"""Local RAG pipeline: ingest → chunk → embed → store → search.

A *knowledge base* (KB) is a named, self-contained directory under
``~/.aero/knowledge/<name>/``:

    sources/        copies of the ingested files (the source of truth)
    chunks.lance/   the LanceDB vector table (derived; safe to rebuild)
    kb.json         manifest: embedder + dim, chunk params, and per-file records
                    (sha/mtime/chunk-count) used for incremental re-ingest

Embeddings come from the engine's co-resident embedder slot (Phase g1), so a KB is
**bound to the embedder it was built with** — search must embed the query with the
same model (the dim must match), and changing the embedder means re-ingesting.

The store is reached only through ``search()`` / ``ingest()`` here, so a future
lexical/FTS or agentic-search mode can be added behind the same surface without
touching callers.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

from . import engine, store
from .store_ops import sanitize_name

_TABLE = "chunks"
_MANIFEST = "kb.json"
DEFAULT_CHUNK_SIZE = 1200       # characters (~300 tokens; safe for small embedders' context)
DEFAULT_CHUNK_OVERLAP = 150

# Files we read as text directly; ``.pdf`` is special-cased; anything else is skipped.
_TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx",
    ".json", ".yaml", ".yml", ".toml", ".csv", ".html", ".css", ".sh", ".java",
    ".go", ".rs", ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".sql", ".tex", ".log",
}

# Progress callback: (file_index_1based, total_files, source_name, status).
IngestProgress = Callable[[int, int, str, str], None]


# --------------------------------------------------------------------------- #
# Manifest helpers
# --------------------------------------------------------------------------- #


def _manifest_path(home: Path, kb: str) -> Path:
    return store.kb_dir(home, kb) / _MANIFEST


def kb_exists(home: Path, kb: str) -> bool:
    return _manifest_path(home, kb).is_file()


def _load_manifest(home: Path, kb: str) -> dict:
    p = _manifest_path(home, kb)
    if not p.is_file():
        raise ValueError(f"no knowledge base named {kb!r}")
    return json.loads(p.read_text())


def _save_manifest(home: Path, kb: str, data: dict) -> None:
    _manifest_path(home, kb).write_text(json.dumps(data, indent=2))


# --------------------------------------------------------------------------- #
# Parsing + chunking
# --------------------------------------------------------------------------- #


def _supported(p: Path) -> bool:
    s = p.suffix.lower()
    return s == ".pdf" or s in _TEXT_SUFFIXES


def parse(path: Path) -> str:
    """Extract plain text from a file (PDF via pypdf; everything else as UTF-8)."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk(text: str, size: int, overlap: int) -> list[tuple[int, int, str]]:
    """Sliding-window character chunks → ``(start, end, text)`` (offsets for citations)."""
    out: list[tuple[int, int, str]] = []
    n = len(text)
    if n == 0:
        return out
    step = max(1, size - overlap)
    i = 0
    while i < n:
        end = min(n, i + size)
        piece = text[i:end]
        if piece.strip():
            out.append((i, end, piece))
        if end >= n:
            break
        i += step
    return out


# --------------------------------------------------------------------------- #
# Vector store (LanceDB)
# --------------------------------------------------------------------------- #


def _connect(home: Path, kb: str):
    import lancedb

    return lancedb.connect(str(store.kb_dir(home, kb)))

def _open_table(db):
    try:
        return db.open_table(_TABLE)
    except Exception:  # noqa: BLE001 - table doesn't exist yet (empty KB)
        return None


def _sql_quote(s: str) -> str:
    return s.replace("'", "''")


# --------------------------------------------------------------------------- #
# KB lifecycle
# --------------------------------------------------------------------------- #


def create_kb(
    home: Path, name: str, embedder: str, *,
    chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    """Create an empty KB bound to ``embedder``. Validates the embedder by loading it."""
    name = sanitize_name(name)
    if kb_exists(home, name):
        raise ValueError(f"knowledge base {name!r} already exists")
    dim = engine.embedder_dim(embedder)  # loads/validates the embedder; also fixes the dim
    if not dim:
        raise ValueError(f"could not determine embedding dimension for {embedder!r}")
    (store.kb_dir(home, name) / "sources").mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": name, "embedder": embedder, "dim": dim,
        "chunk_size": chunk_size, "overlap": overlap,
        "created": time.time(), "files": [],
    }
    _save_manifest(home, name, manifest)
    return manifest


def _expand(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files += [f for f in sorted(p.rglob("*")) if f.is_file() and _supported(f)]
        elif p.is_file() and _supported(p):
            files.append(p)
    return files


def ingest(home: Path, kb: str, paths: list[Path], *, progress_cb: Optional[IngestProgress] = None) -> dict:
    """Ingest files/dirs into a KB: parse → chunk → embed → store, incrementally.

    Files are copied into ``sources/`` (keyed by basename) and skipped on re-ingest if
    unchanged (sha match). Returns counts of files/chunks added and files skipped.
    """
    manifest = _load_manifest(home, kb)
    embedder = manifest["embedder"]
    by_source = {f["source"]: f for f in manifest["files"]}

    files = _expand(paths)
    db = _connect(home, kb)
    tbl = _open_table(db)
    sources_dir = store.kb_dir(home, kb) / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    added_files = added_chunks = skipped = 0
    total = len(files)
    for idx, fpath in enumerate(files, start=1):
        source = fpath.name
        sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
        if source in by_source and by_source[source]["sha"] == sha:
            skipped += 1
            if progress_cb:
                progress_cb(idx, total, source, "skipped")
            continue

        chunks = chunk(parse(fpath), manifest["chunk_size"], manifest["overlap"])
        if not chunks:
            skipped += 1
            if progress_cb:
                progress_cb(idx, total, source, "empty")
            continue

        vectors = engine.embed(embedder, [c[2] for c in chunks])
        shutil.copyfile(fpath, sources_dir / source)
        rows = [
            {"id": f"{source}:{s}", "vector": v, "text": t, "source": source, "start": s, "end": e}
            for (s, e, t), v in zip(chunks, vectors)
        ]
        if tbl is None:
            tbl = db.create_table(_TABLE, data=rows)
        else:
            if source in by_source:  # re-ingest: drop this file's old rows first
                tbl.delete(f"source = '{_sql_quote(source)}'")
            tbl.add(rows)

        by_source[source] = {"source": source, "sha": sha,
                             "mtime": fpath.stat().st_mtime, "chunks": len(chunks)}
        added_files += 1
        added_chunks += len(chunks)
        if progress_cb:
            progress_cb(idx, total, source, "ingested")

    manifest["files"] = list(by_source.values())
    _save_manifest(home, kb, manifest)
    return {"files_ingested": added_files, "chunks_added": added_chunks,
            "skipped": skipped, "total_files": len(manifest["files"])}


def search(home: Path, kb: str, query: str, k: int = 4) -> list[dict]:
    """Retrieve the top-``k`` chunks for ``query`` (embedded with the KB's embedder)."""
    manifest = _load_manifest(home, kb)
    tbl = _open_table(_connect(home, kb))
    if tbl is None:
        return []
    qv = engine.embed(manifest["embedder"], [query])[0]
    rows = tbl.search(qv).metric("cosine").limit(k).to_list()
    return [
        {"text": r["text"], "source": r["source"], "start": r["start"], "end": r["end"],
         "score": round(1.0 - float(r.get("_distance", 0.0)), 4)}
        for r in rows
    ]


def get_kb(home: Path, kb: str) -> Optional[dict]:
    """Full manifest (incl. per-file records) for ``kb``, or None if it doesn't exist."""
    if not kb_exists(home, kb):
        return None
    m = _load_manifest(home, kb)
    m["chunks"] = sum(f["chunks"] for f in m["files"])
    return m


def list_kbs(home: Path) -> list[dict]:
    """Summary of every KB (name, embedder, dim, #files, #chunks)."""
    d = store.knowledge_dir(home)
    out: list[dict] = []
    if d.is_dir():
        for sub in sorted(d.iterdir()):
            if (sub / _MANIFEST).is_file():
                m = json.loads((sub / _MANIFEST).read_text())
                out.append({
                    "name": m["name"], "embedder": m["embedder"], "dim": m["dim"],
                    "files": len(m["files"]), "chunks": sum(f["chunks"] for f in m["files"]),
                })
    return out


def delete_kb(home: Path, kb: str) -> None:
    if not kb_exists(home, kb):
        raise ValueError(f"no knowledge base named {kb!r}")
    shutil.rmtree(store.kb_dir(home, kb))


def remove_file(home: Path, kb: str, source: str) -> None:
    """Drop one ingested file from a KB (its rows, its source copy, its manifest entry)."""
    manifest = _load_manifest(home, kb)
    if source not in {f["source"] for f in manifest["files"]}:
        raise ValueError(f"{source!r} is not in {kb!r}")
    tbl = _open_table(_connect(home, kb))
    if tbl is not None:
        tbl.delete(f"source = '{_sql_quote(source)}'")
    (store.kb_dir(home, kb) / "sources" / source).unlink(missing_ok=True)
    manifest["files"] = [f for f in manifest["files"] if f["source"] != source]
    _save_manifest(home, kb, manifest)
