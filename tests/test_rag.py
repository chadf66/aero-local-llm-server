"""RAG pipeline tests (Phase g2) — stub embedder + a real LanceDB store, temp home.

The stub embedder is deterministic (identical text → identical vector), so searching
with the exact text of an ingested chunk puts that chunk's source at distance 0 — a
reliable way to assert retrieval round-trips without a real embedding model.
"""

from __future__ import annotations

import pytest

from aero import engine, rag, store


@pytest.fixture
def home(tmp_path):
    engine.configure({}, backend="stub", idle_timeout=0, home=tmp_path)
    return tmp_path


def _write(home, rel, text):
    p = home / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


# --------------------------------------------------------------------------- #
# Chunking / parsing
# --------------------------------------------------------------------------- #


def test_chunk_offsets_and_overlap():
    text = "abcdefghij"  # 10 chars
    chunks = rag.chunk(text, size=4, overlap=1)
    assert chunks[0] == (0, 4, "abcd")
    assert chunks[1][0] == 3  # step = size - overlap = 3
    # reconstructable, covers the whole string
    assert chunks[0][2] == text[chunks[0][0]:chunks[0][1]]
    assert chunks[-1][1] == len(text)


def test_chunk_empty_text():
    assert rag.chunk("", 100, 10) == []
    assert rag.chunk("   \n  ", 100, 10) == []


def test_parse_reads_text(home):
    p = _write(home, "docs/a.md", "# Title\nbody text")
    assert "body text" in rag.parse(p)


# --------------------------------------------------------------------------- #
# KB lifecycle + retrieval
# --------------------------------------------------------------------------- #


def test_create_kb_records_embedder_and_dim(home):
    m = rag.create_kb(home, "docs", "bge-small")
    assert m["embedder"] == "bge-small" and m["dim"] == engine._STUB_EMBED_DIM
    assert (store.kb_dir(home, "docs") / "sources").is_dir()
    with pytest.raises(ValueError):
        rag.create_kb(home, "docs", "bge-small")  # duplicate


def test_ingest_and_search_round_trip(home):
    rag.create_kb(home, "docs", "bge-small")
    a = _write(home, "src/alpha.txt", "alpha alpha the quick brown fox jumps")
    b = _write(home, "src/beta.txt", "beta beta lorem ipsum dolor sit amet")
    result = rag.ingest(home, "docs", [a, b])
    assert result["files_ingested"] == 2 and result["chunks_added"] == 2

    # Querying with alpha's exact text -> alpha is the top hit (distance 0).
    hits = rag.search(home, "docs", "alpha alpha the quick brown fox jumps", k=2)
    assert hits[0]["source"] == "alpha.txt"
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-3)
    assert {h["source"] for h in hits} == {"alpha.txt", "beta.txt"}
    # sources copied in
    assert (store.kb_dir(home, "docs") / "sources" / "alpha.txt").is_file()


def test_ingest_is_incremental(home):
    rag.create_kb(home, "docs", "bge-small")
    f = _write(home, "src/a.txt", "hello world")
    assert rag.ingest(home, "docs", [f])["files_ingested"] == 1
    # unchanged -> skipped
    r2 = rag.ingest(home, "docs", [f])
    assert r2["files_ingested"] == 0 and r2["skipped"] == 1
    # changed -> re-ingested, no duplicate file entry
    f.write_text("hello world, again, with more text")
    r3 = rag.ingest(home, "docs", [f])
    assert r3["files_ingested"] == 1
    assert rag.get_kb(home, "docs")["files"][0]["chunks"] >= 1
    assert len(rag.get_kb(home, "docs")["files"]) == 1


def test_ingest_directory_recurses(home):
    rag.create_kb(home, "docs", "bge-small")
    _write(home, "tree/a.md", "first doc")
    _write(home, "tree/sub/b.txt", "second doc")
    _write(home, "tree/skip.bin", "binary-ish, unsupported suffix")
    result = rag.ingest(home, "docs", [home / "tree"])
    assert result["files_ingested"] == 2  # .bin skipped (unsupported)


def test_list_and_get_kb(home):
    rag.create_kb(home, "docs", "bge-small")
    rag.ingest(home, "docs", [_write(home, "src/a.txt", "content here")])
    listed = rag.list_kbs(home)
    assert listed[0]["name"] == "docs" and listed[0]["files"] == 1 and listed[0]["chunks"] >= 1
    assert rag.get_kb(home, "missing") is None


def test_remove_file(home):
    rag.create_kb(home, "docs", "bge-small")
    a = _write(home, "src/a.txt", "keep me")
    b = _write(home, "src/b.txt", "delete me later")
    rag.ingest(home, "docs", [a, b])
    rag.remove_file(home, "docs", "b.txt")
    assert {f["source"] for f in rag.get_kb(home, "docs")["files"]} == {"a.txt"}
    assert not (store.kb_dir(home, "docs") / "sources" / "b.txt").exists()
    assert all(h["source"] != "b.txt" for h in rag.search(home, "docs", "delete me later", k=5))


def test_delete_kb(home):
    rag.create_kb(home, "docs", "bge-small")
    rag.delete_kb(home, "docs")
    assert not rag.kb_exists(home, "docs")
    with pytest.raises(ValueError):
        rag.delete_kb(home, "docs")


def test_search_missing_kb_raises(home):
    with pytest.raises(ValueError):
        rag.search(home, "nope", "q")
