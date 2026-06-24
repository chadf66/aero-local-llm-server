"""The local model store: a directory of GGUF files.

There's no database and no manifest — a model *is* a `.gguf` file in the models
directory, served under its filename stem. `pull` drops files in, `list`/`rm`/
`show` work over them, and `serve` scans the same directory. Keeping the store a
plain folder is the whole point: it's inspectable, and `pull` is just a download.
"""

from __future__ import annotations

from pathlib import Path

# Where models live by default. `aero pull` writes here and `aero serve` scans it.
DEFAULT_MODELS_DIR = Path.home() / ".aero" / "models"


def scan(models_dir: Path) -> dict[str, str]:
    """Map model name -> GGUF path for every ``*.gguf`` in ``models_dir``.

    A model's name is its filename stem, so ``qwen2.5-3b.Q4_K_M.gguf`` is served
    as ``qwen2.5-3b.Q4_K_M``. Returns an empty dict if the directory is absent.
    """
    if not models_dir.is_dir():
        return {}
    return {p.stem: str(p) for p in sorted(models_dir.glob("*.gguf"))}


def find(models_dir: Path, name: str) -> Path | None:
    """Return the GGUF path for ``name`` (its stem), or None if not in the store."""
    path = scan(models_dir).get(name)
    return Path(path) if path else None


def remove(models_dir: Path, name: str) -> Path:
    """Delete the model named ``name`` and return the path removed.

    Raises ``KeyError`` if no such model exists in the store.
    """
    path = find(models_dir, name)
    if path is None:
        raise KeyError(name)
    path.unlink()
    return path


def human_size(num_bytes: int) -> str:
    """Format a byte count as a short human-readable string (e.g. ``2.3 GB``)."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
