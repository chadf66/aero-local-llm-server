"""The on-disk layout of `~/.aero` and small filesystem helpers.

The store splits weights from definitions:

    ~/.aero/
      gguf/      raw weights (what `aero pull` downloads)
      models/    model definitions (*.toml) -- what you actually run

A *model* is a definition in `models/`, which references weights in `gguf/` (by
name, or an explicit path) or, for the zero-config case, the same-named GGUF. A
bare GGUF with no definition still auto-registers as a model (see
`config.build_registry`). Decoupling the two is what lets one GGUF back many named
models (e.g. different system prompts) without copying weights.
"""

from __future__ import annotations

from pathlib import Path

# The aero home. Everything else lives under here.
DEFAULT_HOME = Path.home() / ".aero"


def gguf_dir(home: Path) -> Path:
    """Directory of raw GGUF weights."""
    return home / "gguf"


def config_dir(home: Path) -> Path:
    """Directory of model definitions (`*.toml`)."""
    return home / "models"


def human_size(num_bytes: int) -> str:
    """Format a byte count as a short human-readable string (e.g. ``2.3 GB``)."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
