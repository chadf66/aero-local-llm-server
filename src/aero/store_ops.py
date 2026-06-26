"""Model-store write operations, shared by the CLI and the web admin API.

The pull/create/delete logic lives here as plain functions (no Typer, no FastAPI)
so both front ends call one implementation and can't drift. The engine's live
``reload_from_disk`` (see engine.py) picks up whatever these write to disk, so a
pull or config edit takes effect without restarting the server.

Hugging Face is used only for *metadata and URLs* (`list_repo_files`,
`get_paths_info`, `hf_hub_url`); the actual GGUF transfer is a plain streamed HTTP
GET so we can report real progress (and so the CLI gets a progress bar too).
"""

from __future__ import annotations

import os
import re
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from . import store
from .config import ModelConfig, SamplingConfig, resolve_weights

# A progress callback receives (bytes_downloaded, total_bytes); total may be 0 if
# the server didn't send Content-Length.
ProgressCb = Callable[[int, int], None]

# Starter `.toml` written next to a freshly pulled GGUF (every field commented out
# with its default). Lives here so both `aero pull` and the web pull use one copy.
STARTER_TEMPLATE = """\
# aero model definition for `{name}`.
# Weights: gguf/{name}.gguf (resolved from this file's name).
# Every field is optional and shown commented out with its default. Uncomment and
# edit what you want; an empty file is a valid config (pure defaults).
# Tip: `aero show {name}` prints the config the server will actually use.

# system = "You are a helpful assistant."   # default system prompt
# n_ctx = 4096                              # context window: an int, or "auto" to size to memory
# kv_cache_type = "f16"                     # KV-cache precision: f16 | q8_0 | q4_0  (q8_0/q4_0 fit more context)
# max_tokens = 2048                         # default completion cap
# tools = true                              # enable tool/function calling (agents)
# chat_format = "chatml"                    # override the GGUF's chat template (rarely needed)

# To make a variant that reuses these weights, create another .toml (any name) with:
#   from = "{name}"

# [sampling]                                # defaults used when the request doesn't set them
# temperature = 0.7
# top_p = 0.95
# top_k = 40
# stop = ["</s>"]
"""


def write_starter_config(home: Path, stem: str) -> Path:
    """Write the commented starter `.toml` for a GGUF, unless one already exists."""
    config_dir = store.config_dir(home)
    config_dir.mkdir(parents=True, exist_ok=True)
    toml_path = config_dir / f"{stem}.toml"
    if not toml_path.exists():
        toml_path.write_text(STARTER_TEMPLATE.format(name=stem))
    return toml_path

# Field order for the generated .toml (only set fields are emitted).
_TOP_FIELDS = ("from", "system", "n_ctx", "kv_cache_type", "max_tokens", "tools",
               "chat_format", "knowledge", "knowledge_top_k")
_SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "stop")


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #


def sanitize_name(name: str) -> str:
    """A model name becomes a ``models/<name>.toml`` filename, so keep it tame."""
    name = (name or "").strip()
    if not name or name.startswith(".") or not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise ValueError(
            f"invalid model name {name!r}: use letters, digits, '.', '_', '-' "
            "(no slashes or spaces)"
        )
    return name


# --------------------------------------------------------------------------- #
# Hugging Face: list + download
# --------------------------------------------------------------------------- #


def list_repo_ggufs(repo: str) -> list[dict]:
    """The GGUF files in a HF repo, each ``{filename, size}`` (size may be None)."""
    from huggingface_hub import HfApi

    api = HfApi()
    files = sorted(f for f in api.list_repo_files(repo) if f.endswith(".gguf"))
    if not files:
        return []
    sizes: dict[str, Optional[int]] = {}
    try:
        for info in api.get_paths_info(repo, files):
            sizes[info.path] = getattr(info, "size", None)
    except Exception:  # noqa: BLE001 - sizes are a nicety; the list still works
        pass
    return [{"filename": f, "size": sizes.get(f)} for f in files]


def download_gguf(repo: str, filename: str, dest_dir: Path, progress_cb: Optional[ProgressCb] = None) -> Path:
    """Stream a GGUF from HF to ``dest_dir``, reporting progress, then return its path.

    Downloads to ``<name>.part`` and renames on success so an interrupted pull never
    looks like a complete model. Honors ``HF_TOKEN`` for gated repos.
    """
    from huggingface_hub import hf_hub_url

    dest_dir.mkdir(parents=True, exist_ok=True)
    local_name = Path(filename).name
    final = dest_dir / local_name
    part = dest_dir / (local_name + ".part")

    req = urllib.request.Request(hf_hub_url(repo_id=repo, filename=filename))
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    with urllib.request.urlopen(req) as resp:  # follows the redirect to the LFS CDN
        total = int(resp.headers.get("Content-Length") or 0)
        downloaded = 0
        with open(part, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                if progress_cb is not None:
                    progress_cb(downloaded, total)
    part.replace(final)
    return final


# --------------------------------------------------------------------------- #
# Config files: write (with validation) + a tiny TOML serializer
# --------------------------------------------------------------------------- #


def _toml_str(s: str) -> str:
    """A TOML basic string: quote and escape so newlines/quotes round-trip."""
    out = (
        s.replace("\\", "\\\\").replace('"', '\\"')
        .replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
    )
    return f'"{out}"'


def _toml_value(v) -> str:
    if isinstance(v, bool):  # bool is a subclass of int — check it first
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_str(str(x)) for x in v) + "]"
    return _toml_str(str(v))  # str, or the literal "auto"


def dump_config_toml(fields: dict) -> str:
    """Serialize a (validated) config dict to TOML. Only set fields are written."""
    lines: list[str] = []
    for key in _TOP_FIELDS:
        if fields.get(key) is not None:
            lines.append(f"{key} = {_toml_value(fields[key])}")
    sampling = fields.get("sampling") or {}
    set_sampling = {k: sampling[k] for k in _SAMPLING_FIELDS if sampling.get(k) is not None}
    if set_sampling:
        lines.append("")
        lines.append("[sampling]")
        for k in _SAMPLING_FIELDS:
            if k in set_sampling:
                lines.append(f"{k} = {_toml_value(set_sampling[k])}")
    return "\n".join(lines) + "\n"


def write_model_config(home: Path, name: str, fields: dict) -> Path:
    """Validate a config (via ModelConfig) and write ``models/<name>.toml``.

    ``fields`` mirrors the TOML: ``from``, ``system``, ``n_ctx`` (int or "auto"),
    ``kv_cache_type``, ``max_tokens``, ``tools``, ``chat_format``, and a ``sampling``
    sub-dict. Raises ValueError on anything invalid (bad name, missing weights, bad
    kv_cache_type, …) so the caller can surface a clear message.
    """
    name = sanitize_name(name)
    base = fields.get("from")
    weights = resolve_weights(base, name, store.gguf_dir(home))
    if not weights.is_file():
        raise ValueError(
            f"weights not found at {weights}" + (f" (from={base!r})" if base else "")
            + " — pull the GGUF first, or set `from` to an installed one."
        )

    # Construct the model to validate every field the same way the server does at load.
    ModelConfig(
        name=name,
        path=str(weights),
        base=base,
        system=fields.get("system"),
        n_ctx=fields.get("n_ctx", 4096),
        kv_cache_type=fields.get("kv_cache_type", "f16"),
        max_tokens=fields.get("max_tokens"),
        chat_format=fields.get("chat_format"),
        tools=bool(fields.get("tools", False)),
        knowledge=fields.get("knowledge"),
        knowledge_top_k=fields.get("knowledge_top_k", 4),
        sampling=SamplingConfig(**(fields.get("sampling") or {})),
    )

    config_dir = store.config_dir(home)
    config_dir.mkdir(parents=True, exist_ok=True)
    toml_path = config_dir / f"{name}.toml"
    toml_path.write_text(dump_config_toml(fields))
    return toml_path


# --------------------------------------------------------------------------- #
# Delete (the orphan-safety logic, mirrored from cli.rm)
# --------------------------------------------------------------------------- #


def delete_model(
    home: Path, name: str, registry: dict[str, ModelConfig], *, weights: bool = False
) -> dict:
    """Delete a model's definition (and, if asked and safe, its weights).

    Mirrors ``aero rm``: a derived model drops only its `.toml`; a base model can also
    drop its GGUF unless another model still references those weights. Returns
    ``{"deleted": [paths], "note": str | None}``.
    """
    cfg = registry.get(name)
    if cfg is None:
        raise ValueError(f"no model named {name!r}")

    toml_path = store.config_dir(home) / f"{name}.toml"
    weights_path = Path(cfg.path)
    referenced_by = [n for n, c in registry.items() if n != name and c.path == cfg.path]

    to_delete: list[Path] = []
    if toml_path.is_file():
        to_delete.append(toml_path)

    note: Optional[str] = None
    drop_weights = weights or (cfg.base is None and not toml_path.is_file())
    if drop_weights:
        if cfg.base is not None:
            note = "Kept weights: they belong to the base model, not this derived one."
        elif referenced_by:
            note = f"Kept weights: still referenced by {', '.join(sorted(referenced_by))}."
        elif weights_path.is_file():
            to_delete.append(weights_path)

    for p in to_delete:
        p.unlink()
    return {"deleted": [str(p) for p in to_delete], "note": note}
