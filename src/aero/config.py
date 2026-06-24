"""Model definitions — a Modelfile, as TOML, decoupled from the weights.

A model is defined by a `models/<name>.toml` file. It may carry a default system
prompt, sampling defaults, context size, KV-cache precision, a `max_tokens`
default, and a `chat_format` override. It points at weights via `from`:

    from = "Qwen2.5-3B-Instruct-Q4_K_M"   # a GGUF name in gguf/
    from = "/abs/path/to/weights.gguf"      # or an explicit path

If `from` is omitted, the model uses the same-named GGUF (`gguf/<name>.gguf`) — the
zero-config base case. Because `from` is just a reference, several definitions can
share one GGUF (e.g. different system prompts) without copying weights, and the
engine can switch between them without reloading (see `engine._acquire_handle`).

A bare GGUF with no definition still registers as a model with defaults
(`build_registry`). Request-time precedence is request field > definition default >
built-in (see `engine._effective_kwargs`).
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

# KV-cache precisions we support (llama.cpp type_k/type_v). Lives here (pure data)
# so both config validation and the engine can share it without a circular import.
KV_CACHE_TYPES = ("f16", "q8_0", "q4_0")


class SamplingConfig(BaseModel):
    """Per-model sampling defaults. `None` means 'no opinion — use the built-in'."""

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop: Optional[list[str]] = None


class ModelConfig(BaseModel):
    """Everything the engine needs to load and serve one model."""

    name: str
    path: str                              # resolved absolute GGUF path (the weights)
    base: Optional[str] = None             # the `from` reference, if any (for display)
    system: Optional[str] = None           # default system prompt (injected if absent)
    n_ctx: Union[int, Literal["auto"]] = 4096   # "auto" = size to memory at load
    kv_cache_type: str = "f16"
    max_tokens: Optional[int] = None        # default completion cap
    chat_format: Optional[str] = None       # llama.cpp chat_format override (e.g. "chatml")
    tools: bool = False                     # enable tool calling (chatml-function-calling)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)

    @field_validator("kv_cache_type")
    @classmethod
    def _check_kv(cls, v: str) -> str:
        if v not in KV_CACHE_TYPES:
            raise ValueError(f"kv_cache_type must be one of {KV_CACHE_TYPES}, got {v!r}")
        return v

    @property
    def effective_chat_format(self) -> Optional[str]:
        """The chat_format used at load (None = the GGUF's own template).

        Tool calling deliberately does NOT force a handler here: the model's native
        template both renders the tools and makes the model emit calls reliably in
        'auto' mode, where the generic function-calling handlers fail. aero parses
        the native tool-call output itself (see engine._parse_tool_calls)."""
        return self.chat_format

    def load_key(self) -> tuple:
        """What actually determines the loaded llama context. Two models with the
        same key share weights and can be swapped without a reload — only the prompt
        and sampling (applied per request) differ."""
        return (self.path, self.n_ctx, self.kv_cache_type, self.effective_chat_format)


def resolve_weights(base: Optional[str], name: str, gguf_dir: Path) -> Path:
    """Resolve a model's GGUF path from its `from` value (or its name)."""
    if base is None:
        return gguf_dir / f"{name}.gguf"
    if base.endswith(".gguf") or "/" in base:
        p = Path(base).expanduser()
        return p if p.is_absolute() else gguf_dir / p
    return gguf_dir / f"{base}.gguf"


def load_config_file(
    toml_path: Path,
    gguf_dir: Path,
    *,
    default_n_ctx: Union[int, str] = 4096,
    default_kv_cache_type: str = "f16",
) -> ModelConfig:
    """Load one `models/<name>.toml` definition, resolving and checking its weights."""
    with open(toml_path, "rb") as fh:
        data = tomllib.load(fh)

    name = toml_path.stem
    base = data.get("from")
    weights = resolve_weights(base, name, gguf_dir)
    if not weights.is_file():
        raise FileNotFoundError(
            f"model {name!r}: weights not found at {weights}"
            + (f" (from={base!r})" if base else "")
        )

    return ModelConfig(
        name=name,
        path=str(weights),
        base=base,
        system=data.get("system"),
        n_ctx=data.get("n_ctx", default_n_ctx),
        kv_cache_type=data.get("kv_cache_type", default_kv_cache_type),
        max_tokens=data.get("max_tokens"),
        chat_format=data.get("chat_format"),
        tools=data.get("tools", False),
        sampling=SamplingConfig(**data.get("sampling", {})),
    )


def build_registry(
    gguf_dir: Path,
    config_dir: Path,
    *,
    default_n_ctx: Union[int, str] = 4096,
    default_kv_cache_type: str = "f16",
) -> dict[str, ModelConfig]:
    """The full model list: every definition in `config_dir`, plus any GGUF in
    `gguf_dir` that has no definition (auto-registered with defaults)."""
    models: dict[str, ModelConfig] = {}

    if config_dir.is_dir():
        for toml_path in sorted(config_dir.glob("*.toml")):
            cfg = load_config_file(
                toml_path, gguf_dir,
                default_n_ctx=default_n_ctx, default_kv_cache_type=default_kv_cache_type,
            )
            models[cfg.name] = cfg

    if gguf_dir.is_dir():
        for gguf in sorted(gguf_dir.glob("*.gguf")):
            if gguf.stem not in models:  # orphan GGUF -> default model
                models[gguf.stem] = ModelConfig(
                    name=gguf.stem, path=str(gguf),
                    n_ctx=default_n_ctx, kv_cache_type=default_kv_cache_type,
                )

    return models
