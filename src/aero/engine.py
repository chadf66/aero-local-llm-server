"""The inference engine: a set of known models, at most one resident at a time.

This is the memory-first heart of the project, and the explicit inversion of the
fleet worker's LRU cache. On a single memory-constrained box we would rather pay a
cold reload than risk two models resident at once, so:

  * Load-on-demand   -- a model is loaded the first time a request names it.
  * Evict-before-load -- switching models frees the old one *before* loading the
                         new (never the two-resident peak a fleet would tolerate).
  * Idle-unload      -- a background timer frees the resident model after a stretch
                         of inactivity, handing unified memory back to the system.

At most one model is ever in memory. KV-cache quantization (q8_0/q4_0) trades a
little quality for a smaller KV cache, the knob that dominates memory as context
grows: KV ~= 2 x layers x kv_dim x n_ctx x 2 bytes (f16).

Two backends:
  * "llama" -- llama-cpp-python running a local GGUF, all layers on the Metal GPU.
  * "stub"  -- echoes the prompt back. No dependencies, used by the tests.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Iterator, Optional

from .config import KV_CACHE_TYPES, ModelConfig
from .schemas import ChatCompletionRequest, Usage

logger = logging.getLogger("aero.engine")

# ggml tensor types for the KV cache. These enum values are stable in ggml.
# f16 is the default (we don't override type_k/type_v); the quantized types need
# flash attention, which we enable alongside them. Keys mirror config.KV_CACHE_TYPES.
_GGML_TYPE = {"f16": 1, "q8_0": 8, "q4_0": 2}

# --------------------------------------------------------------------------- #
# Configuration + single-slot model state, set by configure() at startup.
# --------------------------------------------------------------------------- #

_models: dict[str, ModelConfig] = {}   # model name -> its per-model config
_backend = "llama"
_idle_timeout = 300.0                  # seconds; 0 disables idle-unload

# The one resident model, guarded by _lock. _handle is a llama_cpp.Llama (or the
# name string for the stub backend). All loads, unloads, inference, and the idle
# sweep serialize on _lock -- correct and simple for a single-user box.
_handle: Any = None
_loaded_name: Optional[str] = None
_loaded_key: Optional[tuple] = None    # load_key() of the resident model (weights/ctx/kv)
_last_used = 0.0
_load_calls = 0                        # how many times _load actually ran (test seam)
_lock = threading.RLock()

_idle_thread: Optional[threading.Thread] = None
_idle_stop = threading.Event()


def configure(
    models: dict[str, ModelConfig],
    *,
    backend: str = "llama",
    idle_timeout: float = 300.0,
) -> None:
    """Install the per-model configs and load policy. Nothing is loaded yet."""
    global _models, _backend, _idle_timeout
    with _lock:
        _unload()  # drop any model loaded under the previous config
        _models = dict(models)
        _backend = backend
        _idle_timeout = float(idle_timeout)
    _start_idle_thread()


def available_models() -> list[str]:
    """Every model this server can serve (loaded on demand), sorted by name."""
    return sorted(_models)


def loaded_model() -> Optional[str]:
    """The model currently resident in memory, or None."""
    return _loaded_name


# --------------------------------------------------------------------------- #
# Load / unload (all callers hold _lock).
# --------------------------------------------------------------------------- #


def _load(name: str) -> None:
    global _handle, _loaded_name, _loaded_key, _load_calls
    cfg = _models[name]

    if _backend == "stub":
        _handle = name
    elif _backend == "llama":
        from llama_cpp import Llama

        # n_gpu_layers=-1 puts every layer on the Metal GPU -- the whole point on
        # Apple Silicon. Quantized KV cache needs flash attention enabled.
        kwargs: dict[str, Any] = dict(model_path=cfg.path, n_ctx=cfg.n_ctx, n_gpu_layers=-1, verbose=False)
        if cfg.kv_cache_type != "f16":
            ggml = _GGML_TYPE[cfg.kv_cache_type]
            kwargs.update(type_k=ggml, type_v=ggml, flash_attn=True)
        if cfg.chat_format:
            kwargs["chat_format"] = cfg.chat_format
        _handle = Llama(**kwargs)
    else:
        raise ValueError(f"unknown backend {_backend!r}")

    _loaded_name = name
    _loaded_key = cfg.load_key()
    _load_calls += 1
    logger.info("loaded %s (n_ctx=%d, kv=%s)", name, cfg.n_ctx, cfg.kv_cache_type)


def _unload() -> None:
    global _handle, _loaded_name, _loaded_key
    if _loaded_name is None:
        return
    name = _loaded_name
    # Free the Metal context deterministically rather than waiting for GC.
    if _backend == "llama" and hasattr(_handle, "close"):
        try:
            _handle.close()
        except Exception:  # noqa: BLE001 - a failed free shouldn't wedge the server
            logger.warning("error freeing %s", name, exc_info=True)
    _handle = None
    _loaded_name = None
    _loaded_key = None
    logger.info("unloaded %s", name)


def _acquire_handle(name: str) -> Any:
    """Return a resident handle for ``name``, loading (and evicting) as needed.

    The caller must hold _lock for the duration of the work that uses the handle,
    so the idle sweep can't unload mid-request.
    """
    global _last_used, _loaded_name
    if name not in _models:
        raise KeyError(name)
    if _handle is not None and _models[name].load_key() == _loaded_key:
        # Same weights/context already resident -- a different model that only
        # differs in system prompt / sampling. Switch persona, no reload.
        _loaded_name = name
    elif _loaded_name != name:
        _unload()          # EVICT BEFORE LOAD: free the old model first.
        _load(name)
    _last_used = time.monotonic()
    return _handle


# --------------------------------------------------------------------------- #
# Idle-unload sweep.
# --------------------------------------------------------------------------- #


def _unload_if_idle() -> bool:
    """Free the resident model if it has been idle past the timeout. Testable seam."""
    with _lock:
        if _loaded_name is None or _idle_timeout <= 0:
            return False
        if time.monotonic() - _last_used > _idle_timeout:
            logger.info("idle-unloading %s after %.0fs", _loaded_name, _idle_timeout)
            _unload()
            return True
        return False


def _idle_loop() -> None:
    # Wake often enough to be responsive without busy-looping; the actual unload
    # decision is made in _unload_if_idle against the real timeout.
    interval = max(1.0, min(_idle_timeout, 30.0))
    while not _idle_stop.wait(interval):
        try:
            _unload_if_idle()
        except Exception:  # noqa: BLE001 - keep the sweeper alive
            logger.warning("idle sweep error", exc_info=True)


def _start_idle_thread() -> None:
    global _idle_thread
    if _idle_timeout <= 0 or (_idle_thread is not None and _idle_thread.is_alive()):
        return
    _idle_stop.clear()
    _idle_thread = threading.Thread(target=_idle_loop, name="aero-idle-unload", daemon=True)
    _idle_thread.start()


# --------------------------------------------------------------------------- #
# Inference (adapted from the fleet worker; usage is computed under the lock).
# --------------------------------------------------------------------------- #


def _effective_kwargs(request: ChatCompletionRequest, cfg: ModelConfig) -> dict:
    """Build create_chat_completion args, layering per-model config under the request.

    Two kinds of merging:
      * Default system prompt -- injected only if the request has no system message.
      * Sampling / max_tokens  -- the request wins for any field it set explicitly
        (tracked via Pydantic's model_fields_set); otherwise the model's config
        default applies; otherwise the schema's built-in default.
    """
    messages = [m.model_dump() for m in request.messages]
    if cfg.system and not any(m["role"] == "system" for m in messages):
        messages = [{"role": "system", "content": cfg.system}] + messages

    def pick(field: str, cfg_val: Any) -> Any:
        if field in request.model_fields_set:
            return getattr(request, field)
        return cfg_val if cfg_val is not None else getattr(request, field)

    return {
        "messages": messages,
        "temperature": pick("temperature", cfg.sampling.temperature),
        "top_p": pick("top_p", cfg.sampling.top_p),
        "top_k": pick("top_k", cfg.sampling.top_k),
        "max_tokens": pick("max_tokens", cfg.max_tokens),
        "stop": pick("stop", cfg.sampling.stop),
        "seed": request.seed,
    }


def _usage(prompt_tokens: int, completion_tokens: int) -> Usage:
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _stub_prompt_tokens(request: ChatCompletionRequest) -> int:
    """A whitespace-token stand-in for the stub backend (no real tokenizer)."""
    return sum(len(m.content.split()) for m in request.messages)


def run_inference(request: ChatCompletionRequest) -> tuple[str, str, Usage]:
    """Run a (non-streaming) chat completion, loading the model if needed.

    Returns ``(text, finish_reason, usage)``. finish_reason is "stop" for a
    natural end or "length" when the output hit ``max_tokens``; usage carries the
    prompt/completion token counts (exact, straight from llama.cpp's result).
    """
    with _lock:
        handle = _acquire_handle(request.model)
        cfg = _models[request.model]

        if _backend == "stub":
            last = request.messages[-1].content if request.messages else ""
            text = f"[stub:{request.model}] echo: {last}"
            return text, "stop", _usage(_stub_prompt_tokens(request), len(text.split()))

        result = handle.create_chat_completion(**_effective_kwargs(request, cfg))
        choice, u = result["choices"][0], result["usage"]
        return (
            choice["message"]["content"],
            choice.get("finish_reason") or "stop",
            _usage(u["prompt_tokens"], u["completion_tokens"]),
        )


def stream_inference(
    request: ChatCompletionRequest,
) -> Iterator[tuple[str, Any]]:
    """Yield streaming events: ``("content", piece)`` chunks, then a single
    ``("end", (finish_reason, usage))``.

    The lock is held for the whole generation -- through the final usage tally --
    so the idle sweep can't unload the model out from under a stream. (Single-user
    box: serializing requests is fine.)
    """
    with _lock:
        handle = _acquire_handle(request.model)
        cfg = _models[request.model]
        finish_reason = "stop"
        pieces: list[str] = []

        if _backend == "stub":
            last = request.messages[-1].content if request.messages else ""
            for word in f"[stub:{request.model}] echo: {last}".split(" "):
                pieces.append(word + " ")
                yield "content", word + " "
        else:
            for chunk in handle.create_chat_completion(**_effective_kwargs(request, cfg), stream=True):
                choice = chunk["choices"][0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                piece = choice["delta"].get("content") or ""
                if piece:
                    pieces.append(piece)
                    yield "content", piece

        global _last_used
        _last_used = time.monotonic()
        yield "end", (finish_reason, _stream_usage(handle, request, "".join(pieces)))


def _stream_usage(handle: Any, request: ChatCompletionRequest, completion_text: str) -> Usage:
    """Approximate token usage for a streamed response.

    llama.cpp doesn't surface usage in stream chunks (this version has no
    stream_options), so we estimate: completion by re-tokenizing the output, and
    the total from the model's context state after generation (handle.n_tokens),
    with prompt as the difference. This stays within ~1 token of the non-streaming
    counts, and unlike tokenizing the raw messages it includes the chat-template
    tokens that dominate short prompts.
    """
    if _backend == "stub":
        return _usage(_stub_prompt_tokens(request), len(completion_text.split()))

    completion_tokens = len(handle.tokenize(completion_text.encode("utf-8"), add_bos=False))
    total_tokens = getattr(handle, "n_tokens", 0) or completion_tokens
    return _usage(max(0, total_tokens - completion_tokens), completion_tokens)
