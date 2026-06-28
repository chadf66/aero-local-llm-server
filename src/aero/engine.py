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

import json
import logging
import re
import threading
import time
import uuid
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
_mem_fraction = 0.60                   # fraction of total memory for `n_ctx = "auto"`

# Registry-build context, remembered so the engine can rebuild the model set from
# disk after the admin API pulls/edits/deletes a model (reload_from_disk). None when
# serving an ad-hoc set (e.g. tests, or `serve --model <file>` only).
_home = None                           # type: Optional[object]  (pathlib.Path)
_registry_defaults: dict = {}          # {"default_n_ctx", "default_kv_cache_type"}

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

# A SECOND resident slot, for an embedding model (Phase g). Embedders are tiny
# (~90-300 MB) next to chat weights, so we let one stay loaded *alongside* the chat
# model -- a deliberate exception to single-resident, since RAG needs to embed the
# query at the same time the chat model answers. Guarded by the same _lock.
_embedder_handle: Any = None
_embedder_name: Optional[str] = None
_embedder_dim: Optional[int] = None
_embedder_last_used = 0.0
_STUB_EMBED_DIM = 64                    # fixed dim for the stub backend's fake vectors


def configure(
    models: dict[str, ModelConfig],
    *,
    backend: str = "llama",
    idle_timeout: float = 300.0,
    mem_fraction: float = 0.60,
    home=None,
    registry_defaults: Optional[dict] = None,
) -> None:
    """Install the per-model configs and load policy. Nothing is loaded yet.

    ``home`` + ``registry_defaults`` (the ``default_n_ctx`` / ``default_kv_cache_type``
    passed to ``config.build_registry``) are remembered so ``reload_from_disk`` can
    rebuild the same registry after the model store changes on disk.
    """
    global _models, _backend, _idle_timeout, _mem_fraction, _home, _registry_defaults
    with _lock:
        _unload()  # drop any model loaded under the previous config
        _unload_embedder()
        _models = dict(models)
        _backend = backend
        _idle_timeout = float(idle_timeout)
        _mem_fraction = float(mem_fraction)
        _home = home
        _registry_defaults = dict(registry_defaults or {})
    _start_idle_thread()


def reload(models: dict[str, ModelConfig]) -> None:
    """Swap in a new model set, keeping the resident model loaded if still valid.

    Unlike ``configure``, this does *not* unconditionally unload: if the currently
    resident model is still present with the same ``load_key`` (weights/ctx/kv/format),
    its loaded llama context is preserved — so editing an *unrelated* model, or pulling
    a new one, doesn't evict what you're using. Only a resident that vanished or whose
    load-affecting config changed is unloaded (its next request reloads it cold).
    """
    global _models
    with _lock:
        keep = _loaded_name in models and _loaded_key == models[_loaded_name].load_key()
        if _loaded_name is not None and not keep:
            _unload()
        _models = dict(models)


def reload_from_disk() -> list[dict]:
    """Rebuild the registry from ``home`` and reload. Returns the new ``model_info()``.

    Called by the admin API after a pull/create/edit/delete so the served model set
    refreshes with no restart. Raises if the engine wasn't configured with a ``home``.
    """
    if _home is None:
        raise RuntimeError("engine has no home; model management requires `aero serve`")
    from . import config, store

    registry = config.build_registry(
        store.gguf_dir(_home), store.config_dir(_home),
        default_n_ctx=_registry_defaults.get("default_n_ctx", 4096),
        default_kv_cache_type=_registry_defaults.get("default_kv_cache_type", "f16"),
    )
    reload(registry)
    return model_info()


def home():
    """The configured aero home (a pathlib.Path), or None."""
    return _home


def available_models() -> list[str]:
    """Every model this server can serve (loaded on demand), sorted by name."""
    return sorted(_models)


def loaded_model() -> Optional[str]:
    """The model currently resident in memory, or None."""
    return _loaded_name


def loaded_embedder() -> Optional[str]:
    """The embedding model currently resident (second slot), or None."""
    return _embedder_name


def model_info() -> list[dict]:
    """Per-model config detail for the web UI's `/api/state` (one dict per model)."""
    info = []
    for name in sorted(_models):
        cfg = _models[name]
        info.append({
            "name": name,
            "base": cfg.base,
            "n_ctx": cfg.n_ctx,
            "kv_cache_type": cfg.kv_cache_type,
            "max_tokens": cfg.max_tokens,
            "tools": supports_tools(name),
            "system": cfg.system,
            "knowledge": cfg.knowledge,
        })
    return info


def get_config(name: str) -> Optional[ModelConfig]:
    """The full ModelConfig for ``name`` (for the admin config editor), or None."""
    return _models.get(name)


def current_models() -> dict[str, ModelConfig]:
    """A snapshot copy of the registry (name -> ModelConfig)."""
    return dict(_models)


def context_preview(name: str, kv_cache_type: str) -> Optional[int]:
    """Largest context that fits memory for ``name`` at ``kv_cache_type``.

    Powers the UI's live f16→q8_0→q4_0 trade display. Returns None when it can't be
    computed (unknown model, missing weights, or the stub backend has no GGUF to
    read dimensions from) so the endpoint can degrade gracefully.
    """
    cfg = _models.get(name)
    if cfg is None or kv_cache_type not in KV_CACHE_TYPES:
        return None
    try:
        import os

        from . import sizing

        dims = sizing.read_gguf_dims(cfg.path)
        budget = int(sizing.total_memory_bytes() * _mem_fraction)
        n_ctx = sizing.compute_fit(
            dims, os.path.getsize(cfg.path), kv_cache_type, budget,
            reserve_bytes=_embedder_reserve_bytes(cfg),
        )
        return n_ctx or None
    except Exception:  # noqa: BLE001 - preview is best-effort; never fail the request
        logger.debug("context_preview failed for %s", name, exc_info=True)
        return None


# llama.cpp chat handlers that parse tool calls themselves (if one is set explicitly).
_TOOL_CHAT_FORMATS = {"chatml-function-calling", "functionary", "functionary-v1", "functionary-v2"}

# Hermes/Qwen-style tool calls in the native template output: <tool_call>{json}</tool_call>.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def supports_tools(name: str) -> bool:
    """Whether model ``name`` is tool-enabled (the `tools` flag, or an explicit
    function-calling handler)."""
    cfg = _models.get(name)
    if cfg is None:
        return False
    return cfg.tools or cfg.effective_chat_format in _TOOL_CHAT_FORMATS


def _parse_tool_calls(text: Optional[str], tool_names: Optional[set] = None) -> Optional[list[dict]]:
    """Extract OpenAI-shaped tool calls from a model's native tool-call output.

    Handles the common open-model conventions, which differ by model:
      * ``<tool_call>{json}</tool_call>`` blocks  (Qwen / Hermes)
      * a bare JSON object ``{"name":..., "arguments"|"parameters":...}``  (Llama-3.1)
      * a bare JSON array of such objects  (Ministral)

    For the bare-JSON forms (which look like ordinary content), a call is only
    accepted when its ``name`` matches one of the request's ``tool_names`` -- so a
    model that happens to answer with JSON isn't mistaken for a tool call.
    """
    if not text:
        return None

    blocks = _TOOL_CALL_RE.findall(text)
    raw: list = []
    if blocks:
        for b in blocks:
            try:
                raw.append(json.loads(b))
            except json.JSONDecodeError:
                pass
    else:
        try:
            parsed = json.loads(text.strip())
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            raw = [parsed]
        elif isinstance(parsed, list):
            raw = [o for o in parsed if isinstance(o, dict)]

    calls = []
    for obj in raw:
        name = obj.get("name")
        if not name or (tool_names is not None and not blocks and name not in tool_names):
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        if not isinstance(args, str):
            args = json.dumps(args)
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": args},
        })
    return calls or None


# --------------------------------------------------------------------------- #
# Load / unload (all callers hold _lock).
# --------------------------------------------------------------------------- #


def _embedder_reserve_bytes(cfg: ModelConfig) -> int:
    """Memory to hold back in auto-sizing for a co-resident embedder.

    When a model has a knowledge base, its embedder is loaded *alongside* the chat
    model for retrieval (the second resident slot). Auto-sizing must leave room for it,
    or the chat model gets a context it can load but can't decode once the embedder is
    also resident -- which surfaces as ``llama_decode returned -3``. Estimated from the
    embedder's GGUF size plus runtime overhead (MoE experts, KV, scratch)."""
    if not cfg.knowledge or _home is None:
        return 0
    try:
        from . import rag, store

        kb = rag.get_kb(_home, cfg.knowledge)
        if not kb:
            return 0
        path = store.embedders_dir(_home) / f"{kb['embedder']}.gguf"
        if path.is_file():
            return int(path.stat().st_size * 1.4)
    except Exception:  # noqa: BLE001 - best-effort; reserve nothing if we can't size it
        logger.warning("could not size embedder reserve for kb=%s", cfg.knowledge, exc_info=True)
    return 0


def _load(name: str) -> None:
    global _handle, _loaded_name, _loaded_key, _load_calls
    cfg = _models[name]

    n_ctx = cfg.n_ctx
    if _backend == "stub":
        _handle = name
    elif _backend == "llama":
        from llama_cpp import Llama

        if n_ctx == "auto":
            from . import sizing
            n_ctx = sizing.auto_n_ctx(
                cfg.path, cfg.kv_cache_type, _mem_fraction,
                reserve_bytes=_embedder_reserve_bytes(cfg),
            )

        # n_gpu_layers=-1 puts every layer on the Metal GPU -- the whole point on
        # Apple Silicon. Quantized KV cache needs flash attention enabled.
        kwargs: dict[str, Any] = dict(model_path=cfg.path, n_ctx=n_ctx, n_gpu_layers=-1, verbose=False)
        if cfg.kv_cache_type != "f16":
            ggml = _GGML_TYPE[cfg.kv_cache_type]
            kwargs.update(type_k=ggml, type_v=ggml, flash_attn=True)
        if cfg.effective_chat_format:
            kwargs["chat_format"] = cfg.effective_chat_format
        _handle = Llama(**kwargs)
    else:
        raise ValueError(f"unknown backend {_backend!r}")

    _loaded_name = name
    _loaded_key = cfg.load_key()
    _load_calls += 1
    logger.info("loaded %s (n_ctx=%s, kv=%s)", name, n_ctx, cfg.kv_cache_type)


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
# Embeddings (the second resident slot; Phase g). Co-resident with the chat model.
# --------------------------------------------------------------------------- #


def available_embedders() -> list[str]:
    """Embedding models installed under ``embedders/`` (GGUF stems), sorted."""
    if _home is None:
        return []
    from . import store

    d = store.embedders_dir(_home)
    return sorted(p.stem for p in d.glob("*.gguf")) if d.is_dir() else []


def _stub_embed(text: str) -> list[float]:
    """A deterministic, normalized fake embedding for the stub backend (no model).

    Same text -> same vector, different text -> different; good enough to exercise
    ingest/search/round-trip in tests without downloading an embedder."""
    import hashlib
    import math
    import struct

    raw = b""
    i = 0
    while len(raw) < _STUB_EMBED_DIM * 4:
        raw += hashlib.sha256(f"{text}#{i}".encode()).digest()
        i += 1
    vals = [struct.unpack("<I", raw[j:j + 4])[0] / 2**32 - 0.5 for j in range(0, _STUB_EMBED_DIM * 4, 4)]
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]


def _load_embedder(name: str) -> None:
    global _embedder_handle, _embedder_name, _embedder_dim
    if _backend == "stub":
        _embedder_handle = name
        _embedder_dim = _STUB_EMBED_DIM
    elif _backend == "llama":
        from llama_cpp import Llama

        from . import store

        if _home is None:
            raise RuntimeError("no aero home configured; cannot locate embedders")
        path = store.embedders_dir(_home) / f"{name}.gguf"
        if not path.is_file():
            raise FileNotFoundError(f"embedder {name!r} not found at {path}")
        _embedder_handle = Llama(model_path=str(path), embedding=True, n_gpu_layers=-1, verbose=False)
        _embedder_dim = int(_embedder_handle.n_embd())
    else:
        raise ValueError(f"unknown backend {_backend!r}")
    _embedder_name = name
    logger.info("loaded embedder %s (dim=%s)", name, _embedder_dim)


def _unload_embedder() -> None:
    global _embedder_handle, _embedder_name, _embedder_dim
    if _embedder_name is None:
        return
    name = _embedder_name
    if _backend == "llama" and hasattr(_embedder_handle, "close"):
        try:
            _embedder_handle.close()
        except Exception:  # noqa: BLE001
            logger.warning("error freeing embedder %s", name, exc_info=True)
    _embedder_handle = None
    _embedder_name = None
    _embedder_dim = None
    logger.info("unloaded embedder %s", name)


def embedder_dim(name: str) -> Optional[int]:
    """Output dimension of an embedder (loading it if needed)."""
    with _lock:
        if _embedder_name != name:
            _unload_embedder()
            _load_embedder(name)
        return _embedder_dim


def embed(model: str, texts: list[str]) -> list[list[float]]:
    """Embed ``texts`` with embedding model ``model`` (load co-resident as needed).

    Returns one vector per input. Used by ``/v1/embeddings`` and the RAG pipeline.
    The chat model, if any, stays loaded -- this is the second slot.
    """
    global _embedder_last_used
    with _lock:
        if _backend == "stub":
            _embedder_name_set(model)
            _embedder_last_used = time.monotonic()
            return [_stub_embed(t) for t in texts]

        if _embedder_name != model:
            _unload_embedder()
            _load_embedder(model)
        out = _embedder_handle.create_embedding(input=texts)
        _embedder_last_used = time.monotonic()
        return [d["embedding"] for d in out["data"]]


def _embedder_name_set(name: str) -> None:
    """Stub helper: record the embedder name/dim without loading anything real."""
    global _embedder_name, _embedder_dim
    _embedder_name = name
    _embedder_dim = _STUB_EMBED_DIM


# --------------------------------------------------------------------------- #
# Idle-unload sweep.
# --------------------------------------------------------------------------- #


def _unload_if_idle() -> bool:
    """Free the resident chat model and/or embedder if idle past the timeout.

    Testable seam. Returns True if anything was unloaded."""
    with _lock:
        if _idle_timeout <= 0:
            return False
        now = time.monotonic()
        freed = False
        if _loaded_name is not None and now - _last_used > _idle_timeout:
            logger.info("idle-unloading %s after %.0fs", _loaded_name, _idle_timeout)
            _unload()
            freed = True
        if _embedder_name is not None and now - _embedder_last_used > _idle_timeout:
            logger.info("idle-unloading embedder %s after %.0fs", _embedder_name, _idle_timeout)
            _unload_embedder()
            freed = True
        return freed


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


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Render tool-calling messages into plain system/user/assistant text.

    The native GGUF templates only know system/user/assistant with string content,
    so an assistant ``tool_calls`` message (no content) or a ``tool`` result would
    crash them. Since aero does tool calling at the prompt level, we serialize those
    back into the Hermes ``<tool_call>`` / ``<tool_response>`` text the model expects.
    """
    id_to_name = {tc.get("id"): tc.get("function", {}).get("name")
                  for m in messages for tc in (m.get("tool_calls") or [])}
    out: list[dict] = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if role == "assistant" and m.get("tool_calls"):
            blocks = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = fn.get("arguments")
                blocks.append(f'<tool_call>\n{json.dumps({"name": fn.get("name"), "arguments": args})}\n</tool_call>')
            text = (f"{content}\n" if content else "") + "\n".join(blocks)
            out.append({"role": "assistant", "content": text})
        elif role == "tool":
            name = m.get("name") or id_to_name.get(m.get("tool_call_id"))
            payload = json.dumps({"name": name, "content": content})
            out.append({"role": "user", "content": f"<tool_response>\n{payload}\n</tool_response>"})
        else:
            out.append({"role": role, "content": content if content is not None else ""})
    return out


def _tool_system_prompt(request: ChatCompletionRequest) -> str:
    """The function-calling instructions aero injects when a request carries tools.

    We render the tools into a system prompt ourselves (the widely-trained Hermes
    `<tools>`/`<tool_call>` convention) rather than rely on the GGUF's chat template
    -- many templates don't render tools at all, so passing ``tools`` to llama.cpp
    would silently drop them. aero then parses the `<tool_call>` output back into
    OpenAI tool_calls (see _parse_tool_calls)."""
    sigs = "\n".join(json.dumps(t.model_dump(exclude_none=True)) for t in request.tools)
    prompt = (
        "You are a function calling AI model. You are provided with function "
        "signatures within <tools></tools> XML tags. You may call one or more "
        "functions to assist with the user query. Don't make assumptions about what "
        "values to plug into functions.\n"
        f"<tools>\n{sigs}\n</tools>\n\n"
        "To call a function, respond with ONLY a JSON object wrapped in "
        "<tool_call></tool_call> tags, and nothing else:\n"
        '<tool_call>\n{"name": "<function-name>", "arguments": <arguments-object>}\n</tool_call>'
    )
    # A specific function / "required" -> insist on a call; "auto"/None -> model's choice.
    if request.tool_choice not in (None, "auto"):
        prompt += "\nYou MUST call one of the functions; do not answer in prose."
    return prompt


# Safety cap on injected RAG context, so retrieval can't blow the (small) context
# budget regardless of chunk size x top_k.
# Absolute ceiling on injected context, regardless of how big the window is -- we
# never want to dump an unbounded wall of text at the model. The *real* limit is
# computed per-request against the live n_ctx (see _context_token_budget).
_MAX_CONTEXT_CHARS = 12000
# Conservative chars-per-token used to convert a token budget into a char budget.
# Deliberately low (code/markup tokenize to fewer chars/token than prose) so we
# under-fill rather than overflow; the safety margin absorbs the rest.
_CHARS_PER_TOKEN = 3.2
# Tokens held back for the model's answer when the request/config set no max_tokens.
_DEFAULT_COMPLETION_RESERVE = 512
# Slack for chat-template tokens and tokenizer estimation error.
_CONTEXT_SAFETY_MARGIN = 256
# Below this, there isn't enough room for context to be worth injecting -- answer
# ungrounded instead of crowding out the conversation (or overflowing the window).
_MIN_CONTEXT_TOKENS = 64


def _retrieve(request: ChatCompletionRequest, cfg: ModelConfig) -> list[dict]:
    """Retrieve grounding chunks when the model has a knowledge base, else [].

    Queries on the last user message. Never raises: a missing or broken KB degrades
    to an ungrounded answer (logged), so attaching a KB can't take chat down."""
    if not cfg.knowledge:
        return []
    query = next((m.content for m in reversed(request.messages)
                  if m.role == "user" and m.content), None)
    if not query:
        return []
    try:
        from . import rag

        return rag.search(_home, cfg.knowledge, query, k=cfg.knowledge_top_k)
    except Exception:  # noqa: BLE001 - retrieval is best-effort
        logger.warning("knowledge retrieval failed (kb=%s)", cfg.knowledge, exc_info=True)
        return []


def _format_context(sources: list[dict], max_chars: int) -> str:
    """Render retrieved chunks into a system-prompt context block with [n] tags.

    Caps total chunk text at ``max_chars``, keeping highest-ranked chunks first and
    truncating/dropping the rest -- so the block always fits the caller's budget."""
    blocks, used = [], 0
    for i, s in enumerate(sources, start=1):
        if used >= max_chars:
            break
        text = s["text"]
        if used + len(text) > max_chars:
            text = text[: max(0, max_chars - used)]
        if not text:
            break
        blocks.append(f"[{i}] (source: {s['source']})\n{text}")
        used += len(text)
    body = "\n\n".join(blocks)
    return (
        "You have access to the following context from a knowledge base. Use it to "
        "answer the user's question. If the answer is not in the context, say you don't "
        "know rather than guessing. Cite sources inline using their [n] tags.\n\n"
        f"<context>\n{body}\n</context>"
    )


def _estimate_prompt_tokens(handle: Any, request: ChatCompletionRequest, cfg: ModelConfig) -> int:
    """Rough token count for the conversation (system + messages), *excluding* RAG
    context. Uses the model's own tokenizer when available, plus a small per-message
    overhead for chat-template/role markers."""
    texts: list[str] = []
    if cfg.system and not any(m.role == "system" for m in request.messages):
        texts.append(cfg.system)
    texts.extend(m.content for m in request.messages if m.content)
    blob = "\n".join(texts)
    try:
        n = len(handle.tokenize(blob.encode("utf-8"), add_bos=True, special=True))
    except Exception:  # noqa: BLE001 - fall back to a char heuristic
        n = int(len(blob) / _CHARS_PER_TOKEN)
    overhead = 4 * (len(request.messages) + (1 if cfg.system else 0))
    return n + overhead


def _build_context(
    handle: Any, request: ChatCompletionRequest, cfg: ModelConfig, sources: list[dict]
) -> Optional[str]:
    """Format retrieved chunks into a context block sized to fit the live window.

    Budgets against ``handle.n_ctx()``, reserving room for the conversation and the
    answer, so the injected context can't push the prompt past the model's context
    window (which surfaces as ``llama_decode returned -3``). Returns None when there's
    too little room -- the model then answers ungrounded rather than crashing."""
    if _backend != "llama":  # stub: no real window/tokenizer, keep the old behavior
        return _format_context(sources, _MAX_CONTEXT_CHARS)

    n_ctx = handle.n_ctx()
    eff_max = request.max_tokens if "max_tokens" in request.model_fields_set else cfg.max_tokens
    completion_reserve = min(eff_max or _DEFAULT_COMPLETION_RESERVE, n_ctx // 2)
    convo_tokens = _estimate_prompt_tokens(handle, request, cfg)
    budget_tokens = n_ctx - completion_reserve - convo_tokens - _CONTEXT_SAFETY_MARGIN
    if budget_tokens < _MIN_CONTEXT_TOKENS:
        logger.warning(
            "no room for RAG context (n_ctx=%s, convo≈%s tokens); answering ungrounded",
            n_ctx, convo_tokens,
        )
        return None
    max_chars = min(_MAX_CONTEXT_CHARS, int(budget_tokens * _CHARS_PER_TOKEN))
    return _format_context(sources, max_chars)


def _effective_kwargs(
    request: ChatCompletionRequest, cfg: ModelConfig, context: Optional[str] = None
) -> dict:
    """Build create_chat_completion args, layering per-model config under the request.

    Merging:
      * Default system prompt -- injected only if the request has no system message.
      * Tool instructions      -- when the request carries tools, appended to the
        system message (model-agnostic; see _tool_system_prompt).
      * RAG context            -- when ``context`` is given (the model has a knowledge
        base), appended to the system message too (see _format_context / _retrieve).
      * Sampling / max_tokens   -- the request wins for any field it set explicitly
        (tracked via Pydantic's model_fields_set); otherwise the model's config
        default applies; otherwise the schema's built-in default.
    """
    # exclude_none so optional tool fields only appear when set, then render any
    # tool-calling messages into plain text the native template can handle.
    messages = _normalize_messages([m.model_dump(exclude_none=True) for m in request.messages])
    if cfg.system and not any(m["role"] == "system" for m in messages):
        messages = [{"role": "system", "content": cfg.system}] + messages

    # Tool instructions and RAG context both augment the system message.
    additions = []
    if request.tools:
        additions.append(_tool_system_prompt(request))
    if context:
        additions.append(context)
    if additions:
        extra = "\n\n".join(additions)
        sys_idx = next((i for i, m in enumerate(messages) if m["role"] == "system"), None)
        if sys_idx is not None:
            base = messages[sys_idx].get("content") or ""
            messages[sys_idx] = {**messages[sys_idx], "content": f"{base}\n\n{extra}".strip()}
        else:
            messages = [{"role": "system", "content": extra}] + messages

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
    return sum(len((m.content or "").split()) for m in request.messages)


def _stub_tool_call(request: ChatCompletionRequest) -> dict:
    """A canned tool call (first tool, empty args) so tests exercise the tool path
    without a real model."""
    fn = request.tools[0].function.name
    return {"id": "call_stub0", "type": "function", "function": {"name": fn, "arguments": "{}"}}


def run_inference(request: ChatCompletionRequest) -> tuple[dict, str, Usage, list[dict]]:
    """Run a (non-streaming) chat completion, loading the model if needed.

    Returns ``(message, finish_reason, usage, sources)`` where ``message`` is the
    assistant message dict (``content`` and/or ``tool_calls``) and ``sources`` is the
    list of retrieved RAG chunks (empty unless the model has a knowledge base).
    finish_reason is "stop", "length" (hit max_tokens), or "tool_calls".
    """
    with _lock:
        handle = _acquire_handle(request.model)
        cfg = _models[request.model]
        sources = _retrieve(request, cfg)

        if _backend == "stub":
            if request.tools:
                msg = {"role": "assistant", "content": None, "tool_calls": [_stub_tool_call(request)]}
                return msg, "tool_calls", _usage(_stub_prompt_tokens(request), 1), sources
            last = request.messages[-1].content if request.messages else ""
            text = f"[stub:{request.model}] echo: {last}"
            return ({"role": "assistant", "content": text}, "stop",
                    _usage(_stub_prompt_tokens(request), len(text.split())), sources)

        context = _build_context(handle, request, cfg, sources) if sources else None
        result = handle.create_chat_completion(**_effective_kwargs(request, cfg, context))
        choice, u = result["choices"][0], result["usage"]
        message = choice["message"]
        finish_reason = choice.get("finish_reason") or "stop"
        # Native templates emit tool calls as <tool_call> text; parse them ourselves
        # unless an explicit function-calling handler already produced tool_calls.
        if request.tools and not message.get("tool_calls"):
            parsed = _parse_tool_calls(message.get("content"), {t.function.name for t in request.tools})
            if parsed:
                message = {"role": "assistant", "content": None, "tool_calls": parsed}
                finish_reason = "tool_calls"
        return message, finish_reason, _usage(u["prompt_tokens"], u["completion_tokens"]), sources


def stream_inference(
    request: ChatCompletionRequest,
) -> Iterator[tuple[str, Any]]:
    """Yield streaming events: ``("delta", delta_dict)`` chunks (each an OpenAI
    delta with ``content`` and/or ``tool_calls``), then ``("end", (finish_reason,
    usage))``.

    The lock is held for the whole generation -- through the final usage tally --
    so the idle sweep can't unload the model out from under a stream. (Single-user
    box: serializing requests is fine.)
    """
    with _lock:
        handle = _acquire_handle(request.model)
        cfg = _models[request.model]
        finish_reason = "stop"
        pieces: list[str] = []

        # Retrieval happens before generation, so citations are known up front.
        sources = _retrieve(request, cfg)
        if sources:
            yield "sources", sources
        context = _build_context(handle, request, cfg, sources) if sources else None

        if _backend == "stub":
            if request.tools:
                yield "delta", {"role": "assistant", "tool_calls": [{"index": 0, **_stub_tool_call(request)}]}
                finish_reason = "tool_calls"
            else:
                last = request.messages[-1].content if request.messages else ""
                for word in f"[stub:{request.model}] echo: {last}".split(" "):
                    pieces.append(word + " ")
                    yield "delta", {"content": word + " "}
        elif request.tools:
            # Tool calls arrive as <tool_call> text, not structured deltas, so we
            # buffer the stream and parse at the end, emitting one tool_calls delta
            # (or the content if it wasn't a tool call). Plain text isn't token-
            # streamed here, but a tool-using turn isn't useful half-parsed anyway.
            kwargs = _effective_kwargs(request, cfg, context)
            buf: list[str] = []
            for chunk in handle.create_chat_completion(**kwargs, stream=True):
                choice = chunk["choices"][0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                content = (choice.get("delta") or {}).get("content")
                if content:
                    buf.append(content)
            text = "".join(buf)
            if text:
                pieces.append(text)
            parsed = _parse_tool_calls(text, {t.function.name for t in request.tools})
            if parsed:
                yield "delta", {"role": "assistant",
                                "tool_calls": [{"index": i, **tc} for i, tc in enumerate(parsed)]}
                finish_reason = "tool_calls"
            elif text:
                yield "delta", {"content": text}
        else:
            kwargs = _effective_kwargs(request, cfg, context)
            for chunk in handle.create_chat_completion(**kwargs, stream=True):
                choice = chunk["choices"][0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    pieces.append(delta["content"])
                if delta:
                    yield "delta", delta

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

    # This runs at the very end of the stream; a failure here must NOT abort the
    # response before its [DONE] frame (re-tokenizing a long completion has been seen
    # to raise). Fall back to a whitespace estimate rather than killing the stream.
    try:
        completion_tokens = len(handle.tokenize(completion_text.encode("utf-8"), add_bos=False))
        total_tokens = getattr(handle, "n_tokens", 0) or completion_tokens
        return _usage(max(0, total_tokens - completion_tokens), completion_tokens)
    except Exception:  # noqa: BLE001 - usage is best-effort; never break the stream
        logger.warning("stream usage tally failed; estimating", exc_info=True)
        return _usage(0, len(completion_text.split()))
