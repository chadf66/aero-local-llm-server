"""The HTTP server: one FastAPI app exposing an OpenAI-compatible API.

A single localhost app -- no router/worker split, health-check fleet, or auth.
It serves a *set* of known models (see engine.py), loading whichever one a request
names on demand and keeping at most one resident.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from . import db, engine, store, store_ops
from .schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)

logger = logging.getLogger("aero.server")

app = FastAPI(title="aero")


@app.get("/healthz")
def healthz() -> dict:
    """Liveness + which models are available and which (if any) is resident."""
    return {
        "status": "ok",
        "models": engine.available_models(),
        "loaded": engine.loaded_model(),
    }


@app.get("/v1/models")
def list_models() -> dict:
    """OpenAI-compatible list of every model this server can serve."""
    now = int(time.time())
    data = [
        {"id": name, "object": "model", "created": now, "owned_by": "aero"}
        for name in engine.available_models()
    ]
    return {"object": "list", "data": data}


def _sse(data: dict) -> str:
    """Format one Server-Sent Events frame the way OpenAI clients expect."""
    return f"data: {json.dumps(data)}\n\n"


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if request.model not in engine.available_models():
        raise HTTPException(
            status_code=404,
            detail=f"model {request.model!r} not available (have: {engine.available_models()})",
        )
    if request.tools and not engine.supports_tools(request.model):
        raise HTTPException(
            status_code=400,
            detail=f"model {request.model!r} is not tool-enabled; set `tools = true` in its "
                   f"config (e.g. a derived model with `from`) to use the `tools` parameter.",
        )

    if not request.stream:
        message, finish_reason, usage = engine.run_inference(request)
        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(
                        role="assistant",
                        content=message.get("content"),
                        tool_calls=message.get("tool_calls"),
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )

    # Streaming: relay the engine's events as OpenAI chat.completion.chunk frames.
    # The whole loop is guarded so that *any* failure (including in the end-of-stream
    # usage tally) still closes the response cleanly with an error frame + [DONE],
    # rather than resetting the connection — which the browser surfaces as an opaque
    # "network error" even though the answer already arrived.
    def event_stream():
        try:
            for kind, payload in engine.stream_inference(request):
                if kind == "delta":
                    yield _sse(
                        {
                            "object": "chat.completion.chunk",
                            "model": request.model,
                            "choices": [{"index": 0, "delta": payload}],
                        }
                    )
                else:  # "end" -> (finish_reason, usage)
                    finish_reason, usage = payload
                    yield _sse(
                        {
                            "object": "chat.completion.chunk",
                            "model": request.model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                        }
                    )
                    # OpenAI-style trailing usage chunk (empty choices, usage populated).
                    yield _sse(
                        {
                            "object": "chat.completion.chunk",
                            "model": request.model,
                            "choices": [],
                            "usage": usage.model_dump(),
                        }
                    )
        except Exception as exc:  # noqa: BLE001 - report, then still close cleanly
            logger.exception("streaming inference failed for %s", request.model)
            yield _sse({"error": {"message": str(exc), "type": "server_error"}})
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# =========================================================================== #
# Web-UI API (Phase f). Distinct from the OpenAI surface above: these endpoints
# serve the bundled UI — live server state, a context-size preview, and durable
# conversation history (db.py). They are *not* on the inference path: the UI still
# generates via /v1/chat/completions and posts the resulting turns back here.
# =========================================================================== #


@app.get("/api/state")
def ui_state() -> dict:
    """Everything the UI needs to render the model picker + resident badge."""
    return {"models": engine.model_info(), "loaded": engine.loaded_model()}


@app.get("/api/sizing")
def ui_sizing(model: str, kv_cache_type: str = "f16") -> dict:
    """Largest context that fits memory for a model at a KV precision (or null)."""
    return {
        "model": model,
        "kv_cache_type": kv_cache_type,
        "n_ctx": engine.context_preview(model, kv_cache_type),
    }


class ConversationCreate(BaseModel):
    title: str = "New chat"
    model: Optional[str] = None
    system: Optional[str] = None


class ConversationPatch(BaseModel):
    title: Optional[str] = None
    model: Optional[str] = None
    system: Optional[str] = None


class MessageCreate(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[Any] = None


@app.get("/api/conversations")
def list_conversations(q: Optional[str] = None) -> dict:
    """List conversations (most-recent first), or search by title/content with ``q``."""
    rows = db.search(q) if q else db.list_conversations()
    return {"conversations": rows}


@app.post("/api/conversations")
def create_conversation(body: ConversationCreate) -> dict:
    return db.create_conversation(body.title, model=body.model, system=body.system)


@app.get("/api/conversations/{cid}")
def get_conversation(cid: str) -> dict:
    conv = db.get_conversation(cid)
    if conv is None:
        raise HTTPException(status_code=404, detail=f"conversation {cid!r} not found")
    return conv


@app.patch("/api/conversations/{cid}")
def patch_conversation(cid: str, body: ConversationPatch) -> dict:
    if not db.update_conversation(cid, title=body.title, model=body.model, system=body.system):
        raise HTTPException(status_code=404, detail=f"conversation {cid!r} not found")
    return db.get_conversation(cid)


@app.delete("/api/conversations/{cid}")
def delete_conversation(cid: str) -> dict:
    if not db.delete_conversation(cid):
        raise HTTPException(status_code=404, detail=f"conversation {cid!r} not found")
    return {"deleted": cid}


@app.post("/api/conversations/{cid}/messages")
def add_message(cid: str, body: MessageCreate) -> dict:
    msg = db.add_message(cid, body.role, body.content, tool_calls=body.tool_calls)
    if msg is None:
        raise HTTPException(status_code=404, detail=f"conversation {cid!r} not found")
    return msg


@app.delete("/api/conversations/{cid}/messages/last")
def delete_last_message(cid: str) -> dict:
    """Drop the latest message — the UI's 'regenerate' deletes the old reply first."""
    msg = db.delete_last_message(cid)
    if msg is None:
        raise HTTPException(status_code=404, detail="no messages to delete")
    return msg


# =========================================================================== #
# Model management (Phase f2). Pull from Hugging Face, create/edit/delete model
# configs — all reusing store_ops (the same logic the CLI uses) and applying live
# via engine.reload_from_disk(), so changes take effect with no server restart.
# =========================================================================== #

# Only one download at a time on a single-user box.
_pull_active = threading.Lock()


def _require_home() -> Path:
    home = engine.home()
    if home is None:
        raise HTTPException(status_code=503, detail="model management requires `aero serve`")
    return home


def _model_detail(name: str, cfg, registry: dict) -> dict:
    p = Path(cfg.path)
    home = engine.home()
    toml_path = (store.config_dir(home) / f"{name}.toml") if home else None
    return {
        "name": name,
        "base": cfg.base,
        "path": cfg.path,
        "size": p.stat().st_size if p.is_file() else None,
        "exists": p.is_file(),
        "n_ctx": cfg.n_ctx,
        "kv_cache_type": cfg.kv_cache_type,
        "max_tokens": cfg.max_tokens,
        "tools": engine.supports_tools(name),
        "chat_format": cfg.chat_format,
        "system": cfg.system,
        "sampling": cfg.sampling.model_dump(exclude_none=True),
        "has_config_file": bool(toml_path and toml_path.is_file()),
        "referenced_by": sorted(n for n, c in registry.items() if n != name and c.path == cfg.path),
    }


@app.get("/api/models")
def list_installed_models() -> dict:
    """Installed models with size, base, config fields, and reference info (for delete)."""
    registry = engine.current_models()
    return {"models": [_model_detail(n, c, registry) for n, c in sorted(registry.items())]}


@app.get("/api/models/repo")
def list_repo_models(repo: str) -> dict:
    """The GGUF quants available in a Hugging Face repo (for the pull picker)."""
    try:
        return {"repo": repo, "files": store_ops.list_repo_ggufs(repo)}
    except Exception as exc:  # noqa: BLE001 - surface HF errors as a clean 400
        raise HTTPException(status_code=400, detail=f"could not list {repo!r}: {exc}")


@app.get("/api/models/pull")
def pull_model(repo: str, filename: str):
    """Stream a GGUF download as SSE progress, then register it and reload (no restart)."""
    home = _require_home()
    if not _pull_active.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="a download is already in progress")

    def event_stream():
        events: queue.Queue = queue.Queue()

        def worker():
            try:
                store_ops.download_gguf(
                    repo, filename, store.gguf_dir(home),
                    progress_cb=lambda d, t: events.put(("progress", (d, t))),
                )
                stem = Path(filename).stem
                store_ops.write_starter_config(home, stem)
                engine.reload_from_disk()
                events.put(("done", {"name": stem}))
            except Exception as exc:  # noqa: BLE001 - report to the client, don't crash
                events.put(("error", str(exc)))
            finally:
                events.put(("__end__", None))

        threading.Thread(target=worker, name="aero-pull", daemon=True).start()
        last = 0.0
        try:
            while True:
                kind, payload = events.get()
                if kind == "__end__":
                    break
                if kind == "progress":
                    d, t = payload
                    now = time.monotonic()
                    if now - last >= 0.3 or (t and d >= t):  # throttle to ~3/sec
                        last = now
                        yield _sse({"type": "progress", "downloaded": d, "total": t,
                                    "pct": round(d / t * 100, 1) if t else None})
                elif kind == "done":
                    yield _sse({"type": "done", **payload})
                elif kind == "error":
                    yield _sse({"type": "error", "detail": payload})
            yield "data: [DONE]\n\n"
        finally:
            _pull_active.release()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class ModelConfigBody(BaseModel):
    """Editable config fields (mirrors the `.toml`). All optional; unset = use defaults."""
    model_config = ConfigDict(populate_by_name=True)

    name: Optional[str] = None                       # required for create (POST)
    base: Optional[str] = Field(default=None, alias="from")
    system: Optional[str] = None
    n_ctx: Optional[Union[int, str]] = None          # int or "auto"
    kv_cache_type: Optional[str] = None
    max_tokens: Optional[int] = None
    tools: Optional[bool] = None
    chat_format: Optional[str] = None
    sampling: Optional[dict] = None


def _fields(body: ModelConfigBody) -> dict:
    f: dict = {}
    if body.base is not None:
        f["from"] = body.base
    for k in ("system", "n_ctx", "kv_cache_type", "max_tokens", "tools", "chat_format"):
        v = getattr(body, k)
        if v is not None:
            f[k] = v
    if body.sampling:
        f["sampling"] = body.sampling
    return f


def _save_model(home: Path, name: str, body: ModelConfigBody) -> dict:
    try:
        store_ops.write_model_config(home, name, _fields(body))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    models = engine.reload_from_disk()
    return {"name": store_ops.sanitize_name(name), "models": models}


@app.post("/api/models")
def create_model(body: ModelConfigBody) -> dict:
    home = _require_home()
    if not body.name:
        raise HTTPException(status_code=400, detail="`name` is required to create a model")
    return _save_model(home, body.name, body)


@app.put("/api/models/{name}")
def edit_model(name: str, body: ModelConfigBody) -> dict:
    home = _require_home()
    return _save_model(home, name, body)


@app.delete("/api/models/{name}")
def remove_model(name: str, weights: bool = False) -> dict:
    home = _require_home()
    try:
        result = store_ops.delete_model(home, name, engine.current_models(), weights=weights)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    result["models"] = engine.reload_from_disk()
    return result


# --------------------------------------------------------------------------- #
# Static UI. A single catch-all registered LAST so it can't shadow the API routes
# above (Starlette matches in registration order; the specific /api and /v1 routes
# win). It resolves the built assets at *request* time via store.webui_dist(): any
# real file is served directly, anything else falls back to index.html (SPA
# routing), and if the UI hasn't been built (`make ui`) it returns a short hint.
# --------------------------------------------------------------------------- #

_UI_HINT = (
    "<html><body style='font-family:system-ui;max-width:40rem;margin:4rem auto'>"
    "<h1>aero</h1><p>The web UI isn't built yet. From the repo root run:</p>"
    "<pre>make ui</pre><p>then reload this page. The API is already live at "
    "<code>/v1</code> and <code>/api</code>.</p></body></html>"
)


@app.get("/{full_path:path}", include_in_schema=False)
def serve_ui(full_path: str):
    dist = store.webui_dist()
    if dist is None:
        return HTMLResponse(_UI_HINT)
    # Serve a real asset if the path points at one inside dist (guard traversal),
    # else hand back index.html so the SPA can route the URL itself.
    candidate = (dist / full_path).resolve()
    if full_path and dist.resolve() in candidate.parents and candidate.is_file():
        return FileResponse(candidate)
    return FileResponse(dist / "index.html")
