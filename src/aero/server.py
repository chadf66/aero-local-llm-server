"""The HTTP server: one FastAPI app exposing an OpenAI-compatible API.

A single localhost app -- no router/worker split, health-check fleet, or auth.
It serves a *set* of known models (see engine.py), loading whichever one a request
names on demand and keeping at most one resident.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from . import db, engine, store
from .schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)

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
    def event_stream():
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
