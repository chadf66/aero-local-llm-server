"""The HTTP server: one FastAPI app exposing an OpenAI-compatible API.

A single localhost app -- no router/worker split, health-check fleet, or auth.
It serves a *set* of known models (see engine.py), loading whichever one a request
names on demand and keeping at most one resident.
"""

from __future__ import annotations

import json
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from . import engine
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

    if not request.stream:
        text, finish_reason, usage = engine.run_inference(request)
        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )

    # Streaming: relay the engine's events as OpenAI chat.completion.chunk frames.
    def event_stream():
        for kind, payload in engine.stream_inference(request):
            if kind == "content":
                yield _sse(
                    {
                        "object": "chat.completion.chunk",
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {"content": payload}}],
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
