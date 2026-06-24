"""Pydantic models for the (subset of the) OpenAI chat-completions API we support.

These types are intentionally small. They cover the fields a single-user local
server needs and ignore the rest. Pydantic drops unknown fields by default, so a
real OpenAI client can send extra keys (``presence_penalty``, ...) without
breaking anything here.

Adapted from the online server's ``shared/schemas.py`` — same wire shape, minus
the fleet-only ``WorkerInfo`` (there is no router/worker split here).
"""

from __future__ import annotations

import time
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Request types
# --------------------------------------------------------------------------- #


class ChatMessage(BaseModel):
    """A single turn in the conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """Body of ``POST /v1/chat/completions``.

    ``model`` is the name the client asks for; we serve exactly one model at a
    time, so a request for any other name is a 404.
    """

    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    # Standard sampling controls, passed through to the inference backend.
    top_p: float = 0.95
    top_k: int = 40
    seed: Optional[int] = None          # set for reproducible output
    stop: Optional[list[str]] = None    # strings that halt generation


# --------------------------------------------------------------------------- #
# Response types
# --------------------------------------------------------------------------- #


class Usage(BaseModel):
    """Token accounting. Populated with real counts from the inference result."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    """Body of a non-streaming chat-completions response."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)
