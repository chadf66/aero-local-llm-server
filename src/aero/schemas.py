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
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --------------------------------------------------------------------------- #
# Tool-calling types (OpenAI "function" tools)
# --------------------------------------------------------------------------- #


class FunctionDef(BaseModel):
    """A tool's function signature, as sent by the client in ``tools``."""

    name: str
    description: Optional[str] = None
    parameters: dict[str, Any] = Field(default_factory=dict)  # JSON Schema


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDef


class FunctionCall(BaseModel):
    name: str
    arguments: str          # a JSON string, per the OpenAI wire format


class ToolCall(BaseModel):
    """A tool call the model emitted (and the client should execute)."""

    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}")
    type: Literal["function"] = "function"
    function: FunctionCall


# --------------------------------------------------------------------------- #
# Request types
# --------------------------------------------------------------------------- #


class JsonSchemaSpec(BaseModel):
    """The ``json_schema`` payload of a ``response_format`` (OpenAI Structured Outputs)."""

    model_config = ConfigDict(populate_by_name=True)

    name: Optional[str] = None
    # 'schema' collides with BaseModel.schema(); accept the wire key via alias.
    schema_: dict[str, Any] = Field(alias="schema")
    strict: Optional[bool] = None  # accepted; grammar decoding is always strict


class ResponseFormat(BaseModel):
    """Constrains generation. ``text`` (default), ``json_object`` (any valid JSON),
    or ``json_schema`` (output must conform to the given JSON Schema)."""

    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: Optional[JsonSchemaSpec] = None

    @model_validator(mode="after")
    def _require_schema(self) -> "ResponseFormat":
        if self.type == "json_schema" and self.json_schema is None:
            raise ValueError("response_format.type 'json_schema' requires a 'json_schema'")
        return self


class ChatMessage(BaseModel):
    """A single turn in the conversation.

    ``content`` is optional because an assistant turn that *only* calls tools has
    null content, and a ``tool`` turn carries a result keyed by ``tool_call_id``.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None   # assistant -> tool calls it made
    tool_call_id: Optional[str] = None            # tool -> which call this answers


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
    # Tool calling. ``tool_choice`` is "auto" | "none" | {"type":"function",...}.
    tools: Optional[list[Tool]] = None
    tool_choice: Optional[Union[str, dict]] = None
    # Constrained/structured output (OpenAI-compatible). Ignored when ``tools`` is set
    # (tool calling is its own constrained mode).
    response_format: Optional[ResponseFormat] = None


# --------------------------------------------------------------------------- #
# Response types
# --------------------------------------------------------------------------- #


class EmbeddingRequest(BaseModel):
    """Body of ``POST /v1/embeddings`` (OpenAI-compatible).

    ``input`` is one string or a list of strings; ``model`` names an installed
    embedder (see engine's embedder slot)."""

    model: str
    input: Union[str, list[str]]


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
    # aero extension: retrieved RAG sources, when the model has a knowledge base.
    # Additive and omitted when empty, so standard OpenAI clients are unaffected.
    sources: Optional[list[dict]] = None
