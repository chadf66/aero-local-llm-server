"""Constrained / structured output (response_format) tests, stub backend.

The stub can't truly constrain decoding, but it returns schema-conformant JSON so the
whole response_format path — parsing, OpenAI->llama translation, and the API surface —
is exercised with no model. Real grammar enforcement is llama.cpp's job.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from aero import engine, server
from aero.config import ModelConfig
from aero.schemas import ChatCompletionRequest

FAKE = "/fake/a.gguf"

SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "label": {"type": "string", "enum": ["spam", "not_spam"]},
    },
    "required": ["reasoning", "label"],
}


@pytest.fixture
def client():
    engine.configure({"m": ModelConfig(name="m", path=FAKE)}, backend="stub", idle_timeout=0)
    return TestClient(server.app)


def _req(**rf):
    body = {"model": "m", "messages": [{"role": "user", "content": "is this spam?"}]}
    if rf:
        body.update(rf)
    return ChatCompletionRequest(**body)


# --- parsing / validation -------------------------------------------------

def test_accepts_all_three_modes():
    assert _req(response_format={"type": "text"}).response_format.type == "text"
    assert _req(response_format={"type": "json_object"}).response_format.type == "json_object"
    r = _req(response_format={"type": "json_schema", "json_schema": {"schema": SCHEMA}})
    assert r.response_format.json_schema.schema_ == SCHEMA


def test_bad_type_rejected():
    with pytest.raises(ValidationError):
        _req(response_format={"type": "yaml"})


def test_json_schema_requires_a_schema():
    with pytest.raises(ValidationError):
        _req(response_format={"type": "json_schema"})  # no json_schema payload


# --- translation to llama.cpp shape --------------------------------------

def test_translate_text_and_default_are_none():
    assert engine._response_format_kwarg(_req()) is None
    assert engine._response_format_kwarg(_req(response_format={"type": "text"})) is None


def test_translate_json_object():
    assert engine._response_format_kwarg(
        _req(response_format={"type": "json_object"})
    ) == {"type": "json_object"}


def test_translate_json_schema_unwraps_to_llama_shape():
    kw = engine._response_format_kwarg(
        _req(response_format={"type": "json_schema", "json_schema": {"schema": SCHEMA}})
    )
    assert kw == {"type": "json_object", "schema": SCHEMA}


def test_tools_take_precedence_over_response_format():
    r = _req(
        response_format={"type": "json_object"},
        tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
    )
    assert engine._response_format_kwarg(r) is None  # tool calling is its own constrained mode


# --- end-to-end via the API (stub returns valid JSON) --------------------

def test_json_object_returns_parseable_json(client):
    body = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "x"}],
              "response_format": {"type": "json_object"}},
    ).json()
    content = body["choices"][0]["message"]["content"]
    assert json.loads(content) == {}  # valid JSON


def test_json_schema_output_conforms(client):
    body = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "x"}],
              "response_format": {"type": "json_schema", "json_schema": {"schema": SCHEMA}}},
    ).json()
    obj = json.loads(body["choices"][0]["message"]["content"])
    assert set(obj) == {"reasoning", "label"}        # required fields present
    assert obj["label"] in ["spam", "not_spam"]      # enum respected
    assert isinstance(obj["reasoning"], str)


def test_streaming_json_concatenates_to_valid_json(client):
    with client.stream(
        "POST", "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "x"}],
              "response_format": {"type": "json_schema", "json_schema": {"schema": SCHEMA}},
              "stream": True},
    ) as resp:
        frames = [json.loads(l[6:]) for l in resp.iter_lines()
                  if l.startswith("data: ") and l != "data: [DONE]"]
    text = "".join(f["choices"][0]["delta"].get("content", "")
                   for f in frames if f.get("choices"))
    obj = json.loads(text)
    assert obj["label"] in ["spam", "not_spam"]
