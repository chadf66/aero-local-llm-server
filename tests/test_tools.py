"""Tool-calling tests using the stub engine (no model needed).

`model-a` is plain; `qwen-tools` is tool-enabled (effective chat_format is a
function-calling handler) so the stub simulates a tool call for it.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from aero import engine, server
from aero.config import ModelConfig

CONFIGS = {
    "model-a": ModelConfig(name="model-a", path="/fake/a.gguf"),
    "qwen-tools": ModelConfig(name="qwen-tools", path="/fake/a.gguf", tools=True),
}

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


@pytest.fixture(autouse=True)
def stub_engine():
    engine.configure(CONFIGS, backend="stub", idle_timeout=0)
    yield


@pytest.fixture
def client():
    return TestClient(server.app)


def test_supports_tools_reflects_config():
    assert engine.supports_tools("qwen-tools") is True
    assert engine.supports_tools("model-a") is False


def test_parse_tool_calls_hermes_format():
    text = 'sure!\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "SF"}}\n</tool_call>'
    calls = engine._parse_tool_calls(text, {"get_weather"})
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "SF"}


def test_parse_tool_calls_bare_object_llama_format():
    # Llama-3.1 emits a bare object using "parameters".
    text = '{"name": "get_weather", "parameters": {"city": "San Francisco"}}'
    calls = engine._parse_tool_calls(text, {"get_weather"})
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "San Francisco"}


def test_parse_tool_calls_bare_array_ministral_format():
    text = '[{"name": "get_weather", "arguments": {"city": "SF"}}]'
    calls = engine._parse_tool_calls(text, {"get_weather"})
    assert len(calls) == 1 and calls[0]["function"]["name"] == "get_weather"


def test_parse_tool_calls_ignores_unknown_name_in_bare_json():
    # A plain JSON answer that isn't one of the tools must not be treated as a call.
    assert engine._parse_tool_calls('{"name": "Bob", "age": 30}', {"get_weather"}) is None


def test_parse_tool_calls_none_for_plain_text():
    assert engine._parse_tool_calls("Just a normal answer.", {"get_weather"}) is None


def test_normalize_renders_tool_messages_to_text():
    # assistant tool_calls + tool result -> plain system/user/assistant strings the
    # native template can handle (no missing 'content', no tool/assistant-calls roles).
    raw = [
        {"role": "user", "content": "weather in SF?"},
        {"role": "assistant", "tool_calls": [
            {"id": "call_0", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city": "SF"}'}}]},
        {"role": "tool", "tool_call_id": "call_0", "content": "72F sunny"},
    ]
    out = engine._normalize_messages(raw)
    assert all(isinstance(m.get("content"), str) for m in out)
    assert all(m["role"] in {"system", "user", "assistant"} for m in out)
    assert "<tool_call>" in out[1]["content"] and "get_weather" in out[1]["content"]
    assert "<tool_response>" in out[2]["content"] and "72F sunny" in out[2]["content"]


def test_tool_system_prompt_injected_into_kwargs():
    from aero.config import ModelConfig
    from aero.schemas import ChatCompletionRequest
    req = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], tools=[WEATHER_TOOL]
    )
    kw = engine._effective_kwargs(req, ModelConfig(name="m", path="/x", tools=True))
    assert kw["messages"][0]["role"] == "system"
    assert "<tools>" in kw["messages"][0]["content"] and "get_weather" in kw["messages"][0]["content"]
    # tools are handled via the prompt, not passed to llama.cpp
    assert "tools" not in kw and "tool_choice" not in kw


def test_tools_on_non_tool_model_returns_400(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}], "tools": [WEATHER_TOOL]},
    )
    assert resp.status_code == 400
    assert "tools = true" in resp.json()["detail"]


def test_non_streaming_tool_call(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "qwen-tools", "messages": [{"role": "user", "content": "weather in SF?"}],
              "tools": [WEATHER_TOOL]},
    )
    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    calls = choice["message"]["tool_calls"]
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "get_weather"
    assert choice["message"]["content"] is None


def test_streaming_tool_call_delta(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "qwen-tools", "messages": [{"role": "user", "content": "weather in SF?"}],
              "tools": [WEATHER_TOOL], "stream": True},
    ) as resp:
        frames = [f for f in resp.iter_lines() if f.startswith("data: ") and f != "data: [DONE]"]
    objs = [json.loads(f[6:]) for f in frames]
    # A delta frame carries tool_calls, and a later frame reports finish_reason.
    assert any(o["choices"] and o["choices"][0].get("delta", {}).get("tool_calls") for o in objs)
    assert any(o["choices"] and o["choices"][0].get("finish_reason") == "tool_calls" for o in objs)


def test_tool_result_message_round_trips(client):
    """A follow-up turn with an assistant tool_calls message and a tool result
    must validate and complete (the second half of an agent loop)."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen-tools",
            "messages": [
                {"role": "user", "content": "weather in SF?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_0", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\": \"SF\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_0", "content": "72F and sunny"},
            ],
        },
    )
    assert resp.status_code == 200
