"""Tests for the engine's request/config merge (_effective_kwargs)."""

from __future__ import annotations

from aero import engine
from aero.config import ModelConfig, SamplingConfig
from aero.schemas import ChatCompletionRequest


def _req(**kw):
    base = dict(model="m", messages=[{"role": "user", "content": "hi"}])
    base.update(kw)
    return ChatCompletionRequest(**base)


def test_system_prompt_injected_when_absent():
    cfg = ModelConfig(name="m", path="/x", system="You are helpful.")
    msgs = engine._effective_kwargs(_req(), cfg)["messages"]
    assert msgs[0] == {"role": "system", "content": "You are helpful."}
    assert msgs[1]["role"] == "user"


def test_system_prompt_not_injected_when_request_has_one():
    cfg = ModelConfig(name="m", path="/x", system="default")
    req = _req(messages=[{"role": "system", "content": "custom"}, {"role": "user", "content": "hi"}])
    msgs = engine._effective_kwargs(req, cfg)["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == "custom"  # the request's system wins; no duplicate


def test_sampling_precedence_request_over_config_over_default():
    cfg = ModelConfig(name="m", path="/x", sampling=SamplingConfig(temperature=0.2, top_p=0.5))
    kw = engine._effective_kwargs(_req(temperature=0.9), cfg)
    assert kw["temperature"] == 0.9  # explicit request field wins
    assert kw["top_p"] == 0.5        # falls back to config
    assert kw["top_k"] == 40         # falls back to schema default (neither set)


def test_max_tokens_from_config_then_request():
    cfg = ModelConfig(name="m", path="/x", max_tokens=256)
    assert engine._effective_kwargs(_req(), cfg)["max_tokens"] == 256
    assert engine._effective_kwargs(_req(max_tokens=10), cfg)["max_tokens"] == 10


def test_no_reload_when_personas_share_weights():
    shared = "/fake/shared.gguf"
    engine.configure(
        {
            "persona-a": ModelConfig(name="persona-a", path=shared, system="A"),
            "persona-b": ModelConfig(name="persona-b", path=shared, system="B"),
            "other": ModelConfig(name="other", path="/fake/other.gguf"),
        },
        backend="stub",
        idle_timeout=0,
    )
    start = engine._load_calls

    engine.run_inference(_req(model="persona-a"))
    assert engine.loaded_model() == "persona-a"
    assert engine._load_calls == start + 1

    # Same weights/context, different system prompt -> switch persona, no reload.
    engine.run_inference(_req(model="persona-b"))
    assert engine.loaded_model() == "persona-b"
    assert engine._load_calls == start + 1

    # Different weights -> real reload (evict-before-load).
    engine.run_inference(_req(model="other"))
    assert engine.loaded_model() == "other"
    assert engine._load_calls == start + 2
