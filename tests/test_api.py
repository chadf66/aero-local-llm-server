"""API + lifecycle tests using the stub engine.

The stub backend means no GGUF and no llama-cpp-python are needed, so this runs
anywhere `pip install -e .[dev]` works. Because there's no router/worker split,
we drive the FastAPI app directly with TestClient -- no background server needed.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from aero import engine, server
from aero.config import ModelConfig

# A two-model config set; the stub backend ignores the (fake) paths.
CONFIGS = {
    "model-a": ModelConfig(name="model-a", path="/fake/a.gguf"),
    "model-b": ModelConfig(name="model-b", path="/fake/b.gguf"),
}
MODEL = "model-a"


@pytest.fixture(autouse=True)
def stub_engine():
    """Configure the engine in stub mode before each test (idle timer off)."""
    engine.configure(CONFIGS, backend="stub", idle_timeout=0)
    yield


@pytest.fixture
def client():
    return TestClient(server.app)


# --------------------------------------------------------------------------- #
# OpenAI API surface
# --------------------------------------------------------------------------- #


def test_non_streaming_completion(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"model": MODEL, "messages": [{"role": "user", "content": "ping"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == MODEL
    assert body["choices"][0]["message"]["content"] == f"[stub:{MODEL}] echo: ping"
    assert body["choices"][0]["finish_reason"] == "stop"
    usage = body["usage"]
    assert usage["prompt_tokens"] > 0 and usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_streaming_completion(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": MODEL, "messages": [{"role": "user", "content": "ping"}], "stream": True},
    ) as resp:
        assert resp.status_code == 200
        frames = [line for line in resp.iter_lines() if line.startswith("data: ")]

    assert frames[-1] == "data: [DONE]"
    assert any('"delta"' in f and '"content"' in f for f in frames[:-1])
    assert any('"finish_reason": "stop"' in f for f in frames)
    assert any('"usage"' in f for f in frames)


def test_list_models_shows_all(client):
    data = client.get("/v1/models").json()["data"]
    assert [m["id"] for m in data] == ["model-a", "model-b"]


def test_unknown_model_returns_404(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "does-not-exist", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Phase b: model lifecycle
# --------------------------------------------------------------------------- #


def test_load_on_demand(client):
    # Nothing resident until the first request names a model.
    assert engine.loaded_model() is None
    client.post("/v1/chat/completions", json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]})
    assert engine.loaded_model() == "model-a"
    assert client.get("/healthz").json()["loaded"] == "model-a"


def test_evict_before_load(client):
    client.post("/v1/chat/completions", json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]})
    assert engine.loaded_model() == "model-a"
    # Requesting a different model swaps it in; only one is ever resident.
    client.post("/v1/chat/completions", json={"model": "model-b", "messages": [{"role": "user", "content": "hi"}]})
    assert engine.loaded_model() == "model-b"


def test_idle_unload_seam(client):
    # With a tiny timeout, the idle check frees the model once it's stale.
    engine.configure(CONFIGS, backend="stub", idle_timeout=0.01)
    client.post("/v1/chat/completions", json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]})
    assert engine.loaded_model() == "model-a"
    time.sleep(0.05)
    assert engine._unload_if_idle() is True
    assert engine.loaded_model() is None


def test_idle_timeout_zero_never_unloads(client):
    # idle_timeout=0 (the autouse default) disables idle-unload entirely.
    client.post("/v1/chat/completions", json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]})
    time.sleep(0.02)
    assert engine._unload_if_idle() is False
    assert engine.loaded_model() == "model-a"
