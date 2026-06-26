"""RAG-in-chat tests (Phase g3): retrieval injection + citations, stub engine.

A model with a `knowledge` base retrieves chunks for the last user message and
returns them as `sources` (non-streaming) / a `sources` SSE frame (streaming). The
stub backend means no real model/embedder is needed; LanceDB does the real storage.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from aero import engine, rag, server
from aero.config import ModelConfig
from aero.schemas import ChatCompletionRequest

FAKE = "/fake/a.gguf"


@pytest.fixture
def home(tmp_path):
    # Configure first so rag.create_kb/ingest can use the stub embedder.
    engine.configure(
        {
            "plain": ModelConfig(name="plain", path=FAKE),
            "grounded": ModelConfig(name="grounded", path=FAKE, knowledge="docs"),
        },
        backend="stub", idle_timeout=0, home=tmp_path,
    )
    rag.create_kb(tmp_path, "docs", "stub-embedder")
    doc = tmp_path / "facts.txt"
    doc.write_text("the capital of france is paris")
    rag.ingest(tmp_path, "docs", [doc])
    return tmp_path


@pytest.fixture
def client():
    return TestClient(server.app)


def _req(model, text):
    return ChatCompletionRequest(model=model, messages=[{"role": "user", "content": text}])


def test_knowledge_not_in_load_key():
    # Same weights/ctx/kv/format -> a KB difference is a no-reload persona swap.
    a = ModelConfig(name="a", path=FAKE)
    b = ModelConfig(name="b", path=FAKE, knowledge="docs")
    assert a.load_key() == b.load_key()


def test_run_inference_returns_sources_for_grounded_model(home):
    _msg, _fin, _usage, sources = engine.run_inference(
        _req("grounded", "the capital of france is paris")
    )
    assert sources and sources[0]["source"] == "facts.txt"


def test_run_inference_no_sources_without_knowledge(home):
    *_, sources = engine.run_inference(_req("plain", "anything"))
    assert sources == []


def test_toggling_knowledge_does_not_reload(home):
    engine.run_inference(_req("grounded", "x"))
    calls = engine._load_calls
    engine.run_inference(_req("plain", "y"))  # same load_key -> persona swap, no reload
    assert engine._load_calls == calls and engine.loaded_model() == "plain"


def test_nonstreaming_response_includes_sources(home, client):
    body = client.post(
        "/v1/chat/completions",
        json={"model": "grounded", "messages": [{"role": "user", "content": "capital of france?"}]},
    ).json()
    assert body["sources"] and body["sources"][0]["source"] == "facts.txt"
    # Plain model omits the field entirely (standard OpenAI shape).
    plain = client.post(
        "/v1/chat/completions",
        json={"model": "plain", "messages": [{"role": "user", "content": "hi"}]},
    ).json()
    assert plain.get("sources") is None


def test_streaming_emits_a_sources_frame(home, client):
    with client.stream(
        "POST", "/v1/chat/completions",
        json={"model": "grounded", "messages": [{"role": "user", "content": "capital?"}],
              "stream": True},
    ) as resp:
        frames = [json.loads(l[6:]) for l in resp.iter_lines()
                  if l.startswith("data: ") and l != "data: [DONE]"]
    src_frames = [f for f in frames if f.get("sources")]
    assert src_frames and src_frames[0]["sources"][0]["source"] == "facts.txt"
    # The sources frame is a valid (empty-delta) chunk so OpenAI parsers don't choke.
    assert src_frames[0]["choices"][0]["delta"] == {}


def test_missing_kb_degrades_gracefully(home):
    engine.reload({"ghost": ModelConfig(name="ghost", path=FAKE, knowledge="does-not-exist")})
    *_, sources = engine.run_inference(_req("ghost", "q"))
    assert sources == []  # no crash, just ungrounded


def test_admin_sets_knowledge_on_model(home, client):
    # Needs a real GGUF for write_model_config's weights check; create one.
    from aero import store
    (store.gguf_dir(home)).mkdir(parents=True, exist_ok=True)
    (store.gguf_dir(home) / "base.gguf").write_bytes(b"\x00")
    resp = client.post("/api/models", json={"name": "kbmodel", "from": "base", "knowledge": "docs"})
    assert resp.status_code == 200, resp.text
    detail = next(m for m in client.get("/api/models").json()["models"] if m["name"] == "kbmodel")
    assert detail["knowledge"] == "docs"
