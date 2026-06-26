"""Embedding tests (Phase g1) — stub backend, temp home, no model/network.

The stub backend returns deterministic fake vectors, so the co-resident embedder
slot, the /v1/embeddings endpoint, and discovery can all be exercised without a GGUF.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aero import engine, server, store
from aero.config import ModelConfig
from aero.schemas import ChatCompletionRequest

CONFIGS = {"model-a": ModelConfig(name="model-a", path="/fake/a.gguf")}


@pytest.fixture
def home(tmp_path):
    emb = store.embedders_dir(tmp_path)
    emb.mkdir(parents=True)
    (emb / "bge-small.gguf").write_bytes(b"\x00")  # discovery stand-in
    engine.configure(CONFIGS, backend="stub", idle_timeout=0, home=tmp_path)
    return tmp_path


@pytest.fixture
def client():
    return TestClient(server.app)


def test_embed_is_deterministic_and_normalized(home):
    import math

    a1 = engine.embed("bge-small", ["hello world"])[0]
    a2 = engine.embed("bge-small", ["hello world"])[0]
    b = engine.embed("bge-small", ["something else"])[0]
    assert a1 == a2                      # same text -> same vector
    assert a1 != b                       # different text -> different vector
    assert abs(math.sqrt(sum(x * x for x in a1)) - 1.0) < 1e-6   # unit length


def test_available_embedders_scans_dir(home):
    assert engine.available_embedders() == ["bge-small"]
    # Chat registry must NOT include the embedder.
    assert "bge-small" not in engine.available_models()


def test_embedder_is_co_resident_with_chat_model(home):
    # Load a chat model, then embed: the chat model must stay resident (second slot).
    engine.run_inference(
        ChatCompletionRequest(model="model-a", messages=[{"role": "user", "content": "hi"}])
    )
    assert engine.loaded_model() == "model-a"
    engine.embed("bge-small", ["x"])
    assert engine.loaded_model() == "model-a"      # not evicted
    assert engine.loaded_embedder() == "bge-small"


def test_embeddings_endpoint_single_and_batch(home, client):
    single = client.post("/v1/embeddings", json={"model": "bge-small", "input": "hello"}).json()
    assert single["object"] == "list" and len(single["data"]) == 1
    assert single["data"][0]["object"] == "embedding"
    assert isinstance(single["data"][0]["embedding"][0], float)

    batch = client.post("/v1/embeddings", json={"model": "bge-small", "input": ["a", "b", "c"]}).json()
    assert [d["index"] for d in batch["data"]] == [0, 1, 2]
    assert batch["usage"]["total_tokens"] >= 3


def test_embeddings_empty_input_is_400(home, client):
    assert client.post("/v1/embeddings", json={"model": "bge-small", "input": []}).status_code == 400


def test_api_embedders_lists_installed(home, client):
    body = client.get("/api/embedders").json()
    assert body["embedders"] == ["bge-small"]
