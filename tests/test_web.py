"""Web-UI API tests (server.py /api routes) with the stub engine + a temp db."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aero import db, engine, store
from aero.config import ModelConfig

CONFIGS = {
    "model-a": ModelConfig(name="model-a", path="/fake/a.gguf", n_ctx=4096),
    "qwen-tools": ModelConfig(name="qwen-tools", path="/fake/a.gguf", tools=True),
}


@pytest.fixture(autouse=True)
def setup(tmp_path):
    engine.configure(CONFIGS, backend="stub", idle_timeout=0)
    db.connect(tmp_path)
    yield


@pytest.fixture
def client():
    from aero import server
    return TestClient(server.app)


# --------------------------------------------------------------------------- #
# State + sizing
# --------------------------------------------------------------------------- #


def test_state_lists_models_and_resident(client):
    body = client.get("/api/state").json()
    names = {m["name"] for m in body["models"]}
    assert names == {"model-a", "qwen-tools"}
    tools = {m["name"]: m["tools"] for m in body["models"]}
    assert tools["qwen-tools"] is True and tools["model-a"] is False
    assert body["loaded"] is None  # nothing loaded until a request names a model


def test_sizing_returns_context_number(client, monkeypatch):
    # context_preview needs a real GGUF/Metal backend; stub it for the wiring test.
    monkeypatch.setattr(engine, "context_preview", lambda name, kv: 50944)
    body = client.get("/api/sizing?model=model-a&kv_cache_type=q8_0").json()
    assert body == {"model": "model-a", "kv_cache_type": "q8_0", "n_ctx": 50944}


def test_sizing_degrades_to_null_on_stub(client):
    # No monkeypatch: the stub path can't read a fake GGUF, so n_ctx is null (not 500).
    body = client.get("/api/sizing?model=model-a").json()
    assert body["n_ctx"] is None


# --------------------------------------------------------------------------- #
# Conversation history CRUD
# --------------------------------------------------------------------------- #


def test_conversation_crud_round_trip(client):
    created = client.post("/api/conversations", json={"title": "hi", "model": "model-a"}).json()
    cid = created["id"]

    client.post(f"/api/conversations/{cid}/messages", json={"role": "user", "content": "hello"})
    client.post(f"/api/conversations/{cid}/messages",
                json={"role": "assistant", "content": "hi there"})

    conv = client.get(f"/api/conversations/{cid}").json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]

    client.patch(f"/api/conversations/{cid}", json={"title": "renamed"})
    assert client.get(f"/api/conversations/{cid}").json()["title"] == "renamed"

    listed = client.get("/api/conversations").json()["conversations"]
    assert any(c["id"] == cid for c in listed)

    assert client.delete(f"/api/conversations/{cid}").status_code == 200
    assert client.get(f"/api/conversations/{cid}").status_code == 404


def test_search_endpoint(client):
    a = client.post("/api/conversations", json={"title": "about pelicans"}).json()
    b = client.post("/api/conversations", json={"title": "other"}).json()
    client.post(f"/api/conversations/{b['id']}/messages",
                json={"role": "user", "content": "explain quantization"})
    hits_title = {c["id"] for c in client.get("/api/conversations?q=pelican").json()["conversations"]}
    hits_content = {c["id"] for c in client.get("/api/conversations?q=quantization").json()["conversations"]}
    assert a["id"] in hits_title
    assert b["id"] in hits_content


def test_delete_last_message_endpoint(client):
    cid = client.post("/api/conversations", json={"title": "c"}).json()["id"]
    client.post(f"/api/conversations/{cid}/messages", json={"role": "user", "content": "q"})
    client.post(f"/api/conversations/{cid}/messages", json={"role": "assistant", "content": "a"})
    removed = client.delete(f"/api/conversations/{cid}/messages/last").json()
    assert removed["role"] == "assistant"
    conv = client.get(f"/api/conversations/{cid}").json()
    assert [m["role"] for m in conv["messages"]] == ["user"]


def test_missing_conversation_is_404(client):
    assert client.get("/api/conversations/conv_nope").status_code == 404
    assert client.post("/api/conversations/conv_nope/messages",
                       json={"role": "user", "content": "x"}).status_code == 404


# --------------------------------------------------------------------------- #
# Static-mount fallback
# --------------------------------------------------------------------------- #


def test_root_shows_build_hint_when_ui_absent(client, monkeypatch):
    # With no built UI, the root returns the build hint (not a 404).
    monkeypatch.setattr(store, "webui_dist", lambda: None)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "make ui" in resp.text


def test_root_serves_index_when_ui_built(client, monkeypatch, tmp_path):
    # When a build exists, the catch-all serves index.html for app routes.
    (tmp_path / "index.html").write_text("<!doctype html><div id=app></div>")
    monkeypatch.setattr(store, "webui_dist", lambda: tmp_path)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "id=app" in resp.text
