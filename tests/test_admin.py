"""Model-management API tests (Phase f2) — stub engine, temp home, no network.

These exercise store_ops + the /api/models endpoints + the live reload_from_disk
path. A real GGUF is never needed: empty files stand in for weights (the stub
backend doesn't read them), which is all build_registry / ModelConfig require.
"""

from __future__ import annotations

import json
import tomllib

import pytest
from fastapi.testclient import TestClient

from aero import config, engine, server, store, store_ops


@pytest.fixture
def home(tmp_path):
    """A temp aero home with one base GGUF (`base.gguf`) and an engine pointed at it."""
    gguf = store.gguf_dir(tmp_path)
    gguf.mkdir(parents=True)
    (gguf / "base.gguf").write_bytes(b"\x00")  # stand-in weights
    store.config_dir(tmp_path).mkdir(parents=True)
    registry = config.build_registry(gguf, store.config_dir(tmp_path))
    engine.configure(registry, backend="stub", idle_timeout=0, home=tmp_path,
                     registry_defaults={"default_n_ctx": 4096, "default_kv_cache_type": "f16"})
    return tmp_path


@pytest.fixture
def client():
    return TestClient(server.app)


# --------------------------------------------------------------------------- #
# TOML writer
# --------------------------------------------------------------------------- #


def test_toml_writer_round_trips():
    fields = {
        "from": "base", "system": 'Be terse.\nUse "quotes".', "n_ctx": "auto",
        "kv_cache_type": "q8_0", "tools": True, "max_tokens": 2048,
        "sampling": {"temperature": 0.3, "stop": ["</s>"]},
    }
    parsed = tomllib.loads(store_ops.dump_config_toml(fields))
    assert parsed["from"] == "base"
    assert parsed["system"] == 'Be terse.\nUse "quotes".'  # newline + quotes survive
    assert parsed["n_ctx"] == "auto" and parsed["tools"] is True
    assert parsed["sampling"] == {"temperature": 0.3, "stop": ["</s>"]}


def test_toml_writer_omits_unset_fields():
    parsed = tomllib.loads(store_ops.dump_config_toml({"system": "hi"}))
    assert parsed == {"system": "hi"}  # nothing else leaks in


# --------------------------------------------------------------------------- #
# Create / edit / delete
# --------------------------------------------------------------------------- #


def test_create_derived_model(home, client):
    resp = client.post("/api/models", json={
        "name": "pirate", "from": "base", "system": "Arr!", "tools": True})
    assert resp.status_code == 200, resp.text
    # Written to disk and live in the registry without a restart.
    assert (store.config_dir(home) / "pirate.toml").is_file()
    names = {m["name"] for m in client.get("/api/models").json()["models"]}
    assert {"base", "pirate"} <= names
    pirate = next(m for m in client.get("/api/models").json()["models"] if m["name"] == "pirate")
    assert pirate["base"] == "base" and pirate["tools"] is True
    # It also serves immediately (shows up on the chat-side state endpoint).
    assert any(m["name"] == "pirate" for m in client.get("/api/state").json()["models"])


def test_create_requires_name(home, client):
    assert client.post("/api/models", json={"from": "base"}).status_code == 400


def test_create_rejects_missing_weights(home, client):
    resp = client.post("/api/models", json={"name": "ghost", "from": "nonexistent"})
    assert resp.status_code == 400
    assert "weights not found" in resp.json()["detail"]


def test_create_rejects_bad_kv_cache_type(home, client):
    resp = client.post("/api/models", json={"name": "x", "from": "base", "kv_cache_type": "q3_0"})
    assert resp.status_code == 400


def test_edit_model_updates_config(home, client):
    client.post("/api/models", json={"name": "v", "from": "base"})
    resp = client.put("/api/models/v", json={"from": "base", "kv_cache_type": "q4_0", "n_ctx": "auto"})
    assert resp.status_code == 200
    parsed = tomllib.loads((store.config_dir(home) / "v.toml").read_text())
    assert parsed["kv_cache_type"] == "q4_0" and parsed["n_ctx"] == "auto"


def test_delete_derived_model(home, client):
    client.post("/api/models", json={"name": "tmp", "from": "base"})
    resp = client.delete("/api/models/tmp")
    assert resp.status_code == 200
    assert not (store.config_dir(home) / "tmp.toml").exists()
    names = {m["name"] for m in client.get("/api/models").json()["models"]}
    assert "tmp" not in names and "base" in names  # base (the shared weights) untouched


def test_delete_keeps_referenced_weights(home, client):
    client.post("/api/models", json={"name": "child", "from": "base"})
    # base.gguf is an orphan GGUF (no toml) referenced by `child`; deleting base with
    # --weights must refuse to remove the still-referenced GGUF.
    resp = client.delete("/api/models/base?weights=true")
    assert resp.status_code == 200
    assert "still referenced" in (resp.json()["note"] or "")
    assert (store.gguf_dir(home) / "base.gguf").is_file()


def test_management_requires_home(client):
    engine.configure({}, backend="stub", idle_timeout=0)  # no home
    assert client.post("/api/models", json={"name": "x", "from": "base"}).status_code == 503


# --------------------------------------------------------------------------- #
# Pull (SSE) + repo listing — network stubbed
# --------------------------------------------------------------------------- #


def test_repo_listing(home, client, monkeypatch):
    monkeypatch.setattr(store_ops, "list_repo_ggufs",
                        lambda repo: [{"filename": "M-Q4_K_M.gguf", "size": 123}])
    body = client.get("/api/models/repo?repo=some/repo").json()
    assert body["files"][0]["filename"] == "M-Q4_K_M.gguf"


def test_pull_streams_progress_and_registers(home, client, monkeypatch):
    def fake_download(repo, filename, dest_dir, progress_cb=None):
        from pathlib import Path
        dest = Path(dest_dir) / Path(filename).name
        if progress_cb:
            progress_cb(50, 100)
            progress_cb(100, 100)
        dest.write_bytes(b"\x00")
        return dest

    monkeypatch.setattr(store_ops, "download_gguf", fake_download)
    with client.stream("GET", "/api/models/pull?repo=r&filename=NewModel.gguf") as resp:
        events = [json.loads(l[6:]) for l in resp.iter_lines()
                  if l.startswith("data: ") and l != "data: [DONE]"]
    types = [e["type"] for e in events]
    assert "progress" in types and types[-1] == "done"
    assert events[-1]["name"] == "NewModel"
    # Registered live: the pulled model now serves, with a starter config written.
    names = {m["name"] for m in client.get("/api/models").json()["models"]}
    assert "NewModel" in names
    assert (store.config_dir(home) / "NewModel.toml").is_file()
