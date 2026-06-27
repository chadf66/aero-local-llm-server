"""KB admin API tests (Phase g4) — stub engine + real LanceDB, temp home."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from aero import engine, server


@pytest.fixture
def home(tmp_path):
    engine.configure({}, backend="stub", idle_timeout=0, home=tmp_path)
    return tmp_path


@pytest.fixture
def client():
    return TestClient(server.app)


def _frames(resp) -> list[dict]:
    return [json.loads(l[6:]) for l in resp.text.splitlines()
            if l.startswith("data: ") and l.strip() != "data: [DONE]"]


def test_kb_crud_and_ingest_flow(home, client):
    # create
    created = client.post("/api/kb", json={"name": "docs", "embedder": "bge-small"})
    assert created.status_code == 200 and created.json()["dim"] == engine._STUB_EMBED_DIM
    assert any(k["name"] == "docs" for k in client.get("/api/kb").json()["kbs"])

    # ingest (upload two files) -> SSE progress + done
    resp = client.post(
        "/api/kb/docs/ingest",
        files=[
            ("files", ("a.txt", b"alpha content about pelicans", "text/plain")),
            ("files", ("b.txt", b"beta content about quantization", "text/plain")),
        ],
    )
    frames = _frames(resp)
    assert any(f["type"] == "progress" for f in frames)
    done = [f for f in frames if f["type"] == "done"][0]
    assert done["files_ingested"] == 2

    # detail reflects files
    detail = client.get("/api/kb/docs").json()
    assert {f["source"] for f in detail["files"]} == {"a.txt", "b.txt"}
    assert detail["chunks"] >= 2

    # remove one file
    after = client.delete("/api/kb/docs/files/b.txt").json()
    assert {f["source"] for f in after["files"]} == {"a.txt"}


def test_sync_endpoint_streams(home, client):
    client.post("/api/kb", json={"name": "k", "embedder": "e"})
    client.post("/api/kb/k/ingest",
                files=[("files", ("x.txt", b"hello world", "text/plain"))])
    frames = _frames(client.post("/api/kb/k/sync"))
    done = [f for f in frames if f["type"] == "done"][0]
    assert "pruned" in done and done["skipped"] >= 1  # unchanged file skipped on re-index


def test_create_duplicate_is_400(home, client):
    client.post("/api/kb", json={"name": "dup", "embedder": "e"})
    assert client.post("/api/kb", json={"name": "dup", "embedder": "e"}).status_code == 400


def test_get_and_delete_missing(home, client):
    assert client.get("/api/kb/nope").status_code == 404
    assert client.delete("/api/kb/nope").status_code == 404


def test_delete_kb(home, client):
    client.post("/api/kb", json={"name": "tmp", "embedder": "e"})
    assert client.delete("/api/kb/tmp").json() == {"deleted": "tmp"}
    assert not any(k["name"] == "tmp" for k in client.get("/api/kb").json()["kbs"])


def test_kb_requires_home(client):
    engine.configure({}, backend="stub", idle_timeout=0)  # no home
    assert client.get("/api/kb").status_code == 503
    assert client.post("/api/kb", json={"name": "x", "embedder": "e"}).status_code == 503
