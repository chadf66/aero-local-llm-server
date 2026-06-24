"""Tests for the local model store (filesystem only, no network)."""

from __future__ import annotations

import pytest

from aero import store


@pytest.fixture
def models_dir(tmp_path):
    """A store with two fake GGUF files and one non-GGUF that must be ignored."""
    (tmp_path / "model-a.Q4_K_M.gguf").write_bytes(b"x" * 2048)
    (tmp_path / "model-b.gguf").write_bytes(b"y" * 1024)
    (tmp_path / "README.md").write_text("not a model")
    return tmp_path


def test_scan_returns_stems_for_ggufs_only(models_dir):
    found = store.scan(models_dir)
    assert set(found) == {"model-a.Q4_K_M", "model-b"}
    assert found["model-b"].endswith("model-b.gguf")


def test_scan_missing_dir_is_empty(tmp_path):
    assert store.scan(tmp_path / "nope") == {}


def test_find(models_dir):
    assert store.find(models_dir, "model-b").name == "model-b.gguf"
    assert store.find(models_dir, "ghost") is None


def test_remove(models_dir):
    removed = store.remove(models_dir, "model-b")
    assert not removed.exists()
    assert "model-b" not in store.scan(models_dir)
    with pytest.raises(KeyError):
        store.remove(models_dir, "model-b")


def test_human_size():
    assert store.human_size(512) == "512 B"
    assert store.human_size(2048) == "2.0 KB"
    assert store.human_size(5 * 1024**3) == "5.0 GB"
