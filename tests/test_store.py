"""Tests for the store layout helpers."""

from __future__ import annotations

from pathlib import Path

from aero import store


def test_dir_helpers():
    home = Path("/tmp/aero-home")
    assert store.gguf_dir(home) == home / "gguf"
    assert store.config_dir(home) == home / "models"


def test_human_size():
    assert store.human_size(512) == "512 B"
    assert store.human_size(2048) == "2.0 KB"
    assert store.human_size(5 * 1024**3) == "5.0 GB"
