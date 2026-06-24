"""Tests for model definitions: `from` resolution and registry building."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aero import config


@pytest.fixture
def home(tmp_path):
    """An aero home with gguf/ and models/ dirs."""
    (tmp_path / "gguf").mkdir()
    (tmp_path / "models").mkdir()
    return tmp_path


def _gguf(home, stem):
    p = home / "gguf" / f"{stem}.gguf"
    p.write_bytes(b"x" * 1024)
    return p


def _toml(home, name, body):
    (home / "models" / f"{name}.toml").write_text(body)


# --------------------------------------------------------------------------- #
# from-resolution
# --------------------------------------------------------------------------- #


def test_resolve_weights_default_is_same_name(home):
    assert config.resolve_weights(None, "Foo", home / "gguf") == home / "gguf" / "Foo.gguf"


def test_resolve_weights_from_name(home):
    assert config.resolve_weights("Base", "Derived", home / "gguf") == home / "gguf" / "Base.gguf"


def test_resolve_weights_from_path(home):
    p = config.resolve_weights("/abs/w.gguf", "Derived", home / "gguf")
    assert str(p) == "/abs/w.gguf"


# --------------------------------------------------------------------------- #
# registry building
# --------------------------------------------------------------------------- #


def test_orphan_gguf_auto_registers(home):
    _gguf(home, "Solo")  # no definition file
    reg = config.build_registry(home / "gguf", home / "models")
    assert set(reg) == {"Solo"}
    assert reg["Solo"].base is None


def test_base_plus_derived_share_weights(home):
    _gguf(home, "Base")
    _toml(home, "Base", 'system = "base"\nn_ctx = 8192\n')
    _toml(home, "pirate", 'from = "Base"\nsystem = "arr"\nn_ctx = 8192\n')

    reg = config.build_registry(home / "gguf", home / "models")
    assert set(reg) == {"Base", "pirate"}
    # Both resolve to the same weights...
    assert reg["Base"].path == reg["pirate"].path
    # ...so they share a load key (the engine can swap without reloading).
    assert reg["Base"].load_key() == reg["pirate"].load_key()
    assert reg["pirate"].base == "Base"
    assert reg["pirate"].system == "arr"


def test_definition_overrides_orphan(home):
    _gguf(home, "Foo")
    _toml(home, "Foo", "n_ctx = 2048\n")
    reg = config.build_registry(home / "gguf", home / "models")
    assert reg["Foo"].n_ctx == 2048  # the definition wins over the bare-GGUF default


def test_missing_weights_raises(home):
    _toml(home, "ghost", 'from = "DoesNotExist"\n')
    with pytest.raises(FileNotFoundError):
        config.build_registry(home / "gguf", home / "models")


def test_bad_kv_cache_type_raises(home):
    _gguf(home, "Foo")
    _toml(home, "Foo", 'kv_cache_type = "q3_0"\n')
    with pytest.raises(ValidationError):
        config.build_registry(home / "gguf", home / "models")


def test_n_ctx_auto_is_allowed(home):
    _gguf(home, "Foo")
    _toml(home, "Foo", 'n_ctx = "auto"\n')
    reg = config.build_registry(home / "gguf", home / "models")
    assert reg["Foo"].n_ctx == "auto"


def test_n_ctx_bad_string_rejected(home):
    _gguf(home, "Foo")
    _toml(home, "Foo", 'n_ctx = "huge"\n')
    with pytest.raises(ValidationError):
        config.build_registry(home / "gguf", home / "models")
