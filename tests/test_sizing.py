"""Tests for memory-aware context sizing (pure math; no model load)."""

from __future__ import annotations

import pytest

from aero import sizing
from aero.sizing import GGUFDims, compute_fit, kv_bytes_per_token


def test_kv_bytes_per_token_scales_with_precision():
    d = GGUFDims(n_layers=32, n_kv_heads=8, head_dim=128, n_ctx_train=131072)
    # kv_dim = 8*128 = 1024; f16: 2*32*1024*2 = 131072 bytes/token
    assert kv_bytes_per_token(d, "f16") == 131072
    assert kv_bytes_per_token(d, "q8_0") == 65536
    assert kv_bytes_per_token(d, "q4_0") == 32768


def test_compute_fit_caps_at_trained_context():
    d = GGUFDims(n_layers=8, n_kv_heads=8, head_dim=128, n_ctx_train=4096)
    # Effectively unlimited budget -> capped at the trained context.
    assert compute_fit(d, weights_bytes=0, kv_cache_type="f16", budget_bytes=10**12) == 4096


def test_compute_fit_limited_by_budget_and_rounded():
    d = GGUFDims(n_layers=32, n_kv_heads=8, head_dim=128, n_ctx_train=131072)  # 128 KB/token f16
    budget = sizing._OVERHEAD_BYTES + 1 * 1024**3  # ~1 GB left for KV after overhead
    n = compute_fit(d, weights_bytes=0, kv_cache_type="f16", budget_bytes=budget)
    assert n % sizing._ROUND_TO == 0
    assert 7000 <= n <= 8192  # ~1 GB / 128 KB ≈ 8192 tokens


def test_compute_fit_quantized_kv_fits_more():
    d = GGUFDims(n_layers=32, n_kv_heads=8, head_dim=128, n_ctx_train=131072)
    budget = sizing._OVERHEAD_BYTES + 1 * 1024**3
    f16 = compute_fit(d, 0, "f16", budget)
    q4 = compute_fit(d, 0, "q4_0", budget)
    assert q4 >= 3 * f16  # quartering the bytes/token roughly quadruples the fit


def test_compute_fit_reserve_shrinks_context():
    # Reserving memory for a co-resident embedder leaves less room for the KV cache,
    # so the fitted context must shrink (the RAG-model fix).
    d = GGUFDims(n_layers=32, n_kv_heads=8, head_dim=128, n_ctx_train=131072)
    budget = sizing._OVERHEAD_BYTES + 4 * 1024**3
    base = compute_fit(d, 0, "f16", budget)
    with_embedder = compute_fit(d, 0, "f16", budget, reserve_bytes=2 * 1024**3)
    assert with_embedder < base


def test_compute_fit_zero_when_weights_exceed_budget():
    d = GGUFDims(n_layers=32, n_kv_heads=8, head_dim=128, n_ctx_train=131072)
    assert compute_fit(d, weights_bytes=10**12, kv_cache_type="f16", budget_bytes=1) == 0


def test_auto_n_ctx_raises_when_model_too_big(monkeypatch):
    monkeypatch.setattr(sizing, "read_gguf_dims", lambda p: GGUFDims(32, 8, 128, 131072))
    monkeypatch.setattr(sizing, "total_memory_bytes", lambda: 2 * 1024**3)  # 2 GB total
    monkeypatch.setattr("os.path.getsize", lambda p: 4 * 1024**3)            # 4 GB weights
    with pytest.raises(RuntimeError):
        sizing.auto_n_ctx("/fake.gguf", "f16", 0.70)
