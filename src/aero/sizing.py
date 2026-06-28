"""Memory-aware context sizing — pick `n_ctx` to fit the machine.

On a memory-constrained box the right context window is a function of *both* the
model's trained context and the memory available. Because of evict-before-load,
only one model is ever resident, so the budget only has to fit one model:

    fit_ctx = (budget − weights − overhead) / kv_bytes_per_token
    n_ctx   = min(model_n_ctx_train, fit_ctx)

`budget` is a fraction of total unified memory (predictable run-to-run). The KV
cache, not the weights, is what grows with context, so it's the term that decides
how much context fits — and `kv_cache_type` (f16/q8_0/q4_0) scales it directly.

The model's dimensions come from the GGUF metadata header, read via a quick
vocab-only llama.cpp load (no weights, no GPU) — no extra dependency.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("aero.sizing")

# KV-cache bytes per element by precision (q4_0 is ~0.5 incl. its block scales).
_KV_BYTES = {"f16": 2.0, "q8_0": 1.0, "q4_0": 0.5}
# Allowance for llama.cpp compute/context buffers beyond weights + KV cache. This is
# the decode-time graph (activations, attention scratch, output logits) that exists
# only while generating, so a model can *load* and then fail to *decode* if it's not
# reserved. 768 MB is a conservative cover for the models aero targets.
_OVERHEAD_BYTES = 768 * 1024 * 1024
# Don't bother loading below this many tokens; error out instead.
_MIN_CTX = 512
# Round the chosen context down to a tidy multiple.
_ROUND_TO = 256

_GiB = 1024 ** 3


@dataclass
class GGUFDims:
    """The few model dimensions that determine KV-cache size."""

    n_layers: int
    n_kv_heads: int
    head_dim: int
    n_ctx_train: int

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim


def total_memory_bytes() -> int:
    """Total physical (unified) memory on this machine."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):  # pragma: no cover - very unusual
        import subprocess
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))


def read_gguf_dims(path: str) -> GGUFDims:
    """Read layer/head/context dimensions from a GGUF's metadata header.

    Uses a vocab-only llama.cpp load: it parses the header but loads no weights and
    touches no GPU, so it's cheap.
    """
    from llama_cpp import Llama

    llm = Llama(model_path=str(path), vocab_only=True, verbose=False)
    try:
        md = llm.metadata
    finally:
        if hasattr(llm, "close"):
            llm.close()

    arch = md["general.architecture"]

    def get_int(key: str) -> int:
        return int(md[f"{arch}.{key}"])

    n_heads = get_int("attention.head_count")
    n_kv_heads = int(md.get(f"{arch}.attention.head_count_kv", n_heads))  # MHA if absent
    n_embd = get_int("embedding_length")
    key_len = md.get(f"{arch}.attention.key_length")
    head_dim = int(key_len) if key_len else n_embd // n_heads

    return GGUFDims(
        n_layers=get_int("block_count"),
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        n_ctx_train=get_int("context_length"),
    )


def kv_bytes_per_token(dims: GGUFDims, kv_cache_type: str) -> float:
    """KV-cache bytes used per token of context: 2 (K+V) × layers × kv_dim × bytes."""
    return 2 * dims.n_layers * dims.kv_dim * _KV_BYTES[kv_cache_type]


def compute_fit(
    dims: GGUFDims,
    weights_bytes: int,
    kv_cache_type: str,
    budget_bytes: int,
    reserve_bytes: int = 0,
) -> int:
    """The largest context that fits the budget, capped at the trained context and
    rounded down. Returns 0 if nothing reasonable fits (caller decides what to do).

    ``reserve_bytes`` is held back for anything else that must be resident at the same
    time — chiefly a co-resident embedder when the model has a knowledge base (RAG)."""
    available = budget_bytes - weights_bytes - _OVERHEAD_BYTES - reserve_bytes
    if available <= 0:
        return 0
    fit = int(available / kv_bytes_per_token(dims, kv_cache_type))
    n_ctx = min(dims.n_ctx_train, fit)
    return n_ctx - (n_ctx % _ROUND_TO)


def auto_n_ctx(path: str, kv_cache_type: str, mem_fraction: float, reserve_bytes: int = 0) -> int:
    """Choose `n_ctx` for a model so it fits `mem_fraction` of total memory.

    ``reserve_bytes`` leaves room for a co-resident embedder (RAG) so the chat model
    doesn't load at a context it can't actually decode alongside the embedder."""
    dims = read_gguf_dims(path)
    weights = os.path.getsize(path)
    budget = int(total_memory_bytes() * mem_fraction)
    n_ctx = compute_fit(dims, weights, kv_cache_type, budget, reserve_bytes)

    if n_ctx < _MIN_CTX:
        raise RuntimeError(
            f"auto n_ctx: {os.path.basename(path)} doesn't fit the memory budget "
            f"(weights {weights / _GiB:.1f} GB, budget {budget / _GiB:.1f} GB of "
            f"{total_memory_bytes() / _GiB:.1f} GB). Try a smaller --kv-cache-type "
            f"(q8_0/q4_0), a smaller model, or a higher --mem-fraction."
        )

    logger.info(
        "auto n_ctx=%d (trained %d, budget %.1f GB, weights %.1f GB, reserve %.1f GB, kv=%s)",
        n_ctx, dims.n_ctx_train, budget / _GiB, weights / _GiB, reserve_bytes / _GiB, kv_cache_type,
    )
    return n_ctx
