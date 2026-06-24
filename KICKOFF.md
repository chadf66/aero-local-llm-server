# Project kickoff prompt

Paste the block below into a fresh Claude Code session opened in this repo to
start the project. It's written to make Claude ask before building and propose a
plan first.

---

I want to build a new, from-scratch project in this repo: a local LLM serving
application for macOS (Apple Silicon), like Ollama — a personal, educational
replacement to reduce my dependence on third-party tools. I learn by building,
so favor clarity and good docs over cleverness.

## Context (why this is a *different* design than a typical server)
I just built an online LLM *server* (router + worker fleet, Docker/GPU lanes,
OpenAI-compatible API) and learned a lot. This is a deliberately different
design point: the Mac itself is the product — single user, single machine,
constrained unified memory, native Metal, no Docker, no fleet. Choices that are
right for an online server are often wrong here. Concretely, this project should be:
- A single process, localhost, single user. No router/worker split, no replicas,
  no health-check fleet, no network auth by default.
- Inference on the Metal GPU. No Docker (Docker on Mac can't reach the GPU).
- Memory-first: assume ~16 GB unified memory as the tight case.

## Target feature set (Ollama-like)
- A `serve` command exposing a local HTTP API.
- OpenAI-compatible endpoints: `/v1/chat/completions` (streaming SSE +
  non-streaming) and `/v1/models`.
- Model management: `pull` (GGUF from Hugging Face), `list`, `rm`, `show`, and a
  `run` interactive terminal chat.
- Load-on-demand with an idle-unload timer; when swapping models, EVICT-BEFORE-LOAD
  (free the old model before loading the new — on a single memory-constrained box,
  avoiding the two-models-resident peak matters more than keeping the old one
  serving, which is the opposite of the right call for a multi-client server).
- A Modelfile-like per-model config: default system prompt, sampling params,
  context size, chat template.

## Lessons to carry over (don't rediscover these)
- Populate `usage` with real token counts from the inference result, and a real
  `finish_reason` ("stop" vs "length").
- Sampling params: temperature, top_p, top_k, seed, stop, max_tokens.
- KV-cache budgeting is the main memory lever: KV ≈ 2 × layers × kv_dim × n_ctx ×
  2 bytes (f16). Expose n_ctx, and consider KV-cache quantization (q8_0/q4) to
  stretch context on fixed unified memory.
- Reasoning models emit <think>…</think> in the content and need generous max_tokens.

## Decisions I want you to RAISE with me before coding (don't assume)
1. Inference engine: llama.cpp via llama-cpp-python (familiar, huge GGUF
   ecosystem) vs MLX / mlx-lm (Apple-native, most Mac-optimized) vs llama.cpp or
   Rust directly. Trade-offs for a Mac-native, low-dependency, educational build?
2. Stack: Python (FastAPI + Typer CLI) vs Go/Rust for a single distributable binary.
3. API surface: OpenAI-compatible only, or also mirror Ollama's native API for
   drop-in tool compatibility?
4. The v1/MVP scope.

## How to start
1. Ask me decisions 1–4 first.
2. Then propose a short architecture + a phased roadmap. Suggested phases: (a) MVP
   — single GGUF, Metal, OpenAI-compatible chat with streaming + usage; (b) model
   management with load-on-demand + evict-before-load + idle unload; (c) CLI
   (serve/pull/run/list/rm); (d) Modelfile config + polish.
3. Scaffold the repo once we agree on the stack.
