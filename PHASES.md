# aero — phased roadmap

A personal, educational, Mac-native (Apple Silicon) local LLM server. The
**design point is deliberately the inverse of a fleet server**: the Mac itself is
the product — single process, localhost, single user, inference on the Metal GPU,
no Docker, no router/worker split, no auth, memory-first (~16 GB unified memory is
the tight case). Choices that are right for a multi-client online server are often
wrong here, and where they differ this doc says so.

Phase a is **implemented**. Phases b–d are recorded here so the work isn't lost.

---

## Phase a — MVP (done)

Serve a single GGUF on the Metal GPU and answer OpenAI-compatible chat requests.

- `aero serve --model <gguf>` — load one model (all layers on Metal,
  `n_gpu_layers=-1`) and start the HTTP server on localhost.
- `POST /v1/chat/completions` — streaming SSE **and** non-streaming, with **real
  `usage` token counts** and a **real `finish_reason`** ("stop" vs "length").
- `GET /v1/models` — lists the one served model.
- `GET /healthz` — liveness + current model.
- Sampling params plumbed through: temperature, top_p, top_k, seed, stop, max_tokens.
- Stub engine backend so the test suite runs with no model download.

Layout: `src/aero/{schemas,engine,server,cli}.py`, `tests/test_api.py`.

## Phase b — model lifecycle & memory (done)

The heart of a memory-constrained box: hold several models on disk but at most one
in RAM at a time. `aero serve` now takes a `--models-dir` (scanned for `*.gguf`,
name = file stem) plus any extra `--model` files; nothing is resident until a
request names a model.

- **Load-on-demand:** serve a *set* of known models (`/v1/models` lists all);
  `engine._acquire_handle()` loads a model the first time it's requested.
- **Evict-before-load:** switching models frees the old one *before* loading the
  new — a cold gap, never a two-models-resident peak. The explicit inversion of
  the multi-client rule (a fleet keeps the old model serving during the swap).
- **Idle-unload timer:** a daemon thread (`--idle-timeout`, default 300s, `0`
  disables) frees the resident model after inactivity, handing unified memory
  back to the system. `_unload_if_idle()` is the testable seam.
- **KV-cache budgeting:** `--n-ctx` exposed. KV cost is
  `KV ≈ 2 × layers × kv_dim × n_ctx × 2 bytes` (f16) — KV cache, not weights, is
  the knob that blows the budget as context grows.
- **KV-cache quantization:** `--kv-cache-type f16|q8_0|q4_0` (sets llama.cpp's
  `type_k`/`type_v`; quantized types enable flash attention) to stretch context on
  fixed memory.

All loads/unloads/inference and the idle sweep serialize on one lock — correct and
simple for a single-user box. Layout adds nothing new; `engine.py` grew the
registry + lifecycle, `server.py`/`cli.py` updated, tests cover load-on-demand,
evict-before-load, and the idle seam.

## Phase c — CLI & model store (done)

Usable day to day, Ollama-style, without hand-placing GGUF files. The model store
is just a folder of `.gguf` files (`~/.aero/models` by default) — no manifest, no
database; a model *is* a file, served under its filename stem. Store logic lives in
`store.py` (`scan`/`find`/`remove`/`human_size`), shared by `serve` and the new
commands.

- `pull <repo> [filename]` — download a GGUF from Hugging Face (`huggingface-hub`)
  into the store. Omit the filename to list the repo's available GGUF quants.
- `list` — show locally available models with their on-disk sizes.
- `rm <name>` — delete a local model (confirm, or `--yes`).
- `show <name>` — print a model's path, size, and quant (parsed from the filename).
- `run <name>` — interactive terminal chat. **Auto-starts a server** if one isn't
  already healthy on the target host/port, reuses it if it is, and tears down any
  server it started on exit (`/bye`). Streams tokens over the OpenAI API via stdlib
  `urllib` (no extra runtime dep); keeps multi-turn history, `/reset` clears it.

## Phase d — Modelfile config & polish

- **Modelfile-like per-model config:** default system prompt, sampling defaults,
  context size, chat template — stored alongside each model and applied on load.
- **Reasoning-model handling:** models that emit `<think>…</think>` in the content
  need generous `max_tokens`; surface/handle this cleanly.
- README/docs polish, examples, and a short architecture writeup.

---

## Permanently out of scope (by design, not deferral)

Auth, a router/worker split, replicas/health-check fleet, and Docker. Docker on
Mac can't reach the Metal GPU, and the rest solves multi-client problems this
single-user box doesn't have.
