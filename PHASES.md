# aero — phased roadmap

A personal, educational, Mac-native (Apple Silicon) local LLM server. The
**design point is deliberately the inverse of a fleet server**: the Mac itself is
the product — single process, localhost, single user, inference on the Metal GPU,
no Docker, no router/worker split, no auth, memory-first (~16 GB unified memory is
the tight case). Choices that are right for a multi-client online server are often
wrong here, and where they differ this doc says so.

Phases a–c are **implemented** (the MVP). Phases d–f are the roadmap, recorded here
so the work isn't lost.

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

## Beyond the MVP: d → e → f

Phases d, e, and f turn the MVP into a tool-capable, UI-having local stack. They
build in order on one **guiding principle: the OpenAI-compatible API stays the single
source of truth.** A web UI and an agent harness are both just clients of
`/v1/chat/completions`, so the UI dogfoods the exact path agents use. The **inference
path stays stateless** (every request is self-contained — what keeps agents and the
CLI clean); the only state we add is a UI-only conversation store (Phase f), which
never touches inference.

Both the UI and tool calling depend on the Modelfile (per-model chat template /
format), so **d comes first**, then **e** (tools, with an agent harness as the
acceptance test), then **f** (the UI, which visualizes tools + reasoning).

## Phase d — Modelfile config & per-model loading

The shared foundation. Per-model config replaces today's process-global knobs.
**Done.**

- **Modelfile-like per-model config — done.** A TOML model definition
  (`models/<name>.toml`, `config.py`) carries: default `system` prompt, `sampling`
  defaults, `n_ctx`, `kv_cache_type`, `max_tokens`, and a `chat_format` override.
  `n_ctx`/`kv_cache_type` are now per-model (the `serve` CLI flags are the fallbacks);
  the engine loads each model with its own config and `show` displays it. Request-time
  precedence is request field > config default > built-in: the default system prompt
  is injected only when the request has no system message, and explicit sampling
  fields (tracked via Pydantic `model_fields_set`) always win.
- **Config/weights decoupling — done.** The store splits into `~/.aero/gguf/`
  (weights) and `~/.aero/models/` (definitions). A definition points at weights via
  `from = "<gguf name>"` or a path, defaulting to the same-named GGUF; a bare GGUF
  with no definition auto-registers. So **one GGUF can back many named models**
  (e.g. different system prompts) without copying weights. The engine's
  `_acquire_handle` compares a `load_key` of `(path, n_ctx, kv_cache_type,
  chat_format)`: switching between two models that share it is a **persona swap with
  no reload** — only the per-request prompt/sampling differ. `pull` writes the GGUF
  plus a starter definition; `rm` drops the definition (and weights for a base model,
  if nothing else references them).
- **Memory-aware auto context sizing — done.** Set `n_ctx = "auto"` (opt-in) and the
  engine sizes the context to fit memory at load (`sizing.py`):

  ```
  fit_ctx = (budget − weights − overhead) / (2 × n_layers × kv_dim × bytes_per_token)
  n_ctx   = min(model_n_ctx_train, fit_ctx)         # rounded down to a multiple of 256
  ```

  - **budget** = total unified memory (`os.sysconf`) × `--mem-fraction` (default 0.70).
    Chosen over *free* memory for run-to-run reproducibility.
  - **weights** = the GGUF file size; plus a fixed compute/context **overhead**.
  - **KV-per-token** = `2 (K+V) × n_layers × kv_dim × bytes`, `bytes` = 2/1/0.5 for
    f16/q8_0/q4_0 — so `kv_cache_type` is the same lever: quantizing the KV cache frees
    memory for more context (a *joint* (`n_ctx`, `kv_cache_type`) choice the user makes).
  - `n_layers`, `kv_dim`, `n_ctx_train` come from the **GGUF metadata header**, read via
    a quick vocab-only llama.cpp load (no weights, no GPU, no extra dependency).
  - Resolved decisions: budget off *total* memory; size at the chosen KV precision (no
    auto-quant — the user picks `kv_cache_type`); if even a minimal context won't fit,
    refuse to load with a clear message (suggesting a smaller quant/model or higher
    `--mem-fraction`). Verified on 16 GB: Llama-3.1-8B → 50,944 ctx (f16) / 102,144
    (q8_0) / full 131,072 (q4_0).
- **Reasoning-model handling (engine side):** models that emit `<think>…</think>` in
  the content need generous `max_tokens`; carry that as a per-model default and
  surface it cleanly. (Rendering of `<think>` is Phase f.)

## Phase e — Tool calling & agent support

Make the models usable in agent harnesses. The acceptance test: point an existing
agent framework at `http://127.0.0.1:8317/v1` and have it *just work*. Of the
installed models, **Llama-3.1-8B, Qwen2.5-3B, and Ministral-8B** are the tool-capable
ones; TinyLlama/SmolLM/Gemma won't, and DeepSeek-R1 is reasoning-only.

- **Schema extensions (OpenAI-faithful):** add `tools` / `tool_choice` to the request;
  `tool_calls` on assistant messages; accept `role:"tool"` messages with
  `tool_call_id`; allow non-string / `null` message content.
- **Engine:** pass `tools` into llama.cpp's `create_chat_completion` using the
  per-model chat template / tool format from the Modelfile (Llama-3.1, Qwen, Mistral/
  Ministral, Hermes, functionary all differ — this is *why* it's gated on Phase d);
  parse `tool_calls` out of the output and emit `finish_reason:"tool_calls"`.
- **Sequencing within the phase:** non-streaming tool calls first (clean), then the
  genuinely hard part — streaming partial `tool_calls` deltas.
- **Acceptance test:** a real tool loop via the OpenAI Python SDK against localhost,
  then one framework (e.g. Pydantic AI or the OpenAI Agents SDK). Framework choice TBD
  when we start.

## Phase f — Web chat UI

A web UI that beats the typical local experience (Open WebUI is the bar) by showing
off what the engine and models actually do — served by the same FastAPI app.

- **Stack:** a modern SPA (Svelte/React via Vite; framework TBD), built to static
  assets and mounted by FastAPI at `/`. **Node is a build-time-only dependency** — the
  runtime stays one `aero serve`, no Node. (Decide later: commit built assets vs build
  on install.)
- **Persistence:** server-side SQLite at `~/.aero/aero.db` for **searchable, durable
  history** (conversations + messages + tool calls) — a concrete "better than Ollama"
  win. New conversation-CRUD endpoints; this is the *only* server state, and it's kept
  off the inference path.
- **Showcase features:** model picker + **live resident-model state** (load / evict /
  idle-unload, which the engine already tracks); streaming with stop/regenerate;
  collapsible **`<think>`** blocks for reasoning models; inline **tool-call cards**
  (the model requested `f(args)`, here's the result); per-conversation system prompt +
  sampling controls wired to Modelfile defaults; markdown + code highlighting.
- **Surface the memory knobs, not just sampling.** Alongside temperature / top_p /
  top_k, expose **`n_ctx` (incl. `auto`)** and **`kv_cache_type` (f16/q8_0/q4_0)** as
  first-class controls, with a short inline explainer. These are what make bigger
  open models *feasible* on a constrained Mac, yet they're obscure even to
  experienced practitioners — surfacing + explaining them in the UI is a genuine
  differentiator and a teaching moment. Show the trade live: changing `kv_cache_type`
  should display the resulting max context (we already compute it in `sizing.py`), so
  the f16→q8_0→q4_0 context gain is visible. Stretch: separate **K vs V** cache
  precision (K is more sensitive; llama.cpp supports distinct `type_k`/`type_v`, which
  the engine currently sets equal) — the highest-leverage quality knob when leaning on
  `q4_0`.
- **Conversation compaction (summarize & truncate):** long chats overflow `n_ctx`
  (today `aero run` re-sends the full transcript with no windowing — see `_stream_chat`
  in `cli.py`). When the prompt nears the context budget, summarize the oldest turns
  into a compact recap and keep the recent tail verbatim; the pinned system prompt must
  always survive. Applies to both the UI and `run`. Decisions: trigger (~75% of
  `n_ctx`), how much tail to keep verbatim, same-model vs cheaper summarizer, and
  excluding `<think>` blocks from what's summarized.
- README/docs polish, examples, and a short architecture writeup.

---

## Permanently out of scope (by design, not deferral)

Auth, a router/worker split, replicas/health-check fleet, and Docker. Docker on
Mac can't reach the Metal GPU, and the rest solves multi-client problems this
single-user box doesn't have.
