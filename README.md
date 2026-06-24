# aero

A lean, Mac-native local LLM server with an OpenAI-compatible API — a personal,
educational alternative to Ollama for Apple Silicon.

The name is air-derived, and it doubles as the design ethos: light enough to fly
on a memory-constrained MacBook Air. Unlike a fleet server, **the Mac itself is the
product** — a single process on localhost, single user, one model resident at a
time, inference on the Metal GPU. No Docker, no router/worker split, no auth.

> **Status:** Phase c (MVP). Pull models from Hugging Face, serve them on demand,
> and chat from the terminal. See [PHASES.md](PHASES.md) for the roadmap — Modelfile
> per-model config is the next phase.

## Install

Apple Silicon, with the Metal-accelerated inference backend:

```sh
make install-metal      # compiles llama-cpp-python with -DGGML_METAL=on into ./.venv
```

CPU-only (enough to run the stub-backed tests, no model needed):

```sh
make install
make test
```

## Quickstart

Pull a model from Hugging Face and chat with it — `run` starts a server for you:

```sh
# List the GGUF quants in a repo, then pull one into the store (~/.aero/models):
aero pull bartowski/Qwen2.5-3B-Instruct-GGUF
aero pull bartowski/Qwen2.5-3B-Instruct-GGUF Qwen2.5-3B-Instruct-Q4_K_M.gguf

aero list                               # what's in the store
aero run Qwen2.5-3B-Instruct-Q4_K_M     # interactive chat (auto-starts a server)
```

In the chat, `/bye` quits and `/reset` clears the conversation. The model store is
just a folder of `.gguf` files; each is served under its filename stem.

## Model store

```sh
aero pull <repo> [filename]   # download a GGUF (omit filename to list the repo's quants)
aero list                     # list local models + sizes
aero show <name>              # path, size, quantization
aero rm <name>                # delete a local model (--yes to skip the prompt)
```

## Serve models

Drop your GGUF files in a directory and point `serve` at it. Each file is served
as a model named after its filename stem; they're loaded on demand, one at a time.

```sh
aero serve --models-dir ~/.aero/models          # the default dir
# or serve specific files (repeatable), and/or override the port:
.venv/bin/aero serve --model /path/to/a.gguf --model /path/to/b.gguf --port 8317
# via make:
make serve MODEL=/path/to/model.gguf            # add PORT=8088 to override
```

It listens on port `8317` by default (chosen to avoid Ollama's `11434` and common
dev ports). Use any OpenAI client against `http://127.0.0.1:8317/v1`:

```sh
curl http://127.0.0.1:8317/v1/models            # lists every available model
curl http://127.0.0.1:8317/v1/chat/completions \
  -d '{"model": "<name>", "messages": [{"role": "user", "content": "hello"}]}'
```

Streaming (`"stream": true`) and non-streaming are both supported, and responses
carry real `usage` token counts and a real `finish_reason`.

### Memory & context tuning

On a memory-constrained box, the KV cache — not the weights — is what blows the
budget as context grows (`KV ≈ 2 × layers × kv_dim × n_ctx × 2 bytes` at f16).

- `--n-ctx 8192` — context window size.
- `--kv-cache-type q8_0` — quantize the KV cache (`f16` | `q8_0` | `q4_0`) to fit
  more context in the same memory.
- `--idle-timeout 300` — free the resident model after N idle seconds (`0` = never).

Only one model is ever resident: requesting a different one frees the current
model **before** loading the next (evict-before-load), so you never hit a
two-models-in-memory peak.

## API

- `POST /v1/chat/completions` — streaming SSE + non-streaming chat completions.
- `GET /v1/models` — lists every available model.
- `GET /healthz` — liveness, available models, and the one currently resident.

Supported sampling params: `temperature`, `top_p`, `top_k`, `seed`, `stop`,
`max_tokens`.
