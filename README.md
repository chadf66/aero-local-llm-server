# aero

A lean, Mac-native local LLM server with an OpenAI-compatible API — a personal,
educational alternative to Ollama for Apple Silicon.

The name is air-derived, and it doubles as the design ethos: light enough to fly
on a memory-constrained MacBook Air. Unlike a fleet server, **the Mac itself is the
product** — a single process on localhost, single user, one model resident at a
time, inference on the Metal GPU. No Docker, no router/worker split, no auth.

> **Status:** feature-complete. MVP plus per-model config, config/weights decoupling
> (one GGUF, many models), memory-aware auto context sizing, **tool calling** for agent
> harnesses, and a **web UI** with searchable history *and* full **model management**
> (pull, create/edit configs, delete — applied live, no restart).

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
# List the GGUF quants in a repo, then pull one into the store (~/.aero):
aero pull bartowski/Qwen2.5-3B-Instruct-GGUF
aero pull bartowski/Qwen2.5-3B-Instruct-GGUF Qwen2.5-3B-Instruct-Q4_K_M.gguf

aero list                               # what's in the store
aero run Qwen2.5-3B-Instruct-Q4_K_M     # interactive chat (auto-starts a server)
```

In the chat, `/bye` quits and `/reset` clears the conversation.

## Web UI

A browser chat UI is served by the same `aero serve` process at `http://127.0.0.1:8317/`.
It streams responses, collapses `<think>` reasoning blocks, renders tool-call cards,
keeps **searchable conversation history** (SQLite at `~/.aero/aero.db`), and surfaces
the **memory knobs** (`kv_cache_type` and the resulting max context) with an inline
explainer — the levers that make bigger models fit on a constrained Mac.

**Manage models** (the sidebar button) is the same store, in the browser: pull a GGUF
from Hugging Face with a live progress bar, create/edit a model's config from a form
(system prompt, `n_ctx`/`auto`, `kv_cache_type`, sampling, tools, derived `from`), and
delete models. Changes apply **live** — the server reloads its model set with no
restart, keeping the resident model loaded if it's unaffected. It's the same
`store_ops` code path as the `aero pull`/`rm` CLI, so the two never drift.

The UI is a Svelte SPA built to static assets and served by FastAPI. **Node is a
build-time-only tool** — once built, the runtime is still just `aero serve`, no Node.

```sh
make ui          # build the UI into the package (needs Node; one-time per change)
aero serve       # serves the API and the UI; open the printed http://… URL
```

For UI development, `make ui-dev` runs Vite's dev server (hot reload) and proxies
`/api` and `/v1` to a running `aero serve`. If you start `aero serve` before running
`make ui`, the root page shows a short "run `make ui`" hint; the API is live either way.

## Model store

Everything lives under `~/.aero/`, split into **weights** and **definitions**:

```
~/.aero/
  gguf/      raw weights (what `aero pull` downloads)
  models/    model definitions (*.toml) — what you actually run
```

```sh
aero pull <repo> [filename]   # download a GGUF (omit filename to list the repo's quants)
aero list                     # list models + sizes (derived models show their base)
aero show <name>              # weights, size, and effective config
aero rm <name>                # delete a model (--weights also drops the GGUF; --yes skips the prompt)
```

A *model* is a definition that points at weights. A bare GGUF with no definition
still works (it auto-registers with defaults), so `pull` + `run` just works.

## Serve models

`serve` reads the aero home (`~/.aero` by default), serving every model in it,
loaded on demand, one resident at a time.

```sh
aero serve                                      # serves ~/.aero
# point at a different home, serve ad-hoc files, and/or override the port:
.venv/bin/aero serve --home /path/to/home --model /extra/a.gguf --port 8317
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

- `--n-ctx 8192` — context window size, or `--n-ctx auto` to size it to memory.
- `--kv-cache-type q8_0` — quantize the KV cache (`f16` | `q8_0` | `q4_0`) to fit
  more context in the same memory.
- `--mem-fraction 0.7` — with `auto`, the fraction of total memory to budget.
- `--idle-timeout 300` — free the resident model after N idle seconds (`0` = never).

**Auto context sizing.** Set `n_ctx = "auto"` (in a model's config, or `--n-ctx auto`)
and aero picks the largest context that fits your machine — `min(trained context,
what fits in budget)` — reading the model's dimensions from its GGUF header. Because
only one model is resident at a time, the budget only has to fit that one model. The
KV-cache precision is part of the same trade: on 16 GB, Llama-3.1-8B sizes to ~51K
tokens at `f16`, ~102K at `q8_0`, and its full 131K at `q4_0`.

> **Two different "quantizations" — don't confuse them:**
> - **Weight quant** (the `Q4_K_M` in a GGUF's filename) compresses the model's
>   *weights*. It's baked into the file, chosen when you `pull`, and each level is a
>   **separate GGUF**.
> - **KV-cache quant** (`kv_cache_type`) compresses the attention *cache* at runtime
>   to fit more context. It's a config knob on the **same GGUF** — no new download.
>
> So one `…Q4_K_M.gguf` serves any of the `f16` / `q8_0` / `q4_0` context sizes above
> just by changing `kv_cache_type`. The two are independent.

Only one model is ever resident: requesting a different one frees the current
model **before** loading the next (evict-before-load), so you never hit a
two-models-in-memory peak.

### Per-model config & derived models

A model definition is a TOML file in `~/.aero/models/`. The **filename is the model
name** — `~/.aero/models/<name>.toml` defines the model you call as `<name>`. That's
the whole rule: to configure a model, create (or edit) the `.toml` with its name.

#### How to create one

There's no special command and no DSL — it's just a file. Three ways to get one:

1. **From `pull`** — `aero pull` already drops a starter `models/<name>.toml` next to
   the weights, with every field present but commented out. Open it and uncomment what
   you want.
2. **For a model you already have** — write the file yourself. The name must match the
   model (for a plain GGUF, that's its filename stem):
   ```sh
   $EDITOR ~/.aero/models/Qwen2.5-3B-Instruct-Q4_K_M.toml
   ```
3. **As a new persona over existing weights** — give it any new name and point `from`
   at the base (see *Derived models* below).

After editing, check what the server will actually use:

```sh
aero show Qwen2.5-3B-Instruct-Q4_K_M    # prints the effective config
```

#### Every field

All fields are optional; omit one and it falls back to the `serve` flag, then the
built-in default. Nothing is required — an empty file is a valid config.

```toml
# ~/.aero/models/Qwen2.5-3B-Instruct-Q4_K_M.toml
system = "You are a helpful assistant. Be concise."  # default system prompt
n_ctx = 8192               # context window; an int, or "auto" to size to memory [default 4096]
kv_cache_type = "q8_0"     # KV-cache precision: f16|q8_0|q4_0 [default f16]
max_tokens = 2048          # default completion cap (reasoning models want headroom)
chat_format = "chatml"     # override the GGUF's chat template (rarely needed)
from = "..."               # use another model's weights (see below)

[sampling]                 # defaults applied when the request doesn't set them
temperature = 0.7
top_p = 0.9
top_k = 40
stop = ["</s>"]
```

Precedence is **request field > config default > built-in**: a request's own
`system` message or sampling values always win; the config fills in the rest.

> **Tip:** the default `n_ctx` is a conservative **4096** — short enough that long
> chats or documents get truncated. Set `n_ctx = "auto"` (or a larger number) to use
> the context your model and memory actually allow. See
> [Auto context sizing](#memory--context-tuning).

#### Derived models

Point `from` at another model's weights so one GGUF can back several personas without
copying it. The new file's name is the new model's name:

```toml
# ~/.aero/models/qwen-pirate.toml
from = "Qwen2.5-3B-Instruct-Q4_K_M"   # a GGUF name in gguf/, or a path to a .gguf
system = "You are a pirate. Answer in pirate speak."
```

```sh
aero run qwen-pirate     # same weights as Qwen, different personality
```

If two models resolve to the same weights and context settings, switching between
them is a **persona swap with no reload** — the engine keeps the model resident and
just changes the prompt.

## Tool calling (agents)

aero speaks the OpenAI tool-calling API, so agent frameworks pointed at it work
unmodified. Enable tools on a model with `tools = true`:

```sh
aero pull NousResearch/Hermes-2-Pro-Mistral-7B-GGUF Hermes-2-Pro-Mistral-7B.Q4_K_M.gguf
printf 'from = "Hermes-2-Pro-Mistral-7B.Q4_K_M"\ntools = true\n' > ~/.aero/models/hermes-tools.toml
aero serve
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8317/v1", api_key="not-needed")
client.chat.completions.create(model="hermes-tools", tools=[...], messages=[...])
# -> finish_reason="tool_calls", message.tool_calls=[...]
```

Runnable examples in [`examples/`](examples/): a raw OpenAI-SDK tool loop and an
OpenAI Agents SDK agent (`pip install -e ".[examples]"`).

How it works: rather than rely on each GGUF's chat template (many don't render tools
at all), aero injects the tool definitions into a system prompt itself and parses the
model's tool-call output back into OpenAI `tool_calls` — handling the common
`<tool_call>` (Qwen/Hermes), bare-object (Llama-3.1), and bare-array (Ministral)
formats. Non-streaming and streaming are both supported.

**Model matters for reliability.** Deciding *when* to call a tool is a model skill.
A model fine-tuned for it (e.g. **Hermes-2-Pro**) calls reliably in `auto` mode; a
general small model is hit-or-miss. `tool_choice` (a specific function or
`"required"`) forces a call regardless.

## API

- `POST /v1/chat/completions` — streaming SSE + non-streaming chat completions,
  including `tools` / `tool_choice` and `tool_calls` for tool-enabled models.
- `GET /v1/models` — lists every available model.
- `GET /healthz` — liveness, available models, and the one currently resident.

Supported sampling params: `temperature`, `top_p`, `top_k`, `seed`, `stop`,
`max_tokens`.
