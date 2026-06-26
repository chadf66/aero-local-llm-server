# aero — a lean, Mac-native local LLM server.
#
# Two install lanes: a CPU-only one (enough for the stub-backed tests) and a
# Metal one that compiles llama-cpp-python against Apple's GPU. Override MODEL on
# the `serve` target: `make serve MODEL=/path/to/model.gguf`.

VENV   := .venv
BIN    := $(VENV)/bin
PYTHON ?= python3

# Default model for `make serve`; override on the command line.
MODEL  ?=
# Port for `make serve`. Default avoids Ollama (11434) and common dev ports.
PORT   ?= 8317

# Where the web UI source lives; `make ui` builds it into the package (src/aero/webui_dist).
WEBUI  := webui

.PHONY: help install install-metal serve test ui ui-dev

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

$(VENV):
	$(PYTHON) -m venv $(VENV)

install: $(VENV)     ## Install deps into ./.venv, CPU only (enough for tests/stub)
	$(BIN)/pip install -e ".[dev,rag]"

install-metal: $(VENV)  ## Install deps into ./.venv with the Metal backend (Apple Silicon)
	CMAKE_ARGS="-DGGML_METAL=on" $(BIN)/pip install -e ".[llama,dev,rag]"

serve:           ## Serve a model: make serve MODEL=/path/to/model.gguf [PORT=8317]
	@test -n "$(MODEL)" || { echo ">> set MODEL=/path/to/model.gguf"; exit 1; }
	$(BIN)/aero serve --model $(MODEL) --port $(PORT)

test:            ## Run the test suite in ./.venv (stub backend, no models needed)
	$(BIN)/pytest

ui:              ## Build the web UI into the package (needs Node; runtime stays Node-free)
	cd $(WEBUI) && npm ci && npm run build

ui-dev:          ## Run the Vite dev server (proxies /api and /v1 to a running `aero serve`)
	cd $(WEBUI) && npm install && npm run dev
