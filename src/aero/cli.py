"""The `aero` command-line interface.

Commands:
  serve  -- run the OpenAI-compatible server over a set of models.
  pull   -- download a GGUF from Hugging Face into the model store.
  list   -- list locally available models.
  rm     -- delete a local model.
  show   -- print a model's details.
  run    -- interactive terminal chat (auto-starts a server if none is running).
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator, Optional

import typer

from . import store

app = typer.Typer(help="aero — a lean, Mac-native local LLM server.", no_args_is_help=True)

# Reusable option so every command points at the same store by default.
_models_dir_opt = typer.Option(store.DEFAULT_MODELS_DIR, "--models-dir", help="Model store directory.")


@app.callback()
def _main() -> None:
    """aero — a lean, Mac-native local LLM server."""


# =========================================================================== #
# serve
# =========================================================================== #


@app.command()
def serve(
    models_dir: Path = _models_dir_opt,
    model: list[Path] = typer.Option(
        [], "--model", "-m", help="Extra GGUF file(s) to serve (repeatable), beyond --models-dir."
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address (localhost only by default)."),
    port: int = typer.Option(8317, "--port", "-p", help="Port to listen on."),
    n_ctx: int = typer.Option(4096, "--n-ctx", help="Context window size (tokens)."),
    kv_cache_type: str = typer.Option(
        "f16", "--kv-cache-type", help="KV-cache precision: f16 | q8_0 | q4_0 (quantize to fit more context)."
    ),
    idle_timeout: int = typer.Option(
        300, "--idle-timeout", help="Free the resident model after N idle seconds (0 = never)."
    ),
) -> None:
    """Serve a set of GGUF models on the Metal GPU, loaded on demand (one at a time)."""
    # Imported here so other commands (and `--help`) don't pull in the heavy engine.
    import uvicorn

    from . import engine, server

    if kv_cache_type not in engine.KV_CACHE_TYPES:
        raise typer.BadParameter(f"--kv-cache-type must be one of {', '.join(engine.KV_CACHE_TYPES)}")

    registry = store.scan(models_dir)
    for path in model:
        if not path.exists():
            raise typer.BadParameter(f"model file not found: {path}")
        registry[path.stem] = str(path)
    if not registry:
        raise typer.BadParameter(
            f"no models found in {models_dir} and none passed with --model; "
            "run `aero pull <repo>` or pass --model <file>"
        )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    engine.configure(
        registry, n_ctx=n_ctx, kv_cache_type=kv_cache_type, backend="llama", idle_timeout=idle_timeout
    )

    typer.echo(f"Serving {len(registry)} model(s) on http://{host}:{port}  (base_url: http://{host}:{port}/v1)")
    typer.echo(f"  models: {', '.join(sorted(registry))}")
    typer.echo(f"  load-on-demand, one resident; n_ctx={n_ctx}, kv={kv_cache_type}, idle_timeout={idle_timeout}s")

    uvicorn.run(server.app, host=host, port=port)


# =========================================================================== #
# model store: pull / list / rm / show
# =========================================================================== #


@app.command()
def pull(
    repo: str = typer.Argument(..., help="Hugging Face repo id, e.g. bartowski/Qwen2.5-3B-Instruct-GGUF."),
    filename: Optional[str] = typer.Argument(
        None, help="GGUF filename in the repo. Omit to list the repo's available GGUF files."
    ),
    models_dir: Path = _models_dir_opt,
) -> None:
    """Download a GGUF from Hugging Face into the model store."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise typer.BadParameter("huggingface-hub not installed; run `make install-metal` (or pip install .[llama])") from exc

    if filename is None:
        # Discovery: show the GGUF files in the repo so the user can pick one.
        ggufs = sorted(f for f in HfApi().list_repo_files(repo) if f.endswith(".gguf"))
        if not ggufs:
            raise typer.BadParameter(f"no .gguf files found in {repo}")
        typer.echo(f"GGUF files in {repo}:")
        for f in ggufs:
            typer.echo(f"  {f}")
        typer.echo(f"\nRe-run with one, e.g.:  aero pull {repo} {ggufs[0]}")
        return

    models_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Downloading {repo}/{filename} -> {models_dir} ...")
    path = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(models_dir))
    typer.echo(f"Pulled {Path(path).stem}  ({store.human_size(Path(path).stat().st_size)})")


@app.command("list")
def list_models(models_dir: Path = _models_dir_opt) -> None:
    """List locally available models."""
    models = store.scan(models_dir)
    if not models:
        typer.echo(f"No models in {models_dir}. Pull one with `aero pull <repo>`.")
        return
    width = max(len(n) for n in models)
    for name, path in models.items():
        size = store.human_size(Path(path).stat().st_size)
        typer.echo(f"{name:<{width}}  {size:>9}")


@app.command()
def rm(
    name: str = typer.Argument(..., help="Model name (the filename stem) to delete."),
    models_dir: Path = _models_dir_opt,
    yes: bool = typer.Option(False, "--yes", "-y", help="Delete without confirmation."),
) -> None:
    """Delete a local model from the store."""
    path = store.find(models_dir, name)
    if path is None:
        raise typer.BadParameter(f"no model named {name!r} in {models_dir}")
    if not yes:
        typer.confirm(f"Delete {name} ({store.human_size(path.stat().st_size)})?", abort=True)
    store.remove(models_dir, name)
    typer.echo(f"Removed {name}")


@app.command()
def show(
    name: str = typer.Argument(..., help="Model name (the filename stem) to inspect."),
    models_dir: Path = _models_dir_opt,
) -> None:
    """Print a model's details."""
    path = store.find(models_dir, name)
    if path is None:
        raise typer.BadParameter(f"no model named {name!r} in {models_dir}")
    typer.echo(f"name:  {name}")
    typer.echo(f"path:  {path}")
    typer.echo(f"size:  {store.human_size(path.stat().st_size)}")
    # Quantization label parsed from the filename (e.g. Q4_K_M) — cheap and useful
    # without loading the model. Richer GGUF metadata can come with Phase d.
    quant = next((p for p in path.stem.split(".") if p.upper().startswith("Q") and "_" in p), None)
    if quant:
        typer.echo(f"quant: {quant}")


# =========================================================================== #
# run — interactive chat (auto-starts a server if needed)
# =========================================================================== #


def _server_healthy(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/healthz", timeout=1) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _stream_chat(base_url: str, model: str, messages: list[dict]) -> Iterator[str]:
    """POST a streaming chat completion and yield assistant content pieces."""
    payload = json.dumps({"model": model, "messages": messages, "stream": True}).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode("utf-8").rstrip("\n")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            choices = json.loads(data).get("choices") or []
            if choices:
                piece = choices[0].get("delta", {}).get("content")
                if piece:
                    yield piece


@app.command()
def run(
    name: str = typer.Argument(..., help="Model name (the filename stem) to chat with."),
    models_dir: Path = _models_dir_opt,
    host: str = typer.Option("127.0.0.1", "--host", help="Server address."),
    port: int = typer.Option(8317, "--port", "-p", help="Server port."),
    n_ctx: int = typer.Option(4096, "--n-ctx", help="Context window if we start the server."),
    kv_cache_type: str = typer.Option("f16", "--kv-cache-type", help="KV-cache precision if we start the server."),
) -> None:
    """Interactive terminal chat. Auto-starts a server if one isn't already running."""
    if store.find(models_dir, name) is None:
        raise typer.BadParameter(f"no model named {name!r} in {models_dir}; pull it with `aero pull <repo>`")

    base_url = f"http://{host}:{port}"
    started: Optional[subprocess.Popen] = None

    if not _server_healthy(base_url):
        log_path = models_dir.parent / "serve.log"
        models_dir.parent.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Starting server (logs: {log_path}) ...")
        started = subprocess.Popen(
            [sys.executable, "-m", "aero.cli", "serve", "--models-dir", str(models_dir),
             "--host", host, "--port", str(port), "--n-ctx", str(n_ctx), "--kv-cache-type", kv_cache_type],
            stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
        )
        deadline = time.time() + 60
        while time.time() < deadline and not _server_healthy(base_url):
            if started.poll() is not None:
                typer.echo(f"server exited during startup; see {log_path}")
                raise typer.Exit(1)
            time.sleep(0.3)
        if not _server_healthy(base_url):
            started.terminate()
            typer.echo(f"server did not become healthy; see {log_path}")
            raise typer.Exit(1)

    typer.echo(f"Chatting with {name}. Type /bye to quit, /reset to clear history.\n")
    messages: list[dict] = []
    try:
        while True:
            try:
                user = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user:
                continue
            if user in ("/bye", "/exit", "/quit"):
                break
            if user == "/reset":
                messages = []
                typer.echo("(history cleared)")
                continue
            messages.append({"role": "user", "content": user})
            reply = ""
            try:
                for piece in _stream_chat(base_url, name, messages):
                    print(piece, end="", flush=True)
                    reply += piece
            except urllib.error.HTTPError as exc:
                typer.echo(f"\n[error] {exc.code}: {exc.read().decode()[:200]}")
                messages.pop()
                continue
            print("\n")
            messages.append({"role": "assistant", "content": reply})
    finally:
        if started is not None:
            started.terminate()
            try:
                started.wait(timeout=5)
            except subprocess.TimeoutExpired:
                started.kill()


if __name__ == "__main__":
    app()
