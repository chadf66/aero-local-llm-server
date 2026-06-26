"""The `aero` command-line interface.

Commands:
  serve  -- run the OpenAI-compatible server over the model set.
  pull   -- download a GGUF from Hugging Face and create a model definition.
  list   -- list available models.
  rm     -- delete a model (definition, and weights if unreferenced).
  show   -- print a model's details.
  run    -- interactive terminal chat (auto-starts a server if none is running).

Everything operates over an aero home (`~/.aero` by default): weights live in
`<home>/gguf/`, model definitions in `<home>/models/` (see store.py / config.py).
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

from . import config, store, store_ops

app = typer.Typer(help="aero — a lean, Mac-native local LLM server.", no_args_is_help=True)

# Reusable option so every command points at the same home by default.
_home_opt = typer.Option(store.DEFAULT_HOME, "--home", help="aero home directory.")


@app.callback()
def _main() -> None:
    """aero — a lean, Mac-native local LLM server."""


def _registry(home: Path) -> dict[str, config.ModelConfig]:
    return config.build_registry(store.gguf_dir(home), store.config_dir(home))


# =========================================================================== #
# serve
# =========================================================================== #


@app.command()
def serve(
    home: Path = _home_opt,
    model: list[Path] = typer.Option(
        [], "--model", "-m", help="Extra GGUF file(s) to serve (repeatable), beyond the home."
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address (localhost only by default)."),
    port: int = typer.Option(8317, "--port", "-p", help="Port to listen on."),
    n_ctx: str = typer.Option("4096", "--n-ctx", help="Default context window, or 'auto' to size to memory."),
    kv_cache_type: str = typer.Option(
        "f16", "--kv-cache-type", help="Default KV-cache precision: f16 | q8_0 | q4_0."
    ),
    mem_fraction: float = typer.Option(
        0.70, "--mem-fraction", help="Fraction of total memory to budget when n_ctx is 'auto'."
    ),
    idle_timeout: int = typer.Option(
        300, "--idle-timeout", help="Free the resident model after N idle seconds (0 = never)."
    ),
) -> None:
    """Serve the model set on the Metal GPU, loaded on demand (one resident at a time)."""
    # Imported here so other commands (and `--help`) don't pull in the heavy engine.
    import uvicorn

    from . import db, engine, server

    if kv_cache_type not in config.KV_CACHE_TYPES:
        raise typer.BadParameter(f"--kv-cache-type must be one of {', '.join(config.KV_CACHE_TYPES)}")
    if n_ctx == "auto":
        default_n_ctx: object = "auto"
    elif n_ctx.isdigit():
        default_n_ctx = int(n_ctx)
    else:
        raise typer.BadParameter("--n-ctx must be a positive integer or 'auto'")

    registry = config.build_registry(
        store.gguf_dir(home), store.config_dir(home),
        default_n_ctx=default_n_ctx, default_kv_cache_type=kv_cache_type,
    )
    for path in model:
        if not path.exists():
            raise typer.BadParameter(f"model file not found: {path}")
        registry[path.stem] = config.ModelConfig(
            name=path.stem, path=str(path), n_ctx=default_n_ctx, kv_cache_type=kv_cache_type
        )
    if not registry:
        raise typer.BadParameter(
            f"no models in {home}; run `aero pull <repo>` or pass --model <file>"
        )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    engine.configure(
        registry, backend="llama", idle_timeout=idle_timeout, mem_fraction=mem_fraction,
        home=home,
        registry_defaults={"default_n_ctx": default_n_ctx, "default_kv_cache_type": kv_cache_type},
    )
    db.connect(home)  # web-UI conversation history (off the inference path)

    typer.echo(f"Serving {len(registry)} model(s) on http://{host}:{port}  (base_url: http://{host}:{port}/v1)")
    typer.echo(f"  models: {', '.join(sorted(registry))}")
    typer.echo(f"  load-on-demand, one resident; defaults n_ctx={n_ctx}, kv={kv_cache_type}, "
               f"idle_timeout={idle_timeout}s (per-model config overrides apply)")
    if store.webui_dist() is not None:
        typer.echo(f"  web UI:  http://{host}:{port}/")
    else:
        typer.echo("  web UI:  not built (run `make ui` to enable it)")

    # Cap graceful shutdown so Ctrl+C doesn't hang waiting on an open stream — a
    # single-user local box would rather exit promptly than drain connections.
    uvicorn.run(server.app, host=host, port=port, timeout_graceful_shutdown=5)


# =========================================================================== #
# model store: pull / list / rm / show
# =========================================================================== #


@app.command()
def pull(
    repo: str = typer.Argument(..., help="Hugging Face repo id, e.g. bartowski/Qwen2.5-3B-Instruct-GGUF."),
    filename: Optional[str] = typer.Argument(
        None, help="GGUF filename in the repo. Omit to list the repo's available GGUF files."
    ),
    home: Path = _home_opt,
    embedder: bool = typer.Option(
        False, "--embedder", help="Install as an embedding model (into embedders/, no chat config)."
    ),
) -> None:
    """Download a GGUF from Hugging Face and create a model definition.

    With ``--embedder`` the GGUF goes into ``embedders/`` and no chat definition is
    written, so it's available to `/v1/embeddings` and RAG but never offered for chat."""
    try:
        import huggingface_hub  # noqa: F401  (store_ops imports it lazily)
    except ImportError as exc:
        raise typer.BadParameter("huggingface-hub not installed; run `make install-metal` (or pip install .[llama])") from exc

    if filename is None:
        ggufs = store_ops.list_repo_ggufs(repo)
        if not ggufs:
            raise typer.BadParameter(f"no .gguf files found in {repo}")
        typer.echo(f"GGUF files in {repo}:")
        for f in ggufs:
            size = f"  ({store.human_size(f['size'])})" if f.get("size") else ""
            typer.echo(f"  {f['filename']}{size}")
        typer.echo(f"\nRe-run with one, e.g.:  aero pull {repo} {ggufs[0]['filename']}")
        return

    dest = store.embedders_dir(home) if embedder else store.gguf_dir(home)
    typer.echo(f"Downloading {repo}/{filename} -> {dest} ...")

    def progress(downloaded: int, total: int) -> None:
        if total:
            print(f"\r  {store.human_size(downloaded)} / {store.human_size(total)} "
                  f"({downloaded / total * 100:5.1f}%)", end="", flush=True)
        else:
            print(f"\r  {store.human_size(downloaded)}", end="", flush=True)

    path = store_ops.download_gguf(repo, filename, dest, progress_cb=progress)
    print()  # end the progress line
    typer.echo(f"Pulled {path.stem}  ({store.human_size(path.stat().st_size)})")
    if embedder:
        typer.echo("  installed as an embedding model (use it for /v1/embeddings and RAG).")
    else:
        toml_path = store_ops.write_starter_config(home, path.stem)
        typer.echo(f"  definition: {toml_path}")


@app.command("list")
def list_models(home: Path = _home_opt) -> None:
    """List available models."""
    registry = _registry(home)
    if not registry:
        typer.echo(f"No models in {home}. Pull one with `aero pull <repo>`.")
        return
    width = max(len(n) for n in registry)
    for name, cfg in sorted(registry.items()):
        p = Path(cfg.path)
        size = store.human_size(p.stat().st_size) if p.is_file() else "missing"
        suffix = f"  → {cfg.base}" if cfg.base else ""
        typer.echo(f"{name:<{width}}  {size:>9}{suffix}")


@app.command()
def show(
    name: str = typer.Argument(..., help="Model name to inspect."),
    home: Path = _home_opt,
) -> None:
    """Print a model's details."""
    registry = _registry(home)
    cfg = registry.get(name)
    if cfg is None:
        raise typer.BadParameter(f"no model named {name!r} in {home}")

    p = Path(cfg.path)
    toml_path = store.config_dir(home) / f"{name}.toml"
    typer.echo(f"name:    {name}")
    typer.echo(f"weights: {cfg.path}" + ("" if p.is_file() else "  (missing!)"))
    if p.is_file():
        typer.echo(f"size:    {store.human_size(p.stat().st_size)}")
    if cfg.base:
        typer.echo(f"from:    {cfg.base}")
    typer.echo(f"config:  {toml_path if toml_path.is_file() else '(auto-registered GGUF, no definition file)'}")
    typer.echo(f"  n_ctx={cfg.n_ctx}  kv_cache_type={cfg.kv_cache_type}"
               + (f"  max_tokens={cfg.max_tokens}" if cfg.max_tokens else "")
               + (f"  chat_format={cfg.effective_chat_format}" if cfg.effective_chat_format else ""))
    if cfg.tools or cfg.effective_chat_format in {"chatml-function-calling", "functionary", "functionary-v1", "functionary-v2"}:
        typer.echo("  tools: enabled")
    if cfg.system:
        preview = cfg.system if len(cfg.system) <= 60 else cfg.system[:57] + "..."
        typer.echo(f"  system: {preview}")
    s = cfg.sampling
    set_fields = [f"{k}={v}" for k, v in (("temperature", s.temperature), ("top_p", s.top_p),
                  ("top_k", s.top_k), ("stop", s.stop)) if v is not None]
    if set_fields:
        typer.echo(f"  sampling: {', '.join(set_fields)}")


@app.command()
def rm(
    name: str = typer.Argument(..., help="Model name to delete."),
    home: Path = _home_opt,
    weights: bool = typer.Option(
        False, "--weights", help="Also delete the GGUF weights (base models only; refused if still referenced)."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Delete without confirmation."),
) -> None:
    """Delete a model. Derived models drop just their definition; base models can
    also drop their weights (unless another model still references them)."""
    registry = _registry(home)
    if name not in registry:
        raise typer.BadParameter(f"no model named {name!r} in {home}")

    if not yes:
        extra = " and its weights" if weights else ""
        typer.confirm(f"Delete model {name!r}{extra}?", abort=True)

    result = store_ops.delete_model(home, name, registry, weights=weights)
    for p in result["deleted"]:
        typer.echo(f"  removed {p}")
    if result["note"]:
        typer.echo(result["note"])
    typer.echo(f"Removed {name}" if result["deleted"]
               else f"Nothing to delete for {name} (weights still referenced, or already gone).")


# =========================================================================== #
# kb — knowledge bases (RAG)
# =========================================================================== #

kb_app = typer.Typer(help="Manage knowledge bases (RAG).", no_args_is_help=True)
app.add_typer(kb_app, name="kb")


def _engine_for_embedding(home: Path) -> None:
    """Configure the engine (real backend) so embedding works for CLI kb commands."""
    from . import engine
    engine.configure({}, backend="llama", idle_timeout=0, home=home)


@kb_app.command("create")
def kb_create(
    name: str = typer.Argument(..., help="Knowledge base name."),
    embedder: str = typer.Option(..., "--embedder", "-e", help="Embedding model (in embedders/)."),
    home: Path = _home_opt,
    chunk_size: int = typer.Option(1200, "--chunk-size", help="Chunk size in characters."),
    overlap: int = typer.Option(150, "--overlap", help="Chunk overlap in characters."),
) -> None:
    """Create an empty knowledge base bound to an embedding model."""
    from . import rag
    _engine_for_embedding(home)
    try:
        m = rag.create_kb(home, name, embedder, chunk_size=chunk_size, overlap=overlap)
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc))
    typer.echo(f"Created knowledge base {m['name']!r}  (embedder={m['embedder']}, dim={m['dim']})")
    typer.echo(f"  add files with:  aero kb add {m['name']} <path> ...")


@kb_app.command("add")
def kb_add(
    name: str = typer.Argument(..., help="Knowledge base name."),
    paths: list[Path] = typer.Argument(..., help="Files or directories to ingest."),
    home: Path = _home_opt,
) -> None:
    """Ingest files/directories into a knowledge base (incremental; skips unchanged)."""
    from . import rag
    if not rag.kb_exists(home, name):
        raise typer.BadParameter(f"no knowledge base named {name!r} (create it first)")
    _engine_for_embedding(home)

    def progress(i: int, total: int, source: str, status: str) -> None:
        typer.echo(f"  [{i}/{total}] {status:<8} {source}")

    try:
        result = rag.ingest(home, name, paths, progress_cb=progress)
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc))
    typer.echo(f"Ingested {result['files_ingested']} file(s), {result['chunks_added']} chunk(s); "
               f"{result['skipped']} skipped. KB now holds {result['total_files']} file(s).")


@kb_app.command("list")
def kb_list(home: Path = _home_opt) -> None:
    """List knowledge bases."""
    from . import rag
    kbs = rag.list_kbs(home)
    if not kbs:
        typer.echo(f"No knowledge bases in {home}. Create one with `aero kb create`.")
        return
    width = max(len(k["name"]) for k in kbs)
    for k in kbs:
        typer.echo(f"{k['name']:<{width}}  {k['files']:>3} files  {k['chunks']:>5} chunks  "
                   f"(embedder={k['embedder']}, dim={k['dim']})")


@kb_app.command("rm")
def kb_rm(
    name: str = typer.Argument(..., help="Knowledge base name."),
    home: Path = _home_opt,
    yes: bool = typer.Option(False, "--yes", "-y", help="Delete without confirmation."),
) -> None:
    """Delete a knowledge base (sources, index, and manifest)."""
    from . import rag
    if not rag.kb_exists(home, name):
        raise typer.BadParameter(f"no knowledge base named {name!r}")
    if not yes:
        typer.confirm(f"Delete knowledge base {name!r} and all its data?", abort=True)
    rag.delete_kb(home, name)
    typer.echo(f"Removed knowledge base {name}")


@kb_app.command("search")
def kb_search(
    name: str = typer.Argument(..., help="Knowledge base name."),
    query: str = typer.Argument(..., help="Query text."),
    home: Path = _home_opt,
    k: int = typer.Option(4, "-k", help="Number of chunks to return."),
) -> None:
    """Search a knowledge base (debugging aid; prints the top chunks)."""
    from . import rag
    if not rag.kb_exists(home, name):
        raise typer.BadParameter(f"no knowledge base named {name!r}")
    _engine_for_embedding(home)
    hits = rag.search(home, name, query, k=k)
    if not hits:
        typer.echo("(no results — is the KB empty?)")
        return
    for i, h in enumerate(hits, start=1):
        preview = " ".join(h["text"].split())[:160]
        typer.echo(f"{i}. [{h['score']:.3f}] {h['source']} (chars {h['start']}-{h['end']})")
        typer.echo(f"     {preview}…")


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
    name: str = typer.Argument(..., help="Model name to chat with."),
    home: Path = _home_opt,
    host: str = typer.Option("127.0.0.1", "--host", help="Server address."),
    port: int = typer.Option(8317, "--port", "-p", help="Server port."),
) -> None:
    """Interactive terminal chat. Auto-starts a server if one isn't already running."""
    if name not in _registry(home):
        raise typer.BadParameter(f"no model named {name!r} in {home}; pull it with `aero pull <repo>`")

    base_url = f"http://{host}:{port}"
    started: Optional[subprocess.Popen] = None

    if not _server_healthy(base_url):
        log_path = home / "serve.log"
        home.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Starting server (logs: {log_path}) ...")
        started = subprocess.Popen(
            [sys.executable, "-m", "aero.cli", "serve", "--home", str(home),
             "--host", host, "--port", str(port)],
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
