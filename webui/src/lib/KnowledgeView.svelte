<script>
  import { createEventDispatcher, onMount } from "svelte";
  import {
    listKbs, getKb, createKb, deleteKb, removeKbFile,
    ingestKb, syncKb, listEmbedders,
  } from "./api.js";

  const dispatch = createEventDispatcher();

  let kbs = [];
  let embedders = [];
  let loading = true;
  let error = "";

  // create form
  let creating = false;
  let newName = "";
  let newEmbedder = "";

  // selected KB detail
  let detail = null; // manifest with files
  let busy = ""; // "" | "ingest" | "sync"
  let progress = null; // {i, total, source, status}
  let fileInput;

  onMount(load);

  async function load() {
    loading = true;
    try {
      kbs = (await listKbs()).kbs;
      embedders = (await listEmbedders()).embedders;
      if (!newEmbedder) newEmbedder = embedders[0] ?? "";
    } catch (e) {
      error = String(e);
    } finally {
      loading = false;
    }
  }

  async function reloadAndNotify() {
    await load();
    if (detail) detail = await getKb(detail.name).catch(() => null);
    dispatch("changed"); // refresh the model editor's KB list
  }

  async function open(name) {
    error = "";
    detail = await getKb(name);
  }

  async function create() {
    error = "";
    if (!newName.trim()) { error = "Give the KB a name."; return; }
    if (!newEmbedder) { error = "No embedder installed — pull one first (aero pull <repo> <file> --embedder)."; return; }
    try {
      await createKb({ name: newName.trim(), embedder: newEmbedder });
      creating = false;
      newName = "";
      await reloadAndNotify();
    } catch (e) {
      error = String(e).replace(/^Error:\s*\d+:\s*/, "");
    }
  }

  async function onFiles(e) {
    const files = [...e.target.files];
    if (!files.length || !detail) return;
    busy = "ingest";
    progress = { i: 0, total: files.length, source: "", status: "" };
    error = "";
    try {
      await ingestKb(detail.name, files, (ev) => {
        if (ev.type === "progress") progress = ev;
        else if (ev.type === "error") error = ev.detail;
      });
      await reloadAndNotify();
    } catch (e) {
      error = String(e);
    } finally {
      busy = "";
      progress = null;
      if (fileInput) fileInput.value = "";
    }
  }

  async function sync() {
    if (!detail) return;
    busy = "sync";
    progress = { i: 0, total: 0, source: "", status: "" };
    error = "";
    try {
      await syncKb(detail.name, (ev) => {
        if (ev.type === "progress") progress = ev;
        else if (ev.type === "error") error = ev.detail;
      });
      await reloadAndNotify();
    } catch (e) {
      error = String(e);
    } finally {
      busy = "";
      progress = null;
    }
  }

  async function removeFile(source) {
    if (!confirm(`Remove "${source}" from ${detail.name}?`)) return;
    detail = await removeKbFile(detail.name, source);
    dispatch("changed");
  }

  async function destroy(name) {
    if (!confirm(`Delete knowledge base "${name}" and all its data? This is permanent.`)) return;
    await deleteKb(name);
    if (detail && detail.name === name) detail = null;
    await reloadAndNotify();
  }
</script>

<div class="kb">
  <header>
    <button class="back" on:click={() => dispatch("back")}>← Chat</button>
    <h2>Knowledge bases</h2>
    <div class="spacer"></div>
    <button class="primary" on:click={() => (creating = !creating)}>+ New</button>
  </header>

  <div class="body">
    {#if error}<p class="error small">{error}</p>{/if}

    {#if creating}
      <section class="panel form">
        <h3>New knowledge base</h3>
        <div class="row2">
          <label class="field"><span>Name</span>
            <input bind:value={newName} placeholder="my-docs" /></label>
          <label class="field"><span>Embedder</span>
            <select bind:value={newEmbedder}>
              {#each embedders as e}<option value={e}>{e}</option>{/each}
            </select>
          </label>
        </div>
        {#if embedders.length === 0}
          <p class="small muted">No embedders installed. Pull one first:
            <code>aero pull &lt;repo&gt; &lt;file&gt; --embedder</code></p>
        {/if}
        <div class="actions">
          <button on:click={() => (creating = false)}>Cancel</button>
          <button class="primary" on:click={create}>Create</button>
        </div>
      </section>
    {/if}

    <section class="panel">
      <h3>Bases</h3>
      {#if loading}
        <p class="muted small">Loading…</p>
      {:else if kbs.length === 0}
        <p class="muted small">No knowledge bases yet — create one above.</p>
      {:else}
        <table>
          <tbody>
            {#each kbs as k (k.name)}
              <tr class:active={detail && detail.name === k.name}>
                <td class="nm">{k.name}</td>
                <td class="badges">
                  <span class="badge">{k.embedder}</span>
                  <span class="badge">dim {k.dim}</span>
                  <span class="badge">{k.files} files</span>
                  <span class="badge">{k.chunks} chunks</span>
                </td>
                <td class="ops">
                  <button class="small" on:click={() => open(k.name)}>Open</button>
                  <button class="small del" on:click={() => destroy(k.name)}>Delete</button>
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      {/if}
    </section>

    {#if detail}
      <section class="panel">
        <div class="detail-head">
          <h3>{detail.name}</h3>
          <div class="spacer"></div>
          <button class="small" on:click={() => fileInput.click()} disabled={busy}>+ Add files</button>
          <button class="small" on:click={sync} disabled={busy} title="Refresh changed files and prune deleted">↻ Re-index</button>
          <input type="file" multiple bind:this={fileInput} on:change={onFiles} hidden />
        </div>

        {#if busy}
          <div class="prog">
            <div class="bar"><div class="fill" style="width:{progress && progress.total ? (progress.i / progress.total * 100) : 30}%"></div></div>
            <span class="small muted">
              {busy === "sync" ? "Re-indexing" : "Ingesting"}
              {#if progress && progress.source}— {progress.status} {progress.source} ({progress.i}/{progress.total}){/if}
            </span>
          </div>
        {/if}

        {#if detail.files.length === 0}
          <p class="muted small">No files yet. Add some above.</p>
        {:else}
          <table>
            <tbody>
              {#each detail.files as f (f.source)}
                <tr>
                  <td class="fname">{f.source}</td>
                  <td class="muted small">{f.chunks} chunks</td>
                  <td class="ops"><button class="small del" on:click={() => removeFile(f.source)}>Remove</button></td>
                </tr>
              {/each}
            </tbody>
          </table>
        {/if}
      </section>
    {/if}
  </div>
</div>

<style>
  .kb { flex: 1; display: flex; flex-direction: column; min-width: 0; height: 100vh; }
  header { display: flex; align-items: center; gap: 0.8rem; padding: 0.6rem 1rem; border-bottom: 1px solid var(--border); }
  header h2 { margin: 0; font-size: 1.1rem; }
  .back { border: 1px solid var(--border); }
  .spacer { flex: 1; }
  .primary { background: var(--accent); border: 1px solid var(--accent); border-radius: 8px; padding: 0.45rem 0.9rem; }
  .primary:hover { background: #5f97ff; }

  .body { flex: 1; overflow-y: auto; padding: 1.2rem; max-width: 56rem; width: 100%; margin: 0 auto; display: flex; flex-direction: column; gap: 1.2rem; }
  .panel { border: 1px solid var(--border); border-radius: 12px; padding: 1rem 1.1rem; background: var(--sidebar); }
  .panel h3 { margin: 0 0 0.7rem; }
  .form .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; }
  .field { display: flex; flex-direction: column; gap: 0.3rem; }
  .field > span { font-size: 0.85rem; color: var(--muted); }
  .field input, .field select { background: var(--elevated); }
  .actions { display: flex; justify-content: flex-end; gap: 0.6rem; margin-top: 0.8rem; }
  .actions .primary, .actions button { border: 1px solid var(--border); }
  .actions .primary { border-color: var(--accent); }

  table { width: 100%; border-collapse: collapse; }
  td { padding: 0.55rem 0.4rem; border-bottom: 1px solid var(--border); vertical-align: middle; }
  tr:last-child td { border-bottom: none; }
  tr.active .nm { color: var(--accent); }
  .nm, .fname { font-weight: 500; }
  .fname { font-family: ui-monospace, Menlo, monospace; font-size: 0.85rem; font-weight: 400; }
  .badges { display: flex; flex-wrap: wrap; gap: 0.3rem; }
  .badge { background: var(--elevated); border-radius: 5px; padding: 0.1rem 0.45rem; font-size: 0.74rem; color: var(--muted); white-space: nowrap; }
  .ops { text-align: right; white-space: nowrap; }
  .ops button { border: 1px solid var(--border); }
  .ops .del:hover { color: var(--danger); border-color: var(--danger); }

  .detail-head { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.7rem; }
  .detail-head h3 { margin: 0; }
  .detail-head button { border: 1px solid var(--border); }
  .prog { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.8rem; }
  .bar { flex: 0 0 10rem; height: 0.5rem; background: var(--elevated); border-radius: 4px; overflow: hidden; }
  .fill { height: 100%; background: var(--accent); transition: width 0.2s ease; }
  .error { color: var(--danger); }
</style>
