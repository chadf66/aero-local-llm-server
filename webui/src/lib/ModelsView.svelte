<script>
  import { createEventDispatcher, onMount } from "svelte";
  import {
    listInstalledModels, listRepoModels, deleteModel, pullModel,
  } from "./api.js";
  import ModelEditor from "./ModelEditor.svelte";

  const dispatch = createEventDispatcher();

  let models = [];
  let loading = true;
  let editing = null; // null | "new" | <model detail>
  let actionError = "";

  // Pull-from-HF state
  let repo = "";
  let quants = null; // null until listed; [] if none
  let listingRepo = false;
  let repoError = "";
  let pulling = null; // filename being pulled
  let progress = null; // {pct, downloaded, total}
  let pullError = "";

  $: bases = [...new Set(models.map((m) => m.path.split("/").pop().replace(/\.gguf$/, "")))].sort();

  onMount(load);

  async function load() {
    loading = true;
    try {
      models = (await listInstalledModels()).models;
    } catch (e) {
      actionError = String(e);
    } finally {
      loading = false;
    }
  }

  async function reloadAndNotify() {
    await load();
    dispatch("changed"); // tell the chat view to refresh its model picker/state
  }

  function fmtSize(b) {
    if (b == null) return "—";
    const u = ["B", "KB", "MB", "GB", "TB"];
    let i = 0, n = b;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
  }

  async function remove(m) {
    const withWeights = m.base == null && m.referenced_by.length === 0;
    const msg = withWeights
      ? `Delete "${m.name}" and its weights (${fmtSize(m.size)})? This is permanent.`
      : `Delete "${m.name}"?`;
    if (!confirm(msg)) return;
    actionError = "";
    try {
      const res = await deleteModel(m.name, withWeights);
      if (res.note) actionError = res.note; // e.g. "kept weights: still referenced"
      await reloadAndNotify();
    } catch (e) {
      actionError = String(e);
    }
  }

  async function listQuants() {
    if (!repo.trim()) return;
    listingRepo = true;
    repoError = "";
    quants = null;
    try {
      quants = (await listRepoModels(repo.trim())).files;
    } catch (e) {
      repoError = String(e).replace(/^Error:\s*\d+:\s*/, "");
    } finally {
      listingRepo = false;
    }
  }

  async function doPull(filename) {
    pulling = filename;
    pullError = "";
    progress = { pct: 0, downloaded: 0, total: 0 };
    try {
      await pullModel(repo.trim(), filename, (ev) => {
        if (ev.type === "progress") progress = ev;
        else if (ev.type === "error") pullError = ev.detail;
      });
      if (!pullError) {
        quants = null;
        repo = "";
        await reloadAndNotify();
      }
    } catch (e) {
      pullError = String(e);
    } finally {
      pulling = null;
      progress = null;
    }
  }

  function onSaved() {
    editing = null;
    reloadAndNotify();
  }
</script>

<div class="models">
  <header>
    <button class="back" on:click={() => dispatch("back")}>← Chat</button>
    <h2>Models</h2>
    <div class="spacer"></div>
    {#if !editing}
      <button class="primary" on:click={() => (editing = "new")}>+ New model</button>
    {/if}
  </header>

  <div class="body">
    {#if editing}
      <ModelEditor
        model={editing === "new" ? null : editing}
        {bases}
        on:saved={onSaved}
        on:cancel={() => (editing = null)}
      />
    {:else}
      <!-- Pull from Hugging Face -->
      <section class="panel">
        <h3>Pull from Hugging Face</h3>
        <div class="pullbar">
          <input
            placeholder="repo id, e.g. bartowski/Qwen2.5-3B-Instruct-GGUF"
            bind:value={repo}
            on:keydown={(e) => e.key === "Enter" && listQuants()}
          />
          <button on:click={listQuants} disabled={listingRepo || !repo.trim()}>
            {listingRepo ? "Listing…" : "List quants"}
          </button>
        </div>
        {#if repoError}<p class="error small">{repoError}</p>{/if}
        {#if quants}
          {#if quants.length === 0}
            <p class="small muted">No GGUF files in that repo.</p>
          {:else}
            <ul class="quants">
              {#each quants as q}
                <li>
                  <span class="qname">{q.filename}</span>
                  <span class="muted small">{fmtSize(q.size)}</span>
                  {#if pulling === q.filename}
                    <div class="bar"><div class="fill" style="width:{progress?.pct ?? 0}%"></div></div>
                    <span class="small muted">{progress?.pct != null ? progress.pct + "%" : "…"}</span>
                  {:else}
                    <button on:click={() => doPull(q.filename)} disabled={pulling != null}>Pull</button>
                  {/if}
                </li>
              {/each}
            </ul>
          {/if}
        {/if}
        {#if pullError}<p class="error small">{pullError}</p>{/if}
      </section>

      <!-- Installed models -->
      <section class="panel">
        <h3>Installed</h3>
        {#if actionError}<p class="error small">{actionError}</p>{/if}
        {#if loading}
          <p class="muted small">Loading…</p>
        {:else if models.length === 0}
          <p class="muted small">No models yet — pull one above.</p>
        {:else}
          <table>
            <tbody>
              {#each models as m (m.name)}
                <tr>
                  <td class="nm">
                    {m.name}
                    {#if !m.exists}<span class="badge warn">missing weights</span>{/if}
                  </td>
                  <td class="badges">
                    <div class="badgewrap">
                      {#if m.base}
                        <span class="badge kind derived">derived → {m.base}</span>
                      {:else if m.has_config_file}
                        <span class="badge kind configured">configured</span>
                      {:else}
                        <span class="badge kind raw" title="No models/{m.name}.toml — runs on the serve-time defaults">raw GGUF · defaults</span>
                      {/if}
                      {#if m.tools}<span class="badge ok">tools</span>{/if}
                      <span class="badge">{m.kv_cache_type}</span>
                      <span class="badge">ctx {m.n_ctx}</span>
                    </div>
                  </td>
                  <td class="size muted small">{fmtSize(m.size)}</td>
                  <td class="ops">
                    <button class="small" on:click={() => (editing = m)}>
                      {m.base == null && !m.has_config_file ? "Configure" : "Edit"}
                    </button>
                    <button class="small del" on:click={() => remove(m)}>Delete</button>
                  </td>
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
  .models { flex: 1; display: flex; flex-direction: column; min-width: 0; height: 100vh; }
  header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.6rem 1rem;
    border-bottom: 1px solid var(--border);
  }
  header h2 { margin: 0; font-size: 1.1rem; }
  .back { border: 1px solid var(--border); }
  .spacer { flex: 1; }
  .primary { background: var(--accent); border: 1px solid var(--accent); border-radius: 8px; padding: 0.45rem 0.9rem; }
  .primary:hover { background: #5f97ff; }

  .body { flex: 1; overflow-y: auto; padding: 1.2rem; max-width: 56rem; width: 100%; margin: 0 auto; display: flex; flex-direction: column; gap: 1.2rem; }
  .panel { border: 1px solid var(--border); border-radius: 12px; padding: 1rem 1.1rem; background: var(--sidebar); }
  .panel h3 { margin: 0 0 0.7rem; }

  .pullbar { display: flex; gap: 0.5rem; }
  .pullbar input { flex: 1; background: var(--elevated); }
  .pullbar button, .quants button, .ops button { border: 1px solid var(--border); }

  .quants { list-style: none; margin: 0.7rem 0 0; padding: 0; display: flex; flex-direction: column; gap: 0.3rem; }
  .quants li { display: flex; align-items: center; gap: 0.7rem; padding: 0.35rem 0; }
  .qname { flex: 1; font-family: ui-monospace, Menlo, monospace; font-size: 0.85rem; }
  .bar { flex: 0 0 8rem; height: 0.5rem; background: var(--elevated); border-radius: 4px; overflow: hidden; }
  .fill { height: 100%; background: var(--accent); transition: width 0.2s ease; }

  table { width: 100%; border-collapse: collapse; }
  td { padding: 0.55rem 0.4rem; border-bottom: 1px solid var(--border); vertical-align: middle; }
  tr:last-child td { border-bottom: none; }
  .nm { font-weight: 500; }
  .badgewrap { display: flex; flex-wrap: wrap; gap: 0.3rem; }
  .badge { background: var(--elevated); border-radius: 5px; padding: 0.1rem 0.45rem; font-size: 0.74rem; color: var(--muted); white-space: nowrap; }
  .badge.ok { color: var(--ok); }
  .badge.warn { color: var(--danger); }
  /* "kind" badge: outlined (not filled) so it reads as a label, not a value chip. */
  .kind { background: transparent; border: 1px solid var(--border); }
  .kind.raw { color: var(--muted); border-style: dashed; }
  .kind.configured { color: var(--text); }
  .kind.derived { color: var(--accent); border-color: color-mix(in srgb, var(--accent) 50%, transparent); }
  .size { text-align: right; white-space: nowrap; }
  .ops { text-align: right; white-space: nowrap; }
  .ops .del:hover { color: var(--danger); border-color: var(--danger); }
  .error { color: var(--danger); }
</style>
