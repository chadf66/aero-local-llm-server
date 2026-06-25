<script>
  import { createEventDispatcher, onMount } from "svelte";
  import { createModel, editModel, getSizing } from "./api.js";

  export let model = null; // existing detail dict to edit, or null to create
  export let bases = []; // installed model/GGUF names available as `from`
  const dispatch = createEventDispatcher();

  const editing = model != null;

  let name = model?.name ?? "";
  let from = model?.base ?? (editing ? "" : bases[0] ?? "");
  let system = model?.system ?? "";
  let nctxAuto = model?.n_ctx === "auto";
  let nctx = typeof model?.n_ctx === "number" ? model.n_ctx : 4096;
  let kv = model?.kv_cache_type ?? "f16";
  let maxTokens = model?.max_tokens ?? "";
  let tools = model?.tools ?? false;
  let temperature = model?.sampling?.temperature ?? "";
  let top_p = model?.sampling?.top_p ?? "";
  let top_k = model?.sampling?.top_k ?? "";

  let saving = false;
  let error = "";

  // Live max-context preview for the chosen kv precision (uses an existing model:
  // the one we're editing, or the base we derive from).
  let preview = null;
  $: previewTarget = editing ? name : from;
  $: if (previewTarget) loadPreview(previewTarget, kv);
  async function loadPreview(target, kvType) {
    try {
      preview = (await getSizing(target, kvType)).n_ctx;
    } catch {
      preview = null;
    }
  }

  function buildBody() {
    const body = {};
    if (!editing) body.name = name.trim();
    if (from) body.from = from;
    if (system.trim()) body.system = system;
    body.n_ctx = nctxAuto ? "auto" : Number(nctx);
    body.kv_cache_type = kv;
    if (maxTokens !== "" && maxTokens != null) body.max_tokens = Number(maxTokens);
    if (tools) body.tools = true;
    const s = {};
    if (temperature !== "") s.temperature = Number(temperature);
    if (top_p !== "") s.top_p = Number(top_p);
    if (top_k !== "") s.top_k = Number(top_k);
    if (Object.keys(s).length) body.sampling = s;
    return body;
  }

  async function save() {
    error = "";
    if (!editing && !name.trim()) {
      error = "Give the model a name.";
      return;
    }
    saving = true;
    try {
      if (editing) await editModel(name, buildBody());
      else await createModel(buildBody());
      dispatch("saved");
    } catch (e) {
      error = String(e).replace(/^Error:\s*\d+:\s*/, "");
    } finally {
      saving = false;
    }
  }
</script>

<div class="editor">
  <h3>{editing ? `Edit ${name}` : "New model"}</h3>

  {#if !editing}
    <label class="field">
      <span>Name</span>
      <input bind:value={name} placeholder="my-model" />
      <small class="muted">Becomes <code>models/{name || "name"}.toml</code>.</small>
    </label>
  {/if}

  <label class="field">
    <span>Weights (<code>from</code>)</span>
    <select bind:value={from}>
      {#if editing && !bases.includes(from)}<option value={from}>{from}</option>{/if}
      {#each bases as b}<option value={b}>{b}</option>{/each}
    </select>
    <small class="muted">Which GGUF this model runs on. Several models can share one.</small>
  </label>

  <label class="field">
    <span>System prompt</span>
    <textarea rows="3" bind:value={system} placeholder="Optional — sets default behavior"></textarea>
  </label>

  <div class="row2">
    <label class="field">
      <span>Context (<code>n_ctx</code>)</span>
      <div class="inline">
        <label class="check"><input type="checkbox" bind:checked={nctxAuto} /> auto</label>
        <input type="number" min="256" bind:value={nctx} disabled={nctxAuto} />
      </div>
    </label>
    <label class="field">
      <span>KV cache</span>
      <select bind:value={kv}>
        <option value="f16">f16 (full)</option>
        <option value="q8_0">q8_0 (half)</option>
        <option value="q4_0">q4_0 (quarter)</option>
      </select>
    </label>
  </div>
  {#if preview != null}
    <p class="small muted fits">At <code>{kv}</code>, this model fits <strong>{preview.toLocaleString()}</strong> tokens of context.</p>
  {/if}

  <div class="row2">
    <label class="field">
      <span>max_tokens</span>
      <input type="number" min="1" bind:value={maxTokens} placeholder="model default" />
    </label>
    <label class="field check-field">
      <span>Tools</span>
      <label class="check"><input type="checkbox" bind:checked={tools} /> enable tool/function calling</label>
    </label>
  </div>

  <div class="row3">
    <label class="field"><span>temperature</span><input type="number" step="0.05" bind:value={temperature} placeholder="0.7" /></label>
    <label class="field"><span>top_p</span><input type="number" step="0.01" bind:value={top_p} placeholder="0.95" /></label>
    <label class="field"><span>top_k</span><input type="number" step="1" bind:value={top_k} placeholder="40" /></label>
  </div>

  {#if error}<p class="error small">{error}</p>{/if}

  <div class="actions">
    <button on:click={() => dispatch("cancel")}>Cancel</button>
    <button class="primary" on:click={save} disabled={saving}>
      {saving ? "Saving…" : editing ? "Save changes" : "Create model"}
    </button>
  </div>
</div>

<style>
  .editor {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--sidebar);
    padding: 1.1rem 1.2rem;
    display: flex;
    flex-direction: column;
    gap: 0.8rem;
  }
  h3 { margin: 0; }
  .field { display: flex; flex-direction: column; gap: 0.3rem; }
  .field > span { font-size: 0.85rem; color: var(--muted); }
  .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; }
  .row3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.8rem; }
  .inline { display: flex; align-items: center; gap: 0.6rem; }
  .inline input[type="number"] { flex: 1; }
  .check { display: inline-flex; align-items: center; gap: 0.35rem; color: var(--text); font-size: 0.9rem; }
  .check-field { justify-content: flex-start; }
  .fits {
    background: var(--elevated);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.5rem 0.6rem;
    margin: 0;
  }
  .actions { display: flex; justify-content: flex-end; gap: 0.6rem; margin-top: 0.3rem; }
  .primary { background: var(--accent); border-radius: 8px; padding: 0.5rem 0.9rem; }
  .primary:hover { background: #5f97ff; }
  .error { color: var(--danger); margin: 0; }
  button { border: 1px solid var(--border); }
  .primary { border-color: var(--accent); }
</style>
