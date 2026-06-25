<script>
  import { getSizing } from "./api.js";

  export let knobs; // { temperature, top_p, top_k, max_tokens }
  export let model; // selected model name
  export let modelInfo = null; // { n_ctx, kv_cache_type, ... } for the selected model

  // Memory explainer: the KV-cache precision is a *load-time* model setting, not a
  // per-request knob, so here it's a live calculator — pick a precision, see the
  // max context that fits memory. Applying it is done in the model config (Phase f2)
  // or via `aero serve --kv-cache-type`.
  let previewKv = "f16";
  let preview = null;
  let loading = false;

  $: if (model) loadPreview(model, previewKv);

  async function loadPreview(m, kv) {
    loading = true;
    try {
      const r = await getSizing(m, kv);
      preview = r.n_ctx;
    } catch {
      preview = null;
    } finally {
      loading = false;
    }
  }
</script>

<div class="knobs">
  <h3>Sampling</h3>
  <label class="small">
    <span class="lbl">temperature <em>{knobs.temperature.toFixed(2)}</em></span>
    <input type="range" min="0" max="2" step="0.05" bind:value={knobs.temperature} />
  </label>
  <label class="small">
    <span class="lbl">top_p <em>{knobs.top_p.toFixed(2)}</em></span>
    <input type="range" min="0" max="1" step="0.01" bind:value={knobs.top_p} />
  </label>
  <label class="small">
    <span class="lbl">top_k <em>{knobs.top_k}</em></span>
    <input type="range" min="0" max="100" step="1" bind:value={knobs.top_k} />
  </label>
  <label class="small">
    <span class="lbl">max_tokens</span>
    <input type="number" min="1" placeholder="model default" bind:value={knobs.max_tokens} />
  </label>

  <h3>Memory <span class="muted">— fit bigger models</span></h3>
  <p class="small muted explain">
    The KV cache, not the weights, is what grows with context. Quantizing it
    (f16 → q8_0 → q4_0) roughly halves its size each step, buying more context on a
    fixed-memory Mac for a little quality. This is a model-load setting (set it in the
    model config or <code>aero serve --kv-cache-type</code>); below is a live preview of
    the max context each precision fits.
  </p>
  {#if modelInfo}
    <div class="small">
      current: <code>n_ctx={modelInfo.n_ctx}</code>
      <code>kv={modelInfo.kv_cache_type}</code>
    </div>
  {/if}
  <label class="small">
    <span class="lbl">preview precision</span>
    <select bind:value={previewKv}>
      <option value="f16">f16 (full)</option>
      <option value="q8_0">q8_0 (half)</option>
      <option value="q4_0">q4_0 (quarter)</option>
    </select>
  </label>
  <div class="small fits">
    {#if loading}
      <span class="muted">computing…</span>
    {:else if preview != null}
      fits <strong>{preview.toLocaleString()}</strong> tokens of context
    {:else}
      <span class="muted">preview needs the real (Metal) backend with the weights present</span>
    {/if}
  </div>
</div>

<style>
  .knobs { display: flex; flex-direction: column; gap: 0.6rem; }
  h3 {
    margin: 0.5rem 0 0.1rem;
    font-weight: 600;
    font-size: 0.92rem;
  }
  label { display: flex; flex-direction: column; gap: 0.3rem; }
  .lbl { display: flex; justify-content: space-between; align-items: baseline; }
  .lbl em { font-style: normal; color: var(--muted); }
  input[type="number"], select { background: var(--elevated); }
  input[type="range"] { width: 100%; accent-color: var(--accent); }
  .explain { line-height: 1.5; margin: 0.2rem 0; }
  .fits {
    background: var(--elevated);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.5rem 0.6rem;
  }
</style>
