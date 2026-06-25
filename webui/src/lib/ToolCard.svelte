<script>
  // Renders a tool call the assistant emitted (and, if present, its result).
  export let call; // { id, function: { name, arguments } }
  export let result = null; // the matching role:"tool" content, if any

  function pretty(json) {
    try {
      return JSON.stringify(JSON.parse(json), null, 2);
    } catch {
      return json;
    }
  }
</script>

<div class="card">
  <div class="head small">
    <span class="badge">tool call</span>
    <code>{call.function.name}</code>
  </div>
  <pre class="small">{pretty(call.function.arguments || "{}")}</pre>
  {#if result != null}
    <div class="head small"><span class="badge ok">result</span></div>
    <pre class="small">{result}</pre>
  {/if}
</div>

<style>
  .card {
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--elevated);
    padding: 0.5rem 0.6rem;
    margin: 0.4rem 0;
  }
  .head { display: flex; align-items: center; gap: 0.4rem; margin-bottom: 0.25rem; }
  .badge {
    background: rgba(79, 140, 255, 0.16);
    color: var(--accent);
    border-radius: 4px;
    padding: 0.05rem 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 0.7rem;
  }
  .badge.ok { background: #143a2a; color: var(--ok); }
  pre { margin: 0.2rem 0; }
</style>
