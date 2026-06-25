<script>
  export let text = "";
  export let streaming = false;
  let open = false;
  $: open = open || streaming; // auto-expand while the model is still thinking
</script>

<div class="think">
  <button class="ghost head small" on:click={() => (open = !open)}>
    {open ? "▾" : "▸"} reasoning{streaming ? "…" : ""}
  </button>
  {#if open}
    <pre class="body small">{text}</pre>
  {/if}
</div>

<style>
  .think { margin: 0.3rem 0; }
  .head { color: var(--muted); padding: 0.1rem 0.2rem; }
  .body {
    margin: 0.3rem 0 0;
    white-space: pre-wrap;
    color: var(--muted);
    background: #1a1a1a;
    border-left: 2px solid var(--border);
  }
</style>
