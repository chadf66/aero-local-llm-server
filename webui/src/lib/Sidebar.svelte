<script>
  import { createEventDispatcher } from "svelte";
  export let conversations = [];
  export let currentId = null;
  const dispatch = createEventDispatcher();

  let query = "";
  let timer;
  function onSearch() {
    clearTimeout(timer);
    timer = setTimeout(() => dispatch("search", query.trim()), 200);
  }
</script>

<aside class="sidebar">
  <button class="new" on:click={() => dispatch("new")}>
    <span class="plus">+</span> New chat
  </button>

  <input class="search" placeholder="Search chats" bind:value={query} on:input={onSearch} />

  <div class="list">
    {#each conversations as c (c.id)}
      <div
        class="item"
        class:active={c.id === currentId}
        role="button"
        tabindex="0"
        on:click={() => dispatch("open", c.id)}
        on:keydown={(e) => (e.key === "Enter" || e.key === " ") && dispatch("open", c.id)}
      >
        <span class="title" title={c.title}>{c.title}</span>
        <button
          class="del"
          title="Delete chat"
          on:click|stopPropagation={() => dispatch("delete", c.id)}>✕</button>
      </div>
    {:else}
      <p class="small muted empty">No chats yet.</p>
    {/each}
  </div>
</aside>

<style>
  .sidebar {
    width: 260px;
    flex: 0 0 260px;
    background: var(--sidebar);
    display: flex;
    flex-direction: column;
    padding: 0.6rem;
    gap: 0.4rem;
    height: 100vh;
  }
  .new {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    width: 100%;
    justify-content: flex-start;
    font-weight: 500;
    border: 1px solid var(--border);
  }
  .plus { font-size: 1.1rem; line-height: 1; }
  .search { width: 100%; background: var(--elevated); margin-bottom: 0.3rem; }
  .list { overflow-y: auto; display: flex; flex-direction: column; gap: 0.1rem; }
  .item {
    position: relative;
    display: flex;
    align-items: center;
    padding: 0.5rem 0.6rem;
    border-radius: 8px;
    cursor: pointer;
    user-select: none;
  }
  .item:hover { background: var(--hover); }
  .item.active { background: var(--elevated); }
  .title {
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 0.9rem;
  }
  .del {
    opacity: 0;
    padding: 0.15rem 0.35rem;
    font-size: 0.75rem;
    border-radius: 6px;
    color: var(--muted);
  }
  .item:hover .del { opacity: 1; }
  .del:hover { color: var(--danger); background: var(--hover); }
  .empty { padding: 0.6rem; }
</style>
