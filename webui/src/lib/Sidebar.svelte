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
  <div class="brand">
    <svg class="mark" viewBox="0 0 64 40" width="30" height="19" aria-hidden="true">
      <path d="M6 18 C16 9,26 9,33 14 C41 20,50 20,58 13 C51 23,41 24,33 18 C26 13,16 13,6 18 Z" fill="#8fb8ff"/>
      <path d="M5 25 C16 17,26 17,34 22 C42 27,51 27,59 20 C52 30,42 31,34 26 C26 21,16 21,5 25 Z" fill="#4f8cff"/>
      <path d="M10 32 C20 26,28 26,35 30 C42 34,49 34,55 29 C49 36,42 37,35 33 C28 30,20 30,10 32 Z" fill="#2b6fe0"/>
    </svg>
    <span>aero</span>
  </div>

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

  <button class="manage" on:click={() => dispatch("manage")}>
    <svg class="ico" viewBox="0 0 24 24" width="15" height="15" fill="none"
         stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="3"/>
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
    </svg>
    Manage models
  </button>
  <button class="manage" on:click={() => dispatch("knowledge")}>
    <svg class="ico" viewBox="0 0 24 24" width="15" height="15" fill="none"
         stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <ellipse cx="12" cy="5" rx="8" ry="3"/>
      <path d="M4 5v6c0 1.66 3.58 3 8 3s8-1.34 8-3V5"/>
      <path d="M4 11v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6"/>
    </svg>
    Knowledge bases
  </button>
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
  .brand {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    padding: 0.35rem 0.5rem 0.5rem;
    color: var(--text);
  }
  .brand .mark { flex: 0 0 auto; }
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
  .manage {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    width: 100%;
    justify-content: flex-start;
    text-align: left;
    border: 1px solid var(--border);
    color: var(--muted);
    font-size: 0.9rem;
  }
  .manage:first-of-type { margin-top: auto; }
  .manage .ico { flex: 0 0 auto; opacity: 0.85; }
  .manage:hover { color: var(--text); }
</style>
