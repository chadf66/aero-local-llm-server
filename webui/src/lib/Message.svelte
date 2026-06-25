<script>
  import { renderMarkdown, splitThinking } from "./markdown.js";
  import ThinkBlock from "./ThinkBlock.svelte";
  import ToolCard from "./ToolCard.svelte";

  export let message; // { role, content, tool_calls }
  export let toolResults = {}; // tool_call_id -> result content

  $: isUser = message.role === "user";
  $: segments = message.content ? splitThinking(message.content) : [];
</script>

<div class="row" class:user={isUser}>
  {#if !isUser}
    <div class="avatar" aria-label="assistant" title="aero">
      <!-- A sparkle — the common visual shorthand for AI. -->
      <svg viewBox="0 0 24 24" width="15" height="15" fill="currentColor" aria-hidden="true">
        <path d="M12 2.5l1.9 5.2a4 4 0 0 0 2.4 2.4L21.5 12l-5.2 1.9a4 4 0 0 0-2.4 2.4L12 21.5l-1.9-5.2a4 4 0 0 0-2.4-2.4L2.5 12l5.2-1.9a4 4 0 0 0 2.4-2.4L12 2.5z"/>
        <path d="M19 3.5l.7 1.8.0 0 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7.7-1.8z" opacity="0.7"/>
      </svg>
    </div>
  {/if}
  <div class="content">
    <div class="bubble" class:userbubble={isUser}>
      {#each segments as seg}
        {#if seg.type === "think"}
          <ThinkBlock text={seg.text} streaming={seg.streaming} />
        {:else if seg.text.trim()}
          <div class="md">{@html renderMarkdown(seg.text)}</div>
        {/if}
      {/each}

      {#if message.tool_calls}
        {#each message.tool_calls as call}
          <ToolCard {call} result={toolResults[call.id] ?? null} />
        {/each}
      {/if}
    </div>
  </div>
</div>

<style>
  .row {
    display: flex;
    gap: 0.85rem;
    padding: 0.6rem 0;
    align-items: flex-start;
  }
  .row.user { justify-content: flex-end; }
  .avatar {
    flex: 0 0 1.7rem;
    width: 1.7rem;
    height: 1.7rem;
    border-radius: 50%;
    background: var(--accent);
    color: #fff;
    display: grid;
    place-items: center;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 0.15rem;
  }
  .content { min-width: 0; max-width: 100%; }
  .row.user .content { max-width: 80%; }

  /* Assistant: full-width plain text. User: a rounded grey bubble. */
  .bubble :global(.md:first-child > :first-child) { margin-top: 0; }
  .bubble :global(.md:last-child > :last-child) { margin-bottom: 0; }
  .userbubble {
    background: var(--elevated);
    border-radius: 1.25rem;
    padding: 0.55rem 0.95rem;
  }
  .md :global(p) { margin: 0.6rem 0; }
  .md :global(pre) { margin: 0.7rem 0; }
  .md :global(ul), .md :global(ol) { margin: 0.5rem 0; padding-left: 1.4rem; }
  .md :global(h1), .md :global(h2), .md :global(h3) { margin: 1rem 0 0.5rem; line-height: 1.3; }
  .md :global(table) { border-collapse: collapse; margin: 0.6rem 0; }
  .md :global(th), .md :global(td) { border: 1px solid var(--border); padding: 0.35rem 0.6rem; }
</style>
