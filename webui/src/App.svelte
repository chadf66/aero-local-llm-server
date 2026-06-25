<script>
  import { onMount, onDestroy, tick } from "svelte";
  import * as api from "./lib/api.js";
  import Sidebar from "./lib/Sidebar.svelte";
  import Message from "./lib/Message.svelte";
  import Knobs from "./lib/Knobs.svelte";

  let serverState = { models: [], loaded: null };
  let conversations = [];
  let current = null; // { id, title, model, system, messages: [] }
  let model = "";
  let input = "";
  let generating = false;
  let controller = null;
  let error = "";
  let scroller;

  let knobs = { temperature: 0.7, top_p: 0.95, top_k: 40, max_tokens: null };

  $: modelInfo = serverState.models.find((m) => m.name === model) || null;
  // Map tool results (role:"tool") to their call id so ToolCards can show them.
  $: toolResults = current
    ? Object.fromEntries(
        current.messages
          .filter((m) => m.role === "tool" && m.tool_call_id)
          .map((m) => [m.tool_call_id, m.content]),
      )
    : {};

  let statePoll;
  onMount(async () => {
    await refreshState();
    await refreshList();
    statePoll = setInterval(refreshState, 3000);
    if (!model && serverState.models.length) model = serverState.models[0].name;
    newChat();
  });
  onDestroy(() => clearInterval(statePoll));

  async function refreshState() {
    try {
      serverState = await api.getState();
      if (!model && serverState.models.length) model = serverState.models[0].name;
    } catch (e) {
      error = String(e);
    }
  }
  async function refreshList(q) {
    conversations = (await api.listConversations(q)).conversations;
  }

  function newChat() {
    current = { id: null, title: "New chat", model, system: "", messages: [] };
  }

  async function openChat(id) {
    stop();
    current = await api.getConversation(id);
    if (current.model) model = current.model;
    await scrollDown();
  }

  async function removeChat(id) {
    await api.deleteConversation(id);
    if (current && current.id === id) newChat();
    await refreshList();
  }

  async function scrollDown() {
    await tick();
    if (scroller) scroller.scrollTop = scroller.scrollHeight;
  }

  function buildRequestMessages() {
    const msgs = [];
    if (current.system && current.system.trim())
      msgs.push({ role: "system", content: current.system });
    for (const m of current.messages) {
      const out = { role: m.role, content: m.content };
      if (m.tool_calls) out.tool_calls = m.tool_calls;
      if (m.tool_call_id) out.tool_call_id = m.tool_call_id;
      msgs.push(out);
    }
    return msgs;
  }

  async function send() {
    const text = input.trim();
    if (!text || generating || !model) return;
    input = "";
    error = "";

    if (current.id == null) {
      const conv = await api.createConversation({
        title: text.slice(0, 48),
        model,
        system: current.system,
      });
      current.id = conv.id;
      current.title = conv.title;
      await refreshList();
    }
    const userMsg = { role: "user", content: text };
    current.messages = [...current.messages, userMsg];
    await api.addMessage(current.id, userMsg);
    await generate();
  }

  async function generate() {
    generating = true;
    controller = new AbortController();
    const assistant = { role: "assistant", content: "", tool_calls: null };
    current.messages = [...current.messages, assistant];
    await scrollDown();

    const body = {
      model,
      messages: buildRequestMessages().slice(0, -1), // drop the empty assistant
      temperature: knobs.temperature,
      top_p: knobs.top_p,
      top_k: knobs.top_k,
    };
    if (knobs.max_tokens) body.max_tokens = Number(knobs.max_tokens);

    try {
      for await (const chunk of api.streamChat(body, controller.signal)) {
        const delta = chunk.choices?.[0]?.delta;
        if (!delta) continue;
        if (delta.content) assistant.content += delta.content;
        if (delta.tool_calls)
          assistant.tool_calls = delta.tool_calls.map((tc) => ({
            id: tc.id,
            type: "function",
            function: tc.function,
          }));
        current.messages = current.messages; // poke reactivity
        await scrollDown();
      }
    } catch (e) {
      if (e.name !== "AbortError") error = String(e);
    } finally {
      generating = false;
      controller = null;
      await api.addMessage(current.id, assistant);
      await refreshList();
    }
  }

  function stop() {
    if (controller) controller.abort();
  }

  async function regenerate() {
    if (generating || !current || current.id == null) return;
    // Drop the last assistant turn (UI + history), then re-generate from the user turn.
    const last = current.messages[current.messages.length - 1];
    if (last?.role === "assistant") {
      current.messages = current.messages.slice(0, -1);
      await api.deleteLastMessage(current.id);
    }
    await generate();
  }

  function onKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  const canRegen =
    () => current && current.messages.length >= 2 && !generating;
</script>

<div class="app">
  <Sidebar
    {conversations}
    currentId={current?.id}
    on:new={newChat}
    on:open={(e) => openChat(e.detail)}
    on:delete={(e) => removeChat(e.detail)}
    on:search={(e) => refreshList(e.detail)}
  />

  <main>
    <header>
      <select bind:value={model} disabled={generating}>
        {#each serverState.models as m}
          <option value={m.name}>{m.name}{m.tools ? "  ⚙︎" : ""}</option>
        {/each}
      </select>
      <span class="resident small">
        resident:
        {#if serverState.loaded}
          <span class="dot ok"></span>{serverState.loaded}
        {:else}
          <span class="dot"></span><span class="muted">none (idle)</span>
        {/if}
      </span>
      <div class="spacer"></div>
      <input
        class="sysprompt"
        placeholder="System prompt (optional)…"
        bind:value={current.system}
        disabled={generating || (current && current.id != null && current.messages.length > 0)}
      />
    </header>

    <div class="scroll" bind:this={scroller}>
      {#if current}
        {#each current.messages.filter((m) => m.role !== "tool") as m (m)}
          <Message message={m} {toolResults} />
        {/each}
      {/if}
      {#if error}<div class="error small">{error}</div>{/if}
    </div>

    <footer>
      {#if generating}
        <button class="stop" on:click={stop}>■ Stop</button>
      {:else}
        <button class="ghost" on:click={regenerate} disabled={!canRegen()}>↻ Regenerate</button>
      {/if}
      <textarea
        rows="1"
        placeholder="Message {model || "—"}…  (Enter to send, Shift+Enter for newline)"
        bind:value={input}
        on:keydown={onKey}
        disabled={generating}
      ></textarea>
      <button class="primary" on:click={send} disabled={generating || !input.trim()}>Send</button>
    </footer>
  </main>

  <Knobs {knobs} {model} {modelInfo} />
</div>

<style>
  .app { display: flex; height: 100vh; }
  main { flex: 1; display: flex; flex-direction: column; min-width: 0; }
  header {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.6rem 0.9rem;
    border-bottom: 1px solid var(--border);
    background: var(--panel);
  }
  .spacer { flex: 1; }
  .sysprompt { width: 40%; min-width: 12rem; }
  .resident { display: flex; align-items: center; gap: 0.35rem; }
  .dot {
    width: 0.55rem;
    height: 0.55rem;
    border-radius: 50%;
    background: var(--muted);
    display: inline-block;
  }
  .dot.ok { background: var(--ok); }
  .scroll { flex: 1; overflow-y: auto; padding: 1rem 1.2rem; }
  .error {
    color: var(--danger);
    border: 1px solid var(--danger);
    border-radius: 6px;
    padding: 0.5rem 0.7rem;
    margin: 0.5rem 0;
  }
  footer {
    display: flex;
    gap: 0.5rem;
    align-items: flex-end;
    padding: 0.7rem 0.9rem;
    border-top: 1px solid var(--border);
    background: var(--panel);
  }
  textarea {
    flex: 1;
    resize: none;
    max-height: 9rem;
    min-height: 2.4rem;
    line-height: 1.4;
  }
  .stop { border-color: var(--danger); color: var(--danger); }
</style>
