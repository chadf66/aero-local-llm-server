<script>
  import { onMount, onDestroy, tick } from "svelte";
  import * as api from "./lib/api.js";
  import Sidebar from "./lib/Sidebar.svelte";
  import Message from "./lib/Message.svelte";
  import Knobs from "./lib/Knobs.svelte";
  import ModelsView from "./lib/ModelsView.svelte";
  import KnowledgeView from "./lib/KnowledgeView.svelte";

  const emptyChat = () => ({ id: null, title: "New chat", model: "", system: "", messages: [] });

  let serverState = { models: [], loaded: null };
  let conversations = [];
  let current = emptyChat(); // always a valid chat object (never null) so the first render is safe
  let model = "";
  let input = "";
  let generating = false;
  let controller = null;
  let error = "";
  let offline = false; // true once a backend request fails (proxy/`aero serve` not reachable)
  let scroller;

  let knobs = { temperature: 0.7, top_p: 0.95, top_k: 40, max_tokens: null };
  let showSettings = false;
  let view = "chat"; // "chat" | "models" | "knowledge"

  async function onModelsChanged() {
    await refreshState();
    // If the selected model was deleted, fall back to the first available one.
    if (model && !serverState.models.some((m) => m.name === model)) {
      model = serverState.models[0]?.name ?? "";
    }
  }

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
    newChat();
  });
  onDestroy(() => clearInterval(statePoll));

  async function refreshState() {
    try {
      serverState = await api.getState();
      offline = false;
      if (!model && serverState.models.length) model = serverState.models[0].name;
    } catch (e) {
      offline = true; // backend unreachable — keep the UI usable and say so
    }
  }
  async function refreshList(q) {
    try {
      conversations = (await api.listConversations(q)).conversations;
    } catch (e) {
      offline = true;
    }
  }

  function newChat() {
    current = { ...emptyChat(), model };
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
        if (chunk.error) { error = chunk.error.message || "server error"; continue; }
        if (chunk.sources) { assistant.sources = chunk.sources; current.messages = current.messages; }
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
      // If the answer already arrived, a tail/transport hiccup isn't worth a scary
      // banner — keep what we got. Only surface errors that lost us the response.
      const gotAnswer = assistant.content || assistant.tool_calls;
      if (e.name !== "AbortError" && !gotAnswer) error = String(e);
    } finally {
      generating = false;
      controller = null;
      try {
        if (assistant.content || assistant.tool_calls)
          await api.addMessage(current.id, assistant);
        await refreshList();
      } catch {
        /* persistence is best-effort; the offline poll will surface a dead server */
      }
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
    on:new={() => { newChat(); view = "chat"; }}
    on:open={(e) => { openChat(e.detail); view = "chat"; }}
    on:delete={(e) => removeChat(e.detail)}
    on:search={(e) => refreshList(e.detail)}
    on:manage={() => (view = "models")}
    on:knowledge={() => (view = "knowledge")}
  />

  {#if view === "models"}
    <ModelsView on:back={() => (view = "chat")} on:changed={onModelsChanged} />
  {:else if view === "knowledge"}
    <KnowledgeView on:back={() => (view = "chat")} on:changed={onModelsChanged} />
  {:else}
  <main>
    <header>
      <div class="modelpick">
        <select bind:value={model} disabled={generating}>
          {#each serverState.models as m}
            <option value={m.name}>{m.name}{m.tools ? "  ⚙︎" : ""}</option>
          {/each}
        </select>
        <span class="chev" aria-hidden="true">▾</span>
      </div>
      <span class="resident small" title="Model currently held in memory">
        {#if serverState.loaded}
          <span class="dot ok"></span><span class="muted">{serverState.loaded}</span>
        {:else}
          <span class="dot"></span><span class="muted">idle</span>
        {/if}
      </span>
      <div class="spacer"></div>
      <button class="iconbtn" class:active={showSettings} title="Settings"
              on:click={() => (showSettings = !showSettings)}>⚙</button>
    </header>

    <div class="scroll" bind:this={scroller}>
      <div class="col">
        {#if offline}
          <div class="notice">
            <strong>Can't reach the aero server.</strong>
            <p class="small muted">
              Start it in another terminal with <code>aero serve</code> (it listens on
              <code>:8317</code>). In dev mode the Vite server proxies <code>/api</code> and
              <code>/v1</code> there.
            </p>
          </div>
        {:else if serverState.models.length === 0}
          <div class="notice">
            <strong>No models available.</strong>
            <p class="small muted">
              Pull one with <code>aero pull &lt;repo&gt;</code>, then restart
              <code>aero serve</code>.
            </p>
          </div>
        {:else if current.messages.length === 0}
          <div class="hero">
            <h1>What can I help with?</h1>
            <p class="muted small">Chatting with <strong>{model}</strong></p>
          </div>
        {/if}

        {#each current.messages.filter((m) => m.role !== "tool") as m (m)}
          <Message message={m} {toolResults}
                   thinking={generating && m === current.messages[current.messages.length - 1]} />
        {/each}
        {#if error}<div class="error small">{error}</div>{/if}
      </div>
    </div>

    <footer>
      <div class="col composer-col">
        {#if canRegen()}
          <div class="actions">
            <button class="pill small" on:click={regenerate}>↻ Regenerate</button>
          </div>
        {/if}
        <div class="composer" class:disabled={!model}>
          <textarea
            rows="1"
            placeholder={model ? `Message ${model}…` : "No model available"}
            bind:value={input}
            on:keydown={onKey}
            disabled={generating || !model}
          ></textarea>
          {#if generating}
            <button class="send stop" on:click={stop} title="Stop">■</button>
          {:else}
            <button class="send" on:click={send}
                    disabled={!input.trim() || !model} title="Send">↑</button>
          {/if}
        </div>
        <p class="disclaimer small muted">Local models can make mistakes. Responses run on your Mac.</p>
      </div>
    </footer>
  </main>

  {#if showSettings}
    <aside class="settings">
      <div class="settings-head">
        <span>Settings</span>
        <button class="iconbtn" title="Close" on:click={() => (showSettings = false)}>✕</button>
      </div>
      <label class="field small">
        System prompt
        <textarea
          class="sysprompt"
          rows="3"
          placeholder="Optional — sets the assistant's behavior for this chat"
          bind:value={current.system}
          disabled={generating || (current.id != null && current.messages.length > 0)}
        ></textarea>
      </label>
      <Knobs {knobs} {model} {modelInfo} />
    </aside>
  {/if}
  {/if}
</div>

<style>
  .app { display: flex; height: 100vh; overflow: hidden; }
  main { flex: 1; display: flex; flex-direction: column; min-width: 0; }

  /* Centered conversation + composer column, ChatGPT-style. */
  .col { width: 100%; max-width: var(--col); margin: 0 auto; }

  header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.55rem 0.9rem;
    height: 3rem;
  }
  .modelpick { position: relative; display: inline-flex; align-items: center; }
  .modelpick select {
    appearance: none;
    background: transparent;
    border: none;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.35rem 1.6rem 0.35rem 0.5rem;
    border-radius: 8px;
    cursor: pointer;
    max-width: 22rem;
  }
  .modelpick select:hover { background: var(--hover); }
  .modelpick .chev { position: absolute; right: 0.55rem; pointer-events: none; color: var(--muted); font-size: 0.75rem; }
  .resident { display: flex; align-items: center; gap: 0.35rem; }
  .dot { width: 0.5rem; height: 0.5rem; border-radius: 50%; background: var(--muted); display: inline-block; }
  .dot.ok { background: var(--ok); box-shadow: 0 0 6px var(--ok); }
  .spacer { flex: 1; }
  .iconbtn { font-size: 1.05rem; line-height: 1; padding: 0.4rem 0.5rem; border-radius: 8px; }
  .iconbtn.active { background: var(--hover); }

  .scroll { flex: 1; overflow-y: auto; padding: 1rem 1rem 0; }

  .hero { text-align: center; padding: 18vh 1rem 2rem; }
  .hero h1 { font-size: 1.7rem; font-weight: 600; margin: 0 0 0.4rem; }

  .notice {
    max-width: 34rem;
    margin: 14vh auto 2rem;
    text-align: center;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.3rem 1.5rem;
    background: var(--sidebar);
  }
  .notice p { line-height: 1.5; margin: 0.5rem 0 0; }
  .error {
    color: var(--danger);
    border: 1px solid var(--danger);
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    margin: 0.6rem 0;
  }

  /* Composer */
  footer { padding: 0.4rem 1rem 0.6rem; }
  .composer-col { display: flex; flex-direction: column; align-items: stretch; }
  .actions { display: flex; justify-content: center; margin-bottom: 0.5rem; }
  .pill { border: 1px solid var(--border); border-radius: 999px; padding: 0.3rem 0.85rem; }
  .composer {
    display: flex;
    align-items: flex-end;
    gap: 0.5rem;
    background: var(--elevated);
    border: 1px solid var(--border);
    border-radius: 1.6rem;
    padding: 0.4rem 0.5rem 0.4rem 1rem;
  }
  .composer:focus-within { border-color: var(--elevated-2); }
  .composer.disabled { opacity: 0.6; }
  .composer textarea {
    flex: 1;
    background: transparent;
    border: none;
    resize: none;
    max-height: 12rem;
    min-height: 1.6rem;
    padding: 0.45rem 0;
    line-height: 1.5;
  }
  .composer textarea:focus { border: none; }
  .send {
    flex: 0 0 auto;
    width: 2.1rem;
    height: 2.1rem;
    border-radius: 50%;
    background: var(--text);
    color: #111;
    font-size: 1.05rem;
    display: grid;
    place-items: center;
    padding: 0;
  }
  .send:hover { background: #fff; }
  .send:disabled { background: var(--elevated-2); color: var(--muted); }
  .send.stop { background: var(--text); color: #111; font-size: 0.8rem; }
  .disclaimer { text-align: center; margin: 0.45rem 0 0; }

  /* Settings panel */
  .settings {
    width: 320px;
    flex: 0 0 320px;
    background: var(--sidebar);
    border-left: 1px solid var(--border);
    height: 100vh;
    overflow-y: auto;
    padding: 0.9rem;
    display: flex;
    flex-direction: column;
    gap: 0.8rem;
  }
  .settings-head { display: flex; align-items: center; justify-content: space-between; font-weight: 600; }
  .field { display: flex; flex-direction: column; gap: 0.35rem; }
  .sysprompt { background: var(--elevated); resize: vertical; line-height: 1.45; }
</style>
