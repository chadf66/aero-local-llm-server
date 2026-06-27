// Thin wrappers over aero's HTTP API. Two distinct surfaces:
//   /v1/...  — the OpenAI-compatible inference API (the source of truth)
//   /api/... — the web-UI-only state + conversation history endpoints
// Generation goes through /v1 exactly as an agent would; history is persisted
// separately, so the UI never adds state to the inference path.

async function json(res) {
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

// --- server state ---------------------------------------------------------
export const getState = () => fetch("/api/state").then(json);
export const getSizing = (model, kv) =>
  fetch(`/api/sizing?model=${encodeURIComponent(model)}&kv_cache_type=${kv}`).then(json);

// --- conversation history -------------------------------------------------
export const listConversations = (q) =>
  fetch("/api/conversations" + (q ? `?q=${encodeURIComponent(q)}` : "")).then(json);
export const getConversation = (id) => fetch(`/api/conversations/${id}`).then(json);
export const createConversation = (body) =>
  fetch("/api/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then(json);
export const patchConversation = (id, body) =>
  fetch(`/api/conversations/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then(json);
export const deleteConversation = (id) =>
  fetch(`/api/conversations/${id}`, { method: "DELETE" }).then(json);
export const addMessage = (id, msg) =>
  fetch(`/api/conversations/${id}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(msg),
  }).then(json);
export const deleteLastMessage = (id) =>
  fetch(`/api/conversations/${id}/messages/last`, { method: "DELETE" }).then(json);

// --- model management (Phase f2) ------------------------------------------
export const listInstalledModels = () => fetch("/api/models").then(json);
export const listRepoModels = (repo) =>
  fetch(`/api/models/repo?repo=${encodeURIComponent(repo)}`).then(json);
export const createModel = (body) =>
  fetch("/api/models", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then(json);
export const editModel = (name, body) =>
  fetch(`/api/models/${encodeURIComponent(name)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then(json);
export const deleteModel = (name, weights = false) =>
  fetch(`/api/models/${encodeURIComponent(name)}?weights=${weights}`, {
    method: "DELETE",
  }).then(json);

// Stream a pull's SSE progress. `onEvent` gets each parsed frame
// ({type:"progress"|"done"|"error", ...}); resolves when the stream ends.
export async function pullModel(repo, filename, onEvent, signal, embedder = false) {
  const url = `/api/models/pull?repo=${encodeURIComponent(repo)}&filename=${encodeURIComponent(filename)}`
    + (embedder ? "&embedder=true" : "");
  const res = await fetch(url, { signal });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${detail}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop();
    for (const line of lines) {
      const t = line.trim();
      if (!t.startsWith("data: ")) continue;
      const data = t.slice(6);
      if (data === "[DONE]") return;
      try {
        onEvent(JSON.parse(data));
      } catch {
        /* ignore */
      }
    }
  }
}

// --- knowledge bases (Phase g4) -------------------------------------------
export const listEmbedders = () => fetch("/api/embedders").then(json);
export const listKbs = () => fetch("/api/kb").then(json);
export const getKb = (name) => fetch(`/api/kb/${encodeURIComponent(name)}`).then(json);
export const createKb = (body) =>
  fetch("/api/kb", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then(json);
export const deleteKb = (name) =>
  fetch(`/api/kb/${encodeURIComponent(name)}`, { method: "DELETE" }).then(json);
export const removeKbFile = (name, source) =>
  fetch(`/api/kb/${encodeURIComponent(name)}/files/${encodeURIComponent(source)}`, {
    method: "DELETE",
  }).then(json);

// Read an SSE response, calling onEvent per parsed frame; resolves on [DONE].
async function readSSE(res, onEvent) {
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${detail}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop();
    for (const line of lines) {
      const t = line.trim();
      if (!t.startsWith("data: ")) continue;
      const data = t.slice(6);
      if (data === "[DONE]") return;
      try {
        onEvent(JSON.parse(data));
      } catch {
        /* ignore */
      }
    }
  }
}

// Upload files to a KB and stream ingest progress (onEvent per SSE frame).
export async function ingestKb(name, files, onEvent) {
  const form = new FormData();
  for (const f of files) form.append("files", f, f.name);
  const res = await fetch(`/api/kb/${encodeURIComponent(name)}/ingest`, {
    method: "POST",
    body: form,
  });
  return readSSE(res, onEvent);
}

// Re-index a KB from its sources (refresh + prune), streaming progress.
export async function syncKb(name, onEvent) {
  const res = await fetch(`/api/kb/${encodeURIComponent(name)}/sync`, { method: "POST" });
  return readSSE(res, onEvent);
}

// --- streaming chat over /v1/chat/completions -----------------------------
// Yields parsed OpenAI chunk objects. `signal` lets the caller stop generation.
export async function* streamChat(body, signal) {
  const res = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...body, stream: true }),
    signal,
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${detail}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop(); // keep the partial line
    for (const line of lines) {
      const t = line.trim();
      if (!t.startsWith("data: ")) continue;
      const data = t.slice(6);
      if (data === "[DONE]") return;
      try {
        yield JSON.parse(data);
      } catch {
        /* ignore keep-alives / partial frames */
      }
    }
  }
}
