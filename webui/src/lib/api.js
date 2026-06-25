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
