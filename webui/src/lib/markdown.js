// Render assistant markdown to safe HTML, and split out <think>...</think>
// reasoning blocks so the UI can collapse them.
import { marked } from "marked";
import DOMPurify from "dompurify";

marked.setOptions({ gfm: true, breaks: true });

export function renderMarkdown(text) {
  return DOMPurify.sanitize(marked.parse(text || ""));
}

// Split content into ordered segments: { type: "think" | "text", text }.
// Reasoning models emit <think>...</think>; we keep them as separate, collapsible
// segments rather than rendering them inline with the answer.
export function splitThinking(content) {
  const out = [];
  const re = /<think>([\s\S]*?)<\/think>/gi;
  let last = 0;
  let m;
  while ((m = re.exec(content)) !== null) {
    if (m.index > last) out.push({ type: "text", text: content.slice(last, m.index) });
    out.push({ type: "think", text: m[1].trim() });
    last = re.lastIndex;
  }
  if (last < content.length) out.push({ type: "text", text: content.slice(last) });
  // An unterminated <think> (mid-stream) — show the tail as thinking.
  const open = content.lastIndexOf("<think>");
  if (open !== -1 && content.indexOf("</think>", open) === -1) {
    const before = content.slice(0, open);
    return [
      ...(before ? [{ type: "text", text: before }] : []),
      { type: "think", text: content.slice(open + 7), streaming: true },
    ];
  }
  return out.length ? out : [{ type: "text", text: content }];
}
