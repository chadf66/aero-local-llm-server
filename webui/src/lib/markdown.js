// Render assistant markdown to safe HTML, and split out <think>...</think>
// reasoning blocks so the UI can collapse them.
import { marked } from "marked";
import { markedHighlight } from "marked-highlight";
import DOMPurify from "dompurify";

// highlight.js core + a curated language set (keeps the bundle lean rather than
// pulling in all ~190 grammars). Unknown languages fall back to plaintext.
import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import c from "highlight.js/lib/languages/c";
import cpp from "highlight.js/lib/languages/cpp";
import csharp from "highlight.js/lib/languages/csharp";
import css from "highlight.js/lib/languages/css";
import go from "highlight.js/lib/languages/go";
import java from "highlight.js/lib/languages/java";
import javascript from "highlight.js/lib/languages/javascript";
import json from "highlight.js/lib/languages/json";
import markdown from "highlight.js/lib/languages/markdown";
import python from "highlight.js/lib/languages/python";
import rust from "highlight.js/lib/languages/rust";
import sql from "highlight.js/lib/languages/sql";
import typescript from "highlight.js/lib/languages/typescript";
import xml from "highlight.js/lib/languages/xml";
import yaml from "highlight.js/lib/languages/yaml";

for (const [name, lang] of Object.entries({
  bash, c, cpp, csharp, css, go, java, javascript, json, markdown,
  python, rust, sql, typescript, xml, yaml,
})) {
  hljs.registerLanguage(name, lang);
}
// Common aliases so ```sh / ```py / ```js / ```html etc. resolve.
hljs.registerAliases(["sh", "shell", "zsh"], { languageName: "bash" });
hljs.registerAliases(["py"], { languageName: "python" });
hljs.registerAliases(["js", "jsx"], { languageName: "javascript" });
hljs.registerAliases(["ts", "tsx"], { languageName: "typescript" });
hljs.registerAliases(["html", "svelte", "vue"], { languageName: "xml" });
hljs.registerAliases(["yml"], { languageName: "yaml" });

marked.use(
  markedHighlight({
    langPrefix: "hljs language-",
    highlight(code, lang) {
      const language = lang && hljs.getLanguage(lang) ? lang : "plaintext";
      return hljs.highlight(code, { language }).value;
    },
  }),
);
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
