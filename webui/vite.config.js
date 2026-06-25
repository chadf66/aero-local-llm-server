import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// The build lands in the Python package so `aero serve` can find it relative to
// itself (store.webui_dist()) — no Node at runtime. In dev, proxy the API/SSE
// routes to a running `aero serve` on 8317 so the SPA talks to the real backend.
export default defineConfig({
  plugins: [svelte()],
  build: {
    outDir: "../src/aero/webui_dist",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8317",
      "/v1": "http://127.0.0.1:8317",
      "/healthz": "http://127.0.0.1:8317",
    },
  },
});
