import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite dev server. We proxy /api and /ws to the Rust server on :8088 so the
// frontend can stay on :5173 in dev without CORS gymnastics.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8088", changeOrigin: true },
      "/ws": { target: "ws://127.0.0.1:8088", ws: true, changeOrigin: true },
      "/healthz": { target: "http://127.0.0.1:8088", changeOrigin: true },
    },
  },
});
