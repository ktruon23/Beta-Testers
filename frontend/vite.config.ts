import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

// The browser calls /api/...; Vite forwards it to the orchestrator, so no CORS setup is needed.
// Point at a different orchestrator with ORCHESTRATOR_URL=http://host:8000 npm run dev
const orchestrator = process.env.ORCHESTRATOR_URL || "http://localhost:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: orchestrator,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
        timeout: 300_000,
        proxyTimeout: 300_000,
      },
    },
  },
});
