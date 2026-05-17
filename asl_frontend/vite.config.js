import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // Proxy all /ws and /api calls to FastAPI
      "/ws": {
        target: "ws://localhost:8000",
        ws: true,
      },
      "/predict": "http://localhost:8000",
      "/translate": "http://localhost:8000",
      "/reverse_landmarks": "http://localhost:8000",
      "/tts": "http://localhost:8000",
      "/clear": "http://localhost:8000",
    },
  },
});
