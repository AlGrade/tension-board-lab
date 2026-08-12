import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  // onnxruntime-web ships wasm binaries as assets; the dep optimizer mangles them.
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  test: {
    globals: true,
    environment: "node",
  },
});
