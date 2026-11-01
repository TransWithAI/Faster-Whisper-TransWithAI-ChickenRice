import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // 使用相对路径，保证 PyInstaller 打包后（本地静态目录）也能正确加载 assets，
  // 避免 "/assets/..." 这类绝对路径被解析到磁盘根目录而 404。
  base: "./",
  server: {
    port: 5173,
    strictPort: true,
  },
  build: {
    outDir: "dist",
    chunkSizeWarningLimit: 2000,
  },
});
