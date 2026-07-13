import path from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": path.resolve(import.meta.dirname, "./src") },
  },
  server: {
    // 5173 is not incidental: it is one of the two origins serving/app.py's
    // CORS_ALLOW_ORIGINS enumerates. The API refuses any other origin by design
    // (it does not use "*"), so changing this port means editing that list too
    // -- deliberately, rather than discovering it as a CORS error.
    port: 5173,
    strictPort: true,
  },
})
