import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The dashboard talks to api.py, not to Gradio. Proxying keeps every request
// same-origin so there is no CORS preflight and no hardcoded host in the app
// code — fetch('/api/identify') just works.
const API = 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    // The Three.js hero is deliberately a large lazy chunk: it is dynamically
    // imported on idle, so it never touches first paint. Raising the threshold
    // keeps the warning meaningful for chunks that WOULD block startup.
    chunkSizeWarningLimit: 700,
  },
  server: {
    proxy: {
      '/api': {
        target: API,
        changeOrigin: true,
        // Server-sent events for /api/chat/stream: without this, the proxy
        // buffers the whole response and tokens arrive all at once at the end.
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            if (proxyRes.headers['content-type']?.includes('text/event-stream')) {
              proxyRes.headers['cache-control'] = 'no-cache'
            }
          })
        },
      },
    },
  },
})
