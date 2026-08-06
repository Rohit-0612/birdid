import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const BACKEND = 'http://127.0.0.1:7860'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/gradio_api': {
        target: BACKEND,
        changeOrigin: true,
        ws: true,
      },
      '/upload': { target: BACKEND, changeOrigin: true },
      '/file=': { target: BACKEND, changeOrigin: true },
      '/file/': { target: BACKEND, changeOrigin: true },
      '/config': { target: BACKEND, changeOrigin: true },
      '/heartbeat': { target: BACKEND, changeOrigin: true },
      '/api': { target: BACKEND, changeOrigin: true },
      '/run': { target: BACKEND, changeOrigin: true },
      '/call': { target: BACKEND, changeOrigin: true },
      '/queue': { target: BACKEND, changeOrigin: true },
    },
  },
})
