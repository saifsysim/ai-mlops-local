import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
    plugins: [react()],
    server: {
        port: 5173,
        proxy: {
            // Proxy API calls to the FastAPI backend so no CORS issues during dev
            '/agent': { target: 'http://localhost:8000', changeOrigin: true },
            '/knowledge': { target: 'http://localhost:8000', changeOrigin: true },
            '/health': { target: 'http://localhost:8000', changeOrigin: true },
        },
    },
})
