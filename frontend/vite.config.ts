/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import visualizer from 'rollup-plugin-visualizer'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss(), visualizer({ open: true, gzipSize: true })],
  test: {
    environment: 'happy-dom',
    globals: true,
    setupFiles: './src/shared/test/setup.ts',
    exclude: ['**/node_modules/**', '**/dist/**', '**/e2e/**']
  },
})
