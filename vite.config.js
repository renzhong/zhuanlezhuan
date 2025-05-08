import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    tailwindcss(),
  ],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5050', // 请根据实际后端服务地址修改
        changeOrigin: true,
        // rewrite: (path) => path.replace(/^\/api/, '') // 如需去掉 /api 前缀可取消注释
      },
      '/static': {
        target: 'http://127.0.0.1:5050', // Flask 静态资源目录
        changeOrigin: true,
      }
    }
  }
})
