import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    host: true,
  },
  optimizeDeps: {
    // onnxruntime-webはesbuildの事前バンドルでwasm/WebGPU初期化が壊れる
    // (webgpuInit is not a function)。素のESMのまま配信する
    exclude: ['onnxruntime-web'],
  },
});
