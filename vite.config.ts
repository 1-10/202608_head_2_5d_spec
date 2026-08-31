import { readFileSync } from 'node:fs';
import { defineConfig } from 'vite';

// `guest.json` の `exporter_version` に入る値。デスクトップ側が
// `importlib.metadata.version` でパッケージのバージョンを引くのと同じ役割。
const { version } = JSON.parse(readFileSync('./package.json', 'utf-8')) as { version: string };

export default defineConfig({
  define: {
    __EXPORTER_VERSION__: JSON.stringify(version),
  },
  server: {
    host: true,
  },
  optimizeDeps: {
    // onnxruntime-web は esbuild の事前バンドルで wasm/WebGPU 初期化が壊れる
    // (webgpuInit is not a function)。素の ESM のまま配信する。
    exclude: ['onnxruntime-web'],
  },
  test: {
    // domain は純粋計算なので DOM を要らない。infrastructure と presentation は
    // ブラウザでしか動かないのでテストしない（実機で確認する）。
    include: ['tests/**/*.test.ts'],
    environment: 'node',
  },
});
