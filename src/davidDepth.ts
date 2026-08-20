// DAViD (Microsoft, ICCV 2025) による人物特化の単眼相対Depth推定。
// 100%合成データ (SynthHuman, CDLA-Permissive-2.0) 学習 + モデルMIT =
// 「学習データまで商用クリーン」を満たす、ARPortraitDepthの上位互換候補。
// https://github.com/microsoft/DAViD
//
// 実行: onnxruntime-web。WebGPUが使えればfp16 (~215MB)、
// 使えなければWASMでint8 (~110MB)。モデルは public/david/ から配信され
// 初回選択時のみDLされる (tools/prepare_david_model.py で生成)。
// すべての処理はブラウザ内で完結し、画像を外部へ送信しない。

import { type ScalarField } from './fields';
import { cleanupDepthField, computeHeadCrop } from './portraitDepth';

const INPUT_SIZE = 512; // モデル入力 (batch, 3, 512, 512)
const MODEL_FP16_URL = '/david/david-depth-vitb16-fp16.onnx';
const MODEL_INT8_URL = '/david/david-depth-vitb16-int8.onnx';

type OrtModule = typeof import('onnxruntime-web/webgpu');

export class DavidDepthEstimator {
  private ort: OrtModule | null = null;
  private session: import('onnxruntime-web').InferenceSession | null = null;

  async init(): Promise<void> {
    if (this.session) return;
    // onnxruntime-web本体もモデル同様に遅延ロードする (メインバンドルへ含めない)。
    // 注意: vite.config の optimizeDeps.exclude が必須 — esbuildの事前バンドルは
    // wasmランタイムの相対パス解決を壊す (wasmPathsの手動上書きも
    // bundle内部ローダーと不整合を起こすため使わない)
    const ort = await import('onnxruntime-web/webgpu');
    this.ort = ort;

    const hasWebGpu = 'gpu' in navigator;
    try {
      if (!hasWebGpu) throw new Error('WebGPU未対応');
      this.session = await ort.InferenceSession.create(MODEL_FP16_URL, {
        executionProviders: ['webgpu'],
      });
      console.debug('DAViD: WebGPU + fp16 で初期化');
    } catch (err) {
      console.debug('DAViD: WebGPU初期化に失敗、WASM + int8 へフォールバック', err);
      this.session = await ort.InferenceSession.create(MODEL_INT8_URL, {
        executionProviders: ['wasm'],
      });
    }
  }

  /**
   * 人物相対Depthを推定してScalarField (画像UV対応・crop内で0-1正規化) を返す。
   * 頭部を中心に正方形cropして推論する (頭部の実効解像度を上げる。
   * DAViDは背景込みで学習されているため背景の白塗りは不要)。
   */
  async estimate(
    source: HTMLCanvasElement,
    personMask: ScalarField,
    headCenterPx: { x: number; y: number },
    faceWidthPx: number,
  ): Promise<ScalarField> {
    if (!this.session || !this.ort) throw new Error('DavidDepthEstimatorが初期化されていません。');

    const crop = computeHeadCrop(source.width, source.height, headCenterPx, faceWidthPx, 1);

    const canvas = document.createElement('canvas');
    canvas.width = INPUT_SIZE;
    canvas.height = INPUT_SIZE;
    const cctx = canvas.getContext('2d')!;
    cctx.drawImage(source, crop.x, crop.y, crop.w, crop.h, 0, 0, INPUT_SIZE, INPUT_SIZE);
    const img = cctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;

    // 公式前処理 (cv2由来): BGR順・[0,1]正規化のみ・CHW
    const area = INPUT_SIZE * INPUT_SIZE;
    const x = new Float32Array(3 * area);
    for (let i = 0; i < area; i++) {
      x[i] = img[i * 4 + 2] / 255; // B
      x[area + i] = img[i * 4 + 1] / 255; // G
      x[2 * area + i] = img[i * 4] / 255; // R
    }

    const feeds = { input: new this.ort.Tensor('float32', x, [1, 3, INPUT_SIZE, INPUT_SIZE]) };
    const outputs = await this.session.run(feeds);
    const y = outputs.output.data as Float32Array; // (1, 512, 512) inverse depth (手前が大)

    // crop内でmin-max正規化 (手前=1側)。絶対スケールは後段の
    // fitDepthToGnmZ (GNM表面zへの線形フィット) が決めるため相対で十分
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < area; i++) {
      if (y[i] < min) min = y[i];
      if (y[i] > max) max = y[i];
    }
    const span = Math.max(1e-9, max - min);
    const data = new Float32Array(area);
    for (let i = 0; i < area; i++) data[i] = (y[i] - min) / span;

    const rect = {
      u0: crop.x / source.width,
      u1: (crop.x + crop.w) / source.width,
      v0: 1 - (crop.y + crop.h) / source.height,
      v1: 1 - crop.y / source.height,
    };
    const field: ScalarField = { width: INPUT_SIZE, height: INPUT_SIZE, data, rect };

    // ARPortraitDepthと同じシルエット際halo/外れ値対策
    cleanupDepthField(field, personMask);
    return field;
  }
}
