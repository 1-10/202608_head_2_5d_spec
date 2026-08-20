// DAViD (Microsoft, ICCV 2025) による人物特化の単眼推定 (相対Depth / 表面法線)。
// 100%合成データ (SynthHuman, CDLA-Permissive-2.0) 学習 + モデルMIT =
// 「学習データまで商用クリーン」を満たす。
// https://github.com/microsoft/DAViD
//
// 実行: onnxruntime-web。WebGPUが使えればfp16 (~215MB/タスク)、
// 使えなければWASMでint8 (~110MB/タスク)。モデルは public/david/ から配信され
// 初回選択時のみDLされる (tools/prepare_david_model.py で生成)。
// すべての処理はブラウザ内で完結し、画像を外部へ送信しない。

import { type ScalarField } from './fields';
import { cleanupDepthField, computeHeadCrop } from './portraitDepth';

const INPUT_SIZE = 512; // モデル入力 (batch, 3, 512, 512)
const NORMAL_CANVAS_MAX_DIM = 1024; // 法線マップcanvas (画像全体空間) の解像度上限

type OrtModule = typeof import('onnxruntime-web/webgpu');
type OrtSession = import('onnxruntime-web').InferenceSession;

let ortModule: OrtModule | null = null;

/**
 * DAViDのタスク別セッションを作る。WebGPU+fp16 → WASM+int8 の順に試す。
 * 注意: vite.config の optimizeDeps.exclude が必須 — esbuildの事前バンドルは
 * wasmランタイムの相対パス解決を壊す (wasmPathsの手動上書きも
 * bundle内部ローダーと不整合を起こすため使わない)
 */
async function createDavidSession(task: 'depth' | 'normal'): Promise<{ ort: OrtModule; session: OrtSession }> {
  const ort = (ortModule ??= await import('onnxruntime-web/webgpu'));
  const hasWebGpu = 'gpu' in navigator;
  try {
    if (!hasWebGpu) throw new Error('WebGPU未対応');
    const session = await ort.InferenceSession.create(`/david/david-${task}-vitb16-fp16.onnx`, {
      executionProviders: ['webgpu'],
    });
    console.debug(`DAViD(${task}): WebGPU + fp16 で初期化`);
    return { ort, session };
  } catch (err) {
    console.debug(`DAViD(${task}): WebGPU初期化に失敗、WASM + int8 へフォールバック`, err);
    const session = await ort.InferenceSession.create(`/david/david-${task}-vitb16-int8.onnx`, {
      executionProviders: ['wasm'],
    });
    return { ort, session };
  }
}

/** 頭部正方形cropを512x512へ描き、公式前処理 (BGR順・[0,1]・CHW) のテンソルデータを返す。 */
function preprocessCrop(
  source: HTMLCanvasElement,
  crop: { x: number; y: number; w: number; h: number },
): Float32Array {
  const canvas = document.createElement('canvas');
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;
  const cctx = canvas.getContext('2d')!;
  cctx.drawImage(source, crop.x, crop.y, crop.w, crop.h, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const img = cctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;

  const area = INPUT_SIZE * INPUT_SIZE;
  const x = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    x[i] = img[i * 4 + 2] / 255; // B
    x[area + i] = img[i * 4 + 1] / 255; // G
    x[2 * area + i] = img[i * 4] / 255; // R
  }
  return x;
}

export class DavidDepthEstimator {
  private ort: OrtModule | null = null;
  private session: OrtSession | null = null;

  async init(): Promise<void> {
    if (this.session) return;
    const { ort, session } = await createDavidSession('depth');
    this.ort = ort;
    this.session = session;
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
    const x = preprocessCrop(source, crop);
    const feeds = { input: new this.ort.Tensor('float32', x, [1, 3, INPUT_SIZE, INPUT_SIZE]) };
    const outputs = await this.session.run(feeds);
    const y = outputs.output.data as Float32Array; // (1, 512, 512) inverse depth (手前が大)

    // crop内でmin-max正規化 (手前=1側)。絶対スケールは後段の
    // fitDepthToGnmZ (GNM表面zへの線形フィット) が決めるため相対で十分
    const area = INPUT_SIZE * INPUT_SIZE;
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

export class DavidNormalEstimator {
  private ort: OrtModule | null = null;
  private session: OrtSession | null = null;

  async init(): Promise<void> {
    if (this.session) return;
    const { ort, session } = await createDavidSession('normal');
    this.ort = ort;
    this.session = session;
  }

  /**
   * 表面法線を推定し、three.jsのObjectSpaceNormalMapとして貼れるRGBエンコード
   * (rgb = (n+1)/2) のcanvasを「画像全体のUV空間」で返す。crop外は+Z (平坦)。
   *
   * 座標変換 (male実測で確認したDAViDの規約 → three.js object空間):
   * DAViDはX=画像左が正・Y=上が正・Z=カメラ向きが正。three.jsはX=右が正なので
   * Xだけ反転する。headのモデル空間は正面写真と同じ向きなので、
   * カメラ空間の法線をそのままobject空間として使える。
   */
  async estimate(
    source: HTMLCanvasElement,
    headCenterPx: { x: number; y: number },
    faceWidthPx: number,
  ): Promise<HTMLCanvasElement> {
    if (!this.session || !this.ort) throw new Error('DavidNormalEstimatorが初期化されていません。');

    const crop = computeHeadCrop(source.width, source.height, headCenterPx, faceWidthPx, 1);
    const x = preprocessCrop(source, crop);
    const feeds = { input: new this.ort.Tensor('float32', x, [1, 3, INPUT_SIZE, INPUT_SIZE]) };
    const outputs = await this.session.run(feeds);
    const y = outputs.output.data as Float32Array; // (1, 3, 512, 512) camera-space normal

    // crop領域の法線をRGBエンコード
    const area = INPUT_SIZE * INPUT_SIZE;
    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = INPUT_SIZE;
    cropCanvas.height = INPUT_SIZE;
    const cc = cropCanvas.getContext('2d')!;
    const imgData = cc.createImageData(INPUT_SIZE, INPUT_SIZE);
    for (let i = 0; i < area; i++) {
      const nx = -y[i]; // X反転 (DAViDは画像左が正)
      const ny = y[area + i];
      const nz = y[2 * area + i];
      const len = Math.max(1e-6, Math.hypot(nx, ny, nz));
      imgData.data[i * 4] = Math.round(((nx / len) * 0.5 + 0.5) * 255);
      imgData.data[i * 4 + 1] = Math.round(((ny / len) * 0.5 + 0.5) * 255);
      imgData.data[i * 4 + 2] = Math.round(((nz / len) * 0.5 + 0.5) * 255);
      imgData.data[i * 4 + 3] = 255;
    }
    cc.putImageData(imgData, 0, 0);

    // 画像全体空間のcanvasへ配置 (crop外は+Z=平坦)。headのUVは画像UVのため
    const scale = Math.min(1, NORMAL_CANVAS_MAX_DIM / Math.max(source.width, source.height));
    const w = Math.max(2, Math.round(source.width * scale));
    const h = Math.max(2, Math.round(source.height * scale));
    const full = document.createElement('canvas');
    full.width = w;
    full.height = h;
    const fc = full.getContext('2d')!;
    fc.fillStyle = 'rgb(128,128,255)';
    fc.fillRect(0, 0, w, h);
    fc.imageSmoothingEnabled = true;
    fc.imageSmoothingQuality = 'high';
    fc.drawImage(
      cropCanvas,
      (crop.x / source.width) * w,
      (crop.y / source.height) * h,
      (crop.w / source.width) * w,
      (crop.h / source.height) * h,
    );
    return full;
  }
}
