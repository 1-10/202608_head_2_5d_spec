// DAViD (Microsoft, ICCV 2025) による人物特化の単眼推定。
// multi-taskモデル (ViT-L) が1回の推論で 相対Depth / 表面法線 / ソフト前景 を
// 同時に返す。100%合成データ (SynthHuman, CDLA-Permissive-2.0) 学習 +
// モデルMIT = 「学習データまで商用クリーン」を満たす。
// https://github.com/microsoft/DAViD
//
// 実行: onnxruntime-web。WebGPUが使えればfp16、無ければWASMでint8。
// モデル本体 (fp16 691MB / int8 338MB) はリポジトリに含めず、Hugging Face Hub
// (公開モデルリポジトリ、無料枠のCDN配信+CORS許可) から初回選択時のみDLされる
// (tools/prepare_david_model.py で生成したファイルを `hf upload` でアップロード済み)。
// すべての処理はブラウザ内で完結し、画像を外部へ送信しない。

import { fullImageRect, sampleField, type ScalarField } from './fields';
import { cleanupDepthField, computeHeadCrop } from './portraitDepth';

const INPUT_SIZE = 512; // モデル入力 (batch, 3, 512, 512)
const NORMAL_CANVAS_MAX_DIM = 1024; // 法線マップcanvas (画像全体空間) の解像度上限
const PERSON_MERGE_MAX_DIM = 512; // 前景マスクの全体マージ解像度上限

const HF_MODEL_BASE = 'https://huggingface.co/harry00902/202608_head_2_5d_spec/resolve/main/david';
const MODEL_FP16_URL = `${HF_MODEL_BASE}/david-multitask-vitl16-fp16.onnx`;
const MODEL_INT8_URL = `${HF_MODEL_BASE}/david-multitask-vitl16-int8.onnx`;

// DAViD出力はARPortraitDepthよりノイズが格段に少ないため、cleanupは弱く掛ける
// (強いままだと鼻先・耳などレンジ端の実起伏までパーセンタイルclampで丸まる)
const DAVID_DEPTH_CLEANUP = {
  erodeIterations: 1,
  clampLoPct: 0.005,
  clampHiPct: 0.995,
  smoothPasses: 1,
};

type OrtModule = typeof import('onnxruntime-web/webgpu');
type OrtSession = import('onnxruntime-web').InferenceSession;

export interface DavidResult {
  /** 相対Depth (crop内0-1正規化, cleanup済み)。 */
  depth: ScalarField;
  /** ObjectSpaceNormalMapとして貼れるRGBエンコード法線 (画像全体UV空間)。 */
  normalCanvas: HTMLCanvasElement;
  /** ソフト前景マスク (画像全体UV空間。crop外はfallback personで補完済み)。 */
  person: ScalarField;
}

export class DavidEstimator {
  private ort: OrtModule | null = null;
  private session: OrtSession | null = null;

  /**
   * multi-taskセッションを作る。WebGPU+fp16 → WASM+int8 の順に試す。
   * 注意: vite.config の optimizeDeps.exclude が必須 — esbuildの事前バンドルは
   * wasmランタイムの相対パス解決を壊す (wasmPathsの手動上書きも
   * bundle内部ローダーと不整合を起こすため使わない)
   */
  async init(): Promise<void> {
    if (this.session) return;
    const ort = (this.ort = await import('onnxruntime-web/webgpu'));
    const hasWebGpu = 'gpu' in navigator;
    try {
      if (!hasWebGpu) throw new Error('WebGPU未対応');
      this.session = await ort.InferenceSession.create(MODEL_FP16_URL, {
        executionProviders: ['webgpu'],
      });
      console.debug('DAViD(multitask): WebGPU + fp16 で初期化');
    } catch (err) {
      console.debug('DAViD(multitask): WebGPU初期化に失敗、WASM + int8 へフォールバック', err);
      this.session = await ort.InferenceSession.create(MODEL_INT8_URL, {
        executionProviders: ['wasm'],
      });
    }
  }

  /**
   * 頭部正方形cropに対して1回推論し、Depth/法線/前景を同時に返す。
   * fallbackPerson: crop外の前景補完に使うマスク (SelfieMulticlassのperson。
   * null なら crop外=背景扱い)。DepthのcleanupにはDAViD自身の前景を使う
   * (境界ソースが揃い、縁取りの二重化を防ぐ)。
   */
  async estimate(
    source: HTMLCanvasElement,
    headCenterPx: { x: number; y: number },
    faceWidthPx: number,
    fallbackPerson: ScalarField | null,
  ): Promise<DavidResult> {
    if (!this.session || !this.ort) throw new Error('DavidEstimatorが初期化されていません。');

    const crop = computeHeadCrop(source.width, source.height, headCenterPx, faceWidthPx, 1);
    const rect = {
      u0: crop.x / source.width,
      u1: (crop.x + crop.w) / source.width,
      v0: 1 - (crop.y + crop.h) / source.height,
      v1: 1 - crop.y / source.height,
    };

    // --- 前処理 (公式準拠: BGR順・[0,1]・CHW) ---
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

    // --- 推論 (出力順は公式ランタイム準拠: [depth, normal, foreground]) ---
    const feeds = { [this.session.inputNames[0]]: new this.ort.Tensor('float32', x, [1, 3, INPUT_SIZE, INPUT_SIZE]) };
    const outputs = await this.session.run(feeds);
    const names = this.session.outputNames;
    const depthRaw = outputs[names[0]].data as Float32Array; // (1, 512, 512)
    const normalRaw = outputs[names[1]].data as Float32Array; // (1, 3, 512, 512)
    const fgRaw = outputs[names[2]].data as Float32Array; // (1, 1, 512, 512)

    // --- 前景: crop内=DAViD、crop外=fallback person で全体マージ ---
    const fgCrop: ScalarField = {
      width: INPUT_SIZE,
      height: INPUT_SIZE,
      data: Float32Array.from(fgRaw.subarray(0, area), (v) => Math.min(1, Math.max(0, v))),
      rect,
    };
    const person = mergePersonWithForeground(fgCrop, fallbackPerson, source.width, source.height);

    // --- Depth: crop内min-max正規化 → cleanup (personはDAViD前景を使う) ---
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < area; i++) {
      if (depthRaw[i] < min) min = depthRaw[i];
      if (depthRaw[i] > max) max = depthRaw[i];
    }
    const span = Math.max(1e-9, max - min);
    const depthData = new Float32Array(area);
    for (let i = 0; i < area; i++) depthData[i] = (depthRaw[i] - min) / span;
    const depth: ScalarField = { width: INPUT_SIZE, height: INPUT_SIZE, data: depthData, rect };
    cleanupDepthField(depth, fgCrop, DAVID_DEPTH_CLEANUP);

    // --- 法線: three.js object空間RGBエンコード (画像全体UV空間, crop外=+Z) ---
    const normalCanvas = encodeNormalCanvas(normalRaw, crop, source.width, source.height);

    return { depth, normalCanvas, person };
  }
}

/**
 * DAViD前景 (crop領域) とfallback person (画像全体) をマージした
 * 画像全体UV空間のマスクを作る。crop縁の帯 (UV 0.02) でクロスフェードする。
 */
function mergePersonWithForeground(
  fg: ScalarField,
  fallbackPerson: ScalarField | null,
  imageWidth: number,
  imageHeight: number,
): ScalarField {
  const scale = Math.min(1, PERSON_MERGE_MAX_DIM / Math.max(imageWidth, imageHeight));
  const w = Math.max(2, Math.round(imageWidth * scale));
  const h = Math.max(2, Math.round(imageHeight * scale));
  const data = new Float32Array(w * h);
  const { u0, v0, u1, v1 } = fg.rect;
  const FADE = 0.02;
  for (let y = 0; y < h; y++) {
    const v = 1 - (y + 0.5) / h;
    for (let x = 0; x < w; x++) {
      const u = (x + 0.5) / w;
      // crop縁からの内側距離 (UV) → 0..1 のブレンド重み
      const edge = Math.min(u - u0, u1 - u, v - v0, v1 - v);
      const wFg = Math.min(1, Math.max(0, edge / FADE));
      const pFallback = fallbackPerson ? sampleField(fallbackPerson, u, v) : 0;
      data[y * w + x] = wFg > 0 ? sampleField(fg, u, v) * wFg + pFallback * (1 - wFg) : pFallback;
    }
  }
  return { width: w, height: h, data, rect: fullImageRect() };
}

/**
 * 法線 (CHW, カメラ空間) をthree.jsのObjectSpaceNormalMapとして貼れる
 * RGBエンコードcanvas (画像全体UV空間, crop外=+Z平坦) へ変換する。
 *
 * 座標変換 (male実測で確認したDAViDの規約 → three.js object空間):
 * DAViDはX=画像左が正・Y=上が正・Z=カメラ向きが正。three.jsはX=右が正なので
 * Xだけ反転する。headのモデル空間は正面写真と同じ向きなので、
 * カメラ空間の法線をそのままobject空間として使える。
 */
function encodeNormalCanvas(
  normalRaw: Float32Array,
  crop: { x: number; y: number; w: number; h: number },
  imageWidth: number,
  imageHeight: number,
): HTMLCanvasElement {
  const area = INPUT_SIZE * INPUT_SIZE;
  const cropCanvas = document.createElement('canvas');
  cropCanvas.width = INPUT_SIZE;
  cropCanvas.height = INPUT_SIZE;
  const cc = cropCanvas.getContext('2d')!;
  const imgData = cc.createImageData(INPUT_SIZE, INPUT_SIZE);
  for (let i = 0; i < area; i++) {
    const nx = -normalRaw[i]; // X反転 (DAViDは画像左が正)
    const ny = normalRaw[area + i];
    const nz = normalRaw[2 * area + i];
    const len = Math.max(1e-6, Math.hypot(nx, ny, nz));
    imgData.data[i * 4] = Math.round(((nx / len) * 0.5 + 0.5) * 255);
    imgData.data[i * 4 + 1] = Math.round(((ny / len) * 0.5 + 0.5) * 255);
    imgData.data[i * 4 + 2] = Math.round(((nz / len) * 0.5 + 0.5) * 255);
    imgData.data[i * 4 + 3] = 255;
  }
  cc.putImageData(imgData, 0, 0);

  const scale = Math.min(1, NORMAL_CANVAS_MAX_DIM / Math.max(imageWidth, imageHeight));
  const w = Math.max(2, Math.round(imageWidth * scale));
  const h = Math.max(2, Math.round(imageHeight * scale));
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
    (crop.x / imageWidth) * w,
    (crop.y / imageHeight) * h,
    (crop.w / imageWidth) * w,
    (crop.h / imageHeight) * h,
  );
  return full;
}
