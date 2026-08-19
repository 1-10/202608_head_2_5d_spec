// TF.js ARPortraitDepth による実測人物Depth。
// GNM髪シェルの厚み推定に使う「計測された頭部凹凸」の供給源。
// モデルはGoogle自前収集データで学習・Apache-2.0 (学習データまで商用クリーン)。
// すべての処理はブラウザ内 (WebGL) で完結し、画像を外部へ送信しない。

import '@tensorflow/tfjs-backend-webgl';
import * as depthEstimation from '@tensorflow-models/depth-estimation';
import { sampleField, type ScalarField } from './fields';

// ARPortraitDepthの入力解像度 (h x w = 256 x 192)。crop比率をこれに合わせる。
const MODEL_ASPECT = 192 / 256; // w / h

export class PortraitDepthEstimator {
  private estimator: depthEstimation.DepthEstimator | null = null;

  async init(): Promise<void> {
    if (this.estimator) return;
    this.estimator = await depthEstimation.createEstimator(depthEstimation.SupportedModels.ARPortraitDepth);
  }

  /**
   * 人物Depthを推定してScalarField (画像UV対応・値は相対Depth 0-1) を返す。
   * - personMaskで背景を白へ置換してから推論する (公式パイプラインと同じ前処理)
   * - 頭部を中心にモデルのアスペクト比でcropし、頭部の実効解像度を上げる
   */
  async estimate(
    source: HTMLCanvasElement,
    personMask: ScalarField,
    headCenterPx: { x: number; y: number },
    faceWidthPx: number,
  ): Promise<ScalarField> {
    if (!this.estimator) throw new Error('PortraitDepthEstimatorが初期化されていません。');

    const crop = computeHeadCrop(source.width, source.height, headCenterPx, faceWidthPx);

    // crop領域を切り出し、背景をpersonMaskで白へ
    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = crop.w;
    cropCanvas.height = crop.h;
    const ctx = cropCanvas.getContext('2d')!;
    ctx.drawImage(source, crop.x, crop.y, crop.w, crop.h, 0, 0, crop.w, crop.h);
    const imgData = ctx.getImageData(0, 0, crop.w, crop.h);
    for (let y = 0; y < crop.h; y++) {
      const v = 1 - (crop.y + y + 0.5) / source.height;
      for (let x = 0; x < crop.w; x++) {
        const u = (crop.x + x + 0.5) / source.width;
        const m = sampleField(personMask, u, v);
        if (m < 0.5) {
          const i = (y * crop.w + x) * 4;
          imgData.data[i] = 255;
          imgData.data[i + 1] = 255;
          imgData.data[i + 2] = 255;
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);

    const depthMap = await this.estimator.estimateDepth(cropCanvas, { minDepth: 0, maxDepth: 1 });
    const rows = await depthMap.toArray();
    const height = rows.length;
    const width = rows[0]?.length ?? 0;
    if (width === 0) throw new Error('Depth推定結果が空です。');

    const data = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) data[y * width + x] = rows[y][x];
    }

    const rect = {
      u0: crop.x / source.width,
      u1: (crop.x + crop.w) / source.width,
      v0: 1 - (crop.y + crop.h) / source.height,
      v1: 1 - crop.y / source.height,
    };

    // Depthのクリーンアップ (シルエット際のhalo/外れ値対策の定石):
    // 1. 前景コアをerode(境界2pxの混合画素を信頼しない) 2. コアのパーセンタイルでclamp
    // 3. コアからBFS dilate(境界・背景をコアのDepthで塗る) 4. 平滑化
    cleanupDepthField({ width, height, data, rect }, personMask);

    return { width, height, data, rect };
  }
}

const CORE_ERODE_ITERATIONS = 2; // 境界の混合画素(halo)を信頼しない幅
const SMOOTH_PASSES = 2; // 3x3 box blurの回数

/** Depth場のシルエット際halo/外れ値対策 (erode→clamp→前景dilation→平滑化)。他のDepth供給源からも使う。 */
export function cleanupDepthField(depth: ScalarField, personMask: ScalarField): void {
  const { width, height, data, rect } = depth;
  const total = width * height;

  // 前景フラグ (personMask >= 0.5)
  let fg = new Uint8Array(total);
  let fgCount = 0;
  for (let y = 0; y < height; y++) {
    const v = rect.v1 - ((y + 0.5) / height) * (rect.v1 - rect.v0);
    for (let x = 0; x < width; x++) {
      const u = rect.u0 + ((x + 0.5) / width) * (rect.u1 - rect.u0);
      if (sampleField(personMask, u, v) >= 0.5) {
        fg[y * width + x] = 1;
        fgCount++;
      }
    }
  }
  if (fgCount === 0) return;

  // erodeして「コア」(境界halo画素を除いた信頼できる前景)を得る
  for (let it = 0; it < CORE_ERODE_ITERATIONS; it++) {
    const next = new Uint8Array(fg);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = y * width + x;
        if (!fg[i]) continue;
        const edge =
          x === 0 ||
          y === 0 ||
          x === width - 1 ||
          y === height - 1 ||
          !fg[i - 1] ||
          !fg[i + 1] ||
          !fg[i - width] ||
          !fg[i + width];
        if (edge) next[i] = 0;
      }
    }
    // erodeし尽くして空になるならその前の状態を使う
    let count = 0;
    for (let i = 0; i < total; i++) count += next[i];
    if (count === 0) break;
    fg = next;
  }

  // コアのDepth分布のパーセンタイルで全体をclamp (外れ値スパイク除去)
  const coreValues: number[] = [];
  for (let i = 0; i < total; i++) if (fg[i]) coreValues.push(data[i]);
  coreValues.sort((a, b) => a - b);
  const lo = coreValues[Math.floor(coreValues.length * 0.02)];
  const hi = coreValues[Math.min(coreValues.length - 1, Math.floor(coreValues.length * 0.98))];
  for (let i = 0; i < total; i++) data[i] = Math.min(hi, Math.max(lo, data[i]));

  // コアからBFSで外側(halo+背景)へDepthを伝播
  const known = new Uint8Array(fg);
  const queue = new Int32Array(total);
  let head = 0;
  let tail = 0;
  for (let i = 0; i < total; i++) if (known[i]) queue[tail++] = i;
  while (head < tail) {
    const i = queue[head++];
    const x = i % width;
    const y = (i / width) | 0;
    const z = data[i];
    if (x > 0 && !known[i - 1]) {
      known[i - 1] = 1;
      data[i - 1] = z;
      queue[tail++] = i - 1;
    }
    if (x < width - 1 && !known[i + 1]) {
      known[i + 1] = 1;
      data[i + 1] = z;
      queue[tail++] = i + 1;
    }
    if (y > 0 && !known[i - width]) {
      known[i - width] = 1;
      data[i - width] = z;
      queue[tail++] = i - width;
    }
    if (y < height - 1 && !known[i + width]) {
      known[i + width] = 1;
      data[i + width] = z;
      queue[tail++] = i + width;
    }
  }

  // 3x3 box blurで残るノイズを平滑化
  for (let p = 0; p < SMOOTH_PASSES; p++) {
    const src = data.slice();
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = 0;
        let n = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const yy = y + dy;
          if (yy < 0 || yy >= height) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx;
            if (xx < 0 || xx >= width) continue;
            sum += src[yy * width + xx];
            n++;
          }
        }
        data[y * width + x] = sum / n;
      }
    }
  }
}

/** 頭部中心に指定アスペクト比(w/h)のcrop矩形を取る (顔幅の約3.2倍を横幅目安)。 */
export function computeHeadCrop(
  imageWidth: number,
  imageHeight: number,
  headCenterPx: { x: number; y: number },
  faceWidthPx: number,
  aspect: number = MODEL_ASPECT,
): { x: number; y: number; w: number; h: number } {
  let w = Math.min(imageWidth, faceWidthPx * 3.2);
  let h = w / aspect;
  if (h > imageHeight) {
    h = imageHeight;
    w = h * aspect;
  }
  // 頭頂側に多め(0.55)・顎下側に少なめの余白で配置
  let x = headCenterPx.x - w / 2;
  let y = headCenterPx.y - h * 0.55;
  x = Math.min(Math.max(0, x), imageWidth - w);
  y = Math.min(Math.max(0, y), imageHeight - h);
  return { x: Math.round(x), y: Math.round(y), w: Math.round(w), h: Math.round(h) };
}
