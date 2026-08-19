// MediaPipe Image Segmenter (SelfieMulticlass 256x256) による実測セグメンテーション。
// 楕円近似を置き換える「実シルエット・実髪マスク」の供給源。
// モデルはGoogle自社収集データで学習・Apache-2.0 (学習データまで商用クリーン)。
// すべての処理はブラウザ内で完結し、画像を外部へ送信しない。

import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';
import { fullImageRect, type ScalarField } from './fields';
import type { NormalizedFaceLandmark } from './faceTopology';
import { FACE_KEY_INDICES } from './faceTopology';

const WASM_BASE_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17/wasm';
const MULTICLASS_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite';

// SelfieMulticlassの出力クラスindex
const CLASS_BACKGROUND = 0;
const CLASS_HAIR = 1;
const CLASS_BODY_SKIN = 2;
const CLASS_FACE_SKIN = 3;
// 4 = clothes (頭部に含めない)
const CLASS_ACCESSORIES = 5; // 帽子・メガネ等。頭部シルエットに含める

export interface SegmentationResult {
  person: ScalarField; // 1 - background
  hair: ScalarField;
  faceSkin: ScalarField;
  /** 頭部シルエット: hair + faceSkin + accessories + (顎より上のbodySkin=耳など) */
  head: ScalarField;
}

export class PersonSegmenter {
  private segmenter: ImageSegmenter | null = null;

  async init(): Promise<void> {
    if (this.segmenter) return;
    const vision = await FilesetResolver.forVisionTasks(WASM_BASE_URL);
    const baseConfig = {
      runningMode: 'IMAGE' as const,
      outputConfidenceMasks: true,
      outputCategoryMask: false,
    };
    try {
      this.segmenter = await ImageSegmenter.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MULTICLASS_MODEL_URL, delegate: 'GPU' },
        ...baseConfig,
      });
    } catch {
      this.segmenter = await ImageSegmenter.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MULTICLASS_MODEL_URL, delegate: 'CPU' },
        ...baseConfig,
      });
    }
  }

  /**
   * 静止画をセグメントし、頭部関連のScalarFieldを返す。
   * chinV: 顎下端の画像v座標 (0-1, 上が1)。bodySkin(首・耳含む)のうち顎より上だけを頭部へ含める。
   */
  segment(source: HTMLCanvasElement, landmarks: NormalizedFaceLandmark[]): SegmentationResult {
    if (!this.segmenter) throw new Error('PersonSegmenterが初期化されていません。');
    const result = this.segmenter.segment(source);
    const masks = result.confidenceMasks;
    if (!masks || masks.length <= CLASS_ACCESSORIES) {
      result.close();
      throw new Error('セグメンテーション結果が不正です。');
    }

    const width = masks[0].width;
    const height = masks[0].height;
    const bg = Float32Array.from(masks[CLASS_BACKGROUND].getAsFloat32Array());
    const hair = Float32Array.from(masks[CLASS_HAIR].getAsFloat32Array());
    const bodySkin = Float32Array.from(masks[CLASS_BODY_SKIN].getAsFloat32Array());
    const faceSkin = Float32Array.from(masks[CLASS_FACE_SKIN].getAsFloat32Array());
    const accessories = Float32Array.from(masks[CLASS_ACCESSORIES].getAsFloat32Array());
    result.close();

    const person = new Float32Array(width * height);
    for (let i = 0; i < person.length; i++) person[i] = clamp01(1 - bg[i]);

    // 顎下端: landmark chin の v。少し下へ余裕を持たせ、そこから下のbodySkin(首)を頭部から除外する。
    const chinV = landmarks[FACE_KEY_INDICES.chin].v;
    const chinRow = (1 - chinV) * (height - 1); // row換算 (row 0=画像上端)
    const fadeRows = height * 0.04;

    const head = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
      // 顎より上=1、顎から下へfadeRowsかけて0
      const aboveChin = clamp01(1 - (y - chinRow) / Math.max(1, fadeRows));
      for (let x = 0; x < width; x++) {
        const i = y * width + x;
        // bodySkin(首)とaccessories(服の一部を誤って拾うことがある)は顎より上に限定する。
        // 帽子・メガネは顎より上なので実害はない。髪は顎下(ロングヘア)も含める。
        const h = hair[i] + (faceSkin[i] + accessories[i] + bodySkin[i]) * aboveChin;
        // personマスクとの積で背景誤検出を抑える
        head[i] = clamp01(h) * person[i];
      }
    }

    const rect = fullImageRect();
    return {
      person: { width, height, data: person, rect },
      hair: { width, height, data: hair, rect },
      faceSkin: { width, height, data: faceSkin, rect },
      head: { width, height, data: head, rect },
    };
  }
}

function clamp01(x: number): number {
  return Math.min(1, Math.max(0, x));
}

/**
 * ScalarFieldマスクを入力画像UV空間のalphaMap用Canvasへラスタライズする。
 * bilinearサンプル+軽いblurで、256pxマスクのブロック感を抑えたsoft featherを作る。
 */
export function rasterizeMaskCanvas(field: ScalarField, size = 512, blurPx = 2): HTMLCanvasElement {
  const raw = document.createElement('canvas');
  raw.width = field.width;
  raw.height = field.height;
  const rawCtx = raw.getContext('2d')!;
  const imageData = rawCtx.createImageData(field.width, field.height);
  for (let i = 0; i < field.data.length; i++) {
    const g = Math.round(Math.min(1, Math.max(0, field.data[i])) * 255);
    imageData.data[i * 4] = g;
    imageData.data[i * 4 + 1] = g;
    imageData.data[i * 4 + 2] = g;
    imageData.data[i * 4 + 3] = 255;
  }
  rawCtx.putImageData(imageData, 0, 0);

  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  ctx.filter = `blur(${blurPx}px)`;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(raw, 0, 0, size, size);
  return canvas;
}
