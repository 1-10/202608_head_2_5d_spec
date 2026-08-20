// transformers.jsによる高品質ニューラル供給源 (品質比較用ティア):
// - MODNet: ポートレートアルファマット (重みApache-2.0)
//
// マットの第一候補だったBiRefNet_lite (MIT・毛先品質最上位) は2026-08時点で
// ブラウザ実行不可: ONNX入力が1024x1024固定で、WebGPUはstorage buffer 16個上限に
// 抵触し、WASMは1024²のアクティベーションでヒープ不足 (std::bad_alloc) になる。
// ランタイム側の制限解消かdynamic shape版の配布が出たら差し替える。
//
// 【ライセンス注意】重みは商用可ライセンスだが、学習データが非商用/非開示
// (MODNet: 私有データ非開示)。Google構成 (MEASURED) との品質比較・評価用。
// 推論はすべてブラウザ内 (WebGPU / WASM) で完結し、画像を外部へ送信しない。
//
// Depth側のNEURAL (Depth Anything V2) は廃止済み — Depth比較は
// ARPortraitDepth (MEASURED) と DAViD (商用クリーン) の2本で行う。

import { RawImage, pipeline, type ImageSegmentationPipeline } from '@huggingface/transformers';
import { sampleField, type ScalarField } from './fields';
import type { SegmentationResult } from './personSegmentation';

const MATTE_MODEL_ID = 'Xenova/modnet';

/** RawImage (grayscale想定) をScalarField (0-1) へ変換する。 */
function rawImageToField(image: RawImage, rect: ScalarField['rect']): ScalarField {
  const { width, height, data, channels } = image;
  const out = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    out[i] = data[i * channels] / 255;
  }
  return { width, height, data: out, rect };
}

export class NeuralMatteEstimator {
  private pipe: ImageSegmentationPipeline | null = null;

  async init(): Promise<void> {
    if (this.pipe) return;
    try {
      this.pipe = (await pipeline('image-segmentation', MATTE_MODEL_ID, {
        device: 'webgpu',
        dtype: 'fp16',
      })) as ImageSegmentationPipeline;
    } catch {
      this.pipe = (await pipeline('image-segmentation', MATTE_MODEL_ID)) as ImageSegmentationPipeline;
    }
  }

  /** 人物全体のソフトアルファマット (画像UV全体, 0-1) を返す。 */
  async estimate(source: HTMLCanvasElement): Promise<ScalarField> {
    if (!this.pipe) throw new Error('NeuralMatteEstimatorが初期化されていません。');

    const image = await RawImage.fromCanvas(source);
    const results = await this.pipe(image);
    const mask = results[0]?.mask;
    if (!mask) throw new Error('マット推定結果が空です。');
    return rawImageToField(mask, { u0: 0, v0: 0, u1: 1, v1: 1 });
  }
}

/**
 * BiRefNetの高解像度マットとMediaPipeの意味クラスを合成し、
 * 「毛先レベルのエッジ × 頭部/髪の意味分け」を持つSegmentationResultを作る。
 * - エッジ(アルファ)はBiRefNetを正とする
 * - どの領域が頭部/髪か(意味)はMediaPipeを正とする
 */
export function refineSegmentationWithMatte(seg: SegmentationResult, matte: ScalarField): SegmentationResult {
  const { width, height, rect } = matte;
  const person = new Float32Array(width * height);
  const head = new Float32Array(width * height);
  const hair = new Float32Array(width * height);
  const faceSkin = new Float32Array(width * height);

  for (let y = 0; y < height; y++) {
    const v = 1 - (y + 0.5) / height;
    for (let x = 0; x < width; x++) {
      const u = (x + 0.5) / width;
      const i = y * width + x;
      const a = matte.data[i];
      // 意味重みはMediaPipe側をやや広め(smoothstep)に取り、エッジ形状はマットに任せる
      person[i] = a * smoothWeight(sampleField(seg.person, u, v));
      head[i] = a * smoothWeight(sampleField(seg.head, u, v));
      hair[i] = a * smoothWeight(sampleField(seg.hair, u, v));
      faceSkin[i] = a * smoothWeight(sampleField(seg.faceSkin, u, v));
    }
  }

  return {
    person: { width, height, data: person, rect },
    head: { width, height, data: head, rect },
    hair: { width, height, data: hair, rect },
    faceSkin: { width, height, data: faceSkin, rect },
  };
}

function smoothWeight(x: number): number {
  const t = Math.min(1, Math.max(0, (x - 0.15) / (0.55 - 0.15)));
  return t * t * (3 - 2 * t);
}
