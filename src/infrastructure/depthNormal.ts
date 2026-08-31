// 深度・法線推定のアダプタ。
//
// DAViD（Microsoft, ICCV 2025）の multi-task モデルを ONNX Runtime Web で走らせ、深度・法線・人物
// 前景を `domain/field.DepthNormalResult` へ変換する。100% 合成データ（SynthHuman,
// CDLA-Permissive-2.0）学習 + モデル MIT = 「学習データまで商用クリーン」。
//
// **どこを切るかは決めない** — 正方領域を引数で受け取る。決め方は `domain/gnm/crop`。
//
// モデルのグラフ仕様（実測値。これが正本）:
//
//     入力  "input"  (batch, 3, 512, 512) float32
//     出力  3 本 — 名前は自動生成なので**名前で決め打ちしない**（`classifyOutputs`）
//           (batch, 512, 512)     相対深度
//           (batch, 3, 512, 512)  表面法線
//           (batch, 1, 512, 512)  人物前景（ソフト）
//
// 入力解像度はファイル名が `_384` でもグラフは 512 固定。
//
// web だから変わるところ
// ----------------------
// デスクトップ側は CUDA EP で走らせ、取れなければ**落とす**（CPU は実測 150 倍遅い）。ブラウザは
// 実行環境を利用者が選べないので、**WebGPU + fp16 → WASM + int8 の順に落ちる**。どちらで動いているかは
// `provider` で読めるようにして、黙って遅くならないよう画面へ出す。
//
// モデル本体（fp16 691MB / int8 338MB）はリポジトリに含めず、Hugging Face Hub の公開モデルリポジトリ
// から配信する。**デスクトップ側は配布元 URL から落として `onnxconverter_common` で fp16 へ変換する**
// （`tools/prepare_david_model.py` が同じ変換を行い、その出力を HF Hub へ上げてある）。
//
// **前処理は RGB の [0,1] のみ**（ImageNet 系の mean/std は引かない）。デスクトップ側が実測で決めた
// 規約で、人物前景の出力を MediaPipe SelfieMulticlass の前景と比べると [0,1] が IoU 0.983〜0.986、
// ImageNet 正規化は 0.43〜0.75 だった。

import type { InferenceSession } from 'onnxruntime-web';
import { DepthNormalEstimator } from '../application/ports';
import { GpuUnavailableError, ModelFileNotFoundError } from '../domain/errors';
import { DepthNormalResult, makeField, rectFromPixels, validateDepthNormal } from '../domain/field';
import { PhotoRgb } from '../domain/photo';
import { cropSquareToRgb } from './photoCanvas';

/** グラフが固定している入力の一辺。ファイル名の `_384` はグラフと一致しない。 */
export const INPUT_RESOLUTION = 512;

const NORMAL_CHANNELS = 3;
const FOREGROUND_CHANNELS = 1;

const HF_MODEL_BASE =
  'https://huggingface.co/harry00902/202608_head_2_5d_spec/resolve/main/david';
const MODEL_FP16_URL = `${HF_MODEL_BASE}/david-multitask-vitl16-fp16.onnx`;
const MODEL_INT8_URL = `${HF_MODEL_BASE}/david-multitask-vitl16-int8.onnx`;

/**
 * 出力 3 本を形で depth / normal / foreground に振り分ける。
 *
 * 名前で決め打ちしないのは、法線と前景の出力名が自動生成された数値（`2929` / `2917`）で、モデルを
 * 差し替えると黙って変わるため。形は意味に結びついているので、名前より壊れにくい。
 *
 * 判別（バッチ次元を含む形で見る。推論は常に batch=1 で走らせる）:
 *     3 次元       → 深度
 *     4 次元 3ch   → 法線
 *     4 次元 1ch   → 前景
 */
export function classifyOutputs(
  outputs: readonly { name: string; dims: readonly number[]; data: Float32Array }[],
): { depth: Float32Array; normal: Float32Array; foreground: Float32Array } {
  const found = new Map<string, Float32Array>();
  for (const output of outputs) {
    let kind: string;
    if (output.dims.length === 3) kind = 'depth';
    else if (output.dims.length === 4 && output.dims[1] === NORMAL_CHANNELS) kind = 'normal';
    else if (output.dims.length === 4 && output.dims[1] === FOREGROUND_CHANNELS) kind = 'foreground';
    else throw new Error(`想定外の出力 ${output.name}: dims=${output.dims.join('x')}`);
    if (found.has(kind)) {
      throw new Error(`${kind} に相当する出力が 2 本ある（${output.name} で重複）`);
    }
    found.set(kind, output.data);
  }
  const missing = ['depth', 'normal', 'foreground'].filter((kind) => !found.has(kind));
  if (missing.length > 0) {
    throw new Error(
      `出力が足りない: ${missing.join(', ')}` +
        `（受け取った形: ${outputs.map((o) => `${o.name}=${o.dims.join('x')}`).join(', ')}）`,
    );
  }
  return {
    depth: found.get('depth') as Float32Array,
    normal: found.get('normal') as Float32Array,
    foreground: found.get('foreground') as Float32Array,
  };
}

export class DavidDepthNormalEstimator implements DepthNormalEstimator {
  private session: InferenceSession | null = null;
  private ort: typeof import('onnxruntime-web/webgpu') | null = null;
  /** どの実行環境で動いているか（画面に出して、黙って遅くならないようにする）。 */
  provider: 'webgpu-fp16' | 'wasm-int8' | null = null;

  /**
   * セッションを作る。WebGPU + fp16 → WASM + int8 の順に試す。
   *
   * `vite.config` の `optimizeDeps.exclude` が必須 — esbuild の事前バンドルは wasm ランタイムの
   * 相対パス解決を壊す。
   */
  async init(): Promise<void> {
    if (this.session !== null) return;
    const ort = (this.ort = await import('onnxruntime-web/webgpu'));
    const hasWebGpu = 'gpu' in navigator;
    try {
      if (!hasWebGpu) throw new GpuUnavailableError('WebGPU 未対応');
      this.session = await ort.InferenceSession.create(MODEL_FP16_URL, {
        executionProviders: ['webgpu'],
      });
      this.provider = 'webgpu-fp16';
    } catch (webgpuError) {
      try {
        this.session = await ort.InferenceSession.create(MODEL_INT8_URL, {
          executionProviders: ['wasm'],
        });
        this.provider = 'wasm-int8';
      } catch (wasmError) {
        throw new ModelFileNotFoundError(
          `DAViD のモデルを読み込めません（WebGPU: ${String(webgpuError)} /` +
            ` WASM: ${String(wasmError)}）`,
        );
      }
    }
  }

  /** 画像の正方領域を切り出して推論する（port の入口）。 */
  async estimateSquare(
    photo: PhotoRgb,
    square: { x: number; y: number; size: number },
  ): Promise<DepthNormalResult> {
    await this.init();
    if (this.session === null || this.ort === null) {
      throw new Error('DavidDepthNormalEstimator が初期化されていない');
    }
    if (
      square.size <= 0 ||
      square.x < 0 ||
      square.y < 0 ||
      square.x + square.size > photo.width ||
      square.y + square.size > photo.height
    ) {
      throw new Error(
        `切り出し領域が画像の外に出ている: x=${square.x} y=${square.y} size=${square.size}` +
          ` 画像=${photo.width}x${photo.height}`,
      );
    }

    const rgba = cropSquareToRgb(photo, square, INPUT_RESOLUTION);
    const area = INPUT_RESOLUTION * INPUT_RESOLUTION;
    // RGB の [0,1]・CHW。**チャンネル順は RGB**（デスクトップ側が実測で決めた規約）。
    const tensorData = new Float32Array(3 * area);
    for (let pixel = 0; pixel < area; pixel++) {
      tensorData[pixel] = rgba[pixel * 4] / 255;
      tensorData[area + pixel] = rgba[pixel * 4 + 1] / 255;
      tensorData[2 * area + pixel] = rgba[pixel * 4 + 2] / 255;
    }
    const feeds: Record<string, unknown> = {
      [this.session.inputNames[0]]: new this.ort.Tensor('float32', tensorData, [
        1,
        3,
        INPUT_RESOLUTION,
        INPUT_RESOLUTION,
      ]),
    };
    const results = await this.session.run(feeds as never);
    const outputs = this.session.outputNames.map((name) => {
      const tensor = results[name];
      return {
        name,
        dims: tensor.dims as readonly number[],
        data: tensor.data as Float32Array,
      };
    });
    const { depth, normal, foreground } = classifyOutputs(outputs);

    const rect = rectFromPixels(
      square.x,
      square.y,
      square.size,
      square.size,
      photo.width,
      photo.height,
    );
    const result: DepthNormalResult = {
      depth: makeField(
        Float32Array.from(depth.subarray(0, area)),
        INPUT_RESOLUTION,
        INPUT_RESOLUTION,
        rect,
      ),
      normal: Float32Array.from(normal.subarray(0, area * 3)),
      foreground: makeField(
        Float32Array.from(foreground.subarray(0, area)),
        INPUT_RESOLUTION,
        INPUT_RESOLUTION,
        rect,
      ),
    };
    validateDepthNormal(result);
    return result;
  }

  async close(): Promise<void> {
    await this.session?.release();
    this.session = null;
    this.provider = null;
  }
}
