// MediaPipe Tasks Vision の取得元と wasm の共有。
//
// **書き出し用の顔検出（`faceLandmarks`）とリアルタイム表情トラッカー（`faceTracker`）が同じ
// モデルと同じ wasm を使う。** URL を各アダプタへ書き写すと、片方だけ版が動いたときに「同じ
// モデルのはずなのに結果が違う」が黙って起きる。取得元はここが正本。
//
// **`FilesetResolver.forVisionTasks` の結果を使い回す。** アダプタごとに解決すると wasm を
// 二重に落とす（実測で数 MB）。解決は 1 回だけ行い、その Promise を共有する。

import { FilesetResolver } from '@mediapipe/tasks-vision';

/**
 * `FilesetResolver.forVisionTasks` の戻り値。
 *
 * `WasmFileset` はパッケージから export されていないので、戻り値の型をその場で引く
 * （同じ形を手で写すと、パッケージの側が変わったときに黙って嘘になる）。
 */
type VisionFileset = Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;

/** wasm の配布元。`package.json` の `@mediapipe/tasks-vision` と同じ版に固定する。 */
export const WASM_BASE_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17/wasm';

/**
 * FaceLandmarker のモデルの取得元。**バージョン付きパス**（`/1/`）を使う。
 *
 * `/latest/` は中身が動くので使わない（デスクトップ側の `tools/fetch_models.py` が同じ URL を
 * ハッシュ付きで固定している）。
 */
export const FACE_LANDMARKER_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker' +
  '/float16/1/face_landmarker.task';

let pending: Promise<VisionFileset> | null = null;

/** wasm の解決（最初の呼び出しだけが実際に落とす）。 */
export function visionFileset(): Promise<VisionFileset> {
  if (pending === null) pending = FilesetResolver.forVisionTasks(WASM_BASE_URL);
  return pending;
}
