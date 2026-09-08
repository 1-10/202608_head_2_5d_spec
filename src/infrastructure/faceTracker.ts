// リアルタイム表情トラッカーのアダプタ（`<video>` → blendshape スコア）。
//
// **書き出し用の `MediaPipeFaceLandmarkDetector` とは別インスタンス。** あちらは
// `runningMode: 'IMAGE'` / `numFaces: 5` / blendshape 無しで、主役の選定（`domain/faceSubject`）と
// 二段検出（`domain/faceLadder`）という書き出し固有の関心を持っている。同じ検出器を使い回すには
// `runningMode` を毎回切り替えることになり、**リアルタイムの都合で書き出しの経路が歪む**。
// モデルと wasm は `mediapipeVision` で共有しているので、二重に落ちることはない。
//
// **こちらは `numFaces: 1` でよい。** 主役を選ぶ必要があるのは「1 枚の写真から誰の頭を起こすか」
// という取り返しのつかない判断だからで、カメラの前に居る人は 1 人という前提で足りる（違えば
// 画面を見てすぐ分かるし、次のフレームでやり直せる）。
//
// **返すのはカテゴリ名 → スコアだけ。** MediaPipe の型（`Classifications` / `Category`）を
// presentation まで漏らさない。

import { FaceLandmarker } from '@mediapipe/tasks-vision';
import { FaceExpressionTracker, VideoFrameSource } from '../application/ports';
import { ModelFileNotFoundError } from '../domain/errors';
import { FACE_LANDMARKER_MODEL_URL, visionFileset } from './mediapipeVision';

export class MediaPipeFaceExpressionTracker implements FaceExpressionTracker {
  private landmarker: FaceLandmarker | null = null;
  private starting: Promise<void> | null = null;
  /** `detectForVideo` は単調増加する時刻を要求するので、渡した最後の値を覚える。 */
  private lastTimestampMs = -1;

  async start(): Promise<void> {
    if (this.landmarker !== null) return;
    // 連打で 2 つ作らない（作りかけの Promise を共有する）。
    if (this.starting !== null) return this.starting;
    this.starting = this.create();
    try {
      await this.starting;
    } finally {
      this.starting = null;
    }
  }

  private async create(): Promise<void> {
    try {
      const vision = await visionFileset();
      this.landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: FACE_LANDMARKER_MODEL_URL, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: false,
      });
      this.lastTimestampMs = -1;
    } catch (error) {
      this.landmarker = null;
      throw new ModelFileNotFoundError(
        `表情トラッカーを初期化できません: ${String(error)}`,
      );
    }
  }

  detect(source: VideoFrameSource, timestampMs: number): ReadonlyMap<string, number> | null {
    if (this.landmarker === null) return null;
    // 同じ時刻を 2 回渡すと MediaPipe が例外を投げる。1ms ずらして単調増加を保つ。
    const timestamp =
      timestampMs > this.lastTimestampMs ? timestampMs : this.lastTimestampMs + 1;
    this.lastTimestampMs = timestamp;
    const result = this.landmarker.detectForVideo(source, timestamp);
    const blendshapes = result.faceBlendshapes;
    if (blendshapes === undefined || blendshapes.length === 0) return null;
    const scores = new Map<string, number>();
    for (const category of blendshapes[0].categories) {
      // `categoryName` が空の版があるので `displayName` へ落ちる（どちらも無ければ捨てる）。
      const name = category.categoryName || category.displayName;
      if (name === '') continue;
      scores.set(name, category.score);
    }
    return scores;
  }

  stop(): void {
    this.landmarker?.close();
    this.landmarker = null;
    this.lastTimestampMs = -1;
  }
}
