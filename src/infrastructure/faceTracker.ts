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
import { FaceExpressionTracker, FaceTrackingFrame, VideoFrameSource } from '../application/ports';
import { ModelFileNotFoundError } from '../domain/errors';
import { FACE_LANDMARKER_MODEL_URL, visionFileset } from './mediapipeVision';

export class MediaPipeFaceExpressionTracker implements FaceExpressionTracker {
  private landmarker: FaceLandmarker | null = null;
  private starting: Promise<void> | null = null;
  /** `detectForVideo` は単調増加する時刻を要求するので、渡した最後の値を覚える。 */
  private lastTimestampMs = -1;
  /** `stop()` のたびに進む。取得中の `create()` が自分の世代と比べて用済みを判る。 */
  private generation = 0;

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
    // **取得中に `stop()` が来たら、出来上がったものをその場で閉じる。** `await` の後で無条件に
    // 代入すると、止めたつもりのトラッカーが生き残って誰も `close()` しない（世代で見分ける）。
    const generation = this.generation;
    try {
      const vision = await visionFileset();
      const landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: FACE_LANDMARKER_MODEL_URL, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFaceBlendshapes: true,
        // 首を動かすモードで使う（表情だけのときは読まない）。
        outputFacialTransformationMatrixes: true,
      });
      if (generation !== this.generation) {
        landmarker.close();
        return;
      }
      this.landmarker = landmarker;
      this.lastTimestampMs = -1;
    } catch (error) {
      if (generation === this.generation) this.landmarker = null;
      throw new ModelFileNotFoundError(
        `表情トラッカーを初期化できません: ${String(error)}`,
      );
    }
  }

  detect(source: VideoFrameSource, timestampMs: number): FaceTrackingFrame | null {
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
    const matrix = result.facialTransformationMatrixes?.[0]?.data;
    return {
      scores,
      points: normalizedPoints(result.faceLandmarks?.[0]),
      headMatrix: matrix === undefined ? null : Float32Array.from(matrix),
    };
  }

  stop(): void {
    // 世代を進めると、取得中の `create()` が「自分はもう用済み」と分かる。
    this.generation++;
    this.landmarker?.close();
    this.landmarker = null;
    this.lastTimestampMs = -1;
  }
}

/**
 * 正規化座標（x, y）を平たい配列へ落とす。
 *
 * **z は捨てる。** 使うのは映像へ点を重ねることだけで、奥行きは絵に出ない。
 */
function normalizedPoints(landmarks: { x: number; y: number }[] | undefined): Float32Array {
  if (landmarks === undefined) return new Float32Array(0);
  const points = new Float32Array(landmarks.length * 2);
  for (let index = 0; index < landmarks.length; index++) {
    points[index * 2] = landmarks[index].x;
    points[index * 2 + 1] = landmarks[index].y;
  }
  return points;
}
