// 顔ランドマークのアダプタ。
//
// MediaPipe Tasks の FaceLandmarker（float16 / 1）をブラウザで実行し、478 点（顔メッシュ 468 + 虹彩
// 10）を**画像画素座標**で返す。
//
// **点を選ばない。** どの点を使うかは消費側の関心（形状フィットは 468 未満、眼球は虹彩 10 点）。
//
// **主役の選び方は実装の裁量ではない。** `numFaces` を 1 にして検出器の「最も確からしい 1 つ」へ
// 委ねると、選ばれる顔が解像度やモデルの版で黙って変わる。複数返させて `domain/faceSubject` の規則
// （得点 = 一辺 − 画像中心からの距離）で選ぶ。
//
// **二段検出は `domain/faceLadder` が持つ。** ここは「1 枚の画像から顔を全部返す」だけを担い、
// 解像度の階段・主役の選定・主役の周りを切っての再検出はあちらが組む（純粋なので検出器なしで
// 検証できる）。1 回だけの検出で済ませてはいけない — 大きな写真では顔幅が数十画素になり、口の位置
// がずれる。

import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { FaceNotDetectedError, ModelFileNotFoundError } from '../domain/errors';
import { detectTwoPass } from '../domain/faceLadder';
import { PhotoRgb } from '../domain/photo';
import { FACE_LANDMARK_COUNT, FACE_MESH_LANDMARK_COUNT, FaceLandmarkDetector } from '../application/ports';
import { photoToCanvas } from './photoCanvas';

const WASM_BASE_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17/wasm';

/**
 * モデルの取得元。**バージョン付きパス**（`/1/`）を使う。
 *
 * `/latest/` は中身が動くので使わない（デスクトップ側の `tools/fetch_models.py` が同じ URL を
 * ハッシュ付きで固定している）。
 */
const MODEL_ASSET_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker' +
  '/float16/1/face_landmarker.task';

/**
 * 検出させる顔の数の上限（デスクトップ側 `MAX_DETECTED_FACES` と同値）。
 *
 * 1 にしてはいけない（主役の判断が検出器の「確からしさ」に移る）。**上限であってコストではない** —
 * 増えるのは写っている顔 1 つあたりの推論だけで、使われない枠のぶんは払わない。
 */
const MAX_FACES = 5;

export class MediaPipeFaceLandmarkDetector implements FaceLandmarkDetector {
  private landmarker: FaceLandmarker | null = null;

  async init(): Promise<void> {
    if (this.landmarker !== null) return;
    try {
      const vision = await FilesetResolver.forVisionTasks(WASM_BASE_URL);
      this.landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_ASSET_URL, delegate: 'GPU' },
        runningMode: 'IMAGE',
        numFaces: MAX_FACES,
        // 虹彩 10 点は refine 有効時にしか出ない。眼球テクスチャの半径と中心がこれで決まる。
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: false,
      });
    } catch (error) {
      throw new ModelFileNotFoundError(
        `FaceLandmarker を初期化できません: ${String(error)}`,
      );
    }
  }

  async detect(photo: PhotoRgb): Promise<Float64Array> {
    await this.init();
    const result = detectTwoPass({
      detectFaces: (image) => this.detectOnce(image),
      photo,
      faceMeshCount: FACE_MESH_LANDMARK_COUNT,
    });
    return result.landmarks;
  }

  /** 1 枚の画像から写っている顔を全部返す（階段の 1 段ぶん）。 */
  private detectOnce(photo: PhotoRgb): Float64Array[] {
    if (this.landmarker === null) throw new Error('FaceLandmarker が初期化されていない');
    const faces = this.landmarker.detect(photoToCanvas(photo)).faceLandmarks ?? [];
    const candidates: Float64Array[] = [];
    for (const face of faces) {
      // 虹彩を含む 478 点が返らない段は候補にしない（眼球テクスチャの半径が取れない）。
      if (face.length < FACE_LANDMARK_COUNT) continue;
      const points = new Float64Array(FACE_LANDMARK_COUNT * 2);
      for (let point = 0; point < FACE_LANDMARK_COUNT; point++) {
        points[point * 2] = face[point].x * photo.width;
        points[point * 2 + 1] = face[point].y * photo.height;
      }
      candidates.push(points);
    }
    if (candidates.length === 0) {
      throw new FaceNotDetectedError(
        `${photo.width}x${photo.height} では虹彩を含む ${FACE_LANDMARK_COUNT} 点が返らなかった`,
      );
    }
    return candidates;
  }

  close(): void {
    this.landmarker?.close();
    this.landmarker = null;
  }
}
