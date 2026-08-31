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
// **解像度の階段は持たない（デスクトップ側との差分）。** あちらは長辺 256〜3840 の階段を全段回して
// 検出を束ねる（どの解像度で当たるかが写真ごとに違うため）。ブラウザでは 1 枚あたり数百 ms × 段数が
// 体感に直に出るので、**写真の解像度そのままで 1 回だけ検出する**。取り逃がしの向きは同じ（顔が
// 出なければ `FaceNotDetectedError`）で、主役の規則は共有している。

import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { FaceNotDetectedError, ModelFileNotFoundError } from '../domain/errors';
import {
  FaceSquare,
  faceSquareOfLandmarks,
  imageCenter,
  subjectIndex,
} from '../domain/faceSubject';
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
 * 検出させる顔の数の上限。
 *
 * 1 にしてはいけない（主役の判断が検出器の「確からしさ」に移る）。集合写真でも主役の規則が採点
 * できるだけの数を返させる。
 */
const MAX_FACES = 10;

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
    if (this.landmarker === null) throw new Error('FaceLandmarker が初期化されていない');
    const canvas = photoToCanvas(photo);
    const result = this.landmarker.detect(canvas);
    const faces = result.faceLandmarks ?? [];
    if (faces.length === 0) {
      throw new FaceNotDetectedError('顔を検出できませんでした。');
    }

    const candidates: Float64Array[] = [];
    const squares: FaceSquare[] = [];
    for (const face of faces) {
      if (face.length < FACE_LANDMARK_COUNT) continue;
      const points = new Float64Array(FACE_LANDMARK_COUNT * 2);
      for (let point = 0; point < FACE_LANDMARK_COUNT; point++) {
        points[point * 2] = face[point].x * photo.width;
        points[point * 2 + 1] = face[point].y * photo.height;
      }
      candidates.push(points);
      squares.push(faceSquareOfLandmarks(points, FACE_MESH_LANDMARK_COUNT));
    }
    if (candidates.length === 0) {
      throw new FaceNotDetectedError(
        `虹彩を含む ${FACE_LANDMARK_COUNT} 点が返りませんでした（検出器の設定を確認してください）。`,
      );
    }
    // **束ねない。** 束ねるのは解像度の階段が同じ顔を複数回返すからで、1 回しか検出しないこちらでは
    // 同じ顔が 2 件出ることが無い。得点の規則だけを共有する。
    return candidates[subjectIndex(squares, imageCenter([photo.width, photo.height]))];
  }

  close(): void {
    this.landmarker?.close();
    this.landmarker = null;
  }
}
