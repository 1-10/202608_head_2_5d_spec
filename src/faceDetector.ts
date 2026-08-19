// MediaPipe Face Landmarker のロードと推論。
// すべての処理はブラウザ内 (WASM / ローカルGPU) で完結し、画像を外部へ送信しない。
// モデル重み(.task)とWASMランタイムのみ初回に公式CDNから取得する。

import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

const WASM_BASE_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17/wasm';
const MODEL_ASSET_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

export interface FaceLandmark {
  x: number; // normalized [0,1], image space (left=0)
  y: number; // normalized [0,1], image space (top=0)
  z: number; // MediaPipeの相対Depth。実寸ではない。
}

export class FaceDetectionError extends Error {}

export class FaceDetector {
  private landmarker: FaceLandmarker | null = null;

  async init(): Promise<void> {
    if (this.landmarker) return;
    const vision = await FilesetResolver.forVisionTasks(WASM_BASE_URL);
    const baseConfig = {
      runningMode: 'IMAGE' as const,
      numFaces: 1,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    };
    try {
      this.landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_ASSET_URL, delegate: 'GPU' },
        ...baseConfig,
      });
    } catch (err) {
      // GPU delegateが使えない環境向けにCPUへフォールバック。
      this.landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_ASSET_URL, delegate: 'CPU' },
        ...baseConfig,
      });
    }
  }

  /** 静止画から顔ランドマークを検出する。検出できない場合は FaceDetectionError を投げる。 */
  detect(source: HTMLCanvasElement): FaceLandmark[] {
    if (!this.landmarker) {
      throw new FaceDetectionError('Face Landmarkerが初期化されていません。');
    }
    const result = this.landmarker.detect(source);
    const landmarks = result.faceLandmarks?.[0];
    if (!landmarks || landmarks.length === 0) {
      throw new FaceDetectionError('顔を検出できませんでした。正面を向いた顔全体が写る画像を使用してください。');
    }
    return landmarks;
  }
}
