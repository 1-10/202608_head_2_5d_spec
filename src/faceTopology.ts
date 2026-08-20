// MediaPipe Face Landmarkerの生ランドマークをモデル空間・テクスチャ空間へ正規化する。

import type { FaceLandmark } from './faceDetector';

export interface NormalizedFaceLandmark {
  x: number; // モデル空間: (imageX - headCenterX) / faceWidth
  y: number; // モデル空間: (headCenterY - imageY) / faceWidth  (上方向が+)
  zMediapipe: number; // MediaPipe由来Z (faceWidthで正規化済み)
  u: number; // texture座標 [0,1]
  v: number; // texture座標 [0,1] (上が1)
  px: number; // 画像ピクセル空間X
  py: number; // 画像ピクセル空間Y
}

export interface NormalizedFaceResult {
  landmarks: NormalizedFaceLandmark[];
  faceWidth: number; // ピクセル単位
  headCenterPx: { x: number; y: number };
}

/** MediaPipeの生ランドマークをモデル空間・テクスチャ空間へ正規化する。 */
export function normalizeFaceLandmarks(
  raw: FaceLandmark[],
  imageWidth: number,
  imageHeight: number,
): NormalizedFaceResult {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const lm of raw) {
    const px = lm.x * imageWidth;
    const py = lm.y * imageHeight;
    if (px < minX) minX = px;
    if (px > maxX) maxX = px;
    if (py < minY) minY = py;
    if (py > maxY) maxY = py;
  }

  const faceWidth = Math.max(1e-6, maxX - minX);
  const headCenterPx = { x: (minX + maxX) / 2, y: (minY + maxY) / 2 };

  const landmarks: NormalizedFaceLandmark[] = raw.map((lm) => {
    const px = lm.x * imageWidth;
    const py = lm.y * imageHeight;
    // MediaPipeのzはxと同じスケール(画像幅基準)で正規化されている。
    const pz = lm.z * imageWidth;
    return {
      x: (px - headCenterPx.x) / faceWidth,
      y: (headCenterPx.y - py) / faceWidth,
      zMediapipe: -pz / faceWidth, // MediaPipeはカメラに近いほど負値 → 手前を+へ反転
      u: lm.x,
      v: 1 - lm.y,
      px,
      py,
    };
  });

  return { landmarks, faceWidth, headCenterPx };
}
