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

// MediaPipe FACEMESH_FACE_OVAL の輪郭リング36点 (顎先152から時計回り相当の公式順)。
// 髪に被られてバイアスしうるのはこの輪郭点 (+額側) — bald再検出での差し替え対象
export const FACE_OVAL_LANDMARKS: readonly number[] = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
  148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];

export interface OvalMergeStats {
  replaced: number; // 差し替えた輪郭点数
  rejected: number; // 外れ値として棄却した点数
  meanShiftFrac: number; // 差し替え点の平均シフト (faceWidth比)
  maxShiftFrac: number; // 差し替え点の最大シフト (faceWidth比)
}

/**
 * 顔輪郭 (FACE_OVAL) だけをbald画像の再検出値へ差し替えた生ランドマークを返す。
 * 目・口などはfillが触らないため元画像の検出値を保持する。
 * bald側がfillの不自然さで大きく外れた点 (シフト > maxShiftFrac×faceWidth) は
 * 元の値のまま残す (外れ値ガード)。
 */
export function mergeFaceOvalLandmarks(
  base: FaceLandmark[],
  bald: FaceLandmark[],
  imageWidth: number,
  imageHeight: number,
  maxShiftFrac = 0.12,
): { landmarks: FaceLandmark[]; stats: OvalMergeStats } {
  // faceWidthは元検出のbboxから求める (normalizeFaceLandmarksと同じ定義)
  let minX = Infinity;
  let maxX = -Infinity;
  for (const lm of base) {
    const px = lm.x * imageWidth;
    if (px < minX) minX = px;
    if (px > maxX) maxX = px;
  }
  const faceWidth = Math.max(1e-6, maxX - minX);

  const merged = base.slice();
  const stats: OvalMergeStats = { replaced: 0, rejected: 0, meanShiftFrac: 0, maxShiftFrac: 0 };
  for (const idx of FACE_OVAL_LANDMARKS) {
    const b = base[idx];
    const n = bald[idx];
    if (!b || !n) continue;
    const dx = (n.x - b.x) * imageWidth;
    const dy = (n.y - b.y) * imageHeight;
    const shiftFrac = Math.hypot(dx, dy) / faceWidth;
    if (shiftFrac > maxShiftFrac) {
      stats.rejected++;
      continue;
    }
    merged[idx] = n;
    stats.replaced++;
    stats.meanShiftFrac += shiftFrac;
    if (shiftFrac > stats.maxShiftFrac) stats.maxShiftFrac = shiftFrac;
  }
  if (stats.replaced > 0) stats.meanShiftFrac /= stats.replaced;
  return { landmarks: merged, stats };
}
