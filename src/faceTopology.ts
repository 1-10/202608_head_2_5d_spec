// Face Mesh topology関連: landmarkの正規化、三角形分割、目/口などのkey landmark index定義。
// Face Mesh topologyとHead Grid topologyは分離する (spec方針)。このファイルはFace側のみを扱う。

import Delaunator from 'delaunator';
import type { FaceLandmark } from './faceDetector';

export interface NormalizedFaceLandmark {
  x: number; // モデル空間: (imageX - headCenterX) / faceWidth
  y: number; // モデル空間: (headCenterY - imageY) / faceWidth  (上方向が+)
  zMediapipe: number; // MediaPipe由来Z (faceWidthで正規化済み, canonicalMix前)
  u: number; // texture座標 [0,1]
  v: number; // texture座標 [0,1] (上が1)
  px: number; // 画像ピクセル空間X (三角形分割・距離場計算用)
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

export interface FaceTriangulation {
  triangles: Uint32Array; // 3頂点ずつのindex列
  hull: number[]; // 外周(輪郭)を構成する頂点indexのCCW列
}

/** 2D位置からDelaunay三角形分割を行い、Face Mesh topologyの代替として使用する。 */
export function triangulateFaceLandmarks(landmarks: NormalizedFaceLandmark[]): FaceTriangulation {
  const coords = new Float64Array(landmarks.length * 2);
  for (let i = 0; i < landmarks.length; i++) {
    coords[i * 2] = landmarks[i].px;
    coords[i * 2 + 1] = landmarks[i].py;
  }
  const delaunay = new Delaunator(coords);
  return {
    triangles: delaunay.triangles,
    hull: Array.from(delaunay.hull),
  };
}

// 高信頼度のkey landmark index (MediaPipe FaceMesh 468/478点トポロジーで広く使われる定数)。
// Blink(EAR計算と同型の6点)・Talk(口角/上下唇)・canonical depth基準点として使用する。
export const FACE_KEY_INDICES = {
  noseTip: 1,
  chin: 152,
  foreheadTop: 10,
  eyeA: { outer: 33, upper1: 160, upper2: 158, inner: 133, lower1: 153, lower2: 144 },
  eyeB: { outer: 263, upper1: 387, upper2: 385, inner: 362, lower1: 373, lower2: 380 },
  eyebrowA: [70, 63, 105, 66, 107],
  eyebrowB: [300, 293, 334, 296, 336],
  mouth: { cornerA: 61, cornerB: 291, upperCenter: 13, lowerCenter: 14, upperOuter: 0, lowerOuter: 17 },
};

export function getLandmarkSafe(landmarks: NormalizedFaceLandmark[], index: number): NormalizedFaceLandmark {
  return landmarks[Math.min(index, landmarks.length - 1)];
}
