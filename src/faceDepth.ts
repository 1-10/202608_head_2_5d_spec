// Face Depthの構築: canonical顔Depth profileの生成、MediaPipe Zとの混合、
// FULL HEAD用のFace Depth Field (2Dラスタ) 生成。

import { CANONICAL_FEATURES } from './params';
import { FACE_KEY_INDICES, type FaceTriangulation, type NormalizedFaceLandmark } from './faceTopology';

function gaussianFalloff(distance: number, radius: number): number {
  const t = distance / radius;
  return Math.exp(-(t * t));
}

function landmarkCenter(landmarks: NormalizedFaceLandmark[], indices: number[]): { x: number; y: number } {
  let x = 0;
  let y = 0;
  for (const i of indices) {
    x += landmarks[i].x;
    y += landmarks[i].y;
  }
  return { x: x / indices.length, y: y / indices.length };
}

/**
 * canonicalな顔Depth profileを構築する (spec: canonical側の特徴量)。
 * 顔幅正規化されたmodel空間座標上に、鼻・頬・唇・顎・眼窩・輪郭のガウス型bumpを配置する。
 * 特定landmark indexの正確なトポロジーに依存しすぎないよう、信頼度の高いkey point (鼻先・顎・眉間)
 * と、そこから幾何学的に導出した領域中心を組み合わせる。
 */
export function buildCanonicalFaceDepth(landmarks: NormalizedFaceLandmark[]): Float32Array {
  const k = FACE_KEY_INDICES;
  const noseTip = landmarks[k.noseTip];
  const chin = landmarks[k.chin];
  const eyeACenter = landmarkCenter(landmarks, [k.eyeA.outer, k.eyeA.inner, k.eyeA.upper1, k.eyeA.lower1]);
  const eyeBCenter = landmarkCenter(landmarks, [k.eyeB.outer, k.eyeB.inner, k.eyeB.upper1, k.eyeB.lower1]);
  const noseBridge = {
    x: (landmarks[k.eyeA.inner].x + landmarks[k.eyeB.inner].x) / 2,
    y: (landmarks[k.eyeA.inner].y + landmarks[k.eyeB.inner].y) / 2,
  };
  const mouthCenter = landmarkCenter(landmarks, [k.mouth.upperCenter, k.mouth.cornerA, k.mouth.cornerB]);
  // 頬は鼻先から左右へオフセットし、目の高さに合わせた幾何推定位置。
  const cheekA = { x: eyeACenter.x, y: (eyeACenter.y + mouthCenter.y) / 2 };
  const cheekB = { x: eyeBCenter.x, y: (eyeBCenter.y + mouthCenter.y) / 2 };

  const features: Array<{ pos: { x: number; y: number }; value: number; radius: number }> = [
    { pos: { x: noseTip.x, y: noseTip.y }, value: CANONICAL_FEATURES.noseTip, radius: 0.11 },
    { pos: noseBridge, value: CANONICAL_FEATURES.noseBridge, radius: 0.14 },
    { pos: cheekA, value: CANONICAL_FEATURES.cheek, radius: 0.2 },
    { pos: cheekB, value: CANONICAL_FEATURES.cheek, radius: 0.2 },
    { pos: mouthCenter, value: CANONICAL_FEATURES.upperLip, radius: 0.1 },
    { pos: { x: chin.x, y: chin.y }, value: CANONICAL_FEATURES.chin, radius: 0.13 },
    { pos: eyeACenter, value: CANONICAL_FEATURES.eyeSocket, radius: 0.09 },
    { pos: eyeBCenter, value: CANONICAL_FEATURES.eyeSocket, radius: 0.09 },
  ];

  const result = new Float32Array(landmarks.length);
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    let z = 0;
    for (const f of features) {
      const dx = lm.x - f.pos.x;
      const dy = lm.y - f.pos.y;
      const d = Math.sqrt(dx * dx + dy * dy);
      z += f.value * gaussianFalloff(d, f.radius);
    }
    // 輪郭(外周)ほど奥へ: 楕円近似の半径比を使った滑らかな減衰。
    const rx = 0.5;
    const ry = 0.65;
    const r2 = (lm.x / rx) * (lm.x / rx) + (lm.y / ry) * (lm.y / ry);
    const edge = smoothstep(0.4, 1.1, r2);
    z += CANONICAL_FEATURES.faceContour * edge;
    result[i] = z;
  }
  return result;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

function clamp01(x: number): number {
  return Math.min(1, Math.max(0, x));
}

/** MediaPipe ZとcanonicalなZをspec式で混合する。 */
export function blendFaceDepth(
  zMediapipe: number,
  zCanonical: number,
  canonicalMix: number,
  faceDepthScale: number,
): number {
  return zMediapipe * faceDepthScale * (1 - canonicalMix) + zCanonical * canonicalMix;
}

export function computeFinalFaceDepthPerVertex(
  landmarks: NormalizedFaceLandmark[],
  canonicalDepth: Float32Array,
  canonicalMix: number,
  faceDepthScale: number,
): Float32Array {
  const out = new Float32Array(landmarks.length);
  for (let i = 0; i < landmarks.length; i++) {
    out[i] = blendFaceDepth(landmarks[i].zMediapipe, canonicalDepth[i], canonicalMix, faceDepthScale);
  }
  return out;
}

export interface FaceDepthField {
  size: number;
  depth: Float32Array; // size*size, model空間Z (faceWidth正規化)
  coverage: Float32Array; // size*size, 1=顔三角形内部, 0=外部
  imageWidth: number;
  imageHeight: number;
}

/**
 * Face MeshのtriangleをUV空間(画像全体)にラスタライズし、2D Face Depth Fieldを構築する。
 * Head Gridの各頂点はこのFieldをUV参照してFace由来のDepthを取得する。
 */
export function buildFaceDepthField(
  landmarks: NormalizedFaceLandmark[],
  zFinalPerVertex: Float32Array,
  triangulation: FaceTriangulation,
  imageWidth: number,
  imageHeight: number,
  size: number,
): FaceDepthField {
  const depth = new Float32Array(size * size);
  const coverage = new Float32Array(size * size);
  const sx = size / imageWidth;
  const sy = size / imageHeight;

  const tris = triangulation.triangles;
  for (let t = 0; t < tris.length; t += 3) {
    const i0 = tris[t];
    const i1 = tris[t + 1];
    const i2 = tris[t + 2];
    const p0 = landmarks[i0];
    const p1 = landmarks[i1];
    const p2 = landmarks[i2];

    const minFx = Math.max(0, Math.floor(Math.min(p0.px, p1.px, p2.px) * sx));
    const maxFx = Math.min(size - 1, Math.ceil(Math.max(p0.px, p1.px, p2.px) * sx));
    const minFy = Math.max(0, Math.floor(Math.min(p0.py, p1.py, p2.py) * sy));
    const maxFy = Math.min(size - 1, Math.ceil(Math.max(p0.py, p1.py, p2.py) * sy));
    if (minFx > maxFx || minFy > maxFy) continue;

    const denom = (p1.py - p2.py) * (p0.px - p2.px) + (p2.px - p1.px) * (p0.py - p2.py);
    if (Math.abs(denom) < 1e-9) continue;

    for (let fy = minFy; fy <= maxFy; fy++) {
      const qy = (fy + 0.5) / sy;
      for (let fx = minFx; fx <= maxFx; fx++) {
        const qx = (fx + 0.5) / sx;
        const w0 = ((p1.py - p2.py) * (qx - p2.px) + (p2.px - p1.px) * (qy - p2.py)) / denom;
        const w1 = ((p2.py - p0.py) * (qx - p2.px) + (p0.px - p2.px) * (qy - p2.py)) / denom;
        const w2 = 1 - w0 - w1;
        if (w0 < -1e-4 || w1 < -1e-4 || w2 < -1e-4) continue;

        const idx = fy * size + fx;
        const z = w0 * zFinalPerVertex[i0] + w1 * zFinalPerVertex[i1] + w2 * zFinalPerVertex[i2];
        depth[idx] = z;
        coverage[idx] = 1;
      }
    }
  }

  return { size, depth, coverage, imageWidth, imageHeight };
}

/** Face Depth FieldをUV座標(u: 0-1, v: 0-1 上が1)でサンプルする。 */
export function sampleFaceDepthField(field: FaceDepthField, u: number, v: number): { depth: number; coverage: number } {
  const fx = clampIndex(Math.floor(u * field.size), field.size);
  const fy = clampIndex(Math.floor((1 - v) * field.size), field.size);
  const idx = fy * field.size + fx;
  return { depth: field.depth[idx], coverage: field.coverage[idx] };
}

function clampIndex(i: number, size: number): number {
  return Math.min(size - 1, Math.max(0, i));
}
