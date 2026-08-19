// GNM Head (github.com/google/GNM, Apache-2.0) による真3D頭部バックエンドの
// アセット読込と写真へのフィッティング。
//
// - アセットは tools/export_gnm_assets.py が生成する gnm_head_lite.bin
//   (skin_exterior+eye_exteriorsサブセット / identity基底上位64成分int16 / iBUG-68 barycentric)
// - フィットは「2D相似 (x,y) + identity係数の正則化最小二乗」を交互に数回。
//   zは相似の等方スケールに従う (=頭部は実比率の奥行きを持つ)。既存reliefの
//   平坦化されたfaceZFinalにはフィットさせない — それがGNM導入の目的のため。
// - MediaPipe 468点からiBUG-68への対応表はコミュニティで広く使われる定数。

import type { NormalizedFaceLandmark } from './faceTopology';

export interface GnmModel {
  vertexCount: number;
  triangleCount: number;
  basisCount: number;
  positions: Float32Array; // (N,3) メートル, +Y上/+Z前
  triangles: Uint32Array; // (T,3)
  basisQ: Int16Array; // (K,N,3) int16量子化
  basisScales: Float32Array; // (K,) 量子化スケール (値 = q * scale / 32767)
  landmarkIndices: Uint32Array; // (68,3)
  landmarkWeights: Float32Array; // (68,3)
  earWeight: Uint8Array; // (N,) 耳グループ重み 0-255
  expressionCount: number; // 0 = 旧アセット (表情なし)
  expressionBasisQ: Int16Array; // (M,N,3) int16量子化
  expressionScales: Float32Array; // (M,)
  expressionNames: string[];
  // 密対応 (MediaPipe 468点→GNM表面barycentric)。0 = 旧アセット (68点のみ)
  denseCount: number;
  denseMpIndices: Uint16Array; // (M,) MediaPipe landmark index
  denseTriIndices: Uint32Array; // (M,3) GNM頂点index
  denseBaryWeights: Float32Array; // (M,3)
  denseFitWeights: Float32Array; // (M,) フィット重み (信頼度×領域重み)
}

/** MediaPipe FaceMesh 468点 → iBUG-68 の対応表 (顎17/眉10/鼻9/目12/口20)。 */
export const MEDIAPIPE_IBUG68: number[] = [
  // 顎ライン (向かって左→右)
  162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
  // 眉 (左5, 右5)
  70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
  // 鼻梁4 + 鼻底5
  168, 197, 5, 4, 75, 97, 2, 326, 305,
  // 目 (左6, 右6)
  33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
  // 口 外周12 + 内周8
  61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317, 14, 87,
];

interface BinHeader {
  vertexCount: number;
  triangleCount: number;
  identityBasisCount: number;
  identityBasisScales: number[];
  expressionBasisCount?: number;
  expressionBasisScales?: number[];
  expressionNames?: string[];
  denseLandmarkCount?: number;
  landmarkCount: number;
  sections: Record<string, { offset: number; byteLength: number; dtype: string }>;
}

/** gnm_head_lite.bin を読み込む。形式は tools/export_gnm_assets.py が正本。 */
export async function loadGnmModel(url = 'gnm/gnm_head_lite.bin'): Promise<GnmModel> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(
      `GNMアセットを取得できません (${res.status})。tools/export_gnm_assets.py で public/gnm/gnm_head_lite.bin を生成してください。`,
    );
  }
  const buf = await res.arrayBuffer();
  const view = new DataView(buf);
  const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 4));
  if (magic !== 'GNML') throw new Error('GNMアセットの形式が不正です (magic不一致)。');
  const headerLen = view.getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, headerLen))) as BinHeader;
  const payloadStart = 8 + headerLen;

  const section = (name: string) => {
    const s = header.sections[name];
    if (!s) throw new Error(`GNMアセットにセクション ${name} がありません。`);
    return { start: payloadStart + s.offset, byteLength: s.byteLength };
  };
  const f32 = (name: string) => {
    const { start, byteLength } = section(name);
    return new Float32Array(buf.slice(start, start + byteLength));
  };
  const u32 = (name: string) => {
    const { start, byteLength } = section(name);
    return new Uint32Array(buf.slice(start, start + byteLength));
  };

  const basisSec = section('identityBasisQ');
  const earSec = section('earWeight');
  const hasExpression = !!header.sections['expressionBasisQ'] && (header.expressionBasisCount ?? 0) > 0;
  const exprSec = hasExpression ? section('expressionBasisQ') : null;
  return {
    vertexCount: header.vertexCount,
    triangleCount: header.triangleCount,
    basisCount: header.identityBasisCount,
    positions: f32('positions'),
    triangles: u32('triangles'),
    basisQ: new Int16Array(buf.slice(basisSec.start, basisSec.start + basisSec.byteLength)),
    basisScales: new Float32Array(header.identityBasisScales),
    landmarkIndices: u32('landmarkIndices'),
    landmarkWeights: f32('landmarkWeights'),
    earWeight: new Uint8Array(buf.slice(earSec.start, earSec.start + earSec.byteLength)),
    expressionCount: hasExpression ? (header.expressionBasisCount ?? 0) : 0,
    expressionBasisQ: exprSec
      ? new Int16Array(buf.slice(exprSec.start, exprSec.start + exprSec.byteLength))
      : new Int16Array(0),
    expressionScales: new Float32Array(header.expressionBasisScales ?? []),
    expressionNames: header.expressionNames ?? [],
    denseCount: header.denseLandmarkCount ?? 0,
    denseMpIndices: header.sections['denseMpIndices']
      ? (() => {
          const s = section('denseMpIndices');
          return new Uint16Array(buf.slice(s.start, s.start + s.byteLength));
        })()
      : new Uint16Array(0),
    denseTriIndices: header.sections['denseTriIndices'] ? u32('denseTriIndices') : new Uint32Array(0),
    denseBaryWeights: header.sections['denseBaryWeights'] ? f32('denseBaryWeights') : new Float32Array(0),
    denseFitWeights: header.sections['denseFitWeights'] ? f32('denseFitWeights') : new Float32Array(0),
  };
}

/** 2D相似変換 (x,yの回転+等方スケール+平行移動)。zは s*z + tz。 */
export interface SimilarityTransform {
  s: number;
  cos: number;
  sin: number;
  tx: number;
  ty: number;
  tz: number;
}

export interface GnmFitResult {
  vertices: Float32Array; // (N,3) モデル空間 (faceWidth正規化・手前+Z)
  coeffs: Float32Array; // (K,) identity係数 (z-scoreスケール)
  sim: SimilarityTransform;
  landmarkZ: Float32Array; // (68,) フィット後モデル空間z (髪Depthのスケール合わせ用)
  centerZ: number; // 頂点zの平均 (回転pivot用)
}

/** barycentric対応 (indices (M,3) + weights (M,3)) で表面点位置を求める。 */
function barycentricPositions(
  verts: Float32Array,
  indices: Uint32Array,
  weights: Float32Array,
  count: number,
): Float32Array {
  const out = new Float32Array(count * 3);
  for (let k = 0; k < count; k++) {
    for (let j = 0; j < 3; j++) {
      const vi = indices[k * 3 + j];
      const w = weights[k * 3 + j];
      out[k * 3 + 0] += verts[vi * 3 + 0] * w;
      out[k * 3 + 1] += verts[vi * 3 + 1] * w;
      out[k * 3 + 2] += verts[vi * 3 + 2] * w;
    }
  }
  return out;
}

/** HEAD_SPARSE_68のbarycentricでGNMランドマーク位置を求める。 */
function gnmLandmarkPositions(model: GnmModel, verts: Float32Array): Float32Array {
  return barycentricPositions(verts, model.landmarkIndices, model.landmarkWeights, model.landmarkIndices.length / 3);
}

/** identity係数を適用した頂点位置 (GNM座標系のまま)。 */
export function applyIdentity(model: GnmModel, coeffs: Float32Array): Float32Array {
  const { vertexCount: n, basisCount: k } = model;
  const out = new Float32Array(model.positions);
  for (let i = 0; i < k; i++) {
    const c = coeffs[i];
    if (c === 0) continue;
    const cs = (c * model.basisScales[i]) / 32767;
    const base = i * n * 3;
    for (let j = 0; j < n * 3; j++) out[j] += model.basisQ[base + j] * cs;
  }
  return out;
}

/** 2D相似フィット (重み付き最小二乗閉形式)。zはスケール共有でオフセットのみ合わせる。 */
function fitSimilarity2D(
  src: Float32Array,
  dst: Float32Array,
  count: number,
  weights: Float32Array | null = null,
): SimilarityTransform {
  const w = (i: number) => (weights ? weights[i] : 1);
  let wSum = 0;
  let sxm = 0;
  let sym = 0;
  let dxm = 0;
  let dym = 0;
  for (let i = 0; i < count; i++) {
    const wi = w(i);
    wSum += wi;
    sxm += src[i * 3] * wi;
    sym += src[i * 3 + 1] * wi;
    dxm += dst[i * 3] * wi;
    dym += dst[i * 3 + 1] * wi;
  }
  wSum = Math.max(1e-12, wSum);
  sxm /= wSum;
  sym /= wSum;
  dxm /= wSum;
  dym /= wSum;

  let dot = 0;
  let cross = 0;
  let norm = 0;
  for (let i = 0; i < count; i++) {
    const wi = w(i);
    const ax = src[i * 3] - sxm;
    const ay = src[i * 3 + 1] - sym;
    const bx = dst[i * 3] - dxm;
    const by = dst[i * 3 + 1] - dym;
    dot += (ax * bx + ay * by) * wi;
    cross += (ax * by - ay * bx) * wi;
    norm += (ax * ax + ay * ay) * wi;
  }
  const s = Math.hypot(dot, cross) / Math.max(1e-12, norm);
  const theta = Math.atan2(cross, dot);
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);
  const tx = dxm - s * (cos * sxm - sin * sym);
  const ty = dym - s * (sin * sxm + cos * sym);

  let tz = 0;
  for (let i = 0; i < count; i++) tz += (dst[i * 3 + 2] - s * src[i * 3 + 2]) * w(i);
  tz /= wSum;

  return { s, cos, sin, tx, ty, tz };
}

export function applySimilarityInPlace(verts: Float32Array, sim: SimilarityTransform): void {
  const { s, cos, sin, tx, ty, tz } = sim;
  for (let i = 0; i < verts.length; i += 3) {
    const x = verts[i];
    const y = verts[i + 1];
    verts[i] = s * (cos * x - sin * y) + tx;
    verts[i + 1] = s * (sin * x + cos * y) + ty;
    verts[i + 2] = s * verts[i + 2] + tz;
  }
}

/** 対称正定値行列のCholesky解 (A x = b, in-place破壊)。 */
function solveSPD(a: Float64Array, b: Float64Array, n: number): Float64Array {
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = a[i * n + j];
      for (let m = 0; m < j; m++) sum -= a[i * n + m] * a[j * n + m];
      if (i === j) {
        a[i * n + j] = Math.sqrt(Math.max(1e-12, sum));
      } else {
        a[i * n + j] = sum / a[j * n + j];
      }
    }
  }
  const x = Float64Array.from(b);
  for (let i = 0; i < n; i++) {
    for (let m = 0; m < i; m++) x[i] -= a[i * n + m] * x[m];
    x[i] /= a[i * n + i];
  }
  for (let i = n - 1; i >= 0; i--) {
    for (let m = i + 1; m < n; m++) x[i] -= a[m * n + i] * x[m];
    x[i] /= a[i * n + i];
  }
  return x;
}

const FIT_ITERATIONS = 5;
const COEFF_CLAMP = 3.0; // 係数はz-scoreスケール。統計的に妥当な範囲へクランプ
// z方向の弱拘束の重み (xy比)。MediaPipeのzは相対値でノイズが多いが、
// xyだけではz方向にしか効かない係数が正則化任せ (≒0埋め) になる縮退を切る
const Z_CONSTRAINT_WEIGHT = 0.15;
// Σw正規化後の正則化の単位。正規化前の実装 (λ×1e-3を非正規化AᵀAへ加算, 対応数~458)
// と実効強度が揃うよう 1e-3/(2×458×平均重み1) ≈ 1e-6 とする
const REG_UNIT = 1e-6;

/**
 * GNM Headを顔ランドマークへフィットする。
 * targets: モデル空間の468点 (x,yのみ使用。zはGNMの実比率に任せる)。
 */
export function fitGnmToLandmarks(
  model: GnmModel,
  landmarks: NormalizedFaceLandmark[],
  identityReg: number,
): GnmFitResult {
  // 対応点: 密対応 (最大468点。目・口・輪郭の精度と横幅追従のため) があればそれを、
  // 無ければ旧68点 (HEAD_SPARSE_68) を使う
  const useDense = model.denseCount > 0;
  const lmCount = useDense ? model.denseCount : MEDIAPIPE_IBUG68.length;
  const corrIdx = useDense ? model.denseTriIndices : model.landmarkIndices;
  const corrBary = useDense ? model.denseBaryWeights : model.landmarkWeights;
  const fitWeights = useDense ? model.denseFitWeights : null;

  const targets = new Float32Array(lmCount * 3);
  for (let m = 0; m < lmCount; m++) {
    const lm = landmarks[useDense ? model.denseMpIndices[m] : MEDIAPIPE_IBUG68[m]];
    targets[m * 3 + 0] = lm.x;
    targets[m * 3 + 1] = lm.y;
    targets[m * 3 + 2] = lm.zMediapipe; // 弱拘束としてのみ使用 (Z_CONSTRAINT_WEIGHT)
  }

  // 各基底の対応点位置への寄与 (K,M,3) を先に射影しておく
  const k = model.basisCount;
  const lmBasis = new Float32Array(k * lmCount * 3);
  for (let i = 0; i < k; i++) {
    const scale = model.basisScales[i] / 32767;
    const base = i * model.vertexCount * 3;
    for (let m = 0; m < lmCount; m++) {
      for (let j = 0; j < 3; j++) {
        const vi = corrIdx[m * 3 + j];
        const w = corrBary[m * 3 + j] * scale;
        lmBasis[(i * lmCount + m) * 3 + 0] += model.basisQ[base + vi * 3 + 0] * w;
        lmBasis[(i * lmCount + m) * 3 + 1] += model.basisQ[base + vi * 3 + 1] * w;
        lmBasis[(i * lmCount + m) * 3 + 2] += model.basisQ[base + vi * 3 + 2] * w;
      }
    }
  }
  const meanLm = barycentricPositions(model.positions, corrIdx, corrBary, lmCount); // 平均形状の対応点 (固定)

  // shapedLm(c) = meanLm + Σ cᵢ · lmBasisᵢ
  const shapedLm = (coeffs: Float32Array): Float32Array => {
    const out = new Float32Array(meanLm);
    for (let i = 0; i < k; i++) {
      const c = coeffs[i];
      if (c === 0) continue;
      for (let j = 0; j < lmCount * 3; j++) out[j] += lmBasis[i * lmCount * 3 + j] * c;
    }
    return out;
  };

  let coeffs = new Float32Array(k);
  let sim = fitSimilarity2D(meanLm, targets, lmCount, fitWeights);

  // coarse-to-fine: 前半の反復は強い正則化で大域 (サイズ・輪郭) を決め、後半で緩めて細部を追従させる
  const lambdaSchedule = [3, 1.7, 1, 0.55, 0.3];

  for (let iter = 0; iter < FIT_ITERATIONS; iter++) {
    // z拘束のスケール/オフセット: 現在の形状zとMediaPipe zの重み付き1D最小二乗。
    // 大域の奥行きスケール差はszZ/tzZに吸収させ、係数には「相対的な凹凸」だけ効かせる
    const shaped = shapedLm(coeffs);
    let szZ = 1;
    let tzZ = 0;
    {
      let n = 0;
      let sd = 0;
      let st = 0;
      let sdd = 0;
      let sdt = 0;
      for (let m = 0; m < lmCount; m++) {
        const w = fitWeights ? fitWeights[m] : 1;
        const d = shaped[m * 3 + 2];
        const t = targets[m * 3 + 2];
        n += w;
        sd += d * w;
        st += t * w;
        sdd += d * d * w;
        sdt += d * t * w;
      }
      const denom = n * sdd - sd * sd;
      if (Math.abs(denom) > 1e-9) {
        szZ = (n * sdt - sd * st) / denom;
        tzZ = (st - szZ * sd) / n;
      }
    }

    // 相似固定で絶対係数を解く: sim(meanLm + B c) = sim(meanLm) + sR·(B c)
    // → A c = targets − sim(meanLm)。x,y + 弱いz (3×M行)。
    const rows = lmCount * 3;
    const A = new Float64Array(rows * k);
    const r = new Float64Array(rows);
    for (let m = 0; m < lmCount; m++) {
      const gx = meanLm[m * 3];
      const gy = meanLm[m * 3 + 1];
      const gz = meanLm[m * 3 + 2];
      r[m * 3] = targets[m * 3] - (sim.s * (sim.cos * gx - sim.sin * gy) + sim.tx);
      r[m * 3 + 1] = targets[m * 3 + 1] - (sim.s * (sim.sin * gx + sim.cos * gy) + sim.ty);
      r[m * 3 + 2] = targets[m * 3 + 2] - (szZ * gz + tzZ);
      for (let i = 0; i < k; i++) {
        const bx = lmBasis[(i * lmCount + m) * 3];
        const by = lmBasis[(i * lmCount + m) * 3 + 1];
        const bz = lmBasis[(i * lmCount + m) * 3 + 2];
        A[m * 3 * k + i] = sim.s * (sim.cos * bx - sim.sin * by);
        A[(m * 3 + 1) * k + i] = sim.s * (sim.sin * bx + sim.cos * by);
        A[(m * 3 + 2) * k + i] = szZ * bz;
      }
    }
    // 重み付き正規方程式 ((AᵀWA + λI)/Σw基準) c = AᵀWr/Σw
    const ata = new Float64Array(k * k);
    const atr = new Float64Array(k);
    let wTotal = 0;
    for (let row = 0; row < rows; row++) {
      const pw = fitWeights ? fitWeights[(row / 3) | 0] : 1;
      const rw = row % 3 === 2 ? pw * Z_CONSTRAINT_WEIGHT : pw; // z行は弱く
      wTotal += rw;
      const rv = r[row] * rw;
      for (let i = 0; i < k; i++) {
        const av = A[row * k + i];
        atr[i] += av * rv;
        for (let j = 0; j <= i; j++) ata[i * k + j] += av * A[row * k + j] * rw;
      }
    }
    // Σwで正規化する — しないと対応点数に比例してAᵀAが育ち、
    // λの実効強度が点数で変わってしまう (68点→458点で1/6に弱まる)
    const invW = 1 / Math.max(1e-12, wTotal);
    for (let i = 0; i < k * k; i++) ata[i] *= invW;
    for (let i = 0; i < k; i++) atr[i] *= invW;
    for (let i = 0; i < k; i++) {
      for (let j = i + 1; j < k; j++) ata[i * k + j] = ata[j * k + i];
      // 係数はz-scoreスケール (公式デモが±3スライダーで直接使用しており確認済み) なので
      // 一様な単位行列リッジがガウス事前分布として正しい
      ata[i * k + i] += identityReg * REG_UNIT * lambdaSchedule[Math.min(iter, lambdaSchedule.length - 1)];
    }
    const solved = solveSPD(ata, atr, k);
    const next = new Float32Array(k);
    for (let i = 0; i < k; i++) next[i] = Math.min(COEFF_CLAMP, Math.max(-COEFF_CLAMP, solved[i]));
    coeffs = next;

    // 係数を反映した形状で相似を取り直す
    sim = fitSimilarity2D(shapedLm(coeffs), targets, lmCount, fitWeights);
  }

  // 最終頂点の生成とtzの決定: 鼻先(30)のzを既存reliefの鼻位置感覚(≈0.1)に合わせ、
  // カメラフレーミングをGRIDバックエンドと揃える。
  const vertices = applyIdentity(model, coeffs);
  const finalLm = gnmLandmarkPositions(model, vertices);
  const noseZGnm = finalLm[30 * 3 + 2];
  sim = { ...sim, tz: 0.1 - sim.s * noseZGnm };
  applySimilarityInPlace(vertices, sim);

  const landmarkZ = new Float32Array(MEDIAPIPE_IBUG68.length);
  for (let m = 0; m < landmarkZ.length; m++) landmarkZ[m] = sim.s * finalLm[m * 3 + 2] + sim.tz;

  let centerZ = 0;
  for (let i = 2; i < vertices.length; i += 3) centerZ += vertices[i];
  centerZ /= model.vertexCount;

  return { vertices, coeffs, sim, landmarkZ, centerZ };
}
