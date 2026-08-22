// GNMフィット後の残差ワープ。
//
// identity係数 (統計モデル) では張り切れない目・唇の位置残差を、ランドマークの
// 残差ベクトル場としてneutral頂点へ焼き込む。まばたき・開口が「写真の目・口の位置」で
// 起きるようにする (表情basisはワープ後頂点へそのまま加算。数px・XYのみ・目唇局所の
// 範囲なら一次近似として整合する)。フィット自体には手を入れない。

import type { NormalizedFaceLandmark } from './faceTopology';
import { MEDIAPIPE_IBUG68, type GnmFitResult, type GnmModel } from './gnmHead';

// MediaPipe意味領域 (tools/export_gnm_assets.py と同一の定数)。
// 密対応があれば唇40点・目32点の全点がワープのアンカーになる (68点対応表では部分集合)
// (ランドマーク重畳デバッグ表示の色分けにも使うためexport)
export const MP_LIPS = new Set([
  61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
  95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78,
]);
export const MP_EYES = new Set([
  33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
  263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
]);
const MP_BROWS = new Set([
  70, 63, 105, 66, 107, 46, 53, 52, 65, 55, 300, 293, 334, 296, 336, 276, 283, 282, 295, 285,
]);
const MP_NOSE = new Set([1, 2, 4, 5, 6, 19, 94, 97, 98, 326, 327, 168, 195, 197]);

// ワープの局所化重み: 表情で動く目・唇を完全一致させ、鼻は弱く追従。
// 顎ライン・輪郭は0 — 広域をワープすると統計形状の滑らかさを壊すリスクの方が大きい
function warpWeight(mpIdx: number): number {
  if (MP_EYES.has(mpIdx) || MP_LIPS.has(mpIdx)) return 1.0;
  if (MP_BROWS.has(mpIdx)) return 0.8;
  if (MP_NOSE.has(mpIdx)) return 0.5;
  return 0;
}

// 内唇リングの上下ペア (口角78/308を除くMediaPipe標準トポロジ)。
// 開いた口の写真では内唇ランドマークが上下に離れており、そのままワープすると
// GNMの閉じた唇シームが裂ける。上下ペアの中点 (=写真の歯の中心線) を共通の
// 目標にしてシームを閉じたまま移動する
const INNER_LIP_PAIRS: [number, number][] = [
  [191, 95],
  [80, 88],
  [81, 178],
  [82, 87],
  [13, 14],
  [312, 317],
  [311, 402],
  [310, 318],
  [415, 324],
];
const INNER_LIP_PARTNER = new Map<number, number>();
for (const [a, b] of INNER_LIP_PAIRS) {
  INNER_LIP_PARTNER.set(a, b);
  INNER_LIP_PARTNER.set(b, a);
}

const WARP_SIGMA = 0.045; // ガウス核の幅 (モデル空間, faceWidth≈1)
const WARP_KAPPA = 0.3; // Shepard分母の正則化 (対応点から離れるとワープ→0)
const WARP_CLAMP = 0.04; // ワープ量上限 (モデル空間 ≈ faceWidthの4%)
// 反復回数。2回にするとκ正則化の塗り残しが減り位置精度が上がるが、
// 変形も強くなる (口の形の自然さ優先で1回にしている)。
// 注意: 唇だけ狭い核 (σ=0.016〜0.022) で開口シームをさらに追い込む案は試したが、
// どの強度でも口の形が徐々に崩れたため全て撤回した (開口位置の残差は許容し、
// 唇の形の自然さを優先する)
const WARP_PASSES = 1;

// まばたき変位量の補正に使う瞼上下ペア (MediaPipe index)
const APERTURE_PAIRS = [
  { top: 160, bottom: 144 },
  { top: 158, bottom: 153 }, // 画像向かって左の目
  { top: 385, bottom: 380 },
  { top: 387, bottom: 373 }, // 画像向かって右の目
];

/**
 * 鼻孔の内壁頂点を近傍平均のラプラシアン平滑化で膜状に塞ぐ。
 * 穴のジオメトリは角度によって黒い穴/影として破綻するが、写真テクスチャの
 * 鼻孔の暗さだけで見た目は十分なため、凹みを均して閉じてしまう。
 * 縁 (nostrilWeight=0の隣接頂点) は動かさないので鼻の表面とは連続。
 */
/**
 * 眼球貫通の拘束。表情で動くのは瞼だけで、眼球頂点は表情基底上まったく動かない
 * (実測: 目領域20成分すべてで眼球頂点の基底変位=0) ため、瞼が眼球の内側へ
 * 潜り込むと眼球が瞼を貫通して見える。これは表情の組み合わせ・写真のフィット・
 * 残差ワープのいずれからも起こりうるので、原因側を塞ぐのではなく
 * 「瞼は眼球の外側」という不変条件を毎フレーム保証する。
 *
 * 眼球は球で十分に近似できる (実測: 半径0.107に対し平均残差0.0025=2.3%)。
 * 各頂点の許容最小距離はneutral時の距離で下限を取るため、neutralでは
 * 一切動かず (見た目不変)、そこより深く潜ることだけを禁じる。
 */
export interface EyeballContainment {
  /** 表情適用後の頂点配列 (neutral基準の未変換空間) をin-placeで補正する。 */
  apply(vertices: Float32Array): void;
}

const EYE_CLEARANCE = 0.0025; // 眼球表面からの最小離隔 (モデル空間, faceWidth=1)
const EYE_CANDIDATE_SCALE = 1.9; // 拘束対象にする「中心からの距離 / 半径」上限

export function buildEyeballContainment(
  model: GnmModel,
  neutralVertices: Float32Array,
): EyeballContainment | null {
  const eyeIdx: number[] = [];
  for (let i = 0; i < model.vertexCount; i++) if (model.eyeWeight[i] > 128) eyeIdx.push(i);
  if (eyeIdx.length < 32) return null;

  // 左右に分ける (モデル空間の符号で判定。GNMの左右規約に依存しない)
  const groups: number[][] = [[], []];
  for (const i of eyeIdx) groups[neutralVertices[i * 3] < 0 ? 0 : 1].push(i);

  const spheres: { cx: number; cy: number; cz: number; r: number; targets: Int32Array; floors: Float32Array }[] = [];
  for (const group of groups) {
    if (group.length < 16) continue;
    const s = fitSphere(neutralVertices, group);
    if (!s) continue;
    // 拘束対象は眼球以外 (瞼・目周りの肌) で球の近傍にあるもの
    const targets: number[] = [];
    const floors: number[] = [];
    const limit = s.r * EYE_CANDIDATE_SCALE;
    for (let i = 0; i < model.vertexCount; i++) {
      if (model.eyeWeight[i] > 128) continue;
      const d = Math.hypot(
        neutralVertices[i * 3] - s.cx,
        neutralVertices[i * 3 + 1] - s.cy,
        neutralVertices[i * 3 + 2] - s.cz,
      );
      if (d > limit) continue;
      targets.push(i);
      // neutralで既に球内にある頂点はその距離を下限にする (neutralの見た目を変えない)
      floors.push(Math.min(d, s.r + EYE_CLEARANCE));
    }
    if (targets.length === 0) continue;
    spheres.push({ ...s, targets: new Int32Array(targets), floors: new Float32Array(floors) });
  }
  if (spheres.length === 0) return null;

  return {
    apply(vertices: Float32Array): void {
      for (const s of spheres) {
        for (let k = 0; k < s.targets.length; k++) {
          const i = s.targets[k];
          const dx = vertices[i * 3] - s.cx;
          const dy = vertices[i * 3 + 1] - s.cy;
          const dz = vertices[i * 3 + 2] - s.cz;
          const d = Math.hypot(dx, dy, dz);
          const floor = s.floors[k];
          if (d >= floor || d < 1e-6) continue;
          const scale = floor / d;
          vertices[i * 3] = s.cx + dx * scale;
          vertices[i * 3 + 1] = s.cy + dy * scale;
          vertices[i * 3 + 2] = s.cz + dz * scale;
        }
      }
    },
  };
}

/** 頂点群へ球を最小二乗フィットする (|p|²-2p·c+|c|²=r² の線形化)。 */
function fitSphere(
  vertices: Float32Array,
  idx: number[],
): { cx: number; cy: number; cz: number; r: number } | null {
  // 4x4正規方程式 (未知数: cx, cy, cz, r²-|c|²)
  const A = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ];
  const b = [0, 0, 0, 0];
  for (const i of idx) {
    const x = vertices[i * 3];
    const y = vertices[i * 3 + 1];
    const z = vertices[i * 3 + 2];
    const row = [2 * x, 2 * y, 2 * z, 1];
    const q = x * x + y * y + z * z;
    for (let p = 0; p < 4; p++) {
      for (let s = 0; s < 4; s++) A[p][s] += row[p] * row[s];
      b[p] += row[p] * q;
    }
  }
  // ガウス消去 (部分ピボット)
  const m = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < 4; c++) {
    let piv = c;
    for (let r = c + 1; r < 4; r++) if (Math.abs(m[r][c]) > Math.abs(m[piv][c])) piv = r;
    if (Math.abs(m[piv][c]) < 1e-12) return null;
    [m[c], m[piv]] = [m[piv], m[c]];
    for (let r = 0; r < 4; r++) {
      if (r === c) continue;
      const f = m[r][c] / m[c][c];
      for (let k = c; k < 5; k++) m[r][k] -= f * m[c][k];
    }
  }
  const cx = m[0][4] / m[0][0];
  const cy = m[1][4] / m[1][1];
  const cz = m[2][4] / m[2][2];
  const r2 = m[3][4] / m[3][3] + cx * cx + cy * cy + cz * cz;
  if (!(r2 > 1e-9)) return null;
  return { cx, cy, cz, r: Math.sqrt(r2) };
}

export function fillNostrils(model: GnmModel, vertices: Float32Array): void {
  if (model.nostrilWeight.length === 0) return;
  const targets: number[] = [];
  for (let i = 0; i < model.vertexCount; i++) {
    if (model.nostrilWeight[i] >= 128) targets.push(i);
  }
  if (targets.length === 0) return;

  // 対象頂点の隣接リスト (三角形から構築。対象以外の隣接=縁も平均には含める)
  const isTarget = new Uint8Array(model.vertexCount);
  for (const i of targets) isTarget[i] = 1;
  const neighbors = new Map<number, number[]>();
  for (const i of targets) neighbors.set(i, []);
  const tris = model.triangles;
  for (let t = 0; t < tris.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      const a = tris[t + e];
      const b = tris[t + ((e + 1) % 3)];
      if (isTarget[a]) neighbors.get(a)!.push(b);
      if (isTarget[b]) neighbors.get(b)!.push(a);
    }
  }

  for (let pass = 0; pass < 20; pass++) {
    const next = new Map<number, [number, number, number]>();
    for (const i of targets) {
      const ns = neighbors.get(i)!;
      if (ns.length === 0) continue;
      let x = 0;
      let y = 0;
      let z = 0;
      for (const nIdx of ns) {
        x += vertices[nIdx * 3];
        y += vertices[nIdx * 3 + 1];
        z += vertices[nIdx * 3 + 2];
      }
      next.set(i, [x / ns.length, y / ns.length, z / ns.length]);
    }
    for (const [i, p] of next) {
      vertices[i * 3] = p[0];
      vertices[i * 3 + 1] = p[1];
      vertices[i * 3 + 2] = p[2];
    }
  }
}

/**
 * ランドマークの残差 (写真位置 − フィット済表面位置) をXYのみ・目唇局所の
 * ガウス核Shepard補間で fit.vertices へ焼き込む。
 * 対応点は密対応 (唇40点・目32点を含む458点) があればそれを、無ければ68点対応表を使う。
 * κ正則化と逆向き残差の相殺で1回では取り切れないため、適用→再計測→適用を
 * WARP_PASSES回反復してほぼ収束させる。
 * 返り値: 表情成分ごとの振幅スケール。ワープで瞼開口幅が変わると、GNM開口幅を
 * 基準に定義された開閉変位が「閉じきらない/閉じすぎる」ため、
 * 目領域成分を (写真の開口幅 / フィット表面の開口幅) 比でスケールする。
 */
export function applyResidualWarp(
  model: GnmModel,
  fit: GnmFitResult,
  landmarks: NormalizedFaceLandmark[],
  strength: number,
): Float32Array {
  const exprScales = new Float32Array(Math.max(1, model.expressionCount)).fill(1);
  if (strength <= 0) return exprScales;
  const verts = fit.vertices;

  // 対応表 (MediaPipe index → GNM表面のbarycentric) を選ぶ
  const useDense = model.denseCount > 0;
  const corrCount = useDense ? model.denseCount : MEDIAPIPE_IBUG68.length;
  const corrIdx = useDense ? model.denseTriIndices : model.landmarkIndices;
  const corrBary = useDense ? model.denseBaryWeights : model.landmarkWeights;
  const corrMp = (k: number) => (useDense ? model.denseMpIndices[k] : MEDIAPIPE_IBUG68[k]);

  const surfaceY = new Map<number, number>(); // 開口幅計測用 (ワープ前=1パス目の表面y)

  // 対応点の現在の表面位置と残差を集める (mpFilterで領域を絞れる)
  const collectAnchors = (
    firstPass: boolean,
    mpFilter?: (mp: number) => boolean,
  ): { x: number; y: number; rx: number; ry: number; w: number }[] => {
    const anchors: { x: number; y: number; rx: number; ry: number; w: number }[] = [];
    for (let k = 0; k < corrCount; k++) {
      const mp = corrMp(k);
      let px = 0;
      let py = 0;
      for (let j = 0; j < 3; j++) {
        const vi = corrIdx[k * 3 + j];
        const bw = corrBary[k * 3 + j];
        px += verts[vi * 3] * bw;
        py += verts[vi * 3 + 1] * bw;
      }
      if (firstPass && !surfaceY.has(mp)) surfaceY.set(mp, py);
      if (mpFilter && !mpFilter(mp)) continue;
      const w = warpWeight(mp);
      if (w === 0) continue;
      const lm = landmarks[mp];
      const partner = INNER_LIP_PARTNER.get(mp);
      const targetX = partner !== undefined ? (lm.x + landmarks[partner].x) / 2 : lm.x;
      const targetY = partner !== undefined ? (lm.y + landmarks[partner].y) / 2 : lm.y;
      anchors.push({ x: px, y: py, rx: targetX - px, ry: targetY - py, w });
    }
    return anchors;
  };

  // 残差場を頂点へ塗る (Shepard補間)
  const applyField = (
    anchors: { x: number; y: number; rx: number; ry: number; w: number }[],
    sigma: number,
    kappa: number,
  ): void => {
    if (anchors.length < 4) return;
    const supportSq = (3 * sigma) ** 2;
    const invTwoSigmaSq = 1 / (2 * sigma * sigma);
    for (let i = 0; i < model.vertexCount; i++) {
      const vx = verts[i * 3];
      const vy = verts[i * 3 + 1];
      let numX = 0;
      let numY = 0;
      let den = 0;
      for (const a of anchors) {
        const dx = vx - a.x;
        const dy = vy - a.y;
        const dsq = dx * dx + dy * dy;
        if (dsq > supportSq) continue;
        const k = Math.exp(-dsq * invTwoSigmaSq) * a.w;
        numX += k * a.rx;
        numY += k * a.ry;
        den += k;
      }
      if (den < 1e-6) continue;
      let wx = (strength * numX) / (den + kappa);
      let wy = (strength * numY) / (den + kappa);
      const mag = Math.hypot(wx, wy);
      if (mag > WARP_CLAMP) {
        wx *= WARP_CLAMP / mag;
        wy *= WARP_CLAMP / mag;
      }
      verts[i * 3] += wx;
      verts[i * 3 + 1] += wy;
    }
  };

  // 広域パス: 目・唇・眉・鼻の全アンカー
  for (let pass = 0; pass < WARP_PASSES; pass++) {
    applyField(collectAnchors(pass === 0), WARP_SIGMA, WARP_KAPPA);
  }

  // --- 目領域成分の振幅スケール ---
  if (model.expressionCount > 0) {
    // GNMの"left_eye"がモデル空間±xどちら側かをbasis変位の重心xから判定する
    // (GNMの左右規約に依存しないため)
    const sideSignOf = (prefix: string): number => {
      const ci = model.expressionNames.findIndex((n) => n.startsWith(prefix));
      if (ci < 0) return 0;
      const base = ci * model.vertexCount * 3;
      let wSum = 0;
      let xSum = 0;
      for (let i = 0; i < model.vertexCount; i++) {
        const d =
          Math.abs(model.expressionBasisQ[base + i * 3]) +
          Math.abs(model.expressionBasisQ[base + i * 3 + 1]) +
          Math.abs(model.expressionBasisQ[base + i * 3 + 2]);
        wSum += d;
        xSum += d * model.positions[i * 3];
      }
      return wSum > 0 ? Math.sign(xSum / wSum) : 0;
    };
    const leftSign = sideSignOf('left_eye');

    // 左右それぞれ、ペア2組の平均で開口比を求める
    for (const sideSign of [-1, 1]) {
      let photoSum = 0;
      let surfaceSum = 0;
      for (const pair of APERTURE_PAIRS) {
        const pairSign = Math.sign(landmarks[pair.top].x + landmarks[pair.bottom].x);
        if (pairSign !== sideSign) continue;
        const sy0 = surfaceY.get(pair.top);
        const sy1 = surfaceY.get(pair.bottom);
        if (sy0 === undefined || sy1 === undefined) continue;
        surfaceSum += Math.abs(sy0 - sy1);
        photoSum += Math.abs(landmarks[pair.top].y - landmarks[pair.bottom].y);
      }
      if (surfaceSum < 1e-6) continue;
      const scale = Math.min(1.6, Math.max(0.6, photoSum / surfaceSum));
      for (let ci = 0; ci < model.expressionCount; ci++) {
        const name = model.expressionNames[ci] ?? '';
        const isLeft = name.startsWith('left_eye');
        const isRight = name.startsWith('right_eye');
        if (!isLeft && !isRight) continue;
        const compSign = isLeft ? leftSign : -leftSign;
        if (compSign !== 0 && compSign === sideSign) exprScales[ci] = scale;
      }
    }
  }

  return exprScales;
}
