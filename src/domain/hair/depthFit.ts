// 段1: DAViD の相対 Depth を GNM 空間の z スケールへ合わせる。
//
// DAViD の Depth は相対値（スケールもオフセットも不定）なので、そのままでは殻の厚みをメートルで
// 語れない。**Depth が主役で法線は飾り**という方針の根拠がここにある: 法線は傾きしか持たないので
// 積分すると誤差が累積して髪全体が倒れる。Depth を絶対量として使えるようにする対価がこのフィット
// 1段。
//
//     z_gnm = scale × depth + offset
//
// サンプルの取り方
// ----------------
// GNM の頂点を相似変換で写真へ投影し、その位置の Depth と頂点の z を対応させる。ただし信用できる
// 対応だけを残す:
//
//   - **前を向いている面だけ**（法線 z が閾値以上）。側面・背面の Depth は視線方向にほぼ平行な面を
//     見ているので、わずかな投影誤差が大きな深度差になる
//   - **手前にある頂点だけ**。眼球・口腔・歯は前を向いているが皮膚の裏に隠れている
//   - **髪マスクの外だけ**。髪の Depth は頭皮の z ではない
//
// 外れ値への強さ: RANSAC ではなく**トリム最小二乗**（LTS）
// --------------------------------------------------------
// 求めるのは 1 次式の 2 パラメータで、サンプルは数千点あり、上の3条件を通った時点で多数が正常。
// この条件なら LTS が RANSAC と同等に効き、かつ**乱数を使わないので出力が決定的**になる。

import { ScalarField, insideRect, sampleField, uSpan, vSpan } from '../field';
import { Similarity2d } from '../gnm/fit';
import { Grid2d, gridOverRect, nearestGridPoint, gridCellSize } from './grid';
import { gnmXyToImageUv } from './projection';
import { rasterizeFrontZ } from './scalp';

/** 採用する頂点法線の z の下限（単位法線）。約 70 度より正面を向いている面だけ使う。 */
export const FRONT_FACING_MIN_NORMAL_Z = 0.35;

/**
 * 可視判定に使う z バッファの一辺の格子点数（頭部の XY bbox を覆う）。
 *
 * セルの寸法が GNM の三角形の辺長と同程度になる点数。これより粗いと、隠れている頂点と表面の
 * 頂点を同じセルに落として区別できなくなる。
 */
export const VISIBILITY_GRID_SIDE = 128;

/**
 * 可視と見なす z の許容差（可視判定格子のセル何個分か）。
 *
 * z バッファは格子点で表面を拾うので、格子点の間にある頂点は表面の傾きの分だけバッファより後ろに
 * 出る。傾き 45 度でセル 1 個分ずれるため 2 セル分を許容する。
 */
export const VISIBILITY_TOLERANCE_CELLS = 2.0;

/** トリム最小二乗で毎周落とす割合（残差の大きい側から）。 */
export const TRIM_FRACTION = 0.25;

/** トリム最小二乗の反復回数。1 周目は全点、以降は前周の inlier で解き直す。 */
export const TRIM_ITERATIONS = 4;

/** フィットに必要なサンプル数の下限。これを下回る写真は顔が写っていない扱い。 */
export const MIN_FIT_SAMPLES = 64;

/** Depth の分散がこれ以下ならスケールを決められない（一定値の Depth）。 */
export const MIN_DEPTH_SPREAD = 1e-6;

/** 旧実装が Depth フィットに要求した 68 点中の最小標本数。 */
export const LEGACY_MIN_LANDMARK_SAMPLES = 20;

/** Depth を GNM の z へ合わせられない（顔の前面が見えていない / Depth が一定）。 */
export class DepthFitError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'DepthFitError';
  }
}

/** Depth → GNM z の 1 次式と、その当てはまり具合。 */
export interface DepthZFit {
  /** メートル / Depth 単位。DAViD の Depth の符号は不定なので負にもなる。 */
  readonly scale: number;
  readonly offset: number;
  /** inlier の残差 RMS（メートル）。検査の指標。 */
  readonly residualRmsMeters: number;
  /** 3 条件を通ったサンプル数。 */
  readonly sampleCount: number;
  /** 最終周で採用されたサンプル数。 */
  readonly inlierCount: number;
  /**
   * フィットに使った Depth 値の範囲。**この外側は外挿。** 髪は顔より手前にあるので、髪の Depth は
   * ほぼ必ずこの範囲の外に出る。範囲がどれだけ狭いかが「この1次式をどこまで伸ばして信じるか」を
   * 決める。
   */
  readonly inlierDepthMin: number;
  readonly inlierDepthMax: number;
}

/** Depth 値を GNM 空間の z（メートル）へ写す。 */
export function depthToZ(fit: DepthZFit, depth: number): number {
  return fit.scale * depth + fit.offset;
}

/** フィットに使った Depth の幅。傾きを決めるてこの長さ。 */
export function inlierDepthSpan(fit: DepthZFit): number {
  return fit.inlierDepthMax - fit.inlierDepthMin;
}

/**
 * 渡した Depth が、フィット範囲の何倍の距離まで外挿になっているか。
 *
 * 0 なら全部が範囲の内側（内挿）。1 なら範囲の幅と同じだけ外へ出ている。**大きいほど厚みが
 * 信用できない。**
 */
export function extrapolationRatio(fit: DepthZFit, depths: Float64Array): number {
  const span = inlierDepthSpan(fit);
  if (span <= 0) return Infinity;
  if (depths.length === 0) return 0;
  let beyond = 0;
  for (const depth of depths) {
    beyond = Math.max(beyond, fit.inlierDepthMin - depth, depth - fit.inlierDepthMax);
  }
  return Math.max(0, beyond) / span;
}

/**
 * 旧実装と同じく、iBUG 68 点だけで単純な一次最小二乗を解く。
 *
 * 外向き法線・可視判定・髪除外・トリムは行わない。持ち上げ量を写真ごとに出すため、旧実装に
 * なかった残差 RMS だけを同じ標本から追加で計算する。
 */
export function fitDepthToLandmarkZ(
  depth: ScalarField,
  photoLandmarksPixels: Float64Array,
  targetZ: Float64Array,
  imageSize: readonly [number, number],
): DepthZFit {
  if (photoLandmarksPixels.length !== 68 * 2 || targetZ.length !== 68) {
    throw new Error(
      '旧Depthフィットは landmarks=(68,2), z=(68,): ' +
        `${photoLandmarksPixels.length / 2}, ${targetZ.length}`,
    );
  }
  const [width, height] = imageSize;
  const depthValues: number[] = [];
  const targets: number[] = [];
  for (let point = 0; point < 68; point++) {
    const u = photoLandmarksPixels[point * 2] / width;
    const v = photoLandmarksPixels[point * 2 + 1] / height;
    if (!insideRect(depth.rect, u, v)) continue;
    depthValues.push(sampleField(depth, u, v));
    targets.push(targetZ[point]);
  }
  if (depthValues.length < LEGACY_MIN_LANDMARK_SAMPLES) {
    throw new DepthFitError(
      `旧Depthフィットのランドマークが足りない: ${depthValues.length} 点` +
        `（下限 ${LEGACY_MIN_LANDMARK_SAMPLES}）`,
    );
  }
  const depthArray = Float64Array.from(depthValues);
  const targetArray = Float64Array.from(targets);
  const { scale, offset } = solveLine(depthArray, targetArray);
  let squared = 0;
  for (let sample = 0; sample < depthArray.length; sample++) {
    const residual = targetArray[sample] - (scale * depthArray[sample] + offset);
    squared += residual * residual;
  }
  return {
    scale,
    offset,
    residualRmsMeters: Math.sqrt(squared / depthArray.length),
    sampleCount: depthArray.length,
    inlierCount: depthArray.length,
    inlierDepthMin: Math.min(...depthArray),
    inlierDepthMax: Math.max(...depthArray),
  };
}

/**
 * 面積重みの頂点法線（単位ベクトル、外向き）。
 *
 * 外向きの定義は GNM の巻き順に従う: `(v1 − v0) × (v2 − v0)`。
 */
export function vertexNormals(vertices: Float64Array, triangles: Uint32Array): Float64Array {
  const out = new Float64Array(vertices.length);
  for (let triangle = 0; triangle < triangles.length / 3; triangle++) {
    const a = triangles[triangle * 3];
    const b = triangles[triangle * 3 + 1];
    const c = triangles[triangle * 3 + 2];
    const abx = vertices[b * 3] - vertices[a * 3];
    const aby = vertices[b * 3 + 1] - vertices[a * 3 + 1];
    const abz = vertices[b * 3 + 2] - vertices[a * 3 + 2];
    const acx = vertices[c * 3] - vertices[a * 3];
    const acy = vertices[c * 3 + 1] - vertices[a * 3 + 1];
    const acz = vertices[c * 3 + 2] - vertices[a * 3 + 2];
    const nx = aby * acz - abz * acy;
    const ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;
    for (const vertex of [a, b, c]) {
      out[vertex * 3] += nx;
      out[vertex * 3 + 1] += ny;
      out[vertex * 3 + 2] += nz;
    }
  }
  for (let vertex = 0; vertex < out.length / 3; vertex++) {
    const length = Math.max(
      Math.hypot(out[vertex * 3], out[vertex * 3 + 1], out[vertex * 3 + 2]),
      1e-20,
    );
    out[vertex * 3] /= length;
    out[vertex * 3 + 1] /= length;
    out[vertex * 3 + 2] /= length;
  }
  return out;
}

/** 3x3 近傍の最大値でマスクを引く（膨張）。境界の滲みを内側へ入れないため。 */
export function dilatedMaskSample(
  mask: ScalarField,
  u: number,
  v: number,
  stepU: number,
  stepV: number,
): number {
  let maximum = 0;
  for (const dv of [-1, 0, 1]) {
    for (const du of [-1, 0, 1]) {
      const value = sampleField(mask, u + du * stepU, v + dv * stepV);
      if (value > maximum) maximum = value;
    }
  }
  return maximum;
}

/**
 * 顔の前面で Depth と GNM の z を突き合わせ、`z = scale × depth + offset` を解く。
 *
 * **髪と判定された画素の 3x3 近傍にある頂点を捨てる。** 髪の Depth を頭皮のフィットへ混ぜない
 * ため。判定（`present`）で捨てるので、**「確信度をどこで切るか」という調整可能な値が無い。**
 */
export function fitDepthToGnmZ(input: {
  vertices: Float64Array;
  triangles: Uint32Array;
  similarity: Similarity2d;
  depth: ScalarField;
  hairPresent: ScalarField;
  imageSize: readonly [number, number];
  frontFacingMinNormalZ?: number;
  trimFraction?: number;
  trimIterations?: number;
}): DepthZFit {
  const frontFacingMinNormalZ = input.frontFacingMinNormalZ ?? FRONT_FACING_MIN_NORMAL_Z;
  const trimFraction = input.trimFraction ?? TRIM_FRACTION;
  const trimIterations = input.trimIterations ?? TRIM_ITERATIONS;

  const positions = input.vertices;
  const normals = vertexNormals(positions, input.triangles);
  const vertexCount = positions.length / 3;
  const frontmost = isFrontmost(positions, input.triangles);

  const xy = new Float64Array(vertexCount * 2);
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    xy[vertex * 2] = positions[vertex * 3];
    xy[vertex * 2 + 1] = positions[vertex * 3 + 1];
  }
  const uv = gnmXyToImageUv(xy, input.similarity, input.imageSize);

  const candidates: number[] = [];
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    if (normals[vertex * 3 + 2] < frontFacingMinNormalZ) continue;
    if (frontmost[vertex] === 0) continue;
    const u = uv[vertex * 2];
    const v = uv[vertex * 2 + 1];
    if (u < 0 || u > 1 || v < 0 || v > 1) continue;
    if (!insideRect(input.depth.rect, u, v)) continue;
    candidates.push(vertex);
  }

  const stepU = uSpan(input.hairPresent.rect) / input.hairPresent.width;
  const stepV = vSpan(input.hairPresent.rect) / input.hairPresent.height;
  const kept = candidates.filter(
    (vertex) =>
      dilatedMaskSample(input.hairPresent, uv[vertex * 2], uv[vertex * 2 + 1], stepU, stepV) <= 0,
  );

  if (kept.length < MIN_FIT_SAMPLES) {
    throw new DepthFitError(
      `Depth フィットのサンプルが足りない: ${kept.length} 点（下限 ${MIN_FIT_SAMPLES}）。` +
        '顔の前面が写真に写っていない可能性がある',
    );
  }
  const depthValues = new Float64Array(kept.length);
  const targetZ = new Float64Array(kept.length);
  kept.forEach((vertex, slot) => {
    depthValues[slot] = sampleField(input.depth, uv[vertex * 2], uv[vertex * 2 + 1]);
    targetZ[slot] = positions[vertex * 3 + 2];
  });
  return trimmedLeastSquares(depthValues, targetZ, trimFraction, trimIterations);
}

/**
 * GNM 表面の最前面 z バッファと比べ、隠れていない頂点を 1 で返す。
 *
 * 眼球・口腔・歯は前を向いた面を持つが皮膚の裏にある。法線だけで弾けないので、XY ビンごとの
 * 最前面 z と比べて後ろにいる頂点を落とす。バッファに覆われていない格子点しか近くに無い頂点
 * （輪郭の外側）も落とす — 判断材料が無いため。
 */
function isFrontmost(positions: Float64Array, triangles: Uint32Array): Uint8Array {
  const vertexCount = positions.length / 3;
  let xMin = Infinity;
  let yMin = Infinity;
  let xMax = -Infinity;
  let yMax = -Infinity;
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    xMin = Math.min(xMin, positions[vertex * 3]);
    xMax = Math.max(xMax, positions[vertex * 3]);
    yMin = Math.min(yMin, positions[vertex * 3 + 1]);
    yMax = Math.max(yMax, positions[vertex * 3 + 1]);
  }
  const grid: Grid2d = gridOverRect(
    xMin,
    yMin,
    xMax,
    yMax,
    VISIBILITY_GRID_SIDE,
    VISIBILITY_GRID_SIDE,
  );
  const { z, covered } = rasterizeFrontZ(positions, triangles, grid);
  const tolerance = VISIBILITY_TOLERANCE_CELLS * gridCellSize(grid);
  const out = new Uint8Array(vertexCount);
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    const [column, row] = nearestGridPoint(grid, positions[vertex * 3], positions[vertex * 3 + 1]);
    const index = row * grid.columns + column;
    out[vertex] = covered[index] !== 0 && positions[vertex * 3 + 2] >= z[index] - tolerance ? 1 : 0;
  }
  return out;
}

/** 残差の大きい側を毎周切り落としながら 1 次式を解き直す（LTS）。 */
function trimmedLeastSquares(
  depthValues: Float64Array,
  targetZ: Float64Array,
  trimFraction: number,
  iterations: number,
): DepthZFit {
  if (!(trimFraction >= 0 && trimFraction < 1)) {
    throw new Error(`trimFraction が [0, 1) の外: ${trimFraction}`);
  }
  const sampleCount = depthValues.length;
  let inlier = new Uint8Array(sampleCount).fill(1);
  let scale = 0;
  let offset = 0;

  for (let iteration = 0; iteration < Math.max(1, iterations); iteration++) {
    const selectedDepth: number[] = [];
    const selectedTarget: number[] = [];
    for (let sample = 0; sample < sampleCount; sample++) {
      if (inlier[sample] === 0) continue;
      selectedDepth.push(depthValues[sample]);
      selectedTarget.push(targetZ[sample]);
    }
    ({ scale, offset } = solveLine(
      Float64Array.from(selectedDepth),
      Float64Array.from(selectedTarget),
    ));
    const residual = new Float64Array(sampleCount);
    for (let sample = 0; sample < sampleCount; sample++) {
      residual[sample] = Math.abs(targetZ[sample] - (scale * depthValues[sample] + offset));
    }
    const keepCount = Math.max(MIN_FIT_SAMPLES, Math.round(sampleCount * (1 - trimFraction)));
    if (keepCount >= sampleCount) break;
    // 全サンプルから残差の小さい方を選び直す（前周の inlier に閉じ込めない）。
    const sorted = Array.from(residual).sort((a, b) => a - b);
    const threshold = sorted[keepCount - 1];
    inlier = new Uint8Array(sampleCount);
    for (let sample = 0; sample < sampleCount; sample++) {
      inlier[sample] = residual[sample] <= threshold ? 1 : 0;
    }
  }

  let squared = 0;
  let inlierCount = 0;
  let inlierDepthMin = Infinity;
  let inlierDepthMax = -Infinity;
  for (let sample = 0; sample < sampleCount; sample++) {
    if (inlier[sample] === 0) continue;
    inlierCount++;
    const residual = targetZ[sample] - (scale * depthValues[sample] + offset);
    squared += residual * residual;
    inlierDepthMin = Math.min(inlierDepthMin, depthValues[sample]);
    inlierDepthMax = Math.max(inlierDepthMax, depthValues[sample]);
  }
  return {
    scale,
    offset,
    residualRmsMeters: Math.sqrt(squared / inlierCount),
    sampleCount,
    inlierCount,
    inlierDepthMin,
    inlierDepthMax,
  };
}

/** `z = scale × depth + offset` の最小二乗解（閉形式）。 */
function solveLine(
  depthValues: Float64Array,
  targetZ: Float64Array,
): { scale: number; offset: number } {
  let depthMean = 0;
  let targetMean = 0;
  for (let sample = 0; sample < depthValues.length; sample++) {
    depthMean += depthValues[sample];
    targetMean += targetZ[sample];
  }
  depthMean /= depthValues.length;
  targetMean /= targetZ.length;

  let spread = 0;
  let covariance = 0;
  for (let sample = 0; sample < depthValues.length; sample++) {
    const centered = depthValues[sample] - depthMean;
    spread += centered * centered;
    covariance += centered * (targetZ[sample] - targetMean);
  }
  if (spread <= MIN_DEPTH_SPREAD) {
    throw new DepthFitError(
      `Depth の分散が小さすぎてスケールを決められない（Σ(d − d̄)² = ${spread.toExponential(3)}）`,
    );
  }
  const scale = covariance / spread;
  return { scale, offset: targetMean - scale * depthMean };
}
