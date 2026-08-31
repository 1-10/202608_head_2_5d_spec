// 旧実装の法線融合。Depth由来Zをアンカーに120回のJacobi反復を行う。

import { Rect, makeField, sampleField } from '../field';
import { smoothstep } from '../ramp';
import { Grid2d } from './grid';

/** 旧実装の LAMBDA / MAX_STEP / ITERATIONS。MAX_STEPだけメートル換算。 */
export const LEGACY_DATA_WEIGHT = 0.08;
export const LEGACY_MAX_STEP_METERS = 0.00745;
export const LEGACY_ITERATIONS = 120;

/**
 * 旧実装と同じ法線融合後のZとアンカーZとの差を返す。
 *
 * 旧実装は法線を回転せず、96x120格子で反復数も固定だったため、相似変換と基準格子サイズは
 * 受け取らない（現行の他の段と引数を揃える必要が無い）。
 *
 * @param normal (3, h, w) DAViD の生の法線。`rect` の覆う範囲に乗る
 * @param u / v (rows, columns) 格子点が指す画像 UV
 * @param weight (rows, columns) 髪らしさ（`presenceMean`）
 * @param anchorZ (rows, columns) 融合の事前値になる z
 */
export function reliefFromNormal(input: {
  normal: Float32Array;
  normalWidth: number;
  normalHeight: number;
  rect: Rect;
  u: Float64Array;
  v: Float64Array;
  grid: Grid2d;
  weight: Float64Array;
  anchorZ: Float64Array;
  strength?: number;
  dataWeight?: number;
  maxStepMeters?: number;
  iterations?: number;
}): Float64Array {
  const strength = input.strength ?? 1;
  const dataWeight = input.dataWeight ?? LEGACY_DATA_WEIGHT;
  const maxStepMeters = input.maxStepMeters ?? LEGACY_MAX_STEP_METERS;
  const iterations = input.iterations ?? LEGACY_ITERATIONS;
  const { grid } = input;
  const area = input.normalWidth * input.normalHeight;
  if (input.normal.length !== area * 3) {
    throw new Error(`normal の形が (3,h,w) でない: ${input.normal.length}`);
  }
  const pointCount = grid.rows * grid.columns;
  for (const [name, values] of [
    ['u', input.u],
    ['v', input.v],
    ['weight', input.weight],
    ['anchorZ', input.anchorZ],
  ] as const) {
    if (values.length !== pointCount) {
      throw new Error(`${name} の形が格子と合わない: ${values.length}`);
    }
  }

  const channels = [0, 1, 2].map((channel) =>
    makeField(
      input.normal.subarray(channel * area, (channel + 1) * area),
      input.normalWidth,
      input.normalHeight,
      input.rect,
    ),
  );

  // 現行Grid2dはrowがY上向きに増える。上下反転すると旧実装のrow0=上、row+1=下と一致する。
  const rows = grid.rows;
  const columns = grid.columns;
  const flip = (row: number): number => rows - 1 - row;

  const trust = new Float64Array(pointCount);
  const gx = new Float64Array(pointCount);
  const gy = new Float64Array(pointCount);
  const anchor = new Float64Array(pointCount);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const source = flip(row) * columns + column;
      const target = row * columns + column;
      const u = input.u[source];
      const v = input.v[source];
      const nx = sampleField(channels[0], u, v);
      const ny = sampleField(channels[1], u, v);
      const nz = sampleField(channels[2], u, v);
      const mask = input.weight[source];
      const trustValue = smoothstep(0.35, 0.6, nz) * smoothstep(0.25, 0.55, mask);
      trust[target] = trustValue;
      const nzSafe = Math.max(0.35, nz);
      gx[target] = trustValue > 0 ? -nx / nzSafe : 0;
      gy[target] = trustValue > 0 ? -ny / nzSafe : 0;
      anchor[target] = input.anchorZ[source];
    }
  }

  let z = Float64Array.from(anchor);
  let nextZ = new Float64Array(pointCount);
  const total = new Float64Array(pointCount);
  const totalWeight = new Float64Array(pointCount);
  const clampStep = (value: number): number =>
    Math.min(maxStepMeters, Math.max(-maxStepMeters, value));

  for (let iteration = 0; iteration < Math.max(0, iterations); iteration++) {
    for (let index = 0; index < pointCount; index++) {
      total[index] = dataWeight * anchor[index];
      totalWeight[index] = dataWeight;
    }
    // 列方向（x）。
    for (let row = 0; row < rows; row++) {
      for (let column = 1; column < columns; column++) {
        const right = row * columns + column;
        const left = right - 1;
        const edge = Math.min(trust[right], trust[left]) * strength;
        const step = clampStep(0.5 * (gx[right] + gx[left]) * grid.xStep * edge);
        total[right] += z[left] + step;
        totalWeight[right] += 1;
        total[left] += z[right] - step;
        totalWeight[left] += 1;
      }
    }
    // 行方向（y）。旧実装の row+1 は下なので、反転済みの配列でそのまま隣を見る。
    for (let row = 1; row < rows; row++) {
      for (let column = 0; column < columns; column++) {
        const lower = row * columns + column;
        const upper = lower - columns;
        const edge = Math.min(trust[lower], trust[upper]) * strength;
        const step = clampStep(0.5 * (gy[lower] + gy[upper]) * grid.yStep * edge);
        total[lower] += z[upper] - step;
        totalWeight[lower] += 1;
        total[upper] += z[lower] + step;
        totalWeight[upper] += 1;
      }
    }
    for (let index = 0; index < pointCount; index++) nextZ[index] = total[index] / totalWeight[index];
    const swap = z;
    z = nextZ;
    nextZ = swap;
  }

  // 上下反転を戻して返す。
  const out = new Float64Array(pointCount);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      out[flip(row) * columns + column] =
        z[row * columns + column] - anchor[row * columns + column];
    }
  }
  return out;
}
