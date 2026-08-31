// 頭皮 z バッファ — 格子点ごとの「GNM 表面の最前面 z」。
//
// 髪シェルはこの z をアンカーにして手前へ押し出す。**深度フィットの外挿を z に使ってはいけない。**
// 深度フィットは顔の前面で合わせた1次式なので、頭頂や側頭部では外挿になり過大な z を返す。それを
// 殻の基準にすると殻が頭蓋から浮く。ここが返すのは GNM の実ジオメトリそのものの z。
//
// 覆われていない格子点（頭の輪郭より外）
// --------------------------------------
// 髪は頭の輪郭より外へ張り出すので、格子の一部は GNM 表面に覆われない。そこは「輪郭の z を外へ
// 広げた値」で埋める（`covered` が false として区別できる）。輪郭の外に実在の頭蓋は無いので、
// そこでは殻が z バッファより後ろへ回り込んでよい（縁の巻き込み）。覆われている格子点では後ろへ
// 回り込ませない。この区別のために値と被覆フラグを別に持つ。
//
// 巻き順の規約
// ------------
// GNM の三角形は「外向き法線 = (v1 − v0) × (v2 − v0)」。ここではラスタライズに面の向きを使わない
// ので巻き順に依存しないが、髪シェルが張る三角形は同じ規約に揃える。

import { Grid2d, boxBlur3x3, gridPointCount, gridXCoordinate, gridYCoordinate, makeGrid } from './grid';

/** barycentric の内外判定に許す誤差。隣接三角形が辺を共有する格子点を取りこぼさない幅。 */
export const COVERAGE_EPSILON = 1e-9;

/** 旧実装の頭皮 Z バッファ解像度と全面 box blur 回数。 */
export const LEGACY_SCALP_COLUMNS = 96;
export const LEGACY_SCALP_ROWS = 112;
export const LEGACY_SCALP_BLUR_PASSES = 2;

/** 格子の範囲に GNM 表面が1つも掛かっていない（相似変換かマスクが破綻している）。 */
export class EmptyScalpError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'EmptyScalpError';
  }
}

/** 格子上の頭皮 z。 */
export interface ScalpSurface {
  readonly grid: Grid2d;
  /** (rows, columns) 最前面 z（メートル）。covered が 0 の点は外挿値。 */
  readonly z: Float64Array;
  /** (rows, columns) GNM の三角形が実際に覆っていた格子点。 */
  readonly covered: Uint8Array;
}

/** GNM 表面に覆われた格子点の割合。 */
export function scalpCoverageRatio(scalp: ScalpSurface): number {
  let count = 0;
  for (const value of scalp.covered) if (value !== 0) count++;
  return count / scalp.covered.length;
}

/** 旧実装の 96x112 Z バッファを作り、BFS・2回blur・bilinear参照する。 */
export function buildScalpSurface(
  vertices: Float64Array,
  triangles: Uint32Array,
  grid: Grid2d,
): ScalpSurface {
  const xMax = grid.xMin + grid.xStep * (grid.columns - 1);
  const yMax = grid.yMin + grid.yStep * (grid.rows - 1);
  const spanX = xMax - grid.xMin;
  const spanY = yMax - grid.yMin;
  const bufferGrid = makeGrid(
    grid.xMin + (0.5 * spanX) / LEGACY_SCALP_COLUMNS,
    grid.yMin + (0.5 * spanY) / LEGACY_SCALP_ROWS,
    spanX / LEGACY_SCALP_COLUMNS,
    spanY / LEGACY_SCALP_ROWS,
    LEGACY_SCALP_COLUMNS,
    LEGACY_SCALP_ROWS,
  );
  const buffer = rasterizeFrontZ(vertices, triangles, bufferGrid);
  let anyCovered = false;
  for (const value of buffer.covered) {
    if (value !== 0) {
      anyCovered = true;
      break;
    }
  }
  if (!anyCovered) {
    throw new EmptyScalpError(
      '格子の範囲に GNM 表面が掛かっていない' +
        `（格子 x=${grid.xMin.toFixed(3)}..${xMax.toFixed(3)}` +
        ` y=${grid.yMin.toFixed(3)}..${yMax.toFixed(3)} m）`,
    );
  }
  let filled = legacyNearestFill(buffer.z, buffer.covered, LEGACY_SCALP_ROWS, LEGACY_SCALP_COLUMNS);
  filled = boxBlur3x3(filled, LEGACY_SCALP_ROWS, LEGACY_SCALP_COLUMNS, LEGACY_SCALP_BLUR_PASSES);

  const z = new Float64Array(gridPointCount(grid));
  for (let row = 0; row < grid.rows; row++) {
    const y = gridYCoordinate(grid, row);
    for (let column = 0; column < grid.columns; column++) {
      z[row * grid.columns + column] = sampleLegacyBuffer(
        filled,
        LEGACY_SCALP_ROWS,
        LEGACY_SCALP_COLUMNS,
        gridXCoordinate(grid, column),
        y,
        grid.xMin,
        xMax,
        grid.yMin,
        yMax,
      );
    }
  }
  const { covered } = rasterizeFrontZ(vertices, triangles, grid);
  return { grid, z, covered };
}

/** 旧実装と同じ行優先multi-source BFSで空ビンへ最近傍値をコピーする。 */
function legacyNearestFill(
  z: Float64Array,
  covered: Uint8Array,
  rows: number,
  columns: number,
): Float64Array {
  const result = Float64Array.from(z);
  const known = Uint8Array.from(covered);
  const queue: number[] = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      if (known[row * columns + column] !== 0) queue.push(row * columns + column);
    }
  }
  let head = 0;
  while (head < queue.length) {
    const index = queue[head++];
    const row = Math.floor(index / columns);
    const column = index - row * columns;
    for (const [otherRow, otherColumn] of [
      [row, column - 1],
      [row, column + 1],
      [row - 1, column],
      [row + 1, column],
    ] as const) {
      if (otherRow < 0 || otherRow >= rows || otherColumn < 0 || otherColumn >= columns) continue;
      const other = otherRow * columns + otherColumn;
      if (known[other] !== 0) continue;
      known[other] = 1;
      result[other] = result[index];
      queue.push(other);
    }
  }
  return result;
}

/** 旧Zバッファの `*size - 0.5` 規約でbilinear参照する。 */
function sampleLegacyBuffer(
  values: Float64Array,
  rows: number,
  columns: number,
  x: number,
  y: number,
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number,
): number {
  const fx = Math.min(Math.max(((x - xMin) / (xMax - xMin)) * columns - 0.5, 0), columns - 1.001);
  const fy = Math.min(Math.max(((y - yMin) / (yMax - yMin)) * rows - 0.5, 0), rows - 1.001);
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const tx = fx - x0;
  const ty = fy - y0;
  return (
    (values[y0 * columns + x0] * (1 - tx) + values[y0 * columns + x0 + 1] * tx) * (1 - ty) +
    (values[(y0 + 1) * columns + x0] * (1 - tx) + values[(y0 + 1) * columns + x0 + 1] * tx) * ty
  );
}

/**
 * 格子点ごとに、その XY を覆う三角形のうち最も手前（z 最大）の z を拾う。
 *
 * 頂点の z をビンへ放り込むのではなく三角形を走査するのは、GNM の三角形の辺長が格子のセルと
 * 同程度で、頂点だけでは格子点を取りこぼすため。三角形を辿れば「その格子点に本当に表面がある
 * か」も同時に決まる。
 *
 * @returns z（覆われていない点は 0）と covered
 */
export function rasterizeFrontZ(
  vertices: Float64Array,
  triangles: Uint32Array,
  grid: Grid2d,
): { z: Float64Array; covered: Uint8Array } {
  const pointCount = gridPointCount(grid);
  const frontZ = new Float64Array(pointCount).fill(-Infinity);
  const triangleCount = triangles.length / 3;

  for (let triangle = 0; triangle < triangleCount; triangle++) {
    const a = triangles[triangle * 3];
    const b = triangles[triangle * 3 + 1];
    const c = triangles[triangle * 3 + 2];
    // 格子 index 空間（格子点が整数座標）へ移す。以降の bbox・barycentric はこの空間。
    const ax = (vertices[a * 3] - grid.xMin) / grid.xStep;
    const ay = (vertices[a * 3 + 1] - grid.yMin) / grid.yStep;
    const az = vertices[a * 3 + 2];
    const bx = (vertices[b * 3] - grid.xMin) / grid.xStep;
    const by = (vertices[b * 3 + 1] - grid.yMin) / grid.yStep;
    const bz = vertices[b * 3 + 2];
    const cx = (vertices[c * 3] - grid.xMin) / grid.xStep;
    const cy = (vertices[c * 3 + 1] - grid.yMin) / grid.yStep;
    const cz = vertices[c * 3 + 2];

    const doubledArea = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (!(Math.abs(doubledArea) > COVERAGE_EPSILON)) continue;

    const minX = Math.min(ax, bx, cx);
    const maxX = Math.max(ax, bx, cx);
    const minY = Math.min(ay, by, cy);
    const maxY = Math.max(ay, by, cy);
    // クランプ前に「格子と交わるか」を決める。先にクランプすると、格子の外にある三角形が端の
    // 列・行を覆っていることになってしまう。
    if (maxX < 0 || minX > grid.columns - 1 || maxY < 0 || minY > grid.rows - 1) continue;

    const firstColumn = Math.min(Math.max(Math.ceil(minX), 0), grid.columns - 1);
    const lastColumn = Math.min(Math.max(Math.floor(maxX), 0), grid.columns - 1);
    const firstRow = Math.min(Math.max(Math.ceil(minY), 0), grid.rows - 1);
    const lastRow = Math.min(Math.max(Math.floor(maxY), 0), grid.rows - 1);
    if (lastColumn < firstColumn || lastRow < firstRow) continue;

    for (let row = firstRow; row <= lastRow; row++) {
      for (let column = firstColumn; column <= lastColumn; column++) {
        const weightA =
          ((bx - column) * (cy - row) - (by - row) * (cx - column)) / doubledArea;
        const weightB =
          ((cx - column) * (ay - row) - (cy - row) * (ax - column)) / doubledArea;
        const weightC = 1 - weightA - weightB;
        if (
          weightA < -COVERAGE_EPSILON ||
          weightB < -COVERAGE_EPSILON ||
          weightC < -COVERAGE_EPSILON
        ) {
          continue;
        }
        const sampled = weightA * az + weightB * bz + weightC * cz;
        const index = row * grid.columns + column;
        if (sampled > frontZ[index]) frontZ[index] = sampled;
      }
    }
  }

  const z = new Float64Array(pointCount);
  const covered = new Uint8Array(pointCount);
  for (let index = 0; index < pointCount; index++) {
    if (Number.isFinite(frontZ[index])) {
      covered[index] = 1;
      z[index] = frontZ[index];
    }
  }
  return { z, covered };
}
