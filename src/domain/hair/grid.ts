// 髪シェルの格子と、格子上のフィルタ。
//
// 髪シェルは「GNM 空間 XY に張った等間隔の格子」の各点を奥行き方向に押し出して作る。この格子と、
// 格子上で走らせる平滑化をここに閉じる。
//
// 格子点の index 空間（正本）:
//     `index = row * columns + column`。column は GNM の +X 方向、row は GNM の +Y 方向（上）に
//     増える。画像 UV 空間は v が下向きなので、検査画像に描くときは row を反転する。
//
// 平滑化のパス数を解像度に追随させる理由:
//     3x3 blur の物理半径は「セルサイズ × パス数の平方根」なので、格子を細かくすると同じパス数
//     でも物理的な平滑量が縮む。解像度を変えた瞬間に見た目が変わるのを避けるため、パス数は
//     格子点数に比例させる（`passesForGrid`）。

/** 格子の最小辺。1 だと格子間隔が定義できず、三角形も張れない。 */
export const MIN_GRID_SIDE = 2;

/** GNM 空間 XY の等間隔格子。値を持つのは格子点（セルの角）。 */
export interface Grid2d {
  /** 格子点 (row=0, column=0) の GNM 座標（メートル）。 */
  readonly xMin: number;
  readonly yMin: number;
  /** 格子間隔（メートル）。 */
  readonly xStep: number;
  readonly yStep: number;
  /** X 方向の格子点数。 */
  readonly columns: number;
  /** Y 方向の格子点数。 */
  readonly rows: number;
}

export function makeGrid(
  xMin: number,
  yMin: number,
  xStep: number,
  yStep: number,
  columns: number,
  rows: number,
): Grid2d {
  if (columns < MIN_GRID_SIDE || rows < MIN_GRID_SIDE) {
    throw new Error(`格子が小さすぎる: columns=${columns} rows=${rows}`);
  }
  if (!(xStep > 0 && yStep > 0)) {
    throw new Error(`格子間隔が正でない: xStep=${xStep} yStep=${yStep}`);
  }
  return { xMin, yMin, xStep, yStep, columns, rows };
}

/** 矩形を覆う格子。両端の格子点が矩形の辺に乗る（間隔は点数 − 1 で割る）。 */
export function gridOverRect(
  xMin: number,
  yMin: number,
  xMax: number,
  yMax: number,
  columns: number,
  rows: number,
): Grid2d {
  if (!(xMax > xMin && yMax > yMin)) {
    throw new Error(`矩形の幅・高さが正でない: x=${xMin}..${xMax} y=${yMin}..${yMax}`);
  }
  if (columns < MIN_GRID_SIDE || rows < MIN_GRID_SIDE) {
    throw new Error(`格子が小さすぎる: columns=${columns} rows=${rows}`);
  }
  return makeGrid(xMin, yMin, (xMax - xMin) / (columns - 1), (yMax - yMin) / (rows - 1), columns, rows);
}

export function gridPointCount(grid: Grid2d): number {
  return grid.rows * grid.columns;
}

/** セルの代表寸法（X と Y の間隔の相乗平均）。物理半径の基準に使う。 */
export function gridCellSize(grid: Grid2d): number {
  return Math.sqrt(grid.xStep * grid.yStep);
}

export function gridXCoordinate(grid: Grid2d, column: number): number {
  return grid.xMin + grid.xStep * column;
}

export function gridYCoordinate(grid: Grid2d, row: number): number {
  return grid.yMin + grid.yStep * row;
}

/** 全格子点の GNM XY 座標。並びは `row * columns + column`。 */
export function gridPointsXy(grid: Grid2d): Float64Array {
  const out = new Float64Array(gridPointCount(grid) * 2);
  for (let row = 0; row < grid.rows; row++) {
    const y = gridYCoordinate(grid, row);
    for (let column = 0; column < grid.columns; column++) {
      const index = row * grid.columns + column;
      out[index * 2] = gridXCoordinate(grid, column);
      out[index * 2 + 1] = y;
    }
  }
  return out;
}

/** GNM XY を最も近い格子点の (column, row) に丸める（範囲外は端にクランプ）。 */
export function nearestGridPoint(grid: Grid2d, x: number, y: number): [number, number] {
  const column = Math.round((x - grid.xMin) / grid.xStep);
  const row = Math.round((y - grid.yMin) / grid.yStep);
  return [
    Math.min(Math.max(column, 0), grid.columns - 1),
    Math.min(Math.max(row, 0), grid.rows - 1),
  ];
}

/**
 * 基準解像度で決めたパス数を、実際の格子の密度に合わせて換算する。
 *
 * 3x3 blur を n 回かけた実効 σ は `セルサイズ × sqrt(n / 2)`。物理 σ を保つには
 * `n ∝ 1 / セルサイズ²`、すなわち格子点数に比例させればよい。
 */
export function passesForGrid(
  basePasses: number,
  grid: Grid2d,
  baseColumns: number,
  baseRows: number,
): number {
  const ratio = (grid.columns * grid.rows) / (baseColumns * baseRows);
  return Math.max(1, Math.round(basePasses * ratio));
}

/**
 * 分離型の 3x3 二項 blur（`[1, 2, 1] / 4`）を `passes` 回かける。
 *
 * 端は値を複製して外挿する（0 で埋めると縁の値が勝手に薄まる）。
 */
export function blur3x3(
  values: Float64Array,
  rows: number,
  columns: number,
  passes: number,
): Float64Array {
  let result = Float64Array.from(values);
  for (let pass = 0; pass < Math.max(0, passes); pass++) {
    result = binomialPass(result, rows, columns, 0);
    result = binomialPass(result, rows, columns, 1);
  }
  return result;
}

function binomialPass(
  values: Float64Array,
  rows: number,
  columns: number,
  axis: number,
): Float64Array {
  const out = new Float64Array(values.length);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const lowerRow = axis === 0 ? Math.max(row - 1, 0) : row;
      const upperRow = axis === 0 ? Math.min(row + 1, rows - 1) : row;
      const lowerColumn = axis === 1 ? Math.max(column - 1, 0) : column;
      const upperColumn = axis === 1 ? Math.min(column + 1, columns - 1) : column;
      out[row * columns + column] =
        0.25 * values[lowerRow * columns + lowerColumn] +
        0.5 * values[row * columns + column] +
        0.25 * values[upperRow * columns + upperColumn];
    }
  }
  return out;
}

/** 旧髪実装の、端では標本数を減らす一様3x3平均。 */
export function boxBlur3x3(
  values: Float64Array,
  rows: number,
  columns: number,
  passes: number,
): Float64Array {
  let result = Float64Array.from(values);
  for (let pass = 0; pass < Math.max(0, passes); pass++) {
    const next = new Float64Array(result.length);
    for (let row = 0; row < rows; row++) {
      for (let column = 0; column < columns; column++) {
        let total = 0;
        let count = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const otherRow = row + dy;
          if (otherRow < 0 || otherRow >= rows) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const otherColumn = column + dx;
            if (otherColumn < 0 || otherColumn >= columns) continue;
            total += result[otherRow * columns + otherColumn];
            count++;
          }
        }
        next[row * columns + column] = total / count;
      }
    }
    result = next;
  }
  return result;
}

/**
 * 重み付き blur（normalized convolution）。重みが 0 の領域の値を混ぜない。
 *
 * 髪の外側の厚みは「髪ではない場所の深度」から出た無意味な値なので、素の blur で混ぜると髪の
 * 内側の厚みが汚れる。重み（髪らしさ）で正規化すると、髪の外側は髪の内側の値で埋まる方向に
 * 外挿される。
 */
export function normalizedBlur(
  values: Float64Array,
  weights: Float64Array,
  rows: number,
  columns: number,
  passes: number,
  fallback = 0,
): Float64Array {
  const weighted = new Float64Array(values.length);
  for (let index = 0; index < values.length; index++) weighted[index] = values[index] * weights[index];
  const numerator = blur3x3(weighted, rows, columns, passes);
  const denominator = blur3x3(weights, rows, columns, passes);
  const out = new Float64Array(values.length);
  for (let index = 0; index < out.length; index++) {
    out[index] =
      denominator[index] > 1e-12 ? numerator[index] / Math.max(denominator[index], 1e-12) : fallback;
  }
  return out;
}
