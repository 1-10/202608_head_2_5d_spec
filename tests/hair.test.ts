// 髪シェル（`domain/hair`）の検査。
//
// 格子・頭皮 z バッファ・主役の選択は推論なしで検証できる（座標系と相似変換を知らない関数に本体が
// 切り出してある）。

import { describe, expect, it } from 'vitest';
import {
  blur3x3,
  boxBlur3x3,
  gridCellSize,
  gridOverRect,
  gridPointsXy,
  nearestGridPoint,
  passesForGrid,
} from '../src/domain/hair/grid';
import { EmptyScalpError, buildScalpSurface, rasterizeFrontZ } from '../src/domain/hair/scalp';
import {
  DIAGONAL_STEP,
  SUBJECT_GEODESIC_DISTANCE,
  SUBJECT_SEED_DISTANCE,
  chamferDistance,
  subjectHairReach,
} from '../src/domain/hair/subject';
import {
  GRID_COLUMNS,
  GRID_ROWS,
  LEGACY_LIFT_METERS,
  LEGACY_ROLLOFF_METERS,
  validateHairShellParams,
  DEFAULT_HAIR_SHELL_PARAMS,
} from '../src/domain/hair/shell';
import { verticesOf } from '../src/domain/gnm/model';
import { loadAsset } from './asset';

describe('格子', () => {
  it('両端の格子点が矩形の辺に乗る', () => {
    const grid = gridOverRect(-1, -2, 1, 2, 5, 9);
    const points = gridPointsXy(grid);
    expect(points[0]).toBeCloseTo(-1, 12);
    expect(points[1]).toBeCloseTo(-2, 12);
    expect(points[(9 * 5 - 1) * 2]).toBeCloseTo(1, 12);
    expect(points[(9 * 5 - 1) * 2 + 1]).toBeCloseTo(2, 12);
    expect(gridCellSize(grid)).toBeCloseTo(Math.sqrt(0.5 * 0.5), 12);
  });

  it('row は GNM の +Y（上）方向に増える', () => {
    const grid = gridOverRect(0, 0, 1, 1, 2, 3);
    const points = gridPointsXy(grid);
    // index = row * columns + column。row が増えると y が増える。
    expect(points[(0 * 2 + 0) * 2 + 1]).toBeLessThan(points[(2 * 2 + 0) * 2 + 1]);
  });

  it('最近傍の格子点は範囲外を端へクランプする', () => {
    const grid = gridOverRect(0, 0, 1, 1, 3, 3);
    expect(nearestGridPoint(grid, -5, -5)).toEqual([0, 0]);
    expect(nearestGridPoint(grid, 5, 5)).toEqual([2, 2]);
    expect(nearestGridPoint(grid, 0.5, 0.5)).toEqual([1, 1]);
  });

  it('平滑化のパス数は格子点数に比例する（物理半径を保つ）', () => {
    const base = gridOverRect(0, 0, 1, 1, GRID_COLUMNS, GRID_ROWS);
    expect(passesForGrid(6, base, GRID_COLUMNS, GRID_ROWS)).toBe(6);
    const dense = gridOverRect(0, 0, 1, 1, GRID_COLUMNS * 2, GRID_ROWS * 2);
    expect(passesForGrid(6, dense, GRID_COLUMNS, GRID_ROWS)).toBe(24);
  });

  it('blur は端で値を複製し、box blur は標本数を減らす', () => {
    const values = Float64Array.from([1, 0, 0, 0, 0]);
    const binomial = blur3x3(values, 1, 5, 1);
    // 二項 blur は端の値を複製するので、端は 0.75 になる（0.25*1 + 0.5*1）。
    expect(binomial[0]).toBeCloseTo(0.75, 12);
    const box = boxBlur3x3(values, 1, 5, 1);
    // box blur は端で標本 2 つ（自分 + 右）なので 0.5。
    expect(box[0]).toBeCloseTo(0.5, 12);
  });
});

describe('chamfer 距離', () => {
  it('斜めは √2、縦横は 1', () => {
    const seed = new Uint8Array(9);
    seed[4] = 1; // 中心
    const distance = chamferDistance(seed, 3, 3);
    expect(distance[4]).toBe(0);
    expect(distance[1]).toBeCloseTo(1, 12);
    expect(distance[0]).toBeCloseTo(DIAGONAL_STEP, 12);
  });

  it('種が無ければ Infinity', () => {
    const distance = chamferDistance(new Uint8Array(9), 3, 3);
    for (const value of distance) expect(value).toBe(Infinity);
  });
});

describe('主役の髪の選択', () => {
  it('種が空なら何も切らない（主役の髪を消さない）', () => {
    const hair = new Uint8Array(16).fill(1);
    const result = subjectHairReach(hair, new Uint8Array(16), 4, 4, 4);
    expect(Array.from(result)).toEqual(Array.from(hair));
  });

  it('種から届かない髪は落ちる', () => {
    const rows = 8;
    const columns = 16;
    const hair = new Uint8Array(rows * columns);
    const seed = new Uint8Array(rows * columns);
    // 左端に主役の髪と種、右端に隣人の髪（繋がっていない）。
    for (let row = 0; row < rows; row++) {
      hair[row * columns] = 1;
      hair[row * columns + 1] = 1;
      hair[row * columns + columns - 1] = 1;
    }
    seed[0] = 1;
    const result = subjectHairReach(
      hair,
      seed,
      rows,
      columns,
      2,
      SUBJECT_SEED_DISTANCE,
      SUBJECT_GEODESIC_DISTANCE,
      64,
    );
    // 主役側は残る。
    expect(result[0]).toBe(1);
    expect(result[1]).toBe(1);
    // 隣人側は種からも測地でも届かない。
    expect(result[columns - 1]).toBe(0);
  });

  it('縮小率を変えても主役の髪の保持は 100%（作業解像度は速さだけを決める）', () => {
    const rows = 64;
    const columns = 64;
    const hair = new Uint8Array(rows * columns);
    const seed = new Uint8Array(rows * columns);
    for (let row = 8; row < 40; row++) {
      for (let column = 8; column < 40; column++) hair[row * columns + column] = 1;
    }
    seed[20 * columns + 20] = 1;
    const faceSquare = 32;
    // **保持が常に 100% であることがこの実装の要件**（除去率は落ちてよい）。作業解像度を振っても
    // 主役の髪は 1 画素も落ちない。
    for (const workingFacePixels of [1e9, 32, 16, 8]) {
      const reach = subjectHairReach(hair, seed, rows, columns, faceSquare, 0.7, 3, workingFacePixels);
      let kept = 0;
      let total = 0;
      for (let pixel = 0; pixel < hair.length; pixel++) {
        if (hair[pixel] === 0) continue;
        total++;
        if (reach[pixel] !== 0) kept++;
      }
      expect(kept).toBe(total);
    }
  });
});

describe('頭皮 z バッファ', () => {
  it('三角形を走査して最前面 z を拾う', () => {
    // z = 0 の三角形と z = 1 の三角形を重ねる。手前（z 大）が勝つ。
    const vertices = Float64Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1]);
    const triangles = Uint32Array.from([0, 1, 2, 3, 4, 5]);
    const grid = gridOverRect(0, 0, 1, 1, 3, 3);
    const { z, covered } = rasterizeFrontZ(vertices, triangles, grid);
    expect(covered[0]).toBe(1);
    expect(z[0]).toBeCloseTo(1, 12);
  });

  it('格子の外の三角形は端の行・列を覆わない', () => {
    const vertices = Float64Array.from([10, 10, 0, 11, 10, 0, 10, 11, 0]);
    const triangles = Uint32Array.from([0, 1, 2]);
    const grid = gridOverRect(0, 0, 1, 1, 3, 3);
    const { covered } = rasterizeFrontZ(vertices, triangles, grid);
    for (const value of covered) expect(value).toBe(0);
  });

  it('実アセットの頭部を覆う格子では被覆が高く、外れた格子では落ちる', () => {
    const asset = loadAsset();
    const vertices = verticesOf(asset, new Float64Array(asset.vertexIdentityBasis.componentCount));
    let lowX = Infinity;
    let lowY = Infinity;
    let highX = -Infinity;
    let highY = -Infinity;
    for (let vertex = 0; vertex < asset.mesh.vertexCount; vertex++) {
      lowX = Math.min(lowX, vertices[vertex * 3]);
      highX = Math.max(highX, vertices[vertex * 3]);
      lowY = Math.min(lowY, vertices[vertex * 3 + 1]);
      highY = Math.max(highY, vertices[vertex * 3 + 1]);
    }
    const grid = gridOverRect(lowX, lowY, highX, highY, GRID_COLUMNS, GRID_ROWS);
    const scalp = buildScalpSurface(vertices, asset.mesh.triangles, grid);
    let coveredCount = 0;
    for (const value of scalp.covered) if (value !== 0) coveredCount++;
    expect(coveredCount / scalp.covered.length).toBeGreaterThan(0.4);

    const away = gridOverRect(100, 100, 101, 101, 8, 8);
    expect(() => buildScalpSurface(vertices, asset.mesh.triangles, away)).toThrow(EmptyScalpError);
  }, 60_000);
});

describe('髪シェルのパラメータ', () => {
  it('lift < rolloff を割らせない（殻の外周が頭皮の外へ浮く）', () => {
    expect(LEGACY_LIFT_METERS).toBeLessThan(LEGACY_ROLLOFF_METERS);
    expect(() => validateHairShellParams(DEFAULT_HAIR_SHELL_PARAMS)).not.toThrow();
    expect(() =>
      validateHairShellParams({ ...DEFAULT_HAIR_SHELL_PARAMS, liftMeters: 0.02 }),
    ).toThrow(/rolloff/);
  });
});
