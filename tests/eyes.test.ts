// 眼球 UV の測定を**実アセット**で押さえる。
//
// `domain/eyes/geometry` は「UV の角度 ↔ 解剖学的な向き」と「UV の半径 ↔ 正面投影半径」をメッシュから
// 測る。**測った結果がデスクトップ側の実測値と一致すること**が、この移植が同じ幾何を見ている証拠に
// なる（写しを定数で置いていないので、ここが唯一の突き合わせ）。

import { describe, expect, it } from 'vitest';
import {
  MAXIMUM_ANGLE_RESIDUAL_DEGREES,
  angleResidualRmsDegrees,
  eyeUvGeometries,
  eyeInteriorVertexMask,
  limbusFraction,
} from '../src/domain/eyes/geometry';
import {
  EYE_SIDES,
  IRIS_OUTER_RADIUS_UV,
  PUPIL_RADIUS_UV,
  SCLERA_OUTER_RADIUS_UV,
} from '../src/domain/eyes/layout';
import { irisBandMask, scleraBandMask, uvPolarGrid } from '../src/domain/eyes/bake';
import { loadAsset } from './asset';

/** 内殻の頂点数（v3_0 / head の実測値）。 */
const INTERIOR_VERTEX_COUNT = 385;

describe('眼球 UV の角度', () => {
  it('左右どちらも「回転のみ・反転なし」で表せる（残差 RMS は 1° 未満）', () => {
    const geometries = eyeUvGeometries(loadAsset().mesh);
    for (const side of EYE_SIDES) {
      const geometry = geometries[side];
      expect(geometry.angleFlipped).toBe(false);
      expect(angleResidualRmsDegrees(geometry)).toBeLessThan(1);
      expect(angleResidualRmsDegrees(geometry)).toBeLessThan(MAXIMUM_ANGLE_RESIDUAL_DEGREES);
      // offset は ±0.04° の桁（実測）。
      expect(Math.abs((geometry.angleOffsetRadians * 180) / Math.PI)).toBeLessThan(0.5);
    }
  });
});

describe('眼球 UV の半径', () => {
  it('内殻を選び出せている（UV 半径の上限が強膜の外径）', () => {
    const mesh = loadAsset().mesh;
    const geometries = eyeUvGeometries(mesh);
    for (const side of EYE_SIDES) {
      expect(geometries[side].vertexCount).toBe(INTERIOR_VERTEX_COUNT);
      const mask = eyeInteriorVertexMask(mesh, side);
      let count = 0;
      for (const value of mask) if (value !== 0) count++;
      expect(count).toBe(INTERIOR_VERTEX_COUNT);
    }
  });

  it('limbus の環が UV 半径 0.1849 に来る', () => {
    const geometries = eyeUvGeometries(loadAsset().mesh);
    for (const side of EYE_SIDES) {
      const geometry = geometries[side];
      // limbus の投影半径比は定義から 1.0。
      expect(limbusFraction(geometry, IRIS_OUTER_RADIUS_UV)).toBeCloseTo(1, 2);
      expect(limbusFraction(geometry, 0)).toBeCloseTo(0, 2);
      expect(geometry.limbusRadiusMeters).toBeGreaterThan(0.004);
      expect(geometry.limbusRadiusMeters).toBeLessThan(0.008);
    }
  });

  it('半径の写像は線形でない（線形に写すと瞳孔が 1.6 倍に膨らむ）', () => {
    const geometries = eyeUvGeometries(loadAsset().mesh);
    for (const side of EYE_SIDES) {
      const measured = limbusFraction(geometries[side], PUPIL_RADIUS_UV);
      const linear = PUPIL_RADIUS_UV / IRIS_OUTER_RADIUS_UV;
      // 実測 0.498 に対し線形なら 0.270。
      expect(measured).toBeGreaterThan(linear * 1.5);
      expect(measured).toBeCloseTo(0.498, 2);
    }
  });

  it('シルエットの外は写真のどの画素も指さない（NaN）', () => {
    const geometries = eyeUvGeometries(loadAsset().mesh);
    for (const side of EYE_SIDES) {
      const geometry = geometries[side];
      expect(geometry.silhouetteRadiusUv).toBeGreaterThan(IRIS_OUTER_RADIUS_UV);
      expect(geometry.silhouetteRadiusUv).toBeLessThan(SCLERA_OUTER_RADIUS_UV);
      expect(Number.isNaN(limbusFraction(geometry, SCLERA_OUTER_RADIUS_UV))).toBe(true);
    }
  });

  it('左右の profile は一致する（同じ絵を左右で別に測っている裏付け）', () => {
    const geometries = eyeUvGeometries(loadAsset().mesh);
    const left = geometries.left;
    const right = geometries.right;
    expect(left.ringRadiiUv.length).toBe(right.ringRadiiUv.length);
    for (let ring = 0; ring < left.ringRadiiUv.length; ring++) {
      expect(left.ringRadiiUv[ring]).toBeCloseTo(right.ringRadiiUv[ring], 5);
      expect(left.ringLimbusFractions[ring]).toBeCloseTo(right.ringLimbusFractions[ring], 2);
    }
  });

  /**
   * デスクトップ側の `domain/eyes/geometry` の docstring が載せている実測表。
   *
   * **移植が同じ幾何を測っていることの唯一の突き合わせ**（写しを定数で置いていないので、profile が
   * 一致することがそのまま「同じアセットを同じ規則で測った」ことになる）。
   */
  const DOCUMENTED_PROFILE: readonly [number, number][] = [
    [0.0, 0.0],
    [0.0246, 0.262],
    [0.0293, 0.31],
    [0.0499, 0.498],
    [0.1008, 0.687],
    [0.1544, 0.886],
    [0.176, 0.967],
    [0.1849, 1.0],
    [0.1912, 1.041],
    [0.2072, 1.195],
    [0.2434, 1.476],
    [0.2913, 1.847],
    [0.34, 2.166],
    [0.3839, 2.464],
    [0.4184, 2.685],
    [0.4427, 2.773],
    [0.4565, 2.719],
    [0.4608, 2.546],
  ];

  it('環の半径と投影半径比がデスクトップ側の実測表と一致する', () => {
    const geometry = eyeUvGeometries(loadAsset().mesh).left;
    expect(geometry.ringRadiiUv.length).toBe(DOCUMENTED_PROFILE.length);
    DOCUMENTED_PROFILE.forEach(([radiusUv, fraction], ring) => {
      expect(geometry.ringRadiiUv[ring]).toBeCloseTo(radiusUv, 4);
      expect(geometry.ringLimbusFractions[ring]).toBeCloseTo(fraction, 2);
    });
    // シルエット（投影半径比が最大の環）は表の 0.4427 の行。
    expect(geometry.silhouetteRadiusUv).toBeCloseTo(0.4427, 4);
  });
});

describe('極座標の格子', () => {
  it('v は上向き（行 0 が v = 1 側）', () => {
    const { radius, angle } = uvPolarGrid(4);
    // 行 0 / 列 3 は右上 → 角度は +45° 付近。
    expect((angle[3] * 180) / Math.PI).toBeCloseTo(45, 0);
    // 行 3 / 列 3 は右下 → −45°。
    expect((angle[15] * 180) / Math.PI).toBeCloseTo(-45, 0);
    expect(radius[0]).toBeCloseTo(radius[15], 10);
  });

  it('帯のマスクは半径の定義どおり', () => {
    const size = 64;
    const iris = irisBandMask(size);
    const sclera = scleraBandMask(size);
    // 中心（瞳孔の中）は虹彩の帯に入らない。
    const center = (size / 2) * size + size / 2;
    expect(iris[center]).toBe(0);
    expect(sclera[center]).toBe(0);
    // 隅は強膜の外側（半径 0.5 付近）。
    expect(sclera[0]).toBe(1);
  });
});
