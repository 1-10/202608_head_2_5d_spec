// フィット（`domain/gnm/fit`）の検査。
//
// **推論を走らせずに検証できる**のがこの段の設計目標（入力は 478 点と GNM アセットだけ）。合成した
// ランドマークで往復させ、対応の交差検出が実際に効くことも見る。

import { describe, expect, it } from 'vitest';
import {
  IBUG68_POINT_COUNT,
  verticesOf,
} from '../src/domain/gnm/model';
import {
  DISAGREEMENT_SWEEP_RATIOS,
  LandmarkCorrespondenceError,
  MEDIAPIPE_FACE_MESH_COUNT,
  MEDIAPIPE_IBUG68,
  MEDIAPIPE_LANDMARK_COUNT,
  REFERENCE_OBSERVATION_COUNT,
  Similarity2d,
  assertLandmarkChainOrientation,
  buildDenseLandmarkModel,
  buildLandmarkModel,
  coarseSimilarity,
  evaluateModel,
  fitHead,
  regularizationScheduleFor,
  selectIbug68,
  solveSimilarity2d,
  solveSymmetricPositiveDefinite,
  xyOf,
} from '../src/domain/gnm/fit';
import { headOnlySquare, headInferenceSquare, meanChinHeight } from '../src/domain/gnm/crop';
import { loadAsset } from './asset';

/** 対応表は顔メッシュの範囲だけを指す（虹彩を指すとフィットが眼球に引かれる）。 */
describe('MEDIAPIPE_IBUG68', () => {
  it('68 点あり、すべて顔メッシュの範囲', () => {
    expect(MEDIAPIPE_IBUG68.length).toBe(IBUG68_POINT_COUNT);
    for (const index of MEDIAPIPE_IBUG68) {
      expect(index).toBeGreaterThanOrEqual(0);
      expect(index).toBeLessThan(MEDIAPIPE_FACE_MESH_COUNT);
    }
    expect(new Set(MEDIAPIPE_IBUG68).size).toBe(IBUG68_POINT_COUNT);
  });
});

describe('相似変換', () => {
  it('回転 + スケール + 平行移動を厳密に解ける', () => {
    const source = Float64Array.from([0, 0, 1, 0, 0, 1, 1, 1]);
    const angle = 0.3;
    const scale = 2.5;
    const target = new Float64Array(source.length);
    for (let point = 0; point < 4; point++) {
      const x = source[point * 2];
      const y = source[point * 2 + 1];
      target[point * 2] = scale * (Math.cos(angle) * x - Math.sin(angle) * y) + 7;
      target[point * 2 + 1] = scale * (Math.sin(angle) * x + Math.cos(angle) * y) - 3;
    }
    const similarity = solveSimilarity2d(source, target);
    expect(similarity.scale).toBeCloseTo(scale, 10);
    expect(similarity.isMirrored).toBe(false);
    const applied = similarity.apply(source);
    for (let index = 0; index < applied.length; index++) {
      expect(applied[index]).toBeCloseTo(target[index], 10);
    }
  });

  it('鏡映を含む対応も解ける（画像座標は y が下向き）', () => {
    const source = Float64Array.from([0, 0, 1, 0, 0, 1, 1, 1]);
    const target = new Float64Array(source.length);
    for (let point = 0; point < 4; point++) {
      target[point * 2] = source[point * 2];
      target[point * 2 + 1] = -source[point * 2 + 1];
    }
    expect(solveSimilarity2d(source, target).isMirrored).toBe(true);
  });

  it('逆変換は往復する', () => {
    const similarity = new Similarity2d(
      Float64Array.from([2, -1, 1, 2]),
      Float64Array.from([5, -4]),
    );
    const points = Float64Array.from([1, 2, -3, 4]);
    const roundTrip = similarity.inverseApply(similarity.apply(points));
    for (let index = 0; index < points.length; index++) {
      expect(roundTrip[index]).toBeCloseTo(points[index], 10);
    }
  });
});

describe('Cholesky', () => {
  it('対称正定値を解ける', () => {
    const matrix = Float64Array.from([4, 1, 1, 3]);
    const solution = solveSymmetricPositiveDefinite(matrix, Float64Array.from([1, 2]), 2);
    expect(4 * solution[0] + solution[1]).toBeCloseTo(1, 10);
    expect(solution[0] + 3 * solution[1]).toBeCloseTo(2, 10);
  });

  it('正定値でなければ落ちる', () => {
    expect(() =>
      solveSymmetricPositiveDefinite(Float64Array.from([0, 0, 0, 0]), Float64Array.from([1, 1]), 2),
    ).toThrow(/正定値/);
  });
});

describe('正則化のスケジュール', () => {
  it('強 → 弱で、比は (6, 2, 1)', () => {
    expect([...DISAGREEMENT_SWEEP_RATIOS]).toEqual([6, 2, 1]);
    const model = buildLandmarkModel(loadAsset(), loadAsset().landmarks);
    const schedule = regularizationScheduleFor(model, 1);
    expect(schedule.length).toBe(3);
    expect(schedule[0]).toBeGreaterThan(schedule[1]);
    expect(schedule[1]).toBeGreaterThan(schedule[2]);
    // λ = ずれ² / 136（点数に依らない）。
    expect(schedule[2]).toBeCloseTo(0.01 ** 2 / REFERENCE_OBSERVATION_COUNT, 12);
  });

  it('倍率はλを 2 乗で効かせる', () => {
    const model = buildLandmarkModel(loadAsset(), loadAsset().landmarks);
    const base = regularizationScheduleFor(model, 1);
    const doubled = regularizationScheduleFor(model, 2);
    expect(doubled[2] / base[2]).toBeCloseTo(4, 10);
  });
});

/** 平均顔を相似変換で写真へ写して 478 点を合成する（推論を使わずにフィットを回す）。 */
function syntheticLandmarks(similarity: Similarity2d): Float64Array {
  const asset = loadAsset();
  const model = buildDenseLandmarkModel(asset, asset.dense);
  const mean = evaluateModel(model, new Float64Array(model.identityComponentCount));
  const projected = similarity.apply(xyOf(mean));
  const landmarks = new Float64Array(MEDIAPIPE_LANDMARK_COUNT * 2);
  for (let point = 0; point < model.pointCount; point++) {
    const target = model.photoIndices[point];
    landmarks[target * 2] = projected[point * 2];
    landmarks[target * 2 + 1] = projected[point * 2 + 1];
  }
  // 虹彩 10 点は形状フィットに使わないが、形として埋めておく（眼球の段が読む）。
  for (let point = MEDIAPIPE_FACE_MESH_COUNT; point < MEDIAPIPE_LANDMARK_COUNT; point++) {
    landmarks[point * 2] = projected[0];
    landmarks[point * 2 + 1] = projected[1];
  }
  return landmarks;
}

/** 正面写真に相当する相似変換（y を反転してスケールを掛ける）。 */
function frontalSimilarity(scale = 2000, x = 500, y = 400): Similarity2d {
  return new Similarity2d(
    Float64Array.from([scale, 0, 0, -scale]),
    Float64Array.from([x, y]),
  );
}

describe('fitHead', () => {
  it('平均顔をそのまま写した写真では identity がほぼ 0 になる', () => {
    const asset = loadAsset();
    const model = buildDenseLandmarkModel(asset, asset.dense);
    const similarity = frontalSimilarity();
    const fit = fitHead(syntheticLandmarks(similarity), model);
    let maximum = 0;
    for (const value of fit.identity) maximum = Math.max(maximum, Math.abs(value));
    expect(maximum).toBeLessThan(1e-6);
    // 残差はほぼ 0（写真が平均顔そのもの）。
    expect(fit.residualRmsPixels[fit.residualRmsPixels.length - 1]).toBeLessThan(1e-6);
    expect(fit.similarity.scale).toBeCloseTo(similarity.scale, 6);
  });

  it('coarseSimilarity は第1周が解く相似変換と同一', () => {
    const asset = loadAsset();
    const model = buildDenseLandmarkModel(asset, asset.dense);
    const landmarks = syntheticLandmarks(frontalSimilarity());
    const coarse = coarseSimilarity(landmarks, model);
    const fit = fitHead(landmarks, model, { regularizationSchedule: [1e9] });
    for (let index = 0; index < 4; index++) {
      expect(coarse.linear[index]).toBeCloseTo(fit.similarity.linear[index], 8);
    }
  });

  it('identityClip は係数を挟む', () => {
    const asset = loadAsset();
    const model = buildDenseLandmarkModel(asset, asset.dense);
    const landmarks = syntheticLandmarks(frontalSimilarity());
    // 事前分布を最大まで緩めても clip の外へは出ない。
    const fit = fitHead(landmarks, model, {
      regularizationSchedule: [1e-12],
      identityClip: 0.5,
    });
    for (const value of fit.identity) expect(Math.abs(value)).toBeLessThanOrEqual(0.5 + 1e-12);
  });
});

describe('対応の交差検出', () => {
  it('正しい対応は通る', () => {
    const asset = loadAsset();
    const model = buildDenseLandmarkModel(asset, asset.dense);
    const landmarks = syntheticLandmarks(frontalSimilarity());
    expect(() => fitHead(landmarks, model)).not.toThrow();
    void selectIbug68(landmarks);
  });

  it('顎ラインを反転させると落ちる', () => {
    const modelPoints = new Float64Array(IBUG68_POINT_COUNT * 2);
    const photoPoints = new Float64Array(IBUG68_POINT_COUNT * 2);
    for (let point = 0; point < IBUG68_POINT_COUNT; point++) {
      modelPoints[point * 2] = point;
      modelPoints[point * 2 + 1] = 0;
      photoPoints[point * 2] = point;
      photoPoints[point * 2 + 1] = 0;
    }
    // 顎（0..16）だけ並びを逆にする。
    for (let point = 0; point < 17; point++) photoPoints[point * 2] = 16 - point;
    expect(() => assertLandmarkChainOrientation(modelPoints, photoPoints, 2)).toThrow(
      LandmarkCorrespondenceError,
    );
  });
});

describe('推論の切り出し', () => {
  it('頭部だけの正方形はメッシュ全体の正方形より小さい', () => {
    const asset = loadAsset();
    const model = buildDenseLandmarkModel(asset, asset.dense);
    const landmarks = syntheticLandmarks(frontalSimilarity(2000, 1200, 900));
    const meshXy = new Float64Array(asset.mesh.vertexCount * 2);
    for (let vertex = 0; vertex < asset.mesh.vertexCount; vertex++) {
      meshXy[vertex * 2] = asset.mesh.templateVertexPositions[vertex * 3];
      meshXy[vertex * 2 + 1] = asset.mesh.templateVertexPositions[vertex * 3 + 1];
    }
    const imageSize: [number, number] = [2400, 3200];
    const head = headOnlySquare(landmarks, model, meshXy, imageSize);
    const body = headInferenceSquare(landmarks, model, meshXy, imageSize);
    expect(head.size).toBeLessThan(body.size);
    // 一辺は顔幅の 2.1〜2.3 倍（デスクトップ側の実測 2.136〜2.230）。
    const faceWidthPixels = model.faceWidth * 2000;
    expect(head.size / faceWidthPixels).toBeGreaterThan(2.1);
    expect(head.size / faceWidthPixels).toBeLessThan(2.3);
    // どちらも画像の中に収まる。
    for (const square of [head, body]) {
      expect(square.x).toBeGreaterThanOrEqual(0);
      expect(square.y).toBeGreaterThanOrEqual(0);
      expect(square.x + square.size).toBeLessThanOrEqual(imageSize[0]);
      expect(square.y + square.size).toBeLessThanOrEqual(imageSize[1]);
    }
  });

  it('顎の高さは 68 点の最下点', () => {
    const asset = loadAsset();
    const model = buildDenseLandmarkModel(asset, asset.dense);
    const chin = meanChinHeight(model);
    for (let slot = 0; slot < IBUG68_POINT_COUNT; slot++) {
      expect(model.meanPositions[model.guardRows[slot] * 3 + 1]).toBeGreaterThanOrEqual(chin);
    }
    // 頭頂は顎より上にある（GNM 空間は +Y が上）。
    const vertices = verticesOf(asset, new Float64Array(asset.vertexIdentityBasis.componentCount));
    let highest = -Infinity;
    for (let vertex = 0; vertex < asset.mesh.vertexCount; vertex++) {
      highest = Math.max(highest, vertices[vertex * 3 + 1]);
    }
    expect(highest).toBeGreaterThan(chin);
  });
});
