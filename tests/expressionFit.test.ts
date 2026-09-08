// MediaPipe の点から表情係数を解く経路の検査。
//
// **合成観測で測る。** 既知の係数で顔を動かし、密対応の点をそのまま「MediaPipe の出力」として
// 与えて、係数が戻るかを見る。実カメラでしか測れないもの（MediaPipe が本物の表情をどれだけ正しく
// 置くか）はここでは測らない — 測れるのは「点の位置から基底が同定できるか」だけで、それが
// `tools/experiments/expression_fit_feasibility.py` と同じ問いである。
//
// 期待する精度の根拠もあちらの実測: 舌と瞳は原理的に解けない / 目は弱い / 下顔面は通る。

import { describe, expect, it } from 'vitest';
import { loadBundle } from './asset';
import { verticesOf } from '../src/domain/gnm/model';
import { addExpression, addPresetCoefficients, zeroCoefficients } from '../src/domain/preview/expression';
import {
  ANCHOR_MOTION_LIMIT_METERS,
  MINIMUM_ANCHOR_COUNT,
  buildExpressionFitPlan,
  captureNeutralResidual,
  createExpressionFitScratch,
  smoothCoefficients,
  solveExpressionCoefficients,
  solveSimilarity3d,
} from '../src/domain/preview/expressionFit';
import { splitIndexOf } from '../src/domain/gnm/model';

/** 密対応の点を、与えた頂点から作って「MediaPipe の出力」の形（468 点 × 3）に詰める。 */
function landmarksFrom(
  bundle: ReturnType<typeof loadBundle>,
  vertices: Float64Array,
  transform: (point: number[]) => number[] = (point) => point,
): Float64Array {
  const { asset } = bundle;
  const dense = asset.dense;
  let widest = 0;
  for (const index of dense.mediapipeIndices) widest = Math.max(widest, index);
  const out = new Float64Array((widest + 1) * 3);
  for (let point = 0; point < dense.pointCount; point++) {
    const position = [0, 0, 0];
    for (let corner = 0; corner < 3; corner++) {
      const vertex = splitIndexOf(asset.mesh, dense.vertexIndices[point * 3 + corner]);
      const weight = dense.weights[point * 3 + corner];
      for (let axis = 0; axis < 3; axis++) {
        position[axis] += vertices[vertex * 3 + axis] * weight;
      }
    }
    const moved = transform(position);
    const landmark = dense.mediapipeIndices[point];
    for (let axis = 0; axis < 3; axis++) out[landmark * 3 + axis] = moved[axis];
  }
  return out;
}

function relativeError(estimate: Float64Array, truth: Float64Array): number {
  let difference = 0;
  let size = 0;
  for (let index = 0; index < truth.length; index++) {
    difference += (estimate[index] - truth[index]) ** 2;
    size += truth[index] ** 2;
  }
  return Math.sqrt(difference / size);
}

describe('相似変換（3D）', () => {
  it('尺度・回転・平行移動を掛けた点群から元の変換を取り戻す', () => {
    const source = Float64Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]);
    // z 軸まわり 30 度・2 倍・(0.1, -0.2, 0.3) 平行移動。
    const angle = Math.PI / 6;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const target = new Float64Array(source.length);
    for (let point = 0; point < source.length / 3; point++) {
      const x = source[point * 3];
      const y = source[point * 3 + 1];
      const z = source[point * 3 + 2];
      target[point * 3] = 2 * (cos * x - sin * y) + 0.1;
      target[point * 3 + 1] = 2 * (sin * x + cos * y) - 0.2;
      target[point * 3 + 2] = 2 * z + 0.3;
    }
    const solved = solveSimilarity3d(source, target);
    expect(solved.scale).toBeCloseTo(2, 10);
    expect(solved.rotation[0]).toBeCloseTo(cos, 10);
    expect(solved.rotation[1]).toBeCloseTo(-sin, 10);
    expect(solved.rotation[3]).toBeCloseTo(sin, 10);
    expect(solved.translation[0]).toBeCloseTo(0.1, 10);
    expect(solved.translation[1]).toBeCloseTo(-0.2, 10);
    expect(solved.translation[2]).toBeCloseTo(0.3, 10);
  });

  it('鏡像は作らない（回転行列の行列式は +1）', () => {
    const source = Float64Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const mirrored = Float64Array.from([0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const { rotation } = solveSimilarity3d(source, mirrored);
    const determinant =
      rotation[0] * (rotation[4] * rotation[8] - rotation[5] * rotation[7]) -
      rotation[1] * (rotation[3] * rotation[8] - rotation[5] * rotation[6]) +
      rotation[2] * (rotation[3] * rotation[7] - rotation[4] * rotation[6]);
    expect(determinant).toBeCloseTo(1, 10);
  });
});

describe('表情フィット', () => {
  // **手で選ぶと外れる。** 一度 14 点を手書きしたが、そのうち 8 点は表情で数 mm 動いていた
  // （目尻・額・鼻先）。笑うとそこが動き、相似変換が頭の動きと解釈して顔全体を歪めた。
  it('相似変換に使う点は、表情で動かない点だけ（基底から選ぶ）', () => {
    const bundle = loadBundle();
    const { asset, preview } = bundle;
    const rest = verticesOf(asset, new Float64Array(asset.vertexIdentityBasis.componentCount));
    const plan = buildExpressionFitPlan(asset, preview, rest);
    expect(plan.anchorSlots.length).toBeGreaterThanOrEqual(MINIMUM_ANCHOR_COUNT);
    // 全点の 1/3 未満（顔の大半は表情で動くので、そこは姿勢に使えない）。
    expect(plan.anchorSlots.length).toBeLessThan(plan.pointIndices.length / 3);

    // 選ばれた点は、どのプリセットを立てても閾値より動かない。
    for (const name of preview.expressionPresetNames) {
      const truth = zeroCoefficients(preview);
      const weights = new Float64Array(preview.presetCount);
      weights[preview.expressionPresetNames.indexOf(name)] = 1;
      addPresetCoefficients(preview, truth, weights);
      const moved = Float64Array.from(rest);
      addExpression(preview, moved, truth);
      const before = landmarksFrom(bundle, rest);
      const after = landmarksFrom(bundle, moved);
      for (const slot of plan.anchorSlots) {
        const landmark = plan.pointIndices[slot];
        const distance = Math.hypot(
          after[landmark * 3] - before[landmark * 3],
          after[landmark * 3 + 1] - before[landmark * 3 + 1],
          after[landmark * 3 + 2] - before[landmark * 3 + 2],
        );
        expect(distance, `${name} / MediaPipe ${landmark}`).toBeLessThan(
          ANCHOR_MOTION_LIMIT_METERS,
        );
      }
    }
  });

  it('無表情の点を渡すと係数がほぼ 0（勝手に表情が付かない）', () => {
    const bundle = loadBundle();
    const { asset, preview } = bundle;
    const rest = verticesOf(asset, new Float64Array(asset.vertexIdentityBasis.componentCount));
    const plan = buildExpressionFitPlan(asset, preview, rest);
    const scratch = createExpressionFitScratch(plan);
    const landmarks = landmarksFrom(bundle, rest);
    const solved = zeroCoefficients(preview);
    solveExpressionCoefficients(plan, landmarks, null, solved, scratch);

    const displacement = new Float64Array(rest.length);
    addExpression(preview, displacement, solved);
    let maximum = 0;
    for (const value of displacement) maximum = Math.max(maximum, Math.abs(value));
    // 量子化と相似変換の丸めぶんしか出ない（1 マイクロメートル未満）。
    expect(maximum).toBeLessThan(1e-6);
  });

  it('プリセットで動かした点から、その顔がほぼ戻る（下顔面が主のもの）', () => {
    const bundle = loadBundle();
    const { asset, preview } = bundle;
    const rest = verticesOf(asset, new Float64Array(asset.vertexIdentityBasis.componentCount));
    const plan = buildExpressionFitPlan(asset, preview, rest);
    const scratch = createExpressionFitScratch(plan);
    const solved = zeroCoefficients(preview);

    for (const name of ['smile_wide', 'pucker', 'stretch_face', 'wink_left', 'mouth_left']) {
      const truth = zeroCoefficients(preview);
      const weights = new Float64Array(preview.presetCount);
      weights[preview.expressionPresetNames.indexOf(name)] = 1;
      addPresetCoefficients(preview, truth, weights);

      const moved = Float64Array.from(rest);
      addExpression(preview, moved, truth);
      solveExpressionCoefficients(plan, landmarksFrom(bundle, moved), null, solved, scratch);

      // 係数そのものは一致しない（同じ顔を作る係数が何通りもある）。**顔で比べる。**
      const wanted = new Float64Array(rest.length);
      addExpression(preview, wanted, truth);
      const got = new Float64Array(rest.length);
      addExpression(preview, got, solved);
      // 舌は点に出ないので比較から外す。
      const tongue = preview.expressionBasisRegions.find((region) => region.name === 'tongue');
      if (tongue === undefined) throw new Error('舌の領域が無い');
      const hidden = new Uint8Array(preview.vertexCount);
      for (let slot = 0; slot < tongue.vertexCount; slot++) {
        hidden[preview.expressionBasisVertices[tongue.vertexOffset + slot]] = 1;
      }
      const visibleWanted: number[] = [];
      const visibleGot: number[] = [];
      for (let vertex = 0; vertex < preview.vertexCount; vertex++) {
        if (hidden[vertex] !== 0) continue;
        for (let axis = 0; axis < 3; axis++) {
          visibleWanted.push(wanted[vertex * 3 + axis]);
          visibleGot.push(got[vertex * 3 + axis]);
        }
      }
      const error = relativeError(Float64Array.from(visibleGot), Float64Array.from(visibleWanted));
      // 事前分布がプリセットの部分空間なので、プリセットそのものはよく戻る。**ここを緩めると
      // 錨の選び方が壊れても気付けない**（手書きの 14 点だったときは wink_left が 56% だった）。
      expect(error, name).toBeLessThan(0.15);
    }
  });

  it('頭を動かしても表情は変わらない（相似変換で姿勢を落としている）', () => {
    const bundle = loadBundle();
    const { asset, preview } = bundle;
    const rest = verticesOf(asset, new Float64Array(asset.vertexIdentityBasis.componentCount));
    const plan = buildExpressionFitPlan(asset, preview, rest);
    const scratch = createExpressionFitScratch(plan);

    const truth = zeroCoefficients(preview);
    const weights = new Float64Array(preview.presetCount);
    weights[preview.expressionPresetNames.indexOf('smile_wide')] = 1;
    addPresetCoefficients(preview, truth, weights);
    const moved = Float64Array.from(rest);
    addExpression(preview, moved, truth);

    const still = zeroCoefficients(preview);
    solveExpressionCoefficients(plan, landmarksFrom(bundle, moved), null, still, scratch);

    // 20 度回して 1.7 倍にして平行移動した観測。
    const angle = (20 * Math.PI) / 180;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const turnedLandmarks = landmarksFrom(bundle, moved, ([x, y, z]) => [
      1.7 * (cos * x + sin * z) + 0.4,
      1.7 * y - 0.2,
      1.7 * (-sin * x + cos * z) + 0.9,
    ]);
    const turned = zeroCoefficients(preview);
    solveExpressionCoefficients(plan, turnedLandmarks, null, turned, scratch);

    expect(relativeError(turned, still)).toBeLessThan(0.02);
  });

  it('無表情の残差を取ると、顔が違っても無表情から始まる', () => {
    const bundle = loadBundle();
    const { asset, preview } = bundle;
    const identity = new Float64Array(asset.vertexIdentityBasis.componentCount);
    const rest = verticesOf(asset, identity);
    const plan = buildExpressionFitPlan(asset, preview, rest);
    const scratch = createExpressionFitScratch(plan);

    // 別人の顔（identity を振った頭）を観測として渡す。
    identity[0] = 2.5;
    identity[3] = -1.5;
    const otherFace = verticesOf(asset, identity);
    const neutralLandmarks = landmarksFrom(bundle, otherFace);
    const residual = captureNeutralResidual(plan, neutralLandmarks);

    const withoutNeutral = zeroCoefficients(preview);
    solveExpressionCoefficients(plan, neutralLandmarks, null, withoutNeutral, scratch);
    const withNeutral = zeroCoefficients(preview);
    solveExpressionCoefficients(plan, neutralLandmarks, residual, withNeutral, scratch);

    const displacementOf = (coefficients: Float64Array): number => {
      const displacement = new Float64Array(rest.length);
      addExpression(preview, displacement, coefficients);
      let maximum = 0;
      for (const value of displacement) maximum = Math.max(maximum, Math.abs(value));
      return maximum;
    };
    // 残差を取らないと顔の違いが表情として出る。取れば消える。
    expect(displacementOf(withoutNeutral)).toBeGreaterThan(1e-4);
    expect(displacementOf(withNeutral)).toBeLessThan(1e-6);
  });

  it('時定数は目標へ近づけるだけ（追い越さない）', () => {
    const current = Float64Array.from([0, 0]);
    const target = Float64Array.from([1, -1]);
    smoothCoefficients(current, target, 0.016);
    expect(current[0]).toBeGreaterThan(0);
    expect(current[0]).toBeLessThan(1);
    expect(current[1]).toBeLessThan(0);
    expect(current[1]).toBeGreaterThan(-1);
    // 十分な時間で届く。
    for (let step = 0; step < 100; step++) smoothCoefficients(current, target, 0.016);
    expect(current[0]).toBeCloseTo(1, 6);
  });
});
