// カメラの顔から首を駆動する経路の検査（`domain/preview/headTracking`）。
//
// 表情はここに無い（`tests/expressionFit.test.ts`）。ここが見るのは、実機で見えていた症状 2 つを
// そのまま押さえる検査:
//
// - **上を向いたら下を向く**（行列から出る pitch とビューの `HeadPose` は上下が逆）
// - **少し振っただけで端に張り付く**（生の角度は可動域よりずっと広い）

import { describe, expect, it } from 'vitest';
import {
  HEAD_TIME_CONSTANT_SECONDS,
  headPoseFromMatrix,
  mapHeadAngle,
  smoothScalar,
  smoothingFactor,
} from '../src/domain/preview/headTracking';

describe('頭の姿勢の取り出し', () => {
  /** `Ry(yaw) * Rx(pitch)` を**行優先**で並べた 4x4（MediaPipe と同じ並び）。 */
  function matrixFor(yawRadians: number, pitchRadians: number): Float32Array {
    const cy = Math.cos(yawRadians);
    const sy = Math.sin(yawRadians);
    const cx = Math.cos(pitchRadians);
    const sx = Math.sin(pitchRadians);
    return Float32Array.from([
      cy, sy * sx, sy * cx, 0,
      0, cx, -sx, 0,
      -sy, cy * sx, cy * cx, 0,
      0, 0, 0, 1,
    ]);
  }

  it('正面はゼロ', () => {
    const pose = headPoseFromMatrix(matrixFor(0, 0), false);
    expect(pose).not.toBeNull();
    expect(pose!.yawDegrees).toBeCloseTo(0, 4);
    expect(pose!.pitchDegrees).toBeCloseTo(0, 4);
  });

  // **pitch は符号が返る。** 行列の pitch とビューの `HeadPose.headPitchDegrees` は上下が逆で、
  // そのまま渡すと上を向いたら下を向く（実機でそう見えていた）。
  it('yaw と pitch を分けて取り出せる（pitch は符号が返る）', () => {
    const pose = headPoseFromMatrix(matrixFor((20 * Math.PI) / 180, (-12 * Math.PI) / 180), false);
    expect(pose!.yawDegrees).toBeCloseTo(20, 3);
    expect(pose!.pitchDegrees).toBeCloseTo(12, 3);
  });

  it('上を向いたら上を向く（ビューの HeadPose と符号が揃う）', () => {
    const pose = headPoseFromMatrix(matrixFor(0, (10 * Math.PI) / 180), false);
    expect(pose!.pitchDegrees).toBeCloseTo(-10, 3);
  });

  it('鏡にすると yaw だけ変わる（pitch は鏡に依らない）', () => {
    const direct = headPoseFromMatrix(matrixFor((20 * Math.PI) / 180, (10 * Math.PI) / 180), false);
    const mirrored = headPoseFromMatrix(matrixFor((20 * Math.PI) / 180, (10 * Math.PI) / 180), true);
    expect(mirrored!.yawDegrees).toBeCloseTo(-direct!.yawDegrees, 6);
    expect(mirrored!.pitchDegrees).toBeCloseTo(direct!.pitchDegrees, 6);
  });

  it('短い配列と退化した行列は null（atan2 が 0 を返すのに任せない）', () => {
    expect(headPoseFromMatrix(new Float32Array(4))).toBeNull();
    expect(headPoseFromMatrix(new Float32Array(16))).toBeNull();
  });
});

describe('首の角度をリグの可動域へ写す', () => {
  it('範囲いっぱいで可動域いっぱい、外はクランプ', () => {
    expect(mapHeadAngle(0, 35, 15)).toBe(0);
    expect(mapHeadAngle(35, 35, 15)).toBeCloseTo(15, 6);
    expect(mapHeadAngle(-35, 35, 15)).toBeCloseTo(-15, 6);
    expect(mapHeadAngle(70, 35, 15)).toBe(15);
    expect(mapHeadAngle(-70, 35, 15)).toBe(-15);
  });

  it('範囲の途中は線形（少し振っただけで端に張り付かない）', () => {
    expect(mapHeadAngle(17.5, 35, 15)).toBeCloseTo(7.5, 6);
    // 生の角度をそのまま渡していたときは 17.5° で既に上限（15°）に張り付いていた。
    expect(mapHeadAngle(17.5, 35, 15)).toBeLessThan(15);
  });

  it('範囲が 0 なら動かさない（0 除算にしない）', () => {
    expect(mapHeadAngle(10, 0, 15)).toBe(0);
  });
});

describe('時定数', () => {
  it('フレーム間隔が違っても同じ時間で同じ所まで来る', () => {
    const tau = HEAD_TIME_CONSTANT_SECONDS;
    let coarse = 0;
    for (let step = 0; step < 6; step++) {
      coarse = smoothScalar(coarse, 1, smoothingFactor(1 / 30, tau));
    }
    let fine = 0;
    for (let step = 0; step < 12; step++) {
      fine = smoothScalar(fine, 1, smoothingFactor(1 / 60, tau));
    }
    expect(fine).toBeCloseTo(coarse, 12);
  });

  it('経過 0 では動かない（同じフレームで 2 回進めない）', () => {
    expect(smoothingFactor(0, HEAD_TIME_CONSTANT_SECONDS)).toBe(0);
    expect(smoothScalar(0.4, 1, smoothingFactor(0, HEAD_TIME_CONSTANT_SECONDS))).toBe(0.4);
  });

  it('行き過ぎない（1 フレームで目標を越えない）', () => {
    const value = smoothScalar(0, 1, smoothingFactor(1, HEAD_TIME_CONSTANT_SECONDS));
    expect(value).toBeLessThanOrEqual(1);
    expect(value).toBeGreaterThan(0.9);
  });
});
