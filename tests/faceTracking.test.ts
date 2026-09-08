// blendshape → 表情プリセットの対応（`domain/preview/faceTracking`）。
//
// **プリセットの本数は検査しない。** アセットは増える（口形が足される予定がある）ので、本数を
// 書いた検査は増えた瞬間に落ちるだけで何も守らない。守るのは「対応表が名指しするプリセットが
// アセットに在ること」。

import { describe, expect, it } from 'vitest';
import {
  CATEGORY_DEADBAND,
  EXPRESSION_TIME_CONSTANT_SECONDS,
  MAX_TOTAL_WEIGHT,
  TRACKING_ROWS,
  WINK_ASYMMETRY_THRESHOLD,
  applyDeadband,
  blendshapesToTargets,
  limitTotalWeight,
  missingCategories,
  resolveTrackingPlan,
  smoothScalar,
  smoothToward,
  smoothingFactor,
  splitBlinkAndWink,
  strongestPreset,
} from '../src/domain/preview/faceTracking';
import { loadPreview } from './asset';

/** 対応表そのものから作った並び（アセットに依らない単体検査用）。 */
const PRESET_NAMES: readonly string[] = [
  ...TRACKING_ROWS.map((row) => row.preset),
  'wink_left',
  'wink_right',
];

function plan(names: readonly string[] = PRESET_NAMES): ReturnType<typeof resolveTrackingPlan> {
  return resolveTrackingPlan(names);
}

function weightOf(names: readonly string[], weights: Float64Array, preset: string): number {
  const index = names.indexOf(preset);
  expect(index).toBeGreaterThanOrEqual(0);
  return weights[index];
}

/** デッドバンドを通したあとに `value` になるスコア。 */
function scoreFor(value: number): number {
  return CATEGORY_DEADBAND + value * (1 - CATEGORY_DEADBAND);
}

describe('対応表', () => {
  it('名指しするプリセットが実アセットに全部在る', () => {
    const preview = loadPreview();
    const resolved = resolveTrackingPlan(preview.expressionPresetNames);
    expect(resolved.unknownPresets).toEqual([]);
  });

  it('ウィンクのプリセットも実アセットに在る（まばたきの左右差がここへ流れる）', () => {
    const names = loadPreview().expressionPresetNames;
    expect(names).toContain('wink_left');
    expect(names).toContain('wink_right');
  });

  it('まばたきのカテゴリは対応表に無い（プリセットではなく置き換えで扱う）', () => {
    const categories = plan().categories;
    expect(categories).not.toContain('eyeBlinkLeft');
    expect(categories).not.toContain('eyeBlinkRight');
  });

  it('対応の理由が全部書いてある', () => {
    for (const row of TRACKING_ROWS) expect(row.reason.length).toBeGreaterThan(0);
  });

  it('アセットに無いプリセットは unknownPresets に出る', () => {
    const resolved = resolveTrackingPlan(['happy']);
    expect(resolved.unknownPresets).toContain('surprise');
    expect(resolved.unmappedPresets).toEqual([]);
  });

  it('対応表に無いプリセットは unmappedPresets に出る（後から足されたものはここへ来る）', () => {
    const resolved = resolveTrackingPlan([...PRESET_NAMES, 'viseme_a']);
    expect(resolved.unmappedPresets).toContain('viseme_a');
  });
});

describe('blendshape → 重み', () => {
  it('jawOpen が stretch_face を立てる', () => {
    const resolved = plan();
    const weights = new Float64Array(PRESET_NAMES.length);
    blendshapesToTargets(resolved, new Map([['jawOpen', scoreFor(1)]]), weights);
    expect(weightOf(PRESET_NAMES, weights, 'stretch_face')).toBeCloseTo(1, 6);
    expect(weightOf(PRESET_NAMES, weights, 'happy')).toBe(0);
  });

  it('左右対のカテゴリは両方満点で 1 になる', () => {
    const resolved = plan();
    const weights = new Float64Array(PRESET_NAMES.length);
    blendshapesToTargets(
      resolved,
      new Map([
        ['mouthSmileLeft', scoreFor(1)],
        ['mouthSmileRight', scoreFor(1)],
      ]),
      weights,
    );
    expect(weightOf(PRESET_NAMES, weights, 'happy')).toBeCloseTo(1, 6);
  });

  it('片側だけなら半分', () => {
    const resolved = plan();
    const weights = new Float64Array(PRESET_NAMES.length);
    blendshapesToTargets(resolved, new Map([['mouthSmileLeft', scoreFor(1)]]), weights);
    expect(weightOf(PRESET_NAMES, weights, 'happy')).toBeCloseTo(0.5, 6);
  });

  it('知らないカテゴリ名で落ちない（無い名前は 0）', () => {
    const resolved = plan();
    const weights = new Float64Array(PRESET_NAMES.length);
    const result = blendshapesToTargets(
      resolved,
      new Map([
        ['thereIsNoSuchCategory', 1],
        ['jawOpen', scoreFor(0.5)],
      ]),
      weights,
    );
    expect(result.blink).toBe(0);
    expect(weightOf(PRESET_NAMES, weights, 'stretch_face')).toBeCloseTo(0.5, 6);
  });

  it('空のスコアでも落ちない', () => {
    const resolved = plan();
    const weights = new Float64Array(PRESET_NAMES.length);
    expect(() => blendshapesToTargets(resolved, new Map(), weights)).not.toThrow();
    expect([...weights].every((value) => value === 0)).toBe(true);
  });

  it('重みの長さが違えば落とす', () => {
    expect(() => blendshapesToTargets(plan(), new Map(), new Float64Array(1))).toThrow();
  });

  it('デッドバンド以下は 0、超えたぶんは 0〜1 へ伸ばす', () => {
    expect(applyDeadband(CATEGORY_DEADBAND)).toBe(0);
    expect(applyDeadband(CATEGORY_DEADBAND / 2)).toBe(0);
    expect(applyDeadband(1)).toBeCloseTo(1, 12);
    expect(applyDeadband(scoreFor(0.25))).toBeCloseTo(0.25, 12);
  });

  it('無表情の雑音（全カテゴリが薄く立つ）を通さない', () => {
    const resolved = plan();
    const weights = new Float64Array(PRESET_NAMES.length);
    const noise = new Map(resolved.categories.map((name) => [name, CATEGORY_DEADBAND * 0.9]));
    blendshapesToTargets(resolved, noise, weights);
    expect([...weights].every((value) => value === 0)).toBe(true);
  });

  it('全部が満点でも合計が上限を超えない（重ねると顔が壊れる）', () => {
    const resolved = plan();
    const weights = new Float64Array(PRESET_NAMES.length);
    const all = new Map(resolved.categories.map((name) => [name, 1]));
    const result = blendshapesToTargets(resolved, all, weights);
    const total = [...weights].reduce((sum, value) => sum + value, 0);
    expect(total).toBeLessThanOrEqual(MAX_TOTAL_WEIGHT + 1e-9);
    // 縮める前の合計は上限を超えていた（＝抑制が実際に効いた）。
    expect(result.rawTotal).toBeGreaterThan(MAX_TOTAL_WEIGHT);
  });

  it('抑制は比例で縮める（表情の釣り合いを変えない）', () => {
    const weights = Float64Array.from([1, 2, 1]);
    limitTotalWeight(weights, 2);
    expect([...weights]).toEqual([0.5, 1, 0.5]);
    expect(weights[1] / weights[0]).toBe(2);
  });

  it('上限を超えていなければ触らない', () => {
    const weights = Float64Array.from([0.3, 0.2]);
    limitTotalWeight(weights, 2);
    expect([...weights]).toEqual([0.3, 0.2]);
  });

  it('missingCategories が返ってこなかったカテゴリを挙げる', () => {
    const resolved = plan();
    expect(missingCategories(resolved, resolved.categories)).toEqual([]);
    const missing = missingCategories(resolved, ['jawOpen']);
    expect(missing).not.toContain('jawOpen');
    expect(missing).toContain('mouthPucker');
  });

  it('strongestPreset がいちばん強いものを返す（弱ければ null）', () => {
    const resolved = plan();
    const weights = new Float64Array(PRESET_NAMES.length);
    expect(strongestPreset(resolved, weights)).toBeNull();
    weights[PRESET_NAMES.indexOf('pucker')] = 0.9;
    weights[PRESET_NAMES.indexOf('happy')] = 0.3;
    expect(strongestPreset(resolved, weights)).toBe('pucker');
  });
});

describe('まばたきとウィンクの切り分け', () => {
  it('両目を同じだけ閉じたらまばたき（ウィンクは立たない）', () => {
    const split = splitBlinkAndWink(0.95, 0.95);
    expect(split.blink).toBeCloseTo(0.95, 6);
    expect(split.winkLeft).toBe(0);
    expect(split.winkRight).toBe(0);
  });

  it('左右のわずかなズレは雑音として均す（min で下げない）', () => {
    const split = splitBlinkAndWink(0.9, 0.8);
    expect(split.blink).toBeCloseTo(0.85, 6);
    expect(split.winkLeft).toBe(0);
  });

  it('片目だけ閉じたらウィンク、まばたきは残りの共通ぶんだけ', () => {
    const split = splitBlinkAndWink(0.95, 0.05);
    expect(split.winkLeft).toBeGreaterThan(0.5);
    expect(split.winkRight).toBe(0);
    expect(split.blink).toBeLessThan(0.2);
  });

  it('もう片方も同じ形で立つ', () => {
    const split = splitBlinkAndWink(0.05, 0.95);
    expect(split.winkRight).toBeGreaterThan(0.5);
    expect(split.winkLeft).toBe(0);
  });

  it('閾値をまたいでも飛ばない（連続）', () => {
    const below = splitBlinkAndWink(0.5 + WINK_ASYMMETRY_THRESHOLD / 2 - 1e-6, 0.5 - WINK_ASYMMETRY_THRESHOLD / 2 + 1e-6);
    const above = splitBlinkAndWink(0.5 + WINK_ASYMMETRY_THRESHOLD / 2 + 1e-6, 0.5 - WINK_ASYMMETRY_THRESHOLD / 2 - 1e-6);
    expect(Math.abs(above.blink - below.blink)).toBeLessThan(1e-3);
    expect(Math.abs(above.winkLeft - below.winkLeft)).toBeLessThan(1e-3);
  });

  it('まばたきはプリセットへ流れず、ウィンクだけがプリセットへ出る', () => {
    const resolved = plan();
    const weights = new Float64Array(PRESET_NAMES.length);
    const result = blendshapesToTargets(
      resolved,
      new Map([
        ['eyeBlinkLeft', scoreFor(1)],
        ['eyeBlinkRight', scoreFor(1)],
      ]),
      weights,
    );
    expect(result.blink).toBeCloseTo(1, 6);
    expect(weightOf(PRESET_NAMES, weights, 'wink_left')).toBe(0);
    expect(weightOf(PRESET_NAMES, weights, 'wink_right')).toBe(0);
  });

  it('片目だけならウィンクのプリセットが立つ', () => {
    const resolved = plan();
    const weights = new Float64Array(PRESET_NAMES.length);
    const result = blendshapesToTargets(
      resolved,
      new Map([
        ['eyeBlinkLeft', scoreFor(1)],
        ['eyeBlinkRight', 0],
      ]),
      weights,
    );
    expect(weightOf(PRESET_NAMES, weights, 'wink_left')).toBeGreaterThan(0.5);
    expect(result.blink).toBeLessThan(0.3);
  });
});

describe('スムージング', () => {
  it('繰り返すと目標へ収束する', () => {
    const current = Float64Array.from([0, 1]);
    const target = Float64Array.from([1, 0]);
    const factor = smoothingFactor(1 / 60, EXPRESSION_TIME_CONSTANT_SECONDS);
    for (let step = 0; step < 600; step++) smoothToward(current, target, factor);
    expect(current[0]).toBeCloseTo(1, 6);
    expect(current[1]).toBeCloseTo(0, 6);
  });

  it('行き過ぎない（1 フレームで目標を越えない）', () => {
    const current = Float64Array.from([0]);
    const target = Float64Array.from([1]);
    smoothToward(current, target, smoothingFactor(1, EXPRESSION_TIME_CONSTANT_SECONDS));
    expect(current[0]).toBeLessThanOrEqual(1);
    expect(current[0]).toBeGreaterThan(0.9);
  });

  it('フレーム間隔が違っても同じ時間で同じ所まで来る', () => {
    const tau = EXPRESSION_TIME_CONSTANT_SECONDS;
    let coarse = 0;
    for (let step = 0; step < 6; step++) coarse = smoothScalar(coarse, 1, smoothingFactor(1 / 30, tau));
    let fine = 0;
    for (let step = 0; step < 12; step++) fine = smoothScalar(fine, 1, smoothingFactor(1 / 60, tau));
    expect(fine).toBeCloseTo(coarse, 12);
  });

  it('経過 0 では動かない（同じフレームで 2 回進めない）', () => {
    expect(smoothingFactor(0, EXPRESSION_TIME_CONSTANT_SECONDS)).toBe(0);
    expect(smoothScalar(0.4, 1, smoothingFactor(0, EXPRESSION_TIME_CONSTANT_SECONDS))).toBe(0.4);
  });
});
