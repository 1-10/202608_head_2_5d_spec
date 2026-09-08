// 表情の駆動源が「渡された枠を埋める」という約束を守っているかの検査。
//
// **この検査が無かったせいで詰まった。** ビューアーの差し込み口がプリセットの重み（長さ 25）を
// 渡していて、駆動源は係数（383）を期待して黙って何もしなかった。同じ `Float64Array` なので
// TypeScript では捕まらない。**今は名前の付いた枠（`ExpressionSlots`）で渡す**が、どちらの枠を
// 埋めるかは駆動源ごとに違うので、実際に呼んで確かめる。

import { describe, expect, it } from 'vitest';
import { loadPreview } from './asset';
import { zeroCoefficients } from '../src/domain/preview/expression';
import type { ExpressionSlots } from '../src/presentation/viewer';
import {
  appendFrame,
  coefficientQuanta,
  startRecording,
} from '../src/domain/preview/recording';
import { RecordingPlayer } from '../src/presentation/recordingPlayer';
import { VisemeDriver } from '../src/presentation/visemeDriver';
import { DEFAULT_VIEW_SETTINGS } from '../src/presentation/viewSettings';

function peak(values: Float64Array): number {
  let maximum = 0;
  for (const value of values) maximum = Math.max(maximum, Math.abs(value));
  return maximum;
}

/** ビューアーが毎フレーム渡す枠と同じもの（どちらも 0 で来る）。 */
function slotsFor(presetCount: number, componentCount: number): ExpressionSlots {
  return {
    weights: new Float64Array(presetCount),
    coefficients: new Float64Array(componentCount),
  };
}

describe('口形の連続再生の駆動源', () => {
  // 口形はプリセットなので**重みの枠**を埋める（速い経路に乗る）。
  it('重みの枠を埋める（係数の枠は触らない）', () => {
    const preview = loadPreview();
    const driver = new VisemeDriver();
    driver.setPreview(preview);
    driver.apply(DEFAULT_VIEW_SETTINGS);
    driver.play();
    const frame = driver.frame;
    expect(frame).not.toBeNull();

    const slots = slotsFor(preview.presetCount, preview.componentCount);
    // 1 回目は経過 0 なので重み 0（台形の始まり）。立ち上がりの途中まで進めてから見る。
    const step = DEFAULT_VIEW_SETTINGS.visemeFadeSeconds / 2;
    frame!(slots, step);
    const label = frame!(slots, step);
    expect(label).toContain('口形');
    expect(peak(slots.weights)).toBeGreaterThan(0);
    expect(peak(slots.coefficients)).toBe(0);
  });

  it('止めたら口を返す（駆動しない）', () => {
    const preview = loadPreview();
    const driver = new VisemeDriver();
    driver.setPreview(preview);
    driver.apply(DEFAULT_VIEW_SETTINGS);
    expect(driver.frame).toBeNull();
  });
});

describe('録画の再生の駆動源', () => {
  // 収録は係数で持っているのでプリセットの重みへは戻せない。**係数の枠**を埋める。
  it('係数の枠を埋める（重みの枠は触らない）', () => {
    const preview = loadPreview();
    const names = preview.expressionComponentNames;
    const quanta = coefficientQuanta(preview.expressionBasisScales);
    const frames = new Float64Array(names.length);
    frames[0] = 0.5;
    let recording = startRecording(names);
    recording = appendFrame(recording, 0, new Float64Array(names.length), quanta).recording;
    recording = appendFrame(recording, 0.5, frames, quanta).recording;

    const player = new RecordingPlayer();
    player.setRecording(recording);
    expect(player.play()).toBe(true);

    const slots = slotsFor(preview.presetCount, preview.componentCount);
    const label = player.expression(slots, 0.5);
    expect(label).not.toBeNull();
    expect(slots.coefficients[0]).toBeCloseTo(0.5, 3);
    expect(peak(slots.weights)).toBe(0);
    // まばたきは収録した係数に畳んである。別に返すと二重に掛かって閉じ切ったまま固まる。
    expect(player.blink()).toBe(0);
  });

  it('再生していなければ何も埋めない', () => {
    const preview = loadPreview();
    const player = new RecordingPlayer();
    const slots = slotsFor(preview.presetCount, preview.componentCount);
    expect(player.expression(slots, 0.1)).toBeNull();
    expect(peak(slots.coefficients)).toBe(0);
    expect(peak(slots.weights)).toBe(0);
  });
});
