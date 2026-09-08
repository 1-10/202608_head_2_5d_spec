// 表情の駆動源が「係数を埋める」という約束を守っているかの検査。
//
// **この検査が無かったせいで詰まった。** ビューアーの差し込み口がプリセットの重み（長さ 25）を
// 渡していて、駆動源は係数（383）を期待して黙って何もしなかった。TypeScript はどちらも
// `Float64Array` なので捕まえられない。**長さの契約を実際に呼んで確かめる。**

import { describe, expect, it } from 'vitest';
import { loadPreview } from './asset';
import { zeroCoefficients } from '../src/domain/preview/expression';
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

describe('口形の連続再生の駆動源', () => {
  it('渡された係数の配列を埋める（長さは成分数）', () => {
    const preview = loadPreview();
    const driver = new VisemeDriver();
    driver.setPreview(preview);
    driver.apply(DEFAULT_VIEW_SETTINGS);
    driver.play();
    const frame = driver.frame;
    expect(frame).not.toBeNull();

    const coefficients = zeroCoefficients(preview);
    expect(coefficients.length).toBe(preview.componentCount);
    // 1 回目は経過 0 なので重み 0（台形の始まり）。立ち上がりの途中まで進めてから見る。
    const step = DEFAULT_VIEW_SETTINGS.visemeFadeSeconds / 2;
    frame!(coefficients, step);
    const label = frame!(coefficients, step);
    expect(label).toContain('口形');
    expect(peak(coefficients)).toBeGreaterThan(0);
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
  it('渡された係数の配列を埋める（長さは成分数）', () => {
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

    const coefficients = zeroCoefficients(preview);
    const label = player.expression(coefficients, 0.5);
    expect(label).not.toBeNull();
    expect(coefficients[0]).toBeCloseTo(0.5, 3);
    // まばたきは係数に畳んであるので、別に返さない（二重に掛かると閉じ切ったまま固まる）。
    expect(player.blink()).toBe(0);
  });

  it('再生していなければ何も埋めない', () => {
    const preview = loadPreview();
    const player = new RecordingPlayer();
    const coefficients = zeroCoefficients(preview);
    expect(player.expression(coefficients, 0.1)).toBeNull();
    expect(peak(coefficients)).toBe(0);
  });
});
