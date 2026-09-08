// 表情アニメーションの収録と再生（`domain/preview/recording`）。

import { describe, expect, it } from 'vitest';
import { RecordingFileError } from '../src/domain/errors';
import {
  MAX_RECORDING_SECONDS,
  RECORDING_FORMAT_VERSION,
  appendFrame,
  parseRecording,
  recordingDurationSeconds,
  sampleRecording,
  serializeRecording,
  startRecording,
} from '../src/domain/preview/recording';
import { loadPreview } from './asset';

const NAMES = ['happy', 'surprise', 'pucker'] as const;

/** 時刻と重みの列から収録を作る。 */
function build(
  frames: readonly (readonly [number, readonly number[], number])[],
  names: readonly string[] = NAMES,
): ReturnType<typeof startRecording> {
  let recording = startRecording(names);
  for (const [time, weights, blink] of frames) {
    recording = appendFrame(recording, time, Float64Array.from(weights), blink).recording;
  }
  return recording;
}

describe('収録', () => {
  it('プリセット名を持つ（重みの意味は名前で決まる）', () => {
    const recording = startRecording(NAMES);
    expect(recording.presetNames).toEqual([...NAMES]);
    expect(recording.formatVersion).toBe(RECORDING_FORMAT_VERSION);
    expect(recordingDurationSeconds(recording)).toBe(0);
  });

  it('上限を超えた時刻は積まず full を立てる', () => {
    const recording = build([[1, [1, 0, 0], 0]]);
    const step = appendFrame(
      recording,
      MAX_RECORDING_SECONDS + 0.01,
      Float64Array.from([1, 0, 0]),
      0,
    );
    expect(step.full).toBe(true);
    expect(step.recording.frames).toHaveLength(1);
  });

  it('ちょうど上限の時刻は積む', () => {
    const step = appendFrame(
      startRecording(NAMES),
      MAX_RECORDING_SECONDS,
      Float64Array.from([1, 0, 0]),
      0,
    );
    expect(step.full).toBe(false);
    expect(step.recording.frames).toHaveLength(1);
  });

  it('時刻が進まないフレームは落とす（rAF は同じ時刻を 2 回渡してくる）', () => {
    let recording = startRecording(NAMES);
    recording = appendFrame(recording, 0.1, Float64Array.from([1, 0, 0]), 0).recording;
    recording = appendFrame(recording, 0.1, Float64Array.from([0, 1, 0]), 0).recording;
    recording = appendFrame(recording, 0.05, Float64Array.from([0, 0, 1]), 0).recording;
    expect(recording.frames).toHaveLength(1);
    expect(recording.frames[0].weights[0]).toBe(1);
  });

  it('重みの数が合わなければ落とす', () => {
    expect(() => appendFrame(startRecording(NAMES), 0.1, Float64Array.from([1]), 0)).toThrow();
  });

  it('保存する値は丸める（保存して読み直しても動きが変わらない）', () => {
    const recording = build([[0.123456, [0.1234567, 0, 0], 0.98765]]);
    const frame = recording.frames[0];
    const round = (value: number): number => Math.round(value * 1000);
    expect(frame.weights[0] * 1000).toBeCloseTo(round(0.1234567), 9);
    expect(frame.blink * 1000).toBeCloseTo(round(0.98765), 9);
  });
});

describe('再生', () => {
  it('フレームの上で正確に一致する', () => {
    const recording = build([
      [0, [0, 0, 0], 0],
      [0.5, [1, 0, 0], 0.4],
    ]);
    const weights = new Float64Array(NAMES.length);
    expect(sampleRecording(recording, 0.5, weights)).toBeCloseTo(0.4, 6);
    expect(weights[0]).toBeCloseTo(1, 6);
  });

  it('非等間隔のフレームでも時刻で補間する', () => {
    // 0 → 0.1 → 1.1（間隔が 10 倍違う）。番号で引くと 2 本目が「真ん中」になってしまう。
    const recording = build([
      [0, [0, 0, 0], 0],
      [0.1, [1, 0, 0], 1],
      [1.1, [0, 0, 0], 0],
    ]);
    const weights = new Float64Array(NAMES.length);
    sampleRecording(recording, 0.05, weights);
    expect(weights[0]).toBeCloseTo(0.5, 6);
    // 0.6 秒は 0.1〜1.1 の真ん中。
    const blink = sampleRecording(recording, 0.6, weights);
    expect(weights[0]).toBeCloseTo(0.5, 6);
    expect(blink).toBeCloseTo(0.5, 6);
  });

  it('範囲の外は端で留める', () => {
    const recording = build([
      [0.2, [0.5, 0, 0], 0.5],
      [0.4, [1, 0, 0], 1],
    ]);
    const weights = new Float64Array(NAMES.length);
    expect(sampleRecording(recording, 0, weights)).toBeCloseTo(0.5, 6);
    expect(weights[0]).toBeCloseTo(0.5, 6);
    expect(sampleRecording(recording, 99, weights)).toBeCloseTo(1, 6);
    expect(weights[0]).toBeCloseTo(1, 6);
  });

  it('フレームが無ければ全部 0', () => {
    const weights = Float64Array.from([1, 1, 1]);
    expect(sampleRecording(startRecording(NAMES), 1, weights)).toBe(0);
    expect([...weights]).toEqual([0, 0, 0]);
  });

  it('たくさんのフレームでも中央付近を正しく引く（二分探索）', () => {
    const frames: [number, number[], number][] = [];
    for (let index = 0; index < 500; index++) {
      frames.push([index * 0.02, [index / 500, 0, 0], 0]);
    }
    const recording = build(frames);
    const weights = new Float64Array(NAMES.length);
    sampleRecording(recording, 250 * 0.02, weights);
    expect(weights[0]).toBeCloseTo(250 / 500, 3);
  });
});

describe('読み込みの検証', () => {
  const roundTrip = (
    recording: ReturnType<typeof startRecording>,
    names: readonly string[],
  ): ReturnType<typeof parseRecording> =>
    parseRecording(serializeRecording(recording), names);

  it('保存して読むと同じものが返る', () => {
    const recording = build([
      [0, [0, 0, 0], 0],
      [0.4, [0.5, 0.25, 0], 0.75],
    ]);
    const loaded = roundTrip(recording, NAMES);
    expect(loaded.droppedPresets).toEqual([]);
    expect(loaded.recording.frames).toEqual(recording.frames);
  });

  it('プリセットの並びが変わっても名前で合わせ直す', () => {
    const recording = build([[0.1, [1, 0.5, 0], 0]]);
    const loaded = roundTrip(recording, ['pucker', 'surprise', 'happy']);
    expect(loaded.recording.frames[0].weights).toEqual([0, 0.5, 1]);
  });

  it('プリセットが増えていても読める（増えたぶんは 0）', () => {
    const recording = build([[0.1, [1, 0, 0], 0]]);
    const loaded = roundTrip(recording, [...NAMES, 'viseme_a']);
    expect(loaded.recording.frames[0].weights).toEqual([1, 0, 0, 0]);
    expect(loaded.droppedPresets).toEqual([]);
  });

  it('今のアセットに無い名前は落とす（黙って別の表情にしない）', () => {
    const recording = build([[0.1, [1, 0.5, 0.25], 0]], ['happy', 'gone_away', 'pucker']);
    const loaded = parseRecording(serializeRecording(recording), NAMES);
    expect(loaded.droppedPresets).toEqual(['gone_away']);
    // happy=1 / surprise=0（元に無い）/ pucker=0.25。0.5 はどこにも入らない。
    expect(loaded.recording.frames[0].weights).toEqual([1, 0, 0.25]);
  });

  it('名前が 1 つも一致しなければ落とす（別のアセットで録ったもの）', () => {
    const recording = build([[0.1, [1, 0, 0], 0]], ['a', 'b', 'c']);
    expect(() => parseRecording(serializeRecording(recording), NAMES)).toThrow(RecordingFileError);
  });

  it('上限を超えるフレームは切る', () => {
    const text = JSON.stringify({
      formatVersion: RECORDING_FORMAT_VERSION,
      presetNames: NAMES,
      frames: [
        { timeSeconds: 1, weights: [1, 0, 0], blink: 0 },
        { timeSeconds: MAX_RECORDING_SECONDS + 5, weights: [0, 1, 0], blink: 0 },
      ],
    });
    const loaded = parseRecording(text, NAMES);
    expect(loaded.truncated).toBe(true);
    expect(loaded.recording.frames).toHaveLength(1);
  });

  it('時刻が逆行したフレームは落として数を返す', () => {
    const text = JSON.stringify({
      formatVersion: RECORDING_FORMAT_VERSION,
      presetNames: NAMES,
      frames: [
        { timeSeconds: 1, weights: [1, 0, 0], blink: 0 },
        { timeSeconds: 0.5, weights: [0, 1, 0], blink: 0 },
        { timeSeconds: 2, weights: [0, 0, 1], blink: 0 },
      ],
    });
    const loaded = parseRecording(text, NAMES);
    expect(loaded.droppedFrames).toBe(1);
    expect(loaded.recording.frames).toHaveLength(2);
  });

  it('壊れた JSON', () => {
    expect(() => parseRecording('{', NAMES)).toThrow(RecordingFileError);
  });

  it('オブジェクトでない', () => {
    expect(() => parseRecording('[1,2,3]', NAMES)).toThrow(RecordingFileError);
    expect(() => parseRecording('12', NAMES)).toThrow(RecordingFileError);
  });

  it('形式のバージョンが違う', () => {
    const text = JSON.stringify({
      formatVersion: RECORDING_FORMAT_VERSION + 1,
      presetNames: NAMES,
      frames: [{ timeSeconds: 0, weights: [0, 0, 0], blink: 0 }],
    });
    expect(() => parseRecording(text, NAMES)).toThrow(RecordingFileError);
  });

  it('weights の長さが presetNames と合わない', () => {
    const text = JSON.stringify({
      formatVersion: RECORDING_FORMAT_VERSION,
      presetNames: NAMES,
      frames: [{ timeSeconds: 0, weights: [0, 0], blink: 0 }],
    });
    expect(() => parseRecording(text, NAMES)).toThrow(RecordingFileError);
  });

  it('weights に数でない値がある', () => {
    const text = JSON.stringify({
      formatVersion: RECORDING_FORMAT_VERSION,
      presetNames: NAMES,
      frames: [{ timeSeconds: 0, weights: [0, 'x', 0], blink: 0 }],
    });
    expect(() => parseRecording(text, NAMES)).toThrow(RecordingFileError);
  });

  it('フレームが 1 つも無い', () => {
    const text = JSON.stringify({
      formatVersion: RECORDING_FORMAT_VERSION,
      presetNames: NAMES,
      frames: [],
    });
    expect(() => parseRecording(text, NAMES)).toThrow(RecordingFileError);
  });

  it('実アセットのプリセット名でそのまま往復する', () => {
    const names = loadPreview().expressionPresetNames;
    const weights = new Float64Array(names.length);
    weights[0] = 0.5;
    const recording = appendFrame(startRecording(names), 0.1, weights, 0.25).recording;
    const loaded = parseRecording(serializeRecording(recording), names);
    expect(loaded.droppedPresets).toEqual([]);
    expect(loaded.recording.frames[0].weights[0]).toBeCloseTo(0.5, 6);
  });
});
