// 表情アニメーションの収録と再生（`domain/preview/recording`）。
//
// 収録するのは**表情基底の成分の係数**。一時プリセットの重みで持っていたが、あれは web 固有の
// 25 本に依存する形で Unity では再生できなかった。

import { describe, expect, it } from 'vitest';
import { RecordingFileError } from '../src/domain/errors';
import {
  DISPLACEMENT_QUANTUM_METERS,
  MAX_RECORDING_SECONDS,
  RECORDING_FORMAT_VERSION,
  appendFrame,
  coefficientQuanta,
  parseRecording,
  recordingDurationSeconds,
  sampleRecording,
  serializeRecording,
  startRecording,
} from '../src/domain/preview/recording';
import { loadPreview } from './asset';

const NAMES = ['lower_face_region_000', 'left_eye_region_000', 'tongue_000'] as const;
/** 刻みは成分ごと。検査では 1/1000 に揃えて読みやすくする。 */
const QUANTA = Float64Array.from([0.001, 0.001, 0.001]);

/** 時刻と係数の列から収録を作る。 */
function build(
  frames: readonly (readonly [number, readonly number[]])[],
  names: readonly string[] = NAMES,
  quanta: Float64Array = QUANTA,
): ReturnType<typeof startRecording> {
  let recording = startRecording(names);
  for (const [time, coefficients] of frames) {
    recording = appendFrame(recording, time, Float64Array.from(coefficients), quanta).recording;
  }
  return recording;
}

describe('収録', () => {
  it('成分名を持つ（係数の意味は名前で決まる）', () => {
    const recording = startRecording(NAMES);
    expect(recording.componentNames).toEqual([...NAMES]);
    expect(recording.formatVersion).toBe(RECORDING_FORMAT_VERSION);
    expect(recordingDurationSeconds(recording)).toBe(0);
  });

  it('上限を超えた時刻は積まず full を立てる', () => {
    const recording = build([[1, [1, 0, 0]]]);
    const step = appendFrame(
      recording,
      MAX_RECORDING_SECONDS + 0.01,
      Float64Array.from([1, 0, 0]),
      QUANTA,
    );
    expect(step.full).toBe(true);
    expect(step.recording.frames).toHaveLength(1);
  });

  it('ちょうど上限の時刻は積む', () => {
    const step = appendFrame(
      startRecording(NAMES),
      MAX_RECORDING_SECONDS,
      Float64Array.from([1, 0, 0]),
      QUANTA,
    );
    expect(step.full).toBe(false);
    expect(step.recording.frames).toHaveLength(1);
  });

  it('時刻が進まないフレームは落とす（rAF は同じ時刻を 2 回渡してくる）', () => {
    let recording = startRecording(NAMES);
    recording = appendFrame(recording, 0.1, Float64Array.from([1, 0, 0]), QUANTA).recording;
    recording = appendFrame(recording, 0.1, Float64Array.from([0, 1, 0]), QUANTA).recording;
    recording = appendFrame(recording, 0.05, Float64Array.from([0, 0, 1]), QUANTA).recording;
    expect(recording.frames).toHaveLength(1);
    expect(recording.frames[0].coefficients[0]).toBe(1);
  });

  it('係数の数が合わなければ落とす', () => {
    expect(() =>
      appendFrame(startRecording(NAMES), 0.1, Float64Array.from([1]), QUANTA),
    ).toThrow();
  });

  it('刻みの数が合わなければ落とす', () => {
    expect(() =>
      appendFrame(
        startRecording(NAMES),
        0.1,
        Float64Array.from([1, 0, 0]),
        Float64Array.from([0.001]),
      ),
    ).toThrow();
  });

  it('保存する値は丸める（保存して読み直しても動きが変わらない）', () => {
    const recording = build([[0.123456, [0.1234567, -0.9876543, 0]]]);
    const frame = recording.frames[0];
    expect(frame.coefficients[0] * 1000).toBeCloseTo(123, 9);
    expect(frame.coefficients[1] * 1000).toBeCloseTo(-988, 9);
    expect(frame.timeSeconds * 1000).toBeCloseTo(123, 9);
  });

  // **刻みは成分ごとに変える。** 係数 1 あたりの変位は成分によって 100 倍近く違うので、一律の
  // 刻みだと小さい成分に無駄な桁を使い、大きい成分は粗くなる。
  it('刻みは「係数 1 あたりの変位」から作る（変位で 1um に揃う）', () => {
    const scales = Float64Array.from([0.0001, 0.01]);
    const quanta = coefficientQuanta(scales);
    expect(quanta[0] * scales[0]).toBeCloseTo(DISPLACEMENT_QUANTUM_METERS, 15);
    expect(quanta[1] * scales[1]).toBeCloseTo(DISPLACEMENT_QUANTUM_METERS, 15);
    // 変位の大きい成分ほど刻みが細かい。
    expect(quanta[1]).toBeLessThan(quanta[0]);
  });
});

describe('再生', () => {
  it('フレームの上で正確に一致する', () => {
    const recording = build([
      [0, [0, 0, 0]],
      [0.5, [1, 0.4, 0]],
    ]);
    const coefficients = new Float64Array(NAMES.length);
    sampleRecording(recording, 0.5, coefficients);
    expect(coefficients[0]).toBeCloseTo(1, 6);
    expect(coefficients[1]).toBeCloseTo(0.4, 6);
  });

  it('非等間隔のフレームでも時刻で補間する', () => {
    // 0 → 0.1 → 1.1（間隔が 10 倍違う）。番号で引くと 2 本目が「真ん中」になってしまう。
    const recording = build([
      [0, [0, 0, 0]],
      [0.1, [1, 1, 0]],
      [1.1, [0, 0, 0]],
    ]);
    const coefficients = new Float64Array(NAMES.length);
    sampleRecording(recording, 0.05, coefficients);
    expect(coefficients[0]).toBeCloseTo(0.5, 6);
    // 0.6 秒は 0.1〜1.1 の真ん中。
    sampleRecording(recording, 0.6, coefficients);
    expect(coefficients[0]).toBeCloseTo(0.5, 6);
    expect(coefficients[1]).toBeCloseTo(0.5, 6);
  });

  it('範囲の外は端で留める', () => {
    const recording = build([
      [0.2, [0.5, 0, 0]],
      [0.4, [1, 0, 0]],
    ]);
    const coefficients = new Float64Array(NAMES.length);
    sampleRecording(recording, 0, coefficients);
    expect(coefficients[0]).toBeCloseTo(0.5, 6);
    sampleRecording(recording, 99, coefficients);
    expect(coefficients[0]).toBeCloseTo(1, 6);
  });

  it('フレームが無ければ全部 0', () => {
    const coefficients = Float64Array.from([1, 1, 1]);
    sampleRecording(startRecording(NAMES), 1, coefficients);
    expect([...coefficients]).toEqual([0, 0, 0]);
  });

  it('たくさんのフレームでも中央付近を正しく引く（二分探索）', () => {
    const frames: [number, number[]][] = [];
    for (let index = 0; index < 500; index++) {
      frames.push([index * 0.02, [index / 500, 0, 0]]);
    }
    const recording = build(frames);
    const coefficients = new Float64Array(NAMES.length);
    sampleRecording(recording, 250 * 0.02, coefficients);
    expect(coefficients[0]).toBeCloseTo(250 / 500, 3);
  });
});

describe('読み込みの検証', () => {
  const roundTrip = (
    recording: ReturnType<typeof startRecording>,
    names: readonly string[],
  ): ReturnType<typeof parseRecording> => parseRecording(serializeRecording(recording), names);

  it('保存して読むと同じものが返る', () => {
    const recording = build([
      [0, [0, 0, 0]],
      [0.4, [0.5, 0.25, -0.125]],
    ]);
    const loaded = roundTrip(recording, NAMES);
    expect(loaded.droppedComponents).toEqual([]);
    expect(loaded.recording.frames).toEqual(recording.frames);
  });

  it('成分の並びが変わっても名前で合わせ直す', () => {
    const recording = build([[0.1, [1, 0.5, 0]]]);
    const loaded = roundTrip(recording, [NAMES[2], NAMES[1], NAMES[0]]);
    expect(loaded.recording.frames[0].coefficients).toEqual([0, 0.5, 1]);
  });

  it('成分が増えていても読める（増えたぶんは 0）', () => {
    const recording = build([[0.1, [1, 0, 0]]]);
    const loaded = roundTrip(recording, [...NAMES, 'pupils_mean']);
    expect(loaded.recording.frames[0].coefficients).toEqual([1, 0, 0, 0]);
    expect(loaded.droppedComponents).toEqual([]);
  });

  it('今のアセットに無い名前は落とす（黙って別の表情にしない）', () => {
    const recording = build([[0.1, [1, 0.5, 0.25]]], [NAMES[0], 'gone_away', NAMES[2]]);
    const loaded = parseRecording(serializeRecording(recording), NAMES);
    expect(loaded.droppedComponents).toEqual(['gone_away']);
    // 1 本目 = 1 / 2 本目（元に無い）= 0 / 3 本目 = 0.25。0.5 はどこにも入らない。
    expect(loaded.recording.frames[0].coefficients).toEqual([1, 0, 0.25]);
  });

  it('名前が 1 つも一致しなければ落とす（別のアセットで録ったもの）', () => {
    const recording = build([[0.1, [1, 0, 0]]], ['a', 'b', 'c']);
    expect(() => parseRecording(serializeRecording(recording), NAMES)).toThrow(RecordingFileError);
  });

  it('上限を超えるフレームは切る', () => {
    const text = JSON.stringify({
      formatVersion: RECORDING_FORMAT_VERSION,
      componentNames: NAMES,
      frames: [
        { timeSeconds: 1, coefficients: [1, 0, 0] },
        { timeSeconds: MAX_RECORDING_SECONDS + 5, coefficients: [0, 1, 0] },
      ],
    });
    const loaded = parseRecording(text, NAMES);
    expect(loaded.truncated).toBe(true);
    expect(loaded.recording.frames).toHaveLength(1);
  });

  it('時刻が逆行したフレームは落として数を返す', () => {
    const text = JSON.stringify({
      formatVersion: RECORDING_FORMAT_VERSION,
      componentNames: NAMES,
      frames: [
        { timeSeconds: 1, coefficients: [1, 0, 0] },
        { timeSeconds: 0.5, coefficients: [0, 1, 0] },
        { timeSeconds: 2, coefficients: [0, 0, 1] },
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
    for (const version of [RECORDING_FORMAT_VERSION - 1, RECORDING_FORMAT_VERSION + 1]) {
      const text = JSON.stringify({
        formatVersion: version,
        componentNames: NAMES,
        frames: [{ timeSeconds: 0, coefficients: [0, 0, 0] }],
      });
      expect(() => parseRecording(text, NAMES)).toThrow(RecordingFileError);
    }
  });

  it('coefficients の長さが componentNames と合わない', () => {
    const text = JSON.stringify({
      formatVersion: RECORDING_FORMAT_VERSION,
      componentNames: NAMES,
      frames: [{ timeSeconds: 0, coefficients: [0, 0] }],
    });
    expect(() => parseRecording(text, NAMES)).toThrow(RecordingFileError);
  });

  it('coefficients に数でない値がある', () => {
    const text = JSON.stringify({
      formatVersion: RECORDING_FORMAT_VERSION,
      componentNames: NAMES,
      frames: [{ timeSeconds: 0, coefficients: [0, 'x', 0] }],
    });
    expect(() => parseRecording(text, NAMES)).toThrow(RecordingFileError);
  });

  it('フレームが 1 つも無い', () => {
    const text = JSON.stringify({
      formatVersion: RECORDING_FORMAT_VERSION,
      componentNames: NAMES,
      frames: [],
    });
    expect(() => parseRecording(text, NAMES)).toThrow(RecordingFileError);
  });

  it('実アセットの成分名でそのまま往復する', () => {
    const preview = loadPreview();
    const names = preview.expressionComponentNames;
    const quanta = coefficientQuanta(preview.expressionBasisScales);
    const coefficients = new Float64Array(names.length);
    coefficients[0] = 0.5;
    const recording = appendFrame(startRecording(names), 0.1, coefficients, quanta).recording;
    const loaded = parseRecording(serializeRecording(recording), names);
    expect(loaded.droppedComponents).toEqual([]);
    // **係数そのものは丸まる。** 揃うのは変位で、そこが 1um 以内であることが契約。
    const stored = loaded.recording.frames[0].coefficients[0];
    const displacementError = Math.abs(stored - 0.5) * preview.expressionBasisScales[0];
    expect(displacementError).toBeLessThanOrEqual(DISPLACEMENT_QUANTUM_METERS);
  });
});

describe('時刻の丸めと単調増加', () => {
  it('丸めると同じ時刻になるフレームは積まない（保存 → 読み込みで食い違わせない）', () => {
    const names = ['a', 'b'];
    const quanta = Float64Array.from([0.001, 0.001]);
    let recording = startRecording(names);
    const coefficients = new Float64Array([0.5, 0.25]);
    recording = appendFrame(recording, 0.0016, coefficients, quanta).recording;
    recording = appendFrame(recording, 0.00201, coefficients, quanta).recording;
    expect(recording.frames).toHaveLength(1);

    // 丸めて別の値になるものは積む。
    recording = appendFrame(recording, 0.0031, coefficients, quanta).recording;
    expect(recording.frames).toHaveLength(2);
    const times = recording.frames.map((frame) => frame.timeSeconds);
    expect(times[1]).toBeGreaterThan(times[0]);
  });

  it('積んだ時刻は厳密に増加する（読み込みが逆行として落とさない）', () => {
    const names = ['a'];
    const quanta = Float64Array.from([0.001]);
    let recording = startRecording(names);
    const coefficients = new Float64Array([0]);
    for (let step = 0; step < 200; step++) {
      recording = appendFrame(recording, step * 0.0004, coefficients, quanta).recording;
    }
    const times = recording.frames.map((frame) => frame.timeSeconds);
    for (let index = 1; index < times.length; index++) {
      expect(times[index]).toBeGreaterThan(times[index - 1]);
    }
  });
});
