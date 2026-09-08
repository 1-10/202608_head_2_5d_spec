// 表情アニメーションの収録と再生（純粋計算）。
//
// 収録するのは**プリセットの重みとまばたき量**だけで、カメラ映像も landmark も残さない。3D ビューを
// 駆動するのに要るのはそれだけであり、顔の画像を保存しないで済む方が扱いが軽い。
//
// ## 重みは名前で持つ
//
// 保存するのは「重みの並び」ではなく「プリセット名 → 重み」。アセットのプリセットが増減・並び替え
// されると、index で持った重みは**黙って別の表情になる**（`viseme_*` のようなプリセットが後から
// 足されるのは実際に起きる）。名前で突き合わせて、**アセットに無い名前は落とす** — 黙って別の顔に
// なるより、その表情が出ない方がよい。
//
// ## 時刻は等間隔ではない
//
// フレームは requestAnimationFrame の刻みで積む。**等間隔で来ない**（重いフレーム・タブの非表示・
// 表示更新レートの違い）ので、時刻を明示して持ち、再生は時刻で補間する。「n 番目のフレーム」で
// 引くと、収録時と再生時で表示更新レートが違うだけで速さが変わる。

import { RecordingFileError } from '../errors';

/** 収録データの形式。**互換を壊す変更のたびに上げる**（読む側は一致しなければ落とす）。 */
export const RECORDING_FORMAT_VERSION = 1;

/** 収録の上限（秒）。これを超える時刻のフレームは積まない。 */
export const MAX_RECORDING_SECONDS = 30;

/**
 * 保存する重みの刻み。
 *
 * 60fps × 30 秒 × プリセット数のぶんだけ数が並ぶので、桁を落とさないと JSON が無駄に太る。
 * **積む時点で丸める** — 保存してから読み直すと動きが変わる、という差を作らないため。
 * 1/1000 は変位にすると 0.1mm 未満で、画面では見えない。
 */
export const WEIGHT_QUANTUM = 0.001;

/** 1 フレーム。 */
export interface RecordedFrame {
  /** 収録開始からの秒。**厳密に増加する**（同じ時刻・逆行は積む側で落とす）。 */
  readonly timeSeconds: number;
  /** プリセットごとの重み。並びは `ExpressionRecording.presetNames`。 */
  readonly weights: readonly number[];
  /** まばたき量（0〜1）。 */
  readonly blink: number;
}

/** 収録したもの。**そのまま JSON にできる形**にしておく（保存の段で詰め替えない）。 */
export interface ExpressionRecording {
  readonly formatVersion: number;
  /** 収録時のプリセット名の並び（重みの意味の正本）。 */
  readonly presetNames: readonly string[];
  readonly frames: readonly RecordedFrame[];
}

/** 空の収録を作る。 */
export function startRecording(presetNames: readonly string[]): ExpressionRecording {
  return {
    formatVersion: RECORDING_FORMAT_VERSION,
    presetNames: [...presetNames],
    frames: [],
  };
}

/** 収録の長さ（秒）。フレームが無ければ 0。 */
export function recordingDurationSeconds(recording: ExpressionRecording): number {
  const frames = recording.frames;
  return frames.length === 0 ? 0 : frames[frames.length - 1].timeSeconds;
}

/**
 * 1 フレーム積む。
 *
 * **上限を超えたら積まない**（`full` が立つので、呼ぶ側はそこで録画を止める）。時刻が前のフレーム
 * 以下なら落とす — requestAnimationFrame は同じ時刻を 2 回渡してくることがあり、そのまま積むと
 * 再生の補間で 0 除算になる。
 */
export function appendFrame(
  recording: ExpressionRecording,
  timeSeconds: number,
  weights: Float64Array | readonly number[],
  blink: number,
  maximumSeconds = MAX_RECORDING_SECONDS,
): { recording: ExpressionRecording; full: boolean } {
  if (!Number.isFinite(timeSeconds) || timeSeconds < 0) return { recording, full: false };
  if (timeSeconds > maximumSeconds) return { recording, full: true };
  const frames = recording.frames;
  const last = frames.length === 0 ? null : frames[frames.length - 1];
  // **量子化した後の値で比べる。** 格納するのは丸めた時刻なので、生の値で比べると 0.0016 と
  // 0.00201 のように「生では増えているが丸めると同じ」フレームが並び、`timeSeconds` が厳密に
  // 増加するという契約が破れる（保存 → 読み込みで `parseRecording` が逆行として落とすため、
  // 録ったものと読み直したものが食い違う）。
  const quantized = quantize(timeSeconds);
  if (last !== null && quantized <= last.timeSeconds) return { recording, full: false };
  if (weights.length !== recording.presetNames.length) {
    throw new Error(
      `重みが ${weights.length} 個（期待 ${recording.presetNames.length}）`,
    );
  }
  const rounded: number[] = [];
  for (let index = 0; index < weights.length; index++) rounded.push(quantize(weights[index]));
  return {
    recording: {
      ...recording,
      frames: [
        ...frames,
        { timeSeconds: quantized, weights: rounded, blink: quantize(clamp01(blink)) },
      ],
    },
    full: false,
  };
}

function quantize(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.round(value / WEIGHT_QUANTUM) * WEIGHT_QUANTUM;
}

/**
 * 任意の時刻の重みを取り出す（`weights` を破壊的に埋める）。
 *
 * フレーム間は線形補間する。**等間隔を仮定しない**（前後のフレームの時刻から比を作る）。範囲外は
 * 端のフレームで留める。
 *
 * @param weights 長さは `recording.presetNames.length`
 * @returns その時刻のまばたき量
 */
export function sampleRecording(
  recording: ExpressionRecording,
  timeSeconds: number,
  weights: Float64Array,
): number {
  const frames = recording.frames;
  if (weights.length !== recording.presetNames.length) {
    throw new Error(`重みが ${weights.length} 個（期待 ${recording.presetNames.length}）`);
  }
  weights.fill(0);
  if (frames.length === 0) return 0;
  if (timeSeconds <= frames[0].timeSeconds) return copyFrame(frames[0], weights);
  const lastFrame = frames[frames.length - 1];
  if (timeSeconds >= lastFrame.timeSeconds) return copyFrame(lastFrame, weights);

  // 時刻で二分探索（フレームは厳密増加）。`high` が「時刻を超える最初のフレーム」になる。
  let low = 0;
  let high = frames.length - 1;
  while (high - low > 1) {
    const middle = (low + high) >> 1;
    if (frames[middle].timeSeconds <= timeSeconds) low = middle;
    else high = middle;
  }
  const before = frames[low];
  const after = frames[high];
  const span = after.timeSeconds - before.timeSeconds;
  const t = span > 0 ? (timeSeconds - before.timeSeconds) / span : 0;
  for (let index = 0; index < weights.length; index++) {
    weights[index] = before.weights[index] + (after.weights[index] - before.weights[index]) * t;
  }
  return before.blink + (after.blink - before.blink) * t;
}

function copyFrame(frame: RecordedFrame, weights: Float64Array): number {
  for (let index = 0; index < weights.length; index++) weights[index] = frame.weights[index];
  return frame.blink;
}

/** 保存する形（JSON 文字列）。 */
export function serializeRecording(recording: ExpressionRecording): string {
  return JSON.stringify(recording);
}

/** 読み込みの結果。**落としたものを黙らせない。** */
export interface LoadedRecording {
  /** 今のアセットの並びへ移し替えた収録。 */
  readonly recording: ExpressionRecording;
  /** 今のアセットに無くて落としたプリセット名。 */
  readonly droppedPresets: readonly string[];
  /** 時刻が逆行・重複していて落としたフレーム数。 */
  readonly droppedFrames: number;
  /** 上限を超えて切り捨てたか。 */
  readonly truncated: boolean;
}

/**
 * 保存したものを読み、**今のアセットのプリセットの並びへ移し替える**。
 *
 * 壊れていれば `RecordingFileError` を投げる。名前が 1 つも一致しなければ「別のアセットで録ったもの」
 * として落とす — 全部 0 の収録を黙って再生すると、無表情なのが「そういう収録」なのか「読み違え」なのか
 * 区別が付かない。
 */
export function parseRecording(
  text: string,
  presetNames: readonly string[],
  maximumSeconds = MAX_RECORDING_SECONDS,
): LoadedRecording {
  let raw: unknown;
  try {
    raw = JSON.parse(text) as unknown;
  } catch (error) {
    throw new RecordingFileError(`JSON として読めません: ${String(error)}`);
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    throw new RecordingFileError('収録データではありません（オブジェクトではない）。');
  }
  const record = raw as Record<string, unknown>;
  if (record.formatVersion !== RECORDING_FORMAT_VERSION) {
    throw new RecordingFileError(
      `形式のバージョンが違います（${String(record.formatVersion)}、` +
        `読めるのは ${RECORDING_FORMAT_VERSION}）。`,
    );
  }
  const sourceNames = record.presetNames;
  if (!Array.isArray(sourceNames) || sourceNames.some((name) => typeof name !== 'string')) {
    throw new RecordingFileError('presetNames がプリセット名の配列ではありません。');
  }
  const rawFrames = record.frames;
  if (!Array.isArray(rawFrames)) {
    throw new RecordingFileError('frames が配列ではありません。');
  }

  // 収録時の index → 今の index。無い名前は -1（重みを捨てる）。
  const names = sourceNames as string[];
  const mapping = names.map((name) => presetNames.indexOf(name));
  const droppedPresets = names.filter((_, index) => mapping[index] < 0);
  if (droppedPresets.length === names.length) {
    throw new RecordingFileError(
      '今のアセットと一致するプリセット名が 1 つもありません（別のアセットで録ったもの）。',
    );
  }

  const frames: RecordedFrame[] = [];
  let droppedFrames = 0;
  let truncated = false;
  for (const item of rawFrames) {
    if (typeof item !== 'object' || item === null) {
      throw new RecordingFileError('frames の要素がオブジェクトではありません。');
    }
    const frame = item as Record<string, unknown>;
    const timeSeconds = frame.timeSeconds;
    const sourceWeights = frame.weights;
    if (typeof timeSeconds !== 'number' || !Number.isFinite(timeSeconds) || timeSeconds < 0) {
      throw new RecordingFileError('frames の timeSeconds が 0 以上の数ではありません。');
    }
    if (!Array.isArray(sourceWeights) || sourceWeights.length !== names.length) {
      throw new RecordingFileError(
        `frames の weights が presetNames と同じ長さ（${names.length}）ではありません。`,
      );
    }
    if (timeSeconds > maximumSeconds) {
      truncated = true;
      continue;
    }
    const previous = frames.length === 0 ? null : frames[frames.length - 1];
    if (previous !== null && timeSeconds <= previous.timeSeconds) {
      droppedFrames++;
      continue;
    }
    const weights = new Array<number>(presetNames.length).fill(0);
    for (let index = 0; index < names.length; index++) {
      const target = mapping[index];
      if (target < 0) continue;
      const value = sourceWeights[index];
      if (typeof value !== 'number' || !Number.isFinite(value)) {
        throw new RecordingFileError('frames の weights に数でない値があります。');
      }
      weights[target] = value;
    }
    const blink = frame.blink;
    frames.push({
      timeSeconds,
      weights,
      blink: typeof blink === 'number' ? clamp01(blink) : 0,
    });
  }
  if (frames.length === 0) {
    throw new RecordingFileError('再生できるフレームがありません。');
  }
  return {
    recording: {
      formatVersion: RECORDING_FORMAT_VERSION,
      presetNames: [...presetNames],
      frames,
    },
    droppedPresets,
    droppedFrames,
    truncated,
  };
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}
