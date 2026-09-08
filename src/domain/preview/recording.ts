// 表情アニメーションの収録と再生（純粋計算）。
//
// 収録するのは**表情基底 383 成分の係数**だけで、カメラ映像も landmark も残さない。3D ビューを
// 駆動するのに要るのはそれだけであり、顔の画像を保存しないで済む方が扱いが軽い。
//
// ## なぜ係数で持つのか
//
// 一時「プリセット名 → 重み」で持っていた。あれは web 側の 25 本（うち口形 5 本は web 固有）に
// 依存する形で、**Unity では再生できない**。係数なら公式の基底そのものへの指示なので、同じ基底を
// 持つ側はどこでも同じ顔を出せる。まばたきも係数へ畳んであるので、専用の経路が要らない。
//
// 版は上げていない。**まだ誰にも配っていないので、外に古い形のファイルが無い。**
//
// ## 係数は成分名で持つ
//
// 保存するのは「係数の並び」ではなく「成分名の並び + その順の係数」。公式アセットの版が上がって
// 成分が増減すると、index で持った係数は**黙って別の表情になる**。名前で突き合わせて、**今の
// アセットに無い名前は落とす** — 黙って別の顔になるより、その成分が出ない方がよい。
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

/** 時刻の刻み（秒）。 */
export const TIME_QUANTUM = 0.001;

/**
 * 係数を丸める基準にする変位（メートル）。
 *
 * 383 成分 × 60fps × 30 秒ぶんの数が並ぶので、桁を落とさないと JSON が無駄に太る。**成分ごとに
 * 刻みを変える** — 係数 1 あたりの変位は成分によって 100 倍近く違うので（実測 0.13mm 〜 11.5mm）、
 * 一律の刻みだと小さい成分に無駄な桁を使い、大きい成分は粗くなる。
 *
 * 1um は頭部の寸法 0.3m に対して 3ppm で、画面では見えない。**積む時点で丸める** — 保存してから
 * 読み直すと動きが変わる、という差を作らないため。
 */
export const DISPLACEMENT_QUANTUM_METERS = 1e-6;

/** 成分ごとの刻みを、成分ごとの「係数 1 あたりの最大変位」から作る。 */
export function coefficientQuanta(
  scalesMeters: Float64Array | readonly number[],
): Float64Array {
  const quanta = new Float64Array(scalesMeters.length);
  for (let index = 0; index < quanta.length; index++) {
    const scale = scalesMeters[index];
    quanta[index] = scale > 0 ? DISPLACEMENT_QUANTUM_METERS / scale : DISPLACEMENT_QUANTUM_METERS;
  }
  return quanta;
}

/** 1 フレーム。 */
export interface RecordedFrame {
  /** 収録開始からの秒。**厳密に増加する**（同じ時刻・逆行は積む側で落とす）。 */
  readonly timeSeconds: number;
  /** 成分ごとの係数。並びは `ExpressionRecording.componentNames`。 */
  readonly coefficients: readonly number[];
}

/** 収録したもの。**そのまま JSON にできる形**にしておく（保存の段で詰め替えない）。 */
export interface ExpressionRecording {
  readonly formatVersion: number;
  /** 収録時の成分名の並び（係数の意味の正本）。 */
  readonly componentNames: readonly string[];
  readonly frames: readonly RecordedFrame[];
}

/** 空の収録を作る。 */
export function startRecording(componentNames: readonly string[]): ExpressionRecording {
  return {
    formatVersion: RECORDING_FORMAT_VERSION,
    componentNames: [...componentNames],
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
  coefficients: Float64Array | readonly number[],
  quanta: Float64Array | readonly number[],
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
  const quantized = round(timeSeconds, TIME_QUANTUM);
  if (last !== null && quantized <= last.timeSeconds) return { recording, full: false };
  if (coefficients.length !== recording.componentNames.length) {
    throw new Error(`係数が ${coefficients.length} 個（期待 ${recording.componentNames.length}）`);
  }
  if (quanta.length !== coefficients.length) {
    throw new Error(`刻みが ${quanta.length} 個（期待 ${coefficients.length}）`);
  }
  const rounded: number[] = [];
  for (let index = 0; index < coefficients.length; index++) {
    rounded.push(round(coefficients[index], quanta[index]));
  }
  return {
    recording: {
      ...recording,
      frames: [...frames, { timeSeconds: quantized, coefficients: rounded }],
    },
    full: false,
  };
}

function round(value: number, quantum: number): number {
  if (!Number.isFinite(value) || !(quantum > 0)) return 0;
  return Math.round(value / quantum) * quantum;
}

/**
 * 任意の時刻の係数を取り出す（`coefficients` を破壊的に埋める）。
 *
 * フレーム間は線形補間する。**等間隔を仮定しない**（前後のフレームの時刻から比を作る）。範囲外は
 * 端のフレームで留める。
 *
 * @param coefficients 長さは `recording.componentNames.length`
 */
export function sampleRecording(
  recording: ExpressionRecording,
  timeSeconds: number,
  coefficients: Float64Array,
): void {
  const frames = recording.frames;
  if (coefficients.length !== recording.componentNames.length) {
    throw new Error(`係数が ${coefficients.length} 個（期待 ${recording.componentNames.length}）`);
  }
  coefficients.fill(0);
  if (frames.length === 0) return;
  if (timeSeconds <= frames[0].timeSeconds) {
    copyFrame(frames[0], coefficients);
    return;
  }
  const lastFrame = frames[frames.length - 1];
  if (timeSeconds >= lastFrame.timeSeconds) {
    copyFrame(lastFrame, coefficients);
    return;
  }

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
  for (let index = 0; index < coefficients.length; index++) {
    coefficients[index] =
      before.coefficients[index] + (after.coefficients[index] - before.coefficients[index]) * t;
  }
}

function copyFrame(frame: RecordedFrame, coefficients: Float64Array): void {
  for (let index = 0; index < coefficients.length; index++) {
    coefficients[index] = frame.coefficients[index];
  }
}

/** 保存する形（JSON 文字列）。 */
export function serializeRecording(recording: ExpressionRecording): string {
  return JSON.stringify(recording);
}

/** 読み込みの結果。**落としたものを黙らせない。** */
export interface LoadedRecording {
  /** 今のアセットの並びへ移し替えた収録。 */
  readonly recording: ExpressionRecording;
  /** 今のアセットに無くて落とした成分名。 */
  readonly droppedComponents: readonly string[];
  /** 時刻が逆行・重複していて落としたフレーム数。 */
  readonly droppedFrames: number;
  /** 上限を超えて切り捨てたか。 */
  readonly truncated: boolean;
}

/**
 * 保存したものを読み、**今のアセットの成分の並びへ移し替える**。
 *
 * 壊れていれば `RecordingFileError` を投げる。名前が 1 つも一致しなければ「別のアセットで録ったもの」
 * として落とす — 全部 0 の収録を黙って再生すると、無表情なのが「そういう収録」なのか「読み違え」なのか
 * 区別が付かない。
 */
export function parseRecording(
  text: string,
  componentNames: readonly string[],
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
  const sourceNames = record.componentNames;
  if (!Array.isArray(sourceNames) || sourceNames.some((name) => typeof name !== 'string')) {
    throw new RecordingFileError('componentNames が成分名の配列ではありません。');
  }
  const rawFrames = record.frames;
  if (!Array.isArray(rawFrames)) {
    throw new RecordingFileError('frames が配列ではありません。');
  }

  // 収録時の index → 今の index。無い名前は -1（係数を捨てる）。
  const names = sourceNames as string[];
  const mapping = names.map((name) => componentNames.indexOf(name));
  const droppedComponents = names.filter((_, index) => mapping[index] < 0);
  if (droppedComponents.length === names.length) {
    throw new RecordingFileError(
      '今のアセットと一致する成分名が 1 つもありません（別のアセットで録ったもの）。',
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
    const source = frame.coefficients;
    if (typeof timeSeconds !== 'number' || !Number.isFinite(timeSeconds) || timeSeconds < 0) {
      throw new RecordingFileError('frames の timeSeconds が 0 以上の数ではありません。');
    }
    if (!Array.isArray(source) || source.length !== names.length) {
      throw new RecordingFileError(
        `frames の coefficients が componentNames と同じ長さ（${names.length}）ではありません。`,
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
    const coefficients = new Array<number>(componentNames.length).fill(0);
    for (let index = 0; index < names.length; index++) {
      const target = mapping[index];
      if (target < 0) continue;
      const value = source[index];
      if (typeof value !== 'number' || !Number.isFinite(value)) {
        throw new RecordingFileError('frames の coefficients に数でない値があります。');
      }
      coefficients[target] = value;
    }
    frames.push({ timeSeconds, coefficients });
  }
  if (frames.length === 0) {
    throw new RecordingFileError('再生できるフレームがありません。');
  }
  return {
    recording: {
      formatVersion: RECORDING_FORMAT_VERSION,
      componentNames: [...componentNames],
      frames,
    },
    droppedComponents,
    droppedFrames,
    truncated,
  };
}
