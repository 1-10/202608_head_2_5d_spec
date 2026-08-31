// 表情プリセットとまばたき。3D ビューだけが使う。
//
// **プリセットの正本は Unity 側の `Tools/export_expression_presets.py`**（公式 CVAE デコーダを latent 0
// = クラス条件付き平均で回して 20 本に焼いたもの）。自動再生のしかたは `Viewer/GnmExpressionPlayer`。
//
// **公式 383 成分をそのまま出さない。** 成分名は領域ごとの統計方向で、表情としての意味を持たない。
// 旧 web 版は 383 成分の一部を領域で切って合成していたが、その領域分割は公式に無い操作だった。
// 焼いた 20 本だけを持つ方が Unity と同じ絵になる。
//
// **加算変位なので同時に立てるのは 1 本だけ。** 重ねると顔が壊れるうえ、確認用途では「今どれか」が
// 分かる方が役に立つ。まばたきだけは例外で、独立した層として上に足す（旧 web 版から残した機能）。

import { GnmPreviewAsset } from './asset';

/** 自動再生のしかた。 */
export type ExpressionPlayMode = 'off' | 'sequence' | 'random';

/** 立ち上がり / 抜けにかける秒数。正本は Unity 側 `_fadeSeconds`。 */
export const FADE_SECONDS = 0.35;

/** 最大の重みで留める秒数。同 `_holdSeconds`。 */
export const HOLD_SECONDS = 0.8;

/** まばたきの周期（秒）。旧 web 版 `blinkPeriodMinSec` / `blinkPeriodMaxSec`。 */
export const BLINK_PERIOD_MIN_SECONDS = 3;
export const BLINK_PERIOD_MAX_SECONDS = 5;

/** まばたき 1 回の長さ（ミリ秒）。旧 web 版 `blinkDurationMinMs` / `blinkDurationMaxMs`。 */
export const BLINK_DURATION_MIN_MS = 150;
export const BLINK_DURATION_MAX_MS = 250;

/** 両目を閉じるために立てるプリセット。片目ずつのウインクを両方立てる。 */
export const BLINK_PRESET_NAMES: readonly string[] = ['wink_left', 'wink_right'];

/** 台形エンベロープ。0 → 1 → 1 → 0 で、両端は smoothstep で丸める。 */
export function envelope(elapsedSeconds: number, fadeSeconds = FADE_SECONDS, holdSeconds = HOLD_SECONDS): number {
  const cycle = fadeSeconds * 2 + holdSeconds;
  if (fadeSeconds <= 0) return elapsedSeconds < cycle ? 1 : 0;
  if (elapsedSeconds < fadeSeconds) return smoothStep(elapsedSeconds / fadeSeconds);
  const fadeOutStart = cycle - fadeSeconds;
  if (elapsedSeconds < fadeOutStart) return 1;
  return smoothStep(1 - (elapsedSeconds - fadeOutStart) / fadeSeconds);
}

function smoothStep(t: number): number {
  const clamped = Math.min(1, Math.max(0, t));
  return clamped * clamped * (3 - 2 * clamped);
}

/** 自動再生の状態。`advance` が新しい状態を返す（保持は呼び側）。 */
export interface ExpressionPlayback {
  /** 今かかっているプリセットの index。何もかかっていなければ -1。 */
  readonly index: number;
  readonly elapsedSeconds: number;
}

export const IDLE_PLAYBACK: ExpressionPlayback = { index: -1, elapsedSeconds: 0 };

/**
 * 自動再生を 1 フレーム進める。
 *
 * `random` は直前と同じものを引かない（同じものが 2 回続くと止まって見える）。
 *
 * @param pick 0〜1 の乱数を返す関数（テストから差し替えられる形にしてある）
 */
export function advancePlayback(
  playback: ExpressionPlayback,
  mode: ExpressionPlayMode,
  presetCount: number,
  deltaSeconds: number,
  pick: () => number = Math.random,
  fadeSeconds = FADE_SECONDS,
  holdSeconds = HOLD_SECONDS,
): { playback: ExpressionPlayback; index: number; weight: number } {
  if (mode === 'off' || presetCount === 0) {
    return { playback: IDLE_PLAYBACK, index: -1, weight: 0 };
  }
  const cycle = fadeSeconds * 2 + holdSeconds;
  let { index, elapsedSeconds } = playback;
  if (index < 0 || elapsedSeconds >= cycle) {
    index = nextIndex(index, mode, presetCount, pick);
    elapsedSeconds = 0;
  }
  const weight = envelope(elapsedSeconds, fadeSeconds, holdSeconds);
  return {
    playback: { index, elapsedSeconds: elapsedSeconds + deltaSeconds },
    index,
    weight,
  };
}

function nextIndex(
  current: number,
  mode: ExpressionPlayMode,
  presetCount: number,
  pick: () => number,
): number {
  if (mode === 'sequence') return (current + 1) % presetCount;
  if (presetCount === 1) return 0;
  const picked = Math.min(presetCount - 2, Math.floor(pick() * (presetCount - 1)));
  return picked >= current ? picked + 1 : picked;
}

/** まばたきの状態。 */
export interface BlinkState {
  /** 次のまばたきが始まるまでの残り秒。 */
  readonly waitSeconds: number;
  /** 今のまばたきの残り秒。閉じていなければ 0。 */
  readonly remainingSeconds: number;
  /** 今のまばたき 1 回の長さ（秒）。 */
  readonly durationSeconds: number;
}

/** まばたきの初期状態（最初の 1 回まで待つ）。 */
export function startBlink(pick: () => number = Math.random): BlinkState {
  return {
    waitSeconds: lerp(BLINK_PERIOD_MIN_SECONDS, BLINK_PERIOD_MAX_SECONDS, pick()),
    remainingSeconds: 0,
    durationSeconds: 0,
  };
}

/**
 * まばたきを 1 フレーム進める。
 *
 * @returns 閉眼の重み（0〜1）と次の状態
 */
export function advanceBlink(
  state: BlinkState,
  deltaSeconds: number,
  pick: () => number = Math.random,
): { state: BlinkState; weight: number } {
  if (state.remainingSeconds > 0) {
    const remaining = state.remainingSeconds - deltaSeconds;
    if (remaining <= 0) {
      return {
        state: {
          waitSeconds: lerp(BLINK_PERIOD_MIN_SECONDS, BLINK_PERIOD_MAX_SECONDS, pick()),
          remainingSeconds: 0,
          durationSeconds: 0,
        },
        weight: 0,
      };
    }
    // 閉じ切りで留めない。半分で最大、両端で 0 の三角波を smoothstep で丸める。
    const progress = 1 - remaining / state.durationSeconds;
    const shape = smoothStep(1 - Math.abs(progress * 2 - 1));
    return { state: { ...state, remainingSeconds: remaining }, weight: shape };
  }
  const wait = state.waitSeconds - deltaSeconds;
  if (wait > 0) return { state: { ...state, waitSeconds: wait }, weight: 0 };
  const duration = lerp(BLINK_DURATION_MIN_MS, BLINK_DURATION_MAX_MS, pick()) / 1000;
  return {
    state: { waitSeconds: 0, remainingSeconds: duration, durationSeconds: duration },
    weight: 0,
  };
}

function lerp(low: number, high: number, t: number): number {
  return low + (high - low) * Math.min(1, Math.max(0, t));
}

/**
 * 表情の重みを頂点へ加算する（`vertices` を破壊的に更新）。
 *
 * @param weights プリセットごとの重み。長さは `preview.presetCount`
 */
export function addExpression(
  preview: GnmPreviewAsset,
  vertices: Float64Array,
  weights: Float64Array,
): void {
  if (weights.length !== preview.presetCount) {
    throw new Error(`表情の重みが ${weights.length} 個（期待 ${preview.presetCount}）`);
  }
  const stride = preview.vertexCount * 3;
  for (let preset = 0; preset < preview.presetCount; preset++) {
    const weight = weights[preset];
    if (weight === 0) continue;
    const factor = (weight * preview.expressionPresetScales[preset]) / 32767;
    const base = preset * stride;
    for (let index = 0; index < stride; index++) {
      vertices[index] += preview.expressionPresetBasisQ[base + index] * factor;
    }
  }
}

/** プリセット名から重み配列を作る（無い名前は無視せず落とす）。 */
export function weightsFor(
  preview: GnmPreviewAsset,
  entries: readonly (readonly [string, number])[],
): Float64Array {
  const weights = new Float64Array(preview.presetCount);
  for (const [name, weight] of entries) {
    const index = preview.expressionPresetNames.indexOf(name);
    if (index < 0) {
      throw new Error(
        `表情プリセット '${name}' がアセットに無い` +
          `（あるのは: ${preview.expressionPresetNames.join(', ')}）`,
      );
    }
    weights[index] += weight;
  }
  return weights;
}
