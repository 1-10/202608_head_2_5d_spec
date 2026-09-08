// 表情プリセットとまばたき。3D ビューだけが使う。
//
// **表情はすべて「383 成分の係数」で表す。** GNM の表情基底は identity 基底と同じ「係数 × 基底」の
// 形なので、プリセットも口形もまばたきもトラッキングも、行き先は 1 本の係数ベクトルになる。
// 変位に潰した形を持たない — 潰すと GNM 本来の形が web の中だけで消え、Unity へ持って行けない。
//
// - プリセット 20 本: 正本は Unity 側の `Tools/export_expression_presets.py`（公式 CVAE デコーダを
//   latent 0 = クラス条件付き平均で回したもの）。アセットには**係数の行**として入っている
// - 口形 5 本: 正本は web 側の `tools/viseme_presets.json`（上の 20 本の係数行の線形結合）。
//   表情と口形の切り分けは `viseme.ts`。**このファイルは本数を持たない**
// - まばたき: `wink_left` + `wink_right` の係数を目の成分だけ残したもの
// - トラッキング: MediaPipe の blendshape スコアからプリセットの重みを作り、係数へ畳む
//   （`faceTracking.ts`。**点から係数を直接解く形も試したが実機で負けた** — 経緯は git log）
//
// 自動再生のしかたは Unity 側 `Viewer/GnmExpressionPlayer`。**同時に立てるのは 1 本だけ** —
// 係数は足せるが、確認用途では「今どのプリセットか」が分かる方が役に立つ。
//
// ## まばたきは加算ではなく「目の成分だけ置き換え」
//
// 正本は旧 web 版（`blink.ts` の波形と `gnmHeadMesh` のクロスフェード）。**加算にすると surprise の
// ような開瞼系と打ち消し合い、まばたき中も瞼が閉じ切らずに眼球が瞼を貫いて見える。**
//
// 置き換えは**係数の側**で行う（旧実装は頂点の側だった）。目領域の成分は連続した 1 区間なので
// 区間の係数を寄せるだけで済み、結果が 1 本の係数ベクトルに収まる — だから録画にもそのまま乗り、
// Unity 側にまばたき専用の経路が要らない。頂点の側で混ぜると、目と口が共有する頂点で口の寄与まで
// 薄まってしまう（領域の支持は重なっている）。

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
    // 波形は `sin(pi t)`（旧 web 版 `updateBlink` と同じ）。閉じ切りで留めない。
    const progress = 1 - remaining / state.durationSeconds;
    return {
      state: { ...state, remainingSeconds: remaining },
      weight: Math.sin(Math.PI * progress),
    };
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

/** 表情の係数ベクトル（長さ `componentCount`）を作る。 */
export function zeroCoefficients(preview: GnmPreviewAsset): Float64Array {
  return new Float64Array(preview.componentCount);
}

/**
 * プリセットごとの重みを係数ベクトルへ足し込む（`coefficients` を破壊的に更新）。
 *
 * @param weights プリセットごとの重み。長さは `preview.presetCount`
 */
export function addPresetCoefficients(
  preview: GnmPreviewAsset,
  coefficients: Float64Array,
  weights: Float64Array,
): void {
  if (weights.length !== preview.presetCount) {
    throw new Error(`表情の重みが ${weights.length} 個（期待 ${preview.presetCount}）`);
  }
  if (coefficients.length !== preview.componentCount) {
    throw new Error(`係数が ${coefficients.length} 個（期待 ${preview.componentCount}）`);
  }
  const count = preview.componentCount;
  for (let preset = 0; preset < preview.presetCount; preset++) {
    const weight = weights[preset];
    if (weight === 0) continue;
    const base = preset * count;
    for (let component = 0; component < count; component++) {
      coefficients[component] += preview.expressionPresetCoefficients[base + component] * weight;
    }
  }
}

/**
 * まばたきを係数へ混ぜる（`coefficients` を破壊的に更新）。
 *
 * 目の成分だけ「表情由来の係数」と「まばたきの係数」をクロスフェードする。**加算しない** —
 * 開瞼系の表情と打ち消し合って瞼が閉じ切らなくなる。
 *
 * @param amount 0（開眼）〜1（閉眼）
 */
export function blendBlinkCoefficients(
  preview: GnmPreviewAsset,
  coefficients: Float64Array,
  amount: number,
): void {
  if (amount <= 0) return;
  const blend = Math.min(1, amount);
  const end = preview.blinkComponentOffset + preview.blinkComponentCount;
  for (let component = preview.blinkComponentOffset; component < end; component++) {
    const blink = preview.blinkCoefficients[component];
    coefficients[component] = coefficients[component] * (1 - blend) + blink * blend;
  }
}

/**
 * 係数ベクトルを頂点へ加算する（`vertices` を破壊的に更新）。
 *
 * **毎フレーム走る所なので、ブロックの並び（頂点, 成分, xyz）に沿って頂点優先で回す。** 内側の
 * 読みが連番になり、頂点への書き戻しが 1 頂点 1 回で済む。成分優先だと同じ演算数でも 2 倍以上遅い。
 *
 * 係数に 0 を掛ける演算は残す。**分岐で飛ばすと遅くなる** — 内側は 1 頂点あたり成分数ぶん回るので、
 * 分岐予測の外れる方が乗算より高い。呼び側で係数を間引くのは自由（`factors` が 0 なら無害）。
 */
export function addExpression(
  preview: GnmPreviewAsset,
  vertices: Float64Array,
  coefficients: Float64Array,
  scratch = new Float64Array(preview.componentCount),
): void {
  if (coefficients.length !== preview.componentCount) {
    throw new Error(`係数が ${coefficients.length} 個（期待 ${preview.componentCount}）`);
  }
  const quantized = preview.expressionBasisQ;
  const blockVertices = preview.expressionBasisVertices;
  // 係数 × 量子化スケールを先に畳む（内側で毎回掛けない）。
  for (let component = 0; component < coefficients.length; component++) {
    scratch[component] = (coefficients[component] * preview.expressionBasisScales[component]) / 32767;
  }
  for (const region of preview.expressionBasisRegions) {
    const { componentOffset, componentCount, vertexOffset, vertexCount } = region;
    let from = region.quantizedOffset;
    for (let slot = 0; slot < vertexCount; slot++) {
      let x = 0;
      let y = 0;
      let z = 0;
      for (let local = 0; local < componentCount; local++) {
        const factor = scratch[componentOffset + local];
        x += quantized[from] * factor;
        y += quantized[from + 1] * factor;
        z += quantized[from + 2] * factor;
        from += 3;
      }
      const to = blockVertices[vertexOffset + slot] * 3;
      vertices[to] += x;
      vertices[to + 1] += y;
      vertices[to + 2] += z;
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
