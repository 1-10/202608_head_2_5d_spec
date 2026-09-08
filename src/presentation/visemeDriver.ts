// 口形（あいうえお）の連続再生。`Viewer.expressionOverride` へ差す駆動源。
//
// **手動の 5 本はここを通らない。** 口形は表情プリセットと同じ「係数の行」1 本なので、手で立てるのは
// 既存の `Viewer.setManualExpression` で足りる（表情のスライダーと同じ経路）。ここが要るのは
// 「時間で切り替える」ぶんだけ。
//
// **ビューアーの中へ入れない。** `Viewer` は「表情の重みを外から差し込む口」を 1 つ持つだけにして
// あり、そこへ何を差すかはビューアーの関心ではない（口形・トラッキング・録画の再生と、駆動源は
// これから増える）。ここをビューアーへ書くと、増えるたびにビューアーが状態を抱える。
//
// **同時に立てるのは 1 本だけ**（`domain/preview/viseme` の原則）。連続再生はそれを機械で守る —
// 毎フレーム 0 で来る係数へ、今の 1 本ぶんの行しか足さない。

import { GnmPreviewAsset } from '../domain/preview/asset';
import { addPresetCoefficients } from '../domain/preview/expression';
import {
  IDLE_VISEME_PLAYBACK,
  VisemePlayback,
  advanceVisemePlayback,
  splitPresetIndices,
  visemeLabel,
} from '../domain/preview/viseme';
import { ViewSettings } from './viewSettings';

/** `Viewer.expressionOverride` へ差せる形（埋めるのは表情基底 383 成分の係数）。 */
export type ExpressionFrame = (
  coefficients: Float64Array,
  deltaSeconds: number,
) => string | null;

/**
 * 口形の連続再生。
 *
 * 速さとループはビューの設定が正本で、`apply` のたびに丸ごと受け取る（片方だけ更新する経路を
 * 作らない）。**「今 再生しているか」の正本はここ** — 再生は再生ボタンで始まるが、ループ無しなら
 * 終端で自分から止まるので、押した側が状態を持つと終端で古くなる。止まったことは `onFinished` で
 * 知らせる。
 */
export class VisemeDriver {
  /** アセットの中の口形プリセットの index（並びはアセットの順 = あいうえお）。 */
  private presets: readonly number[] = [];
  private preview: GnmPreviewAsset | null = null;
  /** プリセットの重み → 係数へ畳むための作業領域。 */
  private weights: Float64Array = new Float64Array(0);
  private names: readonly string[] = [];
  private playback: VisemePlayback = IDLE_VISEME_PLAYBACK;
  private running = false;
  private loop = true;
  private fadeSeconds = 0;
  private holdSeconds = 0;

  /** アセットを読んだ（か差し替えた）ときに呼ぶ。 */
  setPreview(preview: GnmPreviewAsset): void {
    this.preview = preview;
    this.weights = new Float64Array(preview.presetCount);
    this.presets = splitPresetIndices(preview).visemes;
    this.names = this.presets.map((preset) => preview.expressionPresetNames[preset]);
    this.stop();
  }

  /** 今 連続再生しているか。ボタンのラベルの正本。 */
  get isPlaying(): boolean {
    return this.running;
  }

  /** 終端まで行って自分から止まったときに呼ばれる（ボタンを「再生」へ戻すため）。 */
  onFinished: (() => void) | null = null;

  /**
   * あ から再生し直す。
   *
   * **必ず頭から始める。** 終端まで行って止まった状態のまま押されることがあり、そこから続けると
   * 何も起きずに壊れているように見える。
   */
  play(): void {
    this.playback = IDLE_VISEME_PLAYBACK;
    this.running = true;
  }

  /** 止める。手のスライダーと表情の自動再生へ顔を返す。 */
  stop(): void {
    this.running = false;
    this.playback = IDLE_VISEME_PLAYBACK;
  }

  /** ビューの設定（速さとループ）を取り込む。再生しているかはここでは変えない。 */
  apply(view: ViewSettings): void {
    this.loop = view.visemeLoop;
    this.fadeSeconds = view.visemeFadeSeconds;
    this.holdSeconds = view.visemeHoldSeconds;
  }

  /**
   * `Viewer.expressionOverride` へ差すもの。**連続再生していなければ `null`**（そのとき顔を駆動する
   * のは手のスライダーと表情の自動再生）。
   */
  get frame(): ExpressionFrame | null {
    return this.running && this.presets.length > 0 ? this.step : null;
  }

  /**
   * 1 フレームぶんの係数を埋め、読み出しに出す名前を返す。
   *
   * 名前は `口形 あ` のように**自分が何なのかまで名乗る**。読み出し側で「表情」と決め打ちすると、
   * 別の駆動源が同じ口へ差さったときに黙って嘘のラベルになる。
   */
  private readonly step: ExpressionFrame = (coefficients, deltaSeconds) => {
    const step = advanceVisemePlayback(
      this.playback,
      this.presets.length,
      deltaSeconds,
      this.loop,
      this.fadeSeconds,
      this.holdSeconds,
    );
    this.playback = step.playback;
    if (step.index < 0) {
      // ループ無しで終端まで行った。**自分で止まる** — 止まったまま口を占有し続けると、手の
      // スライダーも表情の自動再生も効かない顔になる。
      if (this.running) {
        this.running = false;
        this.onFinished?.();
      }
      return null;
    }
    if (this.preview === null) return null;
    this.weights.fill(0);
    this.weights[this.presets[step.index]] = step.weight;
    addPresetCoefficients(this.preview, coefficients, this.weights);
    return `口形 ${visemeLabel(this.names[step.index])}`;
  };
}
