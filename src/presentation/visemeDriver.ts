// 口形（あいうえお）の連続再生。`Viewer.expressionOverride` へ差す駆動源。
//
// **手動の 5 本はここを通らない。** 口形はアセットへ焼いた 1 本のプリセットなので、手で立てるのは
// 既存の `Viewer.setManualExpression` で足りる（表情のスライダーと同じ経路）。ここが要るのは
// 「時間で切り替える」ぶんだけ。
//
// **ビューアーの中へ入れない。** `Viewer` は「表情の重みを外から差し込む口」を 1 つ持つだけにして
// あり、そこへ何を差すかはビューアーの関心ではない（口形・トラッキング・録画の再生と、駆動源は
// これから増える）。ここをビューアーへ書くと、増えるたびにビューアーが状態を抱える。
//
// **同時に立てるのは 1 本だけ**（`domain/preview/viseme` の原則）。連続再生はそれを機械で守る —
// 毎フレーム 0 で来る `weights` に、今の 1 本ぶんしか書かない。

import { GnmPreviewAsset } from '../domain/preview/asset';
import {
  IDLE_VISEME_PLAYBACK,
  VisemePlayback,
  advanceVisemePlayback,
  splitPresetIndices,
  visemeLabel,
} from '../domain/preview/viseme';
import { ViewSettings } from './viewSettings';

/** `Viewer.expressionOverride` へ差せる形。 */
export type ExpressionFrame = (weights: Float64Array, deltaSeconds: number) => string | null;

/**
 * 口形の連続再生。
 *
 * 状態は「どこまで進んだか」だけ。速さとループはビューの設定が正本で、`apply` のたびに丸ごと
 * 受け取る（片方だけ更新する経路を作らない）。
 */
export class VisemeDriver {
  /** アセットの中の口形プリセットの index（並びは焼いた順 = あいうえお）。 */
  private presets: readonly number[] = [];
  private names: readonly string[] = [];
  private playback: VisemePlayback = IDLE_VISEME_PLAYBACK;
  private running = false;
  private loop = true;
  private fadeSeconds = 0;
  private holdSeconds = 0;

  /** アセットを読んだ（か差し替えた）ときに呼ぶ。 */
  setPreview(preview: GnmPreviewAsset): void {
    this.presets = splitPresetIndices(preview).visemes;
    this.names = this.presets.map((preset) => preview.expressionPresetNames[preset]);
    this.rewind();
  }

  /** 連続再生を あ の手前へ戻す（「先頭から再生」）。 */
  rewind(): void {
    this.playback = IDLE_VISEME_PLAYBACK;
  }

  /**
   * ビューの設定を取り込む。
   *
   * 駆動を切り替えたときは頭から始める。終端まで行って止まった状態のまま「連続再生」へ戻すと、
   * 何も起きずに壊れているように見える。
   */
  apply(view: ViewSettings): void {
    const running = view.visemeMode === 'sequence';
    if (running !== this.running) this.rewind();
    this.running = running;
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
   * 1 フレームぶんの重みを埋め、読み出しに出す名前を返す。
   *
   * 名前は `口形 あ` のように**自分が何なのかまで名乗る**。読み出し側で「表情」と決め打ちすると、
   * 別の駆動源が同じ口へ差さったときに黙って嘘のラベルになる。
   */
  private readonly step: ExpressionFrame = (weights, deltaSeconds) => {
    const step = advanceVisemePlayback(
      this.playback,
      this.presets.length,
      deltaSeconds,
      this.loop,
      this.fadeSeconds,
      this.holdSeconds,
    );
    this.playback = step.playback;
    if (step.index < 0) return null;
    weights[this.presets[step.index]] = step.weight;
    return `口形 ${visemeLabel(this.names[step.index])}`;
  };
}
