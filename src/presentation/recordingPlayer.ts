// 収録した表情アニメーションの置き場と再生。`Viewer.expressionOverride` へ差す駆動源。
//
// **カメラの箱に置かない。** 再生と読み込みは `<video>` を一切見ない — カメラが止まっていても、
// そもそも繋がっていなくても動く。Webカメラのパネルへ置くと「カメラの機能」に見えるうえ、パネルを
// 閉じたら再生が止まる（実体と画面の形がねじれる）。**録るのはカメラ、持つのと再生するのはここ。**
//
// **クリップの持ち主はここひとつ。** 録画中に積むのも、読み込んだものも、保存するのも同じ 1 本を
// 指す。録る側が自分の複製を持つと、読み込んだ後に「保存」を押したときどちらが出るかが分からない。
//
// 作りは `presentation/visemeDriver` に揃えてある（状態を持ち、`main` が駆動を差す）。**再生中か
// どうかの正本はここ** — 終端まで行ったら自分から止まるので、押した側が持つと止まった後も
// 「停止」のまま残る。止まったことは `onFinished` で知らせる。

import {
  ExpressionRecording,
  appendFrame,
  recordingDurationSeconds,
  sampleRecording,
  startRecording,
} from '../domain/preview/recording';

/** 録る側が使う口。**積むだけ** — クリップを持たせない。 */
export interface RecordingSink {
  /** 新しい収録を始める（前のクリップは捨てる）。 */
  begin(presetNames: readonly string[]): void;
  /**
   * 1 フレーム積む。
   *
   * @returns 上限（`MAX_RECORDING_SECONDS`）に達したか。達したら録る側が録画を止める
   */
  append(timeSeconds: number, weights: Float64Array, blink: number): boolean;
  /** 今のクリップの長さ（秒）。無ければ 0。 */
  durationSeconds(): number;
  frameCount(): number;
  /** 保存に使う。無ければ `null`。 */
  current(): ExpressionRecording | null;
}

export class RecordingPlayer implements RecordingSink {
  private recording: ExpressionRecording | null = null;
  private playSeconds = 0;
  private playing = false;
  private blinkAmount = 0;

  /** ループ再生するか。画面の設定が正本で、`apply` で受け取る。 */
  private loop = false;

  /** 終端まで行って自分から止まったときに呼ばれる（ボタンを「再生」へ戻すため）。 */
  onFinished: (() => void) | null = null;

  /** クリップが差し替わった（録り直し・読み込み）ときに呼ばれる。表示を合わせるのに使う。 */
  onRecordingChanged: (() => void) | null = null;

  get isPlaying(): boolean {
    return this.playing;
  }

  get hasRecording(): boolean {
    return this.recording !== null;
  }

  /** ループの設定を取り込む。再生しているかはここでは変えない。 */
  setLoop(loop: boolean): void {
    this.loop = loop;
  }

  /** 読み込んだクリップを入れる（再生中なら止める。中身が変わったら位置も頭へ戻す）。 */
  setRecording(recording: ExpressionRecording): void {
    this.stop();
    this.recording = recording;
    this.playSeconds = 0;
    this.onRecordingChanged?.();
  }

  // ---- 録る側の口（`RecordingSink`）----

  begin(presetNames: readonly string[]): void {
    this.stop();
    this.recording = startRecording(presetNames);
    this.playSeconds = 0;
    this.onRecordingChanged?.();
  }

  append(timeSeconds: number, weights: Float64Array, blink: number): boolean {
    if (this.recording === null) return false;
    const step = appendFrame(this.recording, timeSeconds, weights, blink);
    this.recording = step.recording;
    return step.full;
  }

  durationSeconds(): number {
    return this.recording === null ? 0 : recordingDurationSeconds(this.recording);
  }

  frameCount(): number {
    return this.recording === null ? 0 : this.recording.frames.length;
  }

  current(): ExpressionRecording | null {
    return this.recording;
  }

  // ---- 再生 ----

  /**
   * 頭から再生する。クリップが無ければ何もせず `false` を返す。
   *
   * **必ず頭から始める。** 終端まで行って止まった状態のまま押されることがあり、そこから続けると
   * 何も起きずに壊れているように見える（`VisemeDriver.play` と同じ理由）。
   */
  play(): boolean {
    if (this.recording === null || this.recording.frames.length === 0) return false;
    this.playSeconds = 0;
    this.blinkAmount = 0;
    this.playing = true;
    return true;
  }

  stop(): void {
    this.playing = false;
    this.playSeconds = 0;
    this.blinkAmount = 0;
  }

  /** 今の再生位置（秒）。表示に使う。 */
  get positionSeconds(): number {
    return this.playSeconds;
  }

  /** 1 フレームぶんの重みを埋め、読み出しに出す名前を返す。 */
  expression(weights: Float64Array, deltaSeconds: number): string | null {
    const recording = this.recording;
    if (!this.playing || recording === null) return null;
    // クリップは**プリセット名で**重みを持つ。アセットが差し替わって本数が変わったら鳴らさない
    // （黙って別の表情になるより出ない方がよい）。読み込みの段で名前を突き合わせてある。
    if (weights.length !== recording.presetNames.length) return null;

    const duration = recordingDurationSeconds(recording);
    this.playSeconds += deltaSeconds;
    if (this.playSeconds > duration) {
      if (this.loop && duration > 0) {
        this.playSeconds %= duration;
      } else {
        this.playSeconds = duration;
        this.blinkAmount = sampleRecording(recording, duration, weights);
        // **自分で止まる。** 止まったまま口を占有し続けると、手のスライダーも表情の自動再生も
        // 効かない顔になる。
        this.playing = false;
        this.onFinished?.();
        return null;
      }
    }
    this.blinkAmount = sampleRecording(recording, this.playSeconds, weights);
    return '録画の再生';
  }

  blink(): number {
    return this.blinkAmount;
  }
}
