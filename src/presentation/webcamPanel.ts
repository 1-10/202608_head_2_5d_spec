// Webカメラのパネル（映像・写真の取り込み・表情トラッキング・録画・保存）。
//
// **再生と読み込みはここに置かない。** どちらも `<video>` を一切見ない（カメラが繋がっていなくても
// 動く）ので、カメラの箱に置くと「カメラの機能」に見える。置き場は 3D ビューのパネルで、クリップの
// 持ち主は `presentation/recordingPlayer`。ここは**録って積む側**。
//
// **右パネル（lil-gui）へ足さない。** あちらは「書き出しの値」と「見え方の値」を並べる一覧で、
// カメラは値ではなく**状態を持つ装置**（開始 / 停止・トラッキング中・録画中）。同じ列に混ぜると、スライダー
// の間に「今なにが動いているか」が埋もれる。独立した箱にして、状態を箱の頭に出す。
//
// **判断はここが持つ。** どの状態でどのボタンが押せるか、どの駆動源がビューを占有しているかは
// この画面の関心。`main.ts` は配線だけを持つ。
//
// ## 駆動源はひとつだけ
//
// 表情を動かすものは「手のスライダー / 自動再生」「トラッキング」「録画の再生」の 3 つあるが、
// **同時に動かない**。ビューアーの `expressionOverride` / `blinkOverride` を占有する形にして、
// 占有していない間はビューアーが元から持つ経路で動く（ビューアーの中に「今どのモードか」という
// 状態を増やさない）。
//
// ## 映像はミラー、写真は非反転
//
// 表示だけを CSS で左右反転する（`style.css` の `.webcam-video`）。`captureWebcamFrame` は
// `<video>` の画素をそのまま読むので、**取り込んだ写真は非反転のまま** — 既存の約束
// （`input.ts` の「正面撮影のプレビューはミラー表示だが、写真は非反転の実画像として扱う」）を
// 壊さない。

import { FaceExpressionTracker } from '../application/ports';
import { describeFailure } from '../application/exportGuest';
import {
  BLINK_TIME_CONSTANT_SECONDS,
  EXPRESSION_TIME_CONSTANT_SECONDS,
  TrackingPlan,
  blendshapesToTargets,
  missingCategories,
  resolveTrackingPlan,
  smoothScalar,
  smoothToward,
  smoothingFactor,
  strongestPreset,
} from '../domain/preview/faceTracking';
import { MAX_RECORDING_SECONDS, serializeRecording } from '../domain/preview/recording';
import { RecordingSink } from './recordingPlayer';
import { PhotoRgb } from '../domain/photo';
import { InputManager } from './input';

/** ビューアーの表情とまばたきを占有する駆動源。 */
export interface ExpressionDriver {
  /** 1 フレームぶんの重みを埋める。返り値は読み出しに出す名前。 */
  expression(weights: Float64Array, deltaSeconds: number): string | null;
  /** まばたき量（0〜1）。 */
  blink(): number;
}

/** パネルが外へ求めるもの。**ビューアーそのものは渡さない**（占有する口だけを渡す）。 */
export interface WebcamPanelHost {
  /** 表情プリセット名の並び。3D ビューにシーンが無ければ空。 */
  presetNames(): readonly string[];
  /** 表情とまばたきの駆動を占有する / 返す（`null` で返す）。 */
  setExpressionDriver(driver: ExpressionDriver | null): void;
  /** 取り込んだ写真を書き出しへ渡す。 */
  acceptPhoto(photo: PhotoRgb): Promise<void>;
  /** ツールバーの状態表示。 */
  setStatus(message: string, isError?: boolean): void;
  /**
   * 収録の置き場。**録る側は積むだけで、クリップは持たない。**
   *
   * 再生と読み込みは別の画面（3D ビューのパネル）にある — カメラを使わない操作なので。持ち主を
   * 分けると、読み込んだ後に「保存」を押したときどちらのクリップが出るか分からなくなる。
   */
  readonly recording: RecordingSink;
}

/** 今なにが動いているか。**録画の再生はここに無い**（カメラを使わないので別の画面が持つ）。 */
type WebcamMode = 'off' | 'tracking' | 'recording';

const MODE_LABELS: Readonly<Record<WebcamMode, string>> = {
  off: '停止中',
  tracking: 'トラッキング中',
  recording: '録画中',
};

/**
 * 顔を見失ってからプリセットを 0 へ戻すまでの猶予（秒）。
 *
 * 1 フレーム外しただけで 0 へ引っ張ると、検出が飛び飛びのときに顔が痙攣する。逆に猶予が無限だと
 * カメラの前から離れても最後の表情が貼り付く。
 */
const FACE_LOST_SECONDS = 0.5;

export class WebcamPanel {
  private readonly input: InputManager;
  private readonly tracker: FaceExpressionTracker;
  private readonly host: WebcamPanelHost;
  private readonly elements: PanelElements;

  private mode: WebcamMode = 'off';
  private cameraOn = false;
  /** トラッカーの準備（モデルの取得）が走っている間だけ true。 */
  private preparing = false;

  /** アセットのプリセットの並びへ解決した対応表（並びが変わったら作り直す）。 */
  private plan: TrackingPlan | null = null;
  private planNames: readonly string[] | null = null;
  private targetWeights = new Float64Array(0);
  private smoothedWeights = new Float64Array(0);
  private targetBlink = 0;
  private smoothedBlink = 0;
  private faceLostSeconds = 0;
  /** `<video>` の同じフレームを 2 回推論しない。 */
  private lastVideoTime = -1;
  /** 対応の取りこぼしは開発者向けに 1 回だけ出す。 */
  private reportedCategories = false;

  private recordSeconds = 0;
  /** 画面の書き換えを毎フレーム行わないための直近値。 */
  private shownTimer = '';

  private readonly driver: ExpressionDriver = {
    expression: (weights, deltaSeconds) => this.advance(weights, deltaSeconds),
    blink: () => this.smoothedBlink,
  };

  /** パネルの開閉が変わったときに呼ばれる（ツールバーのボタンの同期）。 */
  onOpenChanged: (() => void) | null = null;

  constructor(input: InputManager, tracker: FaceExpressionTracker, host: WebcamPanelHost) {
    this.input = input;
    this.tracker = tracker;
    this.host = host;
    this.elements = collectElements();
    this.attach();
    this.refresh();
  }

  get isOpen(): boolean {
    return !this.elements.panel.hidden;
  }

  /**
   * 開閉を切り替える。
   *
   * **閉じるときはカメラを止める。** パネルを隠すと `<video>` が `display: none` になり、そこへ
   * 新しいフレームが来るかはブラウザ任せ（来なくなると、最後の表情が貼り付いたまま「トラッキング中」
   * の表示だけが残る）。見えていないカメラを回し続ける利点も無い。
   * **再生は止めない** — こちらは `<video>` を見ないので、パネルを畳んで 3D ビューだけを見られる。
   */
  toggle(): void {
    const closing = !this.elements.panel.hidden;
    this.elements.panel.hidden = closing;
    if (closing && this.cameraOn) this.stopCamera();
    this.onOpenChanged?.();
    this.refresh();
  }

  open(): void {
    if (this.elements.panel.hidden) this.toggle();
  }

  /** カメラを止める（写真をファイルから選んだときなど、外から止めたい場合に呼ぶ）。 */
  stopCamera(): void {
    if (this.mode === 'tracking' || this.mode === 'recording') this.releaseDriver('off');
    this.tracker.stop();
    this.input.stopWebcam();
    this.cameraOn = false;
    this.lastVideoTime = -1;
    this.refresh();
  }

  /**
   * 表情の駆動だけを手放す（**カメラは止めない**）。
   *
   * 口形の連続再生など、別の駆動源が顔を取ったときに外から呼ぶ。**駆動源はひとつだけ**なので、
   * 黙って上書きされると「トラッキング中」の表示だけが残って顔が動かない状態になる。カメラを
   * 止めないのは、映像を見ながら口形を確かめたい場合があるから。
   */
  stopDriving(): void {
    if (this.mode === 'off') return;
    this.releaseDriver('off');
    this.refresh();
  }

  /** Reset で呼ぶ。カメラを止めて表示を戻す（**クリップは持っていないので捨てない**）。 */
  reset(): void {
    this.releaseDriver('off');
    this.stopCamera();
    this.recordSeconds = 0;
    this.setNote('');
    this.refresh();
  }

  /** 押せるボタンと表示を今の状態へ合わせる。**3D ビューが変わったら外から呼ぶ。** */
  refresh(): void {
    const elements = this.elements;
    const hasPresets = this.host.presetNames().length > 0;

    elements.state.textContent = MODE_LABELS[this.mode];
    elements.state.dataset.mode = this.mode;

    elements.camera.textContent = this.cameraOn ? 'カメラ停止' : 'カメラ開始';
    elements.camera.disabled = this.preparing;
    // 録画中は取り込ませない。書き出しの間 3D ビューが止まる（シーンを差し替える）ので、
    // 録画にだけ「時計は進まないのにフレームが飛ぶ」穴が空く。
    elements.capture.disabled = !this.cameraOn || this.mode === 'recording';

    elements.track.textContent = this.preparing ? '準備中…' : '表情トラッキング';
    elements.track.disabled = this.preparing || !this.cameraOn || !hasPresets;
    elements.track.setAttribute(
      'aria-pressed',
      String(this.mode === 'tracking' || this.mode === 'recording'),
    );

    elements.record.textContent = this.mode === 'recording' ? '録画停止' : '録画';
    elements.record.disabled = this.mode !== 'tracking' && this.mode !== 'recording';

    // **録画中は保存させない。** 押せると途中までのクリップが落ちてきて、しかも録画は続くので
    // 「保存したもの」と「録り終えたもの」が食い違う。止めてから保存する。
    elements.save.disabled = this.mode === 'recording' || !this.host.recording.current();

    this.updateTimer(true);
  }

  // ---- 配線 ----

  private attach(): void {
    const elements = this.elements;
    elements.close.addEventListener('click', () => this.toggle());
    elements.camera.addEventListener('click', () => void this.toggleCamera());
    elements.capture.addEventListener('click', () => void this.capture());
    elements.track.addEventListener('click', () => void this.toggleTracking());
    elements.record.addEventListener('click', () => this.toggleRecording());
    elements.save.addEventListener('click', () => this.save());
  }

  private async toggleCamera(): Promise<void> {
    if (this.cameraOn) {
      this.stopCamera();
      this.setNote('');
      return;
    }
    try {
      await this.input.startWebcam();
      this.cameraOn = true;
      this.lastVideoTime = -1;
      this.setNote('正面を向いてください。');
    } catch (error) {
      console.error(error);
      this.reportFailure(error);
    }
    this.refresh();
  }

  private async capture(): Promise<void> {
    try {
      const photo = this.input.captureWebcamFrame();
      await this.host.acceptPhoto(photo);
    } catch (error) {
      console.error(error);
      this.reportFailure(error);
    }
    this.refresh();
  }

  private async toggleTracking(): Promise<void> {
    if (this.mode === 'tracking' || this.mode === 'recording') {
      this.releaseDriver('off');
      this.refresh();
      return;
    }
    this.preparing = true;
    this.refresh();
    try {
      await this.tracker.start();
      this.resetTrackingState();
      this.takeDriver('tracking');
      this.setNote('');
    } catch (error) {
      console.error(error);
      this.reportFailure(error);
    } finally {
      this.preparing = false;
      this.refresh();
    }
  }

  private toggleRecording(): void {
    if (this.mode === 'recording') {
      this.mode = 'tracking';
      this.setNote(
        `録画しました（${this.host.recording.durationSeconds().toFixed(1)} 秒）。` +
          '再生と読み込みは右の「表情アニメーション」にあります。',
      );
      this.refresh();
      return;
    }
    if (this.mode !== 'tracking') return;
    this.host.recording.begin(this.host.presetNames());
    this.recordSeconds = 0;
    this.mode = 'recording';
    this.setNote(`最大 ${MAX_RECORDING_SECONDS} 秒で自動的に止まります。`);
    this.refresh();
  }

  private save(): void {
    const recording = this.host.recording.current();
    if (recording === null) return;
    const blob = new Blob([serializeRecording(recording)], { type: 'application/json' });
    const anchor = document.createElement('a');
    anchor.href = URL.createObjectURL(blob);
    anchor.download = `expression-${timestamp()}.json`;
    anchor.click();
    URL.revokeObjectURL(anchor.href);
    this.setNote(`保存しました（${anchor.download}）。`);
  }

  // ---- 駆動 ----

  private takeDriver(mode: Exclude<WebcamMode, 'off'>): void {
    this.mode = mode;
    this.host.setExpressionDriver(this.driver);
  }

  private releaseDriver(mode: WebcamMode): void {
    this.mode = mode;
    this.host.setExpressionDriver(null);
  }

  private resetTrackingState(): void {
    this.ensurePlan();
    this.targetWeights.fill(0);
    this.smoothedWeights.fill(0);
    this.targetBlink = 0;
    this.smoothedBlink = 0;
    this.faceLostSeconds = 0;
    this.lastVideoTime = -1;
  }

  /** プリセットの並びが変わっていれば対応表を作り直す。 */
  private ensurePlan(): TrackingPlan {
    const names = this.host.presetNames();
    if (this.plan === null || this.planNames !== names) {
      this.plan = resolveTrackingPlan(names);
      this.planNames = names;
      this.targetWeights = new Float64Array(names.length);
      this.smoothedWeights = new Float64Array(names.length);
      this.reportedCategories = false;
      if (this.plan.unknownPresets.length > 0) {
        console.warn(
          '表情の対応表が参照するプリセットがアセットに無い（対応表が古い）: ' +
            this.plan.unknownPresets.join(', '),
        );
      }
      if (this.plan.unmappedPresets.length > 0) {
        console.info(
          'カメラでは駆動されない表情プリセット（ARKit 側に対応する項目が無い）: ' +
            this.plan.unmappedPresets.join(', '),
        );
      }
    }
    return this.plan;
  }

  /** ビューアーから毎フレーム呼ばれる。 */
  private advance(weights: Float64Array, deltaSeconds: number): string | null {
    if (this.mode !== 'tracking' && this.mode !== 'recording') return null;
    return this.advanceTracking(weights, deltaSeconds);
  }

  private advanceTracking(weights: Float64Array, deltaSeconds: number): string | null {
    const plan = this.ensurePlan();
    if (weights.length !== plan.presetCount) return null;

    const video = this.input.video;
    // 同じ映像フレームを 2 回推論しない（描画は 60fps、カメラは 30fps のことが多い）。
    if (this.cameraOn && video.readyState >= 2 && video.currentTime !== this.lastVideoTime) {
      this.lastVideoTime = video.currentTime;
      const scores = this.tracker.detect(video, performance.now());
      if (scores === null) {
        this.faceLostSeconds += deltaSeconds;
      } else {
        this.faceLostSeconds = 0;
        this.targetBlink = blendshapesToTargets(plan, scores, this.targetWeights).blink;
        this.reportCategories(plan, scores);
      }
    } else if (!this.cameraOn) {
      this.faceLostSeconds += deltaSeconds;
    }
    if (this.faceLostSeconds > FACE_LOST_SECONDS) {
      this.targetWeights.fill(0);
      this.targetBlink = 0;
    }

    smoothToward(
      this.smoothedWeights,
      this.targetWeights,
      smoothingFactor(deltaSeconds, EXPRESSION_TIME_CONSTANT_SECONDS),
    );
    this.smoothedBlink = smoothScalar(
      this.smoothedBlink,
      this.targetBlink,
      smoothingFactor(deltaSeconds, BLINK_TIME_CONSTANT_SECONDS),
    );
    weights.set(this.smoothedWeights);

    if (this.mode === 'recording') {
      this.recordSeconds += deltaSeconds;
      const full = this.host.recording.append(
        this.recordSeconds,
        this.smoothedWeights,
        this.smoothedBlink,
      );
      if (full) {
        this.mode = 'tracking';
        this.setNote(`${MAX_RECORDING_SECONDS} 秒に達したので録画を止めました。`);
        this.refresh();
      }
    }
    this.updateTimer(false);
    return strongestPreset(plan, this.smoothedWeights);
  }

  /** 対応表が求めるカテゴリのうち返ってこなかったものを 1 回だけ出す。 */
  private reportCategories(plan: TrackingPlan, scores: ReadonlyMap<string, number>): void {
    if (this.reportedCategories) return;
    this.reportedCategories = true;
    const missing = missingCategories(plan, scores.keys());
    if (missing.length > 0) {
      console.warn(
        '対応表が参照する blendshape が返ってこない（0 として扱う。モデルの版が違う可能性）: ' +
          missing.join(', '),
      );
    }
  }

  // ---- 表示 ----

  private updateTimer(force: boolean): void {
    const text = this.timerText();
    if (!force && text === this.shownTimer) return;
    this.shownTimer = text;
    this.elements.timer.textContent = text;
  }

  private timerText(): string {
    if (this.mode === 'recording') {
      return `録画 ${this.recordSeconds.toFixed(1)} / ${MAX_RECORDING_SECONDS.toFixed(1)} s`;
    }
    const frames = this.host.recording.frameCount();
    if (frames === 0) return '未収録';
    return `収録 ${this.host.recording.durationSeconds().toFixed(1)} s / ${frames} フレーム`;
  }

  private setNote(message: string, isError = false): void {
    this.elements.note.textContent = message;
    this.elements.note.classList.toggle('error', isError);
  }

  /** 失敗はパネルの中とツールバーの両方へ出す（パネルを閉じていても気付ける）。 */
  private reportFailure(error: unknown): void {
    const report = describeFailure(error);
    const message = report.remedy === null ? report.cause : `${report.cause}\n${report.remedy}`;
    this.setNote(message, true);
    this.host.setStatus(report.cause, true);
  }
}

interface PanelElements {
  readonly panel: HTMLElement;
  readonly state: HTMLElement;
  readonly close: HTMLButtonElement;
  readonly camera: HTMLButtonElement;
  readonly capture: HTMLButtonElement;
  readonly track: HTMLButtonElement;
  readonly record: HTMLButtonElement;
  readonly save: HTMLButtonElement;
  readonly timer: HTMLElement;
  readonly note: HTMLElement;
}

function collectElements(): PanelElements {
  return {
    panel: requireElement<HTMLElement>('webcam-panel'),
    state: requireElement<HTMLElement>('webcam-state'),
    close: requireElement<HTMLButtonElement>('btn-webcam-close'),
    camera: requireElement<HTMLButtonElement>('btn-camera'),
    capture: requireElement<HTMLButtonElement>('btn-capture'),
    track: requireElement<HTMLButtonElement>('btn-track'),
    record: requireElement<HTMLButtonElement>('btn-record'),
    save: requireElement<HTMLButtonElement>('btn-save'),
    timer: requireElement<HTMLElement>('webcam-timer'),
    note: requireElement<HTMLElement>('webcam-note'),
  };
}

function requireElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (element === null) throw new Error(`要素 ${id} が index.html に無い`);
  return element as T;
}

/** ファイル名に入れる時刻（ローカル時刻の `YYYYMMDD-HHMMSS`）。 */
function timestamp(): string {
  const now = new Date();
  const pad = (value: number): string => String(value).padStart(2, '0');
  return (
    `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}` +
    `-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`
  );
}
