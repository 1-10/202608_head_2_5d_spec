// 入口。UI 配線・書き出しの起動・3Dビューと検査画像の表示。
//
// **ここに判断を置かない。** 失敗時の扱い（どの段で何が起きて、どうすればよいか）は
// `application/exportGuest.describeFailure` が持ち、パラメータの既定値と範囲は
// `application/settings` が持つ。ここがするのは、それを画面へ出すことだけ。

import { bakeReport } from '../domain/atlas/bake';
import { LAYER_ORDER } from '../domain/preview/asset';
import { buildPreviewScene } from '../domain/preview/scene';
import { irisToLimbusRatio } from '../domain/eyes/bake';
import { EYE_SIDES } from '../domain/eyes/layout';
import { depthCoverage } from '../domain/hair/shell';
import { PhotoRgb } from '../domain/photo';
import {
  ExportOutcome,
  STAGE_NAMES,
  describeFailure,
  isPipelineError,
} from '../application/exportGuest';
import { Exporter, GnmAssetBundle, createFaceExpressionTracker } from '../composition';
import {
  GuiHandle,
  createPanelState,
  setupGui,
  toExportSettings,
  toViewSettings,
} from './gui';
import { InputManager } from './input';
import { renderInspection } from './inspectionView';
import { Viewer } from './viewer';
import { ViewSettings } from './viewSettings';
import { RecordingPlayer } from './recordingPlayer';
import { VisemeDriver } from './visemeDriver';
import { ExpressionDriver, WebcamPanel } from './webcamPanel';
import { parseRecording } from '../domain/preview/recording';

const elements = {
  buttonWebcam: requireElement<HTMLButtonElement>('btn-webcam'),
  buttonUpload: requireElement<HTMLButtonElement>('btn-upload'),
  fileInput: requireElement<HTMLInputElement>('file-input'),
  recordingInput: requireElement<HTMLInputElement>('recording-input'),
  statusBox: requireElement<HTMLElement>('viewport-status'),
  statusTitle: requireElement<HTMLElement>('status-title'),
  statusStages: requireElement<HTMLElement>('status-stages'),
  buttonStatusClose: requireElement<HTMLButtonElement>('btn-status-close'),
  report: requireElement<HTMLElement>('report'),
  viewport: requireElement<HTMLElement>('canvas-head'),
  video: requireElement<HTMLVideoElement>('webcam-video'),
  guiExport: requireElement<HTMLElement>('gui-export'),
  guiView: requireElement<HTMLElement>('gui-view'),
  inspection: requireElement<HTMLElement>('inspection'),
  overlay: requireElement<HTMLElement>('overlay'),
  overlayTitle: requireElement<HTMLElement>('overlay-title'),
  buttonInspection: requireElement<HTMLButtonElement>('btn-inspection'),
  buttonReport: requireElement<HTMLButtonElement>('btn-report'),
  buttonOverlayClose: requireElement<HTMLButtonElement>('btn-overlay-close'),
};

/** オーバーレイに出せるもの。切り替えはツールバーのボタンなので、中にタブは持たない。 */
const OVERLAY_PANES = {
  inspection: { element: () => elements.inspection, title: '検査画像（各段の出力そのもの）' },
  report: { element: () => elements.report, title: '内訳' },
} as const;

type OverlayPane = keyof typeof OVERLAY_PANES;

/**
 * 検査画像と内訳のオーバーレイ。
 *
 * **状態は DOM が正本**（`#overlay` の `hidden` と各 pane の `hidden`）。別の変数で持つと二重管理に
 * なり、片方だけ更新した状態が画面に残る。
 */
const overlay = {
  /** 出すものを選ぶ。同じものが既に出ていれば閉じる（ツールバーのボタンがトグルになる）。 */
  toggle(name: OverlayPane): void {
    if (!elements.overlay.hidden && overlay.current() === name) {
      overlay.close();
      return;
    }
    elements.overlay.hidden = false;
    for (const [key, pane] of Object.entries(OVERLAY_PANES)) {
      pane.element().hidden = key !== name;
    }
    elements.overlayTitle.textContent = OVERLAY_PANES[name].title;
    overlay.syncButtons();
  },

  close(): void {
    elements.overlay.hidden = true;
    overlay.syncButtons();
  },

  current(): OverlayPane | null {
    const keys = Object.keys(OVERLAY_PANES) as OverlayPane[];
    return keys.find((key) => !OVERLAY_PANES[key].element().hidden) ?? null;
  },

  syncButtons(): void {
    const open = elements.overlay.hidden ? null : overlay.current();
    elements.buttonInspection.setAttribute('aria-pressed', String(open === 'inspection'));
    elements.buttonReport.setAttribute('aria-pressed', String(open === 'report'));
  },
};

function requireElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (element === null) throw new Error(`要素 ${id} が index.html に無い`);
  return element as T;
}

// **パラメータは保存しない。** 毎回 `application/settings` と `presentation/viewSettings` の既定から
// 始める。触った値が残っていると「前回いじった値のまま書き出した」に気付けないし、ブラウザや
// プロファイルを変えると再現しないので、保存されている方が混乱の種になる。
const panelState = createPanelState();
const exporter = new Exporter();
const inputManager = new InputManager(elements.video);
const viewer = new Viewer(elements.viewport);
// 口形の連続再生。手で立てるぶんは表情と同じ経路（`setManualExpression`）を通るので、ここが持つのは
// 「時間で切り替える」ぶんだけ。
const visemeDriver = new VisemeDriver();
// 収録した表情アニメーションの持ち主。**Webカメラは録って積むだけ** — 再生と読み込みは
// カメラを使わない操作なので、右パネルの「表情アニメーション」から動かす。
const recordingPlayer = new RecordingPlayer();

let photo: PhotoRgb | null = null;
let bundle: GnmAssetBundle | null = null;
let busy = false;
/** Webカメラが表情の駆動を握っているときの差し込み口（握っていなければ `null`）。 */
let webcamDriver: ExpressionDriver | null = null;

/**
 * Webカメラ（映像・写真の取り込み・表情トラッキング・録画・再生）。
 *
 * **判断はパネル側が持つ。** ここは「表情の駆動を誰が握るか」の口を渡すだけ — ビューアーへ直に
 * 触らせないので、駆動源が増えても配線の形は変わらない。
 */
const webcamPanel = new WebcamPanel(inputManager, createFaceExpressionTracker(), {
  presetNames: () => viewer.expressionNames(),
  setExpressionDriver: (driver) => {
    webcamDriver = driver;
    // カメラが顔を取ったら他の駆動源は止める（**駆動源はひとつだけ**）。黙って無視すると、
    // 再生ボタンが「停止」のままそちらが出ない状態になる。
    if (driver !== null) stopOtherDrivers('webcam');
    applyExpressionDriver();
  },
  acceptPhoto: (next) => acceptPhoto(next),
  setStatus: (message, isError) => setStatus(message, isError),
  recording: recordingPlayer,
  setHeadPose: (yawDegrees, pitchDegrees) => {
    // **マウス追従は切る。** あちらが入ったままだと `setHeadPose` を受け付けず、カメラで首を
    // 動かしているつもりでカーソルが向きを決め続ける（どちらが動かしているか読めなくなる）。
    if (panelState.view.followPointer) {
      panelState.view.followPointer = false;
      viewer.followPointer = false;
    }
    viewer.setHeadPose({
      headYawDegrees: yawDegrees,
      headPitchDegrees: pitchDegrees,
      gazeYawDegrees: 0,
      gazePitchDegrees: 0,
    });
  },
});
webcamPanel.onOpenChanged = (): void => {
  elements.buttonWebcam.setAttribute('aria-pressed', String(webcamPanel.isOpen));
};

/** ビューの値をまとめてビューアーへ移す。**片方だけ適用する経路を作らない。** */
function applyViewSettings(view: ViewSettings): void {
  viewer.fovDegrees = view.fovDegrees;
  viewer.setCameraTransform(
    [view.cameraPositionX, view.cameraPositionY, view.cameraPositionZ],
    view.cameraPitchDegrees,
    view.cameraYawDegrees,
  );
  viewer.setBackground(view.background);
  viewer.setLighting({
    lightColor: view.lightColor,
    lightIntensity: view.lightIntensity,
    ambientColor: view.ambientColor,
    ambient: view.ambient,
  });
  viewer.setWireframe(view.showWireframe);
  viewer.neckShare = view.neckShare;
  viewer.followPointer = view.followPointer;
  viewer.setHeadPose({
    headYawDegrees: view.headYawDegrees,
    headPitchDegrees: view.headPitchDegrees,
    gazeYawDegrees: view.gazeYawDegrees,
    gazePitchDegrees: view.gazePitchDegrees,
  });
  viewer.playMode = view.playMode;
  viewer.fadeSeconds = view.fadeSeconds;
  viewer.holdSeconds = view.holdSeconds;
  viewer.expressionIntensity = view.expressionIntensity;
  viewer.blinkEnabled = view.blinkEnabled;
  visemeDriver.apply(view);
  recordingPlayer.setLoop(view.recordingLoop);
  applyExpressionDriver();
}

/**
 * 表情とまばたきの駆動を誰が握るかを決める。**ここが唯一の決め所。**
 *
 * 駆動源は Webカメラのトラッキング・収録した表情アニメーションの再生・口形の連続再生の 3 つあり、
 * **同時には動かない**。
 * ビューアーの口（`expressionOverride` / `blinkOverride`）へ各々が勝手に差すと、あとから
 * `applyViewSettings` が走っただけでカメラの駆動が黙って外れる（実際にそうなっていた）。
 *
 * 順は**カメラが先**。カメラは利用者が今まさに顔を映しているもので、再生より意図が強い。ただし
 * 取り合いにはしない — **新しく始めるものが、他を止めてから取る**（後から押した方が勝つ）ので、
 * ここへ来る時点で動いているのは 1 つだけ。順は取りこぼしへの保険。
 */
function applyExpressionDriver(): void {
  const driver = webcamDriver ?? (recordingPlayer.isPlaying ? recordingPlayer : null);
  if (driver !== null) {
    viewer.expressionOverride = (weights, delta) => driver.expression(weights, delta);
    viewer.blinkOverride = () => driver.blink();
    return;
  }
  // 連続再生していなければ `null`（顔を駆動するのは手のスライダーと表情の自動再生）。
  viewer.expressionOverride = visemeDriver.frame;
  viewer.blinkOverride = null;
}

/**
 * これから顔を駆動するもの以外を止める。
 *
 * **止めるだけでなく画面も戻す。** 止めた側のボタンが「停止」のまま残ると、押しても何も起きない
 * ボタンになる。
 */
function stopOtherDrivers(next: 'webcam' | 'recording' | 'viseme'): void {
  if (next !== 'webcam') webcamPanel.stopDriving();
  if (next !== 'recording' && recordingPlayer.isPlaying) recordingPlayer.stop();
  if (next !== 'viseme' && visemeDriver.isPlaying) visemeDriver.stop();
  syncPlaybackControls();
}

/**
 * 再生系のボタンと状態表示を、今の駆動の状態へ合わせる。
 *
 * **毎フレーム呼ぶ**（`animate`）。録画中は秒数が増え、再生中は位置が進み、どちらも終端で自分から
 * 止まるので、押した瞬間だけ合わせても追いつかない。**変わったときだけ DOM へ書く** — lil-gui の
 * `listen()` は毎フレーム読みに行くので、増える表示のたびに使うと数が増えるほど重くなる。
 */
let shownPlayback = '';
function syncPlaybackControls(): void {
  const summary = recordingSummary();
  const key = `${visemeDriver.isPlaying}|${recordingPlayer.isPlaying}|${summary}`;
  if (key === shownPlayback) return;
  shownPlayback = key;
  gui.syncVisemePlayback(visemeDriver.isPlaying);
  gui.syncRecording(recordingPlayer.isPlaying, summary);
}

/** 「表情アニメーション」の状態表示（1 行）。 */
function recordingSummary(): string {
  if (!recordingPlayer.hasRecording) return '未収録';
  const duration = recordingPlayer.durationSeconds().toFixed(1);
  const frames = recordingPlayer.frameCount();
  if (recordingPlayer.isPlaying) {
    return `再生 ${recordingPlayer.positionSeconds.toFixed(1)} / ${duration} s`;
  }
  return `${duration} s / ${frames} フレーム`;
}

/** 口形の連続再生を切り替え、ビューアーの駆動とボタンのラベルを合わせる。 */
function toggleVisemePlayback(): void {
  if (visemeDriver.isPlaying) {
    visemeDriver.stop();
  } else {
    stopOtherDrivers('viseme');
    visemeDriver.play();
  }
  applyExpressionDriver();
  syncPlaybackControls();
}

/** 収録した表情アニメーションの再生を切り替える。 */
function toggleRecordingPlayback(): void {
  if (recordingPlayer.isPlaying) {
    recordingPlayer.stop();
  } else {
    stopOtherDrivers('recording');
    if (!recordingPlayer.play()) {
      setStatus('再生できる収録がありません。Webカメラで録画するか、読み込んでください。', true);
      syncPlaybackControls();
      return;
    }
  }
  applyExpressionDriver();
  syncPlaybackControls();
}

/** 収録した表情アニメーション（JSON）を読み込む。 */
async function loadRecording(file: File): Promise<void> {
  // **読み込む前に、まだ 3D ビューへ頭が出ていないなら断る。** プリセット名が空のまま
  // `parseRecording` へ渡すと「一致する名前が 1 つも無い」枝へ落ち、**別のアセットで録った**という
  // 嘘の理由が出る（ページを開いて最初に押すだけで踏める）。
  if (viewer.expressionNames().length === 0) {
    setStatus('先に写真を通してください（収録の表情を当てる頭がまだありません）。', true);
    return;
  }
  // **録画中なら止めてから差し替える。** クリップの持ち主はひとつなので、差し替えただけでは
  // Webカメラは録画を続け、読み込んだクリップの後ろへ接ぎ木される（「保存」で混ざったものが出る）。
  stopOtherDrivers('recording');
  webcamPanel.stopRecording();
  try {
    const loaded = parseRecording(await file.text(), viewer.expressionNames());
    recordingPlayer.setRecording(loaded.recording);
    applyExpressionDriver();
    const notes = [
      `収録を読み込みました（${loaded.recording.frames.length} フレーム）。`,
    ];
    if (loaded.droppedPresets.length > 0) {
      notes.push(`今のアセットに無い表情を落としました: ${loaded.droppedPresets.join(', ')}`);
    }
    if (loaded.truncated) notes.push('上限を超えるぶんは切りました。');
    if (loaded.droppedFrames > 0) {
      notes.push(`時刻が逆行した ${loaded.droppedFrames} フレームを落としました。`);
    }
    setStatus(notes.join(' '));
  } catch (error) {
    console.error(error);
    const report = describeFailure(error);
    setStatus(report.remedy === null ? report.cause : `${report.cause} ${report.remedy}`, true);
  }
  syncPlaybackControls();
}

const gui: GuiHandle = setupGui(
  { exportPanel: elements.guiExport, viewPanel: elements.guiView },
  panelState,
  {
    onLayerVisibilityChanged: (layer, visible) => viewer.setLayerVisible(layer, visible),
    onLayerTextureChanged: (layer, enabled) => viewer.setLayerTextureEnabled(layer, enabled),
    onAllTexturesToggled: () => viewer.toggleAllTextures(),
    onResetView: () => viewer.resetView(),
    onLookAtTarget: () => viewer.lookAtTarget(),
    onResetCamera: () => viewer.resetCamera(),
    onViewSettingsChanged: (view) => applyViewSettings(view),
    onExpressionChanged: (name, weight) => viewer.setManualExpression(name, weight),
    onVisemePlayToggled: () => toggleVisemePlayback(),
    onRecordingPlayToggled: () => toggleRecordingPlayback(),
    onRecordingLoad: () => elements.recordingInput.click(),
  },
);

// ループ無しの連続再生は終端で自分から止まる。ボタンのラベルはそのときにも合わせ直す。
visemeDriver.onFinished = (): void => {
  applyExpressionDriver();
  syncPlaybackControls();
};

// 収録の再生も終端で自分から止まる。録り直し・読み込みで中身が変われば状態表示も変える。
recordingPlayer.onFinished = (): void => {
  applyExpressionDriver();
  syncPlaybackControls();
};
recordingPlayer.onRecordingChanged = (): void => syncPlaybackControls();

viewer.onViewChanged = (): void => {
  gui.syncViewControls(viewer.layerStates(), viewer.textureStates());
  gui.syncHeadPose(viewer.headPose);
  gui.syncCameraPose(viewer.cameraPose);
};

// `Esc` でオーバーレイ（検査画像・内訳）を閉じる。**それ以外のキーは持たない** — 層やテクスチャの
// 切り替えは右パネルが入口で、同じ操作の入口を 2 つ持つと片方だけ状態が動く経路を塞ぎ続けることに
// なる。
window.addEventListener('keydown', (event) => {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  if (event.code === 'Escape' && !elements.overlay.hidden) {
    overlay.close();
    event.preventDefault();
  }
});

/**
 * 3D ビューの中央に出す状態表示。**置き場はここだけ。**
 *
 * 進行（段の一覧）・完了・失敗・案内を同じ箱が出す。ツールバーの隅にも出していたが、処理の結果が
 * 出るのは 3D ビューなので、隅だけだと「どこで何が起きているか」が結び付かない — 2 か所へ出すのも
 * やめた（同じことを 2 か所が言うと、片方だけ古くなる）。
 *
 * **段の一覧は `STAGE_NAMES` から作る。** ここへ書き写すと、段が増減したとき画面だけ古くなる。
 */
const status = {
  /** 段の `<li>`（一覧は 1 回だけ作る）。 */
  items: new Map<string, HTMLElement>(),

  build(): void {
    for (const stage of STAGE_NAMES) {
      const item = document.createElement('li');
      item.textContent = stage;
      item.dataset.state = 'todo';
      elements.statusStages.appendChild(item);
      status.items.set(stage, item);
    }
  },

  /**
   * 段ごとの所要（秒）。**画面に出すために測る** — どの段が重いかは、写真の大きさや実行環境で
   * 変わるので、都度見えないと当てられない。
   */
  timings: [] as { stage: string; seconds: number }[],
  /** 今の段が始まった時刻（`performance.now()`）。 */
  startedAtMs: 0,
  /** 今出している段（走っていなければ `null`）。経過を毎フレーム書き直すのに使う。 */
  runningStage: null as string | null,
  /** 書き出し全体が始まった時刻。 */
  runStartedAtMs: 0,

  /** ひとこと出す。空文字で閉じる（段の一覧も隠す）。 */
  message(text: string, isError = false): void {
    status.runningStage = null;
    elements.statusTitle.textContent = text;
    elements.statusTitle.classList.toggle('error', isError);
    elements.statusStages.hidden = true;
    elements.statusBox.hidden = text === '';
  },

  /** 書き出しを始める（時計を初期化する）。 */
  beginRun(): void {
    status.runningStage = null;
    status.timings = [];
    status.runStartedAtMs = performance.now();
    status.startedAtMs = status.runStartedAtMs;
  },

  /** 今の段を出す（段の一覧つき）。前の段の所要をここで締める。 */
  stage(stage: string): void {
    const now = performance.now();
    const previous = STAGE_NAMES.indexOf(stage) - 1;
    if (previous >= 0 && status.timings.length === previous) {
      status.timings.push({
        stage: STAGE_NAMES[previous],
        seconds: (now - status.startedAtMs) / 1000,
      });
    }
    status.startedAtMs = now;
    status.runningStage = stage;
    const index = STAGE_NAMES.indexOf(stage);
    for (const [name, item] of status.items) {
      const at = STAGE_NAMES.indexOf(name);
      item.textContent = name;
      item.dataset.state = at < index ? 'done' : at === index ? 'active' : 'todo';
    }
    status.tick();
    elements.statusTitle.classList.remove('error');
    elements.statusStages.hidden = false;
    elements.statusBox.hidden = false;
  },

  /**
   * 経過を書き直す。**`animate` から毎フレーム呼ぶ。**
   *
   * 段の切り替わりでしか書かないと、重い段（推論は 30 秒かかることがある）の間ずっと同じ数が
   * 出たままで、進んでいるのか固まったのか分からない。
   */
  tick(): void {
    const stage = status.runningStage;
    if (stage === null) return;
    const index = STAGE_NAMES.indexOf(stage);
    const elapsed = ((performance.now() - status.runStartedAtMs) / 1000).toFixed(1);
    elements.statusTitle.textContent =
      index < 0
        ? `${stage}　経過 ${elapsed}s`
        : `${stage}（${index + 1} / ${STAGE_NAMES.length}）　経過 ${elapsed}s`;
  },

  /**
   * 失敗を出す。**段の一覧は出したまま**にする — どこまで進んで落ちたかが画面に残る方が原因を
   * 追える。
   */
  failure(text: string): void {
    status.runningStage = null;
    elements.statusTitle.textContent = text;
    elements.statusTitle.classList.add('error');
    elements.statusBox.hidden = false;
  },

  /**
   * 書き出しが終わったことと、かかった時間を出す。
   *
   * **閉じない。** 段の一覧を全部「済んだ」にして残す — どの段に時間がかかったかは、次に触る値を
   * 決めるのに使う（重いのがアトラスなら一辺を落とす、など）。
   */
  finishRun(): void {
    status.runningStage = null;
    const now = performance.now();
    const last = STAGE_NAMES.length - 1;
    if (status.timings.length === last) {
      status.timings.push({
        stage: STAGE_NAMES[last],
        seconds: (now - status.startedAtMs) / 1000,
      });
    }
    const total = ((now - status.runStartedAtMs) / 1000).toFixed(1);
    elements.statusTitle.textContent = `書き出し完了　${total}s`;
    elements.statusTitle.classList.remove('error');
    for (const [name, item] of status.items) {
      const timing = status.timings.find((entry) => entry.stage === name);
      item.textContent = timing === undefined ? name : `${name} ${timing.seconds.toFixed(1)}s`;
      item.dataset.state = 'done';
    }
    elements.statusStages.hidden = false;
    elements.statusBox.hidden = false;
  },

  /** 段の一覧を「まだ」へ戻して閉じる（名前も所要を外した形へ戻す）。 */
  reset(): void {
    for (const [name, item] of status.items) {
      item.textContent = name;
      item.dataset.state = 'todo';
    }
    status.timings = [];
    status.message('');
  },
};

/** 状態表示への入口（Webカメラなど外からも使う）。 */
function setStatus(message: string, isError = false): void {
  status.message(message, isError);
}



/** 書き出しを走らせ、3Dビューと検査画像と内訳を更新する。 */
async function runExport(): Promise<void> {
  if (photo === null || busy) return;
  busy = true;
  status.beginRun();
  try {
    const result = await exporter.run(photo, toExportSettings(panelState), (stage) =>
      status.stage(stage),
    );
    if (bundle === null) throw new Error('アセットが読めていない');
    const source = result.previewSceneSource;
    const scene = buildPreviewScene({
      vertices: source.vertices,
      headMesh: source.asset.mesh,
      preview: bundle.preview,
      skinAlbedo: {
        data: source.skinAlbedo,
        width: source.atlasSize,
        height: source.atlasSize,
      },
      eyeAlbedos: {
        left: {
          data: source.eyeAlbedos.left,
          width: source.eyeTextureSize,
          height: source.eyeTextureSize,
        },
        right: {
          data: source.eyeAlbedos.right,
          width: source.eyeTextureSize,
          height: source.eyeTextureSize,
        },
      },
      hair: source.hair,
      hairAlbedo: source.hairAlbedo,
      hairAlpha: source.hairAlpha,
    });
    if (scene.unassignedTriangleCount > 0) {
      console.warn(
        `どの領域にも入らない三角形が ${scene.unassignedTriangleCount} 個ある` +
          '（3D ビューでマゼンタに出る）。領域の設定かアセットが変わっている',
      );
    }
    viewer.setScene(scene, {
      preview: bundle.preview,
      restVertices: source.vertices,
      identity: result.headFit.identity,
      triangles: source.asset.mesh.triangles,
      uvSplitSource: source.asset.mesh.uvSplitSource,
    });
    // シーンを差し替えると表示状態と姿勢が初期化されるので、パネルを合わせ直す。
    for (const layer of LAYER_ORDER) {
      viewer.setLayerVisible(layer, panelState.visibleLayers[layer]);
      viewer.setLayerTextureEnabled(layer, panelState.texturedLayers[layer]);
    }
    applyViewSettings(toViewSettings(panelState));
    gui.setExpressionPresets(viewer.expressionNames());
    for (const [name, weight] of Object.entries(panelState.expressions)) {
      viewer.setManualExpression(name, weight);
    }
    // 3D ビューに頭が出て初めてトラッキングと再生が使える（重みを当てる先がある）。
    webcamPanel.refresh();
    renderInspection(elements.inspection, result.inspection);
    elements.report.textContent = buildReport(result);
    status.finishRun();
  } catch (error) {
    console.error(error);
    const report = describeFailure(error);
    const stage = report.stage === null ? '' : `段「${report.stage}」で`;
    const remedy = report.remedy === null ? '' : `\n${report.remedy}`;
    status.failure(`${stage}失敗しました（${report.errorType}）: ${report.cause}${remedy}`);
    if (!isPipelineError(error)) console.warn('想定外の失敗（バグの可能性）', error);
  } finally {
    busy = false;
  }
}

/** 内訳を人が読める形にまとめる（デスクトップ側が標準出力へ出しているもの）。 */
function buildReport(result: ExportOutcome): string {
  const lines: string[] = [];
  const manifest = result.artifacts.manifest;
  lines.push(
    `guest.json: format_version ${manifest.format_version} /` +
      ` identity ${manifest.identity_count} 成分 /` +
      ` GNM ${manifest.gnm_version} ${manifest.gnm_variant} /` +
      ` exporter ${manifest.exporter_version}`,
  );
  lines.push(
    'フィット残差 RMS（写真ピクセル）: ' +
      result.headFit.residualRmsPixels.map((value) => value.toFixed(2)).join(' → '),
  );
  for (const side of EYE_SIDES) {
    const albedo = result.eyeAlbedos[side];
    lines.push(
      `眼球 ${side}: 虹彩 ${albedo.irisRadiusPx.toFixed(1)}px /` +
        ` limbus ${albedo.limbusRadiusPx.toFixed(1)}px` +
        `（比 ${irisToLimbusRatio(albedo).toFixed(3)}）`,
    );
  }
  lines.push(bakeReport(result.atlas));
  if (result.hairShell === null) {
    lines.push('髪シェル: 髪が写っていないので作られていない（zip に髪系 3 つは入らない）');
  } else {
    lines.push(
      `髪シェル: 頂点 ${result.hairShell.vertexCount} / 三角形 ${result.hairShell.triangleCount} /` +
        ` Depth 被覆 ${(depthCoverage(result.hairShell) * 100).toFixed(1)}% /` +
        ` Depth 残差 ${(result.hairShell.depthFit.residualRmsMeters * 1000).toFixed(2)}mm`,
    );
  }
  const provider = exporter.depthNormalProvider;
  if (provider !== null) lines.push(`DAViD の実行環境: ${provider}`);
  // 段ごとの所要も内訳へ。画面の一覧は次の書き出しで消えるが、こちらは残して見比べられる。
  if (status.timings.length > 0) {
    const total = status.timings.reduce((sum, entry) => sum + entry.seconds, 0);
    lines.push(
      `所要 ${total.toFixed(1)}s（` +
        status.timings.map((entry) => `${entry.stage} ${entry.seconds.toFixed(1)}s`).join(' / ') +
        '）',
    );
  }
  return lines.join('\n');
}

async function acceptPhoto(next: PhotoRgb): Promise<void> {
  photo = next;
  elements.report.textContent = '';
  await runExport();
}

elements.buttonUpload.addEventListener('click', () => elements.fileInput.click());
elements.fileInput.addEventListener('change', async () => {
  const file = elements.fileInput.files?.[0];
  if (file === undefined) return;
  webcamPanel.stopCamera();
  try {
    await acceptPhoto(await inputManager.loadFromFile(file));
  } catch (error) {
    setStatus(describeFailure(error).cause, true);
  }
  elements.fileInput.value = '';
});

elements.recordingInput.addEventListener('change', async () => {
  const file = elements.recordingInput.files?.[0];
  elements.recordingInput.value = '';
  if (file === undefined) return;
  await loadRecording(file);
});

elements.buttonWebcam.addEventListener('click', () => webcamPanel.toggle());

window.addEventListener('resize', () => viewer.resize());

function animate(): void {
  requestAnimationFrame(animate);
  viewer.render();
  syncPlaybackControls();
  status.tick();
}

// 済んだ表示は自分で閉じられるようにする（終わった後も残り続けるので）。
elements.buttonStatusClose.addEventListener('click', () => status.message(''));
elements.buttonInspection.addEventListener('click', () => overlay.toggle('inspection'));
elements.buttonReport.addEventListener('click', () => overlay.toggle('report'));
elements.buttonOverlayClose.addEventListener('click', () => overlay.close());
status.build();
overlay.syncButtons();
animate();

// GNM アセットは 32MB あるので、写真を待たずに落とし始める（初回の書き出しの待ちを短くする）。
void exporter
  .loadAsset()
  .then((loaded) => {
    bundle = loaded;
    visemeDriver.setPreview(loaded.preview);
    gui.setExpressionPresets(loaded.preview.expressionPresetNames);
    setStatus('写真を選んでください（写真を選ぶ / Webカメラ）。');
  })
  .catch((error) => setStatus(describeFailure(error).cause, true));
