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
import { Exporter, GnmAssetBundle, buildGuestZip, createFaceExpressionTracker } from '../composition';
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
  buttonReset: requireElement<HTMLButtonElement>('btn-reset'),
  buttonExport: requireElement<HTMLButtonElement>('btn-export'),
  fileInput: requireElement<HTMLInputElement>('file-input'),
  recordingInput: requireElement<HTMLInputElement>('recording-input'),
  status: requireElement<HTMLElement>('status-message'),
  report: requireElement<HTMLElement>('report'),
  viewport: requireElement<HTMLElement>('canvas-head'),
  progress: requireElement<HTMLElement>('viewport-progress'),
  progressTitle: requireElement<HTMLElement>('progress-title'),
  progressStages: requireElement<HTMLElement>('progress-stages'),
  viewReadout: requireElement<HTMLElement>('readout-view'),
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
let outcome: ExportOutcome | null = null;
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
  updateViewReadout();
  gui.syncViewControls(viewer.layerStates(), viewer.textureStates());
  gui.syncHeadPose(viewer.headPose);
  gui.syncCameraPose(viewer.cameraPose);
};

// キー操作は 3Dビューが持つ（層・テクスチャ・視点のリセット）。入力欄にフォーカスがあるときは
// 拾わない。
window.addEventListener('keydown', (event) => {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  if (event.code === 'Escape' && !elements.overlay.hidden) {
    overlay.close();
    event.preventDefault();
    return;
  }
  const target = event.target as HTMLElement | null;
  if (target !== null && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
  if (viewer.handleKey(event.code)) event.preventDefault();
});

/**
 * 3D ビューの中の進行表示。
 *
 * **段の一覧は `STAGE_NAMES` から作る。** ここへ書き写すと、段が増減したとき画面だけ古くなる。
 * 済んだ段・今の段・まだの段を `data-state` で示し、色は CSS が持つ。
 */
const progress = {
  /** 段の `<li>`（一覧は 1 回だけ作る）。 */
  items: new Map<string, HTMLElement>(),

  build(): void {
    for (const stage of STAGE_NAMES) {
      const item = document.createElement('li');
      item.textContent = stage;
      item.dataset.state = 'todo';
      elements.progressStages.appendChild(item);
      progress.items.set(stage, item);
    }
  },

  /** 今の段を出す。`null` で閉じる。 */
  show(stage: string | null): void {
    if (stage === null) {
      elements.progress.hidden = true;
      return;
    }
    elements.progress.hidden = false;
    const index = STAGE_NAMES.indexOf(stage);
    elements.progressTitle.textContent =
      index < 0
        ? stage
        : `${stage}（${index + 1} / ${STAGE_NAMES.length}）`;
    for (const [name, item] of progress.items) {
      const at = STAGE_NAMES.indexOf(name);
      item.dataset.state = at < index ? 'done' : at === index ? 'active' : 'todo';
    }
  },

  /** 次の書き出しのために全部「まだ」へ戻す。 */
  reset(): void {
    for (const item of progress.items.values()) item.dataset.state = 'todo';
    elements.progressTitle.textContent = '';
    elements.progress.hidden = true;
  },
};

function setStatus(message: string, isError = false): void {
  elements.status.textContent = message;
  elements.status.classList.toggle('error', isError);
}

function updateViewReadout(): void {
  const pose = viewer.headPose;
  // **パネルと同じ言い方にする。** 右パネルの「カメラ」節に出るのと同じ位置 (m) と回転 (°) で、
  // 別の言い換え（拡大率など）をここだけで作らない。
  const camera = viewer.cameraPose;
  const meters = (value: number): string => value.toFixed(3);
  // **駆動源の名前はそのまま出す。** 表情の自動再生はプリセット名（英字）を返し、口形の連続再生は
  // 「口形 あ」と自分で名乗る。ここで「表情」と決め打ちすると、別の駆動源が差さったときに黙って
  // 嘘のラベルになる。
  const expression = viewer.currentExpression === null ? '' : ` / ${viewer.currentExpression}`;
  elements.viewReadout.textContent =
    `カメラ 位置 ${meters(camera.position[0])}, ${meters(camera.position[1])},` +
    ` ${meters(camera.position[2])} m /` +
    ` 回転 X ${camera.pitchDegrees.toFixed(1)}° / Y ${camera.yawDegrees.toFixed(1)}°` +
    ` — 首 ${pose.headYawDegrees.toFixed(1)}° / ${pose.headPitchDegrees.toFixed(1)}° /` +
    ` 視線 ${pose.gazeYawDegrees.toFixed(1)}° / ${pose.gazePitchDegrees.toFixed(1)}°${expression}`;
}

function updateButtons(): void {
  elements.buttonExport.disabled = busy || photo === null;
}

/** 書き出しを走らせ、3Dビューと検査画像と内訳を更新する。 */
async function runExport(): Promise<void> {
  if (photo === null || busy) return;
  busy = true;
  updateButtons();
  try {
    const result = await exporter.run(photo, toExportSettings(panelState), (stage) => {
      setStatus(`段「${stage}」を実行しています…`);
      progress.show(stage);
    });
    outcome = result;
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
    progress.reset();
    setStatus('');
  } catch (error) {
    console.error(error);
    const report = describeFailure(error);
    const stage = report.stage === null ? '' : `段「${report.stage}」で`;
    const remedy = report.remedy === null ? '' : `\n${report.remedy}`;
    setStatus(`${stage}失敗しました（${report.errorType}）: ${report.cause}${remedy}`, true);
    // **失敗した段を出したまま閉じる。** どこまで進んで落ちたかが画面に残る方が原因を追える。
    elements.progressTitle.textContent = `失敗: ${report.cause}`;
    if (!isPipelineError(error)) console.warn('想定外の失敗（バグの可能性）', error);
  } finally {
    busy = false;
    updateButtons();
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
  return lines.join('\n');
}

/** guest zip をダウンロードさせる。 */
async function downloadZip(): Promise<void> {
  if (outcome === null) return;
  const { blob, filename } = await buildGuestZip(outcome.artifacts);
  const anchor = document.createElement('a');
  anchor.href = URL.createObjectURL(blob);
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(anchor.href);
  setStatus(`書き出し完了: ${filename}（${(blob.size / 1024 / 1024).toFixed(1)}MB）`);
}

async function acceptPhoto(next: PhotoRgb): Promise<void> {
  photo = next;
  outcome = null;
  elements.report.textContent = '';
  updateButtons();
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

elements.buttonReset.addEventListener('click', () => {
  webcamPanel.reset();
  photo = null;
  outcome = null;
  viewer.dispose();
  elements.inspection.replaceChildren();
  elements.report.textContent = '';
  progress.reset();
  overlay.close();
  webcamPanel.refresh();
  updateButtons();
  setStatus('');
});

elements.buttonExport.addEventListener('click', () => {
  if (outcome === null) void runExport().then(() => downloadZip());
  else void downloadZip();
});

window.addEventListener('resize', () => viewer.resize());

function animate(): void {
  requestAnimationFrame(animate);
  viewer.render();
  syncPlaybackControls();
}

elements.buttonInspection.addEventListener('click', () => overlay.toggle('inspection'));
elements.buttonReport.addEventListener('click', () => overlay.toggle('report'));
elements.buttonOverlayClose.addEventListener('click', () => overlay.close());
progress.build();
overlay.syncButtons();
updateViewReadout();
updateButtons();
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
