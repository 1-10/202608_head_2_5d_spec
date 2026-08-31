// 入口。UI 配線・書き出しの起動・3Dビューと検査画像の表示。
//
// **ここに判断を置かない。** 失敗時の扱い（どの段で何が起きて、どうすればよいか）は
// `application/exportGuest.describeFailure` が持ち、パラメータの既定値と範囲は
// `application/settings` が持つ。ここがするのは、それを画面へ出すことだけ。

import './style.css';
import { bakeReport } from '../domain/atlas/bake';
import { LAYER_ORDER } from '../domain/preview/asset';
import { buildPreviewScene } from '../domain/preview/scene';
import { irisToLimbusRatio } from '../domain/eyes/bake';
import { EYE_SIDES } from '../domain/eyes/layout';
import { depthCoverage } from '../domain/hair/shell';
import { PhotoRgb } from '../domain/photo';
import { ExportOutcome, describeFailure, isPipelineError } from '../application/exportGuest';
import { Exporter, GnmAssetBundle, buildGuestZip } from '../composition';
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

const elements = {
  buttonWebcam: requireElement<HTMLButtonElement>('btn-webcam'),
  buttonUpload: requireElement<HTMLButtonElement>('btn-upload'),
  buttonReset: requireElement<HTMLButtonElement>('btn-reset'),
  buttonExport: requireElement<HTMLButtonElement>('btn-export'),
  fileInput: requireElement<HTMLInputElement>('file-input'),
  status: requireElement<HTMLElement>('status-message'),
  report: requireElement<HTMLElement>('report'),
  viewport: requireElement<HTMLElement>('canvas-head'),
  viewReadout: requireElement<HTMLElement>('readout-view'),
  video: requireElement<HTMLVideoElement>('webcam-video'),
  guiExport: requireElement<HTMLElement>('gui-export'),
  guiView: requireElement<HTMLElement>('gui-view'),
  inspection: requireElement<HTMLElement>('inspection'),
  bottomPanel: requireElement<HTMLElement>('bottom-panel'),
};

/**
 * 下段のタブ（検査画像 / 内訳）。
 *
 * `aria-selected` を状態の正本にして、pane の `hidden` をそこから作る。**別の変数で状態を持たない**
 * （二重管理になり、片方だけ更新した状態が画面に残る）。
 */
function setupTabs(panel: HTMLElement): void {
  const tabs = [...panel.querySelectorAll<HTMLButtonElement>('.tab')];
  const select = (name: string): void => {
    for (const tab of tabs) {
      const selected = tab.dataset.pane === name;
      tab.setAttribute('aria-selected', String(selected));
      const pane = document.getElementById(tab.dataset.pane ?? '');
      if (pane !== null) pane.hidden = !selected;
    }
  };
  for (const tab of tabs) {
    tab.addEventListener('click', () => select(tab.dataset.pane ?? ''));
  }
}

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

let photo: PhotoRgb | null = null;
let outcome: ExportOutcome | null = null;
let bundle: GnmAssetBundle | null = null;
let busy = false;

/** ビューの値をまとめてビューアーへ移す。**片方だけ適用する経路を作らない。** */
function applyViewSettings(view: ViewSettings): void {
  viewer.fovDegrees = view.fovDegrees;
  viewer.distanceMeters = view.distanceMeters;
  viewer.setBackground(view.background);
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
}

const gui: GuiHandle = setupGui(
  { exportPanel: elements.guiExport, viewPanel: elements.guiView },
  panelState,
  {
    onLayerVisibilityChanged: (layer, visible) => viewer.setLayerVisible(layer, visible),
    onLayerTextureChanged: (layer, enabled) => viewer.setLayerTextureEnabled(layer, enabled),
    onAllTexturesToggled: () => viewer.toggleAllTextures(),
    onResetView: () => viewer.resetView(),
    onViewSettingsChanged: (view) => applyViewSettings(view),
    onExpressionChanged: (name, weight) => viewer.setManualExpression(name, weight),
  },
);

viewer.onViewChanged = (): void => {
  updateViewReadout();
  gui.syncViewControls(viewer.layerStates(), viewer.textureStates());
  gui.syncHeadPose(viewer.headPose);
};

// キー操作は 3Dビューが持つ（層・テクスチャ・視点のリセット）。入力欄にフォーカスがあるときは
// 拾わない。
window.addEventListener('keydown', (event) => {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  const target = event.target as HTMLElement | null;
  if (target !== null && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
  if (viewer.handleKey(event.code)) event.preventDefault();
});

function setStatus(message: string, isError = false): void {
  elements.status.textContent = message;
  elements.status.classList.toggle('error', isError);
}

function updateViewReadout(): void {
  const degrees = (radians: number): string => ((radians * 180) / Math.PI).toFixed(1);
  const pose = viewer.headPose;
  const expression = viewer.currentExpression === null ? '' : ` / 表情 ${viewer.currentExpression}`;
  elements.viewReadout.textContent =
    `カメラ Yaw ${degrees(viewer.orbitYaw)}° / Pitch ${degrees(viewer.orbitPitch)}° /` +
    ` Zoom ${viewer.zoom.toFixed(2)}x` +
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
    const result = await exporter.run(photo, toExportSettings(panelState), (stage) =>
      setStatus(`段「${stage}」を実行しています…`),
    );
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
    renderInspection(elements.inspection, result.inspection);
    elements.report.textContent = buildReport(result);
    setStatus('');
  } catch (error) {
    console.error(error);
    const report = describeFailure(error);
    const stage = report.stage === null ? '' : `段「${report.stage}」で`;
    const remedy = report.remedy === null ? '' : `\n${report.remedy}`;
    setStatus(`${stage}失敗しました（${report.errorType}）: ${report.cause}${remedy}`, true);
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
  if (inputManager.isWebcamActive) {
    inputManager.stopWebcam();
    elements.buttonWebcam.textContent = 'Webcam';
  }
  try {
    await acceptPhoto(await inputManager.loadFromFile(file));
  } catch (error) {
    setStatus(describeFailure(error).cause, true);
  }
  elements.fileInput.value = '';
});

elements.buttonWebcam.addEventListener('click', async () => {
  try {
    if (!inputManager.isWebcamActive) {
      setStatus('Webcam を起動しています…');
      await inputManager.startWebcam();
      elements.buttonWebcam.textContent = 'Capture';
      setStatus('正面を向いて Capture を押してください。');
      return;
    }
    const captured = inputManager.captureWebcamFrame();
    inputManager.stopWebcam();
    elements.buttonWebcam.textContent = 'Webcam';
    await acceptPhoto(captured);
  } catch (error) {
    console.error(error);
    setStatus('Webcam にアクセスできませんでした。', true);
  }
});

elements.buttonReset.addEventListener('click', () => {
  inputManager.stopWebcam();
  elements.buttonWebcam.textContent = 'Webcam';
  photo = null;
  outcome = null;
  viewer.dispose();
  elements.inspection.replaceChildren();
  elements.report.textContent = '';
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
}

setupTabs(elements.bottomPanel);
updateViewReadout();
updateButtons();
animate();

// GNM アセットは 32MB あるので、写真を待たずに落とし始める（初回の書き出しの待ちを短くする）。
void exporter
  .loadAsset()
  .then((loaded) => {
    bundle = loaded;
    gui.setExpressionPresets(loaded.preview.expressionPresetNames);
    setStatus('写真を選んでください（ファイル / Webcam）。');
  })
  .catch((error) => setStatus(describeFailure(error).cause, true));
