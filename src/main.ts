// アプリケーションのエントリポイント。UI配線、Three.jsシーンの構築、
// 検出→GNM頭部構築→アニメーションループまでを統括する。

import * as THREE from 'three';
import './style.css';
import { InputManager, type CapturedImage } from './input';
import { FaceDetectionError, FaceDetector, type FaceLandmark } from './faceDetector';
import { normalizeFaceLandmarks, type NormalizedFaceResult } from './faceTopology';
import { refineMaskWithGuide } from './maskRefine';
import { PersonSegmenter } from './personSegmentation';
import { PortraitDepthEstimator } from './portraitDepth';
import { createBlinkState, updateBlink, type BlinkState } from './blink';
import { loadGnmModel, type GnmModel } from './gnmHead';
import {
  buildGnmHead,
  type GnmBuildContext,
  type GnmHeadBuild,
  type MeasuredHeadData,
} from './gnmHeadMesh';
import { loadExpressionSampler, type ExpressionSampler } from './gnmSampler';
import { createParams, type Params } from './params';
import { applyDebugVisualization, refreshEmotionOptions, setupDebugGui } from './debugView';
import { OrbitDragController } from './interaction';

class Viewport {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;
  private container: HTMLElement;
  private currentGroup: THREE.Object3D | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    container.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(30, 1, 0.05, 50);

    const ambient = new THREE.AmbientLight(0xffffff, 0.65);
    const key = new THREE.DirectionalLight(0xffffff, 0.9);
    key.position.set(0.6, 0.8, 1.2);
    this.scene.add(ambient, key);

    this.resize();
    // window resizeイベントだけだとコンテナ単独のレイアウト変化を取りこぼす
    new ResizeObserver(() => this.resize()).observe(container);
  }

  setGroup(group: THREE.Object3D): void {
    if (this.currentGroup) this.scene.remove(this.currentGroup);
    this.currentGroup = group;
    this.scene.add(group);
  }

  clear(): void {
    if (this.currentGroup) {
      this.scene.remove(this.currentGroup);
      this.currentGroup = null;
    }
  }

  setBackground(color: string): void {
    this.scene.background = new THREE.Color(color);
  }

  updateCamera(fovDeg: number, distance: number): void {
    this.camera.fov = fovDeg;
    this.camera.position.set(0, 0, distance);
    this.camera.lookAt(0, 0, 0);
    this.camera.updateProjectionMatrix();
  }

  resize(): void {
    // pixelRatioは毎回読み直す — ページズーム変更でdevicePixelRatioが変わるため。
    // 構築時の値で固定すると、ズーム状態でロードした場合に低解像度のまま
    // 引き伸ばされて全体がぼやける
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    const w = this.container.clientWidth || 1;
    const h = this.container.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  render(): void {
    this.renderer.render(this.scene, this.camera);
  }
}

interface SceneState {
  ctx: GnmBuildContext;
  sourceCanvas: HTMLCanvasElement; // DAViDの遅延推論で再利用する入力画像
  normalized: NormalizedFaceResult;
  rawLandmarks: FaceLandmark[]; // 元画像でのMediaPipe生検出値
  gnmHead: GnmHeadBuild | null;
  texture: THREE.Texture;
}

const params: Params = createParams();

const els = {
  btnWebcam: document.getElementById('btn-webcam') as HTMLButtonElement,
  btnUpload: document.getElementById('btn-upload') as HTMLButtonElement,
  btnReset: document.getElementById('btn-reset') as HTMLButtonElement,
  btnExport: document.getElementById('btn-export') as HTMLButtonElement,
  btnExportTemplate: document.getElementById('btn-export-template') as HTMLButtonElement,
  fileInput: document.getElementById('file-input') as HTMLInputElement,
  status: document.getElementById('status-message') as HTMLElement,
  paneHead: document.getElementById('canvas-head') as HTMLElement,
  yawReadout: document.getElementById('readout-yaw') as HTMLElement,
  pitchReadout: document.getElementById('readout-pitch') as HTMLElement,
  toggleBlink: document.getElementById('toggle-blink') as HTMLInputElement,
  video: document.getElementById('webcam-video') as HTMLVideoElement,
  guiContainer: document.getElementById('gui-container') as HTMLElement,
};

const viewport = new Viewport(els.paneHead);
viewport.setBackground(params.backgroundColor);

const inputManager = new InputManager(els.video);
const faceDetector = new FaceDetector();
const personSegmenter = new PersonSegmenter();
const portraitDepth = new PortraitDepthEstimator();

// GNMアセット (gnm_head_lite.bin 約8.5MB) は初回構築時に一度だけロードする。
let gnmModel: GnmModel | null = null;
let exprSampler: ExpressionSampler | null = null;
let gnmBusy = false;

let sceneState: SceneState | null = null;
let yawDeg = 0;
let pitchDeg = 0;
let blinkState: BlinkState = createBlinkState(performance.now(), params);
// --- GNM表情の自動巡回 (Emotion=AUTO時): 感情→ニュートラル→別の感情… ---
let gnmExprNextChangeAt = 0; // 次回遷移時刻
let gnmAutoTarget: number[] | null = null; // 現在の目標表情 (null=ニュートラル区間)
let gnmLastEmotion = ''; // 直前の感情 (同じ感情の連続を避ける)

// AUTO/RANDOM巡回のタイミング (Unityエクスポートのmeta.jsonにもこの値が入る)
const GNM_AUTO_CYCLE = {
  neutralMinMs: 800, // 感情の合間に挟むニュートラル区間
  neutralRandMs: 1200,
  holdMinMs: 2000, // 1つの感情を保持する時間
  holdRandMs: 2500,
};

/** 公式クラス名 → クラス番号。サンプラーの classNames が正本 */
function classIndex(name: string): number {
  return exprSampler ? exprSampler.classNames.indexOf(name) : -1;
}
/**
 * 公式 sample_expression でクラスの表情を作り、アセットの成分並びへ射影する。
 * latent=null は潜在空間の中心 (=クラスの代表)。公式には無い決め打ちだが、
 * 固定表情・パーツ別スライダーの基準として再現性が要る
 */
function sampleClass(name: string, latent: Float32Array | null = null): number[] | null {
  if (!exprSampler || !gnmModel) return null;
  const ci = classIndex(name);
  if (ci < 0) return null;
  return exprSampler.toModelCoeffs(exprSampler.sample(ci, latent), gnmModel);
}

// Emotion=MANUAL用のパーツ別スライダー定義。公式クラスの代表表情を領域で分離して合成する。
// 領域の判定は成分名 (model.expressionNames) から導出する — 成分数や並びを
// 変えても嘘にならないようにするため (位置決め打ちは舌成分の追加で壊れた)
const GNM_MANUAL_CONTROLS: { param: keyof Params; classes: string[]; region: 'eyes' | 'lower' }[] = [
  { param: 'gnmMouthOpen', classes: ['surprise'], region: 'lower' },
  { param: 'gnmSmile', classes: ['smile_wide'], region: 'lower' },
  { param: 'gnmPucker', classes: ['pucker'], region: 'lower' },
  { param: 'gnmCornersDown', classes: ['corners_down'], region: 'lower' },
  // 閉眼は片目ウインクの左右合成 (公式に「両目を閉じる」クラスは無い)
  { param: 'gnmEyesClose', classes: ['wink_left', 'wink_right'], region: 'eyes' },
  { param: 'gnmEyesWide', classes: ['surprise'], region: 'eyes' },
  { param: 'gnmSquint', classes: ['squint'], region: 'eyes' },
];

/** 表情成分ごとの領域ラベルを成分名から導出する (アセットの並びに依存しない)。 */
type ExprRegion = 'eyes' | 'lower' | 'other';
let exprRegionsCache: ExprRegion[] | null = null;
function expressionRegions(model: GnmModel): ExprRegion[] {
  if (exprRegionsCache) return exprRegionsCache;
  exprRegionsCache = model.expressionNames.map((n) =>
    /^(left|right)_eye/.test(n) ? 'eyes' : n.startsWith('lower_face') ? 'lower' : 'other',
  );
  return exprRegionsCache;
}

/**
 * クラスの代表表情を足し合わせる (キャッシュ付き)。
 * 出力ベクトルの線形和は公式の blend_expressions とは別物なので、
 * まばたきとMANUALモードのパーツ合成にだけ使う。
 */
const sampleSumCache = new Map<string, number[] | null>();
function sampleClassSum(classes: string[]): number[] | null {
  const key = classes.join('+');
  const hit = sampleSumCache.get(key);
  if (hit !== undefined) return hit;
  let acc: number[] | null = null;
  for (const c of classes) {
    const v = sampleClass(c);
    if (!v) return null;
    acc = acc ? acc.map((x, i) => x + v[i]) : v.slice();
  }
  sampleSumCache.set(key, acc);
  return acc;
}

/**
 * まばたきベクトル: 左右ウインクの合成から目領域だけを残す
 * (下顔面を0にしないと、まばたきのたびに口が動いてしまう)。
 */
function buildBlinkVector(model: GnmModel): number[] {
  const both = sampleClassSum(['wink_left', 'wink_right']);
  if (!both) return new Array<number>(model.expressionCount).fill(0);
  return both.map((v, i) => (/^(left|right)_eye/.test(model.expressionNames[i] ?? '') ? v : 0));
}

function setStatus(message: string, isError = false): void {
  els.status.textContent = message;
  els.status.classList.toggle('error', isError);
}

/** Yaw/Pitch角を更新し、頭部Groupへ反映する。 */
function updateYaw(deg: number): void {
  yawDeg = Math.min(params.maxYawDeg, Math.max(-params.maxYawDeg, deg));
  updateOrientationReadout();
  if (sceneState?.gnmHead) sceneState.gnmHead.group.rotation.y = THREE.MathUtils.degToRad(yawDeg);
}

function updatePitch(deg: number): void {
  pitchDeg = Math.min(params.maxPitchDeg, Math.max(-params.maxPitchDeg, deg));
  updateOrientationReadout();
  if (sceneState?.gnmHead) sceneState.gnmHead.group.rotation.x = THREE.MathUtils.degToRad(pitchDeg);
}

function updateOrientationReadout(): void {
  els.yawReadout.textContent = yawDeg.toFixed(1);
  els.pitchReadout.textContent = pitchDeg.toFixed(1);
}

new OrbitDragController([els.paneHead], {
  getYaw: () => yawDeg,
  setYaw: (v) => updateYaw(v),
  getMaxYawDeg: () => params.maxYawDeg,
  getPitch: () => pitchDeg,
  setPitch: (v) => updatePitch(v),
  getMaxPitchDeg: () => params.maxPitchDeg,
});

function disposeSceneState(): void {
  if (!sceneState) return;
  viewport.clear();
  sceneState.gnmHead?.dispose();
  sceneState.texture.dispose();
  sceneState = null;
  updateExportButton();
}

/**
 * UnityエクスポートボタンはGNM頭部が表示できているときだけ有効。
 * (Templateはモデル+サンプラーがあれば作れるが、両方とも初回構築時にロードされるので同条件でよい)
 */
function updateExportButton(): void {
  const disabled = exportBusy || !sceneState?.gnmHead;
  els.btnExport.disabled = disabled;
  els.btnExportTemplate.disabled = disabled;
}

/**
 * 実測ソース (セグメンテーション+Depth) を取得する。
 * 失敗した項目はnullとし、UVクランプ・髪シェルが自動的に省略される。
 */
async function acquireMeasuredData(
  captured: CapturedImage,
  normalized: NormalizedFaceResult,
): Promise<MeasuredHeadData> {
  const measured: MeasuredHeadData = {
    segmentation: null,
    segmentationRefined: null,
    depth: null,
    davidDepth: null,
    davidNormalCanvas: null,
    davidPerson: null,
  };

  try {
    setStatus('頭部をセグメントしています…');
    await personSegmenter.init();
    measured.segmentation = personSegmenter.segment(captured.canvas, normalized.landmarks);
  } catch (err) {
    console.warn('セグメンテーションに失敗。UVクランプ・髪シェルなしで表示します。', err);
  }

  // 髪系マスクをGuided Filterで写真エッジへ整合 (256px→768px)。
  // 「髪のみ」と「髪+帽子」は独立に精細化する — 合成後に掛けると帽子の縁が
  // 髪の縁と混ざる。失敗しても生マスクで続行できるよう分離してtryする
  if (measured.segmentation) {
    try {
      const t0 = performance.now();
      const seg = measured.segmentation;
      measured.segmentationRefined = {
        ...seg,
        hair: refineMaskWithGuide(captured.canvas, seg.hair),
        hairWithAccessories: refineMaskWithGuide(captured.canvas, seg.hairWithAccessories),
      };
      console.debug(`髪マスク精細化 (Guided Filter x2): ${(performance.now() - t0).toFixed(0)}ms`);
    } catch (err) {
      console.warn('髪マスクの精細化に失敗。生マスクで続行します。', err);
      measured.segmentationRefined = null;
    }
  }

  if (measured.segmentation) {
    try {
      setStatus('Depthを推定しています…');
      await portraitDepth.init();
      measured.depth = await portraitDepth.estimate(
        captured.canvas,
        measured.segmentation.person,
        normalized.headCenterPx,
        normalized.faceWidth,
      );
    } catch (err) {
      console.warn('Depth推定に失敗。髪シェルなしで表示します。', err);
      measured.depth = null;
    }
  }

  return measured;
}

let davidAcquisitionBusy = false;
let davidEstimator: import('./david').DavidEstimator | null = null;

/**
 * DAViD multi-task (Depth / 表面法線 / ソフト前景を1回の推論で同時取得) を
 * 必要時に遅延取得する。モデルDLが大きいため、いずれかのソースとして
 * 選択されて初めてロードする。
 */
async function ensureDavid(): Promise<void> {
  if (!sceneState || davidAcquisitionBusy) return;
  const s = sceneState;
  const m = s.ctx.measured;
  if (!m) return;
  const need =
    (params.depthSource === 'DAVID' && !m.davidDepth) ||
    (params.normalSource === 'DAVID' && !m.davidNormalCanvas) ||
    (params.personSource === 'DAVID' && !m.davidPerson);
  if (!need) return;

  davidAcquisitionBusy = true;
  try {
    setStatus('DAViDでDepth/法線/前景を推定しています… (初回はモデルDLで時間がかかります)');
    const mod = await import('./david');
    davidEstimator ??= new mod.DavidEstimator();
    await davidEstimator.init();
    const t0 = performance.now();
    const result = await davidEstimator.estimate(
      s.sourceCanvas,
      s.normalized.headCenterPx,
      s.normalized.faceWidth,
      m.segmentation?.person ?? null,
    );
    m.davidDepth = result.depth;
    m.davidNormalCanvas = result.normalCanvas;
    m.davidPerson = result.person;
    console.debug(`DAViD multi-task推定: ${(performance.now() - t0).toFixed(0)}ms`);
    await rebuildGnmHead();
    if (!els.status.classList.contains('error')) setStatus('');
  } catch (err) {
    console.error('DAViDの取得に失敗しました。', err);
    setStatus('DAViDの取得に失敗しました。ARPortraitDepth等へフォールバックして表示しています。', true);
  } finally {
    davidAcquisitionBusy = false;
  }
}

/**
 * GNM頭部を(再)構築してビューポートへ載せる。
 * アセット(gnm_head_lite.bin 約8.5MB)は初回のみロードする。
 */
async function rebuildGnmHead(): Promise<void> {
  if (!sceneState || gnmBusy) return;
  const s = sceneState;

  gnmBusy = true;
  try {
    if (!gnmModel) {
      setStatus('GNM Headアセットを読み込んでいます…');
      gnmModel = await loadGnmModel();
    }
    if (!exprSampler) {
      setStatus('表情サンプラーを読み込んでいます…');
      exprSampler = await loadExpressionSampler();
      sampleSumCache.clear();
      refreshEmotionOptions(gui, params, exprSampler.classNames);
    }
    setStatus('GNM Headをフィットしています…');
    s.gnmHead?.dispose();
    s.gnmHead = null; // 構築失敗時にdispose済みの旧buildが残らないように

    const normalized = normalizeFaceLandmarks(s.rawLandmarks, s.ctx.imageWidth, s.ctx.imageHeight);
    s.normalized = normalized;
    s.ctx.landmarks = normalized.landmarks;
    s.ctx.headCenterPx = normalized.headCenterPx;
    s.ctx.faceWidthPx = normalized.faceWidth;

    const build = buildGnmHead(gnmModel, s.ctx, s.sourceCanvas, s.texture, params, {
      blink: buildBlinkVector(gnmModel),
    });
    build.group.rotation.order = 'YXZ'; // yaw→pitchの順で直感的に合成されるよう明示する
    s.gnmHead = build;
    // デバッグ用: コンソールから表情係数や対応残差を直接調べられるようにする
    (window as unknown as Record<string, unknown>).__gnmHead = build;
    (window as unknown as Record<string, unknown>).__gnmDebug = { model: gnmModel, ctx: s.ctx, build, params };
    viewport.setGroup(build.group);
    updateYaw(yawDeg);
    updatePitch(pitchDeg);
    applyDebugVisualization(build, params);
    if (!build.hairMesh && params.gnmShowHair) {
      setStatus('GNM: 髪シェルを構築できなかったため頭部のみ表示しています。');
    } else {
      setStatus('');
    }
  } catch (err) {
    console.error('GNM Headの構築に失敗しました。', err);
    setStatus('GNM Headの構築に失敗しました。', true);
  } finally {
    gnmBusy = false;
    updateExportButton();
  }
}

async function processImage(captured: CapturedImage): Promise<void> {
  setStatus('顔を検出しています…');
  try {
    await faceDetector.init();
    const rawLandmarks = faceDetector.detect(captured.canvas);
    const normalized = normalizeFaceLandmarks(rawLandmarks, captured.width, captured.height);

    const measured = await acquireMeasuredData(captured, normalized);

    const texture = new THREE.CanvasTexture(captured.canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true;

    disposeSceneState();

    sceneState = {
      ctx: {
        landmarks: normalized.landmarks,
        headCenterPx: normalized.headCenterPx,
        faceWidthPx: normalized.faceWidth,
        imageWidth: captured.width,
        imageHeight: captured.height,
        measured,
      },
      sourceCanvas: captured.canvas,
      normalized,
      rawLandmarks,
      gnmHead: null,
      texture,
    };

    updateCameras();
    updateYaw(0);
    updatePitch(0);

    await rebuildGnmHead();

    // DAVIDが選択済みの状態で新しい画像が来た場合は遅延取得を開始する
    void ensureDavid();
  } catch (err) {
    if (err instanceof FaceDetectionError) {
      setStatus(err.message, true);
    } else {
      console.error(err);
      setStatus('画像の処理中にエラーが発生しました。', true);
    }
  }
}

function updateCameras(): void {
  viewport.updateCamera(params.cameraFovDeg, params.cameraDistanceRatio);
}

// --- Toolbar actions ---

els.btnUpload.addEventListener('click', () => els.fileInput.click());
els.fileInput.addEventListener('change', async () => {
  const file = els.fileInput.files?.[0];
  if (!file) return;
  if (inputManager.isWebcamActive) {
    inputManager.stopWebcam();
    els.btnWebcam.textContent = 'Webcam';
  }
  const captured = await inputManager.loadFromFile(file);
  await processImage(captured);
  els.fileInput.value = '';
});

els.btnWebcam.addEventListener('click', async () => {
  try {
    if (!inputManager.isWebcamActive) {
      setStatus('Webcamを起動しています…');
      await inputManager.startWebcam();
      els.btnWebcam.textContent = 'Capture';
      setStatus('正面を向いてCaptureを押してください。');
    } else {
      const captured = inputManager.captureWebcamFrame();
      inputManager.stopWebcam();
      els.btnWebcam.textContent = 'Webcam';
      await processImage(captured);
    }
  } catch (err) {
    console.error(err);
    setStatus('Webcamにアクセスできませんでした。', true);
  }
});

els.btnReset.addEventListener('click', () => {
  inputManager.stopWebcam();
  els.btnWebcam.textContent = 'Webcam';
  disposeSceneState();
  updateYaw(0);
  updatePitch(0);
  setStatus('');
});

els.toggleBlink.addEventListener('change', () => {
  params.blinkEnabled = els.toggleBlink.checked;
});

// --- Unityエクスポート (本番構成: Guest=毎回 / Template=1回。docs/unity_integration.md参照) ---
let exportBusy = false;

/** blobをファイルとしてダウンロードし、完了メッセージを出す。 */
function downloadBlob(blob: Blob, filename: string): void {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
  setStatus(`エクスポート完了: ${filename} (${(blob.size / 1024 / 1024).toFixed(1)}MB)`);
}

async function runExport(kind: 'guest' | 'template'): Promise<void> {
  const s = sceneState;
  if (exportBusy || !s?.gnmHead || !gnmModel) return;
  exportBusy = true;
  updateExportButton();
  try {
    setStatus('Unity向けパッケージを書き出しています…');
    // GLTFExporterごと遅延ロードする (エクスポートしない起動では読まない)
    const mod = await import('./unityExport');
    if (kind === 'guest') {
      const blob = await mod.exportUnityGuest({
        model: gnmModel,
        build: s.gnmHead,
        ctx: s.ctx,
        sourceCanvas: s.sourceCanvas,
        params,
      });
      const stamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14); // YYYYMMDDhhmmss
      downloadBlob(blob, `gnm_head_guest_${stamp}.zip`);
    } else {
      const blob = await mod.exportUnityTemplate({
        model: gnmModel,
        blinkVector: buildBlinkVector(gnmModel),
        params,
        autoCycle: GNM_AUTO_CYCLE,
      });
      // テンプレートはゲスト非依存なので固定名 (更新はGNMアセットを変えたときだけ)
      downloadBlob(blob, 'gnm_unity_template.zip');
    }
  } catch (err) {
    console.error('Unityエクスポートに失敗しました。', err);
    setStatus('Unityエクスポートに失敗しました。', true);
  } finally {
    exportBusy = false;
    updateExportButton();
  }
}

els.btnExport.addEventListener('click', () => void runExport('guest'));
els.btnExportTemplate.addEventListener('click', () => void runExport('template'));

window.addEventListener('resize', () => viewport.resize());

// --- GUIパラメータパネル ---
const gui = setupDebugGui(els.guiContainer, params, {
  onSourceChanged: () => {
    // まず取得済みソースで即時再構築し、DAVIDが未取得なら裏で取得して再構築する
    void rebuildGnmHead().then(() => ensureDavid());
  },
  onGnmParamsChanged: () => {
    void rebuildGnmHead();
  },
  onCameraChanged: () => updateCameras(),
  onBackgroundChanged: () => viewport.setBackground(params.backgroundColor),
  onYawRangeChanged: () => updateYaw(yawDeg),
  onPitchRangeChanged: () => updatePitch(pitchDeg),
  getGnmHead: () => sceneState?.gnmHead ?? null,
});

// --- レンダーループ ---
function animate(): void {
  requestAnimationFrame(animate);
  const now = performance.now();

  const gnmHead = sceneState?.gnmHead;
  if (gnmHead) {
    const blinkAmount = params.blinkEnabled ? updateBlink(now, blinkState, params) : 0;

    // GNM表情: 感情プリセットを目標に設定し、tickExpressionの指数遷移で滑らかに繋ぐ
    let preset: number[] | null = null;
    if (params.gnmEmotion === 'AUTO' || params.gnmEmotion === 'RANDOM') {
      if (now >= gnmExprNextChangeAt) {
        if (gnmAutoTarget) {
          // 感情の保持が終わったら短いニュートラル区間を挟む
          gnmAutoTarget = null;
          gnmExprNextChangeAt = now + GNM_AUTO_CYCLE.neutralMinMs + Math.random() * GNM_AUTO_CYCLE.neutralRandMs;
        } else {
          if (params.gnmEmotion === 'RANDOM') {
            // 公式 randomize_expressions: 2〜3クラスをランダムに選んで公式blendする
            gnmAutoTarget =
              exprSampler && gnmModel
                ? exprSampler.toModelCoeffs(exprSampler.randomize(Math.random), gnmModel)
                : null;
          } else {
            // 公式Expressionクラスを巡回。潜在zも引き直すので同じクラスでも毎回変わる
            const classes = (exprSampler?.classNames ?? []).filter((c) => c !== gnmLastEmotion);
            const cls = classes[Math.floor(Math.random() * classes.length)];
            gnmLastEmotion = cls;
            const latent = exprSampler?.randomLatent(Math.random) ?? null;
            gnmAutoTarget = cls ? sampleClass(cls, latent) : null;
          }
          gnmExprNextChangeAt = now + GNM_AUTO_CYCLE.holdMinMs + Math.random() * GNM_AUTO_CYCLE.holdRandMs;
        }
      }
      preset = gnmAutoTarget;
    } else if (params.gnmEmotion === 'MANUAL') {
      // パーツ別スライダーの合成。公式クラスの代表表情を目/下顔面領域に分けて加算する
      // (領域分割は公式に無い操作なのでMANUALモード限定)
      const region = gnmModel ? expressionRegions(gnmModel) : [];
      let vec: number[] | null = null;
      for (const c of GNM_MANUAL_CONTROLS) {
        const amount = params[c.param] as number;
        if (amount === 0) continue;
        const p = sampleClassSum(c.classes);
        if (!p) continue;
        vec ??= new Array<number>(p.length).fill(0);
        for (let i = 0; i < p.length; i++) {
          if (region[i] === c.region) vec[i] += p[i] * amount;
        }
      }
      preset = vec;
    } else if (params.gnmEmotion !== 'NEUTRAL') {
      // 感情固定: 値は公式Expressionクラス名そのもの
      preset = sampleClassSum([params.gnmEmotion]);
    }
    if (preset) {
      gnmHead.setExpressionTarget(preset.map((v) => v * params.gnmExprIntensity));
    } else {
      gnmHead.setNeutralExpression();
    }
    gnmHead.tickExpression(blinkAmount);
  }

  viewport.render();
}

updateCameras();
animate();
