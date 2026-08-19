// アプリケーションのエントリポイント。UI配線、Three.jsシーンの構築、
// 検出→メッシュ生成→アニメーションループまでを統括する。

import * as THREE from 'three';
import './style.css';
import { InputManager, type CapturedImage } from './input';
import { FaceDetectionError, FaceDetector } from './faceDetector';
import { normalizeFaceLandmarks, triangulateFaceLandmarks, type NormalizedFaceResult } from './faceTopology';
import { buildCanonicalFaceDepth, buildFaceDepthField, computeFinalFaceDepthPerVertex } from './faceDepth';
import { buildFaceOnlyMesh, recomputeFaceOnlyDepth, type FaceOnlyBuild } from './faceOnlyMesh';
import {
  buildHeadGridGeometry,
  recomputeFullHeadDepth,
  type FullHeadBuild,
  type FullHeadBuildContext,
  type MeasuredHeadData,
} from './fullHeadMesh';
import { PersonSegmenter } from './personSegmentation';
import { fitDepthToModelSpace, PortraitDepthEstimator } from './portraitDepth';
import {
  applyFaceOnlyBlink,
  applyFullHeadBlink,
  buildFullHeadAnimationMasks,
  createBlinkState,
  updateBlink,
  type BlinkState,
  type FullHeadAnimationMasks,
} from './animation';
import {
  applyMouthTalkDeform,
  buildMouthCavityMesh,
  buildMouthDeformTable,
  computeMouthAnchors,
  createTalkState,
  setMouthCavityDarkness,
  updateMouthCavityGeometry,
  updateTalkOpen,
  type MouthAnchors,
  type MouthCavityBuild,
  type MouthDeformEntry,
  type TalkState,
} from './mouthTalk';
import { createParams, type Params } from './params';
import { applyDebugVisualization, setupDebugGui } from './debugView';
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
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(30, 1, 0.05, 50);

    const ambient = new THREE.AmbientLight(0xffffff, 0.65);
    const key = new THREE.DirectionalLight(0xffffff, 0.9);
    key.position.set(0.6, 0.8, 1.2);
    this.scene.add(ambient, key);

    this.resize();
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

  updateCamera(fovDeg: number, distance: number): void {
    this.camera.fov = fovDeg;
    this.camera.position.set(0, 0, distance);
    this.camera.lookAt(0, 0, 0);
    this.camera.updateProjectionMatrix();
  }

  resize(): void {
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
  ctx: FullHeadBuildContext;
  sourceCanvas: HTMLCanvasElement; // NEURAL系の遅延推論で再利用する入力画像
  normalized: NormalizedFaceResult;
  faceOnly: FaceOnlyBuild;
  fullHead: FullHeadBuild;
  gnmHead: import('./gnmHeadMesh').GnmHeadBuild | null; // GNMバックエンド (遅延構築)
  fullHeadMasks: FullHeadAnimationMasks;
  texture: THREE.Texture;
  mouthAnchors: MouthAnchors;
  faceOnlyMouthTable: MouthDeformEntry[];
  fullHeadMouthTable: MouthDeformEntry[];
  faceOnlyCavity: MouthCavityBuild;
  fullHeadCavity: MouthCavityBuild;
}

const params: Params = createParams();

const els = {
  btnWebcam: document.getElementById('btn-webcam') as HTMLButtonElement,
  btnUpload: document.getElementById('btn-upload') as HTMLButtonElement,
  btnReset: document.getElementById('btn-reset') as HTMLButtonElement,
  fileInput: document.getElementById('file-input') as HTMLInputElement,
  status: document.getElementById('status-message') as HTMLElement,
  paneFaceOnly: document.getElementById('canvas-face-only') as HTMLElement,
  paneFullHead: document.getElementById('canvas-full-head') as HTMLElement,
  paneFaceOnlyRoot: document.getElementById('pane-face-only') as HTMLElement,
  paneFullHeadRoot: document.getElementById('pane-full-head') as HTMLElement,
  yawReadout: document.getElementById('readout-yaw') as HTMLElement,
  pitchReadout: document.getElementById('readout-pitch') as HTMLElement,
  toggleBlink: document.getElementById('toggle-blink') as HTMLInputElement,
  toggleTalk: document.getElementById('toggle-talk') as HTMLInputElement,
  selectHeadMode: document.getElementById('select-head-mode') as HTMLSelectElement,
  video: document.getElementById('webcam-video') as HTMLVideoElement,
  guiContainer: document.getElementById('gui-container') as HTMLElement,
};

const faceOnlyViewport = new Viewport(els.paneFaceOnly);
const fullHeadViewport = new Viewport(els.paneFullHead);

const inputManager = new InputManager(els.video);
const faceDetector = new FaceDetector();
const personSegmenter = new PersonSegmenter();
const portraitDepth = new PortraitDepthEstimator();

// NEURAL系 (transformers.js) はモデル・ランタイムとも大きいため、
// ソースとして選択されて初めてdynamic importする。
let neuralModule: typeof import('./neuralSources') | null = null;
let neuralDepthEst: import('./neuralSources').NeuralDepthEstimator | null = null;
let neuralMatteEst: import('./neuralSources').NeuralMatteEstimator | null = null;

// GNM Headバックエンドはアセット(約5MB)ごと遅延ロードする。
let gnmMeshModule: typeof import('./gnmHeadMesh') | null = null;
let gnmModel: import('./gnmHead').GnmModel | null = null;
let gnmBusy = false;

let sceneState: SceneState | null = null;
let yawDeg = 0;
let pitchDeg = 0;
let blinkState: BlinkState = createBlinkState(performance.now(), params);
let talkState: TalkState = createTalkState(performance.now());

function setStatus(message: string, isError = false): void {
  els.status.textContent = message;
  els.status.classList.toggle('error', isError);
}

function currentMaxYaw(): number {
  return params.maxYawDeg;
}

function currentMaxPitch(): number {
  return params.maxPitchDeg;
}

/** Yaw/Pitch角を更新し、FACE ONLY/FULL HEAD両方のGroupへ同期反映する。 */
function updateYaw(deg: number): void {
  yawDeg = Math.min(currentMaxYaw(), Math.max(-currentMaxYaw(), deg));
  updateOrientationReadout();
  if (sceneState) {
    sceneState.faceOnly.group.rotation.y = THREE.MathUtils.degToRad(yawDeg);
    sceneState.fullHead.group.rotation.y = THREE.MathUtils.degToRad(yawDeg);
    if (sceneState.gnmHead) sceneState.gnmHead.group.rotation.y = THREE.MathUtils.degToRad(yawDeg);
  }
}

function updatePitch(deg: number): void {
  pitchDeg = Math.min(currentMaxPitch(), Math.max(-currentMaxPitch(), deg));
  updateOrientationReadout();
  if (sceneState) {
    sceneState.faceOnly.group.rotation.x = THREE.MathUtils.degToRad(pitchDeg);
    sceneState.fullHead.group.rotation.x = THREE.MathUtils.degToRad(pitchDeg);
    if (sceneState.gnmHead) sceneState.gnmHead.group.rotation.x = THREE.MathUtils.degToRad(pitchDeg);
  }
}

function updateOrientationReadout(): void {
  els.yawReadout.textContent = yawDeg.toFixed(1);
  els.pitchReadout.textContent = pitchDeg.toFixed(1);
}

new OrbitDragController([els.paneFaceOnly, els.paneFullHead], {
  getYaw: () => yawDeg,
  setYaw: (v) => updateYaw(v),
  getMaxYawDeg: () => currentMaxYaw(),
  getPitch: () => pitchDeg,
  setPitch: (v) => updatePitch(v),
  getMaxPitchDeg: () => currentMaxPitch(),
});

function disposeSceneState(): void {
  if (!sceneState) return;
  faceOnlyViewport.clear();
  fullHeadViewport.clear();
  sceneState.faceOnly.geometry.dispose();
  sceneState.fullHead.geometry.dispose();
  (sceneState.faceOnly.mesh.material as THREE.Material).dispose();
  const fullHeadMat = sceneState.fullHead.mesh.material as THREE.MeshStandardMaterial;
  fullHeadMat.alphaMap?.dispose();
  fullHeadMat.dispose();
  sceneState.faceOnlyCavity.geometry.dispose();
  sceneState.fullHeadCavity.geometry.dispose();
  (sceneState.faceOnlyCavity.mesh.material as THREE.Material).dispose();
  (sceneState.fullHeadCavity.mesh.material as THREE.Material).dispose();
  sceneState.gnmHead?.dispose();
  sceneState.texture.dispose();
  sceneState = null;
}

/** FACE ONLY / FULL HEAD 双方の口周辺頂点にMouth Cavityメッシュを追加し、Pivotへ揃える。 */
function attachMouthCavity(build: MouthCavityBuild, parentGroup: THREE.Group): void {
  build.mesh.position.z = -params.pivotZRatio;
  parentGroup.add(build.mesh);
  setMouthCavityDarkness(build, params.mouthCavityDarkness);
}

/**
 * 実測ソース (セグメンテーション+Depth) を取得する。
 * 失敗した項目はnullとし、fullHeadMeshが自動的にヒューリスティックへフォールバックする。
 */
async function acquireMeasuredData(
  captured: CapturedImage,
  normalized: NormalizedFaceResult,
  faceZFinal: Float32Array,
): Promise<MeasuredHeadData> {
  const measured: MeasuredHeadData = {
    segmentation: null,
    depth: null,
    depthFit: null,
    neuralSegmentation: null,
    neuralDepth: null,
    neuralDepthFit: null,
  };

  try {
    setStatus('頭部をセグメントしています…');
    await personSegmenter.init();
    measured.segmentation = personSegmenter.segment(captured.canvas, normalized.landmarks);
  } catch (err) {
    console.warn('セグメンテーションに失敗。楕円マスクへフォールバックします。', err);
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
      measured.depthFit = fitDepthToModelSpace(measured.depth, normalized.landmarks, faceZFinal);
      if (!measured.depthFit) {
        console.warn('Depthフィットに失敗。ヒューリスティックDepthへフォールバックします。');
        measured.depth = null;
      }
    } catch (err) {
      console.warn('Depth推定に失敗。ヒューリスティックDepthへフォールバックします。', err);
      measured.depth = null;
      measured.depthFit = null;
    }
  }

  return measured;
}

let neuralAcquisitionBusy = false;

/**
 * NEURALソース (BiRefNetマット / Depth Anything V2) を必要時に遅延取得する。
 * モデルDLが大きい(数十〜数百MB)ため、ソースとして選択されて初めてロードする。
 * 取得済みならそのまま、未取得なら推論してctx.measuredへ格納し、FULL HEADを再構築する。
 */
async function ensureNeuralSources(): Promise<void> {
  if (!sceneState || neuralAcquisitionBusy) return;
  const s = sceneState;
  const m = s.ctx.measured;
  if (!m) return;

  const needMatte = params.maskSource === 'NEURAL' && !m.neuralSegmentation;
  const needDepth = params.depthSource === 'NEURAL' && !m.neuralDepth;
  if (!needMatte && !needDepth) return;

  neuralAcquisitionBusy = true;
  try {
    const mod = (neuralModule ??= await import('./neuralSources'));

    if (needMatte) {
      if (!m.segmentation) {
        setStatus('NEURALマスクにはMediaPipeセグメンテーションが必要です (意味分けに使用)。', true);
      } else {
        setStatus('BiRefNetでマットを推定しています… (初回はモデルDLで時間がかかります)');
        neuralMatteEst ??= new mod.NeuralMatteEstimator();
        await neuralMatteEst.init();
        const matte = await neuralMatteEst.estimate(s.sourceCanvas);
        m.neuralSegmentation = mod.refineSegmentationWithMatte(m.segmentation, matte);
      }
    }

    if (needDepth) {
      const personMask = m.neuralSegmentation?.person ?? m.segmentation?.person ?? null;
      if (!personMask) {
        setStatus('NEURAL Depthには人物マスクが必要です (セグメンテーション取得失敗)。', true);
      } else {
        setStatus('Depth Anything V2でDepthを推定しています… (初回はモデルDLで時間がかかります)');
        neuralDepthEst ??= new mod.NeuralDepthEstimator();
        await neuralDepthEst.init();
        const field = await neuralDepthEst.estimate(
          s.sourceCanvas,
          personMask,
          s.normalized.headCenterPx,
          s.normalized.faceWidth,
        );
        const fit = fitDepthToModelSpace(field, s.normalized.landmarks, s.ctx.faceZFinal);
        if (fit) {
          m.neuralDepth = field;
          m.neuralDepthFit = fit;
        } else {
          setStatus('NEURAL Depthのフィットに失敗しました。MEASUREDへフォールバックします。', true);
        }
      }
    }

    rebuildFullHead();
    if (!els.status.classList.contains('error')) setStatus('');
  } catch (err) {
    console.error('NEURALソースの取得に失敗しました。', err);
    setStatus('NEURALソースの取得に失敗しました。フォールバックで表示しています。', true);
    rebuildFullHead();
  } finally {
    neuralAcquisitionBusy = false;
  }
}

/** headBackendに応じてFULL HEADビューへ表示するgroupを差し替える。 */
function applyHeadBackend(): void {
  if (!sceneState) return;
  if (params.headBackend === 'GNM' && sceneState.gnmHead) {
    fullHeadViewport.setGroup(sceneState.gnmHead.group);
  } else {
    fullHeadViewport.setGroup(sceneState.fullHead.group);
  }
  updateYaw(yawDeg);
  updatePitch(pitchDeg);
}

/**
 * GNMバックエンドを必要時に遅延構築する。
 * アセット(gnm_head_lite.bin 約5MB)とモジュールは初回のみロードし、
 * 画像・パラメータが変わったときはrebuild=trueで作り直す。
 */
async function ensureGnmHead(rebuild = false): Promise<void> {
  if (!sceneState || gnmBusy) return;
  if (params.headBackend !== 'GNM') return;
  const s = sceneState;
  if (s.gnmHead && !rebuild) {
    applyHeadBackend();
    return;
  }

  gnmBusy = true;
  try {
    if (!gnmModel) setStatus('GNM Headアセットを読み込んでいます…');
    gnmMeshModule ??= await import('./gnmHeadMesh');
    if (!gnmModel) {
      const headModule = await import('./gnmHead');
      gnmModel = await headModule.loadGnmModel();
    }
    setStatus('GNM Headをフィットしています…');
    s.gnmHead?.dispose();
    const build = gnmMeshModule.buildGnmHead(gnmModel, s.ctx, s.sourceCanvas, s.texture, params);
    build.group.rotation.order = 'YXZ';
    s.gnmHead = build;
    applyHeadBackend();
    applyDebugVisualization(sceneState, params);
    if (!build.hairMesh) {
      setStatus('GNM: 髪シェルを構築できなかったため頭部のみ表示しています。');
    } else {
      setStatus('');
    }
  } catch (err) {
    console.error('GNM Headの構築に失敗しました。', err);
    setStatus('GNM Headの構築に失敗しました。GRIDバックエンドで表示しています。', true);
    params.headBackend = 'GRID';
    applyHeadBackend();
  } finally {
    gnmBusy = false;
  }
}

async function processImage(captured: CapturedImage): Promise<void> {
  setStatus('顔を検出しています…');
  try {
    await faceDetector.init();
    const rawLandmarks = faceDetector.detect(captured.canvas);

    const normalized = normalizeFaceLandmarks(rawLandmarks, captured.width, captured.height);
    const triangulation = triangulateFaceLandmarks(normalized.landmarks);
    const canonicalDepth = buildCanonicalFaceDepth(normalized.landmarks);
    const faceZFinal = computeFinalFaceDepthPerVertex(
      normalized.landmarks,
      canonicalDepth,
      params.canonicalMix,
      params.faceDepthScale,
    );
    const depthField = buildFaceDepthField(
      normalized.landmarks,
      faceZFinal,
      triangulation,
      captured.width,
      captured.height,
      params.faceDepthFieldSize,
    );

    const measured = await acquireMeasuredData(captured, normalized, faceZFinal);

    const texture = new THREE.CanvasTexture(captured.canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true;

    disposeSceneState();

    const faceOnly = buildFaceOnlyMesh(normalized.landmarks, triangulation, texture, params);

    const ctx: FullHeadBuildContext = {
      landmarks: normalized.landmarks,
      triangulation,
      faceZFinal,
      depthField,
      headCenterPx: normalized.headCenterPx,
      faceWidthPx: normalized.faceWidth,
      imageWidth: captured.width,
      imageHeight: captured.height,
      measured,
    };
    const fullHead = buildHeadGridGeometry(ctx, texture, params);
    const fullHeadMasks = buildFullHeadAnimationMasks(fullHead, normalized.landmarks);

    const mouthAnchors = computeMouthAnchors(normalized.landmarks, (i) => faceZFinal[i]);
    const faceOnlyMouthTable = buildMouthDeformTable(
      normalized.landmarks.length,
      (i) => normalized.landmarks[i],
      mouthAnchors,
    );
    const fullHeadCount = fullHead.cols * fullHead.rows;
    const fullHeadMouthTable = buildMouthDeformTable(
      fullHeadCount,
      (i) => ({ x: fullHead.basePositions[i * 3], y: fullHead.basePositions[i * 3 + 1] }),
      mouthAnchors,
    );
    const faceOnlyCavity = buildMouthCavityMesh();
    const fullHeadCavity = buildMouthCavityMesh();

    sceneState = {
      ctx,
      sourceCanvas: captured.canvas,
      normalized,
      faceOnly,
      fullHead,
      gnmHead: null,
      fullHeadMasks,
      texture,
      mouthAnchors,
      faceOnlyMouthTable,
      fullHeadMouthTable,
      faceOnlyCavity,
      fullHeadCavity,
    };

    // yaw→pitchの順で直感的に合成されるよう回転順序を明示する。
    faceOnly.group.rotation.order = 'YXZ';
    fullHead.group.rotation.order = 'YXZ';

    faceOnlyViewport.setGroup(faceOnly.group);
    fullHeadViewport.setGroup(fullHead.group);
    attachMouthCavity(faceOnlyCavity, faceOnly.group);
    attachMouthCavity(fullHeadCavity, fullHead.group);
    updateCameras();
    updateYaw(0);
    updatePitch(0);
    applyDebugVisualization(sceneState, params);

    if (!measured.segmentation && !measured.depth) {
      setStatus('実測ソースを取得できず、ヒューリスティック方式で表示しています。', true);
    } else if (!measured.depth) {
      setStatus('Depth計測に失敗したため、シルエットのみ実測を使用しています。');
    } else {
      setStatus('');
    }

    // NEURAL/GNMが選択済みの状態で新しい画像が来た場合は遅延取得・構築を開始する
    void ensureNeuralSources();
    void ensureGnmHead();
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
  const distance = params.cameraDistanceRatio;
  faceOnlyViewport.updateCamera(params.cameraFovDeg, distance);
  fullHeadViewport.updateCamera(params.cameraFovDeg, distance);
}

/**
 * Mask/Depthソース切替時にFULL HEADメッシュを再構築する。
 * grid境界・UV・alphaMapがソースに依存するため、Depth再計算では足りない。
 */
function rebuildFullHead(): void {
  if (!sceneState) return;
  const s = sceneState;
  fullHeadViewport.clear();
  s.fullHead.geometry.dispose();
  const oldMat = s.fullHead.mesh.material as THREE.MeshStandardMaterial;
  oldMat.alphaMap?.dispose();
  oldMat.dispose();

  const fullHead = buildHeadGridGeometry(s.ctx, s.texture, params);
  fullHead.group.rotation.order = 'YXZ';
  s.fullHead = fullHead;
  s.fullHeadMasks = buildFullHeadAnimationMasks(fullHead, s.normalized.landmarks);
  s.fullHeadMouthTable = buildMouthDeformTable(
    fullHead.cols * fullHead.rows,
    (i) => ({ x: fullHead.basePositions[i * 3], y: fullHead.basePositions[i * 3 + 1] }),
    s.mouthAnchors,
  );
  fullHeadViewport.setGroup(fullHead.group);
  attachMouthCavity(s.fullHeadCavity, fullHead.group);
  updateYaw(yawDeg);
  updatePitch(pitchDeg);
  applyDebugVisualization(sceneState, params);

  // GNM表示中はソース変更が髪シェル/テクスチャゲートにも効くため作り直す
  if (params.headBackend === 'GNM') {
    applyHeadBackend();
    void ensureGnmHead(true);
  }
}

/** Depth系GUIパラメータ変更時、landmark再検出なしでgeometry/mouthAnchorsのZのみ再計算する。 */
export function rebuildDepthOnly(): void {
  if (!sceneState) return;
  recomputeFaceOnlyDepth(sceneState.faceOnly, params);
  recomputeFullHeadDepth(sceneState.fullHead, sceneState.ctx, params);
  sceneState.fullHeadMasks = buildFullHeadAnimationMasks(sceneState.fullHead, sceneState.normalized.landmarks);

  // Mouth SeamのZは顔Depthに追従させる (X/Yはlandmarkから不変なのでtableは再利用でよい)。
  const posAttr = sceneState.faceOnly.geometry.getAttribute('position') as THREE.BufferAttribute;
  sceneState.mouthAnchors = computeMouthAnchors(sceneState.normalized.landmarks, (i) => posAttr.getZ(i));
  updateCameras();
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
els.toggleTalk.addEventListener('change', () => {
  params.talkEnabled = els.toggleTalk.checked;
  syncTalkEnabledControllers();
});
els.selectHeadMode.addEventListener('change', () => {
  params.fullHeadMode = els.selectHeadMode.value as Params['fullHeadMode'];
  rebuildDepthOnly();
  applyDebugVisualization(sceneState, params);
});

window.addEventListener('resize', () => {
  faceOnlyViewport.resize();
  fullHeadViewport.resize();
});

// --- GUIパラメータパネル ---
const debugGui = setupDebugGui(els.guiContainer, params, {
  onDepthParamsChanged: () => rebuildDepthOnly(),
  onSourceChanged: () => {
    // まず取得済みソースで即時再構築し、NEURAL系が未取得なら裏で取得して再構築する
    rebuildFullHead();
    void ensureNeuralSources();
  },
  onBackendChanged: () => {
    applyHeadBackend();
    void ensureGnmHead();
  },
  onGnmParamsChanged: () => {
    void ensureGnmHead(true);
  },
  onYawRangeChanged: () => updateYaw(yawDeg),
  onPitchRangeChanged: () => updatePitch(pitchDeg),
  getSceneState: () => sceneState,
  onTalkEnabledChangedFromGui: (value) => {
    els.toggleTalk.checked = value;
  },
  onMouthCavityDarknessChanged: () => {
    if (!sceneState) return;
    setMouthCavityDarkness(sceneState.faceOnlyCavity, params.mouthCavityDarkness);
    setMouthCavityDarkness(sceneState.fullHeadCavity, params.mouthCavityDarkness);
  },
});

function syncTalkEnabledControllers(): void {
  debugGui.syncTalkEnabled();
}

// --- レンダーループ ---
function animate(): void {
  requestAnimationFrame(animate);
  const now = performance.now();

  if (sceneState) {
    const blinkAmount = params.blinkEnabled ? updateBlink(now, blinkState, params) : 0;

    let talkOpen: number;
    if (params.talkManualOverride) {
      talkOpen = params.talkOpenManual;
    } else if (params.talkEnabled) {
      talkOpen = updateTalkOpen(now, talkState);
    } else {
      talkOpen = 0;
    }

    applyFaceOnlyBlink(sceneState.faceOnly, blinkAmount);
    applyFullHeadBlink(sceneState.fullHead, sceneState.fullHeadMasks, blinkAmount);

    const faceOnlyPosAttr = sceneState.faceOnly.geometry.getAttribute('position') as THREE.BufferAttribute;
    const fullHeadPosAttr = sceneState.fullHead.geometry.getAttribute('position') as THREE.BufferAttribute;
    applyMouthTalkDeform(
      faceOnlyPosAttr,
      sceneState.faceOnly.basePositions,
      sceneState.faceOnlyMouthTable,
      talkOpen,
      sceneState.mouthAnchors,
      params,
    );
    applyMouthTalkDeform(
      fullHeadPosAttr,
      sceneState.fullHead.basePositions,
      sceneState.fullHeadMouthTable,
      talkOpen,
      sceneState.mouthAnchors,
      params,
    );
    faceOnlyPosAttr.needsUpdate = true;
    fullHeadPosAttr.needsUpdate = true;

    updateMouthCavityGeometry(sceneState.faceOnlyCavity, sceneState.mouthAnchors, talkOpen, params);
    updateMouthCavityGeometry(sceneState.fullHeadCavity, sceneState.mouthAnchors, talkOpen, params);
  }

  faceOnlyViewport.render();
  fullHeadViewport.render();
}

updateCameras();
animate();
