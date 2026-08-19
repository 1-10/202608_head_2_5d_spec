// アプリケーションのエントリポイント。UI配線、Three.jsシーンの構築、
// 検出→GNM頭部構築→アニメーションループまでを統括する。

import * as THREE from 'three';
import './style.css';
import { InputManager, type CapturedImage } from './input';
import { FaceDetectionError, FaceDetector } from './faceDetector';
import { normalizeFaceLandmarks, type NormalizedFaceResult } from './faceTopology';
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
import { GNM_EXPRESSION_PRESETS } from './gnmExpressions';
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
  ctx: GnmBuildContext;
  sourceCanvas: HTMLCanvasElement; // NEURAL系の遅延推論で再利用する入力画像
  normalized: NormalizedFaceResult;
  gnmHead: GnmHeadBuild | null;
  texture: THREE.Texture;
}

const params: Params = createParams();

const els = {
  btnWebcam: document.getElementById('btn-webcam') as HTMLButtonElement,
  btnUpload: document.getElementById('btn-upload') as HTMLButtonElement,
  btnReset: document.getElementById('btn-reset') as HTMLButtonElement,
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

const inputManager = new InputManager(els.video);
const faceDetector = new FaceDetector();
const personSegmenter = new PersonSegmenter();
const portraitDepth = new PortraitDepthEstimator();

// NEURAL系 (transformers.js) はモデル・ランタイムとも大きいため、
// ソースとして選択されて初めてdynamic importする。
let neuralModule: typeof import('./neuralSources') | null = null;
let neuralDepthEst: import('./neuralSources').NeuralDepthEstimator | null = null;
let neuralMatteEst: import('./neuralSources').NeuralMatteEstimator | null = null;

// GNMアセット (gnm_head_lite.bin 約8.5MB) は初回構築時に一度だけロードする。
let gnmModel: GnmModel | null = null;
let gnmBusy = false;

let sceneState: SceneState | null = null;
let yawDeg = 0;
let pitchDeg = 0;
let blinkState: BlinkState = createBlinkState(performance.now(), params);
// --- GNM表情の自動巡回 (Emotion=AUTO時): 感情→ニュートラル→別の感情… ---
// 感情 → 公式ExpressionSamplerプリセット群 (Varはseed違いの変化形。巡回のたびに選ぶ)
const GNM_EMOTION_VARIANTS: Record<string, string[]> = {
  joy: ['happy', 'happyVar1', 'happyVar2'],
  fun: ['smileWide', 'smileWideVar1', 'smileWideVar2'],
  sad: ['cornersDown', 'cornersDownVar1', 'cornersDownVar2'],
  anger: ['snarl', 'snarlVar1', 'snarlVar2'],
  surprise: ['surprise', 'surpriseVar1', 'surpriseVar2'],
};
let gnmExprNextChangeAt = 0; // 次回遷移時刻
let gnmAutoTarget: number[] | null = null; // 現在の目標プリセット (null=ニュートラル区間)
let gnmLastEmotion = ''; // 直前の感情 (同じ感情の連続を避ける)

// Emotion=MANUAL用のパーツ別スライダー定義。公式プリセットを領域で分離して合成する。
// 領域はアセットの表情成分レイアウト (前半=目20成分, 後半=下顔面20成分) に対応
const GNM_MANUAL_CONTROLS: { param: keyof Params; preset: string; region: 'eyes' | 'lower' }[] = [
  { param: 'gnmMouthOpen', preset: 'surprise', region: 'lower' },
  { param: 'gnmSmile', preset: 'smileWide', region: 'lower' },
  { param: 'gnmPucker', preset: 'pucker', region: 'lower' },
  { param: 'gnmCornersDown', preset: 'cornersDown', region: 'lower' },
  { param: 'gnmEyesClose', preset: 'blink', region: 'eyes' },
  { param: 'gnmEyesWide', preset: 'surprise', region: 'eyes' },
  { param: 'gnmSquint', preset: 'squint', region: 'eyes' },
];

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
    depth: null,
    neuralSegmentation: null,
    neuralDepth: null,
  };

  try {
    setStatus('頭部をセグメントしています…');
    await personSegmenter.init();
    measured.segmentation = personSegmenter.segment(captured.canvas, normalized.landmarks);
  } catch (err) {
    console.warn('セグメンテーションに失敗。UVクランプ・髪シェルなしで表示します。', err);
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

let neuralAcquisitionBusy = false;

/**
 * NEURALソース (BiRefNetマット / Depth Anything V2) を必要時に遅延取得する。
 * モデルDLが大きい(数十〜数百MB)ため、ソースとして選択されて初めてロードする。
 * 取得済みならそのまま、未取得なら推論してctx.measuredへ格納し、頭部を再構築する。
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
        m.neuralDepth = await neuralDepthEst.estimate(
          s.sourceCanvas,
          personMask,
          s.normalized.headCenterPx,
          s.normalized.faceWidth,
        );
      }
    }

    await rebuildGnmHead();
    if (!els.status.classList.contains('error')) setStatus('');
  } catch (err) {
    console.error('NEURALソースの取得に失敗しました。', err);
    setStatus('NEURALソースの取得に失敗しました。フォールバックで表示しています。', true);
    await rebuildGnmHead();
  } finally {
    neuralAcquisitionBusy = false;
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
    setStatus('GNM Headをフィットしています…');
    s.gnmHead?.dispose();
    const build = buildGnmHead(gnmModel, s.ctx, s.sourceCanvas, s.texture, params);
    build.group.rotation.order = 'YXZ'; // yaw→pitchの順で直感的に合成されるよう明示する
    s.gnmHead = build;
    // デバッグ用: コンソールから表情係数や対応残差を直接調べられるようにする
    (window as unknown as Record<string, unknown>).__gnmHead = build;
    (window as unknown as Record<string, unknown>).__gnmDebug = { model: gnmModel, ctx: s.ctx, build };
    viewport.setGroup(build.group);
    updateYaw(yawDeg);
    updatePitch(pitchDeg);
    applyDebugVisualization(build, params);
    if (!build.hairMesh) {
      setStatus('GNM: 髪シェルを構築できなかったため頭部のみ表示しています。');
    } else {
      setStatus('');
    }
  } catch (err) {
    console.error('GNM Headの構築に失敗しました。', err);
    setStatus('GNM Headの構築に失敗しました。', true);
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
      gnmHead: null,
      texture,
    };

    updateCameras();
    updateYaw(0);
    updatePitch(0);

    await rebuildGnmHead();

    // NEURALが選択済みの状態で新しい画像が来た場合は遅延取得を開始する
    void ensureNeuralSources();
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

window.addEventListener('resize', () => viewport.resize());

// --- GUIパラメータパネル ---
setupDebugGui(els.guiContainer, params, {
  onSourceChanged: () => {
    // まず取得済みソースで即時再構築し、NEURAL系が未取得なら裏で取得して再構築する
    void rebuildGnmHead().then(() => ensureNeuralSources());
  },
  onGnmParamsChanged: () => {
    void rebuildGnmHead();
  },
  onCameraChanged: () => updateCameras(),
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
    if (params.gnmEmotion === 'AUTO') {
      if (now >= gnmExprNextChangeAt) {
        if (gnmAutoTarget) {
          // 感情の保持が終わったら短いニュートラル区間を挟む
          gnmAutoTarget = null;
          gnmExprNextChangeAt = now + 800 + Math.random() * 1200;
        } else {
          const emotions = Object.keys(GNM_EMOTION_VARIANTS).filter((e) => e !== gnmLastEmotion);
          const emotion = emotions[Math.floor(Math.random() * emotions.length)];
          gnmLastEmotion = emotion;
          const variants = GNM_EMOTION_VARIANTS[emotion];
          gnmAutoTarget =
            GNM_EXPRESSION_PRESETS[variants[Math.floor(Math.random() * variants.length)]] ?? null;
          gnmExprNextChangeAt = now + 2000 + Math.random() * 2500;
        }
      }
      preset = gnmAutoTarget;
    } else if (params.gnmEmotion === 'MANUAL') {
      // パーツ別スライダーの合成 (公式プリセットの目/下顔面領域を強度倍して加算)
      let vec: number[] | null = null;
      for (const c of GNM_MANUAL_CONTROLS) {
        const amount = params[c.param] as number;
        if (amount === 0) continue;
        const p = GNM_EXPRESSION_PRESETS[c.preset];
        if (!p) continue;
        vec ??= new Array<number>(p.length).fill(0);
        const half = p.length / 2;
        for (let i = 0; i < p.length; i++) {
          const inRegion = c.region === 'lower' ? i >= half : i < half;
          if (inRegion) vec[i] += p[i] * amount;
        }
      }
      preset = vec;
    } else if (params.gnmEmotion !== 'NEUTRAL') {
      preset = GNM_EXPRESSION_PRESETS[GNM_EMOTION_VARIANTS[params.gnmEmotion]?.[0] ?? ''] ?? null;
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
