// アプリケーションのエントリポイント。UI配線、Three.jsシーンの構築、
// 検出→メッシュ生成→アニメーションループまでを統括する。

import * as THREE from 'three';
import './style.css';
import { InputManager, type CapturedImage } from './input';
import { FaceDetectionError, FaceDetector } from './faceDetector';
import { normalizeFaceLandmarks, triangulateFaceLandmarks, type NormalizedFaceResult } from './faceTopology';
import { buildCanonicalFaceDepth, buildFaceDepthField, computeFinalFaceDepthPerVertex } from './faceDepth';
import { buildFaceOnlyMesh, recomputeFaceOnlyDepth, type FaceOnlyBuild } from './faceOnlyMesh';
import { buildHeadGridGeometry, recomputeFullHeadDepth, type FullHeadBuild, type FullHeadBuildContext } from './fullHeadMesh';
import {
  applyBlinkToFaceOnly,
  applyBlinkToFullHead,
  buildEyeAnchorPair,
  buildEyeDeformTable,
  createBlinkState,
  updateBlinkAmount,
  type BlinkState,
  type EyeAnchors,
  type EyeDeformEntry,
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
import { applyDebugVisualization, setupDebugGui, updateBlinkDebugOverlays } from './debugView';
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
  normalized: NormalizedFaceResult;
  faceOnly: FaceOnlyBuild;
  fullHead: FullHeadBuild;
  texture: THREE.Texture;
  mouthAnchors: MouthAnchors;
  faceOnlyMouthTable: MouthDeformEntry[];
  fullHeadMouthTable: MouthDeformEntry[];
  faceOnlyCavity: MouthCavityBuild;
  fullHeadCavity: MouthCavityBuild;
  eyeAnchors: [EyeAnchors, EyeAnchors];
  fullHeadEyeTable: EyeDeformEntry[];
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
  }
}

function updatePitch(deg: number): void {
  pitchDeg = Math.min(currentMaxPitch(), Math.max(-currentMaxPitch(), deg));
  updateOrientationReadout();
  if (sceneState) {
    sceneState.faceOnly.group.rotation.x = THREE.MathUtils.degToRad(pitchDeg);
    sceneState.fullHead.group.rotation.x = THREE.MathUtils.degToRad(pitchDeg);
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
  (sceneState.fullHead.mesh.material as THREE.Material).dispose();
  sceneState.faceOnlyCavity.geometry.dispose();
  sceneState.fullHeadCavity.geometry.dispose();
  (sceneState.faceOnlyCavity.mesh.material as THREE.Material).dispose();
  (sceneState.fullHeadCavity.mesh.material as THREE.Material).dispose();
  sceneState.texture.dispose();
  sceneState = null;
}

/** FACE ONLY / FULL HEAD 双方の口周辺頂点にMouth Cavityメッシュを追加し、Pivotへ揃える。 */
function attachMouthCavity(build: MouthCavityBuild, parentGroup: THREE.Group): void {
  build.mesh.position.z = -params.pivotZRatio;
  parentGroup.add(build.mesh);
  setMouthCavityDarkness(build, params.mouthCavityDarkness);
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
    };
    const fullHead = buildHeadGridGeometry(ctx, texture, params);

    const eyeAnchors = buildEyeAnchorPair(normalized.landmarks, (i) => faceZFinal[i]);
    const fullHeadEyeTable = buildEyeDeformTable(
      fullHead.cols * fullHead.rows,
      (i) => ({ x: fullHead.basePositions[i * 3], y: fullHead.basePositions[i * 3 + 1] }),
      eyeAnchors,
    );

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
      normalized,
      faceOnly,
      fullHead,
      texture,
      mouthAnchors,
      faceOnlyMouthTable,
      fullHeadMouthTable,
      faceOnlyCavity,
      fullHeadCavity,
      eyeAnchors,
      fullHeadEyeTable,
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

    setStatus('');
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

/** Depth系GUIパラメータ変更時、landmark再検出なしでgeometry/mouthAnchorsのZのみ再計算する。 */
export function rebuildDepthOnly(): void {
  if (!sceneState) return;
  recomputeFaceOnlyDepth(sceneState.faceOnly, params);
  recomputeFullHeadDepth(sceneState.fullHead, sceneState.ctx, params);

  // Mouth Seam / Eye AnchorsのZは顔Depthに追従させる (X/Yはlandmarkから不変なのでtableは再利用でよい)。
  const posAttr = sceneState.faceOnly.geometry.getAttribute('position') as THREE.BufferAttribute;
  sceneState.mouthAnchors = computeMouthAnchors(sceneState.normalized.landmarks, (i) => posAttr.getZ(i));
  sceneState.eyeAnchors = buildEyeAnchorPair(sceneState.normalized.landmarks, (i) => posAttr.getZ(i));
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
  syncBlinkEnabledControllers();
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
  onBlinkEnabledChangedFromGui: (value) => {
    els.toggleBlink.checked = value;
  },
});

function syncTalkEnabledControllers(): void {
  debugGui.syncTalkEnabled();
}

function syncBlinkEnabledControllers(): void {
  debugGui.syncBlinkEnabled();
}

// --- レンダーループ ---
function renderFrame(now: number): void {
  if (sceneState) {
    let blinkAmount: number;
    if (params.blinkManualOverride) {
      blinkAmount = params.blinkAmountManual;
    } else if (params.blinkEnabled) {
      blinkAmount = updateBlinkAmount(now, blinkState, params);
    } else {
      blinkAmount = 0;
    }

    let talkOpen: number;
    if (params.talkManualOverride) {
      talkOpen = params.talkOpenManual;
    } else if (params.talkEnabled) {
      talkOpen = updateTalkOpen(now, talkState);
    } else {
      talkOpen = 0;
    }

    applyBlinkToFaceOnly(sceneState.faceOnly, sceneState.eyeAnchors, blinkAmount, params);
    applyBlinkToFullHead(sceneState.fullHead, sceneState.fullHeadEyeTable, sceneState.eyeAnchors, blinkAmount, params);

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
    sceneState.faceOnly.geometry.computeVertexNormals();
    sceneState.fullHead.geometry.computeVertexNormals();

    updateMouthCavityGeometry(sceneState.faceOnlyCavity, sceneState.mouthAnchors, talkOpen, params);
    updateMouthCavityGeometry(sceneState.fullHeadCavity, sceneState.mouthAnchors, talkOpen, params);

    if (params.showUpperLidLine || params.showLowerLidLine || params.showBlinkTargets) {
      updateBlinkDebugOverlays(sceneState, params);
    }
  }

  faceOnlyViewport.render();
  fullHeadViewport.render();
}

function animate(): void {
  requestAnimationFrame(animate);
  renderFrame(performance.now());
}

updateCameras();
animate();
