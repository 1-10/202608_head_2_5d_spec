// 品質比較用GUIパラメータパネルとデバッグ表示
// (Wireframe / Landmarks / Head Mask / Face Depth / Final Depth / Mouth Seam / Mouth Region)。

import GUI, { type Controller } from 'lil-gui';
import * as THREE from 'three';
import type { Params } from './params';
import type { FaceOnlyBuild } from './faceOnlyMesh';
import type { FullHeadBuild } from './fullHeadMesh';
import type { NormalizedFaceResult } from './faceTopology';
import type { FullHeadBuildContext } from './fullHeadMesh';
import { seamPointAt, type MouthAnchors, type MouthDeformEntry } from './mouthTalk';
import { closeTargetAt, sampleCurveXYZ, type EyeAnchors, type EyeDeformEntry } from './animation';

export interface SceneStateLike {
  ctx: FullHeadBuildContext;
  normalized: NormalizedFaceResult;
  faceOnly: FaceOnlyBuild;
  fullHead: FullHeadBuild;
  texture: THREE.Texture;
  mouthAnchors: MouthAnchors;
  fullHeadMouthTable: MouthDeformEntry[];
  eyeAnchors: [EyeAnchors, EyeAnchors];
  fullHeadEyeTable: EyeDeformEntry[];
}

export interface DebugGuiOptions {
  onDepthParamsChanged: () => void;
  onSourceChanged: () => void;
  onYawRangeChanged: () => void;
  onPitchRangeChanged: () => void;
  getSceneState: () => SceneStateLike | null;
  onTalkEnabledChangedFromGui: (value: boolean) => void;
  onMouthCavityDarknessChanged: () => void;
  onBlinkEnabledChangedFromGui: (value: boolean) => void;
}

export interface DebugGuiHandle {
  gui: GUI;
  /** paramsに外部(下部パネル)から書き込まれたtalkEnabledをGUI表示へ反映する。 */
  syncTalkEnabled: () => void;
  /** paramsに外部(下部パネル)から書き込まれたblinkEnabledをGUI表示へ反映する。 */
  syncBlinkEnabled: () => void;
}

let landmarkPoints: THREE.Points | null = null;
let mouthSeamLine: THREE.Line | null = null;
let upperLidLines: THREE.Line[] = [];
let lowerLidLines: THREE.Line[] = [];
let blinkTargetLines: THREE.LineSegments | null = null;

function ensureLandmarkPoints(faceOnly: FaceOnlyBuild): THREE.Points {
  if (landmarkPoints) {
    landmarkPoints.parent?.remove(landmarkPoints);
    landmarkPoints.geometry.dispose();
    (landmarkPoints.material as THREE.Material).dispose();
    landmarkPoints = null;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', faceOnly.geometry.getAttribute('position'));
  const material = new THREE.PointsMaterial({ color: 0x4dff8d, size: 0.012, sizeAttenuation: true });
  landmarkPoints = new THREE.Points(geometry, material);
  landmarkPoints.position.copy(faceOnly.mesh.position);
  faceOnly.group.add(landmarkPoints);
  return landmarkPoints;
}

const SEAM_LINE_SEGMENTS = 24;

function ensureMouthSeamLine(faceOnly: FaceOnlyBuild, anchors: MouthAnchors): THREE.Line {
  if (mouthSeamLine) {
    mouthSeamLine.parent?.remove(mouthSeamLine);
    mouthSeamLine.geometry.dispose();
    (mouthSeamLine.material as THREE.Material).dispose();
    mouthSeamLine = null;
  }
  const positions = new Float32Array(SEAM_LINE_SEGMENTS * 3);
  for (let i = 0; i < SEAM_LINE_SEGMENTS; i++) {
    const t = i / (SEAM_LINE_SEGMENTS - 1);
    const p = seamPointAt(t, anchors);
    positions[i * 3] = p.x;
    positions[i * 3 + 1] = p.y;
    positions[i * 3 + 2] = p.z + 0.002; // 唇よりわずかに手前へ、Z-fightingを避ける
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({ color: 0x00e5ff, linewidth: 2 });
  mouthSeamLine = new THREE.Line(geometry, material);
  mouthSeamLine.position.copy(faceOnly.mesh.position);
  faceOnly.group.add(mouthSeamLine);
  return mouthSeamLine;
}

const LID_LINE_SEGMENTS = 12;

function buildLidLine(anchors: EyeAnchors, curve: 'upperCurve' | 'lowerCurve', color: number): THREE.Line {
  const positions = new Float32Array(LID_LINE_SEGMENTS * 3);
  for (let i = 0; i < LID_LINE_SEGMENTS; i++) {
    const eyeU = i / (LID_LINE_SEGMENTS - 1);
    const p = sampleCurveXYZ(anchors[curve], eyeU);
    positions[i * 3] = p.x;
    positions[i * 3 + 1] = p.y;
    positions[i * 3 + 2] = p.z + 0.0015;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({ color });
  return new THREE.Line(geometry, material);
}

function disposeLines(lines: THREE.Line[]): void {
  for (const line of lines) {
    line.parent?.remove(line);
    line.geometry.dispose();
    (line.material as THREE.Material).dispose();
  }
}

function ensureLidLines(faceOnly: FaceOnlyBuild, eyeAnchors: [EyeAnchors, EyeAnchors], which: 'upperCurve' | 'lowerCurve', color: number, store: THREE.Line[]): THREE.Line[] {
  disposeLines(store);
  const lines = eyeAnchors.map((a) => buildLidLine(a, which, color));
  for (const line of lines) {
    line.position.copy(faceOnly.mesh.position);
    faceOnly.group.add(line);
  }
  return lines;
}

/** 上瞼landmarkの現在位置からcloseTargetへの線分(×印代わりに端点マーカー付き)を構築する。 */
function buildBlinkTargetLines(faceOnly: FaceOnlyBuild, eyeAnchors: [EyeAnchors, EyeAnchors], params: Params): THREE.LineSegments {
  const posAttr = faceOnly.geometry.getAttribute('position') as THREE.BufferAttribute;
  const segments: number[] = [];
  for (const anchors of eyeAnchors) {
    for (const idx of anchors.lidIndices.upper) {
      const from = { x: posAttr.getX(idx), y: posAttr.getY(idx), z: posAttr.getZ(idx) };
      const bx = faceOnly.basePositions[idx * 3];
      const by = faceOnly.basePositions[idx * 3 + 1];
      const eyeU = ((bx - anchors.inner.x) * anchors.dirX + (by - anchors.inner.y) * anchors.dirY) / anchors.eyeWidth;
      const target = closeTargetAt(anchors, eyeU, params.blinkCloseTargetBias);
      segments.push(from.x, from.y, from.z + 0.0015, target.x, target.y, target.z + 0.0015);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(segments), 3));
  const material = new THREE.LineBasicMaterial({ color: 0xffb400 });
  const lines = new THREE.LineSegments(geometry, material);
  lines.position.copy(faceOnly.mesh.position);
  faceOnly.group.add(lines);
  return lines;
}

function heatColor(t: number): THREE.Color {
  // 0=青(奥) 〜 1=赤(手前) の簡易ヒートマップ
  const c = new THREE.Color();
  c.setHSL(THREE.MathUtils.clamp(0.66 - 0.66 * t, 0, 0.66), 0.85, 0.5);
  return c;
}

function buildDebugVertexColors(values: Float32Array): Float32Array {
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = Math.max(1e-6, max - min);
  const colors = new Float32Array(values.length * 3);
  for (let i = 0; i < values.length; i++) {
    const t = (values[i] - min) / range;
    const c = heatColor(t);
    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }
  return colors;
}

function buildMaskVertexColors(mask: Float32Array): Float32Array {
  const colors = new Float32Array(mask.length * 3);
  for (let i = 0; i < mask.length; i++) {
    colors[i * 3] = mask[i];
    colors[i * 3 + 1] = mask[i];
    colors[i * 3 + 2] = mask[i];
  }
  return colors;
}

/** Upper Lip=赤 / Lower Lip=緑 / Jaw=青 として口周辺の影響領域を可視化する。 */
function buildMouthRegionVertexColors(vertexCount: number, table: MouthDeformEntry[]): Float32Array {
  const colors = new Float32Array(vertexCount * 3).fill(0.04);
  for (const e of table) {
    colors[e.index * 3] = Math.min(1, e.upperLipW);
    colors[e.index * 3 + 1] = Math.min(1, e.lowerLipW);
    colors[e.index * 3 + 2] = Math.min(1, e.jawW);
  }
  return colors;
}

/** Upper Lid=赤 / Lower Lid=緑 / Eye Interior=青 として目周辺の影響領域を可視化する。 */
function buildEyeRegionVertexColors(vertexCount: number, table: EyeDeformEntry[]): Float32Array {
  const colors = new Float32Array(vertexCount * 3).fill(0.04);
  for (const e of table) {
    colors[e.index * 3] = Math.min(1, e.upperLidW);
    colors[e.index * 3 + 1] = Math.min(1, e.lowerLidW);
    colors[e.index * 3 + 2] = Math.min(1, e.interiorW);
  }
  return colors;
}

/** Blink関連のdebug overlay(まぶたライン・ターゲットライン)を最新のgeometry位置へ更新する。show*がOFFなら何もしない。 */
export function updateBlinkDebugOverlays(state: SceneStateLike | null, params: Params): void {
  if (!state) return;
  if (params.showUpperLidLine) {
    upperLidLines = ensureLidLines(state.faceOnly, state.eyeAnchors, 'upperCurve', 0x00e5ff, upperLidLines);
  } else if (upperLidLines.length) {
    disposeLines(upperLidLines);
    upperLidLines = [];
  }
  if (params.showLowerLidLine) {
    lowerLidLines = ensureLidLines(state.faceOnly, state.eyeAnchors, 'lowerCurve', 0xff59d3, lowerLidLines);
  } else if (lowerLidLines.length) {
    disposeLines(lowerLidLines);
    lowerLidLines = [];
  }
  if (params.showBlinkTargets) {
    if (blinkTargetLines) {
      blinkTargetLines.parent?.remove(blinkTargetLines);
      blinkTargetLines.geometry.dispose();
      (blinkTargetLines.material as THREE.Material).dispose();
    }
    blinkTargetLines = buildBlinkTargetLines(state.faceOnly, state.eyeAnchors, params);
  } else if (blinkTargetLines) {
    blinkTargetLines.parent?.remove(blinkTargetLines);
    blinkTargetLines.geometry.dispose();
    (blinkTargetLines.material as THREE.Material).dispose();
    blinkTargetLines = null;
  }
}

/** GUIのデバッグ表示チェックボックス状態を実際のThree.jsシーンへ反映する。 */
export function applyDebugVisualization(state: SceneStateLike | null, params: Params): void {
  if (!state) return;

  const foMat = state.faceOnly.mesh.material as THREE.MeshStandardMaterial;
  const fhMat = state.fullHead.mesh.material as THREE.MeshStandardMaterial;
  foMat.wireframe = params.showWireframe;
  fhMat.wireframe = params.showWireframe;

  if (params.showLandmarks) {
    const pts = ensureLandmarkPoints(state.faceOnly);
    pts.visible = true;
  } else if (landmarkPoints) {
    landmarkPoints.visible = false;
  }

  if (params.showMouthSeam) {
    const line = ensureMouthSeamLine(state.faceOnly, state.mouthAnchors);
    line.visible = true;
  } else if (mouthSeamLine) {
    mouthSeamLine.visible = false;
  }

  updateBlinkDebugOverlays(state, params);

  let mode: 'none' | 'mask' | 'hairMask' | 'faceDepth' | 'finalDepth' | 'mouthRegion' | 'eyeRegion' = 'none';
  if (params.showHeadMask) mode = 'mask';
  else if (params.showHairMask) mode = 'hairMask';
  else if (params.showFaceDepth) mode = 'faceDepth';
  else if (params.showFinalDepth) mode = 'finalDepth';
  else if (params.showMouthRegion) mode = 'mouthRegion';
  else if (params.showEyeRegion) mode = 'eyeRegion';

  if (mode === 'none') {
    fhMat.vertexColors = false;
    fhMat.map = state.texture;
    fhMat.needsUpdate = true;
    return;
  }

  let colors: Float32Array;
  if (mode === 'mask') {
    colors = buildMaskVertexColors(state.fullHead.maskValues);
  } else if (mode === 'hairMask') {
    colors = buildMaskVertexColors(state.fullHead.hairMaskValues);
  } else if (mode === 'faceDepth') {
    colors = buildDebugVertexColors(state.fullHead.debug.faceDepth);
  } else if (mode === 'finalDepth') {
    colors = buildDebugVertexColors(state.fullHead.debug.finalDepth);
  } else if (mode === 'mouthRegion') {
    colors = buildMouthRegionVertexColors(state.fullHead.cols * state.fullHead.rows, state.fullHeadMouthTable);
  } else {
    colors = buildEyeRegionVertexColors(state.fullHead.cols * state.fullHead.rows, state.fullHeadEyeTable);
  }
  state.fullHead.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  fhMat.vertexColors = true;
  fhMat.map = null;
  fhMat.needsUpdate = true;
}

const EXCLUSIVE_DEBUG_MODES = [
  'showHeadMask',
  'showHairMask',
  'showFaceDepth',
  'showFinalDepth',
  'showMouthRegion',
  'showEyeRegion',
] as const;

export function setupDebugGui(container: HTMLElement, params: Params, options: DebugGuiOptions): DebugGuiHandle {
  const gui = new GUI({ container, title: 'Quality Parameters' });

  const notifyDepth = () => {
    options.onDepthParamsChanged();
    applyDebugVisualization(options.getSceneState(), params);
  };

  const sourceFolder = gui.addFolder('Data Sources');
  // MEASURED=Google公式モデル(商用クリーン) / NEURAL=BiRefNet・DepthAnythingV2(高品質・比較用) / 旧方式
  sourceFolder
    .add(params, 'maskSource', ['MEASURED', 'NEURAL', 'ELLIPSE'])
    .name('Mask Source')
    .onChange(() => {
      options.onSourceChanged();
    });
  sourceFolder
    .add(params, 'depthSource', ['MEASURED', 'NEURAL', 'HEURISTIC'])
    .name('Depth Source')
    .onChange(() => {
      options.onSourceChanged();
    });
  sourceFolder.add(params, 'measuredRegularize', 0, 1, 0.01).name('Depth Regularize').onChange(notifyDepth);
  sourceFolder.add(params, 'measuredDepthGain', 0, 3, 0.05).name('Depth Gain').onChange(notifyDepth);
  sourceFolder.open();

  const depthFolder = gui.addFolder('Depth Parameters');
  depthFolder.add(params, 'faceDepthScale', 0, 3, 0.01).name('Face Depth').onChange(notifyDepth);
  depthFolder.add(params, 'canonicalMix', 0, 1, 0.01).name('Canonical Mix').onChange(notifyDepth);
  depthFolder.add(params, 'headDepthScale', 0, 0.5, 0.005).name('Head Depth').onChange(notifyDepth);
  depthFolder.add(params, 'edgeStart', 0.3, 0.98, 0.01).name('Edge Start').onChange(notifyDepth);
  depthFolder.add(params, 'edgeDepth', 0, 0.4, 0.005).name('Edge Roll').onChange(notifyDepth);
  depthFolder.add(params, 'blendWidthRatio', 0.02, 0.5, 0.005).name('Face/Head Blend').onChange(notifyDepth);
  depthFolder.add(params, 'hairVolumeMax', 0, 0.15, 0.002).name('Hair Volume').onChange(notifyDepth);
  depthFolder.open();

  const cameraFolder = gui.addFolder('Camera / Rotation');
  cameraFolder.add(params, 'pivotZRatio', -0.2, 0.05, 0.005).name('Pivot Z').onChange(notifyDepth);
  cameraFolder
    .add(params, 'maxYawDeg', 5, 45, 1)
    .name('Max Yaw')
    .onChange(() => options.onYawRangeChanged());
  cameraFolder
    .add(params, 'maxPitchDeg', 0, 45, 1)
    .name('Max Pitch')
    .onChange(() => options.onPitchRangeChanged());
  cameraFolder.add(params, 'cameraFovDeg', 15, 60, 1).name('Camera FOV').onChange(notifyDepth);
  cameraFolder.add(params, 'cameraDistanceRatio', 1.5, 8, 0.1).name('Camera Distance').onChange(notifyDepth);

  // --- Talk Animation / Mouth Cavity ---
  const talkFolder = gui.addFolder('Talk / Mouth');
  const talkEnabledController = talkFolder
    .add(params, 'talkEnabled')
    .name('Talk Animation')
    .onChange((v: boolean) => options.onTalkEnabledChangedFromGui(v));

  const talkOpenController = talkFolder
    .add(params, 'talkOpenManual', 0, 1, 0.01)
    .name('Talk Open Manual')
    .onChange((v: number) => {
      params.talkOpenManual = v;
      params.talkManualOverride = true;
      manualOverrideController.updateDisplay();
    });
  const manualOverrideController = talkFolder.add(params, 'talkManualOverride').name('Manual Override');

  talkFolder.add(params, 'upperLipMoveScale', 0, 0.02, 0.001).name('Upper Lip Move');
  talkFolder.add(params, 'lowerLipMoveScale', 0, 0.06, 0.001).name('Lower Lip Move');
  talkFolder.add(params, 'jawMoveScale', 0, 0.05, 0.001).name('Jaw Move');
  talkFolder.add(params, 'cornerInwardScale', 0, 0.01, 0.0005).name('Corner Inward');
  talkFolder.add(params, 'mouthCavityDepthRatio', -0.06, 0, 0.001).name('Mouth Cavity Depth');
  talkFolder
    .add(params, 'mouthCavityDarkness', 0, 1, 0.01)
    .name('Mouth Cavity Darkness')
    .onChange(() => options.onMouthCavityDarknessChanged());
  talkFolder.open();

  // --- Blink Animation ---
  const blinkFolder = gui.addFolder('Blink');
  const blinkEnabledController = blinkFolder
    .add(params, 'blinkEnabled')
    .name('Blink Animation')
    .onChange((v: boolean) => options.onBlinkEnabledChangedFromGui(v));

  blinkFolder
    .add(params, 'blinkAmountManual', 0, 1, 0.01)
    .name('Blink Amount Manual')
    .onChange((v: number) => {
      params.blinkAmountManual = v;
      params.blinkManualOverride = true;
      blinkManualOverrideController.updateDisplay();
      applyDebugVisualization(options.getSceneState(), params);
    });
  const blinkManualOverrideController = blinkFolder.add(params, 'blinkManualOverride').name('Manual Override');

  blinkFolder.add(params, 'blinkUpperLidMoveScale', 0, 3, 0.05).name('Upper Lid Move');
  blinkFolder.add(params, 'blinkLowerLidMove', 0, 0.2, 0.005).name('Lower Lid Move');
  blinkFolder.add(params, 'blinkCloseTargetBias', 0, 0.3, 0.005).name('Close Target Bias');
  blinkFolder.add(params, 'blinkClosingDurationMs', 20, 400, 5).name('Closing Duration');
  blinkFolder.add(params, 'blinkClosedHoldMs', 0, 400, 5).name('Closed Hold');
  blinkFolder.add(params, 'blinkOpeningDurationMs', 20, 400, 5).name('Opening Duration');
  blinkFolder.add(params, 'blinkIntervalMinSec', 0.5, 15, 0.1).name('Interval Min');
  blinkFolder.add(params, 'blinkIntervalMaxSec', 0.5, 15, 0.1).name('Interval Max');
  blinkFolder.add(params, 'blinkIntervalRandomize').name('Randomize Interval');
  blinkFolder.add(params, 'blinkUpperLidZEpsilonRatio', 0, 0.005, 0.0001).name('Upper Lid Z Epsilon');
  blinkFolder.add(params, 'showUpperLidLine').name('Show Upper Lid Line').onChange(() => applyDebugVisualization(options.getSceneState(), params));
  blinkFolder.add(params, 'showLowerLidLine').name('Show Lower Lid Line').onChange(() => applyDebugVisualization(options.getSceneState(), params));
  blinkFolder.add(params, 'showBlinkTargets').name('Show Blink Targets').onChange(() => applyDebugVisualization(options.getSceneState(), params));

  const debugFolder = gui.addFolder('Debug View');
  const onDebugToggle = () => applyDebugVisualization(options.getSceneState(), params);
  debugFolder.add(params, 'showWireframe').name('Show Wireframe').onChange(onDebugToggle);
  debugFolder.add(params, 'showLandmarks').name('Show Landmarks').onChange(onDebugToggle);
  debugFolder.add(params, 'showMouthSeam').name('Show Mouth Seam').onChange(onDebugToggle);

  const exclusiveControllers: Controller[] = [];
  const makeExclusiveToggle = (key: (typeof EXCLUSIVE_DEBUG_MODES)[number], label: string) => {
    const controller = debugFolder
      .add(params, key)
      .name(label)
      .onChange((v: boolean) => {
        if (v) {
          for (const other of EXCLUSIVE_DEBUG_MODES) {
            if (other !== key) params[other] = false;
          }
          exclusiveControllers.forEach((c) => c.updateDisplay());
        }
        onDebugToggle();
      });
    exclusiveControllers.push(controller);
  };
  makeExclusiveToggle('showHeadMask', 'Show Head Mask');
  makeExclusiveToggle('showHairMask', 'Show Hair Mask');
  makeExclusiveToggle('showFaceDepth', 'Show Face Depth');
  makeExclusiveToggle('showFinalDepth', 'Show Final Depth');
  makeExclusiveToggle('showMouthRegion', 'Show Mouth Region');
  makeExclusiveToggle('showEyeRegion', 'Show Eye Region');

  return {
    gui,
    syncTalkEnabled: () => talkEnabledController.updateDisplay(),
    syncBlinkEnabled: () => blinkEnabledController.updateDisplay(),
  };
}
