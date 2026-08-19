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

export interface SceneStateLike {
  ctx: FullHeadBuildContext;
  normalized: NormalizedFaceResult;
  faceOnly: FaceOnlyBuild;
  fullHead: FullHeadBuild;
  texture: THREE.Texture;
  mouthAnchors: MouthAnchors;
  fullHeadMouthTable: MouthDeformEntry[];
}

export interface DebugGuiOptions {
  onDepthParamsChanged: () => void;
  onSourceChanged: () => void;
  onYawRangeChanged: () => void;
  onPitchRangeChanged: () => void;
  getSceneState: () => SceneStateLike | null;
  onTalkEnabledChangedFromGui: (value: boolean) => void;
  onMouthCavityDarknessChanged: () => void;
}

export interface DebugGuiHandle {
  gui: GUI;
  /** paramsに外部(下部パネル)から書き込まれたtalkEnabledをGUI表示へ反映する。 */
  syncTalkEnabled: () => void;
}

let landmarkPoints: THREE.Points | null = null;
let mouthSeamLine: THREE.Line | null = null;

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

  let mode: 'none' | 'mask' | 'hairMask' | 'faceDepth' | 'finalDepth' | 'mouthRegion' = 'none';
  if (params.showHeadMask) mode = 'mask';
  else if (params.showHairMask) mode = 'hairMask';
  else if (params.showFaceDepth) mode = 'faceDepth';
  else if (params.showFinalDepth) mode = 'finalDepth';
  else if (params.showMouthRegion) mode = 'mouthRegion';

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
  } else {
    colors = buildMouthRegionVertexColors(state.fullHead.cols * state.fullHead.rows, state.fullHeadMouthTable);
  }
  state.fullHead.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  fhMat.vertexColors = true;
  fhMat.map = null;
  fhMat.needsUpdate = true;
}

const EXCLUSIVE_DEBUG_MODES = ['showHeadMask', 'showHairMask', 'showFaceDepth', 'showFinalDepth', 'showMouthRegion'] as const;

export function setupDebugGui(container: HTMLElement, params: Params, options: DebugGuiOptions): DebugGuiHandle {
  const gui = new GUI({ container, title: 'Quality Parameters' });

  const notifyDepth = () => {
    options.onDepthParamsChanged();
    applyDebugVisualization(options.getSceneState(), params);
  };

  const sourceFolder = gui.addFolder('Data Sources');
  sourceFolder
    .add(params, 'maskSource', ['MEASURED', 'ELLIPSE'])
    .name('Mask Source')
    .onChange(() => {
      options.onSourceChanged();
    });
  sourceFolder
    .add(params, 'depthSource', ['MEASURED', 'HEURISTIC'])
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

  return {
    gui,
    syncTalkEnabled: () => talkEnabledController.updateDisplay(),
  };
}
