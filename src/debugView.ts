// GUIパラメータパネルとデバッグ表示 (Wireframe)。

import GUI from 'lil-gui';
import * as THREE from 'three';
import type { Params } from './params';
import type { GnmHeadBuild } from './gnmHeadMesh';

export interface DebugGuiOptions {
  onSourceChanged: () => void; // Mask/Depthソース切替 (NEURALは遅延取得)
  onGnmParamsChanged: () => void; // フィット/髪シェルパラメータ変更 (再構築)
  onCameraChanged: () => void;
  onYawRangeChanged: () => void;
  onPitchRangeChanged: () => void;
  getGnmHead: () => GnmHeadBuild | null;
}

/** GUIのデバッグ表示チェックボックス状態を実際のThree.jsシーンへ反映する。 */
export function applyDebugVisualization(gnmHead: GnmHeadBuild | null, params: Params): void {
  if (!gnmHead) return;
  (gnmHead.headMesh.material as THREE.MeshStandardMaterial).wireframe = params.showWireframe;
  if (gnmHead.hairMesh) {
    (gnmHead.hairMesh.material as THREE.MeshStandardMaterial).wireframe = params.showWireframe;
  }
  gnmHead.landmarkOverlay.visible = params.showLandmarks;
}

export function setupDebugGui(container: HTMLElement, params: Params, options: DebugGuiOptions): GUI {
  const gui = new GUI({ container, title: 'Parameters' });

  const sourceFolder = gui.addFolder('Data Sources');
  // MEASURED=Google公式モデル(商用クリーン) / NEURAL=BiRefNet・DepthAnythingV2(高品質・比較用) / NONE=不使用
  sourceFolder
    .add(params, 'maskSource', ['MEASURED', 'NEURAL', 'NONE'])
    .name('Mask Source')
    .onChange(() => options.onSourceChanged());
  sourceFolder
    .add(params, 'depthSource', ['MEASURED', 'NEURAL', 'NONE'])
    .name('Depth Source')
    .onChange(() => options.onSourceChanged());
  sourceFolder
    .add(params, 'measuredDepthGain', 0, 3, 0.05)
    .name('Depth Gain')
    .onFinishChange(() => options.onGnmParamsChanged());
  sourceFolder.open();

  const fitFolder = gui.addFolder('GNM Fit');
  fitFolder
    .add(params, 'gnmIdentityReg', 0.05, 10, 0.05)
    .name('Identity Reg')
    .onFinishChange(() => options.onGnmParamsChanged());
  fitFolder
    .add(params, 'gnmDenseFit')
    .name('Dense Fit (468pt)')
    .onChange(() => options.onGnmParamsChanged());
  fitFolder
    .add(params, 'gnmWarpStrength', 0, 1.5, 0.05)
    .name('Residual Warp')
    .onFinishChange(() => options.onGnmParamsChanged());
  fitFolder
    .add(params, 'gnmHairLift', 0, 0.2, 0.005)
    .name('Hair Lift')
    .onFinishChange(() => options.onGnmParamsChanged());
  fitFolder
    .add(params, 'gnmHairRolloff', 0, 1, 0.01)
    .name('Hair Rolloff')
    .onFinishChange(() => options.onGnmParamsChanged());
  fitFolder
    .add(params, 'gnmHairSkinFill')
    .name('Hair Skin Fill')
    .onChange(() => options.onGnmParamsChanged());
  fitFolder
    .add(params, 'gnmHairFillStrength', 0, 2, 0.05)
    .name('Hair Fill Strength')
    .onFinishChange(() => options.onGnmParamsChanged());
  fitFolder.open();

  // プリセットは公式ExpressionSampler由来 (gnmExpressions.ts)
  const exprFolder = gui.addFolder('Expression');
  exprFolder
    .add(params, 'gnmEmotion', {
      'Auto (喜怒哀楽を巡回)': 'AUTO',
      Neutral: 'NEUTRAL',
      'Manual (下のスライダー)': 'MANUAL',
      '喜 Happy': 'joy',
      '楽 Smile': 'fun',
      '哀 Sad': 'sad',
      '怒 Snarl': 'anger',
      '驚 Surprise': 'surprise',
    })
    .name('Emotion');
  exprFolder.add(params, 'gnmExprIntensity', 0, 2, 0.05).name('Intensity');
  // パーツ別スライダー (Emotion=Manual時に有効。公式クラスを目/下顔面領域で分離)
  exprFolder.add(params, 'gnmMouthOpen', 0, 1.5, 0.05).name('Mouth Open');
  exprFolder.add(params, 'gnmSmile', 0, 1.5, 0.05).name('Smile');
  exprFolder.add(params, 'gnmPucker', 0, 1.5, 0.05).name('Pucker');
  exprFolder.add(params, 'gnmCornersDown', 0, 1.5, 0.05).name('Corners Down');
  exprFolder.add(params, 'gnmEyesClose', 0, 1.2, 0.05).name('Eyes Close');
  exprFolder.add(params, 'gnmEyesWide', 0, 1.5, 0.05).name('Eyes Wide');
  exprFolder.add(params, 'gnmSquint', 0, 1.5, 0.05).name('Squint');

  const cameraFolder = gui.addFolder('Camera / Rotation');
  cameraFolder
    .add(params, 'maxYawDeg', 5, 45, 1)
    .name('Max Yaw')
    .onChange(() => options.onYawRangeChanged());
  cameraFolder
    .add(params, 'maxPitchDeg', 0, 45, 1)
    .name('Max Pitch')
    .onChange(() => options.onPitchRangeChanged());
  cameraFolder
    .add(params, 'cameraFovDeg', 15, 60, 1)
    .name('Camera FOV')
    .onChange(() => options.onCameraChanged());
  cameraFolder
    .add(params, 'cameraDistanceRatio', 1.5, 8, 0.1)
    .name('Camera Distance')
    .onChange(() => options.onCameraChanged());

  const debugFolder = gui.addFolder('Debug View');
  debugFolder
    .add(params, 'showWireframe')
    .name('Show Wireframe')
    .onChange(() => applyDebugVisualization(options.getGnmHead(), params));
  debugFolder
    .add(params, 'showLandmarks')
    .name('Show Landmarks')
    .onChange(() => applyDebugVisualization(options.getGnmHead(), params));

  return gui;
}
