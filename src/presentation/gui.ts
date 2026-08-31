// 調整パラメータと 3D ビューの操作パネル。
//
// **表示の都合しか持たない。** 書き出しの既定値・範囲・選べる値は `application/settings` が持ち、
// 3D ビューの既定値と範囲は `presentation/viewSettings` が持つ（入口ごとに違う既定を持つと、どちらで
// 動かしたかで結果が変わる）。ここがするのは、その値を lil-gui のコントロールへ結ぶことだけ。
//
// デスクトップ側は CLI と GUI の 2 入口を持ち、`--help` がパラメータの一覧を出す。ブラウザは入口が
// 1 つなので、**このパネルが一覧そのもの**になる。
//
// 「首と視線」「表情」の節は Unity 側の Viewer パネル（`Viewer/GnmViewerUi`）と同じ並びにしてある。
// 同じものを同じ順で触れる方が、web と Unity を見比べるときに迷わない。

import GUI from 'lil-gui';
import {
  DEFAULT_IDENTITY_CLIP,
  DEFAULT_SETTINGS,
  EYE_TEXTURE_SIZE_CHOICES,
  ExportSettings,
  MAXIMUM_ATLAS_FOREGROUND_EXPONENT,
  MAXIMUM_ATLAS_FOREGROUND_THRESHOLD,
  MAXIMUM_ATLAS_HARMONIC_SCREENING,
  MAXIMUM_DISAGREEMENT_SCALE,
  MAXIMUM_HAIR_LIFT_MM,
  MAXIMUM_HAIR_ROLLOFF_MM,
  MAXIMUM_IDENTITY_CLIP,
  MINIMUM_ATLAS_FOREGROUND_EXPONENT,
  MINIMUM_ATLAS_FOREGROUND_THRESHOLD,
  MINIMUM_ATLAS_HARMONIC_SCREENING,
  MINIMUM_DISAGREEMENT_SCALE,
  MINIMUM_HAIR_LIFT_MM,
  MINIMUM_HAIR_ROLLOFF_MM,
  MINIMUM_IDENTITY_CLIP,
  TEXTURE_SIZE_CHOICES,
} from '../application/settings';
import { LAYER_ORDER } from '../domain/preview/asset';
import { ExpressionPlayMode } from '../domain/preview/expression';
import {
  GAZE_LIMIT_DEGREES,
  HeadPose,
  PITCH_LIMIT_DEGREES,
  YAW_LIMIT_DEGREES,
} from '../domain/preview/pose';
import {
  ALL_TEXTURES_KEY,
  LAYER_KEYS,
  MAXIMUM_DISTANCE_METERS,
  MAXIMUM_FOV_DEGREES,
  MINIMUM_DISTANCE_METERS,
  MINIMUM_FOV_DEGREES,
  RESET_KEY,
  TEXTURE_KEYS,
  WIREFRAME_KEY,
} from './viewer';
import {
  DEFAULT_VIEW_SETTINGS,
  MAXIMUM_EXPRESSION_INTENSITY,
  MAXIMUM_FADE_SECONDS,
  MAXIMUM_HOLD_SECONDS,
  MINIMUM_EXPRESSION_INTENSITY,
  MINIMUM_FADE_SECONDS,
  MINIMUM_HOLD_SECONDS,
  PLAY_MODES,
  PLAY_MODE_LABELS,
  ViewSettings,
} from './viewSettings';

/** 層の日本語ラベル（デスクトップ側の `LAYER_LABELS` と同じ）。 */
export const LAYER_LABELS: Readonly<Record<string, string>> = {
  skin: '肌',
  eyes: '眼球',
  mouth: '口腔内',
  hair: '髪シェル',
};

/** パネルが編集する状態（`ExportSettings` + `ViewSettings` + 表示切り替え + 表情の重み）。 */
export interface PanelState {
  settings: {
    disagreementScale: number;
    /** identity 係数の上限を置くか。**置かない**のが既定（公式 GNM も置いていない）。 */
    clipEnabled: boolean;
    identityClip: number;
    skinAtlasSize: number;
    eyeTextureSize: number;
    hairTextureSize: number;
    atlasForegroundThreshold: number;
    atlasForegroundExponent: number;
    atlasHarmonicScreening: number;
    hairLiftMm: number;
    hairRolloffMm: number;
  };
  /** 3D ビューの調整値（書き出しには影響しない）。 */
  view: {
    fovDegrees: number;
    distanceMeters: number;
    background: string;
    showWireframe: boolean;
    headYawDegrees: number;
    headPitchDegrees: number;
    gazeYawDegrees: number;
    gazePitchDegrees: number;
    neckShare: number;
    followPointer: boolean;
    playMode: ExpressionPlayMode;
    fadeSeconds: number;
    holdSeconds: number;
    expressionIntensity: number;
    blinkEnabled: boolean;
  };
  /** 手で立てる表情の重み（プリセット名 → 0〜1）。アセットを読むまで空。 */
  expressions: Record<string, number>;
  /** 層ごとの表示。 */
  visibleLayers: Record<string, boolean>;
  /** 層ごとのテクスチャ。OFF では `baseColor` と陰影だけになる。 */
  texturedLayers: Record<string, boolean>;
}

export function createPanelState(
  settings: ExportSettings = DEFAULT_SETTINGS,
  view: ViewSettings = DEFAULT_VIEW_SETTINGS,
): PanelState {
  const visibleLayers: Record<string, boolean> = {};
  const texturedLayers: Record<string, boolean> = {};
  for (const layer of LAYER_ORDER) {
    visibleLayers[layer] = true;
    texturedLayers[layer] = true;
  }
  return {
    settings: {
      disagreementScale: settings.disagreementScale,
      clipEnabled: settings.identityClip !== null,
      identityClip: settings.identityClip ?? DEFAULT_IDENTITY_CLIP,
      skinAtlasSize: settings.skinAtlasSize,
      eyeTextureSize: settings.eyeTextureSize,
      hairTextureSize: settings.hairTextureSize,
      atlasForegroundThreshold: settings.atlasForegroundThreshold,
      atlasForegroundExponent: settings.atlasForegroundExponent,
      atlasHarmonicScreening: settings.atlasHarmonicScreening,
      hairLiftMm: settings.hairLiftMm,
      hairRolloffMm: settings.hairRolloffMm,
    },
    view: { ...view },
    expressions: {},
    visibleLayers,
    texturedLayers,
  };
}

/** パネルの状態を `ExportSettings` へ移す。 */
export function toExportSettings(state: PanelState): ExportSettings {
  const { settings } = state;
  return {
    disagreementScale: settings.disagreementScale,
    identityClip: settings.clipEnabled ? settings.identityClip : null,
    skinAtlasSize: settings.skinAtlasSize,
    eyeTextureSize: settings.eyeTextureSize,
    hairTextureSize: settings.hairTextureSize,
    atlasForegroundThreshold: settings.atlasForegroundThreshold,
    atlasForegroundExponent: settings.atlasForegroundExponent,
    atlasHarmonicScreening: settings.atlasHarmonicScreening,
    hairLiftMm: settings.hairLiftMm,
    hairRolloffMm: settings.hairRolloffMm,
  };
}

/** パネルの状態を `ViewSettings` へ移す。 */
export function toViewSettings(state: PanelState): ViewSettings {
  return { ...state.view };
}

/** キーコードを人に見せる短い表記（`KeyA` → `A` / `Digit1` → `1`）。 */
function keyLabel(code: string): string {
  return code.replace(/^Key/, '').replace(/^Digit/, '');
}

export interface GuiCallbacks {
  onLayerVisibilityChanged: (layer: string, visible: boolean) => void;
  onLayerTextureChanged: (layer: string, enabled: boolean) => void;
  onAllTexturesToggled: () => void;
  onResetView: () => void;
  onSaveParameters: () => void;
  /** ビューの値が変わった（まとめて適用する）。 */
  onViewSettingsChanged: (view: ViewSettings) => void;
  /** 手で立てる表情の重みが変わった。 */
  onExpressionChanged: (name: string, weight: number) => void;
}

export interface GuiHandle {
  /** ビュー側で状態が変わったとき、パネルのチェックを合わせる。 */
  syncViewControls(
    layerStates: readonly [string, boolean][],
    textureStates: readonly [string, boolean][],
  ): void;
  /** ドラッグやマウス追従で動いた首と視線をスライダーへ戻す。 */
  syncHeadPose(pose: HeadPose): void;
  /** アセットを読んだ後に表情プリセットのスライダーを作る（名前はアセットが正本）。 */
  setExpressionPresets(names: readonly string[]): void;
}

export function setupGui(
  container: HTMLElement,
  state: PanelState,
  callbacks: GuiCallbacks,
): GuiHandle {
  const gui = new GUI({ container, title: '書き出しパラメータ', width: 300 });
  const pushView = (): void => callbacks.onViewSettingsChanged(toViewSettings(state));

  const fit = gui.addFolder('フィット');
  fit
    .add(
      state.settings,
      'disagreementScale',
      MINIMUM_DISAGREEMENT_SCALE,
      MAXIMUM_DISAGREEMENT_SCALE,
      0.05,
    )
    .name('事前分布の倍率');
  fit.add(state.settings, 'clipEnabled').name('係数の上限を置く');
  fit
    .add(state.settings, 'identityClip', MINIMUM_IDENTITY_CLIP, MAXIMUM_IDENTITY_CLIP, 0.1)
    .name('identity 係数の上限');

  const texture = gui.addFolder('テクスチャ');
  texture.add(state.settings, 'skinAtlasSize', [...TEXTURE_SIZE_CHOICES]).name('肌アトラスの一辺');
  texture
    .add(state.settings, 'eyeTextureSize', [...EYE_TEXTURE_SIZE_CHOICES])
    .name('眼球テクスチャの一辺');
  texture
    .add(state.settings, 'hairTextureSize', [...TEXTURE_SIZE_CHOICES])
    .name('髪テクスチャの長辺');

  const atlas = gui.addFolder('アトラス');
  atlas
    .add(
      state.settings,
      'atlasForegroundThreshold',
      MINIMUM_ATLAS_FOREGROUND_THRESHOLD,
      MAXIMUM_ATLAS_FOREGROUND_THRESHOLD,
      0.01,
    )
    .name('前景しきい値');
  atlas
    .add(
      state.settings,
      'atlasForegroundExponent',
      MINIMUM_ATLAS_FOREGROUND_EXPONENT,
      MAXIMUM_ATLAS_FOREGROUND_EXPONENT,
      0.5,
    )
    .name('前景の指数');
  atlas
    .add(
      state.settings,
      'atlasHarmonicScreening',
      MINIMUM_ATLAS_HARMONIC_SCREENING,
      MAXIMUM_ATLAS_HARMONIC_SCREENING,
      0.05,
    )
    .name('harmonic screening');

  const hair = gui.addFolder('髪シェル');
  hair
    .add(state.settings, 'hairLiftMm', MINIMUM_HAIR_LIFT_MM, MAXIMUM_HAIR_LIFT_MM, 0.1)
    .name('持ち上げ (mm)');
  hair
    .add(state.settings, 'hairRolloffMm', MINIMUM_HAIR_ROLLOFF_MM, MAXIMUM_HAIR_ROLLOFF_MM, 0.1)
    .name('巻き込み (mm)');

  const actions = { 保存: callbacks.onSaveParameters };
  gui.add(actions, '保存').name('パラメーターを保存（次回起動時に復元）');

  const view = gui.addFolder('3Dビュー');

  const camera = view.addFolder('カメラ');
  camera
    .add(state.view, 'fovDegrees', MINIMUM_FOV_DEGREES, MAXIMUM_FOV_DEGREES, 1)
    .name('画角 (°)')
    .onChange(pushView);
  camera
    .add(state.view, 'distanceMeters', MINIMUM_DISTANCE_METERS, MAXIMUM_DISTANCE_METERS, 0.05)
    .name('距離 (m)')
    .onChange(pushView);
  camera.addColor(state.view, 'background').name('背景色').onChange(pushView);
  const wireframeController = camera
    .add(state.view, 'showWireframe')
    .name(`ワイヤーフレーム   [${keyLabel(WIREFRAME_KEY)}]`)
    .onChange(pushView);

  // 可動域の上限は `domain/preview/pose` が持つ（Unity 側と同じ値）。
  const pose = view.addFolder('首と視線');
  const poseControllers = [
    pose
      .add(state.view, 'headYawDegrees', -YAW_LIMIT_DEGREES, YAW_LIMIT_DEGREES, 0.5)
      .name('首 yaw (°)')
      .onChange(pushView),
    pose
      .add(state.view, 'headPitchDegrees', -PITCH_LIMIT_DEGREES, PITCH_LIMIT_DEGREES, 0.5)
      .name('首 pitch (°)')
      .onChange(pushView),
    pose
      .add(state.view, 'gazeYawDegrees', -GAZE_LIMIT_DEGREES, GAZE_LIMIT_DEGREES, 0.5)
      .name('視線 yaw (°)')
      .onChange(pushView),
    pose
      .add(state.view, 'gazePitchDegrees', -GAZE_LIMIT_DEGREES, GAZE_LIMIT_DEGREES, 0.5)
      .name('視線 pitch (°)')
      .onChange(pushView),
  ];
  pose.add(state.view, 'neckShare', 0, 1, 0.05).name('首へ配る割合').onChange(pushView);
  pose.add(state.view, 'followPointer').name('マウス追従').onChange(pushView);

  const expression = view.addFolder('表情');
  // ラベルは `PLAY_MODE_LABELS` から作る（一覧をここに書き写すと増減で黙って古くなる）。
  const playModeChoices: Record<string, ExpressionPlayMode> = {};
  for (const mode of PLAY_MODES) playModeChoices[PLAY_MODE_LABELS[mode]] = mode;
  expression.add(state.view, 'playMode', playModeChoices).name('自動再生').onChange(pushView);
  expression
    .add(state.view, 'fadeSeconds', MINIMUM_FADE_SECONDS, MAXIMUM_FADE_SECONDS, 0.05)
    .name('立ち上がり (秒)')
    .onChange(pushView);
  expression
    .add(state.view, 'holdSeconds', MINIMUM_HOLD_SECONDS, MAXIMUM_HOLD_SECONDS, 0.05)
    .name('保持 (秒)')
    .onChange(pushView);
  expression
    .add(
      state.view,
      'expressionIntensity',
      MINIMUM_EXPRESSION_INTENSITY,
      MAXIMUM_EXPRESSION_INTENSITY,
      0.05,
    )
    .name('強さ')
    .onChange(pushView);
  expression.add(state.view, 'blinkEnabled').name('自動まばたき').onChange(pushView);
  // プリセットの名前はアセットが正本なので、ここに一覧を書かない（増減で黙って古くなる）。
  const presets = expression.addFolder('プリセット');
  presets.hide();

  const layerControllers = new Map<string, ReturnType<typeof view.add>>();
  const textureControllers = new Map<string, ReturnType<typeof view.add>>();
  const layerKeyOf = (layer: string): string =>
    Object.entries(LAYER_KEYS).find(([, value]) => value === layer)?.[0] ?? '';
  const textureKeyOf = (layer: string): string =>
    Object.entries(TEXTURE_KEYS).find(([, value]) => value === layer)?.[0] ?? '';

  const layers = view.addFolder('表示する層');
  for (const layer of LAYER_ORDER) {
    layerControllers.set(
      layer,
      layers
        .add(state.visibleLayers, layer)
        .name(`${LAYER_LABELS[layer] ?? layer}   [${keyLabel(layerKeyOf(layer))}]`)
        .onChange((value: boolean) => callbacks.onLayerVisibilityChanged(layer, value)),
    );
  }
  const textures = view.addFolder('テクスチャを貼る層');
  for (const layer of LAYER_ORDER) {
    textureControllers.set(
      layer,
      textures
        .add(state.texturedLayers, layer)
        .name(`${LAYER_LABELS[layer] ?? layer}   [${keyLabel(textureKeyOf(layer))}]`)
        .onChange((value: boolean) => callbacks.onLayerTextureChanged(layer, value)),
    );
  }
  const viewActions = {
    全テクスチャ: callbacks.onAllTexturesToggled,
    視点: callbacks.onResetView,
  };
  view.add(viewActions, '全テクスチャ').name(`全テクスチャを切り替え   [${keyLabel(ALL_TEXTURES_KEY)}]`);
  view.add(viewActions, '視点').name(`正面・無表情に戻す   [${keyLabel(RESET_KEY)}]`);

  return {
    syncViewControls(layerStates, textureStates) {
      for (const [layer, visible] of layerStates) {
        state.visibleLayers[layer] = visible;
        layerControllers.get(layer)?.updateDisplay();
      }
      for (const [layer, enabled] of textureStates) {
        state.texturedLayers[layer] = enabled;
        textureControllers.get(layer)?.updateDisplay();
      }
      wireframeController.updateDisplay();
    },
    syncHeadPose(pose_) {
      state.view.headYawDegrees = pose_.headYawDegrees;
      state.view.headPitchDegrees = pose_.headPitchDegrees;
      state.view.gazeYawDegrees = pose_.gazeYawDegrees;
      state.view.gazePitchDegrees = pose_.gazePitchDegrees;
      for (const controller of poseControllers) controller.updateDisplay();
    },
    setExpressionPresets(names) {
      presets.children.slice().forEach((child) => child.destroy());
      for (const name of names) {
        if (state.expressions[name] === undefined) state.expressions[name] = 0;
        presets
          .add(state.expressions, name, 0, 1, 0.01)
          .onChange((value: number) => callbacks.onExpressionChanged(name, value));
      }
      if (names.length > 0) presets.show();
    },
  };
}
