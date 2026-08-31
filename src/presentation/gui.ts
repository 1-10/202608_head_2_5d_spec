// 調整パラメータと 3D ビューの操作パネル。
//
// **表示の都合しか持たない。** 既定値・範囲・選べる値はすべて `application/settings` が持つ（入口
// ごとに違う既定を持つと、どちらで動かしたかで結果が変わる）。ここがするのは、その値を lil-gui の
// コントロールへ結ぶことだけ。
//
// デスクトップ側は CLI と GUI の 2 入口を持ち、`--help` がパラメータの一覧を出す。ブラウザは入口が
// 1 つなので、**このパネルが一覧そのもの**になる。

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
import { LAYER_ORDER } from '../domain/debugScene';
import { ALL_TEXTURES_KEY, LAYER_KEYS, RESET_KEY, TEXTURE_KEYS } from './viewer';

/** 層の日本語ラベル（デスクトップ側の `LAYER_LABELS` と同じ）。 */
export const LAYER_LABELS: Readonly<Record<string, string>> = {
  skin: '肌',
  eyes: '眼球',
  mouth: '口腔内',
  hair: '髪シェル',
};

/** パネルが編集する状態（`ExportSettings` + ビューの表示切り替え）。 */
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
  /** 層ごとの表示。 */
  visibleLayers: Record<string, boolean>;
  /** 層ごとのテクスチャ。OFF では `baseColor` と陰影だけになる。 */
  texturedLayers: Record<string, boolean>;
}

export function createPanelState(settings: ExportSettings = DEFAULT_SETTINGS): PanelState {
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
}

export interface GuiHandle {
  /** ビュー側で状態が変わったとき、パネルのチェックを合わせる。 */
  syncViewControls(layerStates: readonly [string, boolean][], textureStates: readonly [string, boolean][]): void;
}

export function setupGui(
  container: HTMLElement,
  state: PanelState,
  callbacks: GuiCallbacks,
): GuiHandle {
  const gui = new GUI({ container, title: '書き出しパラメータ', width: 300 });

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
  view.add(viewActions, '視点').name(`正面・等倍に戻す   [${keyLabel(RESET_KEY)}]`);

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
    },
  };
}
