// 調整パラメータのパネル。
//
// **表示の都合しか持たない。** 既定値・範囲・選べる値はすべて `application/settings` が持つ
// （入口ごとに違う既定を持つと、どちらで動かしたかで結果が変わる）。ここがするのは、その値を
// lil-gui のコントロールへ結ぶことだけ。
//
// デスクトップ側は CLI と GUI の 2 入口を持ち、`--help` がパラメータの一覧を出す。ブラウザは入口が
// 1 つなので、**このパネルが一覧そのもの**になる。

import GUI from 'lil-gui';
import {
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

/** パネルが編集する状態（`ExportSettings` + ビューの表示切り替え）。 */
export interface PanelState {
  settings: {
    disagreementScale: number;
    /** 0 = 上限なし（`identityClip: null`）。lil-gui は null を扱えないので 0 を「無し」に使う。 */
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
  visibleLayers: Record<string, boolean>;
}

export function createPanelState(): PanelState {
  const visibleLayers: Record<string, boolean> = {};
  for (const layer of LAYER_ORDER) visibleLayers[layer] = true;
  return {
    settings: {
      disagreementScale: DEFAULT_SETTINGS.disagreementScale,
      identityClip: DEFAULT_SETTINGS.identityClip ?? 0,
      skinAtlasSize: DEFAULT_SETTINGS.skinAtlasSize,
      eyeTextureSize: DEFAULT_SETTINGS.eyeTextureSize,
      hairTextureSize: DEFAULT_SETTINGS.hairTextureSize,
      atlasForegroundThreshold: DEFAULT_SETTINGS.atlasForegroundThreshold,
      atlasForegroundExponent: DEFAULT_SETTINGS.atlasForegroundExponent,
      atlasHarmonicScreening: DEFAULT_SETTINGS.atlasHarmonicScreening,
      hairLiftMm: DEFAULT_SETTINGS.hairLiftMm,
      hairRolloffMm: DEFAULT_SETTINGS.hairRolloffMm,
    },
    visibleLayers,
  };
}

/** パネルの状態を `ExportSettings` へ移す（0 の `identityClip` は「上限なし」）。 */
export function toExportSettings(state: PanelState): ExportSettings {
  const { settings } = state;
  return {
    disagreementScale: settings.disagreementScale,
    identityClip: settings.identityClip <= 0 ? null : settings.identityClip,
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

export function setupGui(
  container: HTMLElement,
  state: PanelState,
  callbacks: { onLayersChanged: () => void },
): GUI {
  const gui = new GUI({ container, title: '書き出しパラメータ', width: 300 });

  const fit = gui.addFolder('フィット');
  fit
    .add(state.settings, 'disagreementScale', MINIMUM_DISAGREEMENT_SCALE, MAXIMUM_DISAGREEMENT_SCALE, 0.05)
    .name('事前分布の倍率');
  fit
    .add(state.settings, 'identityClip', 0, MAXIMUM_IDENTITY_CLIP, 0.1)
    .name(`identity 上限（0=無し / ${MINIMUM_IDENTITY_CLIP}〜）`);

  const texture = gui.addFolder('テクスチャ');
  texture
    .add(state.settings, 'skinAtlasSize', [...TEXTURE_SIZE_CHOICES])
    .name('肌アトラスの一辺');
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

  const layers = gui.addFolder('3Dビューの層');
  for (const layer of LAYER_ORDER) {
    layers.add(state.visibleLayers, layer).name(layer).onChange(callbacks.onLayersChanged);
  }
  return gui;
}
