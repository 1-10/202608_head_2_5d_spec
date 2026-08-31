// 3D ビューの定数の検査。
//
// 描画そのものはブラウザでしか動かないので、ここで押さえるのは**Unity 側から写した値**。
// カメラ・光・背景・alpha clip は 1-10/2607_Obayashi_Avatar_Mockup_3DGS の
// `Assets/Sandbox/Ooba/GNM` が正本で、写しなのでズレたら気付ける形にしておく。
//
// 首と視線・表情・領域分け・法線の数値そのものは `tests/preview.test.ts`（純粋計算と実アセット）で見る。

import { describe, expect, it } from 'vitest';
import {
  ALL_TEXTURES_KEY,
  AMBIENT_LIGHT,
  DEFAULT_BACKGROUND,
  DEFAULT_DISTANCE_METERS,
  DEFAULT_FOV_DEGREES,
  LAYER_KEYS,
  LIGHT_DIRECTION,
  MAXIMUM_ZOOM,
  MINIMUM_ZOOM,
  RESET_KEY,
  TARGET_HEIGHT_METERS,
  TEXTURE_KEYS,
  WIREFRAME_KEY,
} from '../src/presentation/viewer';
import { LAYER_ORDER } from '../src/domain/preview/asset';
import { DEFAULT_VIEW_SETTINGS } from '../src/presentation/viewSettings';

describe('Unity 側から写したカメラと光', () => {
  it('カメラは Viewer.unity の MainCamera と同じ', () => {
    expect(DEFAULT_FOV_DEGREES).toBe(20);
    expect(DEFAULT_DISTANCE_METERS).toBeCloseTo(1.3, 10);
    expect(TARGET_HEIGHT_METERS).toBeCloseTo(0.297, 10);
    expect(DEFAULT_BACKGROUND).toBe('#26292e');
  });

  it('光は上・前・被写体から見て右から来る（Unity の DirectionalLight と同じ向き）', () => {
    const [x, y, z] = LIGHT_DIRECTION;
    // GNM 空間の +X は解剖学的な左。Unity 空間は X 反転なので、あちらの +X 側の光は
    // こちらでは負になる。**ここの符号を間違えると顔の陰の向きが左右反転する。**
    expect(x).toBeLessThan(0);
    expect(y).toBeGreaterThan(0);
    expect(z).toBeGreaterThan(0);
    expect(Math.hypot(x, y, z)).toBeCloseTo(1, 3);
  });

  it('環境光は旧 web 版と同じ 0.65（写真の陰影の上へ影を重ねすぎない）', () => {
    expect(AMBIENT_LIGHT).toBeCloseTo(0.65, 10);
  });

  it('拡大率の範囲は 0.3〜5.0', () => {
    expect(MINIMUM_ZOOM).toBe(0.3);
    expect(MAXIMUM_ZOOM).toBe(5.0);
  });
});

describe('キー割り当て', () => {
  it('層とテクスチャのキーは LAYER_ORDER と同じ並び', () => {
    expect(Object.values(LAYER_KEYS)).toEqual([...LAYER_ORDER]);
    expect(Object.values(TEXTURE_KEYS)).toEqual([...LAYER_ORDER]);
  });

  it('単独キーが重複していない', () => {
    const codes = [
      ...Object.keys(LAYER_KEYS),
      ...Object.keys(TEXTURE_KEYS),
      ALL_TEXTURES_KEY,
      RESET_KEY,
      WIREFRAME_KEY,
    ];
    expect(new Set(codes).size).toBe(codes.length);
  });
});

describe('3D ビューの既定値', () => {
  it('カメラの既定は Unity 側の値そのまま', () => {
    expect(DEFAULT_VIEW_SETTINGS.fovDegrees).toBe(DEFAULT_FOV_DEGREES);
    expect(DEFAULT_VIEW_SETTINGS.distanceMeters).toBe(DEFAULT_DISTANCE_METERS);
    expect(DEFAULT_VIEW_SETTINGS.background).toBe(DEFAULT_BACKGROUND);
  });

  it('起動時は無表情・正面・自動再生なし（まばたきだけ動く）', () => {
    expect(DEFAULT_VIEW_SETTINGS.playMode).toBe('off');
    expect(DEFAULT_VIEW_SETTINGS.headYawDegrees).toBe(0);
    expect(DEFAULT_VIEW_SETTINGS.headPitchDegrees).toBe(0);
    expect(DEFAULT_VIEW_SETTINGS.followPointer).toBe(false);
    expect(DEFAULT_VIEW_SETTINGS.blinkEnabled).toBe(true);
  });
});
