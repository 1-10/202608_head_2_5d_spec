// 3D ビューの定数とパラメータ永続化の検査。
//
// 描画そのものはブラウザでしか動かないので、ここで押さえるのは**Unity 側から写した値**と、値の
// 保存が壊れていないこと。カメラ・光・背景・alpha clip は 1-10/2607_Obayashi_Avatar_Mockup_3DGS の
// `Assets/Sandbox/Ooba/GNM` が正本で、写しなのでズレたら気付ける形にしておく。
//
// 首と視線・表情・領域分けの数値そのものは `tests/preview.test.ts`（純粋計算と実アセット）で見る。

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
import { DEFAULT_SETTINGS } from '../src/application/settings';
import { LocalStorageParameterStore } from '../src/presentation/parameterStore';
import { DEFAULT_VIEW_SETTINGS, normalizeViewSettings } from '../src/presentation/viewSettings';

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
    expect(AMBIENT_LIGHT).toBeGreaterThan(0);
    expect(AMBIENT_LIGHT).toBeLessThan(1);
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

describe('3D ビューの値の正規化', () => {
  it('既定値はそのまま通る', () => {
    expect(normalizeViewSettings(DEFAULT_VIEW_SETTINGS)).toEqual(DEFAULT_VIEW_SETTINGS);
  });

  it('範囲外は**捨てずに丸める**（書き出しの値とは扱いが違う）', () => {
    const normalized = normalizeViewSettings({
      ...DEFAULT_VIEW_SETTINGS,
      fovDegrees: 999,
      headYawDegrees: -999,
      neckShare: 5,
    });
    expect(normalized.fovDegrees).toBe(60);
    expect(normalized.headYawDegrees).toBe(-15);
    expect(normalized.neckShare).toBe(1);
  });

  it('型が違う値・不正な色・知らない再生モードは既定へ戻す', () => {
    const normalized = normalizeViewSettings({
      fovDegrees: 'wide',
      background: 'red',
      playMode: 'loop',
      blinkEnabled: 'yes',
    });
    expect(normalized.fovDegrees).toBe(DEFAULT_VIEW_SETTINGS.fovDegrees);
    expect(normalized.background).toBe(DEFAULT_VIEW_SETTINGS.background);
    expect(normalized.playMode).toBe('off');
    expect(normalized.blinkEnabled).toBe(DEFAULT_VIEW_SETTINGS.blinkEnabled);
  });

  it('オブジェクトでなければ丸ごと既定', () => {
    expect(normalizeViewSettings(null)).toEqual(DEFAULT_VIEW_SETTINGS);
    expect(normalizeViewSettings([1, 2])).toEqual(DEFAULT_VIEW_SETTINGS);
  });
});

/** `localStorage` の代わり（Node にはブラウザの Storage が無い）。 */
class FakeStorage implements Storage {
  private readonly map = new Map<string, string>();
  get length(): number {
    return this.map.size;
  }
  clear(): void {
    this.map.clear();
  }
  getItem(key: string): string | null {
    return this.map.get(key) ?? null;
  }
  key(index: number): string | null {
    return [...this.map.keys()][index] ?? null;
  }
  removeItem(key: string): void {
    this.map.delete(key);
  }
  setItem(key: string, value: string): void {
    this.map.set(key, value);
  }
}

describe('パラメータの永続化', () => {
  it('保存して読み戻せる', () => {
    const store = new LocalStorageParameterStore(new FakeStorage());
    expect(store.load()).toBeNull();
    const settings = { ...DEFAULT_SETTINGS, skinAtlasSize: 1024, identityClip: 3 };
    store.save(settings);
    expect(store.load()).toEqual(settings);
  });

  it('壊れた値は使わず null を返す（application の既定へ戻す）', () => {
    const storage = new FakeStorage();
    const store = new LocalStorageParameterStore(storage);
    storage.setItem('export_parameters/v1', '{壊れた');
    expect(store.load()).toBeNull();
    // 検査を通らない値（選べない一辺）も使わない。
    storage.setItem(
      'export_parameters/v1',
      JSON.stringify({ ...DEFAULT_SETTINGS, skinAtlasSize: 777 }),
    );
    expect(store.load()).toBeNull();
  });

  it('検査を通らない値は保存しない（保存したつもりの値が黙って消えないように）', () => {
    const store = new LocalStorageParameterStore(new FakeStorage());
    expect(() =>
      store.save({ ...DEFAULT_SETTINGS, hairLiftMm: DEFAULT_SETTINGS.hairRolloffMm + 1 }),
    ).toThrow(/hairLiftMm/);
  });

  it('3D ビューの値は別のキーへ保存し、片方が壊れても他方に影響しない', () => {
    const storage = new FakeStorage();
    const store = new LocalStorageParameterStore(storage);
    store.save(DEFAULT_SETTINGS);
    store.saveView({ ...DEFAULT_VIEW_SETTINGS, fovDegrees: 35, playMode: 'random' });
    expect(store.loadView().fovDegrees).toBe(35);
    expect(store.loadView().playMode).toBe('random');
    storage.setItem('view_parameters/v1', '{壊れた');
    expect(store.loadView()).toEqual(DEFAULT_VIEW_SETTINGS);
    expect(store.load()).toEqual(DEFAULT_SETTINGS);
  });
});
