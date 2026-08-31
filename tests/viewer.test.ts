// 3D ビューの投影とパラメータ永続化の検査。
//
// 描画そのものはブラウザでしか動かないが、**投影行列は純粋計算**なので枠に収まることを機械で押さえ
// られる（境界球が回転に関わらず NDC に収まる、という保証がこの行列の存在理由）。

import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import {
  DEPTH_FILL,
  FRAME_FILL,
  MAXIMUM_ZOOM,
  MINIMUM_ZOOM,
  normalRotation,
  viewProjection,
} from '../src/presentation/viewer';
import { DEFAULT_SETTINGS } from '../src/application/settings';
import { LocalStorageParameterStore } from '../src/presentation/parameterStore';

const CENTER: readonly [number, number, number] = [0.01, 0.3, 0.02];
const RADIUS = 0.18;

function project(
  point: readonly [number, number, number],
  options: { yaw?: number; pitch?: number; zoom?: number; pan?: [number, number] } = {},
): THREE.Vector3 {
  const matrix = viewProjection({
    center: CENTER,
    radius: RADIUS,
    width: 800,
    height: 600,
    zoom: options.zoom ?? 1,
    yaw: options.yaw ?? 0,
    pitch: options.pitch ?? 0,
    pan: options.pan ?? [0, 0],
  });
  return new THREE.Vector3(point[0], point[1], point[2]).applyMatrix4(matrix);
}

describe('viewProjection', () => {
  it('中心は原点へ来る', () => {
    const projected = project(CENTER);
    expect(projected.x).toBeCloseTo(0, 10);
    expect(projected.y).toBeCloseTo(0, 10);
    expect(projected.z).toBeCloseTo(0, 10);
  });

  it('境界球はどの向きへ回しても NDC に収まる（FRAME_FILL の余白が残る）', () => {
    // 境界球の表面の点を球面上に散らして、全部が枠と深度の内側に入ることを見る。
    for (const yaw of [0, 0.7, 1.9, 3.0]) {
      for (const pitch of [-1.45, -0.5, 0, 1.45]) {
        for (let sample = 0; sample < 40; sample++) {
          const theta = (sample / 40) * Math.PI * 2;
          const phi = Math.acos(1 - (2 * (sample % 7)) / 6);
          const point: [number, number, number] = [
            CENTER[0] + RADIUS * Math.sin(phi) * Math.cos(theta),
            CENTER[1] + RADIUS * Math.sin(phi) * Math.sin(theta),
            CENTER[2] + RADIUS * Math.cos(phi),
          ];
          const projected = project(point, { yaw, pitch });
          // 短辺（高さ）方向は FRAME_FILL まで、長辺はそれより余裕がある。
          expect(Math.abs(projected.y)).toBeLessThanOrEqual(FRAME_FILL + 1e-9);
          expect(Math.abs(projected.x)).toBeLessThanOrEqual(FRAME_FILL + 1e-9);
          // 深度は [-DEPTH_FILL, DEPTH_FILL]（1 未満なので near/far に触れない）。
          expect(Math.abs(projected.z)).toBeLessThanOrEqual(DEPTH_FILL + 1e-9);
        }
      }
    }
  });

  it('拡大は x と y だけに掛かる（深度は動かない）', () => {
    const point: [number, number, number] = [CENTER[0], CENTER[1], CENTER[2] + RADIUS];
    const plain = project(point);
    const zoomed = project(point, { zoom: MAXIMUM_ZOOM });
    expect(zoomed.z).toBeCloseTo(plain.z, 10);
    // 拡大しても深度が [-1, 1] を越えない（越えると手前と奥が切られる）。
    expect(Math.abs(zoomed.z)).toBeLessThan(1);
  });

  it('GNM の +Z（前）は NDC の手前（z が小さい側）へ来る', () => {
    const front = project([CENTER[0], CENTER[1], CENTER[2] + RADIUS]);
    const back = project([CENTER[0], CENTER[1], CENTER[2] - RADIUS]);
    expect(front.z).toBeLessThan(back.z);
  });

  it('平行移動は NDC 単位で効き、深度に掛からない', () => {
    const point: [number, number, number] = [...CENTER] as [number, number, number];
    const moved = project(point, { pan: [0.25, -0.125] });
    expect(moved.x).toBeCloseTo(0.25, 10);
    expect(moved.y).toBeCloseTo(-0.125, 10);
    expect(moved.z).toBeCloseTo(0, 10);
  });

  it('yaw は縦軸まわり（+Y が動かない）', () => {
    const rotation = normalRotation(Math.PI / 2, 0);
    const up = new THREE.Vector3(0, 1, 0).applyMatrix4(rotation);
    expect(up.x).toBeCloseTo(0, 10);
    expect(up.y).toBeCloseTo(1, 10);
    expect(up.z).toBeCloseTo(0, 10);
  });

  it('半径が正でなければ落ちる', () => {
    expect(() =>
      viewProjection({
        center: CENTER,
        radius: 0,
        width: 100,
        height: 100,
        zoom: 1,
        yaw: 0,
        pitch: 0,
        pan: [0, 0],
      }),
    ).toThrow(/境界球の半径/);
  });

  it('拡大率の範囲は 0.3〜5.0', () => {
    expect(MINIMUM_ZOOM).toBe(0.3);
    expect(MAXIMUM_ZOOM).toBe(5.0);
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
});
