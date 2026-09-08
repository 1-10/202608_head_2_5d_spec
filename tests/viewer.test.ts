// 3D ビューの定数の検査。
//
// 描画そのものはブラウザでしか動かないので、ここで押さえるのは**Unity 側から写した値**。
// カメラ・光・背景・alpha clip は 1-10/2607_Obayashi_Avatar_Mockup_3DGS の
// `Assets/Sandbox/Ooba/GNM` が正本で、写しなのでズレたら気付ける形にしておく。
//
// 首と視線・表情・領域分け・法線の数値そのものは `tests/preview.test.ts`（純粋計算と実アセット）で見る。

import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import {
  ALL_TEXTURES_KEY,
  AMBIENT_LIGHT,
  DEFAULT_AMBIENT_COLOR,
  DEFAULT_LIGHT_COLOR,
  DEFAULT_LIGHT_INTENSITY,
  DEFAULT_BACKGROUND,
  FRAGMENT_SHADER,
  LAYER_KEYS,
  LIGHT_DIRECTION,
  RESET_KEY,
  TEXTURE_KEYS,
  WIREFRAME_KEY,
  srgbBaseColor,
} from '../src/presentation/viewer';
import { LAYER_ORDER } from '../src/domain/preview/asset';
import {
  DEFAULT_FOV_DEGREES,
  DEFAULT_ORBIT_RADIUS_METERS,
  MAXIMUM_ORBIT_RADIUS_METERS,
  MINIMUM_ORBIT_RADIUS_METERS,
  TARGET_HEIGHT_METERS,
  cameraPoseAt,
} from '../src/domain/preview/camera';
import { DEFAULT_VIEW_SETTINGS } from '../src/presentation/viewSettings';

describe('Unity 側から写したカメラと光', () => {
  it('背景は Viewer.unity の MainCamera と同じ', () => {
    // カメラの姿勢そのもの（画角・距離・注視点の高さ）は `tests/camera.test.ts` が見る。
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

  it('平行光の色と強さは Unity の DirectionalLight と同じ', () => {
    expect(DEFAULT_LIGHT_COLOR).toBe('#ffffff');
    expect(DEFAULT_LIGHT_INTENSITY).toBe(1);
  });

  it('環境光は旧 web 版と同じ 0.65・色は白（skybox の SH は単色で合わせられない）', () => {
    expect(AMBIENT_LIGHT).toBeCloseTo(0.65, 10);
    expect(DEFAULT_AMBIENT_COLOR).toBe('#ffffff');
  });

  it('既定では環境光と拡散の和が 1（写真の明るさをそのまま出す）', () => {
    // シェーダは ambient + intensity * (1 - ambient) * NdotL。NdotL = 1 の面で 1 になる。
    expect(AMBIENT_LIGHT + DEFAULT_LIGHT_INTENSITY * (1 - AMBIENT_LIGHT)).toBeCloseTo(1, 10);
  });

  // 旧実装は「距離 0.35〜3m」と「拡大 0.3〜5 倍」を別々にクランプしていた。周回半径へ畳んだ後も
  // 寄れる所・引ける所は同じ（0.35/5 〜 3/0.3）。
  it('周回半径の範囲は旧実装の 距離 × 拡大率 で届いた範囲のまま', () => {
    expect(MINIMUM_ORBIT_RADIUS_METERS).toBeCloseTo(0.35 / 5, 10);
    expect(MAXIMUM_ORBIT_RADIUS_METERS).toBeCloseTo(3 / 0.3, 10);
  });
});

// 色の空間は絵の見え方をそのまま決める。**ここが抜けると暗く・彩度が上がり、肌が赤く寄る** —
// 実際にそうなっていた（`ShaderMaterial` は組込みマテリアルと違って自分で戻さないと戻らない）。
describe('色の空間', () => {
  it('線形で計算した色を出力の色空間へ戻す（gl_FragColor への代入より後で）', () => {
    const assignment = FRAGMENT_SHADER.lastIndexOf('gl_FragColor =');
    const conversion = FRAGMENT_SHADER.indexOf('#include <colorspace_fragment>');
    expect(conversion).toBeGreaterThan(assignment);
  });

  it('平坦色は sRGB として受けて線形で渡す', () => {
    // 端は動かない。中間は sRGB の伝達関数ぶん下がる（0.5 → 0.2140）。
    expect(srgbBaseColor([0, 0, 0]).x).toBeCloseTo(0, 10);
    expect(srgbBaseColor([255, 255, 255]).x).toBeCloseTo(1, 10);
    const half = srgbBaseColor([128, 128, 128]);
    expect(half.x).toBeCloseTo(((128 / 255 + 0.055) / 1.055) ** 2.4, 6);
    expect(half.x).toBeLessThan(128 / 255);
    expect(half.w).toBe(1);

    // 口の中の色（Unity の _BaseColor）。灰色へ寄らず、赤が主のまま線形へ落ちる。
    const sock = srgbBaseColor([80, 37, 37]);
    expect(sock.x).toBeGreaterThan(sock.y);
    expect(sock.y).toBeCloseTo(sock.z, 10);
  });

  it('色の表記を線形へ直すのは setStyle が済ませる（二重に変換していない）', () => {
    // `ColorManagement.enabled` の既定が true で、作業色空間は線形。`convertSRGBToLinear` を
    // 続けて呼ぶと二重変換になる（白では気付けないが、パネルで色を選ぶと沈む）。
    const gray = new THREE.Color().setStyle('#808080');
    expect(gray.r).toBeCloseTo(((128 / 255 + 0.055) / 1.055) ** 2.4, 6);
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
  it('カメラの既定は Unity 側の値そのまま（注視点の正面 1.3m・正対）', () => {
    expect(DEFAULT_VIEW_SETTINGS.fovDegrees).toBe(DEFAULT_FOV_DEGREES);
    expect(DEFAULT_VIEW_SETTINGS.background).toBe(DEFAULT_BACKGROUND);
    const expected = cameraPoseAt([0, TARGET_HEIGHT_METERS, 0]);
    expect(DEFAULT_VIEW_SETTINGS.cameraPositionX).toBe(expected.position[0]);
    expect(DEFAULT_VIEW_SETTINGS.cameraPositionY).toBeCloseTo(TARGET_HEIGHT_METERS, 10);
    expect(DEFAULT_VIEW_SETTINGS.cameraPositionZ).toBeCloseTo(DEFAULT_ORBIT_RADIUS_METERS, 10);
    expect(DEFAULT_VIEW_SETTINGS.cameraPitchDegrees).toBe(0);
    expect(DEFAULT_VIEW_SETTINGS.cameraYawDegrees).toBe(0);
  });

  it('ライトの既定はビューアーの定数そのまま', () => {
    expect(DEFAULT_VIEW_SETTINGS.lightColor).toBe(DEFAULT_LIGHT_COLOR);
    expect(DEFAULT_VIEW_SETTINGS.lightIntensity).toBe(DEFAULT_LIGHT_INTENSITY);
    expect(DEFAULT_VIEW_SETTINGS.ambientColor).toBe(DEFAULT_AMBIENT_COLOR);
    expect(DEFAULT_VIEW_SETTINGS.ambient).toBe(AMBIENT_LIGHT);
  });

  it('起動時は無表情・正面・自動再生なし（まばたきだけ動く）', () => {
    expect(DEFAULT_VIEW_SETTINGS.playMode).toBe('off');
    expect(DEFAULT_VIEW_SETTINGS.headYawDegrees).toBe(0);
    expect(DEFAULT_VIEW_SETTINGS.headPitchDegrees).toBe(0);
    expect(DEFAULT_VIEW_SETTINGS.followPointer).toBe(false);
    expect(DEFAULT_VIEW_SETTINGS.blinkEnabled).toBe(true);
  });
});
