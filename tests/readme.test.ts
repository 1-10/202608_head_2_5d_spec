// README に写した値が `application/settings` とズレていないことを見る。
//
// README の表は**写し**なので、片方だけ動くと黙って嘘になる。デスクトップ側も同じ検査を持っている
// （`tests/test_cli_settings.py`）。ここが落ちたら README を直すか、写すのをやめること。

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  DEFAULT_SETTINGS,
  EYE_TEXTURE_SIZE_CHOICES,
  MAXIMUM_DISAGREEMENT_SCALE,
  MAXIMUM_IDENTITY_CLIP,
  MINIMUM_DISAGREEMENT_SCALE,
  MINIMUM_IDENTITY_CLIP,
  TEXTURE_SIZE_CHOICES,
} from '../src/application/settings';
import { PREVIEW_REGIONS } from '../src/domain/preview/asset';
import {
  GAZE_LIMIT_DEGREES,
  NECK_SHARE,
  PITCH_LIMIT_DEGREES,
  YAW_LIMIT_DEGREES,
} from '../src/domain/preview/pose';
import { FADE_SECONDS, HOLD_SECONDS } from '../src/domain/preview/expression';
import {
  DEFAULT_BACKGROUND,
  DEFAULT_DISTANCE_METERS,
  DEFAULT_FOV_DEGREES,
  TARGET_HEIGHT_METERS,
} from '../src/presentation/viewer';

const README = readFileSync(resolve(__dirname, '..', 'README.md'), 'utf-8');

describe('README の調整パラメータの表', () => {
  it('肌アトラスの一辺', () => {
    expect(README).toContain(
      `| 肌アトラスの一辺 | ${DEFAULT_SETTINGS.skinAtlasSize}（${TEXTURE_SIZE_CHOICES.join(' / ')}） |`,
    );
  });

  it('眼球テクスチャの一辺', () => {
    expect(README).toContain(
      `| 眼球テクスチャの一辺 | ${DEFAULT_SETTINGS.eyeTextureSize}` +
        `（${EYE_TEXTURE_SIZE_CHOICES.join(' / ')}） |`,
    );
  });

  it('髪テクスチャの長辺', () => {
    expect(README).toContain(
      `| 髪テクスチャの長辺 | ${DEFAULT_SETTINGS.hairTextureSize}` +
        `（${TEXTURE_SIZE_CHOICES.join(' / ')}） |`,
    );
  });

  it('事前分布の強さの倍率', () => {
    expect(README).toContain(
      `| 事前分布の強さの倍率 | ${DEFAULT_SETTINGS.disagreementScale.toFixed(1)}` +
        `（${MINIMUM_DISAGREEMENT_SCALE}〜${MAXIMUM_DISAGREEMENT_SCALE}。大きいほど平均顔寄り） |`,
    );
  });

  it('identity 係数の上限（既定は上限なし）', () => {
    expect(DEFAULT_SETTINGS.identityClip).toBeNull();
    expect(README).toContain(
      `| identity 係数の上限 | 上限なし（置くなら ${MINIMUM_IDENTITY_CLIP}〜${MAXIMUM_IDENTITY_CLIP}） |`,
    );
  });
});

describe('README の 3D ビューの表', () => {
  it('カメラ（Unity 側 MainCamera の写し）', () => {
    expect(README).toContain(
      `| 投影 | 透視 FOV ${DEFAULT_FOV_DEGREES}° / 距離 ${DEFAULT_DISTANCE_METERS}m |`,
    );
    expect(README).toContain(`| 背景 | \`${DEFAULT_BACKGROUND}\` |`);
  });

  it('注視点は頭部の中心（Unity の固定値と違えている理由まで書く）', () => {
    expect(README).toContain(
      `| 注視点 | **頭部の外接箱の中心**（あちらは眼の高さ y=${TARGET_HEIGHT_METERS}m 固定） |`,
    );
  });

  it('領域の並び（先勝ちなので順序そのものが仕様）', () => {
    expect(README).toContain(
      `（${PREVIEW_REGIONS.map((region) => `\`${region.name}\``).join(' → ')}）`,
    );
  });

  it('口腔内の固定色（Unity 側 Material の写し）', () => {
    const colorOf = (name: string): string => {
      const region = PREVIEW_REGIONS.find((item) => item.name === name);
      if (region === undefined) throw new Error(`領域 ${name} が無い`);
      return `(${region.color.join(',')})`;
    };
    expect(colorOf('Gums')).toBe(colorOf('Tongue'));
    expect(README).toContain(
      `歯 \`${colorOf('Teeth')}\` / 歯茎・舌 \`${colorOf('Gums')}\`` +
        ` / 口腔壁 \`${colorOf('MouthSock')}\``,
    );
  });

  it('可動域と表情（Unity 側 Viewer の写し）', () => {
    expect(README).toContain(
      `| 可動域 | 首 yaw ±${YAW_LIMIT_DEGREES}° / pitch ±${PITCH_LIMIT_DEGREES}°` +
        ` / 視線 ±${GAZE_LIMIT_DEGREES}°、首へ ${NECK_SHARE * 100}%・頭へ ${(1 - NECK_SHARE) * 100}% |`,
    );
    expect(README).toContain(
      `立ち上がり ${FADE_SECONDS}s / 保持 ${HOLD_SECONDS}s |`,
    );
  });
});
