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
