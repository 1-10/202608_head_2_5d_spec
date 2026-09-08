// テストが共有する GNM アセットの読み込み。
//
// `public/gnm/gnm_head.gnmb` は `tools/export_gnm_assets.py` が生成する実アセット。**実アセットで
// 測る**のがこのテスト群の要点で、合成データだと「アセットの前提が崩れたこと」を検出できない
// （デスクトップ側のテストも同じ方針）。

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { GnmHeadAsset } from '../src/domain/gnm/model';
import { GnmPreviewAsset } from '../src/domain/preview/asset';
import {
  addExpression,
  addPresetCoefficients,
  zeroCoefficients,
} from '../src/domain/preview/expression';
import { GnmAssetBundle, parseGnmAssetBundle } from '../src/infrastructure/gnmAsset';

const ASSET_PATH = resolve(__dirname, '..', 'public', 'gnm', 'gnm_head.gnmb');

let cached: GnmAssetBundle | null = null;

/** アセットが無ければテストを落とす（生成し忘れを黙って通さない）。 */
export function loadBundle(): GnmAssetBundle {
  if (cached !== null) return cached;
  let bytes: Buffer;
  try {
    bytes = readFileSync(ASSET_PATH);
  } catch (error) {
    throw new Error(
      `${ASSET_PATH} が無い。python tools/fetch_gnm_assets.py && ` +
        `python tools/export_gnm_assets.py で生成してください（${String(error)}）`,
    );
  }
  cached = parseGnmAssetBundle(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer,
  );
  return cached;
}

/** 書き出しが使う側だけ。 */
export function loadAsset(): GnmHeadAsset {
  return loadBundle().asset;
}

/** 3D ビューが使う側だけ。 */
export function loadPreview(): GnmPreviewAsset {
  return loadBundle().preview;
}

/**
 * プリセット 1 本を重み 1 で当てたときの変位 (頂点数, 3)。
 *
 * アセットは変位を持たず**係数の行**だけを持つので、変位が見たいテストはここで基底へ当てる。
 */
export function presetDisplacement(preview: GnmPreviewAsset, name: string): Float64Array {
  const weights = new Float64Array(preview.presetCount);
  const index = preview.expressionPresetNames.indexOf(name);
  if (index < 0) throw new Error(`プリセット '${name}' がアセットに無い`);
  weights[index] = 1;
  const coefficients = zeroCoefficients(preview);
  addPresetCoefficients(preview, coefficients, weights);
  const displacement = new Float64Array(preview.vertexCount * 3);
  addExpression(preview, displacement, coefficients);
  return displacement;
}
