// テストが共有する GNM アセットの読み込み。
//
// `public/gnm/gnm_head.gnmb` は `tools/export_gnm_assets.py` が生成する実アセット。**実アセットで
// 測る**のがこのテスト群の要点で、合成データだと「アセットの前提が崩れたこと」を検出できない
// （デスクトップ側のテストも同じ方針）。

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { GnmHeadAsset } from '../src/domain/gnm/model';
import { parseGnmHeadAsset } from '../src/infrastructure/gnmAsset';

const ASSET_PATH = resolve(__dirname, '..', 'public', 'gnm', 'gnm_head.gnmb');

let cached: GnmHeadAsset | null = null;

/** アセットが無ければテストを落とす（生成し忘れを黙って通さない）。 */
export function loadAsset(): GnmHeadAsset {
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
  cached = parseGnmHeadAsset(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer,
  );
  return cached;
}
