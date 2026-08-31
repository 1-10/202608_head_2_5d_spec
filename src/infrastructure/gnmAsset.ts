// ブラウザ用 GNM 頭部アセット（GNMB content="head_asset"）の読込。
//
// **デスクトップ側（1-10/2608_Obayashi_GNMHeadExporter）はここで公式 npz を直接読む。** ブラウザは
// npz を読めないので、`tools/export_gnm_assets.py` が同じ値を GNMB へ詰め替えたものを読む。読んだ後の
// 型（`domain/gnm/model.GnmHeadAsset`）はあちらと同じで、**判断はアセット生成側にある**（何を読むか・
// 領域の作り方・密対応の作り方）。ここは詰め替えを解くだけ。
//
// identity 基底は int16 量子化のまま持つ（実数へ展開すると 56MB になる）。値へ戻すのは
// `domain/gnm/model.basisValue` / `verticesOf` の側。

import { ModelFileNotFoundError } from '../domain/errors';
import { GnmHeadAsset, GnmHeadMesh } from '../domain/gnm/model';
import { GNMB_CONTENT_HEAD_ASSET, readGnmbContainer, requireArray } from './gnmb';

/** 既定の配置先（`tools/export_gnm_assets.py` の出力先と揃える）。 */
export const DEFAULT_ASSET_URL = 'gnm/gnm_head.gnmb';

/**
 * per-vertex UV 化後の頂点数（v3_0 / head の実測値: 元 17,821 + 複製 616）。
 *
 * split の実装が壊れたことを検出するための固定値。アセットのバージョンが上がってこの数が変わったなら、
 * 変わったこと自体に気付いてから更新する。
 */
export const EXPECTED_SPLIT_VERTEX_COUNT = 18437;

/** `componentId` の値の意味。アセットが持つ名前がこれと一致することを読込時に検証する。 */
export const EXPECTED_MESH_COMPONENT_NAMES: readonly string[] = [
  'skin',
  'left_eye',
  'right_eye',
  'upper_teeth_and_gums',
  'lower_teeth_and_gums',
  'tongue',
];

/** GNMB を fetch して `GnmHeadAsset` にする。 */
export async function loadGnmHeadAsset(url = DEFAULT_ASSET_URL): Promise<GnmHeadAsset> {
  let response: Response;
  try {
    response = await fetch(url);
  } catch (error) {
    throw new ModelFileNotFoundError(
      `GNM アセットを取得できません（${url}）: ${String(error)}。` +
        ' python tools/export_gnm_assets.py で生成してください。',
    );
  }
  if (!response.ok) {
    throw new ModelFileNotFoundError(
      `GNM アセットを取得できません（${response.status} ${url}）。` +
        ' python tools/export_gnm_assets.py で public/gnm/gnm_head.gnmb を生成してください。',
    );
  }
  return parseGnmHeadAsset(await response.arrayBuffer(), url);
}

/** GNMB のバイト列を `GnmHeadAsset` にする（テストから直接呼べる形）。 */
export function parseGnmHeadAsset(buffer: ArrayBuffer, source = 'gnm_head.gnmb'): GnmHeadAsset {
  const container = readGnmbContainer(buffer, GNMB_CONTENT_HEAD_ASSET);
  const header = container.header;

  const gnmVersion = requireString(header, 'gnm_version');
  const gnmVariant = requireString(header, 'gnm_variant');
  const componentNames = requireStringArray(header, 'component_names');
  if (
    componentNames.length !== EXPECTED_MESH_COMPONENT_NAMES.length ||
    componentNames.some((name, index) => name !== EXPECTED_MESH_COMPONENT_NAMES[index])
  ) {
    throw new Error(
      `component_names が ${componentNames.join(', ')}` +
        `（期待 ${EXPECTED_MESH_COMPONENT_NAMES.join(', ')}）。componentId の意味が変わる`,
    );
  }
  const identityScales = Float64Array.from(requireNumberArray(header, 'identity_basis_scales'));
  const denseEdgeMeters = requireNumber(header, 'dense_edge_meters');

  const templateVertexPositions = requireArray(container, 'templateVertexPositions', Float32Array);
  const vertexUvs = requireArray(container, 'vertexUvs', Float32Array);
  const triangles = requireArray(container, 'triangles', Uint32Array);
  const uvSplitSource = requireArray(container, 'uvSplitSource', Uint32Array);
  const componentId = requireArray(container, 'componentId', Uint8Array);
  const earRegion = requireArray(container, 'earRegion', Uint8Array);
  const atlasPhotoOnlyRegion = requireArray(container, 'atlasPhotoOnlyRegion', Uint8Array);
  const mouthRimRegion = requireArray(container, 'mouthRimRegion', Float32Array);
  const identityBasisQ = requireArray(container, 'identityBasisQ', Int16Array);

  const vertexCount = uvSplitSource.length;
  if (vertexCount !== EXPECTED_SPLIT_VERTEX_COUNT) {
    throw new Error(
      `per-vertex UV 化後の頂点数が ${vertexCount}（期待 ${EXPECTED_SPLIT_VERTEX_COUNT}）。` +
        ' split の実装かアセットのどちらかが変わっている',
    );
  }
  for (const [name, array, expected] of [
    ['templateVertexPositions', templateVertexPositions, vertexCount * 3],
    ['vertexUvs', vertexUvs, vertexCount * 2],
    ['componentId', componentId, vertexCount],
    ['earRegion', earRegion, vertexCount],
    ['atlasPhotoOnlyRegion', atlasPhotoOnlyRegion, vertexCount],
    ['mouthRimRegion', mouthRimRegion, vertexCount],
  ] as const) {
    if (array.length !== expected) {
      throw new Error(`${name} の要素数が ${array.length}（期待 ${expected}）`);
    }
  }
  if (triangles.length % 3 !== 0) throw new Error('triangles の要素数が 3 の倍数でない');
  const triangleCount = triangles.length / 3;
  for (const index of triangles) {
    if (index >= vertexCount) throw new Error('triangles が split 空間の範囲外を指している');
  }
  // `uvSplitSource` の単調非減少は `splitIndexOf` の二分探索が依存する契約。
  for (let vertex = 1; vertex < vertexCount; vertex++) {
    if (uvSplitSource[vertex] < uvSplitSource[vertex - 1]) {
      throw new Error(`uvSplitSource が単調非減少でない（${vertex} 番目）`);
    }
  }
  let maximumComponent = 0;
  for (const value of componentId) if (value > maximumComponent) maximumComponent = value;
  if (maximumComponent >= componentNames.length) {
    throw new Error(`componentId が ${maximumComponent} で構成要素の数を超えている`);
  }

  const componentCount = identityScales.length;
  if (identityBasisQ.length !== componentCount * vertexCount * 3) {
    throw new Error(
      `identityBasisQ の要素数が ${identityBasisQ.length}` +
        `（期待 ${componentCount * vertexCount * 3}）`,
    );
  }

  const mesh: GnmHeadMesh = {
    templateVertexPositions,
    vertexUvs,
    triangles,
    uvSplitSource,
    componentId,
    componentNames,
    earRegion,
    atlasPhotoOnlyRegion,
    mouthRimRegion,
    vertexCount,
    triangleCount,
  };

  const denseMediapipeIndices = requireArray(container, 'denseMediapipeIndices', Uint16Array);
  const denseVertexIndices = requireArray(container, 'denseVertexIndices', Int32Array);
  const denseWeights = requireArray(container, 'denseWeights', Float32Array);
  const denseResidualMeters = requireArray(container, 'denseResidualMeters', Float32Array);
  const densePointCount = denseMediapipeIndices.length;
  if (
    denseVertexIndices.length !== densePointCount * 3 ||
    denseWeights.length !== densePointCount * 3 ||
    denseResidualMeters.length !== densePointCount
  ) {
    throw new Error('密対応の配列の要素数が揃っていない');
  }

  return {
    source: `${source} (${requireString(header, 'source')})`,
    gnmVersion,
    gnmVariant,
    mesh,
    vertexIdentityBasis: {
      quantized: identityBasisQ,
      scales: identityScales,
      componentCount,
      vertexCount,
    },
    landmarks: {
      vertexIndices: requireArray(container, 'sparse68VertexIndices', Int32Array),
      weights: requireArray(container, 'sparse68Weights', Float32Array),
    },
    dense: {
      mediapipeIndices: Int32Array.from(denseMediapipeIndices),
      vertexIndices: denseVertexIndices,
      weights: denseWeights,
      residualMeters: denseResidualMeters,
      edgeMeters: denseEdgeMeters,
      pointCount: densePointCount,
    },
  };
}

function requireString(header: Record<string, unknown>, key: string): string {
  const value = header[key];
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`GNMB header の ${key} が文字列でない`);
  }
  return value;
}

function requireNumber(header: Record<string, unknown>, key: string): number {
  const value = header[key];
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`GNMB header の ${key} が数でない`);
  }
  return value;
}

function requireStringArray(header: Record<string, unknown>, key: string): string[] {
  const value = header[key];
  if (!Array.isArray(value) || value.some((item) => typeof item !== 'string')) {
    throw new Error(`GNMB header の ${key} が文字列の配列でない`);
  }
  return value as string[];
}

function requireNumberArray(header: Record<string, unknown>, key: string): number[] {
  const value = header[key];
  if (!Array.isArray(value) || value.some((item) => typeof item !== 'number')) {
    throw new Error(`GNMB header の ${key} が数の配列でない`);
  }
  return value as number[];
}
