// ブラウザ用 GNM 頭部アセット（GNMB content="head_asset"）の読込。
//
// **デスクトップ側（1-10/2608_Obayashi_GNMHeadExporter）はここで公式 npz を直接読む。** ブラウザは
// npz を読めないので、`tools/export_gnm_assets.py` が同じ値を GNMB へ詰め替えたものを読む。読んだ後の
// 型（`domain/gnm/model.GnmHeadAsset`）はあちらと同じで、**判断はアセット生成側にある**（何を読むか・
// 領域の作り方・密対応の作り方）。ここは詰め替えを解くだけ。
//
// identity 基底は int16 量子化のまま持つ（実数へ展開すると 56MB になる）。値へ戻すのは
// `domain/gnm/model.basisValue` / `verticesOf` の側。
//
// **`GnmModel`（アセット + 68 点定義 + 密対応を束ねる型）を持たない。** デスクトップ側は npz と
// `head_sparse_68.txt` が別のファイルから来るので「揃っていること」を型で保証する必要があるが、
// こちらは 1 つの GNMB に全部入っているので `GnmHeadAsset` がそのまま揃った組である。

import { ModelFileNotFoundError } from '../domain/errors';
import { GnmHeadAsset, GnmHeadMesh } from '../domain/gnm/model';
import { ExpressionBasisRegion, GnmPreviewAsset } from '../domain/preview/asset';
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

/** 公式 npz の `joint_names`。3D ビューの首と視線がこの並びに依存する。 */
export const EXPECTED_JOINT_NAMES: readonly string[] = [
  'neck',
  'head',
  'left_eye',
  'right_eye',
];

/**
 * 1 つの GNMB から出てくる組。
 *
 * `asset` は書き出しが使う値だけ（デスクトップ側の `GnmHeadAsset` と同じ型）、`preview` は 3D ビュー
 * だけが使う値。**分けてあるのは書き出しの型に確認用の道具を混ぜないため。**
 */
export interface GnmAssetBundle {
  readonly asset: GnmHeadAsset;
  readonly preview: GnmPreviewAsset;
}

/** GNMB を fetch して `GnmAssetBundle` にする。 */
export async function loadGnmAssetBundle(url = DEFAULT_ASSET_URL): Promise<GnmAssetBundle> {
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
  return parseGnmAssetBundle(await response.arrayBuffer(), url);
}

/** GNMB のバイト列を `GnmAssetBundle` にする（テストから直接呼べる形）。 */
export function parseGnmAssetBundle(
  buffer: ArrayBuffer,
  source = 'gnm_head.gnmb',
): GnmAssetBundle {
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

  const asset: GnmHeadAsset = {
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

  return { asset, preview: parsePreview(container, header, vertexCount, componentCount) };
}

/** 3D ビュー用の配列を取り出す。要素数はここで全部突き合わせる。 */
function parsePreview(
  container: ReturnType<typeof readGnmbContainer>,
  header: Record<string, unknown>,
  vertexCount: number,
  identityComponentCount: number,
): GnmPreviewAsset {
  const vertexGroupNames = requireStringArray(header, 'vertex_group_names');
  const jointNames = requireStringArray(header, 'joint_names');
  if (
    jointNames.length !== EXPECTED_JOINT_NAMES.length ||
    jointNames.some((name, index) => name !== EXPECTED_JOINT_NAMES[index])
  ) {
    throw new Error(
      `joint_names が ${jointNames.join(', ')}` +
        `（期待 ${EXPECTED_JOINT_NAMES.join(', ')}）。首と視線の割り当てが変わる`,
    );
  }
  const expressionPresetNames = requireStringArray(header, 'expression_preset_names');

  const vertexGroups = requireArray(container, 'vertexGroups', Uint8Array);
  const jointParentIndices = requireArray(container, 'jointParentIndices', Int32Array);
  const templateJointPositions = requireArray(container, 'templateJointPositions', Float32Array);
  const jointIdentityBasis = requireArray(container, 'jointIdentityBasis', Float32Array);
  const skinJointIndices = requireArray(container, 'skinJointIndices', Uint8Array);
  const skinJointWeights = requireArray(container, 'skinJointWeights', Float32Array);
  const expressionComponentNames = requireStringArray(header, 'expression_component_names');
  const expressionBasisScales = Float64Array.from(
    requireNumberArray(header, 'expression_basis_scales'),
  );
  if (expressionBasisScales.length !== expressionComponentNames.length) {
    throw new Error('expression_basis_scales と expression_component_names の数が合わない');
  }
  const expressionBasisRegions = parseExpressionBasisRegions(
    header,
    expressionComponentNames.length,
  );
  const expressionBasisVertices = requireArray(container, 'expressionBasisVertices', Int32Array);
  const expressionBasisQ = requireArray(container, 'expressionBasisQ', Int16Array);
  const expressionPresetCoefficients = requireArray(
    container,
    'expressionPresetCoefficients',
    Float32Array,
  );
  const blinkCoefficients = requireArray(container, 'blinkCoefficients', Float32Array);
  const blinkComponentOffset = requireNumber(header, 'blink_component_offset');
  const blinkComponentCount = requireNumber(header, 'blink_component_count');
  const componentCount = expressionComponentNames.length;
  if (blinkComponentOffset < 0 || blinkComponentOffset + blinkComponentCount > componentCount) {
    throw new Error(
      `まばたきが置き換える区間 ${blinkComponentOffset}〜` +
        `${blinkComponentOffset + blinkComponentCount} が成分数 ${componentCount} を超えている`,
    );
  }
  // **区間の外に係数が残っていないこと。** 残っていると「目の成分だけ置き換える」という設計が
  // 破れていて、口や舌がまばたきで動く。
  for (let component = 0; component < componentCount; component++) {
    const inside =
      component >= blinkComponentOffset &&
      component < blinkComponentOffset + blinkComponentCount;
    if (!inside && blinkCoefficients[component] !== 0) {
      throw new Error(
        `まばたきの係数が置き換え区間の外（成分 ${component} =` +
          ` ${blinkCoefficients[component]}）に出ている`,
      );
    }
  }

  const jointCount = jointNames.length;
  const presetCount = expressionPresetNames.length;
  for (const [name, actual, expected] of [
    ['vertexGroups', vertexGroups.length, vertexGroupNames.length * vertexCount],
    ['jointParentIndices', jointParentIndices.length, jointCount],
    ['templateJointPositions', templateJointPositions.length, jointCount * 3],
    ['jointIdentityBasis', jointIdentityBasis.length, identityComponentCount * jointCount * 3],
    ['skinJointIndices', skinJointIndices.length, vertexCount * 2],
    ['skinJointWeights', skinJointWeights.length, vertexCount * 2],
    [
      'expressionPresetCoefficients',
      expressionPresetCoefficients.length,
      presetCount * componentCount,
    ],
    ['blinkCoefficients', blinkCoefficients.length, componentCount],
    [
      'expressionBasisVertices',
      expressionBasisVertices.length,
      expressionBasisRegions.reduce((total, region) => total + region.vertexCount, 0),
    ],
    [
      'expressionBasisQ',
      expressionBasisQ.length,
      expressionBasisRegions.reduce(
        (total, region) => total + region.componentCount * region.vertexCount * 3,
        0,
      ),
    ],
  ] as const) {
    if (actual !== expected) {
      throw new Error(`${name} の要素数が ${actual}（期待 ${expected}）`);
    }
  }
  for (let joint = 0; joint < jointCount; joint++) {
    if (jointParentIndices[joint] >= joint) {
      throw new Error(
        `ジョイント ${jointNames[joint]} の親が後ろにある（親を先に並べる前提が崩れている）`,
      );
    }
  }
  for (const index of skinJointIndices) {
    if (index >= jointCount) throw new Error('skinJointIndices がジョイント数の範囲外を指している');
  }
  for (const index of expressionBasisVertices) {
    if (index < 0 || index >= vertexCount) {
      throw new Error(`expressionBasisVertices が頂点数の範囲外 (${index}) を指している`);
    }
  }

  return {
    vertexGroupNames,
    vertexGroups,
    jointNames,
    jointParentIndices,
    templateJointPositions,
    jointIdentityBasis,
    skinJointIndices,
    skinJointWeights,
    expressionPresetNames,
    expressionPresetCoefficients,
    blinkCoefficients,
    blinkComponentOffset,
    blinkComponentCount,
    expressionComponentNames,
    expressionBasisScales,
    expressionBasisRegions,
    expressionBasisVertices,
    expressionBasisQ,
    vertexCount,
    jointCount,
    presetCount,
    componentCount,
  };
}

/**
 * 表情基底の領域ブロックの切り出しを読む。
 *
 * **成分の範囲が隙間なく 0 から全成分を覆うことを検査する。** 領域は解くときの分割の単位なので、
 * 抜けや重なりがあると「解いていない成分」が黙って生まれる。
 */
function parseExpressionBasisRegions(
  header: Record<string, unknown>,
  componentCount: number,
): ExpressionBasisRegion[] {
  const value = header['expression_basis_regions'];
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error('GNMB header の expression_basis_regions が空か配列でない');
  }
  const regions = value.map((item, index) => {
    if (typeof item !== 'object' || item === null) {
      throw new Error(`expression_basis_regions[${index}] が object でない`);
    }
    const record = item as Record<string, unknown>;
    const numberOf = (key: string): number => {
      const found = record[key];
      if (typeof found !== 'number' || !Number.isInteger(found) || found < 0) {
        throw new Error(`expression_basis_regions[${index}] の ${key} が非負整数でない`);
      }
      return found;
    };
    const name = record['name'];
    if (typeof name !== 'string' || name.length === 0) {
      throw new Error(`expression_basis_regions[${index}] の name が文字列でない`);
    }
    return {
      name,
      componentOffset: numberOf('component_offset'),
      componentCount: numberOf('component_count'),
      vertexOffset: numberOf('vertex_offset'),
      vertexCount: numberOf('vertex_count'),
      quantizedOffset: numberOf('quantized_offset'),
    };
  });

  let expectedComponent = 0;
  let expectedVertex = 0;
  let expectedQuantized = 0;
  for (const region of regions) {
    if (region.componentOffset !== expectedComponent) {
      throw new Error(
        `領域 ${region.name} の成分が ${region.componentOffset} から始まっている` +
          `（期待 ${expectedComponent}）。領域の並びに隙間か重なりがある`,
      );
    }
    if (region.vertexOffset !== expectedVertex || region.quantizedOffset !== expectedQuantized) {
      throw new Error(`領域 ${region.name} のブロックの頭が連結の順と合わない`);
    }
    expectedComponent += region.componentCount;
    expectedVertex += region.vertexCount;
    expectedQuantized += region.componentCount * region.vertexCount * 3;
  }
  if (expectedComponent !== componentCount) {
    throw new Error(
      `領域が覆う成分が ${expectedComponent} 本（期待 ${componentCount}）。解けない成分が残る`,
    );
  }
  return regions;
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
