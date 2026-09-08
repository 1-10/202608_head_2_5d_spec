// 3D ビューだけが使うアセット（vertex group / ジョイント / 表情プリセット）。
//
// **この層は書き出しの契約に一切関わらない。** `application/exportGuest` はここの型を読まないし、
// guest zip にもここの値は入らない。あるのは「書き出した guest が Unity でどう出るか」を web でも
// 同じ形で見るためだけ。
//
// **正本は Unity 側（1-10/2607_Obayashi_Avatar_Mockup_3DGS の `Assets/Sandbox/Ooba/GNM`）** で、
// デスクトップ側（1-10/2608_Obayashi_GNMHeadExporter）の 3D ビューではない。guest を実際に組み立てて
// 画にするのは Unity なので、確認の絵が違うと「web で見て良かったが Unity で崩れる」が起きる。
// 領域の分け方・固定色・除外領域はあちらの `Editor/GnmHeadAssetBuilder` の設定と同じ値を持つ。
//
// 配列は `tools/export_gnm_assets.py` が GNMB へ詰める。閾値（重み > 1e-4）は生成時に適用済みなので、
// ここに来る `vertexGroups` は 0/1 だけ。

/** 3D ビュー用の配列。`GnmHeadAsset` とは別に持つ（書き出しの型を汚さない）。 */
export interface GnmPreviewAsset {
  /** 公式 npz の `vertex_group_names` そのまま。 */
  readonly vertexGroupNames: readonly string[];
  /** (group 数, 頂点数) の 0/1。閾値は生成時に適用済み。 */
  readonly vertexGroups: Uint8Array;
  /** 公式 npz の `joint_names`（`neck` / `head` / `left_eye` / `right_eye`）。 */
  readonly jointNames: readonly string[];
  /** 親の index。根は -1。 */
  readonly jointParentIndices: Int32Array;
  /** (ジョイント数, 3) GNM 空間・メートルの**絶対**位置。 */
  readonly templateJointPositions: Float32Array;
  /** (identity 成分数, ジョイント数, 3)。ジョイント位置は identity で動く（実測 6.9mm）。 */
  readonly jointIdentityBasis: Float32Array;
  /** (頂点数, 2) 影響ジョイントの index。v3_0 / head は最大 2 本。 */
  readonly skinJointIndices: Uint8Array;
  /** (頂点数, 2) 重み。和は 1。 */
  readonly skinJointWeights: Float32Array;
  /**
   * Unity 側 `Tools/GnmExpressionPresets_v3_0.npz` の `class_names` に、web 側で足した口形
   * （`viseme_*`）を後ろから連結したもの。**表情と口形はこの名前で切り分ける**（`preview/viseme`）。
   */
  readonly expressionPresetNames: readonly string[];
  /**
   * (プリセット数, 成分数) プリセットを表す**係数の行**。
   *
   * **変位に潰して持たない。** プリセットは元から「383 成分の係数の並び」で、潰すと GNM 本来の
   * 「基底 × 係数」という形が web の中だけで消える。係数のままなら Unity へそのまま持って行ける。
   */
  readonly expressionPresetCoefficients: Float32Array;
  /**
   * (成分数,) まばたきの係数（`wink_left` + `wink_right` を目の成分だけ残したもの）。
   *
   * **表情プリセットの 1 本として持たない。** まばたきは他の表情へ加算するのではなく、目の成分を
   * 置き換える（正本は旧 web 版 `gnmHeadMesh` のクロスフェード）。
   */
  readonly blinkCoefficients: Float32Array;
  /** まばたきが置き換える成分の区間（= 目領域の成分）。 */
  readonly blinkComponentOffset: number;
  readonly blinkComponentCount: number;
  /** 公式 npz の `expression_names`（383 本）。並びは領域ごとに連続。 */
  readonly expressionComponentNames: readonly string[];
  /** 成分ごとの量子化スケール。値 = q * scale / 32767 メートル。 */
  readonly expressionBasisScales: Float64Array;
  /** 領域の切り出し（成分の範囲と、その領域が動かす頂点の範囲）。 */
  readonly expressionBasisRegions: readonly ExpressionBasisRegion[];
  /** 領域ブロックを連結した split 空間の頂点 index。 */
  readonly expressionBasisVertices: Int32Array;
  /** 領域ブロックを連結した int16。領域ごとに **(頂点数, 成分数, 3)** の順。 */
  readonly expressionBasisQ: Int16Array;
  readonly vertexCount: number;
  readonly jointCount: number;
  readonly presetCount: number;
  /** 表情基底の成分数（383）。 */
  readonly componentCount: number;
}

/**
 * 表情基底の領域ブロック 1 つ。
 *
 * **成分が動かす頂点は自分の領域のまわりに限られる。** だから密に持たず、領域ごとに「動く頂点の
 * 一覧」と「その頂点だけのブロック」で持つ（実測の容量差はアセット生成が毎回表示する）。
 *
 * **領域は独立した部分空間ではない。** 左目と右目、目と下顔面、下顔面と舌は頂点を共有し、
 * 基底として見ても直交しない。**だから係数を領域ごとに分けて解いてはいけない** — 分けるのは
 * 容量のためだけで、解くときは全成分をまとめて 1 回で解く。
 */
export interface ExpressionBasisRegion {
  /** 公式の領域名（`lower_face_region` / `left_eye_region` / `tongue` など）。 */
  readonly name: string;
  /** この領域が持つ成分の先頭 index（`expressionComponentNames` の中で）。 */
  readonly componentOffset: number;
  readonly componentCount: number;
  /** この領域が動かす頂点の先頭 index（`expressionBasisVertices` の中で）。 */
  readonly vertexOffset: number;
  readonly vertexCount: number;
  /** この領域のブロックの先頭（`expressionBasisQ` の中で）。 */
  readonly quantizedOffset: number;
}

/**
 * 表情基底の値（メートル）。`slot` は領域内の頂点の並び順（`vertexOffset` からの相対）。
 *
 * 頂点 index そのものではなく slot で引くのは、領域ブロックが「動く頂点だけ」を詰めているため。
 * 対応する split 頂点は `expressionBasisVertices[region.vertexOffset + slot]`。
 */
export function expressionBasisValue(
  preview: GnmPreviewAsset,
  region: ExpressionBasisRegion,
  component: number,
  slot: number,
  axis: number,
): number {
  const local = component - region.componentOffset;
  if (local < 0 || local >= region.componentCount) {
    throw new Error(`成分 ${component} は領域 ${region.name} の外`);
  }
  const base = region.quantizedOffset + (slot * region.componentCount + local) * 3 + axis;
  return (preview.expressionBasisQ[base] * preview.expressionBasisScales[component]) / 32767;
}

/** プリセット 1 本の係数の行（長さ `componentCount`）。 */
export function presetRow(preview: GnmPreviewAsset, preset: number): Float64Array {
  if (preset < 0 || preset >= preview.presetCount) {
    throw new Error(`プリセット ${preset} は範囲外（0〜${preview.presetCount - 1}）`);
  }
  const base = preset * preview.componentCount;
  return Float64Array.from(
    preview.expressionPresetCoefficients.subarray(base, base + preview.componentCount),
  );
}

/**
 * 描画する層。表示の切り替えはこの単位。
 *
 * Unity 側に層の概念は無い（submesh を全部出す）。旧 web 版が持っていた「口腔内だけ消す」「髪だけ
 * 消す」は前後関係の崩れを切り分けるのに効くので、web 側の道具として残す。
 */
export const LAYER_SKIN = 'skin';
export const LAYER_EYES = 'eyes';
export const LAYER_MOUTH = 'mouth';
export const LAYER_HAIR = 'hair';

/** 表示を切る単位。並びは数字キーの順でもある。 */
export const LAYER_ORDER: readonly string[] = [LAYER_SKIN, LAYER_EYES, LAYER_MOUTH, LAYER_HAIR];

/** 領域が何で塗られるか。 */
export type PreviewRegionKind = 'skin_texture' | 'eye_left' | 'eye_right' | 'flat_color';

/** submesh 1 つに切り出す領域。`selector` の構文は公式 `vertex_group_mask` と同じ。 */
export interface PreviewRegion {
  readonly name: string;
  readonly layer: string;
  readonly kind: PreviewRegionKind;
  readonly selector: readonly string[];
  /** `flat_color` のときの色（sRGB 0〜255）。他の kind では使わない。 */
  readonly color: readonly [number, number, number];
}

/**
 * メッシュから外す領域。
 *
 * 角膜は写真テクスチャが無く、公式の可視化も描いていない。3 頂点すべてが角膜の三角形だけを外す
 * （Unity 側 `ClassifyTriangles` と同じ判定）。
 */
export const EXCLUDED_SELECTOR: readonly string[] = ['eye_exteriors'];

/**
 * 領域と固定色。**並びが分類の優先順（先勝ち）** で、Unity 側 `GnmHeadBuildSettings.regions` と同順。
 *
 * `MouthSock` を `Skin` より前に置くこと。`mouth_sock` は `skin` の部分集合なので後ろに置くと Skin が
 * 先に全部取る。`Skin` 側を `-mouth_sock` で引く形は使えない（skin と mouth_sock は連続面で、境界の
 * 三角形は mouth_sock 頂点を一部だけ含む。引くとどちらのマスクも 3 頂点揃わず未割り当てになる）。
 *
 * 固定色は Unity 側の Material が正本:
 * `MT_GnmMouthSock` / `MT_GnmTeeth` / `MT_GnmGums` / `MT_GnmTongue` の `_BaseColor`。
 * 歯・歯茎・舌をゲスト共通の固定色にした以上、口腔壁だけ写真の肌色に追随すると開口時に隣り合う面で
 * 色の決め方が割れる。だから口腔壁も専用色にする — **exporter は今も 0.7 × 顔の肌の平均色を
 * `skin_albedo` へ焼くが、この領域はそれを読まない**（Unity も読まない）。
 */
export const PREVIEW_REGIONS: readonly PreviewRegion[] = [
  {
    name: 'MouthSock',
    layer: LAYER_MOUTH,
    kind: 'flat_color',
    selector: ['mouth_sock'],
    color: [80, 37, 37],
  },
  { name: 'Skin', layer: LAYER_SKIN, kind: 'skin_texture', selector: ['skin'], color: [185, 145, 130] },
  { name: 'Teeth', layer: LAYER_MOUTH, kind: 'flat_color', selector: ['teeth'], color: [190, 164, 164] },
  { name: 'Gums', layer: LAYER_MOUTH, kind: 'flat_color', selector: ['gums'], color: [114, 53, 53] },
  { name: 'Tongue', layer: LAYER_MOUTH, kind: 'flat_color', selector: ['tongue'], color: [114, 53, 53] },
  {
    name: 'EyeLeft',
    layer: LAYER_EYES,
    kind: 'eye_left',
    selector: ['eye_interiors', '&left_eye'],
    color: [220, 220, 220],
  },
  {
    name: 'EyeRight',
    layer: LAYER_EYES,
    kind: 'eye_right',
    selector: ['eye_interiors', '&right_eye'],
    color: [220, 220, 220],
  },
];

/**
 * どの領域にも入らなかった三角形の色（マゼンタ）。
 *
 * Unity 側は未割り当てを Console のエラーで知らせるが、ブラウザでは 3D ビューが主役なので**画で**
 * 分かる形にする。この色が見えたら領域の設定かアセットが変わっている。
 */
export const UNASSIGNED_REGION: PreviewRegion = {
  name: 'Unassigned',
  layer: LAYER_SKIN,
  kind: 'flat_color',
  selector: [],
  color: [255, 0, 255],
};

/**
 * `vertex_group_mask` 構文を評価して頂点マスクを返す。
 *
 * 各要素は先頭に演算子 `|`（既定）/ `&` / `-`、続けて否定 `~` を置ける。
 * 例: `['eye_interiors', '&left_eye']` は eye_interiors ∩ left_eye。
 */
export function evaluateSelector(
  preview: GnmPreviewAsset,
  selector: readonly string[],
): Uint8Array {
  const mask = new Uint8Array(preview.vertexCount);
  for (const token of selector) {
    let name = token.trim();
    if (name.length === 0) continue;
    let operator = '|';
    if (name[0] === '|' || name[0] === '&' || name[0] === '-') {
      operator = name[0];
      name = name.slice(1);
    }
    let inverse = false;
    if (name[0] === '~') {
      inverse = true;
      name = name.slice(1);
    }
    const group = preview.vertexGroupNames.indexOf(name);
    if (group < 0) {
      throw new Error(
        `vertex group '${name}' がアセットに無い（あるのは: ${preview.vertexGroupNames.join(', ')}）`,
      );
    }
    const offset = group * preview.vertexCount;
    for (let vertex = 0; vertex < preview.vertexCount; vertex++) {
      const member = inverse
        ? preview.vertexGroups[offset + vertex] === 0
        : preview.vertexGroups[offset + vertex] !== 0;
      if (operator === '|') mask[vertex] = mask[vertex] !== 0 || member ? 1 : 0;
      else if (operator === '&') mask[vertex] = mask[vertex] !== 0 && member ? 1 : 0;
      else mask[vertex] = mask[vertex] !== 0 && !member ? 1 : 0;
    }
  }
  return mask;
}

/** 三角形を領域へ振り分けた結果。 */
export interface RegionClassification {
  /** `regions[i]` に入った三角形の index。 */
  readonly perRegion: readonly (readonly number[])[];
  readonly regions: readonly PreviewRegion[];
  /** 3 頂点すべてが除外領域だった三角形の数。 */
  readonly excludedCount: number;
}

/**
 * 三角形を `PREVIEW_REGIONS` へ振り分ける（先勝ち・3 頂点すべてがマスク内）。
 *
 * 3 頂点すべてが `EXCLUDED_SELECTOR` の三角形は落とす。どこにも入らなかった三角形は
 * `UNASSIGNED_REGION` へまとめる（黙って消すと「無いこと」に気付けない）。
 */
export function classifyTriangles(
  preview: GnmPreviewAsset,
  triangles: Uint32Array,
  triangleCount: number,
  regions: readonly PreviewRegion[] = PREVIEW_REGIONS,
): RegionClassification {
  const excluded = evaluateSelector(preview, EXCLUDED_SELECTOR);
  const masks = regions.map((region) => evaluateSelector(preview, region.selector));
  const perRegion: number[][] = regions.map(() => []);
  const unassigned: number[] = [];
  let excludedCount = 0;

  for (let triangle = 0; triangle < triangleCount; triangle++) {
    const first = triangles[triangle * 3];
    const second = triangles[triangle * 3 + 1];
    const third = triangles[triangle * 3 + 2];
    if (excluded[first] !== 0 && excluded[second] !== 0 && excluded[third] !== 0) {
      excludedCount++;
      continue;
    }
    let assigned = -1;
    for (let region = 0; region < regions.length; region++) {
      const mask = masks[region];
      if (mask[first] !== 0 && mask[second] !== 0 && mask[third] !== 0) {
        assigned = region;
        break;
      }
    }
    if (assigned < 0) unassigned.push(triangle);
    else perRegion[assigned].push(triangle);
  }

  if (unassigned.length === 0) {
    return { perRegion, regions, excludedCount };
  }
  return {
    perRegion: [...perRegion, unassigned],
    regions: [...regions, UNASSIGNED_REGION],
    excludedCount,
  };
}
