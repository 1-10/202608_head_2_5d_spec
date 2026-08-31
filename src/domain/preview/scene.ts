// 3D 確認ビューへ渡すシーン。書き出しには含めない。
//
// **領域の分け方と固定色は Unity 側（1-10/2607_Obayashi_Avatar_Mockup_3DGS の
// `Assets/Sandbox/Ooba/GNM`）が正本**で、デスクトップ側の 3D ビューではない。guest を実際に組み立てて
// 画にするのは Unity なので、確認の絵が違うと「web で見て良かったが Unity で崩れる」が起きる。
// 分け方そのものは `preview/asset.PREVIEW_REGIONS`。
//
// **シーンは「そのまま描ける形」で渡す。** メッシュごとに自分の頂点配列（位置・UV）と、自分の配列を
// 指す三角形 index を持つ。頭部メッシュ全体の配列を共有して index で一部を指す形にしない — 領域ごとに
// 法線が変わる（同じ頂点でも隣り合う三角形が違う）ので、共有すると「どの領域として集約した法線か」が
// 曖昧になり、GPU へ送る量も領域数ぶん増える。
//
// **位置は毎フレーム変わる（首・視線・表情）ので、シーンは形と対応だけを持つ。** 頭部の各メッシュは
// split 頂点配列への index（`sourceVertices`）を持ち、フレームごとに `gatherPositions` で集める。
// 髪シェルだけは自分の配列を持ち、`head` ジョイントの剛体変換で追従する（髪は変形しない）。

import { AlphaImage, HairShell, RgbImage } from '../contract';
import { NormalPlan, planNormals } from './normals';
import { EyeSide } from '../eyes/layout';
import { GnmHeadMesh, unsplitVertexCount } from '../gnm/model';
import {
  GnmPreviewAsset,
  LAYER_HAIR,
  LAYER_ORDER,
  PreviewRegion,
  RegionClassification,
  classifyTriangles,
} from './asset';

/** 1 マテリアルで描くビュー用メッシュ。頂点配列は自分のぶんだけを持つ。 */
export interface PreviewMesh {
  readonly name: string;
  /** 表示を切る単位（`LAYER_ORDER` のいずれか）。 */
  readonly layer: string;
  /**
   * 頭部 split 頂点配列への index（長さ = このメッシュの頂点数）。
   *
   * 髪シェルだけ `null`。髪はスキニングを受けず、`head` ジョイントの剛体変換で追従する。
   */
  readonly sourceVertices: Uint32Array | null;
  /** bind 姿勢での位置。(頂点数, 3) GNM 空間・メートル。 */
  readonly restPositions: Float32Array;
  readonly uvs: Float32Array;
  /** (三角形数, 3) **このメッシュの頂点配列**への index。 */
  readonly triangles: Uint32Array;
  /** RGB。無ければ `baseColor` で塗る。 */
  readonly texture: RgbImage | null;
  /** sRGB 0〜255。テクスチャが無い領域の固定色（正本は Unity 側の Material）。 */
  readonly baseColor: readonly [number, number, number];
  /** テクスチャと同じ大きさの不透明度。**あれば半透明扱い**。 */
  readonly alpha: AlphaImage | null;
}

/**
 * 半透明パスで描くか。**不透明度のテクスチャを持つメッシュだけが半透明。**
 *
 * メッシュ名で判定しない（髪だけを名前で特別扱いすると、後で半透明の要素が増えたときに黙って
 * 不透明パスへ落ちる）。
 */
export function isTransparent(mesh: PreviewMesh): boolean {
  return mesh.alpha !== null;
}

/** 3D ビューへ渡すシーン。 */
export interface PreviewScene {
  readonly meshes: readonly PreviewMesh[];
  /**
   * 法線をどう作るか（どの頂点が実法線か・どの三角形を回すか）。
   *
   * **シーンが持つのは領域分けの結果だから。** ビューアー側で領域名から作り直すと、領域の定義が
   * 変わったときに黙ってズレる。
   */
  readonly normalPlan: NormalPlan;
  /** どの領域にも入らなかった三角形の数。0 でないなら領域設定かアセットが変わっている。 */
  readonly unassignedTriangleCount: number;
  /** 3 頂点すべてが角膜だった三角形の数（メッシュから外した分）。 */
  readonly excludedTriangleCount: number;
}

/** このシーンに実在する層を `LAYER_ORDER` の順で返す。 */
export function sceneLayerNames(scene: PreviewScene): string[] {
  const present = new Set(scene.meshes.map((mesh) => mesh.layer));
  return LAYER_ORDER.filter((layer) => present.has(layer));
}

/**
 * 隠す層を除いて、不透明パスと半透明パスに分ける。
 *
 * 半透明を先に描くと、その後ろに隠れる不透明が上書きしてしまう。不透明を先に描いて深度を埋め、
 * 半透明は深度書き込みを止めて重ねる。
 */
export function drawPasses(
  scene: PreviewScene,
  hidden: ReadonlySet<string> = new Set(),
): { opaque: PreviewMesh[]; transparent: PreviewMesh[] } {
  const visible = scene.meshes.filter((mesh) => !hidden.has(mesh.layer));
  return {
    opaque: visible.filter((mesh) => !isTransparent(mesh)),
    transparent: visible.filter((mesh) => isTransparent(mesh)),
  };
}

// カメラは Unity 側の固定値（FOV 20° / 距離 1.3m / 注視点 y=0.297m）で置くので、シーンの
// 境界からフレーミングを決める関数は持たない。**あちらと同じ絵にすることが目的**なので、
// こちらで勝手に枠へ合わせると「web では入っていたが Unity では切れる」が起きる。

/** split 頂点配列から、このメッシュの頂点だけを `out` へ集める（位置でも法線でも使う）。 */
export function gatherVertexVectors(
  mesh: PreviewMesh,
  values: Float64Array | Float32Array,
  out: Float32Array,
): void {
  const source = mesh.sourceVertices;
  if (source === null) throw new Error(`${mesh.name} は頭部メッシュではない`);
  if (out.length !== source.length * 3) {
    throw new Error(`${mesh.name} の出力配列が ${out.length}（期待 ${source.length * 3}）`);
  }
  for (let vertex = 0; vertex < source.length; vertex++) {
    const from = source[vertex] * 3;
    out[vertex * 3] = values[from];
    out[vertex * 3 + 1] = values[from + 1];
    out[vertex * 3 + 2] = values[from + 2];
  }
}

/**
 * 実法線を残す頂点を返す（1 = 実法線 / 0 = +Z 固定）。
 *
 * **順序が判断そのもの。** まず `flat_color` の領域（口腔内）の頂点を全部 1 にし、**その後で**
 * 写真を貼る領域の頂点を 0 に落とす。口腔壁は `skin` の部分集合で肌と頂点を共有するので、共有頂点は
 * 肌側が勝って +Z になる — 唇の内縁だけ実法線が残ると、開口時にそこへ筋状の陰影が出る。
 */
export function keepRealNormalMask(
  vertexCount: number,
  triangles: Uint32Array,
  classification: RegionClassification,
): Uint8Array {
  const keepReal = new Uint8Array(vertexCount);
  const mark = (flatColor: boolean, value: number): void => {
    classification.regions.forEach((region, index) => {
      if ((region.kind === 'flat_color') !== flatColor) return;
      for (const triangle of classification.perRegion[index]) {
        for (let corner = 0; corner < 3; corner++) {
          keepReal[triangles[triangle * 3 + corner]] = value;
        }
      }
    });
  };
  mark(true, 1);
  mark(false, 0);
  return keepReal;
}

/** bind 姿勢のフィット結果から確認用シーンを作る（形と対応だけ。位置は毎フレーム更新する）。 */
export function buildPreviewScene(input: {
  vertices: Float64Array;
  headMesh: GnmHeadMesh;
  preview: GnmPreviewAsset;
  skinAlbedo: RgbImage;
  eyeAlbedos: Readonly<Record<EyeSide, RgbImage>>;
  hair: HairShell | null;
  hairAlbedo: RgbImage | null;
  hairAlpha: AlphaImage | null;
}): PreviewScene {
  const { headMesh, preview } = input;
  const classification = classifyTriangles(preview, headMesh.triangles, headMesh.triangleCount);
  const meshes: PreviewMesh[] = [];

  classification.regions.forEach((region, index) => {
    const triangles = classification.perRegion[index];
    if (triangles.length === 0) return;
    meshes.push(compacted(region, headMesh, input.vertices, triangles, textureFor(region, input)));
  });

  if (input.hair !== null && input.hairAlbedo !== null) {
    meshes.push({
      name: 'hair_shell',
      layer: LAYER_HAIR,
      sourceVertices: null,
      restPositions: input.hair.positions,
      uvs: input.hair.uvs,
      triangles: input.hair.triangles,
      texture: input.hairAlbedo,
      baseColor: [45, 35, 30],
      alpha: input.hairAlpha,
    });
  }
  if (meshes.length === 0) throw new Error('確認用シーンにメッシュが無い');
  return {
    meshes,
    normalPlan: planNormals(
      headMesh.triangles,
      keepRealNormalMask(headMesh.vertexCount, headMesh.triangles, classification),
      unsplitVertexCount(headMesh),
    ),
    unassignedTriangleCount:
      classification.regions.length > 0 &&
      classification.regions[classification.regions.length - 1].name === 'Unassigned'
        ? classification.perRegion[classification.perRegion.length - 1].length
        : 0,
    excludedTriangleCount: classification.excludedCount,
  };
}

function textureFor(
  region: PreviewRegion,
  input: {
    skinAlbedo: RgbImage;
    eyeAlbedos: Readonly<Record<EyeSide, RgbImage>>;
  },
): RgbImage | null {
  if (region.kind === 'skin_texture') return input.skinAlbedo;
  if (region.kind === 'eye_left') return input.eyeAlbedos.left;
  if (region.kind === 'eye_right') return input.eyeAlbedos.right;
  return null;
}

/** 頭部メッシュ全体の配列から、この領域が使う頂点だけを抜いて index を振り直す。 */
function compacted(
  region: PreviewRegion,
  headMesh: GnmHeadMesh,
  vertices: Float64Array,
  triangleIndices: readonly number[],
  texture: RgbImage | null,
): PreviewMesh {
  const corners: number[] = [];
  for (const triangle of triangleIndices) {
    corners.push(
      headMesh.triangles[triangle * 3],
      headMesh.triangles[triangle * 3 + 1],
      headMesh.triangles[triangle * 3 + 2],
    );
  }
  const used = [...new Set(corners)].sort((first, second) => first - second);
  const remap = new Map<number, number>();
  used.forEach((vertex, index) => remap.set(vertex, index));
  const restPositions = new Float32Array(used.length * 3);
  const uvs = new Float32Array(used.length * 2);
  used.forEach((vertex, index) => {
    restPositions[index * 3] = vertices[vertex * 3];
    restPositions[index * 3 + 1] = vertices[vertex * 3 + 1];
    restPositions[index * 3 + 2] = vertices[vertex * 3 + 2];
    uvs[index * 2] = headMesh.vertexUvs[vertex * 2];
    uvs[index * 2 + 1] = headMesh.vertexUvs[vertex * 2 + 1];
  });
  return {
    name: region.name,
    layer: region.layer,
    sourceVertices: Uint32Array.from(used),
    restPositions,
    uvs,
    triangles: Uint32Array.from(corners, (vertex) => remap.get(vertex) as number),
    texture,
    baseColor: region.color,
    alpha: null,
  };
}
