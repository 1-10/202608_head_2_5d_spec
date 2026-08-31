// 書き出しには含めない、任意の 3D デバッグシーン。
//
// Exporter の契約は identity 係数・テクスチャ・髪シェルだけであり、頭部頂点を出力しない。この型は
// 実行中にだけ存在し、presentation のビューアーへ渡した後に破棄される。
//
// **シーンは「そのまま描ける形」で渡す。** メッシュごとに自分の頂点配列（位置・UV）と、自分の配列を
// 指す三角形 index を持つ。頭部メッシュ全体の頂点配列を共有して index で一部を指す形にしない — 成分
// ごとに法線が変わる（同じ頂点でも隣り合う三角形が違う）ので、共有すると「どの成分として集約した
// 法線か」が曖昧になり、GPU へ送る量も成分数ぶん増える。
//
// **層（layer）で表示を切れるようにする。** 検査は「誰がこの画素を持っているか」を切り分ける作業
// なので、肌・眼球・口腔内・髪を個別に消せることが要る。層の分け方は GNM 頭部の部位の分け方そのもの
// なので、ビューアー側ではなくここに置く。

import { AlphaImage, HairShell, RgbImage } from './contract';
import { EYE_COMPONENT_NAMES, EYE_SIDES, EyeSide } from './eyes/layout';
import { eyeInteriorVertexMask } from './eyes/geometry';
import { GnmHeadMesh } from './gnm/model';

export const LAYER_SKIN = 'skin';
export const LAYER_EYES = 'eyes';
export const LAYER_MOUTH = 'mouth';
export const LAYER_HAIR = 'hair';

/** 表示を切る単位。描画順もこの順（不透明→半透明はパスで分ける）。 */
export const LAYER_ORDER: readonly string[] = [LAYER_SKIN, LAYER_EYES, LAYER_MOUTH, LAYER_HAIR];

/** 口腔内の構成要素。閉じた口では見えないので、見えたら前後関係が壊れている印。 */
export const MOUTH_COMPONENT_NAMES: readonly string[] = [
  'upper_teeth_and_gums',
  'lower_teeth_and_gums',
  'tongue',
];

/** 1 マテリアルで描くビュー用メッシュ。頂点配列は自分のぶんだけを持つ。 */
export interface DebugMesh {
  readonly name: string;
  /** 表示を切る単位（`LAYER_ORDER` のいずれか）。 */
  readonly layer: string;
  /** (頂点数, 3) GNM 空間・メートル。 */
  readonly positions: Float32Array;
  readonly uvs: Float32Array;
  /** (三角形数, 3) **このメッシュの頂点配列**への index。 */
  readonly triangles: Uint32Array;
  /** RGB。無ければ `baseColor` で塗る。 */
  readonly texture: RgbImage | null;
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
export function isTransparent(mesh: DebugMesh): boolean {
  return mesh.alpha !== null;
}

/** 3D ビューアーへ渡す一時シーン。 */
export interface DebugScene {
  readonly meshes: readonly DebugMesh[];
}

/** このシーンに実在する層を `LAYER_ORDER` の順で返す。 */
export function sceneLayerNames(scene: DebugScene): string[] {
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
  scene: DebugScene,
  hidden: ReadonlySet<string> = new Set(),
): { opaque: DebugMesh[]; transparent: DebugMesh[] } {
  const visible = scene.meshes.filter((mesh) => !hidden.has(mesh.layer));
  return {
    opaque: visible.filter((mesh) => !isTransparent(mesh)),
    transparent: visible.filter((mesh) => isTransparent(mesh)),
  };
}

/**
 * (中心, 境界球の半径) を返す。
 *
 * 半径は**箱の辺の長さではなく中心からの最大距離**。回転しても外接しない量が要るのは、正射影の深度
 * スケールと画面に収める倍率をこれで決めるから。
 */
export function sceneBounds(scene: DebugScene): { center: [number, number, number]; radius: number } {
  const low = [Infinity, Infinity, Infinity];
  const high = [-Infinity, -Infinity, -Infinity];
  for (const mesh of scene.meshes) {
    for (let vertex = 0; vertex < mesh.positions.length / 3; vertex++) {
      for (let axis = 0; axis < 3; axis++) {
        const value = mesh.positions[vertex * 3 + axis];
        if (value < low[axis]) low[axis] = value;
        if (value > high[axis]) high[axis] = value;
      }
    }
  }
  const center: [number, number, number] = [
    0.5 * (low[0] + high[0]),
    0.5 * (low[1] + high[1]),
    0.5 * (low[2] + high[2]),
  ];
  let radius = 0;
  for (const mesh of scene.meshes) {
    for (let vertex = 0; vertex < mesh.positions.length / 3; vertex++) {
      radius = Math.max(
        radius,
        Math.hypot(
          mesh.positions[vertex * 3] - center[0],
          mesh.positions[vertex * 3 + 1] - center[1],
          mesh.positions[vertex * 3 + 2] - center[2],
        ),
      );
    }
  }
  return { center, radius: Math.max(radius, 1e-6) };
}

/** 実行中のフィット結果から3D確認用シーンを作る。 */
export function buildDebugScene(input: {
  vertices: Float64Array;
  headMesh: GnmHeadMesh;
  skinAlbedo: RgbImage;
  eyeAlbedos: Readonly<Record<EyeSide, RgbImage>>;
  hair: HairShell | null;
  hairAlbedo: RgbImage | null;
  hairAlpha: AlphaImage | null;
}): DebugScene {
  const { headMesh } = input;
  const positions = Float32Array.from(input.vertices);
  const uvs = headMesh.vertexUvs;
  const triangles = headMesh.triangles;
  const meshes: DebugMesh[] = [];

  const addComponent = (
    name: string,
    layer: string,
    texture: RgbImage | null,
    base: readonly [number, number, number],
    vertexMask: Uint8Array | null,
  ): void => {
    const componentIndex = headMesh.componentNames.indexOf(name);
    if (componentIndex < 0) return;
    const picked: number[] = [];
    for (let triangle = 0; triangle < headMesh.triangleCount; triangle++) {
      let ok = true;
      for (let corner = 0; corner < 3; corner++) {
        const vertex = triangles[triangle * 3 + corner];
        if (headMesh.componentId[vertex] !== componentIndex) ok = false;
        if (vertexMask !== null && vertexMask[vertex] === 0) ok = false;
      }
      if (!ok) continue;
      picked.push(
        triangles[triangle * 3],
        triangles[triangle * 3 + 1],
        triangles[triangle * 3 + 2],
      );
    }
    if (picked.length === 0) return;
    meshes.push(compacted(name, layer, positions, uvs, picked, texture, base));
  };

  addComponent('skin', LAYER_SKIN, input.skinAlbedo, [185, 145, 130], null);
  for (const side of EYE_SIDES) {
    addComponent(
      EYE_COMPONENT_NAMES[side],
      LAYER_EYES,
      input.eyeAlbedos[side],
      [220, 220, 220],
      eyeInteriorVertexMask(headMesh, side),
    );
  }
  for (const name of MOUTH_COMPONENT_NAMES) {
    addComponent(name, LAYER_MOUTH, null, [125, 70, 70], null);
  }

  if (input.hair !== null && input.hairAlbedo !== null) {
    meshes.push({
      name: 'hair_shell',
      layer: LAYER_HAIR,
      positions: input.hair.positions,
      uvs: input.hair.uvs,
      triangles: input.hair.triangles,
      texture: input.hairAlbedo,
      baseColor: [45, 35, 30],
      alpha: input.hairAlpha,
    });
  }
  if (meshes.length === 0) throw new Error('デバッグシーンにメッシュが無い');
  return { meshes };
}

/** 頭部メッシュ全体の配列から、この成分が使う頂点だけを抜いて index を振り直す。 */
function compacted(
  name: string,
  layer: string,
  positions: Float32Array,
  uvs: Float32Array,
  triangles: readonly number[],
  texture: RgbImage | null,
  baseColor: readonly [number, number, number],
): DebugMesh {
  const used = [...new Set(triangles)].sort((first, second) => first - second);
  const remap = new Map<number, number>();
  used.forEach((vertex, index) => remap.set(vertex, index));
  const outPositions = new Float32Array(used.length * 3);
  const outUvs = new Float32Array(used.length * 2);
  used.forEach((vertex, index) => {
    outPositions[index * 3] = positions[vertex * 3];
    outPositions[index * 3 + 1] = positions[vertex * 3 + 1];
    outPositions[index * 3 + 2] = positions[vertex * 3 + 2];
    outUvs[index * 2] = uvs[vertex * 2];
    outUvs[index * 2 + 1] = uvs[vertex * 2 + 1];
  });
  return {
    name,
    layer,
    positions: outPositions,
    uvs: outUvs,
    triangles: Uint32Array.from(triangles, (vertex) => remap.get(vertex) as number),
    texture,
    baseColor,
    alpha: null,
  };
}
