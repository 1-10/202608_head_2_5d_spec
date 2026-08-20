// 口腔内メッシュ (口腔壁・歯・歯茎・舌)。
//
// このファイルの数値はすべて google/GNM (Apache-2.0) のソースからの移植で、
// 対応箇所をコメントに書く。写真から取るのは基準色だけ。
//
// 移植元と、そこから確認した事実:
// - gnm/shape/gnm_utils.py `_EXPRESSION_REGION_VERTEX_GROUP_MAP`
//     'lower_face_region': ['expression_basis_mouth_nose_ears',
//                           'lower_teeth_and_gums', 'tongue']
//   → 口の表情領域は「下顎の歯・歯茎」と「舌」を含み、上顎の歯は含まない。
//     実測も一致 (lower_face全150成分で upper_teeth_and_gums の変位0.000mm /
//     lower_teeth_and_gums 最大5.915mm)。下顎は顎関節まわりの剛体回転
//     (公式デモの係数で12.9°回転、剛体フィット残差RMS 0.53mm)
// - gnm/shape/gnm_common.py `vertex_positions_bind_pose`
//   → 頂点 = template + Σ(identity係数×基底) + Σ(表情係数×基底) の素の線形和。
//     こちらの実装と同一
// - gnm/shape/gnm_common.py `compute_pose_correctives`
//   → pose_features = R - I なので回転0で厳密に0
// - gnm/shape/gnm_common.py `linear_blend_skinning` + skinning_weightsの列和=1.0 (実測)
//   → 回転0・平行移動0でLBSは恒等写像
//   ⇒ 上記2点より、ジョイントを持たないこちらのパイプラインは
//     「回転0の公式パイプライン」と厳密一致する (公式デモの係数で頂点差0.000mm)
// - gnm/shape/visualization/vertex_colors.py `_VERTEX_GROUP_COLOR_MODIFIERS`
//   → 口腔内パーツの色は肌色に対する scale/offset (下記 OFFICIAL_COLOR_MODIFIERS)
// - gnm/shape/visualization/gnm_pyrender.py
//   → マテリアルは MetallicRoughnessMaterial(metallicFactor=0.0, roughnessFactor=1.0)、
//     法線は毎フレーム compute_vertex_normals、パーツごとに別プリミティブ
//
// 頭部表面と別メッシュにする理由: 頭部は「写真の陰影だけで立体を見せる」ため頂点法線を
// +Z固定にしている。口腔内は写真に陰影が無いので実法線で描く必要があり、法線の扱いが
// 正反対で同居できない。公式もパーツごとに別プリミティブなので構成としても一致する。

import * as THREE from 'three';
import type { GnmModel } from './gnmHead';

// パーツID。tools/export_gnm_assets.py が正本で、値は公式の頂点グループ名に対応する
const PART_SOCK = 1; // mouth_sock
const PART_TEETH = 2; // teeth
const PART_GUMS = 3; // gums
const PART_TONGUE = 4; // tongue

/**
 * 公式 visualization/vertex_colors.py の `_VERTEX_GROUP_COLOR_MODIFIERS` の該当分。
 * 肌色 color に対して `color * scale + offset` で各パーツの色を決める。
 * 公式の定数は「Color Pickerへ貼れるUINT8値」なので非線形(sRGB)空間の値として扱う。
 */
const OFFICIAL_COLOR_MODIFIERS: Record<number, { scale: number; offset: number }> = {
  [PART_SOCK]: { scale: 0.7, offset: 0.0 },
  [PART_TEETH]: { scale: 0.6, offset: 0.4 },
  [PART_GUMS]: { scale: 0.7, offset: 0.0 },
  [PART_TONGUE]: { scale: 0.7, offset: 0.0 },
};
// 公式の 'skin' は (1.0, 0.0) = 肌色そのまま。橋渡し三角形で入ってくる唇の内縁がこれ
const SKIN_MODIFIER = { scale: 1.0, offset: 0.0 };

/**
 * 開口した口の中で舌が収まる姿勢。公式デモGIF (gnm/shape/assets/readme/gnm_head_demo.gif)
 * のExpressionタブ tongue列で実際に操作されているスライダー値そのもの
 * (e350..e353 = tongue_mean, tongue_000..002)。
 *
 * 適用すると舌が奥へ5.3mm・上へ3.4mm動く (実測。頂点最大23.3mm=局所的な変形を伴う)。
 * 口を閉じたままでも安全: 舌→口腔壁の最短距離 0.92mm→0.90mm、
 * 舌→上顎 1.27mm→1.90mm で、潜り込みは起きない (実測)。
 *
 * GNM Headに顎ジョイントは無く (joint_names = neck/head/left_eye/right_eye)、
 * 舌を動かす機構はこの表情成分だけ。公式デモは手動スライダーなので顎の開閉とは
 * 連動しない — ここでも連動させない (振幅1.0 = 公式デモと同じ姿勢)。
 */
const OFFICIAL_TONGUE_POSE: Record<string, number> = {
  tongue_mean: 0.7,
  tongue_000: -1.7,
  tongue_001: 0,
  tongue_002: 0,
};

export interface MouthInteriorBuild {
  mesh: THREE.Mesh;
  /** 表情適用後の頭部頂点配列 (相似変換済み) から位置を取り込み、法線を作り直す。 */
  update(headVertices: Float32Array): void;
  dispose(): void;
}

/**
 * 口腔内メッシュを作る。旧アセット (interiorTriangles無し) ではnull。
 * skinPhotoColor: 写真から測った肌の平均色 (linear空間)。公式の色式の基準色 `color`。
 * fallbackColor: セグメンテーションが無く肌色を測れないときの代用 (写真の唇色)。
 */
export function buildMouthInterior(
  model: GnmModel,
  headVertices: Float32Array,
  skinPhotoColor: THREE.Color | null,
  fallbackColor: THREE.Color,
): MouthInteriorBuild | null {
  const tris = model.interiorTriangles;
  if (tris.length === 0) return null;

  // 口腔内の三角形が参照する頂点だけを詰め直す (橋渡し三角形経由で唇の内縁も入る)
  const localOf = new Int32Array(model.vertexCount).fill(-1);
  const globalOf: number[] = [];
  const index = new Uint32Array(tris.length);
  for (let k = 0; k < tris.length; k++) {
    const g = tris[k];
    if (localOf[g] < 0) {
      localOf[g] = globalOf.length;
      globalOf.push(g);
    }
    index[k] = localOf[g];
  }
  const count = globalOf.length;
  const map = new Uint32Array(globalOf);

  const positions = new Float32Array(count * 3);
  const copyPositions = (src: Float32Array): void => {
    for (let i = 0; i < count; i++) {
      const g = map[i] * 3;
      positions[i * 3] = src[g];
      positions[i * 3 + 1] = src[g + 1];
      positions[i * 3 + 2] = src[g + 2];
    }
  };
  copyPositions(headVertices);

  // 公式の色式は非線形(sRGB)空間の値で書かれているので、そこへ戻して計算し、
  // three.jsの頂点色 (linear空間) へ変換して戻す
  const base = (skinPhotoColor ?? fallbackColor).clone().convertLinearToSRGB();
  const modified = new THREE.Color();
  const partColor = (id: number): THREE.Color => {
    const { scale, offset } = OFFICIAL_COLOR_MODIFIERS[id] ?? SKIN_MODIFIER;
    return modified
      .setRGB(
        Math.min(1, base.r * scale + offset),
        Math.min(1, base.g * scale + offset),
        Math.min(1, base.b * scale + offset),
        THREE.SRGBColorSpace,
      );
  };

  const colors = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const c = partColor(model.mouthPartId[map[i]] ?? 0);
    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }

  const geometry = new THREE.BufferGeometry();
  const posAttr = new THREE.BufferAttribute(positions, 3);
  geometry.setAttribute('position', posAttr);
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.setIndex(new THREE.BufferAttribute(index, 1));
  geometry.computeVertexNormals();

  // 公式 gnm_pyrender.py の MetallicRoughnessMaterial(metallicFactor=0, roughnessFactor=1)。
  // FrontSide: 巻き方向は表面が「肉の外側」向き = 口腔壁の面法線は空洞側を向く
  // (実測: mouth_sockの面法線が空洞中心を向く割合 100%)。空洞を覗くと表面が見える
  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 1.0,
    metalness: 0.0,
  });

  const mesh = new THREE.Mesh(geometry, material);
  return {
    mesh,
    update(headVerts: Float32Array): void {
      copyPositions(headVerts);
      posAttr.needsUpdate = true;
      // 公式も毎フレーム compute_vertex_normals を呼ぶ (顎が開くと面の向きが変わる)
      geometry.computeVertexNormals();
    },
    dispose(): void {
      geometry.dispose();
      material.dispose();
    },
  };
}

/**
 * 舌の表情係数へ公式デモの姿勢を書き込む (in-place)。
 * amount: 0=GNMのneutral姿勢そのまま / 1=公式デモのスライダー姿勢そのまま。
 */
export function applyOfficialTonguePose(model: GnmModel, coeffs: Float32Array, amount: number): void {
  if (amount === 0) return;
  for (let i = 0; i < model.expressionCount; i++) {
    const v = OFFICIAL_TONGUE_POSE[model.expressionNames[i] ?? ''];
    if (v) coeffs[i] += v * amount;
  }
}
