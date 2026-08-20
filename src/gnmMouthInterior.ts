// 口腔内メッシュ (口腔壁・歯・歯茎・舌) と舌の姿勢駆動。
//
// ジオメトリはGNM Head公式アセットの mesh_component_names
// (skin / left_eye / right_eye / upper_teeth_and_gums / lower_teeth_and_gums / tongue)
// のうち口腔側そのまま。頂点位置はGNMの同一頂点配列を共有するので、identity基底・
// 残差ワープ・表情基底すべて頭部と同じ変形を受ける。
//
// 頭部表面とは別メッシュにする理由:
// - 写真に色情報が存在しない (入力は口を閉じた正面写真1枚)。頭部は「写真の陰影だけで
//   立体を見せる」ため頂点法線を+Z固定にしているが、口腔内は写真の陰影が無いので
//   実法線+ライティングでしか凹みに見えない。両者は法線の扱いが正反対で同居できない
//
// 顎の開閉は lower_face_region の表情基底が担う。GNM Headに顎ジョイントは無く
// (joint_names = neck / head / left_eye / right_eye)、口の開閉に使える機構は
// この表情基底だけ。実測でも下顎の歯茎・舌・口腔壁は最大5.9mm動き、上顎の歯は
// 変位0 = 解剖学的に正しい。

import * as THREE from 'three';
import type { GnmModel } from './gnmHead';

// パーツID (tools/export_gnm_assets.py が正本)
const PART_SOCK = 1;
const PART_TEETH = 2;
const PART_GUMS = 3;
const PART_TONGUE = 4;

// 写真の下唇色 (linear) に対する各パーツの反射率比。
// 口腔内の照明は写真に写っていないため、唯一の手掛かりである唇の色を基準に取る
// (唇と口腔粘膜は同じ組織なので反射率もほぼ同じ。露出・ホワイトバランスもそのまま
// 引き継げる)。頭部は法線+Z固定なので、視線を向いた面のライティング係数は
// 頭部の肌と一致する = 遮蔽1の面が写真の唇と同じ明るさで描かれる。
//
// 比率はGNM公式の可視化 visualization/vertex_colors.py の
// _VERTEX_GROUP_COLOR_MODIFIERS を踏襲する:
// - gums / tongue / mouth_sock は 肌色×0.7。ここでは基準を唇色にする (口腔粘膜は
//   唇と同じ組織なので、肌より唇の色の方が近い)
// - teeth は 肌色×0.6 + 白×0.4 = 肌の約1.5倍の輝度で大きく脱色。歯は粘膜ではなく
//   エナメル質なので、基準も唇ではなく肌色 (写真から測った平均)
const MUCOSA_GAIN = 0.7;
const TEETH_LUMA_GAIN = 1.5; // 肌に対する歯の輝度比 (公式の 0.6+0.4 に相当)
const TEETH_DESATURATE = 0.7; // 肌の色味を同輝度のニュートラルへ寄せる比 (公式の白オフセット相当)

// 遮蔽 (AO)。口腔内へ光が入る経路は唇の開口だけなので、暗さは反射率ではなく
// ほぼ全部この遮蔽で決まる。開口を半径Rの円窓とみると、奥行きdの点に届く
// 立体角の比は R²/(R²+d²) — 逆二乗の減衰。頂点色へ焼くので直接光も一緒に
// 落ちるが、開口から入らない直接光が奥へ届かないのは物理的に正しい
const AO_APERTURE = 0.12; // 開口の実効半径 (モデル空間。頭部の幅≈1.9)
const AO_FLOOR = 0.04; // 咽頭側の下限 (完全な黒は避ける)

export interface MouthInteriorBuild {
  mesh: THREE.Mesh;
  /** 表情適用後の頭部頂点配列 (相似変換済み) から位置を取り込み、法線を作り直す。 */
  update(headVertices: Float32Array): void;
  dispose(): void;
}

/**
 * 口腔内メッシュを作る。旧アセット (interiorTriangles無し) ではnull。
 * lipPhotoColor: 写真の下唇色 (linear空間)。粘膜 (口腔壁・歯茎・舌) の基準色。
 * skinPhotoColor: 写真の肌の平均色 (linear空間)。歯の基準色。nullなら唇色で代用。
 */
export function buildMouthInterior(
  model: GnmModel,
  headVertices: Float32Array,
  lipPhotoColor: THREE.Color,
  skinPhotoColor: THREE.Color | null,
  brightness: number,
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

  // AOの基準面 = 口腔内の最前面 (唇の内縁)。そこからの奥行きで暗くする
  let zFront = -Infinity;
  for (let i = 0; i < count; i++) zFront = Math.max(zFront, positions[i * 3 + 2]);

  const skin = skinPhotoColor ?? lipPhotoColor;
  const skinLuma = 0.2126 * skin.r + 0.7152 * skin.g + 0.0722 * skin.b;
  const teeth: [number, number, number] = [
    (skin.r + (skinLuma - skin.r) * TEETH_DESATURATE) * TEETH_LUMA_GAIN,
    (skin.g + (skinLuma - skin.g) * TEETH_DESATURATE) * TEETH_LUMA_GAIN,
    (skin.b + (skinLuma - skin.b) * TEETH_DESATURATE) * TEETH_LUMA_GAIN,
  ];
  const mucosa: [number, number, number] = [
    lipPhotoColor.r * MUCOSA_GAIN,
    lipPhotoColor.g * MUCOSA_GAIN,
    lipPhotoColor.b * MUCOSA_GAIN,
  ];
  const rim: [number, number, number] = [lipPhotoColor.r, lipPhotoColor.g, lipPhotoColor.b];
  const partColor = (id: number): [number, number, number] => {
    if (id === PART_TEETH) return teeth;
    if (id === PART_SOCK || id === PART_GUMS || id === PART_TONGUE) return mucosa;
    return rim; // 橋渡し頂点 (唇の内縁) は唇そのまま
  };

  const apertureSq = AO_APERTURE * AO_APERTURE;
  const colors = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const [r, g, b] = partColor(model.mouthPartId[map[i]] ?? 0);
    const d = Math.max(0, zFront - positions[i * 3 + 2]);
    const ao = AO_FLOOR + (1 - AO_FLOOR) * (apertureSq / (apertureSq + d * d));
    const k = ao * brightness;
    colors[i * 3] = r * k;
    colors[i * 3 + 1] = g * k;
    colors[i * 3 + 2] = b * k;
  }

  const geometry = new THREE.BufferGeometry();
  const posAttr = new THREE.BufferAttribute(positions, 3);
  geometry.setAttribute('position', posAttr);
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.setIndex(new THREE.BufferAttribute(index, 1));
  geometry.computeVertexNormals();

  // DoubleSide: 口腔壁・歯は閉じたシェルだが、頭部の内側という「本来見えない面」を
  // 見せる用途なので、巻き方向の想定違いで穴が空くリスクを取らない
  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.6,
    metalness: 0,
    side: THREE.DoubleSide,
  });

  const mesh = new THREE.Mesh(geometry, material);
  return {
    mesh,
    update(headVerts: Float32Array): void {
      copyPositions(headVerts);
      posAttr.needsUpdate = true;
      // 顎が開くと下顎側の面の向きが変わるため法線は毎フレーム作り直す
      geometry.computeVertexNormals();
    },
    dispose(): void {
      geometry.dispose();
      material.dispose();
    },
  };
}

export interface TongueDrive {
  /** 表情係数から顎の開き量を測り、舌成分の係数を書き込む (in-place)。 */
  apply(coeffs: Float32Array, amount: number): void;
}

/**
 * 舌の姿勢を「GNM公式の舌成分」で駆動する。
 *
 * GNMのneutralは口を閉じた姿勢なので、舌は口蓋に張り付いている。実際の口では
 * 開口時に舌が口蓋から離れて下顎の床へ落ちるが、GNMがそれを表現する機構は
 * tongue成分 (tongue_mean + tongue_000..) だけで、顎ジョイントは存在しない。
 * そして公式ExpressionSampler (CVAE) が出す舌成分の係数は全プリセットで実質0
 * (実測: |c|<=0.06 / 変位<=0.2mm) — 学習データに舌のアニメーションが無い。
 * 公式デモGUI (gnm_head_demo.ipynb) も舌成分を ±3 の手動スライダーとして
 * 露出しているだけで、顎との連動は持たない。
 *
 * よって「顎が開いたら舌を下げる」連動はこちら側の設計になる。係数は公式の機構
 * (tongue成分) のまま、量だけ顎の開き量に比例させる:
 * - 「下向き」の方向 = 舌頂点の平均yを最も下げる係数ベクトル (成分ごとの
 *   平均y変位から実測で決める。意味ラベルを推測しない)
 * - 顎の開き量 = 舌以外の成分が舌頂点を下げている量。公式surpriseプリセットの
 *   値を1.0の基準にする (振幅の基準も実測から取る)
 */
export function buildTongueDrive(
  model: GnmModel,
  openReferencePreset: readonly number[] | undefined,
): TongueDrive | null {
  const n = model.vertexCount;
  const tongue: number[] = [];
  for (let i = 0; i < n; i++) if (model.mouthPartId[i] === PART_TONGUE) tongue.push(i);
  const tongueComps: number[] = [];
  for (let i = 0; i < model.expressionCount; i++) {
    if ((model.expressionNames[i] ?? '').startsWith('tongue')) tongueComps.push(i);
  }
  if (tongue.length === 0 || tongueComps.length === 0) return null;

  // 成分ごとの「舌頂点の平均y変位」(係数1あたり、モデル空間)
  const dyPerComp = new Float32Array(model.expressionCount);
  for (let c = 0; c < model.expressionCount; c++) {
    const base = c * n * 3;
    let sum = 0;
    for (const v of tongue) sum += model.expressionBasisQ[base + v * 3 + 1];
    dyPerComp[c] = (sum / tongue.length) * (model.expressionScales[c] / 32767);
  }

  // 舌成分だけで最も下げる方向 (L2ノルム1)。振幅1で平均y変位 = -dirNorm
  const dir = tongueComps.map((c) => -dyPerComp[c]);
  const dirNorm = Math.hypot(...dir);
  if (dirNorm < 1e-9) return null;
  for (let k = 0; k < dir.length; k++) dir[k] /= dirNorm;

  const isTongue = new Uint8Array(model.expressionCount);
  for (const c of tongueComps) isTongue[c] = 1;
  /** 舌以外の成分が舌を下げている量 (顎の開き量の実測プロキシ)。 */
  const jawDrop = (coeffs: ArrayLike<number>): number => {
    let drop = 0;
    for (let c = 0; c < model.expressionCount; c++) {
      if (!isTongue[c]) drop -= (coeffs[c] ?? 0) * dyPerComp[c];
    }
    return drop;
  };
  const reference = openReferencePreset ? jawDrop(openReferencePreset) : 0;
  if (!(reference > 1e-9)) return null;

  return {
    apply(coeffs: Float32Array, amount: number): void {
      if (amount === 0) return;
      // 開き量は基準の2倍で打ち切る (Intensityを上げても舌が突き抜けないように)
      const open = Math.min(2, Math.max(0, jawDrop(coeffs) / reference));
      if (open === 0) return;
      const a = amount * open;
      for (let k = 0; k < tongueComps.length; k++) coeffs[tongueComps[k]] += a * dir[k];
    },
  };
}
