// 写真の左右の眼を、公式 GNM の眼球 UV へそのまま焼く。
//
// この段が行うのは座標変換だけである。虹彩・強膜・瞼を分類せず、まぶたの外側も含めて写真を
// 読む。そのため、瞼、髪による遮蔽、影、反射、左右差は写真に写った見た目のまま残る。反対側の
// 眼や肌色、手続き的な模様で置き換えない。
//
// **虹彩の大きさを公式に合わせない。** 以前は「写真の虹彩の縁」を UV 半径
// `IRIS_OUTER_RADIUS_UV` へ写していた。2 つを一致させると決めていたので、虹彩の大きさは誰でも
// 公式のサイズになり、**その人の虹彩の大きさが失われていた**（目の大きさは個人差が最も出る
// 場所）。今は半径の基準を眼球メッシュ自身に取り、相似変換で写真の画素へ直す。写真の虹彩が
// メッシュの limbus より大きければ強膜の帯に虹彩の色が乗り、小さければ虹彩の帯の外周に白目が
// 入る — どちらも写真どおりの見た目である。
//
// 由来をテクセルごとに残す。焼いた所と伸ばした所が絵から区別できないと、白目のまつ毛のような
// 問題が「写真の内容」なのか「こちらの処理」なのかを人が判断できない。

import { RgbImage } from '../contract';
import { Similarity2d } from '../gnm/fit';
import { GnmHeadMesh } from '../gnm/model';
import { provenancePaletteImage } from '../inspection';
import { PhotoRgb, linearToSrgb8Array, samplePhotoLinearAt, validatePhoto } from '../photo';
import {
  EyeUvGeometry,
  anatomicalAngleOf,
  assignEyeSides,
  limbusFraction,
} from './geometry';
import { EYE_GROUPS, eyeLandmarks, irisCenterAndRadius } from './landmarks';
import {
  EYE_SIDES,
  EYE_UV_CENTER,
  EyeSide,
  IRIS_OUTER_RADIUS_UV,
  PUPIL_RADIUS_UV,
  SCLERA_INNER_RADIUS_UV,
} from './layout';

/** 写真の画素をそのまま読んだテクセル。 */
export const PROVENANCE_PHOTO = 1;

/**
 * 眼球の裏側に回る UV。シルエット上の同じ方位の画素を伸ばしている。
 *
 * 面が裏側へ回るとその UV は写真のどの画素も指さない。別の色を作るのではなく縁を伸ばすのは
 * mipmap 用の縁拡張で、Unity が正面から見るぶんには出てこない。
 */
export const PROVENANCE_STRETCHED = 2;

/**
 * 投影先が写真の外だったテクセル。写真の縁の画素で埋めている。
 *
 * 顔が枠際に写った写真で出る。**以前は例外にしていた**が、眼球の外周（強膜の外側）は写真の外へ
 * 出やすく、絵の中身としては縁の 1 列だけの話なので、落とすより由来を残して続ける方が扱いやすい。
 * 虹彩の帯まで写真の外なら別（`bakeEyeAlbedos` が落とす）。
 */
export const PROVENANCE_OUTSIDE_PHOTO = 3;

/** `provenance` の値と表示名。内訳の表示と検査画像の凡例の正本。 */
export const PROVENANCE_NAMES: readonly [number, string][] = [
  [PROVENANCE_PHOTO, '写真'],
  [PROVENANCE_STRETCHED, '伸ばし'],
  [PROVENANCE_OUTSIDE_PHOTO, '写真外'],
];

/**
 * 検査画像の色分け。写真=緑 / 伸ばし=黄 / 写真外=赤。
 *
 * `domain/atlas` と同じ配色にしてある（写真は緑、作った色は暖色）。2 つの段で色の意味が違うと、
 * 検査画像を並べたときに読み替えが要る。
 */
export const PROVENANCE_INSPECTION_COLORS: readonly [number, [number, number, number]][] = [
  [PROVENANCE_PHOTO, [0, 200, 60]],
  [PROVENANCE_STRETCHED, [240, 210, 40]],
  [PROVENANCE_OUTSIDE_PHOTO, [220, 40, 40]],
];

/** 眼球テクスチャの由来を色分けした検査画像を作る。 */
export function provenanceInspectionImage(provenance: Uint8Array, size: number): RgbImage {
  return provenancePaletteImage(provenance, size, size, PROVENANCE_INSPECTION_COLORS);
}

/** 片側の写真由来眼球テクスチャ。 */
export interface EyeAlbedo {
  readonly side: EyeSide;
  /** (S, S, 3) sRGB。 */
  readonly image: Uint8Array;
  /** (S, S) テクセルごとの由来（`PROVENANCE_*`）。 */
  readonly provenance: Uint8Array;
  /** 写真に写った虹彩の半径（画素）。**測定値で、見た目には使わない**。 */
  readonly irisRadiusPx: number;
  /** メッシュの limbus を相似変換で写真の画素へ直した半径。半径の写像の基準はこちら。 */
  readonly limbusRadiusPx: number;
  readonly size: number;
}

/**
 * 写真の虹彩が公式の limbus より何倍大きいか（1.0 = 一致）。
 *
 * 以前この比を 1 に潰していた（虹彩の標準化）。**残しているのは、潰していないことを検査から
 * 確かめられるようにするため。**
 */
export function irisToLimbusRatio(albedo: EyeAlbedo): number {
  return albedo.irisRadiusPx / albedo.limbusRadiusPx;
}

/** 由来ごとのテクセル数。検査と報告に使う。 */
export function eyeProvenanceCounts(albedo: EyeAlbedo): Map<number, number> {
  const tally = new Map<number, number>();
  for (const [value] of PROVENANCE_NAMES) tally.set(value, 0);
  for (const value of albedo.provenance) tally.set(value, (tally.get(value) ?? 0) + 1);
  return tally;
}

/**
 * 写真の各眼を、対応する側の公式眼球 UV へ焼く。
 *
 * 片眼をもう片眼で代用しない。写真外を参照する入力は、別の見た目を捏造せず明示的に失敗させる。
 *
 * @param size 一辺（左右で同じ。契約が左右同じ一辺を要求する）。写真から逆算せず入口が選ぶ
 */
export function bakeEyeAlbedos(input: {
  photo: PhotoRgb;
  landmarks478: Float64Array;
  mesh: GnmHeadMesh;
  similarity: Similarity2d;
  geometries: Record<EyeSide, EyeUvGeometry>;
  size: number;
}): Record<EyeSide, EyeAlbedo> {
  validatePhoto(input.photo);

  const measured = new Map<string, { center: [number, number]; radius: number }>();
  for (const group of EYE_GROUPS) {
    measured.set(group.name, irisCenterAndRadius(eyeLandmarks(input.landmarks478, group)));
  }
  const centers: Record<string, readonly [number, number]> = {};
  for (const [name, value] of measured) centers[name] = value.center;

  const sides = assignEyeSides(input.mesh, input.similarity, centers);
  const result: Partial<Record<EyeSide, EyeAlbedo>> = {};
  for (const [name, side] of Object.entries(sides)) {
    const value = measured.get(name) as { center: [number, number]; radius: number };
    result[side] = bakeOneEye({
      photo: input.photo,
      side,
      geometry: input.geometries[side],
      similarity: input.similarity,
      irisCenter: value.center,
      irisRadiusPx: value.radius,
      size: input.size,
    });
  }
  for (const side of EYE_SIDES) {
    if (!result[side]) throw new Error(`${side} の眼球テクスチャが作れなかった`);
  }
  return result as Record<EyeSide, EyeAlbedo>;
}

/**
 * 片眼を焼く。
 *
 * 半径の基準は**メッシュの limbus を写真の画素へ直した長さ**。写真の虹彩半径は使わない（使うと
 * 虹彩の大きさが公式に揃ってしまう）。位置の基準は写真の虹彩中心 — 目が写真のどこにあるかは
 * 写真が正本で、フィットのずれをここへ持ち込まない。
 */
function bakeOneEye(input: {
  photo: PhotoRgb;
  side: EyeSide;
  geometry: EyeUvGeometry;
  similarity: Similarity2d;
  irisCenter: readonly [number, number];
  irisRadiusPx: number;
  size: number;
}): EyeAlbedo {
  const { photo, side, geometry, similarity, irisCenter, irisRadiusPx, size } = input;
  if (!Number.isFinite(irisRadiusPx) || irisRadiusPx <= 0) {
    throw new Error(`${side} の虹彩半径が正でない: ${irisRadiusPx}`);
  }
  const limbusRadiusPx = geometry.limbusRadiusMeters * similarity.scale;
  if (!Number.isFinite(limbusRadiusPx) || limbusRadiusPx <= 0) {
    throw new Error(`${side} の limbus 半径が正でない: ${limbusRadiusPx}`);
  }

  const { radius: radiusUv, angle: uvAngle } = uvPolarGrid(size);
  const rotation = similarity.rotation;
  const provenance = new Uint8Array(size * size);
  const colors = new Float32Array(size * size * 3);
  const outside = new Uint8Array(size * size);
  const sampleX = new Float64Array(size * size);
  const sampleY = new Float64Array(size * size);

  for (let texel = 0; texel < size * size; texel++) {
    // 眼球の裏側に回る UV と未使用余白は、シルエット上の同じ方位の写真画素を伸ばす。これは
    // mipmap 用の縁拡張であり、別の色や別眼を生成する処理ではない。
    const sampleRadius = Math.min(radiusUv[texel], geometry.silhouetteRadiusUv);
    const fraction = limbusFraction(geometry, sampleRadius);
    const anatomical = anatomicalAngleOf(geometry, uvAngle[texel]);
    const length = fraction * limbusRadiusPx;
    const frontalX = length * Math.cos(anatomical);
    const frontalY = length * Math.sin(anatomical);
    // `frontal @ rotation.T`（回転部分だけを掛ける。相似のスケールは length に入っている）。
    sampleX[texel] = irisCenter[0] + rotation[0] * frontalX + rotation[1] * frontalY;
    sampleY[texel] = irisCenter[1] + rotation[2] * frontalX + rotation[3] * frontalY;
    provenance[texel] =
      radiusUv[texel] > geometry.silhouetteRadiusUv ? PROVENANCE_STRETCHED : PROVENANCE_PHOTO;
  }

  let outsideCount = 0;
  for (let texel = 0; texel < size * size; texel++) {
    const inside = samplePhotoLinearAt(photo, sampleX[texel], sampleY[texel], colors, texel * 3);
    if (!inside) {
      outside[texel] = 1;
      outsideCount++;
    }
  }
  if (outsideCount > 0) {
    failIfTheIrisBandLeftThePhoto(side, outside, size);
    // 写真の縁へ寄せて引き直す。捏造はしないが、由来は残す。**外のテクセルだけ**引き直す。
    for (let texel = 0; texel < size * size; texel++) {
      if (outside[texel] === 0) continue;
      samplePhotoLinearAt(
        photo,
        Math.min(Math.max(sampleX[texel], 0), photo.width - 1e-6),
        Math.min(Math.max(sampleY[texel], 0), photo.height - 1e-6),
        colors,
        texel * 3,
      );
      provenance[texel] = PROVENANCE_OUTSIDE_PHOTO;
    }
  }

  return {
    side,
    image: linearToSrgb8Array(colors),
    provenance,
    irisRadiusPx,
    limbusRadiusPx,
    size,
  };
}

/**
 * 虹彩の帯が写真の外なら落とす。
 *
 * 強膜の外周が枠から出るのは写真の縁 1 列の話だが、**虹彩まで出ているなら眼が写真に写っていない**。
 * そこを縁の画素で埋めると、目の色を捏造したことになる。
 */
function failIfTheIrisBandLeftThePhoto(side: EyeSide, outside: Uint8Array, size: number): void {
  const band = irisBandMask(size);
  let count = 0;
  for (let texel = 0; texel < outside.length; texel++) {
    if (outside[texel] !== 0 && band[texel] !== 0) count++;
  }
  if (count > 0) {
    throw new Error(
      `${side} 眼の虹彩の帯が写真の外を ${count} テクセル参照した` +
        '（眼が写真の枠に入っていない）',
    );
  }
}

/** 各テクセル中心の UV 中心からの半径と角度。v は上向き。 */
export function uvPolarGrid(size: number): { radius: Float64Array; angle: Float64Array } {
  if (size < 2) throw new Error(`size が小さすぎる: ${size}`);
  const radius = new Float64Array(size * size);
  const angle = new Float64Array(size * size);
  for (let row = 0; row < size; row++) {
    const dv = -((row + 0.5) / size - EYE_UV_CENTER);
    for (let column = 0; column < size; column++) {
      const du = (column + 0.5) / size - EYE_UV_CENTER;
      radius[row * size + column] = Math.hypot(du, dv);
      angle[row * size + column] = Math.atan2(dv, du);
    }
  }
  return { radius, angle };
}

/** 強膜帯のテクセル。検査用。 */
export function scleraBandMask(size: number): Uint8Array {
  const { radius } = uvPolarGrid(size);
  const out = new Uint8Array(size * size);
  for (let texel = 0; texel < out.length; texel++) {
    out[texel] = radius[texel] >= SCLERA_INNER_RADIUS_UV ? 1 : 0;
  }
  return out;
}

/** 瞳孔の外から limbus までのテクセル。検査用。 */
export function irisBandMask(size: number): Uint8Array {
  const { radius } = uvPolarGrid(size);
  const out = new Uint8Array(size * size);
  for (let texel = 0; texel < out.length; texel++) {
    out[texel] =
      radius[texel] > PUPIL_RADIUS_UV && radius[texel] <= IRIS_OUTER_RADIUS_UV ? 1 : 0;
  }
  return out;
}
