// 髪シェルが覆う対象のマスクを、クラス別の確信度から組み立てる。
//
// セグメンタが返すのは 6 クラスの確信度で、髪シェルが覆いたいのは **髪（class 1）と、頭に載って
// いる装飾品（class 5）**。帽子とメガネを髪と同じ殻に乗せたいのでこの 2 つを足すが、**装飾品は
// そのまま足せない**。
//
// 装飾品クラスは頭の外でも当たる
// ------------------------------
// class 5 は「帽子・メガネ」だけでなく、**バッグの肩紐・ネックレス・服の装飾**にも当たる。足した
// だけだと、髪シェルの格子が髪マスクの bbox から張られる関係で、胸を斜めに走る肩紐の上まで殻が
// 伸びる（実測: 門を入れると髪シェルの頂点が 7.0% 減り、減った分は首から下に生えていた殻）。
//
// **顎より下の装飾品を落とす。** 帽子もメガネも顎より上にあるので、落として困るものが無い。髪
// そのものは落とさない — ロングヘアは顎より下へ続く。
//
// 違うのは**フェードの幅を画像の高さではなく顔幅で測る**こと。画像の高さで測ると、同じ人物を引きで
// 撮るか寄りで撮るかで落とす範囲が変わる（`domain/gnm/crop` が切り出しで踏んだのと同じ罠）。
//
// 主役以外の人の髪も同じ 1 枚に入る
// ----------------------------------
// セグメンタはセマンティック分割なので、写っている全員の髪が同じマスクに入る。隣人が居ると髪シェル
// の格子が主役に割けなくなるので、**主役の頭から髪を伝って届く範囲だけ**を残す（`domain/hair/subject`）。
// この段もここを通す — `hairShellMask` が髪マスクの唯一の入口である性質を保つため。
//
// 調整できる値が 1 つ増えたことについて
// --------------------------------------
// 「そこに髪があるか」の判定はもともとモデルの側にあり、調整できる値がどこにも無かった。この門は
// そこへ `DEFAULT_ACCESSORY_FADE_FACE_WIDTHS` を 1 つ持ち込む。**避けられない** — モデルには
// 「頭に載っている装飾品」と「体に載っている装飾品」を分けるクラスが無いので、その区別はモデルの外で
// 与えるしかない。位置（顎）はランドマークから決まるので任意ではなく、任意なのは**境界をどれだけ
// ぼかすか**だけに閉じてある。

import {
  HairMask,
  Rect,
  ScalarField,
  denoisedHairMask,
  isFullRect,
  makeField,
  makeHairMask,
  rectEquals,
} from '../field';
import { LandmarkModel, Similarity2d, coarseSimilarity, selectIbug68 } from '../gnm/fit';
import {
  SUBJECT_GEODESIC_DISTANCE,
  SUBJECT_SEED_DISTANCE,
  SUBJECT_WORKING_FACE_PIXELS,
  subjectHairSelection,
} from './subject';

/** iBUG-68 の顎先（顎ライン 0〜16 の中央）。門の位置はこの点で決まる。 */
export const IBUG68_CHIN_INDEX = 8;

/**
 * 装飾品の門が顎から下へ 0 になるまでの距離（顔幅に対する割合）。
 *
 * **ぼかすのは硬い縁を作らないため。** フードやマフラーのように顎をまたいで続く装飾品が写った
 * 場合、切り落とすなら alpha が段差にならない形で落としたい。
 *
 * 0.25 は旧実装（画像の高さの 4%）を顔幅に読み替えた値。縦長写真（顔幅が画像の高さの約 15%）では
 * 4% ≒ 顔幅の 0.27 に当たるので、**同じ写真でほぼ同じ落とし方になる**。
 */
export const DEFAULT_ACCESSORY_FADE_FACE_WIDTHS = 0.25;

/**
 * 髪と装飾品の確信度から、髪シェルが覆う対象のマスクを作る。
 *
 * 3 つの段の順序に意味がある — `装飾品の門 → 雑音床 → 主役の選択` の順で、入れ替えられない:
 *
 *   - 床は「髪でないと判定された画素」の中央値なので、**門の後**でなければならない
 *   - 主役の選択は**床の後**でなければならない。先に隣人を 0 にすると、その画素が「髪でない画素」の
 *     分布に 0 として大量に混ざり、床が下駄より下へ引きずられる
 *
 * 判定（`present`）の作り方: 6 クラスの確信度は和が 1 なので、門をかけた後も
 * `覆わない = 1 − 覆う` が成り立つ。だから判定は 2 つの和の大小比較のままで、**新しいしきい値は
 * 要らない**。
 *
 * 主役の選択は `confidence` と `present` の**両方**に掛ける。片方だけに掛けると「確信度 0 なのに
 * 髪がある画素」ができ、消費側が隣人の髪を主役の髪として扱う。
 */
export function hairShellMask(input: {
  hair: ScalarField;
  accessory: ScalarField;
  photoLandmarks: Float64Array;
  landmarkModel: LandmarkModel;
  meshXy: Float64Array;
  imageSize: readonly [number, number];
  fadeFaceWidths?: number;
  seedDistance?: number;
  geodesicDistance?: number;
  workingFacePixels?: number;
}): HairMask {
  const [width, height] = input.imageSize;
  const fadeFaceWidths = input.fadeFaceWidths ?? DEFAULT_ACCESSORY_FADE_FACE_WIDTHS;
  if (!rectEquals(input.hair.rect, input.accessory.rect)) {
    throw new Error('髪と装飾品の rect が食い違っている');
  }
  if (input.hair.width !== input.accessory.width || input.hair.height !== input.accessory.height) {
    throw new Error('髪と装飾品の形が揃っていない');
  }
  // 門は場の格子を画像画素へ直に読み替えるので、画像全体を覆う場でなければ位置がずれる。
  if (!isFullRect(input.hair.rect)) {
    throw new Error('画像全体を覆う場を渡すこと');
  }
  if (!(fadeFaceWidths > 0)) throw new Error(`フェードの幅が ${fadeFaceWidths}`);

  // 相似変換はここで 1 回だけ解く。門も種も同じものを要るので、それぞれが解き直すと同じ最小二乗を
  // 2 回走らせたうえ「2 つが同じ変換か」を誰も保証しなくなる。
  const similarity = coarseSimilarity(input.photoLandmarks, input.landmarkModel);

  const gateField = accessoryGate(
    input.hair.height,
    input.hair.width,
    input.photoLandmarks,
    similarity,
    input.landmarkModel,
    [width, height],
    fadeFaceWidths,
  );

  const covered = new Float32Array(input.hair.values.length);
  const present = new Float32Array(input.hair.values.length);
  for (let pixel = 0; pixel < covered.length; pixel++) {
    const value = input.hair.values[pixel] + input.accessory.values[pixel] * gateField[pixel];
    covered[pixel] = Math.min(1, Math.max(0, value));
    // `覆う > 覆わない`。和が 1 なので `覆わない` は残り。0.5 という数字を書かないのは、それが
    // 調整できる値だと誤解されないため。
    present[pixel] = value > 1 - value ? 1 : 0;
  }
  const rect: Rect = input.hair.rect;
  const denoised = denoisedHairMask(
    makeHairMask(
      makeField(covered, input.hair.width, input.hair.height, rect),
      makeField(present, input.hair.width, input.hair.height, rect),
    ),
  );

  const reach = subjectHairSelection({
    present: denoised.present.values,
    rows: input.hair.height,
    columns: input.hair.width,
    similarity,
    landmarkModel: input.landmarkModel,
    meshXy: input.meshXy,
    imageSize: [width, height],
    seedDistance: input.seedDistance ?? SUBJECT_SEED_DISTANCE,
    geodesicDistance: input.geodesicDistance ?? SUBJECT_GEODESIC_DISTANCE,
    workingFacePixels: input.workingFacePixels ?? SUBJECT_WORKING_FACE_PIXELS,
  });

  const finalConfidence = new Float32Array(covered.length);
  const finalPresent = new Float32Array(covered.length);
  for (let pixel = 0; pixel < covered.length; pixel++) {
    finalConfidence[pixel] = denoised.confidence.values[pixel] * reach[pixel];
    finalPresent[pixel] = denoised.present.values[pixel] * reach[pixel];
  }
  return makeHairMask(
    makeField(finalConfidence, input.hair.width, input.hair.height, rect),
    makeField(finalPresent, input.hair.width, input.hair.height, rect),
    denoised.noiseFloor,
  );
}

/**
 * 顎より上を 1、そこから下へ `fade` かけて 0 になる場（場の格子の上）。
 *
 * 「下」は画像の行方向ではなく **GNM 空間の −Y を写真へ写した向き**。顔が傾いた写真でも門が顔に
 * ついて回る（画像の行で切ると、傾いた分だけ片側の頬が削れる）。
 */
function accessoryGate(
  rows: number,
  columns: number,
  photoLandmarks: Float64Array,
  similarity: Similarity2d,
  landmarkModel: LandmarkModel,
  imageSize: readonly [number, number],
  fadeFaceWidths: number,
): Float64Array {
  const [width, height] = imageSize;
  const ibug68 = selectIbug68(photoLandmarks);
  const chinX = ibug68[IBUG68_CHIN_INDEX * 2];
  const chinY = ibug68[IBUG68_CHIN_INDEX * 2 + 1];

  // GNM 空間の −Y（下）を写真へ写した向き。
  let downX = -similarity.linear[1];
  let downY = -similarity.linear[3];
  const norm = Math.hypot(downX, downY);
  if (!(norm > 0)) throw new Error('相似変換が退化していて顎の向きを取れない');
  downX /= norm;
  downY /= norm;

  const fade = fadeFaceWidths * landmarkModel.faceWidth * similarity.scale;
  if (!(fade > 0)) throw new Error('顔幅かスケールが 0 でフェードの幅を作れない');

  const out = new Float64Array(rows * columns);
  for (let row = 0; row < rows; row++) {
    const y = ((row + 0.5) / rows) * height;
    for (let column = 0; column < columns; column++) {
      const x = ((column + 0.5) / columns) * width;
      const below = (x - chinX) * downX + (y - chinY) * downY;
      out[row * columns + column] = Math.min(1, Math.max(0, 1 - below / fade));
    }
  }
  return out;
}
