// GNM 空間 XY と画像 UV 空間の行き来。
//
// フィットが出す相似変換は「GNM 空間 xy → 写真ピクセル」なので、ScalarField（画像 UV 空間）を
// 引くには写真の大きさで割る一段が必ず挟まる。この割り算を各所で書き写すと、片方だけ画像サイズを
// 更新したときに黙ってズレる。式の正本をここに置く。
//
// 座標変換はしない。GNM 空間の値（メートル・右手系）はそのまま保つ。ここでやるのは「同じ点を
// 別の座標系でどう呼ぶか」の読み替えだけ。

import { Similarity2d } from '../gnm/fit';

/** GNM 空間の XY (N, 2) を画像 UV 空間 (N, 2) へ写す。 */
export function gnmXyToImageUv(
  pointsXy: Float64Array,
  similarity: Similarity2d,
  imageSize: readonly [number, number],
): Float64Array {
  const [width, height] = validatedImageSize(imageSize);
  const pixels = similarity.apply(pointsXy);
  const out = new Float64Array(pixels.length);
  for (let point = 0; point < pixels.length / 2; point++) {
    out[point * 2] = pixels[point * 2] / width;
    out[point * 2 + 1] = pixels[point * 2 + 1] / height;
  }
  return out;
}

/** 1 点だけの `gnmXyToImageUv`。 */
export function gnmXyToImageUvPoint(
  x: number,
  y: number,
  similarity: Similarity2d,
  imageSize: readonly [number, number],
): [number, number] {
  const [width, height] = validatedImageSize(imageSize);
  const [pixelX, pixelY] = similarity.applyPoint(x, y);
  return [pixelX / width, pixelY / height];
}

/** 画像 UV 空間の (N, 2) を GNM 空間の XY (N, 2) へ戻す。 */
export function imageUvToGnmXy(
  pointsUv: Float64Array,
  similarity: Similarity2d,
  imageSize: readonly [number, number],
): Float64Array {
  const [width, height] = validatedImageSize(imageSize);
  const pixels = new Float64Array(pointsUv.length);
  for (let point = 0; point < pointsUv.length / 2; point++) {
    pixels[point * 2] = pointsUv[point * 2] * width;
    pixels[point * 2 + 1] = pointsUv[point * 2 + 1] * height;
  }
  return similarity.inverseApply(pixels);
}

function validatedImageSize(imageSize: readonly [number, number]): [number, number] {
  const width = Math.trunc(imageSize[0]);
  const height = Math.trunc(imageSize[1]);
  if (width <= 0 || height <= 0) throw new Error(`画像サイズが正でない: ${imageSize}`);
  return [width, height];
}
