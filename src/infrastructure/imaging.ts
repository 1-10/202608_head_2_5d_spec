// 画像のエンコード。
//
// **JPEG は自分で書く（`infrastructure/jpeg`）。** canvas の `toBlob('image/jpeg', q)` は色差を
// 4:2:0 に間引き、止める手段が無い。デスクトップ側は `subsampling=0`（4:4:4）を明示していて、理由も
// 「4:2:0 はアトラスの chart 境界と髪の縁で色をにじませ、Unity 側の継ぎ目の原因を切り分けられなく
// する」と書いてある。canvas を使うと**あちらが消した不具合を出力に戻す**ことになる。
//
// **PNG も自分で書く（`infrastructure/png`）。** canvas は何を渡してもカラータイプ 6（RGBA）で書く
// が、デスクトップ側は配列の次元で mode を決めていて `hair_alpha` は mode "L"（1 チャンネル）で出る。
// 契約も「単一チャンネルの uint8 画像」と言っているので、RGB へ膨らませて書くと**申告と違う形の
// ファイルを渡す**ことになる。
//
// canvas に残るのは検査画像の表示だけ（`drawRgbImage`）。

import { AlphaImage, RgbImage } from '../domain/contract';
import { encodeJpeg444 } from './jpeg';
import { encodeGrayPng, encodeRgbPng } from './png';

/** JPEG の品質。デスクトップ側の `JPEG_QUALITY` と同じ値。 */
export const JPEG_QUALITY = 90;

/** RGB を canvas が扱う RGBA へ広げる（アルファは 255 固定）。 */
function toRgba(image: RgbImage): Uint8ClampedArray {
  const out = new Uint8ClampedArray(image.width * image.height * 4);
  for (let pixel = 0; pixel < image.width * image.height; pixel++) {
    out[pixel * 4] = image.data[pixel * 3];
    out[pixel * 4 + 1] = image.data[pixel * 3 + 1];
    out[pixel * 4 + 2] = image.data[pixel * 3 + 2];
    out[pixel * 4 + 3] = 255;
  }
  return out;
}

/**
 * RGB を JPEG（4:4:4）にする。
 *
 * `async` のままなのは、呼び出し側（`packaging`）が PNG と同じ形で扱えるようにするため。
 */
export async function encodeJpeg(image: RgbImage, quality = JPEG_QUALITY): Promise<Uint8Array> {
  return encodeJpeg444(image.data, image.width, image.height, quality);
}

/** RGB を PNG（カラータイプ 2）にする。 */
export async function encodePng(image: RgbImage): Promise<Uint8Array> {
  return encodeRgbPng(image.data, image.width, image.height);
}

/** 単一チャンネルを PNG（カラータイプ 0 = グレースケール）にする。 */
export async function encodeAlphaPng(image: AlphaImage): Promise<Uint8Array> {
  return encodeGrayPng(image.data, image.width, image.height);
}

/** RGBA の `ImageData` を RGB の写真へ落とす（入力の段で 1 回だけ）。 */
export function imageDataToPhotoRgb(imageData: ImageData): {
  data: Uint8Array;
  width: number;
  height: number;
} {
  const out = new Uint8Array(imageData.width * imageData.height * 3);
  for (let pixel = 0; pixel < imageData.width * imageData.height; pixel++) {
    out[pixel * 3] = imageData.data[pixel * 4];
    out[pixel * 3 + 1] = imageData.data[pixel * 4 + 1];
    out[pixel * 3 + 2] = imageData.data[pixel * 4 + 2];
  }
  return { data: out, width: imageData.width, height: imageData.height };
}

/** RGB 画像を canvas に描いて `<img>` などから見えるようにする（検査画像の表示用）。 */
export function drawRgbImage(image: RgbImage, canvas: HTMLCanvasElement): void {
  canvas.width = image.width;
  canvas.height = image.height;
  const context = canvas.getContext('2d');
  if (context === null) throw new Error('canvas の 2d コンテキストが取れない');
  context.putImageData(new ImageData(toRgba(image), image.width, image.height), 0, 0);
}
