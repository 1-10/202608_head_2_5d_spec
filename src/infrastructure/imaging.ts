// 画像のエンコード（ブラウザの canvas 経由）。
//
// デスクトップ側は Pillow で書く。ブラウザには JPEG / PNG のエンコーダが canvas しかないので、
// **ここだけは web で置き換わる**。
//
// **JPEG のクロマサブサンプリングは選べない。** デスクトップ側は `subsampling=0`（4:4:4）で品質 90 を
// 指定しているが、canvas の `toBlob('image/jpeg', 0.9)` は実装依存で 4:2:0 になる。**色差の解像度が
// 半分になるのはブラウザ側の制約**で、こちらから指定する手が無い。肌アトラスは低周波の色が主なので
// 実害は小さいが、**同じ写真から作った zip がバイト単位で一致しないのはこれが理由**。
//
// PNG は可逆なので実装差が出ない（眼球テクスチャと `hair_alpha` はこちら）。

import { AlphaImage, RgbImage } from '../domain/contract';

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

/** 単一チャンネルを RGBA のグレースケールへ広げる。 */
function alphaToRgba(image: AlphaImage): Uint8ClampedArray {
  const out = new Uint8ClampedArray(image.width * image.height * 4);
  for (let pixel = 0; pixel < image.width * image.height; pixel++) {
    out[pixel * 4] = image.data[pixel];
    out[pixel * 4 + 1] = image.data[pixel];
    out[pixel * 4 + 2] = image.data[pixel];
    out[pixel * 4 + 3] = 255;
  }
  return out;
}

/**
 * RGBA を canvas に載せて指定の形式へエンコードする。
 *
 * `OffscreenCanvas` があればそちらを使う（DOM を汚さず、Worker でも動く）。
 */
async function encode(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  type: 'image/jpeg' | 'image/png',
  quality?: number,
): Promise<Uint8Array> {
  const imageData = new ImageData(rgba, width, height);
  if (typeof OffscreenCanvas !== 'undefined') {
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext('2d');
    if (context === null) throw new Error('OffscreenCanvas の 2d コンテキストが取れない');
    context.putImageData(imageData, 0, 0);
    const blob = await canvas.convertToBlob({ type, quality });
    return new Uint8Array(await blob.arrayBuffer());
  }
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  if (context === null) throw new Error('canvas の 2d コンテキストが取れない');
  context.putImageData(imageData, 0, 0);
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, type, quality));
  if (blob === null) throw new Error(`${type} のエンコードに失敗した`);
  return new Uint8Array(await blob.arrayBuffer());
}

/** RGB を JPEG にする。 */
export async function encodeJpeg(image: RgbImage, quality = JPEG_QUALITY): Promise<Uint8Array> {
  return encode(toRgba(image), image.width, image.height, 'image/jpeg', quality / 100);
}

/** RGB を PNG にする。 */
export async function encodePng(image: RgbImage): Promise<Uint8Array> {
  return encode(toRgba(image), image.width, image.height, 'image/png');
}

/** 単一チャンネルを PNG にする（グレースケールとして 3ch へ広げる）。 */
export async function encodeAlphaPng(image: AlphaImage): Promise<Uint8Array> {
  return encode(alphaToRgba(image), image.width, image.height, 'image/png');
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
