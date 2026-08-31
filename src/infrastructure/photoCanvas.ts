// 写真（RGB）とブラウザの canvas の行き来。
//
// MediaPipe も DAViD の前処理も入力に canvas / ImageBitmap を要求するので、`PhotoRgb` から 1 箇所で
// 作る。**推論のたびに作り直さない**よう、同じ写真に対しては呼び出し側がキャッシュする
// （`composition` が写真ごとに 1 枚だけ持つ）。

import { PhotoRgb } from '../domain/photo';

const cache = new WeakMap<Uint8Array, HTMLCanvasElement>();

/** 写真を canvas に描いて返す（同じ写真なら同じ canvas を返す）。 */
export function photoToCanvas(photo: PhotoRgb): HTMLCanvasElement {
  const cached = cache.get(photo.data);
  if (cached !== undefined && cached.width === photo.width && cached.height === photo.height) {
    return cached;
  }
  const canvas = document.createElement('canvas');
  canvas.width = photo.width;
  canvas.height = photo.height;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (context === null) throw new Error('canvas の 2d コンテキストが取れない');
  const rgba = new Uint8ClampedArray(photo.width * photo.height * 4);
  for (let pixel = 0; pixel < photo.width * photo.height; pixel++) {
    rgba[pixel * 4] = photo.data[pixel * 3];
    rgba[pixel * 4 + 1] = photo.data[pixel * 3 + 1];
    rgba[pixel * 4 + 2] = photo.data[pixel * 3 + 2];
    rgba[pixel * 4 + 3] = 255;
  }
  context.putImageData(new ImageData(rgba, photo.width, photo.height), 0, 0);
  cache.set(photo.data, canvas);
  return canvas;
}

/**
 * 写真の正方領域を切り出して指定の一辺へ縮めた RGB を返す（DAViD の前処理）。
 *
 * 縮小は canvas の `drawImage`（bilinear 相当）。デスクトップ側は Pillow の
 * `Image.Resampling.BILINEAR` を使っており、**同じ補間の種類**である。
 */
export function cropSquareToRgb(
  photo: PhotoRgb,
  square: { x: number; y: number; size: number },
  resolution: number,
): Uint8ClampedArray {
  const source = photoToCanvas(photo);
  const canvas = document.createElement('canvas');
  canvas.width = resolution;
  canvas.height = resolution;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (context === null) throw new Error('canvas の 2d コンテキストが取れない');
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = 'high';
  context.drawImage(
    source,
    square.x,
    square.y,
    square.size,
    square.size,
    0,
    0,
    resolution,
    resolution,
  );
  return context.getImageData(0, 0, resolution, resolution).data;
}
