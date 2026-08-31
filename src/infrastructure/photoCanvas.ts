// 写真（RGB）とブラウザの canvas の行き来。
//
// MediaPipe が入力に canvas / ImageBitmap を要求するので、`PhotoRgb` から 1 箇所で作る。
// **推論のたびに作り直さない**よう、同じ写真については WeakMap で使い回す。
//
// **縮小はここでやらない。** canvas の `drawImage` は `imageSmoothingQuality: 'high'` でも中身が
// ブラウザ依存で、デスクトップ側が使う PIL の bilinear / LANCZOS とは別のフィルタになる。縮小は
// `domain/resample`（PIL と同じ実装）が持つ。

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
