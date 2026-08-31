// Pillow と同じ縮小（`Image.resize` の再標本化）。
//
// **正本はデスクトップ側が呼んでいる Pillow の `ImagingResample`。** あちらが縮小に PIL を使う 2 箇所
// と同じ絵を作るためにここに置く:
//
// | 使う場所 | フィルタ | 正本 |
// |:--|:--|:--|
// | 顔検出の階段（`domain/faceLadder`） | LANCZOS | `infrastructure/face_landmarks._resample_to_long_side` |
// | DAViD の入力 512x512（`infrastructure/depthNormal`） | BILINEAR | `infrastructure/depth_normal._preprocess` |
//
// **canvas の `drawImage` で代用してはいけない。** `imageSmoothingQuality: 'high'` の中身は
// ブラウザの実装依存で、bilinear でも LANCZOS でもない。顔検出の階段が直しているのは縮小の
// エイリアシングそのもの、DAViD の方は**モデルの入力**なので、どちらもフィルタの違いが結果に出る。
//
// ## Pillow のアルゴリズム
//
// 軸ごとに 1 パスずつ（分離可能）。縮小のときは**支持幅を縮小率で広げる**のが要点で、これが無いと
// 2 倍を超える縮小で入力画素の大半を読まずに捨てる（点標本化になる）。
//
//     filterScale = max(1, inSize / outSize)
//     support     = kernel.support * filterScale
//     center      = (i + 0.5) * inSize / outSize
//     taps        = [floor(center - support), ceil(center + support))
//     weight[j]   = kernel(( j + 0.5 - center ) / filterScale)      // 総和で正規化
//
// **8bit の sRGB 値のまま畳む**（線形光へ直さない）。Pillow がそうしているからで、ここを変えると
// あちらと違う絵になる。線形光で畳む面積平均は `domain/photo.resampleLongestSide` の側にあり、
// あちらの `domain/photo.resample_longest_side` と対応している — **用途が別なので混ぜない。**
//
// 係数は Pillow が 16bit 固定小数で持つのに対しこちらは float64。差は最終の丸めより小さい。

import { PhotoRgb, validatePhoto } from './photo';

/** 再標本化のフィルタ。`support` は縮小率 1 のときの片側の支持幅。 */
export interface ResampleKernel {
  readonly name: string;
  readonly support: number;
  weight(x: number): number;
}

/** Pillow の `BILINEAR`（三角フィルタ）。 */
export const TRIANGLE: ResampleKernel = {
  name: 'bilinear',
  support: 1,
  weight(x: number): number {
    const distance = Math.abs(x);
    return distance < 1 ? 1 - distance : 0;
  },
};

/** Pillow の `LANCZOS`（a = 3）。 */
export const LANCZOS3: ResampleKernel = {
  name: 'lanczos',
  support: 3,
  weight(x: number): number {
    if (x === 0) return 1;
    const distance = Math.abs(x);
    if (distance >= 3) return 0;
    const pi = Math.PI * distance;
    return (3 * Math.sin(pi) * Math.sin(pi / 3)) / (pi * pi);
  },
};

interface AxisWeights {
  /** 出力画素ごとの最初の入力 index。 */
  readonly offsets: Int32Array;
  /** 出力画素ごとの tap 数。 */
  readonly counts: Int32Array;
  /** (出力画素, span) の重み。 */
  readonly weights: Float64Array;
  readonly span: number;
}

/** Pillow と同じ tap と重みを軸 1 本ぶん作る。 */
function axisWeights(
  inSize: number,
  outSize: number,
  kernel: ResampleKernel,
): AxisWeights {
  const scale = inSize / outSize;
  const filterScale = Math.max(1, scale);
  const support = kernel.support * filterScale;
  const span = Math.ceil(support) * 2 + 1;
  const offsets = new Int32Array(outSize);
  const counts = new Int32Array(outSize);
  const weights = new Float64Array(outSize * span);

  for (let out = 0; out < outSize; out++) {
    const center = (out + 0.5) * scale;
    const first = Math.max(0, Math.floor(center - support));
    const last = Math.min(inSize, Math.ceil(center + support));
    offsets[out] = first;
    counts[out] = last - first;
    if (counts[out] > span) throw new Error(`tap が span を超えた（${counts[out]} > ${span}）`);
    let total = 0;
    for (let tap = 0; tap < counts[out]; tap++) {
      const weight = kernel.weight((first + tap + 0.5 - center) / filterScale);
      weights[out * span + tap] = weight;
      total += weight;
    }
    // Pillow と同じく総和で正規化する（総和が 0 になる形はフィルタの性質上起きない）。
    if (total !== 0) {
      for (let tap = 0; tap < counts[out]; tap++) weights[out * span + tap] /= total;
    }
  }
  return { offsets, counts, weights, span };
}

/** 1 軸ぶん畳む（`horizontal` なら列方向）。 */
function convolveAxis(
  source: Float64Array,
  width: number,
  height: number,
  outSize: number,
  axis: AxisWeights,
  horizontal: boolean,
): Float64Array {
  const outWidth = horizontal ? outSize : width;
  const outHeight = horizontal ? height : outSize;
  const out = new Float64Array(outWidth * outHeight * 3);
  for (let row = 0; row < outHeight; row++) {
    for (let column = 0; column < outWidth; column++) {
      const index = horizontal ? column : row;
      const first = axis.offsets[index];
      const count = axis.counts[index];
      let red = 0;
      let green = 0;
      let blue = 0;
      for (let tap = 0; tap < count; tap++) {
        const weight = axis.weights[index * axis.span + tap];
        if (weight === 0) continue;
        const sourceRow = horizontal ? row : first + tap;
        const sourceColumn = horizontal ? first + tap : column;
        const base = (sourceRow * width + sourceColumn) * 3;
        red += source[base] * weight;
        green += source[base + 1] * weight;
        blue += source[base + 2] * weight;
      }
      const target = (row * outWidth + column) * 3;
      out[target] = red;
      out[target + 1] = green;
      out[target + 2] = blue;
    }
  }
  return out;
}

/**
 * Pillow と同じ再標本化で `width` x `height` へ変える。
 *
 * 拡大もできるが（`filterScale` が 1 に留まる）、このリポジトリで使うのは縮小だけ。
 */
export function resamplePil(
  photo: PhotoRgb,
  width: number,
  height: number,
  kernel: ResampleKernel,
): PhotoRgb {
  validatePhoto(photo);
  if (width < 1 || height < 1) throw new Error(`出力の大きさが ${width}x${height}`);
  if (width === photo.width && height === photo.height) return photo;

  const source = new Float64Array(photo.data.length);
  for (let index = 0; index < photo.data.length; index++) source[index] = photo.data[index];

  // 横 → 縦の順（Pillow も同じ順）。片方が同じ大きさなら畳まない。
  let current = source;
  let currentWidth = photo.width;
  if (width !== photo.width) {
    current = convolveAxis(
      current,
      currentWidth,
      photo.height,
      width,
      axisWeights(photo.width, width, kernel),
      true,
    );
    currentWidth = width;
  }
  if (height !== photo.height) {
    current = convolveAxis(
      current,
      currentWidth,
      photo.height,
      height,
      axisWeights(photo.height, height, kernel),
      false,
    );
  }

  const data = new Uint8Array(width * height * 3);
  for (let index = 0; index < data.length; index++) {
    // Pillow と同じ「四捨五入して 0〜255 へ飽和」。
    const value = Math.round(current[index]);
    data[index] = value < 0 ? 0 : value > 255 ? 255 : value;
  }
  return { data, width, height };
}

/** 長辺を `longSide` 以下へ縮める（比は保つ）。既に小さければそのまま返す。 */
export function resamplePilToLongSide(
  photo: PhotoRgb,
  longSide: number,
  kernel: ResampleKernel,
): PhotoRgb {
  const longest = Math.max(photo.width, photo.height);
  if (longest <= longSide) return photo;
  const ratio = longSide / longest;
  return resamplePil(
    photo,
    Math.max(1, Math.round(photo.width * ratio)),
    Math.max(1, Math.round(photo.height * ratio)),
    kernel,
  );
}
