// 検査画像（各段の出力をそのまま見る画像）。
//
// パイプラインは段に分かれていて、**検査画像は各段の出力そのもの**である。ここにはその画像を
// 作る純関数と、段ごとに集めて持ち回る型を置く。
//
// 出力契約（`domain/contract`）とは分けている。検査画像は zip に入らず、Unity 側の仕様でも
// ない。契約の型に混ぜると「出力仕様の一部なのか」が読めなくなる。
//
// 解像度
// ------
// 写真は数千画素あるが、検査に必要なのは「ずれているか」「境目がどこか」なので、一辺
// `INSPECTION_MAX_SIDE` へ落として描く。落とす倍率は整数の間引きに丸めるので、provenance の
// ような値そのものに意味がある画像でも色が混ざらない。
//
// 座標
// ----
// 写真ピクセル座標で来た点・線は `PhotoCanvas` が間引き倍率をかけて描く。倍率を呼び出し側で
// 掛けると、写真ごとに変わる数を各所で書き写すことになる。

import { ScalarField, sampleField } from './field';
import { RgbImage } from './contract';

/** 検査画像の一辺の上限（画素）。 */
export const INSPECTION_MAX_SIDE = 1024;

/**
 * 線 1 本を描くときのサンプル点数の上限。
 *
 * 上限に当たった線は点線に見える — 髪シェルのワイヤは辺長が揃っているので、点線に見えたら
 * 格子解像度が想定と違うことの合図になる。
 */
export const MAX_SEGMENT_SAMPLES = 96;

/**
 * スカラー場をグレースケールへ写すときに使う下端・上端のパーセンタイル。
 *
 * 最小値・最大値で正規化すると、深度の外れ値 1 画素で階調が潰れる。
 */
export const ROBUST_PERCENTILES: readonly [number, number] = [1.0, 99.0];

/** 一辺を `maxSide` 以下にする整数の間引き幅。1 以上。 */
export function downscaleStride(width: number, height: number, maxSide = INSPECTION_MAX_SIDE): number {
  return Math.max(1, Math.ceil(Math.max(width, height) / maxSide));
}

/**
 * スカラーの配列をロバストな範囲で 0..255 のグレースケールにする。
 *
 * `mask` を渡すと、その内側だけで正規化の範囲を決める（外側も同じ範囲で写す）。範囲が潰れて
 * いる（全て同じ値）場合は 0 で埋める。
 */
export function normalizeToUint8(
  values: Float64Array | Float32Array,
  mask: Uint8Array | null = null,
): Uint8Array {
  const sampled: number[] = [];
  for (let index = 0; index < values.length; index++) {
    if (mask === null || mask[index] !== 0) sampled.push(values[index]);
  }
  const out = new Uint8Array(values.length);
  if (sampled.length === 0) return out;
  sampled.sort((a, b) => a - b);
  const low = percentile(sampled, ROBUST_PERCENTILES[0]);
  const high = percentile(sampled, ROBUST_PERCENTILES[1]);
  if (!Number.isFinite(low) || !Number.isFinite(high) || high <= low) return out;
  for (let index = 0; index < values.length; index++) {
    out[index] = Math.min(255, Math.max(0, ((values[index] - low) / (high - low)) * 255));
  }
  return out;
}

/** ソート済み配列の線形補間パーセンタイル（numpy の既定と同じ）。 */
function percentile(sorted: readonly number[], percent: number): number {
  const position = ((sorted.length - 1) * percent) / 100;
  const low = Math.floor(position);
  const high = Math.ceil(position);
  if (low === high) return sorted[low];
  return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
}

/** グレースケールを RGB 画像にする。 */
export function grayToRgb(gray: Uint8Array, width: number, height: number): RgbImage {
  const data = new Uint8Array(width * height * 3);
  for (let pixel = 0; pixel < width * height; pixel++) {
    data[pixel * 3] = gray[pixel];
    data[pixel * 3 + 1] = gray[pixel];
    data[pixel * 3 + 2] = gray[pixel];
  }
  return { data, width, height };
}

/**
 * 由来の値を色分けした検査画像を作る。
 *
 * パレットの大きさを**表の最大値**から取る。表の要素数から取ると、値が連番でなくなった瞬間に
 * 添字が範囲外になる。
 */
export function provenancePaletteImage(
  provenance: Uint8Array,
  width: number,
  height: number,
  colors: readonly [number, [number, number, number]][],
): RgbImage {
  let maximum = 0;
  for (const [value] of colors) if (value > maximum) maximum = value;
  const palette = new Uint8Array((maximum + 1) * 3);
  for (const [value, rgb] of colors) {
    palette[value * 3] = rgb[0];
    palette[value * 3 + 1] = rgb[1];
    palette[value * 3 + 2] = rgb[2];
  }
  const data = new Uint8Array(width * height * 3);
  for (let pixel = 0; pixel < width * height; pixel++) {
    const value = provenance[pixel];
    data[pixel * 3] = palette[value * 3];
    data[pixel * 3 + 1] = palette[value * 3 + 1];
    data[pixel * 3 + 2] = palette[value * 3 + 2];
  }
  return { data, width, height };
}

/** 生の法線 (3, h, w) を RGB に写す（`n * 0.5 + 0.5`）。 */
export function encodeNormalRgb(normal: Float32Array, width: number, height: number): RgbImage {
  const area = width * height;
  if (normal.length !== area * 3) throw new Error(`normal の形が (3, h, w) ではない`);
  const data = new Uint8Array(area * 3);
  for (let pixel = 0; pixel < area; pixel++) {
    for (let channel = 0; channel < 3; channel++) {
      data[pixel * 3 + channel] = Math.min(
        255,
        Math.max(0, (normal[channel * area + pixel] * 0.5 + 0.5) * 255),
      );
    }
  }
  return { data, width, height };
}

/** 大きい画像を整数間引きで検査用の解像度へ落とす。 */
export function downscaled(image: RgbImage, maxSide = INSPECTION_MAX_SIDE): RgbImage {
  const stride = downscaleStride(image.width, image.height, maxSide);
  if (stride === 1) return image;
  const width = Math.ceil(image.width / stride);
  const height = Math.ceil(image.height / stride);
  const data = new Uint8Array(width * height * 3);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const source = (row * stride * image.width + column * stride) * 3;
      const target = (row * width + column) * 3;
      data[target] = image.data[source];
      data[target + 1] = image.data[source + 1];
      data[target + 2] = image.data[source + 2];
    }
  }
  return { data, width, height };
}

/** 三角形の配列から辺の両端の頂点 index を作る。 */
export function triangleEdges(triangles: Uint32Array): { starts: Int32Array; ends: Int32Array } {
  const triangleCount = triangles.length / 3;
  const starts = new Int32Array(triangleCount * 3);
  const ends = new Int32Array(triangleCount * 3);
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    for (let corner = 0; corner < 3; corner++) {
      starts[triangle * 3 + corner] = triangles[triangle * 3 + corner];
      ends[triangle * 3 + corner] = triangles[triangle * 3 + ((corner + 1) % 3)];
    }
  }
  return { starts, ends };
}

/** 切り出した検査領域に足す余白（領域の辺長に対する比）。 */
export const REGION_MARGIN_FRACTION = 0.08;

/** 検査画像を描く土台（元写真を検査用の解像度へ間引いたもの）。 */
export class PhotoCanvas {
  readonly data: Uint8Array;
  readonly width: number;
  readonly height: number;
  readonly photoWidth: number;
  readonly photoHeight: number;

  private constructor(
    data: Uint8Array,
    width: number,
    height: number,
    photoWidth: number,
    photoHeight: number,
  ) {
    this.data = data;
    this.width = width;
    this.height = height;
    this.photoWidth = photoWidth;
    this.photoHeight = photoHeight;
  }

  /** 写真から土台を作る。 */
  static of(photo: RgbImage, maxSide = INSPECTION_MAX_SIDE): PhotoCanvas {
    const stride = downscaleStride(photo.width, photo.height, maxSide);
    const width = Math.ceil(photo.width / stride);
    const height = Math.ceil(photo.height / stride);
    const data = new Uint8Array(width * height * 3);
    for (let row = 0; row < height; row++) {
      for (let column = 0; column < width; column++) {
        const source = (row * stride * photo.width + column * stride) * 3;
        const target = (row * width + column) * 3;
        data[target] = photo.data[source];
        data[target + 1] = photo.data[source + 1];
        data[target + 2] = photo.data[source + 2];
      }
    }
    return new PhotoCanvas(data, width, height, photo.width, photo.height);
  }

  /** 写真全体を土台と同じ大きさへ引き伸ばした画像（描き込み前の複製）。 */
  image(): RgbImage {
    return { data: Uint8Array.from(this.data), width: this.width, height: this.height };
  }

  /** 写真ピクセル座標を土台の座標へ写す。 */
  toCanvasXy(x: number, y: number): [number, number] {
    return [(x * this.width) / this.photoWidth, (y * this.height) / this.photoHeight];
  }

  /** スカラー場を土台と同じ大きさの配列へ引き直す（rect の外は 0）。 */
  rasterizeField(field: ScalarField): Float64Array {
    const out = new Float64Array(this.width * this.height);
    for (let row = 0; row < this.height; row++) {
      const v = (row + 0.5) / this.height;
      for (let column = 0; column < this.width; column++) {
        out[row * this.width + column] = sampleField(field, (column + 0.5) / this.width, v);
      }
    }
    return out;
  }

  /** スカラー場をグレースケールの検査画像にする（場の内側で正規化）。 */
  fieldImage(field: ScalarField): RgbImage {
    const values = this.rasterizeField(field);
    const mask = new Uint8Array(values.length);
    for (let pixel = 0; pixel < values.length; pixel++) mask[pixel] = values[pixel] !== 0 ? 1 : 0;
    return grayToRgb(normalizeToUint8(values, mask), this.width, this.height);
  }

  /** 0..1 の重みで写真の上に色を被せる（マスクの検査用）。 */
  tinted(weight: Float64Array, color: readonly [number, number, number]): RgbImage {
    const image = this.image();
    for (let pixel = 0; pixel < this.width * this.height; pixel++) {
      const alpha = Math.min(1, Math.max(0, weight[pixel]));
      for (let channel = 0; channel < 3; channel++) {
        image.data[pixel * 3 + channel] = Math.min(
          255,
          Math.max(0, image.data[pixel * 3 + channel] * (1 - alpha) + color[channel] * alpha),
        );
      }
    }
    return image;
  }

  /** 写真ピクセル座標の点を四角で描く。 */
  withPoints(
    pointsXy: Float64Array,
    color: readonly [number, number, number],
    image: RgbImage | null = null,
    radius = 2,
  ): RgbImage {
    const target = image ?? this.image();
    for (let point = 0; point < pointsXy.length / 2; point++) {
      const [canvasX, canvasY] = this.toCanvasXy(pointsXy[point * 2], pointsXy[point * 2 + 1]);
      const centerColumn = Math.round(canvasX - 0.5);
      const centerRow = Math.round(canvasY - 0.5);
      for (let dy = -radius; dy <= radius; dy++) {
        const row = centerRow + dy;
        if (row < 0 || row >= this.height) continue;
        for (let dx = -radius; dx <= radius; dx++) {
          const column = centerColumn + dx;
          if (column < 0 || column >= this.width) continue;
          const pixel = (row * this.width + column) * 3;
          target.data[pixel] = color[0];
          target.data[pixel + 1] = color[1];
          target.data[pixel + 2] = color[2];
        }
      }
    }
    return target;
  }

  /** 写真ピクセル座標の線分をまとめて描く（ワイヤフレーム用）。 */
  withSegments(
    startsXy: Float64Array,
    endsXy: Float64Array,
    color: readonly [number, number, number],
    image: RgbImage | null = null,
  ): RgbImage {
    const target = image ?? this.image();
    const count = startsXy.length / 2;
    if (count === 0) return target;
    let longest = 0;
    const starts: [number, number][] = [];
    const ends: [number, number][] = [];
    for (let segment = 0; segment < count; segment++) {
      const start = this.toCanvasXy(startsXy[segment * 2], startsXy[segment * 2 + 1]);
      const end = this.toCanvasXy(endsXy[segment * 2], endsXy[segment * 2 + 1]);
      starts.push(start);
      ends.push(end);
      longest = Math.max(longest, Math.abs(end[0] - start[0]), Math.abs(end[1] - start[1]));
    }
    const steps = Math.min(MAX_SEGMENT_SAMPLES, Math.max(2, Math.ceil(longest) + 1));
    for (let segment = 0; segment < count; segment++) {
      for (let step = 0; step < steps; step++) {
        const t = step / (steps - 1);
        const x = starts[segment][0] + (ends[segment][0] - starts[segment][0]) * t;
        const y = starts[segment][1] + (ends[segment][1] - starts[segment][1]) * t;
        const column = Math.round(x - 0.5);
        const row = Math.round(y - 0.5);
        if (row < 0 || row >= this.height || column < 0 || column >= this.width) continue;
        const pixel = (row * this.width + column) * 3;
        target.data[pixel] = color[0];
        target.data[pixel + 1] = color[1];
        target.data[pixel + 2] = color[2];
      }
    }
    return target;
  }
}

/**
 * 点群を囲む領域だけを切り出した土台と、その中での点の座標を返す。
 *
 * 写真全体を間引いた土台ではワイヤが検査できない。全身写真だと頭が数百画素しか無く、髪シェルの
 * 格子（96x120）が間引き後の 1 画素より細かくなって塗り潰しに見える。切り出してから間引けば、
 * **写真の中で顔がどれだけ小さく写っていても同じ密度で見える**。
 *
 * 領域を `maxSide` に収まる最大の整数倍へ引き伸ばす。補間はしない（画素の複製だけ）ので色は
 * 混ざらない。
 */
export function regionCanvas(
  photo: RgbImage,
  pointsXy: Float64Array,
  marginFraction = REGION_MARGIN_FRACTION,
  maxSide = INSPECTION_MAX_SIDE,
): { canvas: PhotoCanvas; localPoints: Float64Array } {
  let lowX = Infinity;
  let lowY = Infinity;
  let highX = -Infinity;
  let highY = -Infinity;
  for (let point = 0; point < pointsXy.length / 2; point++) {
    lowX = Math.min(lowX, pointsXy[point * 2]);
    highX = Math.max(highX, pointsXy[point * 2]);
    lowY = Math.min(lowY, pointsXy[point * 2 + 1]);
    highY = Math.max(highY, pointsXy[point * 2 + 1]);
  }
  const marginX = (highX - lowX) * marginFraction;
  const marginY = (highY - lowY) * marginFraction;
  const xMin = Math.min(Math.max(Math.floor(lowX - marginX), 0), photo.width - 1);
  const yMin = Math.min(Math.max(Math.floor(lowY - marginY), 0), photo.height - 1);
  const xMax = Math.min(Math.max(Math.ceil(highX + marginX), xMin + 1), photo.width);
  const yMax = Math.min(Math.max(Math.ceil(highY + marginY), yMin + 1), photo.height);

  const cropWidth = xMax - xMin;
  const cropHeight = yMax - yMin;
  const zoom = Math.max(1, Math.floor(maxSide / Math.max(cropWidth, cropHeight)));
  const width = cropWidth * zoom;
  const height = cropHeight * zoom;
  const data = new Uint8Array(width * height * 3);
  for (let row = 0; row < height; row++) {
    const sourceRow = yMin + Math.floor(row / zoom);
    for (let column = 0; column < width; column++) {
      const sourceColumn = xMin + Math.floor(column / zoom);
      const source = (sourceRow * photo.width + sourceColumn) * 3;
      const target = (row * width + column) * 3;
      data[target] = photo.data[source];
      data[target + 1] = photo.data[source + 1];
      data[target + 2] = photo.data[source + 2];
    }
  }
  const localPoints = new Float64Array(pointsXy.length);
  for (let point = 0; point < pointsXy.length / 2; point++) {
    localPoints[point * 2] = (pointsXy[point * 2] - xMin) * zoom;
    localPoints[point * 2 + 1] = (pointsXy[point * 2 + 1] - yMin) * zoom;
  }
  return { canvas: PhotoCanvas.of({ data, width, height }, maxSide), localPoints };
}

/**
 * 各段の検査画像。段の名前ではなく**何を見ている画像か**で名前を付ける。
 *
 * 段が増えても呼び出し側の unpack が壊れないよう、値の集合として持つ。
 */
export interface InspectionImages {
  /** 段1: 検出した 478 点を写真の上に打ったもの。 */
  photoLandmarks?: RgbImage;
  /** 段1: 髪シェルが覆う対象のマスク（写真の上に色を被せたもの）。 */
  hairMask?: RgbImage;
  /** 段1: DAViD の相対深度（頭部だけの切り出し）。 */
  depth?: RgbImage;
  /** 段1: DAViD の表面法線（RGB エンコード）。 */
  normal?: RgbImage;
  /** 段1: DAViD の人物前景（メッシュ全体を覆う切り出し）。 */
  foreground?: RgbImage;
  /** 段2: 対応点とフィット後の点を重ねたもの。 */
  landmarkFit?: RgbImage;
  /** 段2: 耳・首の輪郭フィットの観測と結果。 */
  silhouetteFit?: RgbImage;
  /** 段3: 焼いた眼球テクスチャ（解剖学的左）。 */
  leftEyeAlbedo?: RgbImage;
  /** 段3: 焼いた眼球テクスチャ（解剖学的右）。 */
  rightEyeAlbedo?: RgbImage;
  /** 段3: 眼球テクスチャの由来（左右を横に並べたもの）。 */
  eyeAlbedoProvenance?: RgbImage;
  /** 段4: アトラスのテクセルを写真へ投げた点。 */
  atlasProjection?: RgbImage;
  /** 段4: 投影の門（前景の確信度）を写真の上に見たもの。 */
  atlasProjectionGate?: RgbImage;
  /** 段4: 門で棄却されたテクセルをアトラスの上に見たもの。 */
  atlasAlbedoGate?: RgbImage;
  /** 段4: 焼いたアトラス。 */
  atlasAlbedo?: RgbImage;
  /** 段4: アトラスのテクセルごとの由来。 */
  atlasProvenance?: RgbImage;
  /** 段5: 髪シェルのワイヤフレーム。 */
  hairShellWire?: RgbImage;
  /** 段5: 髪の厚み（格子上の場）。 */
  hairThickness?: RgbImage;
}
