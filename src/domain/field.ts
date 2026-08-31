// ScalarField — 画像 UV 空間の 2D スカラー場。
//
// 深度・マスク・法線成分といった「写真の画素に一対一で対応する値の場」を、サンプリング
// 規約ごと1つの型に閉じ込める。呼び出し側が配列の向き・原点・正規化を各所で解釈し直さず
// に済むようにするのが目的。
//
// 部分矩形（rect）に対応する。推論器は写真全体ではなく顔まわりの切り出しに対して走ること
// があり、その出力を「元画像のどこを覆う場か」を保ったまま扱えるようにする。rect 外の
// 参照は 0 を返す（外挿しない）。深度もマスクも前景も「そこには何も無い」が 0 で表せる
// ため、無効値を別に持つより呼び出し側が分岐せずに済む。
//
// 画像 UV 空間の規約（ここが正本）:
//     u = x / 画像幅、v = y / 画像高さ。原点は画像の左上、v は下向き。
//     つまり配列の行方向がそのまま v の正方向。
//     GNM 空間（Y 上向き）とは v の向きが逆。変換はフィット側の責務。

/** 画像 UV 空間の軸並行矩形。閉区間として扱う（境界上は内側）。 */
export interface Rect {
  readonly uMin: number;
  readonly vMin: number;
  readonly uMax: number;
  readonly vMax: number;
}

export function makeRect(uMin: number, vMin: number, uMax: number, vMax: number): Rect {
  if (!(uMax > uMin && vMax > vMin)) {
    throw new Error(`rect の幅・高さが正でない: ${uMin},${vMin},${uMax},${vMax}`);
  }
  return { uMin, vMin, uMax, vMax };
}

/** 画像全体を覆う矩形。 */
export function fullRect(): Rect {
  return { uMin: 0, vMin: 0, uMax: 1, vMax: 1 };
}

/**
 * 画素座標の矩形（左上 x, y と大きさ）を画像 UV 空間へ移す。
 *
 * 切り出しは画素で決まるが、場の座標系は解像度に依らない UV で持つ。変換式を1箇所に
 * 閉じておかないと、切り出し側とサンプル側で 0.5 画素の取り違えが黙って入る。
 */
export function rectFromPixels(
  x: number,
  y: number,
  width: number,
  height: number,
  imageWidth: number,
  imageHeight: number,
): Rect {
  return makeRect(
    x / imageWidth,
    y / imageHeight,
    (x + width) / imageWidth,
    (y + height) / imageHeight,
  );
}

export function rectEquals(a: Rect, b: Rect): boolean {
  return a.uMin === b.uMin && a.vMin === b.vMin && a.uMax === b.uMax && a.vMax === b.vMax;
}

export function isFullRect(rect: Rect): boolean {
  return rectEquals(rect, fullRect());
}

export function uSpan(rect: Rect): number {
  return rect.uMax - rect.uMin;
}

export function vSpan(rect: Rect): number {
  return rect.vMax - rect.vMin;
}

/**
 * rect が覆う範囲の 2D スカラー場。
 *
 * 画素中心の規約: `values[j * width + i]` は rect 内の
 * `u = uMin + (i + 0.5) / width * uSpan` / `v = vMin + (j + 0.5) / height * vSpan`
 * における値。この半画素ずらしが `sampleField` の逆写像と対応している。
 */
export interface ScalarField {
  readonly values: Float32Array;
  readonly width: number;
  readonly height: number;
  readonly rect: Rect;
}

export function makeField(
  values: Float32Array,
  width: number,
  height: number,
  rect: Rect,
): ScalarField {
  if (values.length !== width * height) {
    throw new Error(`values の長さ ${values.length} が ${width}x${height} と合わない`);
  }
  return { values, width, height, rect };
}

/** 画像全体を覆う場として包む。 */
export function fieldOverFullImage(
  values: Float32Array,
  width: number,
  height: number,
): ScalarField {
  return makeField(values, width, height, fullRect());
}

/**
 * 画像 UV 座標で bilinear サンプルする。rect 外は 0。
 *
 * rect の内側では端の画素をクランプして外挿する（画素中心より外の半画素分）。rect の
 * 外側は 0。境界をまたぐ点が端の値を引きずらないよう、判定はクランプ前の UV で行う。
 */
export function sampleField(field: ScalarField, u: number, v: number): number {
  const { rect, width, height, values } = field;
  if (u < rect.uMin || u > rect.uMax || v < rect.vMin || v > rect.vMax) return 0;

  const fx = ((u - rect.uMin) / uSpan(rect)) * width - 0.5;
  const fy = ((v - rect.vMin) / vSpan(rect)) * height - 0.5;
  const x0Real = Math.floor(fx);
  const y0Real = Math.floor(fy);
  const tx = fx - x0Real;
  const ty = fy - y0Real;
  const x0 = Math.min(Math.max(x0Real, 0), width - 1);
  const x1 = Math.min(Math.max(x0Real + 1, 0), width - 1);
  const y0 = Math.min(Math.max(y0Real, 0), height - 1);
  const y1 = Math.min(Math.max(y0Real + 1, 0), height - 1);

  const top = values[y0 * width + x0] * (1 - tx) + values[y0 * width + x1] * tx;
  const bottom = values[y1 * width + x0] * (1 - tx) + values[y1 * width + x1] * tx;
  return top * (1 - ty) + bottom * ty;
}

/** 点が場の rect の内側か（`sampleField` が 0 を返すかどうかの判定と同じ式）。 */
export function insideRect(rect: Rect, u: number, v: number): boolean {
  return u >= rect.uMin && u <= rect.uMax && v >= rect.vMin && v <= rect.vMax;
}

/** 場を別の解像度・画像全体の格子へ引き直す（画素中心の規約で）。 */
export function resampleFieldToImage(
  field: ScalarField,
  width: number,
  height: number,
): Float32Array {
  if (isFullRect(field.rect) && field.width === width && field.height === height) {
    return field.values;
  }
  const out = new Float32Array(width * height);
  for (let row = 0; row < height; row++) {
    const v = (row + 0.5) / height;
    for (let column = 0; column < width; column++) {
      out[row * width + column] = sampleField(field, (column + 0.5) / width, v);
    }
  }
  return out;
}

/**
 * 1 回のセグメンテーションから取る場をまとめたもの。
 *
 * セグメンタは 6 クラスの確信度を一度に返すので、そこから要る場を全部ここへ束ねる。
 * **別々のメソッドで取らない** — 分けると同じ写真に対して 2 回推論する経路ができ、
 * しかも「同じ推論から出た揃った組か」を呼び出し側が保証することになる。
 *
 * **クラスを混ぜずにそのまま持つ。** 髪と装飾品を足すには位置の門が要り、それを知って
 * いるのはアダプタではなく段の合成側なので、足す場所は domain
 * （`domain/hair/mask.hairShellMask`）。
 */
export interface PersonSegmentation {
  /** 髪クラスの確信度（0..1）。 */
  readonly hair: ScalarField;
  /** 装飾品クラス（帽子・メガネ・バッグの肩紐など）の確信度（0..1）。 */
  readonly accessory: ScalarField;
  /** 顔の肌の確信度（0..1）。**基準の肌色を測る領域**。 */
  readonly faceSkin: ScalarField;
  /** 体の肌の確信度（0..1）。耳と首の輪郭フィットに使う。 */
  readonly bodySkin: ScalarField;
}

export function validateSegmentation(segmentation: PersonSegmentation): void {
  const reference = segmentation.hair;
  for (const [name, field] of [
    ['accessory', segmentation.accessory],
    ['faceSkin', segmentation.faceSkin],
    ['bodySkin', segmentation.bodySkin],
  ] as const) {
    if (field.width !== reference.width || field.height !== reference.height) {
      throw new Error(
        `${name} と髪の場の形が揃っていない: ${name}=${field.width}x${field.height}` +
          ` hair=${reference.width}x${reference.height}`,
      );
    }
    if (!rectEquals(field.rect, reference.rect)) {
      throw new Error(`${name} と髪の場の rect が食い違っている`);
    }
  }
}

/**
 * 髪 + 装飾品のマスク。連続の重みと、モデル自身の判定を束ねる。
 *
 * 2 つを別々に返さないのは、同じ 1 回の推論の出力であり、分けると「同じ推論から出た
 * 揃った組か」を呼び出し側が保証することになるため。
 */
export interface HairMask {
  /** 髪 + 装飾品の確信度の和（0..1）。**連続値が要る用途に使う**。 */
  readonly confidence: ScalarField;
  /**
   * 髪があると判定された画素を 1、それ以外を 0 とした場。
   *
   * **「そこに髪があるか」の判定はこちらを使う。** 判定を下すのはセグメンタ（クラス
   * 確率の比較）なので、消費側に「確信度をどこで切るか」という調整可能な値が現れない。
   */
  readonly present: ScalarField;
  /** `confidence` から引いた雑音床。null なら生（引いていない）。 */
  readonly noiseFloor: number | null;
}

export function makeHairMask(
  confidence: ScalarField,
  present: ScalarField,
  noiseFloor: number | null = null,
): HairMask {
  if (confidence.width !== present.width || confidence.height !== present.height) {
    throw new Error('confidence と present の形が揃っていない');
  }
  if (!rectEquals(confidence.rect, present.rect)) {
    throw new Error('confidence と present の rect が食い違っている');
  }
  return { confidence, present, noiseFloor };
}

/**
 * 雑音床を測る: **髪でないと判定された画素での確信度の中央値**。
 *
 * セグメンタの確信度は髪が無い画素でも 0 を返さず、写真全体に一様な下駄を履く。これを
 * alpha にそのまま使うと殻の輪郭が一度も切られず、薄い膜が顔・首・服の上に残る。
 *
 * 測る場所を「髪でないと判定された画素」にするのは、**崩れる条件が無いから**。背景で
 * 測る案は背景が写らない写真で画素が尽きるが、こちらは顔が必ず写っている。写真ごとに
 * 測るので、サンプル由来の固定値にもならない。
 *
 * 中央値を採るのは、床が「大多数の非髪画素が示す下駄」であって最小値でも平均でもない
 * から。最小値は雑音の下端に張り付き、平均は誤検出の裾に引かれる。
 *
 * 戻り値 null = 測れない（モデルが全画素を髪と判定した）。
 */
export function measuredNoiseFloor(mask: HairMask): number | null {
  const outside: number[] = [];
  for (let pixel = 0; pixel < mask.present.values.length; pixel++) {
    if (mask.present.values[pixel] <= 0) outside.push(mask.confidence.values[pixel]);
  }
  if (outside.length === 0) return null;
  outside.sort((a, b) => a - b);
  const middle = outside.length >> 1;
  return outside.length % 2 === 1
    ? outside[middle]
    : (outside[middle - 1] + outside[middle]) / 2;
}

/**
 * 雑音床を引いて再正規化した `HairMask` を返す（`present` は変わらない）。
 *
 *     alpha = clip((確信度 − 床) / (1 − 床), 0, 1)
 *
 * **閾値で切らない。** 髪の柔らかい縁（後れ毛・毛束の隙間）の中間値は本物なので、切ると
 * シルエットが硬くなり alpha を持っている目的そのものを失う。減算なら床より下は厳密に
 * 0 になり、床より上の階調は（再正規化のぶん伸びるだけで）残る。
 *
 * 床が測れないときと、引く意味が無いとき（床が 0 以下）と、再正規化できないとき（床が
 * 1 以上）は**自分自身を返す**。`noiseFloor` が null のままなので、「引いていない」ことは
 * 戻り値から読める。
 */
export function denoisedHairMask(mask: HairMask): HairMask {
  const floor = measuredNoiseFloor(mask);
  if (floor === null || !(floor > 0 && floor < 1)) return mask;
  const values = new Float32Array(mask.confidence.values.length);
  for (let pixel = 0; pixel < values.length; pixel++) {
    values[pixel] = Math.min(1, Math.max(0, (mask.confidence.values[pixel] - floor) / (1 - floor)));
  }
  return {
    confidence: makeField(values, mask.confidence.width, mask.confidence.height, mask.confidence.rect),
    present: mask.present,
    noiseFloor: floor,
  };
}

/**
 * 深度・法線・人物前景を1回の推論で得た結果。
 *
 * 3つが同じ切り出し矩形・同じ解像度に乗っていることを型で束ねる。別々に返すと呼び出し側
 * が「同じ推論の出力か」を保証できず、矩形の食い違いが黙って通る。
 */
export interface DepthNormalResult {
  /** 相対深度。絶対距離ではなく単調な順序だけが意味を持つ。 */
  readonly depth: ScalarField;
  /**
   * 生の表面法線。RGB エンコードしない。長さ `3 * height * width` で、チャンネルが
   * 外側（`normal[c * height * width + row * width + column]`）。デスクトップ側の
   * `float32 (3, h, w)` と同じ並び。
   */
  readonly normal: Float32Array;
  /** 人物前景のソフトマスク（0..1）。 */
  readonly foreground: ScalarField;
}

export function validateDepthNormal(result: DepthNormalResult): void {
  const { depth, normal, foreground } = result;
  if (normal.length !== 3 * depth.height * depth.width) {
    throw new Error(
      `normal の長さが depth と揃っていない: normal=${normal.length}` +
        ` depth=${depth.width}x${depth.height}`,
    );
  }
  if (foreground.width !== depth.width || foreground.height !== depth.height) {
    throw new Error('foreground の形が depth と揃っていない');
  }
  if (!rectEquals(foreground.rect, depth.rect)) {
    throw new Error('foreground と depth の rect が食い違っている');
  }
}

/** 法線の 1 チャンネルを `ScalarField` として見る（`depth` と同じ rect・解像度）。 */
export function normalChannelField(result: DepthNormalResult, channel: number): ScalarField {
  const { depth, normal } = result;
  const area = depth.width * depth.height;
  return makeField(
    normal.subarray(channel * area, (channel + 1) * area),
    depth.width,
    depth.height,
    depth.rect,
  );
}
