// 写真の色空間と再標本化。
//
// 写真を読む段はどこも同じ2つを必要とする — **sRGB とリニア光の往復**と、**ピクセル
// 座標での標本化**。アトラス（`domain/atlas`）と眼球（`domain/eyes`）と髪テクスチャが
// 同じものを使うので、どれか1つの下に置くと他の2つがそこへ依存する。
//
// 色の演算はすべてリニア光で行う。sRGB のまま平均やぼかしをかけると暗い側へ寄る
// （sRGB は明度に対して非線形なので、2 値の平均が知覚上の中間にならない）。

/** sRGB の線形区間と冪区間の境界（エンコード側の値）。 */
export const SRGB_LINEAR_CUTOFF = 0.04045;

/**
 * uint8 の RGB 写真。
 *
 * ブラウザの canvas は RGBA を返すが、この型は **RGB の 3 チャンネル**で持つ
 * （デスクトップ側の `photo_rgb: uint8 (H, W, 3)` と同じ形）。入力の段で 1 回だけ
 * 落とすので、以降の段はアルファの有無を気にしない。
 */
export interface PhotoRgb {
  readonly data: Uint8Array;
  readonly width: number;
  readonly height: number;
}

/** RGB (H, W, 3) の妥当性を確かめる（各段の入口で同じ検査を書かないため）。 */
export function validatePhoto(photo: PhotoRgb): void {
  if (photo.width <= 0 || photo.height <= 0) {
    throw new Error(`写真の大きさが ${photo.width}x${photo.height}`);
  }
  if (photo.data.length !== photo.width * photo.height * 3) {
    throw new Error(
      `写真は uint8 の (H, W, 3): data=${photo.data.length}` +
        ` 期待 ${photo.width * photo.height * 3}`,
    );
  }
}

/** sRGB エンコード値（0..1）をリニア光にする。 */
export function srgbToLinear(encoded: number): number {
  return encoded <= SRGB_LINEAR_CUTOFF
    ? encoded / 12.92
    : Math.pow((encoded + 0.055) / 1.055, 2.4);
}

/** リニア光を sRGB の 0..255 にする（丸めまで含む）。 */
export function linearToSrgb8(linear: number): number {
  const clamped = Math.min(1, Math.max(0, linear));
  const encoded =
    clamped <= SRGB_LINEAR_CUTOFF / 12.92
      ? clamped * 12.92
      : 1.055 * Math.pow(clamped, 1 / 2.4) - 0.055;
  return Math.round(Math.min(1, Math.max(0, encoded)) * 255);
}

/** リニア光の配列を sRGB の uint8 配列にする。 */
export function linearToSrgb8Array(linear: Float32Array | Float64Array): Uint8Array {
  const out = new Uint8Array(linear.length);
  for (let index = 0; index < linear.length; index++) out[index] = linearToSrgb8(linear[index]);
  return out;
}

let cachedLut: Float32Array | null = null;

/**
 * uint8 の sRGB → リニア光の変換表 (256,)。
 *
 * 写真全体をリニアの float に展開すると 4000×3000 で 140 MB になる。サンプルした
 * 4 タップだけを表引きすれば同じ結果を得られる。
 */
export function srgb8ToLinearLut(): Float32Array {
  if (cachedLut) return cachedLut;
  const lut = new Float32Array(256);
  for (let value = 0; value < 256; value++) lut[value] = srgbToLinear(value / 255);
  cachedLut = lut;
  return lut;
}

/**
 * 写真をピクセル座標で bilinear サンプルし、リニア光で返す。
 *
 * ピクセル座標の規約: 画素 `(row, col)` の中心が `(col + 0.5, row + 0.5)`。原点は
 * 画像の左上、y は下向き（`domain/field` の画像 UV 空間と同じ向き）。
 *
 * @param out 長さ `3 * count` の書き込み先。画像外の色は 0
 * @param inside 長さ `count`。画像内かどうか（`domain.photo.sample_photo_linear` と
 *   同じ判定 — 端は `<= width` / `<= height` を内側とする）
 */
export function samplePhotoLinear(
  photo: PhotoRgb,
  x: Float64Array,
  y: Float64Array,
  out: Float32Array,
  inside: Uint8Array,
): void {
  const { data, width, height } = photo;
  const lut = srgb8ToLinearLut();
  const count = x.length;
  for (let index = 0; index < count; index++) {
    const px = x[index];
    const py = y[index];
    const within = px >= 0 && px <= width && py >= 0 && py <= height;
    inside[index] = within ? 1 : 0;
    if (!within) {
      out[index * 3] = 0;
      out[index * 3 + 1] = 0;
      out[index * 3 + 2] = 0;
      continue;
    }
    const fx = px - 0.5;
    const fy = py - 0.5;
    const x0Real = Math.floor(fx);
    const y0Real = Math.floor(fy);
    const tx = fx - x0Real;
    const ty = fy - y0Real;
    const x0 = Math.min(Math.max(x0Real, 0), width - 1);
    const x1 = Math.min(Math.max(x0Real + 1, 0), width - 1);
    const y0 = Math.min(Math.max(y0Real, 0), height - 1);
    const y1 = Math.min(Math.max(y0Real + 1, 0), height - 1);
    const row0 = y0 * width;
    const row1 = y1 * width;
    for (let channel = 0; channel < 3; channel++) {
      const top =
        lut[data[(row0 + x0) * 3 + channel]] * (1 - tx) + lut[data[(row0 + x1) * 3 + channel]] * tx;
      const bottom =
        lut[data[(row1 + x0) * 3 + channel]] * (1 - tx) + lut[data[(row1 + x1) * 3 + channel]] * tx;
      out[index * 3 + channel] = top * (1 - ty) + bottom * ty;
    }
  }
}

/** 1 点だけ引くときの `samplePhotoLinear`（RGB を長さ 3 の配列へ書く）。 */
export function samplePhotoLinearAt(
  photo: PhotoRgb,
  px: number,
  py: number,
  out: Float32Array,
  offset: number,
): boolean {
  const { data, width, height } = photo;
  const lut = srgb8ToLinearLut();
  const within = px >= 0 && px <= width && py >= 0 && py <= height;
  if (!within) {
    out[offset] = 0;
    out[offset + 1] = 0;
    out[offset + 2] = 0;
    return false;
  }
  const fx = px - 0.5;
  const fy = py - 0.5;
  const x0Real = Math.floor(fx);
  const y0Real = Math.floor(fy);
  const tx = fx - x0Real;
  const ty = fy - y0Real;
  const x0 = Math.min(Math.max(x0Real, 0), width - 1);
  const x1 = Math.min(Math.max(x0Real + 1, 0), width - 1);
  const y0 = Math.min(Math.max(y0Real, 0), height - 1);
  const y1 = Math.min(Math.max(y0Real + 1, 0), height - 1);
  const row0 = y0 * width;
  const row1 = y1 * width;
  for (let channel = 0; channel < 3; channel++) {
    const top =
      lut[data[(row0 + x0) * 3 + channel]] * (1 - tx) + lut[data[(row0 + x1) * 3 + channel]] * tx;
    const bottom =
      lut[data[(row1 + x0) * 3 + channel]] * (1 - tx) + lut[data[(row1 + x1) * 3 + channel]] * tx;
    out[offset + channel] = top * (1 - ty) + bottom * ty;
  }
  return true;
}

/**
 * 長辺が `longestSide` になるまで縮小する。縦横比は変えない。
 *
 * **面積平均**で縮める。bilinear で点標本化すると、2 倍を超える縮小で入力画素の大半を
 * 読まずに捨てるので、髪の細い筋が飛ぶ（エイリアシング）。面積平均は縮小率にかかわらず
 * 入力の全画素が寄与する。
 *
 * **拡大はしない。** 長辺が既に `longestSide` 以下なら写真をそのまま返す（写真に無い
 * 情報を作らないため）。
 *
 * UV は正規化座標なので、縦横比を保った縮小なら対応は変わらない。髪シェルの UV
 * （`domain/field` の画像 UV 空間）がそのまま使える。
 *
 * **行と列を別の段で畳む**（面積重みは軸ごとに分離できる）。
 */
export function resampleLongestSide(photo: PhotoRgb, longestSide: number): PhotoRgb {
  validatePhoto(photo);
  if (longestSide <= 0) throw new Error(`長辺が ${longestSide}`);
  const { width, height } = photo;
  if (Math.max(width, height) <= longestSide) return photo;

  const scale = longestSide / Math.max(width, height);
  const targetHeight = Math.max(1, Math.round(height * scale));
  const targetWidth = Math.max(1, Math.round(width * scale));

  const rows = areaWeights(height, targetHeight);
  const columns = areaWeights(width, targetWidth);
  const lut = srgb8ToLinearLut();

  // 段1 行方向: 入力の列をそのまま保ったまま、出力の行へ畳む。
  const alongRows = new Float32Array(targetHeight * width * 3);
  for (let outRow = 0; outRow < targetHeight; outRow++) {
    const first = rows.offsets[outRow];
    for (let tap = 0; tap < rows.span; tap++) {
      const weight = rows.weights[outRow * rows.span + tap];
      if (weight === 0) continue;
      const sourceRow = Math.min(first + tap, height - 1);
      const sourceBase = sourceRow * width * 3;
      const targetBase = outRow * width * 3;
      for (let index = 0; index < width * 3; index++) {
        alongRows[targetBase + index] += lut[photo.data[sourceBase + index]] * weight;
      }
    }
  }

  // 段2 列方向。
  const out = new Uint8Array(targetHeight * targetWidth * 3);
  for (let outRow = 0; outRow < targetHeight; outRow++) {
    const rowBase = outRow * width * 3;
    for (let outColumn = 0; outColumn < targetWidth; outColumn++) {
      const first = columns.offsets[outColumn];
      let red = 0;
      let green = 0;
      let blue = 0;
      for (let tap = 0; tap < columns.span; tap++) {
        const weight = columns.weights[outColumn * columns.span + tap];
        if (weight === 0) continue;
        const sourceColumn = Math.min(first + tap, width - 1) * 3;
        red += alongRows[rowBase + sourceColumn] * weight;
        green += alongRows[rowBase + sourceColumn + 1] * weight;
        blue += alongRows[rowBase + sourceColumn + 2] * weight;
      }
      const target = (outRow * targetWidth + outColumn) * 3;
      out[target] = linearToSrgb8(red);
      out[target + 1] = linearToSrgb8(green);
      out[target + 2] = linearToSrgb8(blue);
    }
  }
  return { data: out, width: targetWidth, height: targetHeight };
}

/**
 * 面積重みを `(先頭 index, 重み, span)` で返す。行の和は 1。
 *
 * 出力画素 `i` は入力の区間 `[i·source/target, (i+1)·source/target)` を覆う。入力画素
 * との重なりの長さがそのまま重みになる。**`(target, source)` の密行列にしない** —
 * 1 つの出力画素が触る入力は `ceil(source/target) + 1` 個しかない。
 */
function areaWeights(
  source: number,
  target: number,
): { offsets: Int32Array; weights: Float64Array; span: number } {
  if (!(target > 0 && target <= source)) throw new Error(`縮小のみ: ${source} → ${target}`);
  const ratio = source / target;
  const span = Math.ceil(ratio) + 1;
  const offsets = new Int32Array(target);
  const weights = new Float64Array(target * span);
  for (let index = 0; index < target; index++) {
    const low = index * ratio;
    const high = (index + 1) * ratio;
    const first = Math.floor(low);
    offsets[index] = first;
    let total = 0;
    for (let tap = 0; tap < span; tap++) {
      const column = first + tap;
      const overlap =
        column < source ? Math.max(0, Math.min(column + 1, high) - Math.max(column, low)) : 0;
      weights[index * span + tap] = overlap;
      total += overlap;
    }
    for (let tap = 0; tap < span; tap++) weights[index * span + tap] /= total;
  }
  return { offsets, weights, span };
}

/** 重み付き平均に入れる確信度の下限（**この値より大きい**画素だけ使う）。 */
export const MASKED_AVERAGE_MIN_WEIGHT = 0.2;

/** 平均が成立する重みの総和の下限。1 未満は「確信度 1 の画素 1 個ぶんも無い」状態。 */
export const MASKED_AVERAGE_MIN_MASS = 1.0;

/**
 * 確信度で重み付けした写真の平均色を **sRGB の [0,1]** で返す（足りなければ null）。
 *
 * **リニアへ直さない。** これは表示する色ではなく、公式の頂点色の規則
 * `色 = 肌色 × scale + offset` へ渡す入力で、その式は sRGB の値に対して書かれている。
 * この値を実際に使うのは `domain/atlas/bake` の口腔壁の塗り（`interiorScale`）で、
 * そこも sRGB で掛ける。リニアで平均を取ってから sRGB へ戻すと明るい側へ寄り、焼き込む
 * 口腔壁の色が公式の演算からずれる。
 *
 * @param weight 確信度 0..1。**写真と同じ解像度**（`photo.width * photo.height`）
 */
export function maskedAverageSrgb(
  photo: PhotoRgb,
  weight: Float32Array,
  minWeight = MASKED_AVERAGE_MIN_WEIGHT,
  minMass = MASKED_AVERAGE_MIN_MASS,
): [number, number, number] | null {
  validatePhoto(photo);
  if (weight.length !== photo.width * photo.height) {
    throw new Error(
      `重みの長さ ${weight.length} が写真 ${photo.width * photo.height} と揃っていない`,
    );
  }
  let mass = 0;
  const total = [0, 0, 0];
  for (let pixel = 0; pixel < weight.length; pixel++) {
    const used = weight[pixel] > minWeight ? weight[pixel] : 0;
    if (used === 0) continue;
    mass += used;
    total[0] += used * photo.data[pixel * 3];
    total[1] += used * photo.data[pixel * 3 + 1];
    total[2] += used * photo.data[pixel * 3 + 2];
  }
  if (mass < minMass) return null;
  return [total[0] / mass / 255, total[1] / mass / 255, total[2] / mass / 255];
}
