// 写真をガイドにした髪マスク精細化と、半透明境界の色補正。
//
// SelfieMulticlass のマスクを単に bilinear で写真解像度へ広げると、モデル解像度のぼけとブロックが
// そのまま `hair_alpha` に残る。写真 RGB と粗いマスクの局所線形関係を解き、写真に実在する色エッジへ
// マスクを整合させる。
//
// ここに置くのは純粋計算。推論モデル固有の処理ではなく、別のセグメンタが返した `HairMask` にも同じ
// 段を適用できる。実装は局所平均・共分散・3x3 対称行列の線形代数から組み立てており、外部の参照実装や
// 学習済みモデルを含まない。
//
// 精細化後も役割は分ける:
//
//   - `confidence`: ソフト alpha・厚み・縁の重み
//   - `present`: 髪シェルが覆うかという判定。二値場を -1..1 の符号付き場にして同じ写真ガイドで
//     整合し、0 との比較で戻すため、消費側の調整閾値は増えない
//
// `decontaminateHairTexture` は alpha 境界の写真色から近傍背景を引き、透明側へ近傍の髪色を余白として
// 延ばす。alpha が正しくても RGB に背景色が残ると、bilinear・mipmap・JPEG がその色を可視側へ混ぜる
// ため、マスク精細化とは別に必要な段。

import { AlphaImage, RgbImage } from '../contract';
import { HairMask, makeHairMask, fieldOverFullImage, resampleFieldToImage } from '../field';
import {
  PhotoRgb,
  linearToSrgb8,
  resampleLongestSide,
  srgb8ToLinearLut,
  validatePhoto,
} from '../photo';
import { smoothstep } from '../ramp';

/**
 * 写真ガイドで精細化する最大長辺。最終 alpha はここから出力解像度へ引き直す。
 *
 * 2048 で局所統計を十数枚持つと、写真に無いマスク情報を増やさないまま一時メモリだけが 4 倍になる。
 * 1024 なら 256px マスクの各セルを最大4倍へ解き、髪テクスチャ側のbilinearで境界を滑らかに保つのに
 * 十分。
 */
export const GUIDED_MASK_MAX_DIMENSION = 1024;

/** 局所窓の半径を作業画像の長辺に対する比で持つ。 */
export const GUIDED_RADIUS_FRACTION = 0.02;

/** RGB 共分散の正則化。RGB は 0..1。小さすぎると写真ノイズへ追従する。 */
export const GUIDED_EPSILON = 1e-3;

/** 境界色補正で前景・背景色を測る半径。2048px で約12px。 */
export const EDGE_COLOR_RADIUS_FRACTION = 0.006;

/** 写真 RGB をガイドに `HairMask` の境界を高解像度化する。 */
export function refineHairMaskWithPhoto(
  photo: PhotoRgb,
  mask: HairMask,
  options: {
    maximumDimension?: number;
    radiusFraction?: number;
    epsilon?: number;
  } = {},
): HairMask {
  validatePhoto(photo);
  const maximumDimension = options.maximumDimension ?? GUIDED_MASK_MAX_DIMENSION;
  const radiusFraction = options.radiusFraction ?? GUIDED_RADIUS_FRACTION;
  const epsilon = options.epsilon ?? GUIDED_EPSILON;
  if (maximumDimension <= 0) throw new Error(`maximumDimension は正: ${maximumDimension}`);
  if (radiusFraction <= 0) throw new Error(`radiusFraction は正: ${radiusFraction}`);
  if (epsilon <= 0) throw new Error(`epsilon は正: ${epsilon}`);

  const guidePhoto = resampleLongestSide(photo, maximumDimension);
  const { width, height } = guidePhoto;
  const confidence = resampleFieldToImage(mask.confidence, width, height);
  // 二値判定を符号付きにして精細化する。0 を境に戻すので、任意の確信度閾値ではない。
  const decisionSource = resampleFieldToImage(mask.present, width, height);
  const decision = new Float32Array(width * height);
  for (let pixel = 0; pixel < decision.length; pixel++) {
    decision[pixel] = decisionSource[pixel] * 2 - 1;
  }

  const guide = new Float32Array(width * height * 3);
  for (let index = 0; index < guide.length; index++) guide[index] = guidePhoto.data[index] / 255;
  const radius = Math.max(2, Math.round(Math.max(width, height) * radiusFraction));
  const refined = guidedFilterColor(guide, [confidence, decision], width, height, radius, epsilon);

  const refinedConfidence = new Float32Array(width * height);
  const refinedPresent = new Float32Array(width * height);
  for (let pixel = 0; pixel < width * height; pixel++) {
    refinedConfidence[pixel] = Math.min(1, Math.max(0, refined[0][pixel]));
    refinedPresent[pixel] = refined[1][pixel] > 0 ? 1 : 0;
  }
  return makeHairMask(
    fieldOverFullImage(refinedConfidence, width, height),
    fieldOverFullImage(refinedPresent, width, height),
    mask.noiseFloor,
  );
}

/**
 * 半透明境界から背景色を除き、透明側へ近傍の髪色を延ばす。
 *
 * 写真画素 `C` を `C = alpha * F + (1-alpha) * B` とみなし、近傍から測った背景 `B` で前景 `F` を
 * 戻す。alpha が小さい所は除算が不安定なので、alpha 重み付きで測った近傍前景色へ滑らかに寄せる。
 * 完全透明な遠方は見えないので触らず、局所窓の中に髪色が届く余白だけを書き換える。
 */
export function decontaminateHairTexture(photo: RgbImage, alpha: AlphaImage): RgbImage {
  if (alpha.width !== photo.width || alpha.height !== photo.height) {
    throw new Error(
      `写真 ${photo.width}x${photo.height} と alpha ${alpha.width}x${alpha.height} の形が` +
        '揃っていない',
    );
  }
  const area = photo.width * photo.height;
  const alphaValues = new Float32Array(area);
  let anyOpaque = false;
  let anyTransparent = false;
  for (let pixel = 0; pixel < area; pixel++) {
    alphaValues[pixel] = alpha.data[pixel] / 255;
    if (alphaValues[pixel] > 0) anyOpaque = true;
    if (alphaValues[pixel] < 1) anyTransparent = true;
  }
  // 全透明・全不透明なら境界が無い。0/1 のハード境界は透明側への色延長が必要なので処理を続ける。
  if (!anyOpaque || !anyTransparent) return photo;

  const lut = srgb8ToLinearLut();
  const color = new Float32Array(area * 3);
  for (let index = 0; index < color.length; index++) color[index] = lut[photo.data[index]];
  const radius = Math.max(
    4,
    Math.round(Math.max(photo.width, photo.height) * EDGE_COLOR_RADIUS_FRACTION),
  );

  // 半透明画素は背景を含むので、前景色の標本では高い alpha を強く優先する。
  const foregroundWeight = new Float32Array(area);
  const backgroundWeight = new Float32Array(area);
  for (let pixel = 0; pixel < area; pixel++) {
    foregroundWeight[pixel] = Math.pow(alphaValues[pixel], 4);
    backgroundWeight[pixel] = Math.pow(1 - alphaValues[pixel], 2);
  }
  const foregroundMass = boxMean(foregroundWeight, photo.width, photo.height, radius);
  const backgroundMass = boxMean(backgroundWeight, photo.width, photo.height, radius);
  const foreground = weightedBoxColor(
    color,
    foregroundWeight,
    foregroundMass,
    photo.width,
    photo.height,
    radius,
  );
  const background = weightedBoxColor(
    color,
    backgroundWeight,
    backgroundMass,
    photo.width,
    photo.height,
    radius,
  );

  const out = new Uint8Array(area * 3);
  for (let pixel = 0; pixel < area; pixel++) {
    const alphaValue = alphaValues[pixel];
    const safeAlpha = Math.max(alphaValue, 0.08);
    // alpha が小さい所では逆算誤差を増幅せず近傍髪色を使う。0.2..0.8 の間だけ遷移。
    const trust = smoothstep(0.2, 0.8, alphaValue);
    const boundary =
      alphaValue > 0 && alphaValue < 0.98 && foregroundMass[pixel] > 1e-6 &&
      // 背景標本が無い（写真全体が髪等）場所では色を逆算しない。
      backgroundMass[pixel] > 1e-6;
    const padding = alphaValue <= 0 && foregroundMass[pixel] > 1e-4;
    for (let channel = 0; channel < 3; channel++) {
      const source = color[pixel * 3 + channel];
      let value = source;
      if (boundary) {
        const unmixed = Math.min(
          1,
          Math.max(0, (source - (1 - alphaValue) * background[pixel * 3 + channel]) / safeAlpha),
        );
        value = foreground[pixel * 3 + channel] * (1 - trust) + unmixed * trust;
      } else if (padding) {
        // alpha 0 の直外側にも髪色を置き、JPEG/bilinear/mipmap が背景色を縁へ戻さないようにする。
        value = foreground[pixel * 3 + channel];
      }
      out[pixel * 3 + channel] = linearToSrgb8(value);
    }
  }
  return { data: out, width: photo.width, height: photo.height };
}

/** カラーガイドの局所線形フィルタ。`sources` は同じ形のスカラー場の並び。 */
function guidedFilterColor(
  guide: Float32Array,
  sources: readonly Float32Array[],
  width: number,
  height: number,
  radius: number,
  epsilon: number,
): Float32Array[] {
  if (radius < 0) throw new Error(`radius は 0 以上: ${radius}`);
  const area = width * height;
  const red = new Float32Array(area);
  const green = new Float32Array(area);
  const blue = new Float32Array(area);
  for (let pixel = 0; pixel < area; pixel++) {
    red[pixel] = guide[pixel * 3];
    green[pixel] = guide[pixel * 3 + 1];
    blue[pixel] = guide[pixel * 3 + 2];
  }
  const meanRed = boxMean(red, width, height, radius);
  const meanGreen = boxMean(green, width, height, radius);
  const meanBlue = boxMean(blue, width, height, radius);
  const product = (first: Float32Array, second: Float32Array): Float32Array => {
    const out = new Float32Array(area);
    for (let pixel = 0; pixel < area; pixel++) out[pixel] = first[pixel] * second[pixel];
    return out;
  };
  const meanRedRed = boxMean(product(red, red), width, height, radius);
  const meanRedGreen = boxMean(product(red, green), width, height, radius);
  const meanRedBlue = boxMean(product(red, blue), width, height, radius);
  const meanGreenGreen = boxMean(product(green, green), width, height, radius);
  const meanGreenBlue = boxMean(product(green, blue), width, height, radius);
  const meanBlueBlue = boxMean(product(blue, blue), width, height, radius);

  const minorRr = new Float32Array(area);
  const minorRg = new Float32Array(area);
  const minorRb = new Float32Array(area);
  const minorGg = new Float32Array(area);
  const minorGb = new Float32Array(area);
  const minorBb = new Float32Array(area);
  const inverseDeterminant = new Float32Array(area);
  for (let pixel = 0; pixel < area; pixel++) {
    const rr = meanRedRed[pixel] - meanRed[pixel] * meanRed[pixel] + epsilon;
    const rg = meanRedGreen[pixel] - meanRed[pixel] * meanGreen[pixel];
    const rb = meanRedBlue[pixel] - meanRed[pixel] * meanBlue[pixel];
    const gg = meanGreenGreen[pixel] - meanGreen[pixel] * meanGreen[pixel] + epsilon;
    const gb = meanGreenBlue[pixel] - meanGreen[pixel] * meanBlue[pixel];
    const bb = meanBlueBlue[pixel] - meanBlue[pixel] * meanBlue[pixel] + epsilon;
    minorRr[pixel] = gg * bb - gb * gb;
    minorRg[pixel] = rb * gb - rg * bb;
    minorRb[pixel] = rg * gb - rb * gg;
    minorGg[pixel] = rr * bb - rb * rb;
    minorGb[pixel] = rg * rb - rr * gb;
    minorBb[pixel] = rr * gg - rg * rg;
    const determinant = rr * minorRr[pixel] + rg * minorRg[pixel] + rb * minorRb[pixel];
    inverseDeterminant[pixel] = Math.abs(determinant) > 1e-12 ? 1 / determinant : 0;
  }

  return sources.map((values) => {
    const meanValues = boxMean(values, width, height, radius);
    const covarianceRed = boxMean(product(red, values), width, height, radius);
    const covarianceGreen = boxMean(product(green, values), width, height, radius);
    const covarianceBlue = boxMean(product(blue, values), width, height, radius);
    const coefficientRed = new Float32Array(area);
    const coefficientGreen = new Float32Array(area);
    const coefficientBlue = new Float32Array(area);
    const intercept = new Float32Array(area);
    for (let pixel = 0; pixel < area; pixel++) {
      const rp = covarianceRed[pixel] - meanRed[pixel] * meanValues[pixel];
      const gp = covarianceGreen[pixel] - meanGreen[pixel] * meanValues[pixel];
      const bp = covarianceBlue[pixel] - meanBlue[pixel] * meanValues[pixel];
      coefficientRed[pixel] =
        (minorRr[pixel] * rp + minorRg[pixel] * gp + minorRb[pixel] * bp) *
        inverseDeterminant[pixel];
      coefficientGreen[pixel] =
        (minorRg[pixel] * rp + minorGg[pixel] * gp + minorGb[pixel] * bp) *
        inverseDeterminant[pixel];
      coefficientBlue[pixel] =
        (minorRb[pixel] * rp + minorGb[pixel] * gp + minorBb[pixel] * bp) *
        inverseDeterminant[pixel];
      intercept[pixel] =
        meanValues[pixel] -
        coefficientRed[pixel] * meanRed[pixel] -
        coefficientGreen[pixel] * meanGreen[pixel] -
        coefficientBlue[pixel] * meanBlue[pixel];
    }
    const meanCoefficientRed = boxMean(coefficientRed, width, height, radius);
    const meanCoefficientGreen = boxMean(coefficientGreen, width, height, radius);
    const meanCoefficientBlue = boxMean(coefficientBlue, width, height, radius);
    const meanIntercept = boxMean(intercept, width, height, radius);
    const out = new Float32Array(area);
    for (let pixel = 0; pixel < area; pixel++) {
      out[pixel] =
        meanCoefficientRed[pixel] * red[pixel] +
        meanCoefficientGreen[pixel] * green[pixel] +
        meanCoefficientBlue[pixel] * blue[pixel] +
        meanIntercept[pixel];
    }
    return out;
  });
}

/** 積分画像による、画像端で窓を切り詰めた正規化 box mean。 */
export function boxMean(
  values: Float32Array,
  width: number,
  height: number,
  radius: number,
): Float32Array {
  const integral = new Float64Array((height + 1) * (width + 1));
  for (let row = 0; row < height; row++) {
    let rowTotal = 0;
    for (let column = 0; column < width; column++) {
      rowTotal += values[row * width + column];
      integral[(row + 1) * (width + 1) + column + 1] =
        integral[row * (width + 1) + column + 1] + rowTotal;
    }
  }
  const out = new Float32Array(width * height);
  for (let row = 0; row < height; row++) {
    const rowLow = Math.max(row - radius, 0);
    const rowHigh = Math.min(row + radius + 1, height);
    for (let column = 0; column < width; column++) {
      const columnLow = Math.max(column - radius, 0);
      const columnHigh = Math.min(column + radius + 1, width);
      const total =
        integral[rowHigh * (width + 1) + columnHigh] -
        integral[rowLow * (width + 1) + columnHigh] -
        integral[rowHigh * (width + 1) + columnLow] +
        integral[rowLow * (width + 1) + columnLow];
      out[row * width + column] = total / ((rowHigh - rowLow) * (columnHigh - columnLow));
    }
  }
  return out;
}

function weightedBoxColor(
  color: Float32Array,
  weight: Float32Array,
  mass: Float32Array,
  width: number,
  height: number,
  radius: number,
): Float32Array {
  const area = width * height;
  const out = new Float32Array(area * 3);
  for (let channel = 0; channel < 3; channel++) {
    const weighted = new Float32Array(area);
    for (let pixel = 0; pixel < area; pixel++) {
      weighted[pixel] = color[pixel * 3 + channel] * weight[pixel];
    }
    const mean = boxMean(weighted, width, height, radius);
    for (let pixel = 0; pixel < area; pixel++) {
      if (mass[pixel] > 1e-12) out[pixel * 3 + channel] = mean[pixel] / mass[pixel];
    }
  }
  return out;
}
