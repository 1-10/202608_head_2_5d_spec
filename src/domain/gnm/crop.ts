// 推論の切り出し矩形。
//
// DAViD の入力は正方形なので、写真のどこを正方形に切って推論するかを必ず誰かが決める。
// ここはその1箇所。
//
// **切り出しは「メッシュが写真のどこを占めるか」から決める。** 深度・法線・人物前景を
// 消費するのはアトラスのベイクと髪シェルの 2 つで、**メッシュの投影域に縛られているのは
// 前者だけ**。だから覆うべき範囲の下限はメッシュの投影域で、写真の額縁とは関係が無い。
//
// 正方形は 2 つある
// -----------------
// **深度と人物前景で、覆うべき範囲が違う。** 前景はメッシュ全体（胸まで）を覆わないと
// アトラスが壊れ、深度は広く覆うほど隣人が入って壊れる。1 枚で両方は満たせない:
//
//     headInferenceSquare   メッシュ全体を覆う広い方。人物前景に使う
//     headOnlySquare        頭部だけを覆う詰めた方。深度・法線に使う
//
// 画像の中央正方形で切ってはいけない
// ----------------------------------
// 以前は画像に収まる最大の中央正方形で切っていた。切り出しが**被写体ではなく写真の額縁で
// 決まる**ので、同じ人物でも縦長に撮るか横長に撮るかで推論の入力が別物になる。実害が出た
// 例（540x960 の縦写真）: 切り出しは行 210〜750。顎から下がまるごと推論の外に落ちる。
// `ScalarField` は rect 外を 0 で返すので、胸と肩のテクセルは人物前景 0 = 背景と読まれ、
// `BakeSettings.foregroundThreshold` の門で棄却される。写真に服が写っていても
// `skin_albedo` の胸は補完色（肌色）で埋まり、**服を着ていない見た目になる**。
//
// 粗い相似変換で足りる
// --------------------
// 切り出しを決める時点ではフィットがまだ走っていない。使うのは**平均顔を写真のランドマーク
// へ重ねただけの相似変換**で、これは `fitHead` の第1周が解くものと同一
// （`coarseSimilarity`）。平均顔なので最終フィットとはずれる。そのぶんを
// `DEFAULT_INFERENCE_CROP_MARGIN` の余白で吸収する。

import { LandmarkModel, Similarity2d, coarseSimilarity } from './fit';
import { IBUG68_POINT_COUNT } from './model';

/**
 * メッシュ投影域の各辺へ足す余白（投影域の長辺に対する割合）。
 *
 * 吸収するのは**粗い相似変換と最終フィットのずれ**。粗い側は平均顔なので、identity が
 * 効いた最終形とは大きさも位置も違う。`refineEarNeckFit` が耳と首を外へ動かす分もここに
 * 入る。**実測で決めた**（デスクトップ側の実測で片側はみ出しの最大 0.084 に対して 0.15）。
 * 余白を増やすほど 512² に占める頭の割合が減るので、実測の倍弱で止める。
 */
export const DEFAULT_INFERENCE_CROP_MARGIN = 0.15;

/** 推論に渡す正方領域（画素）。`DepthNormalEstimator.estimateSquare` にそのまま渡せる形。 */
export interface Square {
  readonly x: number;
  readonly y: number;
  readonly size: number;
}

/**
 * 写真のランドマークから、メッシュを覆う正方形の切り出しを決める。
 *
 * @param photoLandmarks (478, 2) 検出器の出力そのまま（画像ピクセル座標）
 * @param meshXy (頂点数, 2) GNM 空間の**平均形状**の xy。identity を当てた形ではない
 * @param imageSize (幅, 高さ) 画素
 */
export function headInferenceSquare(
  photoLandmarks: Float64Array,
  landmarkModel: LandmarkModel,
  meshXy: Float64Array,
  imageSize: readonly [number, number],
  margin = DEFAULT_INFERENCE_CROP_MARGIN,
): Square {
  assertImageSize(imageSize);
  if (margin < 0) throw new Error(`余白が負: ${margin}`);
  return projectedSquare(
    coarseSimilarity(photoLandmarks, landmarkModel),
    meshXy,
    imageSize,
    margin,
  );
}

/**
 * 相似変換で写真へ射影したメッシュを覆う正方形（画素）。
 *
 * **bbox の 4 隅を変換してから包む**（変換してから min/max を取る）。相似変換は回転を
 * 含むので、GNM 空間の bbox をそのまま写真の bbox として扱うと回った分だけ足りない。
 *
 * 覆いきれないときは頭頂側を残す
 * ------------------------------
 * 正方形が画像の短辺に収まらないとき、投影域の中心に置くと**頭頂がはみ出す**（メッシュは
 * 頭から胸へ下に伸びているので、中心は顔より下にある）。落とすなら胸側を落とす — **胸は
 * 肌アトラスだけの領域で、写真が届かなければ補完へ回せる。頭頂は髪シェルの材料で、代わりが
 * 無い**。
 *
 * **寄せるのは頭頂の向きが乗っている軸だけ、しかも投影域そのものが収まらないときだけ。**
 * どちらの条件を落としても、覆えるものを落とす（もう一方の軸では `crown` の成分がほぼ 0 で
 * 数値誤差の符号で左右どちらを切るかが決まってしまい、余白の分まで収まるかで判定すると
 * 投影域は収まる軸でも端へ寄る）。
 */
export function projectedSquare(
  similarity: Similarity2d,
  meshXy: Float64Array,
  imageSize: readonly [number, number],
  margin = DEFAULT_INFERENCE_CROP_MARGIN,
): Square {
  const [width, height] = imageSize;
  const pixels = projectedCorners(similarity, meshXy);
  const low = [Math.min(...pixels.map((p) => p[0])), Math.min(...pixels.map((p) => p[1]))];
  const high = [Math.max(...pixels.map((p) => p[0])), Math.max(...pixels.map((p) => p[1]))];
  const span = [high[0] - low[0], high[1] - low[1]];
  const maximumSpan = Math.max(span[0], span[1]);

  let side = maximumSpan * (1 + 2 * margin);
  if (!(side > 0)) {
    throw new Error('メッシュの投影域が 1 点に潰れている（相似変換が退化している）');
  }
  side = Math.min(side, Math.min(width, height));

  // 頭頂の向き（画像画素）。GNM 空間の +Y が上、画像は行方向が下なので普通は y が負。
  const crown = [
    similarity.linear[1],
    similarity.linear[3],
  ];
  const crownAxis = Math.abs(crown[0]) >= Math.abs(crown[1]) ? 0 : 1;
  // 寄せるときは頭頂側へ `inset` だけ踏み越す。メッシュの頭頂の**さらに外側**に髪があるので、
  // 頭頂を正方形の端ちょうどに置くと髪が切り出しの外へ出る。
  const inset = margin * maximumSpan;
  const center = [0, 0];
  for (let axis = 0; axis < 2; axis++) {
    // **投影域そのものが収まるなら中心に置く。**
    if (axis !== crownAxis || high[axis] - low[axis] <= side) {
      center[axis] = (low[axis] + high[axis]) / 2;
    } else if (crown[axis] >= 0) {
      center[axis] = high[axis] + inset - side / 2;
    } else {
      center[axis] = low[axis] - inset + side / 2;
    }
  }

  const x = Math.min(Math.max(center[0] - side / 2, 0), width - side);
  const y = Math.min(Math.max(center[1] - side / 2, 0), height - side);
  // 丸めは最後に1回だけ。size を先に丸めると x + size が画像を 1 画素はみ出しうる。
  const size = Math.max(1, Math.round(side));
  return {
    x: Math.round(Math.min(x, width - size)),
    y: Math.round(Math.min(y, height - size)),
    size,
  };
}

/**
 * 写真のランドマークから、**頭部だけ**を覆う詰めた正方形の切り出しを決める。
 *
 * なぜ詰めるのか — 隣人が深度を壊す。`headInferenceSquare` は一辺が顔幅の約 3.0 倍あり、
 * メッシュの x 半幅は 0.86 顔幅しかないので**横 0.6 顔幅は純粋な余り**で、そこに隣の人が
 * 入る。隣人が入ると本人の頭部の z が 0.91〜5.22mm ずれる（深度の場が非アフィンに歪むので
 * 相似変換で戻せない）。一辺を 2.2 顔幅へ詰めると z のずれが 0.26〜1.18mm に落ち、頭の
 * 実効解像度が 172 → 233px に上がる。
 *
 * 詰めると胸が落ちる（メッシュ被覆 0.944〜0.974）。**呼び出し側は同じ写真に推論を 2 回
 * 掛ける** — 深度・法線はこの正方形、人物前景は `headInferenceSquare`。
 */
export function headOnlySquare(
  photoLandmarks: Float64Array,
  landmarkModel: LandmarkModel,
  meshXy: Float64Array,
  imageSize: readonly [number, number],
  margin = DEFAULT_INFERENCE_CROP_MARGIN,
): Square {
  assertImageSize(imageSize);
  if (margin < 0) throw new Error(`余白が負: ${margin}`);
  return projectedHeadOnlySquare(
    coarseSimilarity(photoLandmarks, landmarkModel),
    meshXy,
    meanChinHeight(landmarkModel),
    imageSize,
    margin,
  );
}

/**
 * 相似変換で写真へ射影した**頭部**を覆う正方形（画素）。
 *
 * 覆う範囲は狭め、余白は狭めない — 頭部の bbox に足す余白は
 * `margin × メッシュ全体の投影域の長辺`。**頭部の投影域に対する割合ではない。** 余白が
 * 吸収するのは 2 つとも「覆う範囲を狭めても小さくならない絶対量」だから（粗い相似変換と
 * 最終フィットのずれ / メッシュの外側にある髪）。頭部の割合で置き直すと、狭めたぶんだけ
 * 余白まで一緒に縮んで髪が切れる。
 *
 * @param meshXy (頂点数, 2) メッシュ**全体**の xy。頭部だけを渡さないこと
 * @param chinHeight GNM 空間の y。**これより上の頂点を頭部とみなす**
 */
export function projectedHeadOnlySquare(
  similarity: Similarity2d,
  meshXy: Float64Array,
  chinHeight: number,
  imageSize: readonly [number, number],
  margin = DEFAULT_INFERENCE_CROP_MARGIN,
): Square {
  if (margin < 0) throw new Error(`余白が負: ${margin}`);
  const headPoints: number[] = [];
  for (let vertex = 0; vertex < meshXy.length / 2; vertex++) {
    if (meshXy[vertex * 2 + 1] >= chinHeight) {
      headPoints.push(meshXy[vertex * 2], meshXy[vertex * 2 + 1]);
    }
  }
  if (headPoints.length === 0) throw new Error(`顎の高さ ${chinHeight} より上に頂点が無い`);
  const head = Float64Array.from(headPoints);

  const wholeSpan = spanOfProjection(similarity, meshXy);
  const headSpan = spanOfProjection(similarity, head);
  if (!(headSpan > 0)) {
    throw new Error('頭部の投影域が 1 点に潰れている（相似変換が退化している）');
  }
  return projectedSquare(similarity, head, imageSize, (margin * wholeSpan) / headSpan);
}

/**
 * 平均顔の 68 点の最下点（GNM 空間の y）= 顎先の高さ。
 *
 * index を直に書かずに最小値で取るのは、iBUG-68 の番号を読み手が覚えていなくても意味が
 * 読めるようにするため。
 */
export function meanChinHeight(landmarkModel: LandmarkModel): number {
  let lowest = Infinity;
  for (let slot = 0; slot < IBUG68_POINT_COUNT; slot++) {
    const y = landmarkModel.meanPositions[landmarkModel.guardRows[slot] * 3 + 1];
    if (y < lowest) lowest = y;
  }
  return lowest;
}

function spanOfProjection(similarity: Similarity2d, pointsXy: Float64Array): number {
  const corners = projectedCorners(similarity, pointsXy);
  const spanX =
    Math.max(...corners.map((p) => p[0])) - Math.min(...corners.map((p) => p[0]));
  const spanY =
    Math.max(...corners.map((p) => p[1])) - Math.min(...corners.map((p) => p[1]));
  return Math.max(spanX, spanY);
}

/**
 * 点群の bbox の 4 隅を写真へ射影する（画素）。
 *
 * **bbox を作ってから変換する。** 相似変換は回転を含むので、GNM 空間の bbox をそのまま
 * 写真の bbox として扱うと回った分だけ足りない。
 */
function projectedCorners(
  similarity: Similarity2d,
  pointsXy: Float64Array,
): [number, number][] {
  if (pointsXy.length === 0 || pointsXy.length % 2 !== 0) {
    throw new Error(`meshXy は (頂点数, 2): length=${pointsXy.length}`);
  }
  let lowX = Infinity;
  let lowY = Infinity;
  let highX = -Infinity;
  let highY = -Infinity;
  for (let vertex = 0; vertex < pointsXy.length / 2; vertex++) {
    const x = pointsXy[vertex * 2];
    const y = pointsXy[vertex * 2 + 1];
    if (x < lowX) lowX = x;
    if (x > highX) highX = x;
    if (y < lowY) lowY = y;
    if (y > highY) highY = y;
  }
  return [
    similarity.applyPoint(lowX, lowY),
    similarity.applyPoint(highX, lowY),
    similarity.applyPoint(lowX, highY),
    similarity.applyPoint(highX, highY),
  ];
}

function assertImageSize(imageSize: readonly [number, number]): void {
  if (!(imageSize[0] > 0 && imageSize[1] > 0)) {
    throw new Error(`画像の大きさが ${imageSize[0]}x${imageSize[1]}`);
  }
}
