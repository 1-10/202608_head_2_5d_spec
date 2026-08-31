// 主役の頭から繋がっている髪だけを残す。
//
// セグメンタはセマンティック分割なので、**写真に何人写っていても髪は 1 枚のマスクに入る**。
// 「誰の髪か」はモデルの側に無い情報で、モデルの外で与えるしかない。
//
// 隣人が居ると格子が主役に割けなくなる
// ------------------------------------
// 髪シェルの格子は髪マスクの bbox から張られる。隣に人が立つと bbox が横へ膨らむが、格子の列数
// （`GRID_COLUMNS`）は変わらない。実測: 横並びの隣人を足すと bbox 幅が 3.6 倍になり、96 列のうち
// 主役の髪に当たる列が 1/2.3〜1/4 に落ちた。**主役の髪の解像度がそのまま落ちる。**
//
// 主役の頭皮から髪を伝って届く範囲だけを採る
// ------------------------------------------
// 1. GNM の頂点を相似変換で写真へ射影し、落ちた画素を**種**にする
// 2. 種からの chamfer 距離 d を、顔の外接正方形の一辺で割って正規化する
// 3. `(d ≤ SUBJECT_SEED_DISTANCE) ∩ 髪マスク` を出発点に、**髪マスクの中だけを** 4 近傍 BFS で
//    成長させる
// 4. 採用 = `(d ≤ SUBJECT_SEED_DISTANCE) ∪ (測地距離 ≤ SUBJECT_GEODESIC_DISTANCE)`
//
// **他人の顔を一切読まない。** これが本質的な利点で、後ろの人の顔が検出できなくても結果が 1 画素も
// 変わらない。他人の顔位置で領域を分ける案（Voronoi）は却下した — **後ろに立つ人の配置で主役の髪が
// 21.5% 消えた**。主役の髪を削る失敗は、隣人の髪が残る失敗より重い（殻の材料には代わりが無い）。
//
// 種は「GNM メッシュの投影域」でなければならない
// ----------------------------------------------
// 種を**顔のランドマークの凸包**にした案では保持が 0.0% に壊れた（顔の凸包は髪マスクとほとんど
// 交わらず、出発点が空になる）。GNM の頂点は頭皮・耳・首・胸まで覆うので、髪マスクと必ず重なる。
//
// **保持が常に 100% であることがこの実装の要件**で、除去率は落ちてよい。隣人が近いほど除去は
// 落ちるが、保持は落ちない。この非対称は狙ったもので、逆にはしない。
//
// 低い解像度で解く
// ----------------
// chamfer も BFS も画素数に比例するので、写真の解像度そのままで解くと 4000px 級の写真で数秒
// かかる。**顔が `SUBJECT_WORKING_FACE_PIXELS` 相当になる格子まで落として解き、採用の可否を
// ブロックごと元の解像度へ戻す。** 落とすときは「ブロック内に髪が 1 画素でもあれば髪」（OR）で、
// 戻すときはブロックごと採用する — どちらも**採用側へ膨らむ向き**なので、縮小が原因で主役の髪が
// 削れることが無い。

import { LandmarkModel, Similarity2d } from '../gnm/fit';
import { IBUG68_POINT_COUNT } from '../gnm/model';

/**
 * 種からの chamfer 距離の上限（顔の外接正方形の一辺に対する比）。
 *
 * この距離までは**髪マスクに繋がっていなくても採用する**。髪はセグメンタの縁で切れることがあり、
 * 繋がりだけを頼りにすると縁の外側の後れ毛が落ちるため。実測で「髪が頭部投影から離れる距離」の
 * 最大が 0.40。その 1.75 倍を採った。**0.40 を下回らせないこと。**
 */
export const SUBJECT_SEED_DISTANCE = 0.7;

/**
 * 出発点からの測地距離（髪マスクの中の 4 近傍歩数）の上限（同じく顔の外接正方形比）。
 *
 * **下げてはいけない。** 擬似ロングヘア（下へ +4.5 顔幅）を通す下限がここで、上限 1.5 では保持が
 * 82.0% に落ちた。上へは ∞ まで安全（隣人は髪マスクの中で繋がっていないので、緩めても入って
 * 来ない）。3.0 は「安全側の下限」であって最適値ではない。
 */
export const SUBJECT_GEODESIC_DISTANCE = 3.0;

/**
 * 解くときに顔の外接正方形の一辺を何画素にするか。
 *
 * **答えではなく速さだけを決める値**（縮小しても採用の集合が変わらない）。32px では 1 画素が顔の
 * 3% になり、細い毛束が縮小の丸めで太って隣人と橋が架かりうる。取り分が小さいので安全な側に置く。
 */
export const SUBJECT_WORKING_FACE_PIXELS = 64.0;

/** chamfer の斜め方向の重み。縦横を 1 とした 3x3 chamfer（Euclid との差は数%）。 */
export const DIAGONAL_STEP = Math.SQRT2;

/**
 * 顔の外接正方形の一辺（写真の画素）。
 *
 * **平均顔の 68 点の外接正方形を相似変換で写真へ写した長さ**。写真のランドマークの bbox から直に
 * 測らないのは、そちらが顔の傾きで伸び縮みするため。相似変換は等方なので `scale` を掛けるだけで
 * 足りる。`LandmarkModel.faceWidth`（68 点の x 幅）ではなく正方形の一辺を使うのは、髪が伸びる
 * 向きが縦横どちらでもありうるから。
 */
export function faceSquareSide(similarity: Similarity2d, landmarkModel: LandmarkModel): number {
  let lowX = Infinity;
  let lowY = Infinity;
  let highX = -Infinity;
  let highY = -Infinity;
  for (let slot = 0; slot < IBUG68_POINT_COUNT; slot++) {
    const row = landmarkModel.guardRows[slot];
    const x = landmarkModel.meanPositions[row * 3];
    const y = landmarkModel.meanPositions[row * 3 + 1];
    lowX = Math.min(lowX, x);
    highX = Math.max(highX, x);
    lowY = Math.min(lowY, y);
    highY = Math.max(highY, y);
  }
  const side = Math.max(highX - lowX, highY - lowY) * similarity.scale;
  if (!(side > 0)) throw new Error('顔の外接正方形が 1 点に潰れている（相似変換が退化している）');
  return side;
}

/**
 * GNM の頂点を写真へ射影し、場の格子で落ちた画素を立てたマスク。
 *
 * 面ではなく**頂点が落ちた画素**でよい。GNM の三角形は辺長が写真の数画素なので頂点だけで投影域は
 * ほぼ埋まり、埋まらない隙間も chamfer 距離が数画素で吸収する。
 *
 * @param meshXy (頂点数, 2) GNM 空間の xy。**平均形状で足りる**
 */
export function projectSubjectSeed(
  similarity: Similarity2d,
  meshXy: Float64Array,
  rows: number,
  columns: number,
  imageSize: readonly [number, number],
): Uint8Array {
  const [width, height] = imageSize;
  const seed = new Uint8Array(rows * columns);
  for (let vertex = 0; vertex < meshXy.length / 2; vertex++) {
    const [x, y] = similarity.applyPoint(meshXy[vertex * 2], meshXy[vertex * 2 + 1]);
    // 場の格子は画像全体を覆うので、画素中心の規約 x = (列 + 0.5) / 列数 * 幅 の逆を取る。
    const column = Math.floor((x / width) * columns);
    const row = Math.floor((y / height) * rows);
    if (column < 0 || column >= columns || row < 0 || row >= rows) continue;
    seed[row * columns + column] = 1;
  }
  return seed;
}

/**
 * 主役の頭から届く画素を 1、届かない画素を 0 とした場（`present` と同じ形）。
 *
 * **判定（`present`）を渡すこと。** 成長を這わせる先が確信度だと「どこで切るか」という調整可能な
 * 値がもう 1 つ現れる。判定はセグメンタのクラス比較で決まりきる。
 */
export function subjectHairSelection(input: {
  present: Float32Array;
  rows: number;
  columns: number;
  similarity: Similarity2d;
  landmarkModel: LandmarkModel;
  meshXy: Float64Array;
  imageSize: readonly [number, number];
  seedDistance?: number;
  geodesicDistance?: number;
  workingFacePixels?: number;
}): Uint8Array {
  const { rows, columns } = input;
  const [width, height] = input.imageSize;
  const hair = new Uint8Array(rows * columns);
  for (let pixel = 0; pixel < hair.length; pixel++) hair[pixel] = input.present[pixel] > 0 ? 1 : 0;

  // chamfer 距離も BFS の歩数も「格子の 1 画素」を単位に測るので、格子の画素が正方でないと距離が
  // 縦横で歪む。**この要求は `PersonSegmenter.segment` の契約に書いてある**（場は写真と縦横比の
  // 等しい格子に乗る）。ここはその写しではなく機械の検査。
  const columnScale = columns / width;
  const rowScale = rows / height;
  if (Math.abs(columnScale - rowScale) > 1e-6 * Math.max(columnScale, rowScale)) {
    throw new Error(`場の格子の画素が正方でない: 列 ${columnScale} 行 ${rowScale}`);
  }

  const seed = projectSubjectSeed(input.similarity, input.meshXy, rows, columns, input.imageSize);
  const faceSquare = faceSquareSide(input.similarity, input.landmarkModel) * columnScale;
  return subjectHairReach(
    hair,
    seed,
    rows,
    columns,
    faceSquare,
    input.seedDistance ?? SUBJECT_SEED_DISTANCE,
    input.geodesicDistance ?? SUBJECT_GEODESIC_DISTANCE,
    input.workingFacePixels ?? SUBJECT_WORKING_FACE_PIXELS,
  );
}

/**
 * 種の近傍と、そこから髪を伝って届く範囲の和。
 *
 * 座標系も相似変換も知らない**この関数が本体**。写真の座標へ結び付ける仕事は
 * `subjectHairSelection` が持つので、こちらは推論もアセットも無しで検証できる。
 *
 * 種が空のときは**何も切らない**（全部 1 を返す）。主役の居場所が分からないのに切ると、主役の髪
 * ごと落としうる。
 */
export function subjectHairReach(
  hair: Uint8Array,
  seed: Uint8Array,
  rows: number,
  columns: number,
  faceSquare: number,
  seedDistance = SUBJECT_SEED_DISTANCE,
  geodesicDistance = SUBJECT_GEODESIC_DISTANCE,
  workingFacePixels = SUBJECT_WORKING_FACE_PIXELS,
): Uint8Array {
  if (hair.length !== seed.length) throw new Error('髪と種の形が揃っていない');
  if (!(faceSquare > 0)) throw new Error(`顔の外接正方形の一辺が ${faceSquare}`);
  if (seedDistance < 0) throw new Error(`種からの距離が負: ${seedDistance}`);
  if (geodesicDistance < 0) throw new Error(`測地距離が負: ${geodesicDistance}`);
  if (!(workingFacePixels > 0)) throw new Error(`作業解像度が ${workingFacePixels}`);

  let anySeed = false;
  for (const value of seed) {
    if (value !== 0) {
      anySeed = true;
      break;
    }
  }
  if (!anySeed) return new Uint8Array(hair.length).fill(1);

  const factor = Math.max(1, Math.floor(faceSquare / workingFacePixels));
  // 縮小は OR（ブロック内に 1 画素でもあれば立てる）。平均や間引きだと細い毛束が消えて測地の
  // 経路が途切れる — 途切れた先は主役の髪でも落ちる。
  const small = reduceAny(hair, rows, columns, factor);
  const smallSeed = reduceAny(seed, rows, columns, factor);
  const unit = faceSquare / factor;

  const distance = chamferDistance(smallSeed.values, small.rows, small.columns);
  const near = new Uint8Array(distance.length);
  const start = new Uint8Array(distance.length);
  for (let pixel = 0; pixel < distance.length; pixel++) {
    near[pixel] = distance[pixel] <= seedDistance * unit ? 1 : 0;
    start[pixel] = near[pixel] !== 0 && small.values[pixel] !== 0 ? 1 : 0;
  }
  const reached = geodesicWithin(
    small.values,
    start,
    small.rows,
    small.columns,
    Math.trunc(geodesicDistance * unit),
  );
  const selected = new Uint8Array(distance.length);
  for (let pixel = 0; pixel < selected.length; pixel++) {
    selected[pixel] = near[pixel] !== 0 || reached[pixel] !== 0 ? 1 : 0;
  }
  return expand(selected, small.rows, small.columns, factor, rows, columns);
}

/** `factor` × `factor` のブロックごとの OR。端は 0 で埋めて割り切る。 */
function reduceAny(
  values: Uint8Array,
  rows: number,
  columns: number,
  factor: number,
): { values: Uint8Array; rows: number; columns: number } {
  if (factor === 1) return { values, rows, columns };
  const outRows = Math.ceil(rows / factor);
  const outColumns = Math.ceil(columns / factor);
  const out = new Uint8Array(outRows * outColumns);
  for (let row = 0; row < rows; row++) {
    const outRow = Math.floor(row / factor);
    for (let column = 0; column < columns; column++) {
      if (values[row * columns + column] === 0) continue;
      out[outRow * outColumns + Math.floor(column / factor)] = 1;
    }
  }
  return { values: out, rows: outRows, columns: outColumns };
}

/** ブロックごとの結果を元の格子へ戻す（ブロック内は同じ値）。 */
function expand(
  values: Uint8Array,
  rows: number,
  columns: number,
  factor: number,
  targetRows: number,
  targetColumns: number,
): Uint8Array {
  if (factor === 1) return values;
  const out = new Uint8Array(targetRows * targetColumns);
  for (let row = 0; row < targetRows; row++) {
    const sourceRow = Math.min(Math.floor(row / factor), rows - 1);
    for (let column = 0; column < targetColumns; column++) {
      const sourceColumn = Math.min(Math.floor(column / factor), columns - 1);
      out[row * targetColumns + column] = values[sourceRow * columns + sourceColumn];
    }
  }
  return out;
}

/**
 * 種からの chamfer 距離（縦横 1・斜め √2 の 3x3、前進と後退の 2 走査）。
 *
 * 厳密な Euclid 距離変換にしないのは、要るのが「しきい値を跨ぐか」だけで、chamfer の誤差（数%）が
 * `SUBJECT_SEED_DISTANCE` の感度（0.4〜1.6）に完全に埋もれるため。
 */
export function chamferDistance(seed: Uint8Array, rows: number, columns: number): Float64Array {
  const distance = new Float64Array(rows * columns);
  for (let pixel = 0; pixel < distance.length; pixel++) {
    distance[pixel] = seed[pixel] !== 0 ? 0 : Infinity;
  }
  // 前進走査（上から下、左から右）。
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const index = row * columns + column;
      let best = distance[index];
      if (row > 0) {
        best = Math.min(best, distance[index - columns] + 1);
        if (column > 0) best = Math.min(best, distance[index - columns - 1] + DIAGONAL_STEP);
        if (column < columns - 1) best = Math.min(best, distance[index - columns + 1] + DIAGONAL_STEP);
      }
      if (column > 0) best = Math.min(best, distance[index - 1] + 1);
      distance[index] = best;
    }
  }
  // 後退走査（下から上、右から左）。
  for (let row = rows - 1; row >= 0; row--) {
    for (let column = columns - 1; column >= 0; column--) {
      const index = row * columns + column;
      let best = distance[index];
      if (row < rows - 1) {
        best = Math.min(best, distance[index + columns] + 1);
        if (column > 0) best = Math.min(best, distance[index + columns - 1] + DIAGONAL_STEP);
        if (column < columns - 1) best = Math.min(best, distance[index + columns + 1] + DIAGONAL_STEP);
      }
      if (column < columns - 1) best = Math.min(best, distance[index + 1] + 1);
      distance[index] = best;
    }
  }
  return distance;
}

/**
 * `hair` の中だけを 4 近傍で `limitSteps` 歩まで成長させた集合。
 *
 * 8 近傍にしないのは、`SUBJECT_GEODESIC_DISTANCE` を 4 近傍の歩数で測って決めたから。近傍の
 * 定義を変えるなら下限を測り直すこと。
 */
function geodesicWithin(
  hair: Uint8Array,
  start: Uint8Array,
  rows: number,
  columns: number,
  limitSteps: number,
): Uint8Array {
  const reached = new Uint8Array(hair.length);
  let anyStart = false;
  for (const value of start) {
    if (value !== 0) {
      anyStart = true;
      break;
    }
  }
  if (!anyStart) return reached;
  if (limitSteps <= 0) return Uint8Array.from(start);

  const steps = new Int32Array(hair.length);
  const queue: number[] = [];
  for (let pixel = 0; pixel < start.length; pixel++) {
    if (start[pixel] === 0) continue;
    reached[pixel] = 1;
    queue.push(pixel);
  }
  let head = 0;
  while (head < queue.length) {
    const index = queue[head++];
    const nextStep = steps[index] + 1;
    if (nextStep > limitSteps) continue;
    const row = Math.floor(index / columns);
    const column = index - row * columns;
    for (const [otherRow, otherColumn] of [
      [row - 1, column],
      [row + 1, column],
      [row, column - 1],
      [row, column + 1],
    ] as const) {
      if (otherRow < 0 || otherRow >= rows || otherColumn < 0 || otherColumn >= columns) continue;
      const other = otherRow * columns + otherColumn;
      if (reached[other] !== 0 || hair[other] === 0) continue;
      reached[other] = 1;
      steps[other] = nextStep;
      queue.push(other);
    }
  }
  return reached;
}
