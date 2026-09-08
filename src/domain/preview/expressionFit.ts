// MediaPipe の点から GNM の表情係数 383 成分を解く。
//
// **identity と同じ機構。** 書き出しのフィット（`domain/gnm/fit`）が写真の 68 点から identity 係数を
// 解くのと同じで、こちらは毎フレームの 468 点から表情係数を解く。Tikhonov 正則化つき最小二乗。
//
// 旧実装（消した `faceTracking.ts` の対応表）は MediaPipe の blendshape スコア 52 個を「顔全体のプリセット
// 20 本」へ手で割り振っていた。プリセットは顔全体が動く完成品なので、口を開けると目や頬も動く。
// デッドバンド・強さ・合計の上限はどれもその無理を抑える当て木だった。**ここは値を割り振らず、
// 点の位置から係数を解く。**
//
// ## 解く空間
//
// **すべて GNM 空間（メートル）で解く。** MediaPipe の点は正規化画像座標なので、毎フレーム相似変換で
// GNM 空間へ持ってくる。相似変換は**動かない所の点だけ**から解く（`RIGID_ANCHORS`）— 全点から解くと
// 口を開けたときに重心がずれ、表情が頭の動きへ吸われる。
//
// ## 無表情を基準にする
//
// 起こした頭は本人の顔と完全には一致しない（identity 253 成分で表せる範囲までしか寄らない）。その
// 差を表情として解くと、無表情のときから顔が歪む。**だから追い始めに「無表情での残差」を取り、
// 以降はそこからの差を解く。** 表情は「その人の素の顔からの変化」なので、これが定義とも合う。
//
// ## 何が解けて何が解けないか
//
// `tools/experiments/expression_fit_feasibility.py` がオフラインで測った値が判断の根拠:
//
// - 舌 32 成分と瞳 1 成分は**原理的に解けない**（顔の表面に出ない）。正則化で 0 に落ちる
// - 目は弱い（片目 100 成分に対し実質 6 前後）。ウィンクの再現が苦手なのはここ
// - 下顔面 150 成分は実質 20 前後
//
// つまり 383 成分のうち実質 30〜45 自由度ぶんしか情報が無い。**それを正則化に決めさせる** —
// 「何成分まで使う」という切り方を持たない（公式の並びは寄与の降順ではないので切れない）。

import { GnmHeadAsset, splitIndexOf } from '../gnm/model';
import { GnmPreviewAsset } from './asset';

/**
 * 相似変換を解くのに使う MediaPipe の点（頭の動きだけを拾う）。
 *
 * **表情で動かない所だけを選ぶ。** 全点から解くと、口を開けた瞬間に重心と尺度がずれて、表情が
 * 「頭が動いた」として吸われる（そのぶん表情の残差が消える）。
 *
 * 選んだのは 3 群:
 *
 * - **耳の前と頬骨の外側**（顎ラインの両端）: 表情筋の外
 * - **鼻梁**: 骨の上。表情でほとんど動かない
 * - **目尻の外側**: 瞼は動くが、目尻そのものの位置は動きが小さい
 *
 * index は `domain/gnm/fit.MEDIAPIPE_IBUG68` と同じ MediaPipe 顔メッシュの番号。
 */
export const RIGID_ANCHORS: readonly number[] = [
  // 顎ラインの両端（耳の前 → 頬骨）
  162, 234, 389, 454, 127, 356,
  // 鼻梁
  168, 197, 6, 4,
  // 目尻の外側
  33, 263,
  // 額の中央（眉より上）
  10, 151,
];

/** 観測に使う軸。MediaPipe の z は x, y ほど信用できないので既定では捨てる。 */
export type ObservationAxes = 'xy' | 'xyz';

/**
 * 仮定する観測ノイズ（メートル）。正則化の強さがこれで決まる。
 *
 * **大きいほど解が保守的になる**（成分が伸びず、顔が動かない）。小さいと点のふらつきが顔の
 * 震えになる。1mm は `expression_fit_feasibility.py` の掃きで「有効自由度が 30 前後・20 プリセットの
 * 復元誤差が 16%」だった所で、実機で詰める前の出発点。
 */
export const DEFAULT_OBSERVATION_NOISE_METERS = 0.001;

/** 係数の時定数（秒）。点のふらつきを均す。旧実装の表情の時定数と同じ。 */
export const COEFFICIENT_TIME_CONSTANT_SECONDS = 0.06;

/** 解く準備（guest ごとに 1 回作る）。 */
export interface ExpressionFitPlan {
  readonly componentCount: number;
  /** 観測に使う MediaPipe の点の index（密対応が付いた点）。 */
  readonly pointIndices: Int32Array;
  /** 軸の数（`xy` なら 2）。 */
  readonly axisCount: number;
  /**
   * 設計行列 (成分数, 点数 × 軸数)。**成分ごとに連続**（正規方程式を作るとき行で舐める）。
   *
   * 値は「係数を 1 動かしたとき、その点が GNM 空間で何メートル動くか」。
   */
  readonly design: Float64Array;
  /** 成分ごとの事前分布の標準偏差（プリセットの散らばりから取る）。 */
  readonly priorStd: Float64Array;
  /** `design × priorStd` の正規方程式の Cholesky 下三角 (成分数, 成分数)。 */
  readonly factor: Float64Array;
  /** 無表情の点の位置 (点数, 3) GNM 空間・メートル。 */
  readonly restPoints: Float64Array;
  /** 相似変換に使う点の、`pointIndices` の中での位置。 */
  readonly anchorSlots: Int32Array;
}

/** 1 フレームぶんの観測（MediaPipe の点）。 */
export interface LandmarkFrame {
  /** (468 以上, 3) 正規化座標。検出器の出力そのまま。 */
  readonly points: Float64Array;
}

/**
 * 解く準備を作る。**guest ごとに 1 回**（密対応と無表情の点が identity で決まる）。
 *
 * @param restVertices identity を当てた無表情の頂点（split 空間・メートル）
 */
export function buildExpressionFitPlan(
  asset: GnmHeadAsset,
  preview: GnmPreviewAsset,
  restVertices: Float64Array,
  options: {
    axes?: ObservationAxes;
    noiseMeters?: number;
  } = {},
): ExpressionFitPlan {
  const axes = options.axes ?? 'xy';
  const axisCount = axes === 'xy' ? 2 : 3;
  const noise = options.noiseMeters ?? DEFAULT_OBSERVATION_NOISE_METERS;
  if (!(noise > 0)) throw new Error(`観測ノイズが ${noise}`);

  const dense = asset.dense;
  const pointCount = dense.pointCount;
  const componentCount = preview.componentCount;

  // 密対応の角の頂点を split 空間へ写す（表情基底は split 空間で持っている）。
  const corners = new Int32Array(pointCount * 3);
  for (let index = 0; index < corners.length; index++) {
    corners[index] = splitIndexOf(asset.mesh, dense.vertexIndices[index]);
  }

  // 領域ブロックは「動く頂点だけ」を持つので、頂点 → 領域内の位置を引く表を作る。
  const design = new Float64Array(componentCount * pointCount * axisCount);
  const slotOf = new Int32Array(preview.vertexCount);
  for (const region of preview.expressionBasisRegions) {
    slotOf.fill(-1);
    for (let slot = 0; slot < region.vertexCount; slot++) {
      slotOf[preview.expressionBasisVertices[region.vertexOffset + slot]] = slot;
    }
    for (let point = 0; point < pointCount; point++) {
      for (let corner = 0; corner < 3; corner++) {
        const slot = slotOf[corners[point * 3 + corner]];
        if (slot < 0) continue;
        const weight = dense.weights[point * 3 + corner];
        if (weight === 0) continue;
        const from = region.quantizedOffset + slot * region.componentCount * 3;
        for (let local = 0; local < region.componentCount; local++) {
          const component = region.componentOffset + local;
          const factor = (weight * preview.expressionBasisScales[component]) / 32767;
          const row = component * pointCount * axisCount + point * axisCount;
          for (let axis = 0; axis < axisCount; axis++) {
            design[row + axis] += preview.expressionBasisQ[from + local * 3 + axis] * factor;
          }
        }
      }
    }
  }

  // 事前分布はプリセットの散らばりから取る（写しを持たない）。舌や瞳のように解けない成分も
  // ここでは普通の広さを持つ — 落とすのは正則化の仕事で、こちらで切らない。
  const priorStd = new Float64Array(componentCount);
  for (let component = 0; component < componentCount; component++) {
    let sum = 0;
    let squares = 0;
    for (let preset = 0; preset < preview.presetCount; preset++) {
      const value = preview.expressionPresetCoefficients[preset * componentCount + component];
      sum += value;
      squares += value * value;
    }
    const mean = sum / preview.presetCount;
    priorStd[component] = Math.sqrt(Math.max(0, squares / preview.presetCount - mean * mean));
  }
  let widest = 0;
  for (const value of priorStd) widest = Math.max(widest, value);
  if (!(widest > 0)) throw new Error('プリセットの係数が全部同じで事前分布を作れない');
  // 0 のままだと正規方程式が特異になる。下限は「一番広い成分の 1/1000」。
  for (let component = 0; component < componentCount; component++) {
    priorStd[component] = Math.max(priorStd[component], widest / 1000);
  }

  const factor = choleskyOfNormalEquations(
    design,
    priorStd,
    componentCount,
    pointCount * axisCount,
    noise,
  );

  const restPoints = new Float64Array(pointCount * 3);
  for (let point = 0; point < pointCount; point++) {
    for (let corner = 0; corner < 3; corner++) {
      const vertex = corners[point * 3 + corner];
      const weight = dense.weights[point * 3 + corner];
      for (let axis = 0; axis < 3; axis++) {
        restPoints[point * 3 + axis] += restVertices[vertex * 3 + axis] * weight;
      }
    }
  }

  const positionOf = new Map<number, number>();
  for (let point = 0; point < pointCount; point++) {
    positionOf.set(dense.mediapipeIndices[point], point);
  }
  const anchors: number[] = [];
  for (const landmark of RIGID_ANCHORS) {
    const slot = positionOf.get(landmark);
    if (slot === undefined) {
      throw new Error(`相似変換に使う点 ${landmark} に密対応が無い`);
    }
    anchors.push(slot);
  }

  return {
    componentCount,
    pointIndices: Int32Array.from(dense.mediapipeIndices),
    axisCount,
    design,
    priorStd,
    factor,
    restPoints,
    anchorSlots: Int32Array.from(anchors),
  };
}

/** 相似変換（尺度 + 回転 + 平行移動）。 */
export interface Similarity3d {
  readonly scale: number;
  /** 行優先 3x3。 */
  readonly rotation: Float64Array;
  readonly translation: Float64Array;
}

/**
 * 対応の付いた 3D 点群から相似変換を解く（Umeyama / Horn）。
 *
 * 回転は共分散行列から作った 4x4 対称行列の最大固有ベクトル（四元数）で取る。**SVD を持たずに
 * 済ませるため** — 3x3 の SVD を自前で書くと退化した配置で符号を取り違える。
 */
export function solveSimilarity3d(source: Float64Array, target: Float64Array): Similarity3d {
  const count = source.length / 3;
  if (count < 3 || target.length !== source.length) {
    throw new Error(`相似変換に渡す点が ${count} 個（3 個以上・両者同数が必要）`);
  }
  const sourceMean = [0, 0, 0];
  const targetMean = [0, 0, 0];
  for (let point = 0; point < count; point++) {
    for (let axis = 0; axis < 3; axis++) {
      sourceMean[axis] += source[point * 3 + axis];
      targetMean[axis] += target[point * 3 + axis];
    }
  }
  for (let axis = 0; axis < 3; axis++) {
    sourceMean[axis] /= count;
    targetMean[axis] /= count;
  }

  const covariance = new Float64Array(9);
  let sourceVariance = 0;
  for (let point = 0; point < count; point++) {
    const sx = source[point * 3] - sourceMean[0];
    const sy = source[point * 3 + 1] - sourceMean[1];
    const sz = source[point * 3 + 2] - sourceMean[2];
    const tx = target[point * 3] - targetMean[0];
    const ty = target[point * 3 + 1] - targetMean[1];
    const tz = target[point * 3 + 2] - targetMean[2];
    sourceVariance += sx * sx + sy * sy + sz * sz;
    covariance[0] += sx * tx;
    covariance[1] += sx * ty;
    covariance[2] += sx * tz;
    covariance[3] += sy * tx;
    covariance[4] += sy * ty;
    covariance[5] += sy * tz;
    covariance[6] += sz * tx;
    covariance[7] += sz * ty;
    covariance[8] += sz * tz;
  }
  if (!(sourceVariance > 0)) throw new Error('相似変換の元の点が 1 点に潰れている');

  const rotation = rotationFromCovariance(covariance);
  // 尺度は Umeyama: trace(R^T C) / var(source)。
  let numerator = 0;
  for (let row = 0; row < 3; row++) {
    for (let column = 0; column < 3; column++) {
      numerator += rotation[row * 3 + column] * covariance[column * 3 + row];
    }
  }
  const scale = numerator / sourceVariance;
  const translation = new Float64Array(3);
  for (let axis = 0; axis < 3; axis++) {
    let rotated = 0;
    for (let column = 0; column < 3; column++) {
      rotated += rotation[axis * 3 + column] * sourceMean[column];
    }
    translation[axis] = targetMean[axis] - scale * rotated;
  }
  return { scale, rotation, translation };
}

/** 共分散行列 (source^T target) から回転を作る（Horn の四元数法）。 */
function rotationFromCovariance(covariance: Float64Array): Float64Array {
  const [xx, xy, xz, yx, yy, yz, zx, zy, zz] = covariance;
  // 対称 4x4。最大固有値の固有ベクトルが回転の四元数（w, x, y, z）。
  // **符号を間違えると共役（= 逆回転）が返る。** 恒等回転では気付けず、頭を回した観測でだけ
  // 崩れる（実際にそう間違えて、姿勢を落とす検査が捕まえた）。
  const matrix = [
    [xx + yy + zz, yz - zy, zx - xz, xy - yx],
    [yz - zy, xx - yy - zz, xy + yx, zx + xz],
    [zx - xz, xy + yx, yy - xx - zz, yz + zy],
    [xy - yx, zx + xz, yz + zy, zz - xx - yy],
  ];
  const quaternion = largestEigenvector4(matrix);
  const [w, x, y, z] = quaternion;
  return Float64Array.from([
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ]);
}

/** 対称 4x4 の最大固有値に属する固有ベクトル（Jacobi 回転）。 */
function largestEigenvector4(input: number[][]): number[] {
  const matrix = input.map((row) => row.slice());
  const vectors = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
  // 4x4 の Jacobi は 30 掃きあれば収束する（実測で 10 掃き目に非対角が 1e-18）。
  for (let sweep = 0; sweep < 30; sweep++) {
    let offDiagonal = 0;
    for (let row = 0; row < 4; row++) {
      for (let column = row + 1; column < 4; column++) {
        offDiagonal += Math.abs(matrix[row][column]);
      }
    }
    if (offDiagonal < 1e-15) break;
    for (let row = 0; row < 4; row++) {
      for (let column = row + 1; column < 4; column++) {
        const pivot = matrix[row][column];
        if (Math.abs(pivot) < 1e-18) continue;
        const theta = (matrix[column][column] - matrix[row][row]) / (2 * pivot);
        const sign = theta >= 0 ? 1 : -1;
        const tangent = sign / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const cosine = 1 / Math.sqrt(tangent * tangent + 1);
        const sine = tangent * cosine;
        for (let index = 0; index < 4; index++) {
          const left = matrix[index][row];
          const right = matrix[index][column];
          matrix[index][row] = cosine * left - sine * right;
          matrix[index][column] = sine * left + cosine * right;
        }
        for (let index = 0; index < 4; index++) {
          const left = matrix[row][index];
          const right = matrix[column][index];
          matrix[row][index] = cosine * left - sine * right;
          matrix[column][index] = sine * left + cosine * right;
        }
        for (let index = 0; index < 4; index++) {
          const left = vectors[index][row];
          const right = vectors[index][column];
          vectors[index][row] = cosine * left - sine * right;
          vectors[index][column] = sine * left + cosine * right;
        }
      }
    }
  }
  let best = 0;
  for (let index = 1; index < 4; index++) {
    if (matrix[index][index] > matrix[best][best]) best = index;
  }
  const vector = [0, 1, 2, 3].map((index) => vectors[index][best]);
  const length = Math.hypot(...vector);
  return vector.map((value) => value / length);
}

/**
 * MediaPipe の点を GNM 空間へ写す（`out` を破壊的に埋める。長さは 点数 × 3）。
 *
 * 相似変換は**動かない所の点だけ**から解く（`RIGID_ANCHORS`）。
 */
export function alignLandmarksToGnm(
  plan: ExpressionFitPlan,
  landmarks: Float64Array,
  out: Float64Array,
): Similarity3d {
  const anchorCount = plan.anchorSlots.length;
  const source = new Float64Array(anchorCount * 3);
  const target = new Float64Array(anchorCount * 3);
  for (let index = 0; index < anchorCount; index++) {
    const slot = plan.anchorSlots[index];
    const landmark = plan.pointIndices[slot];
    for (let axis = 0; axis < 3; axis++) {
      source[index * 3 + axis] = landmarks[landmark * 3 + axis];
      target[index * 3 + axis] = plan.restPoints[slot * 3 + axis];
    }
  }
  const similarity = solveSimilarity3d(source, target);
  const { scale, rotation, translation } = similarity;
  const pointCount = plan.pointIndices.length;
  for (let point = 0; point < pointCount; point++) {
    const landmark = plan.pointIndices[point];
    const x = landmarks[landmark * 3];
    const y = landmarks[landmark * 3 + 1];
    const z = landmarks[landmark * 3 + 2];
    for (let axis = 0; axis < 3; axis++) {
      out[point * 3 + axis] =
        scale *
          (rotation[axis * 3] * x + rotation[axis * 3 + 1] * y + rotation[axis * 3 + 2] * z) +
        translation[axis];
    }
  }
  return similarity;
}

/**
 * 無表情での残差 (点数, 3) を作る。**追い始めに 1 回**。
 *
 * 起こした頭は本人の顔と完全には一致しないので、その差をここで取り分ける。取らないと無表情の
 * ときから顔が歪む。
 */
export function captureNeutralResidual(
  plan: ExpressionFitPlan,
  landmarks: Float64Array,
): Float64Array {
  const pointCount = plan.pointIndices.length;
  const aligned = new Float64Array(pointCount * 3);
  alignLandmarksToGnm(plan, landmarks, aligned);
  const residual = new Float64Array(pointCount * 3);
  for (let index = 0; index < residual.length; index++) {
    residual[index] = aligned[index] - plan.restPoints[index];
  }
  return residual;
}

/** 解く作業領域（毎フレーム作らないために持ち回る）。 */
export interface ExpressionFitScratch {
  readonly aligned: Float64Array;
  readonly observation: Float64Array;
  readonly projected: Float64Array;
}

export function createExpressionFitScratch(plan: ExpressionFitPlan): ExpressionFitScratch {
  const pointCount = plan.pointIndices.length;
  return {
    aligned: new Float64Array(pointCount * 3),
    observation: new Float64Array(pointCount * plan.axisCount),
    projected: new Float64Array(plan.componentCount),
  };
}

/**
 * 1 フレームの点から表情係数を解く（`out` を破壊的に埋める）。
 *
 * @param neutralResidual `captureNeutralResidual` の結果。`null` なら素の残差を使う
 */
export function solveExpressionCoefficients(
  plan: ExpressionFitPlan,
  landmarks: Float64Array,
  neutralResidual: Float64Array | null,
  out: Float64Array,
  scratch: ExpressionFitScratch,
): void {
  if (out.length !== plan.componentCount) {
    throw new Error(`係数が ${out.length} 個（期待 ${plan.componentCount}）`);
  }
  alignLandmarksToGnm(plan, landmarks, scratch.aligned);
  const pointCount = plan.pointIndices.length;
  for (let point = 0; point < pointCount; point++) {
    for (let axis = 0; axis < plan.axisCount; axis++) {
      const index = point * 3 + axis;
      const rest = plan.restPoints[index] + (neutralResidual === null ? 0 : neutralResidual[index]);
      scratch.observation[point * plan.axisCount + axis] = scratch.aligned[index] - rest;
    }
  }
  solveWithFactor(plan, scratch.observation, out, scratch.projected);
}

/**
 * MediaPipe の正規化座標を等方な尺度へ直す（`out` を破壊的に埋める）。
 *
 * 正規化座標は x を幅・y を高さで割ったものなので、**正方形でない映像ではそのまま使うと顔が横へ
 * 潰れる**（相似変換は尺度を 1 つしか持たないので、潰れは表情の残差として残る）。z は MediaPipe の
 * 申告どおり x と同じ尺度なので、x と一緒に直す。
 */
export function toIsotropicLandmarks(
  points: Float32Array,
  aspect: number,
  out: Float64Array,
): void {
  const scale = aspect > 0 ? aspect : 1;
  const count = Math.min(points.length, out.length) / 3;
  for (let point = 0; point < count; point++) {
    out[point * 3] = points[point * 3] * scale;
    out[point * 3 + 1] = points[point * 3 + 1];
    out[point * 3 + 2] = points[point * 3 + 2] * scale;
  }
}

/**
 * 解いた係数が観測をどれだけ説明できたか（残差の大きさ ÷ 観測の大きさ）。
 *
 * **合わないときの切り分けに使う。** 0 に近ければ「点の動きは基底で表せている」= 解き方は効いて
 * いる。1 に近ければ基底で表せない動きを見ている（検出の暴れ・相似変換の失敗・そもそも表情基底に
 * 無い動き）。画面の診断へ出す。
 */
export function observationResidualRatio(
  plan: ExpressionFitPlan,
  coefficients: Float64Array,
  scratch: ExpressionFitScratch,
): number {
  const observationCount = scratch.observation.length;
  let residual = 0;
  let size = 0;
  for (let index = 0; index < observationCount; index++) {
    let predicted = 0;
    for (let component = 0; component < plan.componentCount; component++) {
      const coefficient = coefficients[component];
      if (coefficient === 0) continue;
      predicted += plan.design[component * observationCount + index] * coefficient;
    }
    const observed = scratch.observation[index];
    residual += (observed - predicted) ** 2;
    size += observed * observed;
  }
  return size > 0 ? Math.sqrt(residual / size) : 0;
}

/** 係数を時定数で均す（`current` を破壊的に更新）。 */
export function smoothCoefficients(
  current: Float64Array,
  target: Float64Array,
  deltaSeconds: number,
  timeConstantSeconds = COEFFICIENT_TIME_CONSTANT_SECONDS,
): void {
  const blend =
    timeConstantSeconds <= 0 || deltaSeconds <= 0
      ? 1
      : 1 - Math.exp(-deltaSeconds / timeConstantSeconds);
  for (let index = 0; index < current.length; index++) {
    current[index] += (target[index] - current[index]) * blend;
  }
}

/**
 * `design × priorStd` の正規方程式 + σ²I を Cholesky 分解する。
 *
 * 事前分布で列を伸ばしてから単位行列を足すので、正則化の強さが**成分ごとの散らばりに比例**する
 * （素の λI だと、散らばりの小さい成分ほど強く縛られる）。
 */
function choleskyOfNormalEquations(
  design: Float64Array,
  priorStd: Float64Array,
  componentCount: number,
  observationCount: number,
  noise: number,
): Float64Array {
  const gram = new Float64Array(componentCount * componentCount);
  for (let row = 0; row < componentCount; row++) {
    const rowBase = row * observationCount;
    const rowScale = priorStd[row];
    for (let column = row; column < componentCount; column++) {
      const columnBase = column * observationCount;
      let total = 0;
      for (let index = 0; index < observationCount; index++) {
        total += design[rowBase + index] * design[columnBase + index];
      }
      const value = total * rowScale * priorStd[column];
      gram[row * componentCount + column] = value;
      gram[column * componentCount + row] = value;
    }
    gram[row * componentCount + row] += noise * noise;
  }

  const factor = new Float64Array(componentCount * componentCount);
  for (let row = 0; row < componentCount; row++) {
    for (let column = 0; column <= row; column++) {
      let total = gram[row * componentCount + column];
      for (let index = 0; index < column; index++) {
        total -= factor[row * componentCount + index] * factor[column * componentCount + index];
      }
      if (row === column) {
        if (!(total > 0)) {
          throw new Error(`Cholesky 分解できない（対角 ${total} が正でない）`);
        }
        factor[row * componentCount + column] = Math.sqrt(total);
      } else {
        factor[row * componentCount + column] = total / factor[column * componentCount + column];
      }
    }
  }
  return factor;
}

/** 分解済みの因子で解く（`out` を破壊的に埋める）。 */
function solveWithFactor(
  plan: ExpressionFitPlan,
  observation: Float64Array,
  out: Float64Array,
  projected: Float64Array,
): void {
  const count = plan.componentCount;
  const observationCount = observation.length;
  // A_s^T d（A_s = design × priorStd）。
  for (let row = 0; row < count; row++) {
    const base = row * observationCount;
    let total = 0;
    for (let index = 0; index < observationCount; index++) {
      total += plan.design[base + index] * observation[index];
    }
    projected[row] = total * plan.priorStd[row];
  }
  // 前進代入 L y = b。
  for (let row = 0; row < count; row++) {
    let total = projected[row];
    const base = row * count;
    for (let index = 0; index < row; index++) total -= plan.factor[base + index] * out[index];
    out[row] = total / plan.factor[base + row];
  }
  // 後退代入 L^T x = y。
  for (let row = count - 1; row >= 0; row--) {
    let total = out[row];
    for (let index = row + 1; index < count; index++) {
      total -= plan.factor[index * count + row] * out[index];
    }
    out[row] = total / plan.factor[row * count + row];
  }
  // 事前分布で伸ばした空間から戻す。
  for (let row = 0; row < count; row++) out[row] *= plan.priorStd[row];
}

/**
 * 追従の強さ（解いた係数へ掛ける倍率）。**解き方は触らない。**
 *
 * 正則化を弱めて強く出す手もあるが、そうすると「解が保守的で動かない」のと「弱く出している」のが
 * 画面から分けられなくなる。ここは**解いた後の誇張**として分けてある。
 *
 * 既定を 1 にしてあるのは、フィットが観測をそのまま写す設計だから（旧実装の対応表は MediaPipe の
 * スコアが 1.0 へ届かないぶんを 1.8 倍で埋めていた。あれは当て木で、ここには要らない）。
 */
export const DEFAULT_TRACKING_GAIN = 1;
export const MINIMUM_TRACKING_GAIN = 0.5;
export const MAXIMUM_TRACKING_GAIN = 3;
