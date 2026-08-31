// 相似変換 + identity 係数のフィット。
//
// GNM を正面（Ortho）から見たときの 68 点を、写真から検出した 68 点に重ねる。求めるのは
// 2つ:
//
//   - 相似変換（スケール・回転・平行移動）: GNM 空間 ↔ 画像 UV 空間の対応。アトラスの
//     ベイクと髪シェルの座標変換がこれに乗る
//   - identity: 顔の個人差を表す係数。guest.json に入る唯一の数値
//
// 入口は **MediaPipe の 468 点**（画像ピクセル座標）と GNM モデルだけ。468 点のうちどれが
// GNM の 68 点に対応するかは GNM 固有の知識なので、`MEDIAPIPE_IBUG68` による 468 → 68 の
// 絞り込みはここで行う（検出器のアダプタは点を選ばない）。
//
// 解く空間
// --------
// **すべて GNM 空間（メートル）で解く。** 写真の 68 点は相似変換の逆で GNM 空間へ持って
// きてから残差にする。写真空間で解くと identity 基底を毎周スケール・回転させる必要があり、
// さらに正則化の実効強度が「写真に顔がどれだけ大きく写っているか」で変わってしまう。
//
// そのうえで残差と基底を平均顔の顔幅（`LandmarkModel.faceWidth`、定数）で割って無次元化
// する。GNM 空間はメートル単位（顔幅 約 0.15 m）なので、割らずに λ を置くと λ の値が
// 「メートルで測った長さ²」と釣り合う量になり、係数の事前分布としての意味が読めなくなる。
//
// 交互最適化
// ----------
// 相似変換と identity は互いに影響するので交互に解く。相似変換は閉形式（反復なし）、
// identity は Tikhonov 正則化つき最小二乗（Cholesky）。正則化を強→中→弱にしながら 3 周
// する。強い正則化から始めるのは、初回の相似変換が平均顔に対して当てられていてまだ粗い
// ため、そこで係数を伸ばしきると誤った形に固まるから。

import {
  DenseLandmarks,
  GnmHeadAsset,
  IBUG68_POINT_COUNT,
  SparseLandmarks68,
  denseConfidence,
  denseMedianResidualRatio,
  sampleBarycentricBasis,
  sampleBarycentricPositions,
} from './model';

/**
 * 入力として受け取る MediaPipe の点数（顔メッシュ 468 + 虹彩 10）。
 *
 * **検出器の出力をそのまま受ける。** 虹彩 10 点は形状フィットに使わないが、落とすのは
 * 呼び出し側の仕事ではない（同じ 478 点を眼球テクスチャの焼き込みも読む）。
 */
export const MEDIAPIPE_LANDMARK_COUNT = 478;

/** 顔メッシュの点数。`MEDIAPIPE_IBUG68` が指すのはこの範囲だけ（虹彩 10 点は後ろ）。 */
export const MEDIAPIPE_FACE_MESH_COUNT = 468;

/**
 * MediaPipe FaceLandmarker の点 → iBUG-68 の並び（顎17 / 眉10 / 鼻9 / 目12 / 口20）。
 *
 * この定数がフィット側にあるのは、68 点の意味を決めているのがフィットだから。検出器の
 * アダプタ（infrastructure）は 468 点をそのまま渡し、点を選ばない。
 *
 * **顎ラインの index 2〜6 が `149, 136, 172, 58, 93` で、標準的な iBUG 順
 * （`93, 58, 172, 136, 149`）の逆順になっている。これは誤りではない。** GNM の
 * `head_sparse_68` が向かって左の顎を空間的に逆順（顎寄り → 耳寄り）で定義しているため、
 * MediaPipe 側もその並びに合わせている。標準順に「直す」と左顎の対応が交差し、フィットが
 * 静かに歪む（例外にはならない — 相似変換が誤差を吸収してしまう）。
 * `assertLandmarkChainOrientation` がこの交差を検出するための番人。
 */
export const MEDIAPIPE_IBUG68: readonly number[] = [
  // 顎ライン 0-16
  162, 234, 149, 136, 172, 58, 93, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
  // 眉 17-21（左）/ 22-26（右）
  70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
  // 鼻梁 27-30 + 鼻底 31-35
  168, 197, 5, 4, 75, 97, 2, 326, 305,
  // 目 36-41（左）/ 42-47（右）
  33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
  // 口 外周 48-59 + 内周 60-67
  61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317, 14, 87,
];

function range(start: number, stop: number): number[] {
  const out: number[] = [];
  for (let value = start; value < stop; value++) out.push(value);
  return out;
}

/**
 * iBUG-68 の連続チェーン（名前, 点の並び, 閉じているか）。
 *
 * 「隣り合う点は空間的にも隣り合う」という並びの正本。対応の交差検出に使う。
 */
export const IBUG68_CHAINS: readonly [string, readonly number[], boolean][] = [
  ['顎', range(0, 17), false],
  ['眉左', range(17, 22), false],
  ['眉右', range(22, 27), false],
  ['鼻梁', range(27, 31), false],
  ['鼻底', range(31, 36), false],
  ['目左', range(36, 42), true],
  ['目右', range(42, 48), true],
  ['唇外周', range(48, 60), true],
  ['唇内周', range(60, 68), true],
];

/** チェーンの辺のうち、方向が定まらないとして交差判定から除く長さの閾値（中央値比）。 */
export const DEGENERATE_EDGE_RATIO = 0.25;

/**
 * 交差と見なす cos の上限（これより小さい辺を反転とみなす）。
 *
 * 内積の符号（cos < 0）で切ると誤検出する。−0.5 は「120°以上ずれている」に相当し、
 * ノイズ由来のふらつきと系統的な反転を分ける。
 */
export const REVERSAL_COSINE = -0.5;

/**
 * 同一チェーン内で何本の辺が反転していたら「対応が交差している」と判定するか。
 *
 * 1 本で判定すると誤検出する。並びの反転は必ず複数の辺を巻き込むが、検出ノイズによる
 * 単発の反転は 1 本で終わる。
 */
export const MIN_REVERSED_EDGES = 2;

/**
 * 各周で仮定する「写真の 68 点と GNM の 68 点のずれ」。平均顔の顔幅に対する比。
 *
 * 長さが交互最適化の周回数。強 → 弱に緩めるのは、初回の相似変換が平均顔に対して当てて
 * いてまだ粗く、そこで係数を伸ばしきると誤った形に固まるため。最終周の 1% は σ=0〜1% の
 * 全域で最悪値が最小になる選び方（ミニマックス）。
 */
export const ASSUMED_LANDMARK_DISAGREEMENT: readonly number[] = [0.06, 0.02, 0.01];

/**
 * λ の目盛りにする観測数（68 点 × x,y = 136）。**点数に依らない定数**。
 *
 * `fitHead` は正規方程式を実際の観測数で割ってから λ を足す。だから λ の分母にも実際の
 * 観測数を入れると割り算が二重にかかり、λ の実効強度が点数で変わる。136 に固定するのは
 * 「68 点フィットのとき何倍か」という目盛りを点数から切り離すため。
 */
export const REFERENCE_OBSERVATION_COUNT = 2 * IBUG68_POINT_COUNT;

/**
 * 仮定するずれ（顔幅比）から無次元の λ を出す。
 *
 * 係数の事前分布を `c ~ N(0, I)`、ずれを座標ごとに独立な分散 `σ²` と置いた MAP 推定
 * そのもの。**分母は実際の観測数ではなく 136 固定** — 密対応 468 点は 68 点と同じ範囲を
 * 細かく見るだけで、頭幅・後頭部を 1 点も拘束しないため、観測が増えたぶん事前分布を
 * 緩めてよいという前提が成り立たない。
 */
export function regularizationFor(disagreement: number): number {
  return (disagreement * disagreement) / REFERENCE_OBSERVATION_COUNT;
}

/** 交互最適化の各周で、最終周のずれの何倍を仮定するか（強 → 弱）= (6, 2, 1)。 */
export const DISAGREEMENT_SWEEP_RATIOS: readonly number[] = ASSUMED_LANDMARK_DISAGREEMENT.map(
  (disagreement) => disagreement / ASSUMED_LANDMARK_DISAGREEMENT[ASSUMED_LANDMARK_DISAGREEMENT.length - 1],
);

/** 68 点モデルが仮定する最終周のずれ（顔幅比）。ミニマックスで選んだ 1%。 */
export const IBUG68_ASSUMED_DISAGREEMENT =
  ASSUMED_LANDMARK_DISAGREEMENT[ASSUMED_LANDMARK_DISAGREEMENT.length - 1];

/** 信頼度が 1/2 になる残差を、メッシュの辺の長さの何倍で表すか。 */
export const CONFIDENCE_HALF_EDGES = 0.5;

/** 68 点の対応が交差している（並びが局所的に反転している）。 */
export class LandmarkCorrespondenceError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'LandmarkCorrespondenceError';
  }
}

/**
 * 2D 相似変換 `p ↦ linear @ p + translation`。
 *
 * `linear` は `scale × 回転` または `scale × 鏡映`。鏡映を許すのは、画像座標が行方向
 * （下）を y の正にとるため、正面写真では GNM 空間（y は上）との対応が必ず鏡映を含むから。
 * どちらの側かは最小二乗が決めるので、`isMirrored` で結果を確認できる。
 */
export class Similarity2d {
  /** 行優先の 2x2（`[a, b, c, d]` が `[[a, b], [c, d]]`）。 */
  readonly linear: Float64Array;
  readonly translation: Float64Array;

  constructor(linear: Float64Array, translation: Float64Array) {
    if (linear.length !== 4 || translation.length !== 2) {
      throw new Error('相似変換の形が (2,2) / (2,) ではない');
    }
    this.linear = linear;
    this.translation = translation;
  }

  private determinant(): number {
    return this.linear[0] * this.linear[3] - this.linear[1] * this.linear[2];
  }

  /** 等方スケール。鏡映込みでも正の値。 */
  get scale(): number {
    return Math.sqrt(Math.abs(this.determinant()));
  }

  /** 向きを反転しているか（画像座標が y 下向きなら正面写真では true）。 */
  get isMirrored(): boolean {
    return this.determinant() < 0;
  }

  /** 1 点を変換する。 */
  applyPoint(x: number, y: number): [number, number] {
    const m = this.linear;
    return [m[0] * x + m[1] * y + this.translation[0], m[2] * x + m[3] * y + this.translation[1]];
  }

  /** (N, 2) を変換する。 */
  apply(points: Float64Array): Float64Array {
    const out = new Float64Array(points.length);
    const m = this.linear;
    for (let index = 0; index < points.length; index += 2) {
      const x = points[index];
      const y = points[index + 1];
      out[index] = m[0] * x + m[1] * y + this.translation[0];
      out[index + 1] = m[2] * x + m[3] * y + this.translation[1];
    }
    return out;
  }

  /** 1 点を逆変換する。 */
  inverseApplyPoint(x: number, y: number): [number, number] {
    const m = this.linear;
    const determinant = this.determinant();
    if (determinant === 0) throw new Error('相似変換が退化していて逆写像が無い');
    const dx = x - this.translation[0];
    const dy = y - this.translation[1];
    return [(m[3] * dx - m[1] * dy) / determinant, (-m[2] * dx + m[0] * dy) / determinant];
  }

  /** (N, 2) を逆変換する。 */
  inverseApply(points: Float64Array): Float64Array {
    const out = new Float64Array(points.length);
    const m = this.linear;
    const determinant = this.determinant();
    if (determinant === 0) throw new Error('相似変換が退化していて逆写像が無い');
    for (let index = 0; index < points.length; index += 2) {
      const dx = points[index] - this.translation[0];
      const dy = points[index + 1] - this.translation[1];
      out[index] = (m[3] * dx - m[1] * dy) / determinant;
      out[index + 1] = (-m[2] * dx + m[0] * dy) / determinant;
    }
    return out;
  }

  /** 回転部分だけ（スケールを割った直交行列）。法線と極座標の向きに使う。 */
  get rotation(): Float64Array {
    const scale = this.scale;
    if (!(scale > 0)) throw new Error('相似変換のスケールが 0');
    return new Float64Array([
      this.linear[0] / scale,
      this.linear[1] / scale,
      this.linear[2] / scale,
      this.linear[3] / scale,
    ]);
  }
}

/**
 * 対応点を identity で動かすための線形モデル（GNM 空間 / メートル）。
 *
 * GNM は「平均顔 + 成分ごとのツマミ」で頭部を作るので、表面上のどの点も同じ形に落ちる:
 * `点 = meanPositions + Σ cᵢ · identityBasis[i]`。写真ごとに変わらないので事前計算して
 * 持ち回る。
 *
 * **点の数は 68 に限らない。** GNM アセットの 68 点定義でも MediaPipe 顔メッシュの密対応
 * でも同じ形になる。密対応の方が観測が多く、identity の成分数を超える。
 */
export interface LandmarkModel {
  /** (P, 3) 平均顔の対応点。 */
  readonly meanPositions: Float64Array;
  /** (K, P, 3) 各成分が対応点をどう動かすか。 */
  readonly identityBasis: Float64Array;
  /** (P,) 各点に対応する MediaPipe 顔メッシュの点 index。 */
  readonly photoIndices: Int32Array;
  /** (P,) フィットの重み。密対応では投影の信頼度。 */
  readonly weights: Float64Array;
  /** (68,) iBUG 68 の並びに対応する行。**交差検出と顔幅の基準**。 */
  readonly guardRows: Int32Array;
  /** 写真の点とモデルの点のずれ（顔幅比）。**対応の付け方で変わる**。 */
  readonly assumedDisagreement: number;
  readonly pointCount: number;
  readonly identityComponentCount: number;
  /** 平均顔の **68 点** の x 範囲。無次元化の基準長（写真によらない定数）。 */
  readonly faceWidth: number;
}

function buildModel(
  meanPositions: Float64Array,
  identityBasis: Float64Array,
  photoIndices: Int32Array,
  weights: Float64Array,
  guardRows: Int32Array,
  assumedDisagreement: number,
): LandmarkModel {
  const pointCount = meanPositions.length / 3;
  const identityComponentCount = identityBasis.length / (pointCount * 3);
  if (photoIndices.length !== pointCount) throw new Error('photoIndices の長さが点数と合わない');
  if (weights.length !== pointCount) throw new Error('weights の長さが点数と合わない');
  if (guardRows.length !== IBUG68_POINT_COUNT) {
    throw new Error(`guardRows の長さが ${guardRows.length}（期待 ${IBUG68_POINT_COUNT}）`);
  }
  for (const weight of weights) {
    if (!(weight > 0)) throw new Error('重みが 0 以下の点がある（観測にならない）');
  }
  if (!(assumedDisagreement > 0)) {
    throw new Error(`仮定するずれが ${assumedDisagreement}（事前分布が消える）`);
  }
  let low = Infinity;
  let high = -Infinity;
  for (let slot = 0; slot < guardRows.length; slot++) {
    const x = meanPositions[guardRows[slot] * 3];
    if (x < low) low = x;
    if (x > high) high = x;
  }
  const faceWidth = high - low;
  if (!(faceWidth > 0)) throw new Error('平均顔の 68 点の x 範囲が 0（無次元化できない）');
  return {
    meanPositions,
    identityBasis,
    photoIndices,
    weights,
    guardRows,
    assumedDisagreement,
    pointCount,
    identityComponentCount,
    faceWidth,
  };
}

/** identity 係数から対応点 (P, 3) を作る。 */
export function evaluateModel(model: LandmarkModel, identity: Float64Array): Float64Array {
  if (identity.length !== model.identityComponentCount) {
    throw new Error(`identity の長さが ${identity.length}`);
  }
  const out = Float64Array.from(model.meanPositions);
  const stride = model.pointCount * 3;
  for (let component = 0; component < model.identityComponentCount; component++) {
    const coefficient = identity[component];
    if (coefficient === 0) continue;
    const base = component * stride;
    for (let index = 0; index < stride; index++) {
      out[index] += model.identityBasis[base + index] * coefficient;
    }
  }
  return out;
}

/** identity 係数から iBUG 68 点だけ (68, 3) を作る（交差検出と顔幅の用）。 */
export function evaluateIbug68(model: LandmarkModel, identity: Float64Array): Float64Array {
  const all = evaluateModel(model, identity);
  const out = new Float64Array(IBUG68_POINT_COUNT * 3);
  for (let slot = 0; slot < IBUG68_POINT_COUNT; slot++) {
    const row = model.guardRows[slot];
    out[slot * 3] = all[row * 3];
    out[slot * 3 + 1] = all[row * 3 + 1];
    out[slot * 3 + 2] = all[row * 3 + 2];
  }
  return out;
}

/**
 * 仮定するずれを `disagreementScale` 倍したときの λ（強 → 弱）。
 *
 * **倍率で受けるのは、ずれの既定が model ごとに違うため。** λ はずれの 2 乗に比例するので、
 * 倍率 `s` は λ を `s²` 倍する。
 */
export function regularizationScheduleFor(
  model: LandmarkModel,
  disagreementScale: number,
): number[] {
  if (!(disagreementScale > 0)) throw new Error(`倍率が ${disagreementScale}`);
  return DISAGREEMENT_SWEEP_RATIOS.map((ratio) =>
    regularizationFor(model.assumedDisagreement * disagreementScale * ratio),
  );
}

/** フィットの結果。 */
export interface HeadFit {
  /** (成分数,) 唯一の成果物。上限は置かない（公式 GNM も置いていない）。 */
  readonly identity: Float64Array;
  /** GNM 空間の xy → 写真ピクセル。出力はしない（検査とベイクに使う）。 */
  readonly similarity: Similarity2d;
  /** 各周の残差 RMS（写真ピクセル）。周を追って減る。 */
  readonly residualRmsPixels: readonly number[];
}

/**
 * アセットの頂点データを 68 点へ落として LandmarkModel にする。
 *
 * **68 点だけでは観測 136 個 < identity の成分数**なので係数が一意に決まらない。密対応が
 * 使えるなら `buildDenseLandmarkModel` を使う。
 */
export function buildLandmarkModel(
  asset: GnmHeadAsset,
  landmarks: SparseLandmarks68,
): LandmarkModel {
  const mesh = asset.mesh;
  const weights = new Float64Array(IBUG68_POINT_COUNT).fill(1);
  const guardRows = new Int32Array(IBUG68_POINT_COUNT);
  for (let slot = 0; slot < IBUG68_POINT_COUNT; slot++) guardRows[slot] = slot;
  return buildModel(
    sampleBarycentricPositions(
      mesh,
      mesh.templateVertexPositions,
      landmarks.vertexIndices,
      landmarks.weights,
    ),
    sampleBarycentricBasis(
      mesh,
      asset.vertexIdentityBasis,
      landmarks.vertexIndices,
      landmarks.weights,
    ),
    Int32Array.from(MEDIAPIPE_IBUG68),
    weights,
    guardRows,
    IBUG68_ASSUMED_DISAGREEMENT,
  );
}

/**
 * 密対応（MediaPipe 顔メッシュ）から LandmarkModel を作る。
 *
 * 重みは投影の信頼度。交差検出と顔幅の基準に使う iBUG 68 の行は `mediapipeIndices` から
 * 引き当てる（**68 点が 1 つでも欠けていたら落とす** — 交差検出が回らなくなるうえ、顔幅の
 * 基準が変わってアトラスの一辺が黙って動く）。
 */
export function buildDenseLandmarkModel(
  asset: GnmHeadAsset,
  dense: DenseLandmarks,
): LandmarkModel {
  const mesh = asset.mesh;
  const order = new Map<number, number>();
  for (let row = 0; row < dense.pointCount; row++) order.set(dense.mediapipeIndices[row], row);
  const missing = MEDIAPIPE_IBUG68.filter((index) => !order.has(index));
  if (missing.length > 0) {
    throw new Error(
      `密対応に iBUG 68 の点が ${missing.length} 個足りない（MediaPipe index ${missing}）。` +
        ' 整列が壊れているか投影残差の上限が厳しすぎる',
    );
  }
  const meanPositions = sampleBarycentricPositions(
    mesh,
    mesh.templateVertexPositions,
    dense.vertexIndices,
    dense.weights,
  );
  const guardRows = Int32Array.from(MEDIAPIPE_IBUG68.map((index) => order.get(index) as number));

  let low = Infinity;
  let high = -Infinity;
  for (let slot = 0; slot < guardRows.length; slot++) {
    const x = meanPositions[guardRows[slot] * 3];
    if (x < low) low = x;
    if (x > high) high = x;
  }
  // 仮定するずれは**実測**（投影残差の中央値を顔幅で割る）。68 点のように意味の対応では
  // ないので、ミニマックスで選ぶ必要が無い。
  return buildModel(
    meanPositions,
    sampleBarycentricBasis(mesh, asset.vertexIdentityBasis, dense.vertexIndices, dense.weights),
    Int32Array.from(dense.mediapipeIndices),
    denseConfidence(dense, CONFIDENCE_HALF_EDGES),
    guardRows,
    denseMedianResidualRatio(dense, high - low),
  );
}

/**
 * MediaPipe の 468 点から iBUG-68 の 68 点を抜き出す。
 *
 * @param photoLandmarks (478, 2) 画像ピクセル座標。検出器の出力そのまま
 */
export function selectIbug68(photoLandmarks: Float64Array): Float64Array {
  assertLandmarkShape(photoLandmarks);
  const out = new Float64Array(IBUG68_POINT_COUNT * 2);
  for (let slot = 0; slot < IBUG68_POINT_COUNT; slot++) {
    const source = MEDIAPIPE_IBUG68[slot];
    out[slot * 2] = photoLandmarks[source * 2];
    out[slot * 2 + 1] = photoLandmarks[source * 2 + 1];
  }
  return out;
}

/**
 * 検出器の出力から、モデルの対応点に当たる写真上の点を抜き出す。
 *
 * **虹彩の 10 点は対象外** — 顔メッシュの点ではないので barycentric 対応が無い。
 */
export function selectModelPoints(
  photoLandmarks: Float64Array,
  model: LandmarkModel,
): Float64Array {
  assertLandmarkShape(photoLandmarks);
  const out = new Float64Array(model.pointCount * 2);
  for (let point = 0; point < model.pointCount; point++) {
    const source = model.photoIndices[point];
    if (source < 0 || source >= MEDIAPIPE_FACE_MESH_COUNT) {
      throw new Error(
        `photoIndices が顔メッシュの範囲 [0, ${MEDIAPIPE_FACE_MESH_COUNT}) の外にある`,
      );
    }
    out[point * 2] = photoLandmarks[source * 2];
    out[point * 2 + 1] = photoLandmarks[source * 2 + 1];
  }
  return out;
}

function assertLandmarkShape(photoLandmarks: Float64Array): void {
  if (photoLandmarks.length !== MEDIAPIPE_LANDMARK_COUNT * 2) {
    throw new Error(
      `MediaPipe の ${MEDIAPIPE_LANDMARK_COUNT} 点 (N, 2) を渡すこと:` +
        ` ${photoLandmarks.length / 2}`,
    );
  }
}

/**
 * 平均顔（identity = 0）を写真のランドマークへ重ねる相似変換。
 *
 * **`fitHead` の第1周が解くものと同一。** identity を求めずに GNM 空間 ↔ 写真の対応だけが
 * 欲しい呼び出し側のためにある（推論の切り出しを決めるとき）。**当然フィット後の相似変換
 * より粗い**ので、使う側は余白を持つこと。
 */
export function coarseSimilarity(
  photoLandmarks: Float64Array,
  landmarkModel: LandmarkModel,
): Similarity2d {
  const photo = selectModelPoints(photoLandmarks, landmarkModel);
  return solveSimilarity2d(xyOf(landmarkModel.meanPositions), photo);
}

/** (P, 3) から xy だけ抜いた (P, 2)。 */
export function xyOf(points3: Float64Array): Float64Array {
  const count = points3.length / 3;
  const out = new Float64Array(count * 2);
  for (let point = 0; point < count; point++) {
    out[point * 2] = points3[point * 3];
    out[point * 2 + 1] = points3[point * 3 + 1];
  }
  return out;
}

/**
 * `source → target` の 2D 相似変換を最小二乗で閉形式に解く（反復なし）。
 *
 * 重心を合わせてから複素数で解く。点を `z = x + iy` と書くと、回転を含む相似変換は
 * `w = a·z`、鏡映を含む相似変換は `w = b·conj(z)` の1つの複素係数で表せる。どちらも
 * 最小二乗解は内積の商で出るので、残差の小さい側を採る。
 */
export function solveSimilarity2d(sourceXy: Float64Array, targetXy: Float64Array): Similarity2d {
  if (sourceXy.length !== targetXy.length || sourceXy.length % 2 !== 0) {
    throw new Error(`相似変換の入力の形が合わない: ${sourceXy.length} / ${targetXy.length}`);
  }
  const count = sourceXy.length / 2;
  let sourceCentroidX = 0;
  let sourceCentroidY = 0;
  let targetCentroidX = 0;
  let targetCentroidY = 0;
  for (let point = 0; point < count; point++) {
    sourceCentroidX += sourceXy[point * 2];
    sourceCentroidY += sourceXy[point * 2 + 1];
    targetCentroidX += targetXy[point * 2];
    targetCentroidY += targetXy[point * 2 + 1];
  }
  sourceCentroidX /= count;
  sourceCentroidY /= count;
  targetCentroidX /= count;
  targetCentroidY /= count;

  let energy = 0;
  // Σ w·conj(z) と Σ w·z（それぞれ回転側・鏡映側の分子）。
  let directReal = 0;
  let directImaginary = 0;
  let mirroredReal = 0;
  let mirroredImaginary = 0;
  for (let point = 0; point < count; point++) {
    const zx = sourceXy[point * 2] - sourceCentroidX;
    const zy = sourceXy[point * 2 + 1] - sourceCentroidY;
    const wx = targetXy[point * 2] - targetCentroidX;
    const wy = targetXy[point * 2 + 1] - targetCentroidY;
    energy += zx * zx + zy * zy;
    directReal += wx * zx + wy * zy;
    directImaginary += wy * zx - wx * zy;
    mirroredReal += wx * zx - wy * zy;
    mirroredImaginary += wx * zy + wy * zx;
  }
  if (energy <= 0) throw new Error('相似変換を解けない: source の点がすべて同じ位置にある');

  const direct = [directReal / energy, directImaginary / energy];
  const mirrored = [mirroredReal / energy, mirroredImaginary / energy];
  // 残差は Σ|w|² − |係数の分子|²/energy なので、分子の絶対値が大きい側が残差が小さい。
  const useMirrored = Math.hypot(mirrored[0], mirrored[1]) > Math.hypot(direct[0], direct[1]);
  const [real, imaginary] = useMirrored ? mirrored : direct;
  const linear = useMirrored
    ? new Float64Array([real, imaginary, imaginary, -real])
    : new Float64Array([real, -imaginary, imaginary, real]);
  return new Similarity2d(
    linear,
    new Float64Array([
      targetCentroidX - (linear[0] * sourceCentroidX + linear[1] * sourceCentroidY),
      targetCentroidY - (linear[2] * sourceCentroidX + linear[3] * sourceCentroidY),
    ]),
  );
}

/**
 * 68 点の対応が交差していないことを確かめる（同じ空間の 2 点集合を渡す）。
 *
 * iBUG-68 の各チェーンに沿って隣接点の差分ベクトルを取り、モデル側と写真側で向きが揃って
 * いることを見る。並びがどこかで反転していれば、その区間の辺が揃って逆を向く。
 *
 * これが無いと、対応表（`MEDIAPIPE_IBUG68`）や GNM の barycentric 定義を触ったときに
 * **例外もエラーも出ないまま、相似変換が誤差を吸収してフィットが歪むだけ**になる。
 *
 * 2 つの点集合は同じ空間で渡すこと。**2 次元でも 3 次元でも受ける**（判定は差分ベクトルの
 * 向きなので次元に依らない）。
 */
export function assertLandmarkChainOrientation(
  modelPoints: Float64Array,
  photoPoints: Float64Array,
  dimensions: number,
): void {
  if (
    modelPoints.length !== IBUG68_POINT_COUNT * dimensions ||
    photoPoints.length !== IBUG68_POINT_COUNT * dimensions
  ) {
    throw new Error(
      `68 点 × ${dimensions} 次元を渡すこと: ${modelPoints.length} / ${photoPoints.length}`,
    );
  }
  const violations: string[] = [];
  for (const [name, chain, isRing] of IBUG68_CHAINS) {
    const heads = isRing ? chain : chain.slice(0, -1);
    const tails = isRing ? chain.map((_, slot) => chain[(slot + 1) % chain.length]) : chain.slice(1);

    const modelEdges: number[][] = [];
    const photoEdges: number[][] = [];
    const modelLengths: number[] = [];
    const photoLengths: number[] = [];
    for (let edge = 0; edge < heads.length; edge++) {
      const head = heads[edge];
      const tail = tails[edge];
      const modelEdge: number[] = [];
      const photoEdge: number[] = [];
      for (let axis = 0; axis < dimensions; axis++) {
        modelEdge.push(modelPoints[tail * dimensions + axis] - modelPoints[head * dimensions + axis]);
        photoEdge.push(photoPoints[tail * dimensions + axis] - photoPoints[head * dimensions + axis]);
      }
      modelEdges.push(modelEdge);
      photoEdges.push(photoEdge);
      modelLengths.push(Math.hypot(...modelEdge));
      photoLengths.push(Math.hypot(...photoEdge));
    }
    const modelMedian = median(modelLengths);
    const photoMedian = median(photoLengths);

    const reversed: number[] = [];
    const cosines: number[] = [];
    for (let edge = 0; edge < heads.length; edge++) {
      // 短い辺は向きが定まらないので判定から外す（中央値基準なので写真の大きさに依らない）。
      const usable =
        modelLengths[edge] > DEGENERATE_EDGE_RATIO * modelMedian &&
        photoLengths[edge] > DEGENERATE_EDGE_RATIO * photoMedian;
      let dot = 0;
      for (let axis = 0; axis < dimensions; axis++) dot += modelEdges[edge][axis] * photoEdges[edge][axis];
      const cosine = usable ? dot / (modelLengths[edge] * photoLengths[edge]) : 1;
      cosines.push(cosine);
      if (usable && cosine < REVERSAL_COSINE) reversed.push(edge);
    }
    if (reversed.length < MIN_REVERSED_EDGES) continue;
    for (const edge of reversed) {
      violations.push(
        `${name}: 点 ${heads[edge]} → ${tails[edge]} の差分が逆を向いている` +
          `（cos ${cosines[edge].toFixed(3)}）`,
      );
    }
  }
  if (violations.length > 0) {
    throw new LandmarkCorrespondenceError(
      '68 点の対応が交差している。対応表かランドマーク定義の並びが合っていない:\n' +
        violations.map((line) => `  - ${line}`).join('\n'),
    );
  }
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/**
 * 写真のランドマークに GNM を重ねて identity 係数を求める。
 *
 * @param photoLandmarks (478, 2) MediaPipe の出力そのまま。画像ピクセル座標
 * @param identityClip identity 係数の絶対値の上限。null で上限なし（既定）。事前分布を
 *   緩めた端で何が壊れるかを見るための安全弁で、公式 GNM は持たない
 * @throws LandmarkCorrespondenceError 対応が交差している
 */
export function fitHead(
  photoLandmarks: Float64Array,
  landmarkModel: LandmarkModel,
  options: { regularizationSchedule?: readonly number[]; identityClip?: number | null } = {},
): HeadFit {
  const photo = selectModelPoints(photoLandmarks, landmarkModel);
  const schedule =
    options.regularizationSchedule ?? regularizationScheduleFor(landmarkModel, 1);
  if (schedule.length === 0) throw new Error('λ のスケジュールが空');
  const identityClip = options.identityClip ?? null;

  const componentCount = landmarkModel.identityComponentCount;
  const pointCount = landmarkModel.pointCount;
  const faceWidth = landmarkModel.faceWidth;
  const observationCount = pointCount * 2;

  // 無次元化した設計行列（観測数 × 成分数）。identity 基底は写真に依らないので周回の外で
  // 1 回だけ作る。並びは残差を平坦化した (点0.x, 点0.y, 点1.x, ...) と揃える。
  //
  // 重みは **√w を両辺に掛ける**形で入れる（重み付き最小二乗の標準形）。密対応では投影の
  // 信頼度が点ごとに違い、対応が甘い点を同じ重さで扱うと顔を引っ張る。
  const rootWeights = new Float64Array(pointCount);
  for (let point = 0; point < pointCount; point++) {
    rootWeights[point] = Math.sqrt(landmarkModel.weights[point]);
  }
  const design = new Float64Array(observationCount * componentCount);
  for (let component = 0; component < componentCount; component++) {
    const base = component * pointCount * 3;
    for (let point = 0; point < pointCount; point++) {
      const weight = rootWeights[point] / faceWidth;
      design[(point * 2) * componentCount + component] =
        landmarkModel.identityBasis[base + point * 3] * weight;
      design[(point * 2 + 1) * componentCount + component] =
        landmarkModel.identityBasis[base + point * 3 + 1] * weight;
    }
  }
  const gram = new Float64Array(componentCount * componentCount);
  for (let row = 0; row < componentCount; row++) {
    for (let column = row; column < componentCount; column++) {
      let total = 0;
      for (let observation = 0; observation < observationCount; observation++) {
        total +=
          design[observation * componentCount + row] * design[observation * componentCount + column];
      }
      const value = total / observationCount;
      gram[row * componentCount + column] = value;
      gram[column * componentCount + row] = value;
    }
  }

  let identity = new Float64Array(componentCount);
  let similarity: Similarity2d | null = null;
  const residualRms: number[] = [];
  const residual = new Float64Array(observationCount);

  for (let sweep = 0; sweep < schedule.length; sweep++) {
    const gnmLandmarks = evaluateModel(landmarkModel, identity);
    similarity = solveSimilarity2d(xyOf(gnmLandmarks), photo);
    const target = similarity.inverseApply(photo);

    // 対応の交差は identity に依らないので初回だけ見る。ここで見るのは、写真の点を
    // GNM 空間へ持ってきた後（＝両者が同じ空間にある）でないと内積の符号が座標系の向きの
    // 違いを拾ってしまうため。
    if (sweep === 0) {
      const rows = landmarkModel.guardRows;
      const modelGuard = new Float64Array(IBUG68_POINT_COUNT * 2);
      const targetGuard = new Float64Array(IBUG68_POINT_COUNT * 2);
      for (let slot = 0; slot < IBUG68_POINT_COUNT; slot++) {
        modelGuard[slot * 2] = gnmLandmarks[rows[slot] * 3];
        modelGuard[slot * 2 + 1] = gnmLandmarks[rows[slot] * 3 + 1];
        targetGuard[slot * 2] = target[rows[slot] * 2];
        targetGuard[slot * 2 + 1] = target[rows[slot] * 2 + 1];
      }
      assertLandmarkChainOrientation(modelGuard, targetGuard, 2);
    }

    for (let point = 0; point < pointCount; point++) {
      const weight = rootWeights[point] / faceWidth;
      residual[point * 2] =
        (target[point * 2] - landmarkModel.meanPositions[point * 3]) * weight;
      residual[point * 2 + 1] =
        (target[point * 2 + 1] - landmarkModel.meanPositions[point * 3 + 1]) * weight;
    }
    const rightHand = new Float64Array(componentCount);
    for (let component = 0; component < componentCount; component++) {
      let total = 0;
      for (let observation = 0; observation < observationCount; observation++) {
        total += design[observation * componentCount + component] * residual[observation];
      }
      rightHand[component] = total / observationCount;
    }
    const matrix = Float64Array.from(gram);
    for (let component = 0; component < componentCount; component++) {
      matrix[component * componentCount + component] += schedule[sweep];
    }
    identity = solveSymmetricPositiveDefinite(matrix, rightHand, componentCount);
    if (identityClip !== null) {
      // 周ごとに切る。最後だけ切ると、次の周の相似変換が切る前の形に合わせたままになり、
      // 係数と相似変換が別の形を指す。
      for (let component = 0; component < componentCount; component++) {
        identity[component] = Math.min(identityClip, Math.max(-identityClip, identity[component]));
      }
    }

    const fitted = similarity.apply(xyOf(evaluateModel(landmarkModel, identity)));
    let squared = 0;
    for (let point = 0; point < pointCount; point++) {
      const dx = photo[point * 2] - fitted[point * 2];
      const dy = photo[point * 2 + 1] - fitted[point * 2 + 1];
      squared += dx * dx + dy * dy;
    }
    residualRms.push(Math.sqrt(squared / pointCount));
  }

  if (similarity === null) throw new Error('相似変換が解けなかった');
  return { identity, similarity, residualRmsPixels: residualRms };
}

/**
 * 対称正定値の連立方程式を Cholesky 分解で解く。
 *
 * `matrix = L Lᵀ` に分解してから 2 段で解く。一般の solve ではなく分解を挟むのは、
 * 正定値でない行列（λ が小さすぎる・基底に重複がある）が来たときに例外で止まるため。
 */
export function solveSymmetricPositiveDefinite(
  matrix: Float64Array,
  rightHand: Float64Array,
  size: number,
): Float64Array {
  const factor = new Float64Array(size * size);
  for (let row = 0; row < size; row++) {
    for (let column = 0; column <= row; column++) {
      let total = matrix[row * size + column];
      for (let inner = 0; inner < column; inner++) {
        total -= factor[row * size + inner] * factor[column * size + inner];
      }
      if (row === column) {
        if (!(total > 0)) {
          throw new Error(`Cholesky 分解できない（対角 ${total} が正でない = 正定値でない）`);
        }
        factor[row * size + column] = Math.sqrt(total);
      } else {
        factor[row * size + column] = total / factor[column * size + column];
      }
    }
  }
  const intermediate = new Float64Array(size);
  for (let row = 0; row < size; row++) {
    let total = rightHand[row];
    for (let inner = 0; inner < row; inner++) total -= factor[row * size + inner] * intermediate[inner];
    intermediate[row] = total / factor[row * size + row];
  }
  const solution = new Float64Array(size);
  for (let row = size - 1; row >= 0; row--) {
    let total = intermediate[row];
    for (let inner = row + 1; inner < size; inner++) {
      total -= factor[inner * size + row] * solution[inner];
    }
    solution[row] = total / factor[row * size + row];
  }
  return solution;
}
