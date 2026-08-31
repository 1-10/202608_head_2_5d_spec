// 抽象インターフェース。
//
// ユースケースが外の世界に対して必要とする能力を、実装から切り離して宣言する。infrastructure 側が
// これを実装し、composition が組み立てて注入する。
//
// このモジュールは具体実装を import しない。ここに `onnxruntime-web` や `@mediapipe/tasks-vision` が
// 現れたら、依存の向きが壊れている。

import { AtlasBake, BakeSettings } from '../domain/atlas/bake';
import { AlphaImage, RgbImage } from '../domain/contract';
import { DepthNormalResult, HairMask, PersonSegmentation, ScalarField } from '../domain/field';
import { Similarity2d } from '../domain/gnm/fit';
import { PhotoRgb } from '../domain/photo';

/**
 * `FaceLandmarkDetector.detect` が返す点数（顔メッシュ 468 + 虹彩 10）。
 *
 * **port は検出器の出力を落とさずそのまま返す。** どの点を使うかは消費側の関心で、検出器のアダプタは
 * 点を選ばない（形状フィットは 468 未満だけを見る `MEDIAPIPE_IBUG68`、眼球は虹彩 10 点を見る
 * `domain/eyes`）。ここで虹彩を落とすと、後から必要になった消費者が推論をもう一度走らせる以外に
 * 取り戻す手が無くなる。
 */
export const FACE_LANDMARK_COUNT = 478;

/** 顔メッシュの点数。虹彩 10 点はこの後ろに並ぶ。 */
export const FACE_MESH_LANDMARK_COUNT = 468;

/** 虹彩の点数（片目 5 点 × 2）。 */
export const IRIS_LANDMARK_COUNT = 10;

/** 写真から顔ランドマークを取る能力。 */
export interface FaceLandmarkDetector {
  /**
   * 写真から顔ランドマークを取る。
   *
   * 返すのは `(FACE_LANDMARK_COUNT, 2)` を平坦化した**画像画素座標** (x, y)。正規化座標ではなく
   * 画素で返すのは、切り出し矩形の計算が画素で行われるため。
   *
   * 並び（MediaPipe FaceLandmarker の出力そのまま）:
   *
   *       0〜467  顔メッシュ
   *     468〜472  片目の虹彩（中心 + 縁 4 点）
   *     473〜477  もう片方の虹彩（中心 + 縁 4 点）
   *
   * 顔が検出できなければ `domain/errors.FaceNotDetectedError` を投げる。複数写っていても失敗させず、
   * **主役 1 人ぶんだけ**を返す。**主役の選び方は実装の裁量ではない** — 規則は
   * `domain/faceSubject` が持ち、実装はそれを適用するだけ。
   */
  detect(photo: PhotoRgb): Promise<Float64Array>;
}

/** 写真から人物のクラス別の領域を取る能力。 */
export interface PersonSegmenter {
  /**
   * クラス別の確信度を取る。
   *
   * **1 回の推論で全部返す。** セグメンタはクラス別の確信度を一度に出すので、用途ごとにメソッドを
   * 分けると同じ写真へ複数回推論する経路ができる。
   *
   * **場は写真と縦横比の等しい格子に乗せる**（写真そのものの解像度なら自明に満たす）。4 枚とも同じ
   * 形で、`rect` は画像全体。消費側は格子の 1 画素を距離の単位として使うので（`domain/hair/subject`
   * の chamfer 距離と BFS の歩数）、画素が正方でないと距離が縦横で歪む。
   *
   * **クラスを混ぜない。** 髪と装飾品を足すのは髪シェル側の判断で、しかも装飾品には位置の門が要る。
   * 門の位置はランドマークから決まるが、**この port の契約は画像 1 枚**なのでアダプタはそれを
   * 知らない。足す・床を引く・判定する は `domain/hair/mask.hairShellMask` が行う。
   */
  segment(photo: PhotoRgb): Promise<PersonSegmentation>;
}

/** 写真から相対深度・表面法線・人物前景を取る能力。 */
export interface DepthNormalEstimator {
  /**
   * 正方領域から深度・法線・前景を1回の推論で取る。
   *
   * 3つを別メソッドに分けないのは、同一モデルの multi-task 出力であり、分けると同じ推論を3回走らせる
   * か結果の整合を呼び出し側が保証することになるため。
   *
   * **切り出しは呼び出し側が決める。** 実装側に既定の切り出しを持たせない — 持たせた瞬間に「呼び出し
   * 側が何も考えなくても動く」経路ができ、その既定が写真の額縁で決まる限り被写体によっては黙って
   * 外れる（`domain/gnm/crop` の冒頭に実例）。
   *
   * **正方形しか受けない。** モデルの入力が正方形なので、長方形を許すと呼び出し側ごとに縦横比の
   * 潰し方が分かれる。
   *
   * 返る場の `rect` はこの正方領域を覆う。**rect の外は 0**（`ScalarField` の規約）。
   */
  estimateSquare(
    photo: PhotoRgb,
    square: { x: number; y: number; size: number },
  ): Promise<DepthNormalResult>;
}

/**
 * 写真をGNM公式UVアトラスへ焼く能力。
 *
 * レイアウトの計算は重い外部資源（GPU / キャッシュ）を使いうるので、ユースケースはその実装を知らず
 * この Port だけを呼ぶ。
 */
export interface AtlasBaker {
  bake(input: {
    photo: PhotoRgb;
    vertices: Float64Array;
    triangles: Uint32Array;
    vertexUvs: Float32Array;
    componentId: Uint8Array;
    similarity: Similarity2d;
    personMask: ScalarField;
    skinBaseColor: readonly [number, number, number];
    settings: BakeSettings;
    fillRegionId: Uint8Array | null;
    photoOnlyRegion: Uint8Array | null;
    mouthRimRegion: Float32Array | null;
  }): AtlasBake;
}

/**
 * 髪マスク精細化と半透明境界の色補正を行う能力。
 *
 * 品質判断は domain の実装が正本で、アダプタは重い box filter を別の実行環境へ移すだけ。
 */
export interface HairImageProcessor {
  /** 写真RGBをガイドに粗い髪マスクを精細化する。 */
  refineMask(photo: PhotoRgb, mask: HairMask, maximumDimension: number): HairMask;
  /** 半透明境界から背景色を除き、透明側へ近傍の髪色を延ばす。 */
  decontaminateTexture(photo: RgbImage, alpha: AlphaImage): RgbImage;
}
