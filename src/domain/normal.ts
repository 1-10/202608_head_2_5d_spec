// カメラ空間の法線を GNM 空間へ移す。
//
// DAViD が返すのは**カメラ空間**の法線（x = 画像の右、z = カメラ向き）。GNM 空間は右手系 /
// X = 解剖学的左 / Y = 上 / Z = 前。この 2 つの間の変換をここに 1 つだけ置く。
//
// **法線は出力契約に入っていない。** 使うのは `domain/hair/relief`（毛束の起伏）だけ。それでも
// 独立したモジュールにしているのは、**y 軸の向きの正本を 1 箇所にする**ため。
//
// y 軸の向き
// ----------
// DAViD の法線の y が画像の下向きか上向きかは、`NORMAL_Y_AXIS_DOWNWARD` が正本。取り違えても
// 例外にならず起伏の凹凸が上下入れ替わるだけなので、テストだけが番人になる。

import { Similarity2d } from './gnm/fit';

/**
 * DAViD の法線の y 軸が画像の v（下向き）と同じ向きか。**実測で決めた**。
 *
 * 以前は「要実測」のまま `true` が置かれていて、**それが間違っていた**。
 *
 * 測り方: 写真をそのまま焼いた texel（重み 1.0）で、測った法線と GNM メッシュの幾何法線の角度差を
 * 見る。どちらも同じ面を正面から見ているので近い値になるはずで、y を取り違えると上下が反転して
 * 大きく開く。実写 3 枚（アトラス一辺 1024）で、下向き（旧）は角度差 中央 42.4 / 41.1 / 47.0°、
 * 上向き（現行）は 12.8 / 14.0 / 18.2°。3 倍の差があるので判定は曖昧でない。
 */
export const NORMAL_Y_AXIS_DOWNWARD = false;

/** この長さ未満の法線はデータ無しとして扱う（rect の外は 0 ベクトルが返る）。 */
export const MIN_NORMAL_LENGTH = 0.5;

/**
 * カメラ空間の法線を GNM 空間の単位法線へ写す。
 *
 * xy だけ回して z は触らない。カメラ空間の xy と GNM 空間の xy はどちらもメートルで、相似変換の
 * 回転部分は直交行列なので、その転置が画像フレーム → GNM フレームの写像になる（スケールは単位
 * ベクトルに効かない）。
 *
 * @returns 単位法線 (3,) と、有効かどうか
 */
export function cameraNormalToGnm(
  nx: number,
  ny: number,
  nz: number,
  similarity: Similarity2d,
  yAxisDownward = NORMAL_Y_AXIS_DOWNWARD,
): { normal: [number, number, number]; valid: boolean } {
  const length = Math.hypot(nx, ny, nz);
  const valid = length > MIN_NORMAL_LENGTH;
  const safe = Math.max(length, 1e-12);
  const unitX = nx / safe;
  const unitY = ny / safe;
  const unitZ = nz / safe;

  // 画像フレームの xy（x = 右 / y = 下）へ揃えてから GNM フレームへ回す。
  const imageX = unitX;
  const imageY = yAxisDownward ? unitY : -unitY;
  const rotation = similarity.rotation;
  // `imageXy @ rotation` = `rotation.T @ v`（rotation が直交行列なので）。
  const gnmX = imageX * rotation[0] + imageY * rotation[2];
  const gnmY = imageX * rotation[1] + imageY * rotation[3];
  const gnmLength = Math.max(Math.hypot(gnmX, gnmY, unitZ), 1e-12);
  return {
    normal: [gnmX / gnmLength, gnmY / gnmLength, unitZ / gnmLength],
    valid,
  };
}
