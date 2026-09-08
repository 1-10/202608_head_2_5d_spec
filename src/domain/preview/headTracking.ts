// カメラの顔から 3D ビューの**首**を駆動する（純粋計算）。
//
// 表情はここに無い。**表情は点から係数を解く**（`expressionFit.ts`）。ここが持つのは
// 「MediaPipe が返す頭の姿勢行列 → リグの首の角度」だけ。
//
// **首は既定では駆動しない。** 表情だけを写すのが既定で、頭部姿勢はビューの `headPose`（手の
// スライダーとマウス追従）が持っている。そこへ黙ってカメラの姿勢を足すと「どちらが今の向きを
// 決めているか」が画面から読めなくなる。**明示的に選んだときだけ**首も動かす — どちらで動いて
// いるかは画面のスイッチが示す。

/**
 * 首の追従の時定数（秒）。**表情より長くする。**
 *
 * 首は表情より大きく、ゆっくり動く。表情と同じ 60ms で追うと、検出の跳ねがそのまま首の震えに
 * なって酔う。
 */
export const HEAD_TIME_CONSTANT_SECONDS = 0.12;

/**
 * カメラで振れる首の範囲（度）。**リグの可動域より広く取る。**
 *
 * リグの可動域は Unity 側の首 ±15° / pitch ±12°（`domain/preview/pose`）。人が実際に首を振る幅は
 * それよりずっと広いので、生の角度をそのまま渡すと**すぐ上限に張り付いて壁に当たった感じになる**
 * （実際にそう見えていた）。ここの幅をリグの可動域へ**線形に写す**ので、大きく振っても端で
 * 止まらず、range いっぱいまで滑らかに動く。
 */
export const HEAD_TRACKING_YAW_RANGE_DEGREES = 35;
export const HEAD_TRACKING_PITCH_RANGE_DEGREES = 25;

/**
 * 左右の向きを鏡にするか。
 *
 * 利用者は自分を映した画面を見ながら 3D の頭を合わせるので、自分が右を向いたら画面の頭も画面の
 * 右へ動く方が合わせやすい（映像をミラー表示しているのと同じ理由）。false にすれば実際の向きへ揃う。
 */
export const MIRROR_YAW = true;

/**
 * MediaPipe の頭の姿勢行列から yaw / pitch（度）を取り出す。
 *
 * **行優先の 4x4 として読む**（`facialTransformationMatrixes[].data` の並び）。回転部を
 * `Ry(yaw) * Rx(pitch)` として分解する — 正面で (0, 0)、可動域へ入れるのは呼ぶ側
 * （`domain/preview/pose.clampPose`）。
 *
 * **ロールは捨てる。** ビューの `HeadPose` が持たないし、確認用途で首を傾ける場面が無い。
 *
 * **pitch は符号を返す。** 行列から出る pitch とビューの `HeadPose.headPitchDegrees` は上下が逆で、
 * そのまま渡すと**上を向いたら下を向く**（実機でそう見えていた）。
 */
export function headPoseFromMatrix(
  matrix: Float32Array,
  mirrorYaw = MIRROR_YAW,
): { yawDegrees: number; pitchDegrees: number } | null {
  if (matrix.length < 16) return null;
  // 行優先: m[行][列] = matrix[行 * 4 + 列]。
  const m01 = matrix[1];
  const m02 = matrix[2];
  const m11 = matrix[5];
  const m12 = matrix[6];
  const m22 = matrix[10];
  // 回転行列なら列の長さは 1。全ゼロや壊れた行列をここで落とす（atan2 は 0 を返してしまう）。
  const firstColumn = Math.hypot(matrix[0], matrix[4], matrix[8]);
  if (!Number.isFinite(firstColumn) || firstColumn < 0.5) return null;
  const pitch = Math.atan2(-m12, m11);
  const yaw = Math.atan2(m02, m22);
  if (!Number.isFinite(pitch) || !Number.isFinite(yaw)) return null;
  // 使わないが、分解の前提（`Ry * Rx` にロールが混じっていない）を壊した行列を黙って通さない。
  if (!Number.isFinite(m01)) return null;
  const degrees = 180 / Math.PI;
  return {
    yawDegrees: yaw * degrees * (mirrorYaw ? -1 : 1),
    pitchDegrees: -pitch * degrees,
  };
}

/**
 * カメラの首の角度をリグの可動域へ写す。
 *
 * `range` を可動域（`limit`）へ**線形に**写し、外はクランプする。生の角度をそのまま渡すと、人の
 * 首の振り幅（±35° 程度）に対して可動域が ±15° しかないので、少し振っただけで上限に張り付く。
 */
export function mapHeadAngle(degrees: number, range: number, limit: number): number {
  if (!(range > 0)) return 0;
  const scaled = (degrees / range) * limit;
  return Math.min(limit, Math.max(-limit, scaled));
}

/**
 * 時定数から 1 フレームの混ぜ率を出す。
 *
 * `1 - exp(-dt / tau)` なので**フレームレートに依らず同じ速さ**で追う。`dt / tau` で近似すると
 * 60fps と 30fps で追従の速さが変わる。
 */
export function smoothingFactor(deltaSeconds: number, timeConstantSeconds: number): number {
  if (!(deltaSeconds > 0)) return 0;
  if (!(timeConstantSeconds > 0)) return 1;
  return 1 - Math.exp(-deltaSeconds / timeConstantSeconds);
}

/** 1 つの値を目標へ近づける。 */
export function smoothScalar(current: number, target: number, factor: number): number {
  return current + (target - current) * factor;
}
