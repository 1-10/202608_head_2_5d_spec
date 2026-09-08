// 3D 確認ビューのカメラ（フリーカメラ）。three.js に依存しない純粋計算。
//
// ## 状態は 4 つだけ
//
// **位置（ワールド）+ pitch + yaw + 周回半径。** 周回の中心は持たない — `位置 − 後ろ向き × 半径`
// で毎回導出する。中心を状態として持つと、位置や回転を打ち込んだ瞬間に中心が古くなり、次に回した
// ときへ古い中心が効く。
//
// **「拡大率」と「注視点からの距離」も持たない。** どちらも周回半径の言い換えで、旧実装は
// `距離 / 拡大率` を毎フレーム掛けていた。同じことを 2 つの数で持つと、片方を打ち込んだときに
// もう片方が黙って効き続ける。届く範囲は旧実装のまま — 距離 0.35〜3m と拡大 0.3〜5 倍の組で届いた
// 0.07〜10m を、そのまま半径の範囲にしてある。
//
// ## 角度
//
// 回転は pitch（X）と yaw（Y）だけで**ロールを持たない**。姿勢は `Ry(yaw) * Rx(pitch)`
// （three.js の `Euler(pitch, yaw, 0, 'YXZ')` と同じ順）で、カメラは自分の -Z を見る。旧実装は
// 「その Euler で置いた位置から `lookAt(注視点)`」だったが、pan が注視点と位置を同じだけ動かす
// ので**向きは常にこの Euler と厳密に一致していた** — つまりフリーカメラは旧実装の言い換えで、
// 絵は変わらない（`tests/camera.test.ts` が行列で突き合わせている）。
//
// **位置も角度も 3D ビューのワールド（GNM 空間）の値そのもの。** 右パネルにはこれをそのまま出す
// （導出も変換も挟まないので、パネルの数とカメラの実体がズレようがない）。既定は Unity 側
// `Scenes/Viewer.unity` の `MainCamera` の写し（画角 20° / y 0.297m / z 1.3m）で、**同じ絵**になる。
//
// **ただしあちらの回転の数はそのまま打てない。** Unity のカメラは自分の +Z を、three.js のカメラは
// -Z を見るので、同じ向きを向くのに要る yaw が 180° ずれる（あちらは同じ位置に置いて 180° 振って
// いる）。**Unity の数へ直して出す変換は持たない** — 既定の回転が 180 と出ても web で見る人に意味が
// 通らないし、変換を挟むと画面の数がどちらの空間のものか読めなくなる。加えて Unity 空間は GNM 空間
// の X を反転した左手系なので（根拠は `domain/preview/pose` の「角度の向き」）、正面から外れると
// x と yaw の符号も入れ替わる。

/** ワールド座標（メートル）。 */
export type Vector3 = readonly [number, number, number];

/** 画角（度）。正本は Unity 側 `MainCamera` の `field of view`。 */
export const DEFAULT_FOV_DEGREES = 20;
export const MINIMUM_FOV_DEGREES = 10;
export const MAXIMUM_FOV_DEGREES = 60;

/** 周回半径（メートル）の既定。正本は Unity 側 `MainCamera` の z。 */
export const DEFAULT_ORBIT_RADIUS_METERS = 1.3;

/**
 * 周回半径の範囲。
 *
 * 旧実装は「距離 0.35〜3m」と「拡大 0.3〜5 倍」を別々にクランプしていた。半径に畳んだので、
 * その組で届いた範囲（0.35/5 〜 3/0.3）をそのまま範囲にする — 寄れる所も引ける所も変わらない。
 */
export const MINIMUM_ORBIT_RADIUS_METERS = 0.07;
export const MAXIMUM_ORBIT_RADIUS_METERS = 10;

/**
 * 注視点の高さ（メートル）。シーンが無いときの既定。
 *
 * Unity 側 `MainCamera` の y（GNM の眼の高さ）。**シーンがあるときは頭部の外接箱の中心を使う** —
 * あちらのカメラは体に載せた状態で使う前提のリグで、頭だけを見るこちらでは頭が枠の中心へ来る方が
 * 確認しやすい（旧 web 版も頭の中心を見ていた）。判断は `domain/preview/scene.previewTarget`。
 */
export const TARGET_HEIGHT_METERS = 0.297;

/**
 * pitch の上限（度）。真上・真下の手前で止める。
 *
 * ドラッグでも入力でも同じ上限にする（操作ごとに別の上限を持つと、打ち込める値と回せる値が食い違う）。
 */
export const MAXIMUM_PITCH_DEGREES = 85;

/** yaw の折り返し（度）。上限ではなく、`[-180, 180)` へ畳む先。 */
export const MAXIMUM_YAW_DEGREES = 180;

/** ドラッグ 1 画素あたりのカメラ回転（度）。旧実装の 0.01 rad/px を度へ直した値。 */
export const ORBIT_DEGREES_PER_PIXEL = (0.01 * 180) / Math.PI;

/** ホイールの delta がこれだけ動くと半径が e 倍になる。旧実装（デスクトップ側と同じ）の値。 */
export const ZOOM_PIXELS_PER_E = 1200;

/**
 * カメラの姿勢。
 *
 * `orbitRadiusMeters` は**ドラッグで回るときの中心までの距離**。中心はここに持たず
 * `orbitCenter()` で導出する。
 */
export interface CameraPose {
  readonly position: Vector3;
  /** X 回転（度）。正で上を向く（three.js の Euler と同じ向き）。 */
  readonly pitchDegrees: number;
  /** Y 回転（度）。 */
  readonly yawDegrees: number;
  readonly orbitRadiusMeters: number;
}

/** カメラの基底（ワールド）。`back` はカメラ局所 +Z で、**視線と逆向き**。 */
export interface CameraBasis {
  readonly right: Vector3;
  readonly up: Vector3;
  readonly back: Vector3;
}

const RADIANS_PER_DEGREE = Math.PI / 180;

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

/** 角度を `[-MAXIMUM_YAW_DEGREES, MAXIMUM_YAW_DEGREES)` へ畳む。回し続けても読める数のままにする。 */
function wrapDegrees(value: number): number {
  const turn = 2 * MAXIMUM_YAW_DEGREES;
  return ((((value + MAXIMUM_YAW_DEGREES) % turn) + turn) % turn) - MAXIMUM_YAW_DEGREES;
}

function add(a: Vector3, b: Vector3, scale = 1): Vector3 {
  return [a[0] + b[0] * scale, a[1] + b[1] * scale, a[2] + b[2] * scale];
}

/** 可動域へ入れる（pitch は上限でクランプ・yaw は畳む・半径は範囲へ）。 */
export function clampCameraPose(pose: CameraPose): CameraPose {
  return {
    position: pose.position,
    pitchDegrees: clamp(pose.pitchDegrees, -MAXIMUM_PITCH_DEGREES, MAXIMUM_PITCH_DEGREES),
    yawDegrees: wrapDegrees(pose.yawDegrees),
    orbitRadiusMeters: clamp(
      pose.orbitRadiusMeters,
      MINIMUM_ORBIT_RADIUS_METERS,
      MAXIMUM_ORBIT_RADIUS_METERS,
    ),
  };
}

/** 注視点の正面（+Z 側）から見る姿勢。`resetView` と既定値がここから始まる。 */
export function cameraPoseAt(
  target: Vector3,
  radiusMeters = DEFAULT_ORBIT_RADIUS_METERS,
): CameraPose {
  return clampCameraPose({
    position: [target[0], target[1], target[2] + radiusMeters],
    pitchDegrees: 0,
    yawDegrees: 0,
    orbitRadiusMeters: radiusMeters,
  });
}

/**
 * 姿勢の基底を作る。
 *
 * `Ry(yaw) * Rx(pitch)` の 3 列（right / up / back）。three.js の
 * `Vector3(1,0,0).applyEuler(new Euler(pitch, yaw, 0, 'YXZ'))` などと同じ値になる。
 */
export function cameraBasis(pose: {
  readonly pitchDegrees: number;
  readonly yawDegrees: number;
}): CameraBasis {
  const pitch = pose.pitchDegrees * RADIANS_PER_DEGREE;
  const yaw = pose.yawDegrees * RADIANS_PER_DEGREE;
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  return {
    right: [cy, 0, -sy],
    up: [sy * sp, cp, cy * sp],
    back: [sy * cp, -sp, cy * cp],
  };
}

/** 周回の中心（`位置 − 後ろ向き × 半径`）。**状態としては持たない。** */
export function orbitCenter(pose: CameraPose): Vector3 {
  return add(pose.position, cameraBasis(pose).back, -pose.orbitRadiusMeters);
}

/** 画面の高さの半分に相当するワールドの長さ（周回の中心の面での量）。 */
export function viewHalfHeightMeters(pose: CameraPose, fovDegrees: number): number {
  return pose.orbitRadiusMeters * Math.tan((fovDegrees * RADIANS_PER_DEGREE) / 2);
}

/** 中心を固定して回す（左ドラッグ）。半径が変わらないので被写体からの距離も変わらない。 */
export function orbitBy(
  pose: CameraPose,
  deltaYawDegrees: number,
  deltaPitchDegrees: number,
): CameraPose {
  const center = orbitCenter(pose);
  const turned = clampCameraPose({
    ...pose,
    pitchDegrees: pose.pitchDegrees + deltaPitchDegrees,
    yawDegrees: pose.yawDegrees + deltaYawDegrees,
  });
  return {
    ...turned,
    position: add(center, cameraBasis(turned).back, turned.orbitRadiusMeters),
  };
}

/** 中心を固定して半径を掛け算で変える（ホイール）。向きは変わらない。 */
export function scaleOrbitRadius(pose: CameraPose, scale: number): CameraPose {
  const center = orbitCenter(pose);
  const scaled = clampCameraPose({ ...pose, orbitRadiusMeters: pose.orbitRadiusMeters * scale });
  return {
    ...scaled,
    position: add(center, cameraBasis(scaled).back, scaled.orbitRadiusMeters),
  };
}

/** 画面に平行に動かす（右ドラッグ）。向きも半径も変わらず、中心が一緒に動く。 */
export function moveInView(
  pose: CameraPose,
  rightMeters: number,
  upMeters: number,
): CameraPose {
  const basis = cameraBasis(pose);
  const moved = add(add(pose.position, basis.right, rightMeters), basis.up, upMeters);
  return { ...pose, position: moved };
}

/** 位置と回転を差し替える（パネルの入力）。中心は導出なので勝手に付いてくる。 */
export function withTransform(
  pose: CameraPose,
  position: Vector3,
  pitchDegrees: number,
  yawDegrees: number,
): CameraPose {
  return clampCameraPose({ ...pose, position, pitchDegrees, yawDegrees });
}

/**
 * 位置はそのままで注視点を向く。半径も注視点までの距離に合わせる。
 *
 * 手で回転を打って被写体が枠から外れたときに戻る口。注視点と位置が同じ点なら向きが決まらないので
 * 何もしない。
 */
export function lookAt(pose: CameraPose, target: Vector3): CameraPose {
  const dx = target[0] - pose.position[0];
  const dy = target[1] - pose.position[1];
  const dz = target[2] - pose.position[2];
  const distance = Math.hypot(dx, dy, dz);
  if (distance < MINIMUM_ORBIT_RADIUS_METERS) return pose;
  // 視線は -back = (-sy*cp, sp, -cy*cp)。cp >= 0（|pitch| <= 90）なので atan2 で yaw が決まる。
  const pitch = Math.asin(clamp(dy / distance, -1, 1)) / RADIANS_PER_DEGREE;
  const yaw = Math.atan2(-dx / distance, -dz / distance) / RADIANS_PER_DEGREE;
  return clampCameraPose({
    position: pose.position,
    pitchDegrees: pitch,
    yawDegrees: yaw,
    orbitRadiusMeters: distance,
  });
}
