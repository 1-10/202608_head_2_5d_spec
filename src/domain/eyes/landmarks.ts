// 写真の各眼について虹彩中心と半径を測る MediaPipe 点の定義。
//
// 瞼の輪郭は扱わない。眼球テクスチャでは瞼や遮蔽物も写真の見た目として焼くため、開口部による
// 画素の採否判定を行わない。

export const FACE_MESH_POINT_COUNT = 468;
export const IRIS_POINT_COUNT = 5;
export const LANDMARK_COUNT = FACE_MESH_POINT_COUNT + 2 * IRIS_POINT_COUNT;

/** 片眼の MediaPipe 虹彩5点。`name` は解剖学的な左右を表さない。 */
export interface EyeLandmarkGroup {
  readonly name: string;
  readonly irisCenter: number;
  readonly irisRim: readonly number[];
}

function irisGroup(name: string, first: number): EyeLandmarkGroup {
  const irisRim: number[] = [];
  for (let offset = 1; offset < IRIS_POINT_COUNT; offset++) irisRim.push(first + offset);
  for (const index of [first, ...irisRim]) {
    if (!(index >= FACE_MESH_POINT_COUNT && index < LANDMARK_COUNT)) {
      throw new Error(`虹彩点の範囲が不正: ${name}`);
    }
  }
  return { name, irisCenter: first, irisRim };
}

export const EYE_GROUPS: readonly EyeLandmarkGroup[] = [
  irisGroup('mediapipe_right', FACE_MESH_POINT_COUNT),
  irisGroup('mediapipe_left', FACE_MESH_POINT_COUNT + IRIS_POINT_COUNT),
];

/** 片眼の虹彩5点を (5, 2) で返す。 */
export function eyeLandmarks(landmarks: Float64Array, group: EyeLandmarkGroup): Float64Array {
  if (landmarks.length !== LANDMARK_COUNT * 2) {
    throw new Error(
      `ランドマークの形が (${LANDMARK_COUNT}, 2) ではない: ${landmarks.length / 2}` +
        '（虹彩10点を含む検出器の生の出力を渡すこと）',
    );
  }
  const indices = [group.irisCenter, ...group.irisRim];
  const out = new Float64Array(IRIS_POINT_COUNT * 2);
  indices.forEach((index, slot) => {
    out[slot * 2] = landmarks[index * 2];
    out[slot * 2 + 1] = landmarks[index * 2 + 1];
  });
  return out;
}

/** 虹彩5点から中心と、縁4点までの平均半径（写真画素）を返す。 */
export function irisCenterAndRadius(
  irisPoints: Float64Array,
): { center: [number, number]; radius: number } {
  if (irisPoints.length !== IRIS_POINT_COUNT * 2) {
    throw new Error(`虹彩の点の形が (${IRIS_POINT_COUNT}, 2) ではない: ${irisPoints.length / 2}`);
  }
  const center: [number, number] = [irisPoints[0], irisPoints[1]];
  let total = 0;
  for (let slot = 1; slot < IRIS_POINT_COUNT; slot++) {
    total += Math.hypot(irisPoints[slot * 2] - center[0], irisPoints[slot * 2 + 1] - center[1]);
  }
  return { center, radius: total / (IRIS_POINT_COUNT - 1) };
}
