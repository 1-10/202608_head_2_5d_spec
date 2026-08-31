// 首と視線（ジョイント階層 + LBS）。3D ビューだけが使う。
//
// **正本は Unity 側の `Scripts/GnmHeadInstance` / `GnmSkeleton` と `Viewer/GnmHeadPoseController`。**
// 可動域・首と頭の配分・マウス追従の写し方はあちらの既定値と同じ値を持つ。
//
// v3_0 / head で確認済みの前提に乗っている（アセット生成時に毎回検査し、崩れたら生成が落ちる）:
//
// - `bone_aligned_template_joint_orientations` が単位行列 → bindpose は bind 位置の平行移動の逆だけ
// - `pose_correctives_regressor` が全ゼロ → ジョイントを回しても LBS だけで公式と一致する
// - 重み和がちょうど 1 / 影響ボーンは最大 2 本
//
// ## 角度の向き
//
// **UI の角度は Unity 側と同じ数**（正の yaw で被写体から見て右を向く / 正の pitch で下を向く）にする。
// Unity 空間は GNM 空間の X を反転した左手系なので、GNM 空間での回転は
// `Ry(-yaw) * Rx(pitch)`（右手系）になる。鏡像変換 M = diag(-1,1,1) による共役
// `M Ry(a) M = Ry(-a)` / `M Rx(a) M = Rx(a)` がその根拠。
//
// ジョイント位置は identity で動く（実測 6.9mm）ので、guest ごとに bind 位置を作り直す。テンプレート
// のまま使うと bind pose がずれる。

import { GnmPreviewAsset } from './asset';

/** 首の可動域（度）。正本は Unity 側 `GnmHeadPoseController._yawLimit`。 */
export const YAW_LIMIT_DEGREES = 15;

/** 同 `_pitchLimit`。 */
export const PITCH_LIMIT_DEGREES = 12;

/** 視線の可動域（度）。同 `_gazeLimit`。 */
export const GAZE_LIMIT_DEGREES = 10;

/** 首へ配る割合。残りが頭。同 `_neckShare`。 */
export const NECK_SHARE = 0.3;

/** マウス追従の指数追従の速さ（1/秒）。同 `_followSpeed`。 */
export const FOLLOW_SPEED = 8;

/** ドラッグ 1 画素あたりの角度（度）。旧 web 版 `OrbitDragController` の既定値。 */
export const DEGREES_PER_PIXEL = 0.25;

/** 首と視線の角度（度）。 */
export interface HeadPose {
  readonly headYawDegrees: number;
  readonly headPitchDegrees: number;
  readonly gazeYawDegrees: number;
  readonly gazePitchDegrees: number;
}

export const NEUTRAL_POSE: HeadPose = {
  headYawDegrees: 0,
  headPitchDegrees: 0,
  gazeYawDegrees: 0,
  gazePitchDegrees: 0,
};

function clamp(value: number, limit: number): number {
  return Math.min(limit, Math.max(-limit, value));
}

/** 可動域でクランプする。上限を持つのはこの層（GNM 本体はジョイントを公開するだけ）。 */
export function clampPose(pose: HeadPose): HeadPose {
  return {
    headYawDegrees: clamp(pose.headYawDegrees, YAW_LIMIT_DEGREES),
    headPitchDegrees: clamp(pose.headPitchDegrees, PITCH_LIMIT_DEGREES),
    gazeYawDegrees: clamp(pose.gazeYawDegrees, GAZE_LIMIT_DEGREES),
    gazePitchDegrees: clamp(pose.gazePitchDegrees, GAZE_LIMIT_DEGREES),
  };
}

/**
 * 画面上の位置（中心を原点に -1〜1）から首と視線を作る。
 *
 * カーソル下のワールド座標を見に行くより画角に依存しない。視線は首より速く動く（実際の人間もそう）
 * ので、平滑化は首だけに掛ける。
 */
export function followPointerPose(
  previous: HeadPose,
  x: number,
  y: number,
  deltaSeconds: number,
): HeadPose {
  const clampedX = Math.min(1, Math.max(-1, x));
  const clampedY = Math.min(1, Math.max(-1, y));
  const smoothing = FOLLOW_SPEED <= 0 ? 1 : 1 - Math.exp(-FOLLOW_SPEED * deltaSeconds);
  const lerp = (from: number, to: number): number => from + (to - from) * smoothing;
  return clampPose({
    headYawDegrees: lerp(previous.headYawDegrees, clampedX * YAW_LIMIT_DEGREES),
    headPitchDegrees: lerp(previous.headPitchDegrees, -clampedY * PITCH_LIMIT_DEGREES),
    gazeYawDegrees: clampedX * GAZE_LIMIT_DEGREES,
    gazePitchDegrees: -clampedY * GAZE_LIMIT_DEGREES,
  });
}

/**
 * guest の identity を当てたジョイントの bind 位置（GNM 空間・絶対）を返す。
 *
 * @returns (ジョイント数, 3)
 */
export function jointRestPositions(
  preview: GnmPreviewAsset,
  identity: Float64Array,
): Float64Array {
  const { jointCount } = preview;
  const out = new Float64Array(jointCount * 3);
  for (let index = 0; index < out.length; index++) {
    out[index] = preview.templateJointPositions[index];
  }
  const componentCount = preview.jointIdentityBasis.length / (jointCount * 3);
  const limit = Math.min(componentCount, identity.length);
  for (let component = 0; component < limit; component++) {
    const coefficient = identity[component];
    if (coefficient === 0) continue;
    const base = component * jointCount * 3;
    for (let index = 0; index < jointCount * 3; index++) {
      out[index] += preview.jointIdentityBasis[base + index] * coefficient;
    }
  }
  return out;
}

/** 3x3 を行優先 9 要素で持つ。 */
type Matrix3 = Float64Array;

function rotationYawPitch(yawDegrees: number, pitchDegrees: number): Matrix3 {
  // Unity の Quaternion.Euler(pitch, yaw, 0) を GNM 空間（右手系）へ写したもの。
  const yaw = (-yawDegrees * Math.PI) / 180;
  const pitch = (pitchDegrees * Math.PI) / 180;
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  // Ry(yaw) * Rx(pitch)
  return Float64Array.from([
    cy,
    sy * sp,
    sy * cp,
    0,
    cp,
    -sp,
    -sy,
    cy * sp,
    cy * cp,
  ]);
}

function multiply(left: Matrix3, right: Matrix3): Matrix3 {
  const out = new Float64Array(9);
  for (let row = 0; row < 3; row++) {
    for (let column = 0; column < 3; column++) {
      let sum = 0;
      for (let inner = 0; inner < 3; inner++) {
        sum += left[row * 3 + inner] * right[inner * 3 + column];
      }
      out[row * 3 + column] = sum;
    }
  }
  return out;
}

/**
 * ジョイントごとの局所回転を作る。
 *
 * 首と頭で角度を分けるのは Unity 側と同じ（`NECK_SHARE` が首、残りが頭）。眼球は左右同じ視線。
 */
export function jointLocalRotations(
  preview: GnmPreviewAsset,
  pose: HeadPose,
  neckShare = NECK_SHARE,
): Float64Array {
  const clamped = clampPose(pose);
  const out = new Float64Array(preview.jointCount * 9);
  const identity = Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  for (let joint = 0; joint < preview.jointCount; joint++) {
    const name = preview.jointNames[joint];
    let rotation = identity;
    if (name === 'neck') {
      rotation = rotationYawPitch(
        clamped.headYawDegrees * neckShare,
        clamped.headPitchDegrees * neckShare,
      );
    } else if (name === 'head') {
      rotation = rotationYawPitch(
        clamped.headYawDegrees * (1 - neckShare),
        clamped.headPitchDegrees * (1 - neckShare),
      );
    } else if (name === 'left_eye' || name === 'right_eye') {
      rotation = rotationYawPitch(clamped.gazeYawDegrees, clamped.gazePitchDegrees);
    }
    out.set(rotation, joint * 9);
  }
  return out;
}

/**
 * ジョイントごとの `世界行列 * bind の逆` を返す（回転 3x3 + 平行移動 3 の 12 要素/ジョイント）。
 *
 * bind pose に回転が無いので `bind の逆` は bind 位置の平行移動の逆だけ。よって
 * `skin = R_world * (v - rest) + t_world`。
 */
export function jointSkinMatrices(
  preview: GnmPreviewAsset,
  jointRest: Float64Array,
  localRotations: Float64Array,
): Float64Array {
  const { jointCount } = preview;
  const out = new Float64Array(jointCount * 12);
  const worldRotation = new Float64Array(jointCount * 9);
  const worldTranslation = new Float64Array(jointCount * 3);

  for (let joint = 0; joint < jointCount; joint++) {
    const parent = preview.jointParentIndices[joint];
    if (parent >= joint) {
      throw new Error(`ジョイント ${joint} の親が ${parent} で、親が先に来ていない`);
    }
    const local = localRotations.subarray(joint * 9, joint * 9 + 9) as Matrix3;
    if (parent < 0) {
      worldRotation.set(local, joint * 9);
      for (let axis = 0; axis < 3; axis++) {
        worldTranslation[joint * 3 + axis] = jointRest[joint * 3 + axis];
      }
      continue;
    }
    const parentRotation = worldRotation.subarray(parent * 9, parent * 9 + 9) as Matrix3;
    worldRotation.set(multiply(parentRotation, local), joint * 9);
    // 親の bind 位置からの相対位置を親の世界回転で回して足す。
    for (let axis = 0; axis < 3; axis++) {
      let sum = worldTranslation[parent * 3 + axis];
      for (let inner = 0; inner < 3; inner++) {
        sum +=
          parentRotation[axis * 3 + inner] *
          (jointRest[joint * 3 + inner] - jointRest[parent * 3 + inner]);
      }
      worldTranslation[joint * 3 + axis] = sum;
    }
  }

  for (let joint = 0; joint < jointCount; joint++) {
    for (let index = 0; index < 9; index++) {
      out[joint * 12 + index] = worldRotation[joint * 9 + index];
    }
    for (let axis = 0; axis < 3; axis++) {
      let sum = worldTranslation[joint * 3 + axis];
      for (let inner = 0; inner < 3; inner++) {
        sum -= worldRotation[joint * 9 + axis * 3 + inner] * jointRest[joint * 3 + inner];
      }
      out[joint * 12 + 9 + axis] = sum;
    }
  }
  return out;
}

/**
 * LBS を掛けた頂点を返す（入力を破壊しない）。
 *
 * @param vertices (頂点数, 3) identity と表情を当てた後の位置
 */
export function skinVertices(
  preview: GnmPreviewAsset,
  vertices: Float64Array,
  skinMatrices: Float64Array,
): Float64Array {
  const { vertexCount } = preview;
  if (vertices.length !== vertexCount * 3) {
    throw new Error(`頂点数が ${vertices.length / 3}（期待 ${vertexCount}）`);
  }
  const out = new Float64Array(vertices.length);
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    const x = vertices[vertex * 3];
    const y = vertices[vertex * 3 + 1];
    const z = vertices[vertex * 3 + 2];
    let outX = 0;
    let outY = 0;
    let outZ = 0;
    for (let slot = 0; slot < 2; slot++) {
      const weight = preview.skinJointWeights[vertex * 2 + slot];
      if (weight === 0) continue;
      const base = preview.skinJointIndices[vertex * 2 + slot] * 12;
      outX +=
        weight *
        (skinMatrices[base] * x +
          skinMatrices[base + 1] * y +
          skinMatrices[base + 2] * z +
          skinMatrices[base + 9]);
      outY +=
        weight *
        (skinMatrices[base + 3] * x +
          skinMatrices[base + 4] * y +
          skinMatrices[base + 5] * z +
          skinMatrices[base + 10]);
      outZ +=
        weight *
        (skinMatrices[base + 6] * x +
          skinMatrices[base + 7] * y +
          skinMatrices[base + 8] * z +
          skinMatrices[base + 11]);
    }
    out[vertex * 3] = outX;
    out[vertex * 3 + 1] = outY;
    out[vertex * 3 + 2] = outZ;
  }
  return out;
}

/**
 * 髪シェルへ当てる剛体変換（`head` ジョイントの `世界行列 * bind の逆`）。
 *
 * 髪は変形しないのでスキニングは要らない。メッシュ空間の座標のまま作って bind 位置ぶん引き戻すと、
 * bind 時の見た目を変えずに追従する（Unity 側と同じ扱い）。
 */
export function rigidTransformFor(
  preview: GnmPreviewAsset,
  skinMatrices: Float64Array,
  jointName: string,
): Float64Array {
  const joint = preview.jointNames.indexOf(jointName);
  if (joint < 0) throw new Error(`ジョイント '${jointName}' がアセットに無い`);
  return skinMatrices.slice(joint * 12, joint * 12 + 12);
}

/** 12 要素の剛体変換を (頂点数, 3) の Float32Array へ当てる（新しい配列を返す）。 */
export function applyRigidTransform(
  positions: Float32Array,
  transform: Float64Array,
): Float32Array {
  const out = new Float32Array(positions.length);
  for (let vertex = 0; vertex < positions.length / 3; vertex++) {
    const x = positions[vertex * 3];
    const y = positions[vertex * 3 + 1];
    const z = positions[vertex * 3 + 2];
    out[vertex * 3] =
      transform[0] * x + transform[1] * y + transform[2] * z + transform[9];
    out[vertex * 3 + 1] =
      transform[3] * x + transform[4] * y + transform[5] * z + transform[10];
    out[vertex * 3 + 2] =
      transform[6] * x + transform[7] * y + transform[8] * z + transform[11];
  }
  return out;
}
