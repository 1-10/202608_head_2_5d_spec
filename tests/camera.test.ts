// フリーカメラの姿勢の検査。
//
// 見るのは 2 つ:
//
// - **状態が 1 つに畳まっている**こと。周回の中心も拡大率も持たないので、回しても寄せても
//   平行移動しても、他の値が黙って古くならない（中心・半径・向きのどれが保たれるかを操作ごとに測る）
// - **絵が移植前と同じ**こと。旧実装は `Euler(pitch, yaw, 0, 'YXZ')` で置いた位置から
//   `lookAt(注視点)` していた。ここで作る姿勢のワールド行列がそれと一致すれば、view 行列
//   （= その逆行列）も一致する — つまり同じ画になる。
//
// 既定値は Unity 側 `Scenes/Viewer.unity` の `MainCamera` の写しなので、そこも押さえる。

import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import {
  CameraPose,
  DEFAULT_FOV_DEGREES,
  DEFAULT_ORBIT_RADIUS_METERS,
  MAXIMUM_ORBIT_RADIUS_METERS,
  MAXIMUM_PITCH_DEGREES,
  MINIMUM_ORBIT_RADIUS_METERS,
  TARGET_HEIGHT_METERS,
  Vector3,
  cameraBasis,
  cameraPoseAt,
  lookAt,
  moveInView,
  orbitBy,
  orbitCenter,
  scaleOrbitRadius,
  viewHalfHeightMeters,
  withTransform,
} from '../src/domain/preview/camera';

const TARGET: Vector3 = [0.01, 0.29, 0.02];

function expectVectorClose(actual: Vector3, expected: Vector3, digits = 10): void {
  for (let axis = 0; axis < 3; axis++) {
    expect(actual[axis]).toBeCloseTo(expected[axis], digits);
  }
}

/** ワールド行列（列優先 16 要素）。列は right / up / back / 位置。 */
function worldMatrix(pose: CameraPose): number[] {
  const { right, up, back } = cameraBasis(pose);
  return [
    right[0], right[1], right[2], 0,
    up[0], up[1], up[2], 0,
    back[0], back[1], back[2], 0,
    pose.position[0], pose.position[1], pose.position[2], 1,
  ];
}

describe('Unity 側から写した既定値', () => {
  it('既定の姿勢は注視点の正面 1.3m・画角 20°', () => {
    expect(DEFAULT_FOV_DEGREES).toBe(20);
    expect(DEFAULT_ORBIT_RADIUS_METERS).toBeCloseTo(1.3, 10);
    expect(TARGET_HEIGHT_METERS).toBeCloseTo(0.297, 10);
    const pose = cameraPoseAt([0, TARGET_HEIGHT_METERS, 0]);
    expectVectorClose(pose.position, [0, 0.297, 1.3]);
    expect(pose.pitchDegrees).toBe(0);
    expect(pose.yawDegrees).toBe(0);
  });

  it('既定の姿勢は注視点を見ている（-Z を見るカメラが被写体の正面に立つ）', () => {
    const pose = cameraPoseAt([0, TARGET_HEIGHT_METERS, 0]);
    expectVectorClose(orbitCenter(pose), [0, TARGET_HEIGHT_METERS, 0]);
    expectVectorClose(cameraBasis(pose).back, [0, 0, 1]);
  });
});

describe('周回（左ドラッグ）', () => {
  it('中心と半径が変わらない', () => {
    const start = cameraPoseAt(TARGET, 0.9);
    const turned = orbitBy(orbitBy(start, 37, -21), -12, 8);
    expectVectorClose(orbitCenter(turned), TARGET);
    expect(turned.orbitRadiusMeters).toBeCloseTo(0.9, 10);
    expect(turned.yawDegrees).toBeCloseTo(25, 10);
    expect(turned.pitchDegrees).toBeCloseTo(-13, 10);
  });

  it('被写体からの距離が変わらない（位置は中心から半径の球の上）', () => {
    const start = cameraPoseAt(TARGET, 0.9);
    const turned = orbitBy(start, 120, 40);
    const distance = Math.hypot(
      turned.position[0] - TARGET[0],
      turned.position[1] - TARGET[1],
      turned.position[2] - TARGET[2],
    );
    expect(distance).toBeCloseTo(0.9, 10);
  });

  it('pitch は上限で止まり、yaw は畳まれる（真上を越えない・数が発散しない）', () => {
    const start = cameraPoseAt(TARGET);
    expect(orbitBy(start, 0, 200).pitchDegrees).toBe(MAXIMUM_PITCH_DEGREES);
    expect(orbitBy(start, 0, -200).pitchDegrees).toBe(-MAXIMUM_PITCH_DEGREES);
    expect(orbitBy(start, 540, 0).yawDegrees).toBeCloseTo(-180, 10);
    // 止まった後も中心は動かない。
    expectVectorClose(orbitCenter(orbitBy(start, 0, 200)), TARGET);
  });
});

describe('寄り引き（ホイール）', () => {
  it('半径だけが変わり、向きも中心も変わらない', () => {
    const start = orbitBy(cameraPoseAt(TARGET, 1.2), 33, 17);
    const zoomed = scaleOrbitRadius(start, 0.5);
    expect(zoomed.orbitRadiusMeters).toBeCloseTo(0.6, 10);
    expect(zoomed.pitchDegrees).toBeCloseTo(start.pitchDegrees, 10);
    expect(zoomed.yawDegrees).toBeCloseTo(start.yawDegrees, 10);
    expectVectorClose(orbitCenter(zoomed), orbitCenter(start));
  });

  it('掛け算なので往復すると元へ戻る', () => {
    const start = orbitBy(cameraPoseAt(TARGET, 1.2), 33, 17);
    const round = scaleOrbitRadius(scaleOrbitRadius(start, 2.5), 1 / 2.5);
    expectVectorClose(round.position, start.position);
    expect(round.orbitRadiusMeters).toBeCloseTo(start.orbitRadiusMeters, 10);
  });

  it('半径は範囲でクランプされる（旧実装の 距離 × 拡大率 で届いた範囲）', () => {
    const start = cameraPoseAt(TARGET);
    expect(scaleOrbitRadius(start, 1e-6).orbitRadiusMeters).toBe(MINIMUM_ORBIT_RADIUS_METERS);
    expect(scaleOrbitRadius(start, 1e6).orbitRadiusMeters).toBe(MAXIMUM_ORBIT_RADIUS_METERS);
  });
});

describe('平行移動（右ドラッグ）', () => {
  it('向きと半径が変わらず、中心が位置と同じだけ動く', () => {
    const start = orbitBy(cameraPoseAt(TARGET, 1.1), 25, -15);
    const moved = moveInView(start, 0.13, -0.07);
    expect(moved.pitchDegrees).toBe(start.pitchDegrees);
    expect(moved.yawDegrees).toBe(start.yawDegrees);
    expect(moved.orbitRadiusMeters).toBe(start.orbitRadiusMeters);
    const centerShift: Vector3 = [
      orbitCenter(moved)[0] - orbitCenter(start)[0],
      orbitCenter(moved)[1] - orbitCenter(start)[1],
      orbitCenter(moved)[2] - orbitCenter(start)[2],
    ];
    const positionShift: Vector3 = [
      moved.position[0] - start.position[0],
      moved.position[1] - start.position[1],
      moved.position[2] - start.position[2],
    ];
    expectVectorClose(centerShift, positionShift);
  });

  it('正面から見ているときは right / up がワールドの X / Y そのもの', () => {
    const start = cameraPoseAt(TARGET, 1.1);
    const moved = moveInView(start, 0.2, 0.3);
    expectVectorClose(moved.position, [TARGET[0] + 0.2, TARGET[1] + 0.3, TARGET[2] + 1.1]);
  });
});

describe('注視点（頭部中心）を見る', () => {
  it('その位置から注視点への向きに一致し、半径も距離に合う', () => {
    const strayed = withTransform(cameraPoseAt(TARGET), [0.5, 0.9, -0.7], 40, -150);
    const aimed = lookAt(strayed, TARGET);
    expectVectorClose(aimed.position, [0.5, 0.9, -0.7]);
    const distance = Math.hypot(0.5 - TARGET[0], 0.9 - TARGET[1], -0.7 - TARGET[2]);
    expect(aimed.orbitRadiusMeters).toBeCloseTo(distance, 10);
    // 視線は -back。注視点への単位ベクトルと一致する。
    const { back } = cameraBasis(aimed);
    expectVectorClose(
      [-back[0], -back[1], -back[2]],
      [
        (TARGET[0] - 0.5) / distance,
        (TARGET[1] - 0.9) / distance,
        (TARGET[2] + 0.7) / distance,
      ],
    );
    // 向いた後は中心が注視点そのもの（次に回すと頭のまわりを回る）。
    expectVectorClose(orbitCenter(aimed), TARGET);
  });

  it('注視点の上に立っていると向きが決まらないので何もしない', () => {
    const onTop = withTransform(cameraPoseAt(TARGET), TARGET, 30, 60);
    expect(lookAt(onTop, TARGET)).toEqual(onTop);
  });
});

describe('位置と回転の入力', () => {
  it('打った値がそのまま入り、中心は導出で付いてくる', () => {
    const typed = withTransform(cameraPoseAt(TARGET, 1.3), [0.2, 0.4, 1.0], 0, 0);
    expectVectorClose(typed.position, [0.2, 0.4, 1.0]);
    expectVectorClose(orbitCenter(typed), [0.2, 0.4, 1.0 - 1.3]);
  });

  it('可動域を越えた入力はクランプ・yaw は畳まれる', () => {
    const typed = withTransform(cameraPoseAt(TARGET), [0, 0, 0], 120, 200);
    expect(typed.pitchDegrees).toBe(MAXIMUM_PITCH_DEGREES);
    expect(typed.yawDegrees).toBeCloseTo(-160, 10);
  });
});

// 移植で絵が変わっていないことの担保。旧実装は「注視点 → pan → 位置 → lookAt」で向きを作って
// いたが、pan が注視点と位置を同じだけ動かすので、向きは常に `Euler(pitch, yaw, 0, 'YXZ')` と
// 厳密に一致していた。
describe('旧実装（Euler + lookAt）との一致', () => {
  const cases = [
    { yaw: 0, pitch: 0, distance: 1.3, zoom: 1, pan: [0, 0] },
    { yaw: 37, pitch: -21, distance: 1.3, zoom: 1.7, pan: [0.3, -0.2] },
    { yaw: -128, pitch: 64, distance: 0.8, zoom: 0.6, pan: [-0.45, 0.35] },
  ] as const;
  const fovDegrees = 20;
  const aspect = 16 / 9;

  function legacyCamera(input: (typeof cases)[number]): THREE.PerspectiveCamera {
    const camera = new THREE.PerspectiveCamera(fovDegrees, aspect, 0.01, 20);
    const distance = input.distance / input.zoom;
    const rotation = new THREE.Euler(
      (input.pitch * Math.PI) / 180,
      (input.yaw * Math.PI) / 180,
      0,
      'YXZ',
    );
    const right = new THREE.Vector3(1, 0, 0).applyEuler(rotation);
    const up = new THREE.Vector3(0, 1, 0).applyEuler(rotation);
    const forward = new THREE.Vector3(0, 0, 1).applyEuler(rotation);
    const halfHeight = distance * Math.tan((fovDegrees * Math.PI) / 360);
    const target = new THREE.Vector3(...TARGET);
    target.addScaledVector(right, -input.pan[0] * halfHeight * aspect);
    target.addScaledVector(up, -input.pan[1] * halfHeight);
    camera.position.copy(target).addScaledVector(forward, distance);
    camera.up.copy(up);
    camera.lookAt(target);
    camera.updateMatrixWorld();
    return camera;
  }

  function ported(input: (typeof cases)[number]): CameraPose {
    const turned = orbitBy(cameraPoseAt(TARGET, input.distance / input.zoom), input.yaw, input.pitch);
    const halfHeight = viewHalfHeightMeters(turned, fovDegrees);
    return moveInView(
      turned,
      -input.pan[0] * halfHeight * aspect,
      -input.pan[1] * halfHeight,
    );
  }

  it('ワールド行列が一致する（= view 行列も一致する = 同じ画）', () => {
    for (const input of cases) {
      const legacy = legacyCamera(input).matrixWorld.elements;
      const actual = worldMatrix(ported(input));
      for (let index = 0; index < 16; index++) {
        expect(actual[index]).toBeCloseTo(legacy[index], 12);
      }
    }
  });

  it('基底は three.js の Euler をそのまま当てたものと同じ', () => {
    for (const input of cases) {
      const pose = ported(input);
      const rotation = new THREE.Euler(
        (pose.pitchDegrees * Math.PI) / 180,
        (pose.yawDegrees * Math.PI) / 180,
        0,
        'YXZ',
      );
      const basis = cameraBasis(pose);
      for (const [actual, axis] of [
        [basis.right, new THREE.Vector3(1, 0, 0)],
        [basis.up, new THREE.Vector3(0, 1, 0)],
        [basis.back, new THREE.Vector3(0, 0, 1)],
      ] as [Vector3, THREE.Vector3][]) {
        const expected = axis.applyEuler(rotation);
        expectVectorClose(actual, [expected.x, expected.y, expected.z], 12);
      }
    }
  });
});
