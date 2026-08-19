// FACE ONLY: MediaPipe顔ランドマークをそのまま頂点とするメッシュ。
// 顔輪郭(Delaunay hullで構成される範囲)の外は生成しない。耳・髪・首・背景は含めない。

import * as THREE from 'three';
import { FACE_KEY_INDICES, type FaceTriangulation, type NormalizedFaceLandmark } from './faceTopology';
import { buildCanonicalFaceDepth, computeFinalFaceDepthPerVertex } from './faceDepth';
import type { Params } from './params';

export interface FaceOnlyBuild {
  group: THREE.Group; // yaw pivot用のGroup (pivotZだけ後方にずらす)
  mesh: THREE.Mesh;
  geometry: THREE.BufferGeometry;
  landmarks: NormalizedFaceLandmark[];
  canonicalDepth: Float32Array;
  basePositions: Float32Array; // blink/talk前のベース位置 (アニメーションの基準)
  deformIndices: number[]; // blink/talkの影響を受けるlandmark index一覧
}

/** blink/talkの影響を受けうるlandmark indexの集合 (アニメーションで毎フレーム書き換える範囲を限定する)。 */
function collectDeformIndices(): number[] {
  const k = FACE_KEY_INDICES;
  const set = new Set<number>([
    k.eyeA.upper1,
    k.eyeA.upper2,
    k.eyeA.lower1,
    k.eyeA.lower2,
    k.eyeB.upper1,
    k.eyeB.upper2,
    k.eyeB.lower1,
    k.eyeB.lower2,
    k.mouth.upperCenter,
    k.mouth.upperOuter,
    k.mouth.lowerCenter,
    k.mouth.lowerOuter,
    k.mouth.cornerA,
    k.mouth.cornerB,
    k.chin,
  ]);
  return Array.from(set);
}

export function buildFaceOnlyMesh(
  landmarks: NormalizedFaceLandmark[],
  triangulation: FaceTriangulation,
  texture: THREE.Texture,
  params: Params,
): FaceOnlyBuild {
  const canonicalDepth = buildCanonicalFaceDepth(landmarks);
  const zFinal = computeFinalFaceDepthPerVertex(landmarks, canonicalDepth, params.canonicalMix, params.faceDepthScale);

  const count = landmarks.length;
  const positions = new Float32Array(count * 3);
  const uvs = new Float32Array(count * 2);
  for (let i = 0; i < count; i++) {
    const lm = landmarks[i];
    positions[i * 3 + 0] = lm.x;
    positions[i * 3 + 1] = lm.y;
    positions[i * 3 + 2] = zFinal[i];
    uvs[i * 2 + 0] = lm.u;
    uvs[i * 2 + 1] = lm.v;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geometry.setIndex(new THREE.BufferAttribute(triangulation.triangles, 1));
  geometry.computeVertexNormals();

  const material = new THREE.MeshStandardMaterial({
    map: texture,
    roughness: 0.95,
    metalness: 0.0,
    side: THREE.DoubleSide,
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.z = -params.pivotZRatio;

  const group = new THREE.Group();
  group.position.z = params.pivotZRatio;
  group.add(mesh);

  return {
    group,
    mesh,
    geometry,
    landmarks,
    canonicalDepth,
    basePositions: positions.slice(),
    deformIndices: collectDeformIndices(),
  };
}

/** GUIパラメータ変更時にDepthのみ再計算する (再検出は行わない)。 */
export function recomputeFaceOnlyDepth(build: FaceOnlyBuild, params: Params): void {
  const zFinal = computeFinalFaceDepthPerVertex(
    build.landmarks,
    build.canonicalDepth,
    params.canonicalMix,
    params.faceDepthScale,
  );
  const posAttr = build.geometry.getAttribute('position') as THREE.BufferAttribute;
  for (let i = 0; i < build.landmarks.length; i++) {
    build.basePositions[i * 3 + 2] = zFinal[i];
    posAttr.setZ(i, zFinal[i]);
  }
  posAttr.needsUpdate = true;
  build.geometry.computeVertexNormals();
  build.mesh.position.z = -params.pivotZRatio;
  build.group.position.z = params.pivotZRatio;
}
