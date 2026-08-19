// メッシュ共通ユーティリティ。

import * as THREE from 'three';

/**
 * 全頂点の法線を+Z(正面)に固定する。
 * 写真テクスチャのrelief表示では、Depth段差(生え際・輪郭)の斜面が
 * ジオメトリ陰影で「偽の影の帯」として見えてしまう。
 * 法線を+Zへ固定するとライティング応答が全頂点で一様になり、
 * 立体感はテクスチャ(写真の陰影)と視差だけで表現される。
 * FACE ONLY / FULL HEAD両方に適用し、比較の公平性を保つ。
 */
export function applyFlatNormals(geometry: THREE.BufferGeometry): void {
  const count = (geometry.getAttribute('position') as THREE.BufferAttribute).count;
  const normals = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    normals[i * 3 + 2] = 1;
  }
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
}
