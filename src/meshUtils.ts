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

/** cols×rowsの格子(行優先)を三角形2枚/セルでインデックス化する。 */
export function buildGridIndices(cols: number, rows: number): Uint32Array {
  const indices = new Uint32Array((cols - 1) * (rows - 1) * 6);
  let p = 0;
  for (let row = 0; row < rows - 1; row++) {
    for (let col = 0; col < cols - 1; col++) {
      const a = row * cols + col;
      const b = a + 1;
      const c = a + cols;
      const d = c + 1;
      indices[p++] = a;
      indices[p++] = c;
      indices[p++] = b;
      indices[p++] = b;
      indices[p++] = c;
      indices[p++] = d;
    }
  }
  return indices;
}
