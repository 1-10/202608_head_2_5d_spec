// アセットの不変条件を**実アセット**で押さえる。
//
// ここが落ちたら、`tools/export_gnm_assets.py` の出力かアセットのバージョンが変わっている。数値を
// テストに書き写しているのは、**変わったこと自体に気付いてから更新する**ため（黙って通ると、後段が
// 別のアセットを前提に動き続ける）。

import { describe, expect, it } from 'vitest';
import { GNMB_CONTENT_HEAD_ASSET, buildGnmbContainerBytes, readGnmbContainer } from '../src/infrastructure/gnmb';
import { EXPECTED_SPLIT_VERTEX_COUNT } from '../src/infrastructure/gnmAsset';
import {
  IBUG68_POINT_COUNT,
  splitIndexOf,
  unsplitVertexCount,
  verticesOf,
} from '../src/domain/gnm/model';
import { loadAsset } from './asset';

/** 公式 npz の頂点数（split 前）。 */
const UNSPLIT_VERTEX_COUNT = 17821;
/** 公式 npz の三角形数。 */
const TRIANGLE_COUNT = 35324;
/** v3_0 / head の identity 成分数。**絞らない**ので公式の全成分。 */
const IDENTITY_COMPONENT_COUNT = 253;

describe('GNMB コンテナ', () => {
  it('書いて読み戻せる', () => {
    const arrays = new Map([
      ['positions', { array: Float32Array.from([1, 2, 3]), shape: [1, 3] }],
      ['triangles', { array: Uint32Array.from([0, 1, 2]), shape: [1, 3] }],
    ]);
    const bytes = buildGnmbContainerBytes('hair_shell', arrays, { note: 'テスト' });
    const container = readGnmbContainer(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer,
      'hair_shell',
    );
    expect(container.header.content).toBe('hair_shell');
    expect(container.header['note']).toBe('テスト');
    expect(Array.from(container.arrays.get('positions') as Float32Array)).toEqual([1, 2, 3]);
    expect(Array.from(container.arrays.get('triangles') as Uint32Array)).toEqual([0, 1, 2]);
  });

  it('content が違えば落ちる', () => {
    const bytes = buildGnmbContainerBytes('hair_shell', new Map(), {});
    expect(() =>
      readGnmbContainer(
        bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer,
        GNMB_CONTENT_HEAD_ASSET,
      ),
    ).toThrow(/content/);
  });
});

describe('実アセット', () => {
  it('split 空間の頂点数・三角形数・成分数が想定どおり', () => {
    const asset = loadAsset();
    expect(asset.mesh.vertexCount).toBe(EXPECTED_SPLIT_VERTEX_COUNT);
    expect(unsplitVertexCount(asset.mesh)).toBe(UNSPLIT_VERTEX_COUNT);
    expect(asset.mesh.triangleCount).toBe(TRIANGLE_COUNT);
    expect(asset.vertexIdentityBasis.componentCount).toBe(IDENTITY_COMPONENT_COUNT);
    expect(asset.gnmVersion).toBe('3.0');
    expect(asset.gnmVariant).toBe('head');
  });

  it('uvSplitSource は単調非減少で、公式の頂点がすべて現れる', () => {
    const asset = loadAsset();
    const seen = new Uint8Array(UNSPLIT_VERTEX_COUNT);
    for (let vertex = 0; vertex < asset.mesh.vertexCount; vertex++) {
      if (vertex > 0) {
        expect(asset.mesh.uvSplitSource[vertex]).toBeGreaterThanOrEqual(
          asset.mesh.uvSplitSource[vertex - 1],
        );
      }
      seen[asset.mesh.uvSplitSource[vertex]] = 1;
    }
    expect(seen.every((value) => value === 1)).toBe(true);
  });

  it('splitIndexOf は split 前 index を引き直せる', () => {
    const asset = loadAsset();
    for (const unsplit of [0, 1, 1000, UNSPLIT_VERTEX_COUNT - 1]) {
      const split = splitIndexOf(asset.mesh, unsplit);
      expect(asset.mesh.uvSplitSource[split]).toBe(unsplit);
    }
  });

  it('構成要素は分割（どの頂点もちょうど1つに属する）', () => {
    const asset = loadAsset();
    expect(asset.mesh.componentNames).toEqual([
      'skin',
      'left_eye',
      'right_eye',
      'upper_teeth_and_gums',
      'lower_teeth_and_gums',
      'tongue',
    ]);
    for (const id of asset.mesh.componentId) {
      expect(id).toBeLessThan(asset.mesh.componentNames.length);
    }
    // 三角形は構成要素の境界を跨がない（跨ぐとアトラスの被覆が欠ける）。
    for (let triangle = 0; triangle < asset.mesh.triangleCount; triangle++) {
      const first = asset.mesh.componentId[asset.mesh.triangles[triangle * 3]];
      for (let corner = 1; corner < 3; corner++) {
        expect(asset.mesh.componentId[asset.mesh.triangles[triangle * 3 + corner]]).toBe(first);
      }
    }
  });

  it('68 点の barycentric の行和は 1', () => {
    const asset = loadAsset();
    expect(asset.landmarks.vertexIndices.length).toBe(IBUG68_POINT_COUNT * 3);
    for (let point = 0; point < IBUG68_POINT_COUNT; point++) {
      const total =
        asset.landmarks.weights[point * 3] +
        asset.landmarks.weights[point * 3 + 1] +
        asset.landmarks.weights[point * 3 + 2];
      expect(total).toBeCloseTo(1, 5);
    }
  });

  it('密対応は 468 点すべてに付いている', () => {
    const asset = loadAsset();
    expect(asset.dense.pointCount).toBe(468);
    expect(asset.dense.edgeMeters).toBeGreaterThan(0.002);
    expect(asset.dense.edgeMeters).toBeLessThan(0.005);
    for (let point = 0; point < asset.dense.pointCount; point++) {
      const total =
        asset.dense.weights[point * 3] +
        asset.dense.weights[point * 3 + 1] +
        asset.dense.weights[point * 3 + 2];
      expect(total).toBeCloseTo(1, 4);
    }
  });

  it('領域は空でない（口腔縁・写真専用・耳）', () => {
    const asset = loadAsset();
    const count = (values: Uint8Array | Float32Array): number =>
      Array.from(values).filter((value) => value > 0).length;
    expect(count(asset.mesh.mouthRimRegion)).toBeGreaterThan(100);
    expect(count(asset.mesh.atlasPhotoOnlyRegion)).toBeGreaterThan(1000);
    expect(count(asset.mesh.earRegion)).toBeGreaterThan(500);
  });

  it('identity = 0 の頂点は平均形状そのもの', () => {
    const asset = loadAsset();
    const vertices = verticesOf(asset, new Float64Array(IDENTITY_COMPONENT_COUNT));
    for (const index of [0, 100, 10_000, vertices.length - 1]) {
      expect(vertices[index]).toBeCloseTo(asset.mesh.templateVertexPositions[index], 10);
    }
  });

  it('identity を振ると頂点が動く（量子化で潰れていない）', () => {
    const asset = loadAsset();
    const identity = new Float64Array(IDENTITY_COMPONENT_COUNT);
    identity[0] = 3;
    const vertices = verticesOf(asset, identity);
    let maximum = 0;
    for (let index = 0; index < vertices.length; index++) {
      maximum = Math.max(
        maximum,
        Math.abs(vertices[index] - asset.mesh.templateVertexPositions[index]),
      );
    }
    // 第1成分を 3 振ると、少なくとも数ミリは動く。
    expect(maximum).toBeGreaterThan(0.003);
  });
});
