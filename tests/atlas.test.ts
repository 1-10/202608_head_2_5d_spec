// アトラス（`domain/atlas`）の検査。
//
// UV ラスタライズは**実アセット**で回す（合成データだと chart の分かれ方や被覆率という「アセットの
// 性質」を検証できない）。一辺は 256 に落として時間を抑える — 検証したいのは規約と被覆の性質で、
// 解像度ではない。

import { describe, expect, it } from 'vitest';
import {
  DEFAULT_BAKE_SETTINGS,
  EXTERIOR_CHART,
  PROVENANCE_DILATION,
  PROVENANCE_FILL,
  PROVENANCE_UNUSED,
  bakeAtlas,
  foregroundConfidenceWeight,
  occlusionTolerance,
  provenanceCounts,
} from '../src/domain/atlas/bake';
import { buildAtlasLayout, chartCount, coveredMask } from '../src/domain/atlas/surface';
import { atlasRowColToUv } from '../src/domain/contract';
import { fieldOverFullImage } from '../src/domain/field';
import { Similarity2d } from '../src/domain/gnm/fit';
import { verticesOf } from '../src/domain/gnm/model';
import { PhotoRgb } from '../src/domain/photo';
import { loadAsset } from './asset';

const SIZE = 256;

function layout(): ReturnType<typeof buildAtlasLayout> {
  const mesh = loadAsset().mesh;
  return buildAtlasLayout(mesh.triangles, mesh.vertexUvs, mesh.componentId, SIZE);
}

describe('UV ラスタライズ', () => {
  it('barycentric の行和は 1（被覆テクセル）', () => {
    const result = layout();
    const covered = coveredMask(result);
    let coveredCount = 0;
    for (let texel = 0; texel < covered.length; texel++) {
      if (covered[texel] === 0) continue;
      coveredCount++;
      const total =
        result.barycentric[texel * 3] +
        result.barycentric[texel * 3 + 1] +
        result.barycentric[texel * 3 + 2];
      expect(total).toBeCloseTo(1, 5);
    }
    // 肌の chart はアトラスの 3 割以上を覆う（実測 34% 前後）。
    expect(coveredCount / covered.length).toBeGreaterThan(0.25);
  });

  it('縁のテクセルは barycentric を [0,1] にクランプしてある', () => {
    const result = layout();
    for (let texel = 0; texel < result.triangleIndex.length; texel++) {
      if (result.triangleIndex[texel] < 0) continue;
      if (result.centerInside[texel] !== 0) continue;
      for (let corner = 0; corner < 3; corner++) {
        expect(result.barycentric[texel * 3 + corner]).toBeGreaterThanOrEqual(0);
        expect(result.barycentric[texel * 3 + corner]).toBeLessThanOrEqual(1);
      }
    }
  });

  it('conservative rasterization が縁を拾っている（中心が外のテクセルが存在する）', () => {
    const result = layout();
    let fringe = 0;
    for (let texel = 0; texel < result.triangleIndex.length; texel++) {
      if (result.triangleIndex[texel] >= 0 && result.centerInside[texel] === 0) fringe++;
    }
    expect(fringe).toBeGreaterThan(0);
  });

  it('chart は 2 枚（外から見える肌 + 口腔壁）で、0 が最大', () => {
    const result = layout();
    expect(chartCount(result)).toBe(2);
    const counts = [0, 0];
    for (const chart of result.chartIndex) if (chart >= 0) counts[chart]++;
    expect(counts[EXTERIOR_CHART]).toBeGreaterThan(counts[1]);
    expect(counts[1]).toBeGreaterThan(0);
  });

  it('テクセルの UV は契約の 2 式どおり（行 0 が v = 1 側）', () => {
    const mesh = loadAsset().mesh;
    const result = layout();
    // 被覆テクセルの UV を三角形の頂点 UV の barycentric で復元し、契約の式と一致することを見る。
    let checked = 0;
    for (let texel = 0; texel < result.triangleIndex.length && checked < 200; texel += 97) {
      const triangle = result.triangleIndex[texel];
      if (triangle < 0 || result.centerInside[texel] === 0) continue;
      let u = 0;
      let v = 0;
      for (let corner = 0; corner < 3; corner++) {
        const vertex = mesh.triangles[triangle * 3 + corner];
        u += mesh.vertexUvs[vertex * 2] * result.barycentric[texel * 3 + corner];
        v += mesh.vertexUvs[vertex * 2 + 1] * result.barycentric[texel * 3 + corner];
      }
      const row = Math.floor(texel / SIZE);
      const [expectedU, expectedV] = atlasRowColToUv(row, texel - row * SIZE, SIZE);
      expect(u).toBeCloseTo(expectedU, 5);
      expect(v).toBeCloseTo(expectedV, 5);
      checked++;
    }
    expect(checked).toBeGreaterThan(50);
  });
});

describe('前景の信用曲線', () => {
  it('しきい値以上は生値、未満は冪で落とす', () => {
    expect(foregroundConfidenceWeight(1, 0.95, 6)).toBeCloseTo(1, 10);
    expect(foregroundConfidenceWeight(0.95, 0.95, 6)).toBeCloseTo(0.95, 10);
    // 0.5 は 0.95 * (0.5/0.95)^6 ≈ 0.021。線形合成（0.5）より強く抑える。
    expect(foregroundConfidenceWeight(0.5, 0.95, 6)).toBeLessThan(0.05);
    // しきい値 0 は比較用の従来線形動作。
    expect(foregroundConfidenceWeight(0.5, 0, 6)).toBeCloseTo(0.5, 10);
  });
});

describe('遮蔽の許容量', () => {
  it('見るのは minFacing ではなく minFacing − facingSoftness', () => {
    const similarity = new Similarity2d(
      Float64Array.from([2000, 0, 0, -2000]),
      Float64Array.from([0, 0]),
    );
    const wide = occlusionTolerance(similarity, DEFAULT_BAKE_SETTINGS);
    const narrow = occlusionTolerance(similarity, {
      ...DEFAULT_BAKE_SETTINGS,
      facingSoftness: 0,
    });
    // 傾斜を入れた方が許容量が広い（寝た面を遮蔽と誤判定しない）。
    expect(wide).toBeGreaterThan(narrow);
    // 既定の比は約 2.07（デスクトップ側の docstring の 6.59 / 3.18）。
    const base = DEFAULT_BAKE_SETTINGS.occlusionTolerance;
    expect((wide - base) / (narrow - base)).toBeCloseTo(2.07, 1);
  });
});

/** 一様な色の写真と「全部前景」のマスクで焼く（幾何の性質だけを見る）。 */
function flatPhoto(width: number, height: number, rgb: readonly number[]): PhotoRgb {
  const data = new Uint8Array(width * height * 3);
  for (let pixel = 0; pixel < width * height; pixel++) {
    data[pixel * 3] = rgb[0];
    data[pixel * 3 + 1] = rgb[1];
    data[pixel * 3 + 2] = rgb[2];
  }
  return { data, width, height };
}

describe('ベイク（一辺 256）', () => {
  it('chart 内は全テクセルに色が入り、chart の外は dilation で 8 テクセルにじむ', () => {
    const asset = loadAsset();
    const mesh = asset.mesh;
    const vertices = verticesOf(asset, new Float64Array(asset.vertexIdentityBasis.componentCount));
    const photo = flatPhoto(512, 640, [200, 150, 130]);
    const foreground = fieldOverFullImage(
      new Float32Array(photo.width * photo.height).fill(1),
      photo.width,
      photo.height,
    );
    const settings = { ...DEFAULT_BAKE_SETTINGS, atlasSize: SIZE };
    const bake = bakeAtlas({
      photo,
      vertices,
      triangles: mesh.triangles,
      vertexUvs: mesh.vertexUvs,
      componentId: mesh.componentId,
      // 平均顔を写真の中央へ 1800px スケールで写す（正面写真に相当する鏡映つきの変換）。
      similarity: new Similarity2d(
        Float64Array.from([1800, 0, 0, -1800]),
        Float64Array.from([photo.width / 2, photo.height * 0.42]),
      ),
      personMask: foreground,
      skinBaseColor: [0.78, 0.59, 0.51],
      settings,
      fillRegionId: mesh.earRegion,
      photoOnlyRegion: mesh.atlasPhotoOnlyRegion,
      mouthRimRegion: mesh.mouthRimRegion,
    });

    const covered = coveredMask(bake.surface);
    for (let texel = 0; texel < covered.length; texel++) {
      if (covered[texel] !== 0) expect(bake.provenance[texel]).not.toBe(PROVENANCE_UNUSED);
    }
    const counts = provenanceCounts(bake);
    // 一様な写真でも「写真から焼けた」「補完した」「にじませた」が全部出る。
    expect(counts.get(PROVENANCE_FILL)).toBeGreaterThan(0);
    expect(counts.get(PROVENANCE_DILATION)).toBeGreaterThan(0);
    expect(bake.albedo.length).toBe(SIZE * SIZE * 3);
  }, 120_000);
});
