// アトラスベイクのアダプタ。
//
// **品質判断は `domain/atlas` が持つ。** ここが担うのは外部資源だけ — 同じ GNM アセットと一辺なら
// identity に依らない `AtlasLayout` を再利用できるので、その使い回しをここで持つ。
//
// デスクトップ側はレイアウトを `%LOCALAPPDATA%` へ memory-map して永続化し、写真投影を CUDA カーネル
// で走らせる。**ブラウザには永続キャッシュもカーネルも無いので、セッション内の再利用だけを持つ**
// （パラメータを振って焼き直すときにラスタライズをやり直さない）。web だから縮む段。

import { AtlasBaker } from '../application/ports';
import {
  AtlasBake,
  BakeSettings,
  bakeAtlas,
  validateBakeSettings,
} from '../domain/atlas/bake';
import { AtlasLayout, bindAtlasSurface, buildAtlasLayout } from '../domain/atlas/surface';
import { ScalarField } from '../domain/field';
import { Similarity2d } from '../domain/gnm/fit';
import { PhotoRgb } from '../domain/photo';

/**
 * セッション内でレイアウトを使い回すベイカー。
 *
 * 鍵は「アセットのトポロジと一辺」。頂点 UV の配列そのものを鍵にするのは、同じアセットなら同じ
 * 参照が来るから（アセットは 1 回しか読まない）。
 */
export class CachingAtlasBaker implements AtlasBaker {
  private cachedLayout: AtlasLayout | null = null;
  private cachedKey: { vertexUvs: Float32Array; triangles: Uint32Array; size: number } | null = null;

  bake(input: {
    photo: PhotoRgb;
    vertices: Float64Array;
    triangles: Uint32Array;
    vertexUvs: Float32Array;
    componentId: Uint8Array;
    similarity: Similarity2d;
    personMask: ScalarField;
    skinBaseColor: readonly [number, number, number];
    settings: BakeSettings;
    fillRegionId: Uint8Array | null;
    photoOnlyRegion: Uint8Array | null;
    mouthRimRegion: Float32Array | null;
  }): AtlasBake {
    const settings = validateBakeSettings(input.settings);
    const layout = this.layoutFor(
      input.triangles,
      input.vertexUvs,
      input.componentId,
      settings.atlasSize,
    );
    const surface = bindAtlasSurface(layout, input.vertices, input.triangles);
    return bakeAtlas({ ...input, settings, surface });
  }

  private layoutFor(
    triangles: Uint32Array,
    vertexUvs: Float32Array,
    componentId: Uint8Array,
    size: number,
  ): AtlasLayout {
    if (
      this.cachedLayout !== null &&
      this.cachedKey !== null &&
      this.cachedKey.vertexUvs === vertexUvs &&
      this.cachedKey.triangles === triangles &&
      this.cachedKey.size === size
    ) {
      return this.cachedLayout;
    }
    const layout = buildAtlasLayout(triangles, vertexUvs, componentId, size);
    this.cachedLayout = layout;
    this.cachedKey = { vertexUvs, triangles, size };
    return layout;
  }
}
