// GNM アセットの型。
//
// GNM 公式アセットのうち **Exporter が実際に使う要素** — identity 基底・UV・三角形・
// 部位・68 点ランドマークの barycentric 定義 — を保持する型を置く。
//
// 表情基底・ジョイント・スキニング重みは持たない。フィットもアトラスも髪も読まないので
// 持てば死荷重になる。消費側が表情やスキニングを要るときは公式アセットから直接取れる。
//
// 読み込み（GNMB コンテナ）は infrastructure の責務。ここは読み込み済みの配列を受け取る
// 型定義と評価だけを持つ。
//
// ここで評価した頂点は出力しない。フィットの計算と検査画像の生成にだけ使う。
//
// 座標系は GNM 空間のまま扱う: 右手系 / X=解剖学的左 / Y=上 / Z=前 / 単位はメートル。
// 左手系への変換は消費側（Unity）の責務なので、ここでは符号も三角形の巻き順も触らない。

/** ランドマークの点数。iBUG-68 規約。 */
export const IBUG68_POINT_COUNT = 68;

/**
 * identity 基底（int16 量子化）。値 = `quantized * scales[component] / 32767`。
 *
 * **量子化は web だから増えた差分。** デスクトップ側は公式 npz の float32 をそのまま
 * 持つ（ローカルのファイルを読むので送る必要がない）。ブラウザは 56MB を毎回落とすので
 * 28MB へ落とす。フィットへの影響はあちらの実測（残差の 5 桁目・誤差 146nm）が根拠で、
 * 実際の最大誤差は `tools/export_gnm_assets.py` が生成のたびに表示する。
 */
export interface QuantizedIdentityBasis {
  /** (成分数, 頂点数, 3) を平坦化したもの。 */
  readonly quantized: Int16Array;
  /** (成分数,) 成分ごとの量子化スケール。 */
  readonly scales: Float64Array;
  readonly componentCount: number;
  readonly vertexCount: number;
}

/** 量子化された基底の 1 要素を実数へ戻す。 */
export function basisValue(
  basis: QuantizedIdentityBasis,
  component: number,
  vertex: number,
  axis: number,
): number {
  const offset = (component * basis.vertexCount + vertex) * 3 + axis;
  return (basis.quantized[offset] * basis.scales[component]) / 32767;
}

/**
 * 頭部メッシュ。頂点は per-vertex UV へ分割済み（split 空間）。
 *
 * GNM の UV は face-varying（同じ頂点が面ごとに別の UV を持つ）ため、GPU が扱える
 * per-vertex UV にするには UV の切れ目で頂点を複製する必要がある。この型の頂点数は
 * 複製後の数で、`uvSplitSource` が複製前の index への写像を持つ。
 */
export interface GnmHeadMesh {
  /** (頂点数, 3) 平均形状の頂点位置。 */
  readonly templateVertexPositions: Float32Array;
  /** (頂点数, 2) 公式 UV。 */
  readonly vertexUvs: Float32Array;
  /** (三角形数, 3) split 空間の index。 */
  readonly triangles: Uint32Array;
  /** (頂点数,) split 後 → 複製前の頂点 index。**単調非減少**（`splitIndexOf` が依存）。 */
  readonly uvSplitSource: Uint32Array;
  /** (頂点数,) `componentNames` への index。**分割**（どの頂点もちょうど1つに属する）。 */
  readonly componentId: Uint8Array;
  /** メッシュ構成要素の名前（`componentId` の並び順）。 */
  readonly componentNames: readonly string[];
  /** (頂点数,) 公式GNMの `ears` 頂点グループ。耳の輪郭フィットとアトラス補完に使う。 */
  readonly earRegion: Uint8Array;
  /** (頂点数,) 青い混合投影を補完色100%へ置き換える領域（chin より下の首・胴体）。 */
  readonly atlasPhotoOnlyRegion: Uint8Array;
  /** (頂点数,) 開口部の縁（唇のインナーロール）の重み。0 / 0.5 / 1.0 を取る。 */
  readonly mouthRimRegion: Float32Array;
  readonly vertexCount: number;
  readonly triangleCount: number;
}

/**
 * split 前の頂点数（= 公式アセットの頂点数）。
 *
 * `uvSplitSource` は単調非減少で、公式の頂点はすべて 1 回以上現れるため、末尾の値 + 1 が
 * そのまま個数になる（現れることは読み込み時に検証する）。
 */
export function unsplitVertexCount(mesh: GnmHeadMesh): number {
  return mesh.uvSplitSource[mesh.vertexCount - 1] + 1;
}

/**
 * split 前の頂点 index を split 後の index に写す。
 *
 * split は UV だけを分けるので、同じ元頂点から出た複製は位置・基底・重みがすべて一致
 * する。よって「どれか1つ」を返せば十分で、ここは先頭を返す。`uvSplitSource` は単調
 * 非減少なので二分探索で引ける。写せたことは戻り値で引き直して検証する。
 */
export function splitIndexOf(mesh: GnmHeadMesh, unsplitIndex: number): number {
  const source = mesh.uvSplitSource;
  let low = 0;
  let high = source.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (source[middle] < unsplitIndex) low = middle + 1;
    else high = middle;
  }
  if (low >= source.length || source[low] !== unsplitIndex) {
    throw new Error(
      `uv_split_source から split 前 index ${unsplitIndex} を引き直せなかった` +
        '（単調非減少でなくなっているか、参照されていない元頂点がある）',
    );
  }
  return low;
}

/**
 * iBUG-68 の 68 点を頭部メッシュ表面の barycentric として定義したもの。
 *
 * 68 点は頂点そのものではなく三角形の内部にあるため、`(頂点, 重み) × 3` で表す。
 * index は **split 前の index 空間**（`splitIndexOf` で写す）。
 */
export interface SparseLandmarks68 {
  /** (68, 3) split 前の頂点 index。 */
  readonly vertexIndices: Int32Array;
  /** (68, 3) 行和は 1.0。 */
  readonly weights: Float32Array;
}

/**
 * MediaPipe の顔メッシュ点を頭部メッシュ表面の barycentric として定義したもの。
 *
 * `SparseLandmarks68` と同じ形だが、点の出どころが違う。68 点は GNM アセットが定義を
 * 持っているが、MediaPipe の顔メッシュ 468 点はどちらの公式にも対応が無いので**計算で
 * 作る**（`tools/export_gnm_assets.py` がビルド時に作り、アセットへ焼き込む）。
 *
 * 観測を増やすために持つ。68 点では観測 136 個に対し identity が 253 成分あり、未知数が
 * 観測を超えて係数が一意に決まらない。
 */
export interface DenseLandmarks {
  /** (M,) MediaPipe の顔メッシュ点 index（0..467）。 */
  readonly mediapipeIndices: Int32Array;
  /** (M, 3) split 前の頂点 index。 */
  readonly vertexIndices: Int32Array;
  /** (M, 3) 行和は 1.0。 */
  readonly weights: Float32Array;
  /** (M,) 表面までの距離。**この対応がどれだけ確かか**。 */
  readonly residualMeters: Float32Array;
  /** 投影先メッシュの辺の長さの中央値。残差を無次元化する基準。 */
  readonly edgeMeters: number;
  readonly pointCount: number;
}

/** 点ごとの重み `1 / (1 + 残差 / (辺 × halfEdges))`（残差 0 で 1）。 */
export function denseConfidence(dense: DenseLandmarks, halfEdges: number): Float64Array {
  const half = halfEdges * dense.edgeMeters;
  if (!(half > 0)) throw new Error(`重みの基準長が ${half}`);
  const out = new Float64Array(dense.pointCount);
  for (let point = 0; point < dense.pointCount; point++) {
    out[point] = 1 / (1 + dense.residualMeters[point] / half);
  }
  return out;
}

/** 残差の中央値を顔幅比で返す。**フィットが仮定するずれの実測値**。 */
export function denseMedianResidualRatio(dense: DenseLandmarks, faceWidthMeters: number): number {
  if (!(faceWidthMeters > 0)) throw new Error(`顔幅が ${faceWidthMeters}`);
  const sorted = Array.from(dense.residualMeters).sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  const median =
    sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  return median / faceWidthMeters;
}

/** 公式 GNMB から読み込んだ GNM 頭部アセット一式。 */
export interface GnmHeadAsset {
  /** 出自を人が読める形で記した文字列。機械照合には使わない。 */
  readonly source: string;
  /** GNM アセットのバージョン（例 "3.0"）。照合はこちらで行う。 */
  readonly gnmVersion: string;
  /** GNM アセットの variant（例 "head"）。 */
  readonly gnmVariant: string;
  readonly mesh: GnmHeadMesh;
  readonly vertexIdentityBasis: QuantizedIdentityBasis;
  readonly landmarks: SparseLandmarks68;
  readonly dense: DenseLandmarks;
}

export function identityComponentCount(asset: GnmHeadAsset): number {
  return asset.vertexIdentityBasis.componentCount;
}

/**
 * identity 係数から頭部の頂点 (V, 3) を作る（GNM 空間・メートル）。
 *
 * ここで評価した頂点は出力しない。アトラスのベイクと髪シェルのアンカーに使うだけで、
 * Unity 側は同じ係数から自分で再構成する。
 *
 *     頂点 = template_vertex_positions + Σ identity[i] * vertex_identity_basis[i]
 */
export function verticesOf(asset: GnmHeadAsset, identity: Float64Array): Float64Array {
  const basis = asset.vertexIdentityBasis;
  if (identity.length !== basis.componentCount) {
    throw new Error(
      `identity の長さが ${identity.length}（期待 ${basis.componentCount}）`,
    );
  }
  const vertexCount = basis.vertexCount;
  const out = new Float64Array(vertexCount * 3);
  const template = asset.mesh.templateVertexPositions;
  for (let index = 0; index < out.length; index++) out[index] = template[index];

  const quantized = basis.quantized;
  for (let component = 0; component < basis.componentCount; component++) {
    const coefficient = identity[component];
    if (coefficient === 0) continue;
    const factor = (coefficient * basis.scales[component]) / 32767;
    const base = component * vertexCount * 3;
    for (let index = 0; index < out.length; index++) {
      out[index] += quantized[base + index] * factor;
    }
  }
  return out;
}

/**
 * barycentric 定義で頂点位置を補間する（(P, 3) の float64）。
 *
 * index は **split 前の index 空間**で受け取り、`splitIndexOf` で写す。
 */
export function sampleBarycentricPositions(
  mesh: GnmHeadMesh,
  values: Float32Array | Float64Array,
  vertexIndices: Int32Array,
  weights: Float32Array,
): Float64Array {
  const pointCount = vertexIndices.length / 3;
  const out = new Float64Array(pointCount * 3);
  for (let point = 0; point < pointCount; point++) {
    for (let corner = 0; corner < 3; corner++) {
      const vertex = splitIndexOf(mesh, vertexIndices[point * 3 + corner]);
      const weight = weights[point * 3 + corner];
      for (let axis = 0; axis < 3; axis++) {
        out[point * 3 + axis] += values[vertex * 3 + axis] * weight;
      }
    }
  }
  return out;
}

/**
 * barycentric 定義で identity 基底を補間する（(K, P, 3) を平坦化した float64）。
 *
 * 対応点は数百なので、ここだけは実数へ展開して持つ（フィットの設計行列がこれを何度も
 * 読むので、量子化のままだと毎回の割り算が効く）。
 */
export function sampleBarycentricBasis(
  mesh: GnmHeadMesh,
  basis: QuantizedIdentityBasis,
  vertexIndices: Int32Array,
  weights: Float32Array,
): Float64Array {
  const pointCount = vertexIndices.length / 3;
  const out = new Float64Array(basis.componentCount * pointCount * 3);
  const corners = new Int32Array(pointCount * 3);
  for (let index = 0; index < corners.length; index++) {
    corners[index] = splitIndexOf(mesh, vertexIndices[index]);
  }
  for (let component = 0; component < basis.componentCount; component++) {
    const factor = basis.scales[component] / 32767;
    const basisBase = component * basis.vertexCount * 3;
    const outBase = component * pointCount * 3;
    for (let point = 0; point < pointCount; point++) {
      for (let corner = 0; corner < 3; corner++) {
        const vertex = corners[point * 3 + corner];
        const weight = weights[point * 3 + corner];
        for (let axis = 0; axis < 3; axis++) {
          out[outBase + point * 3 + axis] +=
            basis.quantized[basisBase + vertex * 3 + axis] * factor * weight;
        }
      }
    }
  }
  return out;
}

/** 指定した頂点だけの identity 基底 (K, N, 3)（シルエットフィットのアンカー用）。 */
export function gatherBasisAtVertices(
  basis: QuantizedIdentityBasis,
  vertices: Int32Array,
): Float64Array {
  const out = new Float64Array(basis.componentCount * vertices.length * 3);
  for (let component = 0; component < basis.componentCount; component++) {
    const factor = basis.scales[component] / 32767;
    const basisBase = component * basis.vertexCount * 3;
    const outBase = component * vertices.length * 3;
    for (let slot = 0; slot < vertices.length; slot++) {
      const vertex = vertices[slot];
      for (let axis = 0; axis < 3; axis++) {
        out[outBase + slot * 3 + axis] = basis.quantized[basisBase + vertex * 3 + axis] * factor;
      }
    }
  }
  return out;
}
