// 写真 → GNM 公式 UV アトラスのベイク（3 段）。
//
// 正面写真 1 枚では、アトラスの半分も写真の色で埋まらない（既定で chart 内の 3 割が写真その
// まま、1.5 割が写真と埋めの混合）。残りをどう埋めるかがこの段の本体で、由来を `provenance`
// としてテクセルごとに残す。**由来が分からないと、継ぎ目が写真の焼き付けの誤りなのか生成した
// 色なのかを人が判断できない。**
//
//     段1 写真      法線が前を向いていて投影先が写真内なら、DAViD人物前景確信度をしきい値
//                   未満で非線形に落とした**0..1 の合成重み**として置く
//     段2 埋め      重み 1 未満のテクセルを、確信度がしきい値以上の写真色だけメッシュ表面に
//                   沿って伝播させた場と混ぜる（重み 0 なら埋めた場そのもの）
//     段3 dilation  chart の外へ 8 テクセル、色をにじませる（mip の染み出し対策）
//
// **法線の門もDAViD前景も合成では連続値。** 写真色の重みは法線の門×前景信用度。生の確信度を
// そのまま線形重みにせず、`foregroundThreshold` 未満を `foregroundExponent` 乗で落とす。
// 背景色を頭部全体へ広げないため、しきい値未満は段2の伝播元からも外す。地点自身には小さな
// 写真比率を残すので、髪の細部や人物境界を硬く切らない。
//
// **左右反転で埋める段は持たない。** 以前は段1 で片側だけ埋まったテクセルを反対側から取って
// いた。撤去した理由は、ほくろ・傷・分け目のような左右非対称な特徴が反対側へ複製されること。
// 写真に無い色を作るなら、複製ではなく段2 で作ったと分かる形にする。
//
// 段3 の理由: アトラスの未使用領域が黒のまま残ると、mipmap を降りたときに chart の外の色が
// 縫い目に染み出す。
//
// 投影は平行投影（遠近法なし）。GNM 空間の xy を相似変換で写真ピクセルへ写し、z は遮蔽の判定に
// 使う。**法線が前を向いているだけでは足りない** — 耳のうしろの頭皮と鼻孔の縁は前を向いていても
// 手前の面（耳・鼻）に隠れていて、写真の誤った色を拾う。z の粗い深度バッファで落とす。

import { RgbImage } from '../contract';
import { ScalarField, sampleField } from '../field';
import { Similarity2d } from '../gnm/fit';
import { provenancePaletteImage } from '../inspection';
import { PhotoRgb, linearToSrgb8Array, samplePhotoLinearAt, srgbToLinear } from '../photo';
import { gate } from '../ramp';
import {
  DEFAULT_SURFACE_HARMONIC_SCREENING,
  MAXIMUM_SURFACE_HARMONIC_SCREENING,
  MINIMUM_SURFACE_HARMONIC_SCREENING,
  fillSurfaceClampedSmooth,
  meanColor,
  propagateOutward,
} from './fill';
import { AtlasSurface, buildAtlasSurface, coveredMask } from './surface';

export const PROVENANCE_UNUSED = 0;
export const PROVENANCE_PHOTO = 1;
export const PROVENANCE_FILL = 2;
export const PROVENANCE_DILATION = 3;
export const PROVENANCE_BLEND = 4;

/**
 * `provenance` の値と表示名。内訳の表示と検査画像の凡例の正本。
 *
 * `PROVENANCE_PHOTO` は**写真の色そのもの**（重み 1.0）に限る。門の傾斜の中にいるテクセルは
 * `PROVENANCE_BLEND`。ここを分けないと「写真の色と一致すること」を検査できなくなる。
 */
export const PROVENANCE_NAMES: readonly [number, string][] = [
  [PROVENANCE_PHOTO, '写真'],
  [PROVENANCE_BLEND, '混合'],
  [PROVENANCE_FILL, '埋め'],
  [PROVENANCE_DILATION, 'dilation'],
  [PROVENANCE_UNUSED, '未使用'],
];

/**
 * 検査画像の色分け。写真=緑 / 混合=青 / 埋め=赤 / dilation=黄 / 未使用=黒。
 *
 * **青の帯の幅が門の傾斜の効き方そのもの**なので、softness を振ったときはこの画像で幅を見る。
 */
export const PROVENANCE_INSPECTION_COLORS: readonly [number, [number, number, number]][] = [
  [PROVENANCE_UNUSED, [0, 0, 0]],
  [PROVENANCE_PHOTO, [0, 200, 60]],
  [PROVENANCE_FILL, [220, 40, 40]],
  [PROVENANCE_DILATION, [240, 210, 40]],
  [PROVENANCE_BLEND, [40, 120, 240]],
];

/**
 * 外から見える肌の chart。`AtlasSurface` は三角形数の降順で id を振るので必ず 0。
 *
 * これ以外の chart（GNM v3_0 / head では口腔壁 `mouth_sock` の 1 枚だけ）はメッシュの内側の
 * 面で、正面写真にも他のどんな写真にも絶対に写らない。
 */
export const EXTERIOR_CHART = 0;

export const DEFAULT_FOREGROUND_THRESHOLD = 0.95;
export const MINIMUM_FOREGROUND_THRESHOLD = 0.0;
export const MAXIMUM_FOREGROUND_THRESHOLD = 1.0;

export const DEFAULT_FOREGROUND_CONFIDENCE_EXPONENT = 6.0;
export const MINIMUM_FOREGROUND_CONFIDENCE_EXPONENT = 1.0;
export const MAXIMUM_FOREGROUND_CONFIDENCE_EXPONENT = 12.0;

/**
 * DAViDの生確信度を、しきい値未満だけ非線形な写真信用度へ写す。
 *
 * しきい値以上は生値を保つ。未満はしきい値で連続になる冪曲線
 * `threshold * (confidence / threshold) ** exponent` とし、中低確信度の背景混入を線形合成より
 * 強く抑える。しきい値0は比較用の従来線形動作。
 */
export function foregroundConfidenceWeight(
  confidence: number,
  threshold: number,
  exponent = DEFAULT_FOREGROUND_CONFIDENCE_EXPONENT,
): number {
  const clipped = Math.min(1, Math.max(0, confidence));
  if (threshold <= 0) return clipped;
  return clipped < threshold ? threshold * Math.pow(clipped / threshold, exponent) : clipped;
}

/** ベイクのパラメータ。既定値が実写での判断の記録。 */
export interface BakeSettings {
  /** アトラスの一辺。2048 のとき正面領域の実効解像度は 1110×1110 相当。 */
  readonly atlasSize: number;
  /**
   * 写真から焼く法線 z の下限（GNM 空間の +Z がカメラ方向）。
   *
   * 下げるほど写真由来の面積は増えるが、面が寝ているぶん写真の 1 画素が横に `1 / minFacing`
   * 倍へ伸びて焼かれる。実写で 0.15 / 0.30 / 0.45 を比べて 0.30 にした（0.15 は入射角 81°・
   * 伸び 6.7 倍で頬の外側と耳の前が髪の長い筋に伸びた像になり、埋めより悪い）。
   *
   * `facingSoftness` が 0 でないときは**下限ではなく傾斜の中点**になる。
   */
  readonly minFacing: number;
  /**
   * `minFacing` の門をぼかす半幅。0 で二値の門。
   *
   * 二値だと、法線が中点をまたぐ隣り合ったテクセルの間で色が写真から埋めへ跳び、**その線が
   * アトラスに焼き付く**。半幅を振って**写真由来と埋めが直接隣り合うテクセル対の数**で決めた
   * （0.00 で 12,900 対 → 0.15 で 649 対）。0.05 は量では 9 割落ちるが画像では線として見え、
   * 0.25 は線が消える代わりに頬の外側が埋めの色へ白茶ける。
   */
  readonly facingSoftness: number;
  /**
   * DAViD人物前景確信度の信用曲線の折れ点兼、補完色の伝播元の下限。
   *
   * しきい値未満は背景が混ざった可能性があるため、未撮影領域への伝播元には使わない。1.0は
   * 補間後の値では欠けやすく、0.95なら人物内部のほぼ確実な色を補完元として十分に残せる。
   */
  readonly foregroundThreshold: number;
  /** しきい値未満のDAViD確信度を落とす非線形曲線の指数。 */
  readonly foregroundExponent: number;
  /** 調和補完をowner色へ寄せる強さ。0で純粋なharmonicになる。 */
  readonly harmonicScreening: number;
  /**
   * 遮蔽判定に使う深度バッファのセルの大きさ（写真ピクセル）。
   *
   * アトラスのテクセルを写真へ投げた点をそのまま深度のサンプルにするので、1 セルに複数の
   * サンプルが入る大きさが必要。2048² のアトラスなら 4 px で十分に密になる。
   */
  readonly depthCellPx: number;
  /**
   * 手前の面より何メートル奥までを「見えている」とするか。
   *
   * 実効の許容量はここに「1 セルの中で表面の z がどれだけ変わりうるか」を足した値
   * （`occlusionTolerance()`）。
   */
  readonly occlusionTolerance: number;
  /** chart の外へ色をにじませるテクセル数。 */
  readonly chartDilationTexels: number;
  /**
   * 内側の chart（口腔壁）を塗る色 = 基準の肌色 × この係数。
   *
   * **公式 GNM の値**（`_VERTEX_GROUP_COLOR_MODIFIERS` が `mouth_sock: (0.7, 0.0)` と定めて
   * いる）。写真には写らないので焼けない。**修飾は sRGB で掛ける**（公式が sRGB の色へ掛けて
   * いるのと同じ演算にするため。リニアで掛けると口腔壁が白く浮く）。
   */
  readonly interiorScale: number;
}

export const DEFAULT_BAKE_SETTINGS: BakeSettings = {
  atlasSize: 2048,
  minFacing: 0.3,
  facingSoftness: 0.15,
  foregroundThreshold: DEFAULT_FOREGROUND_THRESHOLD,
  foregroundExponent: DEFAULT_FOREGROUND_CONFIDENCE_EXPONENT,
  harmonicScreening: DEFAULT_SURFACE_HARMONIC_SCREENING,
  depthCellPx: 4,
  occlusionTolerance: 0.004,
  chartDilationTexels: 8,
  interiorScale: 0.7,
};

export function validateBakeSettings(settings: BakeSettings): BakeSettings {
  if (
    !(
      MINIMUM_FOREGROUND_THRESHOLD <= settings.foregroundThreshold &&
      settings.foregroundThreshold <= MAXIMUM_FOREGROUND_THRESHOLD
    )
  ) {
    throw new Error(
      `foregroundThreshold が ${settings.foregroundThreshold}（範囲` +
        ` ${MINIMUM_FOREGROUND_THRESHOLD}〜${MAXIMUM_FOREGROUND_THRESHOLD}）`,
    );
  }
  if (
    !(
      MINIMUM_FOREGROUND_CONFIDENCE_EXPONENT <= settings.foregroundExponent &&
      settings.foregroundExponent <= MAXIMUM_FOREGROUND_CONFIDENCE_EXPONENT
    )
  ) {
    throw new Error(
      `foregroundExponent が ${settings.foregroundExponent}（範囲` +
        ` ${MINIMUM_FOREGROUND_CONFIDENCE_EXPONENT}〜${MAXIMUM_FOREGROUND_CONFIDENCE_EXPONENT}）`,
    );
  }
  if (
    !(
      MINIMUM_SURFACE_HARMONIC_SCREENING <= settings.harmonicScreening &&
      settings.harmonicScreening <= MAXIMUM_SURFACE_HARMONIC_SCREENING
    )
  ) {
    throw new Error(
      `harmonicScreening が ${settings.harmonicScreening}（範囲` +
        ` ${MINIMUM_SURFACE_HARMONIC_SCREENING}〜${MAXIMUM_SURFACE_HARMONIC_SCREENING}）`,
    );
  }
  return settings;
}

/** ベイクの結果。 */
export interface AtlasBake {
  /** (size, size, 3) sRGB。書き出すアトラス。 */
  readonly albedo: Uint8Array;
  /** (size, size) テクセルごとの由来（`PROVENANCE_*`）。 */
  readonly provenance: Uint8Array;
  /** テクセル → GNM 表面の対応（検査と再ベイクに使える）。 */
  readonly surface: AtlasSurface;
  readonly settings: BakeSettings;
}

/** 由来ごとのテクセル数。 */
export function provenanceCounts(bake: AtlasBake): Map<number, number> {
  const tally = new Map<number, number>();
  for (const [value] of PROVENANCE_NAMES) tally.set(value, 0);
  for (const value of bake.provenance) tally.set(value, (tally.get(value) ?? 0) + 1);
  return tally;
}

/**
 * 内訳を人が読める表にする。分母は 2 つ。
 *
 * chart 内（= 三角形が触れているテクセル）を分母にした割合が、実測の「正面 34% / 側面 16.7% /
 * 背面 49.3%」と比べられる数。dilation は chart の外を埋めるので、この分母では 0 になる。
 */
export function bakeReport(bake: AtlasBake): string {
  const counts = provenanceCounts(bake);
  const covered = coveredMask(bake.surface);
  let coveredCount = 0;
  for (const value of covered) if (value !== 0) coveredCount++;
  const total = bake.provenance.length;
  const lines = [
    `アトラス ${bake.surface.size}x${bake.surface.size} = ${total} テクセル /` +
      ` chart 内 ${coveredCount}（${((coveredCount / total) * 100).toFixed(1)}%）`,
    `段1 の門: 法線 nz ${bake.settings.minFacing}±${bake.settings.facingSoftness}` +
      ` / 補完元のDAViD前景 ≥${bake.settings.foregroundThreshold}`,
  ];
  for (const [value, name] of PROVENANCE_NAMES) {
    let inside = 0;
    for (let texel = 0; texel < total; texel++) {
      if (covered[texel] !== 0 && bake.provenance[texel] === value) inside++;
    }
    const count = counts.get(value) ?? 0;
    lines.push(
      `${name}: ${count}（全体 ${((count / total) * 100).toFixed(1)}% /` +
        ` chart 内 ${((inside / coveredCount) * 100).toFixed(1)}%）`,
    );
  }
  return lines.join('\n');
}

/**
 * 段1「写真投影」の中間結果。
 *
 * この値を段2以降から分けることで、写真投影だけを別実装へ置き換えても、色の伝播・口腔壁の
 * 塗り・dilationという品質判断は同じ純粋計算を通せる。
 *
 * **段2はこの配列を書き換えない**（`fillAtlasRemainder` が写してから塗る）。
 */
export interface AtlasPhotoProjection {
  readonly color: Float32Array;
  readonly photoWeight: Float64Array;
  readonly provenance: Uint8Array;
  readonly fillSourceWeight: Float64Array | null;
}

/** 補完済みで、まだリニア光のままのアトラス。 */
export interface AtlasLinearBake {
  readonly color: Float32Array;
  readonly provenance: Uint8Array;
}

/** 段1「写真投影」だけを実行する。 */
export function projectAtlasPhoto(
  surface: AtlasSurface,
  photo: PhotoRgb,
  similarity: Similarity2d,
  personMask: ScalarField,
  settings: BakeSettings,
): AtlasPhotoProjection {
  if (surface.size !== settings.atlasSize) {
    throw new Error(
      `渡された surface の一辺が ${surface.size}（settings は ${settings.atlasSize}）`,
    );
  }
  const texelCount = surface.size * surface.size;
  const color = new Float32Array(texelCount * 3);
  const photoWeight = new Float64Array(texelCount);
  const fillSourceWeight = new Float64Array(texelCount);
  const provenance = new Uint8Array(texelCount).fill(PROVENANCE_UNUSED);
  const exterior = exteriorMask(surface);
  bakeFromPhoto(
    color,
    photoWeight,
    fillSourceWeight,
    provenance,
    surface,
    exterior,
    photo,
    similarity,
    personMask,
    settings,
  );
  return { color, photoWeight, provenance, fillSourceWeight };
}

function exteriorMask(surface: AtlasSurface): Uint8Array {
  const out = new Uint8Array(surface.triangleIndex.length);
  for (let texel = 0; texel < out.length; texel++) {
    out[texel] =
      surface.triangleIndex[texel] >= 0 && surface.chartIndex[texel] === EXTERIOR_CHART ? 1 : 0;
  }
  return out;
}

/** 写真を GNM 公式 UV アトラスへ焼く。 */
export function bakeAtlas(input: {
  photo: PhotoRgb;
  vertices: Float64Array;
  triangles: Uint32Array;
  vertexUvs: Float32Array;
  componentId: Uint8Array;
  similarity: Similarity2d;
  personMask: ScalarField;
  skinBaseColor: readonly [number, number, number];
  settings?: BakeSettings;
  surface?: AtlasSurface;
  photoProjection?: AtlasPhotoProjection;
  fillRegionId?: Uint8Array | null;
  photoOnlyRegion?: Uint8Array | null;
  mouthRimRegion?: Float32Array | null;
}): AtlasBake {
  const settings = input.settings ?? DEFAULT_BAKE_SETTINGS;
  const surface =
    input.surface ??
    buildAtlasSurface(
      input.vertices,
      input.triangles,
      input.vertexUvs,
      input.componentId,
      settings.atlasSize,
    );
  if (surface.size !== settings.atlasSize) {
    throw new Error(
      `渡された surface の一辺が ${surface.size}（settings は ${settings.atlasSize}）`,
    );
  }
  const projection =
    input.photoProjection ??
    projectAtlasPhoto(surface, input.photo, input.similarity, input.personMask, settings);
  const linear = fillAtlasRemainder({
    photoProjection: projection,
    surface,
    vertices: input.vertices,
    triangles: input.triangles,
    skinBaseColor: input.skinBaseColor,
    settings,
    fillRegionId: input.fillRegionId ?? null,
    photoOnlyRegion: input.photoOnlyRegion ?? null,
    mouthRimRegion: input.mouthRimRegion ?? null,
  });
  dilateAtlasLinear(linear, surface, settings);
  return finalizeAtlasBake(linear, surface, settings);
}

/**
 * 段1の結果を受け、chart内の補完までを実行する。
 *
 * **`photoProjection` は読むだけで、書き換える配列は写しを作る。** frozen な値の中身を
 * 書き換えると、同じ投影を2回目の `settings` で焼いたときに1回目の塗りが混ざり、しかも
 * 例外は出ない。
 */
export function fillAtlasRemainder(input: {
  photoProjection: AtlasPhotoProjection;
  surface: AtlasSurface;
  vertices: Float64Array;
  triangles: Uint32Array;
  skinBaseColor: readonly [number, number, number];
  settings: BakeSettings;
  fillRegionId: Uint8Array | null;
  photoOnlyRegion: Uint8Array | null;
  mouthRimRegion: Float32Array | null;
}): AtlasLinearBake {
  const { photoProjection, surface, settings } = input;
  const texelCount = surface.size * surface.size;
  if (
    photoProjection.color.length !== texelCount * 3 ||
    photoProjection.photoWeight.length !== texelCount ||
    photoProjection.provenance.length !== texelCount
  ) {
    throw new Error('photoProjection の配列の形が surface と揃っていない');
  }
  const color = Float32Array.from(photoProjection.color);
  const provenance = Uint8Array.from(photoProjection.provenance);
  let photoWeight = Float64Array.from(photoProjection.photoWeight);
  let fillSourceWeight = Float64Array.from(
    photoProjection.fillSourceWeight ?? photoProjection.photoWeight,
  );

  const exterior = exteriorMask(surface);
  const interior = new Uint8Array(texelCount);
  for (let texel = 0; texel < texelCount; texel++) {
    interior[texel] = surface.triangleIndex[texel] >= 0 && exterior[texel] === 0 ? 1 : 0;
  }

  photoOrFillWeights(
    photoWeight,
    fillSourceWeight,
    exterior,
    surface,
    input.triangles,
    input.photoOnlyRegion,
  );

  const mouthRim = regionTexelMask(
    input.mouthRimRegion,
    exterior,
    surface,
    input.triangles,
    'mouthRimRegion',
  );
  if (mouthRim !== null) {
    // **首・後頭部とまったく同じ扱いにする** — 写真の重みを隠れている度合いだけ落として、段2の
    // 補完（表面に沿った伝播）に譲る。この帯に写っているのは唇の合わせ目の暗線だけで、写真の
    // 色として正しくない。**専用の色は作らない** — 周囲の唇から伝播した色がそのまま入るので、
    // 帯は写真の唇と地続きに馴染む。
    //
    // **掛け算にするのが要点。** 二値で捨てると帯の縁で色が跳ぶ（前の実装がそれで、切れ目が
    // 線として見えた）。頂点の重みは 3 値だが、barycentric 補間で三角形の中では連続になる。
    let any = false;
    for (let texel = 0; texel < texelCount; texel++) {
      if (mouthRim[texel] > 0) {
        any = true;
        break;
      }
    }
    if (any) {
      for (let texel = 0; texel < texelCount; texel++) {
        const keep = 1 - mouthRim[texel];
        photoWeight[texel] *= keep;
        fillSourceWeight[texel] *= keep;
        // 由来を合わせ直す。**緑は重み 1.0 の地点だけ**という約束を、重みを落とした後も守る。
        if (mouthRim[texel] > 0 && photoWeight[texel] > 0) provenance[texel] = PROVENANCE_BLEND;
      }
    }
  }

  fillRemainder(
    color,
    photoWeight,
    fillSourceWeight,
    provenance,
    exterior,
    interior,
    surface,
    input.vertices,
    input.triangles,
    input.skinBaseColor,
    settings,
    input.fillRegionId,
  );
  return { color, provenance };
}

/**
 * 頂点ごとの領域（0/1 か 0..1 の重み）を、外側 chart のテクセルの重みへ落とす。
 *
 * テクセルが乗る三角形の3頂点の値を barycentric で補間する。**閾値を掛けずに返す** — 呼び側が
 * 二値にしたいなら自分で切る。ここで切ってしまうと、頂点の重みを連続にした意味が三角形の
 * 境界で消える。
 */
function regionTexelMask(
  region: Uint8Array | Float32Array | null,
  exterior: Uint8Array,
  surface: AtlasSurface,
  triangles: Uint32Array,
  name: string,
): Float64Array | null {
  if (region === null) return null;
  let maximumVertex = 0;
  for (const vertex of triangles) if (vertex > maximumVertex) maximumVertex = vertex;
  if (region.length !== maximumVertex + 1) {
    throw new Error(`${name}の長さが${region.length}（期待${maximumVertex + 1}）`);
  }
  const out = new Float64Array(exterior.length);
  for (let texel = 0; texel < exterior.length; texel++) {
    if (exterior[texel] === 0) continue;
    const triangle = surface.triangleIndex[texel];
    if (triangle < 0) continue;
    let total = 0;
    for (let corner = 0; corner < 3; corner++) {
      total += surface.barycentric[texel * 3 + corner] * region[triangles[triangle * 3 + corner]];
    }
    out[texel] = total;
  }
  return out;
}

/**
 * 指定領域の青い混合投影を、補完色100%へ置き換える（in-place）。
 *
 * 緑（写真重み1）は写真色のまま補完元にも使う。青（0より大きく1未満）は地点自身の写真色にも
 * 補完元にも使わない。DAViDしきい値による `fillSourceWeight` の判定は先に済んでおり、その
 * 条件は変更しない。
 */
function photoOrFillWeights(
  photoWeight: Float64Array,
  fillSourceWeight: Float64Array,
  exterior: Uint8Array,
  surface: AtlasSurface,
  triangles: Uint32Array,
  photoOnlyRegion: Uint8Array | null,
): void {
  const membership = regionTexelMask(
    photoOnlyRegion,
    exterior,
    surface,
    triangles,
    'photoOnlyRegion',
  );
  if (membership === null) return;
  for (let texel = 0; texel < photoWeight.length; texel++) {
    // ここは二値のまま。首・胴体は「写真100%か補完100%の二択」にする段で、傾斜を付ける相手
    // （青）を消すのが目的だから。
    if (membership[texel] < 0.5) continue;
    if (photoWeight[texel] > 0 && photoWeight[texel] < 1) {
      photoWeight[texel] = 0;
      fillSourceWeight[texel] = 0;
    }
  }
}

/** 段3 dilation。`linear` をその場で更新する。 */
export function dilateAtlasLinear(
  linear: AtlasLinearBake,
  surface: AtlasSurface,
  settings: BakeSettings,
): void {
  // ここへ来る時点で chart 内は全テクセルに色が入っているので、出発点は covered そのもの。
  const covered = coveredMask(surface);
  const outside = new Uint8Array(covered.length);
  for (let texel = 0; texel < covered.length; texel++) outside[texel] = covered[texel] === 0 ? 1 : 0;
  const newly = propagateOutward(
    linear.color,
    Uint8Array.from(covered),
    outside,
    surface.size,
    surface.size,
    settings.chartDilationTexels,
  );
  for (let texel = 0; texel < newly.length; texel++) {
    if (newly[texel] !== 0) linear.provenance[texel] = PROVENANCE_DILATION;
  }
}

/** リニア光の中間値を検証し、公開するsRGB uint8結果へ量子化する。 */
export function finalizeAtlasBake(
  linear: AtlasLinearBake,
  surface: AtlasSurface,
  settings: BakeSettings,
): AtlasBake {
  const covered = coveredMask(surface);
  for (let texel = 0; texel < covered.length; texel++) {
    if (covered[texel] !== 0 && linear.provenance[texel] === PROVENANCE_UNUSED) {
      throw new Error('chart 内に色を置いていないテクセルが残っている');
    }
  }
  return {
    albedo: linearToSrgb8Array(linear.color),
    provenance: linear.provenance,
    surface,
    settings,
  };
}

/**
 * 段1: 前を向いていて、手前の面に隠れていないテクセルへ写真の色と重みを置く。
 *
 * 写真の合成重みは法線の門×非線形化したDAViD前景信用度。しきい値未満も地点自身では小さく写真を
 * 混ぜるが、`fillSourceWeight` を0にして未撮影領域へは伝播させない。混ぜるのは段2。
 *
 * **遮蔽だけは二値のまま。** 手前に面があるかは幾何の事実で、確信度ではない。実測で落ちるのは
 * chart 内 0.27%、場所は耳のうしろと鼻孔の縁に集中しており、いずれも「耳の像」と「埋めた頭皮」
 * という別物の境界なので、間を混ぜても意味のある色にならない。
 *
 * **髪に覆われた画素も焼く。** 写真の前髪が額へ入るのは意図した動きで、髪シェルの柔らかい
 * alpha の裏当てになっている。落とすと隙間から肌色が見えて頭皮が透ける。
 *
 * 内側の chart（口腔壁）は除く。口腔壁は唇の裏から後ろへ伸びる袋で、袋の底が +Z を向いている
 * ため法線の判定だけでは通ってしまい、唇や顎の色を拾う。
 */
function bakeFromPhoto(
  color: Float32Array,
  photoWeight: Float64Array,
  fillSourceWeight: Float64Array,
  provenance: Uint8Array,
  surface: AtlasSurface,
  exterior: Uint8Array,
  photo: PhotoRgb,
  similarity: Similarity2d,
  personMask: ScalarField,
  settings: BakeSettings,
): void {
  const texels: number[] = [];
  for (let texel = 0; texel < exterior.length; texel++) if (exterior[texel] !== 0) texels.push(texel);
  if (texels.length === 0) return;

  const pixelX = new Float64Array(texels.length);
  const pixelY = new Float64Array(texels.length);
  const depth = new Float64Array(texels.length);
  const facing = new Float64Array(texels.length);
  for (let slot = 0; slot < texels.length; slot++) {
    const texel = texels[slot];
    const [x, y] = similarity.applyPoint(
      surface.position[texel * 3],
      surface.position[texel * 3 + 1],
    );
    pixelX[slot] = x;
    pixelY[slot] = y;
    depth[slot] = surface.position[texel * 3 + 2];
    facing[slot] = gate(surface.normal[texel * 3 + 2], settings.minFacing, settings.facingSoftness);
  }

  const unoccluded = computeUnoccluded(pixelX, pixelY, depth, photo, similarity, settings);
  const sampled = new Float32Array(3);
  for (let slot = 0; slot < texels.length; slot++) {
    if (!(facing[slot] > 0) || unoccluded[slot] === 0) continue;
    const inside = samplePhotoLinearAt(photo, pixelX[slot], pixelY[slot], sampled, 0);
    const foreground = Math.min(
      1,
      Math.max(0, sampleField(personMask, pixelX[slot] / photo.width, pixelY[slot] / photo.height)),
    );
    const confidenceWeight = foregroundConfidenceWeight(
      foreground,
      settings.foregroundThreshold,
      settings.foregroundExponent,
    );
    const weight = facing[slot] * confidenceWeight;
    // 写真の枠の外は二値。枠の外に色は無いので、傾斜を付ける先が無い。
    if (!inside || !(weight > 0)) continue;
    const texel = texels[slot];
    color[texel * 3] = sampled[0];
    color[texel * 3 + 1] = sampled[1];
    color[texel * 3 + 2] = sampled[2];
    photoWeight[texel] = weight;
    fillSourceWeight[texel] = foreground >= settings.foregroundThreshold ? weight : 0;
    // 重み1.0だけを「写真」と呼ぶ。法線とDAViD確信度の両方が1の地点に限る。
    provenance[texel] = weight >= 1 ? PROVENANCE_PHOTO : PROVENANCE_BLEND;
  }
}

/**
 * 手前の面に隠れていないテクセルを返す（平行投影の深度判定）。
 *
 * 深度バッファは投影した点そのものから作る。表面をもう一度写真空間へラスタライズする必要が
 * ないうえ、アトラスのテクセルは写真の画素より密なので、粗いセルにまとめれば穴が空かない。
 *
 * 写真の外へ出る点は 1（範囲の判定は呼び出し側の仕事）。
 */
function computeUnoccluded(
  pixelX: Float64Array,
  pixelY: Float64Array,
  depth: Float64Array,
  photo: PhotoRgb,
  similarity: Similarity2d,
  settings: BakeSettings,
): Uint8Array {
  const out = new Uint8Array(depth.length).fill(1);
  const cell = Math.max(1, settings.depthCellPx);
  const columns = Math.ceil(photo.width / cell);
  const rows = Math.ceil(photo.height / cell);
  const front = new Float64Array(columns * rows).fill(-Infinity);
  const cellIndex = new Int32Array(depth.length).fill(-1);

  for (let slot = 0; slot < depth.length; slot++) {
    const x = pixelX[slot];
    const y = pixelY[slot];
    if (!(x >= 0 && x < photo.width && y >= 0 && y < photo.height)) continue;
    const index = Math.floor(y / cell) * columns + Math.floor(x / cell);
    cellIndex[slot] = index;
    if (depth[slot] > front[index]) front[index] = depth[slot];
  }

  const tolerance = occlusionTolerance(similarity, settings);
  for (let slot = 0; slot < depth.length; slot++) {
    const index = cellIndex[slot];
    if (index < 0) continue;
    out[slot] = depth[slot] >= front[index] - tolerance ? 1 : 0;
  }
  return out;
}

/**
 * 遮蔽判定の許容量（メートル）。
 *
 * `occlusionTolerance` に「1 セルの対角ぶん横に動いたときに表面の z が動きうる幅」を足す。
 * 傾きの上限は**受け入れる最も寝た面**から決まるので、この 2 つから機械的に出せる。
 *
 * **見るのは `minFacing` ではなく `minFacing − facingSoftness`。** 門が二値だった間は
 * `minFacing` が下限そのものだったが、傾斜にしたので重みが 0 を超えるのは中点の
 * `facingSoftness` 下まで。狭く見積もると、寝た面（こめかみ・頬の外側）のセル内 z 変化を遮蔽と
 * 誤判定して埋めに落とす。
 */
export function occlusionTolerance(similarity: Similarity2d, settings: BakeSettings): number {
  const facing = Math.min(Math.max(settings.minFacing - settings.facingSoftness, 1e-3), 1);
  const slope = Math.sqrt(1 - facing * facing) / facing;
  const cellMeters = Math.max(1, settings.depthCellPx) / Math.max(similarity.scale, 1e-9);
  return settings.occlusionTolerance + Math.SQRT2 * cellMeters * slope;
}

/**
 * 段2: 残った chart 内のテクセル（頭皮・後頭部・首の裏）と内側の chart を埋める。
 *
 * 重みが 1 未満のテクセルは**埋めた場と写真色を重みで混ぜる**（重み 0 なら埋めた場そのもの）。
 * 混ぜる相手を平らな基準色ではなく埋めた場にするのは、埋めた場が写真の色から伝播して出来て
 * いるので**境界の両側で連続している**ため。平らな色を相手にすると傾斜の外側の端に新しい段差が
 * 生まれる。
 *
 * 伝播はメッシュの表面に沿って広がるので、頭皮のソースになるのは表面上の隣 = 生え際と前髪で
 * ある。前髪は段1 で額へ焼かれているため、**頭皮は指示なしで髪の色を受け取る。**
 *
 * 左右は平均しない。正面写真にも照明・髪・耳・服の左右差があり、反対側との平均は正しい取得元の
 * 色を変えてしまう。UV seam の複製頂点は3D位置で同じnodeへ統合してから伝播するため、シームだけは
 * 同じ頂点色を共有できる。
 */
function fillRemainder(
  color: Float32Array,
  photoWeight: Float64Array,
  fillSourceWeight: Float64Array,
  provenance: Uint8Array,
  exterior: Uint8Array,
  interior: Uint8Array,
  surface: AtlasSurface,
  vertices: Float64Array,
  triangles: Uint32Array,
  skinBaseColor: readonly [number, number, number],
  settings: BakeSettings,
  fillRegionId: Uint8Array | null,
): void {
  const texelCount = photoWeight.length;
  const trustedSource = new Uint8Array(texelCount);
  let anyTrusted = false;
  for (let texel = 0; texel < texelCount; texel++) {
    if (fillSourceWeight[texel] > 0) {
      trustedSource[texel] = 1;
      anyTrusted = true;
    }
  }
  const skinMean = anyTrusted
    ? meanColor(color, trustedSource)
    : new Float32Array([
        srgbToLinear(skinBaseColor[0]),
        srgbToLinear(skinBaseColor[1]),
        srgbToLinear(skinBaseColor[2]),
      ]);

  const blended = new Uint8Array(texelCount);
  let anyBlended = false;
  for (let texel = 0; texel < texelCount; texel++) {
    if (exterior[texel] !== 0 && photoWeight[texel] < 1) {
      blended[texel] = 1;
      anyBlended = true;
    }
  }

  if (anyBlended) {
    // `skinMean` は**ソースがどこからも届かなかったところの色**として渡す。届かない所を黒の
    // ままにするとピラミッドの既定の受け皿（0）が残るので、肌の色を受け皿にする。
    const field = fillSurfaceClampedSmooth({
      color,
      sourceWeight: fillSourceWeight,
      region: exterior,
      triangleIndex: surface.triangleIndex,
      barycentric: surface.barycentric,
      triangles,
      vertexPositions: vertices,
      fallback: skinMean,
      vertexRegionId: fillRegionId,
      harmonicScreening: settings.harmonicScreening,
    });
    for (let texel = 0; texel < texelCount; texel++) {
      if (blended[texel] === 0) continue;
      const weight = photoWeight[texel];
      for (let channel = 0; channel < 3; channel++) {
        const fill = field[texel * 3 + channel];
        color[texel * 3 + channel] = fill + (color[texel * 3 + channel] - fill) * weight;
      }
    }
  }

  // 口腔壁は写真に写らないので、公式の頂点色の規則で塗る（sRGB で掛ける）。
  const interiorColor = [0, 1, 2].map((channel) =>
    srgbToLinear(Math.min(1, Math.max(0, skinBaseColor[channel] * settings.interiorScale))),
  );
  for (let texel = 0; texel < texelCount; texel++) {
    if (interior[texel] === 0) continue;
    color[texel * 3] = interiorColor[0];
    color[texel * 3 + 1] = interiorColor[1];
    color[texel * 3 + 2] = interiorColor[2];
  }

  for (let texel = 0; texel < texelCount; texel++) {
    if ((exterior[texel] !== 0 && photoWeight[texel] <= 0) || interior[texel] !== 0) {
      provenance[texel] = PROVENANCE_FILL;
    }
  }
}

/** 由来を色分けした検査画像を作る。 */
export function provenanceInspectionImage(provenance: Uint8Array, size: number): RgbImage {
  return provenancePaletteImage(provenance, size, size, PROVENANCE_INSPECTION_COLORS);
}
