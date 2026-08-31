// 書き出しの調整パラメータ。
//
// **入口（GUI）から与える値をここに1つだけ集める。** 段ごとに引数を生やすと、入口が段の内訳を
// 知ることになり、段を1つ足すたびに入口が増える。
//
// 既定値は「この値で書き出して良い」という判断そのものなので、根拠を各属性に置く。入口は既定を
// 持たない — 入口ごとに違う既定を持つと、どちらで動かしたかで結果が変わる。
//
// **自動決定を持たない。** 以前はアトラスと眼球の一辺を写真の顔の大きさから逆算していた。逆算は
// 「写真の情報を捨てない最小の一辺」という測った根拠を持っていたが、消費側（Unity）のテクスチャ
// 予算を Exporter が代わりに決めることでもあった。**捨てた情報は戻せないが、余った解像度は消費側で
// 縮められる**ので、決めるのは消費側に寄せる。

import {
  DEFAULT_FOREGROUND_CONFIDENCE_EXPONENT,
  DEFAULT_FOREGROUND_THRESHOLD,
  MAXIMUM_FOREGROUND_CONFIDENCE_EXPONENT,
  MAXIMUM_FOREGROUND_THRESHOLD,
  MINIMUM_FOREGROUND_CONFIDENCE_EXPONENT,
  MINIMUM_FOREGROUND_THRESHOLD,
} from '../domain/atlas/bake';
import {
  DEFAULT_SURFACE_HARMONIC_SCREENING,
  MAXIMUM_SURFACE_HARMONIC_SCREENING,
  MINIMUM_SURFACE_HARMONIC_SCREENING,
} from '../domain/atlas/fill';
import {
  LEGACY_LIFT_METERS,
  LEGACY_ROLLOFF_METERS,
  MAXIMUM_LIFT_METERS,
  MAXIMUM_ROLLOFF_METERS,
  MINIMUM_LIFT_METERS,
  MINIMUM_ROLLOFF_METERS,
} from '../domain/hair/shell';

/**
 * 入口はミリで扱い、`domain` はメートルで扱う。その換算はここ1箇所だけ。
 *
 * 殻の持ち上げは 3 mm、巻き込みは 12 mm の桁で、メートルだと入力欄が `0.00298` になる。**値を2つの
 * 単位で書き写さない** — ミリ側の定数はすべて `domain` のメートル値から導く。
 */
export const MILLIMETERS_PER_METER = 1000.0;

/**
 * 肌アトラスと髪テクスチャで選べる一辺（2 の冪）。
 *
 * 上限が 4096 なのはベイクの実測から（一辺の 2 乗で伸びるので 8192 は 2 分を超える）。
 */
export const TEXTURE_SIZE_CHOICES: readonly number[] = [512, 1024, 2048, 4096];

/**
 * 眼球テクスチャで選べる一辺。
 *
 * 肌より 2 段小さいのは焼く対象が小さいため。**テクセルと写真画素が 1:1 になる一辺は**
 * `limbus半径px / IRIS_OUTER_RADIUS_UV` で、実写（顔幅 530〜632 px）では 116 前後になる。
 */
export const EYE_TEXTURE_SIZE_CHOICES: readonly number[] = [128, 256, 512, 1024];

/**
 * 肌アトラスの既定。
 *
 * 基準は「正面を向いた面の隣接テクセルが写真の 1 画素ぶん離れる」。実写 10 枚の顔幅は 530〜632 px
 * で、この一辺がちょうど良い顔幅はその範囲に入る。4096 はこの顔の大きさでは写真に無い情報のために
 * テクセルを持つだけになる。
 */
export const DEFAULT_SKIN_ATLAS_SIZE = 2048;

/**
 * 眼球テクスチャの既定。
 *
 * 写真画素と 1:1 になる一辺は実写で 116 前後なので 128 がちょうどだが、**余裕が無い**。眼球は顔の
 * 大きさに比例するので、顔幅が 2 倍の写真では適正が 256 になる。捨てた解像度は戻せず、余った解像度は
 * 消費側で縮められるので 1 段上を既定にする。
 */
export const DEFAULT_EYE_TEXTURE_SIZE = 256;

/**
 * 髪テクスチャ（`hair_albedo.jpg` / `hair_alpha.png`）の**長辺**の既定。
 *
 * **この値は測っていない。** 髪の UV は写真への平行投影なので、肌アトラス（曲面を UV へ広げるぶん
 * 1.7 倍の解像度が要る）とは必要な密度が違う。揃えたのは「肌と同程度」以上の根拠が無いため。
 */
export const DEFAULT_HAIR_TEXTURE_SIZE = 2048;

/** 事前分布の強さの倍率の既定（1.0 = `LandmarkModel` が持つ値そのまま）。 */
export const DEFAULT_DISAGREEMENT_SCALE = 1.0;

export const DEFAULT_ATLAS_FOREGROUND_THRESHOLD = DEFAULT_FOREGROUND_THRESHOLD;
export const MINIMUM_ATLAS_FOREGROUND_THRESHOLD = MINIMUM_FOREGROUND_THRESHOLD;
export const MAXIMUM_ATLAS_FOREGROUND_THRESHOLD = MAXIMUM_FOREGROUND_THRESHOLD;

export const DEFAULT_ATLAS_FOREGROUND_EXPONENT = DEFAULT_FOREGROUND_CONFIDENCE_EXPONENT;
export const MINIMUM_ATLAS_FOREGROUND_EXPONENT = MINIMUM_FOREGROUND_CONFIDENCE_EXPONENT;
export const MAXIMUM_ATLAS_FOREGROUND_EXPONENT = MAXIMUM_FOREGROUND_CONFIDENCE_EXPONENT;

export const DEFAULT_ATLAS_HARMONIC_SCREENING = DEFAULT_SURFACE_HARMONIC_SCREENING;
export const MINIMUM_ATLAS_HARMONIC_SCREENING = MINIMUM_SURFACE_HARMONIC_SCREENING;
export const MAXIMUM_ATLAS_HARMONIC_SCREENING = MAXIMUM_SURFACE_HARMONIC_SCREENING;

/** 髪シェル全体を頭皮の手前へ出す固定量（ミリ）。 */
export const DEFAULT_HAIR_LIFT_MM = LEGACY_LIFT_METERS * MILLIMETERS_PER_METER;
export const MINIMUM_HAIR_LIFT_MM = MINIMUM_LIFT_METERS * MILLIMETERS_PER_METER;
export const MAXIMUM_HAIR_LIFT_MM = MAXIMUM_LIFT_METERS * MILLIMETERS_PER_METER;

/**
 * 髪シェルの外周を頭皮の内側へ巻き込む量（ミリ）。
 *
 * 縁を頭皮より奥へ潜らせて、斜めから見た継ぎ目を隠す。**持ち上げ量より大きい必要がある** — 下回ると
 * 縁が頭皮の外に出て、殻が浮いて見える。
 */
export const DEFAULT_HAIR_ROLLOFF_MM = LEGACY_ROLLOFF_METERS * MILLIMETERS_PER_METER;
export const MINIMUM_HAIR_ROLLOFF_MM = MINIMUM_ROLLOFF_METERS * MILLIMETERS_PER_METER;
export const MAXIMUM_HAIR_ROLLOFF_MM = MAXIMUM_ROLLOFF_METERS * MILLIMETERS_PER_METER;

/**
 * `identityClip` を**置くと決めたとき**の値（既定は上限なし = null）。
 *
 * 3 なのは、公式デモの identity スライダーが ±3 で、係数が z-score スケールであることから。ただし
 * 公式 GNM 自体は上限を持たず、成分ごとの標準偏差も定義していないので、**±3σ という言い方は成り
 * 立たない**。事前分布を緩めた端で何が壊れるかを見るための足場としての値。
 */
export const DEFAULT_IDENTITY_CLIP = 3.0;

/**
 * `identityClip` を挟む範囲。
 *
 * 下限は「平均顔に潰れる手前」、上限は実測の上（事前分布を最大まで緩めても係数は 10 を超えなかった）。
 * **上限を撤去しないのは、入口の入力欄が範囲を持たないと桁を打ち間違えたときに黙って平均顔が出る
 * ため。**
 */
export const MINIMUM_IDENTITY_CLIP = 0.1;
export const MAXIMUM_IDENTITY_CLIP = 20.0;

/**
 * `disagreementScale` を挟む範囲。
 *
 * λ は仮定するずれの 2 乗に比例するので、この範囲は λ の ×0.01〜×100 に当たる。
 */
export const MINIMUM_DISAGREEMENT_SCALE = 0.1;
export const MAXIMUM_DISAGREEMENT_SCALE = 10.0;

/** 1 枚を書き出すときの調整パラメータ。 */
export interface ExportSettings {
  /** 事前分布の強さの倍率。大きいほど平均顔寄り。 */
  readonly disagreementScale: number;
  /** identity 係数の絶対値の上限。null で上限なし（既定）。 */
  readonly identityClip: number | null;
  /** `skin_albedo.jpg` の一辺。 */
  readonly skinAtlasSize: number;
  /** 眼球テクスチャの一辺（左右同じ）。 */
  readonly eyeTextureSize: number;
  /** 髪テクスチャの**長辺**（縦横比は写真のまま）。 */
  readonly hairTextureSize: number;
  /** DAViD人物前景の信用曲線の折れ点兼、Atlas補完元の下限。 */
  readonly atlasForegroundThreshold: number;
  /** しきい値未満の信用度を落とす非線形指数。 */
  readonly atlasForegroundExponent: number;
  /** harmonic補完をowner色へ寄せる強さ。 */
  readonly atlasHarmonicScreening: number;
  /** 髪シェルを頭皮の手前へ出す固定量（ミリ）。 */
  readonly hairLiftMm: number;
  /** 髪シェルの外周を頭皮の内側へ巻き込む量（ミリ）。 */
  readonly hairRolloffMm: number;
}

/**
 * 既定のパラメータ一式。
 *
 * 入口が何も指定しないときの値。**既定を持つ場所をここ1つにする** — 入口ごとに既定を書くと、
 * 入口の数だけ結果が変わる状態を作れてしまう。
 */
export const DEFAULT_SETTINGS: ExportSettings = {
  disagreementScale: DEFAULT_DISAGREEMENT_SCALE,
  identityClip: null,
  skinAtlasSize: DEFAULT_SKIN_ATLAS_SIZE,
  eyeTextureSize: DEFAULT_EYE_TEXTURE_SIZE,
  hairTextureSize: DEFAULT_HAIR_TEXTURE_SIZE,
  atlasForegroundThreshold: DEFAULT_ATLAS_FOREGROUND_THRESHOLD,
  atlasForegroundExponent: DEFAULT_ATLAS_FOREGROUND_EXPONENT,
  atlasHarmonicScreening: DEFAULT_ATLAS_HARMONIC_SCREENING,
  hairLiftMm: DEFAULT_HAIR_LIFT_MM,
  hairRolloffMm: DEFAULT_HAIR_ROLLOFF_MM,
};

export function validateExportSettings(settings: ExportSettings): ExportSettings {
  const inRange = (value: number, low: number, high: number): boolean =>
    low <= value && value <= high;
  if (!inRange(settings.disagreementScale, MINIMUM_DISAGREEMENT_SCALE, MAXIMUM_DISAGREEMENT_SCALE)) {
    throw new Error(
      `disagreementScale が ${settings.disagreementScale}` +
        `（範囲 ${MINIMUM_DISAGREEMENT_SCALE}〜${MAXIMUM_DISAGREEMENT_SCALE}）`,
    );
  }
  if (
    !inRange(
      settings.atlasForegroundExponent,
      MINIMUM_ATLAS_FOREGROUND_EXPONENT,
      MAXIMUM_ATLAS_FOREGROUND_EXPONENT,
    )
  ) {
    throw new Error(
      `atlasForegroundExponent が ${settings.atlasForegroundExponent}` +
        `（範囲 ${MINIMUM_ATLAS_FOREGROUND_EXPONENT}〜${MAXIMUM_ATLAS_FOREGROUND_EXPONENT}）`,
    );
  }
  if (
    !inRange(
      settings.atlasHarmonicScreening,
      MINIMUM_ATLAS_HARMONIC_SCREENING,
      MAXIMUM_ATLAS_HARMONIC_SCREENING,
    )
  ) {
    throw new Error(
      `atlasHarmonicScreening が ${settings.atlasHarmonicScreening}` +
        `（範囲 ${MINIMUM_ATLAS_HARMONIC_SCREENING}〜${MAXIMUM_ATLAS_HARMONIC_SCREENING}）`,
    );
  }
  if (
    settings.identityClip !== null &&
    !inRange(settings.identityClip, MINIMUM_IDENTITY_CLIP, MAXIMUM_IDENTITY_CLIP)
  ) {
    throw new Error(
      `identityClip が ${settings.identityClip}` +
        `（範囲 ${MINIMUM_IDENTITY_CLIP}〜${MAXIMUM_IDENTITY_CLIP}、上限なしは null）`,
    );
  }
  if (
    !inRange(
      settings.atlasForegroundThreshold,
      MINIMUM_ATLAS_FOREGROUND_THRESHOLD,
      MAXIMUM_ATLAS_FOREGROUND_THRESHOLD,
    )
  ) {
    throw new Error(
      `atlasForegroundThreshold が ${settings.atlasForegroundThreshold}` +
        `（範囲 ${MINIMUM_ATLAS_FOREGROUND_THRESHOLD}〜${MAXIMUM_ATLAS_FOREGROUND_THRESHOLD}）`,
    );
  }
  for (const [name, value, low, high] of [
    ['hairLiftMm', settings.hairLiftMm, MINIMUM_HAIR_LIFT_MM, MAXIMUM_HAIR_LIFT_MM],
    ['hairRolloffMm', settings.hairRolloffMm, MINIMUM_HAIR_ROLLOFF_MM, MAXIMUM_HAIR_ROLLOFF_MM],
  ] as const) {
    if (!inRange(value, low, high)) throw new Error(`${name} が ${value}（範囲 ${low}〜${high}）`);
  }
  if (settings.hairLiftMm >= settings.hairRolloffMm) {
    // 縁の z は `lift - rolloff`。ここが正だと殻が頭皮の外へ浮き、斜めから見た継ぎ目が出る。
    // 片方だけ見ても検査できないのでここに置く。
    throw new Error(
      `hairLiftMm (${settings.hairLiftMm}) が hairRolloffMm (${settings.hairRolloffMm}) 以上です。` +
        '外周が頭皮へ巻き込めず、髪シェルが浮いて見えます。',
    );
  }
  for (const [name, value, choices] of [
    ['skinAtlasSize', settings.skinAtlasSize, TEXTURE_SIZE_CHOICES],
    ['eyeTextureSize', settings.eyeTextureSize, EYE_TEXTURE_SIZE_CHOICES],
    ['hairTextureSize', settings.hairTextureSize, TEXTURE_SIZE_CHOICES],
  ] as const) {
    if (!choices.includes(value)) {
      throw new Error(`${name} が ${value}（選べる値 ${choices.join(', ')}）`);
    }
  }
  return settings;
}
