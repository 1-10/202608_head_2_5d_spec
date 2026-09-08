// あいうえおの口形（viseme）。3D ビューだけが使う。
//
// **口形はアセットへ焼いた 1 本のプリセット。** 実行時に複数のプリセットを重ねて作るのではない。
// 焼く段は `tools/export_gnm_assets.py`、混ぜ方の定義は `tools/viseme_presets.json`（そこが正本）。
// 中身は公式 20 クラスの**係数行の線形結合**なので、実行時に 2 本重ねるのと数値的に同一 —
// **それでも焼くのは、口形が 1 本のプリセットになると `expression.ts` の「同時に立てるのは 1 本だけ」を
// 壊さずに済むから。** 実行時合成にすると、口形を出すためだけに「重ねてよい経路」をビューアーへ
// 作ることになり、原則が例外だらけになる。
//
// **正本の向きがここだけ逆になる。** 表情 20 本の正本は Unity 側（`Tools/export_expression_presets.py`）
// だが、**口形 5 本の正本は web 側**。Unity 側は 20 本しか持たないので、これは既知の乖離である
// （README の「web だから増えたもの」に、Unity へ持っていくものを書いてある）。
//
// ## 表情と口形は名前で分ける
//
// アセットの `expressionPresetNames` は 25 本を 1 本の並びで持つ。**一覧をここへ書き写さない** —
// 増減したとき黙って古くなる。代わりに名前の頭（`viseme_`）で切る。頭の正本は焼く側の
// `VISEME_PRESET_PREFIX` で、ズレたら口形がどちらのフォルダにも出なくなる（テストが見ている）。
//
// ## 同時に立てるのは 1 本だけ（口形も同じ）
//
// 口形が 1 本のプリセットになった以上、`expression.ts` の原則がそのまま当てはまる。実測でも口形は
// 口まわりだけでなく顔全体を動かす（目領域も数 mm 動く）ので、重ねれば表情と同じように壊れる。
// **自動再生と連続再生は機械で 1 本に保つ。** 手のスライダーは既存の表情スライダーと同じ扱いで、
// 重ねること自体は止めない（利用者が意図して混ぜる余地は残す）。

import { GnmPreviewAsset } from './asset';
// 台形エンベロープは表情と同じものを使う（口形だけ別の波形にする理由が無い）。
import { envelope } from './expression';

/**
 * 口形プリセットの名前の頭。
 *
 * 正本は `tools/export_gnm_assets.py` の `VISEME_PRESET_PREFIX`。焼く側と読む側で頭が違うと、
 * 口形が「表情」の一覧に紛れ込む（`tests/viseme.test.ts` が突き合わせている）。
 */
export const VISEME_PREFIX = 'viseme_';

/**
 * 口形の日本語ラベル。
 *
 * **アセットに焼いた名前を鍵にする。** ここに無い口形が焼かれたら `visemeLabel` は名前をそのまま
 * 返すので画面からは消えない（テストが「ラベルの無い口形」を落とす）。
 */
export const VISEME_LABELS: Readonly<Record<string, string>> = {
  viseme_a: 'あ',
  viseme_i: 'い',
  viseme_u: 'う',
  viseme_e: 'え',
  viseme_o: 'お',
};

/** 口形プリセットか。 */
export function isVisemePreset(name: string): boolean {
  return name.startsWith(VISEME_PREFIX);
}

/** 画面に出す名前。ラベルを持たない口形は焼いた名前のまま出す（黙って消さない）。 */
export function visemeLabel(name: string): string {
  return VISEME_LABELS[name] ?? name;
}

/** プリセットの index を表情と口形へ分けた結果。並びはアセットの並びのまま。 */
export interface PresetSplit {
  /** 表情（口形でないもの）。自動再生が回すのはこちらだけ。 */
  readonly expressions: readonly number[];
  /** 口形。連続再生が回すのはこちら。並びは焼いた順 = あいうえお。 */
  readonly visemes: readonly number[];
}

/**
 * プリセットを表情と口形へ分ける。
 *
 * **並びはアセットが正本。** 連続再生の順（あ→い→う→え→お）もアセットへ焼いた順そのもので、
 * ここで並べ直さない（並べ直すと `tools/viseme_presets.json` を触ったときに黙ってズレる）。
 */
export function splitPresetIndices(preview: GnmPreviewAsset): PresetSplit {
  const expressions: number[] = [];
  const visemes: number[] = [];
  preview.expressionPresetNames.forEach((name, index) => {
    if (isVisemePreset(name)) visemes.push(index);
    else expressions.push(index);
  });
  return { expressions, visemes };
}

/**
 * 口形の駆動のしかた。
 *
 * `off` では口形を一切立てない（そのとき顔を駆動するのは従来どおり表情の側）。表情の
 * `ExpressionPlayMode` へ混ぜないのは、**口形と表情が別の駆動源**だから — 1 つの列挙に混ぜると
 * 「表情を手で立てたまま口形を連続再生する」が型の上で表せなくなる。
 */
export type VisemeMode = 'off' | 'manual' | 'sequence';

/**
 * 連続再生の立ち上がりと保持（秒）。
 *
 * **表情の `FADE_SECONDS` / `HOLD_SECONDS` を使い回さない。** あちらは「どの表情か見せる」ための
 * 長さ（1 本あたり 1.5 秒）で、口形は連なって初めて言葉に見えるので桁が違う。ここの既定は
 * 1 音 0.32 秒 — あいうえおを 1 音ずつはっきり言うのと同じくらいの速さ。
 */
export const VISEME_FADE_SECONDS = 0.12;
export const VISEME_HOLD_SECONDS = 0.08;

/**
 * 連続再生の状態。`advanceVisemePlayback` が新しい状態を返す（保持は呼び側）。
 *
 * `expression.ts` の `ExpressionPlayback` と同じ作り。違うのは終端を持つことだけ — あちらは
 * 止まらないので「終わった」を表す必要が無い。
 */
export interface VisemePlayback {
  /**
   * 今かかっている口形の位置（口形の並びの中での添字）。
   *
   * まだ始まっていなければ -1、**ループ無しで最後まで行ったら口形の本数**。「始まる前」と
   * 「終わった後」を同じ -1 にすると、終端で止めたつもりが次のフレームで頭から再生し直す。
   */
  readonly index: number;
  readonly elapsedSeconds: number;
}

/** 再生前の状態。「先頭から再生」もここへ戻す。 */
export const IDLE_VISEME_PLAYBACK: VisemePlayback = { index: -1, elapsedSeconds: 0 };

/**
 * 連続再生を 1 フレーム進める。
 *
 * あ→い→う→え→お を台形エンベロープ（`expression.ts` の `envelope`）で 1 つずつ立てる。
 * **乱数を取らない** — 並びは決まっているので、`advancePlayback` の `pick` にあたるものが要らない。
 *
 * @param count 口形の本数（`splitPresetIndices` の `visemes.length`）
 * @returns `index` は今かかっている口形の位置。何もかかっていなければ -1
 */
export function advanceVisemePlayback(
  playback: VisemePlayback,
  count: number,
  deltaSeconds: number,
  loop = true,
  fadeSeconds = VISEME_FADE_SECONDS,
  holdSeconds = VISEME_HOLD_SECONDS,
): { playback: VisemePlayback; index: number; weight: number } {
  const stopped = { index: count, elapsedSeconds: 0 };
  if (count === 0) return { playback: IDLE_VISEME_PLAYBACK, index: -1, weight: 0 };
  if (playback.index >= count) return { playback: stopped, index: -1, weight: 0 };
  const cycle = fadeSeconds * 2 + holdSeconds;
  let { index, elapsedSeconds } = playback;
  if (index < 0 || elapsedSeconds >= cycle) {
    index += 1;
    if (index >= count) {
      if (!loop) return { playback: stopped, index: -1, weight: 0 };
      index = 0;
    }
    elapsedSeconds = 0;
  }
  return {
    playback: { index, elapsedSeconds: elapsedSeconds + deltaSeconds },
    index,
    weight: envelope(elapsedSeconds, fadeSeconds, holdSeconds),
  };
}
