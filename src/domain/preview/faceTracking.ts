// カメラの顔から 3D ビューの表情を駆動する（純粋計算）。
//
// 入力は MediaPipe FaceLandmarker の **blendshape スコア**（ARKit 系のカテゴリ名 → 0〜1）。出力は
// **プリセットの重み**（長さ `presetCount`）と**まばたき量**で、呼ぶ側がそれを表情基底 383 成分の
// 係数へ畳んでから顔へ当てる（`domain/preview/expression.addPresetCoefficients`）。
//
// **点から係数を直接解く形も作ったが、実機で負けたのでやめた。** 合成観測では旧実装より良い数字が
// 出たのに、カメラの前では顔がぐにゃぐにゃ動いた。検証の残骸は
// `tools/experiments/expression_fit_feasibility.py` と git log にある（消した実装は
// `src/domain/preview/expressionFit.ts`）。
//
// **首・顔の向きは既定では駆動しない。** 表情だけを写す。頭部姿勢はビューの `headPose`（手の
// スライダーとマウス追従）が持っており、そこへ黙ってカメラの姿勢を足すと「どちらが今の向きを
// 決めているか」が画面から読めなくなる。**明示的に選んだときだけ**首も動かす
// （`headPoseFromMatrix`）— どちらで動いているかは画面のスイッチが示す。
//
// ## 対応表は名前ではなく意味で作る
//
// ARKit 52 とプリセット 20 本は別々に作られたもので、1 対 1 では対応しない。**片方に無い動きが
// ある**（例: `suck` は頬をすぼめる動きで、ARKit には `cheekPuff` の逆向きが無い）。無い対応を
// 近そうな項目で埋めると、**別の表情のときに勝手に立つ**方が害が大きいので埋めない。どのプリセットが
// 駆動されないかは `TrackingPlan.unmappedPresets` に出る。
//
// **カテゴリ名の一覧をここへ焼かない。** 実行時に返ってくる名前が正本で、表に書いた名前が返って
// こなければ 0 として扱う（`missingCategories` に出るので開発者は気付ける）。
//
// ## まばたきは表情プリセットではない
//
// `expression.ts` の冒頭にある通り、まばたきは目の成分の**置き換え**で、プリセットの加算とは別の
// 経路。トラッキングでも同じ扱いにする — `eyeBlinkLeft` / `eyeBlinkRight` はプリセットへ流さず
// `splitBlinkAndWink` でまばたき量とウィンクへ分ける。
//
// 分けた後の混ぜ込みは**呼ぶ側が係数の上で行う**（`presentation/webcamPanel`）。ビューアーへ別の口で
// 渡すと、録るのが混ぜる前の係数になって録画からまばたきが落ちる。

/** 対応表の 1 項（カテゴリ名と、そのスコアへ掛ける係数）。 */
export interface TrackingTerm {
  /** MediaPipe が返す blendshape のカテゴリ名。 */
  readonly category: string;
  /** 係数。左右対の項は 0.5 ずつにして、両側が満点で重み 1 になるようにする。 */
  readonly gain: number;
}

/** プリセット 1 本を何から作るか。 */
export interface TrackingRow {
  /** 表情プリセット名（正本はアセットの `expressionPresetNames`）。 */
  readonly preset: string;
  readonly terms: readonly TrackingTerm[];
  /** なぜその対応なのか。**対応の理由が自明でないものには必ず書く。** */
  readonly reason: string;
}

/**
 * ウィンクのプリセット名。**対応表には載せない** — `splitBlinkAndWink` が blendshape から直に
 * 立てるもので、カテゴリ 1 つに対応する動きではない。
 */
export const WINK_PRESETS: readonly string[] = ['wink_left', 'wink_right'];

/**
 * 左右を入れ替えるプリセットの対。
 *
 * **画面のミラー表示と揃える。** 映像は左右反転して出しているので、利用者が「映像の左に見える目」
 * を閉じたら、3D の頭も**見た目の左**が閉じる方が合わせやすい。解剖学的な左右（`eyeBlinkLeft` は
 * 被写体の左目 / `wink_left` は頭の左）で素直に繋ぐと、画面上では逆に出る — **実機で
 * 「右目を動かすと CG の左が動く」と見えていたのがこれ**。yaw を鏡にしているのと同じ理由。
 */
export const MIRRORED_PRESET_PAIRS: readonly (readonly [string, string])[] = [
  ['wink_left', 'wink_right'],
  ['mouth_left', 'mouth_right'],
];

/** 左右を入れ替えて出すか。false にすると解剖学的な左右へ揃う。 */
export const MIRROR_SIDES = true;

/** 左右を入れ替えたプリセット名（対に無い名前はそのまま）。 */
export function mirroredPreset(name: string, mirror = MIRROR_SIDES): string {
  if (!mirror) return name;
  for (const [left, right] of MIRRORED_PRESET_PAIRS) {
    if (name === left) return right;
    if (name === right) return left;
  }
  return name;
}

/** 左右対のカテゴリを 0.5 ずつで 1 本にする（両側満点で 1）。 */
function pair(left: string, right: string, gain = 1): TrackingTerm[] {
  return [
    { category: left, gain: gain / 2 },
    { category: right, gain: gain / 2 },
  ];
}

/**
 * blendshape → プリセットの対応表。
 *
 * **`eyeBlinkLeft` / `eyeBlinkRight` はここに無い。** まばたきとウィンクは `splitBlinkAndWink` が
 * 別に扱う（上のコメント）。`wink_left` / `wink_right` もそちらが立てるのでここには置かない。
 */
export const TRACKING_ROWS: readonly TrackingRow[] = [
  {
    preset: 'surprise',
    terms: [
      { category: 'browInnerUp', gain: 0.5 },
      ...pair('browOuterUpLeft', 'browOuterUpRight', 0.5),
    ],
    reason:
      '驚きの顔で動くのは眉。眉の内側と外側が両方上がって重み 1 になるよう半分ずつ配る。' +
      '`eyeWide*`（見開き）は 20 本の中に対応する変位が無いので使わない',
  },
  {
    preset: 'disgust',
    terms: pair('browDownLeft', 'browDownRight'),
    reason:
      'ARKit に怒り・しかめ面の項目は無く、眉を下げる `browDown*` が唯一の「眉を寄せる」信号。' +
      '20 本の中でその顔に最も近いのが disgust',
  },
  {
    preset: 'compress_face',
    terms: pair('mouthPressLeft', 'mouthPressRight'),
    reason: '唇を押し付けて顔を縮める動き。`mouthPress*` がそのまま「口を圧する」',
  },
  {
    preset: 'stretch_face',
    terms: [{ category: 'jawOpen', gain: 1 }],
    reason:
      '顎が落ちると顔が縦に伸びる。**開口はこのプリセットが担う** — 20 本に「口を開く」単独の' +
      'プリセットは無く、縦に伸びる変位を持つのが stretch_face だけ',
  },
  {
    preset: 'happy',
    terms: pair('mouthSmileLeft', 'mouthSmileRight'),
    reason: '口角が上がる = 笑み',
  },
  {
    preset: 'smile_wide',
    terms: pair('mouthStretchLeft', 'mouthStretchRight'),
    reason:
      'ARKit は「口角が上がる」(`mouthSmile*`) と「口角が横へ引かれる」(`mouthStretch*`) を分けて' +
      'いる。アセットも happy と smile_wide で分かれているので、同じ分け方で当てる' +
      '（両方を `mouthSmile*` から作ると笑うたびに 2 本が重なって口が壊れる）',
  },
  {
    preset: 'corners_down',
    terms: pair('mouthFrownLeft', 'mouthFrownRight'),
    reason: '口角が下がる',
  },
  {
    preset: 'squint',
    terms: pair('eyeSquintLeft', 'eyeSquintRight'),
    reason:
      '目を細める。**`eyeBlink*` とは別の項目**で、こちらは瞼を閉じ切らないのでプリセット側へ流す',
  },
  {
    preset: 'platysma',
    terms: pair('mouthLowerDownLeft', 'mouthLowerDownRight'),
    reason:
      '広頸筋が張ると下唇が下へ引かれて下の歯が見える。ARKit でその動きは `mouthLowerDown*`',
  },
  {
    preset: 'blow',
    terms: [{ category: 'cheekPuff', gain: 1 }],
    reason: '頬を膨らませる',
  },
  {
    preset: 'funneler',
    terms: [{ category: 'mouthFunnel', gain: 1 }],
    reason: '唇を漏斗状に開く',
  },
  {
    preset: 'pucker',
    terms: [{ category: 'mouthPucker', gain: 1 }],
    reason: '唇をすぼめる',
  },
  {
    preset: 'lips_roll_in',
    terms: pair('mouthRollUpper', 'mouthRollLower'),
    reason: '上下の唇を巻き込む。左右ではなく上下の対だが、半分ずつ配る扱いは同じ',
  },
  {
    preset: 'snarl',
    terms: [
      ...pair('noseSneerLeft', 'noseSneerRight', 0.7),
      ...pair('mouthUpperUpLeft', 'mouthUpperUpRight', 0.3),
    ],
    reason:
      '歯をむく顔は「鼻に皺を寄せる」+「上唇を持ち上げる」の合成。鼻の方が特徴的なので 0.7 / 0.3',
  },
  {
    preset: 'mouth_left',
    terms: [
      { category: 'mouthLeft', gain: 0.7 },
      { category: 'jawLeft', gain: 0.3 },
    ],
    reason:
      '口を左へ寄せる。顎が同じ側へずれることが多いので `jawLeft` も足す' +
      '（左右の意味は ARKit と同じく**被写体の左**。ミラー表示の画面上では右に見える）',
  },
  {
    preset: 'mouth_right',
    terms: [
      { category: 'mouthRight', gain: 0.7 },
      { category: 'jawRight', gain: 0.3 },
    ],
    reason: '`mouth_left` の対',
  },
  {
    preset: 'tongue_center',
    terms: [{ category: 'tongueOut', gain: 1 }],
    reason: '舌を出す（正面へ出す 1 本しか無いので中央だけ）',
  },
];

/**
 * カテゴリのスコアがここ以下なら 0 として扱う（超えたぶんは 0〜1 へ引き伸ばす）。
 *
 * **無表情でも MediaPipe は 52 カテゴリすべてに小さな値を返す。** 素通しすると全プリセットが常時
 * 薄く立ち、顔が眠たくぼやける。0.08 は「無表情で観測される揺らぎ」より上、「意図した弱い表情」より
 * 下に置いた値。**引き伸ばす**のは、切るだけだと最大でも 1 - 0.08 までしか届かないため。
 */
export const CATEGORY_DEADBAND = 0.08;

/**
 * デッドバンドの後に掛ける倍率（追従の強さ）。
 *
 * **MediaPipe のスコアは滅多に 1.0 へ届かない。** はっきり作った表情でも 0.4〜0.7 あたりで、
 * 素通しするとプリセットの重みがその値のままになり、**顔がほとんど動かない**（実機でそう見えて
 * いた）。1.8 は「はっきり作った表情でプリセットが満点近くまで立つ」線。上げすぎると無表情の
 * 揺らぎまで拾うので、画面のスライダーで触れるようにしてある。
 */
export const DEFAULT_TRACKING_GAIN = 1.8;
export const MINIMUM_TRACKING_GAIN = 0.5;
export const MAXIMUM_TRACKING_GAIN = 4;

/**
 * プリセットの重みの合計の上限。
 *
 * プリセットは**加算変位**なので重ねると顔が壊れる（`expression.ts` 冒頭）。自動再生は同時に 1 本
 * しか立てないが、**カメラは必ず複数本を同時に立てる**（笑いながら喋れば happy + stretch_face）ので
 * 1.0 で切ると自然な顔が作れない。合計が上限を超えたら**全体を比例で縮める**（どれかを切り捨てると
 * 表情の釣り合いが変わる）。1.5 は「同時に 1 本」の 1.5 倍までを許す線で、実測ではなく設計上の線。
 */
export const MAX_TOTAL_WEIGHT = 1.5;

/**
 * 左右のまばたき量の差がこれ以下なら「両目のまばたき」として扱う。
 *
 * 両目を閉じても推定は左右でズレる（照明・角度・髪）。閾値が無いとまばたきのたびに wink が薄く
 * 立って片目だけ形が変わる。0.25 は「同時に閉じたときの左右差」より上、「片目だけ閉じたとき」
 * （差が 0.7 以上出る）より十分下。
 */
export const WINK_ASYMMETRY_THRESHOLD = 0.25;

/**
 * 左右差がここまで開いたら**完全にウィンク**として扱う（まばたきは 0 にする）。
 *
 * **1.0 にしてはいけない。** MediaPipe は片目を閉じたとき開いている側にも 0.1〜0.3 を返す
 * （クロストーク）ので、左右差は 1.0 に届かない。上限を 1.0 に置くと、実際のウィンクでも
 * まばたきが 0.2〜0.25 残り、**両目が薄く閉じる**（実際にそう見えていた）。
 *
 * まばたきが残ると害が二重になる — `blendBlink` は目領域を**置き換える**ので、まばたきが立つと
 * その割合だけウィンクのプリセットが打ち消される（`expression.ts` 冒頭の「加算にすると開瞼系と
 * 打ち消し合う」と同じ理屈）。片目を閉じたのに両目が薄く閉じ、しかもウィンクが弱くなる。
 */
export const WINK_FULL_ASYMMETRY = 0.5;

/**
 * 表情の一次遅れの時定数（秒）。
 *
 * カメラの推定はフレームごとに跳ねるので、そのまま当てると顔が震える。60ms は 30fps の 2 フレーム弱で、
 * 跳ねを均しつつ**遅れを 100ms 以下に収める**線（自分の顔を映すと 100ms を超えたあたりから遅れとして
 * 見え始める）。
 */
export const EXPRESSION_TIME_CONSTANT_SECONDS = 0.06;

/**
 * まばたきの時定数（秒）。**表情より短くする。**
 *
 * まばたき 1 回は `expression.BLINK_DURATION_MIN_MS`〜`BLINK_DURATION_MAX_MS`（150〜250ms）しか
 * 続かない。表情と同じ 60ms で均すと山が削れて瞼が閉じ切らず、`blendBlink` の置き換えが効かない。
 */
export const BLINK_TIME_CONSTANT_SECONDS = 0.02;

/** 読み出しに名前を出す下限。これ未満しか立っていなければ「表情なし」とする。 */
export const READOUT_MINIMUM_WEIGHT = 0.15;

/** アセットのプリセットの並びへ解決した対応表。**毎フレーム名前で引かないための前計算。** */
export interface TrackingPlan {
  readonly presetCount: number;
  /** `TRACKING_ROWS` のうちアセットに在るものを、プリセット index 付きで持つ。 */
  readonly rows: readonly { readonly index: number; readonly row: TrackingRow }[];
  /** アセットに在るが対応表に無いプリセット名（カメラでは駆動されない）。 */
  readonly unmappedPresets: readonly string[];
  /** 対応表に在るがアセットに無いプリセット名（表が古い）。 */
  readonly unknownPresets: readonly string[];
  /** 対応表が参照するカテゴリ名の全体。 */
  readonly categories: readonly string[];
  /** プリセット名（`strongestPreset` の読み出し用）。 */
  readonly presetNames: readonly string[];
  /** 対応表が動かすプリセットの index（ウィンクを含まない）。合計の上限をここへ掛ける。 */
  readonly mouthIndices: readonly number[];
  /** ウィンクのプリセットの index。上限を別に持つ（口の予算に巻き込まれないように）。 */
  readonly winkIndices: readonly number[];
  /** 左右を入れ替えて出しているか。 */
  readonly mirror: boolean;
}

/** 対応表をアセットのプリセットの並びへ解決する。 */
export function resolveTrackingPlan(
  presetNames: readonly string[],
  rows: readonly TrackingRow[] = TRACKING_ROWS,
  mirror = MIRROR_SIDES,
): TrackingPlan {
  const resolved: { index: number; row: TrackingRow }[] = [];
  const unknownPresets: string[] = [];
  for (const row of rows) {
    // **左右は鏡で解決する。** 対応表は解剖学的な左右で書き、出す先だけ入れ替える（表の側を
    // 書き換えると「どちらの左右で書いてあるか」が読めなくなる）。
    const index = presetNames.indexOf(mirroredPreset(row.preset, mirror));
    if (index < 0) unknownPresets.push(row.preset);
    else resolved.push({ index, row });
  }
  // **ウィンクは対応表に無いが駆動される。** `blendshapesToTargets` が `splitBlinkAndWink` から
  // 直に立てるので、「駆動されない」の一覧へ入れると診断が嘘になる。
  const mapped = new Set([...resolved.map((entry) => entry.row.preset), ...WINK_PRESETS]);
  const categories = new Set<string>();
  for (const row of rows) for (const term of row.terms) categories.add(term.category);
  return {
    presetCount: presetNames.length,
    rows: resolved,
    unmappedPresets: presetNames.filter((name) => !mapped.has(name)),
    unknownPresets,
    categories: [...categories],
    presetNames,
    mouthIndices: resolved.map((entry) => entry.index),
    winkIndices: WINK_PRESETS.map((name) =>
      presetNames.indexOf(mirroredPreset(name, mirror)),
    ).filter((index) => index >= 0),
    mirror,
  };
}

/**
 * 対応表が求めるカテゴリのうち、実行時に返ってこなかったものを挙げる。
 *
 * **毎フレーム呼ぶものではない**（最初の 1 フレームで一度だけ見て開発者へ知らせる用）。無い名前は
 * 0 として扱われるので、黙って「そのプリセットだけ立たない」になるのを防ぐ。
 */
export function missingCategories(
  plan: TrackingPlan,
  available: Iterable<string>,
): readonly string[] {
  const present = new Set(available);
  return plan.categories.filter((category) => !present.has(category));
}

/** デッドバンドを引いて 0〜1 へ引き伸ばし、追従の強さを掛ける。 */
export function applyDeadband(
  score: number,
  deadband = CATEGORY_DEADBAND,
  gain = DEFAULT_TRACKING_GAIN,
): number {
  if (!Number.isFinite(score) || score <= deadband) return 0;
  if (deadband >= 1) return 0;
  return Math.min(1, ((score - deadband) / (1 - deadband)) * gain);
}

/**
 * 重みの合計を上限まで比例で縮める（`weights` を破壊的に更新）。
 *
 * `only` を渡すと、その index だけを見て、その index だけを縮める。
 *
 * @returns 縮める前の合計
 */
export function limitTotalWeight(
  weights: Float64Array,
  maximum = MAX_TOTAL_WEIGHT,
  only?: readonly number[],
): number {
  const indices = only ?? Array.from({ length: weights.length }, (_, index) => index);
  let total = 0;
  for (const index of indices) total += weights[index];
  if (total > maximum && total > 0) {
    const scale = maximum / total;
    for (const index of indices) weights[index] *= scale;
  }
  return total;
}

/**
 * ウィンクに使える重みの上限。**口の予算とは別に持つ。**
 *
 * 上限を全プリセットで 1 つにすると、喋りながら片目を閉じたときにウィンクが道連れで縮む
 * （口が開くと `stretch_face` などで予算が埋まり、比例縮小がウィンクにも掛かる）。**実際に
 * 「片目を閉じても閉じない」と見えていた原因がこれ。** 予算を分ける根拠は、加算変位が壊すのは
 * 同じ領域を動かし合ったときで、目と口は別の領域だから。
 */
export const MAX_WINK_WEIGHT = 1;

/**
 * 左右のまばたき量を「両目のまばたき」と「ウィンク」へ分ける。
 *
 * **左右対称ならまばたき、非対称ならウィンク**、という切り分け。差が
 * `WINK_ASYMMETRY_THRESHOLD` 以下なら雑音とみなしてまばたきだけを左右の**平均**で駆動し
 * （min だと左右のわずかなズレのぶんだけ毎回閉じ切らない）、`WINK_FULL_ASYMMETRY` まで開いたら
 * **まばたきは 0** にしてウィンクだけを立てる:
 *
 *     blink = 平均 × (1 − t)
 *     wink  = t × 差
 *
 * **min（共通ぶん）をまばたきにしない。** 片目を閉じたとき開いている側にもクロストークで
 * 0.1〜0.3 が乗るので、min を残すと両目が薄く閉じたままになり、しかも `blendBlink` の置き換えで
 * ウィンクのプリセットまで打ち消される。開いている側の残りは**雑音として捨てる**方が絵が合う。
 *
 * `t` は閾値をまたぐところで飛ばないよう連続に上げる。
 */
export function splitBlinkAndWink(
  left: number,
  right: number,
  threshold = WINK_ASYMMETRY_THRESHOLD,
  fullAsymmetry = WINK_FULL_ASYMMETRY,
): { blink: number; winkLeft: number; winkRight: number } {
  const clampedLeft = clamp01(left);
  const clampedRight = clamp01(right);
  const asymmetry = Math.abs(clampedLeft - clampedRight);
  const average = (clampedLeft + clampedRight) / 2;
  const span = fullAsymmetry - threshold;
  const t = span <= 0 ? (asymmetry > threshold ? 1 : 0) : clamp01((asymmetry - threshold) / span);
  const wink = t * asymmetry;
  return {
    blink: clamp01(average * (1 - t)),
    winkLeft: clampedLeft > clampedRight ? wink : 0,
    winkRight: clampedRight > clampedLeft ? wink : 0,
  };
}

/** 1 フレームぶんの目標値。 */
export interface TrackingTargets {
  /** 目標のまばたき量（0〜1）。 */
  readonly blink: number;
  /** 上限で縮める前の重みの合計（どれだけ重なったかの目安）。 */
  readonly rawTotal: number;
}

/**
 * blendshape のスコアからプリセットの重みとまばたき量を作る。
 *
 * `weights` は破壊的に埋める（長さ `plan.presetCount`）。**呼ぶ側は毎フレーム同じ配列を渡してよい**
 * （中で 0 埋めしてから書く）。返り値のまばたき量は `blinkOverride` へ流す。
 *
 * 知らないカテゴリ名は 0 として扱う（無い名前で落とさない）。
 */
export function blendshapesToTargets(
  plan: TrackingPlan,
  scores: ReadonlyMap<string, number>,
  weights: Float64Array,
  deadband = CATEGORY_DEADBAND,
  gain = DEFAULT_TRACKING_GAIN,
): TrackingTargets {
  if (weights.length !== plan.presetCount) {
    throw new Error(`重みが ${weights.length} 個（期待 ${plan.presetCount}）`);
  }
  weights.fill(0);
  for (const { index, row } of plan.rows) {
    let value = 0;
    for (const term of row.terms) {
      value += applyDeadband(scores.get(term.category) ?? 0, deadband, gain) * term.gain;
    }
    weights[index] = clamp01(value);
  }

  // まばたきは加算のプリセットではないので、対応表とは別に扱う。
  const blinkSplit = splitBlinkAndWink(
    applyDeadband(scores.get('eyeBlinkLeft') ?? 0, deadband, gain),
    applyDeadband(scores.get('eyeBlinkRight') ?? 0, deadband, gain),
  );
  // **ウィンクは対応表の後に置き、予算も別で持つ。** 先に口の合計を抑えてから、目の枠で抑える。
  const rawTotal = limitTotalWeight(weights, MAX_TOTAL_WEIGHT, plan.mouthIndices);
  setPreset(plan, weights, mirroredPreset(WINK_PRESETS[0], plan.mirror), blinkSplit.winkLeft);
  setPreset(plan, weights, mirroredPreset(WINK_PRESETS[1], plan.mirror), blinkSplit.winkRight);
  limitTotalWeight(weights, MAX_WINK_WEIGHT, plan.winkIndices);
  return { blink: blinkSplit.blink, rawTotal };
}

function setPreset(
  plan: TrackingPlan,
  weights: Float64Array,
  name: string,
  weight: number,
): void {
  const index = plan.presetNames.indexOf(name);
  if (index >= 0) weights[index] = weight;
}

/**
 * 一次遅れの係数。**フレーム間隔に依らない**形（`1 - exp(-dt/tau)`）で作る。
 *
 * requestAnimationFrame は等間隔で来ないので、固定の係数（`y += (x-y) * 0.2` の類）だと重い
 * フレームでだけ余計に追従して跳ねる。
 */
export function smoothingFactor(deltaSeconds: number, timeConstantSeconds: number): number {
  if (!(deltaSeconds > 0)) return 0;
  if (!(timeConstantSeconds > 0)) return 1;
  return 1 - Math.exp(-deltaSeconds / timeConstantSeconds);
}

/** `current` を `target` へ寄せる（破壊的）。 */
export function smoothToward(
  current: Float64Array,
  target: Float64Array,
  factor: number,
): void {
  for (let index = 0; index < current.length; index++) {
    current[index] += (target[index] - current[index]) * factor;
  }
}

/** スカラー版。 */
export function smoothScalar(current: number, target: number, factor: number): number {
  return current + (target - current) * factor;
}

/** いちばん強く立っているプリセット名（弱ければ null）。読み出しに出すだけ。 */
export function strongestPreset(
  plan: TrackingPlan,
  weights: Float64Array,
  minimum = READOUT_MINIMUM_WEIGHT,
): string | null {
  let best = -1;
  let bestWeight = minimum;
  for (let index = 0; index < weights.length; index++) {
    if (weights[index] > bestWeight) {
      bestWeight = weights[index];
      best = index;
    }
  }
  return best < 0 ? null : plan.presetNames[best];
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

/**
 * 首の追従の時定数（秒）。**表情より長くする。**
 *
 * 首は表情より大きく、ゆっくり動く。表情と同じ 60ms で追うと、検出の跳ねがそのまま首の震えに
 * なって酔う。
 */
export const HEAD_TIME_CONSTANT_SECONDS = 0.12;

/**
 * カメラで振れる首の範囲（度）。**リグの可動域より広く取る。**
 *
 * リグの可動域は Unity 側の首 ±15° / pitch ±12°（`domain/preview/pose`）。人が実際に首を振る幅は
 * それよりずっと広いので、生の角度をそのまま渡すと**すぐ上限に張り付いて壁に当たった感じになる**
 * （実際にそう見えていた）。ここの幅をリグの可動域へ**線形に写す**ので、大きく振っても端で
 * 止まらず、range いっぱいまで滑らかに動く。
 */
export const HEAD_TRACKING_YAW_RANGE_DEGREES = 35;
export const HEAD_TRACKING_PITCH_RANGE_DEGREES = 25;

/**
 * MediaPipe の頭の姿勢行列から yaw / pitch（度）を取り出す。
 *
 * **行優先の 4x4 として読む**（`facialTransformationMatrixes[].data` の並び）。回転部を
 * `Ry(yaw) * Rx(pitch)` として分解する — 正面で (0, 0)、可動域へ入れるのは呼ぶ側
 * （`domain/preview/pose.clampPose`）。
 *
 * **ロールは捨てる。** ビューの `HeadPose` が持たないし、確認用途で首を傾ける場面が無い。
 *
 * **左右の向きは鏡にする。** 利用者は自分を映した画面を見ながら 3D の頭を合わせるので、自分が
 * 右を向いたら画面の頭も画面の右へ動く方が合わせやすい（映像をミラー表示しているのと同じ理由）。
 * `mirrorYaw` を false にすれば実際の向きに揃う。
 *
 * **pitch は符号を返す。** 行列から出る pitch とビューの `HeadPose.headPitchDegrees` は上下が逆で、
 * そのまま渡すと**上を向いたら下を向く**（実機でそう見えていた）。
 */
export const MIRROR_YAW = true;

export function headPoseFromMatrix(
  matrix: Float32Array,
  mirrorYaw = MIRROR_YAW,
): { yawDegrees: number; pitchDegrees: number } | null {
  if (matrix.length < 16) return null;
  // 行優先: m[行][列] = matrix[行 * 4 + 列]。
  const m01 = matrix[1];
  const m02 = matrix[2];
  const m11 = matrix[5];
  const m12 = matrix[6];
  const m22 = matrix[10];
  // 回転行列なら列の長さは 1。全ゼロや壊れた行列をここで落とす（atan2 は 0 を返してしまう）。
  const firstColumn = Math.hypot(matrix[0], matrix[4], matrix[8]);
  if (!Number.isFinite(firstColumn) || firstColumn < 0.5) return null;
  const pitch = Math.atan2(-m12, m11);
  const yaw = Math.atan2(m02, m22);
  if (!Number.isFinite(pitch) || !Number.isFinite(yaw)) return null;
  // 使わないが、分解の前提（`Ry * Rx` にロールが混じっていない）を壊した行列を黙って通さない。
  if (!Number.isFinite(m01)) return null;
  const degrees = 180 / Math.PI;
  return {
    yawDegrees: yaw * degrees * (mirrorYaw ? -1 : 1),
    pitchDegrees: -pitch * degrees,
  };
}

/**
 * カメラの首の角度をリグの可動域へ写す。
 *
 * `range` を可動域（`limit`）へ**線形に**写し、外はクランプする。生の角度をそのまま渡すと、人の
 * 首の振り幅（±35° 程度）に対して可動域が ±15° しかないので、少し振っただけで上限に張り付く。
 */
export function mapHeadAngle(degrees: number, range: number, limit: number): number {
  if (!(range > 0)) return 0;
  const scaled = (degrees / range) * limit;
  return Math.min(limit, Math.max(-limit, scaled));
}
