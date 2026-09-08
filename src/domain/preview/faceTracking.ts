// カメラの顔から 3D ビューの表情を駆動する（純粋計算）。
//
// 入力は MediaPipe FaceLandmarker の **blendshape スコア**（ARKit 系のカテゴリ名 → 0〜1）。出力は
// `domain/preview/expression` が食う**プリセットの重み**（長さ `presetCount`）と**まばたき量**。
//
// **首・顔の向きは駆動しない。** 表情だけを写す。頭部姿勢はビューの `headPose`（手のスライダーと
// マウス追従）が持っており、そこへカメラの姿勢を足すと「どちらが今の向きを決めているか」が
// 画面から読めなくなる。
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
// `expression.ts` の冒頭にある通り、まばたきは目領域の**置き換え**（`blinkBasisQ`）で、プリセットの
// 加算とは別の経路。トラッキングでも同じ扱いにする — `eyeBlinkLeft` / `eyeBlinkRight` はプリセットへ
// 流さず `splitBlinkAndWink` でまばたき量とウィンクへ分ける。

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
}

/** 対応表をアセットのプリセットの並びへ解決する。 */
export function resolveTrackingPlan(
  presetNames: readonly string[],
  rows: readonly TrackingRow[] = TRACKING_ROWS,
): TrackingPlan {
  const resolved: { index: number; row: TrackingRow }[] = [];
  const unknownPresets: string[] = [];
  for (const row of rows) {
    const index = presetNames.indexOf(row.preset);
    if (index < 0) unknownPresets.push(row.preset);
    else resolved.push({ index, row });
  }
  const mapped = new Set(resolved.map((entry) => entry.row.preset));
  const categories = new Set<string>();
  for (const row of rows) for (const term of row.terms) categories.add(term.category);
  return {
    presetCount: presetNames.length,
    rows: resolved,
    unmappedPresets: presetNames.filter((name) => !mapped.has(name)),
    unknownPresets,
    categories: [...categories],
    presetNames,
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

/** デッドバンドを引いて 0〜1 へ引き伸ばす。 */
export function applyDeadband(score: number, deadband = CATEGORY_DEADBAND): number {
  if (!Number.isFinite(score) || score <= deadband) return 0;
  if (deadband >= 1) return 0;
  return Math.min(1, (score - deadband) / (1 - deadband));
}

/**
 * 重みの合計を上限まで比例で縮める（`weights` を破壊的に更新）。
 *
 * @returns 縮める前の合計
 */
export function limitTotalWeight(weights: Float64Array, maximum = MAX_TOTAL_WEIGHT): number {
  let total = 0;
  for (const weight of weights) total += weight;
  if (total > maximum && total > 0) {
    const scale = maximum / total;
    for (let index = 0; index < weights.length; index++) weights[index] *= scale;
  }
  return total;
}

/**
 * 左右のまばたき量を「両目のまばたき」と「ウィンク」へ分ける。
 *
 * 両目を同じだけ閉じたぶんはまばたき（目領域の置き換え）へ、左右差はウィンクのプリセットへ流す。
 * 差が `WINK_ASYMMETRY_THRESHOLD` 以下なら差は雑音とみなし、まばたきは左右の**平均**で駆動する
 * （min だと左右のわずかなズレのぶんだけ毎回閉じ切らない）。
 *
 * 閾値をまたぐところで飛ばないよう、`t`（0→1）で平均から min へ連続に移す:
 *
 *     blink = 平均 − t × 差 / 2   （t=1 で min に一致）
 *     wink  = t × 差
 */
export function splitBlinkAndWink(
  left: number,
  right: number,
  threshold = WINK_ASYMMETRY_THRESHOLD,
): { blink: number; winkLeft: number; winkRight: number } {
  const clampedLeft = clamp01(left);
  const clampedRight = clamp01(right);
  const asymmetry = Math.abs(clampedLeft - clampedRight);
  const average = (clampedLeft + clampedRight) / 2;
  const t = threshold >= 1 ? 0 : clamp01((asymmetry - threshold) / (1 - threshold));
  const wink = t * asymmetry;
  return {
    blink: clamp01(average - (t * asymmetry) / 2),
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
): TrackingTargets {
  if (weights.length !== plan.presetCount) {
    throw new Error(`重みが ${weights.length} 個（期待 ${plan.presetCount}）`);
  }
  weights.fill(0);
  for (const { index, row } of plan.rows) {
    let value = 0;
    for (const term of row.terms) {
      value += applyDeadband(scores.get(term.category) ?? 0, deadband) * term.gain;
    }
    weights[index] = clamp01(value);
  }

  // まばたきは加算のプリセットではないので、対応表とは別に扱う。
  const blinkSplit = splitBlinkAndWink(
    applyDeadband(scores.get('eyeBlinkLeft') ?? 0, deadband),
    applyDeadband(scores.get('eyeBlinkRight') ?? 0, deadband),
  );
  setPreset(plan, weights, 'wink_left', blinkSplit.winkLeft);
  setPreset(plan, weights, 'wink_right', blinkSplit.winkRight);

  const rawTotal = limitTotalWeight(weights);
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
