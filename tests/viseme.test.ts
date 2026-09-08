// あいうえおの口形（viseme）の検査。
//
// **実アセットで測る。** 口形はアセットへ焼いた 1 本のプリセットなので、「意図した口になっているか」
// は焼いた変位でしか確かめられない（合成データで測っても、焼く段が壊れたことを検出できない）。
//
// 測るのは 4 つの量で、`tools/viseme_presets.json` の重みを決めたときと同じ測り方:
//
// | 量 | 測り方 | 向き |
// |:--|:--|:--|
// | 開き | 下唇の y − 上唇の y の平均 | 負で開く |
// | 顎 | `chin_region` の y の平均 | 負で落ちる |
// | 突き出し | 唇の z の平均 | 正で前 |
// | 横の広がり | 唇の x の標準偏差 | 大きいほど横に広い |
//
// **しきい値に絶対値をほとんど書かない。** 口形どうしの**順序**（あ が一番開く / う が一番前へ出る /
// い が一番横に広い）で見る。絶対値を書くと、公式のクラスが焼き直されたときに「意味は保たれている
// のに落ちる」検査になる。順序なら、崩れたときだけ落ちる。

import { describe, expect, it } from 'vitest';
import { loadPreview } from './asset';
import { GnmPreviewAsset, evaluateSelector, presetDisplacement } from '../src/domain/preview/asset';
import { addExpression, weightsFor } from '../src/domain/preview/expression';
import {
  IDLE_VISEME_PLAYBACK,
  VISEME_FADE_SECONDS,
  VISEME_HOLD_SECONDS,
  VISEME_LABELS,
  VISEME_PREFIX,
  VisemePlayback,
  advanceVisemePlayback,
  isVisemePreset,
  splitPresetIndices,
  visemeLabel,
} from '../src/domain/preview/viseme';

function indicesOf(mask: Uint8Array): number[] {
  const list: number[] = [];
  for (let vertex = 0; vertex < mask.length; vertex++) if (mask[vertex] !== 0) list.push(vertex);
  return list;
}

/** 唇まわりの 4 つの量（mm）。 */
interface MouthShape {
  readonly open: number;
  readonly jaw: number;
  readonly protrusion: number;
  readonly spread: number;
}

function mouthShape(preview: GnmPreviewAsset, preset: number): MouthShape {
  const upper = indicesOf(evaluateSelector(preview, ['upper_lip']));
  const lower = indicesOf(evaluateSelector(preview, ['lower_lip']));
  const chin = indicesOf(evaluateSelector(preview, ['chin_region']));
  const lips = indicesOf(evaluateSelector(preview, ['upper_lip', 'lower_lip']));
  const mean = (list: readonly number[], axis: number): number =>
    list.reduce((total, vertex) => total + presetDisplacement(preview, preset, vertex, axis), 0) /
    list.length;
  const lipsX = lips.map((vertex) => presetDisplacement(preview, preset, vertex, 0));
  const centre = lipsX.reduce((total, value) => total + value, 0) / lipsX.length;
  const variance =
    lipsX.reduce((total, value) => total + (value - centre) ** 2, 0) / lipsX.length;
  return {
    open: (mean(lower, 1) - mean(upper, 1)) * 1000,
    jaw: mean(chin, 1) * 1000,
    protrusion: mean(lips, 2) * 1000,
    spread: Math.sqrt(variance) * 1000,
  };
}

/** 焼いた口形を、あいうえおの並び（= 焼いた順）で名前 → 形にする。 */
function visemeShapes(preview: GnmPreviewAsset): Map<string, MouthShape> {
  const shapes = new Map<string, MouthShape>();
  for (const preset of splitPresetIndices(preview).visemes) {
    shapes.set(preview.expressionPresetNames[preset], mouthShape(preview, preset));
  }
  return shapes;
}

describe('口形プリセット（アセットに焼いてある）', () => {
  it('あいうえおの 5 本が焼かれていて、並びは あ→い→う→え→お', () => {
    const preview = loadPreview();
    const { visemes } = splitPresetIndices(preview);
    const names = visemes.map((preset) => preview.expressionPresetNames[preset]);
    // 並びは `tools/viseme_presets.json` の並び = 連続再生の順。
    expect(names).toEqual(['viseme_a', 'viseme_i', 'viseme_u', 'viseme_e', 'viseme_o']);
  });

  it('名前の頭が焼く側と揃っている（ズレると表情の一覧へ紛れ込む）', () => {
    const preview = loadPreview();
    // 正本は `tools/export_gnm_assets.py` の VISEME_PRESET_PREFIX。
    expect(VISEME_PREFIX).toBe('viseme_');
    for (const name of preview.expressionPresetNames) {
      // 表情 20 本の側にこの頭を持つものが混ざっていないこと。
      if (!isVisemePreset(name)) expect(name.startsWith(VISEME_PREFIX)).toBe(false);
    }
  });

  it('焼いた口形には日本語のラベルがある（無いと画面に生の名前が出る）', () => {
    const preview = loadPreview();
    for (const preset of splitPresetIndices(preview).visemes) {
      const name = preview.expressionPresetNames[preset];
      expect(VISEME_LABELS[name], name).toBeDefined();
      expect(visemeLabel(name)).not.toBe(name);
    }
  });

  it('表情プリセットの index は口形を足しても動かない（口形は後ろへ足す）', () => {
    const preview = loadPreview();
    const { expressions } = splitPresetIndices(preview);
    expect(expressions).toEqual(expressions.map((_, index) => index));
  });

  it('あ が一番開き、う が一番閉じる（顎も あ が一番落ちる）', () => {
    const shapes = visemeShapes(loadPreview());
    const open = (name: string): number => shapes.get(name)!.open;
    // 開きは負で開く。日本語の母音は あ > お > え > い > う の順に開く。
    expect(open('viseme_a')).toBeLessThan(open('viseme_o'));
    expect(open('viseme_o')).toBeLessThan(open('viseme_e'));
    expect(open('viseme_e')).toBeLessThan(open('viseme_i'));
    expect(open('viseme_i')).toBeLessThan(open('viseme_u'));
    // 顎が落ちるのは あ（1 音だけ立てたときに口が開いて見える根拠）。
    const jaw = (name: string): number => shapes.get(name)!.jaw;
    for (const name of ['viseme_i', 'viseme_u', 'viseme_e', 'viseme_o']) {
      expect(jaw('viseme_a'), name).toBeLessThan(jaw(name));
    }
  });

  it('丸める母音（う・お）だけが唇を前へ出す', () => {
    const shapes = visemeShapes(loadPreview());
    const forward = (name: string): number => shapes.get(name)!.protrusion;
    // う が最も前。お も前だが う より浅い（日本語の お は英語ほど丸めない）。
    expect(forward('viseme_u')).toBeGreaterThan(forward('viseme_o'));
    expect(forward('viseme_o')).toBeGreaterThan(0);
    // 丸めない母音は前へ出ない（引っ込む）。
    for (const name of ['viseme_a', 'viseme_i', 'viseme_e']) {
      expect(forward(name), name).toBeLessThan(0);
    }
  });

  it('い が一番横に広がり、あ が一番広がらない', () => {
    const shapes = visemeShapes(loadPreview());
    const spread = (name: string): number => shapes.get(name)!.spread;
    for (const name of ['viseme_a', 'viseme_u', 'viseme_e', 'viseme_o']) {
      expect(spread('viseme_i'), name).toBeGreaterThan(spread(name));
    }
    for (const name of ['viseme_i', 'viseme_u', 'viseme_e', 'viseme_o']) {
      expect(spread('viseme_a'), name).toBeLessThan(spread(name));
    }
  });

  it('え は あ と い の中間（開きも横の広がりも両者の間に入る）', () => {
    const shapes = visemeShapes(loadPreview());
    const e = shapes.get('viseme_e')!;
    const a = shapes.get('viseme_a')!;
    const i = shapes.get('viseme_i')!;
    expect(e.open).toBeGreaterThan(a.open);
    expect(e.open).toBeLessThan(i.open);
    expect(e.spread).toBeGreaterThan(a.spread);
    expect(e.spread).toBeLessThan(i.spread);
  });

  it('重みを立てると顔が動き、口形も表情と同じ加算変位として当たる', () => {
    const preview = loadPreview();
    const rest = new Float64Array(preview.vertexCount * 3);
    const moved = Float64Array.from(rest);
    addExpression(preview, moved, weightsFor(preview, [['viseme_a', 1]]));
    let maximum = 0;
    for (let index = 0; index < moved.length; index++) {
      maximum = Math.max(maximum, Math.abs(moved[index] - rest[index]));
    }
    // 表情プリセットと同じ桁（weight 1.0 で数 mm 〜 3cm）。
    expect(maximum).toBeGreaterThan(0.002);
    expect(maximum).toBeLessThan(0.03);
  });
});

describe('口形の連続再生', () => {
  const CYCLE = VISEME_FADE_SECONDS * 2 + VISEME_HOLD_SECONDS;

  /** 1 周期ごとに 1 歩進めて、かかった口形の並びを取る。 */
  function play(steps: number, loop: boolean, count = 5): number[] {
    let playback: VisemePlayback = IDLE_VISEME_PLAYBACK;
    const seen: number[] = [];
    for (let step = 0; step < steps; step++) {
      const result = advanceVisemePlayback(playback, count, CYCLE, loop);
      playback = result.playback;
      seen.push(result.index);
    }
    return seen;
  }

  it('あ→い→う→え→お の順に進む', () => {
    expect(play(5, false)).toEqual([0, 1, 2, 3, 4]);
  });

  it('ループ有りは お の次に あ へ戻る', () => {
    expect(play(7, true)).toEqual([0, 1, 2, 3, 4, 0, 1]);
  });

  it('ループ無しは お で止まり、そのまま止まり続ける', () => {
    // 6 歩目以降は -1（何もかかっていない）。頭へ戻らないことまで見る。
    expect(play(8, false)).toEqual([0, 1, 2, 3, 4, -1, -1, -1]);
  });

  it('台形エンベロープで 0 → 1 → 0 を通る（1 音の中で立ち上がって抜ける）', () => {
    let playback: VisemePlayback = IDLE_VISEME_PLAYBACK;
    const weights: number[] = [];
    const delta = CYCLE / 8;
    for (let step = 0; step < 8; step++) {
      const result = advanceVisemePlayback(playback, 5, delta, true);
      playback = result.playback;
      expect(result.index).toBe(0);
      weights.push(result.weight);
    }
    expect(weights[0]).toBeCloseTo(0, 10);
    expect(Math.max(...weights)).toBeCloseTo(1, 10);
    expect(weights[weights.length - 1]).toBeLessThan(0.5);
  });

  it('同時に立つのは 1 本だけ（口形も表情と同じ原則）', () => {
    let playback: VisemePlayback = IDLE_VISEME_PLAYBACK;
    const delta = CYCLE / 7;
    for (let step = 0; step < 40; step++) {
      const result = advanceVisemePlayback(playback, 5, delta, true);
      playback = result.playback;
      // 返るのは常に 1 本ぶんの index と重み。複数を同時に返す形にしていない。
      expect(result.index).toBeGreaterThanOrEqual(-1);
      expect(result.index).toBeLessThan(5);
    }
  });

  it('口形が 1 本も焼かれていなければ何も起きない（落ちない）', () => {
    const result = advanceVisemePlayback(IDLE_VISEME_PLAYBACK, 0, 0.016);
    expect(result.index).toBe(-1);
    expect(result.weight).toBe(0);
  });
});
