// GNM公式 ExpressionSampler (gnm/shape/semantic_sampler.py) のブラウザ移植。
//
// 公式は Conditional VAE のデコーダで、入力 concat(latent(64), one-hot label(20)) から
// 383成分の表情ベクトルを出す。重みは tools/export_gnm_sampler.py が
// public/gnm/gnm_expression_decoder.bin へ書き出す (float16, 0.75MB)。
//
// 移植したのは公式の3つの操作:
// - sample_expression(class)      : クラスのone-hot + 潜在zをデコード
// - blend_expressions(weights)    : 潜在とone-hotを重み付きで混ぜて「1回」デコード
// - randomize_expressions()       : 2〜max個のクラスをランダムに選んでblend
//
// blendを出力ベクトルの線形合成で代用してはいけない。デコーダはReLUの非線形なので
// 結果が別物になる (実測: 顔頂点変位で RMS 0.63〜0.79mm / 最大 3.6〜4.6mm ずれ。
// 表情そのものの変位量が RMS 0.83〜1.33mm なので同程度の誤差)。

import type { GnmModel } from './gnmHead';

interface LayerMeta {
  name: string;
  in: number;
  out: number;
  activation: 'relu' | 'linear';
  kernel: { offset: number; byteLength: number };
  bias: { offset: number; byteLength: number };
}

interface Layer {
  in: number;
  out: number;
  relu: boolean;
  kernel: Float32Array; // (in, out) 行優先
  bias: Float32Array;
}

export interface ExpressionSampler {
  /** 公式 Expression enum の名前 (小文字)。indexがクラス番号 */
  classNames: string[];
  latentDim: number;
  /** 標準正規の潜在ベクトルを引く (公式 rng.normal(size=latent_dim) 相当)。 */
  randomLatent(rand: () => number): Float32Array;
  /**
   * 公式 sample_expression。latentを省略すると0 (潜在空間の中心=クラスの代表)。
   * z=0は公式には無い概念で、固定表情を決め打ちしたいときのこちらの選択。
   */
  sample(classIndex: number, latent?: Float32Array | null): Float32Array;
  /** 公式 blend_expressions。weightsは正規化される。潜在は各クラスごとに引いて重み付き和。 */
  blend(weights: Map<number, number>, rand: () => number): Float32Array;
  /** 公式 randomize_expressions。2〜maxCategories個のクラスを選んでblendする。 */
  randomize(rand: () => number, maxCategories?: number): Float32Array;
  /** 383成分ベクトルをアセットが持つ成分の並びへ射影する (成分名で対応づけ)。 */
  toModelCoeffs(full: Float32Array, model: GnmModel): number[];
}

/** Box-Muller。公式の rng.normal に対応する標準正規乱数。 */
function gaussian(rand: () => number): number {
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

export async function loadExpressionSampler(
  url = 'gnm/gnm_expression_decoder.bin',
): Promise<ExpressionSampler> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(
      `表情サンプラーを取得できません (${res.status})。` +
        'tools/export_gnm_sampler.py で public/gnm/gnm_expression_decoder.bin を生成してください。',
    );
  }
  const buf = await res.arrayBuffer();
  const view = new DataView(buf);
  if (new TextDecoder().decode(new Uint8Array(buf, 0, 4)) !== 'GNMS') {
    throw new Error('表情サンプラーの形式が不正です (magic不一致)。');
  }
  const headerLen = view.getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, headerLen))) as {
    latentDim: number;
    numClasses: number;
    classNames: string[];
    outputNames: string[];
    layers: LayerMeta[];
  };
  const payload = 8 + headerLen;
  const f16 = (m: { offset: number; byteLength: number }): Float32Array => {
    const half = new Uint16Array(buf, payload + m.offset, m.byteLength / 2);
    const out = new Float32Array(half.length);
    // Float16Array がまだ使えないため手で展開する
    for (let i = 0; i < half.length; i++) {
      const h = half[i];
      const s = h >> 15 ? -1 : 1;
      const e = (h >> 10) & 0x1f;
      const f = h & 0x3ff;
      if (e === 0) out[i] = s * 2 ** -24 * f;
      else if (e === 31) out[i] = f ? NaN : s * Infinity;
      else out[i] = s * 2 ** (e - 25) * (1024 + f);
    }
    return out;
  };
  const layers: Layer[] = header.layers.map((l) => ({
    in: l.in,
    out: l.out,
    relu: l.activation === 'relu',
    kernel: f16(l.kernel),
    bias: f16(l.bias),
  }));

  const { latentDim, numClasses, classNames, outputNames } = header;
  const inputBuf = new Float32Array(latentDim + numClasses);
  const decode = (latent: Float32Array | null, label: Float32Array): Float32Array => {
    inputBuf.fill(0);
    if (latent) inputBuf.set(latent.subarray(0, latentDim), 0);
    inputBuf.set(label.subarray(0, numClasses), latentDim);
    let x: Float32Array = inputBuf;
    for (const l of layers) {
      const y = new Float32Array(l.out);
      y.set(l.bias);
      for (let i = 0; i < l.in; i++) {
        const xi = x[i];
        if (xi === 0) continue;
        const row = i * l.out;
        for (let j = 0; j < l.out; j++) y[j] += xi * l.kernel[row + j];
      }
      if (l.relu) {
        for (let j = 0; j < l.out; j++) if (y[j] < 0) y[j] = 0;
      }
      x = y;
    }
    return x;
  };

  // 383成分 → アセットの成分 の対応 (成分名で引く。並びや個数に依存しない)
  const outputIndexOf = new Map<string, number>();
  outputNames.forEach((n, i) => outputIndexOf.set(n, i));

  const label = new Float32Array(numClasses);
  const sampler: ExpressionSampler = {
    classNames,
    latentDim,
    randomLatent(rand) {
      const z = new Float32Array(latentDim);
      for (let i = 0; i < latentDim; i++) z[i] = gaussian(rand);
      return z;
    },
    sample(classIndex, latent = null) {
      label.fill(0);
      label[classIndex] = 1;
      return decode(latent, label);
    },
    blend(weights, rand) {
      let total = 0;
      for (const w of weights.values()) {
        if (w < 0) throw new Error('blend_expressions: 重みは非負でなければなりません');
        total += w;
      }
      if (!(total > 0)) throw new Error('blend_expressions: 重みの合計が0です');
      // 公式実装と同じく「クラスごとに潜在を引いて重み付きで足し、one-hotも重み付きで足す」
      const z = new Float32Array(latentDim);
      label.fill(0);
      for (const [classIndex, w] of weights) {
        const nw = w / total;
        for (let i = 0; i < latentDim; i++) z[i] += gaussian(rand) * nw;
        label[classIndex] += nw;
      }
      return decode(z, label);
    },
    randomize(rand, maxCategories = 3) {
      const upper = Math.min(maxCategories, numClasses);
      const count = 2 + Math.floor(rand() * Math.max(1, upper - 1));
      const pool = Array.from({ length: numClasses }, (_, i) => i);
      const chosen = new Map<number, number>();
      for (let k = 0; k < count && pool.length > 0; k++) {
        const pick = Math.floor(rand() * pool.length);
        chosen.set(pool.splice(pick, 1)[0], rand());
      }
      return sampler.blend(chosen, rand);
    },
    toModelCoeffs(full, model) {
      const out = new Array<number>(model.expressionCount).fill(0);
      for (let i = 0; i < model.expressionCount; i++) {
        const src = outputIndexOf.get(model.expressionNames[i] ?? '');
        if (src !== undefined) out[i] = full[src];
      }
      return out;
    },
  };
  return sampler;
}
