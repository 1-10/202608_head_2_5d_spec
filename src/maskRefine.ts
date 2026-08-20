// Guided Filter による髪マスクの精細化 (エッジ整合アップサンプル)。
//
// SelfieMulticlass の髪マスクは 256x256 しかなく、写真に重ねると境界が
// 数px分ボケ/ブロック状になる。写真自体をガイドにした Guided Filter
// (He et al. 2010) で粗いソフトマスクを作業解像度へ持ち上げると、
// マスク境界が写真の実エッジ (髪⇔背景の色の変わり目) へスナップし、
// 後れ毛は半透明のまま残る。
//
// 学習モデル不要 = 商用ライセンス問題ゼロ。実装は論文の式から独自に
// 書いたもの (作者の参照MATLABコードは学術用途限定のため参照していない)。
// カラーガイド版: a = (Σ + εU)⁻¹ cov(I,p), b = mean(p) − a·mean(I),
// q = mean(a)·I + mean(b)。box filter は積分画像で O(N)。

import { fullImageRect, sampleField, type ScalarField } from './fields';

const WORK_MAX_DIM = 768; // 精細化の作業解像度 (出力ScalarFieldの解像度)
const RADIUS_FRAC = 0.02; // box filter半径 (作業解像度の長辺比)
const EPS = 1e-3; // 正則化 (小さいほどエッジに強く追従し、ノイズも拾う)

/**
 * 写真をガイドに粗いマスクを精細化した ScalarField を返す。
 * 出力は作業解像度 (長辺768) で、境界が写真のエッジへ整合したソフトマスク。
 */
export function refineMaskWithGuide(sourceCanvas: HTMLCanvasElement, mask: ScalarField): ScalarField {
  const scale = Math.min(1, WORK_MAX_DIM / Math.max(sourceCanvas.width, sourceCanvas.height));
  const w = Math.max(8, Math.round(sourceCanvas.width * scale));
  const h = Math.max(8, Math.round(sourceCanvas.height * scale));
  const size = w * h;

  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const cctx = canvas.getContext('2d')!;
  cctx.drawImage(sourceCanvas, 0, 0, w, h);
  const img = cctx.getImageData(0, 0, w, h).data;

  // ガイドI (RGB, 0-1) と入力p (マスクのbilinearアップサンプル)
  const Ir = new Float32Array(size);
  const Ig = new Float32Array(size);
  const Ib = new Float32Array(size);
  const p = new Float32Array(size);
  for (let y = 0; y < h; y++) {
    const v = 1 - (y + 0.5) / h;
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      Ir[i] = img[i * 4] / 255;
      Ig[i] = img[i * 4 + 1] / 255;
      Ib[i] = img[i * 4 + 2] / 255;
      p[i] = sampleField(mask, (x + 0.5) / w, v);
    }
  }

  const r = Math.max(2, Math.round(Math.max(w, h) * RADIUS_FRAC));
  const box = makeBoxFilter(w, h, r);

  const meanR = box(Ir);
  const meanG = box(Ig);
  const meanB = box(Ib);
  const meanP = box(p);

  // 分散共分散 (対称3x3の上三角 + I·p の3成分)
  const mul = (a: Float32Array, b: Float32Array): Float32Array => {
    const out = new Float32Array(size);
    for (let i = 0; i < size; i++) out[i] = a[i] * b[i];
    return out;
  };
  const covRR = box(mul(Ir, Ir));
  const covRG = box(mul(Ir, Ig));
  const covRB = box(mul(Ir, Ib));
  const covGG = box(mul(Ig, Ig));
  const covGB = box(mul(Ig, Ib));
  const covBB = box(mul(Ib, Ib));
  const covRP = box(mul(Ir, p));
  const covGP = box(mul(Ig, p));
  const covBP = box(mul(Ib, p));

  const aR = new Float32Array(size);
  const aG = new Float32Array(size);
  const aB = new Float32Array(size);
  const b = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    // Σ = E[II^T] − mean mean^T + εU
    const s11 = covRR[i] - meanR[i] * meanR[i] + EPS;
    const s12 = covRG[i] - meanR[i] * meanG[i];
    const s13 = covRB[i] - meanR[i] * meanB[i];
    const s22 = covGG[i] - meanG[i] * meanG[i] + EPS;
    const s23 = covGB[i] - meanG[i] * meanB[i];
    const s33 = covBB[i] - meanB[i] * meanB[i] + EPS;
    const c1 = covRP[i] - meanR[i] * meanP[i];
    const c2 = covGP[i] - meanG[i] * meanP[i];
    const c3 = covBP[i] - meanB[i] * meanP[i];

    // 3x3対称行列の逆行列 (余因子展開)
    const m11 = s22 * s33 - s23 * s23;
    const m12 = s13 * s23 - s12 * s33;
    const m13 = s12 * s23 - s13 * s22;
    const det = s11 * m11 + s12 * m12 + s13 * m13;
    if (Math.abs(det) < 1e-12) {
      b[i] = meanP[i];
      continue;
    }
    const inv = 1 / det;
    const m22 = s11 * s33 - s13 * s13;
    const m23 = s12 * s13 - s11 * s23;
    const m33 = s11 * s22 - s12 * s12;
    aR[i] = (m11 * c1 + m12 * c2 + m13 * c3) * inv;
    aG[i] = (m12 * c1 + m22 * c2 + m23 * c3) * inv;
    aB[i] = (m13 * c1 + m23 * c2 + m33 * c3) * inv;
    b[i] = meanP[i] - aR[i] * meanR[i] - aG[i] * meanG[i] - aB[i] * meanB[i];
  }

  const meanAR = box(aR);
  const meanAG = box(aG);
  const meanAB = box(aB);
  const meanBb = box(b);

  const out = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    const q = meanAR[i] * Ir[i] + meanAG[i] * Ig[i] + meanAB[i] * Ib[i] + meanBb[i];
    out[i] = Math.min(1, Math.max(0, q));
  }

  return { width: w, height: h, data: out, rect: fullImageRect() };
}

/**
 * 積分画像による正規化box filter (半径r, 画像端はウィンドウを切り詰めて正規化)。
 * 同一サイズ画像へ繰り返し使うためクロージャで返す。
 */
function makeBoxFilter(w: number, h: number, r: number): (src: Float32Array) => Float32Array {
  const sat = new Float64Array((w + 1) * (h + 1));
  return (src: Float32Array): Float32Array => {
    for (let y = 0; y < h; y++) {
      let rowSum = 0;
      for (let x = 0; x < w; x++) {
        rowSum += src[y * w + x];
        sat[(y + 1) * (w + 1) + (x + 1)] = sat[y * (w + 1) + (x + 1)] + rowSum;
      }
    }
    const out = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      const y0 = Math.max(0, y - r);
      const y1 = Math.min(h - 1, y + r);
      for (let x = 0; x < w; x++) {
        const x0 = Math.max(0, x - r);
        const x1 = Math.min(w - 1, x + r);
        const sum =
          sat[(y1 + 1) * (w + 1) + (x1 + 1)] -
          sat[y0 * (w + 1) + (x1 + 1)] -
          sat[(y1 + 1) * (w + 1) + x0] +
          sat[y0 * (w + 1) + x0];
        out[y * w + x] = sum / ((y1 - y0 + 1) * (x1 - x0 + 1));
      }
    }
    return out;
  };
}
