// 額の髪画素を肌色で置換した「髪なし顔テクスチャ」の生成。
//
// 動機: GNM headは写真を平行投影するため、額にかかった髪 (まばらな前髪・
// 生え際の細い毛) が肌の頂点色/UVへ焼き付く。髪は髪シェルが手前に描くので、
// 視差 (yaw/pitch回転) が付くと「シェルの髪」と「肌に焼き付いた髪」が
// 二重に見える。CompHairHead等の compositional 手法が採る「bald画像で顔を作り
// 髪を別レイヤーで重ねる」発想の簡易版として、髪画素を周辺肌色の pull-push
// 補間で埋める。
//
// 置換するのは「肌の上に重なった髪」だけに限定する:
// - faceSkin共在ゲート … hairとfaceSkinの信頼度が同一画素で同時に立つのは
//   「肌が透ける、まばらな毛」だけ。生え際より上の密な髪帯はfaceSkin=0で
//   置換されない (GNM頭皮はUVクランプで髪色を拾い、髪シェルalpha縁の
//   ギザギザを裏から隠す既存設計をそのまま維持できる)
// - 眉より上限定 … 髭・もみあげ (hairクラスに入りうる) の誤消去を防ぐ
//
// 依存はSelfieMulticlassのマスクのみ (ニューラルinpainting不使用 = 商用クリーン)。

import { sampleField } from './fields';
import type { NormalizedFaceLandmark } from './faceTopology';
import { MEDIAPIPE_IBUG68 } from './gnmHead';
import { smoothstep } from './meshUtils';
import type { SegmentationResult } from './personSegmentation';

const WORK_MAX_DIM = 512; // 補間処理の作業解像度上限 (fill色は低周波なので十分)
const REPLACE_MIN_TOTAL = 3; // 置換重みの総量がこれ未満なら「髪被りなし」として何もしない

export interface HairFillInput {
  landmarks: NormalizedFaceLandmark[];
  faceWidthPx: number; // 元画像ピクセルでの顔幅 (眉ゲートのフェード幅の基準)
}

/**
 * 置換モード。
 * - overlay: 肌の上に重なったまばらな毛だけ置換 (髪シェルと併用する通常モード)
 * - bald: 髪をすべて置換 (髪シェル非表示の「髪を外した頭」用)。
 *   髭・もみあげの保護は眉ゲートではなく鼻下端ゲートで行う (それより上の
 *   側頭部の髪は側面もすべて肌化する)
 */
export type HairFillMode = 'overlay' | 'bald';

/**
 * 髪なし顔テクスチャを作る。置換対象が実質無い場合はnull (元写真をそのまま使う)。
 * strength: 置換強度 (1=マスク通り置換, 小さいほど元の毛を残す)。
 */
export function buildHairFreeFaceCanvas(
  sourceCanvas: HTMLCanvasElement,
  seg: SegmentationResult,
  input: HairFillInput,
  strength: number,
  mode: HairFillMode = 'overlay',
): HTMLCanvasElement | null {
  if (strength <= 0) return null;
  const fullW = sourceCanvas.width;
  const fullH = sourceCanvas.height;
  const scale = Math.min(1, WORK_MAX_DIM / Math.max(fullW, fullH));
  const w = Math.max(2, Math.round(fullW * scale));
  const h = Math.max(2, Math.round(fullH * scale));

  const work = document.createElement('canvas');
  work.width = w;
  work.height = h;
  const workCtx = work.getContext('2d')!;
  workCtx.drawImage(sourceCanvas, 0, 0, w, h);
  const img = workCtx.getImageData(0, 0, w, h);

  const faceWidthW = input.faceWidthPx * (w / fullW);

  // 髭・もみあげ保護ゲートの基準row (作業解像度)。
  // overlay: 眉の上端 (iBUG 17-26)。bald: 鼻の下端 (iBUG 33) —
  // 髪を全部外すため側頭部 (眉〜鼻の高さ) も置換対象に含める
  let gateRow = Infinity;
  if (mode === 'bald') {
    const lm = input.landmarks[MEDIAPIPE_IBUG68[33]];
    if (lm) gateRow = lm.py * (h / fullH);
  } else {
    for (let k = 17; k <= 26; k++) {
      const lm = input.landmarks[MEDIAPIPE_IBUG68[k]];
      if (lm) gateRow = Math.min(gateRow, lm.py * (h / fullH));
    }
  }
  if (!Number.isFinite(gateRow)) return null;
  const gateBand = Math.max(1, 0.06 * faceWidthW); // ゲートのフェード幅

  const srcWeight = new Float32Array(w * h); // fill色の供給源 (髪でない顔皮膚)
  const replaceW = new Float32Array(w * h);
  let total = 0;
  for (let y = 0; y < h; y++) {
    const v = 1 - (y + 0.5) / h;
    const vertGate = 1 - smoothstep(gateRow, gateRow + gateBand, y);
    for (let x = 0; x < w; x++) {
      const u = (x + 0.5) / w;
      const i = y * w + x;
      const hairSoft = smoothstep(0.1, 0.6, sampleField(seg.hair, u, v));
      const face = sampleField(seg.faceSkin, u, v);
      srcWeight[i] = face * (1 - hairSoft);
      if (vertGate <= 0) continue;
      // overlay: faceSkin共在ゲート (肌が透けて見える画素の髪だけ置換)。
      // bald: 髪はすべて置換
      const coGate = mode === 'bald' ? 1 : smoothstep(0.15, 0.5, face);
      const rw = Math.min(1, hairSoft * coGate * vertGate * strength);
      replaceW[i] = rw;
      total += rw;
    }
  }
  if (total < REPLACE_MIN_TOTAL) return null;

  // 肌色fill (pull-push)。sRGBのまま平均するが、肌色の低周波fillには十分
  const rgb = new Float32Array(w * h * 3);
  for (let i = 0; i < w * h; i++) {
    rgb[i * 3] = img.data[i * 4];
    rgb[i * 3 + 1] = img.data[i * 4 + 1];
    rgb[i * 3 + 2] = img.data[i * 4 + 2];
  }
  const filled = pullPushFill(rgb, srcWeight, w, h);

  // 置換レイヤー (fill色 + alpha=replaceW) を作り、フル解像度の元写真へ重ねる。
  // fill色は低周波なので作業解像度からの拡大で劣化しない
  const layer = document.createElement('canvas');
  layer.width = w;
  layer.height = h;
  const layerCtx = layer.getContext('2d')!;
  const layerData = layerCtx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    layerData.data[i * 4] = Math.round(filled[i * 3]);
    layerData.data[i * 4 + 1] = Math.round(filled[i * 3 + 1]);
    layerData.data[i * 4 + 2] = Math.round(filled[i * 3 + 2]);
    layerData.data[i * 4 + 3] = Math.round(replaceW[i] * 255);
  }
  layerCtx.putImageData(layerData, 0, 0);

  const out = document.createElement('canvas');
  out.width = fullW;
  out.height = fullH;
  const outCtx = out.getContext('2d')!;
  outCtx.drawImage(sourceCanvas, 0, 0);
  outCtx.imageSmoothingEnabled = true;
  outCtx.imageSmoothingQuality = 'high';
  outCtx.drawImage(layer, 0, 0, fullW, fullH);
  return out;
}

/**
 * pull-push補間: weight>0の画素の色を、weight=0の画素へ滑らかに外挿する。
 * pull=重み付き色をピラミッドへ縮小、push=粗いレベルの色でweight不足分を埋める。
 */
function pullPushFill(rgb: Float32Array, weight: Float32Array, w: number, h: number): Float32Array {
  interface Level {
    w: number;
    h: number;
    color: Float32Array; // 非premultiplied (正規化済み)
    wt: Float32Array; // 0-1
  }

  const levels: Level[] = [];
  {
    const color = new Float32Array(rgb);
    const wt = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) wt[i] = Math.min(1, weight[i]);
    levels.push({ w, h, color, wt });
  }

  // pull: 2x2の重み付き平均で縮小
  while (levels[levels.length - 1].w > 1 || levels[levels.length - 1].h > 1) {
    const src = levels[levels.length - 1];
    const dw = Math.max(1, src.w >> 1);
    const dh = Math.max(1, src.h >> 1);
    const color = new Float32Array(dw * dh * 3);
    const wt = new Float32Array(dw * dh);
    for (let y = 0; y < dh; y++) {
      for (let x = 0; x < dw; x++) {
        let r = 0;
        let g = 0;
        let b = 0;
        let sw = 0;
        for (let dy = 0; dy < 2; dy++) {
          const sy = y * 2 + dy;
          if (sy >= src.h) continue;
          for (let dx = 0; dx < 2; dx++) {
            const sx = x * 2 + dx;
            if (sx >= src.w) continue;
            const si = sy * src.w + sx;
            const sWt = src.wt[si];
            r += src.color[si * 3] * sWt;
            g += src.color[si * 3 + 1] * sWt;
            b += src.color[si * 3 + 2] * sWt;
            sw += sWt;
          }
        }
        const di = y * dw + x;
        if (sw > 0) {
          color[di * 3] = r / sw;
          color[di * 3 + 1] = g / sw;
          color[di * 3 + 2] = b / sw;
        }
        wt[di] = Math.min(1, sw);
      }
    }
    levels.push({ w: dw, h: dh, color, wt });
  }

  // push: 粗→細へ、weight不足分を親のbilinear補間色で埋める
  for (let li = levels.length - 2; li >= 0; li--) {
    const fine = levels[li];
    const coarse = levels[li + 1];
    for (let y = 0; y < fine.h; y++) {
      const fy = Math.min(coarse.h - 1, Math.max(0, (y + 0.5) / 2 - 0.5));
      const y0 = Math.floor(fy);
      const y1 = Math.min(coarse.h - 1, y0 + 1);
      const ay = fy - y0;
      for (let x = 0; x < fine.w; x++) {
        const i = y * fine.w + x;
        const fwt = fine.wt[i];
        if (fwt >= 1) continue;
        const fx = Math.min(coarse.w - 1, Math.max(0, (x + 0.5) / 2 - 0.5));
        const x0 = Math.floor(fx);
        const x1 = Math.min(coarse.w - 1, x0 + 1);
        const ax = fx - x0;
        for (let c = 0; c < 3; c++) {
          const c00 = coarse.color[(y0 * coarse.w + x0) * 3 + c];
          const c10 = coarse.color[(y0 * coarse.w + x1) * 3 + c];
          const c01 = coarse.color[(y1 * coarse.w + x0) * 3 + c];
          const c11 = coarse.color[(y1 * coarse.w + x1) * 3 + c];
          const up = (c00 * (1 - ax) + c10 * ax) * (1 - ay) + (c01 * (1 - ax) + c11 * ax) * ay;
          fine.color[i * 3 + c] = fine.color[i * 3 + c] * fwt + up * (1 - fwt);
        }
        fine.wt[i] = 1;
      }
    }
  }

  return levels[0].color;
}
