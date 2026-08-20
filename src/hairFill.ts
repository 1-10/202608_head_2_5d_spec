// 髪画素を肌色で置換した「bald (髪なし) 頭部テクスチャ」の生成。
//
// 用途: Show Hair off時の「髪を外した頭」表示。髪シェルを外すだけだと
// 写真の髪が頭皮テクスチャに残るため、髪画素を周辺肌色の pull-push 補間で
// 埋めた画像に差し替える。CompHairHead等の compositional 手法が採る
// 「bald画像で顔を作る」発想の簡易版。
// 髭・もみあげ (hairクラスに入りうる) は鼻下端ゲートで保護する。
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
 * bald (髪なし) 頭部テクスチャを作る。置換対象が実質無い場合はnull (元写真をそのまま使う)。
 */
export function buildBaldHeadCanvas(
  sourceCanvas: HTMLCanvasElement,
  seg: SegmentationResult,
  input: HairFillInput,
): HTMLCanvasElement | null {
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

  // 髭・もみあげ保護ゲートの基準row (作業解像度): 鼻の下端 (iBUG 33)。
  // それより上の髪 (側頭部含む) はすべて置換対象
  let gateRow = Infinity;
  const noseLm = input.landmarks[MEDIAPIPE_IBUG68[33]];
  if (noseLm) gateRow = noseLm.py * (h / fullH);
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
      const rw = Math.min(1, hairSoft * vertGate);
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
