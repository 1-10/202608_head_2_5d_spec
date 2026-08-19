// 頭部シルエットマスク。
// MediaPipe Image Segmenterの追加導入は依存とモデル取得の複雑さが増すため、
// spec が許容する「簡易マスク」方針を採用: Face Landmarksから頭部ROIを楕円近似する。
// (人物マスク+Face Landmarks+頭部ROIからの推定、というspecの代替手段に相当)
// クラウド処理・外部アップロードは一切発生しない。

import type { NormalizedFaceLandmark } from './faceTopology';

export interface HeadMaskEllipse {
  cx: number; // モデル空間中心X
  cy: number; // モデル空間中心Y (顔landmark中心から上方へオフセット)
  rx: number; // 頭部横半径 (faceWidth正規化)
  ry: number; // 頭部縦半径
  featherStart: number; // r2がこの値からフェザー開始
  featherEnd: number; // r2がこの値で完全に透明
}

// 顔landmark(頬幅=faceWidth=1.0)に対する頭部全体のスケール係数。
// 頭部は頬幅よりわずかに広く(耳・髪を含む)、縦には額上〜頭頂・顎下まで広がる。
const HEAD_WIDTH_FACTOR = 0.66; // rx
const HEAD_HEIGHT_FACTOR = 0.92; // ry
const HEAD_CENTER_Y_OFFSET = 0.12; // 顔landmark中心より上へシフト(頭頂を含めるため)

export function estimateHeadMaskEllipse(landmarks: NormalizedFaceLandmark[]): HeadMaskEllipse {
  let minY = Infinity;
  let maxY = -Infinity;
  for (const lm of landmarks) {
    if (lm.y < minY) minY = lm.y;
    if (lm.y > maxY) maxY = lm.y;
  }
  const faceHeight = maxY - minY;

  return {
    cx: 0,
    cy: HEAD_CENTER_Y_OFFSET,
    rx: HEAD_WIDTH_FACTOR,
    ry: Math.max(HEAD_HEIGHT_FACTOR, faceHeight * 0.62 + HEAD_CENTER_Y_OFFSET),
    featherStart: 0.88,
    featherEnd: 1.04,
  };
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/** モデル空間(x,y)における頭部マスク値 (0=透明, 1=不透明)。輪郭にsoft featherを持つ。 */
export function sampleHeadMask(x: number, y: number, ellipse: HeadMaskEllipse): number {
  const nx = (x - ellipse.cx) / ellipse.rx;
  const ny = (y - ellipse.cy) / ellipse.ry;
  const r2 = nx * nx + ny * ny;
  return 1 - smoothstep(ellipse.featherStart, ellipse.featherEnd, r2);
}

/**
 * 頭部マスクを入力画像と同じUV空間 (u,v: 0-1, vは上が1) にラスタライズしたCanvasを生成する。
 * Three.jsのalphaMapとしてHead Grid Meshに適用し、輪郭のsoft featherを描画時に表現する。
 */
export function rasterizeHeadMaskCanvas(
  ellipse: HeadMaskEllipse,
  headCenterPx: { x: number; y: number },
  faceWidthPx: number,
  imageWidth: number,
  imageHeight: number,
  size = 256,
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  const imageData = ctx.createImageData(size, size);
  for (let row = 0; row < size; row++) {
    const v = 1 - row / (size - 1);
    const py = (1 - v) * imageHeight;
    for (let col = 0; col < size; col++) {
      const u = col / (size - 1);
      const px = u * imageWidth;
      const x = (px - headCenterPx.x) / faceWidthPx;
      const y = (headCenterPx.y - py) / faceWidthPx;
      const mask = sampleHeadMask(x, y, ellipse);
      const g = Math.round(clamp01(mask) * 255);
      const idx = (row * size + col) * 4;
      imageData.data[idx] = g;
      imageData.data[idx + 1] = g;
      imageData.data[idx + 2] = g;
      imageData.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

function clamp01(x: number): number {
  return Math.min(1, Math.max(0, x));
}
