// FULL HEAD用のDepth計算群: Pseudo Head Depth, Edge Rolloff, Face/Head Blend Weight,
// 額領域の独立補間, Hair Volume。すべて後から比較・置換しやすいよう純粋関数として分離する。

import type { HeadMaskEllipse } from './headMask';

export function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function clamp01(x: number): number {
  return Math.min(1, Math.max(0, x));
}

function ellipseR2(x: number, y: number, ellipse: HeadMaskEllipse): number {
  const nx = (x - ellipse.cx) / ellipse.rx;
  const ny = (y - ellipse.cy) / ellipse.ry;
  return nx * nx + ny * ny;
}

/** 正面楕円球表面を模した頭部基本Depth。中央が手前、輪郭に近づくほど奥へ。 */
export function computePseudoHeadDepth(x: number, y: number, ellipse: HeadMaskEllipse, headDepthScale: number): number {
  const r2 = ellipseR2(x, y, ellipse);
  return headDepthScale * Math.sqrt(Math.max(0, 1 - r2));
}

/** 外周を後方へ巻き込み、Yaw回転時の「紙の断面」を防ぐ。 */
export function computeEdgeRolloff(x: number, y: number, ellipse: HeadMaskEllipse, edgeStart: number, edgeDepth: number): number {
  const r = Math.sqrt(ellipseR2(x, y, ellipse));
  const edge = clamp01((r - edgeStart) / (1 - edgeStart));
  return -edgeDepth * smoothstep(0, 1, edge);
}

export interface HeadDepthResult {
  zHead: number;
  zRolloff: number;
  zHeadFinal: number;
}

export function computeHeadDepthFinal(
  x: number,
  y: number,
  ellipse: HeadMaskEllipse,
  headDepthScale: number,
  edgeStart: number,
  edgeDepth: number,
): HeadDepthResult {
  const zHead = computePseudoHeadDepth(x, y, ellipse, headDepthScale);
  const zRolloff = computeEdgeRolloff(x, y, ellipse, edgeStart, edgeDepth);
  return { zHead, zRolloff, zHeadFinal: zHead + zRolloff };
}

// --- Face / Head Blend Weight ---

interface Point {
  x: number;
  y: number;
}

function pointInPolygon(p: Point, poly: Point[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const pi = poly[i];
    const pj = poly[j];
    const intersects = pi.y > p.y !== pj.y > p.y && p.x < ((pj.x - pi.x) * (p.y - pi.y)) / (pj.y - pi.y) + pi.x;
    if (intersects) inside = !inside;
  }
  return inside;
}

function distanceToSegment(p: Point, a: Point, b: Point): number {
  const abx = b.x - a.x;
  const aby = b.y - a.y;
  const lenSq = abx * abx + aby * aby;
  let t = lenSq > 1e-12 ? ((p.x - a.x) * abx + (p.y - a.y) * aby) / lenSq : 0;
  t = clamp01(t);
  const cx = a.x + abx * t;
  const cy = a.y + aby * t;
  const dx = p.x - cx;
  const dy = p.y - cy;
  return Math.sqrt(dx * dx + dy * dy);
}

function distanceToPolygonBoundary(p: Point, poly: Point[]): number {
  let min = Infinity;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const d = distanceToSegment(p, poly[i], poly[j]);
    if (d < min) min = d;
  }
  return min;
}

/**
 * Face境界(顔landmark hull)からの距離に基づくFace優先度の重み。
 * hull内部は距離0とみなしWface=1、境界からblendWidth離れるとWface=0へ滑らかに減衰する。
 */
export function computeFaceHeadBlendWeight(x: number, y: number, hullPoints: Point[], blendWidth: number): number {
  if (hullPoints.length < 3) return 0;
  const p = { x, y };
  const inside = pointInPolygon(p, hullPoints);
  const dist = inside ? 0 : distanceToPolygonBoundary(p, hullPoints);
  const t = clamp01(dist / Math.max(1e-6, blendWidth));
  return 1 - smoothstep(0, 1, t);
}

// --- 額領域の独立補間 ---

/** browY(眉上端)〜headTopY(頭頂)間で0→1になる正規化パラメータ。 */
export function computeForeheadWeight(y: number, browY: number, headTopY: number): number {
  if (headTopY <= browY) return 0;
  return smoothstep(0, 1, clamp01((y - browY) / (headTopY - browY)));
}

export function computeForeheadDepth(zBrow: number, zHeadFinal: number, foreheadT: number): number {
  return zBrow * (1 - foreheadT) + zHeadFinal * foreheadT;
}

// --- Hair Volume ---

/**
 * 髪の擬似ボリューム。頭蓋表面(Zhead)より少し手前に厚みを加える。
 * hairMask: 頭部マスクからFace優先度を除いた「髪らしさ」重み (0-1)
 * verticalT: 0(側面/下部)〜1(頭頂)の高さ係数。上部ほど厚くする。
 */
export function computeHairVolume(hairMask: number, verticalT: number, hairVolumeMax: number): number {
  const profile = 0.4 + 0.6 * clamp01(verticalT);
  return hairVolumeMax * clamp01(hairMask) * profile;
}
