// Blink Animation。
// 生成AIによる閉眼画像生成・眼球テクスチャ生成は行わない。MediaPipeの目周辺landmarkを使い、
// 上瞼を下瞼ラインへ収束させ、虹彩を含む目内部geometryを縦方向に収縮させることで瞬きを表現する。
// FACE ONLYとFULL HEADは同一のBlinkController(blinkAmount)を共有する。

import * as THREE from 'three';
import { FACE_KEY_INDICES, type NormalizedFaceLandmark } from './faceTopology';
import type { FaceOnlyBuild } from './faceOnlyMesh';
import type { FullHeadBuild } from './fullHeadMesh';
import type { Params } from './params';

// ---------------------------------------------------------------------------
// Eye Anchors: 目頭・目尻・上下瞼カーブ・虹彩クラスタ
// ---------------------------------------------------------------------------

export interface EyeCurve {
  inner: { x: number; y: number; z: number };
  outer: { x: number; y: number; z: number };
  control: { x: number; y: number; z: number }; // quadratic bezier制御点
}

export interface EyeAnchors {
  inner: { x: number; y: number; z: number };
  outer: { x: number; y: number; z: number };
  dirX: number; // inner→outerの単位方向
  dirY: number;
  normalX: number; // dirに直交し上方向を向く単位ベクトル
  normalY: number;
  eyeWidth: number; // モデル空間 (faceWidth正規化)
  eyeHeightNorm: number; // 中央(eyeU=0.5)でのupper-lower間の開き量
  upperCurve: EyeCurve; // 目頭→目尻の滑らかなquadratic bezier
  lowerCurve: EyeCurve;
  lidIndices: { upper: number[]; lower: number[] }; // FACE ONLY直接変形用の厳密なlandmark index
  irisIndices: number[]; // このeyeに属す虹彩landmark index (478点モデルのみ)
}

function projectEyeU(px: number, py: number, inner: { x: number; y: number }, dirX: number, dirY: number, eyeWidth: number): number {
  return ((px - inner.x) * dirX + (py - inner.y) * dirY) / eyeWidth;
}

/**
 * 目頭→目尻を通る滑らかなquadratic bezierを構築する。
 * 生MediaPipe landmark(上瞼2点・下瞼2点)をそのまま折れ線として使うと、点間隔が不均一な場合に
 * 上瞼・下瞼の収束先が局所的に凸凹し、Blink時に隣接頂点の上下関係が反転してメッシュが自己交差する
 * (口のMouth Seamと同じ理由でmouthTalk.tsのseamPointAt同様の平滑化が必要)。
 */
function buildSmoothCurve(
  landmarks: NormalizedFaceLandmark[],
  getZ: (index: number) => number,
  inner: { x: number; y: number; z: number },
  outer: { x: number; y: number; z: number },
  midIndices: number[],
): EyeCurve {
  let mx = 0;
  let my = 0;
  let mz = 0;
  for (const i of midIndices) {
    mx += landmarks[i].x;
    my += landmarks[i].y;
    mz += getZ(i);
  }
  mx /= midIndices.length;
  my /= midIndices.length;
  mz /= midIndices.length;
  // control point B(0.5) = mid になるよう逆算する。
  const control = {
    x: 2 * mx - 0.5 * (inner.x + outer.x),
    y: 2 * my - 0.5 * (inner.y + outer.y),
    z: 2 * mz - 0.5 * (inner.z + outer.z),
  };
  return { inner, outer, control };
}

function buildEyeAnchors(
  landmarks: NormalizedFaceLandmark[],
  getZ: (index: number) => number,
  key: { outer: number; upper1: number; upper2: number; inner: number; lower1: number; lower2: number },
  irisIndices: number[],
): EyeAnchors {
  const innerLm = landmarks[key.inner];
  const outerLm = landmarks[key.outer];
  const inner = { x: innerLm.x, y: innerLm.y, z: getZ(key.inner) };
  const outer = { x: outerLm.x, y: outerLm.y, z: getZ(key.outer) };

  const dx = outer.x - inner.x;
  const dy = outer.y - inner.y;
  const eyeWidth = Math.max(1e-5, Math.sqrt(dx * dx + dy * dy));
  const dirX = dx / eyeWidth;
  const dirY = dy / eyeWidth;
  let normalX = -dirY;
  let normalY = dirX;
  if (normalY < 0) {
    normalX = -normalX;
    normalY = -normalY;
  }

  const upperCurve = buildSmoothCurve(landmarks, getZ, inner, outer, [key.upper1, key.upper2]);
  const lowerCurve = buildSmoothCurve(landmarks, getZ, inner, outer, [key.lower1, key.lower2]);

  const midUpper = sampleCurveXYZ(upperCurve, 0.5);
  const midLower = sampleCurveXYZ(lowerCurve, 0.5);
  const eyeHeightNorm = Math.max(1e-4, Math.hypot(midUpper.x - midLower.x, midUpper.y - midLower.y));

  return {
    inner,
    outer,
    dirX,
    dirY,
    normalX,
    normalY,
    eyeWidth,
    eyeHeightNorm,
    upperCurve,
    lowerCurve,
    lidIndices: { upper: [key.upper1, key.upper2], lower: [key.lower1, key.lower2] },
    irisIndices,
  };
}

/** eyeU(0=inner,1=outer)に対応するcurve上の位置を返す (quadratic bezier)。 */
export function sampleCurveXYZ(curve: EyeCurve, eyeU: number): { x: number; y: number; z: number } {
  const t = Math.min(1, Math.max(0, eyeU));
  const u = 1 - t;
  return {
    x: u * u * curve.inner.x + 2 * u * t * curve.control.x + t * t * curve.outer.x,
    y: u * u * curve.inner.y + 2 * u * t * curve.control.y + t * t * curve.outer.y,
    z: u * u * curve.inner.z + 2 * u * t * curve.control.z + t * t * curve.outer.z,
  };
}

function sampleCurveVertical(curve: EyeCurve, anchors: EyeAnchors, eyeU: number): number {
  const p = sampleCurveXYZ(curve, eyeU);
  return (p.x - anchors.inner.x) * anchors.normalX + (p.y - anchors.inner.y) * anchors.normalY;
}

/** 478点モデル(虹彩付き)の場合、末尾10点を目頭中心からの距離で左右の目へ振り分ける。 */
function assignIrisIndices(landmarks: NormalizedFaceLandmark[], eyeACenter: { x: number; y: number }, eyeBCenter: { x: number; y: number }) {
  const irisA: number[] = [];
  const irisB: number[] = [];
  if (landmarks.length >= 478) {
    for (let i = 468; i < 478; i++) {
      const lm = landmarks[i];
      const dA = Math.hypot(lm.x - eyeACenter.x, lm.y - eyeACenter.y);
      const dB = Math.hypot(lm.x - eyeBCenter.x, lm.y - eyeBCenter.y);
      if (dA <= dB) irisA.push(i);
      else irisB.push(i);
    }
  }
  return { irisA, irisB };
}

export function buildEyeAnchorPair(landmarks: NormalizedFaceLandmark[], getZ: (index: number) => number): [EyeAnchors, EyeAnchors] {
  const k = FACE_KEY_INDICES;
  const centerOf = (key: { inner: number; outer: number }) => ({
    x: (landmarks[key.inner].x + landmarks[key.outer].x) / 2,
    y: (landmarks[key.inner].y + landmarks[key.outer].y) / 2,
  });
  const { irisA, irisB } = assignIrisIndices(landmarks, centerOf(k.eyeA), centerOf(k.eyeB));
  return [buildEyeAnchors(landmarks, getZ, k.eyeA, irisA), buildEyeAnchors(landmarks, getZ, k.eyeB, irisB)];
}

function mix3(
  a: { x: number; y: number; z: number },
  b: { x: number; y: number; z: number },
  t: number,
): { x: number; y: number; z: number } {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, z: a.z + (b.z - a.z) * t };
}

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

function lidWeightOf(eyeU: number): number {
  return 0.25 + 0.75 * Math.sin(Math.PI * clamp01(eyeU));
}

/** closeTarget = mix(targetLowerLid, originalUpperPoint, closeTargetBias) を任意のeyeUで求める。 */
export function closeTargetAt(anchors: EyeAnchors, eyeU: number, closeTargetBias: number): { x: number; y: number; z: number } {
  const lowerTarget = sampleCurveXYZ(anchors.lowerCurve, eyeU);
  const upperOriginal = sampleCurveXYZ(anchors.upperCurve, eyeU);
  return mix3(lowerTarget, upperOriginal, closeTargetBias);
}

// ---------------------------------------------------------------------------
// FACE ONLY: landmark厳密indexで直接変形する
// ---------------------------------------------------------------------------

/** Blinkのみを適用する。法線再計算は呼び出し側で一括して行う。 */
export function applyBlinkToFaceOnly(build: FaceOnlyBuild, eyeAnchors: [EyeAnchors, EyeAnchors], t: number, params: Params): void {
  const posAttr = build.geometry.getAttribute('position') as THREE.BufferAttribute;
  const base = build.basePositions;
  const zEps = params.blinkUpperLidZEpsilonRatio;

  for (const anchors of eyeAnchors) {
    // 上瞼: closeTargetへ収束。目頭・目尻の隣接点(upper1/upper2)のみを動かし、
    // 角(inner/outer)そのものは上下瞼で共有される単一頂点のため動かさない。
    for (const idx of anchors.lidIndices.upper) {
      const eyeU = projectEyeU(base[idx * 3], base[idx * 3 + 1], anchors.inner, anchors.dirX, anchors.dirY, anchors.eyeWidth);
      const original = { x: base[idx * 3], y: base[idx * 3 + 1], z: base[idx * 3 + 2] };
      const target = closeTargetAt(anchors, eyeU, params.blinkCloseTargetBias);
      const factor = clamp01(t * lidWeightOf(eyeU) * params.blinkUpperLidMoveScale);
      const p = mix3(original, target, factor);
      posAttr.setXYZ(idx, p.x, p.y, p.z + zEps * factor);
    }

    // 下瞼: ごく少量だけ上方向へ。
    for (const idx of anchors.lidIndices.lower) {
      const dy = t * anchors.eyeHeightNorm * params.blinkLowerLidMove;
      posAttr.setXYZ(idx, base[idx * 3], base[idx * 3 + 1] + dy, base[idx * 3 + 2]);
    }

    // 目内部(虹彩等): 上瞼と同じ(eyeU依存の)ペースでcloseTargetへ収束させる。
    // 上瞼より速く収束させると、目尻付近で虹彩が上瞼を追い越して突き抜けて見える(自己交差)ため、
    // 上瞼と全く同じfactorを使い、常に上瞼と同期して閉じるようにする。
    for (const idx of anchors.irisIndices) {
      const eyeU = projectEyeU(base[idx * 3], base[idx * 3 + 1], anchors.inner, anchors.dirX, anchors.dirY, anchors.eyeWidth);
      const original = { x: base[idx * 3], y: base[idx * 3 + 1], z: base[idx * 3 + 2] };
      const target = closeTargetAt(anchors, eyeU, params.blinkCloseTargetBias);
      const factor = clamp01(t * lidWeightOf(eyeU) * params.blinkUpperLidMoveScale);
      const p = mix3(original, target, factor);
      posAttr.setXYZ(idx, p.x, p.y, p.z + zEps * factor);
    }
  }
}

// ---------------------------------------------------------------------------
// FULL HEAD: Head Grid頂点を位置ベースのfuzzy weightで変形する
// ---------------------------------------------------------------------------

export interface EyeDeformEntry {
  index: number;
  eyeIndex: 0 | 1;
  eyeU: number;
  upperLidW: number;
  lowerLidW: number;
  interiorW: number;
}

const VERTICAL_BAND = 0.4; // openHeight比のガウス半径
const HORIZONTAL_MARGIN = 0.35;
const DEFORM_EPS = 0.003;

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function computeEyeFieldWeights(x: number, y: number, anchors: EyeAnchors): { eyeU: number; upperLidW: number; lowerLidW: number; interiorW: number } {
  const relX = x - anchors.inner.x;
  const relY = y - anchors.inner.y;
  const along = (relX * anchors.dirX + relY * anchors.dirY) / anchors.eyeWidth; // eyeU (未clamp)
  const vertical = relX * anchors.normalX + relY * anchors.normalY;

  const horizontalFalloff = 1 - smoothstep(0.5, 0.5 + HORIZONTAL_MARGIN, Math.abs(along - 0.5));
  const eyeU = clamp01(along);

  if (horizontalFalloff <= 0) {
    return { eyeU, upperLidW: 0, lowerLidW: 0, interiorW: 0 };
  }

  const upperVert = sampleCurveVertical(anchors.upperCurve, anchors, eyeU);
  const lowerVert = sampleCurveVertical(anchors.lowerCurve, anchors, eyeU);
  const openHeight = Math.max(1e-4, upperVert - lowerVert);
  const normVert = (vertical - lowerVert) / openHeight; // 0=下瞼, 1=上瞼

  const upperLidW = horizontalFalloff * Math.exp(-(((normVert - 1) / VERTICAL_BAND) ** 2));
  const lowerLidW = horizontalFalloff * Math.exp(-((normVert / VERTICAL_BAND) ** 2));
  const interiorRaw = horizontalFalloff * clamp01(normVert) * clamp01(1 - normVert) * 4;
  const interiorW = interiorRaw * (1 - clamp01(upperLidW)) * (1 - clamp01(lowerLidW));

  return { eyeU, upperLidW, lowerLidW, interiorW };
}

/** 口と同様、影響を受ける頂点だけを抽出したテーブルを構築する。 */
export function buildEyeDeformTable(count: number, getXY: (index: number) => { x: number; y: number }, eyeAnchors: [EyeAnchors, EyeAnchors]): EyeDeformEntry[] {
  const entries: EyeDeformEntry[] = [];
  for (let i = 0; i < count; i++) {
    const p = getXY(i);
    for (let e = 0; e < 2; e++) {
      const w = computeEyeFieldWeights(p.x, p.y, eyeAnchors[e]);
      if (w.upperLidW > DEFORM_EPS || w.lowerLidW > DEFORM_EPS || w.interiorW > DEFORM_EPS) {
        entries.push({ index: i, eyeIndex: e as 0 | 1, eyeU: w.eyeU, upperLidW: w.upperLidW, lowerLidW: w.lowerLidW, interiorW: w.interiorW });
        break; // 1頂点は最も強く影響する片目にのみ属させる(両目の中間に頂点は通常存在しない)
      }
    }
  }
  return entries;
}

/** Blinkのみを適用する。法線再計算は呼び出し側で一括して行う。 */
export function applyBlinkToFullHead(
  build: FullHeadBuild,
  table: EyeDeformEntry[],
  eyeAnchors: [EyeAnchors, EyeAnchors],
  t: number,
  params: Params,
): void {
  const posAttr = build.geometry.getAttribute('position') as THREE.BufferAttribute;
  const base = build.basePositions;
  const zEps = params.blinkUpperLidZEpsilonRatio;

  for (const e of table) {
    const anchors = eyeAnchors[e.eyeIndex];
    const idx = e.index;
    const bx = base[idx * 3];
    const by = base[idx * 3 + 1];
    const bz = base[idx * 3 + 2];

    const target = closeTargetAt(anchors, e.eyeU, params.blinkCloseTargetBias);
    // interiorも上瞼と同じlidWeightペースで収束させ、目尻付近での追い越し(自己交差)を防ぐ。
    const lidPace = t * lidWeightOf(e.eyeU) * params.blinkUpperLidMoveScale;
    const upperFactor = e.upperLidW * lidPace;
    const interiorFactor = e.interiorW * lidPace;
    const pullFactor = clamp01(upperFactor + interiorFactor);

    const mixed = mix3({ x: bx, y: by, z: bz }, target, pullFactor);
    const lowerDy = e.lowerLidW * t * anchors.eyeHeightNorm * params.blinkLowerLidMove;

    posAttr.setXYZ(idx, mixed.x, mixed.y + lowerDy, mixed.z + zEps * pullFactor);
  }
}

// ---------------------------------------------------------------------------
// Blink Controller: closing → closed → opening の状態遷移
// ---------------------------------------------------------------------------

export type BlinkPhase = 'idle' | 'closing' | 'closed' | 'opening';

export interface BlinkState {
  phase: BlinkPhase;
  phaseStartMs: number;
  nextBlinkAt: number;
}

function randRange(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

function nextInterval(nowMs: number, params: Params): number {
  const sec = params.blinkIntervalRandomize
    ? randRange(params.blinkIntervalMinSec, params.blinkIntervalMaxSec)
    : (params.blinkIntervalMinSec + params.blinkIntervalMaxSec) / 2;
  return nowMs + sec * 1000;
}

export function createBlinkState(nowMs: number, params: Params): BlinkState {
  return { phase: 'idle', phaseStartMs: nowMs, nextBlinkAt: nextInterval(nowMs, params) };
}

function smoothstep01(t: number): number {
  const c = clamp01(t);
  return c * c * (3 - 2 * c);
}

/** 周期Blinkの状態機械を進め、blinkAmountにsmoothstepを適用したt(0-1)を返す。 */
export function updateBlinkAmount(nowMs: number, state: BlinkState, params: Params): number {
  switch (state.phase) {
    case 'idle': {
      if (nowMs >= state.nextBlinkAt) {
        state.phase = 'closing';
        state.phaseStartMs = nowMs;
      } else {
        return 0;
      }
      return 0;
    }
    case 'closing': {
      const elapsed = nowMs - state.phaseStartMs;
      if (elapsed >= params.blinkClosingDurationMs) {
        state.phase = 'closed';
        state.phaseStartMs = nowMs;
        return 1;
      }
      return smoothstep01(elapsed / params.blinkClosingDurationMs);
    }
    case 'closed': {
      const elapsed = nowMs - state.phaseStartMs;
      if (elapsed >= params.blinkClosedHoldMs) {
        state.phase = 'opening';
        state.phaseStartMs = nowMs;
      }
      return 1;
    }
    case 'opening': {
      const elapsed = nowMs - state.phaseStartMs;
      if (elapsed >= params.blinkOpeningDurationMs) {
        state.phase = 'idle';
        state.phaseStartMs = nowMs;
        state.nextBlinkAt = nextInterval(nowMs, params);
        return 0;
      }
      return smoothstep01(1 - elapsed / params.blinkOpeningDurationMs);
    }
  }
}
