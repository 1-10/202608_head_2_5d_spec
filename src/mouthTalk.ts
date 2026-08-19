// Talk Animation / Mouth Cavity。
// 生成AIによる歯・舌・口腔内テクスチャの補完は行わない。
// 閉口写真のMouth Seam(上下唇の境界)をMediaPipe landmarkから求め、
// 古典的なgeometry deformationのみで上下唇を分離し、露出した隙間に単純な暗色Surface
// (Mouth Cavity)を表示することで口パクを表現する。

import * as THREE from 'three';
import { FACE_KEY_INDICES, type NormalizedFaceLandmark } from './faceTopology';
import type { Params } from './params';

// ---------------------------------------------------------------------------
// Mouth Seam / Anchors
// ---------------------------------------------------------------------------

export interface MouthAnchors {
  cornerA: { x: number; y: number; z: number };
  cornerB: { x: number; y: number; z: number };
  seamCenter: { x: number; y: number; z: number };
  controlPoint: { x: number; y: number; z: number }; // quadratic bezier制御点
  mouthWidth: number; // モデル空間 (faceWidth正規化)
  faceHeightNorm: number; // landmark全体のY範囲 (faceWidth正規化)
  dirX: number; // 口角A→Bの単位方向ベクトル
  dirY: number;
  normalX: number; // dirに直交し、上方向を向く単位ベクトル
  normalY: number;
  cornerAInwardDir: number; // cornerA付近の頂点をX方向へ引き寄せる符号 (+1/-1)
  cornerBInwardDir: number;
}

/**
 * MediaPipe口周辺landmarkからMouth Seamの基準点を求める。真正面・閉口を前提とした単純化。
 * getZは口角・唇中央landmarkのZ値を返す関数 (初回はcomputeFinalFaceDepthPerVertexの結果、
 * Depth系GUIパラメータ変更後はgeometryの現在position.zを渡すことでZだけを追従させられる)。
 */
export function computeMouthAnchors(landmarks: NormalizedFaceLandmark[], getZ: (index: number) => number): MouthAnchors {
  const k = FACE_KEY_INDICES.mouth;
  const a = landmarks[k.cornerA];
  const b = landmarks[k.cornerB];
  const upperC = landmarks[k.upperCenter];
  const lowerC = landmarks[k.lowerCenter];

  const cornerA = { x: a.x, y: a.y, z: getZ(k.cornerA) };
  const cornerB = { x: b.x, y: b.y, z: getZ(k.cornerB) };
  const seamCenter = {
    x: (upperC.x + lowerC.x) / 2,
    y: (upperC.y + lowerC.y) / 2,
    z: (getZ(k.upperCenter) + getZ(k.lowerCenter)) / 2,
  };
  // seamPointAt(0.5)がseamCenterを通るようcontrol pointを逆算する (quadratic bezier)。
  const controlPoint = {
    x: 2 * seamCenter.x - 0.5 * (cornerA.x + cornerB.x),
    y: 2 * seamCenter.y - 0.5 * (cornerA.y + cornerB.y),
    z: 2 * seamCenter.z - 0.5 * (cornerA.z + cornerB.z),
  };

  const dx = cornerB.x - cornerA.x;
  const dy = cornerB.y - cornerA.y;
  const mouthWidth = Math.max(1e-5, Math.sqrt(dx * dx + dy * dy));
  const dirX = dx / mouthWidth;
  const dirY = dy / mouthWidth;
  let normalX = -dirY;
  let normalY = dirX;
  if (normalY < 0) {
    normalX = -normalX;
    normalY = -normalY;
  }

  let minY = Infinity;
  let maxY = -Infinity;
  for (const lm of landmarks) {
    if (lm.y < minY) minY = lm.y;
    if (lm.y > maxY) maxY = lm.y;
  }

  const cornerAInwardDir = cornerA.x - seamCenter.x >= 0 ? -1 : 1;
  const cornerBInwardDir = cornerB.x - seamCenter.x >= 0 ? -1 : 1;

  return {
    cornerA,
    cornerB,
    seamCenter,
    controlPoint,
    mouthWidth,
    faceHeightNorm: Math.max(1e-3, maxY - minY),
    dirX,
    dirY,
    normalX,
    normalY,
    cornerAInwardDir,
    cornerBInwardDir,
  };
}

/** t=0(cornerA)..1(cornerB)のquadratic bezier上の点。t=0.5でseamCenterを通る。 */
export function seamPointAt(t: number, anchors: MouthAnchors): { x: number; y: number; z: number } {
  const u = 1 - t;
  return {
    x: u * u * anchors.cornerA.x + 2 * u * t * anchors.controlPoint.x + t * t * anchors.cornerB.x,
    y: u * u * anchors.cornerA.y + 2 * u * t * anchors.controlPoint.y + t * t * anchors.cornerB.y,
    z: u * u * anchors.cornerA.z + 2 * u * t * anchors.controlPoint.z + t * t * anchors.cornerB.z,
  };
}

// ---------------------------------------------------------------------------
// 領域weight
// ---------------------------------------------------------------------------

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

const UPPER_BAND = 0.22; // mouthWidth比のガウス半径。唇の高さ程度に抑え、鼻・頬まで広がらないようにする
const LOWER_BAND = 0.2;
const JAW_PEAK_OFFSET = 1.4; // lowerBandに対する倍率。この距離あたりでjawWがピーク
const JAW_SIGMA = 0.32;
const CORNER_RADIUS = 0.22;
const HORIZONTAL_MARGIN = 0.55;

export interface MouthWeights {
  upperLipW: number;
  lowerLipW: number;
  jawW: number;
  cornerDX: number; // 符号済み。cornerInwardパラメータに直接掛けるだけでよい
}

/** モデル空間(x,y)におけるUpper Lip / Lower Lip / Jaw Influence / corner-inwardの重みを求める。 */
export function computeMouthWeights(x: number, y: number, anchors: MouthAnchors): MouthWeights {
  const relX = x - anchors.seamCenter.x;
  const relY = y - anchors.seamCenter.y;
  const along = (relX * anchors.dirX + relY * anchors.dirY) / anchors.mouthWidth;
  const vertical = (relX * anchors.normalX + relY * anchors.normalY) / anchors.mouthWidth;

  const horizontalFalloff = 1 - smoothstep(0.5, 0.5 + HORIZONTAL_MARGIN, Math.abs(along));

  let upperLipW = 0;
  let lowerLipW = 0;
  let jawW = 0;
  if (horizontalFalloff > 0) {
    if (vertical >= 0) {
      upperLipW = horizontalFalloff * Math.exp(-((vertical / UPPER_BAND) ** 2));
    } else {
      const d = -vertical;
      lowerLipW = horizontalFalloff * Math.exp(-((d / LOWER_BAND) ** 2));
      const jawRaw = horizontalFalloff * Math.exp(-(((d - LOWER_BAND * JAW_PEAK_OFFSET) / JAW_SIGMA) ** 2));
      jawW = jawRaw * (1 - lowerLipW);
    }
  }

  const distToA = Math.hypot(x - anchors.cornerA.x, y - anchors.cornerA.y) / anchors.mouthWidth;
  const distToB = Math.hypot(x - anchors.cornerB.x, y - anchors.cornerB.y) / anchors.mouthWidth;
  const cornerAW = Math.exp(-((distToA / CORNER_RADIUS) ** 2));
  const cornerBW = Math.exp(-((distToB / CORNER_RADIUS) ** 2));
  const cornerDX = cornerAW * anchors.cornerAInwardDir + cornerBW * anchors.cornerBInwardDir;

  return { upperLipW, lowerLipW, jawW, cornerDX };
}

// ---------------------------------------------------------------------------
// 変形テーブル (FACE ONLY landmark / FULL HEAD grid vertexで共用)
// ---------------------------------------------------------------------------

export interface MouthDeformEntry {
  index: number;
  upperLipW: number;
  lowerLipW: number;
  jawW: number;
  cornerDX: number;
}

const DEFORM_EPS = 0.002;

/** 口周辺の影響を受ける頂点だけを抽出したテーブルを構築する (毎フレームの全頂点走査を避けるため)。 */
export function buildMouthDeformTable(
  count: number,
  getXY: (index: number) => { x: number; y: number },
  anchors: MouthAnchors,
): MouthDeformEntry[] {
  const entries: MouthDeformEntry[] = [];
  for (let i = 0; i < count; i++) {
    const p = getXY(i);
    const w = computeMouthWeights(p.x, p.y, anchors);
    if (w.upperLipW > DEFORM_EPS || w.lowerLipW > DEFORM_EPS || w.jawW > DEFORM_EPS || Math.abs(w.cornerDX) > DEFORM_EPS) {
      entries.push({ index: i, upperLipW: w.upperLipW, lowerLipW: w.lowerLipW, jawW: w.jawW, cornerDX: w.cornerDX });
    }
  }
  return entries;
}

/** talkOpenに応じてUpper/Lower Lip・Jaw・口角の変形を適用する。法線再計算は呼び出し側で行う。 */
export function applyMouthTalkDeform(
  posAttr: THREE.BufferAttribute,
  basePositions: Float32Array,
  table: MouthDeformEntry[],
  talkOpen: number,
  anchors: MouthAnchors,
  params: Params,
): void {
  const upperLipOffset = talkOpen * anchors.faceHeightNorm * params.upperLipMoveScale;
  const lowerLipOffset = talkOpen * anchors.faceHeightNorm * params.lowerLipMoveScale;
  const jawOffset = talkOpen * anchors.faceHeightNorm * params.jawMoveScale;
  const cornerInward = talkOpen * params.cornerInwardScale;

  for (const e of table) {
    const idx = e.index;
    const bx = basePositions[idx * 3];
    const by = basePositions[idx * 3 + 1];
    const bz = basePositions[idx * 3 + 2];
    const dy = upperLipOffset * e.upperLipW - lowerLipOffset * e.lowerLipW - jawOffset * e.jawW;
    const dx = cornerInward * e.cornerDX;
    posAttr.setXYZ(idx, bx + dx, by + dy, bz);
  }
}

// ---------------------------------------------------------------------------
// Mouth Cavity
// ---------------------------------------------------------------------------

export interface MouthCavityBuild {
  geometry: THREE.BufferGeometry;
  mesh: THREE.Mesh;
  segments: number;
}

const CAVITY_SEGMENTS = 14;
const CAVITY_BASE_COLOR = new THREE.Color(0x180c0c);

export function buildMouthCavityMesh(): MouthCavityBuild {
  const segments = CAVITY_SEGMENTS;
  const positions = new Float32Array(segments * 2 * 3);
  const indices: number[] = [];
  for (let i = 0; i < segments - 1; i++) {
    const topA = i;
    const topB = i + 1;
    const botA = segments + i;
    const botB = segments + i + 1;
    indices.push(topA, botA, topB, topB, botA, botB);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(indices);

  const material = new THREE.MeshBasicMaterial({ color: CAVITY_BASE_COLOR.clone(), side: THREE.DoubleSide });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.visible = false;
  mesh.name = 'mouth-cavity';

  return { geometry, mesh, segments };
}

/** talkOpenに応じてMouth Cavityの形状(唇の隙間)を更新する。閉口時は高さ0で不可視になる。 */
export function updateMouthCavityGeometry(build: MouthCavityBuild, anchors: MouthAnchors, talkOpen: number, params: Params): void {
  const upperLipOffset = talkOpen * anchors.faceHeightNorm * params.upperLipMoveScale;
  const lowerLipOffset = talkOpen * anchors.faceHeightNorm * params.lowerLipMoveScale;
  const cavityDepth = params.mouthCavityDepthRatio;

  const posAttr = build.geometry.getAttribute('position') as THREE.BufferAttribute;
  const segments = build.segments;
  for (let i = 0; i < segments; i++) {
    const t = i / (segments - 1);
    const seam = seamPointAt(t, anchors);
    const profile = Math.sin(Math.PI * t); // 口角で0、中央で1のレンズ状プロファイル
    const z = seam.z + cavityDepth;
    posAttr.setXYZ(i, seam.x, seam.y + upperLipOffset * profile, z);
    posAttr.setXYZ(segments + i, seam.x, seam.y - lowerLipOffset * profile, z);
  }
  posAttr.needsUpdate = true;
  build.geometry.computeVertexNormals();

  build.mesh.visible = talkOpen > 0.006 || params.showMouthRegion;
}

export function setMouthCavityDarkness(build: MouthCavityBuild, darkness: number): void {
  const material = build.mesh.material as THREE.MeshBasicMaterial;
  material.color.copy(CAVITY_BASE_COLOR).lerp(new THREE.Color(0x000000), THREE.MathUtils.clamp(darkness, 0, 1));
}

// ---------------------------------------------------------------------------
// Talk Controller (周期アニメーション)
// ---------------------------------------------------------------------------

const TALK_PATTERNS: number[][] = [
  [0, 0.25, 0.6, 0.35, 0.75, 0.2, 0.5, 0.1, 0],
  [0, 0.4, 0.15, 0.65, 0.3, 0.55, 0.1, 0],
  [0, 0.3, 0.7, 0.45, 0.2, 0.6, 0.35, 0.05, 0],
];

const STEP_DURATION_MIN_MS = 80;
const STEP_DURATION_MAX_MS = 180;

export interface TalkState {
  pattern: number[];
  stepIndex: number;
  stepStartMs: number;
  stepDurationMs: number;
}

function pickPattern(): number[] {
  return TALK_PATTERNS[Math.floor(Math.random() * TALK_PATTERNS.length)];
}

export function createTalkState(nowMs: number): TalkState {
  return {
    pattern: pickPattern(),
    stepIndex: 0,
    stepStartMs: nowMs,
    stepDurationMs: randRange(STEP_DURATION_MIN_MS, STEP_DURATION_MAX_MS),
  };
}

function randRange(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

function smoothstep01(t: number): number {
  const c = Math.min(1, Math.max(0, t));
  return c * c * (3 - 2 * c);
}

/** 周期的な疑似発話パターンからtalkOpen(0-1)を求める。音声解析・phoneme認識は行わない。 */
export function updateTalkOpen(nowMs: number, state: TalkState): number {
  let elapsed = nowMs - state.stepStartMs;
  while (elapsed >= state.stepDurationMs) {
    elapsed -= state.stepDurationMs;
    state.stepStartMs += state.stepDurationMs;
    state.stepIndex++;
    if (state.stepIndex >= state.pattern.length - 1) {
      state.stepIndex = 0;
      state.pattern = pickPattern();
    }
    state.stepDurationMs = randRange(STEP_DURATION_MIN_MS, STEP_DURATION_MAX_MS);
  }
  const from = state.pattern[state.stepIndex];
  const to = state.pattern[state.stepIndex + 1];
  const t = smoothstep01(elapsed / state.stepDurationMs);
  return from + (to - from) * t;
}
