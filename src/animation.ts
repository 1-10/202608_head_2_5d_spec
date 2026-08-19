// 目パチ(Blink)の周期アニメーション。
// 音声・カメラによるリアルタイム表情解析は行わない。FACE ONLYとFULL HEADで完全同期させるため、
// 経過時刻(performance.now()基準)だけから決定論的にamountを求める。
// 口パク(Talk)はmouthTalk.tsのTalkControllerが担当する。

import * as THREE from 'three';
import { FACE_KEY_INDICES, type NormalizedFaceLandmark } from './faceTopology';
import type { FaceOnlyBuild } from './faceOnlyMesh';
import type { FullHeadBuild } from './fullHeadMesh';
import type { Params } from './params';

export interface BlinkState {
  nextBlinkAt: number;
  blinkStartAt: number | null;
  currentDurationMs: number;
}

function randRange(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

export function createBlinkState(nowMs: number, params: Params): BlinkState {
  return {
    nextBlinkAt: nowMs + randRange(params.blinkPeriodMinSec, params.blinkPeriodMaxSec) * 1000,
    blinkStartAt: null,
    currentDurationMs: 200,
  };
}

/** 現在時刻からBlink量(0=開眼, 1=完全に閉眼)を求め、次回スケジュールを内部状態として更新する。 */
export function updateBlink(nowMs: number, state: BlinkState, params: Params): number {
  if (state.blinkStartAt === null) {
    if (nowMs >= state.nextBlinkAt) {
      state.blinkStartAt = nowMs;
      state.currentDurationMs = randRange(params.blinkDurationMinMs, params.blinkDurationMaxMs);
    } else {
      return 0;
    }
  }
  const elapsed = nowMs - state.blinkStartAt!;
  if (elapsed >= state.currentDurationMs) {
    state.blinkStartAt = null;
    state.nextBlinkAt = nowMs + randRange(params.blinkPeriodMinSec, params.blinkPeriodMaxSec) * 1000;
    return 0;
  }
  const t = elapsed / state.currentDurationMs;
  return Math.sin(Math.PI * t); // 0→1→0 の滑らかなbell curve
}

// --- FACE ONLY: landmark頂点を直接変形する ---

const EYELID_TRAVEL = 0.045; // faceWidth比

/** Blinkのみを適用する。法線再計算は呼び出し側で(Talk適用後に)一括して行う。 */
export function applyFaceOnlyBlink(build: FaceOnlyBuild, blinkAmount: number): void {
  const k = FACE_KEY_INDICES;
  const posAttr = build.geometry.getAttribute('position') as THREE.BufferAttribute;
  const base = build.basePositions;

  const setDelta = (index: number, dx: number, dy: number, dz: number) => {
    posAttr.setXYZ(index, base[index * 3] + dx, base[index * 3 + 1] + dy, base[index * 3 + 2] + dz);
  };

  for (const eye of [k.eyeA, k.eyeB]) {
    setDelta(eye.upper1, 0, -EYELID_TRAVEL * 0.7 * blinkAmount, 0);
    setDelta(eye.upper2, 0, -EYELID_TRAVEL * 0.7 * blinkAmount, 0);
    setDelta(eye.lower1, 0, EYELID_TRAVEL * 0.3 * blinkAmount, 0);
    setDelta(eye.lower2, 0, EYELID_TRAVEL * 0.3 * blinkAmount, 0);
  }
}

// --- FULL HEAD: Head Grid頂点をUV近傍の重みマスクで変形する ---

export interface FullHeadAnimationMasks {
  eyeUpper: Float32Array;
  eyeLower: Float32Array;
}

function gaussianFalloff(dx: number, dy: number, radius: number): number {
  const d2 = dx * dx + dy * dy;
  return Math.exp(-d2 / (radius * radius));
}

function anchorOf(landmarks: NormalizedFaceLandmark[], indices: number[]): { x: number; y: number } {
  let x = 0;
  let y = 0;
  for (const i of indices) {
    x += landmarks[i].x;
    y += landmarks[i].y;
  }
  return { x: x / indices.length, y: y / indices.length };
}

export function buildFullHeadAnimationMasks(build: FullHeadBuild, landmarks: NormalizedFaceLandmark[]): FullHeadAnimationMasks {
  const k = FACE_KEY_INDICES;
  const eyeUpperAnchors = [anchorOf(landmarks, [k.eyeA.upper1, k.eyeA.upper2]), anchorOf(landmarks, [k.eyeB.upper1, k.eyeB.upper2])];
  const eyeLowerAnchors = [anchorOf(landmarks, [k.eyeA.lower1, k.eyeA.lower2]), anchorOf(landmarks, [k.eyeB.lower1, k.eyeB.lower2])];

  const eyeRadius = 0.1;

  const count = build.cols * build.rows;
  const eyeUpper = new Float32Array(count);
  const eyeLower = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const x = build.basePositions[i * 3 + 0];
    const y = build.basePositions[i * 3 + 1];

    for (const a of eyeUpperAnchors) eyeUpper[i] += gaussianFalloff(x - a.x, y - a.y, eyeRadius);
    for (const a of eyeLowerAnchors) eyeLower[i] += gaussianFalloff(x - a.x, y - a.y, eyeRadius);
  }

  return { eyeUpper, eyeLower };
}

/** Blinkのみを適用する。法線再計算は呼び出し側で一括して行う。 */
export function applyFullHeadBlink(build: FullHeadBuild, masks: FullHeadAnimationMasks, blinkAmount: number): void {
  const posAttr = build.geometry.getAttribute('position') as THREE.BufferAttribute;
  const count = build.cols * build.rows;

  for (let i = 0; i < count; i++) {
    const bx = build.basePositions[i * 3 + 0];
    const by = build.basePositions[i * 3 + 1];
    const bz = build.basePositions[i * 3 + 2];

    const dy = -EYELID_TRAVEL * 0.7 * blinkAmount * masks.eyeUpper[i] + EYELID_TRAVEL * 0.3 * blinkAmount * masks.eyeLower[i];

    posAttr.setXYZ(i, bx, by + dy, bz);
  }
}
