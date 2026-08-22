// 目パチ(Blink)の周期エンベロープ。
// 音声・カメラによるリアルタイム表情解析は行わず、経過時刻(performance.now()基準)だけから
// 決定論的にamount(0=開眼, 1=閉眼)を求める。メッシュへの適用はGNM表情基底が担う。

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
