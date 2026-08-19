// マウス/タッチドラッグによるYaw/Pitch操作。Pointer Eventsで両方を統一的に扱う。
// どちらのpaneをドラッグしても同じYaw/Pitch値を共有するため、コールバックで一元的に更新する。

export interface OrbitDragOptions {
  getYaw: () => number;
  setYaw: (yawDeg: number) => void;
  getMaxYawDeg: () => number;
  getPitch: () => number;
  setPitch: (pitchDeg: number) => void;
  getMaxPitchDeg: () => number;
  degreesPerPixel?: number;
}

export class OrbitDragController {
  private options: OrbitDragOptions;
  private dragging = false;
  private startX = 0;
  private startY = 0;
  private startYaw = 0;
  private startPitch = 0;
  private degreesPerPixel: number;
  private cleanupFns: Array<() => void> = [];

  constructor(elements: HTMLElement[], options: OrbitDragOptions) {
    this.options = options;
    this.degreesPerPixel = options.degreesPerPixel ?? 0.25;
    for (const el of elements) {
      this.attach(el);
    }
  }

  private attach(el: HTMLElement): void {
    const onPointerDown = (e: PointerEvent) => {
      this.dragging = true;
      this.startX = e.clientX;
      this.startY = e.clientY;
      this.startYaw = this.options.getYaw();
      this.startPitch = this.options.getPitch();
      el.setPointerCapture(e.pointerId);
      el.classList.add('dragging');
    };
    const onPointerMove = (e: PointerEvent) => {
      if (!this.dragging) return;
      const deltaX = e.clientX - this.startX;
      const deltaY = e.clientY - this.startY;

      const maxYaw = this.options.getMaxYawDeg();
      const yaw = clamp(this.startYaw + deltaX * this.degreesPerPixel, -maxYaw, maxYaw);
      this.options.setYaw(yaw);

      // 画面を上へドラッグ(deltaY<0)すると見上げる(pitchを正)方向に動かす。
      const maxPitch = this.options.getMaxPitchDeg();
      const pitch = clamp(this.startPitch - deltaY * this.degreesPerPixel, -maxPitch, maxPitch);
      this.options.setPitch(pitch);
    };
    const endDrag = (e: PointerEvent) => {
      if (!this.dragging) return;
      this.dragging = false;
      el.classList.remove('dragging');
      try {
        el.releasePointerCapture(e.pointerId);
      } catch {
        // no-op: ポインタが既に解放されている場合がある
      }
    };

    el.addEventListener('pointerdown', onPointerDown);
    el.addEventListener('pointermove', onPointerMove);
    el.addEventListener('pointerup', endDrag);
    el.addEventListener('pointercancel', endDrag);
    el.style.touchAction = 'none';

    this.cleanupFns.push(() => {
      el.removeEventListener('pointerdown', onPointerDown);
      el.removeEventListener('pointermove', onPointerMove);
      el.removeEventListener('pointerup', endDrag);
      el.removeEventListener('pointercancel', endDrag);
    });
  }

  dispose(): void {
    for (const fn of this.cleanupFns) fn();
    this.cleanupFns = [];
  }
}

function clamp(v: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, v));
}
