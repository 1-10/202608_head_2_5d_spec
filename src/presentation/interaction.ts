// マウス/タッチドラッグによる Yaw/Pitch 操作と、ホイールによる拡大縮小。
//
// デスクトップ側の 3D ビューウィンドウと同じ操作（回転・拡大）をブラウザで与える。Pointer Events で
// マウスとタッチを統一的に扱う。

export interface OrbitDragOptions {
  getYaw: () => number;
  setYaw: (yawDeg: number) => void;
  getMaxYawDeg: () => number;
  getPitch: () => number;
  setPitch: (pitchDeg: number) => void;
  getMaxPitchDeg: () => number;
  /** 拡大縮小（省略するとホイールを無視する）。 */
  getZoom?: () => number;
  setZoom?: (zoom: number) => void;
  degreesPerPixel?: number;
}

export class OrbitDragController {
  private readonly options: OrbitDragOptions;
  private dragging = false;
  private startX = 0;
  private startY = 0;
  private startYaw = 0;
  private startPitch = 0;
  private readonly degreesPerPixel: number;
  private readonly cleanups: (() => void)[] = [];

  constructor(elements: readonly HTMLElement[], options: OrbitDragOptions) {
    this.options = options;
    this.degreesPerPixel = options.degreesPerPixel ?? 0.25;
    for (const element of elements) this.attach(element);
  }

  private attach(element: HTMLElement): void {
    const onPointerDown = (event: PointerEvent): void => {
      this.dragging = true;
      this.startX = event.clientX;
      this.startY = event.clientY;
      this.startYaw = this.options.getYaw();
      this.startPitch = this.options.getPitch();
      element.setPointerCapture(event.pointerId);
      element.classList.add('dragging');
    };
    const onPointerMove = (event: PointerEvent): void => {
      if (!this.dragging) return;
      const deltaX = event.clientX - this.startX;
      const deltaY = event.clientY - this.startY;
      const maxYaw = this.options.getMaxYawDeg();
      this.options.setYaw(clamp(this.startYaw + deltaX * this.degreesPerPixel, -maxYaw, maxYaw));
      // 画面を上へドラッグ（deltaY<0）すると見上げる（pitch を正）方向に動かす。
      const maxPitch = this.options.getMaxPitchDeg();
      this.options.setPitch(clamp(this.startPitch - deltaY * this.degreesPerPixel, -maxPitch, maxPitch));
    };
    const endDrag = (event: PointerEvent): void => {
      if (!this.dragging) return;
      this.dragging = false;
      element.classList.remove('dragging');
      try {
        element.releasePointerCapture(event.pointerId);
      } catch {
        // ポインタが既に解放されている場合がある。
      }
    };
    const onWheel = (event: WheelEvent): void => {
      const { getZoom, setZoom } = this.options;
      if (getZoom === undefined || setZoom === undefined) return;
      event.preventDefault();
      setZoom(clamp(getZoom() * Math.exp(-event.deltaY * 0.001), 0.2, 8));
    };

    element.addEventListener('pointerdown', onPointerDown);
    element.addEventListener('pointermove', onPointerMove);
    element.addEventListener('pointerup', endDrag);
    element.addEventListener('pointercancel', endDrag);
    element.addEventListener('wheel', onWheel, { passive: false });
    element.style.touchAction = 'none';

    this.cleanups.push(() => {
      element.removeEventListener('pointerdown', onPointerDown);
      element.removeEventListener('pointermove', onPointerMove);
      element.removeEventListener('pointerup', endDrag);
      element.removeEventListener('pointercancel', endDrag);
      element.removeEventListener('wheel', onWheel);
    });
  }

  dispose(): void {
    for (const cleanup of this.cleanups) cleanup();
    this.cleanups.length = 0;
  }
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
