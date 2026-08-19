// 入力: Webカメラ撮影 / ローカルファイル選択。
// どちらの経路でも最終的に HTMLCanvasElement (RGB, 撮影済み静止画) を生成して返す。

export interface CapturedImage {
  canvas: HTMLCanvasElement;
  width: number;
  height: number;
}

export class InputManager {
  private videoEl: HTMLVideoElement;
  private stream: MediaStream | null = null;

  constructor(videoEl: HTMLVideoElement) {
    this.videoEl = videoEl;
  }

  get isWebcamActive(): boolean {
    return this.stream !== null;
  }

  async startWebcam(): Promise<void> {
    if (this.stream) return;
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 960 } },
      audio: false,
    });
    this.videoEl.srcObject = this.stream;
    await this.videoEl.play();
    await new Promise<void>((resolve) => {
      if (this.videoEl.readyState >= 2) {
        resolve();
      } else {
        this.videoEl.onloadedmetadata = () => resolve();
      }
    });
  }

  stopWebcam(): void {
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
      this.stream = null;
    }
    this.videoEl.srcObject = null;
  }

  captureWebcamFrame(): CapturedImage {
    const width = this.videoEl.videoWidth;
    const height = this.videoEl.videoHeight;
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d')!;
    // 正面撮影のプレビューはミラー表示だが、テクスチャ/ランドマークは非反転の実画像として扱う。
    ctx.drawImage(this.videoEl, 0, 0, width, height);
    return { canvas, width, height };
  }

  async loadFromFile(file: File): Promise<CapturedImage> {
    const image = await loadImageElement(file);
    const canvas = document.createElement('canvas');
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(image, 0, 0);
    return { canvas, width: canvas.width, height: canvas.height };
  }
}

function loadImageElement(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = (err) => {
      URL.revokeObjectURL(url);
      reject(err);
    };
    img.src = url;
  });
}
