// 入力: ローカルファイル選択 / Webカメラ撮影。
//
// **入力は正面写真 1 枚。** どちらの経路でも `PhotoRgb`（uint8 の RGB）へ落とす — 以降の段は
// canvas も File も知らない（デスクトップ側が `imaging.load_photo_rgb` で同じ形にするのと同じ）。
//
// **写真そのものの解像度を落とさない。** 肌アトラスの解像度は写真の顔の大きさで決まるので、入力で
// 縮めると戻せない情報を捨てることになる。大きすぎる写真は `MAX_PHOTO_PIXELS` で落とす。

import { InputImageError } from '../domain/errors';
import { PhotoRgb } from '../domain/photo';
import { imageDataToPhotoRgb } from '../infrastructure/imaging';

/**
 * 受け付ける画素数の上限。デスクトップ側の `imaging.MAX_PHOTO_PIXELS` と同じ値。
 *
 * ブラウザではさらに、セグメンテーションの 4 枚（写真と同じ格子の float32）と肌アトラスが同時に
 * メモリへ乗る。8000 万画素は現実に扱えないので**落とす**（黙って縮めると、アトラスの解像度が
 * 写真から決まるという関係が崩れる）。
 */
export const MAX_PHOTO_PIXELS = 80_000_000;

export class InputManager {
  private readonly video: HTMLVideoElement;
  private stream: MediaStream | null = null;

  constructor(video: HTMLVideoElement) {
    this.video = video;
  }

  get isWebcamActive(): boolean {
    return this.stream !== null;
  }

  async startWebcam(): Promise<void> {
    if (this.stream !== null) return;
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 960 } },
      audio: false,
    });
    this.video.srcObject = this.stream;
    await this.video.play();
    if (this.video.readyState < 2) {
      await new Promise<void>((resolve) => {
        this.video.onloadedmetadata = (): void => resolve();
      });
    }
  }

  stopWebcam(): void {
    if (this.stream !== null) {
      for (const track of this.stream.getTracks()) track.stop();
      this.stream = null;
    }
    this.video.srcObject = null;
  }

  /** 正面撮影のプレビューはミラー表示だが、写真は非反転の実画像として扱う。 */
  captureWebcamFrame(): PhotoRgb {
    return drawToPhoto(this.video, this.video.videoWidth, this.video.videoHeight);
  }

  async loadFromFile(file: File): Promise<PhotoRgb> {
    const image = await loadImageElement(file);
    return drawToPhoto(image, image.naturalWidth, image.naturalHeight);
  }
}

function drawToPhoto(
  source: CanvasImageSource,
  width: number,
  height: number,
): PhotoRgb {
  if (width <= 0 || height <= 0) {
    throw new InputImageError('画像の大きさが取れませんでした。');
  }
  if (width * height > MAX_PHOTO_PIXELS) {
    throw new InputImageError(
      `写真が大きすぎます（${width}x${height} = ${(width * height) / 1e6}メガ画素、` +
        `上限 ${MAX_PHOTO_PIXELS / 1e6}メガ画素）。`,
    );
  }
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (context === null) throw new Error('canvas の 2d コンテキストが取れない');
  context.drawImage(source, 0, 0, width, height);
  return imageDataToPhotoRgb(context.getImageData(0, 0, width, height));
}

function loadImageElement(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = (): void => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = (error): void => {
      URL.revokeObjectURL(url);
      reject(new InputImageError(`画像を読み込めませんでした: ${String(error)}`));
    };
    image.src = url;
  });
}
