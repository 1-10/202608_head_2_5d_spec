// セグメンテーションのアダプタ。
//
// MediaPipe Tasks の ImageSegmenter（SelfieMulticlass 256x256）をブラウザで実行し、クラス別の確信度を
// `domain/field.ScalarField` へ変換する。
//
// **クラスの確信度をそのまま返す。** 髪（class 1）・装飾品（class 5）・顔の肌・体の肌の 4 枚で、混ぜ
// ない。足す・床を引く・判定する は `domain/hair/mask.hairShellMask` の仕事（装飾品クラスは頭の外でも
// 当たり、落とすには「顎より下か」という位置の門が要る。その位置はランドマークから決まるが、
// **アダプタはランドマークを知らないし、知るべきでもない** — port の契約は画像 1 枚）。
//
// 人物前景はこのモデルからは取らない。DAViD が深度・法線と同じ推論で前景を出すので、そちらを使う
// （クラス 0 の反転と二重に持つと、境界が食い違ったときにどちらが正しいか決められない）。

import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';
import { PersonSegmenter } from '../application/ports';
import { ModelFileNotFoundError } from '../domain/errors';
import { PersonSegmentation, fieldOverFullImage, validateSegmentation } from '../domain/field';
import { PhotoRgb } from '../domain/photo';
import { photoToCanvas } from './photoCanvas';

const WASM_BASE_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17/wasm';

/**
 * モデルの取得元。**バージョン付きパス**（`/1/`）を使う。
 *
 * `/latest/` は中身が動くので使わない（デスクトップ側の `tools/fetch_models.py` が同じ URL を
 * ハッシュ付きで固定している）。
 */
const MULTICLASS_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter' +
  '/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite';

// SelfieMulticlass のクラス並び（モデルの仕様。ここが正本）
export const CLASS_BACKGROUND = 0;
export const CLASS_HAIR = 1;
export const CLASS_BODY_SKIN = 2;
export const CLASS_FACE_SKIN = 3;
export const CLASS_CLOTHES = 4;
export const CLASS_ACCESSORY = 5;
export const CLASS_COUNT = 6;

export class MediaPipePersonSegmenter implements PersonSegmenter {
  private segmenter: ImageSegmenter | null = null;

  async init(): Promise<void> {
    if (this.segmenter !== null) return;
    try {
      const vision = await FilesetResolver.forVisionTasks(WASM_BASE_URL);
      this.segmenter = await ImageSegmenter.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MULTICLASS_MODEL_URL, delegate: 'GPU' },
        runningMode: 'IMAGE',
        // ソフトマスクが要る。二値だけだと厚みと縁の重みが階段状になる。判定は確信度から導けるので、
        // category mask を別系統でもらう必要はない（あれは 6 択の argmax で、欲しい 2 択の答えでは
        // ない）。
        outputConfidenceMasks: true,
        outputCategoryMask: false,
      });
    } catch (error) {
      throw new ModelFileNotFoundError(`ImageSegmenter を初期化できません: ${String(error)}`);
    }
  }

  /**
   * クラス別の確信度を 4 枚返す。
   *
   * 確信度マスクは入力画像と同じ解像度で返るので、場は画像全体を覆う（切り出しをしない = rect は
   * 画像全体）。**写真と同じ格子なので、重み付き平均を取るときに写真を縮める必要が無い**。
   *
   * **混ぜない・引かない・判定しない。** ここが返すのはモデルが出した確信度そのもの。
   */
  async segment(photo: PhotoRgb): Promise<PersonSegmentation> {
    await this.init();
    if (this.segmenter === null) throw new Error('ImageSegmenter が初期化されていない');
    const canvas = photoToCanvas(photo);
    const result = this.segmenter.segment(canvas);
    const masks = result.confidenceMasks;
    if (masks === undefined || masks.length !== CLASS_COUNT) {
      result.close();
      throw new Error(
        `確信度マスクが ${masks?.length ?? 0} 枚。SelfieMulticlass は ${CLASS_COUNT} クラスなので` +
          '想定と違う（モデルが差し替わった可能性）。',
      );
    }
    const width = masks[0].width;
    const height = masks[0].height;
    // クリップするのは 0..1 を前提に重みとして使う消費側のため（softmax なので本来は範囲内だが、
    // 数値誤差でわずかに外れうる）。**足し合わせはここでしない。**
    const clipped = (classIndex: number): Float32Array => {
      const source = masks[classIndex].getAsFloat32Array();
      const out = new Float32Array(width * height);
      for (let pixel = 0; pixel < out.length; pixel++) {
        out[pixel] = Math.min(1, Math.max(0, source[pixel]));
      }
      return out;
    };
    const hair = clipped(CLASS_HAIR);
    const accessory = clipped(CLASS_ACCESSORY);
    const faceSkin = clipped(CLASS_FACE_SKIN);
    const bodySkin = clipped(CLASS_BODY_SKIN);
    result.close();

    const segmentation: PersonSegmentation = {
      hair: fieldOverFullImage(hair, width, height),
      accessory: fieldOverFullImage(accessory, width, height),
      faceSkin: fieldOverFullImage(faceSkin, width, height),
      bodySkin: fieldOverFullImage(bodySkin, width, height),
    };
    validateSegmentation(segmentation);
    return segmentation;
  }

  close(): void {
    this.segmenter?.close();
    this.segmenter = null;
  }
}
