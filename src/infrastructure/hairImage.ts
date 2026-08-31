// 髪画像処理のアダプタ。
//
// **品質判断は `domain/hair/maskRefine` が正本。** デスクトップ側はここで反復する box filter だけを
// ONNX Runtime CUDA EP へ移すが、ブラウザには置き換える先が無いので **domain の実装をそのまま呼ぶ**。
// アダプタを残しているのは、application が実装の置き換えを知らずに済む形（port）を崩さないため。

import { HairImageProcessor } from '../application/ports';
import { AlphaImage, RgbImage } from '../domain/contract';
import { HairMask } from '../domain/field';
import { decontaminateHairTexture, refineHairMaskWithPhoto } from '../domain/hair/maskRefine';
import { PhotoRgb } from '../domain/photo';

export class DomainHairImageProcessor implements HairImageProcessor {
  refineMask(photo: PhotoRgb, mask: HairMask, maximumDimension: number): HairMask {
    return refineHairMaskWithPhoto(photo, mask, { maximumDimension });
  }

  decontaminateTexture(photo: RgbImage, alpha: AlphaImage): RgbImage {
    return decontaminateHairTexture(photo, alpha);
  }
}
