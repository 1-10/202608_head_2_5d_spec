// 検査画像の表示。
//
// **検査画像は各段の出力そのもの**（`domain/inspection`）。デスクトップ側は写真ごとのディレクトリへ
// PNG で書き出すが、ブラウザには書き出す先が無いので**画面に並べる**。並びは段の順で、名前は
// `InspectionImages` のキーそのまま（どの段の何を見ているかを名前で引けるようにする）。

import { RgbImage } from '../domain/contract';
import { InspectionImages } from '../domain/inspection';
import { drawRgbImage } from '../infrastructure/imaging';

/** 表示する順（段の順）と、人向けの見出し。 */
const ORDER: readonly [keyof InspectionImages, string][] = [
  ['photoLandmarks', '段1 推論: 検出した478点'],
  ['hairMask', '段1 推論: 髪シェルが覆う対象'],
  ['depth', '段1 推論: 深度（頭部の切り出し）'],
  ['normal', '段1 推論: 表面法線'],
  ['foreground', '段1 推論: 人物前景（全体の切り出し）'],
  ['landmarkFit', '段2 フィット: 対応点（緑）とフィット後（赤）'],
  ['silhouetteFit', '段2 フィット: 耳・首の輪郭'],
  ['leftEyeAlbedo', '段3 眼球: 解剖学的左'],
  ['rightEyeAlbedo', '段3 眼球: 解剖学的右'],
  ['eyeAlbedoProvenance', '段3 眼球: 由来（左右）'],
  ['atlasProjection', '段4 アトラス: 投影位置（緑=写真 / 青=混合）'],
  ['atlasProjectionGate', '段4 アトラス: 前景の門（紫=棄却）'],
  ['atlasAlbedoGate', '段4 アトラス: 門で棄却されたテクセル'],
  ['atlasAlbedo', '段4 アトラス: 焼いたアトラス'],
  ['atlasProvenance', '段4 アトラス: 由来'],
  ['hairShellWire', '段5 髪シェル: ワイヤ'],
  ['hairThickness', '段5 髪シェル: 厚み'],
];

export function renderInspection(container: HTMLElement, inspection: InspectionImages): void {
  container.replaceChildren();
  for (const [key, caption] of ORDER) {
    const image = inspection[key] as RgbImage | undefined;
    if (image === undefined) continue;
    const figure = document.createElement('figure');
    figure.className = 'inspection-item';
    const canvas = document.createElement('canvas');
    drawRgbImage(image, canvas);
    const label = document.createElement('figcaption');
    label.textContent = `${caption}（${image.width}x${image.height}）`;
    figure.append(canvas, label);
    container.append(figure);
  }
}
