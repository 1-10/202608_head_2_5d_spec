// 配線（composition root）。具体実装を組み立てて Port として注入する。
//
// **具体実装の組み立てはここだけが行う。** `presentation/` は `infrastructure/` を import しない。
// 入口ごとに配線を持つと、アダプタの構築が変わったとき直す箇所が入口の数だけ増え、**片方を忘れても
// もう片方は動き続けるので気づけない**。

import { ExportOutcome, exportGuest } from './application/exportGuest';
import { ExportSettings } from './application/settings';

import { PhotoRgb } from './domain/photo';
import { CachingAtlasBaker } from './infrastructure/atlasBaker';
import { DavidDepthNormalEstimator } from './infrastructure/depthNormal';
import { MediaPipeFaceLandmarkDetector } from './infrastructure/faceLandmarks';
import {
  DEFAULT_ASSET_URL,
  GnmAssetBundle,
  loadGnmAssetBundle,
} from './infrastructure/gnmAsset';
import { DomainHairImageProcessor } from './infrastructure/hairImage';
import { buildGuestZip } from './infrastructure/packaging';
import { MediaPipePersonSegmenter } from './infrastructure/segmentation';

/**
 * 書き出したパッケージのバージョン。
 *
 * `vite.config.ts` が `package.json` の `version` を差し込む（デスクトップ側が
 * `importlib.metadata.version` で引くのと同じ値の役割）。
 */
declare const __EXPORTER_VERSION__: string;

/** 実行時にずっと持ち回る一式（写真ごとに作り直さない）。 */
export class Exporter {
  private readonly landmarkDetector = new MediaPipeFaceLandmarkDetector();
  private readonly segmenter = new MediaPipePersonSegmenter();
  private readonly depthNormal = new DavidDepthNormalEstimator();
  private readonly atlasBaker = new CachingAtlasBaker();
  private readonly hairImageProcessor = new DomainHairImageProcessor();
  private bundle: GnmAssetBundle | null = null;

  /** GNM アセットの読込（約 32MB）。初回だけ走る。 */
  async loadAsset(url = DEFAULT_ASSET_URL): Promise<GnmAssetBundle> {
    if (this.bundle === null) this.bundle = await loadGnmAssetBundle(url);
    return this.bundle;
  }

  /** DAViD がどの実行環境で動いているか（画面に出す）。まだ推論していなければ null。 */
  get depthNormalProvider(): string | null {
    return this.depthNormal.provider;
  }

  /** 写真 1 枚を書き出す。 */
  async run(
    photo: PhotoRgb,
    settings: ExportSettings,
    onStage?: (stage: string) => void,
  ): Promise<ExportOutcome> {
    const bundle = await this.loadAsset();
    return exportGuest({
      photo,
      asset: bundle.asset,
      landmarkDetector: this.landmarkDetector,
      segmenter: this.segmenter,
      depthNormal: this.depthNormal,
      atlasBaker: this.atlasBaker,
      hairImageProcessor: this.hairImageProcessor,
      settings,
      exporterVersion: exporterVersion(),
      onStage,
    });
  }
}

/** インストール済みのバージョン。取れなければ既定値。 */
export function exporterVersion(): string {
  try {
    return typeof __EXPORTER_VERSION__ === 'string' ? __EXPORTER_VERSION__ : '0+unknown';
  } catch {
    return '0+unknown';
  }
}

export { buildGuestZip };
export type { GnmAssetBundle };
