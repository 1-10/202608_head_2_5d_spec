// 調整可能パラメータの一元管理。
// GUI (debugView.ts) から書き換えられる値は Params インスタンスのフィールドとして公開する。
// マジックナンバーはここに集約し、他モジュールはこのオブジェクトを参照する。

// 実測ソース (テクスチャUVクランプ・髪シェル用) の供給源。
// MEASURED = Google公式モデル (SelfieMulticlass / ARPortraitDepth。学習データまで商用クリーン)
// NEURAL   = 高品質ニューラル (BiRefNet / Depth Anything V2。重みは商用可だが学習データはグレー。比較用)
// NONE     = 不使用 (マスクなし=UVクランプ無効、Depthなし=髪シェル無効)
export type MaskSource = 'MEASURED' | 'NEURAL' | 'NONE';
export type DepthSource = 'MEASURED' | 'NEURAL' | 'NONE';

// GNMの表情感情 (AUTO=自動巡回, MANUAL=パーツ別スライダー)。
// キーは main.ts の感情→プリセット表と対応する
export type GnmEmotion = 'AUTO' | 'NEUTRAL' | 'MANUAL' | 'joy' | 'fun' | 'sad' | 'anger' | 'surprise';

export interface Params {
  // --- 回転操作 ---
  maxYawDeg: number;
  maxPitchDeg: number;

  // --- 実測ソース (UVクランプ・髪シェル) ---
  maskSource: MaskSource;
  depthSource: DepthSource;
  measuredDepthGain: number; // 計測Depthの振幅倍率 (髪シェルの厚み推定に乗算)

  // --- GNM Head フィット ---
  gnmIdentityReg: number; // identity係数のL2正則化強度 (大=平均顔寄り)
  // 468点密対応フィットを使う (false=68点フィット。密対応の品質比較用スイッチ)
  gnmDenseFit: boolean;
  // 残差ワープ強度 (0=無効)。identity係数では張り切れない目・唇の位置残差を
  // neutral頂点へ焼き込み、まばたき・開口を写真の目・口の位置で起こす
  gnmWarpStrength: number;
  gnmHairLift: number; // 髪シェルをGNM表面手前へ持ち上げる量 (モデル空間)
  gnmHairRolloff: number; // 髪シェル縁を後方へ巻き込む量 (モデル空間)
  // 額の髪画素を肌色で埋めた写真をheadテクスチャに使う (髪シェルとの二重描画対策)
  gnmHairSkinFill: boolean;
  gnmHairFillStrength: number; // 置換強度 (1=マスク通り, 小さいほど元の毛を残す)
  // 髪の表示。offで髪シェルを外し、テクスチャも全髪画素を肌色化したbald写真に切替
  gnmShowHair: boolean;

  // --- GNM 表情 ---
  gnmExprIntensity: number; // 感情表情の強さ (プリセット係数への乗数)
  // 表情の感情選択。AUTO=喜怒哀楽を自動巡回 (感情→ニュートラル→別の感情…)、
  // NEUTRAL=無表情、MANUAL=下のパーツ別スライダーで合成、それ以外=その感情で固定。
  // プリセットはGNM公式ExpressionSampler由来
  gnmEmotion: GnmEmotion;
  // --- パーツ別スライダー (Emotion=MANUAL時に有効) ---
  // 公式ExpressionSamplerのクラスを領域 (目成分/下顔面成分) で分離した強度
  gnmMouthOpen: number; // SURPRISEの下顔面 (顎開き)
  gnmSmile: number; // SMILE_WIDEの下顔面
  gnmPucker: number; // PUCKERの下顔面 (口すぼめ)
  gnmCornersDown: number; // CORNERS_DOWNの下顔面 (口角下げ)
  gnmEyesClose: number; // WINK合成の目領域 (閉眼)
  gnmEyesWide: number; // SURPRISEの目領域 (見開き)
  gnmSquint: number; // SQUINTの目領域 (細目)

  // --- アニメーション (Blink) ---
  blinkEnabled: boolean;
  blinkPeriodMinSec: number;
  blinkPeriodMaxSec: number;
  blinkDurationMinMs: number;
  blinkDurationMaxMs: number;

  // --- デバッグ表示 ---
  showWireframe: boolean;
  // 写真ランドマーク(色点)とGNM表面対応点からの残差(白線)の重畳表示
  showLandmarks: boolean;

  // --- カメラ ---
  cameraFovDeg: number;
  cameraDistanceRatio: number; // faceWidth比

  // --- 髪シェルGrid解像度 ---
  hairGridCols: number;
  hairGridRows: number;
}

export const DEFAULT_PARAMS: Params = {
  maxYawDeg: 15,
  maxPitchDeg: 12,

  maskSource: 'MEASURED',
  depthSource: 'MEASURED',
  measuredDepthGain: 1.0,

  gnmIdentityReg: 1.0,
  gnmDenseFit: true,
  gnmWarpStrength: 1.0,
  gnmHairLift: 0.02,
  gnmHairRolloff: 0.08,
  gnmHairSkinFill: true,
  gnmHairFillStrength: 1.0,
  gnmShowHair: true,

  gnmExprIntensity: 1.0,
  gnmEmotion: 'AUTO',
  gnmMouthOpen: 0,
  gnmSmile: 0,
  gnmPucker: 0,
  gnmCornersDown: 0,
  gnmEyesClose: 0,
  gnmEyesWide: 0,
  gnmSquint: 0,

  blinkEnabled: true,
  blinkPeriodMinSec: 3,
  blinkPeriodMaxSec: 5,
  blinkDurationMinMs: 150,
  blinkDurationMaxMs: 250,

  showWireframe: false,
  showLandmarks: false,

  cameraFovDeg: 30,
  cameraDistanceRatio: 3.4,

  hairGridCols: 64,
  hairGridRows: 80,
};

export function createParams(): Params {
  return { ...DEFAULT_PARAMS };
}
