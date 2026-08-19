// 調整可能パラメータの一元管理。
// GUI (debugView.ts) から書き換えられる値は Params インスタンスのフィールドとして公開する。
// マジックナンバーはここに集約し、他モジュールはこのオブジェクトを参照する。

export type FullHeadMode = 'HEAD_DEPTH_ONLY' | 'FACE_HEAD' | 'FACE_HEAD_HAIR';

// FULL HEADのデータ供給源。
// MEASURED = Google公式モデル (SelfieMulticlass / ARPortraitDepth。学習データまで商用クリーン)
// NEURAL   = 高品質ニューラル (BiRefNet / Depth Anything V2。重みは商用可だが学習データはグレー。比較用)
// ELLIPSE/HEURISTIC = 楕円+ヒューリスティック (旧方式。比較用)
export type MaskSource = 'MEASURED' | 'NEURAL' | 'ELLIPSE';
export type DepthSource = 'MEASURED' | 'NEURAL' | 'HEURISTIC';

// FULL HEADビューの頭部バックエンド。
// GRID = Head Grid Mesh (2.5D relief。従来方式)
// GNM  = Google GNM Head (真3Dパラメトリック頭部, Apache-2.0) + 実測髪シェル。
//        アセットは tools/export_gnm_assets.py で生成し、選択時に遅延ロードする
export type HeadBackend = 'GRID' | 'GNM';

// GNMバックエンドの表情感情 (AUTO=自動巡回, MANUAL=パーツ別スライダー)。
// キーは main.ts の感情→プリセット表と対応する
export type GnmEmotion = 'AUTO' | 'NEUTRAL' | 'MANUAL' | 'joy' | 'fun' | 'sad' | 'anger' | 'surprise';

export interface Params {
  // --- GUIへ露出する主要パラメータ（spec: 品質比較用UI表） ---
  faceDepthScale: number; // Face Depth: MediaPipe顔凹凸倍率
  canonicalMix: number; // Canonical Mix: canonical顔Depth混合率
  headDepthScale: number; // Head Depth: 頭部全体の膨らみ
  edgeStart: number; // Edge Start: 輪郭巻き込み開始位置
  edgeDepth: number; // Edge Roll: 輪郭後退量
  blendWidthRatio: number; // Face/Head Blend: Face→Head遷移幅 (faceWidth比)
  hairVolumeMax: number; // Hair Volume: 髪の厚み
  pivotZRatio: number; // Pivot Z: 回転中心奥行き (faceWidth比)
  maxYawDeg: number; // Max Yaw
  maxPitchDeg: number; // Max Pitch (縦ドラッグの可動範囲。spec外の追加操作)

  // --- Full Head 表示モード ---
  fullHeadMode: FullHeadMode;

  // --- FULL HEAD データ供給源 (実測 vs ヒューリスティック比較) ---
  maskSource: MaskSource;
  depthSource: DepthSource;
  measuredRegularize: number; // 0-1: 計測Depthを楕円Head Depthへ引き戻す正則化強度
  measuredDepthGain: number; // 計測Depthの振幅倍率 (フィット後のscaleに乗算)

  // --- GNM Head バックエンド ---
  headBackend: HeadBackend;
  gnmIdentityReg: number; // identity係数のL2正則化強度 (大=平均顔寄り)
  // 468点密対応フィットを使う (false=68点フィット。密対応の品質比較用スイッチ)
  gnmDenseFit: boolean;
  // 残差ワープ強度 (0=無効)。identity係数では張り切れない目・唇の位置残差を
  // neutral頂点へ焼き込み、まばたき・開口を写真の目・口の位置で起こす
  gnmWarpStrength: number;
  gnmHairLift: number; // 髪シェルをGNM表面手前へ持ち上げる量 (モデル空間)
  gnmHairRolloff: number; // 髪シェル縁を後方へ巻き込む量 (モデル空間)
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

  // --- Talk Animation / Mouth Cavity ---
  talkEnabled: boolean;
  talkOpenManual: number; // 0.0-1.0, Manual Override時に使用
  talkManualOverride: boolean; // ONの間は周期アニメーションを止めtalkOpenManualを使う
  upperLipMoveScale: number; // upperLipOffset = talkOpen * faceHeight * this
  lowerLipMoveScale: number; // lowerLipOffset = talkOpen * faceHeight * this
  jawMoveScale: number; // jawOffset = talkOpen * faceHeight * this
  cornerInwardScale: number; // cornerInward = talkOpen * faceWidth * this
  mouthCavityDepthRatio: number; // 唇表面から後方へのオフセット (faceWidth比、負値)
  mouthCavityDarkness: number; // 0.0=#180C0C, 1.0=黒に近づく

  // --- デバッグ表示 ---
  showWireframe: boolean;
  showLandmarks: boolean;
  showHeadMask: boolean;
  showHairMask: boolean;
  showFaceDepth: boolean;
  showFinalDepth: boolean;
  showMouthSeam: boolean;
  showMouthRegion: boolean;

  // --- カメラ/共通表示設定 ---
  cameraFovDeg: number;
  cameraDistanceRatio: number; // faceWidth比

  // --- Head Grid 解像度 ---
  headGridCols: number;
  headGridRows: number;

  // --- Face Depth Field 解像度 ---
  faceDepthFieldSize: number;
}

export const DEFAULT_PARAMS: Params = {
  faceDepthScale: 1.0,
  canonicalMix: 0.3,
  headDepthScale: 0.18,
  edgeStart: 0.75,
  edgeDepth: 0.1,
  blendWidthRatio: 0.15,
  hairVolumeMax: 0.04,
  pivotZRatio: -0.07,
  maxYawDeg: 15,
  maxPitchDeg: 12,

  fullHeadMode: 'FACE_HEAD_HAIR',

  maskSource: 'MEASURED',
  depthSource: 'MEASURED',
  measuredRegularize: 0.25,
  measuredDepthGain: 1.0,

  headBackend: 'GRID',
  gnmIdentityReg: 1.0,
  gnmDenseFit: true,
  gnmWarpStrength: 1.0,
  gnmHairLift: 0.02,
  gnmHairRolloff: 0.08,
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

  talkEnabled: true,
  talkOpenManual: 0,
  talkManualOverride: false,
  upperLipMoveScale: 0.005,
  lowerLipMoveScale: 0.025,
  jawMoveScale: 0.015,
  cornerInwardScale: 0.003,
  mouthCavityDepthRatio: -0.02,
  mouthCavityDarkness: 0,

  showWireframe: false,
  showLandmarks: false,
  showHeadMask: false,
  showHairMask: false,
  showFaceDepth: false,
  showFinalDepth: false,
  showMouthSeam: false,
  showMouthRegion: false,

  cameraFovDeg: 30,
  cameraDistanceRatio: 3.4,

  headGridCols: 64,
  headGridRows: 80,

  faceDepthFieldSize: 256,
};

// canonical face depth profile (顔幅正規化値)。 buildCanonicalFaceDepth() が参照する。
export const CANONICAL_FEATURES = {
  noseTip: 0.1,
  noseBridge: 0.06,
  cheek: 0.03,
  upperLip: 0.035,
  chin: 0.015,
  eyeSocket: -0.015,
  faceContour: -0.04,
};

export function createParams(): Params {
  return { ...DEFAULT_PARAMS };
}
