// 調整可能パラメータの一元管理。
// GUI (debugView.ts) から書き換えられる値は Params インスタンスのフィールドとして公開する。
// マジックナンバーはここに集約し、他モジュールはこのオブジェクトを参照する。

export type FullHeadMode = 'HEAD_DEPTH_ONLY' | 'FACE_HEAD' | 'FACE_HEAD_HAIR';

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

  // --- Blink Animation ---
  blinkEnabled: boolean;
  blinkAmountManual: number; // 0.0-1.0, Manual Override時に使用
  blinkManualOverride: boolean; // ONの間は周期Blinkを止めblinkAmountManualを使う
  blinkUpperLidMoveScale: number; // 上瞼closeTargetへの寄り具合の全体倍率
  blinkLowerLidMove: number; // lowerOffset = t * eyeHeight * this
  blinkCloseTargetBias: number; // closeTarget = mix(targetLowerLid, originalUpperLid, this)
  blinkUpperLidZEpsilonRatio: number; // 完全閉眼時、上瞼を下瞼よりわずかに手前へ (faceWidth比)
  blinkClosingDurationMs: number;
  blinkClosedHoldMs: number;
  blinkOpeningDurationMs: number;
  blinkIntervalMinSec: number;
  blinkIntervalMaxSec: number;
  blinkIntervalRandomize: boolean;

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
  showFaceDepth: boolean;
  showFinalDepth: boolean;
  showMouthSeam: boolean;
  showMouthRegion: boolean;
  showUpperLidLine: boolean;
  showLowerLidLine: boolean;
  showBlinkTargets: boolean;
  showEyeRegion: boolean;

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

  blinkEnabled: true,
  blinkAmountManual: 0,
  blinkManualOverride: false,
  blinkUpperLidMoveScale: 2.0,
  blinkLowerLidMove: 0.07,
  blinkCloseTargetBias: 0.08,
  blinkUpperLidZEpsilonRatio: 0.001,
  blinkClosingDurationMs: 90,
  blinkClosedHoldMs: 40,
  blinkOpeningDurationMs: 120,
  blinkIntervalMinSec: 2.5,
  blinkIntervalMaxSec: 5.5,
  blinkIntervalRandomize: true,

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
  showFaceDepth: false,
  showFinalDepth: false,
  showMouthSeam: false,
  showMouthRegion: false,
  showUpperLidLine: false,
  showLowerLidLine: false,
  showBlinkTargets: false,
  showEyeRegion: false,

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
