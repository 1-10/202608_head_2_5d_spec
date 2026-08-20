// 調整可能パラメータの一元管理。
// GUI (debugView.ts) から書き換えられる値は Params インスタンスのフィールドとして公開する。
// マジックナンバーはここに集約し、他モジュールはこのオブジェクトを参照する。

// 実測ソース (テクスチャUVクランプ・髪シェル用) の供給源。
// 値は実モデル名で持つ (すべて学習データまで商用クリーン):
// SELFIE_MULTICLASS = Google SelfieMulticlass 256px (意味分け: 髪/肌/人物)
// ARPORTRAIT_DEPTH  = Google ARPortraitDepth (低解像度Depth。DAViDとの比較用)
// DAVID             = Microsoft DAViD multi-task (人物特化のDepth/法線/前景。
//                     100%合成データ学習+MIT。初回のみモデルDL ~660MB)
// NONE              = 不使用 (マスクなし=UVクランプ無効、Depthなし=髪シェル無効)
export type MaskSource = 'SELFIE_MULTICLASS' | 'NONE';
export type DepthSource = 'DAVID' | 'ARPORTRAIT_DEPTH' | 'NONE';
// 人物シルエット (UVクランプ・Depth cleanup境界) の供給源。
// DAVID=ソフト前景セグ (512px)。crop外はSelfieMulticlassで補完。
export type PersonSource = 'DAVID' | 'SELFIE_MULTICLASS';
// 表面法線の供給源。DAVID=実測法線をObjectSpaceNormalMapとしてhead/髪に貼り、
// 回転時の照明応答を与える (写真の陰影 + 実測法線のシェーディング)。NONE=平坦(+Z)
export type NormalSource = 'DAVID' | 'NONE';

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
  normalSource: NormalSource;
  personSource: PersonSource;
  gnmNormalStrength: number; // 実測法線の強さ (0=平坦, 1=実測のまま)
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
  gnmShowHair: boolean; // 髪シェルメッシュの表示切替 (offで頭部のみ)
  // 実測法線から髪シェルの起伏 (毛束の凹凸) を作る強さ (0=Depthのみの滑らかな面)。
  // Depthは絶対位置、法線は高周波と役割を分けて融合する
  gnmHairRelief: number;
  // 髪マスクをGuided Filterで写真エッジへ整合させた精細版を使う (offで生の256px)
  gnmMaskRefine: boolean;
  gnmShowMouthInterior: boolean; // 口腔内 (口腔壁・歯・歯茎・舌) の表示切替
  // 口腔内の明るさ倍率。基準色は写真の下唇色なので、写真の露出のまま暗すぎ/明るすぎる
  // ときの手動補正 (1=写真の唇色そのままの比率)
  gnmMouthBrightness: number;
  // 開口時に舌を下げる量。1.0 = 公式デモGIFの舌スライダー姿勢そのまま
  // (tongue_mean=0.7 / tongue_000=-1.7。舌が奥へ11mm・下へ3.9mm)。顎の開き量に比例。
  // 0だとGNMのneutral姿勢のまま = 舌が口蓋に張り付いて開口部を埋め、歯も口腔も見えない
  gnmTongueDown: number;

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
  // レイヤー分離画像 (headテクスチャ / 髪だけの画像) を画面隅に表示
  showLayerImages: boolean;
  layerImageScale: number; // レイヤー画像の表示倍率 (1=基準高さ160px)

  // --- カメラ ---
  cameraFovDeg: number;
  cameraDistanceRatio: number; // faceWidth比

  // --- シーン ---
  backgroundColor: string; // 3Dビューポートの背景色 (hex)

  // --- 髪シェルGrid解像度 ---
  // DAViD depth (512px crop) の情報量を拾える密度が基準。
  // ARPortraitDepth (192x256) しか無い環境では下げても見た目は変わらない
  hairGridCols: number;
  hairGridRows: number;
}

export const DEFAULT_PARAMS: Params = {
  maxYawDeg: 15,
  maxPitchDeg: 12,

  maskSource: 'SELFIE_MULTICLASS',
  depthSource: 'DAVID',
  normalSource: 'DAVID',
  personSource: 'DAVID',
  gnmNormalStrength: 1.0,
  measuredDepthGain: 1.0,

  gnmIdentityReg: 1.0,
  gnmDenseFit: true,
  gnmWarpStrength: 1.0,
  gnmHairLift: 0.02,
  gnmHairRolloff: 0.08,
  gnmShowHair: true,
  gnmHairRelief: 1.0,
  gnmMaskRefine: true,
  gnmShowMouthInterior: true,
  gnmMouthBrightness: 1.0,
  gnmTongueDown: 1.0,

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
  showLayerImages: false,
  layerImageScale: 1.0,

  cameraFovDeg: 30,
  cameraDistanceRatio: 3.4,

  backgroundColor: '#14161a', // style.cssの--bgと同じ初期値 (透過時と見た目が変わらないように)

  hairGridCols: 96,
  hairGridRows: 120,
};

export function createParams(): Params {
  return { ...DEFAULT_PARAMS };
}
