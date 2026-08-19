# Claude Code 実装指示: 正面写真からの2.5D Face / Full Head 比較WebGLプロトタイプ

## 目的

1枚の人物正面写真から、以下2方式の疑似3D表現を生成し、同一画面で左右比較できる品質評価用Webアプリケーションを実装する。

- 左: `FACE ONLY`
  - MediaPipe Face Landmarker由来の顔メッシュのみ
- 右: `FULL HEAD`
  - MediaPipe由来のFace Depth Field
  - 擬似Head Depth
  - 頭部シルエット
  - 髪の擬似ボリューム
  - これらを連続した1枚のHead Grid Meshとして合成

完全な3D復元は目的としない。正面を中心にYaw ±15°程度の限定視点で、「写真を平面のまま回している」ように見えず、頭部に立体感があることを目標とする。

本プロジェクトは技術・品質検証用であり、実装の理解性、調整可能性、比較の公平性を優先する。

---

## 禁止事項

以下は使用しないこと。

- 3D Gaussian Splatting
- NeRF
- Diffusion Model
- Novel View Synthesis
- AIによる横顔・不可視領域生成
- 有料API
- 有料SDK
- クラウド画像処理
- サーバーサイド画像処理
- 外部サービスへの画像アップロード

処理はブラウザ内で完結させる。

---

## 想定技術

- HTML / CSS / JavaScript または TypeScript
- Three.js
- WebGL
- MediaPipe Face Landmarker
- 必要に応じて MediaPipe Image Segmenter
- Canvas API
- Web Camera API
- Vite程度の軽量な開発環境は使用可
- Reactは不要

依存ライブラリは最小限にする。

---

## 入力

以下2種類に対応する。

1. Webカメラから正面写真を撮影
2. ローカルファイルから正面写真を選択

対象画像は真正面を向いた人物写真に限定する。

斜め顔の補正対応は不要。

入力後にMediaPipe Face Landmarkerで顔ランドマークを取得する。

顔が検出できない場合は明示的なエラーを表示する。

---

## 画面構成

中央を左右に2分割する。

左:

`FACE ONLY`

右:

`FULL HEAD`

両ビューは以下を共通化する。

- 入力画像
- Face Landmark
- 基準スケール
- カメラFOV
- カメラ距離
- Yaw角
- ライティング
- Blinkタイミング
- Talkタイミング

どちらか一方をドラッグすると、左右両方が同じYaw角に同期して回転する。

Yaw可動範囲:

-15°〜+15°

Pitch / Rollは固定でよい。

マウスドラッグとタッチドラッグの両方に対応する。

---

# FACE ONLY 実装

MediaPipe Face Landmarkerの顔ランドマークを用いて顔メッシュを生成する。

可能ならMediaPipeが提供する既知のFace Mesh topologyを使用する。

各ランドマークの2D位置と相対Zを利用し、正面写真に対応した2.5D顔面を構築する。

ただしMediaPipeのZ値をそのまま実寸Depthとして扱わない。

顔形状を安定化するため、MediaPipe Zとcanonicalな顔Depth profileを混合する。

概念:

```text
Zface = ZmediapipeNormalized * (1 - canonicalMix)
      + Zcanonical           * canonicalMix
```

初期値:

```text
canonicalMix = 0.3
```

canonical側では最低限以下の特徴を持たせる。

```text
noseTip      +0.10
noseBridge   +0.06
cheek        +0.03
upperLip     +0.035
chin         +0.015
eyeSocket    -0.015
faceContour  -0.04
```

単位は顔幅基準の正規化値。

入力画像を正面投影テクスチャとして使用する。

顔輪郭外は表示しない。

耳、髪、首、背景はFACE ONLYに含めない。

---

# FULL HEAD 実装方針

FULL HEADでは、Face Meshそのものを描画メッシュとして拡張しない。

代わりに、以下の方式を採用する。

```text
MediaPipe Face Landmarks
        ↓
Face Depth Field生成
        ↓
Head Grid Meshへ転写
        +
Pseudo Head Depth
        +
Head Silhouette Mask
        +
Hair Volume
        ↓
1枚の連続したFULL HEAD Mesh
```

これによりFace領域とHead領域の継ぎ目を抑える。

---

## Head Grid Mesh

FULL HEADは一定間隔のGrid Meshを使う。

初期候補:

```text
64 x 80 vertices
```

負荷に問題がなければ品質を見ながら増やしてよい。

各頂点は以下を持つ。

```text
X
Y
Z
U
V
alpha / mask相当
```

入力画像と正面投影でUVを対応させる。

---

## 正規化座標

顔幅を基準スケールにする。

```text
faceWidth = rightFaceX - leftFaceX
```

座標を概念的に以下へ正規化する。

```text
X = (imageX - headCenterX) / faceWidth
Y = (headCenterY - imageY) / faceWidth
Z = depth / faceWidth
```

物理単位ではなく、見た目調整用の正規化空間として扱う。

---

# Face Depth Field

MediaPipe Face Meshから、FULL HEAD用の2D Face Depth Fieldを作る。

手順:

1. Face Landmarkを2D平面上に配置
2. Face Mesh topologyでtriangleを構成
3. triangle内部のZをbarycentric interpolationする
4. 2D Depth FieldまたはTextureとして保持

内部解像度の初期値:

```text
256 x 256
```

Head Gridの各頂点は、UVに対応するFace Depth Fieldを参照する。

Face Mesh topologyとHead Grid topologyは分離する。

---

# Pseudo Head Depth

頭部の基本Depthは、正面から見た楕円球表面を模した関数で作る。

頭部中心:

```text
(cx, cy)
```

頭部横半径:

```text
rx
```

頭部縦半径:

```text
ry
```

正規化:

```text
nx = (x - cx) / rx
ny = (y - cy) / ry
r2 = nx * nx + ny * ny
```

基本Depth:

```text
Zhead = headDepthScale * sqrt(max(0, 1 - r2))
```

初期値:

```text
headDepthScale = 0.18
```

中央が手前、シルエットに近づくほど奥へ向かう曲面を作る。

---

# Head silhouetteからの巻き込み

Yaw ±15°でメッシュ端が紙の断面のように見えるのを防ぐため、外周を意図的に後方へ巻き込む。

```text
edge = clamp((r - edgeStart) / (1 - edgeStart), 0, 1)
```

```text
Zrolloff = -edgeDepth * smoothstep(0, 1, edge)
```

```text
ZheadFinal = Zhead + Zrolloff
```

初期値:

```text
edgeStart = 0.75
edgeDepth = 0.10
```

この2値はGUIから調整可能にする。

---

# Face DepthとHead Depthのブレンド

Face領域ではFace Depth Fieldを優先し、顔外へ向かうにつれてPseudo Head Depthへ滑らかに遷移する。

Face boundaryからの距離を求める。

```text
t = clamp(distanceFromFaceBoundary / blendWidth, 0, 1)
Wface = 1 - smoothstep(0, 1, t)
```

```text
Zfinal = ZfaceProjected * Wface
       + ZheadFinal    * (1 - Wface)
```

初期値:

```text
blendWidth = faceWidth * 0.15
```

Face boundary上で段差が発生しないことを優先する。

必要なら距離場を一度Canvas / Textureへ生成してよい。

---

# 額領域

MediaPipe Face Meshだけでは額上部〜頭頂方向の形状が不足する。

眉上から髪際に向かって、Face DepthからHead Depthへ独立した補間を行う。

概念:

```text
t = foreheadYNormalized

Zforehead = mix(
  Zbrow,
  Zhead,
  smoothstep(0, 1, t)
)
```

眉上から頭頂までが急激に凹まないこと。

---

# Hair Volume

髪をHead surfaceに完全に貼り付けない。

髪領域には頭蓋表面より少し手前方向の厚みを加える。

```text
Zhair = Zhead + hairVolume
```

実際には以下のように領域依存にする。

```text
hairVolume = hairVolumeMax
           * hairMask
           * volumeProfile
```

初期値:

```text
hairVolumeMax = 0.04
```

目安:

```text
top hair   +0.03〜0.06
side hair  +0.01〜0.03
```

精密な髪形状復元は行わない。

前髪がある場合に顔面へ完全に貼り付いて見えない程度の厚みを狙う。

---

# Head / Hair silhouette mask

FULL HEADでは頭部のみを描画し、矩形画像全体を曲げない。

人物頭部マスク:

```text
M(x, y) = 0.0〜1.0
```

を用意する。

Head Gridの透明度またはgeometry生成条件として使用する。

輪郭には2〜4px程度のsoft featherを入れる。

優先実装:

- MediaPipe Image Segmenter等をブラウザ内ローカル処理で使う

ただし、導入が複雑になりすぎる場合は初期フェーズでは簡易マスクでもよい。

重要なのはクラウド処理を使用しないこと。

髪領域を個別に得られない場合、人物マスク + Face Landmarks + 頭部ROIから推定してよい。

---

# Texture

入力画像をそのまま正面テクスチャとして使用する。

```text
u = imageX / imageWidth
v = 1 - imageY / imageHeight
```

FULL HEADもFACE ONLYも同一入力画像を使用する。

新しい側面画像を生成してはいけない。

Yaw ±15°で側面のtexture不足が見える場合は、以下程度の古典的処理は可。

- Edge Stretch
- UV Clamp
- 輪郭付近の軽微なtexture延長

生成AIによる補完は不可。

---

# Rotation Pivot

頭部のYaw回転中心を画像平面上に置かない。

顔中心より少し後方にPivotを配置する。

初期値:

```text
pivotZ = -0.07 * faceWidth
```

調整範囲の例:

```text
-0.02〜-0.15
```

GUIで変更可能にする。

鼻先中心で回っているような不自然な動きではなく、頭蓋中心に近い回転感を目指す。

---

# 目パチ

周期アニメーションでよい。

音声・カメラによるリアルタイム表情解析は不要。

MediaPipeの目周辺ランドマークを利用してMorphまたは頂点変形を行う。

左右同時Blink。

周期:

```text
3〜5秒に1回程度
```

Blink時間:

```text
150〜250ms
```

変形:

- 上まぶたを下へ
- 下まぶたを少し上へ

FACE ONLYとFULL HEADで完全同期する。

FULL HEADではFace Depth Fieldまたは該当Grid vertexへの変形として反映する。

---

# 口パク

周期アニメーションでよい。

音声入力同期は不要。

最低限以下の変形を行う。

- 下唇を下へ
- 上唇を少し上へ
- 口角を少し内側へ
- 顎をわずかに下・後方へ

アニメーション例:

```text
closed
→ small open
→ medium open
→ small open
→ closed
```

FACE ONLYとFULL HEADで完全同期する。

---

# 品質比較用UI

最低限以下を実装する。

上部:

- Webcam
- Upload Image
- Reset

中央:

- FACE ONLY
- FULL HEAD

下部またはサイドパネル:

- Current Yaw
- Blink ON/OFF
- Talk ON/OFF

品質評価用に以下の調整パラメータをGUIへ露出する。

| Parameter | Initial | Meaning |
|---|---:|---|
| Face Depth | 1.0 | MediaPipe顔凹凸倍率 |
| Canonical Mix | 0.3 | canonical顔Depth混合率 |
| Head Depth | 0.18 | 頭部全体の膨らみ |
| Edge Start | 0.75 | 輪郭巻き込み開始位置 |
| Edge Roll | 0.10 | 輪郭後退量 |
| Face/Head Blend | 0.15 | Face→Head遷移幅 |
| Hair Volume | 0.04 | 髪の厚み |
| Pivot Z | -0.07 | 回転中心奥行き |
| Max Yaw | 15° | 最大Yaw |

デバッグ表示:

- Show Wireframe
- Show Landmarks
- Show Head Mask
- Show Face Depth
- Show Final Depth

可能ならFULL HEADのモードを以下で切り替えられるようにする。

1. `HEAD DEPTH ONLY`
2. `FACE + HEAD`
3. `FACE + HEAD + HAIR VOLUME`

これは品質比較に重要なので、実装コストが低ければ優先する。

---

# 比較の公平性

FACE ONLYとFULL HEADで、顔そのものに関する入力条件を変えない。

以下は共通化する。

```text
same source image
same MediaPipe detection
same face landmarks
same canonical depth profile
same animation values
same camera
same lighting
same yaw
```

FULL HEADのみ、外周のHead surface / silhouette / hair volumeを追加する。

この構成により、頭部情報の追加が立体感にどの程度寄与するか評価可能にする。

---

# 推奨ファイル構成

過度に抽象化せず、責務ごとに分離する。

```text
index.html
src/
  main.ts
  input.ts
  faceDetector.ts
  faceTopology.ts
  faceDepth.ts
  headMask.ts
  headDepth.ts
  faceOnlyMesh.ts
  fullHeadMesh.ts
  animation.ts
  interaction.ts
  debugView.ts
  params.ts
```

JavaScriptで実装する場合は`.js`でよい。

---

# 実装フェーズ

## Phase 1

最優先で以下を動かす。

- ローカル画像入力
- MediaPipe Face Landmarker
- Face Mesh取得
- FACE ONLY生成
- 入力画像texture
- Three.js描画
- ±15°ドラッグ

Webcamはこのフェーズ後でもよい。

## Phase 2

- Head Grid Mesh
- 単純な楕円Pseudo Head Depth
- 頭部マスク
- FULL HEAD表示
- 左右同期Yaw

この時点でまずFULL HEADが「板」ではなく立体に見えることを確認する。

## Phase 3

- Face Depth Field
- Face / Head blend
- 額補間
- Edge rolloff
- Hair Volume

## Phase 4

- Blink
- Talk
- アニメーション同期

## Phase 5

- Webcam
- Wireframe
- Depth debug
- Mask debug
- GUIパラメータ調整

---

# 品質目標

## 正面 0°

入力写真との差を可能な限り小さくする。

## ±5°

明確に平面ではなく、顔・頭部の立体として見える。

## ±10°

鼻、頬、額、頭部外周の前後関係が自然に感じられる。

## ±15°

多少のtexture stretchは許容する。

ただし以下は避ける。

- 紙を曲げただけに見える
- Face MeshとHeadの境界に段差がある
- 髪が顔面に貼り付いて見える
- 頭部輪郭に極端な薄い断面が見える
- 回転中心が鼻先に見える

±15°より外は品質保証対象外。

---

# Claude Codeへの実装指示

まず既存リポジトリの構造を確認すること。

既存コードがある場合は、それを無視して全面書き換えせず、最小変更で上記機能を組み込むこと。

リポジトリが空の場合のみ、新規Vite + TypeScript + Three.js構成を作成してよい。

最初から全機能を一括実装しようとせず、Phase 1から順に動作確認可能な状態を作ること。

各Phaseの終了時点で、ブラウザ上で視覚確認できる状態にする。

アルゴリズム部分ではマジックナンバーを散在させず、`params.ts`等で調整値を集約すること。

特に以下の処理は関数として明確に分離すること。

```text
normalizeFaceLandmarks()
buildCanonicalFaceDepth()
buildFaceDepthField()
computePseudoHeadDepth()
computeEdgeRolloff()
computeFaceHeadBlendWeight()
computeHairVolume()
buildHeadGridGeometry()
updateYaw()
updateBlink()
updateTalk()
```

Depth計算は後から比較・置換しやすい実装にする。

FULL HEADの最終Z値について、デバッグ時に各成分を確認できるようにする。

最低限、以下を個別表示可能にする。

```text
FaceDepth
HeadDepth
EdgeRolloff
HairVolume
FinalDepth
```

GPU shader化は初期実装では不要。

まずCPU側でgeometry頂点を生成・更新する方式で実装し、品質確認を優先する。

パフォーマンス上必要になった場合のみShaderへの移行を検討する。

---

# 最終目的

このプロトタイプで確認したい問いは以下。

> 1枚の正面人物写真、MediaPipe Face Landmark、古典的なDepth補間、シルエット処理、WebGLだけで、Yaw ±15°の限定条件ならどこまで自然な疑似3D人物頭部を作れるか。

特に以下を比較評価する。

```text
FACE ONLY
vs
FULL HEAD
```

さらにFULL HEAD内部でも、

```text
Pseudo Head Depth only
vs
Face + Head
vs
Face + Head + Hair Volume
```

を比較できる設計にすること。
