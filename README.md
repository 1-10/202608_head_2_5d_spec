# 202608_head_2_5d_spec

1枚の正面人物写真から、Google GNM Head (真3Dパラメトリック頭部モデル, Apache-2.0) を
468点顔ランドマークへフィットし、写真を投影テクスチャとして貼った疑似3D頭部を
ブラウザ内で生成・アニメーションさせるWebGLプロトタイプです。

3D Gaussian Splatting・NeRF・Diffusion・Novel View Synthesis・生成AIによる不可視領域補完は使用していません。
すべての処理はブラウザ内(クライアントサイド)で完結し、画像を外部サーバーへ送信することはありません。

初期の2.5D relief方式 (FACE ONLY / FULL HEAD比較プロトタイプ) の設計資料は
[claude_code_head_2_5d_spec.md](./claude_code_head_2_5d_spec.md) に残っていますが、
実装はGNM方式へ完全移行済みです。

## できること

- Webカメラ撮影 / ローカル画像ファイルからの正面写真入力
- MediaPipe Face Landmarkerによる顔ランドマーク検出 (468点)
- GNM Headのフィッティングと描画:
  - 468点密対応 (barycentric) の正則化最小二乗フィット (GUIで68点フィットと比較可)
  - 残差ワープ: identity係数では張り切れない目・唇の位置残差をneutral頂点へ焼き込み、
    まばたき・開口が写真の目・口の位置で起こる
  - 鼻孔の内壁を平滑化で封止 (穴のジオメトリは不要 — 写真の鼻孔の暗さで表現)
  - 正面写真の平行投影テクスチャ + シルエット外/背面は写真色の頂点色へフェード
- 口腔内 (口腔壁・歯・歯茎・舌): GNM Head同梱のジオメトリをそのまま別メッシュで描く。
  色・マテリアル・法線の扱いはGNM公式の可視化コード (`visualization/vertex_colors.py`,
  `visualization/gnm_pyrender.py`) からの移植で、写真から取るのは基準色 (肌の平均色) だけ。
  舌の姿勢は公式デモGIFのスライダー値そのまま (`Tongue Pose`)
- 実測髪シェル: 実測髪マスク+実測Depthの前面シェルをGNMの手前に重ねる。
  シルエット・髪マスク・Depthの供給源はGUIで切替できる:
  - `MEASURED`(既定): MediaPipe Image Segmenter (SelfieMulticlass)による実測シルエット/髪マスク +
    TensorFlow.js ARPortraitDepthによる実測人物Depth(前景dilation・外れ値clamp・平滑化済み)
  - `NEURAL`: BiRefNetのアルファマット(MediaPipeの意味分けと合成) + Depth Anything V2 SmallのDepth。
    選択時に初めてtransformers.jsごと遅延ロードする(下記ライセンス注意を参照)
  - `NONE`: 不使用 (UVクランプ・髪シェルなしの素のGNM)
- 表情アニメーション (GNM公式ExpressionSamplerをブラウザ内で実行):
  - 公式のCVAEデコーダ (`semantic_sampler.py`) をTypeScriptへ移植し、`sample_expression` /
    `blend_expressions` / `randomize_expressions` をそのまま使う (重み0.75MB, float16)
  - `Auto`: 公式Expressionクラス20種を巡回。潜在zも引き直すので同じクラスでも毎回変わる
  - `Random`: 公式 `randomize_expressions` (2〜3クラスをランダムに公式blend)
  - クラス固定: 公式Expressionクラス20種すべて選択可
  - `Manual`: パーツ別スライダー合成 (公式クラスの代表表情を目/下顔面領域に分けて加算。
    領域分割は公式に無い操作なのでこのモード限定)
  - 周期的なBlink(目パチ)を表情へ合成 (左右ウインククラスの合成から目領域だけ使う)
- ビューをドラッグするとYaw(±可変)/Pitch角に回転
- GUI: フィット/髪シェルパラメータ、表情、Camera/Rotation、Wireframe表示
- Unityエクスポート (本番構成の2段構え):
  - Export Template: GNM Headの型 (44 blendshape+口腔内+サンプラー重み) — Unityに1回だけ常駐させる
  - Export Guest: ゲスト固有データ (neutral頂点・写真・髪シェル・meta) — 毎回転送する
  - Unity側の読み込み仕様は [docs/unity_integration.md](./docs/unity_integration.md)

既知の制約: テクスチャは正面写真の焼き付きのため大表情では歪む。

## セットアップ

```bash
npm install
npm run dev
```

`npm run dev`後、表示されたローカルURL(既定 `http://localhost:5173`)をブラウザで開いてください。
Webカメラ機能を使う場合はHTTPS、またはlocalhost経由でのアクセスが必要です。

### GNMアセットの再生成

生成済みの `public/gnm/gnm_head_lite.bin` (約11.7MB) はリポジトリに同梱している。
成分数などを変えて再生成する場合のみ以下を実行する (Python + numpy が必要):

```bash
git clone --depth 1 https://github.com/google/GNM.git /tmp/GNM
curl -LO https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj
python tools/export_gnm_assets.py /tmp/GNM/gnm/shape/data/versions/v3_0/gnm_head.npz canonical_face_model.obj
# → public/gnm/gnm_head_lite.bin (約11.7MB) が生成される
python tools/export_gnm_sampler.py /tmp/GNM/gnm/shape/data/semantic_sampler/expression_decoder_model.h5                                    /tmp/GNM/gnm/shape/data/versions/v3_0/gnm_head.npz
# → public/gnm/gnm_expression_decoder.bin (約0.76MB) が生成される
# .obj (MediaPipe canonical face model) を省略すると468点密対応が省かれ、フィットが68点フォールバックになる
```

### その他コマンド

```bash
npm run build    # 型チェック + 本番ビルド
npm run preview  # ビルド結果のプレビュー
```

## 技術スタック

- Three.js (WebGL)
- Google GNM Head (真3Dパラメトリック頭部。identity 64成分 + 表情44成分の線形basis)
- MediaPipe Face Landmarker / Image Segmenter (SelfieMulticlass) (`@mediapipe/tasks-vision`)
- TensorFlow.js ARPortraitDepth (`@tensorflow-models/depth-estimation`)
- transformers.js (`@huggingface/transformers`): Depth Anything V2 Small / BiRefNet (NEURALソース選択時のみ遅延ロード)
- lil-gui (パラメータパネル)
- Vite + TypeScript

### モデルのライセンス

- `MEASURED`系 (MediaPipe / ARPortraitDepth): すべてGoogle公式配布(Apache-2.0)で、モデルカード上、
  学習データもGoogle自社収集(同意取得済み)のもののみ。**学習データまで商用クリーン**
- `NEURAL`系 (Depth Anything V2 Small / BiRefNet): **重みは商用可**(Apache-2.0)だが、
  学習データに非商用/非開示のもの(VKITTI2, SA-1B, 私有データ等)を含む。
  MEASURED系との**品質比較・評価用**の位置づけ。商用出荷物に含める場合は要法務判断
- `GNM Head` (google/GNM): **Apache-2.0**。学習データは約5,000人の自社スタジオ収録3Dスキャン
  (arXiv:2607.23687) だが、被写体同意の明示記載はモデルカード/論文で未確認。
  商用出荷物に含める場合は同意取得の確認を推奨

## ディレクトリ構成

```text
index.html
tools/
  export_gnm_assets.py   # GNM Head npz → ブラウザ用軽量アセット変換 (ビルド時)
  export_gnm_sampler.py  # 公式ExpressionSamplerのCVAEデコーダ重み → ブラウザ用 (ビルド時)
  verify_gnm_asset.py    # 生成アセットと公式npzのデータ突き合わせ検証
src/
  main.ts          # エントリポイント。UI配線・シーン構築・レンダーループ
  input.ts         # Webcam / ファイル入力
  faceDetector.ts  # MediaPipe Face Landmarkerのロードと推論
  faceTopology.ts  # landmark正規化 (モデル空間・テクスチャ空間)
  fields.ts        # 画像UV空間の2Dスカラー場 (マスク・Depthの共通表現)
  personSegmentation.ts # MediaPipe SelfieMulticlassによる実測シルエット/髪マスク
  portraitDepth.ts # TF.js ARPortraitDepthによる実測人物Depthとクリーンアップ
  neuralSources.ts # Depth Anything V2 / BiRefNet (NEURALソース。遅延ロード)
  gnmHead.ts       # GNM Headのアセット読込と写真へのフィッティング
  gnmHeadMesh.ts   # GNMメッシュ構築 (真3D頭部+実測髪シェル) と表情機構
  gnmRefine.ts     # 残差ワープ・鼻孔封止・眼球非貫通拘束などフィット後の品質改善
  gnmMouthInterior.ts # 口腔内 (口腔壁・歯・歯茎・舌) メッシュと舌の姿勢駆動
  gnmSampler.ts    # 公式ExpressionSampler (CVAEデコーダ) のブラウザ移植
  meshUtils.ts     # メッシュ共通処理 (法線+Z固定・格子index・smoothstep)
  blink.ts         # Blink(目パチ)の周期エンベロープ
  interaction.ts   # ドラッグによるYaw/Pitch操作
  debugView.ts     # GUIパラメータパネル
  params.ts        # 調整パラメータの一元管理
  unityExport.ts   # Unity向けzip (head.glb + meta.json) の書き出し
docs/
  unity_integration.md # Unity側の読み込み・再生ランタイム指示書
```

## 品質目標・評価対象外の範囲

Yaw ±15°程度の限定視点を対象とし、完全な3D復元は目的としていません。
