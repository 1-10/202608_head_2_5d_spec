# 202608_head_2_5d_spec

1枚の正面人物写真から、MediaPipe Face Landmarkerと古典的なDepth補間だけで疑似3D頭部を生成し、
`FACE ONLY`(顔メッシュのみ)と`FULL HEAD`(頭部シルエット・髪ボリュームを含む1枚の連続メッシュ)を
同一画面で左右比較できる、品質検証用WebGLプロトタイプです。

詳細な設計方針・アルゴリズムは [claude_code_head_2_5d_spec.md](./claude_code_head_2_5d_spec.md) を参照してください。

3D Gaussian Splatting・NeRF・Diffusion・Novel View Synthesis・生成AIによる不可視領域補完は使用していません。
すべての処理はブラウザ内(クライアントサイド)で完結し、画像を外部サーバーへ送信することはありません。

## できること

- Webカメラ撮影 / ローカル画像ファイルからの正面写真入力
- MediaPipe Face Landmarkerによる顔ランドマーク検出
- `FACE ONLY`: 顔ランドマークとcanonicalな顔Depth profileを混合した2.5D顔メッシュ
- `FULL HEAD`: Face Depth Field + 擬似Head Depth + 輪郭ロールオフ + 髪ボリュームを合成した
  頭部全体の連続メッシュ(`HEAD DEPTH ONLY` / `FACE + HEAD` / `FACE + HEAD + HAIR VOLUME`の3モード)
- 左右いずれかのビューをドラッグすると、両ビューが同じYaw(±可変)/Pitch角に同期回転
- 周期的なBlink(目パチ)アニメーション
- Mouth Seam(唇の境界)分離による古典的なTalk Animation / Mouth Cavity(生成AIによる口腔内補完は不使用)
- 品質比較用GUI: Depthパラメータ、Camera/Rotation、Talk/Mouth、各種デバッグ表示
  (Wireframe / Landmarks / Head Mask / Face Depth / Final Depth / Mouth Seam / Mouth Region)

## セットアップ

```bash
npm install
npm run dev
```

`npm run dev`後、表示されたローカルURL(既定 `http://localhost:5173`)をブラウザで開いてください。
Webカメラ機能を使う場合はHTTPS、またはlocalhost経由でのアクセスが必要です。

### その他コマンド

```bash
npm run build    # 型チェック + 本番ビルド
npm run preview  # ビルド結果のプレビュー
```

## 技術スタック

- Three.js (WebGL, CPU側でgeometry頂点を生成・更新する方式)
- MediaPipe Face Landmarker (`@mediapipe/tasks-vision`)
- Delaunator (Face Mesh topologyのDelaunay三角形分割)
- lil-gui (品質比較用パラメータパネル)
- Vite + TypeScript

## ディレクトリ構成

```text
index.html
src/
  main.ts          # エントリポイント。UI配線・シーン構築・レンダーループ
  input.ts         # Webcam / ファイル入力
  faceDetector.ts  # MediaPipe Face Landmarkerのロードと推論
  faceTopology.ts  # landmark正規化・三角形分割・key landmark index
  faceDepth.ts     # canonical/MediaPipe Depthの合成、Face Depth Field
  headMask.ts      # 頭部シルエットマスク(楕円近似)
  headDepth.ts     # Pseudo Head Depth / Edge Rolloff / Face-Head Blend / Hair Volume
  faceOnlyMesh.ts  # FACE ONLYメッシュ生成
  fullHeadMesh.ts  # Head Grid Mesh生成 (FULL HEAD)
  mouthTalk.ts     # Mouth Seam / Talk Animation / Mouth Cavity
  animation.ts     # Blinkアニメーション
  interaction.ts   # ドラッグによるYaw/Pitch操作
  debugView.ts     # GUIパラメータパネル・デバッグ表示
  params.ts        # 調整パラメータの一元管理
```

## 品質目標・評価対象外の範囲

Yaw ±15°程度の限定視点を対象とし、完全な3D復元は目的としていません。品質目標や比較の観点の詳細は
[claude_code_head_2_5d_spec.md](./claude_code_head_2_5d_spec.md) の「品質目標」章を参照してください。
