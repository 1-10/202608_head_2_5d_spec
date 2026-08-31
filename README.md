# GNM Head Exporter (Web)

正面写真 1 枚から、Unity 向けの guest zip をブラウザ内で書き出す。

**実装の正本は [1-10/2608_Obayashi_GNMHeadExporter](https://github.com/1-10/2608_Obayashi_GNMHeadExporter)**
（Windows デスクトップアプリ）。このリポジトリはその**ブラウザ移植**で、パイプラインの段・アルゴリズム・
既定値・出力契約はすべてあちらと同じものを持つ。**差分は「web だから必要になった / 消えた」ものだけ**で、
一覧は下の「web だから違うところ」にある。それ以外の違いを見つけたら移植の漏れなので、あちらへ揃える。

すべての処理はブラウザ内（クライアントサイド）で完結し、写真を外部サーバーへ送信しない。

出力は `guest_<YYYYMMDDhhmmss>.zip` のダウンロード。

| zip の中身 | 内容 |
|:--|:--|
| `guest.json` | identity 係数・テクスチャの一辺・UV の原点・色空間・版とアセットの識別 |
| `skin_albedo.jpg` | 肌のアトラス（GNM 公式 UV） |
| `left_eye_albedo.png` / `right_eye_albedo.png` | 眼球テクスチャ |
| `hair_shell.bin` / `hair_albedo.jpg` / `hair_alpha.png` | 髪シェル（髪が写っていなければ入らない） |

## 動作要件

- Chrome / Edge の最近の版（**WebGPU があれば DAViD を fp16 で走らせる**。無ければ WASM + int8 へ
  落ちる — どちらで動いたかは画面に出る）
- 初回に GNM アセット約 29MB と DAViD モデル（fp16 691MB / int8 338MB）を読む

## セットアップ

```bash
npm install
npm run dev
```

`npm run dev` 後、表示されたローカル URL（既定 `http://localhost:5173`）をブラウザで開く。Webカメラを
使う場合は HTTPS または localhost 経由でのアクセスが必要。

### GNM アセットの生成

`public/gnm/gnm_head.gnmb`（約 29MB）はリポジトリに同梱している。**再生成が要るのは公式アセットの版を
上げるときだけ**（Python 3.10 以上 + numpy が必要）:

```bash
python tools/fetch_gnm_assets.py     # 公式 npz / 68 点定義 / canonical を取得（URL とハッシュは固定）
python tools/export_gnm_assets.py    # public/gnm/gnm_head.gnmb を生成
```

生成物は**デスクトップ側が npz から作る値と同じもの**を、ブラウザが読める形（GNMB コンテナ）にした
ものである。判断（何を読むか・領域の作り方・密対応の作り方）はあちらの
`infrastructure/gnm_asset.py` と `domain/gnm/dense.py` が正本で、`tools/export_gnm_assets.py` は
その移植。

### DAViD モデルの配信

DAViD の ONNX（fp16 691MB / int8 338MB）はリポジトリに含めず、Hugging Face Hub の公開モデル
リポジトリ（`harry00902/202608_head_2_5d_spec`）から配信している。`public/david/` は Git 管理外なので、
Vercel 等へ Git ベースでデプロイしてもこのファイルは含まれない — ブラウザが直接 HF Hub へ `fetch`
する構成なので、デプロイ先には何もアップロードしなくてよい。

再生成・再アップロードする場合:

```bash
python tools/prepare_david_model.py
hf upload harry00902/202608_head_2_5d_spec public/david/david-multitask-vitl16-fp16.onnx david/david-multitask-vitl16-fp16.onnx --repo-type model
hf upload harry00902/202608_head_2_5d_spec public/david/david-multitask-vitl16-int8.onnx david/david-multitask-vitl16-int8.onnx --repo-type model
```

### その他コマンド

```bash
npm run build    # 型チェック + 本番ビルド
npm run preview  # ビルド結果のプレビュー
npm test         # domain / application の検査（実アセットを読む）
```

## パイプライン

6 段。段の名前・順序・各段の中身はデスクトップ側と同じ（`src/application/exportGuest.ts`）。

    1. 推論      ランドマーク / 髪マスク / 深度・法線・前景
    2. フィット   468 点密対応を重ねて相似変換と identity 係数を得る（耳・首は体肌輪郭へ追加フィット）
    3. 眼球      写真の画素を眼球の極座標 UV へ焼く（左右 2 枚）
    4. アトラス   写真を公式 UV アトラスへ焼き、残りを表面沿いの伝播で埋める
    5. 髪シェル   深度 + 髪マスクからグリッドメッシュを作る
    6. 組み立て   出力契約の値にまとめる

**顔ランドマークが先**で、そこから DAViD に渡す切り出しを決める（GNM メッシュが写真のどこを占めるかで
決まる。画像の中央正方形で切ると、縦長の写真で顎から下が推論の外へ落ちて肌アトラスの胸が肌色に埋まる）。
切り出しは 2 つ走らせる — 深度・法線は頭部だけを覆う詰めた正方形、人物前景はメッシュ全体を覆う広い方。

各段の出力は**検査画像**としてそのまま画面に並ぶ。

### 調整パラメータ

既定は `src/application/settings.ts` が正本で、**全部の一覧は画面右のパネルが出す**（ここに写すと片方
だけ動いたときに黙って嘘になる）。下の表はよく触るものだけ。

| パラメータ | 既定 |
|:--|:--|
| 肌アトラスの一辺 | 2048（512 / 1024 / 2048 / 4096） |
| 眼球テクスチャの一辺 | 256（128 / 256 / 512 / 1024） |
| 髪テクスチャの長辺 | 2048（512 / 1024 / 2048 / 4096） |
| 事前分布の強さの倍率 | 1.0（0.1〜10。大きいほど平均顔寄り） |
| identity 係数の上限 | 上限なし（置くなら 0.1〜20） |

## web だから違うところ

**ここに挙がっていない差分は移植の漏れ**として扱う。

### web だから増えたもの

| 差分 | 理由 |
|:--|:--|
| `tools/export_gnm_assets.py`（npz → GNMB） | ブラウザは npz を読めない。あちらは npz を直接読むのでこの段を持たない |
| identity 基底の int16 量子化 | 送るバイト数（float32 で 56MB / int16 で 28MB）。最大誤差は生成のたびに表示され、あちらの実測（残差の 5 桁目・誤差 146nm）が根拠 |
| WebGPU → WASM のフォールバック | 実行環境を利用者が選べない。あちらは CUDA が無ければ起動時に落とす。どちらで動いたかは画面に出す（黙って遅くしない） |
| Webカメラ入力 | ブラウザにしかない入力経路 |
| 検査画像を画面に並べる | 書き出す先が無い（あちらは写真ごとのディレクトリへ PNG） |

### web だから消えたもの

| 差分 | 理由 |
|:--|:--|
| CLI・複数枚のまとめ書き出し | 入口が 1 つ（画面）。既定値と失敗時の扱いは同じ `application` にある |
| 推論モデルのローカル取得 | 配布元 URL から実行時に読む（`/1/` のバージョン付きパスで固定） |
| アトラスレイアウトの永続キャッシュ | ブラウザに置き場が無い。セッション内の使い回しだけ持つ（実測 2048² で layout 173ms / bake 1.8s） |
| CUDA カーネル・ONNX CUDA EP の box filter | 置き換える先が無いので domain の実装をそのまま呼ぶ。**品質判断は元から domain にある** |
| zip の一時ファイル → rename | Blob を作り終えてから初めてダウンロードが始まるので、「最終名のファイルが現れた時点で必ず完成している」が構造的に満たされる |

### web だから揃わないもの

| 差分 | 理由 |
|:--|:--|
| JPEG のクロマサブサンプリング | canvas の `toBlob('image/jpeg', 0.9)` は 4:2:0 になり、指定できない（あちらは Pillow で 4:4:4）。**同じ写真から作った zip がバイト単位で一致しないのはこれが理由** |
| 顔検出の解像度の階段 | あちらは長辺 256〜3840 を全段回して検出を束ねる。ブラウザでは 1 枚あたり数百 ms × 段数が体感に出るので、写真の解像度で 1 回だけ検出する。**主役の規則（得点 = 一辺 − 対象点からの距離）は共有している** |

## 消費側へ渡すもの

**guest zip だけ。** 頭部のジオメトリは渡さない — 頂点は消費側が
[公式 GNM](https://github.com/google/GNM) の `template_vertex_positions` と `vertex_identity_basis` に
`guest.json` の `identity` を当てて作る。

```
頂点 = template_vertex_positions + Σ identity[i] * vertex_identity_basis[i]
```

契約の詳細（UV の向き・色空間・口腔内の色を運ばない理由・眼球が左右 2 枚である理由）は
`src/domain/contract.ts` が正本。

## ディレクトリ構成

デスクトップ側と同じレイヤー構成（依存は内向きのみ）。

```text
index.html
tools/
  fetch_gnm_assets.py     # 公式 npz / 68 点定義 / canonical の取得（URL とハッシュを固定）
  export_gnm_assets.py    # npz → ブラウザ用 GNMB アセット
  prepare_david_model.py  # DAViD の fp16 / int8 変換
src/
  domain/          # 純粋計算（contract / field / photo / ramp / normal / faceSubject / inspection /
                   #   debugScene / gnm / atlas / eyes / hair）
  application/     # ユースケースと Port（settings / ports / exportGuest）
  infrastructure/  # Port の実装（gnmb / gnmAsset / packaging / imaging / photoCanvas /
                   #   faceLandmarks / segmentation / depthNormal / atlasBaker / hairImage）
  composition.ts   # 配線（具体実装を組み立てて Port として注入するのはここだけ）
  presentation/    # 入口（main / viewer / gui / inspectionView / input / interaction / style.css）
tests/             # domain / application の検査（実アセットを読む。推論は偽の Port）
```

## 技術スタック

- Google GNM Head（真3Dパラメトリック頭部。identity 253 成分の線形 basis）— Apache-2.0
- MediaPipe Face Landmarker / Image Segmenter (SelfieMulticlass)（`@mediapipe/tasks-vision`）
- DAViD multi-task（Depth / 法線 / 前景を 1 回の推論で。`onnxruntime-web`）
- Three.js（3D ビュー）/ lil-gui（パラメータパネル）/ fflate（zip）
- Vite + TypeScript + Vitest

### モデルのライセンス

- MediaPipe（FaceLandmarker / SelfieMulticlass）: Google 公式配布（Apache-2.0）。モデルカード上、
  学習データも Google 自社収集（同意取得済み）のもののみ。**学習データまで商用クリーン**
- DAViD: モデル MIT、学習データ SynthHuman は CDLA-Permissive-2.0（100% 合成データ）。
  **学習データまで商用クリーン**
- GNM Head (google/GNM): **Apache-2.0**。学習データは約 5,000 人の自社スタジオ収録3Dスキャン
  (arXiv:2607.23687) だが、被写体同意の明示記載はモデルカード/論文で未確認。商用出荷物に含める場合は
  同意取得の確認を推奨
