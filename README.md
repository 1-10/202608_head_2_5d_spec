# GNM Head Exporter (Web)

正面写真 1 枚から、Unity 向けの guest zip をブラウザ内で書き出す。

**実装の正本は [1-10/2608_Obayashi_GNMHeadExporter](https://github.com/1-10/2608_Obayashi_GNMHeadExporter)**
（Windows デスクトップアプリ）。このリポジトリはその**ブラウザ移植**で、パイプラインの段・アルゴリズム・
既定値・出力契約はすべてあちらと同じものを持つ。**差分は「web だから必要になった / 消えた」ものだけ**で、
一覧は下の「web だから違うところ」にある。それ以外の違いを見つけたら移植の漏れなので、あちらへ揃える。

**3D 確認ビューだけは正本が違う。** guest zip を実際に組み立てて画にするのは Unity 側
（[1-10/2607_Obayashi_Avatar_Mockup_3DGS](https://github.com/1-10/2607_Obayashi_Avatar_Mockup_3DGS) の
`Assets/Sandbox/Ooba/GNM`）なので、**ビューの絵はあちらに揃える**。デスクトップ側の 3D ビューは
パイプラインの検査が目的で、消費側の絵とは別物 — こちらの絵がそちらに揃っていると
「web で見て良かったが Unity で崩れる」が起きる。領域の分け方・口腔内の固定色・カメラ・光・
alpha clip・首と視線・表情プリセットは Unity 側が正本。

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
- 初回に GNM アセット約 32MB と DAViD モデル（fp16 691MB / int8 338MB）を読む

## セットアップ

```bash
npm install
npm run dev
```

`npm run dev` 後、表示されたローカル URL（既定 `http://localhost:5173`）をブラウザで開く。Webカメラを
使う場合は HTTPS または localhost 経由でのアクセスが必要。

### GNM アセットの生成

`public/gnm/gnm_head.gnmb`（約 32MB）はリポジトリに同梱している。**再生成が要るのは公式アセットの版を
上げるときだけ**（Python 3.10 以上 + numpy が必要）:

```bash
python tools/fetch_gnm_assets.py     # 公式 npz / 68 点定義 / canonical を取得（URL とハッシュは固定）
python tools/export_gnm_assets.py    # public/gnm/gnm_head.gnmb を生成
```

表情プリセット（`tools/GnmExpressionPresets_v3_0.npz`）は同梱してある。正本は Unity 側の
`Tools/export_expression_presets.py`（公式 CVAE デコーダを latent 0 = クラス条件付き平均で回して
20 本に焼いたもの）で、**GNM の版を上げたときだけ**あちらで作り直して差し替える。

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
npm test         # domain / application の検査（実アセットを読む。87件）
```

### 正本との突き合わせ

**正本の `domain` は numpy 以外を import しない**ので、PySide6 / mediapipe / onnxruntime を入れずに
そのまま動かせる。同じ合成入力で `export_guest` を通し、その結果と移植の結果を数値で比べる:

```bash
git clone https://github.com/1-10/2608_Obayashi_GNMHeadExporter ../gnm-exporter
python tools/golden_export_guest.py ../gnm-exporter/src tests/golden/exportGuest.json
npm test
```

**定数の一致はテキストで確かめられるが、アルゴリズムの一致は数値でしか確かめられない。** 基準値を
作り直したときに差が桁で動いたら、移植のどこかが変わっている。

現時点の実測（相対差）:

| 突き合わせた値 | 差 |
|:--|:--|
| 密対応（`denseVertexIndices` / 重み / 残差 / 辺の中央値） | **完全一致** |
| 眼球テクスチャ（左右）・`hair_alpha`・`hair_albedo` | **完全一致** |
| 髪シェルの頂点数・三角形数 | **完全一致**（9,766 / 19,138） |
| `skin_albedo` の総和 | 8.4e-8（786,432 テクセルのうち数個が 1 階調） |
| 髪シェルの `positions` | 1.8e-5（float32 蓄積 vs float64） |
| `identity` 係数 | 3.0e-4（**int16 量子化のぶん**。あちらが量子化をやめて測った差 7e-4 の内側） |

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

### 3D ビュー

書き出しの直後にフィット結果からシーンを作る。**絵は Unity 側（`Assets/Sandbox/Ooba/GNM`）に揃えて
ある** — 下の値はあちらの写しなので、ズレたら `tests/viewer.test.ts` と `tests/preview.test.ts` が
落ちる。

| | 値 | 正本 |
|:--|:--|:--|
| 投影 | 透視 FOV 20° / 距離 1.3m / 注視点 y=0.297m | `Scenes/Viewer.unity` の `MainCamera` |
| 背景 | `#26292e` | 同 `MainCamera` の `m_BackGroundColor` |
| 光 | 平行光 1 灯（上・前・被写体の右から）+ 環境光 | 同 `DirectionalLight` |
| 法線 | **+Z 固定**（立体感は写真に焼き込まれている） | `MT_GnmHeadOpaque` が `SG_Lit` を引く条件 |
| 髪 | alpha clip 0.3 | `MT_GnmHairTransparent` の `_Cutoff` |
| 領域 | 7 つ・先勝ち（`MouthSock` → `Skin` → `Teeth` → `Gums` → `Tongue` → `EyeLeft` → `EyeRight`） | `Editor/GnmHeadAssetBuilder` の `regions` |
| 除外 | 角膜（`eye_exteriors`）は描かない | 同 `excludedSelector` |
| 可動域 | 首 yaw ±15° / pitch ±12° / 視線 ±10°、首へ 30%・頭へ 70% | `Viewer/GnmHeadPoseController` |
| 表情 | 20 プリセット・同時に 1 本・立ち上がり 0.35s / 保持 0.8s | `Viewer/GnmExpressionPlayer` |

**口腔内は写真の色を読まない。** 歯 `(190,164,164)` / 歯茎・舌 `(114,53,53)` / 口腔壁 `(80,37,37)` の
固定色で塗る（正本は `MT_GnmTeeth` / `MT_GnmGums` / `MT_GnmTongue` / `MT_GnmMouthSock` の `_BaseColor`）。
歯・歯茎・舌がゲスト共通の固定色になった以上、口腔壁だけ写真の肌色に追随すると開口時に隣り合う面で
色の決め方が割れる。exporter は今も `0.7 × 顔の肌の平均色` を `skin_albedo` へ焼くが、**Unity も web も
その領域を読まない**。

`mouth_sock` は `skin` の部分集合なので `MouthSock` を `Skin` より前に置く（`Skin` 側を `-mouth_sock`
で引く形は使えない — 境界の三角形は `mouth_sock` 頂点を一部だけ含むので、引くとどちらのマスクも 3 頂点
揃わず未割り当てになる）。どの領域にも入らなかった三角形はマゼンタで出る（見えたら領域の設定か
アセットが変わっている）。

| 操作 | 割り当て |
|:--|:--|
| カメラを周回 | 左ドラッグ |
| 首と視線を振る（0.25°/px） | **Shift + 左ドラッグ** |
| 平行移動 | 右ドラッグ |
| 拡大 | ホイール |
| 層の表示（肌 / 眼球 / 口腔内 / 髪シェル） | `1` `2` `3` `4` |
| 層ごとのテクスチャ（OFF で下地色と陰影だけ） | `A` `S` `D` `F` |
| 全テクスチャをまとめて切り替え | `T` |
| ワイヤーフレーム | `W` |
| 正面・無表情に戻す | `R` |

首と視線はパネルのスライダーでも動かせ、「マウス追従」で画面上のカーソルを追う（視線は首より速く
動く）。表情は 20 本のスライダーと自動再生（順番に / ランダム）で駆動する。**同時に立てるのは 1 本
だけ** — プリセットは加算変位なので重ねると顔が壊れるうえ、確認用途では「今どれか」が分かる方が役に
立つ。自動まばたきだけは独立した層として上に足す。

テクスチャを外せるのは、絵の中の暗い所が「写真にそう写っていた」のか「面が無い・法線が逆・前後関係が
壊れている」のかを分けるため。

パラメータは「パラメーターを保存」で `localStorage` へ書き、次回起動時に復元する。**書き出しの値と
ビューの値はキーを分けてある**（`export_parameters/v1` / `view_parameters/v1`）。書き出しの値は検査を
通らなければ使わず `application` 側の既定へ戻し、ビューの値は結果に影響しないので範囲へ丸めて使う。

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
| 3D ビューの姿勢・表情（`src/domain/preview/`） | **書き出しの契約には入らない。** guest が Unity でどう出るかを確認するために、あちらの `GnmHeadInstance` / `GnmSkeleton` / `GnmExpressionPlayer` と同じ動きを持つ。デスクトップ側の 3D ビューは姿勢を持たない |
| GNMB へ vertex group / ジョイント / スキニング重み / 表情プリセット（+3.3MB） | 上記の入力。`export_guest` は 1 つも読まない。前提（bind pose に回転が無い・pose correctives が全ゼロ・重み和 1・影響ボーン 2 本）はアセット生成時に毎回検査して、崩れたら生成が落ちる |
| 自動まばたき・ワイヤーフレーム・背景色・FOV と距離の調整 | 旧 web 版から残したもの。Unity 側には無いが、写真 1 枚から起こした頭を確認するのに効く |

### web だから消えたもの

| 差分 | 理由 |
|:--|:--|
| CLI・複数枚のまとめ書き出し | 入口が 1 つ（画面）。既定値と失敗時の扱いは同じ `application` にある |
| 推論モデルのローカル取得 | 配布元 URL から実行時に読む（`/1/` のバージョン付きパスで固定） |
| アトラスレイアウトの永続キャッシュ | ブラウザに置き場が無い。セッション内の使い回しだけ持つ（実測 2048² で layout 173ms / bake 1.8s） |
| CUDA カーネル・ONNX CUDA EP の box filter | 置き換える先が無いので domain の実装をそのまま呼ぶ。**品質判断は元から domain にある** |
| zip の一時ファイル → rename | Blob を作り終えてから初めてダウンロードが始まるので、「最終名のファイルが現れた時点で必ず完成している」が構造的に満たされる |
| 段ごとの開発用スクリプト（`tools/bake_atlas.py` 等） | 同じ役割を画面の検査画像とパラメータパネルが担う（写真を差し替えて即座に全段の絵が出る） |
| numpy のチャンク上限（`CANDIDATE_CHUNK_BUDGET` 等） | ベクトル化のための中間配列そのものが無い（テクセルごとの素直なループ） |
| 3Dビューの遅延生成（`build_viewer_scene`） | 3Dビューが画面の主役で常に見えているので、門を置く先が無い |
| デスクトップ側の 3D ビューの正射影・領域の分け方 | **ビューの正本を Unity 側へ替えた。** 消費側の絵に揃える方が確認として意味があるので、あちらの `debug_scene` の分け方（成分ごと・口腔内を一律の色）は使わない |
| Unity 側のトゥーン表示（`SH_Toon`） | 移植していない。実法線（内角重み）と輪郭線の 2 パスが要り、**写真そのままの「リアル」の側が確認したい絵**なので後回しにした |

### web だから揃わないもの

| 差分 | 理由 |
|:--|:--|
| JPEG のクロマサブサンプリング | canvas の `toBlob('image/jpeg', 0.9)` は 4:2:0 になり、指定できない（あちらは Pillow で 4:4:4）。**同じ写真から作った zip がバイト単位で一致しないのはこれが理由** |
| 顔検出の解像度の階段 | あちらは長辺 256〜3840 を全段回して検出を束ねる。ブラウザでは 1 枚あたり数百 ms × 段数が体感に出るので、写真の解像度で 1 回だけ検出する。**主役の規則（得点 = 一辺 − 対象点からの距離）は共有している** |
| 内部の蓄積精度 | numpy が float32 で足すところを JS は倍精度で足す（JS の数は倍精度しか無い）。**移植の方が精度が高い**側の差で、実測 1.8e-5。**guest.json に出る `identity` だけは float32 の精度へ丸める**（あちらが `astype(np.float32)` している位置と同じ） |

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
  fetch_gnm_assets.py       # 公式 npz / 68 点定義 / canonical の取得（URL とハッシュを固定）
  export_gnm_assets.py      # npz → ブラウザ用 GNMB アセット
  prepare_david_model.py    # DAViD の fp16 / int8 変換
  golden_export_guest.py    # 正本の domain を動かして突き合わせ基準を作る
  GnmExpressionPresets_v3_0.npz  # 表情プリセット 20 本（正本は Unity 側の export_expression_presets.py）
src/
  domain/          # 純粋計算（contract / field / photo / ramp / normal / faceSubject / inspection /
                   #   gnm / atlas / eyes / hair）
    preview/       # 3D ビューだけが使う層（asset / pose / expression / scene）。**正本は Unity 側**
  application/     # ユースケースと Port（settings / ports / exportGuest）
  infrastructure/  # Port の実装（gnmb / gnmAsset / packaging / imaging / photoCanvas /
                   #   faceLandmarks / segmentation / depthNormal / atlasBaker / hairImage）
  composition.ts   # 配線（具体実装を組み立てて Port として注入するのはここだけ）
  presentation/    # 入口（main / viewer / gui / inspectionView / input / parameterStore /
                   #   viewSettings / style.css）
tests/             # domain / application の検査（実アセットを読む。推論は偽の Port）
  golden/          # 正本の domain を動かして作った突き合わせ基準
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
