# Head 2.5D

正面写真 1 枚から頭部を起こし、ブラウザの中で動かして確かめるツール。写真の顔を
[Google GNM Head](https://github.com/google/GNM) のパラメトリック頭部へフィットさせ、肌・眼球・髪の
テクスチャを写真から焼き、そのまま 3D ビューへ出す。表情・口形・Webカメラの表情トラッキングで、
起こした頭がどう動くかまで見られる。

**処理はすべてブラウザ内で完結する。写真を外部サーバーへ送らない。** 推論に使うモデルは配布元から
読み込むが、写真そのものがブラウザの外へ出ることはない。

実装の正本は Windows デスクトップアプリの
[1-10/2608_Obayashi_GNMHeadExporter](https://github.com/1-10/2608_Obayashi_GNMHeadExporter)。この
リポジトリはそのブラウザ移植で、パイプラインの段・アルゴリズム・既定値・出力契約はあちらへ揃える。
3D ビューの絵だけは正本が違い、
[1-10/2607_Obayashi_Avatar_Mockup_3DGS](https://github.com/1-10/2607_Obayashi_Avatar_Mockup_3DGS) の
`Assets/Sandbox/Ooba/GNM`（Unity 側の Viewer）に揃える。

## 動作要件

- Chrome / Edge の最近の版
- WebGPU があれば深度・法線の推論（DAViD）を fp16 で走らせる。無ければ WASM + int8 へ落ちる。
  どちらで動いたかは画面に出る
- 初回に GNM アセット約 39MB と DAViD の ONNX（数百 MB）を読み込む
- Webカメラを使うなら HTTPS か localhost 経由で開く

## セットアップ

```bash
npm install
npm run dev
```

表示されたローカル URL（既定 `http://localhost:5173`）をブラウザで開く。

| コマンド | すること |
|:--|:--|
| `npm run dev` | 開発サーバーを起動する |
| `npm test` | domain / application の検査を走らせる（実アセットを読む） |
| `npm run build` | 型チェックと本番ビルド |
| `npm run preview` | ビルド結果をプレビューする |

### アセットの生成

ブラウザが読む GNM アセット `public/gnm/gnm_head.gnmb` はリポジトリに同梱してある。作り直しが要るのは
**公式 GNM アセットの版を上げるとき**か、**口形プリセット（`tools/viseme_presets.json`）を触ったとき**
だけ。Python と numpy が要る。

```bash
python tools/fetch_gnm_assets.py     # 公式 npz / 68 点定義 / canonical を取得する
python tools/export_gnm_assets.py    # public/gnm/gnm_head.gnmb を生成する
```

DAViD の ONNX はリポジトリに含めず、Hugging Face Hub の公開リポジトリから配信している。ブラウザが
直接読むので、デプロイ先へ上げるものは無い。作り直すときは `python tools/prepare_david_model.py` で
fp16 / int8 を生成し、同じ Hub リポジトリの `david/` へ上げる。

## 何ができるか

画面はツールバー・中央の 3D ビュー・左右のパネルでできている。左パネルが書き出しの値、右パネルが
3D ビューの値を持つ。

- **写真を選ぶ** — ファイルから読むか、Webカメラの映像から取り込む。書き出しが 6 段を走り、
  終わると 3D ビューに頭が出る
- **3D ビュー** — 左ドラッグでカメラ周回、Shift+左で首と視線、右ドラッグで平行移動、ホイールで
  寄り引き。位置と回転は右パネルへ数で出ていて、打ち込んでも動く
- **表情** — プリセットを手で立てるか、自動で切り替える。自動まばたきもここ
- **口形（あいうえお）** — 5 本の口形を手で立てるか、連続再生する
- **Webカメラの表情トラッキング** — 自分の表情を 3D の顔へ写す。「首も動かす」を入れると首も追従する
- **録画と再生** — トラッキング中の表情を収録して JSON へ保存し、読み込んで再生する（再生と読み込みは
  右パネルの「表情アニメーション」）
- **検査画像** — 各段の出力を並べる。結果が合わないときにどの段で崩れたかを見る
- **内訳** — 肌アトラスのテクセルが写真から来たのか補完で埋めたのかを数で出す
- **調整パラメータ** — 既定値・範囲・選べる値は左右のパネルがそのまま一覧を出す

`Esc` で検査画像と内訳のオーバーレイを閉じる。それ以外のキー操作は持たない。

書き出しは成果物（identity 係数・肌アトラス・眼球テクスチャ・髪シェル）を最後まで組み立てる。ただし
zip として落とすボタンは今の画面に無い。詰める側（`src/infrastructure/packaging.ts`）は残してある。

## どう動いているか

写真 1 枚から成果物までを 6 段で通す（正本は `src/application/exportGuest.ts` の `STAGE_NAMES`）。

| 段 | すること |
|:--|:--|
| 1. 推論 | 顔ランドマーク / 髪マスク / 深度・法線・前景を推定する |
| 2. フィット | 密対応を重ねて相似変換と identity 係数を得る |
| 3. 眼球 | 写真の画素を眼球の極座標 UV へ焼く（左右 2 枚） |
| 4. アトラス | 写真を公式 UV アトラスへ焼き、残りを表面沿いの伝播で埋める |
| 5. 髪シェル | 深度と髪マスクからグリッドメッシュを作る |
| 6. 組み立て | 出力契約の値（`GuestArtifacts`）にまとめる |

顔ランドマークが先で、そこから深度推定へ渡す切り出しを決める。各段の出力は検査画像としてそのまま
画面に並ぶ。段ごとの失敗は固有の例外で上がり、段の名前と人向けの対処は例外の型から引く
（`describeFailure`）。

消費側へ渡すのは成果物だけで、頭部のジオメトリは渡さない。頂点は公式 GNM の基底へ `guest.json` の
identity を当てて消費側が作る。

## ディレクトリ構成

デスクトップ側と同じレイヤー構成。依存は内向きだけ。

```text
index.html
src/
  domain/          # 純粋計算（contract / field / photo / normal / ramp / resample /
                   #   faceSubject / faceLadder / inspection / gnm / atlas / eyes / hair）
    preview/       # 3D ビューだけが使う層（asset / camera / pose / expression / viseme /
                   #   faceTracking / recording / normals / scene）
  application/     # ユースケースと Port（exportGuest / ports / settings）
  infrastructure/  # Port の実装（gnmb / gnmAsset / packaging / imaging / jpeg / png /
                   #   photoCanvas / faceLandmarks / faceTracker / segmentation /
                   #   depthNormal / atlasBaker / hairImage / mediapipeVision）
  presentation/    # 入口（main / viewer / gui / input / inspectionView / viewSettings /
                   #   webcamPanel / visemeDriver / recordingPlayer / style.css）
  composition.ts   # 配線（具体実装を組み立てて Port として注入するのはここだけ）
tools/             # アセット生成と突き合わせ基準の作成（Python）
tests/             # domain / application の検査（実アセットを読む。推論は偽の Port）
public/gnm/        # ブラウザが読む GNM アセット
```

## 詳しいことはどこに書いてあるか

**設計判断はコード側に置いてある。** 「なぜこの実装なのか」「やってはいけない形とその理由」
「実測値の根拠」は各ファイル冒頭のコメントが正本なので、そこを読む。

| 知りたいこと | 読む場所 |
|:--|:--|
| 出力契約（UV の向き・色空間・眼球が 2 枚である理由） | `src/domain/contract.ts` |
| 6 段の合成と失敗の伝え方 | `src/application/exportGuest.ts` |
| 書き出しパラメータの既定値と範囲 | `src/application/settings.ts` |
| 3D ビューの値と Unity 側との対応 | `src/presentation/viewSettings.ts` / `src/presentation/viewer.ts` |
| 表情トラッキングの当て方 | `src/domain/preview/faceTracking.ts` |
| アセット生成と量子化 | `tools/export_gnm_assets.py` |

正本との突き合わせは `tools/golden_export_guest.py` がデスクトップ側の `domain` を動かして基準値を
作り、`npm test` がその値と比べる（手順はスクリプトの docstring）。

## 技術スタック

- Google GNM Head（真 3D パラメトリック頭部。identity の線形基底）— Apache-2.0
- MediaPipe Face Landmarker / Image Segmenter（`@mediapipe/tasks-vision`）
- DAViD multi-task（深度・法線・前景を 1 回の推論で。`onnxruntime-web`）
- Three.js（3D ビュー）/ lil-gui（パラメータパネル）/ fflate（zip）
- Vite + TypeScript + Vitest

モデルのライセンスは MediaPipe が Apache-2.0、DAViD がモデル MIT + 学習データ CDLA-Permissive-2.0
（100% 合成データ）、GNM Head が Apache-2.0。GNM の学習データは実写の 3D スキャンで、被写体同意の
明示記載は未確認なので、商用出荷物へ含めるなら確認を推奨する。
