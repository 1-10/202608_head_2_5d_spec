"""アセット生成の入力（GNM 公式 npz / 68 点定義 / canonical）を取得する.

``tools/export_gnm_assets.py`` がブラウザ用アセットを作るのに要る 3 つを落とす。
**取得先とハッシュは 1-10/2608_Obayashi_GNMHeadExporter の ``tools/fetch_models.py``
と同じ値**（あちらが正本）。片方だけ動かすと同じアセットが作れなくなる。

```powershell
python tools/fetch_gnm_assets.py
```

**ブラウザが実行時に読むモデルはここでは落とさない。** MediaPipe の
FaceLandmarker / SelfieMulticlass は配布元の固定 URL から直接読み、DAViD は
Hugging Face Hub から読む（``src/infrastructure/`` の各アダプタが正本）。デスクトップ側が
ローカルへ落とすのは Python が実行時にファイルを要るからで、**ブラウザは URL のまま
読めるので落とす段が要らない** — web だから消える差分。
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RemoteFile:
    """取得元と、そこから来るはずのバイト列.

    ``url`` と ``sha256`` を 1 つの値にまとめているのは、**URL を書き換えたのに
    ハッシュを直し忘れる**という壊れ方を型で塞ぐため。片方だけ変えることができない。
    """

    url: str
    sha256: str
    destination: Path


# git の raw はコミットハッシュで固定する（``main`` / ``master`` は動く）。
_GNM_COMMIT = "29c8e0163d9c3d0335a22367407f1a9ad85c008e"
_GNM_SPARSE_68_COMMIT = "4fa7372b17efe0e324bd579706b718b248cfdea2"
_GNM_RAW = "https://raw.githubusercontent.com/google/GNM"
_MEDIAPIPE_COMMIT = "a908d668c730da128dfa8d9f6bd25d519d006692"

REMOTE_FILES: tuple[RemoteFile, ...] = (
    RemoteFile(
        f"{_GNM_RAW}/{_GNM_COMMIT}/gnm/shape/data/versions/v3_0/gnm_head.npz",
        "7cf8c9916c199d7cdd8a6609ab8b0a91d1fb98f089437c6d32dd6a3a597c371b",
        REPOSITORY_ROOT / "assets" / "gnm" / "gnm_head.npz",
    ),
    RemoteFile(
        f"{_GNM_RAW}/{_GNM_SPARSE_68_COMMIT}/gnm/shape/data/landmarks/head_sparse_68.txt",
        "8b4b759042cae8b67062794306dae9d60fc7ba11ddad60461ba3e2bfaaeac222",
        REPOSITORY_ROOT / "assets" / "gnm" / "head_sparse_68.txt",
    ),
    RemoteFile(
        f"https://raw.githubusercontent.com/google-ai-edge/mediapipe/{_MEDIAPIPE_COMMIT}"
        "/mediapipe/modules/face_geometry/data/canonical_face_model.obj",
        "8bac80443397e113f41a8b565ea72c59390bc031d9defab289dba7bc0c54e618",
        REPOSITORY_ROOT / "assets" / "mediapipe" / "canonical_face_model.obj",
    ),
)


def sha256_of(path: Path) -> str:
    """ファイルの SHA-256 を 16 進で返す。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(remote: RemoteFile) -> None:
    """``remote`` を落として SHA-256 で照合する.

    一時ファイルへ書いてから rename する。途中で切れたファイルが正しい名前で残ると、
    次回「もうある」と判断して壊れた入力を使い続けることになる。

    **既にあるファイルも照合する。** 存在するだけで skip すると、上流が差し替わった
    あとに入れたマシンと前から在るマシンで中身が違っても誰も気づけない。
    """
    destination = remote.destination
    if destination.is_file():
        actual = sha256_of(destination)
        if actual == remote.sha256:
            print(f"[skip] 既にある: {destination}")
            return
        raise SystemExit(
            f"{destination} のハッシュが合いません\n"
            f"       期待 {remote.sha256}\n"
            f"       実際 {actual}\n"
            f"       取得元 {remote.url}\n"
            "       手で置いた別物かもしれないので消しません。確認して削除してください"
        )

    print(f"[get ] {remote.url}\n       -> {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        with urllib.request.urlopen(remote.url) as response:  # noqa: S310 - URL は定数
            temporary.write_bytes(response.read())
        actual = sha256_of(temporary)
        if actual != remote.sha256:
            raise SystemExit(
                f"落としたファイルのハッシュが合いません: {remote.url}\n"
                f"       期待 {remote.sha256}\n"
                f"       実際 {actual}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    for remote in REMOTE_FILES:
        download(remote)
    print("\n次: python tools/export_gnm_assets.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
