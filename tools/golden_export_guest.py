"""正本の domain を直接動かして、移植の突き合わせ基準（golden）を作る.

**あちらの ``domain`` は numpy 以外を import しない**（`.claude/rules/architecture.md` の規約）ので、
PySide6 / mediapipe / onnxruntime を入れずにそのまま呼べる。同じ合成入力で `export_guest` を通し、
その結果の要約を JSON へ出す。TS 側は `tests/exportGuest.test.ts` が同じ合成入力で同じ要約を作り、
この JSON と突き合わせる。

**これが「移植が同じ計算をしている」ことの唯一の直接証拠**である。定数の一致はテキストで確かめられる
が、アルゴリズムの一致は数値でしか確かめられない。

    git clone https://github.com/1-10/2608_Obayashi_GNMHeadExporter <正本>
    python tools/fetch_gnm_assets.py
    python tools/golden_export_guest.py <正本>/src tests/golden/exportGuest.json

合成入力（写真の大きさ・相似変換・領域の境目・虹彩の置き方）は **TS 側のテストと同じ決め方**にして
ある。片方だけ変えたら突き合わせが無意味になるので、変えるときは両方を一緒に直すこと。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CANONICAL_SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
WEB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CANONICAL_SRC))

from gnm_head_exporter.application.export_guest import GnmModel, export_guest  # noqa: E402
from gnm_head_exporter.application.settings import ExportSettings  # noqa: E402
from gnm_head_exporter.domain.atlas import BakeSettings, bake_atlas  # noqa: E402
from gnm_head_exporter.domain.field import (  # noqa: E402
    DepthNormalResult,
    HairMask,
    PersonSegmentation,
    Rect,
    ScalarField,
)
from gnm_head_exporter.domain.gnm.dense import build_dense_correspondence  # noqa: E402
from gnm_head_exporter.domain.gnm.fit import (  # noqa: E402
    MEDIAPIPE_IBUG68,
    build_dense_landmark_model,
)
from gnm_head_exporter.domain.hair.mask_refine import (  # noqa: E402
    decontaminate_hair_texture,
    refine_hair_mask_with_photo,
)
from gnm_head_exporter.infrastructure.canonical_face import load_canonical_face_obj  # noqa: E402
from gnm_head_exporter.infrastructure.gnm_asset import (  # noqa: E402
    load_gnm_head_npz,
    load_sparse_68_barycentric,
)

WIDTH = 480
HEIGHT = 600


def layout(model: GnmModel):
    """合成写真の相似変換と領域の境目（TS 側のテストと同じ決め方）。"""
    # **float64 で bbox を取る。** float32 のまま引くと、TS 側（JS の数は倍精度）との差が
    # 相似変換のスケールへ float32 eps ぶん乗り、合成入力側の丸めが移植の差に見える。
    positions = model.asset.mesh.template_vertex_positions.astype(np.float64)
    low = positions.min(axis=0)
    high = positions.max(axis=0)
    scale = min(WIDTH * 0.9 / (high[0] - low[0]), HEIGHT * 0.9 / (high[1] - low[1]))
    linear = np.array([[scale, 0.0], [0.0, -scale]])
    translation = np.array(
        [
            WIDTH / 2 - scale * (low[0] + high[0]) / 2,
            HEIGHT / 2 + scale * (low[1] + high[1]) / 2,
        ]
    )
    from gnm_head_exporter.domain.gnm.crop import mean_chin_height
    from gnm_head_exporter.domain.gnm.fit import Similarity2d

    similarity = Similarity2d(linear=linear, translation=translation)
    landmark_model = model.landmark_model()
    chin_row = similarity.apply(np.array([[0.0, mean_chin_height(landmark_model)]]))[0, 1]
    top_row = similarity.apply(np.array([[0.0, high[1]]]))[0, 1]
    return similarity, top_row + (chin_row - top_row) * 0.3, chin_row


def synthetic_landmarks(model: GnmModel, similarity) -> np.ndarray:
    landmark_model = model.landmark_model()
    mean = landmark_model.evaluate(np.zeros(landmark_model.identity_component_count))
    projected = similarity.apply(mean[:, :2])
    landmarks = np.zeros((478, 2))
    landmarks[landmark_model.photo_indices] = projected

    centroids = []
    for name in ("right_eye", "left_eye"):
        index = model.asset.mesh.component_names.index(name)
        mask = np.asarray(model.asset.mesh.component_id) == index
        # float64 で平均する。float32 のままだと合成入力側の丸めが移植との差に見える。
        centre = model.asset.mesh.template_vertex_positions[mask][:, :2].astype(np.float64).mean(axis=0)
        centroids.append(similarity.apply(centre[None, :])[0])
    iris_radius = 0.0059 * similarity.scale
    offsets = np.array(
        [[0.0, 0.0], [iris_radius, 0.0], [-iris_radius, 0.0], [0.0, iris_radius], [0.0, -iris_radius]]
    )
    for group, centre in enumerate(centroids):
        landmarks[468 + group * 5 : 468 + group * 5 + 5] = centre[None, :] + offsets
    return landmarks


def synthetic_photo(hair_bottom_row: float) -> np.ndarray:
    photo = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    rows = np.arange(HEIGHT)[:, None]
    hair = np.broadcast_to(rows < hair_bottom_row, (HEIGHT, WIDTH))
    photo[..., 0] = np.where(hair, 60, 205)
    photo[..., 1] = np.where(hair, 45, 155)
    photo[..., 2] = np.where(hair, 40, 135)
    return photo


class FakeLandmarks:
    def __init__(self, landmarks: np.ndarray) -> None:
        self._landmarks = landmarks

    def detect(self, image_rgb: np.ndarray) -> np.ndarray:
        return self._landmarks


class FakeSegmenter:
    def __init__(self, hair_bottom_row: float, face_bottom_row: float) -> None:
        self._hair = hair_bottom_row
        self._face = face_bottom_row

    def segment(self, image_rgb: np.ndarray) -> PersonSegmentation:
        rows = np.arange(HEIGHT)[:, None]
        hair = np.broadcast_to(rows < self._hair, (HEIGHT, WIDTH)).astype(np.float32)
        face = np.broadcast_to(
            (rows >= self._hair) & (rows < self._face), (HEIGHT, WIDTH)
        ).astype(np.float32)
        body = np.broadcast_to(rows >= self._face, (HEIGHT, WIDTH)).astype(np.float32)
        return PersonSegmentation(
            hair=ScalarField.over_full_image(np.ascontiguousarray(hair)),
            accessory=ScalarField.over_full_image(np.zeros((HEIGHT, WIDTH), dtype=np.float32)),
            face_skin=ScalarField.over_full_image(np.ascontiguousarray(face)),
            body_skin=ScalarField.over_full_image(np.ascontiguousarray(body)),
        )


class FakeDepthNormal:
    def estimate_square(self, image_rgb: np.ndarray, *, x: int, y: int, size: int):
        resolution = 128
        rect = Rect.from_pixels(x, y, size, size, WIDTH, HEIGHT)
        depth = np.tile(
            (1.0 - np.arange(resolution) / (resolution - 1))[:, None], (1, resolution)
        ).astype(np.float32)
        normal = np.zeros((3, resolution, resolution), dtype=np.float32)
        normal[2] = 1.0
        return DepthNormalResult(
            depth=ScalarField(np.ascontiguousarray(depth), rect),
            normal=normal,
            foreground=ScalarField(np.ones((resolution, resolution), dtype=np.float32), rect),
        )


class DomainAtlasBaker:
    """domain の実装をそのまま呼ぶ Port（CUDA を使わない品質基準そのもの）。"""

    def bake(self, **kwargs):
        return bake_atlas(**kwargs)


class DomainHairImageProcessor:
    def refine_mask(self, photo_rgb, mask: HairMask, *, maximum_dimension: int) -> HairMask:
        return refine_hair_mask_with_photo(photo_rgb, mask, maximum_dimension=maximum_dimension)

    def decontaminate_texture(self, photo_rgb, alpha_uint8):
        return decontaminate_hair_texture(photo_rgb, alpha_uint8)


def summary(array: np.ndarray) -> dict:
    values = np.asarray(array, dtype=np.float64).ravel()
    return {
        "count": int(values.size),
        "sum": float(values.sum()),
        "min": float(values.min()) if values.size else 0.0,
        "max": float(values.max()) if values.size else 0.0,
        "mean": float(values.mean()) if values.size else 0.0,
    }


def main() -> None:
    npz = WEB_ROOT / "assets" / "gnm" / "gnm_head.npz"
    sparse = WEB_ROOT / "assets" / "gnm" / "head_sparse_68.txt"
    canonical_path = WEB_ROOT / "assets" / "mediapipe" / "canonical_face_model.obj"
    asset = load_gnm_head_npz(npz)
    landmarks68 = load_sparse_68_barycentric(sparse)
    dense = build_dense_correspondence(
        load_canonical_face_obj(canonical_path), asset.mesh, landmarks68, MEDIAPIPE_IBUG68
    )
    model = GnmModel(asset=asset, landmarks=landmarks68, dense=dense)

    similarity, hair_bottom_row, face_bottom_row = layout(model)
    landmarks = synthetic_landmarks(model, similarity)
    photo = synthetic_photo(hair_bottom_row)

    settings = ExportSettings(skin_atlas_size=512, eye_texture_size=128, hair_texture_size=512)
    outcome = export_guest(
        photo,
        model,
        FakeLandmarks(landmarks),
        FakeSegmenter(hair_bottom_row, face_bottom_row),
        FakeDepthNormal(),
        DomainAtlasBaker(),
        DomainHairImageProcessor(),
        settings,
    )

    landmark_model = build_dense_landmark_model(asset, dense)
    result = {
        "dense": {
            "point_count": int(dense.point_count),
            "edge_meters": float(dense.edge_meters),
            "residual_meters": summary(dense.residual_meters),
            "assumed_disagreement": float(landmark_model.assumed_disagreement),
            "face_width": float(landmark_model.face_width),
        },
        "identity": summary(outcome.artifacts.manifest.identity),
        "identity_head": [float(v) for v in outcome.artifacts.manifest.identity[:8]],
        "skin_albedo": summary(outcome.artifacts.skin_albedo),
        "eye_albedo_left": summary(outcome.artifacts.eye_albedos["left"]),
        "eye_albedo_right": summary(outcome.artifacts.eye_albedos["right"]),
        "eye_left_limbus_px": float(outcome.eye_albedos["left"].limbus_radius_px),
        "eye_left_iris_px": float(outcome.eye_albedos["left"].iris_radius_px),
        "hair": None
        if outcome.artifacts.hair is None
        else {
            "vertex_count": int(outcome.artifacts.hair.vertex_count),
            "triangle_count": int(outcome.artifacts.hair.triangle_count),
            "positions": summary(outcome.artifacts.hair.positions),
            "uvs": summary(outcome.artifacts.hair.uvs),
        },
        "hair_alpha": None
        if outcome.artifacts.hair_alpha is None
        else summary(outcome.artifacts.hair_alpha),
        "hair_albedo": None
        if outcome.artifacts.hair_albedo is None
        else summary(outcome.artifacts.hair_albedo),
        "similarity": {
            "linear": [float(v) for v in np.asarray(similarity.linear).ravel()],
            "translation": [float(v) for v in np.asarray(similarity.translation).ravel()],
        },
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{OUT} を書きました")
    print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
