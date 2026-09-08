"""GNM 公式 npz → ブラウザ用アセット（GNMB content="head_asset"）を生成する.

使い方::

    python tools/fetch_gnm_assets.py     # npz / head_sparse_68.txt / canonical を取得
    python tools/export_gnm_assets.py    # public/gnm/gnm_head.gnmb を生成

**この生成物は 1-10/2608_Obayashi_GNMHeadExporter の
``infrastructure/gnm_asset.load_gnm_head_npz`` が npz から作る値と同じもの**を、
ブラウザが読める形（GNMB コンテナ）にしたものである。あちらは Python が npz を直接
読めるので変換を持たない。ブラウザは npz を読めないので、この 1 段だけが web 側に
増える — **web だから必要な差分**であって、内容の差ではない。

正本はあちらの ``infrastructure/gnm_asset.py`` と ``domain/gnm/dense.py``。ここの関数は
その移植で、**判断（何を読むか・領域の作り方・密対応の作り方）を変えない**。変えたら
出力が別物になるので、変えるときはあちらと一緒に変えること。

例外がひとつある: あいうえおの口形（``tools/viseme_presets.json``）
--------------------------------------------------------------------
表情プリセットの正本は Unity 側だが、**その後ろへ足す口形の正本はここ（web 側）**。あちらは
公式クラスぶんしか持たないので、この口形は既知の乖離である。中身は公式クラスの係数行の線形結合
なので、同じ重みを同じクラスへ与えれば Unity でも同じ口が作れる（持っていくものは
``tools/viseme_presets.json`` の重みだけ）。

GNMB のバイト配置（あちらの ``infrastructure.gnm_asset`` が正本）::

    magic  "GNMB"  4 bytes ASCII
    uint32 headerLen                    little endian
    JSON   header  headerLen bytes      UTF-8
    payload                             各配列は 4 byte 境界

identity 基底だけ int16 に量子化する
------------------------------------
**成分は絞らない**（公式の全成分をその並びのまま持つ）。絞ると「どこで切るか」の判断が
残り続けるうえ、公式の並びは寄与の厳密な降順ではない。

量子化するのはブラウザへ送るバイト数のため（float32 で 56MB、int16 で 28MB）。あちらは
量子化しない — ローカルの npz を読むので送る必要がない。フィットへの影響はあちらが
実測しており（残差の 5 桁目・係数の差 7e-4 = 誤差 146nm）、その測定がこちらの根拠に
なる。実際の最大誤差はこのスクリプトが毎回表示する。
"""

from __future__ import annotations

import itertools
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NPZ = REPOSITORY_ROOT / "assets" / "gnm" / "gnm_head.npz"
DEFAULT_SPARSE_68 = REPOSITORY_ROOT / "assets" / "gnm" / "head_sparse_68.txt"
DEFAULT_CANONICAL = REPOSITORY_ROOT / "assets" / "mediapipe" / "canonical_face_model.obj"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "public" / "gnm" / "gnm_head.gnmb"
DEFAULT_EXPRESSION_PRESETS = REPOSITORY_ROOT / "tools" / "GnmExpressionPresets_v3_0.npz"
DEFAULT_VISEME_PRESETS = REPOSITORY_ROOT / "tools" / "viseme_presets.json"

MAGIC = b"GNMB"
GNMB_FORMAT = "GNMB"
GNMB_FORMAT_VERSION = 1
GNMB_CONTENT_HEAD_ASSET = "head_asset"
PAYLOAD_ALIGNMENT = 4
UV_ORIGIN = "bottom-left"

_DTYPE_TO_TOKEN = {
    np.dtype("<f4"): "f32",
    np.dtype("<i2"): "int16",
    np.dtype("<u2"): "u16",
    np.dtype("<u4"): "u32",
    np.dtype("<i4"): "i32",
    np.dtype("u1"): "u8",
}

EXPECTED_MESH_COMPONENT_NAMES: tuple[str, ...] = (
    "skin",
    "left_eye",
    "right_eye",
    "upper_teeth_and_gums",
    "lower_teeth_and_gums",
    "tongue",
)
EXPECTED_SPLIT_VERTEX_COUNT = 18437
IBUG68_POINT_COUNT = 68
MEDIAPIPE_FACE_MESH_COUNT = 468
SPARSE_68_WEIGHT_SUM_TOLERANCE = 5e-3

MOUTH_RIM_APERTURE_RING_WEIGHTS = (1.0, 1.0, 0.5)
EYE_COMPONENT_NAMES = ("left_eye", "right_eye")
NEIGHBOURHOOD_EDGES = 8.0

MEDIAPIPE_IBUG68: tuple[int, ...] = (
    # 顎ライン 0-16。index 2〜6 が標準の iBUG 順の逆なのは誤りではない
    # （GNM の head_sparse_68 が向かって左の顎を顎寄り → 耳寄りで定義している）。
    162, 234, 149, 136, 172, 58, 93, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
    # 眉 17-21（左）/ 22-26（右）
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    # 鼻梁 27-30 + 鼻底 31-35
    168, 197, 5, 4, 75, 97, 2, 326, 305,
    # 目 36-41（左）/ 42-47（右）
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
    # 口 外周 48-59 + 内周 60-67
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317, 14, 87,
)

IBUG68_CHAINS: tuple[tuple[str, tuple[int, ...], bool], ...] = (
    ("顎", tuple(range(0, 17)), False),
    ("眉左", tuple(range(17, 22)), False),
    ("眉右", tuple(range(22, 27)), False),
    ("鼻梁", tuple(range(27, 31)), False),
    ("鼻底", tuple(range(31, 36)), False),
    ("目左", tuple(range(36, 42)), True),
    ("目右", tuple(range(42, 48)), True),
    ("唇外周", tuple(range(48, 60)), True),
    ("唇内周", tuple(range(60, 68)), True),
)
# --- 3D ビューだけが使う値 ---------------------------------------------------
# **書き出しの契約には入らない。** guest が Unity（1-10/2607_Obayashi_Avatar_Mockup_3DGS の
# ``Assets/Sandbox/Ooba/GNM``）でどう出るかを web でも同じ形で見るために持つ。
# 正本は Unity 側の ``Editor/GnmHeadAssetBuilder`` と ``Scripts/GnmHeadInstance``。

GROUP_THRESHOLD = 1e-4
"""vertex group の重みをブール化する閾値（正本は Unity の ``GnmHeadAssetBuilder.GroupThreshold``）。"""

EXPECTED_JOINT_NAMES: tuple[str, ...] = ("neck", "head", "left_eye", "right_eye")

SKIN_INFLUENCE_LIMIT = 2
"""1 頂点が受けるボーンの本数。v3_0 / head では実測で 2 本。超えたら落とす。"""

EYE_EXPRESSION_COMPONENT = re.compile(r"^(left|right)_eye")
"""まばたきに使う成分の名前（正本は旧 web 版 ``main.buildBlinkVector``）。"""

BLINK_PRESET_CLASSES = ("wink_left", "wink_right")
"""まばたき = 公式 ExpressionSampler の WINK_LEFT + WINK_RIGHT を目領域の成分だけ残したもの。"""

EYE_EXPRESSION_GROUPS = ("expression_basis_left_eye", "expression_basis_right_eye")
"""目領域の成分が動かす頂点。実測でこの外側の変位は厳密に 0（クロスフェードをこの範囲へ閉じられる）。"""

VISEME_PRESET_PREFIX = "viseme_"
"""口形プリセットの名前の頭。**GUI と 3D ビューはこの頭で「表情」と「口形」を分ける** — 一覧を
コードへ手で写すと、ここが増減したとき黙って古くなる。"""

EXPRESSION_REGIONS: tuple[str, ...] = (
    "left_eye_region",
    "right_eye_region",
    "lower_face_region",
    "tongue",
    "pupils",
)
"""``expression_names`` に現れる領域。**この並びがそのまま成分の並び**（領域ごとに連続、実測）。

**領域は「動かす頂点が重ならない」という意味では独立していない。** 左目と右目、目と下顔面、
下顔面と舌は頂点を共有し、基底として見ても直交しない（重なりの実測はこのスクリプトが毎回表示する）。

だから**係数は領域ごとに分けて解かず、383 成分をまとめて解く**。領域を分けて持つのは容量のため
だけ — 密に持つと 41MB、領域ブロックなら 1/4 で済む。"""

VISEME_WEIGHT_SUM_LIMIT = 1.0
"""1 つの口形に混ぜるクラス係数の重みの合計の上限。

口形は公式クラスの**線形結合**なので、重みをいくらでも積めば顔は壊れる。「口まわりに効く少数の
クラスを合計 1.0 以下で混ぜたもの」という設計そのものを、生成時に機械で守る。"""

DEGENERATE_EDGE_RATIO = 0.25
REVERSAL_COSINE = -0.5
MIN_REVERSED_EDGES = 2

NPZ_KEYS: dict[str, str] = {
    "version": "読む",
    "variant": "読む",
    "template_vertex_positions": "読む",
    "vertex_identity_basis": "読む",
    "triangles": "読む",
    "triangle_uvs": "読む",
    "vertex_groups": "読む",
    "vertex_group_names": "読む",
    "mesh_component_names": "読む",
    "expression_basis": (
        "読む: 3D ビューの表情プリセットへ焼き、領域ブロックとしてそのまま載せる"
        "（書き出しには入らない）"
    ),
    "skinning_weights": "読む: 3D ビューで首と視線を回す（同上）",
    "joint_names": "読む: 同上",
    "joint_parent_indices": "読む: 同上",
    "joint_identity_basis": "読む: 同上",
    "template_joint_positions": "読む: 同上",
    "pose_correctives_regressor": "読む: 全要素ゼロであることの確認にだけ使う",
    "bone_aligned_template_joint_orientations": "読む: 単位行列であることの確認にだけ使う",
    "expression_names": "読む: まばたきに使う目領域の成分を名前で選び、領域ブロックへ分ける",
    "identity_names": "読まない: 係数は index で送る",
    "joint_regressor": "読まない: 位置は template_joint_positions + joint_identity_basis で作る",
    "mirror_indices": "読まない: 左右対称で色を複製する段を持たない",
    "quads": "読まない: 三角形の側を使う",
    "quad_uvs": "読まない: 同上",
}
"""公式 npz の全キーと、読む / 読まない理由（正本はあちらの ``NPZ_KEYS``）。

キーが増減したらこのスクリプトが落ちる。落ちたら扱いを決めてから足すこと。
"""


# ---------------------------------------------------------------------------
# npz → split 空間のメッシュ
# ---------------------------------------------------------------------------
def names_of(array: np.ndarray) -> list[str]:
    return [str(name) for name in array.tolist()]


def split_by_face_varying_uv(
    triangles: np.ndarray, triangle_uvs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """face-varying UV を per-vertex UV にするため、UV の切れ目で頂点を複製する.

    ``(頂点 index, u, v)`` の辞書順ソートなので ``uv_split_source`` は単調非減少になる。
    **この単調性は契約** — TS 側の ``splitIndicesOf`` が二分探索で引く。
    """
    corner_vertices = triangles.reshape(-1).astype(np.int32)
    corner_uvs = triangle_uvs.reshape(-1, 2).astype(np.float32)

    key = np.empty(
        corner_vertices.shape[0],
        dtype=[("vertex", np.int32), ("u", np.float32), ("v", np.float32)],
    )
    key["vertex"] = corner_vertices
    key["u"] = corner_uvs[:, 0]
    key["v"] = corner_uvs[:, 1]

    unique_key, inverse = np.unique(key, return_inverse=True)
    return (
        inverse.reshape(triangles.shape).astype(np.uint32),
        unique_key["vertex"].astype(np.uint32),
        np.stack([unique_key["u"], unique_key["v"]], axis=1).astype(np.float32),
    )


def component_ids(
    vertex_groups: np.ndarray, vertex_group_names: list[str], mesh_component_names: list[str]
) -> np.ndarray:
    """頂点グループからメッシュ構成要素の id（split 前の index 空間）を作る。"""
    if tuple(mesh_component_names) != EXPECTED_MESH_COMPONENT_NAMES:
        raise SystemExit(
            f"mesh_component_names が {tuple(mesh_component_names)}"
            f"（期待 {EXPECTED_MESH_COMPONENT_NAMES}）。component_id の意味が変わる"
        )
    membership = np.stack(
        [vertex_groups[vertex_group_names.index(name)] > 0 for name in mesh_component_names]
    )
    group_count = membership.sum(axis=0)
    if (group_count == 0).any():
        raise SystemExit(f"どの構成要素にも属さない頂点が {int((group_count == 0).sum())} 個ある")
    if (group_count > 1).any():
        raise SystemExit(f"複数の構成要素に属する頂点が {int((group_count > 1).sum())} 個ある")
    return np.argmax(membership, axis=0).astype(np.uint8)


def photo_only_atlas_region(npz: Any, vertex_group_names: list[str]) -> np.ndarray:
    """首・胴体を「写真100%か補完100%の二択」にする領域（split 前）。"""
    positions = np.asarray(npz["template_vertex_positions"])
    groups = np.asarray(npz["vertex_groups"])
    chin = groups[vertex_group_names.index("chin_region")] > 0
    ears = groups[vertex_group_names.index("ears")] > 0
    if not chin.any() or not ears.any():
        raise SystemExit("公式GNMの chin_region または ears が空")
    chin_bottom = float(positions[chin, 1].min())
    chin_half_width = float(np.abs(positions[chin, 0]).max())
    ear_bottom = float(positions[ears, 1].min())
    face_region_names = [name for name in vertex_group_names if name.endswith("_region")]
    face_region = np.any(
        np.stack([groups[vertex_group_names.index(name)] > 0 for name in face_region_names]),
        axis=0,
    )
    lower_neck_and_torso = positions[:, 1] < chin_bottom
    side_neck = (
        (positions[:, 1] < ear_bottom)
        & (np.abs(positions[:, 0]) > chin_half_width)
        & ~face_region
    )
    return np.asarray(lower_neck_and_torso | side_neck, dtype=bool)


def mouth_aperture_ring_weight(
    triangles: np.ndarray, lips: np.ndarray, sock: np.ndarray
) -> np.ndarray:
    """口腔壁との接続面から唇の中を位相 BFS し、リングごとの重みを立てる。"""
    neighbours: dict[int, set[int]] = {}
    seeds: set[int] = set()
    for triangle in triangles:
        members = [int(value) for value in triangle]
        if sock[members].any():
            seeds.update(vertex for vertex in members if lips[vertex])
        for index in range(3):
            first, second = members[index], members[(index + 1) % 3]
            if lips[first] and lips[second]:
                neighbours.setdefault(first, set()).add(second)
                neighbours.setdefault(second, set()).add(first)

    weight = np.zeros(lips.shape[0], dtype=np.float64)
    visited = set(seeds)
    frontier = list(seeds)
    for level_weight in MOUTH_RIM_APERTURE_RING_WEIGHTS:
        if not frontier:
            break
        weight[frontier] = np.maximum(weight[frontier], level_weight)
        nearer: list[int] = []
        for vertex in frontier:
            for neighbour in neighbours.get(vertex, ()):
                if neighbour not in visited:
                    visited.add(neighbour)
                    nearer.append(neighbour)
        frontier = nearer
    return weight


def mouth_rim_region(npz: Any, vertex_group_names: list[str]) -> np.ndarray:
    """開口部の縁（唇のインナーロール）の重み 0..1（split 前）。"""
    groups = np.asarray(npz["vertex_groups"])
    triangles = np.asarray(npz["triangles"]).reshape(-1, 3)
    missing = [
        name for name in ("upper_lip", "lower_lip", "mouth_sock") if name not in vertex_group_names
    ]
    if missing:
        raise SystemExit(f"公式GNMの口まわりのグループが無い: {missing}")
    lips = np.any(
        np.stack(
            [groups[vertex_group_names.index(name)] > 0 for name in ("upper_lip", "lower_lip")]
        ),
        axis=0,
    )
    sock = groups[vertex_group_names.index("mouth_sock")] > 0
    if not lips.any() or not sock.any():
        raise SystemExit("公式GNMの唇グループまたは mouth_sock が空")
    rim = mouth_aperture_ring_weight(triangles, lips, sock).astype(np.float32)
    if not rim.any():
        raise SystemExit("開口部の縁が空（唇と mouth_sock が接していない）")
    return rim


def load_sparse_68(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``head_sparse_68.txt`` を読んで (頂点 index (68,3), 重み (68,3)) を返す。"""
    rows = np.loadtxt(path, dtype=np.float64)
    if rows.shape != (IBUG68_POINT_COUNT, 6):
        raise SystemExit(f"head_sparse_68.txt の形が {rows.shape}（期待 (68, 6)）")
    vertex_indices = rows[:, 0::2]
    weights = rows[:, 1::2]
    if not np.array_equal(vertex_indices, np.rint(vertex_indices)):
        raise SystemExit("頂点 index の列に整数でない値がある")
    row_sums = weights.sum(axis=1)
    worst = int(np.argmax(np.abs(row_sums - 1.0)))
    if abs(row_sums[worst] - 1.0) > SPARSE_68_WEIGHT_SUM_TOLERANCE:
        raise SystemExit(f"barycentric の行和が 1.0 から離れすぎている: 行 {worst}")
    return vertex_indices.astype(np.int32), (weights / row_sums[:, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# 密対応（あちらの domain/gnm/dense.py の移植）
# ---------------------------------------------------------------------------
def load_canonical_obj(path: Path) -> np.ndarray:
    """MediaPipe canonical_face_model.obj（頂点 i = landmark i）を読む。"""
    vertices: list[list[float]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
    points = np.asarray(vertices, dtype=np.float64)
    if points.shape[0] != MEDIAPIPE_FACE_MESH_COUNT:
        raise SystemExit(f"canonical の頂点数が {points.shape[0]}（期待 468）")
    return points


def solve_similarity_3d(
    source: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """3D 相似変換（Umeyama 1991）。鏡映は許さない（別人の顔になる）。"""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    covariance = centered_target.T @ centered_source / len(source)
    left, singular, right = np.linalg.svd(covariance)
    reflection = np.diag([1.0, 1.0, np.sign(np.linalg.det(left @ right))])
    rotation = left @ reflection @ right
    scale = float(
        np.trace(np.diag(singular) @ reflection) / (centered_source**2).sum(axis=1).mean()
    )
    return scale, rotation, target_mean - scale * rotation @ source_mean


def thin_plate_warp(
    control_source: np.ndarray,
    control_target: np.ndarray,
    points: np.ndarray,
    regularization: float = 1e-6,
) -> np.ndarray:
    """3D thin-plate spline（カーネル ``U(r) = r``）。制御点は厳密に一致する。"""
    count = len(control_source)

    def kernel(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)

    affine_basis = np.hstack([np.ones((count, 1)), control_source])
    system = np.zeros((count + 4, count + 4))
    system[:count, :count] = (
        kernel(control_source, control_source) + np.eye(count) * regularization
    )
    system[:count, count:] = affine_basis
    system[count:, :count] = affine_basis.T
    right_hand = np.zeros((count + 4, 3))
    right_hand[:count] = control_target
    solution = np.linalg.solve(system, right_hand)
    bending, affine = solution[:count], solution[count:]
    return kernel(points, control_source) @ bending + np.hstack(
        [np.ones((len(points), 1)), points]
    ) @ affine


def project_to_triangles(
    points: np.ndarray, triangle_vertices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """各点を三角形群の最近点へ落とす（三角形 index / barycentric / 距離）。"""
    query = np.asarray(points, dtype=np.float64)
    corner_a = triangle_vertices[:, 0]
    edge_ab = triangle_vertices[:, 1] - corner_a
    edge_ac = triangle_vertices[:, 2] - corner_a

    dot_aa = np.einsum("ij,ij->i", edge_ab, edge_ab)
    dot_ab = np.einsum("ij,ij->i", edge_ab, edge_ac)
    dot_bb = np.einsum("ij,ij->i", edge_ac, edge_ac)
    denominator = np.maximum(1e-12, dot_aa * dot_bb - dot_ab * dot_ab)

    offset = query[:, None, :] - corner_a[None, :, :]
    dot_pa = np.einsum("pij,ij->pi", offset, edge_ab)
    dot_pb = np.einsum("pij,ij->pi", offset, edge_ac)

    v = np.clip((dot_bb * dot_pa - dot_ab * dot_pb) / denominator, 0.0, 1.0)
    w = np.clip((dot_aa * dot_pb - dot_ab * dot_pa) / denominator, 0.0, 1.0)
    overflow = np.maximum(1.0, v + w)
    v /= overflow
    w /= overflow

    closest = corner_a[None] + edge_ab[None] * v[..., None] + edge_ac[None] * w[..., None]
    distances = np.linalg.norm(closest - query[:, None, :], axis=2)
    chosen = np.argmin(distances, axis=1)
    rows = np.arange(len(query))
    picked_v = v[rows, chosen]
    picked_w = w[rows, chosen]
    return (
        chosen,
        np.stack([1.0 - picked_v - picked_w, picked_v, picked_w], axis=1),
        distances[rows, chosen],
    )


def assert_landmark_chain_orientation(model_xyz: np.ndarray, other_xyz: np.ndarray) -> None:
    """68 点の対応が交差していないことを確かめる（あちらの domain/gnm/fit.py の移植）。"""
    violations: list[str] = []
    for name, chain, is_ring in IBUG68_CHAINS:
        heads = np.asarray(chain)
        tails = np.roll(heads, -1) if is_ring else heads[1:]
        heads = heads if is_ring else heads[:-1]

        model_edges = model_xyz[tails] - model_xyz[heads]
        other_edges = other_xyz[tails] - other_xyz[heads]
        model_lengths = np.linalg.norm(model_edges, axis=1)
        other_lengths = np.linalg.norm(other_edges, axis=1)
        usable = (model_lengths > DEGENERATE_EDGE_RATIO * np.median(model_lengths)) & (
            other_lengths > DEGENERATE_EDGE_RATIO * np.median(other_lengths)
        )
        cosines = np.einsum("ij,ij->i", model_edges, other_edges) / np.where(
            usable, model_lengths * other_lengths, 1.0
        )
        reversed_edges = np.flatnonzero(usable & (cosines < REVERSAL_COSINE))
        if reversed_edges.size < MIN_REVERSED_EDGES:
            continue
        violations += [
            f"{name}: 点 {heads[edge]} → {tails[edge]} の差分が逆を向いている"
            f"（cos {cosines[edge]:+.3f}）"
            for edge in reversed_edges
        ]
    if violations:
        raise SystemExit(
            "68 点の対応が交差している:\n" + "\n".join(f"  - {line}" for line in violations)
        )


def build_dense_correspondence(
    canonical: np.ndarray,
    template: np.ndarray,
    triangles: np.ndarray,
    uv_split_source: np.ndarray,
    component_id: np.ndarray,
    component_names: tuple[str, ...],
    sparse68_indices: np.ndarray,
    sparse68_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """canonical の 468 点を GNM 平均顔の表面へ写して barycentric 対応を作る.

    戻り値:
        ``(mediapipe_indices, vertex_indices（split 前）, weights, residual_meters,
        edge_meters)``
    """
    split_source = np.asarray(uv_split_source, dtype=np.int64)
    resolved = np.searchsorted(split_source, sparse68_indices.astype(np.int64), side="left")
    if not np.array_equal(split_source[resolved], sparse68_indices):
        raise SystemExit("uv_split_source から split 前 index を引き直せなかった")
    gnm68 = np.einsum("ijk,ij->ik", template[resolved], sparse68_weights.astype(np.float64))

    picks = np.asarray(MEDIAPIPE_IBUG68, dtype=np.int64)
    scale, rotation, translation = solve_similarity_3d(canonical[picks], gnm68)
    aligned = canonical @ (scale * rotation).T + translation
    # 交差していると次の TPS が「正しく」補間して面が折り返る。例外にならないので止める。
    assert_landmark_chain_orientation(gnm68, aligned[picks])
    warped = thin_plate_warp(aligned[picks], gnm68, aligned)

    eye_components = [
        index for index, name in enumerate(component_names) if name in EYE_COMPONENT_NAMES
    ]
    if not eye_components:
        raise SystemExit(f"眼球の構成要素が見つからない: {component_names}")
    excluded = np.isin(component_id, eye_components)
    usable = ~excluded[triangles].any(axis=1)
    target_triangles = np.asarray(triangles, dtype=np.int64)[usable]
    if target_triangles.size == 0:
        raise SystemExit("投影先の三角形が無い（除外が広すぎる）")
    corners = template[target_triangles]
    centroids = corners.mean(axis=1)

    # 長さの基準はメッシュ自身の辺。アセットの解像度が変わっても意味が変わらない。
    edge = float(
        np.median(
            np.concatenate(
                [
                    np.linalg.norm(corners[:, (corner + 1) % 3] - corners[:, corner], axis=1)
                    for corner in range(3)
                ]
            )
        )
    )
    neighbourhood = NEIGHBOURHOOD_EDGES * edge

    mediapipe_indices: list[int] = []
    vertex_indices: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    residuals: list[float] = []
    for index in range(MEDIAPIPE_FACE_MESH_COUNT):
        near = np.flatnonzero(np.linalg.norm(centroids - warped[index], axis=1) < neighbourhood)
        if near.size == 0:
            continue
        chosen, barycentric, distance = project_to_triangles(
            warped[index : index + 1], corners[near]
        )
        mediapipe_indices.append(index)
        vertex_indices.append(target_triangles[near[chosen[0]]])
        weights.append(barycentric[0])
        residuals.append(float(distance[0]))

    if not mediapipe_indices:
        raise SystemExit("対応が 1 点も付かなかった（整列が壊れている）")
    return (
        np.asarray(mediapipe_indices, dtype=np.uint16),
        # split 空間の index を split 前へ戻す（TS 側が split 前で受ける）。
        split_source[np.asarray(vertex_indices, dtype=np.int64)].astype(np.int32),
        np.asarray(weights, dtype=np.float32),
        np.asarray(residuals, dtype=np.float32),
        edge,
    )


# ---------------------------------------------------------------------------
# GNMB コンテナ
# ---------------------------------------------------------------------------
def build_gnmb_container_bytes(
    content: str, arrays: dict[str, np.ndarray], metadata: dict[str, Any]
) -> bytes:
    """GNMB bin の全バイトを返す（``arrays`` の並びがそのまま payload の並び）。"""
    entries: dict[str, Any] = {}
    chunks: list[bytes] = []
    offset = 0
    for name, array in arrays.items():
        token = _DTYPE_TO_TOKEN.get(array.dtype)
        if token is None:
            raise SystemExit(f"{name} の dtype {array.dtype} は GNMB で表せない")
        padding = -offset % PAYLOAD_ALIGNMENT
        if padding:
            chunks.append(b"\0" * padding)
            offset += padding
        data = np.ascontiguousarray(array).tobytes()
        entries[name] = {
            "offset": offset,
            "byteLength": len(data),
            "dtype": token,
            "shape": list(array.shape),
        }
        chunks.append(data)
        offset += len(data)

    header = {
        "format": GNMB_FORMAT,
        "version": GNMB_FORMAT_VERSION,
        "content": content,
        "uv_origin": UV_ORIGIN,
        **metadata,
        "arrays": entries,
    }
    header_bytes = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return b"".join([MAGIC, struct.pack("<I", len(header_bytes)), header_bytes, *chunks])


def build_viseme_coefficients(
    coefficients: np.ndarray,
    preset_names: list[str],
    visemes_path: Path,
) -> tuple[np.ndarray, list[str]]:
    """あいうえおの口形を、公式 20 クラスの**係数行の線形結合**として作る。

    **383 成分を手で触らない。** 成分名は領域ごとの統計方向であって表情としての意味を持たない
    （警告の正本は ``src/domain/preview/expression.ts`` の冒頭）。合成は必ずクラス係数行の側で行う。

    **実行時に 2 本重ねるのと数値的に同一なのに、それでも焼く理由**: 頂点変位は係数について線形
    なので絵は変わらない。変わるのは「口形が 1 本のプリセットになる」こと — 3D ビューの
    **同時に立てるのは 1 本だけ**という原則を壊さずに口形を足せる。実行時合成にすると、口形を出す
    ためだけに「複数本を重ねてよい経路」を viewer へ作ることになり、原則が例外だらけになる。

    定義の置き場所を ``tools/viseme_presets.json``（テキスト）にしてあるのは、この 5 本の重みが
    **web と Unity の 2 つのツールチェーンで必要になる値**だから。Python のリテラルに書くと Unity
    側が読めず、書き写した瞬間に写しが増える。npz にするとレビューの差分で数値が見えない。
    """
    # ``utf-8-sig`` で読む。Windows のエディタは BOM 付きで保存することがあり、素の ``utf-8`` だと
    # 「JSON が壊れている」という分かりにくい例外で落ちる（BOM が無ければ何も変わらない）。
    with visemes_path.open(encoding="utf-8-sig") as handle:
        document = json.load(handle)
    blends = document.get("presets")
    if not isinstance(blends, dict) or not blends:
        raise SystemExit(f"{visemes_path} の presets が空か dict でない")

    rows: list[np.ndarray] = []
    names: list[str] = []
    for name, blend in blends.items():
        if not name.startswith(VISEME_PRESET_PREFIX):
            raise SystemExit(
                f"口形 '{name}' の名前が '{VISEME_PRESET_PREFIX}' で始まっていない。"
                "GUI と 3D ビューは名前で表情と口形を分けるので、頭を変えるとどちらにも出なくなる"
            )
        if name in preset_names:
            raise SystemExit(f"口形 '{name}' と同じ名前の表情プリセットが既にある")
        if not isinstance(blend, dict) or not blend:
            raise SystemExit(f"口形 '{name}' の結合が空か dict でない")
        row = np.zeros(coefficients.shape[1], dtype=np.float64)
        total = 0.0
        for class_name, weight in blend.items():
            if class_name not in preset_names:
                raise SystemExit(
                    f"口形 '{name}' が混ぜる '{class_name}' が表情プリセット npz に無い"
                    f"（あるのは: {', '.join(preset_names)}）"
                )
            value = float(weight)
            if not 0.0 < value <= 1.0:
                raise SystemExit(f"口形 '{name}' の '{class_name}' の重み {value} が 0〜1 の外")
            row += coefficients[preset_names.index(class_name)] * value
            total += value
        if total > VISEME_WEIGHT_SUM_LIMIT + 1e-9:
            raise SystemExit(
                f"口形 '{name}' の重みの合計が {total:.3f}"
                f"（上限 {VISEME_WEIGHT_SUM_LIMIT}）。混ぜすぎると顔が壊れる"
            )
        rows.append(row)
        names.append(name)
    return np.array(rows, dtype=np.float64), names


def build_expression_basis_blocks(
    expression_basis: np.ndarray,
    source: np.ndarray,
    expression_names: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, float]]:
    """公式の表情基底 383 成分を**領域ブロック**の int16 として詰める.

    密に持つと 383 × 18,437 × 3 × 2B = 41MB になる。**成分は自分の領域の頂点しか動かさない**
    ので、領域ごとに「動く頂点の一覧」と「その頂点だけのブロック」に分けると 1/5 で済む。

    プリセット（``expressionPresetBasisQ``）とは役割が違う。あちらは係数を当て終わった変位で、
    こちらは**係数を解くための基底そのもの**。MediaPipe の点から係数を解く経路（表情フィット）が
    これを読む。プリセットを基底から実行時に作り直すためのものではない。

    量子化のスケールは**成分ごと**。領域ごとにすると弱い成分が 1 目盛りに埋もれる。
    """
    seen: list[str] = []
    for name in expression_names:
        prefix = name.rsplit("_", 1)[0]
        if prefix not in seen:
            seen.append(prefix)
    if seen != list(EXPRESSION_REGIONS):
        raise SystemExit(f"表情基底の領域が想定と違う: {seen}（期待 {list(EXPRESSION_REGIONS)}）")

    vertex_blocks: list[np.ndarray] = []
    quantized_blocks: list[np.ndarray] = []
    scales = np.zeros(len(expression_names), dtype=np.float64)
    regions: list[dict[str, Any]] = []
    vertex_offset = 0
    quantized_offset = 0
    worst_error = 0.0
    supports: dict[str, np.ndarray] = {}
    blocks: dict[str, np.ndarray] = {}
    for region in EXPRESSION_REGIONS:
        columns = [
            index for index, name in enumerate(expression_names) if name.rsplit("_", 1)[0] == region
        ]
        if columns != list(range(columns[0], columns[-1] + 1)):
            raise SystemExit(f"領域 {region} の成分が連続していない（{columns[0]}〜{columns[-1]}）")
        block = expression_basis[columns[0] : columns[-1] + 1][:, source].astype(np.float64)
        blocks[region] = block

        region_scales = np.abs(block).max(axis=(1, 2))
        region_scales[region_scales == 0.0] = 1.0
        # **1 目盛り未満の頂点は落とす。** int16 にすると 0 になる = 送っても何も動かない。
        step = region_scales[:, None] / 32767.0
        moves = (np.abs(block).max(axis=2) > step).any(axis=0)
        vertices = np.flatnonzero(moves).astype(np.int32)
        if vertices.size == 0:
            raise SystemExit(f"領域 {region} が動かす頂点が 1 つも無い")
        supports[region] = moves

        picked = block[:, vertices]
        quantized = np.rint(picked / region_scales[:, None, None] * 32767.0).astype(np.int16)
        worst_error = max(
            worst_error,
            float(
                np.abs(
                    quantized.astype(np.float64) * region_scales[:, None, None] / 32767.0 - picked
                ).max()
            ),
        )
        # 落とした頂点は必ず 1 目盛り未満であること。ここが破れると「動くはずの頂点を捨てた」。
        dropped = block[:, ~moves]
        if dropped.size:
            over = float((np.abs(dropped) / region_scales[:, None, None] * 32767.0).max())
            if over > 1.0:
                raise SystemExit(
                    f"領域 {region} で落とした頂点に {over:.2f} 目盛りの変位が残っている"
                )

        scales[columns[0] : columns[-1] + 1] = region_scales
        vertex_blocks.append(vertices)
        quantized_blocks.append(np.ascontiguousarray(quantized).reshape(-1))
        regions.append(
            {
                "name": region,
                "component_offset": columns[0],
                "component_count": len(columns),
                "vertex_offset": vertex_offset,
                "vertex_count": int(vertices.size),
                # ブロックは領域ごとに大きさが違うので、平坦化した先の頭を明示して持つ
                # （頂点数 × 成分数 から計算し直させると、どちらかが変わったとき黙ってずれる）。
                "quantized_offset": quantized_offset,
            }
        )
        vertex_offset += int(vertices.size)
        quantized_offset += int(quantized.size)

    # 領域の重なりを毎回測る。**分けて解けるかどうかがここで決まる**ので、数字を残さず表示する。
    shared_vertices = 0
    worst_cosine = 0.0
    worst_pair = ""
    for first, second in itertools.combinations(EXPRESSION_REGIONS, 2):
        shared = int((supports[first] & supports[second]).sum())
        shared_vertices += shared
        left = blocks[first].reshape(blocks[first].shape[0], -1)
        right = blocks[second].reshape(blocks[second].shape[0], -1)
        left = left / np.linalg.norm(left, axis=1, keepdims=True)
        right = right / np.linalg.norm(right, axis=1, keepdims=True)
        cosine = float(np.abs(left @ right.T).max())
        if cosine > worst_cosine:
            worst_cosine, worst_pair = cosine, f"{first} ↔ {second}"

    arrays = {
        "expressionBasisVertices": np.concatenate(vertex_blocks),
        "expressionBasisQ": np.concatenate(quantized_blocks),
    }
    metadata = {
        "expression_component_names": list(expression_names),
        "expression_basis_scales": [float(value) for value in scales],
        "expression_basis_regions": regions,
    }
    report = {
        "expression_basis_vertices": float(vertex_offset),
        "expression_basis_bytes": float(arrays["expressionBasisQ"].nbytes),
        "expression_basis_error": worst_error,
        "expression_region_shared_vertices": float(shared_vertices),
        "expression_region_cosine": worst_cosine,
    }
    metadata["expression_region_overlap"] = (
        f"領域の支持は重なる（延べ {shared_vertices} 頂点）。基底も直交せず、"
        f"正規化内積の最大は {worst_cosine:.3f}（{worst_pair}）。"
        "だから係数は領域ごとに分けず全成分をまとめて解く"
    )
    return arrays, metadata, report


def build_preview_arrays(
    npz: Any,
    source: np.ndarray,
    vertex_group_names: list[str],
    presets_path: Path,
    visemes_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, float]]:
    """3D ビューが使う領域・姿勢・表情の配列を作る。

    **書き出しの契約には 1 つも入らない。** ``export_guest`` はここで作る配列を読まない。
    guest が Unity でどう組み立てられるかを web でも同じ形で確認するために持つ。

    Unity 側が v3_0 / head の実データで確認した前提を毎回検査して、崩れたら落とす:
    bind pose に回転が無い / ``pose_correctives_regressor`` が全ゼロ / 重み和がちょうど 1 /
    影響ボーンが 2 本以下。近似で通すと「Unity と同じ絵」を名乗れなくなる。
    """
    joint_names = names_of(npz["joint_names"])
    if tuple(joint_names) != EXPECTED_JOINT_NAMES:
        raise SystemExit(
            f"ジョイント構成が変わっている: {joint_names}（期待 {list(EXPECTED_JOINT_NAMES)}）"
        )
    if not np.allclose(npz["bone_aligned_template_joint_orientations"], np.eye(3), atol=1e-6):
        raise SystemExit(
            "bind pose に回転がある。bindpose を平行移動の逆だけでは表せないので、"
            "回転を持つ bind pose を実装してから進めること"
        )
    if not np.allclose(npz["pose_correctives_regressor"], 0.0):
        raise SystemExit(
            "pose_correctives_regressor が非ゼロ。LBS だけでは公式の出力と一致しないので、"
            "補正を実装してから進めること"
        )

    weights = npz["skinning_weights"]
    sums = weights.sum(axis=0)
    if not np.allclose(sums, 1.0, atol=1e-5):
        raise SystemExit(
            f"skinning_weights の重み和が 1 でない（{float(sums.min())}〜{float(sums.max())}）"
        )
    influences = int((weights > 0.0).sum(axis=0).max())
    if influences > SKIN_INFLUENCE_LIMIT:
        raise SystemExit(f"1 頂点の影響ボーンが {influences} 本（上限 {SKIN_INFLUENCE_LIMIT}）")

    order = np.argsort(-weights, axis=0)[:SKIN_INFLUENCE_LIMIT]
    skin_indices = np.ascontiguousarray(order.T[source].astype(np.uint8))
    picked = np.take_along_axis(weights, order, axis=0).T[source].astype(np.float32)
    skin_weights = np.ascontiguousarray(picked / picked.sum(axis=1, keepdims=True))

    groups = np.ascontiguousarray(
        (npz["vertex_groups"] > GROUP_THRESHOLD)[:, source].astype(np.uint8)
    )

    with np.load(presets_path, allow_pickle=False) as preset_npz:
        preset_keys = set(preset_npz.files)
        if preset_keys != {"expression_presets", "class_names"}:
            raise SystemExit(f"表情プリセット npz のキーが想定と違う: {sorted(preset_keys)}")
        coefficients = preset_npz["expression_presets"].astype(np.float64)
        preset_names = names_of(preset_npz["class_names"])

    # あいうえおの口形を 20 クラスの係数行から作って後ろへ足す。**表情の後ろに置く** — 既存の
    # プリセットの index が動くと、Unity 側と突き合わせるときに番号がずれる。
    viseme_coefficients, viseme_names = build_viseme_coefficients(
        coefficients, preset_names, visemes_path
    )
    coefficients = np.concatenate([coefficients, viseme_coefficients], axis=0)
    preset_names = preset_names + viseme_names

    expression_basis = npz["expression_basis"]
    if coefficients.shape[1] != expression_basis.shape[0]:
        raise SystemExit(
            f"表情プリセットの成分数 {coefficients.shape[1]} が npz の"
            f" expression_basis {expression_basis.shape[0]} と合わない"
        )
    flat = expression_basis.reshape(expression_basis.shape[0], -1).astype(np.float64)
    displacement = (coefficients @ flat).reshape(-1, expression_basis.shape[1], 3)[:, source]

    preset_scales = np.abs(displacement).max(axis=(1, 2))
    preset_scales[preset_scales == 0.0] = 1.0
    preset_q = np.rint(displacement / preset_scales[:, None, None] * 32767.0).astype(np.int16)
    preset_error = float(
        np.abs(
            preset_q.astype(np.float64) * preset_scales[:, None, None] / 32767.0 - displacement
        ).max()
    )

    # --- まばたき ---------------------------------------------------------
    # 旧 web 版と同じ作り方: WINK_LEFT + WINK_RIGHT の係数を**目領域の成分だけ**残して基底へ当てる。
    # 表情プリセットの 1 本として持たない — まばたきは他の表情へ加算するのではなく、目領域だけ
    # **置き換える**（加算だと surprise のような開瞼系と打ち消し合い、閉じ切らずに眼球が瞼を貫く）。
    expression_names = names_of(npz["expression_names"])
    eye_components = [
        index
        for index, name in enumerate(expression_names)
        if EYE_EXPRESSION_COMPONENT.match(name)
    ]
    if not eye_components:
        raise SystemExit(
            f"目領域の成分が 1 つも無い（成分名の付け方が変わっている: {expression_names[:4]}）"
        )
    missing = [name for name in BLINK_PRESET_CLASSES if name not in preset_names]
    if missing:
        raise SystemExit(f"まばたきに使うプリセット {missing} が表情プリセット npz に無い")
    blink_coefficients = np.zeros(coefficients.shape[1])
    for name in BLINK_PRESET_CLASSES:
        row = coefficients[preset_names.index(name)]
        blink_coefficients[eye_components] += row[eye_components]
    blink = (blink_coefficients @ flat).reshape(expression_basis.shape[1], 3)[source]

    eye_region = np.zeros(npz["vertex_groups"].shape[1], dtype=bool)
    for name in EYE_EXPRESSION_GROUPS:
        if name not in vertex_group_names:
            raise SystemExit(f"vertex group '{name}' が npz に無い")
        eye_region |= npz["vertex_groups"][vertex_group_names.index(name)] > GROUP_THRESHOLD
    blink_scale = float(np.abs(blink).max()) or 1.0
    # **判定は int16 の 1 目盛りで行う。** 目領域の境目は重み 1e-4 で切っているので、外側にも基底の
    # ごく小さな値が残る。1 目盛り未満なら量子化で 0 に落ちる = 送る値としては存在しない。
    outside = float(np.abs(blink[~eye_region[source]]).max())
    step = blink_scale / 32767.0
    if outside > step:
        raise SystemExit(
            f"まばたきの変位が目領域の外へ {outside * 1e6:.3f} um 出ている"
            f"（int16 の 1 目盛り {step * 1e6:.3f} um より大きい）。"
            "クロスフェードを目領域へ閉じられないので、範囲の決め方を見直すこと"
        )
    blink_q = np.rint(blink / blink_scale * 32767.0).astype(np.int16)

    basis_arrays, basis_metadata, basis_report = build_expression_basis_blocks(
        expression_basis, source, expression_names
    )

    joint_identity = np.ascontiguousarray(npz["joint_identity_basis"], dtype=np.float32)
    arrays: dict[str, np.ndarray] = {
        "vertexGroups": groups,
        "jointParentIndices": npz["joint_parent_indices"].astype(np.int32),
        "templateJointPositions": npz["template_joint_positions"].astype(np.float32),
        "jointIdentityBasis": joint_identity,
        "skinJointIndices": skin_indices,
        "skinJointWeights": skin_weights,
        "expressionPresetBasisQ": np.ascontiguousarray(preset_q),
        "blinkBasisQ": np.ascontiguousarray(blink_q),
        **basis_arrays,
    }
    metadata: dict[str, Any] = {
        **basis_metadata,
        "vertex_group_names": list(vertex_group_names),
        "vertex_group_threshold": GROUP_THRESHOLD,
        "joint_names": list(joint_names),
        "expression_preset_names": list(preset_names),
        "expression_preset_scales": [float(value) for value in preset_scales],
        "blink_scale": blink_scale,
        "blink_source": (
            "WINK_LEFT + WINK_RIGHT の係数を "
            + "/".join(EYE_EXPRESSION_COMPONENT.pattern.split("|"))
            + " の成分だけ残したもの（正本は旧 web 版 main.buildBlinkVector）"
        ),
        "eye_expression_groups": list(EYE_EXPRESSION_GROUPS),
        "expression_presets_source": (
            "1-10/2607_Obayashi_Avatar_Mockup_3DGS"
            " Assets/Sandbox/Ooba/GNM/Tools/export_expression_presets.py"
            "（公式 CVAE デコーダの latent 0 = クラス条件付き平均）"
        ),
        "viseme_preset_prefix": VISEME_PRESET_PREFIX,
        "viseme_presets_source": (
            f"{visemes_path.name}（正本は web 側。上のクラス係数行の線形結合なので、"
            "Unity 側は同じ重みを同じクラスへ与えれば同じ口になる）"
        ),
    }
    report: dict[str, float] = {
        **basis_report,
        "group_count": float(groups.shape[0]),
        "preset_count": float(len(preset_names)),
        "viseme_count": float(len(viseme_names)),
        "preset_max_displacement": float(np.abs(displacement).max()),
        "preset_error": preset_error,
        "joint_identity_max": float(np.abs(joint_identity).max()),
        "blink_max_displacement": float(np.linalg.norm(blink, axis=1).max()),
        "blink_outside_micrometers": outside * 1e6,
        "eye_region_vertices": float(eye_region[source].sum()),
    }
    return arrays, metadata, report


# ---------------------------------------------------------------------------
def main() -> None:
    arguments = sys.argv[1:]
    npz_path = Path(arguments[0]) if len(arguments) > 0 else DEFAULT_NPZ
    sparse_path = Path(arguments[1]) if len(arguments) > 1 else DEFAULT_SPARSE_68
    canonical_path = Path(arguments[2]) if len(arguments) > 2 else DEFAULT_CANONICAL
    output_path = Path(arguments[3]) if len(arguments) > 3 else DEFAULT_OUTPUT
    presets_path = (
        Path(arguments[4]) if len(arguments) > 4 else DEFAULT_EXPRESSION_PRESETS
    )
    visemes_path = Path(arguments[5]) if len(arguments) > 5 else DEFAULT_VISEME_PRESETS

    for label, path in (
        ("gnm_head.npz", npz_path),
        ("head_sparse_68.txt", sparse_path),
        ("canonical_face_model.obj", canonical_path),
        ("GnmExpressionPresets_v3_0.npz", presets_path),
        ("viseme_presets.json", visemes_path),
    ):
        if not path.is_file():
            raise SystemExit(
                f"{label} が無い: {path}\n"
                "  python tools/fetch_gnm_assets.py で取得してください"
            )

    with np.load(npz_path, allow_pickle=False) as npz:
        actual = set(npz.files)
        known = set(NPZ_KEYS)
        if actual != known:
            raise SystemExit(
                f"npz のキーが想定と違う: 増えた={sorted(actual - known)}"
                f" 消えた={sorted(known - actual)}。NPZ_KEYS で扱いを決めてから進めること"
            )
        gnm_version = str(npz["version"])
        gnm_variant = str(npz["variant"])
        vertex_group_names = names_of(npz["vertex_group_names"])
        mesh_component_names = names_of(npz["mesh_component_names"])

        triangles, uv_split_source, vertex_uvs = split_by_face_varying_uv(
            npz["triangles"], npz["triangle_uvs"]
        )
        if uv_split_source.shape[0] != EXPECTED_SPLIT_VERTEX_COUNT:
            raise SystemExit(
                f"per-vertex UV 化後の頂点数が {uv_split_source.shape[0]}"
                f"（期待 {EXPECTED_SPLIT_VERTEX_COUNT}）。split かアセットが変わっている"
            )
        source = uv_split_source.astype(np.int64)
        if np.unique(source).size != int(source[-1]) + 1:
            raise SystemExit("uv_split_source に現れない公式頂点がある")

        template = npz["template_vertex_positions"][source].astype(np.float32)
        component = component_ids(
            npz["vertex_groups"], vertex_group_names, mesh_component_names
        )[source]
        ear_region = (npz["vertex_groups"][vertex_group_names.index("ears")] > 0)[source]
        photo_only = photo_only_atlas_region(npz, vertex_group_names)[source]
        rim = mouth_rim_region(npz, vertex_group_names)[source]
        identity_basis = np.ascontiguousarray(
            npz["vertex_identity_basis"][:, source], dtype=np.float32
        )
        preview_arrays, preview_metadata, preview_report = build_preview_arrays(
            npz, source, vertex_group_names, presets_path, visemes_path
        )

    sparse68_indices, sparse68_weights = load_sparse_68(sparse_path)
    canonical = load_canonical_obj(canonical_path)
    (
        dense_mediapipe,
        dense_vertices,
        dense_weights,
        dense_residuals,
        dense_edge,
    ) = build_dense_correspondence(
        canonical,
        template.astype(np.float64),
        triangles.astype(np.int64),
        uv_split_source,
        component,
        tuple(mesh_component_names),
        sparse68_indices,
        sparse68_weights,
    )
    missing = [index for index in MEDIAPIPE_IBUG68 if index not in set(dense_mediapipe.tolist())]
    if missing:
        raise SystemExit(
            f"密対応に iBUG 68 の点が {len(missing)} 個足りない（MediaPipe index {missing}）"
        )

    # 成分ごとの絶対最大で int16 へ。値 = q * scale / 32767。
    scales = np.abs(identity_basis).max(axis=(1, 2)).astype(np.float64)
    scales[scales == 0.0] = 1.0
    quantized = np.rint(identity_basis / scales[:, None, None] * 32767.0).astype(np.int16)
    error = float(
        np.abs(
            quantized.astype(np.float64) * scales[:, None, None] / 32767.0 - identity_basis
        ).max()
    )

    arrays: dict[str, np.ndarray] = {
        "templateVertexPositions": template,
        "vertexUvs": vertex_uvs,
        "triangles": triangles,
        "uvSplitSource": uv_split_source,
        "componentId": component.astype(np.uint8),
        "earRegion": ear_region.astype(np.uint8),
        "atlasPhotoOnlyRegion": photo_only.astype(np.uint8),
        "mouthRimRegion": rim.astype(np.float32),
        "identityBasisQ": quantized,
        "sparse68VertexIndices": sparse68_indices.astype(np.int32),
        "sparse68Weights": sparse68_weights.astype(np.float32),
        "denseMediapipeIndices": dense_mediapipe.astype(np.uint16),
        "denseVertexIndices": dense_vertices.astype(np.int32),
        "denseWeights": dense_weights.astype(np.float32),
        "denseResidualMeters": dense_residuals.astype(np.float32),
        **preview_arrays,
    }
    metadata: dict[str, Any] = {
        "source": (
            f"google/GNM gnm_head.npz (version={gnm_version},"
            f" variant={gnm_variant}, Apache-2.0)"
        ),
        "gnm_version": gnm_version,
        "gnm_variant": gnm_variant,
        "component_names": list(mesh_component_names),
        "identity_basis_scales": [float(value) for value in scales],
        "dense_edge_meters": dense_edge,
        **preview_metadata,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(build_gnmb_container_bytes(GNMB_CONTENT_HEAD_ASSET, arrays, metadata))

    print(
        f"{output_path} を生成しました\n"
        f"  split 頂点 {template.shape[0]:,} / 三角形 {triangles.shape[0]:,}\n"
        f"  identity 成分 {identity_basis.shape[0]}"
        f"（int16 量子化の最大誤差 {error * 1e9:.1f} nm）\n"
        f"  密対応 {dense_mediapipe.shape[0]}/468 点"
        f"（辺の中央値 {dense_edge * 1000:.2f} mm /"
        f" 残差 中央 {float(np.median(dense_residuals)) * 1000:.2f} mm"
        f" 最大 {float(dense_residuals.max()) * 1000:.2f} mm）\n"
        f"  口腔縁 {int((rim > 0).sum()):,} 頂点 /"
        f" 写真専用領域 {int(photo_only.sum()):,} 頂点 /"
        f" 耳 {int(ear_region.sum()):,} 頂点\n"
        f"  3D ビュー用: vertex group {int(preview_report['group_count'])} 本 /"
        f" ジョイント {len(EXPECTED_JOINT_NAMES)} 本"
        f"（identity で最大 {preview_report['joint_identity_max'] * 1000:.1f} mm 動く）/"
        f" 表情プリセット {int(preview_report['preset_count'])} 本"
        f"（うち口形 {int(preview_report['viseme_count'])} 本 /"
        f" 最大変位 {preview_report['preset_max_displacement'] * 1000:.1f} mm /"
        f" int16 量子化の最大誤差 {preview_report['preset_error'] * 1e6:.1f} um）\n"
        f"  まばたき: 最大変位 {preview_report['blink_max_displacement'] * 1000:.2f} mm /"
        f" 目領域 {int(preview_report['eye_region_vertices']):,} 頂点\n"
        f"  表情基底: 領域ブロック {int(preview_report['expression_basis_vertices']):,} 頂点 /"
        f" {preview_report['expression_basis_bytes'] / 1e6:.1f} MB"
        f"（int16 量子化の最大誤差 {preview_report['expression_basis_error'] * 1e9:.1f} nm）\n"
        f"    領域の重なり: 延べ"
        f" {int(preview_report['expression_region_shared_vertices']):,} 頂点 /"
        f" 正規化内積の最大 {preview_report['expression_region_cosine']:.3f}"
        "（分けずにまとめて解く根拠）\n"
        f"  {output_path.stat().st_size / 1e6:.1f} MB"
    )


if __name__ == "__main__":
    main()
