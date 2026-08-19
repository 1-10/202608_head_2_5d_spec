# GNM Head (github.com/google/GNM, Apache-2.0) の gnm_head.npz から、
# ブラウザ用の軽量アセット gnm_head_lite.bin を生成するビルド時スクリプト。
#
# 使い方:
#   git clone --depth 1 https://github.com/google/GNM.git <どこか>
#   curl -LO https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj
#   python tools/export_gnm_assets.py <GNM>/gnm/shape/data/versions/v3_0/gnm_head.npz canonical_face_model.obj
#   → public/gnm/gnm_head_lite.bin (既定) が生成される
#   .obj (MediaPipe canonical face model, Apache-2.0) を渡すと、468点の密対応表
#   (Umeyama+TPS整列→GNM表面へ投影) も焼き込まれ、フィット精度が上がる
#
# 含まれるもの (見える表面だけに間引く):
# - skin_exterior + eye_exteriors の頂点/三角形サブセット (歯・舌・口腔/眼窩内部を除外)
# - identity基底の上位K成分 (int16量子化。基底はPCAで分散降順・係数はz-scoreスケール)
# - HEAD_SPARSE_68 ランドマークのbarycentric定義 (iBUG-68順)
# - 耳の頂点グループ重み (髪との整合処理用)
#
# バイナリ形式: b'GNML' + uint32(jsonバイト長) + JSONヘッダ + ペイロード(4バイト境界)。
# JSONヘッダに各セクションのオフセット/型を書くので、レイアウトはヘッダが正本。

import json
import struct
import sys
from pathlib import Path

import numpy as np

IDENTITY_BASIS_COUNT = 64  # 上位K成分 (ノルム降順を確認済み。K以降は寄与が微小)

# 表情基底は領域ごとにPCA順 (ノルム降順を確認済み)。ランダム表情デモ用に
# 主要領域の上位成分だけ持つ (舌・瞳孔は正面写真デモでは効果が薄いため除外)
EXPRESSION_PICKS = {'left_eye_region': 10, 'right_eye_region': 10, 'lower_face_region': 20}

# MediaPipe FaceMesh 468点 → iBUG-68 の対応表 (src/gnmHead.ts と同一の定数)。
# 密対応構築時の初期整列 (Umeyama+TPS) の制御点に使う
MEDIAPIPE_IBUG68 = [
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    168, 197, 5, 4, 75, 97, 2, 326, 305,
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317, 14, 87,
]

# MediaPipe FACEMESH_FACE_OVAL の36点 (顔輪郭。フィット時の横幅追従のため高重み)
MEDIAPIPE_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
    377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

DENSE_MAX_RESIDUAL_M = 0.008  # 整列後の投影残差がこれを超える点は対応から除外


def load_canonical_obj(path: Path) -> np.ndarray:
    """MediaPipe canonical_face_model.obj (頂点i = landmark i, 468頂点) を読む。"""
    verts = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.startswith('v '):
                verts.append([float(x) for x in line.split()[1:4]])
    v = np.asarray(verts, dtype=np.float64)
    assert v.shape[0] == 468, f'canonical face modelの頂点数が468ではない: {v.shape[0]}'
    return v


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """3D相似変換 (s, R, t) を最小二乗で求める (Umeyama 1991)。"""
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    sc = src - mu_s
    dc = dst - mu_d
    cov = dc.T @ sc / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    diag = np.diag([1.0, 1.0, d])
    r = u @ diag @ vt
    scale = np.trace(np.diag(s) @ diag) / (sc ** 2).sum(axis=1).mean()
    t = mu_d - scale * r @ mu_s
    return scale, r, t


def tps_warp(ctrl_src: np.ndarray, ctrl_dst: np.ndarray, points: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """3D thin-plate spline (カーネルU(r)=r) で points を warp する。"""
    n = len(ctrl_src)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)

    k = kernel(ctrl_src, ctrl_src) + np.eye(n) * reg
    p = np.hstack([np.ones((n, 1)), ctrl_src])
    a = np.zeros((n + 4, n + 4))
    a[:n, :n] = k
    a[:n, n:] = p
    a[n:, :n] = p.T
    b = np.zeros((n + 4, 3))
    b[:n] = ctrl_dst
    sol = np.linalg.solve(a, b)
    w, aff = sol[:n], sol[n:]
    kq = kernel(points, ctrl_src)
    return kq @ w + np.hstack([np.ones((len(points), 1)), points]) @ aff


def project_point_to_triangles(point: np.ndarray, tri_verts: np.ndarray) -> tuple[int, np.ndarray, float]:
    """点を三角形群へ投影し (三角形index, barycentric, 距離) を返す。tri_verts: (T,3,3)。"""
    a, b, c = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
    ab = b - a
    ac = c - a
    ap = point[None, :] - a
    d00 = (ab * ab).sum(1)
    d01 = (ab * ac).sum(1)
    d11 = (ac * ac).sum(1)
    d20 = (ap * ab).sum(1)
    d21 = (ap * ac).sum(1)
    denom = np.maximum(1e-12, d00 * d11 - d01 * d01)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    v = np.clip(v, 0, 1)
    w = np.clip(w, 0, 1)
    over = v + w > 1
    scale_over = np.where(over, v + w, 1)
    v = v / scale_over
    w = w / scale_over
    closest = a + ab * v[:, None] + ac * w[:, None]
    dist = np.linalg.norm(closest - point[None, :], axis=1)
    ti = int(np.argmin(dist))
    return ti, np.array([1 - v[ti] - w[ti], v[ti], w[ti]]), float(dist[ti])


def build_dense_correspondence(
    canonical: np.ndarray,
    gnm_positions: np.ndarray,
    gnm_triangles: np.ndarray,
    lm_indices: np.ndarray,
    lm_weights: np.ndarray,
    exclude_vertex_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """canonical 468頂点をGNM表面へ整列・投影し、barycentric対応表を作る。

    返り値: (mpIndices uint16 (M,), triIndices (M,3) uint32, weights (M,3) f32, conf (M,) f32)
    """
    gnm68 = (gnm_positions[lm_indices] * lm_weights[..., None]).sum(axis=1)  # (68,3)
    can68 = canonical[MEDIAPIPE_IBUG68]

    scale, r, t = umeyama_similarity(can68, gnm68)
    aligned = canonical @ (scale * r).T + t
    # 68点対応を厳密一致させるTPSで残差 (頬・額のトポロジ差) を吸収する
    aligned = tps_warp(aligned[MEDIAPIPE_IBUG68], gnm68, aligned)

    # 投影対象: 除外グループ (眼球・耳) を含まない三角形のみ
    tri_ok = ~exclude_vertex_mask[gnm_triangles].any(axis=1)
    tris = gnm_triangles[tri_ok]
    tri_verts_all = gnm_positions[tris]
    centroids = tri_verts_all.mean(axis=1)

    mp_out, tri_out, w_out, conf_out = [], [], [], []
    for mp_idx in range(468):
        p = aligned[mp_idx]
        near = np.linalg.norm(centroids - p[None, :], axis=1) < 0.03
        if not near.any():
            continue
        ti_local, bary, dist = project_point_to_triangles(p, tri_verts_all[near])
        if dist > DENSE_MAX_RESIDUAL_M:
            continue
        tri = tris[np.nonzero(near)[0][ti_local]]
        mp_out.append(mp_idx)
        tri_out.append(tri)
        w_out.append(bary)
        conf_out.append(1.0 / (1.0 + dist / 0.002))

    return (
        np.asarray(mp_out, dtype=np.uint16),
        np.asarray(tri_out, dtype=np.uint32),
        np.asarray(w_out, dtype=np.float32),
        np.asarray(conf_out, dtype=np.float32),
    )


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    npz_path = Path(sys.argv[1])
    canonical_path = None
    out_path = Path(__file__).parent.parent / 'public' / 'gnm' / 'gnm_head_lite.bin'
    for arg in sys.argv[2:]:
        if arg.endswith('.obj'):
            canonical_path = Path(arg)
        else:
            out_path = Path(arg)

    # allow_pickle=False (既定): npzは素のndarrayのみでpickleを含まない。
    # 入力はGoogle公式リポジトリ配布の gnm_head.npz を想定。
    d = np.load(npz_path)
    group_names = [str(n) for n in d['vertex_group_names']]
    groups = d['vertex_groups']

    def group(name: str) -> np.ndarray:
        return groups[group_names.index(name)]

    # skin_exteriorではなくskin全体から口腔内(mouth_sock)だけ除外する。
    # skin_exteriorは鼻孔内部を含まず、鼻孔がメッシュの穴 (黒い点) として見えるため
    vertex_mask = ((group('skin') > 0.5) & (group('mouth_sock') < 0.5)) | (group('eye_exteriors') > 0.5)
    old_to_new = np.full(len(vertex_mask), -1, dtype=np.int64)
    old_to_new[vertex_mask] = np.arange(vertex_mask.sum())

    positions = d['template_vertex_positions'][vertex_mask].astype(np.float32)  # (N,3) メートル
    tris_all = d['triangles']
    tri_keep = vertex_mask[tris_all].all(axis=1)
    triangles = old_to_new[tris_all[tri_keep]].astype(np.uint32)  # (T,3)

    basis_full = d['vertex_identity_basis'][:IDENTITY_BASIS_COUNT][:, vertex_mask, :]  # (K,N,3)
    scales = np.abs(basis_full).max(axis=(1, 2)).astype(np.float32)  # (K,)
    scales[scales == 0] = 1.0
    basis_q = np.round(basis_full / scales[:, None, None] * 32767).astype(np.int16)

    lm = np.loadtxt(npz_path.parent.parent.parent / 'landmarks' / 'head_sparse_68.txt')
    lm_idx_old = lm[:, 0::2].astype(np.int64)
    lm_w = lm[:, 1::2].astype(np.float32)
    assert vertex_mask[lm_idx_old].all(), 'ランドマーク頂点がサブセット外です'
    lm_idx = old_to_new[lm_idx_old].astype(np.uint32)

    ear_weight = np.clip(group('ears')[vertex_mask] * 255, 0, 255).astype(np.uint8)

    expr_names = [str(n) for n in d['expression_names']]
    expr_indices = []
    for prefix, count in EXPRESSION_PICKS.items():
        expr_indices += [i for i, n in enumerate(expr_names) if n.startswith(prefix)][:count]
    expr_full = d['expression_basis'][expr_indices][:, vertex_mask, :]  # (M,N,3)
    expr_scales = np.abs(expr_full).max(axis=(1, 2)).astype(np.float32)
    expr_scales[expr_scales == 0] = 1.0
    expr_q = np.round(expr_full / expr_scales[:, None, None] * 32767).astype(np.int16)

    # 密対応 (MediaPipe 468点 → GNM表面barycentric)。canonical_face_model.obj があれば構築
    dense = None
    if canonical_path is not None:
        canonical = load_canonical_obj(canonical_path)
        exclude = (group('ears') > 0.5) | (group('eyes') > 0.5)
        dense = build_dense_correspondence(
            canonical, positions.astype(np.float64), triangles.astype(np.int64), lm_idx.astype(np.int64), lm_w, exclude[vertex_mask],
        )
        mp_idx_arr, dense_tri, dense_w, dense_conf = dense
        # 顔輪郭 (face oval) は横幅追従のため高重みにする
        oval = np.isin(mp_idx_arr, MEDIAPIPE_FACE_OVAL)
        dense_weight = (dense_conf * np.where(oval, 1.6, 1.0)).astype(np.float32)
        print(f'密対応: {len(mp_idx_arr)}/468 点 (残差>{DENSE_MAX_RESIDUAL_M*1000:.0f}mm除外, 輪郭{oval.sum()}点は重み1.6)')

    sections = {
        'positions': positions,       # float32 (N,3)
        'triangles': triangles,       # uint32 (T,3)
        'identityBasisQ': basis_q,    # int16 (K,N,3)
        'landmarkIndices': lm_idx,    # uint32 (68,3)
        'landmarkWeights': lm_w,      # float32 (68,3)
        'earWeight': ear_weight,      # uint8 (N,)
        'expressionBasisQ': expr_q,   # int16 (M,N,3)
    }
    if dense is not None:
        sections['denseMpIndices'] = dense[0]      # uint16 (M,) MediaPipe landmark index
        sections['denseTriIndices'] = dense[1]     # uint32 (M,3)
        sections['denseBaryWeights'] = dense[2]    # float32 (M,3)
        sections['denseFitWeights'] = dense_weight  # float32 (M,) フィット重み (信頼度×領域重み)

    payload = bytearray()
    section_meta = {}
    for name, arr in sections.items():
        if len(payload) % 4:
            payload.extend(b'\x00' * (4 - len(payload) % 4))
        raw = np.ascontiguousarray(arr).tobytes()
        section_meta[name] = {'offset': len(payload), 'byteLength': len(raw), 'dtype': str(arr.dtype)}
        payload.extend(raw)

    header = {
        'source': 'google/GNM gnm_head.npz v3_0 (Apache-2.0)',
        'vertexCount': int(positions.shape[0]),
        'triangleCount': int(triangles.shape[0]),
        'identityBasisCount': IDENTITY_BASIS_COUNT,
        'identityBasisScales': [float(s) for s in scales],
        'expressionBasisCount': len(expr_indices),
        'expressionBasisScales': [float(s) for s in expr_scales],
        'expressionNames': [expr_names[i] for i in expr_indices],
        'landmarkCount': int(lm_idx.shape[0]),
        'denseLandmarkCount': int(len(dense[0])) if dense is not None else 0,
        'sections': section_meta,
    }
    header_bytes = json.dumps(header).encode('utf-8')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(b'GNML')
        f.write(struct.pack('<I', len(header_bytes)))
        f.write(header_bytes)
        f.write(payload)

    total_mb = out_path.stat().st_size / 1e6
    print(f'{out_path} を生成しました: 頂点 {positions.shape[0]:,} / 三角形 {triangles.shape[0]:,} / 基底 {IDENTITY_BASIS_COUNT} / {total_mb:.1f} MB')


if __name__ == '__main__':
    main()
