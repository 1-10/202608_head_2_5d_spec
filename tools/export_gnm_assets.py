# GNM Head (github.com/google/GNM, Apache-2.0) の gnm_head.npz から、
# ブラウザ用の軽量アセット gnm_head_lite.bin を生成するビルド時スクリプト。
#
# 使い方:
#   git clone --depth 1 https://github.com/google/GNM.git <どこか>
#   python tools/export_gnm_assets.py <GNMリポジトリ>/gnm/shape/data/versions/v3_0/gnm_head.npz
#   → public/gnm/gnm_head_lite.bin (既定) が生成される
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


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    npz_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent.parent / 'public' / 'gnm' / 'gnm_head_lite.bin'

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

    sections = {
        'positions': positions,       # float32 (N,3)
        'triangles': triangles,       # uint32 (T,3)
        'identityBasisQ': basis_q,    # int16 (K,N,3)
        'landmarkIndices': lm_idx,    # uint32 (68,3)
        'landmarkWeights': lm_w,      # float32 (68,3)
        'earWeight': ear_weight,      # uint8 (N,)
    }

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
        'landmarkCount': int(lm_idx.shape[0]),
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
