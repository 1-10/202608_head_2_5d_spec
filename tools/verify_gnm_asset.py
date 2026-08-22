# 生成済み gnm_head_lite.bin を直接パースし、公式 gnm_head.npz とデータで突き合わせる。
import json
import struct
import sys
from pathlib import Path

import numpy as np

BIN = Path(sys.argv[1])
NPZ = Path(sys.argv[2])

raw = BIN.read_bytes()
assert raw[:4] == b'GNML'
hlen = struct.unpack('<I', raw[4:8])[0]
hdr = json.loads(raw[8:8 + hlen].decode('utf-8'))
base = 8 + hlen


def sec(name, dtype, shape=None):
    s = hdr['sections'][name]
    n = s['byteLength'] // np.dtype(dtype).itemsize
    a = np.frombuffer(raw, dtype=dtype, count=n, offset=base + s['offset'])
    return a.reshape(shape) if shape else a


d = np.load(NPZ)
names = [str(n) for n in d['vertex_group_names']]
groups = d['vertex_groups']


def G(name):
    return groups[names.index(name)] > 0.5


P_off = d['template_vertex_positions']
T_off = d['triangles']
mouth_interior = G('mouth_sock') | G('teeth') | G('gums') | G('tongue')
vertex_mask = G('skin') | G('eye_exteriors') | mouth_interior
old_to_new = np.full(len(vertex_mask), -1, dtype=np.int64)
old_to_new[vertex_mask] = np.arange(vertex_mask.sum())
N = int(vertex_mask.sum())

pos = sec('positions', '<f4', (hdr['vertexCount'], 3))
tri = sec('triangles', '<u4', (hdr['triangleCount'], 3))
itri = sec('interiorTriangles', '<u4', (hdr['interiorTriangleCount'], 3))
part = sec('mouthPartId', '<u1')

print('アセット: 頂点 %d / 頭部三角形 %d / 口腔内三角形 %d'
      % (hdr['vertexCount'], hdr['triangleCount'], hdr['interiorTriangleCount']))
print('npzから再計算した subset 頂点数 = %d  一致=%s' % (N, N == hdr['vertexCount']))

print()
print('[1] 頂点座標 (float32へのキャスト以外の改変が無いか)')
for nm in ['teeth', 'gums', 'tongue', 'mouth_sock', 'upper_teeth_and_gums', 'lower_teeth_and_gums']:
    m = G(nm)
    idx = old_to_new[m]
    same = np.array_equal(pos[idx], P_off[m].astype(np.float32))
    dmax = np.abs(pos[idx].astype(np.float64) - P_off[m].astype(np.float64)).max()
    print('  %-22s n=%5d  bit一致=%-5s 最大差=%.3e m' % (nm, m.sum(), same, dmax))

print()
print('[2] 三角形の接続 (歯並び・歯の面の張り方)')
tri_keep = vertex_mask[T_off].all(axis=1)
kept = T_off[tri_keep]
is_int = mouth_interior[kept].any(axis=1)
exp_head = old_to_new[kept[~is_int]].astype(np.uint32)
exp_int = old_to_new[kept[is_int]].astype(np.uint32)
print('  頭部三角形   : %d 枚 / 期待 %d 枚  順序込み完全一致=%s'
      % (len(tri), len(exp_head), np.array_equal(tri, exp_head)))
print('  口腔内三角形 : %d 枚 / 期待 %d 枚  順序込み完全一致=%s'
      % (len(itri), len(exp_int), np.array_equal(itri, exp_int)))


def canon(t):
    return set(map(tuple, np.sort(t, axis=1)))


for nm in ['teeth', 'gums', 'tongue', 'mouth_sock', 'upper_teeth_and_gums', 'lower_teeth_and_gums']:
    m = G(nm)
    off_t = T_off[m[T_off].all(axis=1)]
    off_c = canon(old_to_new[off_t])
    inside = np.zeros(N, dtype=bool)
    inside[old_to_new[m]] = True
    our_t = itri[inside[itri].all(axis=1)]
    our_c = canon(our_t)
    print('  %-22s 公式 %5d 枚 / アセット %5d 枚  集合一致=%-5s 脱落=%d 余剰=%d'
          % (nm, len(off_t), len(our_t), our_c == off_c, len(off_c - our_c), len(our_c - off_c)))

print()
print('[3] 巻き方向 (三角形の頂点の並び順)')
for nm in ['teeth', 'tongue', 'mouth_sock']:
    m = G(nm)
    off_t = T_off[m[T_off].all(axis=1)]
    off_map = {tuple(sorted(t)): tuple(t) for t in old_to_new[off_t]}
    inside = np.zeros(N, dtype=bool)
    inside[old_to_new[m]] = True
    our_t = itri[inside[itri].all(axis=1)]
    same = 0
    for t in our_t:
        o = off_map.get(tuple(sorted(t)))
        if o is None:
            continue
        # 巡回一致 (同じ向き) か
        if o == tuple(t) or o == (t[1], t[2], t[0]) or o == (t[2], t[0], t[1]):
            same += 1
    print('  %-12s 向きまで一致 %d / %d' % (nm, same, len(our_t)))

print()
print('[4] int16量子化の誤差 (歯の頂点)')
K = hdr['identityBasisCount']
ib = sec('identityBasisQ', '<i2', (K, N, 3)).astype(np.float64)
isc = np.asarray(hdr['identityBasisScales'])
ours_ib = ib * isc[:, None, None] / 32767
off_ib = d['vertex_identity_basis'][:K][:, vertex_mask, :]
tm = old_to_new[G('teeth')]
e = np.abs(ours_ib[:, tm] - off_ib[:, tm])
print('  identity基底 %d成分: 最大誤差=%.3f um  基底の最大振幅=%.3f mm  相対=%.2e'
      % (K, e.max() * 1e6, np.abs(off_ib[:, tm]).max() * 1000, e.max() / max(1e-12, np.abs(off_ib[:, tm]).max())))

M = hdr['expressionBasisCount']
eb = sec('expressionBasisQ', '<i2', (M, N, 3)).astype(np.float64)
esc = np.asarray(hdr['expressionBasisScales'])
ours_eb = eb * esc[:, None, None] / 32767
en = [str(n) for n in d['expression_names']]
picks = []
for pre, cnt in [('left_eye_region', 10), ('right_eye_region', 10), ('lower_face_region', 20), ('tongue', 4)]:
    picks += [i for i, n in enumerate(en) if n.startswith(pre)][:cnt]
print('  ヘッダの成分名 == 期待する選び方: %s' % (hdr['expressionNames'] == [en[i] for i in picks]))
off_eb = d['expression_basis'][picks][:, vertex_mask, :]
e2 = np.abs(ours_eb[:, tm] - off_eb[:, tm])
print('  表情基底 %d成分:  最大誤差=%.3f um  基底の最大振幅=%.3f mm  相対=%.2e'
      % (M, e2.max() * 1e6, np.abs(off_eb[:, tm]).max() * 1000, e2.max() / max(1e-12, np.abs(off_eb[:, tm]).max())))

print()
print('[5] mouthPartId が公式グループ定義と一致するか')
expect = np.zeros(len(vertex_mask), dtype=np.uint8)
expect[G('gums')] = 3
expect[G('teeth')] = 2
expect[G('tongue')] = 4
expect[G('mouth_sock')] = 1
print('  一致=%s' % np.array_equal(part, expect[vertex_mask]))
for i, nm in [(1, 'mouth_sock'), (2, 'teeth'), (3, 'gums'), (4, 'tongue')]:
    print('    id=%d %-11s %5d 頂点 (公式グループ %5d)' % (i, nm, int((part == i).sum()), int(G(nm).sum())))
