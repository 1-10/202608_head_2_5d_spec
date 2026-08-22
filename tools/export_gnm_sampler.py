# GNM 公式 ExpressionSampler (gnm/shape/semantic_sampler.py) の CVAE デコーダを
# ブラウザで動かすための重みファイルへ変換するビルド時スクリプト。
#
# 使い方:
#   python tools/export_gnm_sampler.py <GNM>/gnm/shape/data/semantic_sampler/expression_decoder_model.h5 \
#                                      <GNM>/gnm/shape/data/versions/v3_0/gnm_head.npz
#   → public/gnm/gnm_expression_decoder.bin (既定) が生成される
#
# デコーダの構造は h5 の model_config が正本で、そこから読み取ったものを検証する:
#   入力 = concat(latent_input(64), decoder_label_input(20))  ※この順
#   dense_13(64,relu) → dense_14(128,relu) → dense_15(256,relu)
#   → dense_16(512,relu) → dense_17(383,linear)
#
# 重みは float16 で書く。fp32との差は実測で係数最大 0.00268 /
# 顔頂点変位で最大 0.0065mm なので影響しない (サイズは 1.50MB → 0.75MB)。
#
# バイナリ形式: b'GNMS' + uint32(jsonバイト長) + JSONヘッダ + ペイロード(4バイト境界)。

import json
import re
import struct
import sys
from pathlib import Path

import h5py
import numpy as np

EXPECTED = [
    ('dense_13', 84, 64, 'relu'),
    ('dense_14', 64, 128, 'relu'),
    ('dense_15', 128, 256, 'relu'),
    ('dense_16', 256, 512, 'relu'),
    ('dense_17', 512, 383, 'linear'),
]


def read_class_names(h5_path: Path) -> list[str]:
    """公式 semantic_sampler.py の Expression enum からクラス名を読む。

    ExpressionSampler.expression_names は member.name.lower() なので同じ形にする。
    ソースの場所が変わったら黙って古い名前を使わずエラーにする。
    """
    src = h5_path.parent.parent.parent / 'semantic_sampler.py'
    if not src.exists():
        raise SystemExit(f'{src} が見つかりません (Expression enum の取得元)')
    text = src.read_text(encoding='utf-8')
    m = re.search(r'class Expression\(enum\.IntEnum\):(.*?)\n\n\n', text, re.S)
    if not m:
        raise SystemExit('semantic_sampler.py から Expression enum を読めません')
    pairs = re.findall(r'^\s+([A-Z_]+)\s*=\s*(\d+)\s*$', m.group(1), re.M)
    names: dict[int, str] = {int(v): n.lower() for n, v in pairs}
    if sorted(names) != list(range(len(names))):
        raise SystemExit(f'Expression enum の値が連番ではありません: {sorted(names)}')
    return [names[i] for i in range(len(names))]


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    h5_path = Path(sys.argv[1])
    npz_path = Path(sys.argv[2])
    out_path = (
        Path(sys.argv[3]) if len(sys.argv) > 3
        else Path(__file__).parent.parent / 'public' / 'gnm' / 'gnm_expression_decoder.bin'
    )

    f = h5py.File(h5_path, 'r')
    mw = f['model_weights']
    order = [n.decode() if isinstance(n, bytes) else n for n in mw.attrs['layer_names']]
    dense = [n for n in order if n.startswith('dense')]
    if dense != [name for name, *_ in EXPECTED]:
        raise SystemExit(f'デコーダの層構成が想定と違います: {dense}')

    layers = []
    payload = bytearray()
    for name, n_in, n_out, act in EXPECTED:
        k = np.array(mw[name][name]['kernel:0'], dtype=np.float32)
        b = np.array(mw[name][name]['bias:0'], dtype=np.float32)
        if k.shape != (n_in, n_out) or b.shape != (n_out,):
            raise SystemExit(f'{name} の形が想定と違います: kernel {k.shape} bias {b.shape}')
        meta = {'name': name, 'in': n_in, 'out': n_out, 'activation': act}
        for key, arr in (('kernel', k), ('bias', b)):
            if len(payload) % 4:
                payload.extend(b'\x00' * (4 - len(payload) % 4))
            raw = np.ascontiguousarray(arr.astype(np.float16)).tobytes()
            meta[key] = {'offset': len(payload), 'byteLength': len(raw)}
            payload.extend(raw)
        layers.append(meta)

    class_names = read_class_names(h5_path)
    d = np.load(npz_path)
    output_names = [str(n) for n in d['expression_names']]
    if len(output_names) != EXPECTED[-1][2]:
        raise SystemExit(f'出力次元 {EXPECTED[-1][2]} と expression_names {len(output_names)} が不一致')
    if len(class_names) != EXPECTED[0][1] - 64:
        raise SystemExit(f'クラス数 {len(class_names)} が入力次元と不整合 (latent 64 + label)')

    header = {
        'source': 'google/GNM expression_decoder_model.h5 + gnm_head.npz (Apache-2.0)',
        'latentDim': 64,
        'numClasses': len(class_names),
        # 入力は concat(latent, label)。h5 の model_config の入力順がこの順
        'classNames': class_names,
        # 出力383成分の名前 (アセット側の成分と名前で対応づけるため)
        'outputNames': output_names,
        'dtype': 'float16',
        'layers': layers,
    }
    header_bytes = json.dumps(header).encode('utf-8')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as fp:
        fp.write(b'GNMS')
        fp.write(struct.pack('<I', len(header_bytes)))
        fp.write(header_bytes)
        fp.write(payload)
    total = sum(l['kernel']['byteLength'] + l['bias']['byteLength'] for l in layers)
    print(f'{out_path} を生成しました: {len(class_names)} クラス / 出力 {len(output_names)} 成分 / '
          f'重み {total / 1e6:.2f} MB / 全体 {out_path.stat().st_size / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
