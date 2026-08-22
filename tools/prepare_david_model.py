"""DAViDモデル (Microsoft, ICCV 2025) のダウンロードと量子化。

DAViD: 人物特化の単眼推定 (相対Depth / 表面法線 / ソフト前景)。
100%合成データ (SynthHuman) 学習で「学習データまで商用クリーン」を満たす。
- モデル/コード: MIT (https://github.com/microsoft/DAViD)
- SynthHuman: CDLA-Permissive-2.0 (データ商用可・学習成果物に義務なし)

fp32 428MB/タスクのままではブラウザ配布に大きすぎるため、
- fp16 (~215MB): WebGPU実行用 (精度劣化ほぼなし)
- int8 (~110MB): WASMフォールバック用 (実写でfp32比 平均誤差1.9%/レンジ)
の2種を public/david/ へ生成する (gitignore対象)。

使い方:
    pip install onnx onnxruntime onnxconverter-common
    python tools/prepare_david_model.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

BASE_URL = "https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models"

# タスク名 → (配布ファイル名, バリアント)。
# multitask (ViT-L) は1回の推論でdepth/normal/foregroundを同時に返す。
# アプリはmultitaskのみを使う (タスク別ViT-Bは過去に検証用で使用)
TASKS = {
    "multitask": ("multi-task-model-vitl16_384.onnx", "vitl16"),
}

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "public" / "david"


def fix_cast_nodes(model) -> int:
    """onnxconverter-commonの既知の取りこぼしを修正する。

    Castノードのto属性がFLOATのまま残り、宣言型 (float16) と不整合になって
    ロードに失敗するため、宣言がfloat16なのにto=FLOATのCastをFLOAT16へ直す。
    """
    from onnx import TensorProto

    declared = {
        vi.name: vi.type.tensor_type.elem_type
        for vi in list(model.graph.value_info) + list(model.graph.output) + list(model.graph.input)
    }
    fixed = 0
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        to_attr = next(a for a in node.attribute if a.name == "to")
        if declared.get(node.output[0]) == TensorProto.FLOAT16 and to_attr.i == TensorProto.FLOAT:
            to_attr.i = TensorProto.FLOAT16
            fixed += 1
    return fixed


def prepare(task: str, filename: str, variant: str) -> None:
    fp32 = OUT_DIR / f"david-{task}-{variant}-fp32.onnx"
    fp16 = OUT_DIR / f"david-{task}-{variant}-fp16.onnx"
    int8 = OUT_DIR / f"david-{task}-{variant}-int8.onnx"

    if not fp32.exists():
        url = f"{BASE_URL}/{filename}"
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, fp32)
    print(f"{task} fp32: {fp32.stat().st_size // 2**20} MB")

    if not fp16.exists():
        import onnx
        from onnxconverter_common import float16

        model = onnx.load(str(fp32))
        # keep_io_types=True: 入出力はfloat32のまま (JS側の扱いを単純にする)
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        print(f"{task} fixed Cast nodes: {fix_cast_nodes(model_fp16)}")
        onnx.save(model_fp16, str(fp16))
    print(f"{task} fp16: {fp16.stat().st_size // 2**20} MB")

    if not int8.exists():
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    print(f"{task} int8: {int8.stat().st_size // 2**20} MB")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for task, (filename, variant) in TASKS.items():
        prepare(task, filename, variant)


if __name__ == "__main__":
    main()
