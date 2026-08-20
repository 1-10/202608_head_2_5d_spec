"""DAViD Depthモデル (Microsoft, ICCV 2025) のダウンロードと量子化。

DAViD: 人物特化の単眼相対Depth。100%合成データ (SynthHuman) 学習で
「学習データまで商用クリーン」を満たす。
- モデル/コード: MIT (https://github.com/microsoft/DAViD)
- SynthHuman: CDLA-Permissive-2.0 (データ商用可・学習成果物に義務なし)

fp32 428MBのままではブラウザ配布に大きすぎるため、
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

MODEL_URL = (
    "https://facesyntheticspubwedata.z6.web.core.windows.net"
    "/iccv-2025/models/depth-model-vitb16_384.onnx"
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "public" / "david"
FP32 = OUT_DIR / "david-depth-vitb16-fp32.onnx"
FP16 = OUT_DIR / "david-depth-vitb16-fp16.onnx"
INT8 = OUT_DIR / "david-depth-vitb16-int8.onnx"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not FP32.exists():
        print(f"downloading {MODEL_URL} ...")
        urllib.request.urlretrieve(MODEL_URL, FP32)
    print(f"fp32: {FP32.stat().st_size // 2**20} MB")

    if not FP16.exists():
        import onnx
        from onnx import TensorProto
        from onnxconverter_common import float16

        model = onnx.load(str(FP32))
        # keep_io_types=True: 入出力はfloat32のまま (JS側の扱いを単純にする)
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)

        # onnxconverter-commonの既知の取りこぼし: Castノードのto属性が
        # FLOATのまま残り、宣言型 (float16) と不整合になってロードに失敗する。
        # 宣言がfloat16なのにto=FLOATのCastをto=FLOAT16へ修正する
        declared = {
            vi.name: vi.type.tensor_type.elem_type
            for vi in list(model_fp16.graph.value_info)
            + list(model_fp16.graph.output)
            + list(model_fp16.graph.input)
        }
        fixed = 0
        for node in model_fp16.graph.node:
            if node.op_type != "Cast":
                continue
            to_attr = next(a for a in node.attribute if a.name == "to")
            if declared.get(node.output[0]) == TensorProto.FLOAT16 and to_attr.i == TensorProto.FLOAT:
                to_attr.i = TensorProto.FLOAT16
                fixed += 1
        print(f"fixed Cast nodes: {fixed}")
        onnx.save(model_fp16, str(FP16))
    print(f"fp16: {FP16.stat().st_size // 2**20} MB")

    if not INT8.exists():
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(FP32), str(INT8), weight_type=QuantType.QInt8)
    print(f"int8: {INT8.stat().st_size // 2**20} MB")


if __name__ == "__main__":
    main()
