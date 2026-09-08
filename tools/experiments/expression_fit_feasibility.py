"""MediaPipe の点から GNM の表情係数 383 本がどこまで解けるかをオフラインで測る.

    python tools/experiments/expression_fit_feasibility.py

**検証用のスクリプト**で、アセットにも実行時にも何も足さない。測るのは「MediaPipe の
468 点が乗る位置から、公式の表情基底 383 成分が線形代数として同定できるか」だけ。
MediaPipe が本物の表情をどれだけ正しく捉えているかは、ここでは測れない（カメラでしか
分からない）。

出す数字は 2 つ:

- **有効自由度 (N_eff)**: リッジ回帰の effective degrees of freedom。事前分布の分散のうち
  観測で決まった割合の総和。「150 成分のうち実質いくつ決まるか」を表す
- **復元誤差**: 公式 20 プリセットを真値として、点の変位から係数を解き直し、全頂点の
  変位がどれだけ合うか（見た目に直結するのはこちら）

観測は 2 通り試す。MediaPipe の z は x, y ほど信用できないので `xy` が本命、`xyz` は上限。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NOISE_MILLIMETERS = (0.0, 0.2, 0.5, 1.0, 2.0)
REGION_OF = {
    "lower_face_region": "下顔面",
    "left_eye_region": "左目",
    "right_eye_region": "右目",
    "tongue": "舌",
    "pupils": "瞳",
}


def load_exporter():
    """アセット生成スクリプトを module として読む（密対応の作り方の正本はあちら）。"""
    path = REPOSITORY_ROOT / "tools" / "export_gnm_assets.py"
    spec = importlib.util.spec_from_file_location("export_gnm_assets", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_observation_basis(exporter, npz):
    """expression_basis を密対応の点の上へ落とす。戻り値 (383, M, 3) ほか。"""
    group_names = [str(name) for name in npz["vertex_group_names"]]
    components = [str(name) for name in npz["mesh_component_names"]]
    triangles, uv_split_source, _ = exporter.split_by_face_varying_uv(
        npz["triangles"], npz["triangle_uvs"]
    )
    source = uv_split_source.astype(np.int64)
    template = npz["template_vertex_positions"][source].astype(np.float64)
    component_id = exporter.component_ids(npz["vertex_groups"], group_names, components)[source]

    sparse_indices, sparse_weights = exporter.load_sparse_68(
        REPOSITORY_ROOT / "assets" / "gnm" / "head_sparse_68.txt"
    )
    canonical = exporter.load_canonical_obj(
        REPOSITORY_ROOT / "assets" / "mediapipe" / "canonical_face_model.obj"
    )
    _, vertex_indices, weights, residuals, edge = exporter.build_dense_correspondence(
        canonical,
        template,
        triangles.astype(np.int64),
        uv_split_source,
        component_id,
        tuple(components),
        sparse_indices,
        sparse_weights,
    )
    basis = npz["expression_basis"].astype(np.float64)
    sampled = np.einsum("kmjc,mj->kmc", basis[:, vertex_indices.astype(np.int64)], weights)
    return sampled, basis, float(np.median(residuals)), edge


def region_slices(names):
    """成分名から領域ごとの列範囲を作る（並びが連続していることも確かめる）。"""
    prefixes = [name.rsplit("_", 1)[0] for name in names]
    bounds = {}
    for index, prefix in enumerate(prefixes):
        low, high = bounds.get(prefix, (index, index))
        bounds[prefix] = (min(low, index), max(high, index))
    for prefix, (low, high) in bounds.items():
        if any(name != prefix for name in prefixes[low : high + 1]):
            raise SystemExit(f"領域 {prefix} の列が連続していない")
    return {prefix: slice(low, high + 1) for prefix, (low, high) in bounds.items()}


def effective_degrees_of_freedom(design, prior_std, noise):
    """リッジ回帰の有効自由度 trace(Gram (Gram + I)^-1)。

    事前分布 N(0, diag(prior_std^2))・観測ノイズ σ の下で、事前分散のうち観測で決まった
    割合の総和。列数が上限（全部決まる）、0 が下限（何も決まらない）。
    """
    scaled = design * prior_std[None, :]
    gram = scaled.T @ scaled / noise**2
    return float(np.trace(gram @ np.linalg.inv(gram + np.eye(gram.shape[0]))))


def solve_ridge(design, observation, prior_std, noise):
    """事前分布込みの MAP 解（Tikhonov。正則化の重みを成分ごとの事前分散から作る）。"""
    scaled = design * prior_std[None, :]
    gram = scaled.T @ scaled + noise**2 * np.eye(scaled.shape[1])
    return prior_std * np.linalg.solve(gram, scaled.T @ observation)


def report_effective_freedom(design_full, prior_std, slices):
    print("  領域          成分  " + "  ".join(f"{s:>6.1f}mm" for s in NOISE_MILLIMETERS[1:]))
    for prefix, columns in slices.items():
        design = design_full[:, columns]
        row = [
            effective_degrees_of_freedom(design, prior_std[columns], sigma / 1000.0)
            for sigma in NOISE_MILLIMETERS[1:]
        ]
        count = columns.stop - columns.start
        label = REGION_OF[prefix]
        print(f"  {label:<12} {count:>5}  " + "  ".join(f"{value:>8.1f}" for value in row))


def report_reconstruction(design_full, full_basis, prior_std, coefficients, visible):
    print("  σ[mm]   全体 中央値   全体 最悪   舌の外 中央値   舌の外 最悪")
    rng = np.random.default_rng(20260908)
    for sigma in NOISE_MILLIMETERS:
        whole, outside = [], []
        for row in range(coefficients.shape[0]):
            truth = coefficients[row]
            observed = design_full @ truth
            if sigma > 0:
                observed = observed + rng.normal(0.0, sigma / 1000.0, observed.shape)
            estimate = solve_ridge(design_full, observed, prior_std, max(sigma, 0.05) / 1000.0)
            difference = np.einsum("k,kvc->vc", estimate - truth, full_basis)
            target = np.einsum("k,kvc->vc", truth, full_basis)
            whole.append(np.linalg.norm(difference) / np.linalg.norm(target))
            outside.append(np.linalg.norm(difference[visible]) / np.linalg.norm(target[visible]))
        whole, outside = np.asarray(whole), np.asarray(outside)
        print(
            f"  {sigma:>5.1f}   {np.median(whole):>10.1%}  {whole.max():>10.1%}"
            f"   {np.median(outside):>13.1%}  {outside.max():>12.1%}"
        )


def report_per_preset(design_full, full_basis, prior_std, coefficients, class_names, visible):
    rng = np.random.default_rng(20260908)
    rows = []
    for row in range(coefficients.shape[0]):
        truth = coefficients[row]
        observed = design_full @ truth + rng.normal(0.0, 1e-3, design_full.shape[0])
        estimate = solve_ridge(design_full, observed, prior_std, 1e-3)
        difference = np.einsum("k,kvc->vc", estimate - truth, full_basis)[visible]
        target = np.einsum("k,kvc->vc", truth, full_basis)[visible]
        rows.append((np.linalg.norm(difference) / np.linalg.norm(target), class_names[row]))
    for error, name in sorted(rows, reverse=True):
        print(f"  {error:>7.1%}  {name}")


def main() -> None:
    exporter = load_exporter()
    with np.load(REPOSITORY_ROOT / "assets" / "gnm" / "gnm_head.npz", allow_pickle=False) as npz:
        sampled, full_basis, residual, edge = build_observation_basis(exporter, npz)
        names = [str(name) for name in npz["expression_names"]]

    presets = np.load(REPOSITORY_ROOT / "tools" / "GnmExpressionPresets_v3_0.npz")
    coefficients = presets["expression_presets"].astype(np.float64)
    class_names = [str(name) for name in presets["class_names"]]

    print(f"密対応の点数 {sampled.shape[1]} / 468（残差 中央値 {residual * 1000:.2f} mm）")
    print(f"メッシュの辺 中央値 {edge * 1000:.2f} mm")

    # 事前分布は公式 20 プリセットの散らばりから取る。0 除算を避けるため下限を置く。
    prior_std = coefficients.std(axis=0)
    prior_std = np.maximum(prior_std, 1e-3 * prior_std.max())

    slices = region_slices(names)
    # 舌は原理的に見えない。舌が動かす頂点を外した誤差も併記して、見える所の当たりを分ける。
    visible = np.abs(full_basis[slices["tongue"]]).max(axis=(0, 2)) <= 1e-6
    print(f"舌が動かす頂点 {int((~visible).sum())} / {full_basis.shape[1]}")

    for kind, axes in (("xy（z を捨てる）", [0, 1]), ("xyz", [0, 1, 2])):
        design_full = sampled[:, :, axes].reshape(sampled.shape[0], -1).T
        print("")
        print(f"=== 観測 {kind}: 有効自由度 / 成分数 ===")
        report_effective_freedom(design_full, prior_std, slices)
        print("")
        print(f"=== 観測 {kind}: 20 プリセットを解き直したときの全頂点変位の誤差 ===")
        report_reconstruction(design_full, full_basis, prior_std, coefficients, visible)
        print("")
        print(f"=== 観測 {kind}: プリセットごと（σ=1.0mm・舌の外）===")
        report_per_preset(design_full, full_basis, prior_std, coefficients, class_names, visible)


if __name__ == "__main__":
    main()
