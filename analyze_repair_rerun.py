#!/usr/bin/env python3
"""Create a fail-closed old-to-new report for the incremental repair run."""

from __future__ import annotations

import argparse
import collections
import html
import json
from pathlib import Path


OOM = {"oom_preflight", "oom_runtime"}
SUCCESS = "success"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def status_counts(rows: list[dict]) -> dict[str, int]:
    return dict(sorted(collections.Counter(row.get("status", "missing") for row in rows).items()))


def svg_bars(path: Path, groups: list[tuple[str, int, int, int]]) -> None:
    width, row_h, left, right = 940, 44, 250, 100
    max_value = max((old + fixed + remaining for _, old, fixed, remaining in groups), default=1)
    height = 90 + row_h * len(groups)
    scale = (width - left - right) / max(max_value, 1)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:DejaVu Sans,Arial,sans-serif;font-size:14px}.title{font-size:20px;font-weight:bold}</style>',
             '<text class="title" x="20" y="30">Non-OOM failures: base run and repaired rerun</text>',
             '<rect x="20" y="48" width="14" height="14" fill="#c44e52"/><text x="40" y="60">base failures</text>',
             '<rect x="170" y="48" width="14" height="14" fill="#55a868"/><text x="190" y="60">fixed to success</text>',
             '<rect x="350" y="48" width="14" height="14" fill="#dd8452"/><text x="370" y="60">remaining/new non-success</text>']
    for index, (label, old, fixed, remaining) in enumerate(groups):
        y = 80 + index * row_h
        parts.append(f'<text x="20" y="{y + 18}">{html.escape(label)}</text>')
        x = left
        for value, color in ((old, "#c44e52"), (fixed, "#55a868"), (remaining, "#dd8452")):
            bar = value * scale
            parts.append(f'<rect x="{x:.1f}" y="{y}" width="{bar:.1f}" height="24" fill="{color}"/>')
            if value:
                parts.append(f'<text x="{x + bar + 4:.1f}" y="{y + 18}">{value}</text>')
            x += bar + 34
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-result", type=Path, required=True)
    parser.add_argument("--rerun-dir", type=Path, required=True)
    args = parser.parse_args()
    base_raw = args.base_result / "v13_640/full_subgraphs/raw"
    selection = json.loads((args.rerun_dir / "manifests/selection.json").read_text(encoding="utf-8"))
    old_whole = read_jsonl(base_raw / "llm_640_pytorch_triton.jsonl")
    old_boundary = read_jsonl(base_raw / "llm_640_boundary_layout_sweep.jsonl")
    new_whole = read_jsonl(args.rerun_dir / "raw/whole_moe_non_oom.jsonl")
    new_boundary = read_jsonl(args.rerun_dir / "raw/boundary_moe_mamba_non_oom.jsonl")

    whole_ids = set(selection["whole_case_ids"])
    boundary_ids = set(selection["boundary_case_ids"])
    old_whole_target = [r for r in old_whole if r.get("case_id") in whole_ids]
    old_boundary_target = [r for r in old_boundary if r.get("case_id") in boundary_ids]

    old_whole_fail_keys = {
        (r.get("case_id"), r.get("backend")) for r in old_whole_target
        if r.get("status") not in {SUCCESS, *OOM}
    }
    new_whole_by_key = {(r.get("case_id"), r.get("backend")): r for r in new_whole}
    fixed_whole = sorted(key for key in old_whole_fail_keys
                         if new_whole_by_key.get(key, {}).get("status") == SUCCESS)
    unresolved_whole = sorted(key for key in old_whole_fail_keys
                              if new_whole_by_key.get(key, {}).get("status") != SUCCESS)

    old_boundary_fail_ids = {
        r.get("case_id") for r in old_boundary_target
        if r.get("status") not in {SUCCESS, *OOM}
    }
    new_boundary_by_case: dict[str, list[dict]] = collections.defaultdict(list)
    for row in new_boundary:
        new_boundary_by_case[row.get("case_id")].append(row)
    fixed_boundary = sorted(
        case_id for case_id in old_boundary_fail_ids
        if len(new_boundary_by_case.get(case_id, [])) == 3
        and all(row.get("status") == SUCCESS for row in new_boundary_by_case[case_id])
    )
    unresolved_boundary = sorted(
        case_id for case_id in old_boundary_fail_ids if case_id not in fixed_boundary
    )
    remaining_new_whole = [r for r in new_whole if r.get("status") not in {SUCCESS, *OOM}]
    remaining_new_boundary = [r for r in new_boundary if r.get("status") not in {SUCCESS, *OOM}]

    report_data = {
        "base_result": str(args.base_result.resolve()),
        "rerun_dir": str(args.rerun_dir.resolve()),
        "selection": selection,
        "old_whole_status": status_counts(old_whole_target),
        "new_whole_status": status_counts(new_whole),
        "old_boundary_status": status_counts(old_boundary_target),
        "new_boundary_status": status_counts(new_boundary),
        "old_whole_non_oom_failures": len(old_whole_fail_keys),
        "fixed_whole_failures": [list(key) for key in fixed_whole],
        "unresolved_whole_failures": [list(key) for key in unresolved_whole],
        "old_boundary_non_oom_failure_cases": len(old_boundary_fail_ids),
        "fixed_boundary_failure_cases": fixed_boundary,
        "unresolved_boundary_failure_cases": unresolved_boundary,
        "new_whole_non_oom_failures": remaining_new_whole,
        "new_boundary_non_oom_failures": remaining_new_boundary,
    }
    (args.rerun_dir / "REPAIR_RERUN_ANALYSIS.json").write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    figures = args.rerun_dir / "figures"
    figures.mkdir(exist_ok=True)
    svg_bars(figures / "repair_failure_delta.svg", [
        ("MoE whole graph", len(old_whole_fail_keys), len(fixed_whole), len(remaining_new_whole)),
        ("MoE/Mamba boundary cases", len(old_boundary_fail_ids), len(fixed_boundary),
         len({r.get("case_id") for r in remaining_new_boundary})),
    ])

    complete = not unresolved_whole and not unresolved_boundary and not remaining_new_whole and not remaining_new_boundary
    lines = [
        "# 非 OOM 修复增量重跑报告",
        "",
        f"- 旧结果（只读基线）：`{args.base_result.resolve()}`",
        f"- 新结果（未覆盖旧结果）：`{args.rerun_dir.resolve()}`",
        f"- 判定：**{'所有目标失败均已修复' if complete else '仍有目标需要审计'}**",
        "",
        "![失败数变化](figures/repair_failure_delta.svg)",
        "",
        "## 选择范围",
        "",
        f"- whole graph：{selection['whole_case_count']} 个旧基线中非 OOM 的 MoE case；每个 case 分别执行 PyTorch eager 与 TorchInductor/Triton backend。",
        f"- boundary：{selection['boundary_case_count']} 个旧基线中非 OOM 的 MoE/Mamba2 case；每个 case 必须各有 native、alternate、repair 三条独立结果。",
        "- OOM case 明确排除；本次不把 OOM 当作代码修复目标。",
        "",
        "## 旧→新结果",
        "",
        f"- whole graph 旧非 OOM 失败：{len(old_whole_fail_keys)}；已转为 success：{len(fixed_whole)}；旧失败仍未解决：{len(unresolved_whole)}。",
        f"- boundary 旧非 OOM 失败 case：{len(old_boundary_fail_ids)}；三策略均成功：{len(fixed_boundary)}；仍未解决：{len(unresolved_boundary)}。",
        f"- 新运行额外非 OOM 失败行：whole={len(remaining_new_whole)}，boundary={len(remaining_new_boundary)}。",
        "",
        "### 状态分布",
        "",
        f"- whole old: `{json.dumps(status_counts(old_whole_target), ensure_ascii=False)}`",
        f"- whole new: `{json.dumps(status_counts(new_whole), ensure_ascii=False)}`",
        f"- boundary old: `{json.dumps(status_counts(old_boundary_target), ensure_ascii=False)}`",
        f"- boundary new: `{json.dumps(status_counts(new_boundary), ensure_ascii=False)}`",
        "",
        "## 对 RQ 的证据含义",
        "",
        "- v13 L-RQ1/H1.3：alternate stride 与 native/repair 的三方配对不再因单条路径异常而整例丢失，可检验 stride-polymorphic direct read 是否减少 edge regret。",
        "- v13 L-RQ3/H3.4：显式 repair 的逐调用拷贝成本与 zero-copy alternate 分开计时，可检验 materialization 是否值得。",
        "- v13 L-RQ7/H7.1、H7.3：实际 stride、合法 direct consumption、unsupported 和 repair 成为不同状态，不再把“不支持”冒充数值/系统错误。",
        "- v13 L-RQ4：修复后的 MoE 多 shape/phase 数据可用于 PyTorch 与 TorchInductor 路径的 workload crossover；但它本身不证明跨硬件 H4.4。",
        "- v10 RQ1/O-LRQ1-EDGE-INVERSION：boundary 三策略配对恢复了 edge-level 观测；本增量实验不替代 v10 CUDA producer/consumer 核心实验。",
        "",
        "## 框架范围（防止过度宣称）",
        "",
        "- 本次实际重跑：PyTorch eager、PyTorch TorchInductor（其生成后端为 Triton）、PyTorch boundary harness。",
        "- 未重跑且沿用旧目录证据：显式 Triton adapter、TVM、CUTLASS/CuTe、vLLM、SGLang；它们不调用本次修改的 `real_world_workloads.py`/`llm_boundary_layout_sweep.py` 路径。",
        "- Hexcute 在 A10 上仍为 architecture-blocked，不以其他框架结果替代。",
        "- 因此本报告证明的是“受修复代码路径上的 RQ 证据恢复”，不是重新声称所有框架的所有 RQ 都由本次增量运行覆盖。",
        "",
        "## 未纳入修复的已知问题",
        "",
        "- 两个 linear-attention prefill 与一个 Mamba2 prefill 的 Inductor 数值差异已在独立 smoke 中复现；没有高精度独立 oracle 前保持 fail-closed，不放宽阈值。",
        "- OOM 仍按资源边界记录，不算实现错误。",
    ]
    if unresolved_whole:
        lines += ["", "### whole graph 未解决键", "", f"`{json.dumps(unresolved_whole, ensure_ascii=False)}`"]
    if unresolved_boundary:
        lines += ["", "### boundary 未解决 case", "", f"`{json.dumps(unresolved_boundary, ensure_ascii=False)}`"]
    (args.rerun_dir / "REPAIR_RERUN_ANALYSIS_CN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"complete": complete, "report": str(args.rerun_dir / 'REPAIR_RERUN_ANALYSIS_CN.md'),
                      "fixed_whole": len(fixed_whole), "fixed_boundary": len(fixed_boundary),
                      "remaining_whole": len(remaining_new_whole),
                      "remaining_boundary": len(remaining_new_boundary)}, ensure_ascii=False))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
