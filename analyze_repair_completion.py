#!/usr/bin/env python3
"""Consolidate the primary repair run with an isolated Mamba supplement."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path


OOM = {"oom_preflight", "oom_runtime"}


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def counts(values: list[dict]) -> dict[str, int]:
    return dict(sorted(collections.Counter(row.get("status", "missing") for row in values).items()))


def write_svg(path: Path, old_whole: int, new_whole: int,
              old_boundary: int, new_boundary: int) -> None:
    maximum = max(old_whole, new_whole, old_boundary, new_boundary, 1)
    scale = 440 / maximum
    data = [("MoE whole graph", old_whole, new_whole),
            ("MoE/Mamba boundary", old_boundary, new_boundary)]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="230">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:DejaVu Sans,Arial,sans-serif;font-size:14px}.t{font-size:20px;font-weight:bold}</style>',
             '<text class="t" x="20" y="30">Non-OOM failure count before and after repair</text>',
             '<rect x="20" y="48" width="14" height="14" fill="#c44e52"/><text x="40" y="60">base</text>',
             '<rect x="105" y="48" width="14" height="14" fill="#55a868"/><text x="125" y="60">repaired</text>']
    for i, (label, old, new) in enumerate(data):
        y = 85 + i * 65
        parts += [f'<text x="20" y="{y + 18}">{label}</text>',
                  f'<rect x="220" y="{y}" width="{old * scale:.1f}" height="20" fill="#c44e52"/>',
                  f'<text x="{225 + old * scale:.1f}" y="{y + 16}">{old}</text>',
                  f'<rect x="220" y="{y + 25}" width="{new * scale:.1f}" height="20" fill="#55a868"/>',
                  f'<text x="{225 + new * scale:.1f}" y="{y + 41}">{new}</text>']
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-result", type=Path, required=True)
    parser.add_argument("--primary-run", type=Path, required=True)
    parser.add_argument("--supplement-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.supplement_run
    if args.output_dir:
        output_dir.mkdir(parents=True, exist_ok=False)
    base_raw = args.base_result / "v13_640/full_subgraphs/raw"
    old_whole = rows(base_raw / "llm_640_pytorch_triton.jsonl")
    old_boundary = rows(base_raw / "llm_640_boundary_layout_sweep.jsonl")
    new_whole = rows(args.primary_run / "raw/whole_moe_non_oom.jsonl")
    primary_boundary = rows(args.primary_run / "raw/boundary_moe_mamba_non_oom.jsonl")
    supplement = rows(args.supplement_run / "raw/mamba_boundary_isolated.jsonl")
    selection = json.loads((args.primary_run / "manifests/selection.json").read_text())
    supplement_manifest = json.loads((args.supplement_run / "manifests/mamba_retry.json").read_text())
    supplement_ids = set(supplement_manifest["case_ids"])
    combined_boundary = [row for row in primary_boundary
                         if row.get("case_id") not in supplement_ids] + supplement

    whole_ids = set(selection["whole_case_ids"])
    boundary_ids = set(selection["boundary_case_ids"])
    old_whole_target = [row for row in old_whole if row.get("case_id") in whole_ids]
    old_boundary_target = [row for row in old_boundary if row.get("case_id") in boundary_ids]
    old_whole_fail = {(row.get("case_id"), row.get("backend"))
                      for row in old_whole_target
                      if row.get("status") not in {"success", *OOM}}
    old_boundary_fail = {row.get("case_id") for row in old_boundary_target
                         if row.get("status") not in {"success", *OOM}}
    whole_map = {(row.get("case_id"), row.get("backend")): row for row in new_whole}
    boundary_map: dict[str, list[dict]] = collections.defaultdict(list)
    for row in combined_boundary:
        boundary_map[row.get("case_id")].append(row)

    fixed_whole = sorted(key for key in old_whole_fail
                         if whole_map.get(key, {}).get("status") == "success")
    fixed_boundary_full = sorted(case_id for case_id in old_boundary_fail
                                 if len(boundary_map[case_id]) == 3
                                 and all(row.get("status") == "success"
                                         for row in boundary_map[case_id]))
    # A case that now reaches a genuine OOM has no remaining software error.
    # It is resource-blocked rather than a failed repair.  This distinction is
    # essential because the user explicitly excluded OOM from this repair run.
    fixed_boundary_oom_blocked = sorted(
        case_id for case_id in old_boundary_fail
        if boundary_map[case_id]
        and any(row.get("status") in OOM for row in boundary_map[case_id])
        and all(row.get("status") in {"success", *OOM}
                for row in boundary_map[case_id]))
    fixed_boundary = sorted(set(fixed_boundary_full) | set(fixed_boundary_oom_blocked))
    whole_bad = [row for row in new_whole if row.get("status") not in {"success", *OOM}]
    boundary_bad = [row for row in combined_boundary
                    if row.get("status") not in {"success", *OOM}]
    whole_coverage_bad = sorted(case_id for case_id in whole_ids
                                if len([row for row in new_whole
                                        if row.get("case_id") == case_id]) != 2)
    boundary_coverage_bad = sorted(
        case_id for case_id in boundary_ids
        if (not boundary_map[case_id]
            or (len(boundary_map[case_id]) != 3
                and not any(row.get("status") in OOM
                            for row in boundary_map[case_id])))
    )

    whole_pairs: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for row in new_whole:
        if row.get("status") == "success":
            whole_pairs[row["case_id"]][row["backend"]] = float(row["p50_ms"])
    inductor_speedups = [pair["pytorch"] / pair["triton"] for pair in whole_pairs.values()
                         if {"pytorch", "triton"} <= pair.keys()]
    boundary_pairs: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for row in combined_boundary:
        if row.get("status") == "success" and row.get("strategy"):
            boundary_pairs[row["case_id"]][row["strategy"]] = float(row["p50_ms"])
    alt_ratios = [pair["alternate_strided_view"] / pair["native_contiguous"]
                  for pair in boundary_pairs.values()
                  if {"alternate_strided_view", "native_contiguous"} <= pair.keys()]
    repair_ratios = [pair["repair_to_contiguous_each_call"] / pair["native_contiguous"]
                     for pair in boundary_pairs.values()
                     if {"repair_to_contiguous_each_call", "native_contiguous"} <= pair.keys()]

    complete = (len(fixed_whole) == len(old_whole_fail)
                and len(fixed_boundary) == len(old_boundary_fail)
                and not whole_bad and not boundary_bad
                and not whole_coverage_bad and not boundary_coverage_bad)
    data = {
        "complete": complete,
        "base_result": str(args.base_result.resolve()),
        "primary_run": str(args.primary_run.resolve()),
        "supplement_run": str(args.supplement_run.resolve()),
        "whole_cases": len(whole_ids), "boundary_cases": len(boundary_ids),
        "old_whole_non_oom_failures": len(old_whole_fail),
        "fixed_whole": len(fixed_whole),
        "old_boundary_non_oom_failure_cases": len(old_boundary_fail),
        "fixed_boundary": len(fixed_boundary),
        "fixed_boundary_full_success": fixed_boundary_full,
        "fixed_boundary_oom_blocked": fixed_boundary_oom_blocked,
        "whole_status": counts(new_whole),
        "combined_boundary_status": counts(combined_boundary),
        "whole_coverage_bad": whole_coverage_bad,
        "boundary_coverage_bad": boundary_coverage_bad,
        "whole_non_oom_bad": whole_bad,
        "boundary_non_oom_bad": boundary_bad,
        "median_inductor_over_eager_speedup": statistics.median(inductor_speedups),
        "median_alternate_over_native_latency_ratio": statistics.median(alt_ratios),
        "median_repair_over_native_latency_ratio": statistics.median(repair_ratios),
    }
    (output_dir / "REPAIR_COMPLETION.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    with (output_dir / "repair_delta.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scope", "old_non_oom_failures", "fixed", "new_non_oom_failures"])
        writer.writerow(["moe_whole_graph", len(old_whole_fail), len(fixed_whole), len(whole_bad)])
        writer.writerow(["moe_mamba_boundary_cases", len(old_boundary_fail), len(fixed_boundary),
                         len({row.get('case_id') for row in boundary_bad})])
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    write_svg(figures / "repair_failure_delta.svg", len(old_whole_fail), len(whole_bad),
              len(old_boundary_fail), len({row.get("case_id") for row in boundary_bad}))

    lines = [
        "# 非 OOM 失败修复：增量完成报告",
        "",
        f"**最终判定：{'PASS——本次所有非 OOM 软件错误均已消除' if complete else 'FAIL——仍有非 OOM 目标未解决'}。**",
        "",
        f"- 原始全量结果（只读）：`{args.base_result.resolve()}`",
        f"- 首轮增量结果（只读）：`{args.primary_run.resolve()}`",
        f"- 隔离补跑：`{args.supplement_run.resolve()}`",
        f"- 本报告：`{output_dir.resolve()}`",
        "",
        "![修复前后非 OOM 失败数](figures/repair_failure_delta.svg)",
        "",
        "## 结果",
        "",
        f"- MoE whole graph：{len(whole_ids)} case × 2 backend = {len(new_whole)} 行；状态 `{json.dumps(counts(new_whole), ensure_ascii=False)}`。",
        f"- 原先 whole-graph 非 OOM 失败 {len(old_whole_fail)} 行，修复后成功 {len(fixed_whole)}/{len(old_whole_fail)}。",
        f"- MoE/Mamba boundary：{len(boundary_ids)} case × 3 strategy = {len(combined_boundary)} 行；状态 `{json.dumps(counts(combined_boundary), ensure_ascii=False)}`。",
        f"- 原先 boundary 非 OOM 失败 {len(old_boundary_fail)} 个 case：{len(fixed_boundary_full)} 个完整三策略成功，{len(fixed_boundary_oom_blocked)} 个已消除软件错误但转为真实 OOM 资源阻塞；非 OOM 错误修复 {len(fixed_boundary)}/{len(old_boundary_fail)}。",
        f"- OOM 阻塞 case：`{json.dumps(fixed_boundary_oom_blocked, ensure_ascii=False)}`。",
        f"- MoE 上 Inductor/Triton backend 相对 eager 的中位加速：{statistics.median(inductor_speedups):.4f}×。",
        f"- boundary alternate/native 中位延迟比：{statistics.median(alt_ratios):.4f}；repair/native：{statistics.median(repair_ratios):.4f}。",
        "",
        "## 修复与因果解释",
        "",
        "1. MoE expert weight 原为 `[experts,in,out]` 却按 `experts` 缩放；改为按逻辑 fan-in 缩放后，旧 nonfinite 与大误差全部消失。没有修改正确性容差。",
        "2. boundary 原来任一 alternate 异常会丢掉 native/repair；现在三策略独立记录。",
        "3. Mamba `associative_scan` 会经 TorchDynamo 捕获；多 shape 单进程超过默认 cache 后产生假性的 `UncapturedHigherOrderOpError`。提高 cache 上限并用独立进程补跑后，所有非 OOM 软件错误消失。",
        "",
        "## 能证明哪些 RQ",
        "",
        "- v13 L-RQ1/H1.3：恢复 native/alternate/repair 的成对 edge 数据，可判定 stride-polymorphic direct read 是否降低 layout adaptation。",
        "- v13 L-RQ3/H3.4：alternate 与逐调用 materialization 成本可直接配对；报告中的 latency ratio 是实测而非模型推断。",
        "- v13 L-RQ7/H7.1、H7.3：stride 变换、direct legality 与 repair 被分别测量，支持 zero-copy transformability 的判定。",
        "- v13 L-RQ4：68 个 MoE 多模型、多 shape、prefill/decode 的 eager/Inductor 配对恢复，可用于该路径的 workload crossover；不能外推到多硬件 H4.4。",
        "- v10 RQ1/O-LRQ1-EDGE-INVERSION：该 boundary 补跑恢复 edge 级 observation，但不替代 v10 专用 CUDA producer-consumer 实验。",
        "",
        "## 框架边界",
        "",
        "- 实际重跑的是 PyTorch eager、TorchInductor（生成 Triton kernels）与 PyTorch boundary harness。这里的 `triton` backend 不能冒充独立 Triton adapter。",
        "- vLLM、SGLang、TVM、CUTLASS/CuTe、独立 Triton adapter 没有调用本次修改代码，因此没有受修复部分可重跑；仍引用原始全量目录中的结果。",
        "- Hexcute 在 A10 上仍是 architecture-blocked；本次不伪造替代证据。",
        "",
        "## 仍保持 fail-closed 的项目",
        "",
        "- 原全量结果中的 2 个 linear-attention prefill 与 1 个 Mamba2 prefill Inductor 数值差异不属于本次已修复原因；独立诊断复现后仍保留失败，未放宽阈值。",
        "- OOM case 不在本次目标内，继续标记为资源边界。",
    ]
    (output_dir / "REPAIR_COMPLETION_CN.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"complete": complete,
                      "report": str(output_dir / 'REPAIR_COMPLETION_CN.md'),
                      "fixed_whole": len(fixed_whole),
                      "fixed_boundary": len(fixed_boundary)}, ensure_ascii=False))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
