#!/usr/bin/env python3
"""Create a fail-closed inventory for a one-RQ reproduction run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def rows_in(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    if path.suffix == ".jsonl":
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines()
                   if line.strip())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--rq", required=True)
    args = parser.parse_args()
    root = args.result_dir
    selection_path = root / "cases" / "SINGLE_RQ_SELECTION.json"
    selection = (json.loads(selection_path.read_text(encoding="utf-8"))
                 if selection_path.exists() else {})
    status_path = root / "status.jsonl"
    statuses = ([json.loads(line) for line in status_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
        if status_path.exists() else [])

    artifacts = [
        ("PyTorch/TorchInductor whole graph", root / "raw" / "whole_graph.jsonl"),
        ("PyTorch boundary", root / "raw" / "boundary_layout.jsonl"),
        ("CUDA reference", root / "raw" / "cuda_reference.csv"),
        ("Triton explicit", root / "raw" / "triton_explicit.csv"),
        ("TVM", root / "raw" / "tvm_projected.csv"),
        ("CUTLASS", root / "raw" / "cutlass_attention.csv"),
        ("vLLM serving", root / "raw" / "vllm_native.jsonl"),
        ("SGLang serving", root / "raw" / "sglang_native.jsonl"),
        ("vLLM native subgraph", root / "native_subgraphs" / "raw" / "vllm_subgraph_native.jsonl"),
        ("SGLang native subgraph", root / "native_subgraphs" / "raw" / "sglang_subgraph_native.jsonl"),
        ("vLLM KV batch", root / "kv_request_batch" / "raw" / "vllm_kv_request_batch.jsonl"),
        ("SGLang KV batch", root / "kv_request_batch" / "raw" / "sglang_kv_request_batch.jsonl"),
    ]
    lines = [
        f"# {args.rq} 单 RQ 复现实验清单", "",
        "> 只声明本次实际生成的证据；`skipped_by_rq` 表示该适配器对该 RQ "
        "没有预先定义的可证伪干预，不等于框架失败。", "",
        "## Case 选择", "",
        f"- 原始 real-world contracts：{selection.get('source_case_count', 'unknown')}",
        f"- 适用于 {args.rq} 的 contracts：{selection.get('selected_case_count', 'unknown')}",
        f"- 其中 attention/KV 可执行 contracts：{selection.get('selected_attention_case_count', 'unknown')}",
        "- 选择规则：`RQ in case.target_rqs`；没有用前 N 个 case 代替全量。", "",
        "## 框架/适配器执行状态", "", "| step | status | detail |",
        "|---|---|---|",
    ]
    housekeeping = {"persuasive_case_tests", "native_adapter_tests",
                    "native_subgraph_adapter_tests", "rq_adequacy_tests",
                    "build_cases", "shape_diversity_audit", "gpu_preflight",
                    "select_single_rq_cases", "analyze"}
    for row in statuses:
        if row.get("step") in housekeeping:
            continue
        detail = str(row.get("detail", "")).replace("|", "\\|")
        lines.append(f"| {row.get('step', '')} | {row.get('status', '')} | {detail} |")
    lines += ["", "## 原始结果行数", "", "| artifact | rows | path |",
              "|---|---:|---|"]
    for label, path in artifacts:
        lines.append(f"| {label} | {rows_in(path)} | `{path}` |")
    lines += ["", "## 判读边界", "",
        "- `success` 只表示适配器执行成功；RQ 假设是否成立仍须按该 RQ 的 paired counterfactual、正确性和噪声门槛判读。",
        "- `unavailable`/`failed` 不能当成反例；`skipped_by_rq` 是设计上的 N/A。",
        "- CUDA reference 是共享因果微基准，可能同时输出相邻 RQ 可复用的 counterfactual 行；本报告按所选 RQ 和 manifest 限定证据范围。",
        "- vLLM/SGLang serving 是固定模型端到端支持证据；native-subgraph 才是逐 contract 的算子/子图证据。", ""]
    report = root / "SINGLE_RQ_REPRODUCTION_CN.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rq": args.rq, "report": str(report)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
