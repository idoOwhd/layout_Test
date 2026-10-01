#!/usr/bin/env python3
"""Infrastructure and provenance gate for the RQ adequacy run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    root = args.result_dir
    statuses = jsonl(root / "status.jsonl")
    latest = {row["step"]: row for row in statuses}
    required = [
        "persuasive_case_tests", "native_adapter_tests", "native_subgraph_adapter_tests",
        "rq_adequacy_tests", "build_cases", "shape_diversity_audit", "gpu_preflight",
        "whole_graph_pytorch_torchinductor", "pytorch_boundary_counterfactual",
        "build_cuda_reference", "cuda_reference", "explicit_triton",
        "tvm_projected_boundaries", "cutlass_attention_boundary",
        "vllm_sglang_native_subgraph_completion", "vllm_sglang_kv_request_batch",
        "analyze", "analyze_rq_adequacy", "audit_attention_kv_multiconsumer",
        "sglang_auto_nhd", "sglang_auto_hnd", "sglang_flashinfer_p1",
        "sglang_flashinfer_p16", "vllm_LBNHC", "vllm_LBHNC",
        "vllm_block16", "vllm_block32",
    ]
    step_checks = {name: latest.get(name, {}).get("status") == "success" for name in required}
    manifest_path = root / "cases" / "executed_manifest.json"
    cases = []
    if manifest_path.is_file():
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        cases = value.get("cases", value) if isinstance(value, dict) else value
    expected_cases = 32 if args.mode == "smoke" else 1024
    provenance_ok = bool(cases) and all(
        case.get("parent_case_id") and case.get("model_id") and case.get("model_revision")
        and str(case.get("source_url", "")).startswith("https://")
        and case.get("shape_counterfactual") is True
        for case in cases)
    cuda_rows = []
    cuda_path = root / "raw" / "cuda_reference.csv"
    if cuda_path.is_file():
        with cuda_path.open(newline="", encoding="utf-8") as handle:
            cuda_rows = list(csv.DictReader(handle))
    process_repetitions = {row.get("process_repetition") for row in cuda_rows}
    boundary_rows = jsonl(root / "raw" / "boundary_layout.jsonl")
    extended_rows = [row for row in boundary_rows
                     if row.get("experiment") == "boundary_reuse_counterfactual"]
    minimum_process_repetitions = 2 if args.mode == "smoke" else 5
    artifact_checks = {
        "executed_case_count": len(cases) == expected_cases,
        "all_cases_from_real_world_parents": provenance_ok,
        "cuda_independent_process_repetitions": (
            len(process_repetitions) >= minimum_process_repetitions),
        "cuda_weighted_fanout_present": any(
            row.get("benchmark") == "weighted_multi_consumer_pipeline" for row in cuda_rows),
        "cuda_state_trace_present": any(
            row.get("benchmark") == "state_migration_trace" for row in cuda_rows),
        "cuda_nondivisible_page_present": any(
            int(row.get("allocated_tokens", 0) or 0) > int(row.get("tokens", 0) or 0)
            for row in cuda_rows),
        "whole_subgraph_reuse_present": bool(extended_rows),
        "adequacy_report_present": (root / "RQ_ADEQUACY_ANALYSIS_CN.md").is_file(),
        "attention_kv_multiconsumer_audit_passed": (
            (root / "ATTENTION_KV_MULTICONSUMER_AUDIT.json").is_file()
            and json.loads((root / "ATTENTION_KV_MULTICONSUMER_AUDIT.json").read_text(
                encoding="utf-8")).get("complete") is True),
    }
    complete = all(step_checks.values()) and all(artifact_checks.values())
    payload = {"schema_version": 1, "mode": args.mode, "complete": complete,
               "required_steps": step_checks, "artifact_checks": artifact_checks,
               "guardrail": "Completion is infrastructure completion, not universal support of every RQ."}
    (root / "RQ_ADEQUACY_COMPLETENESS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# RQ adequacy 运行完整性", "", f"- mode: `{args.mode}`",
             f"- 结论：**{'COMPLETE' if complete else 'INCOMPLETE'}**", "",
             "## 必需步骤", "", "| 步骤 | 结果 |", "|---|---|"]
    lines += [f"| {name} | {'PASS' if passed else 'FAIL'} |"
              for name, passed in step_checks.items()]
    lines += ["", "## 产物约束", "", "| 检查 | 结果 |", "|---|---|"]
    lines += [f"| {name} | {'PASS' if passed else 'FAIL'} |"
              for name, passed in artifact_checks.items()]
    lines += ["", "> COMPLETE 只表示预注册入口和产物齐全；各 RQ 的科学结论以 `RQ_ADEQUACY_ANALYSIS_CN.md` 为准。", ""]
    (root / "RQ_ADEQUACY_COMPLETENESS_CN.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"complete": complete,
                      "report": str(root / "RQ_ADEQUACY_COMPLETENESS_CN.md")},
                     ensure_ascii=False))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
