#!/usr/bin/env python3
"""Fail-closed adequacy analysis for the v10/v13 RQ supplement.

This analyzer does not equate row count with scientific support.  It checks
whether each RQ has the required intervention, counterfactual candidate set,
correctness gate and independent real-model/shape strata.  Missing or modeled
evidence remains partial/blocked.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    return value.get("cases", value) if isinstance(value, dict) else value


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def valid(row: dict[str, str]) -> bool:
    try:
        value = float(row.get("max_abs_error", "nan"))
        return math.isfinite(value) and value <= 1e-3 and float(row["p50_ms"]) > 0
    except (KeyError, ValueError):
        return False


def winners(rows: list[dict[str, str]], group_fields: tuple[str, ...]) -> dict[tuple, str]:
    candidates: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        if valid(row):
            key = tuple(row.get(field, "") for field in group_fields)
            candidates[key + (row["strategy"],)].append(float(row["p50_ms"]))
    medians: dict[tuple, list[tuple[str, float]]] = defaultdict(list)
    for key, values in candidates.items():
        ordered = sorted(values)
        count = len(ordered)
        median = (ordered[count // 2] if count % 2
                  else (ordered[count // 2 - 1] + ordered[count // 2]) / 2)
        medians[key[:-1]].append((key[-1], median))
    return {key: min(values, key=lambda item: item[1])[0]
            for key, values in medians.items() if values}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    root = args.result_dir
    manifest_path = args.manifest or root / "cases" / "executed_manifest.json"
    cases = load_manifest(manifest_path)
    cuda = read_csv(root / "raw" / "cuda_reference.csv")
    triton = read_csv(root / "raw" / "triton_explicit.csv")
    tvm = read_csv(root / "raw" / "tvm_projected.csv")
    cutlass = read_csv(root / "raw" / "cutlass_attention.csv")
    boundary = read_jsonl(root / "raw" / "boundary_layout.jsonl")
    status_rows = []
    if (root / "status.jsonl").is_file():
        status_rows = [json.loads(line) for line in (root / "status.jsonl").read_text().splitlines()
                       if line.strip()]
    latest_status = {row["step"]: row["status"] for row in status_rows}

    structures = Counter(case.get("structure") for case in cases)
    parents = defaultdict(set)
    batches = set()
    for case in cases:
        parents[case.get("structure")].add(case.get("parent_case_id"))
        batches.add(int(case.get("shape", {}).get("batch", 1)))
    direct = [row for row in cuda if valid(row)]
    by_benchmark = Counter(row.get("benchmark") for row in direct)

    # RQ1: both primitive consumers and complete producer+consumer candidates.
    rq1_cases = {row["case_id"] for row in direct
                 if row.get("benchmark") in {"kv_write", "decode_head_scan", "token_major_scan"}}
    rq1_layouts = {row.get("layout") for row in direct
                   if row.get("benchmark") in {"kv_write", "decode_head_scan", "token_major_scan"}}

    # RQ2: direct fanout factorial, not the historical token-only equal-vote proxy.
    weighted = [row for row in direct if row.get("benchmark") == "weighted_multi_consumer_pipeline"]
    fanouts = {(int(row.get("fanout_head", 0)), int(row.get("fanout_token", 0)))
               for row in weighted}
    rq2_winners = winners(weighted, ("case_id", "fanout_head", "fanout_token"))
    winner_by_case = defaultdict(set)
    for (case_id, _head, _token), winner in rq2_winners.items():
        winner_by_case[case_id].add(winner)
    rq2_context_changes = sum(len(value) > 1 for value in winner_by_case.values())

    # RQ3: dense measured reuse and direct contention rows.
    reuse = [row for row in direct if row.get("benchmark") == "conversion_reuse_pipeline"]
    reuse_levels = sorted({int(row.get("reuse_count", 0)) for row in reuse})
    reuse_winners = winners(reuse, ("case_id", "reuse_count"))
    reuse_by_case = defaultdict(list)
    for (case_id, count), winner in reuse_winners.items():
        reuse_by_case[case_id].append((int(count), winner))
    reuse_crossovers = sum(len({winner for _, winner in values}) > 1
                           for values in reuse_by_case.values())
    contention_cases = {row["case_id"] for row in direct
                        if row.get("benchmark") == "decode_under_hbm_contention"}
    boundary_reuse = [row for row in boundary
                      if row.get("experiment") == "boundary_reuse_counterfactual"
                      and row.get("status") == "success" and row.get("correctness_passed") is True]
    boundary_reuse_structures = {row.get("subgraph") for row in boundary_reuse}
    boundary_reuse_levels = {int(row.get("reuse_count", 0)) for row in boundary_reuse}

    # RQ6: direct policy execution over multiple trace regimes.
    state = [row for row in direct if row.get("benchmark") == "state_migration_trace"]
    trace_kinds = {row.get("trace_kind") for row in state}
    policies = {row.get("strategy") for row in state}
    state_winners = winners(state, ("case_id", "trace_kind"))
    state_winner_set = set(state_winners.values())

    # RQ8/RQ10: real metadata layouts, table permutations and non-divisible pages.
    metadata = [row for row in direct if row.get("benchmark") == "data_metadata_scan"]
    metadata_strategies = {row.get("strategy") for row in metadata}
    paged = [row for row in direct
             if row.get("benchmark") in {"paged_decode_head", "sparse_paged_decode_head"}]
    table_orders = {row.get("trace_kind") for row in paged}
    page_sizes = {int(row.get("page_size", 0)) for row in paged}
    fragmented = [row for row in paged
                  if int(row.get("allocated_tokens", 0) or 0) > int(row.get("tokens", 0) or 0)]
    # Rows include process repetitions, route sparsities and block-table
    # orders.  They are measurements, not independent fragmentation cases.
    fragmented_cases = {row.get("case_id") for row in fragmented}
    fragmented_case_pages = {
        (row.get("case_id"), int(row.get("page_size", 0) or 0))
        for row in fragmented
    }

    # RQ7/RQ9 are also checked against their dedicated output paths.  Runtime
    # rows alone do not prove a candidate search is complete.
    boundary_exists = bool(boundary)
    triton_layouts = {row.get("layout") for row in triton}
    # For valid-token accesses paged_NHD has the same affine address map as
    # NHD; its padded allocation extent belongs to RQ10, not a fourth RQ9 map.
    triton_address_maps = {
        "NHD_equivalent" if layout in {"NHD", "paged_NHD"} else layout
        for layout in triton_layouts
    }

    minimum_cases = 4 if len(cases) <= 32 else 64
    assessments = {
        "L-RQ1": {
            "status": "adequate_attention_scope" if len(rq1_cases) >= minimum_cases and {"NHD", "HND"} <= rq1_layouts else "partial",
            "measured": {"direct_cases": len(rq1_cases), "layouts": sorted(rq1_layouts)},
            "remaining": "Native framework rows are supporting slices; only a framework-owned complete pipeline can establish its end-to-end optimum.",
        },
        "L-RQ2": {
            "status": "adequate_attention_scope" if len(fanouts) >= 7 and rq2_context_changes > 0 else "partial",
            "measured": {"fanout_pairs": sorted(fanouts), "cases_with_winner_change": rq2_context_changes},
            "remaining": "A zero winner-change result is valid evidence only after all seven direct fanout pairs complete; token-count formula proxies do not count.",
        },
        "L-RQ3": {
            "status": ("adequate_single_gpu_scope"
                       if len(reuse_levels) >= 12 and contention_cases and reuse_crossovers > 0
                       and len(boundary_reuse_structures) >= 6
                       and {1, 4, 16} <= boundary_reuse_levels else "partial"),
            "measured": {"reuse_levels": reuse_levels, "reuse_crossover_cases": reuse_crossovers,
                         "direct_contention_cases": len(contention_cases),
                         "whole_subgraph_reuse_structures": sorted(boundary_reuse_structures),
                         "whole_subgraph_reuse_levels": sorted(boundary_reuse_levels)},
            "remaining": "Single-GPU HBM contention is direct local evidence, not communication overlap.",
        },
        "L-RQ4": {
            "status": "adequate_single_gpu_scope" if len(cases) >= 1024 and len(structures) == 8 and batches == {1, 2, 4, 8} else "partial",
            "measured": {"cases": len(cases), "structures": dict(structures),
                         "real_parent_counts": {key: len(value) for key, value in parents.items()},
                         "batches": sorted(batches)},
            "remaining": "A later run should use group-held-out splits by parent model; random row splits leak nearly identical shapes.",
        },
        "L-RQ5": {
            "status": "blocked_single_gpu",
            "measured": {},
            "remaining": "Requires a real multi-rank communication intervention; memcpy/HBM pressure is not accepted as validation.",
        },
        "L-RQ6": {
            "status": "adequate_attention_operator_scope" if len(trace_kinds) >= 6 and len(policies) >= 5 and len(state_winners) >= minimum_cases * 6 else "partial",
            "measured": {"trace_kinds": sorted(trace_kinds), "policies": sorted(policies),
                         "policy_trace_pairs": len(state_winners), "winning_policies": sorted(state_winner_set)},
            "remaining": "This is a directly timed operator-state controller; vLLM/SGLang engine-wide live KV migration still lacks a stable public intervention API.",
        },
        "L-RQ7": {
            "status": "adequate_single_consumer_scope" if boundary_exists and len(cases) >= 1024 else "partial",
            "measured": {"boundary_output_present": boundary_exists, "cases": len(cases)},
            "remaining": "Require correctness-gated two/three-consumer intersections; a hard-coded legality intersection does not count.",
        },
        "L-RQ8": {
            "status": "adequate_scale_page_scope" if len(metadata_strategies) >= 8 and len(table_orders) >= 3 else "partial",
            "measured": {"data_metadata_candidates": len(metadata_strategies),
                         "block_table_orders": sorted(table_orders)},
            "remaining": "Scale and page-table metadata are direct; MoE route IDs, sparse indices and zero-points need their own native subgraphs before generalizing.",
        },
        "L-RQ9": {
            "status": "adequate_exposed_candidate_scope" if len(triton_address_maps) >= 3 and latest_status.get("explicit_triton") == "success" else "partial",
            "measured": {"triton_layouts": sorted(triton_layouts),
                         "canonical_address_maps": sorted(triton_address_maps),
                         "triton_step": latest_status.get("explicit_triton", "missing")},
            "remaining": "Candidate maps must be canonicalized independent of tensor extents; equal flat address maps count once.",
        },
        "L-RQ10": {
            "status": "adequate_kernel_page_scope" if len(page_sizes) >= 8 and len(table_orders) >= 3 and fragmented else "partial",
            "measured": {"page_sizes": sorted(page_sizes), "block_table_orders": sorted(table_orders),
                         "fragmented_measurement_rows": len(fragmented),
                         "fragmented_unique_cases": len(fragmented_cases),
                         "fragmented_unique_case_page_contracts": len(fragmented_case_pages)},
            "remaining": "Direct kernel/page geometry is covered; serving-level prefix sharing still requires controlled hit-rate/fanout/concurrency traces.",
        },
    }

    framework_artifacts = {
        "CUDA-reference": len(cuda), "Triton": len(triton), "TVM": len(tvm),
        "CUTLASS/CuTe": len(cutlass),
        "vLLM-native-step": latest_status.get("vllm_sglang_native_subgraph_completion", "missing"),
        "SGLang-native-step": latest_status.get("vllm_sglang_native_subgraph_completion", "missing"),
    }
    payload = {
        "schema_version": 1,
        "guardrail": "adequate means the redesigned single-GPU intervention completed; it does not imply every framework owns that RQ",
        "manifest": str(manifest_path),
        "framework_artifacts": framework_artifacts,
        "benchmark_rows": dict(by_benchmark),
        "rqs": assessments,
    }
    (root / "RQ_ADEQUACY_ANALYSIS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# v10/v13 RQ 实验充分性补充审计", "",
             "> `adequate` 只表示本次重新设计的单卡干预充分完成；不表示每个框架都拥有该层决策，也不把代理模型当成直接测量。", "",
             "## 框架产物", "", "| 框架/入口 | 行数或状态 |", "|---|---:|"]
    lines += [f"| {key} | {value} |" for key, value in framework_artifacts.items()]
    lines += ["", "## 每个 RQ", "", "| RQ | 充分性 | 直接观察 | 尚不能声称 |", "|---|---|---|---|"]
    for rq, item in assessments.items():
        measured = json.dumps(item["measured"], ensure_ascii=False).replace("|", "\\|")
        lines.append(f"| {rq} | **{item['status']}** | `{measured}` | {item['remaining']} |")
    lines += ["", "## 统计单位与防伪重复", "",
              "- 同一模型父结构的多个 shape 是配对 counterfactual，不是多个独立模型。",
              "- 泛化统计先按 `parent_case_id` 聚合，再在模型父结构之间 bootstrap；不得按原始行数给置信区间。",
              "- correctness 未通过、候选缺失或运行步骤失败的 cell 保持 partial/missing。",
              "- L-RQ5 在单卡上保持 blocked；Hexcute 与跨架构实验不在本补充结论范围。", ""]
    (root / "RQ_ADEQUACY_ANALYSIS_CN.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(root / "RQ_ADEQUACY_ANALYSIS_CN.md"),
                      "statuses": {rq: item["status"] for rq, item in assessments.items()}},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
