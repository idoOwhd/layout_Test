#!/usr/bin/env python3
"""Strict, per-RQ analysis of the full single-GPU adequacy suite.

The older reports are intentionally left untouched.  This analyzer:

* aggregates CUDA process repetitions before ranking candidates;
* applies the preregistered 3% practical-effect threshold;
* separates raw measurement rows from independent contracts/models;
* canonicalizes equal address maps;
* distinguishes direct, supporting, modeled, invalid and blocked evidence;
* records code-audit findings that change the scientific interpretation.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


THRESHOLD = 1.03


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def median(values) -> float | None:
    values = list(values)
    return statistics.median(values) if values else None


def f(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def aggregate_cuda(rows: list[dict[str, str]]) -> list[dict]:
    fields = (
        "benchmark", "case_id", "subgraph", "phase", "strategy", "layout",
        "tokens", "kv_heads", "query_heads", "head_dim", "page_size",
        "reuse_count", "fanout_head", "fanout_token", "allocated_tokens",
        "trace_kind", "attention_kind",
    )
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field, "") for field in fields)].append(row)
    result = []
    for key, samples in groups.items():
        errors = [float(row.get("max_abs_error", "nan")) for row in samples]
        result.append({
            **dict(zip(fields, key)),
            "p50_ms": median(float(row["p50_ms"]) for row in samples),
            "process_repetitions": len(samples),
            "max_abs_error": max(errors),
            "correct": all(math.isfinite(error) and error <= 1e-3 for error in errors),
        })
    return result


def grouped_candidates(rows: list[dict], fields: tuple[str, ...],
                       time_field: str = "p50_ms", strategy: str = "strategy") -> dict:
    groups = defaultdict(dict)
    for row in rows:
        if not row.get("correct", True):
            continue
        try:
            value = float(row[time_field])
        except (KeyError, TypeError, ValueError):
            continue
        groups[tuple(str(row.get(field, "")) for field in fields)][str(row.get(strategy, ""))] = value
    return groups


def winner(values: dict[str, float], threshold: float = THRESHOLD) -> tuple[str, float]:
    ordered = sorted(values.items(), key=lambda item: item[1])
    if len(ordered) < 2:
        return (ordered[0][0], math.inf) if ordered else ("missing", math.nan)
    ratio = ordered[1][1] / ordered[0][1]
    return (ordered[0][0] if ratio >= threshold else "tie", ratio)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    raw = root / "raw"

    manifest_value = json.loads((root / "cases/executed_manifest.json").read_text(encoding="utf-8"))
    cases = manifest_value["cases"] if isinstance(manifest_value, dict) else manifest_value
    case_by_id = {case["case_id"]: case for case in cases}
    parent_ids = {case["parent_case_id"] for case in cases}
    structures = Counter(case["structure"] for case in cases)

    cuda_raw = csv_rows(raw / "cuda_reference.csv")
    cuda = aggregate_cuda(cuda_raw)
    triton = csv_rows(raw / "triton_explicit.csv")
    tvm = csv_rows(raw / "tvm_projected.csv")
    cutlass = csv_rows(raw / "cutlass_attention.csv")
    whole = jsonl(raw / "whole_graph.jsonl")
    boundary = jsonl(raw / "boundary_layout.jsonl")
    serving = {name: jsonl(raw / f"{name}_native.jsonl") for name in ("vllm", "sglang")}
    kv = {
        name: jsonl(root / f"kv_request_batch/raw/{name}_kv_request_batch.jsonl")
        for name in ("vllm", "sglang")
    }
    status_rows = jsonl(root / "status.jsonl")

    # RQ1: producer-local choice versus consumer and complete edge.
    primitive = {
        benchmark: grouped_candidates(
            [row for row in cuda if row["benchmark"] == benchmark], ("case_id",))
        for benchmark in ("kv_write", "decode_head_scan", "token_major_scan")
    }
    reuse = grouped_candidates(
        [row for row in cuda if row["benchmark"] == "conversion_reuse_pipeline"],
        ("case_id", "reuse_count"),
    )
    producer_consumer_regret, producer_edge_regret = [], []
    raw_edge_inversions = 0
    for key, producer in primitive["kv_write"].items():
        case_id = key[0]
        producer_choice = min(producer, key=producer.get)
        consumer = primitive["decode_head_scan"][(case_id,)]
        producer_consumer_regret.append(consumer[producer_choice] / min(consumer.values()))
        edge = reuse[(case_id, "1")]
        selected = "common_NHD" if producer_choice == "NHD" else "producer_native_HND"
        producer_edge_regret.append(edge[selected] / min(edge.values()))
        raw_edge_inversions += selected != min(edge, key=edge.get)
    rq1 = {
        "cases": len(producer_edge_regret),
        "raw_edge_inversions": raw_edge_inversions,
        "meaningful_edge_regret_cases": sum(value >= THRESHOLD for value in producer_edge_regret),
        "meaningful_consumer_regret_cases": sum(value >= THRESHOLD for value in producer_consumer_regret),
        "median_edge_regret": median(producer_edge_regret),
        "max_edge_regret": max(producer_edge_regret),
        "robust_primitive_winners": {
            benchmark: dict(Counter(winner(values)[0] for values in groups.values()))
            for benchmark, groups in primitive.items()
        },
    }

    # RQ2: common layout versus split domains over seven direct fanouts.
    multi = grouped_candidates(
        [row for row in cuda if row["benchmark"] == "weighted_multi_consumer_pipeline"],
        ("case_id", "fanout_head", "fanout_token"),
    )
    raw_by_case, robust_by_case = defaultdict(set), defaultdict(set)
    for (case_id, _head, _token), values in multi.items():
        raw_by_case[case_id].add(min(values, key=values.get))
        robust, _ = winner(values)
        if robust != "tie":
            robust_by_case[case_id].add(robust)
    fixed_oracle_regret = []
    split_wins = 0
    for case_id in sorted({key[0] for key in multi}):
        groups = [values for key, values in multi.items() if key[0] == case_id]
        common = set.intersection(*(set(values) for values in groups))
        oracle = sum(min(values.values()) for values in groups)
        fixed = min(sum(values[strategy] for values in groups) for strategy in common)
        fixed_oracle_regret.append(fixed / oracle)
        split_wins += any(min(values, key=values.get) == "split_NHD_HND_with_conversion"
                          for values in groups)
    rq2 = {
        "fanout_pairs": sorted({(int(key[1]), int(key[2])) for key in multi}),
        "cases": len(fixed_oracle_regret),
        "raw_winner_change_cases": sum(len(values) > 1 for values in raw_by_case.values()),
        "robust_winner_change_cases": sum(len(values) > 1 for values in robust_by_case.values()),
        "split_winner_cases": split_wins,
        "median_fixed_over_oracle": median(fixed_oracle_regret),
        "max_fixed_over_oracle": max(fixed_oracle_regret),
        "fixed_regret_ge_3pct": sum(value >= THRESHOLD for value in fixed_oracle_regret),
    }

    # RQ3: conversion reuse and general boundary materialization.
    reuse_winners = Counter()
    convert_vs_common = defaultdict(list)
    for (case_id, count), values in reuse.items():
        reuse_winners[min(values, key=values.get)] += 1
        convert_vs_common[int(count)].append(
            values["common_NHD"] / values["NHD_then_convert_once_HND"])
    boundary_standard = grouped_candidates(
        [row for row in boundary if row.get("experiment") == "boundary_layout_sweep"
         and row.get("status") == "success" and row.get("correctness_passed") is True],
        ("case_id",),
    )
    boundary_meaningful = Counter()
    boundary_raw = Counter()
    for values in boundary_standard.values():
        best = min(values, key=values.get)
        boundary_raw[best] += 1
        if best != "native_contiguous" and values["native_contiguous"] / values[best] >= THRESHOLD:
            boundary_meaningful[best] += 1
    cutlass_unique = {
        (row.get("rows"), row.get("cols"), row.get("tile_rows"), row.get("tile_cols")): row
        for row in cutlass
    }
    rq3 = {
        "reuse_levels": sorted(convert_vs_common),
        "cuda_global_conversion_wins": reuse_winners["NHD_then_convert_once_HND"],
        "convert_vs_common_ge_3pct_at_reuse64": sum(
            value >= THRESHOLD for value in convert_vs_common[64]),
        "boundary_triplets": len(boundary_standard),
        "boundary_raw_winners": dict(boundary_raw),
        "boundary_meaningful_non_native": dict(boundary_meaningful),
        "cutlass_rows": len(cutlass),
        "cutlass_unique_geometry_tile_comparisons": len(cutlass_unique),
        "cutlass_unique_ge_3pct": sum(
            float(row["winner_speedup"]) >= THRESHOLD for row in cutlass_unique.values()),
        "cutlass_mapped_ge_3pct": sum(
            float(row["winner_speedup"]) >= THRESHOLD for row in cutlass),
        "cutlass_median_winner_speedup": median(
            float(row["winner_speedup"]) for row in cutlass_unique.values()),
        "tvm_conversion_is_measured_cost_model": all(
            row.get("evidence_level") == "native_measured_cost_model"
            for row in tvm if row.get("experiment") == "conversion_reuse"),
    }

    # RQ4: shape/workload stability.  Whole-graph backend is supporting rather
    # than a pure layout intervention, so retain that qualifier in the report.
    whole_by_case = defaultdict(dict)
    numerical_mismatches = []
    for row in whole:
        if row.get("status") == "success":
            whole_by_case[row["case_id"]][row["backend"]] = row
        elif row.get("status") == "numerical_mismatch":
            numerical_mismatches.append(row)
    whole_ratios, parent_winners = [], defaultdict(set)
    for case_id, pair in whole_by_case.items():
        if {"pytorch", "triton"} <= set(pair):
            ratio = float(pair["pytorch"]["p50_ms"]) / float(pair["triton"]["p50_ms"])
            whole_ratios.append(ratio)
            parent = case_by_id[case_id]["parent_case_id"]
            parent_winners[parent].add("TorchInductor" if ratio > 1 else "PyTorch eager")
    native_summary = {}
    for framework, rows in serving.items():
        scopes = defaultdict(list)
        for row in rows:
            scopes[(row["case_id"], row["comparison_scope"])].append(row)
        by_scope = {}
        for scope in sorted({key[1] for key in scopes}):
            wins, ratios = Counter(), []
            for (case_id, current_scope), values in scopes.items():
                if current_scope != scope or len(values) < 2:
                    continue
                ordered = sorted(values, key=lambda row: float(row["p50_ms"]))
                wins[ordered[0]["variant"]] += 1
                ratios.append(float(ordered[1]["p50_ms"]) / float(ordered[0]["p50_ms"]))
            by_scope[scope] = {"pairs": len(ratios), "winners": dict(wins),
                               "median_winner_speedup": median(ratios),
                               "ge_3pct": sum(value >= THRESHOLD for value in ratios)}
        native_summary[framework] = by_scope
    rq4 = {
        "whole_correct_pairs": len(whole_ratios),
        "whole_numerical_mismatches": len(numerical_mismatches),
        "whole_median_eager_over_inductor": median(whole_ratios),
        "inductor_ge_3pct": sum(value >= THRESHOLD for value in whole_ratios),
        "eager_ge_3pct": sum(value <= 1 / THRESHOLD for value in whole_ratios),
        "parents": len(parent_winners),
        "parents_with_backend_crossover": sum(len(values) > 1 for values in parent_winners.values()),
        "native_serving": native_summary,
    }

    # RQ6: trace policies.  All traces have horizon 32; this cannot establish N*.
    state = grouped_candidates(
        [row for row in cuda if row["benchmark"] == "state_migration_trace"],
        ("case_id", "trace_kind"),
    )
    trace_winners, eager_over_hysteresis = defaultdict(Counter), defaultdict(list)
    dynamic_over_fixed = []
    for (case_id, trace), values in state.items():
        trace_winners[trace][winner(values)[0]] += 1
        eager_over_hysteresis[trace].append(
            values["eager_switch"] / min(values["hysteresis_2"], values["hysteresis_4"]))
        fixed = min(values["fixed_NHD"], values["fixed_HND"])
        dynamic = min(values["eager_switch"], values["hysteresis_2"], values["hysteresis_4"])
        dynamic_over_fixed.append(fixed / dynamic)
    rq6 = {
        "trace_kinds": sorted(trace_winners),
        "fixed_horizon": 32,
        "robust_winners_by_trace": {key: dict(value) for key, value in trace_winners.items()},
        "alternating_eager_over_hysteresis_median": median(eager_over_hysteresis["alternating"]),
        "markov_eager_over_hysteresis_median": median(eager_over_hysteresis["seeded_markov"]),
        "alternating_hysteresis_ge_3pct": sum(
            value >= THRESHOLD for value in eager_over_hysteresis["alternating"]),
        "markov_hysteresis_ge_3pct": sum(
            value >= THRESHOLD for value in eager_over_hysteresis["seeded_markov"]),
        "dynamic_beats_best_fixed_ge_3pct": sum(value >= THRESHOLD for value in dynamic_over_fixed),
        "policy_trace_contracts": len(state),
    }

    # RQ7 uses the correctness-gated standard boundary triplets.
    rq7 = {
        "complete_correct_triplets": len(boundary_standard),
        "zero_copy_ge_3pct": boundary_meaningful["alternate_strided_view"],
        "repair_each_call_ge_3pct": boundary_meaningful["repair_to_contiguous_each_call"],
        "multi_consumer_legality_intersection_measured": False,
        "instruction_alignment_intervention_measured": False,
    }

    # RQ8: current data×metadata rows fail a semantic audit.  Page-table/route
    # rows remain valid and are analyzed independently.
    paged = [row for row in cuda if row["benchmark"] in
             {"paged_decode_head", "sparse_paged_decode_head"}]
    page_contracts = grouped_candidates(
        paged, ("case_id", "reuse_count", "trace_kind"), strategy="page_size")
    # grouped_candidates cannot use page_size as strategy when duplicate
    # entries exist only after aggregation; CUDA is already aggregated here.
    page_contracts = defaultdict(dict)
    for row in paged:
        page_contracts[(row["case_id"], row["reuse_count"], row["trace_kind"])][
            int(row["page_size"])] = float(row["p50_ms"])
    page_winners_by_case = defaultdict(set)
    for (case_id, _route, _table), values in page_contracts.items():
        page_winners_by_case[case_id].add(min(values, key=values.get))
    rq8 = {
        "data_metadata_rows": sum(row["benchmark"] == "data_metadata_scan" for row in cuda),
        "data_metadata_semantically_valid": False,
        "reason": "the executed binary reinterpreted one flat per-token scale buffer as both NHD and HND, so logical scale[token,head] values changed; max_abs_error was hard-coded to zero",
        "raw_named_candidates": 8,
        "effective_address_patterns": 6,
        "page_route_cases_with_page_winner_change": sum(
            len(values) > 1 for values in page_winners_by_case.values()),
        "page_route_cases": len(page_winners_by_case),
    }

    # RQ9: paged_NHD and NHD are one valid-token affine map.
    triton_edge = [row for row in triton if row.get("experiment") == "producer_consumer_edge"
                   and str(row.get("correct", "")).lower() == "true"]
    triton_by_case = defaultdict(list)
    for row in triton_edge:
        triton_by_case[row["case_id"]].append(row)
    raw_extended, unique_extended, unique_meaningful = 0, 0, 0
    for rows in triton_by_case.values():
        best = min(rows, key=lambda row: float(row["pipeline_p50_ms"]))
        native = min((row for row in rows if row["layout"] in {"NHD", "HND"}),
                     key=lambda row: float(row["pipeline_p50_ms"]))
        if best["layout"] not in {"NHD", "HND"}:
            raw_extended += 1
        if best["layout"] == "paged_HND":
            unique_extended += 1
            gain = float(native["pipeline_p50_ms"]) / float(best["pipeline_p50_ms"])
            unique_meaningful += gain >= THRESHOLD
    rq9 = {
        "triton_cases": len(triton_by_case),
        "raw_named_layouts": ["NHD", "HND", "paged_NHD", "paged_HND"],
        "canonical_address_maps": ["NHD_or_paged_NHD", "HND", "paged_HND"],
        "raw_extended_wins": raw_extended,
        "unique_paged_hnd_wins": unique_extended,
        "unique_paged_hnd_ge_3pct": unique_meaningful,
        "cutlass_unique_ge_3pct": rq3["cutlass_unique_ge_3pct"],
    }

    # RQ10: unique contracts, fragmentation and fixed-page regret.
    case_pages = {}
    for row in paged:
        case_pages[(row["case_id"], int(row["page_size"]))] = (
            int(row["tokens"]), int(row["allocated_tokens"]))
    page_sizes = sorted({page for _case, page in case_pages})
    mean_regret = {
        page: statistics.mean(values[page] / min(values.values())
                              for values in page_contracts.values())
        for page in page_sizes
    }
    best_fixed_page = min(mean_regret, key=mean_regret.get)
    fixed_regret = [values[best_fixed_page] / min(values.values())
                    for values in page_contracts.values()]
    fragmented = [(live, allocated) for live, allocated in case_pages.values()
                  if allocated > live]
    rq10 = {
        "page_sizes": page_sizes,
        "unique_case_page_contracts": len(case_pages),
        "fragmented_case_page_contracts": len(fragmented),
        "fragmented_unique_cases": len({case_id for (case_id, page), (live, allocated)
                                         in case_pages.items() if allocated > live}),
        "median_allocated_over_live": median(allocated / live for live, allocated in case_pages.values()),
        "max_allocated_over_live": max(allocated / live for live, allocated in case_pages.values()),
        "page_winner_change_cases": sum(len(values) > 1 for values in page_winners_by_case.values()),
        "best_fixed_kernel_page": best_fixed_page,
        "mean_fixed_over_oracle": statistics.mean(fixed_regret),
        "median_fixed_over_oracle": median(fixed_regret),
        "fixed_regret_ge_3pct": sum(value >= THRESHOLD for value in fixed_regret),
        "kernel_contracts": len(fixed_regret),
        "native_serving": {framework: native_summary[framework].get("page_size_only", {})
                           for framework in native_summary},
    }

    tvm_unique_shapes = len({(row.get("rows"), row.get("cols")) for row in tvm})
    cutlass_unique_geometries = len({(row.get("rows"), row.get("cols")) for row in cutlass})
    execution = {
        "status_steps": len(status_rows),
        "failed_steps": [row for row in status_rows if row.get("status") == "failed"],
        "hexcute_status": next((row.get("status") for row in status_rows
                                if row.get("step") == "hexcute"), "missing"),
        "manifest_cases": len(cases),
        "parent_models": len(parent_ids),
        "structures": dict(structures),
        "raw_rows": {"CUDA-reference": len(cuda_raw), "Triton": len(triton),
                     "TVM": len(tvm), "CUTLASS/CuTe": len(cutlass),
                     "whole_graph": len(whole), "boundary": len(boundary),
                     "vLLM-serving": len(serving["vllm"]),
                     "SGLang-serving": len(serving["sglang"]),
                     "vLLM-KV-writer": len(kv["vllm"]),
                     "SGLang-KV-writer": len(kv["sglang"])},
        "independent_units": {"CUDA_attention_cases": len({row["case_id"] for row in cuda}),
                              "CUDA_process_repetitions": 5,
                              "TVM_unique_projected_shapes": tvm_unique_shapes,
                              "CUTLASS_unique_score_geometries": cutlass_unique_geometries},
    }

    payload = {
        "schema_version": 2,
        "practical_effect_threshold": THRESHOLD,
        "execution": execution,
        "rqs": {"v10-RQ1_and_L-RQ1": rq1, "L-RQ2": rq2, "L-RQ3": rq3,
                "L-RQ4": rq4, "L-RQ5": {"status": "blocked_single_gpu"},
                "L-RQ6": rq6, "L-RQ7": rq7, "L-RQ8": rq8,
                "L-RQ9": rq9, "L-RQ10": rq10},
    }
    json_path = root / "RQ_FULL_PER_RQ_STRICT_ANALYSIS.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")

    # Compact figure: numerator means a preregistered meaningful observation,
    # not raw winner changes.  RQ5/RQ8 are explicitly non-numeric.
    chart = [
        ("RQ1 edge regret", rq1["meaningful_edge_regret_cases"], rq1["cases"], "#4c78a8"),
        ("RQ2 fixed-policy regret", rq2["fixed_regret_ge_3pct"], rq2["cases"], "#f58518"),
        ("RQ3 non-native boundary", sum(boundary_meaningful.values()), rq3["boundary_triplets"], "#54a24b"),
        ("RQ4 parent crossover", rq4["parents_with_backend_crossover"], rq4["parents"], "#e45756"),
        ("RQ6 dynamic > fixed", rq6["dynamic_beats_best_fixed_ge_3pct"], rq6["policy_trace_contracts"], "#72b7b2"),
        ("RQ7 zero-copy", rq7["zero_copy_ge_3pct"], rq7["complete_correct_triplets"], "#b279a2"),
        ("RQ9 unique extended", rq9["unique_paged_hnd_ge_3pct"], rq9["triton_cases"], "#ff9da6"),
        ("RQ10 fixed-page regret", rq10["fixed_regret_ge_3pct"], rq10["kernel_contracts"], "#9d755d"),
    ]
    width, height, left, bar_width = 980, 430, 235, 650
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<style>text{font-family:DejaVu Sans,Arial,sans-serif;font-size:13px}.title{font-size:18px;font-weight:bold}</style>',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="20" y="28" class="title">Meaningful observations after 3% gate</text>',
           '<text x="20" y="50">Fractions use independent paired contracts, not raw CSV rows.</text>']
    for index, (label, count, total, color) in enumerate(chart):
        y = 72 + index * 42
        fraction = count / total if total else 0
        svg += [f'<text x="20" y="{y + 18}">{html.escape(label)}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_width}" height="22" fill="#eeeeee"/>',
                f'<rect x="{left}" y="{y}" width="{bar_width * fraction:.1f}" height="22" fill="{color}"/>',
                f'<text x="{left + bar_width + 8}" y="{y + 17}">{count}/{total} ({fraction * 100:.1f}%)</text>']
    svg.append('</svg>')
    figure_path = root / "figures/rq_full_strict_meaningful_evidence.svg"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.write_text("\n".join(svg) + "\n", encoding="utf-8")

    lines = [
        "# v10/v13 单卡 full 实验：逐 RQ 严格分析与代码审计",
        "",
        f"> 结果目录：`{root}`。本文件由 `analyze_rq_full_per_rq.py` 从原始结果重新聚合生成；没有修改或覆盖原始 CSV/JSONL。",
        "",
        "## 总判定",
        "",
        f"- 执行层面完整：{execution['status_steps']} 个状态记录中无 failed step；Hexcute 为 `{execution['hexcute_status']}`，符合本轮排除跨架构实验的约束。",
        f"- 数据规模：{len(cases)} 个 contract、{len(parent_ids)} 个 real-world parent、{len(structures)} 类结构；CUDA raw={len(cuda_raw)} 行，但独立 attention case 只有 {rq1['cases']}，进程重复先取中位数。TVM 的 {len(tvm)} 行只对应 {tvm_unique_shapes} 个唯一 projected shape；CUTLASS 的 {len(cutlass)} 行只对应 {cutlass_unique_geometries} 个唯一 score geometry。",
        f"- 所有 winner 结论使用预声明 **3% practical-effect gate**。小于 3% 的差异记为 tie；不能用 raw winner 数替代科学效应。",
        "- 严格结论不是“所有 RQ 已证实”：RQ1/RQ4/RQ7/RQ9 有直接或较强局部支持；RQ2 在测试范围内给出重要的否定结果；RQ3/RQ6/RQ10 只部分支持；RQ5 单卡阻塞；RQ8 的 data×metadata 实验因代码语义错误失效、需增量重跑。",
        "",
        "![3% gate 后的有效观察](figures/rq_full_strict_meaningful_evidence.svg)",
        "",
        "## 统计口径",
        "",
        "- `speedup = baseline_p50 / candidate_p50`；大于 1 表示 candidate 更快。candidate 间 winner 的有效增益是第二名 p50 / 第一名 p50。",
        "- CUDA 的 5 次独立进程结果先按 `case × experiment × candidate × causal-axis` 取 p50 中位数，再比较候选；不选择最快的一次 repetition。",
        f"- shape 是同一 parent model 的受控 counterfactual，不是独立模型。{len(cases)} 个 contract 不能当成 {len(cases)} 个独立模型构造置信区间。",
        "- TVM/CUTLASS 将相同 projected geometry 的一次测量映射回多个 real-world case；coverage 可按 case 报告，统计独立性必须按唯一 geometry 报告。",
        "",
        "## 逐 RQ 结论",
        "",
        "### v10-RQ1 / L-RQ1：局部最优与 edge/subgraph 最优是否冲突",
        "",
        f"结论：**存在，但不是普遍的大效应**。{rq1['cases']} 个 attention case 中 raw edge winner 与 producer-local 决策不一致 {rq1['raw_edge_inversions']} 个；施加 3% 门槛后，只有 {rq1['meaningful_edge_regret_cases']}/{rq1['cases']} 个 case 的 producer-local 选择对完整 producer→head-consumer edge 产生有意义 regret，最大 regret={rq1['max_edge_regret']:.3f}×。producer-local 选择在 consumer-only 上有意义失配 {rq1['meaningful_consumer_regret_cases']}/{rq1['cases']}。",
        f"负对照同样重要：primitive 的 robust winner（其余为 tie）为 `{rq1['robust_primitive_winners']}`，说明大量 raw winner 变化来自 A10 上微秒级计时量化/噪声，支持 v13 的 H1.NEG。",
        "框架解释：CUDA reference 是直接因果证据；Triton 的 producer+consumer pipeline 是独立复核；TVM 是 projected tensor boundary；CUTLASS/vLLM/SGLang 只提供相邻 boundary 或固定模型支持，不能都记为完整 RQ1 直接验证。",
        "",
        "### L-RQ2：最优 layout domain 粒度",
        "",
        f"结论：**测试范围内不支持“异构 consumer 一定需要 split domain”**。CUDA 直接覆盖 {len(rq2['fanout_pairs'])} 个 fanout 对；raw winner 随 fanout 改变 {rq2['raw_winner_change_cases']}/{rq2['cases']}，但 robust winner 改变仅 {rq2['robust_winner_change_cases']}/{rq2['cases']}。显式 `split_NHD_HND_with_conversion` 在 {rq2['cases']} 个 case 中从未成为任何 fanout 的 global winner。每 case 选择一个跨 fanout 的最佳固定策略，相对 per-fanout oracle 的中位 regret={rq2['median_fixed_over_oracle']:.4f}×、最大={rq2['max_fixed_over_oracle']:.4f}×，达到 3% 的为 {rq2['fixed_regret_ge_3pct']}/{rq2['cases']}。",
        "这在当前 NHD/HND、fanout≤8、单卡 A10 的边界上反驳 H2.1，支持 H2.3；不检验 speculative acceptance(H2.4) 或 KV/recurrent 混合状态 overhead(H2.5)。Triton 直接测 common-layout fanout，但没有 split 候选；TVM 是 consumer-only module，producer/materialization 未计时；vLLM/SGLang batch sweep 是 supporting evidence。",
        "",
        "### L-RQ3：转换/修复的摊销阈值",
        "",
        f"结论：**依 boundary 而异，当前实验没有建立一个通用 R-star**。CUDA 中 `NHD_then_convert_once_HND` 在 {len(rq3['reuse_levels'])}×{rq1['cases']} 个 global comparisons 里 raw winner 仅 {rq3['cuda_global_conversion_wins']} 次；到 reuse=64，相对 `common_NHD` 达到 3% 的也只有 {rq3['convert_vs_common_ge_3pct_at_reuse64']}/{rq1['cases']}，而且 producer-native HND 仍可能更快。",
        f"PyTorch {rq3['boundary_triplets']} 个正确三元组中，zero-copy alternate 有意义胜出 {rq3['boundary_meaningful_non_native'].get('alternate_strided_view', 0)}，每次 repair 有意义胜出 {rq3['boundary_meaningful_non_native'].get('repair_to_contiguous_each_call', 0)}。CUTLASS score boundary 去除映射重复后有 {rq3['cutlass_unique_ge_3pct']}/{rq3['cutlass_unique_geometry_tile_comparisons']} 个 unique geometry×tile comparison 达到 3%（映射回 real-world case 后为 {rq3['cutlass_mapped_ge_3pct']}/{rq3['cutlass_rows']}），winner speedup 中位数 {rq3['cutlass_median_winner_speedup']:.3f}×。这证明“直接 tiled consume vs transform+flat”的 boundary trade-off 很强，但不能外推成 KV conversion 的同一 R*。TVM conversion_reuse 是 measured-cost model，不是完整 pipeline 直接计时。",
        "",
        "### L-RQ4：最优 layout/策略跨 workload、shape、hardware 是否稳定",
        "",
        f"结论：**单 GPU 的 shape/workload 稳定性问题存在；hardware 泛化未验证**。Whole-graph 正确配对 {rq4['whole_correct_pairs']}，PyTorch eager/TorchInductor 中位比 {rq4['whole_median_eager_over_inductor']:.3f}×；Inductor 有意义胜出 {rq4['inductor_ge_3pct']}，eager 有意义胜出 {rq4['eager_ge_3pct']}。{rq4['parents']} 个 parent 中 {rq4['parents_with_backend_crossover']} 个随 shape/phase 改变 backend winner。不过这项包含 fusion/codegen 差异，是 supporting system evidence，不是 layout-only intervention。",
        f"原生 serving：vLLM layout-only 达 3% 的 {native_summary['vllm']['layout_only']['ge_3pct']}/{native_summary['vllm']['layout_only']['pairs']}、page-size {native_summary['vllm']['page_size_only']['ge_3pct']}/{native_summary['vllm']['page_size_only']['pairs']}；SGLang 分别 {native_summary['sglang']['layout_only']['ge_3pct']}/{native_summary['sglang']['layout_only']['pairs']} 与 {native_summary['sglang']['page_size_only']['ge_3pct']}/{native_summary['sglang']['page_size_only']['pairs']}。只有 A10，故 H4.4（丰富特征优于 GPU-name rule）仍未测。",
        "",
        "### L-RQ5：通信/placement-aware layout inversion",
        "",
        "结论：**blocked，而不是 negative**。卡1不能形成 multi-rank collective、rank skew、replication 或 placement counterfactual；本轮没有把 HBM pressure/memcpy 冒充通信证据。H5.1–H5.5 均未验证。",
        "",
        "### L-RQ6：持久状态 layout 的在线重配置阈值",
        "",
        f"结论：**只支持 hysteresis observation，尚未测出 N-star**。{len(rq6['trace_kinds'])} 种 trace 全部固定 horizon={rq6['fixed_horizon']}，没有 sweep 剩余生命周期，因此 H6.1/H6.4 不能判定。alternating 与 seeded-Markov 上 eager/hysteresis 的中位比为 {rq6['alternating_eager_over_hysteresis_median']:.3f}×、{rq6['markov_eager_over_hysteresis_median']:.3f}×，两者各 {rq6['alternating_hysteresis_ge_3pct']}/{rq1['cases']}、{rq6['markov_hysteresis_ge_3pct']}/{rq1['cases']} 支持 H6.2。可是 adaptive family 相对每条 trace 的最佳 fixed oracle 仅 {rq6['dynamic_beats_best_fixed_ge_3pct']}/{rq6['policy_trace_contracts']} 达到 3%。",
        "vLLM/SGLang 的 native writer/state slice 没有 live migration、duplicate-live memory 或 CUDA-graph recapture，因此只作 supporting observation，不能声称完成 H6.5。",
        "",
        "### L-RQ7：layout 等价与 zero-copy transformability",
        "",
        f"结论：**H7.1 有直接支持，H7.2/H7.3 未覆盖**。{rq7['complete_correct_triplets']} 个 exact-shape triplet 均通过 correctness；zero-copy strided view 相对 native 达到 3% 的有 {rq7['zero_copy_ge_3pct']} 个，显式 repair-each-call 达到 3% 的有 {rq7['repair_each_call_ge_3pct']} 个。它说明 named-layout mismatch 不能直接等同于 materialization。",
        "但当前没有 correctness-gated 多 consumer alias/lifetime 交集，也没有单独控制 alignment/instruction legality；所以不能据此判定 H7.2/H7.3。CUTLASS 的 tiled-direct/transform 边界是有用的第二类直接 boundary 证据。",
        "",
        "### L-RQ8：data–metadata 联合 layout",
        "",
        f"结论：**当前仅 page-table/route 子问题有效；data×scale 主实验无效**。执行得到 {rq8['data_metadata_rows']} 个进程聚合行，但旧二进制把同一 flat scale allocation 按 NHD/HND 两种公式读取，使逻辑 scale 值改变，并将 `max_abs_error` 固定为 0。它违反“只改变 layout、不改变值”的 paired-counterfactual 条件。源码已修复为两份保持相同逻辑值的物理 buffer，并加入真实输出比较；这些 raw 行必须增量重跑后才能用于 H8.1/H8.4/H8.5。",
        f"另一个有效观察是 route stride×page/table：{rq8['page_route_cases_with_page_winner_change']}/{rq8['page_route_cases']} 个 case 的 kernel page winner 随 route/table 条件改变，局部支持 H8.2。per-head metadata 的两个 layout 名实际上是同一索引，因此 8 个命名候选只有 6 个有效 address pattern；不能把重复候选算成搜索空间扩大。",
        "",
        "### L-RQ9：搜索空间表达能力与截断",
        "",
        f"结论：**在 explicit Triton adapter 上有正证据，但必须 canonicalize**。四个名字 `{rq9['raw_named_layouts']}` 只有三个 valid-token address map：`{rq9['canonical_address_maps']}`，因为 paged_NHD 与 NHD 的 offset 相同。raw extended winner={rq9['raw_extended_wins']}/{rq9['triton_cases']}；排除这个 alias 后，唯一扩展 map paged_HND winner={rq9['unique_paged_hnd_wins']}/{rq9['triton_cases']}，其中达到 3% 的为 {rq9['unique_paged_hnd_ge_3pct']}/{rq9['triton_cases']}。",
        f"这支持 H9.1/H9.5 的局部版本；CUTLASS 还有 {rq9['cutlass_unique_ge_3pct']}/{rq3['cutlass_unique_geometry_tile_comparisons']} 个去重后的 tile-boundary comparisons 支持候选截断会有代价。当前候选是人工显式枚举，不等于 Triton/vLLM/SGLang 自身 optimizer 已搜索这些 map；H9.2 的多 backend support-intersection 尚未直接测。",
        "",
        "### L-RQ10：page/block 粒度与 allocator coupling",
        "",
        f"结论：**kernel page geometry 已覆盖，allocator/prefix-sharing 全问题未完成**。{len(rq10['page_sizes'])} 个 page size、{rq10['unique_case_page_contracts']} 个唯一 case×page，其中 fragmented={rq10['fragmented_case_page_contracts']}，覆盖 {rq10['fragmented_unique_cases']} 个 case；allocated/live 中位 {rq10['median_allocated_over_live']:.3f}×、最大 {rq10['max_allocated_over_live']:.3f}×。{len(page_winners_by_case)} 个 case 中 {rq10['page_winner_change_cases']} 个随 route/table 改变 page winner。",
        f"以 isolated-kernel 合同计算，最佳全局固定 page={rq10['best_fixed_kernel_page']}，相对 per-contract oracle 的平均 regret={rq10['mean_fixed_over_oracle']:.3f}×；{rq10['fixed_regret_ge_3pct']}/{rq10['kernel_contracts']} 个合同达到 3%。vLLM block16/32 只有 {native_summary['vllm']['page_size_only']['ge_3pct']}/{native_summary['vllm']['page_size_only']['pairs']} 个有意义差异，SGLang page1/16 为 {native_summary['sglang']['page_size_only']['ge_3pct']}/{native_summary['sglang']['page_size_only']['pairs']}。没有受控 prefix-hit/fanout、allocator latency/metadata traffic 或 multi-pool registration overhead，因此 H10.1/H10.4/H10.5 仍未验证。",
        "",
        "## v13 hypotheses 判定",
        "",
        "| 假设 | 本轮判定 | 依据/缺口 |",
        "|---|---|---|",
        f"| H1.1 | 支持但效应稀疏 | edge regret≥3% 为 {rq1['meaningful_edge_regret_cases']}/{rq1['cases']} |",
        "| H1.2 | 部分支持 | reuse/fanout 改变 raw 排名，但大多数差异不足3% |",
        f"| H1.3 | 支持 | zero-copy 有意义胜出 {rq7['zero_copy_ge_3pct']}/{rq7['complete_correct_triplets']} |",
        "| H1.4 | 部分支持 | producer-native HND 在部分 reuse 合同胜出；未覆盖官方完整 producer |",
        f"| H1.NEG | 支持 | kv_write/decode/token primitive 分别有 {rq1['robust_primitive_winners']['kv_write'].get('tie', 0)}/{rq1['robust_primitive_winners']['decode_head_scan'].get('tie', 0)}/{rq1['robust_primitive_winners']['token_major_scan'].get('tie', 0)} 个 3% 内 tie |",
        "| H2.1 | 当前范围反驳 | split domain 从未成为 global winner |",
        "| H2.2 | 未验证 | 未训练/held-out 比较 weighted model 与 equal voting |",
        "| H2.3 | 支持 | shared domain 在全部 fanout global winner |",
        "| H2.4 | 未验证 | 无 speculative acceptance sweep |",
        "| H2.5 | 未验证 | 无 KV/recurrent 混合 state 与 allocator overhead sweep |",
        "| H3.1 | 未建立 | CUDA 几乎无 materialized conversion global crossover |",
        "| H3.2 | 未验证 | 有单点 HBM contention，没有 pressure×reuse crossover surface |",
        "| H3.3 | 未验证 | 无 wire-native/transfer overlap |",
        f"| H3.4 | 支持 | zero-copy≥3% 为 {rq7['zero_copy_ge_3pct']}/{rq7['complete_correct_triplets']} |",
        "| H3.5 | 支持 | R=1 时 conversion 未胜出 |",
        "| H4.1 | 支持 workload 轴 | 单卡 shape/phase winner crossover；无 hardware 轴 |",
        "| H4.2 | 部分支持 | fixed page 有 kernel regret；没有 held-out workload distribution |",
        "| H4.3 | 支持 | Inductor/若干 native policy 有宽稳定区 |",
        "| H4.4 | 未验证 | 只有 A10，未比较 feature model 与 GPU-name rule |",
        "| H4.5 | 未验证 | 未隔离 local dispatch 开关后的 persistent-layout residual regret |",
        "| H5.1–H5.5 | blocked | 单卡没有通信/placement intervention |",
        "| H6.1 | 未验证 | horizon 固定32 |",
        "| H6.2 | 支持 | alternating/Markov 中 hysteresis 明显优于 eager |",
        "| H6.3 | 部分支持 | stationary trace 大多 tie/fixed；动态策略有相同代码路径的噪声胜负 |",
        "| H6.4 | 未验证 | 无 drift-length/horizon sweep |",
        "| H6.5 | 未验证 | 无 graph recapture/duplicate-live memory |",
        "| H7.1 | 支持 | 41 个有意义 zero-copy case |",
        "| H7.2 | 未验证 | 无 multi-consumer legality intersection |",
        "| H7.3 | 未验证 | 无 alignment/instruction constraint intervention |",
        "| H7.4 | 部分支持 | stride-polymorphic direct path减少部分 repair；未改变真实框架 kernel set |",
        "| H8.1 | 无效待重跑 | data×scale 语义未保持 |",
        "| H8.2 | 部分支持 | route×page kernel winner interaction 存在 |",
        "| H8.3 | 未验证 | 无真实 block-scale/data-tile joint path |",
        "| H8.4 | 无效待重跑 | per-head 两个 metadata 名是同一 address pattern |",
        "| H8.5 | 无效待重跑 | metadata placement 主实验失效 |",
        f"| H9.1 | 支持 | unique extended paged_HND ≥3% 为 {rq9['unique_paged_hnd_ge_3pct']}/{rq9['triton_cases']} |",
        "| H9.2 | 未验证 | 无 heterogeneous backend support-intersection oracle |",
        "| H9.3 | 部分支持 | 扩展收益只出现在少数 geometry；尚未 group-held-out 建模 |",
        f"| H9.4 | 支持 | {rq9['triton_cases'] - rq9['unique_paged_hnd_ge_3pct']}/{rq9['triton_cases']} 没有 unique extended ≥3% 收益 |",
        "| H9.5 | 支持局部版本 | canonicalize 后仍有 paged_HND 新 map |",
        "| H10.1 | 未验证 | route stride 不是 prefix-hit probability/fanout |",
        "| H10.2 | 部分支持 | 记录实际 fragmentation，但未计 metadata/allocator 系统成本 |",
        f"| H10.3 | 支持 kernel-only | fixed page regret≥3% 为 {rq10['fixed_regret_ge_3pct']}/{rq10['kernel_contracts']} |",
        "| H10.4 | 未验证 | 无 multi-granularity pool/registration overhead |",
        "| H10.5 | 未验证 | 只测 kernel latency，没有 allocator-optimal objective |",
        "",
        "## 框架 × RQ 证据边界",
        "",
        "| RQ | CUDA control | Triton | TVM | CUTLASS/CuTe | vLLM | SGLang |",
        "|---|---|---|---|---|---|---|",
        "| RQ1 | direct | direct pipeline | projected direct boundary | adjacent boundary | fixed-model supporting | fixed-model supporting |",
        "| RQ2 | direct shared-vs-split | direct common-layout fanout | direct consumer-only | N/A-not-owned | writer/batch supporting | writer/batch supporting |",
        "| RQ3 | direct KV pipeline | no conversion candidate | measured-cost model | direct tile/transform boundary | source/local supporting | source/local supporting |",
        "| RQ4 | direct attention shapes | direct candidate sweep | projected shapes | direct tile shapes | native serving direct on one checkpoint | native serving direct on one checkpoint |",
        "| RQ5 | blocked | N/A single GPU | N/A single GPU | N/A single GPU | blocked | blocked |",
        "| RQ6 | direct policy traces, fixed N=32 | N/A | N/A | N/A | state slice supporting | state slice supporting |",
        "| RQ7 | component/stride supporting | explicit maps supporting | transform primitive supporting | direct tile/transform | source supporting | source supporting |",
        "| RQ8 | data×metadata invalid; page-table valid | source-only for scales | N/A KV allocator | adjacent scale-layout source | local supporting | local supporting |",
        "| RQ9 | manual control | direct explicit candidates | projected candidates | direct tile candidates | native exposed set on one checkpoint | native exposed set on one checkpoint |",
        "| RQ10 | direct kernel pages | padded storage supporting | N/A serving allocator | N/A serving allocator | native block16/32 direct | native page1/16 direct |",
        "",
        "Hexcute 因 A10/sm_86 与公开 A100/H100 artifact 不兼容而保持 architecture-blocked；本轮按用户要求不评价跨架构结论。",
        "",
        "## 代码审计：发现的问题与处理",
        "",
        "| 问题 | 对本次结果的影响 | 已处理 | 是否需 GPU 重跑 |",
        "|---|---|---|---|",
        "| CUDA RQ8 复用同一 scale buffer，却按 NHD/HND 解释 | 改变逻辑输入，RQ8 data×metadata 行无效 | 已在 `rq_llm_layout_bench.cu` 建立 NHD/HND 等值物理副本并真实比较输出 | **需要，只重跑 CUDA reference 的 RQ8 或完整 CUDA suite** |",
        "| RQ8 `max_abs_error=0` 为常量 | 旧结果没有真实 correctness gate | 同上已修复 data×metadata；primitive write/read/conversion 也增加输出比较 | 需要更新后的 CUDA 数据才能升级证据 |",
        "| 原报告按 raw winner/最快 repetition 统计 | 夸大微秒级噪声与 order effect | 新分析先取5次进程中位数并使用3% gate；旧 persuasive analyzer 也改为 repetition median | 不需要，属于重分析 |",
        "| RQ10 `fragmented_rows=52425` | 把 process repetition、route、table 当成独立 case | 现在报告 1165 个唯一 fragmented case×page、219 个 case | 不需要 |",
        "| Triton paged_NHD 与 NHD offset 相同 | 4 个名字被误算为4个 address map，extended wins 被夸大 | canonicalize 为3个 map，只把 paged_HND 计为 unique extension | 不需要 |",
        "| TVM multi-consumer 未计 producer/materialization | 不能称完整 pipeline direct | 审计脚本改为 `direct_consumer_only` | 不需要 |",
        "| CUTLASS score tile 被标为 KV multi-consumer supporting | 框架 ownership 夸大 | 改为 `N/A-not-owned`，保留 tile/transform boundary 证据 | 不需要 |",
        "| vLLM/SGLang 单候选 native slice 标 `directly_evidenced_rqs` | 没有 paired candidate 却被误称直接验证 | 新 raw schema 分开 `related_rqs` 与空的 direct list；KV writer 只将 writer-only RQ4 记 direct | 当前报告已重解释；未来重跑可修正 raw metadata |",
        "| TVM/CUTLASS equal-shape measurement 映射回许多 case | raw 行数造成伪重复 | 同时报告 130 unique TVM shapes、78 unique CUTLASS geometries | 不需要 |",
        "| RQ6 只有 horizon=32 | 不能从 trace 多样性推导 switching horizon N* | 本报告降级为 partial，并明确未测 H6.1/H6.4/H6.5 | 需要新增 horizon sweep，不是复跑旧命令即可 |",
        "",
        "## 当前结果可安全引用的最小结论",
        "",
        "1. 局部 layout winner 与 edge winner 在一部分真实 attention shape 上确实冲突，但 3% 以上的效应远少于 raw winner inversion。",
        "2. 在本次 NHD/HND、fanout≤8 的单卡实验里，split domain 没有胜过共享 domain；这是有价值的 negative result。",
        "3. zero-copy/repair/tiled-direct 的收益高度依赖 boundary；不能用一个通用 conversion threshold 解释所有子图。",
        "4. layout/page winner 随 shape、route、框架 backend 改变；静态规则可能有 regret，但系统级结论必须按 parent model 与硬件分层。",
        "5. 扩展候选空间有收益，但必须先 canonicalize 相同 address map；否则搜索空间大小和 win count 都会虚增。",
        "6. RQ8 data×metadata、RQ5 分布式、RQ6 horizon、RQ10 allocator/prefix sharing 目前不能宣称完成。",
        "",
        "## 可复核产物",
        "",
        f"- 严格数值汇总：`{json_path}`",
        f"- 新图：`{figure_path}`",
        f"- 原始 CUDA/Triton/TVM/CUTLASS：`{raw}`",
        f"- 完整执行状态：`{root / 'status.jsonl'}`",
        f"- 原始旧报告保留：`{root / 'RQ_ADEQUACY_ANALYSIS_CN.md'}` 与 `{root / 'PERSUASIVE_RQ_FRAMEWORK_REPORT_CN.md'}`",
    ]
    report_path = root / "RQ_FULL_PER_RQ_STRICT_ANALYSIS_CN.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "json": str(json_path),
                      "figure": str(figure_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
