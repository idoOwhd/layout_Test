#!/usr/bin/env python3
"""Turn controlled LLM-layout measurements into falsifiable RQ verdicts.

The report deliberately keeps three evidence tiers separate:

* ``source_static``: an immutable repository line shows a framework decision;
* ``runtime_empirical``: a strategy was timed on the selected GPU;
* ``measured_cost_model``: a planner composes measured primitive costs.

The CUDA reference kernels isolate layout mechanisms.  They are not relabeled
as native vLLM, SGLang, CUTLASS, Triton, TVM, or Hexcute measurements.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
from statistics import geometric_mean
from typing import Iterable

from rq_planner import (
    Consumer,
    Epoch,
    Repair,
    Representation,
    common_layout_plan,
    fixed_epoch_policy,
    joint_representation_plan,
    per_request_oracle,
    run_epoch_controller,
)


RQ_NAMES = {
    "RQ1": "Persistent control-plane versus local adaptivity",
    "RQ2": "Layout granularity",
    "RQ3": "Conversion-aware optimization",
    "RQ4": "GQA/MLA/MoE/parallelism-coupled objective",
    "RQ5": "Speculative multi-consumer negotiation",
    "RQ6": "Static commitment versus workload drift",
    "RQ7": "Cross-subgraph layout-contract synthesis",
    "RQ8": "Joint pass-order/fusion/layout search",
    "RQ9": "Portable threshold/heuristic synthesis",
    "RQ10": "Repair-oriented optimization",
}

FRAMEWORK_ARTIFACTS = {
    "PyTorch": ("llm_boundary_layout_sweep.jsonl", "llm_representative_pytorch_triton.jsonl"),
    "vLLM": ("vllm_native_serving.jsonl", "native_vllm.jsonl", "native_vllm.csv"),
    "SGLang": ("sglang_native_serving.jsonl", "native_sglang.jsonl", "native_sglang.csv"),
    "CUTLASS": ("cutlass_softmax_boundary.csv",),
    "Triton": ("llm_representative_pytorch_triton.jsonl", "triton_kv_layout.csv"),
    "TVM": ("tvm_layout.csv",),
    "Hexcute": ("hexcute.jsonl", "hexcute.csv"),
    "TileLang": ("llm_representative_tilelang.jsonl", "llm_640_tilelang.jsonl"),
    "TensorRT-LLM": ("native_tensorrt_llm.jsonl",),
    "FlashInfer": ("native_flashinfer.jsonl",),
}

SOURCE_ALIASES = {
    "CUTLASS": ("CUTLASS", "CUTLASS/CuTe"),
    "TVM": ("TVM Relax", "TVM Relax/TIR", "TVM MetaSchedule", "TVM TIR"),
}


def load_csv(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric = {"tokens", "kv_heads", "query_heads", "head_dim", "page_size",
               "reuse_count", "bytes", "p20_ms", "p50_ms", "p80_ms", "max_abs_error"}
    for row in rows:
        for key in numeric:
            if key in row and row[key] != "":
                row[key] = float(row[key])
    return rows


def load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def summarize_boundary(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "success" and "strategy" in row:
            grouped[row["case_id"]].append(row)
    alternate_wins, native_wins, repair_penalties, details = [], [], [], []
    subgraph_shapes: dict[str, set[str]] = defaultdict(set)
    for case_id, candidates in grouped.items():
        by_strategy = strategies(candidates)
        required = {"native_contiguous", "alternate_axis12_strided", "repair_to_contiguous_each_call"}
        if not required.issubset(by_strategy):
            continue
        best, robust, margin = winner(by_strategy.values())
        if robust and best["strategy"] == "alternate_axis12_strided":
            alternate_wins.append(case_id)
        elif robust and best["strategy"] == "native_contiguous":
            native_wins.append(case_id)
        direct = min(by_strategy["native_contiguous"]["p50_ms"],
                     by_strategy["alternate_axis12_strided"]["p50_ms"])
        penalty = by_strategy["repair_to_contiguous_each_call"]["p50_ms"] / direct
        repair_penalties.append(penalty)
        sample = candidates[0]
        subgraph_shapes[sample["subgraph"]].add(json.dumps(sample["shape"], sort_keys=True))
        details.append({"case_id": case_id, "subgraph": sample["subgraph"],
                        "phase": sample["phase"], "winner": best["strategy"],
                        "robust": robust, "margin": margin,
                        "repair_slowdown": penalty})
    return {
        "status": "measured" if details else "not_run",
        "case_count": len(details),
        "subgraphs": sorted(subgraph_shapes),
        "shape_count_by_subgraph": {key: len(value) for key, value in sorted(subgraph_shapes.items())},
        "alternate_layout_wins": alternate_wins,
        "native_layout_wins": native_wins,
        "max_repair_slowdown": max(repair_penalties, default=1.0),
        "median_repair_slowdown": (sorted(repair_penalties)[len(repair_penalties) // 2]
                                   if repair_penalties else 1.0),
        "cases": details,
    }


def indexed(rows: Iterable[dict]) -> dict[tuple[str, str], list[dict]]:
    result: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        result[(row["case_id"], row["benchmark"])].append(row)
    return result


def strategies(rows: Iterable[dict]) -> dict[str, dict]:
    return {str(row["strategy"]): row for row in rows}


def winner(rows: Iterable[dict]) -> tuple[dict, bool, float]:
    ordered = sorted(rows, key=lambda row: row["p50_ms"])
    if not ordered:
        raise ValueError("winner() received no rows")
    if len(ordered) == 1:
        return ordered[0], True, math.inf
    best, second = ordered[:2]
    relative = (second["p50_ms"] - best["p50_ms"]) / second["p50_ms"]
    # Non-overlapping central quantile bands are the primary robustness test.
    # A 5% median effect is retained as a useful observation but marked as a
    # weaker, distribution-free threshold rather than a significance test.
    robust = best["p80_ms"] < second["p20_ms"] or relative >= 0.05
    return best, robust, relative


def speedup(baseline: float, optimized: float) -> float:
    return baseline / optimized if optimized > 0 else math.inf


def status(condition: bool, available: bool = True) -> str:
    if not available:
        return "not_run"
    return "supported" if condition else "not_supported"


def make_record(rq: str, problem_status: str, problem: str,
                solution_status: str, solution: str, evidence: list[str],
                metrics: dict, limitations: list[str] | None = None) -> dict:
    return {
        "rq": rq,
        "question": RQ_NAMES[rq],
        "problem_status": problem_status,
        "problem_observation": problem,
        "solution_status": solution_status,
        "solution_observation": solution,
        "evidence_levels": evidence,
        "metrics": metrics,
        "limitations": limitations or [],
    }


def analyze(rows: list[dict]) -> list[dict]:
    if not rows:
        return [make_record(
            rq, "not_run", "Controlled CUDA CSV is absent.", "not_run",
            "Run the GPU suite before drawing an empirical conclusion.", [], {},
            ["No runtime data were available."])
            for rq in RQ_NAMES]

    by = indexed(rows)
    case_ids = sorted({row["case_id"] for row in rows})
    consumer_conflicts = []
    local_gains = []
    primitive = {}
    for case_id in case_ids:
        decode = strategies(by.get((case_id, "decode_head_scan"), []))
        token = strategies(by.get((case_id, "token_major_scan"), []))
        write = strategies(by.get((case_id, "kv_write"), []))
        if not all(layout in decode and layout in token and layout in write for layout in ("NHD", "HND")):
            continue
        dw, dr, _ = winner(decode.values())
        tw, tr, _ = winner(token.values())
        conflict = dw["strategy"] != tw["strategy"] and dr and tr
        if conflict:
            consumer_conflicts.append(case_id)
        adaptive = min(decode["NHD"]["p50_ms"], decode["HND"]["p50_ms"]) + min(
            token["NHD"]["p50_ms"], token["HND"]["p50_ms"])
        fixed = min(
            decode["NHD"]["p50_ms"] + token["NHD"]["p50_ms"],
            decode["HND"]["p50_ms"] + token["HND"]["p50_ms"],
        )
        local_gains.append(speedup(fixed, adaptive))
        primitive[case_id] = {"decode": decode, "token": token, "write": write}

    rq1_gain = max(local_gains, default=1.0)
    result = [make_record(
        "RQ1", status(bool(consumer_conflicts)),
        f"{len(consumer_conflicts)}/{len(primitive)} shapes have robust, opposite layout winners for head-local and token-local consumers.",
        status(rq1_gain > 1.03),
        f"A consumer-aware choice reaches up to {rq1_gain:.3f}x over the best one-layout choice in the measured-cost composition.",
        ["runtime_empirical", "measured_cost_model"],
        {"cases": len(primitive), "robust_conflict_cases": consumer_conflicts,
         "max_consumer_aware_speedup": rq1_gain},
        ["The CUDA kernels isolate access order; this does not identify a native serving framework's end-to-end speedup."])]

    # RQ2 and RQ5 share the same full, directly timed multi-consumer pipelines.
    split_wins, common_wins, split_speedups, multi_details = [], [], [], []
    for case_id in case_ids:
        candidates = strategies(by.get((case_id, "multi_consumer_pipeline"), []))
        required = {"common_NHD", "common_HND", "split_NHD_HND_with_conversion"}
        if not required.issubset(candidates):
            continue
        best, robust, margin = winner(candidates.values())
        common = min(candidates["common_NHD"]["p50_ms"], candidates["common_HND"]["p50_ms"])
        split = candidates["split_NHD_HND_with_conversion"]["p50_ms"]
        split_speedups.append(speedup(common, split))
        if robust and best["strategy"] == "split_NHD_HND_with_conversion":
            split_wins.append(case_id)
        elif robust:
            common_wins.append(case_id)
        multi_details.append({"case_id": case_id, "winner": best["strategy"],
                              "robust": robust, "margin": margin,
                              "split_vs_best_common_speedup": speedup(common, split)})
    rq2_solution = bool(split_wins) and bool(common_wins)
    result.append(make_record(
        "RQ2", status(bool(consumer_conflicts)),
        "One persistent global layout cannot simultaneously realize both consumers' local winners where their preferences conflict.",
        status(rq2_solution),
        ("Adaptive granularity is supported only when split storage wins some shapes and common storage wins others; "
         f"observed split wins={len(split_wins)}, common wins={len(common_wins)}."),
        ["runtime_empirical"],
        {"split_wins": split_wins, "common_wins": common_wins,
         "max_split_speedup": max(split_speedups, default=1.0), "cases": multi_details},
        ["Split storage here is NHD plus a materialized HND repair, not a production KV-pool implementation."]))

    # RQ3: directly timed write/convert/reuse pipelines.
    reuse_switches, explicit_convert_wins, reuse_gains, reuse_details = [], [], [], []
    reuse_by_case: dict[str, dict[int, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row["benchmark"] == "conversion_reuse_pipeline":
            reuse_by_case[row["case_id"]][int(row["reuse_count"])][row["strategy"]] = row
    for case_id, levels in reuse_by_case.items():
        winners = []
        for reuse, candidates in sorted(levels.items()):
            if len(candidates) < 2:
                continue
            best, robust, margin = winner(candidates.values())
            winners.append(best["strategy"])
            baseline = candidates.get("common_NHD", best)["p50_ms"]
            reuse_gains.append(speedup(baseline, best["p50_ms"]))
            if robust and best["strategy"] == "NHD_then_convert_once_HND":
                explicit_convert_wins.append(f"{case_id}@{reuse}")
            reuse_details.append({"case_id": case_id, "reuse": reuse,
                                  "winner": best["strategy"], "robust": robust,
                                  "margin": margin})
        if len(set(winners)) > 1:
            reuse_switches.append(case_id)
    rq3_problem = bool(reuse_switches or explicit_convert_wins)
    rq3_gain = max(reuse_gains, default=1.0)
    result.append(make_record(
        "RQ3", status(rq3_problem),
        f"Reuse-dependent winner changes occur in {len(reuse_switches)} shapes; explicit conversion wins {len(explicit_convert_wins)} shape/reuse points.",
        status(rq3_gain > 1.03),
        f"Pricing conversion once and amortizing it across reuse selects a measured plan up to {rq3_gain:.3f}x faster than always-common-NHD.",
        ["runtime_empirical"],
        {"winner_switch_cases": reuse_switches, "explicit_conversion_wins": explicit_convert_wins,
         "max_conversion_aware_speedup": rq3_gain, "cases": reuse_details}))

    # RQ4: isolated versus concurrent HBM pressure. Distributed collectives are
    # intentionally not claimed by a one-GPU proxy.
    inversions, contention_gains, contention_details = [], [], []
    for case_id in case_ids:
        isolated = strategies(by.get((case_id, "decode_head_scan"), []))
        contended = strategies(by.get((case_id, "decode_under_hbm_contention"), []))
        if not {"NHD", "HND"}.issubset(isolated) or not {"NHD", "HND"}.issubset(contended):
            continue
        iso, iso_robust, _ = winner(isolated.values())
        con, con_robust, _ = winner(contended.values())
        selected_cost = contended[iso["strategy"]]["p50_ms"]
        gain = speedup(selected_cost, con["p50_ms"])
        contention_gains.append(gain)
        inversion = iso["strategy"] != con["strategy"] and iso_robust and con_robust
        if inversion:
            inversions.append(case_id)
        contention_details.append({"case_id": case_id, "isolated_winner": iso["strategy"],
                                   "contended_winner": con["strategy"],
                                   "robust_inversion": inversion, "speedup": gain})
    rq4_gain = max(contention_gains, default=1.0)
    result.append(make_record(
        "RQ4", status(bool(inversions)),
        f"Single-GPU HBM contention changes the robust layout winner in {len(inversions)} shapes.",
        status(rq4_gain > 1.03),
        f"A contention-aware chooser reaches up to {rq4_gain:.3f}x versus replaying the isolated winner under pressure.",
        ["runtime_empirical"],
        {"contention_inversion_cases": inversions, "max_contention_aware_speedup": rq4_gain,
         "cases": contention_details, "distributed_status": "out_of_scope_single_gpu"},
        ["TP/EP collectives, rank placement, MoE all-to-all, and multi-GPU overlap remain unverified."]))

    result.append(make_record(
        "RQ5", status(bool(consumer_conflicts)),
        f"The two consumers request opposite local layouts in {len(consumer_conflicts)} shapes, creating a concrete negotiation conflict.",
        status(bool(split_wins)),
        f"Jointly timing common and repaired split candidates finds {len(split_wins)} robust split wins and preserves {len(common_wins)} common-layout negative cases.",
        ["runtime_empirical"],
        {"conflict_cases": consumer_conflicts, "split_wins": split_wins,
         "common_wins": common_wins, "cases": multi_details},
        ["Consumers approximate draft/target access styles; a native speculative decoder run is a separate experiment."]))

    # RQ6: compose measured write/read/conversion costs into explicit workload
    # epochs.  Report this as a measured cost model, never direct runtime.
    drift_cases, controller_wins, epoch_details = [], [], []
    for case_id, item in primitive.items():
        conversions = strategies(by.get((case_id, "layout_conversion"), []))
        if not {"NHD_to_HND", "HND_to_NHD"}.issubset(conversions):
            continue
        costs = {}
        for layout in ("NHD", "HND"):
            costs[layout] = {
                "prefill": item["write"][layout]["p50_ms"] + item["token"][layout]["p50_ms"],
                "decode": item["write"][layout]["p50_ms"] + 16 * item["decode"][layout]["p50_ms"],
            }
        prefill_winner = min(costs, key=lambda x: costs[x]["prefill"])
        decode_winner = min(costs, key=lambda x: costs[x]["decode"])
        if prefill_winner != decode_winner:
            drift_cases.append(case_id)
        epochs = [
            Epoch("prefill", 1, {x: costs[x]["prefill"] for x in costs}),
            Epoch("short_decode", 1, {x: costs[x]["decode"] for x in costs}),
            Epoch("long_decode", 32, {x: costs[x]["decode"] for x in costs}),
            Epoch("next_prefill", 1, {x: costs[x]["prefill"] for x in costs}),
            Epoch("next_long_decode", 32, {x: costs[x]["decode"] for x in costs}),
        ]
        switch_cost = {
            ("NHD", "HND"): conversions["NHD_to_HND"]["p50_ms"],
            ("HND", "NHD"): conversions["HND_to_NHD"]["p50_ms"],
        }
        controlled = run_epoch_controller(epochs, ("NHD", "HND"), initial="NHD",
                                          switch_cost_ms=switch_cost, hysteresis_ratio=1.2)
        fixed = min(
            (fixed_epoch_policy(epochs, x) for x in ("NHD", "HND")),
            key=lambda item: item.total_ms,
        )
        aggressive = per_request_oracle(epochs, ("NHD", "HND"), initial="NHD",
                                        switch_cost_ms=switch_cost)
        gain = speedup(fixed.total_ms, controlled.total_ms)
        if gain > 1.03:
            controller_wins.append(case_id)
        epoch_details.append({"case_id": case_id, "prefill_winner": prefill_winner,
                              "decode_winner": decode_winner, "fixed_ms": fixed.total_ms,
                              "epochal_ms": controlled.total_ms,
                              "aggressive_ms": aggressive.total_ms,
                              "epochal_switches": controlled.switch_count,
                              "aggressive_switches": aggressive.switch_count,
                              "speedup_vs_fixed": gain})
    result.append(make_record(
        "RQ6", status(bool(drift_cases)),
        f"Prefill-like and decode-like regimes prefer different layouts in {len(drift_cases)} measured shape models.",
        status(bool(controller_wins)),
        f"A hysteretic epoch controller beats the best static layout by >3% in {len(controller_wins)} shapes while charging measured migration costs.",
        ["measured_cost_model"],
        {"drift_cases": drift_cases, "controller_win_cases": controller_wins,
         "cases": epoch_details},
        ["Epoch timings are a composition of measured primitives, not an end-to-end serving trace."]))

    # RQ7/RQ10: typed edge contracts and measured repair costs.
    planner_repairs, planner_gains, planner_details = [], [], []
    reps = {
        "NHD": Representation("NHD", ("token", "head", "dim"), persistent=True),
        "HND": Representation("HND", ("head", "token", "dim"), persistent=True),
    }
    for case_id, item in primitive.items():
        conversions = strategies(by.get((case_id, "layout_conversion"), []))
        if not {"NHD_to_HND", "HND_to_NHD"}.issubset(conversions):
            continue
        consumers = [
            Consumer("head_local", frozenset({"HND"}), {"HND": item["decode"]["HND"]["p50_ms"]}),
            Consumer("token_local", frozenset({"NHD"}), {"NHD": item["token"]["NHD"]["p50_ms"]}),
        ]
        repairs = [
            Repair("transpose", "NHD", "HND", conversions["NHD_to_HND"]["p50_ms"],
                   temporary_bytes=int(conversions["NHD_to_HND"]["bytes"]),
                   persistent_bytes=int(conversions["NHD_to_HND"]["bytes"] / 2)),
            Repair("transpose", "HND", "NHD", conversions["HND_to_NHD"]["p50_ms"],
                   temporary_bytes=int(conversions["HND_to_NHD"]["bytes"]),
                   persistent_bytes=int(conversions["HND_to_NHD"]["bytes"] / 2)),
        ]
        common = common_layout_plan(reps, consumers)
        plan = joint_representation_plan(reps, consumers, repairs,
                                         base_bytes=int(conversions["NHD_to_HND"]["bytes"] / 2))
        if common is None and plan is not None:
            planner_repairs.append(case_id)
        # Compare the exact plan with a forced NHD representation where the
        # head consumer must pay the same measured repair every use.
        forced = (item["token"]["NHD"]["p50_ms"] +
                  item["decode"]["HND"]["p50_ms"] +
                  conversions["NHD_to_HND"]["p50_ms"])
        gain = speedup(forced, plan.total_ms) if plan else 1.0
        planner_gains.append(gain)
        planner_details.append({"case_id": case_id, "direct_intersection": common is not None,
                                "repair_plan": None if plan is None else {
                                    "cost_ms": plan.total_ms, "storage": plan.storage,
                                    "repairs": [repair.name for repair in plan.repairs],
                                    "assignments": plan.consumer_representations},
                                "speedup_vs_forced_nhd_repair": gain})
    result.append(make_record(
        "RQ7", status(bool(planner_repairs)),
        f"Typed consumers have an empty direct-layout intersection in {len(planner_repairs)} measured cases.",
        status(bool(planner_repairs)),
        "Legality-first edge-contract synthesis finds a finite repair plan for every reported empty intersection.",
        ["measured_cost_model", "cpu_unit_test"],
        {"repaired_cases": planner_repairs, "cases": planner_details},
        ["The prototype exact search is intentionally finite and not a production compiler pass."]))

    # RQ8: actual full pipeline timings test pass ordering and fusion jointly.
    order_effects, fused_wins, fusion_gains, rope_details = [], [], [], []
    for case_id in case_ids:
        candidates = strategies(by.get((case_id, "rope_kv_pipeline"), []))
        required = {"rope_then_convert", "convert_then_rope", "fused_rope_native_HND"}
        if not required.issubset(candidates):
            continue
        a, b = candidates["rope_then_convert"], candidates["convert_then_rope"]
        order_ratio = max(a["p50_ms"], b["p50_ms"]) / min(a["p50_ms"], b["p50_ms"])
        if order_ratio > 1.05:
            order_effects.append(case_id)
        best, robust, margin = winner(candidates.values())
        separate_best = min(a["p50_ms"], b["p50_ms"])
        fused = candidates["fused_rope_native_HND"]["p50_ms"]
        fusion_gains.append(speedup(separate_best, fused))
        if robust and best["strategy"] == "fused_rope_native_HND":
            fused_wins.append(case_id)
        max_error = max(row["max_abs_error"] for row in candidates.values())
        rope_details.append({"case_id": case_id, "winner": best["strategy"],
                             "robust": robust, "margin": margin,
                             "order_ratio": order_ratio, "max_abs_error": max_error,
                             "fusion_speedup": speedup(separate_best, fused)})
    result.append(make_record(
        "RQ8", status(bool(order_effects)),
        f"Changing pass order changes median pipeline time by >5% in {len(order_effects)} shapes.",
        status(bool(fused_wins)),
        f"Jointly admitting the fused layout-native candidate gives {len(fused_wins)} robust wins; maximum speedup is {max(fusion_gains, default=1.0):.3f}x.",
        ["runtime_empirical"],
        {"order_effect_cases": order_effects, "fused_win_cases": fused_wins,
         "max_fusion_speedup": max(fusion_gains, default=1.0), "cases": rope_details}))

    # RQ9: compare one portable fixed policy with a per-case/per-consumer upper
    # bound.  This is explicitly an oracle ceiling, not a trained policy.
    global_cost = {"NHD": 0.0, "HND": 0.0}
    oracle_cost = 0.0
    layout_winners = Counter()
    observations = 0
    for item in primitive.values():
        for consumer in ("decode", "token", "write"):
            values = item[consumer]
            if not {"NHD", "HND"}.issubset(values):
                continue
            for layout in global_cost:
                global_cost[layout] += values[layout]["p50_ms"]
            best, robust, _ = winner(values.values())
            oracle_cost += best["p50_ms"]
            if robust:
                layout_winners[best["strategy"]] += 1
            observations += 1
    best_static_layout = min(global_cost, key=global_cost.get) if observations else "none"
    portable_gain = speedup(global_cost.get(best_static_layout, 1.0), oracle_cost) if observations else 1.0
    heterogeneity = len(layout_winners) > 1
    result.append(make_record(
        "RQ9", status(heterogeneity),
        f"Robust winners are heterogeneous ({dict(layout_winners)}) across {observations} shape/consumer points, so one fixed threshold/layout is not portable.",
        status(portable_gain > 1.03),
        f"The case-and-consumer oracle ceiling is {portable_gain:.3f}x over the best global static layout ({best_static_layout}); this bounds, but does not prove, a learned policy's gain.",
        ["runtime_empirical", "oracle_upper_bound"],
        {"robust_winner_counts": dict(layout_winners), "best_static_layout": best_static_layout,
         "best_static_total_ms": global_cost.get(best_static_layout),
         "oracle_total_ms": oracle_cost, "oracle_speedup": portable_gain},
        ["A portable predictor still requires train/validation splits across GPUs and framework versions."]))

    result.append(make_record(
        "RQ10", status(bool(planner_repairs)),
        f"Direct layout intersection is empty in {len(planner_repairs)} typed cross-consumer cases; rejecting the candidate would discard legal repaired executions.",
        status(bool(planner_repairs)),
        f"The repair graph restores feasibility in {len(planner_repairs)} cases after legality and memory filtering; observed maximum modeled gain over a forced repair point is {max(planner_gains, default=1.0):.3f}x.",
        ["measured_cost_model", "cpu_unit_test"],
        {"repaired_cases": planner_repairs,
         "max_speedup_vs_forced_repair": max(planner_gains, default=1.0),
         "cases": planner_details},
        ["Repair feasibility is proven; end-to-end native-framework integration is not."]))
    return result


def framework_matrix(output_dir: Path, source_audit: dict) -> list[dict]:
    source_frameworks = {
        framework
        for rq in source_audit.get("rq_coverage", {}).values()
        for framework in rq.get("frameworks", [])
    }
    matrix = []
    for framework, candidates in FRAMEWORK_ARTIFACTS.items():
        found = [name for name in candidates if artifact_has_measurement(output_dir / name)]
        source_names = (framework, *SOURCE_ALIASES.get(framework, ()))
        has_source = any(alias in source_frameworks for alias in source_names)
        matrix.append({
            "framework": framework,
            "source_static": has_source,
            "runtime_artifacts": found,
            "runtime_status": "measured" if found else "not_run_or_unavailable",
            "claim_scope": ("native_or_framework_generated" if found else
                            "source decision only" if has_source else "no evidence in this run"),
        })
    return matrix


def artifact_has_measurement(path: Path) -> bool:
    """Reject empty/header-only/error-only files as runtime measurements."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status", "success") == "success" and (
                        "p50_ms" in row or "throughput_requests_per_second" in row):
                    return True
            return False
        if path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                return any(any(key.endswith("_ms") and value not in (None, "")
                               for key, value in row.items())
                           for row in csv.DictReader(handle))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return False


def markdown(summary: dict) -> str:
    coverage = summary["coverage"]
    lines = [
        "# ref_talks RQ validation report",
        "",
        "## Evidence guardrail",
        "",
        "CUDA-reference results are controlled mechanism experiments, not native-framework timings. "
        "Source inspection proves that a decision exists, not that it is slow. Measured-cost models "
        "compose GPU primitives and are labeled separately from directly timed pipelines.",
        "",
        "## Shape coverage",
        "",
        f"- Cases timed: {coverage['case_count']}",
        f"- Distinct shapes: {coverage['distinct_shape_count']}",
        f"- Subgraphs: {', '.join(coverage['subgraphs']) or '-'}",
        f"- Phases: {', '.join(coverage['phases']) or '-'}",
        f"- Attention kinds: {json.dumps(coverage['attention_kinds'], ensure_ascii=False)}",
        f"- All-subgraph boundary sweep: {summary['all_subgraph_boundary_sweep']['case_count']} cases over "
        f"{', '.join(summary['all_subgraph_boundary_sweep']['subgraphs']) or '-'}",
        "",
        "## RQ verdicts",
        "",
        "| RQ | Problem | Proposed observation/solution | Evidence |",
        "|---|---|---|---|",
    ]
    for row in summary["rqs"]:
        problem = f"**{row['problem_status']}** — {row['problem_observation']}"
        solution = f"**{row['solution_status']}** — {row['solution_observation']}"
        lines.append(f"| {row['rq']} | {problem} | {solution} | {', '.join(row['evidence_levels']) or '-'} |")
    lines += ["", "## Framework evidence matrix", "",
              "| Framework | Immutable/source audit | Runtime artifact | Claim scope |",
              "|---|---|---|---|"]
    for row in summary["framework_matrix"]:
        lines.append(f"| {row['framework']} | {'yes' if row['source_static'] else 'no'} | "
                     f"{', '.join(row['runtime_artifacts']) or '-'} | {row['claim_scope']} |")
    lines += ["", "## Requirements audit", ""]
    for name, item in summary["requirements"].items():
        lines.append(f"- **{item['status']}** `{name}`: {item['detail']}")
    lines += ["", "## Remaining validity work", ""]
    for item in summary["remaining_work"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-csv", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--case-summary", type=Path)
    parser.add_argument("--boundary-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_csv(args.cuda_csv)
    source_audit = load_json(args.source_audit)
    case_summary = load_json(args.case_summary)
    boundary = summarize_boundary(load_jsonl(args.boundary_jsonl))
    rqs = analyze(rows)
    for row in rqs:
        if row["rq"] in {"RQ2", "RQ3", "RQ7", "RQ9", "RQ10"}:
            row["metrics"]["all_subgraph_boundary_sweep"] = boundary
    shapes = {(int(r["tokens"]), int(r["query_heads"]), int(r["kv_heads"]), int(r["head_dim"])) for r in rows}
    coverage = {
        "case_count": len({r["case_id"] for r in rows}),
        "distinct_shape_count": len(shapes),
        "subgraphs": sorted({r["subgraph"] for r in rows}),
        "phases": sorted({r["phase"] for r in rows}),
        "attention_kinds": dict(Counter(r["attention_kind"] for r in rows)),
        "selected_manifest": case_summary,
        "all_subgraph_boundary_sweep": boundary,
    }
    matrix = framework_matrix(args.output_dir, source_audit)
    rq_complete = all(row["problem_status"] != "not_run" for row in rqs)
    expected_subgraphs = {"gqa", "sliding_attention", "sparse_attention", "mla",
                          "swiglu", "moe", "mamba2", "linear_attention"}
    boundary_complete = (set(boundary["subgraphs"]) == expected_subgraphs and
                         all(count >= (1 if case_summary.get("per_group") == 1 else 2)
                             for count in boundary["shape_count_by_subgraph"].values()))
    multi_shape = coverage["distinct_shape_count"] >= 3 and len(coverage["phases"]) >= 2
    framework_runtime = [row["framework"] for row in matrix if row["runtime_status"] == "measured"]
    requested_frameworks = {"vLLM", "SGLang", "CUTLASS", "Triton", "TVM", "Hexcute"}
    measured_requested = requested_frameworks.intersection(framework_runtime)
    missing_requested = sorted(requested_frameworks - measured_requested)
    requirements = {
        "ref_talks_source_chain": {
            "status": "PASS" if source_audit.get("source_chain_pass") else "MISSING",
            "detail": "65 canonical rules and 50 forensic rows expected; URL fetching is optional.",
        },
        "common_llm_multiple_shapes": {
            "status": "PASS" if multi_shape and boundary_complete else "PARTIAL" if multi_shape else "MISSING",
            "detail": (f"attention mechanisms: {coverage['case_count']} cases / {coverage['distinct_shape_count']} contracts; "
                       f"all-subgraph boundary sweep: {boundary['case_count']} cases, subgraphs={boundary['subgraphs']}."),
        },
        "rq1_rq10_controlled_validation": {
            "status": "PASS" if rq_complete else "MISSING",
            "detail": "Every RQ has a falsifiable status; not_supported is a valid negative result.",
        },
        "native_framework_runtime": {
            "status": ("PASS" if measured_requested == requested_frameworks else
                       "PARTIAL" if measured_requested else "MISSING"),
            "detail": (f"Requested frameworks measured: {', '.join(sorted(measured_requested)) or 'none'}; "
                       f"still missing: {', '.join(missing_requested) or 'none'}. "
                       "Source evidence is not counted as runtime."),
        },
        "all_framework_all_subgraph_layout_sweep": {
            "status": "PARTIAL" if boundary["status"] == "measured" else "MISSING",
            "detail": ("PyTorch boundary-layout alternatives cover all eight subgraphs when successful; "
                       "serving engines and kernel/compiler libraries do not expose an equivalent executable "
                       "for every subgraph, so unsupported framework×subgraph cells remain explicit rather than substituted."),
        },
        "distributed_rq4": {
            "status": "MISSING",
            "detail": "One physical GPU cannot validate TP/EP rank placement or all-to-all overlap.",
        },
        "cross_gpu_portability_rq9": {
            "status": "MISSING",
            "detail": "Requires held-out GPU generations and framework versions; this run only estimates an oracle ceiling.",
        },
    }
    summary = {
        "schema_version": 1,
        "coverage": coverage,
        "all_subgraph_boundary_sweep": boundary,
        "rqs": rqs,
        "framework_matrix": matrix,
        "requirements": requirements,
        "remaining_work": [
            "Run native SGLang/vLLM with a real model path to attribute serving-level behavior.",
            "Run at least two GPUs for TP/EP/all-to-all and placement-coupled RQ4 claims.",
            "Repeat on held-out GPU architectures for RQ9 portability rather than fitting and evaluating on one A10.",
            "Integrate the typed contract/repair planner into each framework before claiming framework-level speedup.",
            "Controlled CUDA attention covers the KV-layout mechanism; PyTorch/Triton representative runs provide external validity for SwiGLU, MoE, MLA, Mamba2, linear and sparse attention when available.",
        ],
    }
    (args.output_dir / "rq_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "rq_observations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rqs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "RQ_VALIDATION_REPORT.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({"report": str(args.output_dir / "RQ_VALIDATION_REPORT.md"),
                      "cases": coverage["case_count"], "distinct_shapes": coverage["distinct_shape_count"],
                      "rq_status": {row["rq"]: [row["problem_status"], row["solution_status"]] for row in rqs}},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
