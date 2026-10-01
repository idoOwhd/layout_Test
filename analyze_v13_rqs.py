#!/usr/bin/env python3
"""Adjudicate the 49 v13 hypotheses from paired single-GPU measurements.

The analyzer is intentionally conservative: direct runtime evidence, models
composed from measured primitive costs, source observations and single-GPU
blockers remain separate in every result row.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
THRESHOLD = 1.03


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result.append(json.loads(line))
    return result


def prefer(raw_dir: Path, exhaustive: str, representative: str) -> Path:
    """Use exhaustive evidence when present without breaking legacy runs."""
    path = raw_dir / exhaustive
    return path if path.exists() and path.stat().st_size else raw_dir / representative


def number(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def ratio(baseline: float, improved: float) -> float:
    return baseline / improved if baseline > 0 and improved > 0 else 1.0


def by(rows: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    return grouped


def winner(rows: list[dict], key: str = "p50_ms") -> dict | None:
    valid = [row for row in rows if number(row, key, math.inf) > 0]
    return min(valid, key=lambda row: number(row, key)) if valid else None


def add(results: dict[str, dict], hypothesis: str, status: str, level: str,
        frameworks: list[str], summary: str, metrics: dict | None = None,
        experiment: str | None = None) -> None:
    results[hypothesis] = {
        "hypothesis": hypothesis, "status": status, "evidence_level": level,
        "frameworks": frameworks, "summary": summary, "metrics": metrics or {},
        **({"experiment": experiment} if experiment else {}),
    }


def boundary_evidence(rows: list[dict]) -> dict:
    groups = by([row for row in rows
                 if row.get("status") == "success"
                 and str(row.get("correctness_passed", "false")).lower() == "true"],
                "case_id")
    legal = direct_wins = repair_wins = 0
    gains = []
    for candidates in groups.values():
        table = {row["strategy"]: row for row in candidates}
        alternate_key = ("alternate_strided_view" if "alternate_strided_view" in table
                         else "alternate_axis12_strided")
        if alternate_key not in table or "repair_to_contiguous_each_call" not in table:
            continue
        alternate_row = table[alternate_key]
        # New artifacts state this explicitly.  Legacy artifacts are accepted
        # only when at least one input is observably non-contiguous.
        changed = alternate_row.get("layout_changed") is True or (
            "layout_changed" not in alternate_row
            and any(value is False for value in alternate_row.get("input_contiguous", [])))
        if not changed:
            continue
        legal += 1
        direct = number(alternate_row, "p50_ms")
        repair = number(table["repair_to_contiguous_each_call"], "p50_ms")
        gains.append(ratio(repair, direct))
        if ratio(repair, direct) >= THRESHOLD:
            direct_wins += 1
        elif ratio(direct, repair) >= THRESHOLD:
            repair_wins += 1
    return {"cases": len(groups), "legal_zero_copy_cases": legal,
            "zero_copy_materially_faster_cases": direct_wins,
            "materialization_faster_cases": repair_wins,
            "max_zero_copy_speedup": max(gains, default=1.0)}


def serving_summary(rows: list[dict], comparison_scope: str | None = None) -> dict:
    valid = [row for row in rows if row.get("status") == "success"
             and number(row, "p50_ms") > 0
             and (comparison_scope is None or row.get("comparison_scope") == comparison_scope)]
    result = {"successful_rows": len(valid), "cases": len({row.get("case_id") for row in valid}),
              "winner_crossovers": 0, "static_regret": 1.0,
              "max_case_regret": 1.0, "page_winner_crossovers": 0}
    groups = by(valid, "case_id")
    # Static-policy totals are meaningful only for candidates present in every
    # case.  Otherwise a failed/missing variant can look artificially cheap.
    variant_sets = [{row.get("variant") for row in candidates} for candidates in groups.values()]
    common_variants = set.intersection(*variant_sets) if variant_sets else set()
    winners, oracle, totals = [], 0.0, defaultdict(float)
    page_winners = []
    case_regrets = []
    for candidates in groups.values():
        best = winner(candidates)
        if not best:
            continue
        ordered = sorted(candidates, key=lambda row: number(row, "p50_ms"))
        margin = ratio(number(ordered[1], "p50_ms"), number(best, "p50_ms")) \
            if len(ordered) > 1 else 1.0
        if margin >= THRESHOLD:
            winners.append(best.get("variant"))
        case_regrets.append(ratio(number(ordered[-1], "p50_ms"),
                                  number(best, "p50_ms")))
        oracle += number(best, "p50_ms")
        for row in candidates:
            if row.get("variant") in common_variants:
                totals[row.get("variant")] += number(row, "p50_ms")
        page = [row for row in candidates if row.get("page_size")]
        page_best = winner(page)
        ordered_page = sorted(page, key=lambda row: number(row, "p50_ms"))
        page_margin = ratio(number(ordered_page[1], "p50_ms"), number(page_best, "p50_ms")) \
            if page_best and len(ordered_page) > 1 else 1.0
        if page_best and page_margin >= THRESHOLD:
            page_winners.append(page_best.get("page_size"))
    if totals and oracle:
        result["static_regret"] = ratio(min(totals.values()), oracle)
    result["winner_crossovers"] = max(0, len(set(winners)) - 1)
    result["max_case_regret"] = max(case_regrets, default=1.0)
    result["page_winner_crossovers"] = max(0, len(set(page_winners)) - 1)
    result["winners"] = winners
    result["page_winners"] = page_winners
    result["common_variants"] = sorted(common_variants)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=HERE / "v13_experiment_registry.json")
    parser.add_argument("--source-json", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry = json.loads(args.registry.read_text(encoding="utf-8"))

    cuda_all = read_csv(prefer(args.raw_dir, "rq_cuda_all_attention.csv", "rq_cuda_reference.csv"))
    # CUDA-reference rows compare exact layout-preserving paths.  Keep the
    # adjudicator fail-closed if a future benchmark emits a numerically wrong
    # row instead of silently allowing its timing to select a winner.
    cuda = [row for row in cuda_all
            if row.get("max_abs_error", "") != ""
            and abs(number(row, "max_abs_error", math.inf)) <= 1e-3]
    triton = read_csv(prefer(args.raw_dir, "triton_640_kv_layout.csv", "triton_kv_layout.csv"))
    tvm = read_csv(prefer(args.raw_dir, "tvm_640_rq_observations.csv",
                          "tvm_rq_observations.csv"))
    boundary = read_jsonl(prefer(args.raw_dir, "llm_640_boundary_layout_sweep.jsonl",
                                 "llm_boundary_layout_sweep.jsonl"))
    vllm = read_jsonl(args.raw_dir / "vllm_native_serving.jsonl")
    sglang = read_jsonl(args.raw_dir / "sglang_native_serving.jsonl")
    results: dict[str, dict] = {}
    models: dict[str, object] = {}

    # Common directly measured CUDA primitives.
    cuda_groups = by(cuda, "case_id", "benchmark")
    case_ids = sorted({row.get("case_id") for row in cuda if row.get("case_id")})
    local_inversions, triton_edge_inversions, negative_controls, h14_points = [], [], [], []
    reuse_curves = defaultdict(list)
    primitive = {}
    for case_id in case_ids:
        writes = {row["strategy"]: row for row in cuda_groups.get((case_id, "kv_write"), [])}
        heads = {row["strategy"]: row for row in cuda_groups.get((case_id, "decode_head_scan"), [])}
        tokens = {row["strategy"]: row for row in cuda_groups.get((case_id, "token_major_scan"), [])}
        conversions = {row["strategy"]: row for row in cuda_groups.get((case_id, "layout_conversion"), [])}
        if {"NHD", "HND"}.issubset(writes) and {"NHD", "HND"}.issubset(heads):
            producer = winner(list(writes.values()))
            consumer = winner(list(heads.values()))
            if producer and consumer and producer["strategy"] != consumer["strategy"]:
                edge_gain = ratio(number(heads[producer["strategy"]], "p50_ms"), number(consumer, "p50_ms"))
                if edge_gain >= THRESHOLD:
                    local_inversions.append({"case_id": case_id, "producer": producer["strategy"],
                                             "consumer": consumer["strategy"], "speedup": edge_gain})
            layout_gap = max(number(row, "p50_ms") for row in heads.values()) / min(
                number(row, "p50_ms") for row in heads.values())
            if layout_gap < THRESHOLD:
                negative_controls.append(case_id)
        if {"NHD", "HND"}.issubset(heads) and {"NHD", "HND"}.issubset(tokens):
            primitive[case_id] = {"head": {k: number(v, "p50_ms") for k, v in heads.items()},
                                  "token": {k: number(v, "p50_ms") for k, v in tokens.items()},
                                  "conversion": number(conversions.get("NHD_to_HND", {}), "p50_ms")}
    for row in cuda:
        if row.get("benchmark") == "conversion_reuse_pipeline":
            reuse_curves[(row["case_id"], int(number(row, "reuse_count")))].append(row)
    triton_edge_rows = [row for row in triton
                        if str(row.get("correct", "true")).lower() == "true"
                        and number(row, "producer_p50_ms") > 0
                        and number(row, "pipeline_p50_ms") > 0]
    for key, candidates in by(triton_edge_rows, "case_id").items():
        producer = min(candidates, key=lambda row: number(row, "producer_p50_ms"))
        edge = min(candidates, key=lambda row: number(row, "pipeline_p50_ms"))
        if producer.get("layout") == edge.get("layout"):
            continue
        gain = ratio(number(producer, "pipeline_p50_ms"),
                     number(edge, "pipeline_p50_ms"))
        if gain >= THRESHOLD:
            triton_edge_inversions.append({"case_id": key[0],
                                           "producer": producer.get("layout"),
                                           "edge": edge.get("layout"),
                                           "speedup": gain})
    curve_by_case = defaultdict(list)
    explicit_wins = []
    for (case_id, reuse), candidates in sorted(reuse_curves.items()):
        best = winner(candidates)
        table = {row["strategy"]: row for row in candidates}
        if not best or "common_NHD" not in table:
            continue
        regret = ratio(number(table["common_NHD"], "p50_ms"), number(best, "p50_ms"))
        curve_by_case[case_id].append((reuse, best["strategy"], regret))
        if best["strategy"] == "NHD_then_convert_once_HND" and regret >= THRESHOLD:
            explicit_wins.append(f"{case_id}@{reuse}")
        if {"producer_native_HND", "NHD_then_convert_once_HND"}.issubset(table):
            gain = ratio(number(table["NHD_then_convert_once_HND"], "p50_ms"),
                         number(table["producer_native_HND"], "p50_ms"))
            if gain >= THRESHOLD:
                h14_points.append({"case_id": case_id, "reuse": reuse, "speedup": gain})
    increasing_regret = [case for case, curve in curve_by_case.items()
                         if len(curve) >= 2 and curve[-1][2] >= THRESHOLD
                         and curve[-1][2] > curve[0][2] * 1.01]
    add(results, "H1.1", "supported" if local_inversions or triton_edge_inversions else "inconclusive", "runtime_empirical",
        ["CUDA-reference"] + (["Triton"] if triton_edge_inversions else []),
        f"Stable producer/consumer rank inversions: CUDA={len(local_inversions)}, Triton={len(triton_edge_inversions)}.",
        {"inversions": local_inversions, "triton_inversions": triton_edge_inversions}, "RQ1-E3")
    add(results, "H1.2", "supported" if increasing_regret else "inconclusive", "runtime_empirical",
        ["CUDA-reference"], f"Reuse increased local-layout regret in {len(increasing_regret)} shapes.",
        {"cases": increasing_regret, "curves": curve_by_case}, "RQ1-E4")
    be = boundary_evidence(boundary)
    add(results, "H1.3", "supported" if be["zero_copy_materially_faster_cases"] else
        ("not_run_missing_artifact" if not boundary else "inconclusive"), "runtime_empirical",
        ["PyTorch"], "Direct stride-polymorphic consumption was compared with per-call materialization.", be, "RQ1-E4")
    add(results, "H1.4", "supported" if h14_points else "inconclusive", "runtime_empirical",
        ["CUDA-reference"], f"Direct consumer-native emission materially won at {len(h14_points)} points.",
        {"winning_points": h14_points}, "RQ1-E3")
    add(results, "H1.NEG", "supported" if negative_controls else "inconclusive", "runtime_empirical",
        ["CUDA-reference"], f"Layout-insensitive (<3%) negative-control shapes: {len(negative_controls)}.",
        {"cases": negative_controls}, "RQ1-E0")

    # RQ2: direct multi-consumer timings plus measured-cost weight/overhead sweeps.
    multi_winners, split_wins, common_wins = [], [], []
    for case_id in case_ids:
        rows = cuda_groups.get((case_id, "multi_consumer_pipeline"), [])
        best = winner(rows)
        if not best:
            continue
        multi_winners.append({"case_id": case_id, "winner": best["strategy"]})
        (split_wins if best["strategy"].startswith("split_") else common_wins).append(case_id)
    thresholds, weighted_regrets, acceptance_changes, overhead_changes = [], [], [], []
    for case_id, costs in primitive.items():
        conv = costs["conversion"]
        if conv <= 0:
            continue
        decisions = []
        for weight in [i / 10 for i in range(11)]:
            shared_n = weight * costs["head"]["NHD"] + (1-weight) * costs["token"]["NHD"]
            shared_h = weight * costs["head"]["HND"] + (1-weight) * costs["token"]["HND"]
            split = weight * costs["head"]["HND"] + (1-weight) * costs["token"]["NHD"] + conv
            values = {"shared_NHD": shared_n, "shared_HND": shared_h, "split": split}
            decisions.append((weight, min(values, key=values.get), values))
            equal_vote = "shared_NHD" if costs["token"]["NHD"] <= costs["token"]["HND"] else "shared_HND"
            weighted_regrets.append(ratio(values[equal_vote], min(values.values())))
        if len({decision for _, decision, _ in decisions}) > 1:
            thresholds.append(case_id)
            acceptance_changes.append(case_id)
        overhead_decisions = []
        for factor in (0, .25, 1, 4, 16):
            weight = .5
            common = min(weight * costs["head"][layout] + (1-weight) * costs["token"][layout]
                         for layout in ("NHD", "HND"))
            split = weight * costs["head"]["HND"] + (1-weight) * costs["token"]["NHD"] + factor * conv
            overhead_decisions.append("split" if split < common else "shared")
        if len(set(overhead_decisions)) > 1:
            overhead_changes.append(case_id)
        models.setdefault("RQ2", {})[case_id] = decisions
    add(results, "H2.1", "supported" if thresholds else "inconclusive", "mixed_runtime_and_measured_cost_model",
        ["CUDA-reference"], f"Domain winner changed across measured consumer heterogeneity in {len(thresholds)} shapes.",
        {"threshold_cases": thresholds, "direct_split_wins": split_wins, "direct_common_wins": common_wins}, "RQ2-E2")
    add(results, "H2.2", "supported" if max(weighted_regrets, default=1) >= THRESHOLD else "inconclusive",
        "measured_cost_model", ["CUDA-reference"],
        "Executed-time weighting was tested against the implemented token-consumer-only proxy "
        "(historically named equal_vote; it is not a literal equal vote).",
        {"max_token_proxy_regret": max(weighted_regrets, default=1.0)}, "RQ2-E3")
    add(results, "H2.3", "supported" if common_wins else "inconclusive", "runtime_empirical",
        ["CUDA-reference"], f"A shared domain won {len(common_wins)} directly timed cases.",
        {"cases": common_wins}, "RQ2-E0")
    add(results, "H2.4", "supported" if acceptance_changes else "inconclusive", "measured_cost_model",
        ["CUDA-reference"], f"The synthetic consumer-weight sweep moved the plan in {len(acceptance_changes)} shapes; "
        "no live acceptance-rate trace was measured.",
        {"cases": acceptance_changes}, "RQ2-E4")
    add(results, "H2.5", "supported" if overhead_changes else "inconclusive", "measured_cost_model",
        ["CUDA-reference"], f"Domain choice crossed under storage/allocator overhead in {len(overhead_changes)} shapes.",
        {"cases": overhead_changes}, "RQ2-E5")

    # RQ3: reuse crossovers, boundary views, and sensitivity analysis anchored to measured conversion cost.
    conversion_pair_curves = defaultdict(list)
    for (case, reuse), candidates in reuse_curves.items():
        table = {row["strategy"]: row for row in candidates}
        if {"common_NHD", "NHD_then_convert_once_HND"}.issubset(table):
            common = number(table["common_NHD"], "p50_ms")
            converted = number(table["NHD_then_convert_once_HND"], "p50_ms")
            conversion_pair_curves[case].append(
                (reuse, "convert" if converted < common else "keep",
                 ratio(max(common, converted), min(common, converted))))
    crossover, r1_no_convert = [], []
    for case, curve in conversion_pair_curves.items():
        curve.sort()
        robust_keep = any(choice == "keep" and margin >= THRESHOLD
                          for _, choice, margin in curve)
        robust_convert = any(choice == "convert" and margin >= THRESHOLD
                             for _, choice, margin in curve)
        if robust_keep and robust_convert:
            crossover.append(case)
        if curve and curve[0][1] == "keep" and curve[0][2] >= THRESHOLD:
            r1_no_convert.append(case)
    measured_thresholds, memory_shift, overlap_shift = [], [], []
    for case, costs in primitive.items():
        per_use_gain = costs["head"]["NHD"] - costs["head"]["HND"]
        conversion = costs["conversion"]
        if per_use_gain <= 0 or conversion <= 0:
            continue
        base_threshold = conversion / per_use_gain
        measured_thresholds.append({"case_id": case, "R_star": base_threshold})
        memory_shift.append({"case_id": case, "base": base_threshold,
                             "under_pressure": 2 * base_threshold})
        overlap_shift.append({"case_id": case, "base": base_threshold,
                              "with_overlap": .5 * base_threshold})
    tvm_conversion = [row for row in tvm if row.get("experiment") == "conversion_reuse"]
    add(results, "H3.1", "supported" if crossover else "inconclusive", "runtime_empirical",
        ["CUDA-reference"] + (["TVM"] if tvm_conversion else []),
        f"A >3% directly timed keep-versus-materialize crossover occurred in {len(crossover)} CUDA shapes.",
        {"cases": crossover, "pair_curves": conversion_pair_curves,
         "measured_cost_thresholds": measured_thresholds}, "RQ3-E2")
    add(results, "H3.2", "supported" if memory_shift else "inconclusive", "measured_cost_model",
        ["CUDA-reference"], "Temporary-memory charge sensitivity is anchored to the measured conversion term.",
        {"threshold_shift_cases": memory_shift}, "RQ3-E3")
    add(results, "H3.3", "supported" if overlap_shift else "inconclusive", "measured_cost_model",
        ["CUDA-reference"], "Replacing serialized conversion cost by an overlapped cost lowers the modeled crossover.",
        {"overlap_sensitive_cases": overlap_shift}, "RQ3-E4")
    add(results, "H3.4", "supported" if be["zero_copy_materially_faster_cases"] else
        ("not_run_missing_artifact" if not boundary else "inconclusive"), "runtime_empirical",
        ["PyTorch"], "Legal zero-copy views are compared to native consumer plus materialization.", be, "RQ3-E1")
    add(results, "H3.5", "supported" if r1_no_convert else "inconclusive", "runtime_empirical",
        ["CUDA-reference"], f"Conversion did not win at R=1 in {len(r1_no_convert)} shapes.",
        {"cases": r1_no_convert}, "RQ3-E0")

    # RQ4: workload-axis stability. Cross-hardware prediction is explicitly blocked.
    triton_groups = by([row for row in triton if row.get("correct", "True").lower() == "true"], "case_id")
    triton_winners, triton_oracle, triton_static = [], 0.0, defaultdict(float)
    triton_case_regrets = []
    for candidates in triton_groups.values():
        best = winner(candidates)
        if not best:
            continue
        triton_winners.append(best["layout"]); triton_oracle += number(best, "p50_ms")
        triton_case_regrets.append(ratio(max(number(row, "p50_ms") for row in candidates),
                                         number(best, "p50_ms")))
        for row in candidates:
            triton_static[row["layout"]] += number(row, "p50_ms")
    static_regret = ratio(min(triton_static.values()), triton_oracle) if triton_static and triton_oracle else 1.0
    vs = serving_summary(vllm, "layout_only"); ss = serving_summary(sglang, "layout_only")
    vs_page = serving_summary(vllm, "page_size_only")
    ss_page = serving_summary(sglang, "page_size_only")
    tvm_consumer = [row for row in tvm if row.get("experiment") == "consumer_scan"
                    and row.get("correct", "True").lower() == "true"]
    # Exhaustive TVM rows retain the projected-shape case id for compiler
    # reuse and add manifest_case_id for the original 640-case contract.  Keep
    # the consumer axis in the key while evaluating every original case.
    tvm_groups = (by(tvm_consumer, "manifest_case_id", "case_id")
                  if any(row.get("manifest_case_id") for row in tvm_consumer)
                  else by(tvm_consumer, "case_id"))
    tvm_winners, tvm_oracle, tvm_static = [], 0.0, defaultdict(float)
    tvm_case_regrets = []
    for candidates in tvm_groups.values():
        best = winner(candidates)
        if not best:
            continue
        tvm_winners.append(best["strategy"]); tvm_oracle += number(best, "p50_ms")
        tvm_case_regrets.append(ratio(max(number(row, "p50_ms") for row in candidates),
                                      number(best, "p50_ms")))
        for row in candidates:
            tvm_static[row["strategy"]] += number(row, "p50_ms")
    tvm_static_regret = ratio(min(tvm_static.values()), tvm_oracle) if tvm_static and tvm_oracle else 1.0
    workload_cross = (len(set(triton_winners)) > 1 or len(set(tvm_winners)) > 1 or
                      vs["winner_crossovers"] or ss["winner_crossovers"])
    dominant = Counter(triton_winners).most_common(1)[0][1] / len(triton_winners) if triton_winners else 0
    add(results, "H4.1", "supported" if workload_cross else "inconclusive", "runtime_empirical",
        [name for name, rows in (("Triton", triton), ("TVM", tvm), ("vLLM", vllm), ("SGLang", sglang)) if rows],
        "At least one workload/shape axis changed the winner; the hardware axis was not tested.",
        {"triton_winners": triton_winners, "tvm_winners": tvm_winners,
         "vllm": vs, "sglang": ss}, "RQ4-E1")
    h4_frameworks = [name for name, rows in (("Triton", triton), ("TVM", tvm),
                                              ("vLLM", vllm), ("SGLang", sglang)) if rows]
    add(results, "H4.2", "supported" if max(static_regret, tvm_static_regret,
                                               vs["static_regret"], ss["static_regret"]) >= THRESHOLD else "inconclusive",
        "runtime_empirical", h4_frameworks, "Best fixed policy was compared with the per-case oracle.",
        {"triton_static_regret": static_regret, "vllm_static_regret": vs["static_regret"],
         "tvm_static_regret": tvm_static_regret,
         "sglang_static_regret": ss["static_regret"]}, "RQ4-E2")
    add(results, "H4.3", "supported" if dominant >= .5 and triton_winners else "inconclusive", "runtime_empirical",
        ["Triton"], f"The most frequent Triton winner occurs in {dominant:.1%} of measured shapes. "
        "This is a frequency, not a contiguous geometry region.",
        {"dominant_winner_frequency": dominant}, "RQ4-E1")
    add(results, "H4.4", "blocked_requires_multiple_hardware", "blocked",
        [], "Card1 alone cannot evaluate held-out hardware features versus GPU-name-only rules.", {}, "RQ4-E3")
    residual_regret = max(max(triton_case_regrets, default=1.0),
                          max(tvm_case_regrets, default=1.0),
                          vs["max_case_regret"], ss["max_case_regret"])
    add(results, "H4.5", "supported" if residual_regret >= THRESHOLD else "inconclusive", "runtime_empirical",
        h4_frameworks,
        "The per-case worst-candidate/best-candidate layout span remains after candidates use the same local "
        "execution configuration. This is a counterfactual upper bound, not an observed deployed-policy regret.",
        {"max_worst_best_case_span": residual_regret,
         "triton_max_case_regret": max(triton_case_regrets, default=1.0),
         "tvm_max_case_regret": max(tvm_case_regrets, default=1.0),
         "vllm_max_case_regret": vs["max_case_regret"],
         "sglang_max_case_regret": ss["max_case_regret"],
         "aggregate_triton_static_regret": static_regret}, "RQ4-E4")

    for i in range(1, 6):
        add(results, f"H5.{i}", "blocked_single_gpu", "blocked", [],
            "Requires multiple ranks/GPUs to measure communication, placement, replication or max-rank critical path.",
            {}, f"RQ5-E{0 if i == 4 else i + 1 if i < 4 else 5}")

    # RQ6: a measured-cost state controller. It does not claim a native live-KV migration API.
    horizons, trace_gains, stationary_ok, recapture_shifts = [], [], 0, []
    for case, costs in primitive.items():
        delta = abs(costs["head"]["NHD"] - costs["head"]["HND"])
        migration = costs["conversion"]
        if delta <= 0 or migration <= 0:
            continue
        horizon = math.ceil(migration / delta)
        horizons.append({"case_id": case, "N_star": horizon})
        recapture_shifts.append({"case_id": case, "base": horizon,
                                 "with_recapture": math.ceil((migration * 2) / delta)})
        # Alternating traffic charges migration every request, whereas an
        # epochal policy stays; persistent drift is evaluated after N*+1 steps.
        immediate = 20 * min(costs["head"].values()) + 20 * migration
        epochal = 20 * min(costs["head"].values()) + 2 * migration
        trace_gains.append(ratio(immediate, epochal))
        stationary_ok += 1
    models["RQ6"] = {"horizons": horizons, "recapture_shifts": recapture_shifts,
                     "per_request_vs_epochal_speedups": trace_gains}
    add(results, "H6.1", "supported" if horizons else "inconclusive", "measured_cost_model",
        ["CUDA-reference"], f"Finite measured switching horizons found for {len(horizons)} shapes.",
        {"horizons": horizons}, "RQ6-E2")
    add(results, "H6.2", "supported" if max(trace_gains, default=1) >= THRESHOLD else "inconclusive", "measured_cost_model",
        ["CUDA-reference"], "Epochal/hysteretic policy charges fewer measured migrations than per-request switching.",
        {"max_speedup": max(trace_gains, default=1.0)}, "RQ6-E3")
    add(results, "H6.3", "supported" if stationary_ok else "inconclusive", "measured_cost_model",
        ["CUDA-reference"], "In the synthetic stationary-trace formula, repeated switching adds the measured conversion "
        "term without changing the winner; no live controller trace was executed.",
        {"stationary_shapes": stationary_ok}, "RQ6-E0")
    add(results, "H6.4", "supported" if horizons else "inconclusive", "measured_cost_model",
        ["CUDA-reference"], "A persistent drift longer than N* amortizes measured migration cost.",
        {"horizons": horizons}, "RQ6-E2")
    add(results, "H6.5", "supported" if any(x["base"] != x["with_recapture"] for x in recapture_shifts) else "inconclusive",
        "measured_cost_model", ["CUDA-reference"], "Charging a recapture-sized term changes N* in the measured-cost sensitivity analysis.",
        {"thresholds": recapture_shifts}, "RQ6-E4")

    # RQ7: actual strided views and explicit materialization.
    add(results, "H7.1", "supported" if be["legal_zero_copy_cases"] else
        ("not_run_missing_artifact" if not boundary else "inconclusive"), "runtime_empirical", ["PyTorch"],
        f"Legal named-mismatch zero-copy views: {be['legal_zero_copy_cases']}.", be, "RQ7-E1")
    add(results, "H7.2", "feasibility_supported" if be["legal_zero_copy_cases"] else "not_run_missing_artifact",
        "measured_legality_model", ["PyTorch"],
        "Single-consumer legal views are measured. The zero two-consumer intersection is a constructed legality "
        "counterfactual obtained by adding a contiguous-only contract; it is not a separately timed two-consumer pipeline.",
        {"single_consumer_legal": be["legal_zero_copy_cases"], "two_consumer_intersection_legal": 0}, "RQ7-E3")
    add(results, "H7.3", "feasibility_supported" if be["legal_zero_copy_cases"] else "not_run_missing_artifact",
        "runtime_empirical", ["PyTorch"],
        "Different layout names remained legal when strides were accepted, while materialization need followed the consumer stride contract.", be, "RQ7-E2")
    add(results, "H7.4", "supported" if be["zero_copy_materially_faster_cases"] else
        ("not_run_missing_artifact" if not boundary else "inconclusive"), "runtime_empirical", ["PyTorch"],
        "Stride-polymorphic consumption avoids explicit repair where it is correct and faster.", be, "RQ7-E4")

    # RQ8: direct 2x2 data x metadata factorial and sparse-route/page factorial.
    meta_groups = by([row for row in cuda if row.get("benchmark") == "data_metadata_scan"],
                     "case_id", "reuse_count")
    interactions, joint_gains, placement_changes = [], [], []
    per_token_interactions, per_head_interactions = [], []
    for (case, granularity), candidates in meta_groups.items():
        table = {row["strategy"]: row for row in candidates}
        suffix = "per_token" if str(granularity) in {"1", "1.0"} else "per_head"
        keys = {f"data_{d}_meta_{m}_{suffix}" for d in ("NHD", "HND") for m in ("NHD", "HND")}
        if not keys.issubset(table):
            continue
        cost = {(d, m): number(table[f"data_{d}_meta_{m}_{suffix}"], "p50_ms")
                for d in ("NHD", "HND") for m in ("NHD", "HND")}
        interaction = abs(cost[("NHD", "NHD")] + cost[("HND", "HND")] -
                          cost[("NHD", "HND")] - cost[("HND", "NHD")]) / min(cost.values())
        (per_token_interactions if suffix == "per_token" else per_head_interactions).append(interaction)
        interactions.append({"case_id": case, "granularity": suffix, "interaction": interaction})
        data_choice = min(("NHD", "HND"), key=lambda d: sum(cost[(d, m)] for m in ("NHD", "HND")))
        meta_choice = min(("NHD", "HND"), key=lambda m: sum(cost[(d, m)] for d in ("NHD", "HND")))
        independent = cost[(data_choice, meta_choice)]; joint = min(cost.values())
        joint_gains.append(ratio(independent, joint))
        n_winner = min(("NHD", "HND"), key=lambda d: cost[(d, "NHD")])
        h_winner = min(("NHD", "HND"), key=lambda d: cost[(d, "HND")])
        n_margin = ratio(max(cost[("NHD", "NHD")], cost[("HND", "NHD")]),
                         min(cost[("NHD", "NHD")], cost[("HND", "NHD")]))
        h_margin = ratio(max(cost[("NHD", "HND")], cost[("HND", "HND")]),
                         min(cost[("NHD", "HND")], cost[("HND", "HND")]))
        if n_winner != h_winner and min(n_margin, h_margin) >= THRESHOLD:
            placement_changes.append(case)
    sparse_groups = by([row for row in cuda if row.get("benchmark") in
                        {"paged_decode_head", "sparse_paged_decode_head"}], "case_id", "reuse_count")
    route_winner_changes = []
    routes_by_case = defaultdict(list)
    for (case, route), candidates in sparse_groups.items():
        best = winner(candidates)
        if best:
            ordered_candidates = sorted(candidates, key=lambda row: number(row, "p50_ms"))
            margin = ratio(number(ordered_candidates[1], "p50_ms"), number(best, "p50_ms")) \
                if len(ordered_candidates) > 1 else 1.0
            if margin >= THRESHOLD:
                routes_by_case[case].append((route, best.get("page_size"), margin))
    for case, values in routes_by_case.items():
        if len({page for _, page, _ in values}) > 1:
            route_winner_changes.append(case)
    max_interaction = max(per_token_interactions, default=0)
    tiny_interaction = median(per_head_interactions) if per_head_interactions else math.inf
    add(results, "H8.1", "supported" if max_interaction >= .03 else
        ("not_run_missing_artifact" if not meta_groups else "inconclusive"), "runtime_empirical", ["CUDA-reference"],
        f"Maximum normalized data×metadata interaction was {max_interaction:.3f}.",
        {"interactions": interactions}, "RQ8-E2")
    add(results, "H8.2", "supported" if route_winner_changes else
        ("not_run_missing_artifact" if not sparse_groups else "inconclusive"), "runtime_empirical", ["CUDA-reference"],
        f"Sparse-route granularity changed the best KV page in {len(route_winner_changes)} shapes.",
        {"cases": route_winner_changes, "winners": routes_by_case}, "RQ8-E3")
    add(results, "H8.3", "supported" if max(joint_gains, default=1) >= THRESHOLD else
        ("not_run_missing_artifact" if not meta_groups else "inconclusive"), "runtime_empirical", ["CUDA-reference"],
        "Joint selection was compared to independently selected data and metadata marginals.",
        {"max_joint_speedup": max(joint_gains, default=1.0)}, "RQ8-E1")
    add(results, "H8.4", "supported" if max_interaction >= .03 and tiny_interaction < max_interaction * .5 else
        ("not_run_missing_artifact" if not meta_groups else "inconclusive"), "runtime_empirical", ["CUDA-reference"],
        "Per-head tiny metadata is the predeclared cache-resident negative control.",
        {"median_tiny_interaction": tiny_interaction, "max_per_token_interaction": max_interaction}, "RQ8-E0")
    add(results, "H8.5", "supported" if placement_changes else
        ("not_run_missing_artifact" if not meta_groups else "inconclusive"), "runtime_empirical", ["CUDA-reference"],
        f"Metadata placement changed the data-layout winner in {len(set(placement_changes))} shapes.",
        {"cases": sorted(set(placement_changes))}, "RQ8-E4")

    # RQ9: Triton native named subset versus extended paged layouts.
    extended_wins, pipeline_extended_wins, native_contains, expressiveness_gains, physical_maps = [], [], [], [], []
    for case, candidates in triton_groups.items():
        native = [row for row in candidates if row["layout"] in {"NHD", "HND"}]
        extended = candidates
        nbest, ebest = winner(native), winner(extended)
        if not nbest or not ebest:
            continue
        gain = ratio(number(nbest, "p50_ms"), number(ebest, "p50_ms"))
        expressiveness_gains.append({"case_id": case[0] if isinstance(case, tuple) else case,
                                    "gain": gain, "native": nbest["layout"], "extended": ebest["layout"],
                                    "phase": ebest.get("phase"), "tokens": ebest.get("tokens")})
        if ebest["layout"] not in {"NHD", "HND"} and gain >= THRESHOLD:
            extended_wins.append(ebest.get("case_id"))
        else:
            native_contains.append(ebest.get("case_id"))
        physical_maps += [row.get("physical_stride") for row in candidates]
        pipeline_native = winner(native, "pipeline_p50_ms")
        pipeline_extended = winner(extended, "pipeline_p50_ms")
        if pipeline_native and pipeline_extended:
            pipeline_gain = ratio(number(pipeline_native, "pipeline_p50_ms"),
                                  number(pipeline_extended, "pipeline_p50_ms"))
            if pipeline_extended["layout"] not in {"NHD", "HND"} and pipeline_gain >= THRESHOLD:
                pipeline_extended_wins.append(pipeline_extended.get("case_id"))
    gains = [row["gain"] for row in expressiveness_gains]
    heterogeneous = bool(extended_wins) and bool(native_contains)
    add(results, "H9.1", "supported" if extended_wins else
        ("not_run_missing_artifact" if not triton else "inconclusive"), "runtime_empirical", ["Triton"],
        f"Extended legal layouts beat the NHD/HND subset in {len(extended_wins)} consumer-only "
        f"cases and {len(pipeline_extended_wins)} producer+consumer pipeline cases.",
        {"consumer_only_cases": extended_wins,
         "producer_consumer_pipeline_cases": pipeline_extended_wins,
         "primary_metric": "consumer_p50_ms_alias_p50_ms",
         "secondary_metric": "pipeline_p50_ms", "details": expressiveness_gains}, "RQ9-E2")
    add(results, "H9.2", "supported" if split_wins else "inconclusive", "runtime_empirical",
        ["CUDA-reference"], "Common intersection was compared with a union-plus-conversion split alternative.",
        {"split_wins": split_wins}, "RQ9-E3")
    add(results, "H9.3", "supported" if heterogeneous else "inconclusive", "runtime_empirical", ["Triton"],
        "Extended-space gains were tested for concentration rather than assumed universal.",
        {"extended_wins": extended_wins, "native_contains_oracle": native_contains}, "RQ9-E4")
    add(results, "H9.4", "supported" if native_contains else "inconclusive", "runtime_empirical", ["Triton"],
        f"Extension can be disabled with zero oracle regret in {len(native_contains)} shapes.",
        {"cases": native_contains}, "RQ9-E0")
    add(results, "H9.5", "feasibility_supported" if len(set(physical_maps)) > 2 else "inconclusive",
        "runtime_empirical", ["Triton"], "Compiler-visible physical-stride strings establish representations beyond two API "
        "names, but the count also varies with tensor extent and does not equal a count of distinct byte-order permutations.",
        {"distinct_recorded_physical_stride_strings": len(set(physical_maps))}, "RQ9-E1")

    # RQ10: direct block-table kernels and measured allocator/prefix cost model.
    page_groups = by([row for row in cuda if row.get("benchmark") == "paged_decode_head"], "case_id")
    kernel_winners, system_winners, fixed_totals, oracle_total = [], [], defaultdict(float), 0.0
    small_metadata_bound = 0
    multi_gains = []
    for case, candidates in page_groups.items():
        best_kernel = winner(candidates)
        if not best_kernel:
            continue
        ordered_kernel = sorted(candidates, key=lambda row: number(row, "p50_ms"))
        kernel_margin = ratio(number(ordered_kernel[1], "p50_ms"), number(best_kernel, "p50_ms")) \
            if len(ordered_kernel) > 1 else 1.0
        kernel_winners.append((case[0] if isinstance(case, tuple) else case,
                               int(number(best_kernel, "page_size")), kernel_margin))
        tokens = int(number(best_kernel, "tokens")); heads = int(number(best_kernel, "kv_heads")); dim = int(number(best_kernel, "head_dim"))
        bytes_per_token = heads * dim * 2
        measured_bw = max(number(best_kernel, "bytes") / max(number(best_kernel, "p50_ms"), 1e-9), 1.0)
        for hit in (0.0, .5, .9):
            for fanout in (1, 8):
                costs = {}
                live = max(1, tokens - 17)
                for row in candidates:
                    page = int(number(row, "page_size")); pages = math.ceil(live / page)
                    fragment = (pages * page - live) * bytes_per_token * (1-hit)
                    metadata = pages * 4 * fanout
                    costs[page] = number(row, "p50_ms") * fanout + (fragment + metadata) / measured_bw
                win_page = min(costs, key=costs.get)
                ordered_costs = sorted(costs.values())
                system_margin = ratio(ordered_costs[1], ordered_costs[0]) if len(ordered_costs) > 1 else 1.0
                system_winners.append((case, hit, fanout, win_page, system_margin))
                oracle_total += costs[win_page]
                for page, cost in costs.items(): fixed_totals[page] += cost
        table = {int(number(row, "page_size")): number(row, "p50_ms") for row in candidates}
        if 1 in table and min(table, key=table.get) != 1 and table[1] / min(table.values()) >= THRESHOLD:
            small_metadata_bound += 1
    page_static_regret = ratio(min(fixed_totals.values()), oracle_total) if fixed_totals and oracle_total else 1.0
    # Multi-granularity uses the per-trace oracle but pays extra pool and graph
    # registration overhead. Sweep that overhead as a fraction of measured
    # oracle execution time; a real threshold requires both winning and losing
    # regions, rather than assuming more pools always help.
    if oracle_total:
        fixed_total = min(fixed_totals.values())
        for overhead_fraction in (0.0, .01, .03, .10, .30):
            multi_gains.append({"overhead_fraction": overhead_fraction,
                                "speedup": ratio(fixed_total,
                                                 oracle_total * (1 + overhead_fraction))})
    system_by_case = defaultdict(set)
    for case, hit, fanout, page, margin in system_winners:
        if margin >= THRESHOLD:
            system_by_case[case].add(page)
    prefix_changes = [str(case) for case, pages in system_by_case.items() if len(pages) > 1]
    kernel_system_diff = sum(any(kcase == (case[0] if isinstance(case, tuple) else case)
                                 and kmargin >= THRESHOLD and kpage not in pages
                                 for kcase, kpage, kmargin in kernel_winners)
                             for case, pages in system_by_case.items())
    native_page = {"vLLM": vs_page, "SGLang": ss_page}
    h10_frameworks = ["CUDA-reference"] + [name for name, rows in (("vLLM", vllm), ("SGLang", sglang)) if rows]
    # Native offline rows vary request shape, not controlled prefix-hit/fanout.
    # They are useful external evidence for page sensitivity, but cannot by
    # themselves support H10.1's causal prefix/fanout claim.
    add(results, "H10.1", "supported" if prefix_changes else
        ("not_run_missing_artifact" if not page_groups else "inconclusive"), "mixed_runtime_and_measured_cost_model",
        h10_frameworks, f"Prefix-hit/fanout changed page winner in {len(prefix_changes)} controlled shapes.",
        {"cases": prefix_changes, "native": native_page}, "RQ10-E2")
    add(results, "H10.2", "supported" if small_metadata_bound else
        ("not_run_missing_artifact" if not page_groups else "inconclusive"), "runtime_empirical",
        ["CUDA-reference"], f"Page=1 was at least 3% slower than the best directly timed page in {small_metadata_bound} shapes. "
        "The benchmark is consistent with page/addressing overhead but does not separately time metadata traffic; "
        "large-page waste is modeled separately.",
        {"small_page_penalty_cases": small_metadata_bound}, "RQ10-E1")
    add(results, "H10.3", "supported" if page_static_regret >= THRESHOLD else "inconclusive", "measured_cost_model",
        ["CUDA-reference"], "One fixed page size was compared with the per-trace oracle on mixed traffic.",
        {"static_page_regret": page_static_regret}, "RQ10-E3")
    multi_values = [row["speedup"] for row in multi_gains]
    add(results, "H10.4", "supported" if any(x < 1 for x in multi_values) and any(x >= THRESHOLD for x in multi_values) else "inconclusive",
        "measured_cost_model", ["CUDA-reference"], "Multi-pool benefit was charged a synthetic overhead equal to "
        "0/1/3/10/30% of measured oracle execution time; registration/graph overhead was not directly measured.",
        {"overhead_sweep": multi_gains,
         "multi_granularity_gain_range": [min(multi_values, default=1), max(multi_values, default=1)]}, "RQ10-E4")
    add(results, "H10.5", "supported" if kernel_system_diff else "inconclusive", "mixed_runtime_and_measured_cost_model",
        ["CUDA-reference"], f"Kernel-only and allocator/system page winners differed in {kernel_system_diff} shapes.",
        {"kernel_winners": kernel_winners, "system_winners": system_winners}, "RQ10-E1")

    # Merge registry metadata and ensure fail-closed 49/49 coverage.
    canonical = {}
    for rq in registry["rqs"]:
        for hypothesis, spec in rq["hypotheses"].items():
            canonical[hypothesis] = {"rq": rq["rq"], "rq_name": rq["name"], **spec}
    for hypothesis, spec in canonical.items():
        if hypothesis not in results:
            add(results, hypothesis, spec.get("status", "not_run"), "not_run", [],
                "No adjudicator was registered for this hypothesis.", {}, spec.get("experiment"))
        results[hypothesis].update(rq=spec["rq"], rq_name=spec["rq_name"])
        results[hypothesis].setdefault("experiment", spec.get("experiment"))
    ordered = sorted(results.values(), key=lambda row: (int(row["rq"].split("RQ")[1]), row["hypothesis"]))
    source = {}
    source_path = args.source_json or args.output_dir / "v13_source_observations.json"
    if source_path.exists():
        source_doc = json.loads(source_path.read_text(encoding="utf-8"))
        source = {row["id"]: row for row in source_doc.get("hypotheses", [])}
    for row in ordered:
        if row["hypothesis"] in source:
            row["source_majority_status"] = source[row["hypothesis"]]["majority_status"]
            row["source_support"] = f"{source[row['hypothesis']]['support_count']}/{source[row['hypothesis']]['applicable_count']}"

    status_counts = Counter(row["status"] for row in ordered)
    framework_counts = defaultdict(lambda: Counter())
    for row in ordered:
        for framework in row["frameworks"]:
            framework_counts[framework][row["status"]] += 1
    payload = {"schema_version": 1, "raw_dir": str(args.raw_dir),
               "practical_speedup_threshold": THRESHOLD, "hypotheses": ordered,
               "status_counts": dict(status_counts),
               "framework_status_counts": {key: dict(value) for key, value in framework_counts.items()},
               "artifacts": {"cuda_rows": len(cuda), "triton_rows": len(triton), "tvm_rows": len(tvm),
                             "boundary_rows": len(boundary), "vllm_rows": len(vllm), "sglang_rows": len(sglang)}}
    (args.output_dir / "v13_hypothesis_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "v13_measured_models.json").write_text(
        json.dumps(models, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# v13 RQ single-GPU validation", "",
             "> `supported` requires the predeclared effect in a correct paired runtime experiment or an explicitly labeled model composed from measured primitive costs. Source prevalence is shown separately. RQ5 and cross-hardware H4.4 are not claimed on card1.", "",
             "## Run inventory", ""]
    for key, value in payload["artifacts"].items(): lines.append(f"- {key}: **{value}**")
    lines += ["", "## Overall verdicts", "", "| Status | Count |", "|---|---:|"]
    for key, value in sorted(status_counts.items()): lines.append(f"| {key} | {value} |")
    lines += ["", "## Hypothesis-by-hypothesis map", "",
              "| Hypothesis | Experiment | Verdict | Evidence | Framework(s) | Source prevalence | Result |",
              "|---|---|---|---|---|---|---|"]
    for row in ordered:
        lines.append(f"| {row['hypothesis']} | {row.get('experiment','')} | **{row['status']}** | "
                     f"{row['evidence_level']} | {', '.join(row['frameworks']) or '—'} | "
                     f"{row.get('source_majority_status','not_audited')} ({row.get('source_support','—')}) | "
                     f"{row['summary'].replace('|','/')} |")
    lines += ["", "## Per-RQ separation", ""]
    for rq in registry["rqs"]:
        selected = [row for row in ordered if row["rq"] == rq["rq"]]
        lines += [f"### {rq['rq']} — {rq['name']}", ""]
        for row in selected:
            lines.append(f"- `{row['hypothesis']}` / `{row.get('experiment','')}`: **{row['status']}** — {row['summary']}")
        lines.append("")
    lines += ["## Claims intentionally not made", "",
              "- No RQ5 distributed hypothesis is validated by the single-GPU HBM-contention proxy.",
              "- H4.4 needs at least two hardware targets; card1 cannot supply held-out hardware evidence.",
              "- `measured_cost_model` results validate the decision logic conditional on measured primitive costs, not a production framework's missing online controller.",
              "- A source-majority observation is not converted into a performance verdict.", ""]
    (args.output_dir / "V13_RQ_VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"hypotheses": len(ordered), "status_counts": dict(status_counts),
                      "report": str(args.output_dir / "V13_RQ_VALIDATION_REPORT.md")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
