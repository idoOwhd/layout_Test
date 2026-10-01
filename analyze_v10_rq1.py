#!/usr/bin/env python3
"""Adjudicate v10 L-RQ1 using its preregistered statistical rule."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics


HERE = Path(__file__).resolve().parent
SCHEMA = HERE / "ref_talks" / "layout_summary_v10_RQ1_measurement_schema.csv"


def read_csvs(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    numeric = {"batch", "q_len", "kv_len", "Hq", "Hkv", "page_size", "reuse_count",
               "producer_us", "boundary_us", "consumer_us", "edge_us", "edge_p95_us",
               "temp_bytes", "bytes_copied", "correctness_pass"}
    for row in rows:
        for key in numeric:
            if row.get(key, "") != "":
                row[key] = float(row[key])
    return rows


def required_fields() -> list[str]:
    with SCHEMA.open(encoding="utf-8", newline="") as handle:
        return [row["field"] for row in csv.DictReader(handle) if row["status"] == "required"]


def cv(values: list[float]) -> float:
    if len(values) < 2 or statistics.mean(values) == 0:
        return 0.0
    return statistics.stdev(values) / statistics.mean(values)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    return ordered[round((len(ordered) - 1) * q)]


def bootstrap_mean_ci(values: list[float], samples: int = 10_000) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    rng = random.Random(0x10_01)
    means = []
    for _ in range(samples):
        means.append(statistics.mean(rng.choice(values) for _ in values))
    return percentile(means, .025), percentile(means, .975)


def median_by_strategy(case_rows: list[dict], metric: str) -> dict[str, float]:
    grouped = defaultdict(list)
    for row in case_rows:
        if row.get("status") == "success" and metric in row and row[metric] != "":
            grouped[row["strategy"]].append(float(row[metric]))
    return {strategy: statistics.median(values) for strategy, values in grouped.items()}


def adjudicate_case(case_id: str, rows: list[dict]) -> dict:
    success = [row for row in rows if row.get("status") == "success"]
    runs = sorted({row["run_id"] for row in success})
    per_run = []
    correctness = bool(success) and all(bool(row["correctness_pass"]) for row in success)
    for run_id in runs:
        current = [row for row in success if row["run_id"] == run_id]
        by_strategy = {row["strategy"]: row for row in current}
        if not {"NN", "NH-copy", "HH-native", "HN-copy", "NH-view"}.issubset(by_strategy):
            continue
        producer_cost = {
            "NHD": statistics.median(float(row["producer_us"]) for row in current
                                     if row["layout_producer"] == "NHD"),
            "HND": statistics.median(float(row["producer_us"]) for row in current
                                     if row["layout_producer"] == "HND"),
        }
        producer_winner = min(producer_cost, key=producer_cost.get)
        edge_winner = min(by_strategy, key=lambda name: float(by_strategy[name]["edge_us"]))
        producer_direct = "NN" if producer_winner == "NHD" else "HH-native"
        baseline = float(by_strategy[producer_direct]["edge_us"])
        oracle = float(by_strategy[edge_winner]["edge_us"])
        per_run.append({
            "run_id": run_id, "producer_winner": producer_winner,
            "edge_winner": edge_winner,
            "edge_winner_producer": by_strategy[edge_winner]["layout_producer"],
            "producer_direct": producer_direct, "baseline_us": baseline,
            "oracle_us": oracle, "regret_us": baseline - oracle,
        })
    producer_winners = Counter(row["producer_winner"] for row in per_run)
    edge_winners = Counter(row["edge_winner"] for row in per_run)
    stable_producer = len(producer_winners) == 1 and bool(per_run)
    stable_edge = len(edge_winners) == 1 and bool(per_run)
    producer_winner = next(iter(producer_winners), None) if stable_producer else None
    edge_winner = next(iter(edge_winners), None) if stable_edge else None
    regrets = [row["regret_us"] for row in per_run]
    baselines = [row["baseline_us"] for row in per_run]
    oracles = [row["oracle_us"] for row in per_run]
    ci_low, ci_high = bootstrap_mean_ci(regrets)
    if not regrets:
        ci_low, ci_high = None, None
    regret_pct = (statistics.mean(regrets) / statistics.mean(oracles) * 100.0
                  if regrets and statistics.mean(oracles) else None)
    baseline_cv = cv(baselines)
    epsilon_pct = max(1.0, 3.0 * baseline_cv * 100.0)
    edge_producer = (per_run[0]["edge_winner_producer"]
                     if stable_edge and per_run else None)
    positive = all((
        len(per_run) >= 5,
        stable_producer,
        stable_edge,
        producer_winner != edge_producer,
        ci_low is not None and ci_low > 0,
        regret_pct is not None and regret_pct > epsilon_pct,
        correctness,
    ))
    first = success[0] if success else (rows[0] if rows else {})
    medians = median_by_strategy(success, "edge_us")
    return {
        "case_id": case_id, "stage": first.get("stage"), "subgraph": first.get("subgraph"),
        "source": first.get("source"), "dtype": first.get("dtype"),
        "batch": first.get("batch"), "q_len": first.get("q_len"),
        "kv_len": first.get("kv_len"), "Hq": first.get("Hq"), "Hkv": first.get("Hkv"),
        "page_size": first.get("page_size"), "reuse_count": first.get("reuse_count"),
        "successful_process_repetitions": len(per_run),
        "producer_winner_counts": dict(producer_winners),
        "edge_winner_counts": dict(edge_winners),
        "stable_producer_winner": stable_producer, "stable_edge_winner": stable_edge,
        "producer_winner": producer_winner, "edge_winner": edge_winner,
        "edge_winner_producer": edge_producer,
        "producer_baseline_median_us": statistics.median(baselines) if baselines else None,
        "edge_oracle_median_us": statistics.median(oracles) if oracles else None,
        "edge_process_p95_us": percentile(oracles, .95) if oracles else None,
        "baseline_cv": baseline_cv, "epsilon_pct": epsilon_pct,
        "edge_regret_pct": regret_pct, "regret_ci95_us": [ci_low, ci_high],
        "correctness_pass": correctness, "positive_rank_inversion": positive,
        "strategy_median_edge_us": medians,
        "unsupported_rows": sum(row.get("status") != "success" for row in rows),
    }


def compare_solution(case_results: list[dict]) -> dict:
    measurable = [row for row in case_results if row["strategy_median_edge_us"]]
    positive = [row for row in measurable if row["positive_rank_inversion"]]
    core_positive = [row for row in positive if row["stage"] == "3-edge" and row["dtype"] == "BF16"]
    # H1.2: identical center shape, explicit reuse count varied.
    reuse = [row for row in measurable if row["case_id"] == "v10-core-r8" or
             row["case_id"].startswith("v10-reuse-r8-x")]
    reuse.sort(key=lambda row: row["reuse_count"])
    reuse_regrets = [(int(row["reuse_count"]), row["edge_regret_pct"]) for row in reuse]
    h12 = (len(reuse_regrets) >= 3 and
           reuse_regrets[-1][1] > reuse_regrets[0][1] + 1.0 and
           sum(b > a for (_, a), (_, b) in zip(reuse_regrets, reuse_regrets[1:])) >= len(reuse_regrets) - 2)
    view_wins, native_wins = [], []
    for row in measurable:
        med = row["strategy_median_edge_us"]
        if {"NH-view", "NH-copy"}.issubset(med) and med["NH-view"] < med["NH-copy"] * .99:
            view_wins.append(row["case_id"])
        if {"HH-native", "NH-copy"}.issubset(med) and med["HH-native"] < med["NH-copy"] * .99:
            native_wins.append(row["case_id"])
    negative = next((row for row in measurable if row["case_id"] == "v10-neg-bf16-small"), None)
    neg_layout_gap = None
    if negative and {"NN", "HH-native"}.issubset(negative["strategy_median_edge_us"]):
        a = negative["strategy_median_edge_us"]["NN"]
        b = negative["strategy_median_edge_us"]["HH-native"]
        neg_layout_gap = abs(a - b) / min(a, b) * 100.0
    hneg = bool(negative and not negative["positive_rank_inversion"] and
                neg_layout_gap is not None and neg_layout_gap <= negative["epsilon_pct"])
    oracle_speedups = [row["producer_baseline_median_us"] / row["edge_oracle_median_us"]
                       for row in positive if row["edge_oracle_median_us"]]
    return {
        "H1.1": {
            "status": "supported" if core_positive else "not_supported" if measurable else "not_run",
            "experiment": "Stage3 NN/NH-copy/HH-native/HN-copy; 5 process repetitions; bootstrap CI",
            "positive_cases": [row["case_id"] for row in positive],
            "core_positive_cases": [row["case_id"] for row in core_positive],
        },
        "H1.2": {
            "status": "supported" if h12 else "not_supported" if reuse else "not_run",
            "experiment": "Fixed B=8,q=1,kv=2048,Hq/Hkv=8; reuse=1/4/16/64",
            "reuse_regret_pct": reuse_regrets,
        },
        "H1.3": {
            "status": "supported" if view_wins else "not_supported" if measurable else "not_run",
            "experiment": "NH-view zero-copy stride-polymorphic consumer versus NH-copy materialization",
            "view_beats_copy_cases": view_wins,
        },
        "H1.4": {
            "status": "supported" if native_wins else "not_supported" if measurable else "not_run",
            "experiment": "HH-native direct HND emission versus NHD producer plus materialized NH-copy",
            "native_beats_copy_cases": native_wins,
        },
        "H1.NEG": {
            "status": "supported" if hneg else "not_supported" if negative else "not_run",
            "experiment": "BF16 batch=1,q=1,kv=512,Hq/Hkv=1 negative control",
            "layout_gap_pct": neg_layout_gap,
            "noise_threshold_pct": negative["epsilon_pct"] if negative else None,
        },
        "edge_oracle_solution": {
            "status": "supported" if positive else "not_supported" if measurable else "not_run",
            "experiment": "Offline exhaustive legal edge oracle versus producer-local direct baseline",
            "max_speedup": max(oracle_speedups, default=1.0),
            "positive_case_count": len(positive),
        },
        "stage4_go": bool(core_positive),
    }


def framework_evidence(source_audit: dict, output_dir: Path) -> list[dict]:
    # v14's RQ1 is the closest source-level predecessor to v10 L-RQ1. Do not
    # leak evidence from unrelated RQ2--RQ10 rows into this column.
    source = set(source_audit.get("rq_coverage", {}).get("RQ1", {}).get("frameworks", []))
    artifacts = {
        "vLLM": "vllm_native_serving.jsonl", "SGLang": "sglang_native_serving.jsonl",
        "CUTLASS/CuTe": "cutlass_softmax_boundary.csv", "Triton": "llm_representative_pytorch_triton.jsonl",
        "TVM Relax": "tvm_layout.csv", "Hexcute": "hexcute.csv",
        "FlashInfer": "flashinfer_v10_rq1.csv", "TensorRT-LLM": "tensorrt_llm_v10_rq1.csv",
    }
    result = []
    for framework, artifact in artifacts.items():
        path = output_dir / artifact
        result.append({
            "framework": framework,
            "source_static_rq1": framework in source,
            "runtime_artifact": artifact if path.exists() and path.stat().st_size else None,
            "v10_edge_decomposition": artifact if framework in {"FlashInfer", "TensorRT-LLM"} and path.exists() else None,
        })
    return result


def write_adjudicated_csv(path: Path, results: list[dict]) -> None:
    fields = ["case_id", "stage", "subgraph", "dtype", "batch", "q_len", "kv_len", "Hq", "Hkv",
              "page_size", "reuse_count", "successful_process_repetitions", "producer_winner",
              "edge_winner", "edge_winner_producer", "producer_baseline_median_us",
              "edge_oracle_median_us", "edge_process_p95_us", "baseline_cv", "epsilon_pct",
              "edge_regret_pct", "regret_ci95_us", "correctness_pass", "positive_rank_inversion"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({key: json.dumps(row[key]) if isinstance(row.get(key), (dict, list))
                             else row.get(key) for key in fields})


def markdown(summary: dict) -> str:
    lines = [
        "# layout_summary_v10 L-RQ1 empirical adjudication",
        "",
        "> v10 defines L-RQ1 only. H1.1/H1.2/H1.3/H1.4/H1.NEG are hypotheses under L-RQ1, not separate RQs.",
        "",
        "## Experiment → RQ mapping",
        "",
        "| Hypothesis | Exact experiment | Verdict |",
        "|---|---|---|",
    ]
    for key in ("H1.1", "H1.2", "H1.3", "H1.4", "H1.NEG"):
        item = summary["hypotheses"][key]
        lines.append(f"| {key} | {item['experiment']} | **{item['status']}** |")
    solution = summary["hypotheses"]["edge_oracle_solution"]
    lines += [
        "", "## Problem and solution observation", "",
        f"- L-RQ1 phenomenon: **{summary['hypotheses']['H1.1']['status']}**.",
        f"- Edge-oracle solution observation: **{solution['status']}**, max observed speedup "
        f"{solution['max_speedup']:.3f}x over producer-local direct choice.",
        f"- Stage 4 crossover gate: **{'GO' if summary['stage4_gate']['go'] else 'STOP'}**.",
        f"- L-RQ2 permission: **{summary['stage4_gate']['l_rq2_permission']}**.",
        "", "## Per-case adjudication", "",
        "| Case | Shape | Producer winner | Edge winner | Regret | 95% CI (us) | epsilon | Inversion |",
        "|---|---|---|---|---:|---|---:|---|",
    ]
    for row in summary["cases"]:
        shape = f"B{int(row['batch']) if row['batch'] is not None else '-'} Q{int(row['q_len']) if row['q_len'] is not None else '-'} K{int(row['kv_len']) if row['kv_len'] is not None else '-'} H{int(row['Hq']) if row['Hq'] is not None else '-'}/{int(row['Hkv']) if row['Hkv'] is not None else '-'}"
        regret = "-" if row["edge_regret_pct"] is None else f"{row['edge_regret_pct']:.2f}%"
        lines.append(f"| {row['case_id']} | {shape} | {row['producer_winner'] or '-'} | {row['edge_winner'] or '-'} | {regret} | {row['regret_ci95_us']} | {row['epsilon_pct']:.2f}% | {'yes' if row['positive_rank_inversion'] else 'no'} |")
    lines += ["", "## Framework evidence boundary", "",
              "| Framework | Source-level L-RQ1 evidence | Runtime artifact | Exact v10 edge decomposition |",
              "|---|---|---|---|"]
    for row in summary["framework_evidence"]:
        lines.append(f"| {row['framework']} | {'yes' if row['source_static_rq1'] else 'no'} | {row['runtime_artifact'] or '-'} | {row['v10_edge_decomposition'] or '-'} |")
    lines += ["", "## Requirements audit", ""]
    for name, item in summary["requirements"].items():
        lines.append(f"- **{item['status']}** `{name}`: {item['detail']}")
    lines += ["", "## Requirements still open", ""]
    for item in summary["remaining_requirements"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--pattern", default="v10_rq1_*_rep*.csv")
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()
    paths = list(args.input)
    if args.input_dir:
        paths.extend(sorted(args.input_dir.glob(args.pattern)))
    rows = read_csvs(paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing_schema = sorted(set(required_fields()) - set(rows[0] if rows else ()))
    by_case = defaultdict(list)
    for row in rows:
        by_case[row.get("case_id", "missing")].append(row)
    results = [adjudicate_case(case_id, current) for case_id, current in sorted(by_case.items())]
    hypotheses = compare_solution(results)
    source_audit = {}
    if args.source_audit and args.source_audit.exists():
        source_audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    native_edge_frameworks = []
    matrix = framework_evidence(source_audit, args.output_dir)
    for row in matrix:
        if row["v10_edge_decomposition"]:
            native_edge_frameworks.append(row["framework"])
    gate = {
        "go": hypotheses["stage4_go"],
        "reason": "at least one BF16 Stage3 core case passed the preregistered inversion rule" if hypotheses["stage4_go"] else "no BF16 Stage3 core case passed all inversion criteria",
        "l_rq2_permission": "GO" if hypotheses["stage4_go"] and native_edge_frameworks else "BLOCKED",
        "independent_native_edge_frameworks": native_edge_frameworks,
    }
    measured = [row for row in results if row["successful_process_repetitions"] >= 5]
    real_shapes = [row for row in measured
                   if row.get("source") and not str(row["source"]).startswith("v10:")]
    required_centers = {"v10-core-r1", "v10-core-r8", "v10-core-r32"}
    measured_ids = {row["case_id"] for row in measured}
    nvfp4_success = any(row["dtype"] == "NVFP4" and row["successful_process_repetitions"] >= 5
                        for row in results)
    requirements = {
        "v10_measurement_schema": {
            "status": "PASS" if rows and not missing_schema else "MISSING",
            "detail": f"missing required fields: {missing_schema or 'none'}",
        },
        "five_independent_processes": {
            "status": "PASS" if measured and all(row["successful_process_repetitions"] >= 5 for row in results if row["dtype"] == "BF16") else "MISSING",
            "detail": f"{len(measured)}/{len(results)} cases have at least five successful process repetitions.",
        },
        "stage0_to_stage3_core": {
            "status": "PASS" if required_centers.issubset(measured_ids) else "MISSING",
            "detail": f"measured required Stage-3 centers: {sorted(required_centers.intersection(measured_ids))}",
        },
        "common_llm_multiple_shapes": {
            "status": "PASS" if len(real_shapes) >= 3 else "PARTIAL" if real_shapes else "MISSING",
            "detail": f"{len(real_shapes)} real-model attention shapes have complete repetitions; Stage-5 all-subgraph expansion is gate-controlled.",
        },
        "native_nvfp4": {
            "status": "PASS" if nvfp4_success else "MISSING",
            "detail": "Requires a real native NVFP4 implementation; unsupported rows are deliberate.",
        },
        "multi_framework_native_edge_decomposition": {
            "status": "PASS" if len(native_edge_frameworks) >= 2 else "PARTIAL" if native_edge_frameworks else "MISSING",
            "detail": f"exact native edge artifacts: {native_edge_frameworks or 'none'}; ordinary server throughput is not accepted.",
        },
        "stage4_protocol_order": {
            "status": "PASS",
            "detail": "runner executes Stage 4/5 only when the Stage-3 statistical gate is GO.",
        },
    }
    summary = {
        "schema_version": 1, "rq": "L-RQ1", "input_files": [str(path) for path in paths],
        "input_row_count": len(rows), "missing_required_schema_fields": missing_schema,
        "cases": results, "hypotheses": hypotheses, "stage4_gate": gate,
        "framework_evidence": matrix, "requirements": requirements,
        "remaining_requirements": [
            "Native NVFP4 requires a supported H100/B200-class path; unsupported rows are not replaced by BF16.",
            "FlashInfer or TensorRT-LLM must repeat the exact producer/boundary/consumer decomposition before backend external validity is claimed.",
            "vLLM/SGLang full-engine runs are Stage-5 evidence only and cannot replace Stage-3 causal decomposition.",
            "MLA, sparse-attention metadata, and MoE scale-layout boundaries are conditional Stage-5 experiments after a positive Stage-3 gate.",
            "An independent hardware/backend check is required before L-RQ2 is unblocked.",
        ],
    }
    output_json = args.output_dir / ("v10_rq1_gate.json" if args.gate_only else "v10_rq1_validation.json")
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_adjudicated_csv(args.output_dir / "v10_rq1_adjudicated.csv", results)
    if not args.gate_only:
        (args.output_dir / "V10_RQ1_VALIDATION_REPORT.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({"output": str(output_json), "rows": len(rows), "cases": len(results),
                      "H1.1": hypotheses["H1.1"]["status"], "stage4_go": gate["go"]}))
    return 1 if missing_schema else 0


if __name__ == "__main__":
    raise SystemExit(main())
