#!/usr/bin/env python3
"""Cross-document observation audit for canonical RQ1--RQ10 and v10 L-RQ1.

This script never turns source prevalence into a performance claim.  It emits
three separate verdicts for each observation:

1. source-majority: the mapped mechanism appears in a majority of applicable
   pinned framework sources;
2. controlled-causal: counterfactual GPU/planner experiments show performance
   benefit on multiple shapes;
3. native-majority: the same counterfactual has been measured inside a majority
   of applicable frameworks (normally still OPEN).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_CATALOG = HERE / "layout_observation_catalog.json"
LEDGER = HERE / "ref_talks" / "layout_summary_v4_evidence_solution_roles.csv"

DENOMINATORS = {
    "serving": ["vLLM", "SGLang", "TensorRT-LLM", "FlashInfer"],
    "compiler": ["CUTLASS", "Triton", "TVM", "IREE", "MLIR"],
    "all": ["vLLM", "SGLang", "TensorRT-LLM", "FlashInfer",
            "CUTLASS", "Triton", "TVM", "IREE", "MLIR"],
}

SPEEDUP_KEYS = {
    "RQ1": "max_consumer_aware_speedup",
    "RQ2": "max_split_speedup",
    "RQ3": "max_conversion_aware_speedup",
    "RQ4": "max_contention_aware_speedup",
    "RQ5": None,
    "RQ6": "speedup_vs_fixed",
    "RQ7": "speedup_vs_forced_nhd_repair",
    "RQ8": "max_fusion_speedup",
    "RQ9": "oracle_speedup",
    "RQ10": "max_speedup_vs_forced_repair",
}


def normalize_framework(name: str) -> str:
    if name.startswith("TVM"):
        return "TVM"
    if name.startswith("CUTLASS"):
        return "CUTLASS"
    return name


def load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def native_results(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def evidence_details() -> dict[str, list[dict]]:
    details = defaultdict(list)
    with LEDGER.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for token in row["rq"].split("/"):
                token = token.strip()
                if token.startswith("RQ"):
                    details[token].append({
                        "id": row["id"], "framework": row["framework"],
                        "direct_fact": row["direct_fact"], "immutable_url": row["immutable_url"],
                        "solution_component": row["solution_component"],
                    })
    return details


def nested_numbers(value, key_contains: str = "speedup") -> list[float]:
    result = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key_contains in key.lower() and isinstance(item, (int, float)) and math.isfinite(item):
                result.append(float(item))
            result.extend(nested_numbers(item, key_contains))
    elif isinstance(value, list):
        for item in value:
            result.extend(nested_numbers(item, key_contains))
    return result


def performance_for(rq: str, general: dict, v10: dict) -> dict:
    if rq in {"v10", "L-RQ1-v10"}:
        hypotheses = v10.get("hypotheses", {})
        problem = hypotheses.get("H1.1", {}).get("status", "not_run")
        solution = hypotheses.get("edge_oracle_solution", {})
        return {
            "problem_status": problem,
            "solution_status": solution.get("status", "not_run"),
            "max_speedup": solution.get("max_speedup"),
            "evidence": "five-process runtime_empirical_controlled",
            "case_count": len(v10.get("cases", [])),
        }
    row = next((item for item in general.get("rqs", []) if item.get("rq") == rq), None)
    if row is None:
        return {"problem_status": "not_run", "solution_status": "not_run",
                "max_speedup": None, "evidence": None, "case_count": 0}
    values = nested_numbers(row.get("metrics", {}))
    return {
        "problem_status": row.get("problem_status", "not_run"),
        "solution_status": row.get("solution_status", "not_run"),
        "max_speedup": max(values) if values else None,
        "evidence": row.get("evidence_levels", []),
        "case_count": len(row.get("metrics", {}).get("cases", []))
                      if isinstance(row.get("metrics", {}).get("cases"), list) else None,
        "limitations": row.get("limitations", []),
    }


def source_for(observation: dict, audit: dict, detail_map: dict[str, list[dict]]) -> dict:
    rq = observation["source_rq"]
    raw = audit.get("rq_coverage", {}).get(rq, {})
    declared = observation.get("source_framework_override", raw.get("frameworks", []))
    frameworks = sorted({normalize_framework(name) for name in declared})
    denominator = DENOMINATORS[observation["source_denominator"]]
    applicable = sorted(set(frameworks).intersection(denominator))
    threshold = len(denominator) // 2 + 1
    details = [row for row in detail_map.get(rq, [])
               if normalize_framework(row["framework"]) in denominator]
    return {
        "applicable_frameworks": denominator,
        "supporting_frameworks": applicable,
        "support_count": len(applicable),
        "majority_threshold": threshold,
        "coverage_ratio": len(applicable) / len(denominator),
        "majority_status": "supported" if len(applicable) >= threshold else "not_supported",
        "evidence_ids": raw.get("evidence_ids", []),
        "forensic_details": details,
        "inference_guardrail": "Sources expose local rules/contracts; the cross-framework observation is an inference, not a direct quotation.",
    }


def native_for(observation: dict, rows: list[dict]) -> dict:
    denominator = DENOMINATORS[observation["source_denominator"]]
    problem_rows = [row for row in rows if row.get("observation_id") == observation["id"]
                    and row.get("claim_kind") == "problem"
                    and row.get("status") == "supported"]
    matching = [row for row in rows if row.get("observation_id") == observation["id"]
                and row.get("status") == "supported"
                and row.get("claim_kind", "solution") == "solution"
                and float(row.get("speedup", 0)) > 1.01]
    feasibility_rows = [row for row in rows if row.get("observation_id") == observation["id"]
                        and row.get("status") == "feasibility_supported"]
    frameworks = sorted({normalize_framework(row["framework"]) for row in matching
                         if normalize_framework(row["framework"]) in denominator})
    threshold = len(denominator) // 2 + 1
    return {
        "supporting_frameworks": frameworks,
        "support_count": len(frameworks),
        "majority_threshold": threshold,
        "status": "supported" if len(frameworks) >= threshold else
                  "partial" if frameworks else "not_run",
        "rows": matching,
        "problem_frameworks": sorted({normalize_framework(row["framework"]) for row in problem_rows}),
        "feasibility_frameworks": sorted({normalize_framework(row["framework"]) for row in feasibility_rows}),
    }


def grade(source: dict, performance: dict, native: dict) -> str:
    if source["majority_status"] == "supported" and performance["solution_status"] == "supported":
        if native["status"] == "supported":
            return "A_native_majority_causal"
        return "B_source_majority_plus_controlled_causal"
    if source["majority_status"] == "supported":
        return "C_source_majority_only"
    if performance["solution_status"] == "supported":
        return "D_controlled_only"
    return "OPEN"


def markdown(summary: dict) -> str:
    lines = [
        "# Cross-document critical observation validation",
        "",
        "> Source-majority, controlled causal speedup and native-framework causal reproduction are independent columns. "
        "A source vote never proves a speedup.",
        "",
        "## Summary",
        "",
        "| Observation | RQ | Source majority | Controlled performance | Native majority | Grade |",
        "|---|---|---|---|---|---|",
    ]
    for row in summary["observations"]:
        source = row["source_validation"]
        perf = row["performance_validation"]
        native = row["native_validation"]
        speedup = perf.get("max_speedup")
        perf_text = perf["solution_status"] + (f" ({speedup:.3f}x max)" if isinstance(speedup, (int, float)) else "")
        lines.append(f"| {row['id']} | {row['rq']} | {source['majority_status']} "
                     f"({source['support_count']}/{len(source['applicable_frameworks'])}) | "
                     f"{perf_text} | {native['status']} "
                     f"({native['support_count']}/{len(source['applicable_frameworks'])}) | {row['grade']} |")
    for row in summary["observations"]:
        source = row["source_validation"]
        perf = row["performance_validation"]
        lines += [
            "", f"## {row['id']} — {row['rq']}", "",
            f"**关键 observation（跨源码推论）**：{row['observation']}", "",
            f"**为什么对解题重要**：{row['why_important']}", "",
            f"**对应决策**：{row['solution_decision']}", "",
            f"**验证实验**：{row['experiment']}", "",
            f"- 源码多数性：{source['majority_status']}；"
            f"{source['support_count']}/{len(source['applicable_frameworks'])}："
            f"{', '.join(source['supporting_frameworks']) or '-'}。",
            f"- 受控性能：问题={perf['problem_status']}，方案={perf['solution_status']}，"
            f"最大观测加速={perf.get('max_speedup')}。",
            f"- 原生框架因果复现：{row['native_validation']['status']}；"
            f"{', '.join(row['native_validation']['supporting_frameworks']) or '-'}。",
        ]
        if perf.get("limitations"):
            lines.append(f"- 限制：{'；'.join(perf['limitations'])}")
    lines += ["", "## Claim boundary", ""]
    for item in summary["claim_boundary"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--general-validation", type=Path, required=True)
    parser.add_argument("--v10-validation", type=Path, required=True)
    parser.add_argument("--native-results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    catalog = load_json(args.catalog)
    audit = load_json(args.source_audit)
    general = load_json(args.general_validation)
    v10 = load_json(args.v10_validation)
    native = native_results(args.native_results)
    details = evidence_details()
    results = []
    for observation in catalog.get("observations", []):
        source = source_for(observation, audit, details)
        performance = performance_for(observation["performance_result"], general, v10)
        native_result = native_for(observation, native)
        results.append({**observation, "source_validation": source,
                        "performance_validation": performance,
                        "native_validation": native_result,
                        "grade": grade(source, performance, native_result)})
    summary = {
        "schema_version": 1,
        "observations": results,
        "grade_definition": {
            "A": "majority source mechanisms + controlled causal gain + native causal reproduction in a majority",
            "B": "majority source mechanisms + controlled causal gain; native majority still open",
            "C": "majority source mechanisms only",
            "D": "controlled gain but no source majority",
            "OPEN": "insufficient source and performance evidence",
        },
        "claim_boundary": [
            "Pinned source evidence proves that mapped mechanisms/decision rules exist, not that the inferred global observation is explicitly endorsed by a framework.",
            "Controlled CUDA and measured-cost experiments establish a causal opportunity on the tested GPU/shapes, not performance in every named framework.",
            "Grade A requires framework-native counterfactual rows in native_observation_results.jsonl; ordinary default throughput is insufficient.",
            "RQ4 distributed claims and RQ9 portability require multi-GPU and held-out GPU generations respectively.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "observation_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "OBSERVATION_VALIDATION_REPORT.md").write_text(
        markdown(summary), encoding="utf-8")
    with (args.output_dir / "observation_validation.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["id", "rq", "source_majority", "source_support", "source_denominator",
                  "controlled_problem", "controlled_solution", "max_speedup",
                  "native_majority", "native_support", "grade"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({
                "id": row["id"], "rq": row["rq"],
                "source_majority": row["source_validation"]["majority_status"],
                "source_support": row["source_validation"]["support_count"],
                "source_denominator": len(row["source_validation"]["applicable_frameworks"]),
                "controlled_problem": row["performance_validation"]["problem_status"],
                "controlled_solution": row["performance_validation"]["solution_status"],
                "max_speedup": row["performance_validation"].get("max_speedup"),
                "native_majority": row["native_validation"]["status"],
                "native_support": row["native_validation"]["support_count"],
                "grade": row["grade"],
            })
    print(json.dumps({"report": str(args.output_dir / "OBSERVATION_VALIDATION_REPORT.md"),
                      "observations": len(results),
                      "grades": {row["id"]: row["grade"] for row in results}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
