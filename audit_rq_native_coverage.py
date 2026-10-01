#!/usr/bin/env python3
"""Fail-closed audit of problem, solution and native-framework RQ evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_status(path: Path) -> dict[str, dict]:
    return {row["step"]: row for row in load_jsonl(path)}


def shape_counts(canonical: Path) -> dict[str, int]:
    result = {"vLLM": 0, "SGLang": 0, "TVM": 0}
    for framework, filename in (("vLLM", "vllm_native_serving.jsonl"),
                                ("SGLang", "sglang_native_serving.jsonl")):
        rows = load_jsonl(canonical / filename)
        result[framework] = len({row.get("case_id") for row in rows
                                 if row.get("status") == "success"
                                 and row.get("comparison_scope") == "layout_only"})
    path = canonical / "tvm_rq_observations.csv"
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            result["TVM"] = len({row["case_id"] for row in csv.DictReader(handle)
                                  if row.get("experiment") == "consumer_scan"
                                  and row.get("correct") in {"1", "True", "true"}})
    return result


def evidence_state(rows: list[dict], observation_ids: set[str], framework: str) -> dict:
    selected = [row for row in rows if row.get("observation_id") in observation_ids
                and row.get("framework") == framework]
    solution = [row for row in selected if row.get("claim_kind", "solution") == "solution"
                and row.get("status") == "supported" and float(row.get("speedup", 0)) > 1.01]
    problem = [row for row in selected if row.get("claim_kind") == "problem"
               and row.get("status") == "supported"]
    feasibility = [row for row in selected if row.get("status") == "feasibility_supported"]
    inconclusive = [row for row in selected if row.get("status") == "inconclusive"]
    if solution:
        state = "solution_performance_supported"
    elif problem:
        state = "problem_only_supported"
    elif feasibility:
        state = "solution_feasibility_only"
    elif inconclusive:
        state = "measured_inconclusive"
    else:
        state = "not_measured"
    return {"status": state, "max_speedup": max((float(row["speedup"]) for row in solution), default=None),
            "rows": selected}


def is_solution_applicable(text: str) -> bool:
    return text.startswith("native:") or text.startswith("native_single_gpu:")


def markdown(summary: dict) -> str:
    lines = [
        "# RQ × native-framework completion audit",
        "",
        "> PASS is fail-closed: source evidence, controlled CUDA evidence, native problem reproduction and native solution speedup are separate. Import success or a default throughput row is never counted as an RQ validation.",
        "",
        "## Run coverage",
        "",
        "| Framework | Adapter status | Distinct native shapes |",
        "|---|---|---:|",
    ]
    for framework, row in summary["frameworks"].items():
        lines.append(f"| {framework} | {row['adapter_status']} | {row['distinct_shapes']} |")
    lines += ["", "## RQ mapping and verdicts", "",
              "| RQ | Source observation majority | Controlled problem | Controlled solution | vLLM | SGLang | TVM | Installed-applicable solution |",
              "|---|---|---|---|---|---|---|---|"]
    for row in summary["rqs"]:
        lines.append(f"| {row['rq']} | {row['source_status']} | {row['controlled_problem']} | {row['controlled_solution']} | "
                     f"{row['native']['vLLM']['status']} | {row['native']['SGLang']['status']} | "
                     f"{row['native']['TVM']['status']} | {row['installed_solution_status']} |")
    lines += ["", "## Exact experiment-to-RQ map", ""]
    for row in summary["rqs"]:
        lines += [f"### {row['rq']}", "", f"- Important observation(s): `{', '.join(row['observations'])}`.",
                  f"- Source prevalence: {row['source_status']} ({'; '.join(row['source_details']) or 'not available'}).",
                  f"- Controlled experiment: {row['controlled_experiment']}",
                  f"- Controlled result: problem={row['controlled_problem']}; solution={row['controlled_solution']}."]
        for framework in ("vLLM", "SGLang", "TVM"):
            applicability = row["native_applicability"][framework]
            evidence = row["native"][framework]
            suffix = f"; max paired speedup={evidence['max_speedup']:.3f}x" if evidence["max_speedup"] else ""
            lines.append(f"- {framework}: {applicability} Result: **{evidence['status']}**{suffix}.")
        lines.append("")
    lines += ["## What remains unproved", ""]
    for item in summary["remaining_unproved"]:
        lines.append(f"- {item}")
    lines += ["", "## Interpretation", "",
              "- `solution_performance_supported` means a correct paired native counterfactual exceeded 1% in this run.",
              "- `problem_only_supported` does not validate the proposed controller/repair.",
              "- `solution_feasibility_only` proves legality/correctness, not speedup.",
              "- `measured_inconclusive` is a valid negative result; it must not be rewritten as support.",
              "- The broad cross-framework majority claim remains in `OBSERVATION_VALIDATION_REPORT.md`; this report only audits the three installed frameworks.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=HERE / "rq_experiment_registry.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.result_root
    canonical = root / "canonical"
    out = args.output_dir or root
    registry = load_json(args.registry)
    validation = load_json(canonical / "rq_validation.json")
    observation_validation = load_json(root / "observation_validation.json")
    source_by_id = {row["id"]: row.get("source_validation", {})
                    for row in observation_validation.get("observations", [])}
    controlled = {row["rq"]: row for row in validation.get("rqs", [])}
    native_rows = load_jsonl(root / "native_observation_results.jsonl")
    statuses = latest_status(canonical / "status.jsonl")
    counts = shape_counts(canonical)
    step_names = {"vLLM": "vllm_native_serving", "SGLang": "sglang_native_serving",
                  "TVM": "tvm_rq_observations"}
    frameworks = {}
    for framework, step in step_names.items():
        row = statuses.get(step, {})
        frameworks[framework] = {"adapter_status": row.get("status", "missing"),
                                 "detail": row.get("detail", ""),
                                 "distinct_shapes": counts[framework]}
    rqs = []
    for spec in registry["rqs"]:
        rq = spec["rq"]
        controlled_row = controlled.get(rq, {})
        observations = set(spec["observations"])
        source_states = [source_by_id.get(observation, {}).get("majority_status", "not_run")
                         for observation in spec["observations"]]
        source_status = ("supported" if source_states and all(x == "supported" for x in source_states)
                         else "partial" if any(x == "supported" for x in source_states) else "not_supported")
        source_details = []
        for observation in spec["observations"]:
            source = source_by_id.get(observation, {})
            source_details.append(f"{observation}={source.get('support_count', 0)}/{len(source.get('applicable_frameworks', []))}")
        native = {framework: evidence_state(native_rows, observations, framework)
                  for framework in ("vLLM", "SGLang", "TVM")}
        applicable = [framework for framework, text in spec["native_applicability"].items()
                      if is_solution_applicable(text)]
        supported = [framework for framework in applicable
                     if native[framework]["status"] == "solution_performance_supported"]
        threshold = len(applicable) // 2 + 1 if applicable else None
        installed_solution = ("supported" if threshold and len(supported) >= threshold else
                              "not_supported" if applicable else "no_native_solution_adapter")
        rqs.append({**spec,
                    "controlled_problem": controlled_row.get("problem_status", "not_run"),
                    "controlled_solution": controlled_row.get("solution_status", "not_run"),
                    "source_status": source_status, "source_details": source_details,
                    "native": native, "solution_applicable_frameworks": applicable,
                    "solution_supporting_frameworks": supported,
                    "installed_solution_threshold": threshold,
                    "installed_solution_status": installed_solution})
    remaining = []
    for row in rqs:
        if row["controlled_problem"] != "supported" or row["controlled_solution"] != "supported":
            remaining.append(f"{row['rq']}: controlled problem/solution pair is not both supported in this run.")
        if row["installed_solution_status"] != "supported":
            remaining.append(f"{row['rq']}: installed-framework solution status is {row['installed_solution_status']}.")
    remaining += [
        "RQ4 distributed TP/EP/All2All claims require at least two GPUs; card 1 alone cannot validate them.",
        "RQ9 portability requires a held-out GPU architecture and framework version; a single A10 run only establishes within-device adaptivity.",
        "vLLM/SGLang whole-engine timings cannot identify producer, conversion edge and consumer costs separately.",
    ]
    summary = {"schema_version": 1, "result_root": str(root), "frameworks": frameworks,
               "rqs": rqs, "remaining_unproved": list(dict.fromkeys(remaining))}
    out.mkdir(parents=True, exist_ok=True)
    (out / "rq_native_coverage.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "RQ_NATIVE_COVERAGE.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({"report": str(out / "RQ_NATIVE_COVERAGE.md"),
                      "frameworks": frameworks,
                      "rqs": {row["rq"]: row["installed_solution_status"] for row in rqs}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
