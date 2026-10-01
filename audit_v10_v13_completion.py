#!/usr/bin/env python3
"""Fail-closed completion audit for a combined strict-v10 and v13-640 run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def is_boundary_alternate(row: dict) -> bool:
    return row.get("strategy") in {
        "alternate_strided_view", "alternate_axis12_strided"}


def boundary_layout_changed(row: dict) -> bool:
    """Handle both current rows and legacy rows without trusting the label."""
    if "layout_changed" in row:
        return row.get("layout_changed") is True
    contiguous = row.get("input_contiguous")
    return isinstance(contiguous, list) and any(value is False for value in contiguous)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v10", type=Path, required=True)
    parser.add_argument("--v13", type=Path, required=True,
                        help="v13_640 run root containing full_combined")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    combined = args.v13 / "full_combined"
    raw = combined / "raw"
    matrix_path = combined / "v13_640_matrix_summary.json"
    hypothesis_path = combined / "v13_hypothesis_results.json"
    v13_run_path = args.v13 / "V13_640_RUN_COMPLETENESS.json"
    matrix = json.loads(matrix_path.read_text()) if matrix_path.exists() else {}
    hypotheses = json.loads(hypothesis_path.read_text()) if hypothesis_path.exists() else {}
    v13_run = json.loads(v13_run_path.read_text()) if v13_run_path.exists() else {}
    v10_status = jsonl(args.v10 / "status.jsonl")
    whole = jsonl(raw / "llm_640_pytorch_triton.jsonl")
    boundary = jsonl(raw / "llm_640_boundary_layout_sweep.jsonl")
    vllm = jsonl(raw / "vllm_native_serving.jsonl")
    sglang = jsonl(raw / "sglang_native_serving.jsonl")

    v10_reps = sum(row.get("step", "").startswith("v10_core_rep")
                   and row.get("status") == "success" for row in v10_status)
    whole_status = Counter(str(row.get("status")) for row in whole)
    boundary_status = Counter(str(row.get("status")) for row in boundary)
    successful_alternates = [row for row in boundary
                             if row.get("status") == "success"
                             and is_boundary_alternate(row)]
    boundary_noops = sum(not boundary_layout_changed(row)
                         for row in successful_alternates)
    boundary_case_strategies: dict[str, set[str]] = {}
    for row in boundary:
        case_id = row.get("case_id")
        if not case_id:
            continue
        boundary_case_strategies.setdefault(str(case_id), set())
        if row.get("status") == "success" and row.get("strategy"):
            boundary_case_strategies[str(case_id)].add(str(row["strategy"]))
    required_boundary_strategies = {
        "native_contiguous", "alternate_strided_view",
        "repair_to_contiguous_each_call"}
    incomplete_success_groups = sum(
        bool(strategies) and strategies != required_boundary_strategies
        for strategies in boundary_case_strategies.values())
    native_ok = {
        "vllm": sum(row.get("status") == "success" for row in vllm),
        "sglang": sum(row.get("status") == "success" for row in sglang),
    }
    matrix_schema_current = matrix.get("schema_version", 0) >= 2
    # Execution completeness means the requested programs ran and emitted the
    # full fail-closed evidence surface.  Numerical mismatches and OOM are
    # scientific outcomes, not launcher failures; conflating them previously
    # made the global command return failure after a perfectly valid run.
    execution_complete = (
        v10_reps >= 5
        and matrix_schema_current
        and matrix.get("matrix_cell_count") == 640 * 6 * 10
        and v13_run.get("infrastructure_complete") is True
        and len(whole) >= 640 * 2
        and whole_status.get("timeout", 0) == 0
        and len(boundary_case_strategies) == 640
        and native_ok == {"vllm": 16, "sglang": 16}
    )
    non_oom_software_failures = sum(
        count for status, count in whole_status.items()
        if status not in {"success", "oom_preflight", "oom_runtime"}) + sum(
        count for status, count in boundary_status.items()
        if status not in {"success", "oom_preflight", "oom_runtime",
                          "not_applicable_no_layout_change"})
    clean_non_oom_results = (
        non_oom_software_failures == 0
        and incomplete_success_groups == 0
        and boundary_noops == 0
    )
    blockers = {
        "single_gpu_rq5": True,
        "cross_hardware_h4_4": True,
        "hexcute_a10_architecture": True,
        "v14_hopper_blackwell_distributed": True,
        "not_applicable_framework_ownership_cells":
            matrix.get("status_counts", {}).get("not_applicable_framework_owner", 0),
        "source_only_cells": matrix.get("status_counts", {}).get("source_observation_only", 0),
        "external_not_case_matched_cells":
            matrix.get("status_counts", {}).get("native_external_not_case_matched", 0),
        "projected_runtime_cells":
            matrix.get("status_counts", {}).get("paired_native_projected_runtime", 0),
    }
    scientific_all_claim_valid = execution_complete and clean_non_oom_results and not any(
        value for value in blockers.values())
    payload = {
        "schema_version": 2,
        "v10_scope": "L-RQ1_only",
        "v10_core_successful_process_repetitions": v10_reps,
        "v13_matrix_cells": matrix.get("matrix_cell_count", 0),
        "v13_matrix_schema_version": matrix.get("schema_version", 0),
        "v13_matrix_schema_current": matrix_schema_current,
        "v13_runner_infrastructure_complete":
            v13_run.get("infrastructure_complete", False),
        "v13_exact_paired_native_cells": matrix.get("paired_native_runtime_cell_count", 0),
        "v13_projected_paired_native_cells":
            matrix.get("paired_native_projected_runtime_cell_count", 0),
        "hypothesis_status_counts": hypotheses.get("status_counts", {}),
        "whole_subgraph_status_counts": dict(whole_status),
        "boundary_status_counts": dict(boundary_status),
        "successful_boundary_alternates": len(successful_alternates),
        "boundary_noop_alternates": boundary_noops,
        "boundary_case_count": len(boundary_case_strategies),
        "boundary_incomplete_success_groups": incomplete_success_groups,
        "native_serving_success_rows": native_ok,
        "card1_execution_infrastructure_complete": execution_complete,
        "clean_non_oom_results": clean_non_oom_results,
        "non_oom_software_failure_rows": non_oom_software_failures,
        "scientific_all_frameworks_all_rqs_all_640_claim_valid": scientific_all_claim_valid,
        "irreducible_or_unimplemented_blockers": blockers,
        "interpretation": (
            "A complete card1 execution is not equivalent to empirical validation of "
            "not-applicable, distributed, cross-hardware, architecture-blocked, projected, "
            "source-only, or non-case-matched cells."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "CROSS_VERSION_COMPLETENESS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# v10 + v13 card1 completion audit", "",
             f"- Card1 execution infrastructure complete: **{execution_complete}**",
             f"- Clean non-OOM scientific results: **{clean_non_oom_results}**",
             f"- Non-OOM software-failure rows: **{non_oom_software_failures}**",
             f"- Scientific all-framework/all-RQ/all-640 claim valid: **{scientific_all_claim_valid}**",
             f"- v10 scope: **L-RQ1 only**, successful core process repetitions: **{v10_reps}**",
             f"- v13 matrix cells: **{payload['v13_matrix_cells']}**",
             f"- v13 matrix schema current: **{matrix_schema_current}** (version {payload['v13_matrix_schema_version']})",
             f"- v13 runner infrastructure complete: **{payload['v13_runner_infrastructure_complete']}**",
             f"- Exact paired native cells: **{payload['v13_exact_paired_native_cells']}**",
             f"- Projected paired native cells: **{payload['v13_projected_paired_native_cells']}**",
             f"- Boundary represented cases: **{payload['boundary_case_count']} / 640**",
             f"- Boundary incomplete successful triplets: **{incomplete_success_groups}**",
             f"- Boundary no-op alternatives: **{boundary_noops}**", "",
             "## Fail-closed blockers", ""]
    for key, value in blockers.items():
        lines.append(f"- `{key}`: **{value}**")
    lines += ["", "## Interpretation", "", payload["interpretation"], ""]
    (args.output_dir / "CROSS_VERSION_COMPLETENESS.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    # Return success when execution itself completed.  Scientific blockers and
    # measured counterexamples remain explicit in the JSON/Markdown verdict.
    return 0 if execution_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
