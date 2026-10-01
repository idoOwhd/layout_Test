#!/usr/bin/env python3
"""Fail-closed 640-case × framework × v13-RQ coverage audit.

The Cartesian product is always materialized.  A cell is counted as native
validation only when an artifact contains the exact manifest case id and at
least two successful, correct alternatives for the RQ decision.  Whole-model
serving rows, source evidence, generic microbenchmarks and CUDA reference rows
remain useful evidence, but are never copied into a framework-native cell.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE.parent / "real_world_shapes" / "real_world_shape_manifest.json"
DEFAULT_REGISTRY = HERE / "v13_experiment_registry.json"


def json_rows(path: Path) -> list[dict]:
    if not path.exists() or not path.stat().st_size:
        return []
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return value["cases"] if isinstance(value, dict) and "cases" in value else value
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def cases(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value["cases"] if isinstance(value, dict) else value


def success(row: dict) -> bool:
    status = str(row.get("status", "success")).lower()
    correct = str(row.get("correct", "true")).lower()
    return status in {"success", "true", "1", ""} and correct in {"true", "1", ""}


def candidate(row: dict) -> str | None:
    for key in ("layout", "strategy", "variant", "backend"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    if row.get("tile_rows") not in (None, "") and row.get("tile_cols") not in (None, ""):
        return f"tile_{row['tile_rows']}x{row['tile_cols']}"
    return None


def native_candidates(framework: str, rows: list[dict]) -> list[str]:
    """Return alternatives for the decision variable, not incidental tile IDs."""
    if framework == "CUTLASS/CuTe":
        choices = set()
        for row in rows:
            if number(row, "flat_ms") > 0:
                choices.add("flat_row_major")
            if number(row, "tiled_direct_ms") > 0:
                choices.add("tiled_direct")
        return sorted(choices)
    return sorted({value for row in rows if (value := candidate(row))})


def evidence_granularity(framework: str, rows: list[dict]) -> str:
    if framework == "TVM" and any(row.get("manifest_case_id") for row in rows):
        return "projected_boundary_shape"
    if framework == "CUTLASS/CuTe" and rows:
        return "projected_attention_score_boundary"
    if framework in {"vLLM", "SGLang"}:
        return "whole_model_external_regime"
    return "case_matched_native_boundary"


def rq_native_rows(framework: str, rq: str, rows: list[dict]) -> list[dict]:
    """Select only alternatives that manipulate the named RQ decision."""
    if framework == "TVM":
        experiment = "conversion_reuse" if rq == "L-RQ3" else "consumer_scan"
        return [row for row in rows if row.get("experiment") == experiment]
    if framework == "Triton":
        if rq == "L-RQ1":
            return [row for row in rows if number(row, "pipeline_p50_ms") > 0]
        return rows
    if framework in {"vLLM", "SGLang"}:
        scope = "page_size_only" if rq == "L-RQ10" else "layout_only"
        return [row for row in rows if row.get("comparison_scope") == scope]
    if framework == "CUTLASS/CuTe":
        return [row for row in rows if row.get("flat_ms") and row.get("tiled_direct_ms")]
    return rows


def number(row: dict, key: str) -> float:
    try:
        return float(row.get(key, 0))
    except (TypeError, ValueError):
        return 0.0


def in_scope(case: dict, rq: dict) -> bool:
    structures = rq["case_scope"]["structures"]
    return structures == "all" or case["structure"] in structures


def framework_class(spec: dict, rq: str) -> str:
    if rq in spec.get("blocked_rqs", []):
        return "blocked_single_gpu"
    if rq in spec.get("architecture_blocked_rqs", []):
        return "blocked_architecture_unsupported"
    if rq in spec.get("not_applicable_rqs", []):
        return "not_applicable_framework_owner"
    if rq in spec.get("missing_adapter_rqs", []):
        return "missing_native_adapter"
    if rq in spec.get("runtime_rqs", []):
        return "runtime"
    if rq in spec.get("source_only_rqs", []):
        return "source_observation_only"
    return "unmapped_registry_error"


def load_artifacts(raw: Path, registry: dict) -> tuple[dict[str, list[dict]], dict[str, Path]]:
    rows, paths = {}, {}
    preferred = {
        "Triton": ["triton_640_kv_layout.csv", "triton_kv_layout.csv"],
        "vLLM": ["vllm_native_serving.jsonl"],
        "SGLang": ["sglang_native_serving.jsonl"],
        "TVM": ["tvm_640_rq_observations.csv", "tvm_rq_observations.csv"],
        "CUTLASS/CuTe": ["cutlass_640_softmax_boundary.csv", "cutlass_softmax_boundary.csv"],
        "Hexcute": ["hexcute_640.jsonl", "hexcute.jsonl"],
    }
    for framework in registry["framework_adapters"]:
        selected = next((raw / name for name in preferred[framework]
                         if (raw / name).exists()), raw / preferred[framework][0])
        paths[framework] = selected
        rows[framework] = json_rows(selected)
    return rows, paths


def controlled_indexes(raw: Path) -> dict[str, tuple[Path, dict[str, list[dict]]]]:
    paths = {
        "boundary": raw / "llm_640_boundary_layout_sweep.jsonl",
        "triton": raw / "triton_640_kv_layout.csv",
        "cuda": raw / "rq_cuda_all_attention.csv",
    }
    if not paths["boundary"].exists():
        paths["boundary"] = raw / "llm_boundary_layout_sweep.jsonl"
    result = {}
    for name, path in paths.items():
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in json_rows(path):
            if row.get("case_id") and success(row):
                grouped[str(row["case_id"])].append(row)
        result[name] = (path, grouped)
    return result


def controlled_case_status(indexes: dict, case: dict, rq: str) -> tuple[str, list[str]]:
    cid = case["case_id"]
    files: list[str] = []
    boundary, boundary_index = indexes["boundary"]
    boundary_rows = boundary_index.get(cid, [])
    changed_boundary = [row for row in boundary_rows
                        if row.get("strategy") in {"alternate_strided_view",
                                                   "alternate_axis12_strided"}
                        and (row.get("layout_changed") is True or (
                            "layout_changed" not in row
                            and any(value is False for value in row.get("input_contiguous", []))))]
    if rq in {"L-RQ1", "L-RQ3", "L-RQ7"} and changed_boundary and \
            any(row.get("strategy") == "repair_to_contiguous_each_call" for row in boundary_rows):
        files.append(str(boundary))
        return "paired_controlled_runtime", files

    triton, triton_index = indexes["triton"]
    triton_rows = triton_index.get(cid, [])
    if rq in {"L-RQ4", "L-RQ9"} and len({row.get("layout") for row in triton_rows}) >= 2:
        files.append(str(triton))
        return "paired_controlled_runtime", files

    cuda, cuda_index = indexes["cuda"]
    cuda_rows = cuda_index.get(cid, [])
    if rq in {"L-RQ1", "L-RQ2", "L-RQ3", "L-RQ8", "L-RQ10"} and cuda_rows:
        files.append(str(cuda))
        return "controlled_runtime_present", files
    return "not_measured_for_exact_case", files


def markdown(summary: dict) -> str:
    lines = [
        "# v13 validation over 640 LLM subgraphs × all frameworks × all RQs",
        "",
        "> The matrix contains every Cartesian-product cell. `paired_native_runtime` is the only native performance-validation state. Other states are preserved as gaps, ownership boundaries, source observations, or single-GPU blockers.",
        "",
        "## Cardinality and verdict",
        "",
        f"- Manifest cases: **{summary['case_count']}**.",
        f"- Frameworks: **{summary['framework_count']}**.",
        f"- RQs: **{summary['rq_count']}**.",
        f"- Matrix cells: **{summary['matrix_cell_count']}** (expected {summary['expected_matrix_cell_count']}).",
        f"- Exact paired native cells: **{summary['status_counts'].get('paired_native_runtime', 0)}**.",
        f"- All applicable cells natively validated: **{summary['all_applicable_cells_natively_validated']}**.",
        "",
        "## Per-framework × RQ status counts",
        "",
        "| Framework | RQ | In-scope cases | Exact paired | Projected paired | Unpaired native | External/not case-matched | Missing adapter/runtime | Source-only | Blocked |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["framework_rq_summary"]:
        c = row["counts"]
        lines.append(
            f"| {row['framework']} | {row['rq']} | {row['in_scope_cases']} | "
            f"{c.get('paired_native_runtime', 0)} | "
            f"{c.get('paired_native_projected_runtime', 0)} | "
            f"{c.get('native_runtime_unpaired', 0)} | "
            f"{c.get('native_external_not_case_matched', 0)} | "
            f"{c.get('missing_native_adapter', 0) + c.get('native_runtime_missing', 0)} | "
            f"{c.get('source_observation_only', 0)} | "
            f"{c.get('blocked_single_gpu', 0) + c.get('blocked_architecture_unsupported', 0)} |"
        )
    lines += [
        "", "## Guardrails", "",
        "- vLLM/SGLang whole-model rows are external-validity evidence and are not relabeled as exact 640-case execution.",
        "- CUDA/PyTorch controlled rows are recorded in the case×RQ table and are not copied into any of the six native-framework cells.",
        "- `not_applicable_framework_owner` is not a failure: the framework does not own the RQ decision variable.",
        "- RQ5 remains `blocked_single_gpu` on card 1; H4.4 remains a cross-hardware blocker in the hypothesis report.",
        "- The full row-level evidence is in `v13_640_framework_rq_matrix.jsonl`; exact case-level controlled coverage is in `v13_640_case_rq_matrix.jsonl`.", "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-640", action="store_true", default=True)
    args = parser.parse_args()
    manifest = cases(args.manifest)
    if args.require_640 and len(manifest) != 640:
        raise SystemExit(f"expected exactly 640 cases, found {len(manifest)} in {args.manifest}")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    frameworks = registry["framework_adapters"]
    rqs = registry["rqs"]
    artifact_rows, artifact_paths = load_artifacts(args.raw_dir, registry)
    controlled = controlled_indexes(args.raw_dir)

    exact: dict[str, dict[str, list[dict]]] = {}
    for framework, rows in artifact_rows.items():
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            case_id = row.get("manifest_case_id") or row.get("case_id")
            if case_id:
                grouped[str(case_id)].append(row)
        exact[framework] = grouped

    case_rq_rows = []
    framework_rows = []
    for case in manifest:
        for rq_spec in rqs:
            rq = rq_spec["rq"]
            scoped = in_scope(case, rq_spec)
            controlled_status, controlled_artifacts = (controlled_case_status(controlled, case, rq)
                                                       if scoped else ("not_applicable_case_trigger", []))
            case_rq_rows.append({
                "case_id": case["case_id"], "model_id": case["model_id"],
                "structure": case["structure"], "phase": case["phase"],
                "rq": rq, "rq_name": rq_spec["name"], "case_in_scope": scoped,
                "scope_reason": rq_spec["case_scope"]["reason"],
                "hypotheses": sorted(rq_spec["hypotheses"]),
                "controlled_status": controlled_status,
                "controlled_artifacts": controlled_artifacts,
            })
            for framework, framework_spec in frameworks.items():
                classification = framework_class(framework_spec, rq)
                rows = rq_native_rows(framework, rq, [
                    row for row in exact[framework].get(case["case_id"], []) if success(row)])
                choices = native_candidates(framework, rows)
                granularity = evidence_granularity(framework, rows)
                if not scoped:
                    status = "not_applicable_case_trigger"
                elif classification != "runtime":
                    status = classification
                elif len(choices) >= 2:
                    status = ("paired_native_projected_runtime"
                              if granularity.startswith("projected_")
                              else "paired_native_runtime")
                elif rows:
                    status = "native_runtime_unpaired"
                elif artifact_rows[framework]:
                    status = "native_external_not_case_matched"
                else:
                    status = "native_runtime_missing"
                framework_rows.append({
                    "case_id": case["case_id"], "model_id": case["model_id"],
                    "structure": case["structure"], "phase": case["phase"],
                    "framework": framework, "rq": rq, "rq_name": rq_spec["name"],
                    "hypotheses": sorted(rq_spec["hypotheses"]),
                    "case_in_scope": scoped, "framework_classification": classification,
                    "status": status, "native_row_count": len(rows),
                    "native_candidates": choices, "artifact": str(artifact_paths[framework]),
                    "evidence_granularity": granularity,
                    "controlled_case_status": controlled_status,
                })

    expected = len(manifest) * len(frameworks) * len(rqs)
    if len(framework_rows) != expected:
        raise AssertionError(f"matrix cardinality {len(framework_rows)} != {expected}")
    by_frq: dict[tuple[str, str], Counter] = defaultdict(Counter)
    in_scope_counts = Counter()
    for row in framework_rows:
        by_frq[(row["framework"], row["rq"])][row["status"]] += 1
        if row["case_in_scope"]:
            in_scope_counts[(row["framework"], row["rq"])] += 1
    status_counts = Counter(row["status"] for row in framework_rows)
    required = [row for row in framework_rows
                if row["case_in_scope"] and row["framework_classification"] == "runtime"]
    summary = {
        # Version 2 distinguishes exact native case execution from an explicit
        # projection of the case onto a framework-owned boundary.
        "schema_version": 2, "manifest": str(args.manifest), "raw_dir": str(args.raw_dir),
        "case_count": len(manifest), "framework_count": len(frameworks), "rq_count": len(rqs),
        "matrix_cell_count": len(framework_rows), "expected_matrix_cell_count": expected,
        "status_counts": dict(status_counts),
        "applicable_runtime_cell_count": len(required),
        "paired_native_runtime_cell_count": sum(row["status"] == "paired_native_runtime" for row in required),
        "paired_native_projected_runtime_cell_count": sum(
            row["status"] == "paired_native_projected_runtime" for row in required),
        "all_applicable_cells_natively_validated": bool(required) and all(
            row["status"] == "paired_native_runtime" for row in required),
        "framework_rq_summary": [
            {"framework": framework, "rq": rq, "in_scope_cases": in_scope_counts[(framework, rq)],
             "counts": dict(by_frq[(framework, rq)])}
            for framework in frameworks for rq in (item["rq"] for item in rqs)
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "v13_640_case_rq_matrix.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in case_rq_rows),
        encoding="utf-8")
    (args.output_dir / "v13_640_framework_rq_matrix.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in framework_rows),
        encoding="utf-8")
    (args.output_dir / "v13_640_matrix_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "V13_640_FRAMEWORK_RQ_MATRIX.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({"cases": len(manifest), "cells": len(framework_rows),
                      "paired_native": summary["paired_native_runtime_cell_count"],
                      "complete": summary["all_applicable_cells_natively_validated"],
                      "report": str(args.output_dir / "V13_640_FRAMEWORK_RQ_MATRIX.md")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
