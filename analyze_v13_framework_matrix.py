#!/usr/bin/env python3
"""Produce a fail-closed v13 RQ x framework native-evidence matrix."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent


def artifact_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def is_success(row: dict) -> bool:
    return row.get("status", "success") in {"success", "True", "true", "1", ""}


def relevant_runtime_rows(framework: str, rq: str, rows: list[dict]) -> list[dict]:
    """Return only rows that manipulate the decision named by this RQ."""
    if framework in {"vLLM", "SGLang"}:
        if rq in {"L-RQ4", "L-RQ9"}:
            return [row for row in rows if row.get("comparison_scope") == "layout_only"]
        if rq == "L-RQ10":
            return [row for row in rows if row.get("comparison_scope") == "page_size_only"
                    and row.get("page_size") not in {None, "", 0, "0"}]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=HERE / "v13_experiment_registry.json")
    parser.add_argument("--source-json", type=Path)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    source_by_rq_stack = defaultdict(int)
    if args.source_json and args.source_json.exists():
        source = json.loads(args.source_json.read_text(encoding="utf-8"))
        for hypothesis in source.get("hypotheses", []):
            for stack in hypothesis.get("supporting_stacks", []):
                source_by_rq_stack[(hypothesis["rq"], stack)] += 1

    aliases = {
        "TVM": ("TVM Relax", "TVM TIR", "TVM MetaSchedule"),
        "CUTLASS/CuTe": ("CUTLASS/CuTe",),
        "vLLM": ("vLLM",), "SGLang": ("SGLang",),
        "Triton": ("Triton",), "Hexcute": ("Hexcute",),
    }
    rows = []
    frameworks = registry["framework_adapters"]
    for framework, spec in frameworks.items():
        artifact = args.raw_dir / spec["artifact"]
        if framework == "Triton" and (args.raw_dir / "triton_640_kv_layout.csv").exists():
            artifact = args.raw_dir / "triton_640_kv_layout.csv"
        elif framework == "TVM" and (args.raw_dir / "tvm_640_rq_observations.csv").exists():
            artifact = args.raw_dir / "tvm_640_rq_observations.csv"
        elif framework == "CUTLASS/CuTe" and (args.raw_dir / "cutlass_640_softmax_boundary.csv").exists():
            artifact = args.raw_dir / "cutlass_640_softmax_boundary.csv"
        elif framework == "Hexcute" and (args.raw_dir / "hexcute_640.jsonl").exists():
            artifact = args.raw_dir / "hexcute_640.jsonl"
        artifact_data = artifact_rows(artifact)
        total = len(artifact_data)
        all_success = sum(is_success(row) for row in artifact_data)
        for index in range(1, 11):
            rq = f"L-RQ{index}"
            relevant = relevant_runtime_rows(framework, rq, artifact_data)
            success = sum(is_success(row) for row in relevant)
            source_rows = sum(source_by_rq_stack[(rq, alias)] for alias in aliases[framework])
            if rq in spec.get("blocked_rqs", []):
                status = "blocked_single_gpu"
            elif rq in spec.get("architecture_blocked_rqs", []):
                status = "blocked_architecture_unsupported"
            elif rq in spec.get("not_applicable_rqs", []):
                status = "not_applicable_framework_does_not_own_decision"
            elif rq in spec.get("missing_adapter_rqs", []):
                status = "native_adapter_missing"
            elif rq in spec.get("runtime_rqs", []):
                status = "native_runtime_present" if success else "native_runtime_missing"
            elif rq in spec.get("source_only_rqs", []):
                status = "source_observation_only" if source_rows else "source_evidence_missing"
            else:
                status = "unmapped"
            runtime_used = success if status == "native_runtime_present" else 0
            rows.append({"rq": rq, "framework": framework, "status": status,
                         "artifact": str(artifact), "artifact_rows": total,
                         "successful_rows": runtime_used,
                         "available_artifact_rows": all_success,
                         "relevant_artifact_rows": len(relevant),
                         "source_hypothesis_links": source_rows,
                         "limitations": spec["limitations"]})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "rows": rows}
    (args.output_dir / "v13_framework_matrix.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# v13 RQ × framework evidence matrix", "",
             "> `native_runtime_present` only means the registered framework-native artifact ran. It does not automatically support every hypothesis in that RQ; see `V13_RQ_VALIDATION_REPORT.md` for hypothesis verdicts.", "",
             "| RQ | Framework | Status | Relevant native rows | Source hypothesis links |",
             "|---|---|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['rq']} | {row['framework']} | **{row['status']}** | "
                     f"{row['successful_rows']} | {row['source_hypothesis_links']} |")
    lines += ["", "## Framework limitations", ""]
    for framework, spec in frameworks.items():
        lines.append(f"- **{framework}**: {spec['limitations']}")
    lines += ["", "## Guardrail", "",
              "A CUDA-reference result is never copied into a vLLM, SGLang, CUTLASS, Triton, TVM, or Hexcute cell. A missing native adapter remains missing.", ""]
    (args.output_dir / "V13_FRAMEWORK_MATRIX.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "report": str(args.output_dir / "V13_FRAMEWORK_MATRIX.md")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
