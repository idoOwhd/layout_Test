#!/usr/bin/env python3
"""Select exactly the non-OOM 640-case rows affected by the repair.

The MoE fan-in repair changes every materialized MoE workload, so all MoE
cases that were runnable in the base run must be repeated for both eager and
Inductor.  The boundary harness repair changes per-strategy accounting, so all
runnable MoE and Mamba2 boundary cases are repeated.  Existing OOM cases are
deliberately excluded: this is an incremental non-OOM repair run, not an OOM
policy experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


OOM = {"oom_preflight", "oom_runtime"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_cases(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value["cases"] if isinstance(value, dict) else value


def write_manifest(path: Path, cases: list[dict], *, reason: str) -> None:
    path.write_text(json.dumps({
        "schema_version": 1,
        "selection_reason": reason,
        "case_count": len(cases),
        "cases": cases,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = args.base_result / "v13_640/full_subgraphs/raw"
    manifest_path = raw / "rq_llm_all_640_cases.json"
    whole_path = raw / "llm_640_pytorch_triton.jsonl"
    boundary_path = raw / "llm_640_boundary_layout_sweep.jsonl"
    for path in (manifest_path, whole_path, boundary_path):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    cases = load_cases(manifest_path)
    by_id = {case["case_id"]: case for case in cases}
    whole_rows = read_jsonl(whole_path)
    boundary_rows = read_jsonl(boundary_path)

    whole_status: dict[str, list[str]] = {}
    for row in whole_rows:
        whole_status.setdefault(row["case_id"], []).append(row.get("status", "missing"))
    boundary_status: dict[str, list[str]] = {}
    for row in boundary_rows:
        boundary_status.setdefault(row["case_id"], []).append(row.get("status", "missing"))

    whole_ids = sorted(
        case_id for case_id, statuses in whole_status.items()
        if by_id[case_id]["structure"] == "moe"
        and any(status not in OOM for status in statuses)
    )
    boundary_ids = sorted(
        case_id for case_id, statuses in boundary_status.items()
        if by_id[case_id]["structure"] in {"moe", "mamba2"}
        and any(status not in OOM for status in statuses)
    )
    write_manifest(
        args.output_dir / "whole_moe_non_oom.json",
        [by_id[case_id] for case_id in whole_ids],
        reason="all runnable MoE cases affected by logical-fan-in initialization repair",
    )
    write_manifest(
        args.output_dir / "boundary_moe_mamba_non_oom.json",
        [by_id[case_id] for case_id in boundary_ids],
        reason="all runnable MoE/Mamba2 cases affected by data or per-strategy boundary accounting repair",
    )
    selection = {
        "base_result": str(args.base_result.resolve()),
        "source_manifest": str(manifest_path.resolve()),
        "whole_case_count": len(whole_ids),
        "boundary_case_count": len(boundary_ids),
        "whole_case_ids": whole_ids,
        "boundary_case_ids": boundary_ids,
        "excluded_policy": "base-run OOM-only cases excluded",
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
