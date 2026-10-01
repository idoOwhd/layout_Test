#!/usr/bin/env python3
"""Build a manifest containing only non-OOM boundary rows needing a retry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads((args.primary_run / "manifests/boundary_moe_mamba_non_oom.json").read_text())
    cases = source["cases"] if isinstance(source, dict) else source
    by_id = {case["case_id"]: case for case in cases}
    rows = [json.loads(line) for line in
            (args.primary_run / "raw/boundary_moe_mamba_non_oom.jsonl").read_text().splitlines()
            if line.strip()]
    retry = sorted({
        row["case_id"] for row in rows
        if row.get("subgraph") == "mamba2"
        and row.get("status") not in {"success", "oom_preflight", "oom_runtime"}
    })
    value = {"schema_version": 1,
             "selection_reason": "non-OOM Mamba boundary failures from primary repair run",
             "primary_run": str(args.primary_run.resolve()),
             "case_count": len(retry),
             "case_ids": retry,
             "cases": [by_id[case_id] for case_id in retry]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"case_count": len(retry), "case_ids": retry}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
