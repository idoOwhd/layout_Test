#!/usr/bin/env python3
"""Run ordinary boundary cases together and Mamba cases in isolation.

Mamba's higher-order associative scan has process-global Dynamo cache state.
Keeping its cases in fresh processes avoids cache exhaustion without paying
640 Python startups for the full catalog.  The merged output preserves source
manifest order and strategy order.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


STRATEGY_ORDER = {"native_contiguous": 0, "alternate_strided_view": 1,
                  "repair_to_contiguous_each_call": 2}


def load(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value["cases"] if isinstance(value, dict) else value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--isolated-runner", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--physical-device-index", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--extended-rq-counterfactuals", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    parts = args.output.parent / f".{args.output.stem}_hybrid_parts"
    parts.mkdir(parents=True, exist_ok=False)
    cases = load(args.manifest)
    ordinary = [case for case in cases if case["structure"] != "mamba2"]
    mamba = [case for case in cases if case["structure"] == "mamba2"]
    ordinary_manifest = parts / "ordinary.json"
    mamba_manifest = parts / "mamba.json"
    ordinary_manifest.write_text(json.dumps(ordinary, ensure_ascii=False, indent=2) + "\n")
    mamba_manifest.write_text(json.dumps(mamba, ensure_ascii=False, indent=2) + "\n")
    ordinary_output = parts / "ordinary.jsonl"
    mamba_output = parts / "mamba.jsonl"
    env = os.environ.copy()

    commands = []
    if ordinary:
        command = [args.python, str(args.runner),
            "--manifest", str(ordinary_manifest), "--output", str(ordinary_output),
            "--physical-device-index", str(args.physical_device_index),
            "--warmup", str(args.warmup), "--iterations", str(args.iterations)]
        if args.extended_rq_counterfactuals:
            command.append("--extended-rq-counterfactuals")
        commands.append(("ordinary", command))
    if mamba:
        command = [args.python, str(args.isolated_runner),
            "--runner", str(args.runner), "--python", args.python,
            "--manifest", str(mamba_manifest), "--output", str(mamba_output),
            "--log-dir", str(args.log_dir / "mamba_isolated"),
            "--physical-device-index", str(args.physical_device_index),
            "--warmup", str(args.warmup), "--iterations", str(args.iterations),
            "--timeout", str(args.timeout)]
        if args.extended_rq_counterfactuals:
            command.append("--extended-rq-counterfactuals")
        commands.append(("mamba_isolated", command))
    driver_status = []
    for name, command in commands:
        log = args.log_dir / f"{name}.log"
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, check=False)
        driver_status.append({"step": name, "exit_code": completed.returncode,
                              "log": str(log)})
        if completed.returncode:
            raise RuntimeError(f"{name} boundary driver failed with exit={completed.returncode}; {log}")

    result_rows = []
    for path in (ordinary_output, mamba_output):
        if path.exists():
            result_rows.extend(json.loads(line) for line in path.read_text().splitlines()
                               if line.strip())
    case_order = {case["case_id"]: index for index, case in enumerate(cases)}
    result_rows.sort(key=lambda row: (
        case_order.get(row.get("case_id"), len(cases)),
        STRATEGY_ORDER.get(row.get("strategy"), 99)))
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                   for row in result_rows), encoding="utf-8")
    status = {"case_count": len(cases), "ordinary_cases": len(ordinary),
              "isolated_mamba_cases": len(mamba), "row_count": len(result_rows),
              "drivers": driver_status}
    args.output.with_suffix(".hybrid_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
