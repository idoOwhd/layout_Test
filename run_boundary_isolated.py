#!/usr/bin/env python3
"""Run boundary cases in fresh processes to isolate Dynamo and CUDA state."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_cases(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value["cases"] if isinstance(value, dict) else value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
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
    cases = load_cases(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    parts = args.output.parent / f".{args.output.stem}_isolated_parts"
    parts.mkdir(parents=True, exist_ok=False)
    all_rows, statuses = [], []
    for index, case in enumerate(cases, 1):
        case_id = case["case_id"]
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in case_id)
        part = parts / f"{index:04d}_{safe}.jsonl"
        log = args.log_dir / f"isolated_{index:04d}_{safe}.log"
        command = [args.python, str(args.runner), "--manifest", str(args.manifest),
                   "--output", str(part), "--case-id", case_id,
                   "--physical-device-index", str(args.physical_device_index),
                   "--warmup", str(args.warmup), "--iterations", str(args.iterations)]
        if args.extended_rq_counterfactuals:
            command.append("--extended-rq-counterfactuals")
        try:
            with log.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                           timeout=args.timeout, check=False)
            state = "success" if completed.returncode == 0 else "failed"
            detail = f"exit={completed.returncode}"
        except subprocess.TimeoutExpired:
            state, detail = "timeout", f"timeout_seconds={args.timeout}"
        rows = []
        if part.exists():
            rows = [json.loads(line) for line in part.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            all_rows.extend(rows)
        if state != "success" or not rows:
            all_rows.append({"case_id": case_id, "model_id": case.get("model_id"),
                             "subgraph": case.get("structure"), "phase": case.get("phase"),
                             "framework": "PyTorch", "experiment": "boundary_layout_sweep",
                             "status": state, "error": detail, "isolated_log": str(log)})
        statuses.append({"case_id": case_id, "status": state, "rows": len(rows),
                         "detail": detail, "log": str(log)})
        print(f"[{index}/{len(cases)}] {case_id}: {state} rows={len(rows)}", flush=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                   for row in all_rows), encoding="utf-8")
    status_path = args.output.with_suffix(".isolated_status.json")
    status_path.write_text(json.dumps(statuses, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    failures = sum(row["status"] != "success" for row in statuses)
    print(json.dumps({"cases": len(cases), "failures": failures,
                      "output": str(args.output), "status": str(status_path)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
