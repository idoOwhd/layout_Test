#!/usr/bin/env python3
"""Run CUDA counterfactuals in independent processes with order reversal."""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    all_rows: list[dict[str, str]] = []
    fields: list[str] | None = None
    for repetition in range(args.repetitions):
        command = [str(args.binary), "--case-file", str(args.case_file),
                   "--warmup", str(args.warmup), "--iterations", str(args.iterations),
                   "--order-seed", str(repetition)]
        if args.quick:
            command.append("--quick")
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode:
            raise SystemExit(
                f"CUDA repetition {repetition} failed ({result.returncode}):\n{result.stderr}")
        reader = csv.DictReader(io.StringIO(result.stdout))
        if reader.fieldnames is None:
            raise SystemExit(f"CUDA repetition {repetition} emitted no CSV header")
        if fields is None:
            fields = list(reader.fieldnames) + ["process_repetition"]
        elif list(reader.fieldnames) != fields[:-1]:
            raise SystemExit("CUDA CSV schema changed between process repetitions")
        for row in reader:
            row["process_repetition"] = str(repetition)
            all_rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"repetitions={args.repetitions} rows={len(all_rows)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
