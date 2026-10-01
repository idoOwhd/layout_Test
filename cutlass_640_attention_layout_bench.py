#!/usr/bin/env python3
"""Run the CUTLASS/CUDA softmax layout boundary on exact attention shapes.

The adapter compiles once, deduplicates equal (rows, cols) contracts, and maps
the measurements back to every source case.  Oversized explicit score matrices
are recorded as preflight skips rather than replaced by smaller proxy shapes.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ATTENTION = {"gqa", "sliding_attention", "sparse_attention", "mla"}
TILES = ((1, 16), (8, 8), (16, 16), (16, 32), (32, 32))


def load(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value["cases"] if isinstance(value, dict) else value


def geometry(case: dict) -> tuple[int, int]:
    shape = case["shape"]
    heads = int(shape.get("num_query_heads", shape.get("num_heads", 1)))
    rows = int(shape.get("batch", 1)) * int(shape.get("query_length", 1)) * heads
    cols = int(shape.get("kv_length", shape.get("query_length", 1)))
    return rows, cols


def balanced_cases(cases: list[dict], limit: int) -> list[dict]:
    selected, seen = [], set()
    for case in cases:
        key = (case["structure"], case["phase"])
        if key not in seen:
            selected.append(case)
            seen.add(key)
            if len(selected) == limit:
                return selected
    selected_ids = {case["case_id"] for case in selected}
    for case in cases:
        if case["case_id"] not in selected_ids:
            selected.append(case)
            if len(selected) == limit:
                break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True,
                        help="run_softmax_layout_boundary.py")
    parser.add_argument("--cutlass", type=Path, required=True)
    parser.add_argument("--arch", default="sm_86")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--max-score-elements", type=int, default=100_000_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--balanced-limit", type=int)
    parser.add_argument("--phase", choices=("prefill", "decode"))
    parser.add_argument("--full-verify-max-elements", type=int, default=4_194_304,
                        help="fully verify matrices no larger than this many elements")
    args = parser.parse_args()
    source = [case for case in load(args.manifest) if case["structure"] in ATTENTION]
    if args.phase:
        source = [case for case in source if case["phase"] == args.phase]
    if args.balanced_limit:
        source = balanced_cases(source, args.balanced_limit)
    elif args.limit:
        source = source[:args.limit]
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for case in source:
        grouped[geometry(case)].append(case)

    subprocess.run([sys.executable, str(args.driver), "--cutlass", str(args.cutlass),
                    "--arch", args.arch, "--binary", str(args.binary), "--build-only"],
                   check=True)
    records = []
    for (rows, cols), mapped_cases in sorted(grouped.items()):
        if rows * cols > args.max_score_elements:
            for case in mapped_cases:
                records.append({"manifest_case_id": case["case_id"],
                                "model_id": case["model_id"],
                                "structure": case["structure"], "phase": case["phase"],
                                "rows": rows, "cols": cols, "status": "oom_preflight",
                                "reason": "explicit score matrix exceeds max_score_elements"})
            continue
        shape_records = []
        verify_rows = rows if rows * cols <= args.full_verify_max_elements else min(16, rows)
        verification_scope = "full_output" if verify_rows == rows else "leading_rows_sample"
        for tile_rows, tile_cols in TILES:
            if rows % tile_rows or cols % tile_cols or tile_cols % 8:
                continue
            command = [str(args.binary), "--rows", str(rows), "--cols", str(cols),
                       "--tile-rows", str(tile_rows), "--tile-cols", str(tile_cols),
                       "--warmup", str(args.warmup), "--iterations", str(args.iterations),
                       "--verify-rows", str(verify_rows), "--csv"]
            completed = subprocess.run(command, check=False, text=True, capture_output=True)
            if completed.returncode:
                shape_records.append({"rows": rows, "cols": cols,
                                      "tile_rows": tile_rows, "tile_cols": tile_cols,
                                      "status": "failed", "error": completed.stderr[-2000:]})
                continue
            parsed = list(csv.DictReader(io.StringIO(completed.stdout)))
            if len(parsed) != 1:
                raise RuntimeError(f"unexpected CUTLASS output for {rows}x{cols}: {completed.stdout}")
            shape_records.append({**parsed[0], "status": "success",
                                  "verification_scope": verification_scope,
                                  "verified_rows": verify_rows,
                                  "projection_kind": "explicit_attention_score_matrix"})
        for case in mapped_cases:
            for row in shape_records:
                records.append({**row, "manifest_case_id": case["case_id"],
                                "model_id": case["model_id"],
                                "structure": case["structure"], "phase": case["phase"],
                                "measurement_reused_for_equal_shape": len(mapped_cases) > 1})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in records for key in row})
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)
    print(json.dumps({"source_cases": len(source), "unique_shapes": len(grouped),
                      "rows": len(records), "output": str(args.output)}, ensure_ascii=False))
    attempted = [row for row in records if row.get("status") != "oom_preflight"]
    # An all-failed executable sweep is infrastructure failure, not a valid
    # zero-speedup experiment.  OOM preflight-only catalogs remain valid data.
    if attempted and not any(row.get("status") == "success" for row in attempted):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
