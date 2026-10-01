#!/usr/bin/env python3
"""Select every executable case applicable to one layout research question."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


VALID_RQS = {f"L-RQ{index}" for index in range(1, 11)}


def normalize_rq(value: str) -> str:
    value = value.strip().upper()
    if value.startswith("RQ"):
        value = f"L-{value}"
    if value not in VALID_RQS:
        raise argparse.ArgumentTypeError(
            f"RQ must be one of L-RQ1..L-RQ10, got {value!r}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter a real-world case manifest by its target_rqs field")
    parser.add_argument("--rq", type=normalize_rq, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--attention-tsv", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-attention-tsv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_cases = payload["cases"] if isinstance(payload, dict) else payload
    selected = [case for case in source_cases
                if args.rq in case.get("target_rqs", [])]
    selected_ids = {case["case_id"] for case in selected}

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": payload.get("schema_version", 1)
        if isinstance(payload, dict) else 1,
        "selected_rq": args.rq,
        "source_manifest": str(args.manifest.resolve()),
        "source_case_count": len(source_cases),
        "case_count": len(selected),
        "cases": selected,
    }
    args.output_manifest.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    attention_rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    with args.attention_tsv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        attention_rows = [row for row in reader if row.get("case_id") in selected_ids]
    with args.output_attention_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(attention_rows)

    summary = {
        "rq": args.rq,
        "source_case_count": len(source_cases),
        "selected_case_count": len(selected),
        "selected_attention_case_count": len(attention_rows),
        "single_gpu_blocked": args.rq == "L-RQ5",
        "selection_rule": "rq in case.target_rqs",
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
