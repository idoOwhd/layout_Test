#!/usr/bin/env python3
"""Analyze TVM-native RQ observations without upgrading proxies into facts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def grouped(rows: list[dict], experiment: str, extra: str | None = None):
    result = defaultdict(dict)
    for row in rows:
        if row.get("experiment") != experiment or row.get("correct") not in {"1", "True", "true"}:
            continue
        key = (row["case_id"], row.get(extra, "")) if extra else row["case_id"]
        result[key][row["strategy"]] = float(row["p50_ms"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.input)

    scans = grouped(rows, "consumer_scan")
    scan_winners, scan_details = Counter(), []
    static_scan = defaultdict(float)
    oracle_scan = 0.0
    for case, values in sorted(scans.items()):
        winner = min(values, key=values.get)
        scan_winners[winner] += 1
        oracle_scan += values[winner]
        for strategy, value in values.items():
            static_scan[strategy] += value
        scan_details.append({"case": case, "winner": winner,
                             "speedup_over_other": max(values.values()) / values[winner], "times_ms": values})
    scan_oracle_gain = min(static_scan.values()) / oracle_scan if oracle_scan else 1.0

    reuse = grouped(rows, "conversion_reuse", "reuse")
    conversion_details, conversion_wins, direct_wins = [], [], []
    for (case, reuse_count), values in sorted(reuse.items(), key=lambda item: (item[0][0], int(item[0][1]))):
        winner = min(values, key=values.get)
        detail = {"case": case, "reuse": int(reuse_count), "winner": winner, "times_ms": values}
        if winner == "convert_once_then_tiled":
            detail["speedup_vs_keep"] = values["keep_row_major"] / values[winner]
            conversion_wins.append(detail)
        elif winner == "keep_row_major":
            direct_wins.append(detail)
        conversion_details.append(detail)
    rq3_gain = max((row["speedup_vs_keep"] for row in conversion_wins), default=1.0)
    rq10_gain = max((values["convert_each_use_then_tiled"] / values["convert_once_then_tiled"]
                     for values in reuse.values()), default=1.0)

    fusion = grouped(rows, "fusion_order")
    fusion_winners, fusion_details = Counter(), []
    static_fusion = defaultdict(float)
    oracle_fusion = 0.0
    for case, values in sorted(fusion.items()):
        winner = min(values, key=values.get)
        fusion_winners[winner] += 1
        oracle_fusion += values[winner]
        for strategy, value in values.items():
            static_fusion[strategy] += value
        fusion_details.append({"case": case, "winner": winner,
                               "speedup_over_worst": max(values.values()) / values[winner], "times_ms": values})
    fusion_oracle_gain = min(static_fusion.values()) / oracle_fusion if oracle_fusion else 1.0

    summary = {
        "schema_version": 1,
        "input": str(args.input),
        "shape_count": len({(row["rows"], row["cols"]) for row in rows}),
        "RQ3": {"status": "supported" if conversion_wins and direct_wins and rq3_gain > 1.01 else "inconclusive",
                "observation": "conversion break-even depends on reuse",
                "max_conversion_aware_speedup": rq3_gain, "cases": conversion_details},
        "RQ7": {"status": "feasibility_supported",
                "observation": "a typed row-to-tiled conversion restores a legal edge",
                "performance_claim": False},
        "RQ8": {"status": "supported" if len(fusion_winners) > 1 and fusion_oracle_gain > 1.01 else "inconclusive",
                "observation": "materialization/conversion/fusion winner changes by shape",
                "winner_counts": dict(fusion_winners), "joint_oracle_speedup_vs_best_static": fusion_oracle_gain,
                "cases": fusion_details},
        "RQ9": {"status": "supported" if len(scan_winners) > 1 and scan_oracle_gain > 1.01 else "inconclusive",
                "observation": "per-shape layout selection versus one static layout",
                "winner_counts": dict(scan_winners), "oracle_speedup_vs_best_static": scan_oracle_gain,
                "cases": scan_details},
        "RQ10": {"status": "supported" if rq10_gain > 1.01 else "inconclusive",
                 "observation": "materialize a legal repair once instead of at every use",
                 "max_speedup_vs_repeated_repair": rq10_gain},
    }
    lines = ["# TVM native RQ observation analysis", "",
             f"- Physical shapes: {summary['shape_count']}",
             "- Direct kernel rows and measured-cost compositions remain separately labeled in the CSV.", "",
             "| RQ | Verdict | Key result |",
             "|---|---|---|"]
    lines += [
        f"| RQ3 | {summary['RQ3']['status']} | convert-once wins {len(conversion_wins)} points, direct wins {len(direct_wins)}; max conversion-aware gain {rq3_gain:.3f}x |",
        f"| RQ7 | feasibility_supported | correctness-checked row-to-tiled edge exists; no speedup claim |",
        f"| RQ8 | {summary['RQ8']['status']} | winners {dict(fusion_winners)}; per-shape joint choice {fusion_oracle_gain:.3f}x over best static strategy |",
        f"| RQ9 | {summary['RQ9']['status']} | winners {dict(scan_winners)}; oracle gain {scan_oracle_gain:.3f}x over best static layout |",
        f"| RQ10 | {summary['RQ10']['status']} | convert-once up to {rq10_gain:.3f}x over repeated repair |",
        "", "## Interpretation", "",
        "- RQ3 is supported only if both a no-conversion region and a convert-once region occur.",
        "- RQ8 support comes from winner inversion plus the aggregate per-shape oracle, not from assuming fusion always wins.",
        "- RQ9 is inconclusive when heterogeneous winners yield at most 1% aggregate regret; heterogeneity alone is not enough.",
        "- RQ7 establishes legality/correctness. It is intentionally excluded from native speedup-majority counts.",
        "- These rows cover TVM only. They do not fill vLLM/SGLang or distributed/cross-GPU cells.", "",
        "## Per-shape RQ8 winners", ""]
    for row in fusion_details:
        lines.append(f"- `{row['case']}`: {row['winner']}, worst/best={row['speedup_over_worst']:.3f}x")
    lines += ["", "## Conversion break-even points", ""]
    for row in conversion_wins:
        lines.append(f"- `{row['case']}`, reuse={row['reuse']}: convert once, {row['speedup_vs_keep']:.3f}x versus keeping row-major")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tvm_rq_analysis.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "TVM_RQ_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output_dir / "TVM_RQ_ANALYSIS.md"),
                      "verdicts": {rq: summary[rq]["status"] for rq in ("RQ3", "RQ7", "RQ8", "RQ9", "RQ10")}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
