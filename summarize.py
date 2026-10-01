#!/usr/bin/env python3
"""Summarize one run directory without third-party dependencies."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(encoding="utf-8") as f: return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(); p.add_argument("run_dir", type=Path); a = p.parse_args()
    status_path = a.run_dir / "status.jsonl"
    status = [json.loads(x) for x in status_path.read_text().splitlines()] if status_path.exists() else []
    lines = ["# Layout experiment run", "", "## Execution status", "",
             "| step | status | detail |", "|---|---|---|"]
    for row in status:
        lines.append(f"| {row['step']} | {row['status']} | {str(row.get('detail', '')).replace('|', '/')} |")
    audit_path = a.run_dir / "requirements_audit.json"
    if audit_path.exists():
        audit_rows = json.loads(audit_path.read_text(encoding="utf-8"))["requirements"]
        code_counts = defaultdict(int)
        for row in audit_rows: code_counts[row["code_status"]] += 1
        lines += ["", "## Requirements audit", "",
                  f"Code coverage: complete={code_counts['complete']}, partial={code_counts['partial']}, missing={code_counts['missing']}.", "",
                  "See [REQUIREMENTS_AUDIT.md](REQUIREMENTS_AUDIT.md) for the evidence and validation gaps."]
    vision_summary = a.run_dir / "vision_catalog" / "vision_manifest_summary.json"
    if vision_summary.exists():
        vision = json.loads(vision_summary.read_text(encoding="utf-8"))
        lines += ["", "## Modern vision corpus", "",
                  f"2025–2026 families={vision['models']}; common motifs={len(vision['common_motifs'])}; multi-shape cases={vision['shape_cases']}."]
    shared_validation = a.run_dir / "shared_link_validation.json"
    vision_pins = a.run_dir / "vision_source_revisions.json"
    source_evidence = a.run_dir / "source_evidence.json"
    if shared_validation.exists() or vision_pins.exists() or source_evidence.exists():
        lines += ["", "## Reproducible source evidence", ""]
        if shared_validation.exists():
            links = json.loads(shared_validation.read_text(encoding="utf-8"))
            lines.append(f"ChatGPT shares accessible={sum(row['accessible'] for row in links['sources'])}/{len(links['sources'])}; Codex local references are intentionally excluded.")
        if vision_pins.exists():
            pins = json.loads(vision_pins.read_text(encoding="utf-8"))
            lines.append(f"Visual-model sources pinned={sum(row['status'] == 'pinned' for row in pins['sources'])}/{len(pins['sources'])}.")
        if source_evidence.exists():
            evidence = json.loads(source_evidence.read_text(encoding="utf-8"))
            lines.append(f"Frameworks with source hits={sum(row['status'] == 'success' for row in evidence)}/{len(evidence)}.")
    coverage_path = a.run_dir / "layout_analysis_coverage.json"
    if coverage_path.exists():
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        lines += ["", "## Native-layout optimality coverage", "",
                  f"Measured cells={coverage['measured_cells']}/{coverage['expected_cells']}; native layout observed={coverage['native_observed_cells']}; native regret measurable={coverage['optimality_measured_cells']}.", "",
                  "See [layout_analysis_coverage.md](layout_analysis_coverage.md)."]
    lines += ["", "## Paired layout winners", "", "Only rows with a numeric `p50_ms` participate.", "",
              "| file | case | winner | p50 ms | compared layouts |", "|---|---|---|---:|---|"]
    for path in sorted(a.run_dir.glob("*.csv")):
        try: rows = read_csv(path)
        except Exception: continue
        groups = defaultdict(list)
        for row in rows:
            if row.get("p50_ms") not in (None, ""):
                try: groups[row.get("case_id", "unknown")].append((float(row["p50_ms"]), row.get("layout", "unknown")))
                except ValueError: pass
        for case, values in groups.items():
            values.sort()
            lines.append(f"| {path.name} | {case} | {values[0][1]} | {values[0][0]:.6f} | {', '.join(v[1] for v in values)} |")
        if path.name == "cutlass_softmax_boundary.csv":
            for row in rows:
                winner = row.get("winner", "unknown")
                candidates = []
                for key in ("tiled_direct_ms", "transform_plus_flat_ms", "transformed_flat_ms"):
                    if row.get(key): candidates.append(f"{key}={row[key]}")
                lines.append(f"| {path.name} | rows={row.get('rows')},cols={row.get('cols')},tile={row.get('tile_rows')}x{row.get('tile_cols')} | {winner} | — | {', '.join(candidates)} |")
    lines += ["", "## Complete-subgraph coverage", "", "| file | backend | successful / total | GeoMean p50 ms |", "|---|---|---:|---:|"]
    for path in sorted(a.run_dir.glob("llm_640_*.jsonl")):
        per_backend = defaultdict(list); total = defaultdict(int)
        for text in path.read_text(encoding="utf-8").splitlines():
            if not text.strip(): continue
            row = json.loads(text); backend = row.get("backend", "unknown"); total[backend] += 1
            if row.get("status") == "success" and row.get("p50_ms"):
                per_backend[backend].append(float(row["p50_ms"]))
        for backend in sorted(total):
            values = per_backend[backend]
            gm = math.exp(sum(math.log(x) for x in values) / len(values)) if values else float("nan")
            lines.append(f"| {path.name} | {backend} | {len(values)} / {total[backend]} | {gm:.6f} |")
    lines += ["", "## Interpretation guardrails", "",
              "- Static address metrics are mechanism checks, not latency predictions.",
              "- A microbenchmark establishes a local causal effect; a complete-subgraph speedup is required for the end-to-end claim.",
              "- Serving systems that delegate register/shared layout to an attention backend are not credited with that backend's compiler decision.",
              "- Layout-conversion time, padding, cache capacity and correctness are included in the selected end-to-end route.", ""]
    report = a.run_dir / "REPORT.md"; report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__": main()
