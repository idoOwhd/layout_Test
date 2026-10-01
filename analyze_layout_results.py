#!/usr/bin/env python3
"""Compute native-layout regret and framework x subgraph coverage.

The analyzer refuses to invent a native layout: a row must carry
``layout_role=native``/``is_native=true`` or be named by ``--native-map``.
Without that evidence it reports the best measured candidate but leaves native
regret null.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
TARGET_FRAMEWORKS = ("sglang", "vllm", "cutlass", "triton", "tvm", "hexcute")
ALIASES = {"triton-explicit": "triton", "torchinductor-triton": "triton", "inductor": "triton"}


def truth(value) -> bool:
    return str(value).lower() in {"1", "true", "yes", "success"}


def load_rows(run_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(run_dir.glob("*.csv")):
        try:
            with path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f): rows.append({**row, "_file": path.name})
        except Exception:
            pass
    for path in sorted(run_dir.glob("*.jsonl")):
        if path.name in {"status.jsonl", "layout_optimality.jsonl", "compiler_layout_evidence.jsonl"}: continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip(): rows.append({**json.loads(line), "_file": path.name})
        except Exception:
            pass
    return rows


def normalize(row: dict, native_map: dict[str, str]) -> dict | None:
    framework = str(row.get("framework") or row.get("backend") or "").lower()
    framework = ALIASES.get(framework, framework)
    case_id = row.get("case_id"); layout = row.get("layout")
    try: p50 = float(row["p50_ms"])
    except (KeyError, TypeError, ValueError): return None
    if not case_id or not framework or not math.isfinite(p50) or p50 <= 0: return None
    status_ok = str(row.get("status", "success")).lower() == "success"
    correct = status_ok and (truth(row["correct"]) if "correct" in row else True)
    key = f"{framework}|{case_id}"
    native = truth(row.get("is_native", False)) or str(row.get("layout_role", "")).lower() == "native" or (layout is not None and native_map.get(key) == layout)
    return {"framework": framework, "case_id": str(case_id), "layout": layout or "implicit_unknown", "p50_ms": p50,
            "correct": correct, "is_native": native, "source_file": row["_file"],
            "comparison_scope": row.get("comparison_scope", "unspecified")}


def manifest_cases(path: Path, nested: bool) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["cases"] if nested else data
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--llm-manifest", type=Path, default=HERE.parent / "real_world_shapes" / "real_world_shape_manifest.json")
    p.add_argument("--vision-manifest", type=Path, default=HERE.parent / "vision_shapes" / "vision_common_subgraph_manifest.json")
    p.add_argument("--native-map", type=Path)
    p.add_argument("--output-prefix", type=Path)
    a = p.parse_args(); native_map = json.loads(a.native_map.read_text()) if a.native_map else {}
    measured = [x for row in load_rows(a.run_dir) if (x := normalize(row, native_map)) is not None]
    groups = defaultdict(list)
    for row in measured:
        if row["correct"]: groups[(row["framework"], row["case_id"])].append(row)
    optimality = []
    for (framework, case_id), rows in sorted(groups.items()):
        best = min(rows, key=lambda x: x["p50_ms"]); natives = [x for x in rows if x["is_native"]]
        native = min(natives, key=lambda x: x["p50_ms"]) if natives else None
        confounded = any(x["comparison_scope"] == "serving_policy_confounded" for x in rows)
        optimality.append({"framework": framework, "case_id": case_id, "candidate_count": len(rows),
                           "best_layout": best["layout"], "best_p50_ms": best["p50_ms"],
                           "native_layout": native["layout"] if native else None,
                           "native_p50_ms": native["p50_ms"] if native else None,
                           "native_regret": ((native["p50_ms"] - best["p50_ms"]) / best["p50_ms"]) if native and not confounded else None,
                           "optimality_status": "confounded" if confounded else "measured" if native and len(rows) >= 2 else "native_missing" if not native else "alternatives_missing",
                           "source_files": sorted({x["source_file"] for x in rows})})
    llm = manifest_cases(a.llm_manifest, False); vision = manifest_cases(a.vision_manifest, True)
    expected = []
    for row in llm:
        for fw in TARGET_FRAMEWORKS: expected.append((fw, row["case_id"], "llm", row["structure"]))
    # Keep every requested framework visible for every visual case.  A serving
    # system may ultimately report unsupported/delegated, but omitting its cell
    # would silently improve coverage and fail the user's all-framework matrix.
    for row in vision:
        for fw in TARGET_FRAMEWORKS:
            expected.append((fw, row["case_id"], row["domain"], row["motif"]))
    seen = {(x["framework"], x["case_id"]) for x in measured}
    counts = Counter((domain, fw, "measured" if (fw, case) in seen else "missing") for fw, case, domain, motif in expected)
    prefix = a.output_prefix or a.run_dir / "layout_analysis"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{prefix}_optimality.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in optimality), encoding="utf-8")
    matrix = []
    for framework, case_id, domain, motif in expected:
        measured_cell = (framework, case_id) in seen
        if measured_cell:
            reason = None
        elif domain in {"image", "video"} and framework in {"sglang", "vllm"}:
            reason = "native_visual_adapter_missing_or_backend_delegated"
        else:
            reason = "no_valid_performance_row"
        matrix.append({"framework": framework, "case_id": case_id, "domain": domain, "subgraph": motif,
                       "status": "measured" if measured_cell else "missing", "missing_reason": reason})
    Path(f"{prefix}_matrix.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in matrix), encoding="utf-8")
    coverage = {"expected_cells": len(expected), "measured_cells": sum((fw, case) in seen for fw, case, _, _ in expected),
                "native_observed_cells": sum(x["native_layout"] is not None for x in optimality),
                "optimality_measured_cells": sum(x["optimality_status"] == "measured" for x in optimality),
                "by_domain_framework": [{"domain": d, "framework": f, "measured": counts[d, f, "measured"], "missing": counts[d, f, "missing"]}
                                        for d in ("llm", "image", "video") for f in TARGET_FRAMEWORKS if counts[d, f, "measured"] + counts[d, f, "missing"]]}
    Path(f"{prefix}_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    lines = ["# Framework × subgraph layout coverage", "", f"- Expected applicable cells: {coverage['expected_cells']}",
             f"- Cells with a performance row: {coverage['measured_cells']}", f"- Cells with an observed native layout: {coverage['native_observed_cells']}",
             f"- Cells where native regret is measurable: {coverage['optimality_measured_cells']}", "", "| domain | framework | measured | missing |", "|---|---|---:|---:|"]
    for row in coverage["by_domain_framework"]: lines.append(f"| {row['domain']} | {row['framework']} | {row['measured']} | {row['missing']} |")
    lines += ["", "The companion `layout_analysis_matrix.jsonl` retains all requested framework/case cells and an explicit missing reason.",
              "A fastest row is not called optimal unless a native layout and at least one legal alternative were both measured.", ""]
    Path(f"{prefix}_coverage.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(coverage, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
