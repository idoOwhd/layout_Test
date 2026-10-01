#!/usr/bin/env python3
"""Generate an explicit implementation/validation gap audit.

The audit is intentionally non-failing: missing research coverage is a report,
not a reason to hide the results that did run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent


def resolve_evidence(text: str) -> Path:
    return (HERE / text).resolve()


def load_run_status(run_dir: Path | None) -> dict[str, dict]:
    if run_dir is None or not (run_dir / "status.jsonl").exists():
        return {}
    rows = [json.loads(line) for line in (run_dir / "status.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["step"]: row for row in rows}


def runtime_note(req_id: str, status: dict[str, dict]) -> str:
    gates = {
        "R05": ["source_evidence"], "R06": ["triton_kv_layout", "shared_bank", "torch_layout", "tvm_layout", "cutlass_softmax"],
        "R07": ["sglang_native_serving"], "R08": ["vllm_native_serving"],
        "R09": ["cutlass_softmax"], "R10": ["triton_kv_layout", "llm_640_pytorch_triton"],
        "R11": ["tvm_layout"], "R12": ["hexcute_artifact"], "R16": ["graph_layout_optimizer"],
        "R13": ["llm_640_pytorch_triton", "llm_640_tilelang"], "R15": ["shared_bank_ncu"],
        "R18": ["environment", "static_unit_tests"], "R21": ["shared_bank", "triton_kv_layout", "torch_layout"],
        "R22": ["shared_link_validation"], "R23": ["vision_manifest"], "R25": ["vision_onnx_discovery"],
        "R26": ["vision_layout_reference"], "R27": ["layout_optimality_analysis"], "R28": ["shared_link_validation"],
        "R29": ["vision_source_pins"]
    }
    names = gates.get(req_id, [])
    if not names or not status:
        return ""
    return "; ".join(f"{name}={status[name]['status']}" for name in names if name in status)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", type=Path, default=HERE / "requirements_status.json")
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--markdown", type=Path)
    p.add_argument("--json-output", type=Path)
    args = p.parse_args()
    data = json.loads(args.spec.read_text(encoding="utf-8"))
    run_status = load_run_status(args.run_dir)
    rows = []
    for req in data["requirements"]:
        evidence = []
        for item in req["evidence"]:
            path = resolve_evidence(item)
            evidence.append({"declared": item, "path": str(path), "exists": path.exists()})
        row = {**req, "evidence": evidence, "all_evidence_exists": all(x["exists"] for x in evidence),
               "runtime_observation": runtime_note(req["id"], run_status)}
        rows.append(row)
    target_dir = args.run_dir or HERE
    markdown = args.markdown or target_dir / "REQUIREMENTS_AUDIT.md"
    json_output = args.json_output or target_dir / "requirements_audit.json"
    markdown.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps({"requirements": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    counts = Counter(row["code_status"] for row in rows)
    lines = ["# Requirements completion audit", "",
             f"Code status: complete={counts['complete']}, partial={counts['partial']}, missing={counts['missing']}.", "",
             "`code_status` answers whether the requested code exists. `validation_status` answers whether its claim has actually been tested.", "",
             "| ID | requirement | code | validation | evidence | runtime observation |", "|---|---|---|---|---|---|"]
    for row in rows:
        ev = ", ".join(f"{Path(x['path']).name}:{'yes' if x['exists'] else 'NO'}" for x in row["evidence"])
        lines.append(f"| {row['id']} | {row['requirement']} | {row['code_status']} | {row['validation_status']} | {ev} | {row['runtime_observation']} |")
    lines += ["", "## Unfinished requirements", ""]
    for row in rows:
        if row["code_status"] != "complete" or row["validation_status"] not in {"complete", "existing_results_and_rerun_pending", "smoke_checked_without_gpu"}:
            lines += [f"### {row['id']} — {row['code_status']} / {row['validation_status']}", "", row["gap"], ""]
    lines += ["## Completion rule", "",
              "The project is not complete while any row has `code_status=missing`, and no performance claim is complete while its validation is pending, blocked, source-dependent, or not run.", ""]
    markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"markdown": str(markdown), "json": str(json_output), "counts": counts}, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
