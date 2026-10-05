#!/usr/bin/env python3
"""Read progress without inventing success or touching existing results."""
import argparse
from collections import Counter
import json
from pathlib import Path


def read_rows(path):
    if not path.exists():return [],False
    text=path.read_text();lines=text.splitlines();rows=[];partial=False
    for index,line in enumerate(lines):
        if not line:continue
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A live writer can have published only part of its last record.
            # Corruption in any completed/middle record must still raise.
            if index==len(lines)-1 and not text.endswith("\n"):partial=True
            else:raise
    return rows,partial


def progress(root,snapshot=None):
    approved=root/"SMOKE_APPROVAL.json"
    frameworks=json.loads(approved.read_text())["approved_frameworks"] if approved.exists() else \
        ["pytorch","triton","cutlass","tvm","vllm","sglang"]
    manifest=json.loads((root/"design_manifest.json").read_text())
    full_ids={c["case_id"] for c in manifest["cases"]}
    statuses={}
    status_file=root/"process_status.tsv"
    if status_file.exists():statuses=dict(line.split("\t") for line in status_file.read_text().splitlines())
    records=[]
    for framework in frameworks:
        rows,partial=snapshot[framework] if snapshot is not None else read_rows(root/(framework+".jsonl"))
        counts=Counter(r["record_type"] for r in rows)
        graphs=[r for r in rows if r["record_type"]=="graph"]
        ids={r["case"]["case_id"] for r in graphs}
        verified_ids={r["case"]["case_id"] for r in graphs if
                      r["executed_graph_status"]=="correctness_verified_complete_controlled_graph"}
        completion=next((r for r in rows if r["record_type"]=="completion"),None)
        meta=next((r for r in rows if r["record_type"]=="run_metadata"),{})
        full_mode=meta.get("args",{}).get("mode")=="full"
        exact_full_ids=full_mode and verified_ids==full_ids and len(graphs)==len(full_ids)
        records.append({"framework":framework,"mode":meta.get("args",{}).get("mode"),
            "declared_full_graphs":len(full_ids),"graph_records":len(graphs),
            "unique_graphs":len(ids),"correctness_verified_unique_graphs":len(verified_ids),
            "growth_pairs":counts["growth_pair"],
            "correct_candidate_count":sum(r["measured_candidate_count"] for r in graphs),
            "incorrect_candidate_records":counts["incorrect_candidate"],
            "group_failure_records":counts["group_failure"],
            "not_completed_case_records":counts["not_completed_case"],
            "partial_last_record":partial,"last_completed_case":graphs[-1]["case"]["case_id"] if graphs else None,
            "process_exit":statuses.get(framework),"completion_record_present":completion is not None,
            "all_declared_full_H2_1_cases_correct":exact_full_ids and completion is not None and
                statuses.get(framework)=="0" and counts["incorrect_candidate"]==counts["group_failure"]==0,
            "state":"finished" if completion is not None else "not_started" if not rows else "no_completion_yet"})
    return {"run":str(root.resolve()),"frameworks":records,
            "all_declared_H2_1_full_matrix_passed":all(r["all_declared_full_H2_1_cases_correct"] for r in records),
            "all_RQ2_requirements_complete":False,
            "remaining":"H2.2 real partitions/K_epsilon, H2.3 causal/compiler propagation, 207 native source replays, default whole-engine policy are separate unmet requirements",
            "note":"A missing completion record is not proof the process is alive or dead; inspect logs/process separately."}


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True)
    p.add_argument("--save",type=Path);args=p.parse_args();value=progress(args.run)
    if args.save:
        with args.save.open("x") as f:json.dump(value,f,ensure_ascii=False,indent=2)
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
