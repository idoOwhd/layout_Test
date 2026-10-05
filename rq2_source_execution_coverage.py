#!/usr/bin/env python3
"""Preserve the entire deduplicated source-replay ledger, fail closed."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    value=json.loads(args.audit.read_text())
    records=[]
    for row in value["records"]:
        keys=("repo","kind","number","url","base_sha","head_sha","contract_audit_status",
              "contract_audit_note","rq2_whole_graph_native_replay_status")
        records.append({k:row.get(k) for k in keys})
    if len({r["url"] for r in records})!=207:raise ValueError("source catalog is incomplete or duplicated")
    with args.output.open("x") as f:
        json.dump({"unique_sources":len(records),"source_replay_complete":False,"records":records,
            "note":"Controlled Qwen3 graph measurements do not change not_run source patch replay statuses."},
            f,ensure_ascii=False,indent=2)


if __name__=="__main__":main()
