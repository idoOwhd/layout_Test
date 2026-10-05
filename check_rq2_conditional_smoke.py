#!/usr/bin/env python3
"""Approve EXACT conditional-worker code, not a claim all RQ2 is solved."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(smoke,run):
    status=dict(line.split("\t") for line in (smoke/"process_status.tsv").read_text().splitlines())
    required=("pytorch","triton","cutlass","tvm","vllm","sglang")
    for framework in required:
        if status.get(framework)!="0":raise ValueError(f"conditional smoke failed/missing: {framework}")
        rows=[json.loads(l) for l in (smoke/(framework+".jsonl")).read_text().splitlines() if l]
        completion=next((r for r in rows if r["record_type"]=="completion"),None)
        if completion is None or completion["expected_graphs"]<=0:raise ValueError("no smoke completion")
        counts=completion["counts"]
        if counts["failures"] or counts["domain_graphs"]!=completion["expected_graphs"] or \
                counts["anchor_graphs"]+counts["anchor_semantically_inapplicable"]!=counts["domain_graphs"]:
            raise ValueError("incomplete conditional smoke")
        cases=[r["case"]["case_id"] for r in rows if r["record_type"]=="conditional_domain_result"]
        if len(set(cases))!=len(cases) or len(cases)!=completion["expected_graphs"]:
            raise ValueError("duplicate/missing conditional case")
    matched=[]
    for name in ("rq2_domain_anchor_bench.py","rq2_growth_bench.py","rq2_growth_search.py",
                 "rq2_growth_kernels.py","rq2_growth_cutlass.cu","rq1_diverse_kernels.py",
                 "rq1_flashinfer_workspace.py","audit_rq2_holdout_statistics.py"):
        old=hashlib.sha256((smoke/"source_snapshot"/name).read_bytes()).hexdigest()
        new=hashlib.sha256((run/"source_snapshot"/name).read_bytes()).hexdigest()
        if old!=new:raise ValueError(f"conditional worker changed after smoke: {name}")
        matched.append({"file":name,"sha256":new})
    if (smoke/"design_manifest.json").read_bytes()!=(run/"design_manifest.json").read_bytes():
        raise ValueError("conditional manifest changed after smoke")
    return {"approved_smoke":str(smoke.resolve()),"frameworks":required,"matching_code":matched,
            "scope":"conditional runtime tests only, not full RQ2 or native source replay"}


def main():
    p=argparse.ArgumentParser();p.add_argument("--smoke",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True);args=p.parse_args();value=verify(args.smoke,args.run)
    with (args.run/"CONDITIONAL_SMOKE_APPROVAL.json").open("x") as f:json.dump(value,f,indent=2)
    print(json.dumps(value,indent=2))


if __name__=="__main__":main()
