#!/usr/bin/env python3
"""Require a passing smoke of the same GPU worker/manifest before a full run."""
import argparse
import hashlib
import json
from pathlib import Path

GPU_SOURCES=("rq2_growth_bench.py","rq2_growth_search.py","rq2_growth_kernels.py",
             "rq2_growth_cutlass.cu","rq1_diverse_kernels.py","rq1_flashinfer_workspace.py")


def verify(smoke, run, frameworks):
    records=[]
    for name in (*GPU_SOURCES,"design_manifest.json"):
        relative=Path(name) if name=="design_manifest.json" else Path("source_snapshot")/name
        old=hashlib.sha256((smoke/relative).read_bytes()).hexdigest()
        new=hashlib.sha256((run/relative).read_bytes()).hexdigest()
        if old!=new:raise ValueError(f"not smoke-approved: changed {relative}")
        records.append({"file":str(relative),"sha256":new})
    statuses=dict(line.split("\t") for line in (smoke/"process_status.tsv").read_text().splitlines())
    for framework in frameworks:
        if statuses.get(framework)!="0":raise ValueError(f"smoke did not pass: {framework}")
        rows=[json.loads(line) for line in (smoke/(framework+".jsonl")).read_text().splitlines() if line]
        completion=[r for r in rows if r["record_type"]=="completion"]
        if len(completion)!=1:raise ValueError(f"missing/duplicate smoke completion: {framework}")
        c=completion[0];counts=c["counts"]
        graphs=[r for r in rows if r["record_type"]=="graph"]
        if c["expected_graphs"]<=0 or len(graphs)!=c["expected_graphs"] or \
                counts["graphs"]!=len(graphs) or counts["failures"] or counts["incorrect_candidates"]:
            raise ValueError(f"incomplete or incorrect smoke: {framework}")
        if len({r["case"]["case_id"] for r in graphs})!=len(graphs):
            raise ValueError(f"duplicate smoke case: {framework}")
        if any(r["executed_graph_status"]!="correctness_verified_complete_controlled_graph" for r in graphs):
            raise ValueError(f"rejected smoke candidate: {framework}")
    return {"smoke":str(smoke.resolve()),"approved_frameworks":frameworks,
            "matching_gpu_sources_and_manifest":records,
            "claim":"same code smoke-approved; not proof that every full shape will pass"}


def main():
    p=argparse.ArgumentParser();p.add_argument("--smoke",required=True,type=Path)
    p.add_argument("--run",required=True,type=Path);p.add_argument("--frameworks",required=True)
    args=p.parse_args();value=verify(args.smoke,args.run,args.frameworks.split(","))
    with (args.run/"SMOKE_APPROVAL.json").open("x") as f:json.dump(value,f,indent=2)
    print(json.dumps(value,indent=2))


if __name__=="__main__":main()
