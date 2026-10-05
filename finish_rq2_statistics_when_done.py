#!/usr/bin/env python3
"""Finish a RUN ALREADY STARTED, without modifying its GPU code or raw data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError("refusing to overwrite post-run analysis")
    here=Path(__file__).resolve().parent
    snapshot=args.output.with_name(args.output.name+"_source_snapshot")
    snapshot.mkdir(parents=True)
    hashes={}
    for name in ("audit_rq2_holdout_statistics.py","check_rq2_graph_growth_progress.py"):
        target=snapshot/name;shutil.copy2(here/name,target)
        hashes[name]=hashlib.sha256(target.read_bytes()).hexdigest()
    with (snapshot/"source_sha256.json").open("x") as f:json.dump(hashes,f,indent=2)
    runner_pid=int((args.run/"runner.pid").read_text().strip())
    print(f"Waiting for existing run={args.run} runner_pid={runner_pid}; no GPU work launched",flush=True)
    while not (args.run/"RQ2_GRAPH_GROWTH_RESULTS_CN.md").exists():
        try:os.kill(runner_pid,0)
        except ProcessLookupError:
            raise RuntimeError("runner exited before its analysis was produced; inspect saved logs, not a success")
        time.sleep(30)
    subprocess.run([sys.executable,str(snapshot/"audit_rq2_holdout_statistics.py"),
                    "--run",str(args.run),"--output",str(args.output)],check=True)
    subprocess.run([sys.executable,str(snapshot/"check_rq2_graph_growth_progress.py"),
                    "--run",str(args.run),"--save",str(args.output/"FINAL_CASE_COVERAGE.json")],check=True)
    print(f"Post-run analysis complete={args.output}; case coverage still decides pass/fail",flush=True)


if __name__=="__main__":main()
