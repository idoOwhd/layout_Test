#!/usr/bin/env python3
"""Wait for an existing run, then require its full controlled matrix passed."""
import argparse
import json
import os
from pathlib import Path
import time
from check_rq2_graph_growth_progress import progress


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True,type=Path)
    p.add_argument("--require-full",action="store_true");args=p.parse_args()
    pid=int((args.run/"runner.pid").read_text().strip())
    print(f"Waiting for existing growth run={args.run}; no new GPU work",flush=True)
    while not (args.run/"RQ2_GRAPH_GROWTH_RESULTS_CN.md").exists():
        try:os.kill(pid,0)
        except ProcessLookupError:raise RuntimeError("growth runner exited without a final report")
        time.sleep(30)
    value=progress(args.run)
    print(json.dumps(value,indent=2,ensure_ascii=False),flush=True)
    if args.require_full and not value["all_declared_H2_1_full_matrix_passed"]:
        raise RuntimeError("growth full matrix failed/incomplete; do not certify or silently reuse missing cases")


if __name__=="__main__":main()
