#!/usr/bin/env python3
"""Capture exact tested implementation/environment without secrets."""
import argparse
import hashlib
import importlib.metadata as M
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    for name in ("warmup","iterations","repetitions"):p.add_argument("--"+name,type=int,required=True)
    args=p.parse_args()
    if args.repetitions<1 or args.warmup<1 or args.iterations<3:p.error("invalid timing protocol")
    import torch
    if not torch.cuda.is_available():raise RuntimeError("CUDA not available")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="1":raise RuntimeError("GPU1 must be isolated")
    versions={}
    for name in ("torch","triton","vllm","sglang","flashinfer-python"):
        try:versions[name]=M.version(name)
        except M.PackageNotFoundError:versions[name]=None
    sources={}
    for name in ("rq1_diverse_kernels.py","rq1_diverse_edge_bench.py","rq1_flashinfer_workspace.py","build_rq1_diversity_cases.py",
                 "analyze_rq1_diversity.py","build_rq1_source_registry.py","rq1_shared_github_urls.json",
                 "rq1_shared_source_contexts.json","run_rq1_diversity_gpu1.sh"):
        sources[name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    revision=subprocess.run(["git","rev-parse","HEAD"],cwd=HERE,text=True,capture_output=True).stdout.strip()
    out={"python":sys.executable,"python_version":platform.python_version(),"versions":versions,
        "git_head":revision,"source_sha256":sources,
        "manifest_sha256":hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "visible_devices":os.environ["CUDA_VISIBLE_DEVICES"],"device":torch.cuda.get_device_name(),
        "capability":torch.cuda.get_device_capability(),"total_device_bytes":torch.cuda.get_device_properties(0).total_memory,
        "warmup":args.warmup,"iterations":args.iterations,"independent_process_repetitions":args.repetitions,
        "workload_dtype":"float16","native_GDN_state_dtype":"float32",
        "reference_math_dtype":"float32","TF32_enabled":False,
        "native_pr_reproduction_is_complete":False}
    with args.output.open("x") as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write("\n")


if __name__=="__main__":main()
