#!/usr/bin/env python3
"""Retry exact failed RQ2 growth GROUPS, not cherry-picked stages/candidates.

Growth stage i determines the near-set frozen at i+1, so an affected group must
be rerun from its earliest stage through its last. This tool never launches GPU.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from check_rq2_graph_growth_progress import read_rows

FRAMES=("pytorch","triton","cutlass","tvm","vllm","sglang")


def key(c):
    return (c["model_id"],c["phase"],c["batch"],c["query_length"],c["kv_length"],
            tuple(c.get("kv_lengths",[c["kv_length"]]*c["batch"])))


def graph_valid(r):
    if r.get("executed_graph_status")!="correctness_verified_complete_controlled_graph":return False
    rows=r.get("candidates",[])
    if not rows or len(rows)!=r["attempted_candidate_count"] or len(rows)!=r["measured_candidate_count"]:return False
    for candidate in rows:
        samples=candidate.get("train_samples_ms",[])
        if not candidate.get("correctness",{}).get("correct") or not samples:return False
        if any(not isinstance(v,(float,int)) or not math.isfinite(v) or v<=0 for v in samples):return False
        if not math.isclose(statistics.median(samples),candidate["train_median_ms"],rel_tol=1e-12,abs_tol=1e-12):return False
    return True


def pair_valid(row):
    raw=row.get("holdout",{});a=raw.get("frozen_ms",[]);b=raw.get("free_ms",[])
    if not a or len(a)!=len(b) or any(not isinstance(v,(float,int)) or not math.isfinite(v) or v<=0 for v in a+b):return False
    if not all(raw.get(k,{}).get("correct") is True for k in ("free_post_correctness","frozen_post_correctness")):return False
    return math.isclose(statistics.median(a)/statistics.median(b),row["speedup_frozen_over_free"],rel_tol=1e-12,abs_tol=1e-12)


def expected_pairs(cases):
    groups=defaultdict(list)
    for c in cases:groups[key(c)].append(c)
    return {(a["case_id"],b["case_id"]) for values in groups.values() for a,b in zip(values,values[1:])}


def plan(root):
    manifest=json.loads((root/"design_manifest.json").read_text());cases=manifest["cases"]
    byid={c["case_id"]:c for c in cases}
    if not byid or len(byid)!=len(cases):raise ValueError("duplicate/empty manifest")
    approval=root/"SMOKE_APPROVAL.json"
    frames=json.loads(approval.read_text())["approved_frameworks"] if approval.exists() else list(FRAMES)
    statuses=[s.split("\t") for s in (root/"process_status.tsv").read_text().splitlines()]
    if len(statuses)!=len(frames) or {s[0] for s in statuses}!=set(frames):
        raise RuntimeError("wait for every declared framework to exit before scheduling retries")
    bad=defaultdict(set)
    for fw in frames:
        rows,partial=read_rows(root/(fw+".jsonl"));graphs={};completion=[];pairs=set()
        for row in rows:
            kind=row["record_type"]
            if kind=="graph":
                cid=row["case"]["case_id"]
                if cid not in byid or cid in graphs:raise ValueError("undeclared/duplicate graph")
                graphs[cid]=row
                if not graph_valid(row):bad[fw,key(byid[cid])].add("graph_correctness_or_raw_timing_rejected")
            elif kind in {"incorrect_candidate","not_completed_case"}:
                cid=row["case_id"]
                if cid not in byid:raise ValueError("unresolved failed case")
                bad[fw,key(byid[cid])].add(kind)
            elif kind=="group_failure":
                g=row["group"];g=tuple(g[:-1])+ (tuple(g[-1]),)
                if g not in {key(c) for c in cases}:raise ValueError("unresolved failed group")
                bad[fw,g].add("group_failure")
            elif kind=="growth_pair":
                pair=(row["small_case_id"],row["expanded_case_id"])
                if pair not in expected_pairs(cases) or pair in pairs:raise ValueError("duplicate/undeclared growth pair")
                pairs.add(pair)
                if not pair_valid(row):bad[fw,key(byid[pair[1]])].add("growth_pair_timing_or_post_correctness_rejected")
            elif kind=="completion":completion.append(row)
        for cid in set(byid)-set(graphs):bad[fw,key(byid[cid])].add("graph_missing")
        for a,b in expected_pairs(cases)-pairs:bad[fw,key(byid[b])].add("growth_pair_missing")
        if len(completion)!=1 or partial:
            raise RuntimeError("incomplete/corrupt worker tail: audit launcher or fix data writing before measured-case retry")
        exitcode=next(s[1] for s in statuses if s[0]==fw)
        if exitcode!="0" and not any(f==fw for f,g in bad):
            raise RuntimeError("nonzero exit without failed graph: inspect infrastructure, do not blindly rerun valid timings")
    groups={fw:sorted({g for f,g in bad if f==fw}) for fw in frames}
    groups={fw:gs for fw,gs in groups.items() if gs}
    selected={fw:[c["case_id"] for c in cases if key(c) in set(gs)] for fw,gs in groups.items()}
    return {"prior_run":str(root.resolve()),"prior_manifest_sha256":hashlib.sha256((root/"design_manifest.json").read_bytes()).hexdigest(),
            "group_keys_by_framework":groups,"case_ids_by_framework":selected,
            "reasons":[{"framework":f,"group":g,"reasons":sorted(why)} for (f,g),why in sorted(bad.items())],
            "policy":"rerun complete affected model/workload growth sequences and every candidate arm; keep unrelated successful groups untouched",
            "all_RQ2_or_source_PRs_complete":False,"old_results_modified":False}


def require_complete(root):
    manifest=json.loads((root/"design_manifest.json").read_text());ids={c["case_id"] for c in manifest["cases"]}
    statuses=[s.split("\t") for s in (root/"process_status.tsv").read_text().splitlines()]
    if len(statuses)!=1 or statuses[0][1]!="0":raise RuntimeError("retry worker failed")
    rows,partial=read_rows(root/(statuses[0][0]+".jsonl"))
    graphs=[r for r in rows if r["record_type"]=="graph"]
    completions=[r for r in rows if r["record_type"]=="completion"]
    if partial or len(completions)!=1 or len(graphs)!=len(ids) or {r["case"]["case_id"] for r in graphs}!=ids or not all(graph_valid(r) for r in graphs):
        raise RuntimeError("retry failed/incomplete selected growth sequence")
    counts=completions[0]["counts"]
    if counts["graphs"]!=len(ids) or counts["incorrect_candidates"] or counts["failures"]:raise RuntimeError("retry correctness rejected")
    pairs=[r for r in rows if r["record_type"]=="growth_pair"]
    expected=expected_pairs(manifest["cases"])
    if len(pairs)!=len(expected) or {(r["small_case_id"],r["expanded_case_id"]) for r in pairs}!=expected or not all(pair_valid(r) for r in pairs):
        raise RuntimeError("retry growth pairs missing or incorrect; do not advance to full")
    return {"declared_subset_correct":True,"selected_graphs":len(ids),"all_1020_graphs_or_all_PRs_complete":False}


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path);p.add_argument("--check-selected-complete",action="store_true");a=p.parse_args()
    if a.check_selected_complete:print(json.dumps(require_complete(a.run)));return
    if a.output is None:raise ValueError("--output required for retry plan")
    value=plan(a.run);a.output.mkdir(parents=True,exist_ok=False)
    with (a.output/"RQ2_FAILED_GROUP_PLAN.json").open("x") as f:json.dump(value,f,indent=2)
    original=json.loads((a.run/"design_manifest.json").read_text())
    for fw,ids in value["case_ids_by_framework"].items():
        subset={**original,"cases":[c for c in original["cases"] if c["case_id"] in set(ids)]}
        subset["case_count"]=len(subset["cases"]);subset["retry_subset_of"]=str(a.run.resolve())
        with (a.output/f"{fw}_manifest.json").open("x") as f:json.dump(subset,f,indent=2)
    print(json.dumps({"selected_cases":{k:len(v) for k,v in value["case_ids_by_framework"].items()},"output":str(a.output)}))


if __name__=="__main__":main()
