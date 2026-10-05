#!/usr/bin/env python3
"""Independently verify exported data/formulas, pinned source identities and figures.

Read-only except for a NEW validation record in the NEW report directory.
Uses no GPU and no third-party packages. Never overwrites a validation record.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


def main():
    p=argparse.ArgumentParser();p.add_argument("--report",required=True,type=Path);root=p.parse_args().report
    audit=json.loads((root/"AUDIT.json").read_text())
    provenance=json.loads((root/"PROVENANCE.json").read_text());sha={x["path"]:x["sha256"] for x in provenance}
    checks={}
    for pathkey,metadata_key,filename in [("growth_run","metadata","rq2_growth_bench.py"),
                                          ("conditional_run","conditional_metadata","rq2_domain_anchor_bench.py")]:
        run=Path(audit["rq2"][pathkey]);source=str(run/"source_snapshot"/filename)
        manifest=str(Path(audit["rq2"]["growth_run"])/"design_manifest.json")
        for fw,meta in audit["rq2"][metadata_key].items():
            assert meta["executor_code_sha256"]==sha[source]
            assert meta["manifest_sha256"]==sha[manifest]
            if metadata_key=="conditional_metadata":
                assert meta["baseline_results_sha256"]==sha[str(Path(audit["rq2"]["growth_run"])/(fw+".jsonl"))]
    checks["RQ2_recorded_executor_manifest_and_baseline_SHAs_match"]=True
    with (root/"data/RQ1_ALL_OBJECTS.csv").open() as f:r1=list(csv.DictReader(f))
    assert len(r1)==854 and len({(r["case_id"],r["consumer"]) for r in r1})==854
    assert dict(Counter(r["status"] for r in r1))==audit["rq1"]["classifications"]
    for r in r1:
        if not r["speedup_median_of_process_ratios"]:continue
        a=json.loads(r["local_edge_ms"]);b=json.loads(r["joint_edge_ms"])
        assert len(a)==len(b)==3
        assert math.isclose(statistics.median(x/y for x,y in zip(a,b)),float(r["speedup_median_of_process_ratios"]),rel_tol=1e-10)
        assert math.isclose(statistics.mean((x-y)/x for x,y in zip(a,b)),float(r["mean_latency_reduction"]),abs_tol=1e-12)
    checks["RQ1_export_unique_classifications_and_all_heldout_formulas_match"]=True
    with (root/"data/RQ2_ALL_COMPARISONS.csv").open() as f:r2=list(csv.DictReader(f))
    assert len(r2)==15624 and len({(r["protocol"],r["framework"],r["case_id"]) for r in r2})==15624
    positives=Counter();same_choice_positives=[]
    for r in r2:
        a=json.loads(r["restricted_ms"]);b=json.loads(r["free_ms"]);ci=json.loads(r["ci"])
        assert len(a)==len(b)==5 and all(x>0 and math.isfinite(x) for x in a+b)
        assert math.isclose(statistics.median(a)/statistics.median(b),float(r["ratio"]),rel_tol=1e-10)
        positive=r["same_choice"]=="False" and ci[0]>1.03
        assert (r["positive"]=="True")==positive
        positives[r["protocol"]]+=positive
        if r["same_choice"]=="True" and ci[0]>1.03:same_choice_positives.append(r["case_id"])
    assert positives=={"GROWTH-NEARSET":20,"QKV-DUAL":1,"REMOTE-KV-ANCHOR":1}
    checks["RQ2_all_exported_ratios_and_positive_gates_match"]=True
    checks["same_choice_CI_lower_over_1_03_not_layout_positives"]=same_choice_positives
    with (root/"data/RQ2_GRAPH_CHOICES.csv").open() as f:graphs=list(csv.DictReader(f))
    assert len(graphs)==6120
    for g in graphs:
        axes=json.loads(g["axes"]);bits=json.loads(g["best_train_bits"])
        assert len(axes)==len(bits) and all(x in (0,1) for x in bits)
        if g["search_scope"]=="exact_declared_space":assert int(g["candidate_count"])==2**len(axes)
    checks["all_6120_tensor_layout_graphs_have_valid_bits_and_exact_space_counts"]=True
    figures=json.loads((root/"FIGURE_MANIFEST.json").read_text());assert len(figures)==8
    for fig in figures:
        stem=root/"figures"/fig["figure"]
        assert stem.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
        ET.parse(stem.with_suffix(".svg"))
        assert stem.with_suffix(".png").read_bytes().startswith(b"\x89PNG")
        assert all((root/"data"/d).exists() for d in fig["data_tables"])
    checks["8_PDF_SVG_PNG_figures_and_source_tables_exist_and_parse"]=True
    # Re-hash the script/snapshots (small files); raw hashes were computed during streaming.
    for item in provenance:
        if item["path"].endswith((".py",".sh",".cu")):
            assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()==item["sha256"]
    checks["small_source_files_unchanged_since_report_generation"]=True
    with (root/"VALIDATION.json").open("x") as f:json.dump(checks,f,indent=2,ensure_ascii=False)
    print(json.dumps({"passed":True,"report":str(root),"figures":8,"RQ1_objects":len(r1),"RQ2_comparisons":len(r2),"same_choice_noise_examples":len(same_choice_positives)},ensure_ascii=False))


if __name__=="__main__":main()
