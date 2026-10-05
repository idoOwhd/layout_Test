#!/usr/bin/env python3
"""Add real serving's heterogeneous context lengths without changing geometry.

Lengths are controlled workloads, NOT a fabricated production trace. Original
uniform case IDs and source config/forward provenance are preserved.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from build_rq2_real_graph_growth_cases import digest

HERE=Path(__file__).resolve().parent


def build(parent):
    cases=copy.deepcopy(parent["cases"]);new=[];pairs=[]
    configs={}
    for c in cases:
        c["parent_graph_contract_sha256"]=c["graph_contract_sha256"]
        key=(c["model_id"],c["model_revision"])
        if key not in configs:
            path=HERE.parent/"real_world_shapes/source_cache/configs"/(key[0].replace("/","__")+"--"+key[1]+".json")
            raw=path.read_text()
            if hashlib.sha256(raw.encode()).hexdigest()!=c["config_sha256"]:
                raise ValueError("pinned model configuration changed")
            configs[key]=raw
        # Tiny config text, not model weights. Embed exact bytes so the new
        # validation suite no longer requires a sibling directory on a new host.
        c["pinned_config_json"]=configs[key]
        c["kv_lengths"]=[c["kv_length"]]*c["batch"]
        c["request_length_policy"]="uniform"
        c["graph_contract_sha256"]=digest({k:v for k,v in c.items() if k!="graph_contract_sha256"})
    for source in cases:
        if source["batch"]==1 or source["phase"] not in {"decode","extend"}:continue
        c=copy.deepcopy(source);kv=c["kv_length"];q=c["query_length"]
        pattern=[kv,max(q,kv//2+1),max(q,kv//4-1),max(q,kv//8+3)]
        c["kv_lengths"]=[pattern[i%4] for i in range(c["batch"])]
        c["request_length_policy"]="mixed_controlled_uneven_page_tails"
        c["case_id"]+="_mixed"
        c["length_provenance"]="controlled heterogeneous request metadata, supported by real serving forward; not measured traffic"
        for node in c["nodes"]:
            if node["operator_kind"]=="causal_gqa_with_kv_update":
                node["attrs"]["kv_lengths_per_request"]=c["kv_lengths"]
                node["attrs"]["physical_padded_kv_extent"]=kv
                node["semantic_signature"]=digest({k:v for k,v in node.items() if k!="semantic_signature"})
        c["graph_contract_sha256"]=digest({k:v for k,v in c.items() if k!="graph_contract_sha256"})
        new.append(c)
    by_group={}
    for c in new:
        key=(c["model_id"],c["phase"],c["batch"],c["query_length"],c["kv_length"])
        by_group.setdefault(key,[]).append(c)
    for group in by_group.values():
        group.sort(key=lambda c:c["stage"])
        for small,large in zip(group,group[1:]):
            old={n["id"]:n["semantic_signature"] for n in small["nodes"]}
            extended={n["id"]:n["semantic_signature"] for n in large["nodes"]}
            if not all(extended.get(k)==v for k,v in old.items()):raise ValueError("mixed request expansion changed old math")
            pairs.append({"small_case_id":small["case_id"],"expanded_case_id":large["case_id"],
                          "old_node_ids":list(old),"added_node_ids":[k for k in extended if k not in old],
                          "same_old_math_dtype_shape":True,"er_set_measured":False})
    result=copy.deepcopy(parent);result["cases"]=cases+new
    result["nested_pairs"]+=pairs;result["case_count"]=len(result["cases"])
    result["nested_pair_count"]=len(result["nested_pairs"])
    result["schema"]="RQ2-real-forward-growth-with-mixed-request-design-v2"
    result["uniform_case_count"]=len(cases);result["mixed_case_count"]=len(new)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--parent",type=Path,default=HERE/"ref_talks/rq2_qwen3_real_forward_growth_design_20261001.json")
    p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError("refusing to overwrite manifest")
    value=build(json.loads(args.parent.read_text()))
    value["parent_sha256"]=hashlib.sha256(args.parent.read_bytes()).hexdigest()
    with args.output.open("x") as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write("\n")
    print(json.dumps({k:value[k] for k in ("case_count","nested_pair_count","uniform_case_count","mixed_case_count")}))
