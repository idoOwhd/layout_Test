#!/usr/bin/env python3
"""Build operator-edge contracts without clipping pinned architecture widths."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILIES = {
    "attention_bmm_oproj": "gqa",
    "projection_sdpa": "gqa",
    "sparse_gather_sdpa": "sparse_attention",
    "projection_swiglu_down": "swiglu",
    "moe_dispatch_expert": "moe",
    "moe_gemm_combine": "moe",
    "state_prefill_decode": "linear_attention",
    "projection_gdn": "linear_attention",
    "gemm_bias_gemm": "swiglu",
    "reduce_norm_gemm": "swiglu",
    "host_weight_gemm": "swiglu",
    "host_kv_sdpa": "gqa",
    "kv_writer_flashinfer": "gqa",
    "kv_writer_swa_flashinfer": "sliding_attention",
}


def build(source: Path, smoke: bool):
    data = json.loads(source.read_text())
    parents = data["cases"] if isinstance(data, dict) else data
    cases = []
    contract_to_case = {}
    for family, structure in FAMILIES.items():
        pool = [p for p in parents if p["structure"] == structure and p["phase"] == "prefill"]
        # Prefer modern model names, then minimize complete projection/expert
        # weight size. Architecture dimensions are always kept exact.
        def weight(p):
            d = p["dimensions"]
            h = int(d["hidden_size"])
            width = int(d.get("moe_intermediate_size") or d.get("intermediate_size") or h)
            return h*width*(int(d.get("num_experts") or 1) if "moe" in family else 1)
        pool.sort(key=lambda p: (not any(x in p["model_id"].lower()
                                      for x in ("qwen3", "kimi", "deepseek-v4", "granite-4")),weight(p)))
        unique = []
        dimensions_seen = set()
        for p in pool:
            key = json.dumps(p["dimensions"],sort_keys=True)
            if key not in dimensions_seen:
                dimensions_seen.add(key)
                unique.append(p)
        if not unique:
            raise ValueError(f"no pinned model config for {family}")
        chosen = unique[:1 if smoke else 4]
        profiles = [(1,16,128)] if smoke else [
            (1,1,128),(2,1,512),(8,1,4096),
            (1,16,128),(2,32,512),(1,127,511),(1,128,2048),(2,256,2048),
            (4,1,8193),(1,32,8192),(4,33,1025),(1,257,4097)]
        if family in {"moe_dispatch_expert","moe_gemm_combine","host_weight_gemm"}:
            profiles = [(1,4,128)] if smoke else [(1,1,128),(2,1,512),(8,1,4096),
                (1,4,128),(2,8,512),(1,16,512),(1,31,2048),(2,32,2048),
                (4,1,8193),(1,32,8192),(4,17,1025),(1,63,4097)]
        expanded_profiles=[(*profile,page) for profile in profiles
                           for page in ([16] if smoke or "kv_writer" not in family else [16,64])]
        for p in chosen:
            for b,q,kv,page in expanded_profiles:
                record = {k:p[k] for k in ("model_id","model_revision","source_url","dimensions")}
                record.update(family=family,parent_case_id=p["case_id"],
                    request_batch=b,query_length=q,kv_length=kv,
                    phase="decode" if q==1 else "prefill",
                    shape_origin="pinned architecture dimensions; controlled B/q/kv workload",
                    dtype="float16",page_size=page,decode_horizon=4 if smoke else (1 if q==1 else 16),
                    algorithm_scope="exact stated local operator edge; not full-model or upstream PR replay")
                # Provenance does not create an independent experimental
                # tensor contract. Keep all sources on one executable case.
                contract={k:record[k] for k in ("family","dtype")}
                flat_token_families={"projection_swiglu_down","gemm_bias_gemm","reduce_norm_gemm","host_weight_gemm","moe_dispatch_expert","moe_gemm_combine"}
                if family in flat_token_families:contract["tokens"]=b*q
                else:contract.update(request_batch=b,query_length=q)
                fields=set() if family in {"sparse_gather_sdpa","host_kv_sdpa","kv_writer_flashinfer","kv_writer_swa_flashinfer","state_prefill_decode"} else {"hidden_size"}
                if family in {"attention_bmm_oproj","projection_sdpa","sparse_gather_sdpa","host_kv_sdpa","kv_writer_flashinfer","kv_writer_swa_flashinfer"}:
                    fields.update(("num_attention_heads","num_key_value_heads","head_dim","index_n_heads","index_heads","index_head_dim"))
                if "moe" in family:fields.update(("num_experts","experts_per_token","moe_intermediate_size","intermediate_size"))
                elif family in {"projection_swiglu_down","gemm_bias_gemm","reduce_norm_gemm","host_weight_gemm"}:
                    fields.update(("intermediate_size","moe_intermediate_size"))
                if family in {"projection_gdn","state_prefill_decode"}:
                    fields.update(("linear_num_key_heads","linear_num_value_heads","linear_key_head_dim","linear_value_head_dim"))
                contract["dimensions"]={k:record["dimensions"].get(k) for k in sorted(fields)}
                if family in {"attention_bmm_oproj","sparse_gather_sdpa","host_kv_sdpa","kv_writer_flashinfer","kv_writer_swa_flashinfer"}:
                    contract["kv_length"]=kv
                if family=="state_prefill_decode":contract["decode_horizon"]=record["decode_horizon"]
                if family=="attention_bmm_oproj":fields.discard("num_key_value_heads")
                if family=="sparse_gather_sdpa":fields.discard("num_key_value_heads")
                contract["dimensions"]={k:record["dimensions"].get(k) for k in sorted(fields)}
                if "kv_writer" in family:contract["page_size"]=page
                if family=="kv_writer_swa_flashinfer":
                    contract["sliding_window"]=record["dimensions"].get("sliding_window")
                provenance={k:record[k] for k in ("model_id","model_revision","source_url","parent_case_id")}
                workload={"request_batch":b,"query_length":q,"kv_length":kv,"phase":record["phase"]}
                digest=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
                if digest in contract_to_case:
                    target=contract_to_case[digest]
                    if provenance not in target["provenance"]:target["provenance"].append(provenance)
                    if workload not in target["equivalent_workload_aliases"]:target["equivalent_workload_aliases"].append(workload)
                    continue
                record.update(case_id=family+"-"+digest[:16],tensor_seed=int(digest[:8],16),
                              contract_sha256=digest,executable_contract=contract,provenance=[provenance],
                              equivalent_workload_aliases=[workload],
                              complete_shape_is_observed_production_trace=False)
                cases.append(record)
                contract_to_case[digest]=record
    return cases


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-manifest",type=Path,default=HERE.parent/"real_world_shapes/real_world_shape_manifest.json")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--extended-smoke",action="store_true",
                   help="Single/multi-request, decode/prefill and partial pages for every family")
    p.add_argument("--architecture-smoke",action="store_true",
                   help="Decode and prefill for every selected pinned architecture")
    p.add_argument("--validation-smoke",action="store_true",
                   help="Deduplicated union of extended and architecture smoke")
    args=p.parse_args()
    if args.validation_smoke:args.extended_smoke=True;args.architecture_smoke=True
    if args.output.exists():p.error(f"refusing to overwrite {args.output}")
    cases=build(args.source_manifest,args.smoke and not (args.extended_smoke or args.architecture_smoke))
    all_cases=cases
    if args.extended_smoke:
        selected=[]
        for family in FAMILIES:
            pool=[c for c in cases if c["family"]==family]
            first_model=pool[0]["model_id"];pool=[c for c in pool if c["model_id"]==first_model]
            def choose(pred):
                choices=[c for c in pool if pred(c)]
                if choices:
                    c=choices[0]
                    if c not in selected:selected.append(c)
            choose(lambda c:c["request_batch"]==1 and c["query_length"]==1)
            choose(lambda c:c["request_batch"]==2 and c["query_length"]==1)
            choose(lambda c:c["request_batch"]==2 and c["query_length"]>1)
            choose(lambda c:c["query_length"] in {31,127})
            if "kv_writer" in family:
                choose(lambda c:c["request_batch"]==2 and c["page_size"]==64)
        cases=selected
    if args.architecture_smoke:
        selected=list(cases) if args.extended_smoke else []
        for family in FAMILIES:
            pool=[c for c in all_cases if c["family"]==family]
            for model in sorted({c["model_id"] for c in pool}):
                group=[c for c in pool if c["model_id"]==model and c["request_batch"]==1 and c["page_size"]==16]
                for phase in ("decode","prefill"):
                    matches=[c for c in group if c["phase"]==phase]
                    if matches and matches[0] not in selected:selected.append(matches[0])
        cases=selected
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({"schema_version":1,"source_manifest":str(args.source_manifest.resolve()),
        "case_count":len(cases),"family_counts":dict(Counter(c["family"] for c in cases)),
        "cases":cases},ensure_ascii=False,indent=2)+"\n")
    print(f"contracts={len(cases)} families={len(FAMILIES)} output={args.output}")


if __name__=="__main__":main()
