#!/usr/bin/env python3
"""Additional CONDITIONAL runtime tests, not full compiler-domain proofs.

H2.2 subset: one real packed-QKV output has one/two physical representations.
H2.3 subset: a matched 2x2 intervention of upstream choice and downstream KV
contract, with actual semantic-DAG distance and fresh paired holdout timing.
All other layout decisions stay fixed at a correctness-qualified baseline.
"""
import argparse
from collections import deque
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback
import torch
from rq2_growth_bench import (Adapter,Group,GUARD_ELEMENTS,group_key,
                             independent_graph_reference,tensor_meta,timed)
from rq2_growth_search import layout_axes
from audit_rq2_holdout_statistics import paired_bootstrap


def operator_path(case,source,anchor):
    """Shortest directed semantic-operator path, not register/SSA distance."""
    owner={out:n["id"] for n in case["nodes"] for out in n["outputs"]}
    edges={n["id"]:set() for n in case["nodes"]}
    for node in case["nodes"]:
        for name in node["inputs"]:
            if name in owner:edges[owner[name]].add(node["id"])
    queue=deque([(source,[source])]);seen={source}
    while queue:
        node,path=queue.popleft()
        if node==anchor:return path
        for nxt in sorted(edges.get(node,())):
            if nxt not in seen:seen.add(nxt);queue.append((nxt,path+[nxt]))
    return None


def intervention_axes(case):
    axes=layout_axes(case);last=case["decoder_blocks"]-1
    anchor=f"L{last}.kv_cache"
    if anchor not in axes:return None,"graph has no KV anchor"
    upstream="L0.qkv_projection" if "L0.qkv_projection" in axes else "L0.kv_cache"
    if upstream not in axes or upstream==anchor:return None,"fewer than two distinct physical choices in this declared controlled space"
    source="L0.qkv_projection" if upstream.endswith("qkv_projection") else "L0.attention"
    target=f"L{last}.attention";path=operator_path(case,source,target)
    if not path or len(path)<2:return None,"no directed upstream-to-anchor path"
    return {"upstream_axis":upstream,"anchor_axis":anchor,"source_node":source,
            "anchor_node":target,"semantic_operator_path":path,"semantic_operator_hops":len(path)-1},None


def factorial_assignments(bits,axes,upstream,anchor):
    if upstream==anchor:raise ValueError("two distinct intervention factors required")
    ui=axes.index(upstream);ai=axes.index(anchor);result=[]
    for a in (0,1):
        for u in (0,1):
            value=list(bits);value[ui]=u;value[ai]=a
            result.append({"bits":value,"upstream":u,"anchor":a,"source_column":None})
    return result


def representation_assignments(case,bits):
    axes=layout_axes(case)
    if "L0.qkv_projection" not in axes:
        return [{"bits":list(bits),"source_column":0,"destination_column":0,
                 "explicit_output_representation_count":1,"one_row_aliases_deduplicated":True}]
    index=axes.index("L0.qkv_projection");result=[]
    for src in (0,1):
        for dst in (0,1):
            value=list(bits);value[index]=dst
            result.append({"bits":value,"source_column":src,"destination_column":dst,
                           "explicit_output_representation_count":1+int(src!=dst)})
    return result


class RepresentationAdapter:
    """Reuse genuine primitives; add an ACTUAL guarded copy only when chosen."""
    def __init__(self,base):
        self.base=base;self.target_weight_ptr=None;self.raw=None;self.temp=None
        self.check_copy=False;self.copy_checks=[];self.domain_metadata={}
    def __getattr__(self,key):return getattr(self.base,key)
    def bind(self,group):self.target_weight_ptr=group.layers[0]["wqkv"].data_ptr()
    def clear(self):self.raw=None;self.temp=None;self.copy_checks=[];self.domain_metadata={}
    def prepare(self,plan,source_column):
        self.clear();out=plan["layers"][0]["qkv_projection"]
        dst=int(out.shape[0]>1 and out.stride(0)==1)
        src=dst if source_column is None else source_column
        if src!=dst:
            rows,cols=out.shape
            self.raw=torch.full((out.numel()+2*GUARD_ELEMENTS,),777,device="cuda",dtype=torch.float16)
            region=self.raw[GUARD_ELEMENTS:-GUARD_ELEMENTS]
            self.temp=region.view(cols,rows).t() if src else region.view(rows,cols)
        source=out if self.temp is None else self.temp
        self.domain_metadata={"producer_output":tensor_meta(source),"consumer_shared_input":tensor_meta(out),
            "different_storage_instances":source.untyped_storage().data_ptr()!=out.untyped_storage().data_ptr(),
            "explicit_output_representation_count":1+int(self.temp is not None),
            "copy_logical_payload_bytes":out.numel()*out.element_size() if self.temp is not None else 0,
            "extra_allocated_bytes_including_guards":self.raw.numel()*self.raw.element_size() if self.raw is not None else 0,
            "count_scope":"only first-layer packed-QKV output representations; NOT all graph or memory-hierarchy domains",
            "copy_cost":"actual device copy inside complete-graph timing; byte fields are formula-derived, not measured HBM traffic"}
    def gemm(self,x,w,out):
        if w.data_ptr()!=self.target_weight_ptr or self.temp is None:
            return self.base.gemm(x,w,out)
        self.base.gemm(x,w,self.temp);out.copy_(self.temp)
        if self.check_copy:
            # The shared output backing is reused by later decoder layers.
            # Validate NOW, not after L7 has overwritten the L0 destination.
            self.copy_checks.append(bool(torch.equal(self.temp,out)))
    def verified(self,group,plan,reference):
        self.copy_checks=[];self.check_copy=True
        try:value=group.correctness(plan,reference)
        finally:self.check_copy=False
        guard=True if self.raw is None else bool((self.raw[:GUARD_ELEMENTS]==777).all() and
                                                 (self.raw[-GUARD_ELEMENTS:]==777).all())
        extra=self.temp is None or (bool(self.copy_checks) and all(self.copy_checks))
        value["materialization_bitwise_equal_before_buffer_reuse"]=extra
        value["materialization_guards"]=guard
        value["explicit_representation_evidence"]=self.domain_metadata
        value["correct"]=value["correct"] and extra and guard
        return value


def measure(group,adapter,case,choice,reference,warmup,iterations):
    plan=group.plan(case,choice["bits"]);adapter.prepare(plan,choice.get("source_column"))
    try:
        before=adapter.verified(group,plan,reference)
        if not before["correct"]:raise RuntimeError("pre-timing calculation/materialization correctness failed")
        samples=timed(lambda:group.run(plan),warmup,iterations)
        after=adapter.verified(group,plan,reference)
        if not after["correct"]:raise RuntimeError("post-timing calculation/materialization correctness failed")
        return {"choice":choice,"measured_samples_ms":samples,"measured_median_ms":statistics.median(samples),
                "pre_correctness":before,"post_correctness":after}
    finally:adapter.clear();del plan


def holdout(group,adapter,case,restricted,free,reference,args):
    samples={"restricted_ms":[],"free_ms":[],"orders":[]};checks=[]
    for rep in range(args.holdout_repetitions):
        order=("restricted","free") if rep%2==0 else ("free","restricted")
        for name in order:
            choice=restricted if name=="restricted" else free
            row=measure(group,adapter,case,choice,reference,1,args.iterations)
            samples[name+"_ms"].append(row["measured_median_ms"])
            checks.append({"repetition":rep,"name":name,"correct":row["post_correctness"]["correct"]})
        samples["orders"].append(list(order))
    samples["post_correctness_checks"]=checks
    return {"raw_measured_holdout":samples,"statistics":paired_bootstrap(samples["restricted_ms"],samples["free_ms"]),
            "same_choice":restricted==free,"numerator":"restricted full-graph median GPU time",
            "denominator":"free full-graph median GPU time",
            "selection":"choices selected using fresh training only; holdout not reused to select"}


def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True,type=Path)
    p.add_argument("--baseline-results",required=True,type=Path)
    p.add_argument("--framework",choices=("pytorch","triton","cutlass","tvm","vllm","sglang"),required=True)
    p.add_argument("--output",required=True,type=Path);p.add_argument("--artifacts",required=True,type=Path)
    p.add_argument("--mode",choices=("smoke","full"),default="smoke")
    p.add_argument("--warmup",type=int,default=3);p.add_argument("--iterations",type=int,default=10)
    p.add_argument("--holdout-repetitions",type=int,default=5);p.add_argument("--epsilon",type=float,default=.03)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError("refusing to overwrite supplemental experiment")
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError("single card1 visibility required")
    if min(args.warmup,args.iterations,args.holdout_repetitions)<1:raise ValueError("positive repetitions required")
    if not 0<=args.epsilon<1:raise ValueError("epsilon must be in [0,1)")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    manifest=json.loads(args.manifest.read_text());baseline={};baseline_fingerprints={}
    for line in args.baseline_results.read_text().splitlines():
        row=json.loads(line)
        if row["record_type"]=="run_metadata" and row["framework"]!=args.framework:
            raise ValueError("baseline is from a different framework/adapter")
        if row["record_type"]=="graph":
            key=row["case"]["case_id"]
            if key in baseline:raise ValueError("duplicate baseline case")
            baseline[key]=row
        if row["record_type"]=="group_data_fingerprint":
            baseline_fingerprints[json.dumps(row["group"])]=row["fingerprint"]
    cases=manifest["cases"]
    if args.mode=="smoke":
        cases=[c for c in cases if c["model_id"]=="Qwen/Qwen3-0.6B" and c["phase"]=="extend" and
               c["batch"]==4 and c["query_length"]==8 and len(set(c["kv_lengths"]))>1]
    grouped={}
    for c in cases:grouped.setdefault(group_key(c),[]).append(c)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.artifacts.mkdir(parents=True,exist_ok=True)
    adapter=RepresentationAdapter(Adapter(args.framework,args.artifacts));started=time.time()
    counts={"domain_graphs":0,"anchor_graphs":0,"anchor_semantically_inapplicable":0,"failures":0}
    with args.output.open("x") as f:
        def emit(row):f.write(json.dumps(row,ensure_ascii=False)+"\n");f.flush()
        emit({"record_type":"metadata","framework":args.framework,"mode":args.mode,"expected_graphs":len(cases),
            "manifest_sha256":hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "baseline_results_sha256":hashlib.sha256(args.baseline_results.read_bytes()).hexdigest(),
            "executor_code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "gpu":torch.cuda.get_device_name(0),"visible_gpu":os.environ.get("CUDA_VISIBLE_DEVICES"),
            "primitive_origins":adapter.origins,"dtype":"float16",
            "scope":"conditional runtime representation / remote-KV-anchor tests; NOT whole-engine policy or compiler propagation proof",
            "H2_2_fullgraph_K_epsilon_verified":False,"H2_3_compiler_control_flow_verified":False,
            "native_source_patch_replays_verified":False})
        for group_cases in grouped.values():
            group=None
            try:
                group=Group(group_cases[0],adapter,max(c["stage"] for c in group_cases));adapter.bind(group)
                if baseline_fingerprints.get(json.dumps(list(group_key(group_cases[0]))))!=group.data_fingerprint:
                    raise RuntimeError("baseline vs supplemental input/weight fingerprints disagree")
                emit({"record_type":"group_data_fingerprint","framework":args.framework,
                      "group":list(group_key(group_cases[0])),"fingerprint":group.data_fingerprint})
                for case in group_cases:
                    reference=None
                    try:
                        old=baseline[case["case_id"]]
                        if old["executed_graph_status"]!="correctness_verified_complete_controlled_graph":
                            raise RuntimeError("baseline graph has rejected/missing correctness candidates")
                        if old["case"]["graph_contract_sha256"]!=case["graph_contract_sha256"]:
                            raise RuntimeError("baseline graph semantic contract changed")
                        bits=old["best_train_plan"];axes=layout_axes(case)
                        reference=independent_graph_reference(group,case)
                        choices=representation_assignments(case,bits)
                        train=[measure(group,adapter,case,x,reference,args.warmup,args.iterations) for x in choices]
                        free=min(train,key=lambda r:r["measured_median_ms"])
                        shared=min((r for r in train if r["choice"]["explicit_output_representation_count"]==1),
                                   key=lambda r:r["measured_median_ms"])
                        near=[r for r in train if r["measured_median_ms"]<=(1+args.epsilon)*free["measured_median_ms"]]
                        domain_hold=holdout(group,adapter,case,shared["choice"],free["choice"],reference,args)
                        emit({"record_type":"conditional_domain_result","framework":args.framework,"case":case,
                            "fixed_other_layout_bits":bits,"training":train,"holdout":domain_hold,
                            "conditional_explicit_output_K_epsilon":min(r["choice"]["explicit_output_representation_count"] for r in near),
                            "domain_scope":"only L0 packed-QKV output ownership/representation; all other graph choices fixed",
                            "not_global_K_epsilon":True});counts["domain_graphs"]+=1
                        factors,reason=intervention_axes(case)
                        if factors is None:
                            emit({"record_type":"anchor_semantically_inapplicable","framework":args.framework,
                                  "case_id":case["case_id"],"reason":reason});counts["anchor_semantically_inapplicable"]+=1
                        else:
                            choices=factorial_assignments(bits,axes,factors["upstream_axis"],factors["anchor_axis"])
                            train=[measure(group,adapter,case,x,reference,args.warmup,args.iterations) for x in choices]
                            initial=[r for r in train if r["choice"]["anchor"]==0]
                            initial_best=min(r["measured_median_ms"] for r in initial)
                            old_near={r["choice"]["upstream"] for r in initial if r["measured_median_ms"]<=(1+args.epsilon)*initial_best}
                            changed=[r for r in train if r["choice"]["anchor"]==1]
                            free=min(changed,key=lambda r:r["measured_median_ms"])
                            frozen=min((r for r in changed if r["choice"]["upstream"] in old_near),key=lambda r:r["measured_median_ms"])
                            anchor_hold=holdout(group,adapter,case,frozen["choice"],free["choice"],reference,args)
                            emit({"record_type":"runtime_anchor_result","framework":args.framework,"case":case,
                                "factors":factors,"fixed_other_layout_bits":bits,"training_factorial":train,
                                "old_near_upstream_choices_with_NHD_anchor":sorted(old_near),"holdout":anchor_hold,
                                "distance_scope":"actual semantic DAG shortest path, NOT compiler SSA/layout-pass distance",
                                "compiler_propagation_proved":False});counts["anchor_graphs"]+=1
                        print(f"[{args.framework}] supplemental {case['case_id']} correct",flush=True)
                    except Exception as error:
                        counts["failures"]+=1
                        emit({"record_type":"case_failure","framework":args.framework,"case_id":case["case_id"],
                              "error":str(error),"traceback":traceback.format_exc()})
                        print(traceback.format_exc(),flush=True)
                    finally:adapter.clear();del reference;gc.collect();torch.cuda.empty_cache()
            except Exception as error:
                counts["failures"]+=1;emit({"record_type":"group_failure","framework":args.framework,
                    "group":list(group_key(group_cases[0])),"error":str(error),"traceback":traceback.format_exc()})
            finally:adapter.clear();del group;gc.collect();torch.cuda.empty_cache()
        emit({"record_type":"completion","framework":args.framework,"counts":counts,
              "expected_graphs":len(cases),"elapsed_seconds":time.time()-started,
              "all_RQ2_requirements_complete":False})
    return 1 if counts["failures"] or counts["domain_graphs"]!=len(cases) else 0


if __name__=="__main__":raise SystemExit(main())
