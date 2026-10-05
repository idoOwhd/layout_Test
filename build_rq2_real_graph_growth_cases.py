#!/usr/bin/env python3
"""Build provenance-checked nested REAL Qwen3 forward slices (CPU only).

This is a DESIGN manifest, not a GPU benchmark or evidence of any speedup.
No model weights are downloaded; no framework is imported. Other families
(MoE, MLA, sparse/hybrid/state/offload) require separate verified recipes.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
MODELS=("Qwen/Qwen3-0.6B","Qwen/Qwen3-1.7B","Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B","Qwen/Qwen3-14B","Qwen/Qwen3-32B")
STAGES=("norm_qkv","qk_norm_views","rope","attention_and_kv_update",
        "output_projection","postnorm_gateup_swiglu","one_decoder_block",
        "two_decoder_blocks","four_decoder_blocks","eight_decoder_blocks")


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def method(path,classname,name,required):
    text=path.read_text();tree=ast.parse(text)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==classname)
    func=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    code=ast.get_source_segment(text,func)
    if not all(token in code for token in required):
        raise ValueError(f"source recipe no longer matches {classname}.{name}: {required}")
    return {"path":str(path.resolve()),"class":classname,"method":name,
            "start_line":func.lineno,"end_line":func.end_lineno,
            "file_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "method_sha256":hashlib.sha256(code.encode()).hexdigest()}


def source_contract(vllm_dir,sglang_dir):
    rows=[method(vllm_dir/"qwen3.py","Qwen3Attention","forward",
                 ("self.qkv_proj","self.q_norm","self.k_norm","self.rotary_emb","self.attn","self.o_proj")),
          method(vllm_dir/"qwen3.py","Qwen3DecoderLayer","forward",
                 ("self.input_layernorm","self.self_attn","self.post_attention_layernorm","self.mlp")),
          method(vllm_dir/"qwen2.py","Qwen2MLP","forward",
                 ("self.gate_up_proj","self.act_fn","self.down_proj"))]
    if sglang_dir:
        rows.extend([method(sglang_dir/"qwen3.py","Qwen3Attention","forward_prepare_native",
                     ("self.qkv_proj","apply_qk_norm","self.rotary_emb")),
                     method(sglang_dir/"qwen3.py","Qwen3Attention","forward",
                     ("self.forward_prepare_native","self.attn","self.o_proj"))])
    return rows


def workload_shapes():
    rows=[]
    for batch in (1,4,16):
        for kv in (512,4096):rows.append(("decode",batch,1,kv))
    for batch in (1,4):
        for query in (128,1024):rows.append(("prefill",batch,query,query))
    for batch in (1,4):rows.append(("extend",batch,8,4096))
    return rows


def nodes_for_stage(config,batch,query,kv,stage):
    hidden=config["hidden_size"];hq=config["num_attention_heads"]
    hkv=config["num_key_value_heads"];dim=config["head_dim"]
    intermediate=config["intermediate_size"];tokens=batch*query
    nodes=[]
    def add(layer,name,kind,inputs,outputs,attrs=None,side_effects=False):
        row={"id":f"L{layer}.{name}","operator_kind":kind,"inputs":inputs,
             "outputs":{f"L{layer}.{k}":{"logical_shape":v,"dtype":"float16"} for k,v in outputs.items()},
             "attrs":attrs or {},"state_effects":side_effects,"native_adapter_status":"not_implemented"}
        row["semantic_signature"]=digest(row);nodes.append(row)
    layers={7:2,8:4,9:8}.get(stage,1)
    for layer in range(layers):
        cutoff=6 if stage>=6 else stage;prefix=f"L{layer}."
        hidden_input="x" if layer==0 else f"L{layer-1}.mlp_out"
        residual_input="x" if layer==0 else f"L{layer-1}.residual_after_attn"
        add(layer,"input_norm","rmsnorm" if layer==0 else "add_rmsnorm",
            [hidden_input] if layer==0 else [hidden_input,residual_input],
            {"norm_x":[tokens,hidden],"residual_in":[tokens,hidden]},
            {"eps":config["rms_norm_eps"],"first_residual_aliases_x":layer==0})
        add(layer,"qkv_projection","matmul",[prefix+"norm_x"],
            {"packed_qkv":[tokens,(hq+2*hkv)*dim]},
            {"weight_shape":[hidden,(hq+2*hkv)*dim],"bias":bool(config.get("attention_bias",False))})
        if cutoff==0:continue
        add(layer,"qkv_split","split_view",[prefix+"packed_qkv"],
            {"q_view":[tokens,hq,dim],"k_view":[tokens,hkv,dim],"v_view":[tokens,hkv,dim]},
            {"split_sizes":[hq*dim,hkv*dim,hkv*dim],"is_view_not_copy":True})
        add(layer,"q_norm","rmsnorm",[prefix+"q_view"],{"q_norm_out":[tokens,hq,dim]},
            {"eps":config["rms_norm_eps"],"reduction_axis":"head_dim"})
        add(layer,"k_norm","rmsnorm",[prefix+"k_view"],{"k_norm_out":[tokens,hkv,dim]},
            {"eps":config["rms_norm_eps"],"reduction_axis":"head_dim"})
        if cutoff==1:continue
        add(layer,"rope","rotary_embedding",[prefix+"q_norm_out",prefix+"k_norm_out","positions"],
            {"q_rope":[tokens,hq,dim],"k_rope":[tokens,hkv,dim]},
            {"rotary_dim":dim,"rope_theta":config.get("rope_theta"),"rope_scaling":config.get("rope_scaling")})
        if cutoff==2:continue
        add(layer,"attention","causal_gqa_with_kv_update",
            [prefix+"q_rope",prefix+"k_rope",prefix+"v_view",prefix+"kv_state_prev","request_metadata"],
            {"attn_out":[tokens,hq*dim],"kv_state_next":[2,batch,kv,hkv,dim]},
            {"causal":True,"scale":dim**-0.5,"request_count":batch,
             "query_length_per_request":query,"kv_length_per_request":kv,
             "metadata_types":"int32/int64","accumulation":"float32",
             "state_versioned":True,"kv_storage_layout":"undecided"},True)
        if cutoff==3:continue
        add(layer,"output_projection","matmul",[prefix+"attn_out"],{"attn_projected":[tokens,hidden]},
            {"weight_shape":[hq*dim,hidden],"bias":False})
        if cutoff==4:continue
        add(layer,"post_attention_norm","add_rmsnorm",[prefix+"attn_projected",prefix+"residual_in"],
            {"mlp_x":[tokens,hidden],"residual_after_attn":[tokens,hidden]},
            {"eps":config["rms_norm_eps"]})
        add(layer,"gate_up_projection","matmul",[prefix+"mlp_x"],{"gate_up":[tokens,2*intermediate]},
            {"weight_shape":[hidden,2*intermediate],"packed_order":"gate_then_up","bias":False})
        add(layer,"swiglu","silu_and_mul",[prefix+"gate_up"],{"mlp_act":[tokens,intermediate]},
            {"activation":"silu","order":"silu(gate)*up"})
        if cutoff==5:continue
        add(layer,"down_projection","matmul",[prefix+"mlp_act"],{"mlp_out":[tokens,hidden]},
            {"weight_shape":[intermediate,hidden],"bias":False})
    return nodes


def validate_nodes(nodes):
    # Weights/positions/cache-before/metadata are graph boundary inputs, not
    # fabricated extra operators. Every named activation dependency must exist.
    produced=set()
    for node in nodes:
        for name in node["inputs"]:
            if name in {"x","positions","request_metadata"} or name.endswith("kv_state_prev"):continue
            if name not in produced:raise ValueError(f"missing real graph edge: {name}")
        produced.update(node["outputs"])
    return True


def build(config_cache,manifest,vllm_dir,sglang_dir):
    sources=source_contract(vllm_dir,sglang_dir)
    parents=json.loads(manifest.read_text());cases=[];pairs=[]
    for model in MODELS:
        parent=next((r for r in parents if r["model_id"]==model and r["phase"]=="prefill" and r["structure"]=="gqa"),None)
        paths=sorted(config_cache.glob(model.replace("/","__")+"--*.json"))
        if parent:paths=[p for p in paths if p.stem.endswith(parent["model_revision"])]
        if len(paths)!=1:raise ValueError(f"model config must have one pinned revision: {model}, {paths}")
        path=paths[0];config=json.loads(path.read_text());revision=path.stem.split("--",1)[1]
        if config["model_type"]!="qwen3" or config.get("quantization_config"):
            raise ValueError(f"non-matching or quantized recipe: {model}")
        if (config["num_hidden_layers"]<8 or config["hidden_act"]!="silu" or config.get("use_sliding_window")
                or config.get("attention_dropout",0)!=0 or config.get("partial_rotary_factor",1)!=1
                or config["num_attention_heads"]%config["num_key_value_heads"]!=0):
            raise ValueError(f"unsupported layer/activation/attention contract: {model}")
        if parent:
            for field in ("hidden_size","head_dim","num_attention_heads","num_key_value_heads","intermediate_size"):
                if parent["dimensions"][field]!=config[field]:
                    raise ValueError(f"parent/canonical config mismatch: {model} {field}")
        for phase,batch,query,kv in workload_shapes():
            previous=None
            for stage,label in enumerate(STAGES):
                nodes=nodes_for_stage(config,batch,query,kv,stage);validate_nodes(nodes)
                key=f"{model.replace('/','__')}_{phase}_b{batch}_q{query}_kv{kv}_s{stage}"
                row={"case_id":key,"model_id":model,"model_revision":revision,
                     "parent_640_case_id":parent["case_id"] if parent else None,
                     "config_source":f"https://huggingface.co/{model}/blob/{revision}/config.json",
                     "config_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                     "forward_sources":sources,"phase":phase,"batch":batch,"query_length":query,"kv_length":kv,
                     "input_dtype":"float16","weight_dtype":"float16","kv_dtype":"float16",
                     "accumulation_dtype":"float32","metadata_dtype":"int32/int64",
                     "original_config_dtype":config.get("torch_dtype"),"stage":stage,"stage_label":label,
                     "nodes":nodes,"semantic_node_count":len(nodes),
                     "semantic_operator_kind_count":len({n["operator_kind"] for n in nodes}),
                     "has_actual_residual_branch":stage>=5,"decoder_blocks":{7:2,8:4,9:8}.get(stage,1),
                     "provenance_level":"pinned_real_model_geometry_and_checked_installed_forward_source",
                     "input_weight_values":"not generated here; later use seeded test tensors or real weights and label which",
                     "length_provenance":"controlled serving workload, not measured production traffic",
                     "graph_status":"design_only_not_executable_gpu_adapter","layout_measurements":None}
                row["graph_contract_sha256"]=digest(row)
                cases.append(row)
                if previous:
                    old={n["id"]:n["semantic_signature"] for n in previous["nodes"]}
                    new={n["id"]:n["semantic_signature"] for n in nodes}
                    if not all(new.get(k)==v for k,v in old.items()):raise ValueError("old core changed during growth")
                    pairs.append({"small_case_id":previous["case_id"],"expanded_case_id":key,
                                  "old_node_ids":list(old),"added_node_ids":[k for k in new if k not in old],
                                  "same_old_math_dtype_shape":True,"er_set_measured":False})
                previous=row
    return {"schema":"RQ2-real-forward-growth-design-v1","source_count":len(sources),
            "case_count":len(cases),"nested_pair_count":len(pairs),"model_count":len(MODELS),
            "stage_count":len(STAGES),"cases":cases,"nested_pairs":pairs,
            "new_gpu_experiments_completed":0,
            "limitations":"dense Qwen3 family only; cannot replace MoE/MLA/hybrid/sparse/offload coverage or native backend contract review"}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config-cache",type=Path,default=HERE.parent/"real_world_shapes/source_cache/configs")
    p.add_argument("--manifest",type=Path,default=HERE.parent/"real_world_shapes/real_world_shape_manifest.json")
    p.add_argument("--vllm-model-dir",type=Path,default=HERE/"framework_envs/layout-vllm/lib/python3.12/site-packages/vllm/model_executor/models")
    p.add_argument("--sglang-model-dir",type=Path,default=HERE/"framework_envs/layout-sglang/lib/python3.12/site-packages/sglang/srt/models")
    p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError("refusing to overwrite design manifest")
    value=build(args.config_cache,args.manifest,args.vllm_model_dir,args.sglang_model_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write("\n")
    print(json.dumps({k:v for k,v in value.items() if k not in ("cases","nested_pairs")},ensure_ascii=False))


if __name__=="__main__":main()
