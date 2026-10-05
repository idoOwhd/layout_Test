#!/usr/bin/env python3
"""Execute nested real Qwen3 FP16 graphs and set-aware layout counterfactuals.

All adapters disclose primitive origins. These are controlled/hybrid subgraph
experiments, not the unmodified policy of a complete serving engine. Inputs
are seeded synthetic values at pinned real model geometry, not model weights.
"""
from __future__ import annotations
import argparse
import ctypes
import gc
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch
from rq2_growth_search import (candidates, frozen_eligible, layout_axes, near_optimal,
                               stable_seed, summarize_pair)

CODE_DIR = Path(__file__).resolve().parent
HERE = Path(os.environ.get("RQ2_PROJECT_ROOT",str(CODE_DIR))).resolve()
GUARD_ELEMENTS=128  # 256 bytes: preserve allocator/native-kernel alignment.


def tensor_meta(t):
    return {"shape": list(t.shape), "stride": list(t.stride()), "dtype": str(t.dtype),
            "contiguous": t.is_contiguous(), "storage_offset": t.storage_offset(),
            "data_ptr_mod128":t.data_ptr()%128,"data_ptr_mod256":t.data_ptr()%256}


def sample_indices(length,count,device):
    # Float32 linspace can round n-1 UP to n for >2**24 elements and cause a
    # fatal device-side index assertion. Keep this index arithmetic int64.
    return (torch.arange(count,device=device,dtype=torch.long)*(length-1))//max(1,count-1)


def compare(a, b, propagated_operations=1):
    if a.shape != b.shape:
        return {"correct": False, "reason": "shape mismatch"}
    af,bf=a.float(),b.float()
    delta=(af-bf).abs(); rms=float(bf.square().mean().sqrt())
    # Local operator checks use the strict one-op threshold. Global comparison
    # additionally accounts for FP16 boundary errors propagated down the graph.
    # The independent NRMSE <=1% gate is unchanged; local checks are mandatory.
    absolute=max(1e-5,4*2**-10*rms*math.sqrt(propagated_operations))
    nrmse=float(delta.square().mean().sqrt())/max(rms,1e-12)
    finite=bool(torch.isfinite(af).all() and torch.isfinite(bf).all())
    return {"correct": finite and nrmse <= .01 and bool(torch.allclose(af,bf,rtol=.03,atol=absolute)),
            "max_abs_error": float(delta.max()), "nrmse": nrmse,
            "rtol": .03, "atol": absolute, "nrmse_limit": .01, "finite": finite,
            "propagated_operations":propagated_operations}


def timed(fn, warmup, iterations):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
            for _ in range(iterations)]
    for start,end in events:
        start.record();fn();end.record()
    torch.cuda.synchronize()
    return [float(start.elapsed_time(end)) for start,end in events]


def reference_attention(q,k,v,lengths=None):
    # FP32 reference, right-aligned causal mask for decode AND extend. Chunk
    # queries to bound reference memory; every output element is still checked.
    b,t,h,d=q.shape; factor=h//k.shape[2]; chunks=[]
    lengths=(torch.full((b,),k.shape[1],device=q.device,dtype=torch.long) if lengths is None else lengths.long())
    for start in range(0,t,32):
        row=torch.arange(start,min(start+32,t),device=q.device)[:,None]
        col=torch.arange(k.shape[1],device=q.device)[None,:]
        legal=(col[None,None]<lengths[:,None,None,None]) & (col[None,None]<=lengths[:,None,None,None]-t+row[None,None])
        head_outputs=[];count=min(32,t-start)
        # Fold the GQA group into the GEMM M dimension instead of replicating
        # the whole FP32 KV for Hq heads. This is the SAME independent math and
        # checks every element, but avoids several GB of reference-only memory.
        for kh in range(k.shape[2]):
            qq=q[:,start:start+count,kh*factor:(kh+1)*factor].permute(0,2,1,3).float().reshape(b,factor*count,d)
            kk=k[:,:,kh].float();vv=v[:,:,kh].float()
            scores=(torch.bmm(qq,kk.transpose(1,2))*d**-.5).view(b,factor,count,k.shape[1])
            scores.masked_fill_(~legal,-float("inf"))
            probs=scores.softmax(-1).reshape(b,factor*count,k.shape[1])
            result=torch.bmm(probs,vv).view(b,factor,count,d).permute(0,2,1,3).half()
            head_outputs.append(result)
        chunks.append(torch.cat(head_outputs,2))
    return torch.cat(chunks,1)


def frontier_names(stage):
    return ({"packed_qkv","residual_in"} if stage==0 else
            {"q_norm_out","k_norm_out","v_view","residual_in"} if stage==1 else
            {"q_rope","k_rope","v_view","residual_in"} if stage==2 else
            {"attn_out","residual_in"} if stage==3 else
            {"attn_projected","residual_in"} if stage==4 else
            {"mlp_act","residual_after_attn"} if stage==5 else
            {"mlp_out","residual_after_attn"})


def independent_graph_reference(group,case):
    """Independent functional topology: does NOT call Group.run or adapters.

    Round each semantic boundary to FP16, use FP32 math. Keep only graph
    frontier and new KV payloads; all historical cache is separately checked
    bit-for-bit. Every actual intermediate is also checked locally before its
    backing storage can be reused. This avoids reference-only OOM.
    """
    hq,hk,dim=group.hq,group.hk,group.d
    tokens=group.b*group.q;stage=case["stage"];cutoff=min(stage,6)
    outputs={};hidden=group.x;residual=None
    def rms(x,w):
        z=x.float()
        return (z*torch.rsqrt(z.square().mean(-1,keepdim=True)+group.config["rms_norm_eps"])*w.float()).half()
    def mm(x,w):return (x.float()@w.float()).half()
    def rotated(x):
        cos,sin=group.table[group.positions].float().chunk(2,-1)
        left,right=x.float().chunk(2,-1)
        return torch.cat((left*cos[:,None]-right*sin[:,None],right*cos[:,None]+left*sin[:,None]),-1).half()
    for layer in range(case["decoder_blocks"]):
        data=group.layers[layer];values={}
        if layer==0:
            residual=hidden;norm=rms(hidden,data["wn"])
        else:
            full_sum=hidden.float()+residual.float()
            norm=rms(full_sum,data["wn"]);residual=full_sum.half()
        values["residual_in"]=residual
        packed=mm(norm,data["wqkv"]);values["packed_qkv"]=packed
        if cutoff>=1:
            q=packed[:,:hq*dim].reshape(tokens,hq,dim)
            k=packed[:,hq*dim:(hq+hk)*dim].reshape(tokens,hk,dim)
            v=packed[:,(hq+hk)*dim:].reshape(tokens,hk,dim)
            q=rms(q,data["wqn"]);k=rms(k,data["wkn"])
            values.update(q_norm_out=q,k_norm_out=k,v_view=v)
        if cutoff>=2:
            q=rotated(q);k=rotated(k);values.update(q_rope=q,k_rope=k)
        if cutoff>=3:
            key=data["kprev"].clone();value=data["vprev"].clone()
            # Independent flat slot mapping rather than calling the measured
            # native writer / repair or sharing its multidimensional scatter.
            slots=(group.request_ids*group.kv+group.write_positions).flatten()
            key.view(-1,hk,dim).index_copy_(0,slots,k)
            value.view(-1,hk,dim).index_copy_(0,slots,v)
            outputs[f"L{layer}.kv_k_update"]=k.view(group.b,group.q,hk,dim).clone()
            outputs[f"L{layer}.kv_v_update"]=v.view(group.b,group.q,hk,dim).clone()
            attended=reference_attention(q.view(group.b,group.q,hq,dim),key,value,group.lengths)
            values["attn_out"]=attended.view(tokens,hq*dim)
        if cutoff>=4:
            projected=mm(values["attn_out"],data["wo"]);values["attn_projected"]=projected
        if cutoff>=5:
            total=projected.float()+residual.float();mlp_x=rms(total,data["wp"])
            residual=total.half();values["residual_after_attn"]=residual
            gate_up=mm(mlp_x,data["wgu"]);gate,up=gate_up.float().chunk(2,-1)
            act=(torch.nn.functional.silu(gate)*up).half();values["mlp_act"]=act
        if cutoff>=6:
            hidden=mm(act,data["wd"]);values["mlp_out"]=hidden
        if layer==case["decoder_blocks"]-1:
            for name in frontier_names(stage):outputs[f"L{layer}."+name]=values[name].clone()
    return outputs


class Adapter:
    def __init__(self, name, artifacts):
        self.name=name;self.artifacts=artifacts;self.dumps=set()
        self.first_layer_weight_digests={}
        self.origins={"add": "torch FP32 add with FP16 residual store",
                      "split": "zero-copy torch view", "repair": "timed torch contiguous/copy_",
                      "rope": "torch FP32 rotate with FP16 cos/sin table",
                      "kv_write": "Triton strided append with per-request lengths", "attention": "Triton fixed streaming GQA",
                      "rmsnorm": "Triton weighted RMS/add-RMS", "swiglu": "Triton SiLU*up",
                      "gemm": "Triton fixed tile GEMM"}
        if name in {"vllm","sglang"}:
            if name=="vllm":
                import vllm._custom_ops as ops
                self.ops=ops
                from vllm.v1.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash
                self.writer=triton_reshape_and_cache_flash
            else:
                import sgl_kernel
                self.ops=sgl_kernel
                from sglang.kernels.ops.kvcache.kvcache import store_cache
                self.writer=store_cache
            self.origins.update(rmsnorm=name+" native RMS/add-RMS wrapper",
                                swiglu=name+" native SiLU*up", rope=name+" native rotary_embedding",
                                gemm="torch/cuBLAS FP16 matmul (shared dependency)",
                                kv_write=name+" native NHD writer; HND has explicit timed tail repair")
            # The FlashInfer consumer is a native dependency, NOT a claim that
            # these wrappers reproduce the engine scheduler/default backend.
            import flashinfer
            self.flashinfer=flashinfer
            self.origins["attention"]="FlashInfer BatchPrefillWithPagedKVCacheWrapper backend=fa2"
        elif name=="pytorch":
            self.origins.update(gemm="torch/cuBLAS FP16 matmul",rmsnorm="torch FP32 arithmetic",
                                kv_write="torch per-request advanced-index KV writes",
                                swiglu="torch FP32 SiLU*up", attention="torch SDPA (explicit causal mask)")
        elif name=="reference":
            self.origins={k:"independent FP32 math, semantic FP16 boundary rounding" for k in self.origins}
        elif name=="cutlass":
            self.lib=ctypes.CDLL(str(artifacts/"rq2_cutlass.so"))
            self.lib.rq2_gemm.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4+[ctypes.c_ulonglong]
            self.lib.rq2_gemm.restype=ctypes.c_int
            self.origins["gemm"]="CUTLASS device::Gemm SM80 tensorOp fixed 128x128x32, direct row/column output"
        elif name=="tvm":
            # Import the existing Python-3.12 TVM build from another environment
            # without installing torch or modifying either environment.
            site=HERE/"framework_envs/layout-tvm/lib/python3.12/site-packages"
            sys.path[0:0]=[str(HERE/"framework_sources/tvm/python"),str(site)]
            import tvm
            import tvm_ffi
            self.tvm=tvm;self.ffi=tvm_ffi;self.tvm_cache={}
            self.origins["rmsnorm"]="TVM TE->S-TIR->CUDA weighted RMS/add-RMS with thread reduction"
        if name not in {"pytorch","reference","vllm","sglang"}:
            from rq1_diverse_kernels import gemm_kernel
            self.triton_gemm_kernel=gemm_kernel
        if name not in {"pytorch","reference"}:
            import rq2_growth_kernels as kernels
            self.kernels=kernels

    def dump(self,kernel,kind):
        if kernel is None:return
        ident=getattr(kernel,"hash",None) or hashlib.sha256(str(kernel.asm).encode()).hexdigest()
        key=(kind,ident)
        if key in self.dumps:return
        self.dumps.add(key)
        for ext,value in getattr(kernel,"asm",{}).items():
            if ext in {"ttir","ttgir","ptx"}:
                (self.artifacts/f"{self.name}_{kind}_{ident}.{ext}").write_text(str(value))
        metadata=getattr(kernel,"metadata",None)
        metadata=metadata._asdict() if hasattr(metadata,"_asdict") else str(metadata)
        with (self.artifacts/f"{self.name}_{kind}_{ident}.metadata.json").open("x") as f:
            json.dump({"kind":kind,"compiled_hash":ident,"metadata":metadata,
                       "registers_per_thread":getattr(kernel,"n_regs",None),
                       "spills":getattr(kernel,"n_spills",None)},f,indent=2,default=str)

    def gemm(self,x,w,out):
        if self.name=="reference":out.copy_((x.float()@w.float()).half())
        elif self.name in {"pytorch","vllm","sglang"}:torch.mm(x,w,out=out)
        elif self.name=="cutlass":
            # CUTLASS A contract is row-major; measure this repair if needed.
            a=x.contiguous()
            status=self.lib.rq2_gemm(a.data_ptr(),w.data_ptr(),out.data_ptr(),x.shape[0],
                                    w.shape[1],x.shape[1],int(out.stride(0)==1 and x.shape[0]>1),
                                    torch.cuda.current_stream().cuda_stream)
            if status:raise RuntimeError(f"CUTLASS can_implement/run status {status}")
        else:
            # Same primitive/tile as the preserved RQ1 helper, but retain the
            # compiled handle to save ACTUAL TTGIR/PTX for every specialization.
            import triton
            aa=x.unsqueeze(0);ww=w.unsqueeze(0);oo=out.unsqueeze(0)
            groups,m,k=aa.shape;n=ww.shape[-1]
            kernel=self.triton_gemm_kernel[(triton.cdiv(m,32),triton.cdiv(n,64),groups)](
                aa,ww,oo,m,n,k,*aa.stride(),*ww.stride(),*oo.stride(),0,0,False,0,0,num_warps=4)
            self.dump(kernel,"gemm")

    def norm(self,x,w,eps,residual=None):
        # Residual sum and normalization use FP32 BEFORE residual rounding.
        # Native fused kernels require dense last axis; repair is in the graph.
        if self.name in {"vllm","sglang"}:
            a=x.contiguous();out=torch.empty_like(a)
            if residual is None:
                if self.name=="vllm":self.ops.rms_norm(out,a,w,eps)
                else:self.ops.rmsnorm(a,w,eps=eps,out=out)
                return out,None
            r=residual.contiguous().clone();z=a.clone()
            if self.name=="vllm":self.ops.fused_add_rms_norm(z,r,w,eps)
            else:self.ops.fused_add_rmsnorm(z,r,w,eps=eps)
            return z,r
        out=torch.empty(x.shape,device=x.device,dtype=x.dtype)
        r=residual.contiguous().clone() if residual is not None else None
        if self.name in {"reference","pytorch"}:
            a=x.float()+(r.float() if r is not None else 0)
            out.copy_((a*torch.rsqrt(a.square().mean(-1,keepdim=True)+eps)*w.float()).half())
            if r is not None:r.copy_(a.half())
        elif self.name=="tvm":self.tvm_norm(x,w,out,eps,r)
        else:self.dump(self.kernels.rms(x,w,out,eps,r),"rms")
        return out,r

    def tvm_norm(self,x,w,out,eps,residual):
        from tvm import te
        rows,cols=x.shape;key=(rows,cols,x.stride(),eps,residual is not None)
        if key not in self.tvm_cache:
            # Input view is represented by flat PHYSICAL storage and explicit
            # strides; no hidden DLPack contiguous repair outside timing.
            s0,s1=x.stride();extent=(rows-1)*s0+(cols-1)*s1+1
            a=te.placeholder((extent,),"float16",name="A")
            wt=te.placeholder((cols,),"float16",name="W")
            r=te.placeholder((rows,cols),"float16",name="R") if residual is not None else None
            def value(i,j):return a[i*s0+j*s1].astype("float32")+(r[i,j].astype("float32") if r is not None else 0)
            k=te.reduce_axis((0,cols),"k")
            sums=te.compute((rows,),lambda i:te.sum(value(i,k)*value(i,k),axis=k),name="sums")
            result=te.compute((rows,cols),lambda i,j:(value(i,j)*te.rsqrt(sums[i]/cols+eps)*wt[j].astype("float32")).astype("float16"),name="out")
            updated=te.compute((rows,cols),lambda i,j:value(i,j).astype("float16"),name="residual_out") if r is not None else None
            inputs=[a,wt]+([r] if r is not None else [])
            outputs=[result]+([updated] if updated is not None else [])
            schedule=self.tvm.s_tir.Schedule(te.create_prim_func(inputs+outputs))
            block=schedule.get_sblock("sums");i,kk=schedule.get_loops(block)
            ko,ki=schedule.split(kk,factors=[None,128]);schedule.bind(i,"blockIdx.x");schedule.bind(ki,"threadIdx.x")
            for name in ("out",)+(("residual_out",) if r is not None else ()):
                block=schedule.get_sblock(name);i,j=schedule.get_loops(block)
                jo,ji=schedule.split(j,factors=[None,128]);schedule.bind(i,"blockIdx.x");schedule.bind(ji,"threadIdx.x")
            module=self.tvm.compile(schedule.mod,target={"kind":"cuda","arch":"sm_86"})
            module=module.jit() if hasattr(module,"jit") else module
            self.tvm_cache[key]=module
            ident=hashlib.sha256(str(key).encode()).hexdigest()[:16]
            (self.artifacts/f"tvm_norm_{ident}.tir").write_text(str(schedule.mod))
        extent=(rows-1)*x.stride(0)+(cols-1)*x.stride(1)+1
        physical=x.as_strided((extent,),(1,))
        inputs=[physical,w]+([residual] if residual is not None else [])
        # Native TVM CUDA launch uses its own current stream. Explicitly share
        # torch's stream, otherwise event timings and dependencies are invalid.
        self.tvm.cuda(0).set_raw_stream(torch.cuda.current_stream().cuda_stream)
        outputs=[out]
        if residual is not None:outputs.append(torch.empty_like(residual))
        arguments=[self.ffi.from_dlpack(t,require_alignment=0,require_contiguous=True) for t in inputs+outputs]
        self.tvm_cache[key](*arguments)
        if residual is not None:residual.copy_(outputs[1])

    def swiglu(self,x):
        n=x.shape[1]//2;out=torch.empty(x.shape[0],n,device=x.device,dtype=x.dtype)
        if self.name in {"reference","pytorch"}:
            out.copy_((torch.nn.functional.silu(x[:,:n].float())*x[:,n:].float()).half())
        elif self.name=="vllm":torch.ops._C.silu_and_mul(out,x.contiguous())
        elif self.name=="sglang":self.ops.silu_and_mul(x.contiguous(),out=out)
        else:self.dump(self.kernels.silu(x,out),"silu")
        return out

    def rope(self,q,k,positions,table):
        q=q.contiguous();k=k.contiguous();n,h,d=q.shape
        # Actual Q/K norm outputs are newly allocated each execution, so the
        # real in-place rotary contract needs no repeat-protection clone. The
        # LOCAL oracle must clone to avoid altering the candidate's inputs.
        if self.name=="reference":q=q.clone();k=k.clone()
        if self.name in {"vllm","sglang"}:
            self.ops.rotary_embedding(positions,q.view(n,-1),k.view(n,-1),d,table,True)
        else:
            cos,sin=table[positions].float().chunk(2,-1)
            for x in (q,k):
                a,b=x.float().chunk(2,-1)
                x.copy_(torch.cat((a*cos[:,None]-b*sin[:,None],b*cos[:,None]+a*sin[:,None]),-1).half())
        return q,k

    def attention(self,q,k,v,out,plan):
        if self.name=="reference":out.copy_(reference_attention(q,k,v,plan["lengths"]))
        elif self.name=="pytorch":
            factor=q.shape[2]//k.shape[2]
            kk=k.transpose(1,2).repeat_interleave(factor,1)
            vv=v.transpose(1,2).repeat_interleave(factor,1)
            mask=plan["causal_mask"]
            out.copy_(torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),kk,vv,attn_mask=mask).transpose(1,2))
        elif self.name in {"vllm","sglang"}:
            cache_k,cache_v=plan["flash_k"],plan["flash_v"]
            # Paged views share the actual selected NHD/HND cache, no gather.
            out.copy_(plan["flash_wrapper"].run(q.reshape(-1,q.shape[2],q.shape[3]),(cache_k,cache_v)).view_as(out))
        else:self.dump(self.kernels.attention(q,k,v,out,plan["lengths"]),"attention")


class Group:
    def __init__(self,case,adapter,max_stage):
        self.case=case;self.adapter=adapter
        raw=case.get("pinned_config_json")
        if raw is None:
            raw=(HERE.parent/"real_world_shapes/source_cache/configs"/
                (case["model_id"].replace("/","__")+"--"+case["model_revision"]+".json")).read_text()
        if hashlib.sha256(raw.encode()).hexdigest()!=case["config_sha256"]:
            raise ValueError("pinned config hash mismatch")
        self.config=json.loads(raw)
        c=self.config;self.b=case["batch"];self.q=case["query_length"];self.kv=case["kv_length"]
        if c.get("rope_scaling") or c.get("attention_bias",False) or c["hidden_act"]!="silu":
            raise ValueError("unsupported rope scaling, projection bias or activation contract")
        self.h=c["hidden_size"];self.hq=c["num_attention_heads"];self.hk=c["num_key_value_heads"];self.d=c["head_dim"];self.i=c["intermediate_size"]
        self.key=(case["model_id"],case["phase"],self.b,self.q,self.kv)
        self.lengths_cpu=case.get("kv_lengths",[self.kv]*self.b)
        if len(self.lengths_cpu)!=self.b or not all(self.q<=n<=self.kv for n in self.lengths_cpu):
            raise ValueError("invalid per-request KV lengths")
        self.lengths=torch.tensor(self.lengths_cpu,device="cuda",dtype=torch.long)
        self.write_positions=self.lengths[:,None]-self.q+torch.arange(self.q,device="cuda")[None,:]
        self.request_ids=torch.arange(self.b,device="cuda")[:,None]
        layers={7:2,8:4,9:8}.get(max_stage,1);self.layers=[]
        def rand(shape,name,scale):
            seed_key=(case["model_id"],) if name.startswith("L") and name.split(".",1)[1].startswith("w") else self.key
            generator=torch.Generator(device="cuda");generator.manual_seed(stable_seed(seed_key,name))
            return torch.randn(shape,device="cuda",dtype=torch.float16,generator=generator)*scale
        self.x=rand((self.b*self.q,self.h),"x",.5)
        self.positions=self.write_positions.flatten().long()
        inv=1/(float(c.get("rope_theta",1000000))**(torch.arange(0,self.d,2,device="cuda",dtype=torch.float32)/self.d))
        freqs=torch.arange(self.kv,device="cuda",dtype=torch.float32)[:,None]*inv
        self.table=torch.cat((freqs.cos(),freqs.sin()),-1).half()
        for layer in range(layers):
            prefix=f"L{layer}."
            self.layers.append({
                "wqkv":rand((self.h,(self.hq+2*self.hk)*self.d),prefix+"wqkv",self.h**-.5),
                "wo":rand((self.hq*self.d,self.h),prefix+"wo",(self.hq*self.d)**-.5),
                "wgu":rand((self.h,2*self.i),prefix+"wgu",self.h**-.5),
                "wd":rand((self.i,self.h),prefix+"wd",self.i**-.5),
                "wn":rand((self.h,),prefix+"wn",.05)+1,
                "wp":rand((self.h,),prefix+"wp",.05)+1,
                "wqn":rand((self.d,),prefix+"wqn",.05)+1,
                "wkn":rand((self.d,),prefix+"wkn",.05)+1,
                "kprev":rand((self.b,self.kv,self.hk,self.d),prefix+"kprev",.5),
                "vprev":rand((self.b,self.kv,self.hk,self.d),prefix+"vprev",.5)})
        def digest_tensor(t,full=False):
            flat=t.flatten();hasher=hashlib.sha256()
            if full:
                for start in range(0,flat.numel(),4*1024*1024):
                    hasher.update(flat[start:start+4*1024*1024].cpu().numpy().tobytes())
            else:
                count=min(flat.numel(),8192)
                ids=sample_indices(flat.numel(),count,t.device)
                hasher.update(flat[ids].cpu().numpy().tobytes())
            return hasher.hexdigest()
        model=case["model_id"]
        if model not in adapter.first_layer_weight_digests:
            adapter.first_layer_weight_digests[model]={name:digest_tensor(value,full=True)
                for name,value in self.layers[0].items() if name.startswith("w")}
        self.data_fingerprint={"first_layer_full_weight_sha256":adapter.first_layer_weight_digests[model],
            "sample_protocol":"up to 8192 integer-floor-spaced FP16 elements; not a full-state digest",
            "input_sample_sha256":digest_tensor(self.x),
            "first_layer_kprev_sample_sha256":digest_tensor(self.layers[0]["kprev"]),
            "first_layer_vprev_sample_sha256":digest_tensor(self.layers[0]["vprev"]),
            "rope_table_full_sha256":digest_tensor(self.table,full=True)}

    def plan(self,case,bits):
        if len(bits)!=len(layout_axes(case)):raise ValueError("layout assignment length mismatch")
        choices=dict(zip(layout_axes(case),bits));layerplans=[]
        stage=case["stage"];cutoff=6 if stage>=6 else stage
        shared={}
        for l in range(case["decoder_blocks"]):
            data=self.layers[l];p={};n=self.b*self.q
            for name,width in (("qkv_projection",(self.hq+2*self.hk)*self.d),
                               ("output_projection",self.h),("gate_up_projection",2*self.i),
                               ("down_projection",self.h)):
                col=choices.get(f"L{l}."+name,0)
                # Serial decoder execution permits one backing per tensor KIND,
                # independent row/column views per layer. Old hidden is consumed
                # by input norm before down-projection reuses its backing. All
                # intermediates are checked BEFORE the next layer overwrites it.
                if name not in shared:
                    shared[name]=torch.full((n*width+2*GUARD_ELEMENTS,),777,device="cuda",dtype=torch.float16)
                raw=shared[name]
                out=raw[GUARD_ELEMENTS:-GUARD_ELEMENTS].view(width,n).t() if col else raw[GUARD_ELEMENTS:-GUARD_ELEMENTS].view(n,width)
                p[name]=out;p[name+"_canary"]=raw
            hnd=choices.get(f"L{l}.kv_cache",0)
            p["kv_layout"]="HND" if hnd else "NHD"
            p["native_paged_hnd"]=bool(hnd and cutoff>=3 and self.adapter.name in {"vllm","sglang"})
            for kind in ("k","v"):
                original=data[kind+"prev"]
                # Guard complete physical allocation on both sides.
                raw=torch.full((original.numel()+2*GUARD_ELEMENTS,),777,device="cuda",dtype=torch.float16)
                if p["native_paged_hnd"]:
                    cache=raw[GUARD_ELEMENTS:-GUARD_ELEMENTS].view(self.b,self.kv//16,self.hk,16,self.d)
                    cache.copy_(original.view(self.b,self.kv//16,16,self.hk,self.d).transpose(2,3))
                else:
                    cache=(raw[GUARD_ELEMENTS:-GUARD_ELEMENTS].view(self.b,self.hk,self.kv,self.d).transpose(1,2) if hnd
                           else raw[GUARD_ELEMENTS:-GUARD_ELEMENTS].view_as(original))
                    cache.copy_(original)
                p[kind]=cache;p[kind+"_canary"]=raw
            if cutoff>=3 and self.adapter.name in {"vllm","sglang"}:
                ids=(self.request_ids*self.kv+self.write_positions).flatten()
                p["slots"]=(torch.arange(n,device="cuda",dtype=torch.long) if hnd else ids.long())
                p["scale"]=torch.ones(1,device="cuda",dtype=torch.float32)
                if hnd:
                    # Only NEW tokens use a compact native-writer staging cache.
                    # Existing KV stays in true compact paged HND. The measured
                    # repair scatters these tokens, never reorders the full KV.
                    for kind in ("k","v"):
                        stage_raw=torch.full((((n+15)//16)*16*self.hk*self.d+2*GUARD_ELEMENTS,),777,
                                             device="cuda",dtype=torch.float16)
                        p["write_"+kind]=stage_raw[GUARD_ELEMENTS:-GUARD_ELEMENTS].view(-1,16,self.hk,self.d)
                        p["staging_"+kind+"_canary"]=stage_raw
                    p["target_pages"]=(ids//self.kv*(self.kv//16)+(ids%self.kv)//16).long()
                    p["target_offsets"]=(ids%16).long()
                else:
                    p["write_k"]=p["k"];p["write_v"]=p["v"]
                page=16
                if self.kv%page:raise ValueError("exact paged cache requires divisible KV length")
                for kind in ("k","v"):
                    z=p[kind]
                    p["flash_"+kind]=(z.view(-1,self.hk,page,self.d) if hnd else z.view(-1,page,self.hk,self.d))
                qo=torch.arange(self.b+1,dtype=torch.int32)*(self.q)
                counts=[(n+page-1)//page for n in self.lengths_cpu]
                kp=torch.tensor([0]+list(itertools.accumulate(counts)),dtype=torch.int32)
                ki=torch.tensor([request*(self.kv//page)+i for request,count in enumerate(counts) for i in range(count)],dtype=torch.int32)
                last=torch.tensor([(n-1)%page+1 for n in self.lengths_cpu],dtype=torch.int32)
                from rq1_flashinfer_workspace import plan_with_workspace,INITIAL_BYTES,MAX_BYTES
                def factory(workspace):return self.adapter.flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    workspace,kv_layout="HND" if hnd else "NHD",backend="fa2")
                def planner(wrapper):wrapper.plan(qo,kp,ki,last,self.hq,self.hk,self.d,page,causal=True,
                             q_data_type=torch.float16,kv_data_type=torch.float16)
                initial=torch.empty(INITIAL_BYTES,device="cuda",dtype=torch.uint8)
                wrapper,info=plan_with_workspace(factory,planner,
                    lambda size:torch.empty(size,device="cuda",dtype=torch.uint8),initial)
                p["workspace_info"]=info
                p["flash_wrapper"]=wrapper
            col=torch.arange(self.kv,device="cuda")[None,None,None,:]
            p["causal_mask"]=(col<self.lengths[:,None,None,None]) & (
                col<=self.lengths[:,None,None,None]-self.q+torch.arange(self.q,device="cuda")[None,None,:,None])
            p["lengths"]=self.lengths
            layerplans.append(p)
        return {"bits":list(bits),"axes":layout_axes(case),"layers":layerplans,"case":case}

    def logical_cache(self,p,kind):
        if p["native_paged_hnd"]:
            # Canonical materialization is for REFERENCE/CHECKS only. The timed
            # consumer reads the paged HND allocation directly, without this.
            return p[kind].transpose(2,3).reshape(self.b,self.kv,self.hk,self.d)
        return p[kind]

    def run(self,plan,adapter=None,trace=False,validate=False):
        a=adapter or self.adapter;case=plan["case"];stage=case["stage"]
        cutoff=6 if stage>=6 else stage;hidden=self.x;residual=None;outputs={};layouts={};local={}
        oracle=Adapter("reference",a.artifacts) if validate else None
        def check(l,name,actual,expected):
            if validate:local[f"L{l}."+name]=compare(actual,expected)
        def norm_checked(l,name,x,w,eps,r=None):
            y,rr=a.norm(x,w,eps,r)
            if validate:
                ey,er=oracle.norm(x,w,eps,r);check(l,name,y,ey)
                if rr is not None:check(l,name+"_residual",rr,er)
            return y,rr
        def gemm_checked(l,name,x,w,out):
            a.gemm(x,w,out)
            if validate:check(l,name,out,(x.float()@w.float()).half())
        def save(l,name,t):
            if trace:
                key=f"L{l}."+name;layouts[key]=tensor_meta(t)
                if name in {"kv_k_next","kv_v_next"}:
                    update_key=key.replace("_next","_update")
                    outputs[update_key]=t[self.request_ids,self.write_positions].clone()
                elif l==case["decoder_blocks"]-1 and name in frontier_names(stage):
                    outputs[key]=t.clone()
        for l,p in enumerate(plan["layers"]):
            d=self.layers[l];eps=self.config["rms_norm_eps"]
            norm,r=norm_checked(l,"input_norm",hidden,d["wn"],eps,residual)
            residual=hidden if r is None else r
            save(l,"norm_x",norm);save(l,"residual_in",residual)
            packed=p["qkv_projection"];gemm_checked(l,"qkv_projection",norm,d["wqkv"],packed);save(l,"packed_qkv",packed)
            if cutoff==0:continue
            q,k,v=packed.split((self.hq*self.d,self.hk*self.d,self.hk*self.d),-1)
            q=q.view(-1,self.hq,self.d);k=k.view(-1,self.hk,self.d);v=v.view(-1,self.hk,self.d)
            if validate:
                check(l,"q_split",q,packed[:,:self.hq*self.d].reshape_as(q))
                check(l,"k_split",k,packed[:,self.hq*self.d:(self.hq+self.hk)*self.d].reshape_as(k))
                check(l,"v_split",v,packed[:,(self.hq+self.hk)*self.d:].reshape_as(v))
            save(l,"q_view",q);save(l,"k_view",k);save(l,"v_view",v)
            # view cannot flatten token/head dimensions of a packed split.
            # This real native contract repair is measured, not a free reshape.
            qn,_=norm_checked(l,"q_norm",q.contiguous().view(-1,self.d),d["wqn"],eps)
            kn,_=norm_checked(l,"k_norm",k.contiguous().view(-1,self.d),d["wkn"],eps)
            q=qn.view(-1,self.hq,self.d);k=kn.view(-1,self.hk,self.d)
            save(l,"q_norm_out",q);save(l,"k_norm_out",k)
            if cutoff==1:continue
            if validate:rq,rk=oracle.rope(q,k,self.positions,self.table)
            q,k=a.rope(q,k,self.positions,self.table);save(l,"q_rope",q);save(l,"k_rope",k)
            if validate:check(l,"rope_q",q,rq);check(l,"rope_k",k,rk)
            if cutoff==2:continue
            if a.name in {"vllm","sglang"}:
                keys=k.contiguous()
                # Both native writers accept arbitrary TOKEN row strides.
                # Within the head-vector row, however, they require dense H*D.
                # A packed-row V split is already legal; only column emission
                # needs repair. Forcing contiguous unconditionally fabricates
                # a copy that the original native consumer does not require.
                values=v if v.stride(-1)==1 and v.stride(-2)==self.d else v.contiguous()
                if a.name=="vllm":
                    a.writer(keys,values,p["write_k"].view(-1,16,self.hk,self.d),
                             p["write_v"].view(-1,16,self.hk,self.d),p["slots"],"auto",p["scale"],p["scale"])
                else:
                    a.writer(keys.view(-1,self.hk*self.d),values.view(-1,self.hk*self.d),
                             p["write_k"].view(-1,self.hk*self.d),p["write_v"].view(-1,self.hk*self.d),
                             p["slots"],reserved_skip_index=-1)
                if p["kv_layout"]=="HND":
                    for kind in ("k","v"):
                        new=p["write_"+kind].view(-1,self.hk,self.d)[:self.b*self.q]
                        p["flash_"+kind][p["target_pages"],:,p["target_offsets"],:]=new
            else:
                for kind,new in (("k",k),("v",v)):
                    if a.name in {"reference","pytorch"}:
                        p[kind][self.request_ids,self.write_positions]=new.view(self.b,self.q,self.hk,self.d)
                    else:
                        self.kernels_append(a,new,p[kind])
            if trace:
                save(l,"kv_k_next",self.logical_cache(p,"k"));save(l,"kv_v_next",self.logical_cache(p,"v"))
                layouts[f"L{l}.physical_k_cache"]=tensor_meta(p["k"])
                layouts[f"L{l}.physical_v_cache"]=tensor_meta(p["v"])
                layouts[f"L{l}.physical_cache_contract"]={"layout":p["kv_layout"],
                    "implementation":"compact_paged_HND_native_consumer" if p["native_paged_hnd"] else
                        "compact_paged_NHD_native_consumer" if a.name in {"vllm","sglang"} else "dense_strided_cache",
                    "canonical_check_materialization_outside_timing":p["native_paged_hnd"],
                    "measured_repair":"native NHD new-token staging -> paged HND scatter" if p["native_paged_hnd"] else "none"}
            if validate:
                for kind,new in (("k",k),("v",v)):
                    local[f"L{l}.kv_{kind}_write_bitwise"]={"correct":torch.equal(
                        self.logical_cache(p,kind)[self.request_ids,self.write_positions],new.view(self.b,self.q,self.hk,self.d)),
                        "scope":"every newly written KV element, exact FP16 bit pattern"}
            attn=torch.empty(self.b,self.q,self.hq,self.d,device="cuda",dtype=torch.float16)
            a.attention(q.view(self.b,self.q,self.hq,self.d),p["k"],p["v"],attn,p)
            if validate:check(l,"attention",attn,reference_attention(q.view_as(attn),self.logical_cache(p,"k"),self.logical_cache(p,"v"),self.lengths))
            save(l,"attn_out",attn.view(-1,self.hq*self.d))
            if cutoff==3:continue
            projection=p["output_projection"];gemm_checked(l,"output_projection",attn.view(-1,self.hq*self.d),d["wo"],projection)
            save(l,"attn_projected",projection)
            if cutoff==4:continue
            mlp_x,residual=norm_checked(l,"post_attention_norm",projection,d["wp"],eps,residual)
            save(l,"mlp_x",mlp_x);save(l,"residual_after_attn",residual)
            gu=p["gate_up_projection"];gemm_checked(l,"gate_up_projection",mlp_x,d["wgu"],gu);save(l,"gate_up",gu)
            act=a.swiglu(gu);save(l,"mlp_act",act)
            if validate:check(l,"swiglu",act,oracle.swiglu(gu))
            if cutoff==5:continue
            hidden=p["down_projection"];gemm_checked(l,"down_projection",act,d["wd"],hidden);save(l,"mlp_out",hidden)
        return (outputs,layouts,local) if validate else (outputs,layouts)

    def kernels_append(self,adapter,new,cache):
        adapter.dump(adapter.kernels.append(new,cache,self.lengths,self.q),"kv_append")

    def correctness(self,plan,reference):
        value,layouts,local=self.run(plan,trace=True,validate=True)
        parts={name:compare(t,reference[name],propagated_operations=12*(int(name.split('.')[0][1:])+1))
               for name,t in value.items()}
        guards={}
        for l,p in enumerate(plan["layers"]):
            for name,t in p.items():
                if name.endswith("_canary"):
                    guards[f"L{l}.{name}"]=bool((t[:GUARD_ELEMENTS]==777).all() and (t[-GUARD_ELEMENTS:]==777).all())
            if plan["case"]["stage"]>=3:
                for kind in ("k","v"):
                    actual=self.logical_cache(p,kind);before=self.layers[l][kind+"prev"]
                    guards[f"L{l}.{kind}_unwritten_bitwise"]=all(
                        torch.equal(actual[req,:length-self.q],before[req,:length-self.q]) and
                        torch.equal(actual[req,length:],before[req,length:])
                        for req,length in enumerate(self.lengths_cpu))
                    if p["native_paged_hnd"]:
                        guards[f"L{l}.{kind}_unused_staging_bitwise"]=bool(
                            (p["write_"+kind].flatten()[self.b*self.q*self.hk*self.d:]==777).all())
        return {"correct": set(value)==set(reference) and all(v["correct"] for v in parts.values())
                           and all(v["correct"] for v in local.values()) and all(guards.values()),
                "scope":"every intermediate/residual checked immediately by independent FP32 per-op oracle; independent functional whole-graph frontier and KV updates; complete unwritten/padding KV bitwise and all allocation guards",
                "parts":parts,"local_operator_checks":local,"guards":guards,"observed_layouts":layouts}


def group_key(c):return (c["model_id"],c["phase"],c["batch"],c["query_length"],c["kv_length"],
                         tuple(c.get("kv_lengths",[c["kv_length"]]*c["batch"])))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,default=HERE/"ref_talks/rq2_qwen3_real_forward_growth_mixed_design_20261002_v3.json")
    p.add_argument("--framework",choices=("pytorch","triton","vllm","sglang","tvm","cutlass"),required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--artifacts",type=Path,required=True)
    p.add_argument("--mode",choices=("smoke","full"),default="smoke")
    p.add_argument("--warmup",type=int,default=3);p.add_argument("--iterations",type=int,default=10)
    p.add_argument("--holdout-repetitions",type=int,default=5)
    p.add_argument("--epsilon",type=float,default=.03);p.add_argument("--candidate-budget",type=int,default=48)
    p.add_argument("--only-group");p.add_argument("--group-index",type=int)
    p.add_argument("--smoke-extended",action="store_true")
    args=p.parse_args()
    if args.output.exists():raise FileExistsError("refusing to overwrite benchmark results")
    if min(args.warmup,args.iterations,args.holdout_repetitions)<1:raise ValueError("positive repetitions required")
    args.artifacts.mkdir(parents=True,exist_ok=True);args.output.parent.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError("CUDA not visible to this process")
    if torch.cuda.device_count()!=1:raise RuntimeError("restrict CUDA_VISIBLE_DEVICES to card1 UUID")
    if compare(torch.zeros(128,device="cuda",dtype=torch.float16),torch.ones(128,device="cuda",dtype=torch.float16))["correct"]:
        raise RuntimeError("correctness checker failed zero-output negative control")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    adapter=Adapter(args.framework,args.artifacts)
    manifest=json.loads(args.manifest.read_text());grouped={}
    for c in manifest["cases"]:grouped.setdefault(group_key(c),[]).append(c)
    groups=list(grouped.values())
    if args.mode=="smoke":
        # Still use REAL, UNREDUCED model dimensions. Decode multi-request and
        # prefill exercise distinct layouts; include growth to TWO blocks.
        groups=[g for g in groups if g[0]["model_id"]=="Qwen/Qwen3-0.6B" and
                (g[0]["phase"],g[0]["batch"],g[0]["query_length"],g[0]["kv_length"]) in
                ({("decode",4,1,512),("prefill",1,128,128)} |
                 ({("decode",1,1,4096),("extend",4,8,4096)} if args.smoke_extended else set()))]
        groups=[[c for c in g if c["stage"]<=(9 if args.smoke_extended else 7)] for g in groups]
    if args.group_index is not None:groups=[groups[args.group_index]]
    if args.only_group:groups=[g for g in groups if g[0]["case_id"].rsplit("_s",1)[0]==args.only_group]
    started=time.time();counts={"graphs":0,"pairs":0,"correct_candidates":0,"incorrect_candidates":0,"failures":0}
    with args.output.open("x") as f:
        def emit(row):f.write(json.dumps(row,ensure_ascii=False)+"\n");f.flush()
        emit({"record_type":"run_metadata","framework":args.framework,"torch":torch.__version__,
              "gpu":torch.cuda.get_device_name(0),"visible_gpu":os.environ.get("CUDA_VISIBLE_DEVICES"),
              "args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              "primitive_origins":adapter.origins,"policy_scope":"controlled/hybrid graph; NOT unmodified engine default policy",
              "executor_code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "framework_versions":{name:importlib.metadata.version(name) for name in
                  ("torch","triton") if importlib.metadata.packages_distributions().get(name)},
              "manifest_sha256":hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "values":"synthetic seeded FP16, pinned real-model geometry and checked forward topology",
              "seed_contract":"weights depend only on model_id/layer/tensor; activations/cache on model/workload/tensor; stage, candidate and framework excluded",
              "timing":"CUDA event complete pipeline including producer/repairs/consumers; no sum of primitive medians; JIT/plan/reference outside timing",
              "near_set_selection":"training medians; independent paired AB/BA holdout only for selected free/frozen plans",
              "space":"four matmul outputs row/column per block plus NHD/HND KV; one-row aliases deduplicated; fixed arithmetic/backend/fusion",
              "tactic_scope":"Triton/CUTLASS fixed GEMM tile; torch/cuBLAS may internally change tactic with strides, so native library comparisons are layout+automatic-tactic, not a proven tactic-fixed ablation",
              "limitations":"dense Qwen3 only; native vLLM/SGLang HND is explicit repair strategy, not engine default; TVM/CUTLASS graph has other disclosed primitives; source PR replay is separate"})
        for index,cases in enumerate(groups):
            previous=None;group=None;finished_ids=set()
            try:
                group=Group(cases[0],adapter,max(c["stage"] for c in cases))
                emit({"record_type":"group_data_fingerprint","framework":args.framework,
                      "group":list(group_key(cases[0])),"fingerprint":group.data_fingerprint})
                print(f"[{args.framework}] group {index+1}/{len(groups)} {group.key}",flush=True)
                for case in cases:
                    axes=layout_axes(case);plans,scope=candidates(axes,previous,budget=args.candidate_budget,
                                                               seed=stable_seed(group.key,str(case["stage"])))
                    reference=independent_graph_reference(group,case);rows=[]
                    for bits in plans:
                        plan=group.plan(case,bits)
                        check=group.correctness(plan,reference)
                        counts["correct_candidates" if check["correct"] else "incorrect_candidates"]+=1
                        if not check["correct"]:
                            emit({"record_type":"incorrect_candidate","case_id":case["case_id"],"bits":list(bits),"correctness":check})
                            del plan;continue
                        samples=timed(lambda:group.run(plan),args.warmup,args.iterations)
                        rows.append({"bits":list(bits),"train_median_ms":statistics.median(samples),
                                     "train_samples_ms":samples,"correctness":check})
                        del plan
                    if not rows:raise RuntimeError("no correct candidate")
                    free=min(rows,key=lambda r:r["train_median_ms"])
                    if previous:
                        frozen=min((r for r in rows if frozen_eligible(r["bits"],axes,previous["axes"],previous["near_plans"])),
                                   key=lambda r:r["train_median_ms"])
                        free_plan=group.plan(case,free["bits"])
                        frozen_plan=group.plan(case,frozen["bits"])
                        # Check after repeated stateful execution as well.
                        holdout={"free_ms":[],"frozen_ms":[],"order":[]}
                        for rep in range(args.holdout_repetitions):
                            order=("free","frozen") if rep%2==0 else ("frozen","free")
                            for name in order:
                                pp=free_plan if name=="free" else frozen_plan
                                ss=timed(lambda pp=pp:group.run(pp),1,args.iterations)
                                holdout[name+"_ms"].append(statistics.median(ss))
                            holdout["order"].append(list(order))
                        holdout["free_post_correctness"]=group.correctness(free_plan,reference)
                        holdout["frozen_post_correctness"]=group.correctness(frozen_plan,reference)
                        if not holdout["free_post_correctness"]["correct"] or not holdout["frozen_post_correctness"]["correct"]:
                            raise RuntimeError("post timing/state correctness failed")
                        pair=summarize_pair(case,rows,previous,args.epsilon,holdout)
                        pair.update(record_type="growth_pair",framework=args.framework,search_scope=scope)
                        emit(pair);counts["pairs"]+=1
                        del free_plan,frozen_plan,pp
                    near=near_optimal(rows,args.epsilon)
                    emit({"record_type":"graph","framework":args.framework,"case":case,"axes":axes,
                          "executed_graph_status":"correctness_verified_complete_controlled_graph" if len(rows)==len(plans) else
                              "incomplete_incorrect_candidates_rejected",
                          "attempted_candidate_count":len(plans),
                          "search_scope":scope,"declared_binary_space_size":2**len(axes),
                          "measured_candidate_count":len(rows),"near_plan_count":len(near),
                          "near_plans":[list(x) for x in near],"candidates":rows,
                          "best_train_plan":free["bits"],"best_train_ms":free["train_median_ms"],
                          "source_pr_replay":False,"primitive_origins":adapter.origins})
                    previous={"axes":axes,"near_plans":near,"case_id":case["case_id"]}
                    counts["graphs"]+=1
                    finished_ids.add(case["case_id"])
                    print(f"[{args.framework}] {case['case_id']} candidates={len(rows)} scope={scope}",flush=True)
                    del reference;gc.collect()
            except Exception as error:
                counts["failures"]+=1
                emit({"record_type":"group_failure","framework":args.framework,"group":list(group_key(cases[0])),
                      "error":str(error),"traceback":traceback.format_exc()})
                print(traceback.format_exc(),flush=True)
                for remaining in cases:
                    if remaining["case_id"] not in finished_ids:
                        emit({"record_type":"not_completed_case","framework":args.framework,
                              "case_id":remaining["case_id"],"reason":"group_exception",
                              "error":str(error)})
            finally:
                del group;gc.collect();torch.cuda.empty_cache()
        emit({"record_type":"completion","framework":args.framework,"counts":counts,
              "elapsed_seconds":time.time()-started,"expected_graphs":sum(len(g) for g in groups)})
    return 1 if counts["failures"] or counts["incorrect_candidates"] or not counts["graphs"] else 0


if __name__=="__main__":raise SystemExit(main())
