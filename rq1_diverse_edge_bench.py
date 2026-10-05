#!/usr/bin/env python3
"""True producer-only and complete-edge measurements over diverse operators.

Triton is the controlled producer. Consumer identity is recorded separately.
Native vLLM/SGLang activation and FlashInfer attention are hybrid edges, never
relabeled as an unmodified end-to-end vLLM/SGLang baseline.
"""
from __future__ import annotations
import argparse
import gc
import inspect
import importlib.metadata as package_metadata
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from rq1_diverse_kernels import gemm, gather, rms_emit, paged_append
from rq1_flashinfer_workspace import plan_with_workspace,WorkspaceLimitError,INITIAL_BYTES,MAX_BYTES


class Unsupported(Exception):
    pass


def meta(value):
    if isinstance(value,(tuple,list)):
        return [meta(x) for x in value]
    return {"shape":list(value.shape),"stride":list(value.stride()),
            "dtype":str(value.dtype),"bytes":value.numel()*value.element_size()}


def effective_storage(value):
    """Extent-one strides do not create distinct tensor address mappings."""
    if isinstance(value,(tuple,list)):return tuple(effective_storage(x) for x in value)
    return (str(value.dtype),tuple((n,s) for n,s in zip(value.shape,value.stride()) if n>1))


def same_storage(a,b):
    if isinstance(a,(tuple,list)):
        return len(a)==len(b) and all(same_storage(x,y) for x,y in zip(a,b))
    return a.data_ptr()==b.data_ptr() and effective_storage(a)==effective_storage(b)


def clone(value):
    if isinstance(value,(tuple,list)):return tuple(clone(x) for x in value)
    return value.clone()


def compare(value, reference, atol=1e-5, rtol=.03):
    if isinstance(value,(tuple,list)):
        checks=[compare(a,b,atol,rtol) for a,b in zip(value,reference)]
        return {"correct":len(value)==len(reference) and all(c["correct"] for c in checks),"parts":checks}
    if value.shape!=reference.shape:return {"correct":False,"reason":"shape mismatch"}
    delta=(value.float()-reference.float()).abs()
    rms_ref=reference.float().square().mean().sqrt()
    # FP16 dot/reduction ordering can differ across strides. A scale-aware
    # absolute floor (4 FP16 unit roundoffs * RMS) avoids rejecting cancellation
    # near zero, while the independent NRMSE gate still rejects zero outputs.
    effective_atol=max(atol,4*(2**-10)*float(rms_ref))
    nrmse=float(delta.square().mean().sqrt()/rms_ref.clamp_min(1e-12))
    finite=bool(torch.isfinite(value).all())
    return {"correct":finite and bool(torch.allclose(value,reference,atol=effective_atol,rtol=rtol)) and nrmse<=.01,
            "max_abs_error":float(delta.max()),"absolute_floor":atol,"effective_atol":effective_atol,"rtol":rtol,
            "finite":finite,"normalized_rmse":nrmse,"normalized_rmse_limit":.01,
            "reference_rms":float(rms_ref)}


def gdn_reference(q,k,v,g,beta):
    """Independent FP32 gated delta recurrence, no native FLA called here."""
    b,t,h,dk=q.shape;vh=v.shape[2];dv=v.shape[-1]
    q=q.float().repeat_interleave(vh//h,dim=2)
    k=k.float().repeat_interleave(vh//h,dim=2);v=v.float()
    state=torch.zeros(b,vh,dk,dv,device=q.device,dtype=torch.float32)
    outputs=[]
    for i in range(t):
        state=state*g[:,i].float().exp()[:,:,None,None]
        prediction=torch.einsum("bhk,bhkv->bhv",k[:,i],state)
        delta=(v[:,i]-prediction)*beta[:,i].float()[:,:,None]
        state=state+k[:,i,:,:,None]*delta[:,:,None,:]
        outputs.append(torch.einsum("bhk,bhkv->bhv",q[:,i]/math.sqrt(dk),state))
    # Both native serving implementations store recurrent state as [B,H,V,K].
    return torch.stack(outputs,dim=1).half(),state.transpose(-1,-2).contiguous()


def attention_reference(q,k,v,mask=None,causal=False):
    """Independent FP32 QK / softmax / AV reference (not FlashInfer/FA)."""
    factor=q.shape[1]//k.shape[1]
    k=k.float().repeat_interleave(factor,dim=1)
    v=v.float().repeat_interleave(factor,dim=1)
    scores=(q.float()@k.transpose(-1,-2))/math.sqrt(q.shape[-1])
    if causal:
        mask=torch.arange(k.shape[-2],device=q.device)[None,:]<=torch.arange(q.shape[-2],device=q.device)[:,None]
    if mask is not None:scores=scores.masked_fill(~mask,float("-inf"))
    return (torch.softmax(scores,-1)@v).half()


def timing(fn,warmup,iterations):
    for _ in range(warmup):fn()
    torch.cuda.synchronize()
    samples=[]
    # Event creation outside the measured region; one event pair wraps the
    # actual complete pipeline. P/R/C p50s are diagnostic, not added together.
    events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
            for _ in range(iterations)]
    for a,b in events:
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    samples=sorted(float(a.elapsed_time(b)) for a,b in events)
    return {"p50_ms":samples[len(samples)//2],"samples_ms":samples}


def output_buffer(shape,layout):
    if len(shape)==2:
        m,n=shape
        if layout=="row_major":return torch.empty(shape,device="cuda",dtype=torch.float16)
        if layout=="column_major":return torch.empty(n,m,device="cuda",dtype=torch.float16).t()
        if layout=="padded_row":return torch.empty(m,n+16,device="cuda",dtype=torch.float16)[:,:n]
    g,m,n=shape
    if layout=="head_major":return torch.empty(shape,device="cuda",dtype=torch.float16)
    if layout=="token_major":return torch.empty(m,g,n,device="cuda",dtype=torch.float16).permute(1,0,2)
    if layout=="state_transposed":return torch.empty(g,n,m,device="cuda",dtype=torch.float16).transpose(1,2)
    if layout=="route_interleaved":return torch.empty(n,g,m,device="cuda",dtype=torch.float16).permute(1,2,0)
    raise ValueError(layout)


def make_plan(case,consumer_mode,workspace_initial_bytes=INITIAL_BYTES,workspace_max_bytes=MAX_BYTES):
    torch.manual_seed(case["tensor_seed"])
    d=case["dimensions"]; family=case["family"]
    b,q,kv=case["request_batch"],case["query_length"],case["kv_length"]
    t=b*q; h=int(d["hidden_size"])
    heads=int(d.get("num_attention_heads") or 1)
    kh=int(d.get("num_key_value_heads") or 1)
    width=int(d.get("head_dim") or 128)
    inter=int(d.get("moe_intermediate_size") or d.get("intermediate_size") or h)
    def rand(*shape):return torch.randn(shape,device="cuda",dtype=torch.float16)*.02
    plans=[]
    def add(layout,produce,boundary,logical,consume,repair=None,consumer="torch/cuBLAS"):
        plans.append(dict(layout=layout,produce=produce,boundary=boundary,logical=logical,
                          consume=consume,repair=repair,consumer=consumer))
    if family=="attention_bmm_oproj":
        # Value BMM (AV), followed by the real O-projection. A is identical
        # normalized probability data for both direct output layouts.
        a=torch.softmax(rand(b*heads,q,kv).float(),-1).half()
        v=rand(b*heads,kv,width); w=rand(heads*width,h)
        logical_ref=torch.bmm(a,v)
        def consumer(z):
            return z.reshape(b,heads,q,width).permute(0,2,1,3).contiguous().view(t,heads*width) @ w
        for layout in ("head_major","token_major"):
            # The token-major representation is exactly [B,q,H,D], not
            # [q,B*H,D], so B>1 does not introduce an unintended repair copy.
            out=(torch.empty(b,heads,q,width,device="cuda",dtype=torch.float16)
                 if layout=="head_major" else
                 torch.empty(b,q,heads,width,device="cuda",dtype=torch.float16).permute(0,2,1,3))
            add(layout,lambda o=out:gemm(a,v,o),out,lambda o=out:o.reshape(b*heads,q,width),
                consumer,lambda z:z.contiguous())
        return plans,logical_ref,consumer(logical_ref),"AV BMM → token/head flatten → O-Proj"
    if family in {"projection_sdpa","projection_gdn"}:
        if family=="projection_gdn":
            kh=int(d.get("linear_num_key_heads") or 1)
            vh=int(d.get("linear_num_value_heads") or kh)
            dk=int(d.get("linear_key_head_dim") or 128)
            dv=int(d.get("linear_value_head_dim") or dk)
            pieces=(kh*dk,kh*dk,vh*dv)
            try:
                if consumer_mode=="vllm":
                    from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule
                elif consumer_mode=="sglang":
                    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
                else:
                    raise ImportError("GDN requires a native vLLM/SGLang FLA consumer")
            except ImportError as error:
                raise Unsupported(str(error))
        else:
            pieces=(heads*width,kh*width,kh*width)
        x=rand(t,h);w=rand(h,sum(pieces)); ref=x@w
        def split(z):return torch.split(z,pieces,-1)
        if family=="projection_sdpa":
            def consume(z):
                qq,kk,vv=z
                qq=qq.view(b,q,heads,width).transpose(1,2)
                kk=kk.view(b,q,kh,width).transpose(1,2)
                vv=vv.view(b,q,kh,width).transpose(1,2)
                return F.scaled_dot_product_attention(qq,kk,vv,is_causal=q>1,enable_gqa=heads!=kh)
            cname="torch SDPA (real softmax/QK/AV)"
        else:
            beta=torch.full((b,q,vh),.5,device="cuda",dtype=torch.float16)
            gate=torch.full((b,q,vh),-.1,device="cuda",dtype=torch.float32)
            state_pool=torch.zeros(b,vh,dv,dk,device="cuda",dtype=torch.float32)
            state_indices=torch.arange(b,device="cuda",dtype=torch.int64)
            parameters=inspect.signature(chunk_gated_delta_rule).parameters
            def consume(z):
                qq,kk,vv=z
                qq=F.normalize(qq.view(b,q,kh,dk).float(),dim=-1).half()
                kk=F.normalize(kk.view(b,q,kh,dk).float(),dim=-1).half()
                kwargs={"g":gate,"beta":beta}
                if "output_final_state" in parameters:kwargs["output_final_state"]=True
                if "head_first" in parameters:
                    kwargs["head_first"]=False
                if "initial_state_indices" in parameters:
                    # SGLang returns chunk-initial states in h, not final state.
                    # The actual final state is written into its state pool.
                    # Reset on every call, so timing repetitions are identical
                    # independent sequences rather than evolving state.
                    state_pool.zero_()
                    kwargs.update(initial_state=state_pool,initial_state_indices=state_indices,inplace_update=True)
                result=chunk_gated_delta_rule(qq,kk,vv.view(b,q,vh,dv),**kwargs)
                if "initial_state_indices" in parameters:return result[0],state_pool
                return result
            cname=consumer_mode+" FLA chunk_gated_delta_rule"
        for layout in ("packed_row","dense_separated"):
            raw=torch.empty(t,sum(pieces),device="cuda",dtype=torch.float16)
            if layout=="packed_row":
                boundary=split(raw);producer=lambda o=raw:gemm(x,w,o)
            else:
                flat=raw.view(-1);prefix=0;parts=[]
                for n in pieces:
                    parts.append(flat[prefix*t:(prefix+n)*t].view(t,n));prefix+=n
                boundary=tuple(parts);producer=lambda o=raw:gemm(x,w,o,segments=pieces)
            add(layout,producer,boundary,lambda z=boundary:torch.cat(z,-1),consume,
                lambda z:tuple(a.contiguous() for a in z),cname)
        if family=="projection_gdn":
            qr,kr,vr=split(ref)
            qr=F.normalize(qr.view(b,q,kh,dk).float(),dim=-1).half()
            kr=F.normalize(kr.view(b,q,kh,dk).float(),dim=-1).half()
            edge_ref=gdn_reference(qr,kr,vr.view(b,q,vh,dv),gate,beta)
        else:
            qr,kr,vr=split(ref)
            edge_ref=attention_reference(qr.view(b,q,heads,width).transpose(1,2),
                kr.view(b,q,kh,width).transpose(1,2),vr.view(b,q,kh,width).transpose(1,2),causal=q>1)
        return plans,ref,edge_ref,"QKV projection → strided/split operands → "+cname
    if family in {"sparse_gather_sdpa","host_kv_sdpa"}:
        if family=="sparse_gather_sdpa":
            heads=int(d.get("index_n_heads") or d.get("index_heads") or heads)
            width=int(d.get("index_head_dim") or width)
            kh=1
        selected=min(kv,max(16,q*2)); channels=kh*width
        k=rand(b*kv,channels);v=rand(b,selected,kh,width).transpose(1,2)
        ids=torch.cat([torch.randperm(kv,device="cuda")[:selected]+request*kv for request in range(b)])
        qq=rand(b,heads,q,width);ref=k[ids]
        def consume(z):
            kk=z.view(b,selected,kh,width).transpose(1,2)
            return F.scaled_dot_product_attention(qq,kk,v,enable_gqa=heads!=kh)
        if family=="host_kv_sdpa":
            host=ref.cpu().pin_memory()
            for layout in ("row_major","column_major"):
                out=output_buffer(ref.shape,layout)
                add(layout,lambda o=out:o.copy_(host,non_blocking=True),out,lambda o=out:o,
                    consume,lambda z:z.contiguous(),"torch SDPA + pinned-host copy")
        else:
            for layout in ("row_major","column_major","padded_row"):
                out=output_buffer(ref.shape,layout)
                add(layout,lambda o=out:gather(k,ids,o),out,lambda o=out:o,consume,
                    lambda z:z.contiguous(),"torch SDPA (compact selected-token attention)")
        edge_ref=attention_reference(qq,ref.view(b,selected,kh,width).transpose(1,2),v)
        return plans,ref,edge_ref,"selected-token gather/restore → real selected-KV softmax attention"
    if family=="projection_swiglu_down":
        x=rand(t,h);w=rand(h,2*inter);down=rand(inter,h);ref=x@w
        def consume(z):
            if consumer_mode=="vllm":
                import vllm._custom_ops  # Registers version-specific _C libraries.
                result=torch.empty(t,inter,device="cuda",dtype=torch.float16)
                torch.ops._C.silu_and_mul(result,z)
            elif consumer_mode=="sglang":
                import sgl_kernel
                result=sgl_kernel.silu_and_mul(z)
            else: result=F.silu(z[:,:inter])*z[:,inter:]
            return result@down
        for layout in ("row_major","column_major","padded_row"):
            out=output_buffer(ref.shape,layout)
            # Native kernels have a dense last-axis contract; enumerate repair
            # rather than feeding them arbitrary illegal strides.
            add(layout,lambda o=out:gemm(x,w,o),out,lambda o=out:o,consume,
                lambda z:z.contiguous(),consumer_mode+" SwiGLU + cuBLAS down projection")
            plans[-1]["identity_legal"]=(consumer_mode=="torch" or layout=="row_major")
        reference=(F.silu(ref[:,:inter])*ref[:,inter:])@down
        return plans,ref,reference,"gate/up GEMM → SwiGLU → down GEMM"
    if family=="state_prefill_decode":
        kh=int(d.get("linear_num_key_heads") or 1)
        vh=int(d.get("linear_num_value_heads") or kh)
        dk=int(d.get("linear_key_head_dim") or 128);dv=int(d.get("linear_value_head_dim") or 128)
        k=rand(b*vh,q,dk);v=rand(b*vh,q,dv);kt=k.transpose(1,2)
        ref=torch.bmm(kt,v);qd=rand(b*vh,1,dk);kd=rand(b*vh,1,dk);vd=rand(b*vh,1,dv)
        horizon=case["decode_horizon"]
        def consume(z):
            # A deterministic linear-attention recurrent contraction. This is
            # explicitly NOT KDA/GDN: no delta-rule/checkpoint native claim.
            state=z;outputs=[]
            for _ in range(horizon):
                state=state*.99+torch.bmm(kd.transpose(1,2),vd)
                outputs.append(torch.bmm(qd,state))
            return torch.cat(outputs,1)
        for layout in ("head_major","state_transposed"):
            out=output_buffer(ref.shape,layout)
            add(layout,lambda o=out:gemm(kt,v,o),out,lambda o=out:o,consume,
                lambda z:z.contiguous(),"torch linear-state recurrent contraction (not KDA/GDN)")
        return plans,ref,consume(ref),"KᵀV state accumulation → repeated state update/read"
    if family=="moe_dispatch_expert":
        experts=int(d.get("num_experts") or 1);topk=min(experts,int(d.get("experts_per_token") or 1))
        x=rand(t,h); ids=torch.arange(t*topk,device="cuda")%experts
        source=torch.arange(t,device="cuda").repeat_interleave(topk)
        order=torch.argsort(ids,stable=True);undo=torch.argsort(order)
        active=sorted(set(ids.cpu().tolist()))
        weights={e:rand(h,inter) for e in active}
        sortedids=ids[order].cpu().tolist()
        ranges=[(e,sortedids.index(e),sortedids.count(e)) for e in active]
        ref=x[source]
        def consume(z):return torch.cat([z[start:start+count]@weights[e] for e,start,count in ranges],0)[undo]
        for layout in ("route_major","expert_major"):
            out=output_buffer(ref.shape,"row_major")
            index=source if layout=="route_major" else source[order]
            add(layout,lambda o=out,ix=index:gather(x,ix,o),out,
                (lambda o=out:o) if layout=="route_major" else (lambda o=out:o[undo]),
                consume,(lambda z:z[order]) if layout=="route_major" else (lambda z:z),
                "torch grouped expert GEMMs (local dispatch; no DeepEP communication)")
            plans[-1]["identity_legal"]=layout=="expert_major"
        return plans,ref,consume(ref[order]),"routing-fixed local dispatch → grouped expert GEMM"
    if family=="moe_gemm_combine":
        topk=int(d.get("experts_per_token") or 1);routes=t*topk
        experts=int(d.get("num_experts") or 1)
        active=min(experts,routes)
        expert_weights=rand(active,inter,h)
        route_expert=torch.arange(routes,device="cuda")%active
        a=rand(routes,1,inter);w=expert_weights[route_expert];ref=torch.bmm(a,w)
        scales=torch.softmax(rand(t,topk).float(),-1).half()
        def consume(z):return (z.view(t,topk,h)*scales.unsqueeze(-1)).sum(1)
        for layout in ("head_major","route_interleaved"):
            out=output_buffer(ref.shape,layout)
            add(layout,lambda o=out:gemm(a,w,o),out,lambda o=out:o,consume,lambda z:z.contiguous(),
                "torch weighted local MoE combine (no peer-GPU placement)")
        return plans,ref,consume(ref),"expert GEMM2 → weighted route-slot local combine"
    if family in {"gemm_bias_gemm","reduce_norm_gemm","host_weight_gemm"}:
        if family=="host_weight_gemm":
            x=rand(t,h);host=rand(h,inter).cpu().pin_memory();ref=host.cuda()
            def consume(z):return x@z
            for layout in ("row_major","column_major"):
                out=output_buffer(ref.shape,layout)
                add(layout,lambda o=out:o.copy_(host,non_blocking=True),out,lambda o=out:o,
                    consume,lambda z:z.contiguous(),"torch/cuBLAS host-weight GEMM (FP16; not Marlin/NVFP4)")
            return plans,ref,consume(ref),"pinned host weight fill → GPU GEMM"
        x=rand(t,h);w=rand(h,inter);down=rand(inter,h);bias=rand(inter)
        if family=="gemm_bias_gemm":
            ref=x@w
            consume=lambda z:(z+bias)@down
        else:
            ref=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).half()
            consume=lambda z:z@w
        for layout in ("row_major","column_major","padded_row"):
            out=output_buffer(ref.shape,layout)
            produce=(lambda o=out:gemm(x,w,o)) if family=="gemm_bias_gemm" else (lambda o=out:rms_emit(x,o))
            add(layout,produce,out,lambda o=out:o,consume,lambda z:z.contiguous())
        return plans,ref,consume(ref),family.replace("_"," → ")
    if family in {"kv_writer_flashinfer","kv_writer_swa_flashinfer"}:
        try:
            import flashinfer
        except ImportError as error:raise Unsupported(str(error))
        page=int(case.get("page_size",16)); np=math.ceil(kv/page)
        pages=torch.randperm(b*np,device="cuda",dtype=torch.int64).to(torch.int32)
        indptr=torch.arange(b+1,device="cuda",dtype=torch.int32)*np
        last=torch.full((b,),((kv-1)%page)+1,device="cuda",dtype=torch.int32)
        qo=torch.arange(b+1,device="cuda",dtype=torch.int32)*q
        k=rand(b,kv,kh,width);v=rand(b,kv,kh,width);qq=rand(b*q,heads,width)
        mask=torch.arange(kv,device="cuda")[None,:]<=kv-q+torch.arange(q,device="cuda")[:,None]
        window_left=-1
        if family=="kv_writer_swa_flashinfer":
            window_left=int(d.get("sliding_window") or 0)-1
            if window_left<0:raise Unsupported("model config has no concrete sliding_window")
            mask=mask & (torch.arange(kv,device="cuda")[None,:]>=
                         kv-q+torch.arange(q,device="cuda")[:,None]-window_left)
        refout=attention_reference(qq.view(b,q,heads,width).transpose(1,2),
            k.transpose(1,2),v.transpose(1,2),mask=mask)
        refout=refout.transpose(1,2).reshape(b*q,heads,width)
        def allocate_workspace(size):return torch.zeros(size,device="cuda",dtype=torch.uint8)
        workspace=allocate_workspace(workspace_initial_bytes)
        for layout in ("NHD","HND"):
            cache_shape=(b*np,page,kh,width) if layout=="NHD" else (b*np,kh,page,width)
            kc=torch.zeros(cache_shape,device="cuda",dtype=torch.float16);vc=torch.zeros_like(kc)
            # Initialize historical tokens outside all measured regions. P
            # writes only the newly appended q tokens per request, not full KV.
            paged_append(k,kc,pages,kv,page,layout=="HND")
            paged_append(v,vc,pages,kv,page,layout=="HND")
            if q==1:
                factory=lambda scratch:flashinfer.BatchDecodeWithPagedKVCacheWrapper(scratch,kv_layout=layout,
                    use_tensor_cores=True,backend="fa2")
                plan_native=lambda w:w.plan(indptr,pages,last,heads,kh,width,page,
                    q_data_type=torch.float16,kv_data_type=torch.float16,window_left=window_left)
            else:
                factory=lambda scratch:flashinfer.BatchPrefillWithPagedKVCacheWrapper(scratch,kv_layout=layout,backend="fa2")
                plan_native=lambda w:w.plan(qo,indptr,pages,last,heads,kh,width,page,causal=True,
                    q_data_type=torch.float16,kv_data_type=torch.float16,window_left=window_left)
            wrapper,workspace_meta=plan_with_workspace(factory,plan_native,allocate_workspace,workspace,
                workspace_initial_bytes,workspace_max_bytes)
            def produce(kk=kc,vv=vc,hnd=layout=="HND"):
                paged_append(k,kk,pages,min(q,kv),page,hnd)
                paged_append(v,vv,pages,min(q,kv),page,hnd)
            def logical(kk=kc,vv=vc,hnd=layout=="HND"):
                def unpack(z):
                    if hnd:z=z.permute(0,2,1,3)
                    return z[pages.long()].reshape(b,np*page,kh,width)[:,:kv]
                return unpack(kk),unpack(vv)
            add("paged_"+layout,produce,(kc,vc),logical,
                lambda z,w=wrapper:w.run(qq,z),None,
                "FlashInfer fa2 paged "+("decode" if q==1 else "prefill")+" "+layout)
            plans[-1]["consumer_workspace"]=workspace_meta
        return plans,(k,v),refout,"direct append-only paged KV writer → native FlashInfer real attention"
    raise Unsupported(f"unimplemented family {family}")


def estimate_bytes(case):
    d=case["dimensions"];h=int(d["hidden_size"]);t=case["request_batch"]*case["query_length"]
    i=int(d.get("moe_intermediate_size") or d.get("intermediate_size") or h)
    if case["family"]=="moe_gemm_combine":
        routes=t*int(d.get("experts_per_token") or 1)
        return 2*(routes+min(routes,int(d.get("num_experts") or 1)))*i*h+32*t*h
    if case["family"]=="moe_dispatch_expert":
        active=min(int(d.get("num_experts") or 1),t*int(d.get("experts_per_token") or 1))
        return 2*active*h*i+32*t*h
    return 12*h*i+64*t*(h+i)+16*case["request_batch"]*int(d.get("num_attention_heads") or 1)*case["query_length"]*case["kv_length"]


def run(case,args):
    base={"case_id":case["case_id"],"family":case["family"],"case":case,
          "producer_framework":"Triton" if not case["family"].startswith("host_") else "PyTorch copy",
          "consumer_mode":args.consumer_mode,"process_repetition":args.repetition,
          "evidence_level":"controlled_operator_edge_hybrid",
          "native_auto_baseline":None,"native_pr_replay":False,
          "torch_version":torch.__version__,"device":torch.cuda.get_device_name(),
          "installed_package_versions":args.package_versions,
          "timing_method":"CUDA event wraps actual full pipeline; no sum of medians"}
    if estimate_bytes(case)>args.max_bytes:
        return [{**base,"status":"oom_preflight","estimated_bytes":estimate_bytes(case)}]
    plans,pref,eref,description=make_plan(case,args.consumer_mode,args.workspace_initial_bytes,args.workspace_max_bytes)
    distinct={}
    for plan in plans:
        # Route reordering changes tensor values before logical undo, so retain
        # that algorithmic representation even when tensor strides are equal.
        key=(effective_storage(plan["boundary"]),plan["layout"] if case["family"]=="moe_dispatch_expert" else "")
        if key in distinct:
            distinct[key]["equivalent_layout_aliases"].append(plan["layout"])
        else:
            plan["equivalent_layout_aliases"]=[plan["layout"]];distinct[key]=plan
    plans=list(distinct.values())
    for plan in plans:
        plan["produce"]()
    torch.cuda.synchronize()
    rows=[]
    tasks=[]
    for plan in plans:
        pcheck=compare(plan["logical"](),pref)
        row={**base,"edge_description":description,"producer_layout":plan["layout"],
             "consumer_kernel":plan["consumer"],"boundary":meta(plan["boundary"]),
             "equivalent_layout_aliases":plan["equivalent_layout_aliases"],
             "consumer_workspace":plan.get("consumer_workspace"),
             "stage":"producer","strategy":plan["layout"],"correctness":pcheck}
        if not pcheck["correct"]:
            rows.append({**row,"status":"numerical_mismatch"});continue
        tasks.append((row,plan["produce"]))
        repair_noop=(plan.get("identity_legal",True) and plan["repair"] is not None and
                     same_storage(plan["boundary"],plan["repair"](plan["boundary"])))
        for repair_name in ("identity","materialize"):
            if repair_name=="materialize" and repair_noop:continue
            if repair_name=="identity" and not plan.get("identity_legal",True):
                rows.append({**row,"stage":"edge","repair":repair_name,
                             "status":"illegal_consumer_contract"});continue
            repair=(lambda z:z) if repair_name=="identity" else plan["repair"]
            if repair is None:continue
            # clone() may densify padded/non-overlapping strided tensors and
            # silently change the C-only contract. P writes deterministic data;
            # retain the actual repaired view/buffer for C-only timing.
            ready=repair(plan["boundary"])
            def consume(p=plan,z=ready):return p["consume"](z)
            def edge(p=plan,r=repair):
                p["produce"]()
                return p["consume"](r(p["boundary"]))
            # Each complete path is executed and checked independently.
            echeck=compare(edge(),eref)
            descriptor={**row,"repair":repair_name,
                "strategy":plan["layout"]+"/"+repair_name,
                "correctness":echeck,"consumer_input":meta(ready),
                "equivalent_repair_aliases":["identity","materialize"] if repair_noop else [repair_name],
                "repair_is_view_or_identity":same_storage(plan["boundary"],ready)}
            if not echeck["correct"]:
                rows.append({**descriptor,"stage":"edge","status":"numerical_mismatch"});continue
            tasks.extend([({**descriptor,"stage":"consumer"},consume),
                          ({**descriptor,"stage":"edge"},edge)])
            if repair_name=="materialize":
                tasks.append(({**descriptor,"stage":"repair"},lambda p=plan,r=repair:r(p["boundary"])))
    random.Random(case["tensor_seed"]+args.repetition).shuffle(tasks)
    for descriptor,fn in tasks:
        row={**descriptor,"status":"success",**timing(fn,args.warmup,args.iterations)}
        rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--consumer-mode",choices=["torch","vllm","sglang"],default="torch")
    p.add_argument("--family",action="append");p.add_argument("--case-id")
    p.add_argument("--case-ids-file",type=Path,help="Exact selection; unknown/empty IDs fail closed")
    p.add_argument("--workspace-initial-bytes",type=int,default=int(os.environ.get("RQ1_FLASHINFER_WORKSPACE_BYTES",INITIAL_BYTES)))
    p.add_argument("--workspace-max-bytes",type=int,default=int(os.environ.get("RQ1_FLASHINFER_MAX_WORKSPACE_BYTES",MAX_BYTES)))
    p.add_argument("--repetition",type=int,default=0)
    p.add_argument("--warmup",type=int,default=8);p.add_argument("--iterations",type=int,default=30)
    p.add_argument("--max-bytes",type=int,default=2**31)
    p.add_argument("--physical-device-index",type=int,default=1)
    args=p.parse_args()
    if not 0<args.workspace_initial_bytes<=args.workspace_max_bytes:p.error("invalid workspace limits")
    args.package_versions={}
    for name in ("torch","triton","vllm","sglang","flashinfer-python","sgl-kernel"):
        try:args.package_versions[name]=package_metadata.version(name)
        except package_metadata.PackageNotFoundError:args.package_versions[name]=None
    if args.output.exists():p.error(f"refusing to overwrite {args.output}")
    if args.warmup<1 or args.iterations<3:p.error("warmup>=1, iterations>=3 required")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=str(args.physical_device_index):
        p.error("CUDA_VISIBLE_DEVICES must select the assigned physical GPU alone")
    if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    torch.backends.cuda.matmul.allow_tf32=False
    cases=json.loads(args.manifest.read_text())["cases"]
    if args.case_ids_file:
        selection=json.loads(args.case_ids_file.read_text())
        ids=selection["case_ids"] if isinstance(selection,dict) else selection
        known={c["case_id"] for c in cases}
        if not ids or set(ids)-known:p.error(f"empty/unknown case selection: {sorted(set(ids)-known)}")
        selected=set(ids);cases=[c for c in cases if c["case_id"] in selected]
    if args.family:cases=[c for c in cases if c["family"] in args.family]
    if args.case_id:cases=[c for c in cases if c["case_id"]==args.case_id]
    if not cases:p.error("selection matched no cases")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    failure_count=0
    with torch.inference_mode(),args.output.open("x",encoding="utf-8") as f:
        for idx,case in enumerate(cases):
            try:rows=run(case,args)
            except Exception as error:
                rows=[{"case_id":case["case_id"],"family":case["family"],"case":case,
                       "consumer_mode":args.consumer_mode,"process_repetition":args.repetition,
                       "status":("unsupported_consumer" if isinstance(error,Unsupported) else
                                 "workspace_limit_exceeded" if isinstance(error,WorkspaceLimitError) else "error"),
                       "error":f"{type(error).__name__}: {error}","native_pr_replay":False}]
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+"\n")
            failure_count+=sum(r["status"] not in {"success","illegal_consumer_contract"} for r in rows)
            f.flush()
            print(f"[{idx+1}/{len(cases)}] {case['family']} {case['case_id']} "+
                  str({r['status'] for r in rows}),flush=True)
            gc.collect();torch.cuda.empty_cache()
    return 1 if failure_count else 0


if __name__=="__main__":raise SystemExit(main())
