#!/usr/bin/env python3
"""Run model-free, framework-native layout slices for real LLM subgraphs.

This runner deliberately does *not* call a generic PyTorch implementation and
label it vLLM/SGLang.  Every successful row calls an operator shipped by the
selected framework.  It also deliberately labels the evidence
``native_operator_slice``: a KV writer, router, activation, or state update is
not an end-to-end replay of a model architecture.

The manifest is nevertheless consumed case-for-case.  Thus unsupported or
unsafe cases remain visible instead of silently disappearing from coverage.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from pathlib import Path


ATTENTION = {"gqa", "sliding_attention", "sparse_attention", "mla"}
RELATED_RQS = {
    "gqa": ["L-RQ1", "L-RQ2", "L-RQ4", "L-RQ6", "L-RQ9"],
    "sliding_attention": ["L-RQ1", "L-RQ2", "L-RQ4", "L-RQ6", "L-RQ9"],
    "sparse_attention": ["L-RQ1", "L-RQ2", "L-RQ4", "L-RQ6", "L-RQ9"],
    "mla": ["L-RQ1", "L-RQ2", "L-RQ4", "L-RQ6", "L-RQ9"],
    "swiglu": ["L-RQ1", "L-RQ2", "L-RQ4", "L-RQ9"],
    "moe": ["L-RQ1", "L-RQ2", "L-RQ4", "L-RQ9"],
    "mamba2": ["L-RQ1", "L-RQ2", "L-RQ6", "L-RQ9"],
    "linear_attention": ["L-RQ1", "L-RQ2", "L-RQ6", "L-RQ9"],
}


def attention_family(case: dict) -> str | None:
    structure, shape = case["structure"], case["shape"]
    if structure in {"gqa", "sliding_attention"}:
        qh, kh = int(shape["num_query_heads"]), int(shape["num_kv_heads"])
        kind = "MQA" if kh == 1 else "MHA" if qh == kh else "GQA"
        return f"sliding_{kind}" if structure == "sliding_attention" else kind
    if structure == "sparse_attention":
        return "sparse_attention"
    if structure == "mla":
        return "MLA"
    return None


def load_cases(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value["cases"] if isinstance(value, dict) else value


def percentile(samples: list[float]) -> float:
    return statistics.median(sorted(samples))


def time_cuda(torch, fn, warmup: int, iterations: int) -> tuple[float, list[float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return percentile(samples), samples


def attention_dimensions(case: dict, include_batch: bool = False) -> tuple[int, int, int, int]:
    """Return tokens, heads, K width, V width for the framework cache writer."""
    shape, dims = case["shape"], case["dimensions"]
    tokens = int(shape["query_length"])
    if case["phase"] == "decode":
        tokens = 1  # one new K/V row is appended in decode
    if include_batch:
        tokens *= int(shape.get("batch", 1))
    if case["structure"] in {"gqa", "sliding_attention"}:
        return tokens, int(shape["num_kv_heads"]), int(shape["head_dim"]), int(shape["head_dim"])
    if case["structure"] == "sparse_attention":
        # The layout-sensitive object is the index K cache, not dense Q heads.
        return tokens, 1, int(shape.get("index_dim", dims.get("index_head_dim") or 128)), int(shape.get("value_head_dim") or 128)
    # MLA stores one compressed latent vector plus RoPE dimensions per token.
    latent = int(dims.get("kv_lora_rank") or shape.get("latent_cache_dim") or 512)
    rope = int(dims.get("qk_rope_head_dim") or 0)
    value = int(dims.get("value_head_dim") or latent)
    return tokens, 1, latent + rope, value


def padded_tokens(case: dict) -> int:
    shape = case["shape"]
    per_request = int(shape["query_length"] if case["phase"] == "prefill" else 1)
    return int(shape.get("batch", 1)) * per_request


def tensor_meta(tensor) -> dict:
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype), "contiguous": tensor.is_contiguous()}


def run_vllm(case: dict, warmup: int, iterations: int) -> list[dict]:
    import torch
    import vllm._custom_ops  # registers torch.ops._C and _C_cache_ops

    structure = case["structure"]
    device, dtype = "cuda", torch.float16
    torch.manual_seed(int(case.get("tensor_seed", 0)))
    rows = []
    if structure in ATTENTION:
        from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
            triton_reshape_and_cache_flash,
            triton_reshape_and_cache_flash_diffkv,
        )
        tokens, heads, kdim, vdim = attention_dimensions(case, include_batch=True)
        key = torch.randn(tokens, heads, kdim, device=device, dtype=dtype)
        value = torch.randn(tokens, heads, vdim, device=device, dtype=dtype)
        block = 16
        blocks = (tokens + block - 1) // block
        slot = torch.arange(tokens, device=device, dtype=torch.int64)
        scale = torch.ones(1, device=device, dtype=torch.float32)
        # This public direct writer documents a 4D NHD cache.  Its source only
        # enters the head-major branch for a 5D engine cache; manufacturing a
        # 4D HND-strided view can corrupt adjacent allocations on sm_86.  HND
        # remains covered by the separate full-engine variant, not this slice.
        for layout in ("NHD",):
            if kdim != vdim:
                if layout == "NHD":
                    combined = torch.empty(blocks, block, heads, kdim + vdim,
                                           device=device, dtype=dtype)
                else:
                    combined = torch.empty(blocks, heads, block, kdim + vdim,
                                           device=device, dtype=dtype).permute(0, 2, 1, 3)
                fn = lambda: triton_reshape_and_cache_flash_diffkv(
                    key, value, combined, slot, "auto", scale, scale)
                state = combined
                operator = "triton_reshape_and_cache_flash_diffkv"
            else:
                if layout == "NHD":
                    kc = torch.empty(blocks, block, heads, kdim, device=device, dtype=dtype)
                    vc = torch.empty(blocks, block, heads, vdim, device=device, dtype=dtype)
                else:
                    kc = torch.empty(blocks, heads, block, kdim, device=device, dtype=dtype).permute(0, 2, 1, 3)
                    vc = torch.empty(blocks, heads, block, vdim, device=device, dtype=dtype).permute(0, 2, 1, 3)
                fn = lambda: triton_reshape_and_cache_flash(key, value, kc, vc, slot,
                                                             "auto", scale, scale)
                state = kc
                operator = "triton_reshape_and_cache_flash"
            p50, samples = time_cuda(torch, fn, warmup, iterations)
            # Check a deterministic slot after timing.  This catches kernels
            # that accept a stride but write as though the tensor were packed.
            if kdim != vdim:
                ok = (torch.allclose(combined[0, 0, :, :kdim], key[0]) and
                      torch.allclose(combined[0, 0, :, kdim:], value[0]))
            else:
                ok = torch.allclose(kc[0, 0], key[0]) and torch.allclose(vc[0, 0], value[0])
            rows.append({"native_operator": operator,
                         "variant": layout, "selected_layout": layout,
                         "p50_ms": p50, "samples_ms": samples,
                         "numerically_correct": bool(ok),
                         "input": tensor_meta(key), "state": tensor_meta(state)})
        rows.append({"status": "engine_only_not_direct_operator",
                     "native_operator": "vLLM engine attention backend",
                     "variant": "HND", "selected_layout": "HND",
                     "reason": "direct 4D writer is NHD; HND requires the engine-owned 5D cache contract and is tested only by native_offline_layout_bench"})
    elif structure == "swiglu":
        tokens = padded_tokens(case)
        width = int(case["shape"]["intermediate_size"])
        x = torch.randn(tokens, 2 * width, device=device, dtype=dtype)
        out = torch.empty(tokens, width, device=device, dtype=dtype)
        fn = lambda: torch.ops._C.silu_and_mul(out, x)
        p50, samples = time_cuda(torch, fn, warmup, iterations)
        ref = torch.nn.functional.silu(x[:, :width]) * x[:, width:]
        rows.append({"native_operator": "torch.ops._C.silu_and_mul",
                     "variant": "packed_gate_up", "selected_layout": "[token,2I]",
                     "p50_ms": p50, "samples_ms": samples,
                     "numerically_correct": bool(torch.allclose(out, ref, rtol=2e-3, atol=2e-3)),
                     "input": tensor_meta(x), "state": None})
    elif structure == "moe":
        from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
        tokens = padded_tokens(case)
        hidden = int(case["shape"]["hidden_size"])
        experts = max(1, int(case["shape"]["num_experts"]))
        topk = min(experts, max(1, int(case["shape"]["experts_per_token"])))
        x = torch.randn(tokens, hidden, device=device, dtype=dtype)
        logits = torch.randn(tokens, experts, device=device, dtype=torch.float32)
        fn = lambda: fused_topk(x, logits, topk, True)
        p50, samples = time_cuda(torch, fn, warmup, iterations)
        weights, ids, _ = fn()
        rows.append({"native_operator": "vllm.fused_topk", "variant": "token_expert",
                     "selected_layout": "logits[token,expert]", "p50_ms": p50,
                     "samples_ms": samples,
                     "numerically_correct": bool(torch.isfinite(weights).all() and
                                                   (ids >= 0).all() and (ids < experts).all()),
                     "input": tensor_meta(logits), "state": tensor_meta(ids)})
    elif structure == "mamba2":
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_fn, causal_conv1d_update,
        )
        tokens = padded_tokens(case)
        dim = int(case["shape"]["hidden_size"])
        width = int(case["dimensions"].get("conv_kernel") or 4)
        weight = torch.full((dim, width), 0.01, device=device, dtype=dtype)
        # Use the model activation dtype.  Although the wrapper accepts fp32
        # state, the sm_86 path intermittently returned an uninitialised first
        # sample for that mixed-dtype combination; actual serving config keeps
        # the causal-conv cache in the configured Mamba state dtype.
        batch = int(case["shape"].get("batch", 1))
        per_request_tokens = int(case["shape"]["query_length"])
        state = torch.zeros(batch, dim, width - 1, device=device, dtype=dtype)
        indices = torch.arange(batch, device=device, dtype=torch.int32)
        if case["phase"] == "decode":
            x = torch.full((batch, dim), 0.01, device=device, dtype=dtype)
            out_buffer = torch.empty_like(x)
            def fn():
                state.zero_()
                return causal_conv1d_update(
                    x, state, weight, activation="silu",
                    conv_state_indices=indices, out=out_buffer)
            operator, layout = "vllm.causal_conv1d_update", "x[batch,dim]"
        else:
            x = torch.full((dim, tokens), 0.01, device=device, dtype=dtype)
            starts = torch.arange(0, tokens + 1, per_request_tokens,
                                  device=device, dtype=torch.int32)
            initial = torch.zeros(batch, device=device, dtype=torch.bool)
            def fn():
                # The kernel updates recurrent state in place. Reset it so every
                # sample represents the same input/state contract.
                state.zero_()
                return causal_conv1d_fn(x, weight, None, state, starts, indices,
                                        initial, activation="silu")
            operator, layout = "vllm.causal_conv1d_fn", "x[dim,time]"
        p50, samples = time_cuda(torch, fn, warmup, iterations)
        out = fn()
        rows.append({"native_operator": operator,
                     "variant": "decode_update" if case["phase"] == "decode" else "dimension_major",
                     "selected_layout": layout,
                     "p50_ms": p50, "samples_ms": samples,
                     "numerically_correct": bool(torch.isfinite(out).all()),
                     "input": tensor_meta(x), "state": tensor_meta(state)})
    else:
        rows.append({"status": "source_supported_runtime_not_safe",
                     "reason": "linear-attention FLA JIT probe did not terminate cleanly on sm_86; isolated source evidence retained",
                     "native_operator": "vllm.third_party.flash_linear_attention.chunk_gated_delta_rule"})
    return rows


def run_sglang(case: dict, warmup: int, iterations: int) -> list[dict]:
    import torch
    import sgl_kernel

    structure = case["structure"]
    device, dtype = "cuda", torch.float16
    torch.manual_seed(int(case.get("tensor_seed", 0)))
    rows = []
    if structure in ATTENTION:
        from sglang.kernels.ops.kvcache.kvcache import store_cache
        tokens, heads, kdim, vdim = attention_dimensions(case, include_batch=True)
        k = torch.randn(tokens, heads * kdim, device=device, dtype=dtype)
        v = torch.randn(tokens, heads * vdim, device=device, dtype=dtype)
        # SGLang's native pool is slot-major NHD in this path.
        kc = torch.empty(tokens + 1, heads * kdim, device=device, dtype=dtype)
        vc = torch.empty(tokens + 1, heads * vdim, device=device, dtype=dtype)
        indices = torch.arange(1, tokens + 1, device=device, dtype=torch.int64)
        fn = lambda: store_cache(k, v, kc, vc, indices, reserved_skip_index=-1)
        p50, samples = time_cuda(torch, fn, warmup, iterations)
        ok = torch.allclose(kc[1:tokens + 1], k) and torch.allclose(vc[1:tokens + 1], v)
        rows.append({"native_operator": "sglang.kvcache.store_cache",
                     "variant": "NHD_slot_major", "selected_layout": "cache[slot,H*D]",
                     "p50_ms": p50, "samples_ms": samples,
                     "numerically_correct": bool(ok),
                     "input": tensor_meta(k), "state": tensor_meta(kc)})
    elif structure == "swiglu":
        tokens = padded_tokens(case)
        width = int(case["shape"]["intermediate_size"])
        x = torch.randn(tokens, 2 * width, device=device, dtype=dtype)
        fn = lambda: sgl_kernel.silu_and_mul(x)
        p50, samples = time_cuda(torch, fn, warmup, iterations)
        out = fn()
        ref = torch.nn.functional.silu(x[:, :width]) * x[:, width:]
        rows.append({"native_operator": "sgl_kernel.silu_and_mul",
                     "variant": "packed_gate_up", "selected_layout": "[token,2I]",
                     "p50_ms": p50, "samples_ms": samples,
                     "numerically_correct": bool(torch.allclose(out, ref, rtol=2e-3, atol=2e-3)),
                     "input": tensor_meta(x), "state": None})
    elif structure == "moe":
        tokens = padded_tokens(case)
        experts = max(1, int(case["shape"]["num_experts"]))
        topk = min(experts, max(1, int(case["shape"]["experts_per_token"])))
        logits = torch.randn(tokens, experts, device=device, dtype=torch.float32)
        weights = torch.empty(tokens, topk, device=device, dtype=torch.float32)
        ids = torch.empty(tokens, topk, device=device, dtype=torch.int32)
        fn = lambda: sgl_kernel.topk_softmax(weights, ids, logits, True)
        p50, samples = time_cuda(torch, fn, warmup, iterations)
        fn()
        rows.append({"native_operator": "sgl_kernel.topk_softmax",
                     "variant": "token_expert", "selected_layout": "logits[token,expert]",
                     "p50_ms": p50, "samples_ms": samples,
                     "numerically_correct": bool(torch.isfinite(weights).all() and
                                                   (ids >= 0).all() and (ids < experts).all()),
                     "input": tensor_meta(logits), "state": tensor_meta(ids)})
    elif structure == "mamba2":
        tokens = padded_tokens(case)
        dim = int(case["shape"]["hidden_size"])
        width = int(case["dimensions"].get("conv_kernel") or 4)
        weight = torch.full((dim, width), 0.01, device=device, dtype=dtype)
        batch = int(case["shape"].get("batch", 1))
        per_request_tokens = int(case["shape"]["query_length"])
        state = torch.zeros(batch, dim, width - 1, device=device, dtype=dtype)
        indices = torch.arange(batch, device=device, dtype=torch.int32)
        if case["phase"] == "decode":
            # sgl_kernel's CUDA update op requires the explicit singleton
            # sequence axis even though higher layers often expose [B,D].
            x = torch.full((batch, dim, 1), 0.01, device=device, dtype=dtype)
            def fn():
                state.zero_()
                out = x.clone()
                sgl_kernel.causal_conv1d_update(
                    out, state, weight, None, True, None, indices, -1)
                return out
            operator, layout, variant = ("sgl_kernel.causal_conv1d_update",
                                         "x[batch,dim,1]", "decode_update")
        else:
            x = torch.full((dim, tokens), 0.01, device=device, dtype=dtype)
            starts = torch.arange(0, tokens + 1, per_request_tokens,
                                  device=device, dtype=torch.int32)
            initial = torch.zeros(batch, device=device, dtype=torch.bool)
            def fn():
                state.zero_()
                out = x.clone()
                sgl_kernel.causal_conv1d_fwd(out, weight, None, state, starts,
                                             indices, initial, True, -1)
                return out
            operator, layout, variant = ("sgl_kernel.causal_conv1d_fwd",
                                         "x[dim,time]", "dimension_major")
        p50, samples = time_cuda(torch, fn, warmup, iterations)
        out = fn()
        rows.append({"native_operator": operator,
                     "variant": variant, "selected_layout": layout,
                     "p50_ms": p50, "samples_ms": samples,
                     "numerically_correct": bool(torch.isfinite(out).all()),
                     "input": tensor_meta(x), "state": tensor_meta(state)})
    else:
        rows.append({"status": "source_supported_runtime_not_safe",
                     "reason": "linear-attention FLA JIT probe did not terminate cleanly on sm_86; isolated source evidence retained",
                     "native_operator": "sglang.fla.chunk_gated_delta_rule"})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--structure", action="append", choices=(
        "gqa", "sliding_attention", "sparse_attention", "mla", "swiglu",
        "moe", "mamba2", "linear_attention"))
    parser.add_argument("--phase", action="append", choices=("prefill", "decode"))
    args = parser.parse_args()
    cases = load_cases(args.manifest)
    if args.structure:
        selected = set(args.structure)
        cases = [case for case in cases if case["structure"] in selected]
    if args.phase:
        selected_phases = set(args.phase)
        cases = [case for case in cases if case["phase"] in selected_phases]
    if args.limit:
        cases = cases[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cases:
        # Structure/phase process isolation intentionally asks for partitions
        # that some manifests do not contain (for example Mamba decode in the
        # 128 supplement).  An empty applicable set is not an execution error.
        args.output.write_text("", encoding="utf-8")
        print(json.dumps({"framework": args.framework, "cases": 0, "rows": 0,
                          "status": "no_cases_selected", "output": str(args.output)},
                         ensure_ascii=False))
        return 0
    rows = []
    runner = run_vllm if args.framework == "vllm" else run_sglang
    for case in cases:
        base = {"framework": args.framework, "case_id": case["case_id"],
                "structure": case["structure"], "phase": case["phase"],
                "contract_sha256": case.get("contract_sha256"),
                "coverage_level": "native_operator_slice", "full_subgraph": False,
                "model_id": case.get("model_id"), "target_rqs": case.get("target_rqs", [])}
        base["attention_family"] = attention_family(case)
        if case["structure"] in ATTENTION:
            base["kv_cache_implementation"] = (
                "vllm_block_paged_native_writer" if args.framework == "vllm"
                else "sglang_slot_major_native_writer")
        # One selected native operator layout records the framework decision
        # and its local cost, but is not a paired counterfactual for any RQ.
        # Keep relevance separate from direct causal evidence.
        base["related_rqs"] = RELATED_RQS[case["structure"]]
        base["directly_evidenced_rqs"] = []
        base["evidence_role"] = "framework_native_layout_decision_and_local_cost"
        try:
            results = runner(case, args.warmup, args.iterations)
            for result in results:
                result.setdefault("status", "success")
                rows.append({**base, **result})
        except BaseException as error:
            rows.append({**base, "status": "failed", "error": repr(error),
                         "traceback": traceback.format_exc()})
        args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                       for row in rows), encoding="utf-8")
    success = sum(row["status"] == "success" and row.get("numerically_correct", True)
                  for row in rows)
    print(json.dumps({"framework": args.framework, "cases": len(cases),
                      "rows": len(rows), "successful_correct_rows": success,
                      "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
