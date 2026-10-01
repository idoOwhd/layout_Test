#!/usr/bin/env python3
"""Paired request-batch counterfactuals for native vLLM/SGLang KV writers.

The source manifests describe real-world model shapes, but currently all have
``shape.batch == 1``.  This runner preserves every architectural dimension and
the per-request sequence length, then derives request batches 1/2/4/8.  Each
derived point is explicitly labelled: it is a counterfactual on the original
contract, not a claim that the source checkpoint used that serving batch.

Two physical slot schedules are measured:

* ``request_major_contiguous``: one request follows another in cache storage;
* ``block_interleaved``: 16-token pages from different requests alternate.

For decode, block interleaving places the one appended token for each request
in a different page, which exposes the scattered multi-request write contract.
Successful rows call framework-shipped native KV-cache writers only.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import traceback
from pathlib import Path

from native_framework_subgraph_probe import (
    ATTENTION,
    attention_dimensions,
    load_cases,
    tensor_meta,
    time_cuda,
)


POLICIES = ("request_major_contiguous", "block_interleaved")


def attention_family(case: dict) -> str:
    structure, shape = case["structure"], case["shape"]
    if structure in {"gqa", "sliding_attention"}:
        qh, kh = int(shape["num_query_heads"]), int(shape["num_kv_heads"])
        kind = "MQA" if kh == 1 else "MHA" if qh == kh else "GQA"
        return f"sliding_{kind}" if structure == "sliding_attention" else kind
    return "sparse_attention" if structure == "sparse_attention" else "MLA"


def parse_batches(text: str) -> list[int]:
    values = sorted({int(value) for value in text.split(",") if value.strip()})
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("request batches must be positive comma-separated integers")
    return values


def slot_mapping(request_batch: int, tokens_per_request: int, policy: str,
                 block_size: int = 16) -> list[int]:
    """Return slots for request-major flattened inputs."""
    if policy == "request_major_contiguous":
        return list(range(request_batch * tokens_per_request))
    if policy != "block_interleaved":
        raise ValueError(f"unknown policy: {policy}")
    slots = []
    blocks_per_request = math.ceil(tokens_per_request / block_size)
    for request in range(request_batch):
        for token in range(tokens_per_request):
            token_block, offset = divmod(token, block_size)
            physical_block = token_block * request_batch + request
            slots.append(physical_block * block_size + offset)
    assert len(slots) == request_batch * tokens_per_request
    assert max(slots, default=-1) < blocks_per_request * request_batch * block_size
    return slots


def estimated_bytes(total_tokens: int, cache_slots: int, heads: int,
                    kdim: int, vdim: int) -> int:
    # fp16 K/V inputs plus fp16 K/V caches.  Add 10% for mappings/allocator
    # alignment; this is a preflight guard, not a memory profiler.
    elements = total_tokens * heads * (kdim + vdim)
    elements += cache_slots * heads * (kdim + vdim)
    return int(elements * 2 * 1.10)


def balanced_cases(cases: list[dict], limit: int) -> list[dict]:
    """Prefer one case per structure×phase before filling remaining slots."""
    selected, seen = [], set()
    for case in cases:
        key = (case["structure"], case["phase"])
        if key not in seen:
            selected.append(case)
            seen.add(key)
            if len(selected) == limit:
                return selected
    selected_ids = {case["case_id"] for case in selected}
    for case in cases:
        if case["case_id"] not in selected_ids:
            selected.append(case)
            if len(selected) == limit:
                break
    return selected


def common_row(case: dict, framework: str, request_batch: int,
               tokens_per_request: int, policy: str, slots: list[int]) -> dict:
    original_batch = int(case.get("shape", {}).get("batch", 1))
    return {
        "framework": framework,
        "case_id": case["case_id"],
        "structure": case["structure"],
        "phase": case["phase"],
        "base_contract_sha256": case.get("contract_sha256"),
        "source_manifest_batch": original_batch,
        "request_batch": request_batch,
        "derived_counterfactual": request_batch != original_batch,
        "derived_axis": "request_batch",
        "per_request_new_tokens": tokens_per_request,
        "total_new_tokens": request_batch * tokens_per_request,
        "slot_policy": policy,
        "attention_family": attention_family(case),
        "slot_count": max(slots) + 1,
        "coverage_level": "framework_native_kv_writer_slice",
        "full_attention_or_serving_e2e": False,
        # This paired writer-only intervention directly addresses local
        # workload stability (RQ4).  It is supporting, not direct, evidence
        # for edge/global-domain/state-migration/search-space RQs.
        "directly_evidenced_rqs": ["L-RQ4"],
        "related_rqs": ["L-RQ1", "L-RQ2", "L-RQ4", "L-RQ6", "L-RQ9"],
        "evidence_role": "paired_request_batch_and_slot_schedule_counterfactual",
        "consumer_scope": "native_kv_writer_only",
        "multi_consumer": False,
    }


def run_vllm(case: dict, request_batch: int, policy: str,
             warmup: int, iterations: int, max_bytes: int) -> dict:
    import torch
    import vllm._custom_ops  # noqa: F401 - registers cache operators
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash,
        triton_reshape_and_cache_flash_diffkv,
    )

    per_request, heads, kdim, vdim = attention_dimensions(case)
    slots_cpu = slot_mapping(request_batch, per_request, policy)
    total = request_batch * per_request
    cache_slots = max(slots_cpu) + 1
    estimate = estimated_bytes(total, cache_slots, heads, kdim, vdim)
    base = common_row(case, "vllm", request_batch, per_request, policy, slots_cpu)
    base["estimated_allocation_bytes"] = estimate
    if estimate > max_bytes:
        return {**base, "status": "preflight_memory_skip",
                "reason": f"estimated {estimate} bytes exceeds --max-bytes {max_bytes}"}

    dtype, device, block = torch.float16, "cuda", 16
    torch.manual_seed(int(case.get("tensor_seed", 0)) + request_batch)
    key = torch.randn(total, heads, kdim, device=device, dtype=dtype)
    value = torch.randn(total, heads, vdim, device=device, dtype=dtype)
    slot = torch.tensor(slots_cpu, device=device, dtype=torch.int64)
    blocks = math.ceil(cache_slots / block)
    scale = torch.ones(1, device=device, dtype=torch.float32)
    if kdim != vdim:
        cache = torch.empty(blocks, block, heads, kdim + vdim,
                            device=device, dtype=dtype)
        fn = lambda: triton_reshape_and_cache_flash_diffkv(
            key, value, cache, slot, "auto", scale, scale)
        operator = "triton_reshape_and_cache_flash_diffkv"
    else:
        key_cache = torch.empty(blocks, block, heads, kdim, device=device, dtype=dtype)
        value_cache = torch.empty(blocks, block, heads, vdim, device=device, dtype=dtype)
        fn = lambda: triton_reshape_and_cache_flash(
            key, value, key_cache, value_cache, slot, "auto", scale, scale)
        operator = "triton_reshape_and_cache_flash"
    p50, samples = time_cuda(torch, fn, warmup, iterations)
    fn()
    if kdim != vdim:
        flat = cache.reshape(-1, heads, kdim + vdim)
        ok = (torch.allclose(flat[slot, :, :kdim], key) and
              torch.allclose(flat[slot, :, kdim:], value))
        state = cache
    else:
        flat_k = key_cache.reshape(-1, heads, kdim)
        flat_v = value_cache.reshape(-1, heads, vdim)
        ok = torch.allclose(flat_k[slot], key) and torch.allclose(flat_v[slot], value)
        state = key_cache
    return {
        **base, "status": "success", "native_operator": operator,
        "selected_layout": "NHD cache[block,token,head,dim]",
        "kv_cache_implementation": "vllm_block_paged_separate_or_diffkv_combined",
        "kv_block_size": block,
        "p50_ms": p50, "samples_ms": samples,
        "new_tokens_per_ms": total / p50,
        "numerically_correct": bool(ok), "input": tensor_meta(key),
        "state": tensor_meta(state),
    }


def run_sglang(case: dict, request_batch: int, policy: str,
               warmup: int, iterations: int, max_bytes: int) -> dict:
    import torch
    import sgl_kernel  # noqa: F401
    from sglang.kernels.ops.kvcache.kvcache import store_cache

    per_request, heads, kdim, vdim = attention_dimensions(case)
    slots_cpu = slot_mapping(request_batch, per_request, policy)
    total = request_batch * per_request
    cache_slots = max(slots_cpu) + 1
    estimate = estimated_bytes(total, cache_slots, heads, kdim, vdim)
    base = common_row(case, "sglang", request_batch, per_request, policy, slots_cpu)
    base["estimated_allocation_bytes"] = estimate
    if estimate > max_bytes:
        return {**base, "status": "preflight_memory_skip",
                "reason": f"estimated {estimate} bytes exceeds --max-bytes {max_bytes}"}

    dtype, device = torch.float16, "cuda"
    torch.manual_seed(int(case.get("tensor_seed", 0)) + request_batch)
    key = torch.randn(total, heads * kdim, device=device, dtype=dtype)
    value = torch.randn(total, heads * vdim, device=device, dtype=dtype)
    key_cache = torch.empty(cache_slots, heads * kdim, device=device, dtype=dtype)
    value_cache = torch.empty(cache_slots, heads * vdim, device=device, dtype=dtype)
    slots = torch.tensor(slots_cpu, device=device, dtype=torch.int64)
    fn = lambda: store_cache(key, value, key_cache, value_cache, slots,
                             reserved_skip_index=-1)
    p50, samples = time_cuda(torch, fn, warmup, iterations)
    fn()
    ok = torch.allclose(key_cache[slots], key) and torch.allclose(value_cache[slots], value)
    return {
        **base, "status": "success", "native_operator": "sglang.kvcache.store_cache",
        "selected_layout": "NHD cache[slot,H*D]",
        "kv_cache_implementation": "sglang_slot_major_separate_kv_pool",
        "kv_block_size": 1,
        "p50_ms": p50, "samples_ms": samples,
        "new_tokens_per_ms": total / p50,
        "numerically_correct": bool(ok), "input": tensor_meta(key),
        "state": tensor_meta(key_cache),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-batches", type=parse_batches, default=parse_batches("1,2,4,8"))
    parser.add_argument("--policy", action="append", choices=POLICIES)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--balanced-limit", type=int)
    parser.add_argument("--max-bytes", type=int, default=2 * 1024**3)
    args = parser.parse_args()
    cases = [case for case in load_cases(args.manifest) if case["structure"] in ATTENTION]
    if args.balanced_limit:
        cases = balanced_cases(cases, args.balanced_limit)
    elif args.limit:
        cases = cases[:args.limit]
    policies = args.policy or list(POLICIES)
    runner = run_vllm if args.framework == "vllm" else run_sglang
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for case in cases:
        for request_batch in args.request_batches:
            for policy in policies:
                try:
                    row = runner(case, request_batch, policy, args.warmup,
                                 args.iterations, args.max_bytes)
                except BaseException as error:
                    per_request = attention_dimensions(case)[0]
                    slots = slot_mapping(request_batch, per_request, policy)
                    row = {**common_row(case, args.framework, request_batch,
                                        per_request, policy, slots),
                           "status": "failed", "error": repr(error),
                           "traceback": traceback.format_exc()}
                rows.append(row)
                args.output.write_text(
                    "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows),
                    encoding="utf-8")
    good = sum(row.get("status") == "success" and row.get("numerically_correct")
               for row in rows)
    print(json.dumps({"framework": args.framework, "source_cases": len(cases),
                      "rows": len(rows), "successful_correct_rows": good,
                      "output": str(args.output)}, ensure_ascii=False))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
