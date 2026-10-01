#!/usr/bin/env python3
"""Paired Triton QK microbenchmark over four physical KV layouts.

The arithmetic, values and launch geometry are held fixed.  Only the mapping
from logical [token, kv_head, channel] to physical storage changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from pathlib import Path

import torch
import triton
import triton.language as tl


LAYOUTS = {"NHD": 0, "HND": 1, "paged_NHD": 2, "paged_HND": 3}


def tsv_cases(path: Path) -> list[dict]:
    """Load the MHA/MQA/GQA shapes selected from the pinned 640-case catalog."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [{
        "case_id": row["case_id"], "family": "attention_kv",
        "shape": {
            "tokens": int(row["tokens"]),
            "query_heads": int(row["query_heads"]),
            "kv_heads": int(row["kv_heads"]),
            "head_dim": int(row["head_dim"]),
            "block_size": 16,
        },
        "layouts": list(LAYOUTS),
        "phase": row["phase"], "attention_kind": row["attention_kind"],
        "subgraph": row["subgraph"], "model": row["model"],
    } for row in rows]


@triton.jit
def store_layout_kernel(source, destination, tokens: tl.constexpr,
                        kv_heads: tl.constexpr, head_dim: tl.constexpr,
                        block_size: tl.constexpr, layout_id: tl.constexpr,
                        BLOCK: tl.constexpr):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = tokens * kv_heads * head_dim
    valid = linear < total
    d = linear % head_dim
    quotient = linear // head_dim
    head = quotient % kv_heads
    token = quotient // kv_heads
    if layout_id == 0:
        output = (token * kv_heads + head) * head_dim + d
    elif layout_id == 1:
        output = (head * tokens + token) * head_dim + d
    elif layout_id == 2:
        page = token // block_size
        within = token % block_size
        output = ((page * block_size + within) * kv_heads + head) * head_dim + d
    else:
        page = token // block_size
        within = token % block_size
        output = ((page * kv_heads + head) * block_size + within) * head_dim + d
    value = tl.load(source + linear, mask=valid)
    tl.store(destination + output, value, mask=valid)


@triton.jit
def qk_layout_kernel(q, k, scores, tokens: tl.constexpr, q_heads: tl.constexpr,
                     kv_heads: tl.constexpr, head_dim: tl.constexpr,
                     block_size: tl.constexpr, layout_id: tl.constexpr,
                     BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    block_n = tl.program_id(0)
    qh = tl.program_id(1)
    n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d = tl.arange(0, BLOCK_D)
    kvh = qh // (q_heads // kv_heads)
    qv = tl.load(q + qh * head_dim + d, mask=d < head_dim, other=0.0)
    if layout_id == 0:
        offsets = (n[:, None] * kv_heads + kvh) * head_dim + d[None, :]
    elif layout_id == 1:
        offsets = (kvh * tokens + n[:, None]) * head_dim + d[None, :]
    elif layout_id == 2:
        block = n[:, None] // block_size
        within = n[:, None] % block_size
        offsets = ((block * block_size + within) * kv_heads + kvh) * head_dim + d[None, :]
    else:
        block = n[:, None] // block_size
        within = n[:, None] % block_size
        offsets = ((block * kv_heads + kvh) * block_size + within) * head_dim + d[None, :]
    kval = tl.load(k + offsets, mask=(n[:, None] < tokens) & (d[None, :] < head_dim), other=0.0)
    result = tl.sum(kval.to(tl.float32) * qv[None, :].to(tl.float32), axis=1)
    tl.store(scores + qh * tokens + n, result, mask=n < tokens)


@triton.jit
def token_layout_kernel(k, output, tokens: tl.constexpr, kv_heads: tl.constexpr,
                        head_dim: tl.constexpr, block_size: tl.constexpr,
                        layout_id: tl.constexpr, BLOCK_HD: tl.constexpr):
    """Token-local consumer complementary to the head-local QK traversal."""
    token = tl.program_id(0)
    hd = tl.arange(0, BLOCK_HD)
    head = hd // head_dim
    d = hd % head_dim
    if layout_id == 0:
        offsets = (token * kv_heads + head) * head_dim + d
    elif layout_id == 1:
        offsets = (head * tokens + token) * head_dim + d
    elif layout_id == 2:
        page = token // block_size
        within = token % block_size
        offsets = ((page * block_size + within) * kv_heads + head) * head_dim + d
    else:
        page = token // block_size
        within = token % block_size
        offsets = ((page * kv_heads + head) * block_size + within) * head_dim + d
    values = tl.load(k + offsets, mask=(head < kv_heads) & (d < head_dim), other=0.0)
    tl.store(output + token, tl.sum(values.to(tl.float32), axis=0))


def physical(logical: torch.Tensor, layout: str, block_size: int) -> torch.Tensor:
    tokens, heads, dim = logical.shape
    if layout == "NHD":
        return logical.contiguous()
    if layout == "HND":
        return logical.permute(1, 0, 2).contiguous()
    padded = math.ceil(tokens / block_size) * block_size
    if padded != tokens:
        logical = torch.nn.functional.pad(logical, (0, 0, 0, 0, 0, padded - tokens))
    blocked = logical.view(padded // block_size, block_size, heads, dim)
    if layout == "paged_NHD":
        return blocked.contiguous()
    if layout == "paged_HND":
        return blocked.permute(0, 2, 1, 3).contiguous()
    raise ValueError(layout)


def quantile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def time_ms(fn, warmup: int, iterations: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(iterations):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); b.synchronize()
        values.append(float(a.elapsed_time(b)))
    return {"p20_ms": quantile(values, .2), "p50_ms": quantile(values, .5),
            "p80_ms": quantile(values, .8), "iterations": iterations}


def run_one(case_id: str, tokens: int, q_heads: int, kv_heads: int, head_dim: int,
            block_size: int, layouts: list[str], warmup: int, iterations: int) -> list[dict]:
    torch.manual_seed(7)
    q = torch.randn((q_heads, head_dim), device="cuda", dtype=torch.float16)
    logical = torch.randn((tokens, kv_heads, head_dim), device="cuda", dtype=torch.float16)
    reference = torch.einsum("hd,thd->ht", q, logical[:, torch.arange(q_heads, device="cuda") // (q_heads // kv_heads)]).float()
    token_reference = logical.float().sum((1, 2))
    block_n = 32
    block_d = triton.next_power_of_2(head_dim)
    rows = []
    for layout in layouts:
        expected_cache = physical(logical, layout, block_size)
        cache = torch.empty_like(expected_cache)
        output = torch.empty((q_heads, tokens), device="cuda", dtype=torch.float32)
        token_output = torch.empty((tokens,), device="cuda", dtype=torch.float32)
        grid = (triton.cdiv(tokens, block_n), q_heads)
        total = tokens * kv_heads * head_dim
        store_grid = (triton.cdiv(total, 256),)
        store = lambda: store_layout_kernel[store_grid](
            logical, cache, tokens, kv_heads, head_dim, block_size, LAYOUTS[layout],
            BLOCK=256, num_warps=4, num_stages=1)
        consume = lambda: qk_layout_kernel[grid](
            q, cache, output, tokens, q_heads, kv_heads, head_dim, block_size,
            LAYOUTS[layout], BLOCK_N=block_n, BLOCK_D=block_d,
            num_warps=4, num_stages=2)
        block_hd = triton.next_power_of_2(kv_heads * head_dim)
        token_consume = lambda: token_layout_kernel[(tokens,)](
            cache, token_output, tokens, kv_heads, head_dim, block_size,
            LAYOUTS[layout], BLOCK_HD=block_hd, num_warps=4, num_stages=2)
        pipeline = lambda: (store(), consume())
        store(); consume(); token_consume(); torch.cuda.synchronize()
        error = float((output - reference).abs().max().item())
        token_error = float((token_output - token_reference).abs().max().item())
        producer_error = float((cache - expected_cache).abs().max().item())
        timing = time_ms(consume, warmup, iterations)
        producer_timing = time_ms(store, warmup, iterations)
        pipeline_timing = time_ms(pipeline, warmup, iterations)
        rows.append({"case_id": case_id, "framework": "triton-explicit", "layout": layout,
                     "experiment": "producer_consumer_edge",
                     "tokens": tokens, "q_heads": q_heads, "kv_heads": kv_heads,
                     "head_dim": head_dim, "block_size": block_size,
                     "block_n": block_n, "block_d": block_d, "num_warps": 4,
                     "num_stages": 2, "max_abs_error": error,
                     "producer_max_abs_error": producer_error,
                     "correct": error < 0.5 and producer_error == 0.0,
                     "producer_p20_ms": producer_timing["p20_ms"],
                     "producer_p50_ms": producer_timing["p50_ms"],
                     "producer_p80_ms": producer_timing["p80_ms"],
                     "pipeline_p20_ms": pipeline_timing["p20_ms"],
                     "pipeline_p50_ms": pipeline_timing["p50_ms"],
                     "pipeline_p80_ms": pipeline_timing["p80_ms"],
                     "physical_shape": list(cache.shape),
                     "physical_stride": list(cache.stride()),
                     "storage_bytes": cache.numel() * cache.element_size(), **timing})
        # Same produced cache, two consumers with opposing locality and seven
        # fan-out ratios.  Every row times producer + all consumer launches;
        # this is not a token-count cost model.
        for head_fanout, token_fanout in (
                (1, 8), (1, 4), (1, 2), (1, 1), (2, 1), (4, 1), (8, 1)):
            def multi_pipeline():
                store()
                for _ in range(head_fanout):
                    consume()
                for _ in range(token_fanout):
                    token_consume()
            multi_timing = time_ms(multi_pipeline, warmup, iterations)
            rows.append({
                "case_id": case_id, "framework": "triton-explicit", "layout": layout,
                "experiment": "weighted_multi_consumer_pipeline",
                "strategy": f"common_{layout}",
                "tokens": tokens, "q_heads": q_heads, "kv_heads": kv_heads,
                "head_dim": head_dim, "block_size": block_size,
                "fanout_head": head_fanout, "fanout_token": token_fanout,
                "head_max_abs_error": error, "token_max_abs_error": token_error,
                "max_abs_error": max(error, token_error),
                "correct": error < 0.5 and token_error < 0.5 and producer_error == 0.0,
                "physical_shape": list(cache.shape),
                "physical_stride": list(cache.stride()),
                "storage_bytes": cache.numel() * cache.element_size(),
                "evidence_level": "native_runtime_complete_pipeline",
                **multi_timing,
            })
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=Path, default=Path(__file__).with_name("layout_cases.json"))
    p.add_argument("--attention-tsv", type=Path,
                   help="override --cases with representative MHA/MQA/GQA TSV")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if args.attention_tsv:
        cases = tsv_cases(args.attention_tsv)
    else:
        data = json.loads(args.cases.read_text(encoding="utf-8"))
        cases = data["cases"]
    if args.limit:
        cases = cases[:args.limit]
    rows = []
    for case in cases:
        if case["family"] not in {"attention_kv", "diffusion_attention"}:
            continue
        s = case["shape"]
        current = run_one(case["case_id"], s["tokens"], s["query_heads"], s["kv_heads"],
                          s["head_dim"], s["block_size"], case["layouts"],
                          args.warmup, args.iterations)
        for row in current:
            row.update({key: case[key] for key in
                        ("phase", "attention_kind", "subgraph", "model") if key in case})
        rows += current
    props = torch.cuda.get_device_properties(0)
    for row in rows:
        row.update(device_name=props.name, compute_capability=f"{props.major}.{props.minor}",
                   torch_version=torch.__version__, triton_version=triton.__version__,
                   python_version=platform.python_version())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} Triton layout measurements to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
