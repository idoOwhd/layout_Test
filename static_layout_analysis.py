#!/usr/bin/env python3
"""Dependency-free, falsifiable layout metrics for the layout study.

This is deliberately not a latency predictor.  It reports address-level facts
(segments, span, padding, bank conflicts and conversion bytes) that can explain
GPU measurements without pretending to replace them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def kv_offset(layout: str, token: int, head: int, dim: int, *, tokens: int,
              heads: int, head_dim: int, block_size: int) -> int:
    """Element offset for the four common logical [token, head, dim] layouts."""
    if layout == "NHD":
        return (token * heads + head) * head_dim + dim
    if layout == "HND":
        return (head * tokens + token) * head_dim + dim
    block, within = divmod(token, block_size)
    if layout == "paged_NHD":
        return ((block * block_size + within) * heads + head) * head_dim + dim
    if layout == "paged_HND":
        return ((block * heads + head) * block_size + within) * head_dim + dim
    raise ValueError(layout)


def segments(addresses: list[int], transaction_bytes: int) -> int:
    return len({address // transaction_bytes for address in addresses})


def attention_metrics(case: dict, transaction_bytes: int) -> list[dict]:
    s = case["shape"]
    tokens = s["tokens"]
    heads = s["kv_heads"]
    dim = s["head_dim"]
    block = s["block_size"]
    element_bytes = 2
    records = []
    for layout in case["layouts"]:
        if layout not in {"NHD", "HND", "paged_NHD", "paged_HND"}:
            continue
        # A decode-style warp tile: four tokens, eight lanes/token, each lane
        # requests eight fp16 channels (one aligned 128-bit vector).
        addresses = []
        sample_tokens = min(tokens, 4)
        sample_dim = min(dim, 64)
        for token in range(sample_tokens):
            for d in range(sample_dim):
                addresses.append(kv_offset(layout, token, 0, d, tokens=tokens,
                                           heads=heads, head_dim=dim,
                                           block_size=block) * element_bytes)
        requested = len(addresses) * element_bytes
        transferred = segments(addresses, transaction_bytes) * transaction_bytes
        padded_tokens = ceil_div(tokens, block) * block if layout.startswith("paged") else tokens
        storage_elements = padded_tokens * heads * dim
        records.append({
            "case_id": case["case_id"], "family": case["family"], "layout": layout,
            "metric_scope": "address_model", "sample_requested_bytes": requested,
            "sample_transactions": transferred // transaction_bytes,
            "sample_global_load_efficiency": requested / transferred,
            "sample_address_span_bytes": max(addresses) - min(addresses) + element_bytes,
            "storage_bytes": storage_elements * element_bytes,
            "padding_bytes": (padded_tokens - tokens) * heads * dim * element_bytes,
            "conversion_bytes_one_way": tokens * heads * dim * element_bytes * 2,
            "caveat": "Transaction count is necessary but not sufficient; empirical kernels test cache locality, occupancy and instruction selection."
        })
    return records


def mla_metrics(case: dict) -> list[dict]:
    s = case["shape"]
    t, h, d, latent, block = s["tokens"], s["query_heads"], s["head_dim"], s["latent_dim"], s["block_size"]
    rows = []
    for layout in case["layouts"]:
        if layout == "expanded_NHD":
            elems, padded = t * h * d, 0
        elif layout == "latent_token_major":
            elems, padded = t * latent, 0
        elif layout == "latent_paged":
            pt = ceil_div(t, block) * block
            elems, padded = pt * latent, (pt - t) * latent * 2
        else:
            continue
        rows.append({"case_id": case["case_id"], "family": case["family"], "layout": layout,
                     "metric_scope": "storage_model", "storage_bytes": elems * 2,
                     "padding_bytes": padded, "bytes_vs_expanded": elems / (t * h * d),
                     "scientific_variable": "cache representation changes both layout and information volume; report it separately from within-representation permutations"})
    return rows


def swiglu_metrics(case: dict) -> list[dict]:
    s = case["shape"]
    m, i = s["m"], s["intermediate"]
    tensor_bytes = m * i * 2
    rows = []
    for layout in case["layouts"]:
        if layout == "separate_gate_up":
            launches, intermediate, conversion = 3, 2 * tensor_bytes, 0
        elif layout == "packed_gate_up":
            launches, intermediate, conversion = 2, 2 * tensor_bytes, 0
        else:
            launches, intermediate, conversion = 2, 2 * tensor_bytes, 4 * tensor_bytes
        rows.append({"case_id": case["case_id"], "family": case["family"], "layout": layout,
                     "metric_scope": "traffic_lower_bound", "materialized_intermediate_bytes": intermediate,
                     "minimum_kernel_launches": launches, "explicit_relayout_bytes": conversion,
                     "scientific_variable": "producer/consumer co-design and fusion boundary"})
    return rows


def moe_metrics(case: dict) -> list[dict]:
    s = case["shape"]
    t, h, topk, cap, e = s["tokens"], s["hidden"], s["topk"], s["expert_capacity"], s["experts"]
    for layout in case["layouts"]:
        if layout == "token_major_indirect":
            moved, padding, locality = 0, 0, "irregular"
        else:
            assigned = t * topk
            slots = max(assigned, e * cap)
            moved, padding, locality = assigned * h * 2 * 2, (slots - assigned) * h * 2, "contiguous_per_expert"
        yield {"case_id": case["case_id"], "family": case["family"], "layout": layout,
               "metric_scope": "packing_model", "pack_and_unpack_bytes": moved,
               "capacity_padding_bytes": padding, "expert_access": locality,
               "scientific_variable": "pay packing once to improve grouped-GEMM locality"}


def state_metrics(case: dict) -> list[dict]:
    s = case["shape"]
    b, t, h, d, n, chunk = s["batch"], s["tokens"], s["heads"], s["head_dim"], s["state_dim"], s["chunk"]
    total = b * t * h * (d + n) * 2
    rows = []
    for layout in case["layouts"]:
        relayout = 0 if layout == "sequence_major" else total * 2
        reusable = layout == "chunked_head_major"
        rows.append({"case_id": case["case_id"], "family": case["family"], "layout": layout,
                     "metric_scope": "scan_model", "working_tensor_bytes": total,
                     "explicit_relayout_bytes": relayout, "chunk_local_reuse": reusable,
                     "chunks": ceil_div(t, chunk), "scientific_variable": "scan-axis locality versus head/state-axis locality"})
    return rows


def conv_metrics(case: dict) -> list[dict]:
    s = case["shape"]
    elements = s["batch"] * s["channels"] * s["height"] * s["width"]
    rows = []
    for layout in case["layouts"]:
        rows.append({"case_id": case["case_id"], "family": case["family"], "layout": layout,
                     "metric_scope": "storage_model", "storage_bytes": elements * 2,
                     "conversion_bytes_one_way": elements * 2 * 2,
                     "unit_stride_axis": "width" if layout == "NCHW" else "channels",
                     "scientific_variable": "conv tensor-core channel vectorization versus surrounding spatial/reduction operators"})
    return rows


def reduction_metrics(case: dict) -> list[dict]:
    s = case["shape"]
    elems = s["rows"] * s["cols"]
    return [{"case_id": case["case_id"], "family": case["family"], "layout": layout,
             "metric_scope": "conversion_model", "storage_bytes": elems * 2,
             "explicit_relayout_bytes": elems * 4 if layout == "tiled_to_row_major" else 0,
             "reduction_axis_contiguous": layout in {"row_major", "tiled_to_row_major"},
             "scientific_variable": "conversion amortization boundary"} for layout in case["layouts"]]


def shared_metrics(case: dict, banks: int, bank_width: int) -> list[dict]:
    rows = []
    for layout in case["layouts"]:
        bank_hits = {}
        for lane in range(32):
            if layout == "row_major_stride32":
                index = lane * 32
            elif layout == "padded_stride33":
                index = lane * 33
            else:
                # XOR row bits into the column for a column-wise consumer.
                index = lane * 32 + lane
            bank = (index * case["shape"]["element_bytes"] // bank_width) % banks
            bank_hits[bank] = bank_hits.get(bank, 0) + 1
        rows.append({"case_id": case["case_id"], "family": case["family"], "layout": layout,
                     "metric_scope": "shared_bank_model", "active_banks": len(bank_hits),
                     "max_bank_conflict_degree": max(bank_hits.values()),
                     "serialized_wavefront_lower_bound": max(bank_hits.values()),
                     "scientific_variable": "zero-copy swizzle versus shared-memory bank serialization"})
    return rows


def analyze(data: dict) -> list[dict]:
    rows = []
    tx = data["gpu_transaction_bytes"]
    for case in data["cases"]:
        family = case["family"]
        if family in {"attention_kv", "diffusion_attention"}:
            rows += attention_metrics(case, tx)
        elif family == "mla_cache":
            rows += mla_metrics(case)
        elif family == "swiglu":
            rows += swiglu_metrics(case)
        elif family == "moe":
            rows += list(moe_metrics(case))
        elif family == "state_space":
            rows += state_metrics(case)
        elif family == "diffusion_conv":
            rows += conv_metrics(case)
        elif family == "reduction":
            rows += reduction_metrics(case)
        elif family == "shared_memory":
            rows += shared_metrics(case, data["shared_bank_count"], data["shared_bank_width_bytes"])
        else:
            raise ValueError(f"unsupported family: {family}")
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=Path, default=HERE / "layout_cases.json")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = analyze(json.loads(args.cases.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} static layout records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
