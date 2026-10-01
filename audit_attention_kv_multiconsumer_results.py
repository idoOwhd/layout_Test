#!/usr/bin/env python3
"""Audit multi-consumer, attention-family and KV-cache coverage by framework."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def manifest_cases(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value.get("cases", value) if isinstance(value, dict) else value


def family(case: dict) -> str | None:
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


def successful(row: dict) -> bool:
    status = row.get("status", "success")
    correct = row.get("correct", row.get("numerically_correct", True))
    if isinstance(correct, str):
        correct = correct.lower() in {"1", "true", "yes"}
    return status == "success" and bool(correct)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir
    cases = manifest_cases(args.manifest)
    source_families = {value for case in cases if (value := family(case))}
    source_structures = {case["structure"] for case in cases
                         if case["structure"] in {"gqa", "sliding_attention",
                                                  "sparse_attention", "mla"}}

    cuda = csv_rows(root / "raw" / "cuda_reference.csv")
    triton = csv_rows(root / "raw" / "triton_explicit.csv")
    tvm = csv_rows(root / "raw" / "tvm_projected.csv")
    cutlass = csv_rows(root / "raw" / "cutlass_attention.csv")
    native_dir = root / "native_subgraphs" / "raw"
    native = {
        "vLLM": jsonl(native_dir / "vllm_subgraph_native.jsonl"),
        "SGLang": jsonl(native_dir / "sglang_subgraph_native.jsonl"),
    }
    kv_dir = root / "kv_request_batch" / "raw"
    kv = {
        "vLLM": jsonl(kv_dir / "vllm_kv_request_batch.jsonl"),
        "SGLang": jsonl(kv_dir / "sglang_kv_request_batch.jsonl"),
    }
    serving = {
        "vLLM": jsonl(root / "raw" / "vllm_native.jsonl"),
        "SGLang": jsonl(root / "raw" / "sglang_native.jsonl"),
    }

    cuda_multi = [row for row in cuda
                  if row.get("benchmark") == "weighted_multi_consumer_pipeline"]
    cuda_fanouts = {(row.get("fanout_head"), row.get("fanout_token")) for row in cuda_multi}
    cuda_pages = {row.get("page_size") for row in cuda
                  if row.get("benchmark") in {"paged_decode_head", "sparse_paged_decode_head"}}
    cuda_tables = {row.get("trace_kind") for row in cuda
                   if row.get("benchmark") in {"paged_decode_head", "sparse_paged_decode_head"}}

    matrix: list[dict] = []
    # Triton owns explicit logical->physical mapping and both consumer kernels.
    triton_good = [row for row in triton if successful(row)]
    triton_multi = [row for row in triton_good
                    if row.get("experiment") == "weighted_multi_consumer_pipeline"]
    triton_fanouts = {(row.get("fanout_head"), row.get("fanout_token"))
                      for row in triton_multi}
    matrix += [
        {"framework": "Triton", "dimension": "multi_consumer",
         "status": "direct" if len(triton_fanouts) >= 7 else "missing",
         "detail": f"fanout_pairs={len(triton_fanouts)}"},
        {"framework": "Triton", "dimension": "attention_families",
         "status": "direct" if {"MHA", "MQA", "GQA"} <=
                   {row.get("attention_kind") for row in triton_good} else "missing",
         "detail": str(sorted({row.get('attention_kind') for row in triton_good if row.get('attention_kind')}))},
        {"framework": "Triton", "dimension": "kv_cache_implementations",
         "status": "direct" if {"NHD", "HND", "paged_NHD", "paged_HND"} <=
                   {row.get("layout") for row in triton_good} else "missing",
         "detail": str(sorted({row.get('layout') for row in triton_good}))},
    ]

    tvm_good = [row for row in tvm if successful(row)]
    tvm_multi = [row for row in tvm_good
                 if row.get("experiment") == "weighted_multi_consumer_pipeline"]
    tvm_consumer_only = bool(tvm_multi) and all(
        str(row.get("producer_materialization_timed", "")).lower() == "false"
        for row in tvm_multi)
    matrix += [
        {"framework": "TVM", "dimension": "multi_consumer",
         "status": ("direct_consumer_only" if tvm_consumer_only else
                    "direct" if tvm_multi else "missing"),
         "detail": "row/column consumers in one TVM CUDA module; producer/materialization is not timed"},
        {"framework": "TVM", "dimension": "attention_families",
         "status": "supporting" if source_structures <=
                   {row.get("structure") for row in tvm_good} else "missing",
         "detail": "real-case boundary projection; not full attention"},
        {"framework": "TVM", "dimension": "kv_cache_implementations",
         "status": "N/A", "detail": "generic tensor compiler adapter does not own serving KV allocation"},
    ]

    cutlass_good = [row for row in cutlass if successful(row)]
    matrix += [
        {"framework": "CUTLASS/CuTe", "dimension": "multi_consumer",
         "status": "N/A-not-owned",
         "detail": "the measured attention-score tile boundary has one consumer and no persistent shared-KV allocator"},
        {"framework": "CUTLASS/CuTe", "dimension": "attention_families",
         "status": "supporting" if source_structures <=
                   {row.get("structure") for row in cutlass_good} else "missing",
         "detail": str(sorted({row.get('structure') for row in cutlass_good}))},
        {"framework": "CUTLASS/CuTe", "dimension": "kv_cache_implementations",
         "status": "N/A", "detail": "kernel-template layer; KV allocator is upstream"},
    ]

    for framework in ("vLLM", "SGLang"):
        native_good = [row for row in native[framework] if successful(row)]
        kv_good = [row for row in kv[framework] if successful(row)]
        serving_good = [row for row in serving[framework] if row.get("status", "success") == "success"]
        native_structures = {row.get("structure") for row in native_good}
        batches = {int(row.get("request_batch", 0)) for row in kv_good}
        policies = {row.get("slot_policy") for row in kv_good}
        implementations = {row.get("kv_cache_implementation") for row in kv_good}
        matrix += [
            {"framework": framework, "dimension": "multi_consumer",
             "status": "supporting" if serving_good and {1, 2, 4, 8} <= batches else "missing",
             "detail": "shared persistent cache across prefill/decode plus multi-request; not heterogeneous target/draft consumers"},
            {"framework": framework, "dimension": "attention_families",
             "status": "direct_operator_slice" if source_structures <= native_structures else "missing",
             "detail": str(sorted(native_structures & source_structures))},
            {"framework": framework, "dimension": "kv_cache_implementations",
             "status": "direct" if implementations and {1, 2, 4, 8} <= batches and
                       {"request_major_contiguous", "block_interleaved"} <= policies else "missing",
             "detail": f"implementations={sorted(x for x in implementations if x)}; batches={sorted(batches)}; policies={sorted(x for x in policies if x)}"},
        ]

    required_source = {"MHA", "MQA", "GQA", "sparse_attention", "MLA"}
    source_base_families = {value.removeprefix("sliding_") for value in source_families}
    source_ok = required_source <= source_base_families and any(
        item.startswith("sliding_") for item in source_families)
    cuda_reference_ok = (len(cuda_fanouts) >= 7 and len(cuda_pages) >= 8
                         and {"identity", "reverse", "permuted"} <= cuda_tables)
    no_missing = all(row["status"] != "missing" for row in matrix)
    complete = source_ok and cuda_reference_ok and no_missing
    payload = {"schema_version": 1, "complete": complete,
               "source_attention_families": sorted(source_families),
               "source_attention_structures": sorted(source_structures),
               "cuda_control": {"fanout_pairs": len(cuda_fanouts),
                                "page_sizes": sorted(cuda_pages),
                                "block_table_orders": sorted(cuda_tables),
                                "complete": cuda_reference_ok},
               "matrix": matrix,
               "guardrail": "supporting is executed evidence but does not equal a heterogeneous multi-consumer direct intervention"}
    (root / "ATTENTION_KV_MULTICONSUMER_AUDIT.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Attention / KV cache / 多 consumer 跨框架审计", "",
             f"- real-world attention families：`{sorted(source_families)}`。",
             f"- 完整性：**{'PASS' if complete else 'FAIL'}**。",
             "- `supporting` 表示真实执行过，但不能替代 heterogeneous multi-consumer direct intervention；`direct_consumer_only` 不包含 producer/materialization。", "",
             "| 框架 | 维度 | 证据等级 | 说明 |", "|---|---|---|---|"]
    lines += [f"| {row['framework']} | {row['dimension']} | {row['status']} | {row['detail']} |"
              for row in matrix]
    lines += ["", "## 严格边界", "",
              "- MHA/MQA/GQA 的多 consumer 由 CUDA reference 与 Triton 完整 pipeline 直接测量。",
              "- sparse attention 与 MLA 在 vLLM/SGLang 中是 native writer/operator slice；并非完整 sparse/MLA attention replay。",
              "- vLLM/SGLang 的 multi-request/prefill→decode 是 supporting evidence；target/draft 或多个异构 attention backend 共用同一 cache 仍需独立实验。",
              "- TVM/CUTLASS 不拥有 serving KV allocator，相应 cell 是 N/A，不拿其他框架结果代替。", ""]
    (root / "ATTENTION_KV_MULTICONSUMER_AUDIT_CN.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps({"complete": complete,
                      "report": str(root / "ATTENTION_KV_MULTICONSUMER_AUDIT_CN.md")},
                     ensure_ascii=False))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
