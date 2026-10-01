#!/usr/bin/env python3
"""Build a discriminative, real-model-derived suite for RQ adequacy tests.

The pinned 640-case catalog is an architecture source, not a statistically
independent sample of 640 models.  This builder therefore selects diverse real
model parents first and then applies explicitly-labelled workload
counterfactuals.  The result balances structure, phase, request batch and
tile/page boundaries without pretending that controlled lengths are model
training configurations.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))

from discover_real_world_shapes import contract_hash, shape_payload, tensor_spec
from layout_research.build_llm_shape_diversity_v2 import batchify, rq_axes
from layout_research.build_persuasive_real_world_cases import (
    DEFAULT_MANIFEST,
    STRUCTURES,
    diverse_parents,
    load,
)
from layout_research.build_rq_llm_cases import write_attention_tsv
from real_world_workloads import estimate_case_bytes


# 64 profiles.  Long lengths are paired with smaller batches so the suite is
# executable on a 24 GiB card; every phase still contains B=1/2/4/8 and
# minus/exact/plus tile/page boundaries.
PREFILL_BY_BATCH = {
    1: (7, 15, 16, 17, 63, 64, 65, 513),
    2: (7, 15, 16, 17, 63, 64, 65, 257),
    4: (7, 15, 16, 17, 31, 32, 33, 129),
    8: (7, 8, 15, 16, 17, 31, 32, 33),
}
DECODE_BY_BATCH = {
    1: (31, 32, 33, 127, 128, 129, 4095, 8192),
    2: (31, 32, 33, 127, 128, 129, 2047, 4096),
    4: (31, 32, 33, 127, 128, 129, 1023, 2048),
    8: (31, 32, 33, 127, 128, 129, 511, 1024),
}


def attention_family(case: dict) -> str | None:
    structure = case["structure"]
    shape = case["shape"]
    if structure in {"gqa", "sliding_attention"}:
        query_heads = int(shape["num_query_heads"])
        kv_heads = int(shape["num_kv_heads"])
        kind = "MQA" if kv_heads == 1 else "MHA" if query_heads == kv_heads else "GQA"
        return f"sliding_{kind}" if structure == "sliding_attention" else kind
    if structure == "sparse_attention":
        return "sparse_attention"
    if structure == "mla":
        return "MLA"
    return None


def role(length: int) -> str:
    for boundary in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        if length == boundary - 1:
            return f"boundary_{boundary}_minus_one"
        if length == boundary:
            return f"boundary_{boundary}_exact"
        if length == boundary + 1:
            return f"boundary_{boundary}_plus_one"
    return "long_or_interior"


def profiles() -> list[tuple[str, int, int, str]]:
    values: list[tuple[str, int, int, str]] = []
    # Interleave batch and phase rather than exposing a prefix containing one
    # regime.  Smoke subsets therefore remain causal-axis diverse.
    for index in range(8):
        for batch in (1, 2, 4, 8):
            length = PREFILL_BY_BATCH[batch][index]
            values.append(("prefill", length, batch, role(length)))
        for batch in (1, 2, 4, 8):
            length = DECODE_BY_BATCH[batch][index]
            values.append(("decode", length, batch, role(length)))
    assert len(values) == 64
    return values


def create(parent: dict, parent_index: int, profile_index: int,
           phase: str, length: int, batch: int, boundary_role: str) -> dict:
    shape = shape_payload(parent["structure"], parent["dimensions"], phase,
                          length, 8192 if phase == "prefill" else length)
    shape = batchify(shape, batch)
    seed, spec_hash = tensor_spec(parent["structure"], phase, shape)
    row = {key: parent[key] for key in (
        "model_id", "model_revision", "model_type", "structure",
        "structure_label", "dimensions", "graph_nodes", "source_url")}
    axes = rq_axes(parent["structure"], phase, boundary_role)
    row.update({
        "case_id": (f"adequacy-{parent['structure']}-p{parent_index}-r{profile_index}-"
                    f"{phase}-b{batch}-q{shape['query_length']}-kv{shape['kv_length']}"),
        "phase": phase,
        "shape": shape,
        "tensor_seed": seed,
        "tensor_spec_sha256": spec_hash,
        "parent_case_id": parent["case_id"],
        "derivation_kind": "pinned_real_architecture_plus_controlled_rq_counterfactual",
        "shape_source": "pinned real model dimensions; controlled batch/sequence boundary",
        "shape_counterfactual": True,
        "causal_role": boundary_role,
        "rq_counterfactuals": axes,
        "target_rqs": sorted(axes),
        "single_gpu_blocked_rqs": ["L-RQ5"],
        "precision_scope": "FP16 adequacy suite; v10 precision experiments are separate",
        "benchmark_policy": {
            "request_batch": batch,
            "boundary_role": boundary_role,
            "parent_selection": "diverse pinned real model architecture",
            "profile_is_controlled_counterfactual": True,
        },
    })
    row["contract_sha256"] = contract_hash(row)
    row["estimated_case_bytes"] = estimate_case_bytes(row)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases-per-structure", type=int, default=128)
    parser.add_argument("--max-parents", type=int, default=8)
    parser.add_argument("--max-parent-estimate-gib", type=float, default=12.0)
    args = parser.parse_args()
    if args.cases_per_structure < 64:
        parser.error("--cases-per-structure must be at least 64")
    source = load(args.source_manifest)
    prof = profiles()
    generated: list[dict] = []
    parent_summary: dict[str, list[str]] = {}
    for structure in STRUCTURES:
        candidates = [case for case in source
                      if case["structure"] == structure and case["phase"] == "prefill"
                      and estimate_case_bytes(case) <= args.max_parent_estimate_gib * 2**30]
        parents = diverse_parents(candidates, args.max_parents)
        if len(parents) < 2:
            raise RuntimeError(f"{structure}: fewer than two real parent architectures")
        parent_summary[structure] = [p["case_id"] for p in parents]
        if args.cases_per_structure > len(prof) * len(parents):
            raise RuntimeError(
                f"{structure}: {args.cases_per_structure} exceeds the unique "
                f"parent/profile capacity {len(prof) * len(parents)}")
        # Repeat the complete 64-profile block, shifting its parent assignment
        # on each pass.  At 128 cases this gives exactly 64 prefill/64 decode
        # and 32 cases per batch for every structure, including the two-parent
        # linear-attention catalog.
        for index in range(args.cases_per_structure):
            profile_index = index % len(prof)
            parent_index = (profile_index + index // len(prof)) % len(parents)
            generated.append(create(parents[parent_index], parent_index, profile_index,
                                    *prof[profile_index]))

    expected = len(STRUCTURES) * args.cases_per_structure
    ids = [case["case_id"] for case in generated]
    specs = [case["tensor_spec_sha256"] for case in generated]
    if len(generated) != expected or len(set(ids)) != expected:
        raise AssertionError("adequacy suite requires the expected number of unique case IDs")
    # Two different pinned models can legitimately expose the same executable
    # tensor contract.  Keep both provenance rows, but report the deduplicated
    # contract count so downstream statistics never treat them as independent
    # performance samples.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    full = args.output_dir / f"rq_adequacy_{expected}.json"
    full.write_text(json.dumps({"schema_version": 3, "case_count": expected,
                                "source_manifest": str(args.source_manifest.resolve()),
                                "cases": generated}, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    attention_count = write_attention_tsv(args.output_dir / "rq_adequacy_attention.tsv", generated)

    # 32 smoke cases: two per structure×phase, different parents/batches when
    # the parent catalog allows it.
    smoke: list[dict] = []
    for structure in STRUCTURES:
        for phase in ("prefill", "decode"):
            pool = [case for case in generated
                    if case["structure"] == structure and case["phase"] == phase]
            if structure in {"gqa", "sliding_attention", "sparse_attention", "mla"}:
                # CUTLASS/CuTe exact score-layout kernels require a vectorizable
                # K extent.  Use real-parent cases at exact 16/32/... boundaries
                # in smoke while full mode retains the ±1 counterexamples.
                pool = [case for case in pool
                        if int(case["shape"]["query_length" if phase == "prefill"
                                             else "kv_length"]) % 16 == 0]
            smoke.append(pool[0])
            smoke.append(next(case for case in pool
                              if case["parent_case_id"] != pool[0]["parent_case_id"]
                              and case["shape"]["batch"] != pool[0]["shape"]["batch"]))
    smoke_path = args.output_dir / "rq_adequacy_smoke_32.json"
    smoke_path.write_text(json.dumps({"schema_version": 3, "case_count": len(smoke),
                                      "cases": smoke}, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    smoke_attention = write_attention_tsv(
        args.output_dir / "rq_adequacy_attention_smoke.tsv", smoke)
    smoke_attention_families = {attention_family(case) for case in smoke
                                if attention_family(case)}
    base_attention_kinds = {value.removeprefix("sliding_")
                            for value in smoke_attention_families}
    required_base_kinds = {"MHA", "MQA", "GQA", "sparse_attention", "MLA"}
    family_coverage_ok = (required_base_kinds <= base_attention_kinds
                          and any(value.startswith("sliding_")
                                  for value in smoke_attention_families))
    if not family_coverage_ok:
        raise AssertionError(
            f"smoke attention coverage insufficient: {sorted(smoke_attention_families)}")
    summary = {
        "case_count": expected,
        "unique_tensor_contracts": len(set(specs)),
        "structures": dict(Counter(c["structure"] for c in generated)),
        "phases": dict(Counter(c["phase"] for c in generated)),
        "batches": dict(Counter(str(c["shape"]["batch"]) for c in generated)),
        "real_parent_counts": {key: len(value) for key, value in parent_summary.items()},
        "parent_cases": parent_summary,
        "attention_cases": attention_count,
        "smoke_cases": len(smoke),
        "smoke_attention_cases": smoke_attention,
        "smoke_attention_families": sorted(smoke_attention_families),
        "single_gpu_blocked": ["L-RQ5"],
    }
    (args.output_dir / "RQ_ADEQUACY_CASE_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
