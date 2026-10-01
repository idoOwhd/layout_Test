#!/usr/bin/env python3
"""Build 256 real-world-derived LLM contracts over missing shape axes.

Architecture dimensions and provenance come from the pinned 640-case catalog.
Batch and sequence lengths are controlled serving-workload counterfactuals.
They are labelled as derived; they are not claims about a model's training
configuration.  The suite is balanced by structure, phase and request batch.
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
from layout_research.build_persuasive_real_world_cases import (
    DEFAULT_MANIFEST,
    FEATURES,
    RQ_SCOPE,
    STRUCTURES,
    diverse_parents,
    load,
)
from layout_research.build_rq_llm_cases import write_attention_tsv
from real_world_workloads import estimate_case_bytes


PROFILES = (
    # phase, workload length, request batch, page/tile boundary role
    ("prefill", 15, 1, "tile_minus_one"),
    ("prefill", 16, 2, "tile_exact"),
    ("prefill", 17, 4, "tile_plus_one"),
    ("prefill", 513, 8, "long_prefill_plus_one"),
    ("decode", 127, 1, "page_minus_one"),
    ("decode", 128, 2, "page_exact"),
    ("decode", 129, 4, "page_plus_one"),
    ("decode", 8192, 8, "long_decode"),
)

# The pinned catalog currently has only two distinct linear-attention parent
# architectures.  Do not invent two more models: use a denser workload sweep
# for those two real parents while keeping the same 32-case structure budget.
EXTENDED_PROFILES = (
    ("prefill", 7, 1, "half_tile_minus_one"),
    ("prefill", 8, 2, "half_tile_exact"),
    ("prefill", 15, 4, "tile_minus_one"),
    ("prefill", 16, 8, "tile_exact"),
    ("prefill", 17, 1, "tile_plus_one"),
    ("prefill", 33, 2, "two_tiles_plus_one"),
    ("prefill", 129, 4, "eight_tiles_plus_one"),
    ("prefill", 513, 8, "long_prefill_plus_one"),
    ("decode", 63, 1, "small_page_minus_one"),
    ("decode", 64, 2, "small_page_exact"),
    ("decode", 127, 4, "page_minus_one"),
    ("decode", 128, 8, "page_exact"),
    ("decode", 129, 1, "page_plus_one"),
    ("decode", 513, 2, "four_pages_plus_one"),
    ("decode", 2048, 4, "medium_decode"),
    ("decode", 8192, 8, "long_decode"),
)

BATCHED_KEYS = {
    "x", "q", "k_or_cache", "v_or_cache", "latent_kv", "output", "state",
    "topk_indices", "selected_k", "selected_v",
}


def batchify(shape: dict, batch: int) -> dict:
    shape = json.loads(json.dumps(shape))
    shape["batch"] = batch
    for key in BATCHED_KEYS:
        value = shape.get(key)
        if isinstance(value, list) and value and value[0] == 1:
            value[0] = batch
    tokens = batch * int(shape["query_length"])
    for key in ("gate_up", "router", "expert_output"):
        value = shape.get(key)
        if isinstance(value, list) and value:
            value[0] = tokens
    return shape


def rq_axes(structure: str, phase: str, role: str) -> dict[str, str]:
    axes = {
        "L-RQ1": "producer/consumer layout ranking over the same contract",
        "L-RQ2": "structure domain plus request-batch granularity",
        "L-RQ3": "direct/view/materialize and measured reuse counterfactual",
        "L-RQ4": "shape regime stability across tile/page boundaries",
        "L-RQ7": "stride legality and zero-copy/materialize boundary",
        "L-RQ9": "native versus extended layout candidate set",
    }
    if structure in {"gqa", "sliding_attention", "sparse_attention", "mla", "moe"}:
        axes["L-RQ8"] = "data plus index/route/page metadata co-layout"
    if structure in {"gqa", "sliding_attention", "sparse_attention", "mla"}:
        axes["L-RQ10"] = f"page/block allocator boundary: {role}"
    if structure in {"gqa", "sliding_attention", "sparse_attention", "mla",
                     "mamba2", "linear_attention"}:
        axes["L-RQ6"] = f"persistent state horizon in {phase}"
    # L-RQ5 deliberately absent: a shape cannot create a communication
    # intervention on one GPU.  The matrix must retain single-GPU-blocked.
    return axes


def create(parent: dict, parent_index: int, profile_index: int,
           phase: str, length: int, batch: int, role: str) -> dict:
    shape = (shape_payload(parent["structure"], parent["dimensions"], phase,
                           length, 8192 if phase == "prefill" else length))
    shape = batchify(shape, batch)
    seed, spec_hash = tensor_spec(parent["structure"], phase, shape)
    row = {key: parent[key] for key in (
        "model_id", "model_revision", "model_type", "structure",
        "structure_label", "dimensions", "graph_nodes", "source_url")}
    row.update({
        "case_id": (f"div2-{parent['structure']}-p{parent_index}-r{profile_index}-"
                    f"{phase}-b{batch}-q{shape['query_length']}-kv{shape['kv_length']}"),
        "phase": phase, "shape": shape, "tensor_seed": seed,
        "tensor_spec_sha256": spec_hash, "parent_case_id": parent["case_id"],
        "derivation_kind": "pinned_real_architecture_plus_controlled_shape_counterfactual",
        "shape_source": "pinned real model dimensions; controlled batch/sequence/page-boundary workload",
        "shape_counterfactual": True,
        "causal_role": role,
        "rq_counterfactuals": rq_axes(parent["structure"], phase, role),
        "target_rqs": sorted(rq_axes(parent["structure"], phase, role)),
        "single_gpu_blocked_rqs": ["L-RQ5"],
        "precision_scope": "FP16 executable shape suite; precision-specific v10 experiments remain separate",
        "benchmark_policy": {"request_batch": batch, "boundary_role": role,
                             "prefill_length": shape["query_length"] if phase == "prefill" else None,
                             "decode_kv_length": shape["kv_length"] if phase == "decode" else None},
    })
    row["contract_sha256"] = contract_hash(row)
    row["estimated_case_bytes"] = estimate_case_bytes(row)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-parent-estimate-gib", type=float, default=12.0)
    args = parser.parse_args()
    source = load(args.source_manifest)
    generated, parent_summary = [], {}
    for structure in STRUCTURES:
        candidates = [case for case in source if case["structure"] == structure
                      and case["phase"] == "prefill"
                      and estimate_case_bytes(case) <= args.max_parent_estimate_gib * 2**30]
        parents = diverse_parents(candidates, 4)
        if len(parents) not in (2, 4):
            raise RuntimeError(
                f"{structure}: need two or four distinct real architecture parents; got {len(parents)}")
        parent_summary[structure] = [parent["case_id"] for parent in parents]
        profiles = PROFILES if len(parents) == 4 else EXTENDED_PROFILES
        for parent_index, parent in enumerate(parents):
            for profile_index, profile in enumerate(profiles):
                generated.append(create(parent, parent_index, profile_index, *profile))

    ids = [case["case_id"] for case in generated]
    specs = [case["tensor_spec_sha256"] for case in generated]
    if len(generated) != 256 or len(set(ids)) != 256 or len(set(specs)) != 256:
        raise AssertionError("diversity-v2 requires 256 unique IDs and tensor contracts")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 2, "case_count": 256,
                "source_manifest": str(args.source_manifest.resolve()), "cases": generated}
    (args.output_dir / "llm_shape_diversity_256.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    attention = write_attention_tsv(args.output_dir / "llm_shape_diversity_attention.tsv", generated)
    smoke = []
    for structure in STRUCTURES:
        for phase in ("prefill", "decode"):
            wanted_batch = 2 if phase == "prefill" else 4
            smoke.append(next(case for case in generated
                              if case["structure"] == structure and case["phase"] == phase
                              and case["shape"]["batch"] == wanted_batch))
    (args.output_dir / "llm_shape_diversity_smoke_16.json").write_text(
        json.dumps({"schema_version": 2, "case_count": 16, "cases": smoke},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_attention_tsv(args.output_dir / "llm_shape_diversity_attention_smoke.tsv", smoke)
    summary = {
        "case_count": len(generated), "unique_tensor_contracts": len(set(specs)),
        "structures": dict(Counter(c["structure"] for c in generated)),
        "phases": dict(Counter(c["phase"] for c in generated)),
        "batches": dict(Counter(str(c["shape"]["batch"]) for c in generated)),
        "causal_roles": dict(Counter(c["causal_role"] for c in generated)),
        "attention_tsv_cases": attention, "parent_cases": parent_summary,
        "single_gpu_blocked": ["L-RQ5"],
    }
    (args.output_dir / "LLM_SHAPE_DIVERSITY_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
