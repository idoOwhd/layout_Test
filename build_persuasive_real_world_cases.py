#!/usr/bin/env python3
"""Build 128 discriminative contracts derived from pinned real model configs.

The original 640 catalog fixes prefill at 2048 and mostly fixes decode KV at
8192.  This supplement preserves each selected model's pinned architecture
dimensions while sweeping workload lengths that are causal for layout choices.
Every generated row records its parent catalog case and remains auditable with
the same contract/tensor hash scheme.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))

from discover_real_world_shapes import (actual_shape_reasons, contract_hash,
                                        shape_payload, tensor_spec)
from real_world_workloads import estimate_case_bytes
from layout_research.build_rq_llm_cases import write_attention_tsv


DEFAULT_MANIFEST = BASELINE / "real_world_shapes/real_world_shape_manifest.json"
STRUCTURES = ("gqa", "sliding_attention", "sparse_attention", "mla",
              "swiglu", "moe", "mamba2", "linear_attention")
STATEFUL_ATTENTION = {"gqa", "sliding_attention", "sparse_attention", "mla"}
RQ_SCOPE = {
    "L-RQ1": set(STRUCTURES),
    "L-RQ2": {"gqa", "sliding_attention", "sparse_attention", "mla", "moe"},
    "L-RQ3": set(STRUCTURES),
    "L-RQ4": set(STRUCTURES),
    "L-RQ5": {"gqa", "sliding_attention", "sparse_attention", "mla", "moe"},
    "L-RQ6": {"gqa", "sliding_attention", "sparse_attention", "mla",
               "mamba2", "linear_attention"},
    "L-RQ7": set(STRUCTURES),
    "L-RQ8": {"gqa", "sliding_attention", "sparse_attention", "mla", "moe"},
    "L-RQ9": set(STRUCTURES),
    "L-RQ10": {"gqa", "sliding_attention", "sparse_attention", "mla"},
}


FEATURES = {
    "gqa": ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim"),
    "sliding_attention": ("hidden_size", "num_attention_heads", "num_key_value_heads",
                          "head_dim", "sliding_window"),
    "sparse_attention": ("hidden_size", "num_attention_heads", "head_dim",
                         "value_head_dim", "index_topk", "index_head_dim"),
    "mla": ("hidden_size", "num_attention_heads", "kv_lora_rank",
            "qk_nope_head_dim", "qk_rope_head_dim", "value_head_dim"),
    "swiglu": ("hidden_size", "intermediate_size"),
    "moe": ("hidden_size", "num_experts", "experts_per_token", "moe_intermediate_size"),
    "mamba2": ("hidden_size", "mamba_num_heads", "state_size"),
    "linear_attention": ("hidden_size", "linear_num_key_heads", "linear_num_value_heads",
                         "linear_key_head_dim", "linear_value_head_dim"),
}


def load(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value["cases"] if isinstance(value, dict) else value


def feature(case: dict) -> tuple[float, ...]:
    dims = case["dimensions"]
    values = []
    for key in FEATURES[case["structure"]]:
        raw = dims.get(key)
        values.append(math.log2(max(float(raw or 1), 1.0)))
    return tuple(values)


def diverse_parents(cases: list[dict], count: int = 4) -> list[dict]:
    """Greedy farthest-point selection in normalized architecture space."""
    unique = {}
    for case in cases:
        unique.setdefault(feature(case), case)
    candidates = sorted(unique.values(), key=lambda case: case["case_id"])
    vectors = [feature(case) for case in candidates]
    dimensions = len(vectors[0])
    mins = [min(vector[i] for vector in vectors) for i in range(dimensions)]
    maxs = [max(vector[i] for vector in vectors) for i in range(dimensions)]
    normalized = [tuple((vector[i] - mins[i]) / max(maxs[i] - mins[i], 1e-9)
                        for i in range(dimensions)) for vector in vectors]
    selected = [0]
    if len(candidates) > 1:
        selected.append(max(range(len(candidates)),
                            key=lambda index: sum(value * value
                                                  for value in normalized[index])))
    while len(selected) < min(count, len(candidates)):
        remaining = [index for index in range(len(candidates)) if index not in selected]
        selected.append(max(remaining, key=lambda index: min(
            sum((normalized[index][axis] - normalized[chosen][axis]) ** 2
                for axis in range(dimensions)) for chosen in selected)))
    return [candidates[index] for index in selected]


def regimes(structure: str, parent_count: int) -> list[tuple[str, int]]:
    if structure == "sparse_attention":
        full = [("prefill", 64), ("prefill", 256),
                ("decode", 512), ("decode", 16384),
                ("prefill", 128), ("prefill", 512),
                ("decode", 2048), ("decode", 32768)]
    elif structure in STATEFUL_ATTENTION:
        full = [("prefill", 128), ("prefill", 1024),
                ("decode", 512), ("decode", 16384),
                ("prefill", 64), ("prefill", 512),
                ("decode", 2048), ("decode", 32768)]
    elif structure == "moe":
        full = [("prefill", 16), ("prefill", 64),
                ("prefill", 256), ("prefill", 512),
                ("prefill", 128), ("prefill", 1024),
                ("decode", 512), ("decode", 16384)]
    else:
        full = [("prefill", 32), ("prefill", 128),
                ("prefill", 512), ("prefill", 1024),
                ("prefill", 64), ("prefill", 256),
                ("decode", 512), ("decode", 16384)]
    return full


def axes(case: dict) -> list[str]:
    structure = case["structure"]
    result = ["short_vs_long_sequence", "prefill_decode_or_recurrence_horizon",
              "layout_stride_and_materialization"]
    if structure in {"gqa", "sliding_attention"}:
        result += ["query_to_kv_head_ratio", "head_dim", "kv_cache_length"]
    elif structure == "sparse_attention":
        result += ["selected_token_ratio", "index_metadata", "kv_cache_length"]
    elif structure == "mla":
        result += ["latent_rank", "qk_value_width", "latent_cache_length"]
    elif structure == "moe":
        result += ["expert_count", "experts_per_token", "route_metadata"]
    elif structure == "mamba2":
        result += ["state_size", "scan_horizon"]
    elif structure == "linear_attention":
        result += ["state_matrix_size", "key_value_head_asymmetry", "scan_horizon"]
    elif structure == "swiglu":
        result += ["hidden_to_intermediate_ratio", "token_batching"]
    return result


def create_case(parent: dict, parent_index: int, phase: str, length: int) -> dict:
    if phase == "prefill":
        shape = shape_payload(parent["structure"], parent["dimensions"], phase,
                              length, 16384)
    else:
        shape = shape_payload(parent["structure"], parent["dimensions"], phase,
                              1024, length)
    tensor_seed, tensor_hash = tensor_spec(parent["structure"], phase, shape)
    q, kv = shape["query_length"], shape["kv_length"]
    row = {key: parent[key] for key in (
        "model_id", "model_revision", "model_type", "structure",
        "structure_label", "dimensions", "graph_nodes", "source_url")}
    row.update({
        "case_id": f"persuasive-{parent['structure']}-p{parent_index}-{phase}-q{q}-kv{kv}",
        "phase": phase,
        "shape": shape,
        "tensor_seed": tensor_seed,
        "tensor_spec_sha256": tensor_hash,
        "parent_case_id": parent["case_id"],
        "shape_source": "architecture dimensions from pinned real-world parent config; workload length from RQ-discriminative supplement policy",
        "benchmark_policy": {"prefill_length": q if phase == "prefill" else None,
                             "decode_kv_length": kv if phase == "decode" else None},
        "derivation_kind": "pinned_real_model_architecture_plus_controlled_workload_length",
        "discriminative_axes": axes(parent),
        "target_rqs": [rq for rq, structures in RQ_SCOPE.items()
                       if parent["structure"] in structures],
        "v10_target": parent["structure"] in {"gqa", "sliding_attention"},
    })
    reasons = sorted(set(parent.get("large_shape_reasons", []) + actual_shape_reasons(shape)))
    row["large_shape_reasons"] = reasons
    row["size_class"] = "large" if reasons else "regular"
    row["contract_sha256"] = contract_hash(row)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases-per-structure", type=int, default=16)
    parser.add_argument("--max-parent-estimate-gib", type=float, default=12.0)
    args = parser.parse_args()
    if args.cases_per_structure < 16:
        parser.error("--cases-per-structure must be >=16")
    source = load(args.source_manifest)
    original_hashes = {case["tensor_spec_sha256"] for case in source}
    generated = []
    parent_summary = {}
    for structure in STRUCTURES:
        candidates = [case for case in source
                      if case["structure"] == structure and case["phase"] == "prefill"
                      and estimate_case_bytes(case) <= args.max_parent_estimate_gib * 2**30]
        if not candidates:
            raise RuntimeError(f"no A10-feasible real-world parents for {structure}")
        parents = diverse_parents(candidates, 4)
        current = []
        schedule = regimes(structure, len(parents))
        for phase, length in schedule:
            for parent_index, parent in enumerate(parents):
                row = create_case(parent, parent_index, phase, length)
                if row["tensor_spec_sha256"] not in original_hashes:
                    current.append(row)
        # Round-robin construction normally gives exactly 16.  If context
        # clamping created a duplicate, draw additional declared regimes.
        unique = {}
        for row in current:
            unique.setdefault(row["tensor_spec_sha256"], row)
        current = list(unique.values())
        if len(current) < args.cases_per_structure:
            raise RuntimeError(
                f"{structure}: only {len(current)} new contracts after context clamping; "
                f"need {args.cases_per_structure}")
        current = current[:args.cases_per_structure]
        generated.extend(current)
        parent_summary[structure] = [{"case_id": parent["case_id"],
                                      "model_id": parent["model_id"],
                                      "revision": parent["model_revision"],
                                      "features": feature(parent)} for parent in parents]

    ids = [case["case_id"] for case in generated]
    hashes = [case["tensor_spec_sha256"] for case in generated]
    if len(generated) < 128 or len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)):
        raise AssertionError("supplement must contain >=128 unique case IDs and tensor contracts")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = args.output_dir / "persuasive_real_world_128.json"
    manifest.write_text(json.dumps({"schema_version": 1,
                                    "source_manifest": str(args.source_manifest.resolve()),
                                    "case_count": len(generated),
                                    "cases": generated},
                                   ensure_ascii=False, indent=2) + "\n")
    attention_count = write_attention_tsv(args.output_dir / "persuasive_attention.tsv", generated)
    # One deterministic case per structure is kept beside the full manifest so
    # the launcher can test every implementation path before the long run.
    smoke = [next(case for case in generated if case["structure"] == structure)
             for structure in STRUCTURES]
    (args.output_dir / "persuasive_smoke_8.json").write_text(
        json.dumps({"schema_version": 1, "case_count": len(smoke), "cases": smoke},
                   ensure_ascii=False, indent=2) + "\n")
    write_attention_tsv(args.output_dir / "persuasive_attention_smoke.tsv", smoke)
    summary = {
        "case_count": len(generated),
        "unique_tensor_contracts": len(set(hashes)),
        "overlap_with_original_640_tensor_contracts": len(set(hashes) & original_hashes),
        "cases_by_structure": dict(Counter(case["structure"] for case in generated)),
        "cases_by_phase": dict(Counter(case["phase"] for case in generated)),
        "attention_tsv_cases": attention_count,
        "parent_models": parent_summary,
        "real_world_definition": "architecture dimensions and source provenance come from pinned parent configs; workload lengths are controlled RQ axes",
    }
    (args.output_dir / "PERSUASIVE_CASES_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
