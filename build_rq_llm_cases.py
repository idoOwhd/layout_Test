#!/usr/bin/env python3
"""Select multiple representative shapes per common LLM subgraph.

The source is the pinned 640-case manifest.  Selection is deterministic:
within every (structure, phase), deduplicate tensor contracts and keep the
minimum, median and maximum estimated work/footprint shapes.  The complete
selected manifest feeds the PyTorch/Triton external-validity run; the TSV is a
strict attention subset consumed by the controlled CUDA layout benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE.parent / "real_world_shapes" / "real_world_shape_manifest.json"


def load(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value["cases"] if isinstance(value, dict) else value


def score(case: dict) -> int:
    s = case["shape"]
    q = int(s.get("query_length", 1) or 1)
    kv = int(s.get("kv_length", q) or q)
    hidden = int(s.get("hidden_size", 1) or 1)
    structure = case["structure"]
    if structure in {"gqa", "sliding_attention", "sparse_attention", "mla"}:
        heads = int(s.get("num_kv_heads", 1) or 1)
        dim = int(s.get("head_dim", s.get("kv_lora_rank", hidden)) or hidden)
        return kv * heads * dim + q * hidden
    if structure == "swiglu":
        return q * hidden * int(s.get("intermediate_size", hidden) or hidden)
    if structure == "moe":
        return q * hidden * int(s.get("experts_per_token", 2) or 2)
    if structure in {"mamba2", "linear_attention"}:
        state = int(s.get("state_size", hidden) or hidden)
        return q * hidden * state
    return q * hidden


def choose(cases: list[dict], per_group: int) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for case in cases:
        groups.setdefault((case["structure"], case["phase"]), []).append(case)
    selected = []
    for key in sorted(groups):
        unique: dict[str, dict] = {}
        for case in groups[key]:
            unique.setdefault(case["tensor_spec_sha256"], case)
        ordered = sorted(unique.values(), key=lambda item: (score(item), item["case_id"]))
        if len(ordered) <= per_group:
            picks = ordered
        elif per_group == 1:
            picks = [ordered[len(ordered) // 2]]
        else:
            indices = sorted({round(i * (len(ordered) - 1) / (per_group - 1)) for i in range(per_group)})
            picks = [ordered[index] for index in indices]
        selected.extend(picks)
    return selected


def attention_kind(q_heads: int, kv_heads: int) -> str:
    if kv_heads == 1:
        return "MQA"
    if q_heads == kv_heads:
        return "MHA"
    return "GQA"


def choose_attention(cases: list[dict], per_group: int) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for case in cases:
        if case["structure"] not in {"gqa", "sliding_attention"}:
            continue
        shape = case["shape"]
        qh, kh = shape.get("num_query_heads"), shape.get("num_kv_heads")
        if not qh or not kh or not shape.get("head_dim"):
            continue
        key = (case["structure"], case["phase"], attention_kind(int(qh), int(kh)))
        groups.setdefault(key, []).append(case)
    selected = []
    for key in sorted(groups):
        unique = {case["tensor_spec_sha256"]: case for case in groups[key]}
        ordered = sorted(unique.values(), key=lambda item: (score(item), item["case_id"]))
        if len(ordered) <= per_group:
            selected.extend(ordered)
        elif per_group == 1:
            selected.append(ordered[len(ordered) // 2])
        else:
            indices = sorted({round(i * (len(ordered) - 1) / (per_group - 1)) for i in range(per_group)})
            selected.extend(ordered[index] for index in indices)
    return selected


def write_attention_tsv(path: Path, cases: list[dict]) -> int:
    lines = ["case_id\tmodel\tphase\tsubgraph\tattention_kind\ttokens\tquery_heads\tkv_heads\thead_dim"]
    count = 0
    for case in cases:
        if case["structure"] not in {"gqa", "sliding_attention"}:
            continue
        s = case["shape"]
        qh, kh = s.get("num_query_heads"), s.get("num_kv_heads")
        dim = s.get("head_dim")
        if not qh or not kh or not dim:
            continue
        # CUDA/Triton adapters model a flattened logical token domain rather
        # than a serving scheduler.  Preserve total work for B>1 explicitly;
        # request-aware page/slot topology is tested by the native KV runner.
        tokens_per_request = s["kv_length"] if case["phase"] == "decode" else s["query_length"]
        tokens = int(s.get("batch", 1)) * int(tokens_per_request)
        model = f"{case['model_id']}@{case['model_revision'][:12]}"
        values = [case["case_id"], model, case["phase"], case["structure"],
                  attention_kind(int(qh), int(kh)), tokens, qh, kh, dim]
        lines.append("\t".join(map(str, values)))
        count += 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--all-cases", action="store_true",
        help="emit all 640 source cases and the complete compatible attention TSV",
    )
    args = parser.parse_args()
    if args.quick and args.all_cases:
        parser.error("--quick and --all-cases are mutually exclusive")
    per_group = 1 if args.quick else args.per_group
    all_cases = load(args.manifest)
    selected = list(all_cases) if args.all_cases else choose(all_cases, per_group)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "rq_llm_all_640" if args.all_cases else "rq_llm_representative"
    manifest_path = args.output_dir / f"{prefix}_cases.json"
    manifest_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tsv_path = args.output_dir / ("rq_attention_all_cases.tsv" if args.all_cases
                                  else "rq_attention_cases.tsv")
    attention_cases = [case for case in all_cases
                       if case["structure"] in {"gqa", "sliding_attention"}] \
        if args.all_cases else choose_attention(all_cases, per_group)
    attention_count = write_attention_tsv(tsv_path, attention_cases)
    counts: dict[str, int] = {}
    for case in selected:
        key = f"{case['structure']}:{case['phase']}"
        counts[key] = counts.get(key, 0) + 1
    summary = {
        "source_manifest": str(args.manifest),
        "source_case_count": len(load(args.manifest)),
        "selection": ("all source cases" if args.all_cases else
                      "min/median/max distinct tensor contracts per structure and phase"),
        "per_group": per_group,
        "selected_case_count": len(selected),
        "attention_cuda_case_count": attention_count,
        "attention_kind_counts": {
            kind: sum(1 for case in attention_cases if attention_kind(
                int(case["shape"]["num_query_heads"]), int(case["shape"]["num_kv_heads"])) == kind)
            for kind in ("MHA", "MQA", "GQA")
        },
        "counts": counts,
        "case_ids": [case["case_id"] for case in selected],
    }
    summary_name = "rq_all_640_case_summary.json" if args.all_cases else "rq_case_summary.json"
    (args.output_dir / summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
