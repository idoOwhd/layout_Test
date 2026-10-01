#!/usr/bin/env python3
"""Build preregistered v10 L-RQ1 workloads and experiment/RQ mapping.

The v10 documents contain one research question (L-RQ1) with five hypotheses.
This generator keeps Stage 0--3 separate from Stage 4 so the runner can enforce
the protocol's GO gate instead of executing sensitivity sweeps unconditionally.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from build_rq_llm_cases import DEFAULT_MANIFEST, choose_attention, load


HERE = Path(__file__).resolve().parent
REF = HERE / "ref_talks"
V10_MATRIX = REF / "layout_summary_v10_RQ1_experiment_matrix.csv"
V10_SCHEMA = REF / "layout_summary_v10_RQ1_measurement_schema.csv"

FIELDS = ["case_id", "stage", "subgraph", "source", "dtype", "batch", "q_len",
          "kv_len", "Hq", "Hkv", "head_dim", "page_size", "reuse_count"]


def case(case_id: str, stage: str, subgraph: str, source: str, dtype: str,
         batch: int, q_len: int, kv_len: int, hq: int, hkv: int,
         head_dim: int = 128, page_size: int = 64, reuse: int = 1) -> dict:
    if hq % hkv:
        raise ValueError(f"Hq must be divisible by Hkv: {hq}, {hkv}")
    return dict(zip(FIELDS, (case_id, stage, subgraph, source, dtype, batch, q_len,
                             kv_len, hq, hkv, head_dim, page_size, reuse)))


def core_cases(manifest: Path, per_group: int) -> list[dict]:
    rows = [
        case("v10-neg-bf16-small", "0-control", "kv_decode", "v10:C00+C01",
             "BF16", 1, 1, 512, 8, 8),
        case("v10-core-r1", "3-edge", "gqa_kv_decode", "v10:Stage3:Hq/Hkv=1",
             "BF16", 8, 1, 2048, 8, 8),
        case("v10-core-r8", "3-edge", "gqa_kv_decode", "v10:Stage3:Hq/Hkv=8",
             "BF16", 8, 1, 2048, 32, 4),
        case("v10-core-r32", "3-edge", "gqa_kv_decode", "v10:Stage3:Hq/Hkv=32",
             "BF16", 8, 1, 2048, 32, 1),
    ]
    # Explicit fanout/reuse is H1.2. KV length is deliberately held fixed.
    for reuse in (4, 16, 64):
        rows.append(case(f"v10-reuse-r8-x{reuse}", "3-edge-reuse", "gqa_kv_decode",
                         "v10:H1.2", "BF16", 8, 1, 2048, 32, 4, reuse=reuse))

    # Native NVFP4 is preregistered even on unsupported GPUs. The CUDA program
    # emits explicit unsupported rows rather than substituting FP16/BF16.
    for ratio, hq, hkv in ((1, 8, 8), (8, 32, 4), (32, 32, 1)):
        rows.append(case(f"v10-core-nvfp4-r{ratio}", "3-edge", "gqa_kv_decode",
                         f"v10:Stage3:NVFP4:Hq/Hkv={ratio}", "NVFP4", 8, 1,
                         2048, hq, hkv))

    # Real-model decode shapes give multiple common contracts without changing
    # the controlled edge semantics. Select min/median/max per attention kind.
    selected = choose_attention(load(manifest), per_group)
    decode = [item for item in selected if item["phase"] == "decode"]
    seen = set()
    for item in decode:
        shape = item["shape"]
        key = (shape["kv_length"], shape["num_query_heads"],
               shape["num_kv_heads"], shape["head_dim"], item["structure"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(case(
            f"v10-real-{item['case_id']}", "3-edge-real-shape", item["structure"],
            f"{item['model_id']}@{item['model_revision'][:12]}", "BF16",
            int(shape.get("batch", 1)), 1, int(shape["kv_length"]),
            int(shape["num_query_heads"]), int(shape["num_kv_heads"]),
            int(shape["head_dim"]), 64, 1))
    return rows


def sensitivity_cases() -> list[dict]:
    center = dict(dtype="BF16", batch=8, q_len=1, kv_len=2048,
                  hq=32, hkv=4, head_dim=128, page_size=64)
    rows = []
    for value in (1, 8, 32, 64):
        rows.append(case(f"v10-s-batch-{value}", "4-sensitivity", "gqa_kv_decode",
                         "v10:S-BATCH", center["dtype"], value, center["q_len"],
                         center["kv_len"], center["hq"], center["hkv"],
                         center["head_dim"], center["page_size"]))
    for value in (128, 512, 2048, 8192, 32768):
        rows.append(case(f"v10-s-kv-{value}", "4-sensitivity", "gqa_kv_decode",
                         "v10:S-KVLEN", center["dtype"], center["batch"],
                         center["q_len"], value, center["hq"], center["hkv"],
                         center["head_dim"], center["page_size"]))
    for ratio, hq, hkv in ((1, 8, 8), (4, 32, 8), (8, 32, 4),
                           (16, 32, 2), (32, 32, 1)):
        rows.append(case(f"v10-s-gqa-{ratio}", "4-sensitivity", "gqa_kv_decode",
                         "v10:S-GQA", center["dtype"], center["batch"],
                         center["q_len"], center["kv_len"], hq, hkv,
                         center["head_dim"], center["page_size"]))
    for value in (1, 4, 8):
        rows.append(case(f"v10-s-qlen-{value}", "4-sensitivity", "gqa_kv_decode",
                         "v10:S-QLEN", center["dtype"], center["batch"], value,
                         center["kv_len"], center["hq"], center["hkv"],
                         center["head_dim"], center["page_size"]))
    for value in (16, 64, 128):
        rows.append(case(f"v10-s-page-{value}", "4-sensitivity", "gqa_kv_decode",
                         "v10:S-PAGE", center["dtype"], center["batch"],
                         center["q_len"], center["kv_len"], center["hq"],
                         center["hkv"], center["head_dim"], value))
    return rows


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with V10_MATRIX.open(encoding="utf-8", newline="") as handle:
        matrix = list(csv.DictReader(handle))
    with V10_SCHEMA.open(encoding="utf-8", newline="") as handle:
        schema = list(csv.DictReader(handle))
    core = core_cases(args.manifest, 1 if args.quick else 3)
    sensitivity = sensitivity_cases()
    if args.quick:
        sensitivity = [row for row in sensitivity if row["case_id"] in {
            "v10-s-batch-8", "v10-s-kv-2048", "v10-s-gqa-8",
            "v10-s-qlen-4", "v10-s-page-64"}]
    write_tsv(args.output_dir / "v10_rq1_core_cases.tsv", core)
    write_tsv(args.output_dir / "v10_rq1_sensitivity_cases.tsv", sensitivity)
    mapping = {
        "L-RQ1": {
            "question": "When does producer-local layout differ from producer-to-consumer edge-optimal layout?",
            "experiments": {
                "H1.1": ["Stage1 producer", "Stage2 consumer", "Stage3 NN/NH-copy/HH-native/HN-copy"],
                "H1.2": ["Stage3 explicit reuse_count=1/4/16/64"],
                "H1.3": ["Stage3 NH-view versus NH-copy and HH-native"],
                "H1.4": ["Stage3 HH-native versus NH-copy"],
                "H1.NEG": ["Stage0 BF16 batch=1 q_len=1 kv_len=512 ratio=1"],
                "crossover": ["Stage4 BATCH/KVLEN/GQA/QLEN/PAGE; gated on H1.1"],
                "external_validity": ["real-model GQA/sliding-attention shapes", "conditional all-subgraph boundary sweep"],
            },
        },
        "v10_matrix_rows": len(matrix),
        "v10_required_measurement_fields": [row["field"] for row in schema if row["status"] == "required"],
        "core_case_count": len(core),
        "sensitivity_case_count": len(sensitivity),
        "core_case_ids": [row["case_id"] for row in core],
    }
    (args.output_dir / "v10_rq1_experiment_map.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(mapping, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
