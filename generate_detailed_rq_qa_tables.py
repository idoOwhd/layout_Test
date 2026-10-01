#!/usr/bin/env python3
"""Make compact, per-contract review tables from the 2026-09-27 full run.

This is a read-only reduction of the frozen result ledgers.  It does not
reinterpret correctness flags or combine the smoke/repair runs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results/rq_adequacy_full_20260927_113444"
REVIEW = RUN / "review_20260928"
OUT = REVIEW / "detailed_qa_20260929"
OUT.mkdir(exist_ok=True)


def rows(name: str):
    with (REVIEW / name).open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def write(name: str, records: list[dict], fields: list[str]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def shape_fields(row: dict) -> dict:
    shape = json.loads(row["shape"]) if row.get("shape") else {}
    return {
        "batch": shape.get("batch", ""),
        "q_len": shape.get("query_length", ""),
        "kv_len": shape.get("kv_length", ""),
        "hidden": shape.get("hidden_size", ""),
        "q_heads": shape.get("num_query_heads", shape.get("num_heads", "")),
        "kv_heads": shape.get("num_kv_heads", ""),
        "head_dim": shape.get("head_dim", shape.get("qk_head_dim", "")),
        "intermediate": shape.get("intermediate_size", ""),
        "experts": shape.get("num_experts", ""),
        "selected_experts": shape.get("experts_per_token", ""),
        "state_size": shape.get("state_size", ""),
    }


BASE = ["case_id", "model_id", "subgraph", "phase", "batch", "q_len",
        "kv_len", "hidden", "q_heads", "kv_heads", "head_dim",
        "intermediate", "experts", "selected_experts", "state_size", "shape"]

paired = list(rows("RQ_ALL_PAIRED_COMPARISONS.csv"))
verdicts = list(rows("RQ_ALL_CASE_VERDICTS_POSITIVE_NEGATIVE.csv"))
catalog = {row["case_id"]: row for row in rows("ALL_1024_CASE_CATALOG_AND_BROAD_RESULTS.csv")}


def raw_pair(case_id: str, rq: str, framework: str, contract: str,
             candidates: dict, evidence: str, note: str, **extra) -> dict:
    info = catalog[case_id]
    times = {name: float(row["p50_ms"]) for name, row in candidates.items()}
    correct = {name: row["correct"].lower() == "true" for name, row in candidates.items()}
    ranked = sorted(times, key=times.get)
    valid = all(correct.values()) and len(times) >= 2
    margin = times[ranked[1]] / times[ranked[0]] if len(times) >= 2 else 1.0
    return {**info, "rq": rq, "framework": framework, "contract": contract,
            "candidate_p50_ms": json.dumps(times), "raw_winner": ranked[0],
            "strict_winner_3pct": (ranked[0] if margin >= 1.03 else "tie")
            if valid else "incomplete_or_incorrect",
            "best_p50_ms": times[ranked[0]], "winner_margin": margin,
            "valid": valid, "evidence_kind": evidence, "note": note,
            "candidate_correctness": json.dumps(correct), **extra}


# Rebuild TVM contracts from raw: the older review merged row and column
# consumers, misclassified weighted fanout as RQ4, and compared two different
# primitives (square vs layout conversion) as though they were alternatives.
paired = [row for row in paired if row["framework"] not in {"TVM", "Triton-explicit"}]
tvm_groups = defaultdict(dict)
tvm_primitives = []
with (RUN / "raw/tvm_projected.csv").open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        case_id = row["manifest_case_id"]
        if row["experiment"] == "primitive":
            info = catalog[case_id]
            tvm_primitives.append({**info, **shape_fields(info),
                                   "original_boundary_id": row["case_id"],
                                   "projected_rows": row["rows"], "projected_cols": row["cols"],
                                   "operation": row["strategy"], "p50_ms": row["p50_ms"],
                                   "recorded_correct": row["correct"],
                                   "note": "square correctness is hard-coded; distinct operations are not paired alternatives"})
            continue
        key = (case_id, row["case_id"], row["experiment"], row["reuse"],
               row["fanout_row"], row["fanout_column"])
        tvm_groups[key][row["strategy"]] = row
for (case_id, boundary, experiment, reuse, fanout_row, fanout_col), values in tvm_groups.items():
    rq = {"consumer_scan": "L-RQ1;L-RQ4;L-RQ7;L-RQ9",
          "weighted_multi_consumer_pipeline": "L-RQ2",
          "conversion_reuse": "L-RQ3", "fusion_order": "L-RQ3;L-RQ4"}[experiment]
    sample = next(iter(values.values()))
    axis = "row" if boundary.endswith("_row_consumer") else "column" if boundary.endswith("_column_consumer") else ""
    paired.append(raw_pair(case_id, rq, "TVM",
                           f"experiment={experiment};axis={axis};reuse={reuse};fanout={fanout_row},{fanout_col}",
                           values, "measured_component_cost_model" if experiment == "conversion_reuse"
                           else "mixed_direct_and_component_cost_model" if experiment == "fusion_order"
                           else "direct_consumer_only" if experiment == "weighted_multi_consumer_pipeline"
                           else "direct_projected_primitive",
                           "equal projected geometry measurements are reused; no original subgraph; fanout excludes producer/materialization",
                           original_boundary_id=boundary, projected_rows=sample["rows"], projected_cols=sample["cols"]))
write("TVM_all_primitive_measurements.csv", tvm_primitives,
      BASE + ["original_boundary_id", "projected_rows", "projected_cols", "operation",
              "p50_ms", "recorded_correct", "operators", "note"])

# Preserve all explicit Triton candidates, including failed correctness flags,
# and add the previously omitted seven weighted fanout contracts.
triton_groups = defaultdict(dict)
with (RUN / "raw/triton_explicit.csv").open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        if row["experiment"] == "producer_consumer_edge":
            value = dict(row, p50_ms=row["pipeline_p50_ms"])
            key = (row["case_id"], "L-RQ1;L-RQ9", "producer_plus_consumer_pipeline")
        else:
            value = row
            key = (row["case_id"], "L-RQ2",
                   f"fanout_head={row['fanout_head']};fanout_token={row['fanout_token']}")
        triton_groups[key][row["layout"]] = value
for (case_id, rq, contract), values in triton_groups.items():
    paired.append(raw_pair(case_id, rq, "Triton-explicit", contract, values,
                           "direct_runtime", "paged_NHD aliases NHD for valid tokens; padding verification can falsely fail; split candidate absent"))

for number in range(1, 11):
    selected = []
    for row in paired:
        if f"L-RQ{number}" not in row["rq"].split(";"):
            continue
        selected.append({**row, **shape_fields(row)})
    if selected:
        write(f"RQ{number}_all_paired_framework_contracts.csv", selected,
              BASE + ["framework", "contract", "candidate_p50_ms", "raw_winner",
                      "strict_winner_3pct", "best_p50_ms", "winner_margin",
                      "valid", "candidate_correctness", "original_boundary_id",
                      "projected_rows", "projected_cols", "evidence_kind", "operators", "note"])

for number in range(1, 11):
    selected = []
    for row in verdicts:
        if row["rq"] == f"L-RQ{number}":
            selected.append({**row, **shape_fields(row)})
    if selected:
        write(f"RQ{number}_all_verdicts.csv", selected,
              BASE + ["question", "verdict", "framework", "scope", "details",
                      "evidence_kind", "operators", "note"])
        notable = [row for row in selected if row["verdict"] in
                   {"positive", "positive_zero_copy", "counterexample_materialize_faster"}]
        if notable:
            write(f"RQ{number}_notable_cases.csv", notable,
                  BASE + ["question", "verdict", "framework", "scope", "details",
                          "evidence_kind", "operators", "note"])

# RQ1: one compact row per explicit Triton KV producer/consumer pair.
triton = defaultdict(dict)
with (RUN / "raw/triton_explicit.csv").open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        if row["experiment"] == "producer_consumer_edge":
            triton[row["case_id"]][row["layout"]] = row
triton_out = []
for case_id, candidates in sorted(triton.items()):
    valid = {name: row for name, row in candidates.items() if row["correct"] == "True"}
    if not valid:
        continue
    producer = min(valid, key=lambda name: float(valid[name]["producer_p50_ms"]))
    edge = min(valid, key=lambda name: float(valid[name]["pipeline_p50_ms"]))
    regret = float(valid[producer]["pipeline_p50_ms"]) / float(valid[edge]["pipeline_p50_ms"])
    info = catalog[case_id]
    triton_out.append({"case_id": case_id, "model_id": info["model_id"],
                       "subgraph": info["subgraph"], "phase": info["phase"],
                       **shape_fields(info), "producer_choice": producer,
                       "shape": info["shape"],
                       "edge_choice": edge, "edge_regret": regret,
                       "ge_3pct": regret >= 1.03,
                       "producer_ms": json.dumps({n: float(r["producer_p50_ms"]) for n, r in candidates.items()}),
                       "pipeline_ms": json.dumps({n: float(r["pipeline_p50_ms"]) for n, r in candidates.items()}),
                       "correct": json.dumps({n: r["correct"] for n, r in candidates.items()}),
                       "operators": info["operators"]})
write("RQ1_Triton_producer_vs_pipeline_256.csv", triton_out,
      BASE + ["producer_choice", "edge_choice", "edge_regret", "ge_3pct",
              "producer_ms", "pipeline_ms", "correct", "operators"])
write("RQ1_Triton_regret_positive_77.csv", [r for r in triton_out if r["ge_3pct"]],
      BASE + ["producer_choice", "edge_choice", "edge_regret", "ge_3pct",
              "producer_ms", "pipeline_ms", "correct", "operators"])

# RQ10: one row per case and route/table condition, including all eight pages.
page_rows = []
page_samples = defaultdict(list)
page_wins = Counter()
page_wins_strict = Counter()
for row in paired:
    if row["rq"] != "L-RQ8;L-RQ10" or row["framework"] != "CUDA-reference":
        continue
    candidates = json.loads(row["candidate_p50_ms"])
    page_values = {int(name.split("_")[1]): float(value)
                   for name, value in candidates.items()}
    best = min(page_values.values())
    winner = min(page_values, key=page_values.get)
    page_wins[winner] += 1
    if float(page_values[128]) / best >= 1.03:
        page_wins_strict[winner] += 1
    for page, value in page_values.items():
        page_samples[page].append(value)
    shape = shape_fields(row)
    page_rows.append({**row, **shape,
                      **{f"page_{page}_ms": page_values[page]
                         for page in (1, 4, 8, 16, 32, 64, 128, 256)},
                      "best_page": winner, "best_ms": best,
                      "page128_over_best": page_values[128] / best,
                      "page128_loss_ge_3pct": page_values[128] / best >= 1.03,
                      "flattened_allocated_over_live_by_page": json.dumps(
                          {page: ((int(shape["batch"]) * int(shape["kv_len"]) + page - 1) // page * page)
                           / (int(shape["batch"]) * int(shape["kv_len"]))
                           for page in (1, 4, 8, 16, 32, 64, 128, 256)}),
                      "per_request_allocated_over_live_by_page": json.dumps(
                          {page: ((int(shape["kv_len"]) + page - 1) // page * page)
                           / int(shape["kv_len"])
                           for page in (1, 4, 8, 16, 32, 64, 128, 256)})})
write("RQ10_all_2304_page_contracts.csv", page_rows,
      BASE + ["contract"] + [f"page_{page}_ms" for page in (1, 4, 8, 16, 32, 64, 128, 256)]
      + ["best_page", "best_ms", "page128_over_best", "page128_loss_ge_3pct",
         "flattened_allocated_over_live_by_page",
         "per_request_allocated_over_live_by_page", "valid", "operators"])
page_summary = [{"page": page, "contracts": len(values),
                 "mean_ms": statistics.mean(values),
                 "median_ms": statistics.median(values),
                 "min_ms": min(values), "max_ms": max(values),
                 "raw_wins": page_wins[page],
                 "wins_when_128_loses_3pct": page_wins_strict[page]}
                for page, values in sorted(page_samples.items())]
write("RQ10_page_size_summary.csv", page_summary,
      ["page", "contracts", "mean_ms", "median_ms", "min_ms", "max_ms",
       "raw_wins", "wins_when_128_loses_3pct"])
write("RQ10_page128_loss_464.csv", [r for r in page_rows if r["page128_loss_ge_3pct"]],
      BASE + ["contract"] + [f"page_{p}_ms" for p in (1, 4, 8, 16, 32, 64, 128, 256)]
      + ["best_page", "best_ms", "page128_over_best", "valid", "operators"])

# CUDA shape-size observations compare like with like across all 256 cases.
cuda = defaultdict(dict)
for row in paired:
    if row["framework"] == "CUDA-reference" and row["rq"] == "L-RQ1":
        cuda[row["case_id"]][row["contract"]] = json.loads(row["candidate_p50_ms"])
    if row["framework"] == "CUDA-reference" and row["rq"] == "L-RQ3" and row["contract"] == "reuse=64":
        cuda[row["case_id"]]["reuse64"] = json.loads(row["candidate_p50_ms"])
conversion = defaultdict(list)
with (RUN / "raw/cuda_reference.csv").open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        if row["benchmark"] == "layout_conversion" and row["strategy"] == "NHD_to_HND":
            conversion[row["case_id"]].append(float(row["p50_ms"]))
shape_rows = []
for case_id, tests in sorted(cuda.items()):
    if len(conversion[case_id]) != 5:
        continue
    info = catalog[case_id]
    s = shape_fields(info)
    size = int(s["batch"]) * int(s["kv_len"]) * int(s["kv_heads"]) * int(s["head_dim"]) * 2
    head = tests["decode_head_scan"]
    token = tests["token_major_scan"]
    edge = tests["producer_to_head_edge_reuse=1"]
    r64 = tests["reuse64"]
    shape_rows.append({"case_id": case_id, "model_id": info["model_id"],
                       "subgraph": info["subgraph"], "phase": info["phase"], **s,
                       "shape": info["shape"],
                       "logical_kv_bytes": size,
                       "NHD_to_HND_conversion_ms": statistics.median(conversion[case_id]),
                       "head_NHD_over_HND": head["NHD"] / head["HND"],
                       "token_HND_over_NHD": token["HND"] / token["NHD"],
                       "edge_common_NHD_over_best": edge["common_NHD"] / min(edge.values()),
                       "reuse64_convert_over_best": r64["NHD_then_convert_once_HND"] / min(r64.values()),
                       "operators": info["operators"]})
write("CUDA_256_shape_conversion_and_locality.csv", shape_rows,
      BASE + ["logical_kv_bytes", "NHD_to_HND_conversion_ms",
              "head_NHD_over_HND", "token_HND_over_NHD",
              "edge_common_NHD_over_best", "reuse64_convert_over_best", "operators"])

# RQ4 parent-level crossover verdicts use original parent IDs, so join them
# back to all controlled child shapes and their complete-subgraph timings.
manifest = json.loads((RUN / "cases/executed_manifest.json").read_text(encoding="utf-8"))["cases"]
parent_of = {case["case_id"]: case["parent_case_id"] for case in manifest}
positive_parents = {row["case_id"] for row in verdicts
                    if row["rq"] == "L-RQ4" and row["verdict"] == "positive"
                    and row["question"].startswith("同一父模型")}
parent_children = []
whole_pairs = {row["case_id"]: row for row in paired
               if row["rq"] == "L-RQ4" and row["framework"] == "PyTorch-eager_vs_TorchInductor"}
for case in manifest:
    case_id = case["case_id"]
    parent_id = parent_of[case_id]
    if parent_id not in positive_parents:
        continue
    info = catalog[case_id]
    row = whole_pairs.get(case_id, {
        "candidate_p50_ms": json.dumps({name: float(info[field]) for name, field in
                                        (("pytorch", "whole_pytorch_ms"), ("triton", "whole_torchinductor_ms"))
                                        if info[field]}),
        "raw_winner": "", "strict_winner_3pct": "incomplete_or_incorrect"})
    parent_children.append({**info, **row, **shape_fields(info), "parent_case_id": parent_id})
write("RQ4_six_parent_crossovers_all_children.csv", parent_children,
      ["parent_case_id"] + BASE + ["candidate_p50_ms", "raw_winner",
                                  "strict_winner_3pct", "whole_pytorch_status",
                                  "whole_torchinductor_status", "operators"])

triplets = []
dynamic = []
for row in verdicts:
    if row["rq"] == "L-RQ7":
        values = json.loads(row["details"])["candidate_p50_ms"]
        native, view, repair = (values[name] for name in
                                ("native_contiguous", "alternate_strided_view",
                                 "repair_to_contiguous_each_call"))
        ranked = sorted(values, key=values.get)
        triplets.append({**row, **shape_fields(row), "native_ms": native,
                          "view_ms": view, "repair_ms": repair,
                          "raw_winner": ranked[0],
                          "native_over_view": native / view,
                          "native_over_repair": native / repair,
                          "view_lt_0_97_native": view < .97 * native,
                          "repair_lt_0_97_native": repair < .97 * native,
                          "runner_up_over_best": values[ranked[1]] / values[ranked[0]]})
    if row["rq"] == "L-RQ6" and row["question"].startswith("最佳动态"):
        values = json.loads(row["details"])["candidate_p50_ms"]
        fixed = min(values[name] for name in ("fixed_NHD", "fixed_HND"))
        dyn = min(values[name] for name in ("eager_switch", "hysteresis_2", "hysteresis_4"))
        dynamic.append({**row, **shape_fields(row),
                        **{name + "_ms": value for name, value in values.items()},
                        "best_fixed_ms": fixed, "best_dynamic_ms": dyn,
                        "fixed_over_dynamic": fixed / dyn,
                        "dynamic_lt_0_97_fixed": dyn < .97 * fixed})
write("RQ7_all_1024_triplet_thresholds.csv", triplets,
      BASE + ["native_ms", "view_ms", "repair_ms", "raw_winner", "verdict",
              "native_over_view", "native_over_repair", "view_lt_0_97_native",
              "repair_lt_0_97_native", "runner_up_over_best", "operators"])
for label, name in (("positive_zero_copy", "RQ7_zero_copy_wins_41.csv"),
                    ("counterexample_materialize_faster", "RQ7_materialize_wins_61.csv")):
    write(name, [row for row in triplets if row["verdict"] == label],
          BASE + ["native_ms", "view_ms", "repair_ms", "raw_winner", "verdict",
                  "native_over_view", "native_over_repair", "runner_up_over_best", "operators"])
write("RQ6_dynamic_vs_fixed_1536.csv", dynamic,
      BASE + ["scope", "fixed_NHD_ms", "fixed_HND_ms", "eager_switch_ms",
              "hysteresis_2_ms", "hysteresis_4_ms", "best_fixed_ms",
              "best_dynamic_ms", "fixed_over_dynamic", "dynamic_lt_0_97_fixed", "verdict", "operators"])
write("RQ6_dynamic_positive_5.csv", [row for row in dynamic if row["verdict"] == "positive"],
      BASE + ["scope", "fixed_NHD_ms", "fixed_HND_ms", "eager_switch_ms",
              "hysteresis_2_ms", "hysteresis_4_ms", "fixed_over_dynamic", "dynamic_lt_0_97_fixed"])
write("RQ6_hysteresis_positive_512.csv", [{**row, **shape_fields(row)} for row in verdicts
      if row["rq"] == "L-RQ6" and row["verdict"] == "positive" and row["question"].startswith("滞后")],
      BASE + ["scope", "details", "operators"])

evidence_specs = [
    (ROOT.parent / "real_world_workloads.py", "def _attention(", "1;4;7", "window mask is conditional on query length >1"),
    (ROOT.parent / "real_world_workloads.py", "def _make_attention(", "1;4;7", "historical cache inputs and concatenation; shape window fallback"),
    (ROOT.parent / "discover_real_world_shapes.py", "def shape_payload(", "1;4;7", "window omitted from executable shape; documented cache extent can differ"),
    (ROOT / "build_rq_llm_cases.py", "def write_attention_tsv(", "1;2;3;6;8;9;10", "batch times tokens per request flattened; only GQA/sliding selected"),
    (ROOT / "rq_llm_layout_bench.cu", "__global__ void kv_write_kernel", "1;2;3;6", "producer copies logical KV; no projections"),
    (ROOT / "rq_llm_layout_bench.cu", "__global__ void decode_head_scan_kernel", "1;2;3;6", "head sum reduction consumer"),
    (ROOT / "rq_llm_layout_bench.cu", "__global__ void token_scan_kernel", "1;2;6", "token sum reduction consumer"),
    (ROOT / "rq_llm_layout_bench.cu", "for (int reuse :", "1;3", "conversion reuse pipeline; emitted error hard-coded"),
    (ROOT / "rq_llm_layout_bench.cu", "std::vector<std::pair<int, int>> fanouts", "2", "three shared/split pipelines and seven fanouts"),
    (ROOT / "rq_llm_layout_bench.cu", "const std::vector<std::pair<std::string, std::vector<int>>> traces", "6", "six hard-coded 32-step traces"),
    (ROOT / "rq_llm_layout_bench.cu", "int final_access = accesses.back()", "6", "only final output checked"),
    (ROOT / "rq_llm_layout_bench.cu", "__global__ void scaled_decode_head_scan_kernel", "8", "data and scale address formulas"),
    (ROOT / "rq_llm_layout_bench.cu", "for (int page_size :", "8;10", "page sweeps; writer outside timing; consumer error constant"),
    (ROOT / "triton_kv_layout_bench.py", "def physical(", "1;2;9", "NHD/paged_NHD address alias and padding"),
    (ROOT / "triton_kv_layout_bench.py", "def run_one(", "1;2;9", "QK consumer, empty cache, whole-storage correctness and fanout"),
    (ROOT / "tvm_rq_observation_bench.py", "def projected_boundary_shape(", "1;2;3;4;7;9", "B*Q by hidden projection"),
    (ROOT / "tvm_rq_observation_bench.py", "def build_multi_consumer(", "2", "row/column consumers; identical repeated expressions"),
    (ROOT / "tvm_rq_observation_bench.py", 'square = build_unary(', "3;4", "square correctness constant and component cost model"),
    (ROOT / "llm_boundary_layout_sweep.py", "def alternate_stride(", "3;7", "new backing and copy before timing; not native-storage alias"),
    (ROOT / "llm_boundary_layout_sweep.py", "variants = [", "3;7", "all graph inputs changed; repair timed per call"),
    (ROOT / "prepare_pinned_serving_model.py", "MODEL_ID =", "4;10", "pinned small Qwen checkpoint"),
    (ROOT / "generate_rq_review_artifacts.py", "tvm_groups = defaultdict(dict)", "1;2;4;7;9", "old ledger key omits consumer axis; TVM fanout misclassified"),
]
evidence = []
for path, needle, rqs, claim in evidence_specs:
    data = path.read_bytes()
    lines = data.decode().splitlines()
    matches = [i + 1 for i, line in enumerate(lines) if needle in line]
    if not matches:
        raise ValueError(f"missing source evidence: {path} {needle}")
    evidence.append({"file": str(path), "start_line": matches[0], "needle": needle,
                     "rqs": rqs, "claim": claim, "source_sha256": hashlib.sha256(data).hexdigest()})
write("SOURCE_EVIDENCE_INDEX.csv", evidence,
      ["file", "start_line", "needle", "rqs", "claim", "source_sha256"])

model_groups = defaultdict(list)
case_pairs = defaultdict(Counter)
for case in manifest:
    model_groups[case["parent_case_id"]].append(case)
for row in paired:
    case_pairs[row["case_id"]][row["framework"]] += 1
model_summary = []
for parent_id, cases in sorted(model_groups.items()):
    sample = cases[0]
    ids = [case["case_id"] for case in cases]
    contracts = Counter()
    for case_id in ids:
        contracts.update(case_pairs[case_id])
    status = {name: dict(Counter(catalog[case_id][name] for case_id in ids)) for name in
              ("whole_pytorch_status", "whole_torchinductor_status",
               "vllm_native_slice_status", "sglang_native_slice_status")}
    model_summary.append({"parent_case_id": parent_id, "model_id": sample["model_id"],
                          "model_revision": sample["model_revision"], "subgraph": sample["structure"],
                          "cases": len(cases), "phases": json.dumps(dict(Counter(c["phase"] for c in cases))),
                          "batches": json.dumps(sorted({c["shape"]["batch"] for c in cases})),
                          "query_lengths": json.dumps(sorted({c["shape"]["query_length"] for c in cases})),
                          "kv_lengths": json.dumps(sorted({c["shape"]["kv_length"] for c in cases})),
                          "case_ids": json.dumps(ids), "framework_paired_contracts": json.dumps(dict(contracts)),
                          "broad_statuses": json.dumps(status),
                          "operators": catalog[ids[0]]["operators"],
                          "checkpoint_loaded_for_manifest_cases": False})
write("MODEL_SUBGRAPH_52_CONFIG_COVERAGE.csv", model_summary,
      ["parent_case_id", "model_id", "model_revision", "subgraph", "cases", "phases",
       "batches", "query_lengths", "kv_lengths", "case_ids", "framework_paired_contracts",
       "broad_statuses", "operators", "checkpoint_loaded_for_manifest_cases"])

summary = {
    "run": str(RUN),
    "unique_model_ids": len({case["model_id"] for case in manifest}),
    "unique_model_subgraph_configurations": len(model_summary),
    "tables": {p.name: sum(1 for _ in p.open(encoding="utf-8")) - 1 for p in OUT.glob("*.csv")},
    "triton_regret_ge_3pct": dict(Counter(r["subgraph"] for r in triton_out if r["ge_3pct"])),
    "page_summary": page_summary,
}
(OUT / "table_generation_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

index = ["# 逐项结果表索引", "",
         "对应唯一 full 运行 `rq_adequacy_full_20260927_113444`，没有合并 smoke 或历史运行。",
         "", "先读[逐问审查文档](RQ逐问复核与修正建议_20260929.md)，再查看各表。",
         "所有时间列及 candidate JSON 的数值单位均为 ms。CSV 可按 model_id、subgraph、case_id、contract 筛选。",
         "shape 列保留完整 tensor shape，其他维度列用于快速排序。valid/correctness 只是历史记录的检查状态，",
         "本文发现的硬编码误差、滑窗、padding 等问题仍须按审查文档降级；本次没有 GPU 重跑。", "",
         "| RQ | 所有框架配对候选 | 正反例/持平/无效 |", "|---|---|---|"]
for number in range(1, 11):
    pair_file = f"RQ{number}_all_paired_framework_contracts.csv"
    verdict_file = f"RQ{number}_all_verdicts.csv"
    pair_link = f"[配对表]({pair_file})" if (OUT / pair_file).exists() else "无性能配对，单卡阻塞"
    index.append(f"| RQ{number} | {pair_link} | [判定表]({verdict_file}) |")
index += ["", "还可直接打开：", "",
          "- [39 个 model_id、52 个模型×子图配置](MODEL_SUBGRAPH_52_CONFIG_COVERAGE.csv)",
          "- [RQ1 Triton 77 条旧门槛正例](RQ1_Triton_regret_positive_77.csv)",
          "- [RQ4 六个翻转父配置的全部 144 个子 case](RQ4_six_parent_crossovers_all_children.csv)",
          "- [RQ6 5 条动态表面正例](RQ6_dynamic_positive_5.csv)；[512 条迟滞正例](RQ6_hysteresis_positive_512.csv)",
          "- [RQ7 41 条 view 最快](RQ7_zero_copy_wins_41.csv)；[61 条 materialize 最快](RQ7_materialize_wins_61.csv)",
          "- [RQ10 每 case/route/table 的八页时间](RQ10_all_2304_page_contracts.csv)；[464 条固定128损失](RQ10_page128_loss_464.csv)",
          "- [CUDA shape 与转换成本](CUDA_256_shape_conversion_and_locality.csv)",
          "- [TVM 不同功能原语的独立计时](TVM_all_primitive_measurements.csv)",
          "- [源码位置与哈希](SOURCE_EVIDENCE_INDEX.csv)",
          "- [1024 case 整图、边界和原生切片结果](../ALL_1024_CASE_CATALOG_AND_BROAD_RESULTS.csv)",
          "- [全部单项计时，包括服务和切片](../RQ_ALL_MEASUREMENTS_LEDGER.csv)", "",
          "以下列出本目录全部 CSV。一个合同可对应多候选；一个 case 可有多个合同，不能把行数当独立模型数。", "",
          "| 文件 | 数据行 |", "|---|---:|"]
for name, count in sorted(summary["tables"].items()):
    index.append(f"| [{name}]({name}) | {count} |")
(OUT / "明细表索引.md").write_text("\n".join(index) + "\n", encoding="utf-8")
