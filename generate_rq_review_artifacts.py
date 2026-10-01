#!/usr/bin/env python3
"""Generate exhaustive, auditable ledgers for the 2026-09-27 full RQ run.

The generator does not modify raw results.  It produces:

* one 1024-row case catalogue joining model, operators, shapes and broad results;
* one long-form ledger containing every usable measurement (CUDA repetitions are
  aggregated by process median before being written);
* one comparison ledger containing the paired decisions used by the strict RQ
  interpretation.

The purpose is traceability, not a new statistical adjudication.  Scientific
claims continue to use analyze_rq_full_per_rq.py and its 3% effect gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


THRESHOLD = 1.03


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path):
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dump_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def j(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def as_float(value, default=math.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def aggregate_cuda(rows):
    fields = (
        "benchmark", "case_id", "subgraph", "phase", "strategy", "layout",
        "tokens", "kv_heads", "query_heads", "head_dim", "page_size",
        "reuse_count", "fanout_head", "fanout_token", "allocated_tokens",
        "trace_kind", "attention_kind", "model",
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in fields)].append(row)
    output = []
    for key, values in grouped.items():
        errors = [as_float(row.get("max_abs_error")) for row in values]
        output.append({
            **dict(zip(fields, key)),
            "p50_ms": statistics.median(as_float(row["p50_ms"]) for row in values),
            "process_repetitions": len(values),
            "max_abs_error": max(errors),
            "correct": all(math.isfinite(error) and error <= 1e-3 for error in errors),
        })
    return output


CUDA_RQS = {
    "kv_write": "L-RQ1;L-RQ4",
    "decode_head_scan": "L-RQ1;L-RQ4",
    "token_major_scan": "L-RQ1;L-RQ4",
    "multi_consumer_pipeline": "L-RQ2",
    "weighted_multi_consumer_pipeline": "L-RQ2",
    "layout_conversion": "L-RQ3",
    "conversion_reuse_pipeline": "L-RQ1;L-RQ3",
    "rope_kv_pipeline": "L-RQ3;L-RQ4",
    "decode_under_hbm_contention": "L-RQ3;L-RQ4",
    "state_migration_trace": "L-RQ6",
    "data_metadata_scan": "L-RQ8",
    "paged_decode_head": "L-RQ8;L-RQ10",
    "sparse_paged_decode_head": "L-RQ8;L-RQ10",
}


def strict_choice(values: dict[str, float]):
    ordered = sorted(values.items(), key=lambda item: item[1])
    if not ordered:
        return "missing", math.nan, math.nan
    if len(ordered) == 1:
        return ordered[0][0], ordered[0][1], math.inf
    ratio = ordered[1][1] / ordered[0][1]
    return (ordered[0][0] if ratio >= THRESHOLD else "tie", ordered[0][1], ratio)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw = root / "raw"

    manifest = json.loads((root / "cases/rq_adequacy_1024.json").read_text(encoding="utf-8"))["cases"]
    by_id = {row["case_id"]: row for row in manifest}
    whole = read_jsonl(raw / "whole_graph.jsonl")
    boundary = read_jsonl(raw / "boundary_layout.jsonl")
    cuda = aggregate_cuda(read_csv(raw / "cuda_reference.csv"))
    triton = read_csv(raw / "triton_explicit.csv")
    tvm = read_csv(raw / "tvm_projected.csv")
    cutlass = read_csv(raw / "cutlass_attention.csv")
    serving = {name: read_jsonl(raw / f"{name}_native.jsonl") for name in ("vllm", "sglang")}
    kv_batch = {name: read_jsonl(root / f"kv_request_batch/raw/{name}_kv_request_batch.jsonl")
                for name in ("vllm", "sglang")}
    native = {name: read_jsonl(root / f"native_subgraphs/raw/{name}_subgraph_native.jsonl")
              for name in ("vllm", "sglang")}

    # Long-form measurement ledger.
    measurements = []

    def add_measurement(*, rq, framework, scope, case_id, experiment, candidate,
                        metric, value, unit, correct, evidence, source, axes=None,
                        note="", model="", subgraph="", phase=""):
        case = by_id.get(case_id, {})
        measurements.append({
            "rq": rq,
            "framework": framework,
            "measurement_scope": scope,
            "case_id": case_id,
            "model_id": model or case.get("model_id", ""),
            "subgraph": subgraph or case.get("structure", ""),
            "phase": phase or case.get("phase", ""),
            "experiment": experiment,
            "candidate": candidate,
            "metric": metric,
            "value": value,
            "unit": unit,
            "correct_or_valid": correct,
            "evidence_kind": evidence,
            "causal_axes": j(axes or {}),
            "shape": j(case.get("shape", {})),
            "operators": j(case.get("graph_nodes", [])),
            "source_file": str(source),
            "note": note,
        })

    for row in cuda:
        invalid = row["benchmark"] == "data_metadata_scan"
        add_measurement(
            rq=CUDA_RQS.get(row["benchmark"], ""), framework="CUDA-reference",
            scope="GPU内核或显式管线", case_id=row["case_id"], experiment=row["benchmark"],
            candidate=row["strategy"], metric="p50_ms", value=row["p50_ms"], unit="ms",
            correct=(False if invalid else row["correct"]),
            evidence=("invalid_semantic_counterfactual" if invalid else "direct_runtime_process_median"),
            source=raw / "cuda_reference.csv",
            axes={key: row.get(key, "") for key in
                  ("layout", "tokens", "kv_heads", "query_heads", "head_dim", "page_size",
                   "reuse_count", "fanout_head", "fanout_token", "allocated_tokens", "trace_kind",
                   "process_repetitions")},
            note=("数据与缩放元数据的逻辑值未保持一致；该行不得用于性能结论" if invalid else ""),
            model=row.get("model", ""), subgraph=row.get("subgraph", ""), phase=row.get("phase", ""))

    for row in triton:
        valid = str(row.get("correct", "")).lower() == "true"
        for metric in ("producer_p50_ms", "p50_ms", "pipeline_p50_ms"):
            if row.get(metric, "") == "":
                continue
            add_measurement(
                rq="L-RQ1;L-RQ4;L-RQ9", framework="Triton-explicit",
                scope="GPU显式内核", case_id=row["case_id"], experiment=row.get("experiment", ""),
                candidate=row.get("layout", ""), metric=metric, value=row[metric], unit="ms",
                correct=valid, evidence="direct_runtime", source=raw / "triton_explicit.csv",
                axes={key: row.get(key, "") for key in
                      ("block_size", "block_n", "block_d", "num_warps", "num_stages",
                       "fanout_head", "fanout_token", "physical_shape", "physical_stride")},
                note="paged_NHD与NHD在有效词元上的地址映射相同，RQ9必须合并计数")

    for row in tvm:
        experiment = row.get("experiment", "")
        rq = {
            "consumer_scan": "L-RQ1;L-RQ4;L-RQ7;L-RQ9",
            "multi_consumer": "L-RQ2",
            "conversion_reuse": "L-RQ3",
            "fusion_order": "L-RQ3;L-RQ4",
        }.get(experiment, "L-RQ4")
        add_measurement(
            rq=rq, framework="TVM", scope="投影矩阵原语", case_id=row.get("manifest_case_id", ""),
            experiment=experiment, candidate=row.get("strategy", ""), metric="p50_ms",
            value=row.get("p50_ms", ""), unit="ms", correct=str(row.get("correct", "")).lower() == "true",
            evidence=row.get("evidence_level", ""), source=raw / "tvm_projected.csv",
            axes={key: row.get(key, "") for key in
                  ("rows", "cols", "logical_rows", "logical_cols", "padded_rows", "padded_cols",
                   "reuse", "fanout_row", "fanout_column", "measurement_reused_for_equal_projected_shape",
                   "producer_materialization_timed")},
            note="相同投影形状会复用测量；conversion_reuse是实测原语组成的成本模型")

    for row in cutlass:
        case_id = row.get("manifest_case_id", "")
        valid = row.get("status") == "success" and str(row.get("correct", "")) in {"1", "True", "true"}
        for candidate, metric in (
            ("flat_reference", "flat_ms"), ("tiled_direct", "tiled_direct_ms"),
            ("transform_only", "transform_ms"), ("transform_plus_flat", "transform_plus_flat_ms"),
            ("flat_after_transform", "transformed_flat_ms"),
        ):
            if row.get(metric, "") == "":
                continue
            add_measurement(
                rq="L-RQ3;L-RQ4;L-RQ7;L-RQ9", framework="CUTLASS-CuTe",
                scope="注意力分数矩阵分块边界", case_id=case_id,
                experiment=row.get("projection_kind", ""), candidate=candidate, metric=metric,
                value=row[metric], unit="ms", correct=valid, evidence="direct_runtime_boundary",
                source=raw / "cutlass_attention.csv",
                axes={key: row.get(key, "") for key in
                      ("rows", "cols", "tile_rows", "tile_cols", "measurement_reused_for_equal_shape")},
                note="这是分块/变换边界，不是KV多消费者实验")

    for row in whole:
        add_measurement(
            rq="L-RQ4", framework=("PyTorch-eager" if row.get("backend") == "pytorch" else "TorchInductor"),
            scope="完整合成子图", case_id=row.get("case_id", ""), experiment="whole_graph_backend",
            candidate=row.get("backend", ""), metric="p50_ms", value=row.get("p50_ms", ""), unit="ms",
            correct=row.get("status") == "success" and row.get("correct") is True,
            evidence="supporting_system_runtime", source=raw / "whole_graph.jsonl",
            axes={"compile_seconds": row.get("compile_seconds"), "generated_kernel_count": row.get("generated_kernel_count")},
            note="包含融合、代码生成和内核选择，不是纯布局干预",
            model=row.get("model_id", ""), subgraph=row.get("structure", ""), phase=row.get("phase", ""))

    for row in boundary:
        boundary_rq = "L-RQ3;L-RQ7" if row.get("experiment") == "boundary_layout_sweep" else "L-RQ3"
        add_measurement(
            rq=boundary_rq, framework="PyTorch", scope="子图输入布局边界",
            case_id=row.get("case_id", ""), experiment=row.get("experiment", ""),
            candidate=row.get("strategy", ""), metric="p50_ms", value=row.get("p50_ms", ""), unit="ms",
            correct=row.get("status") == "success" and row.get("correctness_passed") is True,
            evidence="direct_runtime_boundary", source=raw / "boundary_layout.jsonl",
            axes={"input_shapes": row.get("input_shapes"), "input_strides": row.get("input_strides"),
                  "explicit_repair_bytes": row.get("explicit_repair_bytes"),
                  "reuse_count": row.get("reuse_count"),
                  "conversion_placement": row.get("conversion_placement")},
            note="同一逻辑shape和值；比较连续、步长视图和每次连续化修复")

    for framework, rows in serving.items():
        for row in rows:
            add_measurement(
                rq="L-RQ4;L-RQ10", framework=framework, scope="固定Qwen2.5-0.5B完整服务",
                case_id=row.get("case_id", ""), experiment=row.get("comparison_scope", ""),
                candidate=row.get("variant", ""), metric="p50_ms", value=row.get("p50_ms", ""), unit="ms",
                correct=row.get("status") == "success", evidence="direct_serving_wall_time",
                source=raw / f"{framework}_native.jsonl",
                axes={key: row.get(key) for key in
                      ("kv_layout", "page_size", "block_size", "prompt_tokens", "completion_tokens",
                       "request_count", "requested_concurrency", "effective_max_simultaneous_batch")},
                note="只覆盖固定Qwen2.5-0.5B；harness未逐词元比较不同variant输出",
                model="Qwen/Qwen2.5-0.5B-Instruct")
            add_measurement(
                rq="L-RQ4;L-RQ10", framework=framework, scope="固定Qwen2.5-0.5B完整服务",
                case_id=row.get("case_id", ""), experiment=row.get("comparison_scope", ""),
                candidate=row.get("variant", ""), metric="output_tokens_per_second",
                value=row.get("output_tokens_per_second", ""), unit="token/s",
                correct=row.get("status") == "success", evidence="direct_serving_throughput",
                source=raw / f"{framework}_native.jsonl", axes={},
                note="请求级吞吐量，不能与GPU内核毫秒直接相除", model="Qwen/Qwen2.5-0.5B-Instruct")

    for framework, rows in kv_batch.items():
        for row in rows:
            add_measurement(
                rq="L-RQ1;L-RQ2;L-RQ4;L-RQ6;L-RQ9", framework=framework,
                scope="原生KV写入器切片", case_id=row.get("case_id", ""),
                experiment="kv_request_batch_slot_policy", candidate=row.get("slot_policy", ""),
                metric="p50_ms", value=row.get("p50_ms", ""), unit="ms",
                correct=row.get("status") == "success" and row.get("numerically_correct") is True,
                evidence="supporting_native_operator_slice", source=root / f"kv_request_batch/raw/{framework}_kv_request_batch.jsonl",
                axes={key: row.get(key) for key in
                      ("request_batch", "source_manifest_batch", "derived_counterfactual", "per_request_new_tokens",
                       "total_new_tokens", "slot_count", "selected_layout", "kv_block_size")},
                note="只测cache写入，不含attention读取、分配器或完整服务")

    for framework, rows in native.items():
        for row in rows:
            if row.get("p50_ms") is None:
                continue
            add_measurement(
                rq=";".join(row.get("directly_evidenced_rqs", [])), framework=framework,
                scope="框架原生算子切片", case_id=row.get("case_id", ""),
                experiment=row.get("native_operator", ""), candidate=row.get("variant", ""),
                metric="p50_ms", value=row.get("p50_ms", ""), unit="ms",
                correct=row.get("status") == "success" and row.get("numerically_correct") is True,
                evidence="supporting_single_candidate_native_slice",
                source=root / f"native_subgraphs/raw/{framework}_subgraph_native.jsonl",
                axes={"selected_layout": row.get("selected_layout"), "input": row.get("input"), "state": row.get("state")},
                note="没有成对候选且未加载对应checkpoint，不能单独证明RQ胜负")

    measurement_fields = [
        "rq", "framework", "measurement_scope", "case_id", "model_id", "subgraph", "phase",
        "experiment", "candidate", "metric", "value", "unit", "correct_or_valid", "evidence_kind",
        "causal_axes", "shape", "operators", "source_file", "note",
    ]
    dump_csv(out / "RQ_ALL_MEASUREMENTS_LEDGER.csv", measurements, measurement_fields)

    # Paired comparison ledger used to locate every positive, negative and tie.
    comparisons = []

    def add_comparison(rq, framework, case_id, contract, values, evidence, valid=True, note=""):
        case = by_id.get(case_id, {})
        values = {str(k): float(v) for k, v in values.items() if math.isfinite(as_float(v))}
        choice, best, ratio = strict_choice(values) if valid else ("invalid", math.nan, math.nan)
        comparisons.append({
            "rq": rq,
            "framework": framework,
            "case_id": case_id,
            "model_id": case.get("model_id", ""),
            "subgraph": case.get("structure", ""),
            "phase": case.get("phase", ""),
            "contract": contract,
            "candidate_p50_ms": j(values),
            "raw_winner": (min(values, key=values.get) if values and valid else "invalid"),
            "strict_winner_3pct": choice,
            "best_p50_ms": best,
            "winner_margin": ratio,
            "valid": valid,
            "evidence_kind": evidence,
            "shape": j(case.get("shape", {})),
            "operators": j(case.get("graph_nodes", [])),
            "note": note,
        })

    cuda_groups = defaultdict(dict)
    for row in cuda:
        key = (row["benchmark"], row["case_id"], row.get("reuse_count", ""), row.get("fanout_head", ""),
               row.get("fanout_token", ""), row.get("trace_kind", ""))
        cuda_groups[key][row["strategy"]] = row["p50_ms"]
    for (bench, case_id, reuse, fan_h, fan_t, trace), values in cuda_groups.items():
        if bench in {"kv_write", "decode_head_scan", "token_major_scan"}:
            add_comparison("L-RQ1", "CUDA-reference", case_id, bench, values, "direct_runtime")
        elif bench == "conversion_reuse_pipeline":
            add_comparison("L-RQ3", "CUDA-reference", case_id, f"reuse={reuse}", values, "direct_runtime")
            if str(reuse) == "1":
                add_comparison("L-RQ1", "CUDA-reference", case_id,
                               "producer_to_head_edge_reuse=1", values, "direct_runtime")
        elif bench == "weighted_multi_consumer_pipeline":
            add_comparison("L-RQ2", "CUDA-reference", case_id, f"fanout_head={fan_h};fanout_token={fan_t}", values, "direct_runtime")
        elif bench == "multi_consumer_pipeline":
            add_comparison("L-RQ2", "CUDA-reference", case_id, "unweighted_multi_consumer", values, "direct_runtime")
        elif bench in {"rope_kv_pipeline", "decode_under_hbm_contention"}:
            add_comparison("L-RQ3", "CUDA-reference", case_id, bench, values, "direct_runtime")
        elif bench == "state_migration_trace":
            add_comparison("L-RQ6", "CUDA-reference", case_id, f"trace={trace};horizon=32", values, "direct_runtime_synthetic_trace")
        elif bench in {"paged_decode_head", "sparse_paged_decode_head"}:
            add_comparison("L-RQ8;L-RQ10", "CUDA-reference", case_id,
                           f"route={reuse};table={trace}", values, "direct_runtime_kernel_only")
        elif bench == "data_metadata_scan":
            add_comparison("L-RQ8", "CUDA-reference", case_id, "data_x_metadata", values,
                           "invalid_semantic_counterfactual", valid=False,
                           note="逻辑scale值随候选改变且旧二进制错误恒报max_abs_error=0")

    boundary_groups = defaultdict(dict)
    for row in boundary:
        if row.get("experiment") == "boundary_layout_sweep" and row.get("status") == "success" and row.get("correctness_passed") is True:
            boundary_groups[row["case_id"]][row["strategy"]] = row["p50_ms"]
    for case_id, values in boundary_groups.items():
        add_comparison("L-RQ3;L-RQ7", "PyTorch", case_id, "native_vs_view_vs_repair", values, "direct_runtime_boundary")

    boundary_reuse_groups = defaultdict(dict)
    for row in boundary:
        if row.get("experiment") == "boundary_reuse_counterfactual" and row.get("status") == "success" and row.get("correctness_passed") is True:
            boundary_reuse_groups[(row["case_id"], row.get("reuse_count", ""))][row["strategy"]] = row["p50_ms"]
    for (case_id, reuse_count), values in boundary_reuse_groups.items():
        add_comparison("L-RQ3", "PyTorch", case_id, f"boundary_reuse={reuse_count}", values,
                       "direct_runtime_boundary",
                       note="repair_once_then_reuse只在管线开始物化一次；repair_every_use每次消费者前物化")

    whole_groups = defaultdict(dict)
    whole_valid = defaultdict(lambda: True)
    for row in whole:
        if row.get("status") == "success" and row.get("correct") is True:
            whole_groups[row["case_id"]][row["backend"]] = row["p50_ms"]
        else:
            whole_valid[row.get("case_id", "")] = False
    for case_id, values in whole_groups.items():
        if len(values) == 2:
            add_comparison("L-RQ4", "PyTorch-eager_vs_TorchInductor", case_id,
                           "whole_synthetic_subgraph", values, "supporting_system_runtime")

    triton_groups = defaultdict(dict)
    for row in triton:
        if row.get("experiment") == "producer_consumer_edge" and str(row.get("correct", "")).lower() == "true":
            triton_groups[row["case_id"]][row["layout"]] = row["pipeline_p50_ms"]
    for case_id, values in triton_groups.items():
        add_comparison("L-RQ1;L-RQ9", "Triton-explicit", case_id,
                       "producer_plus_consumer_pipeline", values, "direct_runtime",
                       note="RQ9计数时将NHD与paged_NHD合并为同一有效地址映射")

    for row in cutlass:
        if row.get("status") == "success" and str(row.get("correct", "")) in {"1", "True", "true"}:
            add_comparison("L-RQ3;L-RQ7;L-RQ9", "CUTLASS-CuTe", row.get("manifest_case_id", ""),
                           f"tile={row.get('tile_rows')}x{row.get('tile_cols')}",
                           {"tiled_direct": row["tiled_direct_ms"], "transform_plus_flat": row["transform_plus_flat_ms"]},
                           "direct_runtime_boundary")

    tvm_groups = defaultdict(dict)
    for row in tvm:
        if str(row.get("correct", "")).lower() != "true":
            continue
        key = (row.get("manifest_case_id", ""), row.get("experiment", ""), row.get("reuse", ""),
               row.get("fanout_row", ""), row.get("fanout_column", ""))
        tvm_groups[key][row.get("strategy", "")] = row.get("p50_ms", "")
    for (case_id, experiment, reuse_value, fanout_row, fanout_column), values in tvm_groups.items():
        rq = {
            "consumer_scan": "L-RQ1;L-RQ4;L-RQ7;L-RQ9",
            "multi_consumer": "L-RQ2",
            "conversion_reuse": "L-RQ3",
            "fusion_order": "L-RQ3;L-RQ4",
        }.get(experiment, "L-RQ4")
        add_comparison(rq, "TVM", case_id,
                       f"experiment={experiment};reuse={reuse_value};fanout={fanout_row},{fanout_column}",
                       values,
                       ("measured_cost_model" if experiment == "conversion_reuse" else "direct_projected_primitive"),
                       note="相同投影形状可能复用同一测量；TVM多消费者不含生产者和物化")

    for framework, rows in serving.items():
        groups = defaultdict(dict)
        for row in rows:
            groups[(row["case_id"], row["comparison_scope"])][row["variant"]] = row["p50_ms"]
        for (case_id, scope), values in groups.items():
            add_comparison(("L-RQ4" if scope == "layout_only" else "L-RQ10"), framework,
                           case_id, f"serving_{scope}", values, "direct_serving_wall_time",
                           note="固定Qwen2.5-0.5B；未逐词元交叉验证输出")

    for framework, rows in kv_batch.items():
        groups = defaultdict(dict)
        for row in rows:
            if row.get("status") == "success" and row.get("numerically_correct") is True:
                groups[(row["case_id"], row["request_batch"])][row["slot_policy"]] = row["p50_ms"]
        for (case_id, request_batch), values in groups.items():
            add_comparison("L-RQ1;L-RQ2;L-RQ4;L-RQ6;L-RQ9", framework, case_id,
                           f"native_kv_writer;request_batch={request_batch}", values,
                           "supporting_native_operator_slice",
                           note="仅比较KV写入器slot组织，不包含attention读取、分配器和完整服务")

    comparison_fields = [
        "rq", "framework", "case_id", "model_id", "subgraph", "phase", "contract",
        "candidate_p50_ms", "raw_winner", "strict_winner_3pct", "best_p50_ms", "winner_margin",
        "valid", "evidence_kind", "shape", "operators", "note",
    ]
    dump_csv(out / "RQ_ALL_PAIRED_COMPARISONS.csv", comparisons, comparison_fields)

    # Hypothesis-oriented verdicts.  A "positive" row supports the narrowly
    # named observation in `question`; it must not be generalized to the whole
    # RQ without respecting evidence_kind and scope.
    verdicts = []

    def add_verdict(rq, question, case_id, verdict, framework, scope, details,
                    evidence="direct_runtime", note=""):
        case = by_id.get(case_id, {})
        verdicts.append({
            "rq": rq,
            "question": question,
            "case_id": case_id,
            "model_id": case.get("model_id", ""),
            "subgraph": case.get("structure", ""),
            "phase": case.get("phase", ""),
            "verdict": verdict,
            "framework": framework,
            "scope": scope,
            "details": j(details),
            "evidence_kind": evidence,
            "shape": j(case.get("shape", {})),
            "operators": j(case.get("graph_nodes", [])),
            "note": note,
        })

    # RQ1: producer-local choice versus head consumer / complete edge.
    for case_id in sorted({key[1] for key in cuda_groups if key[0] == "kv_write"}):
        producer = cuda_groups[("kv_write", case_id, "0", "0", "0", "")]
        head = cuda_groups[("decode_head_scan", case_id, "0", "0", "0", "")]
        edge = cuda_groups[("conversion_reuse_pipeline", case_id, "1", "0", "0", "")]
        producer_choice = min(producer, key=producer.get)
        selected_edge = "common_NHD" if producer_choice == "NHD" else "producer_native_HND"
        consumer_regret = head[producer_choice] / min(head.values())
        edge_regret = edge[selected_edge] / min(edge.values())
        add_verdict("L-RQ1", "生产者局部最优是否导致完整边至少3%的损失", case_id,
                    "positive" if edge_regret >= THRESHOLD else "negative_or_tie",
                    "CUDA-reference", "producer_to_head_edge",
                    {"producer_p50_ms": producer, "head_consumer_p50_ms": head,
                     "edge_p50_ms": edge, "producer_choice": producer_choice,
                     "consumer_regret": consumer_regret, "edge_regret": edge_regret})

    # RQ2: robust fanout crossover and split-domain necessity are separate.
    multi_by_case = defaultdict(list)
    for key, values in cuda_groups.items():
        bench, case_id, _reuse, fan_h, fan_t, _trace = key
        if bench == "weighted_multi_consumer_pipeline":
            multi_by_case[case_id].append((fan_h, fan_t, values))
    for case_id, contracts in sorted(multi_by_case.items()):
        robust = []
        split_win = False
        packed = {}
        for fan_h, fan_t, values in contracts:
            choice, _best, _ratio = strict_choice(values)
            if choice != "tie":
                robust.append(choice)
            split_win |= choice == "split_NHD_HND_with_conversion"
            packed[f"{fan_h}:{fan_t}"] = values
        add_verdict("L-RQ2", "扇出改变是否造成至少3%的共享布局交叉", case_id,
                    "positive" if len(set(robust)) > 1 else "negative_or_tie",
                    "CUDA-reference", "seven_fanout_pairs", {"fanout_candidates_ms": packed,
                    "robust_winners": robust})
        add_verdict("L-RQ2", "分离布局是否成为至少3%的全局赢家", case_id,
                    "positive" if split_win else "negative",
                    "CUDA-reference", "shared_vs_split_domain", {"fanout_candidates_ms": packed})

    # RQ3 and RQ7: conversion/reuse and zero-copy/materialization boundaries.
    for key, values in sorted(cuda_groups.items()):
        bench, case_id, reuse_value, _fan_h, _fan_t, _trace = key
        if bench != "conversion_reuse_pipeline":
            continue
        choice, _best, _ratio = strict_choice(values)
        add_verdict("L-RQ3", "显式转换一次后复用是否成为至少3%的全局赢家", case_id,
                    "positive" if choice == "NHD_then_convert_once_HND" else "negative_or_tie",
                    "CUDA-reference", f"reuse={reuse_value}",
                    {"candidate_p50_ms": values, "strict_winner": choice})
    for case_id, values in sorted(boundary_groups.items()):
        numeric = {key: float(value) for key, value in values.items()}
        raw_choice = min(numeric, key=numeric.get)
        native_over_best = numeric["native_contiguous"] / numeric[raw_choice]
        if raw_choice == "alternate_strided_view" and native_over_best >= THRESHOLD:
            label = "positive_zero_copy"
        elif raw_choice == "repair_to_contiguous_each_call" and native_over_best >= THRESHOLD:
            label = "counterexample_materialize_faster"
        elif raw_choice == "native_contiguous":
            label = "negative_native_faster"
        else:
            label = "tie"
        add_verdict("L-RQ7", "零复制视图是否至少快3%，以及物化是否构成反例", case_id,
                    label, "PyTorch", "native_vs_view_vs_repair",
                    {"candidate_p50_ms": numeric, "raw_winner": raw_choice,
                     "native_over_best": native_over_best})

    # RQ4 whole-graph sensitivity and parent-level crossover.
    parent_winners = defaultdict(set)
    for case_id, values in sorted(whole_groups.items()):
        if len(values) != 2:
            continue
        numeric = {key: float(value) for key, value in values.items()}
        choice, _best, ratio = strict_choice(numeric)
        add_verdict("L-RQ4", "完整合成子图后端差异是否至少3%（支持性而非纯布局）", case_id,
                    "positive_sensitive" if choice != "tie" else "negative_or_tie",
                    "PyTorch-eager_vs_TorchInductor", "whole_synthetic_subgraph",
                    {"candidate_p50_ms": numeric, "strict_winner": choice, "margin": ratio},
                    evidence="supporting_system_runtime")
        parent_winners[by_id[case_id]["parent_case_id"]].add(choice)
    for parent_id, choices in sorted(parent_winners.items()):
        clean = {choice for choice in choices if choice != "tie"}
        add_verdict("L-RQ4", "同一父模型跨shape/阶段是否改变至少3%的后端赢家", parent_id,
                    "positive" if len(clean) > 1 else "negative_or_tie",
                    "PyTorch-eager_vs_TorchInductor", "parent_shape_phase_crossover",
                    {"strict_winners": sorted(choices)}, evidence="supporting_system_runtime",
                    note="parent行不是manifest case，shape/operators字段为空")

    add_verdict("L-RQ5", "通信与设备放置是否改变布局最优解", "GLOBAL-RQ5", "blocked",
                "all", "single_A10", {}, evidence="blocked_single_gpu",
                note="无多rank collective、rank偏斜、复制或放置反事实；不是负例")

    # RQ6: hysteresis and adaptive-vs-fixed are distinct observations.
    for key, values in sorted(cuda_groups.items()):
        bench, case_id, _reuse, _fan_h, _fan_t, trace = key
        if bench != "state_migration_trace":
            continue
        eager = values["eager_switch"]
        hysteresis = min(values["hysteresis_2"], values["hysteresis_4"])
        fixed = min(values["fixed_NHD"], values["fixed_HND"])
        dynamic = min(eager, hysteresis)
        add_verdict("L-RQ6", "滞后策略是否比立即切换至少快3%", case_id,
                    "positive" if eager / hysteresis >= THRESHOLD else "negative_or_tie",
                    "CUDA-reference", f"trace={trace};horizon=32",
                    {"candidate_p50_ms": values, "eager_over_hysteresis": eager / hysteresis})
        add_verdict("L-RQ6", "最佳动态策略是否比最佳固定布局至少快3%", case_id,
                    "positive" if fixed / dynamic >= THRESHOLD else "negative_or_tie",
                    "CUDA-reference", f"trace={trace};horizon=32",
                    {"candidate_p50_ms": values, "fixed_over_dynamic": fixed / dynamic})

    # RQ8 valid route/page interaction plus invalid data-metadata rows.
    page_by_case = defaultdict(list)
    for key, values in cuda_groups.items():
        bench, case_id, route, _fan_h, _fan_t, table = key
        if bench in {"paged_decode_head", "sparse_paged_decode_head"}:
            page_by_case[case_id].append((route, table, values))
    for case_id, contracts in sorted(page_by_case.items()):
        winners = set()
        for _route, _table, values in contracts:
            strategy = min(values, key=values.get)
            winners.add(int(strategy.split("_", 2)[1]))
        add_verdict("L-RQ8", "路由/表条件改变时页大小赢家是否变化", case_id,
                    "positive" if len(winners) > 1 else "negative",
                    "CUDA-reference", "route_x_page_table",
                    {"page_winners": sorted(winners), "contract_count": len(contracts)},
                    evidence="direct_runtime_kernel_only")
    for key, values in sorted(cuda_groups.items()):
        bench, case_id, metadata_axis, _fan_h, _fan_t, _trace = key
        if bench == "data_metadata_scan":
            add_verdict("L-RQ8", "数据与元数据联合布局", case_id, "invalid", "CUDA-reference",
                        f"metadata_axis={metadata_axis}", {"recorded_p50_ms": values},
                        evidence="invalid_semantic_counterfactual",
                        note="只保留审计痕迹，禁止作为正例或反例")

    # RQ9 canonicalized extended-map win.
    for case_id, values in sorted(triton_groups.items()):
        numeric = {key: float(value) for key, value in values.items()}
        native_best = min(numeric[key] for key in ("NHD", "HND") if key in numeric)
        paged_hnd = numeric.get("paged_HND", math.inf)
        canonical_best = min({"NHD_or_paged_NHD": min(numeric.get("NHD", math.inf), numeric.get("paged_NHD", math.inf)),
                              "HND": numeric.get("HND", math.inf), "paged_HND": paged_hnd}.items(),
                             key=lambda item: item[1])[0]
        gain = native_best / paged_hnd
        add_verdict("L-RQ9", "去除NHD/paged_NHD别名后，paged_HND是否成为至少3%的新赢家", case_id,
                    "positive" if canonical_best == "paged_HND" and gain >= THRESHOLD else "negative",
                    "Triton-explicit", "canonical_three_address_maps",
                    {"candidate_pipeline_p50_ms": numeric, "canonical_winner": canonical_best,
                     "native_over_paged_hnd": gain})

    # RQ10 fixed page=128 regret and fragmentation.
    for key, values in sorted(cuda_groups.items()):
        bench, case_id, route, _fan_h, _fan_t, table = key
        if bench not in {"paged_decode_head", "sparse_paged_decode_head"}:
            continue
        fixed = values.get("page_128_route_1_table_identity")
        if fixed is None:
            fixed_candidates = [value for name, value in values.items() if name.startswith("page_128_")]
            fixed = fixed_candidates[0] if fixed_candidates else math.nan
        oracle = min(values.values())
        regret = fixed / oracle
        add_verdict("L-RQ10", "固定页128相对逐合同最佳页的损失是否至少3%", case_id,
                    "positive" if regret >= THRESHOLD else "negative_or_tie",
                    "CUDA-reference", f"route={route};table={table}",
                    {"page_candidate_p50_ms": values, "fixed128_over_oracle": regret},
                    evidence="direct_runtime_kernel_only")
    fragmentation = defaultdict(list)
    for row in cuda:
        if row["benchmark"] in {"paged_decode_head", "sparse_paged_decode_head"}:
            live = int(row["tokens"])
            allocated = int(row["allocated_tokens"])
            fragmentation[row["case_id"]].append((int(row["page_size"]), live, allocated))
    for case_id, triples in sorted(fragmentation.items()):
        unique = {(page, live, allocated) for page, live, allocated in triples}
        max_ratio = max(allocated / live for _page, live, allocated in unique)
        add_verdict("L-RQ10", "至少一个页大小是否产生尾页碎片", case_id,
                    "positive" if any(allocated > live for _page, live, allocated in unique) else "negative",
                    "CUDA-reference", "case_x_page_fragmentation",
                    {"unique_case_page_contracts": len(unique), "max_allocated_over_live": max_ratio},
                    evidence="calculated_from_controlled_shape",
                    note="这是容量比计算，不是分配器耗时实测")

    verdict_fields = [
        "rq", "question", "case_id", "model_id", "subgraph", "phase", "verdict",
        "framework", "scope", "details", "evidence_kind", "shape", "operators", "note",
    ]
    dump_csv(out / "RQ_ALL_CASE_VERDICTS_POSITIVE_NEGATIVE.csv", verdicts, verdict_fields)

    # Broad one-row-per-case catalogue.
    whole_by_case = defaultdict(dict)
    for row in whole:
        whole_by_case[row.get("case_id", "")][row.get("backend", "")] = row
    boundary_by_case = defaultdict(dict)
    for row in boundary:
        if row.get("experiment") == "boundary_layout_sweep":
            boundary_by_case[row.get("case_id", "")][row.get("strategy", "")] = row
    native_by_case = defaultdict(dict)
    for framework, rows in native.items():
        for row in rows:
            case_id = row.get("case_id", "")
            previous = native_by_case[case_id].get(framework)
            if previous is None or (previous.get("p50_ms") is None and row.get("p50_ms") is not None):
                native_by_case[case_id][framework] = row
    catalog = []
    for case in manifest:
        cid = case["case_id"]
        wg = whole_by_case[cid]
        bd = boundary_by_case[cid]
        catalog.append({
            "case_id": cid,
            "model_id": case["model_id"],
            "model_revision": case.get("model_revision", ""),
            "subgraph": case["structure"],
            "phase": case["phase"],
            "batch": case["shape"].get("batch", ""),
            "query_length": case["shape"].get("query_length", ""),
            "kv_length": case["shape"].get("kv_length", ""),
            "hidden_size": case["shape"].get("hidden_size", ""),
            "shape": j(case["shape"]),
            "operators": j(case.get("graph_nodes", [])),
            "target_rqs": ";".join(case.get("target_rqs", [])),
            "whole_pytorch_ms": wg.get("pytorch", {}).get("p50_ms", ""),
            "whole_torchinductor_ms": wg.get("triton", {}).get("p50_ms", ""),
            "whole_pytorch_status": wg.get("pytorch", {}).get("status", "missing"),
            "whole_torchinductor_status": wg.get("triton", {}).get("status", "missing"),
            "boundary_native_ms": bd.get("native_contiguous", {}).get("p50_ms", ""),
            "boundary_view_ms": bd.get("alternate_strided_view", {}).get("p50_ms", ""),
            "boundary_repair_ms": bd.get("repair_to_contiguous_each_call", {}).get("p50_ms", ""),
            "vllm_native_slice_status": native_by_case[cid].get("vllm", {}).get("status", "missing"),
            "vllm_native_slice_ms": native_by_case[cid].get("vllm", {}).get("p50_ms", ""),
            "sglang_native_slice_status": native_by_case[cid].get("sglang", {}).get("status", "missing"),
            "sglang_native_slice_ms": native_by_case[cid].get("sglang", {}).get("p50_ms", ""),
            "source_url": case.get("source_url", ""),
        })
    catalog_fields = list(catalog[0])
    dump_csv(out / "ALL_1024_CASE_CATALOG_AND_BROAD_RESULTS.csv", catalog, catalog_fields)

    summary = {
        "result_dir": str(root),
        "manifest_cases": len(manifest),
        "measurement_rows": len(measurements),
        "paired_comparison_rows": len(comparisons),
        "hypothesis_verdict_rows": len(verdicts),
        "catalog_rows": len(catalog),
        "measurement_by_framework": dict(sorted(__import__("collections").Counter(row["framework"] for row in measurements).items())),
        "comparison_by_rq": dict(sorted(__import__("collections").Counter(row["rq"] for row in comparisons).items())),
        "warnings": [
            "RQ5没有性能行：单卡无法验证通信与设备放置。",
            "RQ8 data_x_metadata行被保留用于审计，但valid=false，禁止用于性能结论。",
            "TVM相同投影形状会复用测量，不能把映射行当统计独立样本。",
            "CUTLASS相同几何会映射到多个模型case，统计独立性需要去重。",
            "原生算子切片不是完整子图或完整模型服务。",
        ],
    }
    (out / "LEDGER_GENERATION_SUMMARY.json").write_text(j(summary) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
