#!/usr/bin/env python3
"""Build a de-duplicated positive/negative-example ledger for v10/v13 RQs.

The result tree contains smoke runs, repaired reruns and final runs.  This tool
inventories all of them, but uses the latest complete 640-case and 256-case
diversity runs as canonical evidence so repeated measurements are not counted
as independent samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


THRESHOLD = 1.03
FIELDS = [
    "taxonomy", "rq", "hypothesis", "classification", "adjudication_status", "framework",
    "evidence_level", "case_id", "model_id", "model_revision", "phase",
    "subgraph", "graph_nodes", "shape", "compared_layouts", "metric",
    "interpretation", "artifact",
]


def read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def read_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fnum(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def shape_text(shape: dict) -> str:
    if not shape:
        return "-"
    aliases = [
        ("batch", "B"), ("query_length", "q"), ("kv_length", "kv"),
        ("num_query_heads", "Hq"), ("num_kv_heads", "Hkv"),
        ("head_dim", "D"), ("hidden_size", "hidden"),
        ("intermediate_size", "ffn"), ("num_experts", "E"),
        ("experts_per_token", "topk"), ("page_size", "page"),
    ]
    out = [f"{label}={shape[key]}" for key, label in aliases if shape.get(key) is not None]
    return ", ".join(out) if out else json.dumps(shape, ensure_ascii=False, sort_keys=True)


def manifest_map(path: Path):
    data = read_json(path)
    cases = data.get("cases", []) if isinstance(data, dict) else data
    return {row["case_id"]: row for row in cases}


def split_model(source: str):
    if not source or source.startswith("v10:"):
        return ("synthetic/pre-registered-control", "-")
    if "@" in source:
        return tuple(source.split("@", 1))
    return (source, "-")


def add(rows, lookup, *, taxonomy, rq, hypothesis, classification, framework,
        case_id="-", evidence_level="runtime_empirical", compared_layouts="-",
        metric="-", interpretation="-", artifact="-", model_id=None,
        model_revision=None, phase=None, subgraph=None, shape=None,
        graph_nodes=None, adjudication_status="-"):
    meta = lookup.get(case_id, {})
    rows.append({
        "taxonomy": taxonomy,
        "rq": rq,
        "hypothesis": hypothesis,
        "classification": classification,
        "adjudication_status": adjudication_status,
        "framework": framework,
        "evidence_level": evidence_level,
        "case_id": case_id,
        "model_id": model_id or meta.get("model_id", "-"),
        "model_revision": model_revision or meta.get("model_revision", "-"),
        "phase": phase or meta.get("phase", "-"),
        "subgraph": subgraph or meta.get("structure", meta.get("structure_label", "-")),
        "graph_nodes": " -> ".join(graph_nodes or meta.get("graph_nodes", [])) or "-",
        "shape": shape_text(shape or meta.get("shape", {})),
        "compared_layouts": compared_layouts,
        "metric": metric,
        "interpretation": interpretation,
        "artifact": artifact,
    })


def add_metric_cases(rows, lookup, h, key, classification, framework, artifact,
                     layouts="hypothesis-specific candidates", note=None):
    values = (h.get("metrics") or {}).get(key, [])
    for value in values:
        if isinstance(value, str):
            case_id, metric = value, "listed by canonical adjudicator"
        elif isinstance(value, dict):
            case_id = value.get("case_id", "-")
            metric = ", ".join(f"{k}={v}" for k, v in value.items() if k != "case_id")
        else:
            continue
        add(rows, lookup, taxonomy="v13", rq=h["rq"], hypothesis=h["hypothesis"],
            classification=classification, framework=framework, case_id=case_id,
            evidence_level=h.get("evidence_level", "runtime_empirical"),
            compared_layouts=layouts, metric=metric,
            interpretation=note or h.get("summary", "-"), artifact=artifact)


def boundary_evidence(rows, lookup, path: Path):
    groups = defaultdict(dict)
    for row in read_jsonl(path):
        if row.get("status") == "success" and row.get("correctness_passed"):
            groups[row["case_id"]][row["strategy"]] = row
    for case_id, group in groups.items():
        native = group.get("native_contiguous")
        view = group.get("alternate_strided_view")
        repair = group.get("repair_to_contiguous_each_call")
        if not (native and view and repair):
            continue
        vn = fnum(view.get("p50_ms")); rn = fnum(repair.get("p50_ms")); nn = fnum(native.get("p50_ms"))
        if not all(x and x > 0 for x in (vn, rn, nn)):
            continue
        ratio = rn / vn
        if ratio >= THRESHOLD:
            cls = "positive"
            note = "合法 strided view 相对逐次 materialize 至少快 3%，支持零拷贝边界优化。"
        elif ratio <= 1 / THRESHOLD:
            cls = "counterexample"
            note = "materialize 比合法 view 至少快 3%，反驳“零拷贝总是最优”。"
        else:
            cls = "negative_control"
            note = "view 与 materialize 差异小于 3%，属于布局不敏感区。"
        for rq, hyp in (("L-RQ1", "H1.3"), ("L-RQ3", "H3.4"), ("L-RQ7", "H7.1/H7.3/H7.4")):
            add(rows, lookup, taxonomy="v13", rq=rq, hypothesis=hyp,
                classification=cls, framework="PyTorch", case_id=case_id,
                compared_layouts="alternate_strided_view vs repair_to_contiguous_each_call vs native_contiguous",
                metric=f"repair/view={ratio:.4f}x; native={nn:.6g} ms; view={vn:.6g} ms; repair={rn:.6g} ms",
                interpretation=note, artifact=str(path))


def page_evidence(rows, lookup, path: Path):
    groups = defaultdict(list)
    for row in read_csv(path):
        if row.get("benchmark") == "paged_decode_head" and fnum(row.get("p50_ms"), 0) > 0:
            groups[row["case_id"]].append(row)
    for case_id, vals in groups.items():
        p1 = [x for x in vals if int(float(x.get("page_size") or 0)) == 1]
        if not p1:
            continue
        best = min(vals, key=lambda x: fnum(x["p50_ms"], float("inf")))
        one = min(p1, key=lambda x: fnum(x["p50_ms"], float("inf")))
        ratio = fnum(one["p50_ms"]) / fnum(best["p50_ms"])
        cls = "positive" if ratio >= THRESHOLD else "negative_control"
        add(rows, lookup, taxonomy="v13", rq="L-RQ10", hypothesis="H10.2/H10.3",
            classification=cls, framework="CUDA-reference", case_id=case_id,
            compared_layouts=f"page=1 vs best page={best.get('page_size')}",
            metric=f"page1/best={ratio:.4f}x",
            interpretation=("小 page 的 metadata/allocator 代价达到 3% 阈值。" if cls == "positive"
                            else "page=1 与最佳 page 的差异未达到 3%。"), artifact=str(path))


def cutlass_evidence(rows, lookup, path: Path):
    valid = [x for x in read_csv(path) if x.get("status") == "success" and x.get("correct") in {"1", "True", "true"}]
    dominant = Counter(x.get("winner") for x in valid).most_common(1)
    dominant = dominant[0][0] if dominant else None
    for x in valid:
        case_id = x.get("manifest_case_id", "-")
        winner = x.get("winner", "-")
        cls = "positive" if winner != dominant else "negative_control"
        add(rows, lookup, taxonomy="v13", rq="L-RQ4", hypothesis="H4.1/H4.2",
            classification=cls, framework="CUTLASS", case_id=case_id,
            evidence_level="native_runtime_counterfactual",
            compared_layouts="tiled_direct vs transform_plus_flat (flat_ms is recorded reference only)",
            metric=f"winner={winner}; winner_speedup={x.get('winner_speedup')}x",
            interpretation=("winner 偏离 CUTLASS 全局主导策略，证明 shape-specific 决策有价值。" if cls == "positive"
                            else "全局主导策略在该 shape 保持最优，作为稳定区对照。"), artifact=str(path))


def tvm_evidence(rows, lookup, path: Path):
    groups = defaultdict(list)
    for x in read_csv(path):
        if x.get("correct") in {"True", "true", "1"} and fnum(x.get("p50_ms"), 0) > 0:
            groups[(x.get("manifest_case_id"), x.get("experiment"), x.get("reuse"))].append(x)
    winners = []
    for key, vals in groups.items():
        if len({x.get("strategy") for x in vals}) < 2:
            continue
        best = min(vals, key=lambda x: fnum(x["p50_ms"], float("inf")))
        winners.append((key, best, vals))
    dominant = Counter(best.get("strategy") for _, best, _ in winners).most_common(1)
    dominant = dominant[0][0] if dominant else None
    for (case_id, experiment, reuse), best, vals in winners:
        cls = "positive" if best.get("strategy") != dominant else "negative_control"
        worst = max(fnum(x["p50_ms"]) for x in vals)
        gain = worst / fnum(best["p50_ms"])
        levels = {x.get("evidence_level") for x in vals if x.get("evidence_level")}
        evidence_level = ("native_measured_cost_model"
                          if "native_measured_cost_model" in levels
                          else "native_runtime_counterfactual")
        timing_note = ("这些 p50 是由已实测 TVM primitive 的中位数按公式相加，整条策略未直接计时。"
                       if evidence_level == "native_measured_cost_model"
                       else "这些候选 compiled function 由 TVM time_evaluator 直接计时。")
        add(rows, lookup, taxonomy="v13", rq="L-RQ4", hypothesis="H4.1/H4.2",
            classification=cls, framework="TVM", case_id=case_id,
            evidence_level=evidence_level,
            compared_layouts=" vs ".join(sorted({x.get("strategy", "-") for x in vals})),
            metric=f"experiment={experiment}; reuse={reuse or '-'}; winner={best.get('strategy')}; worst/best={gain:.4f}x",
            interpretation=(("TVM winner 偏离全局主导 schedule/layout。" if cls == "positive"
                             else "TVM 全局主导 schedule/layout 在该点仍最优。")
                            + timing_note), artifact=str(path))


def native_serving_evidence(rows, path: Path, framework: str, model: str):
    data = [x for x in read_jsonl(path) if x.get("status") == "success"]
    groups = defaultdict(list)
    for x in data:
        groups[(x.get("comparison_scope"), x.get("case_id"))].append(x)
    for (scope, case_id), vals in groups.items():
        if len(vals) < 2:
            continue
        best = min(vals, key=lambda x: fnum(x.get("p50_ms"), float("inf")))
        worst = max(vals, key=lambda x: fnum(x.get("p50_ms"), 0))
        ratio = fnum(worst["p50_ms"]) / fnum(best["p50_ms"])
        cls = "positive" if ratio >= THRESHOLD else "negative_control"
        for rq, hyp in (("L-RQ4", "H4.1/H4.2/H4.5"), ("L-RQ10", "H10.1/H10.3")):
            add(rows, {}, taxonomy="v13", rq=rq, hypothesis=hyp,
                classification=cls, framework=framework, case_id=case_id,
                model_id=model, model_revision="7ae5576", phase="serving",
                subgraph="full decoder serving + paged KV cache",
                shape={"batch": best.get("effective_max_simultaneous_batch"),
                       "query_length": best.get("requested_prompt_tokens_per_request")},
                compared_layouts=" vs ".join(sorted({str(x.get("variant")) for x in vals})),
                metric=f"scope={scope}; winner={best.get('variant')}; worst/best={ratio:.4f}x",
                interpretation=("原生系统策略差异达到 3%。" if cls == "positive"
                                else "原生系统该 workload 下策略差异小于 3%，或 winner 未变化。"), artifact=str(path))


def kv_batch_evidence(rows, lookup, path: Path, framework: str):
    data = [x for x in read_jsonl(path) if x.get("status") == "success" and x.get("numerically_correct")]
    groups = defaultdict(list)
    for x in data:
        groups[(x.get("case_id"), x.get("request_batch"))].append(x)
    for (case_id, batch), vals in groups.items():
        if len({x.get("slot_policy") for x in vals}) < 2:
            continue
        best = min(vals, key=lambda x: fnum(x.get("p50_ms"), float("inf")))
        worst = max(vals, key=lambda x: fnum(x.get("p50_ms"), 0))
        ratio = fnum(worst["p50_ms"]) / fnum(best["p50_ms"])
        cls = "positive" if ratio >= THRESHOLD else "negative_control"
        for rq, hyp in (("L-RQ2", "H2.1/H2.5"), ("L-RQ4", "H4.1"), ("L-RQ6", "H6.1")):
            add(rows, lookup, taxonomy="v13", rq=rq, hypothesis=hyp,
                classification=cls, framework=framework, case_id=case_id,
                evidence_level="framework_native_kv_writer_slice",
                compared_layouts=" vs ".join(sorted({x.get("slot_policy", "-") for x in vals})),
                metric=f"request_batch={batch}; winner={best.get('slot_policy')}; worst/best={ratio:.4f}x",
                interpretation=("request batch/slot schedule 使局部 KV writer 差异达到 3%。" if cls == "positive"
                                else "KV writer 局部策略差异小于 3%；不能单凭 writer 决定全图 layout。"), artifact=str(path))


def build(args):
    root = args.results.resolve()
    canonical = root / args.canonical_run
    diversity = root / args.diversity_run
    raw = canonical / "v13_640/full_combined/raw"
    hyp_path = canonical / "v13_640/full_combined/v13_hypothesis_results.json"
    v10_path = canonical / "v10_strict/v10_rq1_validation.json"
    lookup = manifest_map(raw / "rq_llm_all_640_cases.json")
    div_lookup = manifest_map(diversity / "cases/llm_shape_diversity_256.json")
    all_lookup = {**lookup, **div_lookup}
    rows = []

    # Strict v10: measured failures are counterevidence; unsupported precision is not a negative.
    v10 = read_json(v10_path)
    pos_ids = set(v10["hypotheses"]["H1.1"].get("positive_cases", []))
    neg_id = "v10-neg-bf16-small"
    for case in v10["cases"]:
        measured = case.get("successful_process_repetitions", 0) > 0 and case.get("correctness_pass")
        if case["case_id"] in pos_ids:
            cls, hyp = "positive", "H1.1"
        elif case["case_id"] == neg_id:
            cls, hyp = "negative_control", "H1.NEG"
        elif measured:
            cls, hyp = "counterevidence", "H1.1"
        else:
            cls, hyp = "uninterpretable", "H1.1"
        mid, rev = split_model(case.get("source", ""))
        shape = {"batch": case.get("batch"), "query_length": case.get("q_len"),
                 "kv_length": case.get("kv_len"), "num_query_heads": case.get("Hq"),
                 "num_kv_heads": case.get("Hkv"), "page_size": case.get("page_size")}
        metric = (f"producer={case.get('producer_winner')}; edge={case.get('edge_winner')}; "
                  f"regret={case.get('edge_regret_pct')}%; epsilon={case.get('epsilon_pct')}%")
        add(rows, {}, taxonomy="v10-strict", rq="v10-RQ1", hypothesis=hyp,
            classification=cls, framework="CUDA-reference", case_id=case["case_id"],
            model_id=mid, model_revision=rev, phase="decode", subgraph=case.get("subgraph"),
            shape=shape, compared_layouts="NN/NH-copy/HH-native/HN-copy/NH-view",
            metric=metric,
            interpretation=("通过预注册 rank-inversion 判据。" if cls == "positive" else
                            "预注册 negative control。" if cls == "negative_control" else
                            "可测但未通过严格 rank-inversion 门槛。" if cls == "counterevidence" else
                            "未成功测量（如 A10 不支持 NVFP4）；不能当作反例。"), artifact=str(v10_path))

    # v10 H1.3/H1.4 are separate supported claims even though strict H1.1 did not pass.
    v10_cases = {x["case_id"]: x for x in v10["cases"]}
    for hyp, key, layouts, note in (
        ("H1.3", "view_beats_copy_cases", "NH-view vs NH-copy",
         "合法 stride view 比显式 copy 快；该证据不等于 H1.1 Stage-4 gate 通过。"),
        ("H1.4", "native_beats_copy_cases", "HH-native vs NHD + NH-copy",
         "直接产生 consumer-native HND 比先产 NHD 再 copy 快。"),
    ):
        for case_id in v10["hypotheses"][hyp].get(key, []):
            case = v10_cases.get(case_id, {})
            mid, rev = split_model(case.get("source", ""))
            shape = {"batch": case.get("batch"), "query_length": case.get("q_len"),
                     "kv_length": case.get("kv_len"), "num_query_heads": case.get("Hq"),
                     "num_kv_heads": case.get("Hkv"), "page_size": case.get("page_size")}
            med = case.get("strategy_median_edge_us", {})
            metric = "; ".join(f"{k}={v} us" for k, v in med.items() if k in {"NH-view", "NH-copy", "HH-native"})
            add(rows, {}, taxonomy="v10-strict", rq="v10-RQ1", hypothesis=hyp,
                classification="positive", framework="CUDA-reference", case_id=case_id,
                model_id=mid, model_revision=rev, phase="decode",
                subgraph=case.get("subgraph", "-"), shape=shape,
                compared_layouts=layouts, metric=metric, interpretation=note,
                artifact=str(v10_path))

    for hyp in ("H1.1", "H1.2", "H1.3", "H1.4", "H1.NEG"):
        h = v10["hypotheses"][hyp]
        status = h.get("status", "not_run")
        add(rows, {}, taxonomy="v10-strict", rq="v10-RQ1", hypothesis=hyp,
            classification="aggregate_verdict", adjudication_status=status,
            framework="CUDA-reference", case_id="aggregate",
            evidence_level="canonical_adjudication", metric=f"status={status}",
            interpretation=h.get("experiment", "-"), artifact=str(v10_path))

    hypotheses = read_json(hyp_path)["hypotheses"]
    hs = {h["hypothesis"]: h for h in hypotheses}
    art = str(hyp_path)

    # Canonical explicit case lists.
    add_metric_cases(rows, lookup, hs["H1.1"], "inversions", "positive", "CUDA-reference", art,
                     "producer NHD/HND vs consumer NHD/HND")
    add_metric_cases(rows, lookup, hs["H1.1"], "triton_inversions", "positive", "Triton", art,
                     "NHD/HND/paged_NHD/paged_HND")
    add_metric_cases(rows, lookup, hs["H1.2"], "cases", "positive", "CUDA-reference", art,
                     "common_NHD vs producer_native_HND across reuse")
    add_metric_cases(rows, lookup, hs["H1.4"], "winning_points", "positive", "CUDA-reference", art,
                     "direct consumer-native emission vs materialization")
    add_metric_cases(rows, lookup, hs["H1.NEG"], "cases", "negative_control", "CUDA-reference", art,
                     "candidate layouts", "布局差异低于 3%，是预声明的不敏感区。")

    add_metric_cases(rows, lookup, hs["H2.1"], "threshold_cases", "positive", "CUDA-reference", art,
                     "common domain vs heterogeneous domains")
    add_metric_cases(rows, lookup, hs["H2.3"], "cases", "negative_control", "CUDA-reference", art,
                     "common layout vs split domains", "共享 domain 直接获胜，是 split-domain 假设的负对照。")
    add_metric_cases(rows, lookup, hs["H2.5"], "cases", "positive", "CUDA-reference", art,
                     "common vs split plus boundary overhead")

    add_metric_cases(rows, lookup, hs["H3.1"], "cases", "positive", "CUDA-reference/TVM", art,
                     "keep vs materialize across reuse")
    add_metric_cases(rows, lookup, hs["H3.2"], "threshold_shift_cases", "positive", "CUDA-reference", art,
                     "conversion threshold with/without temporary pressure")
    add_metric_cases(rows, lookup, hs["H3.3"], "overlap_sensitive_cases", "positive", "CUDA-reference", art,
                     "conversion threshold with/without overlap")
    add_metric_cases(rows, lookup, hs["H3.5"], "cases", "negative_control", "CUDA-reference", art,
                     "keep vs convert at reuse=1", "R=1 时 conversion 不获胜，是 amortization 的负对照。")

    for hyp in ("H6.1", "H6.4"):
        add_metric_cases(rows, lookup, hs[hyp], "horizons", "positive", "CUDA-reference cost model", art,
                         "keep current state vs migrate", "由实测局部成本推得有限切换 horizon；不是 live migration。")
    add_metric_cases(rows, lookup, hs["H6.5"], "thresholds", "positive", "CUDA-reference cost model", art,
                     "migration with/without recapture")
    add(rows, lookup, taxonomy="v13", rq="L-RQ6", hypothesis="H6.3",
        classification="negative_control", framework="CUDA-reference cost model",
        evidence_level="measured_cost_model", case_id="184 stationary traces",
        compared_layouts="no switch vs repeated switch", metric="stationary_shapes=184",
        interpretation="stationary trace 中反复切换只增加转换成本。", artifact=art)

    for point in hs["H8.1"]["metrics"].get("interactions", []):
        interaction = abs(fnum(point.get("interaction"), 0))
        cls = "positive" if interaction >= 0.03 else "negative_control"
        add(rows, lookup, taxonomy="v13", rq="L-RQ8", hypothesis="H8.1",
            classification=cls, framework="CUDA-reference", case_id=point.get("case_id", "-"),
            compared_layouts="data layout × metadata granularity",
            metric=f"granularity={point.get('granularity')}; |interaction|={interaction:.4f}",
            interpretation=("data×metadata interaction 达到 3%。" if cls == "positive"
                            else "交互项低于 3%，作为弱耦合对照。"), artifact=art)
    add_metric_cases(rows, lookup, hs["H8.2"], "cases", "positive", "CUDA-reference", art,
                     "page/data layout under metadata placement")
    add_metric_cases(rows, lookup, hs["H8.5"], "cases", "positive", "CUDA-reference", art,
                     "data layout under metadata placement")
    add(rows, lookup, taxonomy="v13", rq="L-RQ8", hypothesis="H8.4",
        classification="negative_control", framework="CUDA-reference", case_id="aggregate/per-head-metadata",
        compared_layouts="per-head vs per-token metadata",
        metric=f"median per-head interaction={hs['H8.4']['metrics']['median_tiny_interaction']:.4f}",
        interpretation="cache-resident per-head metadata 是预声明负对照。", artifact=art)

    add_metric_cases(rows, lookup, hs["H9.1"], "consumer_only_cases", "positive", "Triton", art,
                     "native NHD/HND vs paged extended layouts")
    add_metric_cases(rows, lookup, hs["H9.1"], "producer_consumer_pipeline_cases", "positive", "Triton", art,
                     "native NHD/HND vs paged extended layouts")
    add_metric_cases(rows, lookup, hs["H9.4"], "cases", "negative_control", "Triton", art,
                     "native-only oracle vs extended set", "native set 已包含 oracle；关闭扩展空间零 regret。")

    # Runtime rows needed to recover exact examples that aggregate JSON only counted.
    boundary_evidence(rows, lookup, raw / "llm_640_boundary_layout_sweep.jsonl")
    # Canonical v13 adjudication prefers the exhaustive 240-attention artifact;
    # rq_cuda_reference.csv is only the smaller representative subset.
    page_evidence(rows, lookup, raw / "rq_cuda_all_attention.csv")
    tvm_evidence(rows, lookup, raw / "tvm_640_rq_observations.csv")
    cutlass_evidence(rows, lookup, raw / "cutlass_640_softmax_boundary.csv")

    # More discriminative 256-shape suite and true native serving/KV-writer slices.
    boundary_evidence(rows, div_lookup, diversity / "raw/boundary_layout.jsonl")
    cutlass_evidence(rows, div_lookup, diversity / "raw/cutlass_attention.csv")
    tvm_evidence(rows, div_lookup, diversity / "raw/tvm_projected.csv")
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    native_serving_evidence(rows, diversity / "raw/vllm_native.jsonl", "vLLM", model)
    native_serving_evidence(rows, diversity / "raw/sglang_native.jsonl", "SGLang", model)
    kv_batch_evidence(rows, div_lookup, diversity / "kv_request_batch/raw/vllm_kv_request_batch.jsonl", "vLLM")
    kv_batch_evidence(rows, div_lookup, diversity / "kv_request_batch/raw/sglang_kv_request_batch.jsonl", "SGLang")

    # Explicit blocked/no-valid-example rows.
    for framework in ("CUDA-reference", "Triton", "TVM", "CUTLASS", "vLLM", "SGLang", "Hexcute"):
        add(rows, all_lookup, taxonomy="v13", rq="L-RQ5", hypothesis="H5.1-H5.5",
            classification="blocked", framework=framework, evidence_level="not_run_single_gpu",
            case_id="-", compared_layouts="distributed rank/device layouts",
            metric="requires multiple GPUs/ranks",
            interpretation="单卡 1 无法产生正例或反例；不得把未运行写成反例。", artifact=art)
    add(rows, all_lookup, taxonomy="v13", rq="L-RQ4", hypothesis="H4.4",
        classification="blocked", framework="all", evidence_level="not_run_multi_hardware",
        metric="requires multiple hardware platforms",
        interpretation="跨硬件稳定性未由 A10 单卡验证。", artifact=art)
    add(rows, all_lookup, taxonomy="v13", rq="L-RQ10", hypothesis="H10.1/H10.5",
        classification="counterevidence", framework="CUDA-reference/vLLM/SGLang",
        case_id="aggregate", metric="winner changes=0",
        compared_layouts="prefix/fanout/page/kernel-vs-system policies",
        interpretation="本轮没有观测到预期 winner change，主假设仍 inconclusive。", artifact=art)
    for rq in ("L-RQ1", "L-RQ3", "L-RQ4", "L-RQ7", "L-RQ9"):
        add(rows, all_lookup, taxonomy="v13", rq=rq, hypothesis="architecture coverage",
            classification="blocked", framework="Hexcute", evidence_level="architecture_blocked",
            metric="A10 sm_86; Hexcute target unavailable/inapplicable",
            interpretation="没有 Hexcute runtime 正/反例；CUDA-reference 不能冒充 Hexcute。", artifact=art)

    # Preserve every canonical hypothesis verdict, including hypotheses whose
    # adjudicator exposes only an aggregate metric rather than case IDs.
    for h in hypotheses:
        status = h.get("status", "inconclusive")
        add(rows, all_lookup, taxonomy="v13", rq=h["rq"], hypothesis=h["hypothesis"],
            classification="aggregate_verdict", adjudication_status=status,
            framework="/".join(h.get("frameworks") or ["none"]),
            case_id="aggregate", evidence_level=h.get("evidence_level", "canonical_adjudication"),
            compared_layouts="hypothesis-level aggregate", metric=f"status={status}",
            interpretation=h.get("summary", "-"), artifact=art)

    # Stable de-duplication across repeated hypothesis lists.
    unique = {}
    for row in rows:
        key = tuple(row[k] for k in FIELDS)
        unique[key] = row
    rows = list(unique.values())
    rows.sort(key=lambda x: (x["taxonomy"], x["rq"], x["hypothesis"], x["classification"], x["framework"], x["case_id"]))

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)

    inventory = sorted(p.name for p in root.iterdir())
    report = render_report(rows, root, canonical, diversity, inventory, hyp_path, v10_path, args.csv_out)
    args.md_out.write_text(report)
    return rows


def md_escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def representative(rows, classification, n=4):
    pool = [r for r in rows if r["classification"] == classification and r["case_id"] not in {"-", "aggregate"}]
    def score(r):
        text = r["metric"]
        # Rank by effect size, not by an unrelated absolute latency or N*.
        ratio_patterns = [r"speedup=([0-9.]+)", r"repair/view=([0-9.]+)",
                          r"worst/best=([0-9.]+)", r"page1/best=([0-9.]+)"]
        ratios = [float(m.group(1)) for pattern in ratio_patterns if (m := re.search(pattern, text))]
        if classification == "counterexample" and ratios:
            return max(max(x, 1 / x) for x in ratios if x > 0)
        if classification == "negative_control" and ratios:
            return -min(abs(x - 1) for x in ratios)
        extras = []
        if m := re.search(r"regret=([0-9.]+)%", text): extras.append(1 + float(m.group(1)) / 100)
        if m := re.search(r"\|interaction\|=([0-9.]+)", text): extras.append(1 + float(m.group(1)))
        return max(ratios + extras, default=0)
    return sorted(pool, key=score, reverse=True)[:n]


def render_table(items):
    if not items:
        return "无可解释的运行时样本。"
    lines = ["| 类别 | 框架 | hypothesis | 模型@revision | case / phase | 子图 | shape | layout 对照与实测 | 原始证据 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in items:
        model = r["model_id"] + ("@" + r["model_revision"][:12] if r["model_revision"] not in {"", "-"} else "")
        evidence = f"`{r['artifact']}`"
        lines.append("| " + " | ".join(md_escape(x) for x in [
            r["classification"], r["framework"], r["hypothesis"], model,
            f"{r['case_id']} / {r['phase']}",
            r["subgraph"] + ("<br>" + r["graph_nodes"] if r["graph_nodes"] != "-" else ""), r["shape"],
            f"{r['compared_layouts']}; {r['metric']}", evidence]) + " |")
    return "\n".join(lines)


def render_report(rows, root, canonical, diversity, inventory, hyp_path, v10_path, csv_path):
    lines = [
        "# 所有 results：v10/v13 RQ 正例、反例与来源总账",
        "",
        "## 结论先行",
        "",
        "此前结果**没有**在一份文件中完整整理“RQ → 正例/反例 → 框架 → 模型 → 子图 → shape → 原始证据”。本文件补齐该缺口。",
        "",
        "本报告使用以下术语：",
        "",
        "- `positive`：达到预声明 3% practical threshold，支持该 hypothesis 的成对运行时证据。",
        "- `negative_control`：预期布局不敏感或扩展空间无收益的对照区；它不是执行失败。",
        "- `counterexample`：反驳强命题（例如“zero-copy 总是更快”）的有效反例。",
        "- `counterevidence`：正确执行但没有通过 hypothesis 门槛，或主假设的 winner change 为 0。",
        "- `uninterpretable/blocked`：不支持的 dtype/架构、OOM、缺硬件或未运行；**不能**当作反例。",
        "- `aggregate_verdict`：canonical adjudicator 的 hypothesis 总裁决，仅用于显示 supported/inconclusive/blocked，不计作新的逐-case 正例。",
        "",
        f"完整逐 case 总账见 `{csv_path}`（Markdown 只展示最有判别力的代表样本）。",
        "",
        "## 去重与结果选择",
        "",
        f"结果根目录共盘点 **{len(inventory)}** 个顶层条目。canonical 640 运行是 `{canonical}`；补充 shape/batch/page-boundary 判别力的 256-case 运行是 `{diversity}`。smoke、修复前失败和重复 rerun 被纳入盘点但不重复计为独立科学样本。",
        "",
        "v10 与 v13 是不同判据：v10 严格预注册 Stage-3 的 H1.1 为 `not_supported`（仅 1 个 real-shape positive，0 个 core positive，Stage4 gate=false）；v13 L-RQ1 在更广的 640-case/CUDA/Triton 判据下有 27+19 个 inversion。两者不能互相替代。",
        "",
        "## 汇总矩阵",
        "",
        "| RQ | positive | negative control | counterexample/counterevidence | blocked/uninterpretable | 结论 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    rq_order = ["v10-RQ1"] + [f"L-RQ{i}" for i in range(1, 11)]
    conclusion = {
        "v10-RQ1": "严格 H1.1 未通过；H1.3/H1.4 与 negative control 通过。",
        "L-RQ1": "存在 edge rank inversion、reuse 与直接 native emission 收益，也存在不敏感区。",
        "L-RQ2": "异构 domain/overhead 会改变阈值，但 shared domain 在大量 case 仍是负对照。",
        "L-RQ3": "转换是否摊销取决于 reuse、临时空间、overlap；zero-copy 并非总胜。",
        "L-RQ4": "winner 随 shape/framework 变化；跨硬件 H4.4 尚未验证。",
        "L-RQ5": "分布式 RQ 单卡 blocked，无科学正/反例。",
        "L-RQ6": "实测成本模型给出 switching horizon；尚非 live migration 端到端验证。",
        "L-RQ7": "stride/view 可避免 repair，但 materialize 在一部分 case 更快。",
        "L-RQ8": "data×metadata 有交互；per-head 小 metadata 是重要负对照。",
        "L-RQ9": "扩展 layout 空间在少数 Triton case 获胜，在多数 case 可关闭而零 regret。",
        "L-RQ10": "小 page penalty 有正例；winner-change/system-coupling 主假设本轮仍 inconclusive。",
    }
    for rq in rq_order:
        sub = [x for x in rows if x["rq"] == rq]
        c = Counter(x["classification"] for x in sub)
        lines.append(f"| {rq} | {c['positive']} | {c['negative_control']} | {c['counterexample'] + c['counterevidence']} | {c['blocked'] + c['uninterpretable']} | {conclusion[rq]} |")

    for rq in rq_order:
        sub = [x for x in rows if x["rq"] == rq]
        lines += ["", f"## {rq}", "", conclusion[rq], ""]
        by_hyp = defaultdict(Counter)
        for r in sub:
            by_hyp[r["hypothesis"]][r["classification"]] += 1
        lines += ["### Hypothesis 覆盖", "",
                  "| hypothesis | canonical verdict | positive | negative control | counterexample/evidence | blocked/uninterpretable |",
                  "|---|---|---:|---:|---:|---:|"]
        for hyp in sorted(by_hyp):
            c = by_hyp[hyp]
            verdicts = sorted({r["adjudication_status"] for r in sub
                               if r["hypothesis"] == hyp and r["adjudication_status"] != "-"})
            lines.append(f"| {hyp} | {', '.join(verdicts) or '-'} | {c['positive']} | {c['negative_control']} | {c['counterexample'] + c['counterevidence']} | {c['blocked'] + c['uninterpretable']} |")
        lines += [""]
        pos = representative(sub, "positive")
        neg = representative(sub, "counterexample") + representative(sub, "negative_control") + representative(sub, "counterevidence")
        lines += ["### 最强正例", "", render_table(pos), "", "### 反例 / 负对照 / 反证", "", render_table(neg[:6])]
        blocked = [x for x in sub if x["classification"] in {"blocked", "uninterpretable"}]
        if blocked:
            lines += ["", "### 不可判定项", "", render_table(blocked[:8])]

    by_framework = defaultdict(Counter)
    for r in rows:
        by_framework[r["framework"]][r["classification"]] += 1
    lines += ["", "## 框架来源审计", "",
              "| 框架 | positive | negative control | counterexample/evidence | blocked/uninterpretable | 解释 |",
              "|---|---:|---:|---:|---:|---|"]
    for fw in sorted(by_framework):
        c = by_framework[fw]
        note = "真实 native runtime/slice" if fw in {"vLLM", "SGLang", "TVM", "Triton", "CUTLASS"} else "机制基线/投影/组合标签"
        if fw == "Hexcute": note = "A10 architecture-blocked；没有伪装成 CUDA-reference"
        lines.append(f"| {fw} | {c['positive']} | {c['negative_control']} | {c['counterexample'] + c['counterevidence']} | {c['blocked'] + c['uninterpretable']} | {note} |")

    lines += ["", "## 所有 results 顶层条目盘点", "",
              "以下条目均被盘点；canonical 选择规则避免把 smoke/rerun 当作独立重复样本：", "",
              "| 条目 | 审计角色 | 是否独立计入结论 |", "|---|---|---|"]
    for name in inventory:
        if name == canonical.name:
            role, used = "canonical 640 + v10 strict", "是"
        elif name == diversity.name:
            role, used = "canonical 256 diversity/native supplement", "是"
        elif "smoke" in name:
            role, used = "smoke/diagnostic", "否"
        elif "repair" in name or "repaired" in name:
            role, used = "repair audit/rerun", "仅用于确认修复，不重复计数"
        elif name.startswith("ALL_RESULTS_"):
            role, used = "本分析生成物", "否"
        elif name.endswith(".log"):
            role, used = "launcher log", "否"
        else:
            role, used = "历史 full/partial run", "否；可能被 canonical 覆盖"
        lines.append(f"| `{name}` | {role} | {used} |")
    lines += ["", "## 可复现", "", "```bash",
              "cd /home/liangyilei/ladder_home",
              "python staged/baseline_framework/layout_research/analyze_all_results_rq_examples.py",
              "```", "",
              "canonical adjudication 输入：", "",
              f"- `{hyp_path}`",
              f"- `{v10_path}`",
              f"- `{canonical / 'v13_640/full_combined/raw'}`",
              f"- `{diversity}`", "",
              "## 解释边界", "",
              "1. 640 manifest 的模型来源是真实 pinned model config；CUDA/PyTorch/Triton/TVM/CUTLASS 的部分记录是按该模型维度抽取的 operator/subgraph slice，不等于启动完整模型。",
              "2. vLLM/SGLang serving 行使用固定 Qwen2.5-0.5B-Instruct；native KV-writer slice 则按 256-case manifest 的真实模型维度构造。",
              "3. L-RQ5 和 H4.4 不能由卡 1 推断；Hexcute 也没有 A10 runtime 证据。",
              "4. `counterevidence` 只否定本轮配置下的预期效应，不自动证明相反的普遍命题。",
              ""]
    return "\n".join(lines)


def main():
    here = Path(__file__).resolve().parent
    results = here / "results"
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=results)
    ap.add_argument("--canonical-run", default="v10_v13_640_all_frameworks_20260920_143843")
    ap.add_argument("--diversity-run", default="llm_shape_diversity_v2_full_20260921_200314")
    ap.add_argument("--md-out", type=Path, default=results / "ALL_RESULTS_V10_V13_RQ_POSITIVE_NEGATIVE_EXAMPLES_CN.md")
    ap.add_argument("--csv-out", type=Path, default=results / "ALL_RESULTS_V10_V13_RQ_EXAMPLE_LEDGER.csv")
    args = ap.parse_args()
    rows = build(args)
    print(f"wrote {args.md_out} and {args.csv_out}; rows={len(rows)}")


if __name__ == "__main__":
    main()
