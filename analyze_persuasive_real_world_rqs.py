#!/usr/bin/env python3
"""Analyze a discriminative real-world supplement without inflating evidence.

Runtime, projected, whole-model external-validity, source-only, not-applicable,
and blocked cells remain distinct.  In particular, vLLM/SGLang measurements on
the pinned Qwen checkpoint are workload counterfactuals rather than fabricated
per-architecture executions of every manifest row.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


RQS = {
    "v10-RQ1": "端到端最优 layout 与局部最优是否冲突",
    "L-RQ1": "跨边 layout 排名反转",
    "L-RQ2": "layout domain 的最优粒度",
    "L-RQ3": "layout 转换/修复的摊销阈值",
    "L-RQ4": "最优 layout 随 shape/phase/workload 的稳定性",
    "L-RQ5": "通信与 placement 引起的 layout 反转",
    "L-RQ6": "持久状态 layout 重配置阈值",
    "L-RQ7": "layout 等价与零拷贝可变换性",
    "L-RQ8": "数据与 metadata 的联合 layout",
    "L-RQ9": "layout 搜索空间表达能力与截断",
    "L-RQ10": "paged/block 粒度与 allocator 耦合",
}
FRAMEWORKS = ("Triton", "TVM", "CUTLASS/CuTe", "vLLM", "SGLang", "Hexcute")


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def csv_rows(path: Path) -> list[dict]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def number(row: dict, key: str) -> float | None:
    try:
        value = row.get(key)
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def fmt(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def success(row: dict) -> bool:
    return row.get("status") == "success" and row.get("correct", True) not in (False, "False", "0", 0)


def whole_metrics(rows: list[dict]) -> dict:
    statuses = Counter(str(row.get("status", "unknown")) for row in rows)
    by_case = defaultdict(dict)
    for row in rows:
        backend = row.get("backend") or row.get("framework")
        if backend in {"pytorch", "triton"} and success(row):
            by_case[row.get("case_id")][backend] = row
    ratios, winners, structure_winners, structure_ratios = [], Counter(), defaultdict(Counter), defaultdict(list)
    for pair in by_case.values():
        if set(pair) >= {"pytorch", "triton"}:
            eager, compiled = number(pair["pytorch"], "p50_ms"), number(pair["triton"], "p50_ms")
            if eager and compiled:
                ratios.append(eager / compiled)
                structure_ratios[pair["pytorch"].get("structure") or pair["pytorch"].get("subgraph")].append(eager / compiled)
                winner = "TorchInductor" if compiled < eager else "PyTorch eager"
                winners[winner] += 1
                structure_winners[pair["pytorch"].get("structure") or pair["pytorch"].get("subgraph")][winner] += 1
    mixed = sum(1 for counts in structure_winners.values() if len(counts) > 1)
    meaningful = {"TorchInductor": sum(value >= 1.03 for value in ratios),
                  "PyTorch eager": sum(value <= 1 / 1.03 for value in ratios),
                  "within_3pct": sum(1 / 1.03 < value < 1.03 for value in ratios)}
    mismatches = [{key: row.get(key) for key in ("case_id", "backend", "structure", "phase",
                                                  "status", "max_abs_error", "rmse", "correctness_rule")}
                  for row in rows if row.get("status") == "numerical_mismatch"]
    return {"rows": len(rows), "statuses": dict(statuses), "paired": len(ratios),
            "median_eager_over_compiled": median(ratios), "winners": dict(winners),
            "meaningful_winners_3pct": meaningful,
            "median_speedup_by_structure": {key: median(value) for key, value in sorted(structure_ratios.items())},
            "structures_with_backend_crossover": mixed, "numerical_mismatches": mismatches}


def boundary_metrics(rows: list[dict]) -> dict:
    by_case = defaultdict(dict)
    for row in rows:
        by_case[row.get("case_id")][row.get("strategy")] = row
    ratios_direct, ratios_repair, winners, meaningful_winners, meaningful_gains = [], [], Counter(), Counter(), []
    complete = 0
    for group in by_case.values():
        if not {"native_contiguous", "alternate_strided_view", "repair_to_contiguous_each_call"} <= set(group):
            continue
        if not all(success(group[key]) for key in group):
            continue
        times = {key: number(row, "p50_ms") for key, row in group.items()}
        if not all(times.values()):
            continue
        complete += 1
        native = times["native_contiguous"]
        ratios_direct.append(native / times["alternate_strided_view"])
        ratios_repair.append(native / times["repair_to_contiguous_each_call"])
        winner = min(times, key=times.get)
        winners[winner] += 1
        gain = native / times[winner]
        if winner != "native_contiguous" and gain >= 1.03:
            meaningful_winners[winner] += 1
            meaningful_gains.append(gain)
    return {"rows": len(rows), "complete_triplets": complete,
            "median_native_over_direct": median(ratios_direct),
            "median_native_over_repair": median(ratios_repair), "winners": dict(winners),
            "direct_or_repair_wins": winners["alternate_strided_view"] + winners["repair_to_contiguous_each_call"],
            "meaningful_non_native_wins_3pct": sum(meaningful_winners.values()),
            "meaningful_winner_counts_3pct": dict(meaningful_winners),
            "median_meaningful_gain": median(meaningful_gains)}


def cuda_metrics(rows: list[dict]) -> dict:
    # Aggregate independent process repetitions per candidate before ranking.
    # Selecting the minimum raw repetition biases the winner toward noise.
    candidate_samples = defaultdict(list)
    for row in rows:
        value = number(row, "p50_ms")
        # Cross-edge rank inversion requires the same candidate set.  Other
        # experiments have intentionally different strategies and must not be
        # mixed merely because their labels differ.
        if value is not None and (row.get("strategy") or row.get("layout")) in {"NHD", "HND"}:
            candidate_samples[(row.get("case_id"), row.get("benchmark"),
                               row.get("strategy") or row.get("layout"))].append(value)
    groups = defaultdict(list)
    for (case_id, benchmark, strategy), values in candidate_samples.items():
        groups[(case_id, benchmark)].append((statistics.median(values), strategy))
    winners_by_case = defaultdict(set)
    for (case_id, _), values in groups.items():
        winners_by_case[case_id].add(min(values)[1])
    return {"rows": len(rows), "case_benchmarks": len(groups),
            "cases_with_cross_edge_rank_inversion": sum(len(values) > 1 for values in winners_by_case.values()),
            "cases": len(winners_by_case)}


def triton_metrics(rows: list[dict]) -> dict:
    by_case = defaultdict(list)
    for row in rows:
        value = number(row, "pipeline_p50_ms")
        if value is not None and row.get("correct", "True") not in ("False", "0"):
            by_case[row.get("case_id")].append((value, row.get("layout")))
    extended_wins, gains, meaningful_gains, winners = 0, [], [], Counter()
    for values in by_case.values():
        best = min(values)
        winners[best[1]] += 1
        native = [item for item in values if item[1] in {"NHD", "HND"}]
        if native and best[1] not in {"NHD", "HND"}:
            extended_wins += 1
            gains.append(min(native)[0] / best[0])
            if min(native)[0] / best[0] >= 1.03:
                meaningful_gains.append(min(native)[0] / best[0])
    result = {"rows": len(rows), "cases": len(by_case), "winners": dict(winners),
              "extended_layout_wins": extended_wins, "median_extended_gain": median(gains)}
    result["meaningful_extended_wins_3pct"] = len(meaningful_gains)
    result["median_meaningful_extended_gain"] = median(meaningful_gains)
    return result


def tvm_metrics(rows: list[dict]) -> dict:
    by_consumer = defaultdict(list)
    by_reuse = defaultdict(list)
    for row in rows:
        value = number(row, "p50_ms")
        manifest_id = row.get("manifest_case_id")
        if value is None or not manifest_id:
            continue
        if row.get("experiment") == "consumer_scan":
            by_consumer[(manifest_id, row.get("case_id"))].append((value, row.get("strategy")))
        elif row.get("experiment") == "conversion_reuse":
            by_reuse[(manifest_id, row.get("reuse"))].append((value, row.get("strategy")))
    consumer_winners, reuse_winners = defaultdict(set), defaultdict(set)
    for (manifest_id, _), values in by_consumer.items():
        consumer_winners[manifest_id].add(min(values)[1])
    for (manifest_id, _), values in by_reuse.items():
        reuse_winners[manifest_id].add(min(values)[1])
    mapped = {row.get("manifest_case_id") for row in rows if row.get("manifest_case_id")}
    return {"rows": len(rows), "mapped_case_experiments": len(mapped) * 4,
            "mapped_cases": len(mapped),
            "cases_with_consumer_rank_inversion": sum(len(values) > 1 for values in consumer_winners.values()),
            "cases_with_reuse_crossover": sum(len(values) > 1 for values in reuse_winners.values())}


def cutlass_metrics(rows: list[dict]) -> dict:
    good = [row for row in rows if row.get("status") == "success"]
    winners = Counter(row.get("winner") for row in good)
    gains = [number(row, "winner_speedup") for row in good]
    gains = [value for value in gains if value is not None]
    skipped = [row for row in rows if row.get("status") == "oom_preflight"]
    return {"rows": len(rows), "successful_rows": len(good), "cases": len({row.get('manifest_case_id') for row in good}),
            "winners": dict(winners), "median_winner_speedup": median(gains),
            "oom_preflight_cases": len(skipped),
            "oom_preflight_case_ids": [row.get("manifest_case_id") for row in skipped]}


def native_metrics(rows: list[dict]) -> dict:
    good = [row for row in rows if row.get("status") == "success" and number(row, "p50_ms")]
    groups = defaultdict(list)
    for row in good:
        groups[(row.get("case_id"), row.get("comparison_scope"))].append(row)
    winners = Counter()
    for values in groups.values():
        if len(values) > 1:
            winners[min(values, key=lambda row: float(row["p50_ms"])).get("variant")] += 1
    batch_groups = defaultdict(list)
    for row in good:
        batch_groups[(row.get("variant"), row.get("comparison_scope"),
                      row.get("requested_prompt_tokens_per_request"),
                      row.get("requested_output_tokens_per_request"),
                      row.get("request_count"))].append(row)
    batch_speedups = []
    batch_pairs = 0
    for values in batch_groups.values():
        by_batch = {int(row.get("effective_max_simultaneous_batch", 0)): row
                    for row in values}
        multi = [batch for batch in by_batch if batch > 1]
        if 1 in by_batch and multi:
            batch_pairs += 1
            target = max(multi)
            batch_speedups.append(float(by_batch[1]["p50_ms"]) /
                                  float(by_batch[target]["p50_ms"]))
    return {"rows": len(rows), "successful_rows": len(good), "paired_regimes": sum(len(v) > 1 for v in groups.values()),
            "winners": dict(winners), "winner_variants_across_regimes": len(winners),
            "single_multi_request_pairs": batch_pairs,
            "median_multi_request_wall_speedup": median(batch_speedups),
            "multi_request_faster_pairs": sum(value > 1 for value in batch_speedups)}


def evidence_state(framework: str, rq: str, available: dict[str, bool]) -> str:
    if rq == "v10-RQ1":
        if framework == "Triton": return "runtime" if available[framework] else "missing"
        if framework in {"TVM", "CUTLASS/CuTe", "vLLM", "SGLang"}: return "supporting-only"
        return "arch-blocked"
    if rq == "L-RQ5": return "single-GPU-blocked"
    if framework == "Hexcute":
        return "arch-blocked" if rq in {"L-RQ1", "L-RQ3", "L-RQ4", "L-RQ7", "L-RQ9"} else "N/A"
    runtime = {
        "Triton": {"L-RQ1", "L-RQ2", "L-RQ4", "L-RQ7", "L-RQ9"},
        "TVM": {"L-RQ1", "L-RQ3", "L-RQ4", "L-RQ9"},
        "CUTLASS/CuTe": {"L-RQ3", "L-RQ4", "L-RQ7", "L-RQ9"},
        "vLLM": {"L-RQ4", "L-RQ9", "L-RQ10"},
        "SGLang": {"L-RQ4", "L-RQ9", "L-RQ10"},
    }
    source = {
        "Triton": {"L-RQ8"}, "TVM": {"L-RQ7"},
        "CUTLASS/CuTe": {"L-RQ1", "L-RQ3", "L-RQ7", "L-RQ8"},
        "vLLM": {"L-RQ1", "L-RQ2", "L-RQ3", "L-RQ6", "L-RQ7", "L-RQ8"},
        "SGLang": {"L-RQ1", "L-RQ2", "L-RQ3", "L-RQ6", "L-RQ7", "L-RQ8"},
    }
    if framework == "TVM" and rq == "L-RQ2":
        return "runtime-consumer-only" if available.get(framework) else "missing"
    if rq in runtime.get(framework, set()):
        return "runtime" if available.get(framework) else "missing"
    if rq in source.get(framework, set()): return "source-only"
    return "N/A"


def make_svg(path: Path, matrix: dict[tuple[str, str], str]) -> None:
    labels = list(RQS); width, row_h, col_w, left = 1120, 38, 128, 250
    colors = {"runtime": "#1b9e77", "runtime-consumer-only": "#2ca25f",
              "source-only": "#7570b3", "supporting-only": "#66a61e",
              "single-GPU-blocked": "#d95f02", "arch-blocked": "#e6ab02", "missing": "#e7298a", "N/A": "#d9d9d9"}
    height = 95 + row_h * len(labels) + 70
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<style>text{font-family:DejaVu Sans,Arial,sans-serif;font-size:12px}.title{font-size:18px;font-weight:bold}.cell{stroke:#fff;stroke-width:2}</style>',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="20" y="28" class="title">v10/v13 RQ × framework evidence on NVIDIA A10</text>']
    for col, framework in enumerate(FRAMEWORKS):
        parts.append(f'<text x="{left + col*col_w + col_w/2}" y="70" text-anchor="middle">{html.escape(framework)}</text>')
    for row_index, rq in enumerate(labels):
        y = 82 + row_index * row_h
        parts.append(f'<text x="15" y="{y+24}">{html.escape(rq + ": " + RQS[rq][:18])}</text>')
        for col, framework in enumerate(FRAMEWORKS):
            state = matrix[(rq, framework)]
            x = left + col * col_w
            parts.append(f'<rect class="cell" x="{x}" y="{y}" width="{col_w}" height="{row_h}" fill="{colors[state]}"/>')
            parts.append(f'<text x="{x+col_w/2}" y="{y+24}" text-anchor="middle" fill="white">{html.escape(state)}</text>')
    legend_y = 82 + len(labels)*row_h + 28
    x = 20
    for state, color in colors.items():
        parts += [f'<rect x="{x}" y="{legend_y}" width="14" height="14" fill="{color}"/>',
                  f'<text x="{x+19}" y="{legend_y+12}">{html.escape(state)}</text>']
        x += 135
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def make_performance_svg(path: Path, metrics: dict) -> None:
    """Render only audited, comparable runtime observations."""
    structure = metrics["whole"]["median_speedup_by_structure"]
    events = [
        ("CUDA NHD/HND edge inversion", metrics["cuda"]["cases_with_cross_edge_rank_inversion"], metrics["cuda"]["cases"]),
        ("TVM consumer inversion", metrics["TVM"]["cases_with_consumer_rank_inversion"], metrics["TVM"]["mapped_cases"]),
        ("TVM reuse crossover", metrics["TVM"]["cases_with_reuse_crossover"], metrics["TVM"]["mapped_cases"]),
        ("PyTorch non-native >=3%", metrics["boundary"]["meaningful_non_native_wins_3pct"], metrics["boundary"]["complete_triplets"]),
        ("Triton extended >=3%", metrics["Triton"]["meaningful_extended_wins_3pct"], metrics["Triton"]["cases"]),
    ]
    width, height, left, chart_w = 1100, 620, 255, 760
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<style>text{font-family:DejaVu Sans,Arial,sans-serif;font-size:13px}.title{font-size:19px;font-weight:bold}.axis{stroke:#555;stroke-width:1}</style>',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="20" y="30" class="title">Audited runtime observations (NVIDIA A10)</text>',
             '<text x="20" y="58">Whole-graph median speedup: PyTorch eager / TorchInductor (correct pairs only)</text>']
    max_speedup = max(3.2, max(structure.values(), default=1.0) * 1.08)
    for index, (name, value) in enumerate(structure.items()):
        y = 78 + index * 35
        bar = chart_w * value / max_speedup
        parts += [f'<text x="20" y="{y+18}">{html.escape(name)}</text>',
                  f'<rect x="{left}" y="{y}" width="{bar:.1f}" height="22" fill="#4c78a8"/>',
                  f'<text x="{left+bar+8:.1f}" y="{y+17}">{value:.3f}x</text>']
    baseline_x = left + chart_w / max_speedup
    parts.append(f'<line class="axis" x1="{baseline_x:.1f}" y1="72" x2="{baseline_x:.1f}" y2="{78+len(structure)*35}" stroke-dasharray="4,3"/>')
    start = 390
    parts.append(f'<text x="20" y="{start-18}">Comparable cases exhibiting the predicted observation</text>')
    for index, (name, count, total) in enumerate(events):
        y = start + index * 42
        fraction = count / total if total else 0
        parts += [f'<text x="20" y="{y+18}">{html.escape(name)}</text>',
                  f'<rect x="{left}" y="{y}" width="{chart_w}" height="22" fill="#eeeeee"/>',
                  f'<rect x="{left}" y="{y}" width="{chart_w*fraction:.1f}" height="22" fill="#59a14f"/>',
                  f'<text x="{left+chart_w+8}" y="{y+17}">{count}/{total} ({fraction*100:.1f}%)</text>']
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root, raw = args.result_dir.resolve(), args.result_dir.resolve() / "raw"
    manifest_path = root / "cases/executed_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"cases": []}
    cases = manifest.get("cases", [])
    artifacts = {
        "whole": jsonl(raw / "whole_graph.jsonl"),
        "boundary": jsonl(raw / "boundary_layout.jsonl"),
        "cuda": csv_rows(raw / "cuda_reference.csv"),
        "Triton": csv_rows(raw / "triton_explicit.csv"),
        "TVM": csv_rows(raw / "tvm_projected.csv"),
        "CUTLASS/CuTe": csv_rows(raw / "cutlass_attention.csv"),
        "vLLM": jsonl(raw / "vllm_native.jsonl"),
        "SGLang": jsonl(raw / "sglang_native.jsonl"),
    }
    metrics = {
        "whole": whole_metrics(artifacts["whole"]),
        "boundary": boundary_metrics(artifacts["boundary"]),
        "cuda": cuda_metrics(artifacts["cuda"]),
        "Triton": triton_metrics(artifacts["Triton"]),
        "TVM": tvm_metrics(artifacts["TVM"]),
        "CUTLASS/CuTe": cutlass_metrics(artifacts["CUTLASS/CuTe"]),
        "vLLM": native_metrics(artifacts["vLLM"]),
        "SGLang": native_metrics(artifacts["SGLang"]),
    }
    kv_batch_path = root / "kv_request_batch/KV_REQUEST_BATCH_SUMMARY.json"
    metrics["kv_request_batch"] = (json.loads(kv_batch_path.read_text(encoding="utf-8"))
                                   if kv_batch_path.exists() else
                                   {"rows": 0, "successful_correct_rows": 0,
                                    "executed_attention_cases": 0,
                                    "status_counts": {"missing": 1}})
    available = {
        "Triton": metrics["Triton"]["cases"] > 0,
        "TVM": metrics["TVM"]["mapped_cases"] > 0,
        "CUTLASS/CuTe": metrics["CUTLASS/CuTe"]["successful_rows"] > 0,
        "vLLM": metrics["vLLM"]["successful_rows"] > 0,
        "SGLang": metrics["SGLang"]["successful_rows"] > 0,
        "Hexcute": False,
    }
    matrix = {(rq, framework): evidence_state(framework, rq, available)
              for rq in RQS for framework in FRAMEWORKS}
    make_svg(root / "figures/rq_framework_evidence.svg", matrix)
    make_performance_svg(root / "figures/audited_runtime_observations.svg", metrics)

    structure_counts = Counter(case.get("structure") for case in cases)
    phase_counts = Counter(case.get("phase") for case in cases)
    statuses = jsonl(root / "status.jsonl")
    failed_steps = [row for row in statuses if row.get("status") == "failed"]

    lines = [
        "# v10/v13 高判别力 real-world 子图补充实验报告",
        "",
        f"结果目录：`{root}`",
        "",
        "## 结论边界",
        "",
        f"本次清单包含 **{len(cases)}** 个新增 tensor contract；结构分布为 " +
        "、".join(f"{key}={value}" for key, value in sorted(structure_counts.items())) +
        "；phase 分布为 " + "、".join(f"{key}={value}" for key, value in sorted(phase_counts.items())) + "。",
        "每个 case 的 architecture dimensions、revision 与 source URL 来自原 640 清单中的 pinned real-world model config；sequence/KV length 是为制造 crossover、reuse、page、stride 因果轴而控制的 workload 变量。它们不是声称某模型训练时只使用该长度。",
        f"vLLM/SGLang 只在本地 Qwen checkpoint 上做原生 engine 的 layout/page 反事实，因此提供 workload 外部有效性，**不冒充** {len(cases)} 个 case 的 architecture 都能由该 checkpoint 执行。TVM 是显式标注的 projected boundary；CUTLASS 是 attention score-matrix boundary。",
        "",
        f"运行步骤失败数：**{len(failed_steps)}**。失败步骤见 `status.jsonl`；测得的 OOM/数值反例应保留为科学结果，启动器或缺依赖才是执行失败。",
        "",
        "![RQ × framework evidence](figures/rq_framework_evidence.svg)",
        "",
        "![Audited runtime observations](figures/audited_runtime_observations.svg)",
        "",
        "## 真实数据总览",
        "",
        "| 实验层 | 结果 |",
        "|---|---|",
        f"| {len(cases)}-case whole graph（PyTorch eager/TorchInductor） | rows={metrics['whole']['rows']}，correct pairs={metrics['whole']['paired']}，eager/compiled 中位比={fmt(metrics['whole']['median_eager_over_compiled'])}×；≥3%：Inductor={metrics['whole']['meaningful_winners_3pct']['TorchInductor']}、eager={metrics['whole']['meaningful_winners_3pct']['PyTorch eager']}、tie={metrics['whole']['meaningful_winners_3pct']['within_3pct']} |",
        f"| PyTorch stride/materialize 边界 | complete triplets={metrics['boundary']['complete_triplets']}；原始非-native winner={metrics['boundary']['direct_or_repair_wins']}，但 ≥3% 的有效胜出={metrics['boundary']['meaningful_non_native_wins_3pct']}，有效胜出中位增益={fmt(metrics['boundary']['median_meaningful_gain'])}× |",
        f"| CUDA-reference attention | rows={metrics['cuda']['rows']}，cases={metrics['cuda']['cases']}，跨 consumer winner 反转 cases={metrics['cuda']['cases_with_cross_edge_rank_inversion']} |",
        f"| explicit Triton | cases={metrics['Triton']['cases']}，extended raw wins={metrics['Triton']['extended_layout_wins']}，≥3% wins={metrics['Triton']['meaningful_extended_wins_3pct']}，有效胜出中位增益={fmt(metrics['Triton']['median_meaningful_extended_gain'])}× |",
        f"| TVM projected boundary | mapped cases={metrics['TVM']['mapped_cases']}，同候选跨 consumer 反转={metrics['TVM']['cases_with_consumer_rank_inversion']}，reuse crossover={metrics['TVM']['cases_with_reuse_crossover']} |",
        f"| CUTLASS/CuTe attention boundary | measured cases={metrics['CUTLASS/CuTe']['cases']}，success rows={metrics['CUTLASS/CuTe']['successful_rows']}，winner speedup 中位数={fmt(metrics['CUTLASS/CuTe']['median_winner_speedup'])}×，显式 score-matrix preflight skip={metrics['CUTLASS/CuTe']['oom_preflight_cases']} |",
        f"| vLLM native engine | success={metrics['vLLM']['successful_rows']}，layout/page paired regimes={metrics['vLLM']['paired_regimes']}，single/multi-request pairs={metrics['vLLM']['single_multi_request_pairs']}，多 request wall-time 中位加速={fmt(metrics['vLLM']['median_multi_request_wall_speedup'])}× |",
        f"| SGLang native engine | success={metrics['SGLang']['successful_rows']}，layout/page paired regimes={metrics['SGLang']['paired_regimes']}，single/multi-request pairs={metrics['SGLang']['single_multi_request_pairs']}，多 request wall-time 中位加速={fmt(metrics['SGLang']['median_multi_request_wall_speedup'])}× |",
        f"| vLLM/SGLang native KV writer request-batch | executed attention cases={metrics['kv_request_batch']['executed_attention_cases']}，rows={metrics['kv_request_batch']['rows']}，数值正确成功={metrics['kv_request_batch']['successful_correct_rows']}，状态={metrics['kv_request_batch']['status_counts']} |",
        "",
        "## 每个 RQ 在各框架上的判定",
        "",
        "图与下表中的 runtime 才是该框架本次实际运行证据；source-only 只证明机制/遗漏变量存在，不能证明性能；supporting-only 只是邻近问题的支持证据；N/A 表示框架不拥有该层决策。",
        "",
        "| RQ | Triton | TVM | CUTLASS/CuTe | vLLM | SGLang | Hexcute |",
        "|---|---|---|---|---|---|---|",
    ]
    for rq, name in RQS.items():
        lines.append("| " + rq + " " + name + " | " + " | ".join(matrix[(rq, framework)] for framework in FRAMEWORKS) + " |")
    lines += ["", "## RQ 逐项解释", ""]

    explanations = {
        "v10-RQ1": f"CUDA-reference 的 {metrics['cuda']['cases_with_cross_edge_rank_inversion']}/{metrics['cuda']['cases']} 个 attention case 出现 consumer/edge winner 改变；explicit Triton 是命名 layout 的独立复核。TVM/CUTLASS/serving 数据只能作邻近支持，不能都写成 v10-RQ1 直接验证。",
        "L-RQ1": f"CUDA 的共同 NHD/HND 候选反转为 {metrics['cuda']['cases_with_cross_edge_rank_inversion']}/{metrics['cuda']['cases']}；TVM 的共同 row-major/tiled 候选在 row/column consumer 间反转为 {metrics['TVM']['cases_with_consumer_rank_inversion']}/{metrics['TVM']['mapped_cases']}。说明只优化 producer 或单 consumer 的 layout 不是充分目标。",
        "L-RQ2": f"CUDA reference 直接比较共享 NHD、共享 HND 与 split+conversion；Triton 直接测七组多 consumer fanout，TVM 直接测 consumer-only row/column fanout（不含 producer/materialization）。vLLM/SGLang 原生 KV writer 在 {metrics['kv_request_batch']['executed_attention_cases']} 个 attention case 上配对 request batch=1/2/4/8，但仍只是 supporting evidence，不能仅凭 batch sweep 宣称解决 domain 分裂。",
        "L-RQ3": f"PyTorch 的 {metrics['boundary']['complete_triplets']} 个成组三策略比较直接测量 zero-copy direct 与每次 materialize；非 native 原始 winner 为 {metrics['boundary']['direct_or_repair_wins']}，达到预注册 ≥3% 门槛的是 {metrics['boundary']['meaningful_non_native_wins_3pct']}。TVM 的 reuse winner 在 {metrics['TVM']['cases_with_reuse_crossover']}/{metrics['TVM']['mapped_cases']} case 改变。",
        "L-RQ4": f"whole graph 配对 {metrics['whole']['paired']} case，且 {metrics['whole']['structures_with_backend_crossover']} 个结构内部出现 eager/compiled winner 改变；各框架多 shape 的 winner 变化支持静态全局策略存在 regret。硬件轴仍只测 A10，不能外推。",
        "L-RQ5": f"全部框架均为 single-GPU-blocked：卡 1 无法形成通信/placement 反事实。{len(cases)} 个 case 只能准备形状，不能将单卡 memcpy proxy 当成 collective 证据。",
        "L-RQ6": "运行时 primitive 可给迁移成本输入，但 vLLM/SGLang 没有安全 live-KV/state migration adapter；阈值与 hysteresis 只能由 measured-cost trace/controller 验证，当前原生框架单次 engine sweep 仍不足以声称完成在线重配置。",
        "L-RQ7": f"PyTorch boundary 有 {metrics['boundary']['complete_triplets']} 个 exact-shape 三策略配对，显式记录 stride、repair bytes 与正确性；这证明 layout 名相同不足以表达 zero-copy legality。Triton/TVM 的支持范围按其 adapter 所拥有的 stride/mapping 层解释。",
        "L-RQ8": "CUDA data×metadata/page-table 反事实是运行时证据；vLLM/SGLang/CUTLASS/Triton 本轮没有暴露可独立控制且配对的数据与 metadata placement，因而这些框架保持 source-only，不能从整体延迟反推 metadata layout。",
        "L-RQ9": f"explicit Triton 中 extended legal layout 原始胜出 {metrics['Triton']['extended_layout_wins']}/{metrics['Triton']['cases']}，达到 ≥3% 的是 {metrics['Triton']['meaningful_extended_wins_3pct']}/{metrics['Triton']['cases']}；TVM/CUTLASS 与 serving policy 的候选 winner 分布用于检验候选集合截断。只有在同 case、同语义、同正确性下超越 native subset 才算表达能力证据。",
        "L-RQ10": f"vLLM 与 SGLang 分别获得 {metrics['vLLM']['successful_rows']}、{metrics['SGLang']['successful_rows']} 条 native engine 成功记录；fixed-layout 的 page/block 配对可测 allocator coupling，但它仍不是这 {len(cases)} 个 case 的逐 architecture checkpoint 映射。CUDA page 粒度实验提供 kernel-level 配对。",
    }
    for rq, text in explanations.items():
        lines += [f"### {rq}: {RQS[rq]}", "", text, ""]

    lines += [
        "## 各框架实测 observation 与适用边界",
        "",
        f"- **PyTorch/TorchInductor**：{metrics['whole']['paired']} 个正确配对的 eager/compiled 中位 speedup 为 {fmt(metrics['whole']['median_eager_over_compiled'])}×；按结构中位数为 " +
        "、".join(f"{key}={value:.3f}×" for key, value in metrics['whole']['median_speedup_by_structure'].items()) + "。boundary 中达到 ≥3% 的非 native 胜出为 " +
        f"{metrics['boundary']['meaningful_non_native_wins_3pct']}/{metrics['boundary']['complete_triplets']}。这主要支持 L-RQ3/L-RQ4/L-RQ7。",
        f"- **Triton explicit adapter**：winner 分布为 `{metrics['Triton']['winners']}`；扩展候选 raw win={metrics['Triton']['extended_layout_wins']}，≥3% win={metrics['Triton']['meaningful_extended_wins_3pct']}。这直接支持 v10-RQ1、L-RQ1/L-RQ4/L-RQ7/L-RQ9，不覆盖持久 allocator/controller。",
        f"- **TVM**：{metrics['TVM']['mapped_cases']} 个 real-world boundary 投影有映射；row-major/tiled 在不同 consumer 间反转 {metrics['TVM']['cases_with_consumer_rank_inversion']} 次，reuse 改变最优策略 {metrics['TVM']['cases_with_reuse_crossover']} 次。它是编译器 boundary 证据，不是 serving KV-pool 证据。",
        f"- **CUTLASS/CuTe**：{metrics['CUTLASS/CuTe']['successful_rows']} 条成功 kernel 行，tiled-direct/transform-plus-flat winner 为 `{metrics['CUTLASS/CuTe']['winners']}`，winner speedup 中位数 {fmt(metrics['CUTLASS/CuTe']['median_winner_speedup'])}×；{metrics['CUTLASS/CuTe']['oom_preflight_cases']} 个显式 score-matrix case 被 preflight 跳过。",
        f"- **vLLM**：{metrics['vLLM']['successful_rows']}/{metrics['vLLM']['rows']} native engine 行成功；layout/page winner `{metrics['vLLM']['winners']}`；single/multi-request 配对 {metrics['vLLM']['single_multi_request_pairs']} 组，其中 multi-request 更快 {metrics['vLLM']['multi_request_faster_pairs']} 组。仍只对应固定 Qwen checkpoint。",
        f"- **SGLang**：{metrics['SGLang']['successful_rows']}/{metrics['SGLang']['rows']} native engine 行成功；winner `{metrics['SGLang']['winners']}`；single/multi-request 配对 {metrics['SGLang']['single_multi_request_pairs']} 组，其中 multi-request 更快 {metrics['SGLang']['multi_request_faster_pairs']} 组。NHD/HND 与 FlashInfer page-size 对照被分组，未把 backend-confounded 行混作 layout-only。",
        f"- **原生 KV writer batch 反事实**：{metrics['kv_request_batch']['successful_correct_rows']}/{metrics['kv_request_batch']['rows']} 行成功；batch>1 明确标记为 derived counterfactual，并将连续 request-major 与 16-token block-interleaved slot 分开。它证明 materialization 边界，不冒充完整 attention/serving。",
        "- **Hexcute**：A10/sm_86 与公开 A100/H100 artifact 不匹配，保持 `arch-blocked`；没有用 CUDA-reference 数据替代 Hexcute runtime。",
        "",
        "## 正确性反例与 preflight 边界",
        "",
        f"whole graph 有 {len(metrics['whole']['numerical_mismatches'])} 条数值不匹配。它们不是 launcher/环境失败，也没有通过放宽阈值伪装成成功；相应性能行从 paired speedup 中排除：",
        "",
        "| case | backend | structure | max abs | RMSE | 预声明规则 |",
        "|---|---|---|---:|---:|---|",
    ]
    for item in metrics["whole"]["numerical_mismatches"]:
        lines.append(f"| {item.get('case_id')} | {item.get('backend')} | {item.get('structure')} | {item.get('max_abs_error')} | {item.get('rmse')} | {item.get('correctness_rule')} |")
    if metrics["whole"]["numerical_mismatches"]:
        mismatch_structures = Counter(
            item.get("structure", "unknown")
            for item in metrics["whole"]["numerical_mismatches"]
        )
        lines += [
            "",
            "数值反例按结构分布为 `" + str(dict(mismatch_structures)) +
            "`。这些行保持 fail-closed；在没有更高精度 oracle 和预注册新阈值前，不通过放宽容差将其“修复”为通过。",
        ]
    else:
        lines += ["", "本次没有数值反例，因而不对其他轮次中的长 scan/cumsum 反例作硬编码归因。"]
    if metrics["CUTLASS/CuTe"]["oom_preflight_cases"]:
        lines += [
            "",
            f"CUTLASS 的 preflight skip 是 `{metrics['CUTLASS/CuTe']['oom_preflight_case_ids']}`。这些 case 超过 harness 的显式 score-matrix 安全上限；这不是 CUTLASS 编译失败，也不进入成功行统计。",
        ]
    else:
        lines += ["", "CUTLASS 本次没有 preflight skip。"]
    lines += [""]

    lines += [
        f"## 为什么这 {len(cases)} 个 case 对原 640 形成补充",
        "",
        f"原 640 的优势是模型覆盖广，但 prefill 多固定为 2048、decode KV 多固定为 8192，适合 prevalence/coverage，不适合识别 crossover。新增 {len(cases)}-case 清单固定真实模型 architecture，同时系统覆盖 short/long、prefill/decode、request batch、reuse/page/stride 敏感区间；因此能形成同一结构内部的正反对照。两者是互补关系，补充清单不替代 640。",
        "",
        "最关键的判据不是“模型更大”，而是同一 causal axis 两侧是否都采样：短/长 sequence、低/高 reuse、contiguous/strided、direct/materialize、native/extended layout、page/block 粒度。报告只将实际 paired counterfactual 计为性能证据。",
        "",
        "## 可复核产物",
        "",
        f"- case 清单：`{manifest_path}`",
        f"- whole graph：`{raw / 'whole_graph.jsonl'}`",
        f"- boundary：`{raw / 'boundary_layout.jsonl'}`",
        f"- CUDA/Triton/TVM/CUTLASS：`{raw}` 下对应 CSV",
        f"- vLLM/SGLang：`{raw / 'vllm_native.jsonl'}`、`{raw / 'sglang_native.jsonl'}`",
        f"- KV request batch：`{root / 'kv_request_batch/KV_REQUEST_BATCH_REPORT_CN.md'}`",
        f"- 执行状态：`{root / 'status.jsonl'}`",
        "",
        "## 与此前完整实验的关系",
        "",
        f"此前 640-case 结果继续作为 coverage 基线；非 OOM 软件失败的增量修复结果继续作为可靠性基线。本报告是新增的 {len(cases)}-case discriminative supplement，不写回或覆盖旧目录。最终论文级结论应同时引用：640 coverage、修复审计和 causal supplement 三类证据。",
    ]
    report = root / "PERSUASIVE_RQ_FRAMEWORK_REPORT_CN.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {"case_count": len(cases), "structures": dict(structure_counts),
               "phases": dict(phase_counts), "metrics": metrics,
               "evidence_matrix": {rq: {fw: matrix[(rq, fw)] for fw in FRAMEWORKS} for rq in RQS},
               "failed_steps": failed_steps, "report": str(report)}
    (root / "PERSUASIVE_RQ_RESULTS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report), "cases": len(cases),
                      "failed_steps": len(failed_steps)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
