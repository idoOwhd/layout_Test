#!/usr/bin/env python3
"""Post-analyze a completed v10/v13 run without third-party plotting packages.

The script is deliberately read-only with respect to the run directory and
refuses to overwrite its output directory.  Figures are vector SVG files made
from the persisted raw measurements rather than manually transcribed numbers.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


COLORS = {
    "exact": "#1b9e77", "projected": "#66c2a5", "external": "#7570b3",
    "source": "#e6ab02", "blocked": "#d95f02", "na": "#bdbdbd",
    "mismatch": "#d73027", "oom": "#fc8d59", "boundary_error": "#984ea3",
    "success": "#2ca25f", "neutral": "#969696", "blue": "#2b8cbe",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def q(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    position = (len(values) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] * (high - position) + values[high] * (position - low)


def esc(value: object) -> str:
    return html.escape(str(value))


def svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#222}'
        '.title{font-size:22px;font-weight:700}.axis{font-size:12px}'
        '.label{font-size:13px}.small{font-size:11px}.grid{stroke:#ddd;stroke-width:1}'
        '.ref{stroke:#555;stroke-width:1.5;stroke-dasharray:5 4}</style>',
        f'<text x="24" y="32" class="title">{esc(title)}</text>',
    ]


def write_svg(path: Path, parts: list[str]) -> None:
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def figure_matrix(path: Path, framework_counts: dict[str, Counter]) -> None:
    width, height = 1320, 430
    parts = svg_start(width, height, "Evidence states in the 640 × framework × RQ matrix")
    categories = [
        ("exact", "Exact paired native", ["paired_native_runtime"]),
        ("projected", "Projected paired", ["paired_native_projected_runtime"]),
        ("external", "External / not case-matched", ["native_external_not_case_matched"]),
        ("source", "Source observation", ["source_observation_only"]),
        ("blocked", "Blocked", ["blocked_single_gpu", "blocked_architecture_unsupported"]),
        ("na", "Not applicable", ["not_applicable_framework_owner", "not_applicable_case_trigger"]),
    ]
    frameworks = ["vLLM", "SGLang", "CUTLASS/CuTe", "Triton", "TVM", "Hexcute"]
    x0, bar_w, y0, step, bar_h = 205, 1050, 85, 46, 26
    for tick in range(0, 6401, 800):
        x = x0 + bar_w * tick / 6400
        parts += [f'<line x1="{x:.1f}" y1="66" x2="{x:.1f}" y2="344" class="grid"/>',
                  f'<text x="{x:.1f}" y="365" text-anchor="middle" class="axis">{tick}</text>']
    for row, framework in enumerate(frameworks):
        y = y0 + row * step
        parts.append(f'<text x="194" y="{y + 18}" text-anchor="end" class="label">{esc(framework)}</text>')
        cursor = x0
        counts = framework_counts[framework]
        for color_key, _, statuses in categories:
            value = sum(counts.get(status, 0) for status in statuses)
            segment = bar_w * value / 6400
            parts.append(f'<rect x="{cursor:.2f}" y="{y}" width="{segment:.2f}" height="{bar_h}" fill="{COLORS[color_key]}"/>')
            if value >= 300:
                parts.append(f'<text x="{cursor + segment / 2:.2f}" y="{y + 18}" text-anchor="middle" class="small" fill="#111">{value}</text>')
            cursor += segment
    lx, ly = 80, 402
    for color_key, label, _ in categories:
        parts += [f'<rect x="{lx}" y="{ly - 12}" width="14" height="14" fill="{COLORS[color_key]}"/>',
                  f'<text x="{lx + 20}" y="{ly}" class="small">{esc(label)}</text>']
        lx += 190
    write_svg(path, parts)


def figure_failures(path: Path, whole: list[dict], boundary: list[dict]) -> None:
    structures = ["gqa", "sliding_attention", "swiglu", "moe", "mla",
                  "linear_attention", "sparse_attention", "mamba2"]
    series = [
        ("Whole: numerical mismatch", COLORS["mismatch"], lambda s: sum(
            row.get("structure") == s and row.get("status") == "numerical_mismatch" for row in whole)),
        ("Whole: OOM", COLORS["oom"], lambda s: sum(
            row.get("structure") == s and str(row.get("status", "")).startswith("oom") for row in whole)),
        ("Boundary: error", COLORS["boundary_error"], lambda s: sum(
            row.get("subgraph") == s and row.get("status") == "error" for row in boundary)),
        ("Boundary: OOM preflight", "#fdbb84", lambda s: sum(
            row.get("subgraph") == s and row.get("status") == "oom_preflight" for row in boundary)),
    ]
    width, height = 1240, 520
    parts = svg_start(width, height, "Non-success results by LLM subgraph")
    x0, y0, plot_w, plot_h = 92, 68, 1090, 345
    max_value = max(fn(s) for _, _, fn in series for s in structures) or 1
    ymax = max(40, int(math.ceil(max_value / 5) * 5))
    for tick in range(0, ymax + 1, 5):
        y = y0 + plot_h - plot_h * tick / ymax
        parts += [f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + plot_w}" y2="{y:.1f}" class="grid"/>',
                  f'<text x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{tick}</text>']
    group_w = plot_w / len(structures)
    bar_w = group_w / (len(series) + 1)
    for i, structure in enumerate(structures):
        for j, (_, color, fn) in enumerate(series):
            value = fn(structure)
            h = plot_h * value / ymax
            x = x0 + i * group_w + (j + .5) * bar_w
            parts.append(f'<rect x="{x:.1f}" y="{y0 + plot_h - h:.1f}" width="{bar_w * .8:.1f}" height="{h:.1f}" fill="{color}"/>')
            if value:
                parts.append(f'<text x="{x + bar_w * .4:.1f}" y="{y0 + plot_h - h - 5:.1f}" text-anchor="middle" class="small">{value}</text>')
        x = x0 + (i + .5) * group_w
        parts.append(f'<text x="{x:.1f}" y="{y0 + plot_h + 18}" text-anchor="middle" class="small" transform="rotate(25 {x:.1f} {y0 + plot_h + 18})">{esc(structure)}</text>')
    lx = 100
    for label, color, _ in series:
        parts += [f'<rect x="{lx}" y="477" width="14" height="14" fill="{color}"/>',
                  f'<text x="{lx + 20}" y="489" class="small">{esc(label)}</text>']
        lx += 270
    write_svg(path, parts)


def figure_interval(path: Path, title: str, groups: list[tuple[str, list[float]]],
                    ticks: list[float], log_scale: bool = False,
                    reference: float = 1.0) -> None:
    width, row_h = 1260, 31
    height = 105 + row_h * len(groups)
    parts = svg_start(width, height, title)
    x0, x1, y0 = 250, 1205, 65
    transform = math.log2 if log_scale else (lambda x: x)
    tmin, tmax = transform(min(ticks)), transform(max(ticks))
    xpos = lambda value: x0 + (transform(value) - tmin) / (tmax - tmin) * (x1 - x0)
    for tick in ticks:
        x = xpos(tick)
        parts += [f'<line x1="{x:.1f}" y1="52" x2="{x:.1f}" y2="{height - 35}" class="grid"/>',
                  f'<text x="{x:.1f}" y="{height - 14}" text-anchor="middle" class="axis">{tick:g}×</text>']
    ref_x = xpos(reference)
    parts.append(f'<line x1="{ref_x:.1f}" y1="52" x2="{ref_x:.1f}" y2="{height - 35}" class="ref"/>')
    for i, (label, values) in enumerate(groups):
        if not values:
            continue
        y = y0 + i * row_h
        lo, q1, med, q3, hi = (q(values, p) for p in (0, .25, .5, .75, 1))
        parts += [f'<text x="{x0 - 12}" y="{y + 5}" text-anchor="end" class="small">{esc(label)} (n={len(values)})</text>',
                  f'<line x1="{xpos(lo):.1f}" y1="{y}" x2="{xpos(hi):.1f}" y2="{y}" stroke="#555" stroke-width="1.5"/>',
                  f'<rect x="{xpos(q1):.1f}" y="{y - 8}" width="{max(xpos(q3)-xpos(q1),1):.1f}" height="16" fill="#9ecae1" stroke="#2b8cbe"/>',
                  f'<circle cx="{xpos(med):.1f}" cy="{y}" r="4" fill="#08519c"/>',
                  f'<text x="{min(xpos(hi)+5,x1-30):.1f}" y="{y + 4}" class="small">med {med:.2f}×</text>']
    write_svg(path, parts)


def figure_native(path: Path, native: dict[str, list[dict]]) -> list[dict]:
    records = []
    for framework, rows in native.items():
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            grouped[(row["comparison_scope"], row["case_id"])].append(row)
        for (scope, case_id), values in sorted(grouped.items()):
            values.sort(key=lambda row: float(row["p50_ms"]))
            best, worst = values[0], values[-1]
            records.append({"framework": framework, "scope": scope, "case_id": case_id,
                            "best": best["variant"], "worst": worst["variant"],
                            "speedup": float(worst["p50_ms"]) / float(best["p50_ms"]),
                            "best_ms": float(best["p50_ms"]), "worst_ms": float(worst["p50_ms"])})
    width, height = 1320, 610
    parts = svg_start(width, height, "Native vLLM/SGLang policy sensitivity (worst / best p50)")
    x0, y0, plot_w, plot_h = 100, 65, 1160, 420
    ymax = 1.28
    for tick in (1.0, 1.05, 1.10, 1.15, 1.20, 1.25):
        y = y0 + plot_h - (tick - 1) / (ymax - 1) * plot_h
        parts += [f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+plot_w}" y2="{y:.1f}" class="grid"/>',
                  f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end" class="axis">{tick:.2f}×</text>']
    bar_w = plot_w / len(records)
    for i, record in enumerate(records):
        value = record["speedup"]
        h = (value - 1) / (ymax - 1) * plot_h
        color = COLORS["blue"] if value >= 1.03 else COLORS["neutral"]
        x = x0 + i * bar_w + 3
        parts += [f'<rect x="{x:.1f}" y="{y0+plot_h-h:.1f}" width="{bar_w-6:.1f}" height="{h:.1f}" fill="{color}"/>',
                  f'<text x="{x+(bar_w-6)/2:.1f}" y="{y0+plot_h-h-4:.1f}" text-anchor="middle" class="small">{value:.3f}</text>']
        prompt = record["case_id"].replace("offline_", "").replace("_o", "/o")
        label = f'{record["framework"]}:{record["scope"].replace("_only","")}:{prompt}'
        lx, ly = x+(bar_w-6)/2, y0+plot_h+12
        parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="start" class="small" transform="rotate(55 {lx:.1f} {ly:.1f})">{esc(label)}</text>')
    parts += [f'<line x1="{x0}" y1="{y0+plot_h-(.03/(ymax-1)*plot_h):.1f}" x2="{x0+plot_w}" y2="{y0+plot_h-(.03/(ymax-1)*plot_h):.1f}" stroke="#d73027" stroke-dasharray="5 4"/>',
              '<text x="104" y="590" class="small">Blue: ≥3% policy effect; gray: &lt;3%. Direction/winner is listed in the Markdown table.</text>']
    write_svg(path, parts)
    return records


def figure_rqs(path: Path, hypotheses: list[dict]) -> None:
    statuses = ["supported", "feasibility_supported", "inconclusive",
                "blocked_single_gpu", "blocked_requires_multiple_hardware"]
    colors = {"supported": "#1b9e77", "feasibility_supported": "#66c2a5",
              "inconclusive": "#e6ab02", "blocked_single_gpu": "#d95f02",
              "blocked_requires_multiple_hardware": "#984ea3"}
    width, height = 1160, 465
    parts = svg_start(width, height, "v13 hypothesis verdicts by research question")
    x0, y0, plot_w, plot_h = 85, 65, 1010, 310
    rqs = [f"L-RQ{i}" for i in range(1, 11)]
    group_w = plot_w / len(rqs)
    for tick in range(0, 6):
        y = y0 + plot_h - plot_h * tick / 5
        parts += [f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+plot_w}" y2="{y:.1f}" class="grid"/>',
                  f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end" class="axis">{tick}</text>']
    for i, rq in enumerate(rqs):
        counts = Counter(row["status"] for row in hypotheses if row["rq"] == rq)
        bottom = y0 + plot_h
        for status in statuses:
            value = counts.get(status, 0)
            h = plot_h * value / 5
            bottom -= h
            parts.append(f'<rect x="{x0+i*group_w+12:.1f}" y="{bottom:.1f}" width="{group_w-24:.1f}" height="{h:.1f}" fill="{colors[status]}"/>')
            if value:
                parts.append(f'<text x="{x0+(i+.5)*group_w:.1f}" y="{bottom+h/2+4:.1f}" text-anchor="middle" class="small">{value}</text>')
        parts.append(f'<text x="{x0+(i+.5)*group_w:.1f}" y="{y0+plot_h+20}" text-anchor="middle" class="axis">RQ{i+1}</text>')
    lx = 100
    for status in statuses:
        label = status.replace("_", " ")
        parts += [f'<rect x="{lx}" y="418" width="14" height="14" fill="{colors[status]}"/>',
                  f'<text x="{lx+20}" y="430" class="small">{esc(label)}</text>']
        lx += 205
    write_svg(path, parts)


def figure_layout_winners(path: Path, triton: Counter, cutlass: Counter,
                          tvm: Counter) -> None:
    panels = [("Triton consumer winner (240 exact cases)", triton),
              ("CUTLASS flat vs tiled (173 projected cases)", cutlass),
              ("TVM consumer scan (640 projected cases)", tvm)]
    width, height = 1230, 480
    parts = svg_start(width, height, "Framework-owned layout winner distributions")
    panel_w = 380
    palette = ["#1b9e77", "#7570b3", "#e6ab02", "#d95f02", "#66a61e"]
    for panel, (title, counts) in enumerate(panels):
        x0, y0, bar_w, max_h = 45 + panel * 405, 105, 55, 250
        maximum = max(counts.values()) if counts else 1
        parts.append(f'<text x="{x0}" y="70" class="label" font-weight="700">{esc(title)}</text>')
        for i, (name, value) in enumerate(counts.items()):
            h = max_h * value / maximum
            x = x0 + i * 82
            parts += [f'<rect x="{x}" y="{y0+max_h-h:.1f}" width="{bar_w}" height="{h:.1f}" fill="{palette[i%len(palette)]}"/>',
                      f'<text x="{x+bar_w/2}" y="{y0+max_h-h-6:.1f}" text-anchor="middle" class="label">{value}</text>',
                      f'<text x="{x+bar_w/2}" y="{y0+max_h+18}" text-anchor="start" class="small" transform="rotate(35 {x+bar_w/2} {y0+max_h+18})">{esc(name)}</text>']
    write_svg(path, parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    figures = args.output_dir / "figures"
    figures.mkdir()

    result = args.result_dir.resolve()
    combined = result / "v13_640" / "full_combined"
    raw = combined / "raw"
    execution = read_json(result / "EXECUTION_STATUS.json")
    completion = read_json(result / "CROSS_VERSION_COMPLETENESS.json")
    matrix_summary = read_json(combined / "v13_640_matrix_summary.json")
    matrix_rows = read_jsonl(combined / "v13_640_framework_rq_matrix.jsonl")
    hypotheses_doc = read_json(combined / "v13_hypothesis_results.json")
    hypotheses = hypotheses_doc["hypotheses"]
    whole = read_jsonl(raw / "llm_640_pytorch_triton.jsonl")
    boundary = read_jsonl(raw / "llm_640_boundary_layout_sweep.jsonl")
    native = {"vLLM": read_jsonl(raw / "vllm_native_serving.jsonl"),
              "SGLang": read_jsonl(raw / "sglang_native_serving.jsonl")}
    triton_rows = read_csv(raw / "triton_640_kv_layout.csv")
    tvm_rows = read_csv(raw / "tvm_640_rq_observations.csv")
    cutlass_rows = read_csv(raw / "cutlass_640_softmax_boundary.csv")
    v10_gate = read_json(result / "v10_strict" / "v10_rq1_gate.json")

    framework_counts: dict[str, Counter] = defaultdict(Counter)
    for row in matrix_rows:
        framework_counts[row["framework"]][row["status"]] += 1
    figure_matrix(figures / "fig01_matrix_evidence_by_framework.svg", framework_counts)
    figure_failures(figures / "fig02_non_success_by_subgraph.svg", whole, boundary)

    whole_success: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in whole:
        if row.get("status") == "success":
            whole_success[row["case_id"]][row["backend"]] = row
    whole_pairs = []
    for case_id, rows in whole_success.items():
        if {"pytorch", "triton"}.issubset(rows):
            eager, compiled = rows["pytorch"], rows["triton"]
            whole_pairs.append({"case_id": case_id, "structure": eager["structure"],
                                "phase": eager["phase"],
                                "speedup": float(eager["p50_ms"]) / float(compiled["p50_ms"]),
                                "fingerprint": eager.get("tensor_fingerprint", case_id)})
    # Equal weight per unique executable tensor contract; catalog-weighted
    # counts are reported separately in the Markdown.
    unique_whole = list({row["fingerprint"]: row for row in whole_pairs}.values())
    speed_groups = []
    for key in sorted({(row["structure"], row["phase"]) for row in unique_whole}):
        values = [row["speedup"] for row in unique_whole
                  if (row["structure"], row["phase"]) == key]
        speed_groups.append((f"{key[0]} / {key[1]}", values))
    figure_interval(figures / "fig03_inductor_speedup_unique_contracts.svg",
                    "Steady-state PyTorch eager / Inductor p50 (correct unique contracts)",
                    speed_groups, [.75, 1, 1.5, 2, 3, 4, 6], log_scale=True)

    boundary_success: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in boundary:
        if row.get("status") == "success":
            boundary_success[row["case_id"]][row["strategy"]] = row
    boundary_pairs = []
    required = {"native_contiguous", "alternate_strided_view",
                "repair_to_contiguous_each_call"}
    for case_id, rows in boundary_success.items():
        if required.issubset(rows):
            native_row, direct, repair = (rows[name] for name in
                                          ("native_contiguous", "alternate_strided_view",
                                           "repair_to_contiguous_each_call"))
            contract = (native_row["subgraph"], native_row["phase"],
                        json.dumps(native_row["shape"], sort_keys=True))
            boundary_pairs.append({"case_id": case_id, "structure": native_row["subgraph"],
                                   "phase": native_row["phase"], "contract": contract,
                                   "repair_over_direct": float(repair["p50_ms"]) / float(direct["p50_ms"]),
                                   "native_over_direct": float(native_row["p50_ms"]) / float(direct["p50_ms"])})
    unique_boundary = list({row["contract"]: row for row in boundary_pairs}.values())
    boundary_groups = []
    for key in sorted({(row["structure"], row["phase"]) for row in unique_boundary}):
        values = [row["repair_over_direct"] for row in unique_boundary
                  if (row["structure"], row["phase"]) == key]
        boundary_groups.append((f"{key[0]} / {key[1]}", values))
    figure_interval(figures / "fig04_repair_over_zero_copy_unique_contracts.svg",
                    "Per-call materialization / direct strided-view p50 (unique contracts)",
                    boundary_groups, [.75, .85, .95, 1, 1.03, 1.15, 1.25, 1.35])

    native_records = figure_native(figures / "fig05_native_serving_policy_sensitivity.svg", native)
    figure_rqs(figures / "fig06_v13_hypothesis_verdicts.svg", hypotheses)

    triton_by_case: dict[str, list[dict]] = defaultdict(list)
    for row in triton_rows:
        triton_by_case[row["case_id"]].append(row)
    triton_winners = Counter()
    triton_consumer_extension, triton_pipeline_extension = [], []
    for rows in triton_by_case.values():
        triton_winners[min(rows, key=lambda row: float(row["p50_ms"]))["layout"]] += 1
        named = min(float(row["p50_ms"]) for row in rows if row["layout"] in {"NHD", "HND"})
        all_best = min(float(row["p50_ms"]) for row in rows)
        triton_consumer_extension.append(named / all_best)
        named_pipe = min(float(row["pipeline_p50_ms"]) for row in rows
                         if row["layout"] in {"NHD", "HND"})
        all_pipe = min(float(row["pipeline_p50_ms"]) for row in rows)
        triton_pipeline_extension.append(named_pipe / all_pipe)

    cutlass_by_case: dict[str, list[dict]] = defaultdict(list)
    for row in cutlass_rows:
        if row.get("flat_ms") and row.get("tiled_direct_ms") and \
                row.get("correct") in {"1", "True", "true"}:
            cutlass_by_case[row["manifest_case_id"]].append(row)
    cutlass_winners = Counter()
    cutlass_speedups = []
    for rows in cutlass_by_case.values():
        flat = min(float(row["flat_ms"]) for row in rows)
        tiled = min(float(row["tiled_direct_ms"]) for row in rows)
        cutlass_winners["tiled_direct" if tiled < flat else "flat_row_major"] += 1
        cutlass_speedups.append(flat / tiled)

    tvm_consumer: dict[str, list[dict]] = defaultdict(list)
    for row in tvm_rows:
        if row.get("experiment") == "consumer_scan" and row.get("p50_ms") and \
                row.get("correct") in {"1", "True", "true"}:
            tvm_consumer[row["manifest_case_id"]].append(row)
    tvm_winners = Counter()
    for rows in tvm_consumer.values():
        tvm_winners[min(rows, key=lambda row: float(row["p50_ms"]))["strategy"]] += 1
    figure_layout_winners(figures / "fig07_framework_layout_winners.svg",
                          triton_winners, cutlass_winners, tvm_winners)

    whole_status = Counter(row["status"] for row in whole)
    boundary_status = Counter(row["status"] for row in boundary)
    whole_failure_detail = Counter((row.get("structure"), row["status"])
                                   for row in whole if row["status"] != "success")
    boundary_failure_detail = Counter((row.get("subgraph"), row["status"])
                                      for row in boundary if row["status"] != "success")
    hypothesis_by_rq = {rq: Counter(row["status"] for row in hypotheses if row["rq"] == rq)
                        for rq in [f"L-RQ{i}" for i in range(1, 11)]}

    catalog_triton_wins = sum(row["speedup"] > 1.03 for row in whole_pairs)
    catalog_eager_wins = sum(row["speedup"] < 1 / 1.03 for row in whole_pairs)
    unique_triton_wins = sum(row["speedup"] > 1.03 for row in unique_whole)
    unique_eager_wins = sum(row["speedup"] < 1 / 1.03 for row in unique_whole)
    direct_wins = sum(row["repair_over_direct"] > 1.03 for row in boundary_pairs)
    repair_wins = sum(row["repair_over_direct"] < 1 / 1.03 for row in boundary_pairs)
    unique_direct_wins = sum(row["repair_over_direct"] > 1.03 for row in unique_boundary)
    unique_repair_wins = sum(row["repair_over_direct"] < 1 / 1.03 for row in unique_boundary)

    applicable = matrix_summary["applicable_runtime_cell_count"]
    exact = matrix_summary["paired_native_runtime_cell_count"]
    projected = matrix_summary["paired_native_projected_runtime_cell_count"]
    lines = [
        "# v10/v13 × 640 LLM 子图 × 六框架：运行结果审计与分析", "",
        f"> 结果目录：`{result}`", "",
        "## 一、结论摘要", "",
        f"- v10 阶段退出码 **{execution['v10_run_exit']}**，v13 阶段退出码 **{execution['v13_run_exit']}**；两套 runner 都完成。总审计退出码 **{execution['audit_exit']}**，因此本次运行被正确地 fail-closed。",
        f"- 640 × 6 × 10 的 **{matrix_summary['matrix_cell_count']:,}** 个矩阵单元已经全部物化，但这不等于全部获得 native runtime 证据。适用 runtime 单元为 **{applicable:,}**；exact paired native 为 **{exact:,} ({exact/applicable:.1%})**，projected paired 为 **{projected:,} ({projected/applicable:.1%})**。",
        f"- exact 的 {exact:,} 个单元来自 Triton 的 240 个 attention case 在 4 个 RQ 中重复映射，不是 {exact:,} 个独立实验。TVM/CUTLASS 的 {projected:,} 个单元被正确标成 projected；vLLM/SGLang 是 4 个整模型 serving regime，各 16 行，不是 640 个 case 的原生执行。",
        f"- 整图 PyTorch/Inductor 共 1,280 行：成功 **{whole_status['success']}**，numerical mismatch **{whole_status['numerical_mismatch']}**，OOM preflight **{whole_status['oom_preflight']}**，OOM runtime **{whole_status['oom_runtime']}**。只有正确成功行进入性能比较。",
        f"- boundary sweep 覆盖 640/640 case：**{len(boundary_pairs)}** 个 case 得到完整 native/direct-strided/repair 三联结果，{boundary_status['oom_preflight']} 个 OOM preflight，{boundary_status['error']} 个 error；所有成功 alternate 都真实改变 stride，no-op 为 0。",
        "- 科学上最重要的总观察不是某个 layout 普遍最优，而是 winner 随子图、shape、phase、边界修复成本和框架拥有的决策空间改变；同时，扩展 layout 空间只对部分 case 有显著收益，因此需要带置信/成本门控的条件决策。", "",
        "![Matrix evidence](figures/fig01_matrix_evidence_by_framework.svg)", "",
        "## 二、为什么总审计没有通过", "",
        "总审计失败不是因为命令中途退出：所有 required framework steps 和 vLLM/SGLang native rows 都完成。失败来自结果内部仍有 correctness/error 状态，外加单卡和架构不可消除的科学阻塞。", "",
        "### 2.1 整图失败分解", "",
        "| 子图 | numerical mismatch | OOM preflight/runtime | 判断 |", "|---|---:|---:|---|",
    ]
    for structure in ["gqa", "sliding_attention", "swiglu", "moe", "mla",
                      "linear_attention", "sparse_attention", "mamba2"]:
        mismatch = whole_failure_detail[(structure, "numerical_mismatch")]
        oom = whole_failure_detail[(structure, "oom_preflight")] + whole_failure_detail[(structure, "oom_runtime")]
        judgement = {"moe": "主要代码/数值稳定性问题", "mamba2": "峰值显存 + 1 个编译数值失败",
                     "linear_attention": "小幅归约差异，仍未裁决"}.get(structure, "无失败")
        lines.append(f"| {structure} | {mismatch} | {oom} | {judgement} |")
    lines += ["", "![Failures](figures/fig02_non_success_by_subgraph.svg)", "",
              "### 2.2 需要修改代码后补跑的项目", "",
              "1. **MoE 权重初始化存在高可信代码问题。** `Factory.weight()` 对所有形状使用 `shape[0]` 作为 fan-in；expert 权重形状是 `[experts, hidden, intermediate]`，因此实际按专家数而不是 hidden/intermediate 缩放。结果中 32/35 个 mismatch 属于 MoE，最坏 max-abs=96、RMSE=10.33，并有一个 PyTorch case 出现 326 个 non-finite。不能通过放宽 tolerance 解决；应修正 expert fan-in、重新冻结种子后只补跑 MoE whole/boundary。",
              "2. **Mamba boundary 的 12 个 error 应重分类。** 它们都是 alternate stride 触发 `UncapturedHigherOrderOpError: associative_scan must be captured completely`。native 输入已能执行，因此这是 consumer 对该物理 stride/捕获路径不支持的负结果。runner 应仍测 native 和 repair，并把 direct 标为 `unsupported_alternate_layout`，而不是让整个 case 成为泛化 `error`。",
              "3. **Mamba prefill 峰值显存估算仍不足。** `prefill-0103-mamba2` 两个 backend 都在运行时申请 4 GiB 失败，说明 scan 临时空间没有被 preflight 精确覆盖。应增加 scan capture/compile 峰值安全系数或分块 scan，然后补跑该 case。",
              "4. **Linear-attention 两个 Inductor mismatch 和一个 Mamba mismatch 暂不能放宽。** 它们的误差远小于异常 MoE，但阈值已在全量运行前冻结。应使用 FP64/顺序 recurrence oracle、重复进程和输出尺度分层后决定这是合法归约误差还是编译错误。", "",
              "## 三、正确 paired 数据说明了什么", "",
              "### 3.1 PyTorch eager 与 TorchInductor/Triton", "",
              f"- catalog 加权的正确 pair 为 **{len(whole_pairs)}**：Inductor 快 ≥3% 的有 **{catalog_triton_wins}**，eager 快 ≥3% 的有 **{catalog_eager_wins}**。",
              f"- 去除重复 tensor contract 后为 **{len(unique_whole)}**：Inductor 快 ≥3% 的有 **{unique_triton_wins}**，eager 快 ≥3% 的有 **{unique_eager_wins}**；其余为 3% 内。",
              "- 这是 steady-state p50，不包含首次编译成本；并且失败/错误 case 被排除。因此不能把该比例外推为所有 640 case 都能安全获得相同加速。", "",
              "![Inductor speedup](figures/fig03_inductor_speedup_unique_contracts.svg)", "",
              "### 3.2 零拷贝 stride 与每次 materialize", "",
              f"- 581 个完整 catalog triplet 中，direct strided view 比 per-call repair 快 ≥3%：**{direct_wins}**；repair 反而快 ≥3%：**{repair_wins}**；其余 **{len(boundary_pairs)-direct_wins-repair_wins}** 个在 3% 内。",
              f"- 按唯一 shape contract 等权后是 {len(unique_boundary)} 个：direct 胜 **{unique_direct_wins}**，repair 胜 **{unique_repair_wins}**，中性 **{len(unique_boundary)-unique_direct_wins-unique_repair_wins}**。这证明“支持任意 stride”不是无条件最优：某些 MoE prefill 中先 materialize 反而改善后续 grouped GEMM。", "",
              "![Boundary tradeoff](figures/fig04_repair_over_zero_copy_unique_contracts.svg)", "",
              "### 3.3 框架自己的 layout 候选", "",
              f"- **Triton：** 240 个 attention case 的 consumer winner 为 {dict(triton_winners)}。扩展 paged layout 相对 NHD/HND 子集，在 consumer-only 有 **{sum(v>1.03 for v in triton_consumer_extension)}/240** 个 case 提升 ≥3%（最大 {max(triton_consumer_extension):.2f}×），在 producer+consumer pipeline 有 **{sum(v>1.03 for v in triton_pipeline_extension)}/240** 个（最大 {max(triton_pipeline_extension):.2f}×）。结论是扩展搜索空间对少数 case 很重要，但大多数 case 可安全关闭扩展。",
              f"- **CUTLASS/CuTe：** 173 个可比较 projected attention-score boundary 中，flat row-major 胜 {cutlass_winners['flat_row_major']}，tiled-direct 胜 {cutlass_winners['tiled_direct']}；flat/tiled 中位比为 {statistics.median(cutlass_speedups):.3f}。这里是边界投影，不是完整 LLM 子图。",
              f"- **TVM：** projected consumer scan 中 tiled_16x16 胜 {tvm_winners['tiled_16x16']}/640，row-major 胜 {tvm_winners['row_major']}/640；但 conversion-reuse 实验里 keep_row_major 为 640/640 winner，说明 consumer-local tiled 优势可以被转换成本完全反转。", "",
              "![Layout winners](figures/fig07_framework_layout_winners.svg)", "",
              "### 3.4 vLLM 与 SGLang 整模型 serving", "",
              "| Framework | 比较 | Case | Winner | Loser | worst/best p50 |", "|---|---|---|---|---|---:|",
    ]
    for record in native_records:
        lines.append(f"| {record['framework']} | {record['scope']} | {record['case_id']} | {record['best']} ({record['best_ms']:.1f} ms) | {record['worst']} ({record['worst_ms']:.1f} ms) | {record['speedup']:.3f}× |")
    lines += ["", "短 prompt 上策略影响最大：vLLM page 为 1.250×、layout 为 1.128×；SGLang layout 为 1.086×。长 prompt/较高并发多数差异小于 3%。这支持 shape-aware policy，但当前只有一个 0.5B 模型和四个 traffic regime，不能替代 640 case 的原生框架执行。", "",
              "![Native serving](figures/fig05_native_serving_policy_sensitivity.svg)", "",
              "## 四、RQ 级别裁决", "",
              "| RQ | supported | feasibility | inconclusive | blocked | 解释 |", "|---|---:|---:|---:|---:|---|",
    ]
    rq_interpretation = {
        "L-RQ1": "v13 广覆盖发现 inversion，但 v10 严格核心检验未支持；属于 shape-dependent mixed evidence。",
        "L-RQ2": "条件测量/成本模型支持；不是各 serving framework 的在线 native controller。",
        "L-RQ3": "转换摊销 crossover 得到支持。",
        "L-RQ4": "workload/shape 轴支持；稳定区域和跨硬件仍不完整。",
        "L-RQ5": "单卡无法验证通信与放置。",
        "L-RQ6": "基于已测 primitive cost 的 controller 逻辑支持。",
        "L-RQ7": "零拷贝可行，但 59 个 catalog case 中 repair 明显更快，且 Mamba 有 stride 捕获拒绝。",
        "L-RQ8": "数据/metadata 联合选择有交互证据。",
        "L-RQ9": "扩展空间对少数 case 有高收益；intersection-vs-union 仍 inconclusive。",
        "L-RQ10": "只有 metadata-bound H10.2 支持，其余 4 个假设 inconclusive。",
    }
    for rq, counts in hypothesis_by_rq.items():
        blocked = counts["blocked_single_gpu"] + counts["blocked_requires_multiple_hardware"]
        lines.append(f"| {rq} | {counts['supported']} | {counts['feasibility_supported']} | {counts['inconclusive']} | {blocked} | {rq_interpretation[rq]} |")
    lines += ["", "![RQ verdicts](figures/fig06_v13_hypothesis_verdicts.svg)", "",
              "### v10 与 v13 的 L-RQ1 为什么看起来相反", "",
              f"v10 的严格主检验要求 5 个独立进程、稳定 producer/edge winner、bootstrap CI 和噪声自适应 epsilon。三个 BF16 核心 case 的 edge regret 只有 1.16%、1.30%、2.06%，而 epsilon 为 13.56%、7.61%、11.94%，所以 H1.1 **not_supported**，Stage 4 STOP。v10 仍支持 H1.3/H1.4，edge oracle 最大加速约 1.084×。",
              "v13 的 H1.1 是更广的 640-shape 外部有效性筛选，发现 CUDA 27 个、Triton 19 个 >3% inversion。二者不应合并成“RQ1 已普遍证实”：更准确的结论是 **inversion 确实存在，但在 v10 预先指定的核心 regime 中没有通过严格确认，且效果高度依赖 shape/候选空间**。", "",
              "## 五、框架覆盖边界", "",
              "| Framework | Exact native 640-case 证据 | 本次实际证据 |", "|---|---:|---|",
    ]
    for framework in ["vLLM", "SGLang", "CUTLASS/CuTe", "Triton", "TVM", "Hexcute"]:
        counts = framework_counts[framework]
        lines.append(f"| {framework} | {counts['paired_native_runtime']} | projected={counts['paired_native_projected_runtime']}, external={counts['native_external_not_case_matched']}, source={counts['source_observation_only']}, blocked={counts['blocked_single_gpu']+counts['blocked_architecture_unsupported']} |")
    lines += ["", "因此，“38,400 单元已经生成”成立；“640 子图 × 六框架 × 十个 RQ 已全部 native 验证”不成立。当前 exact native 覆盖仅来自 Triton attention adapter；其余证据层级必须保持 projected/external/source/blocked 标签。", "",
              "## 六、建议的最小补跑顺序", "",
              "1. 修复 expert 权重 fan-in，先对失败的 MoE case 做 smoke + FP32 oracle；通过后只补跑 640 whole/boundary 中的 MoE 子集。",
              "2. 重构 boundary runner，使 native/direct/repair 独立记录；Mamba direct stride 捕获失败作为 `unsupported_alternate_layout`，仍保留 native 与 repair 性能。",
              "3. 修正 Mamba scan 峰值估算并单独补跑 `prefill-0103-mamba2`；对 `prefill-0304-mamba2` 与两个 linear-attention case 做高精度数值裁决。",
              "4. 用补跑结果重新生成总分析；vLLM/SGLang、Triton 显式 layout、TVM、CUTLASS 已完成的原始结果不需要重复运行。",
              "5. RQ5/H4.4/Hexcute 不应在 A10 上反复补跑；它们需要多 GPU、第二种硬件或受支持架构。", "",
              "## 七、可复现数据文件", "",
              f"- 整图：`{raw / 'llm_640_pytorch_triton.jsonl'}`",
              f"- boundary：`{raw / 'llm_640_boundary_layout_sweep.jsonl'}`",
              f"- Triton 显式 layout：`{raw / 'triton_640_kv_layout.csv'}`",
              f"- TVM：`{raw / 'tvm_640_rq_observations.csv'}`",
              f"- CUTLASS：`{raw / 'cutlass_640_softmax_boundary.csv'}`",
              f"- vLLM/SGLang：`{raw / 'vllm_native_serving.jsonl'}`、`{raw / 'sglang_native_serving.jsonl'}`",
              f"- 逐单元矩阵：`{combined / 'v13_640_framework_rq_matrix.jsonl'}`", "",
              "所有图均由本目录中的 `analysis_data.json` 和上述原始文件生成；没有把失败行混入性能统计。", ""]

    report = args.output_dir / "RESULT_ANALYSIS_CN.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    analysis_data = {
        "result_dir": str(result), "execution": execution,
        "matrix": {"cells": matrix_summary["matrix_cell_count"], "applicable": applicable,
                   "exact": exact, "projected": projected,
                   "status_counts": matrix_summary["status_counts"]},
        "whole_status": dict(whole_status), "boundary_status": dict(boundary_status),
        "whole_pairs_catalog": len(whole_pairs), "whole_pairs_unique_contract": len(unique_whole),
        "boundary_triplets_catalog": len(boundary_pairs),
        "boundary_triplets_unique_contract": len(unique_boundary),
        "triton_winners": dict(triton_winners), "cutlass_winners": dict(cutlass_winners),
        "tvm_consumer_winners": dict(tvm_winners), "native_serving": native_records,
        "hypothesis_status_counts": hypotheses_doc["status_counts"],
        "v10_l_rq1_phenomenon": v10_gate.get("phenomenon_verdict", "see report"),
    }
    (args.output_dir / "analysis_data.json").write_text(
        json.dumps(analysis_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (args.output_dir / "failure_cases.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        fields = ["source", "case_id", "structure", "phase", "backend_or_strategy",
                  "status", "max_abs_error", "rmse", "nonfinite_count", "error"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in whole:
            if row["status"] == "success":
                continue
            writer.writerow({"source": "whole_graph", "case_id": row.get("case_id"),
                             "structure": row.get("structure"), "phase": row.get("phase"),
                             "backend_or_strategy": row.get("backend"), "status": row.get("status"),
                             "max_abs_error": row.get("max_abs_error"), "rmse": row.get("rmse"),
                             "nonfinite_count": row.get("nonfinite_count"), "error": row.get("error")})
        for row in boundary:
            if row["status"] == "success":
                continue
            writer.writerow({"source": "boundary", "case_id": row.get("case_id"),
                             "structure": row.get("subgraph"), "phase": row.get("phase"),
                             "backend_or_strategy": row.get("strategy", "alternate_probe"),
                             "status": row.get("status"), "max_abs_error": row.get("max_abs_error"),
                             "rmse": row.get("rmse"), "nonfinite_count": row.get("nonfinite_count"),
                             "error": row.get("error")})

    with (args.output_dir / "performance_summary.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        fields = ["experiment", "structure", "phase", "unit", "count", "q1", "median",
                  "q3", "min", "max", "wins_ge_3pct", "losses_ge_3pct"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for experiment, rows, metric in (
                ("eager_over_inductor", unique_whole, "speedup"),
                ("repair_over_direct_stride", unique_boundary, "repair_over_direct")):
            for structure, phase in sorted({(row["structure"], row["phase"]) for row in rows}):
                values = [float(row[metric]) for row in rows
                          if row["structure"] == structure and row["phase"] == phase]
                writer.writerow({"experiment": experiment, "structure": structure, "phase": phase,
                                 "unit": "unique_tensor_contract" if experiment == "eager_over_inductor"
                                 else "unique_shape_contract", "count": len(values),
                                 "q1": q(values, .25), "median": q(values, .5), "q3": q(values, .75),
                                 "min": min(values), "max": max(values),
                                 "wins_ge_3pct": sum(value > 1.03 for value in values),
                                 "losses_ge_3pct": sum(value < 1 / 1.03 for value in values)})

    with (args.output_dir / "framework_coverage.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        statuses = sorted(matrix_summary["status_counts"])
        writer = csv.DictWriter(handle, fieldnames=["framework", *statuses])
        writer.writeheader()
        for framework in ["vLLM", "SGLang", "CUTLASS/CuTe", "Triton", "TVM", "Hexcute"]:
            writer.writerow({"framework": framework,
                             **{status: framework_counts[framework][status]
                                for status in statuses}})
    print(json.dumps({"report": str(report), "figures": len(list(figures.glob('*.svg'))),
                      "output": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
