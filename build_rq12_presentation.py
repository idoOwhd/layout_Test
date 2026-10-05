#!/usr/bin/env python3
"""CPU-only audit/plots of completed RQ1/2. Never modifies an input run.

Run with a Python containing numpy and matplotlib, e.g. /usr/bin/python3.
No torch import, model download, kernel launch, or experiment scheduling.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics as S
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FRAMES = ["pytorch", "triton", "cutlass", "tvm", "vllm", "sglang"]
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
STATUSES = ["rq1_supported_in_candidate_space", "no_layout_conflict_in_candidate_space",
            "inconclusive_noise_or_winner_instability", "insufficient_layouts"]
LABELS = ["Supported", "No conflict found", "Inconclusive", "Aliased layouts"]
STATUS_COLORS = ["#009E73", "#0072B2", "#E69F00", "#A0A0A0"]


def load(path):
    return json.loads(path.read_text())


def write_json(path, obj):
    with path.open("x") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def csv_rows(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"no data for {path}")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                        for k, v in r.items()})


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def equal(a, b):
    if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-11):
        raise ValueError(f"derived/stored value mismatch: {a} != {b}")


def rows_hashed(path, provenance):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for number, line in enumerate(f, 1):
            h.update(line)
            if line.strip():
                yield number, json.loads(line)
    provenance.append({"path": str(path.resolve()), "bytes": path.stat().st_size,
                       "sha256": h.hexdigest()})


def shape(case):
    if "dimensions" in case:
        d = case["dimensions"]
        b,q,kv=case['request_batch'],case['query_length'],case['kv_length']
        t=b*q;h=int(d['hidden_size']);hq=int(d.get('num_attention_heads') or 1)
        hk=int(d.get('num_key_value_heads') or 1);width=int(d.get('head_dim') or 128)
        inter=int(d.get('moe_intermediate_size') or d.get('intermediate_size') or h)
        family=case['family']
        # Report actual executor operands, not unused source-model metadata.
        if family in {'projection_gdn','state_prefill_decode'}:
            kh=int(d.get('linear_num_key_heads') or 1);vh=int(d.get('linear_num_value_heads') or kh)
            dk=int(d.get('linear_key_head_dim') or 128);dv=int(d.get('linear_value_head_dim') or (dk if family=='projection_gdn' else 128))
            if family=='projection_gdn':return f"B={b},q={q},hidden={h},Hk_linear={kh},Hv_linear={vh},Dk={dk},Dv={dv}"
            return f"B={b},prefill_q={q},Hv_linear={vh},Dk={dk},Dv={dv},decode_horizon={case['decode_horizon']}"
        if family in {'moe_dispatch_expert','moe_gemm_combine'}:
            experts=int(d.get('num_experts') or 1);topk=int(d.get('experts_per_token') or 1)
            if family=='moe_dispatch_expert':topk=min(experts,topk)
            return f"tokens={t},hidden={h},I={inter},experts={experts},topk={topk},routes={t*topk}"
        if family in {'projection_swiglu_down','gemm_bias_gemm','reduce_norm_gemm','host_weight_gemm'}:
            return f"tokens={t},hidden={h},I={inter}"
        if family in {'sparse_gather_sdpa','host_kv_sdpa'}:
            if family=='sparse_gather_sdpa':
                hq=int(d.get('index_n_heads') or d.get('index_heads') or hq)
                width=int(d.get('index_head_dim') or width);hk=1
            selected=min(kv,max(16,2*q))
            return f"B={b},q={q},source_KV={kv},selected_S={selected},actual_Hq={hq},actual_Hkv={hk},actual_D={width}"
        if family=='projection_sdpa':return f"B={b},q={q},actual_KV={q},Hq={hq},Hkv={hk},D={width},hidden={h}"
        if family in {'kv_writer_flashinfer','kv_writer_swa_flashinfer'}:
            return f"B={b},q={q},KV={kv},Hq={hq},Hkv={hk},D={width},page={case['page_size']}"
        return f"B={b},q={q},KV={kv},Hq={hq},D={width},hidden={h}"
    d = json.loads(case["pinned_config_json"])
    return f"B={case['batch']},q={case['query_length']},kv={case['kv_lengths']},hidden={d['hidden_size']},Hq={d['num_attention_heads']},Hkv={d['num_key_value_heads']},D={d['head_dim']}"


def audit_rq1(run, out, provenance):
    saved = load(run / "RQ1_DIVERSITY_FINDINGS.json")
    analyzer = module(ROOT / "analyze_rq1_diversity.py", "rq1_audit_analysis")
    preflight = load(run / "preflight.json")
    current_sha = hashlib.sha256((ROOT / "analyze_rq1_diversity.py").read_bytes()).hexdigest()
    if current_sha != preflight["source_sha256"]["analyze_rq1_diversity.py"]:
        raise ValueError("RQ1 analyzer changed from merged-run recorded fingerprint")
    grouped = defaultdict(list)
    measurement_rows = []
    origins = Counter()
    statuses = Counter()
    case_data = {}
    versions = {}
    cells = set()
    for path in sorted((run / "raw").glob("*.jsonl")):
        for line, row in rows_hashed(path, provenance):
            key = (row["case_id"], row["consumer_mode"])
            if row["device"] != "NVIDIA A10":
                raise ValueError("unexpected RQ1 device")
            statuses[row["status"]] += 1
            origins[row.get("measurement_origin", str(path))] += 1
            case_data[key] = row["case"]
            versions[row["consumer_mode"]] = row["installed_package_versions"]
            cells.add((*key, row["process_repetition"]))
            compact = {k: row[k] for k in ("stage", "strategy", "process_repetition", "status", "p50_ms", "correctness") if k in row}
            if row["status"] == "success":
                assert row["correctness"]["correct"]
                samples = row["samples_ms"]
                assert len(samples) == preflight["iterations"] == 30
                assert all(math.isfinite(x) and x > 0 for x in samples)
                # The historical worker uses the upper middle order statistic.
                equal(row["p50_ms"], sorted(samples)[len(samples)//2])
            grouped[key].append(compact)
            measurement_rows.append({"case_id": key[0], "consumer_mode": key[1],
                "family": row["family"], "process": row["process_repetition"],
                "stage": row.get("stage"), "strategy": row.get("strategy"),
                "status": row["status"], "measured_p50_ms": row.get("p50_ms"),
                "samples_ms": row.get("samples_ms"), "boundary": row.get("boundary"),
                "consumer_kernel": row.get("consumer_kernel"),
                "correctness": row.get("correctness"), "raw_file": str(path), "line": line,
                "measurement_origin": row.get("measurement_origin")})
    saved_by = {(f["case_id"], f["consumer_mode"]): f for f in saved["findings"]}
    assert len(saved_by) == len(grouped)
    findings = []
    for key, rr in sorted(grouped.items()):
        result = analyzer.assess(rr)
        old = saved_by[key]
        assert result["status"] == old["status"], key
        if "median_held_out_speedup" in result:
            equal(result["median_held_out_speedup"], old["median_held_out_speedup"])
            equal(result["mean_held_out_regret_fraction"], old["mean_held_out_regret_fraction"])
        findings.append({**old, **result})
    counts = Counter(f["status"] for f in findings)
    assert dict(counts) == saved["case_mode_result_counts"]
    assert dict(statuses) == saved["measurement_status_counts"]
    coverage = load(run / "correctness_coverage.json")
    assert coverage["controlled_hybrid_correctness_and_coverage_pass"]
    assert len(cells) == coverage["checked_case_mode_process_cells"]
    unique = {f["case_id"]: f["case"] for f in findings}
    audit = {"run": str(run), "unique_executable_case_ids": len(unique), "case_consumer_objects": len(findings),
        "case_consumer_process_cells": len(cells), "classifications": dict(counts),
        "classification_by_consumer": {m: dict(Counter(f["status"] for f in findings if f["consumer_mode"] == m)) for m in ["torch", "vllm", "sglang"]},
        "unique_case_by_family": dict(Counter(c["family"] for c in unique.values())),
        "objects_by_family": dict(Counter(f["family"] for f in findings)),
        "model_provenance_count": len({p["model_id"] for c in unique.values() for p in c["provenance"]}),
        "batch_values": sorted({c["request_batch"] for c in unique.values()}),
        "query_values": sorted({c["query_length"] for c in unique.values()}),
        "kv_values": sorted({c["kv_length"] for c in unique.values()}),
        "versions": versions, "preflight": preflight, "coverage": coverage,
        "merge_provenance": load(run / "MERGE_PROVENANCE.json"),
        "all_classifications_recomputed_from_raw": True,
        "raw_p50_recomputed": True, "native_PR_replays": saved["native_PR_replays_completed"],
        "source_registry": {k: v for k, v in load(run / "source_coverage/all_source_registry.json").items() if k != "records"},
        "raw_origins": dict(origins)}
    csv_rows(out / "data/RQ1_ALL_MEASUREMENTS.csv", measurement_rows)
    csv_rows(out / "data/RQ1_ALL_OBJECTS.csv", ({"case_id": f["case_id"], "consumer": f["consumer_mode"],
        "family": f["family"], "model": f["case"]["model_id"], "shape": shape(f["case"]),
        "effective_executable_contract": f["case"]["executable_contract"],
        "status": f["status"], "local_layout": f.get("producer_local_layout"),
        "local_edge_strategy": f.get("local_layout_best_legal_edge"),
        "joint_edge_strategy": f.get("complete_edge_selected_strategy"),
        "speedup_median_of_process_ratios": f.get("median_held_out_speedup"),
        "mean_latency_reduction": f.get("mean_held_out_regret_fraction"),
        "regret_95ci": f.get("regret_95ci"), "local_edge_ms": f.get("measured_local_edge_ms"),
        "joint_edge_ms": f.get("measured_oracle_edge_ms"), "source_url": f["case"]["source_url"]} for f in findings))
    return findings, measurement_rows, audit


def compact_case(c):
    return {k: c[k] for k in ("case_id", "model_id", "model_revision", "config_source", "config_sha256",
        "phase", "batch", "query_length", "kv_length", "kv_lengths", "stage", "stage_label", "decoder_blocks",
        "semantic_node_count", "semantic_operator_kind_count", "graph_contract_sha256", "pinned_config_json")}


def audit_rq2(growth, conditional, out, provenance):
    manifest = load(growth / "design_manifest.json")
    cases = {c["case_id"]: compact_case(c) for c in manifest["cases"]}
    corrected = {(r["framework"], r["expanded_case_id"]): r for r in load(growth / "holdout_statistics_strict/holdout_estimand_audit.json")}
    sys.path.insert(0, str(ROOT))
    from audit_rq2_holdout_statistics import paired_bootstrap
    points = []; graphs = []; metadata = {}; counts = {}; rawtypes = {}
    for fw in FRAMES:
        print(f"[offline] audit RQ2 growth {fw}", flush=True)
        tally = Counter(); total_candidates = 0; exact = 0; seen = set(); seen_pairs = set()
        for line, row in rows_hashed(growth / (fw + ".jsonl"), provenance):
            kind = row["record_type"]; tally[kind] += 1
            if kind == "run_metadata":
                metadata[fw] = row
                assert row["gpu"] == "NVIDIA A10" and row["args"]["mode"] == "full"
            elif kind == "graph":
                cid = row["case"]["case_id"]
                assert cid not in seen and cid in cases
                seen.add(cid)
                assert row["executed_graph_status"] == "correctness_verified_complete_controlled_graph"
                for candidate in row["candidates"]:
                    assert candidate["correctness"]["correct"]
                    assert all(x > 0 and math.isfinite(x) for x in candidate["train_samples_ms"])
                    equal(S.median(candidate["train_samples_ms"]), candidate["train_median_ms"])
                assert len(row["candidates"]) == row["measured_candidate_count"] == row["attempted_candidate_count"]
                total_candidates += len(row["candidates"])
                exact += row["search_scope"] == "exact_declared_space"
                graphs.append({"framework": fw, "case_id": cid, "stage": cases[cid]["stage"],
                    "axes": row["axes"], "best_train_bits": row["best_train_plan"],
                    "best_train_ms": row["best_train_ms"], "near_plans": row["near_plans"],
                    "search_scope": row["search_scope"], "candidate_count": len(row["candidates"]),
                    "raw_file": str(growth / (fw+".jsonl")), "line": line})
            elif kind == "growth_pair":
                cid = row["expanded_case_id"]; assert cid not in seen_pairs; seen_pairs.add(cid)
                a = row["holdout"]["frozen_ms"]; b = row["holdout"]["free_ms"]
                assert row["holdout"]["free_post_correctness"]["correct"] and row["holdout"]["frozen_post_correctness"]["correct"]
                assert len(a) == len(b) == 5 and all(x > 0 and math.isfinite(x) for x in a+b)
                ratio = S.median(a)/S.median(b)
                equal(ratio, row["speedup_frozen_over_free"])
                audit = corrected[(fw, cid)]; stat = audit["corrected_estimand_statistics"]
                equal(ratio, stat["ratio_of_medians"])
                assert audit["same_plan"] == row["same_plan"]
                positive = not row["same_plan"] and stat["ratio_of_medians_paired_bootstrap_95_interval"][0] > 1.03
                # Independently replay every reported positive interval and deterministic A/A controls.
                if positive or len(seen_pairs) <= 2:
                    assert paired_bootstrap(a, b) == stat
                points.append({"protocol": "GROWTH-NEARSET", "framework": fw, "case_id": cid,
                    "small_case_id": row["small_case_id"], "case": cases[cid], "restricted_ms": a, "free_ms": b,
                    "ratio": ratio, "ci": stat["ratio_of_medians_paired_bootstrap_95_interval"],
                    "positive": positive, "same_choice": row["same_plan"], "search_scope": row["search_scope"],
                    "old_axes": row["old_axes"], "axes": row["expanded_axes"],
                    "restricted_choice": row["frozen_plan"], "free_choice": row["free_plan"],
                    "old_near_set_size": row["old_near_set_size"],
                    "survival_fraction": row["old_near_set_survival_fraction"],
                    "minimum_changes": row["minimum_old_axis_changes_to_expanded_near_set"],
                    "raw_file": str(growth / (fw+".jsonl")), "line": line})
        assert len(seen) == 1020 and len(seen_pairs) == 918
        assert tally["completion"] == 1 and not any(tally[k] for k in ("incorrect_candidate", "group_failure", "not_completed_case"))
        counts[fw] = {"graphs": len(seen), "growth_pairs": len(seen_pairs), "correct_candidates": total_candidates, "exact_graphs": exact}
        rawtypes[fw] = dict(tally)
    conditional_metadata = {}; conditional_types = {}
    for fw in FRAMES:
        print(f"[offline] audit RQ2 conditional {fw}", flush=True)
        tally = Counter(); domain_ids = set(); anchor_ids = set()
        for line, row in rows_hashed(conditional / (fw + ".jsonl"), provenance):
            kind = row["record_type"]; tally[kind] += 1
            if kind == "metadata":
                conditional_metadata[fw] = row
                assert row["gpu"] == "NVIDIA A10" and row["mode"] == "full"
            elif kind in {"conditional_domain_result", "runtime_anchor_result"}:
                protocol = "QKV-DUAL" if kind == "conditional_domain_result" else "REMOTE-KV-ANCHOR"
                cid = row["case"]["case_id"]; ids = domain_ids if protocol == "QKV-DUAL" else anchor_ids
                assert cid not in ids; ids.add(cid)
                train = row.get("training", row.get("training_factorial"))
                for t in train:
                    assert t["pre_correctness"]["correct"] and t["post_correctness"]["correct"]
                    assert len(t["measured_samples_ms"]) == 10
                    equal(S.median(t["measured_samples_ms"]), t["measured_median_ms"])
                hold = row["holdout"]; raw = hold["raw_measured_holdout"]
                assert all(x["correct"] for x in raw["post_correctness_checks"])
                a = raw["restricted_ms"]; b = raw["free_ms"]
                assert len(a) == len(b) == 5 and all(x > 0 and math.isfinite(x) for x in a+b)
                stat = hold["statistics"]; ratio = S.median(a)/S.median(b); equal(ratio, stat["ratio_of_medians"])
                positive = not hold["same_choice"] and stat["ratio_of_medians_paired_bootstrap_95_interval"][0] > 1.03
                if positive or len(ids) <= 2:
                    assert paired_bootstrap(a, b) == stat
                if protocol == "QKV-DUAL":
                    restricted = min((t for t in train if t["choice"]["explicit_output_representation_count"] == 1), key=lambda t: t["measured_median_ms"])
                    free = min(train, key=lambda t: t["measured_median_ms"])
                else:
                    changed = [t for t in train if t["choice"]["anchor"] == 1]
                    restricted = min((t for t in changed if t["choice"]["upstream"] in row["old_near_upstream_choices_with_NHD_anchor"]), key=lambda t: t["measured_median_ms"])
                    free = min(changed, key=lambda t: t["measured_median_ms"])
                assert hold["same_choice"] == (restricted["choice"] == free["choice"])
                points.append({"protocol": protocol, "framework": fw, "case_id": cid, "case": cases[cid],
                    "restricted_ms": a, "free_ms": b, "ratio": ratio,
                    "ci": stat["ratio_of_medians_paired_bootstrap_95_interval"], "positive": positive,
                    "same_choice": hold["same_choice"], "restricted_choice": restricted["choice"],
                    "free_choice": free["choice"], "training": [{"choice": t["choice"], "median_ms": t["measured_median_ms"]} for t in train],
                    "conditional_K": row.get("conditional_explicit_output_K_epsilon"),
                    "factors": row.get("factors"), "raw_file": str(conditional / (fw+".jsonl")), "line": line})
        assert len(domain_ids) == 1020 and len(anchor_ids) == 666
        assert tally["anchor_semantically_inapplicable"] == 354 and tally["completion"] == 1
        assert not tally["case_failure"] and not tally["group_failure"]
        conditional_types[fw] = dict(tally)
    summary = []
    for fw in FRAMES:
        for protocol in ["GROWTH-NEARSET", "QKV-DUAL", "REMOTE-KV-ANCHOR"]:
            pp = [p for p in points if p["framework"] == fw and p["protocol"] == protocol]
            changed = [p for p in pp if not p["same_choice"]]
            summary.append({"framework": fw, "protocol": protocol, "pairs": len(pp),
                "same_choice": len(pp)-len(changed), "different_choice": len(changed),
                "exploratory_positive": sum(p["positive"] for p in pp),
                "median_ratio_all": S.median(p["ratio"] for p in pp),
                "median_ratio_different": S.median(p["ratio"] for p in changed) if changed else None,
                "conditional_K2_training_only": sum(p.get("conditional_K") == 2 for p in pp),
                "ratio_over_1_03_but_not_positive": sum(p["ratio"] > 1.03 and not p["positive"] for p in pp),
                "AA_CI_lower_over_1_03": sum(p["same_choice"] and p["ci"][0] > 1.03 for p in pp)})
    csv_rows(out / "data/RQ2_ALL_COMPARISONS.csv", ({**{k: v for k, v in p.items() if k != "case"},
        "model": p["case"]["model_id"], "stage": p["case"]["stage"], "shape": shape(p["case"]),
        "measured_restricted_median_ms": S.median(p["restricted_ms"]),
        "measured_free_median_ms": S.median(p["free_ms"])} for p in points))
    csv_rows(out / "data/RQ2_GRAPH_CHOICES.csv", graphs)
    csv_rows(out / "data/RQ2_SUMMARY.csv", summary)
    audit = {"growth_run": str(growth), "conditional_run": str(conditional), "growth_counts": counts,
        "growth_raw_types": rawtypes, "conditional_raw_types": conditional_types,
        "summary": summary, "metadata": metadata, "conditional_metadata": conditional_metadata,
        "all_raw_point_estimates_recomputed": True, "all_reported_positive_CIs_recomputed": True,
        "CI_method": "paired bootstrap, 2000 draws, seed 1234; saved full audit for other growth CIs",
        "source_replay_coverage": {k: v for k, v in load(growth / "SOURCE_REPLAY_COVERAGE.json").items() if k != "records"},
        "requirements": load(growth / "REQUIREMENTS_COMPLETENESS.json"),
        "manifests": {k: v for k, v in manifest.items() if k not in ("cases", "nested_pairs")}}
    return points, graphs, cases, audit


def savefig(fig, out, name, figure_manifest, data, caption):
    # PDF embeds fonts; SVG retains vector text as paths for portability.
    fig.savefig(out / "figures" / (name+".pdf"), bbox_inches="tight")
    fig.savefig(out / "figures" / (name+".svg"), bbox_inches="tight")
    fig.savefig(out / "figures" / (name+".png"), bbox_inches="tight", dpi=160)
    figure_manifest.append({"figure": name, "data_tables": data, "caption": caption})
    plt.close(fig)


def plots_rq1(findings, raw, out, manifest):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                        "pdf.fonttype": 42, "svg.fonttype": "path"})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    modes = ["torch", "vllm", "sglang"]; left = np.zeros(3)
    for status, label, color in zip(STATUSES, LABELS, STATUS_COLORS):
        v = np.array([sum(f["status"] == status and f["consumer_mode"] == m for f in findings) for m in modes])
        ax.barh(modes, v, left=left, color=color, label=label)
        for i, n in enumerate(v):
            if n: ax.text(left[i]+n/2, i, str(n), ha="center", va="center", fontsize=9)
        left += v
    ax.set_xlabel("case x consumer environment objects"); ax.set_title("A. What was actually established")
    ax.legend(loc="upper center", bbox_to_anchor=(.55, -.16), ncol=2, fontsize=9)
    for m, color in zip(modes, COLORS):
        values = sorted(f["median_held_out_speedup"] for f in findings if f["consumer_mode"] == m and "median_held_out_speedup" in f)
        bx.step(values, np.arange(1, len(values)+1)/len(values), where="post", label=f"{m} (n={len(values)})", color=color)
    bx.axvline(1, color="black", linestyle="--", lw=1); bx.set_xscale("log")
    bx.set_xlabel("Median paired-process speedup (local edge / joint edge)")
    bx.set_ylabel("Cumulative fraction"); bx.set_title("B. All identifiable objects, not only positives")
    bx.legend(); bx.grid(alpha=.18)
    fig.suptitle("RQ1 | Local producer optimum is not always an edge optimum", fontsize=15)
    savefig(fig, out, "RQ1_01_coverage_and_distribution", manifest, ["RQ1_ALL_OBJECTS.csv"],
            "左图为854个对象的原判据分类；右图包括全部724个具有至少两个有效layout的对象。不确定结果不是支持证据。")
    positives = [f for f in findings if f["status"] == STATUSES[0]]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.7), layout="constrained")
    labels = [f"P{i+1} {f['family']}\n{f['consumer_mode']}" for i, f in enumerate(positives)]
    y = np.arange(len(positives))
    for j, (key, label, color) in enumerate([("measured_local_edge_ms", "P-only selected + best repair", COLORS[0]),
                                            ("measured_oracle_edge_ms", "Joint edge selection", COLORS[2])]):
        vals = [1000*S.median(f[key]) for f in positives]
        yy = y + (.16 if j else -.16)
        axes[0].barh(yy, vals, height=.3, label=label, color=color, alpha=.8)
        for k, f in enumerate(positives):
            axes[0].scatter(np.array(f[key])*1000, np.repeat(yy[k], 3), s=13, color="black", zorder=5)
    axes[0].set_yticks(y, labels); axes[0].invert_yaxis(); axes[0].set_xlabel("Complete measured edge time (microseconds)")
    axes[0].legend(fontsize=8); axes[0].set_title("A. All 7 classified positives; dots = 3 held-out processes")
    vals = np.array([f["mean_held_out_regret_fraction"]*100 for f in positives])
    lows = np.array([f["regret_95ci"][0]*100 for f in positives]); highs = np.array([f["regret_95ci"][1]*100 for f in positives])
    axes[1].errorbar(vals, y, xerr=[vals-lows, highs-vals], fmt="o", color=COLORS[2], capsize=3)
    axes[1].set_yticks(y, [f"P{i+1}: {f['median_held_out_speedup']:.3f}x" for i, f in enumerate(positives)])
    axes[1].invert_yaxis(); axes[1].axvline(0, color="grey", linestyle="--")
    axes[1].set_xlabel("Mean paired-process latency reduction (%)")
    axes[1].set_title("B. Original regret estimand + bootstrap 95% intervals")
    fig.suptitle("RQ1 positives | Finite candidate-space evidence, not engine-default speedups", fontsize=14)
    savefig(fig, out, "RQ1_02_all_supported_examples", manifest, ["RQ1_ALL_MEASUREMENTS.csv", "RQ1_ALL_OBJECTS.csv"],
            "列出全部7个支持对象而非挑选最大值；完整路径实测时间与3个独立held-out进程散点。区间针对平均regret而非speedup。")
    example = next(f for f in positives if f["family"] == "attention_bmm_oproj")
    rr = [r for r in raw if r["case_id"] == example["case_id"] and r["consumer_mode"] == "torch" and r["status"] == "success" and r["process"] in [2, 3, 4]]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.4), layout="constrained")
    for ax, stage, title in zip(axes, ["producer", "consumer", "edge"], ["A. Producer only", "B. Consumer diagnostic", "C. Complete edge (not A+B)"]):
        strategies = sorted({r["strategy"] for r in rr if r["stage"] == stage})
        values = [np.array([r["measured_p50_ms"]*1000 for r in rr if r["stage"] == stage and r["strategy"] == strategy]) for strategy in strategies]
        ax.bar(np.arange(len(values)), [S.median(v) for v in values], color=[COLORS[k % len(COLORS)] for k in range(len(values))])
        for i, v in enumerate(values):ax.scatter(np.repeat(i, len(v)), v, color="black", s=15)
        ax.set_xticks(np.arange(len(values)), [s.replace("/", "\n") for s in strategies], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Measured time (microseconds)"); ax.set_title(title)
    fig.suptitle("RQ1 mechanism | AV BMM -> O projection | B=1, q=128, H=12, D=128, hidden=1536", fontsize=12)
    savefig(fig, out, "RQ1_03_actual_rank_inversion", manifest, ["RQ1_ALL_MEASUREMENTS.csv"],
            f"真实case {example['case_id']}，head-major producer更快但token-major完整路径更快。consumer诊断不用于相加还原edge。")


def plots_rq2(points, graphs, cases, out, manifest):
    protocols = ["GROWTH-NEARSET", "QKV-DUAL", "REMOTE-KV-ANCHOR"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), layout="constrained")
    for ax, protocol in zip(axes, protocols):
        left = np.zeros(len(FRAMES))
        category_counts = []
        for category, label, color in [(0, "Same choice (A/A)", "#A0A0A0"), (1, "Different, not established", "#E69F00"), (2, "Exploratory positive", "#009E73")]:
            v = np.array([sum(p["framework"] == fw and p["protocol"] == protocol and
                              (0 if p["same_choice"] else 2 if p["positive"] else 1) == category for p in points) for fw in FRAMES])
            ax.barh(FRAMES, v, left=left, color=color, label=label)
            category_counts.append(v)
            for i, n in enumerate(v):
                if n >= 80:ax.text(left[i]+n/2, i, str(n), va="center", ha="center", fontsize=8)
            left += v
        for i in range(len(FRAMES)):
            ax.text(left[i]*1.025, i, f"{category_counts[1][i]} / {category_counts[2][i]}", va="center", ha="left", fontsize=8)
        ax.set_xlim(0, max(left)*1.25)
        ax.text(.97, 1.01, "Other / +", transform=ax.transAxes, fontsize=8, ha="right")
        ax.set_title(protocol); ax.set_xlabel("Paired comparisons per adapter")
        ax.invert_yaxis()
    axes[-1].legend(loc="upper center", bbox_to_anchor=(.4, -.16), fontsize=8)
    fig.suptitle("RQ2 | Execution coverage is not hypothesis confirmation", fontsize=16)
    savefig(fig, out, "RQ2_01_three_protocol_outcomes", manifest, ["RQ2_ALL_COMPARISONS.csv", "RQ2_SUMMARY.csv"],
            "扩图每环境918对、QKV双表示1020对、KV anchor 666对；正例须不同方案且中位数比的paired CI下界>1.03，属同进程探索性结果。")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    for ax, protocol in zip(axes, protocols):
        for fw, color in zip(FRAMES, COLORS):
            vals = sorted(p["ratio"] for p in points if p["framework"] == fw and p["protocol"] == protocol)
            ax.step(vals, np.arange(1, len(vals)+1)/len(vals), where="post", color=color, label=fw)
        ax.axvline(1, color="black", linestyle="--", lw=1); ax.set_xscale("log")
        ax.set_title(protocol); ax.set_xlabel("Restricted / free full-graph time"); ax.grid(alpha=.18)
    axes[0].set_ylabel("Cumulative fraction of ALL comparisons"); axes[-1].legend(fontsize=8)
    fig.suptitle("RQ2 | Raw held-out ratios, including no gains and losses", fontsize=15)
    savefig(fig, out, "RQ2_02_full_speedup_distributions", manifest, ["RQ2_ALL_COMPARISONS.csv"],
            "包含全部5508+6120+3996对；不截掉慢于baseline的结果，不把同方案A/A的时间波动当布局收益。")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), layout="constrained")
    stages = range(10)
    for field, label, color in [("semantic_node_count", "Semantic operator nodes", COLORS[0]), ("semantic_operator_kind_count", "Distinct operator kinds", COLORS[2])]:
        v = [next(c[field] for c in cases.values() if c["stage"] == stage) for stage in stages]
        axes[0].plot(list(stages), v, "o-", label=label, color=color)
    axes[0].set_yscale("log"); axes[0].set_xticks(list(stages), [f"s{x}" for x in stages]); axes[0].set_ylabel("Count (log scale)")
    axes[0].set_title("A. Actual nested forward graph growth"); axes[0].legend(fontsize=9)
    for fw, color in zip(FRAMES, COLORS):
        medians = []; lows = []; highs = []
        for stage in range(1, 10):
            pp = [p for p in points if p["framework"] == fw and p["protocol"] == "GROWTH-NEARSET" and p["case"]["stage"] == stage]
            vals = np.array([100*(p["ratio"]-1) for p in pp])
            medians.append(np.median(vals)); lows.append(np.percentile(vals, 10)); highs.append(np.percentile(vals, 90))
        axes[1].plot(range(1, 10), medians, "o-", label=fw, color=color, lw=1.2, markersize=3)
        axes[1].fill_between(range(1, 10), lows, highs, alpha=.06, color=color)
    axes[1].axhline(0, linestyle="--", color="grey"); axes[1].set_xticks(range(1, 10))
    axes[1].set_xlabel("Expanded graph stage (not a uniform complexity increment)")
    axes[1].set_ylabel("100 x (restricted / free - 1)")
    axes[1].set_title("B. Median and 10-90% case range (NOT confidence intervals)")
    axes[1].legend(ncol=2, fontsize=8)
    fig.suptitle("RQ2 | A larger graph does not imply monotonically increasing layout benefit", fontsize=14)
    savefig(fig, out, "RQ2_03_graph_complexity_and_measured_effect", manifest, ["RQ2_ALL_COMPARISONS.csv", "RQ2_GRAPH_CHOICES.csv"],
            "每阶段102个shape对象；图节点数来自真实forward合同，阴影是不同case的10–90百分位不是CI。不能从图规模直接推出收益必然增长。")
    examples = []
    for fw in FRAMES:
        pp = [p for p in points if p["protocol"] == "GROWTH-NEARSET" and p["framework"] == fw and p["positive"]]
        if pp:examples.append(max(pp, key=lambda p: p["ratio"]))
    examples += [p for p in points if p["protocol"] != "GROWTH-NEARSET" and p["positive"]]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for i, (ax, p) in enumerate(zip(axes.flat, examples)):
        for x, key, label, color in [(0, "restricted_ms", "restricted", COLORS[0]), (1, "free_ms", "free", COLORS[2])]:
            v = np.array(p[key])*1000
            ax.bar(x, S.median(v), color=color, alpha=.8)
            ax.scatter(x+np.linspace(-.08, .08, len(v)), v, color="black", s=18)
        c = p["case"]
        ax.set_xticks([0, 1], ["Restricted", "Free"]); ax.set_ylabel("Full graph time (microseconds)")
        ax.set_title(f"E{i+1}: {p['framework']} | {p['protocol']}\n{c['model_id'].split('/')[-1]} s{c['stage']}, B={c['batch']}, q={c['query_length']}\n{p['ratio']:.3f}x, same-process 95% CI [{p['ci'][0]:.3f},{p['ci'][1]:.3f}]", fontsize=9)
    for ax in list(axes.flat)[len(examples):]:ax.set_visible(False)
    fig.suptitle("RQ2 exploratory examples | Selected for illustration, not a prevalence estimate", fontsize=14)
    savefig(fig, out, "RQ2_04_examples_measured_latency", manifest, ["RQ2_ALL_COMPARISONS.csv"],
            "扩图每个有正例的环境展示最大实测比值，另展示全部双表示/anchor正例。散点为5轮同进程holdout，不是5个独立进程。")
    # A matched trajectory shows real selected tensor layouts, not just a latency ratio.
    selected = max((p for p in points if p["protocol"] == "GROWTH-NEARSET" and p["positive"]
                    and p["case"]["batch"]*p["case"]["query_length"] > 1), key=lambda p:p["ratio"])
    c = selected["case"];fw = selected["framework"]
    trajectory = sorted((x for x in cases.values() if all(x[k] == c[k] for k in
        ("model_id", "phase", "batch", "query_length", "kv_lengths"))), key=lambda x:x["stage"])
    gg = {g["case_id"]:g for g in graphs if g["framework"] == fw}
    axes_order = gg[trajectory[-1]["case_id"]]["axes"]
    matrix = np.full((len(axes_order),10),np.nan)
    for x in trajectory:
        g=gg[x["case_id"]]
        for axis,bit in zip(g["axes"],g["best_train_bits"]):matrix[axes_order.index(axis),x["stage"]]=bit
    from matplotlib.colors import ListedColormap
    cmap=ListedColormap([COLORS[0],COLORS[1]]);cmap.set_bad("#F0F0F0")
    fig, (ax,bx) = plt.subplots(1,2,figsize=(12.5,8),gridspec_kw={"width_ratios":[1.35,1]},layout="constrained")
    ax.imshow(matrix,aspect="auto",cmap=cmap,vmin=0,vmax=1,interpolation="nearest")
    ax.set_yticks(range(len(axes_order)),axes_order,fontsize=7)
    ax.set_xticks(range(10),[f"s{s}" for s in range(10)])
    ax.set_title("A. Actual training-selected tensor layouts\nBlue: row/NHD; orange: column/HND; grey: absent")
    pp=sorted((p for p in points if p["protocol"] == "GROWTH-NEARSET" and p["framework"] == fw
               and p["case_id"] in {x["case_id"] for x in trajectory}),key=lambda p:p["case"]["stage"])
    yy=np.arange(len(pp));vals=np.array([p["ratio"] for p in pp]);lo=np.array([p["ci"][0] for p in pp]);hi=np.array([p["ci"][1] for p in pp])
    # Plot intervals as lines, rather than assuming a percentile CI contains its point estimate.
    for i,p in enumerate(pp):bx.plot([lo[i],hi[i]],[i,i],color="#999999",lw=2)
    bx.scatter(vals,yy,c=[COLORS[2] if p["positive"] else "#555555" for p in pp],zorder=3)
    bx.set_yticks(yy,[f"s{p['case']['stage']-1} -> s{p['case']['stage']}\n{'same choice' if p['same_choice'] else 'different choice'}" for p in pp],fontsize=9)
    bx.invert_yaxis();bx.axvline(1,color="grey",linestyle="--");bx.set_xlabel("Restricted / free held-out time")
    bx.set_title("B. Matched expansion comparisons\nGreen: exploratory positive; CI: same-process repeats")
    fig.suptitle(f"RQ2 matched trajectory | {fw} | {c['model_id'].split('/')[-1]}, B={c['batch']}, q={c['query_length']}\nTraining winner maps are best-measured, NOT proof of unique or globally optimal layouts",fontsize=12)
    savefig(fig,out,"RQ2_05_matched_tensor_layout_choices",manifest,["RQ2_ALL_COMPARISONS.csv","RQ2_GRAPH_CHOICES.csv"],
            f"同一model/请求shape从s0扩至s9：{fw}/{c['case_id']}。左图各真实tensor训练赢家映射，右图配对完整图实测；后期候选有限，不把赢家翻转本身当稳定因果证明。")
    return examples


def reports(out, findings, raw, r1, points, graphs, cases, r2, examples):
    def md(name, lines):
        with (out / name).open("x") as f:f.write("\n".join(lines)+"\n")
    fpositive = [f for f in findings if f["status"] == STATUSES[0]]
    r1lines = ["# RQ1：实验方法、真实数据和汇报结论", "",
        "本文RQ1沿用 **v13 L-RQ1：跨边界layout排名反转**，不是后来 `layout_problem_trial_RQ1_first_20261003.md` 中重新编号的RQ1。所有数值只取卡1 NVIDIA A10 的已完成实验；未使用卡0smoke计时、公式估计耗时或新GPU测试。", "",
        "## 1. 可以如何汇报", "",
        f"在{r1['unique_executable_case_ids']}个去重可执行合同、14种算子边界和3个consumer环境组成的{r1['case_consumer_objects']}个对象中，7个达到原代码支持判据，说明**producer单独最快的layout不一定使完整producer→repair→consumer最快**。这些正例的held-out加速比为{min(f['median_held_out_speedup'] for f in fpositive):.3f}–{max(f['median_held_out_speedup'] for f in fpositive):.3f}倍。", "",
        "不能汇报为‘所有框架的默认layout都不优’：这些是共用Triton producer的controlled/hybrid边界实验，不是原引擎layout policy替换的端到端吞吐量测试。7个是未做多重比较校正的原判据支持对象，只有3个held-out独立进程，需保留探索性限定。", "",
        "## 2. 实际跑了哪些实验", "",
        "对象=一个实际可执行的结构/shape/consumer合同；同geometry的多个模型来源去重，不把名称数当独立实验。模型架构维度来自固定revision的真实config，权重/激活是固定seed随机FP16。B/q/KV来自受控工作负载，不是生产trace；模型名含NVFP4也不表示本实验测试了NVFP4。", "",
        "|子图family（实际执行，不是模型整段forward）|去重case|含consumer环境的对象数|改变/比较什么|", "|---|---:|---:|---|"]
    descriptions = {
        "attention_bmm_oproj": "AV BMM→token/head flatten→O projection；直接写head/token-major",
        "projection_sdpa": "QKV projection→真实softmax attention；packed-row/分别稠密QKV",
        "projection_gdn": "QKV projection→原生FLA chunk gated-delta-rule；同时检查最终state",
        "sparse_gather_sdpa": "selected-token gather→compact SDPA；row/column/padded输出；不是专用DeepSeek MSA/QSA",
        "projection_swiglu_down": "gate-up GEMM→SiLU×up→down GEMM；row/column/padded及合法materialize",
        "moe_dispatch_expert": "固定routing本地dispatch→grouped expert GEMM；route/expert顺序；无DeepEP网络",
        "moe_gemm_combine": "expert GEMM2→weighted combine；head-major/route-interleaved",
        "state_prefill_decode": "KᵀV→线性state更新/读；head-major/state-transposed；不是KDA/GDN完整模型",
        "gemm_bias_gemm": "GEMM→bias→GEMM；输出row/column/padded",
        "reduce_norm_gemm": "归一化producer→GEMM；输出row/column/padded",
        "host_weight_gemm": "pinned-host权重copy→GPU GEMM；placement扩展，不是量化kernel",
        "host_kv_sdpa": "pinned-host selected K copy→真实attention；placement扩展",
        "kv_writer_flashinfer": "append-only paged KV writer→原生FlashInfer fa2 prefill/decode；paged NHD/HND",
        "kv_writer_swa_flashinfer": "同上，使用模型sliding_window；page=1时物理layout可能退化"}
    for family, n in sorted(r1["unique_case_by_family"].items()):r1lines.append(f"|{family}|{n}|{r1['objects_by_family'][family]}|{descriptions[family]}|")
    r1lines += ["", f"来源模型去重数：{r1['model_provenance_count']}。已执行B={r1['batch_values']}；q={r1['query_values']}；KV={r1['kv_values']}。它们不表示14种原模型的完整forward都真实复现；见逐对象CSV里的source_url、shape和consumer_kernel。", "",
        "这些是来源workload字段的取值集合，不是每种算子都使用所有字段。逐对象shape列按实际family执行参数列出：MLP只使用tokens=B×q、hidden和I；GDN/state使用linear head；sparse使用index head和selected S；projection_sdpa没有历史KV，只attention于q个新token。完整可执行合同另存CSV，不能把无效KV字段当shape多样性。", "",
        "|consumer环境|真实producer/consumer来源|支持|未发现冲突|不确定|单一有效layout|", "|---|---|---:|---:|---:|---:|"]
    for m in ["torch", "vllm", "sglang"]:
        cc=r1["classification_by_consumer"][m]
        r1lines.append(f"|{m}|{'共用Triton producer（host类为copy）；torch/cuBLAS/SDPA消费者' if m=='torch' else '共用Triton producer；native SiLU/FLA；KV attention共用FlashInfer，不是整引擎'}|"+"|".join(str(cc.get(k,0)) for k in STATUSES)+"|")
    r1lines += ["", "RQ1这组没有CUTLASS/TVM/Hexcute独立adapter。vLLM与SGLang两种环境调用相同FlashInfer算法时不能算两份独立算法证据。全部173个来源PR/issue的逐base/head原生replay=0；来源登记/机制类比不等于执行成功。", "",
        "## 3. 怎么测、怎么算", "",
        "1. 固定同一case的数学、seed、shape、路由和算法，producer直接计算并写不同实际地址映射；不是仅改tensor名称。记录shape/stride，去除相同storage映射的别名（如一行matrix）。\n2. 枚举identity（无修复）与materialize（真实contiguous/重排）合法边界；native stride合同不合法的860条单独拒绝，不能算失败或负例。\n3. 分别实测P-only、C-only、R-only、完整edge；完整edge由一对CUDA event包住P→R→C，不能用各median求和。\n4. 每条路径先检查producer逻辑值及最终consumer输出，再warmup 8次、计时30次；任务顺序随seed/process打乱。5个不同worker进程串行运行，前2进程训练选择，后3进程只评估。\n5. 按训练P-only时间选layout Lp；在Lp对应合法edge中选最佳repair作为local baseline。另按训练完整edge时间选联合方案Le。baseline不是框架原始默认policy。", "",
        "worker字段 `p50_ms` 实际取sorted(samples_ms)[len//2]：30次时取第16个顺序统计量，不是第15/16的平均。本汇报忠实保留历史统计定义，没有静默重新定义p50。时间是毫秒；图转微秒=ms×1000。CUDA event区间可能包含Python dispatch导致的device空隙，不等于纯kernel指令耗时。JIT/参考计算在计时外；materialize的真实执行在edge内，动态allocation/host控制开销没有单独拆分。", "",
        "对每个held-out进程i：A_i=local baseline完整edge实测p50；B_i=joint选择完整edge实测p50。**speedup=median_i(A_i/B_i)**（不是median(A)/median(B)）；regret_i=(A_i−B_i)/A_i，报告mean(regret_i)。CI为这3个配对进程regret均值的4000次bootstrap 95%区间，seed=17。", "",
        "支持判据：Lp≠Le的producer layout；producer与edge winner在5进程中均≥80%一致；regret CI下界>0；平均regret>max(3%,3×所选路径的最大跨进程CV)。CV=标准差/均值，是噪声估计。winner稳定性gate用了含held-out在内的5进程，因此应理解为原代码探索性筛选，不能声称完全独立于测试数据的最终确认。", "",
        "正确性：FP16输入/主要输出，FP32累加/reference；GDN state/gate允许FP32。rtol=0.03，atol=max(1e-5,4×2^-10×reference RMS)，且全部有限、normalized RMSE≤0.01。RMS为平方均值的平方根；normalized RMSE=误差RMS/reference RMS。这不是bitwise等价或模型任务精度评估。", "",
        "## 4. 全部7个支持对象：原始时间与选择", "",
        "|编号|consumer/结构|真实架构来源与受控shape|local→joint|分子3进程ms|分母3进程ms|speedup|平均延迟下降|", "|---|---|---|---|---|---|---:|---:|"]
    for i, f in enumerate(fpositive):
        r1lines.append(f"|P{i+1}|{f['consumer_mode']}/{f['family']}|{f['case']['model_id']}；{shape(f['case'])}|{f['local_layout_best_legal_edge']}→{f['complete_edge_selected_strategy']}|{[round(x,6) for x in f['measured_local_edge_ms']]}|{[round(x,6) for x in f['measured_oracle_edge_ms']]}|{f['median_held_out_speedup']:.5f}|{100*f['mean_held_out_regret_fraction']:.2f}%|")
    r1lines += ["", "以P1为例：训练P-only选择head-major，但完整AV→O projection选择token-major；直接写目标顺序可能省掉consumer flatten中的contiguous。图3实测P/C/edge的排名，不用P+C估算整体。cuBLAS可能随stride改变tactic（内部实现），所以不把效果全部归因于memory transaction；未采集bank conflict/带宽/指令级因果证据。", "",
        "## 5. 各假设支持到哪里", "",
        "|v13假设|本轮实际证据|不能扩大的结论|", "|---|---|---|",
        "|H1.1 local-best≠edge-best存在|7个原判据支持对象，至少反驳局部最优必然组合成完整路径最优|不支持‘所有/多数框架默认策略均失败’|",
        "|H1.2 reuse/fanout越多regret越大|有线性state重复读对象，但没有跨相同对象完整reuse/fanout匹配干预|未充分验证单调关系或fanout因果效应|",
        "|H1.3 stride-polymorphic consumer减少materialization|部分identity/materialize真实路径已计时，部分native路径按合同拒绝identity|不同consumer算法/框架/tactic未完全固定，不能视为广泛因果证明|",
        "|H1.4 consumer-native emission优于natural+convert|AV、MoE dispatch等支持对象可显示不同producer表示的完整路径收益|不是所有正例都纯粹省copy；要结合每case路径与native合同分析|",
        "|H1.NEG layout-insensitive低收益区域|396个选择无冲突、130个映射退化，并保留321个不确定|这轮FP16不能证明低精度FP8/FP4原机制|", "",
        "## 6. 可信性复核与文件", "",
        f"本次重新从raw核算全部{len(findings)}个分类、全部speedup/regret和36,460条记录中的p50定义，核对4,270个case×consumer×process。100个失败单元按完整cell替换；未把旧失败和retry算两份独立样本。最大已观察normalized RMSE={r1['coverage']['max_observed_normalized_rmse']:.8f}。旧结果不覆盖。", "",
        "RQ1原full没有保存完整冻结源码文件，只保存源码SHA；retry源码SHA与原full有变化（workspace plan修复）。本报告记录两者，不声称能从当前源码完全重建当时每一行实现。计时raw和合并来源可直接复核。", "",
        "核对方法代码：[consumer与producer路径定义]("+str(ROOT/"rq1_diverse_edge_bench.py")+":131)、[完整edge计时]("+str(ROOT/"rq1_diverse_edge_bench.py")+":102)、[训练/留出选择及原支持判据]("+str(ROOT/"analyze_rq1_diversity.py")+":19)。RQ1 bench链接是当前修复源码，不冒称它与原full源码逐字一致。", "",
        "图：`figures/RQ1_01_coverage_and_distribution.pdf/.svg`、`RQ1_02_all_supported_examples.pdf/.svg`、`RQ1_03_actual_rank_inversion.pdf/.svg`。对应明细：`data/RQ1_ALL_MEASUREMENTS.csv`和`data/RQ1_ALL_OBJECTS.csv`。术语和各图阅读说明见 `TERMS_AND_FIGURE_READING_CN.md`。", "",
        f"原始合并结果：`{r1['run']}`。原full/retry关系见 `AUDIT.json` 与 `PROVENANCE.json`。"]
    md("RQ1_METHODS_RESULTS_AND_CLAIMS_CN.md", r1lines)
    r2lines = ["# RQ2：实际做了哪些实验、真实数据和能说明什么", "",
        "本文沿用 **v13 L-RQ2：optimal layout-domain granularity**。扩图代码曾把三组协议局部标成H2.1/H2.2/H2.3，这不等于v13的同编号科学假设；本报告只用协议名称，避免偷换问题。", "",
        "## 1. 汇报结论", "",
        "六个adapter均完成了1020个真实Qwen3几何的嵌套图；每环境918个扩图比较、1020个QKV双表示比较、666个remote-KV-anchor比较（另外354图语义不适用）。结果支持**某些受控shape/图/后端条件下需要重新选择**，但不支持‘图越大，layout收益必然越大’或‘多数场景必须split域’。", "",
        "完整运行≠全部v13 RQ2假设已证实：该实验主干只有dense Qwen3，domain只改变第一层packed-QKV一个值，remote anchor是运行时条件对照，不是compiler pass传播实验。", "",
        "## 2. 测试对象及逐步扩大结构", "",
        "来源为Qwen3-0.6B/1.7B/4B/8B/14B/32B的固定revision config及已安装vLLM/SGLang forward源码（文件和方法SHA在manifest）。保留真实hidden/head/intermediate，没有按GPU显存缩小matrix。随机FP16权重/激活，不是预训练模型任务精度或完整模型服务吞吐。", "",
        "每模型17组受控shape：12组uniform（decode B=1/4/16、q=1、KV=512/4096；prefill B=1/4、q=128/1024、KV=q；extend B=1/4、q=8、KV=4096），再加5组mixed-request KV长度。共6×17=102组，每组10个阶段，共1020图。每组9个相邻扩图对，共918对。B/q/KV是受控请求轴，不是生产分布。", "",
        "|stage|真实forward子图边界|block数|语义operator nodes|operator种类数|", "|---|---|---:|---:|---:|"]
    for stage in range(10):
        c=next(c for c in cases.values() if c["stage"]==stage)
        r2lines.append(f"|s{stage}|{c['stage_label']}|{c['decoder_blocks']}|{c['semantic_node_count']}|{c['semantic_operator_kind_count']}|")
    r2lines += ["", "一个完整block：input RMSNorm→QKV projection→Q/K norm及split/view→RoPE→KV append→真实GQA attention→O projection→residual+RMSNorm→gate/up projection→SiLU×up→down projection，并保留跨block residual branch。不是把同一个独立kernel复制N遍当复杂图。", "",
        "## 3. layout选择空间与框架真实性", "",
        "每个block的四个GEMM输出分别在row-major/column-major中选；KV在NHD/HND中选（vLLM/SGLang走实际paged native consumer合同，并为HND执行收费repair）。row-major二维[m,n] stride=[n,1]，column-major stride=[1,m]。NHD为token→head→dim排列，HND为head→token→dim排列；D在内层。B×q=1的matrix layout物理等价，因此去重不算两种。bit0/1分别对应row/column或NHD/HND；graph CSV保存每个axis的选择。", "",
        "layout轴≤5时穷举2^k；>5时用预算48的有限候选+每个旧near plan合法扩展（保留集合可能超过预算）。每adapter738图穷举、282图bounded；8block非退化空间最多40个二元轴，不能把48个候选冒称2^40全空间最优。不同adapter/阶段因near-set不同可能有不同候选集合，跨框架绝对时间不能直接视为公平框架排名。", "",
        "|adapter|实际原生部分|共用/手写部分|", "|---|---|---|"]
    for fw in FRAMES:
        o=r2["metadata"][fw]["primitive_origins"]
        r2lines.append(f"|{fw}|GEMM={o['gemm']}；RMSNorm={o['rmsnorm']}|attention={o['attention']}；KV={o['kv_write']}|")
    r2lines += ["", "layout controller是本研究显式枚举/约束选择，不是Triton compiler、TVM全图优化器或serving engine的原始默认policy。TVM只原生norm，CUTLASS只原生GEMM，其余来源在raw `primitive_origins`。vLLM/SGLang有原生norm/SiLU/RoPE/KV writer，但GEMM共用cuBLAS，attention共用FlashInfer。cuBLAS tactic可能随stride变化；固定tile的Triton/CUTLASS与它们应分别解释。", "",
        "## 4. 三组实际协议及测量方法", "",
        "### GROWTH-NEARSET：旧图近优集合扩图后是否足够", "",
        "在G_small训练测每个合法layout候选，取时间≤1.03×训练最优的集合Sε（ε=3%）。扩成同一forward的G_large后，在同一个已测候选集合里比较：restricted/frozen=旧layout轴必须投影落在Sε内、新轴自由选择；free=旧新轴都可重选。两者均只按G_large训练时间选。不是把旧图时间与新图时间直接相除。", "",
        "每候选warmup3次、完整图计时10次；按训练median选择。之后在同一worker内做5轮AB/BA配对holdout，每边warmup1次、计时10次，先算该轮median。**分子=median(5个restricted轮median)，分母=median(5个free轮median)，speedup=分子/分母**。ER_set=speedup−1，是额外受限耗时相对free的比例，不是RQ1的延迟下降率；若要restricted为baseline的延迟下降，另算1−1/speedup。", "",
        "near-set survival=扩图near-set在旧轴投影与旧near-set的交集大小/旧集合大小；minimum_old_axis_changes是两集合最小Hamming差（不同bit个数）。它们是候选集合计算量，不是编译器传播距离；训练时间噪声也能改变near-set。", "",
        "### QKV-DUAL：第一层packed-QKV是否需要第二份表示", "",
        "固定其它layout为原growth的best-measured，枚举第一层QKV的producer输出row/column与共同consumer输入row/column的2×2。相同表示共一buffer（K=1）；不同表示真的写一个guarded临时buffer并copy_至consumer buffer（K=2）。完整forward实际收费copy。restricted=训练最优K=1；free=训练最优K=1/2。5轮独立于训练的AB/BA计时协议同上。", "",
        "Kε=min{训练近优候选的显式QKV输出表示数}仅为此值的条件统计，不是整图domain数、register/shared层domain数或global partition oracle。训练Kε=2不等于held-out真的加速；一个共享consumer目的表示也不等于为多个异质consumer分别分域，故本测试未充分扫描consumer heterogeneity。buffer创建/plan在测速外，copy执行在内；allocated-byte字段是公式，不是实测HBM流量或allocator压力。", "",
        "### REMOTE-KV-ANCHOR：远端KV选择是否改变上游选择", "",
        "固定数学、shape、其它layout；在原forward DAG存在有向依赖的两个实际轴上测2×2：upstream=L0 QKV（退化则L0 KV），anchor=最后block KV。先在NHD anchor下找上游3% near choices，再改HND anchor比较保留旧上游集合的restricted与free上游。真正比较的两条holdout路径都在HND anchor，不把NHD/HND算法时间误作reoptimization收益。", "",
        "DAG最短路径以实际operator node计算，记录semantic_operator_hops；它是语义路径距离，不是SSA/layout-pass传播。没有KV、物理双轴退化或无有向依赖才是354个N/A。没有做compiler pass on/off/control-flow ablation。", "",
        "### 共同计时、计算正确性和统计范围", "",
        "CUDA event包完整pipeline，实际producer/consumer/contiguous/KV repair在内，JIT、FlashInfer plan、buffer准备、reference与correctness在外。该时间可能包含Python launch间隙，不是纯kernel或host端wall latency。5轮holdout是同进程重复，不是RQ1的5个独立进程。", "",
        "所有候选先通过独立FP32 per-op reference（按语义FP16边界舍入）；扩图selected路径计时后再检查；conditional每次measure前后检查完整计算、copy、KV未写区域和allocation canary（两侧哨兵防越界）。rtol=0.03，局部atol=max(1e-5,4×2^-10×reference RMS)，整图frontier阈值随sqrt(传播操作数)放宽，但normalized RMSE仍≤1%；并要求每个局部算子单独通过。计时CSV只有正确合法路径，不把失败填0。", "",
        "原growth worker的point estimate是ratio-of-medians，但旧CI针对median-of-paired-ratios；这两者不同。本报告使用已保存strict audit并从raw复核ratio，所有正例CI重新计算同一配对索引bootstrap（2000次，seed=1234）。探索性正例条件：不同方案且ratio-of-medians CI下界>1.03。没有独立进程、A/A显著性校准、FDR等多重比较控制，所以不能叫已确认科研普遍规律。", "",
        "## 5. 完整结果，不能只报最好case", "",
        "|adapter|正确图/候选|扩图探索正例/918|QKV双表示探索正例/1020|anchor探索正例/666|训练QKV Kε=2|", "|---|---|---:|---:|---:|---:|"]
    summaries={(s["framework"],s["protocol"]):s for s in r2["summary"]}
    for fw in FRAMES:
        cc=r2["growth_counts"][fw]
        r2lines.append(f"|{fw}|{cc['graphs']}/{cc['correct_candidates']}|{summaries[fw,'GROWTH-NEARSET']['exploratory_positive']}|{summaries[fw,'QKV-DUAL']['exploratory_positive']}|{summaries[fw,'REMOTE-KV-ANCHOR']['exploratory_positive']}|{summaries[fw,'QKV-DUAL']['conditional_K2_training_only']}|")
    r2lines += ["", "|adapter/协议|相同选择A/A|不同选择数|全部比较中位speedup|不同选择子集中位speedup|", "|---|---:|---:|---:|---:|"]
    for s in r2["summary"]:
        d=s["median_ratio_different"]
        r2lines.append(f"|{s['framework']}/{s['protocol']}|{s['same_choice']}|{s['different_choice']}|{s['median_ratio_all']:.5f}|{d:.5f}|" if d is not None else f"|{s['framework']}/{s['protocol']}|{s['same_choice']}|{s['different_choice']}|{s['median_ratio_all']:.5f}|无不同选择|")
    r2lines += ["", "相同choice在代码中通常被创建为两个独立plan/buffer并分别计时，A/A的memory地址/顺序也可能不同。它们用于展示同方案波动，不是严格同buffer噪声oracle；即使ratio>1也不能算layout improvement。", "",
        "特别核对到3个same-choice比较的CI下界也>1.03（CUTLASS growth、Triton QKV-DUAL、CUTLASS remote-anchor各1个），但全部被different-choice gate排除。它们表明buffer/顺序/运行状态足以产生3%以上表观效应，因此少量different-choice正例也要独立进程、严格同buffer A/A、随机交错顺序和多重比较校准，不能仅以单次CI宣称稳健因果规律。对象见VALIDATION.json及全量CSV。", "",
        "结果解读：扩图有20/5508个探索性正例，主要出现在SGLang/vLLM hybrid adapter，绝大多数方案相同或无法建立3%收益。QKV-DUAL有33/6120个训练Kε=2，独立于训练的holdout比较只出现1个探索性正例；这两类对象并非必然一一对应，训练Kε标签不能代替收益验证。REMOTE-KV-ANCHOR只有1/3996个探索性正例，不足以断言远距离layout传播普遍显著。模型个数6但同一家族，邻近stage/同shape不独立，不能把一万多个比较当一万多个独立统计样本。", "",
        "**尤其重要：唯一QKV-DUAL正例E5位于s0（仅RMSNorm→QKV projection），没有执行后续Q/K norm、RoPE、KV和attention消费者。它只是producer输出+copy的局部完整图收益，可能涉及cuBLAS布局相关tactic；不能作为多consumer异质性阈值或最优域分区的正例。**", "",
        "## 6. 图4展示对象和原始时间", "",
        "扩图按每个有正例的adapter选最大ratio示例（明确事后选择，仅用于解释）；另外展示全部QKV-DUAL/anchor正例。完整分布看图1/2，不据此估计普遍收益。", "",
        "|编号|协议/adapter|模型、stage、shape|restricted 5轮ms|free 5轮ms|ratio/CI|", "|---|---|---|---|---|---|"]
    for i,p in enumerate(examples):
        r2lines.append(f"|E{i+1}|{p['protocol']}/{p['framework']}|{p['case']['model_id']} s{p['case']['stage']}；{shape(p['case'])}|{[round(x,6) for x in p['restricted_ms']]}|{[round(x,6) for x in p['free_ms']]}|{p['ratio']:.5f} / {[round(x,5) for x in p['ci']]}|")
    r2lines += ["", "每个case的方案bit/choice、原raw文件行号、完整shape在 `data/RQ2_ALL_COMPARISONS.csv`，训练graph选择在 `data/RQ2_GRAPH_CHOICES.csv`；不能依据示例绝对耗时做adapter速度排名。", "",
        "## 7. v13原RQ2五个假设，哪些没有真正完成", "",
        "|假设|已执行实验可提供什么|仍不能证明|", "|---|---|---|",
        "|H2.1 异质consumer超过阈值后split获益|QKV-DUAL对一个value的single/dual表示成本提供局部实例|多consumer偏好/频率/bytes异质性阈值，以及所有边的最优域分区|",
        "|H2.2 byte/time加权regret优于等权投票|未实际对照这两种policy|无结论，必须真实consumer trace+独立holdout policy实验|",
        "|H2.3 低异质性/reuse适合shared domain|多数QKV-DUAL对象没有建立收益，和该方向兼容|未控制heterogeneity/reuse，不能宣称因果确认|",
        "|H2.4 acceptance改变target/draft/verify partition|dense Qwen3 forward没有真实speculative接受/验证流程|未验证|",
        "|H2.5 hybrid KV/recurrent state分域存在overhead阈值|这1020图没有实际Mamba/GDN共享page-pool实验|未验证|", "",
        "因此‘RQ2的这三组实验跑完’是真，‘v13 RQ2全部假设、所有framework、所有PR都验证完’是假。来源账本207个PR/issue，全部标not_run的原生whole-graph replay不能冒充完成；Hexcute、其他硬件与多GPU也没有执行。", "",
        "## 8. 本次离线复核与保存", "",
        "本报告逐行读取六个growth和六个conditional原始JSONL，检查完整记录、唯一case、candidate计算正确性与原训练samples的median、所有holdout实测分子分母、所有探索性正例paired CI，以及来源SHA/行号。没有GPU重新测量，也未把终止的conditional旧半程重复计入。", "",
        "这些选择主要干预全局tensor地址映射及表示实例，不是穷举register/shared-memory layout。后者仍由固定kernel实现/编译器处理；没有捕获各框架原始on-chip布局选择器，因此不能由本结果推出其register/shared最优性。", "",
        "核对冻结源码：[扩图选择空间]("+str(Path(r2['growth_run'])/'source_snapshot/rq2_growth_search.py')+":13)、[扩图speedup定义]("+str(Path(r2['growth_run'])/'source_snapshot/rq2_growth_search.py')+":75)、[图的实际执行与holdout]("+str(Path(r2['growth_run'])/'source_snapshot/rq2_growth_bench.py')+":710)、[单/双QKV表示]("+str(Path(r2['conditional_run'])/'source_snapshot/rq2_domain_anchor_bench.py')+":65)、[anchor双轴]("+str(Path(r2['conditional_run'])/'source_snapshot/rq2_domain_anchor_bench.py')+":42)、[conditional配对计时]("+str(Path(r2['conditional_run'])/'source_snapshot/rq2_domain_anchor_bench.py')+":139)。实际worker SHA、manifest SHA与baseline raw SHA已由独立校验脚本核对。", "",
        f"growth：`{r2['growth_run']}`；conditional最终完成集：`{r2['conditional_run']}`。", "",
        "图：`figures/RQ2_01_three_protocol_outcomes.pdf/.svg`、`RQ2_02_full_speedup_distributions.pdf/.svg`、`RQ2_03_graph_complexity_and_measured_effect.pdf/.svg`、`RQ2_04_examples_measured_latency.pdf/.svg`、`RQ2_05_matched_tensor_layout_choices.pdf/.svg`。图5补充同一shape逐次扩图的tensor实际选择与配对收益。阴影10–90%是case百分位，非统计CI。所有输入文件SHA见 `PROVENANCE.json`，术语见 `TERMS_AND_FIGURE_READING_CN.md`。"]
    md("RQ2_METHODS_RESULTS_AND_CLAIMS_CN.md", r2lines)
    md("README_CN.md", ["# RQ1 / RQ2 汇报材料（真实已完成数据）", "",
        "主要材料：", "", "- [RQ1实验、方法、结论](RQ1_METHODS_RESULTS_AND_CLAIMS_CN.md)",
        "- [RQ2实验、方法、结论](RQ2_METHODS_RESULTS_AND_CLAIMS_CN.md)",
        "- [术语、公式及图表阅读说明](TERMS_AND_FIGURE_READING_CN.md)",
        "- `figures/`：3张RQ1、5张RQ2图，每张PDF/SVG/PNG。英文图标签便于移植；中文解读在文档。",
        "- `data/`：真实数据明细、公式输入、case和原始文件行号。",
        "- `AUDIT.json`、`PROVENANCE.json`、`FIGURE_MANIFEST.json`：核对结果、SHA和图数据映射。", "",
        "离线重建（不占GPU，不覆盖）：", "", "```bash", "cd /home/liangyilei/ladder_home",
        "/usr/bin/python3 staged/baseline_framework/layout_research/build_rq12_presentation.py", "```", "",
        "统计口径：RQ1 speedup是3个held-out独立进程配对比值的中位数；RQ2是同进程5轮配对holdout时间中位数之比。框架标签均是声明过原生/共用组件的adapter，不是全引擎默认layout优化器对比。", "",
        "旧结果、源代码和运行中队列全部保持不变。新增脚本只负责分析/绘图，不修改实验代码。"])
    md("TERMS_AND_FIGURE_READING_CN.md", ["# 术语、公式与图表阅读说明", "",
        "## 术语", "",
        "|名称|这里的定义|", "|---|---|",
        "|B / batch|一次子图调用中的request数量；不等于CUDA线程数|",
        "|q / query_length|每request本次处理的新token数量；q=1通常为decode，多token可为prefill/extend|",
        "|KV / kv_length / kv_lengths|attention可读的key/value历史长度；列表表示不同request不同长度，不等于新写入量|",
        "|Hq / Hkv / D / hidden|query头数、key/value头数、每head维度、模型hidden维度；GQA是Hq大于Hkv并共享KV头的attention|",
        "|I / tokens / routes / topk|实际中间MLP宽度、B×q token数、token×topk路由条数、每token选中expert数；MoE宽度优先使用moe_intermediate_size|",
        "|Hk_linear / Hv_linear / Dk / Dv|线性attention的key/value头数与对应维度；不是模型普通attention的Hq/Hkv/D|",
        "|source_KV / selected_S / actual_D|sparse原池容量、实际选中attention token数、代码使用的index_head_dim（不是普通head_dim）|",
        "|layout / stride|张量索引到实际地址的映射；stride是相邻逻辑索引的地址元素间隔，名称不同但地址相同需去重|",
        "|P / C / R / edge|产生边界tensor的算子、读取它的后续算子、必要重排/copy、完整P→R→C路径|",
        "|identity / view / materialize|不修复、零拷贝视图、真实产生新存储的转换；view不是免费改变实际数据顺序|",
        "|QKV / AV / O projection|query/key/value打包投影输出、attention概率A乘value V、attention输出映射回hidden的线性投影|",
        "|RMSNorm / SiLU×up / SwiGLU|按均方根归一化、激活gate后乘up分支、采用这种门控的MLP结构；本文不是替换这些数学|",
        "|SDPA / FLA / GDN|scaled dot-product attention、flash-linear-attention代码来源、gated delta-rule递归状态算子；不同算法不能按名字互换|",
        "|forward / DAG / stage|模型前向算子关系、有向无环依赖图、同一个forward逐步截取/扩大后的阶段|",
        "|adapter / native / hybrid|本研究调度接口、调用框架实际算子、原生算子与共用依赖/手写控制器组合；adapter标签不是全框架默认优化器|",
        "|tactic|库内部实现算法/分块等方案；cuBLAS可能随layout切换tactic，因此不能把所有差异归因到纯地址映射|",
        "|candidate / exact / bounded / best-measured|候选方案、仅声明的小空间穷举、大空间有限搜索、已测候选中的最优；都不自动等于所有合法GPU实现的最优|",
        "|training / held-out / holdout|只用训练数据选方案；选完后新采集留出数据评价，避免在测速噪声中挑最大值。这里没有训练LLM权重|",
        "|AB/BA|同一轮先测A再B、下一轮先B再A，缓解顺序漂移；RQ2轮次不等于独立进程|",
        "|A/A / same choice|两方案layout选择相同的比较；仍可能独立buffer/计时，测得ratio波动不能算layout收益|",
        "|median / p50|中位统计量；RQ1历史p50字段是30次的第16个顺序统计量，RQ2使用两中间值平均|",
        "|CV / CI / bootstrap|标准差除以均值、置信区间、对已有配对样本有放回重采样的统计方法；不能制造独立GPU样本|",
        "|RMS / normalized RMSE|均方根、误差均方根除reference均方根；用来核对FP16数值而非模型任务准确率|",
        "|canary / guard|输出buffer两侧的哨兵值；检查是否越界改写；不是GPU显存压力测试|",
        "|ε / near-set / Kε|容忍训练最优3%额外耗时、满足容忍阈值的候选集合、其中最少显式表示数；本K仅统计一个QKV输出|",
        "|Hamming / survival|不同layout bit的数量、扩图近优集合在旧轴上的保留比例；是集合指标，非测速或编译器传播距离|",
        "|FDR / multiple comparisons|对大量同时检验的偶然阳性进行控制；本轮没有做此校正，正例按探索性解释|", "",
        "## 哪些量真的测了", "",
        "CUDA event样本、producer/consumer/repair/完整路径时间、正确性误差来自GPU实际执行。shape/stride是运行对象读取的元数据。speedup、regret、CI、near-set、Hamming、bytes都从时间/元数据公式计算，不是额外硬件测量。未采集真实HBM流量、bank conflict、register占用或power；不能用bytes公式冒充这些量。", "",
        "RQ1：s=median_i(A_i/B_i)，平均baseline延迟下降=mean_i(1−B_i/A_i)。RQ2：s=median_i(A_i)/median_i(B_i)，额外受限耗时= s−1，baseline延迟下降=1−1/s。两者取中位的顺序不同，不能把1−1/median(speedup)当RQ1代码的mean(regret)。", "",
        "## 图怎么读", "",
        "RQ1图1：分类与完整分布；图2：全部7个原判据支持对象，点是3个留出进程，右侧CI针对regret；图3：具体AV→O projection的producer与完整路径排名反转。", "",
        "RQ2图1：灰色同选择、橙色不同但未建立收益、绿色探索性正例；右侧Other/+分别为橙色/绿色数量。图2：全部比较的累计分布（横轴比值>1表示free更快）。图3：真实图节点/算子种类增长，阴影是case的10–90%范围，非CI。图4：明确事后选择的解释示例，不代表平均收益。图5：同一真实shape从s0到s9的实际训练赢家tensor layout，灰格为该轴尚未出现在图中；右侧配对收益与CI。", "",
        "所有图都有PDF和SVG矢量版本及PNG预览；数据来源和case选择规则见FIGURE_MANIFEST.json，原文件SHA见PROVENANCE.json。"])


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--rq1",type=Path,default=ROOT/"results/rq1_failed_retry_full_20261001_214949_1711030/merged")
    p.add_argument("--rq2",type=Path,default=ROOT/"results/rq2_graph_growth_full_20261002_approved")
    p.add_argument("--rq2-conditional",type=Path,default=ROOT/"results/rq2346_preserved_serial_20261003_v2/rq2/conditional_full")
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    out=a.output or ROOT/"results"/f"rq12_presentation_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/"data").mkdir();(out/"figures").mkdir()
    provenance=[];fm=[]
    print(f"[offline] output={out}; no GPU work",flush=True)
    findings,raw,r1=audit_rq1(a.rq1.resolve(),out,provenance)
    print(f"[offline] RQ1 {r1['classifications']} all rechecked",flush=True)
    points,graphs,cases,r2=audit_rq2(a.rq2.resolve(),a.rq2_conditional.resolve(),out,provenance)
    plots_rq1(findings,raw,out,fm)
    examples=plots_rq2(points,graphs,cases,out,fm)
    reports(out,findings,raw,r1,points,graphs,cases,r2,examples)
    for path in [Path(__file__),a.rq1/"preflight.json",a.rq1/"MERGE_PROVENANCE.json",a.rq1/"RQ1_DIVERSITY_FINDINGS.json",
                 a.rq2/"design_manifest.json",a.rq2/"holdout_statistics_strict/holdout_estimand_audit.json"]:
        provenance.append({"path":str(path.resolve()),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    for run in [a.rq2,a.rq2_conditional]:
        for path in sorted((run/"source_snapshot").iterdir()):
            if path.is_file():provenance.append({"path":str(path.resolve()),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    write_json(out/"AUDIT.json",{"rq1":r1,"rq2":r2,"gpu_experiments_run_this_analysis":0})
    write_json(out/"PROVENANCE.json",provenance)
    write_json(out/"FIGURE_MANIFEST.json",fm)
    write_json(out/"ILLUSTRATIVE_RQ2_EXAMPLES.json",examples)
    print(f"[offline] COMPLETE {out}",flush=True)


if __name__=="__main__":main()
