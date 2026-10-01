#!/usr/bin/env python3
"""Conservatively map native framework artifacts to observation evidence.

Only paired counterfactuals with correctness and a >1% effect are marked
supported. Confounded SGLang backend/page experiments and default-only
PyTorch/Triton rows are deliberately excluded; SGLang's native NHD/HND pair is
eligible because backend, page policy, model and requests remain fixed.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path


def csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def emit(observation_id: str, framework: str, speedup: float, artifact: Path,
         experiment: str, *, status: str = "supported", limitation: str = "",
         claim_kind: str = "solution", evidence_level: str = "native_runtime_counterfactual",
         benefit_kind: str = "performance") -> dict:
    return {
        "observation_id": observation_id, "framework": framework,
        "status": status, "speedup": speedup, "artifact": str(artifact),
        "experiment": experiment, "limitation": limitation,
        "claim_kind": claim_kind, "benefit_kind": benefit_kind,
        "evidence_level": evidence_level,
    }


def cutlass(root: Path) -> list[dict]:
    path = root / "cutlass_softmax_boundary.csv"
    rows = csv_rows(path)
    # RQ3 needs a regime in which paying the transformation is actually the
    # winner.  A large difference with tiled_direct winning is only a negative
    # control and must not be relabelled as conversion-investment evidence.
    valid = [row for row in rows if row.get("correct") in {"1", "True", "true"}
             and row.get("winner") == "transform_plus_flat"]
    gains = [float(row["winner_speedup"]) for row in valid if row.get("winner_speedup")]
    if not gains:
        return []
    gain = max(gains)
    return [emit(
        "O-RQ3-CONVERSION-INVESTMENT", "CUTLASS", gain, path,
        "CUTLASS/CUDA softmax: tiled-direct versus transform-plus-flat",
        status="supported" if gain > 1.01 else "inconclusive",
        limitation="single operator boundary; not a persistent KV conversion")]


def triton(root: Path) -> list[dict]:
    path = root / "triton_kv_layout.csv"
    rows = csv_rows(path)
    grouped = defaultdict(list)
    for row in rows:
        if row.get("correct") in {"1", "True", "true"}:
            grouped[row["case_id"]].append(row)
    winners, gains = set(), []
    for case_rows in grouped.values():
        if len(case_rows) < 2:
            continue
        ordered = sorted(case_rows, key=lambda row: float(row["p50_ms"]))
        winners.add(ordered[0]["layout"])
        gains.append(float(ordered[-1]["p50_ms"]) / float(ordered[0]["p50_ms"]))
    if not gains:
        return []
    heterogeneous = len(winners) > 1
    return [emit(
        "O-RQ9-RESIDUAL-METAPOLICY", "Triton", max(gains), path,
        "Triton NHD/HND/paged layout sweep across cases",
        status="supported" if heterogeneous and max(gains) > 1.01 else "inconclusive",
        limitation="one GPU/version; portability still needs held-out hardware")]


def vllm(root: Path) -> list[dict]:
    path = root / "vllm_native_serving.jsonl"
    rows = [row for row in jsonl(path)
            if row.get("status") == "success" and row.get("comparison_scope") == "layout_only"
            and row.get("p50_ms") is not None]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["case_id"]].append(row)
    gains, winners = [], set()
    for current in grouped.values():
        if len(current) < 2:
            continue
        ordered = sorted(current, key=lambda row: float(row["p50_ms"]))
        gains.append(float(ordered[-1]["p50_ms"]) / float(ordered[0]["p50_ms"]))
        winners.add(ordered[0].get("layout", ordered[0].get("variant")))
    if not gains:
        return []
    heterogeneous = len(winners) > 1
    result = [emit(
        "O-RQ1-TWO-LEVEL-CONTROL", "vLLM", max(gains), path,
        "vLLM whole-engine KV-layout counterfactual across request shapes",
        status="supported" if heterogeneous and max(gains) > 1.01 else "inconclusive",
        limitation="whole-engine evidence; does not decompose producer/boundary/consumer"),
            emit(
        "O-RQ9-RESIDUAL-METAPOLICY", "vLLM", max(gains), path,
        "vLLM layout winner across request shapes",
        status="supported" if heterogeneous and max(gains) > 1.01 else "inconclusive",
        limitation="single GPU/model; portability still open")]
    result.append(emit(
        "O-RQ6-EPOCHAL-STATE", "vLLM", max(gains), path,
        "vLLM persistent-layout winner changes across request regimes",
        status="supported" if heterogeneous and max(gains) > 1.01 else "inconclusive",
        claim_kind="problem",
        limitation="Reproduces static-policy regret only; no online migration/hysteresis action is exposed."))
    return result


def sglang(root: Path) -> list[dict]:
    path = root / "sglang_native_serving.jsonl"
    rows = [row for row in jsonl(path)
            if row.get("status") == "success" and row.get("comparison_scope") == "layout_only"
            and row.get("layout") in {"NHD", "HND"} and row.get("p50_ms") is not None]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["case_id"]].append(row)
    gains, winners = [], set()
    for current in grouped.values():
        if {row["layout"] for row in current} != {"NHD", "HND"}:
            continue
        ordered = sorted(current, key=lambda row: float(row["p50_ms"]))
        gains.append(float(ordered[-1]["p50_ms"]) / float(ordered[0]["p50_ms"]))
        winners.add(ordered[0]["layout"])
    if not gains:
        return []
    heterogeneous = len(winners) > 1
    result = [emit(
        "O-RQ1-TWO-LEVEL-CONTROL", "SGLang", max(gains), path,
        "SGLang whole-engine NHD/HND KV-layout counterfactual across request shapes",
        status="supported" if heterogeneous and max(gains) > 1.01 else "inconclusive",
        limitation="whole-engine evidence; unsupported HND cases are retained but not timed"),
            emit(
        "O-RQ9-RESIDUAL-METAPOLICY", "SGLang", max(gains), path,
        "SGLang NHD/HND winner across request shapes",
        status="supported" if heterogeneous and max(gains) > 1.01 else "inconclusive",
        limitation="single GPU/model; portability still open")]
    result.append(emit(
        "O-RQ6-EPOCHAL-STATE", "SGLang", max(gains), path,
        "SGLang persistent NHD/HND winner changes across request regimes",
        status="supported" if heterogeneous and max(gains) > 1.01 else "inconclusive",
        claim_kind="problem",
        limitation="Reproduces static-policy regret only; no online migration/hysteresis action is exposed."))
    return result


def tvm(root: Path) -> list[dict]:
    path = root / "tvm_rq_observations.csv"
    rows = csv_rows(path)
    if not rows:
        return []
    result = []

    scans = defaultdict(list)
    for row in rows:
        if row.get("experiment") == "consumer_scan" and row.get("correct") in {"1", "True", "true"}:
            scans[row["case_id"]].append(row)
    winners, shape_gains = set(), []
    static_totals = defaultdict(float)
    oracle_total = 0.0
    for current in scans.values():
        if len(current) < 2:
            continue
        ordered = sorted(current, key=lambda row: float(row["p50_ms"]))
        winners.add(ordered[0]["strategy"])
        shape_gains.append(float(ordered[-1]["p50_ms"]) / float(ordered[0]["p50_ms"]))
        oracle_total += float(ordered[0]["p50_ms"])
        for row in current:
            static_totals[row["strategy"]] += float(row["p50_ms"])
    oracle_gain = min(static_totals.values(), default=1.0) / oracle_total if oracle_total else 1.0
    if shape_gains:
        result.append(emit(
            "O-RQ9-RESIDUAL-METAPOLICY", "TVM", oracle_gain, path,
            "TVM row/tiled layout: best global static layout versus per-shape oracle",
            status="supported" if len(winners) > 1 and oracle_gain > 1.01 else "inconclusive",
            limitation="single GPU and hand-enumerated TVM candidates; cross-GPU portability remains open"))

    reuse = defaultdict(dict)
    for row in rows:
        if row.get("experiment") == "conversion_reuse" and row.get("correct") in {"1", "True", "true"}:
            reuse[(row["case_id"], int(row["reuse"]))][row["strategy"]] = float(row["p50_ms"])
    conversion_wins, direct_wins, conversion_gains, repair_gains = [], [], [], []
    for key, values in reuse.items():
        required = {"keep_row_major", "convert_once_then_tiled", "convert_each_use_then_tiled"}
        if not required.issubset(values):
            continue
        winner = min(values, key=values.get)
        if winner == "convert_once_then_tiled":
            conversion_wins.append(key)
            conversion_gains.append(values["keep_row_major"] / values[winner])
        if winner == "keep_row_major":
            direct_wins.append(key)
        repair_gains.append(values["convert_each_use_then_tiled"] / values["convert_once_then_tiled"])
    if reuse:
        result.append(emit(
            "O-RQ3-CONVERSION-INVESTMENT", "TVM", max(conversion_gains, default=1.0), path,
            "TVM measured conversion primitive amortized over reuse 1/4/16/64",
            status="supported" if conversion_wins and direct_wins and max(conversion_gains) > 1.01 else "inconclusive",
            evidence_level="native_measured_cost_model",
            limitation="Totals compose directly timed TVM kernels; they are not one fused end-to-end launch."))
        result.append(emit(
            "O-RQ10-BOUNDED-REPAIR", "TVM", max(repair_gains, default=1.0), path,
            "TVM legal row-to-tiled repair: materialize once versus repeat at every use",
            status="supported" if repair_gains and max(repair_gains) > 1.01 else "inconclusive",
            evidence_level="native_measured_cost_model",
            limitation="Finite row-to-tiled repair family; resource-budget search remains in the controlled planner."))
        result.append(emit(
            "O-RQ7-TYPED-EDGE-CONTRACT", "TVM", 1.0, path,
            "TVM row producer to tiled-only consumer with correctness-checked conversion edge",
            status="feasibility_supported", claim_kind="solution", benefit_kind="feasibility",
            evidence_level="native_runtime_counterfactual",
            limitation="Proves a legal typed repair exists; feasibility is not a speedup claim."))

    fusion = defaultdict(dict)
    for row in rows:
        if row.get("experiment") == "fusion_order" and row.get("correct") in {"1", "True", "true"}:
            fusion[row["case_id"]][row["strategy"]] = float(row["p50_ms"])
    fusion_gains, order_ratios, fusion_winners = [], [], set()
    fusion_static = defaultdict(float)
    fusion_oracle = 0.0
    for values in fusion.values():
        required = {"fused_square_reduce", "square_then_reduce_materialized",
                    "convert_then_square_then_tiled_reduce"}
        if not required.issubset(values):
            continue
        winner = min(values, key=values.get)
        fusion_winners.add(winner)
        fusion_oracle += values[winner]
        for strategy, value in values.items():
            fusion_static[strategy] += value
        separate = min(values["square_then_reduce_materialized"],
                       values["convert_then_square_then_tiled_reduce"])
        fusion_gains.append(separate / values["fused_square_reduce"])
        a, b = values["square_then_reduce_materialized"], values["convert_then_square_then_tiled_reduce"]
        order_ratios.append(max(a, b) / min(a, b))
    if fusion_gains:
        joint_gain = min(fusion_static.values()) / fusion_oracle if fusion_oracle else 1.0
        result.append(emit(
            "O-RQ8-NONCOMMUTATIVE-SEARCH", "TVM", joint_gain, path,
            "TVM materialize/convert order versus fused square-reduction candidate",
            status="supported" if len(fusion_winners) > 1 and joint_gain > 1.01 and max(order_ratios) > 1.01 else "inconclusive",
            evidence_level="native_measured_cost_model",
            limitation="Separate pipelines compose native primitive medians; fused candidate is directly timed; speedup is the per-shape oracle over the best static strategy."))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--v10-dir", type=Path)
    parser.add_argument("--additional", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [*cutlass(args.canonical_dir), *triton(args.canonical_dir),
            *tvm(args.canonical_dir), *sglang(args.canonical_dir), *vllm(args.canonical_dir)]
    if args.v10_dir:
        # v10 may contain the only vLLM Stage-5 run.
        rows.extend(vllm(args.v10_dir))
    for path in args.additional:
        rows.extend(jsonl(path))
    # Deduplicate exact evidence rows while keeping distinct artifacts.
    unique = {(row["observation_id"], row["framework"], row["artifact"]): row for row in rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                   for row in unique.values()), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(unique)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
