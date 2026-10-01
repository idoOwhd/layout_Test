#!/usr/bin/env python3
"""Fail-closed audit of the Chinese RQ methods/results report.

This checker does not benchmark a GPU.  It verifies that the report is tied to
the canonical raw artifacts and to the current adjudicator formulas, so a
documentation edit cannot silently turn a measured-cost model into a directly
timed pipeline (or vice versa).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "results/v10_v13_640_all_frameworks_20260920_143843"
DEFAULT_AUDIT = HERE / "results/method_formula_audit_20260927_v2"
DEFAULT_REPORT = HERE / "results/ALL_RQ_EXPERIMENT_METHODS_AND_RESULTS_CN.md"


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def jsonl_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    raw = args.canonical_run / "v13_640/full_combined/raw"
    report = args.report.read_text(encoding="utf-8")
    analyzer = (HERE / "analyze_v13_rqs.py").read_text(encoding="utf-8")
    tvm_bench = (HERE / "tvm_rq_observation_bench.py").read_text(encoding="utf-8")
    cutlass_bench = (HERE.parent.parent.parent / "Ladder_LL/benchmarks/softmax_layout_boundary.cu")
    cutlass_source = cutlass_bench.read_text(encoding="utf-8")

    cuda = csv_rows(raw / "rq_cuda_all_attention.csv")
    triton = csv_rows(raw / "triton_640_kv_layout.csv")
    tvm = csv_rows(raw / "tvm_640_rq_observations.csv")
    boundary = jsonl_rows(raw / "llm_640_boundary_layout_sweep.jsonl")
    require(len(cuda) == 13_680, f"unexpected CUDA rows: {len(cuda)}")
    require(len({row['case_id'] for row in cuda}) == 240, "CUDA attention case count drifted")
    require(len(triton) == 960, f"unexpected Triton rows: {len(triton)}")
    require(len({row['case_id'] for row in triton}) == 240, "Triton case count drifted")
    require(len(tvm) == 13_440, f"unexpected TVM rows: {len(tvm)}")
    require(len({row['manifest_case_id'] for row in tvm}) == 640, "TVM manifest coverage drifted")
    require(len({row['case_id'] for row in tvm}) == 120, "TVM projected-shape count drifted")
    require(len(boundary) == 1_802, f"unexpected boundary rows: {len(boundary)}")

    payload = json.loads((args.audit_dir / "v13_hypothesis_results.json").read_text(encoding="utf-8"))
    hypotheses = {row["hypothesis"]: row for row in payload["hypotheses"]}
    require(len(hypotheses) == 49, f"expected 49 hypotheses, got {len(hypotheses)}")
    expected_status = {
        "supported": 34,
        "inconclusive": 6,
        "feasibility_supported": 3,
        "blocked_single_gpu": 5,
        "blocked_requires_multiple_hardware": 1,
    }
    require(Counter(row["status"] for row in hypotheses.values()) == expected_status,
            "hypothesis status distribution drifted")
    require(hypotheses["H2.2"]["metrics"]["max_token_proxy_regret"] == 1.0886075949367089,
            "H2.2 metric or meaning drifted")
    require(hypotheses["H4.3"]["metrics"]["dominant_winner_frequency"] == 0.3375,
            "H4.3 must remain a frequency, not a contiguous region")
    require(hypotheses["H10.2"]["metrics"]["small_page_penalty_cases"] == 136,
            "H10.2 direct page penalty count drifted")

    code_markers = [
        'cuda = read_csv(prefer(args.raw_dir, "rq_cuda_all_attention.csv", "rq_cuda_reference.csv"))',
        'equal_vote = "shared_NHD" if costs["token"]["NHD"] <= costs["token"]["HND"] else "shared_HND"',
        'dominant = Counter(triton_winners).most_common(1)[0][1] / len(triton_winners)',
        'for overhead_fraction in (0.0, .01, .03, .10, .30):',
        'and abs(number(row, "max_abs_error", math.inf)) <= 1e-3',
        'str(row.get("correctness_passed", "false")).lower() == "true"',
    ]
    # The raw read is intentionally named cuda_all after the fail-closed
    # correctness patch; accept that exact current spelling below.
    code_markers[0] = 'cuda_all = read_csv(prefer(args.raw_dir, "rq_cuda_all_attention.csv", "rq_cuda_reference.csv"))'
    for marker in code_markers:
        require(marker in analyzer, f"adjudicator formula marker missing: {marker}")
    require('"evidence_level": "native_measured_cost_model"' in tvm_bench,
            "TVM composite rows lost measured-cost-model label")
    require("std::max(tiled_ms, end_to_end_ms)" in cutlass_source and
            "std::min(tiled_ms, end_to_end_ms)" in cutlass_source,
            "CUTLASS winner_speedup formula drifted")

    report_markers = [
        "哪些数是测出来的，哪些是算出来的",
        "所有主要 speedup/regret 的分子与分母",
        "历史名为 `equal_vote`",
        "不是一个连续 geometry region",
        "registration/graph overhead was not directly measured",
    ]
    # The last two concepts are written in Chinese in the report; accept the
    # exact current wording rather than weakening this to an uninformative word.
    report_markers[-2:] = ["不能称为“最大连续区域”", "这些 overhead fraction 没有直接测量"]
    for marker in report_markers:
        require(marker in report, f"report explanation missing: {marker}")

    print(json.dumps({
        "status": "pass",
        "report": str(args.report),
        "canonical_raw": str(raw),
        "hypotheses": len(hypotheses),
        "status_counts": dict(expected_status),
        "rows": {"cuda": len(cuda), "triton": len(triton),
                 "tvm": len(tvm), "boundary": len(boundary)},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
