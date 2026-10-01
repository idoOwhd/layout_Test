#!/usr/bin/env python3
"""Fail-closed diversity audit for an LLM layout experiment manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


STRUCTURES = {"gqa", "sliding_attention", "sparse_attention", "mla",
              "swiglu", "moe", "mamba2", "linear_attention"}
FRAMEWORKS = ("Triton", "TVM", "CUTLASS/CuTe", "vLLM", "SGLang", "Hexcute")
RQS = ("L-RQ1", "L-RQ2", "L-RQ3", "L-RQ4", "L-RQ5",
       "L-RQ6", "L-RQ7", "L-RQ8", "L-RQ9", "L-RQ10")

# This is a decision-ownership matrix, not a claim of runtime success.  The
# post-run analyzer replaces planned-runtime with actual runtime/missing.
OWNERSHIP = {
    "Triton": {"L-RQ1", "L-RQ3", "L-RQ4", "L-RQ7", "L-RQ8", "L-RQ9"},
    "TVM": {"L-RQ1", "L-RQ2", "L-RQ3", "L-RQ4", "L-RQ7", "L-RQ8", "L-RQ9"},
    "CUTLASS/CuTe": {"L-RQ1", "L-RQ3", "L-RQ4", "L-RQ7", "L-RQ8", "L-RQ9"},
    "vLLM": {"L-RQ1", "L-RQ2", "L-RQ3", "L-RQ4", "L-RQ6", "L-RQ7",
             "L-RQ8", "L-RQ9", "L-RQ10"},
    "SGLang": {"L-RQ1", "L-RQ2", "L-RQ3", "L-RQ4", "L-RQ6", "L-RQ7",
                "L-RQ8", "L-RQ9", "L-RQ10"},
    "Hexcute": {"L-RQ1", "L-RQ3", "L-RQ4", "L-RQ7", "L-RQ9"},
}


def load(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value["cases"] if isinstance(value, dict) else value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cases = load(args.manifest)
    structures = Counter(case["structure"] for case in cases)
    phases = Counter(case["phase"] for case in cases)
    batches = Counter(int(case["shape"].get("batch", 1)) for case in cases)
    q_lengths = {int(case["shape"]["query_length"]) for case in cases
                 if case["phase"] == "prefill"}
    kv_lengths = {int(case["shape"]["kv_length"]) for case in cases
                  if case["phase"] == "decode"}
    parents = defaultdict(set)
    for case in cases:
        parents[case["structure"]].add(case.get("parent_case_id"))
    checks = {
        "all_eight_structures": set(structures) == STRUCTURES,
        "at_least_32_per_structure": all(structures[s] >= 32 for s in STRUCTURES),
        "balanced_prefill_decode": phases["prefill"] == phases["decode"] >= 128,
        "request_batches_1_2_4_8": all(batches[b] >= 64 for b in (1, 2, 4, 8)),
        "prefill_tile_boundary_15_16_17": {15, 16, 17} <= q_lengths,
        "decode_page_boundary_127_128_129": {127, 128, 129} <= kv_lengths,
        "long_prefill": max(q_lengths, default=0) >= 513,
        "long_decode": max(kv_lengths, default=0) >= 8192,
        "multiple_real_parents_per_structure": all(len(parents[s]) >= 2 for s in STRUCTURES),
        "four_parents_where_catalog_allows": all(
            len(parents[s]) >= 4 for s in STRUCTURES - {"linear_attention"}),
        "explicit_rq_counterfactual_mapping": all(case.get("rq_counterfactuals")
                                                   for case in cases),
        "distributed_not_faked": all("L-RQ5" in case.get("single_gpu_blocked_rqs", [])
                                      for case in cases),
    }
    design_matrix = []
    for framework in FRAMEWORKS:
        for rq in RQS:
            if rq == "L-RQ5":
                status = "single-GPU-blocked"
            elif framework == "Hexcute":
                status = "architecture-blocked-sm86" if rq in OWNERSHIP[framework] else "N/A-not-owned"
            elif rq in OWNERSHIP[framework]:
                status = "planned-runtime-or-native-supporting"
            else:
                status = "N/A-not-owned"
            design_matrix.append({"framework": framework, "rq": rq, "status": status})
    payload = {"schema_version": 1, "manifest": str(args.manifest.resolve()),
               "case_count": len(cases), "structures": dict(structures),
               "phases": dict(phases), "batches": dict(batches),
               "prefill_lengths": sorted(q_lengths), "decode_kv_lengths": sorted(kv_lengths),
               "real_parent_counts": {key: len(value) for key, value in parents.items()},
               "checks": checks, "all_shape_checks_pass": all(checks.values()),
               "framework_rq_design_matrix": design_matrix}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "LLM_SHAPE_DIVERSITY_AUDIT.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# LLM shape/structure 多样性审计", "",
             f"- case：{len(cases)}；structure：`{dict(structures)}`。",
             f"- phase：`{dict(phases)}`；batch：`{dict(batches)}`。",
             f"- prefill 长度：`{sorted(q_lengths)}`；decode KV 长度：`{sorted(kv_lengths)}`。",
             f"- shape 多样性门槛：**{'通过' if all(checks.values()) else '未通过'}**。", "",
             "## 门槛", "", "| 检查 | 结果 |", "|---|---|"]
    lines += [f"| {name} | {'PASS' if passed else 'FAIL'} |"
              for name, passed in checks.items()]
    lines += ["", "## 框架 × RQ 设计可适用性", "",
              "`planned` 只表示有适用实验入口；必须由运行结果升级为 runtime。N/A 不是失败，表示该框架不拥有该层决策。", "",
              "| RQ | " + " | ".join(FRAMEWORKS) + " |",
              "|---|" + "---|" * len(FRAMEWORKS)]
    lookup = {(row["rq"], row["framework"]): row["status"] for row in design_matrix}
    for rq in RQS:
        lines.append("| " + rq + " | " + " | ".join(lookup[rq, fw] for fw in FRAMEWORKS) + " |")
    lines += ["", "## 严格边界", "",
              "- L-RQ5 需要多 GPU/通信干预，卡1不能验证；所有 cell 保持 blocked。",
              "- Hexcute 公开 artifact 与 A10/sm_86 不兼容，保持 architecture-blocked。",
              "- Triton/TVM/CUTLASS 的 batch 是逻辑张量/flattened token batch；只有 vLLM/SGLang native engine/KV probe 能提供 request-aware scheduler/slot 证据。",
              "- 本清单是 FP16 shape suite；BF16/FP8/INT8/INT4 精度轴由 v10 precision matrix 单独验证，不能冒充 shape 多样性。", ""]
    (args.output_dir / "LLM_SHAPE_DIVERSITY_AUDIT_CN.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps({"case_count": len(cases), "all_shape_checks_pass": all(checks.values()),
                      "report": str(args.output_dir / 'LLM_SHAPE_DIVERSITY_AUDIT_CN.md')},
                     ensure_ascii=False))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
