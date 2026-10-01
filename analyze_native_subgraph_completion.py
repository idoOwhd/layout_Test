#!/usr/bin/env python3
"""Audit vLLM/SGLang exact-case native operator coverage without overclaiming."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


STRUCTURES = ("gqa", "sliding_attention", "sparse_attention", "mla",
              "swiglu", "moe", "mamba2", "linear_attention")
FRAMEWORKS = ("vllm", "sglang")
RELATED_RQS = {
    "gqa": {"L-RQ1", "L-RQ2", "L-RQ4", "L-RQ6", "L-RQ9"},
    "sliding_attention": {"L-RQ1", "L-RQ2", "L-RQ4", "L-RQ6", "L-RQ9"},
    "sparse_attention": {"L-RQ1", "L-RQ2", "L-RQ4", "L-RQ6", "L-RQ9"},
    "mla": {"L-RQ1", "L-RQ2", "L-RQ4", "L-RQ6", "L-RQ9"},
    "swiglu": {"L-RQ1", "L-RQ2", "L-RQ4", "L-RQ9"},
    "moe": {"L-RQ1", "L-RQ2", "L-RQ4", "L-RQ9"},
    "mamba2": {"L-RQ1", "L-RQ2", "L-RQ6", "L-RQ9"},
    "linear_attention": {"L-RQ1", "L-RQ2", "L-RQ6", "L-RQ9"},
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = value["cases"] if isinstance(value, dict) else value
    expected = {(framework, case["case_id"]) for framework in FRAMEWORKS for case in cases}
    rows = []
    for framework in FRAMEWORKS:
        rows += read_jsonl(args.result_dir / "raw" / f"{framework}_subgraph_native.jsonl")
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row.get("framework"), row.get("case_id"))].append(row)

    accounting = []
    for framework, case_id in sorted(expected):
        current = by_key.get((framework, case_id), [])
        good = [row for row in current if row.get("status") == "success" and
                row.get("numerically_correct", True)]
        bad = [row for row in current if row.get("status") not in ("success", None) or
               not row.get("numerically_correct", True)]
        case = next(item for item in cases if item["case_id"] == case_id)
        if good:
            status, level = "native_operator_validated", "native_operator_slice"
        elif current:
            status, level = current[0].get("status", "failed"), "source_only_or_failed"
        else:
            status, level = "missing", "none"
        accounting.append({"framework": framework, "case_id": case_id,
                           "structure": case["structure"], "phase": case["phase"],
                           "status": status, "coverage_level": level,
                           "native_rows": len(current), "successful_rows": len(good),
                           "failed_rows": len(bad), "target_rqs": case.get("target_rqs", []),
                           "full_subgraph_end_to_end": False,
                           "reason_full_subgraph_false":
                               "case architecture checkpoint is not loaded; an operator slice cannot prove the complete graph"})

    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir / "NATIVE_SUBGRAPH_COVERAGE.json").write_text(
        json.dumps({"schema_version": 1, "expected_framework_case_pairs": len(expected),
                    "accounting": accounting}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    lines = ["# vLLM / SGLang 全子图覆盖补充审计", "",
             "## 结论", "",
             f"- manifest case：{len(cases)}；应有框架×case 组合：{len(expected)}。",
             "- `native_operator_validated` 只表示框架原生、layout-sensitive 算子切片实测；不等于该模型架构的完整子图端到端实测。",
             "- 固定 Qwen2.5 serving 不能证明 MLA、稀疏注意力、MoE、Mamba2、线性注意力等不同语义子图；本报告禁止这种外推。",
             "- 单卡不能验证分布式 RQ；相应项必须保留为 not-applicable/blocked。", "",
             "## 按框架和结构", "",
             "| 框架 | 结构 | case | 原生算子已验证 | 源码存在但运行未验证/失败 | 缺失 | 完整子图 E2E |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for framework in FRAMEWORKS:
        for structure in STRUCTURES:
            subset = [row for row in accounting if row["framework"] == framework and
                      row["structure"] == structure]
            counts = Counter(row["status"] for row in subset)
            native = counts["native_operator_validated"]
            missing = counts["missing"]
            other = len(subset) - native - missing
            lines.append(f"| {framework} | {structure} | {len(subset)} | {native} | {other} | {missing} | 0 |")

    lines += ["", "## vLLM attention layout 对照", ""]
    wins = Counter()
    pairs = 0
    for case in cases:
        current = [row for row in by_key.get(("vllm", case["case_id"]), [])
                   if row.get("status") == "success" and row.get("p50_ms") is not None]
        variants = {row.get("variant"): row for row in current}
        if "NHD" in variants and "HND" in variants:
            pairs += 1
            a, b = variants["NHD"]["p50_ms"], variants["HND"]["p50_ms"]
            wins["NHD" if a < b else "HND" if b < a else "tie"] += 1
    lines += [f"直接算子可比较 case：{pairs}；NHD 更快 {wins['NHD']}，HND 更快 {wins['HND']}，相同 {wins['tie']}。",
              "直接 4D cache writer 只执行其受支持的 NHD contract；HND 由独立的固定模型 native engine 实验验证。不能用人工 4D stride 冒充 engine-owned 5D HND cache。", "",
              "## 对 RQ 的相关运行证据范围", "",
              "| RQ | vLLM 成功 case | SGLang 成功 case | 这里能观察什么（不是成对因果验证） |",
              "|---|---:|---:|---|"]
    descriptions = {
        "L-RQ1": "框架原生算子的既定 layout 与局部代价",
        "L-RQ2": "真实 shape 改变时同一原生路径的成本变化",
        "L-RQ4": "cache materialization / fused activation / routing 边界成本",
        "L-RQ6": "KV 或 recurrent state 的原生状态布局",
        "L-RQ9": "局部中间量/状态的 shape、stride 与显存对象",
    }
    for rq in ("L-RQ1", "L-RQ2", "L-RQ4", "L-RQ6", "L-RQ9"):
        counts = {}
        for framework in FRAMEWORKS:
            keys = {(row.get("framework"), row.get("case_id")) for row in rows
                    if row.get("framework") == framework and row.get("status") == "success"
                    and row.get("numerically_correct", True)
                    and rq in row.get("related_rqs",
                                      row.get("directly_evidenced_rqs",
                                              RELATED_RQS.get(row.get("structure"), set())))}
            counts[framework] = len(keys)
        lines.append(f"| {rq} | {counts['vllm']} | {counts['sglang']} | {descriptions[rq]} |")
    lines += ["", "L-RQ3/L-RQ7/L-RQ8/L-RQ10 需要图级、搜索/泛化或跨硬件反事实，不能由本算子切片单独证明；L-RQ5 是分布式问题，卡1不适用。", "",
              "## 仍未完成、且不能由单个固定模型补齐的验证", "",
              "1. 每个 case 对应真实 checkpoint 的完整子图 E2E：需要加载相应架构/权重（或框架正式支持的 dummy-weight 模式），大模型会超过 A10 显存。",
              "2. 线性注意力 FLA runtime：当前 sm_86 首次 JIT 探针未干净结束，因此按 fail-closed 记为源码支持、运行未验证。",
              "3. 分布式 layout/RQ：单卡1不适用。",
              "4. 单候选原生算子切片只记录框架决策与局部成本；没有成对 layout counterfactual 时不能标为 RQ 的直接因果证据。", ""]
    (args.result_dir / "VLLM_SGLANG_SUBGRAPH_COMPLETION_REPORT_CN.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps({"expected_pairs": len(expected),
                      "native_validated_pairs": sum(row["status"] == "native_operator_validated"
                                                    for row in accounting),
                      "missing_pairs": sum(row["status"] == "missing" for row in accounting),
                      "report": str(args.result_dir / "VLLM_SGLANG_SUBGRAPH_COMPLETION_REPORT_CN.md")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
