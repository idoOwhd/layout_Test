#!/usr/bin/env python3
"""Summarize paired single- vs multi-request native KV-cache results."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def ratio_median(values: list[float]) -> str:
    return f"{statistics.median(values):.3f}×" if values else "N/A"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    raw = args.result_dir / "raw"
    rows = read_jsonl(raw / "vllm_kv_request_batch.jsonl")
    rows += read_jsonl(raw / "sglang_kv_request_batch.jsonl")
    value = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = value["cases"] if isinstance(value, dict) else value
    attention_ids = {case["case_id"] for case in cases if case["structure"] in
                     {"gqa", "sliding_attention", "sparse_attention", "mla"}}
    good = [row for row in rows if row.get("status") == "success" and
            row.get("numerically_correct")]
    executed_ids = {row.get("case_id") for row in rows if row.get("case_id")}
    grouped = defaultdict(list)
    for row in good:
        grouped[(row["framework"], row["request_batch"], row["slot_policy"],
                 row["phase"])].append(row)

    pair_ratios = defaultdict(list)
    by_key = {(row["framework"], row["case_id"], row["request_batch"],
               row["slot_policy"]): row for row in good}
    for row in good:
        batch = row["request_batch"]
        if batch == 1:
            continue
        base = by_key.get((row["framework"], row["case_id"], 1, row["slot_policy"]))
        if base:
            # Throughput ratio: >1 means batching improved aggregate append throughput.
            pair_ratios[(row["framework"], batch, row["slot_policy"], row["phase"])].append(
                row["new_tokens_per_ms"] / base["new_tokens_per_ms"])

    policy_ratios = defaultdict(list)
    for framework in ("vllm", "sglang"):
        for case_id in executed_ids:
            for batch in (1, 2, 4, 8):
                contiguous = by_key.get((framework, case_id, batch, "request_major_contiguous"))
                interleaved = by_key.get((framework, case_id, batch, "block_interleaved"))
                if contiguous and interleaved:
                    key = (framework, batch, contiguous["phase"])
                    policy_ratios[key].append(interleaved["p50_ms"] /
                                              contiguous["p50_ms"])

    expected_per_framework = len(executed_ids) * 4 * 2
    status_counts = Counter(row.get("status", "missing") for row in rows)
    summary = {
        "schema_version": 1,
        "source_attention_cases": len(attention_ids),
        "executed_attention_cases": len(executed_ids),
        "source_manifest_batches": sorted({case.get("shape", {}).get("batch") for case in cases}),
        "expected_rows_per_framework_for_batches_1_2_4_8_and_two_policies": expected_per_framework,
        "rows": len(rows), "status_counts": dict(status_counts),
        "successful_correct_rows": len(good),
        "paired_throughput_ratios": {
            "|".join(map(str, key)): values for key, values in pair_ratios.items()
        },
        "interleaved_over_contiguous_latency_ratios": {
            "|".join(map(str, key)): values for key, values in policy_ratios.items()
        },
    }
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir / "KV_REQUEST_BATCH_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# KV-cache 单 request / 多 request layout 验证", "",
             "## 结论边界", "",
             f"- 输入 manifest 中 attention case：{len(attention_ids)}；本次执行：{len(executed_ids)}；原始 batch：`{summary['source_manifest_batches']}`。",
             "- batch>1 行是保持模型维度和单请求长度不变的显式反事实，不冒充原 manifest 合同。",
             "- 本实验调用框架原生 KV writer，证明 cache materialization 局部边界；完整 attention read 和调度器影响由另一个 native serving 并发实验验证。",
             f"- 采集 {len(rows)} 行；数值正确且成功 {len(good)}；状态：`{dict(status_counts)}`。", "",
             "## 原生 KV writer 分层结果", "",
             "| 框架 | batch | slot policy | phase | 成功 case | p50 中位数 (ms) | 吞吐中位数 (new token/ms) | 相对 batch=1 吞吐 |",
             "|---|---:|---|---|---:|---:|---:|---:|"]
    for key in sorted(grouped):
        framework, batch, policy, phase = key
        subset = grouped[key]
        ratios = [] if batch == 1 else pair_ratios[(framework, batch, policy, phase)]
        lines.append(f"| {framework} | {batch} | {policy} | {phase} | {len(subset)} | "
                     f"{statistics.median(row['p50_ms'] for row in subset):.6f} | "
                     f"{statistics.median(row['new_tokens_per_ms'] for row in subset):.3f} | "
                     f"{ratio_median(ratios)} |")
    lines += ["", "## Slot policy 的配对判据", "",
              "`interleaved/contiguous < 1` 表示 block-interleaved 更快；只把 ≥3% 视作有意义差异。", "",
              "| 框架 | batch | phase | 配对 case | interleaved/contiguous 中位数 | interleaved 快 ≥3% | contiguous 快 ≥3% |",
              "|---|---:|---|---:|---:|---:|---:|"]
    for (framework, batch, phase), values in sorted(policy_ratios.items()):
        lines.append(f"| {framework} | {batch} | {phase} | {len(values)} | "
                     f"{statistics.median(values):.4f}× | "
                     f"{sum(value <= 1 / 1.03 for value in values)} | "
                     f"{sum(value >= 1.03 for value in values)} |")
    meaningful = sum(sum(value <= 1 / 1.03 or value >= 1.03 for value in values)
                     for values in policy_ratios.values())
    comparisons = sum(len(values) for values in policy_ratios.values())
    lines += ["", f"全体 slot-policy 配对中，达到 3% 阈值的是 {meaningful}/{comparisons}；必须结合 full attention/serving 结果判断这些少数点是否可复现，而不能把微小 kernel 抖动写成 layout 胜负。", "",
              "## RQ / observation 对应", "",
              "- L-RQ4（直接、writer-only）：比较连续 request-major 与 page/block interleaved 的局部 materialization 成本是否随 batch 改变。",
              "- L-RQ1/L-RQ2（支持）：观察固定 writer layout 跨 request batch 的局部成本；没有 producer→consumer 或 domain-split 反事实。",
              "- L-RQ6（支持）：观测持久 KV state 的 slot/page 组织成本，但没有执行迁移或 horizon sweep。",
              "- L-RQ9（支持）：记录 shape、stride、layout 与时延，但没有搜索未暴露候选。",
              "- 重要 observation：若 batch 或 slot schedule 改变相对性能/吞吐，则 batch 与调度产生的物理 slot 分布必须进入 layout cost model，不能只用 batch=1 shape 决策。", "",
              "## 不能据此声称", "",
              "- 这不是完整 attention kernel，也不是每个 real-world checkpoint 的 E2E replay。",
              "- 它不验证分布式 KV cache；单卡1无法覆盖 L-RQ5。", ""]
    (args.result_dir / "KV_REQUEST_BATCH_REPORT_CN.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "successful_correct_rows": len(good),
                      "report": str(args.result_dir / 'KV_REQUEST_BATCH_REPORT_CN.md')},
                     ensure_ascii=False))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
