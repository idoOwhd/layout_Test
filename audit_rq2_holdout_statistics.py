#!/usr/bin/env python3
"""Recompute holdout estimands from raw samples into NEW analysis files.

The original worker reports ratio-of-medians as its point estimate but a
median-of-paired-ratios bootstrap interval. Keep both estimands explicit.
Nothing here changes measured durations, selected plans, or old reports.
"""
import argparse
import json
import math
from pathlib import Path
import random
import statistics
from check_rq2_graph_growth_progress import read_rows


def paired_bootstrap(frozen,free,draws=2000,seed=1234):
    if not frozen or len(frozen)!=len(free) or any(not math.isfinite(x) or x<=0 for x in [*frozen,*free]):
        raise ValueError("positive, nonempty, equal-length paired durations required")
    rng=random.Random(seed);n=len(free);ratios=[x/y for x,y in zip(frozen,free)]
    ratio_of_medians=[];median_of_ratios=[]
    for _ in range(draws):
        indices=rng.choices(range(n),k=n)
        ratio_of_medians.append(statistics.median(frozen[i] for i in indices)/
                                statistics.median(free[i] for i in indices))
        median_of_ratios.append(statistics.median(ratios[i] for i in indices))
    def interval(values):
        values=sorted(values)
        return [values[max(0,int(draws*.025)-1)],values[min(draws-1,int(draws*.975))]]
    return {"ratio_of_medians":statistics.median(frozen)/statistics.median(free),
            "ratio_of_medians_paired_bootstrap_95_interval":interval(ratio_of_medians),
            "median_of_paired_ratios":statistics.median(ratios),
            "median_of_paired_ratios_bootstrap_95_interval":interval(median_of_ratios),
            "paired_repetitions":n,"draws":draws,"seed":seed,
            "inference_scope":"same-process repeats; neither cross-process CI nor multiplicity-adjusted proof"}


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError("refusing to overwrite statistics analysis")
    records=[]
    for path in sorted(args.run.glob("*.jsonl")):
        rows,partial=read_rows(path)
        if partial:raise ValueError("wait for the last live JSONL record to finish")
        for row in rows:
            if row["record_type"]!="growth_pair":continue
            raw=row["holdout"];stat=paired_bootstrap(raw["frozen_ms"],raw["free_ms"])
            if not abs(stat["ratio_of_medians"]-row["speedup_frozen_over_free"])<1e-10:
                raise ValueError("stored speedup disagrees with raw measured durations")
            records.append({"framework":row["framework"],"small_case_id":row["small_case_id"],
                "expanded_case_id":row["expanded_case_id"],"same_plan":row["same_plan"],
                "search_scope":row["search_scope"],"worker_saved_paired_ratio_interval":row["paired_ratio_bootstrap_95_interval"],
                "corrected_estimand_statistics":stat,
                "same_process_candidate_positive":not row["same_plan"] and
                    stat["ratio_of_medians_paired_bootstrap_95_interval"][0]>1.03})
    args.output.mkdir()
    with (args.output/"holdout_estimand_audit.json").open("x") as f:json.dump(records,f,indent=2,ensure_ascii=False)
    lines=["# RQ2 holdout 统计量复核","",
           "全部时间来自已保存 GPU CUDA-event 测量；本程序没有重新跑 kernel。保留原始报告，不覆盖。","",
           "原 worker 的 point estimate 是 median(frozen_ms)/median(free_ms)，而原区间针对 median(frozen_ms[i]/free_ms[i])。它们不是相同统计量，不能把原区间直接叫作 ratio-of-medians 的置信区间。这里从原始 AB/BA 配对样本同时重算两个区间。","",
           "新 ratio-of-medians 区间使用同一批抽样索引同时重采样分子、分母，再计算中位数比；不是分别打乱两侧样本。","",
           "正例仍仅是同进程有限样本中的 candidate positive；未经独立进程重复、A/A 噪声校准和多重比较控制，不能宣布普遍科学结论。有限候选 best-measured 也不能冒称 exact oracle。","",
           "|adapter|扩图对|ratio-of-medians 下界 >1.03 且方案不同|相同方案 A/A 对|","|---|---:|---:|---:|"]
    for framework in sorted({r["framework"] for r in records}):
        values=[r for r in records if r["framework"]==framework]
        lines.append(f"|{framework}|{len(values)}|{sum(r['same_process_candidate_positive'] for r in values)}|{sum(r['same_plan'] for r in values)}|")
    (args.output/"RQ2_HOLDOUT_ESTIMAND_AUDIT_CN.md").write_text("\n".join(lines)+"\n")
    print(args.output/"RQ2_HOLDOUT_ESTIMAND_AUDIT_CN.md")


if __name__=="__main__":main()
