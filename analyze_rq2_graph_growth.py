#!/usr/bin/env python3
"""Summarize only saved measured records; never invent unavailable baselines."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from audit_rq2_holdout_statistics import paired_bootstrap


def audit_input_fingerprints(all_rows):
    groups={}
    for rows in all_rows:
        for row in rows:
            if row["record_type"]=="group_data_fingerprint":
                key=json.dumps(row["group"],sort_keys=True)
                framework=row["framework"]
                if framework in groups.setdefault(key,{}):
                    raise ValueError("duplicate framework/group data fingerprint")
                groups[key][framework]=row["fingerprint"]
    result=[]
    for key,values in sorted(groups.items()):
        canonical={json.dumps(v,sort_keys=True) for v in values.values()}
        result.append({"group":json.loads(key),"frameworks":sorted(values),
                       "compared_framework_count":len(values),
                       "matching_recorded_fingerprints":len(canonical)==1,
                       "cross_framework_comparison_performed":len(values)>1,
                       "scope":"full first-layer weights and RoPE table; sampled input/history KV; NOT a full-state hash"})
    return {"groups":result,"mismatching_groups":sum(not r["matching_recorded_fingerprints"] for r in result),
            "compared_groups":sum(r["cross_framework_comparison_performed"] for r in result)}


def analyze(root):
    summaries=[];pair_rows=[];failures=[];all_rows=[]
    for path in sorted(root.glob("*.jsonl")):
        rows=[json.loads(line) for line in path.read_text().splitlines() if line]
        if not rows:continue
        all_rows.append(rows)
        graphs=[r for r in rows if r["record_type"]=="graph"]
        pairs=[]
        for r in rows:
            if r["record_type"]!="growth_pair":continue
            stat=paired_bootstrap(r["holdout"]["frozen_ms"],r["holdout"]["free_ms"])
            if abs(stat["ratio_of_medians"]-r["speedup_frozen_over_free"])>1e-10:
                raise ValueError("stored speedup mismatches raw durations")
            pairs.append({**r,"worker_saved_interval":r["paired_ratio_bootstrap_95_interval"],
                          "paired_ratio_bootstrap_95_interval":stat["ratio_of_medians_paired_bootstrap_95_interval"],
                          "analysis_interval_estimand":"ratio-of-medians, same paired indices; raw worker record unmodified"})
        failed=[r for r in rows if r["record_type"] in {"group_failure","incorrect_candidate"}]
        framework=rows[0]["framework"]
        failures.extend(failed);pair_rows.extend(pairs)
        positive=[r for r in pairs if not r["same_plan"] and r["paired_ratio_bootstrap_95_interval"][0]>1.03]
        negative=[r for r in pairs if r["same_plan"] or r["speedup_frozen_over_free"]<=1.03]
        summary={"framework":framework,"graphs":len(graphs),"pairs":len(pairs),
                 "candidates":sum(r["measured_candidate_count"] for r in graphs),
                 "exact_graphs":sum(r["search_scope"]=="exact_declared_space" for r in graphs),
                 "positive_3pct_holdout_CI":len(positive),"non_positive":len(negative),
                 "failed_records":len(failed),
                 "median_speedup":statistics.median(r["speedup_frozen_over_free"] for r in pairs) if pairs else None}
        summaries.append(summary)
    statusfile=root/"process_status.tsv"
    if statusfile.exists():
        for line in statusfile.read_text().splitlines():
            framework,status=line.split("\t")
            if not any(s["framework"]==framework for s in summaries):
                summaries.append({"framework":framework,"graphs":0,"pairs":0,"candidates":0,
                    "exact_graphs":0,"positive_3pct_holdout_CI":0,"non_positive":0,"failed_records":1,
                    "median_speedup":None,"process_exit":int(status)})
                failures.append({"framework":framework,"record_type":"process_failure_before_results",
                    "process_exit":int(status),"log":str(root/"logs"/(framework+".log"))})
            elif int(status)!=0:
                failures.append({"framework":framework,"record_type":"nonzero_process_exit",
                    "process_exit":int(status),"log":str(root/"logs"/(framework+".log"))})
    inputs=audit_input_fingerprints(all_rows)
    for record in inputs["groups"]:
        if not record["matching_recorded_fingerprints"]:
            failures.append({"record_type":"cross_framework_input_mismatch",**record})
    report=root/"RQ2_GRAPH_GROWTH_RESULTS_CN.md"
    if report.exists():raise FileExistsError("refusing to overwrite analysis")
    lines=["# RQ2 FP16 真实模型子图扩展实验（受控布局空间）","",
           "## 证据边界","",
           "这些结果是完整嵌套子图的受控/hybrid 对照，不是未修改 serving engine 的默认 layout policy。逐算子的实际来源保存在 JSONL 的 primitive_origins。TVM 原生 norm、CUTLASS 原生 GEMM 等不能被表述为整个图由该框架原生编译。真实模型来源指 pinned config 和 forward 拓扑；权重/输入是固定种子的 FP16 测试数据，非下载模型权重。","",
           "小图仅对声明的二元 layout 空间穷举；多 block 的 bounded 候选结果只能称 best-measured。207 个 PR/issue 的原生 patch replay 是独立要求，本实验不替代它们。","",
           "## 结果","","|框架/adapter|图|扩图对|正确候选|穷举图|>3% 且 holdout CI 为正|失败记录|中位 speedup|",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        speed="—" if r["median_speedup"] is None else f"{r['median_speedup']:.5f}"
        lines.append(f"|{r['framework']}|{r['graphs']}|{r['pairs']}|{r['candidates']}|{r['exact_graphs']}|{r['positive_3pct_holdout_CI']}|{r['failed_records']}|{speed}|")
    lines += ["","## 如何解释","",
              f"跨框架输入审计：{inputs['compared_groups']} 组进行了至少两个框架的指纹比较，{inputs['mismatching_groups']} 组不一致。第一层全部权重与完整 RoPE 表使用全量 SHA256；激活/历史 KV 使用至多 8192 个 int64 等距索引采样，不冒充全状态哈希。详见 CROSS_FRAMEWORK_INPUT_AUDIT.json。","",
              "S_eps(H) = 小图中 training median 不超过 (1+epsilon)×已测最优的全部正确候选。扩图冻结旧决策在该集合内，仅补齐新增轴，与同一个候选集合的自由选择作比较。先按 training 选择两个方案，再用独立 AB/BA 配对 holdout 测量。","",
              "speedup 的分子是 frozen 扩图方案的 holdout 中位耗时，分母是 free 扩图方案的 holdout 中位耗时；两者均由完整 GPU pipeline 的 CUDA events 实测。speedup = 分子/分母、ER_set = speedup−1 是公式计算，不是额外测量。候选选择、JIT、plan、参考解、正确性检查不计入耗时；实际 contiguous、KV repair、producer、consumer 计入。小 decode 的 Python dispatch 间隙也可能进入 CUDA event 完整区间，不能解读为纯 kernel 时间。","",
              "若存在可复现且显著的正 ER_set，支持旧图近优布局集合对扩图不稳定；零/负值或置信区间跨 1 是反例/不确定，不可强写成问题必然存在。多 block 的测得差距是有限候选下的差距，不是全空间后悔值。","",
              "holdout CI 对同一进程内配对重复 bootstrap，并非跨进程/跨 GPU 置信区间。要推广到多数框架的默认优化策略，还需要 engine policy/IR 捕获、native whole-graph replay、更多模型家族和独立进程重复。","",
              "本分析始终从原始holdout样本重新计算与point estimate一致的ratio-of-medians配对区间；旧worker保存的median-of-paired-ratios区间保留为worker_saved_interval，不再混用。没有改变原始JSONL或重跑GPU。","",
              "near-set survival 和 minimum_old_axis_changes 只统计已测近优集合在旧轴上的投影、最少 Hamming 差异；它们不等于编译器传播距离。原讨论的 H2.2（真实 layout domain 分区及 K_epsilon）和 H2.3（远距离因果传播、control flow）仍需专门实验，不能用二元布局数量或 decoder block 数冒充。","",
              "## 失败与未完成","",
              f"失败记录 {len(failures)} 个，原始原因详见 failure_records.json。Hexcute、跨架构、多 GPU 不在本次执行内；源 PR 的 BF16/FP8/AMD/新 SM 限制不能用 FP16 A10 受控图冒充已验证。"]
    report.write_text("\n".join(lines)+"\n")
    for name,value in (("summary.json",summaries),("growth_pairs.json",pair_rows),("failure_records.json",failures),
                       ("CROSS_FRAMEWORK_INPUT_AUDIT.json",inputs)):
        with (root/name).open("x") as f:json.dump(value,f,ensure_ascii=False,indent=2)
    manifest=json.loads((root/"design_manifest.json").read_text())
    completeness={"declared_case_count":manifest["case_count"],"frameworks":{},
                  "H2_2_K_epsilon_complete":False,"H2_3_causal_propagation_complete":False,
                  "cross_framework_recorded_fingerprints_match":inputs["mismatching_groups"]==0,
                  "all_207_native_source_replays_complete":False,"all_requested_requirements_complete":False}
    for path in sorted(root.glob("*.jsonl")):
        rows=[json.loads(line) for line in path.read_text().splitlines() if line]
        if not rows:continue
        metadata=next(r for r in rows if r["record_type"]=="run_metadata")
        expected=next((r["expected_graphs"] for r in rows if r["record_type"]=="completion"),None)
        success=[r for r in rows if r["record_type"]=="graph" and r.get("executed_graph_status") in
                  (None,"correctness_verified_complete_controlled_graph")]
        complete=next((r for r in rows if r["record_type"]=="completion"),{})
        counts=complete.get("counts",{})
        unique_ids=len({r["case"]["case_id"] for r in success})==len(success)
        status=dict(line.split("\t") for line in statusfile.read_text().splitlines()).get(metadata["framework"]) if statusfile.exists() else None
        completeness["frameworks"][metadata["framework"]]={
            "expected_mode_graphs":expected,"correctness_verified_graphs":len(success),
            "controlled_H2_1_scope_complete":expected is not None and len(success)==expected and
                counts.get("failures",1)==0 and counts.get("incorrect_candidates",1)==0 and unique_ids and status=="0",
            "unique_case_ids":unique_ids,"process_exit":status,
            "mode":metadata["args"]["mode"],"whole_native_engine_policy_verified":False}
    with (root/"REQUIREMENTS_COMPLETENESS.json").open("x") as f:json.dump(completeness,f,ensure_ascii=False,indent=2)
    # Self-contained SVG, all plotted points come from saved holdout data.
    w=1000;h=100+75*len(summaries);svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-size="18">RQ2: frozen/free measured holdout speedup (not engine-default policy)</text>']
    all_speed=[r["speedup_frozen_over_free"] for r in pair_rows];lo=min([.9]+all_speed);hi=max([1.1]+all_speed)
    def xx(x):return 250+(x-lo)/(hi-lo)*700
    for i,s in enumerate(summaries):
        y=70+75*i;svg.append(f'<text x="20" y="{y}" font-size="15">{s["framework"]}: {s["pairs"]} pairs</text>')
        svg.append(f'<line x1="{xx(1):.2f}" x2="{xx(1):.2f}" y1="{y-20}" y2="{y+30}" stroke="#aaa"/>')
        for j,r in enumerate(p for p in pair_rows if p["framework"]==s["framework"]):
            color="#009E73" if r["paired_ratio_bootstrap_95_interval"][0]>1.03 and not r["same_plan"] else "#777"
            svg.append(f'<circle cx="{xx(r["speedup_frozen_over_free"]):.2f}" cy="{y+(j%5)*4:.2f}" r="3" fill="{color}" opacity="0.65"/>')
    svg.append(f'<text x="250" y="{h-15}">{lo:.3f}</text><text x="920" y="{h-15}">{hi:.3f}</text></svg>')
    (root/"rq2_growth_speedup.svg").write_text("\n".join(svg))
    print(json.dumps({"report":str(report),"summaries":summaries},ensure_ascii=False))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    analyze(p.parse_args().output)
