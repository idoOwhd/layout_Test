#!/usr/bin/env python3
"""Fail-closed RQ1 paired-process analysis with held-out winner selection."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import random
import statistics as S


def bootstrap(values,seed=17,draws=4000):
    rng=random.Random(seed)
    means=sorted(S.mean(rng.choices(values,k=len(values))) for _ in range(draws))
    return [means[int(.025*(draws-1))],means[int(.975*(draws-1))]]


def assess(rows):
    failures=[r for r in rows if r.get("status") not in {"success","illegal_consumer_contract"}]
    if failures:
        return {"status":"measurement_failed","failure_statuses":sorted({r["status"] for r in failures}),
                "failed_process_repetitions":sorted({r["process_repetition"] for r in failures})}
    accepted=[r for r in rows if r.get("status")=="success" and
              r.get("correctness",{}).get("correct") and r.get("p50_ms",0)>0]
    by=defaultdict(dict)
    for r in accepted:
        if r["stage"] not in {"producer","edge"}:continue
        key=(r["stage"],r["strategy"])
        rep=r["process_repetition"]
        if rep in by[key]:return {"status":"invalid_duplicate_measurement"}
        by[key][rep]=r["p50_ms"]
    producers={k[1]:v for k,v in by.items() if k[0]=="producer"}
    edges={k[1]:v for k,v in by.items() if k[0]=="edge"}
    if len(producers)<2 or not edges:return {"status":"insufficient_layouts"}
    reps=sorted(set.intersection(*(set(v) for v in [*producers.values(),*edges.values()])))
    if len(reps)<5:return {"status":"smoke_only_insufficient_process_repetitions","paired_process_count":len(reps)}
    train=reps[:len(reps)//2];test=reps[len(reps)//2:]
    pbest=min(producers,key=lambda k:S.median(producers[k][r] for r in train))
    local_edges={k:v for k,v in edges.items() if k.startswith(pbest+"/")}
    if not local_edges:return {"status":"local_layout_has_no_legal_edge"}
    local=min(local_edges,key=lambda k:S.median(local_edges[k][r] for r in train))
    oracle=min(edges,key=lambda k:S.median(edges[k][r] for r in train))
    regrets=[(edges[local][r]-edges[oracle][r])/edges[local][r] for r in test]
    speedups=[edges[local][r]/edges[oracle][r] for r in test]
    win_p=sum(min(producers,key=lambda k:producers[k][r])==pbest for r in reps)/len(reps)
    win_e=sum(min(edges,key=lambda k:edges[k][r])==oracle for r in reps)/len(reps)
    cvs=[]
    for values in (producers[pbest],edges[local],edges[oracle]):
        xs=[values[r] for r in reps];cvs.append(S.stdev(xs)/S.mean(xs))
    noise=max(.03,3*max(cvs));ci=bootstrap(regrets)
    diff=oracle.split("/")[0]!=pbest
    convincing=(diff and win_p>=.8 and win_e>=.8 and ci[0]>0 and S.mean(regrets)>noise)
    status="rq1_supported_in_candidate_space" if convincing else (
        "no_layout_conflict_in_candidate_space" if not diff else "inconclusive_noise_or_winner_instability")
    return dict(status=status,paired_process_count=len(reps),selection_repetitions=train,
        held_out_repetitions=test,producer_local_layout=pbest,local_layout_best_legal_edge=local,
        complete_edge_selected_strategy=oracle,layout_conflict=diff,
        producer_winner_stability=win_p,edge_winner_stability=win_e,
        mean_held_out_regret_fraction=S.mean(regrets),regret_95ci=ci,
        median_held_out_speedup=S.median(speedups),speedup_numerator="measured complete edge under P-only selected layout, with its best legal repair selected on train",
        speedup_denominator="measured complete edge under globally edge-selected strategy, selected on train",
        speedup_is_derived_ratio_of_measurements=True,noise_and_practical_gate_fraction=noise,
        measured_local_edge_ms=[edges[local][r] for r in test],
        measured_oracle_edge_ms=[edges[oracle][r] for r in test],
        native_unmodified_framework_baseline_measured=False,
        scope="Finite enumerated candidate layouts; not a proof of global layout optimality")


def main():
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args();root=args.output_dir
    groups=defaultdict(list);statuses=Counter()
    for path in sorted((root/"raw").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            r=json.loads(line);statuses[r["status"]]+=1
            groups[(r["case_id"],r["consumer_mode"])].append(r)
    findings=[]
    for (case_id,mode),rows in sorted(groups.items()):
        findings.append(dict(case_id=case_id,consumer_mode=mode,family=rows[0]["family"],
            case=rows[0]["case"],**assess(rows)))
    counts=Counter(f["status"] for f in findings)
    source_path=root/"source_coverage/all_source_registry.json"
    source=json.loads(source_path.read_text()) if source_path.exists() else {}
    result={"measurement_status_counts":dict(statuses),"case_mode_result_counts":dict(counts),
        "all_link_native_validation_complete":source.get("all_link_native_validation_complete",False),
        "native_PR_replays_completed":source.get("native_replays_completed",0),"findings":findings}
    target=root/"RQ1_DIVERSITY_FINDINGS.json"
    with target.open("x") as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write("\n")
    lines=["# RQ1 多样性实验：逐 case 结果","",
        "这是 controlled/hybrid 算子边界测量，不是所有框架的默认 layout，也不是全部 PR 的原生复现。",
        f"measurement status：`{dict(statuses)}`。",f"case × consumer mode：`{dict(counts)}`。",
        f"全部 PR/issue 原生复现完成：**{result['all_link_native_validation_complete']}**。","",
        "## 实验与词语定义","",
        "P：真正执行计算并直接写候选布局的 producer。C：实际后续算子。R：显式修复（如 contiguous 或路由重排）。",
        "P-only 只包 producer；C-only 使用预先准备但保持真实 stride 的输入；R-only 只包修复；edge 用一对 CUDA event 包 P→R→C 的实际执行。不能把三项 median 相加冒充 edge 实测。",
        "每条合法路径先检查 P 逻辑输出和最终 C 输出，再计时。FP16 rtol=0.03，atol=max(1e-5,4×2^-10×reference RMS)，另要求 normalized RMSE≤0.01 且全部有限。不合法 stride 不喂给 native kernel。",
        "至少 5 个独立进程；前 floor(n/2) 个选择 P-only 最优 layout 和两条 complete-edge 策略，后面的进程只评估，避免在同一数据上选 winner 后报告乐观 oracle。",
        "speedup = T_edge(P-only 所选 layout＋其最佳合法修复) / T_edge(edge 联合所选策略)。分子/分母均是完整路径实测延迟；比值是计算值。",
        "regret = (分子延迟 − 分母延迟)/分子延迟。CI 是 held-out paired process regret 的 bootstrap mean 95% CI（只有 3 个 held-out 样本时 CI 可信度有限）。",
        "支持 RQ1 要求不同 P layout、P/C-edge winner ≥80% 稳定、CI 下界>0，且平均 regret > max(3%,3×process CV)。不满足只能报告不确定/未发现；不能倒推证明 RQ1 普遍不存在。",
        "候选空间中的最优不等于硬件/所有合法 layout 的全局最优；框架原始默认策略没有在本测试中自动替代为 local winner。",
        "执行 error/OOM/unsupported/numerical mismatch 单列 measurement_failed；不能把没有运行成功的 case 误分类为已测但只有一个独立 layout。",
        "host transfer 条目是 placement extension；无 gated state 的 contraction 不是 KDA/GDN；sparse compact attention 不是 DeepSeek 专用 MSA/QSA。",
        "模型 config 提供真实架构维度，随机 tensor 不需要 checkpoint；B/q/KV 是标明来源的受控 workload，不是线上生产分布。","",
        "## 逐 case × consumer","","| case / family | consumer | status | P-local → edge strategy | speedup（held out） |",
        "|---|---|---|---|---|"]
    for f in findings:
        speed=f.get("median_held_out_speedup")
        lines.append(f"| {f['case_id']} / {f['family']} | {f['consumer_mode']} | {f['status']} | {f.get('producer_local_layout','—')} → {f.get('complete_edge_selected_strategy','—')} | {speed:.4f} |" if speed is not None else
                     f"| {f['case_id']} / {f['family']} | {f['consumer_mode']} | {f['status']} | — | — |")
    with (root/"RQ1_DIVERSITY_REPORT_CN.md").open("x") as f:f.write("\n".join(lines)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="findings"}))


if __name__=="__main__":main()
