#!/usr/bin/env python3
"""Report measured conditional tests without promoting them to full proofs."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True,type=Path);root=p.parse_args().output
    summary=[];points=[];failures=[]
    for path in sorted(root.glob("*.jsonl")):
        rows=[json.loads(l) for l in path.read_text().splitlines() if l]
        if not rows:continue
        metadata=rows[0];framework=metadata["framework"]
        domain=[r for r in rows if r["record_type"]=="conditional_domain_result"]
        anchor=[r for r in rows if r["record_type"]=="runtime_anchor_result"]
        fail=[r for r in rows if r["record_type"] in {"case_failure","group_failure"}];failures.extend(fail)
        completion=next((r for r in rows if r["record_type"]=="completion"),{})
        def positive(r):return not r["holdout"]["same_choice"] and \
            r["holdout"]["statistics"]["ratio_of_medians_paired_bootstrap_95_interval"][0]>1.03
        summary.append({"framework":framework,"expected_graphs":metadata["expected_graphs"],
            "domain_graphs":len(domain),"conditional_K2_near_required":sum(r["conditional_explicit_output_K_epsilon"]==2 for r in domain),
            "domain_holdout_candidate_positive":sum(map(positive,domain)),"anchor_graphs":len(anchor),
            "anchor_holdout_candidate_positive":sum(map(positive,anchor)),
            "anchor_semantically_inapplicable":sum(r["record_type"]=="anchor_semantically_inapplicable" for r in rows),
            "failures":len(fail),"completion":completion})
        for kind,values in (("domain",domain),("anchor",anchor)):
            for r in values:
                points.append({"framework":framework,"kind":kind,"case_id":r["case"]["case_id"],
                    "model":r["case"]["model_id"],"stage":r["case"]["stage"],
                    "batch":r["case"]["batch"],"query_length":r["case"]["query_length"],
                    "kv_lengths":r["case"]["kv_lengths"],"statistics":r["holdout"]["statistics"],
                    "same_choice":r["holdout"]["same_choice"],"candidate_positive":positive(r),
                    "semantic_operator_hops":r.get("factors",{}).get("semantic_operator_hops")})
    statusfile=root/"process_status.tsv"
    if statusfile.exists():
        for line in statusfile.read_text().splitlines():
            framework,status=line.split("\t")
            if not any(s["framework"]==framework for s in summary):
                summary.append({"framework":framework,"expected_graphs":summary[0]["expected_graphs"] if summary else "unknown",
                    "domain_graphs":0,"conditional_K2_near_required":0,"domain_holdout_candidate_positive":0,
                    "anchor_graphs":0,"anchor_holdout_candidate_positive":0,"anchor_semantically_inapplicable":0,
                    "failures":1,"completion":{},"process_exit":int(status)})
                failures.append({"framework":framework,"record_type":"process_failure_before_metadata",
                    "process_exit":int(status),"log":str(root/"logs"/(framework+".log"))})
    lines=["# RQ2 额外 representation 与远端 KV anchor 的受控实验","",
        "## 不能扩大解释的边界","",
        "H2.2 子实验：真的创建第二份 packed-QKV output 存储，producer 写 row/column buffer，再实际 copy 到共同 consumer 表示。K 只统计这个值的显式输出表示数（1/2），不是整图或寄存器/shared-memory domain 数。其余布局固定在原测试的正确 best-measured 方案。","",
        "H2.3 子实验：固定同一个真实 forward、shape、数学和其它所有选择，完整测 2×2 上游选择×远端 KV NHD/HND anchor。图距离由原 forward DAG 的最短有向算子路径计算，不以 decoder block 数或 Hamming 代替。它验证的是运行时规划的条件响应，不是 compiler/layout-pass 的传播距离，也没有 control-flow compiler ablation。","",
        "两类实验均先重新训练选择，再在新的配对 AB/BA holdout 中比较；每次 pipeline 前后都检查完整计算、KV 和 guards。copy 源/目的在下一层重用 backing 前逐 bit 验证。实际转换计入时间。大模型矩阵未缩小。","",
        "原始源码 PR/issue replay 和原生 engine 默认 policy 仍是独立缺口。本报告不宣布 RQ2 全部要求已完成。","",
        "## 实测结果","","|adapter|应测图|domain 图|条件 K2 近优要求|domain 候选正例|anchor 图|anchor 候选正例|anchor 语义不适用|失败|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append("|"+"|".join(str(s[k]) for k in ("framework","expected_graphs","domain_graphs",
            "conditional_K2_near_required","domain_holdout_candidate_positive","anchor_graphs",
            "anchor_holdout_candidate_positive","anchor_semantically_inapplicable","failures"))+"|")
    lines += ["","分子：shared/旧近优上游约束方案在同一扩展图/改变后的 anchor 条件下的实测 holdout 中位时间；分母：训练选出的 free 方案的实测 holdout 中位时间。比值、K_epsilon、bootstrap 都是计算统计量，不是额外测速。","",
        "候选正例要求对应 ratio-of-medians 的配对 bootstrap 下界 >1.03，且两个方案不同；仍需独立进程与多重比较控制。没有正例应按场景负结果报告，不得改口认为所有框架都存在该问题。","",
        "semantic inapplicable 仅限没有 KV anchor 或没有两个不同且有有向依赖的实际物理选择；缺 baseline/环境/计算错误均计 failure，不用 N/A 掩盖。所有详细正反例、模型、结构范围、shape 见 conditional_case_points.json 和原始 JSONL。"]
    with (root/"RQ2_DOMAIN_ANCHOR_RESULTS_CN.md").open("x") as f:f.write("\n".join(lines)+"\n")
    for name,value in (("summary.json",summary),("conditional_case_points.json",points),("failures.json",failures)):
        with (root/name).open("x") as f:json.dump(value,f,indent=2,ensure_ascii=False)
    w=1100;h=90+60*max(1,len(summary));svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-size="18">Conditional runtime representation / anchor holdout ratios (NOT full compiler proofs)</text>']
    values=[x["statistics"]["ratio_of_medians"] for x in points];lo=min([.9]+values);hi=max([1.1]+values)
    def xx(x):return 240+800*(x-lo)/(hi-lo)
    for i,s in enumerate(summary):
        y=65+i*60;svg.append(f'<text x="20" y="{y}" font-size="15">{s["framework"]}</text>')
        svg.append(f'<line x1="{xx(1)}" x2="{xx(1)}" y1="{y-15}" y2="{y+25}" stroke="#aaa"/>')
        for j,r in enumerate(x for x in points if x["framework"]==s["framework"]):
            color="#009E73" if r["candidate_positive"] else "#777"
            svg.append(f'<circle cx="{xx(r["statistics"]["ratio_of_medians"])}" cy="{y+(j%5)*4}" r="3" fill="{color}" opacity="0.6"/>')
    svg.append('</svg>')
    with (root/"conditional_domain_anchor_speedup.svg").open("x") as f:f.write("\n".join(svg))
    print(root/"RQ2_DOMAIN_ANCHOR_RESULTS_CN.md")


if __name__=="__main__":main()
