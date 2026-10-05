#!/usr/bin/env python3
"""Final correctness/expected-coverage gate; process exit 0 alone is insufficient."""
import argparse
from collections import Counter,defaultdict
import json
import hashlib
from pathlib import Path

NATIVE={"projection_gdn","projection_swiglu_down","kv_writer_flashinfer","kv_writer_swa_flashinfer"}
TORCH={"attention_bmm_oproj","projection_sdpa","sparse_gather_sdpa","projection_swiglu_down",
       "moe_dispatch_expert","moe_gemm_combine","state_prefill_decode","gemm_bias_gemm",
       "reduce_norm_gemm","host_weight_gemm","host_kv_sdpa"}


def expected_cells(cases,reps):
    return [(c["case_id"],mode,rep) for c in cases
            for mode in ((["torch"] if c["family"] in TORCH else [])+
                         (["vllm","sglang"] if c["family"] in NATIVE else []))
            for rep in range(reps)]


def verify(root):
    cases=json.loads((root/"manifest.json").read_text())["cases"]
    preflight=json.loads((root/"preflight.json").read_text())
    reps=preflight["independent_process_repetitions"]
    by=defaultdict(list);status=Counter()
    for path in sorted((root/"raw").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row=json.loads(line);status[row["status"]]+=1
            by[(row["case_id"],row["consumer_mode"],row["process_repetition"])].append(row)
    gaps=[];checked=0;max_nrmse=0.
    here=Path(__file__).resolve().parent
    for name,digest in preflight.get("source_sha256",{}).items():
        if hashlib.sha256((here/name).read_bytes()).hexdigest()!=digest:
            gaps.append(dict(reason="implementation changed during run",source=name))
    cell_plan=root/"retry_plan.json"
    if cell_plan.exists():
        plan=json.loads(cell_plan.read_text())
        cells=[(c["case_id"],c["consumer_mode"],c["process_repetition"]) for c in plan["selected_cells"]]
        if len(cells)!=len(set(cells)):gaps.append(dict(reason="duplicate retry cells"))
        valid=set(expected_cells(cases,plan["source_repetitions"]))
        if set(cells)-valid:gaps.append(dict(reason="unknown retry cells"))
    else:cells=expected_cells(cases,reps)
    for case_id,mode,rep in cells:
        rows=by.get((case_id,mode,rep),[])
        successful=[r for r in rows if r["status"]=="success"]
        stages={r["stage"] for r in successful}
        bad=[r for r in rows if r["status"] not in {"success","illegal_consumer_contract"}]
        signatures=[(r["stage"],r["strategy"]) for r in successful]
        if len(signatures)!=len(set(signatures)):
            gaps.append(dict(case_id=case_id,mode=mode,repetition=rep,reason="duplicate stage/strategy measurements"))
        if not {"producer","edge"}.issubset(stages) or bad:
            gaps.append(dict(case_id=case_id,mode=mode,repetition=rep,
                statuses=sorted({r["status"] for r in rows}),error=[r.get("error") for r in bad]));continue
        for row in successful:
            check=row.get("correctness",{})
            if not check.get("correct"):
                gaps.append(dict(case_id=case_id,mode=mode,repetition=rep,reason="successful row lacks passing correctness"))
            def visit(c):
                nonlocal max_nrmse
                if "normalized_rmse" in c:max_nrmse=max(max_nrmse,c["normalized_rmse"])
                for child in c.get("parts",[]):visit(child)
            visit(check)
            if not row.get("boundary"):
                gaps.append(dict(case_id=case_id,mode=mode,reason="missing tensor stride metadata"))
        checked+=1
    return {"controlled_hybrid_correctness_and_coverage_pass":not gaps,"checked_case_mode_process_cells":checked,
        "measurement_status_counts":dict(status),"max_observed_normalized_rmse":max_nrmse,
        "gaps":gaps,"all_173_PR_native_replays_complete":False,
        "note":"Correctness pass does not imply RQ1 supported, global optimality, all-framework default-layout coverage, or native PR replay."}


def main():
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,required=True);args=p.parse_args()
    result=verify(args.output_dir)
    with (args.output_dir/"correctness_coverage.json").open("x") as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write("\n")
    lines=["# 最终计算正确性与覆盖检查","",
        f"controlled/hybrid 计算正确性及预期执行覆盖通过：**{result['controlled_hybrid_correctness_and_coverage_pass']}**。",
        f"已检查 case × consumer × process：{result['checked_case_mode_process_cells']}。",
        f"最大已观察 normalized RMSE：{result['max_observed_normalized_rmse']:.8g}（阈值 0.01）。",
        "FP16 producer 输入/权重/KV/主要输出；FP32 accumulator、GDN state/gate 和数学参考单列，不声称纯 FP16 内部运算。",
        "成功 row 必须存在对应 P-only 和完整 edge、passing correctness 和真实 shape/stride。",
        "numerical mismatch、unsupported、OOM preflight、missing、process timeout 都留在缺口里，不算成功。非法 stride 的预期拒绝单独记录。",
        "所有 173 个 PR 原生 replay：**未完成**。与上面的 controlled suite 检查不同。","",
        "## 未通过条目","", "```json",json.dumps(result['gaps'],ensure_ascii=False,indent=2),"```"]
    with (args.output_dir/"CORRECTNESS_AND_COVERAGE_CN.md").open("x") as f:f.write("\n".join(lines)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="gaps"}))
    return 0 if result["controlled_hybrid_correctness_and_coverage_pass"] else 1


if __name__=="__main__":raise SystemExit(main())
