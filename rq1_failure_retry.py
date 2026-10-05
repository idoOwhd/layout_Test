#!/usr/bin/env python3
"""Select exact failed cells and build a NEW, provenance-preserving merged view."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import shutil

from check_rq1_diversity_run import expected_cells

HERE=Path(__file__).resolve().parent


def dump(path,value):
    with path.open("x",encoding="utf-8") as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write("\n")


def read_rows(root):
    for path in sorted((root/"raw").glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                if line.strip():yield path,json.loads(line)


def key(row):return row["case_id"],row["consumer_mode"],row["process_repetition"]


def passing_cell(rows):
    good=[r for r in rows if r["status"]=="success"]
    signatures=[(r.get("stage"),r.get("strategy")) for r in good]
    return bool(good) and {"producer","edge"}.issubset({r.get("stage") for r in good}) and \
        len(signatures)==len(set(signatures)) and \
        all(r["status"] in {"success","illegal_consumer_contract"} for r in rows) and \
        all(r.get("correctness",{}).get("correct") for r in good)


def snapshot(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def select_failures(cases,reps,rows,smoke):
    by=defaultdict(list)
    for row in rows:by[key(row)].append(row)
    expected=expected_cells(cases,reps)
    failed=[cell for cell in expected if not passing_cell(by[cell])]
    if smoke:
        selected={}
        for cell in failed:selected.setdefault(cell[:2],cell)
        chosen=list(selected.values())
    else:chosen=failed
    return sorted(chosen),sorted(failed),by


def prepare(previous,output,smoke=False):
    previous=previous.resolve();output=output.resolve()
    if output==previous or previous in output.parents:
        raise ValueError("retry output must not be inside the previous results")
    data=json.loads((previous/"manifest.json").read_text());cases=data["cases"]
    pre=json.loads((previous/"preflight.json").read_text())
    if hashlib.sha256((previous/"manifest.json").read_bytes()).hexdigest()!=pre["manifest_sha256"]:
        raise ValueError("previous manifest differs from measured snapshot")
    # Only workspace capacity/selection/analysis is repaired here. Reusing
    # measurements after changing producer arithmetic or case generator would
    # require broader remeasurement, not this failed-only merge.
    for name in ("rq1_diverse_kernels.py","build_rq1_diversity_cases.py"):
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=pre["source_sha256"][name]:
            raise ValueError(f"{name} changed; failed-only reuse is not safe")
    selected,failed,by=select_failures(cases,pre["independent_process_repetitions"],
        (r for _,r in read_rows(previous)),smoke)
    if not selected:raise ValueError("no failed/missing measurement cells to rerun")
    output.mkdir(parents=True,exist_ok=False)
    for folder in ("raw","logs","selections"): (output/folder).mkdir()
    wanted={c[0] for c in selected};subset=[c for c in cases if c["case_id"] in wanted]
    dump(output/"manifest.json",{**data,"cases":subset,"case_count":len(subset),
        "family_counts":dict(Counter(c["family"] for c in subset)),
        "selection_origin":"exact failed cells from previous measured run"})
    jobs=[];cells=[]
    grouped=defaultdict(list)
    for case_id,mode,rep in selected:
        grouped[(mode,rep)].append(case_id)
        prior=by[(case_id,mode,rep)]
        cells.append(dict(case_id=case_id,consumer_mode=mode,process_repetition=rep,
            previous_statuses=sorted({r["status"] for r in prior}) or ["missing"],
            previous_errors=sorted({r.get("error","") for r in prior if r.get("error")})))
    for (mode,rep),ids in sorted(grouped.items()):
        filename=f"{mode}_rep{rep}.json"
        dump(output/"selections"/filename,{"case_ids":sorted(set(ids))})
        jobs.append((mode,rep,filename))
    with (output/"jobs.tsv").open("x") as f:
        for mode,rep,filename in jobs:f.write(f"{mode}\t{rep}\t{filename}\n")
    plan={"previous_dir":str(previous),"retry_mode":"smoke" if smoke else "full",
        "unique_case_count":len(wanted),"selected_cell_count":len(cells),
        "total_failed_cell_count":len(failed),"source_repetitions":pre["independent_process_repetitions"],
        "warmup":pre["warmup"],"iterations":pre["iterations"],
        "previous_preflight":pre,"previous_manifest_sha256":pre["manifest_sha256"],
        "previous_files_sha256":snapshot(previous),
        "selected_cells":cells}
    dump(output/"retry_plan.json",plan)
    lines=["# 只补跑失败 case：选择清单","",
        f"原始目录：`{previous}`（只读）。",
        f"不同 contract：{len(wanted)}；本次 case×consumer×process：{len(cells)}；原始失败单元：{len(failed)}。",
        "成功的历史单元不重测；一个失败单元中的所有布局/阶段一起重测，不能只换掉较慢或错误的单个 strategy。",
        "smoke 只选每个 case×consumer 的一个失败 process；不能当作已补齐全部重复。","",
        "| case | consumer | process | old status |","|---|---|---|---|"]
    lines.extend(f"| {c['case_id']} | {c['consumer_mode']} | {c['process_repetition']} | {', '.join(c['previous_statuses'])} |" for c in cells)
    with (output/"FAILED_CASE_RETRY_PLAN_CN.md").open("x") as f:f.write("\n".join(lines)+"\n")
    print(json.dumps({k:plan[k] for k in ("previous_dir","retry_mode","unique_case_count","selected_cell_count","total_failed_cell_count")}))
    return plan


def merge(retry):
    retry=retry.resolve();plan=json.loads((retry/"retry_plan.json").read_text())
    previous=Path(plan["previous_dir"])
    if hashlib.sha256((previous/"manifest.json").read_bytes()).hexdigest()!=plan["previous_manifest_sha256"]:
        raise ValueError("previous manifest changed after retry selection")
    selected={(c["case_id"],c["consumer_mode"],c["process_repetition"]) for c in plan["selected_cells"]}
    by=defaultdict(list)
    for path,row in read_rows(retry):by[key(row)].append((path,row))
    if set(by)-selected:raise ValueError("retry contains unselected measurement cells")
    replacement={cell for cell in selected if passing_cell([r for _,r in by[cell]])}
    root=retry/"merged";root.mkdir(exist_ok=False);(root/"raw").mkdir()
    shutil.copyfile(previous/"manifest.json",root/"manifest.json")
    if (previous/"source_coverage").exists():shutil.copytree(previous/"source_coverage",root/"source_coverage")
    pre=json.loads((retry/"preflight.json").read_text())
    pre.update(manifest_sha256=plan["previous_manifest_sha256"],
        independent_process_repetitions=plan["source_repetitions"],
        record_type="merged_results_not_a_new_single_full_run",
        previous_measurement_preflight=plan["previous_preflight"],
        workspace_repair_policy="capacity growth at plan time only; previous successful cells reused unchanged")
    dump(root/"preflight.json",pre)
    retained=0;added=0
    with (root/"raw"/"effective_measurements.jsonl").open("x") as f:
        for path,row in read_rows(previous):
            if key(row) in replacement:continue
            row={**row,"measurement_origin":str(path)}
            f.write(json.dumps(row,ensure_ascii=False)+"\n");retained+=1
        for cell in sorted(replacement):
            for path,row in by[cell]:
                row={**row,"measurement_origin":str(path),"replaces_failed_cell_from":str(previous)}
                f.write(json.dumps(row,ensure_ascii=False)+"\n");added+=1
    original_unchanged=(snapshot(previous)==plan["previous_files_sha256"])
    if not original_unchanged:raise ValueError("previous files changed since selection; cannot certify this merge")
    result={"previous_dir":str(previous),"retry_dir":str(retry),"merged_dir":str(root),
        "selected_cells":len(selected),"replaced_cells":len(replacement),
        "unrepaired_selected_cells":[dict(case_id=c[0],consumer_mode=c[1],process_repetition=c[2]) for c in sorted(selected-replacement)],
        "retained_previous_rows":retained,"added_retry_rows":added,
        "all_previous_files_unchanged":original_unchanged,"native_PR_replays_completed":0,
        "scope":"original successful cells plus successful replacements, with row-level origins; no duplicate statistical cells"}
    dump(root/"MERGE_PROVENANCE.json",result)
    with (root/"MERGE_PROVENANCE_CN.md").open("x") as f:
        f.write("# 补跑合并来源\n\n原目录未修改；不是重新执行了一个完整 full run。\n\n```json\n"+
                json.dumps(result,ensure_ascii=False,indent=2)+"\n```\n")
    print(json.dumps(result,ensure_ascii=False))
    return result


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("prepare");q.add_argument("--previous",type=Path,required=True)
    q.add_argument("--output",type=Path,required=True);q.add_argument("--smoke",action="store_true")
    q=sub.add_parser("merge");q.add_argument("--retry-dir",type=Path,required=True)
    args=p.parse_args()
    if args.command=="prepare":prepare(args.previous,args.output,args.smoke)
    else:merge(args.retry_dir)


if __name__=="__main__":main()
