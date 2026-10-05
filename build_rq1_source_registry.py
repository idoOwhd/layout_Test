#!/usr/bin/env python3
"""All-link coverage ledger: no cherry-picking, no inferred native successes."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent


class CopyValues(HTMLParser):
    def __init__(self):
        super().__init__();self.values=[];self.scripts=[];self.in_json=False
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=="clipboard-copy" and len(attrs.get("value",""))>100:
            self.values.append(attrs["value"])
        if tag=="script" and attrs.get("type")=="application/json":self.in_json=True
    def handle_endtag(self,tag):
        if tag=="script":self.in_json=False
    def handle_data(self,data):
        if self.in_json:self.scripts.append(data)


def html_body(path):
    parser=CopyValues();parser.feed(Path(path).read_text())
    if parser.values:return parser.values[0],"clipboard-copy original markdown"
    def walk(obj):
        if isinstance(obj,dict):
            for key,value in obj.items():
                if key in {"body","bodyHTML","bodyHtml","rawBody"} and isinstance(value,str) and len(value)>80:
                    yield value
                yield from walk(value)
        elif isinstance(obj,list):
            for value in obj:yield from walk(value)
    for script in parser.scripts:
        try:
            bodies=list(walk(json.loads(script)))
            if bodies:return bodies[0],"GitHub embedded JSON body"
        except ValueError:pass
    return "","body unavailable"


def candidates(text):
    """Candidate causal analogues, NOT an assertion of exact native coverage."""
    t=text.lower();out=[]
    rules={
        "attention_bmm_oproj":("o_proj","o-proj","attention output","bmm output","attention→"),
        "projection_sdpa":("qkv","q/k/v","q/k layout","dense prefill"),
        "sparse_gather_sdpa":("sparse","gather","ragged","dsv4","deepseek-v4","indexer"),
        "projection_swiglu_down":("swiglu","silu","activation→","gate/up"),
        "moe_dispatch_expert":("dispatch","moe","expert","deepep"),
        "moe_gemm_combine":("combine","gemm2","fc2","moe"),
        "state_prefill_decode":("state","linear attention","linear-attention","kda","gdn","delta"),
        "projection_gdn":("gdn","gated delta","gated_delta","qkvz","packed qkv"),
        "gemm_bias_gemm":("epilogue","bias","transpose","gemm"),
        "reduce_norm_gemm":("norm","reduction","reduce","softmax"),
        "host_weight_gemm":("weight prefetch","weight offload","host-to-gpu weight","host weight","marlin"),
        "host_kv_sdpa":("host-to-gpu","offload","kv restore","pd disaggregation"),
        "kv_writer_flashinfer":("cache","kv writer","paged","hnd","nhd","attention"),
    }
    for family,words in rules.items():
        if any(w in t for w in words):out.append(family)
    return out


def build(contexts,audit,cases):
    source={r["url"]:r for r in audit.get("records",[])}
    ledger={}
    for context in contexts["records"]:
        original=context["url"];row=source.get(original,{})
        canonical=row.get("canonical_url") or original
        canonical=re.sub(r"[#?].*$","",canonical).rstrip("/")
        if canonical in ledger:
            ledger[canonical]["url_aliases"].append(original);continue
        body=row.get("body","");method="GitHub REST issue/PR body" if body else "not fetched"
        if not body and row.get("fetch_origin")=="html" and row.get("artifact"):
            body,method=html_body(row["artifact"])
        text=row.get("title","")+"\n"+body+"\n"+"\n".join(context["shared_discussion_context"])
        families=candidates(text)
        requirements=[]
        # These are hints to be checked before any native replay, not grounds
        # for silently deleting entries from the requested coverage matrix.
        hardware_text=(row.get("title","")+"\n"+body).lower()
        for label,words in {
            "ROCm/AMD":("rocm","mi355","gfx950","mfma"),
            "Blackwell/TMEM/NVFP4":("blackwell","tmem","tcgen05","nvfp4"),
            "Hopper/SM90":("hopper","sm90","wgmma"),
            "multi-GPU/network":("multi-gpu","all-to-all","all2all","nccl","nvshmem","deepep"),
        }.items():
            if any(w in hardware_text for w in words):requirements.append(label)
        matches=[c["case_id"] for c in cases if c["family"] in families]
        ledger[canonical]={
            "url":canonical,"url_aliases":[original],"title":row.get("title",""),
            "source_status":row.get("source_status","not_fetched"),
            "source_body_extraction":method,"source_body_sha256":hashlib.sha256(body.encode()).hexdigest() if body else None,
            "source_body_excerpt":body[:12000],"source_artifact":row.get("artifact"),
            "changed_files":row.get("changed_files",[]),"upstream_test_files":row.get("test_files",[]),
            "shared_discussion_context":context["shared_discussion_context"],
            "candidate_causal_families":families,"candidate_case_ids":matches,
            "mapping_status":"candidate_only_needs_contract_review" if families else "no_implemented_analogue",
            "native_requirement_hints":requirements,
            "native_replay_status":"missing_per_source_adapter",
            "native_replay_completed":False,
            "reason":"Generic operator analogue is NOT execution of this PR/issue. Native replay requires pinned upstream revision, exact P/C contract, dtype and hardware.",
        }
    return sorted(ledger.values(),key=lambda r:r["url"])


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-audit",type=Path);p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    contexts=json.loads((HERE/"rq1_shared_source_contexts.json").read_text())
    audit=json.loads(args.source_audit.read_text()) if args.source_audit else {}
    cases=json.loads(args.manifest.read_text())["cases"]
    rows=build(contexts,audit,cases)
    result={"input_unique_urls":len(contexts["records"]),"canonical_unique_urls":len(rows),
        "source_status_counts":dict(Counter(r["source_status"] for r in rows)),
        "native_replays_completed":sum(r["native_replay_completed"] for r in rows),
        "all_link_native_validation_complete":all(r["native_replay_completed"] for r in rows),"records":rows}
    (args.output_dir/"all_source_registry.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    lines=["# RQ1：共享讨论全部 PR/issue 去重与覆盖账本","",
        f"输入唯一 URL：{len(contexts['records'])}；canonical 去重后：{len(rows)}。",
        "源码核查、机制 analogue、原 PR 原生复现是三个不同的完成度。下面不把 analogue 算为原 PR 测试成功。",
        "所有链接的原生复现目前未完成。硬件提示需人工结合源码核实；不自动排除条目。","",
        "| URL / 实际 title | 源码状态 | 候选机制（未审核等价性） | 硬件提示 | 原生 replay |",
        "|---|---|---|---|---|"]
    for r in rows:
        title=r["title"].replace("|","\\|").replace("\n"," ")
        lines.append(f"| [{title or r['url']}]({r['url']}) | {r['source_status']}; {r['source_body_extraction']} | {', '.join(r['candidate_causal_families']) or '缺少'} | {', '.join(r['native_requirement_hints']) or '待核实'} | {r['native_replay_status']} |")
    lines.extend(["","## 去重规则","",
        "1. URL 去除锚点/query，使用可核查 canonical URL；API/HTML 核查失败仍保留。",
        "2. 重复引用不增加 source 数；全部上下文单独保存在 rq1_shared_source_contexts.json。",
        "3. 可执行 contract 以算子族、真正使用的维度、B/q/KV、dtype 等哈希；模型名不同不产生额外独立样本。",
        "4. 独立 process repetition 不去重，它们是统计重复，不是新 shape。",
        "5. candidate_case_ids 只用于人工找相关机制，不能据此自动填 native replay success。"])
    (args.output_dir/"ALL_PR_ISSUE_COVERAGE_CN.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="records"}))


if __name__=="__main__":main()
