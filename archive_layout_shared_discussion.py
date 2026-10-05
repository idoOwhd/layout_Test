#!/usr/bin/env python3
"""Archive every visible user/final answer and canonical issue/PR reference.

Offline only. Tool streams/reasoning are deliberately excluded. Generated
archives/catalogs are new files and refuse to overwrite an existing file.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

URL_RE=re.compile(r"https://github\.com/([\w.-]+/[\w.-]+)/(pull|issues)/(\d+)")
REPOS={"vllm":"vllm-project/vllm","sglang":"sgl-project/sglang",
    "flashinfer":"flashinfer-ai/flashinfer","triton":"triton-lang/triton",
    "tilelang":"tile-ai/tilelang","cutlass":"NVIDIA/cutlass","tvm":"apache/tvm",
    "tensorrt-llm":"NVIDIA/TensorRT-LLM","trt-llm":"NVIDIA/TensorRT-LLM",
    "flashattention":"Dao-AILab/flash-attention","flash-attention":"Dao-AILab/flash-attention",
    "deepgemm":"deepseek-ai/DeepGEMM","aiter":"ROCm/aiter"}
ALIAS_RE=re.compile(r"\b("+"|".join(sorted(REPOS,key=len,reverse=True))+r")\b",re.I)


def messages(html):
    match=re.search(r'streamController.enqueue\((".*?")\);',html,re.S)
    if not match:raise ValueError("shared page has no serialized message stream")
    array=json.loads(json.loads(match.group(1)))
    def obj(value):
        return {str(array[int(k[1:])]) if k.startswith("_") and k[1:].isdigit() else k:v for k,v in value.items()}
    visible=[];seen=set()
    for raw in array:
        if not isinstance(raw,dict):continue
        value=obj(raw)
        if "author" not in value or "content" not in value:continue
        try:
            author=obj(array[value["author"]]);role=array[author["role"]]
            channel=array[value["channel"]] if isinstance(value.get("channel"),int) else None
            if role!="user" and not (role=="assistant" and channel=="final"):continue
            content=obj(array[value["content"]]);parts=array[content["parts"]]
            text="\n".join(array[p] for p in parts if isinstance(array[p],str))
        except (KeyError,TypeError,IndexError):continue
        digest=hashlib.sha256(text.encode()).hexdigest()
        if text and (role,digest) not in seen:
            seen.add((role,digest));visible.append(dict(role=role,text=text,sha256=digest))
    if not any(m["role"]=="assistant" for m in visible):raise ValueError("no visible final answers")
    return visible


def catalog(visible,source,html_sha):
    by={}
    for index,message in enumerate(visible,1):
        text=message["text"]
        for match in URL_RE.finditer(text):
            repo,kind,number=match.groups();key=(repo.lower(),int(number))
            url=f"https://github.com/{repo}/{kind}/{number}"
            row=by.setdefault(key,dict(url=url,repo=repo,kind=kind,number=int(number),aliases=[],occurrences=[]))
            if url not in row["aliases"]:row["aliases"].append(url)
            # Keep local context to make each proposed relevance independently
            # reviewable, rather than treating a mentioned URL as validated.
            previous_break=text.rfind("\n\n",0,match.start())
            start=previous_break+2 if previous_break>=0 else 0
            end=text.find("\n\n",match.end())
            if end<0:end=len(text)
            row["occurrences"].append(dict(message_index=index,role=message["role"],
                offset=match.start(),context=text[start:end]))
    records=sorted(by.values(),key=lambda r:(r["repo"].lower(),r["number"]))
    return dict(source=source,html_sha256=html_sha,unique_visible_message_count=len(visible),
        role_counts=dict(Counter(m["role"] for m in visible)),unique_pr_issue_count=len(records),
        dedup_key="case-insensitive repository plus GitHub issue/PR number; query/anchor stripped",
        urls=[r["url"] for r in records],records=records,
        native_replays_completed=0,scope="shared-discussion references, not validated experimental evidence")


def unlinked_references(visible,explicit):
    """Resolve bare #numbers by literal preceding stack or section heading.

    These are review candidates, not silently certified PR links. Mixed-stack
    table rows use the closest preceding named stack, not the previous section.
    """
    known={(r["repo"].lower(),r["number"]) for r in explicit["records"]};found={}
    for index,message in enumerate(visible,1):
        section=None;offset=0
        for line in message["text"].splitlines(keepends=True):
            aliases=list(ALIAS_RE.finditer(line))
            if re.match(r"^#{1,6}\s+\S",line) and len(aliases)==1:
                section=REPOS[aliases[0].group(1).lower()]
            numbers=list(re.finditer(r"(?<!\w)#(\d{2,6})\b",line))
            line_owner=section
            if numbers:
                first_prefix=[a for a in aliases if a.end()<numbers[0].start()]
                if first_prefix:
                    line_owner=REPOS[first_prefix[0].group(1).lower()]
            for match in numbers:
                # A descriptive parenthesis such as "AITER direct slices"
                # does not change the owner of subsequent SGLang references.
                # Accept only an immediate stack prefix, otherwise use the
                # explicit section owner. Preserve slash-separated ID lists.
                before=[a for a in aliases if a.end()<=match.start() and
                    re.fullmatch(r"[\s|:：,*()\-]*(?:(?:PR|Issue|RFC)\s*)?",
                                 line[a.end():match.start()],re.I)]
                if before:line_owner=REPOS[before[-1].group(1).lower()]
                repo=line_owner
                following=URL_RE.search(line,match.end())
                if following and line[match.end():following.start()] in ("](",""):
                    # These are already linked labels, not bare references.
                    repo=following.group(1)
                if not repo:continue
                key=(repo.lower(),int(match[1]))
                if key in known:continue
                row=found.setdefault(key,dict(repo=repo,number=int(match[1]),kind="unknown",
                    repository_resolution="explicit linked owner, immediate stack prefix, or section; verify primary title",
                    issue_api=f"https://api.github.com/repos/{repo}/issues/{match[1]}",occurrences=[]))
                row["occurrences"].append(dict(message_index=index,role=message["role"],
                    offset=offset+match.start(),line=line.strip()))
            offset+=len(line)
    return sorted(found.values(),key=lambda r:(r["repo"].lower(),r["number"]))


def main():
    p=argparse.ArgumentParser();p.add_argument("--html",type=Path,required=True)
    p.add_argument("--source",required=True);p.add_argument("--rq",required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--catalog",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or args.catalog.exists():raise FileExistsError("archive/catalog already exists")
    data=args.html.read_bytes();visible=messages(data.decode("utf-8"))
    source=catalog(visible,args.source,hashlib.sha256(data).hexdigest())
    lines=[f"# {args.rq} 共享讨论：完整可见内容归档（去除重复流）","",
        f"来源：{args.source}",f"HTML SHA256：{source['html_sha256']}",
        f"唯一可见消息：{len(visible)}；角色计数：{source['role_counts']}；唯一 PR/issue：{len(source['records'])}。",
        "保留全部 user 与 assistant final；不含工具流、内部 reasoning。URL 按 repo+number 去重；不是原生实验完成声明。"]
    for index,message in enumerate(visible,1):
        lines.extend(["",f"## 可见消息 {index} — {message['role']}","",message["text"]])
    with args.output.open("x",encoding="utf-8") as f:f.write("\n".join(lines)+"\n")
    with args.catalog.open("x",encoding="utf-8") as f:json.dump(source,f,ensure_ascii=False,indent=2);f.write("\n")
    print(json.dumps({k:source[k] for k in ("source","role_counts","unique_pr_issue_count","html_sha256")},ensure_ascii=False))


if __name__=="__main__":main()
