#!/usr/bin/env python3
"""Archive complete visible final assistant messages; dedup streamed copies."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def extract(text):
    match=re.search(r'streamController.enqueue\((".*?")\);',text,re.S)
    if not match:raise ValueError("shared HTML has no serialized message stream")
    array=json.loads(json.loads(match.group(1)))
    def obj(value):
        return {str(array[int(k[1:])]) if k.startswith("_") and k[1:].isdigit() else k:v for k,v in value.items()}
    finals=[];seen=set()
    for value in array:
        if not isinstance(value,dict):continue
        value=obj(value)
        if not all(k in value for k in ("author","content","channel")):continue
        try:
            author=obj(array[value["author"]]);channel=array[value["channel"]]
            if array[author["role"]]!="assistant" or channel!="final":continue
            content=obj(array[value["content"]]);parts=array[content["parts"]]
            body="\n".join(array[p] for p in parts if isinstance(array[p],str))
        except (KeyError,TypeError,IndexError):continue
        digest=hashlib.sha256(body.encode()).hexdigest()
        if body and digest not in seen:seen.add(digest);finals.append(body)
    return finals


def main():
    p=argparse.ArgumentParser();p.add_argument("--html",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    finals=extract(args.html.read_text())
    urls=sorted(set(re.findall(r"https://github\.com/[\w.-]+/[\w.-]+/(?:pull|issues)/\d+", "\n".join(finals))))
    expected=json.loads(Path(__file__).with_name("rq1_shared_github_urls.json").read_text())["urls"]
    if set(urls)!=set(expected):raise ValueError("archive URL set differs from checked-in all-link catalog")
    header=["# RQ1 共享讨论完整可见回答归档（去重）","",
        "来源：https://chatgpt.com/share/6abe1dff-8264-83ec-a1dc-8247f4819946",
        f"HTML SHA256：{hashlib.sha256(args.html.read_bytes()).hexdigest()}",
        f"唯一可见 assistant final 回答：{len(finals)}；唯一 GitHub PR/issue URL：{len(urls)}。",
        "这是共享讨论原文，不是逐项实验已完成的声明。工具/搜索结果流、内部 reasoning 不作为用户可见来源。","",
        "\n\n---\n\n".join(finals)]
    with args.output.open("x") as f:f.write("\n".join(header)+"\n")
    print(f"unique final messages={len(finals)} unique PR/issue URLs={len(urls)}")


if __name__=="__main__":main()
