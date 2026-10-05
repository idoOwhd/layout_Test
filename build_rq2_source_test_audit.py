#!/usr/bin/env python3
"""Offline, fail-closed inventory of the complete RQ2 discussion's sources.

Mentions of a dtype, a test filename, a speedup, or a layout do NOT certify
FP16 support, execution on A10, or an RQ2 causal performance experiment.
This tool never imports frameworks, executes downloaded code, or writes results.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent

# Manually checked restrictions. Unlisted sources remain pending, not successes.
# Each entry is scoped to its upstream reproducer, not to every path in a repo.
RESTRICTIONS = {
    ("flashinfer-ai/flashinfer", 5405): ("outside_fp16_upstream", "原 fused RMSNorm/RoPE/KV-write 实现声明 BF16/FP8、SM90/SM100；不能换 dtype 后仍称原 PR 复现。"),
    ("flashinfer-ai/flashinfer", 4572): ("outside_fp16_upstream", "GDN cache-step stride 复现为 FP32/BF16、SM100/SM103；原 PR 主要是正确性修复。"),
    ("flashinfer-ai/flashinfer", 5757): ("outside_fp16_upstream", "NVFP4 KV 的 FP8 scale stride；不是 FP16 KV scale 性能结论。"),
    ("sgl-project/sglang", 41944): ("outside_llm_cuda_fp16_upstream", "XPU FLUX.2 DiT、BF16、Arc Pro B70；是机制启发，不是本轮 CUDA FP16 LLM 原生实验。"),
    ("sgl-project/sglang", 34299): ("outside_fp16_upstream", "CakeKDA 示例 BF16 导出/FP32 recurrent state，原大模型多卡性能不可视为 FP16 单卡结果。"),
    ("sgl-project/sglang", 40326): ("fp16_contract_tests_runtime_pending", "1/7 stride-hygiene 栈；有 FP16 pool/CPU contract 测试，但不是整图 domain 经济性实验。"),
    ("sgl-project/sglang", 39071): ("blocked_platform_upstream", "ROCm stride-aware QK norm；A10 上的对应机制移植必须单列 analogue。"),
    ("sgl-project/sglang", 41848): ("blocked_platform_upstream", "AMD/AITER 路由输入切片；原生运行不能用 NVIDIA 对照代替。"),
    ("sgl-project/sglang", 41875): ("blocked_platform_upstream", "NPU 启动时 DSA KV layout 能力选择。"),
    ("sgl-project/sglang", 39989): ("blocked_platform_multigpu_upstream", "XPU diffusion packed Ulysses A2A；当前只用卡1不能执行原分布式实验。"),
    ("sgl-project/sglang", 40588): ("outside_llm_multigpu_upstream", "Diffusion USP strided-row A2A；可启发 LLM producer-stride 测试，不是原 LLM 单卡实验。"),
    ("sgl-project/sglang", 38092): ("outside_fp16_upstream", "NVFP4 Marlin fallback 的死 scale 分配；~3.5GiB 内存收益不等于 FP16 layout 加速。"),
    ("sgl-project/sglang", 38771): ("outside_fp16_upstream", "NVFP4 MoE linear scale 消费；量化元数据机制须与 FP16 激活/整数路由元数据对照分开。"),
    ("sgl-project/sglang", 39977): ("outside_fp16_upstream", "DeepGEMM FP8 scale 所有权/生命周期正确性修复。"),
    ("sgl-project/sglang", 40682): ("outside_fp16_upstream", "序列化 CUTLASS NVFP4 scale 转换延后到加载期，不是 FP16 动态图收益。"),
    ("vllm-project/vllm", 59112): ("fp16_contract_tests_runtime_pending", "新增 FP16 测试是 sliding-window manager/cache 几何合同；新增 GPU attention parity/replay 是 BF16。PR 的 B300 BF16/FP8 DFlash 吞吐数据不是本轮 FP16/A10 测量。"),
    ("vllm-project/vllm", 58798): ("fp16_contract_tests_runtime_pending", "MiMo partial RoPE+value scale+DiffKV write 正确性显式参数化 FP16，检查 whole-allocation guards；其整段 prep+attention benchmark 固定 BF16、排除 projection/metadata/serving，需新增 FP16 整图性能实验。"),
    ("vllm-project/vllm", 57823): ("fp16_contract_extension_pending", "新增 padded-router-row regression 固定 BF16、33 tokens、60→64/64→72 experts；普通测试提及 FP16 不等于这个 strided case 已覆盖 FP16，需补 dtype 参数化。"),
    ("vllm-project/vllm", 52240): ("outside_fp16_upstream", "MXFP4 MoE W13 顺序；可另做 FP16 SwiGLU 双投影顺序对照，不冒充原量化 PR。"),
    ("vllm-project/vllm", 50336): ("outside_fp16_upstream", "NVFP4 KV scale-layout validation 是量化契约测试。"),
    ("vllm-project/vllm", 51942): ("outside_fp16_multigpu_upstream", "packed FP8 + all-reduce RMSNorm；卡1不能验证原 all-reduce 收益。"),
    ("vllm-project/vllm", 51947): ("outside_fp16_upstream", "MTP packed FP8 logits 输入复用；另设计 FP16 输入复用对照。"),
    ("triton-lang/triton", 11117): ("outside_pure_fp16_reproducer", "原 chained-division→reduce 使用 FP32；4128→64 是 PTX div 指令数，257s→4s 是编译耗时，不是 GPU 延迟。"),
    ("triton-lang/triton", 10987): ("outside_pure_fp16_reproducer", "原 issue 是 FP32 division/reduction compiler reproducer；FP16 输入、FP32 累加版本须说明混合类型。"),
    ("triton-lang/triton", 11538): ("outside_pure_fp16_reproducer", "原 tf32x3 路径为 FP32 dot，不能直接推断 FP16 dot 存在相同 shared-memory 浪费。"),
    ("triton-lang/triton", 11526): ("outside_pure_fp16_reproducer", "tf32x3 shared layout 路径；FP16 需重新审计实际生成 IR。"),
    ("triton-lang/triton", 8450): ("blocked_platform_upstream", "AMD direct GMEM→REG dot-operand 草案；A10 的类比不等于原生实现复现。"),
    ("triton-lang/triton", 11685): ("compiler_contract_runtime_pending", "Gluon 自动 layout inference 的 lit/MLIR 测试；编译成功不是整图加速。"),
    ("tile-ai/tilelang", 3176): ("outside_pure_fp16_reproducer", "原 H100 FP32 pooling reproducer；cost heuristic 未校准，register count 非峰值存活寄存器。"),
    ("tile-ai/tilelang", 3180): ("compiler_contract_runtime_pending", "connected-component fragment bijection/ownership correctness；需另测 FP16 kernel 与整图。"),
    ("tile-ai/tilelang", 3255): ("outside_fp16_blocked_platform_upstream", "ROCm GLM-5.3 FP8 key、FP32 scale 路径。"),
    ("tile-ai/tilelang", 3283): ("outside_fp16_blocked_architecture_upstream", "SM120 NVFP4 SFA/SFB scale fragment 约束，不能视为 FP16 必须拆分的证据。"),
    ("tile-ai/tilelang", 3284): ("outside_fp16_blocked_architecture_upstream", "SM120 NVFP4 scale fragment layout；FP16 analogue 需重新证明合法约束。"),
    ("tile-ai/tilelang", 3320): ("blocked_platform_upstream", "Ascend VF/shared.dyn 生命周期变换；CUDA 卡1不支持原设备指令。"),
    ("tile-ai/tilelang", 3328): ("blocked_platform_upstream", "Ascend L0C→UB FixPipe；非 A10 原生路径。"),
    ("NVIDIA/cutlass", 3030): ("blocked_architecture_upstream", "原实现要求 SM120；SM80 FP16 替代实验应另标 analogue。"),
    ("NVIDIA/cutlass", 3453): ("blocked_architecture_upstream", "Blackwell kernel/layout 路径；卡1 A10 不能执行其原指令契约。"),
    ("NVIDIA/cutlass", 3256): ("blocked_architecture_upstream", "SM90 grouped SwiGLU aux/TMA；SM80 FP16 epilogue analogue 不等于原 PR。"),
    ("NVIDIA/cutlass", 3273): ("outside_fp16_blocked_architecture_upstream", "SM120 NVFP4 路径。"),
    ("apache/tvm", 17599): ("blocked_platform_upstream", "Adreno texture-scope pass；CUDA generic Relax 对照另列。"),
    ("apache/tvm", 18523): ("blocked_platform_upstream", "Adreno texture lowering，虽有 FP16 packing 仍非 A10 CUDA texture 契约。"),
    ("apache/tvm", 20426): ("blocked_platform_upstream", "WebGPU path；CUDA 上只能测等价传播机制，不能叫原 kernel 复现。"),
    ("apache/tvm", 20329): ("blocked_architecture_upstream", "SM100 path，A10 不具备对应指令。"),
    ("deepseek-ai/DeepGEMM", 440): ("outside_fp16_multigpu_upstream", "SM90 FP8 MegaMoE/NVSHMEM ring overlap；单 GPU FP16 流水不能证明原网络收益。"),
    ("NVIDIA/TensorRT-LLM", 19541): ("outside_fp16_blocked_architecture_upstream", "Rubin locality-domain/MXFP8；不是 A10 FP16 baseline。"),
    ("NVIDIA/TensorRT-LLM", 19115): ("outside_fp16_blocked_architecture_upstream", "Rubin/MXFP8 locality 路径；原生硬件实验待跨平台。"),
    ("NVIDIA/TensorRT-LLM", 18763): ("outside_fp16_blocked_architecture_upstream", "Rubin locality 域与 MXFP8 tactic；FP16 类比必须单列。"),
}

MOTIFS = {
    "persistent_paged_kv": ("paged", "kv cache", "kv-cache", "block table", "block_table", "packed kv"),
    "cross_layer_hybrid_state": ("cross-layer", "cross layer", "mamba", "gdn", "kda", "recurrent", "hybrid"),
    "projection_attention": ("qkv", "q/k/v", "q/k", "qk norm", "qk_norm", "rope"),
    "sparse_data_and_metadata": ("sparse", "top-k", "topk", "top_k", "indexer", "indices", "scale"),
    "moe_dispatch_expert_combine": ("moe", "expert", "dispatch", "combine", "swiglu", "w13", "w31"),
    "host_transfer_offload": ("offload", "host", "hicache", "l3", "nixl", "transfer"),
    "compiler_propagation": ("layout inference", "layout propagation", "convertlayout", "fragment", "reshape", "transpose", "encoding"),
    "memory_hierarchy": ("shared memory", "shared-memory", "smem", "tma", "register", "gmem", "tmem", "swizzle"),
    "lifetime_and_aliasing": ("stride", "sliced", "slice", "ownership", "lifetime", "in-place", "alias", "contiguous"),
}


def dtype_hints(text):
    rules = {"FP16": r"\b(?:float16|fp16|f16|half_t|__half)\b|torch\.half\b",
             "BF16": r"\b(?:bfloat16|bf16)\b", "FP32/TF32": r"\b(?:float32|fp32|tf32|tf32x3)\b",
             "FP8": r"\b(?:fp8|float8|mxfp8)\b", "FP4": r"\b(?:fp4|nvfp4|mxfp4)\b"}
    return [name for name, pattern in rules.items() if re.search(pattern, text, re.I)]


def patches_from_diff(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    parts = re.split(r"(?=^diff --git a/)", text, flags=re.M)
    result = []
    for part in parts:
        m = re.search(r"^diff --git a/(.*?) b/", part, re.M)
        if m: result.append({"filename": m[1], "patch": part})
    return result


def restriction(repo, number):
    if (repo, number) in RESTRICTIONS:
        return RESTRICTIONS[repo, number]
    if repo.lower() == "rocm/aiter":
        return "blocked_platform_upstream", "AMD/ROCm 原生项目；卡1 CUDA 移植须单列 analogue，原 dtype 另核实。"
    return "pending_fp16_contract_review", "源码已归档；dtype、目标设备和完整图合同还需逐路径审查，不能自动记为验证成功。"


def collect(directory, catalog):
    metadata = json.loads((directory/"connector_metadata.json").read_text())["records"]
    extras = json.loads((directory/"connector_unlinked_metadata.json").read_text())["records"]
    raw_audit = json.loads((directory/"source_audit.json").read_text())["records"]
    diffs = {(r["repo"].lower(), r["number"]): r for r in raw_audit}
    patch_map = {}
    for name in ("connector_missing_patches.json", "connector_unlinked_patches.json"):
        for row in json.loads((directory/name).read_text())["records"]:
            patch_map[row["repo"].lower(), row["number"]] = row
    contexts = {(r["repo"].lower(),r["number"]):r["occurrences"] for r in catalog["records"]}
    result=[]; seen=set()
    for meta in metadata+extras:
        key=(meta["repo"].lower(),meta["number"])
        if key in seen: raise ValueError(f"duplicate primary source: {key}")
        seen.add(key)
        row=dict(meta);occurrences=contexts.get(key, meta.get("occurrences", []))
        shared="\n".join(c.get("context", c.get("line", "")) for c in occurrences)
        original=diffs.get(key, {})
        artifact=original.get("diff_artifact")
        artifact_path=None
        if artifact:
            # Older source-audit entries use paths relative to layout_research,
            # not the caller's cwd. Never silently lose those real patches.
            options=[Path(artifact), HERE/Path(artifact), directory/Path(artifact).name]
            artifact_path=next((p for p in options if p.is_file()),None)
            if artifact_path is None:raise FileNotFoundError(f"stored diff missing: {artifact}")
        if artifact_path:
            patches=patches_from_diff(artifact_path);patch_origin=str(artifact_path.resolve())
        else:
            patch_record=patch_map.get(key, {})
            patches=patch_record.get("patches") or []
            patch_origin="connector PR-file patches" if patches else "not available / issue"
        tests=[]
        for patch in patches:
            name=patch["filename"];content=patch.get("patch") or ""
            if re.search(r"test|bench|repro|example",name,re.I):
                tests.append({"path":name,"test_definitions":sorted(set(re.findall(
                    r"^[ +]?\s*(?:async\s+)?def\s+(test_\w+)\s*\(",content,re.M))),
                    "dtype_mentions":dtype_hints(content),
                    "patch_available":bool(content),
                    "head_blob_url": f"https://github.com/{meta.get('head_repo') or meta['repo']}/blob/{meta.get('head_sha')}/{name}" if meta.get("head_sha") else None})
        text=meta.get("title", "")+"\n"+meta.get("body", "")+"\n"+"\n".join(p.get("patch") or "" for p in patches)
        hints=dtype_hints(text)
        status,note=restriction(meta["repo"],meta["number"])
        row.update(shared_occurrences=occurrences,
                   reference_kind="explicit_link" if key in contexts else "bare_number_verified_by_primary_endpoint",
                   source_patch_origin=patch_origin,changed_files=[p["filename"] for p in patches],
                   changed_test_benchmark_example_files=tests,
                   native_dtype_mentions_not_support_proof=hints,
                   extracted_upstream_command_lines=[line.strip() for line in meta.get("body", "").splitlines()
                       if re.match(r"^\s*(?:\$\s*)?(?:python(?:3)?\s|pytest\s|(?:uv\s+run\s+)?(?:bash|ctest)\s)",line)],
                   patch_sha256=hashlib.sha256(json.dumps(patches, sort_keys=True).encode()).hexdigest() if patches else None,
                   missing_per_file_patch_count=sum(not p.get("patch") for p in patches),
                   contract_audit_status=status,contract_audit_note=note,
                   candidate_graph_motifs_not_contract_proof=[m for m,words in MOTIFS.items() if any(w in (text+"\n"+shared).lower() for w in words)],
                   rq2_whole_graph_native_replay_status="not_run",
                   missing_test_file_warning="无变更测试文件不代表上游不存在原测试；仍需查 pinned revision 的原文件。" if not tests else None)
        result.append(row)
    return sorted(result,key=lambda r:(r["repo"].lower(),r["number"]))


def emit(directory, rows, source):
    directory.mkdir(parents=True, exist_ok=False)
    counts=dict(Counter(r["repo"] for r in rows))
    summary={"source":source,"unique_sources":len(rows),"repo_counts":counts,
             "reference_counts":dict(Counter(r["reference_kind"] for r in rows)),
             "primary_metadata_counts":dict(Counter(r["source_status"] for r in rows)),
             "pr_state_counts":dict(Counter("merged" if r.get("merged") else r.get("state", "unknown")+"_unmerged"
                                             for r in rows if r["kind"]=="pull")),
             "contract_audit_counts":dict(Counter(r["contract_audit_status"] for r in rows)),
             "native_rq2_whole_graph_replays":0,"all_native_experiments_complete":False,
             "scope":"source inventory and experiment-design audit only, no new performance results"}
    with (directory/"RQ2_ALL_SOURCE_TEST_AUDIT.json").open("x",encoding="utf-8") as f:
        json.dump({**summary,"records":rows},f,ensure_ascii=False,indent=2);f.write("\n")
    fields=["repo","number","kind","url","title","state","merged","base_sha","head_sha",
            "reference_kind","contract_audit_status","contract_audit_note","dtype_mentions",
            "changed_test_files","upstream_commands","graph_motif_candidates","native_replay"]
    with (directory/"RQ2_ALL_SOURCE_TEST_AUDIT.csv").open("x",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            csvrow={k:row.get(k, "") for k in fields}
            csvrow.update(dtype_mentions=";".join(row["native_dtype_mentions_not_support_proof"]),
                changed_test_files=";".join(t["path"] for t in row["changed_test_benchmark_example_files"]),
                upstream_commands=";".join(row["extracted_upstream_command_lines"]),
                graph_motif_candidates=";".join(row["candidate_graph_motifs_not_contract_proof"]),native_replay="not_run")
            writer.writerow(csvrow)
    def safe(value):return str(value).replace("|","\\|").replace("\n"," ")
    lines=["# RQ2 完整共享讨论：全部去重源码与原测试审计", "",
        f"来源：{source}", f"去重后 {len(rows)} 项；仓库数 {len(counts)}；{summary['reference_counts']}。", "",
        "这是源码/测试入口清单，不是实验通过清单。所有新增 RQ2 整图原生复现均未运行。",
        "dtype mentions 仅表示 PR 描述或变更代码出现该类型，**不能自动证明目标 kernel 支持它**。",
        "base/head SHA 是 API 读取时快照；正式实验须从该 SHA 读取完整文件，核对 patch 后再执行；本次未执行下载代码。",
        "无变更测试文件不代表上游无测试；名字中有 test/bench/example 也不证明其可在 A10/FP16 运行。", "",
        "| 仓库 | 唯一来源数 |", "|---|---|"]
    lines += [f"| {repo} | {count} |" for repo,count in sorted(counts.items())]
    lines += ["", "## 逐来源原始测试、限制与未完成项", "",
        "| PR/issue（真实标题） | 状态 / head | 原测试/benchmark/example 变更入口 | dtype 提及（非支持证明） | 合同审计 |",
        "|---|---|---|---|---|"]
    for row in rows:
        tests="<br>".join(f"`{t['path']}`" for t in row["changed_test_benchmark_example_files"]) or "变更中未发现；须查原 revision"
        state="merged" if row.get("merged") else row["state"]+("/unmerged" if row["kind"]=="pull" else "/issue")
        lines.append(f"| [{safe(row['repo'])} #{row['number']}: {safe(row['title'])}]({row['url']}) | {state}; `{(row.get('head_sha') or 'n/a')[:12]}` | {tests} | {', '.join(row['native_dtype_mentions_not_support_proof']) or '未提及'} | `{row['contract_audit_status']}`<br>{safe(row['contract_audit_note'])} |")
    lines += ["", "## 防止错误归因", "",
        "- PR/issue 按大小写归一的 repo+number 去重；同一 PR 的多次引用/issue 路径不增加样本数。",
        "- 207 个来源不是 207 个独立机制，也不是 207 个性能正例；堆叠 PR 的依赖与归因须单独 ablation。",
        "- 编译耗时、PTX 指令数、显存、kernel 延迟、端到端吞吐分别记录，不互换 speedup 的分母。",
        "- native 原实现、FP16 mechanism analogue、legacy supporting probe 分开计数。",
        "- AMD/XPU/NPU、TMA/TMEM、新架构或多卡依赖明确标 blocked，不用 CUDA reference 填成功。",
        "- 所有 pending 条目在正式 full run 前必须完成逐路径 contract review；仅凭关键词匹配不得补成功。"]
    with (directory/"RQ2_ALL_SOURCE_TEST_AUDIT_CN.md").open("x",encoding="utf-8") as f:f.write("\n".join(lines)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument("--source-dir",type=Path,required=True)
    p.add_argument("--catalog",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args();catalog=json.loads(args.catalog.read_text())
    print(json.dumps(emit(args.output_dir,collect(args.source_dir,catalog),catalog["source"]),ensure_ascii=False))


if __name__ == "__main__":main()
