#!/usr/bin/env python3
"""Generate a non-destructive inventory and synthesis of every results run."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results"
SKIP_WALK = {"compiler_dumps", "torchinductor_cache", "__pycache__"}


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def discover(run: Path) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    targets = {
        "v13_hypothesis_results.json", "v10_rq1_validation.json",
        "PERSUASIVE_RQ_RESULTS.json", "KV_REQUEST_BATCH_SUMMARY.json",
        "NATIVE_SUBGRAPH_COVERAGE.json", "REPAIR_COMPLETION.json",
        "run_summary.json", "EXECUTION_STATUS.json", "status.jsonl",
    }
    for base, dirs, files in os.walk(run):
        dirs[:] = [name for name in dirs if name not in SKIP_WALK]
        for name in files:
            if name in targets:
                found.setdefault(name, []).append(Path(base) / name)
    return found


def role(name: str) -> tuple[str, str]:
    if name == "v10_v13_640_all_frameworks_20260920_143843":
        return "canonical", "v10 strict + v13 640-case 主裁决"
    if name == "llm_shape_diversity_v2_full_20260921_200314":
        return "canonical supplement", "256 个高判别力 shape + native/batch 补充"
    if name == "persuasive_real_world_full_20260921_005344":
        return "canonical supplement", "128 个高判别力 real-world-derived case"
    if name == "kv_request_batch_128_full_20260921_114000":
        return "canonical supplement", "单/多 request 原生 KV writer"
    if name == "vllm_sglang_subgraphs_full_20260921_115500":
        return "canonical supplement", "最新 vLLM/SGLang 原生算子切片覆盖"
    if name.startswith("method_formula_audit_"):
        return "analysis-only", "同一 raw 数据的方法/公式重裁决；不是新测量"
    if name.startswith("repaired_"):
        return "repair increment", "只重跑旧结果中的非 OOM 软件失败"
    if "smoke" in name or "probe" in name or "adapter" in name or "mode_factorial" in name:
        return "diagnostic", "smoke/probe/安装或 adapter 诊断；不独立计入最终样本"
    if name.startswith("vllm_sglang_native_subgraphs_640_") or name.startswith("vllm_sglang_subgraphs_full_"):
        return "superseded development run", "原生子图 adapter 开发轮次；由 20260921_115500 汇总"
    if name == "llm_shape_diversity_v2_full_20260921_200228":
        return "superseded full", "首个 256 full；由 200314 完整版覆盖"
    if name.startswith("v13_640_all_frameworks_") or name.startswith("v13_native_full_") or name.startswith("v13_full_"):
        return "historical full", "v13 早期/中间 full；由 v10_v13 canonical 覆盖"
    if name.startswith("observations_") or name.startswith("card1_all_frameworks_"):
        return "historical observation", "早期 observation/v10 运行；用于演化审计"
    if name.startswith("tvm_native_rq_"):
        return "diagnostic", "TVM native smoke"
    return "historical/auxiliary", "保留的历史或辅助实验"


CURATED = {
    "v10_v13_640_all_frameworks_20260920_143843":
        "v10 H1.1/H1.2 未支持，H1.3/H1.4/H1.NEG 支持；v13 49 hypotheses="
        "34 supported、6 inconclusive、3 feasibility、6 blocked；runner infrastructure complete=true，"
        "但 all-framework×all-RQ×all-640 科学全覆盖=false。",
    "persuasive_real_world_full_20260921_005344":
        "128 case、8 类结构；whole 253/256 success，125 正确配对；boundary 128/128 triplet；"
        "CUDA 1821、Triton 128、TVM 2688、CUTLASS 288 success，vLLM/SGLang 各16 native rows。",
    "llm_shape_diversity_v2_full_20260921_200314":
        "256 case、8 类结构、B=1/2/4/8；whole 250 正确配对，中位 eager/Inductor=1.349×；"
        "boundary 249 triplet；CUDA 64 case；Triton extended≥3% 7/64；KV writer 2048/2048 success。",
    "kv_request_batch_128_full_20260921_114000":
        "64 attention case × 2 framework × B{1,2,4,8} × 2 slot policy=1024 行，全部成功且正确；"
        "slot-policy 达到3%的配对为21/512。",
    "repaired_non_oom_20260920_213642":
        "首轮修复：whole 旧32个非 OOM 失败全部恢复；boundary 27 case 中15个先恢复，12个待隔离补跑。",
    "repaired_boundary_completion_20260920_215519":
        "隔离补跑后 whole 136/136 success；boundary 248 success+2 OOM；旧27个非 OOM boundary 失败恢复25/27，"
        "剩余覆盖缺口来自 OOM，非 OOM bad=0。",
    "vllm_sglang_subgraphs_full_20260921_115500":
        "128 manifest case×2 framework；每框架112个 native operator slice 成功，linear-attention各16源码支持但 runtime 未验证；"
        "完整子图 E2E=0，不能冒充逐模型执行。",
}


def status_summary(found: dict[str, list[Path]]) -> str:
    bits: list[str] = []
    for path in found.get("v13_hypothesis_results.json", [])[:2]:
        data = load_json(path) or {}
        if data.get("status_counts"):
            bits.append(f"v13={data['status_counts']}")
    for path in found.get("v10_rq1_validation.json", [])[:1]:
        data = load_json(path) or {}
        states = {key: value.get("status") for key, value in data.get("hypotheses", {}).items()
                  if isinstance(value, dict) and key.startswith("H1")}
        if states:
            bits.append(f"v10={states}")
    for path in found.get("KV_REQUEST_BATCH_SUMMARY.json", [])[:1]:
        data = load_json(path) or {}
        bits.append(f"KV rows={data.get('rows')}, success/correct={data.get('successful_correct_rows')}")
    for path in found.get("PERSUASIVE_RQ_RESULTS.json", [])[:1]:
        data = load_json(path) or {}
        bits.append(f"case_count={data.get('case_count')}, failed_steps={len(data.get('failed_steps', []))}")
    for path in found.get("REPAIR_COMPLETION.json", [])[:1]:
        data = load_json(path) or {}
        bits.append(f"repair complete={data.get('complete')}, whole={data.get('whole_status')}, boundary={data.get('combined_boundary_status')}")
    for path in found.get("EXECUTION_STATUS.json", [])[:1]:
        data = load_json(path) or {}
        bits.append(f"launcher={data}")
    for path in found.get("run_summary.json", [])[:1]:
        data = load_json(path) or {}
        exits = {key: value for key, value in data.items() if key.endswith("_exit")}
        bits.append(f"exits={exits}")
    return "; ".join(bits) or "见该目录日志/报告；没有统一 machine-readable summary"


def status_log_counts(path: Path) -> Counter:
    counts: Counter = Counter()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            counts[str(row.get("status", row.get("state", "unknown")))] += 1
    except Exception:
        return Counter()
    return counts


def render(results: Path) -> str:
    runs = sorted(path for path in results.iterdir() if path.is_dir())
    lines = [
        "# layout_research：全部历史实验结果总账",
        "",
        "## 1. 汇总原则",
        "",
        "本文盘点 `results/` 下全部历史实验目录。目录被保留并逐项列出，但只有 canonical 与明确的 canonical supplement 用于最终科学结论；smoke、probe、修复前结果、重复 full 和 analysis-only 重裁决不能重复累计为独立样本。OOM、unsupported、blocked 和 numerical mismatch 继续 fail-closed。",
        "",
        "最终结论应联合引用三类互补证据：640-case coverage、128/256 causal/discriminative supplement、以及 native batch/subgraph/repair audit。",
        "",
        "## 2. 最终采用的实验结果",
        "",
        "### 2.1 v10/v13 640-case canonical",
        "",
        "- v10 strict：H1.1、H1.2 未通过；H1.3、H1.4、H1.NEG 通过。严格 H1.1 不能写成已普遍证明。",
        "- v13：49 个 hypothesis 中 34 supported、6 inconclusive、3 feasibility_supported、5 blocked_single_gpu、1 blocked_requires_multiple_hardware。",
        "- canonical raw：CUDA 13,680 行/240 attention case；Triton 960 行/240 case；TVM 13,440 行/640 manifest→120 projected shape；boundary 1,802 行；vLLM/SGLang 各16 serving 行。",
        "- 640 whole-subgraph 状态为 success=1179、OOM preflight=64、numerical mismatch=35、OOM runtime=2；boundary 为 success=1743、OOM preflight=32、error=27。后续 repair 实验单独修复非 OOM 软件失败。",
        "- v13 runner infrastructure complete=true，但顶层 scientific all-framework×all-RQ×all-640 claim=false：RQ5 单卡阻塞、H4.4 缺第二硬件、Hexcute 在 A10 architecture-blocked，且部分 matrix cell 只能 projected/source-only/not-applicable。",
        "",
        "### 2.2 128 与 256 个高判别力子图",
        "",
        "- 128-case：8 类子图；whole 253/256 成功、125 个正确配对；boundary 128/128 triplet；保留3个数值反例。",
        "- 256-case：8 类×32、prefill/decode 各128、B=1/2/4/8；whole 250 个正确配对，中位 eager/Inductor=1.349×；boundary 249 个完整 triplet。",
        "- 256-case CUDA NHD/HND consumer inversion=47/64；Triton extended layout ≥3% winner=7/64；TVM consumer inversion=185/256、reuse crossover=40/256。",
        "- 这些 shape 来自 pinned real-model architecture dimensions，但 sequence/batch/page 边界是受控反事实；不等于加载了256个 checkpoint。",
        "",
        "### 2.3 vLLM/SGLang 单 request、多 request 与原生算子",
        "",
        "- 独立 64-attention-case KV writer：1024/1024 行成功正确；slot policy 只有21/512配对达到3%，多数点布局不敏感。",
        "- 256 supplement 内的扩展 KV writer：2048/2048 行成功正确，显式覆盖 B=1/2/4/8。",
        "- 最新原生子图切片：每框架112/128 case 成功；linear-attention 16 case/框架仍为源码支持但 runtime 未验证；完整子图 E2E=0。",
        "- native serving 使用固定 Qwen2.5-0.5B checkpoint，不能外推为全部模型架构均已在引擎中执行。",
        "",
        "### 2.4 修复实验",
        "",
        "- MoE whole graph 的32条旧非 OOM 失败已全部恢复，最终136/136 success。",
        "- boundary 最终248 success+2 OOM；非 OOM bad=0。OOM 保留为资源边界，不伪装修复。",
        "- 3个长 scan/cumsum 数值反例继续被排除，没有通过放宽容差改写为成功。",
        "",
        "## 3. 每个历史目录的结果与角色",
        "",
        "| 历史目录 | 角色 | 实验含义 | 关键结果/状态 | 是否独立计入最终结论 |",
        "|---|---|---|---|---|",
    ]
    for run in runs:
        kind, meaning = role(run.name)
        found = discover(run)
        summary = CURATED.get(run.name, status_summary(found))
        status_files = found.get("status.jsonl", [])
        if status_files and summary.startswith("见该目录"):
            counts = status_log_counts(status_files[0])
            if counts:
                summary = f"status.jsonl={dict(counts)}"
        independent = {
            "canonical": "是",
            "canonical supplement": "是，作为互补证据",
            "repair increment": "只计修复后的恢复证据",
            "analysis-only": "否；同一 raw 重分析",
        }.get(kind, "否；用于诊断/演化审计")
        lines.append(f"| `{run.name}` | {kind} | {meaning} | {summary} | {independent} |")

    lines += [
        "",
        "## 4. 合并后的科学结论",
        "",
        "1. 局部 layout winner 不能稳定推出跨 edge、跨 consumer 或 pipeline winner；但严格 v10 H1.1 没有通过预注册 core gate，结论应主要表述为 v13 广覆盖 observation。",
        "2. conversion 的价值取决于 reuse、consumer access、stride legality、temporary memory 与 overlap；zero-copy 和 materialize 均存在有效正反例。",
        "3. 静态固定 layout/page 在部分集合有 regret，但 vLLM/SGLang aggregate fixed-policy regret 较小；不能把 microkernel 最坏 case speedup 写成完整 serving 加速。",
        "4. data 与 metadata 存在可测 interaction，但小而 cache-resident 的 metadata 是重要负对照，不能无条件扩大搜索空间。",
        "5. extended layout 空间只在少数 geometry/phase 有明显收益；多数 shape 可关闭扩展空间而无 oracle regret，因此科学问题是候选空间的条件化开启。",
        "6. RQ5 分布式与 H4.4 跨硬件仍未验证；A10 上 Hexcute architecture-blocked，不能用 CUDA reference 冒充。",
        "",
        "## 5. 与方法报告的关系",
        "",
        "- 本文件回答“以前所有运行分别得到了什么、哪些被最终采用”。",
        "- `ALL_RQ_EXPERIMENT_METHODS_AND_RESULTS_CN.md` 回答“每个 RQ 的实验如何计时、公式的分子分母是什么”。",
        "- `ALL_RESULTS_V10_V13_RQ_EXAMPLE_LEDGER_AUDITED_20260927_V2.csv` 提供逐 case 正例、反例和原始 artifact 路径。",
        "",
        f"盘点目录数：**{len(runs)}**。生成器：`{HERE / 'summarize_all_historical_experiments.py'}`。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path,
                        default=DEFAULT_RESULTS / "ALL_HISTORICAL_EXPERIMENTS_SUMMARY_CN.md")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing report: {args.output}")
    text = render(args.results.resolve())
    args.output.write_text(text, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "runs": text.count("| `")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
