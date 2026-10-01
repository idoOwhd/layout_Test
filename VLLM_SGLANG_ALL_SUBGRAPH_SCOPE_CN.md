# vLLM / SGLang “所有子图验证”范围说明

## KV cache 的 request batch 维度

原 640 与 128 manifest 的 `shape.batch` 均为 1，因此旧结果不能外推为多
request 证据。`run_kv_request_batch_gpu1.sh` 现在对每个 attention case 保持
架构维度和单请求长度不变，派生 batch=1/2/4/8，并测量 request-major 连续
slot 与 16-token block 交错 slot。所有 batch>1 行都标记为 derived
counterfactual。`run_persuasive_real_world_rqs_gpu1.sh` 还在固定 Qwen 模型的
native engine 中配对 concurrency=1 与 concurrency>1，以覆盖调度器和完整
attention read/write；两种证据分开报告，互不冒充。

## 严格结论

不能把固定 Qwen2.5 模型的 native serving workload 当成全部 LLM 子图验证。
它只覆盖该 checkpoint 实际包含的结构。当前补充实现采用两层证据：

1. 固定模型 native engine：保留原有 vLLM/SGLang 的 NHD/HND、page/block
   等端到端 serving 对照。
2. manifest 逐 case 原生算子：对每个 case 调用框架自带的 KV writer、
   SwiGLU、MoE router 或 Mamba state kernel，并记录 shape、stride、选择的
   layout、耗时、数值检查和失败原因。

第二层是 `native_operator_slice`，不是完整模型子图 E2E。只有加载该 case
对应的模型架构/checkpoint，才能称为完整子图 E2E；一些模型超出 A10 显存。

## 已实现覆盖

- 128 个判别性 real-world case：vLLM/SGLang 共 256 个框架×case 组合全部
  有 fail-closed 记录，无静默缺失。
- 其中 224 个组合完成原生算子执行和数值检查：两框架的 GQA、滑窗注意力、
  稀疏注意力、MLA、SwiGLU、MoE、Mamba2 各 16 个 case。
- 32 个线性注意力组合（16×2）保留为源码支持但 A10 runtime 未验证；FLA
  JIT 探针在 sm_86 上没有干净结束，因此不能标成功。
- vLLM 的直接 4D cache writer 只按受支持的 NHD contract 执行。HND 属于
  engine-owned 5D cache contract，由固定模型 native engine 实验验证；不能
  用人工 4D stride 冒充。
- 分布式 RQ 在单卡1上明确不适用。

通过的最终结果：

`results/vllm_sglang_subgraphs_full_20260921_115500`

KV request batch=1/2/4/8 的完整 128 attention 结果：

`results/kv_request_batch_128_full_20260921_114000`

64 个 attention case × 4 个 batch × 2 个 slot policy × 2 个框架，共 1024
行全部执行成功并通过数值校验。512 组 slot-policy 配对中，仅 21 组达到 3%
差异阈值，且总体中位差异接近 0；这说明仅看 KV writer 不足以证明某种
slot policy 优越，必须结合 attention reader 与 serving scheduler 的配对结果。

固定模型原生 engine 的 c1/c3 smoke：

`results/kv_request_batch_serving_smoke_20260921_113000`

同样三个 128-token 请求从串行执行改为同时 batch 后，vLLM LBNHC 的
wall-time 方向性加速为 1.795×，SGLang NHD 为 2.942×（仅 2 次重复，属于
smoke，不作统计显著性结论）。

原始 640 case 的最终结果：

`results/vllm_sglang_native_subgraphs_640_20260921_102126`

640 case 共 1280 个框架×case 组合，1272 个完成原生算子执行和数值检查；
剩余 8 个是两个框架各 4 个 linear-attention case，均为源码支持但 A10
runtime 未验证。无静默缺失、无其他意外失败。Mamba prefill/decode 分进程，
避免框架 JIT/原生全局状态跨 phase 污染。

## 独立复现 128 case

```bash
cd /home/liangyilei/ladder_home

OUT=/home/liangyilei/ladder_home/staged/baseline_framework/layout_research/results/vllm_sglang_subgraphs_full_$(date +%Y%m%d_%H%M%S)

GPU_PHYSICAL_INDEX=1 \
PYTHON_BIN=/home/liangyilei/conda/envs/cuda-opt/bin/python \
VLLM_PYTHON=/home/liangyilei/ladder_home/staged/baseline_framework/layout_research/framework_envs/layout-vllm/bin/python \
SGLANG_PYTHON=/home/liangyilei/ladder_home/staged/baseline_framework/layout_research/framework_envs/layout-sglang/bin/python \
NATIVE_SUBGRAPH_WARMUP=3 \
NATIVE_SUBGRAPH_ITERATIONS=10 \
NATIVE_SUBGRAPH_STRUCTURE_TIMEOUT=1800 \
bash staged/baseline_framework/layout_research/run_vllm_sglang_subgraph_completion_gpu1.sh \
  --full \
  --manifest staged/baseline_framework/layout_research/results/persuasive_real_world_full_20260921_005344/cases/persuasive_real_world_128.json \
  --output "$OUT"
```

## 640 case 全局命令

`run_v13_640_all_frameworks_gpu1.sh` 与
`run_v10_v13_640_all_frameworks_gpu1.sh` 已接入同一逐结构、逐框架隔离 runner。
新运行会在其新结果目录中生成：

`vllm_sglang_native_subgraphs_640/VLLM_SGLANG_SUBGRAPH_COMPLETION_REPORT_CN.md`

原有结果目录不会被覆盖；所有入口拒绝已存在的 `--output`。

## 独立复现 KV request batch

```bash
cd /home/liangyilei/ladder_home

OUT=/home/liangyilei/ladder_home/staged/baseline_framework/layout_research/results/kv_request_batch_640_$(date +%Y%m%d_%H%M%S)

GPU_PHYSICAL_INDEX=1 \
PYTHON_BIN=/home/liangyilei/conda/envs/cuda-opt/bin/python \
VLLM_PYTHON=/home/liangyilei/ladder_home/staged/baseline_framework/layout_research/framework_envs/layout-vllm/bin/python \
SGLANG_PYTHON=/home/liangyilei/ladder_home/staged/baseline_framework/layout_research/framework_envs/layout-sglang/bin/python \
KV_REQUEST_BATCHES=1,2,4,8 \
KV_BATCH_WARMUP=3 \
KV_BATCH_ITERATIONS=10 \
KV_BATCH_TIMEOUT=28800 \
bash staged/baseline_framework/layout_research/run_kv_request_batch_gpu1.sh \
  --full \
  --manifest staged/baseline_framework/real_world_shapes/real_world_shape_manifest.json \
  --output "$OUT"
```

完整 v10/v13 × 640 × 全框架入口也已包含此实验，并把 KV batch 的实际行数、
成功数和数值正确性纳入 `V13_640_RUN_COMPLETENESS.json`；不完整时总入口返回
非零退出码。
