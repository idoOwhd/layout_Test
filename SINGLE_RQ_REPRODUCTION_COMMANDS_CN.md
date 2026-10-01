# 单个 RQ 的全量复现命令

## v13：L-RQ1～L-RQ10

把 `RQ` 改成目标编号。`--full` 会先生成 1024 个 real-world-derived
contracts，再按 `case.target_rqs` 选择该 RQ 的全部适用 case；每次输出目录都带时间戳，
不会覆盖旧结果。

```bash
cd /home/liangyilei/ladder_home

RQ=L-RQ8
OUT=/home/liangyilei/ladder_home/staged/baseline_framework/layout_research/results/single_${RQ}_full_$(date +%Y%m%d_%H%M%S)

GPU_PHYSICAL_INDEX=1 \
FRAMEWORK_ENV_FILE=/home/liangyilei/ladder_home/staged/baseline_framework/layout_research/framework_envs.generated.sh \
SERVING_MODEL_PATH=/home/liangyilei/ladder_home/staged/baseline_framework/layout_research/models/Qwen2.5-0.5B-Instruct-7ae5576 \
CUTLASS_DIR=/home/liangyilei/cursor_home/Ladder/3rdparty/cutlass \
CUDA_PROCESS_REPETITIONS=5 \
PERSUASIVE_WARMUP=3 \
PERSUASIVE_ITERATIONS=10 \
CUDA_WARMUP=8 \
CUDA_ITERATIONS=30 \
NATIVE_SUBGRAPH_TIMEOUT=28800 \
KV_BATCH_TIMEOUT=28800 \
REAL_WORLD_CASE_TIMEOUT=14400 \
SERVING_STARTUP_TIMEOUT=3600 \
SERVING_REQUEST_TIMEOUT=3600 \
bash staged/baseline_framework/layout_research/run_rq_adequacy_gpu1.sh \
  --full --rq "$RQ" --output "$OUT"
```

先做小规模连通性检查时，把 `--full` 改为 `--smoke`，并使用另一个新 `OUT`。

预期 full case 数如下；这是“适用全集”，不是所有 RQ 都机械地跑 1024 个：

| RQ | selected contracts | attention/KV contracts | 单卡状态 |
|---|---:|---:|---|
| L-RQ1 | 1024 | 256 | 可运行 |
| L-RQ2 | 1024 | 256 | 可运行 |
| L-RQ3 | 1024 | 256 | 可运行 |
| L-RQ4 | 1024 | 256 | 可运行 |
| L-RQ5 | 0 | 0 | blocked；需要至少 2 GPU/rank |
| L-RQ6 | 768 | 256 | 可运行 |
| L-RQ7 | 1024 | 256 | 可运行 |
| L-RQ8 | 640 | 256 | 可运行 |
| L-RQ9 | 1024 | 256 | 可运行 |
| L-RQ10 | 512 | 256 | 可运行 |

每个 adapter 的状态会写入 `status.jsonl`。对该 RQ 没有可证伪干预的 adapter
记录为 `skipped_by_rq`；环境缺失记录为 `unavailable`；二者都不能当成反例。
case 选择保存在 `cases/SINGLE_RQ_SELECTION.json`，运行清单保存在
`SINGLE_RQ_REPRODUCTION_CN.md`。

## v10

当前 `layout_summary_v10_*` 的可执行 protocol 只定义了严格的 L-RQ1；使用专用的
preregistered runner，而不是 v13 adequacy runner：

```bash
cd /home/liangyilei/ladder_home

OUT=/home/liangyilei/ladder_home/staged/baseline_framework/layout_research/results/v10_rq1_single_$(date +%Y%m%d_%H%M%S)

GPU_PHYSICAL_INDEX=1 \
PYTHON_BIN=/home/liangyilei/conda/envs/cuda-opt/bin/python \
TORCH_PYTHON=/home/liangyilei/conda/envs/cuda-opt/bin/python \
FRAMEWORK_ENV_FILE=/home/liangyilei/ladder_home/staged/baseline_framework/layout_research/framework_envs.generated.sh \
SERVING_MODEL_PATH=/home/liangyilei/ladder_home/staged/baseline_framework/layout_research/models/Qwen2.5-0.5B-Instruct-7ae5576 \
V10_PROCESS_REPETITIONS=5 \
V10_WARMUP=8 \
V10_ITERATIONS=30 \
bash staged/baseline_framework/layout_research/run_v10_rq1_gpu1.sh \
  --full --output "$OUT"
```

v10 Stage 4/5 受 Stage 3 的预注册统计 gate 控制；若 gate 为 STOP，后续未运行是协议
要求，不是脚本失败。
