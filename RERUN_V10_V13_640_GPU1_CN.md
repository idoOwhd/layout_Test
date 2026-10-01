# v10/v13 × 640 LLM 子图 × 六框架：卡1重跑说明

## 结论边界

- `layout_summary_v10_*` 在当前目录中明确只定义 **L-RQ1** 的严格经验裁决；它不是 v10 的 RQ1–RQ10 实验集。
- `layout_summary_v13_*` 定义 **L-RQ1–L-RQ10**。统一命令运行“v10 严格 L-RQ1 + v13 全部十个 RQ”。
- 结果矩阵固定物化 `640 × 6 × 10 = 38,400` 个单元，但只有 exact case-matched、correct、paired alternatives 才标为 `paired_native_runtime`。
- TVM/CUTLASS 的子图边界投影、vLLM/SGLang 的整模型 serving、源码观察、框架不拥有的决策、Hexcute 在 A10 上的架构阻塞、单卡无法执行的 RQ5、以及 H4.4 跨硬件实验均保留为独立状态，不能冒充 exact native 证据。
- 因而该命令可以完成并审计全部卡1可执行实验，但一张 NVIDIA A10 不能使“所有框架 × 所有 RQ × 640 case 均获得原生运行时证据”这一字面命题成立。

## 本轮修复

- MoE 改成固定路由的 dispatch → grouped expert GEMM → combine，避免错误地为每个 token 计算全部专家。
- Mamba2 CUDA prefill 改为仿射 associative scan，消除 Python token 循环造成的长序列超时。
- boundary alternate 对 decode `S=1` tensor 使用真实 padded-stride view；不再把 contiguous no-op 当成 layout 实验。
- 显存预检使用“持久张量 + 实际复制的边界输入”，不再把全部权重机械乘二；FP32 recurrent state 按四字节计算。
- 所有整图结果保存 scale-aware 的 max-abs、max-relative、normalized error、RMSE、non-finite count；数值阈值注明“代表性 smoke 校准、640 全量运行前冻结”，不伪称为观察数据前预注册。
- 640-case 矩阵区分 exact native 与 projected native；CUTLASS tile id 不再被错误当成 layout candidate。
- vLLM/SGLang 请求按声明的 concurrency 分波执行，避免参数被记录但实际一次性并发。
- fail-closed 审计识别旧策略名和 no-op，要求 640 个 boundary case 均有状态，并要求每个可执行 case 有 native/alternate/repair 三联对照。
- 外层脚本即使某个长阶段失败也会保存其退出码、继续生成审计和复现脚本，最后才非零退出。

## 已完成 smoke

- 33 个 CPU/静态单元测试通过。
- 卡1确认是 NVIDIA A10 23028 MiB。
- GQA、SwiGLU、MLA、linear attention、sliding attention、sparse attention、Mamba2 的代表 prefill boundary case 均通过真实 stride 与数值检查。
- 一个 23.7 GB 持久张量估算的 MoE shape 在 A10 上被正确标为 `oom_preflight`，不是伪造成功。
- 旧结果重审计被正确判为不完整：旧 schema、153 个 boundary no-op、43 个 numerical mismatch、8 个 timeout 和 19 个 boundary error 都会阻止完成标志。

## 唯一完整重跑命令

```bash
cd /home/liangyilei/ladder_home

OUT=/home/liangyilei/ladder_home/staged/baseline_framework/layout_research/results/v10_v13_640_all_frameworks_$(date +%Y%m%d_%H%M%S)

GPU_PHYSICAL_INDEX=1 \
PYTHON_BIN=/home/liangyilei/conda/envs/cuda-opt/bin/python \
TORCH_PYTHON=/home/liangyilei/conda/envs/cuda-opt/bin/python \
FRAMEWORK_ENV_FILE=/home/liangyilei/ladder_home/staged/baseline_framework/layout_research/framework_envs.generated.sh \
SERVING_MODEL_PATH=/home/liangyilei/ladder_home/staged/baseline_framework/layout_research/models/Qwen2.5-0.5B-Instruct-7ae5576 \
CUTLASS_DIR=/home/liangyilei/cursor_home/Ladder/3rdparty/cutlass \
CHECK_SOURCE_URLS=1 \
V10_PROCESS_REPETITIONS=5 \
V10_WARMUP=8 \
V10_ITERATIONS=30 \
CUDA_WARMUP=8 \
CUDA_ITERATIONS=30 \
TRITON_WARMUP=8 \
TRITON_ITERATIONS=30 \
ALL_640_WARMUP=3 \
ALL_640_ITERATIONS=10 \
REAL_WORLD_CASE_TIMEOUT=3600 \
SERVING_STARTUP_TIMEOUT=3600 \
SERVING_REQUEST_TIMEOUT=3600 \
bash staged/baseline_framework/layout_research/run_v10_v13_640_all_frameworks_gpu1.sh \
  --output "$OUT" \
  --model /home/liangyilei/ladder_home/staged/baseline_framework/layout_research/models/Qwen2.5-0.5B-Instruct-7ae5576
```

脚本没有全局短超时；本命令让慢的单 case 和 serving 启动/请求各允许 3600 秒。输出目录带秒级时间戳且所有下层写入器拒绝覆盖已有路径。

## 跑完后优先检查

1. `$OUT/EXECUTION_STATUS.json`：三个阶段的退出码。
2. `$OUT/CROSS_VERSION_COMPLETENESS.md`：卡1执行完整性与不可消除阻塞。
3. `$OUT/v13_640/full_combined/V13_RQ_VALIDATION_REPORT.md`：v13 hypothesis/observation 裁决。
4. `$OUT/v13_640/full_combined/V13_640_FRAMEWORK_RQ_MATRIX.md`：38,400 单元的状态汇总。
5. `$OUT/v13_640/full_combined/v13_640_framework_rq_matrix.jsonl`：逐 case × framework × RQ 的证据类型与候选 layout。
6. `$OUT/REPRODUCE_THIS_RUN.sh`：保存本次参数的非覆盖复现入口。

外层命令非零退出不表示数据丢失；它表示 fail-closed 审计发现某个阶段失败或证据不足，应先读取上述两个状态文件再决定补跑范围。
