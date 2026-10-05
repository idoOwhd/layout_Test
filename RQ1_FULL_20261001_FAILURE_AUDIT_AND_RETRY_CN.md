# RQ1 full 失败审计、修复与只补跑失败 case

## 结论

审计对象：`results/rq1_diversity_full_20261001_181601_926575`。原始文件全部保留，不在原目录重新生成报告。

这次实际执行的 100 个 `error` 均由测试 harness 给 FlashInfer 固定分配 **128 MiB 临时工作区**导致。不是计算正确性失败，也不是 CUDA 设备总显存不足。另有一个分析分类错误：没有执行成功的 case 被误记为 `insufficient_layouts`，已改为 `measurement_failed`。

已在物理 GPU1 完成修复 smoke：失败的 **10 个不同 contract × vLLM/SGLang 两个环境 × 1 次独立进程 = 20 个测量单元**全部通过；完整补跑仍需恢复原来的 5 次独立进程，共 100 个单元。smoke 通过不等于全部统计重复已补齐。

## 1. 原始结果的实际范围

| 项目 | 原始结果 |
|---|---:|
| 去重 contract | 590 |
| 机制族 / 模型配置来源 | 14 / 12 |
| case × consumer mode | 854 |
| 独立进程次数 | 5 |
| 预期 case × consumer × process | 4270 |
| 原始 measurement rows | 36260 |
| `success` rows | 35300 |
| `illegal_consumer_contract` rows | 860 |
| `error` rows | 100 |
| 通过计算正确性与执行覆盖的测量单元 | 4170 |
| 失败的不同 case × consumer | 20 |

一个测量单元有 P-only、C-only、R-only 或完整 edge 的多行记录；不能把 36260 行当成 36260 个独立 case。`illegal_consumer_contract` 表示预期拒绝不合法的 native consumer stride，合法 repair 路径另测，不是这次需修复的错误。

原始分析有 7 个 `rq1_supported_in_candidate_space`、396 个 `no_layout_conflict_in_candidate_space`、321 个 `inconclusive_noise_or_winner_instability`、130 个 `insufficient_layouts`。最后一类中有 20 个实际上是执行失败，应单列；另外 110 个才是已测但候选地址映射不足。旧报告保持原样，新报告使用修正后的分类。

## 2. 为什么工作区不够：源码与日志证据

测试调用的库来自两个已安装环境中的 FlashInfer，使用 `backend="fa2"`。q=1 时使用 `BatchDecodeWithPagedKVCacheWrapper(..., use_tensor_cores=True)`；该 tensor-core decode 路径复用 prefill 实现，因此错误中的 `batch_prefill_tmp_*` 不代表把 decode shape 错改成了 prefill。

已安装源码位置（两个环境具有相同相关分配逻辑）：

- `framework_envs/layout-vllm/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/attention/scheduler.cuh:855` 附近：split-KV 先分配中间向量 `batch_prefill_tmp_v`，再分配归约标量 `batch_prefill_tmp_s`。
- 同文件中向量字节数为 `num_qo_heads * padded_batch_size * cta_tile_q * head_dim_vo * sizeof(float)`；标量字节数为 `num_qo_heads * padded_batch_size * cta_tile_q * sizeof(float)`。这是库源码公式，不是实测 GPU 总内存。
- `.../flashinfer/allocator.h:61` 附近：在调用方提供的 buffer 内做对齐分配；容量不足时打印 requested / remaining bytes 并抛异常，并未报告 CUDA `out of memory`。

原始日志归一化到申请类型后的三类失败：

| 申请对象 | 本次申请字节数 | 当前剩余字节数 | 失败单元数 |
|---|---:|---:|---:|
| `batch_prefill_tmp_s` | 262144（256 KiB） | 0 | 40 |
| `batch_prefill_tmp_v` | 268435456（256 MiB） | 134217728（128 MiB） | 20 |
| `batch_prefill_tmp_v` | 536870912（512 MiB） | 134217728（128 MiB） | 40 |

第一类是向量已经恰好占满 128 MiB，后面的标量没有位置。另两类连向量都放不下。只把 buffer 改到与向量申请大小完全相等，仍可能在后续标量分配时失败。

## 3. 精确需要补跑的 10 个 contract

以下 ID 都具有前缀 `kv_writer_swa_flashinfer-`。两种 consumer mode 均失败，每种需补跑原来的 5 次独立进程。所有 case 保持 FP16、Hkv=1、D=512、window=128；B 表示独立 request 数，q 表示每 request 新增 token 数，KV 表示每 request 总缓存 token 数，page 表示每物理页容纳的 token 数。

| case ID 后缀 | manifest 模型配置来源 | Hq | B | q | KV | page |
|---|---|---:|---:|---:|---:|---:|
| `4c8456a213071550` | DeepSeek-V4-Flash-0731 | 64 | 8 | 1 | 4096 | 16 |
| `99b327619d0eb782` | DeepSeek-V4-Flash-0731 | 64 | 8 | 1 | 4096 | 64 |
| `5581a52897f9f918` | DeepSeek-V4-Flash-0731 | 64 | 1 | 32 | 8192 | 16 |
| `4a19bedd9a583012` | DeepSeek-V4-Flash-0731 | 64 | 1 | 32 | 8192 | 64 |
| `3178253fd0233fd2` | DeepSeek-V4-Pro | 128 | 2 | 1 | 512 | 16 |
| `943707209377dfd2` | DeepSeek-V4-Pro | 128 | 2 | 1 | 512 | 64 |
| `31d6f89cd2e3cd11` | DeepSeek-V4-Pro | 128 | 8 | 1 | 4096 | 16 |
| `641d4bc2265dd855` | DeepSeek-V4-Pro | 128 | 8 | 1 | 4096 | 64 |
| `223ca30a3263314e` | DeepSeek-V4-Pro | 128 | 4 | 1 | 8193 | 16 |
| `ac9a9ad58bdbf3ea` | DeepSeek-V4-Pro | 128 | 4 | 1 | 8193 | 64 |

完整模型 revision、配置 URL、tensor seed、维度与 provenance 从原始 `manifest.json` 直接复制，不重新生成。这里的模型名表示配置维度来源：实际执行的是该测试明确规定的同维度滑窗 attention analogue，不是 DeepSeek 专用压缩缓存/非对称 Q/K/V 架构的整模型复现。

## 4. 已修复的代码

| 文件 | 修改内容 |
|---|---|
| [rq1_flashinfer_workspace.py](rq1_flashinfer_workspace.py) | 解析 attention float scratch 的 `AlignedAllocator` 溢出，按申请量与已有占用有界增长；默认 128 MiB 起步、2 GiB 上限。未知异常与真正 CUDA OOM 不吞掉。 |
| [rq1_diverse_edge_bench.py](rq1_diverse_edge_bench.py) | 只在 plan/setup 阶段重建 wrapper 并扩容；记录容量与扩容过程。shape、随机种子、FP16、backend、split 策略、window、layout 和数学计算不变。新增精确 case ID 选择。 |
| [analyze_rq1_diversity.py](analyze_rq1_diversity.py) | 执行失败单列 `measurement_failed`，不再冒充只有一种独立 layout 的负对照。 |
| [rq1_failure_retry.py](rq1_failure_retry.py) | 根据旧数据选择失败/缺失的整个 case×consumer×process；新建结果目录；整单元替换失败记录，成功旧单元原样复用；每行保留来源路径。 |
| [run_rq1_failure_retry_gpu1.sh](run_rq1_failure_retry_gpu1.sh) | 串行使用物理卡1，自动继承旧运行的 warmup、iterations、重复次数；补跑、分析、正确性检查与新目录中的完整合并报告一条命令完成。 |
| [check_rq1_diversity_run.py](check_rq1_diversity_run.py) | 只补跑时按精确 retry cell 清单检查；重复的 stage/strategy 测量不允许混入。 |
| [rq1_diversity_preflight.py](rq1_diversity_preflight.py) | 把工作区修复模块加入源码 hash 快照。 |
| [test_rq1_failure_retry.py](test_rq1_failure_retry.py) | 工作区扩容边界、未知异常/上限拒绝、失败选择去重、分类、坏正确性拒绝、旧文件不变及整单元合并的回归测试。 |

**计时边界未改变：**新分配与重新 plan 均在 CUDA event 计时区域之外；P-only 和完整 edge 仍用原来的实测方法。原来成功的 case 默认仍沿用共享的 128 MiB 工作区；不是把全体改成 1 GiB，也没有通过关闭 split-KV、缩小 shape 或换 backend 绕过错误。

**上次的全局命令依然可用：**`run_rq1_diversity_gpu1.sh --full` 会自动调用修复后的 benchmark，不需要额外工作区参数。若只修本次失败，应使用下面的 failed-only 命令，避免重测所有成功对象。

## 5. 已执行的验证

修复 smoke 保存到新目录：`results/rq1_workspace_failure_retry_smoke_20261001_v1`。

| 检查 | 实测结果 |
|---|---|
| 所有失败 contract × 两个环境，独立进程各一次 | 20 / 20 通过 |
| 成功 stage rows | 60（每单元 P-only、C-only、完整 edge） |
| error / numerical mismatch / OOM / missing | 0 |
| 最大 observed normalized RMSE | 0.0002611743111629039；原阈值 0.01，未放宽 |
| smoke 最终工作区容量分布 | 256 MiB：8 单元；512 MiB：4 单元；1024 MiB：8 单元 |
| CPU unit tests | 原 6 项 + 新 8 项，共 14 项通过 |
| 原结果文件 SHA256 全量校验 | 所有旧文件均未改变 |

normalized RMSE 为误差平方均值开根号后，再除以参考输出的 RMS；数值越小，计算与独立参考越接近。容量分布是实际 plan 记录，不是预估值。

smoke 的合并预览仍保留 80 个旧失败，因为每个 case×consumer 只补了 1/5 次进程。此时 `merged/CORRECTNESS_AND_COVERAGE_CN.md` 为 False 是预期的，不代表这次 20 个修复测试失败。完整补跑不能用 smoke 的 1 次结果复制成 5 次。

## 6. 只补跑失败 case：完整命令

```bash
cd /home/liangyilei/ladder_home

GPU_PHYSICAL_INDEX=1 \
TORCH_PYTHON=/home/liangyilei/conda/envs/cuda-opt/bin/python \
FRAMEWORK_ENV_FILE=/home/liangyilei/ladder_home/staged/baseline_framework/layout_research/framework_envs.generated.sh \
RQ1_PROCESS_TIMEOUT=28800 \
bash staged/baseline_framework/layout_research/run_rq1_failure_retry_gpu1.sh \
  --previous /home/liangyilei/ladder_home/staged/baseline_framework/layout_research/results/rq1_diversity_full_20261001_181601_926575 \
  --full
```

无需模型下载、无需重装环境。这份原始结果会选择 **100 个失败单元**，只启动 vLLM、SGLang 各 5 个独立进程；不重测 Torch 成功对象。warmup=8、iterations=30 从旧 `preflight.json` 自动继承，原输入/shape/seed 保留。运行时应保持卡1空闲且不要同时修改测量源代码。

脚本自动建立 `results/rq1_failed_retry_full_<时间戳>_<PID>`，已存在目录被拒绝。也可用 `--output <不存在的新目录>` 指定路径，不能放到旧结果目录中。

关注新目录中的这些文件：

- `FAILED_CASE_RETRY_PLAN_CN.md` / `retry_plan.json`：精确 case、环境、原 repetition 编号、旧失败原因；保证没有把成功 case 混进来重跑。
- `raw/*.jsonl` / `logs/*.log`：本次补跑的原始测量、正确性与工作区记录。
- `CORRECTNESS_AND_COVERAGE_CN.md`：仅本次 100 个选择单元的检查。
- `merged/RQ1_DIVERSITY_REPORT_CN.md`：旧成功结果加上这次修复结果的完整 RQ1 报告。
- `merged/CORRECTNESS_AND_COVERAGE_CN.md`：完整 4270 个预期单元的检查。
- `merged/MERGE_PROVENANCE.json`：替换数量、来源与所有旧文件未改变的验证。成功补齐时 `replaced_cells=100`，`unrepaired_selected_cells=[]`。

合并是“旧成功测量 + 新修复测量”，不是宣称所有对象在同一次新 full run 中执行。任何未通过正确性/缺失/超时的补跑单元不替换旧记录，完整检查仍失败并保留原因。不会只挑最快的策略或选择有利的重复。

## 7. 修复后的科学解释与仍未完成的范围

这些失败对象都有 Hkv=1：NHD/HND 交换的 head 轴长度为 1，去掉 singleton stride 差别后是相同的实际地址映射。修复后的 20 个 smoke 对象因此正确归为 `insufficient_layouts`，可以检查接口与计算，却不能用来证明两个独立布局的 producer-local/edge-global 冲突。扩容本身也不是 layout 加速方案。

原始 7 个 RQ1 正例未被补跑或修改；其结论仍限于候选空间与受控边界，不代表框架默认策略必然次优。321 个不确定对象不是执行错误，只有希望加强统计结论时才应另外设计更多独立重复；本 failed-only 命令不会挑选性重测它们。

`all-links` / `source_coverage` 账本的 **173 个唯一 PR/issue 的原生 replay 尚未完成**。本次原始运行中 120 个 REST 核查、53 个 html_pending；`native_replays_completed=0`，`all_link_native_validation_complete=False`。修复工作区不会把 URL 可访问、通用算子 analogue 或两个环境里的同一个 FlashInfer kernel 算成逐 PR 原生验证。这是独立的 adapter/硬件覆盖缺口，不是上述 100 个运行错误的根因。

本轮不修改已安装的 FlashInfer/vLLM/SGLang 源码，不重装环境，不覆盖历史结果，也未 push 到远端。
