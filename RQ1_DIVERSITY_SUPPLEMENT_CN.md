# RQ1 多样性补充：范围、去重、shape 来源与运行方法

## 先明确未完成的要求

本次新增代码提供直接 producer emission 的 controlled/hybrid 边界实验，以及共享讨论**全部 173 个唯一 PR/issue**的源码与覆盖账本。

**它尚不等于 173 个 PR/issue 各自的原生性能复现，也不等于所有框架默认 layout 均已验证。** 原生 replay 的逐 source adapter 未补齐，账本对此 fail-closed；不能把相关的通用算子实验或一个源码 URL 可访问，登记为原 PR 验证成功。ROCm/AMD、Blackwell、Hopper、网络/多 GPU 的要求需要逐源码核实，A10 卡 1 无法代替其他硬件。

完整讨论原文保存在 `ref_talks/shared_6abe1dff_rq1_full_visible_deduplicated_20261001.md`。去掉重复流消息后有 19 个可见 final 回答；PR/issue URL 集合与 `rq1_shared_github_urls.json` 严格一致。此前精简归档不删除。

## RQ1 真正在问什么

比较 producer 的局部最优布局 `argmin_L T_P(L)` 与整个边界的联合决策 `argmin_(L,R) T_edge(P(L),R,C)`。

只观察两个 consumer 喜欢不同布局，或测“先按原布局输出再 transpose”的路径，不能证明 producer 局部最优与 edge 最优冲突。本实现的 GEMM/BMM、gather、RMSNorm、KV append producer 都直接写目标地址；计算、输入、dtype 和同族的 Triton tile/launch 保持相同。Pinned-host copy 是额外的 placement 边界，单列，不能代替纯 kernel-layout RQ1。

## 新增结构

| 机制族 | Producer → consumer | 证据范围 |
|---|---|---|
| attention_bmm_oproj | AV BMM → head/token flatten → O-projection | 固定 tile 的 Triton BMM + cuBLAS；B>1 使用真正 `[B,q,H,D]` backing |
| projection_sdpa | QKV projection → split/view → 完整 softmax attention | Packed row 与分量分别 dense 的 direct emission；不是 KV scan |
| sparse_gather_sdpa | selected-token gather → compact selected-KV attention | 通用选中集合 attention；不是 DeepSeek MSA/QSA 原生复现 |
| projection_swiglu_down | gate/up projection → SwiGLU → down GEMM | torch / vLLM `_C.silu_and_mul` / `sgl_kernel.silu_and_mul` consumer |
| moe_dispatch_expert | 固定路由的 token dispatch → grouped expert GEMM | 本地受控边界；不是 DeepEP 网络 dispatch |
| moe_gemm_combine | route-grouped FC2 GEMM → weighted local combine | 受控本地 route contraction；不是框架默认 grouped-MoE 实现 |
| state_prefill_decode | KᵀV state write → recurrent read/update | 明确无 gated delta rule；不能标为 KDA/GDN |
| projection_gdn | packed/separate QKV projection → native chunk GDN | vLLM / SGLang 已安装版本的 FLA kernel；两者 state 接口分别适配 |
| gemm_bias_gemm | GEMM → bias → GEMM | Torch/cuBLAS consumer，显式同数学计算 |
| reduce_norm_gemm | RMSNorm direct emit → GEMM | reduction producer，不限于 KV writer |
| host_weight_gemm | pinned CPU FP16 weight fill → GEMM | placement extension；不是量化 Marlin/NVFP4 replay |
| host_kv_sdpa | pinned CPU KV restore → selected-KV attention | placement extension；不是框架完整 KV offload 子系统 |
| kv_writer_flashinfer | direct append-only paged writer → real FlashInfer decode/prefill | NHD/HND、打乱物理页、多 request 独立 page table |
| kv_writer_swa_flashinfer | direct paged writer → sliding-window attention | 使用模型配置中的 window；不是 DeepSeek 特殊压缩 cache 原生实现 |

这里只把原生算子调用记为 native **consumer**，producer 为新写的 Triton kernel。vLLM/SGLang 环境中的 FlashInfer consumer 仍属于 FlashInfer，不是两份独立框架默认策略证据。TVM、CUTLASS、Triton 编译器 upstream、IREE 等 source 的原生 replay 尚有缺口，不能把上表登记成它们已验证。

## Shape 多样性与真实来源

full manifest 当前为 **590 个去重 contract、14 个机制族、12 个模型配置来源**。确切来源、完整 revision、config URL、使用维度、原 parent case 与所有重复 provenance 在生成的 `manifest.json` 中保留。

来源包括 Qwen3/Qwen3.6、Granite 4、DeepSeek V3.2/V4、Kimi Linear、GLM 5.2、LLaDA2 等配置；一个 GPT-Neo 配置作为真实 SWA/MHA 小型对照。不同机制族只使用其中适用的维度。

- B：1、2、4、8；同时覆盖单 request 和多个独立 request。
- q：1、4/16、31/32/33、63、127/128、256/257 等；decode 和 prefill 分开。
- KV：128、511/512、1025、2048、4096/4097、8192/8193 等；包含末页不满与 tile 边界附近的长度。
- 配置中的 head dimension 范围包含 64、72、128、192、256、512（并非每个族都有全部维度）。
- 普通/滑窗 attention 与 GDN consumer 分开；paged layout 用 page size 16/64；页映射打乱而不是仅测试连续页。
- state recurrence horizon 与矩阵形状分别记录。

**来源边界：**模型配置提供架构维度，随机 tensor 实现这里明确说明的算子/contract；B/q/KV 是受控 workload 扫描，不是采自线上 trace。没有下载或加载整模型权重，也不能据此声称专用 MLA/稀疏/量化架构的完整子图已经原样运行。旧 parent manifest 的标准化 shape 本身也不是 full-model 执行轨迹。特别是模型有非对称 Q/K/V、压缩 cache 等语义时，通用同维度 attention 只能作为标明范围的 analogue。

## 去重规则

1. PR/issue URL：去 query/anchor，以可核查 canonical URL 合并；完整上下文与 aliases 保留。一次引用十次不产生十个验证对象。
2. contract：以族、实际使用的 tensor 维度、dtype、必要的 B/q/KV/page/state horizon 哈希；不把模型名或未使用的 KV 长度当新实验变量。
3. 纯 token 算子按 `T=B*q` 去重，相同 T 的 B/q 分解作为 `equivalent_workload_aliases` 保留。Attention 等真正区分 request 的算子不能这样合并。
4. layout：extent=1 的 stride 差异不产生不同地址映射；同 mapping 的名称记录为 `equivalent_layout_aliases`。例如 q=1 的两种 row/column 布局可能实际等价。
5. 5 个独立 process repetition 是统计重复，不是新 shape，不应被上述规则删除。
6. RQ1 支持率不能按重复来源、环境名或同一个 native kernel 的重复调用人为扩充样本数。

## 计算正确性门槛

- 每个 producer 检查逻辑输出；每条合法 `P→R→C` 完整路径单独检查最终输出。
- 主 workload 的输入、权重、KV、主要输出均 FP16。FP32 accumulator、GDN gate/state、数学参考按算法/原生接口保留并单列；不把其它 dtype 的性能混入这里。
- Attention reference 为独立 FP32 `QK → softmax → AV`，不是再调一次 FlashInfer；SwiGLU reference 为 PyTorch 数学表达式；GDN reference 为独立逐 token FP32 recurrence。
- GDN state 真实存储为 `[B,H,V,K]`。SGLang 返回的 chunk-initial `h` 不是 final state；测其实际写回的 pool，并在每次执行前 reset，使重复计时不会演化输入状态。
- 检查 shape、finite、`rtol=0.03`、`atol=max(1e-5,4×2^-10×reference_RMS)`，并强制 normalized RMSE≤0.01。误差阈值和实际误差均写入原始结果，不只给一个 pass。
- CPU unit tests 包含故意的全零错误输出和错误 transpose，必须拒绝。仅用固定 0.03 绝对容差会误放过小幅值错误，已避免。
- 不把非法 stride 送入 dense-only native kernel；拒绝记录为 `illegal_consumer_contract`，随后测试合法 materialization 路径。OOM/unsupported/timeout/数值不通过不能算成功。
- `check_rq1_diversity_run.py` 逐预期 case×consumer×process 检查 coverage；代码 hash 在测试过程中改变也判为未通过。

## 计时与统计

P-only / C-only / R-only 诊断分开计时；C-only 保留真实 stride，不能因为 `clone()` 被 densify。完整 edge 使用一对 CUDA events 测实际 pipeline，不能把阶段 median 相加冒充整体延迟。

full 默认 warmup=8、iterations=30、独立 process repetition=5；不同 consumer 环境串行，占用同一物理 GPU1。前 2 个进程只选择 P-local layout 与两条 edge 策略，后 3 个进程评估：

`speedup = measured T_edge(P-only 选的 layout + 其最佳合法 repair) / measured T_edge(edge 联合选择的 layout/repair)`。

分子、分母是完整 pipeline 实测；speedup/regret/CI 是派生计算。选择与评估不使用同一组数据。winner 稳定性、paired bootstrap CI 与 `max(3%,3×process CV)` noise/practical gate 决定是否支持 RQ1。5 次进程只是最小门槛，3 个 held-out 样本的 CI 仍有限，可增加为 10/20 次。

单次 smoke 只能验证实现和正确性，不能宣称已证明 RQ1；候选内最优也不能证明全局 layout 最优。没有在原框架默认 layout 上测 native-auto baseline 的条目不报告成默认策略 regret。

## 完整运行命令

不需要模型 checkpoint。每次自动创建带时间戳和 PID 的新结果目录，已存在目录被拒绝；旧脚本、旧结果不修改。

```bash
cd /home/liangyilei/ladder_home

GPU_PHYSICAL_INDEX=1 \
TORCH_PYTHON=/home/liangyilei/conda/envs/cuda-opt/bin/python \
FRAMEWORK_ENV_FILE=/home/liangyilei/ladder_home/staged/baseline_framework/layout_research/framework_envs.generated.sh \
CHECK_SOURCE_URLS=1 \
RQ1_PROCESS_REPETITIONS=5 \
RQ1_WARMUP=8 \
RQ1_ITERATIONS=30 \
RQ1_PROCESS_TIMEOUT=28800 \
bash staged/baseline_framework/layout_research/run_rq1_diversity_gpu1.sh --full
```

若希望复用本次 173 条源码核查，避免 GitHub API 限流，可把 `CHECK_SOURCE_URLS=1` 替换为：

```bash
RQ1_SOURCE_AUDIT_JSON=/home/liangyilei/ladder_home/staged/baseline_framework/layout_research/results/rq1_source_audit_20261001_retry173/source_audit.json
```

该文件只复用源码核查，不复用 GPU 性能结果。不能让它指向 `all_source_registry.json`，两者 schema 不同。

测试可分三种 smoke（使用相同 env 指定方式，把最后的 `--full` 替换为对应参数）：

- `--smoke`：各族最小单例。
- `--extended-smoke`：各族 B=1/B>1、decode/prefill、partial page。
- `--architecture-smoke`：每个被选真实配置的 decode/prefill，避免只在最小模型上通过。
- `--validation-smoke`：上述扩展和架构 smoke 的去重并集，最终交付版复核使用这个入口。

unit tests：

```bash
cd /home/liangyilei/ladder_home/staged/baseline_framework/layout_research
/home/liangyilei/conda/envs/cuda-opt/bin/python -m unittest -v test_rq1_diversity.py
```

每次输出包含 `manifest.json`、`preflight.json`、`raw/*.jsonl`、`logs/*.log`、`RQ1_DIVERSITY_FINDINGS.json`、`RQ1_DIVERSITY_REPORT_CN.md`、`CORRECTNESS_AND_COVERAGE_CN.md`、`source_coverage/ALL_PR_ISSUE_COVERAGE_CN.md`。原始 stage、strategy、shape/stride、每次 timing samples、误差均保留。

full 中 `RQ1_MAX_CASE_BYTES` 默认 8 GiB 是每 case 的保守预算，不是 GPU 总显存；大 MoE 或其它超预算条目可能 `oom_preflight`，这种结果使 coverage gate 未通过，不会假报“全跑完”。可依据剩余显存调整预算，不能仅为 pass 去裁剪架构维度。

## 下一阶段必须补齐的部分

源代码与完整链接账本目前覆盖全部 URL，但各 PR/issue 需要逐一确认其确切 P/C contract、dtype、硬件、upstream revision、patch 前后语义与 native replay adapter。`candidate_causal_families` 为找相关机制的候选索引，**不是人工审核后的等价性判断**。不能将本命令的完成当成“所有链接都已经测过其原生效果”。

## 2026-10-01 最终 smoke 实测

结果目录：`results/rq1_diversity_validation_smoke_20261001_final_v3`。

| 项目 | 实测/检查结果 |
|---|---|
| 去重后 smoke contract | 132（14 个机制族） |
| 预期 case×consumer×process 单元 | 179，全部检查通过 |
| 成功的 stage measurement rows | 1451；这是 P/C/R/edge 行数，不是 1451 个独立 case |
| 非法 native consumer stride 路径 | 35，按预期拒绝；对应合法 repair 路径另测 |
| numerical mismatch / error / OOM / missing | 最终联合 smoke 中均没有 |
| 最大实测 normalized RMSE | 0.0009086995269171894；阈值 0.01 |
| CPU protocol/正确性负对照 unit tests | 6 项全部通过，包括故意的零输出与 transpose 错误 |
| 完整源码集合 | 173 个唯一 PR/issue；另外检查 user messages 无新增 PR/issue URL |
| Source body 获取 | 57 个 REST，86 个 clipboard 原始 markdown，30 个 embedded JSON；完整 body 已在 ledger 中留 hash/来源 |
| 原 PR native performance replay | 未完成，不能据此报所有 PR 已测试 |
| RQ1 科学结论 | smoke 只有 1 个独立进程，不作性能结论 |

`CORRECTNESS_AND_COVERAGE_CN.md` 和 `correctness_coverage.json` 保存了最终逐预期对象检查。`RQ1_DIVERSITY_REPORT_CN.md` 中 42 个单元为 insufficient_layouts（包括 q=1/MQA 等实际只有一个独立地址映射的负对照），137 个为 smoke_only_insufficient_process_repetitions；这两个类别都不是 RQ1 支持结论。

前期 smoke 的失败文件保留，未覆盖：vLLM/SGLang 接口差异、GDN state `[V,K]` 与 `[K,V]` 的误用、过严的 near-zero cancellation 绝对检查等已在最终版修正。后者改用 scale-aware FP16 误差门槛，并保留 normalized RMSE 和故意错误负对照，未通过降低检查标准来放过错误输出。

旧全局入口未改动；新补充入口独立保存结果。没有将本轮代码擅自 push 到远端。

## 2026-10-01 full 失败修复与 failed-only 补跑

对 `results/rq1_diversity_full_20261001_181601_926575` 的审计、10 个失败 contract 清单、源码证据和 smoke 结果见 [RQ1_FULL_20261001_FAILURE_AUDIT_AND_RETRY_CN.md](RQ1_FULL_20261001_FAILURE_AUDIT_AND_RETRY_CN.md)。

100 个错误是 FlashInfer 固定 128 MiB float scratch 不足，已改为 plan 阶段有界扩容；shape、dtype、backend、split 策略和计时方法不变。失败不再误记为 `insufficient_layouts`。卡1 smoke 覆盖全部失败 contract 在两个环境中的 20 个对象，均通过计算正确性；还需按原 5 个进程重复完整补跑 100 个失败单元。

```bash
cd /home/liangyilei/ladder_home

GPU_PHYSICAL_INDEX=1 \
bash staged/baseline_framework/layout_research/run_rq1_failure_retry_gpu1.sh \
  --previous /home/liangyilei/ladder_home/staged/baseline_framework/layout_research/results/rq1_diversity_full_20261001_181601_926575 \
  --full
```

新建带时间戳的结果目录，只重测失败的完整 case×consumer×process 单元；`merged/` 保存旧成功数据与新修复数据的完整报告及每行来源。原文件保持只读；原 `run_rq1_diversity_gpu1.sh --full` 也自动使用修复后的工作区逻辑。
