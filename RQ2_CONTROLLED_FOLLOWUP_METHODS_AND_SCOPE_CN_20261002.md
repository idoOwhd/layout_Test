# RQ2 补充实验的方法、范围与执行状态（2026-10-02）

## 先回答“全部代码是否完成”

**还没有完成共享讨论要求的全部原生验证。** 已实现的主测试与两项补充测试如下；不能将它们扩大成“207 个 PR/issue 全部复现”或“框架原生整图编译器已证明”。源码审计账本仍保留 source replay 的 not_run/pending 状态，不会因为同框架一个受控子图通过就给所有链接盖章。

| 实验 | 代码 | 实际验证 | 不能证明的部分 |
|---|---|---|---|
| H2.1 嵌套扩图 | rq2_growth_bench.py | 保持旧数学、shape、dtype；冻结全部已测近优旧决策，比较新增部分最佳已测延伸与自由方案 | 大图全空间最优、原生 engine default regret |
| H2.2 子问题 | rq2_domain_anchor_bench.py 的 conditional_domain_result | 真的分配一份额外 QKV 表示并转换；第一层 packed-QKV 的条件 K=1/2 与整图实测耗时 | 全图所有值/层级的独立 domain 分区和全局 K_epsilon |
| H2.3 子问题 | 同文件 runtime_anchor_result | 完整匹配上游决策×远端 KV anchor 的 2×2；其余决策固定；独立 holdout；真实语义 DAG 距离 | compiler SSA/layout pass 传播距离、control-flow pass 的限制、原 PR base/head |

六 adapter 为 PyTorch、Triton、CUTLASS、TVM、vLLM、SGLang。它们使用公开标注的实际原语和共享依赖，是 **受控/hybrid forward**，并非六个未修改默认优化器。模型 geometry/forward 来自 pinned Qwen3，数值为固定种子 FP16 测试数据，不是预训练权重或生产 trace。

## H2.2 的实际表示，而不是给变量改名

只针对第一层同一个 packed-QKV 逻辑值，其它决策固定在主实验正确的 best-measured 方案：

| producer 输出存储 | 所有后续 uses 的共同输入存储 | 显式输出表示数 | 计时内动作 |
|---|---|---:|---|
| row | 原 row buffer | 1 | 原生 GEMM 直接写 |
| column | 原 column buffer | 1 | 原生 GEMM 直接写 |
| row | 第二份 column buffer | 2 | GEMM 写第一份；真实 GPU copy |
| column | 第二份 row buffer | 2 | GEMM 写第一份；真实 GPU copy |

每份额外 producer buffer 有 256-byte prefix/suffix guards。记录实际 shape/stride、storage offset、是否不同 storage instance、Tensor payload/guard bytes。bytes 从真实 shape/element_size 计算，**不是测出来的 HBM transactions**。是否分配第二份、producer/store/copy/consumer 的整图耗时均由真实运行决定。

Q/K norm 的必需 contiguous、V writer repair、其它层的表示、register/shared-memory domains 不被计入上述 K。这里的 `conditional_explicit_output_K_epsilon` 是这个单值、四候选控制问题的训练近优统计量，绝不能叫全图 K_epsilon。B×q=1 的 row/column 地址映射相同，去重为一个直接候选，不制造假变体。

`shared` 是 K=1 中训练最快的方案；`free` 是四种方案中训练最快的方案。选好后，在新的 AB/BA 配对 holdout 中重测：

`speedup = median(shared_holdout_ms) / median(free_holdout_ms)`。

分子、分母来自完整 GPU pipeline 实测，包括实际 copy 和必要 repair；除法、K_epsilon、bootstrap 区间是计算统计量。K=2 训练更快不够，必须看独立 holdout；没有收益也必须报告。

## H2.3 的因果轴与距离

对同一个 graph contract，选择第一层 QKV output layout 作为上游变量 U；一行矩阵没有该物理轴时，选择第一层 KV layout（仅多 block 且存在第二个独立 anchor 时）。A 为最后一层 KV NHD/HND contract。

四格全部重新测量，不能从随机大空间候选里挑不匹配的点：

| | A=NHD | A=HND |
|---|---|---|
| U=0 | 其余 bits 同一个固定值 | 其余 bits 同一个固定值 |
| U=1 | 其余 bits 同一个固定值 | 其余 bits 同一个固定值 |

先重新测四格 training。`S_eps(U | A=NHD)` 包含两种 U 中所有训练近优选择。改变 A 为 HND 后，restricted 只能从该旧近优集合选择 U；free 可从两种 U 自由选择。两者只按 training 选，再独立 AB/BA holdout。

`speedup = median(restricted_at_HND_holdout_ms) / median(free_at_HND_holdout_ms)`。

其它轴只是固定在主实验 best-measured 方案上，不是声明在所有其它轴上已求全局最优。形状、权重、激活、有效 request lengths、RoPE 位置、mask、数学不变，只改变两个明示物理决策。

距离从 pinned forward 的节点 inputs/outputs 建有向 DAG，求 source→anchor 的最短算子路径；实际路径全部存入 JSONL。residual 分支可能绕过 MLP，因此不能拿 block 数乘一个常数。这个距离是 **语义 DAG 距离，不是 compiler SSA 或 register-layout 推断距离**。

没有 KV anchor，或声明的物理空间中不存在两个不同且有有向依赖的选择，标为本子实验的 semantic inapplicable；缺环境、缺 baseline、计算错误均为 failure。不能将这个有限方法的不适用外推为原 RQ2 不适用。

## 正确性与数据公平

- 沿用主实验的独立 FP32 单算子与独立 functional 全图 frontier/new-KV 参考，逐元素/NRMSE/finite gate；完整 historical/padding KV bitwise；每个计划在 timing 前后都检查。
- 新 copy 在 GEMM 写完时立即逐 bit 比较 source/destination，之后再允许下一层重用输出 backing；不能等 L7 覆写 L0 后再比较。
- 第二份存储的 guards 同时检查。额外分配/计划/JIT/参考计算在 timing 外，实际 producer、copy、repair、consumer 在 timing 内。
- 与 baseline 的 recorded data fingerprints 必须完全一致：第一层全部权重与完整 RoPE 表全量 SHA256，input/history KV 为采样 SHA256。采样不能称全状态哈希。
- 固定权重采用逻辑 `[K,N]` row-major，这是受控存储，不是原生 `F.linear` 通常 `[N,K]` row-major 参数的完整重放。native reader 固定 FlashInfer fa2；native engine 的其它 decode/backend/weight layout 仍未枚举。
- bootstrap 对分子/分母使用同一配对重采样索引，对应 ratio-of-medians；不能拿 median-of-ratios 的区间混用。仍只是同进程、未经多重比较校准的候选证据。

## 本次正在执行的链，不是已完成的结果

1. 58 个 CPU regressions 已通过，两个启动脚本和辅助脚本通过语法/导入检查。
2. 主 H2.1 最新 GPU smoke：六 adapter 各 10 图，合计 60 图、54 扩图对、1186 正确候选，零失败。
3. 卡1上已启动主 full：`results/rq2_graph_growth_full_20261002_approved`；应为六 adapter 各 1,020 图。PyTorch 部分已经完成：1,020 图、918 扩图对、21,222 正确候选、零错误/失败，退出状态 0，实际用时 12,344.14 秒（约 3 小时 26 分钟）。目前正在运行 Triton；仍不能宣布六框架全部完成。
4. 新对照的 CPU 代码通过，但 **GPU smoke 尚未执行**。已排队于 `results/rq2_controlled_followup_20261002`，等待主 full 完成且每个 declared case/候选全部通过。
5. 队列随后执行六框架新对照 smoke；任一失败停止，不会直接进入 full。全部通过后，使用相同冻结代码和 manifest 执行六框架各 1,020 图的新对照。

卡1同用户锁避免重叠；无 per-case 强制 timeout。所有新输出目录已有则拒绝写入，旧结果不删除、不覆盖。队列开始前已冻结补充 worker/launcher/helper；原主任务也使用自己的 immutable snapshot，不受本次继续开发影响。

新输出中 `conditional_smoke/RQ2_DOMAIN_ANCHOR_RESULTS_CN.md`、`conditional_full/RQ2_DOMAIN_ANCHOR_RESULTS_CN.md` 与对应 SVG/逐 case JSON 是实际运行后产生的文件，未产生前不能引用为通过证据。

## 从头复现当前已实现的三类受控实验：一个命令

```bash
cd /home/liangyilei/ladder_home

GPU_PHYSICAL_INDEX=1 \
FRAMEWORK_ENV_FILE=/home/liangyilei/ladder_home/staged/baseline_framework/layout_research/framework_envs.generated.sh \
RQ2_WARMUP=3 \
RQ2_ITERATIONS=10 \
RQ2_HOLDOUT_REPETITIONS=5 \
bash staged/baseline_framework/layout_research/run_rq2_controlled_suite_gpu1.sh
```

流程是主矩阵 extended smoke→主矩阵 full→完整性 gate→补充对照 smoke→补充对照 full；每步失败会明确退出。当前已有任务在跑，请不要同时执行第二份。

若主 full 完成后只想重跑补充对照，可复用旧文件 **只读**：

```bash
GPU_PHYSICAL_INDEX=1 \
bash staged/baseline_framework/layout_research/run_rq2_controlled_suite_gpu1.sh \
  --reuse-growth-full /home/liangyilei/ladder_home/staged/baseline_framework/layout_research/results/rq2_graph_growth_full_20261002_approved \
  --reuse-growth-smoke /home/liangyilei/ladder_home/staged/baseline_framework/layout_research/results/rq2_graph_growth_smoke_20261002_native_contract
```

## 仍未完成的要求，不能隐藏

全图合法 domain 分区/fusion/backend joint oracle；compiler/control-flow 真实传播 pass 及 bounded-local ablation；MoE/MLA/GDN/Mamba/sparse/offload 等非 dense Qwen3 结构；native module/engine 默认策略及原始权重布局、其它 attention backend；207 个来源各自原生 base/head replay；独立进程重复/多重比较/dispatch-noise 控制。Hexcute、跨架构、多 GPU 按当前限制不执行。源案例的 BF16/FP8/SM90+ 等限制必须逐项审计，不能被本 FP16 A10 类比取代。

所以，这个脚本完成后也只能说 **已实现的三个受控/条件实验全量通过或失败**，不能说共享讨论所有 RQ2 evidence 已经原生完整验证。
