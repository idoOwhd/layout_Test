# RQ2 新增可执行实验、正确性、运行边界（2026-10-02）

## 完成与未完成：不能把“可运行”写成“全部科研要求满足”

共享讨论中的 RQ2 分成 H2.1（扩图后旧布局的最优性失效）、H2.2（近优独立 layout domain 数量改变）、H2.3（跨边界的远距离因果传播）。主矩阵覆盖 **H2.1 的受控空间**；随后新增 `rq2_domain_anchor_bench.py` 的真实 QKV 第二份表示、匹配 2×2 远端 KV anchor 对照。后者分别只是 **H2.2 的单值表示子问题、H2.3 的运行时条件响应子问题**。整图任意 domain 分区/融合及 `K_epsilon`、compiler/control-flow 传播和所有 PR patch 原生复现，**仍未完成**。

| 要求 | 此套代码的实际范围 |
|---|---|
| 真实模型来源 | 六个 pinned Qwen3 配置；已核对 vLLM/SGLang 的 Qwen3 attention/decoder、Qwen2MLP forward |
| 原模型权重 | 不下载权重。真实 geometry/forward + 固定种子 FP16 测试权重、激活、历史 KV；不是 production trace |
| 图逐步扩大 | norm→QKV；+split/QK norm；+RoPE；+attention/KV update；+O-proj；+postnorm/gate-up/SiLU；完整 block；2/4/8 blocks |
| 多 shape | 720 uniform 图 + 300 mixed-request 图 = 1,020 图、918 相邻扩图对；不是旧 640 清单的简单重编号 |
| request 数 | decode B=1/4/16；prefill、extend B=1/4，保留每个 request 的独立 KV |
| 变长 request | B>1 decode/extend 的 KV 长度不同，含不满 page 尾部；每 request 的 RoPE 位置、mask、page counts/last length/slot 全部真实改变 |
| 方法比较 | PyTorch、Triton、CUTLASS、TVM、vLLM、SGLang 的 **受控/hybrid adapter**。每个算子来源公开记录；不是六个未修改框架的默认整图优化器比较 |
| 多布局 | 每 block 的 QKV/O/gate-up/down GEMM 输出 row/column；cache NHD/HND。B×q=1 时去掉同地址映射的 row/column 假变体 |
| 独立选择 | 同时改变多个决策；扩图保留全部小图近优已测选择，补齐新轴；训练选择与独立 holdout 分离 |
| 正确性 | 每个 intermediate/residual 在 buffer 重用前立即作独立 FP32 单算子检查；独立 functional 全图检查 frontier 和各层新 KV；new KV/全部 unwritten/padding KV bitwise；首尾 canary；重复执行后再次检查 |
| 所有链接 | 已对共享讨论 207 个去重 source 做源码/patch/test/dtype 审计；**1,020 图不等于原生执行这些 207 个 source 的全部实验** |
| 科学结论 | 只能由新结果判定支持/反例/不确定；不保证预设猜想成立，不拿不可运行实验补零假装成功 |

## 对照的真实实现

- `rq2_growth_search.py`：有限 layout 空间、近优集合、冻结扩图、paired holdout 的结果公式。最多五个有效轴时穷举。大空间是 bounded measured candidates；为每个旧近优选择至少提供一条已测扩展，因此 proposal budget 不会静默删掉旧近优成员。**大图 free/frozen 都仅在同一最终已测集合内求最优**，并非证明最佳合法扩展或全局最优。
- `rq2_growth_bench.py`：真实 Qwen3 forward 切片。跨层返回 `(MLP hidden, residual_after_attention)`，下一层 input norm 执行 add-RMS；没有凭空加一个 post-MLP residual operator。`head_dim` 来自 config，不能假设 hidden/head_count。
- `rq2_growth_kernels.py`：固定 arithmetic/tile 的 Triton weighted RMS/add-RMS、SiLU*up、streaming causal GQA。decode/extend 的 mask 是右对齐 `key_position <= KV−q+query_position`。
- `build_rq2_mixed_request_cases.py`：保留原 720 ID，追加 300 mixed ID，重新计算 graph contract hash，保持每一组 nested graph 的旧节点语义相同。独立 CPU 测试检查 request 长度、causal mask、signature、去重、负例。
- `rq2_growth_cutlass.cu`：真正 CUTLASS SM80 tensor-core FP16 GEMM，直接写 row/column output，检查 `can_implement` 和 launch status；其余图算子来源另行标注，不声称是 CUTLASS 的整图编译。
- TVM：已有 TVM 环境的 TE→S-TIR→CUDA norm/add-RMS（含线程 reduction），接受显式 physical strides。与 torch 共用 CUDA stream。其余 kernel 是公开标注的共享 Triton 实现，不称为 TVM 原生整图优化。
- vLLM：native RMS/add-RMS、rotary、SiLU*up、native cache writer；GEMM 是其共享 torch/cuBLAS dependency，attention consumer 是固定 `fa2` 的 FlashInfer native dependency。不是 vLLM scheduler、native graph capture、default attention dispatch 的原样重放。
- SGLang：native public wrapper；当前 RMS wrapper 在 FP16 eager 路径实际调用 FlashInfer，因此不是 SGLang 自有 RMS kernel 的独立性能证据。其它算子明确记录来源，同样不冒充整个 serving engine。

固定 GEMM 权重采用 `[K,N]` 的 row-major 测试存储；这也不是原生 `F.linear` 参数通常 `[N,K]` row-major 所对应的同一种物理权重表示。此次控制的是输出 layout 与 cache 表示，不能据此声称所有原生模型权重 layout/tactic 已被重放。native attention 固定用 FlashInfer fa2 的 BatchPrefill wrapper（q=1 也如此），不是完整枚举各 engine 的 decode backend。

### Native HND 的重要修正

不能给只支持 NHD 的 writer 塞一个错误 HND-strided 4D tensor，尤其 vLLM 的 head-major 分支有不同的 5D K / 4D V contract。新增实验采用：

1. NHD：直接写实际 paged NHD cache，再由原生 attention 读取；
2. HND：原生 writer 把 **本次新 token** 写入一个紧凑 NHD staging cache，随后把新 token scatter 到实际 compact paged HND cache，原生 attention 直接读 HND；
3. staging write 和 scatter 都计入整图时间；不对历史 KV 全量转置，减少人为制造的不公平耗时和冗余显存；
4. canonical KV materialization 只为检查/参考发生在计时外，不能省略真实 consumer 要求的修复；
5. staging 和真实 cache 均有 canary，全部更新元素与 producer 的 FP16 输出逐 bit 比较，历史区域也逐 bit 比较。

HND 是一个明确的实验 repair strategy，不是声称当前框架默认 writer 原生支持这一 HND contract。

## 样本与随机值

六个模型：Qwen3-0.6B/1.7B/4B/8B/14B/32B，均有 pinned HF revision。长度组合来自可执行的 controlled serving workload，而非声称某张 trace 中恰好出现的生产请求：

- decode：B=1/4/16，q=1，KV=512/4096；
- prefill：B=1/4，q=KV=128/1024；
- extend：B=1/4，q=8，KV=4096。

权重种子仅依赖 model/layer/tensor，与 shape workload、stage、候选、框架无关。同一个 workload 的 input/cache 种子也不依赖 stage、候选、框架。因此扩图、换框架、同模型换 shape 不会不知不觉换一套权重。

当前 v3 manifest 内嵌 pinned config 的原始 JSON 文本，并逐 case 校验 SHA256，执行不依赖临时下载缓存。逐框架保存第一层全部权重、完整 RoPE 表的全量 SHA256；输入/历史 KV 另记录至多 8192 个整数等距索引的采样 SHA256，不能称为全状态哈希。新增 analyzer 比较相同 group 的这些指纹；不一致时明确列为跨框架数据不可比，而不是继续把 speedup 混合分析。

Mixed 请求采用 `[KV, KV/2+1, KV/4−1, KV/8+3]` 循环且不短于 q；例如 KV=4096 的四请求是 `[4096,2049,1023,515]`。与对应 uniform 组保持相同 input/weights/past-cache 数据，只改变 request metadata 和正确的 RoPE 位置，从而观察长度分布造成的影响。Triton streaming attention 的循环按实际 request length 结束；native FlashInfer 的 page indptr/indices/last-page-length 按实际长度规划。unused/padding cache 区也逐 bit 检查保持不变。长度组合是 controlled workload，不伪称来自 production trace。

## 每个数如何产生

`train_samples_ms`：CUDA events 包住完整图的实际执行，包括 producer、contiguous、scatter、consumer、norm、残差、MLP；JIT/plan/参考解/验证不计时。Python 逐算子 dispatch 的空隙可能进入区间；这不是去掉 launch overhead 的纯 kernel service time。

`train_median_ms`：上述样本的中位数。用于选择方案，**不拿它作最终加速证据**。

`S_eps(H)`：小图中每个已测正确候选，满足 train median <= `(1+epsilon) × best measured train median`。默认 epsilon=0.03。近优集合只是当前已测集合的近优集合，大图不能称全空间近优集合。

Frozen：在扩图候选中，旧轴 assignment 属于 `S_eps(H)`；新轴不冻结。Free：同一个最终候选集合内不冻结旧轴。分别只按 training median 选方案。

Holdout：两方案交错 AB/BA，各重复执行完整图，默认 full 为 5 对重复，每对内 10 次 events，取中位数。最后再检查完整正确性。没有利用 holdout 重新挑赢家。

`speedup_frozen_over_free = measured_frozen_holdout_median_ms / measured_free_holdout_median_ms`。

`ER_set = speedup − 1`。分子和分母是实测耗时，中位数/比值/ER 都是计算得到的统计量。大图的此值是 **best-measured-set regret**，不是严格全空间 `ER_set`。两方案相同时仍记录噪声，但不列为正例。

原始 worker 的 95% interval：对同进程的 paired ratios 做固定种子 bootstrap（2000 次），对应的是 `median(frozen_i/free_i)`，**不是** point estimate `median(frozen_i)/median(free_i)` 的区间。新增 `audit_rq2_holdout_statistics.py` 保留原数据，从原始配对耗时同时重算这两个不同统计量；ratio-of-medians 的新 bootstrap 必须对两侧使用同一个配对重采样索引。正式判断优先查看新报告 `RQ2_HOLDOUT_ESTIMAND_AUDIT_CN.md`，不能将旧 paired-ratio 区间直接附到不同统计量上。

两种区间都只是同进程重复，不是独立进程/跨机器置信区间；lower bound >1.03 且方案不同只能列为待复现的候选正例。大批 case 的多重比较、独立进程重复和 A/A 噪声都未完成，不能强判假设成立。

`old_near_set_survival_fraction`：扩图近优方案投影到旧轴后，与小图近优集合相交的比例；`minimum_old_axis_changes_to_expanded_near_set` 是两个集合间最小 Hamming 差异。它们不是传播距离，也不是 `K_epsilon`。

## 正确性阈值与第一轮 smoke 修复

单算子：独立 FP32 arithmetic 后按语义边界 round FP16；要求 finite、NRMSE <=1%、逐元素 allclose rtol=3%、atol=max(1e−5, 4 × 2^-10 × reference RMS)。`2^-10` 是 FP16 machine epsilon（1 附近相邻可表示数的相对间隔），不是通常定义的 unit roundoff `2^-11`；旧文字的命名已修正，实际测试阈值未改。一旦失败候选不计性能证据，整个 smoke 不通过。

整图：独立 `independent_graph_reference` 不调用 adapter 或实际 `Group.run`，对最终 frontier（图切口上的有效输出、residual tuple）与各层新 KV 作全图数值对比；所有实际 intermediate 另在产生后立即通过独立的局部 FP32 oracle 检查。FP16 边界误差会随层数传播，因此全图绝对阈值按路径上 operations 的平方根增长；**1% NRMSE 门槛不放宽、逐算子严格参考检查必须同时通过**。全部历史/填充 KV 和实际新 KV 写入使用 bitwise gate，不能用浮点宽容差掩盖错 slot/stride。

这不是“保存全部 intermediate，再逐项与全图参考比较”的实现：那会在大模型 shape 上产生验证器自身的 OOM。现在 serial decoder 层重用四种 GEMM activation backing；在下一层覆写之前已完成全部局部检查，仅保留全图 frontier/new-KV 参考。独立 GQA 参考逐 KV head 将 query groups 合并到矩阵 M 轴，避免复制 Hq 份全量 FP32 KV。两项优化均不缩小模型尺寸、不减少被校验的输出元素。

Native RoPE 的 Q/K norm 输出每次执行新分配，遵循原生 in-place contract，计时中不再无故 clone；局部 oracle 必须 clone，防止参考计算改写待测输入。原生 KV writer 接受任意 token 行 stride，但要求 head×D 行内连续，因此 packed-row V 直接使用，仅 column-output V 发生实际必要 repair。这些决定已经在最新 smoke 中分别验证。

第一轮发现并修复：vLLM public SiLU helper 已变化，应走注册的 `torch.ops._C.silu_and_mul`；TVM 旧 Ladder PYTHONPATH 抢先导入，需要显式加载已安装的新 TVM 源码；二层累计误差不能用一层零附近 allclose 阈值替代局部算子验证。扩展 smoke 发现 32-byte canary 前缀会破坏 batch=1 zero-copy 输入的 128-byte 对齐，现使用 256-byte guard，trace 记录指针对齐。所有失败轮次保持原样，未覆盖。

另外修复大 tensor 采样的 float32 linspace 边界取整风险，改成 int64 整数运算；CPU 回归覆盖超过 2^29 元素的索引边界。不能把这种指纹验证器错误误归因于 layout kernel。

## 运行

在 `/home/liangyilei/ladder_home`：

```bash
# CPU contracts + 六框架扩展 smoke（60 图/框架，uniform/mixed、batch=1/4、extend、8-block）
GPU_PHYSICAL_INDEX=1 \
bash staged/baseline_framework/layout_research/run_rq2_graph_growth_gpu1.sh \
  --smoke --smoke-extended

# 六 adapter 的全部 1,020 图、918 扩图对；默认不限制慢 case 的运行时间
GPU_PHYSICAL_INDEX=1 \
RQ2_WARMUP=3 RQ2_ITERATIONS=10 RQ2_HOLDOUT_REPETITIONS=5 \
bash staged/baseline_framework/layout_research/run_rq2_graph_growth_gpu1.sh --full

# 仅重跑一个框架的一组 shape 的全部 growth stages；group index 按 manifest 顺序 0..101
GPU_PHYSICAL_INDEX=1 \
bash staged/baseline_framework/layout_research/run_rq2_graph_growth_gpu1.sh \
  --full --frameworks vllm --group-index 0
```

每次新建独立目录，已有目录会拒绝写入。保留 source snapshot、SHA256、design manifest、GPU UUID、逐框架日志和 JSONL、process exit status；完成后生成 Markdown 和由实测 holdout 数据作图的 SVG。没有删除旧文件，没有改动/安装原框架环境，没有 push。

推荐 full 增加 `--approved-smoke /绝对路径/已通过的smoke目录`：运行器逐 SHA256 验证六个 GPU worker/helper 与 manifest 和 smoke 完全相同，并检查所有目标框架的退出状态、case 去重和正确候选全部通过。分析器/运行器的非 GPU 修改不改变待测 arithmetic。缺失、失败、代码变化时拒绝开始大规模测量。

截至 2026-10-02，最近完整的 uniform/mixed 扩展 smoke（`results/rq2_graph_growth_smoke_20261002_mixed`）六 adapter 各 60 图，共 360 图、324 扩图对、6152 正确候选；最新 memory-safe/native-contract smoke（`results/rq2_graph_growth_smoke_20261002_native_contract`）六 adapter 各 10 图，共 60 图、54 扩图对、1186 正确候选，错误数均为零，跨框架记录指纹一致。两轮都没有 >3% 且配对 holdout CI 为正的 H2.1 正例；smoke 通过意味着实现/正确性在这些对象上通过，不等于科研假设成立，也不等于六模型全部 shape 已通过。

### 本次已启动的 full（不是“已完成”）

输出：`results/rq2_graph_growth_full_20261002_approved`。六 adapter 按 PyTorch→Triton→CUTLASS→TVM→vLLM→SGLang 顺序串行使用物理卡1；共应执行 6,120 图、5,508 相邻扩图对。CPU 回归随后新增统计量、live-record、representation、匹配 factorial 与 smoke gate 测试，当前 58 项通过；GPU worker/manifest 在 full 开始前与最新通过的 smoke 逐 SHA256 一致。新运行器会自动输出严格统计复核；此次已启动、已冻结的旧 runner 由额外 CPU post-run helper 等待原运行结束后生成复核，不改变正在执行的 GPU 测量。

查看状态（只读，不重新占卡）：

```bash
/home/liangyilei/conda/envs/cuda-opt/bin/python \
  /home/liangyilei/ladder_home/staged/baseline_framework/layout_research/check_rq2_graph_growth_progress.py \
  --run /home/liangyilei/ladder_home/staged/baseline_framework/layout_research/results/rq2_graph_growth_full_20261002_approved

# 已完成旧运行的独立统计复核：output 必须是尚不存在的新目录
/home/liangyilei/conda/envs/cuda-opt/bin/python \
  /home/liangyilei/ladder_home/staged/baseline_framework/layout_research/audit_rq2_holdout_statistics.py \
  --run /绝对路径/已完成的旧结果目录 \
  --output /绝对路径/新的统计复核目录
```

`all_declared_H2_1_full_matrix_passed` 只有六框架每个 declared case 都有去重后的正确图记录、所有候选正确、完成记录与成功退出状态时为 true；仅某几个框架或 smoke 完成不会触发。`all_RQ2_requirements_complete` 仍为 false，因为后续科学要求确实未完成。

运行器自身也冻结到临时脚本并保存 snapshot，防止正在运行时编辑脚本使 Bash 的后续读取失效；/tmp 的同用户 card1 lock 避免两份新 runner 重叠占卡。Triton 实际编译的 GEMM/norm/activation/attention/append 特化分别保存 TTIR/TTGIR/PTX、编译 hash、寄存器与 spills metadata；TVM 保存 TIR；CUTLASS 保存 SASS。原生库未公开的内部 tactic/寄存器布局不能因此说已全量抓到。

### 必须补的后续验证，不能遗漏

1. source catalog 的原生 PR/base-vs-head replay；FP16/SM86 不适用项明确说明，149 个 dtype contract 待审项不能直接判 N/A；
2. MoE/MLA/GDN/Mamba/sliding/sparse/offload/connector 的真实 forward 拓扑，避免所有正反例来自一个 dense decoder 家族；
3. H2.2：实际合法 domain partition 候选（含 fusion/materialization/backend 固定与 joint-search 分层），真实 `K_epsilon` 曲线，不能拿 layout 轴数或颜色连续段数冒充 domain 数；
4. H2.3：同 shape/同算子 multiset 的真实 branch/join/control-flow motif、固定旧 core 的可控远端干预及距离，不拿 winner flip 冒充因果传播；
5. 原框架 default decision/IR/tactic 的捕获与同候选空间的 native 整图 oracle；
6. 独立进程重复、kernel-only 或 CUDA graph capture 对照，分离小 decode 的 dispatch noise 与布局收益；
7. 至少多模型家族/多 shape 的显著正反对照之后，再判断是否“多数框架存在”，不得按重复 PR 或相同几何复用计数提高支持率。
