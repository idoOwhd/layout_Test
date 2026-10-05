# RQ2：FP16、真实子图逐级扩大后的整图多 layout 决策——源码审计与实验改造方案

日期：2026-10-01（Asia/Shanghai）。本文件是**分析与设计阶段**的交付，不是新增 GPU 性能报告。

## 1. 这次真正要测什么

目标不是继续增加 `KV write → head scan/token scan` 的次数，而是回答：

> 从真实模型的一个较小 forward 片段开始，逐步加入实际相邻算子、真实分支与后续 decoder block；在不同 shape 下，整图的多个 tensor layout、共享/拆分域以及转换位置如何共同变化？小图上近优的决策何时不能继续用于大图？

“图大”必须拆成两个不同概念：算子/边更多，与 tensor 的元素更多。二者不能混为一个变量。图更复杂也不必然要求更多 layout，更不必然产生加速；需要找出变化的条件、阈值和负例。

所有新增图必须能回指真实模型 forward / 原框架 kernel 合同。不能用随意拼接的链、伪造 fanout、任意混合不同模型的 state，来制造“图复杂了”的正例。

本轮完成：完整共享讨论归档、全部引用去重和一手源码/原测试入口清单、旧代码审计、整图实验规格，以及第一批真实 Qwen3 嵌套图的 CPU 设计 manifest。**新的逐框架 GPU executor、layout 搜索、计时与整图正确性验证还未完成或运行。**

## 2. 共享链接已完整读取，但讨论中的例子不能不加区分地照搬

来源：[完整 RQ2 讨论](https://chatgpt.com/share/6abe41cd-b72c-83ec-881b-57a4602d51d7)。

- 去除同一消息的重复流后，归档保留全部 5 条 user 消息、5 条 assistant final 回答；不保留内部 reasoning/tool 流。
- 原始抓取 HTML 的 SHA256：`0b7b2e106e05a04b196980a08b0b95e92ad6606c1169fd317c355199bfac7cc8`。
- 139 个显式 PR/issue 链接；另有 68 个仅以 `#编号` 出现的引用，已通过章节归属、仓库前缀和 GitHub primary endpoint 核实。
- 按 `lowercase(repo) + number` 去重后：**207 项，11 个仓库，193 个 PR、14 个 issue**。不是 207 个独立机制或已测性能正例。
- PR 快照：44 merged、130 open/unmerged、19 closed/unmerged。未合并 PR 的行为不能当成安装版默认行为。
- 193 个 PR 都有变更路径/patch 记录；connector 中有 3 个文件没有文本 patch，不能宣称获得了每个文件的完整可执行源码。正式复现仍须取 pinned revision 的完整文件。

详细材料：

1. `ref_talks/shared_6abe41cd_rq2_full_visible_deduplicated_20261001.md`：完整可见讨论。
2. `rq2_shared_github_catalog_20261001.json`：139 个显式引用的全部上下文。
3. `ref_talks/rq2_source_audit_20261001_139/`：原始 primary metadata、diff、补充编号与 patch；目录名称的 139 是初始显式引用数，不是最终总数。
4. `ref_talks/rq2_fp16_test_audit_20261001_v3/RQ2_ALL_SOURCE_TEST_AUDIT_CN.md`：207 项逐来源原测试/benchmark/example 路径、原状态、SHA、dtype 提及和限制。
5. 同目录 JSON/CSV：包含 test definitions、原文已有命令、全部原始标题/描述和图机制候选，方便筛选。

这份来源账本保守地把 **149 项保留为 `pending_fp16_contract_review`**；其余项有明确的已审限制或合同测试判断，但也都不是新增 GPU replay success。关键词出现 FP16 不等于目标 strided kernel 已支持 FP16，测试文件出现也不等于在 A10 上可执行。

审计工具早期生成的 v1 没有正确解析部分历史相对 diff 路径；v2 修正路径解析；v3 又细化了 MiMo、repage 和 padded-router 的 dtype 判断。旧版本不删除、不覆盖；**本分析以 v3 为准**。这不涉及改写历史 GPU results。

## 3. 完整讨论各部分的取舍与纠正

| 讨论内容 | 本次采用的结论 | 对新增实验的影响 |
|---|---|---|
| 共享一个 layout 与拆分 layout domains | 保留，但不能只在一块 KV 的两个扫描 consumer 上测 | 多输入、多输出、不同生命周期的真实整图 |
| 从小图最优推广到大图 | 是此次主问题；小图存在多个近优方案时不能只冻结一个任意 winner | 冻结小图近优集合，搜索其在大图的最佳合法延伸 |
| graph expansion / operator heterogeneity | 保留，但来源必须是实际 forward 中的相邻区域 | 实际 QKV/QK norm/RoPE/attention/O-proj/MLP/block 的嵌套片段 |
| connected components / inference propagation | 不只属于 KV allocator；compiler/kernel 内也存在耦合域 | Triton/TileLang/TVM/CuTe 均需记录各自实际决定的层级 |
| sparse data + top-k/scale metadata | 联动机制重要；原 FP8/FP4 scale 实验不能自动改称 FP16 | FP16 data + 整数路由 metadata 可做独立类比；量化原实现单列 |
| hybrid recurrent/KV、prefill/decode、speculation | 保留 state/lifetime 与 phase 轴 | 真正执行 state update、rollback/reuse；不把 consumer 权重叫 acceptance rate |
| GPU/host/L3、distributed transfer | 分开核查硬件、是否需要多卡 | 单卡 host-copy 可测；分布式原实验暂不运行，不以单卡模拟冒充 |
| MoE W13/W31 / activation / dispatch / combine | FP16 的激活/路由/中间矩阵仍值得测 | 原量化权重/scale PR 与 FP16 等价激活图分开标记 |
| GMEM/SMEM/register propagation | 保留，但硬件支持与真实指令合同优先 | A10 不测 TMA/TMEM/Rubin 原指令；不要求人为制造不合法 layout |
| PR 作者报告 speedup | 只作为选择验证对象的线索 | 原 GPU、dtype、分母、图范围、base/head 逐项记录，不能当本机实测 |
| 旧 fanout 实验很弱 | 同意其不能覆盖新版整图问题，但不是完全无价值 | 保留为最小基线/负对照，不删除或改写历史结果 |

特别需要纠正的来源解读：

- Triton [#11117](https://github.com/triton-lang/triton/pull/11117) / [#10987](https://github.com/triton-lang/triton/issues/10987)：原 reproducer 是 FP32 division→reduction。4128→64 是 PTX division 指令数；257s→4s 是编译耗时，**不是 GPU 延迟下降 64 倍**。
- FlashInfer [#5405](https://github.com/flashinfer-ai/flashinfer/pull/5405)：原 fused QK norm/RoPE/KV append 明确是 BF16/FP8、SM90/SM100。不能在 A10 上简单改 dtype 然后称原 PR FP16 复现。
- SGLang [#41944](https://github.com/sgl-project/sglang/pull/41944)：XPU、FLUX.2 DiT、BF16；不是本轮 CUDA FP16 LLM 性能证据。
- vLLM [#59112](https://github.com/vllm-project/vllm/pull/59112)：新增 FP16 测试主要是 manager/cache 几何合同；新增 GPU packed/dense attention parity/replay 为 BF16。原 B300/DFlash 吞吐数据不能当 A10/FP16 结果。
- vLLM [#58798](https://github.com/vllm-project/vllm/pull/58798)：已有 FP16/BF16 写入正确性测试，但 prep+attention 性能 benchmark 固定 BF16，并排除 projection/metadata/serving。它非常适合作为逐级扩大图的来源，但必须补 FP16 benchmark。
- vLLM [#57823](https://github.com/vllm-project/vllm/pull/57823)：新增 padded-router regression 固定 BF16；普通 tests 出现 FP16 并不能证明这个 padded case 已测 FP16。
- TileLang [#3176](https://github.com/tile-ai/tilelang/pull/3176)：原 H100 FP32 reproducer 与未校准 heuristic；静态 register 数不能直接当峰值 live registers/occupancy。
- AITER 是 AMD/ROCm；TVM Adreno/WebGPU、CUTLASS SM90/SM120/Blackwell、TRT-LLM Rubin locality 和量化示例都须单列设备/dtype 边界。

## 4. 旧 RQ2 实验实际验证了什么，有什么缺口

### 4.1 旧结果的有效范围

已有报告：`results/rq_adequacy_full_20260927_113444/RQ_FULL_PER_RQ_STRICT_ANALYSIS_CN.md`，L-RQ2 段。

256 个 CUDA attention 形状、7 个 fanout 配比中：raw winner 改变 199/256；按旧 3% 判据 robust winner 改变 3/256；`split_NHD_HND_with_conversion` 未成为任何 fanout 的最优候选，0/256。固定策略/逐 fanout 最佳候选的中位时间比 1.0002，最大 1.0128，达到 1.03 的 case 为 0/256。

这些数值是**旧报告的数据**，本轮没有新 GPU 测量。这里的“最佳”仅限旧实现提供的三个策略，不是所有硬件合法布局的全局最优。报告没有新版 ER_set 的 held-out 检验，因此不能据此断言复杂图中的 RQ2 不存在。

合理结论是：在旧 NHD/HND、扫描 consumer、fanout≤8、A10、所提供策略范围内，没有看到显式 split 的性能优势。这是必要的负对照；不能检验真实 speculation、hybrid state、完整 MLP/MoE 或多 tensor 域。

### 4.2 按代码定位的缺口

| 现有文件/入口 | 当前实际做法 | 对新版 RQ2 不足 | 修改方式 |
|---|---|---|---|
| `rq_llm_layout_bench.cu:503` | 三个策略：common NHD、common HND、NHD→HND split；重复两个 scan | 单一 tensor 域；非完整 attention；无反向转换/合法替代域；不沿真实 forward 扩展 | 保留 baseline；新增真实图 executor，所有合法候选与完整图计时 |
| 同文件 fanout correctness | consumer 反复写同一个 `reductions`；pipeline 输出 error 写常量 0 | primitive check 不等于所有 fanout 输出、state、alias 都正确 | 每个真实 branch 结果可检查；whole-allocation guards；真实误差字段，不接受硬编码 0 |
| `triton_kv_layout_bench.py:217` | common-layout fanout + QK score/token scan | 没有 split candidate、softmax/PV/O-proj、graph-domain 搜索 | 注册逐级真实图；布局向量与转换选择；IR/pass snapshots |
| `tvm_rq_observation_bench.py:235` | 已物化输入的 row/column consumer module | producer/materialization 不计时；只核对第 0 个 fanout 输出 | 从统一 logical inputs 运行全图；逐输出校验；Relax/TIR 决策分别记录 |
| `build_rq_llm_cases.py:101` | attention 投影后的尺寸，部分请求轴展平 | 多结构被映成同一扫描对象，B 不代表真实 request topology | 显式保留 B/q/kv、indptr/block-table、每个 request 的边界 |
| `native_framework_subgraph_probe.py` | 按多个 RQ 打标签的 native operator probe | 标签不能证明多域/整图反事实已运行 | probe 降级为 supporting；新增 native graph adapters |
| `native_kv_request_batch_probe.py` | vLLM/SGLang native KV writer、batch sweep | writer-only，不是 writer+真实 consumer+全图策略 | 写入→native attention→下游投影，并比较受控合法 layouts |
| `analyze_v13_rqs.py:273–326` | measured primitive + 权重/开销公式 | `equal_vote` 实际是 token-only proxy；acceptance 是权重；overhead 是 factor×conversion | 保留为 model exploration；新 runtime hypothesis 必须独立实测 |
| `analyze_rq_full_per_rq.py:60–92,161–190` | 合并 samples、同批挑 winner、固定策略/逐 fanout 比值 | 缺少新指标、候选完整性、independent process 标识、近优集合及 held-out 选择 | 新 analyzer fail-closed：缺一候选不能补“oracle”；选择与评价数据分开 |
| `v13_experiment_registry.json` | compiler/CUTLASS RQ2 标 N/A，原因是“不拥有 serving allocator” | 对旧 allocator 问题合理，但不能排除 kernel/编译图内 representation domain 问题 | 版本化旧 H2 与新版整图扩展指标，不覆盖原 registry 语义 |

源码账本中 `COMPLETE_STRONG` 指 source-rule/proxy/fallback 链条资料完备，不等于“所有框架已经执行了所有新 RQ2 GPU 实验”。

## 5. RQ2 的严格测量定义

### 5.1 一个 layout 决策不是一个字符串

每个候选 x 至少包含：每个 tensor 的 physical shape/stride/storage offset；data+metadata 的映射；memory level；alias/storage owner；domain 分区；conversion/materialization 的位置；必要的 kernel contract。若联动改变 fusion/tactic/page/backend，还要把这些变动显式记录。

domain 的含义：一组 tensor/use 共享可兼容的物理表示与所有权约束。不同逻辑轴名不一定是不同物理 layout；view/reshape/transpose 如果只是地址映射的别名，不能自动按多域计数。必须给出统一 logical-index→physical-address 映射。

同一框架内部比较的候选必须有相同数学计算、mask、state 更新和有效输出。跨框架只比较共同语义；不把 scan、QK-only、完整 attention 的时间混在一起。

### 5.2 小图近优决策在大图的最佳延伸损失：主指标 ER_set

H 是 G 的真实嵌套子图，旧节点 ID、数学、dtype、shape 不变。`T(G,x)` 是候选 x 的**直接实测整图时间**。

`S_eps(H) = {s | T(H,s) <= (1+eps) * min_z T(H,z)}`。

`ER_set(H→G) = min_{x in X(G), projection_H(x) in S_eps(H)} T(G,x) / min_{x in X(G)} T(G,x) - 1`。

- 分子：冻结小图近优集合中的旧决策，允许新增部分选择**最佳合法延伸**，得到的大图时间。
- 分母：同一大图在已声明完整候选集中的最佳时间。
- 这两个时间都必须实测；除法与减 1 是计算出来的指标。
- 单个任意小图 winner 的 regret 另报，不能取代 ER_set，否则 near-tie 会制造假正例。
- 不允许故意给 frozen 小图接一个很差的转换路径，然后称全图规划获胜。
- frozen 无合法延伸时：`legality_flip`，没有性能 speedup；禁止以 ∞ 或 0 时间代填性能统计。

eps 初始取 3%，还要参考 A/A 噪声和置信区间。若图太大未穷尽候选，分母只能叫 `best_measured_candidate`，不能叫 exact oracle。

### 5.3 还需要测什么

- `K_eps(G)`：在完整候选集中，达到 `(1+eps)*最佳时间` 所需最少 domain 数。不是“最快方案恰有几种 layout”的简单计数。
- `layout_change_count`：大图最优相对 frozen 小图，旧 tensor 的物理决策改变数量；同时报告能保持旧近优决策的最佳延伸，不任意取 tie。
- `decision_influence_distance`：新增约束离被改变旧 tensor 的实际图距离；loops 要按真实迭代依赖处理。
- `conversion_count / bytes / measured_ms`、duplicated storage、peak live bytes、实际 allocation 数和 memory budget。
- `default_regret`、`producer-local/greedy regret`：原框架默认、局部启发式相对整图最佳候选的直接时间比。
- 每 node/edge 的局部时间作解释；完整图时间作为主结论。不同 primitive 的 median 不能直接相加冒充一次实测。
- 编译时间、kernel 时间、host/metadata 时间、端到端 wall time 分栏；GPU events 不覆盖 Python scheduler/host allocator。

若解除 frozen 同时改变 fusion/backend/kernel count，首先报告“联合策略收益”；需要额外固定这些轴的 layout-only ablation，才能归因到 layout。

## 6. 第一批已整理的真实嵌套子图：不是人为 DAG

新增构造器：`build_rq2_real_graph_growth_cases.py`。

读取 pinned configs，以及安装版 vLLM `Qwen3Attention.forward`、`Qwen3DecoderLayer.forward`、`Qwen2MLP.forward` 的完整方法；校验关键调用，保存文件/方法 SHA256、行号。同时记录 SGLang Qwen3 native prepare/attention forward 的来源；SGLang 的 communicator、实际后端/state write 仍要在 executor 阶段继续核实。

生成文件：`ref_talks/rq2_qwen3_real_forward_growth_design_20261001.json`。

**6 个真实模型 × 12 个受控 workload shape × 10 个实际 forward 范围 = 720 个图合同，648 个相邻嵌套图对。** 这些是设计对象，尚不是运行结果。dense Qwen3 族不能代替 MoE/MLA/hybrid/sparse/offload 的结构多样性。

| 范围 | 实际新增内容 | 新出现的决策关系 |
|---|---|---|
| G0 | input RMSNorm → packed QKV projection | norm 输出/矩阵输入，packed 输出 stride |
| G1 | QKV split views、Q/K per-head RMSNorm | 一个 packed producer 的真实三路 uses；view/物化与 head-local reduction |
| G2 | Q/K RoPE | 双输出、位置 metadata 与之前的 Q/K layouts |
| G3 | 实际 causal GQA + KV update | Q、K、V、持久 cache 与 request metadata 的联合合同 |
| G4 | O-proj | attention output→矩阵输入边界反向影响 |
| G5 | residual/add-RMSNorm → gate/up projection → SiLU×up | 真实 residual 分支和 join、不同算子约束 |
| G6 | down-proj，完整第一个 decoder block | 两段矩阵 contraction 与 activation 的联合域 |
| G7 | 原模型的第二个 decoder block | block 边界、residual tuple、前后布局与生命周期 |
| G8 | 前四个真实 decoder blocks | 更多 tensor、不同复用与存活区间 |
| G9 | 前八个真实 decoder blocks | 搜索规模/内存限制与决策传播距离 |

注意 vLLM 的 decoder 返回 `(mlp_hidden, residual)`，下一层才继续 fused add/norm；不人为在 block 尾插入一个不存在的 residual add。

模型几何从原 config 读取，不把 Hq×D 强制等于 hidden：

| 模型 | hidden | Hq/Hkv | D | intermediate |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 1024 | 16/8 | 128 | 3072 |
| Qwen3-1.7B | 2048 | 16/8 | 128 | 6144 |
| Qwen3-4B | 2560 | 32/8 | 128 | 9728 |
| Qwen3-8B | 4096 | 32/8 | 128 | 12288 |
| Qwen3-14B | 5120 | 40/8 | 128 | 17408 |
| Qwen3-32B | 5120 | 64/8 | 128 | 25600 |

12 个 workload：decode `B∈{1,4,16}, q=1, kv∈{512,4096}`；prefill `B∈{1,4}, q=kv∈{128,1024}`；extend `B∈{1,4}, q=8, kv=4096`。

模型结构/几何是 real-world 配置；长度、batch 是合法受控 serving workload，**不是已采集生产 trace**。未下载大模型权重；后续用 seeded tensor 验证时须明确“真实结构+合成数值”，不能冒称完整预训练模型推理。

构造器保留 request 数、每请求长度、metadata 边界和 cache state-before/state-after，不因 activations 展平为 `[B*q,H]` 就丢掉 batch 语义。每次相邻扩大都检查旧节点 semantic signatures 完全一致。

### 6.1 进一步扩大范围的真实来源，不以伪造 fanout 代替

| 真实来源族 | 逐级扩展路径 | 必须额外核查的合同 |
|---|---|---|
| MiMo partial RoPE / DiffKV，vLLM #58798 | packed QKV → partial RoPE/value scale → packed cache write → full/SWA attention → O-proj/实际 block | Q/K D=192、V D=128、rotary=64、packed padding；真实 attention sink/window；原 benchmark 为 BF16 |
| vLLM #44454/#44455/#44456/#44458/#59112 | cache 配置/分组 → allocation views → kernel re-page metadata → update → attention → copy/offload | 不把 manager block size 与 kernel page size 混成同一物理布局；FP16 GPU parity 需补 |
| SGLang #40326/#40327/#38592/#40328/#40329/#40330/#40331 | stride 正确性 → unified paged view → writer → native attention/state → translator →单卡 host write-back | 七个依赖补丁是一组链条，不是七个独立正例；统一池与原池对照、别名与 canary |
| 真实 DeepSeek MLA / sparse 模型 | latent projection + RoPE → cache/index metadata → gather/sparse attention → O-proj | latent/nope/rope/V 维度分开；top-k metadata 为 int；原 FP8/Blackwell 源码不冒充 FP16 |
| 真实 Qwen3 MoE / DeepSeek MoE | router logits → top-k → dispatch/permutation → expert GEMM1 → SwiGLU → GEMM2 → combine | FP16 激活、整数 route metadata、相同 expert 权重/route、真实 padded rows；单卡不声称 A2A 网络收益 |
| 真实 Falcon-H1/Granite/Nemotron 等 hybrid | 实际 attention/SSM 层 → recurrent state update →真实下一层/共同池 | 必须按 config 中真实 layer pattern 和各 state dtype；不得把 KDA/GDN/Mamba 当同一 state |
| real host KV offload/HiCache | device layout → host representation → restore → native attention | layout tag/ownership、D2H/H2D、页/层拓扑；网络/多卡部分 blocked |

这些后续 recipe 仍须从全部 207 项账本逐条映射；上表是机制组织方式，不是只挑上表几个 PR 就宣布“全部覆盖”。未合适映射的来源保留 pending 行。

## 7. 各框架原测试是什么，新测试需要细到什么程度

| 框架 | 原源码/测试例子 | 原测试能证明什么 | 针对整图增长必须新增的实验 |
|---|---|---|---|
| vLLM（34 来源） | #59112 的 `tests/kernels/attention/test_flashinfer.py`、`tests/v1/core/test_contiguous_kv_packing.py`、`tests/v1/worker/test_attn_utils.py`；#58798 的 `tests/kernels/attention/test_cache.py`、MiMo benchmark；#57823 top-k regression | 视图/几何、replay、cache input-preservation、strided routing 合同；部分 FP16 correctness | 同 native backend 的 G0→完整 block、FP16 GPU numerical tests、writer+attention+O-proj；default/frozen/best 三路及每 tensor trace |
| SGLang（34） | stride hygiene、paged view、unified token-major、masked writer、HiCache、KV canary、translate tests | 新 pool/view/ownership 在目标 path 的功能一致性 | 验证 source-stack 依赖；FP16 全链数值和 state；layer-major/unified 布局的实际完整图、host-copy/metadata 分开计时 |
| Triton（22） | #11117 的 `test/TritonGPU/remove-layout-conversions-10987.mlir`；#11685 的 `test/Gluon/auto_encoding.mlir`；#8398 的 `test/TritonGPU/combine.mlir` | pass/encoding/合法性/指令变化，不是 FP16 整图加速 | 真实 LLM reduction/norm→matmul、attention partials→reducer、reshape/broadcast 链；保留每个 pass 前后 IR，执行 FP16 kernel 和多 kernel DAG |
| TileLang（33） | #1559 `testing/python/issue/test_tilelang_issue_layout.py`；#3176 inference/reduction-aware tests；#3180 fragment slicing/reduce；#1386 GQA decode example | infer layout/fragment inverse/ownership 和部分 example 正确性 | 同真实 kernel 的手工/default/合法穷举；真实 GQA split/reduce 与 QKV/norm/attention 链，测复制、shared/barrier、寄存器及整图 |
| TVM（14） | #18579/#18622/#18638/#18642/#18643 的 `tests/python/relax/test_transform_convert_layout.py`；#20076 Buffer.local physical-order fix | Relax propagation 与 IR 合同；不证明 LLM FP16 完整图性能 | 真实 reshape/gather/scatter/norm/matmul 图在 Relax 的转换位置、TIR 的 tile/layout；producer、转换和 all outputs 纳入计时 |
| CUTLASS/CuTe（8） | #3256 grouped SwiGLU aux TMA example；#3453 FMHA runtime strides；#3030 SM120 FA | 指定架构的 kernel contract/示例；不是 arbitrary graph 自动域划分 | A10 合法 SM80 FP16 GEMM→SwiGLU→GEMM、attention→O-proj；driver 负责逐边搜索，CuTe 只声明它实际决定的 thread/value layout |
| FlashInfer（16） | #5405 experimental fused norm/RoPE/append tests；#4572 GDN stride；#4971 Hopper paged prefill；#5709 MLA reducer；#5094 packed sparse metadata | 特定 dtype/架构的 producer/consumer 合同或 kernel 性能 | 可用 FP16 native attention 接真实 producer/downstream；SM90+ 与量化来源原生 blocked，FP16 analogue 独立记录 |
| FlashAttention（12） | varlen sort/swizzle、paged head256、SM120 TMA、non-contiguous QKV 修复 | scheduler/strides/架构路径的正确性与局部性能 | 卡1可用 FP16 kernel 与真实 projection/consumer；页/varlen/stride 都与同一 logical batch 对照；高 SM kernel 原生 blocked |
| TRT-LLM（15） | locality domain、KV connector、mixed KV、DP/disagg 原 tests | 引擎/kernel/transfer 的特定合同，多项依赖新架构/量化/多卡 | 只测单卡 FP16 支持路径；unsupported PR 保留状态；不能用 PyTorch 结果填 native engine 成功 |
| DeepGEMM（8）/AITER（11） | compact/grouped GEMM、swizzle/descriptor lifetime、AMD state与MoE | 多项 FP8、Hopper/AMD/网络相关机制 | 按 native dtype/硬件标记；FP16 对照是 analogue，不是原项目 FP16支持证明 |

注意：vLLM/SGLang 是 serving engine，不是接受任意 DAG 并自动搜索所有张量布局的通用编译器。可通过它们的**真实 native operator / allocator / model-forward path**构建可测子图；新增驱动的 search winner 要叫“实验整图规划结果”，不能冒称框架原本自动做出的决定。

Triton/TileLang/CUTLASS 的 kernel 内 layouts 与 engine 的 global KV storage 是不同层级。统一的是数学计算和报告字段，不是要求每个框架拥有同样的 API。Hexcute 尚未在此讨论提供可执行 native source 映射，不能以 CUDA reference 填成功。

## 8. 如何从测试中发现规律，而不只是列出 speedup

### 8.1 必须控制的轴

1. **真实范围增长**：G0→G1→…，同一次增长的旧 tensor shape/dtype/math 不变。
2. **tensor 大小**：B/q/kv、Hq/Hkv、D、hidden/intermediate；每个形状单独重复同一增长过程。
3. **拓扑**：实际 QKV 三路、residual fork/join、实际 route/combine、跨层 cache/state uses；不增加不存在的 branch。
4. **算子种类**：从投影逐步加入 per-head norm、RoPE、attention/reduction、投影、elementwise/MLP；op kind 按语义和 lowered kernel 分开计数。
5. **存储合同**：contiguous、真实 packed views、padding/storage offset、paged cache、data/metadata；“改变一项”与“native 联合改进”分组。
6. **持久性**：一次 forward、真实多步 cache/state 更新、同请求复用、多请求 ragged batch；不把重复同一 scan 等同多 request。
7. **资源**：peak live bytes、register spill、shared usage、launch 数、conversion bytes、metadata/host成本；allocation 数要实际测，不能 factor 模拟。

不是每一项都能与其余项完全独立。若不同真实模型的几何/结构同时变化，应做同模型面板比较，明确协变量；不把跨模型相关性直接叫 layout 因果效应。

### 8.2 每个图/shape 至少输出以下策略

- 原框架 native/default；记录实际执行 path 和默认值，而不是仅记录指定的 flags。
- 边界局部/greedy；明确逐节点选择规则。
- best frozen-small-near-optimal extension。
- 全图 unrestricted best（小候选集完全枚举）或 best-measured（较大集合）。
- 固定其他 tactic/fusion/page/backend 的 layout-only ablation。
- 真实 joint policy 的收益另报，不能与上项混为一个结论。
- 同 layout A/A 和低异构/小张量/无转换收益的负对照。

每个 framework、graph、shape 的 search space 都先列合法集合和拒绝原因。小图在约 6–8 个二值决策时可做完整枚举；实际候选总数由合法约束决定。大图可以 beam/top-k/DP，但需在小图上校准；搜索结果不能称硬件 layout 全局最优。

### 8.3 规律汇总方式

不要先假设“算子越多越要拆域”。逐项检验：

- ER_set 是否只在进入某个不同消费者（attention→O-proj、top-k→dispatch 等）时跃迁？
- 在相同真实增长边界下，B=1 与 B>1、prefill 与 decode、短/长 KV 是否不同？
- conversion amortization、有效利用带宽、峰值存活 tensor/寄存器是否解释了布局变化？
- K_eps 是否增加、减少或平台化？同一个物理表示能否跨多个 logical view 复用？
- 被新增约束改变的是相邻 tensor，还是经 view/reduction/loop 传播到更远节点？
- 默认 heuristic 的错误由合法性、遗漏 layout、错误 cost proxy、还是融合/页/allocator 冲突造成？

按模型族/geometry、framework implementation family、增长边界分层报告，不把多个模型别名算成独立结构；vLLM 与 SGLang 若调用同一个 FlashInfer kernel，也不能当成三份完全独立的内核证据。

## 9. 每一个 tensor/算子到底记录什么

拟新增的实验结果目录应包含：

```text
NEW_RESULT_DIR/
  input_contracts.json                # 真实模型、revision、forward source、shape、数学合同
  source_coverage.json                # 每个 PR/issue 独立状态；不静默跳过
  environment.json                  # 物理GPU/SM、包/commit、CUDA、计时模式
  graphs/<graph_id>/semantic_graph.json
  graphs/<graph_id>/lowered_graph.json
  layouts/<candidate_id>/tensors.json # 每个 tensor 的 shape/stride/offset/alias/owner
  layouts/<candidate_id>/domains.json
  layouts/<candidate_id>/conversions.json
  layouts/<candidate_id>/ir/          # compiler pass dumps/PTX/可得的SASS/源码 layout
  correctness/<candidate_id>.json    # 所有 outputs、state、guards、参考算法
  measurements.jsonl                 # process/round/order、直接时间、准确分子分母
  search_coverage.json                # expected/legal/measured/rejected 候选计数
  growth_pairs.jsonl                 # 小图集合、投影约束、最佳延伸、ER_set、CI
  PER_FRAMEWORK_RQ2_ANALYSIS_CN.md
  figures/                           # 真实测量 SVG/PDF；本轮没有新 GPU 图
```

`tensors.json` 要记录 framework/operator/kernel symbol、logical dimensions、physical dimensions、byte strides、storage offset、memory level、producer/consumer node IDs、alias group、domain ID、是否物化、read/write bytes 和 layout 的获得依据。

register/shared layouts：有显式 CuTe/Triton/TileLang encoding 时保存源声明/IR；仅能看到 opaque native library 时写 `not_exposed`，不能从 tensor shape 推导一个不存在的 lane/register 证据。load/store 指令来自实际编译产物，不由布局名字猜测。

## 10. 正确性与计时门槛：不正确的“快”不能计入 RQ2

### 10.1 FP16 的定义

本轮主组：输入、未量化权重、激活、KV 为 FP16；softmax/norm/reduction 按原算法使用 FP32 累计；metadata 保留 int32/int64。不能把整数 page table 转成 FP16。原实现强制 BF16/FP8/FP32 persistent state 时单独报告混合类型/blocked，不叫“纯 FP16 原生实现”。

### 10.2 所有候选逐一检查

- 用同一组 logical inputs、positions、causal/window masks、request boundaries、weights/routes、旧 state。
- 核对全部有效 outputs、每层 KV/state 更新，以及整块 storage 的未写入区域；padding/offset、非连续 slices、负 slot、ragged/空 request 均检查。
- view/zero-copy 路径必须检查原 input 未被修改、owner/lifetime 与 replay 实际读取的 storage；clone 正确而原 pool 没更新是失败。
- 不仅看第 0 个 consumer；不写硬编码 max_error=0。
- 纯搬运/索引映射用 bitwise；浮点用定义清楚的 max-abs、relative/NRMSE 与 float32 参考，tolerance 与算子/长度对应。不能只用统一绝对 0.5。
- top-k 控制 ties，索引一致性和权重误差分开；加入错 stride/错 page/错 RoPE/alias 破坏的负测试，确保 checker 真能发现错误。
- 数值/alias 检查与计时分开；debug intermediate dumps 不加入正式计时。原图不存在的 observer/copy 不能人为加入测量路径。
- stateful 路径明确 cold state、steady update、cache replay；每个候选开始时 state 一致，不能前一候选留下的数据帮后一候选。

### 10.3 统计与运行

- 独立进程初期 ≥5，条件允许 ≥10；随机/交替顺序，A/A 噪声检查。
- 编译/初始化/warmup 完成后计 GPU；包括完整实际 graph calls。host metadata/allocator 用独立 wall-clock，总体端到端另报。
- 小图候选选择和大图评价使用分离的 repetitions/rounds；bootstrap CI 的单位是独立 process/round，不是把同一 CUDA event 的 iterations 当独立样本。
- frozen 策略和 unrestricted 候选在相同 cache/timing 环境下 paired 测量；分别报告 eager、CUDA graph/replay，不混合取最小值。
- 缺少某候选、OOM、compile fail、数值 fail、timeout、unsupported hardware 各记独立原因；不能用缺失值制造 split/默认策略优势。
- timeout 给足编译时间；只用 physical GPU1。新结果目录必须是新路径，存在则拒绝覆盖。

## 11. 与 v13 H2.1–H2.5 的映射，避免静默改变旧假设

| 原假设 | 原实验不足 | 新的实际验证 |
|---|---|---|
| H2.1 consumer heterogeneity 下 split 获胜阈值 | 旧三个策略/两个 scan 过窄；winner变化不等于 split 收益 | 真实多算子图与合法 domain alternatives；K_eps、shared/split、ER_set 全图实测 |
| H2.2 time/byte-weighted 比 equal voting 更好 | 旧 token-only proxy 被叫 equal_vote；公式内比较不是 held-out native 决策 | 从原框架源读取真实选择规则，训练代价模型，在未参与拟合的 shape/真实图上检验 regret |
| H2.3 低异构/低复用利于 shared | 旧负例有价值，但来源有限 | 保留旧负例；真实无 view copy/同 consumer 合同/launch-bound 图，允许 shared 继续胜出 |
| H2.4 acceptance rate 改变 target/draft/verify 分区 | 改 consumer weight 不是执行 speculation | 实际 draft/target/verify 合同、accepted mask 与 rollback/update 工作；受控合法 trace 与模型实际接受率分开，不冒称质量实验 |
| H2.5 hybrid KV/recurrent overhead 阈值 | `factor * conversion_ms` 不是实测 allocator/state 成本 | 真实 hybrid forward/pool pattern；实际 allocations/live bytes、translator、state write/read、host/metadata 计时 |

新增 graph-growth 指标用 `RQ2-GROWTH-*` 名字，与旧 hypothesis IDs 并存，不重写历史结论：

- G1：真实相邻子图扩大后，旧近优 layout 决策可能不再有近优延伸。
- G2：这种变化取决于新增算子的表示/生命周期合同，不仅是节点数量。
- G3：shape/reuse/resource 条件控制 shared/split/zero-copy/materialization 的阈值；允许负例。
- G4：编译器的非局部传播与 serving 的跨层/state 表示，在不同层级体现同类耦合，但不能把所有 source 都视为同一个优化器。

这些是待检验假设，不是本轮得出的性能结论。

## 12. 实现顺序与完成度

已经新增：

- `archive_layout_shared_discussion.py`：完整 visible stream/引用去重；bare-number 归属校验。
- `build_rq2_source_test_audit.py`：offline primary source/原测试/dtype 限制清单，拒绝覆盖。
- `build_rq2_real_graph_growth_cases.py`：第一批真实 Qwen3 forward-growth 设计合同；拒绝覆盖；不导入框架、不下载权重。
- `test_rq2_source_audit.py`：CPU 测试，覆盖去重、错误仓库归属、BF16/FP16区分、历史相对路径、输出防覆盖、实际 head_dim、旧节点保持、request/state版本和“设计不是实测”的状态。

本轮 CPU 测试结果：14/14 通过。它们验证来源/合同构造和状态标注，**不验证 GPU 算术正确性，不产生 layout speedup**。连接检查也已通过：工作目录可读、GitHub primary endpoint 可用。

尚需实现，不能声称已完成：

1. 每个 source 的完整 FP16/native 合同 review（149 项 pending，另外受限制条目也有单独待执行项）。
2. Qwen3/MiMo 逐级 graph executor 与 baseline/frozen/default/search adapters。
3. MoE/MLA/sparse/hybrid/offload 的真实嵌套 recipe 与独立参考算法；dense族不能代替。
4. native compiler passes/layout exports、engine cache/state trace、IR/PTX证据。
5. ER_set/K_eps 候选覆盖与 held-out analyzer；联合收益与 layout-only 归因。
6. 所有输出/state/alias guards 的 GPU correctness smoke。
7. 单卡 smoke 通过后再给 full GPU1 运行命令；本轮**不把 design builder 叫 GPU 测试命令**。

不会删除旧 baseline、旧 RQ1 新增文件、模型、源码或 results，不会自动 git push/改变环境，不会覆盖旧报告。

### 12.1 本轮 CPU 检查命令（不是性能实验）

```bash
cd /home/liangyilei/ladder_home/staged/baseline_framework/layout_research
/usr/bin/python3 -m unittest -v test_rq2_source_audit.py
```

重新生成设计合同需要指定**新文件**；同名会拒绝执行：

```bash
cd /home/liangyilei/ladder_home
/usr/bin/python3 staged/baseline_framework/layout_research/build_rq2_real_graph_growth_cases.py \
  --output /home/liangyilei/ladder_home/staged/baseline_framework/layout_research/ref_talks/rq2_growth_design_$(date +%Y%m%d_%H%M%S_%N).json
```

### 12.2 旧 RQ2 入口保留，不代表新要求已经满足

旧 `run_rq_adequacy_gpu1.sh --full --rq L-RQ2 --output NEW_DIR` 仍是旧 runtime/supporting/model 实验路线。后续应新加 graph-growth runner，并在 source coverage 中明确每条原 PR/native/analogue/blocked，不悄悄把旧入口的成功状态当新整图验证成功。

## 13. 阶段性结论

目前能够确认的是：旧代码主要是局部 KV/consumer/layout 模板，不足以回答用户的**真实图逐级增大、算子种类增加与 shape 对全图多 layout 决策的影响**。源码提供了大量真实表示耦合、stride/ownership、传播与域划分机制，但源码/原 regression 不能代替 FP16/A10 的整图性能证据。

新版需要的是按真实 forward 扩展的多 tensor 反事实实验，并保留 near-optimal 集合、所有合法延伸、正确性、计时范围与硬件边界。720 个设计图/648 个嵌套对是起点，不是“所有框架和所有 RQ2 已充分验证”的完成声明。
