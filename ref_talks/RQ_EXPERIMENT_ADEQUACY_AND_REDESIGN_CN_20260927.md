# v10/v13 layout RQ：现有实验充分性审计与单卡补充设计

本文审计 `layout_summary_v10_*` 与 `layout_summary_v13_*` 的实验，不把“产生了很多行”视为“科学问题已经验证”。范围排除 Hexcute、跨硬件架构比较；真正需要多 rank 的 L-RQ5 保持单卡 blocked。

## 1. 总结

现有 640-case 结果的主要价值是**覆盖真实模型结构和发现框架适用性**。它不能单独保证因果充分性，原因有四个：

1. 640 行并非 640 个独立模型；大量 case 是同一模型/同一张量合同的不同投影，TVM 等入口还会进一步压缩为少量唯一形状。
2. 若一个 RQ 的关键变量只有两个点（例如复用次数 1/64）或根本没有进入计时区（例如用公式改变权重），增加 shape 只是在重复同一种实验。
3. source evidence 证明“框架这样实现”，不能证明“该选择在给定 workload 上性能最优”。代理模型可生成假设，不能替代直接 paired counterfactual。
4. 固定候选执行顺序会把编译缓存、温度/频率漂移和候选本身混在一起；同进程 iteration 也不是独立重复。

因此新增 `adequacy-v3`：1024 个 real-model-derived contracts（8 个结构各 128），但统计单位按真实父模型聚合；CUDA 因果实验用 5 个独立进程、候选/shape 顺序反转；所有直接结果先过 correctness gate。

## 2. 每个 RQ 的审计与改正

| RQ | 原实验是否充分 | 主要设计问题 | 本次改正 | 仍然不能声称 |
|---|---|---|---|---|
| v10 strict L-RQ1 / v13 L-RQ1 跨 edge 排名反转 | 部分 | 核心 shape 少；primitive winner 容易被误写成 complete-edge winner；固定顺序 | 256 个 attention contracts；NHD/HND producer、head consumer、token consumer以及完整 pipeline；5 个独立进程、顺序 seed | native operator slice 不等于每个框架完整端到端最优 |
| L-RQ2 layout-domain 粒度 | 原方法偏弱 | 1:1 双 consumer 无法辨别 fan-out；历史 equal-vote 是 token proxy；权重扫面是公式计算 | 直接计时 head:token fan-out = 1:8、1:4、1:2、1:1、2:1、4:1、8:1；共同 NHD、共同 HND、split+一次 conversion 三者做完整 producer→consumer 比较 | 框架 engine 若没有多 pool/迁移 API，只能证明 operator-level 机制，不声称 engine 已支持该策略 |
| L-RQ3 adaptation amortization | 部分 | 复用点仅 1/4/16/64，可能越过 crossover；历史 overlap/pressure 含公式代理 | 直接复用点扩为 1/2/3/4/6/8/12/16/24/32/48/64；保留真实 HBM contention；转换和 consumer 均在计时区 | 单卡 HBM contention 不是网络通信 overlap；该部分不得升级为 L-RQ5 |
| L-RQ4 稳定性/泛化 | 部分 | 用同一 catalog 行随机切 train/test 会 shape leakage；固定 Qwen serving workload 太少；worst/best span 不是实际 policy | 8 类结构×128；真实父模型最多 8 个；B=1/2/4/8、prefill/decode、tile/page ±1；后续按 `parent_case_id` group-held-out | 当前单卡不能回答跨架构稳定性；硬件项保持排除 |
| L-RQ5 placement/communication inversion | 单卡不充分 | memcpy、HBM pressure 或静态公式不能制造真实 collective、rank ownership、拓扑 contention | 不伪造。清单和报告固定为 `blocked_single_gpu` | 需要至少 2 rank 的真实 collective/placement intervention |
| L-RQ6 持久状态与迁移 | 原方法不足 | 过去主要把 primitive median 代入 cost model；没有真的执行 policy | 直接执行 stationary-token、stationary-head、alternating、phase-drift、bursty、seeded-Markov 轨迹；比较 fixed NHD/HND、eager、hysteresis-2/4，conversion 在计时区 | 尚无稳定公共 API 在 vLLM/SGLang 运行中迁移整个已分配 KV pool；不能把 reference controller 冒充 engine feature |
| L-RQ7 legality/repair | 部分 | 单 consumer boundary 较强，但多 consumer legality intersection 曾由硬编码/构造给出；alias/lifetime 不完整 | 1024 contract 的 boundary runner继续做 correctness-gated view/materialize；报告 fail-closed 要求二/三 consumer intersection | 在二/三 consumer repair 产物未齐时状态仍应 partial，不能因为 581 个单 consumer case 就声称全局完成 |
| L-RQ8 data+metadata co-layout | 不足 | 主要只有 scale 的 NHD/HND；“metadata 重要”被过度泛化到 route/page/sparse | 保留 data×scale 2×2 和 per-head negative control；新增 identity/reverse/permuted block table；非整页也实际分配 | MoE route ID、稀疏 index、zero-point 仍需各自 native 子图，不能由 scale/page table 外推 |
| L-RQ9 candidate completeness | 部分 | `paged_NHD` 可能与 NHD flat address 相同；55 个 stride string 可能只是 extent 不同；candidate 数被高估 | 1024 shape 扩展外部有效性；报告要求按 affine address map 归一化去重；Triton/CUTLASS/TVM 分别保留 native candidate evidence | 未实现的 layout 不是可执行候选；候选全集只相对给定框架/API 成立 |
| L-RQ10 page/allocator/layout coupling | 原方法不足 | 旧 CUDA 只测 `tokens % page_size == 0`，fragmentation 完全由公式算；block table 恒 identity；serving workload 少 | page=1/4/8/16/32/64/128/256；ceil allocation、真实 padding；identity/reverse/permuted table；route stride=1/4/16；vLLM/SGLang 增加边界 prompt 与 concurrency 1/N | prefix hit rate/fan-out 的真实 serving trace仍需单独受控；kernel page winner不等于端到端 prefix-cache winner |

## 3. 为什么测试对象这样扩展

新增对象不是任意随机 shape，而是分层设计：

- 结构：GQA、sliding attention、sparse attention、MLA、SwiGLU、MoE、Mamba2、linear attention，各 128 个合同。
- 父模型：每类尽量选择 8 个来自 pinned catalog 的不同真实 architecture；catalog 只有 sparse=4、Mamba2=6、linear-attention=2 时不虚构模型。
- workload：prefill/decode 各 512；request batch 1/2/4/8 各 256。
- 边界：7/8、15/16/17、31/32/33、63/64/65、127/128/129，以及受显存约束的长 prefill/decode。
- 因果含义：模型维度和 provenance 是真实的；batch/length 是明确标记的 controlled counterfactual，并非宣称模型训练时使用该长度。

1024 个 case 中可能出现少量相同 tensor contract；这些行保留不同模型 provenance，但置信区间先按 `parent_case_id` 聚合，不能把它们当独立样本。

## 4. 每类证据的判定规则

1. **direct runtime**：候选执行相同逻辑工作；差异只在目标 layout 决策轴；完整路径进入 CUDA event；`max_abs_error <= 1e-3`。
2. **framework-native supporting**：调用框架发布的 kernel/operator/API，但若未覆盖完整模型图，必须标记 operator slice。
3. **source mechanism**：只能证明决策规则/约束存在，不能提供 speedup。
4. **modeled/synthetic**：只能用于解释、灵敏度分析和提出 crossover 假设；不能单独把 RQ 判为 supported。
5. **N/A**：框架不拥有该决策层，不是失败。比如 CUTLASS 不拥有 serving prefix allocator。
6. **blocked**：单卡不能构造的通信/跨架构干预保持 blocked。

## 5. 统计与复现

- 每个 CUDA causal case 用 5 个独立进程；seed 改变 case、layout、fan-out、policy、metadata/table 的顺序。
- 每进程保留 p20/p50/p80；候选比较先取进程内 p50，再对进程 repetition 取中位数。
- 主要报告：winner-change rate、paired speedup、crossover 所在区间、按父模型聚合的 bootstrap CI、正例和反例数；不只报告全局最大 speedup。
- `analyze_rq_adequacy_supplement.py` fail-closed：关键 intervention 缺失时输出 partial，而不是从历史 proxy 填补。
- 旧结果不被覆盖；所有入口拒绝已存在的 output path。

## 6. 新代码与执行入口

- `build_rq_adequacy_suite.py`：生成 1024/full 与 32/smoke 清单。
- `rq_llm_layout_bench.cu`：dense reuse、weighted fan-out、真实 state policy、padded page、block-table permutation。
- `run_cuda_reference_repetitions.py`：独立进程重复与 order seed。
- `analyze_rq_adequacy_supplement.py`：逐 RQ 充分性判定。
- `audit_attention_kv_multiconsumer_results.py`：逐框架检查 multi-consumer、attention family 与 KV-cache implementation，区分 direct/supporting/N/A/missing。
- `run_rq_adequacy_gpu1.sh`：卡1 smoke/full 单入口。

先运行 smoke；只有 smoke 的 build、correctness、framework import、native adapter 与 analyzer 均完成，再运行 full。最终报告中的 `adequate` 只适用于完成的单卡干预；L-RQ5、Hexcute、跨架构问题不会被自动升级。

## 7. Attention、KV cache 与多 consumer 的补充覆盖

- 多 consumer：CUDA reference 与 Triton 直接计时 7 个 head-local:token-local fan-out；TVM 把 row/column 两类 consumer 编译进同一个 CUDA module。CUTLASS/CuTe 只提供 attention-score output tile/layout supporting evidence，因为它不拥有 persistent KV allocator。
- Attention：smoke/full 都从真实父模型覆盖 MHA、MQA、GQA、sliding attention、sparse attention 和 MLA。vLLM/SGLang 对 sparse/MLA 是 native writer/operator slice，不能写成完整 attention replay。
- KV cache：连续 NHD/HND、paged NHD/HND、page size 1/4/8/16/32/64/128/256、identity/reverse/permuted block table、request-major/block-interleaved slot schedule、B=1/2/4/8，以及 vLLM LBNHC/LBHNC/block16/block32、SGLang NHD/HND/page1/page16。
- vLLM/SGLang 的 prefill→decode 与 multi-request 是真实 supporting evidence；target/draft 或两个异构 attention backend 同时消费同一 cache 尚无稳定、低风险的公开运行时注入点，所以不会伪装成 direct evidence。
