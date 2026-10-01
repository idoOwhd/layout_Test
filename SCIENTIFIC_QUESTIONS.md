# LLM 与 Diffusion 子图 Layout 优化：可行性、假设与验证设计

## 结论摘要

最值得优先验证的科学问题不是“哪个框架最快”，而是：**当一个 tensor 同时受生产者、消费者、GPU 指令、缓存生命周期和动态 shape 约束时，局部最优 layout 何时不再是子图最优？** 现有六个系统处在不同抽象层，不能当作六个同类的 layout 搜索器：SGLang/vLLM 是 serving 与缓存策略层，CUTLASS 是显式 kernel 模板库，Triton 是显式 tile DSL 加编译器 layout lowering，TVM 是带 IndexMap 和 schedule search 的 tensor compiler，Hexcute 才把 register/shared thread-value layout 的约束求解作为核心自动化能力。[^1][^2][^3][^4][^5][^6]

因此，公平实验必须分三层：地址/指令机制微基准、同语义单子图配对消融、真实 shape 的完整子图。只比较端到端 latency 无法把加速归因于 layout；只比较单 kernel 又会隐藏转换、padding、packing 和 KV 容量代价。

## 1. 子图范围与结构

现有 `real_world_shape_manifest.json` 固定了 137 个模型 release、640 个 case（320 prefill + 320 decode），覆盖八类现代 LLM 子结构。每个 case 都包含来源 commit、完整 shape、算子序列和 tensor contract。

视觉侧不能用旧 ViT/ResNet 分类模型替代现代生成 workload。新增的 `vision_model_corpus.json` 严格采用共享讨论中的 2025–2026 十二个图像/视频生成家族（6 image + 6 video），按“每个架构家族一票”计算 all/image/video 三种分母。`build_vision_manifest.py` 只纳入任一适用分母采用率达到 50% 的 motif，并为每类生成至少三个 shape regime；当前得到 13 类高频 motif、83 个 shape case。基准策略 shape 与 ONNX 实测 shape 明确分栏，用户已有 ONNX 通过 `discover_vision_onnx.py` 提取后才能标成 exact model shape。

2025–2026 视觉生成高频集合包括 DiT block、token mixer、FFN、timestep modulation、2D/3D patchify、latent codec、text cross/joint conditioning，以及达到阈值的 RoPE、QK-Norm、image dual-stream/dual+single hybrid、video 3D representation 和 structured attention。Linear Attention、token-MoE、timestep-MoE 仍保留在 corpus 统计中，但因全体或对应域采用率不足 50%，不会被伪装成“通用子图”。详细分子、分母和 shape 数见 `vision_shapes/VISION_COMMON_SUBGRAPHS.md`。

| 子图 | 核心算子链 | 关键长寿命/边界 tensor | 主要 layout 矛盾 |
|---|---|---|---|
| GQA/MHA | RMSNorm → Q/K/V GEMM → RoPE → attention → O GEMM → residual | Q、K/V cache、score/output | token-major 有利于写入/搬运；head-major 可能有利于 decode 的同头连续扫描；paged layout 引入容量与地址计算 |
| Sliding-window attention | 同 GQA，但只读取窗口 | paged K/V、window index | 连续物理窗口、page 边界与 prefix sharing 冲突 |
| Sparse attention | index Q/K → top-k → gather K/V → attention | indices、selected K/V | token-major 与稀疏 gather 冲突；预排序/packing 有额外成本 |
| MLA | Q 投影；latent KV 压缩/缓存；吸收或展开 attention | latent cache、RoPE cache、dequant/expanded scratch | layout 与表示大小耦合；展开整个 pool 会掩盖 latent cache 的收益 |
| SwiGLU FFN | RMSNorm → gate/up GEMM → SiLU× → down GEMM → residual | gate、up、middle | 两个生产者和 down-GEMM 消费者是否共享 packed/interleaved layout；能否消除 materialization |
| MoE | router → top-k → dispatch/pack → grouped expert FFN → combine | token/expert map、expert input/output | token-major 无 packing 但访问不规则；expert-major 需 pack/unpack 与 capacity padding |
| Linear attention | Q/K/V/gate → recurrent state update/scan → output | KV state、normalizer state | sequence 连续、head/state 连续和 chunk 并行化不能同时局部最优 |
| Mamba2 | in-proj/conv → selective scan → out-proj | conv state、SSM state、speculative snapshots | state 生命周期、chunk scan、page-major speculative cache共同约束 layout |
| Diffusion Transformer attention | AdaLN/RMSNorm → self/cross attention → MLP → residual | Q/K/V、condition cache | 与 LLM prefill 类似，但 denoise steps 允许跨 step 摊销持久 layout |
| U-Net/DiT spatial block | Conv/linear → GroupNorm/AdaLN → activation → residual/up/downsample | activation、weight | NHWC 有利于 channel vectorization/tensor core；norm/resize/spatial consumer 可能偏好别的轴 |

常规 decoder 的 embedding/LM head、dense GEMM、RMSNorm/LayerNorm、RoPE、softmax 和 residual/elementwise 也占重要比例，但它们应作为上述子图内部节点研究，不应拆成互不相关的算子排行榜。

## 2. 六个框架究竟做了什么决策

| 框架 | 决策拥有者 | 已解决 | 没有解决/不能声称解决 |
|---|---|---|---|
| SGLang | runtime 选择 attention backend、page/cache pool、MHA/MLA/Mamba 表示及少量 backend/layout 开关 | token/page 分配、prefix sharing、模型特定 cache；当前源码同时存在 NHD、HND、vectorized 5D、MLA latent、Mamba page-major 路径 | 不自动联合求解任意算子的 global/shared/register layout；许多选择由被调用 backend 固定；HND 与 CPU offload/PD disaggregation 等路径存在兼容性限制[^1] |
| vLLM | engine 汇总各 attention backend 的支持列表与优先级，选一个全模型兼容 KV layout | paged KV、hybrid cache group、跨 backend/rank 的可行 layout 交集；当前默认优先 LBNHC（paged NHD），并把旧 NHD/HND 名映射为 LBNHC/LBHNC[^2] | 选择规则首先是兼容性/偏好投票，不是针对当前 shape 的实测全局最优；backend 内部 shared/register layout 仍被委托 |
| CUTLASS/CuTe | kernel 作者/manifest 用模板与 layout algebra 显式指定 | GEMM/attention building block 的 global layout、CTA/warp/instruction tile、shared swizzle、pipeline；能表达硬件友好布局 | 不是任意计算图的自动 layout 搜索器；不同 epilogue/consumer 的边界 layout 与转换摊销需上层决策。例 35 直接把 softmax 相关 tensor 固定为 RowMajor，并以 128-bit 对齐访问[^3] |
| Triton | 程序员写 pointer/index order 与 tile；编译器把 blocked/dot/shared encoding 转为 LinearLayout 并降低 conversion | 以 GF(2) basis 表达 lane/warp/register/CTA 映射；合法化 dot/shared layout、swizzle 和跨 layout 数据移动[^4] | 常规 `@autotune` 只枚举作者给出的配置；不会自动改写整个 serving graph 的持久 KV storage 或联合多个 kernel 的边界 layout |
| TVM | schedule/template/MetaSchedule；IndexMap 表达逻辑→物理映射 | `transform_layout`、`transform_block_layout`、显式 padding/非满射映射与 schedule search；可把 layout 作为 TIR 变换[^5] | 自动搜索质量受 schedule rules、operator coverage、cost model 和 dynamic shape 约束；合法 IndexMap 不等于子图级性能最优；转换生命周期常需显式建模 |
| Hexcute | type-inference/constraint propagation 从 anchor instruction 推导 thread-value/task mapping，共享内存 layout 再选 swizzle | 自动合成 register/shared tensor 的 thread-value layout；冲突时 rearrange；artifact 覆盖 GEMM、attention、MoE、Mamba scan 与 cost-model 实验[^6] | 公开 artifact 主要面向 A100/H100 和 kernel/特定端到端实验；不能据此断言已经联合优化 SGLang/vLLM 的 page allocator、KV 生命周期、动态 batch 或 Diffusion 跨 step storage |

### 按子图展开的框架决策矩阵

表中“显式”表示 kernel/调度作者选择，“推导”表示编译器在给定 program/anchor/contract 后推导内部映射，“委托”表示该框架本身没有作出这层决策。具体 shape/stride 必须由本次运行的 dump 与源码 commit 确认，不能从框架名字反推。

| 子图 | SGLang | vLLM | CUTLASS | Triton | TVM | Hexcute |
|---|---|---|---|---|---|---|
| GQA prefill/decode | MHA pool 决定 slot/page 与 NHD/HND/vectorized storage；attention backend 决定消费 layout；Q/K/V 临时 tensor通常委托 backend | backend 支持 layout 的交集+偏好决定 LBNHC/LBHNC 等 paged KV；Q/K/V shared/register 委托 backend | 作者显式给 Q/K/V global CuTe layout、shared swizzle、MMA fragment 与 epilogue | pointer arithmetic显式 NHD/HND/page；编译器推导 blocked/dot/shared LinearLayout | tensor/layout rewrite 由 IndexMap/schedule 显式或搜索；kernel 内 tile由 schedule | global input contract通常显式；thread-value、task mapping、shared layout/swizzle推导 |
| Sliding attention | 复用/组合 MHA pool与 SWA allocator，只保留/读取窗口；物理 kernel委托 | hybrid KV group统一 page size并按 attention type分配 block；内部 layout 委托 | 需作者把 window mask/block traversal 纳入显式 tile/layout | window/page索引显式，内部映射推导 | schedule/IndexMap显式；是否融合 mask取决于模板 | 在已给 window program 上推导内部 layout；不负责 serving eviction |
| Sparse attention | DSA/index cache与专用 backend决定 selected page/token访问；没有跨 index+KV 的通用 layout solver | attention backend/spec决定支持，统一 cache resolver只保证兼容 | gather/index与 MMA 的 layout/iterator由作者显式实现 | sparse indices 与 gather offset由作者显式，lane/register lowering推导 | 可表达 gather和layout，但自动搜索依赖 op/schedule rule | 可对给定 tile DAG合成内部映射；routing/index storage仍是外部 contract |
| MLA | 单独的 latent+RoPE KV pool；量化/dequant和 backend吸收路径决定是否展开；源码可见 `[slot,1,kv_cache_dim]` 类 buffer | MLAAttentionSpec/对应 backend决定 cache spec与 kernel；全模型 layout需与其他层兼容 | 可实现低层 GEMM/attention，但 latent/absorbed graph需上层拼接 | latent、expanded、scale offset显式；内部 tile推导 | 可联合表达 projection/attention，但需 Relax/TIR fusion与 cost model | 可合成低精度 projection/attention kernel内部 layout；page/lifetime不在核心类型推导中 |
| SwiGLU FFN | 选择 fused activation/quant/GEMM backend；gate/up/middle 的内部 layout委托 | quantization/GEMM backend dispatch；KV resolver不参与 FFN layout | A/B/C、gate/up epilogue和down operand显式；可手写 persistent/fused epilogue | `[M,I]`、`[M,2I]`、interleaved offset与fusion由程序员显式；内部 layout推导 | Relay/Relax fusion + TIR schedule/IndexMap，可搜索候选 | 以 MMA/copy为 anchors 推导 thread-value/shared layout，冲突插 rearrange |
| MoE | router/dispatch 与 fused-MoE backend决定 token/expert packing；没有一个跨所有 backend 的固定 tensor layout | fused-MoE/quant backend决定 expert weight与activation packing；engine负责batch/routing | grouped GEMM 的 expert/problem arrays、A/B/C layout显式 | sort/pack/grouped kernel与 offsets显式；内部 tile推导 | 可表达 sort/segment/grouped GEMM，但动态专家负载使静态 schedule困难 | artifact含 mixed-type MoE；内部 layout自动合成，但 routing/capacity仍需程序策略 |
| Linear attention | recurrent-state pool和模型 backend定义 state shapes；部分 ReplaySSM/ring/page-major layout显式专用 | Mamba/attention-like cache spec与专用 backend；不存在通用 KV layout等价 | scan/GEMM primitives需作者显式布局 | sequence/head/state/chunk offsets显式，compiler推导线程映射 | scan schedule与state IndexMap显式/搜索 | artifact含 selective scan；自动合成 tile内部 mapping |
| Mamba2 | MambaPool明确分 conv/temporal state；可选 page-major envelope、ring与deduplicated window视图 | MambaSpec用独立 block/page约束并参与 hybrid cache page-size协调 | scan/conv/GEMM组件显式 | state/chunk/page offsets显式 | scan/reduction schedule与layout显式 | artifact直接提供 Mamba scan验证，内部 layout/task mapping自动 |
| Diffusion attention | 非主要模型/runtime contract；若外部接入仍委托 attention backend | 非主要 serving contract；若接入仍按 backend支持 | 同普通 attention，作者显式 | 同 prefill attention，pointer/global layout显式 | 可在 Relax/TIR graph中联合 layout | 可合成 kernel内部布局；跨 denoise-step持久 storage未由 artifact证明 |
| Diffusion Conv/Norm | 无通用决策 | 无通用决策 | Conv A/B/C、implicit-GEMM 与 epilogue layout显式 | convolution/normalization程序索引显式 | NCHW/NHWC/blocked IndexMap与schedule可表达并搜索 | 可在可表达的 tile kernel内自动内部 layout；graph-wide activation lifetime需上层 |

### 每类子图中应记录的 tensor layout

以下是实验字段，不是预先假定的“唯一最佳答案”。`source_evidence.py` 会在提供框架 checkout 时保存 commit、文件、行号和上下文，避免版本变化后仍引用旧结论。

| 子图 | 必须记录的 layout 决策 | 代表性候选 |
|---|---|---|
| GQA/SWA/sparse/DiT attention | Q/K/V 逻辑轴顺序；KV page/block 轴；head_dim vector；block table；score/output；shared Q/K/V tile；lane/warp 映射 | NHD、HND、`[block,N,H,D]`、`[block,H,N,D]`、vectorized-5D、shared XOR swizzle |
| MLA | latent、RoPE、scale、expanded scratch 是否分离；每个轴及 page；dequant 是否只访问 live rows | expanded NHD、latent token-major、latent paged；selective/fused dequant |
| SwiGLU | X、Wgate/Wup/Wdown；gate/up output；SiLU× middle；down operand；epilogue store | separate、`[M,2I]` packed、`[M,I,2]` interleaved、producer→consumer persistent tile |
| MoE | routing index/weight；dispatch token；expert input/weight/output；capacity padding | token-major indirect、expert-major packed、blocked expert×token×channel |
| Linear/Mamba | conv/SSM states；sequence/chunk/head/state axes；speculative state | sequence-major、head-major、chunked-head-major、page-major envelope |
| Diffusion conv/norm | activation、filter、norm reduction axis、residual boundary | NCHW/OIHW、NHWC/HWIO、blocked channels、persistent channels-last |
| Norm/softmax | reduction axis、producer output、vector width、shared reduction tile | last-axis-contiguous row major、direct tiled、tiled→row major |

SGLang/vLLM 对 SwiGLU/MoE 等通常是在 kernel/backend dispatch 层选择实现，而不是给每个中间 tensor 求一个统一 layout；CUTLASS/Triton/TVM/Hexcute 则能在不同程度上表达或合成 kernel 内部 layout。表中出现“delegated/unsupported”是实验结果，不应补成猜测值。

## 3. 已有性能证据能说明什么

现有 A10、PyTorch 2.6、Triton 3.2、TileLang 0.1.13 的 640-case 报告中，三方严格可比 552 case：TorchInductor/Triton 相对 PyTorch 的几何平均加速为 1.354×，TileLang 为 0.577×。Triton 在 prefill/decode 分别为 1.432×/1.290×；稀疏 attention prefill 为 4.551×；可比 Mamba2 prefill 样本则出现 Triton 11.197×、TileLang 40.015×。这些结果证明 fusion/specialization 对完整子图很重要，却**不能**把差异单独归因于 layout，因为 kernel 数、算法、编译、fallback 和 schedule 同时改变。

新实验把这个缺口拆成：

1. `static_layout_metrics.jsonl`：地址 span、128B segment、padding、pack/转换 bytes、shared bank conflict；
2. `triton_kv_layout.csv`、`shared_bank.csv`、`cutlass_softmax_boundary.csv`、`tvm_layout.csv`、`torch_layout.csv`：只改变目标 layout family 的配对微基准；
3. `llm_640_*.jsonl`：完整子图外部效度；
4. `source_evidence.json`、compiler dump 和环境：为什么框架做了该决策。
5. `vision_common_subgraph_manifest.json` 与 `vision_layout_reference.jsonl`：2025–2026 图像/视频生成高频 motif 的多 shape 合同与参考配对实验；
6. `layout_analysis_coverage.json` 与 `layout_analysis_optimality.jsonl`：框架×子图覆盖缺口，以及只有观测到 native layout 后才计算的 regret。

## 4. 科学问题、假设、方案与可行性

### RQ1：局部 kernel 最优与子图边界最优何时分离？

假设 H1：若 producer 的原布局为 `L0`，consumer 的最快布局为 `L1`，只有在 `Tconvert(L0→L1) < Tconsumer(L0)-Tconsumer(L1)` 或转换能与 producer 融合时才应选择 `L1`。基于单 kernel 排名的框架会在短 decode、窄 reduction 或一次性 tensor 上做错决策。

方案：对 tiled softmax/norm 比较 direct tiled 与 transform+row-major；对 SwiGLU 比较 separate、packed、interleaved，并把 conversion/packing 纳入总时间。跨 shape 拟合 break-even surface，而不是一个阈值。

可行性：高。已有 CUDA boundary kernel 和本次总控脚本可以直接运行；判据是 end-to-end paired speedup、转换占比与 NCU global transaction/sector 指标一致。

### RQ2：layout 是否应随 prefill/decode、batch、context 和 page size动态选择？

假设 H2：不存在对所有 attention phase 通用的 NHD/HND 最优。decode 的同 head 长序列扫描更偏好 head-local span，prefill 与 cache append/跨层搬运可能更偏好 token row；page size 改变 padding、地址计算与局部性边界。

方案：显式 Triton QK kernel固定算术、tile、warp、stage，只改变 NHD/HND/paged-NHD/paged-HND；扫描 `Q∈{1,16,128,2048}`、KV length、heads、head_dim、page size，并与 SGLang/vLLM 实际支持/默认选择对照。

可行性：高。`triton_kv_layout_bench.py` 已覆盖 Qwen3 decode/prefill 与 DiT attention；下一轮数据可扩展 batch/page sweep。若 layout winner 随 phase/shape 翻转，固定优先级假设被证伪。

### RQ3：缓存“表示”与物理 layout 能否联合优化？

假设 H3：MLA/量化 KV 的最优决策不是先选 dtype/latent representation、再单独选 layout。若 backend 为一次 decode 展开或反量化整个 pool，表示压缩会被额外全池流量抵消；只对 live pages selective dequant 并让输出直接匹配 attention consumer layout 才能保留收益。

方案：三路比较 expanded NHD、latent paged+全池展开、latent paged+live-page fused dequant；同时记录 live-row ratio、读写 bytes、临时峰值与 attention 时间。SGLang 当前源码已经显示 MLA 单独的 latent buffer，并存在量化 cache 的解量化读取路径，说明该问题可在真实实现中定位。[^1]

可行性：中高。静态 storage 模型已实现；要获得真实性能需在对应 MLA backend 加窄范围 instrumentation 或 wrapper，且保证算法/精度一致。

### RQ4：不规则子图中，何时值得付出 packing？

假设 H4：MoE/sparse attention 的 expert/token-major 选择存在由 `tokens_per_expert`、top-k、skew、capacity padding、hidden size 和 reuse 次数共同决定的 phase boundary。仅以 GEMM kernel 吞吐选 expert-major 会在 decode 小 batch 亏掉 pack/unpack；完全不 pack 又会在 prefill 丢失连续 grouped-GEMM 访问。

方案：生成均匀、Zipf、热点三种 routing；比较 token-major indirect 与 stable expert sort/pack，完整计入 prefix sum、scatter、padding、GEMM、combine。对 sparse attention同理比较直接 gather和按 page/segment 重排 indices。

可行性：中。需要框架 native grouped-GEMM adapter 才能避免 PyTorch loop 偏差；现有 50×2 MoE 与 5×2 sparse case 可提供真实 shape。

### RQ5：硬件合法性约束能否与跨 kernel 成本统一求解？

假设 H5：Triton LinearLayout、TVM IndexMap、CuTe Layout 和 Hexcute thread-value layout 可归一成“逻辑 index → storage/lane/warp/bank”的分层映射；先用可逆性、surjectivity、alignment、tensor-core contract、bank conflict作硬剪枝，再用包含 conversion/lifetime/cache 的图成本选择，可比纯 autotune 更少测量且更少局部最优。

方案：以 common layout IR 表示各层映射；候选必须通过等价/可逆/边界检查；目标函数为 `kernel latency + materialized conversion + padding capacity + lifetime-weighted traffic`，用实测校准而不是把 transaction 数直接当 latency。

可行性：中高。仓库已有 `Ladder_LL/ladder_ll/linear_layout.py`、TVM IndexMap bridge 与 bank-conflict checker，可直接作为 solver 前端；难点是从六个框架稳定抽取内部 mapping。

### RQ6：Diffusion 的跨 denoise-step 生命周期是否改变最佳 layout？

假设 H6：若 activation/weight layout 可跨多个 block 或多个 denoise step 持久化，channels-last/blocked layout 的一次转换可被重复使用；逐算子 cost model会低估该收益。反之，若 GroupNorm、resize 或 residual迫使每层转换，局部 NHWC conv 加速不会转化为端到端加速。

方案：先做单 Conv NCHW/NHWC 配对，再测 `Conv→Norm→activation→Conv+residual`，最后比较每层转换、block 内持久、全 denoise loop 持久三种策略；输出每个 tensor stride 和 conversion lifetime。

可行性：高（PyTorch/cuDNN 与 TVM 层），中（Hexcute/Triton 完整空间 block）。`torch_layout_bench.py` 已提供第一层因果实验。

## 5. 判定标准与反例设计

每个假设必须同时满足：相同 logical tensor 值与算术 contract；所有 layout 通过 correctness；预分配输出，计时不含一次性随机初始化；报告 P20/P50/P80；记录 compile time 但不混入 steady-state；至少三种真实 shape；用 GPU counters解释但不以 counter 替代时间；端到端结果计入转换、pack、padding与额外 peak memory。

最简单的反例是 shared transpose：32×32 float tile 的 stride-32 column access 理论上 32-way bank conflict，stride-33 或 XOR swizzle 为 1-way；若实测无差异，说明 kernel 没有按预期发出 load、优化器消除了循环、或测试被其他瓶颈淹没。`shared_bank_bench.cu` 使用 volatile shared load 和大量重复访问，使该假设可直接被 SASS/NCU 验证。

更强的反例是 reduction boundary：`1×16` tile 与 row-major 物理等价，必须接近 row-major；若所有带“tiled”标签的路径都被判慢，说明实验混入了实现质量差异，而不是 layout 原因。

## 6. 一条命令与结果合同

完整运行（物理卡 1，在进程内显示为 `cuda:0`）：

```bash
cd /home/liangyilei/ladder_home
bash staged/baseline_framework/layout_research/run_all_gpu1.sh --full
```

快速 smoke：

```bash
bash staged/baseline_framework/layout_research/run_all_gpu1.sh --quick
```

若本地有源码 checkout，可在命令前设置 `SGLANG_DIR`、`VLLM_DIR`、`CUTLASS_DIR`、`TRITON_DIR`、`TVM_DIR`、`HEXCUTE_DIR`；总控会把 commit 和命中源码行写入 `source_evidence.json`。Hexcute 上游全 artifact耗时数小时且可能锁频，默认只审计源码；显式设置 `RUN_HEXCUTE_ARTIFACT=1 HEXCUTE_BENCH_DIR=/path/to/hexcute-bench` 才运行其 smoke。官方 artifact说明 A100/H100、CUDA≥12.6，且完整 A100 kernel run约五小时，因此 A10 上不能把“不运行”错误解释为性能失败。[^6]

若要加入 serving 系统自己的完整路径，在同一命令前设置 `SERVING_MODEL_PATH=/absolute/local/model`。vLLM adapter 只改变其原生 `VLLM_KV_CACHE_LAYOUT`；SGLang 当前公开接口主要允许改变 backend/page policy，因而对应结果被标为 `serving_policy_confounded`，不能用来声称纯 layout 因果效应。若提供 `VISION_ONNX_ROOT`，同一运行还会提取视觉模型的精确 tensor shape 与 motif witness；否则 83 个视觉 case 保持 `benchmark policy` 标签。

结果目录为 `layout_research/results/<timestamp>/`，核心文件：

| 文件 | 用途 |
|---|---|
| `environment.json` | GPU、CUDA、Python与框架版本 |
| `status.jsonl` | 每个实验 success/failed/unavailable/delegated；失败不吞掉 |
| `source_evidence.json` | 框架 commit、文件、行号、源码片段和官方 URL |
| `shared_link_validation.json` | 十个独立 ChatGPT share 的内容可访问性；不包含 `codex://` |
| `vision_source_revisions.json` | 十二个视觉模型官方来源的不可变 commit URL |
| `static_layout_metrics.jsonl` | 地址与容量机制事实 |
| `triton_kv_layout.csv` | NHD/HND/paged KV 配对时间 |
| `shared_bank.csv` | stride32/padded/XOR shared layout |
| `cutlass_softmax_boundary.csv` | direct tiled 与 transform+row-major 的完整成本 |
| `tvm_layout.csv` | 显式物理 IndexMap-style reduction |
| `torch_layout.csv` | SwiGLU packed 与 diffusion NCHW/NHWC |
| `llm_640_pytorch_triton.jsonl` | 640-case 完整子图 |
| `llm_640_tilelang.jsonl` | 可选的第三个完整子图 backend |
| `vision_catalog/*` | 2025–2026 视觉家族、采用率与 83 个多 shape case |
| `vision_onnx_shapes.jsonl` | 可选 ONNX 精确 shape 与可审计 motif witness |
| `vision_layout_reference.jsonl` | 视觉高频 motif 的配对 reference layout 实验 |
| `sglang_native_serving.jsonl` / `vllm_native_serving.jsonl` | 原生 serving policy/layout 结果与实际 server log |
| `compiler_layout_evidence.jsonl` | PTX/SASS/Triton/TVM/CuTe layout/load/store 归一化证据 |
| `layout_analysis_coverage.json` | 框架×子图覆盖率与可计算 native regret 的数量 |
| `layout_analysis_matrix.jsonl` | 4,338 个 LLM/视觉×六框架单元及逐项 missing reason |
| `REQUIREMENTS_AUDIT.md` | 代码覆盖、验证状态与所有剩余缺口 |
| `REPORT.md` | 自动汇总 winner 与运行状态 |

## Sources

[^1]: SGLang Team, [memory_pool.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/memory_pool.py). 该源码定义 physical KV pool、NHD/HND/vectorized/MLA/Mamba 路径，并明确部分 HND 与 offload/disaggregation 限制。
[^2]: vLLM Project, [attention backend layout resolution](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/utils.py)；[Hybrid KV Cache Manager](https://github.com/vllm-project/vllm/blob/main/docs/design/hybrid_kv_cache_manager.md).
[^3]: NVIDIA CUTLASS, [GEMM with Softmax example](https://github.com/NVIDIA/cutlass/blob/main/examples/35_gemm_softmax/gemm_with_softmax.h)；[CuTe swizzle layout](https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle_layout.hpp)；[Efficient GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md).
[^4]: Triton, [LinearLayout.h](https://github.com/triton-lang/triton/blob/main/include/triton/Tools/LinearLayout.h)；[LinearLayoutConversions.cpp](https://github.com/triton-lang/triton/blob/main/lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp)；[layout-conversion analysis](https://github.com/triton-lang/triton/blob/main/lib/Analysis/Utility.cpp).
[^5]: Apache TVM, [physical buffer layout RFC](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0039-buffer-physical-layout.md)；[layout transformation implementation](https://github.com/apache/tvm/blob/main/src/s_tir/schedule/primitive/layout_transformation.cc)；[MetaSchedule rewrite layout](https://github.com/apache/tvm/blob/main/src/s_tir/meta_schedule/postproc/rewrite_layout.cc).
[^6]: Hexcute authors, [Hexcute reproducibility artifact](https://github.com/hexcute/hexcute-bench)；[paper](https://arxiv.org/abs/2504.16214)；[Hidet CuTe IR](https://github.com/hidet-org/hidet/tree/main/python/hidet/ir/cute).
