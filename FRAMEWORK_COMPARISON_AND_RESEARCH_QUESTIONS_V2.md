# 现代 LLM / Diffusion 子图的 Layout 决策：框架比较与可证伪科研问题（V2）

更新日期：2026-09-13

## 0. 结论先行

现有框架不能按“谁会不会做 layout 优化”一维排序，因为它们控制的决策层不同：

- **SGLang / vLLM** 控制运行时与长寿命状态：attention backend、KV/cache 物理排列、page/block、cache group、连接器/传输兼容性。
- **CUTLASS / CuTe** 提供最强的专家显式 layout vocabulary 和硬件指令模板；`CollectiveBuilder` 做规则化构造，但并不是任意 layout 的全局搜索器。
- **Triton** 由程序员给出 tile、pointer/index 与 autotune 候选，编译器用 distributed encoding / LinearLayout 做传播、合法化和数据移动 lowering；它的强项是编译与 lowering，不等于自动找到全局最优 layout assignment。
- **TVM** 同时具有 IndexMap/Layout IR、schedule transformation 和 MetaSchedule 实测搜索；但“图级 layout 传播”“kernel schedule 搜索”“layout rewrite postproc”是三套相关但未统一的机制。
- **Hexcute** 最接近“在给定 tile-level program 内自动合成 register/shared layout 与指令”的系统：先由关键指令生成约束，再求解、枚举冲突分支并用分析模型选择；但它没有联合求解 serving cache、跨 kernel 持久布局与运行时动态策略。
- **TileLang** 应作为额外参照：它介于 Triton 的 compiler-derived layout 与 Hexcute 的 constraint synthesis 之间；不是本研究中 Hexcute 的替代品。

因此，真正有价值的主问题不是“哪个框架最快”，而是：

> **在 persistent storage、kernel boundary、shared memory、register/thread-value、copy/MMA instruction 和 tensor lifetime 同时存在时，layout 决策在什么条件下可以分层独立优化；何时必须联合优化？怎样在不穷举的情况下保留全局最优候选？**

建议把主线收敛为三个相互衔接的核心问题：

1. **局部最优与子图最优的可分性边界**；
2. **运行时持久布局与 kernel 内部布局的双层联合决策**；
3. **constraint-first、cost-aware 的低测量预算选择方法**。

表示-layout 耦合、Diffusion 跨 step 生命周期可作为两类关键应用域；统一 layout 关系代数是方法基础，不宜单独包装成“造一个新 IR”的工程目标。

---

## 1. 先统一“layout 决策”到底是什么

### 1.1 六个层级

对一个 tensor `x`，至少要区分六类变量：

| 层级 | 变量示例 | 典型 owner |
|---|---|---|
| 逻辑/表示 | MHA 还是 MLA latent；FP16/FP8/FP4；scale 粒度；MoE packed 与否 | 模型、runtime、kernel 作者 |
| 持久 global storage | NHD/HND；`[L,B,H,N,C]` 的排列；page/block；NCHW/NHWC；跨 layer/step 相邻关系 | SGLang、vLLM、graph compiler |
| kernel 边界 | 输入/输出 stride、blocked/interleaved、materialize/alias、producer-consumer contract | runtime、graph compiler、kernel API |
| shared memory | tile shape、leading dimension、padding、XOR/swizzle、stage buffer | CUTLASS、Triton、TVM、Hexcute |
| register/thread-value | 哪个 lane/warp 持有哪些逻辑元素；fragment 排列；replication | CUTLASS、Triton、Hexcute，TVM 的 execution layout |
| instruction / schedule | vector width、ldmatrix/stmatrix/TMA/cp.async、MMA major、warp 数、tile、pipeline stage | kernel compiler/library |

“NHD 与 HND”只回答了持久或边界布局的一小部分，不能代表完整 layout；“使用 Tensor Core”也不是 layout 本身，而是对 operand layout 施加约束的 instruction choice。

### 1.2 三个不可混淆的阶段

每个框架都应分别回答：

1. **表示能力（representation）**：能表达哪些 layout？
2. **可行性推导（feasibility）**：如何判断 layout 合法、兼容指令和上下游？
3. **性能选择（selection）**：在多个合法候选中，依据什么选择？

很多旧比较把“能够表达/合法化某 layout”写成“自动选择了最优 layout”。这是错误的。例如：

- CuTe 能表达复杂 swizzle，不表示 CUTLASS 会在所有 swizzle 中搜索最优解；
- Triton LinearLayout 能精确表示和降低 layout conversion，不表示传播 pass 已求解全 kernel 或全子图的最小代价 assignment；
- TVM `TransformLayout(IndexMap)` 能改写物理索引，不表示 MetaSchedule 默认把所有 IndexMap 当作自由搜索轴；
- vLLM 能在多个 KV layouts 中求兼容交集，不表示排序依据是当前请求 shape 的实测延迟。

### 1.3 正确的待优化对象

令子图为 `G=(V,E)`，每个 tensor edge `e` 可有 persistent/boundary/shared/register layouts，每个 op `v` 可有 tile、copy、MMA、pipeline 和 backend 选择。一个可审计的目标应写成：

```text
min  J = T_compute
       + T_layout_conversion
       + T_pack_or_dequant
       + T_cache_append
       + T_transfer
       + T_padding_and_capacity
       + lambda_memory * peak_memory
       + lambda_search * search_cost

s.t. semantic equivalence
     backend / page / dtype compatibility
     instruction operand contracts
     alignment, vectorization, bank and resource constraints
     tensor lifetime and alias constraints
```

不同框架只是优化该问题的不同投影。公平研究必须固定不研究的变量，否则“换 backend/算法得到加速”不能归因于 layout。

---

## 2. 框架横向比较：不仅比较功能，还比较决策机制

### 2.1 决策机制总表

| 维度 | SGLang | vLLM | CUTLASS / CuTe | Triton | TVM | Hexcute |
|---|---|---|---|---|---|---|
| 主要优化单位 | serving phase、attention layer/cache pool | model/cache group、attention backend | 单 kernel / collective | 单 kernel 与 compiler region | PrimFunc/task；Relax 图传播另算 | 给定 tile-level kernel program |
| 决策时机 | server init + runtime phase | engine 初始化/worker 协商 | C++ 模板实例化/编译时 | 编译时；autotune 运行时测候选 | 编译/调优时，MetaSchedule 实测 | 编译时约束求解 + 分析模型 |
| 显式 layout 表示 | pool shape/stride、NHD/HND、page-major 等专用结构 | `KVCacheLayout`：逻辑 `[L,B,H,N,C]` 的若干物理 stride order | CuTe `Layout`、composition、tile/partition、非线性 swizzle | distributed encodings；LinearLayout 的 GF(2) basis；Gluon 可显式写 layout | IndexMap、Layout IR、buffer/execution axes、schedule trace | CuTe-like layout function、tensor type 中的 thread-value/task layout |
| 候选从哪里来 | 可安装且支持当前硬件/模型/phase 的 backend 与 cache path | 各 backend 声明支持列表的交集、用户/connector 请求 | 作者、example、manifest、`CollectiveBuilder` specialization | 作者的 program/config；compiler propagation；作者列出的 autotune configs | DLight/ScheduleRule 生成 schedule；标记的 layout-free buffer；用户 IndexMap | 性能关键 op/instruction 产生约束；instruction 分支形成搜索树；shared swizzle 枚举 |
| 硬约束 | model 类型、MHA/MLA、GPU 架构、dtype、page size、speculation、backend capability | 所有 backend 的共同支持、各 rank 一致、HNC 异构时 block-compact、block/page/connector 条件 | arch、dtype、alignment、MMA major、tile/cluster/stage、类型系统 | encoding/op contract、shape、target、legal conversion、resource | IndexMap well-formed、dependency、schedule legality、tensor intrinsic、postproc | layout 函数组合/逆、copy/MMA contract、alignment、satisfiability、thread arrangement |
| 多个合法候选如何选 | hardware/model 的 best-effort 分支与人工经验默认值 | preference order/投票；显式 env；connector preference；最终首候选 | expert rule/template；可外接 profiler | compiler heuristics；autotune 对作者给出的 configs 实测 | cost model + runner 测量 + database；DLight 是规则基线 | 指令 issue/completion microbenchmark 驱动的分析 cost model |
| 直接优化的 objective | 正确性、支持性、经验性能与可部署性混合 | 首先兼容性与系统一致性，不是 request-specific latency optimizer | 高性能 kernel 构造，由专家/规则隐式编码 | 局部代码生成质量；autotune 的 kernel latency | 调优 task 的实测 latency/预测分数 | 给定 program candidate 的估计 tile latency |
| 是否显式计 conversion | backend 自己承担，缺少统一跨 backend 模型 | connector/alias 有兼容逻辑，但未联合 attention+transfer 总成本 | 作者显式写 copy/rearrange | compiler 会插入/消除/降低 conversion，但 assignment 未统一全局 cost | 可插 layout rewrite/copy；是否纳入搜索取决于 graph/task 表达 | `rearrange` 显式；论文也承认多 GEMM 可产生 register-layout conversion |
| 动态 shape / phase | 强；prefill/decode、batch/cache 状态都可见 | 强；请求与 cache group 可见，但物理布局多为 allocation-time 决策 | 通常 shape-specialized | 支持动态参数，但高性能 config 常 shape-family 特化 | 支持 symbolic/dynamic，搜索泛化依赖规则/model | 支持 symbolic dimensions，但 layout synthesis 围绕给定 program/shape contract |
| 跨 kernel/长期状态 | 强于 cache 生命周期；弱于内部 kernel | 强于 cache allocation/transfer；弱于内部 kernel | 弱 | 弱 | Relax 有图级能力，但与 kernel layout search 未形成单一联合 solver | 弱；主要是 kernel 内 |
| 原生视觉/Diffusion | 不是核心 | vLLM-Omni 有独立 Diffusion KV layout，当前与 core resolver 分开 | kernel building blocks 可用 | 可写 attention/conv/norm | graph+kernel 均可表达 | 可写 tile kernels，但公开评测重点不是视觉全图 |
| 最大优势 | 看得到 serving phase 与 persistent cache | layout 合同显式、兼容性交集可审计 | 表达能力、硬件贴合、专家 oracle | 易编程、成熟 lowering、跨 NVIDIA/AMD、可观测 conversion | 实测搜索、IndexMap、graph compiler 基础 | 自动合成 register/shared layout 与 instruction，避免大量手工模板 |
| 结构性盲点 | 不生成 lane/shared/register layout；backend 切换与 layout 混杂 | 偏好排序不是 workload cost；内部布局委托 | 自动发现与跨 kernel 目标弱 | 候选/传播局部且受程序员 config 限制 | 三套机制未统一；candidate coverage 取决于 rules | anchor/thread arrangement、跨 GEMM conversion、shared 枚举复杂度、无 serving joint model |

### 2.2 SGLang：选择“实现路径与 cache contract”，不是合成 kernel layout

SGLang 当前 attention backend 文档按 GPU、MHA/MLA、page size、quantization、sliding window、multimodal 和 speculative decoding 列支持矩阵；还允许 prefill/decode 使用不同 backend，并通过 hybrid wrapper 组合。这是**运行时算法/backend/cache contract 决策**。

源码中的 `MHATokenToKVPool` 明确存在 NHD/HND、vectorized 5D 与专用 page-major pool 分支；HND 要求 `(size + page_size)` 可整除 page，vectorized 5D 又要求 page size 满足 16-byte vector 的约束。也就是说，layout 与 allocator 容量、page 合法性和 backend kernel contract 耦合，而不只是 tensor 的 `permute`。[固定 commit 源码](https://github.com/sgl-project/sglang/blob/d6fabb74b45d4fb92796cfb6740810b4811b018e/python/sglang/srt/mem_cache/memory_pool.py#L1949-L2042)

当前 FlashInfer wrapper 中多处把 cache layout 作为 `NHD` 传入，表明内部选择常由具体 backend wrapper 固定，而不是由 SGLang 对 lane/shared/register mapping 做搜索。[FlashInfer backend](https://github.com/sgl-project/sglang/blob/d6fabb74b45d4fb92796cfb6740810b4811b018e/python/sglang/srt/layers/attention/flashinfer_backend.py)

因此：

- 它解决了：服务期可用 backend、cache pool、page/模型专用表示与生命周期管理。
- 它没有解决：给定同一 backend/算法后，如何联合选择 global/shared/register/instruction layout。
- 研究中若把 FA3、FlashInfer、TRT-LLM、Triton backend 的延迟直接叫作“layout 差异”，结论无效，因为算法、fusion、指令和 kernel 数也变了。

来源：[SGLang attention backend guide](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/attention_backend.mdx)。

### 2.3 vLLM：显式的兼容性协商器，而不是 latency optimizer

vLLM 当前把每层 KV 的逻辑形状统一看作 `[L,B,H,N,C]`，`KVCacheLayout` 定义 `LBHNC/LBNHC/LHBNC/BLHNC/BLNHC/BHLNC` 等物理 stride order。[KV cache layout 文档](https://docs.vllm.ai/en/latest/api/vllm/v1/kv_cache_layout/)

其 resolver 的关键过程是：

1. 收集 backend 按偏好排序的支持列表；
2. 求交集；
3. 按各 backend 首选项“投票”排序；
4. heterogeneous HNC specs 时限制为 block-compact；
5. `VLLM_KV_CACHE_LAYOUT` 必须合法，否则报错；
6. connector preference 只有兼容时采用，否则退回首候选；
7. 没有额外请求时直接取 `candidates[0]`。

这些行为可直接由 [resolver 源码 L199–225](https://github.com/vllm-project/vllm/blob/410f6da5c4bb62010728502035bee1b5f0eab2ac/vllm/v1/attention/backends/utils.py#L199-L225) 和 [L240–304](https://github.com/vllm-project/vllm/blob/410f6da5c4bb62010728502035bee1b5f0eab2ac/vllm/v1/attention/backends/utils.py#L240-L304) 证明。backend API 也把返回值定义成“kernel 能消费的 layouts，按偏好排序”，并未要求提供 shape-conditioned latency model。[backend contract](https://github.com/vllm-project/vllm/blob/410f6da5c4bb62010728502035bee1b5f0eab2ac/vllm/v1/attention/backend.py#L359-L363)

vLLM-Omni 已有 Diffusion KV cache layout，但其文档说明 Diffusion backend 当前不进入 core resolver，而是使用自身默认/stride layout。这一点对视觉实验很重要：不能把 vLLM core 的 KV layout 选择能力自动推断到 Diffusion。[vLLM-Omni Diffusion layout](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/diffusion_kv/layout/)

因此：

- 它解决了：全模型、多 backend/rank/cache spec 的 layout **可行性交集**和可审计的物理 allocation/view。
- 它没有解决：针对当前 batch/context/prefix hit/PD transfer 负载，对合法候选做联合 latency/throughput/memory 最优化。
- 这恰好给出了一个高价值研究入口：把“支持列表偏好”升级为“运行时—kernel 双层 cost contract”，同时保留兼容性硬约束。

### 2.4 CUTLASS / CuTe：最强专家表达与硬件 oracle，但自动搜索范围有限

CuTe `Layout` 本质上是从坐标到索引的函数，支持 composition、tiling、partition；swizzle 是与普通 shape/stride layout 组合的非线性映射。[CuTe layout algebra](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/01_layout.md)、[swizzle 源码](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/include/cute/swizzle_layout.hpp#L38-L50)

CUTLASS 3 `CollectiveMma`/builder 的决策包含 global strides、tiled copy、shared layout atom、MMA atom、tile/cluster、stage 与 schedule。builder 会依据架构、dtype、layout、alignment 等选择一个支持的 collective；例如 SM90 builder 对输入 layout/dtype 是否允许某种 GMMA major 有明确的 compile-time 分支。[CUTLASS 3 GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html)、[SM90 layout-to-major 分支](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/include/cutlass/gemm/collective/builders/sm90_common.inl#L58-L107)

因此：

- 它解决了：如何把专家选择编码成类型安全、贴近硬件指令且高性能的 kernel。
- 它没有解决：对任意子图自动生成完整候选集并计入上下游 conversion/lifetime 后找全局最优。
- 在研究中应把 CUTLASS 作为“可表达性上界/专家 oracle”，不是和 vLLM 的 resolver 作为同一种选择算法直接排名。

### 2.5 Triton：强 layout 表示与 lowering，选择仍以程序/局部传播为中心

Triton 的 LinearLayout 用 GF(2) basis 表示 register/lane/warp/block 到 tensor 维度的映射，并支持 `invertAndCompose` 来推导 layout 间数据移动。[LinearLayout 源码](https://github.com/triton-lang/triton/blob/972d18aa0905dfb403091bd4c5746318d097df4d/include/triton/Tools/LinearLayout.h#L47-L65)、[layout conversion 分析](https://github.com/triton-lang/triton/blob/972d18aa0905dfb403091bd4c5746318d097df4d/lib/Analysis/Utility.cpp)

它能统一比较原先不同 encoding，优化 `convert_layout/local_load/local_store`，并在 Gluon 中让作者显式控制 layout。LinearLayout 论文在真实 benchmark 中把收益主要归因于减少或更好 lowering 这些 conversion；这证明“conversion 是一等成本”，但不证明现有 compiler assignment 已经全局最优。[Linear Layouts 论文](https://arxiv.org/html/2505.23819)、[Gluon layout tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/02-layouts.py)

现实选择路径仍是：作者写 pointer/index、tile 和有限 `@autotune` configs；compiler 根据 op/instruction contract 传播并合法化 encoding。公开的 layout propagation RFC 还展示了 loop 中不良 `convert_layout` 的反例，强制改变布局在该个案中有近 5 倍差异。这是“存在反例”的证据，而不是对所有设备的通用加速结论。[Triton issue #8706](https://github.com/triton-lang/triton/issues/8706)

因此：

- 它解决了：高层 tile program 到可执行 distributed layout 的编译、转换分析和高质量 lowering。
- 它没有完全解决：把整个 kernel/子图的 layout assignment 写成一个包含 conversion 的全局性能优化问题；autotune 也只搜索作者列出的配置。

### 2.6 TVM：表示、改写、搜索都具备，但三者不是一个统一 solver

TVM 当前至少要分三件事：

1. `TransformLayout/TransformBlockLayout` 用 IndexMap 改写 buffer 或 iteration 坐标；
2. Layout IR 表达逻辑坐标到 storage/execution axes 的映射，并验证 well-formed；
3. MetaSchedule 生成/评估 schedule，使用 cost model 与真实 runner measurement 建数据库。

`RewriteLayout` 在当前源码中是一个 **postproc**：它只收集被标记为 `layout_free_buffers` 的对象，依据 anchor consumer 建议 IndexMap，插入 cache-read/layout-rewrite block，并沿 cache-read chain 传播。这不是“对所有 tensors 的任意 layout 做全局 ScheduleRule 搜索”。[RewriteLayout 标记与收集](https://github.com/apache/tvm/blob/7fca2e17160bae19d0311c327042d5859361a92e/src/s_tir/meta_schedule/postproc/rewrite_layout.cc#L117-L149)、[anchor 与传播实现](https://github.com/apache/tvm/blob/7fca2e17160bae19d0311c327042d5859361a92e/src/s_tir/meta_schedule/postproc/rewrite_layout.cc#L195-L249)

MetaSchedule 的优势是可以用真实硬件测量 schedule；其 task 权重还能反映图中的调用频率。但 kernel task 搜索与 Relax 图级 ConvertLayout/边界持久化仍不是同一个 joint objective。[MetaSchedule 教程](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html)、[TVM Layout IR](https://tvm.apache.org/docs/tirx/api/layout.html)

因此：

- 它解决了：可编程 IndexMap、合法 schedule transformation 与 measurement-based search。
- 它没有自然保证：搜索空间包含所有高价值 layout；也没有默认联合求解多个 kernel 的边界 conversion、persistent cache 与动态请求分布。

### 2.7 Hexcute：constraint synthesis 最系统，但问题边界仍是 kernel 内

Hexcute 把 layout 作为 tensor type 中的函数。thread-value layout 从 GEMM/copy 等关键指令约束向外传播；存在多个 instruction 时形成搜索树。register layout 确定后，再对 shared-memory layout 求 alignment/satisfiability、枚举 swizzle，并用由指令 issue/completion microbenchmark 支撑的分析模型选择候选。[Hexcute 论文](https://arxiv.org/html/2504.16214)

这与“规则传播”有实质差别：它不只验证一个手工 layout，而是系统求解合法的 thread-value/shared layouts 和 instruction choices。但它仍要求用户给定算法/dataflow/pipeline 骨架，并非从任意 graph 自动发明最佳算法。

论文自己明确列出三个局限：多个 GEMM 通过 register tensor 相连时可能插入昂贵 conversion；用户需给每个 GEMM 标注一致 thread arrangement；shared-memory layouts 的朴素枚举是指数级，其按 buffer 独立分解尚需形式化。[论文 limitations](https://arxiv.org/html/2504.16214#S9)

其公开 artifact 要求 A100/H100 级环境，不能把 A10 上无法复现 artifact 误判成框架性能失败。[Hexcute artifact](https://github.com/hexcute/hexcute-bench)

因此：

- 它解决了：给定 tile program 内 register/shared layout、copy/MMA instruction 的 constraint-driven synthesis。
- 它没有解决：runtime page/KV allocator、跨 kernel/跨 denoise-step持久 layout、请求分布驱动策略；分析模型对未见设备/复杂 subgraph 的泛化也是开放问题。

### 2.8 TileLang：为什么应作为第七个参照

TileLang 将数据流和 schedule（thread binding、layout、tensorization、pipeline）分开，Fragment layout inference 可从操作约束推导 register allocation，也允许 `annotate_layout`/swizzle helper 显式控制。[TileLang overview](https://github.com/tile-ai/tilelang/blob/main/docs/get_started/overview.md)、[DeepSeek MLA example](https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mla/README.md)

它适合作为“inference/propagation + 用户 annotation + autotune”参照，但不应仅因名字含 `layout inference` 就等同于 Hexcute 的完整 constraint search。当前公开 issue 也出现过 attention-like program 的 layout inference conflict，说明可行性传播的冲突处理本身仍是研究对象。[TileLang issue #1117](https://github.com/tile-ai/tilelang/issues/1117)

---

## 3. 按现代子图看：到底是谁在决定哪一层

下表不写一个虚假的“框架固定 layout”。正确对象是决策函数：

```text
L* = F(program, shape, dtype, target, phase, backend,
       page/lifetime constraints, candidate generator, cost objective)
```

同一框架对不同 backend、shape 或 commit 可能输出不同布局。因此实验必须 dump 实际决策和 provenance。

| 子图族 | 持久/边界主要矛盾 | kernel 内主要矛盾 | SGLang / vLLM 决策 | CUTLASS / Triton / TVM / Hexcute 决策 |
|---|---|---|---|---|
| MHA/GQA prefill | append/transfer/prefix sharing偏好 token/page 连续；prefill Q 较长 | QK/PV 的 MMA fragment、Q/K/V shared swizzle、softmax reduction | backend、page、KV physical order、cache group | tile、lane/warp/register、shared、copy/MMA、conversion |
| MHA/GQA decode | 同头长 KV scan 与 page lookup/append 冲突 | query 很短导致 occupancy、vector width、split-K/V 和 reduction布局敏感 | phase/backend/cache contract | decode kernel 内映射和指令 |
| SWA / sparse attention | eviction/window、index/cache page 与 prefix sharing；indices 是否按 page/segment 重排 | gather coalescing 与 selected K/V 后续 MMA/reduction冲突 | cache role、window/page/index backend | indices/gather tile、register/shared layout；packing 是否 materialize 通常仍需程序指定 |
| MLA | latent 与 RoPE/scale 分离、expanded scratch、live-page dequant | mixed-type copy/MMA、latent→attention consumer layout | latent pool、backend、page、量化表示 | projection/absorbed attention 内部 layout 与 instruction；Hexcute可合成部分混合类型路径 |
| SwiGLU / GeGLU FFN | gate/up 是否 separate、packed/interleaved；是否跨 down-proj持久 | 两个 GEMM 与 activation 的 producer-consumer fragment 冲突 | 主要选择 fused/quant backend，不逐 tensor 合成 layout | CUTLASS 手工 epilogue；Triton程序+传播；TVM fusion/schedule；Hexcute anchor constraint+rearrange |
| MoE | token-major indirect vs expert-major pack；routing metadata、capacity padding、skew | grouped GEMM tile、mixed-type weight/scale、load vector | router/fused-MoE backend 与 batch | pack/sort算法通常显式；内部 GEMM/layout自动化程度不同 |
| Mamba/linear attention | conv/SSM state 的 head/state/chunk/page/lifetime；speculative snapshot | scan/reduction 的 thread mapping 与 vector load | 专用 state spec/pool/ring/page policy | scan kernel内部 layout；Hexcute artifact覆盖该类但不管 serving state allocator |
| DiT self/cross/joint attention | condition KV 是否跨 step复用；image/video token ordering；2D/3D RoPE axes | 与 prefill attention类似，但序列、head dim、mask结构不同 | vLLM-Omni仅覆盖自己的 diffusion contract；SGLang非通用 owner | kernel 编译器负责内部；TVM可进一步做graph边界 |
| DiT FFN / modulation / norm | timestep modulation、residual、FFN中间值能否保持blocked layout | reduction轴与 producer fragment、fused store | serving框架基本委托 | CUTLASS/Triton/TVM/Hexcute各自在kernel层处理，只有TVM天然有graph IR入口 |
| image/video patchify、3D conv、codec | NCHW/NHWC/NDHWC、channel block、space-time ordering、skip/residual | implicit GEMM tile、norm/resize消费者、shared swizzle | 六框架中的serving二者一般不负责；vLLM-Omni仅局部 | CUTLASS/Triton可做kernel；TVM可做graph+kernel；Hexcute做可表达tile program内部 |

### 每次运行必须记录的 layout 证据

不能只输出字符串 `NHD` 或延迟。每个 op/tensor 至少记录：

```text
framework + commit + backend + target
subgraph/case/shape/dtype/phase
logical_axes, physical_shape, byte_strides
page_size, block_size, layer/block compactness
memory_scope(global/shared/register)
thread_value_layout, warp/CTA mapping
vector_width, copy_instruction, mma_instruction
shared_swizzle/padding, pipeline_stages
conversion edge: src_layout -> dst_layout, materialized_or_alias, bytes
decision_owner: user / runtime_rule / backend_preference /
                compiler_propagation / constraint_solver / autotuner
candidate_set + rejected_reason + selected_reason
steady_latency + conversion/pack/transfer + peak_memory
```

只有这样才能回答“框架为什么选它”和“它是否接近最优”。没有 candidate set 与 rejected reason，只能观察结果，不能评价选择算法。

---

## 4. 更好的科研问题：假设、验证逻辑和可证伪条件

### RQ1（核心）：局部 layout 最优与子图全局最优何时可分？

**科学问题**

对 producer-consumer DAG，在哪些结构条件下，逐 op 选择最优 layout 与全子图联合选择等价？哪些 conversion、fan-out、reduction、instruction contract 或 tensor lifetime 会破坏这种可分性？

**假设 H1**

若边界 tensor 的 layout 兼容成本为零、或 conversion 可完全融合且不增加资源压力，则局部最优可组合；否则 interaction term

```text
I(Lp, Lc) = T(producer=Lp, consumer=Lc, boundary)
            - Tproducer(Lp) - Tconsumer(Lc)
```

会造成非零 regret。regret 在短 decode、一次性中间 tensor、fan-out、多 anchor GEMM、窄 reduction 中更大，因为 conversion 难以摊销。

**验证逻辑（此处只定义，不写代码）**

- 最小 DAG：`QK -> softmax -> PV`、`QKV -> RoPE -> KV append -> decode attention`、`gate/up -> activation× -> down GEMM`、`GEMM -> norm -> GEMM`。
- 固定算法、dtype、tile family 和 backend，只枚举合法边界/内部 layout；避免把 FlashAttention 与朴素 attention 比成 layout。
- 对可穷举的小 tile/小候选求 joint oracle；分别计算 local policy 与 global oracle 的 regret。
- 用二因素/三因素交互而不只看 winner；同时计 materialized conversion、register spill、shared bytes 和 occupancy。
- 在 prefill/decode、fan-out、reuse count、shape regime 上寻找“可分/不可分”的 phase boundary。

**证伪条件**

在测量噪声之外，所有合法候选上 interaction term 均可忽略，并且 local policy 在未见 shape 上持续达到 joint oracle；此时“需要全局联合选择”的假设不成立。

**科学贡献边界**

贡献应是可分性的条件或预测准则，不是“实现一个 fused kernel”。

**A10 可行性：高。** 先用 Triton/TVM/自定义 CUDA 做统一候选空间，再接框架 native trace。

### RQ2（核心）：runtime persistent layout 与 kernel internal layout 是否必须双层联合决策？

**科学问题**

serving runtime 选择 KV/page/layer order，kernel compiler 选择 shared/register/instruction layout。两层是否可以用一个稳定 contract 独立优化，还是必须联合考虑 attention、append、prefix reuse、KV transfer 与 capacity？

**假设 H2**

存在显著 workload 区域，使 vLLM/SGLang 的固定 backend preference 或单一 cache layout 不是总服务成本最优：decode scan 偏好的 head-local 布局、append/PD-transfer 偏好的 block/layer contiguous 布局以及 kernel MMA 偏好的内部 layout 互相冲突。一个“外层 persistent layout + 内层 synthesized layout”的双层策略，在不改变 attention 数学/算法时能降低：

```text
T_total = T_append + T_attention + T_prefix_restore
          + T_PD_transfer + T_layout_conversion
```

**验证逻辑**

- 外层候选取 vLLM 的合法 `LBNHC/LBHNC/BLHNC/...` 及 SGLang 当前 backend 支持的 NHD/HND/page policy；不得强行运行不支持候选。
- 内层固定一个 kernel family，枚举/合成 register/shared/copy layouts。
- 工作负载轴：prefill/decode、batch、context、page、prefix hit ratio、PD-transfer on/off、layer group、MHA/GQA/MLA。
- 四种 policy：runtime 默认；仅外层 oracle；仅内层 oracle；联合 oracle/预测器。
- attention、append、transfer、可用 cache capacity 分项计量，并按真实 trace 权重合成；报告 Pareto frontier，不强塞成单一 latency。

**证伪条件**

若一个 persistent layout 在所有合法 workload 上 Pareto dominate，或内部 compiler 能以近零 conversion/资源成本完全吸收外部 layout 差异，则双层联合优化没有必要。

**A10 可行性：中高。** Kernel 因果验证可做；SGLang/vLLM 完整 PD/多节点外部效度需额外环境。

### RQ3（应用域）：表示、metadata 与 layout 能否分别优化？

**科学问题**

MLA latent、KV quantization scales、MoE routing indices、sparse attention indices 都改变“哪些字节在何时被访问”。数据表示与 layout 是否具有可分目标？

**假设 H3**

表示选择和 layout 之间存在不可忽略的交互：

- MLA 若为一次 decode 展开/反量化整个 pool，会使 latent storage 的容量收益无法转化为带宽收益；只对 live pages 处理并直接产出 consumer layout 才有优势。
- FP8/FP4 data 与 scale/zero-point 的相对排列决定 vector copy 和 MMA feed；只优化 data tensor 会造成 metadata gather。
- MoE/sparse 的 token/expert 或 token/page 重排只有在 reuse、skew 与 downstream tile 足够大时才摊销 packing。

**验证逻辑**

- 采用 factorial design：representation × data layout × metadata layout × consumer layout。
- MLA：expanded、latent+full-pool materialize、latent+live-page fused conversion。
- Quant：per-tensor/per-channel/per-block scale，separate vs colocated/blocked metadata。
- MoE/sparse：direct indirect access vs stable pack/sort；均匀、Zipf、热点 routing。
- 总成本必须含 dequant/pack/sort/scatter/padding/temporary peak，精度误差作为硬约束。

**证伪条件**

若加性模型已解释性能，交互项不显著，且分别优化 representation/layout 总能命中 joint oracle，则联合假设被否定。

**A10 可行性：中。** 机制与简化 kernel 可做；native MLA/FP4/Hopper 专用 instruction 不能在 A10 上外推。

### RQ4（方法基础）：能否建立“表达力足够但可求解”的跨框架 layout 关系模型？

**科学问题**

CuTe general layout/swizzle、Triton GF(2) LinearLayout、TVM IndexMap/Layout IR、Hexcute layout function 能否被一个分层关系模型统一比较？该模型在表达能力与求解复杂度之间的边界是什么？

**假设 H4**

主流高性能候选可由混合关系系统表达：

```text
logical coordinate
  -> persistent/boundary affine-or-permutation layout
  -> shared affine + finite bitwise swizzle
  -> thread/value finite mapping
  -> instruction operand relation
```

以 semantic equivalence、coverage/bijection/surjection、alignment、bank、instruction contract 为硬约束，可在不排除 oracle winner 的前提下显著压缩搜索空间。单一“纯 affine”或“纯 GF(2)”表示预计都不完整，因此研究对象应是可组合关系，而不是强行统一语法。

**验证逻辑**

- 在小 tile 上穷举坐标表作为 ground truth；从 CuTe/LinearLayout/IndexMap/Hexcute 导入后逐点比较映射。
- 测 coverage：能否表示真实 framework dump；false accept：生成后不合法；false reject：合法高性能 layout 被剪掉。
- 对每个小空间求 oracle，观察 constraint pruning 后 optimum preservation 与 candidate reduction。
- 单列 non-power-of-two、nonlinear swizzle、padding/non-surjective、replication/broadcast，防止只在最容易的 row/column major 上成功。

**证伪条件**

若常见且高性能的候选无法表达，或 pruning 经常删除 oracle winner，则模型不足；若表达完整但求解复杂度没有下降，则方法不实用。

**A10 可行性：正确性高、性能外部效度中。** H100/Blackwell 特有 TMA/WGMMA 布局只能做静态 contract 或异机复现。

### RQ5（核心）：约束之后，怎样以少量测量选择而不依赖脆弱启发式？

**科学问题**

在合法候选图上，结构化分析模型与少量主动测量能否比局部 heuristic propagation 更准确、比穷举 autotune 更省试验，并对未见 shape/subgraph 泛化？

**假设 H5**

`constraint-first + uncertainty-aware measurement` 会优于两端：

- 比纯分析 Hexcute-style model 更能适应 cache/occupancy/conversion 与新设备误差；
- 比 Triton/TVM 的固定 grid 或大预算搜索用更少 trial 达到相同 simple regret；
- 比 backend preference/local propagation 更能识别跨边界 interaction。

模型特征必须来自机制：transaction/segment、bank multiplicity、vector width、instruction count/latency、occupancy、conversion bytes、reuse/lifetime、page padding；不能只用 layout 名称。

**验证逻辑**

- 在一部分可穷举 case 上建立 oracle；评价 `simple regret vs measured trials` 学习曲线。
- baseline：runtime preference、CUTLASS expert configuration、Triton heuristic/autotune、TVM MetaSchedule、Hexcute analytical-only、random。
- leave-one-shape-family、leave-one-subgraph、leave-one-model-family；LLM 与 2025–2026 image/video motif 分别报告，不能随机打散相似 shape 造成泄漏。
- 消融：无 constraint、无 conversion feature、无 lifetime、无 uncertainty、无 real measurement。

**证伪条件**

在相同 trial budget 下不降低 regret，或在未见 family 上系统误排；这时复杂策略不比现有 heuristic/measurement search 更有科学价值。

**A10 可行性：中。** 依赖先完成 RQ1/RQ4 的可控 candidate/oracle 数据。

### RQ6（视觉关键应用）：Diffusion 的跨 block/denoise-step lifetime 是否改变最优 layout？

**科学问题**

视觉生成中同一结构重复数十层/数十 denoise steps。一个单 kernel 较慢但可长时间保持的 layout，何时胜过每个 kernel 的局部最快 layout？

**假设 H6**

若一次转换成本为 `C`，保持 layout 后每次 consumer 节省 `Delta`，理想 break-even 为 `R > C/Delta`；真实阈值还受 residual fan-out、norm/resize、shape change、condition KV reuse 与 memory pressure 影响。因而存在可预测的 lifecycle phase boundary，而不是固定“NCHW 或 NHWC 最优”。

**验证逻辑**

- 现代 motif：DiT self/cross/joint attention、AdaLN/modulation→attention/FFN、2D/3D patchify、video space-time attention、latent codec conv/norm/upsample。
- 策略：逐 op 最快并转换；block 内持久；跨若干 blocks 持久；denoise-loop 持久；condition-only 持久。
- 控制 resolution、frames、token order、steps、CFG batch、condition reuse；相同数学和 precision。
- 先用合成重复次数验证 crossover，再在 2025–2026 模型真实 trace 加权。

**证伪条件**

不存在可重复 crossover，或 lifecycle 特征不能预测 winner；也可能 memory pressure/强制边界使持久策略始终不成立。

**A10 可行性：高（机制/简化 block），中（完整现代模型）。**

### 暂不作为主问题：跨 GPU 代际 portability

“layout policy 是否从 A10 迁移到 H100/Blackwell”是科学问题，但单卡 1 无法充分验证。A10 可以发现 Ampere 上的规律，不能验证 TMA/WGMMA、warp-group、non-power-of-two native instruction 下的结论。它应作为后续多硬件 external validity，而不是当前论文的首要 claim。

---

## 5. 验证这些问题的共同因果标准

每个 RQ 都要经过五层证据，不能只给一个端到端数字：

1. **映射正确性**：相同 logical tensor；坐标→地址/thread/value mapping 可审计。
2. **机制证据**：load/store/vector/MMA 指令、transaction、bank conflict、conversion 与资源占用符合假设。
3. **控制 kernel**：固定算法/fusion/tile family，只改变目标 layout 变量。
4. **完整子图**：加入 conversion、packing、dequant、padding、temporary memory 和 producer/consumer。
5. **真实 runtime/trace**：用真实 prefill/decode 或 denoise workload 权重验证外部效度。

评价“是否最优”也必须分级：

- **observed best**：只在已运行候选中最好；
- **search-space optimal**：对明确且完整的小候选空间穷举得到最优；
- **constraint-space optimal**：solver 能证明合法空间内最优，且 cost 等于真实 objective；
- **global optimal**：一般不能声称，除非已证明候选表达完备并纳入所有相关成本。

建议正文只报告 `regret to enumerated oracle` 与 candidate coverage，不使用无边界的“全局最优”。

---

## 6. 对现有代码的诚实判定

现有 harness **尚不能完成上述科研验证**，即使把当前所有脚本跑完也不够。它目前更接近“case catalog + 若干 layout mechanism microbench + 部分 framework source audit”。关键缺口是：

| 需求 | 当前状态 | 为什么不足 |
|---|---|---|
| 六框架在全部子图上的 native layout | 未完成 | 640-case 主性能路径主要是 PyTorch/Triton/TileLang，不是 SGLang/vLLM/CUTLASS/TVM/Hexcute 全 native；unsupported/delegated 不能当作已测 |
| 评价框架原始选择是否最优 | 未完成 | 没有为每个框架记录真实 candidate set、rejected reason、decision owner，也没有同一合法空间 oracle |
| 科研问题 RQ1 | 部分 | 有 boundary/shared/KV 微基准，但还没有系统的 local-vs-joint 完整枚举与 interaction/regret 分析 |
| RQ2 runtime–kernel 联合 | 未完成 | SGLang backend/page 与 vLLM layout probe 仍是外层；未与同一 kernel family 内层 layout 做交叉积，也未加入 transfer/prefix trace objective |
| RQ3 representation–metadata | 未完成 | 现有静态容量/简化 benchmark 不能证明 live-page dequant、scale layout、routing metadata 的联合效应 |
| RQ4 common relation/coverage | 部分 | 已有 layout utility，但尚未证明跨四种表示的 round-trip、表达 coverage、false reject 和 optimum preservation |
| RQ5 sample-efficient selector | 未完成 | 尚无统一合法候选集、oracle subset、trial-budget learning curve 和跨 family generalization |
| RQ6 Diffusion lifetime | 部分 | 83 个视觉 cases/参考 benchmark 提供范围，但没有跨 block/step persistence 与真实 trace 加权 |
| 2025–2026 视觉范围 | catalog 已补，native 验证未完成 | “出现率≥50%”目录不等于六框架均已运行；policy shape 与 ONNX exact shape 仍需分开 |

所以此轮不应该继续堆 benchmark adapter。应先确认 RQ 与统一证据合同，再重新设计最小实验矩阵；否则 4,338 个框架×case 单元会产生大量不可比较的数据，而不是科学证据。

---

## 7. 推荐研究路线与决策门

### 主线建议

建议把论文/研究主张组织为：

> **Hierarchical layout co-optimization under runtime and instruction constraints**：先用跨层关系约束保留合法候选，再以少量测量学习包含 conversion/lifetime 的子图成本，联合决定 persistent boundary layout 与 kernel internal layout。

RQ1 提供“为什么需要联合”的现象与条件；RQ2 给出现代 LLM serving 的核心实例；RQ5 给出方法；RQ3 与 RQ6 分别验证对 representation-rich LLM 和 lifecycle-rich Diffusion 的泛化；RQ4 是 correctness/solver 基础。

### 在写新代码前应先由用户确认的三点

1. 主线是否采用 **RQ1 + RQ2 + RQ5**，RQ3/RQ6 作为应用；
2. A10 是否只承担机制与 Ampere 外部效度，不声称覆盖 H100/Blackwell 特有 layout；
3. “所有子图×所有框架”是否接受 `native / delegated / unsupported` 三态，并只对具有同一合法候选空间的单元计算 optimality regret。

确认后，再设计代码时应从“最小可证伪矩阵”开始，而不是先扩展全部 case 数。

---

## 8. 用户提供的讨论链接（已按独立 URL 划分）

以下 HTTPS share 在本轮均按独立地址获取成功；`codex://` 是本地客户端引用，不和 HTTPS 拼接，也不能用普通 HTTP 检查器验证。

- [LLM/现代视觉子图讨论](https://chatgpt.com/share/6aa544d5-e610-83e9-a854-f8496c835725)
- [Prefill / Decode Attention 讨论](https://chatgpt.com/share/6aa54a83-ec44-83e9-9115-425ca979b0f7)
- [TileLang layout 讨论](https://chatgpt.com/share/6aa55b6f-e6c4-83ea-90b3-a7d229fec0de)
- [Triton layout 讨论](https://chatgpt.com/share/6aa55ba4-497c-83ea-b309-76b28bf2dd1f)
- [SGLang layout 讨论](https://chatgpt.com/share/6aa55c5f-f6dc-83ea-b2f3-626aeff6dbf3)
- [vLLM layout 讨论 A](https://chatgpt.com/share/6aa55c95-c204-83ea-96fe-85352528dd5a)
- [vLLM layout 讨论 B](https://chatgpt.com/share/6aa55ca9-4db0-83e9-a7d229fec0de)
- [TVM layout 讨论](https://chatgpt.com/share/6aa55d76-995c-83ea-b44a-49fa65cfb947)
- [Layout 约束讨论](https://chatgpt.com/share/6a9f6b96-6044-83ec-97a2-43f439587390)
- [科研问题讨论](https://chatgpt.com/share/6aa56213-0038-83ea-bf5a-a335f2cdb231)
- `codex://threads/01a01e1a-016c-70b0-8407-9e2afdc05360`：仅 Codex 客户端可打开的本地 thread reference。

## 9. 主要一手资料

- SGLang：[attention backend guide](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/attention_backend.mdx)、[pinned memory pool](https://github.com/sgl-project/sglang/blob/d6fabb74b45d4fb92796cfb6740810b4811b018e/python/sglang/srt/mem_cache/memory_pool.py)
- vLLM：[pinned layout resolver](https://github.com/vllm-project/vllm/blob/410f6da5c4bb62010728502035bee1b5f0eab2ac/vllm/v1/attention/backends/utils.py)、[pinned KV cache interface](https://github.com/vllm-project/vllm/blob/410f6da5c4bb62010728502035bee1b5f0eab2ac/vllm/v1/kv_cache_interface.py)
- CUTLASS/CuTe：[CuTe layout](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/01_layout.md)、[CUTLASS 3 GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html)
- Triton：[pinned LinearLayout](https://github.com/triton-lang/triton/blob/972d18aa0905dfb403091bd4c5746318d097df4d/include/triton/Tools/LinearLayout.h)、[Linear Layouts paper](https://arxiv.org/html/2505.23819)
- TVM：[Layout IR](https://tvm.apache.org/docs/tirx/api/layout.html)、[MetaSchedule](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html)、[pinned RewriteLayout](https://github.com/apache/tvm/blob/7fca2e17160bae19d0311c327042d5859361a92e/src/s_tir/meta_schedule/postproc/rewrite_layout.cc)
- Hexcute：[paper v3](https://arxiv.org/html/2504.16214)、[artifact](https://github.com/hexcute/hexcute-bench)、[Hidet CuTe IR](https://github.com/hidet-org/hidet/tree/f962fc32f6c67b581c125d68134d0de0599d4041/python/hidet/ir/cute)
- TileLang：[overview](https://github.com/tile-ai/tilelang/blob/main/docs/get_started/overview.md)、[layout/swizzle API](https://www.tilelang.com/autoapi/tilelang/layout/swizzle/index.html)
