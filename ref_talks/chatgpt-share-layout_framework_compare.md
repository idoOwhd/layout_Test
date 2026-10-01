# 对比布局优化框架 1

> 来源：https://chatgpt.com/share/6aa7b017-feac-83ea-82ff-1b8df2c889b7
> 说明：以下正文按共享页面中的消息顺序导出。正文文字保持原样；标题是为区分消息类型而添加的 Markdown 结构。
> 边界：共享页未提供模型内部隐藏推理；“模型思考”仅包括页面公开的推理摘要、思考用时及工具活动标题。源页面已脱敏的正文只能保留其脱敏提示，无法恢复。

## 用户

现在请综合分析，以及对比: 不同框架的实现，以及问题
为LLM中常见（占比较高的子图）/Difussion 优化layout时，针对每一个子图现有的框架分别是怎么实现的？SGLang/vLLM/Cutlass/Triton/TVM/Hexcute对上述每一个子图， 它的决策是什么？分别解决了什么问题，但是没有解决什么问题，以及每个框架对每个子图的每个算子/每个tensor选的layout是什么，寻找这些框架中Github仓库中代码能证明为什么这么选？ 另外请针对这些子图和框架设计实现证明他们现有的问题是什么？在layout 优化上性能不好的原因是什么？怎么通过一个简单的实例证明，怎么通过更好的什么样的决策 就能为这些框架在layout优化上得到性能加速？】
1. 这些LLM 子图都是什么？结构是什么样？
2. 这些优化框架SGLang/vLLM/Cutlass/Triton/TVM/Hexcute， 分别是怎么实现layout 优化决策的？
3. 当前的这些子图在这些框架上性能如何？
4. 当前的这些子图在这些框架上的优化结果是什么？【如: 哪些算子/tensor/load，store指令，KV cache选什么样的layout, 以及为了得到这些结果，需要如何在这些子图上跑这些不同的框架，命令是什么，结果文件怎么存方便分析出我的目标】
5. 通过上述框架中的Github仓库源码 和 上述结果分析，能总结出这些框架在实现layout 优化时还有什么科研问题？
下面有一些共享链接回答了上述的一些问题：
1. LLM中的大模型占比talks：https://chatgpt.com/share/6aa544d5-e610-83e9-a854-f8496c835725，codex://threads/01a01e1a-016c-70b0-8407-9e2afdc05360，这里有一些关于prefill和Decode Attention的讨论：https://chatgpt.com/share/6aa54a83-ec44-83e9-9115-425ca979b0f7。目前我已经为这些大模型做了部分ONNX生成。
2. 不同框架中的layout优化策略: TileLang:https://chatgpt.com/share/6aa55b6f-e6c4-83ea-90b3-a7d229fec0de; Triton：https://chatgpt.com/share/6aa55ba4-497c-83ea-b309-76b28bf2dd1f; SGLang：https://chatgpt.com/share/6aa55c5f-f6dc-83ea-b2f3-626aeff6dbf3, ; vLLM:https://chatgpt.com/share/6aa55c95-c204-83ea-96fe-85352528dd5a, https://chatgpt.com/share/6aa55ca9-4db0-83ea-9e04-f4c9c8d3982f; TVM:https://chatgpt.com/share/6aa55d76-995c-83ea-b44a-49fa65cfb947（另外如果这里什么地方没有理解，可以找它对应的Github仓库）还有一些layout的相关约束：https://chatgpt.com/share/6a9f6b96-6044-83ec-97a2-43f439587390。
3. 你需要做的就是比较不同的layout选择方法，对比他们的不同 和 优势，以及在解决现代大模型的子图 和 layout选择问题的一些问题。针对这些问题，给出可以研究的科学问题（注意区分 科学问题不是工程问题），然后针对科学问题设计 假设+解决方案（期望利用这些方案之后能得到性能提升）

## 模型回复

我会把问题拆成两层：先建立“子图 × 框架 × layout 决策”的统一坐标系，再用各仓库源码/官方文档和可复现实验把“现状、瓶颈、科研问题、假设与验证方案”串起来。你给的共享链接我会当作线索，但关键结论会尽量回到官方仓库或官方文档做证据闭环。

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 github.com

## 模型思考摘要

已搜索 1 个网站

## 模型回复

一个关键边界已经很清楚：SGLang/vLLM 与 CUTLASS/Triton/TVM/Hexcute 不是同一抽象层。前两者主要决定“选哪个 kernel/backend、KV page/cache 怎么组织、什么时候融合/量化”，后四者才直接决定或生成 thread/register/shared-memory/global-memory layout；把它们硬放在同一层比较，会把真正的科研缺口掩盖掉。

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 github.com

## 模型工具调用参数

{"paths":["GitHub"],"query":"search"}

## 模型思考摘要

已搜索 1 个网站

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Hexcute 相关代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 Hidet 仓库 IR 目录结构

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Hidet 中的布局推断实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 Hidet Cute 变换目录内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 Hidet CUDA 分析目录

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查找 generic/cuda 中的相关 passes

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 CUDA 目录内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取指令选择与资源冲突解析文件

## 模型工具调用参数

{"uri":"/response/turn7","query":"class InstructionSelection"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

对比 Triton 与 CUDA Bank Conflict 优化

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取成本模型与自动注释

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 CUDA 成本模型关键代码

## 模型思考摘要

完成 Hidet 与 Triton 的布局优化及成本模型代码调研

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 github.com

## 模型思考摘要

已搜索 1 个网站

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索标准 CUDA MHA KV 缓存布局 реализации

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 memory_pool 指定提交的原始文件

## 模型思考摘要

搜索CUDA MHA缓存布局并获取指定提交文件

## 模型思考用时

思考了 3m 56s

## 用户

上述提问中还有一些额外要求：针对科学问题设计 假设+解决方案 时需要评估其可行性。

在解决现代大模型的子图 和 layout选择问题的一些问题。这些问题 应该如何通过相关框架这 常见的大模型子图进行验证。

## 模型回复

补充要求后，研究框架需要再加一列：每个“科学问题”不仅要有 Hypothesis → Method，还要有“可行性评估”和“如何在真实常见子图上证伪/证实”。我会把验证设计成可直接落到 benchmark matrix 的形式：子图、shape、layout 决策变量、框架 baseline、指标、消融、判据。

## 模型思考：工具活动

正在搜索 "Hexcute" layout compiler GitHub GPU

## 模型思考：工具活动

正在搜索 github.com

## 模型思考摘要

构建验证框架
构建了验证框架

## 模型思考摘要

已搜索 1 个网站

## 模型思考用时

思考了 22s

## 模型回复

对。加上你这条要求之后，最终目标不能只是“找出各框架 layout 做得哪里不好”，而应该形成一套完整的科研闭环：

**现代 LLM / Diffusion 常见子图 → 现有框架的 layout 决策机制 → 可复现反例 → 科学问题 → 假设 → 新 layout 决策方法 → 可行性 → 子图级验证 → 端到端验证。**

而且我建议把“layout 优化”严格定义成一个**跨层次、跨算子的数据映射问题**，而不只是 `NCHW/NHWC` 或某一个 shared-memory swizzle 的选择。下面是我认为最适合你这个研究方向的整体实验和科研设计。

---

# 一、先明确：真正应该研究哪些“大模型高占比子图”

如果你的目标是做论文，而不是做一个只针对 GEMM 的工程优化器，那么 benchmark suite 至少应该覆盖下面 7 类。

| 子图 | LLM | Diffusion | layout 敏感点 | 优先级 |
|---|---:|---:|---|---:|
| Dense GEMM + Epilogue | ✓ | ✓ | GMEM/SMEM/register、MMA fragment、vectorization | ★★★★★ |
| RMSNorm/LayerNorm → QKV GEMM → RoPE | ✓ | ✓ | producer-consumer layout、fusion boundary | ★★★★★ |
| Prefill Attention | ✓ | ✓ | Q/K/V tile、SMEM swizzle、QK/PV MMA | ★★★★★ |
| Decode/Paged Attention + KV Cache | ✓ |  | KV physical layout、page/block/head/dim order | ★★★★★ |
| SwiGLU/GEGLU FFN | ✓ | ✓ | 两 GEMM 中间 activation layout | ★★★★ |
| MoE / Grouped GEMM / quantized GEMM | ✓ | 部分 | expert dispatch + mixed type + scale layout | ★★★★★ |
| Conv/GroupNorm/SiLU/Conv |  | ✓ | NHWC/NCHW、Tensor Core tile、producer-consumer | ★★★★ |

再增加两个“压力测试”：

- Mamba / selective scan
- Sparse / sliding-window / MLA / DSA attention

Hexcute 自己的 artifact 已经覆盖 GEMM、Attention、FP8 GEMM、warp-specialized kernel、mixed-type MoE 和 Mamba selective scan，因此它实际上已经给出了一个很合适的研究起点。citeturn259371search1turn259371search5

---

# 二、一个非常重要的比较原则：这 6 个框架实际上处在不同层

不能简单写成：

> SGLang layout vs vLLM layout vs CUTLASS layout vs Triton layout ...

这样论文很容易被 reviewer 质疑，因为抽象层不一致。

更合理的层次是：

```text
                     Model / serving
                           │
             ┌─────────────┴──────────────┐
             │                            │
          SGLang                        vLLM
   backend selection              backend selection
   KV allocation/layout           paged KV layout
   page/cache policy              scheduler/cache policy
             │                            │
             └─────────────┬──────────────┘
                           │
                    Kernel / Compiler
                           │
        ┌──────────┬───────┼─────────┬───────────┐
        │          │       │         │           │
     Triton       TVM   CUTLASS    Hexcute   FlashAttn...
```

CUTLASS、Triton、TVM、Hexcute 才是真正应该比较：

> 给定一个 tensor program，怎样选择 thread/value mapping、GMEM/SMEM/register layout、copy instruction、MMA instruction、swizzle、tile 和 pipeline。

而 SGLang/vLLM 应该研究：

> 上层 runtime 做出的 page/cache/backend/quantization/layout 决策，如何限制下面 kernel compiler 的最优 layout。

这是一个很好的科研切入点。

SGLang 官方甚至允许 **prefill 和 decode 使用完全不同的 attention backend**，明确承认不存在一个 backend 在所有 phase 都最优。citeturn854810search0

---

# 三、6 个框架的 layout 决策本质是什么

可以先用下面这个表作为你整个研究的核心 taxonomy。

| Framework | layout 谁决定 | 搜索粒度 | 是否自动 | 是否跨算子 | 主要优势 | 核心限制 |
|---|---|---:|---:|---:|---|---|
| CUTLASS/CuTe | programmer/template | instruction/tile | 低 | 否/弱 | 极强表达能力 | 手工专家设计 |
| Triton | programmer + compiler heuristics | kernel | 中高 | 基本否 | 编程简单、自动 lowering | 搜索空间隐式且受 heuristic 限制 |
| TVM | schedule / MetaSchedule | operator/schedule | 高 | 有限 | 自动调优体系成熟 | layout × instruction 联合建模不足 |
| Hexcute | constraint/type inference + cost model | tensor program | 高 | 部分 | 自动 layout/task mapping synthesis | cost model / search scope 仍有限 |
| vLLM | runtime/kernel implementation | KV/page/backend | 中 | runtime level | paged KV、动态 serving | 不优化 kernel 内 layout |
| SGLang | runtime/backend/kernel selection | KV/page/backend | 中 | runtime level | backend 多、phase-aware | layout 决策与 compiler 分离 |

CUTLASS 官方对 CuTe 的定义本身就说明了这一点：核心对象是 hierarchical multidimensional `Layout`，通过 composition、tiling、partition 来描述 tensor，而 layout 本身主要由 kernel programmer/template 构造。citeturn442683search1

---

# 四、Hexcute 对你的问题特别重要

你之前说的 Hexcute 确实是 2026 CGO 的：

> **Hexcute: A Compiler Framework for Automating Layout Synthesis in GPU Programs**

不是一个泛称。

它的核心思想恰好和你的研究高度重合：

> 把 layout synthesis formalize 成 constraint problem，然后利用 type-inference-based algorithm 自动推导 layout/task mapping。

citeturn259371search0turn259371search3

而且 artifact 的实现实际上建立在 Hidet/CuTe IR 上。代码已经能看到非常关键的几层。

例如自动推导阶段明确区分：

```text
1. logical shape inference
2. logical layout inference
3. task mapping inference
4. memory layout inference
```

对于 Copy，它构造类似：

\[
f\circ p^{-1}=g\circ q^{-1}
\]

这样的 layout constraint。

对于 MMA，又建立 A/B/C 三个 operand 的 thread-value layout compatibility constraint。

源码甚至明确写出：

> 一个 copy 可能有多个可用 instruction，可以 DFS 搜索所有合法 variant，再由 cost model 选择，也可以用 heuristic / beam search 剪枝。

fileciteturn9file0L1-L2

这实际上给你留下了非常明显的科研空间：

> **合法 layout synthesis 已经做得比较好，但“合法 layout 中哪个真正最快”的 performance optimization 问题远没有解决。**

---

# 五、源码已经直接暴露出的 Hexcute/Hidet 问题

这是非常值得你利用的地方，因为不是我们猜的，而是源码自己承认。

它的 CUDA cost model 目前：

\[
Cost = N_{\mathrm{inst}}\times Latency_{\mathrm{inst}}
\]

同时假设 copy 和 MMA 可以充分 overlap。

源码明确写着：

> 当前 cost model 能通过 instruction selection 间接区分不同 shared-memory layouts。

但也明确写：

> **bank conflict 的影响未来才加入。**

以及：

> 假设 pipeline fully overlapped。

并且：

> **没有考虑 `cp_async_wait_group()` 和 `mbarrier`。**

fileciteturn10file0L1-L2

这几乎天然就是一个论文问题。

更有意思的是，另一个 pass 已经确实在做 bank-conflict optimization：

它枚举 swizzle，计算 warp thread 对 shared-memory bank 的访问，然后最小化：

\[
\sum_{\mathrm{copy}} ConflictWays(copy,L)
\]

fileciteturn8file0L1-L2

也就是说系统里面出现了：

```text
layout synthesis cost model
         ↓
不显式建模 bank conflict

后续 resolve_bank_conflict pass
         ↓
再次修改 layout
```

这意味着**layout selection 与 layout repair 是分裂的**。

这是一个非常漂亮的 scientific question。

---

# 六、科学问题 1：Local-layout optimality 是否导致 subgraph-level suboptimality？

这是我建议你最优先研究的问题。

## Scientific question

> 对每个 operator 独立选择最优 layout，是否可以保证整个 Transformer/Diffusion 子图最优？

理论上显然不能。

考虑：

```text
GEMM1
  ↓
RMSNorm / SiLU / RoPE
  ↓
GEMM2
```

假设：

\[
L_1^*=\arg\min_L T_{GEMM1}(L)
\]

而 GEMM2 喜欢另外一个 layout \(L_2^*\)。

那么真实目标不是：

\[
\min T_1+\min T_2
\]

而是：

\[
\min_{L_1,L_2}
T_1(L_1)
+
C_{\text{transform}}(L_1,L_2)
+
T_2(L_2)
\]

很多 compiler 优化了前两项，却没有显式建模：

\[
C_{\text{transform}}
\]

---

## Hypothesis H1

> **在 Transformer 高频 producer-consumer 子图中，允许单个 operator 使用 3–10% 较慢的局部 layout，可以通过消除 intermediate transpose/rearrange/reload，使整个 subgraph 获得更高性能。**

---

# 七、怎样用最简单的实例证明 H1

不要一开始跑 Llama-70B。

先做：

```text
GEMM
 ↓
transpose / reshape / elementwise
 ↓
GEMM
```

例如：

```text
A [M,K]
W1[K,N]

C = A @ W1
D = silu(C)
O = D @ W2
```

建立两个方案。

Baseline：

```text
GEMM1 → L1_opt
           ↓
        convert
           ↓
        L2_opt → GEMM2
```

Proposed：

```text
GEMM1 → Lshared
           ↓
     zero conversion
           ↓
         GEMM2
```

验证：

\[
T_{\mathrm{subgraph}}(L_{shared})
<
T_{\mathrm{subgraph}}(L_1^*,L_2^*)
\]

---

# 八、这个科学问题的可行性

非常高。

| 项目 | 可行性 |
|---|---|
| 实现难度 | 低—中 |
| 是否需要改 CUDA compiler | 不一定 |
| Triton 能否实验 | ✓ |
| Hidet/Hexcute 能否实验 | ✓ |
| CUTLASS 能否 oracle | ✓ |
| TVM 能否 baseline | ✓ |
| Nsight 能否解释 | ✓ |
| 论文新颖度 | 中高 |

而且这是一个真正的**科学问题**：

> local optimality 与 graph optimality 的关系。

而不是：

> “给某个 kernel 改一下 block size”。

---

# 九、科学问题 2：layout 是否应该与运行 phase 联合优化？

现代 LLM 最明显的反例就是：

```text
Prefill:
Q length ≫ 1

Decode:
Q length = 1
KV length ≫ 1
```

两者 roofline 完全不同。

Prefill 更接近 compute-intensive tiled GEMM：

\[
QK^T
\]

Decode 则大量时间是在：

\[
\text{load KV cache}
\]

因此一个对 prefill 最优的 KV/tile/layout 决策不太可能自动也是 decode 最优。

SGLang 已经从工程层面佐证了这一点：它允许分别指定

```bash
--prefill-attention-backend
--decode-attention-backend
```

并且不同 backend 的 page-size、FP8/FP4、speculative decoding 支持不同。citeturn854810search0

---

# 十、Hypothesis H2

> **静态 layout \(L\) 无法在 prefill/decode 及不同 sequence length 下同时最优；shape/phase-conditioned layout policy 可以显著降低 expected latency。**

将 objective 写成：

\[
\min_{\pi}
E_{x\sim workload}
[T(x,\pi(x))]
\]

其中：

\[
x=
(B,S_q,S_{kv},H_q,H_{kv},D,dtype,page\ size)
\]

而不是：

\[
\min_L T(L)
\]

---

# 十一、怎么验证 H2

直接做 phase × sequence grid：

```text
Prefill:
S = 128
    512
    2048
    8192

Decode:
KV = 128
     512
     2048
     8192
     32768
```

再变化：

```text
MHA
GQA 8:1
GQA 4:1
MQA
MLA
```

记录每个 shape 下：

```text
best layout
best tile
best page size
best warp count
best vector width
best backend
```

最后画：

```text
                     Sequence length
                           →
layout A    █████
layout B          █████
layout C                 █████████
```

如果最优 layout 出现明显 phase transition，H2 就成立。

---

# 十二、科学问题 3：KV Cache layout 是否需要和 attention kernel 联合优化？

这个方向尤其适合你的 KV layout 项目。

vLLM 的核心是 paged KV：

```text
logical token
   ↓
block table
   ↓
physical KV block
```

其 PagedAttention 实现中特意安排 thread 读取相邻 memory，从而实现 coalescing。citeturn854810search4

vLLM 新的 hybrid KV manager 又进一步让：

```text
full attention
sliding-window attention
```

可以拥有不同 slot reservation semantics。citeturn854810search5

SGLang 更直接。

其 KV cache memory pool 明确采用：

```text
ReqToTokenPool
      ↓
TokenToKVPool
      ↓
physical KV cache
```

而且 CUDA/ROCm 路径会判断 row width，优先调用专门 `store_cache`，否则才 fallback 到普通 indexing store。fileciteturn12file0L1-L2

所以：

> KV cache 的 physical layout 不是 attention kernel 自己选择的。

它已经是 runtime/kernel interface。

---

# 十三、Hypothesis H3

> **固定 paged-KV physical layout 为 runtime memory management 优化，但不一定最小化 attention 的实际 memory transactions；将 page allocation layout 与 decode kernel thread mapping 联合优化能够改善 HBM efficiency。**

研究变量：

\[
L_{KV} =
(order,\ page,\ vector,\ head\ grouping,\ quant\ grouping)
\]

例如比较：

```text
[P, T, Hkv, D]

[P, Hkv, T, D]

[Hkv, P, T, D]

[P, T, Hkv, D/x, x]

[P, Hkv, D/x, T, x]
```

千万不要只测 logical contiguous layout。

真正要测：

```text
physical block address
→ warp accesses
→ memory sector
→ L2
→ HBM
```

---

# 十四、验证 H3 的一个非常简单反例

例如：

```text
Hq = 32
Hkv = 8
D = 128
page = 16
FP16
decode batch = 32
```

Baseline：

```text
[P,T,H,D]
```

让 warp 内 thread 沿：

```text
D
```

取数据。

然后构造另一个 layout：

```text
[P,H,T,D/vector,vector]
```

令一个 warp 服务：

```text
same KV head
multiple token
vector D
```

观察：

```text
dram__bytes
l2 hit rate
sectors/request
global load efficiency
issue stall
kernel latency
```

如果：

\[
Bytes_{requested}
\approx Bytes_{useful}
\]

但 baseline：

\[
Bytes_{actual}\gg Bytes_{useful}
\]

你就得到了一个非常干净的 layout pathology。

---

# 十五、科学问题 4：layout / instruction / pipeline 是否必须 joint optimization？

目前很多系统实际上做的是：

```text
choose layout
   ↓
choose instruction
   ↓
schedule pipeline
```

但理论上真正的问题是：

\[
\arg\min_{L,I,P}
T(L,I,P)
\]

而不是：

\[
L^*=\arg\min_L C_L
\]

然后：

\[
I^*=\arg\min_I C_I(L^*)
\]

然后：

\[
P^*=\arg\min_P C_P(L^*,I^*)
\]

因为三者耦合。

例如 shared-memory layout 会决定是否：

```text
ldmatrix
lds128
cp.async
TMA
```

能够合法使用。

Hexcute/Hidet 的 instruction selection 中已经明确通过 thread-value layout matching 来判断某个 copy instruction 是否能使用，而且会检查：

```text
alignment
max_common_vector
src layout
dst layout
```

fileciteturn7file0L1-L2

---

# 十六、Hypothesis H4

> **layout synthesis 与 instruction/pipeline 独立优化会丢失全局较优方案；联合搜索能产生单独优化阶段无法到达的 configuration。**

这个实验最好用：

```text
GEMM
Attention
mixed-type GEMM
MoE
```

因为 instruction choice 非常丰富。

比较：

```text
A. layout → instruction → pipeline
B. instruction → layout → pipeline
C. joint(L,I,P)
```

如果 C 找到了 A/B 到不了的 Pareto point，就是非常强的证据。

---

# 十七、科学问题 5：现有 layout cost model 是否足以预测现代 GPU？

这里 Hexcute 的源码已经给你送来了最好的反例。

当前 model 明确不完整建模：

```text
bank conflicts
mbarrier
cp_async_wait_group
```

并假设 copy/MMA overlap。fileciteturn10file0L1-L2

而当前 NVIDIA GPU 又进一步增加：

```text
TMA
WGMMA
warp specialization
TMEM
cluster
2CTA MMA
```

CUTLASS 4.x 仍持续增加更复杂 pipeline/task scheduling、TMA 和 Blackwell/Rubin primitives，本身已经说明简单 latency-addition model 越来越难覆盖现代 kernel。citeturn442683search0

---

# 十八、Hypothesis H5

> **instruction-count based layout cost model 在复杂 pipeline 下会系统性 mis-rank layouts；加入 contention / synchronization / occupancy / cache terms 可以明显提高 ranking accuracy。**

重点不是预测：

```text
kernel = 17.32 us
```

而是：

\[
P[
C(L_i)<C(L_j)
\iff
T(L_i)<T(L_j)
]
\]

即：

> cost model 能不能把 layout 排对。

这是比绝对 latency error 更符合 layout optimizer 需求的 metric。

---

# 十九、怎么验证 H5

生成，例如：

```text
500–5000 layouts/kernel
```

记录：

```text
Predicted Cost
Actual latency
```

计算：

```text
Pearson
Spearman
Kendall τ
Top-1 accuracy
Top-5 recall
regret
```

其中我尤其建议使用：

\[
Regret=
\frac{T(L_{\text{chosen}})}
{T(L_{\text{oracle}})}-1
\]

因为 optimizer 最终关注的不是 RMSE，而是：

> 错选 layout 会慢多少。

---

# 二十、科学问题 6：layout transformation 的 cost 是否应该成为 first-class optimization variable？

目前很多 compiler 将：

```text
transpose
rearrange
reshape
shared-memory staging
```

当作 lowering 中间结果。

但对大模型子图来说，你真正需要优化的是：

\[
Compute+
DataMovement+
LayoutConversion
\]

---

## H6

> **显式把 layout conversion edge 加入 subgraph cost graph，可以找到 operator-local search 找不到的更优方案。**

模型：

```text
          L1a ─── L2a
         /   \    /
Tensor A     convert
         \   /    \
          L1b ─── L2b
```

变成 shortest path / dynamic programming / constraint optimization。

这是一个特别干净、可数学化的问题。

---

# 二十一、科学问题 7：Layout search 是否应该 workload-aware，而不仅 shape-aware？

这是 serving 场景里很有意思的问题。

例如同一个：

```text
Llama/Qwen layer
```

部署在：

```text
offline throughput
online low latency
long context
short context
large batch
small batch
```

最佳 layout 不一定相同。

所以：

## H7

> layout 最优解不仅是 tensor shape 的函数，也是 workload distribution 的函数。

即：

\[
L^*=f(shape)
\]

可能应该变成：

\[
L^*=f(shape,\ workload,\ hardware,\ cache\ state)
\]

这是把 compiler layout optimization 和 serving system 联系起来最自然的方法。

---

# 二十二、这些科学问题应该怎样映射回真实现代大模型子图

我建议最终实验矩阵这样设计。

| Scientific question | GEMM | FFN | Prefill Attn | Decode Attn | MoE | Diffusion | Mamba |
|---|---:|---:|---:|---:|---:|---:|---:|
| H1 cross-op layout | ✓ | ★ | ✓ |  | ★ | ★ | ✓ |
| H2 dynamic/phase layout |  | ✓ | ★ | ★ | ✓ | ✓ | ✓ |
| H3 KV/runtime layout |  |  | ✓ | ★★★ |  |  |  |
| H4 joint L/I/P | ★ | ✓ | ★ | ★ | ★ | ✓ | ★ |
| H5 cost-model fidelity | ★ | ✓ | ★ | ★ | ★ | ✓ | ★ |
| H6 conversion-aware | ✓ | ★★★ | ★ | ✓ | ★ | ★★★ | ✓ |
| H7 workload-aware |  | ✓ | ★ | ★★★ | ★ | ✓ | ✓ |

其中最适合形成一条论文故事线的是：

> **H1 + H4 + H5 + H6**

也就是：

```text
现有系统:
per-op legality/layout synthesis
       +
instruction selection
       +
local cost model

问题:
缺乏 subgraph-global layout optimization

我们的方案:
cross-operator
layout + instruction + conversion
joint optimization
       +
hardware-aware ranking model
```

---

# 二十三、怎么区分“科研问题”和“工程问题”

这是你特别强调的，因此最好从一开始就写死。

下面不是科学问题：

> Triton attention kernel block size 不好。

这是工程问题。

下面才是科学问题：

> 为什么 per-kernel block/layout optimization 无法保证 multi-operator dataflow 的最优性？

---

下面不是科学问题：

> Hexcute 没建模 mbarrier。

这是 implementation limitation。

真正的科学问题是：

> 在 asynchronous GPU pipeline 中，layout 的性能排序是否能够由 instruction-local additive cost model正确预测？

---

下面不是科学问题：

> vLLM KV cache layout 不够快。

而应该变成：

> serving-oriented paging layout 与 compute-oriented thread mapping 之间是否存在系统性的 optimization conflict？

这个 distinction 非常重要。

---

# 二十四、最终 benchmark 不能只看 latency

每次实验至少采：

```text
Latency
TFLOP/s
effective bandwidth

DRAM bytes
L2 hit rate
L1 hit rate

global load/store transactions
memory sectors
shared-memory bank conflicts

register/thread
shared memory/CTA
occupancy

warp issue efficiency
eligible warps
stall reasons

instruction counts:
LDG
STG
LDS
STS
LDMATRIX
MMA
WGMMA
CPASYNC
TMA
```

工具建议：

```bash
ncu
nsys
cuobjdump
nvdisasm
PTX dump
compiler IR dump
```

最后一定要把：

```text
layout decision
↓
generated instruction
↓
hardware counter
↓
performance
```

串起来。

否则 reviewer 很容易说：

> 只是 autotuning。

---

# 二十五、每个实验结果应该怎样保存

我强烈建议不要保存成随意的 txt。

统一：

```text
results/
 ├── machine.json
 ├── commit.json
 │
 ├── attention/
 │    ├── prefill/
 │    │     └── triton/
 │    │          ├── config.json
 │    │          ├── layout.json
 │    │          ├── compile/
 │    │          │    ├── ttir
 │    │          │    ├── ttgir
 │    │          │    ├── llvm.ll
 │    │          │    ├── ptx
 │    │          │    └── sass
 │    │          ├── benchmark.csv
 │    │          └── ncu.csv
```

主表：

```text
framework
subgraph
model
B
Sq
Skv
Hq
Hkv
D
dtype

tensor
memory_scope
shape
stride
layout

tile
warps
stages

load_instruction
store_instruction
mma_instruction

smem_bytes
registers
occupancy

dram_bytes
l2_hit
bank_conflicts

latency_us
throughput
```

这样后面才能真正回答：

> 为什么某一个 framework 的 layout decision 性能差？

---

# 二十六、一个关键的实验原则：一定要做 Oracle

假设 Triton 选：

```text
layout A
```

Hexcute 选：

```text
layout B
```

不能直接说 B 比 A 好。

需要构造：

```text
framework-selected
random
heuristic
oracle search
proposed
```

所以最终比较应该是：

\[
Regret_{framework}
=
\frac{T_{framework}-T_{oracle}}
{T_{oracle}}
\]

这才直接量化：

> “这个框架因为 layout 决策不好损失了多少性能”。

---

# 二十七、可行性评估：我建议优先攻哪三个

综合 novelty、实现复杂度、实验可解释性，我会排序成：

### 第一优先级：Subgraph-global layout synthesis

问题：

> operator-local layout ≠ subgraph-global layout。

假设：

> 保留 producer layout 并改变 consumer task mapping，比 layout conversion 更优。

可行性：

**很高。**

最容易先在：

```text
SwiGLU
QKV→RoPE→Attention
Attention→O projection
Diffusion GEGLU
```

验证。

---

### 第二优先级：KV layout × decode task mapping joint optimization

问题：

> paged memory management 和 kernel memory access 的目标冲突。

假设：

> runtime-aware physical KV layout + kernel thread mapping joint optimization 可以降低 decode memory traffic。

可行性：

**高。**

科研价值可能比普通 GEMM layout 高，因为这是现代 serving 独有问题。

---

### 第三优先级：Modern-GPU-aware layout cost model

问题：

> 当前 additive instruction latency model 会 mis-rank layout。

假设：

加入：

\[
C =
C_{instruction}
+
C_{bank}
+
C_{sync}
+
C_{occupancy}
+
C_{cache}
+
C_{conversion}
\]

能够降低 layout-selection regret。

可行性：

**中高。**

而且 Hexcute 当前源码已经给出了明确的 baseline limitation，非常方便写 motivation。fileciteturn10file0L1-L2

---

# 二十八、我目前认为最值得形成论文主线的题目

可以抽象成：

> **Subgraph-Level Layout Synthesis for Modern LLM Workloads**

核心论点：

```text
Current systems optimize:
operator
 ↓
tile
 ↓
instruction
 ↓
local layout

But modern LLM performance depends on:
producer layout
 ↓
intermediate layout
 ↓
consumer layout
 ↓
KV physical layout
 ↓
runtime phase
```

新的 optimization problem：

\[
\min_{\{L_t\},\{I_o\},\{P_o\}}
\sum_o C_o(L,I,P)
+
\lambda\sum_{(u,v)}
C_{conversion}(L_u,L_v)
\]

subject to：

\[
HardwareConstraints
\]

\[
InstructionConstraints
\]

\[
MemoryCapacityConstraints
\]

\[
LayoutCompatibilityConstraints
\]

这就已经不是简单 kernel engineering 了，而是一个比较完整的 compiler research problem。

---

# 二十九、最重要的一点：如何证明它“确实是科学问题”

最终论文必须完成下面的因果链，而不是只报 speedup：

```text
Observation
   ↓
同一子图存在多个合法 layout

Mechanism
   ↓
不同 layout 改变
coalescing / bank conflict /
instruction selection /
conversion cost / occupancy

Limitation
   ↓
现有 framework 的 optimization objective
没有表达这种 interaction

Hypothesis
   ↓
joint / global model 可以选择更优 layout

Prediction
   ↓
某类 shape / workload 下一定会出现 baseline regret

Experiment
   ↓
真实 LLM/Diffusion 子图验证 prediction

Result
   ↓
性能改善 + hardware counter 与假设一致
```

尤其最后这一点：

> **不仅 proposed 快，而且“为什么快”的硬件计数器必须与 hypothesis 的预测一致。**

这才是比较扎实的科研验证。

---

你后续整套研究中，我建议把 **Hexcute 当作最强的“自动 layout synthesis baseline”**，把 CUTLASS 当作 **expert/oracle-like baseline**，Triton/TVM 代表 compiler heuristic/search 路径，vLLM/SGLang 则代表 **real-world runtime layout constraint**。这样六个系统就不会被混在同一个抽象层，而是形成一条从 serving → graph → kernel → instruction → physical memory 的完整比较链。Hexcute artifact 也已经提供 A100/H100、MoE、attention、Mamba 等官方 benchmark 脚本，可以直接作为复现起点。citeturn259371search1

下一阶段最有价值的工作不是继续泛泛罗列框架，而是把这套思路落实成你最开始要的那张“大表”：**逐子图列出 SGLang/vLLM/CUTLASS/Triton/TVM/Hexcute 的 tensor layout、thread/value mapping、load/store/MMA 指令、源码位置、运行命令、输出文件，以及针对每一个 layout limitation 的最小反例和 proposed experiment。**这张表完成之后，基本就会变成你的实验计划和论文 problem statement。

## 用户

另外你对不同框架的对比讨论有点粗糙，请详细点

## 模型回复

你说得对。上一版把“框架类别”区分出来了，但还没有把真正决定性能的细节展开到：**layout 的表示、搜索空间、约束传播、指令选择、shared-memory swizzle、KV 物理组织、跨算子能力、cost model、动态 shape/phase 处理，以及这些机制分别在哪些子图上会失效**。这次我会按同一套维度逐框架拆开，并把能从源码直接证明的地方标出来。

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 2 个网站

## 模型工具调用参数

{"paths":["GitHub"],"query":"fetch"}

## 模型思考摘要

已搜索 2 个网站

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 SGLang 后端实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查找 SGLang KV 缓存布局代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 MHATokenToKVPool 相关内存池代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 MHATokenToKVPool 类定义

## 模型工具调用参数

{"uri":"/response/turn12","query":"class MHATokenToKVPool"}

## 模型工具调用参数

{"uri":"/response/turn12","query":"buffer_shape = (self.size"}

## 模型工具调用参数

{"uri":"/response/turn12","query":"def _create_buffers(self):"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 SGLang KV 缓冲区实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Triton 布局推断源码证据

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 AccelerateMatmul、点操作数优化与 Coalesce passes

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取相关矩阵乘法优化代码段

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

对比分析主流GPU编程框架

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 内存布局优化代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 TVM 布局重写与张量核心调度源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 apache/tvm 中的 layout_free_buffers

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索多级张量核心分块配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 CUTLASS Builder 与 SM90 布局选择逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 CUTLASS 自动阶段与内核调度实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 vLLM 后端定制规范与分页注意力布局

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 vLLM KV 缓存形状相关实现

## 模型思考摘要

搜索并对比 SGLang、Triton、TVM、CUTLASS 与 vLLM 布局优化实现

## 模型思考：工具活动

正在搜索 "Hexcute" compiler layout synthesis GPU programs CGO 2026

## 模型思考：工具活动

正在搜索 8 个网站

## 模型思考摘要

已搜索 8 个网站

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

比较 LLM 与扩散模型算子布局及后端

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索文档中的精确短语

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 vLLM KV Cache 布局与对齐规范

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查找 vLLM 缓存布局选择函数

## 模型思考摘要

定位布局优化
完成布局定位优化

## 模型思考摘要

比较并检索 LLM扩散模型及vLLM缓存布局规范

## 模型回复

已经能看到两个此前没有展开、但对你的课题非常重要的差别。第一，现代 Triton 本身也分成“经典 Triton：layout 主要由编译器 pass 隐式决定”和 Gluon：“layout 变成显式编程对象”两种模式；第二，vLLM 现在已经不只是单一旧式 PagedAttention layout，而是允许 backend 定制 KV cache spec，并支持 NHD/HND 等不同物理 stride，因此它更适合被视为“有限候选集上的 runtime layout contract”，而不是“固定 layout”。

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 SGLang KV 缓存布局变体

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析 SGLang 注意力后端选择机制

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考摘要

梳理对比边界

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 KVPool 实际缓冲区分配实现

## 模型思考摘要

检索并分析 SGLang KV 缓存与注意力后端实现

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 tvm.apache.org

## 模型思考摘要

完善比较框架
完善了比较框架

## 模型思考摘要

已搜索 1 个网站

## 模型思考用时

思考了 3m 7s

## 模型回复

是的。上一版最大的问题是只说了“谁更自动、谁更底层”，但没有把 **layout decision 到底包含哪些变量，以及六个框架分别在哪些变量上做决策**拆开。真正公平的比较应该先定义一个统一决策空间：

\[
\mathcal D=
\{
L_{\rm global},
L_{\rm shared},
L_{\rm register},
T_{\rm thread},
I_{\rm copy},
I_{\rm mma},
P_{\rm pipeline},
K_{\rm tile},
KV_{\rm page},
KV_{\rm layout},
B_{\rm backend}
\}
\]

这里分别对应全局内存布局、shared-memory layout、register/thread-value layout、线程映射、load/store/copy 指令、MMA 指令、pipeline、tile、KV page、KV 物理布局和 backend。然后才问：**SGLang / vLLM / CUTLASS / Triton / TVM / Hexcute 各自在优化哪一部分？谁负责剩下的部分？**

这一层拆开后，六个框架的差别会非常明显。

---

# 1. 六个框架实际上解决的是六种不同的 layout optimization problem

| 决策层 | SGLang | vLLM | CUTLASS/CuTe | Triton | TVM | Hexcute |
|---|---|---|---|---|---|---|
| Attention backend | ★★★ | ★★★ | — | — | — | — |
| KV page/block | ★★★ | ★★★ | — | — | — | — |
| KV physical layout | ★★ | ★★★ | kernel consumer | kernel consumer | kernel consumer | kernel consumer |
| GMEM tensor layout | 有限 | 有限 | ★★★ | ★★ | ★★★ | ★★★ |
| SMEM layout | — | — | ★★★ | ★★/★★★ | ★★/★★★ | ★★★ |
| Register/thread-value layout | — | — | ★★★ | ★★★ | TIRx: ★★★ | ★★★ |
| Copy instruction | — | — | 显式/heuristic | compiler | TIRx dispatch | 自动推导 |
| MMA instruction | — | — | 显式/heuristic | compiler | tensorize/dispatch | 自动推导 |
| Tile shape | backend 间接 | backend 间接 | programmer/builder | autotune/programmer | MetaSchedule/rule | programmer+推导 |
| Pipeline | backend kernel | backend kernel | 显式/Auto schedule | compiler + num_stages | schedule/TIRx explicit | **主要显式** |
| Layout 自动搜索 | 很弱 | 很弱 | 很弱 | heuristic | search 较强 | **最强** |
| 跨算子 layout | runtime 层有限 | runtime 层有限 | epilogue/mainloop | kernel 内 | Relax + kernel | kernel/tile program 内 |
| runtime workload awareness | ★★★ | ★★★ | 很弱 | autotune shape | 可做 | 很弱 |

所以如果你的论文说：

> “Hexcute 比 vLLM 的 layout optimization 更自动”

其实意义不大，因为两者根本不是在优化同一个变量集合。

更准确的定位应该是：

> **SGLang/vLLM 提供 outer-level runtime layout constraints；CUTLASS 提供 expert oracle；Triton 提供 compiler-heuristic baseline；TVM 提供 search-based baseline；Hexcute 提供 constraint-based layout synthesis baseline。**

---

# 2. SGLang：本质是“runtime layout policy”，不是 kernel layout synthesizer

SGLang最重要的 layout 决策不是：

\[
(thread,register,shared)
\]

而是：

\[
(\text{attention backend},
\text{KV representation},
\text{page size},
\text{cache placement})
\]

这一点从现在的源码很明显。

SGLang允许：

```text
prefill backend ≠ decode backend
```

当前 backend setup 在两者不同时，会显式构造 `HybridAttnBackend`：

```text
prefill_backend
decode_backend
        ↓
HybridAttnBackend
```

而不是强行让两个阶段使用相同 kernel。fileciteturn35file0L1-L7

这是很关键的设计，因为：

\[
S_q^{prefill}\gg1
\]

但：

\[
S_q^{decode}=1
\]

两个阶段一个更偏 compute-bound，一个更偏 KV-memory-bound。SGLang 的设计其实已经隐含承认：

\[
Backend^*_{prefill}\neq Backend^*_{decode}
\]

---

## SGLang 的 KV layout 到底是什么？

传统 MHA pool 的基本思想是 token slot 为主要索引，每层维护 K/V pool。SGLang 的测试实现中直接可看到：

```python
(self.size + self.page_size,
 self.head_num,
 self.head_dim)
```

这样的 K buffer。fileciteturn36file1L12-L21

但这并不是说 SGLang 永远只有一种 layout。

现在代码中特别说明：

> AITER 的 SHUFFLE 5D layout 只有 ROCm + AITER consumer kernel 才有意义，否则仍然采用 legacy NHD layout。fileciteturn34file0L1-L22

因此 SGLang 实际结构更像：

```text
model
  ↓
hardware/backend
  ↓
attention backend
  ↓
required KV representation
  ↓
memory pool implementation
```

而不是：

```text
enumerate arbitrary layouts
        ↓
benchmark
        ↓
choose best layout
```

---

## SGLang 解决得很好的是

它解决的是：

\[
\text{Serving constraints}
+
\text{KV lifetime}
+
\text{backend compatibility}
\]

例如 prefill/decode 分离、Radix cache、KV offload、speculative decoding、PD disaggregation 等。

尤其 PD disaggregation 中，prefill 和 decode 本来就是不同性能区域：官方文档也明确把 prefill 描述为偏计算密集、decode 描述为偏 KV/cache memory 密集。citeturn734032search5

---

## SGLang 没有解决的是

SGLang 不会从：

```text
KV tensor
     ↓
warp mapping
     ↓
vector width
     ↓
shared layout
     ↓
MMA fragment
```

自动生成最优方案。

它实际做的是：

> **选择已经人工实现好的 attention backend / cache representation。**

所以它的问题是一个离散选择：

\[
B^*
=
\arg\min_{B\in\{
FlashInfer,Triton,FA,\ldots
\}} T(B)
\]

而不是：

\[
L^*=\arg\min_{L\in\mathcal{L}}T(L)
\]

这两个 optimization space 差很多。

---

# 3. vLLM：比我上一版说的“PagedAttention 固定 layout”复杂得多

这里需要修正上一版过于简单的表述。

早期经典 vLLM PagedAttention 确实有非常具体的布局：

\[
K:
[num\_blocks,
num\_kv\_heads,
head\_size/x,
block\_size,
x]
\]

而 V 是：

\[
V:
[num\_blocks,
num\_kv\_heads,
head\_size,
block\_size]
\]

官方 PagedAttention 文档仍然清楚展示了这一点。citeturn734032search0

为什么 K 会多一个 `x`？

本质是：

\[
x=\frac{16\ \text{bytes}}{sizeof(dtype)}
\]

通过 vectorized load 让 thread/warp 更容易形成合并访问。

这本身就是一个典型的：

> logical tensor layout 为 kernel memory access pattern 服务。

---

# 4. 但现代 vLLM 已经变成“backend-specific KV layout contract”

当前 vLLM 的 `AttentionBackend` 有：

```python
customize_spec(...)
```

源码的注释非常明确：

> Adjust the layer's KV cache spec for this backend's kernels. Used when the kernels want KV packed in a specific way.

fileciteturn31file0L1-L43

所以现在正确的理解不是：

```text
vLLM
 ↓
one PagedAttention layout
```

而是：

```text
vLLM runtime
      ↓
AttentionBackend
      ↓
KV cache specification
      ↓
physical stride/layout
```

---

# 5. vLLM 现在甚至显式区分 NHD / HND physical layout

源码里已经有：

```text
NHD
HND
```

并映射到内部 layout 表示：

```text
NHD → LBNHC
HND → LBHNC
```

fileciteturn33file0L1-L19 fileciteturn33file1L21-L35

现在很多 kernel 不再假设 KV 是 C-contiguous，而是读取真实 stride。

例如当前 Triton attention backend 明确写：

> physical layout may be NHD, so do not assume C-contiguous HND layout.

fileciteturn32file1L20-L46

甚至 fused QKNorm + RoPE + KV insertion kernel 也是：

> kernel honours whatever physical layout the attention backend allocated.

fileciteturn32file3L69-L89

这对你的研究很重要。

因为 vLLM 已经自然提供了一个实验变量：

\[
KVLayout\in\{NHD,HND,\ldots\}
\]

---

# 6. 但 vLLM 仍然不是 layout synthesizer

它做的是：

\[
\text{select compatible layout}
\]

而不是：

\[
\text{synthesize arbitrary layout}
\]

这意味着它的 layout space 类似：

```text
NHD
HND
special packed layout
quantized layout
backend-specific layout
```

而不是：

```text
arbitrary hierarchical mapping
logical coordinate
      ↓
block
      ↓
warp
      ↓
lane
      ↓
register
```

因此 vLLM 非常适合研究：

> **runtime-oriented KV layout 与 kernel-optimal layout 是否冲突？**

例如一个 layout 可能对 attention kernel 最优，但对：

```text
KV transfer
prefix caching
PD disaggregation
cross-layer RDMA
```

不好。

当前 vLLM 甚至专门提供了 cross-layer contiguous layout 用于 KV transfer。fileciteturn33file2L38-L49

这实际上对应一个很漂亮的科学问题：

\[
\min_L
\left[
T_{\rm attention}(L)
+
T_{\rm KV-transfer}(L)
+
T_{\rm cache-management}(L)
\right]
\]

而不是仅：

\[
\min_L T_{\rm attention}(L)
\]

---

# 7. CUTLASS/CuTe：layout 表达能力最强之一，但 automation 很弱

CUTLASS 与前两个系统完全不同。

CuTe 中：

\[
Layout:
Logical\ Coordinate
\rightarrow
Physical\ Coordinate
\]

核心对象是：

```text
Shape
Stride
Layout
Tensor
Mma_Atom
Copy_Atom
TiledMma
TiledCopy
```

CUTLASS 3.x 的层次非常明确：

```text
Device
 ↓
Kernel
 ↓
Collective
 ↓
TiledMma / TiledCopy
 ↓
MmaAtom / CopyAtom
```

citeturn667413search1

---

# 8. CUTLASS 的真正优势：layout 和 instruction 是统一描述的

例如 GEMM 数据路径可以写成：

```text
A/B global layout
       ↓
GmemTiledCopy
       ↓
SmemLayoutAtom
       ↓
SmemCopyAtom
       ↓
register fragment
       ↓
TiledMma
```

CUTLASS 的 `CollectiveMma` 参数甚至直接包括：

```text
StrideA
StrideB

TiledMma

GmemTiledCopyA
SmemLayoutAtomA
SmemCopyAtomA

GmemTiledCopyB
SmemLayoutAtomB
SmemCopyAtomB
```

citeturn667413search1

因此 layout 不是孤立属性：

\[
L_{smem}
\leftrightarrow
I_{copy}
\leftrightarrow
I_{mma}
\]

是天然耦合的。

---

# 9. CUTLASS 的 `Auto` 到底自动了什么？

这里很容易误解。

CUTLASS有：

```cpp
StageCountAuto
KernelScheduleAuto
```

但它们不是一个：

> 任意 layout search algorithm。

源码可以看到 `KernelScheduleAuto` 本质上是一个 schedule tag，builder 根据：

```text
architecture
dtype
alignment
tile shape
cluster shape
```

等条件选一套已经设计好的 schedule specialization。fileciteturn27file0L1-L16

甚至一些 example 会人工写：

```text
if tile_M == 64:
    Pingpong
else:
    Cooperative
```

式的 schedule selection。fileciteturn27file5L87-L102

所以 CUTLASS 的 automation 更接近：

\[
\text{expert rule selection}
\]

不是：

\[
\text{general layout synthesis}
\]

---

# 10. CUTLASS 对你的研究应该是什么角色？

不是最主要“自动编译器 baseline”。

而应该是：

> **expert implementation / oracle-like upper bound。**

例如你搜索得到一个：

```text
layout L*
```

可以手写一个对应 CUTLASS/CuTe kernel。

如果：

\[
T_{proposed}\approx T_{CUTLASS-expert}
\]

说明自动方案已经逼近专家实现。

但 CUTLASS 最大科研弱点恰好是：

> layout 空间很强，但自动发现能力弱。

也就是：

\[
Expressiveness \uparrow
\quad
Automation \downarrow
\]

---

# 11. Triton：必须分“经典 Triton”和 Gluon

这是上一版没有讲清楚的一个重要问题。

经典 Triton 用户通常写：

```python
offs_m
offs_n
tl.load
tl.dot
tl.store
```

用户主要决定：

```text
BLOCK_M
BLOCK_N
BLOCK_K
num_warps
num_stages
program_id mapping
```

而：

```text
thread → element
register distribution
dot operand layout
shared layout
```

很多由 compiler pass 决定。

---

# 12. 经典 Triton 的 layout optimization 是“pass-driven local heuristics”

例如 `Coalesce` pass 的定义明确说：

> 为 memory operation 选择 coalesced layout，并在 load/store 前后插入 layout conversion 来保持程序其余部分一致。

fileciteturn22file2L45-L74

也就是：

```text
existing layout
      ↓
convert_layout
      ↓
coalesced load/store layout
      ↓
load/store
      ↓
convert_layout
      ↓
consumer layout
```

局部看 load 很漂亮。

但整个子图可能变成：

\[
T=
T_{\rm load-opt}
+
T_{\rm convert}
\]

这正是你应该攻击的点。

---

# 13. Triton 的 MMA layout 也是另一个 pass 决定

当前 `AccelerateMatmul.cpp` 会识别 `dot`，然后：

```text
Blocked layout
      ↓
MMA encoding
      ↓
convert dot operand
      ↓
MMA
      ↓
ConvertLayout back
```

代码里可以直接看到：

```cpp
createMMAEncodingForDot(...)
convertDotOperandForMMA(...)
getSharedMemoryMMAOperand(...)
...
replaceOpWithNewOp<ConvertLayoutOp>
```

fileciteturn20file0L1-L2

这说明 Triton 的核心策略是：

> 一个 compiler pass 针对一个局部 operation 生成一个适合它的 layout。

这和 Hexcute 的 global constraint propagation 非常不一样。

---

# 14. Triton 的科研缺口因此特别清楚

假设：

```text
Load
 ↓
Transpose
 ↓
Dot
 ↓
Reduce
```

那么可能分别出现：

```text
Lload
 ↓ convert
Ldot
 ↓ convert
Lreduce
```

每一个：

\[
L_i
\]

可能都是当前 operation 的好 layout。

但：

\[
\sum_i C_i(L_i)
+
\sum_{i,j} C_{convert}(L_i,L_j)
\]

可能并不好。

这是一个非常适合你论文的反例。

---

# 15. Gluon 改变了 Triton 的这个边界

现代 Triton 的 Gluon 已经开始显式暴露：

```text
BlockedLayout
DotOperandLayout
NVMMADistributedLayout
NVMMASharedLayout
SwizzledSharedLayout
LinearLayout
```

citeturn667413search2

例如：

```python
BlockedLayout(
 size_per_thread,
 threads_per_warp,
 warps_per_cta,
 order
)
```

直接定义：

\[
register
\times lane
\times warp
\times CTA
\rightarrow tensor
\]

citeturn667413search3

这使 Triton/Gluon 在 expressiveness 上开始明显接近 CuTe。

---

# 16. Triton 官方文档甚至已经直接承认 layout conversion trade-off

Gluon layout tutorial 明确说：

> reduction 本身 compiler 通常可以在不同 layout 下生成高效代码，所以有时先 `convert_layout` 到“理想 reduction layout”反而得不偿失。

并明确指出 shared memory bank conflicts 同时受到 shared layout 和 register layout 影响。citeturn667413search0

这几乎直接支持你的 H1：

\[
Local\ layout\ optimality
\not\Rightarrow
Program\ optimality
\]

---

# 17. TVM：这里也需要修正之前“只是 MetaSchedule”的粗略描述

2026 年的 TVM 已经明显分成两个世界：

```text
经典 TensorIR / MetaSchedule
            +
       新 TIRx
```

两者应该分别分析。

---

# 18. TVM 经典路线：schedule search，而不是纯 layout synthesis

MetaSchedule 搜的是：

```text
loop tiling
thread binding
vectorization
unroll
cache read/write
tensorization
software pipeline
```

官方定义就是：

> search different TIR schedules and measure them on real hardware.

citeturn429224search1

它的重要优势是：

\[
Cost(schedule)=MeasuredLatency
\]

而不是完全依靠 analytic model。

这是 Hexcute 没有的优势。

---

# 19. TVM 也确实会搜索 layout-related decisions

例如 CUDA TensorCore schedule rule：

```text
MultiLevelTilingTensorCore
```

搜索空间包含：

```text
intrinsic group
tiling
thread binding
vector load lengths
reuse_read
reuse_write
software pipeline
```

fileciteturn25file0L1-L13 fileciteturn25file2L32-L50

而 `RewriteLayout` 还能改变标记为：

```text
layout_free_buffers
```

的输入 tensor 布局。fileciteturn24file1L14-L26

所以 TVM 绝不能简单说：

> 不做 layout optimization。

它做，只是抽象不同。

---

# 20. 经典 TVM 的限制是什么？

最重要的是 layout optimization 和 schedule optimization 有一定“分裂”。

典型：

```text
weight layout rewrite
loop tiling
cache_read
cooperative fetch
tensorization
```

这些一起形成最终 kernel。

但它不像 CuTe/Hexcute 那样从一开始就统一表示：

\[
ThreadValueLayout
\]

并对：

\[
Copy
\leftrightarrow MMA
\leftrightarrow Tensor
\]

建立显式代数约束。

所以 MetaSchedule 非常擅长：

> 搜一个已有 schedule space。

却不一定擅长：

> 自动发现一个之前 schedule rule 没表达出来的新 layout family。

换句话说：

\[
Search\ quality
\le
Search\ space\ quality
\]

---

# 21. 2026 TVM 的 TIRx 更值得关注

TIRx 已经引入真正的 first-class Tensor Layout：

\[
LogicalTensor
\rightarrow
PhysicalResources
\]

支持：

```text
memory
lane
warp
device
TMEM
```

等具名 hardware axis。

并且把 layout 分成：

\[
D=\text{Shard}
\]

\[
R=\text{Replica}
\]

\[
O=\text{Offset}
\]

还用 `ComposeLayout` 显式表示 shared-memory swizzle。citeturn201960search0turn201960search1

这已经非常接近你的研究对象。

---

# 22. 但 TIRx 与 CuTe 又有一个非常关键的区别

TIRx 官方专门解释：

> CuTe 把 layout 同时作为 storage representation 和 work-partition interface。

而 TIRx：

> layout 是 **storage contract**；真正的 thread partitioning、loop、instruction 由 tile primitive dispatch 根据 layout + execution scope 推出来。

citeturn825805search0

所以：

### CuTe

```text
Layout
 ↓
tile
 ↓
partition
 ↓
thread work
```

### TIRx

```text
Storage Layout
       +
Execution Scope
       ↓
Primitive Dispatch
       ↓
thread partition
instruction
```

这是非常本质的差别。

---

# 23. TVM 现在反而出现了一个很好的研究空白

TIRx 当前的 philosophy 是：

> pipeline、synchronization、role assignment、memory placement 等 frontier optimizations 保留给 programmer。

官方甚至明确说 higher-level：

```text
schedule search
pipelining
performance model
agent tuning
```

可以建立在 TIRx 之上，而不是 TIRx core 本身自动解决。citeturn825805search0

于是一个很自然的科研问题是：

> **能否把 MetaSchedule 的自动搜索能力，与 TIRx 的 first-class layout representation 结合？**

也就是：

\[
MetaSchedule
+
TIRx Layout
\]

可能形成一个比旧 TVM 更强的 layout optimizer。

---

# 24. Hexcute：六个框架里最接近你真正研究对象的 baseline

Hexcute 的核心不是：

```text
enumerate tile sizes
```

而是：

> layout legality 和 instruction compatibility 可以通过 constraint solving 推导。

CGO 2026 官方描述就是：

> formalizes layout synthesis as a constraint programming problem and solves it with a type-inference-based algorithm。

citeturn212797search0turn212797search9

---

# 25. Hexcute 的自动化究竟比 Triton/CUTLASS 多在哪里？

Hidet/Hexcute 当前代码将过程明确拆成：

```text
logical shape inference

logical layout inference

task mapping inference

memory layout inference
```

fileciteturn9file0L1-L2

比如 Copy。

假设 hardware copy instruction 的 src/dst TV mapping 是：

\[
p,q
\]

而程序中的 tensor mapping 是：

\[
f,g
\]

它建立：

\[
f\circ p^{-1}
=
g\circ q^{-1}
\]

然后由：

\[
g
\]

推：

\[
f
=
g\circ q^{-1}\circ p
\]

于是不是 programmer 手写：

```text
thread 0 gets element ...
thread 1 gets element ...
```

而是通过 instruction compatibility 自动推导。

---

# 26. MMA 也一样

对于：

\[
C=A B
\]

它把 A、B、C 的 TV mapping 分解成：

\[
M,N,K
\]

mode，然后要求：

\[
f_{A,M}=f_{C,M}
\]

\[
f_{B,N}=f_{C,N}
\]

\[
f_{A,K}=f_{B,K}
\]

再自动推导兼容 layout。

这个思想和 CUTLASS 完全不同：

CUTLASS：

```text
programmer chooses valid composition
```

Hexcute：

```text
compiler solves valid composition
```

---

# 27. Hexcute 还有一层 instruction selection

当前 instruction selection 会根据：

```text
src layout
dst layout
alignment
max_common_vector
memory scope
```

检查：

```text
ldg128
ldg64
...
lds128
ldmatrix
cp_async
mma_sync
wgmma
TMA
```

等 candidate 是否匹配。fileciteturn7file0L1-L2

所以 Hexcute 的 search variable 实际已经接近：

\[
(L,T,I)
\]

即：

```text
Layout
Task Mapping
Instruction
```

联合选择。

这比经典 Triton 和 TVM MetaSchedule 更接近你的目标。

---

# 28. 但 Hexcute 仍然没有解决整个问题

这是你的论文真正应该深入的地方。

Hexcute 当前的自动 layout synthesis 并不等于：

\[
\min_{\text{all possible GPU programs}} T
\]

它更接近：

\[
\min_{L,T,I}
T(L,T,I\mid
\text{fixed dataflow/pipeline})
\]

因为 Hexcute 的设计目标之一就是：

> 自动 layout，但保留 programmer 对 dataflow 和 pipeline 的显式控制。

官方论文摘要对此也写得很明确：

> automates layout synthesis while providing explicit control over dataflow and pipelining.

citeturn212797search0

这意味着：

\[
Dataflow
\]

和：

\[
Pipeline
\]

仍然会限制 layout search space。

---

# 29. Hexcute 的 cost model 也是一个非常明确的攻击点

当前代码把 tile primitive 大致建模为：

\[
Cost
=
N_{\rm instructions}
\times
Latency_{\rm instruction}
\]

并假定 Copy 与 MMA 可以很好 overlap。

源码又明确写：

```text
bank conflict impact: future work

cp_async_wait_group: not considered

mbarrier: not considered

pipeline assumed fully overlapped
```

fileciteturn10file0L1-L2

然而另外一个 pass 又专门：

```text
enumerate swizzles
 ↓
calculate bank conflicts
 ↓
select minimum-conflict swizzle
```

fileciteturn8file0L1-L2

也就是说当前架构中有：

```text
layout synthesis
      ↓
cost model

      ↓

bank-conflict repair
```

而不是：

\[
\arg\min_L
\left(
InstructionCost+
BankConflictCost
\right)
\]

这是非常明显的科研空间。

---

# 30. 把六个系统放到一个“自动化—表达能力”坐标上会更准确

| Framework | Layout 表达能力 | 自动选择能力 | runtime awareness | 典型目标 |
|---|---:|---:|---:|---|
| SGLang | 低（kernel 内）/中（KV） | 中 | 极高 | Serving |
| vLLM | 中（KV） | 中 | 极高 | Serving/KV |
| CUTLASS | **极高** | 低 | 低 | Expert kernels |
| Triton classic | 中高 | **中高 heuristic** | 中 | Kernel compiler |
| Triton Gluon | **很高** | 较低/中 | 中 | Explicit kernel DSL |
| TVM MetaSchedule | 中高 | **很高 search** | 中 | Auto-tuning |
| TVM TIRx | **极高** | 目前较低 | 中 | Frontier kernel DSL |
| Hexcute | **很高** | **很高 synthesis** | 低 | Automatic layout synthesis |

所以其实不存在：

> “哪个最好？”

而是有四条不同设计哲学。

```text
CUTLASS:
Express everything,
expert chooses.

Triton:
Express less,
compiler heuristics choose.

TVM:
Define schedule/search space,
measure candidates.

Hexcute:
Express dataflow,
solve layout constraints.

SGLang/vLLM:
Select runtime-compatible kernel/cache contracts.
```

---

# 31. 再落回你最关心的 LLM 子图，对比会更明显

| 子图 | SGLang/vLLM | CUTLASS | Triton | TVM | Hexcute |
|---|---|---|---|---|---|
| GEMM | 选择库/kernel | 专家设计 Layout+MMA+pipeline | block + compiler MMA layout | schedule search / TIRx explicit | 自动推 L/T/I |
| RMSNorm | fused kernel 选择 | 手写 epilogue/collective | blocked/reduction heuristic | reduction schedule | 自动 task layout |
| QKV GEMM | backend/kernel | expert GEMM | dot MMA | tensorize/search | auto MMA layout |
| RoPE | fused/独立 kernel | 手写 | elementwise layout | schedule | constraint propagation |
| Prefill Attention | **backend selection** | 手写 FA-style pipeline | compiler/local layout | TIRx/MetaSchedule | auto L/T/I, pipeline mostly固定 |
| Decode Attention | **KV layout/page 最重要** | consumer kernel | consumer kernel | consumer kernel | consumer kernel |
| SwiGLU | kernel/fusion | epilogue/fused GEMM | kernel fusion | Relax+TIR | tile program |
| MoE | routing + kernel | grouped GEMM expert | heuristic/custom Triton | schedule search | **强项** |
| Mamba | state/runtime | 手写 | custom | custom/TIRx | 已评测 |
| Diffusion Conv | 非主要目标 | Conv/GEMM | custom | **较适合 graph+kernel** | 可表达但 benchmark coverage 较少 |

Hexcute 的 CGO 评测已经覆盖 GEMM、Attention、FP8、warp specialization、mixed-type MoE、Mamba；官方 artifact 也提供对应实验。citeturn212797search12

---

# 32. 对 Decode Attention，六者的差距最有研究价值

这是我认为你最应该单独拉出来的一组。

Decode：

\[
Q:[B,H_q,1,D]
\]

KV：

\[
K,V:
[B,H_{kv},S,D]
\]

核心瓶颈通常不是：

\[
QK^T
\]

的 FLOPs，而是：

\[
Bytes(KV)
\]

所以真正的 decision chain 是：

```text
runtime page layout
      ↓
block table
      ↓
KV physical order
      ↓
warp mapping
      ↓
vector load
      ↓
register layout
      ↓
dot/reduce
```

现在的问题是：

```text
vLLM/SGLang
optimize upper half

CUTLASS/Triton/Hexcute
optimize lower half
```

几乎没有系统完整地联合优化：

\[
KV_{\rm runtime}
+
L_{\rm kernel}
+
T_{\rm thread}
+
I_{\rm load}
\]

这比普通 GEMM layout 更可能产生新的科研贡献。

---

# 33. 对 Prefill Attention，研究问题则不同

Prefill 是：

```text
QK^T
 ↓
scale/mask
 ↓
softmax
 ↓
PV
```

这里有至少三个 layout domain：

\[
L_Q,L_K
\]

决定 QK MMA。

中间：

\[
L_P
\]

需要服务：

```text
softmax reduction
```

然后又必须服务：

\[
PV
\]

所以：

\[
L_P^{softmax-opt}
\]

不一定等于：

\[
L_P^{PV-opt}
\]

这是最好的 cross-operator layout example 之一：

\[
QK
\rightarrow
Softmax
\rightarrow
PV
\]

现有系统的典型做法是依赖精心设计的数据流。

但你的研究可以问：

> 能不能自动推导整个 chain 的最优 layout，而不是分别满足 QK、softmax、PV？

这比“自动找 GEMM shared memory swizzle”科学问题更强。

---

# 34. 对 FFN/SwiGLU，又是另外一种 layout conflict

典型：

\[
XW_{gate}
\]

\[
XW_{up}
\]

然后：

\[
SiLU(gate)\odot up
\]

最后：

\[
YW_{down}
\]

最大的 layout 问题不是某一个 GEMM。

而是：

```text
GEMM output layout
       ↓
SiLU/mul layout
       ↓
next GEMM input layout
```

如果 GEMM1 为了自己的 tensor core fragment 得到：

\[
L_1
\]

而 GEMM2 最喜欢：

\[
L_2
\]

那么中间可能发生：

\[
Rearrange(L_1,L_2)
\]

真正目标应该是：

\[
T_{GEMM1}
+
T_{activation}
+
T_{convert}
+
T_{GEMM2}
\]

而不是每个 operator 独立优化。

---

# 35. 对 MoE，六个框架差距又换成了“irregularity”

MoE 中：

```text
token routing
      ↓
variable expert sizes
      ↓
grouped GEMM
      ↓
mixed precision / scales
```

于是 shape 本身是动态的。

CUTLASS 可以写极强 grouped GEMM，但需要大量专家设计。

Triton 对复杂 mixed-type layout 往往需要人工 workaround。

Hexcute 的论文正好在这里展示了优势：mixed-type MoE 相对 Triton 平均达到 6.46× speedup。citeturn212797search0

但这也引出一个你可以继续研究的问题：

> Hexcute 的 layout synthesis 是针对一个 shape；MoE runtime 下 expert workload distribution 变化时，静态最优 layout 是否仍然最优？

这就是：

\[
L^*=f(shape)
\]

扩展成：

\[
L^*=f(shape,\ expert\ distribution)
\]

---

# 36. 对 Diffusion，TVM/Triton/CUTLASS 的价值会明显高于 SGLang/vLLM

Diffusion 的关键高占比子图通常包含：

```text
Conv / Linear
GroupNorm
SiLU
Attention
GEGLU
Upsample/Downsample
```

SGLang/vLLM本质不是 Diffusion compiler。

因此论文实验不应该强行写：

> Diffusion 在 SGLang 上测试 layout。

应该写：

```text
LLM runtime experiments:
SGLang
vLLM

Kernel/subgraph experiments:
CUTLASS
Triton
TVM/TIRx
Hexcute
```

这才公平。

---

# 37. 如果严格评价“谁在 layout optimization 上更先进”，应该分四个指标

| 指标 | 最强候选 |
|---|---|
| Layout expression power | CuTe / TIRx |
| Automatic layout synthesis | Hexcute |
| Empirical search / autotuning | TVM MetaSchedule |
| Runtime-aware KV/cache decision | SGLang / vLLM |
| High-level compiler heuristics | Triton |
| Expert peak performance baseline | CUTLASS |

所以你真正研究的机会不是简单：

> 做一个比六者都好的 layout algorithm。

更合理的是把他们没有连接起来的东西连接起来：

\[
Runtime
\leftrightarrow
Subgraph
\leftrightarrow
Layout
\leftrightarrow
Instruction
\leftrightarrow
Pipeline
\]

---

# 38. 我现在对这六个框架的科研定位会这样定义

| Framework | 在你的论文中的角色 | 核心对比问题 |
|---|---|---|
| SGLang | runtime baseline | backend/KV policy 是否限制 kernel-optimal layout |
| vLLM | runtime/KV baseline | paging/transfer/layout 与 decode access 是否冲突 |
| CUTLASS | expert oracle | 自动方案距离专家 layout 还有多少 |
| Triton | compiler heuristic baseline | local layout heuristic 是否导致 conversion/regret |
| TVM | search baseline | schedule search 能否发现 layout optimum |
| TIRx | expressive explicit-layout baseline | explicit layout 能否自动 synthesis |
| Hexcute | automatic synthesis baseline | constraint synthesis 是否仍存在 global/cost-model limitation |

这一定位比简单做：

```text
Triton vs CUTLASS vs TVM
latency
```

要有意义得多。

---

## 最后，一个我认为需要修正的核心研究方向

上一版我把问题概括成了：

> subgraph-level layout synthesis。

现在进一步看完这些框架以后，我认为应该更精确地改成：

\[
\boxed{
\textbf{Cross-Layer Subgraph Layout Optimization}
}
\]

这里的 `Layer` 不是 Transformer layer，而是：

\[
Runtime
\rightarrow
Physical\ Tensor\ Layout
\rightarrow
Task\ Mapping
\rightarrow
Instruction
\rightarrow
Pipeline
\]

现有六个框架各自优化其中一段，但没有哪个真正把整个链条统一起来。

尤其是：

\[
\boxed{
KV\ Layout
+
Thread/Value\ Layout
+
Instruction
+
Subgraph\ Dataflow
}
\]

我认为是你这个方向最有潜力的一条主线。

下一步真正应该做的不是继续用文字泛谈框架，而是建立一张**源码级别的“子图 × 框架 × Tensor”大表**。具体要细化到例如 Prefill Attention 的 `Q/K/V/S/P/O`、Decode 的 `Q/KV/cache/page/block table`、FFN 的 `gate/up/intermediate/down`，逐个填出六个框架里的 **shape、stride/layout、thread mapping、SMEM layout、load/store、MMA、pipeline、决策代码位置、可调变量以及它为什么这么选**。做到这一层之后，我们才能严谨地设计每一个“现有框架一定会选错”的最小反例，而不是凭经验猜它可能不好。

## 用户

另外你对不同框架的对比也很草率（你可以先看看不同框架的优化链接，这里不清楚的再找找github源码， 总结不同的实现维度，尽可能在这些维度上全面的对比），所以最后的科研问题总结的也不好。如何验证每一个问题总结的也不好， 所以对应的代码设计也不好。

所以你先做不同框架的对比，以及给出更好的科研问题，我看了之后，同意了，我们再设计代码验证 该问题 和 思考解决方法

## 模型回复

我会先把“框架对比”做成真正可用于科研选题的 taxonomy，而不是按框架逐个介绍。重点会统一比较：**优化对象、layout 表示、搜索/推导空间、合法性约束、成本模型、指令选择、数据搬运、shared-memory bank conflict、pipeline、跨算子/子图能力、动态 shape/workload、KV cache、硬件适配、可解释性与搜索开销**。然后只基于这些差异提炼科学问题，不进入代码实现。

## 模型思考：工具活动

正在搜索 Hexcute compiler layout synthesis GPU programs Hidet CGO 2026

## 模型思考：工具活动

正在搜索 SGLang attention backend prefill decode FlashInfer Triton KV cache page size official docs

## 模型思考：工具活动

正在搜索网页

## 模型思考摘要

正在搜索 github.com

## 模型回复

共享 ChatGPT 链接当前无法从外部检索接口直接展开，所以我不会把其中未验证的内容当证据；我会用你之前链接里强调的概念作为检索线索，再回到官方文档和当前 GitHub 主线代码核对。现在已经有一个需要纠正的点：vLLM 当前的 KV layout 已经是模型级物理 layout descriptor + backend preference resolution，而不是旧文档里单一的 K/V 5D/4D PagedAttention 布局。

## 模型工具调用参数

{"paths":["GitHub"],"query":"search"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

Computing SGLang与Hexcute深度对比

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Hexcute 项目布局实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理不同编译框架的布局优化机制

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 Passes.td 相关代码行

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索TVM布局转换与MetaSchedule源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查询 vLLM KV 缓存布局解析

## 模型工具调用参数

{"paths":["GitHub"],"query":"fetch_file"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理主流推理框架内存布局证据

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取注意力后端工具文件内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

评估 SGLang 后端选择机制

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取注意力后端文档与默认推荐

## 模型思考摘要

完成SGLang与Hexcute布局机制及后端实现对比源码调研

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 triton-lang.org

## 模型思考摘要

梳理技术依据
梳理了技术依据

## 模型思考摘要

已搜索 1 个网站

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取并总结 Hidet CUDA 代价模型源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取成本模型与银行冲突信息

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

解决 CUDA 共享内存 Bank 冲突

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索指令选择与TMA布局匹配逻辑

## 模型思考摘要

获取并总结CUDA代价模型及共享内存优化逻辑

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 docs.nvidia.com

## 模型思考摘要

已搜索 1 个网站

## 模型思考用时

思考了 1m 28s

## 用户

另外还有就是不叫不同框架的layout的优化方法的区别。 只有从这些框架中的不同点和相同点中才能看出这些框架中的区别，才能更好的总结科研问题。

## 模型回复

对，这个纠正是关键。真正要比较的不是“六个框架各自怎么做 layout”，而是先建立一套共同维度，比较它们在**程序表示、可优化对象、搜索/推导方式、约束来源、性能评价、运行时信息、硬件映射、跨算子范围、人工介入程度**上的相同点和不同点；只有这些差异稳定地指向某种系统性缺口，才有资格抽象成科学问题。

## 模型思考：工具活动

正在搜索 CUTLASS CuTe layout tensor layout composition TiledMma TiledCopy CollectiveBuilder official docs

## 模型思考：工具活动

正在搜索 Hexcute github artifact benchmark Hidet CGO 2026

## 模型思考摘要

正在搜索 3 个网站

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索CUTLASS调度与布局源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

深入解析 Hexcute 分支及成本模型

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

深入分析 SGLang 后端自动选择机制

## 模型思考摘要

聚焦科学问题
聚焦科学问题完成梳理

## 模型思考摘要

检索并分析CUTLASS、Hexcute及SGLang源码机制

## 模型回复

从现有源码对齐后，差异已经比“谁自动、谁手工”具体很多。例如 Triton 自己就同时存在 coalescing、MMA layout、thread-locality、warp specialization、pipeline、layout-conversion elimination 等相互作用的 pass；Hexcute 则把“合法 layout/task mapping 的推导”系统化了，但 pipeline/dataflow 仍由程序表达，性能模型又与后续 bank-conflict 修复存在分层；vLLM 的决策则是全模型共享的 persistent KV physical layout 与多个 backend 的兼容性交集。  

## 模型思考用时

思考了 1m 9s

## 模型回复

对。前面的比较对象确实选窄了。不能先假设“这些都是 layout optimizer”，然后比较“layout 方法有什么不同”；应该反过来：

> 先看这些系统整体上在解决什么问题、哪些决策由谁做、硬件知识放在哪里、搜索空间在哪里、性能反馈从哪里来、优化边界在哪里。  
> 再从这些系统的共同点与结构性差异中，看哪些问题在所有方法里都反复出现，哪些是某种设计范式天然解决不了的。  
> 最后才能抽象科研问题。

我重新查了当前官方文档和 GitHub 主线。你给的 ChatGPT share 链接我也尝试直接读取，但接口无法展开，所以这版不依赖 share 内容作为证据，而是回到当前源码和官方文档核对。下面先不设计代码，只做两件事：**框架对比**和**科研问题筛选**。

---

# 一、先建立正确的比较对象：不是“六种 layout 方法”，而是六种优化系统

比较 SGLang / vLLM / CUTLASS / Triton / TVM / Hexcute，首先要问的是：

\[
\boxed{
\text{Who decides What, When, and according to Which information?}
}
\]

我建议把整个现代 GPU 大模型优化问题写成：

\[
\mathcal D =
\{
D_{\text{runtime}},
D_{\text{graph}},
D_{\text{dataflow}},
D_{\text{persistent-layout}},
D_{\text{tile}},
D_{\text{thread}},
D_{\text{memory}},
D_{\text{instruction}},
D_{\text{pipeline}}
\}
\]

也就是：

```text
Serving workload / runtime
          │
          ▼
Backend / algorithm
          │
          ▼
Subgraph / fusion / dataflow
          │
          ▼
Persistent tensor representation
(KV cache / weights / states)
          │
          ▼
Tile decomposition
          │
          ▼
Thread / warp / warpgroup mapping
          │
          ▼
GMEM / SMEM / registers / TMEM
          │
          ▼
LD/ST / TMA / cp.async / ldmatrix
MMA / WGMMA / tcgen05
          │
          ▼
Pipeline / barriers / warp specialization
```

六个系统切的是这条链条的不同区间。

---

# 二、第一层本质区别：它们优化的“边界”不同

| 系统 | 主要优化边界 | 最主要的输入信息 | 最主要的输出决策 |
|---|---|---|---|
| SGLang | Serving/runtime | model、hardware、prefill/decode、speculative、page、quantization | backend、cache policy、page/cache representation |
| vLLM | Serving/runtime + persistent state | model、attention backend、connector、KV spec | KV physical layout、block/page、backend compatibility |
| CUTLASS/CuTe | Kernel implementation | dtype、shape、arch、instruction family | tile、copy、MMA、SMEM layout、pipeline |
| Triton classic | Kernel compiler | tensor program、shape、target | distributed layout、coalescing、MMA conversion、pipeline |
| Triton/Gluon | Explicit kernel DSL | programmer-provided layout/dataflow | explicit distributed/shared layouts + lowering |
| TVM MetaSchedule | Schedule search | TIR workload、target、measured latency | tiling、binding、vectorization、reuse、tensorization |
| TVM TIRx | Explicit kernel IR + dispatch | storage layout、execution scope、target | primitive implementation / native instruction |
| Hexcute | Automatic kernel synthesis | tile-level dataflow、constraints、target | task mapping、memory layout、copy/MMA instruction |

这已经揭示第一件很重要的事情：

> **SGLang/vLLM 和 CUTLASS/Triton/TVM/Hexcute 的最大区别，不是 layout 表示不同，而是优化边界根本不同。**

---

# 三、它们又有一个很重要的共同点：所有系统其实都在编码“GPU 专家知识”

只是专家知识放的位置不同。

| 系统 | GPU 专家知识主要藏在哪里 |
|---|---|
| CUTLASS | C++ template specialization、CuTe layout、CollectiveBuilder、schedule tag |
| Triton | compiler passes、encoding rules、pattern rewrite、heuristics |
| TVM MetaSchedule | schedule rules、tensor intrinsics、search space、cost model |
| TIRx | TilePrimitive dispatch predicates + user kernel orchestration |
| Hexcute | layout constraints、instruction constraints、cost model |
| vLLM | attention backend capability、KV layout compatibility/preferences |
| SGLang | backend registry、model/hardware overrides、feature compatibility |

这是后面科研问题特别重要的一条轴：

\[
\boxed{\text{区别不是有没有专家知识，而是专家知识如何表示、组合和搜索}}
\]

---

# 四、CUTLASS：Expert Construction，而不是一般意义上的自动搜索

CUTLASS/CuTe 是六者中非常特殊的一个。

CuTe 的核心不是简单的 row-major / column-major，而是 hierarchical layout algebra：

\[
\text{logical coordinate}
\rightarrow
\text{hierarchical physical coordinate}
\]

可以组合、tiling、partition。

CUTLASS 3.x 的 GEMM Collective 直接把这些东西作为一等参数：

\[
Stride_A,\ Stride_B
\]

\[
TiledMma
\]

\[
GmemTiledCopy_A,\ GmemTiledCopy_B
\]

\[
SmemLayoutAtom_A,\ SmemLayoutAtom_B
\]

\[
SmemCopyAtom_A,\ SmemCopyAtom_B
\]

也就是说，在 CUTLASS 中：

\[
\boxed{
Layout + Copy + MMA + Pipeline
}
\]

从 API 层面就是共同设计的，而不是几个互不相干的优化 pass。citeturn586462search2turn586462search7

这一点非常重要。

CUTLASS 的强项不是“会选 layout”，而是：

> **它允许专家精确构造一个硬件一致的实现。**

所以其搜索空间表达能力极强。

但是它的 `KernelScheduleAuto` 不等于 layout search。

源码里可以直接看到 `KernelScheduleAuto` 最终进入不同 architecture-specific `CollectiveBuilder` specialization；甚至一些 builder 中就是 compile-time 条件判断 cluster shape 后选择某个 schedule family。fileciteturn51file1L19-L34 fileciteturn51file5L83-L97

因此更准确地说：

\[
\text{CUTLASS Auto}
\approx
\text{expert rule selection}
\]

而不是：

\[
\arg\min_{L,I,P} T(L,I,P)
\]

---

# 五、CUTLASS 对研究最重要的意义

它提供的是一个非常好的：

\[
\boxed{\text{Expert oracle / expressiveness baseline}}
\]

也就是说，对某个 GEMM / Attention / MoE：

如果人工 CuTe 可以实现配置 \(C^*\)，而自动系统找不到，

真正的问题不是：

> “这个 layout 不支持。”

而是：

> **自动决策机制没有发现一个已经可以表达的优质解。**

这一区分很重要：

\[
\text{Expressiveness gap}
\neq
\text{Search gap}
\]

后面我们应该专门测这两个 gap。

---

# 六、Triton classic：与 CUTLASS 最大区别不是 DSL 高低，而是“决策被拆到 compiler pass 中”

Triton 当前 GPU lowering pipeline 中，可以明确看到：

`Coalesce`：

> 分析 load/store，并换成更适合 coalescing 的 layout；必要时在前后插入 `ConvertLayout`。

`AccelerateMatmul`：

> 改变 dot 输入/输出 layout，使其适配硬件 accelerator。

`OptimizeDotOperands`：

> rearrange operand layout，尝试利用硬件 transpose。

`RemoveLayoutConversions`：

> 尝试减少 ConvertLayout，并在昂贵 memory access 与 MMA-friendly layout 之间做 heuristic trade-off。

`OptimizeThreadLocality`：

> 对 reduction/gather 尝试减少跨线程通信。

此外还有：

`Pipeline`、`AutomaticWarpSpecialization`、`PartitionScheduling`、`Prefetch`、register-pressure-related reorder 等。fileciteturn40file0L1-L2

这说明 Triton 实际不是：

```text
choose one layout
```

而更像：

```text
Memory access
    ↓ Coalesce

Dot
    ↓ MMA encoding

Reduction
    ↓ locality layout

Pipeline
    ↓ async scheduling

Then:
    remove / propagate / rematerialize
    layout conversions
```

这和 CUTLASS 是根本不同的设计。

CUTLASS：

\[
\boxed{\text{construct a coherent configuration}}
\]

Triton：

\[
\boxed{\text{progressively rewrite toward locally favorable configurations}}
\]

---

# 七、这是 Triton 真正值得研究的点

例如一个 tensor：

\[
X
\]

先被 load 使用。

Coalesce 喜欢：

\[
L_{\text{mem}}
\]

后面 dot 喜欢：

\[
L_{\text{mma}}
\]

后面 reduction 又喜欢：

\[
L_{\text{reduce}}
\]

于是系统面对的是：

\[
L_{\text{mem}}
\rightarrow
L_{\text{mma}}
\rightarrow
L_{\text{reduce}}
\]

而不是：

\[
\min_L C(L)
\]

Triton 已经有 `RemoveLayoutConversions` 去缓和问题，本身就证明：

> layout conversion 并不是偶然 artifact，而是 compiler 中真实存在的优化冲突。fileciteturn40file0L1-L2

这和“Trition layout 优化不够好”的说法完全不一样。

真正现象是：

\[
\boxed{
不同局部优化目标可能要求彼此冲突的表示
}
\]

---

# 八、Gluon 又改变了 Triton 的设计边界

这里必须把 classic Triton 与 Gluon 分开。

Gluon 暴露 `BlockedLayout` 等显式 layout abstraction。

而且官方教程非常有价值：它直接展示 layout 会影响：

\[
\text{global vectorization}
\]

\[
\text{sector accesses}
\]

\[
\text{layout conversion}
\]

\[
\text{reduction communication}
\]

\[
\text{shared-memory bank conflict}
\]

甚至教程明确把 layout-conversion cost 算进整体 throughput。citeturn586462search6

因此 Triton/Gluon 其实形成一个很有趣的对照：

```text
Classic Triton:
programmer hides layout
compiler chooses

Gluon:
programmer exposes layout
compiler lowers
```

所以这里有一个非常重要的研究维度：

\[
\boxed{
谁应该拥有 layout 决策权？
}
\]

程序员？编译器？搜索器？constraint solver？

这比“layout 表示是什么”重要得多。

---

# 九、TVM MetaSchedule：真正不同的是“性能反馈机制”

TVM MetaSchedule 和上述几者最重要的区别不是 TensorIR。

而是：

\[
\boxed{\text{它愿意真正运行 candidate}}
\]

MetaSchedule 搜索：

\[
\text{loop tiling}
\]

\[
\text{thread binding}
\]

\[
\text{vectorization}
\]

\[
\text{cache/reuse}
\]

\[
\text{tensorization}
\]

并在真实硬件上测 latency，再选择 schedule。官方当前文档仍明确将其定义为 search-based auto-tuning。citeturn586462search3

同时还有：

`RewriteLayout`：

> rewrite input tensor layout。

以及 tensor-core specific schedule rules。citeturn586462search1turn586462search8

因此 TVM 的核心哲学更接近：

\[
\boxed{
Generate candidates
\rightarrow
Measure
\rightarrow
Search
}
\]

而 Hexcute 是：

\[
\boxed{
Infer legal candidates
\rightarrow
Analytical ranking
}
\]

Triton 则更多是：

\[
\boxed{
Pattern / heuristic rewriting
}
\]

CUTLASS：

\[
\boxed{
Expert construction / selection rules
}
\]

这是四种完全不同的 decision mechanism。

---

# 十、TVM MetaSchedule 的强点和弱点因此都很明确

强点：

\[
C_{\text{candidate}}
=
T_{\text{measured}}
\]

真实 hardware 能自动捕获：

bank conflicts、cache behavior、pipeline overlap、compiler effects……

不需要分析模型把所有东西都准确建模。

但是它的弱点也来自相同地方：

\[
\boxed{
只能搜索 Search Space 中已经被表达的候选
}
\]

如果 schedule rule 没有产生一种新的 thread/data layout family：

再准确的 measurement 都没用。

所以：

\[
SearchQuality
\le
CandidateSpaceQuality
\]

这和 Hexcute 恰好构成很有意思的互补关系。

---

# 十一、TVM TIRx 又是第三种设计

2026 TVM 的 TIRx 对你的方向很重要。

TIRx 将 layout 定义成 storage-first contract：

\[
\text{logical tensor}
\rightarrow
\text{physical hardware resources}
\]

但和 CuTe 不同：

CuTe 用 layout 同时参与 work partitioning。

TIRx 则：

> layout 描述 storage；TilePrimitive dispatch 根据 operand layouts + execution scope + target，选择具体实现并生成 thread partitioning / loop / instruction。citeturn275868search0turn275868search3

因此：

```text
CuTe
layout
  ↓
programmer composes partitioning

TIRx
storage layout + execution scope
  ↓
dispatch
  ↓
partitioning + instruction
```

这又是一种完全不同的边界划分。

---

# 十二、TIRx 还有一个很重要的特点：不试图自动接管所有 orchestration

当前 TIRx 设计里：

```text
pipeline state
barriers
role selection
warp specialization
low-level synchronization
```

仍然显式存在于程序中。

Tile primitive 只负责 recurring hardware primitive 的 dispatch。citeturn275868search0turn275868search5

这和 Hexcute 很相似：

> **自动化某一类映射问题，但不自动 dataflow/pipeline。**

这条共同点后面非常值得变成科研问题。

---

# 十三、Hexcute：不同于 TVM 的搜索，也不同于 Triton heuristic

Hexcute 的核心贡献是把 layout synthesis 形式化成 constraint problem。

CGO 2026 官方描述很准确：

> automates layout synthesis while providing explicit control over dataflow and pipelining；并用 type-inference-based algorithm 解 constraint。citeturn586462search0

Hexcute/Hidet 实现中可以看到四层：

\[
LogicalShape
\]

\[
LogicalLayout
\]

\[
TaskMapping
\]

\[
MemoryLayout
\]

对于 Copy，建立：

\[
f\circ p^{-1}
=
g\circ q^{-1}
\]

对于 MMA，约束 M/N/K mapping compatibility。

然后从已知 layout 传播未知 layout。fileciteturn47file0L1-L7

因此它回答的是：

> **给定 dataflow 和候选 hardware instructions，哪些 layout/task mapping 是合法的？**

这比 Triton 的 heuristic 更系统。

---

# 十四、Hexcute 与 TVM 的真正差别

这两者特别值得直接比较：

| | Hexcute | TVM MetaSchedule |
|---|---|---|
| 核心机制 | Constraint inference | Search |
| 主要解决 | candidate legality / synthesis | performance selection |
| Candidate 来源 | 从约束推导 | schedule rules 生成 |
| 性能评价 | analytic model | measurement / cost model |
| 非法候选处理 | 很早就剪掉 | schedule construction / postproc |
| 探索代价 | 相对低 | 相对高 |
| 模型错误风险 | 高于 measurement | 很低 |
| search-space completeness | constraint system 决定 | schedule rule 决定 |

这反而给出一个很清晰的研究张力：

\[
\boxed{
Hexcute 擅长“找到合法空间”，
TVM 擅长“在给定空间中识别快的”
}
\]

这比说：

> Hexcute layout 自动化比 TVM 强

准确得多。

---

# 十五、Hexcute 目前性能选择还有一个非常明确的结构性边界

其 cost model 当前大体基于：

\[
N_{\text{instructions}}\times CPI
\]

并估计 copy/MMA overlap。

源码明确注明：

bank conflict 影响尚未进入该 cost model；

pipeline 假设充分 overlap；

没有考虑 `cp_async_wait_group()` 和 `mbarrier`；

tensor manipulation/address calculation 被忽略。fileciteturn48file0L1-L7

与此同时，后续又有单独的 `ResolveBankConflict`：

它分析 shared-memory copy 的 thread access，然后枚举 swizzle，最小化 bank conflicts。fileciteturn49file0L1-L7

因此其内部出现：

```text
layout/task synthesis
       │
       ▼
instruction/cost evaluation
       │
       ▼
bank-conflict repair
```

这不意味着 Hexcute“做错了”。

但它说明：

\[
\boxed{
Legality synthesis,\ Performance ranking,\ Resource-conflict optimization
}
\]

目前仍然不是完全统一的问题。

---

# 十六、现在看 vLLM：它解决的不是 transient kernel layout，而是 persistent representation

这也是之前最容易比较错误的地方。

当前 vLLM 已经将 KV cache 定义成明确的 physical layout descriptor：

逻辑 shape 固定：

\[
[L,B,H,N,C]
\]

但物理 stride order 可为：

\[
LBHNC,\ LBNHC,\ LHBNC,\ BLHNC,\ BLNHC,\ BHLNC
\]

fileciteturn43file0L1-L7

也就是说这里的 layout 是：

> **跨 kernel、跨 token、长期存在的 persistent state representation。**

它和某个 kernel 内 shared-memory swizzle 根本不是一个生命期。

---

# 十七、vLLM 当前的选择机制也很有意思：compatibility negotiation

各 AttentionBackend 声明：

\[
SupportedLayouts_B
\]

并按 preference 排列。

引擎取所有 backend 的交集：

\[
C=
\bigcap_B SupportedLayouts_B
\]

然后根据：

backend preferences、

mixed KV specs、

explicit environment override、

connector requirements，

选一个**整个模型共享的 KV layout**。fileciteturn44file0L1-L7

官方文档也明确说：

> 一个 layout 为整个 model resolve 一次；backend 报支持集合；connector preference 也参与选择。citeturn275868search1turn275868search2

注意这不是：

\[
\arg\min_L T(L)
\]

更准确是：

\[
\boxed{
Find compatible L,
then use preference ordering
}
\]

---

# 十八、vLLM 已经出现非常漂亮的目标冲突

NIXL connector 当前甚至默认喜欢 `LBHNC/HND`，原因是 transfer performance。

而 token-major `LBNHC/NHD` 也支持，但传输能力不同。citeturn275868search7

这意味着同一个 persistent KV layout 有多个 consumer：

```text
Attention kernel
KV insertion
Prefix cache
PD transfer
Offload
Connector / RDMA
```

所以真正目标变成：

\[
T_{\text{total}}(L)
=
T_{\text{attention}}(L)
+
T_{\text{insert}}(L)
+
T_{\text{transfer}}(L)
+
T_{\text{management}}(L)
\]

这比“哪个 KV layout attention 更快”科学问题强很多。

---

# 十九、SGLang：它和 vLLM 又不一样

SGLang更加明显地做：

\[
\boxed{\text{algorithm/backend specialization}}
\]

官方文档直接说：

> 不同 attention backend 在不同模型、硬件和场景中各有优劣；如果用户没指定，会根据 hardware 和 model architecture 尽力选择 performant backend。fileciteturn46file0L1-L7

而且支持：

\[
Backend_{\text{prefill}}
\neq
Backend_{\text{decode}}
\]

当前源码会在两者不同时构造 Hybrid Attention Backend。fileciteturn35file0L1-L7

因此 SGLang 比 vLLM 更明显体现：

\[
\boxed{\text{phase-aware algorithm specialization}}
\]

---

# 二十、但 SGLang 的 backend selection 和 compiler search 也是完全不同的东西

例如当前支持矩阵会考虑：

```text
page size
FP8 KV
FP4 KV
spec decoding
sliding window
multimodal
MHA / MLA
hardware backend
```

fileciteturn46file0L1-L7

本质更像：

\[
Backend=
Rule(Model,Hardware,FeatureSet,Phase)
\]

而不是：

\[
Backend=
\arg\min_C MeasuredPerformance(C)
\]

所以 SGLang 的优势是：

\[
\text{workload awareness}
\]

而不是：

\[
\text{large configuration search}
\]

---

# 二十一、现在可以总结真正的“相同点”

这部分其实比不同点更重要。

六类框架都有以下共性。

第一，**都没有完全摆脱专家知识**。区别只是专家知识放到了 template、compiler pass、schedule rule、constraint、backend capability 还是 runtime policy。

第二，**都在做 constrained optimization，而不是任意 layout permutation**。Hardware instruction、alignment、memory scope、thread participation、backend compatibility 都大量裁剪空间。

第三，**都在把“合法性”和“性能”分成某种形式的两阶段问题**。CUTLASS先有合法 template family；Triton先 rewrite 到合法 encoding；TVM schedule rule 生成候选后测；Hexcute先 constraint inference 后 cost；vLLM先做 capability intersection 再 preference。

第四，**都存在优化边界**。没有一个系统同时拥有 runtime workload、persistent representation、subgraph dataflow、thread-value mapping、instruction、pipeline 的统一决策权。

我认为第四条才是我们后面应该重点挖的。

---

# 二十二、而最重要的“不同点”不是 layout API，而是下面这五条

| 维度 | CUTLASS | Triton | TVM | Hexcute | vLLM/SGLang |
|---|---|---|---|---|---|
| 谁提出 candidate | 专家 | compiler heuristics | search rules | constraint inference | backend/runtime registry |
| 谁判断 legality | template/types | compiler verifier/rules | schedule/postproc | constraint solver | capability intersection |
| 谁判断 performance | 专家/rules/profiler | heuristic + autotune config | **hardware measurement** | **analytic model** | preference / policy |
| 优化 scope | kernel | kernel | operator/task | tile program | runtime/persistent state |
| 动态 workload 信息 | 很少 | shape | task/shape | shape/program | **很多** |

因此这六个框架最核心的设计差异实际上可以压缩为：

\[
\boxed{
Candidate Generation
\times
Legality
\times
Performance Evaluation
\times
Optimization Scope
\times
Runtime Information
}
\]

我认为我们后面所有科研问题都应该从这里产生。

---

# 二十三、基于这个比较，我会把之前的科研问题大幅修改

之前的“cross-op layout”“KV layout”等方向本身不是错，但描述太接近 implementation opportunity。

现在我认为值得继续考虑的是下面五个真正更抽象的问题。

---

## 科学问题 A：GPU 优化决策在多层 optimization boundary 之间是否具有可分解性？

这是我现在最推荐的主问题。

现有系统实际上假定：

\[
Runtime
\]

\[
PersistentLayout
\]

\[
KernelLayout
\]

\[
Instruction
\]

\[
Pipeline
\]

可以相当程度独立优化。

因为不同系统分别负责它们。

但真正的问题是：

\[
\boxed{
\arg\min_{R,L,K,I,P} T(R,L,K,I,P)
\stackrel{?}{=}
\left(
\arg\min R,
\arg\min L,
\arg\min K,
\arg\min I,
\arg\min P
\right)
}
\]

显然数学上一般不成立，但关键科研问题不是证明“不成立”。

而是：

> **在现代 GPU workload 的什么条件下，这种分层优化仍然近似成立；什么条件下会发生严重 coupling，从而产生系统性性能损失？**

这就是科学问题。

---

### 为什么这不是工程问题

因为它研究的是：

\[
\text{optimization decomposition validity}
\]

而不是给某个框架修一个 pass。

得到的结论理论上应该适用于 Triton、TVM、Hexcute、CUTLASS，也适用于后续 compiler。

---

### 最适合观察这个现象的子图

首选不是 GEMM。

最适合的是：

\[
\boxed{Decode Attention}
\]

因为这里天然跨：

```text
runtime page allocation
persistent KV layout
thread mapping
memory instruction
reduction
```

第二类是：

\[
QK^T
\rightarrow Softmax
\rightarrow PV
\]

第三类：

\[
Gate/Up GEMM
\rightarrow SiLU\times
\rightarrow Down GEMM
\]

Diffusion 可以用：

\[
Conv/Linear
\rightarrow Norm/Activation
\rightarrow Conv/Linear
\]

验证 generality。

---

# 二十四、科学问题 B：高质量 GPU optimization 是否应该分成“结构合法性生成”和“性能选择”两个不同问题？

这来自 Hexcute 和 TVM 最明显的互补。

Hexcute：

\[
\text{Constraints}
\rightarrow
\text{legal candidate set}
\]

TVM：

\[
\text{Candidate set}
\rightarrow
\text{measurement search}
\]

Triton：

\[
\text{heuristic directly chooses}
\]

CUTLASS：

\[
\text{expert chooses}
\]

于是非常自然的问题是：

\[
\boxed{
怎样的 candidate-generation mechanism
能够在不爆炸搜索空间的情况下
保留真正的 performance optimum？
}
\]

这个问题非常科学。

因为它研究：

\[
\text{Search Space Completeness}
\quad vs \quad
\text{Search Cost}
\]

---

### 假设

我目前认为值得检验的假设是：

\[
\boxed{
Constraint-guided candidate generation
+
empirical/performance-guided ranking
}
\]

可能比：

pure heuristics、

pure brute-force search、

pure analytic ranking

具有更好的：

\[
\text{quality} / \text{search cost}
\]

Pareto frontier。

注意，这不是说“把 Hexcute 和 TVM 拼起来”。

科学问题是：

> **GPU mapping space 中是否存在足够强的结构约束，使我们能在几乎不丢失优质方案的情况下极大缩小 empirical search space？**

这就很像真正 compiler research。

---

# 二十五、科学问题 C：不同局部 consumer 对同一 tensor 的 layout preference 冲突能否被一个统一的表示与目标函数描述？

这个比之前简单说“跨算子 layout”更准确。

典型 tensor P：

\[
QK^T\rightarrow P\rightarrow Softmax\rightarrow P'\rightarrow PV
\]

Softmax 希望减少跨线程 reduction：

\[
L_{\text{softmax}}
\]

PV 希望适配 MMA：

\[
L_{\text{mma}}
\]

memory write/read 又希望：

\[
L_{\text{mem}}
\]

Triton 当前就通过不同 pass 与 `ConvertLayout` 处理这些冲突；Gluon 文档也直接展示 conversion 与 communication 成本。fileciteturn40file0L1-L2 citeturn586462search6

因此真正的问题不是：

> 跨算子布局能不能优化？

而是：

\[
\boxed{
当一个 tensor 有多个异构 consumer 时，
是否存在可以统一表示各 consumer layout demand
及转换代价的 compositional optimization model？
}
\]

这比“减少 transpose”更一般。

---

### 这个问题的可推广性很好

它同时覆盖：

Attention：

\[
QK\rightarrow Softmax\rightarrow PV
\]

FFN：

\[
GEMM\rightarrow elementwise\rightarrow GEMM
\]

MoE：

\[
routing\rightarrow grouped GEMM
\]

Diffusion：

\[
Norm\rightarrow QKV\rightarrow Attention
\]

以及 Conv residual blocks。

---

# 二十六、科学问题 D：layout / instruction / pipeline 的最优性到底有多强的耦合？

这个问题来自三类系统的共同结构。

CUTLASS：

专家共同设计：

\[
L+I+P
\]

Hexcute：

自动化：

\[
L+I
\]

但 pipeline/dataflow 显式控制。citeturn586462search0

TIRx：

layout + primitive dispatch 自动一部分，但 pipeline/barrier/orchestration 显式。citeturn275868search0

Triton：

不同 pass 分别做 MMA、layout、pipeline、warp specialization。fileciteturn40file0L1-L2

所以多个系统独立得出了一个类似边界：

> 自动 layout/instruction，pipeline 单独处理。

这非常值得追问：

\[
\boxed{
这个边界是因为“理论上可以近似分解”，
还是只是因为 compiler architecture 比较容易这样实现？
}
\]

这就是一个好的科学问题。

---

### Hypothesis 应该变成

不是：

> joint search 更快。

这个太显然。

而应该是：

> **L/I/P coupling strength 与 operational intensity、async-copy depth、shared-memory footprint 和 synchronization topology 存在可预测关系。**

如果这个假设成立，就可以进一步决定：

什么时候：

\[
SeparateOptimization
\]

足够。

什么时候：

\[
JointOptimization
\]

才值得花搜索成本。

这就比无脑联合搜索有研究价值。

---

# 二十七、科学问题 E：Persistent layout 是否应该是一个固定配置，还是 workload-dependent policy？

这个方向来自 SGLang 与 vLLM，而不是来自 kernel compiler。

vLLM 当前做的是：

\[
\boxed{\text{one resolved KV layout for the model}}
\]

fileciteturn44file0L1-L7

但实际 workload 包含：

\[
Prefill
\]

\[
Decode
\]

\[
Speculative
\]

\[
KV transfer
\]

\[
Prefix reuse
\]

\[
Offload
\]

甚至 SGLang 已经认为 prefill/decode backend 可以不同。fileciteturn35file0L1-L7

NIXL 又明确表现出 transfer 对 layout 的独立 preference。citeturn275868search7

所以真正的问题是：

\[
\boxed{
Persistent representation 的最佳选择
是否是 workload distribution 的函数，
而不是 model configuration 的函数？
}
\]

形式上：

\[
L^*
=
f(Model,Hardware)
\]

还是应该：

\[
L^*
=
f(Model,Hardware,\mathcal W)
\]

其中：

\[
\mathcal W
=
P(B,S_q,S_{kv},phase,transfer,reuse,\ldots)
\]

---

# 二十八、这五个问题中，我现在的优先级已经发生变化

我不会再把“cost model 少了 bank conflict”列成主要科研问题。

那首先是一个已知 implementation limitation。

我也不会把：

> Triton convert layout 太多

本身当科研问题。

更不会把：

> KV layout 可以换一下顺序

当科研问题。

现在我认为最有价值的是：

| 优先级 | 科学问题 | 原因 |
|---|---|---|
| **S** | A. Optimization boundary 是否可分解 | 六类系统共同结构性问题 |
| **S** | B. Constraint synthesis × performance search 的 Pareto frontier | Hexcute/TVM/Triton/CUTLASS 可形成非常干净的方法学比较 |
| **A+** | D. Layout–Instruction–Pipeline coupling | 面向 Hopper/Blackwell 越来越重要 |
| **A+** | E. Persistent layout 的 workload dependence | 极有 LLM serving 特色，vLLM/SGLang 提供现实 motivation |
| **A** | C. Multi-consumer tensor layout conflict | 易验证、适合 LLM + Diffusion，但单独做可能创新度稍弱 |

---

# 二十九、我尤其推荐把 A + B 作为主线

如果最终想做一个“layout 优化研究”，我目前最看好的故事不是：

> 我们提出了一个更好的 layout search。

而是：

> **现有 GPU optimization frameworks 将一个本质耦合的空间沿不同 abstraction boundary 分解：runtime systems optimize persistent representation，kernel compilers optimize transient mappings，layout synthesizers optimize legal mappings，autotuners optimize schedules。我们研究这种 decomposition 何时产生 substantial optimization regret，并提出一种 constraint-guided cross-boundary search。**

注意最后“提出什么方法”我们现在先不定。

因为你说得对：

> 先把问题证据做扎实，再决定 solution。

目前只保留研究问题：

\[
\boxed{
How much performance is lost because current GPU optimization systems
partition the decision space differently?
}
\]

以及：

\[
\boxed{
Can structural constraints identify the small subset of cross-boundary
configurations worth evaluating?
}
\]

我认为这两个问题比前面我给出的“跨算子 layout”“改进 cost model”高一个层级。

---

# 三十、下一步我建议也不要立刻写代码

下一阶段应该先做一张非常严格的 **“decision ownership matrix”**。

不是之前那种：

> 谁选什么 layout。

而是对每个框架逐项回答：

| Decision | 谁决定 | 什么时候决定 | 候选从哪里来 | legality 谁检查 | performance 谁评价 | 是否看到邻接算子 | 是否看到 runtime | 可否 rollback/search |
|---|---|---|---|---|---|---|---|---|
| algorithm | | | | | | | | |
| fusion/dataflow | | | | | | | | |
| persistent layout | | | | | | | | |
| tile | | | | | | | | |
| thread mapping | | | | | | | | |
| register layout | | | | | | | | |
| shared layout | | | | | | | | |
| swizzle | | | | | | | | |
| copy instruction | | | | | | | | |
| MMA instruction | | | | | | | | |
| pipeline | | | | | | | | |
| warp specialization | | | | | | | | |
| cache policy | | | | | | | | |

对：

**SGLang / vLLM / CUTLASS / Triton classic / Triton Gluon / TVM MetaSchedule / TVM TIRx / Hexcute**

全部填完。

完成这张表以后，我们才能判断 A–E 中哪些问题是真正由源码支持的，哪些只是我们的猜测。

这一次我建议先不进入任何验证代码。你先看这套“比较维度 + A–E 科学问题”的抽象是否符合你的研究目标；如果你认可，下一轮我会继续把上面这张 **decision ownership matrix 填到源码级**，包括具体文件、class/pass/function 和每个 decision 的依据，然后再进一步筛掉不够强的科研问题。

## 用户

请继续下一轮

## 模型回复

我继续按“decision ownership matrix”往下做，这一轮先把每个系统的**决策边界、候选生成、合法性检查、性能评价、跨算子/运行时可见性**填到源码/官方文档级别。最后我会基于这张表重新筛科研问题，只保留能被这些结构性差异直接支持的。

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 3 个网站

## 模型工具调用参数

{"paths":["GitHub"],"query":"fetch_file"}

## 模型思考摘要

已搜索 3 个网站

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取并分析多级分块调度规则代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取并分析TVM布局重写实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 TIRx 源码中的调度注册逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查找 CUTLASS 自动构建器源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 vLLM 注意力后端支持布局

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查找 supported_kv_cache_layouts 定义

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 SGLang 注意力后端与内存池布局

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

解析 SGLang 注意力后端自动选择机制

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索平台默认注意力后端解析逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理注意力后端默认配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取模型覆盖基础解析规则

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查找 get_default_attn_backend 实现

## 模型工具调用参数

{"uri":"/response/turn62","query":"def get_default_attn_backend"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取模型覆盖参数文件后续内容

## 模型思考摘要

检索并分析注意力后端、布局及调度实现源码

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 triton-lang.org

## 模型思考摘要

已搜索 1 个网站

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

  Comparing Kernel Optimization Ownership Across Systems

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取合并、移除转换与线程局部性相关代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析主流推理框架责任边界

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 vLLM 注意力后端选择器文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 vLLM 注意力选择器代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 CUDA 后端优先级与性能候选

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

调查 vLLM 注意力后端候选选择机制

## 模型思考摘要

分析推理框架内核优化归属与vLLM注意力后端选择机制

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 docs.nvidia.com

## 模型思考摘要

已搜索 1 个网站

## 模型回复

这一轮有一个重要方法学修正：我会把“框架本身决定”与“框架把决定委托给 backend/kernel”分开。比如 SGLang/vLLM 的 thread mapping、SMEM swizzle 不是“不存在”，而是**runtime 不拥有这个决策权，选中的 FlashInfer/TRT-LLM/Triton/CUDA backend 才拥有**；如果不区分 ownership，很容易把框架能力比较错。

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 SGLang KV 池物理布局源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 SHUFFLE 5D 池布局实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取预洗牌相关内存池代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 MHATokenToKVPool 类及行号

## 模型思考摘要

检索并定位SGLang KV池及预洗牌布局源码

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 tvm.apache.org

## 模型思考摘要

改进科研问题
改进了科研问题

## 模型思考摘要

深化科研问题
深化了科研问题

## 模型思考摘要

已搜索 1 个网站

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索Hexcute仓库中的Hidet子模块配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 HexCute Hidet 仓库中的 CUDA 成本模型

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 CUDA 成本模型文件内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 HexCUTE 推理与指令选择实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检查 Hexcute 分支中的银行冲突解析

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 Hexcute 指令选择源码

## 模型思考摘要

检索并获取HexCute Hidet配置及CUDA实现源码

## 模型思考用时

思考了 3m 34s

## 模型回复

继续。这一轮我把比较方式彻底换成“**decision ownership + candidate generation + legality + performance evaluation + optimization scope**”，而不是“不同框架的 layout 方法”。

先给结论：从当前 SGLang、vLLM、CUTLASS/CuTe、Triton classic、Gluon、TVM MetaSchedule/Relax、TIRx、Hexcute 的源码看，真正拉开这些系统差距的不是某一种 layout 表示，而是：

\[
\boxed{
\text{谁拥有某个决策}
\times
\text{谁产生候选}
\times
\text{谁判断合法}
\times
\text{谁判断快不快}
\times
\text{能看到多大的上下文}
}
\]

而且这一轮比较以后，我认为科研问题应该进一步重写。之前“跨算子 layout”“cost model 不准”仍然太像 implementation issue。

---

# 1. 先定义比较符号

下面用：

- **E**：Expert/programmer 显式决定
- **H**：Heuristic/rule 决定
- **M**：Measurement-based search/autotune
- **C**：Constraint/inference 自动推导
- **A**：Compiler 自动 lowering/transform
- **D**：Delegated，框架本身不拥有该决策，由选中的 backend/kernel/library 决定
- **—**：基本不在该组件的优化边界内

这里的关键是 **D 和 — 不一样**。

例如 SGLang 的 shared-memory swizzle 不是“不存在”，而是：

\[
\text{SGLang runtime}
\rightarrow
\text{FlashInfer/TRT-LLM/Triton backend}
\rightarrow
\text{SMEM layout}
\]

即 SGLang 自己没有 ownership。

---

# 2. 第一张关键表：到底“谁决定什么”

| Decision | SGLang | vLLM | CUTLASS/CuTe | Triton classic | Gluon | TVM Relax + MetaSchedule | TIRx | Hexcute |
|---|---|---|---|---|---|---|---|---|
| Serving algorithm/backend | **H/E** | **H/E** | — | — | — | — | — | — |
| Prefill/decode specialization | **H/E** | backend-dependent | — | kernel-level | kernel-level | task-level | kernel-level | kernel-level |
| Graph fusion | custom/D | custom/D | — | — | — | **A/H** | Relax负责 | — |
| Kernel dataflow | D | D | **E** | **E** | **E** | E/search-space | **E** | **E** |
| Persistent KV/state layout | H/D | **H + compatibility resolution** | — | E/external | E/external | 可 rewrite weights | E/external | E/external |
| Tile shape | D | D | **E/H** | **E/M** | **E/M** | **M** | E | E |
| Thread/warp mapping | D | D | **E** | **A/H** | **E** | **M/H** | **A dispatch** | **C** |
| Register/value layout | D | D | **E** | **A** | **E** | 间接 schedule | storage/layout + dispatch | **C** |
| SMEM layout | D | D | **E** | **A** | **E** | cache/schedule | **E** | **C + A** |
| SMEM swizzle | D | D | **E** | **A** | **E** | schedule/intrin | E/primitive | **A conflict pass** |
| Global coalescing | D | D | **E** | **A/H** | **E** | **M/H** | primitive dispatch | **C/A** |
| Copy instruction | D | D | **E/H** | **A** | E/A | tensorization/lowering | **A dispatch** | **C/A** |
| MMA instruction | D | D | **E/H** | **A** | E/A | tensor intrin/search | **A dispatch** | **C/A** |
| Pipeline depth | D | D | **E/H** | **E/A/M** | **E/M** | **M/H** | **E** | **E** |
| Warp specialization | D | D | E/H | **A/E** | **E** | schedule dependent | **E** | **E** |
| Performance ranking | rules | rules/preferences | expert/static rules | **H + optional M** | **optional M** | **learned model + M** | priority/predicate | **analytic model** |
| Live serving context | **高** | **高** | 低 | 低 | 低 | 低 | 低 | 低 |

这张表已经比“谁的 layout abstraction 强”更能解释科研机会。

---

# 3. SGLang：它拥有的是 runtime policy，而不是 kernel mapping

当前 SGLang 的默认 attention backend 并不是笼统地“FlashInfer 优先”。

源码里的 `get_default_attn_backend()` 会显式考虑：

\[
\text{architecture}
,\text{GPU generation},
\text{speculative mode},
\text{page size},
\text{MLA/MHA},
\text{asymmetric KV},
\text{attention sinks}
\]

例如当前规则包括：

- Hopper MHA 条件满足时偏向 FA3；
- SM100 上可以选 TRTLLM MHA；
- asymmetric K/V 时会改走 FA4；
- HIP 走 AITER；
- FlashInfer 不满足约束时回退 Triton；
- MLA 又是另一套规则。fileciteturn64file0L1-L7

官方 attention backend 文档同样明确说不同 backend 在 page size、FP8/FP4 KV、spec decoding、sliding window、多模态等支持不同。fileciteturn46file0L1-L7

更重要的是，SGLang 将：

\[
B_\mathrm{prefill}
\]

和

\[
B_\mathrm{decode}
\]

作为可以独立 resolve 的对象；不同的时候会构造 hybrid attention backend。fileciteturn35file0L1-L7

所以 SGLang 的优化形式更像：

\[
B^*
=
H(
Model,
Hardware,
ServingFeatures,
Phase
)
\]

而不是：

\[
(L,T,I,P)^*
=
\arg\min T
\]

这也是为什么不能拿 SGLang 和 Hexcute 直接比较“layout 自动化程度”。

---

# 4. vLLM：当前已经明显变成“persistent representation manager”

vLLM 这一块比很多旧资料复杂。

当前 `KVCacheLayout` 的逻辑空间固定成：

\[
[L,B,H,N,C]
\]

但物理顺序可以是：

\[
LBHNC,\ LBNHC,\ LHBNC,\ BLHNC,\ BLNHC,\ BHLNC
\]

每种实际上是 logical axes 到 physical stride order 的 permutation。fileciteturn43file0L1-L7

这和 kernel 内的：

\[
register\rightarrow lane\rightarrow warp
\]

完全不是同一种 layout。

这是一个**跨 kernel 长期存在的 storage contract**。

---

## vLLM 的选择机制也非常特殊

每个 backend 可以声明：

```text
supported_kv_cache_layouts()
```

不同 backend 的要求确实不一样。

例如源码中能找到：

- CPU backend 只接受 `LBHNC`
- HPC attention 只接受 `LBNHC`
- 一些 backend 接受 `LBHNC, BLHNC`
- 某些 QSA cache 接受 `BLNHC, BLHNC`

fileciteturn57file0L1-L13 fileciteturn57file2L27-L38 fileciteturn57file6L79-L90 fileciteturn57file8L105-L116

然后 vLLM 做的不是 performance search，而是：

\[
C=
\bigcap_i SupportedLayouts_i
\]

再根据 backend preference、connector preference、mixed KV spec、用户 override 等选一个整个模型共享的 physical layout。源码甚至明确写：

> Resolve one KV cache layout for the whole model.

fileciteturn44file0L1-L7

所以 vLLM 的核心方法是：

\[
\boxed{
Compatibility\ negotiation
+
Preference\ resolution
}
\]

而不是 compiler search。

---

# 5. vLLM backend selection 又是另一层 decision

当前 `AttentionSelectorConfig` 会包含：

\[
head\_size,\ dtype,\ kv\_dtype,\ block\_size,
MLA,\ sparse,\ sinks,
sliding\ window,
KV\ connector,
PCP,\ DCP,\ldots
\]

然后 platform 决定 backend。fileciteturn70file0L1-L7

同时 backend 还能：

```python
customize_spec(...)
```

因为某些 kernel 对 KV packing 有自己的要求。fileciteturn56file0L1-L7

因此 vLLM 的 hierarchy 实际是：

```text
Model / serving configuration
          │
          ▼
Attention backend
          │
          ├── supported block size
          ├── supported dtype
          ├── supported KV layout
          └── customize KV spec
          │
          ▼
Global model KV-layout resolution
          │
          ▼
Backend kernel
```

kernel 内的 thread mapping、SMEM layout、MMA 等仍然属于 backend。

---

# 6. CUTLASS/CuTe：专家把整个低层配置一次性构造出来

CuTe 的 `Layout` 本质上是：

\[
coordinate\ space
\rightarrow
index\ space
\]

而且支持 composition、product、divide、tiling、partition。citeturn415906search0turn415906search3

它甚至直接把 thread/value partition 表成：

\[
(thread,value)\rightarrow(M,N)
\]

citeturn415906search4

这意味着 CUTLASS/CuTe 能表达：

\[
L_{gmem},L_{smem},L_{thread/value},I_{copy},I_{mma},P
\]

之间非常细的耦合。

当前 GEMM Collective API 直接包含：

```text
TileShape
ClusterShape
TiledMma
GmemTiledCopy
SmemLayoutAtom
SmemCopyAtom
StageCount
KernelSchedule
```

fileciteturn51file10L163-L174

---

## 但 `Auto` 并不是 general search

CUTLASS 文档明确说：

`StageCountAuto` 会根据一个 stage 的 shared-memory footprint 自动决定 stage count；

`KernelScheduleAuto` 会从 builder 支持的 schedule 中选择。citeturn419041search0

源码也能看到它最终进入 architecture-specific `CollectiveBuilder` specialization，而不是一个任意配置搜索器。fileciteturn55file0L1-L16

所以 CUTLASS 是：

\[
\boxed{
Maximum\ expressiveness
+
Expert/rule-based\ construction
}
\]

它最适合在我们后续研究中担当：

\[
\textbf{expert oracle}
\]

而不是 automatic compiler baseline。

---

# 7. Triton classic：优化 ownership 被分散在很多 pass 中

这是非常重要的区别。

当前 Triton GPU pipeline 中至少有：

`AccelerateMatmul`

> 把 dot input/output layout 改成 hardware accelerator-compatible layout。

`Coalesce`

> load/store 使用 cache-friendly layout，并在前后插 `ConvertLayout`。

`RemoveLayoutConversions`

> 尝试减少 layout conversion，在 memory-friendly blocked layout 与 tensor-op-friendly MMA layout 间做取舍。

`OptimizeThreadLocality`

> reduction/gather 中降低 cross-thread communication。

`Pipeline`

> 基于 stage number 做 software pipeline / async load / multi-buffering。

`AutomaticWarpSpecialization`

> 分析 loop 并尝试自动创建 warp-specialized partition。

fileciteturn66file0L1-L7 fileciteturn67file0L1-L7

所以 classic Triton 不是：

\[
\text{choose a layout}
\]

而是：

\[
\boxed{
\text{a sequence of interacting local rewrites}
}
\]

---

# 8. Triton 其实同时拥有 heuristic 和 measurement 两种机制

这一点上一轮也说得不够完整。

Compiler 内部大量 decision 属于 heuristic/rule。

但是用户提供：

```python
@triton.autotune(configs=[...], key=[...])
```

后，Triton 会真正 benchmark 不同 config。

一个 Config 可以包含：

\[
BLOCK\_M,BLOCK\_N,BLOCK\_K,
num\_warps,
num\_stages,
num\_ctas,
maxnreg
\]

等。citeturn584227search0turn584227search4

所以 Triton 更准确是：

\[
\boxed{
Compiler\ heuristics
+
User-defined\ empirical\ configuration\ search
}
\]

但这里最关键的限制是：

> autotuner 只能测试**用户给出的 meta-parameter configs**；很多 compiler-internal distributed layout choices并没有天然作为 autotuning variable 暴露出来。

这会成为后面的科研问题。

---

# 9. Gluon 和 classic Triton 是两种不同的 ownership philosophy

Gluon 的 tensor 必须有 layout。

例如 `BlockedLayout` 明确描述：

\[
size\_per\_thread
\]

\[
threads\_per\_warp
\]

\[
warps\_per\_cta
\]

\[
order
\]

也就是 tensor element 如何落到 register/lane/warp/CTA。citeturn419041search5turn419041search10

所以：

```text
Classic Triton
    programmer gives tensor computation
                ↓
      compiler chooses layouts

Gluon
    programmer gives computation
        + explicit layouts
                ↓
          compiler lowers
```

Gluon 甚至把 warp specialization、barrier、buffer management 显式暴露给 programmer；官方示例直接手写 worker partitions 和 `mbarrier`。citeturn419041search3turn419041search4

但 Gluon 仍然可以使用 Triton autotune。citeturn584227search7

所以它代表：

\[
\boxed{
Explicit\ low-level\ control
+
Empirical\ meta-parameter\ tuning
}
\]

---

# 10. TVM MetaSchedule：核心区别不是 IR，而是搜索闭环

MetaSchedule 当前明确由几个组件构成：

\[
SpaceGenerator
\rightarrow
SearchStrategy
\rightarrow
CostModel
\rightarrow
Builder/Runner
\rightarrow
Database
\]

默认 SpaceGenerator 用 schedule rules 产生候选；

默认可以用 evolutionary search；

XGB/MLP cost model 用测量结果学习；

真正候选又会在目标硬件上运行获得 latency。citeturn450339search0turn450339search1

也就是说：

\[
\boxed{
candidate\ generation
\rightarrow
learned\ ranking
\rightarrow
hardware\ measurement
}
\]

和 Hexcute 的结构完全不同。

---

# 11. TVM 的 candidate space 其实很具体

例如 `MultiLevelTilingTensorCore` 可以指定：

\[
intrin\ groups
\]

\[
tiling\ structure
\]

\[
tile\ binding
\]

\[
vector\ load\ lengths
\]

\[
reuse\_read/reuse\_write
\]

\[
software\ pipeline
\]

fileciteturn52file0L1-L7

而 MetaSchedule 还有 `RewriteLayout`，专门针对 `layout_free_buffers` 推导 IndexMap，再插 layout rewrite preprocessing。fileciteturn53file0L1-L7

所以 TVM 的特点是：

\[
\boxed{
Search\ quality
\quad\text{strongly depends on}\quad
ScheduleRule\ coverage
}
\]

不是所有合法 GPU mapping 都天然在 search space 中。

---

# 12. TVM 还有一个其他几者没有的明显优势：graph scope

Relax 有：

```text
FuseOps
FuseTIR
FuseOpsByPattern
```

可以先从 graph 层决定哪些 operator 形成一个 fused kernel，再进入 TIR schedule。citeturn223392search0turn223392search4

因此 TVM 是这里唯一一个比较完整地同时具有：

\[
\text{graph transformation}
+
\text{kernel search}
\]

的系统。

不过这并不等于 graph decision 与 low-level mapping 已经 joint-search。

一般还是：

```text
Graph fusion
    ↓
produce TIR task
    ↓
MetaSchedule tune task
```

这点后面非常重要。

---

# 13. TIRx：和 CuTe/Gluon 又是一个完全不同的 ownership 划分

TIRx 当前最有意思的设计决定是：

> layout 是 storage contract，不是 work-partitioning interface。

TIRx 用户描述：

\[
logical\ tensor
\rightarrow
lane/warp/register/shared/TMEM
\]

但调用 TilePrimitive 后：

\[
operand\ layouts
+
execution\ scope
+
target
\]

交给 primitive dispatcher。

dispatcher 再生成：

\[
thread\ partition
+
loop
+
addressing
+
instruction
\]

官方文档直接用：

- global→shared 可能变 TMA；
- shared→register 可能变 ldmatrix；
- matrix multiply 变 WGMMA/tcgen05 等，

来解释这个机制。citeturn419041search1turn419041search2

---

# 14. TIRx dispatch 的选择机制也不是 autotuning

当前 dispatch table 的每个 implementation 有：

\[
variant,\ priority,\ predicates,\ implementation
\]

运行时按 priority 排序，检查 predicate，选择第一个接受的实现；用户也可以显式 `dispatch=` override。citeturn419041search2

所以：

\[
\boxed{
TIRx:
Explicit\ storage/layout
+
Rule/predicate\ based\ instruction/work\ dispatch
}
\]

而且 pipeline state、barrier、control-flow 等仍然显式存在于 kernel program 中。citeturn419041search1

这和 Hexcute有明显相似性：

> 两者都刻意没有把整个 pipeline/dataflow 全部交给自动优化器。

---

# 15. Hexcute：它真正自动化的是“合法 mapping synthesis”

Hexcute artifact 明确把一个 Hidet fork 作为 submodule。fileciteturn77file0L1-L7

CGO 2026 对它的定位是：

> 自动 layout synthesis，但保留对 dataflow 和 pipelining 的显式控制；将 layout synthesis formalize 为 constraint programming/type inference。citeturn437713search0

实现中自动推导分四层：

\[
LogicalShape
\rightarrow
LogicalLayout
\rightarrow
TaskMapping
\rightarrow
MemoryLayout
\]

fileciteturn80file0L1-L7

对于 Copy：

\[
f\circ p^{-1}
=
g\circ q^{-1}
\]

对于 MMA：

\[
f_{A,M}=f_{C,M}
\]

\[
f_{B,N}=f_{C,N}
\]

\[
f_{A,K}=f_{B,K}
\]

这相当于从 instruction 的 thread/value semantics 反推合法 tensor mapping。fileciteturn80file0L1-L7

---

# 16. Hexcute 的 instruction selection 也和 layout 深度绑定

它的 instruction registry/import 直接覆盖：

```text
ldg/stg
lds/sts
ldmatrix
mma_sync
wgmma
cp_async
TMA
```

并根据 scope、alignment、thread-value layout compatibility 等判断某 instruction 能否匹配。fileciteturn82file0L1-L7

因此 Hexcute 做的不是：

\[
layout\ only
\]

而比较接近：

\[
\boxed{
Layout
+
TaskMapping
+
Instruction
}
\]

的联合合法性推导。

---

# 17. 但它的 performance decision 又是另一层

Hexcute fork 里的 cost model 明确采用：

\[
Cost
\approx
N_{\text{instruction}}
\times
InstructionLatency
\]

并模拟 copy/MMA overlap。

同时源码自己写明：

- bank conflict impact 未来加入；
- pipeline 假设 full overlap；
- `cp_async_wait_group()` 和 `mbarrier` 没有进入模型；
- tensor manipulation/address arithmetic 被视为 negligible。fileciteturn79file0L1-L7

而 bank conflict 又由另一个 pass：

```text
ResolveBankConflict
```

分析 shared tensor 的所有 copy consumer，枚举 swizzle 来降低 conflict。fileciteturn81file0L1-L7

这意味着当前实际上是：

\[
\text{legal mapping synthesis}
\]

\[
\downarrow
\]

\[
\text{instruction/cost ranking}
\]

\[
\downarrow
\]

\[
\text{bank-conflict correction}
\]

并不是一个完全统一的 optimization objective。

---

# 18. 现在再比较“候选是怎么来的”

这一张表我认为对科研问题尤其重要。

| 系统 | Candidate 从哪里来 | Legality | Performance 选择 |
|---|---|---|---|
| SGLang | backend registry + model/hardware rules | feature/backend compatibility | 手写性能规则 |
| vLLM | backend registry + KV layout enum | supported-set intersection | preference ordering |
| CUTLASS | expert template/layout compositions | C++ type/template constraints | expert + builder rules |
| Triton classic | programmer kernel + compiler rewrite | IR/encoding/instruction rules | heuristic；meta-config 可实测 |
| Gluon | programmer explicit low-level configs | compiler verifier | programmer + autotune |
| TVM | ScheduleRule-generated space | schedule/postproc/intrin requirements | learned model + hardware measurement |
| TIRx | registered primitive variants | predicates | priority order |
| Hexcute | constraint-derived mapping space | **constraint unification/backtracking** | analytic cost model |

这比“自动化程度排行榜”有信息得多。

---

# 19. 我认为这些系统有五个非常重要的共同点

第一，所有系统都在**压缩搜索空间**。

没有一个系统真的枚举：

\[
\text{all layouts}
\times
\text{all mappings}
\times
\text{all instructions}
\times
\text{all pipelines}
\]

CUTLASS 用 expert templates；

Triton 用 heuristics；

TVM 用 ScheduleRules；

TIRx 用 registered variants；

Hexcute 用 constraints；

vLLM/SGLang 用 backend capability sets。

---

第二，所有系统都区分了某种意义上的：

\[
\boxed{合法性}
\]

和：

\[
\boxed{性能优劣}
\]

只是形式不同。

---

第三，所有系统都有明确的 **optimization boundary**。

没有一个系统同时拥有：

\[
\begin{aligned}
&Serving workload\\
+&Persistent storage\\
+&Graph fusion\\
+&Kernel dataflow\\
+&Thread mapping\\
+&Instruction\\
+&Pipeline
\end{aligned}
\]

的统一决策权。

---

第四，专家知识没有消失，只是位置变化了。

\[
CUTLASS:
\text{template}
\]

\[
Triton:
\text{passes/heuristics}
\]

\[
TVM:
\text{schedule rules}
\]

\[
TIRx:
\text{dispatch predicates}
\]

\[
Hexcute:
\text{constraints/cost model}
\]

\[
vLLM/SGLang:
\text{backend capability/policy}
\]

---

第五，也是最重要的一条：

> **这些系统最根本的区别，是如何划分“决策权”。**

不是谁的 layout class 更漂亮。

---

# 20. 由这张表重新推科研问题

我现在会把科研问题筛成 5 个，其中前三个我认为明显比上一轮更强。

---

# 科学问题 P1：Optimization Boundary 是否真的可分解？

这是我现在最看好的总问题。

现有系统实际上都在假定：

\[
D=
(D_r,D_p,D_g,D_t,D_m,D_i,D_{pipe})
\]

可以分块处理。

其中可以分别表示：

runtime、

persistent representation、

graph/dataflow、

tile、

mapping、

instruction、

pipeline。

现实实现大致是：

\[
D_r
\rightarrow
D_p
\rightarrow
D_g
\rightarrow
D_{kernel}
\]

每一块被不同系统或 pass 处理。

---

## 科学问题

不是：

> “joint optimization 会不会更快？”

那个太显然。

真正的问题应该是：

\[
\boxed{
现代 GPU workload 中，
这些 decision blocks 在什么条件下近似可分解，
什么条件下会出现强 coupling？
}
\]

定义 joint optimum：

\[
d^*
=
\arg\min_{d\in D}T(d)
\]

而某种分层系统得到：

\[
\hat d
=
Optimize_k(
\cdots Optimize_2(Optimize_1(D))
)
\]

定义：

\[
\boxed{
BoundaryRegret
=
\frac{T(\hat d)}{T(d^*)}-1
}
\]

研究的问题就是：

\[
BoundaryRegret
=
f(
workload,
hardware,
decision\ boundary
)
\]

---

## 为什么这个问题是从所有框架共同结构中来的

SGLang/vLLM 划在 runtime/persistent boundary。

CUTLASS 划在 kernel expert construction。

Triton 又把 kernel 内部切成多个 passes。

TVM 把 graph fusion 与 task tuning 分开。

TIRx 把 storage layout / primitive dispatch 与 pipeline orchestration 分开。

Hexcute 把 layout/instruction synthesis 与 explicit dataflow/pipeline 分开。

因此这是一个真正的**跨系统共同假设**。

---

## 可证伪预测

如果以下情况成立，这个问题就没有我们想象的那么重要：

\[
BoundaryRegret < 1\%-2\%
\]

在大量 LLM/Diffusion workload、shape、GPU 上都成立；

并且改变上游 decision 几乎不会改变下游 optimum。

那就说明现有 decomposition 非常合理。

反之，如果 regret 在特定 workload 上系统性增大，我们才有论文故事。

---

## 最合适的 workload

最强：

\[
DecodeAttention
\]

因为天然连接：

\[
KV\ physical\ layout
\leftrightarrow
thread\ mapping
\leftrightarrow
memory\ access
\]

其次：

\[
QK^T\rightarrow Softmax\rightarrow PV
\]

以及：

\[
Gate/Up\ GEMM
\rightarrow SiLU\times
\rightarrow Down\ GEMM
\]

Diffusion 可以用：

\[
Linear/Conv
\rightarrow Norm/activation
\rightarrow Linear/Conv
\]

验证 generality。

---

# 科学问题 P2：不同 Candidate-Generation 机制会不会系统性丢掉近最优解？

这个问题来自 CUTLASS/Triton/TVM/TIRx/Hexcute 最明显的区别。

设整个 expressible/legal space：

\[
S
\]

框架 \(F\) 实际产生的 candidate set：

\[
G_F\subseteq S
\]

定义：

\[
T^*_S
=
\min_{x\in S}T(x)
\]

\[
T^*_F
=
\min_{x\in G_F}T(x)
\]

则：

\[
\boxed{
CoverageRegret(F)
=
\frac{T^*_F}{T^*_S}-1
}
\]

注意这个 metric 完全不考虑“selector 有没有选错”。

它只问：

> **最优 candidate 到底有没有进入 framework search space？**

---

## 这能把两个长期混在一起的问题分开

### Candidate generation error

真正好配置根本没有产生。

和：

### Candidate ranking error

好配置产生了，但 selector/cost model 没选中。

也就是：

\[
TotalRegret
=
CoverageRegret
+
SelectionRegret
+\text{interaction}
\]

概念上必须拆开。

---

## 不同系统正好提供不同 candidate generator

\[
CUTLASS:
Expert\ template\ family
\]

\[
Triton:
Compiler\ heuristic/rewrite
\]

\[
TVM:
ScheduleRule
\]

\[
TIRx:
PrimitiveVariant
\]

\[
Hexcute:
ConstraintInference
\]

所以这不是针对某一个系统的 bug。

---

## 科学问题可以进一步写成

\[
\boxed{
高性能 GPU mapping 在整个合法空间中
是否具有可利用的结构，
从而可以大幅压缩候选空间而不损失 near-optimal coverage？
}
\]

这就比“用 constraint search 好不好”更基础。

---

## 可证伪预测

如果 high-performing configs 在合法空间中非常随机、分散：

\[
P(x\text{ near-optimal}\mid structural\ constraints)
\]

没有明显提升，

那么 constraint-guided candidate generation 的理论价值会很有限。

反过来，如果：

alignment、

instruction compatibility、

thread/value structure、

memory access pattern

能够把空间减少几个数量级，同时保留绝大多数：

\[
\le5\%
\]

oracle configs，

这就是很强的科学结果。

---

# 科学问题 P3：同一个数据表示面对多个 Consumer 时，optimality 是否可组合？

这是之前“cross-op layout”的更正确版本。

问题不是“operator A 和 B layout 不一样”。

而是一个数据对象：

\[
X
\]

同时被不同 consumer 使用：

\[
C_1,C_2,\ldots,C_n
\]

每个 consumer 对 representation 有一个性能函数：

\[
T_i(L)
\]

于是可能：

\[
\arg\min T_1(L)
\neq
\arg\min T_2(L)
\]

---

## 在不同框架中都能看到这个现象

Triton：

memory load 喜欢 blocked/coalesced layout，而 MMA 喜欢 MMA encoding，因此需要 `ConvertLayout` 和后续 removal/rewriting。fileciteturn67file0L1-L7

Hexcute：

同一个 shared-memory tensor 可以有多个 Copy consumer，所以 memory constraints 需要统一。fileciteturn80file0L1-L7

vLLM：

一个 KV persistent representation 同时服务不同 attention backend / connector / transfer path，而当前整个模型 resolve 一个 layout。fileciteturn44file0L1-L7

TVM：

layout-free buffer rewrite 与 FuseTIR 都是为了让 producer/consumer representation 更适合新的整体实现。fileciteturn53file0L1-L7 citeturn223392search4

---

## 所以真正科学问题是

\[
\boxed{
Multiple-consumer representation preferences
是否可以用 compositional cost model 表达？
}
\]

例如：

\[
Cost(L)
=
\sum_i C_i(L)
+
\sum_{ij}Convert_{ij}(L_i,L_j)
\]

够不够？

还是因为：

cache residency、

occupancy、

pipeline overlap、

resource contention

导致：

\[
Cost(L)
\neq
\sum local\ costs
\]

必须看完整执行上下文？

这个问题比简单“减少 layout conversion”深很多。

---

# 科学问题 P4：Persistent Representation 的 optimum 是静态配置还是 workload policy？

这个问题主要来自 SGLang/vLLM。

当前 vLLM 会为整个 model resolve 一个 KV layout。fileciteturn44file0L1-L7

但 SGLang 已经明确认为：

\[
Backend_{prefill}
\neq
Backend_{decode}
\]

可能是合理的。fileciteturn35file0L1-L7

于是自然出现矛盾：

backend 可以 phase-specific，

为什么 persistent representation 一定要 model-static？

---

设 workload state：

\[
x=
(B,S_q,S_{kv},
prefill/decode,
reuse,
transfer,
spec,\ldots)
\]

当前很多设计近似：

\[
L^*
=
f(Model,Hardware)
\]

科学问题是：

\[
\boxed{
真正 optimum 是否应该是
L^*=f(Model,Hardware,\mathcal W)
}
\]

甚至：

\[
L_t=\pi(x_t)
\]

其中要扣除改变 representation 的 switching cost。

---

## 这不是简单 dynamic layout

真正需要研究的是：

> workload 空间中的 optimum 是否形成少量稳定的“configuration regions”？

如果只是：

```text
prefill region
decode-short region
decode-long region
transfer-heavy region
```

几个大区域，

动态 policy 很可能可行。

如果最优 layout 随每个微小 shape 都剧烈变化，则 policy 会非常复杂。

---

## 可证伪预测

如果同一静态 layout：

\[
L_s
\]

在所有 realistic serving distributions 中都距离 oracle：

\[
<1\%-2\%
\]

则没有必要研究 dynamic persistent policy。

所以这个问题是非常容易被实验否掉的。

这很好。

---

# 科学问题 P5：对现代异步 GPU，什么信息足以正确“排序”候选？

这个是 cost-model 方向的更强版本。

不是：

> Hexcute 没考虑 bank conflict。

那个只是工程缺口。

真正科学问题是：

\[
\boxed{
为了找出 fastest configuration，
性能模型到底需要建模到什么程度？
}
\]

注意，我们甚至不一定需要：

\[
\hat T=T
\]

绝对预测准确。

真正需要的是 ranking：

\[
T(A)<T(B)
\Rightarrow
\hat T(A)<\hat T(B)
\]

---

## 六类系统形成了天然对照

CUTLASS：

expert/static rules。

Triton：

heuristics + empirical autotune。

TVM：

learned cost model + measurement。

TIRx：

priority/predicate。

Hexcute：

instruction-latency analytic model。

SGLang/vLLM：

runtime policy/preference rules。

所以这里真正的问题是：

\[
\boxed{
Analytic/local features 是否足以保持
modern async GPU configuration 的性能排序？
}
\]

---

## 为什么 Hopper/Blackwell 特别值得研究

因为：

\[
TMA
\]

\[
WGMMA/tcgen05
\]

\[
mbarrier
\]

\[
warp specialization
\]

\[
TMEM
\]

\[
dynamic register allocation
\]

使 performance 越来越取决于：

\[
\text{overlap}
+
\text{resource interaction}
\]

而不仅是：

\[
\#instructions\times latency
\]

Hexcute 当前模型的明确假设正好提供一个现实 motivation。fileciteturn79file0L1-L7

---

# 21. 现在我对五个问题的评价

| 问题 | 科学性 | LLM 相关性 | Diffusion generality | 实验可证伪性 | 我现在的评价 |
|---|---:|---:|---:|---:|---|
| P1 Boundary separability | ★★★★★ | ★★★★★ | ★★★★☆ | ★★★★★ | **最强主线** |
| P2 Candidate-space coverage | ★★★★★ | ★★★★☆ | ★★★★★ | ★★★★★ | **最强 compiler 主线** |
| P3 Multi-consumer representation | ★★★★☆ | ★★★★★ | ★★★★★ | ★★★★★ | **非常适合做具体机制** |
| P4 Workload-dependent persistent state | ★★★★☆ | ★★★★★ | ★★☆☆☆ | ★★★★★ | **很强 LLM serving 方向** |
| P5 Ranking sufficiency | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★★★ | **适合 cost-model 论文** |

---

# 22. 我现在认为最值得继续推进的不是五个都做

如果目标是一篇统一的科研故事，我建议进一步集中成：

\[
\boxed{\textbf{P1 + P2}}
\]

作为核心理论/方法学问题。

即：

> 现有 GPU 优化系统通过 abstraction boundaries 将整体决策空间分解，同时通过 templates、heuristics、schedule rules、constraints 等机制压缩候选空间。  
> 我们需要回答两个基础问题：  
> **这种分解会造成多大的 optimization regret？**  
> **这种候选空间压缩会造成多大的 coverage regret？**

然后：

\[
P3/P4/P5
\]

作为产生这些 regret 的三种具体机制：

```text
P3:
多个 consumer 的 representation conflict

P4:
workload/time-varying persistent representation

P5:
候选存在，但 performance ranking 错误
```

这会形成一个非常干净的逻辑结构：

```text
                     Total Optimization Regret
                              │
               ┌──────────────┼──────────────┐
               │              │              │
               ▼              ▼              ▼
        Boundary Regret   Coverage Regret  Ranking Regret
               │              │              │
       decisions split    candidate absent   candidate exists
       across systems      from search space but selected wrong
```

我认为这张图比我们之前所有“科学问题列表”都更接近一个真正可以形成论文的研究框架。

---

# 23. 哪些东西我现在明确不建议单独当科研问题

“Hexcute cost model 不考虑 bank conflict”：

这是 P5 的一个证据，不是科研问题本身。

“Triton 有很多 ConvertLayout”：

这是 P3/P1 的 observable symptom。

“vLLM KV layout 固定”：

这是 P4 的现实设计选择。

“SGLang backend selector 是 heuristic”：

这是 P2/P5 的一个实例。

“CUTLASS 需要人工指定 layout”：

这是 candidate ownership 的一个 design point。

“TVM search 太慢”：

这是 P2 中 search-space quality/cost trade-off 的一个现象。

这样区分以后，科研问题就不会退化成“找 framework 缺点然后修 bug”。

---

# 24. 下一步应该怎么继续

我建议下一轮仍然**不写代码**。

下一轮我们应该针对 P1–P5 做第二次筛选，建立：

\[
\boxed{\text{Problem Evidence Matrix}}
\]

具体逐问题回答：

| 科研问题 | 哪些 framework 提供正面证据 | 哪些 framework 已经部分解决 | 哪些框架可能构成反例 | 最小真实子图 | 需要观察什么现象才能证明“问题确实存在” | 什么实验结果会直接否定这个问题 |
|---|---|---|---|---|---|---|

尤其要做一件很重要的事：

> **主动寻找“这个科研问题可能不存在”的证据。**

例如对 P1，我们不能只是找一个 joint optimization 比 sequential 快的 case；我们需要先看大量现实 workload 中 boundary regret 到底是不是普遍存在、是否可预测。

对 P2，也不能只是找到 Triton 漏掉一个 configuration；要证明不同 candidate-generation paradigm 存在系统性 coverage gap。

如果你认可这一轮的 taxonomy，我下一轮就专门做这张 **P1–P5 Problem Evidence Matrix**，同时把每个问题进一步收敛成真正可以写进论文 Introduction 的 “Observation → Research Question → Hypothesis → Falsifiable Prediction”，仍然先不设计任何代码。

## 用户

请继续下一轮

## 模型回复

这一轮我会做两件事：第一，给 P1–P5 建“**证据—反证—已部分解决**”矩阵；第二，把每个问题改写成真正可证伪的科研命题。重点不再是“哪里有缺点”，而是判断这些问题是否真的跨框架、是否有普遍性、以及什么结果会直接把它否掉。

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 3 个网站

## 模型思考摘要

构建问题矩阵
构建了问题矩阵

## 模型思考摘要

统一研究框架

## 模型思考摘要

已搜索 3 个网站

## 模型思考用时

思考了 43s

## 模型回复

可以。经过这一轮源码和官方文档的交叉核对，我认为现在应该做一次更严格的收敛：不是继续“增加科研问题”，而是判断 P1–P5 中哪些是真正独立、哪些只是某个更大问题的机制。

先给这一轮最重要的结论：

\[
\boxed{
\text{现有框架差异最适合统一成三类 Optimization Regret：}
}
\]

\[
\boxed{
Boundary\ Regret,\quad Coverage\ Regret,\quad Selection\ Regret
}
\]

而我们前面提到的 multi-consumer conflict、静态 KV representation、cost-model fidelity，其实分别是造成这三类 regret 的具体机制。这样科研问题会比“layout 优化哪里不好”清晰很多。

---

# 1. 先区分两种“证据”，否则很容易把科研问题论证错

源码目前能够证明的是：

> 框架确实采用了某种 decomposition、candidate-generation 或 ranking mechanism。

但源码不能直接证明：

> 这种机制一定造成性能损失。

例如 Hexcute 明确把 layout synthesis 和 programmer-controlled pipeline 分开。这个事实可以从设计和源码证明。citeturn839016search1

但我们不能因此直接下结论：

\[
\text{separate layout/pipeline optimization is bad}
\]

真正需要实验回答的是：

\[
\text{这种 boundary 是否产生 measurable regret？}
\]

所以后面我会严格区分：

\[
\textbf{Design Evidence}
\]

和

\[
\textbf{Performance Claim}
\]

前者现在可以由源码支持；后者仍然是研究假设。

---

# 2. 我建议先建立一个统一的“Regret 分解”

假设对于一个真实 LLM/Diffusion workload \(w\)，理论上可以考虑一个完整合法实现空间：

\[
\mathcal U(w)
\]

其中一个 configuration 不只是 layout，而是：

\[
c=
(
A,
F,
L_p,
D,
Tile,
T,
L_s,
L_r,
I,
P
)
\]

分别可以表示：

\[
\begin{aligned}
A &: algorithm/backend\\
F &: fusion/subgraph boundary\\
L_p &: persistent\ representation\\
D &: kernel\ dataflow\\
Tile &: tile\ decomposition\\
T &: thread/warp\ mapping\\
L_s &: shared-memory\ layout\\
L_r &: register/distributed\ layout\\
I &: instruction\ selection\\
P &: pipeline/orchestration
\end{aligned}
\]

真正 oracle 是：

\[
c^*
=
\arg\min_{c\in\mathcal U}T(c)
\]

但现实框架不会搜索 \(\mathcal U\)。

它们首先用 abstraction boundary 把空间分解，然后用自己的 candidate generator 进一步压缩，最后再用 heuristic/model/measurement 选择。

因此可以非常干净地写成三层。

---

# 3. 第一层：Boundary Regret

框架选择了一种 optimization decomposition：

\[
B
\]

例如：

```text
vLLM:
persistent KV layout
        ↓
backend kernel

TVM:
graph fusion
        ↓
TIR task tuning

Hexcute:
dataflow + pipeline fixed
        ↓
layout/task/instruction synthesis

Triton:
coalescing
   ↓
MMA layout
   ↓
pipeline
   ↓
conversion cleanup
```

这相当于把全空间限制成：

\[
\mathcal U_B\subseteq\mathcal U
\]

于是定义：

\[
\boxed{
R_B
=
\frac{
\min_{c\in\mathcal U_B}T(c)
}{
\min_{c\in\mathcal U}T(c)
}
-1
}
\]

这就是 Boundary Regret。

它回答：

> **因为把某些决策分开做，最优方案是否已经不可能被表达/选择？**

---

# 4. 第二层：Coverage Regret

即使 boundary 固定了，框架一般也不会遍历整个：

\[
\mathcal U_B
\]

而是 candidate generator \(G\) 产生：

\[
\mathcal C_G\subseteq\mathcal U_B
\]

例如：

CUTLASS：

\[
G=\text{expert templates/builders}
\]

Triton：

\[
G=\text{compiler rewrites + user autotune configs}
\]

TVM：

\[
G=\text{ScheduleRules}
\]

TIRx：

\[
G=\text{registered primitive variants}
\]

Hexcute：

\[
G=\text{constraint inference}
\]

于是：

\[
\boxed{
R_C
=
\frac{
\min_{c\in\mathcal C_G}T(c)
}{
\min_{c\in\mathcal U_B}T(c)
}
-1
}
\]

这是 Coverage Regret：

> **candidate-generation mechanism 是否漏掉了真正优秀的 configuration？**

---

# 5. 第三层：Selection Regret

就算最佳 configuration 已经在 candidate set：

\[
c^*_{\mathcal C}\in\mathcal C_G
\]

selector 仍然可能选错。

框架最后选：

\[
\hat c
=
S(\mathcal C_G)
\]

于是：

\[
\boxed{
R_S
=
\frac{
T(\hat c)
}{
\min_{c\in\mathcal C_G}T(c)
}
-1
}
\]

这是 Selection Regret。

它回答：

> candidate 明明存在，为什么 heuristic/cost model/ranking 没选中？

---

# 6. 三个 regret 还能形成非常漂亮的总分解

如果使用乘法定义：

\[
1+R_{total}
=
\frac{T(\hat c)}
{\min_{\mathcal U}T(c)}
\]

那么：

\[
\boxed{
1+R_{total}
=
(1+R_B)
(1+R_C)
(1+R_S)
}
\]

这是我这一轮认为最有价值的结果。

因为它第一次把：

> SGLang、vLLM、CUTLASS、Triton、TVM、TIRx、Hexcute 为什么不同

统一到了一个可实验研究的框架里。

不是简单比较性能。

而是回答：

\[
\boxed{\text{性能损失到底来自哪里？}}
\]

---

# 7. 现在重新看 P1–P5

| 原问题 | 实际属于 | 是否保留为独立主问题 |
|---|---|---|
| P1 Optimization Boundary separability | Boundary Regret | **保留，主问题** |
| P2 Candidate-generation coverage | Coverage Regret | **保留，主问题** |
| P3 Multi-consumer representation conflict | Boundary Regret 的重要机制 | **降为机制问题** |
| P4 Workload-dependent persistent representation | Boundary/Policy Regret 的特殊情形 | **可独立成 LLM serving 方向** |
| P5 Performance-ranking sufficiency | Selection Regret | **保留，但作为第三层问题** |

也就是说，我现在不建议再写成五个平行 scientific questions。

更合理的结构是：

\[
\boxed{
Q_1:\ Boundary\ Regret
}
\]

\[
\boxed{
Q_2:\ Coverage\ Regret
}
\]

\[
\boxed{
Q_3:\ Selection\ Regret
}
\]

P3/P4 是用于研究 Q1 的具体机制。

---

# 8. Q1：Optimization Boundary Regret

## 8.1 跨框架的设计证据

这是目前证据最强的问题。

### SGLang

SGLang 有明确的 runtime backend policy。

当前默认 backend selection 会根据：

GPU architecture、

MHA/MLA、

spec decoding、

attention sinks、

asymmetric K/V、

FlashInfer availability 等规则选择 backend。fileciteturn64file0L1-L7

并且 prefill/decode backend 可以分离。

这已经说明：

\[
Backend^*_{prefill}
\neq
Backend^*_{decode}
\]

是系统显式考虑的可能性。

也就是说，SGLang 本身就在利用一种跨 phase 的“不应强制共享同一决策”的事实。

---

### vLLM

vLLM 的另一个方向正好相反。

当前：

\[
KVCacheLayout
\]

在 engine core 为整个 model resolve 一次，然后在 allocation 前固定下来。官方文档明确写：

> Resolve one KV cache layout for the whole model. citeturn887469search4

CacheConfig 也明确说明 layout resolve 后就是 final。citeturn481206search3

而 backend 又可以 per KV-cache group 不同。citeturn481206search0

所以现在可以出现：

```text
backend 1
backend 2
backend 3
    │
    └──────→ one persistent KV layout
```

这就是非常清晰的 optimization boundary。

---

### Triton

Triton kernel compiler内部又采用另一种 decomposition。

例如 `Coalesce` 为 load/store 产生 cache-friendly layout，并允许插入 conversion；`AccelerateMatmul` 使 layout 满足 tensor-core operation；后续 `RemoveLayoutConversions` 又尝试协调两者。fileciteturn66file0L1-L7 fileciteturn67file0L1-L7

还有独立的：

\[
Pipeline
\]

\[
AutomaticWarpSpecialization
\]

\[
OptimizeThreadLocality
\]

等 pass。fileciteturn66file0L1-L7

所以 kernel 内部本身也是 staged optimization。

---

### TVM

TVM 的 boundary 又在：

\[
Relax\ graph
\rightarrow
TIR\ tuning\ task
\]

MetaSchedule 从 Relax program 提取 tuning tasks，并对每个 TIR workload 搜 schedule。citeturn884265search4turn884265search8

也就是说 graph transform 与 low-level schedule search 虽然在一个系统中，但主要仍是不同 optimization stages。

---

### TIRx

TIRx 对 boundary 的设计甚至直接写进 design philosophy：

> pipeline structure、synchronization、role assignment、memory placement 等 orchestration 保持显式；execution scope/layout/tile primitive 则暴露给 compiler。citeturn414668search0turn414668search3

这是非常强的证据，因为它不是偶然实现细节，而是明确的 abstraction-boundary choice。

---

### Hexcute

Hexcute又选择：

\[
\text{dataflow + pipelining explicit}
\]

但：

\[
\text{layout + task mapping + instructions automatic}
\]

CGO 论文摘要明确这样描述。citeturn839016search1

所以 CUTLASS/Triton/TVM/TIRx/Hexcute 实际是在做不同的：

\[
\boxed{\text{programmer/compiler decision partition}}
\]

---

# 9. Q1 已经有哪些部分解决方案？

这很重要，否则我们会把“完全没人做”说过头。

CUTLASS 的 `CollectiveMma` 本身就在一个 mainloop abstraction 内联合表达：

\[
Tile
+
GmemCopy
+
SmemLayout
+
SmemCopy
+
MMA
+
StageCount
+
KernelSchedule
\]

fileciteturn51file10L163-L174

所以它部分降低了 kernel 内 boundary fragmentation。

Hexcute 则把：

\[
Layout
+
TaskMapping
+
Instruction
\]

通过 constraint system 联合起来。fileciteturn80file0L1-L7

Triton 也有 `RemoveLayoutConversions` 等后处理试图协调不同 pass 的局部决策。fileciteturn67file0L1-L7

因此 Q1 不应该写：

> existing frameworks optimize everything independently.

这是不准确的。

应该写成：

\[
\boxed{
现有框架选择不同大小的联合优化区域；
尚不清楚这些边界在哪些 workload 上已经足够，
在哪些 workload 上仍会产生显著 regret。
}
\]

这个表述严谨很多。

---

# 10. Q1 的真正科学问题

我现在会写成：

> **Under what workload and hardware conditions can modern GPU optimization decisions be decomposed across runtime, representation, mapping, instruction, and pipeline boundaries without significant performance regret?**

中文就是：

\[
\boxed{
现代 GPU 优化决策在什么条件下具有近似可分解性？
}
\]

注意关键词是：

\[
\textbf{conditions}
\]

而不是简单问：

> joint optimization 比 sequential 好吗？

后者答案几乎必然是“有时候”。

前者才可能形成 general knowledge。

---

# 11. Q1 的 Hypothesis

目前我认为比较强的假设是：

\[
\boxed{
BoundaryRegret
\text{ 与跨 boundary 的资源/数据依赖强度存在系统关系}
}
\]

例如：

\[
R_B
=
f(
\text{data reuse},
\text{conversion volume},
\text{shared-state lifetime},
\text{async overlap},
\text{memory intensity},
\text{consumer diversity}
)
\]

具体预测：

对于普通 standalone GEMM：

\[
R_B\approx0
\]

因为 workload 很规则，CUTLASS/Triton 的局部 kernel optimization 已经很好。

但对于：

\[
DecodeAttention
\]

中 persistent KV layout 与 warp access pattern 强耦合：

\[
R_B
\]

可能明显变大。

对于：

\[
QK^T\rightarrow Softmax\rightarrow PV
\]

中间 tensor 同时面对 reduction 与 MMA：

\[
R_B
\]

也可能明显变大。

---

# 12. Q1 的反证标准

如果我们以后发现：

对现代 LLM/Diffusion 中大量真实 subgraph：

\[
R_B < 2\%
\]

而且在：

A100/H100/B200、

不同 dtype、

不同 sequence length、

不同 batch、

不同 attention variant

上都成立，

那么：

\[
\boxed{\text{Q1 基本应该放弃}}
\]

因为那说明现有 abstraction decomposition 本身已经非常合理。

这是一个好科研问题，因为它可以被否定。

---

# 13. Q1 最好的验证对象

我建议按顺序：

| 子图 | Boundary coupling 来源 | Q1 价值 |
|---|---|---:|
| Decode Attention | persistent KV ↔ memory mapping ↔ reduction | ★★★★★ |
| Prefill Attention | QK ↔ softmax ↔ PV | ★★★★★ |
| SwiGLU/GEGLU | GEMM ↔ elementwise ↔ GEMM | ★★★★☆ |
| MoE grouped GEMM | routing ↔ dynamic shape ↔ GEMM | ★★★★☆ |
| Diffusion attention | QKV ↔ attention ↔ projection | ★★★★☆ |
| standalone GEMM | coupling 弱 | ★★☆☆☆，适合作 control |

这里 standalone GEMM 反而应该保留。

因为我们需要一个：

\[
\boxed{\text{negative control}}
\]

证明方法不是“所有东西联合搜索总会更快”。

---

# 14. Q2：Candidate-Space Coverage Regret

这个问题我现在认为和 Q1 同样强。

不同框架最根本的区别之一就是：

> 它们如何产生值得考虑的 candidate。

---

# 15. Q2 的跨框架证据

### CUTLASS

官方当前文档自己指出：

`CollectiveBuilder` 尝试根据输入构造高性能 `CollectiveMma`，但 builder 并不覆盖 expert `CollectiveMma` API 的全部 design space。citeturn887469search2

因此天然存在：

\[
\mathcal C_{builder}
\subset
\mathcal U_{CUTLASS}
\]

这是非常漂亮的 coverage 例子。

---

### Triton

classic Triton 的 internal candidate 来源主要是 pass heuristics。

用户 autotune 则需要用户提供 candidate configs：

\[
\{config_1,\ldots,config_n\}
\]

然后 benchmark。

因此：

\[
\text{autotune 能否找到好配置}
\]

首先受限于：

\[
\boxed{\text{用户有没有把那个 configuration 放进去}}
\]

而很多 distributed-layout compiler choices 又不直接是 autotune parameter。

---

### TVM

TVM 更明显。

MetaSchedule 明确：

> search-based tuning explores TIR schedules generated by space generators/schedule rules，并在硬件上测量。citeturn884265search8

TensorCore schedule rule 的 candidate family包括：

tiling structure、

thread binding、

vector load lengths、

reuse read/write、

intrinsic groups、

software pipeline。fileciteturn52file0L1-L7

所以：

\[
\mathcal C_{TVM}
=
G_{\text{ScheduleRules}}(workload)
\]

它不可能搜索没有被 schedule rules 表达的结构。

---

### TIRx

TilePrimitive dispatch 更明确。

候选是 backend 已经注册的 variant；每个有：

\[
predicate,\ priority,\ implementation
\]

dispatcher 从这些 variants 中选择。citeturn884265search0turn884265search5

所以：

\[
\mathcal C_{TIRx}
=
\{\text{registered implementations}\}
\]

---

### Hexcute

Hexcute 则不是人工列举 mapping。

它从 instruction/layout constraint 推导可能合法的 task mapping。

源码甚至明确写：

> copy 可能有多个可执行 instruction，可以 DFS 找 valid variants，再 cost model 选择，也可以 heuristic/beam search prune。fileciteturn80file0L1-L7

因此：

\[
\mathcal C_{Hexcute}
\]

比显式模板更具系统性。

但它仍由：

constraint vocabulary、

instruction registry、

fixed tile-level dataflow

限定。

---

# 16. Q2 的真正问题不是“搜索空间大”

“搜索空间太大”不是一个很好的科学问题。

真正问题是：

\[
\boxed{
高性能 GPU configuration
在合法空间中是否具有可利用的结构，
使我们可以大幅压缩空间而保持 near-optimal coverage？
}
\]

形式化一点：

合法空间：

\[
\mathcal U
\]

candidate generator：

\[
G_\theta:\mathcal U\rightarrow\mathcal C
\]

我们希望同时：

\[
|\mathcal C|\ll|\mathcal U|
\]

以及：

\[
\frac{
\min_{\mathcal C}T
}{
\min_{\mathcal U}T
}
\approx1
\]

这其实是一个：

\[
\boxed{
Search-space compression vs optimum preservation
}
\]

问题。

---

# 17. Q2 的核心 Hypothesis

我现在更倾向：

\[
\boxed{
硬件 instruction legality + dataflow compatibility
提供的结构约束，
可以显著缩小 candidate space，
同时高概率保留 near-optimal configurations。
}
\]

也就是说 Hexcute 的 constraint-based synthesis 提供了一个非常重要的提示：

\[
Legal\ structure
\]

可能比纯：

\[
random/search-rule enumeration
\]

更适合做第一阶段空间压缩。

但我们现在不能直接假定：

> constraint-guided 一定最好。

---

# 18. 一个很关键的反例可能是：合法性太弱，无法预测性能

假设有：

\[
10^5
\]

个合法 mapping。

constraint system 可以把：

\[
10^9
\]

个理论组合剪成：

\[
10^5
\]

这很好。

但如果其中：

\[
99.9\%
\]

性能都差，而高性能配置并不具有进一步的结构可分性，

那么 constraint synthesis 对：

\[
performance\ search
\]

帮助仍然有限。

所以 Q2 必须研究：

\[
\boxed{
Legal structure 与 Performance structure 的相关程度
}
\]

而不仅是“constraint 能不能找到合法程序”。

---

# 19. Q2 的可证伪预测

例如定义：

\[
NearOpt_\epsilon
=
\left\{
c:
T(c)
\le
(1+\epsilon)T^*
\right\}
\]

令：

\[
\epsilon=5\%
\]

一个好的 candidate generator 应做到：

\[
Recall_{5\%}
=
\frac{
|NearOpt_{5\%}\cap\mathcal C|
}{
|NearOpt_{5\%}|
}
\]

比较高，同时：

\[
\frac{|\mathcal C|}{|\mathcal U|}
\]

非常低。

如果发现：

各种 generator：

CUTLASS rules、

Triton heuristics、

TVM ScheduleRules、

Hexcute constraints

的：

\[
CoverageRegret
\]

都接近零，

那 Q2 也没有太大价值。

反过来，如果 framework-selected candidate space 本身已经排除了大量 near-optimal solutions：

那是非常强的结果。

---

# 20. Q3：Selection Regret

假设：

\[
c^*
\in\mathcal C
\]

但框架仍选：

\[
\hat c\neq c^*
\]

这是另一种完全不同的问题。

这一点以前经常和 Q2 混在一起。

---

# 21. Q3 的跨框架证据

### CUTLASS

Builder 用 static/expert rules 在支持集合里挑 schedule。citeturn887469search2

### Triton

compiler passes 用 heuristics，而 autotuning 可以对部分 meta-parameters 做真实 measurement。

### TVM

这是 measurement-heavy 一侧。

MetaSchedule：

\[
candidate
\rightarrow
cost\ model
\rightarrow
hardware\ measurement
\]

明确把真实性能作为 feedback。citeturn884265search8

### TIRx

dispatch 目前主要是：

\[
predicate
+
priority
\]

并非通用 performance measurement/search。citeturn884265search0

### Hexcute

解析模型主要基于：

\[
\#instructions\times instruction\ latency
\]

以及 copy/MMA overlap。

源码同时明确说 bank conflicts 尚未纳入这个 cost model，pipeline 假设充分 overlap，也没有 `cp_async_wait_group`/`mbarrier`。fileciteturn79file0L1-L7

---

# 22. Q3 不应该问“能不能更准确预测 latency”

这是一个容易走偏的地方。

我们并不一定需要：

\[
\hat T(c)=T(c)
\]

真正 selector 只需要：

\[
T(c_1)<T(c_2)
\]

时：

\[
\hat T(c_1)<\hat T(c_2)
\]

也就是：

\[
\boxed{\text{ranking correctness}}
\]

所以真正科研问题应该是：

> **What performance information is minimally sufficient to preserve the ranking of high-quality GPU configurations?**

中文：

\[
\boxed{
正确排序高性能 GPU configuration，
最少需要建模哪些信息？
}
\]

---

# 23. 为什么这个问题在 Hopper/Blackwell 特别值得看

Gluon 官方 warp-specialization 教程自己就强调：

不同 warp partition 数量和 register allocation 很难估计，经常需要 trial-and-error、profiling、autotuning；warp specialization 同时带来 synchronization、shared-memory 和 register-pressure 代价。citeturn887469search0

TIRx 同样把：

pipeline、

synchronization、

role assignment、

memory placement

留在显式 source 里，正是因为 frontier GPU feature 目前往往需要专家控制。citeturn414668search0turn414668search3

所以：

\[
TMA+
WGMMA/tcgen05+
mbarrier+
warp\ specialization+
TMEM
\]

使得配置性能越来越取决于：

\[
\text{resource interactions}
\]

而不仅仅是：

\[
instruction count
\]

---

# 24. 但 Q3 也有很强的反证可能

Hexcute artifact 专门包含：

> analytical cost model accuracy evaluation。citeturn839016search0

这点非常重要。

它说明不能从源码中的简化假设直接推导：

> Hexcute cost model 很差。

有可能虽然绝对模型简化，但在实际候选空间中：

\[
rank\ correlation
\]

已经足够高。

如果：

\[
Top1/Top5
\]

和：

\[
regret
\]

已经很好，那么增加 bank conflict、occupancy、barrier 等复杂项可能没有足够研究价值。

所以 Q3 必须先做 empirical falsification。

---

# 25. P3：Multi-consumer conflict 我现在不建议作为一级科学问题

但是它是 Q1 最重要的一个 mechanism。

观察不同框架可以发现非常一致的现象。

Triton 中同一个 tensor：

memory operation 喜欢 coalesced/blocked layout；

tensor-core op 喜欢 MMA-compatible layout；

因此存在 `ConvertLayout`，后面又有专门的 pass 去降低 conversion。fileciteturn67file0L1-L7

Hexcute 中一个 shared tensor 可以被多个 Copy operation 使用，因此 memory-layout inference 必须统一多个 consumer constraints。fileciteturn80file0L1-L7

`ResolveBankConflict` 甚至会把访问同一个 shared tensor 的 copy operations 分组，然后共同考虑 swizzle。fileciteturn81file0L1-L7

vLLM 也是类似问题，只不过 lifetime 大几个数量级：

多个 backend / connector 对同一个 persistent KV representation 提出 compatibility/preference。citeturn481206search4turn887469search4

所以：

\[
\boxed{
multi-consumer representation conflict
}
\]

是一个跨 abstraction level 反复出现的现象。

---

# 26. P3 更适合作为 Q1 的因果解释

可以定义 consumer preference divergence：

\[
D(X)
=
\mathrm{dispersion}
\left(
L^*_{C_1},
L^*_{C_2},
\ldots,L^*_{C_n}
\right)
\]

然后提出：

\[
\boxed{
BoundaryRegret
\uparrow
\quad\text{as}\quad
ConsumerPreferenceDivergence
\uparrow
}
\]

这比：

> 我们做一个跨算子 layout optimizer

更科学。

因为它提出一个可验证关系。

---

# 27. P4：Persistent representation 是否应该 workload-dependent？

这个问题我认为仍然值得独立保留，但它更适合作为第二条论文路线，而不是 Q1/Q2 的同级核心。

现在有一个非常有意思的现实矛盾。

SGLang 接受：

\[
Backend_{prefill}
\neq Backend_{decode}
\]

vLLM 当前甚至支持不同 KV-cache group 采用不同 backend。citeturn481206search0

但物理 KV layout：

\[
L_{KV}
\]

仍然在 engine 初始化阶段 resolve 一次，并成为 final configuration。citeturn481206search3turn887469search4

同时 backend 又会反过来影响 preferred block size；vLLM 当前还专门调整 attention/Mamba/hybrid page compatibility。citeturn481206search1

这说明 persistent state 已经受到：

\[
backend
+
model
+
hybrid\ state
+
quantization
\]

多个因素约束。

---

# 28. P4 真正的问题

不是简单：

> dynamic KV layout 会不会更快？

应该是：

\[
\boxed{
Persistent representation 的最优解，
在 realistic workload distribution 上是否形成稳定 regime？
}
\]

设：

\[
x=
(B,S_q,S_{kv},phase,reuse,transfer,\ldots)
\]

则：

\[
L^*(x)
=
\arg\min_L T(x,L)
\]

问题是：

\[
L^*(x)
\]

是不是几乎恒定？

还是形成少量 region：

```text
Prefill
        → Layout A

Short-context decode
        → Layout B

Long-context decode
        → Layout C

KV-transfer-heavy
        → Layout D
```

如果只是 2–4 个稳定 region，那么：

\[
Policy(x)\rightarrow L
\]

有现实意义。

---

# 29. P4 最大的反证：switching cost

persistent representation 和 register layout 最大的不同就在这里。

Register/shared layout 可以每个 kernel 改。

KV cache 是：

\[
GB\text{-scale persistent state}
\]

如果从：

\[
L_A\rightarrow L_B
\]

需要搬整个 KV：

\[
C_{switch}
\]

可能巨大。

所以真正目标不是：

\[
\min_L T(x,L)
\]

而是：

\[
\boxed{
\min_{\pi}
\sum_t
T(x_t,\pi(x_t))
+
C_{switch}
(
\pi(x_{t-1}),\pi(x_t)
)
}
\]

这才是科研问题。

---

# 30. P4 因此甚至可以和 Q1 分开形成另一篇论文

如果你更想贴近 KV layout / serving，我认为这一条其实很有潜力：

> **Workload-Aware Persistent Representation for LLM Serving**

核心不是 layout syntax。

而是研究：

\[
\boxed{
何时值得为不同 workload regime
维护不同 persistent representation？
}
\]

甚至可能得出反直觉结论：

> 动态切换完全不值得；但选择 workload-distribution-aware static representation 很重要。

这本身也是有价值的结果。

---

# 31. 现在形成完整的 Problem Evidence Matrix

| 问题 | Framework evidence | 已部分解决 | 最强潜在反证 | 最佳观察子图 | 当前判断 |
|---|---|---|---|---|---|
| Q1 Boundary Regret | SGLang phase split；vLLM persistent KV；Triton staged passes；TVM graph→task；TIRx explicit orchestration；Hexcute explicit pipeline | CUTLASS kernel joint construction；Hexcute L/T/I joint；Triton conversion cleanup | 大多数真实 workload \(R_B\approx0\) | Decode Attn、Prefill Attn、FFN | **最强** |
| Q2 Coverage Regret | CUTLASS builder subset；Triton heuristic/config set；TVM ScheduleRules；TIRx registered variants；Hexcute constraints | TVM broad search；Hexcute systematic synthesis；expert CUTLASS | 所有 generator 都覆盖 near-optimal set | GEMM、Attention、MoE | **最强** |
| Q3 Selection Regret | rules vs heuristics vs measurement vs analytic models | TVM measurement；Triton autotune；Hexcute calibrated model | 简单模型已有近完美 ranking | GEMM/Attention/warp-specialized kernels | **强，但需先验证问题存在** |
| P3 Consumer conflict | Triton conversions；Hexcute constraint unification；vLLM common KV layout | conversion removal、layout unification | compromise layout/convert 成本总很小 | Attention、FFN | **作为 Q1 mechanism** |
| P4 Persistent workload policy | phase-specific backend + model-static KV layout | backend-per-kind、connector preference、block adaptation | static layout 已近 oracle；switching 太贵 | Decode/KV transfer | **独立 LLM serving 方向** |

---

# 32. 我现在会把论文级研究问题进一步压缩成两个主问题

如果你的目标仍然是偏：

> GPU compiler / layout optimization

我现在最推荐：

### Research Question 1

\[
\boxed{
When does decomposing GPU optimization decisions across abstraction boundaries incur significant optimization regret?
}
\]

也就是：

> **什么时候局部/分阶段最优不再近似等于整体最优？**

---

### Research Question 2

\[
\boxed{
How aggressively can the legal GPU configuration space be structurally compressed while preserving near-optimal solutions?
}
\]

也就是：

> **怎样缩小 GPU mapping 搜索空间，同时不把真正好的解剪掉？**

这两问可以很好地统一：

CUTLASS、

Triton、

TVM、

TIRx、

Hexcute。

SGLang/vLLM 则提供更上层的 boundary case。

---

# 33. 如果希望论文更偏 LLM serving，则我会换主线

那就不应该主攻 Q2。

而是：

### RQ1

\[
\text{Kernel/runtime optimization boundary regret}
\]

### RQ2

\[
\text{Workload-dependent persistent representation}
\]

即：

\[
\boxed{
Runtime\ KV\ representation
\leftrightarrow
Kernel\ mapping
}
\]

的 joint optimization。

这与你项目名称 “KV layout” 实际上更加契合。

---

# 34. 还有一个重要结论：不应该一开始就宣称“现有框架性能不好”

现在更严谨的论文 observation 应该是：

> Existing systems make fundamentally different choices about where GPU optimization decisions reside. CUTLASS exposes expert composition; Triton distributes decisions across compiler transformations; TVM searches schedules generated by scheduling rules; TIRx separates explicit orchestration from primitive dispatch; Hexcute synthesizes layouts and instruction-compatible task mappings under fixed dataflow and pipelines; serving systems such as SGLang and vLLM additionally control runtime backends and persistent cache representations. citeturn887469search2turn884265search8turn414668search3turn839016search1turn887469search4

然后：

> It is not yet known how much performance is lost because of these different optimization boundaries, candidate-generation mechanisms, and performance-selection mechanisms.

这个才是一个严谨的 research gap。

而不是：

> Triton/TVM/Hexcute layout optimization 都不够好。

---

# 35. 我认为下一轮可以第一次开始设计“验证问题是否存在”的实验，但还不设计最终解决方案

你之前说：

> “我同意问题以后，再设计代码验证问题和思考解决方法。”

现在我认为已经到了可以做这一层的阶段，但仍然不要直接做新 optimizer。

下一轮最合理的是先针对 Q1/Q2/P4 设计三种**diagnostic experiments**：

1. **Boundary Regret Test**：固定同一个真实子图，只改变哪些 decision 允许联合变化，测 \(R_B\)。
2. **Coverage Test**：把 CUTLASS/Triton/TVM/Hexcute 各自实际能生成的 candidate set 与更大的 oracle set 对齐，测 \(R_C\)。
3. **Persistent-Regime Test**：对真实 decode/prefill/KV-transfer workload grid，画 \(L^*(x)\) 的 phase diagram，看静态 layout 是否真的存在明显 regret。

这三类实验的目的都不是证明我们的方案更快，而是首先回答：

\[
\boxed{\text{这个科研问题到底存不存在？}}
\]

如果 Q1/Q2/P4 中某一个在这个阶段就被实验否掉，我们就应该删掉它，而不是硬做 solution。

这一步之后，再选择一个确实有 substantial regret 的问题，才进入你说的“代码验证 + 思考解决方法”。

## 用户

请继续下一轮

## 模型回复

这一轮我会只设计“**问题存在性验证**”，不进入解决方案实现。每个问题都会给出：实验变量、oracle 定义、baseline、测量指标、判定阈值、反证条件，以及它最适合落在哪些真实 LLM/Diffusion 子图上。

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 3 个网站

## 模型思考摘要

已搜索 3 个网站

## 模型回复

我先补充一个实验方法上的关键约束：Q1、Q2、Q3 必须**解耦测量**。否则“框架慢”可能同时来自 boundary、candidate 没覆盖、selector 选错，最后无法说明科学问题。下一轮实验要刻意做到一次只测一种 regret。

## 模型思考摘要

设计选择实验
设计选择实验

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 2 个网站

## 模型思考摘要

构建实验层级
构建实验层级

## 模型思考摘要

已搜索 2 个网站

## 模型思考用时

思考了 59s

## 模型回复

这一轮我建议把目标限定为：**验证科研问题本身是否存在，以及存在于什么条件下**。不设计新优化器，不先假定我们的方法会更快。

经过前面的框架对比，现在可以把整个实验逻辑固定成下面这条链：

\[
\boxed{
\text{Legal implementation space}
\supset
\text{Boundary-restricted space}
\supset
\text{Framework candidate space}
\ni
\text{Framework-selected config}
}
\]

这样才能分别测 Boundary、Coverage、Selection 三种 regret，而不是最后只得到一个“框架 A 比框架 B 慢”的结果。

---

# 一、先固定统一实验模型，否则三个问题会互相污染

对同一个 workload \(w\)，定义四层空间：

\[
\mathcal U(w)
\]

是我们能够构造出来的、语义正确且硬件合法的较大 implementation space。

然后某种 optimization boundary \(B\) 限制它：

\[
\mathcal U_B(w)\subseteq\mathcal U(w)
\]

比如“KV layout 先固定，再优化 kernel”。

框架 \(F\) 的 candidate generator 实际只生成：

\[
\mathcal C_F(w)\subseteq\mathcal U_B(w)
\]

框架最后选：

\[
\hat c_F\in\mathcal C_F
\]

于是：

\[
R_B=
\frac{\min_{c\in\mathcal U_B}T(c)}
{\min_{c\in\mathcal U}T(c)}-1
\]

\[
R_C=
\frac{\min_{c\in\mathcal C_F}T(c)}
{\min_{c\in\mathcal U_B}T(c)}-1
\]

\[
R_S=
\frac{T(\hat c_F)}
{\min_{c\in\mathcal C_F}T(c)}-1
\]

以及：

\[
1+R_{\rm total}
=
(1+R_B)(1+R_C)(1+R_S)
\]

这一点以后所有实验都不要改。

---

# 二、还有一个非常重要的问题：我们几乎不可能得到真正的全局 oracle

完整空间：

\[
Layout\times Tile\times Mapping\times Instruction\times Pipeline\times\cdots
\]

太大。

所以实验中必须明确区分三种 oracle，而不能都叫 “optimal”。

| Oracle | 定义 | 能声称什么 |
|---|---|---|
| Exact oracle | 一个受控小空间全部穷举 | 可以说这个空间内的 true optimum |
| Strong empirical oracle | 多框架 union + 系统扩展 + 大规模测量 | 只能说 best-known configuration |
| Expert oracle | CUTLASS/FlashAttention/手写专家 kernel | 作为强性能参考，不能证明全局最优 |

CUTLASS 当前官方文档本身就明确说 `CollectiveBuilder` 并不覆盖 expert `CollectiveMma` API 的全部设计空间，这正说明“框架 candidate space”和“expressible space”不能混为一谈。citeturn956072search0

以后论文中也应该避免：

> global optimum

除非那个受控空间真的被我们完整枚举。

---

# 三、Q1：Boundary Regret 怎么验证？

Q1 是：

\[
\boxed{
什么时候将 GPU optimization 拆成多个阶段会产生显著 regret？
}
\]

不能只做：

> joint search 比 sequential search 快。

那只能证明一个 case。

真正应该验证的是“**coupling strength**”。

---

## 3.1 一个非常关键的新指标：Rank Reversal

假设两个 decision：

\[
A\in\{a_1,\ldots,a_m\}
\]

和：

\[
B\in\{b_1,\ldots,b_n\}
\]

例如：

\[
A=KV\ physical\ layout
\]

\[
B=kernel\ thread/tile\ configuration
\]

如果两个 decision 真正近似独立，那么最优 \(B\) 不应该因为 \(A\) 改变而频繁变化：

\[
\arg\min_B T(a_1,B)
\approx
\arg\min_B T(a_2,B)
\]

反之，如果出现大量：

\[
T(a_1,b_1)<T(a_1,b_2)
\]

但：

\[
T(a_2,b_1)>T(a_2,b_2)
\]

就是 **rank reversal**。

这个现象比单纯“joint 快 6%”有科学意义得多，因为它直接证明：

\[
\boxed{
A\text{ 和 }B\text{ 的 performance effect 不可简单分解}
}
\]

---

# 四、Q1 还可以定义一个 Interaction Strength

有性能矩阵：

\[
T(a_i,b_j)
\]

先去掉 A、B 的独立主效应：

\[
I_{ij}
=
T(a_i,b_j)
-
\bar T_{a_i}
-
\bar T_{b_j}
+
\bar T
\]

然后：

\[
\boxed{
InteractionStrength=
\frac{\mathrm{Var}(I)}
{\mathrm{Var}(T)}
}
\]

如果接近 0：

\[
T(A,B)\approx f(A)+g(B)
\]

说明两个 decision 近似可分解。

如果很大：

\[
T(A,B)
\]

主要受 interaction 控制。

这样 Q1 就从：

> “joint optimization 会不会好一点”

升级成：

> **哪些 GPU workload characteristics 会导致 optimization-variable interaction 增强？**

这才像科研问题。

---

# 五、Q1-A：第一个最重要的实验应该是 Decode Attention

这是我认为整个项目最重要的 diagnostic experiment。

固定语义：

\[
Q:[B,H_q,1,D]
\]

以及 paged KV。

第一组 decision：

\[
A=L_{KV}
\]

包括 persistent physical organization。

vLLM 当前实际上已经允许多种：

\[
LBHNC,\ LBNHC,\ LHBNC,\ BLHNC,\ BLNHC,\ BHLNC
\]

但为 whole model resolve 一个 layout，并综合所有 backend support/preference。citeturn238445search2

第二组 decision：

\[
B=
(
token\ tile,
head\ grouping,
vector\ width,
warps,
thread/value\ mapping,
load\ strategy,\ldots
)
\]

也就是 consumer kernel configuration。

真正测试：

\[
T(L_{KV},K)
\]

而不是只测试：

\[
T(L_{KV})
\]

---

# 六、Decode 实验最重要的三种比较

### 独立优化

先选 KV layout：

\[
L^*=
\arg\min_L C_{\rm runtime}(L)
\]

然后在其条件下找：

\[
K^*(L^*)
\]

这模拟“runtime ownership → kernel ownership”的边界。

### Joint oracle

直接：

\[
(L,K)^*
=
\arg\min_{L,K}T(L,K)
\]

### Reverse decomposition

先根据一个 kernel-oriented criterion 选：

\[
K^*
\]

然后适配 KV representation。

如果：

\[
A\rightarrow B
\]

和：

\[
B\rightarrow A
\]

产生不同性能，而且都距离 joint oracle 明显，

就是非常强的 boundary evidence。

---

# 七、Decode Attention 要覆盖哪些 workload region？

不能只做一个：

\[
B=1,S_{KV}=4096
\]

否则意义很弱。

至少要覆盖四个轴：

\[
B\in
\{1,4,16,32,64\}
\]

\[
S_{KV}\in
\{128,512,2K,8K,32K,128K\}
\]

\[
H_q/H_{kv}
\in
\{
1,\ 4,\ 8,\ 16
\}
\]

分别覆盖 MHA/GQA/MQA 类访问。

再做：

\[
D\in\{64,128,256\}
\]

以及 FP16/BF16/FP8 KV。

这样才能画：

\[
\boxed{
BoundaryRegret(B,S_{KV},GQA\ ratio,D,dtype)
}
\]

的 phase diagram。

---

# 八、Q1 的第二个实验：Prefill Attention

这里研究的 boundary 不是 persistent KV。

而是：

\[
QK^T
\rightarrow
Softmax
\rightarrow
PV
\]

中间 representation 的 consumer conflict。

概念上有：

\[
L_{QK}
\]

适合 QK MMA，

\[
L_{softmax}
\]

适合 reduction，

\[
L_{PV}
\]

适合下一次 MMA。

真正要研究：

\[
T(
L_{QK},
L_{softmax},
L_{PV},
Pipeline
)
\]

Triton 当前就明确存在针对 dot 的 accelerator layout、针对 memory coalescing 的 layout，以及后续 `RemoveLayoutConversions` 去协调两者；这是已经存在局部 layout preference conflict 的源码证据。fileciteturn66file0L1-L7 fileciteturn67file0L1-L7

---

# 九、Prefill 不应该只测“有没有 ConvertLayout”

我们真正关心的是三个东西：

\[
\boxed{
conversion\ cost
}
\]

\[
\boxed{
consumer\ local\ speedup
}
\]

\[
\boxed{
whole-chain\ speedup/regret
}
\]

因为某个 ConvertLayout 即使很贵，也可能值得：

\[
+8\%\ conversion
-20\%\ MMA
\]

最后还是赚。

所以不能把：

> layout conversion 数量

本身当问题。

应该测：

\[
\Delta T_{\rm conversion}
\]

相对于：

\[
\Delta T_{\rm consumer}
\]

。

---

# 十、Q1 第三个实验：SwiGLU/GEGLU

真实子图：

\[
G=XW_g
\]

\[
U=XW_u
\]

\[
H=SiLU(G)\odot U
\]

\[
Y=HW_d
\]

这里研究：

\[
L_{G/U\ output}
\]

\[
L_{elementwise}
\]

\[
L_{down\ GEMM\ input}
\]

是否冲突。

这个实验非常适合验证 Q1 是否只存在于 attention。

如果 Decode/Attention 有很大 regret，

但 FFN：

\[
R_B\approx0
\]

反而是很有价值的结果。

它意味着 boundary coupling 不是“GPU 优化普遍现象”，而是跟：

\[
memory\ lifetime
\]

\[
consumer\ diversity
\]

\[
reduction/dataflow
\]

相关。

---

# 十一、Q1 必须加入 Negative Control：普通 GEMM

这一点我很坚持。

普通：

\[
C=AB
\]

应该作为 negative control。

原因是成熟 CUTLASS/Triton 已经非常擅长 GEMM。

如果我们的所谓 boundary theory 得到：

\[
R_B=15\%
\]

连标准 GEMM 都很大，

第一反应应该是：

> 实验 oracle/baseline 定义有问题。

理想结果反而应该类似：

\[
R_B^{GEMM}\approx0
\]

但：

\[
R_B^{DecodeAttention}\gg0
\]

这才支持“存在条件性 boundary coupling”。

---

# 十二、Q1 怎么判定“值得继续”？

我不建议只看 statistical significance。

GPU benchmark 很容易把 0.5% 测成 statistically significant。

应该同时要求：

\[
95\%\ CI\text{ 不跨 }0
\]

和 practical effect size。

可以预先设一个 research gate，比如：

\[
R_B>5\%
\]

出现在至少两个 realistic workload regimes，

并且至少两类子图或者两种 GPU 上能复现。

这个 5% 不是理论常数，后续可以根据 measurement noise 调整；重要的是**提前定义 go/no-go criterion**，不要看到结果后再改标准。

---

# 十三、Q2：Coverage Regret 怎么验证？

Q2 的核心不是比较：

> Hexcute latency vs Triton latency。

而是：

\[
\boxed{
Hexcute/Triton/TVM/CUTLASS 各自有没有把优质 candidate 放进自己的搜索空间？
}
\]

---

# 十四、第一步必须建立统一 Configuration Schema

不同框架 candidate 不能直接拿源代码文本比较。

要归一化成一个共同 descriptor：

\[
c=
(
Tile_M,
Tile_N,
Tile_K,
CTA/cluster,
warps,
stages,
thread\ mapping,
smem\ layout,
copy\ family,
MMA\ family,
pipeline\ family,\ldots
)
\]

对 Attention 再增加：

\[
Block_M,
Block_N,
Q/K/V\ layout,
softmax\ mapping,\ldots
\]

对 decode 增加：

\[
KV\ layout,
page,
head/token\ mapping
\]

。

以后每一个 framework candidate 编译以后都映射到这个 canonical representation。

---

# 十五、然后分别提取框架实际 candidate set

CUTLASS：

\[
\mathcal C_{\rm CUTLASS-builder}
\]

与更大的 expert/CuTe configuration family 区分开。

这非常重要，因为 NVIDIA 当前文档自己明确说明 CollectiveBuilder 并不覆盖 expert API 全设计空间。citeturn956072search0

Triton：

需要区分：

\[
\mathcal C_{\rm Triton-user}
\]

即 `triton.Config` 中实际提供的 meta-configs，

和：

\[
Compiler(c)
\]

最后产生的 layout/instruction configuration。

官方 `Config` 明确暴露：

`num_warps`、`num_stages`、`num_ctas`、`maxnreg` 加用户 meta-parameters。citeturn883889search1turn238445search6

---

# 十六、TVM 的 candidate set 更容易形式化

MetaSchedule 的：

\[
SpaceGenerator
\]

先产生 design spaces；

SearchStrategy 再从这里产生：

\[
MeasureCandidate
\]

；

Runner 真实测量。

官方当前架构就是：

\[
SpaceGenerator
\rightarrow
SearchStrategy
\rightarrow
CostModel
\rightarrow
Builder/Runner
\]

。citeturn956072search2turn956072search1

所以可以分别测：

\[
\mathcal C_{\rm design-space}
\]

和：

\[
\mathcal C_{\rm actually-measured}
\]

。

这是非常有价值的区别。

---

# 十七、TIRx 的 candidate set也非常清晰

当前 TIRx primitive dispatch 是：

\[
(Op,target)
\rightarrow
\{
variant_1,\ldots,variant_n
\}
\]

每个 variant 带：

\[
priority+predicates+implementation
\]

然后按 priority 排序，选第一个 predicate 成功的实现。citeturn883889search3turn883889search4

所以：

\[
\mathcal C_{\rm TIRx}
=
\text{all registered legal variants}
\]

而：

\[
\hat c_{\rm TIRx}
\]

则是 priority/predicate 最终选择的那个。

于是 Coverage 与 Selection 天然就能拆开。

---

# 十八、Hexcute 的 candidate set是最有意思的一组

Hexcute 的 inference 源码直接说：

copy operation 可能存在多个可执行 instruction；

理论上可以 DFS 找出所有 valid variants，再通过 cost model 选，也可以用 heuristic/beam search prune。fileciteturn80file0L1-L7

所以理想实验应该至少区分：

\[
\mathcal C_{\rm legal}
\]

constraint solver 能产生的所有合法候选，

\[
\mathcal C_{\rm pruned}
\]

实际 pruning 后保留的候选，

以及：

\[
\hat c_{\rm cost}
\]

cost model 最终选中的 candidate。

这样一个系统内部就可以直接分解：

\[
CoverageRegret
\]

和：

\[
SelectionRegret
\]

。

---

# 十九、Q2 最重要的三个指标

不能只报告 search space size。

第一：

\[
\boxed{
CompressionRatio
=
\frac{|\mathcal C_F|}
{|\mathcal U|}
}
\]

越小越说明 candidate generation 剪枝强。

第二：

\[
\boxed{
CoverageRegret
=
\frac{\min_{\mathcal C_F}T}
{\min_{\mathcal U}T}-1
}
\]

直接告诉我们剪掉的东西有没有价值。

第三：

设：

\[
NearOpt_\epsilon
=
\{
c:T(c)\le(1+\epsilon)T^*
\}
\]

那么：

\[
\boxed{
NearOptRecall_\epsilon
=
\frac{
|NearOpt_\epsilon\cap\mathcal C_F|
}{
|NearOpt_\epsilon|
}
}
\]

推荐至少：

\[
\epsilon=1\%,5\%,10\%
\]

分别报告。

---

# 二十、为什么 NearOpt Recall 很重要？

假设 framework：

\[
CoverageRegret=0
\]

因为碰巧留下了唯一一个 oracle config。

但其他所有：

\[
\le5\%
\]

的好配置都被删掉了。

这种 candidate generator 非常脆弱。

换 GPU、shape、compiler version 后很可能失败。

所以：

\[
CoverageRegret
\]

回答：

> 有没有留下最好的？

而：

\[
NearOptRecall
\]

回答：

> 有没有保留高性能区域？

后者对科学结论很重要。

---

# 二十一、Q2 应该先在哪些 workload 上做？

这里不要一开始上完整 LLM。

最合理的是：

| Workload | 原因 |
|---|---|
| Dense GEMM | 搜索空间可控，成熟 baseline，negative control |
| Fused attention | 多 layout/instruction choice |
| FP8/mixed-type GEMM | instruction/layout constraint 更强 |
| Grouped/MoE GEMM | irregular shape + heterogeneous type |
| Selective scan | 非 GEMM stress case |

Hexcute artifact 本身已经覆盖 GEMM、Attention、FP8、warp specialization、mixed-type MoE、Mamba scan，并专门有 cost-model accuracy evaluation，因此它的 artifact 可以直接作为我们设计 Q2/Q3 diagnostics 的参考数据来源。citeturn883889search0

---

# 二十二、Q2 的科研命题可以进一步收敛

真正想验证：

\[
\boxed{
Legal structure 对 high-performance region 是否具有 predictive power？
}
\]

也就是说，一个 constraint-based generator 如果把：

\[
10^8
\]

个理论组合剪成：

\[
10^4
\]

个合法 configuration，

它是否同时保留：

\[
90\%+
\]

的 near-optimal region？

如果是：

\[
\text{constraint structure}
\]

就是非常有价值的 search-space prior。

如果不是：

> legality 本身对 performance search 帮助有限。

两种结果都值得知道。

---

# 二十三、Q3：Selection Regret 怎么测？

Q3 必须在**框架自己的 candidate set内部**测。

不能：

> TVM cost model 在 Hexcute candidate 上怎么排序？

这种比较很容易不公平。

对于每个 framework：

\[
\mathcal C_F
\]

先尽可能实际测完或测一个充分大的固定 sample。

得到 measured ranking：

\[
r_{\rm true}
\]

然后比较 framework selector：

\[
r_F
\]

。

---

# 二十四、Q3 建议四个主指标

\[
\boxed{
SelectionRegret
=
\frac{T(\hat c_F)}
{\min_{\mathcal C_F}T}-1
}
\]

最重要。

再报告：

\[
Kendall\ \tau
\]

测整体 pairwise ranking。

\[
TopKRecall
\]

测模型有没有把真正优质 candidate 放进前 K。

以及：

\[
PairwiseAccuracy
=
P[
\operatorname{sign}(\hat T_i-\hat T_j)
=
\operatorname{sign}(T_i-T_j)
]
\]

。

我甚至认为 absolute latency prediction error：

\[
|\hat T-T|
\]

应该放次要位置。

因为 compiler selector 最终只需要挑对。

---

# 二十五、Q3 的实验对象非常自然

Hexcute：

analytic cost model vs measured candidates。

其当前源码明确说：

\[
cost=\#instructions\times latency
\]

并建模 copy/MMA overlap；同时说明 bank conflict 尚未进入模型、pipeline 假设 full overlap、没有考虑 `cp_async_wait_group` 和 `mbarrier`。fileciteturn79file0L1-L7

但不能据此预设它一定失败，因为 artifact 本身已经有 cost-model accuracy test。citeturn883889search0

TVM：

XGB/MLP/random model 与最终 measured candidates。

MetaSchedule 当前默认就是 cost-model-guided evolutionary search + hardware runner。citeturn956072search2

TIRx：

priority/predicate-selected variant vs 同一 primitive 所有 legal registered variants 的 measured best。citeturn883889search3

CUTLASS：

Auto schedule/builder choice vs 手动指定 supported schedules。

Triton：

heuristic configuration vs exhaustive supplied `triton.Config`；其官方 attention example 甚至会先 prune invalid configs，再 autotune剩余集合，非常适合研究 pruning/selection 的分界。citeturn238445search9

---

# 二十六、Q3 最值得研究的不是“平均 ranking accuracy”

要特别研究：

\[
\boxed{
mis-ranking 什么时候发生？
}
\]

例如把误选 case 按下面这些 feature 分组：

\[
SMEM/CTA
\]

\[
register/thread
\]

\[
occupancy
\]

\[
pipeline\ depth
\]

\[
bank\ conflict
\]

\[
TMA/WGMMA
\]

\[
warp\ specialization
\]

\[
memory/compute\ ratio
\]

然后问：

\[
P(misrank\mid feature)
\]

如果发现：

> register-pressure 超过某区间以后 analytic model 大量误排，

这是科学结果。

而不只是：

> Kendall tau = 0.72。

---

# 二十七、P4：Persistent Representation Regime Test

如果项目仍然重点叫 “KV layout”，这个实验应该保留。

vLLM 当前明确：

> whole model resolve one KV-cache layout。

并考虑 backend-supported layouts 和 connector preference。citeturn238445search2

同时 NIXL 当前实现里已经能看到 NHD/HND 与 transfer path 的具体差异，例如部分情况下 host transfer buffer 会改用 HND。citeturn956072search3

这证明 persistent KV representation 确实面对至少两个 consumer：

\[
Attention
\]

和：

\[
Transfer
\]

。

但仍然不能证明 dynamic layout 值得做。

---

# 二十八、P4 应该先构建一个 cost matrix

令 workload state：

\[
x=
(
phase,
B,
S_q,
S_{kv},
reuse,
transfer,
GQA,
dtype,
page
)
\]

对每个 layout：

\[
L
\]

测：

\[
C(x,L)
\]

包括：

\[
T_{\rm attention}
+
T_{\rm insert}
+
T_{\rm transfer}
+
T_{\rm management}
\]

如果只测 attention：

\[
C=T_{\rm attention}
\]

其实没有研究 persistent representation。

---

# 二十九、然后画真正的 Layout Phase Diagram

对于每个 \(x\)：

\[
L^*(x)=\arg\min_L C(x,L)
\]

画：

```text id="eylfy5"
                         KV length
                    short       long

Batch small        Layout A    Layout B

Batch large        Layout C    Layout B
```

再加：

```text id="k1rtun"
transfer-heavy → Layout D
```

我们真正想知道：

\[
\boxed{
L^*(x) 是否形成少量稳定区域？
}
\]

而不是单纯有几个 shape 的最佳 layout 不同。

---

# 三十、P4 要比较三个 oracle

Static oracle：

\[
L^*_{\rm static}
=
\arg\min_L
E_{x\sim\mathcal W}C(x,L)
\]

这已经比当前简单 preference 更公平。

Zero-cost dynamic oracle：

\[
\pi^*(x)=L^*(x)
\]

这个只是理论上界。

最后才是 realistic policy：

\[
\pi^*
=
\arg\min_\pi
\sum_t
C(x_t,\pi(x_t))
+
C_{switch}
(
L_{t-1},L_t
)
\]

。

---

# 三十一、P4 最重要的结果可能是“动态切换根本不值得”

例如发现：

\[
DynamicOracle
\]

理论上快 12%，

但重新排列一个 30 GB KV cache：

\[
C_{switch}
\]

需要的时间根本收不回来。

那合理结论可能是：

> 不应该做 dynamic KV transformation。

但仍可能说明：

\[
\boxed{
workload-distribution-aware static layout
}
\]

优于 model-only static preference。

这也是有价值的科研结果。

---

# 三十二、为了避免实验规模失控，我建议模型/子图覆盖这样设计

| Workload family | 代表子图 | Q1 | Q2 | Q3 | P4 |
|---|---|---:|---:|---:|---:|
| Dense LLM | GEMM | control | ★★★★★ | ★★★★ | — |
| GQA LLM | Decode attention | ★★★★★ | ★★★★ | ★★★★ | ★★★★★ |
| GQA LLM | Prefill attention | ★★★★★ | ★★★★★ | ★★★★★ | ★★★ |
| Dense LLM | SwiGLU | ★★★★ | ★★★★ | ★★★ | — |
| MoE LLM | grouped/mixed GEMM | ★★★★ | ★★★★★ | ★★★★★ | — |
| MLA LLM | decode attention | ★★★★★ | ★★★★ | ★★★★ | ★★★★★ |
| Diffusion/DiT | self/cross attention | ★★★★ | ★★★★ | ★★★★ | — |
| Diffusion | GEGLU/FFN | ★★★★ | ★★★★ | ★★★ | — |
| SSM | selective scan | ★★★ | ★★★★★ | ★★★★ | state-layout 可扩展 |

这里没有要求所有 framework 都跑所有 workload。

这是另一个必须坚持的原则。

---

# 三十三、不同 framework 在 diagnostics 中应该承担不同角色

不要为了表格齐全而强行：

> SGLang 跑 Diffusion GEGLU。

更合理的是：

SGLang/vLLM：

主要用于：

\[
Runtime/Persistent\ Boundary
\]

也就是 Q1/P4。

CUTLASS：

主要作为：

\[
Expressiveness/Expert\ Oracle
\]

和 Q2 的 expert-space reference。

Triton：

用于：

\[
Heuristic/compiler-pass
+
empirical-config-search
\]

三个问题都很重要。

TVM：

重点用于：

\[
CandidateGeneration
+
Search/Selection
\]

也就是 Q2/Q3。

TIRx：

用于：

\[
Explicit-layout
+
predicate/priority\ dispatch
\]

Q2/Q3。

Hexcute：

用于：

\[
ConstraintCandidateGeneration
+
AnalyticSelection
\]

Q2/Q3，同时也是 Q1 中 layout/instruction 与 pipeline boundary 的重要对象。

---

# 三十四、这一轮之后，我建议设明确的 Go/No-Go Gate

这样不会出现“先认定问题，再想办法证明”。

| 问题 | 继续研究的最低证据 | 否则怎么办 |
|---|---|---|
| Q1 Boundary | 在多个 realistic regimes 出现稳定 rank reversal + practical \(R_B\)，并跨至少两类 workload/硬件复现 | 放弃“cross-boundary”主线 |
| Q2 Coverage | 至少一种主流 generator 有显著 coverage regret，或 near-optimal recall 随高压缩明显下降 | 不做新的 candidate generator |
| Q3 Selection | candidate 已包含好配置，但 native selector 仍存在稳定 selection regret，并可与硬件机制相关联 | 不做新 cost model |
| P4 Persistent | 最佳 layout 形成多个稳定 workload regimes，static policy 有实质 regret，而且 gain 能覆盖 switching/management cost | 不做 dynamic/policy KV layout |

我个人会把“实质 regret”的预注册阈值暂定为 kernel-level 约 **5%**，但最终应该根据实际 measurement noise 和 end-to-end contribution 重新定，不能把 5% 当自然常数。

---

# 三十五、这四个 diagnostic 中，我建议实际执行顺序也不是 Q1→Q2→Q3→P4

我现在更建议：

\[
\boxed{
Q2\rightarrow Q3\rightarrow Q1\rightarrow P4
}
\]

原因很重要。

首先要知道：

\[
\mathcal C
\]

里面到底有哪些 candidate。

否则 Q1 里你发现：

> sequential optimization 比 joint 慢 10%

可能只是 sequential 那一侧 candidate generator 较差，

实际上测到的是：

\[
CoverageRegret
\]

不是 BoundaryRegret。

所以先建立 candidate space。

然后测：

\[
SelectionRegret
\]

确认 selector 的影响。

最后才能把这两个因素控制住，比较：

\[
BoundaryRegret
\]

。

P4 是 runtime-level extension，放在最后。

---

# 三十六、因此真正的实验依赖关系应该是

```text id="o1p08f"
                 Build canonical configuration space
                              │
                              ▼
                 Q2: Candidate Coverage
                              │
                  Is good config present?
                       /             \
                     NO               YES
                     │                 │
             Coverage problem          ▼
                                Q3: Selection
                                      │
                             Did selector pick it?
                               /             \
                             NO               YES
                             │                 │
                     Selection problem        ▼
                                      Q1: Boundary
                                              │
                                    Does changing ownership/
                                    optimization ordering
                                    change the optimum?
                                              │
                                              ▼
                                  P4: Runtime/Persistent
                                      workload regimes
```

我认为这是目前为止我们得到的最重要的实验设计框架。

---

# 三十七、下一轮才应该开始把它变成具体 benchmark specification

但仍然不需要立刻写 optimizer。

下一轮应该选择一个最适合建立整个基础设施的 workload。我现在建议首先用：

\[
\boxed{
Dense\ GEMM + Prefill\ Attention
}
\]

建立 Q2/Q3 的 canonical candidate representation，因为这两个 workload：

一方面 CUTLASS/Triton/TVM/Hexcute 都有成熟实现；

另一方面 Hexcute artifact 已经覆盖 GEMM、Attention、FP8 和 cost-model evaluation，可以直接做复现基线。citeturn883889search0

然后再进入：

\[
\boxed{Decode\ Attention}
\]

做真正关键的 Q1/P4。

下一轮我可以具体到：**GEMM / Prefill Attention 各自到底枚举哪些 optimization dimensions、每个框架如何导出 candidate、怎样定义 canonical configuration fingerprint、哪些 shape 从真实现代 LLM 中取、每一步保存什么 IR/PTX/SASS/profile/result 文件，以及最终如何计算 Coverage/Selection Regret**。这一阶段仍然只是验证基础设施设计，不会开始提出我们的优化算法。

## 用户

请继续下一轮

## 模型回复

这一轮我会把前面的研究框架第一次落成“可执行的 benchmark specification”，但仍不设计新算法。重点是两条基础线：**Dense GEMM 用来校准 candidate-space/selection 方法，Prefill Attention 用来暴露真正的多 consumer/layout coupling**；随后它们的基础设施才能可靠迁移到 Decode/KV。

## 模型工具调用参数

{"paths":["GitHub"],"query":"search"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

制定现代LLM基准规格

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 融合注意力配置示例

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Hexcute 基准仓库目录

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 scripts 目录内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 README 并运行 H100 基准测试

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 run_h100 基准测试脚本路径

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查找 Hidet 基准测试源码提交

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 GitHub 主分支文件内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 GEMM 基准测试源码

## 模型思考摘要

制定并运行现代LLM及H100基准测试流程

## 模型思考：工具活动

正在搜索 Qwen3 8B config num_attention_heads num_key_value_heads head_dim hidden_size intermediate_size

## 模型思考：工具活动

正在搜索 huggingface.co

## 模型思考摘要

规划矩阵测试
规划矩阵测试

## 模型思考摘要

已搜索 1 个网站

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析 Qwen3 GEMM 与 GQA 注意力基准

## 模型思考摘要

分析了 Qwen3 GEMM 与 GQA 注意力基准

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 3 个网站

## 模型思考摘要

完善实验分层
完善了实验分层方案

## 模型思考摘要

已搜索 3 个网站

## 模型回复

一个重要细节已经确认：Hexcute 官方 H100 artifact 的 GEMM 搜索空间显式扫 `cluster_m/n`、`TiledMma`、pipeline 深度等；Attention artifact 则在 `TiledMma pair`、warp-group 数、N tile、cluster 等维度搜索，但当前 benchmark 代码里仍直接 `assert num_heads == num_heads_k`，也就是 stock attention 实验是 MHA，不是 Qwen3 这类 GQA。这个差异必须在 benchmark 设计里显式处理，不能把“语义不支持”误算成 coverage regret。 

## 模型思考用时

思考了 1m 42s

## 模型回复

可以。这一轮把前面的 Q2→Q3→Q1 方法真正落成 benchmark specification。先做 **Dense GEMM + Prefill Attention**，目标不是提出新优化方法，而是把不同框架的 candidate 空间、最终选择和实际 codegen 放进同一个可比较坐标系。

这里先做一个很重要的修正：

\[
\boxed{\text{“框架不支持某种语义”不能算 Coverage Regret}}
\]

例如 Hexcute 官方 H100 Attention artifact 当前运行的是 `B=1,H_q=H_k=32,D=128,S_q=S_k=16384`，源码中直接要求 `num_heads == num_heads_k`；因此 stock artifact 是 MHA，不是 Qwen3-8B 的 GQA。若 Qwen3 的 \(H_q=32,H_{kv}=8\) 不能直接运行，我们应记录为 `semantic_unsupported`，而不是说 “Hexcute candidate generator 漏掉了 GQA 最优 layout”。fileciteturn88file0L1-L13 fileciteturn90file0L1-L2

---

# 1. 整套 benchmark 最好分成三层

我建议每个 candidate 都区分：

\[
\boxed{
Problem
\rightarrow
Decision
\rightarrow
Realized\ Code
}
\]

不要只保存“config”。

因为同样：

```text
BLOCK_M=128
BLOCK_N=128
num_warps=8
```

在 Triton、Hexcute、CUTLASS 中最后可能对应完全不同的：

- thread/value mapping；
- SMEM layout；
- copy instruction；
- MMA；
- pipeline；
- register allocation。

所以至少要有三个 ID。

| ID | 表示什么 |
|---|---|
| `problem_id` | 数学语义完全相同的问题 |
| `decision_id` | 框架/用户明确做出的优化选择 |
| `binary_id` | 实际生成代码的 fingerprint |

最终 measurement 应该绑定：

\[
(problem\_id,decision\_id,binary\_id,environment)
\]

而不是只绑定 source config。

---

# 2. 先定义统一的 `problem.json`

Dense GEMM：

```text
op = GEMM
M
N
K
A_dtype
B_dtype
C_dtype
acc_dtype

A_logical_layout
B_logical_layout
C_logical_layout

alpha
beta
```

第一阶段强烈建议限定：

\[
C=AB
\]

不做 bias、activation、split-K 等额外语义。

原因是 Q2 首先要研究：

> candidate generation 本身。

如果把 epilogue/fusion 一开始也放进来，空间很快无法解释。

---

# 3. GEMM 的 canonical decision space 应该至少包含这些维度

| 类别 | Canonical field | 说明 |
|---|---|---|
| CTA tile | `tile_m,n,k` | CTA/workgroup 处理区域 |
| Cluster | `cluster_m,n,k` | Hopper/Blackwell CTA cluster |
| Grid | `raster_order`, `tile_swizzle` | CTA 调度/空间 locality |
| Threads | `num_threads`, `num_warps` | CTA execution resources |
| Roles | `producer_wg`, `consumer_wg` | warp specialization |
| Registers | `producer_regs`, `consumer_regs` | role-specific allocation |
| GMEM→SMEM | `copy_a`, `copy_b` | TMA/cp.async/vector LD 等 |
| GMEM vector | `vec_a`, `vec_b` | copy width/alignment |
| SMEM A | `smem_layout_a`, `swizzle_a` | physical shared layout |
| SMEM B | `smem_layout_b`, `swizzle_b` | 同上 |
| Pipeline | `stages`, `pipeline_family` | buffer 数和 pipeline |
| MMA | `mma_opcode`, `inst_m,n,k` | HMMA/WGMMA/tcgen05 等 |
| Mapping | `mma_thread_value_layout` | thread/value → tile |
| Epilogue | `epilogue_tile`, `store_family` | accumulator→output |
| Resource | `smem_bytes`, `regs`, `occupancy` | 这是 realized attribute |

但这里还要给每个字段增加：

```text
owner
controllable
observable
```

例如：

### CUTLASS

```text
tile_m:
owner = programmer/builder
controllable = yes
```

### Triton classic

```text
mma_thread_value_layout:
owner = compiler
controllable = usually no
observable = yes
```

### Hexcute

```text
mma_thread_value_layout:
owner = compiler/constraint solver
controllable = indirectly
observable = yes
```

这是后面比较 candidate coverage 的前提。

---

# 4. 所以 `decision.json` 不应该只是值

建议每个字段形式类似：

```text
tile_m:
    value: 128
    owner: user
    source: autotune_config
    controllable: true
```

而：

```text
smem_layout_a:
    value: ...
    owner: compiler
    source: inferred
    controllable: false
```

这会让我们第一次可以量化：

\[
\boxed{\text{Decision Ownership}}
\]

而不是笼统说：

> Triton 自动一点，CUTLASS 手工一点。

---

# 5. `realized.json` 更关键

最终 compiler 产生以后，再从 IR/PTX/SASS/profile 提取：

```text
mma_instruction_family
mma_instruction_count

global_load_widths
global_store_widths

tma_count
cp_async_count
ldmatrix_count

shared_load/store_count

shared_layout
bank_conflict_metric

registers_per_thread
smem_per_cta

threads_per_cta
occupancy

barrier_count
mbarrier_count

pipeline_depth_observed
```

于是我们可以区分：

\[
\text{declared decision}
\]

和：

\[
\text{realized hardware implementation}
\]

这对 Triton 尤其重要。

---

# 6. Hexcute GEMM 已经提供了一个很好的真实 candidate-space 样例

当前 artifact 的 H100 script运行：

```bash
python3 benchmark_warp_specialized_gemm_non_persistent.py \
    --m 4096 \
    --n 4096 \
    --k 4096 \
    --search-space 2
```

fileciteturn88file0L1-L13

而源码里的 `@tune.space` 明确扫：

\[
cluster_m\in\{1,2,4,8,16\}
\]

\[
cluster_n\in\{1,2,4,8,16\}
\]

多种 `TiledMma`，

\[
k\_pipe\_max\in\{4,5,6,-1\}
\]

和：

\[
BK=64
\]

。fileciteturn91file0L1-L2

`TiledMma` 本身又枚举不同：

\[
warpgroup_m\in\{1,2\}
\]

以及：

\[
N_{MMA}
\in
\{32,64,96,128,144,\ldots,256\}
\]

。fileciteturn91file0L1-L2

这已经不是一个小 search space。

---

# 7. Hexcute GEMM 的一个优势：我们能够拿到“所有 candidate IR”

源码不是只 build 最佳 kernel。

它调用：

```text
tune.extract_ir_modules(...)
```

得到所有 candidate IR module，然后逐个编译；每个 module 还根据 IR string hash 建独立 cache directory。fileciteturn91file0L1-L2

这对我们的 Q2/Q3 非常理想。

我们应该修改 benchmark harness，使其对每个 candidate 输出：

```text
candidate_id
raw_ir
inferred layouts
selected instructions
cost_model_score
compile_status
measured_latency
```

而不是只保留最终性能图。

---

# 8. CUTLASS 的 candidate extraction 方法又完全不一样

CUTLASS Profiler 可以把**已经实例化进 binary 的 kernel family**基本完整跑一遍。

官方文档现在支持：

```bash
cutlass_profiler \
  --kernels=*gemm* \
  --enable-kernel-performance-search \
  --sort-results-flops-per-sec
```

也支持固定 shape 搜索：

```bash
cutlass_profiler \
  --kernels=*gemm* \
  --enable-best-kernel-for-fixed-shape \
  --m=... --n=... --k=...
```

并可保存 CSV。citeturn318331search1turn318331search0

Profiler 输出本身已经包含：

```text
cta_m
cta_n
cta_k

stages

warps_m
warps_n
warps_k

inst_m
inst_n
inst_k
```

等信息。citeturn318331search0turn318331search3

---

# 9. 但不要把 CUTLASS profiler 结果称为整个 CuTe 空间

要叫：

\[
\boxed{
\mathcal C_{\text{CUTLASS-built}}
}
\]

因为 profiler 只能搜索：

> 编译进 profiler binary 的 kernel。

NVIDIA 文档自己也提醒，全部实例化可能达到数万个 kernel，因此推荐只构建 subset。citeturn318331search0

所以以后至少区分：

\[
\mathcal C_{\rm CUTLASS-built}
\]

和理论上更大的：

\[
\mathcal U_{\rm CuTe-expressible}
\]

这正是 Coverage 研究里很重要的边界。

---

# 10. Triton 必须同时保存 autotune config 和 compiler-realized layout

对于 GEMM，可以自己定义明确的 meta-space，例如：

\[
BLOCK_M\in\{32,64,128,256\}
\]

\[
BLOCK_N\in\{32,64,128,256\}
\]

\[
BLOCK_K\in\{32,64,128\}
\]

\[
num\_warps\in\{4,8\}
\]

\[
num\_stages\in\{2,3,4,5\}
\]

然后**不要只调用 autotune 取 winner**。

必须逐 config 强制运行一次，得到：

\[
\mathcal C_{\rm Triton-meta}
\]

。

---

# 11. 但 Triton 的真实 candidate 比这个 meta-space 更复杂

因为：

```text
BLOCK_M/N/K
num_warps
num_stages
```

并没有完全决定：

\[
L_{register}
\]

\[
L_{smem}
\]

\[
I_{MMA}
\]

Compiler passes 还会重新决定这些。

所以每一个 Triton config 编译时都应该打开：

```bash
export TRITON_ALWAYS_COMPILE=1
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=<candidate-dir>
```

当前 Triton README 明确支持这样 dump 每一阶段 IR 和最终 PTX/AMDGCN。citeturn866526search0turn866526search4

应保存：

```text
ttir
ttgir
llir
ptx
cubin
```

如果使用 compiled kernel object，也可以直接访问：

```text
asm["ttir"]
asm["ttgir"]
asm["llir"]
asm["ptx"]
```

。citeturn866526search2

---

# 12. TVM 则必须区分“设计空间”和“真正测过的 candidate”

这又是另一种区别。

MetaSchedule 的固定 work directory 会保存：

```text
database_workload.json
database_tuning_record.json
```

其中 tuning record 包括：

\[
schedule\ trace + measured\ runtimes
\]

。citeturn318331search5turn318331search6

所以至少有两个集合：

\[
\mathcal C_{\rm TVM-design}
\]

SpaceGenerator 可以产生的设计空间，

和：

\[
\mathcal C_{\rm TVM-measured}
\]

SearchStrategy 真正送去 Runner 的 candidate。

这两个绝对不能混为一谈。

---

# 13. 对 TVM 我建议每一个 measured candidate 保存

```text
workload.json
schedule_trace.json
scheduled_tir.py
scheduled_tir.json
target.json

predicted_cost.json
run_secs.json
```

然后再额外 compile 最终 TIR：

```text
lowered_cuda
ptx/cubin
sass
```

。

这样以后可以研究：

\[
\text{SpaceGenerator}
\rightarrow
\text{SearchStrategy}
\rightarrow
\text{CostModel}
\rightarrow
\text{Measurement}
\]

分别造成什么。

---

# 14. Q2 比较时必须避免一个严重的公平性错误

不能直接算：

\[
\frac{|\mathcal C_{\rm Triton}|}
{|\mathcal C_{\rm Hexcute}|}
\]

然后说谁 coverage 大。

因为它们的 candidate abstraction 不一样。

正确做法是构造：

\[
\boxed{
Canonical\ Realized\ Configuration
}
\]

把最终 kernel 转换成统一 descriptor。

例如：

```text
CTA tile = 128x128x64
consumer warpgroups = 2
producer warpgroups = 1
pipeline = 4
copy = TMA
mma = WGMMA m64n128k16
...
```

然后比较：

\[
\Phi(\mathcal C_F)
\]

其中：

\[
\Phi
\]

是各框架 candidate 到 canonical hardware configuration 的映射。

---

# 15. 即便如此，Q2 还应该分两类结论

第一类：

\[
\boxed{Native Coverage}
\]

问：

> 在框架自己可控制/可生成的空间里，它漏了什么？

例如 Hexcute：

\[
all\ legal\ inferred
\rightarrow
pruned
\rightarrow
selected
\]

最好研究。

---

第二类：

\[
\boxed{Cross-framework Performance Envelope}
\]

问：

> 不同系统产生的 configuration union 中，谁留下了哪些高性能区域？

定义：

\[
\mathcal U_{\rm empirical}
=
\bigcup_F\Phi(\mathcal C_F)
+
\mathcal C_{\rm expert}
\]

然后：

\[
T^*_{\rm known}
=
\min_{c\in\mathcal U_{\rm empirical}}T(c)
\]

这里只能叫：

\[
\boxed{best-known empirical oracle}
\]

不能叫 true global optimum。

---

# 16. 真实 GEMM shape 不应该继续只用 4096³

Hexcute artifact 的：

\[
4096\times4096\times4096
\]

非常适合 artifact reproduction。fileciteturn88file0L1-L13

但科研实验需要真实模型 shape。

我建议第一个真实模型固定用 **Qwen3-8B**，因为它非常干净：

\[
hidden=4096
\]

\[
intermediate=12288
\]

\[
H_q=32
\]

\[
H_{kv}=8
\]

\[
D=128
\]

。citeturn166677search0

因此可以直接产生真实 projection shapes。

---

# 17. Qwen3-8B 对应的 GEMM families

令：

\[
M=N_{\rm tokens}
\]

。

Attention QKV 如果采用 fused projection，输出维度：

\[
32\times128
+
2(8\times128)
=
6144
\]

所以：

\[
\boxed{
(M,4096,6144)
}
\]

这里我用：

\[
(M,K,N)
\]

表示。

Attention output projection：

\[
\boxed{
(M,4096,4096)
}
\]

。

Gate+Up 如果 fused：

\[
N=2\times12288=24576
\]

所以：

\[
\boxed{
(M,4096,24576)
}
\]

。

Down projection：

\[
\boxed{
(M,12288,4096)
}
\]

。

这些比一个孤立 4096³ 更贴近真正 Transformer workload。Qwen3 的模型尺寸来源于官方模型 config。citeturn166677search0

---

# 18. 第一版 M grid 我建议这样

\[
M\in
\{
1,16,64,128,512,2048,8192
\}
\]

意义分别是：

```text
M=1          单请求 decode-like extreme
16/64        small-batch decode / small token group
128/512      chunk/prefill small
2048         common medium prefill
8192         long prefill / high-throughput
```

这样同一个：

\[
K,N
\]

可以观察最优 configuration 随 M 改变的相变。

这一点本身以后也能支持 workload-dependent optimization 的研究。

---

# 19. 第一阶段 GEMM benchmark matrix

我建议不是所有笛卡尔积都跑。

先跑：

| Family | \((K,N)\) | M |
|---|---:|---|
| Artifact control | 4096,4096 | 4096 |
| QKV | 4096,6144 | 1,64,512,2048,8192 |
| O proj | 4096,4096 | 1,64,512,2048,8192 |
| Gate+Up | 4096,24576 | 1,64,512,2048 |
| Down | 12288,4096 | 1,64,512,2048 |

dtype 第一轮只做：

\[
BF16/FP16
\]

第二阶段再做：

\[
FP8
\]

否则 candidate space 一开始就太大。

---

# 20. Prefill Attention 的 `problem.json`

至少应该定义：

```text
B

Sq
Skv

Hq
Hkv

Dqk
Dv

q_dtype
kv_dtype
o_dtype
acc_dtype

causal
window

q_layout
k_layout
v_layout
o_layout
```

不要只记录：

```text
sequence_length
```

因为：

\[
H_q/H_{kv}
\]

对 GQA mapping 极其重要。

---

# 21. Attention 的 canonical decision space 比 GEMM 更复杂

至少要加入：

| 类别 | 字段 |
|---|---|
| Q tile | `block_m` |
| KV tile | `block_n` |
| CTA cluster | `cluster_m,n` |
| Warp roles | producer/consumer WG |
| Q copy | instruction/vector/layout |
| K copy | instruction/vector/layout |
| V copy | instruction/vector/layout |
| Q SMEM | layout/swizzle |
| K SMEM | layout/swizzle |
| V SMEM | layout/swizzle |
| QK MMA | instruction + TV mapping |
| Score P | register/shared representation |
| Softmax | reduction mapping |
| PV MMA | instruction + TV mapping |
| Output | register→SMEM→GMEM strategy |
| Pipeline | K/V stages |
| Sync | barrier/mbarrier structure |
| Resources | regs/SMEM/occupancy |

这里尤其要把：

\[
QK\ MMA
\]

和：

\[
PV\ MMA
\]

分开记录。

---

# 22. 为什么这个字段是关键？

Hexcute 当前 attention benchmark 本身就注册的是：

\[
\boxed{
(TiledMma_{QK},TiledMma_{O})
}
\]

一对 MMA configuration。

其代码分别建立 QK MMA 和 output/PV MMA 的 `TiledMma`，然后把 pair 放入 candidate list。fileciteturn90file0L1-L2

这本身说明：

\[
L_{QK}
\]

和：

\[
L_{PV}
\]

不是一个简单单变量。

这正是后续 Q1 的基础。

---

# 23. Hexcute Attention 当前 candidate space已经相当有信息量

源码显式枚举：

\[
head\_size
\in
\{32,64,96,128,160,192,224,256\}
\]

\[
warpgroup\in\{1,2\}
\]

\[
N_{\rm MMA}
\in
\{32,64,128,256\}
\]

然后在 cooperative kernel 上进一步搜索：

\[
stages=2
\]

\[
cluster_m\in\{1,2\}
\]

。fileciteturn90file0L1-L2

共享内存 Q/K/V 又写成 `layout_auto`，copy 使用 `auto_copy`，所以有一部分 decision 是显式 tune variable，一部分则留给 Hexcute 自动 layout/instruction inference。fileciteturn90file0L1-L2

这正适合我们的 ownership matrix。

---

# 24. Triton官方 attention candidate space 可以作为另一个对照

当前 Triton Fused Attention tutorial 构造 configs：

\[
BLOCK_M\in\{64,128\}
\]

\[
BLOCK_N\in\{32,64,128\}
\]

\[
num\_warps\in\{4,8\}
\]

以及多种 `num_stages`，随后还通过 `keep()` 和 `prune_invalid_configs()` 删除部分 candidate。citeturn318331search4

这非常适合做：

\[
\boxed{
Pre-pruning\ space
\rightarrow
Post-pruning\ space
\rightarrow
Autotune\ winner
}
\]

三阶段研究。

不要只保存 winner。

---

# 25. Prefill Attention 必须分成两个 benchmark tier

这是非常重要的。

### Tier A：跨框架共同控制组

使用 MHA：

\[
B=1
\]

\[
H_q=H_{kv}=32
\]

\[
D=128
\]

因为 Hexcute stock artifact 当前就是这个语义。fileciteturn88file0L1-L13

sequence：

\[
S_q=S_{kv}
\in
\{
512,2048,8192,16384
\}
\]

。

这个 Tier 用来：

\[
CUTLASS/FA-like,\ Triton,\ TVM,\ Hexcute
\]

尽可能公平地比较 candidate generation 和 selection。

---

# 26. Tier B：真实现代 GQA workload

直接使用 Qwen3-8B：

\[
H_q=32,\qquad H_{kv}=8,\qquad D=128
\]

。citeturn166677search0

序列长度：

\[
S\in
\{
128,512,2048,8192,16384,32768
\}
\]

这已经覆盖其 40,960 max-context 内的重要区间。citeturn166677search0

这里：

- Triton/FlashAttention 类实现如果支持 GQA，直接跑；
- SGLang/vLLM 可作为现实 runtime backend baseline；
- Hexcute stock artifact 若不能运行，标：

```text
semantic_status = unsupported_stock
```

绝对不要写：

```text
coverage_regret = infinity
```

。

如果以后我们扩展 Hexcute GQA，那是另一项实验。

---

# 27. DeepSeek MLA 不要混进第一轮 Prefill Attention

DeepSeek-V3/V3.2 的 attention 不是同一个问题。

例如当前官方 config：

\[
H_q=128
\]

\[
qk\_nope\_head\_dim=128
\]

\[
qk\_rope\_head\_dim=64
\]

\[
v\_head\_dim=128
\]

以及：

\[
kv\_lora\_rank=512
\]

。citeturn166677search1turn166677search8

它属于 MLA/latent representation 体系。

第一轮把它和标准 MHA/GQA 混在一起，会导致：

> algorithmic difference 被误解释成 layout difference。

所以 DeepSeek 应该作为第二阶段 stress test。

---

# 28. 还需要一个非常严格的 `support_status`

我建议每个 `(framework,problem)` 首先记录：

```text
supported_native
supported_with_adapter
supported_with_kernel_modification
unsupported_semantics
compile_failure
resource_infeasible
numerical_failure
```

之后只有：

\[
supported\_native
\]

或你明确规定允许的实现层，才能进入 framework-native coverage calculation。

这会避免大量错误科研结论。

---

# 29. Candidate 的合法性也应该分层

一个 config 可能：

1. 数学语义合法；
2. framework IR 合法；
3. hardware instruction 合法；
4. resource 可部署；
5. 编译成功；
6. 数值正确。

所以状态建议：

```text
generated
legal
compiled
correct
measured
```

分别记录。

尤其 Hexcute constraint pruning 与 Triton pruning 可以因此公平比较。

---

# 30. 统一结果目录我建议直接定下来

```text
results/
  machine/
    gpu.json
    cuda.json
    clocks.json

  commits/
    cutlass.json
    triton.json
    tvm.json
    hexcute.json
    sglang.json
    vllm.json

  gemm/
    <problem_id>/
      problem.json

      cutlass/
        candidate_<id>/
          decision.json
          realized.json
          compile/
          measure/

      triton/
        candidate_<id>/
          decision.json
          realized.json
          compile/
            ttir
            ttgir
            llir
            ptx
            cubin
            sass
          measure/

      tvm/
        ...

      hexcute/
        ...

  attention_prefill/
    ...
```

不要把 profiler 文件直接全部扔一个目录。

---

# 31. 每个 candidate 的 measurement 目录

建议：

```text
measure/
  latency.json
  correctness.json

  ncu/
    report.ncu-rep
    summary.csv

  nsys/
    trace.nsys-rep
```

但是注意：

\[
\boxed{
不要对每一个 search candidate 跑完整 NCU
}
\]

那会把实验规模扩大几个数量级。

---

# 32. 正确 profiling pipeline 应该分两阶段

第一阶段所有 candidate 只测：

\[
latency
\]

和基础 runtime resource：

\[
regs,\ smem,\ occupancy
\]

。

筛出：

```text
framework winner
best measured
top-5
near-optimal
bad-but-interesting
```

。

第二阶段才对这些代表 candidate 跑：

- Nsight Compute；
- PTX/SASS；
- instruction statistics；
- bank conflict；
- DRAM/L2；
- stall。

否则 Q2 candidate enumeration 会完全跑不动。

---

# 33. latency measurement 也必须统一

每个 candidate：

先 warm-up，

然后至少多轮采样。

主指标用：

\[
median
\]

同时保存：

\[
p5,p25,p50,p75,p95
\]

以及 bootstrap confidence interval。

不要只保存：

```text
average_ms
```

。

而且 compiler time 必须和 runtime time 分开。

---

# 34. GPU 环境必须进入实验对象

Hexcute artifact 本身就会固定 H100 clock，例如当前 H100 script 用：

```bash
nvidia-smi -pm 1
nvidia-smi -lgc 1410
```

。fileciteturn88file0L1-L13

我们未必要完全照 1410 MHz，但同一实验必须：

- 固定 power/clock policy；
- 记录 GPU SKU；
- driver；
- CUDA；
- compiler commit；
- temperature/power state。

否则 2–5% 的 regret 没有解释意义。

---

# 35. Q2 最终数据表不要只是 candidate count

核心表应该长这样：

| problem | framework | generated | legal | compiled | measured | best latency | 1% near-opt | 5% near-opt | best-known gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

并计算：

\[
CompressionRatio_F
=
\frac{|\mathcal C_F|}
{|\mathcal U_{\rm reference}|}
\]

但只有在我们真的定义了一个有意义的 reference space 时才报告。

更稳妥的早期指标是：

\[
Gap^{known}_F
=
\frac{
\min_{c\in\mathcal C_F}T(c)
}{
T^*_{\rm best-known}
}-1
\]

。

名字一定要叫：

\[
\boxed{best\text{-}known\ gap}
\]

而不是 global coverage regret。

---

# 36. Near-optimal region 比 winner 更重要

以所有可比较 measured candidate 的 best-known 为：

\[
T^*
\]

定义：

\[
\mathcal N_\epsilon
=
\{c:T(c)\le(1+\epsilon)T^*\}
\]

我们应该研究不同 framework 的：

\[
\Phi(\mathcal C_F)
\cap
\mathcal N_{1\%}
\]

\[
\Phi(\mathcal C_F)
\cap
\mathcal N_{5\%}
\]

\[
\Phi(\mathcal C_F)
\cap
\mathcal N_{10\%}
\]

。

如果某框架只碰巧留下一个 optimum，但几乎不覆盖 near-optimal basin，这个 generator 很脆弱。

---

# 37. Q3 的数据可以从同一批实验直接得到

对每个框架自己的 measured candidates：

\[
\mathcal C_F
\]

找到：

\[
c_F^*=
\arg\min_{c\in\mathcal C_F}T(c)
\]

然后拿 framework native selector 选出的：

\[
\hat c_F
\]

计算：

\[
\boxed{
SelectionRegret_F
=
\frac{T(\hat c_F)}{T(c_F^*)}-1
}
\]

这一步非常关键。

因为如果：

```text
framework 最终 kernel 慢 12%
```

但：

\[
SelectionRegret=0
\]

说明 selector 没问题。

问题是：

\[
candidate\ space
\]

。

反之，如果 candidate space 中已经有最快 config，但 native selector 没挑中，才是 Q3。

---

# 38. 对 Hexcute 可以直接进一步分解

因为它同时拥有：

\[
constraint\ generation
\]

和：

\[
analytic\ cost\ model
\]

。

应该为每个 candidate保存：

\[
C_{\rm predicted}
\]

和：

\[
T_{\rm measured}
\]

然后计算：

\[
Kendall\ \tau
\]

\[
Top1Accuracy
\]

\[
Top5Recall
\]

以及：

\[
SelectionRegret
\]

。

其 artifact 已经专门提供 analytical cost-model accuracy evaluation，因此第一步应该先复现原论文这一结果，再扩展到我们的真实模型 shapes，而不是从零发明 benchmark。fileciteturn87file0L1-L13

---

# 39. 对 Triton 则可以分三层

```text
all specified configs
        ↓
prune_invalid_configs
        ↓
actual autotune benchmark set
        ↓
winner
```

官方 Attention tutorial 本身已经存在这样的 pruning。citeturn318331search4

所以可分别计算：

\[
R_{pruning}
\]

和：

\[
R_{selection}
\]

。

如果 best config 在 pruning 阶段被删掉：

是 candidate-generation/pruning 问题。

如果保留下来了但 autotuner winner 不是它：

才是 selection/measurement 问题。

---

# 40. TVM 也应该进行完全相同的拆分

```text
SpaceGenerator
      ↓
design space

SearchStrategy
      ↓
proposed MeasureCandidates

CostModel
      ↓
ranking

Runner
      ↓
actual measured latency

Database
      ↓
chosen best record
```

TVM 当前默认 work directory 已经把 workload 与 tuning records，包括 schedule trace 和 measured run times，持久化为 JSONL。citeturn318331search5turn318331search6

所以 TVM 是最适合做这一类内部因果分解的 baseline 之一。

---

# 41. 第一阶段实验我建议明确分成 Phase 0–3

### Phase 0 — Reproduction

完全不改框架。

跑：

Hexcute H100：

```bash
bash run_h100.sh --gemm --host
bash run_h100.sh --attn --host
```

artifact 支持这些 mode。fileciteturn88file0L1-L13

CUTLASS profiler control。

Triton官方 matmul/attention tutorial。

TVM标准 MetaSchedule GEMM。

目的只验证环境正确。

---

### Phase 1 — Candidate extraction

不研究真实模型。

固定：

\[
GEMM=4096^3
\]

和：

\[
Attention:
B=1,H=32,D=128,S=16384
\]

也就是与 Hexcute artifact 对齐。

目标：

\[
\boxed{
确认四个框架都能导出 candidate set + realized code
}
\]

。

这一步不关心最终谁快。

---

### Phase 2 — Real LLM shapes

加入 Qwen3-8B：

GEMM families和标准 GQA Prefill。

目标：

\[
Coverage/Selection
\]

随真实 shape 的变化。

---

### Phase 3 — Q1 preparation

只有 Phase 1/2 做完以后，才开始修改：

\[
\text{decision ownership}
\]

例如固定/放开：

- QK layout；
- PV layout；
- pipeline；
- consumer mapping。

开始真正测 Boundary Regret。

---

# 42. 为什么不能直接跳 Phase 3？

假设 joint search 最后比 Triton 快 10%。

如果不知道 Triton：

- 搜了哪些 configs；
- prune 了什么；
- compiler 又改变了什么；
- winner 是否是 measured set 中最好；

这 10% 完全无法解释。

可能是：

\[
BoundaryRegret
\]

也可能是：

\[
CoverageRegret
\]

甚至仅仅是：

\[
SelectionRegret
\]

。

所以 Phase 1/2 不是工程前置工作，而是后面科学因果分析的必要条件。

---

# 43. 我建议第一版 central CSV 直接固定这些列

```text
problem_id
candidate_id
framework
framework_commit
gpu
cuda

op
model
phase

B
Sq
Skv
Hq
Hkv
D

M
N
K

dtype

semantic_status
generated
legal
compiled
correct
measured

tile_m
tile_n
tile_k

cluster_m
cluster_n
cluster_k

num_threads
num_warps

producer_warps
consumer_warps

pipeline_family
stages

copy_a
copy_b
copy_q
copy_k
copy_v

smem_layout_a
smem_layout_b
smem_layout_q
smem_layout_k
smem_layout_v

mma_qk
mma_pv
mma_gemm

registers
smem_bytes
occupancy

predicted_cost
latency_us

dram_bytes
l2_hit_rate
bank_conflicts
```

缺失字段填：

```text
NA
```

而不是猜。

---

# 44. 还需要一张 `decision_ownership.csv`

这个其实可能会成为论文里很有价值的数据。

形式：

| candidate | decision | value | owner | explicit/inferred | controllable |
|---|---|---|---|---|---|
| X | BM | 128 | user/autotuner | explicit | yes |
| X | SMEM-A | ... | Hexcute solver | inferred | indirect |
| X | MMA | WGMMA... | compiler | inferred | no |
| X | stages | 4 | user | explicit | yes |

这样可以真正定量研究：

\[
\text{哪个 optimization layer 拥有多少 decision freedom}
\]

。

---

# 45. 最终 Q2 的第一个重要图不是性能柱状图

我建议做：

\[
x=\text{canonical configuration embedding}
\]

\[
y=\text{measured latency}
\]

不同颜色代表：

```text
CUTLASS
Triton
TVM
Hexcute
```

我们想看到：

> 各框架 candidate 分布在 configuration space 的哪些区域？

而不是单纯：

```text
CUTLASS 120 us
Triton 125 us
Hexcute 118 us
```

后者几乎不能解释科研问题。

---

# 46. 第二张图应该是 Coverage–Search-Cost frontier

横轴：

\[
N_{\rm candidates\ evaluated}
\]

或 compile+tuning cost。

纵轴：

\[
BestKnownGap
\]

。

理想曲线：

```text
BestKnownGap
↑
│  Random
│      *
│     *
│ Triton *
│    *
│ TVM *
│  Hexcute
│ *
└────────────────→ Candidates / search cost
```

真正问题：

> 谁用更少候选进入 near-optimal region？

这才对应 Q2。

---

# 47. 第三张图才是 Selection Regret

对框架分别画：

\[
T(\hat c_F)
\]

vs：

\[
\min_{\mathcal C_F}T
\]

。

例如：

```text
Framework        Selected       Native Oracle
CUTLASS-auto       ...
Triton-auto        ...
TVM-cost-model     ...
Hexcute-cost       ...
```

这样 Q3 非常清楚。

---

# 48. 完成这套 GEMM/Prefill 基础设施后，科研问题可能出现三种结果

第一种，也是最理想的：

\[
\boxed{
CoverageRegret\gg SelectionRegret
}
\]

说明问题主要在：

> candidate generator 没表达出好 configuration。

这支持 Q2。

---

第二种：

\[
\boxed{
CoverageRegret\approx0,\quad
SelectionRegret\gg0
}
\]

说明候选已经存在：

> ranking/selector 才是问题。

这支持 Q3。

---

第三种：

\[
CoverageRegret\approx0
\]

\[
SelectionRegret\approx0
\]

但下一阶段仍然发现：

\[
BoundaryRegret\gg0
\]

这反而是最有意思的结果：

> **每一层内部都已经做得很好，但 optimization boundary 本身阻止了全局更优方案。**

这会非常强地支持 Q1。

---

# 49. 现在我对下一步的判断

完成这一轮后，我认为已经不应该继续无限做 conceptual taxonomy 了。

现在已经有足够清楚的链条：

\[
\boxed{
Support
\rightarrow
CandidateGeneration
\rightarrow
Legality
\rightarrow
Pruning
\rightarrow
Selection
\rightarrow
Codegen
\rightarrow
Performance
}
\]

下一轮最合理的是把上面 specification 进一步缩成一个**真正可以开始实现的最小实验版本**，但依然不是新优化器。

我建议下一轮只做：

\[
\boxed{
GEMM\ Q2/Q3\ Minimal\ Experimental\ Plan
}
\]

具体到：

**4–6 个 shape、CUTLASS/Triton/TVM/Hexcute 各自要改哪一个 benchmark 文件、如何强制导出所有 candidate、如何生成统一 `candidate.json`、如何 dump IR/PTX/SASS、如何写统一 runner、第一版应该搜索哪些维度、总 candidate 数控制在多少，以及最终用什么脚本算 Coverage/Selection Regret。**

等 GEMM 这个 negative-control infrastructure 跑通以后，再复制到 Prefill Attention；这样不会一上来被 FlashAttention 的复杂 pipeline 淹没。

## 用户

除此之外还有哪些科研问题呢（这些科研问题应该从当前已有框架中的缺陷/解决layout问题的解决方法以及假设不是很合适 得到），请继续补充。所以上述的有些回答可能证据不充足 或者 没那么重要。

## 模型思考摘要

拓展科学问题
拓展科学问题

## 模型回复

我会补充，但这次不追求“多列几个问题”，而是优先找那些能从现有框架的**设计假设、已知限制、优化边界、搜索空间构造方式**直接推出来的问题。我会把每个候选问题都标成：证据强/中/弱，以及它到底是“科学问题”还是只是工程缺口。

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 3 个网站

## 模型思考摘要

聚焦关键问题
聚焦关键问题

## 模型思考摘要

已搜索 3 个网站

## 模型思考用时

思考了 51s

## 用户

另外上述不同框架方法统计对比若不是很充分的话，也应该 继续完善

## 模型回复

我会同时补两层：一层把现有框架的方法统计补全到“表示—候选生成—合法性—性能模型—资源约束—动态 shape—跨算子—运行时状态”这些维度；另一层从这些方法共同依赖的假设里继续挖新的科研问题。尤其会优先找“某个假设在多个框架里都反复出现，但其实未被严格验证”的问题。

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 docs.nvidia.com

## 模型思考摘要

深化布局抽象
深化了布局抽象

## 模型思考摘要

权衡表示能力
权衡了表示能力

## 模型工具调用参数

{"paths":["GitHub"],"query":"fetch_file"}

## 模型思考摘要

已搜索 1 个网站

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 TileLang 布局推理源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取入门文档并检查循环验证器

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索布局推断实现与传播逻辑

## 模型思考摘要

搜索TileLang布局推理源码并检查循环验证与传播实现

## 模型思考：工具活动

正在搜索 Hexcute CGO 2026 layout synthesis constraint programming type inference explicit dataflow pipelining

## 模型思考：工具活动

正在搜索 2 个网站

## 模型思考摘要

完善研究问题
完善研究问题

## 模型思考摘要

已搜索 2 个网站

## 模型回复

目前我看到几个比前面 Q1/Q2 更“贴 layout 本体”的方向，证据也更直接：例如 **TIRx 明确把 layout 当 storage contract，而 CuTe/Hexcute 把 layout 与 task mapping 更紧地绑定**；Gluon 官方文档明确承认 **没有 canonical layout representation**；TIRx 明确强调 general non-power-of-two shapes，而 Gluon 的布局模型仍强依赖 power-of-two tile；另外 Tawa 2026 又直接挑战了“pipeline/warp specialization 必须由专家显式控制”的假设。

## 模型思考用时

思考了 2m 56s

## 模型回复

是的。前面的 Q1「Boundary Regret」和 Q2「Coverage Regret」更适合作为**分析框架/评价方法**，未必本身就是最强的论文科研问题。继续把现有系统展开后，可以找到一些更具体、证据更直接，而且确实来自“现有方法用了什么假设”的问题。

我这次把比较范围也扩大了：除了 SGLang、vLLM、CUTLASS/CuTe、Triton、TVM、Hexcute，还加入 FlashInfer、TileLang，以及两个很有价值的 2026 参考工作 LEGO 和 Tawa。它们不是都处于同一抽象层，但正好可以暴露不同设计假设之间的矛盾。

## 1. 先把框架比较补完整

下面这张表比之前的 decision ownership 更重要，因为增加了“**方法到底假设了什么**”。

| 系统 | 它认为 layout 是什么 | layout/task mapping 如何产生 | Legality | Performance choice | Dataflow / Pipeline | 当前设计假设 |
|---|---|---|---|---|---|---|
| SGLang | runtime backend/cache contract | backend/model/hardware rules | feature compatibility | 手工策略 | backend 内部处理 | backend 选好以后，可把低层 mapping 委托出去 |
| vLLM | persistent KV physical storage | 有限 KV layout 集合 | backend support 交集 | preference/connector priority | kernel delegated | 整个模型使用一个兼容 persistent layout 足够 |
| FlashInfer | attention/KV 专用 NHD/HND/page layouts | 少量预设计 layout/kernel families | kernel requirements | backend auto/经验规则 | kernel 专门实现 | attention 所需 layout space 可以被有限 family 覆盖 |
| CUTLASS/CuTe | 数据 + thread/work hierarchical mapping | expert composition / Builder | type/template/instruction contracts | expert + heuristics | 显式、深度耦合 | 专家可以构造正确的 joint configuration |
| Triton classic | compiler-owned distributed encoding | compiler passes | encoding/instruction rules | heuristic + autotune | compiler pass | staged local rewrites通常足够接近好方案 |
| Gluon | 显式 register/lane/warp/block ownership | programmer explicit | compiler checking | programmer + autotune | 显式 | layout 应该暴露给专家；当前 layout 甚至没有 canonical representation |
| TVM MetaSchedule | schedule 的结果/部分 buffer layout | ScheduleRules + search | schedule/postproc | learned model + measurement | schedule/search | 好实现应该已经存在于 schedule-rule design space |
| TIRx | **storage contract** | user storage layout + primitive dispatch | dispatch predicate | priority/dispatch | pipeline/orchestration 显式 | storage layout + scope + instruction semantics 足以推导 execution mapping |
| Hexcute | task mapping + memory layout | constraint/type inference | constraint unification | analytic cost model | dataflow/pipeline 显式 | 固定 dataflow/pipeline 后，layout+mapping+instruction 可以自动综合 |
| TileLang | fragment/register distribution | LayoutInference | inference consistency | autotune/heuristics | pipeline/WS 单独处理 | 从 tile program 可以推导 fragment layout |
| LEGO | layout-independent computation + layout specification | layout expression → indexing | mapping semantics | 本身不负责完整 perf search | 独立 | computation 与 hierarchical layout 可以干净解耦 |
| Tawa | 重点不是 storage layout，而是 warp roles/dataflow | 自动 producer/consumer partition | async-reference semantics | compiler | **自动 warp specialization** | pipeline/role assignment 不一定必须专家手写 |

几个特别关键的事实值得单独指出。

TIRx 明确说自己的 layout 是 **storage contract，而不是 work-partitioning interface**；primitive dispatch 再根据 operand layout、scope、target 生成 thread partition、loop 和 instruction。相比之下，CuTe 的 layout 本身参与 partition/tiled copy/tiled MMA，而 Hexcute直接联合推导 task mapping 和 memory layout。citeturn415539search3turn862114search0 fileciteturn80file0L1-L7

Gluon 官方文档则直接说：**没有 canonical layout representation**，多个 layout 可以表示完全相同的 element mapping；同时当前 Triton/Gluon tile model 仍要求 tile dimensions 为 power-of-two。TIRx 的设计文档恰恰把“直接支持 general non-power-of-two shape”列为设计理由。citeturn928928search1turn415539search1

Hexcute 和 TIRx 都选择保留 pipeline/orchestration 的显式控制；但 Tawa 2026 的工作恰好证明另一条设计路线是可行的——它从 tile program 自动生成 warp-specialized producer/consumer pipeline，并在 H100 上报告了与手写 FA3 竞争的性能。这至少说明“pipeline 必须人工控制”不是一个已经被证明正确的原则。citeturn436659search0turn415539search3turn956082academia17

这类“设计假设冲突”才是后面科研问题更可靠的来源。

---

# 2. 新问题一：Storage Layout 和 Task Mapping 真能分开吗？

这是我现在认为**非常强**的一个问题，而且比之前泛化的 Boundary Regret 更具体。

存在三种不同哲学：

\[
\text{CuTe:}\quad
Layout \leftrightarrow WorkPartition
\]

\[
\text{Hexcute:}\quad
Layout + TaskMapping
\quad\text{joint constraint inference}
\]

\[
\text{TIRx:}\quad
StorageLayout
\rightarrow
Dispatch
\rightarrow
TaskMapping
\]

TIRx 明确认为 layout 只需要表达 storage；执行 partition 可以由 primitive dispatch 派生。citeturn415539search3

因此真正的科学问题是：

\[
\boxed{
给定 storage layout、execution scope 和 instruction，
剩余的 task-mapping freedom 到底有多大？
}
\]

定义：

\[
\mathcal M(L_s,I,S)
=
\{
M:
M\text{ compatible with }L_s,I,S
\}
\]

关键不是：

\[
|\mathcal M|>1
\]

而是：

\[
\mathrm{Var}_{M\in\mathcal M}
[T(L_s,M)]
\]

有多大。

如果同一 storage layout 下：

\[
\max T/\min T \approx 1
\]

那么 TIRx 的 factorization 很合理。

如果同一个 storage layout 可以对应大量合法、性能差异 20–50% 的 task mappings，那么：

\[
\boxed{
Storage\ layout\not\Rightarrow Performance\ mapping
}
\]

就成为很重要的结果。

我的假设是：

\[
\text{regular GEMM}
\]

可能接近可分，

但：

\[
Attention,\ Reduction,\ GQA,\ Gather,\ MoE
\]

剩余 mapping freedom 会明显更大。

证据强度：**很强**。  
我目前会把它列入 S 级候选。

---

# 3. 新问题二：Layout Space 中大量“不同 layout”是不是其实同一个东西？

这个以前没有讨论，但我认为很值得认真考虑。

Gluon 官方文档直接给出两个不同 layout expression，说明它们拥有相同 tensor-element mapping，并明确说：

> Gluon layouts have no canonical representation. citeturn928928search1

Hexcute 源码中也频繁调用：

```text
canonicalize_thread_value_layout
```

去规范 thread/value layout 后再进行 instruction matching。fileciteturn82file0L1-L7

CuTe 又允许大量：

\[
composition,
tiling,
product,
partition
\]

所以很自然出现：

\[
L_1\neq L_2
\]

但物理效果可能：

\[
L_1\equiv L_2
\]

。

因此科研问题是：

\[
\boxed{
GPU layout search space 应该搜索 syntactic layouts，
还是 layout equivalence classes？
}
\]

可以定义 equivalence：

\[
L_1\sim L_2
\]

当且仅当两者：

1. logical→thread ownership 相同；
2. memory addresses 相同；
3. 或只差 intra-thread register permutation 这种零成本变换。

然后搜索：

\[
\mathcal L/\sim
\]

而不是：

\[
\mathcal L
\]

。

### Hypothesis

我怀疑：

\[
|\mathcal L/\sim|
\ll
|\mathcal L|
\]

也就是说目前很多搜索/constraint work 可能浪费在等价 configuration 上。

更有意思的是另一种可能：

两个**语义等价** layout expression：

\[
L_1\sim L_2
\]

却由于 compiler pattern matching 不同生成不同代码：

\[
Codegen(L_1)\neq Codegen(L_2)
\]

那么这又会得到一个更深的问题：

\[
\boxed{
semantic\ layout\ equivalence
\neq
compiler\ optimization\ equivalence
}
\]

这可能直接解释 heuristic instability。

证据强度：**很强**。  
重要性：我认为高于普通 cost-model 改进。

---

# 4. 新问题三：Power-of-two layout 假设对现代 LLM 是否已经不合理？

这个问题的证据也非常直接。

Gluon 当前文档明确：

> Triton requires tile dimensions to be powers of 2. citeturn928928search1

而 TIRx 的 layout design rationale 特别强调：

> non-power-of-two shapes are common，因此直接支持 general shapes。citeturn415539search1

Hexcute H100 GEMM candidate 甚至显式使用：

\[
N=96,144,160,176,\ldots
\]

之类非 2 次幂 MMA tiles。fileciteturn91file0L1-L2

TileLang 当前也有真实 layout-inference 问题涉及 width 不整除 thread count、非规则 reduction layout 等；这些只能作为工程侧佐证，不能直接证明方法错误。citeturn356325search5turn356325search9

真正的科学问题不是：

> Triton 不支持某个奇怪 shape。

而是：

\[
\boxed{
Power\text{-}of\text{-}two hierarchical tiling
在 irregular tensor shape 上到底损失了什么？
}
\]

很多人会认为损失只是：

\[
padding/masking
\]

但我怀疑实际还包括：

\[
\text{thread ownership fragmentation}
\]

\[
\text{register waste}
\]

\[
\text{reduction communication}
\]

\[
\text{MMA tile mismatch}
\]

\[
\text{pipeline footprint}
\]

。

因此可以验证：

\[
T_{\text{general-layout}}
\]

versus：

\[
T_{\text{next-power-of-two}}
\]

并分解浪费来源。

特别适合：

\[
D=80,96,160,192,224
\]

mixed-type MoE，

non-uniform quant group，

以及各种新 attention head dimensions。

证据：**很强**。  
LLM relevance：**很强**。  
我也会放进 S 级。

---

# 5. 新问题四：Layout 和 Pipeline/Warp Specialization 真能独立设计吗？

这是 Hexcute、TIRx、Tawa 三者形成的非常漂亮的冲突。

Hexcute：

\[
\boxed{
dataflow/pipeline = programmer
}
\]

\[
layout/task/instruction = compiler
\]

citeturn436659search0

TIRx 同样说：

pipeline structure、synchronization、role assignment 保留在 native source。citeturn415539search3

但 Tawa：

\[
tile\ program
\rightarrow
automatic\ producer/consumer\ partition
\rightarrow
automatic\ warp-specialized\ pipeline
\]

。citeturn956082academia17

所以科研问题应该是：

\[
\boxed{
Layout/TaskMapping
和
Pipeline/RoleAssignment
到底具有多强的性能耦合？
}
\]

不是简单：

> joint search 更快。

需要研究 rank reversal：

对 layout：

\[
L_1,L_2
\]

在 pipeline \(P_1\) 下：

\[
T(L_1,P_1)<T(L_2,P_1)
\]

但在 \(P_2\) 下：

\[
T(L_1,P_2)>T(L_2,P_2)
\]

这种 reversal 有多频繁。

如果频繁：

\[
\boxed{
先定 pipeline 再 synthesis layout
}
\]

就不是一个安全的 decomposition。

我认为 Hopper/Blackwell 上：

\[
TMA+mbarrier+WGMMA/tcgen05
\]

使这个问题特别重要。

证据：**非常强**。

---

# 6. 新问题五：Compiler 不应该只“减少 Layout Conversion”，而应该主动决定什么时候转换

这是一个我认为比之前 P3 更好的表述。

Gluon 官方教程给了两个非常重要但方向相反的 observation。

第一种情况：

input/output global layout 不同，使用两个合适的 layouts + 一次 conversion，比强行用一个 layout 快很多。

第二种情况：

对于 reduction，虽然存在 reduction-friendly layout，但先 `convert_layout` 再 reduce 往往比直接在原 layout 上完成 reduction 更慢。citeturn928928search1

Triton compiler 同时存在：

- `Coalesce`
- `AccelerateMatmul`
- `OptimizeThreadLocality`
- `RemoveLayoutConversions`

这实际上就在解决这个问题。fileciteturn66file0L1-L7 fileciteturn67file0L1-L7

所以真正科研问题是：

\[
\boxed{
什么条件决定一次 layout transition 是“值得”的？
}
\]

而不是：

\[
\min\#ConvertLayout
\]

。

应该优化：

\[
\sum_i C_{\rm op}(O_i,L_i)
+
\sum_i C_{\rm transition}(L_i,L_{i+1})
\]

。

而且：

\[
C_{\rm transition}
\]

不是一个常数。

可能是：

- register rename；
- warp shuffle；
- shared-memory exchange；
- TMA copy；
- global materialization。

这个问题非常适合：

\[
QK^T\rightarrow Softmax\rightarrow PV
\]

以及：

\[
GEMM\rightarrow SiLU\times\rightarrow GEMM
\]

。

证据：**非常强**。  
可验证性：**很高**。  
如果想先做一篇比较聚焦的论文，我认为这个方向比抽象的 Boundary Regret 更容易站住。

---

# 7. 新问题六：Replication 其实应该是 Layout Optimization 的一等变量

这一点目前也容易被忽略。

Layout 不只是 permutation。

TIRx 的 layout 明确包含：

\[
R = Replica
\]

一个逻辑元素可以映射到多个 physical coordinates。citeturn415539search1

而 Triton 甚至有专门的：

`ReduceDataDuplication`

pass，尝试减少 register 中的 duplication，并用 shared tensor 重用来替代某些 distributed→dotOperand conversion。fileciteturn67file0L1-L7

所以 replication 其实有两面：

多复制一份数据：

\[
+\text{register/SMEM usage}
\]

但可能换来：

\[
-\text{communication}
\]

\[
-\text{reload}
\]

\[
-\text{shuffle}
\]

。

于是：

\[
\boxed{
Optimal layout 是否应该同时决定 placement 和 replication degree？
}
\]

这对 GQA 特别有意思。

例如：

\[
H_q/H_{kv}=8
\]

一个 KV head 会被多个 Q heads 使用。

我们可以选择：

\[
\text{share one KV representation}
\]

或：

\[
\text{replicate across warp groups}
\]

。

因此最优 replication：

\[
R^*
=
f(
H_q/H_{kv},
D,
S_{kv},
register\ pressure,
SMEM,
architecture
)
\]

。

这个方向和你的 KV layout 项目非常吻合。

证据：**中强**。  
潜在科研价值：**很高**。

---

# 8. 新问题七：Fusion Boundary 和 Layout 能否分开优化？

这个问题是从 TVM / CUTLASS / Hexcute 的方法区别直接来的。

TVM 大体路径是：

\[
Relax\ fusion
\rightarrow
TIR\ task
\rightarrow
MetaSchedule
\]

MetaSchedule 的 task 本身来自已经抽取/融合后的 TIR workload。citeturn415539search0

CUTLASS 的 Collective / Kernel API 则明确是 mainloop fusion、epilogue fusion、back-to-back GEMM 等优化的 composition point。citeturn862114search0

Hexcute 又明确让 programmer 控制 dataflow，然后在该 dataflow 内综合 layout。citeturn436659search0

于是这些设计实际上普遍有：

\[
\boxed{
先确定 dataflow/fusion，
再优化 layout
}
\]

的倾向。

真正科学问题：

\[
\boxed{
Fusion\ boundary 和 optimal layout 是否具有 rank-changing coupling？
}
\]

例如：

Unfused：

\[
GEMM_1
\rightarrow global
\rightarrow SiLU
\rightarrow GEMM_2
\]

可能希望：

\[
L_1=L_{\rm coalesced-store}
\]

而 fused：

\[
GEMM_1
\rightarrow registers/SMEM
\rightarrow SiLU
\rightarrow GEMM_2
\]

可能完全不需要这个 layout。

因此：

\[
L^*_{\rm fused}
\neq
L^*_{\rm unfused}
\]

并不奇怪。

值得研究的不是这个事实本身，而是：

> **什么时候 fusion decision 可以和 layout optimization 分开？**

适合：

- RMSNorm→QKV
- QKV→RoPE→Attention
- SwiGLU
- Attention→O projection
- DiT AdaLN→Linear
- GEGLU

证据：**强**。

---

# 9. 新问题八：低精度时代，“tensor layout”是否已经必须包含 metadata layout？

这是一个非常新的问题，而且我认为会越来越重要。

传统 layout 假设主要对象是：

\[
TensorValues
\]

但 NVFP4/MXFP4/MXFP8 等 block-scaled formats 实际有：

\[
(Data,\ Scale,\ ZeroPoint,\ Metadata)
\]

。

CUTLASS Blackwell primitives 已经要求 block-scale MMA 的 scale factors 有专门 layout，并且 instruction compatibility 与 `scale_vec_size`、K dimension 等相关。citeturn972082search1

FlashInfer 的 NVFP4 KV cache 更明显：K/V data 有 NHD/HND，而 scale tensor 又有额外要求；例如 TRT-LLM backend 的 V scale 使用 4-token interleaved layout，NHD 甚至可能触发 data 和 scale 一起 transpose/copy。citeturn267638search1turn267638search5

vLLM TurboQuant 更直接把 K+V 压缩进一个：

\[
[num\_blocks,block\_size,H,slot]
\]

的 combined slot。citeturn972082search0

所以科学问题是：

\[
\boxed{
现代 quantized tensor 的 layout 应该是单 tensor layout，
还是 data+metadata 的 coupled layout graph？
}
\]

传统：

\[
L_D
\]

可能已经不够。

应该研究：

\[
(L_D,L_S,L_M)
\]

之间的 coupling。

Hypothesis：

\[
\boxed{
data-optimal layout
\neq
(data+scale)-optimal layout
}
\]

并且在：

- FP4 KV cache
- mixed-type MoE
- block-scaled GEMM

中差异会很大。

证据：**非常强**。  
现代 GPU relevance：**非常高**。

我认为这个方向比普通 FP16 GEMM layout optimization 更有新意。

---

# 10. 新问题九：为什么 Persistent KV Layout 必须是“整个模型一个”？

这是 vLLM 当前设计直接暴露出来的问题。

vLLM 当前明确：

> Resolve one KV cache layout for the whole model. citeturn885075search0

而模型本身可以有多个 KV-cache groups、不同 attention kinds、mixed page sizes，甚至不同 backend；`KVCacheLayout` 还必须满足 block-compact 等 packing 条件。citeturn885075search4turn885075search8

同时 NIXL 有自己的 transfer preference：

\[
LBHNC
\]

用于更好的 transfer performance，而 LBNHC 有不同 TP 能力；现在甚至有 experimental：

\[
LBHNC\leftrightarrow LBNHC
\]

local permute。citeturn885075search1

所以比之前 P4 更准确的问题应该是：

\[
\boxed{
Persistent layout 最合适的 optimization granularity 是什么？
}
\]

候选可能是：

\[
Model
\]

\[
KV\ Cache\ Group
\]

\[
Layer
\]

\[
Attention\ Type
\]

\[
Page
\]

\[
Producer/Consumer
\]

。

这比简单：

> dynamic KV layout

更基础。

### Hypothesis

我比较看好：

\[
\boxed{
Per\text{-}group\ layout
}
\]

可能已经能获得大部分 heterogeneous-layout benefit，

同时避免 per-request/per-phase conversion 的巨额开销。

也就是说研究的不是：

\[
static\ vs\ dynamic
\]

而是：

\[
\boxed{
uniformity\ granularity
}
\]

。

证据：**非常强**。  
和 KV layout 项目的匹配度：**最高之一**。

---

# 11. 新问题十：Page Size / Allocator Geometry 和 Layout 能否独立优化？

这个比“改变 KV layout 顺序”要更科学。

FlashInfer 的 paged KV 同时定义：

\[
page\_size
\]

和：

\[
NHD/HND
\]

，具体 backend 还有 page-size restrictions。citeturn267638search0turn267638search1

vLLM 也有：

\[
block\_size
\]

以及 block-compact、block-outermost 等 physical layout property；NIXL 当前甚至开始支持受限的 heterogeneous P/D block sizes。citeturn885075search4turn885075search1

因此：

\[
\boxed{
Page geometry
\times
within-page layout
\times
kernel task mapping
}
\]

实际可能是同一个问题。

原因是 page size 会改变：

\[
\text{tokens per contiguous region}
\]

\[
\text{page-table pressure}
\]

\[
\text{TMA/load transaction shape}
\]

\[
\text{fragmentation}
\]

\[
\text{reuse}
\]

。

真正问题：

\[
\boxed{
Paged tensor layout 的 optimum
能否被分解为
“先选 page size，再选 layout”？
}
\]

我认为这比单纯 HND/NHD 比较更有科研价值。

证据：**强**。

---

# 12. 新问题十一：Local Primitive Dispatch 是否具有 compositional optimality？

TIRx 是非常好的 motivation。

它会对一个 primitive：

\[
Op + Layouts + Scope + Target
\]

做 local dispatch，选择一个 registered implementation。citeturn415539search8turn415539search9

这个设计非常干净。

但一个 kernel 实际上是：

\[
Copy_1
\rightarrow MMA
\rightarrow Reduce
\rightarrow Copy_2
\]

。

即使：

\[
I_i^*
=
\arg\min I_i
\]

对每个 primitive 都局部最好，

仍然不一定：

\[
(I_1^*,I_2^*,I_3^*)=
\arg\min T_{\rm kernel}
\]

。

因为某个 implementation 的 output layout 可能影响下一个 primitive。

Hexcute恰好采取另一条路线：通过 tensor layout constraint 在操作之间传播信息。fileciteturn80file0L1-L7

Triton也需要 `RemoveLayoutConversions` 去修正不同 local passes 的 interaction。fileciteturn67file0L1-L7

所以这是非常干净的科学问题：

\[
\boxed{
Primitive-local dispatch
在什么条件下具有 compositional optimality？
}
\]

它实际上是 Q1 的一个更具体、更可验证版本。

证据：**非常强**。

---

# 13. 新问题十二：Layout Optimization 有多少能跨 GPU generation 迁移？

这个证据稍弱一些，但值得作为后续方向。

A100：

\[
mma.sync,\ cp.async
\]

H100：

\[
WGMMA,\ TMA,\ mbarrier
\]

Blackwell：

\[
tcgen05,\ TMEM,\ multi\text{-}CTA
\]

导致 layout constraints 和 performance tradeoff 明显变化。

CUTLASS/CuTe 有 architecture-specific atoms/builders；TIRx primitive dispatch 同样 target-specific；Hexcute instruction registry 和 latency model 也依赖 target；Gluon当前甚至有大量 Blackwell-specific `TensorMemoryLayout` constraints。citeturn862114search2turn415539search8turn928928search3

所以问题是：

\[
\boxed{
Layout optimum 中哪些特征是 architecture-invariant，
哪些必须重新搜索？
}
\]

例如：

\[
L^*_{H100}
\rightarrow
L^*_{B200}
\]

的 rank correlation。

如果 layout family 可以迁移，而：

\[
instruction/pipeline
\]

需要重新优化，

这对未来搜索空间 warm-start 非常重要。

但目前我把它列为 **A-/B+**，低于前面几个问题，因为已有证据主要证明“硬件不同”，还没有证明“layout search transfer 是核心瓶颈”。

---

# 14. TileLang 给了另一个值得注意的信号，但我暂时不把它单独升格成科研问题

TileLang 明确采用 `LayoutInference` 去推导 `T.Fragment` 的 register allocation/layout。fileciteturn92file1L22-L50

近期又出现一些真实案例：

- 同一 attention-like tensor 被推导出两个不同 layout；
- warp specialization 改变 thread range 后 fragment layout inference 冲突；
- 大 logical domain / irregular width 下 solver/prover 找不到 layout。citeturn356325search3turn356325search4turn356325search14

这些首先是工程 bug，不能直接写成：

> layout inference 不可行。

但是它们支持一个更大的科学问题：

\[
\boxed{
Layout constraints 的局部推导结果什么时候能够全局一致？
}
\]

这可以归入前面的：

- Storage–Task Mapping factorization；
- Primitive compositionality；
- irregular shape；
- pipeline/layout coupling。

这样更科学，而不是以 bug 作为论文问题。

---

# 15. 现在我会重新排列科研问题优先级

和上一轮相比，我会降低几个太泛的问题的重要性。

| 科研问题 | 源码/文档证据 | 科学性 | KV/LLM 相关性 | 我目前优先级 |
|---|---:|---:|---:|---:|
| Storage layout vs task mapping factorization | ★★★★★ | ★★★★★ | ★★★★★ | **S** |
| Irregular/non-power-of-two layout | ★★★★★ | ★★★★☆ | ★★★★★ | **S** |
| Layout × pipeline/warp specialization coupling | ★★★★★ | ★★★★★ | ★★★★★ | **S** |
| Data + quant metadata co-layout | ★★★★★ | ★★★★★ | ★★★★★ | **S** |
| Persistent-layout granularity | ★★★★★ | ★★★★★ | ★★★★★ | **S** |
| Layout transition placement | ★★★★★ | ★★★★☆ | ★★★★★ | **A+** |
| Page geometry × physical layout × task mapping | ★★★★☆ | ★★★★★ | ★★★★★ | **A+** |
| Layout equivalence/canonicalization | ★★★★★ | ★★★★★ | ★★★★☆ | **A+** |
| Replication as layout variable | ★★★★☆ | ★★★★★ | ★★★★★ | **A+** |
| Fusion × layout coupling | ★★★★☆ | ★★★★☆ | ★★★★☆ | **A** |
| Primitive-local dispatch compositionality | ★★★★★ | ★★★★★ | ★★★★☆ | **A+** |
| Cross-GPU layout transferability | ★★★☆☆ | ★★★★☆ | ★★★★☆ | **B+** |

---

# 16. 对之前几个问题，我会做如下调整

之前的：

\[
\boxed{BoundaryRegret}
\]

我现在不会把它直接当论文题目。它太宽。

它更适合作为**统一评价指标**：

\[
R_B
\]

去测试上面具体的：

- layout/pipeline boundary；
- layout/fusion boundary；
- storage/task-mapping boundary；
- runtime/kernel boundary。

这样更有说服力。

之前的：

\[
\boxed{CoverageRegret}
\]

同样更像实验方法。

除非我们最后发现一个非常明确的机制，例如：

> power-of-two candidate generator 系统性丢失 irregular-shape optimum，

这时真正论文问题应该叫：

\[
\textbf{Irregular-Shape Layout Synthesis}
\]

而不是：

\[
\textbf{Candidate Coverage}
\]

。

之前的 cost-model / Selection Regret 我也会降级。

只有先证明：

\[
\text{好 candidate 已经存在}
\]

但现有 cost model 在某类 workload 上持续 mis-rank，

并且 mis-ranking 能归因到：

\[
bank\ conflict,\ pipeline,\ occupancy,\ldots
\]

才值得单独做。

---

# 17. 如果现在让我选“最值得进一步证实”的五个

我会选择下面五个，而不是之前的 Q1/Q2/P4 组合：

\[
\boxed{\textbf{A. Storage Layout–Task Mapping Separability}}
\]

研究 layout 与 execution mapping 是否真的可分。

\[
\boxed{\textbf{B. Layout–Pipeline Coupling}}
\]

研究固定 pipeline 后综合 layout 是否会系统性错失 optimum。

\[
\boxed{\textbf{C. Irregular-Shape Layout Synthesis}}
\]

研究 power-of-two/hierarchical assumptions 对现代模型 irregular dimensions 的实际限制。

\[
\boxed{\textbf{D. Quantized Data–Metadata Co-Layout}}
\]

研究 FP4/MXFP 等时代 data、scale、metadata 是否必须联合布局。

\[
\boxed{\textbf{E. Persistent KV Layout Granularity}}
\]

研究整个 model 一个 layout、per-group、per-layer、per-consumer 之间真正合理的粒度。

这五个有一个好处：

它们都可以直接指出**现有框架之间互相矛盾的设计选择**，而不是我们主观猜某个 framework“不够好”。

这正是我认为目前更可靠的科研选题来源。

---

# 18. 我建议下一轮不要立即做代码，而是专门做一次“问题真实性审查”

尤其针对这五个 S 级问题，下一轮可以严格建立：

\[
\boxed{
Existing\ Evidence
\rightarrow
Implicit\ Assumption
\rightarrow
Competing\ Framework
\rightarrow
Counterexample
\rightarrow
Falsifiable\ Hypothesis
}
\]

例如 Storage–Task Mapping：

```text
TIRx:
storage layout 与 work partition 分离
        │
        ▼
隐含假设：
storage + scope + instruction 足以很好地推导 mapping

        vs

Hexcute:
task mapping 与 layout 联合 inference

        vs

CuTe:
layout/tiled operation 深度耦合
```

然后我们专门问：

> 三种设计谁在哪些 workload 上成立？

Layout–Pipeline：

```text
Hexcute/TIRx:
pipeline 显式

        vs

Tawa:
warp specialization 自动化
```

Quantized co-layout：

```text
传统 layout:
data tensor

        vs

Blackwell/FP4:
data + scale-factor + metadata
```

Persistent KV：

```text
vLLM:
one layout for whole model

        vs

实际：
multiple attention backends
multiple cache groups
NIXL transfer
FP4 scale layout
```

这种“**已有设计之间的矛盾 → 科研假设**”我认为比继续从单个框架里找 bug 更可靠。

下一轮我建议就按这五个问题逐个做这种审查，并主动查找已有论文是否已经把它解决掉，避免我们最终选到一个“看起来新、实际上已有工作”的问题。

## 用户

我指的科研问题是 抨击当前框架中layout优化的一些假设和方法策略上的不足，越多框架有这个问题越好，但是需要区分这种问题是否是一个值得研究的trade-off（这是我指的科研问题：所有框架都有这个问题，但是都没有确定的答案）， 而不是一个确定性答案。尤其是这些框架的解决策略 遇到现代大模型的常见子图时。所以请继续分析，前面的回答仍然需要调整。另外建议你再就好好的扒一扒不同框架的实现源码，总结他们的区别与共同点，尽可能粒度更细，而不是粗略的回答，每一个框架优化了什么。 这样才方便你总结核心问题。

## 模型思考摘要

梳理框架权衡
梳理了框架权衡

## 模型回复

这次我会把前面的“问题列表”整体重构：不再把某个框架的缺口直接叫科研问题，而是找**多个框架都必须做取舍、但没有统一最优答案的 layout trade-off**。我会先把 SGLang、vLLM、CUTLASS/CuTe、Triton、TVM/TIRx、Hexcute 的源码决策链拆到更细，再从这些共同决策中提炼问题，并明确哪些只是工程问题、哪些才值得研究。

## 模型工具调用参数

{"paths":["GitHub"],"query":"search"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索多框架自动调度与布局解析实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 vLLM 解析器实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取文件代码片段

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 SGLang 注意力后端解析源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

缩小 SGLang 默认注意力后端搜索

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 SGLang 内存池布局与页面置换

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 Triton 编译流水线与优化 Pass 顺序

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 accelerate_matmul Python 后端实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Python 后端中的 add_coalesce

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 NVIDIA 编译器 Pass 流程

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 中的布局转换函数

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 TVM 元调度张量核心规则源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 MultiLevelTilingTensorCore 相关代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Apache TVM 中 TIR 布局实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 ResolveAuto 与代价模型实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

比较 Hexcute 自动注解实例化实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 CUTLASS GEMM API 参数文档源代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 CollectiveBuilder 灵活性与构建权衡

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 vLLM KV Cache 布局实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 GitHub 文件内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 SGLang 内存池布局

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 SGLang KV 缓存布局方法

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索并定位 vectorized_5d 缓冲区实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索MetaSchedule搜索与代价模型资料

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Apache TVM 中的 EvolutionarySearch 类

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索XGBoost成本模型实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

研究并比较各框架自动调度实现

## 模型思考摘要

研究并比较多框架自动调度、布局与编译实现

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 2 个网站

## 模型思考摘要

细化源码比较
细化了源码比较

## 模型思考摘要

已搜索 2 个网站

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Triton 编译器核心源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 NVIDIA 后端编译器源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 TVM、CUTLASS 与 SGLang 源码流程

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取精确的注意力后端配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

评估注意力后端选择与调度抽象

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析TVM多级分块调度机制

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 TVM RewriteLayout 与 VerifyGPUCode 源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 TVM RewriteLayout 源码细节

## 模型思考摘要

检索并分析 Triton、NVIDIA、TVM 等注意力编译调度源码

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 docs.flashinfer.ai

## 模型思考摘要

完善权衡框架
完善了权衡框架

## 模型思考摘要

厘清研究主线
厘清研究主线

## 模型思考摘要

已搜索 1 个网站

## 模型思考用时

思考了 5m 33s

## 模型回复

需要调整。你这次把“科研问题”的标准说得更准确了：

> 不是“框架缺了某个优化，所以把它补上”，而是**多个框架为了让 layout 优化可实现，都不得不采用某种假设或策略；这些策略两边都有合理性，在现代大模型常见子图上不存在显然支配另一方的答案，因此形成值得研究的 trade-off。**

按这个标准，前面我给出的 `Boundary Regret / Coverage Regret / Selection Regret` 更适合作为**诊断指标**，不应该直接作为核心科研问题。“Hexcute 没建模 bank conflict”“Triton 有 convert_layout”“vLLM 只有若干 KV layout”也都只是证据或 symptom，而不是科研问题本身。

我重新扒了一轮当前源码。下面先把框架比较做细，再从这些源码里的共同假设推出更合适的科研问题。

---

## 一、先把这些框架真正做的决策拆开

我建议以后不要再用“这个框架优化 layout / 那个框架也优化 layout”这种分类。把一次 GPU 实现写成：

\[
D=
\{
L_G,L_P,L_S,L_R,
T,
I_C,I_M,I_R,
X,
P,W,
A
\}
\]

其中：

\[
\begin{aligned}
L_G &: \text{global/persistent physical layout}\\
L_P &: \text{page/block geometry}\\
L_S &: \text{shared-memory layout/swizzle}\\
L_R &: \text{register / thread-value layout}\\
T &: \text{tile + thread/warp mapping}\\
I_C &: \text{copy instruction}\\
I_M &: \text{MMA instruction}\\
I_R &: \text{reduction/communication method}\\
X &: \text{layout transition / conversion / replication}\\
P &: \text{pipeline / stage}\\
W &: \text{warp specialization / role assignment}\\
A &: \text{allocator / persistent-state organization}
\end{aligned}
\]

然后才看每个框架决定其中哪一部分。

### 更细的 Decision Ownership Matrix

| 决策 | SGLang | vLLM | CUTLASS/CuTe | Triton classic | Gluon | TVM MetaSchedule | TIRx | Hexcute |
|---|---|---|---|---|---|---|---|---|
| Persistent physical layout | backend/cache-specific | **显式有限集合** | 外部 | 外部 | 外部 | 可 rewrite | 显式 storage | 外部/程序显式 |
| Page/block geometry | runtime/backend | **runtime/backend** | — | — | — | — | 程序 | 程序 |
| Global-memory stride/layout | cache决定 | cache决定 | **显式** | 程序输入 | **显式** | schedule/rewrite | **显式** | 程序输入 |
| CTA tile | backend | backend | **显式/Builder** | meta-program | **显式** | **搜索** | 程序 | 程序/tune |
| Cluster tile | backend | backend | **显式** | `num_ctas`/compiler | 显式 | schedule | scope/program | tune/program |
| Thread/value mapping | backend | backend | **TiledMma/Copy 显式** | **compiler encoding** | **程序显式** | schedule 间接决定 | dispatch 派生 | **constraint inference** |
| Register layout | backend | backend | 显式/atom | compiler | **显式** | 间接 | storage + dispatch | **推导** |
| SMEM layout | backend | backend | **SmemLayoutAtom** | compiler | **显式** | cache/reuse | **显式** | **推导** |
| SMEM swizzle | backend | backend | 显式 | compiler | 显式 | tensorization/schedule | implementation | **单独 bank-conflict pass** |
| GMEM→SMEM copy | backend | backend | **TiledCopy/CopyAtom** | compiler | 显式 | vector/cooperative fetch | primitive dispatch | **instruction inference** |
| SMEM→REG copy | backend | backend | **SmemCopyAtom** | compiler | 显式 | tensor intrinsic | primitive dispatch | **instruction inference** |
| MMA instruction | backend | backend | **MmaAtom/TiledMma** | AccelerateMatmul | 显式 | tensor intrin | dispatch | **constraint/instruction selection** |
| Reduction mapping | backend | backend | kernel/program | OptimizeThreadLocality | 显式 layout | reduction rules | primitive dispatch | layout inference |
| Layout conversion | backend-specific | backend/connector | 程序显式 | **插入+消除+rematerialize** | **程序显式** | RewriteLayout/cache | `permute_layout` | Rearrange/inference |
| Replication | runtime/cache | cache | layout composition | encoding/rematerialize | explicit layout | schedule/cache | **Replica 是 layout 一等结构** | TV-layout 可表示 |
| Pipeline stages | backend | backend | **StageCount** | `num_stages` + compiler | **显式** | schedule rule | **显式** | **程序显式** |
| Warp specialization | backend | backend | **schedule tag** | compiler auto | **程序显式** | schedule-dependent | **程序显式** | **程序显式** |
| Candidate generation | backend registry | backend/layout enum | expert + Builder | passes + user configs | programmer | ScheduleRules | registered variants | **constraint inference** |
| Legality | capability rules | support intersection | C++ types/templates | verifier/encodings | compiler | postproc/VerifyGPU | predicates | **constraint unification/backtracking** |
| Performance selection | 手工规则 | preference | expert/rules/profiler | heuristic + autotune | programmer + autotune | **XGB + measurement** | priority | **analytic cost model** |
| Runtime workload visibility | **高** | **高** | 极低 | 低 | 低 | task/shape | 低 | 低 |
| 跨 operator 可见性 | serving级 | cache/model级 | collective/epilogue | kernel 内 | kernel 内 | Relax→TIR | kernel 内 | tile-program 内 |

这张表已经暴露出一个重要事实：

\[
\boxed{
没有任何一个框架是在“优化 layout”；
它们是在选择不同的 decision decomposition。
}
\]

---

# 二、源码进一步说明这些方法差异不是表面差异

### SGLang：先选 algorithm/backend，再接受 backend 所要求的 layout

SGLang 当前 `get_default_attn_backend()` 本质是规则系统。它会检查 GPU generation、MHA/MLA、speculative decoding、attention sink、asymmetric KV 等，然后在 FA3、TRT-LLM、FlashInfer、AITER、Triton 等 backend 间决策。fileciteturn64file0L1-L7

而且 prefill/decode 是两个独立字段。`resolve_attention_backend_strs()` 返回：

\[
(B_{\rm prefill},B_{\rm decode})
\]

如果不同，`_build_resolved_backend()` 会构造 `HybridAttnBackend`。fileciteturn124file0L1-L10

Persistent KV 也已经出现 backend-specific physical representation。`memory_pool.py` 明确区分 legacy NHD 和 ROCm/AITER 的 `vectorized_5d` SHUFFLE layout；源码甚至说明没有对应 consumer kernel 时 SHUFFLE layout 没有意义。fileciteturn114file0L1-L2 fileciteturn115file0L1-L17

因此 SGLang 的策略是：

\[
\boxed{
\text{workload/model/hardware}
\rightarrow
\text{backend}
\rightarrow
\text{backend-compatible representation}
}
\]

不是从完整 layout space 做优化。

---

# 三、vLLM：比 SGLang 更明确地把 persistent layout 变成 compatibility problem

vLLM 当前的 `KVCacheLayout` 不是旧资料里单一 PagedAttention layout，而是逻辑：

\[
[L,B,H,N,C]
\]

到六种 physical order 的枚举：

\[
LBHNC,LBNHC,LHBNC,BLHNC,BLNHC,BHLNC.
\]

而且源码显式定义：

- layer compact；
- block contiguous；
- block compact；
- block outermost。

fileciteturn113file0L1-L13

每个 backend 返回 ordered：

```text
supported_kv_cache_layouts()
```

系统先做：

\[
\bigcap_i Supported_i
\]

若多个 backend preference 不同，再按“多少 backend 把它列第一”排序。fileciteturn96file0L1-L2

如果 mixed HNC specs 出现，又只留下 `block_compact` layout；connector 也可以提供 preference；最后：

> `Resolve one KV cache layout for the whole model.`

并且结果一旦记录不能随意改变。fileciteturn96file0L1-L2

与此同时 attention backend selection 输入已经包括 head size、dtype、KV dtype、block size、MLA、sink、sparse、sliding window、KV connector、PCP、DCP、adaptive verification 等信息。fileciteturn70file0L1-L7

所以 vLLM 做的是：

\[
\boxed{
\text{rich runtime constraints}
\rightarrow
\text{compatibility set}
\rightarrow
\text{preference}
}
\]

而不是：

\[
\arg\min_L T(L).
\]

这里已经隐藏着几个很强的 trade-off，后面会回来。

---

# 四、CUTLASS/CuTe：几乎把所有耦合都暴露给专家

CUTLASS 的 `CollectiveMma` 不是单纯选择 tile。

它直接接受：

\[
Stride_A,Stride_B,
TiledMma,
GmemTiledCopy_{A/B},
SmemLayoutAtom_{A/B},
SmemCopyAtom_{A/B}
\]

以及 transform。fileciteturn110file0L1-L2

dispatch policy 又把：

\[
Stages,\ ClusterShape,\ KernelSchedule
\]

组合起来。

Hopper 甚至显式提供：

```text
KernelTmaWarpSpecialized
KernelTmaWarpSpecializedPingpong
KernelTmaWarpSpecializedCooperative
```

等 schedule family。fileciteturn110file0L1-L2

因此 CUTLASS 的基本思路是：

\[
\boxed{
L_G+L_S+T+I_C+I_M+P+W
}
\]

由 expert coherent construction 一起决定。

`CollectiveBuilder` 虽然有 `StageCountAuto`、`KernelScheduleAuto`，但它仍然是在 architecture-specific template specialization/rule 中选择，而不是 general measured search。源码甚至能看到某些 low-precision builder 因 SMEM 容量不足，直接采用特殊 stage-count 规则。fileciteturn94file0L1-L16 fileciteturn125file1L21-L31

CUTLASS 代表一个极端：

\[
\boxed{
\text{最大控制能力}
\leftrightarrow
\text{最大 expert burden}
}
\]

---

# 五、Triton classic：不是一次 layout 决策，而是一串互相修正的局部 pass

当前 NVIDIA backend 的 `make_ttgir()` 顺序非常有信息量。

TTIR 转 TTGIR 后，大致经过：

\[
\begin{aligned}
&Coalesce\\
\rightarrow& RemoveLayoutConversions\\
\rightarrow& OptimizeCTALocality\\
\rightarrow& RemoveLayoutConversions\\
\rightarrow& OptimizeThreadLocality\\
\rightarrow& AccelerateMatmul\\
\rightarrow& RemoveLayoutConversions\\
\rightarrow& OptimizeDotOperands\\
\rightarrow&\cdots\\
\rightarrow& Pipeline/WarpSpecialize.
\end{aligned}
\]

这不是推测，当前 `third_party/nvidia/backend/compiler.py` 就是这个 pass pipeline。fileciteturn122file0L1-L2

而各个 pass 的目标又不同：

`Coalesce`：

> 给 load/store 换成 cache-friendly layout，同时插 conversion。

`AccelerateMatmul`：

> 改 dot input/output layout，让它适配 tensor cores。

`OptimizeThreadLocality`：

> 为 reduction/gather 降低 cross-thread communication。

`RemoveLayoutConversions`：

> 在 memory-friendly BlockedEncoding 与 MMA-friendly encoding 等之间重新改写、rematerialize。

fileciteturn66file0L1-L7 fileciteturn67file0L1-L7

这说明 Triton 当前方法本身就在承认：

\[
\boxed{
L_{\rm memory}
\neq
L_{\rm MMA}
\neq
L_{\rm reduction}
}
\]

但解决方式是 staged local rewriting + repair。

此外 `CUDAOptions` 中 `num_warps` 必须是 2 的幂，`num_stages` 等成为外部 meta-parameter，而很多内部 layout choice 并不是 autotune variable。fileciteturn122file0L1-L2

这个实现对后面的科研问题极其重要。

---

# 六、Gluon：把 classic Triton 隐藏的 layout ownership 重新交给 programmer

Gluon 则走相反方向。

程序员显式写：

\[
sizePerThread,
threadsPerWarp,
warpsPerCTA,
order.
\]

官方文档甚至明确说明：

> Gluon 没有 canonical layout representation；不同 layout expression 可以表示相同 element mapping。

同时文档直接展示 layout 会影响：

- global memory coalescing；
- reduction 中的 inter-thread communication；
- shared-memory bank conflict；
- layout conversion cost。

citeturn922154search5

Warp specialization 又要求 programmer 明确分 producer/consumer partition、warp 数、register 数、mbarrier；官方教程明确说 register allocation 很难估计，经常要 profiling/trial-and-error/autotuning，并指出 warp specialization 会增加同步、SMEM 和 register pressure。citeturn922154search2

所以 Gluon 给出的答案其实是：

\[
\boxed{
当 compiler heuristic 不可靠时，
把关键 trade-off 暴露给 expert。
}
\]

这和 Hexcute/TIRx 又不一样。

---

# 七、TVM MetaSchedule：先人为设计“允许搜索的结构”，然后真正测量

`MultiLevelTiling` 的参数非常值得看。

它明确固定/暴露：

\[
structure,
tile\_binds,
max\_innermost\_factor,
vector\_load\_lens,
reuse\_read,
reuse\_write.
\]

TensorCore 版本再增加：

\[
intrin\_groups
\]

和：

\[
use\_software\_pipeline.
\]

fileciteturn126file0L1-L10

然后 MetaSchedule 的默认逻辑是：

\[
ScheduleRules
\rightarrow
DesignSpace
\rightarrow
EvolutionarySearch
\rightarrow
CostModel
\rightarrow
HardwareMeasurement.
\]

官方当前文档明确说默认 `EvolutionarySearch` 由 cost model 引导，而默认模型可以是 XGB；候选最终实际跑在目标硬件上。fileciteturn118file5L81-L95 fileciteturn119file0L1-L28

还有 `VerifyGPUCode`、`RewriteTensorize` 等 postproc；`RewriteLayout` 会针对 `layout_free_buffers` 引入 cache-read/layout rewrite。fileciteturn127file1L16-L30 fileciteturn128file0L1-L11

因此 TVM 的赌注是：

\[
\boxed{
可以先用专家规则把无限空间压成一个“有希望的设计空间”，
再用测量解决性能不可预测性。
}
\]

这和 Hexcute“先通过 constraints 求合法 layout”、CUTLASS“专家直接构造”形成了非常好的对照。

---

# 八、TIRx：layout 和 execution mapping 被有意拆开

TIRx 是目前最明确表达这一哲学的系统之一。

它说：

> layout 是 storage contract，而不是 work-partitioning interface。

layout 描述 logical tensor 到 global/shared/register/TMEM/lane/warp 等资源的映射；primitive dispatch 再使用：

\[
OperandLayouts+
ExecutionScope+
Target
\]

生成：

\[
thread\ partition+
loops+
addressing+
instruction.
\]

citeturn922154search1

dispatch 实现也很具体：

每种 primitive/target 注册多个：

\[
(variant,priority,predicates,implementation)
\]

，然后：

1. 按 priority 排序；
2. 检查 predicates；
3. 采用第一个成功的 implementation。

fileciteturn107file0L1-L38 citeturn922154search0

也就是说它不是 performance search，而是：

\[
\boxed{
storage semantics
\rightarrow
legal/priority dispatch.
}
\]

而 pipeline state、barrier protocol、warp role selection 等仍然明确留在程序中。citeturn922154search1

---

# 九、Hexcute：把 legality propagation 做得最系统，但刻意不接管 dataflow/pipeline

Hexcute 的 auto annotation 实现实际上分四步：

\[
LogicalShape
\rightarrow
LogicalLayout
\rightarrow
TaskMapping
\rightarrow
MemoryLayout.
\]

Copy 的 TV mapping 满足：

\[
f\circ p^{-1}=g\circ q^{-1}.
\]

MMA 则约束：

\[
f_{A,m}=f_{C,m},\qquad
f_{B,n}=f_{C,n},\qquad
f_{A,k}=f_{B,k}.
\]

如果一个 instruction family 有多个合法 variant，可以 DFS 枚举，也可以 heuristic/beam search prune，然后再用 cost model 选。fileciteturn109file0L1-L2

Shared-memory layout 又必须统一所有 consumer copy 的 constraints；冲突则回溯。fileciteturn109file0L1-L2

Instruction selection 再看：

\[
scope,
alignment,
thread/value\ layout
\]

去匹配 `ldg/stg/lds/sts/ldmatrix/cp_async/TMA/mma/wgmma` 等。fileciteturn82file0L1-L7

而 bank conflict 又是后面的独立 pass，通过枚举 swizzle 去处理。fileciteturn81file0L1-L7

性能 model 大致：

\[
\#instruction\times latency
\]

并假设 copy/MMA 充分 overlap；源码明确写 bank conflict、`cp_async_wait_group`、`mbarrier` 等尚未完整进入 model。fileciteturn79file0L1-L7

Hexcute 论文自己把选择表述得非常清楚：

> 自动 layout synthesis，但显式控制 dataflow 和 pipelining。citeturn148696search0

这不是“没做完”，而是一种刻意的 search-space boundary。

---

# 十、这次重新总结：这些框架真正共有的不是某种 layout 技术，而是六个共同假设

源码看完以后，我认为最值得“抨击”的并不是一个具体 pass，而是下面这些假设。

| 共同假设 | 框架为什么这么做 | 潜在问题 |
|---|---|---|
| 先把巨大 layout space 限定成一个结构化子空间 | 搜索否则不可承受 | 是否把重要 optimum 排除？ |
| 同一个 tensor 尽量维持稳定 representation | conversion/metadata/内存成本低 | 多 consumer preference 冲突 |
| regular tile/page/layout family 足够 | 易映射 TensorCore/TMA/SIMT | LLM shape 越来越 irregular |
| layout、instruction、pipeline 可以一定程度分阶段处理 | 模块化、可编译、搜索空间小 | Hopper/Blackwell coupling 越来越强 |
| local cost/heuristic 可以代表整体性能 | full-system optimization 太贵 | Attention/MoE 多阶段相互影响 |
| 静态/离线决策可以代表 runtime workload | 避免在线 tuning/repacking | LLM batch/seq/phase 高度动态 |

这六个假设没有一个是“明显错误”。

也没有一个是“显然正确”。

这才是科研问题来源。

---

# 十一、科研问题 1：稳定统一的 Layout vs Consumer-Specific Layout

我现在把它放在第一位。

当前框架实际上在不同粒度上都要回答：

> 一个 tensor 应该拥有一个稳定 layout，还是不同 consumer 应该看到不同 layout？

vLLM 走得非常靠“稳定”一侧：

\[
\boxed{\text{whole model one KV layout}}
\]

。fileciteturn96file0L1-L2

Hexcute 对一个 shared tensor 会统一多个 copy consumer 的 constraints，努力找到一个同时兼容的 memory layout。fileciteturn109file0L1-L2

Triton 则更允许 consumer-specific representation：

\[
L_{\rm load}\rightarrow convert
\rightarrow L_{\rm MMA}
\]

然后再尝试删除过多 conversions。fileciteturn67file0L1-L7

FlashInfer 甚至给出了现实反例：普通 FP16 KV 的 NHD/HND 差别可能很小，但 low-precision KV 时 HND 对 GPU kernel 更友好；某些 NVFP4 backend 遇到 NHD 会直接 transpose + contiguous copy 成 HND。citeturn124734search1turn124734search3

真正问题因此是：

\[
\boxed{
\text{Layout specialization 的合理粒度是什么？}
}
\]

可能是：

\[
model,\ layer,\ phase,\ consumer,\ kernel,\ tile.
\]

这不存在确定答案，因为更细粒度带来：

\[
+\text{local performance}
\]

但同时：

\[
+\text{conversion}
+\text{metadata}
+\text{search}
+\text{representation management}.
\]

现代 LLM 最典型的两个场景正好是：

\[
QK^T\rightarrow Softmax\rightarrow PV
\]

和：

\[
KVCache
\rightarrow
\{
decode,\ prefill,\ append,\ transfer,\ prefix\ reuse
\}.
\]

我认为这是 **S 级问题**。

---

# 十二、科研问题 2：一个 compromise layout，还是 Convert，还是 Replicate？

这是上一个问题的下一层，但我认为值得单独提出。

当两个 consumer：

\[
C_1,C_2
\]

分别喜欢：

\[
L_1^*,L_2^*
\]

时，其实系统有三个策略。

\[
\text{Shared:}\quad
L_1=L_2=L_c
\]

\[
\text{Convert:}\quad
L_1^*\rightarrow L_2^*
\]

\[
\text{Replicate:}\quad
\{L_1^*,L_2^*\}
\]

长期同时存在。

现有框架都在不同位置暗含选择：

Triton大量处理 `ConvertLayout` 和 rematerialization。fileciteturn67file0L1-L7

TVM 可以 `CacheRead` 并 rewrite layout，相当于创建新的 representation。fileciteturn128file0L1-L11

Hexcute倾向于对 shared tensor unify constraints。

vLLM 倾向于一个 persistent KV representation。

TIRx 又把 `permute_layout` 本身做成 primitive。citeturn922154search3

真正 trade-off：

\[
J_{\rm shared}
\quad vs\quad
J_{\rm convert}
\quad vs\quad
J_{\rm replicate}.
\]

Replication 可以减少 repeated conversion，但：

\[
MemoryFootprint\uparrow
\]

可能反过来降低：

\[
batch\ capacity,\ KV\ capacity,\ occupancy.
\]

对于 GQA：

\[
H_q/H_{kv}=8
\]

甚至可以研究 KV 是共享一份，还是在 warpgroup/register level 复制以避免通信。

这也是 **S/A+ 级**问题。

---

# 十三、科研问题 3：Regular Layout vs Irregular Workload Fidelity

之前我说“power-of-two layout 问题”，这个表述不够准确。

真正的问题是：

\[
\boxed{
为了得到规则、硬件友好的 mapping，
应该把 workload regularize 到什么程度？
}
\]

所有这些系统都大量依赖 regularity：

CUTLASS：

\[
TileShape,\ ClusterShape
\]

基本是规则 tiled hierarchy。

Triton：

`num_warps` 当前甚至必须是 2 的幂；大量 Tensor program 用 regular block + mask。fileciteturn122file0L1-L2

TVM：

`MultiLevelTiling` 首先选择一个固定 tiling structure。fileciteturn126file0L1-L10

Hexcute：

layout synthesis 仍然建立在 tile/instruction hierarchy 上。

vLLM/SGLang/FlashInfer：

使用固定 page/block geometry。

但现代 LLM 越来越不规则：

\[
\text{ragged sequence lengths}
\]

\[
\text{MoE expert token counts}
\]

\[
H_q\neq H_{kv}
\]

\[
\text{speculative verify with variable q length}
\]

\[
\text{sliding windows}
\]

\[
\text{nonuniform quantization groups}.
\]

FlashInfer本身甚至同时提供 ragged tensor 和 paged KV，因为两个表示面对的 trade-off 不同。citeturn124734search1

所以问题应该写成：

\[
\boxed{
规则化带来的 TensorCore/TMA/vectorization 收益，
什么时候超过 padding、mask、fragmentation 和 load imbalance 的代价？
}
\]

不是：

> “支持 non-power-of-two 就好了。”

那是工程问题。

真正要研究的是一个 phase diagram。

这也是我现在非常看好的 **S 级问题**。

---

# 十四、科研问题 4：Layout Locality vs Parallelism / Resource Pressure

这是一个以前严重低估的 layout 核心 trade-off。

假设让一个 thread 拥有更多连续值：

\[
values/thread\uparrow
\]

通常可以：

\[
coalescing\uparrow,\quad
reuse\uparrow,\quad
cross\text{-}thread\ communication\downarrow.
\]

但同时：

\[
register/thread\uparrow
\]

以及：

\[
occupancy\downarrow.
\]

增加 SMEM buffering：

\[
reuse\uparrow,\quad
pipeline\ overlap\uparrow
\]

但：

\[
SMEM/CTA\uparrow
\Rightarrow
resident\ CTAs\downarrow.
\]

增加 replication：

\[
communication\downarrow
\]

但：

\[
register/SMEM\ footprint\uparrow.
\]

Gluon warp-specialization文档直接说 register allocation 难以估计，warp specialization 会增加同步、SMEM 和 register pressure。citeturn922154search2

CUTLASS 把 `Stages`、cluster 和 tile 当作共同参数，甚至因为 SMEM capacity 对低精度大 tile 做特殊 stage 选择。fileciteturn125file1L21-L31

TVM 的 schedule space也同时包含 tiling、vector-load length 和 read/write reuse。fileciteturn126file0L1-L10

因此更基础的问题是：

\[
\boxed{
一个 layout 应该追求多少 locality，
才不会破坏足够的 execution parallelism？
}
\]

现代 Attention、MoE、SwiGLU 都会触发它。

尤其 Decode Attention：

memory-bound，很可能更喜欢 locality；

大型 Prefill GEMM：

compute-bound，occupancy/latency hiding 又可能更重要。

因此不存在固定答案。

这是 **S 级**问题，而且非常“layout”。

---

# 十五、科研问题 5：Layout–Instruction–Pipeline 到底应该联合优化多少？

这个问题需要从“joint optimization 一定更好”调整成：

\[
\boxed{
哪些 coupling 值得联合优化，
哪些可以安全分解？
}
\]

因为完整联合搜索：

\[
L\times I\times P\times W\times Tile
\]

很容易爆炸。

不同系统已经选择了不同答案。

CUTLASS：

几乎全显式联合 composition。

Triton：

layout 优化、MMA、pipeline、warp specialization 是一串 pass。fileciteturn122file0L1-L2

TVM：

software pipeline 是 `MultiLevelTilingTensorCore` 的一个选项，但 search space 是 ScheduleRule 预先定义的。fileciteturn126file0L1-L10

Hexcute：

\[
Layout+TaskMapping+Instruction
\]

自动；

但：

\[
Dataflow+Pipeline
\]

显式。citeturn148696search0

TIRx：

storage + primitive dispatch自动/半自动；

orchestration/pipeline 显式。citeturn922154search1

TileLang 当前甚至有独立的 `PipelinePlanning` 和 `LayoutInference` pass。fileciteturn92file0L1-L19

这说明行业并没有形成答案。

真正问题：

\[
\boxed{
Performance\ interaction\ graph
\quad
G(D_i,D_j)
}
\]

是不是稀疏的？

例如是否只有：

\[
L_S\leftrightarrow I_C
\]

\[
L_R\leftrightarrow I_M
\]

\[
P\leftrightarrow SMEM
\]

需要 joint search，

而其他变量可以分开？

如果能回答这个问题，就能决定：

\[
\text{什么时候需要 joint optimization}
\]

而不是无条件扩大 search space。

我认为这是 **S 级**核心 compiler 问题。

---

# 十六、科研问题 6：Local Primitive Optimality vs Subgraph Optimality

这个和上一条不同。

上一条研究变量之间 coupling。

这一条研究：

> 一个 primitive/operator 自己最快，是否帮助整个 subgraph？

Triton 是最直接的证据。

它分别有：

- memory coalescing；
- thread locality；
- tensor-core layout；
- layout conversion removal。

这本质就是多个 local objectives 之间的协调。fileciteturn66file0L1-L7

TIRx 的 dispatch 更直接：

每个 primitive 按：

\[
priority+predicate
\]

挑第一个实现。citeturn922154search0

但：

\[
Copy^* \rightarrow MMA^* \rightarrow Reduce^*
\]

未必就是：

\[
\arg\min T_{\rm Copy\rightarrow MMA\rightarrow Reduce}.
\]

现代 Attention 是极强的 counter-pressure：

\[
QK^T
\rightarrow
Softmax
\rightarrow
PV.
\]

第一段喜欢 TensorCore operand layout；

Softmax 喜欢 thread-local reduction layout；

第二次 MMA 又有另一套 operand constraints。

类似问题也存在于：

\[
Gate/Up\ GEMM
\rightarrow
SiLU\times
\rightarrow
Down\ GEMM
\]

以及：

\[
RMSNorm
\rightarrow QKV.
\]

真正问题：

\[
\boxed{
什么条件下 local layout cost 是 compositional 的？
}
\]

如果：

\[
C_{\rm global}
\approx
\sum_i C_i
+
\sum_i C_{\rm convert}
\]

就可以局部优化。

如果 cache、occupancy、pipeline overlap 产生强 interaction，就不行。

这是 **A+/S-**。

---

# 十七、科研问题 7：Hard Structural Pruning vs Performance Coverage

这才是之前 Coverage Regret 背后真正的问题。

几乎所有系统都先用结构知识把空间剪掉：

CUTLASS：

template specialization / legal schedule family。

Triton：

encoding / pass heuristics / supplied autotune configs。

TVM：

ScheduleRules。

TIRx：

registered variants + predicates。

Hexcute：

layout/instruction constraints。

vLLM：

backend-supported layout intersection。

SGLang：

backend registry + model/hardware rule。

也就是说大家都假设：

\[
\boxed{
“明显不值得搜索的配置”
可以在看到真实性能前安全删除。
}
\]

但这个假设的 aggressiveness 不同。

真正 trade-off：

\[
SearchCost\downarrow
\]

versus：

\[
NearOptimalCoverage\downarrow?
\]

Hexcute 特别适合作为一端：它用结构 constraints 大幅减少 legal space。fileciteturn109file0L1-L2

TVM 是另一端：结构 schedule space + 实测搜索。

因此真正科学问题是：

\[
\boxed{
GPU layout space 中，
“合法性/结构信息”与“高性能区域”之间到底有多强相关性？
}
\]

如果合法 constraints 已经非常 predictive，就不需要巨大 autotuning。

如果几乎所有合法 configs 性能差异都巨大且缺乏结构规律，则必须 measurement-heavy。

这是 **A+**。

---

# 十八、科研问题 8：Analytic/Rule-Based Choice vs Empirical Measurement

这个也应该独立于“cost model 不准”。

不同框架已经站在非常不同的位置：

\[
\text{CUTLASS}
\rightarrow
expert/rules
\]

\[
\text{Triton}
\rightarrow
heuristics+\text{optional autotune}
\]

\[
\text{TVM}
\rightarrow
cost model + hardware measurement
\]

\[
\text{TIRx}
\rightarrow
priority/predicate
\]

\[
\text{Hexcute}
\rightarrow
analytic instruction-latency model
\]

\[
\text{SGLang/vLLM}
\rightarrow
runtime rules/preferences.
\]

这不是谁“先进”谁“落后”。

真正 trade-off 是：

\[
\boxed{
performance\ fidelity
\leftrightarrow
tuning/compile/runtime\ overhead
}
\]

。

对于传统固定 GEMM：

花很多时间 tune 可能很划算。

但现代 serving workload 有：

\[
(B,S_q,S_{kv},phase,GQA,page,spec,\ldots)
\]

巨大的动态 shape space。

对每个状态实测显然不可行。

反过来纯 rule 又可能严重 mis-rank。

所以真正问题：

\[
\boxed{
为了正确选择 layout，
需要多少 empirical feedback？
}
\]

不是：

> “做一个更准 cost model”。

这是 **S/A+**。

---

# 十九、科研问题 9：Static Specialization vs Runtime Adaptability

这和上一条有关，但目标不同。

CUTLASS/CuTe 基本在 compile-time construction。

Triton/Tensor kernels 基于 shape/meta 编译或 autotune。

TVM tune 一个 workload/task。

Hexcute synthesis 一个 tile program。

而 vLLM 的 persistent KV layout虽然看到丰富 model/runtime configuration，却最后在 engine 初始化阶段 resolve 一次：

\[
L_{KV}=\text{static after resolution}.
\]

fileciteturn96file0L1-L2

SGLang 已经走得更动态一点：

\[
Backend_{\rm prefill}
\neq
Backend_{\rm decode}
\]

是允许的。fileciteturn124file0L1-L10

因此真正 trade-off：

\[
\boxed{
specialization\ quality
\leftrightarrow
recompile/repack/state\ migration/cache\ management
}
\]

。

不是“dynamic 一定更好”。

KV cache 可能几十 GB。

即使：

\[
L_B
\]

比：

\[
L_A
\]

快 15%，把已有 KV 从 A 转成 B 也可能永远收不回来。

所以要研究：

\[
\text{adaptation gain}
\quad vs\quad
\text{switching/amortization cost}.
\]

现代 LLM serving 里这个问题很强。

---

# 二十、科研问题 10：Kernel-Optimal Layout vs System-Optimal Layout

这是之前没有强调够的。

低层 compiler 通常目标是：

\[
\min T_{\rm kernel}.
\]

但 serving runtime 真正关心：

\[
TTFT,\ TPOT,\ throughput,\ memory\ capacity,\ concurrency.
\]

例如某个 layout：

\[
T_{\rm attention}\downarrow10\%
\]

但因为 replication/extra workspace：

\[
KV\ capacity\downarrow15\%
\]

导致 maximum batch 降低。

最终：

\[
throughput\downarrow.
\]

Warp specialization 同样可能提高单 kernel throughput，但增加 SMEM/register footprint。citeturn922154search2

因此真正问题：

\[
\boxed{
Layout optimization 应该把多少 system-level resource opportunity cost
纳入目标？
}
\]

对 interactive serving：

\[
\min TPOT
\]

和 high-throughput serving：

\[
\max tokens/s
\]

甚至可能选择不同 layout。

这是一个真正没有唯一答案的 trade-off。

我认为对于“KV layout”项目尤其值得重视。

---

# 二十一、科研问题 11：Data Layout vs Quantization-Metadata Layout

这是较专门，但现代模型价值很高。

FP4/NVFP4 以后一个“tensor”实际上越来越像：

\[
(Data,Scale,Metadata).
\]

FlashInfer 当前 NVFP4 KV cache 的 scale tensors 会跟 NHD/HND 一起改变形状；部分 backend 甚至要求 V scale 采用特定 4-token interleaved layout。citeturn124734search0turn124734search3

因此：

\[
L_D
\]

不再能单独优化。

需要考虑：

\[
(L_D,L_S,L_M).
\]

但 interleave metadata 有好处：

\[
address/locality\ simpler
\]

可能也会导致：

\[
independent\ reuse/vectorization\ harder.
\]

分开 metadata 则相反。

真正问题：

\[
\boxed{
低精度 tensor 的“layout unit”到底应该是什么？
}
\]

这是 FP4 KV、block-scaled GEMM、mixed-type MoE 中很好的研究问题。

但覆盖框架数量略少，我会给 **A+** 而不是第一主线。

---

# 二十二、几个之前我认为重要的问题，现在应该降级

| 之前的表述 | 现在的判断 |
|---|---|
| Hexcute cost model 没 bank conflict | 工程缺口；属于“analytic vs empirical”证据 |
| Triton convert_layout 太多 | symptom；真正问题是 shared/convert/replicate 与 local/global objective |
| vLLM whole-model KV layout | 某种策略；真正问题是 specialization granularity |
| Triton power-of-two | 具体限制；真正问题是 regularization vs irregularity |
| Gluon 没 canonical layout | 主要是 representation/canonicalization 工程问题 |
| Layout equivalence class | 有理论兴趣，但目前不如资源/consumer trade-off 重要 |
| Boundary Regret | **评价指标**，不是科研问题 |
| Coverage Regret | **评价指标**，不是科研问题 |
| Selection Regret | **评价指标**，不是科研问题 |
| Storage layout vs task mapping 是否可分 | 应并入“modularity/joint optimization”这一 trade-off |

这个调整我认为是必要的。

---

# 二十三、现在按“跨框架共同性”重新排序

符号：● = 框架直接面对；○ = 主要委托给 backend/下一层，但系统选择会间接影响。

| 科研 trade-off | SGLang | vLLM | CUTLASS | Triton | TVM/TIRx | Hexcute | 现代 LLM 重要性 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stable/shared vs specialized layout | ● | ● | ● | ● | ● | ● | ★★★★★ |
| Shared vs convert vs replicate | ● | ● | ● | ● | ● | ● | ★★★★★ |
| Regularity vs irregular workload | ● | ● | ● | ● | ● | ● | ★★★★★ |
| Locality vs parallelism/resources | ○ | ○ | ● | ● | ● | ● | ★★★★★ |
| Joint L/I/P vs modular optimization | ○ | ○ | ● | ● | ● | ● | ★★★★★ |
| Local primitive vs global subgraph | ● | ● | ● | ● | ● | ● | ★★★★★ |
| Structural pruning vs coverage | ● | ● | ● | ● | ● | ● | ★★★★☆ |
| Analytic/rule vs measurement | ● | ● | ● | ● | ● | ● | ★★★★★ |
| Static specialization vs runtime adapt | ● | ● | ● | ● | ● | ● | ★★★★★ |
| Kernel optimum vs system optimum | ● | ● | ○ | ○ | ○ | ○ | ★★★★★ |
| Data vs metadata co-layout | ● | ● | ● | ● | ● | ●/部分 | ★★★★☆ |

这一张表比之前的“科研问题排行榜”更接近你要求的定义。

---

# 二十四、如果要从中再找“最核心”的三个问题

我目前不会先选 cost model，也不会先选“跨框架 candidate coverage”。

我会优先研究三个互相连接的问题。

第一个是：

\[
\boxed{
\textbf{Representation Stability vs Specialization}
}
\]

即一个 tensor 在它的 lifetime 内到底应该多稳定。

第二个是：

\[
\boxed{
\textbf{Regular Hardware Mapping vs Irregular LLM Workload}
}
\]

即为了使用硬件高效规则 mapping，究竟应该容忍多少 padding/masking/imbalance。

第三个是：

\[
\boxed{
\textbf{Optimization Modularity vs Coupled Search}
}
\]

即 layout、mapping、instruction、pipeline、consumer 是否可以分开优化，以及什么时候不能。

这三个问题背后其实是一条更统一的逻辑：

\[
\boxed{
\text{为了让 layout optimization tractable，
现有框架不断“减少自由度”；
但现代 LLM 恰恰不断引入新的 heterogeneity。}
}
\]

传统高性能 GEMM 是：

\[
\text{regular shape}
+
\text{single dominant consumer}
+
\text{short-lived tiles}
+
\text{stable dataflow}.
\]

而现代大模型越来越是：

\[
\text{ragged shapes}
+
\text{multiple consumers}
+
\text{persistent states}
+
\text{quant metadata}
+
\text{phase-dependent workloads}
+
\text{async heterogeneous pipelines}.
\]

我认为这可能是我们现在最值得继续追的总 research thesis：

> **Existing layout systems gain tractability by imposing structural regularity, representation stability, and optimization modularity. Modern LLM workloads increasingly violate all three assumptions. The scientific question is not whether these assumptions are wrong, but where each assumption remains beneficial and where its cost exceeds the complexity it saves.**

这比“现有 layout 框架做得不好”强得多，也比泛泛的“joint optimization”更有科研可检验性。

下一轮最值得做的不是代码，而是针对上面前三到五个 trade-off，逐个建立一张非常严格的 **Assumption → 源码证据 → 为什么框架这样设计 → 现代 LLM 子图如何施加反向压力 → 两种 competing strategy → 未知量 → 可证伪 hypothesis** 矩阵。这样我们可以进一步筛掉那些“听起来像科研问题、实际上答案很明显”的候选。

## 用户

继续下一轮

## 模型思考摘要

继续核查框架差异
继续核查框架差异

## 模型回复

这一轮我会把最有潜力的 trade-off 做成“假设审查表”，不再继续扩散问题数量。重点会放在 4 个最核心方向：**表示稳定性 vs 专用化、规则性 vs 不规则 workload、模块化优化 vs 联合优化、共享 vs 转换 vs 复制**；每个方向都用多框架源码证明“双方都有合理策略，因此不是一个答案显然的问题”。

## 模型工具调用参数

{"paths":["GitHub"],"query":"fetch_file"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查找 Triton 转换源码与注释

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索并优化线程局部性描述

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 线程局部性优化 passes

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析现代深度学习编译器子图优化方法

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 vLLM 多后端布局兼容性

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 TVM TIRx 存储优先布局文档

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 TIRX 布局文档

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

澄清TIRx布局与执行分工

## 模型思考摘要

查找并分析Triton及编译器布局优化资料

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 huggingface.co

## 模型思考摘要

厘清适用边界
厘清适用边界

## 模型思考摘要

补充验证思路
补充了验证思路

## 模型思考摘要

已搜索 1 个网站

## 模型回复

源码细看后，一个很关键的修正出现了：真正跨框架反复出现的 trade-off 不只是“layout 要不要联合优化”，而是**layout 到底要承担多少语义**。CuTe 把 layout 深度参与 work partition；TIRx 刻意把 layout 作为 storage contract、把 thread partition 交给 primitive dispatch；Hexcute则在 constraint inference 里同时推 task mapping 和 memory layout；Triton classic 把这些决策分散在 encoding/pass 中。这个分歧本身就是一个没有确定答案的研究轴。

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 triton-lang.org

## 模型思考摘要

归纳核心取舍
归纳核心取舍

## 模型思考摘要

已搜索 1 个网站

## 模型思考用时

思考了 2m 26s

## 模型回复

这次需要把前面的框架重新校准。你说的“科研问题”应该满足一个更严格的条件：

\[
\boxed{
\text{不是发现一个缺陷，而是发现一个没有支配解的设计 trade-off。}
}
\]

也就是说，现有框架为了让 layout 优化可编译、可搜索、可维护，分别选择了不同的折中点；两边都有明确收益和代价。现代 LLM 的 Attention/GQA/MLA/MoE/SwiGLU/KV cache 又把这些折中推到了新的 workload regime，因此“到底在哪个 regime 应该选择哪一边”没有确定答案。这才值得研究。

下面我先把源码层面对比进一步细化，再重新筛科研问题。

---

# 1. 这轮源码细看以后，框架之间最本质的区别不是“自动 vs 手写”

当前源码大致可以归纳成下面的决策链：

\[
\boxed{
\text{Representation}
\rightarrow
\text{Candidate generation}
\rightarrow
\text{Legality}
\rightarrow
\text{Mapping/instruction}
\rightarrow
\text{Conversion}
\rightarrow
\text{Pipeline}
\rightarrow
\text{Performance selection}
}
\]

不同框架是在这条链的不同位置“冻结自由度”。

| 系统 | Layout 表达了什么 | Candidate 如何产生 | 如何处理多种 layout 需求 | Pipeline / instruction | 性能决策 |
|---|---|---|---|---|---|
| CUTLASS/CuTe | storage + hierarchical thread/value/work mapping | expert composition + Builder | 用户显式组合不同 TiledCopy/MMA/SMEM layouts | 与 layout 高度耦合显式构造 | expert / static rules / profiler |
| Triton classic | compiler-owned distributed encoding | compiler passes + exposed autotune params | 插入 `convert_layout`，再 rematerialize/remove | 后续独立 MMA/pipeline/warp-specialization passes | heuristics + optional measurement |
| Gluon | programmer-visible register/lane/warp layout | programmer | programmer 决定 convert | programmer 显式 async/pipeline/WS | expert + autotune |
| TVM MetaSchedule | schedule 产生的 thread/memory structure | ScheduleRules | cache/rewrite-layout/schedule transformation | tensor intrin + software-pipeline schedule choices | XGB/search + real measurement |
| TIRx | **storage contract**，含 shard/replica/offset/hardware axes | explicit tile layout + primitive variants | `permute_layout` 等 primitive | pipeline/role 显式；primitive locally dispatched | predicates + priority |
| Hexcute | logical layout + TV task mapping + memory layout | constraint inference | 多 consumer constraints 统一/回溯 | instruction 参与 constraint；pipeline显式 | analytic cost model |
| vLLM | persistent `[L,B,H,N,C]` physical storage | finite enum + backend capabilities | 尽量一个 whole-model compatible layout | 委托 attention backend | support intersection + preference |
| SGLang | persistent/backend-specific representation | model/hardware/backend policy | backend-specific pool/layout | kernel backend 拥有低层 mapping | hand-written policy |

这些不是概括出来的“概念差异”，源码本身就是这样组织的。

CUTLASS 的 `CollectiveMma` 直接把 `TiledMma`、global tiled copy、SMEM layout atom、SMEM copy atom等放在同一个 collective type 里；dispatch policy 再同时包含 stage、cluster shape 和 warp-specialized kernel schedule。也就是说它实际上承认 layout、copy、MMA、pipeline、warp roles 是耦合配置。fileciteturn110file0L1-L2

Triton恰好相反。当前 NVIDIA backend 的 TTGIR pipeline 是 `Coalesce → RemoveLayoutConversions → OptimizeCTALocality → RemoveLayoutConversions → OptimizeThreadLocality → AccelerateMatmul → RemoveLayoutConversions → OptimizeDotOperands → ... → Pipeline/WarpSpecialize`。也就是说它刻意用一串局部 transformation 解这个问题。fileciteturn122file0L1-L2

而 `RemoveLayoutConversions` 的官方描述甚至明确写着：对于 expensive load/store 偏好适合 coalescing 的 `BlockedEncoding`，否则偏好适合 tensor ops 的 `NvidiaMmaEncoding`。这直接证明：

\[
L_{\mathrm{memory}}
\neq L_{\mathrm{tensorcore}}
\]

不是我们的假设，而是 Triton 自己必须解决的冲突。fileciteturn129file1L24-L44

TVM 又不同。它先定义 `MultiLevelTiling` 的结构、thread binding、vector load、read/write reuse；TensorCore 版本再指定 intrin groups 和是否 software pipeline，然后 MetaSchedule 才在这个人造 design space 上 evolutionary search + cost model + hardware measurement。fileciteturn126file0L1-L10 fileciteturn118file5L81-L95

TIRx 则明确选择另一条哲学：layout 是 storage contract；程序员描述数据如何分布在 lane/warp/register/shared/TMEM 等资源上，但是 primitive 的 loop、thread partition、addressing 和具体 instruction 由 dispatch 根据 operand layout + scope + target 生成。pipeline、barrier、role assignment仍然留在源代码。citeturn684706search0

Hexcute 则把 task mapping 和 memory layout 放在同一个 inference 中。`ResolveAuto` 对 Copy/MMA 建立 TV-layout constraints，一个 shared tensor 如果被多个 copy 消费，它的 memory layout 必须同时满足这些 consumer；如果矛盾就回溯。甚至源码自己说，同一 Copy 可能有多个 instruction，可以 DFS 全搜，也可以 heuristic/beam prune。fileciteturn109file0L1-L2

而 serving runtime 这一层又完全不同。vLLM 当前不是“让 kernel 自己决定 KV layout”，而是 backend 先声明 `supported_kv_cache_layouts()`，系统做支持集合交集和 preference，再 resolve **one KV cache layout for the whole model**；mixed HNC specs 还会强制候选满足 block-compact。fileciteturn96file0L1-L2 SGLang 则可以让 prefill 和 decode 使用不同 attention backend，同时某些 ROCm/AITER path 又要求特定 SHUFFLE/preshuffle representation，甚至 page-size 与 layout/kernel 必须匹配。fileciteturn124file0L1-L10 fileciteturn132file0L1-L27

这就给了我们比较扎实的出发点。

---

# 2. 什么才算我们现在应该寻找的“科研 trade-off”？

我现在会用四个条件筛选。

| 条件 | 不满足时是什么 |
|---|---|
| 两种策略各有明确优势 | 否则只是工程优化 |
| 多个框架实际选择了不同折中点 | 否则证据太弱 |
| 最优选择依赖 workload/hardware/context | 否则有确定答案 |
| 现代 LLM 会跨越不同 regime | 否则研究价值有限 |

例如：

> “Hexcute 应该加入 bank-conflict model”

不满足。加入更多信息理论上只会让模型更完整，只是 engineering cost。

但是：

> “应该用一个统一 layout，还是为不同 consumer 专门化 layout？”

满足。统一 layout 少 conversion、少存储、简单；专门 layout 每个 consumer 更快。没有确定答案。

---

# 3. 我现在最看好的问题之一：Layout 到底应该包含多少 execution semantics？

这一点前面其实还没有抓准。

不同框架已经给出了三种答案。

CuTe：

\[
\boxed{Layout \approx Storage + WorkPartition}
\]

TiledMma 的 atom layout、value layout、permutation 本身就决定 iteration/work mapping。fileciteturn110file0L1-L2

Hexcute：

\[
\boxed{
MemoryLayout
+
TaskMapping
+
InstructionCompatibility
}
\]

联合 constraint inference。fileciteturn109file0L1-L2

TIRx：

\[
\boxed{
Layout=StorageContract
}
\]

而：

\[
TaskMapping/Loops/Instruction
=
PrimitiveDispatch(Layout,Scope,Target).
\]

citeturn684706search0turn684706search2

Triton classic 更进一步，把 layout/work mapping 隐藏在 distributed encoding 和 compiler transformations 中；Gluon又重新把它显式暴露给 programmer。citeturn432936search1

所以真正科学问题不是：

> 谁的 layout abstraction 更强？

而是：

\[
\boxed{
为了获得 near-optimal GPU implementation，
layout representation 最少需要携带多少 execution semantics？
}
\]

这是一个真正的 trade-off。

如果 layout 包含大量 thread/value/instruction semantics：

\[
+\text{可以精确联合优化}
\]

\[
+\text{legality 更容易局部表达}
\]

但：

\[
-\text{representation 复杂}
\]

\[
-\text{搜索空间大}
\]

\[
-\text{与 instruction/architecture 耦合严重}
\]

\[
-\text{重用和可移植性下降}.
\]

反过来 storage-only：

\[
+\text{modularity}
+\text{primitive reuse}
+\text{portability}
\]

但可能留下大量：

\[
\text{residual execution freedom}.
\]

我们可以定义：

\[
\mathcal M(L)
=
\{(mapping,instruction,pipeline):
legal(L,\cdot)\}.
\]

真正未知的是：

\[
\Delta(L)
=
\frac{
\max_{m\in\mathcal M(L)}T(L,m)
}{
\min_{m\in\mathcal M(L)}T(L,m)
}.
\]

如果一个 storage layout 固定以后：

\[
\Delta(L)\approx1,
\]

说明 layout 和 task mapping 可以很好分离。

如果：

\[
\Delta(L)\gg1,
\]

说明 storage layout 并没有包含足够的性能信息。

### 为什么现代 LLM 特别适合挑战这个假设？

普通 GEMM 很规则：

\[
M\times K \times N
\]

而 Attention 同一个 tile 会经历：

\[
QK^T
\rightarrow reduction/Softmax
\rightarrow PV.
\]

GQA 又有：

\[
H_q/H_{kv}>1.
\]

例如 Qwen3-8B 是 32 个 query heads、8 个 KV heads，即每个 KV head 对应 4 个 Q heads。citeturn541538search0

MLA 更不规则。DeepSeek-V3 同时有 `qk_nope_head_dim=128`、`qk_rope_head_dim=64`、`v_head_dim=128`、`kv_lora_rank=512`。citeturn541538search2

所以同一个 storage placement 背后可能有非常不同的 work mappings。

我现在把这个问题评为：

\[
\boxed{\textbf{S}}
\]

而且比“storage layout vs task mapping 是否可分”原来的表述更准确，因为现在研究的是**semantic richness 的最佳边界**，而不是简单二选一。

---

# 4. 第二个非常强的问题：Representation Stability vs Specialization Granularity

所有框架都在某个层面遇到：

\[
\boxed{
\text{一个 tensor 应该保持多稳定？}
}
\]

vLLM 是最极端的例子：

\[
L_{KV}
\]

最后 resolve 为 whole-model layout。fileciteturn96file0L1-L2

为什么这样合理？

因为 persistent KV cache 可能几十 GB：

\[
+\text{单一 allocator}
+\text{简单 page aliasing}
+\text{无需转换}
+\text{connector/backend compatibility 简单}.
\]

但 backend preference 本身并不一致。源码明确允许不同 backend 声明不同支持顺序，甚至测试代码里为了测试一个不兼容 shared layout 的 backend，会给它复制一份它能接受的 cache。fileciteturn133file3L44-L60

SGLang 又已经往 specialization 另一侧移动：prefill/decode 可以不同 backend，某些 AITER path 还拥有特定 SHUFFLE representation。fileciteturn124file0L1-L10

Triton在 transient tensor 层面更激进：

\[
L_{\rm memory}
\rightarrow ConvertLayout
\rightarrow L_{\rm MMA}.
\]

然后 compiler 又努力删除 conversion。fileciteturn129file0L1-L22

Hexcute则更倾向于让同一个 shared tensor 的多个 consumer 共同约束一个可接受 memory layout。fileciteturn109file0L1-L2

所以这些系统实际上分布在一条轴上：

\[
\boxed{
Uniform
\longleftrightarrow
ConsumerSpecialized
}
\]

真正研究问题是：

\[
\boxed{
Layout specialization 的合理粒度是什么？
}
\]

候选粒度甚至可以排成：

\[
model
\rightarrow
layer
\rightarrow
phase
\rightarrow
consumer
\rightarrow
primitive
\rightarrow
tile.
\]

越向右：

\[
local\ performance\uparrow
\]

但同时：

\[
conversion+
storage+
metadata+
management+
search
\uparrow.
\]

这个问题不存在一个固定答案。

---

# 5. 而且“稳定 vs 专用”实际进一步分成三个策略

当两个 consumer 喜欢：

\[
L_A,\quad L_B
\]

时，至少有三种合理方案：

\[
\text{Compromise: }L_C
\]

让两者都用一个次优 layout。

或者：

\[
\text{Convert: }L_A\rightarrow L_B
\]

或者：

\[
\text{Replicate: }\{L_A,L_B\}.
\]

Triton 的 `ReduceDataDuplication` 很有意思，它甚至会把：

\[
distributed\rightarrow dotOperand
\]

改成：

\[
distributed\rightarrow shared\rightarrow dotOperand
\]

以便利用 CSE 复用 shared tensor，而不是在 register 中重复数据。fileciteturn136file0L1-L30

TIRx 的 layout 本身把 `Replica` 作为一等结构：

\[
L(x)=\{D(x)+r+O\mid r\in R\}.
\]

fileciteturn135file0L1-L7

所以：

\[
\boxed{
Compromise
\quad vs\quad
Conversion
\quad vs\quad
Replication
}
\]

其实应该被看成同一个 scientific design space，而不是三个 compiler trick。

Trade-off 非常明确：

\[
T_{\rm execution}
+
T_{\rm conversion}
+
\lambda_M M_{\rm footprint}.
\]

其中 \(\lambda_M\) 在 standalone kernel 与 LLM serving 中完全不同。

在 kernel benchmark 中，多用 20 MB workspace 可能无所谓。

在 KV serving 中，多复制 20% KV：

\[
\text{最大并发 batch}
\]

可能直接下降。

因此这个问题同时连接 kernel compiler 和 serving runtime。

我认为这一问题是：

\[
\boxed{\textbf{S}}
\]

尤其适合你的 KV-layout 主线。

---

# 6. 第三个核心问题：Hardware Regularity vs Workload Fidelity

我上一轮把它说成“power-of-two layout”，太窄。

真正所有框架都在做的是：

\[
\boxed{
把 workload regularize 成硬件容易处理的形状。
}
\]

CUTLASS 基本世界观就是：

\[
Atom
\rightarrow TiledMma/TiledCopy
\rightarrow Collective
\rightarrow Cluster
\]

规则 hierarchy。fileciteturn110file0L1-L2

TVM `MultiLevelTiling` 更直接把：

\[
structure,\ tile\ binds,\ vector\ lengths,\ reuse
\]

作为 schedule-space skeleton。fileciteturn126file0L1-L10

Triton/Gluon甚至明确要求 tile dimensions 为 power-of-two；元素均匀分给 threads。如果 tensor 比 layout block 小，文档例子中 `32×8` 的 256 个逻辑元素可以占用相当于 `64×16=1024` 个 physical register slots，因为数据会 broadcast 到 warp/layout hierarchy。citeturn432936search0

Serving 层是同样的问题，只是 regularity 单位变成：

\[
page/block.
\]

例如 SGLang 某个 AITER preshuffle path 要求与特定 page-size/layout/kernel组合匹配。fileciteturn132file0L1-L27

问题是现代 LLM 正越来越违反规则性：

Qwen3 GQA：

\[
H_q=32,\quad H_{kv}=8.
\]

citeturn541538search0

DeepSeek-V3：

\[
256\ experts,\quad top8/expert,
\]

意味着每个 expert 实际收到的 token 数动态且极不均匀；同时 MLA 又有不对称 Q/K/V representation。citeturn541538search2

所以真正科研问题是：

\[
\boxed{
什么时候应该把 irregular workload 填充/分桶/规则化，
什么时候应该原生保留 irregularity？
}
\]

规则化：

\[
+\text{TensorCore utilization}
+\text{vectorization}
+\text{coalescing}
+\text{simple scheduling}
\]

但：

\[
+\text{padding}
+\text{masking}
+\text{register waste}
+\text{fragmentation}
+\text{load imbalance}.
\]

Native irregular：

减少浪费，却可能破坏：

\[
MMA/TMA/vectorization
\]

最容易得到研究结果的不是证明：

> irregular layout 更好。

而是画出 phase boundary：

\[
r^*
=
f(
shape\ irregularity,
arithmetic\ intensity,
batch,
hardware,
instruction\ granularity
).
\]

这个问题也是：

\[
\boxed{\textbf{S}}
\]

但我要提醒：**regularity vs irregularity 本身是老问题**。论文新意必须落在现代 LLM 特有的 layout mechanism 上，比如 GQA/MLA/MoE/paged KV，而不是泛泛做 ragged tensor。

---

# 7. 第四个核心问题：Locality / Reuse vs Parallelism / Capacity

这是几乎所有 layout system 都必须处理、但前面我们没有足够重视的 trade-off。

Layout 让更多相邻数据属于同一个 thread：

\[
values/thread\uparrow
\]

通常：

\[
reuse\uparrow,\qquad
shuffle\downarrow,\qquad
coalescing\uparrow.
\]

但是：

\[
register/thread\uparrow
\Rightarrow
occupancy\downarrow.
\]

让更多数据驻留 SMEM：

\[
DRAM\ traffic\downarrow
\]

但：

\[
SMEM/CTA\uparrow
\Rightarrow
resident\ CTA\downarrow.
\]

增加 replication：

\[
communication\downarrow
\]

但：

\[
footprint\uparrow.
\]

这个 trade-off 在源码中到处存在。

CUTLASS 的 stage count 和 tile/cluster/schedule 本来就是共同参数；当前 low-precision builder甚至因为大 tile 的 SMEM footprint 容不下两个完整 stages而使用特殊的 1.5-stage 设计。fileciteturn125file1L21-L31

TVM 把 vector load、reuse read/write 和 tiling structure 放在同一个 schedule rule。fileciteturn126file0L1-L10

Gluon 官方 layout教程明确提醒 register ownership会影响 register budget；前面 `32×8` 对 `64×16` layout 的例子就是规则 layout 导致 4× physical register representation。citeturn432936search0

Triton则专门有 `ReduceDataDuplication` 来减少 register duplication。fileciteturn136file0L1-L30

所以真正科学问题是：

\[
\boxed{
Layout 应该追求多少 locality/reuse，
才不会为了局部数据重用牺牲过多 execution parallelism？
}
\]

这个问题没有固定答案，因为 Decode Attention 很可能：

\[
memory\ bound
\]

所以 locality/reuse 非常值钱。

Prefill GEMM：

\[
compute\ bound
\]

可能更需要 occupancy/parallelism。

MoE 小 expert GEMM 又需要足够 CTA 数填满 GPU。

这是一个真正的 trade-off，但我现在会给：

\[
\boxed{\textbf{A+}}
\]

而不是 S，因为“locality vs occupancy”本身是经典 GPU 问题。要成为主论文，需要证明 **layout hierarchy / replication / modern LLM phase heterogeneity** 产生新的规律。

---

# 8. 第五个核心问题：Optimization Modularity vs Coupled Search

这是我认为最可能成为 compiler 主线的一个问题。

各框架实际上已经选择了完全不同的 coupling boundary。

CUTLASS：

\[
\boxed{
Tile+Layout+Copy+MMA+Pipeline+WarpSchedule
}
\]

高度共同构造。fileciteturn110file0L1-L2

Hexcute：

\[
\boxed{
Layout+TaskMapping+Instruction
}
\]

联合推导；

但：

\[
\boxed{
Dataflow+Pipeline
}
\]

显式固定。fileciteturn109file0L1-L2 citeturn684706search5

TIRx：

\[
Layout+PrimitiveDispatch
\]

局部组合；

pipeline、roles、barrier：

\[
Programmer.
\]

citeturn684706search0

Triton：

\[
Layout
\rightarrow MMA
\rightarrow Pipeline
\rightarrow WarpSpecialization
\]

通过多个 pass 连续修正。fileciteturn122file0L1-L2

TVM：

ScheduleRule 事先决定哪些变量属于 design space，然后 search。fileciteturn126file0L1-L10

为什么大家不直接 joint-search？

因为如果：

\[
L\times Tile\times I\times P\times W
\]

全笛卡尔积：

\[
|\mathcal S|
=
|L||Tile||I||P||W|
\]

很容易爆炸。

但为什么不全部 modular？

因为 interaction 可能导致 rank reversal。

例如：

\[
T(L_1,P_1)<T(L_2,P_1)
\]

但：

\[
T(L_1,P_2)>T(L_2,P_2).
\]

真正科学问题不应该再叫：

> “joint optimization 有没有用？”

这个问题太容易。

应该是：

\[
\boxed{
GPU layout optimization 的 interaction graph 有多稀疏？
}
\]

令：

\[
D=
\{L_G,L_S,L_R,T,I_C,I_M,P,W\}.
\]

定义 decision interaction graph：

\[
G=(D,E).
\]

如果某两类决策几乎没有 rank reversal：

\[
(D_i,D_j)\notin E
\]

就应该模块化。

只有 strong-coupling edge 才 joint optimize。

因此真正未知的是：

\[
\boxed{
哪些 decision edges 在 Attention/GEMM/MoE/MLA 上必须联合，
哪些可以安全分解？
}
\]

这既解释了 CUTLASS 为什么高度 coupled，也解释了 Hexcute/TIRx 为什么刻意留下 pipeline 给 expert、Triton 为什么做 staged repair。

这比抽象的 “Boundary Regret” 强很多。

我会给：

\[
\boxed{\textbf{S}}
\]

---

# 9. 第六个问题：Local Primitive Optimality vs Subgraph Optimality

这个问题与上一个相关，但不是同一个。

上一个是：

> 同一个 kernel 中哪些决策需要 joint。

这里是：

> 每个 primitive 各自最快，整个 subgraph 是否最快？

TIRx 是最漂亮的例子。

每个 TilePrimitive 的多个 implementation：

\[
variant,\ priority,\ predicates
\]

独立注册；

dispatch 会按 priority 检查，并采用第一个合法实现。fileciteturn107file0L1-L38

这非常 modular。

但：

\[
Copy^*
\rightarrow
GEMM^*
\rightarrow
Reduction^*
\]

不必然等于：

\[
\arg\min
T_{\rm Copy\rightarrow GEMM\rightarrow Reduction}.
\]

Triton的一串 layout passes，本质也是在处理这种 local preference conflict。

Attention 是最典型场景：

\[
QK^T
\rightarrow
Softmax
\rightarrow
PV.
\]

QK：

\[
L_{\rm MMA1}
\]

Softmax：

\[
L_{\rm reduction}
\]

PV：

\[
L_{\rm MMA2}.
\]

Gluon官方文档给了一个特别值得注意的 observation：

对于 reduction，虽然确实存在更 reduction-friendly 的 layout，但很多情况下：

\[
ConvertLayout+Reduce
\]

反而比：

\[
Reduce(original\ layout)
\]

更慢，因为 compiler 已经能较好地处理原 layout 上的 reduction。citeturn684706search1

所以绝不是：

> 每个算子都先转成自己的最佳 layout。

真正研究问题：

\[
\boxed{
Local layout utility 在什么条件下具有 compositionality？
}
\]

即是否可以近似：

\[
T_{\rm graph}
\approx
\sum_i T_i(L_i)
+
\sum_i C(L_i,L_{i+1}).
\]

如果成立，局部优化很合理。

如果：

\[
cache,\ occupancy,\ async\ overlap,\ fusion
\]

产生强 interaction，就不成立。

我会给：

\[
\boxed{\textbf{A+/S-}}
\]

---

# 10. 第七个问题：Structural Prior vs Empirical Evidence

这个是前面的 Coverage / Selection 之上真正应该问的问题。

所有框架都不能搜索完整 layout space，因此都必须使用 prior。

CUTLASS：

\[
expert templates + architecture rules.
\]

Triton：

\[
compiler heuristics + limited exposed autotune configs.
\]

TVM：

\[
ScheduleRules
\rightarrow
EvolutionarySearch
\rightarrow
measurement.
\]

fileciteturn118file0L1-L14

Hexcute：

\[
constraint legality
\rightarrow
DFS/heuristic/beam
\rightarrow
analytic cost.
\]

fileciteturn109file0L1-L2

TIRx：

\[
registered variants
\rightarrow
predicate+priority.
\]

citeturn684706search2

vLLM：

\[
supported\ layout\ intersection
\rightarrow preference.
\]

SGLang：

\[
model/hardware/backend rules.
\]

所以所有系统都在回答：

\[
\boxed{
在实际测量之前，可以安全地相信多少结构知识？
}
\]

纯 structural prior：

\[
+\text{zero/low tuning cost}
+\text{可解释}
+\text{适合 dynamic workloads}
\]

但可能：

\[
-\text{漏掉反直觉高性能 configuration}.
\]

Measurement-heavy：

\[
+\text{真实硬件反馈}
\]

但：

\[
-\text{编译/tuning 开销}
-\text{shape explosion}
-\text{online workload 无法穷举}.
\]

现代 serving 的 shape state：

\[
x=
(B,S_q,S_{kv},H_q/H_{kv},phase,page,spec,\ldots)
\]

数量巨大。

所以真正问题：

\[
\boxed{
Layout optimization 至少需要多少 empirical feedback，
才能超过 purely structural selection？
}
\]

或者更强地问：

\[
\boxed{
Structural legality 与 high-performance region
之间到底有多强相关性？
}
\]

这个问题非常普遍，但是 AutoTVM/MetaSchedule/Ansor 等领域已经长期研究 tuning-vs-model，因此 novelty 风险比前几个高。

我目前给：

\[
\boxed{\textbf{A}}
\]

除非我们最后发现 layout-specific constraints 有特别强的新规律。

---

# 11. 第八个问题：Static Specialization vs Runtime Adaptation

这一个跨层，但现代 LLM 特别强。

低层 compiler 几乎都在假设：

\[
shape/workload
\]

在编译或 tuning 时足够确定。

但 serving 的真实状态高度变化。

SGLang 已经认为：

\[
Backend_{\rm prefill}
\neq Backend_{\rm decode}
\]

可能合理。fileciteturn124file0L1-L10

vLLM 的 KV layout却在 engine-core resolve 一次，然后固定。fileciteturn96file0L1-L2

所以这里存在一个非常干净的矛盾：

\[
\text{backend can be phase-specialized}
\]

但：

\[
\text{persistent representation is mostly stable}.
\]

为什么 stable 合理？

因为：

\[
C_{\rm migration}
\]

很大。

为什么 adaptive 合理？

因为 prefill / decode / KV transfer / speculative decoding 的 consumer patterns 完全不同。

因此 objective 是：

\[
\min_{\pi}
\sum_t
T(x_t,L_t)
+
C_{\rm switch}(L_{t-1},L_t).
\]

真正问题：

\[
\boxed{
Layout specialization 的收益需要多长 workload duration 才能 amortize representation switching cost？
}
\]

这显然没有固定答案。

对于 KV layout 项目，我会给：

\[
\boxed{\textbf{S}}
\]

但它主要是 serving/runtime 论文，而不是一般 GPU compiler 论文。

---

# 12. 现代 LLM 为什么正在系统性破坏旧的 layout 假设？

现在可以把两个真实模型放进来。

Qwen3-8B：

\[
H_q=32,\quad H_{kv}=8,\quad D=128,
\]

并支持约 40K context。它带来的核心压力是 GQA：同一 KV head 被多个 query heads 共用，因此共享、复制、head grouping、thread mapping之间产生直接 trade-off。citeturn541538search0

DeepSeek-V3 更极端：

\[
qk_{nope}=128,\quad
qk_{rope}=64,\quad
V=128,\quad
KV_{latent}=512
\]

以及：

\[
256\ routed\ experts,\quad top8,
\]

同时使用 block-wise FP8 quantization。citeturn541538search2turn541538search4

它同时产生：

\[
\text{heterogeneous attention representation}
\]

\[
+\text{dynamic MoE token counts}
\]

\[
+\text{quantization metadata}
\]

\[
+\text{long context}.
\]

这就是为什么旧的：

\[
\text{regular tile}
+
\text{one dominant consumer}
+
\text{fixed dataflow}
+
\text{static shape}
\]

假设越来越容易受到压力。

---

# 13. 重新排序后，我认为目前真正值得继续查的科研问题是这些

| Trade-off | 跨框架证据 | 是否真没有固定答案 | LLM relevance | Novelty 风险 | 当前评价 |
|---|---:|---:|---:|---:|---:|
| Layout semantic richness：storage-only vs execution-coupled | ★★★★★ | ★★★★★ | ★★★★☆ | 低-中 | **S** |
| Representation stability vs specialization granularity | ★★★★★ | ★★★★★ | ★★★★★ | 低 | **S** |
| Compromise vs Convert vs Replicate | ★★★★★ | ★★★★★ | ★★★★★ | 低-中 | **S** |
| Regular hardware layout vs irregular workload fidelity | ★★★★★ | ★★★★★ | ★★★★★ | 中 | **S-/A+** |
| Layout locality/reuse vs parallelism/resource capacity | ★★★★★ | ★★★★★ | ★★★★★ | 高（经典） | **A+** |
| Modular optimization vs selective joint optimization | ★★★★★ | ★★★★★ | ★★★★★ | 低-中 | **S** |
| Primitive-local optimum vs subgraph optimum | ★★★★☆ | ★★★★★ | ★★★★★ | 中 | **A+/S-** |
| Structural prior vs empirical measurement | ★★★★★ | ★★★★★ | ★★★★☆ | 高 | **A** |
| Static specialization vs runtime adaptation | ★★★★☆ | ★★★★★ | ★★★★★ | 中 | **S，serving方向** |
| Data/scale/metadata co-layout | ★★★☆☆ | ★★★★★ | ★★★★★ | 低 | **A+，新兴方向** |

---

# 14. 这也让我再次降低了几个前面问题的重要性

`Boundary Regret`、`Coverage Regret`、`Selection Regret` 我现在仍然保留，但全部降级为：

\[
\boxed{\text{measurement instruments}}
\]

而不是研究问题。

例如对：

\[
\text{Compromise vs Convert vs Replicate}
\]

我们可以用 Boundary/Selection regret 测它，但论文问题不能叫“Boundary Regret”。

同样：

> “支持 irregular layout”

不是科研问题。

正确问题是：

\[
\text{Regularization benefit}
\quad vs\quad
\text{irregularity overhead}.
\]

> “联合优化 layout 和 pipeline”

也不是科研问题。

正确问题是：

\[
\text{interaction strength}
\quad vs\quad
\text{search complexity}.
\]

> “动态 KV layout”

也不是科研问题。

正确问题是：

\[
\text{specialization gain}
\quad vs\quad
\text{state migration/amortization}.
\]

这应该成为后续筛选科研问题的统一标准。

---

# 15. 目前我认为最可能形成一条统一论文主线的是下面这个命题

这些系统为了让 layout optimization tractable，基本都使用了三类“降低自由度”的手段：

\[
\boxed{
\text{Representation stability}
}
\]

尽量少维护不同 physical representations；

\[
\boxed{
\text{Structural regularity}
}
\]

尽量使用规则 tile/page/thread hierarchy；

\[
\boxed{
\text{Optimization modularity}
}
\]

尽量把 layout、instruction、pipeline、runtime 分阶段解决。

这三种策略都非常合理，因为否则搜索空间和软件复杂度会爆炸。

但是现代 LLM 正在同时增加：

\[
\text{consumer heterogeneity}
+
\text{shape irregularity}
+
\text{persistent state}
+
\text{dynamic phases}
+
\text{quant metadata}
+
\text{async hardware coupling}.
\]

所以更强的总研究命题应该是：

\[
\boxed{
\textbf{
How much structural freedom should a layout optimizer preserve
before the cost of optimization exceeds the performance value of that freedom?
}}
\]

换成中文：

> **现代 GPU layout 优化真正的问题可能不是“如何找到更好的 layout”，而是“应该保留多少 layout 自由度”：什么时候应该统一、规则化、模块化以控制复杂度；什么时候现代 LLM 的异构性又强到必须允许 specialization、irregularity 和 cross-stage coupling。**

这个问题本质上是：

\[
\boxed{
Optimization\ tractability
\quad\leftrightarrow\quad
Representation\ freedom
}
\]

我认为它比我们前面单独说“joint optimization”“candidate coverage”“动态 KV layout”都更接近一个核心科研问题。

下一轮最值得继续做的是对这几个 **S 级 trade-off** 逐一检查已有论文：不是只找有没有相同标题，而是检查已有工作到底固定了 trade-off 的哪一侧、做了什么假设、有没有真正回答“什么时候应该切换策略”。然后才能判断哪一个问题具有足够的新颖性，哪几个其实已有文献已经回答得比较充分。

## 用户

现继续审查：抨击当前框架中layout优化的一些假设和方法策略上的不足，越多框架有这个问题越好，但是需要区分这种问题是否是一个值得研究的trade-off（这是我指的科研问题：所有框架都有这个问题，但是都没有确定的答案）， 而不是一个确定性答案。尤其是这些框架的解决策略 遇到现代大模型的常见子图时。所以请继续分析，前面的回答仍然需要调整。另外建议你再就好好的扒一扒不同框架的实现源码，总结他们的区别与共同点，尽可能粒度更细，而不是粗略的回答，每一个框架优化了什么。   这里对框架的整理有没有做好，符合源码。而且细粒度的对比，也确实归纳出了他们的不同点和共同点。

然后再继续哪些已有的实现点/layout优化方法的细节确认哪些确实因为假设不充分，假设错误，有什么模型框架支持它。哪些优化策略不够好，还不是最优的：有什么模型框架支持它。这些支持的作为问题的主要证据。然后据此提炼研究问题。

先不着急检查论文

## 模型回复

我会按你这次要求把工作拆成两层：先做一次“源码审计版框架对比”，把每个框架的 layout 相关决策链、默认假设、修正机制、可控变量、性能选择方式逐项对齐；然后只从**源码能支持的不足/未决取舍**中提炼问题，并明确标注“确定性缺陷”还是“值得研究的 trade-off”。这一轮先不查论文。

## 模型工具调用参数

{"paths":["GitHub"],"query":"search"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Hexcute 成本模型与银行冲突解析

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Hidet 相关仓库

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

解释 Triton 布局优化与代价模型

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 OptimizeThreadLocality 实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

评估FlashInfer量化KV缓存支持

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 TileLang 布局推断与流水线规划

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 TileLang 流水线与布局推断顺序

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 vLLM KV 连接器布局及提交信息

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 vLLM 中的 KV 连接器缓存布局

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 KV 传输连接器工具代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 CUTLASS 自动调度选择逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 SM100 稀疏 UMMA 构建器代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 GPU 注意力布局与后端实现

## 模型思考摘要

提炼跨框架证据
提炼跨框架证据

## 模型思考摘要

检索并梳理GPU布局、成本模型与KV缓存实现

## 模型回复

细化源码后，前面的框架整理确实还要再修正两点。第一，不能把“layout optimizer”当作统一对象：例如 TileLang 明确先 materialize/规划 pipeline，再做 LayoutInference，而 CUTLASS 把 stage/schedule/layout 放在同一 collective 构造里；这说明“layout 与 pipeline 谁先决定”本身就是框架策略差异。第二，vLLM 虽然全模型 resolve 一个 KV layout，但其 disaggregated KV transfer 代码已经包含接收端 block-size/layout postprocess，甚至注明 prefill 可为 LBHNC、小 block，而 decode 可为 LBNHC、大 block——也就是说系统自己已经出现“统一 representation 不够”的补偿路径。 

## 模型思考：工具活动

正在搜索 Qwen3-8B config hidden_size num_attention_heads num_key_value_heads head_dim

## 模型思考：工具活动

正在搜索 huggingface.co

## 模型思考摘要

分析依赖关系
规划研究结构
分析依赖关系并规划研究结构

## 模型思考摘要

已搜索 1 个网站

## 模型思考用时

思考了 1m 59s

## 模型回复

需要再调整。前几轮已经把框架分层了，但还没有完全达到你现在要求的标准：**从源码中的具体实现策略出发，找出多个框架反复采用的假设，再检查这些假设是否只是工程近似、是否已经被框架自己的补偿逻辑“打脸”，最后只有那些两边都有收益、没有确定答案的取舍，才提升为科研问题。**

这一轮我先不查论文，只做这件事。

## 1. 先回答一个关键问题：前面的框架整理够细了吗？

还不够。至少有四个地方需要修正。

第一，不能再简单写“CUTLASS 联合优化、Triton 分阶段优化、TVM 搜索、Hexcute constraint synthesis”。必须继续往下看：**它们分别把哪些变量放进同一个决策里，哪些变量先被冻结，后面的 pass 是否还能反向修改前面的选择。**

第二，vLLM/SGLang 不能只视为“runtime backend selector”。它们已经在直接处理 persistent layout、page geometry、transfer layout、backend-specific representation。例如 vLLM 虽然 resolve “one KV cache layout for the whole model”，但其 KV-transfer 代码已经有显式的 layout/block-size 后处理，甚至源码注释直接写：

\[
\text{prefill: LBHNC + smaller block}
\]

\[
\text{decode: LBNHC + larger block}
\]

然后在接收端做 reshape/permute。也就是说，**框架自身已经遇到“统一 layout 与不同 consumer 最优 layout 不一致”的现实问题**。fileciteturn146file0L1-L2

第三，Triton 的逻辑也不能概括成“compiler chooses layout”。它当前 CUDA pipeline 是反复：

\[
Coalesce
\rightarrow RemoveLayoutConversions
\rightarrow OptimizeThreadLocality
\rightarrow AccelerateMatmul
\rightarrow RemoveLayoutConversions
\rightarrow OptimizeDotOperands
\rightarrow Pipeline/WarpSpecialization
\]

也就是说它不是一次做 layout 决策，而是**不同目标的 pass 不断改变 layout，再用 conversion-removal 修补局部冲突**。fileciteturn122file0L1-L2

第四，TileLang 也给了一个之前忽略的重要证据：当前 CUDA pipeline 会先做 `PipelinePlanning` / software-pipeline injection，再做 `LayoutInference`，并明确写“让 inferred layouts 看到最终 pipelined structure”；warp-specialization 也要先 materialize，后面的 LayoutInference 才看到新的结构。fileciteturn142file0L1-L26 fileciteturn142file2L43-L66

这个 pass 顺序本身就是一种假设：

\[
\boxed{
Pipeline\rightarrow Layout
}
\]

而 CUTLASS 更接近：

\[
\boxed{
Pipeline\leftrightarrow Layout
}
\]

这才是我们要抓的差异。

---

# 2. 重新做一张源码级 Decision Chain

下面这张我认为比前面的表更准确。

| 框架 | Layout 表达层 | Layout 候选从哪里来 | Mapping 如何得到 | 多 consumer 冲突怎么处理 | Pipeline 什么时候决定 | 性能决策 |
|---|---|---|---|---|---|---|
| CUTLASS/CuTe | GMEM stride、SMEM layout、thread/value、MMA/copy tiling | expert / Builder | 与 TiledMma/TiledCopy 一起显式构造 | expert 显式选择/转换 | 与 Collective/Schedule 一起构造 | static rules / expert / profiler |
| Triton classic | compiler-owned distributed encoding | passes + meta-config | encoding lowering | 插 convert，再 remove/rematerialize | layout/MMA 后，pipeline pass | heuristic + autotune |
| Gluon | programmer-visible distributed layout | programmer | programmer layout 直接决定 | programmer `convert_layout` | programmer 显式 | expert + autotune |
| TVM MetaSchedule | schedule-derived memory/thread organization | ScheduleRule | tiling/binding/tensorize | cache/rewrite-layout | schedule space 中可选 | XGB/evolution + measurement |
| TIRx | storage contract：D/R/O + hardware axes | programmer/primitive variants | primitive dispatch 根据 layout/scope/target 产生 | permute/primitive composition | orchestration显式 | predicate + priority |
| Hexcute | logical layout + TV mapping + memory layout | constraint inference | 与 instruction 一起约束推导 | 多 consumer constraint unify/backtrack | dataflow/pipeline 预先显式 | analytic cost model |
| vLLM | persistent `[L,B,H,N,C]` physical layout + block | finite enum + backend support | kernel delegated | 全局 intersection/preference；transfer 时可转换 | backend 内部 | compatibility + preference |
| SGLang | persistent/backend-specific cache layout | backend/model/platform rule | backend delegated | phase/backend-specific路径 | backend 内部 | hand-written runtime rule |
| FlashInfer | attention/KV-specific NHD/HND/page/metadata | specialized kernel families | backend-specific | kernel family选择 | kernel固定/模板 | specialized dispatch |

CUTLASS 的源码尤其清楚。`CollectiveMma` 同时持有 `TiledMma`、GMEM `TiledCopy`、SMEM layout atom、SMEM copy atom；dispatch policy 又同时带 stages、cluster shape、kernel schedule。fileciteturn110file0L1-L2

更重要的是 `KernelScheduleAuto` 并不是性能搜索。例如 SM100 sparse builder 中，“Auto”是在编译期检查 cluster M 是否是 2 的倍数、tile M 是否满足 128 的条件，然后决定 2SM MMA 或 1SM MMA；否则 fallback。StageCountAuto 同样是根据剩余 SMEM capacity 算最大可容纳 stage 数。fileciteturn148file0L1-L2

所以 CUTLASS 的“Auto”实际上是：

\[
\boxed{\text{structural legality + expert heuristic}}
\]

而不是：

\[
\arg\min T.
\]

---

# 3. Triton 的源码暴露了一个非常重要的事实：单一最佳 layout 根本不存在

`RemoveLayoutConversions` 的 pass description 明确写：

对于昂贵 load/store：

\[
BlockedEncoding
\]

更有利于 coalescing；

其他情况：

\[
NvidiaMmaEncoding
\]

更适合 tensor operations。fileciteturn129file1L24-L44

`OptimizeThreadLocality` 又专门为了 reduction、reshape、gather 降低 cross-thread communication。fileciteturn140file0L1-L22

因此一个真实 tensor 会面对：

\[
L_{\rm memory}
\]

\[
L_{\rm reduction}
\]

\[
L_{\rm MMA}.
\]

Triton 的回答不是“选一个”，而是：

\[
\boxed{
允许冲突存在
\rightarrow
插入 conversion
\rightarrow
后续尽量消除 conversion
}
\]

甚至 `ReduceDataDuplication` 会把：

\[
distributed\rightarrow dotOperand
\]

改成：

\[
distributed\rightarrow shared\rightarrow dotOperand
\]

以减少 register duplication，并让 CSE 复用 shared tensor。fileciteturn136file0L1-L30

这三个 pass 本身就是非常好的“现有假设并不充分”的证据。

---

# 4. Hexcute 的源码也比之前总结得更有信息量

Hexcute/Hidet 的 auto annotation 明确分为：

\[
LogicalShape
\rightarrow
LogicalLayout
\rightarrow
TaskMapping
\rightarrow
MemoryLayout.
\]

其中 Copy 和 MMA 都通过 TV-layout constraint 推导；一个 shared tensor 如果被多个 Copy 使用，其 memory layout 要统一所有 consumer constraint；冲突时回溯。fileciteturn109file0L1-L2

但源码还有一句很关键：

> 当 constraints 全部满足以后，剩余未确定 stride 再由 heuristic 决定。

也就是说：

\[
\boxed{
Legality\ constraints
\neq
Complete\ performance\ decision
}
\]

而一个 Copy 有多个合法 instruction 时，源码明确允许：

\[
DFS\ all\ variants
\]

或者：

\[
heuristic / beam\ search
\]

去 prune。fileciteturn109file0L1-L2

再往后，bank conflict 又由独立的 `ResolveBankConflict` pass 枚举 swizzle；性能 cost model 本身主要依赖 instruction count × latency，并假定 copy/MMA overlap，而 bank conflict、barrier 等没有完整进入这个 ranking。fileciteturn79file0L1-L7 fileciteturn81file0L1-L7

所以 Hexcute 实际结构是：

\[
\boxed{
Legal layout synthesis
\rightarrow
analytic ranking
\rightarrow
bank-conflict correction
}
\]

而不是“constraint solver直接找到性能最优 layout”。

这一区分后面非常重要。

---

# 5. TVM 的关键不是“会 search”，而是 search 前已经固定了大量结构

`MultiLevelTiling` 先决定：

\[
structure
\]

\[
tile\ binds
\]

\[
vector\ load\ lengths
\]

\[
read/write\ reuse.
\]

TensorCore 版本再增加：

\[
intrin\ groups
\]

以及：

\[
use\_software\_pipeline.
\]

fileciteturn126file0L1-L10

然后才：

\[
ScheduleRules
\rightarrow DesignSpace
\rightarrow EvolutionarySearch
\rightarrow XGB
\rightarrow Measurement.
\]

fileciteturn118file5L81-L95

所以 TVM 的真正假设不是：

> measurement 会找到最优。

而是：

\[
\boxed{
值得测量的结构已经被 ScheduleRules 覆盖。
}
\]

Hardware measurement 再准确，也无法选到：

\[
c\notin\mathcal C_{\rm ScheduleRule}.
\]

这不是 bug，而是 search tractability 必须做的取舍。

---

# 6. TIRx 与 Hexcute/CuTe 的差异现在可以说得更精确

TIRx Layout 当前显式表示：

\[
D=\text{Shard}
\]

\[
R=\text{Replica}
\]

\[
O=\text{Offset}
\]

并映射到 memory/thread/device axes。一个 logical element 甚至可以映射到多个 physical coordinates：

\[
L(x)=\{D(x)+r+O\mid r\in R\}.
\]

fileciteturn135file0L1-L7

但它刻意让 layout 主要成为 physical resource/storage contract。

之后 primitive dispatch：

\[
(Layouts,Scope,Target)
\rightarrow
Implementation
\]

每个 implementation 带：

\[
variant,\ priority,\ predicates.
\]

按 priority 检查，第一个 predicate 成功的实现胜出。fileciteturn107file0L1-L38

所以这里存在一个很明确的设计哲学：

\[
\boxed{
让 layout 保持 compositional，
execution mapping 在 primitive 内局部决定。
}
\]

CuTe 选择更强耦合。

Hexcute 则居中：

\[
Layout+TaskMapping+Instruction
\]

一起 inference。

这三者之间没有显然的赢家。

---

# 7. vLLM 的源码现在给出了一个尤其强的“假设不充分”证据

vLLM 的 KV layout enum：

\[
LBHNC,\ LBNHC,\ LHBNC,\ BLHNC,\ BLNHC,\ BHLNC
\]

同时显式定义：

- layer compact；
- block contiguous；
- block compact；
- block outermost。

fileciteturn113file0L1-L13

Backend 声明支持的 layout，系统取 intersection，再根据 preference 选一个；最后注释明确：

> Resolve one KV cache layout for the whole model.

fileciteturn96file0L1-L2

但是 connector path 又明确写：

> NIXL disaggregated P/D 中使用 LBHNC 能更快 transfer。

fileciteturn146file0L1-L2

更关键的是已经存在：

```text
kv_postprocess_layout_on_receive
kv_postprocess_blksize_on_receive
kv_postprocess_blksize_and_layout_on_receive
```

最后一个注释直接写：

```text
prefill is LBHNC, smaller block_size
decode(local) is LBNHC, larger block_size
```

fileciteturn146file0L1-L2

这很重要。

它说明：

\[
\boxed{
“一个全模型 layout 足够”
并不是系统普遍正确的事实；
现实中已经需要 boundary-specific conversion。
}
\]

但是这也不意味着：

> 应该永远每 phase 用不同 layout。

因为 conversion 也有成本。

这正是科研 trade-off。

---

# 8. SGLang 提供的是类似但不同的证据

SGLang 当前 `get_default_attn_backend()` 的逻辑直接根据：

- Hopper / Blackwell / HIP；
- MHA / MLA；
- speculative decoding；
- asymmetric KV；
- attention sinks；
- head number；

选择 FA3、FA4、TRTLLM、FlashInfer、AITER 或 Triton。fileciteturn143file0L1-L2

比如 SM100 MHA 有 asymmetric K/V 时，`trtllm_mha` 不适用，会改走 FA4；HIP MLA 当前 AITER 只在特定 head number 上采用，否则回 Triton。fileciteturn143file0L1-L2

同时 prefill/decode 可以不同 backend。fileciteturn124file0L1-L10

Persistent representation 又可能有 AITER SHUFFLE 5D，而源码说明：如果没有对应 consumer kernel，就退回 legacy NHD；另一个 preshuffle path甚至与 `page_size=64` 和特定 MQA/Gluon consumer 配对。fileciteturn114file0L1-L2 fileciteturn132file0L1-L27

这说明 SGLang 选择的是：

\[
\boxed{
backend specialization
+
representation specialization
}
\]

但只在手工已知的 model/hardware regimes 上开启。

---

# 9. 因此先区分：“确定性不足”与“真正科研 trade-off”

这是这轮最重要的一步。

| 现象 | 是否是科研问题 | 原因 |
|---|---|---|
| Hexcute cost model没有完整 bank-conflict/barrier | 否 | 增加模型项是确定方向，首先是工程改进 |
| Triton 存在不必要 `convert_layout` | 否 | 删除确定冗余 conversion 是确定性优化 |
| 某 backend 不支持某种 KV layout | 否 | 增加支持本身是工程覆盖 |
| SGLang AITER 只支持特定 head count | 否 | support limitation |
| TVM 某 ScheduleRule 没生成某 tile | 否，单独看不是 | 加 candidate 本身没有科研问题 |
| vLLM whole-model KV layout | **可能是** | 更细粒度 layout 有性能收益，也有迁移/管理代价 |
| Hexcute unify 多 consumer layout | **是潜在 trade-off** | compromise 可少转换，但 consumer-specific 可更快 |
| Triton insert/remove conversion | **是潜在 trade-off** | conversion 本身贵，但专用 layout 可能更快 |
| TileLang Pipeline→Layout | **是潜在 trade-off** | one-way optimization tractable，但不能自动反馈 layout resource pressure |
| CUTLASS L/I/P joint construction | **是潜在 trade-off** | 高质量但 expert/search burden 大 |
| TVM structural search space + measurement | **是潜在 trade-off** | 搜索成本可控，但可能失去 coverage |
| TIRx local primitive dispatch | **是潜在 trade-off** | modularity 很强，但 local choice未必 global-optimal |

后半张才是科研问题来源。

---

# 10. 科研问题 A：一个 tensor 的 layout 应该“稳定”多久？

我现在认为这是证据最强的问题之一。

不同框架已经选择了不同 granularity：

\[
\text{vLLM}: model\text{-}wide
\]

\[
\text{SGLang}: backend/phase\text{-}specific
\]

\[
\text{Hexcute}: tensor\text{-}level compromise
\]

\[
\text{Triton}: operation/consumer\text{-}specific
\]

\[
\text{Gluon}: programmer\text{-}selected
\]

真正问题不是：

> 专用 layout 比统一 layout 快吗？

而是：

\[
\boxed{
Layout specialization 的 optimal granularity
由什么决定？
}
\]

可以是：

\[
model
\rightarrow layer
\rightarrow phase
\rightarrow consumer
\rightarrow primitive.
\]

细粒度 specialization 得到：

\[
T_{\rm local}\downarrow
\]

但增加：

\[
T_{\rm convert}
+
M_{\rm extra}
+
C_{\rm metadata}
+
C_{\rm management}.
\]

Qwen3-8B 是很好的现实 workload，因为：

\[
H_q=32,\quad H_{kv}=8,\quad D=128
\]

也就是 GQA，一个 KV head被多个 Q heads共享；同一 KV representation 的多个 consumer/mapping天然存在。citeturn264176search1

DeepSeek-V3 则更极端：MLA 同时有 `kv_lora_rank=512`、`qk_nope_head_dim=128`、`qk_rope_head_dim=64`、`v_head_dim=128`，使 K/V representation 本身就非传统对称 MHA。citeturn264176search0

这个问题我给：

\[
\boxed{\textbf{S}}
\]

---

# 11. 科研问题 B：Compromise、Convert、Replicate 到底选哪个？

这是前一个问题更具体、也更 layout-centric 的形式。

假设 consumer：

\[
C_1,C_2
\]

分别希望：

\[
L_1^*,L_2^*.
\]

系统至少有三个合理策略：

\[
\text{Compromise: }L_c
\]

\[
\text{Convert: }L_1^*\rightarrow L_2^*
\]

\[
\text{Replicate: }\{L_1^*,L_2^*\}.
\]

现在已有框架刚好覆盖三种思路。

Hexcute 多 consumer constraint unification偏向 compromise。fileciteturn109file0L1-L2

Triton大量使用 conversion，然后再决定哪些 conversion 应该消掉。fileciteturn129file0L1-L22

TIRx 把 Replica 本身作为 layout 的一等结构。fileciteturn135file0L1-L7

Triton `ReduceDataDuplication`又说明 replication 也不是“越少越好”：它愿意引入 shared representation 来换取 register duplication 减少和 reuse。fileciteturn136file0L1-L30

vLLM transfer path则出现了 persistent conversion。fileciteturn146file0L1-L2

所以这里没有确定答案。

合理 objective 应该类似：

\[
J=
T_{\rm consumer}
+
T_{\rm conversion}
+
\lambda_RR_{\rm register}
+
\lambda_SS_{\rm shared}
+
\lambda_MM_{\rm persistent}.
\]

而 \(\lambda_M\) 在 standalone kernel 和 LLM serving 完全不同。

对于 GQA，这个问题尤其漂亮：

> 一份 KV 被多个 Q groups共享，是保持一份并通信，还是在 warp/warpgroup/register 层复制？

这个我也给：

\[
\boxed{\textbf{S}}
\]

---

# 12. 科研问题 C：Layout、Instruction、Pipeline 到底应该联合多少？

这个问题前面说“joint optimization”太粗。

源码现在支持更精确的表述。

CUTLASS：

\[
L+I+P+W
\]

高度耦合构造。fileciteturn110file0L1-L2

Hexcute：

\[
L+TaskMapping+I
\]

联合 inference，但 pipeline显式。fileciteturn109file0L1-L2

TileLang：

\[
PipelinePlanning
\rightarrow
LayoutInference.
\]

fileciteturn142file0L1-L26

Triton：

\[
Layout\ passes
\rightarrow
MMA
\rightarrow
Pipeline
\rightarrow
WarpSpecialization.
\]

fileciteturn122file0L1-L2

TVM：

由 ScheduleRule 决定哪些变量一起进入 search。fileciteturn126file0L1-L10

所以真正问题应该是：

\[
\boxed{
哪些 optimization variables 存在足够强的 interaction，
必须 joint optimize；
哪些可以安全 modularize？
}
\]

即研究 interaction graph：

\[
G=(D,E)
\]

其中：

\[
D=
\{L_S,L_R,T,I_C,I_M,P,W\}.
\]

如果：

\[
T(L_1,P_1)<T(L_2,P_1)
\]

但：

\[
T(L_1,P_2)>T(L_2,P_2),
\]

说明存在 rank reversal，应有 edge：

\[
(L,P)\in E.
\]

如果没有 reversal，可以拆开。

Attention 的：

\[
QK^T\rightarrow Softmax\rightarrow PV
\]

以及 Hopper/Blackwell 的：

\[
TMA/WGMMA/tcgen05+mbarrier+warp\ specialization
\]

是最适合研究的场景。

我给：

\[
\boxed{\textbf{S}}
\]

---

# 13. 科研问题 D：Layout 到底应该包含多少 execution semantics？

这是这轮新增后我认为很有价值的一条。

CuTe 倾向：

\[
Layout\approx Storage+WorkPartition.
\]

TIRx 倾向：

\[
Layout\approx StorageContract
\]

而 execution mapping由 primitive dispatch产生。

Hexcute介于中间：

\[
Layout+TaskMapping+Instruction
\]

一起 inference。

Gluon把 distributed layout明确暴露给程序员；`BlockedLayout` 直接 partition tensor across threads/warps/CTAs。fileciteturn149file0L1-L17

真正 trade-off：

layout semantic richness 越强：

\[
+\text{more precise optimization}
+\text{better legality propagation}
\]

但：

\[
-\text{larger search space}
-\text{architecture coupling}
-\text{less portability}
-\text{harder composition}.
\]

storage-only 越轻：

\[
+\text{modularity}
+\text{reuse}
\]

但把大量 performance-sensitive freedom留给 dispatch。

因此问题是：

\[
\boxed{
为了保留 near-optimality，
layout abstraction 最少需要编码多少 execution semantics？
}
\]

可以用：

\[
\mathcal M(L)=
\{\text{all legal mappings under the same layout}\}
\]

研究：

\[
\frac{\max_{m\in\mathcal M(L)}T(m)}
{\min_{m\in\mathcal M(L)}T(m)}.
\]

如果很接近 1，storage layout 足够。

如果差异很大，layout representation 太“贫”。

这个问题给：

\[
\boxed{\textbf{S-/A+}}
\]

理论价值很高，但实验设计会比前几个难。

---

# 14. 科研问题 E：Regularization vs Native Irregularity

这也应该重新表述。

不是：

> power-of-two 不够好。

而是：

\[
\boxed{
为了获得规则硬件 mapping，
应该牺牲多少 workload fidelity？
}
\]

几乎所有框架都依赖规则 hierarchy：

CUTLASS：tile/cluster/atom。

Triton：blocked distributed encoding，`num_warps` 当前还必须是 2 的幂。fileciteturn122file0L1-L2

TVM：固定 multi-level tiling skeleton。fileciteturn126file0L1-L10

Hexcute：instruction-compatible tile hierarchy。

vLLM/SGLang/FlashInfer：固定 page/block geometry。

而现代 LLM 正在增加：

\[
ragged\ sequence
\]

\[
GQA
\]

\[
MoE\ dynamic\ expert\ tokens
\]

\[
speculative\ q\ lengths
\]

\[
MLA
\]

\[
block\ quantization.
\]

DeepSeek-V3 有 256 routed experts，每 token 选 8 个 expert，同时使用 128×128 block-wise FP8 quantization。citeturn264176search0

所以 trade-off 是：

\[
\text{regular tile}
\Rightarrow
\text{better MMA/TMA/vectorization}
\]

vs

\[
\text{native irregularity}
\Rightarrow
\text{less padding/masking/imbalance}.
\]

这个问题是：

\[
\boxed{\textbf{A+/S-}}
\]

因为“规则性 vs irregularity”本身已有经典性，真正新意必须落在 GQA/MLA/MoE/paged KV layout。

---

# 15. 科研问题 F：Structural Prior 到底能替代多少 empirical search？

这也是所有框架共有。

CUTLASS Auto 用 static rules。fileciteturn148file0L1-L2

Triton用 compiler heuristics + optional autotuning。

TVM 用 schedule rules压空间，再 empirical measurement。fileciteturn118file5L81-L95

Hexcute先 constraint legality，再 analytic ranking。fileciteturn109file0L1-L2

TIRx是 predicate + priority。fileciteturn107file0L1-L38

vLLM是 compatibility intersection + preference。fileciteturn96file0L1-L2

SGLang是 model/hardware rules。fileciteturn143file0L1-L2

大家都在做：

\[
\boxed{
先相信结构知识，
避免测量整个空间。
}
\]

真正 trade-off：

\[
SearchCost
\quad vs\quad
PerformanceCoverage.
\]

但它不应叫“Coverage Regret”。

更好的科研问题是：

\[
\boxed{
Layout legality / hardware structure
对 high-performance region 到底有多强预测力？
}
\]

如果：

\[
P(\text{fast}\mid\text{structurally favored})
\]

很高，Hexcute/CUTLASS/TIRx式方法非常合理。

如果很低，TVM式 measurement 更必要。

我现在给：

\[
\boxed{\textbf{A}}
\]

因为这是很强的共同问题，但相对不够 LLM-specific。

---

# 16. 科研问题 G：Static Layout vs Runtime Adaptation

这一条主要是 serving。

SGLang 已经允许：

\[
Backend_{prefill}\neq Backend_{decode}.
\]

fileciteturn124file0L1-L10

vLLM layout却主要在 engine initialization resolve。

但 transfer path已经明确支持：

\[
prefill\ layout/block
\rightarrow
decode\ layout/block
\]

的 postprocessing。fileciteturn146file0L1-L2

这意味着实际系统已经处于：

\[
\boxed{
static\ representation
+
selective\ runtime\ adaptation
}
\]

状态。

真正问题不是“动态 layout 好不好”，而是：

\[
\boxed{
什么时候 specialization gain
足以 amortize migration cost？
}
\]

目标是：

\[
\sum_tT(x_t,L_t)
+
C_{\rm migrate}(L_{t-1},L_t).
\]

对长生命周期 KV cache，migration 可能很贵。

对分离 prefill/decode 节点，网络 transfer 本来就必须发生，此时顺手转 layout 的边际成本可能低很多。

这个条件性恰恰让它成为真正科研问题。

我给：

\[
\boxed{\textbf{S，LLM serving方向}}
\]

---

# 17. 哪些“框架自己已经在补偿”的地方尤其值得关注？

这是判断假设不充分的非常好证据。

| 原策略 | 后续补偿 | 说明 |
|---|---|---|
| Triton 各 pass 选择局部 layout | 多次 `RemoveLayoutConversions` | 局部目标之间确实产生冲突 |
| Triton distributed layout | `ReduceDataDuplication` | layout 可能导致过度 register replication |
| Hexcute constraint layout | 独立 bank-conflict resolver | legal layout 不等于性能最优 layout |
| TVM schedule-rule design space | hardware measurement | 静态规则不足以判断性能排序 |
| vLLM whole-model layout | receive-side layout/block conversion | single persistent layout 在 heterogeneous P/D 中不总是终点 |
| SGLang default backend | model/platform-specific overrides | 通用 backend preference 不足以覆盖模型差异 |
| TileLang PipelinePlanning | layout inference放在 pipeline 后 | pipeline structure会改变 layout constraints |
| CUTLASS Auto Builder | expert override接口仍长期存在 | static builder rules无法取代 expert space |

这种“**先采用简化假设，然后专门增加 repair mechanism**”比单纯 benchmark 慢更值得作为 problem evidence。

---

# 18. 当前我会怎样重新排序科研问题？

| 科研 trade-off | 跨框架支持 | 现代 LLM 压力 | 是否没有确定答案 | 当前优先级 |
|---|---:|---:|---:|---:|
| Representation stability vs specialization granularity | ★★★★★ | ★★★★★ | ★★★★★ | **S** |
| Compromise vs Convert vs Replicate | ★★★★★ | ★★★★★ | ★★★★★ | **S** |
| Selective joint optimization：Layout×Instruction×Pipeline | ★★★★★ | ★★★★★ | ★★★★★ | **S** |
| Layout semantic richness vs modularity/portability | ★★★★☆ | ★★★★☆ | ★★★★★ | **S-/A+** |
| Regularization vs workload irregularity | ★★★★★ | ★★★★★ | ★★★★☆ | **A+/S-** |
| Static representation vs runtime adaptation | ★★★★☆ | ★★★★★ | ★★★★★ | **S，serving** |
| Structural prior vs empirical feedback | ★★★★★ | ★★★★☆ | ★★★★★ | **A** |
| Locality/replication vs occupancy/capacity | ★★★★★ | ★★★★☆ | ★★★★★ | **A，但经典问题较多** |
| Local primitive optimum vs subgraph optimum | ★★★★☆ | ★★★★★ | ★★★★★ | **A+** |
| Data/scale/metadata co-layout | ★★★☆☆ | ★★★★★ | ★★★★★ | **A+，新兴** |

---

# 19. 对前面回答最大的修正

前面的主线：

\[
BoundaryRegret,\ CoverageRegret,\ SelectionRegret
\]

现在我不会再把它们称为主要科研问题。

它们应该变成诊断工具：

\[
\text{specialization trade-off}
\rightarrow
BoundaryRegret
\]

\[
\text{structural-prior trade-off}
\rightarrow
CoverageRegret
\]

\[
\text{analytic-vs-measurement trade-off}
\rightarrow
SelectionRegret.
\]

真正论文问题应该落在**为什么存在这些 regret，以及两种策略之间何时切换**。

---

# 20. 我现在认为最核心的总假设也需要改写

不是：

> “现有框架错误地分开优化 layout。”

而应该是：

> 现有框架为了让 GPU layout optimization 可实现，都在主动压缩 representation freedom：保持较稳定的 layout、偏好规则 tile hierarchy、限制候选结构、把 layout/instruction/pipeline 分阶段或局部决定。这些方法在传统规则 GEMM 上非常合理；但现代 LLM 引入 GQA、MLA、MoE、paged persistent KV、quantization metadata、phase-dependent serving 和 async pipelines，使这些压缩自由度的假设可能在部分 workload regime 中失效。

真正科研问题就是：

\[
\boxed{
\textbf{
Which layout freedoms are worth preserving,
and under what workload/hardware regimes?
}}
\]

也就是：

\[
\boxed{
\text{optimization tractability}
\leftrightarrow
\text{layout freedom}
}
\]

这比“找更好 layout”本身更有研究价值。

下一轮如果继续，我建议不查论文，而是对现在排在最前的 **Representation Stability、Convert/Replicate、Layout×Pipeline coupling、Structural Prior、Runtime Adaptation** 五个问题逐个做“证据审判”：把每个框架具体哪段源码支持哪一侧、框架自己的 fallback/repair 又说明什么、Qwen3/DeepSeek 等真实子图具体怎样触发 trade-off、什么观察会证明这个问题其实不重要，全部填成一张严格的 evidence matrix。

## 用户

没有达到标准之前 还是先别总结科研问题了，指导所有框架的统计达到标准了再提炼科研问题把！

## 模型回复

同意。先停止“提炼科研问题”。接下来只做**源码审计与框架统计**，直到比较矩阵足够细、每个结论都有源码支撑、各框架处在可比的共同坐标系里；在达到这个标准前，不再总结科研问题。

我会把“达到标准”定义得更严格：每个框架至少要覆盖 representation、decision ownership、candidate generation、legality、layout propagation/conversion、instruction selection、pipeline/warp specialization、performance selection、dynamic shape/runtime state、persistent layout、multi-consumer handling、fallback/repair mechanism、architecture-specific specialization 这 13 类，并明确哪些是“显式控制”、哪些“推导”、哪些“搜索”、哪些“委托”。

## 模型工具调用参数

{"paths":["GitHub"],"query":"fetch_file"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Hexcute、TileLang、FlashInfer 等项目细节

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 hexcute/hidet 相关仓库

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 TileLang 布局推断与流水线实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 TileLang 布局定义

## 模型思考摘要

搜索 Hexcute、Hidet、TileLang及FlashInfer项目与布局实现细节

## 模型回复

目前 TileLang 的源码补充了一个之前没有统计清楚的维度：它的 layout inference 不是单独的“布局选择器”，而是依赖已经经过 Blackwell 2SM lowering、pipeline planning、software-pipeline injection 后的 IR；另外 copy lowering 对某些 fragment/TMA index 又要求所有线程复制数据。也就是说，TileLang 的 layout 决策明显受 **pipeline structure、target lowering、replication semantics** 共同影响，这些都必须进入后面的统一统计表，而不能只写“有 LayoutInference”。 

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检查布局推理与图传播冲突 algori  go

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析 TileLang 布局推断与成本模型

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 TileLang 布局成本模型与性能排序

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检查 TileLang 布局成本模型源码

## 模型思考摘要

梳理审计范围
梳理了审计范围

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

继续审计各框架布局覆盖情况

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取并审计 vLLM KV 缓存布局文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计 FlashInfer 布局与规划器实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 FlashInfer 解码后端自动选择策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审查Gluon布局与Warp专化源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计 Triton GPU 编译优化流程

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计TIRx分块调度与布局

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取TIRx复制与GEMM布局匹配文档

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

继续完善 TIRx GEMM 实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计 Relax 图融合与布局优化

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计 SGLang 内存池布局与量化实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 `_use_aiter` 与 KV 缓存布局

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析 SGLang HND 内存池布局选择

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 SGLang KV 缓存布局实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 SGLang HND KVCache 配置实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 SGLang HND 分支逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计 CUTLASS CuTe 布局与集合构建器

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 CuTe Layout 映射与分区文档

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计 CUTLASS GEMM API 3.x 文档 વિગતો

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计Hexcute并汇总框架统计

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 vLLM 支持的 KV 缓存布局

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计 TVM RewriteLayout 实现

## 模型思考摘要

审计多框架 GPU 布局推理、KV缓存与性能优化实现

## 模型思考：工具活动

正在搜索 github hexcute hidet instantiate_auto_annotation.py layout synthesis

## 模型思考：工具活动

正在搜索 4 个网站

## 模型思考摘要

已搜索 4 个网站

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 CUDA 自动注解实例化代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 Hexcute 指令选择、代价模型与银行冲突代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析 CUDA Cost Model 实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计 Hexcute 银行冲突解析逻辑

## 模型思考摘要

获取并分析 CUDA 与 Hexcute 代价及银行冲突代码

## 模型思考用时

思考了 3m 1s

## 用户

还要总结每个框架中layout 优化相关的假设，规则和启发式策略，以及他们用于什么样的layout的选择

## 模型回复

会加，而且这应当成为统计表里的独立层，不再只写“谁拥有哪个 decision”。每个框架我会额外固定统计四项：**隐含假设（Assumption）→ 硬约束/规则（Rule）→ 启发式/优先级（Heuristic）→ 它实际作用在哪类 layout 选择上**。这样才能区分“框架不支持”与“框架主动采用了一种有偏好的策略”。

例如 TileLang 现在已经能看到很具体的三层策略：先由 strict/common/free 三档推导 layout；冲突时允许 reducer 向 fully-replicated layout 单调放宽；free mode 又用 register-count 或 I/O-aware traffic model给候选排序，而不是简单“自动推 layout”。 

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理各框架假设、规则与布局策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理各框架布局选择的假设与启发式策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取模型覆盖参数逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

Computing Framework Layout Selection Policies

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 MetaSchedule 调优流水线官方证据

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取完整 Meta Schedule 教程文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 CUTLASS 中 StageCountAuto 相关实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

汇总 Triton 编译器优化流程

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取并审查 NVIDIA 编译器实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Triton 自动调优器实现文档

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Gluon 布局与 Warp 专用化源码

## 模型思考摘要

完善框架审计

## 模型思考摘要

梳理并检索各编译框架布局策略及自动调优实现

## 模型回复

补充后的统计口径我已经按源码改成“决策机制表”，而不是“功能表”。下面我先给出这一版框架审计结果；这一轮只做框架事实和实现策略，不提炼科研问题。

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 FlashInfer 注意力后端选择启发式

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查找 determine_attention_backend 实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取并检查工具函数代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索函数相关代码片段

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 utils.py 后续代码行

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查找 vLLM 传输后处理实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 KV 传输工具函数文档字符串

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取下一段布局函数代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索CUTLASS阶段计数与启发式

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 CUTLASS 自动调度逻辑源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 `resolve_auto_encodings` 的启发式实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 中自动布局与编码解析实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton C++ 编译器 Pass 名称

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 Gluon Pass 详细信息

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

推断合并编码源

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 TileLang 成本模型与推理级别

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

总结 vLLM 注意力后端选择逻辑

## 模型思考摘要

检索并总结注意力后端与KV布局启发式实现

## 模型思考用时

思考了 2m 43s

## 模型回复

对，这一项必须单独统计。前面的矩阵只有“谁拥有 decision”，还缺了更关键的一层：

\[
\boxed{
\text{Assumption}
\rightarrow
\text{Hard Rule}
\rightarrow
\text{Heuristic / Priority}
\rightarrow
\text{实际选择哪一种 Layout}
\rightarrow
\text{Conflict / Fallback}
}
\]

只有把这一层补齐，后面才有资格讨论“这些假设是否充分”。因此现在仍然不提炼科研问题。

目前主矩阵我建议固定为 10 个系统：SGLang、vLLM、FlashInfer、CUTLASS/CuTe、Triton classic、Gluon、TVM MetaSchedule、TIRx、Hexcute、TileLang。它们不是同一层系统，所以所有结论都必须说明“选择的 layout 是 persistent KV、GMEM/SMEM、thread/value、register fragment 还是 schedule-derived layout”，否则会产生错误比较。

---

## 一、先统一“layout 选择”的细粒度分类

后面所有框架都按下面这些对象统计，而不是笼统写一个 `layout`：

\[
L=
\{
L_{\rm persistent},
L_{\rm page},
L_{\rm gmem},
L_{\rm smem},
L_{\rm thread/value},
L_{\rm reg},
L_{\rm MMA},
L_{\rm replica},
L_{\rm convert}
\}.
\]

再加上会直接改变 layout 最优选择的邻接决策：

\[
D=
\{
Tile,
CopyInst,
MMAInst,
Pipeline,
WarpRole,
Backend
\}.
\]

这里非常重要。例如 vLLM 的 `LBHNC/LBNHC` 和 Triton 的 `BlockedEncoding/NvidiaMmaEncoding` 根本不是同一层的东西。

---

# 二、CUTLASS / CuTe：规则不是“选 layout”，而是构造一个 coherent micro-kernel configuration

CUTLASS 目前最容易被错误总结成“手写 layout”。实际上它的 `CollectiveMma` 同时包含：

\[
Stride_{A/B},
TiledMma,
GmemTiledCopy_{A/B},
SmemLayoutAtom_{A/B},
SmemCopyAtom_{A/B}.
\]

也就是说 global stride、thread/value mapping、GMEM copy partition、SMEM layout、SMEM→REG copy 与 MMA mapping 本身就是同一个 collective configuration 的不同部分。fileciteturn176file0L1-L7

它的 layout-related assumption 是：

\[
\boxed{
\text{性能好的 layout 应与 tile、copy、MMA、pipeline schedule 共同构造。}
}
\]

而不是先找到一个抽象 layout 再选 instruction。

硬规则主要来自 architecture/instruction/type system。例如一个 `CollectiveMma` specialization 必须匹配具体 GPU architecture 和 algorithm；SMEM layout、copy atom 和 `TiledMma` 必须满足对应 specialization 的类型约束。Hopper 的 dispatch policy进一步同时携带：

\[
Stages,\quad ClusterShape,\quad KernelSchedule.
\]

fileciteturn176file0L1-L7

启发式策略主要集中在 `CollectiveBuilder`：

`StageCountAuto` 不是 autotuning。它根据单 stage 的 SMEM footprint，把 SMEM 使用最大化，而且官方文档明确说假设：

\[
\boxed{\text{1 threadblock / SM occupancy}}
\]

。fileciteturn190file0L1-L32

`KernelScheduleAuto` 也不是 hardware measurement，而是在当前参数允许的 schedule family 中选择“available schedule”。用户仍可以显式覆盖。Builder本身还明确不覆盖 expert `CollectiveMma` 的完整 design space。fileciteturn176file0L1-L7

所以 CUTLASS 实际选择的是：

| 类型 | CUTLASS 的选择 |
|---|---|
| GMEM layout | `StrideA/B` |
| CTA/cluster tile | `TileShape`, `ClusterShape` |
| thread/value layout | `TiledMma`, `TiledCopy` |
| SMEM layout | `SmemLayoutAtom` |
| SMEM→REG mapping | `SmemCopyAtom` |
| MMA-compatible register layout | `TiledMma` |
| pipeline | `Stages` |
| warp specialization | KernelSchedule tags |

其核心策略属于：

\[
\boxed{\text{expert construction + architecture-specific structural rules}}
\]

而不是 layout search。

---

# 三、Triton classic：大量 layout 规则是“局部目标驱动 + 后续 repair”

Triton现在的源码比“compiler 自动决定 layout”复杂得多。

CUDA TTGIR pipeline 实际顺序包括：

\[
Coalesce
\rightarrow
RemoveLayoutConversions
\rightarrow
OptimizeCTALocality
\rightarrow
RemoveLayoutConversions
\rightarrow
OptimizeThreadLocality
\rightarrow
AccelerateMatmul
\rightarrow
RemoveLayoutConversions
\rightarrow
OptimizeDotOperands
\rightarrow\cdots
\]

之后才进入 latency assignment、loop scheduling、pipeline、warp specialization 等阶段；末尾又继续 optimize dot operands、TMEM layouts、remove conversions、reduce duplication。fileciteturn192file0L1-L7

因此 Triton 的核心 assumption 是：

\[
\boxed{
\text{可以用多个针对局部目标的 layout transformation，
再通过 conversion elimination/rematerialization 协调冲突。}
}
\]

不同 pass 有不同的硬编码偏好。

`Coalesce`：

\[
\text{memory op}\Rightarrow \text{cache-friendly/coalesced layout}
\]

并允许前后插 `ConvertLayout`。fileciteturn163file0L1-L5

`AccelerateMatmul`：

\[
\text{dot}\Rightarrow
\text{hardware-accelerator-compatible input/output layout}.
\]

`RemoveLayoutConversions` 更直接地写明偏好：

\[
\text{expensive load/store}
\Rightarrow BlockedEncoding
\]

而：

\[
\text{tensor operation}
\Rightarrow NvidiaMmaEncoding.
\]

fileciteturn163file0L1-L5

`OptimizeThreadLocality` 又为 reduction、reshape、gather 选择减少跨线程通信的 layout，并明确使用 heuristics 判断何时为 gather 应用 warp-synchronous layout。fileciteturn163file0L1-L5

`ReduceDataDuplication` 会主动把：

\[
distributed\rightarrow dotOperand
\]

分解成：

\[
distributed\rightarrow shared\rightarrow dotOperand
\]

从而用 shared tensor + CSE 减少 register duplication。fileciteturn163file0L1-L5

所以 Triton 当前实际存在至少四类 layout preference：

\[
L_{\rm memory},
L_{\rm MMA},
L_{\rm reduction},
L_{\rm reuse}.
\]

它们并不天然相同。

Autotuner 是另一个层面。`Autotuner` 搜索的是用户提供的 `configs`，可以按 key 缓存，也允许 `prune_configs_by`；它不是把 TTGIR 内部所有 encoding 都作为自由 layout variable 搜索。fileciteturn193file0L1-L20

这一点以后必须一直保持区分：

\[
\boxed{
\text{Triton compiler-internal layout selection}
\neq
\text{Triton autotune configuration search}.
}
\]

---

# 四、Gluon：显式 layout 为主，但保留两种特殊的自动 layout

Gluon 比之前统计得更有层次。

它至少提供：

- `BlockedLayout`
- `DistributedLinearLayout`
- `DotOperandLayout`
- `NVMMADistributedLayout`
- Shared layouts
- `AutoLayout`
- `CoalescedLayout`

`BlockedLayout` 显式指定：

\[
sizePerThread,
threadsPerWarp,
warpsPerCTA,
order,
CGA.
\]

`DistributedLinearLayout` 更直接给出：

\[
reg\ bases,\ lane\ bases,\ warp\ bases,\ block\ bases.
\]

fileciteturn162file0L1-L7

所以它的主要 assumption 与 classic Triton 不同：

\[
\boxed{
关键 distributed layout 应由 programmer 显式控制；
compiler 主要负责验证与 lowering。
}
\]

但是 `AutoLayout` 和 `CoalescedLayout` 说明它并不是“所有 layout 都手写”。

当前 compiler 顺序先：

```text
InferCoalescedEncodings
ResolveAutoEncodings
```

再继续 lowering。fileciteturn192file0L1-L7

其中 `CoalescedLayout` 的实现非常具体：针对带 coalesced encoding 的 load/store，做 axis analysis，然后调用 `buildCoalescedEncoding`；产生的 concrete layout再向前/向后传播。fileciteturn211file0L1-L7

这里还有一个当前硬限制：

\[
numCTAs=1
\]

是 coalesced inference 当前实现的 assertion。fileciteturn211file0L1-L7

`AutoLayout` 则从 `set_auto_layout` 提供的 seed encoding出发，通过 layout inference在 IR 中传播。fileciteturn210file0L1-L7

因此 Gluon 不是简单的“explicit vs automatic”，实际是：

\[
\boxed{
\text{explicit-by-default}
+
\text{seed-based AutoLayout}
+
\text{memory-specific CoalescedLayout}.
}
\]

Warp specialization 又另外显式接受每个 worker partition 的 `num_warps` 和可选 `num_regs`。fileciteturn194file0L1-L24

也就是说 Gluon 当前把：

\[
Layout,\quad WarpRole,\quad RegisterBudget
\]

更多交给 programmer，而 classic Triton把其中更多部分交给 pass。

---

# 五、TVM MetaSchedule：layout search 之前首先由 ScheduleRule 定义什么“值得搜索”

TVM不能概括成“measurement-based layout optimizer”。

MetaSchedule 的真实流程是：

\[
ScheduleRule
\rightarrow
SpaceGenerator
\rightarrow
EvolutionarySearch
\rightarrow
XGBModel
\rightarrow
Builder/Runner.
\]

默认 `PostOrderApply` 根据 ScheduleRules 生成 design space，默认 `EvolutionarySearch` 用 cost model引导，最终通过真实硬件 measurement反馈。fileciteturn189file0L1-L7

因此最重要的 implementation assumption 是：

\[
\boxed{
\text{好的实现首先必须能够被 ScheduleRules 产生。}
}
\]

measurement只负责在生成出来的集合中排序。

`MultiLevelTiling` 自己已经预先固定了 layout/schedule family 的结构参数，包括：

\[
structure,
tile\_binds,
max\_innermost\_factor,
vector\_load\_lens,
reuse\_read,
reuse\_write.
\]

TensorCore版本再增加 intrinsic groups 和 software-pipeline choice。fileciteturn187file0L1-L15

`RewriteLayout` 也很值得注意。它不是 arbitrary layout search，而是只处理标成 `layout_free_buffers` / `layout_free_placeholders` 的 buffer，从 consumer 的访问索引推导 `IndexMap`；源码甚至注明它关注的 buffer预期只由一个 `BufferLoad` 读取。fileciteturn179file0L1-L7

如果中间存在 `cache_read` chain，则同一个 index map沿 cache-read chain 向前传播，并可以插 global cache-read/layout rewrite block。fileciteturn179file0L1-L7

因此 TVM 当前 layout相关策略至少分两层：

\[
\textbf{Schedule layout}
\]

由 tiling/binding/reuse/intrinsic search产生；

以及：

\[
\textbf{Buffer physical rewrite}
\]

从 anchor consumer的访问模式推导。

它们不是同一种优化。

---

# 六、TIRx：Layout 本身尽量描述 storage，具体执行由 primitive dispatch 补完

TIRx 的源码结构现在已经相当明确。

它的 primitive dispatch在 `LowerTIRx()` 的第一阶段运行。对每一个 `TilePrimitiveCall`，系统构造：

\[
DispatchContext=
(Target,Scope,LaunchParams,ValueRanges,Inter/IntraMaps)
\]

然后从注册的 implementation variants中选择。fileciteturn164file0L1-L7

这里的选择策略非常明确：

1. 按 `(Op,target)` 找 variants；
2. 如果用户显式 `dispatch=`，只留下对应 variant；
3. 按 priority 降序；
4. 检查 predicates；
5. 使用**第一个接受该调用的实现**；
6. `DispatchFail` 后继续尝试；
7. 最终没有合法 implementation 才失败。fileciteturn164file0L1-L7

因此 TIRx 的主要 assumption 是：

\[
\boxed{
storage/layout semantics 可以保持相对独立；
执行 implementation 可以通过 local primitive dispatch 推导。
}
\]

它的 copy dispatch已经能看出典型规则。

CUDA copy 当前有：

\[
vec16/32/64/128/256
\]

优先级 20；

`vec_auto`、`ldstmatrix` 优先级 10；

scalar fallback 优先级 0。fileciteturn165file0L1-L7

也就是说这不是 performance measurement，而是非常明确的：

\[
\boxed{\text{predicate legality + hand-coded priority}}
\]

策略。

GEMM规则更严格：当前 `mma.sync` path要求 operands在 register scope、完整 active warps、不能有 replica/broadcast axis，并要求：

\[
M\bmod16=0,\quad
N\bmod8=0,\quad
K\bmod8=0.
\]

它先尝试 `m16n8k16`，再尝试 `m16n8k8`，第一个能完整 tile operand layouts 的 instruction胜出。fileciteturn166file0L1-L7

所以 TIRx 中 layout决定的是 storage/fragment structure，而 dispatch规则选择：

\[
\text{vector width},
\text{thread partition},
\text{ldmatrix},
\text{mma instruction},
\text{loop decomposition}.
\]

这要和 CuTe 的工作划分方式严格区分。

---

# 七、Hexcute：最值得细分为“约束决定合法性 + heuristic补剩余自由度 + cost model排序”

Hexcute 当前源码中的四步已经相当清楚：

\[
LogicalShape
\rightarrow
LogicalLayout
\rightarrow
TaskMapping
\rightarrow
MemoryLayout.
\]

fileciteturn180file0L1-L2

Copy 使用 TV-layout constraint：

\[
f\circ p^{-1}
=
g\circ q^{-1}.
\]

MMA使用：

\[
f_{A,m}=f_{C,m},
\quad
f_{B,n}=f_{C,n},
\quad
f_{A,k}=f_{B,k}.
\]

如果 Copy 有多个 instruction candidates，源码明确写可以 DFS 找全部合法 variants 后用 cost model选，也可以用 simple heuristic / beam search剪枝。fileciteturn180file0L1-L2

Shared-memory layout也不是独立选择。对于一个 shared tensor：

\[
C_1,C_2,\ldots,C_n
\]

每个 Copy consumer都产生 memory constraint；系统不断 unify这些 constraints，冲突则 backtrack。约束都满足以后，**剩余未确定 stride 再用 heuristic 决定**。fileciteturn180file0L1-L2

Instruction selection进一步检查：

\[
scope,
alignment,
thread/value\ layout,
element\ width.
\]

例如 `CopyInstruction.logical_match()` 会 canonicalize thread/value layout，并检查它能否分解成对应 instruction fragment；源码甚至有单独逻辑处理 non-power-of-two tile的 corner cases。fileciteturn181file0L1-L2

Shared-memory swizzle又是另一层策略。`ResolveBankConflict`：

- 聚合同一 shared tensor 的所有 copy accesses；
- 枚举 swizzle candidates；
- 计算 warp bank conflicts；
- 选择所有 accesses 总 conflict较小的 swizzle。

TMA 有四种合法 swizzle，需要针对所有 consumers共同评估；如果 shared tensor直接参与 WGMMA，则该 swizzle可能被 instruction固定，tensor进入 immutable集合。fileciteturn183file0L1-L7

性能排序再使用 analytic cost model：

\[
cost
=
\#instructions\times instruction\ latency
\]

并区分 independent/dependent CPI。模型假设 copy/MMA pipeline可以充分 overlap；源码明确说 bank conflict、`cp_async_wait_group`、`mbarrier` 尚未完整进入 cost model。fileciteturn182file0L1-L7

所以 Hexcute 的完整策略应该统计成：

\[
\boxed{
Constraint legality
\rightarrow
search/prune
\rightarrow
analytic ranking
\rightarrow
bank-conflict swizzle refinement.
}
\]

不能再简化成“constraint-based layout inference”。

---

# 八、TileLang：现在看来比之前统计的复杂得多

TileLang 当前 `LayoutInference` 同时推 fragment layout 和 shared-memory layout。fileciteturn154file0L1-L7

而且它不是单阶段 inference。

当前 engine包含：

\[
Strict
\rightarrow Common
\rightarrow Free
\]

的分级推导。

源码中还存在非常具体的 conflict policies。

例如 reducer 的多个 update site产生不同 layout时，它不会直接失败，而可以沿一个单调 lattice：

\[
unset
\rightarrow narrow
\rightarrow FullyReplicated
\]

向 participant-wide replicated layout放宽。fileciteturn155file0L1-L2

如果两个非-fragment swizzle layouts冲突，则尝试 `MergeSwizzleLayouts`，合并成更小 granularity 的可兼容 layout；否则才 fatal conflict。fileciteturn155file0L1-L2

Floating fragment因为会在 TileOp之外访问，直接初始化为：

\[
FullyReplicated
\]

layout。zero-update reducer也会使用 wide replicated fallback。fileciteturn155file0L1-L2

Alias buffer又有一套特殊规则：相同 storage var但不同 dtype/shape时，layout propagation通过 storage-bit ratio进行 reshape，因此 fp4 等 sub-byte dtype不会简单按照 bytes处理。fileciteturn154file0L1-L7

更值得注意的是 TileLang **已经有 layout cost model**。

当前 free-mode支持至少：

- `register-count`：默认；
- `io-aware`：可选 global-memory traffic model。

fileciteturn156file2L41-L61

I/O-aware模型不是简单 count bytes，它根据 layout代数推导：

\[
(thread,slot)\rightarrow address
\]

然后估计：

- vector width；
- warp coalescing segments；
- per-thread instruction issue depth；

并使用：

\[
T(S)\approx \max(BW,Issue)
\]

再对 statement求和。对于 non-affine、swizzle、non-bijective等模型无法表达的情况，使用 conservative worst-case，而不是让“不透明 candidate”获得不公平低 cost。fileciteturn157file0L1-L2

Pipeline和layout的关系也很具体：

> pipeline planning/software-pipeline rewriting 必须在 LayoutInference 之前执行，使 inferred layouts看到最终 pipeline structure。

Blackwell 2SM lowering同样必须先执行，让 layout inference看到 `use_2cta`。fileciteturn152file0L1-L22

所以 TileLang 当前的假设是：

\[
\boxed{
先物化影响 layout constraint 的 execution structure，
然后做 whole-component layout inference/ranking。
}
\]

这和 Triton classic、Hexcute、CUTLASS都不一样。

---

# 九、vLLM：persistent layout 本身已经成为 compatibility negotiation

vLLM 当前 layout必须按源码重新描述。

其 logical cache统一写成：

\[
[L,B,H,N,C]
\]

但允许六种 physical stride orders：

\[
LBHNC,\;
LBNHC,\;
LHBNC,\;
BLHNC,\;
BLNHC,\;
BHLNC.
\]

并显式定义：

- layer compact；
- block contiguous；
- block compact；
- block outermost。

fileciteturn159file0L1-L7

所以 vLLM layout selection作用对象是：

\[
\boxed{
\text{persistent multi-layer paged KV physical stride order}
}
\]

而不是 kernel内部 thread layout。

每个 backend通过：

```python
supported_kv_cache_layouts()
```

声明支持 layout，按 preference排序。

系统先做：

\[
C=
\bigcap_i Supported_i.
\]

如果各 backend preference不同，则统计各 layout作为第一选择的次数，再排序 intersection中的 candidates。没有共同 layout直接报错。fileciteturn177file0L1-L7

这已经是一个非常明确的 heuristic：

\[
\boxed{
\text{compatibility intersection}
+
\text{first-choice voting}.
}
\]

接着还有额外规则。

如果模型里的 KV specs具有不同：

\[
(H,N,page\_bytes)
\]

shape，就只保留 `block_compact` layout，因为不同 HNC shape需要把每个 page当成连续 byte run才能 alias。fileciteturn177file0L1-L7

Explicit `VLLM_KV_CACHE_LAYOUT` 必须位于合法 candidate set。

KV connector也可以提供 preference；如果兼容就优先，否则 warning后回退到 candidate 0。

最终源码明确：

\[
\boxed{\text{Resolve one KV cache layout for the whole model.}}
\]

fileciteturn177file0L1-L7

不同 backend确实有不同硬需求。例如 CPU只接受 `LBHNC`；TurboQuant/HPC只接受 `LBNHC`；B12X接受 `LBHNC/BLHNC`；Flex Attention要求 `LBNHC`，因为只有这种 stride允许 `(B,N)` 零拷贝 flatten。fileciteturn178file0L1-L12 fileciteturn178file2L39-L50 fileciteturn178file3L60-L70 fileciteturn178file7L140-L150

Block geometry同样参与 layout legality。比如 manager block拆成更小 kernel blocks时，physical block必须是 dense、unpadded page，否则不能简单做 strided view；源码会要求改用更合适的 layout如 LBNHC或改变 block size。fileciteturn158file0L1-L2

而在 disaggregated KV transfer里，统一 representation又允许被显式修正：NIXL偏好 LBHNC做更快 transfer；接收侧存在 layout permutation和 block-size + layout联合 postprocess，源码直接给出：

```text
prefill: LBHNC + smaller block
decode:  LBNHC + larger block
```

。fileciteturn201file0L1-L7 fileciteturn202file0L1-L7

这部分以后必须记入“fallback/repair mechanism”，不能只统计 whole-model resolver。

---

# 十、SGLang：主要是 backend-native layout + hand-written applicability rules

SGLang 的 layout policy比 vLLM更偏 backend-specific。

当前 KV pool源码直接列出：

\[
NHD\;(\text{default}),
\quad
HND,
\quad
vectorized\_5d.
\]

HND把 `(page,head)` 折叠到 paged index，用于 per-KV-head sparse page table；`vectorized_5d` 是 ROCm AITER SHUFFLE representation：

\[
K:
(B,H,D_k/X,N,X)
\]

\[
V:
(B,H,N/X,D_v,X),
\]

其中：

\[
X=16/dtypeBytes.
\]

而且 HND 与 vectorized-5d互斥，HND优先。fileciteturn205file0L1-L37

另一个非常明确的 assumption 是：

> 如果没有消费某种 layout 的 kernel，该 layout 没有意义。

源码因此只有在 HIP+AITER路径启用 SHUFFLE 5D；其他平台会忽略这个选择并保持 legacy NHD。fileciteturn169file0L1-L15

所以 SGLang 的 layout策略更接近：

\[
\boxed{
\text{先确定 consumer backend/kernel family，
再采用它原生消费的 representation。}
}
\]

Backend选择本身又是大量 hand-written rules。

例如 MHA：

- Hopper + 合适 spec条件 → FA3；
- SM100 + compatible → TRTLLM MHA；
- 但 asymmetric K/V 时改 FA4；
- HIP → AITER；
- FlashInfer存在且无 attention sinks → FlashInfer；
- 否则 Triton。

MLA又有另一套 Hopper/SM100/HIP规则；HIP下 AITER目前只在特定 KV head counts上默认使用，否则回 Triton。fileciteturn186file0L1-L7

这意味着 SGLang的 heuristic不是 generic layout score，而是：

\[
\boxed{
H(
architecture,
attention\ type,
model\ feature,
spec\ mode,
kernel\ capability
)
\rightarrow Backend.
}
\]

然后：

\[
Backend\rightarrow PersistentLayout/KernelLayout.
\]

这和 vLLM“先做所有 backend的 layout compatibility negotiation”是显著不同的。

---

# 十一、FlashInfer：layout space较小，但把 layout与 specialized kernel family紧密绑定

FlashInfer当前基础 KV physical layout主要是：

\[
NHD
\]

和：

\[
HND.
\]

Paged KV时：

NHD：

\[
[numPages,pageSize,numKVHeads,headDim]
\]

HND：

\[
[numPages,numKVHeads,pageSize,headDim].
\]

fileciteturn195file3L77-L92

它的主要 assumption 是：

\[
\boxed{
Attention/KV workload 可以由少量规范化 representation
+
specialized kernel families覆盖。
}
\]

Backend `auto` selection也是 rule-based。

当前普通 attention path会检查：

- SM90；
- positional encoding；
- custom mask；
- FP16 QK reduction；
- Q/KV dtype；
- QK/VO head dims。

满足 FA3能力时用 FA3，否则退 FA2；FA3 prefill的 head dimension当前只接受：

\[
64,128,256
\]

以及 asymmetric：

\[
(192,128).
\]

fileciteturn199file0L1-L7

也就是说 FlashInfer的 backend/layout选择核心仍然是：

\[
\boxed{\text{capability rules}}
\]

而不是在 FA2/FA3/layout candidates之间运行 hardware autotuning。

另外它的 `plan()` / `run()` separation也很关键：ragged KV与paged KV会形成不同 plan；某个 wrapper如果为 ragged KV plan，就不能直接拿 paged KV run，必须重新 plan。fileciteturn160file1L12-L22

甚至 MoE tile geometry里也已经出现明确经验 heuristic：

\[
tokens/expert
\approx
\frac{tokens\times topK}{experts}\times1.3
\]

其中 `1.3` 被源码直接称为 expert imbalance factor，然后向上 pad 到下一个 power-of-two，并限制在 kernel支持的 tile范围。fileciteturn198file0L1-L2

这类规则以后也应该进入统计，而不是只统计 KV NHD/HND。

---

# 十二、现在可以得到一张更准确的“规则/启发式统计表”

| 框架 | 主要 layout 对象 | Hard rule 来源 | Heuristic / preference | Performance feedback | Conflict / repair |
|---|---|---|---|---|---|
| CUTLASS/CuTe | GMEM/SMEM/thread-value/MMA | template、instruction、architecture | StageCountAuto、KernelScheduleAuto、expert choice | 通常不在 Builder 内测量 | expert override |
| Triton | distributed encoding、MMA/reduction/memory layouts | IR verifier、hardware encoding | Coalesce、ThreadLocality、MMA preference、remat | optional user autotune | ConvertLayout removal、rematerialize、shared staging |
| Gluon | explicit register/lane/warp/CTA + shared layouts | layout type/IR verifier | AutoLayout propagation、axis-based CoalescedLayout | optional autotune | explicit programmer conversion |
| TVM MetaSchedule | tile/bind/reuse + rewritten buffer layout | ScheduleRule/postproc/intrin | rule-generated design space + XGB ranking | **真实硬件 measurement** | invalid candidate丢弃；database best |
| TIRx | storage/fragment layout | primitive predicates | fixed priority ordering | 无通用 measurement ranking | fallback primitive |
| Hexcute | task mapping + shared/register memory layout | constraint unification、instruction compatibility | DFS/beam/remaining-stride heuristic | analytic latency model | backtrack + swizzle refinement |
| TileLang | fragment/shared layout | operator inference constraints | strict/common/free、register-count / IO-aware | symbolic analytic model | replication widening、swizzle merge、fallback seed |
| vLLM | persistent multi-layer paged KV layout | backend support intersection、HNC/page constraints | first-choice voting、connector preference、default order | 无直接 performance search | receive-side layout/block conversion |
| SGLang | persistent backend-native KV representation | backend/kernel capability | architecture/model/phase rule table | 无通用 layout measurement | fallback backend/layout |
| FlashInfer | NHD/HND/page layout + specialized tile families | kernel capability | architecture/head-dim/dtype/backend rules | 通常 capability dispatch | fallback backend / re-plan |

这张表才是后面应该不断填细的核心表。

---

## 十三、目前我认为“框架统计是否达到标准”的状态

现在比前面明显好，但我仍然不认为已经完全达到你要求的最终标准，所以还不进入科研问题。

当前已经比较扎实的是：

\[
\text{CUTLASS},
\text{Triton},
\text{Gluon},
\text{TIRx},
\text{Hexcute},
\text{TileLang},
\text{vLLM},
\text{SGLang}.
\]

TVM还需要继续把 **Relax fusion → extracted task → schedule rule → tensorization/layout rewrite** 的前后关系查得更细，特别是一个 fused subgraph什么时候共享/重写 representation。

FlashInfer还需要进一步拆 `plan()` 内部的 page size、GQA group、split-KV、CTA/tile selection，而不只是 backend/NHD/HND。

CUTLASS也还需要把 Hopper/Blackwell 的 TMA/WGMMA/tcgen05 builder rules分开，因为当前 `StageCountAuto/KernelScheduleAuto` 总结仍然略粗。

最后，后续每个框架都必须形成类似：

\[
\boxed{
\text{Layout object}
\rightarrow
\text{Candidate source}
\rightarrow
\text{Hard constraints}
\rightarrow
\text{Heuristic}
\rightarrow
\text{Cost/measurement}
\rightarrow
\text{repair/fallback}
}
\]

这样的完整 decision trace。达到这一层之后，再横向统计哪些假设被多个框架共同采用、哪些框架选择了相反策略。到那时再谈科研问题，证据链才够完整。

## 用户

没有达到标准之前 还是先别总结科研问题了，直到所有框架的统计达到标准了再提炼科研问题把！
还要总结每个框架中layout 优化相关的假设，规则和启发式策略，以及他们用于什么样的layout的选择

## 模型思考摘要

继续审查缺口
继续审查缺口

## 模型回复

这轮继续只做“源码审计”，不提炼科研问题。我会把剩余薄弱处补到同一标准，尤其是 TVM、FlashInfer、CUTLASS Hopper/Blackwell、TileLang，然后给出一张“是否达到统计标准”的验收表；未达标的框架继续查，不提前收敛。

## 模型工具调用参数

{"paths":["GitHub"],"query":"fetch_file"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索Blackwell相关构建器与算子实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索CUTLASS、FlashInfer、TVM与TileLang实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 CUTLASS 中的 SM100 调度逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 SM100 GEMM 构建器代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

继续分析SMEM选择器与自动阶段配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析 FlashInfer 规划搜索与调度策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取调度器相关代码行

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 FlashInfer 预填充调度符号

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 PrefillBinarySearchKVChunkSize 引用

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取调度器文件第二部分

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 TVM 融合与调优机制

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Apache TVM MetaSchedule 与 Relax 集成

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查找 TVM Relax 集成代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 Relax 集成文件内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计 Relax FuseOps 相关代码路径

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 TileLang 布局推断与融合流程

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 TileLang GPU 编译器仓库

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析FA2启发式布局与工作负载分组策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 FlashInfer 工具文件内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查询 CUDA 默认调度规则配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 CUDA 调度规则代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取默认 CUDA 非 Tensor Core 配置代码

## 模型思考摘要

继续完善审查
继续完善审查工作

## 模型思考摘要

检索并分析Blackwell算子调度与编译优化实现

## 模型回复

这一轮我把“假设 / 规则 / 启发式”严格拆开，避免把源码中的 `static_assert`、capability check 和“作者默认这样更快”的策略混为一谈。统计口径如下：**硬约束**决定“能不能”；**规则**决定“合法集合里固定怎么选”；**启发式**决定“多个合法解里偏向谁”；**性能反馈**决定是否真的测硬件；**修复机制**记录前面选择冲突后怎么补救。

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 TIRx 调度与布局细节

## 模型思考摘要

检索了TIRx调度与布局细节

## 模型思考用时

思考了 2m 9s

## 模型回复

同意。现在先把“科研问题”完全冻结。下面只做框架统计和源码事实审计；在所有框架达到同一个细粒度之前，不再提炼科研问题。

我把统计标准固定成下面这套模板。以后每个框架必须同时回答：它优化的 layout 对象是什么；layout 如何表示；候选从哪里来；硬约束是什么；确定性规则是什么；启发式是什么；是否用真实性能反馈；layout 与 instruction/pipeline 的先后关系；多 consumer 冲突怎么处理；失败后如何 fallback/repair；哪些 workload/runtime 信息参与决策；最终这些规则分别用于选择哪一级 layout。

其中“假设”只表示设计上的结构前提，不等价于代码里的 correctness constraint；`static_assert`、capability check 一律单列成“硬约束”。

---

## 1. CUTLASS / CuTe

CUTLASS 不能再归类成“手工 layout”。它实际选择的是一个高度耦合的 micro-kernel configuration。

核心 layout 对象包括 GMEM stride、SMEM swizzle/layout、thread/value partition、TiledCopy partition、TiledMma partition、MMA operand representation，以及 cluster-level mapping。`CollectiveMma` 将 `TiledMma`、GMEM tiled copies、SMEM layout atoms、SMEM copy atoms放在同一个 collective 配置中。fileciteturn110file0L1-L2

其设计假设是：**高性能 layout 必须与 copy instruction、MMA instruction、tile hierarchy、pipeline schedule 一致地构造**。因此它不试图独立求一个抽象的“最佳 layout”。

硬约束非常多，而且明显 architecture-specific。例如 SM100 SMEM selector要求基本 tile mode 为 8 的倍数，然后按能否整除依次尝试 `SW128 → SW64 → SW32 → INTER`；TF32 MN-major甚至只能使用特定 `SW128_32B` layout。也就是说“选择 swizzle”首先是 instruction/tile divisibility problem。fileciteturn217file0L1-L7

SM100 的 1SM / 2SM UMMA同样有硬约束。1SM M tile只允许 64/128，2SM只允许 128/256，N必须是 8 的倍数且不超过 256。fileciteturn218file0L1-L2

`KernelScheduleAuto` 的一个非常具体的规则是：如果 cluster shape是静态的、cluster M能被 2 整除，并且 tile M能被128整除，则 Auto倾向生成 2SM MMA；否则回落到 1SM。动态 cluster因为编译期无法保证 2SM 条件，直接选1SM。这里不是benchmark后挑最快，而是结构规则。fileciteturn218file0L1-L2

`StageCountAuto` 同样是规则而非 search。官方文档明确说它根据每 stage 的 SMEM footprint 最大化 shared-memory usage，而且假设 1 threadblock/SM occupancy。fileciteturn190file0L1-L32

因此 CUTLASS 的分类应该是：

| 项 | CUTLASS/CuTe |
|---|---|
| 假设 | layout/copy/MMA/pipeline 应共同构造 |
| 硬约束 | architecture、datatype、tile divisibility、instruction shape、SMEM capacity |
| 规则 | SMEM atom最大可适配 swizzle；1SM/2SM可行性；TMA multicast选择 |
| 启发式 | StageCountAuto、KernelScheduleAuto、Builder specialization |
| 性能反馈 | Builder 本身通常无真实硬件测量；Profiler 可外部测 |
| 用于选择 | SMEM layout、TiledMma、TiledCopy、stage、cluster、warp-specialized schedule |
| repair/fallback | Auto fallback；或用户退回 expert API 显式覆盖 |

还有一个重要统计事实：`CollectiveBuilder` 不覆盖 expert `CollectiveMma` API 的完整设计空间，所以“Auto builder space”和“CuTe expressible space”必须永远分开统计。

---

## 2. Triton classic

Triton classic 的核心特征不是“自动 layout”，而是**多个局部目标 pass 连续改变 distributed layout**。

当前 NVIDIA TTGIR pipeline里明确存在：

\[
Coalesce
\rightarrow RemoveLayoutConversions
\rightarrow OptimizeCTALocality
\rightarrow RemoveLayoutConversions
\rightarrow OptimizeThreadLocality
\rightarrow AccelerateMatmul
\rightarrow RemoveLayoutConversions
\rightarrow OptimizeDotOperands
\]

之后再做 latency assignment、loop scheduling、pipeline、warp specialization，后部还会继续优化 TMEM layout、remove layout conversion、reduce data duplication。fileciteturn192file0L1-L7

因此它的设计假设是：**不同局部 objective 可以先分别改 layout，再通过 conversion-removal、rematerialization 和后续 pass 协调。**

它至少有四套明确的 layout preference：

| 局部目标 | 偏好 layout |
|---|---|
| GMEM load/store | coalesced/cache-friendly distributed encoding |
| Tensor Core dot | MMA-compatible encoding |
| Reduction/gather | 降低 cross-thread communication 的 encoding |
| Reuse/duplication | 降低 register duplication、允许 shared staging 的 encoding |

`Coalesce` 会为了 memory access主动修改 layout并插 conversion；`AccelerateMatmul` 又会修改 dot input/output layout去利用 tensor cores。`OptimizeThreadLocality`针对 reduction/gather重新布局。fileciteturn66file0L1-L7

`RemoveLayoutConversions` 因而不是普通 DCE，它承担的是不同 local layout objective之间的协调。源码/Pass说明里明确体现了 memory-friendly BlockedEncoding与 MMA-friendly encoding 的选择冲突。fileciteturn67file0L1-L7

`ReduceDataDuplication` 又会主动把某些 distributed→dotOperand路径经 shared memory重构，以减少 register duplication并利用CSE。fileciteturn136file0L1-L30

外部 autotune 是另一层机制。`Autotuner` 只搜索用户传入的 `configs`，并支持 `key` cache和 `prune_configs_by`；compiler内部所有 distributed encoding并没有自动成为 autotune变量。fileciteturn193file0L1-L20

因此 Triton 需要这样统计：

| 项 | Triton classic |
|---|---|
| 假设 | 局部 layout transformations + repair 足以形成好的整体实现 |
| 硬约束 | encoding verifier、instruction lowering、GPU architecture |
| 规则 | coalesce / MMA acceleration / locality passes 的 transformation rule |
| 启发式 | gather/reduction locality、rematerialization、conversion elimination |
| 性能反馈 | compiler内部主要规则；用户层可 autotune |
| 用于选择 | GMEM encoding、register/thread distribution、MMA operand encoding、TMEM layouts |
| repair | `ConvertLayout` elimination、rematerialization、shared staging、duplication reduction |

这里必须永久保持一个统计区分：

\[
\boxed{
\text{compiler internal layout selection}
\neq
\text{user autotune configuration search}
}
\]

---

## 3. Gluon

Gluon不是简单的“显式 layout Triton”。

其主要 layout types确实让 programmer直接控制 register/lane/warp/block distribution，例如 `BlockedLayout` 描述 `size_per_thread`、`threads_per_warp`、`warps_per_cta` 和 order。但同时又存在 `AutoLayout` 和 `CoalescedLayout`。

当前 lowering先跑：

\[
InferCoalescedEncodings
\rightarrow ResolveAutoEncodings.
\]

fileciteturn192file0L1-L7

`CoalescedLayout` 的选择规则非常具体：对被标记为 coalesced encoding 的 load/store，根据 AxisInfo analysis 和当前 `numWarps`、threads/warp、shape-per-CTA 调用 `buildCoalescedEncoding`，然后将该 encoding作为 seed前后传播。fileciteturn211file0L1-L7

目前这套自动 coalesced inference甚至有一个硬限制：

\[
numCTAs=1.
\]

fileciteturn211file0L1-L7

`AutoLayout` 则从 `set_auto_layout`产生的 seed encodings出发，通过 IR layout inference传播；如果最终还有 unresolved auto encoding，pass失败。fileciteturn210file0L1-L7

Warp specialization另外显式暴露 worker partitions的 `num_warps` 和可选 `num_regs`。fileciteturn194file0L1-L24

所以 Gluon 的完整策略是：

| 项 | Gluon |
|---|---|
| 假设 | 关键 distributed layout最好暴露给 programmer；简单/明显 layout可推导 |
| 硬约束 | layout type、encoding verifier、当前 coalesced inference 的 CTA限制 |
| 规则 | axis-analysis coalesced encoding |
| 启发式 | programmer经验 + autotune meta-config |
| 性能反馈 | 可借用 Triton autotune |
| 用于选择 | register/lane/warp/CTA layout；memory coalescing layout |
| repair | 显式 `convert_layout`；或用户改 layout |
| pipeline关系 | warp roles/register budgets显式 |

这一点与 classic Triton差异很重要：classic Triton主要让 pass拥有 distributed-layout决策，而 Gluon将大部分 ownership返还程序员，仅保留 auto/coalesced inference。

---

## 4. TVM MetaSchedule

TVM当前最容易被错误归类为“自动搜索 layout”。实际上它是**先人为限定 schedule/layout family，然后在该 family 内做搜索和硬件 measurement**。

默认流程是：

\[
ExtractedTask
\rightarrow SpaceGenerator
\rightarrow ScheduleRules
\rightarrow EvolutionarySearch
\rightarrow CostModel
\rightarrow Builder/Runner
\rightarrow Database.
\]

当前文档/源码默认 `PostOrderApply + EvolutionarySearch + XGBModel + real hardware Runner`。fileciteturn189file0L1-L7

这意味着 TVM 的最重要结构前提是：

\[
\boxed{
\text{真正优秀的 schedule/layout 应先能被 ScheduleRules 表达。}
}
\]

CUDA默认 ScheduleRules本身已经包含大量明确先验。例如普通CUDA `MultiLevelTiling`默认：

\[
structure=SSSRRSRS
\]

并绑定：

\[
blockIdx.x,\;vthread.x,\;threadIdx.x
\]

；vector load candidates是：

\[
\{1,2,3,4,8,16\}
\]

；read reuse被要求进入 shared level 4，write reuse要求 local level 3。CrossThreadReduction只搜索一组离散 thread extents。fileciteturn235file0L1-L7

TensorCore规则更明显：结构固定为 `SSSRRSRS`，预定义 WMMA/MMA intrinsic groups，read reuse必须进入 `shared.dyn`，某个版本write reuse必须进 shared，另一个版本不做write reuse但打开 software pipeline。fileciteturn234file0L1-L7

也就是说 MetaSchedule measurement并不是无先验地决定：

\[
L,\;Tile,\;Reuse,\;Intrinsic,\;Pipeline
\]

而是在规则预设的 family 中搜索参数实例。

`RewriteLayout`是另一套完全不同的机制。它只作用于 `layout_free_buffers/layout_free_placeholders`，根据单个 anchor consumer 的实际 indices使用 `SuggestIndexMap`，然后通过 `TransformLayout`重写 buffer；如果有 cache-read chain，同一个 layout transform向上游传播。fileciteturn179file0L1-L7

因此 TVM 至少有两个 layout mechanism，不能混写：

\[
\boxed{
\text{schedule-induced thread/cache layout}
}
\]

和：

\[
\boxed{
\text{buffer physical layout rewrite}
}
\]

Relax层又位于其上。MetaSchedule从 Relax程序提取 TIR tuning tasks，每个 task带 weight；后续 TuneContext默认只使用 `task.dispatched[0]`作为被搜索 workload。`module_equality`还可以是 structural或 anchor-block，用于不同融合上下文之间复用 tuning records。fileciteturn227file0L1-L7

所以完整统计应该是：

| 项 | TVM MetaSchedule |
|---|---|
| 假设 | ScheduleRules 能构造有价值的 design space；measurement用于空间内排序 |
| 硬约束 | schedule legality、postproc、tensor intrinsic、GPU verifier |
| 规则 | 固定 tiling skeleton、thread binds、reuse scopes、intrinsic groups |
| 启发式 | evolutionary search + XGB prediction + task weights |
| 性能反馈 | **真实硬件测量** |
| 用于选择 | tile、thread binding、cache/reuse、vector width、tensorization、pipeline |
| physical layout | `RewriteLayout` 从 consumer index map 推导 |
| repair | candidate postproc失败淘汰；无 tuning record时可用其他 scheduling path |

TVM至此已经达到我们定义的统计粒度。

---

## 5. TIRx

TIRx不能与 MetaSchedule合成一个框架统计，因为其 layout/dispatch哲学不同。

TIRx layout主要是 storage/resource contract，可以包括 Shard、Replica、Offset等结构。Replica不是副作用，而是 layout representation的一等组成。fileciteturn135file0L1-L7

随后 TilePrimitive dispatch根据：

\[
OperandLayouts
+
ExecutionScope
+
Target
+
LaunchContext
\]

决定实现。

其选择算法是确定性的：查询 `(Op,target)` 注册 variants，如果用户给了 `dispatch=` 就限制候选；否则按 priority排序，逐个检查 predicates，使用第一个成功实现；`DispatchFail` 后尝试下一个。fileciteturn164file0L1-L7

CUDA copy registry就是非常清楚的规则系统：多种 vectorized copy variant拥有较高 priority，auto/ldmatrix等居中，scalar fallback最低。fileciteturn165file0L1-L7

MMA dispatch同样不是 autotune。比如现有 `mma.sync` path要求 register operands、完整 active warps、不能有不允许的 replica/broadcast，并对 M/N/K divisibility做硬约束；之后按实现规则先尝试特定 MMA tile，再fallback。fileciteturn166file0L1-L7

所以：

| 项 | TIRx |
|---|---|
| 假设 | storage layout可以与具体 execution mapping分离；primitive dispatch可局部补全 |
| 硬约束 | operand layout/scope、target、instruction divisibility、active warp条件 |
| 规则 | variant predicates |
| 启发式 | hand-written priority |
| 性能反馈 | dispatch本身无通用 hardware measurement |
| 用于选择 | vector width、copy primitive、ldmatrix、MMA primitive、work partition |
| fallback | 下一个合法 primitive；最终 scalar/general fallback或失败 |
| pipeline关系 | pipeline/barrier/role assignment主要显式存在于程序 |

这与 CUTLASS 的“共同构造”以及 Hexcute 的“约束联合推导”形成了很清楚的事实差异。

---

## 6. Hexcute

Hexcute不能简化为“constraint solver选 layout”。

其当前流程明确分四步：

\[
LogicalShape
\rightarrow
LogicalLayout
\rightarrow
TaskMapping
\rightarrow
MemoryLayout.
\]

fileciteturn180file0L1-L2

Copy 的 task mapping满足 TV-layout composition约束；MMA对 A/B/C 的 M/N/K mapping建立一致性约束。多个 instruction variant都可能合法，源码明确说可以 DFS遍历，也可以 heuristic / beam search prune。fileciteturn180file0L1-L2

Shared tensor的 physical memory layout是通过所有访问它的 copy constraints共同 unify。每增加一个 consumer都会进一步约束 stride；冲突则 backtrack。当硬约束都满足以后，**剩余未决 strides 使用 heuristic补完**。这意味着 constraint系统决定的是合法性空间，不是全部性能最优性。fileciteturn180file0L1-L2

Instruction selection继续根据：

\[
scope,\quad alignment,\quad thread/value layout,\quad element width
\]

匹配 `ldg/stg/lds/sts/ldmatrix/cp_async/TMA/mma/wgmma` 等。对于 non-power-of-two tile还有专门 corner-case路径。fileciteturn181file0L1-L2

Bank-conflict optimization又是后续独立 pass。`ResolveBankConflict`按 underlying shared tensor聚合所有 Copy访问，枚举 swizzles并最小化整体 bank conflict。对于 TMA会比较四种合法 swizzle；WGMMA直接消费的 shared operand可能成为 immutable，其 swizzle不能自由改。fileciteturn183file0L1-L7

最终候选 ranking使用 analytic model：

\[
Cost =
\#Instructions\times InstructionLatency
\]

并显式建模部分 copy/MMA overlap；源码同时写明 bank-conflict影响、`cp_async_wait_group`、`mbarrier`没有完整包含。fileciteturn182file0L1-L7

因此准确统计是：

| 项 | Hexcute |
|---|---|
| 假设 | dataflow/pipeline固定后，可以通过约束联合求合法 task/memory/instruction configuration |
| 硬约束 | TV-layout一致性、alignment、instruction legality、多consumer memory constraints |
| 规则 | constraint propagation/unification/backtracking |
| 启发式 | remaining-stride heuristic、DFS/beam pruning、analytic cost |
| 性能反馈 | instruction microbenchmark-based analytic model，不是候选逐个实测 |
| 用于选择 | TV task mapping、register/shared layout、copy/MMA instruction、swizzle |
| repair | backtracking；后置 bank-conflict resolver |
| pipeline关系 | pipeline/dataflow主要在 auto-layout synthesis之外显式决定 |

这一层已经达到标准。

---

## 7. TileLang

TileLang也不能再写成一句“LayoutInference”。

当前 layout inference会同时处理 fragment/register distribution与shared layout，并采用分阶段的 inference策略。源码中可以看到 strict/common/free 类模式，而不是一次性求解。fileciteturn154file0L1-L7

冲突处理也有明确 policy。例如 reducer多 update sites推导出不同 layout时，可沿：

\[
unset
\rightarrow narrow
\rightarrow FullyReplicated
\]

方向放宽，保证参与线程覆盖正确。Floating fragment等场景也会使用 replicated fallback。两个可合并的 swizzled layouts可以尝试 merge，而不是立即判失败。fileciteturn155file0L1-L2

Alias layout propagation会考虑相同 storage var之间的 shape/dtype变化，特别处理 sub-byte dtype，而不是简单按 element count搬 layout。fileciteturn154file0L1-L7

Free-layout候选存在明确成本策略：

- 默认 `register-count`
- 可选 `io-aware`

。fileciteturn156file2L41-L61

I/O-aware模型会将 layout映射成：

\[
(thread,slot)\rightarrow address
\]

然后估计 vector width、warp memory segments、per-thread issue depth，并用类似：

\[
T(S)\approx \max(BW,Issue)
\]

排序。无法解析的 non-affine/swizzled候选采用 conservative cost，而不是给予零成本。fileciteturn157file0L1-L2

TileLang特别重要的一点是 pass顺序：pipeline planning/software-pipeline materialization以及 Blackwell 2SM相关 lowering需要先完成，之后再做 LayoutInference，以便 layout推导看到最终 pipeline结构和 `use_2cta` 等信息。fileciteturn152file0L1-L22

因此：

| 项 | TileLang |
|---|---|
| 假设 | 先物化影响 layout 的 execution/pipeline structure，再全组件推导 layout |
| 硬约束 | TileOp inference constraints、alias/storage consistency、target constraints |
| 规则 | strict/common/free inference、replication widening、swizzle merge |
| 启发式 | register-count 或 I/O-aware ranking |
| 性能反馈 | symbolic/analytic，不是逐 candidate hardware benchmark |
| 用于选择 | fragment/register layout、shared layout、replication/swizzle |
| repair | FullyReplicated fallback、merge swizzle、放宽 reducer layout |
| pipeline关系 | **pipeline/materialization → layout inference** |

这已经达到同一标准。

---

## 8. vLLM

vLLM优化的不是 kernel内部 layout，而是**persistent multi-layer paged KV physical representation**。

逻辑形状统一视为：

\[
[L,B,H,N,C]
\]

物理枚举包括：

\[
LBHNC,\;LBNHC,\;LHBNC,\;BLHNC,\;BLNHC,\;BHLNC.
\]

并且每个 layout还有 `layer_compact`、`block_contiguous`、`block_compact`、`block_outermost` 等属性。fileciteturn159file0L1-L7

每个 attention backend声明 ordered `supported_kv_cache_layouts()`。Resolver首先取所有 active backends支持集合的交集。

然后如果交集中有多个候选，采用 preference解决：统计 layout作为 backend第一选择的次数，再排序候选。也就是说当前启发式可明确写成：

\[
\boxed{
CompatibilityIntersection
+
FirstPreferenceVoting
}
\]

。fileciteturn177file0L1-L7

接着还有额外硬规则：不同 HNC shape共存时要求 layout具备 `block_compact` 属性，以保证不同 spec能够对 page storage进行安全 alias。Explicit layout override也必须在合法集合中。Connector可给自己的 preference，如果兼容则提升优先级。最终源码明确执行：

\[
\boxed{\text{one KV cache layout for the whole model}}
\]

。fileciteturn177file0L1-L7

不同 backend对 layout的需求并不相同，例如一些 backend只接受 LBHNC，一些接受/偏好 LBNHC，一些支持BLHNC等。因此 resolver是现实 compatibility negotiation，不是纯抽象设计。fileciteturn178file0L1-L13 fileciteturn178file2L39-L50

Backend本身又根据 `head_size`、dtype、KV dtype、block size、MLA、sink、sparse、sliding window、KV connector、PCP/DCP、adaptive verification等条件选择；并允许 per-KV-cache-kind backend override。fileciteturn213file0L1-L7

vLLM还必须统计一个重要 repair path：NIXL P/D transfer偏好 LBHNC以提高传输效率，但接收端允许 layout permutation、block-size转换，甚至源码明确描述：

\[
prefill:
LBHNC + smaller\ block
\]

\[
decode:
LBNHC + larger\ block.
\]

fileciteturn201file0L1-L7 fileciteturn202file0L1-L7

因此：

| 项 | vLLM |
|---|---|
| 假设 | persistent KV最好先找到所有消费者均兼容的稳定物理表示 |
| 硬约束 | backend support、HNC/page alias、block-compact等 |
| 规则 | support intersection |
| 启发式 | first-choice voting、connector preference、default order |
| 性能反馈 | resolver无候选实测 |
| 用于选择 | whole-model KV physical order、block/page compatibility |
| repair | transfer-side permute / block-size conversion |
| runtime visibility | 很高；但 layout resolution仍主要是初始化阶段 |

达到标准。

---

## 9. SGLang

SGLang同样主要涉及 persistent KV/cache representation，但其策略比 vLLM更偏：

\[
\boxed{
BackendSelection
\rightarrow
BackendNativeLayout
}
\]

。

当前 KV pool注释直接列出：

\[
NHD\;default,
\quad
HND,
\quad
vectorized\_5d.
\]

HND为 per-KV-head paged indexing服务；AITER的 vectorized-5D SHUFFLE针对 K/V采用不同物理排列，并且 HND与vectorized-5D互斥，HND优先。fileciteturn205file0L1-L37

其一个非常明确的设计规则是：没有 consumer kernel的 layout不启用。因此 AITER-only SHUFFLE representation在非相应 backend/platform下不会采用。

Backend selector完全是 hand-written capability/performance policy。例如当前 MHA规则大致包括 Hopper优先FA3；SM100可使用TRT-LLM MHA，但 asymmetric K/V切FA4；HIP走AITER；FlashInfer不可处理attention sink时fallback Triton。MLA还有独立规则，HIP/AITER对 head count存在限制，不满足时fallback Triton。fileciteturn185file0L1-L7 fileciteturn186file0L1-L7

而且 prefill/decode backend可分别指定。

因此：

| 项 | SGLang |
|---|---|
| 假设 | 先选择最合适的 backend/kernel family，再采用该 backend原生 representation |
| 硬约束 | GPU arch、attention architecture、head shape、spec mode、backend feature support |
| 规则 | architecture/model feature capability tree |
| 启发式 | hand-written “fastest backend” preference |
| 性能反馈 | runtime selector无通用候选benchmark |
| 用于选择 | backend；间接决定 persistent NHD/HND/SHUFFLE layout |
| repair | backend fallback、layout fallback |
| runtime visibility | 高，且 prefill/decode可分开 |

达到标准。

---

## 10. FlashInfer

FlashInfer之前统计过粗。源码现在能明确拆出三层：persistent representation、kernel/backend selection、attention workload scheduling。

基础 KV layout仍然主要是：

\[
NHD,\quad HND.
\]

Paged representation对应：

\[
[numPages,pageSize,numKVHeads,headDim]
\]

或者：

\[
[numPages,numKVHeads,pageSize,headDim].
\]

fileciteturn195file3L77-L92

Backend选择是 capability rule，不是 autotuning。当前 `determine_attention_backend()`会检查 GPU arch、positional encoding、custom mask、QK reduction mode、Q/KV dtype和QK/VO head dims。Hopper满足FA3能力时选FA3，否则FA2；FA3 prefill可接受的head dim组合还是离散集合。fileciteturn199file0L1-L7

但 FlashInfer还有更深的调度启发式。

Decode scheduler计算：

\[
vec\_size =
\max(16/sizeof(KV),HEAD\_DIM/32)
\]

再由 head dim、GQA group size生成 `bdx/bdy/bdz` 和 threads/block；通过 `cudaOccupancyMaxActiveBlocksPerMultiprocessor`得到最大grid capacity。若 batch×KV-head parallelism足够填满GPU，则不split-KV；否则对每个request的pages做二分，寻找KV chunk size，使拆分后的batch能够填充grid。fileciteturn220file0L1-L2

所以 FlashInfer的 split-KV不是“搜哪个更快”，而是基于 occupancy/grid utilization 的确定性 work-estimation heuristic。

Prefill更细。首先：

\[
packed\_qo\_len
=
qo\_len\times GQAGroupSize.
\]

然后 `CTA_TILE_Q`由 `FA2DetermineCtaTileQ`决定；非CUDA-graph情况下使用整个batch的**平均 packed query length**，CUDA graph下则使用uniform length或最大可变长度上界。之后再通过binary search决定KV chunk size，以控制 split后的work-item数量。fileciteturn223file0L1-L2

kernel实例化本身也根据 head dim只保留 `CTA_TILE_Q`的一小组可达候选；例如大VO dimension时允许的CTA Q tile集合会缩小，因为output fragment register pressure。fileciteturn231file0L1-L10

FlashInfer甚至在MoE工具函数里直接编码：

\[
tokens/expert
\approx
\frac{tokens\times topK}{experts}\times1.3
\]

其中 `1.3` 是经验 expert-imbalance factor，然后将tile tokens向上round到power-of-two并截断到kernel支持区间。fileciteturn198file0L1-L2

因此：

| 项 | FlashInfer |
|---|---|
| 假设 | Attention workload可通过少量规范layout + 专用kernel family + workload scheduler高效覆盖 |
| 硬约束 | NHD/HND、head dim、dtype、backend feature、kernel instantiation family |
| 规则 | FA2/FA3 capability dispatch、vector width、thread/block geometry |
| 启发式 | occupancy-driven split-KV、average/max Q length CTA tile、MoE 1.3 imbalance factor |
| 性能反馈 | 大多数选择不是在线benchmark |
| 用于选择 | KV NHD/HND、backend、CTA Q tile、KV chunk、split-KV geometry |
| repair/fallback | FA3→FA2、重新plan、不同 kernel family |

FlashInfer现在也达到统计标准。

---

# 11. 十个框架的“Assumption / Rule / Heuristic / Layout Target”总表

这是目前最关键的一张统计表。

| Framework | 设计假设/结构前提 | 硬规则 | 启发式/偏好 | 实际选择的 Layout |
|---|---|---|---|---|
| CUTLASS/CuTe | layout要和copy/MMA/pipeline共同构造 | instruction shape、tile divisibility、SMEM、arch | StageCountAuto、KernelScheduleAuto、swizzle优先级 | GMEM stride、SMEM swizzle、TV layout、MMA/copy mapping |
| Triton | 多个局部layout目标可通过后续repair协调 | encoding/instruction legality | coalesce、thread locality、MMA preference、rematerialization | distributed register/thread layout、MMA operands、TMEM |
| Gluon | 关键layout由expert显式控制更可靠 | explicit encoding verifier | AxisInfo coalescing、seed AutoLayout | reg/lane/warp/CTA distribution、coalesced layout |
| TVM | 高性能解存在于ScheduleRules定义空间中 | schedule/intrin/postproc legality | XGB、evolutionary search、task budget | tiling/binding/cache/reuse + buffer IndexMap |
| TIRx | storage layout可与execution mapping分离 | primitive predicates | priority ordering | storage/fragment layout；dispatch决定execution mapping |
| Hexcute | 固定dataflow/pipeline后可由constraint求mapping/layout/instruction | TV constraints、alignment、instruction legality | remaining stride、beam/DFS、analytic latency | TV mapping、SMEM/reg layout、instruction、swizzle |
| TileLang | 先确定pipeline/target structure，再推导layout | op inference/storage constraints | strict/common/free、register-count、I/O-aware | fragment/reg、shared、replicated/swizzled layout |
| vLLM | persistent KV优先采用跨backend统一兼容representation | backend/layout intersection、block/page alias | first-choice voting、connector preference | whole-model paged KV physical order |
| SGLang | backend-native representation优先 | backend/platform/model capability | hand-written backend preference | NHD/HND/AITER vectorized KV |
| FlashInfer | 少量KV layouts + 专用kernel families足够覆盖attention | kernel/head-dim/dtype capability | occupancy split-KV、Q-tile和chunk heuristics | NHD/HND + attention tile/work partition |

---

# 12. 再按“选择机制”统计，而不是按框架名统计

目前可以非常明确地分出六类 mechanism：

| Mechanism | 框架 |
|---|---|
| Expert explicit construction | CUTLASS/CuTe、Gluon |
| Progressive compiler rewriting | Triton classic |
| Rule-generated search space + measurement | TVM MetaSchedule |
| Local predicate/priority dispatch | TIRx |
| Constraint synthesis + analytic ranking | Hexcute |
| Whole-component inference + analytic layout cost | TileLang |
| Runtime compatibility/preference | vLLM |
| Runtime hand-written policy | SGLang、FlashInfer |

这只是实现事实分类，不是在评价哪种方法优越。

---

# 13. 所有框架还需要统一记录的 fallback/repair

这一列以前漏掉很多，现在必须保留：

| Framework | Repair / fallback |
|---|---|
| CUTLASS | 2SM→1SM、Auto→expert override、不同 specialization |
| Triton | ConvertLayout elimination、rematerialization、shared staging、duplication reduction |
| Gluon | programmer conversion / explicit replacement |
| TVM | postproc rejection、下一candidate、未tune函数保留/规则调度 |
| TIRx | 下一priority primitive / general fallback |
| Hexcute | backtracking、remaining-stride heuristic、bank-conflict swizzle |
| TileLang | replicated widening、swizzle merge、fallback layout |
| vLLM | next compatible layout、connector fallback、receive-side permute/block conversion |
| SGLang | backend fallback + representation fallback |
| FlashInfer | FA3→FA2、split/no-split、re-plan |

这一列后续特别重要，因为“一个框架需要专门repair什么”能揭示它前面哪些决策并非全局一致，但目前这里只记录事实，不做问题提炼。

---

## 14. 当前统计是否已经达到你要求的标准？

这一版之后，**10个框架已经第一次达到同一个“机制级”统计标准**：

\[
\boxed{
LayoutObject
\rightarrow
Representation
\rightarrow
CandidateSource
\rightarrow
Assumption
\rightarrow
HardConstraint
\rightarrow
Rule
\rightarrow
Heuristic
\rightarrow
PerformanceFeedback
\rightarrow
PipelineRelation
\rightarrow
Repair/Fallback
}
\]

之前确实没有达到：TVM只写了“会search”、FlashInfer只写了“NHD/HND”、TileLang只写了“LayoutInference”、CUTLASS只写了“expert layout”。现在这些都已经展开到实际规则层。

不过我还不会马上进入科研问题。下一轮仍应该做一次最后的**交叉校验**：把这10个框架按同一个具体 layout decision逐项横向比较，例如“GMEM→SMEM layout如何选”“SMEM swizzle如何选”“thread/value mapping如何选”“replication如何处理”“multi-consumer conflict如何传播”“pipeline何时冻结”“动态shape如何进入选择”等，检查有没有遗漏或把不同抽象层误当成同一件事。

等这一轮横向校验完成后，统计才算真正封版；之后再从源码证据中筛“假设不足/策略未定”的研究问题。

## 用户

先按照你说的做下一轮

## 模型思考摘要

明确下一轮核验
明确了下一轮核验方案

## 模型回复

这一轮我按“具体 layout 决策”横向核对，不再按框架逐个讲。目标是检查：同一个问题（如 SMEM swizzle、thread/value mapping、replication、multi-consumer、pipeline 次序、动态 shape）各框架到底怎么做，哪些其实不可直接比较，哪些是委托给下一层。这样可以把之前可能混淆抽象层的地方清掉。

## 模型工具调用参数

{"paths":["GitHub"],"query":"search"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析 TileLang 与 Hexcute 布局推理

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 TileLang 布局推断实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 TileLang 流水线执行顺序

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

对比复制指令与共享内存布局选择

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索共享编码规则与Swizzle实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Apache TVM 调度规则源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析TIR任务提取与融合决策

## 模型思考摘要

交叉核验版图
完成版图交叉核验

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

完成各框架布局机制交叉验证

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

跨框架验证 KV Cache 布局决策

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 SGLang 精确布局与内核实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 HND 折叠与页面头部实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 SGLang 注意力后端选择逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

评估多消费者布局传播策略

## 模型思考摘要

完成跨框架布局推理、调度实现与KV Cache策略验证

## 模型回复

交叉核对已经发现一个需要明确写进最终统计的边界：**“同名 layout 决策”在不同框架里经常不是同一层。**例如 vLLM 的 KV layout 是跨请求、跨层存活的 persistent stride order；Triton/TileLang 的 layout 是单 kernel 内 thread/register/shared 分布；CUTLASS 的 layout又和具体 MMA/TMA atom绑定。因此横向表里我会把“直接优化 / 间接决定 / 委托 / 不适用”分开，避免制造假对比。

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计并完善CUTLASS Blackwell布局矩阵

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 `_layouts.py` 中 `get_default_for` 的精确代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理低精度布局优化与跨框架假设

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 FlashInfer NVFP4 元数据布局

## 模型思考摘要

审计CUTLASS布局矩阵并检索默认实现与NVFP4布局假设

## 模型回复

横向校验基本完成了：现在能把“同一决策在不同框架里的处理”对齐到具体机制，而不是只按框架名归类。下一步我会在最终表里明确四种状态：**直接选择（O）/ 间接约束（I）/ 委托 backend 或用户（D）/ 不在该框架职责范围（—）**，并把之前容易混淆的 persistent layout、kernel-local layout、schedule-derived layout分开。

## 模型思考用时

思考了 2m 25s

## 模型回复

这一轮只做“横向源码校验 + 统计封版”，不提炼科研问题，也不评价哪种策略优越。前几轮最大的问题是按框架纵向总结比较充分，但还没有彻底解决“不同框架所说的 layout 到底是不是同一个东西”。这一轮主要把这个问题校正掉。

截至这一轮，我认为 10 个框架已经可以放进一个统一、但不混淆抽象层的统计坐标系。

## 1. 首先固定四个状态，否则后面的表会产生假比较

后续统一使用：

| 符号 | 含义 |
|---|---|
| O | Own：框架直接拥有/选择该 decision |
| I | Indirect：框架不直接选择，但其规则会间接决定 |
| D | Delegate：显式委托给 backend、kernel、programmer 或下一层 |
| — | 不属于该框架职责范围 |

另外还必须区分 layout 生命周期：

\[
L_{\rm persistent}
\neq
L_{\rm kernel}
\neq
L_{\rm shared}
\neq
L_{\rm register}.
\]

例如 vLLM 的 `LBNHC` 与 Triton 的 `BlockedEncoding` 不能放在一个“layout candidates”集合里比较。

前者生命周期可能跨：

\[
\text{many requests + many kernels + many layers}
\]

后者通常只活在：

\[
\text{one kernel invocation}.
\]

这是后续所有统计的第一原则。

---

# 2. 第一张横向表：每个系统到底在优化哪一级 layout？

| Framework | 主要 layout 生命周期 | Persistent/page | GMEM/physical | SMEM | Thread/Register | MMA/Instruction-compatible | Pipeline/role |
|---|---|---:|---:|---:|---:|---:|---:|
| CUTLASS/CuTe | kernel/collective | — | O | O | O | O | O |
| Triton classic | kernel | — | I/O | O | O | O | O |
| Gluon | kernel | — | O | O | O | O | O |
| TVM MetaSchedule | TIR task | — | O/I | O/I | O/I | O/I | O/I |
| TIRx | kernel/tile primitive | — | O | O | O | I | D/O |
| Hexcute | tile program | — | program给定 | O | O | O | D |
| TileLang | kernel component | — | I/O | O | O | I/O | O→layout |
| vLLM | model/runtime | **O** | **O** | D | D | D | D |
| SGLang | runtime/backend | **O/I** | **O/I** | D | D | D | D |
| FlashInfer | attention plan/kernel | consumes persistent KV | O | kernel internal | kernel internal | O/I | O/I |

这里最关键的修正是：SGLang/vLLM不能再和 CUTLASS/Triton 在“谁优化 SMEM layout”这一栏直接比较。

它们真正决定的是：

\[
\boxed{
\text{长期物理 representation}
+
\text{backend/kernel family}
}
\]

之后 SMEM/register mapping 交给 backend。

---

# 3. Persistent Layout / Page Geometry：实际上只有 serving/runtime 系统直接面对

这一层，低层 compiler基本没有对应的同类 decision。

### vLLM

当前 resolver明确写的是：

> `Resolve one KV cache layout for the whole model.`

fileciteturn244file0L1-L35

物理候选来自：

\[
LBHNC,\ LBNHC,\ LHBNC,\ BLHNC,\ BLNHC,\ BHLNC.
\]

不同 backend提供 `supported_kv_cache_layouts()`，resolver先做支持集合交集，再处理 preference。它还根据 `block_compact` 等 layout property约束不同 HNC specs能不能共享同一物理 page representation。fileciteturn177file0L1-L7

这一级选择使用的规则实际上是：

\[
\boxed{
Backend\ Compatibility
\rightarrow
Layout\ Intersection
\rightarrow
Preference
}
\]

而不是 kernel latency search。

同时 transfer path已经拥有显式 representation conversion。当前代码里的 `kv_postprocess_layout_on_receive()`做：

\[
[B,H,N,C]
\leftrightarrow
[B,N,H,C]
\]

方向的物理 permutation；更进一步的函数明确注明：

\[
\text{prefill}=LBHNC+\text{smaller block}
\]

\[
\text{decode(local)}=LBNHC+\text{larger block}.
\]

fileciteturn202file0L1-L2

所以 vLLM 需要统计两个东西，不能只记 resolver：

\[
\boxed{\text{steady-state persistent layout}}
\]

和：

\[
\boxed{\text{boundary conversion layout}}.
\]

---

### SGLang

SGLang 当前 memory pool直接承认三类 KV representation：

\[
NHD,\ HND,\ vectorized\_5d.
\]

HND用于将 page/head组织成适合 per-KV-head paged access的表示；AITER vectorized-5D则是更具体的 K/V SHUFFLE表示：

\[
K:(B,H,D/X,N,X)
\]

\[
V:(B,H,N/X,D_v,X).
\]

源码还直接注明 AITER representation在其他路径被忽略，因为：

> no consumer kernel.

同时 HND和vectorized-5D互斥，HND优先。fileciteturn246file0L1-L45

因此 SGLang的 representation选择不是全局 layout optimization，而更接近：

\[
\boxed{
ConsumerKernel
\rightarrow
NativePersistentRepresentation
}
\]

。

---

### FlashInfer

FlashInfer主要消费：

\[
NHD/HND
\]

paged KV，并把 page geometry作为 attention plan的一部分。

其 decode scheduler直接使用 `page_size`、每个请求的 page数量、GQA group和 occupancy 来决定是否 split KV以及KV chunk的大小。fileciteturn220file0L1-L2

所以与 vLLM 不同：

vLLM主要拥有 persistent allocation/layout；

FlashInfer主要拥有：

\[
\boxed{
\text{如何消费该 paged representation}
}
\]

的 work decomposition。

这三个框架现在应该按下面方式统计：

| Decision | vLLM | SGLang | FlashInfer |
|---|---|---|---|
| Persistent stride order | O | O/I | consume |
| Page/block size | O | O/I | input + scheduler |
| Consumer kernel family | O | O | O |
| Split KV | backend | backend | **O** |
| Receive-side layout conversion | **O** | specialized path | —/backend |
| Whole-model layout negotiation | **O** | no，同backend policy为主 | — |

---

# 4. GMEM → SMEM：这里各 compiler 才第一次真正进入同一个比较层

这是第一组可直接横向比较的低层 decision。

### CUTLASS

GMEM→SMEM transfer直接作为：

\[
GmemTiledCopy_A,\quad
GmemTiledCopy_B
\]

存在于 `CollectiveMma` 中。

Blackwell SM100更进一步根据 cluster shape 和 1SM/2SM MMA决定 TMA atom以及是否 multicast。

例如 2SM情况下 cluster满足一定几何结构时选择：

\[
SM100\_TMA\_2SM\_LOAD
\]

否则：

\[
SM100\_TMA\_2SM\_LOAD\_MULTICAST.
\]

动态 cluster由于编译时不知道最终 cluster geometry，会采用保守的 multicast路径。fileciteturn217file0L1-L7

这是：

\[
\boxed{
Tile/Cluster/MMA
\rightarrow
CopyAtom
}
\]

的规则式构造。

---

### Triton

Triton没有类似用户层统一 `GmemTiledCopy` 对象。

它先通过 `Coalesce`改变 memory operation的 distributed encoding，然后 pipeline阶段再判断 load能否变成异步 load。

当前 pipeline utility甚至写出了具体 heuristic：过小的 load不一定 pipeline，因为增加 register pressure可能反而变差；实现只允许满足一定访问宽度的load进入async path。fileciteturn240file0L1-L2

因此这里是：

\[
\boxed{
MemoryLayout
\rightarrow
CompilerAnalysis
\rightarrow
AsyncCopyEligibility
}
\]

。

---

### Gluon

Gluon主要把 shared descriptor、shared layout和数据搬运方式显式暴露给 programmer。

与此同时其 `CoalescedLayout`可以根据 AxisInfo自动推 GMEM访问分布。fileciteturn211file0L1-L7

---

### TVM

TVM通过 schedule决定：

\[
vector\ load\ width
+
shared\ cache\ read
+
thread\ binding.
\]

当前 CUDA `MultiLevelTiling`预先提供：

\[
vector\_load\_lens
=
\{1,2,3,4,8,16\}
\]

并显式要求 read reuse进入 shared层。fileciteturn235file0L1-L7

---

### TIRx

TIRx这里最明显是 primitive dispatch：

\[
Copy(src,dst,layout,scope)
\]

→ 根据 predicates选择具体 vector/ldmatrix/copy variant。

已有 copy registry给不同 vectorized variants更高 priority，general/scalar path作为fallback。fileciteturn165file0L1-L7

---

### Hexcute

Hexcute反过来从 Copy TV-layout constraints推导 task mapping和可匹配 instruction：

\[
(scope,alignment,TVLayout)
\rightarrow
ldg/lds/ldmatrix/cp.async/TMA...
\]

。fileciteturn181file0L1-L2

---

### TileLang

TileLang的 Copy op参与 layout inference，而不是事先拥有一个固定 copy mapping；其 layout inference甚至要求 thread extent在这一阶段为编译期常数。fileciteturn237file1L13-L20

因此 GMEM→SMEM 的横向差异现在可以精确写成：

| Framework | 主要机制 |
|---|---|
| CUTLASS | explicit TiledCopy + structural selection |
| Triton | memory encoding + compiler async-copy lowering |
| Gluon | explicit shared/copy + coalesced auto encoding |
| TVM | schedule vectorization + cache_read |
| TIRx | local primitive dispatch |
| Hexcute | TV constraint → instruction matching |
| TileLang | operator-driven layout inference |
| vLLM/SGLang | delegated |
| FlashInfer | specialized attention kernel内部固定 |

这才是真正的同层比较。

---

# 5. SMEM Layout / Swizzle：各框架的策略差异非常具体

这一栏之前总结得不够细，现在可以补齐。

### CUTLASS

SM100的 `sm100_smem_selector()`直接按 tile geometry选最大可用 UMMA SMEM layout。

例如 MN-major会按：

\[
SW128
\rightarrow
SW64
\rightarrow
SW32
\rightarrow
INTER
\]

依次找能够整除 tile 的最大 swizzle。

TF32某种MN-major甚至只有特定 128B方案合法。fileciteturn217file0L1-L7

即：

\[
\boxed{
\text{largest instruction-compatible swizzle}
}
\]

是显式规则。

---

### Triton

Triton拥有 `SwizzledSharedEncoding`。

但目前它并不是一个“枚举所有 swizzle然后测”的通用优化器。例如 pipeline utility中若无法推得更具体的shared layout，会构造 generic shared encoding，源码直接注明：

> This won't be optimal for 2D tensors.

fileciteturn239file5L93-L112

另外部分 encoding逻辑会基于连续性判断是否需要 swizzle；例如某些 K 不连续场景直接不 swizzle，因为本身访问已经落在不同banks。fileciteturn239file2L48-L63

---

### Gluon

Gluon将这个自由度暴露得更直接。

`NVMMASharedLayout`显式包含：

\[
swizzle\_byte\_width\in\{0,32,64,128\}
\]

以及 element bitwidth、transpose、FP4 padding等。

其 `get_default_for()`规则明确描述为：

> 选择与 block shape兼容的最大 swizzle，使TMA/MMA messages尽可能少。

fileciteturn250file0L2-L2

也就是说 Gluon有：

\[
\boxed{
explicit\ swizzle
+
default\ structural\ swizzle\ heuristic
}
\]

两种方式。

---

### Hexcute

Hexcute在这方面反而最像显式优化 pass。

它将访问同一 shared tensor的所有 Copy operations聚合，枚举合法 swizzle并计算bank conflict，然后选择整体冲突更低的candidate；TMA有自己的有限合法swizzle集合，而WGMMA operand可能因为instruction约束而不可再更改。fileciteturn183file0L1-L7

---

### TileLang

TileLang能够推shared layout，也有 swizzle conflict merge；多个 swizzled layout冲突时尝试合并为兼容粒度，不行才判冲突。fileciteturn155file0L1-L2

---

### TVM / TIRx

TVM更多通过 cache layout + tensor intrinsic schedule决定 shared-memory组织，并没有像 Hexcute那样一个统一“对所有consumer枚举swizzle”的主机制。

TIRx则让 storage layout显式存在，而特定 primitive implementation决定怎样消费它；swizzle通常属于具体 layout/primitive implementation contract。

因此这一栏不能简化成“所有compiler都会优化bank conflicts”。

它们实际上分别是：

\[
\text{CUTLASS: instruction-compatible structural choice}
\]

\[
\text{Triton: lowering/encoding rules}
\]

\[
\text{Gluon: programmer/default layout}
\]

\[
\text{Hexcute: explicit multi-access swizzle optimization}
\]

\[
\text{TileLang: inference + conflict merge}.
\]

---

# 6. Thread / Value / Register Mapping：这是框架差异最大的地方

这项非常重要，因为前面很多“layout优化”其实主要是在优化它。

### CUTLASS / CuTe

`TiledMma`、`TiledCopy`本身直接描述：

\[
(thread,value)
\rightarrow tile\ coordinates.
\]

因此 work partition与layout表达深度耦合。

---

### Triton classic

distributed encoding由compiler拥有。

Programmer通常操作：

\[
tensor\ block
\]

但：

\[
register,\ lane,\ warp
\]

到logical tensor element的分配由 TTGIR encoding决定。

随后 `AccelerateMatmul`、`OptimizeThreadLocality`等pass继续修改这些 encoding。fileciteturn66file0L1-L7

---

### Gluon

它把这一层重新暴露给programmer。

`BlockedLayout`明确表示：

\[
sizePerThread,\ threadsPerWarp,\ warpsPerCTA,\ order.
\]

而 `DistributedLinearLayout`更直接拥有：

\[
regBases,\ laneBases,\ warpBases,\ blockBases.
\]

fileciteturn250file0L2-L2

所以 Gluon的核心layout对象本身就是thread/value mapping。

---

### TIRx

这一点必须避免错误归类。

TIRx layout描述storage/resource placement，但primitive dispatch会进一步生成thread partition、loops和instruction。

所以：

\[
\boxed{
StorageLayout
\neq
完整WorkPartition
}
\]

是其设计边界。

当前 MMA dispatch甚至根据 operand layouts判断能否使用 `m16n8k16`/`m16n8k8` 等实现。fileciteturn166file0L1-L7

---

### Hexcute

Hexcute同时推：

\[
TaskMapping
+
MemoryLayout.
\]

Copy/MMA TV constraints会把 thread/value mapping与instruction fragment联系起来。fileciteturn180file0L1-L2

---

### TileLang

`Fragment`也是明确的：

\[
logical\ coordinates
\rightarrow
thread/index
\]

结构。

layout inference会为 local fragments推导这些映射，并根据 cost model选择free candidates。fileciteturn237file2L30-L38 fileciteturn237file10L204-L212

所以这一维可以非常清楚地分类：

| Mapping ownership | Framework |
|---|---|
| programmer/expert显式 | CUTLASS/CuTe、Gluon |
| compiler pass | Triton |
| schedule生成 | TVM |
| storage显式、execution局部dispatch | TIRx |
| constraint联合推导 | Hexcute |
| component-wide inference | TileLang |
| delegated | vLLM、SGLang |
| specialized kernel固定 | FlashInfer |

---

# 7. Replication / Duplication：以前经常没有单独统计，现在必须保留

这一维不是简单的“坏的冗余数据”。

各框架处理方法完全不同。

### TIRx

最明确：

\[
L(x)=\{D(x)+r+O\mid r\in R\}
\]

中的 Replica \(R\)就是 layout定义的一部分。fileciteturn135file0L1-L7

所以 replication是显式表达能力。

---

### Triton

distributed layout可能导致数据重复，因此有专门：

`ReduceDataDuplication`

pass。

它甚至可以改变：

\[
Register\ duplication
\]

与：

\[
Shared\ materialization
\]

之间的权衡，通过 shared intermediate换取重复减少。fileciteturn136file0L1-L30

---

### TileLang

replication不仅是优化结果，还是 inference fallback。

例如某些 reducer layout conflict会沿着：

\[
narrow
\rightarrow FullyReplicated
\]

放宽。

fileciteturn155file0L1-L2

---

### CUTLASS / CuTe

复制/broadcast可以通过 layout/tile partition表达，但当前Builder并没有独立的“replication optimizer”。通常作为expert mapping构造的一部分。

### TVM

复制更多体现为：

\[
cache\_read/cache\_write
\]

和schedule materialization，而不是统一的layout replica axis。

### Gluon

distributed layouts可以表达元素所有权和重复关系，但主要由programmer控制；当前没有一个类似 Triton `ReduceDataDuplication` 的通用自动pass作为主机制。

### Hexcute

TV mapping/layout inference可以产生/约束映射复用，但 replication不是像TIRx一样单独命名的一维全局policy。

### vLLM/SGLang/FlashInfer

它们的 KV resolver通常不把“同时存两套物理KV layout”作为普通候选；system-level TP/caching复制属于另一层，不应与register replication混为一谈。

因此：

\[
\boxed{
Register\ replication
\neq
Shared\ replication
\neq
Persistent\ KV\ duplication.
}
\]

后续统计必须分别记录。

---

# 8. Multi-consumer：这一栏尤其容易误判

这里现在可以更严格地区分。

### Hexcute

这是最明确的自动 multi-consumer layout propagation。

一个 shared tensor有多个copy consumers时：

\[
C_1(L),C_2(L),\ldots,C_n(L)
\]

会共同生成constraints。

系统逐步unify；冲突会导致backtracking。fileciteturn180file0L1-L2

Bank-conflict resolver同样会将所有访问同一shared tensor的copy一起评估。fileciteturn183file0L1-L7

---

### Triton

不是先为tensor求一个全局多consumer layout，而是不同consumer相关pass可能产生不同 encoding：

\[
Memory\ encoding
\]

\[
MMA\ encoding
\]

\[
Reduction\ encoding
\]

然后由 `ConvertLayout`、rematerialization、conversion removal协调。

因此这是：

\[
\boxed{
consumer specialization
+
conversion reconciliation
}
\]

。

---

### TVM

`RewriteLayout`主要围绕被声明为 layout-free 的buffer及其anchor access推 physical `IndexMap`；如果有 cache-read chain，则把相同transform向上游传播。fileciteturn179file0L1-L7

它与 Hexcute 的“所有 consumers联合constraint”不是同一种机制。

---

### TileLang

inference是whole-component传播，因此多个使用点可以产生conflict；系统有replication widening、swizzle merge等resolve机制。fileciteturn155file0L1-L2

---

### Gluon / CUTLASS

更多由programmer/expert显式协调。

### TIRx

更多通过primitive composition，以及显式layout/permute操作协调。

### vLLM

这里 multi-consumer 指的是完全不同生命周期上的consumer：多个 attention backends、KV connectors、mixed cache specs。

它做的是：

\[
\bigcap SupportedLayouts
\]

而不是 kernel-local constraint unification。fileciteturn244file0L1-L35

这三个机制现在必须分开命名：

\[
\boxed{
\text{Kernel consumer constraints}
}
\]

\[
\boxed{
\text{Encoding conversion reconciliation}
}
\]

\[
\boxed{
\text{Runtime backend compatibility negotiation}
}
\]

不能统称“multi-consumer layout optimization”。

---

# 9. Layout Conversion / Representation Change

这一栏也已经可以统一统计。

| Framework | Conversion mechanism |
|---|---|
| CUTLASS | explicit tiled copies/transforms；无统一通用 conversion optimizer |
| Triton | **first-class `ConvertLayout` + elimination/rematerialization** |
| Gluon | programmer explicit `convert_layout` |
| TVM | `TransformLayout` / cache materialization / schedule rewrite |
| TIRx | explicit layout/permute primitive组合 |
| Hexcute | constraint要求兼容；不兼容时backtrack或程序结构中需出现转换 |
| TileLang | inference传播 + copy/materialization，冲突时改变 inferred representation |
| vLLM | receive-side persistent KV permute/block reshape |
| SGLang | backend-specific write/read format、preshuffle路径 |
| FlashInfer | 多数通过plan/backend匹配现有NHD/HND，而非通用runtime converter |

这里 Triton和vLLM都有“conversion”，但二者成本量级完全不同。

Triton可能转换：

\[
\text{one tile in registers/shared}
\]

而 vLLM可能重排：

\[
\text{many persistent KV pages}.
\]

所以后续绝不能用“convert layout 次数”跨层直接比较。

---

# 10. Pipeline 与 Layout：这次横向校验后，次序已经可以精确统计

这是之前最值得修正的一栏。

### CUTLASS

最接近：

\[
\boxed{
Layout
\leftrightarrow
Pipeline
}
\]

因为 `Stages`、cluster、kernel schedule、SMEM layout、TiledMma/Copy共同存在于collective construction中。

而Blackwell每stage的shared-memory计算甚至包括：

- A/B storage；
- pipeline state；
- block-scale情况下 SFA/SFB storage。

fileciteturn216file1L19-L33 fileciteturn216file4L72-L82

---

### Triton

当前实际顺序更接近：

\[
Layout/Coalesce
\rightarrow
MMA
\rightarrow
Pipeline/WarpSpecialization
\rightarrow
later\ cleanup.
\]

fileciteturn192file0L1-L7

但 pipeline过程中仍会创建/选择 shared encodings，因此不能简单说 layout完全冻结在pipeline之前。fileciteturn239file5L93-L112

---

### Gluon

pipeline/warp specialization主要显式写在程序里。

Shared/register layout也显式。

因此这两个维度主要由programmer共同承担，而不是自动pass做joint search。

---

### TVM

software pipeline本身就是TensorCore `MultiLevelTiling` schedule rule的一个候选结构参数。

例如当前 CUDA TensorCore defaults中一条rule使用 shared read/write且不开software pipeline，另一条使用不同 write-reuse设置并开启software pipeline。fileciteturn234file0L1-L7

也就是说：

\[
Pipeline
\]

作为 schedule family的一部分进入design space。

---

### TIRx

pipeline/barrier/role assignment主要在native program显式存在。

Primitive dispatch不会负责全kernel pipeline search。

---

### Hexcute

自动 synthesis的主要边界是：

\[
Layout+Mapping+Instruction
\]

而 pipeline/dataflow预先由programmer固定。

---

### TileLang

这一栏源码最明确：

Blackwell 2SM lowering必须在 LayoutInference之前，让 `use_2cta` 对layout inference可见；PipelinePlanning和software-pipeline injection也在 LayoutInference之前完成，让inference直接看到最终pipelined structure。fileciteturn238file0L1-L26

因此它是：

\[
\boxed{
PipelineStructure
\rightarrow
LayoutInference
}
\]

。

---

# 11. Dynamic Shape / Runtime Information：各框架能看到的信息差别非常大

这一栏以前容易被漏掉，但对后续比较非常重要。

| Framework | layout选择时看到的动态信息 |
|---|---|
| CUTLASS | 大多compile-time template shape；可有dynamic cluster，但会影响可选路径 |
| Triton | JIT specialization/meta-key；kernel layout仍在compile/JIT阶段确定 |
| Gluon | 类似，layout大多constexpr/program configuration |
| TVM | extracted TIR workload + tuning context；非per-request online layout policy |
| TIRx | dispatch context可看到scope/launch params/value ranges，但选择仍是predicates/priority |
| Hexcute | compile-time tile program/constraint context |
| TileLang | inference context；thread extent当前要求常量 fileciteturn237file1L13-L20 |
| vLLM | model/cache/backend/runtime config很多，但persistent layout主要初始化时resolve |
| SGLang | model/hardware/spec/phase信息较丰富；prefill/decode backend可不同 |
| FlashInfer | **plan阶段直接使用每请求qo/kv lengths、page counts、GQA group、CUDA graph状态和GPU occupancy** |

FlashInfer这里很特别。

Prefill plan会先计算：

\[
packedQOLen=QOLen\times GQAGroupSize
\]

然后决定 CTA Q tile。

非CUDA graph状态使用batch平均 packed Q length；CUDA graph下根据uniform length或最大shape上界决定tile。之后再binary-search KV chunk size。fileciteturn223file0L1-L2

FA2 kernel实例化只保留 scheduler实际会选的 CTA Q tile，例如大VO dimension时只保留更小的 Q tiles，因为 output fragment register pressure。fileciteturn242file1L30-L57

所以 FlashInfer不是 layout compiler，但在：

\[
\boxed{
runtime\ workload\ geometry
}
\]

使用上，比大部分kernel compilers动态得多。

---

# 12. Graph / Subgraph Level：只有 TVM 真正需要单独标一层

CUTLASS/Triton/Gluon/TIRx/Hexcute/TileLang的主要分析单位基本仍然是一个kernel/tile program。

TVM则在其上还有：

\[
Relax
\rightarrow Fusion
\rightarrow ExtractedTIRTask
\rightarrow MetaSchedule.
\]

当前 `tune_relax()`先从Relax程序提取tasks，再为每个task建立TuneContext；每个task有weight，默认使用 `task.dispatched[0]`进入后续MetaSchedule。fileciteturn227file0L1-L7

Relax本身另有 `FuseOps/FuseTIR/FuseOpsByPattern` 等pass。fileciteturn228file1L19-L46

因此当前统计必须把：

\[
\boxed{
FusionDecision
}
\]

和：

\[
\boxed{
Schedule/LayoutDecision
}
\]

分成两个阶段。

这一点此前用一句“TVM支持graph-level”太粗了，现在已经修正。

---

# 13. 新增一列：低精度 data / metadata layout

这不是所有系统都涉及，但现代框架统计中不能省略。

CUTLASS Blackwell block-scaled kernels已经独立存在：

\[
Layout_{A/B}
\]

和：

\[
Layout_{SFA/SFB}.
\]

源码中的block-scaled collective明确分别构造scale-factor SMEM layouts及其TMA。fileciteturn251file7L177-L186

Gluon同样允许scale descriptor使用独立 `NVMMASharedLayout`；scaled tcgen05示例会专门为scale tensor调用 `NVMMASharedLayout.get_default_for()`。fileciteturn249file7L133-L145

FlashInfer的NVFP4工具链也已经显式区分scale-factor layout；量化实现有独立 `SF_LAYOUT` variants，benchmark还提供是否shuffle scale factors的选项。fileciteturn252file0L1-L29 fileciteturn252file4L101-L123

所以总表中以后不能只有：

\[
L_{tensor}
\]

还需要：

\[
L_{\rm data},
L_{\rm scale},
L_{\rm metadata}.
\]

目前这只作为事实统计，不做任何研究问题推断。

---

# 14. 最重要的横向“共同点统计”

现在可以从源码层面确认以下共同结构，但这里只是描述，不评价。

第一，所有系统都把 layout choice限制在某个结构化集合里，只是集合来源不同。

| 来源 | Framework |
|---|---|
| Expert/template family | CUTLASS |
| Encoding/pass transformations | Triton |
| Explicit programmer layouts | Gluon |
| ScheduleRules | TVM |
| Registered primitive variants | TIRx |
| Constraint-derived layouts | Hexcute |
| Component-wide inferred candidates | TileLang |
| Finite persistent layout enum | vLLM |
| Backend-native formats | SGLang |
| Small specialized layout/kernel family | FlashInfer |

第二，所有系统都把“合法”与“偏好”分开，只是方式不同。

\[
\text{CUTLASS: static\_assert \rightarrow AutoRules}
\]

\[
\text{Triton: IR legality \rightarrow heuristics}
\]

\[
\text{TVM: legal schedule \rightarrow XGB+measurement}
\]

\[
\text{TIRx: predicate \rightarrow priority}
\]

\[
\text{Hexcute: constraints \rightarrow analytic cost}
\]

\[
\text{TileLang: inference consistency \rightarrow cost model}
\]

\[
\text{vLLM: support intersection \rightarrow preference}
\]

\[
\text{SGLang: backend capability \rightarrow preferred backend}
\]

\[
\text{FlashInfer: kernel capability \rightarrow scheduler heuristic}.
\]

第三，不同系统都存在“前面做了选择，后面再修正”的路径，但修正对象不同：

| Framework | Repair |
|---|---|
| CUTLASS | Auto fallback / expert override |
| Triton | conversion removal / rematerialization / reduce duplication |
| Gluon | explicit programmer conversion |
| TVM | reject candidate / choose another schedule |
| TIRx | try next primitive |
| Hexcute | backtrack + swizzle refinement |
| TileLang | replication widening + swizzle merge |
| vLLM | transfer-side permute / block reshape |
| SGLang | backend/layout fallback |
| FlashInfer | split/no-split + backend fallback + re-plan |

这一列现在证据也已经比较完整。

---

# 15. 框架之间最核心的“差异统计”，目前也已经可以精确描述

不谈优劣，只说设计选择。

### Decision ownership

\[
\text{CUTLASS/Gluon}
\]

更多给expert。

\[
\text{Triton/TileLang/Hexcute}
\]

更多给compiler，但方法完全不同。

\[
\text{TVM}
\]

更多给search system。

\[
\text{TIRx}
\]

把storage和primitive implementation ownership拆开。

\[
\text{vLLM/SGLang/FlashInfer}
\]

更多在 runtime/kernel-family层决定。

---

### Candidate generation

\[
Template
\neq
Rewrite
\neq
ScheduleSearch
\neq
Dispatch
\neq
ConstraintSolve
\neq
Inference
\neq
RuntimeNegotiation.
\]

这是目前最清晰的一组框架差异。

---

### Layout semantic scope

从“layout包含多少东西”看，当前大致可以排列为：

```text
Persistent physical stride
vLLM / SGLang
        │
        ▼
Storage/resource layout
TIRx
        │
        ▼
Storage + inferred fragment distribution
TileLang / Hexcute
        │
        ▼
Explicit distributed thread/value layout
Gluon
        │
        ▼
Layout deeply embedded in tiled operation composition
CuTe/CUTLASS
```

Triton不完全落在这条线上，因为其layout semantic主要藏在compiler encoding中。

---

# 16. 本轮交叉校验后的“统计验收表”

现在按照我们之前约定的标准逐项检查：

| 统计项 | CUTLASS | Triton | Gluon | TVM | TIRx | Hexcute | TileLang | vLLM | SGLang | FlashInfer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Layout object定义 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 生命周期/scope | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Candidate来源 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Hard constraints | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Deterministic rules | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Heuristics | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Performance feedback | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| GMEM/data movement | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | D | D | ✓ |
| SMEM layout/swizzle | ✓ | ✓ | ✓ | I | I | ✓ | ✓ | D | D | internal |
| Thread/value mapping | ✓ | ✓ | ✓ | ✓ | dispatch | ✓ | ✓ | D | D | internal |
| Conversion/replication | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial | internal |
| Multi-consumer handling | explicit | ✓ | explicit | partial/anchor | primitive | ✓ | ✓ | runtime | backend | kernel |
| Pipeline relation | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | D | D | ✓ |
| Dynamic/runtime state | low | low | low | tuning-task | context | compile | compile | high | high | **high** |
| Fallback/repair | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Quant metadata layout | ✓ | ✓/arch | ✓ | partial | target-dependent | instruction-dependent | partial | backend-dependent | backend-dependent | ✓ |

这里的 `D` 和 `internal` 不是缺失。

例如 vLLM“不决定register layout”是正确的系统边界，不应该因此判未达标。

---

# 17. 这一轮以后，我认为“核心框架统计”已经达到可以封版的标准

和上一轮不同，这次不仅每个框架自己已经足够细，而且完成了你要求的第二步：

\[
\boxed{
\text{同一 layout decision 上进行横向比较}
}
\]

而且已经纠正了几个此前可能误导后续分析的地方：

1. `vLLM KV layout` 与 `Triton distributed layout`不是同一层。
2. CUTLASS的 Auto不是 generic performance autotuner，而主要是architecture-specific structural rules。
3. Triton不是“一次自动选择layout”，而是多组局部pass不断修改和repair。
4. Gluon不是纯手写layout，还存在 `AutoLayout`、`CoalescedLayout` 和default shared swizzle规则。
5. TVM的measurement发生在 ScheduleRules定义的空间之后，`RewriteLayout`又是另一条物理buffer rewrite机制。
6. TIRx layout并不等于完整execution mapping；primitive dispatch补全后半部分。
7. Hexcute constraint inference解决合法layout/mapping，而不是仅靠constraints直接决定全部性能。
8. TileLang不是单纯 LayoutInference，而是 pipeline/target lowering先物化，再做component-wide inference，并有明确cost model/fallback。
9. vLLM虽然resolve whole-model KV layout，但transfer边界已经显式支持layout/block conversion。
10. FlashInfer除了 NHD/HND外，还有很具体的runtime scheduling rules：GQA-aware packed Q length、occupancy-driven split KV、CTA tile heuristic、page/chunk planning。

这一版已经足够作为后续“审查假设和策略”的基础数据，而不需要再回到粗粒度框架描述。

下一轮仍然可以不提炼科研问题。更合适的是对这张已经封版的统计表做最后一个步骤：**逐条抽取每个源码规则背后的具体假设，但先只写“假设证据表”，不判断它是不是科研问题**。例如 `StageCountAuto` 的 1-CTA/SM 前提、TVM固定 `SSSRRSRS` schedule skeleton、vLLM whole-model layout、FlashInfer用平均 packed-Q length决定CTA tile、TileLang pipeline-before-layout、TIRx first-accepted priority dispatch等。这样下一阶段才有足够扎实的原始“假设库”。

## 用户

继续下一轮 请再核查对每一个框架的每一种启发式，每一种规则是否都完全整理，每一种假设，以及在完成layout选择的每一个策略，都做了细粒度的整理统计和对比？

## 模型回复

这一轮我会先做“完整性审计”，而不是直接补表。我不会声称已经覆盖“源码里每一条规则”——这需要逐文件静态分析；但我会把所有与 layout 决策相关的入口按固定 checklist 扫一遍，并明确标注：**已覆盖 / 仅部分覆盖 / 仍有缺口**。只有全部框架都达到“没有已知关键缺口”，才算真正封版。

## 模型工具调用参数

{"paths":["GitHub"],"query":"search"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 CUTLASS 自动调度与阶段计数实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 CUTLASS 中的 SM100 共享内存选择器

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 CUTLASS 中的 sm100_smem_selector实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计各框架启发式布局优化 completeness

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 中的布局 FIXME

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索Triton布局启发式与Pass实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 描述符内存布局与 TMEM 启发式

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

探索 Gluon 默认布局与合并启发式

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 默认块形状与 Swizzle 配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计TVM布局重写与调度启发式策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 TVM 中的 layout_free_buffers

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计 TilePrimitive 调度优先级与操作覆盖

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

系统搜索 TIRx 拷贝优先级与 GEMM 实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Hidet 中所有启发式实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Hidet 中的自动注解启发式实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索并定位 Hexcute Hidet 代码仓库

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审查TileLang启发式与约束完整性

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索布局成本模型相关代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审查 vLLM KV 缓存布局启发式

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 vLLM 中 KV 缓存布局支持

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计混合管理器块大小选择逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

全面复核 SGLang 缓存与后端策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 SGLang 模型覆盖注册表

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计 FlashInfer 全部启发式策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索并检查 FlashInfer 中的 FA2 启发式

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检查 FlashInfer 工具函数启发式细节

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 FlashInfer 解码阶段与共享内存配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

继续扫描框架规则与启发式

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 SM90 的 ss_smem_selector 定义

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 CUTLASS 中的 ss_smem_selector

## 模型思考摘要

审计多框架布局、调度与内存启发式实现

## 模型回复

目前完整性审计已经发现前一版**还不能算“每一种启发式/规则都整理完”**。至少有三个新增缺口：Triton 还有 descriptor-memory/TMEM 专门启发式；vLLM 除 layout resolver 之外还有 backend-specific `get_preferred_block_size()` 规则；FlashInfer 的 kernel specialization还包含 GQA group、head-dim、decode stage、CTA tile 等离散 dispatch 规则。接下来我会把这些遗漏补进“规则目录”，而不是只在总表里加一句话。

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 描述符与 TMEM 实现细节

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取并检查完整源文件内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 TMEM 启发式实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检视 Gluon 默认布局推断逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Gluon 默认布局推断实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 TVM CUDA 调度启发式策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计 CUDA 布局优化调度启发式

## 模型思考摘要

检索并梳理 Triton、Gluon 与 TVM CUDA 布局调度启发式实现

## 模型思考：工具活动

正在搜索 Hexcute GitHub hidet instantiate_auto_annotation.py layout cost_model resolve_bank_conflict 2026

## 模型思考：工具活动

正在搜索 github.com

## 模型思考摘要

已搜索 1 个网站

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Hidet 主分支中的 Hexcute 源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索代码库中的 auto_annotation 实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审核 CUTLASS 启发式规则完整性

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 EpilogueScheduleAuto 构建规则

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检查 CUTLASS Hopper 选择器与 Swizzle配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索并总结 CUTLASS 启发式布局策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

全面审计布局启发式规则

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取布局相关代码与通用重排逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计 Gluon 路径中的搜索启发式

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计TVM布局选择启发式策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 TVM 中 CUDA 平铺原语优先级配置

## 模型思考摘要

审计并总结CUTLASS与TVM布局启发式策略

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 github.com

## 模型思考摘要

已搜索 1 个网站

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取模型覆盖参数配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计框架布局变体与覆盖缺口

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计Triton布局变换源文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计 Triton 与 Gluon 布局策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 TVM 布局相关启发式与搜索

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

Fetching Hidet Auto-Annotation Implementation

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 HexCute 成本模型实现

## 模型思考摘要

核查审计范围
补充审计规则
补充布局审计
完善规则审计
继续核对完整性
继续完整性审计
完成审计范围、规则、布局及完整性核查

## 模型思考摘要

梳理并审计注意力布局及模型覆盖实现配置

## 模型回复

审计结果先给结论：上一轮说“可以封版”仍然太早。现在已经确认至少 6 个框架还有此前未纳入的规则族；因此这一轮我会把“完整”标准进一步收紧到**规则族/入口级穷举**，而不是只覆盖代表性规则。下面我会明确哪些框架已经接近完整、哪些还需要继续扫源码。

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考摘要

核查各框架启发式覆盖范围并确认仍有分支缺口

## 模型思考用时

思考了 6m 52s

## 模型回复

核查后的结论是：**还没有达到“每一种 layout 相关启发式、规则、假设都已整理”的标准。**上一轮把统计称为“封版”过早。这次继续扫源码后，又发现了多组此前没有纳入主表的规则，因此现在应该撤销“已经完整”的判断。

尤其新增发现集中在 CUTLASS、Triton、TIRx、vLLM、SGLang、FlashInfer。它们不是边角功能，而是会实际改变 layout、page geometry、thread mapping、SMEM swizzle 或 instruction mapping 的决策逻辑。

---

## 1. 先把“完整”的标准再提高一级

以后一个框架只有下面 14 项全部核完，才能标记为完成：

\[
\boxed{
Representation
\rightarrow
CandidateGeneration
\rightarrow
Legality
\rightarrow
Rule
\rightarrow
Heuristic
\rightarrow
Ranking
\rightarrow
Materialization
}
\]

具体要检查：

| # | 必须穷举的内容 |
|---|---|
| 1 | 所有 layout representation / encoding 类型 |
| 2 | 哪些 layout 由用户显式给，哪些自动生成 |
| 3 | candidate 产生入口 |
| 4 | 所有 correctness / legality constraints |
| 5 | 所有 deterministic selection rules |
| 6 | 所有显式 heuristic / priority |
| 7 | candidate pruning / fallback 顺序 |
| 8 | analytic cost model / learned model / hardware measurement |
| 9 | layout conversion / rematerialization / replication |
| 10 | SMEM bank-conflict / swizzle规则 |
| 11 | layout 与 pipeline / warp specialization 的先后依赖 |
| 12 | architecture / dtype / quant metadata 的特殊规则 |
| 13 | runtime / model / shape-specific override |
| 14 | user override、cache、fallback 和 repair |

而且每条都必须标注：

\[
\boxed{
\text{作用对象}
+
\text{输入条件}
+
\text{选择结果}
+
\text{规则性质}
}
\]

例如不能再只写：

> CUTLASS 有 SMEM swizzle heuristic。

必须写成类似：

> SM90 `ss_smem_selector`: 对 `GMMA::Major::MN`，在满足 tile divisibility 时按 SW128 → SW64 → SW32 → INTER 的顺序选择；这是 compile-time structural rule，作用于 GMMA operand shared-memory atom，不是 measurement-based choice。fileciteturn296file0L1-L2

这才达到所需粒度。

---

# 2. 当前完整性状态需要重新降级

我现在给的状态是：

| Framework | 上轮判断 | 本轮重新判断 | 主要原因 |
|---|---:|---:|---|
| CUTLASS/CuTe | 接近完成 | **部分完成** | Hopper copy/swizzle、epilogue、低精度规则此前遗漏 |
| Triton classic | 接近完成 | **部分完成** | GenericSwizzling、Descriptor layout、TMEM、pipeline heuristics此前遗漏 |
| Gluon | 接近完成 | **接近完成** | 仍需扫全部 automatic/default layout helpers |
| TVM MetaSchedule | 接近完成 | **接近完成** | RewriteLayout已深入；仍需系统化所有CUDA rule families |
| TIRx | 接近完成 | **部分完成** | primitive variants远比此前统计的多 |
| Hexcute | 接近完成 | **接近完成** | inference/cost已很细；仍需指令选择和bank resolver逐规则目录 |
| TileLang | 接近完成 | **接近完成** | cost model已细；仍需完整枚举 inference/fallback path |
| vLLM | 完成 | **部分完成** | 遗漏 backend-specific block/page-size policy |
| SGLang | 完成 | **部分完成** | 遗漏大批 model-specific override providers |
| FlashInfer | 完成 | **部分完成** | 遗漏大量 kernel-specialization dispatch rules |

所以：

\[
\boxed{\text{目前还不能进入下一阶段。}}
\]

---

# 3. CUTLASS：本轮发现之前漏了相当重要的一组规则

之前我们重点统计了 Blackwell `sm100_smem_selector`、StageCountAuto、KernelScheduleAuto。

但 Hopper `sm90_common.inl` 本身已经包含一整套 layout heuristics。

### 3.1 GMMA major mode选择

输入 global layout 并不直接原样映射到 GMMA major。

例如 A/B 的 MN-major只有在 datatype允许时才映射为：

\[
GMMA::Major::MN.
\]

TF32、部分整数/8-bit类型会被规则改成：

\[
GMMA::Major::K.
\]

fileciteturn296file0L1-L2

因此这里有：

\[
L_{\rm GMEM}
+\text{dtype}
\rightarrow
L_{\rm GMMA-major}.
\]

这是一个之前没单独统计的 layout-selection rule。

---

### 3.2 TMA vs TMA multicast

当前 Hopper helper：

\[
clusterExtent=1
\Rightarrow
SM90\_TMA\_LOAD
\]

否则：

\[
SM90\_TMA\_LOAD\_MULTICAST.
\]

fileciteturn296file0L1-L2

所以 cluster layout 会直接决定 copy semantics。

之前仅写：

> CUTLASS选择TMA copy。

过粗。

---

### 3.3 SIMT GMEM tiled-copy 有明确 coalescing heuristic

源码注释非常直接：

> Maximize the number of threads along the gmem major mode to promote coalesced reads.

然后根据：

\[
ThreadCount,\ Alignment,\ TileMN,\ TileK,\ Stride
\]

计算 major/minor thread layout，同时要求整个 threadblock tile可以整除。fileciteturn296file0L1-L2

所以这是一个真正的：

\[
\boxed{\text{thread-layout heuristic}}
\]

而不是单纯合法性约束。

---

### 3.4 Hopper SMEM selector实际上有不止一种

此前只整理“最大 compatible swizzle”，不够。

`rs_smem_selector()` 某些场景明确**优先 SW32**，原因源码直接写：

> due to free bank conflict.

而对 int8/fp8 transposed-B 等情况，又有专门 SW128要求。

另一方面 `ss_smem_selector()` 更接近：

\[
SW128\rightarrow SW64\rightarrow SW32\rightarrow INTER
\]

的 largest-compatible规则。fileciteturn296file0L1-L2

所以 CUTLASS 至少需要拆成：

\[
\text{SS-GMMA SMEM rule}
\]

和：

\[
\text{RS-GMMA SMEM rule}.
\]

不能统一叫“largest swizzle”。

---

### 3.5 Epilogue 还有独立 layout/schedule decision

之前主表主要统计 mainloop。

但 CUTLASS 还有：

\[
EpilogueScheduleAuto
\]

以及：

\[
sm90\_get\_epilogue\_smem\_swizzle\_layout\_atom.
\]

fileciteturn294file0L1-L10 fileciteturn293file1L35-L66

所以以后 CUTLASS 必须至少分：

```text
Mainloop layout policy
Epilogue layout policy
Aux tensor layout policy
Scale-factor layout policy
```

之前这块明显统计不足。

---

# 4. Triton classic：遗漏比预计还多

这是目前需要继续重点审计的框架之一。

以前集中在：

- Coalesce
- AccelerateMatmul
- OptimizeDotOperands
- OptimizeThreadLocality
- RemoveLayoutConversions
- ReduceDataDuplication

但源码里还有不少独立 layout heuristic。

---

## 4.1 Generic shared-memory swizzle 已经有自己的优化目标

`GenericSwizzling.cpp` 并不是简单构造一个swizzle。

源码明确写：

> Current heuristic: Minimise total bank conflicts.

平局时又看：

> number of rounds we do to move the data.

fileciteturn298file0L1-L16

完整算法还计算 source/destination bank conflicts和segment basis。fileciteturn299file0L1-L2

因此 Triton现在至少有一个：

\[
\boxed{
\text{bank-conflict minimization heuristic}
}
\]

此前主表没有把它单独列出。

---

## 4.2 TMA descriptor layout 又是一套独立选择策略

`OptimizeDescriptorEncoding.cpp` 很重要。

对于 TMA：

1. 只接受 non-transposed `NVMMASharedEncoding`;
2. 先尝试默认 shape/order-based preferred encoding；
3. 如果它和已有 SharedLinearLayout等价，直接采用；
4. 否则枚举：
   \[
   fp4Padded\in\{false,true\}
   \]
   和
   \[
   swizzle\in\{0,32,64,128\}
   \]
   找到第一个语义等价的candidate；
5. 最后还有fallback default encoding。

fileciteturn285file0L1-L13

所以 Triton的shared layout还要进一步拆：

\[
\text{ordinary shared layout}
\]

\[
\text{descriptor/TMA shared layout}
\]

\[
\text{MMA shared layout}.
\]

---

# 5. Triton Blackwell TMEM 还有非常具体的 register/layout heuristic

这是之前完全没有进入主统计表的。

`TensorMemoryUtils.cpp`在选择 tcgen05/TMEM load-store vectorization时直接写：

> Do not use more than half the registers as otherwise it's prone to spilling.

因此：

\[
maxReg=\frac{maxnreg}{2}.
\]

如果：

\[
maxnreg=256
\]

且一次需要多个message，源码又主动把vectorization减半，因为：

> ptxas' scheduler breaks.

fileciteturn286file0L1-L6

然后实现按固定顺序尝试：

\[
I32x32b,
I16x256b,
I16x64b,
I16x128b
\]

再fallback到：

\[
I16x32bx2.
\]

这实际上是：

\[
L_{\rm register}
+
L_{\rm TMEM}
+
register\ budget
\rightarrow
instruction\ shape/vectorization.
\]

之前一句“Blackwell TMEM layout optimization”远远不够。

---

# 6. Triton pipeline 中也存在会影响 layout 的性能假设

例如 pipeline 判断一个global load要不要转为 async load时，源码明确说：

> small loads may hurt performance by increasing register pressure.

因此只在访问宽度达到一定条件后才进入async path。fileciteturn240file0L1-L2

`AssignLatencies` 还有明确 heuristic：

> only pipeline A and B operands of the dot op.

fileciteturn298file4L55-L64

SoftwarePipeliner还存在 MMAv5-specific epilogue peeling heuristic。fileciteturn298file3L44-L53

所以以后 Triton 的统计必须至少分：

```text
distributed layout heuristics
shared swizzle heuristics
descriptor/TMA layout heuristics
TMEM layout/vectorization heuristics
pipeline-layout/resource heuristics
warp-specialization heuristics
conversion/rematerialization heuristics
```

这说明它离真正的“完整规则目录”仍有距离。

---

# 7. Gluon：情况比 Triton简单，但仍有自动规则

Gluon的主要自由度是显式的，因此自动heuristic较少。

但也不能只写“programmer decides”。

### 自动 Coalesced layout

compiler存在：

\[
InferCoalescedEncodings
\]

以及：

\[
ResolveAutoEncodings.
\]

Pass定义已经明确：前者基于 axis analysis推 coalesced encoding。fileciteturn312file0L1-L13

---

### NVMMASharedLayout default

`get_default_for()`明确说：

> picks the largest swizzle pattern compatible with the shape, which allows emitting the fewest TMA or MMA messages.

fileciteturn261file0L1-L12

具体根据 contiguous dimension bytes：

\[
\ge128\text{且整除128}
\Rightarrow128B
\]

其次64B、32B，否则无swizzle。fileciteturn288file0L1-L28

同时有硬约束：

\[
swizzle\in\{0,32,64,128\}
\]

且 FP4 padded要求特定组合。fileciteturn287file0L1-L2

Warp specialization的：

\[
worker\_num\_warps
\]

以及：

\[
worker\_num\_regs
\]

则由programmer直接传入，不是compiler heuristic。fileciteturn306file0L1-L23

因此 Gluon 当前可以分为：

\[
\boxed{
\text{explicit-by-default}
+
\text{small number of structural defaults}
}
\]

但下一轮还要继续扫 Blackwell TensorMemory/scale layout helpers，才能真的标完成。

---

# 8. TVM MetaSchedule：主要缺口现在不是“搜索算法”，而是要穷举规则族

当前 CUDA默认 schedule已经相当明确：

普通CUDA：

\[
structure=SSSRRSRS
\]

\[
vectorLoadLens=\{1,2,3,4,8,16\}
\]

shared read reuse、local write reuse、cross-thread reduction thread-extents等。fileciteturn235file0L1-L7

TensorCore又有自己的：

\[
SSSRRSRS
+
WMMA/MMA\ intrin groups
+
shared.dyn reuse
+
software pipeline choices.
\]

fileciteturn234file0L1-L7

但完整性要求不能只记录 DefaultCUDA和DefaultCUDATensorCore，还需要把：

- MultiLevelTiling
- MultiLevelTilingTensorCore
- CrossThreadReduction
- AddRFactor
- AutoInline
- AutoBind
- ParallelizeVectorizeUnroll
- RewriteCooperativeFetch
- RewriteParallelVectorizeUnroll
- RewriteReductionBlock
- RewriteTensorize
- RewriteLayout

哪些会改变 layout/thread binding/cache placement分别统计。

尤其 `RewriteLayout`现在已经确认一个非常具体的 assumption：

> target buffer expected to be read by only one BufferLoad.

然后从该 consumer 的 indices + loops + predicate推一个 `IndexMap`。fileciteturn289file0L1-L10

如果存在 cache-read chain，同一个 IndexMap被沿链传播。

因此 TVM 必须至少区分：

\[
\boxed{\text{schedule-space layout}}
\]

和：

\[
\boxed{\text{anchor-consumer-driven physical buffer layout}}
\]

两套规则。

TVM目前接近完整，但还没达到“所有 CUDA rule families 已逐项入册”。

---

# 9. TIRx：之前明显低估了 priority registry 的规模

此前写成：

> high-priority vector copy → auto → scalar fallback。

这是不够的。

当前源码/官方源内文档至少已经出现：

### priority 20

- forced vector copies
- `ldgsts` asynchronous copy
- SM100 packed reduction
- `permute_layout/warp_xor_swizzle`

fileciteturn301file0L1-L11 fileciteturn301file2L28-L37 fileciteturn301file3L39-L47 fileciteturn301file4L50-L63

### priority 10

- local reduction
- shared reduction
- `mma_m16n8k` GEMM
- DSMEM copy
- tcgen05 async GEMM
- tcgen05 TMEM↔local copy
- TMA auto path

fileciteturn302file0L1-L14 fileciteturn302file2L35-L50 fileciteturn302file4L67-L82

而且 reduction本身还依赖layout pattern。

例如：

\[
laneid\ shard\rightarrow replica
\]

模式会自动走特殊 shuffle path；否则general path可以根据 `thread_reduce` 再加入shuffle。fileciteturn290file2L27-L43

所以 TIRx真正完整的统计单位应该是：

\[
\boxed{
(Op,\ Variant,\ Priority,\ Predicate,\ LayoutPrecondition,\ Scope,\ Target)
}
\]

而不是“一个dispatch策略”。

我们还没有把所有 tile primitive registry都按这个格式扫完，因此 TIRx目前只能标“部分完成”。

---

# 10. Hexcute：现在信息已经非常接近完整

这一轮拿到了 artifact 对应 Hidet源码中的完整设计说明。

`instantiate_auto_annotation.py` 自己明确列出四阶段：

1. logical shape；
2. logical layout；
3. task mapping；
4. memory layout。

fileciteturn308file0L1-L2

Copy的核心约束：

\[
f\circ p^{-1}=g\circ q^{-1}
\]

MMA：

\[
f_{1m}=f_{3m},
\quad
f_{2n}=f_{3n},
\quad
f_{1k}=f_{2k}.
\]

Multiple instructions时：

> could employ DFS to find all valid variants and choose the best one through a cost model; alternatively simple heuristics or beam search.

Shared tensor的多个consumer constraints持续refine memory layout；冲突backtrack；所有constraints满足以后：

> use a heuristic to determine the remaining undetermined strides.

这些都属于明确的 layout-selection策略。fileciteturn308file0L1-L2

Cost model的假设也必须完整记录：

- instruction count × CPI；
- independent/dependent latency table；
- copy/MMA perfect overlap；
- bank conflicts暂未计入；
- `cp_async_wait_group` / `mbarrier`未计入；
- tensor-manipulation/address arithmetic认为negligible；
- arithmetic latency没有按Ampere/Hopper拆分，因为作者认为微基准差异小。

fileciteturn309file0L1-L2

这里已经不只是“cost model简化”一句话，而是一组明确 assumptions。

Hexcute剩下的完整性工作主要是把 `instruction_selection.py` 和 `resolve_bank_conflict` 中的每个 instruction/layout predicate形成表格。前面已有源码证据，但还没逐variant成册。

---

# 11. TileLang：free-mode ranking现在已经比较明确

当前默认：

\[
tl.layout\_cost\_model=\text{"register-count"}.
\]

fileciteturn270file2L32-L43

另一种：

\[
\text{"io-aware"}
\]

根据所有 fragment↔global copy的：

- vector width；
- coalescing；
- bytes moved；

估计global-memory access cost，并以register count作为tie-breaker。fileciteturn270file8L116-L126

维护目录明确把两者称为：

> selection policy behind `tl.layout_cost_model`.

fileciteturn270file3L43-L52

同时 LayoutInference要求thread extent是compile-time constant。fileciteturn237file1L13-L20

PipelinePlanning、software pipeline materialization以及Blackwell 2SM lowering都先于 LayoutInference：

\[
Pipeline
\rightarrow LayoutInference.
\]

fileciteturn238file0L1-L26

这部分已经较完整。

剩余要补的是：strict/common/free三阶段具体candidate ordering、所有reducer/alias/floating fragment fallback的完整枚举。目前我们只覆盖了代表性路径。

---

# 12. vLLM：之前漏掉的最大部分是 block/page geometry policy

vLLM并不只有：

\[
KVCacheLayout
\]

选择。

Backend API还直接提供：

```python
get_preferred_block_size()
```

。fileciteturn273file0L1-L11

不同backend规则不同。

例如 B12X：

> 如果default block size已经受支持，则保留；否则使用 backend preferred page size。

fileciteturn273file1L12-L21

Sparse MLA/SWA直接：

\[
block\ size=256.
\]

fileciteturn273file3L34-L43

ROCm AITER unified：

\[
block\ size=64.
\]

fileciteturn273file4L45-L54

平台初始化又明确分 Phase 1：

> Pick block size from backend，除非用户显式设置 `--block-size`.

fileciteturn273file5L56-L66

所以 vLLM真正的 persistent-layout决策应写成：

\[
\boxed{
PhysicalStrideOrder
+
Page/BlockGeometry
}
\]

而不是单独 `KVCacheLayout`。

而 backend layout requirements又非常丰富，例如：

- CPU → LBHNC only；
- HPC → LBNHC only；
- B12X → LBHNC/BLHNC；
- QSA → BLNHC/BLHNC；
- FlexAttention → LBNHC，因为只有该stride支持 `(B,N)` zero-copy flatten；
- TurboQuant → LBNHC。

fileciteturn272file1L12-L22 fileciteturn272file2L23-L33 fileciteturn272file3L34-L44 fileciteturn272file11L127-L137

因此 vLLM仍需最后做一件事：

\[
\boxed{
\text{backend-by-backend}
\quad
(LayoutPreference,\ BlockSizePreference)
}
\]

的完整inventory。

目前尚未完成。

---

# 13. SGLang：此前最大遗漏是 Model Override 层

generic：

```python
get_default_attn_backend()
```

只是第一层。

它已经包含比如：

- Hopper FA3；
- Blackwell TRTLLM MHA；
- speculative top-k/page-size限制；
- asymmetric K/V；
- FlashInfer/attention-sink；
- MLA/HIP规则。

fileciteturn275file1L20-L40

但源码还有完整的：

```text
python/sglang/srt/arg_groups/model_overrides/
```

目录。

里面至少有 DeepSeek、Qwen、Gemma、GLM、GPT-OSS等大量model-family policy providers。fileciteturn303file0L1-L2

例如 Qwen3.5 的SM100 override并不是简单沿用generic policy。源码注释明确根据 TRTLLM MHA 的 speculative top-k 和 page-size要求重新选择backend。fileciteturn275file0L1-L19

因此真正的policy层是：

\[
GenericPlatformPolicy
\]

\[
+\ ModelFamilyOverrides
\]

\[
+\ ExplicitUserOverride.
\]

过去只统计第一层，所以远不能称“完整”。

这里下一步需要把整个 `model_overrides/` 目录按作用分类：

```text
attention backend
prefill/decode split
page size
KV pool/layout
speculative mode
MoE backend
Mamba/state layout
quantization
```

而不是逐个模型全文罗列。

---

# 14. FlashInfer：本轮也发现了大量此前没有列出的 kernel specialization rule

此前只写：

- NHD/HND；
- FA2/FA3；
- split-KV；
- CTA Q tile。

仍然过粗。

`utils.cuh` 中直接存在：

### MMA Q count

仅接受：

\[
NUM\_MMA\_Q\in\{1,2\}.
\]

### MMA KV count

按：

\[
8\rightarrow4\rightarrow2\rightarrow1
\]

选择不超过 `max_mma_kv` 的最大档。fileciteturn279file0L1-L2

### CTA Q tile

离散：

\[
\{16,32,64,128\}.
\]

### GQA group size

该路径显式支持：

\[
\{1,2,3,4,6,8\}.
\]

### aligned vector size

离散：

\[
\{16,8,4,2,1\}.
\]

### decode SMEM stages

如果compute capability major：

\[
\ge8
\Rightarrow2\ stages
\]

否则：

\[
1\ stage.
\]

fileciteturn280file0L1-L11

这都是 kernel-layout / work-partition相关的结构规则。

---

# 15. FlashInfer 的 CTA tile甚至直接编码了 register-pressure assumption

当前kernel generator注释：

当：

\[
HEAD\_DIM_{VO}\ge512
\]

时，scheduler只会选：

\[
CTA\_TILE_Q\in\{16,32\}
\]

；

若QK dim很大且VO较小：

\[
\{16\}
\]

；

普通情况：

\[
\{16,64,128\}.
\]

原因注释直接提到 output fragment register pressure。fileciteturn278file0L1-L11

所以 FlashInfer真正的规则链已经是：

\[
(Q/K/V\ shape,dtype,GQA,page,arch)
\]

\[
\Downarrow
\]

\[
Backend
+
CTA\ tile
+
VectorWidth
+
MMA count
+
PipelineStages
+
SplitKV.
\]

其复杂度其实比此前“NHD/HND + attention scheduler”这个总结大很多。

---

# 16. 这次重新建立一个真正的“完整性矩阵”

标记：

- **A** = 已有源码级规则目录较完整
- **B** = 核心已覆盖，但存在已知未整理rule families
- **C** = 仍存在较大规则层缺口

| Framework | Representation | Hard rules | Deterministic selection | Heuristics | Cost/ranking | Pipeline relation | Overrides | 当前 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CUTLASS | A | A | B | B | B | A | B | **B** |
| Triton | A | A | B | **C** | B | B | B | **C/B** |
| Gluon | A | A | B | B | B | A | A | **B+** |
| TVM | A | A | B | B | A | B | B | **B+** |
| TIRx | A | A | **C** | C | —/priority | A | A | **C/B** |
| Hexcute | A | A | B | B | A | A | A | **B+** |
| TileLang | A | A | B | B | A | A | A | **B+** |
| vLLM | A | A | **B** | B | — | D | B | **B** |
| SGLang | A | A | **C** | C | — | D | **C** | **C** |
| FlashInfer | A | A | **B/C** | C | mostly rules | B | A | **C/B** |

也就是说，目前**没有一个框架我愿意标成“已经穷举所有 production layout-selection strategies”**。

Hexcute、TileLang、Gluon、TVM最接近。

SGLang、TIRx、FlashInfer、Triton需要继续扫得最多。

---

# 17. 还必须明确“每一种规则”到底统计到什么边界

这里需要避免无限扩张。

例如 CUTLASS repository里成千上万example specialization；SGLang有大量模型override；TIRx有大量primitive variants。

如果所谓“每一种规则”包含：

> 每一个example里用户手工写的 `if M == 64`，

统计会失去意义。

我建议严谨地把完整性边界定义成：

\[
\boxed{
\textbf{production decision paths that can affect automatic/default layout selection}
}
\]

必须包括：

- compiler passes；
- builders；
- runtime selectors；
- backend capability methods；
- layout inference；
- cost model；
- primitive registries；
- model/runtime override providers；
- architecture specializations；
- default helper functions。

而仅用于单个example展示的手工config：

\[
\boxed{\text{记录为 expert configuration evidence，不计入 framework heuristic catalog}}
\]

否则“框架heuristic”和“example author选择”会被混在一起。

---

# 18. 当前已经建立的规则目录应该扩成这种格式

后续每一行都是一个具体rule，而不是一个框架一个格子：

| Framework | Rule ID | Phase | Target layout | Inputs | Type | Selection | Fallback |
|---|---|---|---|---|---|---|---|
| CUTLASS | C-S90-SMEM-SS | mainloop | SMEM atom | major,dtype,tile | heuristic/rule | largest compatible SW | INTER/fail |
| CUTLASS | C-S90-GMEM-SIMT | copy | thread mapping | stride,tile,alignment | heuristic | maximize major-axis threads | invalid |
| Triton | T-COALESCE | TTGIR | distributed encoding | AxisInfo/load | heuristic | coalesced layout | conversion |
| Triton | T-DESC-NVMMA | descriptor | shared encoding | shape,order,dtype | rule | preferred then scan 0/32/64/128 | default |
| Triton | T-TMEM-VEC | lowering | reg↔TMEM | layout,maxnreg | heuristic | ≤½ regs; vectorize | smaller atom |
| Gluon | G-NVMMA-DEFAULT | program helper | shared layout | shape,dtype | rule | largest compatible swizzle | 0 |
| TVM | V-CUDA-MLT | search | schedule/thread/cache | workload | prior | SSSRRSRS family | other rule |
| TIRx | X-COPY-LDGSTS | dispatch | copy mapping | layout,scope,target | priority | priority 20 | lower variant |
| Hexcute | H-COPY-TV | inference | TV mapping | src/dst,instruction | constraint | unify | backtrack |
| TileLang | TL-FREE-IO | inference | fragment layout | GMEM copies | cost model | min estimated IO | reg tie-break |
| vLLM | V-KV-LAYOUT | init | persistent KV | backends/specs | preference | intersection+vote | fail/override |
| vLLM | V-BLOCK-SIZE | init | page geometry | backend | heuristic | backend preferred | default/user |
| SGLang | S-ATTN-DEFAULT | init | backend→layout | arch/model/spec | rules | backend tree | Triton/etc |
| FlashInfer | F-CTA-Q | plan | CTA Q tile | qlen,D,dtype | heuristic | {16,32,64,128} | unsupported |

达到这种粒度后，我们才能真正检查：

> 有没有整个rule family漏掉。

---

## 19. 所以下一轮应该继续做什么

现在最应该继续的不是再写总表，而是做四个剩余的“高风险规则目录”：

1. **Triton**：把 Nvidia layout相关pass + GenericSwizzling + descriptor + TMEM + pipeline + warp-specialization中所有明确 heuristic逐条列出来。
2. **TIRx**：把 CUDA `tile_primitive` registry中的所有 variant、priority、predicate、作用layout形成表。
3. **SGLang**：把 `model_overrides/` 全目录按“修改了哪个 layout/backend/page/state decision”分类统计。
4. **FlashInfer**：把 decode/prefill/MLA/FP4/MoE scheduler和kernel specialization的离散规则完整列出。

同时对 CUTLASS补 mainloop/epilogue/scale-factor 三套 Builder规则；对 Hexcute/TileLang/Gluon/TVM做漏项复核。

因此，本轮最重要的结论不是“整理完了”，而是：

\[
\boxed{
\text{我们已经从“框架级统计”进入“rule-catalog级统计”，并确认前一版仍有遗漏。}
}
\]

只有这四个高风险目录和剩余复核都完成后，我才会再次说“所有框架达到标准”。在此之前继续不提炼任何科研问题。

## 用户

请继续审查 检查哪里没有整理并补充

## 模型回复

这一轮继续做“漏项审查”，不提炼科研问题。重点放在上一轮仍标为 C/B 的 Triton、TIRx、SGLang、FlashInfer，同时对 CUTLASS、TVM、Gluon、Hexcute、TileLang 做反向检查：专门找“此前没有进入规则目录、但会改变 layout / mapping / page geometry / swizzle / instruction / pipeline 选择”的入口。

## 模型工具调用参数

{"paths":["GitHub"],"query":"fetch"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考摘要

查询TVM CUDA Tile目录并搜索拷贝调度实现

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 github.com

## 模型思考摘要

已搜索 1 个网站

## 模型回复

我会把“漏项”分成两类：一类是**新的决策入口**（例如 Triton 的 TMEM layout pass、vLLM 的 block-size policy）；另一类是**同一入口内部此前只记录了代表性分支，但实际还有更多架构/dtype/操作特化**。后者也会补，但会避免把单个 example 的手工参数误算成框架默认启发式。

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 github.com

## 模型思考摘要

继续核查缺口
梳理审计框架
补充关键缺口
查验Triton与Gluon
补充关键缺口并梳理审计框架

## 模型思考摘要

扩展剩余审计
完成剩余审计扩展

## 模型思考摘要

已搜索 1 个网站

## 模型回复

本轮补查又确认了一个重要点：**vLLM 的“layout 决策”之前仍少了一层 `KVCacheSpec` 物理重写**。backend 不只是声明 `LBHNC/LBNHC` 和 block size，还能通过 `customize_spec()` 改变 page 内部内容，例如 NVFP4 把 K/V 拆成独立 head slots 并附带 scale storage；ROCm shuffle 路径也会改变 slot 组织。这个必须单独作为 `persistent content layout`，不能混进 stride-order 表。 

## 模型思考：工具活动

正在搜索网页

## 模型思考：工具活动

正在搜索 docs.flashinfer.ai

## 模型思考摘要

已搜索 1 个网站

## 模型思考用时

思考了 4m 46s

## 模型回复

继续审查以后，结论仍然是：**现在还不能说“所有框架的 layout 规则已经完整整理”。**不过这一轮已经把几个此前比较大的空洞定位得更清楚了，而且补出的内容不是边角规则，而是会直接改变 persistent layout、page geometry、thread/value mapping、SMEM swizzle、TMEM mapping、copy instruction 或 backend-native representation 的生产路径。

我建议从现在开始把“完整性”判断从“有没有介绍这个模块”提升为“有没有把这个模块的每一个自动/default decision path 编目”。也就是说，某个框架有 `layout inference` 还不够，还要知道 inference 的 candidate 从哪来、按什么顺序、什么情况下 reject、什么情况下 widen/replicate、怎么 rank、最后如何 materialize。

---

## 一、先更新本轮完整性审计结果

目前我会这样标记：

| Framework | 上轮 | 本轮 | 这轮新增发现 | 仍未整理完整的主要部分 |
|---|---:|---:|---|---|
| CUTLASS/CuTe | B | **B+** | mainloop/epilogue/scale/sparse 其实是不同 AUTO policy family | SM100 sparse、mixed-input、block-scaled、epilogue 各 Builder 的完整规则矩阵 |
| Triton | C/B | **B** | GenericSwizzling、descriptor layout、TMEM layout、TMEM interleave 是独立决策器 | `OptimizeTMemLayouts` 内部规则、Coalesce/ThreadLocality 的完整阈值与分支 |
| Gluon | B+ | **B+** | AutoEncoding 与 CoalescedEncoding 是 compiler pass，不只是 explicit layout | 两个 pass 内 exact inference rule；Blackwell TMEM/scale helpers |
| TVM MetaSchedule | B+ | **B+** | RewriteLayout 与 schedule-space layout 是两套独立机制 | CUDA ScheduleRule/Postproc 全部 layout-affecting rule 的逐项目录 |
| TIRx | C/B | **B+** | 当前文档已经能看到完整 primitive dispatch surface，远比此前统计丰富 | copy_async/gemm/reduction 等所有 layout-sensitive variant 的完整 predicate table |
| Hexcute | B+ | **B+** | 约束、backtracking、cost assumptions 已较完整 | 每个 copy/MMA instruction predicate；unknown stride completion heuristic |
| TileLang | B+ | **B** | TMA auto-promotion、alias layout propagation、grid swizzle 是此前遗漏的三个机制 | strict/common/free candidate order、reducer fallback、TMA selection 完整条件 |
| vLLM | B | **B+** | `customize_spec()` + scheduler/hash block granularity 是新的 persistent-layout 层 | 所有 backend 的 layout/spec/block-size 四维矩阵 |
| SGLang | C | **B/C** | model-family override 和 backend-internal policy 远比 generic selector 复杂 | 整个 `model_overrides/` 分类穷举，以及 backend-native representation contracts |
| FlashInfer | C/B | **B/C** | 除 FA2 scheduler 外还有 task-scheduled/Blackwell 等独立 scheduler family | FA2 / decode / MLA / TRTLLM-gen / PrimTS 分别完整建表 |

所以目前依然：

\[
\boxed{\text{没有一个框架应该正式标记为“规则穷举完成”。}}
\]

但其中 Hexcute、TIRx、vLLM、CUTLASS 已经越来越接近可以做最后反向漏项检查的阶段。

---

# 二、CUTLASS：之前把 AUTO 机制合得太粗

这一轮一个重要修正是：以后不能只有一个：

\[
\text{CUTLASS Auto}
\]

这一项。

至少要拆成：

\[
\boxed{
\begin{aligned}
&AutoMainloop\\
&AutoCopy\\
&AutoSmemLayout\\
&AutoKernelSchedule\\
&AutoStageCount\\
&AutoEpilogue\\
&AutoScale/MetadataLayout
\end{aligned}}
\]

因为它们的输入、假设和选择规则不一样。

例如 SM90 SIMT GMEM copy builder明确不是随意构造 thread layout，而是“尽可能增加 global-memory major dimension 上的线程数以提高 coalescing”，同时要求 thread partition 可以整除 tile。fileciteturn296file0L1-L2

这应该正式记成：

| Rule | 输入 | 输出 | 类型 |
|---|---|---|---|
| `make_simt_gmem_tiled_copy` | tile shape、stride major、alignment、thread count | copy thread/value layout | structural heuristic |
| 主要假设 | major-axis thread parallelism 是 coalescing 的主要目标 |
| legality | thread layout 必须整除 CTA tile |

而 SM90 SMEM layout 至少又有两种策略。

`ss_smem_selector()` 基本按照：

\[
SW128\rightarrow SW64\rightarrow SW32\rightarrow INTER
\]

选择能够合法 tile 的最大 swizzle。

但是 `rs_smem_selector()` 并不总采用这个顺序；某些 whole-word/transposition 场景源码明确优先 SW32，理由就是避免 bank conflict。也就是说：

\[
\boxed{
SS\ selector\neq RS\ selector.
}
\]

fileciteturn296file0L1-L2

Blackwell 又是另一套 policy。当前官方文档明确说 `KernelScheduleAuto` 不仅选 1SM/2SM，还会根据 MMA tile size、dtype 和 layout 选择 instruction family，包括 block-scaled 和 sparse 情况；而官方甚至明确表示，对这些 Blackwell kernel，“preferred method”仍然是直接指定 dispatch policy，而不是完全依赖 AUTO。citeturn338619search8

这说明 CUTLASS 的统计以后必须有：

\[
\boxed{
\text{legality-driven AUTO}
\quad\text{vs}\quad
\text{performance-oriented expert policy}
}
\]

两列，而不能把 Auto 当成 autotuner。

另一个此前遗漏的是 epilogue。SM100 epilogue有自己的 SMEM layout selector和 sparse epilogue tile rules，不是 mainloop layout 的简单延伸。citeturn338619search6

因此 CUTLASS 目前最大的剩余工作已经非常具体：不是继续解释 CuTe，而是把 SM90/SM100 的 dense、sparse、mixed-input、block-scaled 四组 `CollectiveBuilder` 与 `EpilogueBuilder` 分支整理成 rule matrix。

---

# 三、Triton：现在确认它实际上有至少七类 layout 决策器

之前把 Triton概括成：

\[
Coalesce\rightarrow MMA\rightarrow Pipeline\rightarrow ConversionRepair
\]

还是太粗。

当前源码应该至少拆成：

| Triton 决策器 | 主要 layout 对象 |
|---|---|
| Coalesce | GMEM distributed layout |
| AccelerateMatmul / OptimizeDotOperands | MMA operand/result layout |
| GenericSwizzling | generic shared-memory layout |
| OptimizeDescriptorEncoding | TMA descriptor shared layout |
| OptimizeTMemLayouts | Blackwell TMEM layout |
| RemoveLayoutConversions / ReduceDataDuplication | representation reconciliation |
| Pipeline / WarpSpecialization / InterleaveTMem | layout-resource-execution coupling |

### GenericSwizzling

它实际上已经有明确的局部优化目标：

\[
\min
\left(
BankConflict_{read}
+
BankConflict_{write}
\right),
\]

平局再比较搬运 rounds。

源码直接称之为 heuristic。fileciteturn298file0L1-L16

因此不能再说 Triton 的 shared layout只是“compiler automatic”。

准确说应该是：

\[
\boxed{
\text{candidate construction}
+
\text{explicit bank-conflict heuristic}
}
\]

。

### TMA descriptor layout

另一个完全独立的规则是 `OptimizeDescriptorEncoding`。

它先尝试默认 shape/order-derived NVMMASharedEncoding；若不能和已有 SharedLinearLayout 保持等价，再扫描：

\[
fp4Padded\in\{0,1\},
\]

\[
swizzle\in\{0,32,64,128\}.
\]

找到第一个等价candidate就接受。fileciteturn285file0L1-L13

这里选择目标首先是：

\[
\boxed{\text{semantic equivalence + TMA legality}}
\]

而不是“所有swizzle测性能”。

Triton官方pass说明也明确指出 descriptor encoding 决定 TMA descriptor 的 swizzling mode 和 message size。citeturn338619search5

### TMEM

Blackwell 又出现了一套此前完全没有纳入的 layout system。

官方 pass列表现在明确有：

\[
OptimizeTMemLayouts
\]

用于选择有利于 subtiling、reduction 等的 TMEM layout；以及：

\[
InterleaveTMem
\]

通过 sink load / hoist store / interleave 来降低 register pressure。citeturn338619search5

其低层 `TensorMemoryUtils` 还有非常具体的 heuristic：

\[
RegistersUsedForTMemMessage
\le
\frac{maxnreg}{2},
\]

因为源码认为超过一半容易 spill；如果 `maxnreg=256` 且需要多个message，还进一步降低vectorization，因为 ptxas scheduler可能表现不好。fileciteturn286file0L1-L6

所以 Triton现在必须分别记录：

\[
L_{GMEM},
L_{SMEM},
L_{TMA-desc},
L_{register},
L_{TMEM}.
\]

之前把这些统称 TTGIR layout显然太粗。

### 目前 Triton 最大缺口

现在缺的已经不是 pass 名字，而是：

\[
\boxed{
\text{每个 pass 内候选是如何产生和排序的}
}
\]

尤其 `OptimizeTMemLayouts`、`OptimizeThreadLocality`、Coalesce、AutomaticWarpSpecialization。这部分还不能标完成。

---

# 四、Gluon：需要修正“explicit layout system”这个过度简化

Gluon仍然比 classic Triton更 explicit，但：

\[
\boxed{\text{explicit}\neq\text{完全没有 compiler layout inference}}
\]

。

现在已经确认 compiler 中有两个专门pass：

\[
GluonResolveAutoEncodings
\]

与：

\[
GluonInferCoalescedEncodings.
\]

后者明确根据 axis analysis推 coalesced encoding。fileciteturn312file0L1-L13

同时 `NVMMASharedLayout.get_default_for()` 是真正的 default heuristic：

\[
128B
\rightarrow64B
\rightarrow32B
\rightarrow0
\]

按 contiguous byte extent合法性取最大的swizzle，目的明确写成减少 TMA/MMA messages。fileciteturn288file0L1-L28

所以 Gluon 当前应该改写为：

\[
\boxed{
\text{explicit distributed layout}
+
\text{small structural auto/default layer}
}
\]

而不是“programmer决定layout，compiler只lower”。

目前仍然没完全整理的是 `ResolveAutoEncodings` 和 `InferCoalescedEncodings` 的具体内部规则。所以 Gluon维持 B+。

---

# 五、TVM：还有一个需要进一步拆开的概念

以前容易把：

\[
MetaSchedule\ layout\ search
\]

和：

\[
RewriteLayout
\]

放在一起。

实际上不是一回事。

MetaSchedule做的是：

\[
ScheduleRule
\rightarrow CandidateSchedule
\rightarrow CostModel
\rightarrow HardwareMeasurement.
\]

而 `RewriteLayout` 是一个 consumer-driven physical representation rewrite。

更重要的是，其当前实现有非常强的结构前提：用于推 IndexMap 的 buffer预期由一个 BufferLoad消费；随后根据 consumer access、loop context和predicate推导IndexMap。如果中间存在 cache-read chain，则同一个transform沿链向上游传播。fileciteturn289file0L1-L10

因此 TVM以后至少要分：

\[
\boxed{
L_\text{schedule}
\neq
L_\text{physical-buffer-rewrite}.
}
\]

TVM剩余的工作也很明确：将 CUDA default rule set 中所有可能改变：

\[
tile,\ thread-binding,\ vector-width,\ cache-level,\ tensorization,\ layout
\]

的 `ScheduleRule/Postproc`逐项建表。

也就是说不能只记 `MultiLevelTiling` 和 `RewriteLayout`。

---

# 六、TIRx：本轮补齐程度提升最大

当前官方文档其实已经给出了一个非常好的穷举入口：TIRx现在文档化了整个 tile primitive dispatch surface。

它的 primitive call并不是只有 op，而是：

\[
(op,args,workspace,config,dispatch,scope).
\]

自动dispatch又隐式使用：

\[
ExecutionScope
+
OperandLayouts
+
Target.
\]

citeturn338619search0

这意味着我们之前的：

> TIRx = priority + predicate

虽然没错，但粒度远远不够。

### synchronous copy

CUDA当前就有八个variant：

| Variant | Priority | layout/work-partition strategy |
|---|---:|---|
| `vec_16b` | 20 | 固定 16-bit transfer width |
| `vec_32b` | 20 | 固定 width |
| `vec_64b` | 20 | 固定 width |
| `vec_128b` | 20 | 固定 width |
| `vec_256b` | 20 | 固定 width |
| `vec_auto` GMEM↔SMEM | 10 | 自动构造 `[outer,threads,vec]` |
| `vec_auto` REG path | 10 | 从 register layout thread axes导出partition |
| `ldstmatrix` | 10 | m8n8 fragment mapping |
| fallback | 0 | scalar/single-thread |

准确说固定宽度五种 + `vec_auto` + `ldstmatrix` + fallback，共八个dispatch variants；`vec_auto`内部又有两个路径。citeturn909065search4turn338619search7

这已经比以前一句“vector copy优先”精确很多。

### `copy_async → ldgsts`

它要求：

\[
global\rightarrow shared,
\]

并把vector候选限制为：

\[
\{128,64,32\}\ bits
\]

也就是 cp.async合法的：

\[
16,8,4\ bytes.
\]

然后在 dtype、alignment、layout、thread count约束下选择最大的合法vector width。priority为20。citeturn338619search1

所以它不是：

> cp.async is available → use cp.async

而是一个真正的：

\[
\boxed{
Layout+Alignment+ThreadCount
\rightarrow
VectorWidth+Partition
}
\]

规则。

### Blackwell TMEM copy

`tcgen05.ld/st` 又按照 register layout匹配：

\[
.16x64b,\quad
.16x128b,\quad
.16x256b,
\]

不匹配时只有满足特殊 Layout-D 条件才fallback到：

\[
.32x32b.
\]

而 Layout B、D、F 又有不同physical lane interpretation。citeturn338619search3

这说明 TIRx真正完整的统计行应该是：

\[
(Op,Variant,Priority,Predicate,LayoutContract,Instruction)
\]

。

TIRx现在已经从 C/B提高到 B+，但还需要把 `copy_async` 的 TMA、DSMEM、SMEM↔TMEM，以及 `gemm/gemm_async/reduction/permute_layout`全部用这个格式走一遍。官方总表已经确认这些是独立variants。citeturn338619search0turn338619search4

---

# 七、Hexcute：现在主要缺的不是概念，而是“展开成规则表”

Hexcute 的大框架目前已经比较完整：

\[
LogicalShape
\rightarrow
LogicalLayout
\rightarrow
TVMapping
\rightarrow
MemoryLayout.
\]

Copy：

\[
f\circ p^{-1}=g\circ q^{-1}.
\]

MMA：

\[
f_{1m}=f_{3m},
\quad
f_{2n}=f_{3n},
\quad
f_{1k}=f_{2k}.
\]

多个consumer共同约束同一个SMEM representation，冲突backtrack；多个instruction variant可以DFS枚举后通过cost model排序。fileciteturn308file0L1-L2

Cost model的假设也已经基本被完整抄出来：

\[
Cost=\#Instruction\times CPI
\]

以及 perfect overlap、忽略bank-conflict penalty、忽略部分barrier/wait、忽略address arithmetic等。fileciteturn309file0L1-L2

剩余两项必须继续展开。

第一是：

\[
InstructionSelection
\]

必须针对：

\[
ldg/stg,\ lds/sts,\ ldmatrix,\ cp.async,\ TMA,\ mma,\ wgmma
\]

分别写出 scope/alignment/layout/task-mapping predicate。

第二是约束解完以后：

> remaining undetermined strides

到底按什么 heuristic补齐。

因此 Hexcute目前缺的是“表格化穷举”，不是体系理解。

还有一点必须保持：我们比较的是 Hexcute artifact pin住的 Hidet fork，不能把2026年当前 upstream Hidet新加的 TMEM/TMA功能自动算成 Hexcute已有能力。

---

# 八、TileLang：这一轮又找到三个此前统计中没有的 layout 机制

这部分值得修正。

## 8.1 `use_swizzle` 和 shared-memory swizzle根本不是一个东西

TileLang现在同时存在：

\[
T.use\_swizzle(...)
\]

以及：

\[
make\_swizzled\_layout(...).
\]

前者是 threadblock scheduling/grid swizzle，用于改变 CTA遍历顺序、改善L2 locality；后者才是 buffer-level shared-memory layout。官方API直接把 `use_swizzle`描述为 threadblock swizzle。citeturn909065search8

因此过去如果把“TileLang swizzle”统一放在 SMEM layout一栏，是错误的。

应该拆成：

\[
L_\text{grid/tile-scheduler}
\]

与：

\[
L_\text{SMEM}.
\]

---

## 8.2 warp-specialized pipeline 会自动把普通 Copy提升到 TMA

当前源码问题追踪揭示了一条实际production path：

\[
T.copy
\]

先被：

\[
ClassifyWarpSpecializedProducerCopy
\]

识别，然后调用：

\[
SelectTmaInst
\]

；符合粗粒度条件后：

\[
RewriteCopyToTmaCopy.
\]

之后 LowerBulk再做更严格的 layout/swizzle legality check。citeturn799964search3

这意味着 TileLang并不是：

> 用户写 `T.tma_copy` 才会使用 TMA。

还有：

\[
\boxed{
\text{ordinary copy}
\rightarrow
\text{compiler auto TMA promotion}
}
\]

这一层。

这一层此前完全没进入我们的规则目录。

这里当前暴露的 fallback bug暂时只作为“机制证据”，不把bug本身提炼成任何科研问题。

---

## 8.3 alias之间还会传播 layout

当前 `T.view`相关源码路径表明，如果多个buffer共享同一底层storage，LayoutInference会把一个buffer已经推导出的layout通过 `Reshape`传播给其alias sibling；cross-dtype时还使用storage bit-width ratio做reshape。citeturn799964search0

因此 TileLang还有一个之前未记录的规则：

\[
\boxed{
L(A)
\rightarrow
Reshape
\rightarrow
L(alias(A))
}
\]

。

这不是candidate ranking，而是：

\[
\boxed{\text{layout propagation assumption}}
\]

。

同样，pipeline multi-versioning也会改变buffer physical shape，然后 LayoutInference需要和后续MMA layout协调；当前已有真实源码交互案例。citeturn799964search2

因此 TileLang后续必须增加两个统计维度：

\[
\text{AliasLayoutPropagation}
\]

和：

\[
\text{PipelineVersionLayout}.
\]

这也是为什么这一轮我反而把 TileLang从 B+降成B：不是理解变差，而是发现其decision surface比之前认为的大。

---

# 九、vLLM：应该从“KV stride layout”升级成“persistent representation system”

这是本轮非常重要的补充。

此前我们主要统计：

\[
LBHNC/LBNHC/\ldots
\]

以及：

\[
supported\_kv\_cache\_layouts().
\]

现在确认这样仍然不够。

vLLM当前至少有四级persistent decisions：

\[
\boxed{
\begin{aligned}
&1.\ PhysicalStrideOrder\\
&2.\ PageContentRepresentation\\
&3.\ Page/BlockGeometry\\
&4.\ Scheduler/HashGranularity
\end{aligned}}
\]

---

## 9.1 `customize_spec()` 可以修改 page内部物理内容

以当前 FlashInfer NVFP4 backend为例，它不是只说自己支持哪个stride layout。

它直接把 K/V变成独立 per-head slots，并给每个FP4 block配置FP8 scale storage：

\[
num\_head\_slots
=
2\times num\_kv\_heads.
\]

同时重新定义：

\[
state\_content\_bytes.
\]

citeturn338619search2turn799964search1

ROCm路径同样存在 `customize_spec`；shuffle representation会为了kernel addressing重新组织K/V为独立head slots，并加入block-size约束。citeturn799964search4turn799964search7

所以以后 vLLM不能只写：

\[
L=LBHNC
\]

而需要：

\[
\boxed{
L=
(
stride\ order,
slot\ structure,
content\ packing,
page\ geometry
)
}
\]

。

---

## 9.2 block size之外还有 scheduler/hash block granularity

最新 `resolve_kv_cache_block_sizes()` 明确区分：

\[
scheduler\_block\_size
\]

和：

\[
hash\_block\_size.
\]

多个KV groups时：

\[
scheduler\_block
=
LCM(effective\ group\ block\ sizes),
\]

而 prefix/hash granularity在条件允许时使用：

\[
GCD(group\ block\ sizes)
\]

或用户override。

Attention under DCP还会把effective block span乘DCP size，而Mamba group不这样处理。citeturn909065search2

这属于 persistent representation 的**granularity policy**，此前完全漏了。

---

## 9.3 backend selector实际上也是 layout candidate的上游

vLLM当前自动backend选择是：

\[
PriorityOrder
\rightarrow
ValidateConfiguration
\rightarrow
FirstCompatible.
\]

官方文档明确如此，并给出了CUDA不同architecture的priority。Blackwell standard attention当前优先FlashInfer，然后FlashAttention、Triton等；Ampere/Hopper顺序又不同。citeturn248398search0

而这些 backend拥有不同：

\[
LayoutSupport,
BlockSize,
CustomizeSpec,
KernelGeometry.
\]

所以：

\[
BackendChoice
\rightarrow
PersistentRepresentationFeasibleSet.
\]

这是此前主表缺少的一条依赖边。

例如 Blackwell FA4 head-size 256的专门kernel要求KV block size 128；若用户固定了不兼容block size，backend本身可能直接变得不可用。citeturn248398search8

因此 vLLM现在接近完整，但还需要真正做一张backend-by-backend矩阵：

\[
\boxed{
Backend
\times
SupportedLayout
\times
CustomizeSpec
\times
BlockSize
\times
CompatibilityPredicate
}
\]

这张没做完之前不标完成。

---

# 十、SGLang：目前最大的统计缺口仍然是 model-specific policy layer

SGLang现在可以确定不是只有：

\[
get\_default\_attn\_backend().
\]

至少存在三层：

\[
\boxed{
GenericPolicy
\rightarrow
ModelFamilyOverride
\rightarrow
BackendInternalPolicy.
}
\]

例如当前 DeepSeek-family override就已经包含实际persistent/layout相关决定。

DeepSeek DSA在GPU路径可把page size改成64；某些ROCm legacy path如果paged-MQA preshuffle不可用会退成page size 1；XPU又用128。SM100上的普通DeepSeek V3路径在没有显式attention backend override时可以改选 `trtllm_mla`。citeturn822457view0

这不是简单“model compatibility”，而已经改了：

\[
Backend
+
PageGeometry.
\]

---

## backend内部还有第二套策略

SGLang的 FlashInfer backend 本身又做：

\[
should\_use\_tensor\_core(...)
\]

决定decode路径。

对一些Qwen/MiMo/VL architecture会增大FlashInfer workspace；deterministic inference又强制 tensor-core decode、固定prefill/decode split tile defaults并关闭 CUDA-graph KV split，同时显著扩大workspace。citeturn909065search0

所以以后不能简单记录：

> SGLang selects FlashInfer.

实际上还存在：

\[
SGLang
\rightarrow
FlashInfer\ configuration.
\]

另一个非常好的backend-native representation例子是 HPC-Ops：它明确直接消费 NHD：

\[
(numPages,pageSize,numKVHeads,headDim)
\]

并硬要求：

\[
pageSize=64,\quad headDim=128,
\]

以及特定GQA ratios。citeturn248398search2

这说明 SGLang最终也必须建立：

\[
Backend
\times
PersistentFormat
\times
PageSize
\times
KernelConstraints.
\]

SGLang目前还不能升到 B+，因为 `model_overrides/`目录还没按decision type完整分类。

下一轮应该把它们按：

`attention/backend`、`page size/KV representation`、`speculative`、`quant/MoE`、`state/Mamba` 五类扫描，而不是机械地一个模型一个模型讲。

---

# 十一、FlashInfer：此前最大的遗漏是把它当成一个 scheduler

实际上至少要拆：

\[
\boxed{
FA2/legacy\ scheduler
\neq
Decode\ scheduler
\neq
MLA\ scheduler
\neq
TRTLLM-gen
\neq
Blackwell\ task-scheduled\ attention.
}
\]

当前FlashInfer官方文档甚至已经单独暴露：

> Experimental Task-Scheduled Attention

在 Blackwell上，FMHA context、decode、block-sparse FMHA、MLA decode的：

\[
scheduling,\ tile\ selection,\ split-KV
\]

都是自动实现细节，并且没有public tuning knobs。citeturn362530search0

这是以前根本没单独统计的一个 scheduler family。

---

## Legacy FA2也不是简单的 CTA size rule

当前源中的离散decision至少包括：

\[
NUM\_MMA_Q\in\{1,2\},
\]

\[
NUM\_MMA_{KV}\in\{8,4,2,1\},
\]

\[
CTA\_TILE_Q\in\{16,32,64,128\},
\]

以及离散GQA group specialization、vector width和SMEM stage。fileciteturn279file0L1-L2

decode pipeline stage甚至直接由compute capability决定：

\[
CC_{major}\ge8
\Rightarrow
2\ stages,
\]

否则1 stage。fileciteturn280file0L1-L11

对于很大的 VO/QK head dimension，CTA-Q候选又因为register pressure主动缩小。fileciteturn278file0L1-L11

因此应该把 FlashInfer的decision vector写成：

\[
D_F=
(
KVLayout,
CTAQ,
MMAQ,
MMAKV,
VectorWidth,
Stages,
SplitKV,
BackendFamily
).
\]

而不是只记：

\[
NHD/HND+SplitKV.
\]

FlashInfer官方目前仍然接受NHD/HND paged KV；MLA又支持把CKV和KPE拼在同一个paged representation里，这也是另一个representation family。citeturn362530search2

所以这里还有很大整理空间。

---

# 十二、现在还缺哪些“横向维度”？

这一轮发现，以前虽然已经有很多列，但仍少了几列。新的统一统计表至少应该扩充到：

| Decision dimension | CUTLASS | Triton | Gluon | TVM | TIRx | Hexcute | TileLang | vLLM | SGLang | FlashInfer |
|---|---|---|---|---|---|---|---|---|---|---|
| Persistent stride | — | — | — | weight rewrite | — | — | — | ✓ | ✓ | consumes |
| Persistent content packing | — | — | — | partial | — | — | — | **✓** | ✓ | ✓ |
| Page geometry | — | — | — | — | — | — | — | **✓** | **✓** | input |
| Grid/CTA scheduling layout | kernel schedule | compiler | explicit | schedule | dispatch | fixed/program | **✓** | runtime | runtime | scheduler |
| GMEM thread/value mapping | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | D | D | kernel |
| SMEM physical layout | ✓ | ✓ | ✓ | ✓/I | ✓ | ✓ | ✓ | D | D | kernel |
| Descriptor/TMA layout | ✓ | **✓** | ✓ | tensorize | ✓ | ✓ | **✓** | D | D | kernel |
| Register layout | ✓ | ✓ | ✓ | schedule | derived | ✓ | ✓ | D | D | kernel |
| TMEM layout | ✓ SM100 | **✓** | ✓ | target-specific | **✓** | artifact有限 | ✓ | D | D | Blackwell kernel |
| Metadata/scale layout | ✓ | ✓ | ✓ | partial | ✓ | ✓ | partial | **✓** | backend | ✓ |
| Alias propagation | C++ composition | SSA/conversion | explicit | IndexMap | explicit | constraint | **✓** | shared specs | pools | — |
| Multi-version/pipeline layout | ✓ | ✓ | explicit | schedule | explicit | explicit pipeline | **✓** | — | — | scheduler |
| Layout conversion | explicit | ✓ | explicit | ✓ | explicit | backtrack/program | inference/copy | persistent conversion | backend | limited |
| Backend→layout coupling | — | — | — | backend target | primitive | instruction | backend | **strong** | **strong** | kernel family |
| Runtime workload geometry | very low | JIT | JIT | tune task | dispatch context | compile | compile | medium | high | **high** |

新增最重要的是四列：

\[
\boxed{
PersistentContentPacking,
PageGeometry,
Descriptor/TMALayout,
Alias/PipelineLayoutPropagation.
}
\]

这些以前没有被充分横向比较。

---

# 十三、还需要把“rule类型”统一，否则后面会继续混淆

我建议下一版完整目录给每条规则标一个类别：

| 类型 | 定义 | 例子 |
|---|---|---|
| C0 Hard constraint | 不满足则非法 | TMA alignment、MMA fragment shape |
| R1 Structural rule | 确定性结构选择 | CUTLASS largest-compatible swizzle |
| H1 Priority heuristic | 多个合法候选按固定顺序挑 | TIRx priority、vLLM backend priority |
| H2 Local analytic heuristic | 用局部代理指标排序 | Triton bank-conflict swizzle |
| M1 Analytic cost model | 显式cost function比较完整candidate | Hexcute、TileLang io-aware |
| M2 Learned/empirical | measurement或learned ranking | TVM MetaSchedule、Triton autotune |
| P1 Runtime policy | 请求/模型状态参与选择 | SGLang、vLLM、FlashInfer |
| F1 Repair/fallback | 前面选择失败后重新协调 | Triton remat、Hexcute backtrack、TIRx fallback |

这个分类很重要，因为：

\[
\text{constraint}
\neq
\text{heuristic}
\neq
\text{cost model}
\neq
\text{runtime policy}.
\]

如果把它们都称为“layout optimization heuristic”，后面还是会失真。

---

# 十四、本轮最重要的修正结论

现在我认为此前统计里最大的遗漏已经从“某个具体规则没看到”转变为下面这些**整个 decision family** 没有单独拆出来：

| 此前漏掉的 family | 主要框架 |
|---|---|
| Persistent content packing，而不仅stride order | vLLM、SGLang、FlashInfer |
| Scheduler/hash/page granularity | vLLM、SGLang |
| Model-family override | SGLang |
| Descriptor/TMA-specific shared layout | Triton、CUTLASS、Gluon、TIRx、TileLang |
| TMEM layout | Triton、CUTLASS、TIRx、TileLang/Gluon |
| Generic bank-conflict-based swizzle search | Triton |
| Alias layout propagation | TileLang |
| Pipeline-versioned storage layout | TileLang，也应反查Triton/TVM |
| Grid/threadblock swizzle | TileLang/CUTLASS scheduler层 |
| Backend-native representation contract | vLLM、SGLang |
| Multiple independent attention scheduler families | FlashInfer |
| Mainloop vs epilogue layout policy | CUTLASS |

这说明之前按：

\[
GMEM/SMEM/Register/KV
\]

四五层统计仍然不够；现在已经开始接近真正的完整 decision graph。

---

## 十五、下一轮审查应该继续在哪里

下一轮我仍然不建议总结科研问题。现在最值得继续完成的是四张“穷举表”：

\[
\boxed{\text{Triton pass-rule table}}
\]

把所有 NVIDIA layout-affecting pass 的 candidate、criterion、fallback精确列出来；

\[
\boxed{\text{TIRx primitive-variant table}}
\]

把 `copy/copy_async/permute/gemm/gemm_async/reduce` 的 variant × priority × predicate × layout contract全部完成；

\[
\boxed{\text{vLLM/SGLang persistent-policy table}}
\]

把 backend × stride-layout × content-spec × page-size × model override完全铺开；

以及：

\[
\boxed{\text{FlashInfer scheduler-family table}}
\]

分别整理 FA2 prefill、decode、MLA、TRTLLM-gen、Blackwell task-scheduled path，避免再把它们混成一个FlashInfer策略。

完成这四张之后，再对 CUTLASS/TVM/Gluon/Hexcute/TileLang做一次**反向检索：专门搜索 `heuristic / auto / default / prefer / priority / fallback / layout / swizzle / vector / stage / dispatch`**，如果不再出现新的production decision family，才有资格把框架统计从 B/B+ 升为“完成”。现在仍然不应该开始提炼科研问题。

## 用户

请继续审查 检查哪里没有整理并补充。这里的证据链是否也整理完毕：哪些已有的实现点/layout优化方法的细节确认哪些确实因为假设不充分，假设错误，有什么模型框架支持它。哪些优化策略不够好，还不是最优的：有什么模型框架支持它。这些支持的作为问题的主要证据，从证据最多开始的逐个凝练科研问题；

## 模型思考摘要

继续审查证据链
继续审查证据链

## 模型回复

这轮我会把“证据链”也纳入验收，而不只是补规则目录。最终只有满足下面链条的问题才会被提炼：**源码中的具体策略/假设 → 另一个框架采取相反或补偿策略 → 现代 LLM 常见子图确实触发该冲突 → 两边都有代价、没有支配解**。缺任何一环，都先停留在“工程缺口/实现细节”，不升级为科研问题。

## 模型工具调用参数

{"paths":["GitHub"],"query":"search"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Triton 内存布局优化实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 OptimizeTMemLayouts 关键词

## 模型工具调用参数

{"paths":["GitHub"],"query":"fetch_file"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 TIRx 优先级与调度实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索精确已知名称

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索更广泛的 TIRx 实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 vLLM 自定义 Spec 证据

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 vLLM 其他后端及 NVFP4 支持

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 SGLang 特定覆盖配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检查 overrides.py 通用机制

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索页面大小与注意力后端覆盖配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 FlashInfer 调度器族系

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 FA2DetermineCtaTileQ 实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索任务调度与注意力分块实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索TIRx变体及文档索引

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

Evaluating GPU Kernel Optimization Evidence Chains

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 TileLang 布局推理与流水线规划代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 TileLang 布局成本模型术语

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计自适应布局研究空白

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 vLLM 注意力后端解析代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

评估跨框架性能代理与布局协同设计

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索“only one BufferLoad”注释

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 TileLang 冲突回退实现源代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Swizzle 合并与布局推导

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 CUTLASS 阶段与 CTA/SM 假设

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索布局转换与调度假设

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审查 SGLang KV Cache 布局实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考摘要

检索并评估 Triton、TIRx 及推理框架布局调度实现

## 模型思考：工具活动

正在搜索 Qwen3-8B config num_attention_heads num_key_value_heads head_dim huggingface

## 模型思考：工具活动

正在搜索 huggingface.co

## 模型思考摘要

推进证据核查
推进了证据核查

## 模型思考摘要

已搜索 1 个网站

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计注意力后端布局与块大小配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检查 vLLM FlashInfer 自定义规范

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 customize_spec 相关实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 NVFP4 K/V 存储相关实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

继续审计布局转换与共享内存冲突

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 中最小化银行冲突的实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Triton 布局转换偏好证据

## 模型思考摘要

审计注意力布局配置并检索相关内核实现

## 模型思考用时

思考了 3m 53s

## 模型回复

继续审查后的结论需要分成两层。

第一层，**“每个框架所有 production layout rule 都已经穷举”仍然没有完成**。特别是 Triton、SGLang、FlashInfer 还有规则族没有完全展开成逐条目录；CUTLASS/TIRx/vLLM 已经比较接近。

第二层，和前面不同的是，**有几条跨框架证据链已经基本闭合**：已经不是“我怀疑某个假设不好”，而是可以从源码看到一种策略、看到该策略的限制/repair，再看到其他框架采取相反策略，最后看到现代 LLM workload 正好会触发这种冲突。因此现在可以开始从证据最强的几条逐个凝练科研问题，但仍然要把“源码已经证明的事实”和“尚需实验验证的性能结论”分开。

---

# 一、先更新：还有哪些框架规则没有完全整理？

目前我不会再用一个简单 A/B/C 分数，而是看“是否还存在未展开的自动决策族”。

| Framework | 当前已基本搞清楚 | 仍需继续穷举 |
|---|---|---|
| CUTLASS/CuTe | SM90/SM100 SMEM swizzle、TiledCopy/MMA、StageCountAuto、KernelScheduleAuto、1SM/2SM、cluster/TMA关系 | sparse、mixed-input、block-scaled、epilogue AUTO 的完整规则矩阵 |
| Triton | Coalesce、MMA layout、RemoveLayoutConversions、ThreadLocality、GenericSwizzling、TMA descriptor、TMEM vectorization、pipeline主要阶段 | `OptimizeTMemLayouts`内部全部候选规则、warp-specialization partition heuristics、Coalesce细分阈值 |
| Gluon | explicit layouts、AutoLayout、CoalescedLayout、NVMMAShared default swizzle、显式 warp specialization | ResolveAuto/InferCoalesced内部所有 propagation branch |
| TVM | MetaSchedule流程、CUDA MultiLevelTiling、TensorCore rules、RewriteLayout | 所有 CUDA ScheduleRule/Postproc 对 layout/thread/cache 的逐项影响 |
| TIRx | dispatch机制、priority/predicate、copy/cp.async/tcgen05/gemm部分variant | 全部 tile primitive 的 `(variant, priority, predicate, layout contract)` |
| Hexcute | 4阶段inference、TV constraints、多consumer memory unification、instruction selection框架、cost model、bank-conflict resolver | 各 instruction family 的逐predicate表；剩余stride heuristic的精确规则 |
| TileLang | strict/common/free思路、replication widening、swizzle merge、register-count/io-aware cost、pipeline-before-layout | free candidate生成顺序、reducer/alias全fallback、TMA auto promotion条件 |
| vLLM | layout enum、whole-model resolver、preference voting、block size、`customize_spec`、transfer conversion | 全backend的 `layout × block-size × customize_spec × capability`矩阵 |
| SGLang | generic backend policy、NHD/HND/vectorized5D、override registry结构 | `model_overrides/`全部分类、每个backend自己的representation/page/tile policy |
| FlashInfer | NHD/HND、FA2/FA3、CTA-Q、split-KV、GQA/vec/stage规则、Blackwell TS存在 | FA2/decode/MLA/TRTLLM-gen/Blackwell TS五套scheduler分别完整穷举 |

所以仍然不能说“统计工作结束”。

但这轮有一个进展：**科研问题所需要的某些证据链，不要求等到每一个冷门 backend 的每一个分支都枚举完成。**只要核心假设在多个主要框架中已经有足够直接的源码证据，就可以先形成待验证的研究命题。

---

# 二、证据链也需要分等级

以后我建议对“某个假设有问题”的证据分成四级。

| 证据级别 | 能证明什么 |
|---|---|
| E0 | 框架采用了某个实现策略 |
| E1 | 源码明确写出了该策略的假设、限制或 heuristic |
| E2 | 同一框架存在专门 repair / fallback，说明原策略不是普遍充分的 |
| E3 | 其他主流框架针对同一决策采用相反策略，说明不存在行业公认的确定答案 |
| E4 | 现代 LLM 常见子图实际进入该策略冲突的 workload regime |

只有至少达到：

\[
E1+E2+E3+E4
\]

我才认为值得提升成我们现在定义的“科研问题”。

这也能避免之前那种：

> Hexcute 没算 bank conflict → 一个科研问题

的错误。

---

# 三、先把“确定性的工程缺口”排除掉

这一点非常重要。

Hexcute cost model明确写了，目前 bank conflict、`cp_async_wait_group`、`mbarrier`没有完整计入，而且假设pipeline充分overlap。fileciteturn309file0L1-L2

单独看：

> “把这些因素加进去。”

不是科研问题。

同样，TVM `RewriteLayout` 明确假定相关buffer只由一个 `BufferLoad`读取。fileciteturn334file0L1-L29

单独做：

> “让 RewriteLayout 支持两个 BufferLoad。”

也是工程扩展。

Triton存在可以被证明冗余的 `ConvertLayout` 时：

> “删除这个冗余 ConvertLayout。”

也是确定性优化。

真正值得研究的是这些工程机制背后的**未决策略**。

---

# 四、目前证据最完整的第一条：一个 Tensor 应该维持一个稳定 Layout，还是为 Consumer 专门化？

这一条现在证据明显最强。

## 4.1 vLLM：明显站在“稳定 representation”一侧

当前 resolver源码明确：

> `Resolve one KV cache layout for the whole model.`

而且先收集所有backend的ordered support set，求intersection；不同backend preference冲突时，通过“多少backend把某layout列为第一选择”排序。fileciteturn331file11L292-L303

也就是说：

\[
L_{KV}
=
\operatorname{Resolve}
\left(
\bigcap_i Supported_i
\right).
\]

这里目标首先是：

\[
\boxed{Compatibility + Representation Stability}
\]

而不是每个consumer分别最快。

而 backend甚至不仅影响 stride order。`customize_spec()` 的API明确允许backend修改KV cache内部packing，因为kernel可能要求KV按特定方式打包；preferred block size也由backend给出。fileciteturn341file0L2-L2

例如FlashInfer的NVFP4 backend明确把K/V变成“separate per-head slots of packed fp4 data plus fp8 block scales”。fileciteturn344file0L1-L11

所以persistent representation实际是：

\[
L_{persistent}
=
(
stride,
page,
slot,
packing,
metadata
).
\]

---

## 4.2 但 vLLM 自己已经需要打破这个稳定假设

KV disaggregation path存在：

```text
kv_postprocess_layout_on_receive
kv_postprocess_blksize_and_layout_on_receive
```

也就是说在P/D边界会进行真实persistent representation转换。源码明确描述对收到的KV同时转换layout和block size。fileciteturn338file0L1-L20

这形成了一个非常强的 E2 证据：

\[
\boxed{
\text{one stable layout}
}
\]

是主策略；

但：

\[
\boxed{
\text{某些 system boundary 又需要 specialization/conversion}.
}
\]

注意，这不能证明 whole-model layout 是“错误的”。

它证明的是：

\[
\boxed{\text{它不是在所有 consumer boundary 上都充分。}}
\]

---

# 五、同一个冲突在 Triton 的 kernel 内部再次出现

Triton当前 `RemoveLayoutConversions` 的说明非常直接：

对于昂贵load/store倾向：

\[
BlockedEncodingAttr
\]

因为适合coalescing；

其他情况倾向：

\[
NvidiaMmaEncodingAttr
\]

因为适合tensor operation。fileciteturn347file0L1-L23

所以：

\[
L_{\text{memory}}
\neq
L_{\text{MMA}}.
\]

Triton的解决方案不是强迫二者共用layout，而是：

\[
L_1
\rightarrow
ConvertLayout
\rightarrow
L_2
\]

再通过 `RemoveLayoutConversions` / rematerialization 判断哪些conversion不值得保留。

甚至GenericSwizzling又以“总bank conflicts最低”为自己的局部目标。fileciteturn346file0L1-L20

因此Triton事实上采取：

\[
\boxed{\text{consumer specialization + transition reconciliation}}
\]

这和vLLM的 whole-model compromise 是完全不同的策略。

这已经提供了 E3。

---

# 六、Hexcute 又给出了第三种答案：让多个 Consumer 共同约束同一个 Layout

Hexcute/Hidet的 layout inference 对shared tensor的多个Copy consumer建立多个memory constraints。

每消费一次shared tensor，就进一步refine其stride；如果constraints矛盾就backtrack，全部满足以后再用heuristic决定剩余未确定strides。fileciteturn308file0L1-L2

所以Hexcute更接近：

\[
\boxed{
\text{寻找一个满足多个 consumers 的 compromise representation}
}
\]

而不是给每个consumer单独layout。

这与Triton形成非常漂亮的对照：

\[
\text{Hexcute: unify}
\]

vs

\[
\text{Triton: specialize + convert}.
\]

---

# 七、TileLang 给出了第四种机制：冲突时扩大 replication

TileLang layout inference遇到reducer多个update site不同意时，源码会：

> widening to the participant-wide plan

即把mapping向更广泛participant-owned / replicated layout放宽。fileciteturn335file1L12-L27

两个non-fragment swizzle layout发生冲突时，又会尝试：

\[
MergeSwizzleLayouts(existing,new)
\]

而不是立即选择其中一个。fileciteturn336file0L1-L16

所以它引入了第四种可能：

\[
\boxed{\text{replicate / widen ownership to satisfy multiple uses}}
\]

。

目前已经出现四种实际production strategy：

\[
\begin{array}{ll}
\text{vLLM} & \text{stable common representation}\\
\text{Hexcute} & \text{constraint-based compromise}\\
\text{Triton} & \text{specialize + convert}\\
\text{TileLang} & \text{merge / replicate / widen}
\end{array}
\]

这已经远远超过“某一个框架做得不好”的证据标准。

---

# 八、现代 LLM 又恰好强化这种 multi-consumer pressure

Qwen3-8B 是：

\[
H_q=32,\qquad H_{kv}=8,\qquad D=128,
\]

即GQA，一个KV head被多组query heads复用，同时最大context约40K。citeturn352329search0

DeepSeek-V3更明显：

\[
KV_{latent}=512,
\]

\[
D_{qk,nope}=128,
\quad
D_{qk,rope}=64,
\quad
D_v=128,
\]

同时还有256个routed experts、每token选择8个expert、128×128 block-wise FP8 quantization。citeturn352329search1

也就是说现代模型正在增加：

\[
\text{shared consumers}
+
\text{heterogeneous representations}
+
\text{persistent states}
+
\text{metadata}.
\]

这完成了 E4。

---

# 九、因此第一条科研问题现在可以比较严谨地凝练

不是：

> “vLLM的KV layout太固定。”

也不是：

> “Triton convert_layout太多。”

而是：

\[
\boxed{
\textbf{RQ1:
At what granularity should a tensor's physical representation be specialized across heterogeneous consumers?}
}
\]

中文：

> **面对具有不同layout偏好的多个consumer，一个tensor的物理representation应该保持多稳定？应当共享一个compromise layout、在consumer边界转换，还是维护replicated/specialized representations？**

这是真正的 trade-off：

\[
T_{\rm consumer}
+
T_{\rm convert}
+
M_{\rm replication}
+
C_{\rm management}.
\]

没有一个策略可以显然支配另一个。

我目前给它的证据评级是：

\[
\boxed{\textbf{E1+E2+E3+E4：完整}}
\]

这是现在证据链最完整的研究问题。

---

# 十、第二条强证据链：Layout / Mapping / Instruction / Pipeline 应该联合多少？

这是当前compiler层面证据最强的一条。

CUTLASS的一端非常清楚。

`StageCountAuto` 会根据每stage的SMEM footprint尽可能提高shared-memory使用，并**明确假设1 threadblock/SM occupancy**。fileciteturn337file0L1-L24

而其 `CollectiveMma` 将：

\[
TiledMma,
GmemTiledCopy,
SmemLayout,
SmemCopy,
Stages,
Cluster,
KernelSchedule
\]

深度组织在同一个collective体系里。

也就是：

\[
\boxed{\text{strongly coupled construction}}
\]

。

---

## Triton 位于另一端

Triton分别存在：

\[
Coalesce
\]

\[
AccelerateMatmul
\]

\[
OptimizeThreadLocality
\]

\[
GenericSwizzling
\]

\[
OptimizeTMemLayouts
\]

\[
Pipeline
\]

\[
AutomaticWarpSpecialization.
\]

然后反复：

\[
RemoveLayoutConversions.
\]

GenericSwizzling自己优化bank conflicts；TMA descriptor又用独立layout chooser；TMEM又使用自己的register-budget/vectorization rules。fileciteturn346file0L1-L20 fileciteturn315file1L12-L28

因此Triton是：

\[
\boxed{\text{progressive local decisions + repair}}
\]

而不是joint construction。

---

# 十一、Hexcute、TileLang、TIRx 又各选了不同切分边界

Hexcute联合：

\[
Layout+TaskMapping+Instruction
\]

但dataflow/pipeline在其自动layout synthesis外部；cost model还明确假设copy/MMA充分overlap。fileciteturn308file0L1-L2 fileciteturn309file0L1-L2

TileLang 当前采取：

\[
PipelinePlanning
\rightarrow
LayoutInference,
\]

即先物化pipeline结构，让layout inference看到最终pipeline。

TIRx则更模块化：

\[
StorageLayout
\rightarrow
PrimitiveDispatch,
\]

后者根据layout/scope/target按priority/predicate选择implementation。例如 `copy_async/ldgsts` priority=20，而SMEM→TMEM `tcgen05` path priority=10。fileciteturn317file1L15-L34 fileciteturn317file2L36-L46

`gemm_async/tcgen05`也是priority 10的局部variant。fileciteturn318file0L1-L20

所以现在实际存在：

\[
\text{CUTLASS: strongly coupled}
\]

\[
\text{Triton: staged + repair}
\]

\[
\text{Hexcute: partial joint synthesis}
\]

\[
\text{TileLang: pipeline first, layout second}
\]

\[
\text{TIRx: storage + local primitive dispatch}.
\]

如果存在一个显然正确的 decomposition，不应该出现这么不同的生产设计。

---

# 十二、第二条研究问题应该这样写，而不是“联合优化更好”

直接问：

> Joint optimization是否比sequential optimization快？

几乎没有科学性，因为搜索空间越大理论上不会更差。

真正的问题应该是：

\[
\boxed{
\textbf{RQ2:
Which layout-related decisions must be optimized jointly, and which can be safely modularized?}
}
\]

更形式化一点：

\[
D=
\{
L_G,L_S,L_R,
Mapping,
CopyInst,
MMAInst,
Pipeline,
WarpRole
\}.
\]

研究decision interaction graph：

\[
G=(D,E).
\]

如果改变 \(D_i\) 会让 \(D_j\) 的performance ranking反转：

\[
T(d_i^1,d_j^1)<T(d_i^1,d_j^2)
\]

但：

\[
T(d_i^2,d_j^1)>T(d_i^2,d_j^2),
\]

则两者具有强coupling。

真正要回答：

> Attention / GEMM / MoE / MLA 中，哪些 edge 必须joint optimize，哪些 edge 可以安全分解以控制搜索复杂度？

这里两侧trade-off明确：

\[
\text{Joint}
\Rightarrow
\text{larger search + better coupling awareness}
\]

\[
\text{Modular}
\Rightarrow
\text{tractability + composability + potential interaction loss}.
\]

这条也已经达到：

\[
\boxed{E1+E2+E3}
\]

而现代Attention/MLA/TMA/WGMMA/TMEM给了很强E4 workload pressure。

所以我把它放第二位。

---

# 十三、第三条证据最广：结构规则 / 启发式到底能替代多少真实性能反馈？

这是覆盖框架数量最多的一条，但是概念比前两条更宽。

现在源码可以非常清楚地看到不同框架站在完全不同位置。

CUTLASS的 `StageCountAuto` 是structural heuristic，而且明确假设1 CTA/SM。fileciteturn337file0L1-L24

Triton GenericSwizzling直接最小化bank conflicts；TMEM又用“最多使用一半register”这类resource heuristic。

TVM是：

\[
ScheduleRules
\rightarrow
XGB
\rightarrow
HardwareMeasurement.
\]

Hexcute则：

\[
Constraints
\rightarrow
InstructionLatencyModel,
\]

同时明确假设perfect overlap且目前不完整考虑bank conflicts/barriers。fileciteturn309file0L1-L2

TileLang甚至把这个分歧直接变成配置：

默认：

\[
\texttt{register-count}
\]

另有：

\[
\texttt{io-aware},
\]

后者估计global-memory vector width/coalescing/bytes moved，再用register count tie-break。fileciteturn330file2L32-L42 fileciteturn330file8L116-L126

TIRx是static：

\[
predicate+priority.
\]

vLLM是：

\[
support\ intersection+preference\ voting.
\]

SGLang当前甚至建立了完整declarative model-override registry，用model architecture conditional logic继续修正generic defaults。fileciteturn322file0L1-L7

FlashInfer的CTA tile同样由固定函数 `FA2DetermineCtaTileQ`选择；对于大VO dim，其kernel实例化甚至只保留 `{16,32}`，普通情况保留 `{16,64,128}`。fileciteturn325file0L1-L10

Blackwell新的task-scheduled attention又把 scheduling、tile selection、split-KV明确设成内部自动策略，而不是用户tuning knob。fileciteturn326file0L1-L27

这是十个框架几乎全部都有的证据。

---

# 十四、但这里要非常小心：不能把“heuristic不是最优”直接当事实

源码只能证明：

\[
\text{decision}=H(features)
\]

而不是：

\[
\text{decision}=\arg\min T.
\]

它不能证明：

\[
H(features)
\]

性能不好。

因此科研问题必须保持trade-off形式：

\[
\boxed{
\textbf{RQ3:
How much empirical performance feedback is necessary for robust layout selection?}
}
\]

或者更layout-specific：

> **硬件结构约束、instruction legality、bank-conflict、register footprint等静态信息，对near-optimal layout区域到底有多强的预测能力？什么时候基于规则/解析模型已经足够，什么时候必须使用hardware measurement？**

两侧代价：

\[
\text{rules/analytic}
\Rightarrow
\text{cheap, fast, deployable for dynamic shapes}
\]

\[
\text{measurement}
\Rightarrow
\text{more faithful but tuning cost explodes}.
\]

现代LLM serving的状态：

\[
(B,S_q,S_{kv},GQA,phase,page,spec,\ldots)
\]

高度动态，因此不可能对所有状态离线测量。

这一条证据广度：

\[
\boxed{\text{10/10 frameworks}}
\]

但它的新颖性需要以后再审查论文；目前只论源码证据，它非常强。

---

# 十五、第四条正在形成：固定规则几何 vs 现代大模型 irregularity

这条证据也越来越完整。

CUTLASS大量规则要求tile/instruction divisibility；其SMEM swizzle选择也是从一组离散、instruction-compatible layout中挑。

TIRx的async copy明确把合法vector width限制到：

\[
\{128,64,32\}\ bit
\]

再根据layout、alignment、thread count找最大合法宽度。fileciteturn317file1L15-L34

FlashInfer把CTA Q tile限制成离散集合，并根据head dimension、packed query length、GQA等分支决定。fileciteturn325file2L23-L32

vLLM不但有layout enum，还允许backend改变preferred block size和内部content packing。fileciteturn319file0L1-L16 fileciteturn341file0L2-L2

SGLang的override系统甚至直接处理page-size constraints；当前generic override路径中可以因为kernel支持限制自动改page size。fileciteturn321file0L1-L24

与此同时现代模型有GQA、MLA、MoE动态token分配、不对称QK/V维度、长context等。Qwen3-8B的32/8 GQA和DeepSeek-V3的MLA+256 experts/top-8就是具体例子。citeturn352329search0turn352329search1

因此这一条可以开始形成：

\[
\boxed{
\textbf{RQ4:
When should irregular LLM workloads be regularized into hardware-friendly layout families, and when should the layout system preserve their native irregularity?}
}
\]

但我暂时不会把它排到前3，因为“regularity vs irregularity”本身是更经典的问题，需要未来证明现代LLM的layout场景带来新的现象。

---

# 十六、另一个很重要但应该作为机制而不是立即独立成题的问题：局部Proxy目标之间互相冲突

现在已经看到很多框架各自在优化不同proxy：

\[
\text{CUTLASS StageCountAuto}
\rightarrow
\max SMEM\ usage
\quad(1CTA/SM)
\]

\[
\text{Triton GenericSwizzle}
\rightarrow
\min bank\ conflicts
\]

\[
\text{Triton Coalesce}
\rightarrow
memory\ coalescing
\]

\[
\text{TileLang default}
\rightarrow
\min register\ count
\]

\[
\text{TileLang io-aware}
\rightarrow
\min estimated\ GMEM\ cost
\]

\[
\text{Hexcute}
\rightarrow
instruction\ count\times latency
\]

\[
\text{FlashInfer}
\rightarrow
occupancy/grid\ utilization/register-pressure\ rules.
\]

这些proxy都合理。

但是它们显然不是同一个目标函数。

这可以作为 RQ2/RQ3 的重要解释机制，而不是现在单独立题：

\[
\boxed{
\text{Local proxy optimality}
\not\Rightarrow
\text{global performance optimality}
}
\]

这里“\(\not\Rightarrow\)”目前仍是待验证命题，不是已经被源码证明的性能结论。

---

# 十七、当前证据链完整度，可以这样重新判断

| 候选问题 | 多框架源码策略 | 同框架 repair/fallback | 相反策略 | 现代LLM压力 | 可以开始凝练？ |
|---|---:|---:|---:|---:|---:|
| Representation稳定性 vs specialization/convert/replicate | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★★ | **是，最强** |
| Layout/Mapping/Instruction/Pipeline joint vs modular | ★★★★★ | ★★★★☆ | ★★★★★ | ★★★★★ | **是** |
| Structural heuristic vs empirical feedback | ★★★★★ | ★★★★☆ | ★★★★★ | ★★★★★ | **是** |
| Regular geometry vs irregular workload | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★★★ | **接近** |
| Local proxy vs global objective | ★★★★★ | ★★★★★ | ★★★★☆ | ★★★★★ | 先作为机制 |
| Storage-only layout vs execution-rich layout | ★★★★☆ | ★★★☆☆ | ★★★★★ | ★★★★☆ | 暂缓 |
| Persistent layout runtime adaptation | ★★★☆☆ | ★★★★★ | ★★★★★ | ★★★★★ | serving子问题 |

所以现在第一次可以比较有把握地说：

\[
\boxed{
\text{前三条已经有足够源码证据成为真正的待验证科研问题。}
}
\]

而不是我们以前那种从一个implementation limitation直接跳到“科研问题”。

---

# 十八、同时必须保留一个严格边界：哪些话现在还不能说

目前**可以说**：

> 多个框架针对同一个layout冲突选择了不同策略。

可以说：

> 框架内部出现专门repair/fallback，说明原始策略不是对所有程序都充分。

可以说：

> 某些性能选择基于显式heuristic或proxy，而不是global hardware search。

可以说：

> 现代GQA/MLA/MoE/long-context workload会产生这些决策变量。

但目前**不能说**：

> Triton staged layout一定比joint layout差。

不能说：

> vLLM one-layout policy导致X%性能损失。

不能说：

> Hexcute constraint compromise一定不如consumer-specialized layout。

不能说：

> CUTLASS StageCountAuto的1 CTA/SM假设在LLM上一定错。

这些都必须进入后面的实验验证。

这一区分很重要，因为科研问题真正要验证的恰恰就是：

\[
\boxed{\text{什么时候这些假设失效，而不是预先宣判它们错误。}}
\]

---

## 当前下一步

接下来我建议继续两件事并行推进，但仍然不查论文。

一方面继续完成剩余规则目录，尤其是 Triton `OptimizeTMemLayouts/warp specialization`、TIRx完整primitive registry、SGLang model overrides、FlashInfer各scheduler family，避免证据存在盲区。

另一方面从现在最强的 **RQ1：representation stability vs specialization** 开始建立真正的“证据审判表”：逐框架列出具体源码策略、隐含假设、repair mechanism、相反框架策略，以及 Qwen3 GQA、DeepSeek MLA、P/D KV transfer、Prefill Attention 这几个真实子图分别怎样施加压力。只有这一条被彻底审判完，再进入 RQ2，而不是一次性列十几个问题。

## 用户

请继续审查 检查哪里没有整理并补充。这里的证据链是否也整理完毕：哪些已有的实现点/layout优化方法的细节确认哪些确实因为假设不充分，假设错误，有什么模型框架支持它。哪些优化策略不够好，还不是最优的：有什么模型框架支持它。这些支持的作为问题的主要证据，从证据最多开始的逐个凝练科研问题；

## 模型回复

我继续按“先补规则目录，再做证据审判”的顺序推进。这次会特别区分三种结论：**源码直接证明假设不充分**、**源码只证明存在相反策略**、**是否更优仍需实验**；只有第一类+第二类都充分，才进入科研问题。

## 模型工具调用参数

{"paths":["GitHub"],"query":"fetch_file"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计 Triton TMem 布局优化实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Triton 布局优化更多代码模式

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton Warp 特化精确启发式策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton CMake 特定文件名

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取自动 Warp 特化实现文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析分区调度启发式算法

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 PartitionScheduling 文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 TritonGPU Warp 特化目录列表

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索分区调度启发式代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

读取 Triton 分区调度源文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取启发式名称相关代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析分区调度启发式实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析优化分区 Warp 启发式策略

## 模型思考摘要

审计并分析 Triton Warp 特化与分区调度启发式实现

## 模型回复

Triton 这一轮已经补出两条此前缺失、而且很关键的证据：`OptimizeTMemLayouts` 会根据后续 reduction / local-load/store consumer 改 TMEM layout；warp-specialization 又用非常粗的寄存器代理模型决定每个 partition 的 warps，并在 warps 改变后**重新跑一遍 layout assignment**。这说明 Triton 自己已经显式承认 `layout ↔ warp count ↔ register pressure ↔ consumer` 是耦合的，而它用的是分阶段 heuristic 近似。 

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

列出 TIRx 相关文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 CUDA Tile Primitive 调度注册大小规律

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 TVM 调度注册与图块原语实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 TVM CUDA Tile Primitive 实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 CUDA copy_async 目录内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 SGLang 模型覆盖实现与研究问题

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取模型覆盖配置初始化文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索模型覆盖配置与缓存参数

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 SGLang 模型覆盖配置字段

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索模型覆盖与注意力后端页面配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审查模型覆盖配置与注意力派生参数

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审查并扩展 Qwen3.5 与 Qwen3-VL 支持

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 Qwen3-VL 模型配置

## 模型思考摘要

检索并梳理 TIRx、CUDA Tile Primitive 与 SGLang 模型覆盖配置

## 模型回复

SGLang 也比前面更明确：它不是一套“通用默认规则”，而是 `generic policy → model-family override → backend-internal policy` 三层叠加。DeepSeek DSA 会根据平台/consumer能力把 page size 设成 1、64 或 128；Qwen3.5 在 SM100 上甚至把 backend 与 page size 联动成 `trtllm_mha+64` 或 `triton+1`。这类模型级修正本身就是“通用规则不充分”的直接 E2 证据。 

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

计算 TileLang 布局推断成本

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检查默认寄存器计数定义

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索布局推断中的候选枚举逻辑

## 模型思考摘要

规划系统审查
完成系统审查规划

## 模型思考摘要

计算并检查 TileLang 布局推断与寄存器计数逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 TVM 默认 CUDA 配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 schedule_rule.cc 默认配置相关代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 TVM 默认 CUDA 后处理配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取后处理默认实现代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 TVM 后处理文件早期代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Triton Gluon 编译器编码推导流程

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

Searching Triton source paths and encoding resolution implementations

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 Triton Gluon 转换目录

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取并分析合并编码推断实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 ResolveAuto 编码实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审查布局推断与冲突处理逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查找并获取 Hexcute 指令选择文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取并评估 Hidet GitHub 仓库

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 Hexcute 指令选择文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索指令候选排序与布局规范化逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

Inspecting FlashInfer Scheduler and Cost Model

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 FA2DetermineCtaTileQ 函数位置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取并检查 FlashInfer 工具函数实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

定位并查看 FA2Determine 实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

Searching FlashInfer’s split-KV scheduler implementation

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 scheduler.cuh 第1至400行内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

评估GPU内核调度与自动搜索策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 CUTLASS SM90 Swizzle 与 Stage 配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检查TIRx变体初始化文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计各模块初始化文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 CUDA tile primitive 拷贝模块内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 GEMM 初始化文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

初始化 GEMM 异步模块

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查看 CUDA 归约原语实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 CUDA 置换布局初始化代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取并梳理 TIRx CUDA 目录注册分发逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 vec_auto、ldstmatrix 及回退实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 gmem_smem 向量拷贝实现

## 模型思考摘要

检索并梳理GPU编译调度、布局推断与CUDA实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 Hexcute 成本模型

## 模型思考摘要

梳理审计框架
核对证据链
细化实现证据
梳理审计框架并核对实现证据链

## 模型思考摘要

获取了 Hexcute 成本模型

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索精确支持的布局

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索并引用 customize_spec 相关实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 SGLang 模型覆盖字段并分类

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计主流模型架构与FlashInfer调度支持

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检查 FlashInfer 的 Cute DSL 目录

## 模型思考摘要

搜索并审计布局、模型覆盖及FlashInfer支持

## 模型思考用时

思考了 8m 25s

## 模型回复

这一轮审查之后，需要对前面的状态再做一次修正：**框架规则目录仍没有 100% 穷举完成，但已经有几条证据链达到“可以开始凝练科研问题”的标准。**后续要同时做两件事：继续补漏；同时只对已经闭环的问题进行凝练，不再把“一个 heuristic 看起来粗糙”直接当科研问题。

我先把证据强度定义固定下来。源码能证明的东西分三类：

\[
\boxed{
A:\ \text{已证明该假设不能普遍成立}
}
\]

典型证据是框架自己存在 override、repair、conversion、重新 layout、fallback。

\[
\boxed{
B:\ \text{已证明它只是 heuristic/proxy，而非最优解}
}
\]

源码明确写 heuristic、rough estimate、priority、固定阈值等；但“性能差”仍需要实验。

\[
\boxed{
C:\ \text{确定性工程缺口}
}
\]

例如某条路径目前只支持一种 shape。它本身不是科研问题。

下面按这个标准继续。

---

# 1. 最新补漏：Triton 比前面统计的还要“启发式化”

这轮把 Triton 的 warp specialization继续扒下去以后，发现证据非常强。

`PartitionScheduling.cpp` 并不是一个形式化优化器。它先让每个 data operation拥有独立 partition，然后用一组**有顺序的 heuristic rules**不断 merge partition直到 fixed point。源码明确这样描述。fileciteturn357file0L2-L2

具体规则甚至包括：

- view应该和 producer 还是 consumer合并：比较通信元素量；
- expensive SFU与dependency合并；
- TMEM load与consumer合并；
- load/store/MMA partition各自合并；
- `NONE` op更倾向并入costly group；
- 当partition之间边的数据量：

\[
>16384
\]

时尝试merge。

最值得注意的是源码自己在这个阈值旁写：

> `FIXME: seemingly arbitrary size...`

fileciteturn358file0L2-L2

所以这里已经不是我们推测：

> Triton可能用了不够精确的heuristic。

而是源码直接证明：

\[
\boxed{
\text{warp partitioning 使用人为 proxy/threshold，而不是性能最优求解。}
}
\]

这是 B 类证据。

---

# 2. Triton register/warp/layout 决策的假设也非常明确

`OptimizePartitionWarps.cpp` 更重要。

源码直接写：

> `Extremely rough estimate of the number of registers needed per partition.`

然后：

1. 找partition中最大的tensor；
2. 假定：

\[
\boxed{
\text{largest tensor accounts for half of registers used by a warpgroup}
}
\]

所以把它的估算乘2；

3. 假定Blackwell整个SM寄存器池：

\[
2^{16}
\]

个32-bit registers；

4. 因为当前不能对partition控制`maxnreg`，又假定：

\[
\boxed{
PTXAS\ evenly\ distributes\ registers\ across\ all\ warps
}
\]

5. TMA-load partition至少保留2 warps；
6. TMEM partition至少4 warps；
7. 如果估计寄存器还容得下，就不断把partition warp数减半。

fileciteturn360file0L1-L7

而最关键的是：

> warp数改变之后，Triton会清掉tensor encodings，并重新执行layout assignment、Coalesce、RemoveLayoutConversions、OptimizeThreadLocality、AccelerateMatmul。

也就是说源码自己明确承认：

\[
\boxed{
WarpCount
\leftrightarrow
RegisterPressure
\leftrightarrow
Layout
}
\]

不是独立变量。fileciteturn360file0L1-L7

这对后面的证据链很重要。

---

# 3. Triton TMEM layout 也不是一个固定 lowering

之前我们把TMEM统计得过粗。

`OptimizeTMemLayouts`现在明确会**根据consumer改变TMEM layout**：

- 后面有沿N维reduction时改变TMEM distribution，减少cross-warp reduction；
- TMEM→local/store vectorization不好时换另一种distribution；
- shared→TMEM path类似；
- 然后插layout conversion，并依赖后续RemoveLayoutConversions去repair。

fileciteturn348file0L2-L2

这说明当前Triton不是：

\[
Storage\ Layout\rightarrow Fixed\ Lowering
\]

而是：

\[
\boxed{
Consumer
\rightarrow LayoutPreference
\rightarrow Conversion
\rightarrow Repair
}
\]

。

因此此前对Triton的总结应该正式修改成：

\[
\boxed{
\text{progressive proxy-driven specialization + reconciliation}
}
\]

而不是简单的“compiler自动layout”。

---

# 4. Gluon 的 AutoLayout 也终于可以说清楚了

以前只知道有AutoLayout，现在其具体选择逻辑已经补齐。

`InferLayoutUtils.cpp` 给每个推导layout一个：

\[
distance
\]

。Join/Split/Reshape/Transpose这类“fuzzy inference”每经过一次就让distance增加。

冲突时：

\[
distance(L_1)<distance(L_2)
\Rightarrow
L_1
\]

即偏向距离seed更近的layout。fileciteturn387file0L1-L7

如果两个layout：

\[
distance_1=distance_2>0
\]

则并不是测性能，而是通过稳定hash ordering决定。

如果：

\[
distance=0
\]

的两个seed直接冲突，则报错。

因此 Gluon AutoLayout的真实策略是：

\[
\boxed{
Seed
+
ConstraintPropagation
+
MinimumInferenceDistance
}
\]

而不是：

\[
\arg\min_L T(L).
\]

此外 AutoEncoding不能跨函数边界，必须fully inline。fileciteturn387file0L1-L7

CoalescedLayout又是一套独立机制：AxisInfo分析load/store，再构建coalesced encoding并前后传播，而且当前要求：

\[
numCTAs=1.
\]

fileciteturn385file0L1-L7

所以 Gluon现在的规则目录基本已经补得比较完整。

---

# 5. TIRx 的规则目录也进一步明确了

TIRx当前CUDA tile primitive surface已经确定包含：

\[
copy,\ copy\_async,\ elementwise,\ gemm,\ gemm\_async,\ permute\_layout,\ reduction.
\]

fileciteturn400file0L1-L7

同步copy至少包括：

- fixed `vec_16b`
- `vec_32b`
- `vec_64b`
- `vec_128b`
- `vec_256b`
- `vec_auto`
- `ld/stmatrix`
- scalar/general fallback

其中forced vector variants：

\[
priority=20
\]

并要求用户显式dispatch，而且shape必须**精确匹配vector instruction宽度**。fileciteturn407file0L1-L7

`vec_auto`：

\[
priority=10
\]

先检查global↔shared，再尝试register path。fileciteturn408file0L1-L7

global↔shared路径又有一个非常强的结构假设：

\[
N_{\rm elements}\bmod N_{\rm threads}=0.
\]

否则直接reject。

dynamic region extent同样reject；源码还明确说，不愿意用一个slow scalar emit去“paper over”这种poorly-shaped copy。fileciteturn409file0L1-L7

所以这里已经可以精确写：

\[
\boxed{
\text{TIRx primitive dispatch 偏好规则、整齐、可均分的 work partition。}
}
\]

但注意，这还只是策略事实。

它不能证明：

> 这个策略性能差。

---

# 6. TVM 现在也基本能把“搜索之前固定了什么”列清楚

DefaultCUDA当前直接规定：

\[
structure=SSSRRSRS
\]

thread-binding：

\[
blockIdx.x,\ vthread.x,\ threadIdx.x
\]

vector-load candidates：

\[
\{1,2,3,4,8,16\}
\]

read reuse：

\[
\text{MUST shared at level 4}
\]

write reuse：

\[
\text{MUST local at level 3}.
\]

CrossThreadReduction只从：

\[
\{4,8,16,32,64,128,256,512\}
\]

搜索thread extent。

AutoBind也只从：

\[
\{32,64,128,256,512,1024\}
\]

选择。fileciteturn378file0L1-L7

TensorCore路径则预先提供一个有限WMMA/MMA intrinsic catalog，并仍使用：

\[
SSSRRSRS.
\]

其中一类schedule：

\[
shared\ read+write,\quad
software\ pipeline=false
\]

另一类：

\[
shared\ read,\ no\ write\ reuse,\quad
software\ pipeline=true.
\]

fileciteturn378file0L1-L7

然后MetaSchedule才去search+measurement。

因此 TVM 真正的策略是：

\[
\boxed{
StructuralPrior
\rightarrow
CandidateSpace
\rightarrow
Measurement
}
\]

而不是“硬件measurement自动解决layout”。

Default CUDA postproc还包括：

- DisallowDynamicLoop
- RewriteCooperativeFetch
- RewriteUnboundBlock
- RewriteParallelVectorizeUnroll
- RewriteReductionBlock
- VerifyGPUCode

TensorCore再在合法性验证之后做RewriteTensorize。fileciteturn381file0L1-L7

这使TVM这一列已经比前面完整很多。

---

# 7. vLLM：现在确认 persistent layout 应该拆成至少四个独立决策

前面的：

\[
LBHNC/LBNHC
\]

还是太粗。

目前应该写成：

\[
\boxed{
R_{KV}
=
(
L_{\rm stride},
L_{\rm content},
B_{\rm page},
L_{\rm metadata}
)
}
\]

。

### stride

backend对stride preference完全不同。

当前源码实例包括：

CPU：

\[
LBHNC
\]

HPC：

\[
LBNHC
\]

B12X：

\[
LBHNC,BLHNC
\]

QSA：

\[
BLNHC,BLHNC
\]

FlexAttention：

\[
LBNHC
\]

而Flex明确说明原因是只有LBNHC能让：

\[
(B,N)
\]

zero-copy flatten。fileciteturn412file3L36-L46 fileciteturn412file4L47-L57 fileciteturn412file7L85-L95 fileciteturn412file12L140-L151

所以backend之间的layout preference不是人为造出来的，它来自真实consumer addressing constraints。

---

### content packing

`AttentionBackend.customize_spec()`的接口注释现在非常关键：

> 当kernel希望KV按特定方式packing时，由backend调整spec。

而且源码还明确称它目前只是temporary compatibility API，未来理想状态是backend直接构造完整spec。fileciteturn413file2L49-L75

具体已经有：

- native ROCm：K/V为两个head groups；
- AITER unified：保持K/V packed在content dim；
- Triton per-head quant：每个head数据后内联FP32 scale；
- TurboQuant：K+V合并成每head一个slot；
- FlashInfer NVFP4：K/V分开的packed-FP4 head slots + FP8 block scales；
- MLA又有自己的latent+scale packing。

fileciteturn413file0L1-L24 fileciteturn413file1L26-L48 fileciteturn413file4L107-L117 fileciteturn413file5L132-L143 fileciteturn413file7L184-L195

因此以后绝不能把：

> KV layout = HND/NHD

继续当完整描述。

---

# 8. vLLM 本身还继续提供了“稳定 representation 不充分”的证据

核心resolver依然：

\[
\boxed{\text{one KV cache layout for the whole model}}
\]

而且backend preference用：

\[
\text{intersection}
+
\text{first-choice voting}.
\]

fileciteturn332file0L1-L7

但NIXL/transfer代码又存在真实：

\[
block\ size + layout
\]

联合转换。fileciteturn338file0L1-L20

所以可以确认：

> “一个representation永远适合所有边界”作为**普遍命题**是不成立的。

但不能说：

> whole-model local KV layout这个设计是错误的。

因为稳定layout避免大量长期KV转换和metadata复杂度。

这是非常典型的：

\[
\boxed{
\text{假设不普适，但策略本身可能仍是正确trade-off。}
}
\]

---

# 9. SGLang 的漏项也进一步收敛了

目前其model override目录至少覆盖 DeepSeek、Gemma、GLM、GPT-OSS、Inkling、Kimi、Llama4、Qwen3.5、Qwen3-MoE、Qwen3-VL、Qwen4-exp等大量真实模型族。fileciteturn367file0L1-L7

DeepSeek这条路径尤其能说明问题。

DSA：

在ROCm，如果AITER preshuffle paged-MQA不可用：

\[
page\_size=1.
\]

如果路径可用：

\[
page\_size=64.
\]

XPU：

\[
page\_size=128.
\]

而SM100普通DeepSeek-V3又会优先：

\[
trtllm\_mla.
\]

fileciteturn371file0L1-L7

Qwen3.5更加直接。

在SM100：

如果满足TRTLLM-MHA的spec/radix等条件：

\[
Backend=TRTLLM\_MHA,
\qquad PageSize=64.
\]

否则：

\[
Backend=Triton,
\qquad PageSize=1.
\]

fileciteturn372file0L1-L7

这是一条很强的源码证据：

\[
\boxed{
Backend
\not\perp
PageGeometry.
}
\]

也就是说 backend、persistent layout/page geometry 不能假设彼此独立。

Qwen3-VL在ROCm AITER unified-attention下又自动：

\[
page\_size=16.
\]

大型Hopper Qwen3-VL甚至采用profiled serving defaults，并默认让decode走FlashInfer。fileciteturn373file0L1-L7

因此SGLang已经非常清楚地证明：

\[
\boxed{
generic\ hardware/model\ rule
}
\]

对现实模型并不足够，需要：

\[
\boxed{
model\text{-}family\ repair/override.
}
\]

这是A类“非普适性”证据。

---

# 10. FlashInfer 的 scheduler 也比我们最初认为的更接近一个 layout/mapping policy engine

其基础dispatch空间已经非常离散：

\[
NUM\_MMA_Q\in\{1,2\}
\]

\[
NUM\_MMA_{KV}\in\{8,4,2,1\}
\]

\[
CTA_Q\in\{16,32,64,128\}
\]

\[
GQA\in\{1,2,3,4,6,8\}
\]

\[
Vec\in\{16,8,4,2,1\}.
\]

decode pipeline：

\[
CC\ge8
\Rightarrow2\ SMEM\ stages,
\]

否则1 stage。fileciteturn394file0L2-L2

而真正有意思的是 split-KV。

它不是固定heuristic，而是先根据：

\[
HeadDim,
GQA,
dtype,
SMEM
\]

构造threadblock geometry，然后调用真实CUDA occupancy API：

\[
cudaOccupancyMaxActiveBlocksPerMultiprocessor.
\]

得到最大grid capacity。

如果当前：

\[
batch\times KVHeads
\]

已经能填满grid，就不split。

否则对每个request的paged KV长度做binary search，选chunk使任务数接近GPU可并行容量。fileciteturn397file0L2-L2

这实际上是很有价值的对照：

FlashInfer不是：

\[
\text{pure static rules}
\]

也不是：

\[
\text{benchmark every configuration}.
\]

它是：

\[
\boxed{
\text{runtime workload geometry}
+
\text{hardware occupancy query}
+
\text{analytic scheduling rules}.
}
\]

后面讨论ranking方法时，它应该被单独分类，不能继续和SGLang纯hand-written backend rule混为一类。

---

# 11. TileLang：现在也可以确认“proxy choice”本身就是显式设计选择

TileLang free-mode不是只产生一个layout。

源码明确说：

> per candidate root枚举完整layout attempts，然后保留cost最低的完整assignment。

当前至少两种cost model：

默认：

\[
C_{\rm reg}=total\ fragment\ register\ slots
\]

以及：

\[
C_{\rm IO}
=
\sum_s
\max(
bytes_{\rm bandwidth}(s),
bytes_{\rm issue}(s)
)
\]

再以register count作为lexicographic tiebreak。fileciteturn375file0L1-L7

I/O-aware甚至把不能建模的：

- non-affine
- dynamic
- swizzled
- non-bijective

candidate直接收取conservative worst-case cost，以防“看不懂的layout反而因为cost缺失而获胜”。fileciteturn374file0L2-L2

这里特别有价值的一点是：

\[
\boxed{
\text{同一个框架提供两个不同的layout ranking proxy。}
}
\]

这本身就说明：

> “应该根据什么信息判断layout好坏？”

并没有一个框架内部公认的确定答案。

---

# 12. Hexcute也给出了同类、但不同的信息假设

Hexcute当前cost model非常明确：

\[
Cost=
\#Instructions\times CPI.
\]

同时使用independent/dependent CPI，并显式近似copy/MMA overlap。fileciteturn410file0L1-L7

但源码自己列出了假设：

- copy/MMA pipeline充分overlap；
- bank conflict penalty暂未计入；
- `cp_async_wait_group` / `mbarrier`未计入；
- address arithmetic / tensor manipulation被认为可忽略；
- arithmetic latency甚至没有区分Ampere/Hopper，因为microbenchmark观察认为差别很小；
- 明确说明该模型专门用于layout synthesis ranking，不能保证普通Hidet程序预测准确。

fileciteturn410file0L1-L7

这里也不能说：

> Hexcute模型错了。

但是可以非常严格地说：

\[
\boxed{
Hexcute认为“这些局部信息足够用于layout candidate differentiation”。
}
\]

这就是后面科研问题要验证的assumption。

---

# 13. 到这一轮，框架统计中还有哪些明显缺口？

现在缺口已经比之前少很多。

我会这样判断：

| Framework | 还缺什么 |
|---|---|
| CUTLASS | SM100 sparse/block-scaled/mixed-input + epilogue的完整Auto矩阵 |
| Triton | `OptimizeTMemLayouts`已大幅补齐；仍需把PartitionScheduling所有rule逐条归档，Coalesce全部branch，TMEM全pattern整理成正式表 |
| Gluon | AutoLayout inference基本补齐；还需Blackwell-specific layout helpers反查 |
| TVM | 默认CUDA/TensorCore rules基本清楚；还要将每个Postproc的layout影响细分 |
| TIRx | family inventory已完整；每个`copy_async/TMA/tcgen05/reduction` predicate仍需逐条入表 |
| Hexcute | 核心机制基本完整；instruction registry逐instruction表尚未完成 |
| TileLang | ranking基本完整；candidate-root生成/strict-common-free的全部branch仍需整理 |
| vLLM | 需要把backend × stride × content packing × block size正式形成最终矩阵 |
| SGLang | model override目录需要全部按“attention/page/KV/state/MoE”等字段分类 |
| FlashInfer | FA2/decode/MLA/Blackwell TS需要分开做完整scheduler rule表 |

因此：

\[
\boxed{\text{实现统计仍未完全封版。}}
\]

但几条科研证据链现在已经不依赖这些剩余冷门分支。

---

# 14. 现在重新审判证据：哪些假设已经能确认“不充分”？

这一部分要非常严格。

| 假设/策略 | 能否说“不充分” | 为什么 |
|---|---|---|
| 一个稳定KV representation适合所有consumer/boundary | **能，作为普遍命题不成立** | vLLM自己存在layout+block conversion |
| 通用model/hardware rule足够选择backend/page | **能，作为普遍命题不成立** | SGLang大量model-family overrides |
| layout可与warp count/register pressure完全独立 | **能，作为普遍命题不成立** | Triton改变partition warps后重新layout |
| 一个layout满足所有kernel-local consumer偏好 | **能，作为普遍命题不成立** | Triton memory/MMA/reduction/TMEM preference不同并插conversion |
| legality constraint足以确定性能最优layout | **不能直接说错误** | Hexcute还需要cost model，但没有证明其他合法layout更快 |
| bank-conflict最小layout就是global最优 | **不能** | Triton只是以此作为局部proxy，性能需实验 |
| register-count最小layout最好 | **不能** | TileLang默认使用，但同时有IO-aware替代；需实验 |
| TVM的SSSRRSRS空间遗漏最优解 | **不能** | 搜索空间受限是事实，漏掉重要optimum需实验 |
| TIRx priority first-match不是最优 | **不能** | priority策略是事实；性能次优需实验 |
| FlashInfer occupancy rule不是最优 | **不能** | rule存在，但没有oracle比较 |

这是以后表述上必须坚持的边界。

---

# 15. “假设错误”和“不是最优”还应该再分开

这是前面回答里容易混淆的地方。

如果一个框架自己已经有：

\[
Strategy_A
\]

但是某种条件必须切到：

\[
Strategy_B,
\]

那么我们可以说：

\[
\boxed{
A\text{ 不是 universally sufficient。}
}
\]

例如SGLang generic backend policy + model overrides。

但是：

\[
Strategy_B
\]

是否比A“最优”，源码并不能证明。

同理Triton写：

> `Extremely rough estimate`

只能证明：

\[
\boxed{
\text{selector使用粗略proxy。}
}
\]

并不能证明：

\[
\boxed{
\text{selector产生了性能损失。}
}
\]

性能损失必须通过后续oracle/exhaustive/measurement验证。

---

# 16. 现在按“证据数量”排序，第一条最值得凝练的问题其实变了

此前我更偏向representation stability。

但把这轮的规则全部补进来以后，**覆盖框架最广的共同现象实际上是：所有框架都在用某种不完整的性能代理来压缩layout决策。**

它几乎覆盖全部十个系统。

具体是：

| Framework | 实际性能代理 |
|---|---|
| CUTLASS | instruction/tile legality、largest compatible swizzle、SMEM capacity、1 CTA/SM StageCountAuto |
| Triton | coalescing、bank conflicts、cross-thread communication、rough register model、fixed thresholds |
| Gluon | explicit expert + coalescing AxisInfo；AutoLayout用inference distance |
| TVM | structural ScheduleRules + XGB +真实measurement |
| TIRx | predicate + hand-written priority |
| Hexcute | instruction count × CPI + overlap model |
| TileLang | register-count / IO-aware symbolic model |
| vLLM | compatibility intersection + first-choice preference |
| SGLang | model/platform/backend rules + overrides |
| FlashInfer | occupancy + workload geometry + resource formulas |

这是目前证据最多的一条。

---

# 17. 科研问题 1：什么信息才足够判断一个 Layout 好不好？

不能写成：

> “现有heuristic不准。”

目前没有证据。

更准确的科研问题是：

\[
\boxed{
\textbf{RQ1:
What information is necessary and sufficient to robustly rank GPU layout configurations for modern LLM workloads?}
}
\]

中文可以更直接：

> **现代GPU layout选择究竟需要多少性能信息？仅依赖结构合法性、coalescing、bank conflict、register footprint、instruction latency、occupancy等局部代理，什么时候已经足以保持正确的候选排序；什么时候必须引入真实硬件measurement或更全局的资源/流水线信息？**

这是真正的trade-off：

\[
\text{cheap structural/analytic model}
\]

vs

\[
\text{expensive empirical feedback}.
\]

为什么它现在证据很强？

因为现有框架已经实际选择了整条谱系：

\[
\text{TIRx priority}
\rightarrow
\text{CUTLASS rules}
\rightarrow
\text{Hexcute analytic}
\rightarrow
\text{TileLang symbolic}
\rightarrow
\text{Triton autotune}
\rightarrow
\text{TVM measurement}.
\]

而且没有统一答案。

这个问题的关键评价指标以后也不是绝对时间预测误差，而应该是：

\[
\boxed{
RankingAccuracy,\ TopKRecall,\ OracleRegret
}
\]

尤其：

\[
T(A)<T(B)
\Rightarrow
\hat T(A)<\hat T(B)?
\]

这个问题当前证据等级：

\[
\boxed{E1:10/10,\quad E2:多框架,\quad E3:10/10,\quad E4:强}
\]

因此目前排第一。

---

# 18. 但 RQ1 还可以进一步变得更 layout-specific

否则容易变成泛化的“GPU cost modeling”。

真正需要研究的是：

\[
\boxed{
\text{layout decision所需要的信息是否局部可分？}
}
\]

例如配置：

\[
c=(L_S,L_R,Mapping,MMA,Pipeline)
\]

是否可以用：

\[
Score(c)
=
w_1BankConflict
+w_2Registers
+w_3Coalescing
+w_4InstructionCount
\]

正确排序？

还是必须知道：

\[
Occupancy,
AsyncOverlap,
WarpRole,
Conversion,
Consumer,
RuntimeShape
\]

才能保持ranking？

这样才是你的layout项目，而不是普通cost model。

---

# 19. 按证据强度，第二条：Layout 相关决策到底能不能分开做？

本轮Triton的源码使这个证据链明显变强。

一个非常关键的事实是：

Triton改变warp partition数后：

\[
\boxed{\text{重新执行layout assignment}}
\]

。fileciteturn360file0L1-L7

TMEM layout也根据downstream reduction/local-load/store修改。fileciteturn348file0L2-L2

这已经直接证明：

\[
L
\not\perp
WarpCount
\]

以及：

\[
L
\not\perp
Consumer.
\]

CUTLASS则直接把tile/copy/SMEM/MMA/stage/schedule共同构造。fileciteturn337file0L1-L24

Hexcute选择：

\[
Layout+TaskMapping+Instruction
\]

一起infer，但把pipeline留给programmer。

TileLang选择：

\[
Pipeline\rightarrow LayoutInference.
\]

TIRx选择：

\[
StorageLayout\rightarrow LocalPrimitiveDispatch.
\]

TVM选择：

\[
ScheduleRuleFamily\rightarrow Search.
\]

这几乎就是五种不同的decomposition。

---

# 20. 科研问题 2：哪些 Layout 决策真正需要联合优化？

现在可以更精确地凝练：

\[
\boxed{
\textbf{RQ2:
Which layout-related decisions exhibit strong performance interactions and therefore require joint optimization, and which can be safely modularized?}
}
\]

变量可以明确限定为：

\[
D=
\{
L_G,L_S,L_R,L_T,
ThreadMapping,
CopyInst,
MMAInst,
Pipeline,
WarpRole
\}.
\]

真正要研究的是：

\[
\boxed{
InteractionGraph\ G=(D,E)
}
\]

而不是简单证明“联合搜索更好”。

一个edge存在的实验定义应该是**rank reversal**：

\[
T(A_1,B_1)<T(A_1,B_2)
\]

但：

\[
T(A_2,B_1)>T(A_2,B_2).
\]

如果没有rank reversal：

\[
A,B
\]

可以安全分阶段。

如果频繁发生，则需要joint optimization。

这使问题变得可以证伪。

---

# 21. 第三条：Representation Stability vs Specialization 证据仍然很强

它的证据比RQ2更偏serving，但非常贴KV layout。

现有系统至少出现四种现实策略：

\[
\text{vLLM}
:
\text{whole-model stable compatible representation}
\]

\[
\text{SGLang}
:
\text{backend/model-specific representation}
\]

\[
\text{Hexcute}
:
\text{multi-consumer constraint compromise}
\]

\[
\text{Triton}
:
\text{consumer-specific layout + conversion}
\]

\[
\text{TileLang}
:
\text{conflict时merge/widen/replicate}.
\]

而vLLM又自己存在persistent receive-side conversion。

这正说明不存在简单答案。

---

# 22. 科研问题 3：一个 Tensor 的 Representation 到底应该多稳定？

现在建议把以前两个问题合并：

> specialization granularity

和：

> compromise/convert/replicate

变成一个更完整的问题：

\[
\boxed{
\textbf{RQ3:
How should a multi-consumer tensor trade representation stability against specialization, conversion, and replication?}
}
\]

中文：

> **一个被多个异构consumer长期或短期复用的tensor，应该维护一个共享的折中layout，还是在consumer边界转换layout，或者同时复制多种representation？最优策略如何随tensor lifetime、reuse、conversion cost和memory pressure变化？**

这非常适合分两个尺度研究：

kernel-local：

\[
QK^T
\rightarrow Softmax
\rightarrow PV
\]

system-level：

\[
KV
\rightarrow
\{
Prefill,\ Decode,\ Transfer,\ QuantizedConsumer
\}.
\]

这也是目前最直接和“KV layout”题目契合的问题。

---

# 23. 第四条暂时还是候选：规则几何 vs 不规则现代模型

它的源码证据在变强。

TIRx global/shared自动copy要求：

\[
N_{\rm elements}\%N_{\rm threads}=0
\]

且extent要常量。fileciteturn409file0L1-L7

TVM的candidate space大量来自离散tile/thread/vector集合。fileciteturn378file0L1-L7

FlashInfer的GQA group、CTA Q、MMA count、head dim也都是有限离散families。fileciteturn394file0L2-L2

SGLang实际模型则不断需要不同page sizes：

\[
1,\ 16,\ 64,\ 128
\]

等特殊路径。fileciteturn371file0L1-L7 fileciteturn373file0L1-L7

这里已经可以形成候选：

\[
\textbf{regular hardware mapping}
\quad vs\quad
\textbf{irregular workload fidelity}.
\]

但目前我仍然不会把它提升到前三。

因为规则tiling与irregularity本身是经典问题；还需要后续证明 GQA/MLA/MoE/paged-KV 产生了不同于传统ragged computation的新layout规律。

---

# 24. 还有一个此前问题应该降级：Storage-only vs execution-rich layout abstraction

CuTe、TIRx、Hexcute之间确实存在：

\[
Storage-only
\leftrightarrow
Storage+TaskMapping+Instruction
\]

的抽象差异。

但经过这几轮源码审查，我认为它暂时更适合作为RQ2的**机制变量**。

因为真正有性能意义的未知量不是：

> “layout abstraction应该丰富还是简单？”

而是：

\[
\boxed{
\text{那些被 abstraction 隐藏/解耦的 execution freedoms 是否产生显著 rank reversal？}
}
\]

如果没有，就应该保持storage-only。

如果有，才值得把更多execution semantics纳入layout。

所以暂不单列主问题。

---

# 25. 当前的“证据→问题”矩阵

这是我认为现在最有价值的一张表。

| 源码事实 | 框架证据 | 能证明什么 | 不能证明什么 | 对应RQ |
|---|---|---|---|---|
| 多种局部proxy决定layout | CUTLASS/Triton/Gluon/TIRx/Hexcute/TileLang | 没有统一ranking信息集 | proxy一定性能差 | RQ1 |
| hardware measurement仅部分系统使用 | TVM/Triton autotune vs其他系统 | measurement cost vs fidelity确实存在 | measurement一定值得 | RQ1 |
| warp数变化必须重新layout | Triton | layout↔warps确有coupling | 必须全量joint search | RQ2 |
| consumer改变TMEM layout | Triton | layout↔consumer确有coupling | 所有consumer都应specialize | RQ2/RQ3 |
| collective共同构造 layout/copy/MMA/pipeline | CUTLASS | 强耦合设计是现实可行路径 | 它是最优decomposition | RQ2 |
| pipeline固定后推layout | Hexcute/TileLang | 另一种decomposition可行 | pipeline-first最好 | RQ2 |
| whole-model KV layout | vLLM | 稳定representation是实际策略 | 这是错误策略 | RQ3 |
| persistent receive conversion | vLLM | stable layout不普适到所有boundary | 动态layout总是值得 | RQ3 |
| model-specific backend/page override | SGLang | generic policy不普适 | override一定达到oracle | RQ1/RQ3 |
| conflict→replication/widen | TileLang | replication是实际协调策略 | replication更优 | RQ3 |
| multi-consumer constraints unify | Hexcute | compromise layout是实际策略 | compromise更优 | RQ3 |
| fixed discrete regular families | TVM/TIRx/FlashInfer/CUTLASS | structured regularity是普遍压缩方式 | irregular layout一定更快 | 候选RQ4 |

---

# 26. 哪些“问题证据”目前最强？

如果严格按**源码支持的框架数量 + 同框架自我修正证据 + 现代LLM相关性**排序，我现在会排：

\[
\boxed{
1.\ Ranking\ Information\ Sufficiency
}
\]

即 RQ1。

\[
\boxed{
2.\ Selective\ Joint\ Optimization
}
\]

即 RQ2。

\[
\boxed{
3.\ Representation\ Stability/Specialization
}
\]

即 RQ3。

第四名才是：

\[
Regularity\ vs\ Irregularity.
\]

这比几轮前的排序更可靠，因为现在排序来自实际rule catalog，而不是概念上的猜测。

---

# 27. 但是“优化策略不够好、还不是最优”的证据，目前要怎样表述？

需要非常谨慎。

目前已经可以确定“不是全局最优求解”的包括：

- Triton partitioning：启发式merge + fixed thresholds；
- Triton warp count：源码称extremely rough estimate；
- Gluon auto-layout：minimum inference distance；
- TIRx：priority-first dispatch；
- CUTLASS：architecture rule-based Auto；
- Hexcute：analytic approximate cost；
- TileLang：register-count或IO-aware proxy；
- vLLM：first-preference voting；
- SGLang：handwritten/model override；
- FlashInfer：occupancy/resource analytic scheduling。

但：

\[
\boxed{
\text{不是全局最优求解}
\neq
\text{实际性能显著次优}.
}
\]

后者仍然必须通过后续实验找：

\[
c^*=
\arg\min_cT(c)
\]

然后测：

\[
Regret=
\frac{T(c_{\rm framework})}
{T(c^*)}-1.
\]

所以目前可以说：

> **现有策略没有最优性保证，且依赖显式proxy/heuristic。**

不能说：

> **现有策略已经被证明性能不好。**

这是下一阶段实验必须回答的地方。

---

# 28. 当前证据链是否“整理完毕”？

答案是：**前三个问题的核心证据链已经基本整理成型；整个十框架rule catalog还没有100%封口。**

具体：

RQ1 已经有十框架策略证据，足够进入深度审判。

RQ2 已经有 CUTLASS、Triton、Hexcute、TileLang、TIRx、TVM/Gluon多种decomposition证据，而且Triton提供了非常强的内部coupling证据。

RQ3 已经有 vLLM、SGLang、Triton、Hexcute、TileLang的相反策略和repair证据。

因此接下来最合适的方式不是继续一次性列更多科研问题，而是**从RQ1开始做一条完整的“问题审判”**：

\[
\text{所有相关框架源码规则}
\rightarrow
\text{每个规则使用的proxy}
\rightarrow
\text{它忽略了哪些变量}
\rightarrow
\text{框架自己的fallback/repair}
\rightarrow
\text{现代Attention/GQA/MLA/MoE触发条件}
\rightarrow
\text{可证伪的科研假设}.
\]

RQ1彻底完成之后再审判RQ2，然后RQ3。这样可以防止又回到前几轮“问题列了很多，但证据深度不够”的状态。

## 用户

请继续审查 检查哪里没有整理并补充。这里的证据链是否也整理完毕：哪些已有的实现点/layout优化方法的细节确认哪些确实因为假设不充分，假设错误，有什么模型框架支持它。哪些优化策略不够好，还不是最优的：有什么模型框架支持它。这些支持的作为问题的主要证据，从证据最多开始的逐个凝练科研问题；

## 模型回复

我继续按“规则目录补漏 → 假设审判 → 次优性证据 → 科研问题排序”推进。重点会把剩余几类高风险漏项补齐：CUTLASS Auto、TIRx dispatch、vLLM persistent-spec、SGLang override、FlashInfer scheduler，并把“源码能证明不充分”与“只能证明无最优性保证”严格分开。

## 模型工具调用参数

{"paths":["GitHub"],"query":"search"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

Auditing Kernel Mapping Evidence and Hardware Assumptions

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 CUTLASS 指定分支文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

在 CUTLASS 中检索自动调度与共享内存选择器

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 TIRx 调度器优先级实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 CUDA TIR 注册优先级配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审核 vLLM KV Cache 布局证据链

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 KV 缓存布局解析器实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索旧文件中的精确短语

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 vLLM 注意力后端工具代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 SGLang 当前配置覆盖类别

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 SGLang 中页面大小与注意力后端配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

评估 SGLang 模型覆盖配置声明

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 FlashInfer 工具函数实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 FA2 CTA Tile Q 选择逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检视 TileLang 成本模型候选生成逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索布局推理与成本模型实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索Hexcute布局搜索与成本模型

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Beam 相关实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索TVM MetaSchedule与结构先验证据

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 base_worker 源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 KV 接收后处理布局实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 vLLM KV 传输工具代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 CUTLASS SM103 调度实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

评估 TileLang 布局系统与硬件映射 extenstion

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索文件中的精确短语

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取布局推断实现代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索布局推断中的候选生成逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

评估注意力优化策略的模型框架支持

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索精确匹配内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索任意大小连接边实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索布局重排与Warp变更代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 MetaSchedule 与 Gluon 布局哈希冲突

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 SGLang 后端性能回归与默认配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

评估CUTLASS自动调度启发式

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计 TVM MetaSchedule 成本模型与测量流程

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Hexcute 成本模型与布局综合限制

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

评估大模型推理内存布局策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 仓库中的 TMem 实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检查 Triton TMem 布局优化实现

## 模型思考摘要

凝练研究问题
凝练了研究问题

## 模型思考摘要

审计解析了主流框架布局推理、调度启发式与成本模型证据

## 模型回复

现在可以先给一个阶段性结论：**“证据最多”的第一问题已经相当稳定，但“现有方法已经性能次优”仍不能仅靠源码下结论。**源码目前足以证明的是两件事：大量系统使用显式 proxy/heuristic 而无全局最优性保证；并且多个系统自己的 override、relayout、conversion、widen/fallback 已经直接否定了若干“决策可独立/单一 representation 普适”的强假设。下面我按这个标准收口。

## 模型思考摘要

收束研究问题

## 模型思考用时

思考了 4m 18s

## 模型回复

继续审查以后，我认为现在可以把“框架实现事实”和“科研问题”之间的中间层真正补起来了。前几轮最大的问题，是容易从“某框架用了 heuristic”直接跳到“这是科研问题”。现在证据应该经过四级审判：

\[
\boxed{
\text{Implementation}
\rightarrow
\text{Assumption / Proxy}
\rightarrow
\text{Counter-evidence / Repair}
\rightarrow
\text{Research Question}
}
\]

而且必须严格区分两件事：

\[
\boxed{\text{假设不普适}}
\neq
\boxed{\text{当前策略已经被证明性能次优}}
\]

前者有些已经可以由源码直接确认；后者绝大多数还需要我们下一阶段做 oracle/measurement 才能成立。

---

# 一、先更新审查结论：证据链还没有 100% 整理完，但主干已经形成

目前十个系统可以分成三种状态。

| 系统 | 规则/机制目录状态 | 当前最重要的补充 |
|---|---|---|
| CUTLASS/CuTe | 约 85% | SM100/103 Auto schedule、stage、TMEM/epilogue 的具体 heuristic 还需继续穷举 |
| Triton | 约 95% | partition、warp、TMEM、coalesce、layout conversion 主干已基本完整 |
| Gluon | 约 90% | AutoEncoding/CoalescedEncoding 已搞清，剩 Blackwell-specific helper |
| TVM MetaSchedule | 约 90% | candidate generation/default CUDA rules 已明确，个别 Postproc 影响还可继续拆 |
| TIRx | 约 90% | dispatcher 总机制完整；每种 primitive 的全部 predicate 尚未逐项成表 |
| Hexcute | 约 90% | constraint/instruction/cost model 主干完整 |
| TileLang | 约 90% | strict/common/free、conflict repair、ranking 已明确，少量 candidate-root branch 待补 |
| vLLM | 约 95% | persistent KV layout/spec/block/content/transfer 已形成完整主线 |
| SGLang | 约 90% | generic policy + model override 已明确；model-family 全部字段尚未机械归档 |
| FlashInfer | 约 85% | FA2/decode/split-KV 主干完整；Blackwell/CuTeDSL scheduler 还需单独穷举 |

所以回答“证据链是否整理完毕”：

\[
\boxed{\text{主问题的证据链已经基本成型；全框架规则目录尚未完全封版。}}
\]

不过现在已经足够判断哪些问题值得继续，哪些问题应该降级。

---

# 二、这一轮最重要的新发现：我们已经能证明若干“强假设”确实不成立

这里不是推测，而是框架源码自己已经提供 counterexample / repair。

## 1. “Layout 可以独立于 warp 数量决定”——已经可以否定

Triton 的 `OptimizePartitionWarps` 明确写：

> `Extremely rough estimate of the number of registers needed per partition.`

它用 partition 最大 tensor 估计寄存器，然后调整 warp 数量。fileciteturn446file0L1-L35

更关键的是，一旦 warp 数变化：

> `Layouts need to be reassigned if the number of warps changed`

随后重新执行 layout assignment。fileciteturn448file0L1-L46

因此：

\[
\boxed{
L_{\text{register/thread}}
\not\perp
N_{\text{warps}}
}
\]

这个不是科研假说了，而是**实现事实**。

它直接说明：

> 如果先固定 layout，再独立优化 warp count，原则上可能失去原先的 layout optimality。

但这里还不能直接说：

> 一定需要全局 joint search。

可能只需要局部联合优化。

这个区别非常重要。

---

# 三、“一个默认 TMEM layout 对所有 consumer 都足够好”也已经被 Triton 自己否定

这是这一轮非常强的一条证据。

Triton `OptimizeTMemLayouts.cpp` 明确根据 downstream consumer 改变 TMEM layout。

默认多个 warpgroup 时 TMEM load 更倾向沿 N distribute，但如果后面存在：

\[
Reduction_N
\]

则改成沿 M distribute，从而避免 cross-warp reduction。fileciteturn456file0L2-L2

另一个 pattern 明确写：

> `layout 16x256b allows better code generation for local_load lowering`

所以如果发现 shared→TMEM 路径中的 local load 无法有效 vectorize，就切换 layout。fileciteturn456file0L2-L2

反方向 TMEM→shared 也有类似优化。

因此已经可以确认：

\[
\boxed{
L^*_{\text{TMEM}}
=
f(\text{consumer})
}
\]

而不是：

\[
L^*_{\text{TMEM}}
=
f(\text{tensor shape only})
\]

。

这是一条非常强的“假设不充分”证据。

而且它不是外部框架反驳 Triton。

是：

\[
\boxed{\text{Triton 自己在后续 pass 修正自己早期的 layout。}}
\]

这种证据比“两个框架策略不同”强得多。

---

# 四、“局部 heuristic 足以决定 warp specialization”也只能说是近似，不是最优求解

Triton `PartitionScheduling` 的实现方式现在已经很清楚。

它首先建立 dataflow graph，再给 data operations 分 partition，然后用 heuristic 不断合并直到 fixed point。

其中有一条：

\[
edge.size>16384
\]

就倾向merge。

源码旁边直接写：

> `FIXME: seemingly arbitrary size...`

fileciteturn447file0L1-L30

这意味着我们可以确定：

\[
\boxed{
\text{Triton partition decision使用人为近似阈值，而没有全局性能最优性保证。}
}
\]

但这里仍然必须保持科学严谨：

我们目前能说：

\[
\boxed{\text{Approximate heuristic}}
\]

不能说：

\[
\boxed{\text{Measured suboptimal}}
\]

因为后者还没有 oracle comparison。

---

# 五、CUTLASS Auto 也必须重新定位：它不是自动搜索，而是 expert-rule construction

这轮 SM100/SM103 源码进一步确认。

SM100 `StageCountAutoCarveout` 基本是：

\[
Stages
=
\left\lfloor
\frac{SMEM_{\rm capacity}-carveout}
{Bytes_A+Bytes_B+PipelineStorage}
\right\rfloor
\]

。fileciteturn418file0L1-L2

也就是说，它首先回答的是：

> 能放多少 stage？

不是：

\[
\arg\min_{stages}T(stages).
\]

更有意思的是 accumulator pipeline。

源码明确写：

> `4 accumulator stages works well ...`

然后：

\[
AccumulatorStages=
\min(4,CapacityDerivedStages).
\]

fileciteturn418file0L1-L2

SM103 block-scaled Auto schedule 更清楚：

如果：

\[
ClusterShape_M \bmod 2=0
\]

Auto直接选 2SM MMA，否则选1SM。fileciteturn440file0L1-L2

所以 CUTLASS Auto 的本质应该修正为：

\[
\boxed{
\text{hardware legality/resource constraints}
+
\text{expert structural heuristics}
}
\]

不是performance search。

因此它是 RQ1/RQ2 的重要证据，但不能拿来证明策略真的差。

---

# 六、Gluon 的 AutoLayout 现在也可以正式判定为“constraint propagation”，不是 performance optimizer

Gluon 的 layout conflict resolution 依据 inference distance。

如果两个推导：

\[
d_1<d_2
\]

选：

\[
L_1.
\]

当距离一样但不是直接seed时，它用确定性 ordering 解决；直接 seed 冲突则报错。之前审查到的 `InferLayoutUtils.cpp` 已经明确这一机制。fileciteturn387file0L1-L7

这意味着：

\[
\boxed{
\text{AutoLayout解决的是“一致地传播什么layout”}
}
\]

而不是：

\[
\boxed{
\text{哪个layout性能最好}
}
\]

。

这个 distinction 前面还没有完全强调。

所以不能把：

> Gluon AutoLayout

与：

> TVM MetaSchedule

归为同一种“自动优化”。

完全不是。

---

# 七、TVM MetaSchedule：measurement 很强，但它仍然没有解决 candidate-space completeness

TVM 是这里最重要的 counterexample。

Default CUDA schedule rule 已经提前固定大量 structural prior：

\[
SSSRRSRS
\]

以及固定 thread binding、vector-load candidate、reuse level、CrossThreadReduction extent、AutoBind 候选等。fileciteturn378file0L1-L7

之后才进入：

\[
SpaceGenerator
\rightarrow SearchStrategy
\rightarrow CostModel
\rightarrow Builder/Runner.
\]

Builder/Runner 确实会在真实硬件上运行 candidate 并获得 measured runtime。fileciteturn452file0L1-L20

因此：

\[
\boxed{
\text{measurement解决的是}
\quad
\min_{x\in G}T(x)
}
\]

而不是：

\[
\boxed{
\min_{x\in S}T(x)
}
\]

其中：

\[
G\subseteq S.
\]

如果最优candidate压根没有被 ScheduleRule 生成：

\[
x^*\notin G
\]

再好的 cost model 和真实 measurement 都没用。

所以这为另一个科研问题提供了非常干净的证据：

\[
\boxed{
\text{candidate generation error}
\neq
\text{candidate ranking error}
}
\]

这个区分应该正式进入我们的研究框架。

---

# 八、TIRx 的证据更直接：它明确采用“最高优先级第一个合法实现”

TIRx dispatcher 当前逻辑已经完全确认。

每个 variant 包含：

\[
(\text{priority}, predicates, implementation)
\]

执行时：

\[
\boxed{
\text{按 priority 降序排序}
\rightarrow
\text{predicate检查}
\rightarrow
\text{第一个成功 implementation 直接返回}
}
\]

。fileciteturn420file0L1-L10

它不会：

\[
Benchmark(\text{all legal variants})
\]

也不会：

\[
CostModel(\text{all legal variants}).
\]

这非常重要，因为它给出了纯粹的：

\[
\boxed{
Legality
+
Priority
}
\]

selector。

同步copy里，显式fixed-width variant甚至 priority=20，而 `vec_auto` 是 priority=10。fileciteturn407file0L1-L7 fileciteturn408file0L1-L7

所以：

\[
\boxed{
\text{TIRx中的高priority代表designer preference，不等于empirical optimum。}
}
\]

这可以作为 RQ1 的非常干净 baseline。

---

# 九、TIRx 还暴露了 candidate-space 的一个强结构假设

global↔shared `vec_auto` 要求：

\[
N_{\text{elements}}\bmod N_{\text{threads}}=0.
\]

dynamic extent 也会被拒绝。

源码明确解释：

> 对这种 badly shaped copy 不想用 slow scalar path 去“paper over”。

fileciteturn409file0L1-L7

所以这里可以确定：

\[
\boxed{
\text{TIRx自动vectorized partition的candidate generator偏好规则可整除结构。}
}
\]

但是仍然不能说：

> 非规则partition一定更快。

真正科学的问题应该是：

> 这种结构约束压掉了多少搜索空间，又会损失多少near-optimal coverage？

这属于 Coverage 问题，而不是“TIRx bug”。

---

# 十、vLLM：Persistent layout 证据链现在已经相当完整

当前逻辑shape统一为：

\[
[L,B,H,N,C].
\]

物理layout至少包括：

\[
LBHNC,LBNHC,LHBNC,BLHNC,BLNHC,BHLNC.
\]

fileciteturn422file0L1-L13

resolver明确写：

> `Resolve one KV cache layout for the whole model.`

fileciteturn425file0L1-L2

backend支持集先求intersection，然后如果 preference 不同，则统计各backend“第一个layout”的票数，再排序；connector preference可覆盖，但必须仍然合法。fileciteturn425file0L1-L2

因此它解决的是：

\[
\boxed{
\text{compatibility negotiation}
+
\text{preference aggregation}
}
\]

不是：

\[
\arg\min_L
T_{\rm whole-system}(L).
\]

---

# 十一、而 vLLM 自己已经直接证明“一种 persistent representation 对所有边界都够用”不成立

这条现在非常强。

vLLM connector source 明确写：

> disaggregated P/D + NIXL 下使用 LBHNC，因为 transfer 更快。

fileciteturn439file0L1-L2

同时 receive-side 存在：

`kv_postprocess_layout_on_receive`

把：

\[
[B,H,N,C]
\]

转换到：

\[
[B,N,H,C].
\]

还存在联合：

\[
\boxed{
block\ size + layout
}
\]

转换：

> prefill 是 LBHNC + smaller block size  
> decode(local) 是 LBNHC + larger block size

fileciteturn439file0L1-L2

这是目前 RQ3 最强的证据之一。

因为它直接证明：

\[
\boxed{
L_{\rm transfer}^{*}
\neq
L_{\rm local\ decode}^{*}
}
\]

至少在当前工程设计的某些路径里，这个冲突已经真实存在。

注意我仍然没有说两者是各自“数学上的最优”。

但：

\[
\boxed{
\text{consumer/boundary preference不同}
}
\]

已经是实现事实。

---

# 十二、vLLM 还说明“KV layout”不能再只讨论 HND/NHD

backend `customize_spec()` 允许kernel改变 KV physical content packing。

目前已确认的设计包括：

ROCm native：

\[
K/V \text{ two head groups}
\]

AITER unified：

\[
K/V \text{ packed in content dimension}
\]

Triton per-head quant：

\[
Data + FP32Scale
\]

TurboQuant：

\[
K+V\text{ packed into one head slot}
\]

FlashInfer NVFP4：

\[
PackedFP4 + FP8BlockScales.
\]

相关backend源码都明确使用 `customize_spec()`。fileciteturn413file0L1-L24 fileciteturn413file1L26-L48 fileciteturn413file4L107-L117 fileciteturn413file5L132-L143 fileciteturn413file7L184-L195

所以后面的科研定义应该使用：

\[
R_{KV}
=
(
StrideOrder,
ContentPacking,
PageGeometry,
QuantMetadata,
TransferRepresentation
)
\]

而不是简单：

\[
Layout\in\{HND,NHD\}.
\]

这是我们前面目录里一个已经修正完成的重要缺口。

---

# 十三、SGLang：generic rule 不充分，已经是非常明确的源码事实

SGLang 的 generic backend chooser 依赖：

\[
Architecture,
Hardware,
SpecDecode,
PageSize,
AsymmetricKV,
AttentionSink,
BackendAvailability,\ldots
\]

例如：

Hopper MHA → FA3；

SM100 compatible MHA → TRTLLM MHA；

asymmetric K/V → FA4；

HIP → AITER；

否则 FlashInfer/Triton 等。fileciteturn428file0L1-L2

但随后又存在大量 model-family overrides。

DeepSeek DSA已经出现：

\[
PageSize=
1,\ 64,\ 128
\]

取决于平台和AITER path。fileciteturn371file0L1-L7

Qwen3.5 SM100 又直接把：

\[
Backend=TRTLLM\_MHA
\Longleftrightarrow
PageSize=64
\]

与：

\[
Backend=Triton
\Longleftrightarrow
PageSize=1
\]

联动。fileciteturn372file0L1-L7

Qwen3-VL ROCm AITER unified 又需要：

\[
PageSize=16.
\]

fileciteturn373file0L1-L7

因此可以直接确认：

\[
\boxed{
Backend
\not\perp
PersistentPageGeometry
}
\]

以及：

\[
\boxed{
GenericPolicy(ModelClass,Hardware)
}
\]

不足以覆盖全部现代模型。

这是**强 A 类证据**。

---

# 十四、SGLang 甚至提供了少数“实际性能变差”的直接证据

generic chooser当前源码写明：

> FlashInfer 0.6.1 在 Hopper attention kernel 上发生 performance regression，因此默认改选 FA3。

fileciteturn450file0L1-L25

这比单纯“heuristic没有理论保证”更强。

它说明：

\[
\boxed{
BackendRanking
}
\]

不只是：

\[
f(Model,Hardware)
\]

而且可能依赖：

\[
f(Model,Hardware,BackendVersion,KernelImplementation).
\]

但我不建议把这个version regression本身变成科研问题。

它只是 RQ1 的证据：

> ranking policy所需的信息可能比当前policy输入更多。

---

# 十五、FlashInfer：介于静态 heuristic 和 measurement optimizer 之间

FlashInfer值得单列，因为它不是简单hard-code。

其candidate families非常结构化，例如：

\[
CTA_Q\in\{16,32,64,128\}
\]

\[
NUM\_MMA_Q\in\{1,2\}
\]

\[
NUM\_MMA_{KV}\in\{1,2,4,8\}
\]

以及固定GQA/vector families。fileciteturn430file0L1-L2

FA2 `CTA_TILE_Q` 还会根据：

\[
avg\_packed\_qo\_len,\quad
QKDim,\quad
VODim,\quad
KVDType
\]

做rule-based选择；源码甚至专门处理大QK维导致的register/shared-memory约束。fileciteturn431file1L27-L58

但 split-KV 又实际调用：

\[
cudaOccupancyMaxActiveBlocksPerMultiprocessor
\]

估计当前kernel可填充多少grid，然后binary-search KV chunk size。fileciteturn397file0L2-L2

所以它属于：

\[
\boxed{
AnalyticGeometry
+
RuntimeWorkload
+
HardwareOccupancyQuery
}
\]

而不是 pure heuristic，也不是 exhaustive tuning。

这会成为 RQ1 中一个很重要的中间点。

---

# 十六、TileLang：这里出现了一条以前没有充分强调的“局部贪心失败修复”证据

这一轮重新检查 `layout_inference.cc` 后，这条值得提升。

它明确分：

\[
Strict
\rightarrow Common
\rightarrow Free
\]

不同 inference level。

而针对 reducer destination，代码专门引入 `ReducerDstSteering`。

源码注释非常明确：过去如果consumer先完成layout，会绕过finalize的planner decision，最后产生：

> `thread-indexed publish copy`

所以现在要让正确的owner先决定layout。fileciteturn443file0L1-L2

同时 reducer partial conflict 使用：

\[
\text{unset}
\rightarrow
\text{narrow}
\rightarrow
\text{wide}
\]

monotone widen lattice。

实在不行还有 universally readable replicated fallback。fileciteturn443file0L1-L2

这是一条很强的证据：

\[
\boxed{
\text{局部“谁先推导出来谁决定layout”并不充分。}
}
\]

因为它会产生后续publish/communication cost。

这直接支持 RQ2/RQ3。

---

# 十七、TileLang ranking 还进一步说明“最简单的proxy并没有唯一答案”

free-mode完整candidate assignment可以用两套不同排序模型。

默认：

\[
C_{\rm reg}
=
\text{fragment register slots}.
\]

IO-aware：

\[
C_{\rm io}
=
\sum_s
\max(
BWBytes(s),
IssueBytes(s)
)
\]

然后register count做tie break。fileciteturn375file0L1-L7

对于cost model无法理解的layout，IO-aware甚至采用conservative worst-case，以防“不可分析layout”错误获胜。fileciteturn374file0L2-L2

因此在同一个框架内部就存在：

\[
\boxed{
\text{Register pressure view}
\neq
\text{Memory traffic view}
}
\]

。

这为 RQ1 提供的是很高质量的证据。

---

# 十八、Hexcute：可以确定它的 ranking 假设是什么，但不能说假设已经失败

Hexcute：

\[
Cost=
\#Instructions\times InstructionLatency
\]

并建模copy/MMA overlap。fileciteturn410file0L1-L7

源码同时明确写：

bank conflict暂未进入cost model；

pipeline假设fully overlapped；

`cp_async_wait_group` / `mbarrier`未计；

address arithmetic忽略；

主要用于layout synthesis candidate differentiation，而非保证普通程序absolute prediction accuracy。fileciteturn410file0L1-L7

所以能得出的严谨结论是：

\[
\boxed{
Hexcute假设这些省略项不会破坏layout candidate ordering到不可接受的程度。
}
\]

而不是：

\[
\boxed{
Hexcute cost model错误。
}
\]

这个假设恰恰应该被我们实验检验。

---

# 十九、目前“确实假设不充分”的证据链，可以正式整理成这张表

| 原假设 | 当前判决 | 最强源码证据 | 支持框架 |
|---|---|---|---|
| Layout与warp count可独立优化 | **否，普遍命题不成立** | warp数改变后重新layout | Triton |
| 一个TMEM layout适合所有consumer | **否** | reduction/local-load/store触发layout替换 | Triton |
| generic backend/page policy覆盖现代模型 | **否** | model-family override | SGLang |
| backend与page geometry可以独立决定 | **否** | Qwen3.5 backend↔page联动 | SGLang |
| 单一persistent representation适合compute+transfer | **否** | NIXL transfer layout + receive转换 | vLLM |
| consumer-local first decision足以决定共享tensor layout | **否** | reducer dst steering修复first-completer问题 | TileLang |
| single consumer preference代表共享tensor最优 | **否，普遍命题不成立** | Triton consumer-aware TMEM；vLLM transfer/local冲突；Hexcute multi-consumer unify | Triton/vLLM/Hexcute/TileLang |
| legality能够唯一决定高性能实现 | **没有证据支持；但不能说已证伪** | TIRx合法variant仍按priority选择；Hexcute还需要cost model | TIRx/Hexcute |
| register最少就是layout最好 | **未证伪，但明显只是proxy** | TileLang同时提供IO-aware替代 | TileLang |
| bank conflict最少就是整体最快 | **未证伪，只是局部proxy** | Triton/Hexcute处理方式不同 | Triton/Hexcute |
| fixed schedule family包含近最优 | **未知，必须实验** | TVM先生成SSSRRSRS等有限空间再measurement | TVM |
| priority最高的合法primitive最优 | **未知，必须实验** | first-success dispatch | TIRx |

这张表基本就是我们之前一直缺的：

\[
\boxed{\text{Assumption Audit}}
\]

。

---

# 二十、再整理“哪些优化策略不够好，还不是最优”

这一句话需要改成更严谨的：

\[
\boxed{
\text{哪些策略明确只是近似/没有最优性保证？}
}
\]

这已经有大量证据。

| 框架 | 当前策略 | 源码可确认的问题 | 实际性能次优是否已证明 |
|---|---|---|---|
| Triton | fixed partition merge heuristics | 有arbitrary threshold | **未全面证明** |
| Triton | rough register proxy | 源码明确称extremely rough | **未全面证明** |
| Triton | consumer-aware repair | 早期layout不适合所有consumer | **局部已确认** |
| CUTLASS | StageCountAuto | 最大化容量可容纳stages | 未证明 |
| CUTLASS | Auto 1SM/2SM | cluster结构rule | 未证明 |
| CUTLASS | accumulator stages | “4 works well”经验rule | 未证明 |
| Gluon | minimum inference distance | 不使用性能ranking | 未证明 |
| TVM | fixed ScheduleRule space + measurement | 最优性只在generated space内 | coverage gap未知 |
| TIRx | priority first-success | 不比较所有合法candidate性能 | 未证明 |
| Hexcute | analytic latency model | 明确省略若干资源交互 | ranking fidelity未知 |
| TileLang | register-count / IO-aware | proxy选择本身存在trade-off | ranking fidelity未知 |
| vLLM | supported-set intersection + preference voting | compatibility不是performance objective | 未证明 |
| SGLang | handcrafted backend rules | 会被model/version-specific override修正 | **存在具体regression证据** |
| FlashInfer | analytic + occupancy scheduler | candidate仍是有限结构族 | oracle gap未知 |

所以目前最重要的结论其实是：

\[
\boxed{
\text{我们已经拥有大量“非最优求解机制”的证据，}
}
\]

但：

\[
\boxed{
\text{尚未拥有大量“性能regret显著”的实验证据。}
}
\]

这正好说明下一阶段为什么必须设计 oracle，而不是继续找更多 FIXME。

---

# 二十一、现在按“证据最多”重新排序科研问题

这次我不再单纯根据直觉排名。

按当前审查的十套系统计，我建议使用四个指标：

\[
E_{\rm systems}
\]

多少框架存在这个现象；

\[
E_{\rm direct}
\]

多少有源码明确heuristic/approximation；

\[
E_{\rm repair}
\]

多少框架自己已经存在override/repair/conversion；

\[
E_{\rm science}
\]

能否形成可证伪科学假设。

重新排序后，我认为是：

| 排名 | 科研问题 | 系统覆盖 | 直接源码证据 | 当前成熟度 |
|---:|---|---:|---|---|
| 1 | **Layout Ranking Information Sufficiency** | 10/10 | 极强 | 已可正式立题 |
| 2 | **Candidate-Space Coverage under Structural Constraints** | 10/10 广义；约7/10低层layout | 强 | 已可正式立题 |
| 3 | **Selective Joint Optimization / Boundary Separability** | 多框架 | 极强，Triton尤其强 | 已可正式立题 |
| 4 | **Representation Stability vs Specialization** | 5+直接 | 极强，vLLM/SGLang尤其强 | 已可作为KV主线 |
| 5 | Regularity vs Irregularity | 多框架 | 中等 | 继续观察，不宜先立题 |

这个排序比上一轮更稳定。

---

# 二十二、科研问题 1：Layout Ranking 到底需要什么信息？

这是目前证据最多的。

十个系统几乎分别选择了不同答案：

\[
\text{CUTLASS}: ExpertRules
\]

\[
\text{Triton}: LocalHeuristics+Autotune
\]

\[
\text{Gluon}: ConstraintDistance+Expert
\]

\[
\text{TVM}: LearnedModel+Measurement
\]

\[
\text{TIRx}: Priority
\]

\[
\text{Hexcute}: AnalyticLatency
\]

\[
\text{TileLang}: Register/IOProxy
\]

\[
\text{vLLM}: PreferenceVoting
\]

\[
\text{SGLang}: RuntimeRules
\]

\[
\text{FlashInfer}: AnalyticResource+Occupancy.
\]

所以科研问题现在可以凝练成：

\[
\boxed{
\textbf{RQ1:
What information is necessary and sufficient to reliably rank GPU layout configurations?}
}
\]

更严格的版本：

> **对于现代LLM kernel中的 layout、thread mapping、instruction 和 pipeline configuration，仅依赖 coalescing、bank conflict、register footprint、instruction latency、occupancy 等局部/解析信息，在哪些条件下可以保持候选的正确性能排序；在哪些条件下 asynchronous execution、resource contention、layout conversion 和 consumer context 会导致 ranking reversal，从而必须引入真实硬件反馈？**

注意这里不是追求：

\[
\hat T(c)=T(c)
\]

而应该追求：

\[
T(c_i)<T(c_j)
\Rightarrow
\hat T(c_i)<\hat T(c_j).
\]

真正指标应该是：

\[
Kendall\text{-}\tau,
\quad
TopKRecall,
\quad
SelectionRegret.
\]

这个问题目前证据最充分。

---

# 二十三、科研问题 2：结构约束究竟能安全地裁掉多少 Layout 搜索空间？

这个问题现在应该重新提升到第二。

因为所有系统都在做candidate compression，只是方法不同。

CUTLASS：

\[
ExpertTemplateFamily
\]

Triton：

\[
CompilerRewriteReachability
\]

TVM：

\[
ScheduleRuleSpace
\]

TIRx：

\[
RegisteredVariants
\]

Hexcute：

\[
ConstraintSynthesis
\]

TileLang：

\[
InferenceLevels+CandidateRoots
\]

FlashInfer：

\[
DiscreteKernelFamilies.
\]

定义完整合法空间：

\[
S
\]

框架生成空间：

\[
G_F\subseteq S.
\]

真正问题不是：

> 谁的搜索空间大？

而是：

\[
\boxed{
CoverageRegret(F)=
\frac{\min_{x\in G_F}T(x)}
{\min_{x\in S}T(x)}-1.
}
\]

科研问题：

\[
\boxed{
\textbf{RQ2:
How aggressively can layout search spaces be structurally constrained while preserving near-optimal coverage?}
}
\]

中文：

> **现代GPU高性能layout是否集中在一个具有强结构的低维子空间中？哪些hardware/instruction/thread-value constraints能够安全地删除绝大多数candidate而不损失near-optimal解，哪些看似合理的结构约束反而会系统性排除好解？**

这个问题的科学价值很高。

因为如果答案是：

\[
|G_F|\ll|S|
\]

同时：

\[
CoverageRegret\approx0
\]

那么约束推导路线是正确方向。

反过来，如果：

\[
CoverageRegret
\]

在现代LLM shapes频繁很大，则需要新的candidate generation。

---

# 二十四、科研问题 3：哪些 Layout 决策真的需要 Joint Optimization？

Triton现在给了这个问题非常强的直接证据。

已有：

\[
WarpCount\leftrightarrow Layout
\]

\[
Consumer\leftrightarrow TMEMLayout
\]

。

SGLang：

\[
Backend\leftrightarrow PageSize
\]

。

vLLM：

\[
Consumer/Transfer
\leftrightarrow
PersistentLayout
\]

。

CUTLASS：

\[
Tile+MMA+SmemLayout+Copy+Stage+Schedule
\]

共同构造。

所以问题不能再写成：

> “joint optimization是否更好？”

太泛。

应该是：

\[
\boxed{
\textbf{RQ3:
Which layout-related decision pairs exhibit strong performance interaction, and which are approximately separable?}
}
\]

定义：

\[
D=
\{
L_G,L_S,L_R,L_T,
WarpCount,
ThreadMapping,
CopyInst,
MMAInst,
Pipeline,
Consumer
\}.
\]

然后寻找真正的interaction。

最清楚的科学定义是 **rank reversal**。

如果：

\[
T(a_1,b_1)<T(a_1,b_2)
\]

但是：

\[
T(a_2,b_1)>T(a_2,b_2),
\]

则：

\[
A\not\perp B
\]

在性能排序意义上不能安全分离。

最终我们希望得到的不是：

\[
\text{full joint search everything}
\]

而是一个：

\[
\boxed{
\text{Sparse Interaction Graph}
}
\]

只联合优化真正耦合的决策。

这比简单“全局优化”强得多，也更可实现。

---

# 二十五、科研问题 4：Representation 应该稳定到什么程度？

这个现在是最适合你的“KV layout”主线的问题。

vLLM提供最直接的实际系统证据：

\[
L_{\rm transfer}
\neq
L_{\rm local}
\]

并显式转换。fileciteturn439file0L1-L2

Triton提供kernel-local：

\[
L_{\rm producer}
\neq
L_{\rm reduction}
\neq
L_{\rm store}.
\]

TileLang采用 widen/replicate/fallback。

Hexcute采用multi-consumer constraint unification。

因此应该凝练为：

\[
\boxed{
\textbf{RQ4:
When should a multi-consumer tensor use one stable representation, versus specialization, conversion, or replication?}
}
\]

可以形式化为：

\[
\min_{\{L_i\}}
\left[
\sum_i f_iT_i(L_i)
+
\sum_{i,j}C(L_i,L_j)
+
\lambda M(\{L_i\})
\right].
\]

其中：

\[
T_i
\]

是consumer性能；

\[
C
\]

是layout conversion/transfer cost；

\[
M
\]

是维护多representation的memory cost；

\[
f_i
\]

是consumer调用频率。

这个问题可以统一两种尺度。

Kernel-local：

\[
QK^T
\rightarrow Softmax
\rightarrow PV.
\]

Serving：

\[
KVCache
\rightarrow
\{
Prefill,
Decode,
Transfer,
QuantizedConsumer
\}.
\]

所以它不再只是“动态KV layout”。

而是：

\[
\boxed{
\text{Representation stability/specialization problem}
}
\]

。

---

# 二十六、这四个问题之间其实已经形成一条逻辑链

现在我认为我们的研究框架终于比较完整了：

```text
                    Optimization System
                           |
                           v
                Generate candidate space
                           |
                  [RQ2: Coverage]
                           |
                           v
                 Rank/select candidates
                           |
                  [RQ1: Ranking]
                           |
                           v
             Decisions made in separate stages
                           |
               [RQ3: Separability]
                           |
                           v
       Same tensor crosses consumers / system boundaries
                           |
          [RQ4: Representation Stability]
```

也可以统一成 regret decomposition：

\[
\boxed{
Regret_{\rm total}
=
Regret_{\rm coverage}
+
Regret_{\rm ranking}
+
Regret_{\rm boundary}
+
Regret_{\rm representation}
+
InteractionTerms.
}
\]

这比最开始“比较不同框架的layout优化方法”已经前进很多。

因为现在研究对象不是框架本身。

而是：

\[
\boxed{
\text{框架共同暴露出的 GPU layout optimization 的基本科学限制。}
}
\]

---

# 二十七、哪些原来的“问题”现在应该明确降级

经过这一轮，我认为下面几项不能再作为独立主问题。

“Hexcute没有考虑bank conflict”只能作为 RQ1 evidence。

“Triton有很多ConvertLayout”只能作为 RQ3/RQ4 的 observable symptom。

“vLLM只选一个whole-model KV layout”只是 RQ4 的一个设计点；因为稳定layout也有明显收益。

“TIRx priority不是最优”目前只能作为 RQ1/RQ2 的机制，不是独立科研问题。

“TVM搜索空间有限”不是问题本身，真正问题是：

\[
CoverageRegret?
\]

“CUTLASS依赖专家layout”也不是问题，真正问题是expert constraints是否保留near-optimal coverage。

“FlashInfer heuristic很多”不是问题，真正问题是这些cheap runtime signals是否足够ranking。

这一轮这些边界应该算是整理清楚了。

---

# 二十八、当前还没有整理完的最后几个地方

接下来不应该再漫无目的扩大框架数量，而应该把剩余证据封口。

最值得继续审查的是四块：

第一，CUTLASS SM100/103 的 Auto decision table，要完整形成：

\[
(InputType,
Tile,
Cluster,
Alignment)
\rightarrow
(MMA,
1/2SM,
SMEMLayout,
Stages,
Schedule)
\]

并标出哪些是 legality，哪些是 heuristic。

第二，TIRx 把每个 `copy/copy_async/gemm/gemm_async/reduction/permute` 的：

\[
(priority,predicate,variant)
\]

做成最终矩阵。Dispatcher 总机制已经不用再查。fileciteturn420file0L1-L10

第三，vLLM 把所有backend形成最终：

\[
Backend
\times
StrideLayout
\times
ContentPacking
\times
BlockSize
\times
Connector
\]

矩阵。目前机制已经基本完成，不需要再重新理解。

第四，SGLang所有model override按字段而不是按model归类：

\[
AttentionBackend
\]

\[
PageSize
\]

\[
KVCacheDType
\]

\[
Prefill/DecodeSplit
\]

\[
Parallelism
\]

\[
MoEBackend.
\]

这样才能判断哪些cross-variable coupling最普遍。

---

# 二十九、这一轮之后，我对主线的推荐也发生了一个小调整

现在不建议只用原来的：

\[
P1\ BoundaryRegret
\]

作为论文总故事。

更强的总框架可能是：

\[
\boxed{
\textbf{Where does optimization regret come from in GPU layout systems?}
}
\]

拆成三个基础来源：

\[
\boxed{
Coverage
\rightarrow
Ranking
\rightarrow
Decomposition
}
\]

再把KV作为最强的系统级case study：

\[
\boxed{
Representation\ Stability
}
\]

。

也就是：

\[
\boxed{
\text{RQ1 Ranking Sufficiency}
}
\]

\[
\boxed{
\text{RQ2 Candidate Coverage}
}
\]

\[
\boxed{
\text{RQ3 Decision Separability}
}
\]

\[
\boxed{
\text{RQ4 Representation Stability}
}
\]

。

其中目前**证据最完整的是 RQ1**；**最可能形成compiler方法论贡献的是 RQ2+RQ3**；**和你的KV-layout主题最直接的是 RQ4**。

下一轮最有价值的工作已经不是继续重复框架介绍，而是从 **RQ1** 开始建立最终的 **Problem Evidence Matrix**：把十个系统每一个具体rule、使用的信息、忽略的信息、已有repair/counterexample、可构造的反例条件、以及“什么实验结果会直接否定这个科研问题”逐项钉死。完成 RQ1 后再依次做 RQ2、RQ3、RQ4；仍然不写代码。

## 用户

请先完成所有框架的证据链（一定要精细：一个框架的每一种启发式，每一种规则是否都完全整理，每一种假设，以及在完成layout选择的每一个策略，都做了细粒度的整理统计和对比 。尤其是上述没有完成的） 到 科研问题的凝练；

然后从RQ1开始做一条完整的“问题审判”：所有相关框架源码规则 → 每个规则使用的proxy → 它忽略了哪些变量 → 框架自己的fallback/repair → 现代Attention/GQA/MLA/MoE触发条件 → 可证伪的科研假设。 RQ1彻底完成之后再审判RQ2，然后后续RQ。这样可以防止又回到前几轮“问题列了很多，但证据深度不够”的状态。

## 模型思考摘要

规划证据链
规划了证据链

## 模型回复

我先按你要求把顺序固定下来：这次先把十个框架的 **production layout decision chain** 补到规则族/入口级，明确哪些是硬约束、结构规则、启发式、cost/ranking、fallback/repair；只有这部分封口后，才进入 RQ1 的完整“问题审判”。本轮不会提前审判 RQ2。

## 模型工具调用参数

{"paths":["GitHub"],"query":"search"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

计算CUTLASS自动调度与混合精度支持

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 CUTLASS 中 SM100 计算阶段配置函数

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 CUTLASS 专用 Epilogue 与自动调度

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 CUTLASS Epilogue 启发式实现

## 模型思考摘要

检索并计算CUTLASS自动调度与Epilogue配置

## 模型回复

CUTLASS 这部分又补出两个此前需要单列的 production 规则族：SM100 dense builder 的 accumulator pipeline 不是纯容量推导，而是明确把 accumulator stages 上限设为 4，并注释“works well”；SM100 epilogue 又有独立的 C/D SMEM swizzle 与 epilogue-tile 选择。因此 mainloop、accumulator、epilogue 不能再合成一个“StageCountAuto”。 

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

全面检索 Triton 启发式与优化 Pass

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索并定位相关优化文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 中 GenericSwizzling 银行冲突实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 TritonGPU 转换目录内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审查 Triton GPU 布局变换文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 中的银行冲突最小化代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Triton 描述符编码优化实现

## 模型思考摘要

检索并审查 Triton 启发式、布局变换及优化实现

## 模型回复

Triton 这一块现在已经可以从“pass 名单”提升到“决策链”：`Coalesce` 明确为每个 memory op 构造更 coalesced 的 distributed layout，再插 conversion；descriptor load/store 则单独限制到最多 16B vectorization。GenericSwizzling 又独立以总 bank conflict 最小为 proxy，平局看搬运 rounds；TMA descriptor layout 还有自己的 swizzle/FP4-padding候选。也就是说 Triton 至少有三套不同 proxy 在分别优化不同 layout 层，而不是一个统一 selector。  

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Triton Gluon 布局推导实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查找 Gluon 方言目录路径

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

继续获取 Transforms 目录

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计 Gluon 布局推断实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

推导合并与自动解析编码布局

## 模型思考摘要

检索并审计 Triton Gluon 布局推导实现，推导布局合并解析

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 TVM MetaSchedule CUDA 默认规则

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取完整调度规则源文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 TVM 默认 CUDA 后处理流程

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 CUDA Tile Primitive 目录 contents

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

统计并分析CUDA Tile原语实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 TVM CUDA 异步复制目录内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 TIRx 调度注册与优先级

## 模型思考摘要

梳理并检索 TVM CUDA 调度规则与原语实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Hexcute/Hidet 源码仓库

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析 Hexcute CUDA 自动调优与验证流程

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 Hexcute 成本模型与指令路径

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析 CUDA 指令选择与存储体冲突

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

解析 CUDA 共享内存银行冲突

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

深入核查 TileLang 布局推导规则

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取布局代价模型源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 TileLang 默认成本配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 vLLM 后端 KV 缓存支持

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索并定位 customize_spec 定义

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索首选块大小实现

## 模型思考摘要

完成 Hexcute 与 TileLang CUDA 调优、布局成本及 vLLM KV 缓存实现分析

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索并归类 SGLang 模型覆盖配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理模型覆盖规则与启发式配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 attention_backend 覆盖实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 DeepSeek 与 Qwen 模型布局规则

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Qwen3.5 文件及混合注意力配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 FlashInfer 注意力调度器族系

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

列出 Blackwell 目录内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检查 Blackwell 任务调度规则

## 模型思考摘要

梳理SGLang模型覆盖与注意力调度配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查找 vLLM 当前后端选择器实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理现代 GPU 内核触发规则

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 CUTLASS 混合输入调度配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析 CUTLASS SM100 混合输入 UMMA 构建器

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索并裁定模型配置与框架规则

## 模型思考摘要

核验审计范围
审查布局规则
核验审计范围并审查布局规则

## 模型思考摘要

查找并分析 vLLM与CUTLASS混合输入调度规则

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

各框架布局决策证据链审计与 RQ1 终审

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 DeepSeek 配置参数定义

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 DeepSeek-V3 配置默认值

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 Qwen3 模型配置定义

## 模型思考摘要

审计布局证据并检索DeepSeek与Qwen3配置定义

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 TileLang TMA 复制重写机制

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 TMA 拷贝实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析 TileLang 生产者消费者与 TMA 自动选择

## 模型思考摘要

检索并分析 TileLang TMA 复制及自动选择机制

## 模型思考用时

思考了 8m 57s

## 模型回复

这一次我按你要求把顺序彻底固定下来：

\[
\boxed{
\text{框架源码规则封口}
\rightarrow
\text{假设/启发式证据链}
\rightarrow
\text{科研问题凝练}
\rightarrow
\text{RQ1完整审判}
}
\]

本回答不会继续审判 RQ2。RQ1 没有完成以前，不向后跳。

有一个“完整性”的边界需要先说明清楚：我这里所谓“整理完成”，指的是**所有 production/default 路径中，会自动改变 physical layout、layout propagation、task mapping、page/content geometry、layout-coupled instruction、pipeline/warp role 或 backend→representation 的决策族，都已经进入目录**。单个 example/tutorial 中用户手写的 tile size 不算框架启发式；成百上千个重复的 `static_assert` 按“硬约束族”归档，而不是机械复制每一行。按这个边界，本轮之后已经没有发现一个新的、尚未归类的主要 production layout-decision family。后续当然仍可能存在某个 dtype/architecture 叶分支需要补充，但它不会再改变下面的机制分类。

---

# 一、CUTLASS / CuTe：最终规则链

CUTLASS 不能描述成“expert chooses layout”，也不能描述成“Auto layout”。生产路径实际上有多组相互关联但独立的 builder policy。

| 决策 | 当前源码机制 | 类型 | 实际 proxy / 假设 |
|---|---|---|---|
| GMEM layout → MMA major | 根据 stride tag、dtype 映射到 UMMA/GMMA Major；某些 dtype 强制 K-major | C0/R1 | instruction legality |
| GMEM→SMEM copy | cluster shape、alignment决定 TMA / multicast / cp.async 路径 | C0/R1 | instruction capability、cluster geometry |
| SIMT copy thread layout | 尽量增加 GMEM major dimension 上的threads | H2 | coalescing |
| SMEM layout | SS/RS分别有 swizzle selector；常见策略为最大合法 SW128→64→32→interleaved，RS又存在bank-conflict特殊规则 | R1/H2 | instruction compatibility + bank conflict |
| 1SM/2SM MMA | SM100/103 Auto根据cluster/tile条件决定1SM或2SM | R1 | cluster divisibility，不benchmark |
| mainloop stage count | 从SMEM capacity减去carveout，再除以per-stage footprint | R1 | 最大可容纳pipeline depth |
| accumulator stages | SM100 dense中上限4，源码直接注明“4 stages works well” | H1 | 经验规则 |
| block-scaled layout | SFA/SFB拥有独立SMEM/TMA layouts与stage footprint | C0/R1 | scale-factor instruction/layout constraints |
| mixed-input stages | 分成 load→transform、transform→MMA、accumulator 三个stage count，同时受TMEM和SMEM约束 | R1 | resource capacity |
| epilogue | 独立选择epilogue tile、C/D SMEM layout/swizzle | R1/H1 | epilogue-specific datatype/SMEM reuse |
| fallback | 用户指定StageCount、KernelSchedule、expert CollectiveMma覆盖Auto | F1 | expert override |

SM100 dense builder本身同时构造 `TiledMma`、TMA copy atoms、SMEM layouts和stage资源；accumulator stage又明确封顶4。fileciteturn418file0L1-L2 fileciteturn459file0L1-L30

SM103 block-scaled路径已经不是相同的stage公式：SFA/SFB也进入stage footprint，而且 `KernelScheduleAuto` 在cluster-M可被2整除时选择2SM，否则1SM；SFB还有N=192时按256粒度组织scale-factor layout的特殊情况。fileciteturn440file0L1-L2

mixed-input更明确地说明“StageCount”不是单一变量：源码计算 `Load2TransformStageCount / Transform2MmaStageCount / AccumulatorStageCount` 三组资源预算，而且input-transform path不能使用TMA 2SM。fileciteturn502file0L1-L2

另外mainloop和epilogue不能再合并统计，SM100 epilogue有自己的SMEM swizzle/tile selector。fileciteturn460file0L1-L11

所以 CUTLASS 最终应定义为：

\[
\boxed{
\text{constraint-heavy coherent construction}
+
\text{expert structural heuristics}
}
\]

而不是“autotuning”。

---

# 二、Triton classic：最终规则链

Triton 是目前规则最分散的框架。审完以后，至少存在下面九套不同的layout决策器。

| 决策器 | 选择什么 | Proxy / rule |
|---|---|---|
| `Coalesce` | GMEM distributed encoding | AxisInfo、最大coalescing/vectorization |
| `AccelerateMatmul` | MMA-compatible operand/result layout | tensor-core compatibility |
| `OptimizeDotOperands` | dot operand layout | instruction/transpose/reuse |
| `OptimizeThreadLocality` | reduction/reshape/gather distributed layout | 减少cross-thread communication |
| `GenericSwizzling` | generic SMEM swizzle | 最小总bank conflicts；tie看data-move rounds |
| `OptimizeDescriptorEncoding` | TMA descriptor shared encoding | descriptor legality/equivalence；扫描swizzle/FP4-padding |
| `OptimizeTMemLayouts` | TMEM→register/shared layout | downstream consumer、reduction、vectorization |
| Warp specialization | partition和warp数 | op-type heuristic、edge size、粗略register模型 |
| pipeline | async load/staging | access width、register pressure、MMA operand role |
| repair | `ConvertLayout`、rematerialization、shared staging | 消除不同局部目标产生的representation mismatch |
| external autotune | 用户给出的meta-config | 真实benchmark |

`Coalesce`的源码目标非常明确：为memory op构造“best memory coalescing”的encoding，并在原layout与memory-friendly layout间插转换；descriptor load/store又单独限制vectorization。fileciteturn465file0L1-L13

GenericSwizzling更直接写明：

> Current heuristic: Minimise total bank conflicts.

平局再看搬运rounds。fileciteturn466file0L1-L11

Warp specialization是目前最强的“启发式证据”。register estimation源码自己称：

> Extremely rough estimate

。fileciteturn446file0L1-L35

Partition scheduling又存在：

\[
edge.size>16384
\]

的merge阈值，旁边源码直接标：

> seemingly arbitrary size.

fileciteturn447file0L1-L30

更重要的是，如果warp数改变，Triton明确重新进行layout assignment：

\[
N_{\rm warp}\rightarrow\text{change}
\Rightarrow
L\rightarrow\text{reassign}.
\]

fileciteturn448file0L1-L46

TMEM也会根据consumer改layout。例如有N维reduction时改成沿M分发以避免cross-warp reduction；shared↔TMEM路径又为了16-bit vectorization使用 `16x256b` alternative layout，随后依赖 `RemoveLayoutConversions` 修复。fileciteturn456file0L1-L2

因此 Triton 最准确的定义现在是：

\[
\boxed{
\text{progressive local proxy optimization}
+
\text{representation reconciliation}
}
\]

。

---

# 三、Gluon：最终规则链

Gluon的关键不是“manual layout”，而是：

\[
\boxed{
\text{explicit-by-default}
+
\text{narrow automatic inference}
}
\]

。

| 决策 | 机制 |
|---|---|
| Distributed layout | programmer显式给 `BlockedLayout` / distributed layout |
| `AutoLayout` | 从layout seeds传播约束 |
| fuzzy propagation | Join/Split/Reshape/Transpose每跨一步增加inference distance |
| 冲突 | 选择distance更小的seed；非零同distance用stable hash；两个distance=0 seed冲突则error |
| `CoalescedLayout` | 对load/store用AxisInfo构造coalesced seed，再forward/backward传播 |
| CGA | 当前Auto Coalesced路径只支持 `numCTAs == 1` |
| SMEM default | `NVMMASharedLayout.get_default_for()`选择最大兼容swizzle以减少TMA/MMA messages |
| warp specialization | `num_warps/num_regs`主要显式 |
| performance search | 可以叠加Triton autotune |

AutoLayout的distance机制说明它优化的是“离哪个seed更可信”，而不是runtime。fileciteturn471file0L1-L10

CoalescedLayout则明确使用AxisInfo和当前warps构造memory-friendly encoding，而且目前硬限制 `numCTAs==1`。fileciteturn472file0L1-L13

所以 Gluon 的自动层主要解决：

\[
\boxed{\text{consistency + coalescing}}
\]

而不是performance ranking。

---

# 四、TVM MetaSchedule：最终规则链

TVM需要拆成“candidate generation”和“ranking/measurement”。

DefaultCUDA已经预先固定了非常强的结构先验：

\[
structure=SSSRRSRS
\]

\[
vector\_loads\in\{1,2,3,4,8,16\}
\]

read cache：

\[
\text{MUST shared @ level 4}
\]

write cache：

\[
\text{MUST local @ level 3}
\]

CrossThreadReduction只从：

\[
\{4,8,16,32,64,128,256,512\}
\]

选择thread extent；AutoBind同样从有限thread集合选择。fileciteturn474file0L1-L2

TensorCore路径仍使用 `SSSRRSRS`，但预先提供有限的WMMA/MMA intrinsic groups，并有两套重要schedule family：

\[
\text{shared read+write,\ no software pipeline}
\]

与：

\[
\text{shared read,\ no write reuse,\ software pipeline}
\]

。fileciteturn474file0L1-L2

随后还有Postproc：

`DisallowDynamicLoop`、`RewriteCooperativeFetch`、`RewriteUnboundBlock`、`RewriteReductionBlock`、`VerifyGPUCode`，TensorCore路径再做 `RewriteTensorize`。fileciteturn475file0L1-L18

而在这些结构规则生成候选后，MetaSchedule的Builder/Runner确实在真实硬件上编译运行candidate。fileciteturn452file0L1-L20

因此：

\[
\boxed{
\text{Structural candidate prior}
\rightarrow
\text{learned/search ranking}
\rightarrow
\text{hardware measurement}
}
\]

。

`RewriteLayout`则应继续与MetaSchedule schedule-space分开：它属于physical-buffer representation rewrite，不是同一个search mechanism。

---

# 五、TIRx：最终规则链

TIRx现在已经可以按注册表严格描述。

全局dispatch语义：

\[
(Op,Target)
\rightarrow
\{Variant_i\}
\]

每个variant拥有：

\[
(priority,predicates,implementation).
\]

运行时：

\[
\boxed{
priority\downarrow
\rightarrow
predicates
\rightarrow
\text{first successful implementation}
}
\]

；失败继续下一个；显式 `dispatch=` 会只留下指定variant。fileciteturn420file0L1-L10

CUDA生产primitive families已经完整落在：

\[
copy,\ copy\_async,\ elementwise,\ gemm,\ gemm\_async,\ permute\_layout,\ reduction.
\]

fileciteturn476file0L1-L10

同步copy进一步包括：

`vec_forced`、`vec_auto_gmem_smem`、`vec_auto_reg`、`ld/stmatrix`、general fallback。fileciteturn477file0L1-L10

异步copy则包括：

`ldgsts`、TMA、DSMEM、`tcgen05_cp`、`tcgen05_ldst`。fileciteturn478file0L1-L10

典型priority层次也是源码事实，例如SM100 packed reduction可以priority 20，而local/shared reduction常见priority 10；TMEM↔local tcgen05 path也属于priority 10 family。fileciteturn421file0L1-L18 fileciteturn421file3L55-L70 fileciteturn479file0L1-L15

所以 TIRx 本质上是：

\[
\boxed{
\text{layout-aware legality predicates}
+
\text{designer priority}
}
\]

不是性能search。

---

# 六、Hexcute：最终规则链

Hexcute现在反而是信息最完整的框架之一。

自动推导流程已经由源码自己明确分为：

\[
LogicalShape
\rightarrow
LogicalLayout
\rightarrow
TaskMapping/TVLayout
\rightarrow
MemoryLayout.
\]

Copy要求：

\[
f\circ p^{-1}=g\circ q^{-1}.
\]

MMA要求：

\[
f_{1m}=f_{3m},
\quad
f_{2n}=f_{3n},
\quad
f_{1k}=f_{2k}.
\]

多个合法instruction可以：

\[
DFS
\]

枚举，也可以heuristic/beam search prune，然后交给cost model选择。fileciteturn481file0L1-L2

Shared tensor如果有多个copy consumers，所有memory constraints共同refine physical layout；冲突则backtrack；约束解决完后，剩余未知stride再用heuristic补齐。fileciteturn481file0L1-L2

Instruction matching进一步检查：

\[
scope,
element\ width,
alignment,
TVLayout,
loop\ structure.
\]

支持的instruction families包括LDG/STG、LDS/STS、`ldmatrix`、`mma.sync`、WGMMA、`cp.async`、TMA等。fileciteturn483file0L1-L2

Bank-conflict resolver又是独立机制：把访问同一个shared tensor的全部Copy合并考虑，枚举swizzle，最小化所有access的bank-conflict ways；TMA专门检查四种合法swizzle；WGMMA operand则可能因为instruction约束成为immutable。fileciteturn484file0L1-L2

最后cost model：

\[
Cost=\#Instructions\times CPI
\]

并区分independent/dependent CPI、估计copy/MMA overlap。源码同时明确写出它没有完整计入bank conflict、`cp_async_wait_group`、`mbarrier`，并假设pipeline充分overlap。fileciteturn482file0L1-L2

因此：

\[
\boxed{
\text{constraint synthesis}
+
\text{instruction-aware analytic ranking}
+
\text{specialized bank-conflict repair}
}
\]

。

---

# 七、TileLang：最终规则链

TileLang经过最近几轮源码变化，已经不能用以前的“一次 LayoutInference”解释。

它目前有：

\[
Strict\rightarrow Common\rightarrow Free
\]

层级式layout inference。

thread extent必须是编译期常量。fileciteturn443file0L1-L2

Free mode不是只推一个解，而是比较完整layout attempts。目前正式支持两套ranking：

\[
\texttt{register-count}
\]

默认最小化fragment register slots；

以及：

\[
\texttt{io-aware}
\]

根据fragment↔global访问的vector width、coalescing、bytes moved估计IO代价，再用register count tie-break。fileciteturn487file0L1-L13

IO-aware的核心近似是：

\[
Time(S)\approx\max(BW(S),Issue(S))
\]

然后对statements求和；对non-affine、dynamic、swizzled或不可bijective的candidate，则收取保守worst-case，防止“模型看不懂反而被判低cost”。fileciteturn486file0L1-L2

多consumer/reducer冲突时又不是简单失败。

`ReducerDstSteering`存在的原因，源码直接解释为：如果consumer先完成layout，会绕过finalize的planner decision，最终产生thread-indexed publish copy。因此现在给finalize destination ownership。fileciteturn443file0L1-L2

Reducer冲突使用：

\[
unset\rightarrow narrow\rightarrow wide
\]

monotonic widening，最后可以seed universally-readable replicated fallback。Alias buffers又自动传播layout；跨dtype alias用storage-bit ratio reshape，专门处理FP4这种sub-byte关系。fileciteturn443file0L1-L2

Pipeline/warp-specialization transformation位于LayoutInference之前，因此最终layout能看到已经物化的producer/consumer结构；warp-specialized路径会显式materialize TMA copies和barriers。fileciteturn512file0L1-L2

因此：

\[
\boxed{
\text{component-wide constraint inference}
+
\text{free-mode proxy ranking}
+
\text{widen/replicate repair}
}
\]

。

---

# 八、vLLM：最终规则链

vLLM现在必须被定义成一个 **persistent representation resolver**，而不是KV stride selector。

表示应写成：

\[
R_{KV}
=
(
StrideOrder,
ContentPacking,
BlockGeometry,
MetadataLayout,
TransferRepresentation
).
\]

Stride候选是：

\[
LBHNC,LBNHC,LHBNC,BLHNC,BLNHC,BHLNC.
\]

而且layout本身带 `block_compact`、`block_contiguous` 等语义属性。fileciteturn422file0L1-L13

每个backend提供ordered `supported_kv_cache_layouts()`。如果多个backend声明不同集合：

1. 求intersection；
2. 统计各backend第一选择；
3. surviving candidates按第一选择票数排序；
4. mixed HNC specs要求block-compact；
5. connector preference如果兼容则参与；
6. user override必须合法；
7. 最终 **one KV cache layout for the whole model**。

fileciteturn425file0L1-L2

而consumer偏好是真实不同的，例如CPU只LBHNC、HPC只LBNHC、B12X支持LBHNC/BLHNC、QSA偏BLNHC/BLHNC、FlexAttention只接受LBNHC，因为需要把 `(B,N)` 零拷贝flatten成token dimension。fileciteturn488file0L1-L12

然后 `customize_spec()` 又允许backend改变page内部表示。源码接口自己说明：

> kernels want KV packed in a specific way.

fileciteturn489file2L27-L38

实际已有ROCм K/V分head groups、AITER content-packed、Triton inline scales、TurboQuant K+V one-slot/head、FlashInfer NVFP4的packed-FP4+FP8-scale等多种表示。fileciteturn489file0L1-L12 fileciteturn489file3L40-L50 fileciteturn489file5L66-L76 fileciteturn489file7L92-L102

Page geometry又有独立 `get_preferred_block_size()`；不同backend可以固定/改变block size，例如Sparse MLA/SWA偏256，ROCm AITER unified偏64。fileciteturn490file0L1-L11 fileciteturn490file3L34-L43 fileciteturn490file4L45-L54

最后NIXL甚至偏好LBHNC以提高P/D transfer效率，而receive path支持layout转换以及layout+block-size联合转换；源码明确给出：

\[
prefill:
LBHNC+\text{smaller block}
\]

\[
decode:
LBNHC+\text{larger block}.
\]

fileciteturn439file0L1-L2

所以完整策略是：

\[
\boxed{
BackendCompatibility
\rightarrow
PersistentRepresentationNegotiation
\rightarrow
BoundaryConversion
}
\]

。

---

# 九、SGLang：最终规则链

SGLang现在应明确分三层：

\[
\boxed{
GenericPolicy
\rightarrow
ModelFamilyOverride
\rightarrow
BackendInternalPolicy
}
\]

generic attention selector本身已经是复杂hand-written decision tree。

MHA例子：

Hopper兼容 → FA3；

SM100兼容 → TRTLLM MHA；

asymmetric K/V → FA4；

HIP → AITER；

attention sink不能走FlashInfer → Triton fallback。

MLA又有独立Hopper/Blackwell/HIP逻辑；HIP AITER只有特定head count才走，否则Triton。fileciteturn428file0L1-L2

而源码直接记录了一个真实性能修正：

> FlashInfer 0.6.1 在 Hopper attention kernel 上有performance regression，因此默认改FA3。

fileciteturn450file0L1-L25

之后还有规模很大的model override registry。当前生产目录包含DeepSeek、Qwen、Gemma、GLM、GPT-OSS、Cohere MoE、GraniteMoEHybrid等大量family。fileciteturn491file0L1-L2

这些override并不只是correctness patch。

DeepSeek DSA中：

ROCm且preshuffle paged-MQA不可用：

\[
page=1
\]

否则GPU：

\[
page=64
\]

XPU：

\[
page=128.
\]

而SM100普通DeepSeekV3又可改用 `trtllm_mla`。fileciteturn494file0L1-L10

Qwen3.5在SM100上更明确地把backend和page size联动，测试确认其中一条resolution是：

\[
\{\texttt{attention\_backend=trtllm\_mha},
\texttt{page\_size=64}\}.
\]

fileciteturn495file0L19-L35

同时framework允许prefill/decode backends分别解析。fileciteturn504file0L1-L18 fileciteturn506file0L1-L18

因此 SGLang 的真实策略是：

\[
\boxed{
\text{expert runtime policy}
+
\text{model-specific exception hierarchy}
}
\]

。

---

# 十、FlashInfer：最终规则链

FlashInfer不能再只记成“NHD/HND + split-KV”。

基础FA2/prefill scheduler首先计算batch平均：

\[
avg\_packed\_qo\_len
\]

再调用：

\[
FA2DetermineCtaTileQ
(avg\_packed\_qo\_len,
head\_dim,
head\_dim_{qk},
kv\_dtype\_bytes).
\]

fileciteturn429file0L1-L25 fileciteturn429file1L26-L46

kernel family又是离散的：

\[
NUM\_MMA_Q\in\{1,2\}
\]

\[
NUM\_MMA_{KV}\in\{8,4,2,1\}
\]

\[
CTA_Q\in\{128,64,32,16\}
\]

\[
GQA\ group\in\{1,2,3,4,6,8\}
\]

\[
Vec\in\{16,8,4,2,1\}
\]

，decode SMEM stages也直接由compute capability决定。fileciteturn430file0L1-L2

大QK/VO dimension又会因为fragment/register pressure减少可选CTA-Q。fileciteturn431file1L27-L58

split-KV比静态rule更复杂：它查询真实CUDA occupancy；如果当前batch×heads已经填满grid则不split，否则binary-search KV chunk去增加parallel work。

Blackwell又是另一套scheduler philosophy。当前planner为每个：

\[
(batch,head,Q\ tile)
\]

生成工作项，并把：

\[
cost\approx KV\ length
\]

加入当前cost最低的bucket，做动态greedy load balancing。fileciteturn498file0L1-L10

所以最终应定义为：

\[
\boxed{
\text{discrete specialized kernels}
+
\text{analytic/runtime workload scheduling}
}
\]

而不是autotune。

---

# 十一、框架证据链现在是否达到“科研问题凝练”的标准？

按上述边界，我认为答案现在是 **是**。

不是说所有repository里的每一个`if`都被逐字复制了；而是：

\[
\boxed{
\text{没有已知的主要production layout-decision mechanism仍未归类。}
}
\]

横向结果可以压缩成这张表：

| Framework | Candidate source | Legality | Performance proxy / selector | Hardware measurement | Repair |
|---|---|---|---|---|---|
| CUTLASS | expert builder family | template/static constraints | capacity、swizzle、coalescing、expert rules | 否 | expert override |
| Triton | compiler rewrite space | encoding/instruction legality | 多个局部proxy | 可外部autotune | conversion/remat/relayout |
| Gluon | explicit+seed inference | encoding consistency | distance、coalescing、expert | 可autotune | explicit conversion/error |
| TVM | ScheduleRules | postproc/intrin | XGB/search | **是** | reject/retune |
| TIRx | registered variants | predicates | priority-first | 否 | next variant |
| Hexcute | constraint-derived | unification | analytic instruction latency | 否 | backtrack/bank resolver |
| TileLang | inferred complete attempts | inference legality | register-count / IO-aware | 否 | widen/replicate/retry |
| vLLM | backend/layout enums | support intersection | preference voting/policy | 否 | conversion/override |
| SGLang | backend/model rules | feature compatibility | hand-written policy | 否 | model override/fallback |
| FlashInfer | specialized kernel family | shape/dtype/arch | analytic geometry/occupancy/work cost | occupancy query，不是benchmark | replanning/fallback |

因此科研问题现在不是从“某一个框架有bug”产生，而是从这些**不相同、互不支配的解法**产生。

目前按证据广度排序，问题应收敛成：

| 顺序 | 科研问题 |
|---:|---|
| **RQ1** | Layout ranking到底需要什么信息？ |
| RQ2 | 结构规则能安全裁掉多少candidate space？ |
| RQ3 | 哪些layout-related decisions可以分离，哪些必须joint？ |
| RQ4 | multi-consumer tensor应稳定representation还是specialize/convert/replicate？ |

下面只审 RQ1。

---

# 十二、RQ1 的精确定义

以前的表述：

> “heuristic vs measurement”

还是过粗。

真正的问题是：

\[
\boxed{
\textbf{RQ1:
What information is necessary and sufficient to reliably rank GPU layout configurations for modern LLM workloads?}
}
\]

设一个完整configuration：

\[
c=
(
L_G,L_S,L_R,L_T,
ThreadMapping,
WarpCount,
CopyInst,
MMAInst,
Pipeline,
WarpRole
).
\]

真实运行时间：

\[
T(c\mid w,h)
\]

依赖workload \(w\) 和hardware \(h\)。

现有framework实际上使用：

\[
\hat T(c)=f(\phi(c))
\]

或者甚至没有显式 \(\hat T\)，只用priority/rules。

核心问题就是：

\[
\boxed{
\phi(c)\text{ 至少必须包含什么，才能保持 }T\text{ 的排序？}
}
\]

这里我们**不要求绝对性能预测准确**。

真正重要的是：

\[
T(c_i)<T(c_j)
\Rightarrow
\hat T(c_i)<\hat T(c_j).
\]

---

# 十三、RQ1：十个框架的完整证据审判

| Framework | 实际ranking信息/proxy | 源码没有直接建模的信息 | 自己的repair / counter-evidence |
|---|---|---|---|
| CUTLASS | tile divisibility、alignment、SMEM/TMEM capacity、compatible swizzle、cluster geometry、“4 stages works well” | 实际cache行为、pipeline stalls、runtime shape distribution、真实candidate latency | 手工Stage/Schedule、expert API、不同builder family |
| Triton | coalescing、bank conflicts、cross-thread communication、rough register estimate、fixed merge threshold、consumer pattern | 所有局部proxy的全局组合效果、真实PTXAS live ranges、consumer间conversion总成本 | `ConvertLayout`、remat、TMEM relayout、warps改变后重新layout、autotune |
| Gluon | inference distance、AxisInfo coalescing、largest compatible shared swizzle、expert config | kernel总runtime、occupancy、future consumer/async overlap | explicit layout、conversion、conflict error、autotune |
| TVM | structural schedule priors + learned cost model + hardware time | generated space之外的candidate；serving distribution/cross-task persistent state | real Runner、postproc rejection、database |
| TIRx | legality predicates + priority | 两个同时合法variant的真实速度、whole-kernel resource interaction | next variant、user forced dispatch |
| Hexcute | instruction count×CPI、dependent/independent latency、idealized copy/MMA overlap | bank-conflict penalty在main cost中、barrier/wait、imperfect overlap、occupancy/cache | backtracking + separate bank-conflict optimization |
| TileLang | register slots或GMEM vector/coalescing/traffic model | register proxy不看IO；IO proxy不完整看compute/SMEM/bank/async overlap | two cost models、free-mode alternatives、replicate/widen |
| vLLM | backend support intersection、first-choice vote、block-size/backend preference | consumer frequency、attention latency+insert+transfer+conversion的统一cost | connector preference、transfer conversion、user override |
| SGLang | hardware/model/features/version-specific rule tree | 当前真实batch/Sq/Skv分布以及多个合法backend的实时速度 | model-family overrides、prefill/decode split、fallback；已有performance-regression workaround |
| FlashInfer | avg packed-Q、head dim、GQA、register/SMEM rules、occupancy、KV-length bucket cost | exact cache/stall/bank/pipeline timing、batch内部variance被平均化的部分 | split/no-split、dynamic plan、different kernel families |

这个表里，Triton证据尤其强，因为源码自己称register model为“Extremely rough”，且有“seemingly arbitrary”阈值。fileciteturn446file0L1-L35 fileciteturn447file0L1-L30

Hexcute同样不是我们推断它省略了信息，而是源码主动列明了perfect-overlap、bank conflict、barrier/wait等假设。fileciteturn482file0L1-L2

TileLang则特别有价值，因为同一系统主动保留两种不同proxy：`register-count` 与 `io-aware`。fileciteturn487file0L1-L13

TVM构成了另一端：它实际跑hardware measurement。fileciteturn452file0L1-L20

因此现有系统实际上已经覆盖完整谱系：

\[
\boxed{
Priority
\rightarrow
Rule
\rightarrow
LocalProxy
\rightarrow
AnalyticModel
\rightarrow
LearnedModel
\rightarrow
HardwareMeasurement
}
\]

却没有统一答案。

这就是 RQ1 的第一层科学证据。

---

# 十四、哪些“proxy assumption”已经能被源码直接证明不充分？

这一点必须和“可能不最优”分开。

现在已经能直接否定的强假设有：

| 强假设 | 判决 | 直接证据 |
|---|---|---|
| layout可以与warp count独立选择 | **不成立** | Triton改变warp count后必须重新layout |
| 一个TMEM layout与consumer无关 | **不成立** | Triton根据reduction/local load/store更换TMEM layout |
| shared tensor谁先局部推导出layout就可以决定它 | **不成立** | TileLang专门增加ReducerDstSteering修复first-completer问题 |
| 一个persistent KV representation适合compute和transfer全部边界 | **不成立，作为普遍命题** | vLLM P/D transfer使用不同layout/block并在receive转换 |
| generic model/hardware backend policy足够 | **不成立，作为普遍命题** | SGLang存在大量model-family overrides |
| backend choice与page geometry独立 | **不成立** | SGLang Qwen/DeepSeek路径联动backend/page |
| tensor layout只由shape决定即可 | **不成立** | Triton consumer-aware TMEM；vLLM backend-specific packing；Hexcute multi-consumer constraints |

Triton的源码尤其直接：

\[
N_{\rm warp}\text{ changes}
\Rightarrow
Layout\text{ reassigned}.
\]

fileciteturn448file0L1-L46

vLLM则直接存在：

\[
LBHNC_{\rm transfer}
\rightarrow
LBNHC_{\rm local}
\]

加block-size转换。fileciteturn439file0L1-L2

TileLang也直接解释旧的first-completer策略为什么会引入额外publish copy。fileciteturn443file0L1-L2

这些不是“我们觉得heuristic粗糙”。

而是：

\[
\boxed{\text{框架自己的repair已经提供了反例。}}
\]

---

# 十五、哪些假设目前还不能说“错误”？

另外一些只能判成“未经证明的性能proxy”。

| Proxy | 现在能说什么 | 现在不能说什么 |
|---|---|---|
| CUTLASS largest-compatible swizzle | 是structural heuristic | 不能说它经常不是最快 |
| CUTLASS 4 accumulator stages | 是经验规则 | 不能说5/3一定更好 |
| Triton min bank conflict | 是局部目标 | 不能说global runtime一定排序错误 |
| Gluon minimum inference distance | 不是performance model | 不能说选出的layout慢 |
| TVM `SSSRRSRS` candidate family | design space受限 | 不能说oracle在空间外 |
| TIRx priority-first | 没有比较合法variants runtime | 不能说priority顺序错误 |
| Hexcute CPI model | 有明确简化 | 不能说ranking fidelity低 |
| TileLang register-count | 只是proxy | 不能说IO-aware一定优 |
| vLLM preference voting | 不是系统cost最小化 | 不能说whole-model chosen layout显著差 |
| FlashInfer avg-Q / occupancy proxy | 是解析heuristic | 不能说实际regret显著 |

这正是 RQ1 必须通过实验回答的部分。

---

# 十六、现代 LLM 为什么特别容易触发 RQ1？

这里不能只说“模型越来越复杂”，需要落实到实际结构。

### Attention / Prefill

Prefill同时涉及：

\[
QK^T
\rightarrow
Softmax
\rightarrow
PV.
\]

同一个kernel/subgraph同时承受：

- GMEM coalescing；
- SMEM swizzle；
- Tensor Core fragment；
- accumulator/register pressure；
- reduction；
- async copy；
- pipeline depth。

因此只优化：

\[
BankConflict
\]

或者：

\[
RegisterCount
\]

很容易与其他局部目标发生竞争。

Triton事实上已经分别为memory、MMA、reduction、TMEM维护不同layout preference，这正是现实证据。fileciteturn465file0L1-L13 fileciteturn456file0L1-L2

---

### GQA / Decode

现代attention implementation显式存在：

\[
num\_key\_value\_groups
=
\frac{num\_attention\_heads}
{num\_key\_value\_heads}.
\]

Qwen3代码本身就按照这个关系组织attention head reuse。fileciteturn503file1L14-L32

Decode具有：

\[
S_q\ll S_{kv}
\]

而且KV cache是paged/persistent的。

此时性能排名更可能依赖：

\[
PageGeometry,
GQAGroup,
KVLength,
Occupancy,
SplitKV,
PersistentLayout
\]

而不是单纯GEMM tile proxy。

FlashInfer正是为此把GQA、head dim、CTA tile、occupancy、split-KV全部放进scheduler。fileciteturn430file0L1-L2

---

### MLA

DeepSeek-V3配置是非常好的压力案例。

当前Transformers配置中：

\[
kv\_lora\_rank=512
\]

\[
qk_{\rm nope}=128
\]

\[
qk_{\rm rope}=64
\]

\[
V_{\rm head}=128.
\]

fileciteturn508file0L1-L13

也就是说：

\[
D_{QK}=192
\neq
D_V=128
\]

并且persistent state采用latent representation。

这会直接影响：

- KV physical representation；
- copy/vector width；
- Tensor Core tile；
- register layout；
- decode work partition；
- transfer representation。

SGLang甚至为DeepSeek/DSA专门建立了backend/page-size override hierarchy。fileciteturn494file0L1-L10

---

### MoE

同一个DeepSeek-V3配置还有：

\[
n_{\rm routed\ experts}=256
\]

\[
experts/token=8.
\]

fileciteturn508file0L1-L13

其EP plan直接把expert projections组织成grouped GEMM。fileciteturn508file0L1-L13

所以单个expert的实际：

\[
M_e
\]

是动态的。

这会改变：

\[
tile\ utilization,
occupancy,
register\ amortization,
tail\ waste.
\]

因此针对固定regular shape建立的layout ranking，未必在动态expert load下保持排序。

这里不是说它“一定失败”，而是出现了明确的潜在rank-reversal regime。

---

### FP8 / FP4 / block-scale

现代低精度又引入：

\[
L_{\rm data}
\neq
L_{\rm scale}.
\]

CUTLASS block-scaled builder已经分别为A/B和SFA/SFB计算layout与pipeline footprint。fileciteturn440file0L1-L2

vLLM FlashInfer NVFP4 backend甚至改变persistent page content，加入独立FP8 scales。fileciteturn489file7L92-L102

因此只根据主tensor data layout做ranking已经不够表达全部representation cost。

---

# 十七、RQ1 真正应该比较的“信息层级”

为了不把RQ1变成泛泛的“做更好的cost model”，我建议严格定义信息层级：

\[
F_0=
\{\text{shape,dtype,legality}\}
\]

\[
F_1=
F_0+
\{\text{coalescing,vector width,bank conflict}\}
\]

\[
F_2=
F_1+
\{\text{registers,SMEM,TMEM,occupancy}\}
\]

\[
F_3=
F_2+
\{\text{instruction count,latency,instruction family}\}
\]

\[
F_4=
F_3+
\{\text{pipeline,barrier,async overlap,warp role,conversion cost}\}
\]

\[
F_5=
F_4+
\{\text{consumer context,phase,B,S_q,S_{kv},
GQA,MLA,page,MoE imbalance}\}
\]

最后：

\[
F_6=
F_5+
\{\text{real hardware measurement}\}.
\]

现有框架几乎刚好分布在这些层级上。

TIRx接近 \(F_0/F_1\)。

CUTLASS主要在 \(F_1\sim F_3\)。

Hexcute主要在 \(F_3\)，带部分 \(F_4\) 近似。

TileLang IO-aware主要在 \(F_1/F_2\)。

FlashInfer开始进入 \(F_2/F_5\)。

TVM最终进入 \(F_6\)。

这使RQ1成为一个非常清晰的问题：

\[
\boxed{
\text{性能排序从哪一级 }F_k\text{ 开始基本稳定？}
}
\]

而不是：

> “我要发明一个更复杂cost model。”

---

# 十八、RQ1 必须和 RQ2 严格隔离

这是实验设计中最容易犯的错误。

RQ1只能研究：

\[
\boxed{\text{同一个固定candidate set里的ranking error}}
\]

不能同时改变candidate generation。

定义固定候选集：

\[
C_{\rm fixed}
=
\{c_1,\ldots,c_n\}.
\]

真实oracle：

\[
c^*
=
\arg\min_{c\in C_{\rm fixed}}T(c).
\]

selector：

\[
\hat c_F
=
\arg\min_{c\in C_{\rm fixed}}\hat T_F(c).
\]

然后定义：

\[
\boxed{
SelectionRegret(F)
=
\frac{T(\hat c_F)}
{T(c^*)}-1.
}
\]

这样：

\[
\text{candidate不存在}
\]

的问题完全不属于RQ1。

它留给RQ2。

这一点必须从现在开始锁死，否则两个问题会再次混起来。

---

# 十九、RQ1 的评价不能主要看 absolute prediction error

目标不是：

\[
|\hat T-T|.
\]

最重要的至少有三个指标：

\[
\boxed{
PairwiseRankingAccuracy
}
\]

或者 Kendall-\(\tau\)；

\[
\boxed{
TopKRecall
}
\]

即真正fast candidates是否进入预测top-k；

以及最关键的：

\[
\boxed{
SelectionRegret.
}
\]

因为即使：

\[
MAPE=30\%
\]

只要candidate顺序正确，selector仍然完全可以工作。

反过来：

\[
MAPE=5\%
\]

但第一、第二名顺序经常反转，也可能实际选择错误。

因此 RQ1 本质上是：

\[
\boxed{\text{ranking sufficiency}}
\]

而不是generic GPU performance prediction。

---

# 二十、RQ1 的可证伪假设

现在可以把它写成真正的scientific hypothesis。

### H0：Local-information sufficiency

存在一个低成本局部信息集合：

\[
F_{\rm local}\subseteq F_0\cup F_1\cup F_2\cup F_3
\]

使得在现代LLM常见subgraphs、shapes和目标GPU上：

\[
SelectionRegret(F_{\rm local})
\le\epsilon
\]

且ranking稳定。

如果这个成立，那么：

\[
Pipeline,
Consumer,
RuntimeState,
HardwareMeasurement
\]

并非layout ranking所必需。

这会直接削弱RQ1继续复杂化模型的研究价值。

---

### H1：Context-sensitive ranking

存在可预测的workload regimes，使得：

\[
F_{\rm local}
\]

发生系统性rank reversal，而加入：

\[
F_4/F_5
\]

显著降低selection regret。

例如预期候选regime：

\[
\text{long-KV decode/GQA}
\]

\[
\text{asymmetric MLA}
\]

\[
\text{block-scaled FP4/FP8}
\]

\[
\text{imbalanced MoE}
\]

\[
\text{Blackwell async/TMEM/warp-specialized kernels}.
\]

如果这些regime中不存在稳定的rank reversal，这个假设被否定。

---

### H2：Measurement necessity

即使加入 \(F_4/F_5\)，仍存在：

\[
\hat T_{\rm analytic}(c_i)
<
\hat T_{\rm analytic}(c_j)
\]

但：

\[
T(c_i)>T(c_j)
\]

的系统性案例，并且只有少量真实measurement才能恢复ranking。

如果一个结构化/解析模型已经在跨shape、跨GPU上持续得到近oracle选择，那么：

\[
F_6
\]

不是必要条件。

---

# 二十一、什么结果会直接判 RQ1 “不值得研究”？

这一点必须提前定义，避免结果出来后硬解释。

如果在 Attention、GQA decode、MLA、MoE、low-precision kernels中，一个简单的 \(F_1\sim F_3\) proxy持续满足：

\[
TopKRecall\approx1
\]

并且：

\[
P95\ SelectionRegret<\epsilon
\]

同时加入pipeline/consumer/runtime信息或真实measurement只产生极小改善，那么：

\[
\boxed{\text{RQ1基本被否定。}}
\]

\(\epsilon\) 应该在正式实验前预注册；例如可以使用2%或5%这种工程上有意义的容忍区间，而不是看到结果后再定。

另外，如果出现ranking error但随机分布、无法由workload/hardware regime预测，那么也会削弱科研价值——因为那更像noise，而不是一个可以建模的规律。

---

# 二十二、什么结果会强力支持 RQ1？

反过来，如果出现下面这种结构：

```text
regular GEMM / long prefill
    local proxy ≈ oracle

long-KV GQA decode
    occupancy + runtime geometry required

MLA
    consumer + asymmetric dimensions required

FP4/block-scaled
    metadata layout + instruction/pipeline required

Blackwell warp-specialized
    register/TMEM/async interaction required

MoE small/imbalanced expert GEMMs
    runtime M distribution required
```

并且这些regime能够跨shape、跨GPU复现，那么我们就得到了真正的scientific result：

\[
\boxed{
\text{不同workload regime具有不同的minimum sufficient information set。}
}
\]

这比：

> “我们的cost model比Hexcute更准”

强得多。

因为最终结论会是：

\[
\boxed{
F^*(w,h)
=
\text{minimum information needed for low-regret layout ranking}
}
\]

。

---

# 二十三、RQ1 到这里的“问题审判”结论

现在我认为 RQ1 已经满足你要求的完整证据链：

\[
\boxed{
\text{源码规则}
\rightarrow
\text{proxy}
\rightarrow
\text{遗漏变量}
\rightarrow
\text{框架自身repair}
\rightarrow
\text{现代LLM触发条件}
\rightarrow
\text{可证伪hypothesis}
}
\]

而且它不是建立在某个框架的单一缺陷上。

证据覆盖：

- CUTLASS：resource/expert rule；
- Triton：大量局部proxy以及明确rough/arbitrary heuristic；
- Gluon：inference-distance而非performance；
- TVM：hardware measurement作为另一端；
- TIRx：priority-first；
- Hexcute：解析instruction model；
- TileLang：两套不同proxy；
- vLLM：preference aggregation；
- SGLang：model/backend policy与真实performance workaround；
- FlashInfer：runtime analytic scheduler。

因此目前可以给RQ1一个相当明确的判决：

\[
\boxed{
\textbf{RQ1 是一个成立的科研问题，但“现有方法存在显著performance regret”尚未被证明。}
}
\]

真正下一步需要证明的不是“这些heuristic很粗糙”，这一点源码已经证明得足够多；而是：

\[
\boxed{
\text{这些不同的信息删减究竟在什么条件下改变layout性能排序。}
}
\]

只有把这个问题通过固定candidate set的oracle ranking彻底回答之后，才应该进入 **RQ2：Candidate-Space Coverage**。下一轮因此应严格只做 RQ2 的同等级“问题审判”，前提是你认可现在 RQ1 的定义、证据边界和可证伪形式。

## 用户

对于凝练到每一个科研问题的 所有框架的证据链【如果上述你确实整理完了，没有的话请继续】（给出一张非常精细统计表：一个框架的每一种启发式，每一种规则是否都完全整理，每一种假设，以及在完成layout选择的每一个策略，都做了细粒度的整理统计和对比 。尤其是上述没有完成的）。也就是顺便将上述的所有回答全部统计进一张表格中

## 模型思考摘要

整合统计表

## 模型回复

我会把“是否完全整理”也作为表中的一列，而不是继续用一句“已完成”带过。对每一行我区分：**机制族已封口**、**叶分支已枚举**、**仍有架构/数据类型叶分支未逐条穷举但不改变机制类别**，这样你能直接看出哪里还能继续补源码。

## 模型思考用时

思考了 32s

## 模型回复

可以。按前面已经固定的边界——只统计会影响 **layout / mapping / persistent representation / layout-coupled instruction / pipeline / backend→representation** 的 production/default 决策——目前十个系统的主要机制族已经封口；但我不会把“机制族完整”冒充成“仓库里所有架构、dtype、model override 的每一个叶子 `if` 都逐行穷举完毕”。因此下面专门增加“完整度”列：

- **A**：该机制的 production 规则、proxy、fallback/repair 已经能够闭环；
- **B**：机制族已经完整，但还有同一机制下的 architecture / dtype / model-specific 叶分支没有逐行枚举；
- **C**：仍存在需要继续追源码才能封口的机制级问题。

就目前证据而言，下面没有 C；主要未逐条穷举的是 CUTLASS 的大量 architecture/dtype builder leaf、TIRx primitive leaf、SGLang model override leaf、vLLM backend leaf 和 FlashInfer 不同 GPU 世代 kernel specialization。这些属于 **B：机制不缺，叶子还可继续枚举**。

下面把前面所有回答压进同一张审计表。`C0=硬约束/legality`，`H=启发式`，`P=性能proxy/cost model`，`M=真实测量`，`E=显式专家决定`，`R=repair/fallback`。

| Framework | Rule ID | 决策点 / layout层级 | Production规则 / 启发式 | 类型 | 使用的输入 / proxy | 隐含假设与主要忽略变量 | 最终选择策略、fallback / repair | 对科研问题的证据 | 完整度 | 源码证据 |
|---|---|---|---|---|---|---|---|---|---|---|
| **CUTLASS/CuTe** | C-01 | GMEM major / MMA operand orientation | 从GMEM stride/layout tag推导UMMA/GMMA major；部分dtype/layout组合只能走特定major | C0/R | stride、dtype、MMA instruction legality | 合法major集合内不比较真实runtime | 非法直接template/static failure；专家可换layout | RQ2：candidate space先被instruction结构强裁剪；RQ3：storage layout与instruction不可完全分离 | A | SM100 builder结构与major/layout构造见 fileciteturn502file0L1-L2 |
| CUTLASS/CuTe | C-02 | GMEM→SMEM copy | 根据alignment、cluster shape、kernel schedule决定TMA / multicast / cp.async等copy atom | C0/R | alignment、cluster geometry、architecture | 只要结构满足，默认认为选择的copy family合理；不做候选runtime比较 | 不满足某builder specialization则走另一builder/fail；expert可显式构造 | RQ2/RQ3 | B | 多个SM100 builder family及TMA/cp.async specialization见 fileciteturn500file0L1-L2 |
| CUTLASS/CuTe | C-03 | SIMT copy thread mapping | copy thread layout倾向让更多thread覆盖GMEM major/contiguous dimension | H | contiguous dimension、alignment、tile geometry | coalescing是dominant proxy；忽略后续consumer、register pressure | builder生成固定thread-value mapping；expert override | RQ1：典型local memory proxy；RQ3：mapping与consumer可能耦合 | B | builder/copy/layout construction归入当前CollectiveBuilder证据链 fileciteturn502file0L1-L2 |
| CUTLASS/CuTe | C-04 | SMEM layout/swizzle | SMEM selector从合法swizzle/layout family中选适配MMA/copy的布局；常见偏最大合法swizzle | C0/H | dtype、major、tile shape、instruction要求 | bank-conflict / instruction friendliness足够代表性能；不看whole-kernel pipeline interaction | 不同SS/RS family使用不同selector；expert显式layout可覆盖 | RQ1：bank/swizzle proxy；RQ2：模板候选覆盖 | B | SM100 builder内独立 `SmemLayoutAtom*` selector fileciteturn502file0L1-L2 |
| CUTLASS/CuTe | C-05 | 1SM vs 2SM MMA | `KernelScheduleAuto`根据cluster/tile等合法条件选择1SM/2SM UMMA path | H/C0 | cluster divisibility、tile shape | 结构条件足够决定较好schedule；没有runtime比较 | expert指定KernelSchedule绕过Auto | RQ1/RQ2 | A | block-scaled SM100/SM103 schedule family前述源码证据 fileciteturn440file0L1-L2 |
| CUTLASS/CuTe | C-06 | mainloop pipeline stages | `StageCountAuto*`：可用SMEM减去carveout，再除per-stage footprint，尽量塞入stage | H/R | SMEM capacity、per-stage bytes、pipeline storage | 更多合法stage通常更好；不直接考虑occupancy/latency hiding拐点 | 手工 `StageCount<N>` 覆盖Auto | **RQ1核心**：resource capacity≠真实最佳stage | A | 当前builder stage公式 fileciteturn418file0L1-L2 |
| CUTLASS/CuTe | C-07 | accumulator pipeline | SM100 dense存在accumulator stage上限4，源码注释“4 stages works well” | H | accumulator/TMEM geometry + 固定经验cap | “4”能泛化到shape/workload；不测runtime | 专家schedule/stage覆盖 | **RQ1强证据：显式经验常数** | A | accumulator stage rule fileciteturn459file0L1-L30 |
| CUTLASS/CuTe | C-08 | mixed-input pipeline | 分别计算Load→Transform、Transform→MMA、Accumulator三个stage count，受SMEM/TMEM共同约束 | C0/H | SMEM、TMEM、transform footprint、MMA geometry | capacity-driven最大化近似性能；忽略具体stall/overlap | 不足2 stages则static failure；manual stage override | RQ1/RQ3：多个资源块本身已不能独立 | A | fileciteturn502file0L1-L2 |
| CUTLASS/CuTe | C-09 | block-scaled data/scale layout | A/B与SFA/SFB拥有不同SMEM/TMA layout及stage footprint | C0/R | data dtype、scale granularity、MMA instruction | data与metadata约束能通过builder局部协调 | 独立layout selector + builder specialization | RQ3/RQ4：一个算子已有多representation coupling | A | fileciteturn440file0L1-L2 |
| CUTLASS/CuTe | C-10 | epilogue layout | epilogue独立选择tile、C/D SMEM layout、swizzle与schedule | H/C0 | output dtype、tile、SMEM reuse | mainloop最优与epilogue最优可阶段化组合 | epilogue builder / expert CollectiveEpilogue | RQ3：阶段决策是否可分离 | A | epilogue selector证据 fileciteturn460file0L1-L11 |
| CUTLASS/CuTe | C-11 | global fallback | Builder不覆盖全部CuTe设计空间；专家直接构造TiledMma/TiledCopy/Layout/Pipeline | E/R | programmer knowledge | expert能够提供coherent high-performance config | 完全绕过Auto builder | RQ2 oracle/expressiveness baseline | A | 当前CollectiveBuilder能力边界见前述builder源码 |
| **Triton classic** | T-01 | GMEM distributed layout | `Coalesce`对load/store构造更coalesced encoding并插layout conversion | H | AxisInfo、contiguity、vectorization | 单memory-op coalescing改进能改善全局性能 | 插`convert_layout`；后续pass再消除/重写 | **RQ1、RQ3核心**：局部memory proxy产生跨layout代价 | A | fileciteturn465file0L1-L13 |
| Triton | T-02 | Descriptor memory vectorization | descriptor load/store有独立vectorization限制，如最多16B等 | C0/H | descriptor legality、element size | 最大合法vectorization通常最佳 | 降低vector width/改encoding | RQ1 | A | 同Coalesce/descriptor路径 fileciteturn465file0L1-L13 |
| Triton | T-03 | tensor-core layout | `AccelerateMatmul`改dot input/output layout以匹配tensor core encoding | C0/H | dot shape、dtype、arch | tensor-core compatible layout优于generic layout | 不可tensorize则保留generic dot | RQ2/RQ3 | A | 当前Passes/实现已在前轮封口 |
| Triton | T-04 | dot operand representation | `OptimizeDotOperands`重排operand layouts、硬件transpose、conversion hoisting | H | dot consumer、transpose capability、reuse | dot局部consumer决定最佳operand representation | hoist/remove conversion；fallback原layout | RQ1/RQ4 | A | 当前Triton pass族证据 |
| Triton | T-05 | layout reconciliation | `RemoveLayoutConversions`在memory-friendly Blocked和MMA-friendly encoding等之间重写/消除转换 | R/H | conversion graph、consumer type、rematerialization机会 | conversion graph可通过局部重写近似全局最优 | rematerialize / shared staging / encoding rewrite | **RQ3/RQ4直接证据** | A | 现行Passes.td/相关实现前述证据 |
| Triton | T-06 | reduction/gather locality | `OptimizeThreadLocality`调整layout减少cross-thread communication | H | reduction/gather axis、thread ownership | 最小通信≈最好runtime；忽略memory/pipeline trade-off | 重新分布，必要时conversion | RQ1 | A | 现行pass族已核对 |
| Triton | T-07 | generic SMEM swizzle | `GenericSwizzling`最小化总bank conflicts；平局看data-movement rounds | **P/H** | predicted bank-conflict ways、move rounds | bank conflict是dominant SMEM proxy | 枚举合法swizzle后取proxy最优 | **RQ1极强证据** | A | fileciteturn466file0L1-L11 |
| Triton | T-08 | TMA/descriptor SMEM encoding | Descriptor path单独枚举/优化swizzle、padding/FP4等encoding | C0/H | descriptor/TMA legality、transaction geometry | descriptor局部规则足够 | fallback至其他合法descriptor encoding | RQ1/RQ2 | A | 已归入descriptor优化链 |
| Triton | T-09 | TMEM layout | `OptimizeTMemLayouts`根据后续reduction/load/store consumer调整TMEM→warp/register mapping | H | consumer operation、reduction dimension、vector width | consumer局部信息决定TMEM layout | alternative TMEM layout + `RemoveLayoutConversions` repair | **RQ3/RQ4直接反例：layout不是shape-only** | A | fileciteturn456file0L1-L2 |
| Triton | T-10 | pipeline | `TritonGPUPipeline`把eligible loads异步化、多buffer，结合num_stages | H/E | load role、dot pipeline、user num_stages | async overlap可由pipeline transform近似决定 | 不合法则同步load/更浅pipeline | RQ1/RQ3 | A | 当前Passes.td生产pass族 |
| Triton | T-11 | warp specialization partition | `PartitionScheduling`对load/MMA/其他op分区，并带edge-size merge heuristic | H | op type、dependence edge、edge size | 固定阈值能代表跨partition通信代价 | merge partitions / 放弃warp specialization | **RQ1强证据** | A | “seemingly arbitrary size”阈值 fileciteturn447file0L1-L30 |
| Triton | T-12 | warp count/register budget | Auto warp specialization使用粗略register estimate决定warp/partition资源 | H | estimated registers、partition role | 粗略register model足以排序 | 失败/资源不合法则普通路径；可优化warps | **RQ1强证据** | A | 源码称“Extremely rough estimate” fileciteturn446file0L1-L35 |
| Triton | T-13 | warp-count→layout feedback | warp数变化后重新做layout assignment | R | num warps、distributed encoding | warp count与layout强耦合 | 重新layout，而非保留旧layout | **RQ3直接证据** | A | fileciteturn448file0L1-L46 |
| Triton | T-14 | SMEM prefetch | `Prefetch`为dot operand做shared-memory prefetch | H | dot reuse、shared operand | prefetch收益大于额外SMEM/pressure | 不满足则不prefetch | RQ1/RQ3 | A | 当前pass族 |
| Triton | T-15 | data duplication | `ReduceDataDuplication`利用distributed→shared→dotOperand路径减少重复数据 | H/R | producer/consumer graph、reuse | 额外shared stage能减少整体成本 | 无收益则原路径 | RQ3/RQ4 | A | 当前pass族 |
| Triton | T-16 | instruction ordering | `ReorderInstructions`面向register pressure/PTXAS行为调整指令顺序 | H | estimated liveness/dependency | 编译器局部ordering能代表PTXAS收益 | 后端仍可再调度 | RQ1 | A | 当前pass族 |
| Triton | T-17 | user autotune | `triton.autotune`测量用户提供的tile/num_warps/num_stages等config | **M** | 真GPU运行时间 | 搜索空间里存在好candidate | benchmark选最快；early pruning/perf model可减少测量 | RQ1 measurement endpoint；RQ2受用户candidate set限制 | A | 官方autotune证据在前述 `turn584227search0/4/1/6` |
| **Gluon** | G-01 | distributed layout ownership | tensor layout通常由programmer显式指定 | E | programmer/expert | programmer知道合理映射 | 无自动全局搜索 | RQ2：显式设计空间oracle；RQ3 ownership轴 | A | 当前Gluon layout API前述官方证据 |
| Gluon | G-02 | AutoEncoding seed propagation | Auto layout从已知seed沿IR forward/backward传播 | C0/R | producer/consumer layout equations | constraint propagation足够决定未知layout | ResolveAutoLayout；无法一致则failure | RQ3/RQ4 | A | fileciteturn471file0L1-L10 |
| Gluon | G-03 | fuzzy inference | Join/Split/reshape/transpose传播layout时增加distance | H | inference distance | “离显式seed更近”更可信 | 选择distance更小的candidate | RQ1：不是performance proxy | A | fileciteturn471file0L1-L10 |
| Gluon | G-04 | equal-distance conflict | 非零等distance可用stable ordering/hash；0-distance explicit冲突直接error | H/C0 | distance、seed identity | deterministic resolution优先于性能选择 | error要求用户修复 | RQ1/RQ4 | A | 同上 |
| Gluon | G-05 | CoalescedEncoding | 对memory op用AxisInfo构造coalesced seed，再前后传播 | H | AxisInfo、num warps、shape | memory coalescing足够决定该auto layout | seed propagation / concrete encoding | RQ1 | A | fileciteturn472file0L1-L13 |
| Gluon | G-06 | CGA restriction | 当前auto coalesced路径 `numCTAs==1` | C0 | CTA count | 暂不覆盖multi-CTA auto layout | multi-CTA需其他显式路径 | RQ2 coverage gap | A | fileciteturn472file0L1-L13 |
| Gluon | G-07 | shared MMA layout | NV MMA shared layout default选兼容swizzle/layout | H/C0 | MMA instruction、dtype、shape | compatible large swizzle/message efficiency是合理proxy | 显式shared layout可覆盖 | RQ1 | B | 当前layout API机制已完整，arch叶子可继续枚举 |
| Gluon | G-08 | warp specialization | warps/register budgets/dataflow通常显式 | E | programmer | expert选择 | 可配合Gluon warp-specialize API | RQ3 ownership | A | 官方Gluon warp specialization前述证据 |
| Gluon | G-09 | autotune | 可复用Triton autotune搜索显式meta-params | M | runtime | candidate set足够 | benchmark最快 | RQ1/RQ2 | A | 官方Gluon/autotune前述证据 |
| **TVM MetaSchedule/Relax** | V-01 | CUDA tiling skeleton | DefaultCUDA固定`SSSRRSRS` multi-level tiling结构 | H | operator structure | 该tiling grammar覆盖大多数高性能GPU schedule | search只在该grammar内展开 | **RQ2核心** | A | fileciteturn474file0L1-L2 |
| TVM | V-02 | tile binding | 默认把前几个tile level绑定blockIdx/vthread/threadIdx | H | schedule template | 固定层级绑定结构足够通用 | ScheduleRule生成候选 | RQ2 | A | fileciteturn474file0L1-L2 |
| TVM | V-03 | vector load | `vector_load_lens={1,2,3,4,8,16}` | H/search set | vector width | 好配置存在于有限集合 | hardware measurement最终排序 | RQ2候选覆盖；RQ1最终measurement | A | fileciteturn474file0L1-L2 |
| TVM | V-04 | read reuse | CUDA默认read cache必须在level 4 shared | H | data reuse pattern | shared read cache placement结构可固定 | schedule rule | RQ2/RQ3 | A | 同上 |
| TVM | V-05 | write reuse | 默认write reuse必须level 3 local | H | output reuse | local write cache placement可固定 | schedule rule | RQ2/RQ3 | A | 同上 |
| TVM | V-06 | CrossThreadReduction | thread extents只从4…512离散集合选 | H/search set | reduction extent | 好thread mapping在预定义集合中 | measurement选 | RQ2 | A | 同上 |
| TVM | V-07 | unroll | unroll steps从固定离散集合选择 | H/search set | loop body | 有效unroll在有限set | measurement | RQ2 | A | 同上 |
| TVM | V-08 | AutoBind | CUDA thread extent从32…1024等候选中生成 | H/search set | loop extent、GPU limit | 离散标准block sizes足够 | search+measurement | RQ2 | A | 同上 |
| TVM | V-09 | TensorCore intrinsic family | TensorCore rule从预注册WMMA/MMA intrinsic groups tensorize | C0/H | dtype、shape、intrinsic pattern | 注册intrinsic集合覆盖目标 | 不匹配可回普通CUDA schedule | RQ2 | A | fileciteturn474file0L1-L2 |
| TVM | V-10 | TensorCore software pipeline | 不同TensorCore schedule family显式区分software pipeline on/off与write reuse | H/search family | intrinsic family、reuse policy | 有限结构families可覆盖好pipeline | measurement选择 | RQ1/RQ2 | A | 同上 |
| TVM | V-11 | Postproc legality | DisallowDynamicLoop、RewriteCooperativeFetch、RewriteReductionBlock、VerifyGPUCode、RewriteTensorize等 | C0/R | generated schedule IR、GPU limits | legality可与ranking阶段分离 | reject invalid candidate | RQ2 candidate legality | A | fileciteturn475file0L1-L18 |
| TVM | V-12 | `RewriteLayout` | 对`layout_free_buffers`根据consumer access建议IndexMap并改physical buffer layout | H/A | consumer indexing | consumer访问模式足以决定layout rewrite | 可插global cache/preprocessing | RQ3/RQ4 | A | fileciteturn53file0L1-L7 |
| TVM | V-13 | Cost model | XGB/其他cost model预估候选性能 | P/ML | schedule features +历史measurement | features能外推ranking | search strategy继续探索 | **RQ1直接对照组** | A | 官方MetaSchedule证据前述 `turn450339search0/1` |
| TVM | V-14 | Hardware Runner | Builder/Runner在目标硬件实际编译运行 | **M** | runtime | 真测量作为ground truth | Database持久化最佳schedule | RQ1 measurement endpoint | A | 前述MetaSchedule官方证据 |
| TVM Relax | V-15 | graph fusion boundary | `FuseOps/FuseTIR/FuseOpsByPattern`先决定graph/subgraph，再进入低层schedule | H/A | graph pattern、fusibility | graph fusion与low-level tuning可以阶段化 | 不融合/后端pattern dispatch | **RQ3核心** | A | Relax transform官方证据前述 `turn223392search*` |
| **TIRx** | X-01 | primitive dispatch | `(Op,target)`注册多个Variant | C0/E | op、target | 高性能实现可编码为有限variant registry | 没variant则verification/failure | RQ2 | A | fileciteturn420file0L1-L10 |
| TIRx | X-02 | variant ranking | variants按priority降序，逐个predicate检查，第一个成功者被采用 | **H** | priority、predicate | designer priority≈performance ordering | candidate fail则尝试下一个 | **RQ1强证据** | A | fileciteturn420file0L1-L10 |
| TIRx | X-03 | explicit dispatch | 用户可指定variant，绕开priority | E/R | dispatch hint | expert override有价值 | 只考虑指定variant | RQ1/RQ2 | A | 同上 |
| TIRx | X-04 | sync copy family | forced vector / auto vector GMEM↔SMEM / reg copy / ld-stmatrix / fallback | C0/H | scopes、layout、alignment、target | finite implementation families足够 | priority/predicate/fallback | RQ2 | A | fileciteturn477file0L1-L10 |
| TIRx | X-05 | async copy family | ldgsts、TMA、DSMEM、tcgen05 cp/ldst等 | C0/H | exec scope、storage scope、arch、layout | priority表达性能 preference | next variant | RQ1/RQ2 | A | fileciteturn478file0L1-L10 |
| TIRx | X-06 | copy predicate | alignment/layout/storage/exec-scope predicate筛掉非法copy variants | C0 | exact hardware constraints | legality与performance基本可分开 | reject→next variant | RQ2 | A | tcgen05示例priority/predicate fileciteturn479file0L1-L15 |
| TIRx | X-07 | GEMM / GEMM async | primitive dispatch根据layout/storage/target选择WGMMA/tcgen05等implementation | C0/H | operand layout、exec scope、target | priority order能代表同类合法variant速度 | fallback variant | RQ1/RQ2/RQ3 | B | family目录完整；每个arch leaf未逐条列出 fileciteturn476file0L1-L10 |
| TIRx | X-08 | reduction | reduction注册不同warp/warpgroup/TMEM实现并用priority | H/C0 | layout、exec scope、arch | priority≃性能 | next variant | RQ1 | B | 已确认priority-based production机制 |
| TIRx | X-09 | unresolved primitive verifier | lowering后若仍有tile primitive未dispatch则报错 | C0/R | unresolved op | 所有低层选择必须显式resolve | compile-time failure | RQ2 legality | A | dispatch/verifier机制已核对 |
| **Hexcute** | H-01 | Logical shape inference | 反复传播shape直到全部resolve，不一致error | C0 | tensor op equations | shape inference纯legality | error | RQ2基础约束 | A | fileciteturn481file0L1-L2 |
| Hexcute | H-02 | Logical layout inference | 根据stride/layout关系推导逻辑vector/matrix layout | C0 | logical strides | logical layout可先于hardware mapping独立求解 | conflict error | RQ3分层假设 | A | 同上 |
| Hexcute | H-03 | Copy task mapping | 使用 \(f\circ p^{-1}=g\circ q^{-1}\) 约束TV layouts | C0 | copy instruction TV layout | instruction constraint足够推导mapping | constraint propagation/backtrack | RQ2/RQ3 | A | fileciteturn481file0L1-L2 |
| Hexcute | H-04 | MMA task mapping | 用M/N/K三组layout相等约束推导A/B/C/D TV layouts | C0 | MMA instruction TV layout | hierarchical TV layout能表达合法mapping | backtrack | RQ2 | A | 同上 |
| Hexcute | H-05 | elementwise mapping | elementwise inputs要求shape/nonzero strides兼容 | C0 | tensor layouts | 共layout利于elementwise | conflict/backtrack | RQ4 | A | 同上 |
| Hexcute | H-06 | multiple instruction variants | 一个copy/MMA可对应多个instruction，理论可DFS枚举全部合法解 | Search/C0 | instruction library + constraints | exhaustive legal enumeration可作为candidate source | DFS | RQ2 ideal candidate coverage | A | 同上 |
| Hexcute | H-07 | search pruning | 源码允许simple heuristic / beam search prune DFS | H | partial solution | 可在不明显损伤coverage时裁剪 | beam/heuristic | **RQ2核心** | A | 同上 |
| Hexcute | H-08 | multi-consumer SMEM layout | 同一shared tensor所有copy constraints一起unify | C0 | all consumers | 单一shared representation可同时满足consumers | conflict→backtracking | **RQ4直接证据** | A | 同上 |
| Hexcute | H-09 | unresolved stride completion | constraints全部满足后，对剩余未知stride使用heuristic补齐 | H | partial memory layout | 任一合法completion性能差异有限/可被后续选择修正 | heuristic completion | **RQ1/RQ2** | A | 同上 |
| Hexcute | H-10 | instruction match | instruction selection检查scope、alignment、TV layout、loop structure等 | C0 | exact layout+scope | legality predicates可筛出合理candidate | 无match则其他instruction | RQ2 | A | fileciteturn483file0L1-L2 |
| Hexcute | H-11 | SMEM bank repair | 为shared tensor枚举swizzle，最小化所有copy bank-conflict ways总量 | P/H/R | bank conflict across consumers | bank conflicts是SMEM ranking的重要proxy | 选最佳swizzle | **RQ1/RQ4** | A | fileciteturn484file0L1-L2 |
| Hexcute | H-12 | TMA swizzle | TMA路径特别枚举4种合法swizzle并考虑其他consumer冲突 | P/R | TMA legal modes + other accesses | 多consumer bank cost可加和 | 最小总conflict | RQ1/RQ4 | A | 同上 |
| Hexcute | H-13 | WGMMA immutability | WGMMA shared operand的swizzle可能由instruction固定，禁止自由修复 | C0 | WGMMA layout | instruction constraint高于bank heuristic | 标记immutable | RQ3 | A | 同上 |
| Hexcute | H-14 | analytic latency model | \(cost=\#inst\times CPI\)，分independent/dependent CPI | **P** | instruction count、microbenchmarked CPI | instruction issue/completion足够保持ranking | 取预测cost最小candidate | **RQ1最直接证据** | A | fileciteturn482file0L1-L2 |
| Hexcute | H-15 | overlap model | copy/MMA按理想化公式假设充分overlap | P假设 | issue/complete CPI | pipeline通常能fully overlap | 无动态repair | **RQ1可证伪假设** | A | 同上 |
| Hexcute | H-16 | omitted barrier/bank/address cost | 主cost model明确不完整计算bank-conflict penalty、`cp_async_wait_group`、`mbarrier`，并忽略tensor-manipulation address arithmetic | P假设 | 简化模型 | 这些项不会频繁改变candidate ranking | 独立bank pass部分弥补bank conflict | **RQ1关键遗漏变量** | A | fileciteturn482file0L1-L2 |
| **TileLang** | L-01 | inference mode hierarchy | LayoutInference按Strict→Common→Free逐层尝试 | C0/Search | layout constraints | 越严格规则优先，free是最终搜索层 | 前层失败进入下一层 | RQ2 | A | 当前layout inference证据链前述 |
| TileLang | L-02 | thread geometry legality | thread extent必须编译期可决定 | C0 | static thread geometry | 动态thread mapping不进入候选空间 | reject | RQ2 | A | 前述 `turn443file0` |
| TileLang | L-03 | free-mode candidates | Free mode生成多个完整layout attempts，而非只推一个解 | Search | constraint solutions | 候选可由proxy排序 | cost model | RQ2/RQ1接口 | A | 当前layout inference实现 |
| TileLang | L-04 | default `register-count` | 默认按fragment总register slots排序 | **P/H** | register slots | 较少register通常意味着更好occupancy/runtime | tie/attempt ordering | **RQ1非常强证据** | A | fileciteturn487file0L1-L13 |
| TileLang | L-05 | optional `io-aware` | 估算fragment↔global的vector width、warp coalescing、bytes moved；register作tie-break | **P** | vectorization、global traffic、registers | memory traffic模型能改善ranking | 可切换cost model | **RQ1天然A/B对照** | A | fileciteturn487file0L1-L13 |
| TileLang | L-06 | IO time approximation | statement cost近似 \(T(S)=\max(BW,Issue)\)，最后求和 | P | transactions、issue depth | statement独立可加；cache/async/compute secondary | symbolic score | RQ1可证伪 | A | fileciteturn486file0L1-L2 |
| TileLang | L-07 | opaque-layout penalty | non-affine/swizzle/non-bijective等模型无法解析时按worst-case收费 | R/P | model evaluability | 不应因“模型看不懂”得到低cost | conservative worst case | RQ1模型robustness | A | fileciteturn486file0L1-L2 |
| TileLang | L-08 | ReducerDstSteering | finalize reducer destination优先决定ownership，避免consumer先完成layout造成额外publish copy | H/R | producer-consumer dependency | first-completer策略可能错误 | steering ownership | **RQ4直接证据** | A | 前述 `turn443file0` |
| TileLang | L-09 | reducer widening | layout冲突按unset→narrow→wide单调放宽 | R | consumer compatibility | widening是安全repair | 逐级放宽 | RQ4 | A | 前述 `turn443file0` |
| TileLang | L-10 | replicated fallback | 无法统一时生成universally-readable replicated representation | R | unsatisfied consumer constraints | replication换性能/空间换兼容性 | replicate | **RQ4 representation specialization证据** | A | 前述 `turn443file0` |
| TileLang | L-11 | alias propagation | alias buffers自动继承/传播layout | C0/R | alias relation | alias共享physical representation | propagate | RQ4 | A | 前述 `turn443file0` |
| TileLang | L-12 | cross-dtype alias reshape | 根据storage-bit ratio reshape layout，处理FP4等sub-byte alias | C0/R | bit widths、alias geometry | bit-level storage relation可精确转换 | reshape | RQ3/RQ4 | A | 前述 `turn443file0` |
| TileLang | L-13 | warp specialization ordering | producer/consumer warp-specialization在LayoutInference之前物化，使layout inference看到角色/branch结构 | A/H | TMA pipeline、dependency graph | 先定pipeline角色、后选layout是合理边界 | materialize TMA/barriers，再layout inference | **RQ3边界顺序核心证据** | A | fileciteturn512file0L1-L2 |
| TileLang | L-14 | auto WS limits | 当前producer/consumer WS只覆盖纯TMA等受限结构 | C0 | pipeline pattern | 结构外candidate不进入auto WS | 不做WS/手写 | RQ2 coverage | A | fileciteturn512file0L1-L2 |
| **vLLM** | VL-01 | persistent stride-order space | KV cache layout枚举LBHNC/LBNHC/LHBNC/BLHNC/BLNHC/BHLNC | Search/C0 | physical stride order | 有限枚举覆盖backend需要的persistent orders | backend support filtering | RQ2/RQ4 | A | current KV layout enum前述证据 |
| vLLM | VL-02 | backend support sets | 每backend声明`ordered supported_kv_cache_layouts()` | C0/H | backend kernel addressing | backend自己最清楚可行/偏好layout | unsupported layout剔除 | RQ2 | A | 多backend实例 fileciteturn488file0L1-L12 |
| vLLM | VL-03 | global intersection | 所有相关backend layout支持集合求intersection | C0 | backend sets | 一个model-wide representation应同时服务所有consumer | 无intersection则error/配置失败 | **RQ4核心证据** | A | current resolver前述证据 |
| vLLM | VL-04 | first-choice voting | surviving candidates按各backend的第一偏好“投票”排序 | **H** | ordered backend preference | 第一偏好票数能代表whole-model性能 | 票数/ordered preference | **RQ1非常强证据** | A | current resolver前述证据 |
| vLLM | VL-05 | mixed HNC restriction | mixed spec要求block-compact等额外layout property | C0 | KV spec geometry | correctness/zero-copy优先 | restrict set | RQ2 | A | current resolver/layout properties |
| vLLM | VL-06 | user layout override | `VLLM_KV_CACHE_LAYOUT`显式指定，但必须属于兼容集合 | E/C0 | user choice | expert override | incompatible直接error | RQ1/RQ2 | A | resolver/env证据前述 |
| vLLM | VL-07 | connector preference | connector/NIXL可以把transfer偏好加入layout resolution | H | transfer backend | transfer是persistent representation的重要consumer | 兼容时提高优先级 | **RQ3/RQ4** | A | NIXL链前述证据 |
| vLLM | VL-08 | one model-wide layout | resolver最终把一个KV stride layout写入整个model cache config | Architectural assumption | all backend constraints | 不同phase/workload下一个static layout近似足够 | boundary conversion而非per-request layout | **RQ4/RQ3可证伪假设** | A | current resolver证据 |
| vLLM | VL-09 | `customize_spec()` | backend可改变K/V/scale在page内部的content packing | C0/H | kernel packing requirement、quant mode | backend-specific internal representation应由backend定义 | per-backend spec rewrite | RQ3/RQ4 | A | API说明及backend实例 fileciteturn489file2L27-L38 |
| vLLM | VL-10 | quantized content variants | Triton inline scale、TurboQuant K+V one slot/head、FlashInfer NVFP4 packed data+FP8 scale等 | C0/H | quantization mode | quant metadata/layout必须协同 | customize spec | RQ4 | B | fileciteturn489file3L40-L50 fileciteturn489file5L66-L76 fileciteturn489file7L92-L102 |
| vLLM | VL-11 | preferred block size | backend独立提供page/block-size preference，如64/256等 | H | backend kernel/page constraints | backend静态preferred page接近性能最优 | 用户显式block size可覆盖 | **RQ1/RQ3** | A | fileciteturn490file0L1-L11 fileciteturn490file3L34-L43 |
| vLLM | VL-12 | backend selection | CUDA/ROCm按backend priority，并逐个`validate_configuration` | H/C0 | head size、dtype、KV dtype、MLA/sparse/sliding等selector config | ordered backend priority能代表性能 | invalid→next backend | RQ1/RQ2 | A | CUDA priority/validation fileciteturn499file0L1-L26 fileciteturn499file1L27-L44 |
| vLLM | VL-13 | transfer-local conversion | NIXL receive path支持layout及block-size转换；transfer representation可以不同于local compute representation | R | network transfer preference、local backend | conversion成本可能小于统一representation损失 | boundary conversion | **RQ3/RQ4最强证据之一** | A | 前述NIXL证据 `turn439file0` |
| **SGLang** | SG-01 | generic MHA Hopper | compatible Hopper MHA优先FA3，但spec decode/top-k/page等条件可阻止 | H/C0 | GPU gen、spec mode、page | architecture+feature rule能代表backend performance | fall through其他backend | RQ1/RQ2 | A | generic policy证据前述 `turn428file0`/`turn492file1` |
| SGLang | SG-02 | SM100 MHA | compatible Blackwell优先TRTLLM MHA；不兼容条件转其他backend | H/C0 | SM100/103、spec conditions、page | default backend priority | fallback | RQ1 | A | fileciteturn492file1L17-L32 |
| SGLang | SG-03 | asymmetric K/V | asymmetric KV维度等情况转FA4而非TRTLLM MHA | H/C0 | K/V geometry | backend specialized capability比generic default重要 | alternate backend | RQ1/RQ3 | A | generic selector链已封口 |
| SGLang | SG-04 | HIP | HIP默认AITER系列backend | H | platform | vendor-specific backend是最好默认 | unsupported→Triton等 | RQ1 | A | generic selector |
| SGLang | SG-05 | MPS | MPS使用torch native | C0/H | platform | capability限制 | native fallback | RQ2 | A | generic selector |
| SGLang | SG-06 | FlashInfer eligibility | 无attention sinks且FlashInfer可用时可选FlashInfer；否则Triton fallback | H/C0 | feature support、library availability | compatibility后priority足够 | Triton fallback | RQ1/RQ2 | A | generic selector |
| SGLang | SG-07 | MLA separate policy | MLA有与MHA不同的Hopper/Blackwell/HIP backend规则 | H | MLA model family、head geometry、platform | attention family需要独立policy | dedicated MLA backends/Triton fallback | RQ1/RQ3 | A | generic model-override链 |
| SGLang | SG-08 | prefill/decode split | prefill和decode backend可分别resolve；相同时合并，否则hybrid backend | H/Architecture | phase | phase-specific backend差异足够重要 | hybrid wrapper | **RQ3/RQ4直接证据** | A | fileciteturn504file0L1-L18 fileciteturn506file0L1-L18 |
| SGLang | SG-09 | model override registry | architecture-specific constant/callable/predicate overrides在generic policy之上修正 | H/R | exact model architecture/config | generic policy不足以覆盖所有model | last-writer/ordered override体系 | **RQ1非常强的框架自修复证据** | A机制 / B叶子 | override体系 fileciteturn492file0L1-L16 |
| SGLang | SG-10 | observed regression repair | FlashInfer某版本在Hopper存在attention performance regression，默认切FA3 | R/H | library version + platform | version-specific performance不能完全由静态shape规则预测 | policy patch | **RQ1直接现实反例** | A | 前述 `turn450file0` |
| SGLang | SG-11 | DeepSeek DSA ROCm page | preshuffle paged-MQA不可用时page=1；可用时GPU page=64 | H/C0 | ROCm、Triton/AITER capability | backend implementation决定page geometry | legacy fallback=1 | RQ1/RQ3 | A | fileciteturn494file0L1-L10 |
| SGLang | SG-12 | DeepSeek DSA XPU page | XPU page=128 | H | platform | platform-specific optimal/required geometry | override | RQ1 | A | 同上 |
| SGLang | SG-13 | DeepSeek SM100 MLA backend | 普通DeepSeekV3在SM100且未显式设置时可默认`trtllm_mla` | H | model family+SM100 | model×hardware决定backend | explicit user choice不覆盖 | RQ1/RQ3 | A | fileciteturn494file0L1-L10 |
| SGLang | SG-14 | Qwen3.5 SM100 coupled backend/page | model override可联动`trtllm_mha`与page=64 | H/R | exact model + hardware | backend与page不可独立选择 | model override | **RQ3直接证据** | A | fileciteturn495file0L19-L35 |
| SGLang | SG-15 | DSA KV dtype/default split resolution | DSA有单独KV-cache dtype default与split backend post-process | H/R | model/quant/backend | representation dtype是backend policy组成部分 | post-process override | RQ3/RQ4 | A机制 / B全部model叶子 | fileciteturn505file2L33-L50 fileciteturn505file3L51-L66 |
| SGLang | SG-16 | MoE/FP8 runner override | 某些model/quant组合把MoE/FP8 GEMM backend从auto修正为DeepGEMM等 | H/R | model family、MXFP8、JIT availability | generic MoE backend policy不足 | model override | RQ1/RQ3；MoE触发证据 | B | HYV4 MXFP8逻辑 fileciteturn494file0L1-L10 |
| **FlashInfer** | FI-01 | API KV layout contract | attention kernels接受有限KV layout形式（如NHD/HND等family），persistent/container layout与kernel path绑定 | C0/E | API/kernel family | 有限layout family足够 | caller选择兼容layout | RQ2/RQ4 | B | production attention目录当前结构 fileciteturn496file0L1-L2 |
| FlashInfer | FI-02 | FA2 prefill CTA-Q | 用`avg_packed_qo_len`、QK/VO head dim、KV dtype bytes决定CTA query tile | **H** | batch平均Q长度、dims、dtype | batch平均值足以代表request分布 | 从离散CTA-Q family选 | **RQ1强证据** | A | 前述 `turn429file0/1` |
| FlashInfer | FI-03 | discrete Q tile family | kernel family只覆盖CTA-Q `{16,32,64,128}`等离散组合 | C0/Search set | head dim、arch | 高性能点落在离散families | 选择合法family | RQ2 | A | 前述 `turn430file0` |
| FlashInfer | FI-04 | GQA specialization | GQA group size从有限编译kernel groups `{1,2,3,4,6,8...}`适配 | C0/H | num Q heads / KV heads | 常见GQA ratios可离散专门化 | generic/其他kernel fallback | RQ2、现代GQA触发 | B | 前述 `turn430file0` |
| FlashInfer | FI-05 | vector width | Vec family `{1,2,4,8,16}`等按head/dtype/alignment选择 | C0/H | alignment、dtype、head size | 最大/合适vector width接近最佳 | lower-width fallback | RQ1/RQ2 | A | 前述 `turn430file0` |
| FlashInfer | FI-06 | register-pressure restriction | 大QK/VO head dimension减少允许的CTA-Q/MMA tile，以控制fragment/register pressure | H/C0 | head dimension、estimated resources | head dim是register pressure主要proxy | 选更小tile | **RQ1** | A | 前述 `turn431file1` |
| FlashInfer | FI-07 | architecture stage defaults | decode/prefill pipeline stage数受compute capability/default parameter family控制 | H | GPU generation | architecture-level固定stage能泛化shape | architecture specialization | RQ1 | B | default decode/prefill params当前目录 fileciteturn496file0L1-L2 |
| FlashInfer | FI-08 | split-KV enablement | 查询真实CUDA occupancy；若batch×heads已填满SM则不split，否则考虑split | **P/runtime geometry** | occupancy、batch、heads | occupancy不足是split-KV主要收益来源 | split/no-split | **RQ1：比静态layout规则多一层runtime context** | A | 前述split-KV scheduler证据 |
| FlashInfer | FI-09 | split-KV chunk search | 在KV chunk尺寸上binary-search/规划以增加parallel work同时控制合并开销 | H/P | KV length、occupancy、work count | work parallelism proxy能代表runtime | planner决定chunk | RQ1 | A | 前述scheduler证据 |
| FlashInfer | FI-10 | Blackwell work scheduler | 每个 `(batch,head,Q tile)` 按KV work cost分配给当前cost最低bucket | **P/H** | Q tile、KV length、causal extent | work≈KV tokens；bucket load balance≈runtime balance | greedy min-cost bucket assignment | **RQ1强证据** | A | fileciteturn498file0L1-L10 |
| FlashInfer | FI-11 | Blackwell tie-break | bucket cost相等时用bucket index确定deterministic winner | H | estimated work | 同cost候选性能近似相同 | deterministic tie | RQ1细节 | A | fileciteturn498file0L1-L10 |
| FlashInfer | FI-12 | kernel-family fallback | 不满足某专门kernel geometry时进入其他prefill/decode/MLA/Blackwell family | R | shape、arch、dtype | specialization边界可以离散划分 | alternate kernel family | RQ2 | B | attention目录含decode/prefill/MLA/Hopper/Blackwell独立families fileciteturn496file0L1-L2 |

这张表给出了现在更严格的“完成度”结论。

**已经真正 A 级封口的是机制本身。** Triton 的局部layout passes、Gluon AutoLayout、TVM DefaultCUDA/MetaSchedule、Hexcute inference+cost model、TileLang free-mode ranking与repair、vLLM resolver机制、TIRx dispatcher语义，都已经能从“输入 → rule/proxy → output → fallback”闭环解释。它们已经足以支撑科研问题，而不再只是框架简介。

**仍然是 B 级、可以继续逐叶穷举的主要区域有五块。** 这不表示科研证据链缺失，而表示如果你的目标是做一个“source-code census appendix”，还可以继续往下枚举：CUTLASS 各SM90/SM100/SM103/SM120、dense/sparse/block-scaled/mixed-input 的每个builder specialization；TIRx每一个registered variant的priority/predicate表；vLLM每一个attention backend的`supported_kv_cache_layouts/customize_spec/get_preferred_block_size`；SGLang所有model override文件中的backend/page/KV-dtype/MoE runner分支；FlashInfer Hopper/Blackwell/MLA/decode/prefill每一组compile-time kernel parameter leaf。

但这五块目前**没有暴露出新的selection paradigm**。它们仍分别落入：

\[
\text{CUTLASS: structural/resource rule}
\]

\[
\text{TIRx: priority+predicate}
\]

\[
\text{vLLM: compatibility+preference negotiation}
\]

\[
\text{SGLang: expert policy+model override}
\]

\[
\text{FlashInfer: specialized kernel family+analytic/runtime scheduling}.
\]

因此，从“科研问题凝练”的标准看，证据链现在已经足够完整；从“逐文件逐叶源码审计”的标准看，还可以继续把上述五个 B 区域展开成 appendix 级 census。

更重要的是，这张总表现在把四个 RQ 的来源区分开了：

\[
\boxed{\text{RQ1 = 同一candidate set内，proxy是否足以正确排序}}
\]

证据主要来自 CUTLASS C-06/C-07、Triton T-07/T-11/T-12、TIRx X-02、Hexcute H-14～H-16、TileLang L-04～L-07、vLLM VL-04/VL-11/VL-12、SGLang SG-01～SG-16、FlashInfer FI-02/FI-06/FI-08～FI-10，以及 TVM 的真实measurement作为对照。

\[
\boxed{\text{RQ2 = candidate generator是否把好解排除在外}}
\]

证据主要来自 CUTLASS builder families、Triton compiler encodings、Gluon `numCTAs==1` auto限制、TVM固定schedule grammar、TIRx有限Variant registry、Hexcute pruning、TileLang inference modes、vLLM有限persistent layout enum、FlashInfer有限kernel families。

\[
\boxed{\text{RQ3 = 哪些decision boundary真正可分离}}
\]

最直接的源码反例是 Triton “warp数改变→重新layout”、TMEM consumer-aware relayout、CUTLASS mainloop/accumulator/epilogue独立但资源耦合、TVM Relax fusion→task tuning阶段化、TileLang warp-specialization在LayoutInference前、vLLM backend/page/layout/transfer representation联动、SGLang prefill/decode split。

\[
\boxed{\text{RQ4 = multi-consumer tensor应该统一representation还是specialize/convert/replicate}}
\]

最强证据来自 Hexcute多consumer constraint unification、TileLang reducer steering+widen+replicate、vLLM model-wide KV layout与NIXL boundary conversion、Triton conversion/rematerialization/TMEM consumer adaptation。

所以现在我不会再把“所有框架证据”散落在不同回答里。**上表可以作为后续所有RQ审判的统一证据索引**。后面审RQ2、RQ3、RQ4时，应只引用这张表中的具体 Rule ID，再向下展开源码，而不是重新做一轮泛泛的框架介绍。

## 用户

所有框架的证据链【如果上述你确实整理完了，没有的话请继续】（给出一张非常精细统计表：一个框架的每一种启发式，每一种规则是否都完全整理，每一种假设，以及在完成layout选择的每一个策略，都做了细粒度的整理统计和对比 。   这一步现在完成了吗，没有完成请继续补充表格

## 模型回复

还不能把它定义为“逐叶完全穷举完成”。上一张表已经把主要 **selection mechanism family** 封口，但仍有几块只到“机制族完整、叶分支未逐项枚举”。我会继续补，重点把这五块拉到同一粒度：**CUTLASS architecture/builder leaves、TIRx primitive variants、vLLM backend persistent-policy leaves、SGLang model overrides、FlashInfer scheduler/kernel leaves**；同时再反向检查 Triton/Gluon/TVM/Hexcute/TileLang 是否还有漏掉的 production/default 决策入口。

## 模型工具调用参数

{"paths":["GitHub"],"query":"search_code"}

## 模型思考摘要

梳理研究路径
梳理了研究路径

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

补充检索 CUTLASS、Triton 与推理框架实现 evidence

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

优化 GitHub 搜索查询

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

补充KV Cache决策规则与边界

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 supported_kv_cache_layouts 相关结果

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 get_preferred_block_size 调用位置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检查模型覆盖字段及目录内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查看模型覆盖注册文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

全局检索模型覆盖声明字段

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Apache TVM 中的 TIRx 优先级实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索所有 register_tile_primitive 用法

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Apache TVM 中的 priority=10 结果

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 FlashInfer 分块与占用率逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 avg_packed_qo_len 相关代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 CUDA 占用率相关实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 FlashInfer 中调度成本计算逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 CUTLASS 叶族与四级累加器阶段配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 CUTLASS SM100 构建器相关代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 AccumulatorPipelineStageCount 定义

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理并补全注意力优化框架实测规则

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

补充整理 Triton 启发式规则

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索“Current heuristic”相关代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

补充各框架布局启发式规则

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 中的自动编码解析实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检查 InferLayoutUtils 启发式精确性

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析 InferLayoutUtils 布局推导实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检查 Gluon 合并编码推断实现

## 模型思考摘要

补充检索并梳理注意力优化框架启发式实现证据

## 模型回复

我会继续补，而不是现在宣布完成。刚才反查已经又发现 Triton 里几条此前没进主表的 layout-related heuristic（例如 `AccelerateMatmul` 的 packed-K 判断、`LinearLayout` 的 warp映射 heuristic、某些 fused-attention conversion 规则），说明上一张总表还不是“逐规则封口”。接下来我会把每个框架再做一次 **关键词反向审计**：`heuristic / auto / default / priority / prefer / fallback / swizzle / layout / vector / stage / dispatch`，把新发现逐条补进主表并重新标注完成度。

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

补充完善所有框架证据链统计

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 sm100_common 中的 Auto 辅助函数

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 TileLang 布局推断策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计 SGLang 配置覆盖字段声明

## 模型思考摘要

继续核对规则
继续核对规则

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

整理各框架启发式规则审计表

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 中的最小范数解启发式

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索精确短语“heuristic here is as follows”

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审计 CUTLASS 中的启发式实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 CUTLASS 启发式算子与调度策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

补充审计 CUTLASS 调度公式与遗漏表格

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 get_rasterization_order 相关代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 CUTLASS 中 RasterOrder 相关逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 CUTLASS 算子启发式文档

## 模型思考摘要

继续完善审计
补充审计条目
完善并补充审计条目

## 模型思考摘要

完成GPU调优框架证据、启发式与调度逻辑审计

## 模型回复

没有完成。按你现在要求的“**每一个 production/default layout 决策规则都进入目录**”标准，刚才的反向审计又发现了此前主表遗漏的生产规则：CUTLASS 的 operator-level `nvidia-matmul-heuristics`、tile raster/grid swizzle、CuTe cooperative-copy permutation；Triton 的 `LinearLayout`、packed-K matmul、pipeliner 与 fused-attention 特化 heuristic；SGLang 的 declarative-resolution 依赖规则等。  
所以前一版“机制族已封口”可以保留，但“规则逐项已经完全整理”这个结论需要撤回。下面先把这些新发现补回总目录，并重新标记每个框架还差什么。

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 CUTLASS 算子启发式文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索并评估矩阵乘法性能排序器

## 模型思考摘要

检索并评估CUTLASS算子启发式排序器

## 模型思考用时

思考了 5m 57s

## 模型回复

没有。按照你现在要求的严格标准——

\[
\boxed{
\text{每个框架所有 production/default 路径中的 layout 相关规则、启发式、假设、候选生成、排序、fallback 都逐项登记}
}
\]

——上一版仍然不能称为最终完成。刚才的反向审计又找到几类此前漏掉的 production 规则，其中 CUTLASS 和 Triton 的遗漏尤其重要。因此我会继续补，不进入新的科研问题审判。

这次最重要的修正有两个。

第一，CUTLASS 不能再只统计 `CollectiveBuilder`。2026 年当前代码中还存在独立的 **Operator heuristic/ranking layer**：它能对已经合法的 Operators 用 `nvidia-matmul-heuristics` 做解析性能排名和裁剪，再只构建/实测其中一部分。这直接改变了我们之前“CUTLASS主要是 expert/static rules”的过粗结论。fileciteturn556file0L1-L7 fileciteturn557file0L1-L7

第二，Triton 再次发现若干此前没有进入总表的显式 heuristic，包括 `LinearLayout` 构造 heuristic、packed-K matmul heuristic、pipeliner 的“dirty heuristic”、fused-attention 特殊 layout-conversion heuristic，以及 distributed-layout contiguity/order heuristic。fileciteturn536file2L30-L40 fileciteturn536file10L159-L168 fileciteturn536file13L208-L218 fileciteturn536file14L224-L235

所以先补表。

## 本轮新增的遗漏规则

| Framework | 新 Rule ID | 源码决策点 | 具体规则 / 启发式 | 类型 | 实际用于选择什么 | 使用的信息 / proxy | 隐含假设 | 明确没有考虑 / 没有保证 | fallback / 后续机制 | 当前完整度 | 证据 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **CUTLASS** | **C-12** | Operator API heuristic layer | 合法 Operators 可以先由 heuristic 按 estimated performance 排序；返回结果可以只是原候选的子集 | P/H | 整个GEMM kernel/operator config，包括tile/schedule等已经实例化的组合 | problem \(M,N,K\)、dtype/layout/alignment、目标GPU | analytical ranking足够先裁掉大量候选 | 不保证exhaustive performance coverage | 保留top configs，再profile/autotune | **新增重要机制** | `Heuristic.rank()`明确允许reorder + prune，但不能invent candidate。fileciteturn556file0L1-L7 |
| CUTLASS | **C-13** | `NvMatmulHeuristics` | 使用 NVIDIA nvMatmulHeuristics 对 Operators 排序，并删除不匹配recommended configs的Operators | P/H | dense GEMM kernel family | M/N/K、batch、precision、layouts、A/B alignment、GPU SKU | 外部解析模型能预测好kernel family | 当前只支持有限GPU/problem classes；非oracle | over-fetch recommended configs，再match现有Operators | B | 当前实现只支持SM100 dense non-blockscaled GEMM，且明确prune不匹配candidate。fileciteturn557file0L1-L7 |
| CUTLASS | **C-14** | heuristic→profile pipeline | GEMM heuristic用于缩小runtime autotuning空间，只构建/profile推荐subset | H→M | kernel candidate set | nvMMH estimate + real profiler runtime | 先用analytical model裁剪，再measurement比全量measurement更划算 | 被heuristic裁掉的candidate不会进入最终measurement | `cutlass_profiler`测试生成的subset | B | 官方当前文档明确说目标是“reduce the search space for runtime autotuning”，且exhaustive coverage不保证。fileciteturn555file0L1-L7 |
| CUTLASS | **C-15** | persistent/static tile scheduler | `RasterOrderOptions::Heuristic` 根据M/N tile-grid aspect ratio选AlongM/AlongN | H | CTA/grid traversal layout | `tiles_m`, `tiles_n` | 沿较合适方向raster有更好locality/scheduling | 不测L2/cache/runtime效果 | 用户可强制AlongM/AlongN | A | 当 `tiles_n > tiles_m` 时选AlongM，否则另一方向。fileciteturn554file0L1-L15 |
| CUTLASS | **C-16** | tile scheduler swizzle | 根据M/N tile count与`max_swizzle_size`求`log_swizzle_size`，round grid到cluster/swizzle几何 | R/H | CTA grid swizzle / scheduler mapping | problem blocks、cluster shape、max swizzle | 规则grid swizzle改善调度/locality | 不直接以runtime比较不同swizzle | 用户可指定max swizzle/raster | A | scheduler保存 `log_swizzle_size_` 和 raster order。fileciteturn552file0L1-L6 |
| CUTLASS/CuTe | **C-17** | cooperative copy | CuTe存在“determine a good permutation”启发式，为后续vectorization与thread assignment排列tensor | H | copy operand permutation/thread assignment前的layout | source/destination tensor layout | 某些permutation更利于vectorization/thread assignment | 没有通用runtime model | 后续copy/thread assignment继续处理 | B | 源码注释直接称其为“heuristic to determine a good permutation”。fileciteturn550file12L196-L206 |
| CUTLASS | **C-18** | builder-family coverage | 当前SM100 builder不是一个family，而有dense、sparse、blockscaled、blockwise、mixed-input、TMA+cp.async、complex等大量独立builder | C0/R | candidate family本身 | dtype、quantization、sparsity、copy path | 各problem class可由独立specialization覆盖 | 不同builder叶规则尚未逐个抄完 | 不适用family不会实例化 | **B：仍未逐叶完成** | 当前builder目录至少包含这些独立production families。fileciteturn542file0L1-L2 |
| **Triton** | **T-18** | `LinearLayout` algebra | 求layout间映射时，在非唯一解中使用显式heuristic，而不是只用minimum-norm solution | H | layout→layout coordinate/warp mapping | 两个LinearLayout的维度对应关系 | 保持相同维度对应关系等结构偏好是好的canonical mapping | 不根据consumer/runtime判断 | 后续layout conversion lowering | **新遗漏** | 源码明确写“minimum norm solution”后采用“heuristic here is as follows”。fileciteturn549file2L41-L64 |
| Triton | **T-19** | Pipeliner `AssignLatencies` | 某些sync-dot loop不pipeline MMA；源码称其为“dirty heuristic for performance drops” | H | 是否pipeline MMA / latency assignment | sync dot、loop structure | 已知结构pattern可预测pipeline regression | 不是真实candidate benchmark；不全局探索peeling方案 | skip pipelining / 使用其他pipeline结构 | **新遗漏，RQ1证据强** | fileciteturn536file2L30-L40 |
| Triton | **T-20** | `AccelerateMatmul` packed-K | 针对packed-K join/bitwidth形态有专门heuristic，决定是否采用特定tensor-core transformation | H/C0 | MMA-compatible layout/transformation | joined K shape、element bitwidth | 某种packed-K结构代表目标量化/packed operand模式 | 不是通用cost comparison | 不match则走普通matmul path | **新遗漏** | 源码直接称“This heuristic is intended for packed-K values...”。fileciteturn536file10L159-L168 |
| Triton | **T-21** | conversion semantics / fused attention | 某类向 `DotOperandEncodingAttr` 的conversion特意不按普通路径处理，以兼容fused attention | H/R | layout conversion legality/lowering | target encoding + attention pattern | fused-attention需要特殊conversion策略 | 不一定泛化到其他consumer graph | 特化conversion behavior | **新遗漏；现代Attention直接相关** | 源码注释“heuristic to accommodate fused attention”。fileciteturn536file13L208-L218 |
| Triton | **T-22** | distributed encoding / thread order | 如果element contiguity与thread order不对齐，使用显式heuristic处理layout/order | H | distributed element/thread mapping | contiguity、thread order | thread order应适应元素连续性 | 不直接看whole-kernel consumer | encoding construction | **新遗漏** | Dialect源码存在显式“Heuristic: If the element contiguity does not align with the thread order...”。fileciteturn536file14L224-L235 |
| Triton | **T-23** | GenericSwizzling | 仍确认：minimize read+write bank conflicts，tie-break data-move rounds | P/H | shared swizzle | bank conflict count、move rounds | 这些指标足够排序SMEM layouts | 不直接考虑occupancy/pipeline/cache | 取proxy最好swizzle | A | fileciteturn536file0L1-L11 |
| Triton | **T-24** | Warp partition graph | partition merge由一串heuristics作用在跨partition edges直到fixed point | H | producer/consumer partitioning | op kind、edges、communication等 | 局部merge规则足以构造好warp specialization | 非全局partition optimizer | merge直到fixed point | A | 当前源码明确说“heuristics ... applied to every edge ... until fixed point”。fileciteturn536file11L174-L184 |
| **Gluon** | **G-10** | AutoLayout conflict | 每次fuzzy inference增加distance；冲突时distance小者胜 | H/C | automatic distributed encoding propagation | seed distance | 离显式seed更近的layout更可信 | 不看performance | fixed-point propagation | **A，现已精确封口** | 完整当前源码。fileciteturn540file0L1-L7 |
| Gluon | **G-11** | equal-distance tie | 相同非零distance用stable hash/order确定winner；distance=0不同seed则error | H/C | encoding + distance | 同等可信自动推导无需performance比较 | tie选择与runtime无关 | explicit conflict报错 | **A** | 同上。fileciteturn540file0L1-L7 |
| Gluon | **G-12** | function boundary | AutoEncoding不允许跨function boundary；函数需fully inline | C0 | function signatures | auto-layout inference保持函数内闭包 | 跨函数global layout optimization不做 | compile error | **A** | 同上。fileciteturn540file0L1-L7 |
| Gluon | **G-13** | CoalescedEncoding seed | load/store上利用AxisInfo + numWarps + threadsPerWarp + shapePerCTA产生coalesced seed | H | memory access analysis | memory-local seed可沿graph传播 | 不考虑非memory consumer性能 | 再交AutoLayout传播 | **A** | fileciteturn541file0L1-L7 |
| Gluon | **G-14** | CGA auto restriction | automatic coalesced inference当前要求`numCTAs == 1` | C0 | CTA count | multi-CTA自动coalescing暂不纳入空间 | candidate coverage缩小 | multi-CTA需显式layout | **A** | `getDefaultCGALayout`直接assert。fileciteturn541file0L1-L7 |
| **TIRx** | **X-10** | `copy/fallback` | priority 0，只要general copy合法即可兜底 | F/C0 | copy legality | 通用慢路径保证coverage | 性能最低优先级 | 高priority都拒绝后采用 | A | fileciteturn525file3L40-L50 |
| TIRx | **X-11** | `copy/vec_auto` | priority 10，predicate通过才构造自动vector mapping | H/C0 | scope/layout/shape | 合法自动vector path优于fallback | 不benchmark与其它同priority variant | first-success | A | fileciteturn525file4L53-L63 |
| TIRx | **X-12** | forced vector copy | forced vector family处于更高priority族 | H/E | explicit vector variant | explicit specialized vectorization优先 | 用户要求可能不是最快 | predicate→implementation | B | fileciteturn523file3L60-L80 |
| TIRx | **X-13** | `copy_async/ldgsts` | priority 20的cp.async/ldgsts路径 | H/C0 | global→shared、alignment、scope | 符合ldgsts时应优先于generic async copy | 无runtime ranking | predicate失败继续priority 10族 | A | fileciteturn523file0L1-L20 |
| TIRx | **X-14** | warp XOR permute | `permute_layout/warp_xor_swizzle` priority 20 | H | source/dest layout pattern | warp-XOR是首选特化layout permutation | 不和其它合法permute benchmark | reject后其它implementation | A | fileciteturn523file1L22-L42 |
| TIRx | **X-15** | SM100 packed reduction | 特化packed reduction priority 20，高于general local/shared priority 10 | H/C0 | SM100 + layout pattern | specialized reduction应该优于general | 不测速度 | predicate不match→local/shared path | A | fileciteturn523file2L43-L59 |
| TIRx | **X-16** | local/shared reduction | local/shared各priority 10，根据storage scope和validity dispatch | H/C0 | storage scope、layout validity | storage scope足够选实现族 | 同priority性能不比较 | first-success | A | fileciteturn526file5L61-L71 fileciteturn526file6L72-L82 |
| TIRx | **X-17** | GEMM | priority 10，要求full active lanes、no replica等layout条件 | H/C0 | active lanes、replica axes、operand layouts | 满足pattern时该MMA mapping优选 | 其它合法MMA性能不测 | rejection/fallback | B | fileciteturn526file3L39-L49 |
| TIRx | **X-18** | async GEMM | gemm_async特化priority 10，并按exec-scope/layout predicates dispatch | H/C0 | warp/thread scope、target/layout | tcgen05/WGMMA local priority能表示preferred implementation | 不benchmark | reject→other variant | B | fileciteturn526file7L83-L93 |
| TIRx | **X-19** | DSMEM / TMEM async copies | DSMEM、SMEM→TMEM、TMEM↔local均为独立priority-10variants | H/C0 | storage scope、execution scope、layout contract | specialized memory-space instruction优先 | 没有joint ranking | first-success | B | fileciteturn526file8L94-L104 fileciteturn526file9L105-L115 fileciteturn526file10L116-L126 |
| **vLLM** | **VL-14** | backend API layout contract | base backend声明“supported layouts, most preferred first”；None表示任意且无偏好 | H/C0 | kernel capability | backend自己的ordered preference有意义 | preference不是系统级runtime cost | whole-model resolver汇总 | A | fileciteturn516file1L14-L24 |
| vLLM | **VL-15** | CPU persistent layout | CPU只接受LBHNC，因为kernel读取head-major block interiors | C0 | CPU addressing | kernel物理寻址决定layout | 无alternative | global intersection | A | fileciteturn516file3L36-L46 |
| vLLM | **VL-16** | HPC layout | HPC只接受LBNHC | C0 | backend contract | 单一kernel physical representation | — | resolver | A | fileciteturn516file4L47-L57 |
| vLLM | **VL-17** | B12X layout | `(LBHNC, BLHNC)` 有序支持 | H/C0 | B12X kernels | first entry是backend preference | whole-model cost不比较 | resolver voting | A | fileciteturn516file2L25-L35 |
| vLLM | **VL-18** | QSA | QSA pages要求BLNHC/BLHNC，因为QSA pages与主KV block并置 | C0/H | page packing | cache packing决定stride family | — | intersection | A | fileciteturn516file7L85-L95 |
| vLLM | **VL-19** | FlexAttention | 强制LBNHC，使B,N可zero-copy flatten | C0 | desired zero-copy view | 避免representation conversion优于支持其它strides | 不比较转换版Flex | backend约束 | A | fileciteturn516file12L140-L150 |
| vLLM | **VL-20** | ROCm native content | `customize_spec()` 把K/V设为两个head groups以匹配native HIP contiguous addressing | C0/R | native HIP kernel | consumer-native packing最重要 | global uniform packing被放弃 | backend spec rewrite | A | fileciteturn517file0L1-L12 |
| vLLM | **VL-21** | AITER unified content | 与native ROCm相反，保留K/V在content dim中packed | C0 | AITER kernel | 不同consumer确实要求不同packing | — | spec rewrite | **A；RQ4非常强证据** | fileciteturn517file1L14-L24 |
| vLLM | **VL-22** | Triton quant content | per-token/head量化在每个head数据后inline FP32 scale | C0 | quant mode | data+metadata co-layout由consumer kernel定义 | 不独立优化scale placement | customize_spec | A | fileciteturn517file3L40-L50 |
| vLLM | **VL-23** | TurboQuant | K+V合并成每head一个slot | C0 | TurboQuant | combined slot更符合quant kernel | 与通用KV representation不一致 | customize_spec | A | fileciteturn517file5L66-L77 |
| vLLM | **VL-24** | FlashInfer NVFP4 | K、V为独立head slots，packed FP4 + FP8 block scales | C0 | NVFP4 consumer | metadata layout是persistent representation的一部分 | 不能只用stride-order描述 | customize_spec | A | fileciteturn517file7L92-L103 |
| vLLM | **VL-25** | composite backend | composed backends的`customize_spec()`结果必须一致，否则报错 | C0 | two consumers | 一个shared persistent spec必须满足所有consumer | 不会自动创建两份spec | error | **RQ4直接证据** | A | fileciteturn517file8L105-L115 |
| vLLM | **VL-26** | block-size resolution | 默认block size由backend preferred size决定，除非用户显式设置 | H/E | backend kernel sizes | backend静态preferred page geometry可泛化 | runtime workload不进入 | user override | A | fileciteturn518file0L1-L11 |
| vLLM | **VL-27** | FA block-size policy | FlashAttention存在特殊FA4/head-dim相关规则，否则至少max(default,64)或base policy | H | kernel/head dimension | 特定kernel需要更大page | 不动态benchmark page size | preferred block | B | fileciteturn518file3L34-L49 |
| vLLM | **VL-28** | Sparse MLA/SWA block | fixed preferred block=256 | H/C0 | sparse backend | 256适合/要求该backend | no dynamic choice | user override可能先占优 | A | fileciteturn518file4L50-L60 |
| vLLM | **VL-29** | ROCm AITER block | fixed preferred block=64 | H/C0 | AITER unified | 64作为backend native geometry | — | user override/capability checks | A | fileciteturn518file5L61-L72 |
| **SGLang** | **SG-17** | model override ownership | 一个architecture可以被多个model override module处理，但同一个field不能被两个module声明 | C0/design rule | field ownership | single-writer让resolution确定性 | 不能自动merge两个对同字段有不同策略的provider | test/registration discipline | **新遗漏** | package源码明确说明否则结果会依赖import order。fileciteturn521file0L1-L7 |
| SGLang | **SG-18** | declarative resolution | `declare_resolution`只记录决策；dependent resolver必须通过`resolving_view/resolved_view`读前面已解析结果 | C0/R | dependency between defaults | layout/backend/page等配置decision存在显式依赖链 | raw-field读取会得到用户原始值而不是上游decision | tests禁止raw reads | **新遗漏，RQ3重要** | 测试说明整个resolution pipeline必须读取resolved view。fileciteturn546file0L1-L7 |
| SGLang | **SG-19** | override declaration surfaces | 决策既可能来自`declare_resolution` keyword，也可能来自`MODEL_OVERRIDES` map、provider返回dict、post-process map | R/Architecture | source/provider | 多种policy source最终归并到统一resolution stash | 需要扫描全部sources才能完整审计 | AST tests自动覆盖新增字段 | **新遗漏** | `_declared_fields()`明确扫描这些来源。fileciteturn546file0L1-L7 |
| SGLang | **SG-20** | model override family inventory | 当前production registry至少28个model-family modules | H/R | exact architecture/model config | generic policy不足，需大量family-local policy | 尚未逐文件列出所有字段分支 | modules import后注册 | **B：最大剩余缺口之一** | 当前registry逐项列出所有family模块。fileciteturn521file0L1-L7 |
| SGLang | **SG-21** | resolved-field surface | 当前resolution系统可决定的不只是backend/page，还包括MoE、parallel、cache等多个配置字段 | H/R | model/platform/runtime | performance policy跨多个配置维度 | layout相关field与非layout field可能耦合 | declarative resolver | B | tests枚举了`attention_backend`, `page_size`, `moe_dp_size`等resolved fields。fileciteturn519file5L89-L104 |
| **FlashInfer** | **FI-13** | avg-Q representative statistic | FA2 scheduler对batch求 `avg_packed_qo_len`，用这个均值决定CTA-Q | **H/P** | batch mean query length | batch平均值足以代表不均匀request distribution | variance/tail信息丢失 | scheduler根据均值选tile | **新增应单列的假设** | fileciteturn528file0L1-L16 fileciteturn528file2L33-L45 |
| FlashInfer | **FI-14** | short-Q CTA rule | `avg_packed_qo_len <= 32`时偏CTA16等short/decode rule | H | mean Q len | short-Q regime可由单阈值划分 | batch heterogeneity被压缩 | longer-Q走其它tile | A | 当前函数注释明确“decode / short-q”。fileciteturn528file0L1-L16 |
| FlashInfer | **FI-15** | occupancy-query scheduling | decode/prefill/Blackwell等生产路径实际调用CUDA occupancy API | P/runtime | kernel resources、SM count | occupancy是work partition/split的重要代理 | 不等于真实runtime | split/no-split、grid/chunk planning | A | decode/prefill/scheduler/Blackwell均有occupancy call。fileciteturn529file9L120-L130 fileciteturn529file10L131-L142 fileciteturn529file11L147-L157 |
| FlashInfer | **FI-16** | Blackwell bucket load balance | 对 `(batch,head,Q-tile)` 工作项，把causal-adjusted KV length作为cost，贪心分到当前最小cost bucket | **P/H** | KV length、Q tile、causality | 工作量近似与KV token长度成比例 | 不含不同tile实际latency/cache/stall差异 | greedy min-cost assignment | A | planner直接把KV length累加到bucket cost。fileciteturn530file0L1-L27 |

这个补充表之后，完整性状态应该重新降级成下面这样，而不是继续沿用上一轮的“基本封口”。

| Framework | 机制族是否完整 | 自动/default production规则是否逐叶枚举 | 当前状态 | 仍需继续审查的具体部分 |
|---|---:|---:|---|---|
| **CUTLASS/CuTe** | 基本完整，但本轮新增operator-level heuristic与tile scheduler | **否** | **B** | SM90/SM100/SM103各builder leaf；dense/sparse/blockscaled/blockwise/mixed-input/complex；epilogue leaves；operator nvMMH与传统Builder之间完整连接；grid scheduler各模式 |
| **Triton classic** | 仍在发现新heuristic | **否** | **B-/C+** | `LinearLayout.cpp`所有heuristic；`AccelerateMatmul`所有packed/quant heuristics；pipeline/AssignLatencies全部特殊分支；Dialect encoding heuristics；fused-attention特例；NVIDIA TMEM/descriptor路径再次关键词扫尾 |
| **Gluon** | **基本完整** | 大部分完成 | **A-** | Blackwell-specific explicit/default layout helper叶规则再扫一遍；自动机制本身已经封口 |
| **TVM MetaSchedule** | 基本完整 | 尚未逐一枚举全部CUDA ScheduleRule/Postproc叶分支 | **A-/B+** | `AutoInline/AddRFactor/PVUnroll/RewriteCooperativeFetch/...`中所有会改变thread/cache/layout的叶规则 |
| **TIRx** | family完整 | **否** | **B** | TMA copy variants、完整gemm/gemm_async predicates、reduction全部variants、每个priority相同variant的注册顺序 |
| **Hexcute** | **基本完整** | 仍可展开instruction leaves | **A-** | 每个CopyInstruction/MmaInstruction exact predicate、remaining-stride heuristic具体算法；artifact fork版本边界继续保持 |
| **TileLang** | **基本完整** | 大部分完成 | **A-** | Strict/Common/Free全部candidate-root生成branch、TMA promotion predicate、reducer全部fallback；目前没发现新机制族 |
| **vLLM** | 机制完整 | **接近但未完全** | **B+** | 将全部backend做成最终 `layout × customize_spec × block-size × capability`矩阵；当前已覆盖主要特殊backend |
| **SGLang** | 机制现在更完整，但override surface非常大 | **否** | **B** | 28个model-family模块逐文件抽取layout/backend/page/KV/MoE/state字段；各hook中的dependent resolution；backend-internal policy |
| **FlashInfer** | scheduler机制族大体完整 | **否** | **B** | FA2 prefill、decode、POD、MLA、Blackwell FMHA、SM100 MLA、TRTLLM-derived paths分别穷举kernel-family leaves |

所以，这一步现在的结论必须是：

\[
\boxed{\textbf{还没有完成。}}
\]

而且最新反向审计证明，“机制族完整”与“每一个启发式规则完整”之间还有明显距离。最典型的是 CUTLASS：我们之前已经理解 `CollectiveBuilder`，但随后又发现完全独立的 Operator heuristic层。`Heuristic.rank()` 当前甚至明确允许**排序并裁剪已经合法的operators**，而 `NvMatmulHeuristics` 会只留下与解析模型recommended configs匹配的候选。fileciteturn556file0L1-L7 fileciteturn557file0L1-L7

这意味着 CUTLASS 当前实际上至少存在三个selection层，而不是一个：

\[
\boxed{
\text{Builder structural rules}
\rightarrow
\text{Operator analytical heuristic}
\rightarrow
\text{Profiler measurement}
}
\]

官方当前heuristics文档也明确说，这一层的目的就是减少runtime autotuning搜索空间，并且“exhaustive functional or performance coverage is not guaranteed”。fileciteturn555file0L1-L7

这会直接修改我们后面对 CUTLASS 在 RQ1/RQ2 中的位置：它不是单纯“expert/rule endpoint”，而已经包含“analytic prune → measurement”的混合路线。

Triton也是同样的问题。前面已经列出了核心passes，但反向关键词审计继续找到生产源码自己称为 `heuristic`、`dirty heuristic`、`heuristic to accommodate fused attention` 的额外分支。fileciteturn536file2L30-L40 fileciteturn536file13L208-L218 因而当前不能说“每一种Triton heuristic已经逐项登记完”。

SGLang的情况更明显：当前 `model_overrides/__init__.py` 已经列出约28个model family module，而且源码明确允许同一个architecture由多个module分别拥有不同字段，但禁止两个module同时拥有同一个field，否则结果会依赖import order。fileciteturn521file0L1-L7 它还有一套专门的resolved-view依赖机制：一个后续resolver必须读取前面已经解析出的decision，而不是用户最初输入值。fileciteturn546file0L1-L7 所以 SGLang 需要进一步审计的不是“再找几个模型例子”，而是要构造真正的：

\[
\boxed{
ModelFamily
\times
DecisionField
\times
Predicate
\times
Value
\times
UpstreamDependencies
}
\]

矩阵。

vLLM则已经更接近完成，因为当前三大persistent decision接口现在比较明确：

\[
\boxed{
supported\_kv\_cache\_layouts()
}
\]

决定stride-order候选；

\[
\boxed{
customize\_spec()
}
\]

决定page内content/metadata packing；

\[
\boxed{
get\_preferred\_block\_size()
}
\]

决定page geometry。基础API和多个backend当前实现都已经明确。fileciteturn515file0L1-L16 Flex、CPU、HPC、B12X、QSA等layout要求也有明确consumer理由。fileciteturn516file2L25-L35 fileciteturn516file3L36-L46 fileciteturn516file4L47-L57 fileciteturn516file7L85-L95 fileciteturn516file12L140-L150

因此我现在不会再说“再查一轮就一定完成”，而会采用一个更机械的封口条件：**对每个仓库完成一次production路径的反向关键词审计，并要求连续一轮搜索不再发现新的decision family；剩余结果只能落入已有Rule ID的architecture/dtype/model leaf。** 目前 Gluon、Hexcute、TileLang比较接近这个状态；CUTLASS、Triton、TIRx、SGLang、FlashInfer还没有，vLLM和TVM介于中间。

所以接下来的工作仍然应该是继续补这张规则表，而不是继续科研问题。等所有十个系统都只剩“已有规则族的重复叶实例”，再把此前主表和本轮新增行合并成唯一的 canonical evidence table，然后才算你要求的“所有框架证据链整理完成”。

## 用户

证据链不完整请继续补充。然后把证据链也凝炼为问题。然后从RQ1开始做一条完整的“问题审判”：所有相关框架源码规则 → 每个规则使用的proxy → 它忽略了哪些变量 → 框架自己的fallback/repair → 现代Attention/GQA/MLA/MoE触发条件 → 可证伪的科研假设。 RQ1彻底完成之后再审判RQ2，然后后续RQ。这样可以防止又回到前几轮“问题列了很多，但证据深度不够”的状态 。当然如果新的证据链形成新的RQ 也请补充。

## 模型回复

我会按“证据先封口、问题后凝练”的顺序继续。这一轮先补目前仍明显不完整的几块，并把每条规则统一压到同一证据格式：**源码规则 → 决策对象 → proxy/输入 → 隐含假设 → 被忽略变量 → fallback/repair → 适用 workload**。之后只对证据最强的 RQ1 做完整“问题审判”，不提前展开 RQ2。

## 模型工具调用参数

{"paths":["GitHub"],"query":"search"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 CUTLASS 自动调度与内存布局实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 NVIDIA CUTLASS 中的 StageCountAuto 定义

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索更多相关代码与引用

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 TVM MetaSchedule 多级分块规则

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 TVM CUDA 默认调度规则

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 schedule_rule.cc 代码片段

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 SGLang 模型后端覆盖分支

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 DeepSeek、Qwen、Nemotron 模型文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 Qwen3-VL 与 Nemotron-H 注意力实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 Nemotron 模型配置代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

深入探究 FlashInfer 解码调度策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 FlashInfer 中的 split_kv 相关代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检查 split_kv_mode_policy 实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理注意力后端选择与现代MLA回退机制

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 Triton 布局启发式及代理不足

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 TritonGPU 优化与内存布局相关文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Hexcute 当前状态及仓库证据

## 模型思考摘要

搜索并梳理 CUTLASS、TVM、SGLang、FlashInfer 与 Triton 调度实现

## 模型思考：工具活动

正在搜索 Hexcute CGO 2026 artifact GitHub hidet instantiate_auto_annotation cost_model bank conflict

## 模型思考：工具活动

正在搜索 7 个网站

## 模型思考摘要

已搜索 7 个网站

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取仓库最新拉取请求

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取仓库中用户最近的拉取请求

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索仓库中的 Hidet 相关代码

## 模型思考摘要

检查子模块、拉取请求及 Hidet 相关代码

## 模型思考：工具活动

正在搜索 github.com

## 模型思考摘要

已搜索 1 个网站

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 GitHub 子模块配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Hidet 自动注解与代价模型实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取并分析自动注解实例化代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查看 CUDA 成本模型文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析 TileLang 布局推断与成本模型

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析布局推断与成本选择逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 TIRx 调度注册优先级实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 TVM 调度注册与优先级逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查找 TVM 中 copy 调度注册路径

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

全量审计 TVM 调度实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 Tile Primitive 调度器代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

审查SGLang文档与代码实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检视 vLLM 注意力后端选择器

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析 CUTLASS SM100 阶段与调度策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析 CUTLASS 阶段计数实现公式

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

计算 SM90 阶段数量公式

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

总结 Gluon 布局推导与编码机制

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析 Hexcute 指令选择与 Bank 冲突

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 CUDA Bank Conflict 解析代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 MetaSchedule 硬件测量与候选选择流程

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 Meta Schedule 教程示例

## 模型思考摘要

梳理研究问题
审查 RQ1 证据
扩展 RQ1 证据
深化代理审判
汇总决策边界
补全现代触发条件
凝练研究问题
梳理并深化研究问题及相关证据

## 模型思考摘要

审计 GPU 调度、布局推断与成本模型实现

## 模型回复

目前证据已经出现一个重要变化：**新增证据没有再产生新的“决策机制族”，而是在已有机制族下不断增加 architecture/model-specific leaves**。例如 CUTLASS 新发现的 SM100 `SW128→SW64→SW32→INTER` 是“静态结构 proxy 选 layout”的新叶；SGLang DeepSeek/Nemotron 是“model/platform feature tree”的新叶；它们没有再引入一种全新的选择机制。  
因此我会把“完成”分成两层：**研究证据链可以封口**（足以支撑/反驳 RQ），但“把十个仓库所有 architecture-specialization 每一行都抄进附录”仍不是穷尽状态。这个区分可以避免无限审计阻止科研问题本身。

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

补强证据并完成 RQ1 全量试验

## 模型思考摘要

完成证据补强并开展RQ1全量试验

## 模型思考用时

思考了 7m 22s

## 模型回复

可以。经过这一轮继续反查，我认为现在可以把“证据链是否完成”分成两个标准来看：

- 如果“完成”要求把十个仓库所有 architecture / dtype / model-specific specialization 的每个 `if constexpr`、每个 override 都逐行抄出来，那么还可以无限继续，尤其 CUTLASS、SGLang、FlashInfer 的 specialization 数量很大。
- 如果“完成”要求的是：**所有不同的 layout 决策机制族都已经找到；每种机制都有源码规则、proxy、隐含假设、未观测变量、fallback/repair；剩下未枚举的只是已经确认机制的重复 specialization leaf，而不再是新的 selection mechanism**，那么现在已经达到可以凝练科研问题的标准。

本轮继续检查没有再发现一种新的 mechanism family，发现的是已有机制的新叶子。例如 CUTLASS SM100 shared-memory selector继续证实“按 tile divisibility 静态选最大 swizzle layout”；Triton继续出现更多局部 heuristic；SGLang继续出现更多 model-specific repair；FlashInfer继续出现结构化 split-KV policy。fileciteturn599file0L1-L11

所以，从这里开始我认为可以正式进入科研问题，但我会严格按你说的：**先只彻底审判 RQ1，不提前替 RQ2 下结论。**

---

# 一、证据链最后一轮补充：此前最关键的空洞已经补上

我先把这轮真正改变证据强度的部分补进去。这里不是重复之前大表，而是补此前没有精确到“决策函数实际看什么”的地方。

| 框架 | 精确决策点 | 实际规则 | 选择对象 | 实际 proxy | 由此形成的假设 | fallback / repair |
|---|---|---|---|---|---|---|
| CUTLASS | `StageCountAutoCarveout` | \((SMEM-capacity-carveout)/stage\_bytes\)，取能容纳的最大 stage 数 | pipeline stages | tile A/B字节数、pipeline metadata、SMEM capacity | 在给定tile下，“尽量多stage”是好默认值 | 可显式指定 `StageCount<N>` | 
| CUTLASS | `sm100_smem_selector()` | SW128 → SW64 → SW32 → INTER，选择第一个满足tile divisibility的layout | SMEM layout/swizzle | `BLK_MN0` / `BLK_K0`、major、dtype | 最大可适配swizzle通常最好 | 小一级swizzle；最终INTER；不合法则编译失败 |
| CUTLASS | `KernelScheduleAuto` | 根据 MMA tile、dtype、layout 等选instruction family和1SM/2SM schedule | MMA/schedule | static structural properties | structural compatibility足以决定好schedule family | 用户显式schedule；之后还可heuristic/profile |
| CUTLASS | Raster heuristic | `tiles_n > tiles_m → AlongM`，否则AlongN | CTA traversal | tiled M/N aspect ratio | grid aspect ratio是cache/scheduling locality的充分统计量 | 用户强制AlongM/AlongN |
| CUTLASS | nvMatmulHeuristics | analytical model先排序/裁剪valid operators | 整个kernel config | M/N/K、batch、dtype、layout、alignment、GPU | analytic model可安全缩小measurement space | CUTLASS profiler实测幸存者 |
| Triton | Coalesce | memory op改为cache-friendly/coalesced encoding，并在边界插conversion | distributed layout | AxisInfo/contiguity/memory access structure | memory-locality收益足以值得conversion | 后续RemoveLayoutConversions/rematerialization |
| Triton | RemoveLayoutConversions | expensive load/store偏Blocked，tensor op偏MMA encoding | encoding propagation | op类别、conversion graph、memory-vs-MMA角色 | 局部producer/consumer preference可以逐步调和 | rematerialization / conversion |
| Triton | GenericSwizzling | 最小化read+write bank conflicts；tie看搬运rounds | SMEM swizzle | bank conflict计数、data-movement rounds | 这两个量可代表swizzle性能 | 选择另一legal swizzle |
| Triton | Pipeliner | sync-dot情况下有明确“dirty heuristic”避免MMA pipeline | pipeline arrangement | IR pattern | 此pattern足以预测performance regression | 不pipeline该MMA |
| Triton | TMEM utility | 不使用超过约一半register，避免spill | register/TMEM mapping | register count fraction | register footprint阈值足以预测spill风险 | 较保守的mapping |
| Triton | gather lowering | warp shuffle与SMEM roundtrip用近似相对成本选择path | gather layout/lowering | approximate shuffle-vs-SMEM cost | 固定局部cost ratio足以做选择 | shared-memory path等 |
| Gluon | AutoLayout | fuzzy propagation每跨可变layout op distance+1，冲突取distance小者 | encoding | graph distance to explicit seed | “离seed近”≈更可信的layout | explicit seed conflict报错 |
| Gluon | Coalesced layout | AxisInfo + warps + threads/warp + CTA shape生成seed | memory layout | coalescing信息 | memory seed可以向producer/consumer传播成为好layout | 显式layout |
| TVM | DefaultCUDA ScheduleRules | 固定 `SSSRRSRS`、固定tile binds、vector lens、reuse level、thread extents等生成搜索空间 | tile/thread/cache layouts | hand-designed candidate grammar | 高性能区域大部分位于此grammar内 | XGB + EvolutionarySearch + hardware Runner |
| TIRx | dispatcher | priority降序、predicate全部通过、first-success | implementation + derived mapping | legality predicates + expert priority | priority足以代表legal implementations的性能偏好 | lower-priority variant |
| Hexcute | constraint inference | layout/instruction/task mapping先满足copy/MMA约束 | legal mapping space | algebraic compatibility、scope、alignment | 高性能布局必须首先满足结构约束，能大量压缩空间 | backtracking / alternate instruction |
| Hexcute | cost model | `#inst × latency`，copy/MMA按full overlap模型 | candidate ranking | instruction count/latency | local instruction latency足以近似ranking | DFS候选中选其他config |
| Hexcute | bank-conflict resolver | 对一个SMEM tensor所有copy consumer求bank-conflict总和最小的swizzle | shared layout | summed conflict ways | consumer间冲突可以由bank-conflict总和组合 | 枚举其它swizzle |
| TileLang | register-count | 默认free-layout候选按register proxy排序 | fragment layout | register count | register footprint是好默认ranking proxy | optional `io-aware` |
| TileLang | io-aware | \(time(S)\approx\max(bw,issue)\)，对global-memory statements求和 | fragment/thread layout | vector width、coalescing segments、issue depth | memory traffic模型足够比较layouts | 不可建模layout→worst-case charge |
| TileLang | reducer repair | conflict可从narrow单调widen到wide；必要时replicated fallback | reducer destination layout | compatibility | 更宽layout牺牲specialization换取全consumer可读 | universally-readable replicated layout |
| vLLM | `get_supported_kv_cache_layouts` | backend supported-set求交；first-choice投票；tie按enum顺序 | persistent KV layout | backend compatibility + ordinal preference | backend preference票可以代理model-wide性能 | incompatible connector→candidate[0]；空交集报错 |
| vLLM | mixed HNC rule | HNC shape不同则只保留block-compact layouts | persistent representation | shape compatibility | aliasability是必要结构约束 | 没有合法layout则报错 |
| vLLM | `customize_spec` | backend改变K/V/content/scale packing | cache content layout | kernel-native representation | consumer-native packing优于统一抽象packing | composite backend要求spec完全一致 |
| vLLM | preferred block size | backend给固定/规则化page size | page geometry | kernel backend/type | backend静态preferred size足够 | 用户显式block-size优先 |
| SGLang | generic MHA backend selector | Hopper→FA3；SM100→TRTLLM/FA4；HIP→AITER；其余FI→Triton等 | backend | architecture、spec mode、asymmetric KV、sink等 | coarse model/hardware feature tree能够代表最佳kernel family | FlashInfer→Triton等 |
| SGLang | MLA selector | Hopper FA3；SM100 FlashInfer；HIP AITER仅KV head=16/128，否则Triton | MLA backend | architecture、KV-head count | few discrete features足够划backend region | Triton |
| SGLang | DeepSeek DSA | ROCm preshuffle可用→page64；不可用→page1；XPU→128 | KV page layout | backend capability/platform | capability class决定page geometry | model-specific override |
| SGLang | Qwen3-VL | 大显存Hopper上使用“profiled serving defaults”，decode→FlashInfer等 | serving/backend policy | model type、GPU family、VRAM threshold | 一个profiled model/platform region可以静态复用 | generic selector |
| SGLang | Nemotron-H | quant/MoE/spec mode改变MoE runner、attention backend/page size | MoE/attention policies | quantization、SM generation、spec top-k、A2A | discrete workload labels代表最佳kernel family | Marlin/FI-CUTLASS/TRTLLM/Triton等 |
| FlashInfer | FA2 CTA-Q | batch求 `avg_packed_qo_len`; <=32倾向CTA16 | Q-tile | **query-length mean** | mean足以代表batch query distribution | 另一CTA tile |
| FlashInfer | split-KV mode | FMHA: cluster→separate-GMEM→inline-GMEM；MLA 1CTA优先cluster | reduction topology | family、topology、support、occupancy | structural support优先级足够，无需shape crossover table | support/occupancy reject后下一个mode |
| FlashInfer | Blackwell balancing | causal-adjusted KV length作为bucket work cost | CTA work allocation | KV token count | compute work约与KV tokens成比例 | greedy least-cost bucket |

CUTLASS 的 StageCountAuto 不是抽象描述，而是当前源码实际通过 per-stage SMEM footprint 和 carveout 计算最大可容纳stage。fileciteturn593file0L1-L2 当前SM100 selector也确实首先从tile shape和major选择共享内存layout，而不是运行一个runtime cost model。fileciteturn591file0L1-L2 `KernelScheduleAuto`则明确允许builder自动选schedule；Blackwell文档指出其根据MMA tile、datatype、layout选择instruction kind和1/2-SM tcgen05路径。fileciteturn560file1L12-L23 fileciteturn560file6L77-L88

TVM这边现在也已经精确到默认candidate grammar，而不是“TVM会搜索”这么粗。DefaultCUDA固定了 `SSSRRSRS`、block/vthread/thread bindings、vector widths、shared/local reuse位置、cross-thread reduction extents和AutoBind候选；TensorCore又是另一套静态intrinsic/tile grammar。fileciteturn563file0L1-L2 随后才由EvolutionarySearch、XGB cost model和真实Builder/Runner测量进行selection。fileciteturn598file0L1-L10

TIRx也已经封到真正的selector：不是泛称“predicate dispatch”，而是明确的

\[
\boxed{
sort(-priority,\ variant)
\rightarrow predicates
\rightarrow first\ successful\ PrimFunc
}
\]

失败会继续下一个variant，只有所有variant都失败才报错；用户还可显式`dispatch=`覆盖。fileciteturn588file0L1-L10

FlashInfer最近这条证据尤其关键：`select_split_kv_modes()`甚至接收 `tile_size_q`、`head_dim`、`head_dim_per_cta_v`，但当前mode ordering刻意不使用这些shape值。源码明确说明希望保持“structural”选择、避免shape-specific measured crossover tables；FMHA固定优先 cluster→separate→inline，MLA则1CTA优先cluster。fileciteturn570file0L1-L13

这不证明它一定选错，但它给出了一个非常干净的、可证伪的科研假设入口。

---

# 二、从证据重新凝练 RQ：排序发生了变化

此前把“boundary regret”放第一位，现在证据量改变以后，我认为不应该机械沿用原顺序。

目前跨框架最强、源码最直接的问题已经变成：

\[
\boxed{\textbf{RQ1 — Proxy Sufficiency / Decision-Sufficient Context}}
\]

中文可以更精确地写成：

> **现有 GPU layout / mapping / schedule selector 使用的低维代理变量，是否足以保持现代 LLM workload 下接近最优的配置选择？如果不充分，哪些被遗漏的状态变量系统性地改变最优决策？**

然后才是：

| 顺位 | 科研问题 | 证据来源 | 与RQ1关系 |
|---|---|---|---|
| **RQ1** | **低维proxy是否是layout决策的充分统计量？** | 几乎所有10个系统都有直接证据 | 本轮完整审判 |
| **RQ2** | 分阶段/跨边界决策是否近似可分；否则产生多少Boundary Regret？ | SGLang/vLLM/Triton/TVM/TIRx/Hexcute/TileLang等 | 下一轮再审判 |
| **RQ3** | Candidate generator的结构剪枝能否保留near-optimal region？ | CUTLASS/Triton/TVM/TIRx/Hexcute/TileLang + runtime backend catalogs | 后续 |
| **RQ4** | 同一persistent/intermediate representation面对多个consumer时，局部layout preference能否组合？ | vLLM/Triton/Hexcute/TileLang/TVM等 | 后续 |
| **RQ5** | workload空间是否能被少量稳定决策区域覆盖，还是需要动态policy？ | SGLang profiled overrides、FlashInfer thresholds、vLLM static model-wide layout等 | **本轮新增证据增强的问题** |

原来的“ranking sufficiency”不再单独放成一个大RQ，因为它实际上是RQ1的一个子问题：

\[
\text{analytic ranking proxy}
\subset
\text{decision proxy}
\]

Hexcute cost model、TileLang cost model、CUTLASS nvMMH属于ranking proxy；而vLLM投票、TIRx priority、SGLang decision tree、FlashInfer thresholds连显式cost model都没有，因此RQ1更一般。

下面只审判 RQ1。

---

# 三、RQ1正式定义：到底什么叫“proxy 不充分”？

设完整执行状态是

\[
x=
(
shape,
dtype,
layout,
consumer,
cache,
resource,
pipeline,
concurrency,
workload\ distribution,
hardware,\ldots
)
\]

候选配置为

\[
c\in\mathcal C(x).
\]

真正oracle选择是

\[
c^*(x)
=
\arg\min_{c\in\mathcal C(x)}
T(c,x).
\]

但现有selector通常不看完整的 \(x\)，而只提取一个较低维proxy：

\[
p=P(x),
\]

然后执行

\[
\hat c=\pi(P(x)).
\]

因此核心不是问“heuristic准不准”，而是问：

\[
\boxed{
P(x)\text{ 是否包含了做这个决策所需的充分信息？}
}
\]

最强的检验方式不是找一个bad case，而是寻找 **proxy collision**：

\[
P(x_1)=P(x_2),
\]

但是

\[
c^*(x_1)\neq c^*(x_2).
\]

如果这种collision在真实Attention/GQA/MLA/MoE workload中系统出现，那么selector无论把规则调得多漂亮，只要仍只看 \(P(x)\)，就不可能同时对两个状态最优。

定义Proxy Regret：

\[
\boxed{
R_{\mathrm{proxy}}(x)
=
\frac{
T(\pi(P(x)),x)
}{
T(c^*(x),x)
}-1
}
\]

这样科研问题就不再是模糊的：

> “heuristic是不是不好？”

而成为：

> **现代GPU kernel的near-optimal configuration是否能由当前这些低维proxy唯一确定？**

---

# 四、RQ1“问题审判”：十个系统逐一检查

下面是核心表。这里“忽略变量”严格表示：

> **这个具体决策函数本身没有读取这些变量。**

绝不等于“整个框架永远不知道这些变量”。

## 1. SGLang

当前 generic selector 对MHA明确使用GPU代际、speculative模式、asymmetric KV、attention sink、backend可用性等离散条件；MLA路径又使用GPU代际和KV-head count。fileciteturn589file0L1-L2

| 规则 | Proxy | 决策函数没有看的信息 | 自己的repair | 现代触发 |
|---|---|---|---|---|
| Hopper MHA→FA3 | Hopper + no-spec/topk1 | 当前batch、\(S_q\)、\(S_{kv}\)、KV-length distribution、cache state、并发负载 | 其它platform branch | Llama/Qwen/GQA |
| SM100 MHA→TRTLLM，asymmetric KV→FA4 | GPU + asym-KV + spec properties | live sequence mix、page reuse、resource contention | FA4/FI/Triton branches | asymmetric-KV attention |
| MLA Hopper→FA3 / SM100→FI | architecture + attention architecture | latent dims以外的动态request state、decode KV distribution | Triton/AITER | DeepSeek MLA |
| HIP MLA AITER iff kv_heads∈{16,128} | KV-head count | sequence/batch/latency profile | Triton | MLA/GQA-like grouping |

更重要的是框架自己不断加入**model-specific repair**。

DeepSeek DSA在ROCm上，AITER preshuffle paged-MQA可用时page size=64；不可用则直接变成1；XPU又固定128。fileciteturn565file0L1-L10

Qwen3-VL在“大显存Hopper”区域直接采用源码称为 **profiled serving defaults** 的策略，包括默认decode FlashInfer、固定prefill/decode interval等。fileciteturn566file0L1-L13

Nemotron-H又针对W4A16-NVFP4 MoE、SM100、speculative top-k等分别改Marlin、FlashInfer-TRTLLM、TRTLLM-MHA、Triton等。fileciteturn567file0L1-L13

### 对RQ1意味着什么？

这形成一条很强但不能过度解释的证据：

\[
generic\ feature\ tree
\rightarrow
model\ specific\ repair
\rightarrow
profiled\ special\ region
\]

这**不是证明generic rule错了**。

但它证明开发者事实上已经认为：

\[
\boxed{
hardware\ class + model\ class
}
\]

并不足以覆盖所有重要case，必须继续加入quantization、speculation、backend capability、model family甚至VRAM threshold等变量。

这是 RQ1 的正证据。

---

# 五、vLLM：最干净的“proxy不是cost”的例子

vLLM当前KV-layout resolver非常适合作为RQ1证据，因为它的数学结构几乎可以直接从代码写出来。

每个backend给：

\[
L_i=(l_{i1},l_{i2},\ldots)
\]

先求：

\[
C=\bigcap_i set(L_i).
\]

然后计算某个layout获得多少backend的第一偏好票：

\[
vote(l)
=
\sum_i
\mathbf 1[l=l_{i1}].
\]

最后在合法intersection里按vote排序；tie由layout enum order解决。fileciteturn571file0L1-L2

所以当前选择不是：

\[
\arg\min_l T(l)
\]

而近似是：

\[
\boxed{
\arg\max_{l\in C} vote(l)
}
\]

### 它看到的proxy

- backend capability set；
- backend preference order；
- mixed HNC compatibility；
- connector layout preference；
- explicit user override。

### 这个resolver本身没有读取

- attention read latency；
- KV append/write latency；
- prefix-cache reuse；
- KV-transfer traffic比例；
- prefill/decode request mix；
- batch/concurrency distribution；
- L2 locality；
- conversion/packing total cost。

而current selector本身甚至有一条很关键的注释：**单个backend selection看不到它的peer backends；全model KV layout在之后单独解决。** fileciteturn590file0L1-L10

这说明context本身已经被分割。

### vLLM自己的repair

它不是“盲选”：

- backend先限制合法layout；
- mixed HNC时强制block-compact；
- connector preference不兼容则warning并退到candidate[0]；
- user可以显式指定；
- composite backend的packing如果不一致直接报错；
- backend还可分别修改content packing和page size。

所以它有很强的**legality repair**。

问题是：

\[
\boxed{
legality\ repair \neq performance\ repair
}
\]

这正是RQ1要审判的点。

---

# 六、CUTLASS：静态proxy、解析proxy和measurement同时存在

CUTLASS是一个重要“反方证人”，因为它不是纯heuristic系统。

### 6.1 StageCountAuto

源码实际计算的是：

\[
stageBytes
=
bytes(A_{tile})
+
bytes(B_{tile})
+
pipelineMetadata
\]

再取：

\[
N_{stage}
=
\left\lfloor
\frac{SMEM_{capacity}-carveout}
{stageBytes}
\right\rfloor .
\]

fileciteturn593file0L1-L2

proxy是：

\[
P_{\text{stage}}
=
(tile,\ dtype,\ SMEM\ footprint).
\]

它没有直接读取：

- actual global-memory latency；
- TMA overlap efficiency；
- register pressure变化；
- barrier latency；
- effective occupancy；
- consumer/epilogue动态行为。

隐含假设是：

\[
\boxed{
在合法范围内，最大可容纳stage数通常是好默认值
}
\]

这是一条可以直接实验反驳的假设。

### 6.2 SM100 SMEM layout

SM100的shared-layout selection按major和tile divisibility优先：

\[
SW128
\rightarrow
SW64
\rightarrow
SW32
\rightarrow
INTER.
\]

fileciteturn591file0L1-L2

它是典型：

\[
layout=f(tile\ divisibility,dtype,major)
\]

而不是：

\[
layout=
\arg\min_L T(L,\ full\ state).
\]

### 6.3 Raster order

当前规则：

\[
tiles_N>tiles_M
\Rightarrow AlongM
\]

否则AlongN。fileciteturn554file0L1-L17

这提供了非常漂亮的proxy-collision实验：

保持

\[
tiles_N>tiles_M
\]

不变，因此selector必定给同一decision；然后改变K、epilogue、batch、cache footprint、cluster geometry等，看oracle raster optimum是否翻转。

### 6.4 但CUTLASS有强repair

当前Operator API还能用nvMatmulHeuristics按

\[
M,N,K,batch,dtype,layout,alignment,GPU
\]

预测whole-kernel config，再用Profiler实际测量幸存者。fileciteturn557file0L1-L7

官方文档也明确说heuristic的目的是先缩小runtime autotuning space，而不是替代measurement。fileciteturn555file0L1-L7

所以CUTLASS实际上给RQ1提供了一个可能的答案：

\[
\boxed{
static\ proxy
\rightarrow
analytic\ proxy
\rightarrow
measurement
}
\]

问题变成：

> 到底在哪一层measurement已经不再必要？

这比“CUTLASS heuristic不够好”科学得多。

---

# 七、Triton：RQ1最丰富的局部proxy证据

Triton不是一个selector，而是一连串局部decision。

例如：

\[
Memory
\rightarrow Coalesce
\rightarrow MMA
\rightarrow DotOperand
\rightarrow LayoutConversion
\rightarrow Pipeline
\rightarrow WS
\]

当前pass定义明确：

- Coalesce选择cache-friendly layout；
- RemoveLayoutConversions在memory-friendly Blocked与tensor-op-friendly MMA layout之间重写；
- OptimizeThreadLocality处理reduction/gather；
- AccelerateMatmul把layout变成hardware accelerator compatible；
- pipeline根据stage数做async/multibuffer。fileciteturn573file0L1-L2

而源码中有大量作者明确称为“heuristic”的proxy。

### Shared swizzle

\[
score(L)
=
bankConflict_{read}(L)
+
bankConflict_{write}(L)
\]

最小化score；tie再看move rounds。fileciteturn536file0L1-L11

未直接观察：

- instruction overlap；
- occupancy；
- register pressure；
- downstream consumers；
- conversion数量；
- global-memory behavior。

### Pipeliner

源码直接称一个分支为：

> “dirty heuristic for performance drops”

用于某些sync-dot loop跳过MMA pipeline。fileciteturn536file2L30-L40

这是很强的证据，因为开发者自己已经承认这是pattern proxy，而不是performance model。

### Tensor memory/register

TensorMemory utility中还有规则“不使用超过大约一半register，否则容易spill”。fileciteturn572file1L19-L30

于是：

\[
registerFraction
\]

成为spill风险proxy。

### Gather

源码指出warp shuffle大约是zero-bank-conflict SMEM roundtrip成本的一半，并明确说未来需要“more precise heuristic”。fileciteturn572file2L35-L46

这几乎直接给出了RQ1：

\[
\text{固定局部cost proxy}
\stackrel{?}{\Rightarrow}
\text{稳定performance ordering}
\]

### Triton repair

它也不是没有repair：

- layout conversions；
- rematerialization；
- fallback lowering；
- 用户可对 `num_warps/num_stages/tile` 做实测autotune。

但这里有关键边界：

\[
\boxed{
user\ autotune\ variables
\neq
all\ compiler\ internal\ layout\ decisions
}
\]

因此measurement并不会自动校正所有内部heuristic。

---

# 八、Gluon：把“confidence proxy”写得最明确

Gluon AutoLayout非常有意思。

它不是按runtime cost传播layout，而给每条推导路径一个`distance`。

Join/Split/Reshape/Transpose等fuzzy inference会：

\[
distance\leftarrow distance+1.
\]

如果两条layout inference冲突：

\[
distance_1<distance_2
\Rightarrow
L_1\ wins.
\]

如果同distance且非零，则使用stable hash/order破tie；两个distance=0显式seed冲突则error。fileciteturn540file0L1-L7

也就是说：

\[
\boxed{
graph\ inference\ distance
}
\]

其实是一个**可信度proxy**。

它没有声称：

\[
distance
\propto latency.
\]

于是非常自然可以问：

> “离显式layout seed更近”是否也更接近performance-optimal layout？

此外CoalescedEncoding由AxisInfo、numWarps、threadsPerWarp、shapePerCTA生成memory seed，再传播。fileciteturn541file0L1-L7

Gluon的repair很简单：

- 用户显式layout；
- explicit seed conflict就报错；
- auto encoding不能跨未inline函数边界。

它没有内建measurement repair。

因此它是RQ1中非常干净的编译器样本。

---

# 九、TVM：RQ1的重要“反例/对照组”

如果只找支持RQ1的框架，会有confirmation bias。TVM必须作为反方证据。

DefaultCUDA candidate generator当然有很强人工先验：

\[
structure=SSSRRSRS
\]

固定thread bindings、reuse positions、vector lens、cross-thread reduction extents、AutoBind extents等。fileciteturn563file0L1-L2

但是最终selection不是靠这些规则排序，而是：

\[
SpaceGenerator
\rightarrow
EvolutionarySearch
\rightarrow
XGB
\rightarrow
Builder/Runner
\rightarrow
real\ latency.
\]

fileciteturn598file0L1-L10

所以在：

\[
candidate\in G_{\mathrm{TVM}}
\]

条件下，TVM有能力让真实measurement纠正低质量ranking proxy。

这对RQ1提出强挑战：

> 如果measurement feedback已经能把proxy regret压到接近0，那么RQ1可能只是static selector的问题，而不是GPU layout优化的基本问题。

因此TVM是我们之后实验中应该保留的control group。

---

# 十、TIRx：把performance proxy压缩成“priority”

当前dispatcher的数学形式最简单：

\[
candidate
=
sorted(cases,-priority,variant)
\]

逐项：

\[
predicate(c)
\]

全部成功之后就立刻返回该PrimFunc。fileciteturn588file0L1-L10

比如当前已有：

\[
ldgsts\ priority=20
>
vec\_auto\ priority=10
>
fallback\ priority=0.
\]

fileciteturn523file0L1-L20 fileciteturn525file4L53-L63 fileciteturn525file3L40-L50

priority是一种隐含性能proxy：

\[
\boxed{
higher\ priority
\approx
prefer\ this\ legal\ implementation
}
\]

但dispatcher本身没有看候选latency。

### repair

非常完整：

1. predicate reject；
2. implementation自己`DispatchFail`；
3. 尝试下一variant；
4. 用户可强制specific dispatch；
5. 全部失败才error。

这保证了**可靠legality fallback**。

但仍不能保证：

\[
first\ legal\ high\ priority
=
fastest\ legal.
\]

因此TIRx对RQ1也是强证据。

---

# 十一、Hexcute：它可能真正反驳“需要复杂模型”

Hexcute不能被简化成“它cost model缺bank conflict”。

真正的机制是：

\[
Constraints
\rightarrow
legal\ layout/task/instruction\ space
\rightarrow
cost\ ranking.
\]

copy约束：

\[
f\circ p^{-1}=g\circ q^{-1}
\]

MMA约束：

\[
f_{1m}=f_{3m},
\quad
f_{2n}=f_{3n},
\quad
f_{1k}=f_{2k}.
\]

不满足就backtrack。fileciteturn580file0L1-L2

instruction selection又进一步用：

- storage scope；
- element width；
- alignment；
- thread-value layout compatibility。

fileciteturn595file0L1-L2

最后cost model才近似：

\[
cost
=
\#instruction
\times
latency
\]

并建模copy/MMA overlap。fileciteturn581file0L1-L2

它明确假设：

- pipeline fully overlapped；
- 不计`cp_async_wait_group`、`mbarrier`；
- address manipulation negligible；
- arithmetic instruction latency不区分Ampere/Hopper；
- bank conflict不在这个latency model内。

与此同时bank-conflict由另一pass处理，而且它**不是只优化某一个copy**，而是把同一个shared tensor的所有copy consumers放在一起，枚举swizzle并最小化总conflict ways；WGMMA固定的swizzle还会把tensor标成immutable。fileciteturn596file0L1-L2

所以Hexcute是RQ1最重要的反方之一：

> 也许只要constraints足够强，剩下的legal space已经足够“规则化”，简单proxy真的就够了。

这必须通过实验反驳或支持，不能预设答案。

---

# 十二、TileLang：一个框架内部已经存在proxy A/B test

TileLang尤其有价值，因为它自己已经有两个ranking proxy：

默认：

\[
P_1=register\ count
\]

以及optional：

\[
P_2=io\text{-}aware.
\]

当前default仍是`register-count`。fileciteturn544file2L32-L43

而io-aware已经明显更丰富：

\[
vector =
\text{stride-1 contiguous width}
\]

\[
bw =
\text{warp coalescing segments}
\]

\[
issue
=
steps\times threads\times laneBytes
\]

\[
T(S)
\approx
\max(bw,issue)
\]

然后对statements求和。fileciteturn582file0L1-L2

对于无法建模的非affine/swizzle等，直接收取conservative worst-case penalty。

这恰好提供RQ1内部对照：

\[
registerCount
\quad vs \quad
memoryGeometry
\]

哪个proxy更接近oracle？

其layout inference还有repair：严格规则优先；reducer conflict允许由narrow单调widen至wide，最后甚至可用universally-readable replicated fallback。fileciteturn583file0L1-L2

---

# 十三、FlashInfer：目前最适合直接“审判RQ1”的自然实验

FlashInfer当前有两条极干净证据。

### 13.1 平均Q长度

scheduler先计算：

\[
\overline{Q}
=
\frac{1}{B}\sum_i Q_i
\]

然后将这个平均值输入CTA tile rule。fileciteturn528file2L33-L45

例如：

\[
\overline Q\le32
\]

进入short-Q/decode倾向CTA16的路径。fileciteturn528file0L1-L16

因此下面两个batch：

\[
Q_A=[32,32,32,32]
\]

和

\[
Q_B=[1,1,1,125]
\]

有同样平均值32。

selector看到：

\[
P(A)=P(B)=32.
\]

但是二者：

- tail；
- tile wastage；
- CTA work imbalance；
- memory traffic distribution

明显不同。

问题是：

\[
c^*(A)
\stackrel{?}{=}
c^*(B).
\]

这是一个近乎教科书式的proxy-sufficiency可证伪问题。

注意：这里不能先声称它们一定不同。**是否不同就是实验要回答的问题。**

### 13.2 Split-KV mode

当前源码对FMHA固定：

\[
cluster
\succ
gmem\_separate
\succ
gmem\_inline.
\]

MLA 1CTA则：

\[
cluster
\succ
gmem\_separate.
\]

fileciteturn570file0L1-L13

尤其关键的是函数参数包括：

\[
tile\_size_q,\ head\_dim,\ head\_dim\_per\_cta_v
\]

但这些shape参数没有进入当前排序公式。

源码还明确解释：

> 不使用shape-specific measured crossover table，而保持structural policy。

所以可以构造：

\[
P(x_1)=P(x_2)
\]

即相同family/topology/split/support状态，

但改变：

- head_dim；
- Q tile；
- KV length；
- batch；
- resource footprint；

然后直接比较：

\[
T_{cluster},
T_{separate},
T_{inline}.
\]

如果winner发生翻转，这就是RQ1极强的直接证据。

如果完全不翻转，那么反而是非常漂亮的反证：

> structural policy确实捕捉到了足够信息。

---

# 十四、把现代 Attention / GQA / MLA / MoE 触发条件放到同一矩阵

这一步很重要，因为否则容易又退化成toy GEMM问题。

| Workload | 哪些proxy最容易失真 | 已有框架证据 | RQ1最关键的hidden variables |
|---|---|---|---|
| **MHA prefill** | tile shape、backend、pipeline、layout | SGLang、Triton、TVM、FlashInfer | \(B,Q,K\) distribution、mask、cache、register/SMEM pressure |
| **GQA decode** | KV representation、head mapping、split KV | vLLM、FlashInfer、SGLang | \(H_q/H_{kv}\)、KV length distribution、page locality、reuse |
| **MLA decode** | latent layout、split-KV topology、backend | SGLang DeepSeek、vLLM MLA、FlashInfer MLA | latent dim、RoPE dim、KV pages、split factor、1CTA/2CTA、occupancy |
| **Speculative verify** | backend/top-k/query tile | SGLang Nemotron、vLLM adaptive verification、FlashInfer short-Q | verification width、acceptance pattern、batch heterogeneity |
| **Attention sinks/SWA/sparse** | backend/layout compatibility | SGLang、vLLM | effective KV span、window locality、consumer-specific cache behavior |
| **MoE grouped GEMM** | kernel/backend/layout/tile | SGLang Nemotron/HYV4、Hexcute mixed-type artifact、CUTLASS grouped/MoE families | expert-token skew、per-expert M/N、quant type、routing skew、A2A overlap |
| **Quantized Attention/MoE** | content packing、instruction choice | vLLM NVFP4/TurboQuant、SGLang W4A16/MXFP8、Hexcute instruction selection | scale metadata locality、mixed width、dequantization resource cost |

其中MoE尤其值得注意。

同样总token数：

\[
N_\text{tokens}=4096
\]

可以是均匀：

\[
[512,512,\ldots]
\]

也可以高度skew：

\[
[3000,500,300,\ldots].
\]

如果selector只看total tokens/model quantization，而不看expert distribution，那么也会产生完全类似的proxy collision。

所以RQ1不是只针对Attention。

---

# 十五、框架自己的 fallback / repair 是不是已经把问题解决了？

这是“审判”的反方部分。不能只找罪证。

各框架实际上已经有大量repair：

| 框架 | repair |
|---|---|
| SGLang | generic fallback、phase-specific backend、model-specific overrides、profiled serving defaults |
| vLLM | compatibility intersection、block-compact restriction、backend-specific packing、explicit override、connector fallback |
| CUTLASS | manual template override、analytical nvMMH、真实Profiler measurement |
| Triton | conversion/rematerialization、special-case heuristics、user autotune |
| Gluon | explicit layout、conflict error |
| TVM | learned cost model + actual hardware Runner |
| TIRx | priority fallback chain、forced variant |
| Hexcute | constraint backtracking、alternate instructions、bank-conflict resolver、analytic cost ranking |
| TileLang | strict/common/free hierarchy、alternative cost model、wide/replicated fallback |
| FlashInfer | support predicates、exact occupancy checks、cluster→GMEM fallback |

这使得我们不能提出：

> “现有heuristic不够精确，所以做一个更好的heuristic。”

这个问题太弱。

真正科学的问题是：

\[
\boxed{
这些repair之后，
是否仍存在系统性的proxy collision？
}
\]

以及：

\[
\boxed{
需要补充多少状态信息，
才能使决策接近充分？
}
\]

---

# 十六、RQ1的可证伪假设

现在可以正式写成一个Null hypothesis和一个research hypothesis。

### \(H_0\)：现有proxy基本充分

对目标workload分布 \(\mathcal W\)，存在一个小阈值 \(\epsilon\)，使：

\[
\Pr_{x\sim\mathcal W}
\left[
R_{\mathrm{proxy}}(x)>\epsilon
\right]
\approx0.
\]

也就是说，即使selector看不到完整状态，现有proxy仍足以让其选择落在near-optimal region。

例如实验可把：

\[
\epsilon=5\%
\]

作为工程上的near-optimal判据，但5%不是理论常数，只是实验阈值。

如果这个 \(H_0\) 成立，那么RQ1基本被否掉：

> 复杂dynamic context没有科研必要性，现有低维proxy已经足够好。

---

### \(H_1\)：存在系统性proxy collision

存在真实workload状态：

\[
x_1,x_2
\]

满足：

\[
P(x_1)=P(x_2)
\]

但是：

\[
c^*(x_1)\neq c^*(x_2),
\]

并且使用同一静态决策导致：

\[
R_{\mathrm{proxy}}(x_i)>\epsilon.
\]

更强版本要求：

\[
\Pr[R_{\mathrm{proxy}}>\epsilon]>\delta
\]

在真实workload分布上有非平凡质量，而不是万里挑一的corner case。

这才构成科研问题。

---

# 十七、RQ1进一步分成三个真正可检验的子假设

这一层很关键，否则RQ仍然太大。

### RQ1-H1：Proxy Collision Hypothesis

相同现有proxy的状态中，oracle winner会发生稳定翻转：

\[
P(x_1)=P(x_2),
\quad
c^*(x_1)\ne c^*(x_2).
\]

最强对象：

- FlashInfer same average-Q / different distribution；
- FlashInfer same structural split-KV class / different head-dim or KV length；
- CUTLASS same M/N raster relation / different K/resource/cache state；
- vLLM same backend votes / different serving mix；
- TIRx same predicates / different shapes/resource states。

如果winner不翻转，这个子假设失败。

---

### RQ1-H2：Repair Concentration Hypothesis

已有框架中大量special-case repair并非随机，而会集中在proxy信息不足的区域。

例如：

\[
GenericPolicy
\rightarrow
DeepSeekOverride
\]

\[
GenericPolicy
\rightarrow
Qwen3VLProfiledDefault
\]

\[
TritonGeneralRule
\rightarrow
FusedAttentionSpecialCase
\]

\[
RegisterCount
\rightarrow
IOAware
\]

\[
StaticBuilder
\rightarrow
nvMMH
\rightarrow
Profiler.
\]

如果repair区域和真实proxy regret完全不相关，那么这条假设失败。

注意：**repair本身只能作为线索，不作为证明。**

---

### RQ1-H3：Compact Augmentation Hypothesis

如果proxy不充分，最重要的科学问题不是“把所有runtime状态全塞进去”，而是：

> 是否只需增加非常少量的missing variables就可以消除大部分regret？

令现有：

\[
P(x)
\]

扩展为：

\[
P'(x)
=
[P(x),z_1,z_2,\ldots,z_k].
\]

研究：

\[
k\ll dim(x)
\]

时是否已经有：

\[
R_{P'}\ll R_P.
\]

例如可能只需：

- variance/tail of Q；
- KV-length distribution；
- GQA ratio；
- expert-token skew；
- estimated register/SMEM pressure；
- consumer count；
- transfer ratio。

如果必须读取完整runtime trace甚至实测所有candidate才能准确选择，则compact augmentation假设失败。

这实际上比“训练一个ML cost model”更有科学价值。

---

# 十八、RQ1最关键的“杀手级”可证伪case

现在已经可以明确哪些实验以后最有判别力，但仍然不写代码。

### Case A：FlashInfer Mean-Q Collision

固定：

\[
\bar Q
\]

让scheduler看到完全一样的proxy。

改变：

\[
Var(Q),\quad Q_{max},\quad distribution.
\]

如果oracle CTA tile不同：

\[
\boxed{\bar Q\text{不是决策充分统计量}}
\]

这是最干净的一条。

---

### Case B：FlashInfer Split-KV Structural Collision

固定：

- `family`
- `topology`
- `split_kv`
- support status

因此policy mode-order完全一样。

改变：

\[
head\_dim,
KVlen,
Qtile,
batch,
resource\ footprint.
\]

如果：

\[
cluster\leftrightarrow gmem
\]

winner翻转，则直接反驳structural-only ranking。

---

### Case C：vLLM Persistent KV Collision

固定：

- backend set；
- supported-layout lists；
- preference vote。

因此：

\[
resolve\_kv\_cache\_layout()
\]

必定选择同一个layout。

改变serving distribution：

\[
prefill:decode,
\quad
KV-transfer:local-read,
\quad
reuse,
\quad
sequence-length distribution.
\]

评价：

\[
T_{total}(L)
=
T_{write}
+
T_{attention}
+
T_{transfer}
+
T_{management}.
\]

若最佳persistent layout翻转，则证明compatibility+vote不是workload-sufficient。

---

### Case D：CUTLASS Raster Collision

固定：

\[
tiles_N>tiles_M
\]

所以always AlongM。

改变：

\[
K,\quad stages,\quad epilogue,\quad batch,\quad cluster.
\]

如果AlongN反而更快：

\[
aspect\ ratio
\]

不是sufficient proxy。

---

### Case E：TIRx Priority Collision

找到两个同时满足predicate的合法variants：

\[
c_1,c_2.
\]

priority固定：

\[
priority(c_1)>priority(c_2).
\]

扫shape/resource状态。

如果存在：

\[
T(c_1,x_1)<T(c_2,x_1)
\]

而

\[
T(c_1,x_2)>T(c_2,x_2),
\]

那么固定priority不能是充分selection policy。

---

### Case F：MoE Expert-Skew Collision

保持：

- total token count；
- dtype；
- quantization；
- number of experts；

只改变expert-token histogram。

如果相同backend/tile policy下oracle kernel/layout发生翻转，这说明：

\[
total\ workload\ size
\]

不能代表grouped GEMM真实shape distribution。

这条会把RQ1从Attention推广到MoE。

---

# 十九、什么结果会让我们直接放弃 RQ1？

必须提前写清楚，否则以后容易“怎么跑都有故事”。

如果实验发现：

1. FlashInfer相同mean-Q下，即便distribution极端变化，当前CTA selection仍始终在oracle 3–5%以内；
2. split-KV cluster/separate/inline的最优边界基本由当前structural eligibility决定；
3. CUTLASS static rule在大量shape/resource变化下几乎从不发生winner flip；
4. TIRx同priority-order下alternative legal variants不存在有意义的crossover；
5. vLLM不同serving mix下同一个persistent layout始终near-optimal；
6. Hexcute/TileLang简单proxy的top-1/top-k ranking稳定；
7. TVM真实measurement相对static proxy只带来极少改善，

那么：

\[
\boxed{\text{RQ1应当被判定为弱问题甚至否定}}
\]

这反而会得到一个有价值的科学结论：

> 现代GPU高性能layout空间具有足够强的结构规律，使很低维的proxy已接近decision-sufficient。

---

# 二十、什么结果才算 RQ1 被真正证实？

不是找到一个“慢20%”的case。

至少要满足三层。

第一层是 **collision**：

\[
P(x_1)=P(x_2),
\quad
c^*(x_1)\neq c^*(x_2).
\]

第二层是 **systematicity**：

collision在Attention/GQA/MLA/MoE一组真实shape region中反复出现，而不是单个corner case。

第三层是 **explanation**：

能够找到缺失变量 \(z\)，使：

\[
P'(x)=[P(x),z]
\]

显著减少：

\[
R_{\mathrm{proxy}}.
\]

如果只满足前两层但不知道为什么winner翻转，那么研究仍然偏benchmark observation。

最强结果会是：

\[
\boxed{
\text{现有proxy}
\rightarrow
\text{系统性collision}
\rightarrow
\text{识别missing state}
\rightarrow
\text{低维augmentation恢复near-optimality}
}
\]

这才是一个完整的科研贡献链。

---

# 二十一、RQ1的最终“审判结论”

基于目前所有源码证据，我不会写成：

> “现有layout heuristic是错的。”

目前证据只足以支持更严谨的结论：

\[
\boxed{
\textbf{现有系统普遍把高维GPU性能决策压缩成低维proxy，}
}
\]

而且不同系统选择了截然不同的proxy：

\[
\begin{aligned}
CUTLASS &: SMEM\ capacity,\ divisibility,\ aspect\ ratio,\ analytic\ GEMM\ model\\
Triton &: coalescing,\ bank\ conflict,\ register\ pressure,\ IR\ patterns\\
Gluon &: inference\ distance,\ AxisInfo\\
TVM &: schedule\ grammar + learned/measurement\\
TIRx &: predicates + priority\\
Hexcute &: constraints + instruction\ latency\\
TileLang &: register\ count\ /\ memory\ traffic\\
vLLM &: compatibility + preferences\\
SGLang &: model/hardware feature tree\\
FlashInfer &: mean\ Q,\ structural\ mode,\ occupancy,\ KV\ token\ work
\end{aligned}
\]

更重要的是，它们各自又提供了不同程度的repair：

\[
fallback,\ override,\ backtracking,\ measurement,\ profiling,\ special\ cases.
\]

所以最终RQ1不是：

> “哪个heuristic最好？”

而是：

\[
\boxed{
\begin{array}{l}
\textbf{对于现代Attention/GQA/MLA/MoE，现有selector所观测的proxy}\\
\textbf{是否构成近似decision-sufficient statistics？}\\
\textbf{如果不是，最小还需要哪些状态变量才能消除系统性selection regret？}
\end{array}
}
\]

我认为这个RQ现在已经通过了“值得研究”的第一轮审判，原因不是我们已经证明它成立，而是同时满足了四个条件：

\[
\boxed{
\text{跨框架普遍存在}
+
\text{源码直接暴露proxy}
+
\text{存在框架自身repair}
+
\text{存在非常明确的反证实验}
}
\]

其中最强的一手证据是FlashInfer明确选择“structural policy而非shape-specific crossover”，vLLM明确以compatibility/preference投票解决persistent layout，TIRx明确用priority-first-success，Gluon明确以inference distance解决冲突；另一端TVM和CUTLASS又提供了measurement/hybrid路线作为真正的反方。这使RQ1不是人为制造的“框架缺陷”，而是不同系统设计之间实际存在的科学张力。fileciteturn570file0L1-L13 fileciteturn571file0L1-L2 fileciteturn588file0L1-L10 fileciteturn540file0L1-L7 fileciteturn598file0L1-L10

因此我建议下一轮严格按你设定的流程：**不再扩展RQ1，也不写代码，正式开始RQ2“决策边界/可分性”的问题审判**，同样逐框架做到 `源码规则 → proxy → omitted variables → repair → Attention/GQA/MLA/MoE触发 → H0/H1 → 杀手级反证case`。RQ2审判完成以后，再进入RQ3。

## 用户

证据链不完整请继续补充。然后把证据链也凝炼为问题。然后从RQ1开始做一条完整的“问题审判”：所有相关框架源码规则 → 每个规则使用的proxy → 它忽略了哪些变量 → 框架自己的fallback/repair → 现代Attention/GQA/MLA/MoE触发条件 → 可证伪的科研假设。 RQ1彻底完成之后再审判RQ2，然后后续RQ。这样可以防止又回到前几轮“问题列了很多，但证据深度不够”的状态 。当然如果新的证据链形成新的RQ 也请补充。

## 模型回复

这轮我会把“证据链缺口”优先补齐，不先扩展新RQ。重点补三类此前仍薄弱的地方：**(1) 每个框架内部真正参与layout/mapping选择的规则分支；(2) 这些规则的proxy与fallback；(3) 现代Attention/GQA/MLA/MoE里能触发这些分支的具体条件**。完成后再重新凝练RQ，并只把RQ1做成完整“问题审判”。

## 模型工具调用参数

{"paths":["GitHub"],"query":"fetch"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton Gluon AutoLayout 精确源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查找 Triton Gluon 目录内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取布局推断工具源文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 InferCoalesced 编码推断文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 vLLM 后端布局与块大小规范

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 vLLM KV 缓存布局支持

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查找首选块大小配置

## 模型思考摘要

检索 Triton Gluon 布局源码及 vLLM 缓存规范

## 模型回复

目前 vLLM 的证据比之前更强：它不是只有“一个模型级KV layout投票”。源码里其实有三层彼此独立的表示决策：**backend支持的物理layout集合 → backend preferred block/page size → backend-specific cache spec/packing**。这说明它的persistent representation不是单变量决策，而是多个局部规则拼成的；后面RQ1审判必须分别检查，不能只拿“投票选layout”代表整个vLLM。

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取并分析 vLLM 后端基础自定义规范

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 customize_spec 覆盖实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

补充 SGLang 页面大小规则与模型覆盖证据

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 SGLang 配置覆盖逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

定位注意力后端与页面大小覆盖函数

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检查 MiniMax 模型覆盖实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Triton 经典启发式实现细节

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取并分析 Triton 源码文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析通用置换工具中的选择成本

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 中冲突与轮次选择逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 TVM 布局优化规则与假设

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Apache TVM 中的布局重写函数

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取并重写布局文件内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

列出 TIRx CUDA 拷贝注册文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检查初始化文件中的显式优先级

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

核查 CUDA 内存拷贝注册优先级

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 CUDA 复制原语初始化文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 `_is_gmem_smem` 相关注册位置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 CUDA 向量自动复制实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 TIRx 高优先级 ld/st 矩阵分派

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 CUDA Tile Copy 回退实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析TileLang布局代价模型实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取布局推断 README 与选择实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索布局成本模型配置与评分默认值

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 Hexcute/Hidet 启发式策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索自动注释实例化相关代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取并检查自动注解实例化文件

## 模型工具调用参数

{"uri":"/response/turn633","query":"heuristic"}

## 模型工具调用参数

{"uri":"/response/turn633","query":"undetermined"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索布局压缩与内存约束实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 FlashInfer 调度策略与成本模型

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取调度器代码与CTA规则定义

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 utils.cuh 中的 CTA 规则

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 FlashInfer 中 FA2 CTA Tile 阈值逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索返回64阈值函数

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析FlashInfer调度器工作负载均衡策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 FlashInfer 中的最小成本调度实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 CUTLASS 矩阵乘启发式源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 CUTLASS 光栅排序实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 nvMatmulHeuristics 启发式文档与实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 Triton 布局启发式规则

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Coalesce 源码这里只精品

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理注意力后端选择证据链

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 CUDA 选择逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索_get_backend_priorities后端优先级规则

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 CUDA 平台代码早期内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析 vLLM 后端优先级缺陷

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 MLA Token 速度实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

评估深度学习框架性能调优策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 CUTLASS SM100 调度与布局选择策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Triton AccelerateMatmul 启发式实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Triton 布局变换性能启发式

## 模型思考摘要

凝练研究问题
凝练研究问题

## 模型思考摘要

扩展研究问题
扩展了研究问题

## 模型思考摘要

分析并梳理多框架布局、调度与后端启发式策略

## 模型回复

又补出两条此前没有充分利用的强证据。第一，vLLM 的 CUDA backend priority 源码自己写着 `TOKENSPEED_MLA` 在特定合法配置下“**bs≈8以后胜出、bs≤2回退**”，但backend selector的输入结构没有当前batch size，且仍按固定priority取第一个合法backend；这是一个源码内生的、几乎直接指向“proxy缺变量”的证据。  
第二，TileLang维护者已经保留了 `register-count` 与 `io-aware` 两套layout ranking，并明确把两者分歧case称为需要硬件benchmark验证的“具体性能主张”；其中 `broadcast_read` 已记录默认register-count保留thread-collapse pathology，而io-aware会选择不同layout。 这比单纯推断“某proxy可能不足”更强，因为框架源码/验证体系自己已经暴露了proxy之间的决策冲突。

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

核查 FlashInfer Split-KV 策略源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

评估布局选择器与经验代价模型局限

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取覆盖参数配置文件内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索并定位 `_page_size_default` 相关代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

Fetching vLLM KV Cache Interface Details

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检查 KV 缓存布局枚举

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检查 FlashInfer KV 缓存布局支持

## 模型思考摘要

核查 FlashInfer 与 vLLM KV 缓存布局及 Split-KV 配置限制

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 TileLang 布局推理代价模型

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 TileLang 布局推导与回退逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索广播读取布局推断源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取并分析 TileLang reduce-broadcast 案例

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Triton 自动 Warp 专门化实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 RemoveLayoutConversions 算法实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Triton 中昂贵加载启发式定义

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取布局转换启发式实现

## 模型思考摘要

梳理 vLLM 决策层
уточению vLLM 选择器
梳理并 уточению vLLM 决策选择器

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

完成证据审计并开展 RQ1 试验

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

Auditing Evidence and Completing the RQ1 Trial

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析 SGLang 量化测试与页面大小策略

## 模型思考摘要

完成 TileLang 与 Triton 布局推理审计及 RQ1 试验

## 模型思考用时

思考了 11m 54s

## 模型回复

这次继续往源码里补，确实又发现了几处此前不能算“证据链闭合”的缺口。最重要的不是又多找了几个框架特例，而是找到了几条能直接把“规则—proxy—遗漏变量—repair—现代 workload”连起来的源码证据。其中最强的两条是：

1. **vLLM 源码自己记录了 MLA backend 的 batch-size 性能 crossover，但当前 backend priority selector 并没有 batch size 这个输入。** 当前代码注释明确写 `TOKENSPEED_MLA` “wins past bs≈8, regresses at bs≤2”，而 `_get_backend_priorities()` 的输入是 MLA、GPU capability、head 数、KV dtype、causal/head-size/mm-prefix 等，并不包含 live batch size；合法性过滤之后直接取静态 priority 最高者。fileciteturn650file0L1-L7 fileciteturn648file0L1-L7
2. **TileLang 当前默认 `register-count` layout proxy 已有源码维护者自己记录的具体 pathology。** `broadcast_read` case 明确写出默认策略因为“寄存器更少”选择部分复制布局，结果把 parallel j 轴塌缩成每个线程串行遍历；`io-aware` 使用 issue/coalescing 信息后选择另一个布局。fileciteturn666file0L1-L7 更重要的是，维护目录明确把两套cost model发生分歧的case称为需要在真实硬件上校准的“concrete claim”。fileciteturn629file0L1-L7

所以，上一轮的 RQ1 方向没有被推翻，反而从“合理猜测”升级成了**有若干直接源码反例/已知 crossover 支撑的科学问题**。但我需要先把完整证据矩阵补齐，再下结论。

下面所有“忽略变量”都严格表示：

\[
\boxed{\text{该具体 selector / rule 没有读取这个变量}}
\]

不是说“整个框架不知道这个变量”。

---

# 一、补完后的细粒度证据链总表

为了避免再次把不同抽象层混在一起，我把每一行都压成同一个形式：

\[
\boxed{
\text{规则}
\to
\text{选择对象}
\to
\text{proxy}
\to
\text{隐含假设}
\to
\text{未观测变量}
\to
\text{repair/fallback}
}
\]

## 1. SGLang

| 源码规则 | 选择对象 | 使用的 proxy | 隐含假设 | 此规则未读取 | repair / fallback |
|---|---|---|---|---|---|
| generic MHA：Hopper→FA3；SM100→TRTLLM-MHA，asym-KV→FA4；HIP→AITER；否则FI→Triton | attention backend | GPU代际、spec状态、asymmetric KV、sink、backend availability | 这些离散特征可以把最佳backend划成稳定区域 | 当前B、Q长度分布、KV长度分布、cache locality、并发 | FlashInfer→Triton；architecture overrides |
| MLA：Hopper→FA3；SM100→FI；HIP仅KV heads=16/128→AITER，否则Triton | MLA backend | GPU、KV-head count | hardware+head family基本决定backend | batch、KV length、latent workload distribution | Triton fallback |
| DeepSeek DSA ROCm preshuffle可用→page64，否则page1；XPU→128 | KV page geometry | backend capability/platform | kernel capability class决定合适page geometry | live reuse、fragmentation、KV length distribution | page 1/64/128 override |
| DeepSeek DSA/MLA CP rules | attention/MoE/parallel representation | architecture、CP mode、TP/DP | 已验证结构可作为safe region | 动态batch/communication contention | assert/禁用/强制DeepEP等 |
| Qwen3-VL “profiled serving path” | backend/cache/transport | Qwen3-VL、Hopper、VRAM≥60GB、高并发阈值 | 一个profiling得到的区域能静态复用 | 实际image reuse、请求组成、动态concurrency变化 | generic policy |
| Nemotron W4A16 NVFP4→Marlin；SM100→FI-TRTLLM等 | MoE runner | quant type、GPU、A2A | quant+arch足以划分runner region | expert skew、per-expert M/N、batch | multiple backends |
| Nemotron spec top-k≤1 SM100→TRTLLM-MHA+page64，否则Triton | attention/page | spec algorithm、top-k、arch | speculative模式类别可代理实际verify workload | actual acceptance/query-width distribution | Triton/FlashInfer draft |
| MiniMax-M3 FP8 e4m3 SM100→TRTLLM-MHA，否则FA4；page128 | sparse/dense attention representation | dtype、arch、backend support | dtype/backend组合决定最佳路径 | sparse selectivity、batch、KV length | Triton widening path / kill switch |
| MiniMax-M3 MXFP8 MoE→DeepGEMM，mixed→FI-TRTLLM-routed；BF16 DeepGEMM修复为Triton | MoE runner | quantization/model identity | quantization是强proxy | routing skew、expert workload distribution | explicit corruption repair |

generic selector源码本身就是一棵feature tree；DeepSeek、Qwen3-VL、Nemotron、MiniMax-M3又在这棵树外围不断增加model-specific规则。fileciteturn589file0L1-L2 fileciteturn565file0L1-L10 fileciteturn566file0L1-L13 fileciteturn567file0L1-L13 fileciteturn612file0L1-L7

这里最值得科研关注的不是“override很多”，而是：

\[
\text{generic proxy}
\rightarrow
\text{发现不够}
\rightarrow
\text{增加 model-specific context}
\]

这是一种**手工逐步补充决策状态**的体系。

---

# 二、vLLM：此前证据链缺失最大，这一轮已经补成四层

## 2.1 Backend selection

CUDA selector首先生成静态backend priority，然后逐个执行合法性检查，最终：

\[
\boxed{
backend=
\arg\min_{\text{valid backend}}
priority
}
\]

不是：

\[
\arg\min T_{\text{measured}}.
\]

fileciteturn648file0L1-L7

其中 MLA/SM100 当前规则特别值得注意：

- quantized KV：FlashInfer sparse优先；
- BF16 KV、head数≤16：FlashInfer sparse优先于FlashMLA sparse；
- head数更高反过来；
- 主backend顺序中 FlashInfer MLA 在 TokenSpeed MLA 之前；
- 但源码同时明确记录：TokenSpeed 在 `bs≈8` 后胜出、`bs≤2` 回退。fileciteturn650file0L1-L7

而实际TokenSpeed `supports_combination()`只检查：

- Blackwell；
- FP8 KV；
- DeepSeek R1 MLA dimensions；
- package可用性等，

并没有用batch-size解决这个性能crossover。fileciteturn652file0L1-L7

这是目前 RQ1 最硬的一条证据：

\[
P(x_1)=P(x_2)
\]

因为所有selector输入都可以保持相同，只变化：

\[
B_1=1,\quad B_2=16,
\]

源码自己却告诉我们两个区域有不同performance behavior。

---

## 2.2 Persistent KV physical layout

当前logical cache统一写为：

\[
[L,B,H,N,C]
\]

但允许六种物理stride permutation。fileciteturn662file0L1-L7

不同backend分别声明自己支持/偏好的layout，例如：

- CPU仅LBHNC；
- HPC仅LBNHC；
- B12X支持LBHNC/BLHNC；
- MLA indexer只LBHNC；
- QSA cache偏BLNHC/BLHNC；
- FlexAttention只有LBNHC，因为它要求 `(B,N)` 能zero-copy flatten。fileciteturn605file0L1-L12

resolver做：

\[
C=\bigcap_i S_i
\]

然后：

\[
priority(L)
=
\sum_i
\mathbf1[L=\text{backend}_i\text{ first preference}]
\]

再按这个计数排candidate；mixed-HNC会强制block-compact。fileciteturn672file0L1-L7

最后connector不兼容时退到：

\[
candidates[0].
\]

fileciteturn673file0L1-L7

所以这个selector的proxy是：

\[
\boxed{
backend\ capability
+
ordinal\ preference
+
shape\ compatibility
}
\]

而不是：

\[
T_\text{attention}
+
T_\text{append}
+
T_\text{transfer}
+
T_\text{reuse}.
\]

---

## 2.3 Page/block geometry

backend还有独立的：

```text
get_preferred_block_size()
```

base rule是：

1. 默认size合法→保留；
2. 否则选择一个最小的legal supported size。

而具体backend又有：

- AITER unified固定64；
- sparse MLA可以固定256；
- FA4某些head dim路径要求更大的block等。fileciteturn606file0L1-L22 fileciteturn607file0L1-L7

也就是说：

\[
\text{stride layout}
\quad\text{和}\quad
\text{page geometry}
\]

是两个分开的规则。

---

## 2.4 KV content packing

`customize_spec()`又是第三种representation：

> backend可以因为kernel要求而修改KV cache spec/packing。fileciteturn607file0L1-L7

已有实现包括：

- native ROCm：K/V拆成两个head groups；
- AITER unified：K/V继续放content dimension；
- Triton quantized KV：head后inline scale；
- TurboQuant：K+V打进一个slot；
- FlashInfer NVFP4：packed FP4 + FP8 block scale；
- MLA：latent data后加入scale pair；
- composite backend若两边packing不同直接报错。fileciteturn608file0L1-L13

因此以前说：

> “vLLM只决定一种KV layout”

是不够准确的。

实际是：

\[
\boxed{
Backend
\rightarrow
PhysicalAxisLayout
\rightarrow
PageGeometry
\rightarrow
ContentPacking
}
\]

这四层相互影响，但目前不是一个统一cost optimization。

---

# 三、CUTLASS/CuTe：这次补到“规则真正假设了什么”

## 3.1 `StageCountAuto`

官方源码文档不是简单说“自动选stage”，而明确写：

> 最大化shared-memory usage，**假设1 threadblock / SM occupancy**。fileciteturn653file0L1-L32

即：

\[
stageBytes
\approx
A_{tile}
+
B_{tile}
+
pipelineStorage
\]

\[
stages
=
\left\lfloor
\frac{SMEM-carveout}{stageBytes}
\right\rfloor.
\]

fileciteturn593file0L1-L2

因此它有非常清楚的假设：

\[
\boxed{
\text{在该kernel family中，多stage带来的latency hiding}
>
\text{提高CTA residency的收益}
}
\]

至少默认builder是这么选择的。

这对 MoE 很重要，因为 Blackwell grouped/MoE示例本身同样使用 `StageCountAuto`，即它不是只针对理想大GEMM。

---

## 3.2 SM100 shared-memory layout

`sm100_smem_selector()`不是搜索bank-conflict/latency，而是按：

\[
dtype,\ major,\ BLK_{MN},BLK_K
\]

选择第一个能整除的最大layout：

\[
SW128
\rightarrow SW64
\rightarrow SW32
\rightarrow INTER.
\]

fileciteturn591file0L1-L2

这等价于：

\[
L=f(tile\ divisibility)
\]

而不是：

\[
L=\arg\min T.
\]

---

## 3.3 Raster-order

当前heuristic：

\[
tiles_N>tiles_M
\Rightarrow AlongM,
\]

否则：

\[
AlongN.
\]

fileciteturn643file1L25-L43

它的性能proxy非常低维：

\[
P=(tiles_M,tiles_N).
\]

---

## 3.4 CUTLASS也给出了RQ1的反方

不能据此说CUTLASS“只能heuristic”。

当前CUTLASS已经支持用nvMatmulHeuristics按estimated performance排序operators。fileciteturn644file3L65-L76

而且用户可以：

- 手工显式覆盖；
- 使用profiling/measurement验证候选。

所以CUTLASS代表的是：

\[
\boxed{
static\ structural\ rule
\rightarrow
analytic\ heuristic
\rightarrow
optional\ measurement
}
\]

这非常重要，因为RQ1真正要比较的不是“有没有heuristic”，而是：

> **什么时候低维rule足够，什么时候必须升级到更丰富的performance evidence？**

---

# 四、Triton：不能再概括成“compiler自动layout”

现在源码可以把它拆成至少六类决策。

| 决策 | proxy/规则 | 源码证据 |
|---|---|---|
| Coalesce | memory access/AxisInfo→coalesced layout | load/store的cache-friendly encoding |
| RemoveLayoutConversions | expensive load/store为layout anchor；dot/TMEM等为另类anchor；传播后解决冲突 | 明确四阶段算法 |
| SMEM swizzle | vectorization + bank geometry + basis choice | GenericSwizzling |
| pipeline | IR operation latency/stage；某些sync-dot loop直接跳MMA pipeline | 源码称“dirty heuristic” |
| gather | shape + existing layout；只尝试warp-synchronous family | explicit heuristic |
| MMA packed-K kWidth | packed-K pattern heuristic | 源码自己说以后可以做layout backprop再检查load contiguity |

`RemoveLayoutConversions`的当前算法特别重要：

1. 找“必须保留”的layout anchor；
2. 每个anchor向descendants传播；
3. 一个value可能收到多个互相冲突的layouts；
4. resolve conflict；
5. 必要时插入`convert_layout`；
6. 再做rematerialization/hoisting。fileciteturn671file0L1-L2

其中“expensive load/store”偏Blocked encoding，而tensor op偏MMA encoding。fileciteturn669file0L1-L19

这已经说明：

\[
L_\text{memory}\neq L_\text{tensorcore}
\]

在Triton不是抽象猜测，而是compiler本身需要解决的冲突。

更直接的是一些源码注释：

- Pipeline中明确称某条件为 **“dirty heuristic for performance drops”**。fileciteturn656file1L28-L51
- Gather lowering直接用warp shuffle与SMEM roundtrip的近似相对成本，并表示需要更精确heuristic。fileciteturn572file2L35-L46
- TMEM/register utility使用“不超过约一半register，否则容易spill”的规则。fileciteturn572file1L19-L30
- packed-K MMA kWidth代码自己写出未来应该尝试更大kWidth、做layout backprop、查看上游load contiguity。fileciteturn655file0L1-L25
- gather layout当前只尝试warp-synchronous方案。fileciteturn645file0L1-L19

所以Triton是目前“局部proxy很多、逐pass repair很多”的最典型样本。

---

# 五、Gluon：此前把“explicit layout”说得太粗，现在已经找到真正的冲突算法

Gluon Auto Encoding有非常明确的“confidence proxy”。

每一个layout inference记录：

\[
(\text{encoding},distance).
\]

对于：

- Join
- Split
- Reshape
- Transpose

这样的fuzzy inference：

\[
distance\leftarrow distance+1.
\]

如果两个encoding冲突：

\[
d_1<d_2\Rightarrow L_1
\]

如果：

\[
d_1=d_2>0,
\]

则用stable hash形成确定性tie break。

如果：

\[
d_1=d_2=0,\quad L_1\ne L_2
\]

说明两个显式seed冲突，直接error。fileciteturn602file0L1-L7

所以Gluon不是简单“用户手工layout”。

它实际有：

\[
\boxed{
\text{explicit/coalesced seed}
+
\text{graph inference}
+
\text{distance-based conflict selection}
}
\]

`CoalescedEncoding` seed又来自：

- AxisInfo；
- numWarps；
- threadsPerWarp；
- shapePerCTA；
- CGA layout。

fileciteturn603file0L1-L7

因此它的核心假设之一其实是：

\[
\boxed{
\text{离explicit/coalesced seed更近的layout推断更可信}
}
\]

注意这个“可信”是**推断可信度**，并没有证明是**性能可信度**。

这正是RQ1里非常漂亮的一种不同proxy。

---

# 六、TVM：既是支持证据，也是RQ1的重要反例

## 6.1 Candidate space本身强依赖规则

当前DefaultCUDA不是“任意schedule搜索”。

它预先规定：

- `"SSSRRSRS"` tiling grammar；
- block/vthread/thread binding；
- vector load lengths；
- shared/local reuse levels；
- reduction thread extents；
- unroll候选；
- thread binding候选。

fileciteturn563file0L1-L2

所以：

\[
\boxed{
search\ space
\neq
all\ legal\ GPU\ layouts
}
\]

---

## 6.2 但最终ranking有真实measurement

MetaSchedule走：

\[
SpaceGenerator
\rightarrow
EvolutionarySearch
\rightarrow
XGB
\rightarrow
Builder/Runner
\rightarrow
measured\ runtime.
\]

fileciteturn598file0L1-L10

因此：

- 对“ranking proxy是否充分”而言，TVM是一种反方；
- 对“candidate space是否充分”而言，TVM反而是重要支持证据。

这两个不能混在一个RQ里。

---

## 6.3 RewriteLayout新发现的强假设

`RewriteLayout`源码有一句以前没有纳入证据链的重要说明：

> layout-free buffer预期只被一个 `BufferLoad` 读取。

随后它：

1. 找到这个consumer；
2. 根据consumer indices + surrounding loops调用`SuggestIndexMap`；
3. 用这个IndexMap改buffer layout；
4. 如果中间存在cache-read chain，把同一transform沿chain向前传播。fileciteturn619file0L1-L7

这将成为后面的 **multi-consumer RQ** 的一级证据：

\[
\boxed{
\text{single-consumer-derived representation}
}
\]

遇到Attention softmax、fused residual、shared intermediates等多consumer数据流时，不一定成立。

但这是 RQ3/RQ4 的材料，不应该偷塞进RQ1。

---

# 七、TIRx：现在已经精确到具体 copy variant，而不是泛称“priority dispatch”

TIRx dispatcher：

\[
(op,target)\mapsto
\{variant_i,priority_i,predicates_i\}
\]

按：

\[
(-priority,\ variant)
\]

排序。

第一个：

\[
predicates=True
\]

且实现成功的variant直接返回。fileciteturn588file0L1-L10

CUDA copy里：

### `vec_auto`

\[
priority=10.
\]

fileciteturn625file0L1-L7

global↔shared predicate检查：

- CUDA；
- exec scope；
- all threads active；
- valid copy；
- legal storage pair；
- region element count能整除thread count。

之后根据execution scope合成thread partition，并尽量vectorize。fileciteturn621file0L1-L7

### fallback

\[
priority=0
\]

而且源码直接警告：

> scalar single-thread fallback；all faster variants rejected. fileciteturn627file0L1-L7

于是决策机制可以准确写成：

\[
\boxed{
\text{legality predicates}
+
\text{expert static priority}
+
\text{first-success}
}
\]

它的repair非常强——总能退到合法实现。

但repair解决的是：

\[
\text{can run?}
\]

不直接解决：

\[
\text{fastest among legal variants?}
\]

---

# 八、Hexcute：这次补上了“约束解完以后仍有heuristic”

Hexcute首先不是heuristic搜索，而是constraint inference。

Copy：

\[
f\circ p^{-1}=g\circ q^{-1}.
\]

MMA：

\[
f_{1m}=f_{3m},\qquad
f_{2n}=f_{3n},\qquad
f_{1k}=f_{2k}.
\]

冲突则backtrack。fileciteturn633file0L1-L2

但是源码还明确写：

> 所有memory constraints满足以后，**remaining undetermined strides由heuristic决定**。fileciteturn633file1

而copy存在多个instruction candidates时，源码描述了：

- DFS找全部；
- cost model选最好；
- 或简单heuristic/beam search剪枝。

fileciteturn633file0

然后独立的bank conflict pass对同一shared tensor的所有copy consumers一起枚举swizzle并最小化总conflict；WGMMA operand会约束shared layout不能任意变化。fileciteturn596file0L1-L2

cost model则明确假设：

\[
cost=N_\text{inst}\cdot latency
\]

并进一步：

- copy/MMA可以充分overlap；
- pipeline充分overlap；
- 不计`mbarrier` / `cp_async_wait_group`；
- address manipulation可忽略；
- bank conflict由另一pass处理。

fileciteturn581file0L1-L2

所以Hexcute的科学张力非常明确：

\[
\boxed{
\text{强结构约束能否把空间缩到足够规则，
使简单剩余ranking proxy已经充分？}
}
\]

这比“cost model漏了bank conflict”强得多。

---

# 九、TileLang：这是目前 RQ1 最直接的“框架自己做A/B”的证据

Layout inference对connected component枚举complete layout assignments，然后用可插拔cost model挑winner。fileciteturn664file0L1-L7

现在有两种：

### 默认 `register-count`

主要排序：

\[
C_\text{reg}(L)
=
localSpillBytes(L),\quad
RegisterSlots(L).
\]

也就是说在没有spill时基本就是：

\[
\min RegisterSlots.
\]

### `io-aware`

对global-memory touching statements：

\[
C_s
=
\max(
BWBytes_s,
IssueBytes_s
)
\]

然后：

\[
C(L)=\sum_s C_s
\]

最后register数只做tie-break。fileciteturn664file0L1-L7

更底层的实现会计算：

- vector width；
- warp coalescing segments；
- issue depth；
- threads；
- memory-system geometry。

无法表达的layout被保守赋予worst-case。fileciteturn582file0L1-L2

---

## 已经存在一个源码记录的proxy failure

`broadcast_read`：

```text
small fragment
      ↓
register-count
      ↓
few register slots
      ↓
partial replication
      ↓
parallel j collapsed
      ↓
every thread serially traverses j
```

而io-aware：

```text
recognizes issue cost
      ↓
fully replicate tiny fragment
      ↓
non-replicated/coalesced output loop
```

源码直接把前者称为 `#1729 pathology`。fileciteturn666file0L1-L7

这不是我们推测“register proxy可能不好”，而是框架的维护验证case已经这么定义。

但它同时给出了反证：

softmax-shaped row reduction + broadcast的case，两套模型可以得到相同布局。fileciteturn667file0L1-L7

因此科学问题不能写成：

> register-count总是错。

而应该问：

> **什么时候它足够，什么时候不够？**

---

# 十、FlashInfer：两条极强的proxy压缩规则

## 10.1 Q-tile selection

planner先求：

\[
\overline Q
=
\frac{
\sum_i packedQ_i
}{B}
\]

再调用：

\[
CTA_Q
=
f(
\overline Q,
D_{VO},
D_{QK},
KVBytes
).
\]

fileciteturn635file0L1-L24

当前明确：

\[
\overline Q\le32
\Rightarrow
CTA_Q=16.
\]

源码注释就是：

> decode / short-q / speculative decode: lean CTA16. fileciteturn638file0L1-L34

large head-dim又限制candidate family：

\[
D_{VO}\ge512:
\{16,32\}
\]

\[
D_{QK}\ge512,D_{VO}<512:
\{16\}
\]

否则：

\[
\{16,64,128\}.
\]

fileciteturn639file0L1-L26 fileciteturn639file1L27-L63

真正值得质疑的不是“32这个threshold对不对”，而是：

\[
\boxed{
\overline Q
}
\]

是否足以代表ragged batch。

因为：

\[
[32,32,32,32]
\]

和：

\[
[1,1,1,125]
\]

有相同平均值，却有非常不同的：

- tail waste；
- CTA imbalance；
- short/long request mix；
- per-request active tiles。

---

## 10.2 Split-KV mode ordering

这个证据更干净。

函数明明拿到了：

- `tile_size_q`
- `head_dim`
- `head_dim_per_cta_v`
- `split_kv`

但当前ordering刻意不使用shape参数。

FMHA：

\[
cluster
\succ
gmem\_separate
\succ
gmem\_inline.
\]

MLA 1CTA：

\[
cluster
\succ
gmem\_separate.
\]

源码明确写：

> 保持selection structural，避免shape-specific measured crossover tables。fileciteturn657file0L1-L7

这已经不是我们“猜它可能忽略shape”。

这是作者明确选择：

\[
\boxed{
structural\ rule
\quad\text{instead of}\quad
shape\text{-}specific\ measured\ crossover
}
\]

一个非常典型、非常值得研究的trade-off。

---

# 十一、证据链现在能凝练出哪些问题？

经过这轮，RQ排序应该再次微调。

我现在认为证据最强的结构是：

| RQ | 科研问题 | 直接支持框架 | 当前证据强度 |
|---|---|---:|---|
| **RQ1** | **低维proxy/静态context何时足以决定near-optimal layout/backend/mapping？** | SGLang、vLLM、CUTLASS、Triton、Gluon、TIRx、Hexcute、TileLang、FlashInfer；TVM作反方 | **极强** |
| RQ2 | 决策被拆到runtime/layout/kernel/pipeline后，是否近似可分？ | 约8个 | 强 |
| RQ3 | rule/constraint/template生成的candidate space是否保留near-optimal configs？ | CUTLASS、Triton、TVM、TIRx、Hexcute、TileLang | 极强 |
| RQ4 | 一个representation面对多个consumer时，本地cost能否组合成global最优？ | Triton、TVM、Hexcute、TileLang、vLLM | 强 |
| RQ5 | workload空间能否用少量static regions覆盖，还是需要动态policy？ | SGLang、vLLM、FlashInfer | 中强 |
| **RQ6（新增）** | **何时应该相信cheap heuristic，何时应该升级到更昂贵的cost model/search/measurement？** | 几乎所有框架形成不同答案 | **很强的新问题** |

RQ6是这轮真正新形成的问题。

它来自一个以前没有单独强调的共同现象：

```text
SGLang       feature tree → model override
vLLM        static priority → legality fallback
CUTLASS      builder rule → nvMMH → profiler
Triton       local heuristic → autotune only some exposed dimensions
Gluon        distance inference → explicit user layout
TVM          schedule prior → model → real measurement
TIRx         priority → next legal variant
Hexcute      constraints → analytic model
TileLang     register-count → optional io-aware
FlashInfer   structural policy → support/occupancy fallback
```

这些实际上是在回答同一个未解决trade-off：

\[
\boxed{
\text{Decision quality}
\quad\leftrightarrow\quad
\text{selection overhead}
}
\]

而各框架没有统一答案。

但按你的要求，现在不审判RQ6，更不进入RQ2。

下面只彻底审判 RQ1。

---

# 十二、RQ1重新精确定义

之前的名字“Proxy Sufficiency”还稍微抽象。

我建议正式定义为：

\[
\boxed{
\textbf{RQ1: Decision-Sufficient State for GPU Layout Optimization}
}
\]

中文：

> **对现代 Attention/GQA/MLA/MoE 的layout、mapping、backend与representation决策，当前框架使用的低维proxy是否构成近似充分的决策状态？若不是，哪些缺失状态是造成系统性selection regret的最小必要信息？**

注意，它不是：

> heuristic准不准。

而是一个更深的问题。

设真实执行状态：

\[
x=
(
shape,
raggedness,
dtype,
consumer,
cache,
reuse,
resource,
pipeline,
concurrency,
routing,
hardware,\ldots
).
\]

candidate：

\[
c\in\mathcal C(x).
\]

真正oracle：

\[
c^*(x)
=
\arg\min_c T(c,x).
\]

框架实际只观察：

\[
p=P(x).
\]

然后：

\[
\hat c=\pi(P(x)).
\]

核心科学问题就是：

\[
\boxed{
P(x)
\text{是否是 }c^*(x)\text{ 的近似充分统计量？}
}
\]

---

# 十三、什么叫“proxy不充分”：必须用collision来证明

不能因为规则“看起来简单”就批判。

最严格的证据应该是：

存在：

\[
x_1,x_2
\]

使：

\[
P(x_1)=P(x_2),
\]

但：

\[
c^*(x_1)\neq c^*(x_2).
\]

这就是：

\[
\boxed{\text{Proxy Collision}}
\]

定义：

\[
R_P(x)
=
\frac{
T(\pi(P(x)),x)
}{
\min_cT(c,x)
}
-1.
\]

如果两个状态proxy完全一样，但最优candidate翻转，那么：

> 只调整同一个proxy上的threshold是不可能同时解决两者的。

这是比“heuristic threshold不准”强得多的科学结论。

---

# 十四、RQ1完整“问题审判”：框架逐一列罪证与反证

这里给一张核心总表。

| 框架 | selector / rule | 实际proxy | 该规则未读取 | fallback / repair | 现代模型触发 | 证据性质 |
|---|---|---|---|---|---|---|
| SGLang | generic attention backend | arch/spec/asymKV/sink/head-count | live B、Q/KV distribution、cache/load | FI→Triton + model overrides | MHA/GQA/MLA/spec | 间接强证据 |
| SGLang | DeepSeek/Nemotron/MiniMax overrides | model+quant+arch+feature | routing/load distribution | 更细special case | MLA/DSA/MoE/sparse | “repair accumulation” |
| vLLM | MLA backend priority | arch/head/KV dtype/static flags | **live batch size** | 下一个合法backend | DeepSeek MLA | **直接已知crossover** |
| vLLM | KV physical layout vote | support sets + first-choice votes | runtime prefill/decode/transfer/reuse | intersection/connector fallback | GQA/MLA serving | 强结构证据 |
| vLLM | block/page preference | backend supported sizes | live fragmentation/reuse/workload | user override | paged attention | 结构证据 |
| vLLM | content packing | backend/quant mode | other consumer total cost | composite compatibility | quantized KV/MLA | representation evidence |
| CUTLASS | StageCountAuto | SMEM footprint | latency/occupancy tradeoff beyond assumed 1 CTA/SM | manual stage/profile | GEMM/MoE | **显式假设** |
| CUTLASS | SMEM selector | divisibility/dtype/major | measured bank/pipeline interaction | smaller swizzle/manual | attention GEMM/MoE | structural heuristic |
| CUTLASS | Raster order | tiles M/N | K/cache/epilogue/cluster state | user override | GEMM/MoE | clean proxy |
| Triton | Coalesce/RLC | memory-vs-MMA anchor class | whole subgraph latency | convert/remat | Attention QK/softmax/PV | strong |
| Triton | Pipeliner | IR sync-dot pattern | full resource/runtime behavior | skip pipeline | attention/GEMM | **源码称dirty heuristic** |
| Triton | Gather | shape/layout + approx local costs | whole-kernel effect | alternate lowering | sparse/gather attention | **源码要求更精确heuristic** |
| Triton | register/TMEM | register fraction | interaction with live ranges/pipeline | conservative mapping | Blackwell kernels | direct heuristic |
| Gluon | AutoEncoding conflict | inference distance | performance entirely | explicit layout/error | fused Attention/reshape/GQA | clean nonperformance proxy |
| TVM | ScheduleRules | fixed schedule grammar | configs outside grammar | search+measurement | GEMM/attention/MoE | ranking反例；coverage证据 |
| TIRx | dispatch | predicates + static priority | measured latency | next legal variant | copy/MMA kernels | clean static ranking |
| Hexcute | constraint inference | structural legality | performance among legal configs | backtracking | Attention/GEMM | legality机制 |
| Hexcute | analytic cost | inst count/latency/ideal overlap | barriers/contention/etc | alternate candidate | async Hopper | explicit model assumption |
| TileLang | register-count | spill + register slots | global coalescing/issue | io-aware opt-in | reduction/broadcast | **已有已知pathology** |
| TileLang | io-aware | global traffic + issue | much of SMEM/async compute interactions | conservative worst case | softmax-like patterns | strongerproxy |
| FlashInfer | CTA-Q | mean packed Q + head dims | variance/tail distribution | legal CTA family | ragged prefill/spec | clean compression |
| FlashInfer | split-KV | family/topology/support | **shape values intentionally excluded** | structural fallback | decode GQA/MLA | **明确设计取舍** |

这已经把RQ1从“九个框架好像都有heuristic”变成了五种不同的proxy范式：

\[
\boxed{
\begin{array}{ll}
\textbf{A. Structural proxy} &
\text{CUTLASS, FlashInfer, TIRx}\\
\textbf{B. Feature-tree proxy} &
\text{SGLang, vLLM}\\
\textbf{C. Graph-distance/locality proxy} &
\text{Triton, Gluon}\\
\textbf{D. Analytic cost proxy} &
\text{Hexcute, TileLang}\\
\textbf{E. Empirical correction} &
\text{TVM, CUTLASS profiling}
\end{array}
}
\]

也就是说RQ1不是针对某一个框架的“bug”。

它是在问：

> 哪一种信息抽象才足以表示GPU layout decision？

---

# 十五、一级证据：不是“怀疑”，而是源码已经暴露不足

证据强度需要分类，否则容易把推测和事实混起来。

## A级：已有直接不足/crossover证据

### A1. vLLM MLA backend

代码自己说：

\[
bs\le2:
Tokenspeed\ regresses
\]

而：

\[
bs\gtrsim8:
Tokenspeed\ wins.
\]

但selection priority不读batch。fileciteturn650file0L1-L7

这是：

\[
\boxed{
known\ relevant\ variable
\notin
selector\ state
}
\]

目前最强。

---

### A2. TileLang register-count

源码维护case已经说明：

\[
\min registers
\]

导致parallel execution collapse pathology。fileciteturn666file0L1-L7

这是：

\[
\boxed{
proxy\ locally\ improves
\quad
but
\quad
global\ mapping\ gets\ worse
}
\]

也是直接证据。

---

## B级：源码作者自己明确称heuristic/不精确

包括：

- Triton `dirty heuristic` pipeline；
- Gather需要more precise heuristic；
- kWidth未来应做layout backprop；
- CUTLASS 1CTA/SM stage assumption；
- FlashInfer明确选择不使用shape crossover table。

fileciteturn656file1L28-L51 fileciteturn572file2L35-L46 fileciteturn655file0L1-L25 fileciteturn653file0L1-L32 fileciteturn657file0L1-L7

这类证据证明“trade-off存在”，但还不能证明性能regret大。

---

## C级：结构上使用非performance proxy

例如：

- Gluon inference distance；
- TIRx priority；
- vLLM KV-layout vote；
- CUTLASS divisibility；
- SGLang architecture tree。

它们说明：

\[
selection
\ne
direct\ performance\ optimization
\]

但不代表selection就一定差。

这类必须通过实验寻找collision。

---

# 十六、现代 Attention/GQA/MLA/MoE 为什么尤其容易触发 RQ1

不是因为“大模型很复杂”这种泛泛理由，而是这些workload恰好产生selector当前proxy容易压缩掉的变量。

## 16.1 Ragged Attention

真实batch：

\[
\{Q_i,K_i\}_{i=1}^B
\]

不是一个单一Q、K。

而FlashInfer CTA rule会压缩成：

\[
\bar Q.
\]

于是：

\[
Var(Q),\ Q_\max,\ tail
\]

全部消失。

---

## 16.2 GQA

GQA引入：

\[
r=\frac{H_Q}{H_{KV}}.
\]

它同时改变：

- KV reuse；
- memory traffic；
- Q-head parallelism；
- KV cache traversal；
- split-KV granularity。

但很多backend/layout selector主要按：

- backend family；
- head size；
- KV-head count；
- page legality

分类。

因此：

\[
(H_Q,H_{KV},B,S_{KV})
\]

之间的interaction可能不能由单个head-count或backend identity表示。

---

## 16.3 MLA

MLA更明显：

\[
QK:
(D_{nope}+D_{rope})
\]

而：

\[
V:
D_v.
\]

再叠加：

- FP8 KV；
- latent cache；
- speculative q_len；
- split-KV；
- DCP。

vLLM当前恰好已经出现了真实batch crossover注释，因此MLA应该成为RQ1第一优先实验域，而不是一般MHA。fileciteturn650file0L1-L7

---

## 16.4 MoE

MoE真实shape不是：

\[
M=N_\text{tokens}.
\]

而是：

\[
M_e=N_{\text{tokens assigned to expert }e}.
\]

因此同样：

\[
\sum_e M_e=4096
\]

可以是：

\[
[512,512,\ldots]
\]

也可以是：

\[
[3000,500,250,100,\ldots].
\]

如果MoE runner只按：

- quantization；
- architecture；
- model identity

选择：

\[
DeepGEMM/Marlin/FI\text{-}CUTLASS/Triton,
\]

就没有显式观察routing histogram。

SGLang Nemotron与MiniMax-M3正是这种规则。fileciteturn567file0L1-L13 fileciteturn612file0L1-L7

CUTLASS `StageCountAuto`又明确基于静态tile/SMEM而不是per-expert occupancy distribution。

因此MoE提供了与Attention完全不同的RQ1验证方向。

---

# 十七、RQ1不能只证明“存在bad case”，必须拆成四个可证伪假设

我现在把RQ1拆成四层。

## H1 — Proxy Collision Hypothesis

\[
\boxed{
\exists x_1,x_2:
P(x_1)=P(x_2)
\land
c^*(x_1)\ne c^*(x_2)
}
\]

也就是说，相同当前proxy不足以决定最佳configuration。

### 否证条件

如果在大量realistic workload里：

\[
P(x_1)=P(x_2)
\Rightarrow
c^*(x_1)\approx c^*(x_2),
\]

那么H1失败。

---

# 十八、H2 — Systematic Regret Hypothesis

一个crossover还不够。

要求：

\[
\Pr_{x\sim\mathcal W}
[
R_P(x)>\epsilon
]
>\delta.
\]

例如可以后续用：

\[
\epsilon=5\%
\]

作为near-optimal阈值，但它只是实验定义，不是理论常数。

核心是：

\[
\delta
\]

不能接近0。

也就是说这些collision必须落在现实Attention/GQA/MLA/MoE的常见区域，而不是人为构造corner case。

---

# 十九、H3 — Compact Augmentation Hypothesis

假设当前：

\[
P(x)
\]

不充分。

不意味着必须训练一个巨大ML模型。

可能只需加入几个变量：

\[
P'(x)
=
[P(x),z_1,\ldots,z_k].
\]

研究：

\[
k\ll dim(x)
\]

能否让：

\[
R_{P'}\ll R_P.
\]

候选missing state目前已经能从源码反推出来：

### Attention

\[
\{
Var(Q),
Q_{max},
KV_{mean},
KV_{tail},
B,
GQA\ ratio
\}
\]

### MLA

\[
\{
B,
q\_len,
split\_kv,
D_{latent},
D_{rope},
D_v
\}
\]

### MoE

\[
\{
\max_eM_e,
Var(M_e),
active\ experts,
routing\ skew
\}
\]

### kernel level

\[
\{
regs,
SMEM,
CTA/SM,
pipeline\ overlap,
conversion\ cost
\}.
\]

如果很少的augmentation就能消除大部分regret，这是非常好的科研结果。

---

# 二十、H4 — Performance-Uncertainty Hypothesis

这是本轮新证据引出的一个关键子问题，也是未来RQ6的来源。

不应该永远使用昂贵搜索。

也不应该永远相信cheap heuristic。

设selector除了candidate外还能估计confidence：

\[
u(x)=
\text{uncertainty of cheap decision}.
\]

策略：

\[
u(x)<\tau
\Rightarrow
\text{accept static decision}
\]

否则：

\[
\text{escalate to analytic/search/measurement}.
\]

研究假设：

> **存在一个低成本的不确定性指标，可以识别大部分高proxy-regret区域，从而只在必要时付出搜索/measurement成本。**

这正对应框架现状两端：

```text
TIRx / FlashInfer / vLLM
cheap static decision
        ↑
        |
      unknown optimal escalation point
        |
        ↓
TVM / profiling
measurement-heavy decision
```

这个问题不是工程补丁，而是真正的：

\[
compile/runtime\ overhead
\leftrightarrow
selection\ regret
\]

trade-off。

---

# 二十一、RQ1必须做的“杀手级”实验，不写代码，只确定证伪结构

## Case 1 — vLLM MLA：目前证据最强

固定：

- DeepSeek R1 dimensions；
- SM100；
- FP8 KV；
- head size；
- num heads；
- page size；
- speculative mode；
- selector全部输入。

只改变：

\[
B.
\]

例如：

\[
B=1,2,4,8,16,32.
\]

比较至少：

\[
FlashInferMLA
\quad vs\quad
TokenSpeedMLA.
\]

源码已经预测：

\[
B\le2
\]

和：

\[
B\gtrsim8
\]

行为不同。fileciteturn650file0L1-L7

如果实际测量确认winner翻转：

\[
\boxed{
\text{H1几乎直接成立}
}
\]

因为当前priority function无法表示这个变量。

如果2026当前kernel已经变化，注释过时，且FlashInfer始终near-optimal，那么这条证据会被反驳。这也正是科学实验应该做的。

---

# 二十二、Case 2 — TileLang：已知proxy冲突是否真的成为性能regret

使用源码自己的：

\[
broadcast\_read.
\]

比较：

\[
L_\text{register-count}
\]

与：

\[
L_\text{io-aware}.
\]

源码已经预测mapping结构不同。fileciteturn666file0L1-L7

真正科学测试是：

\[
T(L_{reg})
\stackrel{?}{>}
T(L_{io}).
\]

如果没有显著performance差别，那么“pathology”只是IR结构不好看，不构成科研价值。

因此这条实验非常重要，因为它能防止我们把编译器结构问题误判为性能问题。

---

# 二十三、Case 3 — FlashInfer Mean-Q collision

保持：

\[
\bar Q=32.
\]

构造：

\[
A=[32,32,32,32]
\]

\[
B=[1,1,1,125].
\]

当前CTA selector看到同一个mean。fileciteturn635file0L1-L24

然后比较CTA：

\[
16,64,128
\]

中legal candidates。

如果：

\[
CTA^*(A)\ne CTA^*(B)
\]

且差距明显，则：

\[
\boxed{
mean\ Q
}
\]

不是充分统计量。

如果winner从不翻转，则这个proxy比直觉中更强，H1受到反证。

---

# 二十四、Case 4 — FlashInfer Split-KV structural policy

固定：

- family；
- topology；
- available modes；
- split count；
- cluster legality。

因此当前排序固定不变。

改变：

\[
head\_dim,
D_v,
KVLen,
B,
Q.
\]

测试：

\[
cluster\_smem,
gmem\_separate,
gmem\_inline.
\]

因为这些shape虽然传给policy，却被刻意从ordering中排除。fileciteturn657file0L1-L7

如果winner发生crossover：

\[
\boxed{
structural\ legality
\ne
performance\ sufficiency.
}
\]

---

# 二十五、Case 5 — CUTLASS StageCountAuto的1CTA/SM假设

当前默认明确：

\[
\text{maximize SMEM stages under 1 CTA/SM}.
\]

fileciteturn653file0L1-L32

所以应该找：

- short-K；
- small-M expert；
- grouped MoE；
- irregular expert sizes；

比较：

\[
StageCountAuto
\]

与显式较少stage，例如：

\[
StageCount<N-1>,
StageCount<N-2>.
\]

关键问题：

> 减少stage后允许更高residency时，是否反而更快？

如果始终不快，那么CUTLASS这个假设得到强支持。

如果在MoE常见small-expert区域系统翻转，就是RQ1跨Attention以外的重要证据。

---

# 二十六、Case 6 — TIRx priority collision

找到两个同时合法的variants：

\[
c_a,c_b
\]

且：

\[
priority_a>priority_b.
\]

保持所有legality predicates均true，改变shape/resource context。

若存在：

\[
T(c_a,x_1)<T(c_b,x_1)
\]

同时：

\[
T(c_a,x_2)>T(c_b,x_2),
\]

则static priority必然不能同时最优。

这条实验尤其干净，因为不需要挑战TIRx的legality inference，只挑战：

\[
\boxed{
legal\ priority
\approx
performance\ priority
}
\]

这个假设。

---

# 二十七、Case 7 — Gluon inference-distance collision

设计一个layout conflict：

```text
seed A ──reshape/join──┐
                       ├── X
seed B ─────elementwise┘
```

让：

\[
distance_A > distance_B
\]

因此Gluon必定选B-derived encoding。fileciteturn602file0L1-L7

然后人为显式固定两种legal layout分别benchmark。

若较远seed反而更快：

\[
distance
\not\Rightarrow
performance\ preference.
\]

这不会说明Gluon算法“错误”，因为distance本来可能只是inference confidence。

真正科学问题是：

> inference-confidence是否可以兼作performance-confidence？

---

# 二十八、Case 8 — Triton local-anchor conflict

构造现代Attention子图：

\[
QK^T
\rightarrow
scale
\rightarrow
softmax/reduction
\rightarrow
PV.
\]

其中：

- load希望Blocked/coalesced；
- dot希望MMA；
- reduction希望thread-local；
- PV又希望另一个dot operand layout。

当前Triton本来就在这些anchor之间传播并插conversion/rematerialization。fileciteturn671file0L1-L2

测试：

\[
\text{compiler chosen conflict solution}
\]

相对不同手工/Gluon layouts的oracle。

如果差距只出现在极少case，则local proxy可能已经很好。

如果：

- Q长度；
- head dim；
- GQA ratio；
- softmax row width

可预测地改变最佳layout conflict resolution，则RQ1得到compiler-level证据。

---

# 二十九、Case 9 — Hexcute analytic-ranking fidelity

重点不是绝对latency误差：

\[
|\hat T-T|.
\]

而是排序：

\[
T(c_i)<T(c_j)
\Rightarrow
\hat T(c_i)<\hat T(c_j).
\]

只在：

\[
\text{同一个constraint-legal candidate set}
\]

内部比较。

改变：

- stages；
- cp.async/TMA；
- WGMMA；
- barrier structure；
- SMEM swizzle；
- register pressure。

如果简单instruction-latency模型仍然保持高top-k recall，那么RQ1得到反证：

> constraints已经把空间压缩得足够结构化。

这其实是非常重要的可能结果。

---

# 三十、Case 10 — MoE routing-skew collision

保持：

\[
N_{total},
E,
dtype,
quant,
GPU
\]

完全一样。

只改变：

\[
M_e
\]

distribution。

例如：

Uniform：

\[
[512,512,512,\ldots]
\]

Skewed：

\[
[3000,400,200,100,\ldots].
\]

比较：

- DeepGEMM；
- Marlin；
- FlashInfer-CUTLASS/TRTLLM routed；
- Triton；

以及可能不同tile/layout。

如果：

\[
backend^*_\text{uniform}
\ne
backend^*_\text{skewed},
\]

而当前SGLang选择规则只看model/quant/arch，则这是Attention之外非常强的RQ1证据。

---

# 三十一、TVM在RQ1实验里应该扮演什么角色？

不能把TVM也当“被告”。

它更适合作为：

\[
\boxed{\text{measurement control}}
\]

因为它已经采取：

\[
candidate\ grammar
+
learned\ model
+
hardware\ measurement.
\]

fileciteturn598file0L1-L10

如果：

- 静态proxy方案频繁失败；
- MetaSchedule measurement能找到显著更优candidate；

支持RQ1。

反过来如果：

- measurement最终总是找到与cheap heuristic相同/极近的solution；

那么证明低维proxy已经接近sufficient。

但要注意，这只能比较：

\[
G_{\text{TVM}}
\]

内部的candidate。

如果好layout根本没被ScheduleRules生成，那是RQ3，不属于RQ1。

这个边界必须严格保持。

---

# 三十二、RQ1的框架repair是否已经把问题解决？

这里是最容易误判的地方。

把repair按类型分类：

\[
\boxed{
\begin{array}{ll}
\text{Legality repair} &
vLLM,\ TIRx,\ Hexcute,\ FlashInfer\\
\text{Structural repair} &
Triton,\ TileLang,\ Gluon\\
\text{Special-case repair} &
SGLang\\
\text{Analytic repair} &
CUTLASS,\ Hexcute,\ TileLang\\
\text{Empirical repair} &
TVM,\ CUTLASS\ profiler
\end{array}
}
\]

问题是，很多fallback只在：

\[
candidate\ invalid
\]

时触发。

而RQ1关心：

\[
candidate\ valid\ but\ suboptimal.
\]

例如：

### vLLM

FlashInfer MLA和TokenSpeed都合法。

static priority不会因为：

\[
B=16
\]

而fallback。

### TIRx

priority=10 variant合法，就直接返回。

不会问priority=0或另一个variant是不是更快。

### FlashInfer

cluster mode通过occupancy/support，就优先。

不会因为某head_dim上GMEM reducer快而自动触发fallback。

所以：

\[
\boxed{
legality\ fallback
\ne
performance\ uncertainty\ repair.
}
\]

这是RQ1能够存活的关键原因。

---

# 三十三、RQ1的真正反方证据

如果只收集支持证据，就还是不够科学。

目前最重要的反方至少有四条。

### 反方1：TVM

measurement可以纠正ranking proxy。

### 反方2：CUTLASS

高性能GPU kernel空间可能具有非常强的结构规律：

\[
divisibility,
instruction\ shape,
SMEM,
alignment
\]

本身就可能把winner几乎确定下来。

### 反方3：Hexcute

constraint inference可能大幅减少candidate entropy。

如果：

\[
|\mathcal C_\text{legal}|
\gg1
\]

但near-optimal区域高度集中，那么简单cost model也完全可能足够。

### 反方4：TileLang softmax-like case

源码自己的典型reduce+broadcast case里，register-count与io-aware可以一致。fileciteturn667file0L1-L7

说明：

\[
\text{简单proxy}
\]

并不是天然错误。

所以RQ1不能预设：

\[
H_1=true.
\]

---

# 三十四、什么结果会直接判 RQ1 “无罪”？

这一步必须现在写清楚。

如果后续真实实验得到：

### vLLM MLA

虽然源码有bs crossover注释，但2026当前kernel上：

\[
\max_B R_P(B)<3\%-5\%
\]

或者winner实际上不再翻转。

### FlashInfer mean-Q

极端改变raggedness：

\[
Var(Q)
\]

也不改变optimal CTA tile，或当前选择始终near-optimal。

### FlashInfer Split-KV

只要structural cluster条件成立：

\[
cluster
\]

几乎始终是winner。

### CUTLASS

`StageCountAuto`即使在short-K/MoE-small-expert也始终near-optimal。

### TIRx

legal variants的performance ordering基本固定。

### Hexcute

简单latency ranking长期保持：

- 高Spearman；
- 高top-k recall；
- 高oracle-near-optimal hit rate。

### TileLang

documented layout pathology对真实latency没有明显伤害。

那么：

\[
\boxed{
RQ1应当被削弱甚至否决
}
\]

科学结论反而会是：

> GPU高性能layout/config空间存在很强的低维结构，当前结构proxy已经接近decision-sufficient。

这是完全有可能的结果。

---

# 三十五、什么结果才允许我们说 RQ1 被证实？

不能只找到一个case快20%。

我建议必须同时满足四条。

### 1. Collision

\[
P(x_1)=P(x_2)
\]

而：

\[
c^*(x_1)\ne c^*(x_2).
\]

### 2. Material regret

\[
R_P(x)>\epsilon
\]

不是噪声。

### 3. Common workload

发生在：

- GQA decode；
- MLA decode；
- ragged prefill；
- speculative verify；
- MoE skew；

这样的真实常见区域。

### 4. Explanation

找到missing variable：

\[
z
\]

加入以后：

\[
P'=[P,z]
\]

显著消除regret。

最强成果链是：

```text
current proxy
      ↓
repeated collision
      ↓
identify missing state
      ↓
compact augmentation
      ↓
most regret disappears
```

而不是：

```text
发现heuristic慢
      ↓
加一个if
```

后者只是工程修补。

---

# 三十六、根据证据，现在RQ1最值得优先验证的顺序已经很清楚

如果按“证据强度 × 可证伪性 × 现代LLM相关性”排：

\[
\boxed{
\textbf{1. vLLM MLA batch crossover}
}
\]

源码已经给出crossover方向，selector又没有该变量。

然后：

\[
\boxed{
\textbf{2. TileLang register-count vs io-aware known pathology}
}
\]

因为框架自己已经准备好A/B selection体系。

然后：

\[
\boxed{
\textbf{3. FlashInfer split-KV structural-vs-shape policy}
}
\]

源码明确解释了为什么不采用measured crossover table。

第四：

\[
\boxed{
\textbf{4. FlashInfer mean-Q ragged compression}
}
\]

数学上最干净。

第五：

\[
\boxed{
\textbf{5. CUTLASS StageCountAuto 1CTA/SM assumption under MoE}
}
\]

很适合把RQ从attention推广到MoE。

然后才是：

- TIRx priority；
- Triton local-layout conflicts；
- Gluon distance；
- Hexcute ranking。

---

# 三十七、所以现在对 RQ1 的“审判结论”是什么？

现在已经有足够源码证据判定：

\[
\boxed{
\textbf{RQ1是成立为“值得研究的问题”的，}
}
\]

但还不能判定：

\[
H_1\textbf{已经成立}.
\]

二者必须区分。

我们现在已经证明的是：

### 事实1

几乎所有系统都必须把完整性能状态：

\[
x
\]

压缩为某种：

\[
P(x).
\]

### 事实2

不同系统选择了完全不同的proxy：

\[
\begin{aligned}
SGLang &: model/hardware/features\\
vLLM &: static priorities/preferences/capabilities\\
CUTLASS &: footprint/divisibility/grid geometry\\
Triton &: local memory/layout/resource heuristics\\
Gluon &: inference distance\\
TVM &: schedule priors + measurement\\
TIRx &: predicates + priorities\\
Hexcute &: constraints + analytic latency\\
TileLang &: registers\ or\ IO\ traffic\\
FlashInfer &: meanQ/structural topology
\end{aligned}
\]

### 事实3

至少两个系统已经存在**非常直接的不足证据**：

- vLLM：known batch crossover未进入selector state；
- TileLang：默认proxy已有维护者记录的layout pathology。

### 事实4

另外几个框架源码自己承认其规则是heuristic、approximation或显式设计trade-off。

因此真正未回答的问题是：

\[
\boxed{
\begin{array}{c}
\text{哪些状态变量真正改变optimal layout decision？}\\[2mm]
\text{现有proxy在哪些区域已经足够？}\\[2mm]
\text{在哪些区域会发生systematic collision？}\\[2mm]
\text{最小需要增加多少context才能修复？}
\end{array}
}
\]

这就是目前证据最强的科研问题。

---

# 三十八、RQ1现在可以正式冻结为这个版本

我建议后续不再随便改写RQ1的定义：

\[
\boxed{
\begin{aligned}
\textbf{RQ1: }&
\text{Are the low-dimensional proxies used by current GPU layout/configuration}\\
&\text{selectors decision-sufficient for modern Attention, GQA, MLA, and MoE workloads?}\\
&\text{If not, what is the minimal additional state required to eliminate}\\
&\text{systematic selection regret without resorting to exhaustive measurement?}
\end{aligned}
}
\]

中文：

> **当前GPU layout/configuration selector使用的低维proxy，对于现代Attention、GQA、MLA与MoE是否是近似决策充分的？若不是，最少还需要哪些状态信息，才能在不进行穷尽实测的情况下消除系统性的selection regret？**

它有真正的trade-off：

\[
\boxed{
state/model\ complexity
+
selection\ overhead
\quad\leftrightarrow\quad
decision\ quality
}
\]

而不是一个确定答案。

---

# 三十九、证据链完整度现在应该怎样评价

这次我不会说“所有仓库所有规则100%穷尽”，那不严谨。

现在能比较有把握地说的是：

### 对 RQ1 所需的 mechanism-level 证据

已经达到闭环标准：

\[
\boxed{
rule
\to proxy
\to assumption
\to omitted\ state
\to fallback
\to modern\ trigger
\to falsification
}
\]

10个系统都已经有位置。

### 仍然没有逐行穷尽的

主要是：

- CUTLASS每个SM90/SM100/SM120 builder specialization；
- SGLang数十个model-family override leaf；
- vLLM每一个独立backend的所有packing/page-size特例；
- FlashInfer每个FA2/FA3/Blackwell kernel family threshold；
- Triton所有architecture-specific pass细节。

但是目前新增这些leaf反复落入已经识别出的机制：

```text
static structural rule
preference/priority
local heuristic
constraint
analytic cost
measurement
repair/fallback
```

还没有发现会改变 RQ1 定义的新selection mechanism。

因此，**RQ1现在可以认为完成了“问题审判”的源码阶段**。

我不在这一轮开始审判 RQ2，否则又会把深度摊薄。下一轮应该严格用同样标准审判 RQ2——不是泛泛说“框架有不同边界”，而是逐个追到：**谁先决定什么 → 下游决策能否反向影响上游 → 源码是否允许joint revision → framework自己的repair → Attention/GQA/MLA/MoE中哪些变量跨边界耦合 → 如何构造完全相同局部状态但不同global optimum的反例 → 什么结果会推翻Boundary Regret这一问题。**

## 用户

证据链不完整请继续补充。然后把证据链也凝炼为问题。然后从RQ1开始做一条完整的“问题审判”：所有相关框架源码规则 → 每个规则使用的proxy → 它忽略了哪些变量 → 框架自己的fallback/repair → 现代Attention/GQA/MLA/MoE触发条件 → 可证伪的科研假设。 RQ1彻底完成之后再审判RQ2，然后后续RQ。这样可以防止又回到前几轮“问题列了很多，但证据深度不够”的状态 。当然如果新的证据链形成新的RQ 也请补充。

## 模型回复

我会继续把 RQ1 的证据链补到“可审判”标准，并且这次只做 RQ1，不提前进入 RQ2。重点补三类缺口：**当前规则的精确触发条件、proxy 没有读取的状态变量、以及框架自己的 repair/fallback 是否真的能覆盖“合法但次优”这一类失配**。如果补证过程中出现一个与 RQ1 正交、且跨框架重复出现的新机制，我会记录成候选新 RQ，但不展开审判。

## 模型工具调用参数

{"paths":["GitHub"],"query":"search"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 SGLang、vLLM 与 FlashInfer 最新性能回归

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取最新 CUDA 平台实现代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 SGLang 当前后端自动选择逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取模型覆盖参数代码片段

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 FlashInfer 当前实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索形状特定的测量交叉表

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查找 FlashInfer 当前平均长度实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 FlashInfer 最新 utils.cuh 内容

## 模型工具调用参数

{"uri":"/response/turn682","query":"FA2DetermineCtaTileQ"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 FlashInfer 中相关线程块优化代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 FlashInfer GQA 调度与 Split-KV 占用 evidence

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查找 FlashInfer 注意力调度器实现

## 模型思考摘要

检索并分析 SGLang、vLLM 与 FlashInfer 性能及调度实现

## 模型回复

补证过程中有一个重要修正：**FlashInfer 的 decode split-KV“要不要切分、切多细”并不只是 occupancy proxy。** 当前 scheduler 会读取每个请求的 page 数数组，并通过二分求 chunk size；GQA 还直接进入 `GROUP_SIZE`、`num_kv_heads` 和 grid-y 计算。也就是说，之前把它概括成“只看grid saturation、忽略KV-length distribution”是不准确的。真正适合 RQ1 审判的，是它后面的 **reduction-mode ordering**（cluster / separate / inline）明确选择 structural policy，而不是 measured shape crossover。

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索CUTLASS当前共享内存选择逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 CUTLASS 中的 SM100 共享内存选择器

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 sm100_common 函数内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 CUTLASS 内核自动调度与阶段计数

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 SM100 自动调度相关代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索Triton布局与流水线优化实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 中 NVMMASharedLayout 默认布局实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 Gluon 布局定义文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索并补全 `get_default_for` 阈值逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 TVM 调度成本与张量核心配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 TVM 调度规则实现文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 TVM 调优测量实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 TIRx 最新调度器优先级实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 TVM 调度器名称

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

完善 TIRx 调度器证据链

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 TIRx MMA 变体与优先级

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Hidet 仓库中的代价模型与布局推导

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Hidet 中的布局映射与代价模型文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 TileLang 复制布局冲突处理实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 LayoutInference 中的线程常量要求

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

Resolving vLLM KV 缓存布局一致性

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 vLLM 布局投票与默认偏好

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查找 SGLang 页面大小默认值与模型覆盖

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 SGLang KV 内存池布局

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索大写标志及KV缓存布局

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 当前“脏启发式”流水线证据

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 中跳过流水线启发式注释

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 通用共享回退实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 中二维张量优化相关代码

## 模型思考摘要

检索并分析多框架GPU布局调度与KV缓存实现

## 模型思考：工具活动

正在搜索 "Hexcute" GPU layout compiler GitHub

## 模型思考：工具活动

正在搜索 5 个网站

## 模型工具调用参数

{"uri":"/response/turn633","query":"display_title"}

## 模型思考摘要

已搜索 5 个网站

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Hexcute/Hidet 当前成本模型源码

## 模型思考摘要

检索 Hexcute/Hidet 当前成本模型源码

## 模型思考：工具活动

正在搜索 "hexcute/hidet" "cost_model" latency bank conflict

## 模型思考摘要

正在搜索 "hexcute/hidet" "cost_model" latency bank conflict

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Hidet CUDA 变换目录

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查找成本模型与银行冲突假设

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取银行冲突解析源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检查TMA回退拷贝实现

## 模型思考摘要

筛选实验依据
筛选实验依据

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

评估并行布局与流水线代理成本

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析现代注意力内核与GQA实现

## 模型工具调用参数

{"uri":"/response/turn721","query":"bank conflict"}

## 模型工具调用参数

{"uri":"/response/turn721","query":"BMM1_DMA_Stage"}

## 模型工具调用参数

{"uri":"/response/turn721","query":"int constexpr BMM1_DMA_Stage"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索仓库默认分支内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

整理 Hexcute Benchmark 主 README

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

Analyzing TileLang Cost Model Design

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

总结 TileLang 布局推断与共享暂存

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析 SGLang 内存池实现

## 模型工具调用参数

{"uri":"/response/turn726","query":"vectorized_5d"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 vLLM NVFP4 量化实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 vLLM Triton 每头缩放实现

## 模型思考摘要

补全证据链
补全证据链

## 模型思考摘要

检索并分析 CUDA变换、成本模型、注意力与推理实现

## 模型回复

这轮补证已经改变了两处判断，我会把修正写进最终审判而不是保留旧结论：一是 FlashInfer decode split-KV 的 chunk/split 规划其实读取了逐请求 page-count 分布，不能再把它当成“只看 occupancy”的弱 proxy；二是 Triton 的 shared-memory swizzle 也比“固定规则”复杂，当前实现会显式计算 vectorization/bank-conflict，只是在若干 tie-break 和未实现的 asymmetric vectorization 上仍使用启发式。也就是说，RQ1 必须攻击**真正被压缩掉的信息**，不能攻击框架已经观察到的信息。 

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

核实 Triton 与 CUTLASS 当前实现

## 模型思考摘要

继续闭合审判
审判 RQ1 状态
完成审判闭合并检查RQ1状态

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型工具调用参数

{"uri":"/response/turn682","query":"return 128"}

## 模型工具调用参数

{"uri":"/response/turn682","query":"if (avg_packed_qo_len > 16)"}

## 模型思考摘要

核实Triton与CUTLASS实现并检索Prefill CTA逻辑

## 模型思考用时

思考了 10m 57s

## 模型回复

对。上一轮把 RQ1 的“源码阶段”判得过早了。继续往下追以后，至少有两处必须修正：FlashInfer 的 decode split-KV 切分规划其实读取了逐请求 page-count 分布，不是简单 occupancy heuristic；Triton 的 shared-memory swizzle 也显式分析 vectorization 和 bank conflict，不能再归类成“固定 swizzle 规则”。fileciteturn685file0L1-L2 fileciteturn615file0L1-L2

这两个修正很重要，因为 RQ1 不能靠“把框架说得比实际更简单”来成立。下面重新按更严格的标准审判 RQ1。本轮仍然不进入 RQ2。

---

# 1. 先把“证据链”凝炼成问题，而不是直接列 RQ

现在 10 个系统里反复出现的机制，实际可以归并成四种证据链。

第一类是**状态压缩**：

\[
x
\overset{P}{\longrightarrow}
p
\overset{\pi}{\longrightarrow}
c
\]

真实执行状态 \(x\) 很高维，但 selector 只读取低维 \(p=P(x)\)。典型是 SGLang backend tree、vLLM backend priority、FlashInfer `avg_packed_qo_len`、CUTLASS tile/cluster structure。

这里产生的问题是：

\[
\boxed{
P(x)\text{ 是否保留了决定最优 candidate 所需的信息？}
}
\]

第二类是**合法性近似性能排序**：

\[
\text{legal candidates}
\overset{\text{priority/order}}{\longrightarrow}
\text{first success}
\]

最典型是 TIRx，也部分出现在 vLLM backend priority。这里真正的问题不是“能否运行”，而是：

\[
\boxed{
c_i,c_j\text{ 都合法时，固定 priority 是否保持真实性能顺序？}
}
\]

第三类是**解析 proxy 代替 runtime**：

\[
\hat T(c,x)
\approx T(c,x)
\]

Hexcute、TileLang 属于这一类，CUTLASS builder 的一些规则也有类似性质。问题变成：

\[
\boxed{
\arg\min_c\hat T(c,x)
\stackrel{?}{\approx}
\arg\min_cT(c,x)
}
\]

第四类是**repair 只修 correctness/compatibility，不修合法但次优**：

\[
\text{bad and illegal}
\rightarrow fallback
\]

不等于：

\[
\text{legal but slow}
\rightarrow repair.
\]

这一点横跨 vLLM、TIRx、Hexcute、FlashInfer 等系统。

因此 RQ1 应该保持一个统一定义，而不是拆成很多零散“某 heuristic 好不好”的问题：

\[
\boxed{
\textbf{RQ1 — Decision-State Sufficiency}
}
\]

> 对现代 Attention/GQA/MLA/MoE，当前 GPU layout/backend/configuration selector 所使用的状态表示和性能 proxy，是否已经包含选择 near-optimal 合法配置所需的充分信息？若没有，最小需要补充哪些状态，才能显著降低 selection regret，而不退化成穷举实测？

---

# 2. RQ1 的审判标准

设完整状态为

\[
x=
(
\text{shape},
\text{raggedness},
\text{dtype},
\text{layout},
\text{reuse},
\text{cache},
\text{resource},
\text{pipeline},
\text{concurrency},
\text{routing},
\text{hardware},
\dots
).
\]

真实最优：

\[
c^*(x)=\arg\min_{c\in\mathcal C(x)}T(c,x).
\]

框架实际只观察：

\[
p=P(x)
\]

并选择：

\[
\hat c(x)=\pi(P(x)).
\]

最强的反例不是“heuristic 慢了一次”，而是找到：

\[
P(x_1)=P(x_2)
\]

但：

\[
c^*(x_1)\ne c^*(x_2).
\]

这叫 **proxy collision**。

selection regret 定义为：

\[
R_P(x)
=
\frac{T(\pi(P(x)),x)}
{\min_cT(c,x)}
-1.
\]

因此下面每个框架都要回答六件事：

\[
\boxed{
Rule
\rightarrow Proxy
\rightarrow MissingState
\rightarrow Repair
\rightarrow ModernTrigger
\rightarrow Falsifier
}
\]

---

# 3. SGLang：feature tree + model-specific repair

SGLang 当前的 `get_default_attn_backend()` 源码甚至直接把目标写成 “Auto select the fastest attention backend”。但实际算法是一棵结构规则树。MHA 下会根据 Hopper/Blackwell/HIP、spec decode、asymmetric KV、attention sink、FlashInfer availability 等决定 FA3、TRTLLM-MHA、FA4、AITER、FlashInfer 或 Triton；MLA 下则根据 Hopper/SM100/HIP，以及 HIP 上 KV-head count 是否为 16/128 选择 FA3、FlashInfer、AITER 或 Triton。fileciteturn678file0L1-L2

| Rule | Proxy / selector state | 该规则没有读取 | framework repair | 现代触发 |
|---|---|---|---|---|
| MHA backend default | architecture、spec/top-k、asymmetric-KV、sink、backend availability | 当前 batch、当前 \(Q_i/K_i\) 分布、server concurrency、当前cache状态 | FlashInfer不可用/不兼容→Triton；model overrides | MHA/GQA/spec decode |
| MLA default | GPU family、KV-head count | current batch、decode q_len、KV-length distribution | unsupported AITER head-count→Triton | MLA |
| prefill/decode split | phase identity | 同一 phase 内的动态请求分布 | prefill/decode 可以独立 backend | chunked prefill / decode |
| model override | model identity + quant + architecture等 | 未进入该override predicate的runtime workload state | 更多 model-specific override | MiniMax、DeepSeek、Nemotron、MoE |

需要特别纠正一点：SGLang 并不是“一次选一个 backend 后 prefill/decode 共用”。它有独立的 `prefill_attention_backend` 和 `decode_attention_backend`，未设置时才回落到基础 backend。fileciteturn720file0L1-L22

所以 phase：

\[
prefill\neq decode
\]

已经进入状态空间。这实际上是 RQ1 的反证：框架已经发现单一 backend state 不够，于是扩展了 selector state。

MiniMax-M3 更能体现这种“不断补状态”的机制。SM100 + FP8 e4m3 会偏向 TRTLLM-MHA，否则 FA4；page-size 又设置 128；MXFP8 MoE 偏 DeepGEMM，而 modelopt mixed 使用 FlashInfer TRTLLM routed；BF16 full weight 下还会因为已知错误把 DeepGEMM 修正回 Triton。fileciteturn612file0L1-L7

因此 SGLang 的源码证据不是“rule 一定错”，而是：

\[
\boxed{
P_0(x)
\rightarrow
P_1(x)
\rightarrow
P_2(x)
}
\]

随着新 workload/model 出现，selector 所需上下文持续扩张。

Persistent KV 也体现同样结构。目前默认 NHD，还支持 HND 和 ROCm AITER 的 `vectorized_5d`；HND 为 per-KV-head sparse page table 折叠 page/head，`vectorized_5d` 则把 16-byte vector 结构直接编码进 K/V physical representation，并要求 page/head dimensions 与 \(X=16/\text{dtype bytes}\) 对齐。没有消费该布局的 kernel 时，特殊 AITER layout 会被忽略。fileciteturn726file0

这条证据对应的可证伪假设是：

\[
H_{SGLang}:
\quad
\exists x_1,x_2:
P_\text{backend-tree}(x_1)=P_\text{backend-tree}(x_2),
\quad
backend^*(x_1)\ne backend^*(x_2).
\]

最干净的实验不是随机扫模型，而是在完全相同 model/arch/quant/spec flags 下，只改变 batch、raggedness、GQA ratio workload 或 MoE routing histogram。如果 backend winner 不发生有意义的 crossover，那么这个假设被否证。

证据等级：**B/C**。源码清楚表明使用 feature tree，并存在持续特化 repair；但还没有像 vLLM 那样直接在同一 selector 源码里留下一个未读取变量的性能 crossover。

---

# 4. vLLM：目前 RQ1 最强的直接证据

vLLM 这里现在有一条非常难回避的证据。

当前 CUDA `_get_backend_priorities()` 的参数包括：

\[
\{
use\_mla,\,
device\_capability,\,
num\_heads,\,
kv\_cache\_dtype,\,
use\_non\_causal,\,
head\_size,\,
use\_mm\_prefix
\}.
\]

但没有 live batch size。SM100 MLA 的 priority list 中，源码却明确写：

> TokenSpeed MLA 在 `bs≈8` 之后胜出，而 `bs≤2` 会退化。

同时它仍排在 FlashInfer MLA 后面。fileciteturn676file0L1-L2

随后 `get_valid_backends()` 对这些静态 priority candidate 做 capability/configuration legality checking，再选择最高优先级的合法 backend。fileciteturn648file0L1-L7

TokenSpeed 自己的 `supports_combination()` 检查 DeepSeek R1 的固定 MLA dimensions、Blackwell、FP8 KV、依赖是否安装等，但同样没有 batch performance branch。fileciteturn652file0L1-L7

于是存在一个源码已经暗示的 collision：

\[
x_1=(B=1,\text{其他selector输入固定})
\]

\[
x_2=(B=16,\text{其他selector输入固定})
\]

从 selector 看：

\[
P(x_1)=P(x_2),
\]

但源码作者留下的性能注释暗示：

\[
backend^*(x_1)\ne backend^*(x_2)
\]

至少具有非常高的可能性。

这是目前 RQ1 的 **A级证据**。

---

## vLLM 的第二条证据：persistent KV representation

当前代码明确：

> Resolve one KV cache layout for the whole model. fileciteturn706file0L1-L35

backend 各自返回 most-preferred-first 的 supported layout 集合；resolver 求交。如果各backend的顺序不同，则“被最多 backend 放在第一位”的 layout 胜出，tie 按 enum 顺序。fileciteturn707file0L1-L43

即它优化的是：

\[
C=
\bigcap_i S_i
\]

然后近似：

\[
score(L)
=
\sum_i
\mathbf 1[
L=first(S_i)
].
\]

不是：

\[
\arg\min_L
\big(
T_\text{prefill}
+
T_\text{decode}
+
T_\text{append}
+
T_\text{transfer}
+
T_\text{connector}
\big).
\]

但是这不能直接判有罪，因为 persistent representation 的转换代价可能让“稳定统一布局”本身就是正确策略。

vLLM 还有另外两个独立 representation decision：

\[
\text{axis layout}
\neq
\text{page/block size}
\neq
\text{content packing}.
\]

例如：

- backend 可通过 `get_preferred_block_size()` 决定合法/偏好的 page geometry；fileciteturn607file0L1-L7
- FlashInfer NVFP4 把 K/V 做成 packed FP4 + FP8 block-scale representation；fileciteturn727file0L1-L36
- Triton per-token-head quant 又将 FP32 scale inline 在每个 head 数据之后。fileciteturn728file0L1-L40

所以 vLLM 的真实 representation policy 至少是：

\[
Backend
\rightarrow
AxisLayout
\rightarrow
PageGeometry
\rightarrow
Content/ScalePacking.
\]

RQ1 需要检验这些局部 selector 的状态是否足够，而不是把它们压成一个“KV layout heuristic”。

可证伪实验中，第一优先级就是直接重测当前 2026 代码：

\[
B\in\{1,2,4,8,16,32\}
\]

固定 SM100、DeepSeek-R1 dimensions、FP8 KV 等全部 selector state，benchmark：

\[
FlashInferMLA
\quad vs\quad
TokenSpeedMLA.
\]

如果当前版本已经没有 crossover，或者 static priority 的 regret 始终小于噪声/例如 3%，那么源码注释可能已经过时，这条 A 级证据会被实证否决。

---

# 5. CUTLASS/CuTe：结构规则非常强，但也公开编码性能假设

CUTLASS 不适合概括成“手工 layout”。SM100 builder 会共同构造 MMA、GMEM copy、SMEM layout、pipeline stages 和 schedule。

最清楚的三个 RQ1 rule 是：

| Rule | Proxy | 未读取的状态 | repair/escalation | 现代触发 |
|---|---|---|---|---|
| `sm100_smem_selector` | element type、major、tile divisibility | 当前完整 consumer 集、实际测得 latency、下游 conversion/pipeline total cost | 手工 layout / expert kernel | GEMM、GQA、MoE |
| `StageCountAuto` | tile A/B bytes、pipeline storage、SMEM carveout | actual problem size、short-K、small-M、其它 occupancy benefit | explicit `StageCount<N>` | GEMM/grouped/MoE |
| 1SM/2SM / pipeline builder rules | cluster/tile structural properties | measured crossover under actual workload | explicit schedule / profiler | Blackwell |

SM100 shared selector非常明确地选择“能放进去的最大 UMMA layout”：

\[
SW128
\rightarrow
SW64
\rightarrow
SW32
\rightarrow
INTER.
\]

TF32 MN-major甚至只有特定 SW128_32B。fileciteturn688file0L1-L2

这个函数的实际 proxy 就是：

\[
P=
(
major,
dtype,
BLK_{MN},
BLK_K
).
\]

它并不读取“这个 shared tensor 还被谁以什么 pattern 访问”。

`StageCountAuto` 的源码则直接算：

\[
stageBytes
=
bytes(A_{tile})
+
bytes(B_{tile})
+
pipelineStorage
\]

\[
stages
=
\left\lfloor
\frac{Capacity-carveout}
{stageBytes}
\right\rfloor.
\]

fileciteturn729file0L1-L2

官方文档进一步明确说，这是最大化 SMEM utilization、**假设 1 threadblock / multiprocessor occupancy**。fileciteturn653file0L1-L32

这是一个非常干净、可以证伪的性能假设：

\[
H_{stage}:
\text{增加pipeline stages的收益通常大于提高CTA residency的收益}.
\]

Blackwell builder 里还有更明确的经验规则：accumulator pipeline stage 数 cap 到 4，注释称 4 stages “works well”，并用于兼顾 buffer accumulator 与小 tile epilogue tail；scheduler pipeline 又用比 accumulator 多一 stage 来隐藏 latency。fileciteturn729file0L1-L2

现代 Attention 的直接 source trigger 也已经找到了。CUTLASS 自己的 Blackwell low-latency GQA kernel 中：

\[
qH=kvH\times qHLocal.
\]

K/Q/V/O 有显式 GQA physical mapping，而且对 softmax intermediate S，源码说明没有候选 swizzle 能做到无 bank conflict，64/128B 至多造成 2-way conflict，但因为 output tile 小而认为 “good enough”，随后显式选择 SW128。fileciteturn721file0

这是非常有价值的一条反证：

> “最小 bank conflict”本身也不是充分 objective。

真实策略已经隐含：

\[
bankConflictCost
\times
accessVolume.
\]

因此 CUTLASS 对 RQ1 的贡献不是“largest swizzle 很蠢”，而是：

\[
\boxed{
结构规则什么时候足够强，以至于不需要完整性能状态？
}
\]

可证伪实验最有价值的是 grouped/MoE small-\(M_e\)、short-K 等场景中比较：

\[
StageCountAuto
\]

和多个显式较少 stage 的合法配置。若 Auto 始终 near-optimal，则这是 RQ1 的强反证。

同时 CUTLASS 已支持 nvMatmulHeuristics / profiling 对 operators 做 estimated/measured ranking，因此它也提供了从 structural rule 向 richer feedback 升级的路径。fileciteturn644file3L65-L76

证据等级：**B + 强反证能力**。

---

# 6. Triton classic：RQ1 必须攻击剩余 heuristic，而不是错误地攻击其 bank-conflict machinery

当前 `RemoveLayoutConversions` 先找 layout anchors，再把 anchor layout 向 descendants 传播；一个 value 可以同时收到多个 layout，之后 conflict resolution 选择 representation，并通过 `convert_layout`、rematerialization、hoisting 等修复不一致。fileciteturn671file0L1-L2

其 pass 目标也写得非常清楚：

- expensive load/store 偏好 `BlockedEncoding`；
- tensor-op 路径偏好 `NvidiaMmaEncoding`。fileciteturn669file0L1-L19

这意味着：

\[
L_\text{memory}
\neq
L_\text{MMA}
\]

是编译器内部显式存在的 conflict。

这里的 RQ1 不是“是否知道两种目标冲突”，Triton显然知道；而是 conflict resolution 的状态是否足以判断：

\[
\text{keep memory layout}
\quad vs\quad
\text{keep MMA layout}
\quad vs\quad
\text{convert/rematerialize}.
\]

---

## Triton 的强 B 级证据

Pipeliner 当前遇到 loop 中存在 sync dot 时，会跳过 MMA pipelining。源码自己称：

> “dirty heuristic for performance drops”，原因涉及最后一个 masked iteration 和 wait。fileciteturn712file0L1-L31

该具体 branch 的核心状态是：

\[
P=\text{hasSyncDots(loop)}.
\]

它没有按不同 runtime/kernel shape 实测：

- loop trip count；
- tail fraction；
- masked final iteration占比；
- resulting register/SMEM pressure

再做 crossover 判断。

所以真正可测的是：

\[
hasSyncDots=1
\]

保持不变时，是否存在两种 loop geometry，使：

\[
pipeline^*(x_1)=off,\qquad
pipeline^*(x_2)=on.
\]

---

## GenericSwizzling 需要更精确地描述

当前 Triton 已经显式计算：

- register/shared linear layouts；
- vectorization；
- bank conflicts；
- source/destination phase；
- lane mapping。

所以不能再说它“忽略 bank conflict”。fileciteturn614file0L1-L7

但代码仍存在明确的 residual heuristics：

- lane mask缺失时假设 lane IDs 在 phase 内顺序排列；
- source/destination vectorization候选同规模时，选择较低 basis，源码写的是“**in the hope that it will avoid PRMTs**”；
- asymmetric vectorization + bank-conflict-free swizzle 尚未实现，因此某些情况下会主动裁剪 vector basis。fileciteturn615file0L1-L2

这给出一个更严谨的 RQ1 子问题：

\[
\boxed{
\text{bank-conflict/vectorization algebra}
\text{ 是否已足以代理最终 instruction-level cost？}
}
\]

而不是错误地问“为什么 Triton 不分析 bank conflict”。

现代 trigger 最自然的是：

\[
QK^T
\rightarrow
softmax/reduction
\rightarrow
PV
\]

因为 load、dot、reduction、第二个 dot 会连续制造不同 layout anchors。GQA 又进一步改变 head sharing 和 work partition。

证据等级：**B/C**。

---

# 7. Triton Gluon：这里的 proxy 甚至不是 performance proxy

Gluon 对 layout conflict 有一个非常干净的机制。

每个传播结果是：

\[
LayoutInfo=(encoding,distance).
\]

Join/Split/Reshape/Transpose 这类 fuzzy inference 会增加 `distance`。发生冲突时：

\[
d_1<d_2
\Rightarrow
L_1
\]

较近者获胜。

若：

\[
d_1=d_2>0,
\]

使用 stable hash 做 deterministic tie-break。

若两个不同 encoding 都是：

\[
d=0,
\]

表示明确 seed 直接冲突，报错。fileciteturn602file0L1-L7

这里必须精确区分：

\[
distance
\]

代表的是 **inference confidence/proximity**，源码没有把它声明成 latency estimator。

所以 Gluon 的 RQ1 不是：

> distance heuristic 错了。

而是：

> 当 AutoLayout 允许编译器在多个一致布局之间选择时，inference confidence 是否偶然也足够作为 performance preference；如果不够，什么情况下应要求程序员显式指定？

Coalesced layout 的 seed 则来自 AxisInfo、numWarps、threadsPerWarp、shape-per-CTA 等；当前实现还明确 TODO 多CTA，现有路径要求 `numCTAs==1`。fileciteturn603file0L1-L7

对于 NVMMA shared layout，`get_default_for()` 又是另一种完全不同的 proxy：

> 选择 shape-compatible 的最大 swizzle，因为这样可以发出更少的 TMA/MMA messages。fileciteturn693file0L1-L2

因此 Gluon 实际混合了：

\[
\boxed{
\text{explicit expert layout}
+
\text{inference-distance selection}
+
\text{message-count structural heuristic}
}
\]

而不是“manual layout system”。

现代 workload 也不是假想：Gluon 当前 MoE fused BMM 示例直接调用 `NVMMASharedLayout.get_default_for()`。fileciteturn692file4L80-L90

可证伪实验应该把“inference正确性”和“performance”分开：

\[
L_{near-seed}
\quad vs\quad
L_{far-seed}
\]

都显式合法，然后测真实时间。若 distance-selected candidate 始终 near-optimal，那就是 RQ1 的反证。

证据等级：**C**，非常干净的 non-performance proxy。

---

# 8. TVM MetaSchedule：RQ1 的重要“无罪对照组”

TVM 这里尤其容易误判。

默认 CUDA schedule space 的确有非常强的 structural prior：

\[
structure=\texttt{SSSRRSRS},
\]

vector load 只取：

\[
\{1,2,3,4,8,16\},
\]

并规定 shared/local reuse level、cross-thread reduction thread extents、AutoBind thread candidates。TensorCore 规则也使用固定的 intrinsic groups、相同 tiling grammar，以及少数 software-pipeline/reuse variants。fileciteturn696file0L1-L2

但这些属于：

\[
\boxed{\text{candidate-source restriction}}
\]

而不是 RQ1 的最终 performance selector。

MetaSchedule 的 Builder/Runner 会真正编译并在硬件上执行 candidates，获得 measured runtimes。fileciteturn697file0L1-L18

所以在固定候选空间内：

\[
P(x)
\approx
T_\text{measured}
\]

而不是 low-dimensional static proxy。

因此 TVM 对 RQ1 是关键反证：

> 如果 hardware measurement 的代价可以接受，就可以避免很多 proxy-sufficiency 问题。

但它把问题转移成另一个独立问题：

\[
\boxed{
\text{真正好的 candidate 有没有被 schedule rules 生成？}
}
\]

这是未来的 RQ3，不应偷进 RQ1。

所以 TVM 在 RQ1 的 verdict 应该是：

**N（negative control / counterexample）**。

这一步很重要，否则会把“candidate coverage”和“candidate ranking”混成一个问题。

---

# 9. TIRx：最纯粹的 legality + priority 系统

当前 dispatcher 的语义非常明确：

1. variants 具有 `priority`；
2. 高 priority 先；
3. 每个 variant 跑 predicates；
4. predicate不合法则继续；
5. implementation可 `DispatchFail`；
6. 第一个成功返回 `PrimFunc` 的 variant 立即结束。fileciteturn700file0L1-L10

所以它的选择函数本质是：

\[
\pi(x)
=
\operatorname{firstSuccess}
\left(
sort_{\downarrow priority}
\{c_i\}
\right).
\]

CUDA synchronous copy 中，`vec_auto` 的 priority 是 10。fileciteturn625file0L1-L7

global↔shared 路径检查：

- CUDA target；
- execution scope；
- all threads active；
- copy legality；
- memory-scope pair；
- region size 能否整除 thread count；

然后根据 scope 合成 work partition/vector width。fileciteturn621file0L1-L7

最后还有 priority 0 的 scalar single-thread fallback，源码甚至会 warning：

> all faster variants rejected. fileciteturn627file0L1-L7

MMA 也类似：当前 CUDA `mma.m16n8k*` priority 10，要求 M/N/K divisibility 和 layout compatibility，并在可用时先试 m16n8k16，再 m16n8k8。fileciteturn701file0L1-L18 fileciteturn701file1L19-L40

这形成 RQ1 非常标准的实验：

找到：

\[
c_a,c_b
\]

都满足 predicates。

当前：

\[
priority_a>priority_b.
\]

然后找到两个 workload：

\[
x_1,x_2
\]

保持 legality 相同。

如果：

\[
T(c_a,x_1)<T(c_b,x_1)
\]

但：

\[
T(c_a,x_2)>T(c_b,x_2),
\]

那么 fixed priority 不可能同时最优。

关键 distinction 是：

\[
\boxed{
fallback\ solves\ legality
\neq
fallback\ solves\ performance\ crossover.
}
\]

证据等级：**C**。源码机制完全确定，但是否真的存在 material crossover 必须测。

---

# 10. Hexcute：这是“解析性能 proxy”最清楚的样本

Hexcute 先通过 algebraic constraints 推导可行 layout/task mapping，而不是直接 heuristic 猜布局。

Copy：

\[
f\circ p^{-1}
=
g\circ q^{-1}.
\]

MMA 又约束 M/N/K mapping 一致。多个 consumer 对同一 shared tensor 的 constraints 会被统一；不可满足则 backtrack；全部约束满足后，仍有未定 stride 时再用 heuristic 填充。fileciteturn633file0 fileciteturn633file1

所以：

\[
\text{constraint system}
\]

主要负责：

\[
\mathcal C
\rightarrow
\mathcal C_\text{legal}.
\]

真正 RQ1 的关键在 ranking。

Hexcute 的 `cost_model.py` 自己把模型定义得很明确：

\[
cost
=
N_\text{instructions}
\times
instructionLatency.
\]

instruction latency 来自 microbenchmark CPI。fileciteturn717file0L1-L2

而且源码直接列出假设：

- copy 与 MMA 可以充分 overlap；
- pipeline fully overlapped；
- 不计 `cp_async_wait_group()`；
- 不计 `mbarrier`；
- address calculation 近似零成本；
- cost model 本身还没有 bank-conflict term。

这些不是我们的推断，是 cost-model 文件自己的说明。fileciteturn717file0L1-L2

但又不能说 Hexcute“完全忽略 bank conflict”，因为它有独立 `ResolveBankConflict` pass：

- 按 underlying shared tensor 聚合所有 copy；
- 枚举 swizzle；
- 计算每个 access 的 bank conflict；
- 最小化所有 copy 的累计 conflict；
- TMA 的四种合法 swizzle全部评估；
- WGMMA operand 会锁定/限制 shared layout。fileciteturn718file0L1-L2

也就是说它采取：

\[
\boxed{
layout/instruction ranking
\rightarrow
bank-conflict refinement
}
\]

而不是一个 monolithic objective。

另一个 repair 更直接：如果程序希望使用 TMA，但 layout 使 TMA 无法执行，专门的 pass 会自动退成 `cp_async`，同时修正 mbarrier transaction counts 并生成 mask。fileciteturn719file0L1-L2

所以 Hexcute 最准确的 RQ1 假设是：

\[
H_{Hexcute}:
\]

> constraint solving 已经把 candidate space 收缩得足够规则，因此 instruction-count × latency + staged bank-conflict repair 已能保持 near-optimal candidate ranking。

检验指标不应该主要看：

\[
|\hat T-T|.
\]

而应该看排序：

\[
T(c_i)<T(c_j)
\Rightarrow
\hat T(c_i)<\hat T(c_j),
\]

例如 Spearman、top-1/top-k oracle recall 和 final regret。

它的 benchmark artifact 确实包含 Attention、FP8 GEMM、warp-specialized kernel、mixed-type MoE 和 cost-model accuracy，因此这些并不是脱离目标workload的假设。fileciteturn723file0L1-L13

证据等级：**B + 很强反证能力**。

---

# 11. TileLang：目前另一条A级证据

TileLang 当前 free-mode inference 会枚举 connected component 内的完整 layout attempts，再通过 pluggable cost model 排序。fileciteturn724file0L1-L13

默认：

\[
C_\text{register}
\]

主要由：

- local spill bytes；
- total fragment register slots

决定。

`io-aware` 则对 fragment↔global 和 direct-global parallel-loop statement 估计：

\[
C_s
=
\max(
BandwidthBytes_s,
IssueEquivalentBytes_s
)
\]

然后 register count 做 lexicographic tie-break。fileciteturn724file0L1-L13

这里最强的证据来自框架自己的 verification harness。

源码明确说：

> 两个 cost model 发生分歧的 case 是 calibration corpus；每次分歧都对应一个可以在硬件上验证的具体性能 claim。fileciteturn725file0L1-L13

更关键的是 `broadcast_read`：

- register-count 会保留 thread-collapsed legacy pathology；
- io-aware 需要选择 fully replicated small fragment + non-replicated/coalesced output loop。

同一个 harness 还把 `transposed_store` 和 `mixed_dtype_chain` 中的分歧称为 benchmark-worthy。fileciteturn725file0L1-L13

所以这里不是我们说：

> “register count 可能不够。”

而是源码维护体系已经明确建立：

\[
\boxed{
\text{proxy disagreement}
\rightarrow
\text{hardware calibration claim}
}
\]

这几乎就是 RQ1 所需的实验方法本身。

但是同一 harness 也有非常重要的反证：

`reduce_broadcast` 是 softmax-shaped row reduction + broadcast，两套模型一致。fileciteturn725file0L1-L13

所以不能得出：

\[
register-count\text{ 总是错误}.
\]

真正问题仍然是：

\[
\boxed{
\text{什么结构区域内 cheap proxy 已经充分？}
}
\]

另外 `shared_staging` 给出一个极重要的边界：

> global→shared→fragment→global 中，shared-side copy 在 io-aware model 之外，因此 fragment 主要由 copy-out 决定。fileciteturn725file0L1-L13

这将来会和 RQ2/RQ4 有关系；本轮只把它记为 RQ1 的 missing-state candidate，不提前展开边界问题。

冲突无法正常解时，TileLang还有 replicated wide fallback，把 reserved destination 固定到“universally readable replicated layout”。fileciteturn704file0L1-L24

同样：

\[
correctness/compatibility\ repair
\neq
performance\ repair.
\]

证据等级：**A**。

---

# 12. FlashInfer：补证后，RQ1 应该只攻击两个真正的信息压缩点

## 12.1 不再把 decode split planning 当弱 proxy

当前 decode planner 实际会：

- 计算 kernel occupancy；
- 使用 GQA `GROUP_SIZE`；
- 推导 `num_kv_heads` 和 grid-y；
- 收集每个请求的 page count；
- 对 pages-per-chunk 做二分；
- 根据 CUDA graph 与 resulting batch size 判断是否 split。

fileciteturn685file0L1-L2

所以：

\[
\{KV_i\}
\]

并没有被完全压缩掉。

这一条应从以前的 RQ1 支持证据中删除。

---

## 12.2 Prefill CTA-Q 才是干净的 state compression

`FA2DetermineCtaTileQ` 接收：

\[
(
\overline{packedQ},
D_{VO},
D_{QK},
bytes_{KV}
).
\]

其中 \(D_{QK}\) 用于真实 shared-memory cost，KV byte width也进入模型。fileciteturn682file0

对于 VO≥512，代码明确：

\[
\overline{packedQ}\le32
\Rightarrow CTA_Q=16
\]

否则 long-Q 可以用 CTA32，因为 VO split 减少 output fragment register pressure。fileciteturn683file0L1-L33

普通区间又会在 16/64/128 中选择；例如 average packed Q 大于64且 head dim<256 会选择128。fileciteturn682file0

因此真正压缩的是：

\[
\{packedQ_i\}_{i=1}^B
\rightarrow
\overline{packedQ}.
\]

于是最干净的 collision test 是：

\[
A=[32,32,32,32]
\]

和例如：

\[
B=[1,1,1,125].
\]

两者平均值相同，但：

- tail waste；
- long/short request mix；
- active CTA distribution

明显不同。

如果 legal CTA family 中最优 tile 翻转，那么 mean 不是 sufficient statistic。

如果不翻转，则这是非常强的反证。

---

## 12.3 Split-KV reduction mode 是更强的 B 级证据

`select_split_kv_modes()` 实际接收：

- family；
- topology；
- tile_size_q；
- head_dim；
- head_dim_per_cta_v；
- split_kv；
- available modes。

但 FMHA 当前自动顺序是：

\[
cluster\_smem
\succ
gmem\_separate
\succ
gmem\_inline.
\]

源码明确解释：

> 保持自动选择为 structural，exact residency/support 检查通过就优先 cluster；这样可以避免 shape-specific measured crossover tables。fileciteturn680file0L1-L17

这是一条极其干净的设计取舍：

\[
\boxed{
structural\ robustness
\quad\text{vs}\quad
shape\text{-}specific\ performance\ crossover
}
\]

所以实验应该只挑：

\[
split\_kv>1
\]

且多个 reduction mode 同时合法的区域，然后改变：

\[
Q,\ KV,\ D,\ split
\]

测三种 mode 的真实 crossover。

证据等级：**B**。

---

# 13. 把 10 个框架放回同一张 RQ1 审判表

下面这张表才是这轮最重要的结果。

| Framework | 被审判的具体 rule | 实际 proxy / state | 明确未进入该 rule 的信息 | 自己的 fallback / repair | 现代触发 | 证据 |
|---|---|---|---|---|---|---|
| SGLang | `get_default_attn_backend` | arch、spec、asym-KV、sink、head count等 | live B、request-length distribution、load | fallback tree + model overrides + split phase backend | MHA/GQA/MLA/spec/MoE | B/C |
| vLLM | `_get_backend_priorities` | arch、MLA、heads、dtype、head size、flags | **live batch size** | next legal backend | SM100 MLA | **A** |
| vLLM | whole-model KV-layout resolver | support intersection + first-choice vote | live prefill/decode mix、request distribution | connector/default candidate | paged GQA/MLA | C |
| CUTLASS | `StageCountAuto` | tile bytes + pipeline bytes + SMEM | problem size、actual residency tradeoff | explicit stages / profiler | GEMM/MoE | B |
| CUTLASS | `sm100_smem_selector` | dtype/major/tile divisibility | full consumer set、measured total latency | explicit layout/expert kernel | GQA/GEMM | C |
| Triton | sync-dot pipeline skip | `hasSyncDots` | per-shape crossover / measured tail cost | skip/other pipeline paths | Attention/GEMM | B |
| Triton | GenericSwizzling residual choices | layout algebra、vectorization、bank model | actual PRMT cost、unsupported asymmetric choice | conservative vector trim | transposed/asymmetric staging | B |
| Triton | RLC anchors | expensive mem-op vs MMA/etc | measured whole-subgraph latency | conversions/remat | QK-softmax-PV | C |
| Gluon | AutoEncoding conflict | inference distance + deterministic tie | measured performance | explicit seed / conflict error | fused Attention/MoE | C |
| Gluon | NVMMA shared default | largest compatible swizzle / fewer messages | full multi-op latency | explicit swizzle | MMA/MoE | C |
| TVM | candidate ranking | **real Builder/Runner runtime** | — within generated candidate set | search iterations | TIR kernels | **N** |
| TIRx | dispatcher | legality predicates + priority | measured latency among simultaneous legal variants | next candidate / scalar fallback | copy/MMA | C |
| Hexcute | layout cost model | instruction count × CPI + overlap model | bank conflict term、barrier/wait cost、imperfect overlap | separate bank pass / backtrack / TMA→cp.async | Attention/MoE | B |
| TileLang | register-count | spills + register slots | global coalescing/issue | io-aware / replication fallback | broadcast/reduction | **A** |
| TileLang | io-aware | global traffic + issue + spill/register | shared-side staging等未建模项 | explicit layout / fallback | softmax/fused kernels | B |
| FlashInfer | CTA-Q | avg packed-Q + QK/VO dim + KV bytes | distribution higher moments/tail | finite legal tile family | ragged prefill/GQA | C |
| FlashInfer | split-KV reduction mode | topology/support/residency structural order | measured shape crossover | next supported mode | long-KV GQA/MLA decode | B |

这里 A/B/C/N 的含义很重要：

\[
A=\text{源码已经暴露具体性能冲突/病态}
\]

\[
B=\text{源码作者明确声明 heuristic/approximation/design tradeoff}
\]

\[
C=\text{源码结构证明是非实测 proxy，但尚无直接性能反例}
\]

\[
N=\text{RQ1反例/negative control}.
\]

目前真正达到 A 的主要就是：

\[
\boxed{
vLLM,\ TileLang
}
\]

而不是“所有框架都已经证明 proxy 失败”。

这比上一轮严谨得多。

---

# 14. 现代 Attention/GQA/MLA/MoE 为什么是 RQ1 的正确 workload，而不是泛泛的 GPU benchmark

Attention 的关键不是只有 \(M,N,K\)。

实际状态包含：

\[
\{Q_i,K_i\}_{i=1}^B,
\quad
H_Q,
\quad
H_{KV},
\quad
D_{QK},
\quad
D_V,
\quad
page_i,
\quad
phase.
\]

GQA 又增加：

\[
g=\frac{H_Q}{H_{KV}},
\]

直接改变：

- KV reuse；
- grid parallelism；
- shared-KV amortization；
- split-KV geometry。

FlashInfer 的 scheduler 已经明确把 GROUP_SIZE 纳入一部分决策，因此实验不能假装框架没看到 GQA；需要寻找 **它还没看到的 interaction**。fileciteturn685file0L1-L2

MLA 状态更高维：

\[
(
D_{nope},
D_{rope},
D_v,
B,
qLen,
KVLen,
dtype_{KV},
splitKV
).
\]

vLLM 当前刚好给出了最强 source clue：同一 R1/FP8/SM100 legal region 中存在 batch-size crossover 注释，而 selector 不读 batch。fileciteturn676file0L1-L2

MoE 则提供另一种完全不同的 state compression：

\[
M_e=
\text{tokens routed to expert }e.
\]

保持：

\[
\sum_e M_e=N
\]

不变时，可能从 uniform：

\[
[512,512,\ldots]
\]

变成极端 skew：

\[
[3000,500,200,100,\ldots].
\]

model identity、quantization、GPU、total token count 全部可以保持不变，但每个 grouped GEMM 的 occupancy/tile efficiency 已经完全不同。

因此如果相同 backend/tree state 下 optimal runner/layout 随：

\[
\{M_e\}
\]

翻转，就构成 MoE 版 proxy collision。

Hexcute 的官方 artifact 本身就包含 mixed-type MoE；CUTLASS 当前 builder规则也直接适用于 grouped/kernel constructions，所以这不是人为把 MoE 拉进问题。fileciteturn723file0L1-L13

---

# 15. RQ1 现在应当拆成四个可证伪假设，而不是一句宽泛口号

## H1 — Proxy Collision

\[
\boxed{
\exists x_1,x_2:
P(x_1)=P(x_2)
\land
c^*(x_1)\neq c^*(x_2)
}
\]

这是最基础条件。

没有 collision，RQ1 基本不成立。

---

## H2 — Systematic Regret

单个 crossover 不够。

需要：

\[
\Pr_{x\sim\mathcal W}
[
R_P(x)>\epsilon
]
>\delta
\]

在 realistic Attention/GQA/MLA/MoE workload distribution 上出现。

否则它只是 corner case。

这里 \(\epsilon\) 不应预先神圣化成某个数字。实验阶段可以例如用 3%、5%、10% 分层报告。

---

## H3 — Minimal State Augmentation

假设原状态：

\[
P(x)
\]

不充分。

寻找很小的：

\[
z(x)
\]

使：

\[
P'(x)=
[P(x),z(x)]
\]

就能显著减少：

\[
R_{P'}.
\]

目前从源码反推最值得测试的 candidate state 包括：

\[
\text{Attention}:
\quad
\{
B,
Var(Q),
Q_{\max},
KV\ tail,
GQA\ ratio
\}
\]

\[
\text{MLA}:
\quad
\{
B,
qLen,
splitKV,
D_{QK},
D_V
\}
\]

\[
\text{MoE}:
\quad
\{
max(M_e),
Var(M_e),
activeExperts,
routingSkew
\}.
\]

注意它们只是 candidate，不应先假定每个都必要。

---

## H4 — Adaptive Fidelity

这是新证据链形成的最有价值的新方向，但我暂时把它保留在 RQ1 内，不单独升格 RQ。

因为 10 个系统实际上形成了一条 fidelity ladder：

\[
\text{static rule}
\rightarrow
\text{structural model}
\rightarrow
\text{analytic cost}
\rightarrow
\text{search}
\rightarrow
\text{hardware measurement}.
\]

问题是：

\[
\boxed{
什么时候 cheap selector 已经足够，
什么时候值得升级？
}
\]

定义 cheap selector 的 uncertainty：

\[
u(x).
\]

可以研究：

\[
u(x)<\tau
\Rightarrow
\text{accept cheap result}
\]

否则：

\[
\text{escalate}.
\]

H4 是：

> 存在低成本 uncertainty signal，可以捕获大部分 high-regret states，使大多数常规case继续走 cheap rule，仅少数困难case进入昂贵搜索/实测。

如果以后它独立形成完整跨框架证据链，可以升为新 RQ；现在不提前扩张。

---

# 16. RQ1 的“问题审判”必须怎么做实验才算真正判决

最优先的实验矩阵现在已经可以明确下来。

| Trial | 固定什么 | 只变化什么 | 对比 candidate | 判有罪条件 | 判无罪条件 |
|---|---|---|---|---|---|
| vLLM MLA | SM100/R1 dims/FP8/heads/page等 | batch B | FI-MLA vs TokenSpeed | winner随B翻转且regret显著 | 当前版本无crossover |
| TileLang | 同一 harness IR | cost model/layout | register vs io-aware winner | pathology对应真实latency损失 | 布局不同但性能等价 |
| FlashInfer Q-tile | mean packed-Q、dims、dtype | variance/tail | 16/64/128等legal tiles | 相同mean下winner翻转 | winner稳定 |
| FlashInfer split reduction | legal topology/support | Q/KV/dim/split | cluster/separate/inline | structural order出现crossover | cluster/order稳定near-opt |
| CUTLASS stage | tile/type/arch | workload M/K / expert size | Auto vs explicit stages | lower stage因residency胜出 | Auto持续near-opt |
| Triton pipeline | `hasSyncDots=true` | trip/tail/resource geometry | pipeline on/off合法版本 | optimal decision翻转 | skip rule持续正确 |
| Triton swizzle | src/dst legality | asymmetric access geometry | legal layouts/swizzles | residual heuristic排序翻转 | analytic swizzle足够 |
| Gluon | same graph legality | seed-distance structure | explicit legal layouts | farther layout显著更快 | distance winner near-opt |
| TIRx | same predicates true | shape/resource state | ≥2 legal variants | fixed priority发生winner inversion | ordering稳定 |
| Hexcute | same legal set | barrier/bank/pipeline situations | synthesized candidates | analytic ranking inversion | high top-k/rank accuracy |
| MoE | model/GPU/quant/total tokens | \(M_e\) histogram | runners/layouts | same static state winner翻转 | runner ranking稳定 |

这里最重要的是第一列中的控制。

比如 FlashInfer Q-tile collision 必须保证：

\[
\overline{packedQ}
\]

真的相同。

TIRx 必须保证：

\[
Pred_i(x_1)
=
Pred_i(x_2)
=
true.
\]

否则只是合法性区域变了，不是 proxy collision。

Hexcute 必须在同一 legal candidate set 中比较，否则可能只是 constraint pruning 的差异。

这就是“问题审判”和普通 benchmark sweep 的区别。

---

# 17. 哪些结果会直接推翻 RQ1？

这一点必须明确，否则 RQ 永远无法被否证。

如果最终出现以下模式：

\[
P(x_1)=P(x_2)
\]

的 states 中，虽然底层状态变化巨大，但：

\[
c^*(x_1)=c^*(x_2)
\]

几乎总成立；或者即便 winner 偶尔翻转：

\[
R_P(x)<\epsilon
\]

始终很小，那么 RQ1 的核心 hypothesis 应被拒绝。

特别是，如果：

- vLLM 当前 kernel 已不存在 source-comment 所说的 batch crossover；
- FlashInfer 的 `avg_packed_qo_len` 在 realistic ragged batches 上仍几乎总选到最佳 tile；
- structural split-KV mode order在所有合法shape上都near-optimal；
- CUTLASS `StageCountAuto` 在 small-expert/short-K仍不输；
- TIRx static priorities在合法candidate之间表现稳定；
- Hexcute简单analytic model的 ranking recall非常高；
- TileLang register-count 的所谓 layout pathology并没有显著影响实测性能，

那么最值得发表的结论反而可能是：

\[
\boxed{
\text{现代 GPU layout/configuration 最优区域具有非常强的低维结构。}
}
\]

也就是说 cheap structural proxies 已经接近 decision-sufficient。

这同样是一个有效科研结论。

---

# 18. 反过来，什么结果才算真正证明 RQ1？

我会要求四项同时满足。

第一，需要**同 proxy collision**：

\[
P(x_1)=P(x_2).
\]

第二，需要**真实 winner 翻转**：

\[
c^*(x_1)\ne c^*(x_2).
\]

第三，需要**material regret**：

\[
R_P>\epsilon
\]

而不是测量噪声。

第四，需要**可解释的最小 missing state**：

\[
P'(x)=[P(x),z(x)]
\]

可以显著降低 regret。

最理想的证据链是：

\[
\boxed{
CurrentProxy
\rightarrow
RepeatedCollision
\rightarrow
MissingState
\rightarrow
CompactAugmentation
\rightarrow
RegretCollapse
}
\]

而不是：

\[
\text{某个 if threshold 慢}
\rightarrow
\text{换一个 threshold}.
\]

后者只是工程修补，不足以构成科研问题。

---

# 19. 现在 RQ1 的“判决”

经过这轮补证，我会比上一轮更谨慎地区分两个结论。

**第一，RQ1 已经通过“值得研究”这一层审判。**

原因不是“很多框架用了 heuristic”，而是现在已经有完整的跨框架机制链：

\[
\begin{array}{rcl}
\text{SGLang} &:& feature-state tree\\
\text{vLLM} &:& static priority + persistent preference\\
\text{CUTLASS} &:& structural resource rules\\
\text{Triton} &:& local analytic/heuristic conflict decisions\\
\text{Gluon} &:& inference confidence + explicit structural defaults\\
\text{TVM} &:& measurement\ as\ counterexample\\
\text{TIRx} &:& legality + static priority\\
\text{Hexcute} &:& constraints + analytic latency\\
\text{TileLang} &:& explicit competing performance proxies\\
\text{FlashInfer} &:& workload-summary + structural policy
\end{array}
\]

而且其中至少 vLLM 与 TileLang 已达到 A 级源码证据。

**第二，RQ1 的科学假设还没有被“判有罪”。**

目前源码最多证明：

\[
\text{现有 selector 有信息压缩/近似/静态排序}
\]

以及少数地方已有已知 crossover/pathology。

它还没有证明：

\[
\text{这些近似在真实现代 workload 上造成系统性 selection regret}.
\]

这一点必须由上面的 hardware falsification 完成。

因此我现在会把 RQ1 状态标记为：

\[
\boxed{
\textbf{Source-evidence chain: closed at mechanism-family level}
}
\]

但：

\[
\boxed{
\textbf{Empirical verdict: open}
}
\]

这里的“mechanism-family level”不是声称已经逐行穷尽 CUTLASS 所有 SM100/SM103 builder specialization、SGLang 所有 model override、Triton 所有 target-specific lowering；而是说目前继续新增这些 leaf rule，绝大多数已经落入上面识别出的 selector mechanism，没有再发现足以改变 RQ1 定义的新机制。

还有一个边界需要保留：TVM 的 candidate-space restriction、`RewriteLayout` 的 single-consumer assumption，以及 pipeline/layout 决策前后顺序，已经形成了很强的下一层证据，但它们分别属于 **candidate coverage、multi-consumer、decision-boundary/separability**，不应该为了让 RQ1 看起来更强而塞进 RQ1。

所以我这轮仍然不开始 RQ2。下一步如果继续按这个标准，最合理的是先把 **RQ1 的每个 trial 再落成具体 experimental state vector / controlled variables / candidate oracle / metric / falsification threshold**；只有连实验的“同 proxy、不同 hidden state”条件也闭合之后，才真正结束 RQ1，进入 RQ2。

## 用户

证据链不完整请继续补充。然后把证据链也凝炼为问题。然后从RQ1开始做一条完整的“问题审判”：所有相关框架源码规则 → 每个规则使用的proxy → 它忽略了哪些变量 → 框架自己的fallback/repair → 现代Attention/GQA/MLA/MoE触发条件 → 可证伪的科研假设。 RQ1彻底完成之后再审判RQ2，然后后续RQ。这样可以防止又回到前几轮“问题列了很多，但证据深度不够”的状态 。当然如果新的证据链形成新的RQ 也请补充。

## 模型回复

我会继续停留在 RQ1，不进入 RQ2。这里我会把上一轮还不够“闭环”的部分补成逐规则证据：**选择函数/Pass 本身到底读取什么状态、没有读取什么、冲突后怎样 repair、以及现代 Attention/GQA/MLA/MoE 中怎样构造“同 proxy、不同 hidden state”的反例**；如果某条旧判断被源码推翻，也会直接撤回。

## 模型工具调用参数

{"paths":["GitHub"],"query":"fetch"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 CUTLASS KernelScheduleAuto 集群调度

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 sm100_common 文件指定代码行

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton RLC 锚点与启发式

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索Trition布局转换优化相关代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取文件内容并查找相关代码行

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 异步加载与共享内存注释

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 Triton 流水线工具代码⊕

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Gluon 精确自动编码实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索文件名 ResolveAutoEncodings.cpp

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取并查看自动编码解析源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

计算 inferLayout 工具中的精确距离

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索融合推理中的 numCTAs=1

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 TVM RewriteLayout 实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 RewriteLayout 文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取布局重写源码文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 TIRx 复制调度实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索“所有更快变体均被拒绝”

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Hidet 中精确约束与统一逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索代码中的“unconstrained”相关实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 TileLang 流水线执行顺序

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 CUDA 流水线实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

Fetching Exact vLLM Backend Resolution Sources

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取剩余解析器代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 RQ1 布局选择证据链

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取工具并确定注意力后端

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取指定行号范围的文件内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 FlashInfer 注意力后端判定逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析 TileLang 注意力后端机制

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索布局推断文件及线程变量常量转换

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取布局推断代码片段

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

为恒定范围延续布局推断

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Hexcute 约束源代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索自动注释实例化与成本模型实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Hidet 自动注解实现源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

计算 KV 缓存布局与后端调度方案

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 CUDA 默认规则与向量加载配置

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取并审查 MetaSchedule 调度规则代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 DefaultCUDA 调度规则代码 հատված

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Apache TVM 中的 TIRx 复制变体

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 CUDA 向量化自动拷贝实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检查 TIRx 回退直接实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 TIRx 调度器当前实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取调度器后续实现内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 CUTLASS SM100 共享内存选择器与阶段数яда

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 Triton、vLLM、SGLang 与 FlashInfer 后端策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 中的 GenericSwizzling 文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取并检查 Triton 通用置换代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 GenericSwizzling 中与 PRMT 相关的实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 中的 PRMT 实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索非对称向量化支持

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取Gluon布局定义文件

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

继续获取 get_default 代码行

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 TileLang 布局代价模型与验证源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取验证 README 指定提交内容

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取并核实 vLLM 后端优先级源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 SGLang 模型覆盖参数源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取后端选择相关代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 SGLang 内存池布局实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索代码库中的五维向量化内存池

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 SGLang 内存池布局代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 FlashInfer 精确调度逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 FlashInfer 当前 utils.cuh 函数实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 utils.cuh 第390至435行阈值分支

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析调度器平均值与CUDA图细节

## 模型思考摘要

凝炼研究问题
凝炼了研究问题

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

澄清布局决策证据链并评估RQ1

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查找 FlashInfer 解码拆分规划器源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索最新提交中的 `select_split_kv_modes`

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Triton Pass Pipeline 顺序

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 NVIDIA 后端编译流水线

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析CUTLASS低延迟GQA访存优化

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

补充核查 CUTLASS 与编译器调度证据

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索高效四阶段累加器流水线实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton ReduceDataDuplication 源码规则

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查找 Triton 2D 张量性能提示源码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Triton 同步点脏状态启发式

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

计算 Hexcute CUDA 成本模型

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

解析 CUDA 共享内存银行冲突流水账

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查看 TMA 回退复制实现源代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 FlashInfer 当前提交的分块 KV 归约模式

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 TileLang Swizzle 合并修复

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

评估GPU注意力后端与布局优化 Strategy

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

梳理 vLLM 与 SGLang 内存布局规则

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取覆盖参数实际内容

## 模型思考摘要

推进RQ1判定

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

核实FlashInfer预填充调度长度策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

评估跨后端并发失败条件

## 模型思考摘要

补充RQ1证据
梳理Triton证据
扩展跨框架证据
补充RQ1与Triton相关证据

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

完成FlashInfer与SGLang分页注意力分析

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 is_fa3_backend_supported 相关源码

## 模型工具调用参数

{"uri":"/response/turn771","query":"def determine_attention_backend"}

## 模型思考摘要

完成 GPU 注意力后端、布局推断与调度优化证据链梳理

## 模型回复

补证后，RQ1 的边界也需要再收紧：**TVM 的 `RewriteLayout`、vLLM 的 whole-model KV layout、TileLang 的 pipeline-before-layout、Hexcute 的 fixed dataflow/pipeline boundary，虽然都很重要，但它们主要回答“优化边界/候选空间/持久化表示粒度”，不是 RQ1 的 proxy-sufficiency 本体。** 我会把这些证据保留并映射到后续 RQ 簇，不再为了增强 RQ1 而混入。这样每个 RQ 的“有罪证据”会更干净。

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 vLLM CUDA 后端回退逻辑源代码

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 vLLM 后端配置验证逻辑

## 模型思考摘要

获取并检索 vLLM 后端回退及配置验证源码

## 模型思考用时

思考了 13m 52s

## 模型回复

证据链确实还不能简单宣布“全部完成”。上一轮已经把 RQ1 的主干建立起来，但继续下钻源码后，可以把若干此前过粗的结论进一步拆成具体 selector rule。尤其需要把“候选合法性”“性能 proxy”“fallback”“实测反馈”分开，否则仍然会把不同机制混成一句 heuristic。

这一轮我仍然只审判 RQ1，不进入 RQ2。下面先把新增证据纳入总证据账本，再重新给出 RQ1 的完整问题链和可证伪条件。

---

# 一、这轮补证后，先修正四个旧结论

第一，FlashInfer decode split-KV 不能再作为“只看 occupancy、忽略请求 KV 分布”的证据。当前 scheduler 会从 `kv_indptr` 得到每个 request 的 `num_pages`，结合 `GROUP_SIZE/num_kv_heads`、occupancy-derived grid size，通过二分计算 pages-per-chunk；短序列且非 CUDA Graph 时还会取消 split。fileciteturn751file0L1-L19

因此：

\[
\boxed{
\text{FlashInfer decode split planning 已观察 KV-length distribution}
}
\]

真正的 RQ1 证据应转到 CTA-Q 的 mean compression 和 split-KV reduction mode 的 structural ordering。

第二，Triton 不能描述成“shared-memory swizzle 不分析 bank conflict”。`GenericSwizzling` 已经把 register/shared linear layout、vectorization、lane phase、bank conflict 纳入代数分析；剩下的证据是一些 residual assumptions，例如 lane mask 缺失时假设 lane ID 在 phase 内顺序排列，以及候选等价时“选低 basis，希望避免 PRMT”。fileciteturn732file0L1-L2 fileciteturn734file0L1-L16

第三，Hexcute 也不能说成“忽略 bank conflict”。更准确的是：

\[
\text{analytic layout cost}
\rightarrow
\text{separate bank-conflict refinement}.
\]

主 cost model 不包含 bank-conflict term，但后续 `ResolveBankConflict` 会对一个 shared tensor 的所有 copy consumer 联合计算冲突并枚举 swizzle。fileciteturn761file0L1-L2 fileciteturn762file0L1-L2

第四，TVM MetaSchedule 的固定 `SSSRRSRS`、reuse level、vector candidate 是候选空间 prior，不等于最终候选 ranking proxy。Builder/Runner 会真正编译、运行 candidate 并取得 measured runtime。fileciteturn719file0L1-L32

因此 TVM 是 RQ1 的重要 negative control：

\[
\boxed{
\text{restricted candidate space}
\neq
\text{proxy-only candidate ranking}
}
\]

前者应该留给后续“candidate coverage/search-space restriction”问题。

---

# 二、目前证据链到底补到什么程度

先给出诚实的完成度。这里的“完成”指针对 RQ1 是否已经把影响 selector state / proxy / ranking / repair 的主要机制找齐，而不是声称逐行枚举整个仓库里每一个架构特化 kernel。

| Framework | RQ1 主机制覆盖 | 当前状态 | 仍未声称穷尽的 leaf |
|---|---:|---|---|
| CUTLASS/CuTe | SMEM selector、TMA selector、1SM/2SM、stage count、accumulator stages | 主机制已闭合 | SM103/Rubin/特殊 quant builder 的所有分支 |
| Triton classic | layout anchors、conversion/remat、async-load gate、sync-dot pipeline、swizzle、duplication repair | 主机制已闭合 | 各 GPU backend 的所有 target-specific lowerings |
| Gluon | AutoLayout、CoalescedLayout、conflict-distance、NVMMA shared default | 主机制已闭合 | 所有显式 expert kernel layout |
| TVM MetaSchedule | schedule prior、TensorCore prior、measurement、RewriteLayout | 主机制已闭合 | 所有自定义 schedule rule |
| TIRx | dispatcher、priority/predicate、vec copy、scalar fallback、MMA variant | 主机制已闭合 | 全部注册 primitive variants |
| Hexcute | constraints、candidate search、cost model、bank repair、TMA fallback | 主机制已闭合 | 所有 instruction family 的 rule leaf |
| TileLang | inference order、constant thread topology、candidate scoring、fallback | 主机制已闭合 | 每个 TileOperator 的 InferLayout leaf |
| vLLM | backend priority、validation、persistent KV resolver | 主机制已闭合 | 所有 backend-specific support predicates |
| SGLang | default backend tree、prefill/decode split、persistent KV family、page geometry coupling | 主机制已闭合 | 大量 model-specific override leaf，仍是开放集合 |
| FlashInfer | FA2/FA3 dispatch、CTA-Q、decode split geometry、split reduction policy | 主机制已闭合 | 各 specialized profile factory 的全部 leaf |

所以现在可以说：

\[
\boxed{
\text{RQ1 的 mechanism-family 证据链已基本闭合}
}
\]

但仍不能说：

\[
\boxed{
\text{10 个仓库所有 layout 相关 if/else 已逐行穷尽}
}
\]

后者是不准确的，我不会再用这种表述。

---

# 三、RQ1 主证据总表：逐规则而不是逐框架一句话

下面是目前能够直接用于“问题审判”的规则。表中的“忽略”严格指**这个具体 selector/cost function 没有读取该变量**，不是说整个框架完全不知道这个变量。

| 框架 / 规则 | 它到底选择什么 | 实际 proxy / state | 硬约束 | 没进入该规则的性能变量 | fallback / repair | 证据 |
|---|---|---|---|---|---|---|
| CUTLASS `sm100_smem_selector` | UMMA SMEM layout | major、dtype、tile divisibility | MN/K tile multiples of 8；TF32特殊布局 | 多 consumer access frequency、最终 measured latency | explicit expert layout | SW128→64→32→INTER，且注释明确选 largest compatible fileciteturn728file0L1-L2 |
| CUTLASS TMA A/B selector | load vs multicast | static/dynamic cluster、MMA atom threads | cluster divisibility | runtime cluster occupancy、实际 traffic crossover | dynamic cluster 强制 conservative multicast path | fileciteturn756file0L1-L2 |
| CUTLASS `KernelScheduleAuto` | 1SM vs 2SM | static cluster + tile M | 2SM需要cluster-M可被2整除且tile-M适配128 | workload M/N/K、batch/expert size、实测1SM/2SM crossover | 否则1SM；dynamic cluster直接1SM | fileciteturn684file0L1-L2 |
| CUTLASS `StageCountAuto` | mainloop stages | tile A/B footprint + pipeline storage + SMEM carveout | SMEM capacity | short-K、small-M、额外CTA residency收益 | explicit `StageCount<N>` | stage bytes结构计算 fileciteturn729file0L1-L19 |
| CUTLASS accumulator stages | accumulator pipeline depth | tile/TMEM resource structure | TMEM capacity | workload distribution、measured epilogue crossover | cap 4 | 注释直接说 4 stages “works well” fileciteturn757file0L1-L59 |
| Triton `isLayoutAnchor` | 哪些layout不可随意改 | expensive LD/ST、dot、atomic、TMEM等结构角色 | IR合法性 | whole-subgraph measured latency | conversion、rematerialization、hoisting | fileciteturn687file0L1-L2 |
| Triton async-load gate | 是否值得 pipeline load | pointer contiguity、mask alignment、load width | cp.async size legality | loop trip count、实际 latency hiding、当前 register headroom | 保留同步 load | width≥32 bits；源码明确提 register pressure fileciteturn689file0L1-L2 |
| Triton sync-dot rule | 是否 pipeline MMA | `hasSyncDots(loop)` | — | tail ratio、trip count、resource usage | skip MMA pipelining | 源码直接叫 “dirty heuristic” fileciteturn760file0L1-L32 |
| Triton GenericSwizzling | shared swizzle/layout | linear layout、vectorization、lane phase、bank conflict | 可逆/合法linear mappings | precise PRMT/instruction cost | conservative candidate construction | 已分析 bank conflict fileciteturn732file0L1-L2 |
| Triton swizzle tie-break | 等价candidate排序 | lowest basis | — | 实际 PRMT count / latency | 后续lowering | “in the hope that it will avoid PRMTs” fileciteturn734file0L1-L16 |
| Triton generic shared fallback | pipeline-created shared encoding | tensor order | IR compatibility | 2D access structure | generic encoding | 源码明确 “won't be optimal for 2D tensors” fileciteturn759file0L1-L38 |
| Triton ReduceDataDuplication | convert路径 | duplication/CSE opportunity | encoding compatibility | complete runtime latency | distributed→shared→dotOperand | fileciteturn758file0L1-L36 |
| Gluon AutoLayout conflict | ambiguous auto encoding | propagation distance | distance-0 explicit seed不能冲突 | runtime latency | conflict→error；explicit seed | fileciteturn693file0L1-L2 |
| Gluon equal-distance tie | auto encoding | stable hash | candidates inferred legal | 所有performance variable | deterministic choice | fileciteturn693file0L1-L2 |
| Gluon CoalescedLayout | memory-friendly distributed mapping | AxisInfo、warps、threads/warp、shape | 当前实现 `numCTAs==1` | multi-CTA performance | programmer explicit layout | fileciteturn694file0L1-L2 |
| Gluon NVMMA shared default | shared swizzle | contiguous bytes、dtype、shape/CGA | swizzle∈0/32/64/128 | other consumer latency | explicit swizzle | largest compatible，为减少TMA/MMA messages fileciteturn737file0L1-L2 |
| TVM DefaultCUDA | candidate family | fixed tiling grammar/reuse/vector candidate | schedule legality | candidate outside family | custom rules | `SSSRRSRS`, vector `{1,2,3,4,8,16}` 等 fileciteturn722file0L1-L2 |
| TVM TensorCore rules | TensorCore candidate family | predefined intrinsics、reuse/pipeline variants | intrinsic matching | candidate outside family | general CUDA rules | fileciteturn721file0L1-L2 |
| TVM MetaSchedule ranking | generated candidates | **hardware runtime** | compilation/execution success | — | continued search | Builder/Runner实测 fileciteturn719file0L1-L32 |
| TVM RewriteLayout | weight physical layout | consumer access→`SuggestIndexMap` | layout-free；源码预期单 BufferLoad | arbitrary multi-consumer joint objective | cache-read chain传播相同map | fileciteturn697file0L1-L2 |
| TIRx dispatcher | primitive variant | legality predicates + static priority | predicates | simultaneous legal candidates真实latency | 下一个candidate | first successful result立刻返回 fileciteturn726file0L1-L2 |
| TIRx `vec_auto` | copy implementation | gmem↔smem/reg-copy applicability | copy legality | measured vector/scalar crossover | fallback | priority 10 fileciteturn724file0L1-L2 |
| TIRx scalar fallback | copy implementation | all faster variants rejected | valid copy | performance不再优化 | scalar single-thread | priority 0 fileciteturn725file0L1-L2 |
| TIRx MMA | mma.sync variant | operand layout、active lanes、dtype、tile divisibility | instruction exact layout | simultaneous legal variant latency | next registered variant | m16n8k16/8 family fileciteturn770file0L1-L2 |
| Hexcute constraint solver | TV/layout/instruction legal set | algebraic mapping constraints | copy/MMA mapping equality | performance本身 | backtrack | `f∘p⁻¹=g∘q⁻¹` 等 fileciteturn718file0L1-L2 |
| Hexcute remaining strides | unresolved shared strides | local heuristic after unification | all constraints satisfied | measured latency | backtracking before heuristic | 源码文档直接说明 fileciteturn718file0L1-L2 |
| Hexcute latency model | candidate ranking | instruction count×microbench CPI | candidate已合法 | bank-conflict cost、barrier/wait、imperfect overlap、address arithmetic | 后续 bank pass | fileciteturn761file0L1-L2 |
| Hexcute bank repair | shared swizzle | aggregate conflict count across consumers | WGMMA layout可锁定 | complete latency impact | choose min-conflict swizzle | fileciteturn762file0L1-L2 |
| Hexcute TMA repair | transfer instruction | TMA layout feasibility | TMA不能执行 | “TMA合法但较慢”的情况 | TMA→cp.async + barrier/mask修复 | fileciteturn763file0L1-L2 |
| TileLang pipeline ordering | inference input structure | materialized WS/2SM/software pipeline | pipeline先合法化 | alternative layout×pipeline joint choice | pipeline固定后 inference | fileciteturn703file0L1-L2 |
| TileLang thread topology | layout inference | constant thread extent | **必须compile-time constant** | runtime-varying topology | hard failure，不是performance fallback | fileciteturn715file0L1-L2 |
| TileLang register-count | complete layout attempt | spill bytes + register slots | complete inferred assignment | global memory coalescing/issue等 | optional io-aware | fileciteturn738file0L1-L2 |
| TileLang io-aware | complete layout attempt | max(bandwidth bytes, issue-equivalent bytes) + regs | modelled global access | shared-side staging等未建模成本 | explicit layout / fallback | fileciteturn738file0L1-L2 |
| TileLang repair | unresolved reducer component | replicated fallback | reserved dst | replication性能代价 | `FullyReplicated`-style universal readable fallback | fileciteturn714file0L1-L2 |
| vLLM backend priority | attention backend | MLA、arch、heads、dtype、head size、mask-ish flags | `validate_configuration` | **live batch size** | next legal backend | TokenSpeed注释直接暴露 bs crossover fileciteturn740file0L1-L2 |
| vLLM backend validation | legal backend set | capability/config support | backend predicates | legal backend之间runtime latency | skip invalid backend | priority后做support filtering |
| vLLM persistent resolver | one whole-model KV layout | support intersection + first-choice votes | all active backend都支持 | prefill/decode traffic weights、dynamic request distribution | explicit env、connector preference、block-compact restriction | fileciteturn704file0L1-L2 fileciteturn705file0L1-L2 |
| SGLang default backend | MHA/MLA backend | GPU family、attention arch、spec mode、asymmetric KV、sink、head count | backend feature support | live B、Q/KV-length distribution、load/concurrency | FlashInfer→Triton等tree fallback | fileciteturn742file0L1-L2 |
| SGLang phase split | backend scope | prefill vs decode | configuration | phase内部更细 workload state | separate fields fallback base backend | fileciteturn741file0L1-L2 |
| SGLang KV representation | NHD/HND/vectorized5D | platform/backend/env | AITER/layout divisibility等 | workload frequency/mix | NHD default、specialized native representation | fileciteturn744file0L1-L16 |
| SGLang AITER page geometry | page size | HIP + vectorized5D | representation compatibility | dynamic sequence mix | user pin page size | default 64 in this configuration fileciteturn767file1L23-L46 |
| FlashInfer FA2/FA3 | backend | architecture、pos enc、mask、dtype、head dims等 | backend support | batch/length performance crossover | FA2 fallback | fileciteturn766file0L1-L20 |
| FlashInfer CTA-Q | prefill Q tile | **avg packed-Q**、QK dim、VO dim、KV bytes、arch/SMEM | register/SMEM feasibility | packed-Q variance、tail/higher moments | finite 16/32/64/128 family | fileciteturn748file0L1-L2 fileciteturn746file1L20-L38 |
| FlashInfer CTA-Q SMEM probe | CTA16 vs64 | approximate SMEM formula | device SMEM | FP4 SF bytes、FP8 repack staging明确没计 | CTA64 fallback | 源码自己写“approximation” fileciteturn747file0L1-L2 |
| FlashInfer decode split geometry | split/chunk | occupancy + **每request page counts** + GQA geometry | grid/resource legality | —不能再称忽略KV distribution | no split / binary searched split | fileciteturn751file0L1-L19 |
| FlashInfer reduction mode | cluster/separate/inline | topology/support/residency | exact support由caller检查 | measured shape-specific crossover | cluster→separate→inline | 源码明确避免 measured crossover table fileciteturn764file0L1-L24 |

这张表有一个重要结果：RQ1 并不是“所有框架都使用一个简单 heuristic”。

实际存在至少四类 selector：

\[
\boxed{
\text{static rule}
,\quad
\text{legality+priority}
,\quad
\text{analytic cost}
,\quad
\text{hardware measurement}
}
\]

所以科研问题必须能够同时容纳这四类，而不是批判某一个 `if`。

---

# 四、证据链凝炼后，RQ1 应该这样定义

完整状态记为：

\[
x=
(
x_\text{shape},
x_\text{ragged},
x_\text{layout},
x_\text{reuse},
x_\text{pipeline},
x_\text{resource},
x_\text{runtime},
x_\text{hardware},
\dots
).
\]

候选集：

\[
\mathcal C(x).
\]

真实最优：

\[
c^*(x)
=
\arg\min_{c\in\mathcal C(x)}
T(c,x).
\]

框架没有直接使用 \(x\)，而是构造某种 decision state：

\[
p=P(x)
\]

再通过：

\[
\hat c(x)=\pi(P(x))
\]

选 candidate。

于是 RQ1 可以精确写成：

\[
\boxed{
\textbf{RQ1: Decision-State / Proxy Sufficiency}
}
\]

> 在现代 Attention/GQA/MLA/MoE 的 layout、instruction、pipeline、backend 与 persistent representation 相关决策中，现有 selector 所观察的状态 \(P(x)\) 是否足以区分性能最优候选？如果不充分，是否存在一个小规模、可在线获得的状态扩展 \(z(x)\)，使 \(P'(x)=[P(x),z(x)]\) 显著降低 selection regret，而不退化成全候选在线实测？

这个问题比“heuristics 准不准”强得多，因为它允许被严格否证。

---

# 五、RQ1 的第一条定罪标准：Proxy Collision

最关键的不是某个规则偶尔慢，而是：

\[
P(x_1)=P(x_2)
\]

但是：

\[
c^*(x_1)\neq c^*(x_2).
\]

定义为：

\[
\boxed{
\text{Proxy Collision}
}
\]

如果找不到这种 collision，说明现有 proxy 可能已经是 sufficient statistic。

这使每个框架都可以构造非常具体的实验。

---

# 六、CUTLASS 的 RQ1 审判

CUTLASS 有三类不同 proxy。

## 6.1 SMEM swizzle：compatibility-order proxy

`sm100_smem_selector` 不是搜索所有 swizzle 再测，而是：

\[
SW128
\succ
SW64
\succ
SW32
\succ
INTER
\]

取最大的 compatible layout。源码注释直接说“returns the largest UMMA::Layout that fits”。fileciteturn728file0L1-L2

所以：

\[
P_\text{smem}
=
(dtype,major,BLK_{MN},BLK_K).
\]

真正需要验证的不是：

> SW128 是否有 bank conflict。

而是：

> 在两个 workload 有完全相同 tile compatibility，但 shared tensor 的其它 consumer/access-frequency 不同时，最大的 compatible swizzle 是否仍然保持最优？

构造：

\[
P(x_1)=P(x_2)
\]

但让：

\[
frequency_\text{consumer A}
\neq
frequency_\text{consumer B}.
\]

若 candidate swizzle winner 翻转：

\[
L^*(x_1)\neq L^*(x_2),
\]

则证明 compatibility state 不充分。

---

## 6.2 `StageCountAuto`：capacity proxy

核心近似：

\[
S_\text{auto}
=
\left\lfloor
\frac{SMEM_\text{capacity}-carveout}
{bytes(A_\text{tile})+bytes(B_\text{tile})+bytes(pipeline)}
\right\rfloor.
\]

fileciteturn729file0L1-L19

所以两个 workload 只要 tile/type 一样：

\[
P(x_1)=P(x_2)
\]

即使：

\[
K_1\ll K_2
\]

或 MoE 中：

\[
M_{e,1}\ll M_{e,2},
\]

自动 stage 决策仍然一样。

因此直接假设：

\[
H_\text{CUTLASS-stage}:
\exists x_1,x_2:
P(x_1)=P(x_2)
\land
stage^*(x_1)\neq stage^*(x_2).
\]

最重要的现代触发：

- MoE 小 expert；
- decode/small-M GEMM；
- short-K projections；
- grouped GEMM 中 expert size 高度不均衡。

---

## 6.3 1SM/2SM：structural eligibility → policy

静态 cluster 时：

\[
cluster_M\bmod2=0
\land
tile_M\bmod128=0
\Rightarrow 2SM;
\]

否则 1SM；dynamic cluster 一律不假设2SM。fileciteturn684file0L1-L2

这是非常适合 RQ1 的例子，因为它首先是 legality/resource structural prior，然后隐含了：

\[
\text{eligible 2SM}
\Rightarrow
\text{2SM is preferred}.
\]

可以固定相同 tile/cluster：

\[
P(x_1)=P(x_2),
\]

只变化 problem extent / number of tiles。

如果 small workload 上 1SM 更快、大 workload 上 2SM 更快，则构成 collision。

CUTLASS 的 fallback 是 expert override：

\[
KernelSchedule1Sm/2Sm,\quad StageCount<N>,\quad explicit layout.
\]

所以这里非常明确：

\[
\boxed{
\text{manual override 是 escape hatch，不是自动性能 repair}
}
\]

---

# 七、Triton classic 的 RQ1 审判

Triton 需要拆成三种不同证据。

## 7.1 async-load eligibility

当前规则：

\[
width\ge32\text{ bits}
\Rightarrow
canBeConvertedToAsyncLoad.
\]

width 来自 pointer contiguity 和 mask alignment。源码解释，小 load 未必值得 pipeline，还可能增加 register pressure。fileciteturn689file0L1-L2

这里的 proxy：

\[
P=(accessWidth).
\]

没有进入这个 gate 的变量包括：

\[
loopTripCount,\quad
consumerDistance,\quad
currentRegisterPressure,\quad
occupancy.
\]

注意 register pressure 虽然被注释作为决策理由，却没有作为 runtime/IR numeric state 被该函数读取。

因此非常干净的 collision：

\[
width(x_1)=width(x_2)=32
\]

但一个：

\[
tripCount=2
\]

另一个：

\[
tripCount=128.
\]

比较 pipelined vs unpipelined。

---

## 7.2 sync-dot “dirty heuristic”

这条更强。

当前 source rule：

\[
hasSyncDots(loop)=true
\Rightarrow
skip\ MMA\ pipelining.
\]

源码自己称其为：

> dirty heuristic for performance drops. fileciteturn760file0L1-L32

所以这是 B 级直接证据。

碰撞实验：

\[
P(x_1)=P(x_2)=hasSyncDots=true
\]

但变化：

\[
tripCount,\ tailFraction,\ registerPressure.
\]

检验：

\[
pipeline^*(x_1)\neq pipeline^*(x_2)?
\]

现代 Attention 中非常自然，因为 QK/PV 的 tiled reduction loop 和 masked tail 会产生这样的差异。

---

## 7.3 layout conflict + conversion repair

`RemoveLayoutConversions` 先把 expensive memory op、dot、atomic 等标记为 layout anchor，再传播 layout；如果目标冲突，会尝试：

- conversion；
- backward rematerialization；
- hoisting；
- conditional conversion movement。

fileciteturn687file0L1-L2

这里体现的不是“编译器没意识到多目标冲突”。

相反：

\[
\boxed{
Triton 已明确意识到
L_\text{memory}
\neq
L_\text{MMA}.
}
\]

RQ1真正问的是：

> 用 anchor structure + conversion/rematerialization profitability 判断，是否已经包含选择 whole-subgraph 最优 representation 所需的信息？

后续 RQ4 才会问：

> 为什么要“先局部选，再 conversion repair”，而不是联合优化？

二者不能混在一起。

---

# 八、Gluon 的 RQ1 审判

这里证据非常“纯”。

`LayoutInfo`：

\[
(\text{encoding},distance).
\]

Join/Split/Reshape/Transpose 之类 fuzzy inference 每经过一次：

\[
distance\leftarrow distance+1.
\]

冲突时：

\[
d_a<d_b
\Rightarrow
a\ wins.
\]

相同非零 distance：

\[
d_a=d_b>0
\]

最终用 stable hash 决定；两个不同的 distance-0 explicit seeds 冲突则报错。fileciteturn693file0L1-L2

这里的 `distance` 明确是：

\[
\boxed{\text{inference ambiguity / proximity}}
\]

不是：

\[
\boxed{\text{latency prediction}}.
\]

因此实验非常简单：

找到两个合法 explicit layout：

\[
L_\text{near},L_\text{far}.
\]

让 AutoLayout propagation 会选择 `near`。

然后实际 benchmark：

\[
T(L_\text{near}),T(L_\text{far}).
\]

如果不同 workload 下 winner 发生翻转，就说明：

\[
inference\ distance
\]

不能作为 performance sufficient statistic。

但要注意 Gluon 的设计允许专家直接写 `BlockedLayout` / `DistributedLinearLayout` / NVMMA layout，因此：

\[
\boxed{
AutoLayout 是 convenience path，而不是声称全局性能最优
}
\]

这会降低其“有罪程度”。

---

# 九、TVM：RQ1 的反方证人

TVM 对这个问题非常重要，因为它证明：

\[
\text{structural prior}
\]

和：

\[
\text{performance ranking}
\]

完全可以分层。

它的 candidate space 很强地限制在类似：

\[
SSSRRSRS
\]

和有限 vector/reuse/pipeline families 内。fileciteturn722file0L1-L2

TensorCore space 同样使用固定 intrinsic family 和有限 software-pipeline choices。fileciteturn721file0L1-L2

但是最终可以：

\[
Builder
\rightarrow
Runner
\rightarrow
T_\text{hardware}.
\]

fileciteturn719file0L1-L32

因此在 RQ1 中它应标成：

\[
\boxed{N=\text{negative control}}
\]

不是有罪框架。

但 TVM 又留下另一个很强的新证据：

`RewriteLayout` 源码明确写：

> buffer is expected to be read by only one BufferLoad

并从 consumer access 推导 `IndexMap`，然后沿 cache-read chain 传播。fileciteturn697file0L1-L2

这不是 RQ1。

它明显属于后续：

\[
\boxed{
\text{single-consumer-derived representation}
\quad vs\quad
\text{multi-consumer joint layout}
}
\]

所以该证据保留，但暂不审判。

---

# 十、TIRx 的 RQ1 审判

TIRx 是最标准的：

\[
\boxed{
legality
+
fixed\ priority
}
\]

系统。

dispatcher 明确：

\[
cases=
sort(-priority,variant)
\]

逐个检查 predicate；第一个成功返回 `PrimFunc` 的 candidate 直接返回。fileciteturn726file0L1-L2 fileciteturn727file0L1-L2

例如 copy：

\[
vec\_auto:priority=10
\]

fileciteturn724file0L1-L2

scalar fallback：

\[
priority=0,
\]

而且 warning 直接说：

> all faster variants rejected. fileciteturn725file0L1-L2

这隐含的 assumption 不是：

> vector candidate永远合法。

而是：

\[
\boxed{
\text{一旦 higher-priority candidate 合法，
它就是足够好的选择。}
}
\]

因此审判条件非常严格：

选：

\[
c_1,c_2
\]

使：

\[
legal(c_1,x)=legal(c_2,x)=true
\]

对两个 workload 都成立。

若：

\[
T(c_1,x_1)<T(c_2,x_1)
\]

但：

\[
T(c_1,x_2)>T(c_2,x_2),
\]

则固定 priority 必然出现 selection regret。

这比拿 scalar fallback 和 vector path 比更科学，因为 scalar fallback 本来就是 correctness repair。

---

# 十一、Hexcute 的 RQ1 审判

Hexcute 需要明确拆成：

\[
\text{feasibility inference}
\]

和：

\[
\text{performance ranking}.
\]

它先通过 copy：

\[
f\circ p^{-1}
=
g\circ q^{-1}
\]

和 MMA M/N/K consistency 等约束构造合法 TV layout / instruction combinations；多 consumer 的 memory constraints 会逐步统一，冲突就 backtrack，最后仍有自由 stride 才使用 heuristic。fileciteturn718file0L1-L2

所以约束求解本身不是 RQ1 的 target。

RQ1 target 是：

\[
\hat T
=
N_\text{inst}
\times CPI.
\]

源码还直接声明：

- copy/MMA perfect overlap；
- pipeline fully overlapped；
- 不计 `cp_async_wait_group`；
- 不计 `mbarrier`；
- bank conflict 不在这个 cost model 内；
- address arithmetic近似零成本。

fileciteturn761file0L1-L2

然后 bank conflict 又有独立优化：

\[
L
\rightarrow
ResolveBankConflict(L).
\]

它会对 shared tensor 的多个 consumer 联合统计冲突，而不是只看一个 copy。fileciteturn762file0L1-L2

因此 Hexcute 的假设应精确写成：

\[
\boxed{
\text{约束已经把候选压缩到足够规则的区域，
使 instruction-CPI 模型 + staged bank repair 足以保持排名。}
}
\]

不是：

> CPI 能精确预测 cycle 数。

真正需要比较的是 rank：

\[
rank_{\hat T}(c_i)
\quad vs\quad
rank_T(c_i)
\]

和：

\[
Top1Regret.
\]

TMA→cp.async fallback 则只处理 layout 不满足 TMA 时的 support failure。fileciteturn763file0L1-L2

它不能反驳：

\[
TMA\ legal\ but\ slower.
\]

---

# 十二、TileLang：RQ1 的另一个A级样本

TileLang 的证据越来越强，因为它自己已经构造了两个竞争 cost model。

默认：

\[
C_R
=
(\text{spill bytes},
\text{register slots}).
\]

可选 IO-aware：

\[
C_{IO}
=
\sum_s
\max(
bytes^\text{bandwidth}_s,
bytes^\text{issue}_s
)
\]

再以 register count tie-break。fileciteturn738file0L1-L2

它不是“某个 op 的局部 cost”，而是在 connected component 上比较 complete layout attempts。

更强的是 verification harness 自己说：

> 两套模型出现 disagreement 的 case 就是 calibration corpus；每一次 disagreement 都是一个可以 hardware benchmark 的 claim。fileciteturn739file0L1-L2

其中：

- `broadcast_read`：register-count 保留 thread-collapsed legacy pathology；
- `transposed_store`：load/store preference冲突；
- `mixed_dtype_chain`：fp16/fp32 vector split冲突；
- `reduce_broadcast`：softmax-shaped reduce+broadcast，两模型却一致；
- `shared_staging`：shared-side copy 不进入 IO-aware cost。

fileciteturn739file0L1-L2

这几乎已经把 RQ1 的 experimental philosophy 写在框架源码旁边：

\[
\boxed{
proxy\ disagreement
\rightarrow
hardware\ adjudication.
}
\]

而 `reduce_broadcast` 又提供必要的反例：

\[
\boxed{
复杂 workload
\not\Rightarrow
cheap proxy 一定失效.
}
\]

所以 TileLang 应同时作为 RQ1 的支持证据和反方证据。

---

# 十三、vLLM：目前最强的直接“proxy collision”源码线索

`_get_backend_priorities()` 输入只有：

\[
useMLA,
arch,
numHeads,
KVdtype,
nonCausal,
headSize,
mmPrefix.
\]

没有 batch size。

但 SM100 MLA 代码旁直接写：

> TokenSpeed MLA wins past `bs≈8`, regresses at `bs≤2`.

fileciteturn740file0L1-L2

同时静态顺序仍是：

\[
FlashInferMLA
\succ
TokenSpeedMLA
\succ
CUTLASSMLA
\succ\dots
\]

这已经接近源码直接给出的：

\[
P(x_1)=P(x_2)
\]

但：

\[
c^*(x_1)\neq c^*(x_2).
\]

例如：

\[
x_1:B=1,
\qquad
x_2:B=16
\]

其它 selector inputs 全固定。

因此它是 A 级证据。

但依然必须重新 benchmark，因为代码注释不是实验真理。kernel 版本可能已经变化。

如果 2026 当前版本测出来：

\[
T_{FI}(B)<T_{TS}(B)
\quad
\forall B,
\]

那么这条看似最强的源码证据会被否证。

这正是我们要的科研问题。

---

## vLLM persistent layout 要移出 RQ1 主定罪链

whole-model resolver 的逻辑非常明确：

1. 每个 backend 给 supported layouts，按 preference 排序；
2. 求 intersection；
3. 若排序不一致，最多 backend 放第一位的 layout优先；
4. mixed HNC specs时只保留 block-compact layout；
5. connector preference只在 compatible 时采用；
6. 最终 whole model 使用一个 layout。

fileciteturn704file0L1-L2 fileciteturn705file0L1-L2

这确实没有 runtime traffic-weighted objective。

但它面对的是：

\[
L_\text{persistent}
\]

而不是 kernel-local candidate。

所以真正核心问题更接近：

\[
\boxed{
\text{stable common representation}
\quad vs\quad
\text{consumer specialization + conversion/replication}
}
\]

这将形成后续非常强的 RQ，不应该硬塞入 RQ1。

---

# 十四、SGLang：静态 feature tree，但已经主动增加过 state granularity

SGLang 默认 selector 直接写：

> Auto select the fastest attention backend.

实际 MHA tree 会读：

- Hopper / SM100 / HIP；
- speculative conditions；
- asymmetric KV；
- attention sinks；
- FlashInfer availability。

MLA 又读：

- architecture；
- KV-head count；
- TP-related model state。

fileciteturn742file0L1-L2

但该 selector 本身没有读取：

\[
live\ batch,\qquad
\{Q_i,KV_i\},\qquad
runtime\ concurrency.
\]

所以其 RQ1 假设是：

\[
H_{SGLang}:
\]

> hardware/model/feature state 已足以稳定决定 backend winner，batch/length distribution 不需要进入默认 backend selector。

但 SGLang 自己已经给出一个有趣的反例机制：

\[
backend_\text{prefill}
\neq
backend_\text{decode}.
\]

fileciteturn741file0L1-L2

这说明框架已经认为：

\[
phase
\]

是重要 decision state，因此把一个 coarse selector：

\[
P_0=(model,arch)
\]

扩张成：

\[
P_1=(model,arch,phase).
\]

这正支持 RQ1 更深层的问题：

> decision state 应扩张到哪里停止？

Persistent representation 也不是单一 NHD。SGLang 有：

- NHD default；
- HND；
- AITER `vectorized_5d`。

HND服务 per-KV-head sparse page tables，vectorized5D直接为 AITER 的消费模式重排 K/V。fileciteturn744file0L1-L16

而 AITER vectorized representation 还会改变 page-size default；相关测试专门指出如果不显式 pin page size，测得 geometry 会漂移到 64。fileciteturn767file1L23-L46

这里已经开始指向后续“representation × page geometry × backend”的联合决策问题。

---

# 十五、FlashInfer：现在可以精确指出两个真正的 RQ1 proxy

## 15.1 CTA-Q：mean compression

scheduler 计算：

\[
\bar Q
=
\frac{1}{B}\sum_i packedQ_i
\]

然后：

\[
CTA_Q
=
f(
\bar Q,
D_{VO},
D_{QK},
bytes_{KV},
arch
).
\]

fileciteturn746file1L20-L38

具体规则包括：

\[
D_{VO}\ge512:
\begin{cases}
16,&\bar Q\le32\\
32,&\bar Q>32
\end{cases}
\]

\[
D_{QK}\ge512,D_{VO}\le256
\Rightarrow16
\]

以及普通区域的：

\[
\bar Q>64,D_{VO}<256
\Rightarrow128
\]

等。fileciteturn748file0L1-L2

因此这里存在真正的信息压缩：

\[
\{Q_1,\dots,Q_B\}
\longrightarrow
\bar Q.
\]

构造：

\[
x_1=[32,32,32,32]
\]

\[
x_2=[1,1,1,125].
\]

两者：

\[
\bar Q=32.
\]

selector state 完全相同。

但 raggedness 差异极大。

如果：

\[
CTA_Q^*(x_1)
\neq
CTA_Q^*(x_2)
\]

即构成 textbook proxy collision。

---

## 15.2 CTA16 SMEM probe 本身还是 approximation

短-Q branch 会估计：

\[
SMEM
\approx
SMEM_Q
+
SMEM_{K+V}.
\]

源码自己明确写：

- FP4 scale-factor bytes 没计；
- FP8 repack staging 没计；
- 这是 approximation。

fileciteturn747file0L1-L2

这里可以单独验证：

\[
\hat{SMEM}
\]

与实际 kernel resource / performance 的 ranking 是否一致。

但这主要是一个很窄的 leaf hypothesis，不应该独立成为 RQ。

---

## 15.3 split-KV reduction mode：结构排序替代 crossover table

当多个 reduction mechanism 都合法时，FMHA：

\[
cluster\_smem
\succ
gmem\_separate
\succ
gmem\_inline.
\]

源码明确说明：

> avoids shape-specific measured crossover tables. fileciteturn764file0L1-L24

这已经直接把研究 trade-off 写出来了：

\[
\boxed{
structural\ robustness
\quad vs\quad
shape\text{-}specific\ adaptation.
}
\]

所以实验必须限制在：

\[
cluster,\ separate,\ inline
\]

多个模式同时合法的 intersection region。

否则只是 support difference，不是 RQ1。

---

# 十六、现代 Attention/GQA/MLA/MoE 的触发条件现在可以精确下来

不能再笼统说“现代模型更动态”。

### Attention / ragged prefill

核心隐藏状态：

\[
\mathcal Q=
\{Q_i\}_{i=1}^{B},
\qquad
\mathcal K=
\{KV_i\}_{i=1}^{B}.
\]

需要区分：

\[
mean,\quad
variance,\quad
max,\quad
tail,\quad
histogram.
\]

FlashInfer CTA-Q 已直接给出：

\[
\mathcal Q\rightarrow mean(\mathcal Q)
\]

的具体压缩点。

### GQA

\[
g=
\frac{H_Q}{H_{KV}}.
\]

影响：

- KV reuse；
- CTA parallelism；
- per-KV-head work；
- shared-memory amortization；
- split-KV grid。

但注意 FlashInfer decode 已经把 `GROUP_SIZE` 纳入 scheduler，因此不能拿 GQA ratio 作为该具体规则“未观察变量”。

需要寻找：

\[
\boxed{
\text{它观察了 }g，
\text{但没有观察 }g\times raggedness/resource
\text{ 的哪些 interaction}
}
\]

才是有效 RQ1。

### MLA

状态至少包括：

\[
(
B,
qLen,
KVLen,
D_{CKV},
D_{KPE},
dtype_{KV},
splitKV,
architecture
).
\]

vLLM 的 batch crossover 注释是这里目前最强的直接 evidence。

### MoE

设 expert token histogram：

\[
\mathbf M=
(M_1,\ldots,M_E).
\]

同样总 token count：

\[
\sum_iM_i=N
\]

可有：

uniform：

\[
M_i\approx N/E
\]

和 heavy skew：

\[
M_1\gg M_2\gg\dots.
\]

如果 backend/tile/stage selector只看到 model/shape class 而没有看到：

\[
max(M_i),
Var(M_i),
activeExperts
\]

就可以构造：

\[
P(x_1)=P(x_2)
\]

但 expert-level kernel optimum不同的 collision。

这正是 CUTLASS StageCount/1SM-2SM 等结构规则最有价值的 MoE testbed。

---

# 十七、把 RQ1 收缩成四条必须分别通过的科学假设

现在不应该再把 RQ1 看成一句话。

## H1：Proxy Collision Exists

\[
\boxed{
\exists x_1,x_2:
P(x_1)=P(x_2)
\land
c^*(x_1)\ne c^*(x_2)
}
\]

没有 H1，就没有 RQ1。

---

## H2：Collision Causes Material Regret

定义：

\[
R(x)
=
\frac{T(\hat c(x),x)}
{T(c^*(x),x)}
-1.
\]

要求不是偶发 0.5% flip，而是：

\[
\Pr_{x\sim\mathcal W}
[
R(x)>\epsilon
]
>\delta.
\]

也就是说问题必须在真实 workload distribution 上有频率和幅度。

---

## H3：存在低维 Missing State

找到：

\[
z(x)
\]

使：

\[
P'(x)
=
[P(x),z(x)]
\]

显著降低：

\[
E[R_{P'}]
<
E[R_P].
\]

并且：

\[
dim(z)\ll dim(x).
\]

否则答案只是：

> “把所有信息都测一遍。”

这不构成有价值的新优化方法。

最值得优先检验的 \(z\)：

Attention：

\[
B,\quad
Var(Q),\quad
Q_{max},\quad
KV_{p95}.
\]

MLA：

\[
B,\quad
qLen,\quad
KVLen,\quad
splitKV.
\]

MoE：

\[
max(M_e),\quad
CV(M_e),\quad
activeExperts.
\]

这些只是候选，实验必须允许它们全部被否决。

---

## H4：Adaptive Fidelity 比单一 Fidelity 更优

证据链现在已经形成：

\[
\text{TIRx/CUTLASS/SGLang}
\rightarrow
cheap structural rule
\]

\[
\text{Hexcute/TileLang}
\rightarrow
analytic proxy
\]

\[
\text{TVM}
\rightarrow
measurement.
\]

因此出现一个很自然但暂时仍属于 RQ1 的子假设：

存在：

\[
u(x)=
\text{selector uncertainty}
\]

使得：

\[
u(x)<\tau
\Rightarrow
cheap\ rule
\]

\[
u(x)\ge\tau
\Rightarrow
richer\ model/search/measurement.
\]

也就是说不是问：

> 静态 rule 还是 autotune 哪个好？

而是：

\[
\boxed{
\text{什么时候值得升级 fidelity？}
}
\]

如果这一条以后在多个 RQ 中独立重复，我会再把它升格成新的 RQ；目前不提前制造新编号。

---

# 十八、RQ1 的实验审判矩阵现在可以闭成这样

| Case | 必须固定的 observable proxy | 只改变 hidden state | Candidate oracle | H1 成立标准 |
|---|---|---|---|---|
| vLLM SM100 MLA | arch/R1 dims/FP8/head config等 | batch 1→32 | FI-MLA / TokenSpeed / others | winner随B翻转 |
| FlashInfer CTA-Q | **same avg packed-Q**、same QK/VO/dtype | variance/max/tail | 所有合法 CTA-Q | 同 mean 下 winner翻转 |
| FlashInfer reduction | same legal mode set/topology | Q/KV/split workload | cluster/separate/inline | fixed structural order翻转 |
| CUTLASS StageCount | same tile/type/carveout | K或expert M | explicit stages | optimal stage翻转 |
| CUTLASS 1SM/2SM | same eligible tile/cluster | total tile count/workload | explicit 1SM vs2SM | winner翻转 |
| CUTLASS swizzle | same compatibility class | consumer access mixture | legal SW128/64/32/... | largest-compatible不是恒优 |
| Triton async | same access width | loop trip/resource pressure | async on/off | threshold相同但winner翻转 |
| Triton sync-dot | `hasSyncDots=true` | tail/trip count | pipeline choices | skip vs pipeline翻转 |
| Triton swizzle | same algebraic legality | access/instruction mix | legal swizzles | residual tie-break翻转 |
| Gluon AutoLayout | same inference-distance ordering | workload shape/use count | explicit candidate layouts | farther seed layout更快 |
| TIRx | same predicates all true | workload/resource context | all legal variants | priority ordering翻转 |
| Hexcute | same legal candidate set | barrier/overlap/bank patterns | hardware-measured candidates | analytic rank inversion |
| TileLang | same program/component | alternative cost policy/layout | all enumerated attempts | default proxy winner显著较慢 |
| SGLang | same model/arch/spec/features | B/Q/KV distribution | all supported backends | default backend winner翻转 |
| MoE cross-framework | same GPU/model/quant/total tokens | routing histogram | backend/tile/stage variants | same proxy state、不同winner |

这张矩阵比“扫 shape 看性能”严格得多。

核心要求始终是：

\[
\boxed{
\text{selector observable state 必须保持相同}
}
\]

否则就无法证明 proxy insufficiency。

---

# 十九、RQ1 还需要三个 negative-control 组

为了防止研究变成“专门寻找 heuristic 输的 case”，必须主动设计反证。

第一是 TVM measurement：

\[
\text{measured candidate ranking}
\]

应显著减少 RQ1 型 selection regret。若它并没有，说明问题可能主要不是 proxy，而是 candidate space。

第二是 TileLang `reduce_broadcast`：

softmax-shaped component 中两套 proxy 已经一致。fileciteturn739file0L1-L2

它可以检验：

\[
\text{某些结构是否天然具有稳定 layout optimum}.
\]

第三是 FlashInfer decode split planning。

因为它已经使用：

\[
\{numPages_i\}
\]

而不是 mean-only state。fileciteturn751file0L1-L19

可以对比：

\[
\text{distribution-aware selector}
\]

与：

\[
\text{mean-compressed selector}.
\]

如果前者的 regret 明显更低，会成为 RQ1 非常干净的同框架对照。

---

# 二十、什么结果会“判 RQ1 无罪”

RQ必须可以失败。

如果实验发现：

\[
P(x_1)=P(x_2)
\]

即使 hidden state差异很大，仍然：

\[
c^*(x_1)=c^*(x_2)
\]

几乎总成立，则 H1 不成立。

或者存在 winner flip，但：

\[
R(x)\approx0
\]

始终是 1–2% 内的小差异，则 H2 不成立。

或者要降低 regret，必须加入几十个复杂 runtime features：

\[
dim(z)\approx dim(x),
\]

则 H3 不成立。

或者 uncertainty estimator 与 regret 没相关性，导致几乎所有 case 都得升级 measurement，则 H4 不成立。

这些都应允许论文结论变成：

> 现代 GPU layout/config selection 的最优区域比预期更低维，现有 structural priors 已经接近 decision-sufficient。

这仍然是有价值的结果。

---

# 二十一、什么结果才足以“判 RQ1 有罪”

我现在会把标准提高到四个条件必须同时出现：

\[
\boxed{
P(x_1)=P(x_2)
}
\]

同时：

\[
\boxed{
c^*(x_1)\neq c^*(x_2)
}
\]

并且：

\[
\boxed{
R_P(x)\text{ 在现代 workload 上有系统性幅度}
}
\]

最后：

\[
\boxed{
P'(x)=[P(x),z(x)]
\text{ 用很小的 }z
\text{ 就能显著降低 regret}
}
\]

也就是完整的科学证据链必须是：

\[
\boxed{
CurrentRule
\rightarrow
Proxy
\rightarrow
Collision
\rightarrow
MaterialRegret
\rightarrow
MissingState
\rightarrow
CompactAugmentation
\rightarrow
RegretCollapse
}
\]

只有做到这一步，才不是简单的 heuristic tuning。

---

# 二十二、新补出的证据正在形成哪些后续问题

我现在还不正式审判它们，但证据已经足以登记为后续 RQ 簇。

| 后续证据簇 | 典型源码证据 | 将来真正要问的问题 |
|---|---|---|
| Persistent representation stability vs specialization | vLLM whole-model KV layout；SGLang backend-native NHD/HND/5D | 一个稳定公共表示何时应该让位于 per-consumer representation？ |
| Compromise vs convert vs replicate | Triton conversion/remat；TileLang replicated fallback；vLLM common-layout intersection | 多consumer冲突时应该 compromise、转换还是复制？ |
| Candidate-space sufficiency | TVM `SSSRRSRS`；TIRx registered variants；FlashInfer finite CTA family；CUTLASS builders | 最优候选是否根本不在当前搜索空间？ |
| Layout×instruction×pipeline separability | CUTLASS共构造；Triton多pass+repair；Hexcute固定pipeline后合成；TileLang pipeline-before-layout | 哪些decision必须联合，哪些可以分阶段？ |
| Static vs runtime adaptation | SGLang/vLLM setup-time；FlashInfer per-plan workload state | layout/backend decision应该在compile/setup/request哪个时间尺度重做？ |
| Multi-consumer physical layout | TVM RewriteLayout single-read assumption；Hexcute multi-consumer unification | 一个physical layout能否由单anchor推导，还是需要joint consumer objective？ |
| Data/scale/metadata co-layout | CUTLASS block scales、FlashInfer FP4 scale、vLLM quant packing | 低精度tensor是否必须把 data/scale/metadata 当作联合layout对象？ |

其中这一轮新增得最清楚的是：

\[
\boxed{
\textbf{Single-anchor layout derivation vs multi-consumer joint layout}
}
\]

因为 TVM `RewriteLayout` 的源码已经明确写出“expected to be read by only one BufferLoad”，而 Hexcute恰好提供了反方向机制：一个 shared tensor 的多个 copy consumer 被统一考虑。fileciteturn697file0L1-L2 fileciteturn718file0L1-L2

这很可能最终独立形成一个 RQ，而不是简单并入“compromise vs convert vs replicate”。

但仍然先不编号，避免在 RQ1 尚未实证闭环之前又堆很多问题。

---

# 二十三、现在 RQ1 到底算不算“彻底完成”？

需要区分两个“完成”。

源码证据层：

\[
\boxed{
\textbf{RQ1 mechanism-family source chain：现在可以认为闭合。}
}
\]

因为我们已经覆盖了四种不同 selector fidelity：

\[
\text{rule}
\rightarrow
\text{priority}
\rightarrow
\text{analytic model}
\rightarrow
\text{measurement}
\]

并在 10 个框架中逐个找到了具体函数、proxy、硬约束、repair 和现代 workload trigger。

特别是现在已经有：

- vLLM：源码直接暴露未进入 selector 的 batch crossover；
- TileLang：框架自己维护 proxy disagreement calibration corpus；
- Triton：源码明确标注 “dirty heuristic”；
- FlashInfer：源码明确选择 structural order 来避免 measured crossover table；
- Hexcute：源码明确列出 analytic model 的省略项；
- CUTLASS：源码明确存在结构型 stage/swizzle/1SM-2SM policy；
- TVM：硬件measurement作为反方对照。

但实证判决层：

\[
\boxed{
\textbf{RQ1 尚未完成。}
}
\]

因为目前还缺：

\[
\boxed{
\text{同 proxy state 的 controlled hardware crossover}
}
\]

以及：

\[
\boxed{
\text{selection regret distribution}
}
\]

和：

\[
\boxed{
\text{minimal-state augmentation 的 regret reduction}
}
\]

所以现在还不应该开始审判 RQ2。

下一轮继续 RQ1 时，不再继续泛搜“还有哪些 heuristic”，而应该把上述每一个 RQ1 case进一步落成一张**实验级审判表**：精确列出 \(P(x)\) 的每一维、必须固定的变量、hidden variable、candidate intersection、oracle、warm-up/repetition、regret metric，以及什么实验结果判 H1/H2/H3/H4 成立或失败。只有这张表也闭合，RQ1 才可以正式封存，然后进入 RQ2。

## 用户

证据链不完整请继续补充。然后把证据链也凝炼为问题。然后从RQ1开始做一条完整的“问题审判”：所有相关框架源码规则 → 每个规则使用的proxy → 它忽略了哪些变量 → 框架自己的fallback/repair → 现代Attention/GQA/MLA/MoE触发条件 → 可证伪的科研假设。 RQ1彻底完成之后再审判RQ2，然后后续RQ。这样可以防止又回到前几轮“问题列了很多，但证据深度不够”的状态 。当然如果新的证据链形成新的RQ 也请补充。

## 模型回复

这一轮继续只做两件事：先补齐 RQ1 还缺的 selector / proxy / repair 证据，尤其是当前还过于粗粒度的 CUTLASS、Triton、FlashInfer、TileLang、vLLM/SGLang；然后把 RQ1 封装成一条可以直接做实验的“源码规则→proxy collision→regret→可证伪假设”链。RQ2 仍不开始。

## 模型工具调用参数

{"paths":["GitHub"],"query":"search"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索并核实GPU推理优化实现细节

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取 SM100 UMMA 构建器文件内容

## 用户

证据链不完整请继续补充。然后把证据链也凝炼为问题。然后从RQ1开始做一条完整的“问题审判”：所有相关框架源码规则 → 每个规则使用的proxy → 它忽略了哪些变量 → 框架自己的fallback/repair → 现代Attention/GQA/MLA/MoE触发条件 → 可证伪的科研假设。 RQ1彻底完成之后再审判RQ2，然后后续RQ。这样可以防止又回到前几轮“问题列了很多，但证据深度不够”的状态 。当然如果新的证据链形成新的RQ 也请补充。

## 模型思考摘要

继续核查实现
继续核查实现

## 模型回复

我继续把 RQ1 当成唯一主线，不进入 RQ2。这一轮重点不是再重复已有结论，而是做“反向漏项审计”：逐仓库搜索 `heuristic / prefer / default / fallback / priority / auto / swizzle / stage / layout`，只记录那些**会改变最终 layout/config 选择**、且此前没有被完整纳入 RQ1 证据链的生产规则。

## 模型工具调用参数

{"paths":["GitHub"],"query":"search"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索核心仓库中的启发式实现

## 模型思考摘要

检索核心仓库中的启发式实现

## 模型回复

反向审计又找到一个需要补入 RQ1 的重要层：**vLLM 自己并不只是 runtime/backend resolver，它还内置量化 GEMM kernel 的 launch/tile heuristics**。例如 Machete 代码生成里有基于 \(M\) 区间的默认 tile/cluster heuristic；环境选项甚至明确说 `relax` 模式“用 heuristic 选择要跑的 GEMM shapes，目标近似全跑，但不保证”。这意味着 vLLM 在 kernel-config 层也存在 candidate compression，不能只用 KV/attention resolver代表整个框架。 

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

分析 vLLM Machete 布局启发式

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

查找 generate.py 中的启发式配置

## 模型工具调用参数

{"uri":"/response/turn778","query":"default_tile_heuristic_config"}

## 模型工具调用参数

{"uri":"/response/turn778","query":"M > 64"}

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 Triton 当前仓库启发式实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 Triton 中的启发式实现

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

调研 Triton 与推理框架中的启发式策略

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 SGLang FP4 后端配置与自动选择 logic

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

检索 FlashInfer 当前启发式逻辑

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

搜索 FA2DetermineCtaTileQ 当前源码定义

## 源页面已脱敏的模型输出

The output of this plugin was redacted.

## 模型思考：工具活动

获取并检查 utils.cuh 完整规则

