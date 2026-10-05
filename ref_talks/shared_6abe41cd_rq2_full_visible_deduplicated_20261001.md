# RQ2 共享讨论：完整可见内容归档（去除重复流）

来源：https://chatgpt.com/share/6abe41cd-b72c-83ec-881b-57a4602d51d7
HTML SHA256：0b7b2e106e05a04b196980a08b0b95e92ad6606c1169fd317c355199bfac7cc8
唯一可见消息：10；角色计数：{'assistant': 5, 'user': 5}；唯一 PR/issue：139。
保留全部 user 与 assistant final；不含工具流、内部 reasoning。URL 按 repo+number 去重；不是原生实验完成声明。

## 可见消息 1 — assistant

会，而且这里需要把 RQ2 的实验定义再向前推进一步。最重要的修正是：

> **一个复杂子图不是“选择一个 layout”，而是在多个中间 tensor / state / metadata 上选择一组 layout，同时决定哪些 tensor 属于同一个 layout domain、哪些边界允许 conversion/view/repack。**

因此真正要验证的不是“子图越大，最快 layout 会不会变”，而应该验证：

\[
\boxed{
\text{扩大图边界或改变拓扑之后，
旧子图的最优 layout assignment 是否仍然是大图的最优可扩展方案？}
}
\]

这个命题可以做得非常严格，而且我认为比原来 RQ2 的 multi-consumer 设计强很多。

---

## 一、先重新定义：复杂子图的“最优 layout”是什么

假设子图是

\[
G=(V,E)
\]

其中算子是 \(V\)，tensor/dataflow edge 是 \(E\)。

不要再定义：

\[
L(G)\in\{NHD,HND\}
\]

而应该定义每条 layout-sensitive edge 都有自己的变量：

\[
\mathbf L_G
=
\{L_{e_1},L_{e_2},\ldots,L_{e_m}\}.
\]

例如：

```text
                x
                │
              GEMM
                │ A
          ┌─────┴─────┐
          │           │
        Norm        Consumer-B
          │ B          │ C
        RoPE            │
          │ D           │
          └──────┬──────┘
                 │
               Join
                 │ E
              Consumer
```

你需要搜索的不是一个 `layout`，而是：

\[
(L_A,L_B,L_C,L_D,L_E).
\]

如果每个 tensor 有 4 个候选 layout，那么五个 tensor 就已经有

\[
4^5=1024
\]

种 assignment。

如果还允许：

- A/B 共用一个 domain；
- B/C 分 domain；
- conversion 插在哪里；
- view 还是 materialize；
- kernel tactic；
- page size；
- memory level；

搜索空间更大。

所以新的 optimization object 应该是：

\[
\boxed{
x_G=(\mathbf L_G,\mathbf P_G,\mathbf C_G,\mathbf K_G)
}
\]

其中：

- \(\mathbf L_G\)：每个 tensor 的 physical layout；
- \(\mathbf P_G\)：layout-domain partition；
- \(\mathbf C_G\)：domain 边界上的 conversion/view/materialization；
- \(\mathbf K_G\)：必要时包括与 layout 绑定的 kernel/tactic。

真正 oracle：

\[
x_G^*
=
\arg\min_{x\in\mathcal X(G)}
T_{\rm whole}(G,x).
\]

---

# 二、和这个问题最直接相关的 PR/Issue

我重新按“**图扩大/拓扑改变会不会改变 layout inference 或 layout winner**”筛了一轮。下面这些比普通 layout optimization PR 更直接。

| 框架 / PR | 它为什么与你的问题直接相关 | 对实验的启示 |
|---|---|---|
| TileLang [#2960](https://github.com/tile-ai/tilelang/pull/2960) + [Issue #1729](https://github.com/tile-ai/tilelang/issues/1729) | 对一个 **connected free-mode component**，会“以 component 中每个成员分别作为 inference root”，得到多个完整 layout assignment，然后用整个 component 的 global-memory IO cost + register cost 选 winner。Issue #1729 里旧选择导致 256 threads 中只有 2 个真正干活 | **非常直接地证明不能只根据一个节点决定 layout；必须评价整个 connected component** |
| TileLang [#3180](https://github.com/tile-ai/tilelang/pull/3180) | PR 原文几乎就是你的问题：`enumerate all implied single-fragment layouts in a connected component (fix one layout = fix all layouts)` | layout 决策是 connected-component assignment，不是单 tensor assignment |
| TileLang [#3284](https://github.com/tile-ai/tilelang/pull/3284) / [Issue #3283](https://github.com/tile-ai/tilelang/issues/3283) | 一个 fragment 同时作为 SFA 和 SFB 后，两条路径对它提出**不兼容 layout requirement**。最终必须使用不同 scale fragments 或 shared-memory scale | 图扩大后甚至可能从“一个 domain 最优”变成“一个 domain 根本不合法”，必须 split |
| TileLang [#2940](https://github.com/tile-ai/tilelang/pull/2940), [RFC #2897](https://github.com/tile-ai/tilelang/issues/2897), [Issue #2408](https://github.com/tile-ai/tilelang/issues/2408) | reducer 跨多个 `T.Parallel`、多个 pipeline iteration；旧实现把 physical replication/layout 与 logical reduction topology 混在一起会产生错误 | branch/reduction/reuse topology 会改变 layout ownership 和 domain |
| Triton [#7447](https://github.com/triton-lang/triton/pull/7447) | `AutoLayout` 从 downstream concrete layout **反向穿过整个图传播**；冲突时直接报错 | downstream operator 可以改变很远 upstream tensor 的 layout |
| Triton [#7718](https://github.com/triton-lang/triton/pull/7718) | reshape 等 operator 存在**多个合法 layout inference result**；backward 和 forward propagation 得到不同候选时需要 conflict resolution | 一个节点不能只有“唯一推导 layout”；需要保留候选集合 |
| Triton [#11685](https://github.com/triton-lang/triton/pull/11685) | `splat→join→transpose→reshape→transpose→dot` 的复杂链中，增加 **distance from layout seed** 来解决多个传播路径冲突 | layout preference 与“离哪个 consumer/anchor 更近”有关，图深度本身成为变量 |
| Triton [#10158](https://github.com/triton-lang/triton/pull/10158) | layout 需要穿过 `scf.if`、`scf.for`，包括 loop iter-arg 最终被 dot 消费 | control-flow topology 也是 layout domain 的一部分 |
| Triton [#11117](https://github.com/triton-lang/triton/pull/11117) + [Issue #10987](https://github.com/triton-lang/triton/issues/10987) | **两个 chained division + downstream reduction** 才触发 replicated layout pathology。去掉 reduction 或去掉其中一个 division，问题都消失。修复后 `div.full.f32` 4128→64 | 这是目前非常漂亮的“拓扑改变→layout 选择后果改变”的因果案例 |
| Triton [#11538](https://github.com/triton-lang/triton/pull/11538) + [Issue #11526](https://github.com/triton-lang/triton/issues/11526) | 同一个 load/value 沿不同路径给多个 dot，conversion hoist/rematerialization placement 会改变 shared-memory 使用；Issue 中 tf32x3 SMEM 1536→3072B | 图中的 reuse/path multiplicity 会改变 conversion/domain placement |
| Triton [#11760](https://github.com/triton-lang/triton/pull/11760) | broadcast 前后 layout compatible 时允许吸收 conversion；同时必须处理 shared-source conflict | broadcast/join 会改变 layout equivalence 和 conversion 数量 |
| vLLM [#54795](https://github.com/vllm-project/vllm/pull/54795) | Pipeline-parallel 各 worker 因本地 attention backend 不同，暴露不同 KV layout 集合；不能要求列表完全相同，而要取 ordered intersection | **拓扑中多一个 PP stage/backend，整个可行 layout 集可能缩小** |
| vLLM [#59112](https://github.com/vllm-project/vllm/pull/59112) | manager block、kernel page、target attention、draft attention、metadata/scheduler 同时出现；packed BLHNC + re-page view 使整个系统行为变化 | storage layout 和 kernel layout 可以属于不同 domain；扩大到 scheduler/metadata 后，consumer-only 比较已经不够 |
| SGLang [#40326](https://github.com/sgl-project/sglang/pull/40326) → [#40327](https://github.com/sgl-project/sglang/pull/40327) → [#38592](https://github.com/sgl-project/sglang/pull/38592) → [#40330](https://github.com/sgl-project/sglang/pull/40330) | token-major/unified-memory layout change会穿过 writer、page view、多个 attention backend、HiCache、PD transfer、translate 等几十个 site | 非常适合实验“传播距离增加以后，固定局部 layout 是否失效” |
| SGLang [#41944](https://github.com/sgl-project/sglang/pull/41944) | fused projection 输出的 q/k/v 是 row-strided slice。直接让 attention 读 stride 后，attention kernel 自身反而略慢，但省掉 144 个 copy，整个 DiT forward 仍更快 | **这是 operator-local winner ≠ expanded-subgraph winner 的现实直接证据** |
| AITER [#5543](https://github.com/ROCm/aiter/pull/5543) | token-major GDN 在 gfx942 上 kernel 本身约慢 4–6%，但如果 framework upstream 原生就是 token-major，则能省 transpose，整图可能反胜 | 与上一个形成 AMD 侧独立复核 |
| TensorRT-LLM [#19541](https://github.com/NVIDIA/TensorRT-LLM/pull/19541) | MoE 从 full-GPU domain 变成两个 Rubin locality domains，并共享 parent reset/quant；domain boundary 本身是 optimization variable | 复杂图不仅选 tensor layout，还要选 domain partition |
| FlashInfer [#4657](https://github.com/flashinfer-ai/flashinfer/pull/4657) | 同时搜索 scale layout `{8×4,128×4}` × kernel tactics；75 shape 中两个 layout 都大量获胜 | 证明搜索空间中多个 layout/tactic 变量不能彼此独立选择 |

我认为其中最关键的六组是：

**TileLang #2960/#3180、TileLang #3284、Triton #11685、Triton #11117/#10987、SGLang #41944/AITER #5543、vLLM #54795。**

因为它们不是简单说“layout A 比 B 快”，而是在直接说明：

\[
\boxed{
\text{graph structure / propagation path / connected component / additional constraint}
\rightarrow
\text{layout decision changes}
}
\]

---

# 三、真正合理的验证方式：不要直接比较“小图 winner”和“大图 winner”

这是最重要的一点。

假设有一个小子图：

\[
H\subset G
\]

例如：

```text
H:
P → A → B
```

扩展后：

```text
G:
P → A → B → C
        ↘ D → E
```

如果你只是发现：

\[
L_H^*\neq L_G^*
\]

其实还不够。

因为：

1. \(G\) 多了新的 layout variables；
2. \(G\) 的搜索空间不同；
3. 小图中的某个 assignment 可能有很多种扩展方式；
4. winner 可能只是 0.5% measurement noise。

真正应该测的是下面这个量。

---

# 四、核心指标：Expansion Regret

小图最优：

\[
x_H^*
=
\arg\min_{x\in\mathcal X(H)}T(H,x)
\]

大图真正最优：

\[
x_G^*
=
\arg\min_{x\in\mathcal X(G)}T(G,x).
\]

现在最关键的一步：

**固定小图内部所有 layout/domain decisions 为 \(x_H^*\)，只允许新增部分自由搜索。**

定义：

\[
T_{\text{freeze}}(H\rightarrow G)
=
\min_{
x\in\mathcal X(G):
\pi_H(x)=x_H^*
}
T(G,x).
\]

这里 \(\pi_H(x)\) 表示把大图 assignment 投影回旧子图。

然后：

\[
\boxed{
ER(H\rightarrow G)
=
\frac{
T_{\text{freeze}}(H\rightarrow G)
}{
T(G,x_G^*)
}
-1
}
\]

这就是我建议你以后 RQ2 最核心的指标。

它回答的不是：

> 大图 winner 名字变了吗？

而是：

> **如果我坚持沿用“小子图时代认为最优的 layout 决策”，扩大图以后究竟会损失多少性能？**

---

# 五、举一个非常具体的例子

假设小图只有：

```text
Projection
    ↓
Attention
```

小图测试得到：

```text
Projection output = contiguous
Attention input   = contiguous
```

最快。

现在扩大：

```text
                        ┌→ Attention
Projection → packed QKV ┤
                        ├→ KV append
                        └→ speculative consumer
```

新的 oracle 可能是：

```text
Projection output = packed / strided
Attention         = stride-aware
KV append         = direct packed
Spec consumer     = packed view
```

那么不能简单说：

> packed 赢了 contiguous。

而应该比较：

```text
Small-graph frozen decision:
Projection → contiguous
             ├→ Attention
             ├→ KV append
             └→ Spec
```

允许 downstream 自己做最优处理。

和：

```text
Large-graph oracle:
Projection → packed
             ├→ stride-aware Attention
             ├→ packed KV append
             └→ packed Spec
```

假如：

\[
T_{\rm freeze}=110\mu s
\]

而

\[
T_{\rm oracle}=91\mu s,
\]

那么：

\[
ER=20.9\%.
\]

这才能说明：

\[
\boxed{
\text{扩大 layout domain 的可见范围确实改变了旧决策的最优性。}
}
\]

---

# 六、还必须区分两种“layout 发生变化”

这个区别很重要。

### 第一种：legality-induced change

例如 TileLang #3284。

小图：

```text
scale fragment S
       ↓
     MMA-A
```

S 有合法 layout \(L_A\)。

扩成：

```text
          ┌→ MMA-A expects LA
scale S ──┤
          └→ MMA-B expects LB
```

而：

\[
L_A\neq L_B.
\]

此时共享一个 S 根本没有合法 layout。

那么这不是：

> 另一个 layout 更快。

而是：

\[
\boxed{
\text{topology expansion 使原 domain partition 不再合法}
}
\]

必须：

```text
SA → MMA-A
SB → MMA-B
```

或者：

```text
shared memory canonical domain
   ├→ layout A view
   └→ layout B view
```

因此结果应该单独统计：

\[
\text{Feasibility Flip}.
\]

---

### 第二种：performance-induced change

旧 layout 在大图中仍然完全合法，但是变慢。

例如 SGLang #41944 / AITER #5543：

```text
Consumer-only:
contiguous input → kernel
```

可能更快。

但是完整图：

```text
packed producer
    ↓
.contiguous()
    ↓
consumer
```

却可能比：

```text
packed producer
    ↓
stride-aware consumer
```

慢。

这才是：

\[
\boxed{
\text{Performance Flip}
}
\]

也是你论文里更强的现象。

---

# 七、不要只使用“winner flip”；要使用 near-optimal set

复杂图里面很容易有：

```text
layout A = 10.00 us
layout B = 10.04 us
```

下一层图：

```text
A = 15.05
B = 15.00
```

这不是科研意义上的 layout flip。

所以定义：

\[
\mathcal S_\epsilon(G)
=
\{
x:
T(G,x)
\le
(1+\epsilon)T_G^*
\}.
\]

比如继续用你现在体系里的 3% practical threshold：

\[
\epsilon=0.03.
\]

小图可能：

\[
\mathcal S_{3\%}(H)=\{A,B\}.
\]

大图：

\[
\mathcal S_{3\%}(G)=\{B,C\}.
\]

那么最好不要声称“A→B winner flip”，而是计算：

\[
T_{\rm freeze-set}
=
\min_{
x\in\mathcal X(G):
\pi_H(x)\in\mathcal S_\epsilon(H)
}
T(G,x)
\]

再定义：

\[
ER_{\rm set}
=
\frac{T_{\rm freeze-set}}{T_G^*}-1.
\]

如果连**小图所有 near-optimal choices**到了大图都显著落后：

\[
ER_{\rm set}>3\%,
\]

这个证据会非常强。

---

# 八、其次应该验证“domain 数量”是不是随着图复杂度改变

你现在的问题还包括：

> 一个子图可以有多个 layout。

所以另一个非常重要的实验变量不是 layout 本身，而是：

\[
K=\#\text{layout domains}.
\]

定义：

\[
T_K^*(G)
=
\min_{
x\in\mathcal X(G),
\#Domains(x)\le K
}
T(G,x).
\]

分别测：

\[
K=1,2,3,\ldots
\]

于是会得到：

```text
          latency
             │
             │ \
             │  \
             │   \______
             │
             └──────────── K
                1 2 3 4
```

例如：

| Graph | K=1 | K=2 | K=3 | Oracle |
|---|---:|---:|---:|---:|
| 2-op | 10.0 | 10.0 | – | 10.0 |
| 4-op | 18.0 | 15.1 | 15.0 | 15.0 |
| 7-op | 31.0 | 25.5 | 22.2 | 22.1 |

这时就可以定义：

\[
K_\epsilon^*(G)
=
\min\{
K:
T_K^*(G)
\le
(1+\epsilon)T^*(G)
\}.
\]

真正研究：

\[
\boxed{
K_\epsilon^*(G)
\text{ 是否随着 graph complexity 增大而上升？}
}
\]

这比问：

> NHD 还是 HND？

要高级很多。

---

# 九、如何证明是“拓扑”影响，而不仅仅是算子数量影响

这个实验必须做，否则 reviewer 很容易问：

> 你只是加了更多工作，当然结果不同。

所以需要两类扩展同时做。

### A. Nested real-subgraph expansion

保持真实 graph：

```text
G1:
A → B

G2:
P → A → B

G3:
P → A → B → C

G4:
          ┌→ C
P → A → B ┤
          └→ D

G5:
          ┌→ C ─┐
P → A → B ┤      ├→ E
          └→ D ─┘
```

也就是说：

\[
G_1\subset G_2\subset\cdots\subset G_n.
\]

shape 不变，核心 operator 不变，只扩大边界。

测每一次：

\[
ER(G_i\rightarrow G_{i+1}).
\]

---

### B. Topology counterfactual

控制：

- 算子数量相同；
- operator type 相同；
- tensor shapes 相同或尽可能匹配；

只改变 topology。

例如四个 operator：

```text
Chain:
A → B → C → D
```

对比：

```text
Fork:
      ┌→ C
A → B ┤
      └→ D
```

对比：

```text
Diamond:
      ┌→ B →┐
A ────┤     ├→ D
      └→ C →┘
```

对比：

```text
Reuse:
A → B → C
    ↑   │
    └───┘
```

你不一定要人为改变真实 model semantics；可以从真实 PR 中分别抽这些 motif，然后做 controlled micrograph replication。

真正关注的是：

\[
\text{same-ish operator set}
+
\text{different topology}
\rightarrow
\text{different oracle assignment/domain partition}.
\]

---

# 十、我建议你的 topology taxonomy 至少包含这 6 种

这不是“只做 multi-consumer”。

| 拓扑 | 真实来源 | 为什么重要 |
|---|---|---|
| Serial chain | Triton #11685 | downstream preference 可跨很多 op 反向传播 |
| Branch/fanout | vLLM KV、SGLang QKV | 一个 representation 面向多个不同 consumer |
| Diamond/join | MoE / residual / combine | 两条不同 layout 路径最终又必须 merge |
| Reduction | Triton #10987、TileLang reducer | reduction 对 replication/layout 有特殊强约束 |
| Loop/state reuse | KV、GDN、Mamba | layout 成本不仅付一次，而会跨 timestep reuse |
| Distributed/memory boundary | vLLM PP/P-D、SGLang HiCache、TRT locality domain | domain 横跨 GPU/CPU/worker/locality domain |

---

# 十一、搜索空间确实会爆炸，所以不要对所有大图都“暴力枚举”

如果有 \(m\) 个 layout-sensitive tensor，每个平均有 \(q\) 个候选：

\[
q^m.
\]

如果 layout domain partition 也自由，理论上还涉及 Bell number：

\[
B_m.
\]

因此大图不能直接声称“我枚举了所有可能 layout”。

更合理的实验应该是三层。

### 第一层：Exact-oracle small graph

对于：

\[
m\le 6\sim 8
\]

左右的真正 layout variables，经过合法性 pruning 后，把所有 assignment **完整执行 whole subgraph**。

这是你真正可以叫：

\[
\text{measured oracle}.
\]

用它验证所有核心 scientific claims。

---

### 第二层：Pruned exact graph

利用 framework/operator layout contract：

```text
Op1:
input ∈ {A,B}
output ∈ {C}

Op2:
input ∈ {C,D}

Op3:
requires stride(-1)=1
```

先删掉不合法 assignment。

还应该把 zero-copy physically-equivalent representations 合并：

\[
L_a\sim L_b
\]

如果只是不同名字，但 shape/stride/storage mapping 等价。

然后再 exhaustive。

---

### 第三层：Large graph search

大图不应该再叫 oracle。

应该叫：

\[
\boxed{\text{best-measured assignment}}
\]

或者：

\[
\text{searched best}.
\]

可以采用：

- component decomposition；
- dynamic programming；
- beam search；
- coordinate search；
- autotuning；

但最重要的是：

> **搜索算法输出的 top candidates 最后必须直接运行 whole subgraph。**

不要仅用：

\[
\sum T_{operator}
+
\sum T_{conversion}
\]

来判 winner。

因为 SGLang #41944、DeepGEMM #440、TRT-LLM #19541 已经说明：

- fusion；
- overlap；
- launch removal；
- register pressure；
- SMEM；
- scheduling；
- communication；

都会让简单 additive model 失真。

---

# 十二、搜索算法本身也需要验证，而不能默认它找到了最优

这是复杂图实验很容易出现的漏洞。

正确方式是：

在小图上你有 exhaustive oracle：

\[
x_{\rm exact}^*.
\]

同时运行你以后想用于大图的 search：

\[
x_{\rm search}^*.
\]

计算：

\[
SearchRegret
=
\frac{
T(x_{\rm search}^*)
}{
T(x_{\rm exact}^*)
}
-1.
\]

比如在 100 个小/中等子图上：

```text
exact search space: 64–4096 candidates
beam/DP search: 8–64 evaluations
```

如果 search consistently：

\[
SearchRegret<1\%
\]

或在你的 threshold 内，再允许把它用于大图。

否则：

> 大图搜索得到的“最优”本身不可相信。

---

# 十三、最值得做的一套 canonical 实验

我建议你直接做一个 `Graph Expansion Ladder`。

假设真实 parent 是 attention block：

```text
G1:
KV → QK consumer
```

```text
G2:
KV writer → KV → QK consumer
```

```text
G3:
projection → RoPE → KV writer → QK
```

```text
G4:
                         ┌→ QK
projection → RoPE → KV ─┤
                         └→ sparse/index consumer
```

```text
G5:
                         ┌→ target attention
projection → RoPE → KV ─┼→ draft attention
                         └→ offload/transfer
```

```text
G6:
allocator
   ↓
projection → RoPE → persistent KV
                  ├→ target
                  ├→ draft
                  ├→ connector
                  └→ offload
                         ↓
                     reload
```

对于每一级都求：

\[
x_{G_i}^*
\]

然后记录：

\[
ER(G_i\rightarrow G_{i+1})
\]

\[
K_\epsilon^*(G_i)
\]

和旧 tensor 的 layout 是否发生变化。

最终你可以画三张非常核心的论文图。

---

# 十四、第一张图：Graph Size vs Frozen-Decision Regret

横轴：

\[
|V|
\quad\text{或}\quad
\#layout\ variables
\]

纵轴：

\[
ER.
\]

例如：

```text
Expansion regret
30% |                     *
    |                *
20% |          *
    |
10% |     *
    | *
 0% +-------------------------
      2  3  4  5  6  8 10
       layout-sensitive ops
```

如果这个趋势成立，你才能真正写：

> local layout decisions become increasingly insufficient as the optimization domain expands.

---

# 十五、第二张图：Optimal Domain Count vs Graph Complexity

横轴：

\[
\#operators
\]

纵轴：

\[
K^*_{3\%}.
\]

如果得到：

```text
2 op  → K*=1
3 op  → K*=1
5 op  → K*=2
7 op  → K*=3
10 op → K*=3
```

这直接回答：

> layout domain 应该划多细？

而不是靠一个 1:1 multi-consumer pipeline 推断。

---

# 十六、第三张图：旧子图中的 layout 究竟在哪里被远端算子改变

定义一个 shared edge \(e\)。

新增 operator \(v\) 距离 \(e\) 的 graph distance：

\[
d(e,v).
\]

如果增加一个距离 1 的 consumer 改变了 \(L_e\)，很正常。

更有意思的是：

```text
e ─ A ─ B ─ C ─ D
                ↑
          新增 constraint
```

最终 \(e\) 的 layout 也改变。

可以定义：

\[
InfluenceDistance(e)
=
\min d(e,v)
\]

或者统计：

\[
P(\text{layout flip}\mid d).
\]

这和 Triton #11685 的 seed-distance 思路非常呼应。

甚至论文可以问：

\[
\boxed{
\text{How far should layout preference propagate through a subgraph?}
}
\]

这比“domain 大小”更具体。

---

# 十七、你的判定标准应该很严格

我建议只有满足下面条件才叫：

> graph expansion materially changes optimal layout selection.

具体就是：

1. shared core 的 shape、dtype、operator implementation 不变；
2. correctness 全通过；
3. 小图最优 layout 在大图中如果仍合法，要计算其**最佳扩展**，而不是随便补 downstream layout；
4. `ER > practical threshold`，例如继续使用 3%；
5. process-level 重复稳定；
6. bootstrap 95% CI 下界 > 0；
7. 如果只是 winner 名变了，但两个都处于 3% near-optimal set，则不算 strong flip；
8. legality change 和 performance change 分开报告。

这套标准会比：

> 小图 NHD 赢、大图 HND 赢，所以 domain matters

严谨很多。

---

# 十八、我认为新版 RQ2 可以正式写成三个子假设

你现在完全可以把 RQ2 从原来的“shared vs split multi-consumer”升级成下面这样：

\[
\boxed{
H2.1:
\text{Expanding a layout optimization domain can invalidate or materially degrade the layout assignment optimal for a smaller nested subgraph.}
}
\]

用：

\[
ER(H\rightarrow G)
\]

验证。

---

\[
\boxed{
H2.2:
\text{The near-optimal number of independent layout domains changes with subgraph topology and complexity.}
}
\]

用：

\[
K_\epsilon^*(G)
\]

验证。

---

\[
\boxed{
H2.3:
\text{Layout preferences can propagate across multiple operator and control-flow boundaries, so bounded local propagation is insufficient for some real workloads.}
}
\]

用：

- influence distance；
- chain/branch/join/control-flow motifs；
- Triton #10158/#11685；
- TileLang #2960/#3180；

验证。

---

# 十九、原 multi-consumer 实验应该放在哪里

不要删除。

但它应该从：

> RQ2 的主体

降级成：

\[
\boxed{\text{RQ2 minimal motif / sanity check}}
\]

完整 evidence hierarchy 应该变成：

```text
Level 0
2-op producer-consumer
        ↓
Level 1
1 producer → heterogeneous consumers
        ↓
Level 2
serial / branch / join / reduction motifs
        ↓
Level 3
real multi-operator framework subgraphs
        ↓
Level 4
nested subgraph expansion
        ↓
Level 5
whole serving/compiler graph best-measured search
```

这样你的 RQ2 就不会再被 reviewer 质疑：

> 你所谓 layout-domain granularity，实际上只验证了 fanout。

---

## 最终我最建议你采用的核心判据

如果只能选一个，我会选这个：

\[
\boxed{
ExpansionRegret(H\rightarrow G)
=
\frac{
\min_{x\in\mathcal X(G):\,\pi_H(x)\in\mathcal S_\epsilon(H)}
T(G,x)
}{
\min_{x\in\mathcal X(G)}T(G,x)
}
-1
}
\]

它非常干净。

含义是：

> “把子图扩大以后，即使我允许新增区域自由优化，但强制保留旧子图中所有原本 near-optimal 的 layout 选择，最好能做到多快？相对于重新联合优化整个大图还差多少？”

如果这个值稳定地大于 3%，那么你就有相当强的证据证明：

\[
\boxed{
\text{layout optimization domain 的扩大本身，
确实影响了最优 layout assignment；
不是单纯因为又多执行了几个算子。}
}
\]

这比简单统计 `winner flip`、multi-consumer winner 或 conversion amortization 都更适合作为新版 RQ2 的主判据。

---

If you want, I can:

- 继续说明合理的验证方式
- 举更多具体案例说明验证策略
- 分享针对大规模复杂图的实用搜索策略

## 可见消息 2 — user

还有个问题，layout的domain 扩大了，比如子图扩大了，不在是只有producer-consumer两个op或两种操作，子图结构更复杂，包含的算子的拓扑结构更多样了，算子更多了，应该如何验证 确实对最优layout的选择是有影响的？因为一个子图不止可以选一个layout, 甚至可以选多个，而且搜索空间明显扩大。请在此筛选和这个主题相关的pr/issue，并且回答更合理的验证方式问题

## 可见消息 3 — assistant

是。按照你现在的研究目标，原来的 RQ2 需要明显扩展。只做

\[
\text{KV writer}\rightarrow\{\text{head consumer},\text{token consumer}\}
\]

即使把 fanout 扫得很全，也只能证明“一个 tensor 的多个 consumer 存在 layout conflict”，不足以支持“真实复杂子图中的 layout-domain granularity / layout propagation”。

我按 2025-01-01 到 2026-10-01 重新扩展检索，范围包括你原来重点看的 vLLM、SGLang、FlashInfer、Triton、CUTLASS、TVM、TileLang，并补了 TensorRT-LLM、FlashAttention、DeepGEMM、AITER。检索不只用了 `layout`，还覆盖了 `stride / packed / swizzle / TMA / relayout / contiguous / page / block / offload / fragment / memory scope / weight layout / scale layout / KV / state / sharding` 等关键词。

先说明一个边界：下面已经比较接近“研究意义上的全量 relevant cases”，但我不能声称 GitHub 上数学意义的 100% 完备——有些 PR 标题和描述完全不出现 layout/stride，或者被 squash/cherry-pick 到别的 PR。不过从“能否形成 RQ2 实验 case”的角度，这一轮已经把主要类别覆盖得相当完整。

---

# 1. 先把新版 RQ2 的实验单位定下来

以后一个 case 不应该再写成：

\[
Producer \rightarrow Consumer
\]

而应该写成：

\[
G=(V,E)
\]

其中一个真实子图可能是：

```text
                  Tensor A
                    │
             ┌──────┴──────┐
             ▼             ▼
          Operator 1    Operator 2
             │             │
          Tensor B       Tensor C
             │             │
             ├──────┐ ┌────┤
             ▼      ▼ ▼    ▼
          Operator 3     Operator 4
               \           /
                \         /
                 Tensor D
                    │
                 Operator 5
```

每个 tensor/representation 都有 layout variable：

\[
L_A,L_B,L_C,L_D,\ldots
\]

而每个 operator 可能对输入、输出分别有 layout contract：

\[
\mathcal L_{O_i}
=
\{
(L_{in},L_{out},M)
\}
\]

这里的 \(M\) 还可以包括 memory level：

\[
HBM,\quad SMEM/LDS,\quad TMEM,\quad REG,\quad CPU,\quad L3,\quad remote\ GPU.
\]

最终真正优化：

\[
\boxed{
\min_{\mathbf L,\mathbf D}
T_{\rm whole\ subgraph}(\mathbf L,\mathbf D)
}
\]

而不是：

\[
\sum_i T_{O_i}^{\rm isolated}.
\]

其中 \(\mathbf D\) 就是 layout-domain partition。

你至少要对每个 real-world case 比较：

\[
\boxed{
FrameworkDefault,\;
LocalGreedy,\;
SharedDomain,\;
PartitionedDomains,\;
GraphOracle
}
\]

并报告：

\[
R_{\rm default}
=
\frac{T_{\rm default}}{T_{\rm oracle}}-1
\]

\[
R_{\rm greedy}
=
\frac{T_{\rm greedy}}{T_{\rm oracle}}-1
\]

和

\[
B_{\rm split}
=
\frac{T_{\rm best\ shared}}
{T_{\rm best\ partitioned}}.
\]

---

# 2. 我建议把找到的真实 PR 分成 8 类，而不是简单堆 PR

这次搜出来的案例实际上形成了非常漂亮的八种 layout-domain archetype：

| 类型 | 核心冲突 |
|---|---|
| A. Persistent-state domain | KV/SSM/Mamba 等持久状态究竟长期保持什么 representation |
| B. Cross-engine domain | Prefill/Decode/Connector/Offload 各自偏好的 layout 不同 |
| C. Multi-input/metadata domain | data、scale、TopK、routing metadata 的 layout 必须联合选择 |
| D. MoE representation domain | dispatch、communication、W13/W31、activation scale、expert GEMM 的 layout 冲突 |
| E. Compiler propagation domain | layout 从 Dot/reduction/store 向前后传播，跨 transpose/reshape/control flow |
| F. Memory-hierarchy domain | HBM→SMEM/LDS→TMEM→REG，每一级可能有不同 layout |
| G. Materialization domain | `.contiguous()`、repack、transpose、conversion 应该在哪里发生，甚至是否应该发生 |
| H. Distributed/locality domain | layout-domain 不只是 tensor layout，还可能是 GPU/locality-domain/TP/DCP partition |

这八类一起做，RQ2 才不再是“multi-consumer benchmark”。

---

# 3. vLLM：这是最适合做 serving-level layout-domain 的框架

下面这些我认为都属于强相关。

| PR / Issue | 真实子图 | 改谁 | 候选 / conflict | 原 PR 已有证据 | 你应该怎么测 |
|---|---|---|---|---|---|
| [Issue #26744](https://github.com/vllm-project/vllm/issues/26744) + [PR #30448](https://github.com/vllm-project/vllm/pull/30448) | Prefill→KV write→postprocess→NIXL→Decode | KV physical layout + block size | Prefill NHD/64 vs Decode HND/16 | Qwen accuracy；异构 block/layout 路径 | 枚举 `Lp×Ltransfer×Ld×Bp×Bd`，完整测 P→conversion→transfer→D |
| [Issue #46204](https://github.com/vllm-project/vllm/issues/46204) + [#46223](https://github.com/vllm-project/vllm/pull/46223) | MiniMax-M3 KV→MSA + NIXL | persistent KV | MSA NHD-only；connector 曾希望 HND | 实际 correctness/startup conflict | shared NHD / transfer-side convert / attention-side convert |
| [#44458](https://github.com/vllm-project/vllm/pull/44458) 及 #44454→#44455→#44456 | allocator→attention→copy→NIXL/Mooncake→CPU offload | 整个 KV representation contract | BLHNC/BHLNC、packed、cross-layer、layer-compact、shared storage | 大量 allocator/copy/connector tests | **旗舰 case**：同一真实 KV graph 上统一枚举 layout propagation |
| [#54795](https://github.com/vllm-project/vllm/pull/54795) | PP worker0 backend + worker1 backend + … | worker candidate sets | 各 PP stage 支持集不同，framework 做 intersection | compatible/disjoint layout tests | shared intersection vs per-stage domain+conversion |
| [#37885](https://github.com/vllm-project/vllm/pull/37885) | hybrid full/sliding KV groups→connector | allocation | per-layer scattered vs canonical contiguous | physical contiguity tests | allocator-local optimum vs transfer-optimal whole graph |
| [#50208](https://github.com/vllm-project/vllm/pull/50208) | packed HMA→Mooncake | physical stride/offset/payload | padded / packed / region offset | K3：write bytes 145.35→128.52 GiB；read 143.02→133.05 GiB | attention+Mooncake 一起测，而非只测 transfer |
| [#50717](https://github.com/vllm-project/vllm/pull/50717) | compressor state + compressed MLA KV→NIXL | block geometry | global min block=4 vs attention block=256 | B300/NIXL correctness | **不同 tensor 使用不同单位的 domain** |
| [#57169](https://github.com/vllm-project/vllm/pull/57169) | packed KV→sparse MLA/indexer→MRv2→spec proposer | physical block stride | packed stride 必须传播给多个模块 | GB200 GSM8K | 随 propagation depth 增大测 default regret |
| [#59112](https://github.com/vllm-project/vllm/pull/59112) | manager block→FlashInfer kernel pages→target+draft attention | manager vs kernel page representation | LBHNC vs packed BLHNC；640 manager page→128 kernel page view | **非常强**：吞吐 +41~49%，KV group 46→5 | 主实验；研究“storage domain≠kernel domain” |
| [#59265](https://github.com/vllm-project/vllm/pull/59265) | mixed FP8/NVFP4 KV→CPU offload→reload→attention | offload layout | allocator-selected packed vs forced LBHNC/BLHNC | 541 tests + byte-exact roundtrip | packed throughout vs canonicalize-before-offload vs after-reload |
| [#59342](https://github.com/vllm-project/vllm/pull/59342) | persistent NVFP4 KV→gather→FP8 staging→sparse MLA | persistent + temporary | native NVFP4 decode vs FP8 staging | B300 correctness/GSM8K | 是否建立 temporary FP8 domain |
| [#49335](https://github.com/vllm-project/vllm/pull/49335) | quantize→DP/EP A2A→MoE | activation-scale layout | comm 要 row-major；FlashInfer CUTLASS 要 swizzled 128×4 | 实际 NaN bug | pre-A2A swizzle vs post-A2A swizzle vs row-major consumer |
| [#52240](https://github.com/vllm-project/vllm/pull/52240) | checkpoint→weight loader→MoE backend | W13 weight | `[w1;w3]` / `[w3;w1]` / interleaved `[g,u]` / `[u,g]` | backend 深度审计 | 同一 checkpoint 对 DeepGEMM/TRTLLM/FlashInfer/Triton 做 graph-level layout test |
| [#56989](https://github.com/vllm-project/vllm/pull/56989) | Engram FP8 table→TP exchange→wkv | exchange representation | BF16 512B/row vs packed FP8+scale 264B | byte volume -48%；无 serving claim | communication-cost crossover |
| [#58766](https://github.com/vllm-project/vllm/pull/58766) | Engram FP8→TP gather→wkv MXFP8 | representation propagation | FP8→BF16→requant vs preserve MXFP8 | GB300 4096 tokens local 60.2→20.4µs；TP4 89.8→15.3µs | **非常好的 preserve-layout-through-subgraph case** |
| [#59337](https://github.com/vllm-project/vllm/pull/59337) | MoE dispatch→receiver→expert compute | DeepEP dispatch layout | eager expanded/exact vs graph contiguous | CUDA graph transitions、GSM8K；未给 speedup | layout 随 runtime phase 动态变化 |
| [#59350](https://github.com/vllm-project/vllm/pull/59350) | receive→gather→reshape→permute→index_copy | receive-side relayout | explicit materialization path | 删除重复 `index_select` | 作为 conversion micro-op accounting case |

vLLM 还有一批我建议放进 appendix / workload pool，而不一定全做主实验：#58196（offload cache 按 layout namespace）、#55907（防止跨 layout cache reuse）、#53544（packed KV canonical mapping）、#59037（packed BLHNC hidden-state extraction）、#59297（attention-group kernel block mapping）、#50336（NVFP4 scale-layout validation）、#55157（NIXL stride descriptors）、#57823（padded row stride）、#58737（Mamba block-table gather）、#58798（fused RoPE/value-scale/packed DiffKV write）、#51947/#51942（packed FP8 representation reuse）。

---

# 4. SGLang：特别适合“一个 layout change 穿过几十个模块”的实验

最值得注意的是 2026 unified-memory 七 PR stack。

| PR | 真实问题 | 为什么非常适合新版 RQ2 |
|---|---|---|
| [#40326](https://github.com/sgl-project/sglang/pull/40326) | unified pool 给 per-layer **strided views**；多个 consumer 却根据 shape 计算地址，或 `.contiguous()` 后写到 private copy | 一次 pool layout 变化传播到 writer、quant kernel、HiCache、debug/canary 等大量 downstream |
| [#40327](https://github.com/sgl-project/sglang/pull/40327) | 各 attention backend 用自己的 `.view(page_size,...)` 建 paged view | 把 page-view stride contract 统一化 |
| [#38592](https://github.com/sgl-project/sglang/pull/38592) | stack 中真正改变行为的 **token-major layout** | 可以作为新的真实 layout candidate |
| #40328 | write location physical semantics | layout propagation 到 write side |
| #40329 | 删除 kernel-page multiplier plumbing | domain 表示简化 |
| [#40330](https://github.com/sgl-project/sglang/pull/40330) | KV translate kernel 支持真实 row/column strides | PD/HiCache/retraction/read/write 共用同一 stride-aware representation |
| #40331 | fused translate contract 命名 | 辅助 |

这个系列特别重要，因为 PR 本身说原始 #38592 涉及 **41 个文件、12 个 serving hot-path attention backend**。这比你的 synthetic multi-consumer 强得多。

其他强 case：

| PR / Issue | 子图 | layout decision | 已有结果 / 价值 |
|---|---|---|---|
| [#33651](https://github.com/sgl-project/sglang/pull/33651) | GPU KV→L2 host→L3/Mooncake | `layer_first/page_head/page_first/page_first_direct/grouped` | GSM8K + PCIe bandwidth；非常强的 GPU→CPU→L3 domain |
| [#39606](https://github.com/sgl-project/sglang/pull/39606) | GPU per-layer→relayout→staging→D2H | per-layer vs page-unified | 2048 tokens：relayout ~0.34ms，D2H ~9.39ms；约95% contiguous D2H |
| [#39039](https://github.com/sgl-project/sglang/pull/39039) | MoonEP dispatch→DeepGEMM→psum→combine | DeepGEMM psum layout + zero-copy shard | Kimi K3 c8 48.5→128.5 tok/s；MoE layer 1.69→0.62ms |
| [#37521](https://github.com/sgl-project/sglang/pull/37521) | packed Ulysses A2A→Q/K/V→FA→output A2A | materialize vs packed split views | attention communication block约4–7%改善；非常适合 collective→attention domain |
| [#39388](https://github.com/sgl-project/sglang/pull/39388) | weight loader→MegaMoE/TRTLLM | W13/W31 | 真实严重 correctness conflict |
| [#34299](https://github.com/sgl-project/sglang/pull/34299) | KDA prefill→checkpoint→decode recurrent state | zero-copy checkpoint / packed decode state | phase-dependent state layout |
| [#41030](https://github.com/sgl-project/sglang/pull/41030) | RMS/quant producer→MoE/GEMM consumer | row-major FP8 scale vs bpreshuffle | 78 relayout launches消失；decode step -1.35% |
| [#41944](https://github.com/sgl-project/sglang/pull/41944) | fused QKVMLP projection→q/k/v slices→XPU attention | row-strided view vs contiguous | 删除144 copies/forward；FA自身慢约1.2ms，但整图仍净省~6ms |
| [#39860](https://github.com/sgl-project/sglang/pull/39860) | speculative conv history→scratch→accept commit | dense scratch vs strip | 6144B→2048B；representation/lifetime domain |
| [#38430](https://github.com/sgl-project/sglang/pull/38430) | GLM NoPE KV→FlashInfer | 656B padded vs 528B compact | -19.5% storage，+24.2% token slots；没声称 serving speedup |
| [#14982](https://github.com/sgl-project/sglang/pull/14982) / [#12196](https://github.com/sgl-project/sglang/issues/12196) | KV→distributed attention→softmax combine | TP replicated vs DCP token-interleaved | memory gain vs communication overhead |
| [#41953](https://github.com/sgl-project/sglang/pull/41953) | unified allocation→MLA/Mamba views→PD transport | tensor offsets/stride/TP shard layout | H200 RDMA correctness |
| [#38373](https://github.com/sgl-project/sglang/pull/38373) | DSv4 persistent KV | paged vs ring | 明确公开成 layout option |

次级但值得收进 case manifest：#38317（packed DCP1→DCP-N transfer）、#32059（shared KV in prefill CP）、#40588（packed QKV A2A）、#39989（XPU packed Ulysses）、#38178（non-contiguous MLA latent slices）、#38771/#39977（MoE scale-layout ownership）、#41875（DSA KV layout startup capability）、#39071（stride-aware QK norm）、#41848（AITER direct slices）、#38092（dead swizzle allocation）、#40682（deferred CUTLASS scale layout）。

---

# 5. FlashInfer：非常适合 data-layout × metadata-layout × tactic 联合选择

这里出现了一个非常关键的新 case。

## #4657：不要只调 tactic，layout 本身也必须进入 autotuning

[FlashInfer PR #4657](https://github.com/flashinfer-ai/flashinfer/pull/4657)

它直接把：

\[
\{\text{8×4 scale layout},\text{128×4 scale layout}\}
\times
\{\text{TRTLLM tactics}\}
\]

做联合 autotune。

75 个 GB200 shape：

- 8×4 赢 43 个；
- 128×4 赢 32 个；
- adaptive 比固定 8×4 geomean +21.9%；
- 比固定 128×4 +27.1%；
- 距离“每 shape 最佳固定 layout oracle”只有约 0.34%。

这几乎可以直接成为你的新版 RQ2 子假设：

\[
\boxed{
\text{layout 和 kernel tactic 不应该独立优化}
}
\]

其他 FlashInfer 强 case：

| PR | 子图 / 决策 |
|---|---|
| [#4574](https://github.com/flashinfer-ai/flashinfer/pull/4574) | MoE FC1 输出 scale layout 由 tactic 决定：`SWIZZLED_8x4` vs `SWIZZLED_128x4`；下游必须知道 producer 实际选了哪个 |
| [#5405](https://github.com/flashinfer-ai/flashinfer/pull/5405) | Q/K RMSNorm→RoPE→Q output→KV append 融成一个 layout domain；B200 两 shape 比 primitive pipeline 约 5.75–6.37x |
| [#5667](https://github.com/flashinfer-ai/flashinfer/pull/5667) | MLA `kv_b_proj`→FP8 cast→K/V concat/layout，原来4个 PyTorch op；真实 Kimi-K3 motivating tail ~380µs、879MB/chunk/layer |
| [#5094](https://github.com/flashinfer-ai/flashinfer/pull/5094) | packed HND KV + token/head-major TopK metadata → sparse MSA；直接 token-major 比 transpose+contiguous eager 快约16–17% |
| [#5272](https://github.com/flashinfer-ai/flashinfer/pull/5272) | KV data + per-(token,head) FP8 scale 的 layout 联合传播 |
| [#4971](https://github.com/flashinfer-ai/flashinfer/pull/4971) | paged HBM→TMA/cp.async→SMEM→MMA；部分 shape TMA 1.17–1.40x |
| [#5447](https://github.com/flashinfer-ai/flashinfer/pull/5447) | packed query + sliding-window pool + compressed pool + sparse route metadata → MLA |
| [#5709](https://github.com/flashinfer-ai/flashinfer/pull/5709) | split attention partials→L2 prefetch→register reducer；split count/layout 和 reducer register pressure 联合 |
| [#5493](https://github.com/flashinfer-ai/flashinfer/pull/5493) | MiniMax-H3 pre-attention：norm/AdaLN/quant/QKV GEMM/QK norm/RoPE/output layout 整链融合；多 shape 2–5.9x baseline chain |
| [#4572](https://github.com/flashinfer-ai/flashinfer/pull/4572) | GDN sliced intermediate cache：physical capacity > logical T；stride-aware alias 消除 contiguous copy |
| [#5117](https://github.com/flashinfer-ai/flashinfer/pull/5117) | DeepSeek V4.1 page layout + 三种 scale format + RoPE/cache + logical→physical mapping；适合定义 layout search space |
| [#5737](https://github.com/flashinfer-ai/flashinfer/pull/5737) | sparse-attention training 直接消费 packed Q/KV 并直接把 dK/dV 写回 packed buffer；消除 contiguous/cat/index_add；某些 step 比 FA reference 约1.15–1.49x |
| [#5757](https://github.com/flashinfer-ai/flashinfer/pull/5757) | flexible NVFP4 KV layout；目前 PR 描述太少，只适合作候选，不宜作为强证据 |

---

# 6. Triton：这是“layout propagation 本身”最完整的一组 real-world evidence

这里应该成为 compiler-side RQ2 的主战场。

| PR / Issue | 图中的 conflict | 你能验证什么 |
|---|---|---|
| [RFC #8281](https://github.com/triton-lang/triton/issues/8281) + [#8450](https://github.com/triton-lang/triton/pull/8450) | Load→Transpose→Reshape→Elementwise→ConvertLayout→Dot | DotOperand layout 是否应反向传播到 GMEM load |
| [#10158](https://github.com/triton-lang/triton/pull/10158) | layout 跨 `scf.if` / `scf.for`，iter arg 最后进入 dot | layout domain 是否应穿过 control-flow boundary |
| [#11117](https://github.com/triton-lang/triton/pull/11117) | elementwise 同时收到 replicated/block layout candidates | 选错造成 division 指令 4128→64；compile 257s→4s |
| [#11685](https://github.com/triton-lang/triton/pull/11685) | splat→join→transpose→reshape→transpose，forward/backward inference 冲突 | seed distance / global propagation |
| [#11760](https://github.com/triton-lang/triton/pull/11760) | source→broadcast→downstream | compatible layout 是否应穿过 broadcast、消除 conversion |
| [#10360](https://github.com/triton-lang/triton/pull/10360) | permute/reshape/split 小 tensor | REG shuffle vs SMEM roundtrip domain |
| [#10563](https://github.com/triton-lang/triton/pull/10563) | WMMA result→epilogue store | downstream store order 反向决定 result layout；Navi3 某 shape 62→73 TFLOP/s |
| [#10837](https://github.com/triton-lang/triton/pull/10837) | CTA layout→elementwise producer→load | CGA layout 是否 rematerialize 到 producer/load |
| [#11360](https://github.com/triton-lang/triton/pull/11360) | MXFP4 canonical→swizzled conversion | materialization fusion；50–91% peak-memory reduction，warm 4.2–21x conversion speedup |
| [#11754](https://github.com/triton-lang/triton/pull/11754) | TMA load→cast/layout convert→MMA | conversion placement + lifetime + buffering；H100 case约1.21x，但+32KiB SMEM、+20 regs |
| [#11704](https://github.com/triton-lang/triton/pull/11704) | MMA/FMA result→convert_layout→store | 保留 compute layout 直接 store 还是为 store 建新 domain；多 FMA kernel约10% |
| [#11295](https://github.com/triton-lang/triton/pull/11295) | load blocked layout vs downstream dot-derived LDS swizzle | producer lane coverage 与 consumer shared encoding 冲突 |
| [#11967](https://github.com/triton-lang/triton/pull/11967) | aligned pointer + mask | vector/coalesced layout 不能只看 pointer alignment |
| [#11780](https://github.com/triton-lang/triton/pull/11780) | warp repartition 后残留旧 convert_layout | layout lifetime / obsolete-domain elimination |
| [#11777](https://github.com/triton-lang/triton/pull/11777) | pipelined shared buffer→logical row view | physical swizzle 必须穿过 nested view |

这已经足够证明：新版 RQ2 不应该再建模成单一 producer-consumer edge。

---

# 7. TileLang：和你想研究的“layout domain connected component”最接近

其中最重要的是：

[PR #3180](https://github.com/tile-ai/tilelang/pull/3180)

它自己的表述就是：

> enumerate all implied single-fragment layouts in a connected component  
> fix one layout = fix all layouts

这基本就是 layout-domain 的编译器定义。

建议重点收：

| PR | 研究意义 |
|---|---|
| [#3180](https://github.com/tile-ai/tilelang/pull/3180) | connected-component layouts；fragment/T.Parallel/slicing；全 component 最小 register |
| [#3176](https://github.com/tile-ai/tilelang/pull/3176) | global memory + local + shuffle + barrier + register + spill 联合 layout cost |
| [#3279](https://github.com/tile-ai/tilelang/pull/3279) | logical tile accesses 在 scheduling/layout inference 前统一暴露；transfer-aware propagation |
| [#3084](https://github.com/tile-ai/tilelang/pull/3084) | reducer natural layout 向 destination 传播；reducer epoch 纳入同一 component |
| [#2785](https://github.com/tile-ai/tilelang/pull/2785) | 任意 TMEM physical layout；不再固定 2-D row-major |
| [#2452](https://github.com/tile-ai/tilelang/pull/2452) | arbitrary sliced SMEM layouts→GMMA/UMMA descriptor |
| [#2953](https://github.com/tile-ai/tilelang/pull/2953) | dtype-changing view 必须保持相同 storage-bit map；zero-copy alias legality |
| [#3284](https://github.com/tile-ai/tilelang/pull/3284) | **同一 fragment 同时作为 SFA/SFB 时两个 layout requirement 冲突**；必须分成两个 fragment/domain 或 shared scales |
| [#2807](https://github.com/tile-ai/tilelang/pull/2807) | reductions 要尊重 swizzled/padded logical layouts |
| [#3320](https://github.com/tile-ai/tilelang/pull/3320) | fragment 跨 VF lifetime → 是否落到 `shared.dyn`；真正的 domain split |
| [#3328](https://github.com/tile-ai/tilelang/pull/3328) | L0C→UB：dual+VF cast vs FixPipe on-path cast |
| [#1509](https://github.com/tile-ai/tilelang/pull/1509) | attention/conv/MLA 从 manual swizzle annotations 转向 auto layout |
| [#1559](https://github.com/tile-ai/tilelang/pull/1559) | buffer-derived vs PlanLoopPartition：选择 replication 更小的合法 layout |
| [#1386](https://github.com/tile-ai/tilelang/pull/1386) / [issue #1336](https://github.com/tile-ai/tilelang/issues/1336) | GQA split decode：fragment→local reduction layout conflict |
| [#3255](https://github.com/tile-ai/tilelang/pull/3255) | production interleaved FP8-key/FP32-scale cache→score→radix top-k→index transform，一次 fused subgraph |

附录还建议保留 #3130、#3063、#3046、#3090、#3091、#3233、#2965、#2137、#2391/#2378、#2737。

---

# 8. CUTLASS：重点不是 serving graph，而是 HBM→SMEM→REG→epilogue domain

| PR | layout chain |
|---|---|
| [#3030](https://github.com/NVIDIA/cutlass/pull/3030) | FlashAttention：GMEM→cp.async/TMA→不同 SMEM swizzle→MMA REG→softmax→PV→store；不同 shape 下 TMA/cp.async winner 会变化 |
| [#3453](https://github.com/NVIDIA/cutlass/pull/3453) | Blackwell FMHA 原来按 shape 重建 packed stride；现在保留 Q/K/V/O/LSE runtime stride |
| [#3256](https://github.com/NVIDIA/cutlass/pull/3256) | persistent grouped GEMM→shape-changing SwiGLU aux output；每 group 必须动态更新 Aux TMA descriptor |
| [#3273](https://github.com/NVIDIA/cutlass/pull/3273) | SM120 NVFP4：A/B TMA + scale TMA + SMEM packed/unpacked + register fragments + MMA |
| [#3345](https://github.com/NVIDIA/cutlass/pull/3345) | C register fragment→SMEM→TMA store 的 TV layout coverage 错误 |
| [Issue #3612](https://github.com/NVIDIA/cutlass/issues/3612) + [#3660](https://github.com/NVIDIA/cutlass/pull/3660) | ThrK slices 的 C register/thread layout；同一 logical C 在多个 K slice 的 ownership |
| [#3194](https://github.com/NVIDIA/cutlass/pull/3194) | grouped FP8 GEMM：TMA descriptor 每 group 更新 + scale layouts + persistent scheduling |

CUTLASS 很适合验证：

\[
L_{\rm GMEM}
\times L_{\rm SMEM}
\times L_{\rm REG}
\times L_{\rm epilogue}
\]

而不是 NHD/HND。

---

# 9. TVM：可以补“graph compiler + WebGPU + physical register layout”

强 case：

| PR | 价值 |
|---|---|
| [#20426](https://github.com/apache/tvm/pull/20426) | WebGPU paged decode：head-dimension tiled layout vs sequence-sharded layout；根据 context length host-side切换。长 context attention 2.9–4x，WebLLM E2E 某些 long-prompt 2x+ |
| [#20080](https://github.com/apache/tvm/pull/20080) | TIRx buffer views：unflatten/flatten/select/narrow/rearrange + ComposeLayout/Swizzle + FlashMLA lowering |
| [#20076](https://github.com/apache/tvm/pull/20076) | `Buffer.local()` 从 logical/storage iterator 语义改为真正 physical register order；gaps/offset/permutation/replication |
| [#19896](https://github.com/apache/tvm/pull/19896) | vector ld/st、shared copy、TMA/tcgen05 descriptors、permute layout、GEMM async |
| [#20329](https://github.com/apache/tvm/pull/20329) | SM100 weight-stationary B collector domain |
| [#17599](https://github.com/apache/tvm/pull/17599) / [#18523](https://github.com/apache/tvm/pull/18523) | Relax graph→texture/custom memory scope→Adreno lowering |
| [#18548](https://github.com/apache/tvm/pull/18548) | PyTorch frontend NCHW/NHWC layout propagation |
| #18579/#18622/#18629/#18638/#18642/#18643 | Relax operator-level `FRelaxInferLayout` 系列，可用于“随着算子数量增多，layout propagation coverage 是否断裂” |

TVM 特别适合做一个：

```text
Conv / layout-sensitive op
  → repeat/tile/gather/scatter
  → view
  → custom scope
  → lower
```

的 graph-level layout propagation benchmark，而不应该只保留当前 `row_reduce/column_reduce`。

---

# 10. TensorRT-LLM：出现了最直接的“locality-domain”真实 PR

这个框架这轮非常值得加入。

最重要的是：

## #19541：名字就叫 MXFP8 locality-domain split

[TensorRT-LLM PR #19541](https://github.com/NVIDIA/TensorRT-LLM/pull/19541)

真实 graph：

```text
routing
   │
parent quant/reset
   │
 ┌─┴─────────────┐
 │               │
locality D0   locality D1
FC1 shard     FC1 shard
 │               │
 └──── join ─────┘
        │
      FC2
        │
      output
```

原 split 需要：

```text
quantize
memset
reset D0
reset D1
GEMM D0
GEMM D1
```

新的 parent-domain 共享 reset/quant 后分裂。

Rubin 某些 workload：

- TEP8 64 tokens：45.85 → 37.74 µs，约 -18%；
- DEP4 32：100.60 → 81.13 µs，约 -19%。

这应该直接成为你的 RQ2 主 case。

其他：

| PR | 价值 |
|---|---|
| [#18763](https://github.com/NVIDIA/TensorRT-LLM/pull/18763) | GVR decode prototype：请求跨两个 Rubin locality domains 分片；producer logits 仍 full-device，因此是“部分 domain split”天然实验 |
| [#19115](https://github.com/NVIDIA/TensorRT-LLM/pull/19115) | PrimsTS MoE：每 locality domain 拥有两个 GEMM 的 output-channel shards 和 scales，FC1 intermediate 又是 shared full-width domain |
| [#19358](https://github.com/NVIDIA/TensorRT-LLM/pull/19358) | strided BF16 Q/K/V→直接 pack 到 cuDNN MXFP8 payload+scale；避免中间 copies；Wan2.2 B200 full generation -7.06% latency |
| [#19731](https://github.com/NVIDIA/TensorRT-LLM/pull/19731) | MiniMax-M3 NVFP4 draft：physical P32 subpage→native FP8/P128 view；作者 B300 128K batch32 报 38.4→20.9 ms |
| [#19749](https://github.com/NVIDIA/TensorRT-LLM/pull/19749) | MoE finalization 应放在 communication combine 前还是后 + FP4 activation-scale contract |
| [#19542](https://github.com/NVIDIA/TensorRT-LLM/pull/19542) | Rubin FC1 NVFP4 gate16/up16 weight layout；公开 PR 当前验证信息不足，可作 candidate |
| [#19338](https://github.com/NVIDIA/TensorRT-LLM/pull/19338) | packed Q heads / compact MLA input；公开 diff evidence 较弱，建议放 supporting set |

此外还有 #17715（NcclEP algorithm/layout）、#19441（unified Dspark KV disagg）、#19391、#18206（mixed KV formats）、#16954（DeepEP NVFP4 transport）、#19369/#19164（attention-DP KV connector）。

---

# 11. FlashAttention：大量真实案例正好证明“不要无脑 contiguous”

最干净的是这一对：

- #2666：先用 `.contiguous()` 修；
- [#2667](https://github.com/Dao-AILab/flash-attention/pull/2667)：再让 kernel 真正 stride-aware。

#2667 的问题是 hd256 SM100 kernel 用 shape 合成 contiguous stride，面对 transpose view 会读错；`.contiguous()` 修复要每 forward 复制约 0.4 GB。最终改成读取真实 stride。

这正好可以构造：

\[
\boxed{
copy\ to\ preferred\ layout
\quad vs\quad
consumer\ reads\ producer\ layout
}
\]

其他：

| PR | case |
|---|---|
| [#2777](https://github.com/Dao-AILab/flash-attention/pull/2777) | backward 的 dQ/dK/dV mutable output layout，不能盲目 `empty_like` 保留非法 stride |
| [#2785](https://github.com/Dao-AILab/flash-attention/pull/2785) | projection 已经产生 packed QKV/KV；用 split views，而不是拆后重新 materialize |
| [#2443](https://github.com/Dao-AILab/flash-attention/pull/2443) | paged KV cp.async vs TMA，甚至 PR 最后建议放弃，因为6–8% prefill gain不足以抵偿复杂度——很好的 negative case |
| [#2336](https://github.com/Dao-AILab/flash-attention/pull/2336) | split-KV partial FP32 output：直接 REG→GMEM，绕开 SMEM，因为 SMEM store/load layout 不匹配 |
| [#2809](https://github.com/Dao-AILab/flash-attention/pull/2809) | 即使用户不需要 LSE，把 LSE state 写回 shared 反而消除 compiler cliff，最高原差异约24%；说明“局部无用 store”可能改善整 kernel physical layout/liveness |
| #2349 | SM120 TMA + warp specialization |
| #2489 | hd256 TMA paged KV |
| #2558 | SM100 MLA + page table |
| #1833/#1823 | varlen sort/head swizzle |

---

# 12. DeepGEMM：非常适合补 MoE、weight layout 和 communication overlap

最值得的是：

## #440：两个 GEMM 不再是两个独立 domain，而变成 ring-activation single-kernel domain

[DeepGEMM #440](https://github.com/deepseek-ai/DeepGEMM/pull/440)

原：

```text
GEMM1
  ↓
large activation pool
  ↓
GEMM2
```

新：

```text
single task stream
GEMM1 ──────┐
            │ lag
ring pool   │
            ▼
          GEMM2
```

H100 EP8：

- 2048 tokens/rank：约 1.44–1.56x；
- 8192：1.33–1.43x；
- symmetric memory 最多缩小约 4–4.6x。

这是非常好的 multi-operator domain case。

另外：

| PR | case |
|---|---|
| [#394](https://github.com/deepseek-ai/DeepGEMM/pull/394) | FP4 expert weight+scale 按实际 CTA-pair access order 打包；row-major vs packet layout；small-token最高约9.5% |
| [#403](https://github.com/deepseek-ai/DeepGEMM/pull/403) | scale-factor physical layout transformation；SM120 `shape=(4,128,2), stride=(256,1,128)` |
| [#311](https://github.com/deepseek-ai/DeepGEMM/pull/311) | paged MQA fused KV `[values|scale] per row`，旧代码错误当成 flat；TMA stride 错导致静默 corruption |
| [#183](https://github.com/deepseek-ai/DeepGEMM/pull/183) | Down GEMM→Combine Send producer-consumer 流水；按 block_m signal，把 compute/communication domain 交叠 |
| #418 | compact M-grouped GEMM execution；PR body 信息不足，作为 candidate |
| #343 | in-flight tensormap update race；descriptor/layout lifetime |
| #86 | swizzle instead of padding，适合 SMEM layout 支持 evidence |

---

# 13. AITER：AMD 侧必须加，不然你的结论太 NVIDIA-centric

这里找到的几个 case 非常合适。

| PR | real-world layout question |
|---|---|
| [#5543](https://github.com/ROCm/aiter/pull/5543) | GDN prefill `w/u/v_new`：head-major `[B,H,T,D]` vs token-major `[B,T,H,D]`。gfx950 token-major≈持平/略快；gfx942 本 kernel 慢约4–6%，但可以消除 framework transpose，因此**必须测整图** |
| [#4057](https://github.com/ROCm/aiter/pull/4057) | GDN recurrent state：`hkv=[N,HV,K,V]` vs `hvk=[N,HV,V,K]`；SGLang pool 是 V-major，直接 stride-aware 后无需 transpose copy |
| [#4550](https://github.com/ROCm/aiter/pull/4550) | packed `mixed_qkv`→q/k/v strided slices→GDR decode；消除三个 contiguous copy。batch32 约13% kernel+copy path gain |
| [#5495](https://github.com/ROCm/aiter/pull/5495) | **layout-dynamic MXFP8 GEMM**：scale mode、plain/preshuffled weights、split/slice-K、tile/stage/wave 联合 tuning |
| [#5094](https://github.com/ROCm/aiter/pull/5094) | MoE scale layout 应由 `shuffle_scale_moe` producer contract 决定 |
| [#2919](https://github.com/ROCm/aiter/pull/2919) | paged attention NHD API，PR 描述不足，可作 supplementary |
| #3959 | sliding-window decode over shuffled FP8 paged KV |
| #3583 | DeepSeek-V4 layout sparse paged prefill |
| #5398/#5744 | MoE layout-v2 GEMM2 / EP 下 domain/overpadding conflict |
| #6001 | FP8 MQA logits 去掉 host row-pad copies |

AITER #5543 特别适合你的研究，因为它给出了非常重要的反例：

> token-major 在 kernel 本身可能慢 4–6%，但如果 upstream 本来就是 token-major，可以省掉 transpose，因此整图仍可能更快。

这就是：

\[
\boxed{
\text{operator-local winner}
\neq
\text{subgraph winner}
}
\]

最直接的现实证据之一。

---

# 14. 所以你不应该“一 PR 一个完全独立实验”

现在至少已经有上百条可关联 PR/issue。如果全部独立做，会导致 workload 很碎，论文反而难讲。

更好的方法是把它们归并成下面 14 个 canonical real-world subgraphs。

| Canonical subgraph | 真实 PR 来源 | 建议布局变量 |
|---|---|---|
| 1. P/D KV handoff | vLLM #30448/#26744/#46204 | prefill layout, transfer layout, decode layout, page size |
| 2. Packed persistent KV | vLLM #44458/#57169/#59112/#59265 | manager layout, kernel view, offload representation |
| 3. Hybrid MLA/Mamba storage | vLLM #50717/#37885 | per-state block geometry + shared allocation |
| 4. Hierarchical KV | SGLang #33651/#39606/#41953 | GPU, staging, CPU/L2, L3 layouts |
| 5. Distributed KV | SGLang #14982, TRT-LLM attention-DP | per-rank token/head/page mapping |
| 6. Projection→attention zero-copy | FlashAttention #2667/#2785, SGLang #41944, AITER #4550 | packed/strided views vs materialize |
| 7. Attention data+metadata | FlashInfer #5094/#5272 | KV, TopK, scale, output layouts |
| 8. Linear/recurrent state | SGLang #12892/#39860/#34299, AITER #4057/#5543 | materialized/lazy, K/V-major, token/head-major |
| 9. MoE weight representation | vLLM #52240, SGLang #39388, DeepGEMM #394 | W13/W31/interleaved/packet |
| 10. MoE dispatch/scale representation | vLLM #49335/#59337, SGLang #39039, FlashInfer #4574 | row-major/swizzled/expanded/contiguous/psum |
| 11. Locality domain | TRT-LLM #19541/#18763/#19115 | full GPU vs two physical locality domains |
| 12. Compiler layout propagation | Triton #8450/#10158/#11117/#11685 | layout anchor placement, propagation through graph/control flow |
| 13. HBM→SMEM→REG | FlashInfer #4971, CUTLASS #3030/#3273, TileLang #2452/#2785 | load primitive, shared swizzle, register fragment |
| 14. Materialization/fusion | Triton #10360/#11360/#11754, FlashInfer #5405/#5667, DeepGEMM #440 | explicit intermediate domain vs fused/alias domain |

这 14 类，比“一个 KV writer + 两个 scan”完善得多。

---

# 15. 每一个 canonical case 应该统一跑同一套 experiment contract

我建议你以后所有 RQ2 case 都输出这些 candidate：

```text
F0: framework_default

F1: framework_default + instrumentation only

S0: maximal-shared-domain
    尽量让整个 subgraph 共享一个 physical representation

G0: local-greedy
    每个 operator 选自己单独最快的 layout
    边界不一致时显式 conversion

P1..Pk: partitioned-domain candidates
    domain boundary 放在不同 operator edges

Z: zero-copy/view candidate
    如果 stride/alias contract 允许

M: materialized candidate
    显式 contiguous/repack/transpose

O: measured graph oracle
    合法候选中完整 subgraph 实测最快
```

然后测四种层级：

\[
T_{\rm operators}
\]

\[
T_{\rm conversions}
\]

\[
T_{\rm full\ subgraph}
\]

\[
T_{\rm E2E}
\]

最后两个最重要。

---

# 16. 一个 vLLM case 应该具体怎么变成实验

以 #59112 为例。

真实图：

```text
allocator
   ↓
manager KV block
   ↓
metadata/page mapping
   ↓
target attention
   ↓
spec draft attention
   ↓
scheduler/cache reuse
```

不要只测：

```text
LBHNC kernel
vs
BLHNC kernel
```

而是：

```text
Candidate A — Framework default
LBHNC manager
→ LBHNC kernel pages
→ baseline metadata

Candidate B — Packed shared
BLHNC manager
→ shared-storage re-page view
→ target+draft read BLHNC

Candidate C — Split domains
BLHNC manager
→ materialize LBHNC kernel pages
→ attention

Candidate D — Different manager/kernel page
BLHNC manager P=640
→ zero-copy/remap
→ kernel P=128

Candidate E
BLHNC manager P=640
→ physical repack
→ kernel P=128
```

然后完整测：

\[
T_{\rm alloc}
+
T_{\rm metadata}
+
T_{\rm target}
+
T_{\rm draft}
+
T_{\rm scheduler}
\]

这才是 layout-domain。

---

# 17. 一个 Triton case 应该怎么变成实验

选：

```text
load
 → transpose
 → reshape
 → elementwise
 → reduction/dot
 → epilogue
 → store
```

设置不同 anchor：

```text
A: layout chosen from load
B: layout chosen from dot
C: layout chosen from store
D: propagate forward only
E: propagate backward only
F: bidirectional
G: forbid propagation across reshape
H: forbid propagation across control flow
I: permit explicit conversion
J: minimize number of conversions
K: measured oracle
```

记录：

- `#convert_layout`
- shared-memory round trips
- register count
- local/spill bytes
- load/store vector width
- PTX/SASS instructions
- kernel latency.

这才是在回答：

> operator 数量增加以后，layout propagation 应如何处理。

---

# 18. 对现有框架默认策略的评估，不要只写“它用了什么”

应该统一变成：

\[
\boxed{
DefaultRegret(c)
=
\frac{T_{\rm framework-default}(c)}
{T_{\rm measured-oracle}(c)}
-1
}
\]

再按照：

- framework；
- subgraph family；
- operator count；
- number of layout-sensitive tensors；
- number of domains；
- memory levels；
- phase；
- shape；

统计。

最后你可以得到非常有价值的一张图：

```text
x = # layout-sensitive decisions in subgraph

y = framework-default regret

         *
       *     *
    *           *
  *
*____________________
  1  2  3  4  5 ... decisions
```

真正要回答：

\[
\boxed{
\text{当需要联合决定的 layout 数量增加时，
现有局部规则的 regret 是否系统性增大？}
}
\]

这比现在 RQ2 的科学含量高很多。

---

# 19. 我建议主论文不要直接塞 100 个 PR，而采用“PR→case”的证据链

例如：

```text
Real PR/Issue
     ↓
抽取真实 model / shape / subgraph / layout conflict
     ↓
重现 framework default
     ↓
增加 controlled counterfactual layouts
     ↓
完整 subgraph measurement
     ↓
graph oracle
     ↓
default regret / split-domain benefit
```

这样 PR 的作用不是：

> “别人做过这个优化。”

而是：

> “这个真实 PR 证明这个 layout conflict 在 production framework 中确实存在；我们把它变成 controlled paired experiment。”

这是论文论证上更强的写法。

---

# 20. 哪些应该优先做成第一批实验

如果你资源有限，我会优先做下面 16 个，不再从 synthetic multi-consumer 起步：

| 优先级 | Framework / case | 原因 |
|---|---|---|
| P0 | vLLM #59112 packed BLHNC re-paging | 有非常完整 E2E performance，storage/kernel domains 明确 |
| P0 | vLLM #30448 P/D heterogeneous layout | 最典型 cross-engine domain |
| P0 | vLLM #52240 MoE W13 matrix | 4 类 backend layout，非 attention |
| P0 | vLLM #58766 Engram MXFP8 preservation | 多 operator + communication + quantization |
| P0 | SGLang #40326→#38592→#40330 | layout change 穿过几十个真实模块 |
| P0 | SGLang #39039 MoonEP/DeepGEMM psum | 真实 MoE graph，性能差异很大 |
| P0 | SGLang #41944 strided QKV | copy elimination 与 slower consumer tradeoff 同时存在 |
| P0 | FlashInfer #4657 | layout×tactic joint selection，有 oracle-like sweep |
| P0 | FlashInfer #5094 | data+metadata layout joint selection |
| P0 | Triton #8450/#8281 | 真正的 operator-chain propagation |
| P0 | Triton #11117 | 错误 conflict resolution 可造成64x指令放大 |
| P0 | TileLang #3180 | connected-component layout domain |
| P0 | TileLang #3284 | 单 domain 不合法、必须 split 的直接例子 |
| P0 | TRT-LLM #19541 | **真实 locality-domain split** |
| P0 | AITER #5543 | local kernel loser 可能是 graph winner |
| P0 | DeepGEMM #440 | 两 GEMM + activation pool 的真正 multi-op domain |

然后第二批补：

- FlashAttention #2667；
- CUTLASS #3030/#3453；
- TVM #20426/#20076；
- SGLang #33651/#39606；
- TRT-LLM #19358；
- AITER #4057/#4550；
- FlashInfer #5405/#5667。

这样你的证据就已经跨越：

\[
\text{serving}
+\text{compiler}
+\text{attention}
+\text{MoE}
+\text{linear attention}
+\text{distributed}
+\text{memory hierarchy}.
\]

---

# 21. 最后，原 RQ2 应该明确废弃的实验表述

不建议再把 RQ2 主结论建立在：

```text
KV writer
 ├─ head scan
 └─ token scan
```

上。

这个实验仍然可以保留，但身份应该降成：

\[
\boxed{\text{minimal controlled sanity check}}
\]

而不是 canonical RQ2 evidence。

真正 canonical evidence 应该变成：

\[
\boxed{
\text{Real multi-operator subgraph}
+
\text{framework-native default}
+
\text{controlled layout/domain counterfactuals}
+
\text{whole-subgraph measured oracle}
}
\]

我认为按照上面这批 PR/issue 来重构以后，你的 RQ2 才真正有能力回答：

> 当子图扩大、layout-sensitive operators 变多、不同 representation 跨越 multiple consumers、multiple phases、multiple memory levels 甚至 multiple GPUs 时，layout 应如何传播，layout domain 应在哪里分裂，以及现有框架默认策略距离整图最优还有多远？

而不再只是回答：

> 两个 consumer 偏好不同时，要不要做一次 NHD→HND conversion。

## 可见消息 4 — user

按照上面的要求，尽可能扒完所有框架的所有相关的pr/issue， 这样我的实验能够足够完善与充分，而不是像原来那样只测multi-consumer

## 可见消息 5 — assistant

可以。你现在真正想研究的已经不是“某一条 producer→consumer edge 应该选 NHD 还是 HND”，而是一个更一般、也更像顶会问题的：

\[
\boxed{
\text{给定一个包含多个 layout-sensitive operators 的真实子图，}
\text{layout 应该在哪些边传播、在哪里分裂成新的 domain、}
\text{以及如何联合选择整张子图的 layout？}
}
\]

这和当前 RQ2 的 `KV writer → head-scan/token-scan` 有本质区别。后者只能验证“一个 tensor 被两个 locality 不同的 consumer 使用”；你现在需要验证的是：

\[
O_1\to O_2\to O_3,\qquad
O_2\to O_4,\qquad
O_3,O_4\to O_5
\]

这样的 DAG 中，多个 layout decision 发生相互传播和冲突时，局部最优是否等于整图最优。

我把 GitHub 2025–2026 的 vLLM、SGLang、FlashInfer、Triton、CUTLASS、TVM、TileLang 相关 PR/issue 又筛了一轮。下面先给方法，然后逐类讨论最值得加入实验的 real-world case，最后给一张总表。检索快照以 2026-10-01 为准；其中有些 PR 仍然是 open，所以我会明确区分“真实工程案例”和“已经成为 framework default”。

---

# 一、RQ2 应该从 producer/consumer pair 改成“layout-constrained DAG”

假设一个子图：

```text
               T1
          O1 --------> O2
                       │ \
                    T2 │  \ T3
                       ▼   ▼
                       O3  O4
                        \  /
                         \/
                         O5
```

不要再说：

```text
Producer = O1
Consumer = O2
```

而应该把每个 tensor edge 都定义一个 layout variable：

\[
L_{T_1},L_{T_2},L_{T_3},...
\]

每个 operator 有自己的 layout contract：

\[
F_{O_i}
=
\{
(L_{in1},L_{in2},...,L_{out})
\mid O_i\text{合法}
\}
\]

例如某个算子可能允许：

\[
NHD\rightarrow NHD
\]

和

\[
HND\rightarrow HND
\]

另一个算子可能只允许：

\[
HND\rightarrow HND
\]

再一个算子可能接受 stride-polymorphic input。

因此完整目标不是：

\[
\min_L C_{\text{consumer}}(L)
\]

而是：

\[
\boxed{
\min_{\mathbf L,\mathbf D}
\left[
\sum_i C_{O_i}(\mathbf L)
+
\sum_j C_{\mathrm{conversion},j}
+
C_{\mathrm{communication}}
+
C_{\mathrm{materialization}}
+
C_{\mathrm{memory}}
\right]
}
\]

其中 \(\mathbf D\) 就是 layout-domain partition。

---

# 二、这里的“layout domain”到底应该怎么定义

我建议你不要把 domain 定义成：

> 一串 operator 使用同一个 layout。

这个定义太弱，也不能覆盖 HBM→SMEM→REG。

更合理的是：

> 一个 layout domain 表示一组共享同一物理 representation/存储实例的 tensor uses；如果某个 consumer 需要另一个 representation，就必须创建新的 domain，并显式支付 conversion / materialization / copy / communication 成本。

例如：

```text
                     KV storage domain D0
                     physical = NHD
                          │
           ┌──────────────┼──────────────┐
           │              │              │
           ▼              ▼              ▼
        O_decode      O_prefix       O_offload
       accepts NHD    wants NHD       wants HND
                                         │
                                 layout conversion
                                         │
                                         ▼
                                  HND domain D1
```

这时不是简单的：

```text
3 consumers
```

而是：

\[
D_0=\{decode,prefix\}
\]

\[
D_1=\{offload\}
\]

以及：

\[
D_0\rightarrow D_1
\]

有显式 conversion。

更复杂的是：

```text
HBM NHD
   │
   │ TMA
   ▼
SMEM swizzled
   │
   │ ldmatrix
   ▼
REG MMA fragment
   │
   │ reduction
   ▼
REG output layout
   │
   │ TMA store
   ▼
HBM row-major
```

这里**不应该强迫 HBM/SMEM/REG 使用“相同 layout”**。

应该把它理解成不同 memory-level domain：

\[
D_{\mathrm{HBM}},
D_{\mathrm{SMEM}},
D_{\mathrm{REG}}
\]

layout propagation 是跨这些 domain 的映射：

\[
L_{\rm HBM}
\xrightarrow{\text{TMA/cp.async}}
L_{\rm SMEM}
\xrightarrow{\text{ldmatrix}}
L_{\rm REG}
\]

这点对 Triton / CUTLASS / TileLang 尤其重要。

---

# 三、整图实验应该比较什么

对于一个有 \(k\) 个 layout-sensitive tensors/operators 的子图，我建议至少比较五种策略：

| 策略 | 含义 |
|---|---|
| Framework default | 原框架当前默认 layout propagation / backend policy |
| Local greedy | 每个 operator 单独选最快 layout，不考虑上下游 |
| Single-domain | 尽量整张子图维持一个公共 representation |
| Partitioned domains | 允许在若干边创建新 layout domain，并支付 conversion |
| Graph oracle | 在所有合法组合里找整图实测最优 |

真正值得报告的是：

\[
R_{\rm default}
=
\frac{T_{\rm default}-T_{\rm oracle}}
{T_{\rm oracle}}
\]

和：

\[
R_{\rm greedy}
=
\frac{T_{\rm local\ greedy}-T_{\rm oracle}}
{T_{\rm oracle}}
\]

以及：

\[
B_{\rm split}
=
\frac{T_{\rm best\ shared}}
{T_{\rm best\ partitioned}}
\]

如果：

\[
B_{\rm split}>1.03
\]

才说明“为不同 operator 创建 layout domains”有实际价值。

反过来，如果：

\[
T_{\rm shared}\approx T_{\rm partitioned}
\]

甚至 shared 更快，那么就是非常重要的 negative result：

> 虽然存在局部 layout conflict，但 conversion/materialization 成本使得 domain split 不值得。

这比当前 RQ2 的单纯 fanout sweep 强很多。

---

# 四、operator 数量变多后，不要暴力枚举所有 \(4^N\)

小子图可以 exhaustive。

例如 5 个 layout-sensitive tensors，每个 `{NHD,HND}`：

\[
2^5=32
\]

完全可以全部直接测。

但是 10 个 tensor、4 layouts：

\[
4^{10}\approx 10^6
\]

就不现实。

建议使用：

```text
framework legality
        ↓
删除非法 layout
        ↓
layout propagation constraint
        ↓
删除等价 physical mappings
        ↓
local measurement pruning
        ↓
保留 top-k layouts/operator
        ↓
enumerate / beam search / DP
        ↓
对 top candidates 直接运行完整子图
```

注意最后一步一定要：

\[
\boxed{\text{直接运行完整子图}}
\]

不能只把 operator median 相加。

cost model 可以用于搜索，但最终 scientific evidence 应来自完整 subgraph execution。

---

# 五、最重要的一点：real-world PR 里已经存在这种“整图 layout conflict”

下面这些 case 比当前 CUDA reference 的两个 scan consumer 强很多。

---

## Case 1：vLLM P/D disaggregation：Prefill 和 Decode 想要不同 KV layout

这是我认为**最适合直接升级成新 RQ2 的 case 之一**。

相关：

urlvLLM Issue #26744 — heterogeneous block sizehttps://github.com/vllm-project/vllm/issues/26744  
urlvLLM PR #30448 — prefill NHD / heterogeneous KV layouthttps://github.com/vllm-project/vllm/pull/30448

真实子图不是：

```text
writer → consumer
```

而是：

```text
Prefill attention
      │
      ▼
Prefill KV write
NHD, block=64
      │
      ▼
KV postprocess
NHD→HND
64→16 block repack
      │
      ▼
NIXL transfer
      │
      ▼
Decode KV cache
HND, block=16
      │
      ▼
Decode attention
```

PR #30448 明确提出：

```text
fwd_prefill
   ↓
KV update (NHD → HND, block_size → 16)
   ↓
send
   ↓
decoder
```

这里至少有四个 decision：

\[
L_{\rm prefill}
\]

\[
L_{\rm transfer}
\]

\[
L_{\rm decode}
\]

\[
P_{\rm block}
\]

也就是说：

\[
\boxed{
\text{layout}+\text{page/block geometry}
}
\]

已经是联合 domain problem。

你应该测试：

```text
A. HND16 throughout
B. NHD64 prefill → HND16 convert → transfer/decode
C. NHD64 prefill → NHD transfer → decode-side convert
D. NHD/HND + {16,32,64,128} block combinations
```

测量：

\[
T_{\rm prefill}
+
T_{\rm conversion}
+
T_{\rm transfer}
+
T_{\rm decode}
\]

而不是只测 conversion。

这是非常好的真实：

\[
\boxed{\text{cross-engine layout domain}}
\]

case。

---

# 六、Case 2：vLLM MiniMax-M3，backend 和 connector 发生直接 layout contract 冲突

这是非常“干净”的真实 layout conflict。

urlIssue #46204https://github.com/vllm-project/vllm/issues/46204  
urlPR #46223https://github.com/vllm-project/vllm/pull/46223

Issue 的根因就是：

```text
MiniMax M3 MSA backend
         wants
          NHD
           │
           │
KV storage ┼──────── connector
           │
           ▼
     NIXL wanted HND
```

Issue 明确指出 HND 对 M3 不是单纯“名字变化”，而是实际 stride permutation。

因此这是：

\[
\boxed{
\text{backend layout contract}
\neq
\text{connector layout contract}
}
\]

非常适合实验：

```text
                    KV
                     │
             ┌───────┴────────┐
             ▼                ▼
       MSA attention       NIXL transfer
         NHD-only            HND-prefer
```

候选：

```text
NHD shared
NHD → HND at transfer
HND storage → NHD before MSA
```

然后测完整：

```text
prefill
→ cache
→ connector
→ decode
→ MSA
```

这里你真正验证的是：

> layout domain boundary 应该放在 storage/attention 之间，还是 storage/connector 之间？

这比人为构造两个 scan consumer 强很多。

---

# 七、Case 3：vLLM packed / HMA KV——layout 已经传播到 allocator、attention、spec decode、connector、offload

这组 PR 非常重要。

urlPR #37885 — canonical HMA allocationhttps://github.com/vllm-project/vllm/pull/37885  
urlPR #50208 — preserve HMA region strideshttps://github.com/vllm-project/vllm/pull/50208  
urlPR #57169 — generic packed KV layouthttps://github.com/vllm-project/vllm/pull/57169  
urlPR #59265 — preserve packed layouts for offloadhttps://github.com/vllm-project/vllm/pull/59265

这里真实图已经变成：

```text
KV allocator
   │
   ├── physical block stride
   ├── region offset
   ├── padding
   └── packed rows
        │
        ▼
 Attention kernel
        │
        ├── sparse MLA metadata
        ├── indexer
        ├── speculative proposer
        ├── NIXL/Mooncake
        └── CPU offload
```

#57169 特别重要，因为它明确要把：

```text
physical block strides
```

传播到：

```text
sparse-MLA metadata builder
model runner
MRv2
speculative proposer
```

这就是你正在找的：

\[
\boxed{
\text{layout propagation across a larger real subgraph}
}
\]

这里 layout variable 不再只是 NHD/HND，而包括：

\[
(\text{shape},\text{stride},\text{offset},
\text{page},\text{packed-row})
\]

所以你的 RQ2 不应该限制在“layout name”。

候选可以是：

```text
per-layer scattered
canonical contiguous
packed physical row
padded canonical block
```

然后比较整个：

```text
allocation
→ attention
→ spec
→ transfer/offload
```

的总时间和峰值 memory。

---

# 八、Case 4：vLLM sparse MLA：Persistent NVFP4 layout 是否值得一直保留

urlvLLM PR #59342https://github.com/vllm-project/vllm/pull/59342

这是另一个特别适合 RQ2 的真实 case。

逻辑：

```text
persistent KV
   NVFP4
      │
      ├──────────── native decode
      │             reads NVFP4 directly
      │
      └→ gather
          │
          ▼
       stage FP8
          │
          ▼
   FlashInfer sparse MLA
```

目前两条路径：

```text
staged:
NVFP4 → gather/dequant → FP8 → attention

native:
NVFP4 ─────────────────→ attention
```

而 prefill 和 decode 的最佳策略可能不同。

所以你可以定义：

```text
D0 = persistent NVFP4 domain
D1 = temporary FP8 staging domain
```

研究：

\[
\boxed{
\text{什么时候值得产生 D1？}
}
\]

这比抽象 conversion-reuse 更真实。

---

# 九、SGLang L3 unified KV：这是一个非常完整的 memory hierarchy domain case

urlSGLang PR #33651https://github.com/sgl-project/sglang/pull/33651  
urlPR #39606 — staged page-unified writebackhttps://github.com/sgl-project/sglang/pull/39606

它明确列了四种 host layout：

```text
layer_first
page_head
page_first
page_first_direct
```

例如：

```text
layer_first:
(2, layer, page*page_size, head, dim)

page_head:
(2, page, head, page_size, layer, dim)

page_first_direct:
(2, page, layer, page_size, head, dim)
```

而且还有：

```text
page_first_direct_group:
(2,page,head_group,layer,page_size,head_in_group,dim)
```

真实图：

```text
GPU KV layout
     │
     ▼
GPU relayout kernel
     │
     ▼
staging page
     │
     ▼
D2H
     │
     ▼
Host L2 layout
     │
     ▼
Mooncake / File / L3
     │
     ▼
other TP/PP topology
```

这里至少有：

\[
L_{\rm GPU}
,L_{\rm staging}
,L_{\rm host},
L_{\rm transport}
\]

四层。

而且 PR 已经提供真实 benchmark：

2048 tokens 时大约：

```text
GPU relayout ≈ 0.34 ms
D2H ≈ 9.39 ms
```

所以这是非常典型的：

> 某个 local layout conversion 看起来有成本，但相对于下一级 transfer 可能几乎可以忽略。

这正是 domain granularity 需要研究的问题。

---

# 十、SGLang MoE W13/W31：consumer 决定 upstream weight layout

urlSGLang PR #39388https://github.com/sgl-project/sglang/pull/39388

这里不是 KV。

这非常重要，因为你需要一些**非 attention case**。

真实子图：

```text
checkpoint
    │
    ▼
weight loader
    │
    ├─ W31
    └─ W13
          │
          ▼
      quant/scales
          │
          ▼
    effective MoE backend
     /             \
TRT-LLM          MegaMoE
 wants W31        wants W13
```

PR 的 bug 就是：

> nominal runner 是 TRT-LLM，因此 loader 选了 W31；但真正的 consumer 是 MegaMoE，它要求 canonical W13。

这非常漂亮地说明：

\[
\boxed{
\text{upstream layout 不应该由名义 producer 决定，
而应该由实际 downstream consumer graph 决定}
}
\]

候选：

```text
W13 throughout
W31 throughout
W13→W31 conversion
W31→W13 conversion
interleaved W13 + deinterleave
```

整图：

```text
load
→ quant metadata
→ expert dispatch
→ grouped expert GEMM
→ SwiGLU/combine
```

都可以进入测量。

这是你论文里非常应该放的一个 case。

---

# 十一、SGLang speculative Mamba/SSM：甚至“是否 materialize layout”本身也是 domain decision

urlSGLang PR #12892https://github.com/sgl-project/sglang/pull/12892

原来：

```text
target verify
    │
    ▼
intermediate SSM state
    │
    ├── copy
    ├── scatter
    └── cache
         │
         ▼
     next kernel
```

修改后：

```text
target verify
     │
     ▼
  last_steps
     │
     ▼
SSM/conv kernel
lazy reconstruct/update
```

这已经超出了传统 layout name。

它研究的是：

\[
\boxed{
\text{materialized state domain}
\quad vs \quad
\text{lazy logical state domain}
}
\]

PR 报告某些配置 E2E 提升到 9.47%。

这个 case 可以帮助你把 RQ2 从：

> 选 NHD/HND

提升成：

> physical representation / materialization domain 应该划在哪里？

---

# 十二、SGLang DCP：layout domain 甚至可以跨 GPU

urlRFC Issue #12196https://github.com/sgl-project/sglang/issues/12196  
urlPR #14982https://github.com/sgl-project/sglang/pull/14982

这里 KV：

```text
TP baseline:
GPU0: entire KV
GPU1: entire KV

DCP:
GPU0: token 0,2,4,...
GPU1: token 1,3,5,...
```

所以 layout domain 变成：

\[
L_{\rm distributed}
=
(\text{rank},\text{token offset},\text{page})
\]

而 attention 之后还需要 online-softmax combine。

这应该作为：

\[
\boxed{\text{distributed layout domain}}
\]

单独一类，不和单卡 NHD/HND 混在一起。

PR 甚至展示了一个很好的 trade-off：

KV capacity 增加，但当时实现的 serving throughput 比 TP baseline 更低。

这正是你的科研问题：

> 局部 memory/layout 收益是否值得其 communication/compute 代价？

---

# 十三、FlashInfer #5094：这是非常理想的“两个输入分别有自己的 layout”的真实 attention case

urlFlashInfer PR #5094https://github.com/flashinfer-ai/flashinfer/pull/5094

输入不是只有 K/V：

```text
Q
│
├──────────────┐
│              │
KV cache       TopK indices
packed HND     token-major
│              │
└──────┬───────┘
       ▼
 sparse MSA
```

支持：

```text
KV:
packed HND view

TopK:
[token,head,16]
或
[head,token,16]
```

PR 直接比较：

```text
transpose + contiguous
vs
direct token-major
```

报告 eager 模式下 direct token-major 大约快 16–17%。

这正是一个：

\[
\boxed{
L_{KV}\neq L_{metadata}
}
\]

但整图最优仍然需要联合选择的 case。

不要只改 KV layout。

应该同时枚举：

\[
L_{KV}
\times
L_{topk}
\times
L_{out}
\]

---

# 十四、FlashInfer #5272：data layout 和 metadata layout 必须联合传播

urlIssue #3617https://github.com/flashinfer-ai/flashinfer/issues/3617  
urlPR #5272https://github.com/flashinfer-ai/flashinfer/pull/5272

FP8 KV：

```text
KV data
[token,head,dim]
      │
      └── scale
          [token,head]
```

scale layout 必须 mirror KV layout：

\[
stride(scale)
\sim
\frac{stride(KV)}{head\_dim}
\]

而 paged path 的 scale 甚至可能是 inline slot 的 non-contiguous view。

所以它是非常好的：

\[
\boxed{
\text{data layout + metadata layout propagation}
}
\]

case。

---

# 十五、FlashInfer paged TMA：layout 决策穿越 HBM→TMA→SMEM→MMA

urlIssue #1474https://github.com/flashinfer-ai/flashinfer/issues/1474  
urlPR #4971https://github.com/flashinfer-ai/flashinfer/pull/4971

真实结构：

```text
Paged K/V in HBM
      │
      ├── cp.async gather
      │
      └── TMA
            │
            ▼
      swizzled SMEM
            │
            ▼
          MMA
            │
            ▼
          P × V
```

候选不仅是 NHD/HND：

```text
HBM:
NHD / HND / paged variants

load:
cp.async / TMA

SMEM:
different swizzles / box geometry

compute:
MMA fragment mapping
```

PR 报告 TMA 在部分 page/shape 上约 1.17–1.40x。

这是你研究“不同计算粒度、内存层级下怎么联合选”最直接的案例之一。

---

# 十六、Triton #8450：这是“多算子 layout propagation”最典型的 compiler case

urlTriton RFC #8281https://github.com/triton-lang/triton/issues/8281  
urlPR #8450https://github.com/triton-lang/triton/pull/8450

它明确描述：

```text
Load
 ↓
Transpose
 ↓
Reshape
 ↓
Elementwise
 ↓
ConvertLayout
 ↓
Dot
```

现有路径可能：

```text
GMEM
 ↓ load
register layout A
 ↓
shared memory
 ↓ ConvertLayout
register DotOperand layout
 ↓
Dot
```

目标路径：

```text
GMEM
 ↓ direct load in required layout
REG DotOperand layout
 ↓
Transpose/reshape/etc propagate layout
 ↓
Dot
```

PR 的核心算法甚至就是：

```text
从 DotOp 向后追到 LoadOp
→ 修改 DotOperand layout
→ backward propagate
→ forward propagate
→ 删除 ConvertLayout
```

所以这个 case 几乎就是你的新 RQ2 的 compiler 版本。

你完全可以测试：

```text
default Triton
forced early layout
forced late layout
conversion at each possible boundary
global propagation
```

然后统计：

```text
#convert_layout
#shared materialization
kernel time
registers
shared memory
HBM traffic
```

---

# 十七、Triton 2026 layout propagation PR 更直接证明“局部规则会冲突”

urlPR #11685 — Improve auto layout propagationhttps://github.com/triton-lang/triton/pull/11685

里面的真实 IR：

```text
splat
 ↓
join
 ↓
transpose
 ↓
reshape
 ↓
transpose
 ↓
set_auto_layout
```

问题正是：

> layout 从一端向后传播，再向前传播，会遇到 join 两边不同 inferred layouts，出现冲突。

修复甚至引入了距离 seed 的 `distance` 来决定哪个 layout 应该占优。

这说明：

\[
\boxed{
\text{layout selection 是 graph propagation problem，
不是 operator-local problem}
}
\]

这应该成为你的核心 compiler case。

另外还有：

urlPR #11760 — absorb compatible layouts into broadcastshttps://github.com/triton-lang/triton/pull/11760  
urlPR #11780 — drop obsolete conversions after warp repartitionhttps://github.com/triton-lang/triton/pull/11780  
urlPR #11967 — masks affect coalesced layout selectionhttps://github.com/triton-lang/triton/pull/11967

它们分别体现：

```text
broadcast compatibility
warp repartition
mask/vectorization
```

都会改变 layout propagation。

---

# 十八、TileLang #3180 几乎已经在实现“layout domain connected component”

urlTileLang PR #3180https://github.com/tile-ai/tilelang/pull/3180

这个 PR 的描述非常关键：

> enumerate all implied single-fragment layouts in a connected component  
> fix one layout = fix all layouts

这几乎就是：

\[
\boxed{\text{layout domain}}
\]

的编译器定义。

它要求：

\[
\text{所有 fragment layouts bijective}
\]

\[
\text{所有 T.Parallel 只能访问自己 thread 持有的数据}
\]

同时：

\[
\min \text{total registers}
\]

真实结构：

```text
fragment A
   │
T.Parallel
   │
fragment slice
   │
T.copy
   │
fragment B
   │
reduction
```

一旦其中一个 layout 固定，connected component 其他 layout 都被限制。

如果你要验证“operator 增多以后 layout propagation 怎么做”，这个 PR 是非常值得直接复现的。

---

# 十九、TileLang #3176：框架自己已经开始用“整 component cost”选 layout

urlTileLang PR #3176https://github.com/tile-ai/tilelang/pull/3176

候选布局会考虑：

```text
global memory
local work
shuffle
barrier
shared reducer
register slots
spill
```

成本：

\[
C=
C_{\rm spill}
+
C_{\rm execution}
+
4\cdot threads_{\max}\cdot registers
\]

这里已经不是单算子 memory layout 了。

它实际上在回答：

> 某个 layout 给 memory load 带来的收益，是否值得 register replication / shuffle / barrier 的代价？

这可以成为你的：

\[
\boxed{\text{microarchitecture-level layout domain}}
\]

实验。

PR 的 weighted-pooling reproducer 某些 shape 从 1.17x 到 7x，但 PR 自己也明确说，这是 regression reproducer，不代表所有 production kernels。

这个边界要保留。

---

# 二十、TileLang #3320：跨执行 region 的值应该留在 fragment 还是落到 shared.dyn

urlTileLang PR #3320https://github.com/tile-ai/tilelang/pull/3320

这是 Ascend，但科学上非常相关。

```text
VF region A
   │
 fragment
   │
   ├── lifetime ends here
   │       → keep fragment
   │
   └── live across VF boundary
           ↓
       shared.dyn
           ↓
       reload
           ↓
       VF region B
```

这实际上是在自动决定：

\[
\boxed{
\text{register/fragment domain boundary}
}
\]

何时必须 split。

而且 memory hierarchy 是：

```text
GM / Cube
   ↕
MTE
   ↕
UB
   ↕
fragment / VF
```

这正是你问的“不同内存层级怎么选择整图 layout/domain”。

---

# 二十一、CUTLASS：更适合作为 memory-level domain，而不是 serving graph domain

比如：

urlCUTLASS PR #3030 — SM120 FlashAttentionhttps://github.com/NVIDIA/cutlass/pull/3030

同一个 FlashAttention 子图就包含：

```text
Q/K/V GMEM
    │
cp.async / TMA
    │
SMEM swizzle
    │
MMA register layout
    │
online softmax
    │
PV
    │
output
```

而且：

```text
cp.async swizzle
≠
TMA-compatible swizzle
```

所以这里可以研究：

\[
L_{\rm gmem}
\rightarrow
L_{\rm smem}
\rightarrow
L_{\rm reg}
\]

是否应该联合选择。

另一个很纯粹的 layout correctness case：

urlIssue #3612https://github.com/NVIDIA/cutlass/issues/3612  
urlPR #3660https://github.com/NVIDIA/cutlass/pull/3660

`ThrK>1` 时，不同 K-slice 的 C register/thread layout 是否应该 alias。

它适合验证：

\[
\boxed{\text{thread/value-level layout propagation}}
\]

但不是 serving-level domain。

---

# 二十二、你要求的总表

其中“默认性能”一栏只写 PR/issue 实际提供的 baseline；没有 benchmark 的我明确标成“未给性能”，避免把 correctness test 冒充性能证据。

| Framework / real case | 子图与 layout-sensitive operations | 改谁的 layout / 候选 | 为什么有 propagation/conflict | PR 原生 baseline/验证 | 建议你的 RQ2 整图实验 |
|---|---|---|---|---|---|
| vLLM url#30448https://github.com/vllm-project/vllm/pull/30448 / urlIssue #26744https://github.com/vllm-project/vllm/issues/26744 | Prefill attention → KV write → postprocess → NIXL → Decode attention | Prefill `NHD/HND`; block 16/64 等；transfer layout；decode layout | Prefill 与 decode 可各有 preferred layout/block geometry | Qwen3 accuracy；PR 没有系统 latency matrix | 枚举 `{Lp,Lx,Ld,page}`；直接测 prefill+convert+transfer+decode；计算 default/oracle regret |
| vLLM url#46223https://github.com/vllm-project/vllm/pull/46223 / url#46204https://github.com/vllm-project/vllm/issues/46204 | M3 KV writer → MSA → connector | MSA NHD；connector 曾偏 HND | backend 与 connector contract 冲突 | 主要 correctness/startup bug | 比较 shared NHD、transfer-boundary conversion、attention-boundary conversion |
| vLLM url#37885https://github.com/vllm-project/vllm/pull/37885 | allocator → hybrid full/sliding KV groups → connector | per-layer scattered / canonical contiguous | eviction group 与 transfer 希望不同物理组织 | unit tests；无完整性能矩阵 | 将 allocator+attention+connector 放进同一 pipeline 测 |
| vLLM url#50208https://github.com/vllm-project/vllm/pull/50208 | packed HMA allocation → Mooncake address construction → transfer | stride/offset/payload 分离 | logical bytes ≠ physical block stride | K3 GSM8K；writes 145.35→128.52 GiB，reads 143.02→133.05 GiB | 枚举 padded/packed/canonical；测 transfer bytes + latency + attention |
| vLLM url#57169https://github.com/vllm-project/vllm/pull/57169 | packed KV → sparse MLA/indexer → MRv2 → speculative proposer | packed physical stride vs inferred logical page stride | 一个 stride 必须传播到多个下游模块 | GB200 GSM8K correctness | 特别适合测试 propagation depth 增长后的错误和性能 regret |
| vLLM url#59265https://github.com/vllm-project/vllm/pull/59265 | layerwise mixed FP8/NVFP4 KV → CPU offload → reload → attention | packed allocator-selected vs forced LBHNC/BLHNC | offload consumer 强制 layout 会破坏 upstream packed domain | 原 baseline startup failure；patched round-trip correctness | `packed throughout` vs `canonicalize before offload` vs `canonicalize after reload` |
| vLLM url#59342https://github.com/vllm-project/vllm/pull/59342 | NVFP4 persistent KV → top-k gather → FP8 staging/native decode → sparse MLA | NVFP4 / staged FP8 | native decode 不需 conversion；prefill等路径仍需 staging | B300 correctness；GSM8K | domain=`persistent NVFP4` + optional temporary FP8 domain；按 phase 搜索 |
| SGLang url#12892https://github.com/sgl-project/sglang/pull/12892 | target verify → SSM/conv state → next speculative step | materialized intermediate state vs `last_steps` lazy state | 是否 materialize 本身就是 representation/domain decision | 最高报告约 9.47% E2E | 同时测 state-copy time、kernel、E2E、peak state bytes |
| SGLang url#14982https://github.com/sgl-project/sglang/pull/14982 / urlRFC #12196https://github.com/sgl-project/sglang/issues/12196 | KV allocation → distributed KV → local attention → online-softmax combine | TP replicated / DCP token-interleaved | memory layout 与 communication jointly determined | TP8 1586 tok/s vs DCP2TP8 1272 tok/s 的一组结果；TPOT overhead <~10% | 分布式 domain：capacity、communication、attention、E2E 联合比较 |
| SGLang url#33651https://github.com/sgl-project/sglang/pull/33651 | GPU KV → HiCache L2 → L3/Mooncake → different topology | layer_first/page_head/page_first/page_first_direct/grouped | layer/head partition 对 host layout 偏好不同 | GSM8K 94.5%；PCIe bandwidth matrix | 枚举 GPU/host/transport 三层 layout，测完整 offload/reload |
| SGLang url#39606https://github.com/sgl-project/sglang/pull/39606 | per-layer GPU KV → relayout → staging page → D2H | per-layer → page-unified | contiguous D2H 偏 page-major，GPU compute 未必 | 2048 token: relayout~0.34ms，D2H~9.39ms；~95% contiguous baseline | 很适合测“额外 domain conversion 是否值得” |
| SGLang url#38373https://github.com/sgl-project/sglang/pull/38373 | DSV4 KV representation → attention | `paged/ring` | phase/backend 对 persistent representation 的要求不同 | PR 主要接口化，无完整 benchmark | 固定同一 DSV4 workload 直接 paged vs ring + migration |
| SGLang url#38430https://github.com/sgl-project/sglang/pull/38430 | GLM NoPE cache → native FlashInfer prefill/decode | 656-byte padded / 528-byte compact row | capacity 最优与 backend support 绑定 | storage -19.5%；token slots +24.2%；未声称 serving speedup | 比较 compact throughout、padded throughout、phase split |
| SGLang url#39388https://github.com/sgl-project/sglang/pull/39388 | weight loader → quant/scales → actual MoE backend → SwiGLU | W13/W31/interleaved | nominal backend 与 effective consumer 不一致 | correctness 从严重错误恢复；性能与正确 W13 baseline 基本 parity | 非 attention 核心 case；联合 loader→expert GEMM→activation |
| SGLang url#41953https://github.com/sgl-project/sglang/pull/41953 | unified allocation → MLA/recurrent tensor views → PD transport | layer/offset/stride/TP shard descriptors | 一个 physical block 包含多种 tensor/domain | 31 tests + H200 RDMA correctness；无性能 claim | 测 whole-block shared domain vs tensor-granular transfer domains |
| FlashInfer url#4971https://github.com/flashinfer-ai/flashinfer/pull/4971 / url#1474https://github.com/flashinfer-ai/flashinfer/issues/1474 | paged HBM K/V → TMA/cp.async → SMEM → MMA | NHD/HND；cp.async/TMA；box/page；SMEM swizzle | HBM layout、load primitive、SMEM layout 联动 | 多 shape TMA 约 1.03–1.40x | memory hierarchy domain 的强 case |
| FlashInfer url#5094https://github.com/flashinfer-ai/flashinfer/pull/5094 | packed KV + top-k metadata → sparse MSA | packed HND KV；token/head-major top-k | data 与 metadata 有独立 layout | direct token-major 比 transpose+contiguous eager 快约16–17%；vLLM TPS +7.04% | 枚举 `L_KV × L_topk × L_out` |
| FlashInfer url#5272https://github.com/flashinfer-ai/flashinfer/pull/5272 / url#3617https://github.com/flashinfer-ai/flashinfer/issues/3617 | FP8 KV + scale → QK/PV | NHD/HND data + mirror scale layout | metadata stride 依赖 KV stride | correctness/layout tests；PR 描述未给完整 speed table | data+metadata joint domain |
| FlashInfer url#5447https://github.com/flashinfer-ai/flashinfer/pull/5447 | packed query + sliding-window pool + compressed pool + sparse routing → MLA | multiple pool/routing representations | 多个 persistent inputs 同时进入一个 consumer | B200 latency benchmarks | 多输入 sparse subgraph；不要只改 KV |
| FlashInfer url#5709https://github.com/flashinfer-ai/flashinfer/pull/5709 | split attention → partials → reducer | L2 load schedule / register bands / split count | upstream split layout 决定 reducer register pressure | 部分配置明显降低 reducer latency，极端有小 regression | 适合作为 producer partials→reducer domain |
| Triton url#8450https://github.com/triton-lang/triton/pull/8450 / urlRFC #8281https://github.com/triton-lang/triton/issues/8281 | Load→Trans→Reshape→Elemwise→Dot | GMEM→SMEM→REG vs direct GMEM→DotOperand REG | downstream Dot layout 应向 Load 反向传播 | RFC 对 implicit conversion 报告明显 dot-kernel gain | 编译器版“整链 propagation”第一优先级 |
| Triton url#8398https://github.com/triton-lang/triton/pull/8398 | loop/prologue/epilogue async-copy chain | pointer layout before/through `forOp` | propagation 被 loop boundary 截断会保留双份 pointer calculations | compiler regression | 比较 stop-at-loop vs cross-loop propagation |
| Triton url#11685https://github.com/triton-lang/triton/pull/11685 | splat→join→transpose→reshape→transpose | multiple Gluon encodings | forward/backward propagation 发生 equal-distance conflict | 已 merged；layout inference tests | 构造多个 anchors，比较 local seed / global seed / oracle |
| Triton url#11760https://github.com/triton-lang/triton/pull/11760 | source→broadcast→consumer | source/result layout | compatible broadcast 可吸收 conversion | 已 merged；conversion regression | measure convert count + runtime |
| Triton url#11967https://github.com/triton-lang/triton/pull/11967 | masked load→downstream compute | elements/thread/coalesced layout | pointer alignment 与 mask vectorizability 冲突 | lit 301 pass；GPU 18 pass | 将 mask 作为 propagation state，测 selected layout regret |
| CUTLASS url#3030https://github.com/NVIDIA/cutlass/pull/3030 | FlashAttention Q/K/V load→SMEM→QK→softmax→PV→store | cp.async/TMA；SMEM swizzle；register MMA | TMA swizzle 与 cp.async swizzle 合法集合不同 | 大量 D=64/128 benchmark；不同 shape winner 会变 | memory-level joint layout search |
| CUTLASS url#3660https://github.com/NVIDIA/cutlass/pull/3660 / urlIssue #3612https://github.com/NVIDIA/cutlass/issues/3612 | MMA thread partition→C fragment→epilogue copy | ThrK slice C TV layout | register/thread mapping alias conflict | 26 configs；main 16/26 vs PR 26/26 contract | thread/value-level domain legality case |
| TVM url#17599https://github.com/apache/tvm/pull/17599 | Relax graph→custom texture scope→Adreno lowering | ordinary vs texture/custom-scope layout | graph layout must propagate到特殊 memory scope | merged；相关 RFC/测试 | 非 NVIDIA cross-scope layout case |
| TVM url#20080https://github.com/apache/tvm/pull/20080 | views: unflatten/flatten/select/narrow/rearrange → FlashMLA lowering | logical views / physical ComposeLayout / swizzle | view 不 materialize 时 layout identity 必须跨 pass 保存 | B200 TIRx 2620 tests；113 workloads | 比较 view propagation vs materialize-at-boundary |
| TileLang url#1386https://github.com/tile-ai/tilelang/pull/1386 / url#1336https://github.com/tile-ai/tilelang/issues/1336 | GQA split decode → reduce → fragment/local | fragment vs local register layout | reducer fragment→local assignment layout 不合法 | bug reproduction + split output correctness | 测 fragment shared domain vs explicit adapter |
| TileLang url#1509https://github.com/tile-ai/tilelang/pull/1509 | attention/conv/MLA examples | manual shared swizzle vs auto-inferred | 移除大量显式 annotations，交给 propagation | merged | 同 kernel `manual annotations` vs `auto inference` vs oracle |
| TileLang url#1559https://github.com/tile-ai/tilelang/pull/1559 | buffer→Parallel loop→stores | buffer-derived vs PlanLoopPartition layout | correctness-equivalent layouts replication 不同 | merged；targeted test | measure register replication + kernel runtime |
| TileLang url#3176https://github.com/tile-ai/tilelang/pull/3176 | global access→reducer→shuffle/shared/barrier→registers | vector width/layout candidates | memory、communication、register costs conflict | H100 reproducer 1.17–7x；非完整 production claim | 直接比较 heuristic winner 与 exhaustive measured oracle |
| TileLang url#3180https://github.com/tile-ai/tilelang/pull/3180 | connected fragments + T.Parallel + slice/copy | connected-component fragment layouts | “fix one layout = fix all layouts” | layout correctness regressions | 最接近真正 layout-domain connected component |
| TileLang url#3320https://github.com/tile-ai/tilelang/pull/3320 | VF region→fragment→cross-region value→shared.dyn→VF | fragment local vs shared.dyn materialized | lifetime 决定 domain split | transform tests；设备性能未完整验证 | Ascend cross-memory-domain case |
| TileLang url#3328https://github.com/tile-ai/tilelang/pull/3328 | GEMM L0C→copy/cast→UB→VF | fp32 dual+VF cast vs on-path f16/bf16 FixPipe | copy layout/type conversion 与 memory level 联合 | 小 K 最多约1.3%，大 K parity | L0C→UB domain boundary case |

---

# 二十三、哪些 case 我认为最值得真正加入你的 RQ2

如果目标不是无限扩大 workload，而是得到一套有说服力、跨层次的 RQ2，我不会把上面 30 多个全部做成主实验。

我会选下面这几个“互相不重复”的类别：

```text
Serving / persistent state
    vLLM #30448       P/D NHD↔HND + block geometry
    vLLM #46204       backend vs connector direct conflict
    vLLM #57169       packed stride propagation through many modules
    SGLang #33651     GPU→Host→L3 hierarchical layout
    SGLang #39388     MoE W13/W31 non-attention case
    SGLang #12892     materialized vs lazy persistent state

Attention library
    FlashInfer #5094  KV layout + metadata layout jointly
    FlashInfer #4971  HBM→SMEM→REG hierarchy
    FlashInfer #5272  data+scale layout

Compiler
    Triton #8450      Load→Transform→Dot propagation
    Triton #11685     bidirectional layout propagation conflict
    TileLang #3180    connected-component layout domain
    TileLang #3176    cost-based domain selection

Low-level kernel
    CUTLASS #3030     TMA/cp.async + SMEM swizzle + MMA
```

这已经覆盖：

\[
\text{operator}
\]

\[
\text{multi-operator subgraph}
\]

\[
\text{persistent state}
\]

\[
\text{metadata}
\]

\[
\text{HBM/SMEM/REG}
\]

\[
\text{CPU/L3}
\]

\[
\text{distributed GPUs}
\]

而不再只是：

\[
KV\ write\rightarrow KV\ read
\]

---

# 二十四、每个框架到底应该怎么“改 layout”

这个也需要非常明确地区分。

对于 vLLM / SGLang，不要直接改底层 load/store 作为主实验。优先改框架已经暴露出来的真实 decision point：

```text
KV cache layout selector
KV cache stride order
block/page size
packed/canonical allocation
backend selection
offload layout
connector representation
DCP/TP sharding layout
```

然后用原生 serving path 测：

```text
TTFT
TPOT
throughput
KV capacity
transfer bandwidth
conversion overhead
```

对于 FlashInfer，适合直接构造：

```text
producer/layout-preparation
→ metadata generation
→ attention
→ reducer
```

保持 wrapper 和 kernel 都是真实实现，显式给：

```text
NHD/HND
paged
packed
token-major/head-major metadata
```

然后 CUDA Graph/eager 都测。

对于 Triton / TileLang / TVM，应该在 IR/layout inference 层插入实验控制：

```text
force layout anchor
disable propagation
propagate forward only
propagate backward only
global propagation
allow/forbid convert_layout
```

这样研究的是：

\[
\boxed{\text{layout propagation policy}}
\]

而不是只比较两个手写 kernel。

对于 CUTLASS，则更适合作为低层补充：

```text
GMEM layout
× TMA/cp.async
× SMEM swizzle
× MMA fragment layout
× epilogue layout
```

它回答 memory hierarchy 的 domain，而不是整个 serving graph。

---

# 二十五、最终建议你把 RQ2 重新表述成这样

原来的：

> Layout domain 应该划多细？

可以升级成：

> **RQ2: How should layout decisions be propagated and partitioned across a multi-operator subgraph?**

或者更具体：

> **When multiple operators impose different layout preferences, where should a framework preserve a shared physical representation, and where should it create a new layout domain and pay for conversion?**

然后拆三个子问题：

\[
\boxed{
RQ2.1:
\text{local layout preference 是否足以预测整图最优？}
}
\]

\[
\boxed{
RQ2.2:
\text{随着 layout-sensitive operators 数量增加，
最优 domain partition 是否发生变化？}
}
\]

\[
\boxed{
RQ2.3:
\text{HBM/SMEM/REG、GPU/CPU、跨 GPU 等不同 memory levels
是否需要不同 domain granularity？}
}
\]

最关键的实验指标则是：

\[
\boxed{
\text{Framework Default Regret}
=
\frac{T_{\rm framework}-T_{\rm graph\ oracle}}
{T_{\rm graph\ oracle}}
}
\]

这能直接回答你最后问的：

> “现有框架的默认性能如何？”

不再说“这个框架支持 NHD/HND”，而是：

> 在这个真实 PR 对应的真实子图上，框架默认 layout propagation/domain policy 距离整图最优到底差多少。

这会比当前 RQ2 的 `common_NHD/common_HND/split` 实验强一个层级。

如果以你现在仓库的实验体系继续做，我认为下一步最合理的是把这些 real-world case 统一转换成一个 **“operator-by-operator layout decision matrix”**：每个 case 明确列出 `O1...On`、每个 tensor、候选 layout、memory level、哪些 layout 是 framework-native、哪些需要 patch，以及具体应该运行哪组 baseline/counterfactual。这样基本就能直接变成你的新 RQ2 实验 manifest。

## 可见消息 6 — user

事实上，对于一个子图可能会有多个算子，如何测试当layout decision的operator/操作变多了之后， 此时会有多个producer/consumer对，怎么验证整图的layout如何选择，而不是说只考虑Producer/Consumer两种操作。此时要验证的问题是整个子图最优的layout，应该如何考虑layout domain的问题。另外上述CUDA reference由于代码不受任何框架的限制，比较灵活，但是如果我想在不同的框架中验证不同子图上的layout domain 问题要如何做呢？

也就是说，当子图不同/范围更大/结构更复杂/包含可选的layout的算子更多，应该如何验证不同算子之间的layout propagation和layout conflict， 并且现有的框架的默认性能如何呢？

**请扒一扒现有的Framework的Github仓库2025\~2026年，我希望补充的符合上述要求的, 来自real world cases, 而且来自真实的issue/pr， 并给出对应的链接，方便我找该pr。而不是简单的上述实验中的KV writer, multi-consumer scan。**
再扒一扒不同框架2025\~2026期间相关的 pr / issue， 回答对于每一个case, 改变谁的layout, 以及为什么，还有它有哪些layout可供选择？以及每一个case中有哪些算子/ld/st等微操作/甚至子图的layout在这些不同计算粒度，内存层级下怎么选一个整个子图/layout domain最优，请逐个pr/issue中的case讨论。最后
能否生成一张表，告诉我应该怎么对每一个相关的pr/issue（尽可能统计所有相关的）， 首先链接， test case， 不同框架怎么测试原有的性能，以及怎么验证？

## 可见消息 7 — assistant

不是。需要区分两个问题：

1. “RQ2 最核心、最完整的 split-domain 实验”的 producer/consumer 是什么？
2. “RQ2 所有框架的实验”是不是都用这一组 producer/consumer？

答案是：**核心 CUDA split-domain 实验基本只有一组 producer/consumer 模式；但整个 RQ2 的不同框架实验并不只有这一种。** 而且这一点正是当前 RQ2 实验覆盖上的一个重要局限。

先看最核心的 CUDA-reference RQ2。

在这里，Producer 是：

`kv_write_kernel`

它接收逻辑上的 `[token, head, dim]` 数据，把它写成指定的物理 layout：

\[
[N,H,D]
\]

或者

\[
[H,N,D]
\]

源码就是：

```cpp
physical[offset_of(layout, t, h, d, ...)] = logical_nhd[logical];
```

也就是说这里的 Producer 更准确地说是：

\[
\boxed{\text{KV-like tensor layout materializer / writer}}
\]

它不是完整的 `K projection GEMM`，也不是完整的 `RoPE → KV cache append`。它是一个受控的 synthetic/reference producer，目的是只控制 layout 这个变量。

源码：urlrq_llm_layout_bench.cu — kv_write_kernelhttps://github.com/idoOwhd/layout_Test/blob/main/rq_llm_layout_bench.cu#L87-L98

CUDA RQ2 有两个 Consumer。

第一个是：

`decode_head_scan_kernel`

它固定一个 head，然后沿着：

\[
[token,dim]
\]

扫描。

因此：

```text
head h0:
token0 → token1 → token2 → ... → tokenN
```

在 HND：

\[
[H,N,D]
\]

中，同一个 head 的所有 token 连续，所以它偏好 HND。

这可以理解成一个非常简化的：

\[
\boxed{\text{decode/head-oriented KV-cache reader}}
\]

但注意：它不是完整 attention。它只是读 KV-like tensor 后做 reduction。

源码：urlrq_llm_layout_bench.cu — decode_head_scan_kernelhttps://github.com/idoOwhd/layout_Test/blob/main/rq_llm_layout_bench.cu#L119-L139

第二个 Consumer 是：

`token_scan_kernel`

它固定 token，然后遍历：

\[
[head,dim]
\]

即：

```text
token t0:
head0 → head1 → head2 → ... → headH
```

因此 NHD：

\[
[N,H,D]
\]

更连续。

源码明确说它只是近似：

> batched producer/prefill-side traversal, not a complete attention implementation

所以它更准确的角色是：

\[
\boxed{\text{token-oriented consumer}}
\]

而不是一个具体的真实 LLM operator。

源码：urlrq_llm_layout_bench.cu — token_scan_kernelhttps://github.com/idoOwhd/layout_Test/blob/main/rq_llm_layout_bench.cu#L141-L161

因此最核心的 CUDA RQ2 实际上是：

```text
                         KV-like logical tensor
                                  │
                                  ▼
                         kv_write_kernel
                           [Producer]
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
           head-local consumer         token-local consumer
       decode_head_scan_kernel         token_scan_kernel
                    │                           │
             prefers HND                 prefers NHD
```

然后测试三种 domain：

```text
A. common_NHD

Producer → NHD
              ├── Head consumer reads NHD
              └── Token consumer reads NHD


B. common_HND

Producer → HND
              ├── Head consumer reads HND
              └── Token consumer reads HND


C. split

Producer → NHD ───────────────→ Token consumer
              │
              └→ NHD→HND → Head consumer
```

其中 `convert_kernel` 是：

\[
\boxed{\text{layout transformer / domain bridge}}
\]

严格来说它既不是原始 Producer，也不是最终 Consumer。

---

## 但是，整个 RQ2 并不只有这一种 producer/consumer

这一点非常重要。

不同框架实际上用了不同的“访问冲突抽象”。

| 实验 | Producer | Consumer 1 | Consumer 2 | 是否真正测 split |
|---|---|---|---|---|
| CUDA-reference | `kv_write_kernel` | `decode_head_scan_kernel` | `token_scan_kernel` | **是** |
| Triton | `store_layout_kernel` | `qk_layout_kernel` | `token_layout_kernel` | **否** |
| TVM | 无 timed producer；input 已物化 | row reduction | column reduction | **否** |
| vLLM/SGLang KV request-batch | native KV writer | 没有隔离 consumer | 没有 | **否** |
| vLLM/SGLang whole serving | 整个模型内部很多 producer | 整个 serving graph | 不单独隔离 | **不是 RQ2 split 因果实验** |

这里尤其值得看 Triton，因为它和 CUDA 已经不是完全一样的 consumer 了。

### Triton：Producer 仍是 layout writer，但 head consumer 变成了真实一点的 QK

Triton 的 Producer 是：

`store_layout_kernel`

作用仍然类似 CUDA 的 `kv_write_kernel`：

\[
logical[N,H,D]\rightarrow physical\ layout
\]

候选甚至不只有 NHD/HND，还有：

- `NHD`
- `HND`
- `paged_NHD`
- `paged_HND`

源码：urltriton_kv_layout_bench.py — store_layout_kernelhttps://github.com/idoOwhd/layout_Test/blob/main/triton_kv_layout_bench.py#L34-L61

但 Triton 的第一个 Consumer 不再只是 scan，而是：

`qk_layout_kernel`

它真的执行：

\[
score_{h,n}
=
\sum_d Q_{h,d}K_{n,h,d}
\]

也就是一个简化后的：

\[
\boxed{QK^T\ attention-score consumer}
\]

这比 CUDA 的 `decode_head_scan_kernel` 更接近真实 attention consumer。

源码：urltriton_kv_layout_bench.py — qk_layout_kernelhttps://github.com/idoOwhd/layout_Test/blob/main/triton_kv_layout_bench.py#L64-L98

第二个 Consumer 是：

`token_layout_kernel`

它固定 token，对：

\[
[head,dim]
\]

做 reduction。

仍然是一个 synthetic token-oriented consumer。

所以 Triton 实际是：

```text
             store_layout_kernel
                 Producer
                    │
           ┌────────┴─────────┐
           ▼                  ▼
   qk_layout_kernel     token_layout_kernel
     QK consumer        token-local consumer
   head/KV oriented         N-oriented
```

这是和 CUDA 不完全相同的 producer/consumer pair。

但有一个关键问题：

Triton RQ2 只测：

```text
common_NHD
common_HND
common_paged_NHD
common_paged_HND
```

以及 7 个 fanout。

它没有：

```text
NHD copy
  ├── NHD consumer
  └── conversion → HND consumer
```

这样的 split-domain candidate。

所以 Triton 提供的是：

\[
\boxed{\text{multi-consumer common-layout preference}}
\]

而不是完整的：

\[
\boxed{\text{shared domain vs split domain}}
\]

源码也明确写的是 `strategy = f"common_{layout}"`。urlTriton multi-consumer RQ2 loophttps://github.com/idoOwhd/layout_Test/blob/main/triton_kv_layout_bench.py#L214-L240

---

## TVM 又是第三种 producer/consumer

TVM 的 RQ2 实际上根本没有 timed producer。

输入 `A` 已经提前被构造成：

- `row_major`
- `tiled_16x16`

然后让两个 consumer 读取。

Consumer 1：

\[
\text{row reduction}
\]

即：

\[
out[r]=\sum_c A[r,c]
\]

Consumer 2：

\[
\text{column reduction}
\]

即：

\[
out[c]=\sum_r A[r,c]
\]

源码：urlTVM build_multi_consumerhttps://github.com/idoOwhd/layout_Test/blob/main/tvm_rq_observation_bench.py#L114-L139

所以它构造的冲突是：

```text
                pre-materialized A
                       │
             row-major / tiled
                       │
              ┌────────┴────────┐
              ▼                 ▼
         Row reducer       Column reducer
        row-oriented      column-oriented
```

这不是 KV cache，也不是 attention。

它是在验证一个更抽象的问题：

\[
\boxed{
\text{两个访问轴偏好不同的 consumer 是否会要求不同 layout？}
}
\]

而且 TVM 的结果里明确记录：

```python
"producer_materialization_timed": False
```

所以你不能把 TVM 说成：

> “直接验证 producer → split-layout → two consumers。”

它没有完整测 producer。

源码：urlTVM RQ2 multi-consumer timinghttps://github.com/idoOwhd/layout_Test/blob/main/tvm_rq_observation_bench.py#L227-L253

---

## vLLM / SGLang 更不一样

vLLM/SGLang 的 `KV request batch` 实验只直接测了原生 KV writer。

这里：

- Producer：框架自己的 native KV writer/cache materialization；
- Consumer：**没有在这个实验中直接测**。

仓库报告自己就写得很清楚：

> 本实验调用框架原生 KV writer，证明 cache materialization 局部边界；完整 attention read 和调度器影响由另一个 native serving 并发实验验证。

并且对 RQ2：

> 没有 producer→consumer 或 domain-split 反事实。

源码：urlanalyze_kv_request_batch.py — evidence boundaryhttps://github.com/idoOwhd/layout_Test/blob/main/analyze_kv_request_batch.py#L89-L127

所以这是 supporting evidence，不能算一个新的完整 producer/consumer pair。

---

# 那么，“RQ2 的完整实验中只有一种 Producer/Consumer 吗？”

如果你所谓“完整实验”是指：

> 真正比较 `common layout` 和 `split layout + conversion` 的 causal experiment

那么答案基本是：

\[
\boxed{\textbf{是，目前实际上只有 CUDA-reference 那一类。}}
\]

也就是：

\[
\boxed{
KV\ writer
\rightarrow
\{
head\text{-}local\ consumer,\;
token\text{-}local\ consumer
\}
}
\]

只有它真的包含：

```text
common_NHD
common_HND
split_NHD_HND_with_conversion
```

---

如果你的问题是：

> RQ2 所有实验里的 Producer/Consumer 是否都完全一样？

那么答案是：

\[
\boxed{\textbf{不是。}}
\]

大致有三种真正不同的 consumer conflict：

1. CUDA：`KV writer → head scan + token scan`
2. Triton：`KV writer → QK attention + token reduction`
3. TVM：`pre-materialized tensor → row reduction + column reduction`

而 vLLM/SGLang 目前主要只提供 native writer / serving 辅助证据。

---

## 这也暴露了当前 RQ2 一个很明显的实验缺口

我认为这个点对你后面完善实验很重要。

当前 L-RQ2 在理论上声称研究的是：

\[
\text{Optimal layout-domain granularity}
\]

这是一个很普遍的问题，例如真实 LLM 里可以出现：

```text
KV cache
 ├── decode attention
 ├── speculative verification
 ├── prefix-cache copy
 └── offload / migration
```

或者：

```text
MoE dispatch buffer
 ├── grouped GEMM
 ├── combine
 └── routing / metadata consumer
```

又或者：

```text
SwiGLU intermediate
 ├── activation
 └── down projection
```

但当前真正做了：

\[
\text{shared domain vs split domain + conversion}
\]

完整对比的，实际上仍然主要只有：

```text
KV-like writer
   ↓
head-local vs token-local consumers
```

也就是说，目前实验很好地验证了：

\[
\boxed{
\text{“两个 locality preference 相反的 consumers 会不会值得 split domain？”}
}
\]

但还没有充分验证：

\[
\boxed{
\text{“不同真实 operator 类型之间是否普遍存在 layout-domain granularity tradeoff？”}
}
\]

这两个命题强度是不一样的。

如果你要把 RQ2 做得更像一篇顶会论文，我会建议至少再增加几类**真实 producer→multi-consumer graph**，例如：

| 子图 | Producer | Consumer A | Consumer B | 可比较 layout-domain |
|---|---|---|---|---|
| KV cache | KV append/RoPE store | decode QK | cache copy/prefix reuse | NHD/HND/paged |
| SwiGLU | gate/up GEMM | SiLU× | down GEMM | separate/packed/interleaved |
| MoE | dispatch/pack | grouped expert GEMM | combine/routing | token-major/expert-major |
| Sparse attention | top-k/index producer | gather K/V | sparse attention | token/page/segment-major |
| MLA | latent KV write | latent attention | expand/dequant consumer | latent/expanded/paged |
| Mamba/linear attention | state update | scan | checkpoint/speculative state reader | sequence/head/page-major |

这样 RQ2 才能从现在的：

\[
\text{“一个精心构造的 KV multi-consumer locality conflict”}
\]

提升到：

\[
\text{“跨真实 LLM subgraph 的 layout-domain granularity 规律”}
\]

这是我认为当前仓库 RQ2 最值得补强的地方。

## 可见消息 8 — user

RQ2中的Producer和consumer分别是什么，所有的测试中，只有这样一种类型的producer/consumer吗？

## 可见消息 9 — assistant

我直接按这个仓库当前 `main` 分支的代码追了一遍。先澄清一个容易混淆的地方：仓库里存在两套“RQ2”表述。较早的 `SCIENTIFIC_QUESTIONS.md` 里的 RQ2 是“layout 是否应随 prefill/decode、batch、context、page size 动态选择”；而当前 v13 / adequacy-v3 里真正执行和裁决的 **L-RQ2** 是：

> **Optimal layout-domain granularity：一个 tensor 被多个访问模式不同的 consumer 使用时，应该所有 consumer 共用一个 layout，还是拆成多个 layout domain，并支付一次 layout conversion？**

你最近一直讨论的“layout domain 应该划多细”对应的是后者。下面我按当前 **L-RQ2** 解释。

---

## 一、RQ2 实际想验证什么

核心冲突非常具体。

假设有一份 KV-like tensor：

\[
K[t,h,d]
\]

仓库构造两种物理布局：

\[
\text{NHD}: [N,H,D]
\]

和

\[
\text{HND}: [H,N,D]
\]

源码里的地址公式就是：

```cpp
NHD: ((token * heads + head) * dim + d)
HND: ((head * tokens + token) * dim + d)
```

源码：[rq_llm_layout_bench.cu](https://github.com/idoOwhd/layout_Test/blob/main/rq_llm_layout_bench.cu#L50-L64)

然后故意构造两个访问偏好相反的 consumer。

`decode_head_scan_kernel` 是 **head-local consumer**：固定一个 head，连续扫描很多 token × dim。HND 下，同一个 head 的 token 都连续，因此 HND 天然适合它。源码注释直接写了：

> HND is physically contiguous; NHD jumps over other KV heads.

见 [rq_llm_layout_bench.cu](https://github.com/idoOwhd/layout_Test/blob/main/rq_llm_layout_bench.cu#L119-L139)。

另一个 `token_scan_kernel` 是 **token-local consumer**：固定一个 token，遍历所有 head × dim。NHD 下这一整行连续，因此 NHD 天然适合它。源码也明确说这个 kernel：

> approximates a batched producer/prefill-side traversal, not a complete attention implementation.

见 [rq_llm_layout_bench.cu](https://github.com/idoOwhd/layout_Test/blob/main/rq_llm_layout_bench.cu#L141-L161)。

所以 RQ2 的本质是：

\[
\boxed{
\text{同一个 tensor 同时面对 HND-preferring consumer 和 NHD-preferring consumer 时，怎么办？}
}
\]

不是单纯问“NHD 还是 HND 更快”。

---

## 二、最核心的实验：直接测 3 种 layout-domain 方案

当前 adequacy-v3 版本最重要的一点，是它不再只把几个 primitive 的时间加起来，而是**直接运行完整 producer → 所有 consumers pipeline**。

核心代码在这里：

[rq_llm_layout_bench.cu, RQ2 multi-consumer](https://github.com/idoOwhd/layout_Test/blob/main/rq_llm_layout_bench.cu#L502-L557)

它比较三个策略：

| 策略 | Producer | Head-local consumer | Token-local consumer | 是否转换 | 含义 |
|---|---|---|---|---|---|
| `common_NHD` | 直接写 NHD | 从 NHD 读 | 从 NHD 读 | 无 | 全部 consumer 共用 NHD domain |
| `common_HND` | 直接写 HND | 从 HND 读 | 从 HND 读 | 无 | 全部 consumer 共用 HND domain |
| `split_NHD_HND_with_conversion` | 先写 NHD | 从转换出的 HND 读 | 继续从原 NHD 读 | `NHD→HND` 一次 | 拆成两个 layout domain |

三个路径实际执行的是：

\[
T_{\text{common-N}}
=
T_{\text{write-N}}
+
F_H T_{\text{head}}(N)
+
F_N T_{\text{token}}(N)
\]

\[
T_{\text{common-H}}
=
T_{\text{write-H}}
+
F_H T_{\text{head}}(H)
+
F_N T_{\text{token}}(H)
\]

而 split 是：

\[
T_{\text{split}}
=
T_{\text{write-N}}
+
T_{\text{convert }N\rightarrow H}
+
F_H T_{\text{head}}(H)
+
F_N T_{\text{token}}(N)
\]

这里最关键的是：**这三个式子只是帮你理解，并不是实验真正用这些 primitive p50 相加。**

代码真正做的是：

```cpp
auto multi_split = measure([&](cudaStream_t stream) {
    kv_write(... NHD);
    convert(... NHD, HND);

    for (...) decode_head_scan(... HND);
    for (...) token_scan(... NHD);
});
```

也就是说从 producer、conversion 到所有 consumer kernels 都放在一个 timed region 里直接执行。

这比早期的 cost-model 更可信。

---

## 三、它怎么改变“consumer heterogeneity”

RQ2 不是只测一个 head consumer + 一个 token consumer。

当前版本显式扫 7 种 fanout：

\[
(F_H,F_N)
\in
\{
(1,8),(1,4),(1,2),(1,1),(2,1),(4,1),(8,1)
\}
\]

源码：

[rq_llm_layout_bench.cu#L506-L509](https://github.com/idoOwhd/layout_Test/blob/main/rq_llm_layout_bench.cu#L506-L509)

其中：

- \(F_H\)：head-local consumer 执行多少次；
- \(F_N\)：token-local consumer 执行多少次。

例如：

\[
(1,8)
\]

表示同一 producer 输出被：

- head-local consumer 用 1 次；
- token-local consumer 用 8 次。

这时你直觉上会认为 NHD 更重要。

反过来：

\[
(8,1)
\]

则 head-local 路径占主导，HND 的价值应该增加。

所以它真正改变的不是 tensor shape，而是：

\[
\boxed{\text{同一 tensor 的不同 consumer 对总执行时间的重要性}}
\]

这正对应 layout-domain granularity 的核心变量。

---

## 四、所谓“测哪些东西”，可以分成四层

第一层是 primitive。CUDA benchmark 单独测：

`kv_write(NHD/HND)`、`decode_head_scan(NHD/HND)`、`token_major_scan(NHD/HND)`、`NHD→HND` 和 `HND→NHD` conversion。

例如 conversion 在这里：

[rq_llm_layout_bench.cu#L452-L465](https://github.com/idoOwhd/layout_Test/blob/main/rq_llm_layout_bench.cu#L452-L465)

这些 primitive 主要用于理解“为什么”某个 domain 策略赢，以及早期 measured-cost model。

第二层是 **真正的 L-RQ2 直接实验**：

`weighted_multi_consumer_pipeline`

对每个 shape、每个 fanout，都直接比较：

`common_NHD`、`common_HND`、`split_NHD_HND_with_conversion`。

第三层是跨 fanout 的策略稳定性。严格分析器把每个：

\[
(\text{case}, F_H,F_N)
\]

的三个策略进行比较，然后问：

“同一个 case 从 `(1,8)` 变到 `(8,1)` 时，winner 是否改变？”

对应源码：

[analyze_rq_full_per_rq.py#L161-L190](https://github.com/idoOwhd/layout_Test/blob/main/analyze_rq_full_per_rq.py#L161-L190)

第四层更重要：它还比较

\[
\text{一个固定 domain 策略}
\]

与

\[
\text{每个 fanout 单独选 oracle}
\]

之间的 regret。

代码实际上计算：

\[
T_{\text{oracle}}
=
\sum_f
\min_s T(s,f)
\]

然后对每一个固定策略 \(s\)：

\[
T_{\text{fixed}}(s)
=
\sum_f T(s,f)
\]

最终：

\[
R
=
\frac{\min_s T_{\text{fixed}}(s)}
{T_{\text{oracle}}}
\]

如果：

\[
R\ge1.03
\]

说明“一个 layout-domain 方案固定到底”相对 context-aware domain selection 至少损失 3%。

这比单纯统计 winner 有没有变化更科学。

---

## 五、怎么计时

CUDA 的 `measure()` 使用 CUDA events。

每个策略：

先 warmup，然后每次：

```text
cudaEventRecord(start)
    整个 producer → consumer pipeline
cudaEventRecord(stop)
cudaEventElapsedTime(...)
```

最后报告：

- p20
- p50
- p80

见：

[rq_llm_layout_bench.cu#L289-L317](https://github.com/idoOwhd/layout_Test/blob/main/rq_llm_layout_bench.cu#L289-L317)

adequacy runner 默认配置：

- warmup = 8
- iterations = 30
- independent process repetitions = 5

而且 5 个进程使用不同 `order-seed`，改变候选运行顺序，降低固定次序带来的 cache / thermal / DVFS 偏差。

见：

[run_cuda_reference_repetitions.py#L18-L30](https://github.com/idoOwhd/layout_Test/blob/main/run_cuda_reference_repetitions.py#L18-L30)

严格分析里不是“选五次最快的一次”，而是先对重复进程聚合，再比较策略，并使用 **3% practical-effect gate**。

也就是说：

\[
\frac{T_{\text{second}}}{T_{\text{best}}}<1.03
\]

原则上不会被当成有意义的 winner 差异。

---

## 六、shape 是怎么来的

这一点非常重要：不是随手编几个 `(N,H,D)`。

最新 adequacy-v3 先从真实模型 catalog 选 parent architecture，再对 workload shape 做 controlled counterfactual。

每种 structure 默认 128 个 contract，总共 8 类 structure：

\[
8\times128=1024
\]

所以单独跑 `L-RQ2 --full` 时，选择文件里会看到：

- selected contracts = 1024；
- attention/KV contracts = 256。

这与你之前看到的数字是一致的。

shape sweep 中 batch 明确覆盖：

\[
B\in\{1,2,4,8\}
\]

prefill length 会围绕：

7、8、15、16、17、31、32、33、63、64、65，以及更长长度。

decode KV length 则有：

31、32、33、127、128、129，以及 511/1023/2047/4095/8192 等。

源码：

[build_rq_adequacy_suite.py#L37-L87](https://github.com/idoOwhd/layout_Test/blob/main/build_rq_adequacy_suite.py#L37-L87)

这里还特意覆盖：

\[
2^k-1,\quad 2^k,\quad 2^k+1
\]

用来打到 tile/page boundary。

---

# 七、但是这里有一个非常重要的实验范围限制

虽然 `L-RQ2` 在 manifest 层被标到了 1024 个 contract 上，但是**核心 CUDA/Triton NHD/HND multi-consumer 实验并没有真的跑全部八种子图**。

`write_attention_tsv()` 明确只接受：

```python
{"gqa", "sliding_attention"}
```

见：

[build_rq_llm_cases.py#L101-L123](https://github.com/idoOwhd/layout_Test/blob/main/build_rq_llm_cases.py#L101-L123)

因此最新 adequacy-v3 中：

\[
1024\text{ selected contracts}
\]

不等于

\[
1024\text{ direct RQ2 split-domain measurements}
\]

真正进入 CUDA/Triton KV layout benchmark 的是其中大约：

\[
128_{\rm GQA}+128_{\rm sliding}=256
\]

个 attention contracts。

而且 batch 在 CUDA/Triton 中被 flatten 成：

\[
N=
B\times
\begin{cases}
Q,&\text{prefill}\\
KV,&\text{decode}
\end{cases}
\]

因此 CUDA/Triton 本身**没有显式保留 multi-request slot/page topology**。

代码甚至明确注释：

> request-aware page/slot topology is tested by the native KV runner.

这也是为什么仓库后来又加了 vLLM/SGLang request-batch 实验。

---

# 八、到底在哪些框架上测了？

这里最好分“真正直接回答 RQ2”和“辅助证据”。

### CUDA-reference：RQ2 的主实验

这是最完整、因果最清楚的。

它有：

\[
\boxed{
common\_NHD
\quad vs\quad
common\_HND
\quad vs\quad
split+conversion
}
\]

并且有 7 个 fanout ratio。

所以它真正回答：

> consumer heterogeneity 改变后，共享 domain 还是 split domain 更好？

这是 L-RQ2 最核心的直接证据。

---

### Triton：直接测 fanout，但没有 split-domain candidate

Triton 也直接做 7 个 fanout：

```python
(1,8), (1,4), (1,2), (1,1), (2,1), (4,1), (8,1)
```

而且注释明确说：

> Every row times producer + all consumer launches; this is not a token-count cost model.

见：

[triton_kv_layout_bench.py#L214-L240](https://github.com/idoOwhd/layout_Test/blob/main/triton_kv_layout_bench.py#L214-L240)

但它生成的是：

```python
strategy = f"common_{layout}"
```

即对一个 layout：

\[
producer(layout)\rightarrow
head\ consumers(layout)+token\ consumers(layout)
\]

它没有实现 CUDA 那种：

\[
NHD\ producer
\rightarrow
\begin{cases}
HND\ head\ consumer\\
NHD\ token\ consumer
\end{cases}
\]

的 `split_NHD_HND_with_conversion`。

所以：

\[
\boxed{\text{Triton 是 RQ2 的独立 common-layout fanout 复核，但不是完整 domain-split 验证。}}
\]

---

### TVM：真正执行 multi-consumer module，但仍缺完整 producer→split pipeline

TVM 现在也比早期版本强了。

它构造：

- row-oriented consumer；
- column-oriented consumer；

二者访问偏好相反。

直接测 fanout：

\[
(1,4),(1,1),(4,1)
\]

并比较：

- `row_major`
- `tiled_16x16`

源码：

[tvm_rq_observation_bench.py#L227-L253](https://github.com/idoOwhd/layout_Test/blob/main/tvm_rq_observation_bench.py#L227-L253)

而且是一整个 TVM module 直接执行，不是简单地：

\[
T_1+T_2
\]

但是它自己在结果中明确记录：

```python
"producer_materialization_timed": False
```

所以 TVM 实验验证的是：

> consumer mix 改变时，共享 input layout 的 preference 会不会改变。

它没有完整测：

\[
producer
+
split-storage
+
conversion
+
two consumers
\]

因此不能等同 CUDA 的完整 domain-split RQ2。

---

### vLLM / SGLang：只有 supporting evidence，不是真正 split-domain RQ2

这是最容易被误读的地方。

仓库的实验 registry 写得很清楚。

对 vLLM：

> `missing_adapter: requires two coexisting cache groups/layout domains in one run.`

对 SGLang：

> `missing_adapter: page-size experiments are not equivalent to split layout domains.`

见：

[rq_experiment_registry.json#L21-L28](https://github.com/idoOwhd/layout_Test/blob/main/rq_experiment_registry.json#L21-L28)

后来仓库又增加了 native KV request-batch sweep。

它调用框架原生 KV writer，并比较 request batch 和 slot policy。

但报告自己明确限定：

> L-RQ1/L-RQ2（支持）：观察固定 writer layout 跨 request batch 的局部成本；**没有 producer→consumer 或 domain-split 反事实。**

见：

[analyze_kv_request_batch.py#L89-L127](https://github.com/idoOwhd/layout_Test/blob/main/analyze_kv_request_batch.py#L89-L127)

所以千万不要写成：

> “RQ2 已经在 vLLM 和 SGLang 上完整验证。”

准确说法应该是：

\[
\boxed{
\text{vLLM/SGLang 为 request-batch / native-KV supporting evidence；
真正的 split-domain 因果实验主要在 CUDA-reference。}
}
\]

---

## 九、CUTLASS 和 Hexcute 呢？

当前 RQ2 runner 不把 CUTLASS 作为 L-RQ2 核心 adapter。

`run_persuasive_real_world_rqs_gpu1.sh` 中 CUTLASS 对应的是：

L-RQ1、L-RQ3、L-RQ4、L-RQ7、L-RQ9，

不包括 L-RQ2。

见 [run_persuasive_real_world_rqs_gpu1.sh](https://github.com/idoOwhd/layout_Test/blob/main/run_persuasive_real_world_rqs_gpu1.sh#L327-L332)。

Hexcute 则因为实验机 A10 / sm86 与公开 artifact 的 A100/H100 目标不一致，被明确标为 architecture blocked，也不是 RQ2 的 runtime 实验。

---

# 十、早期 v13 RQ2 还有一套“后处理模型”，不要和最新直接 fanout 实验混在一起

这是你前面问过的：

\[
C(\text{shared-NHD})
=
wC_{\text{head}}(NHD)
+(1-w)C_{\text{token}}(NHD)
\]

\[
C(\text{shared-HND})
=
wC_{\text{head}}(HND)
+(1-w)C_{\text{token}}(HND)
\]

\[
C(\text{split})
=
wC_{\text{head}}(HND)
+(1-w)C_{\text{token}}(NHD)
+C_{\text{convert}}
\]

其中：

\[
w=0,0.1,\ldots,1
\]

这是 `analyze_v13_rqs.py` 里的老 RQ2 adjudication。

这里 `C_head`、`C_token`、`C_convert` 都是**实测 primitive**，但是三者的组合是后处理计算：

\[
\boxed{\text{measured-cost model}}
\]

不是一条实际 pipeline 的直接计时。

另外它还做：

\[
C_{\text{convert}}\times
\{0,0.25,1,4,16\}
\]

来模拟 storage/allocator overhead 改变后的 domain winner。

这也是为什么你之前看到“H2.5 allocator overhead”时我强调：**它没有真的测 allocator latency，只是拿 measured conversion cost 做 sensitivity sweep。**

而 adequacy-v3 后来增加 7 个直接 fanout，就是为了补这个弱点。

---

# 十一、如果把整个 RQ2 实验画成最简单的结构，就是这样

```text
                  logical KV
                      │
              ┌───────┴────────┐
              │    Producer     │
              └───────┬────────┘
                      │
      ┌───────────────┼────────────────┐
      │               │                │
      ▼               ▼                ▼
 common_NHD       common_HND        split domain
      │               │                │
      │               │           NHD storage
      │               │            │      │
      │               │            │      └── token consumer × FN
      │               │            │
      │               │        NHD→HND conversion
      │               │            │
      │               │            └── head consumer × FH
      │               │
 head × FH        head × FH
 token × FN       token × FN
      │               │
      └──────┬────────┘
             ▼
         total p50
```

然后：

```text
(FH,FN)
   ↓
(1,8)
(1,4)
(1,2)
(1,1)
(2,1)
(4,1)
(8,1)
   ↓
每个 fanout 选 oracle
   ↓
与一个固定 domain 策略比较
   ↓
fixed-policy regret
```

---

## 十二、所以你写论文时，RQ2 最准确的实验描述应该是什么

可以概括成：

> L-RQ2 通过构造两个具有相反 locality preference 的消费者来研究 layout-domain 粒度。CUDA reference 对同一真实模型派生的 attention tensor contract，直接计时三条完整 producer→consumer 路径：统一 NHD、统一 HND，以及 NHD producer 后一次转换为 HND、再分别供 token-local 与 head-local consumer 使用的 split-domain 路径。实验将两个 consumer 的 fanout 从 1:8 扫到 8:1，以改变 consumer heterogeneity，并比较 context-specific oracle 与单一固定 domain 策略的 regret。Triton 直接复核 shared-layout fanout，TVM 直接运行 row/column multi-consumer modules；vLLM/SGLang 仅提供 native KV writer / request-batch supporting evidence，因为当前 harness 没有在一次真实 serving run 中创建两个共存、layout 不同的 KV domains。

其中最重要的一句话其实是：

\[
\boxed{
\text{RQ2 主要不是“比较 NHD/HND”，而是在测：
consumer mix 改变时，是否值得为不同 consumers 创建独立 layout domain。}
}
\]

另外，我检查源码后认为之前有一个地方需要修正：如果把“RQ2 已经在 CUDA、Triton、TVM、vLLM、SGLang 上都完整测了”这么表述，是不准确的。**完整含 split+conversion 的直接 RQ2，目前主要是 CUDA-reference；Triton、TVM 是部分机制复核，vLLM/SGLang 是 supporting native evidence。** 这是你后面写实验章节时最好严格区分的证据层级。

## 可见消息 10 — user

[https://github.com/idoOwhd/layout_Test.git](https://github.com/idoOwhd/layout_Test.git) 这个仓库中RQ2是怎么做实验的，测了哪些东西，怎么测的？在什么框架上测的？
