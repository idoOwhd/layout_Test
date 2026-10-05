# RQ1 共享讨论完整可见回答归档（去重）

来源：https://chatgpt.com/share/6abe1dff-8264-83ec-a1dc-8247f4819946
HTML SHA256：31a6d2b66b7f77dd5bc35a327088ce87cdd13ad6092971d60dcd8b3e66534f22
唯一可见 assistant final 回答：19；唯一 GitHub PR/issue URL：173。
这是共享讨论原文，不是逐项实验已完成的声明。工具/搜索结果流、内部 reasoning 不作为用户可见来源。

有，而且如果按我们前面筛过的 2025–2026 真实 issue/PR 来看，已经有足够多的 case 完全不是最初 v10/v13 那种：

\[
\boxed{\text{KV write}\rightarrow \text{head-local KV scan}}
\]

甚至可以把新增实验分成三组：

1. 真实 **operator-level / subgraph-level** layout conflict；
2. **跨 memory level** 的 representation/layout conflict；
3. KV 仍然存在，但 consumer 已经变成真实 Attention，而不是 synthetic local scan。

其中第 1、2 组已经足够让 RQ1 摆脱“只是 KV read/write benchmark”的问题。

---

## 一、首先：哪些 case 已经完全不是 `KV write → local KV scan`

我建议把下面这些视为最重要的非原始-KV cases。

| Case | Producer → Consumer | 类型 | 真实来源 | 与原始 KV benchmark 的区别 |
|---|---|---|---|---|
| C1 | Attention Value BMM → O-Proj GEMM | Operator→Operator | [SGLang #34498](https://github.com/sgl-project/sglang/pull/34498) | 两端都是计算算子，不是 cache read/write |
| C2 | DeepEP Dispatch → Grouped Expert GEMM | Operator→Operator | [SGLang #37261](https://github.com/sgl-project/sglang/pull/37261) | MoE activation representation |
| C3 | MoE Prepare → FusedMoE Expert | Operator→Operator | [vLLM #54109](https://github.com/vllm-project/vllm/pull/54109) | Standard/Padded/Batched expert activation |
| C4 | Expert GEMM2 → Cross-GPU Combine | Operator→Communication | [FlashInfer #5208](https://github.com/flashinfer-ai/flashinfer/pull/5208) | GEMM epilogue layout影响通信 |
| C5 | QKV Projection → KDA Prefill | Operator→Operator | [FlashInfer #5846](https://github.com/flashinfer-ai/flashinfer/pull/5846) | packed/strided QKV，不涉及 KV-cache scan |
| C6 | KDA Prefill → KDA Decode State | Subgraph/Stage | [SGLang #34299](https://github.com/sgl-project/sglang/pull/34299) | recurrent state layout，不是 Attention KV |
| C7 | MLA Projection → K/V Pack → MLA Attention | Operator/Subgraph | [FlashInfer #5667](https://github.com/flashinfer-ai/flashinfer/pull/5667) | projection output representation |
| C8 | Host Expert Cache Fill → Marlin MoE GEMM | Memory producer→Operator | [vLLM #56177](https://github.com/vllm-project/vllm/pull/56177) | weight representation，而不是 activation/KV |
| C9 | non-contiguous View → Matmul | Framework Operator | [PyTorch #197100](https://github.com/pytorch/pytorch/pull/197100) | stride/view vs contiguous materialization |
| C10 | Dot1 → FFMA/Bias → Dot2 | Compiler Operator | [Triton #10342](https://github.com/triton-lang/triton/issues/10342) | register/MMA encoding conflict |
| C11 | GMEM Load → MFMA/Dot | Micro-op | [Triton #8450](https://github.com/triton-lang/triton/pull/8450), [#7968](https://github.com/triton-lang/triton/pull/7968) | HBM/LDS/register hierarchy |
| C12 | MMA accumulator/TMEM → Epilogue | Micro-op | [CUTLASS #3313](https://github.com/NVIDIA/cutlass/pull/3313) | TMEM/register fragment layout |

所以答案是明确的：

\[
\boxed{\text{有足够多，而且可以让非-KV case 成为 RQ1 主体的一半以上。}}
\]

---

# 二、哪些最适合验证“Operator-level Layout Preference”

这里的标准是：

> Producer 和 Consumer 都是有明确 LLM 语义的真实算子，而不是单纯 memory copy/load/store。

我认为最强的是下面 7 类。

### 1. Attention BMM → O-Proj

[SGLang #34498](https://github.com/sgl-project/sglang/pull/34498)

\[
\boxed{
Attention\ Value\ BMM
\rightarrow
O\text{-}Projection
}
\]

Producer candidates：

\[
[H,T,D_v]
\]

vs

\[
[T,H,D_v].
\]

Consumer `O-Proj` 更自然地消费 token-major：

\[
[T,H,D_v]\rightarrow[T,H D_v].
\]

严格 RQ1 比较：

\[
T_{BMM}([H,T,D])
\]

vs

\[
T_{BMM}([T,H,D])
\]

然后：

\[
BMM([H,T,D])
\rightarrow transpose
\rightarrow OProj
\]

vs

\[
BMM([T,H,D])
\rightarrow OProj.
\]

这是目前最漂亮的非-KV RQ1。

---

### 2. MoE Dispatch → Grouped GEMM

[SGLang #37261](https://github.com/sgl-project/sglang/pull/37261)

\[
\boxed{
Router/Dispatch
\rightarrow
GroupedExpertGEMM
}
\]

Producer representation：

\[
NonExpanded
\]

vs

\[
ExpandedExpertMajor.
\]

传统：

\[
Dispatch
\rightarrow ep\_scatter
\rightarrow ExpertGEMM.
\]

Alternate：

\[
Dispatch_{ExpertMajor}
\rightarrow ExpertGEMM.
\]

这里研究的是：

\[
L_{\rm dispatch-best}
\stackrel?=
L_{\rm dispatch\rightarrow GEMM-best}.
\]

完全不是 KV。

---

### 3. MoE Prepare → Expert kernel

[vLLM #54109](https://github.com/vllm-project/vllm/pull/54109)

\[
\boxed{
MoE\ Prepare
\rightarrow
ExpertKernel
}
\]

真实 representation：

\[
Standard,
\quad
PaddedStandard,
\quad
BatchedExperts.
\]

这里 layout 已经不只是 axis permutation，而是：

> token rows 如何组织、padding 是否成为 physical representation 的一部分。

这非常适合说明：

\[
\text{representation}>\text{NHD/HND}.
\]

---

### 4. QKV Projection → KDA

[FlashInfer #5846](https://github.com/flashinfer-ai/flashinfer/pull/5846)

\[
\boxed{
QKV\ Projection
\rightarrow
KDA\ Prefill
}
\]

Producer：

\[
PackedRow=[T,3HD].
\]

通过 zero-copy split 得到 strided Q/K/V。

另一个候选：

\[
DenseSeparateQKV.
\]

因此：

\[
PackedProjection
\rightarrow StridedViews
\rightarrow KDA
\]

vs

\[
Projection
\rightarrow DenseQKV
\rightarrow KDA.
\]

这是非常好的 stride-layout operator case。

---

### 5. MLA Projection → Attention

[FlashInfer #5667](https://github.com/flashinfer-ai/flashinfer/pull/5667)

Producer：

\[
kv_b\ projection.
\]

Consumer：

\[
MLA\ Attention.
\]

中间可以选择：

\[
Separate\ K/V
\]

或：

\[
PackedFP8K/V.
\]

因此：

\[
Projection
\rightarrow separate
\rightarrow pack
\rightarrow Attention
\]

vs

\[
Projection/packer
\rightarrow consumer-native\ packed
\rightarrow Attention.
\]

也是典型 operator/subgraph edge。

---

### 6. KDA Prefill → Decode

[SGLang #34299](https://github.com/sgl-project/sglang/pull/34299)

\[
\boxed{
KDA\ Prefill
\rightarrow
PersistentState
\rightarrow
KDA\ Decode
}
\]

这里 layout 是：

\[
PrefillNativeCheckpoint
\]

vs

\[
DecodePackedState.
\]

它代表一种很重要的现代 LLM case：

\[
\boxed{\text{persistent recurrent state}}
\]

而不是传统 Transformer KV cache。

---

### 7. Expert GEMM2 → Combine

[FlashInfer #5208](https://github.com/flashinfer-ai/flashinfer/pull/5208)

\[
\boxed{
ExpertGEMM2
\rightarrow
MoECombine
}
\]

Producer layout：

\[
LocalTokenMajor
\]

vs

\[
RouteSlotMajor.
\]

还可以进一步变成：

\[
PeerRouteSlotMajor.
\]

所以这里既是 operator-level，又是 memory-placement case。

---

# 三、Operator-level 主实验可以形成怎样的 coverage

如果只看真正具有 LLM 语义的算子：

| LLM subsystem | P → C |
|---|---|
| Dense Attention | Attention BMM → O-Proj |
| MLA | MLA Projection/Pack → MLA Attention |
| MoE front-end | Dispatch → Grouped GEMM |
| MoE back-end | GEMM2 → Combine |
| Linear Attention | QKV Projection → KDA/GDN |
| Recurrent state | KDA Prefill → KDA Decode |
| Framework stride | View/Permute → Matmul/SDPA |

这已经跨越：

\[
\boxed{
Attention,\ GEMM,\ MoE,\ MLA,\ LinearAttention,\ RecurrentState
}
\]

所以完全可以避免 reviewer 认为：

> “RQ1 只是在研究 KV cache layout。”

---

# 四、哪些是“不同 Memory Level 上的 Layout”

这一类的研究目标稍微不同：

\[
\boxed{
L_{\text{producer-local memory level}}
\stackrel?=
L_{\text{downstream memory-level edge}}
}
\]

这里不是单纯 axis order，而是 physical representation 与 placement 联合。

---

## M1. HBM → Register / LDS → MMA

真实：

- [Triton #8450](https://github.com/triton-lang/triton/pull/8450)
- [Triton #7968](https://github.com/triton-lang/triton/pull/7968)

路径 A：

\[
HBM
\rightarrow Reg(Blocked)
\rightarrow LDS
\rightarrow Reg(DotOperand)
\rightarrow MMA.
\]

路径 B：

\[
HBM_{preshuffled}
\rightarrow Reg(DotCompatible)
\rightarrow MMA.
\]

所以它研究：

\[
\boxed{
HBM\ layout
\leftrightarrow
register/MMA\ layout
}
\]

memory levels：

\[
HBM\rightarrow LDS\rightarrow Register.
\]

这应该作为 RQ1 的 micro-level mechanism case。

---

## M2. HBM → LDS

IREE：

- [Issue #23782](https://github.com/iree-org/iree/issues/23782)
- [PR #22356](https://github.com/iree-org/iree/pull/22356)

比较：

\[
HBM\rightarrow Register\rightarrow LDS
\]

和：

\[
HBM\rightarrow LDS.
\]

这里 producer global layout 与 consumer LDS tile/swizzle 之间可能存在 tension。

---

## M3. TMEM → Register → HBM

CUTLASS：

[PR #3313](https://github.com/NVIDIA/cutlass/pull/3313)

路径：

\[
TensorCore
\rightarrow TMEM
\rightarrow Register
\rightarrow Epilogue
\rightarrow HBM.
\]

这里研究：

\[
TMEM-native fragment
\]

vs

\[
register/epilogue-friendly fragment.
\]

属于 Blackwell 特别重要的 memory-hierarchy case。

---

## M4. Register/SMEM → Peer GPU HBM

FlashInfer：

[PR #5208](https://github.com/flashinfer-ai/flashinfer/pull/5208)

传统：

\[
GEMM2
\rightarrow LocalHBM
\rightarrow Collective
\rightarrow PeerHBM.
\]

alternate：

\[
GEMM2\ epilogue
\rightarrow PeerHBM.
\]

因此跨：

\[
\boxed{
Reg/SMEM
\rightarrow
LocalHBM/PeerHBM
}
\]

而且 layout 从：

\[
LocalTokenMajor
\]

变成：

\[
PeerRouteSlotMajor.
\]

这是非常漂亮的 operator × placement 联合实验。

---

## M5. GPU HBM → Host pinned DRAM → GPU HBM

SGLang：

[PR #41182](https://github.com/sgl-project/sglang/pull/41182)

真实：

\[
PrefillGPU
\rightarrow
PinnedHost
\rightarrow
Network
\rightarrow
PinnedHost
\rightarrow
DecodeGPU.
\]

布局/representation：

\[
GPUPageNative
\]

vs

\[
HostPackedTransport
\]

vs

\[
DecodeNative.
\]

这里是：

\[
\boxed{
HBM\leftrightarrow HostDRAM
}
\]

真实 GLM-5.3、Kimi-K3 PD workload。

---

## M6. Host DRAM → GPU hot buffer → Sparse Attention

vLLM HiSparse：

[PR #46326](https://github.com/vllm-project/vllm/pull/46326)

\[
HostFullKV
\rightarrow
GPUCompactHotBuffer
\rightarrow
SparseMLA.
\]

可以比较：

\[
H2D\ copy-friendly\ layout
\]

和：

\[
SparseAttention\ consumer-native\ hot-buffer\ layout.
\]

这是：

\[
\boxed{
HostDRAM
\rightarrow
GPUHBM
}
\]

真实长上下文 sparse attention case。

---

## M7. Remote → Host vs Remote → GPU

vLLM：

[PR #55398](https://github.com/vllm-project/vllm/pull/55398)

同一 remote P/D KV：

\[
Remote
\rightarrow HostPage
\rightarrow GPU
\]

或：

\[
Remote
\rightarrow FinalGPUPage.
\]

这主要验证：

\[
\boxed{\text{placement preference}}
\]

而不是纯 axis-layout，因此我会把它标成 RQ1 placement extension。

---

## M8. Host Expert → GPU Expert Bank → MoE

vLLM：

[PR #56177](https://github.com/vllm-project/vllm/pull/56177)

真实：

\[
PinnedHostExpert
\rightarrow
GPUExpertBank
\rightarrow
MarlinGEMM.
\]

representation：

\[
Checkpoint/compact\ NVFP4
\]

vs

\[
MarlinConvertedPhysicalRows.
\]

这是非常好的：

\[
\boxed{
HostDRAM\ representation
\rightarrow
GPUCompute\ representation
}
\]

case。

---

## M9. Host → NVMe / mmap → Host → GPU

vLLM：

[PR #54129](https://github.com/vllm-project/vllm/pull/54129)

Qwen3.8-Flash-Next 47.68 GiB PLE：

\[
Disk
\rightarrow mmap/page\ cache
\rightarrow HostGather
\rightarrow GPUBuffer.
\]

SGLang HiCache 也有：

\[
Host
\leftrightarrow NVMe.
\]

这一层更偏：

\[
\boxed{\text{storage representation / placement}}
\]

适合作为 external extension，不建议拿来作为 strict layout RQ1 主判据。

---

# 五、可以把这些 case 放成一个二维矩阵

这样你的实验设计会非常清楚。

| | Register/TMEM | Shared/LDS | GPU HBM | Peer GPU | Host DRAM | NVMe |
|---|---:|---:|---:|---:|---:|---:|
| Attention/GEMM micro | Dot/MMA | HBM→LDS Attention | KV/Attention | — | — | — |
| Dense operator | accumulator | — | BMM→OProj | — | — | — |
| MoE | GEMM epilogue | SMEM | Dispatch→GEMM | GEMM2→Combine | Host expert | — |
| Linear attention | registers | — | PackedQKV→KDA | — | recurrent state offload possible | — |
| Sparse attention | — | — | Gather→MSA | — | HiSparse KV | KV tier |
| Serving | — | — | Prefill/Decode cache | remote GPU | host staging | filesystem tier |

这实际上给 RQ1 带来了两个独立的 diversity 轴：

\[
\boxed{
\text{Operator diversity}
}
\]

和：

\[
\boxed{
\text{Memory-hierarchy diversity}.
}
\]

---

# 六、我建议论文里明确分三类证据

### Class A：Operator-level strict RQ1

主结论最应该依赖这些：

\[
\boxed{
\begin{aligned}
&AttentionBMM\rightarrow OProj\\
&MoEDispatch\rightarrow GroupedGEMM\\
&MoEPrepare\rightarrow ExpertKernel\\
&PackedQKV\rightarrow KDA\\
&MLAProjection/Pack\rightarrow MLAAttention\\
&KDAPrefill\rightarrow KDADecode\\
&View/Stride\rightarrow Matmul
\end{aligned}}
\]

这些 case 直接回答：

> 一个真实 LLM operator 的局部输出 layout 是否与下一真实 operator 的 edge-optimal layout 不同？

---

### Class B：Memory-hierarchy strict/extended RQ1

\[
\boxed{
\begin{aligned}
&HBM\rightarrow LDS/Reg\rightarrow MMA\\
&TMEM\rightarrow Reg\rightarrow Epilogue\\
&GEMM2\rightarrow PeerHBM\rightarrow Combine\\
&HostExpert\rightarrow GPUExpert\rightarrow GEMM\\
&HostKV\rightarrow GPUHotBuffer\rightarrow SparseAttention\\
&GPU\rightarrow Host\rightarrow Network\rightarrow GPU
\end{aligned}}
\]

这些回答：

> representation optimum 是否随着 memory level 变化？

---

### Class C：Real Attention/KV external validity

不要删除，因为它们是现代 LLM 的重要组成：

\[
\boxed{
\begin{aligned}
&KVWriter\rightarrow TRTLLM/FlashInfer\\
&SparseGather\rightarrow MSA\\
&StandardKV\rightarrow AITERShuffle\rightarrow Attention\\
&P/D\ KV\rightarrow MSA
\end{aligned}}
\]

但这里 consumer 必须是真实 Attention，不再使用 synthetic：

\[
head\text{-}local\ scan.
\]

---

# 七、因此，与最初 v10/v13 相比，新 suite 的变化可以非常直观地表达

旧：

```text
KV writer
    ↓
NHD/HND
    ↓
synthetic head-local scan
```

新：

```text
                    ┌→ O-Proj
Attention BMM ──────┤

MoE Dispatch ───────→ Grouped GEMM

QKV Projection ─────→ KDA/GDN

MLA Projection ─────→ MLA Attention

KDA Prefill ─────────→ KDA Decode

GEMM2 Epilogue ──────→ Peer-GPU Combine

Host Expert ─────────→ GPU Marlin GEMM

HBM Operand ─────────→ LDS/Register → MMA

Host Sparse KV ──────→ GPU Hot Buffer → Sparse Attention
```

所以新的 RQ1 不再是：

\[
\boxed{\text{“KV 的读写顺序会不会冲突？”}}
\]

而会真正变成：

\[
\boxed{
\text{Across real LLM operators and memory hierarchies,
when does producer-local representation differ from edge-optimal representation?}
}
\]

这套 coverage 已经足够多样：至少有 6–7 个强 operator-level case、5–6 个跨 memory-level case，以及若干真实 Attention/KV external-validity case。这样在论文里会比继续增加 `KV write → scan` 的 shape 数量有说服力得多。

---

If you want, I can:

- 列出全部非KV读写操作的案例
- 筛选验证算子级布局偏好的案例
- 总结验证不同存储层级布局偏好的案例

---

可以。这里最重要的修正是：

> RQ1 不是比较“Producer 喜欢什么 layout”和“Consumer 喜欢什么 layout”本身，而是比较  
> **Producer-local 最优 representation** 与 **完整 `Producer → repair → Consumer` edge 的最优 producer representation** 是否不同。

因此每个 case 最少要测两个 preference：

\[
L_P^*=\arg\min_{L_P}T_P(L_P)
\]

以及

\[
(L_P,R,L_C)^*
=\arg\min T_{\rm edge}
\]

真正的 RQ1 positive 是：

\[
\boxed{L_P^*\neq L_{P,\rm edge}^*}
\]

`Consumer-only preference` 可以测，但主要用于解释机制，不是判定 RQ1 的充分条件。

下面我把前面提出的 real-world test cases重新整理，并修正了一些之前不够严谨的分类。

| ID | Test case / 真实来源 | Producer | Consumer | 实际要比较的 layout preference | 完整 edge 应比较什么 | 是否适合 strict RQ1 |
|---|---|---|---|---|---|---|
| R1 | SGLang KV→TRTLLM MHA；[源码](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/trtllm_mha_backend.py) | paged KV writer | TRTLLM MHA decode/prefill | **KV writer：NHD vs HND direct-write** | `P(NHD)→convert→C(HND)` vs `P(HND)→C(HND)`；若有合法 NHD consumer path再加 direct NHD | ★★★ |
| R2 | vLLM FlashInfer KV；[源码](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flashinfer.py) | vLLM KV-cache writer | FlashInfer/TRTLLM-gen attention | **writer：LBHNC vs BLHNC** | `write(LBHNC)→FI` vs `write(BLHNC)→FI`，consumer固定 | ★★★ |
| R3 | SGLang MiniMax-M3；[#34525](https://github.com/sgl-project/sglang/pull/34525) | sparse KV gather/prepare | `fmha_sm100` MSA prefill | **gather：Compact-NHD vs Compact-HND direct output** | `gather(NHD)→transpose→MSA` vs `gather(HND)→MSA` | ★★★ |
| R4 | SGLang Attention→OProj；[#34498](https://github.com/sgl-project/sglang/pull/34498) | attention value BMM | O-Proj GEMM | **BMM output `[H,T,D]` vs `[T,H,D]`** | `[H,T,D]→transpose→OProj` vs BMM direct `[T,H,D]→OProj` | ★★★ |
| R5 | SGLang DeepEP→DeepGEMM；[#37261](https://github.com/sgl-project/sglang/pull/37261) | DeepEP dispatch | grouped expert GEMM | **dispatch output non-expanded vs expanded expert-major** | `dispatch(normal)→ep_scatter→GEMM` vs `dispatch(expanded)→GEMM` | ★★★ |
| R6 | vLLM DeepEPV2；[#54109](https://github.com/vllm-project/vllm/pull/54109) | MoE Prepare/Dispatch | FusedMoE expert kernel | **Standard vs PaddedStandard activation contract** | Prepare(Standard/Padded)→同一 expert kernel；只比较 consumer 都合法的 formats | ★★★ |
| R7 | FlashInfer packed QKV→KDA；[#5846](https://github.com/flashinfer-ai/flashinfer/pull/5846) | QKV projection | Cake KDA prefill | **packed-row/strided QKV vs direct dense-separated QKV** | `packed→zero-copy-strided→KDA` vs `packed→densify→KDA` vs patched direct-dense projection→KDA | ★★★ |
| R8 | SGLang KDA Prefill→Decode；[#34299](https://github.com/sgl-project/sglang/pull/34299) | KDA prefill state/checkpoint writer | KDA decode | **prefill-native checkpoint vs decode-packed state** | native state→pack→decode vs prefill direct decode-packed→decode | ★★☆ |
| R9 | FlashInfer MLA pack→Attention；[#5667](https://github.com/flashinfer-ai/flashinfer/pull/5667) | MLA K/V packer after projection | MLA/TRTLLM attention | **separate K/V vs packed consumer-native K/V** | separate→pack→attention vs direct packed producer→attention | ★★★ |
| R10 | FlashInfer MSA NVFP4；[#5402](https://github.com/flashinfer-ai/flashinfer/pull/5402) | NVFP4 KV quantizer/packer | Cake/msa_ops MSA decode | **generic KV+scale packing vs canonical planar MSA page layout** | generic→repack→MSA vs producer direct canonical page layout→MSA | ★★★ |
| R11 | vLLM standard KV→AITER shuffle；[#59111](https://github.com/vllm-project/vllm/issues/59111) | NIXL decode-side importer | AITER attention | **standard KV page vs AITER-shuffled page as importer output** | standard landing→shuffle→AITER vs direct shuffled landing→AITER | ★★★ |
| R12 | vLLM MiniMax-M3 P/D mismatch；[#46204](https://github.com/vllm-project/vllm/issues/46204) | P/D KV importer | MSA | **HND landing vs NHD-native landing** | HND wire/import→NHD repair→MSA vs direct NHD destination→MSA | ★★★ |
| R13 | vLLM KV data+scale alignment；[#56081](https://github.com/vllm-project/vllm/issues/56081) | quantized KV cache writer | gather-based attention | **inline scale vs side/aligned scale representation** | writer-local cost vs aligned data+metadata→gather attention edge | ★★★，但属 data+metadata layout |
| R14 | vLLM QSA sparse gather；[#55394](https://github.com/vllm-project/vllm/issues/55394), [#55430](https://github.com/vllm-project/vllm/pull/55430) | QSA K/V gather | sparse QSA prefill | **per-query gathered layout vs tile-union gathered layout** | per-row gather→QSA vs shared tile-union gather→QSA | ★★★ |
| R15 | vLLM QKNorm/RoPE/QSA prepare；[#57097](https://github.com/vllm-project/vllm/pull/57097) | QKNorm+RoPE+KVPrep subgraph | QSA/indexer/attention | **不同 direct cache output representation** | fixed fused producer under alternate cache mappings→same consumer | ★★☆：需额外 layout-only patch |
| R16 | vLLM/SGLang RoPE→Attention fusion；[#46979](https://github.com/vllm-project/vllm/issues/46979), [SGLang #41698](https://github.com/sgl-project/sglang/pull/41698) | QKNorm/RoPE/KV preparation | real Attention | **producer output KV representation** | direct consumer-native cache vs producer-native+repair | ★★☆；PR before/after不能直接当RQ1 |
| R17 | FlashInfer GEMM2→MoE Combine；[#5208](https://github.com/flashinfer-ai/flashinfer/pull/5208) | expert GEMM2 epilogue | cross-GPU combine/local reduction | **local token output vs `(token,k_slot)` route-slot output** | local HBM→collective vs direct peer route-slot→local reduce | ★★☆：layout+placement+comm |
| R18 | vLLM Host Expert→Marlin；[#56177](https://github.com/vllm-project/vllm/pull/56177) | expert-cache H2D refill | Marlin MoE GEMM | **compact/checkpoint representation vs Marlin-converted physical rows** | compact H2D→convert→GEMM vs host consumer-native→H2D→GEMM | ★★★ |
| R19 | SGLang PD matching DCP；[#41953](https://github.com/sgl-project/sglang/pull/41953), [#41958](https://github.com/sgl-project/sglang/pull/41958) | prefill-side KV/state exporter | decode-side importer/first consumer | **prefill-native page vs canonical-wire vs decode-native page** | pack→transfer→unpack vs direct whole-page consumer-native transfer | ★★☆ system-level |
| R20 | SGLang NIXL Host staging；[#41182](https://github.com/sgl-project/sglang/pull/41182) | GPU→host gather | decode-side host→GPU scatter | **GPU-row-native vs host-packed wire representation** | GPU rows→host pack→network→scatter vs alternate direct/canonical packing | ★★☆ placement/transport RQ1 |
| R21 | vLLM HiSparse；[#46326](https://github.com/vllm-project/vllm/pull/46326) | host→GPU top-k KV swap-in | sparse MLA decode | **copy-friendly hot buffer vs consumer-native compact hot buffer** | host gather→GPU layout→sparse decode | ★★☆；consumer-native alternate需patch |
| R22 | vLLM HiSparse direct landing；[#55398](https://github.com/vllm-project/vllm/pull/55398) | remote P/D receiver | decode cache/attention | **host landing vs final GPU-page landing** | remote→host→GPU→C vs remote→GPU-final-page→C | ★★☆，主要是 placement |
| R23 | SGLang HiCache | GPU KV backup | later restore/attention | **GPU-native representation vs L2-host/storage-native representation** | backup-local optimum vs backup+restore+attention optimum | ★★☆；需要明确同一 logical format |
| R24 | vLLM PLE mmap；[#54129](https://github.com/vllm-project/vllm/pull/54129) | disk/host row gather | GPU PLE consumer | **file/gather-friendly row organization vs GPU-consumer row buffer** | mmap gather→H2D→PLE | ★☆☆：更像 placement/I/O，不是纯 layout RQ1 |
| R25 | PyTorch non-contiguous→Matmul；[#197100](https://github.com/pytorch/pytorch/pull/197100) | upstream tensor/view producer | matmul | **non-contiguous zero-copy stride vs contiguous materialization** | view→matmul vs view→copy→matmul | ★★★ framework-level stride RQ1 |
| R26 | PyTorch permuted K→SDPA；[#195320](https://github.com/pytorch/pytorch/issues/195320) | K producer/permute | SDPA | **permuted stride representation vs expected transpose/contiguous** | stride-preserving→SDPA vs materialize→SDPA | ★★☆，首先是 correctness/legality |
| R27 | Triton Dot1→Dot2；[#10342](https://github.com/triton-lang/triton/issues/10342) | first `tl.dot` | second `tl.dot` | **Dot1 accumulator layout vs Dot2 operand-compatible layout** | Dot1 native→convert→Dot2 vs alternate intermediate direct-compatible→Dot2 | ★★★ micro/operator mechanism |
| R28 | Triton GMEM→MFMA；[#8450](https://github.com/triton-lang/triton/pull/8450), [#7968](https://github.com/triton-lang/triton/pull/7968) | global operand load | Dot/MFMA | **coalesced Blocked vs Dot-compatible/preshuffled** | GMEM→Blocked→LDS→Dot vs GMEM preshuffled→Reg→Dot | ★★★ micro mechanism |
| R29 | Triton WMMA/FMA→Store；[#10563](https://github.com/triton-lang/triton/pull/10563), [#11704](https://github.com/triton-lang/triton/pull/11704) | MMA/FMA accumulator | global store | **accumulator-native vs coalesced-store layout** | compute-native→relayout→store vs alternate producer output→store | ★★★ micro mechanism |
| R30 | IREE HBM→LDS Attention；[#23782](https://github.com/iree-org/iree/issues/23782), [#22356](https://github.com/iree-org/iree/pull/22356) | global-memory tile loader | Attention/MMA | **GMEM-coalesced representation vs LDS/MMA-native tile layout** | HBM→Reg→LDS→MMA vs direct GMEM→LDS→MMA | ★★★ micro memory-hierarchy |
| R31 | CUTLASS TMEM→Epilogue；[#3313](https://github.com/NVIDIA/cutlass/pull/3313) | MMA accumulator in TMEM | register epilogue/store | **TMEM load atom / fragment mapping choices** | TMEM→Reg→epilogue under different mappings | ★★☆：是 micro-layout，不是 tensor-layout |
| R32 | TileLang fragment→GEMM/shared | fragment-producing op | GEMM/shared consumer | **register fragment mapping vs shared/MMA-compatible mapping** | direct register consumer vs shared relayout | ★★☆ mechanism |

这里有几个需要纠正前面回答中的说法。

第一，`vLLM FlexAttention vs FlashInfer` 不应该作为最纯粹的 RQ1。FlexAttention 当前是 `LBNHC`-only，而 SM100 FlashInfer/TRTLLM-gen 是 `LBHNC/BLHNC`。这非常重要，但它测试的是：

\[
\mathcal L_{C_1}\cap\mathcal L_{C_2}
\]

甚至可能为空，更接近 **multi-consumer/layout-domain RQ2**，不是固定同一 Consumer 的 strict RQ1。vLLM 自己的测试也明确举了 “Flex is LBNHC-only next to FlashInfer's LBHNC” 这一例子。  
[FlexAttention 源码](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flex_attention.py)  
[FlashInfer 源码](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flashinfer.py)

第二，`Draft → Target Verify` 也不能默认算 strict RQ1。只有 draft 真正生成一个物理 state/cache，由 target 直接消费时才是 P→C edge。如果只是两个 backend 共享 persistent cache、各自要求不同 layout，那更接近 multi-consumer/domain-selection。

第三，`FP8/NVFP4 vs BF16` 本身不是 layout。比如 vLLM #59035、#54846 或 SGLang HiCache INT8，如果直接比较 BF16 和 FP8，实验改变的是：

\[
dtype+\text{layout/packing}
\]

不能据此声称 RQ1 layout inversion。正确做法是先固定 dtype：

\[
dtype = \text{constant}
\]

比较 physical representation；之后再做：

\[
dtype\times layout
\]

factorial，作为扩展。

第四，`Host vs GPU vs NVMe` 本身是 memory placement，不是传统 tensor layout。因此如果你的 RQ1 定义仍然严格叫 “layout rank inversion”，这些 case 应写成 external extension；如果你把研究对象升级成：

\[
\boxed{\text{physical representation / placement}}
\]

它们才可以统一进入主问题。

第五，SGLang #34299 的 `prefill-native checkpoint vs packed decode` 很有潜力，但只有当你真正实现：

\[
Prefill\ directly\ emits\ DecodePacked
\]

以后，它才是 strict RQ1。仅测：

\[
PrefillNative\rightarrow Pack\rightarrow Decode
\]

只能说明 conversion/reuse trade-off，更像 RQ3。

---

### 最容易混淆的三个 winner

对于每个表里的 case，我建议最终都打印三个 winner：

\[
\boxed{
L_P^*
=
\arg\min_L T_P(L)
}
\]

这是 Producer 自己最喜欢什么。

然后可选：

\[
L_C^*
=
\arg\min_L T_C(L)
\]

这是 Consumer 单独最喜欢什么，只用于解释。

真正决定 RQ1：

\[
\boxed{
S_E^*
=
\arg\min_{L_P,R,L_C}
T_{\rm edge}(P(L_P)\rightarrow R\rightarrow C(L_C))
}
\]

然后取 oracle 中 producer 那一部分：

\[
L_{P,E}^*.
\]

只有：

\[
\boxed{
L_P^*\neq L_{P,E}^*
}
\]

才叫严格的 RQ1 rank inversion。

例如 `Attention BMM → OProj`：

\[
T_{BMM}([H,T,D])
<
T_{BMM}([T,H,D])
\]

说明 producer 喜欢 `[H,T,D]`。

但完整 edge 如果：

\[
T[
BMM([T,H,D])
\rightarrow OProj
]
<
T[
BMM([H,T,D])
\rightarrow transpose
\rightarrow OProj
]
\]

那么：

\[
L_P^*=[H,T,D]
\]

而：

\[
L_{P,E}^*=[T,H,D],
\]

这才是最标准、最容易解释的 RQ1 positive。

如果要从上述 32 类里挑一套最有说服力的主实验，我建议优先保留：

\[
\boxed{
R1,R3,R4,R5,R7,R9,R11,R12,R13,R18,R25,R28
}
\]

它们分别覆盖：

- KV persistent layout；
- sparse gather；
- Attention→GEMM；
- MoE dispatch；
- packed/strided QKV；
- MLA packing；
- transfer→backend-native layout；
- P/D layout mismatch；
- data+metadata alignment；
- host→GPU weight format；
- framework stride；
- HBM/LDS/register micro-layout。

这 12 个 case 的 P/C 类型、memory level 和 representation 类型差异已经足够大，而且每一个都能明确追溯到真实的 2025–2026 issue/PR。

---

可以。这次我把它整理成一个“RQ1 P–C 实验 catalog”。筛选原则是：2025–2026 GitHub 中，能够从真实 issue/PR 还原出 `Producer → representation/layout boundary → Consumer`，并且适合构造 layout/placement 对照。下面优先纳入真实 LLM operator/subgraph/stage；编译器 micro-op 单独放最后。

需要强调：`Producer Layout` / `Consumer Layout` 一栏中，**PR 已明确存在的 representation**我直接写；为了完成 RQ1 需要额外 patch 的候选，我标成“实验扩展”。非法 consumer/layout 组合不应运行，记 `illegal`。

### RQ1 实验总表

| # | Issue / PR | Framework / real case | Test case | Producer | Producer layout / representation | Consumer | Consumer layout / contract | 整图中的其他算子 / repair | RQ1 应改哪里 |
|---|---|---|---|---|---|---|---|---|---|
| A1 | [vLLM #57097](https://github.com/vllm-project/vllm/pull/57097) | vLLM / Qwen3.8-Flash-Next QSA | QK-Norm+RoPE+KVPrep → QSA | `qsa_prepare` / 原三 kernel prepare stage | main Q/K/gate + paged K/V；BF16/FP8 cache | QSA sparse attention/indexer | paged QSA/KV cache contract | QKV projection → QK norm → RoPE → gate split → QSA metadata → KV cache → QSA | 固定 QSA consumer，改变 **prepare 最终 K/V cache 写布局**；不要直接用 fused-vs-unfused 当 layout-only 结论 |
| A2 | [vLLM #46979](https://github.com/vllm-project/vllm/issues/46979) | vLLM / GQA、MLA、SparseMLA | RoPE/KVPrep → Attention | RoPE + cache update family | GQA KV / MLA latent / SparseMLA KV | 对应 attention backend | backend-specific KV layout | projection → norm/RoPE → cache → attention | 把它做成 umbrella：分别对 GQA/MLA/SparseMLA 枚举 producer direct-emission layout |
| A3 | [vLLM #52901](https://github.com/vllm-project/vllm/pull/52901) | vLLM / gated attention | gated QKV+RoPE+KV write → Attention | fused QKV split/QK norm/RoPE/cache write | NHD/HND 类 KV representation | attention | consumer-supported cache layout | QKV split → RMSNorm → RoPE → gate → KV update | 改 fused producer 的 KV store mapping；固定 fusion 与 dtype |
| A4 | [vLLM #57640](https://github.com/vllm-project/vllm/pull/57640) | vLLM / Kimi-K3 MLA decode | MLA KV write + Q prep → MLA decode | fused KV-write/Q-prep | MLA latent / packed cache representation | MLA decode | AITER MLA native contract | projection → Q prep → KV write → MLA | 改 fused producer 的 cache representation；consumer 固定 AITER |
| A5 | [vLLM #59289](https://github.com/vllm-project/vllm/pull/59289) | vLLM / MLA | MLA KV-cache writer → MLA | vectorized KV writer | producer-native vectorized MLA cache | MLA attention | native MLA cache | projection → vectorized store → MLA | 比较 writer 直接生成不同合法 MLA cache layout，不仅比较 vector width |
| A6 | [vLLM #59035](https://github.com/vllm-project/vllm/pull/59035) | vLLM ROCm | QKNorm+RoPE+KVCache → FP8-Q Attention | fused norm/RoPE/cache/Q quant | Q BF16/FP8 + KV representation | attention | FP8-Q + cache contract | Norm → RoPE → Q quant → KV write → attention | layout 与 Q dtype 做 factorial；不能把 FP8 收益全归因于 layout |
| A7 | [vLLM #54846](https://github.com/vllm-project/vllm/pull/54846) | vLLM / Qwen3.8/Qwen4Exp QSA | QSA KV producer → QSA | KV update | BF16 / FP8-e4m3 / NVFP4 QSA cache | QSA | QSA page contract | quantize → cache → sparse attention | 固定 dtype 后比较 cache physical layout；再独立做 dtype×layout |
| A8 | [vLLM #59300](https://github.com/vllm-project/vllm/pull/59300) | vLLM / MiniMax-M3 | NVFP4 KV producer → MSA | KV quant/cache writer | NVFP4 paged KV | MSA sparse attention | native MSA NVFP4 contract | quantization → cache → MSA | direct-write MSA-native vs canonical KV+repair |
| A9 | [vLLM #59111](https://github.com/vllm-project/vllm/issues/59111) | vLLM / ROCm AITER | standard KV pages → AITER decode | NIXL receive/import | standard KV pages | AITER attention | **shuffled pages** | transfer → decode-side conversion → attention | 改 decode-side importer：standard→shuffle 后消费 vs **直接 landing 成 shuffled layout** |
| A10 | [vLLM #59110](https://github.com/vllm-project/vllm/issues/59110) | vLLM / heterogeneous TP | remote KV → ROCm attention | NIXL importer | split K/V slot representation | AITER/ROCm/B12X | backend-shuffled representation | P/D transport → slot map → layout translation → attention | 改 importer destination mapping；是 transfer-layout RQ1 |
| A11 | [vLLM #46204](https://github.com/vllm-project/vllm/issues/46204) | vLLM / MiniMax-M3 + P/D | KV connector → MSA | connector receive | HND forced by connector | MSA | NHD-native（issue 所述冲突） | prefill → transfer → cache mapping → MSA | 最值得测 `HND wire→NHD repair` vs `NHD direct landing` |
| A12 | [vLLM #55394](https://github.com/vllm-project/vllm/issues/55394), [#55430](https://github.com/vllm-project/vllm/pull/55430) | vLLM / Qwen3.8 QSA | KV gather → QSA sparse prefill | sparse gather | per-query gather / tile-union gathered K/V | sparse QSA | sparse tile consumer | index selection → gather → sparse attention | 改 **gather output organization**；per-row vs tile-union/consumer-native |
| A13 | [vLLM #56081](https://github.com/vllm-project/vllm/issues/56081) | vLLM / gather attention | KV quant/write → gather consumer | cache writer with inline scales | inline data+scale causing alignment shift | gather-based attention | 128B-friendly aligned rows | KV data + scale metadata → gather | 改 producer 的 **data/scale physical packing**：inline vs side-array/aligned |
| A14 | [SGLang #41698](https://github.com/sgl-project/sglang/pull/41698) | SGLang / DSA decode | RoPE+KV prep → DSA Attention | fused AITER DSA prepare | DSA KV-prep representation | AITER DSA attention | native DSA contract | RoPE → cache prep → DSA | direct consumer-native cache write vs old preparation |
| A15 | [SGLang #35357](https://github.com/sgl-project/sglang/pull/35357) | SGLang / MiniMax-M3 | sparse QKNorm+RoPE+cache write → MSA | fused sparse prepare | sparse Q/K + cache representation | MSA | MSA cache contract | QKNorm → RoPE → cache write → MSA | 改 fused cache output mapping |
| A16 | [SGLang #34525](https://github.com/sgl-project/sglang/pull/34525) | SGLang / MiniMax-M3 | KV gather → MSA sparse prefill | `gather_kv_hnd` | native pool NHD；候选 Compact-NHD / Compact-HND | `fmha_sm100` MSA | `[page,Hkv,page_size,D]` HND | full NHD pool → gather/transpose → MSA | **核心 case**：实现 `gather_NHD` 与 `gather_HND` direct-emission，再测 edge |
| A17 | [SGLang #35322](https://github.com/sgl-project/sglang/pull/35322) | SGLang / MLA | FP8 MLA cache → FlashInfer MLA | cache producer | native FP8 MLA | FlashInfer MLA | native FP8 contract | cache → 原 full-pool BF16 cast → MLA | 比较 FP8-native consumer vs FP8→BF16 materialization；layout/dtype要拆开 |
| A18 | [SGLang #38592](https://github.com/sgl-project/sglang/pull/38592) | SGLang unified memory | KV/state producer → multiple consumers | unified pool writer | **token-major dense views** vs旧 pool organization | attention / transfer / HiCache | per-consumer views | unified allocation → views → attention/PD/HiCache | 改 unified-pool physical order；这是 multi-consumer extension |
| A19 | [SGLang #41875](https://github.com/sgl-project/sglang/pull/41875) | SGLang NPU / DSA | KV writer → DSA | cache writer | startup-resolved KV layout | DSA backend | capability-advertised layout | startup capability → pool creation → DSA | 强制每种合法 startup layout，与 `auto` 比 |
| A20 | [FlashInfer #5402](https://github.com/flashinfer-ai/flashinfer/pull/5402) | FlashInfer / MSA | NVFP4 KV pack → MSA decode | NVFP4 KV producer | canonical planar data/scale page layout | Cake MSA decode | exact NVFP4 paged contract | quant → pack data/scales → MSA | 改 quantizer/packer direct output；canonical vs generic KV+scale |
| A21 | [FlashInfer #5667](https://github.com/flashinfer-ai/flashinfer/pull/5667) | FlashInfer / MLA | MLA projection output → attention | K/V concat+FP8 pack | separate `k_nope,v,k_pe` vs packed FP8 K/V | TRTLLM ragged/MLA attention | contiguous packed FP8 K/V | kv_b projection → concat → cast → attention | 改 pack producer；separate vs direct packed |
| A22 | [FlashInfer #5720](https://github.com/flashinfer-ai/flashinfer/pull/5720) | FlashInfer / H512 decode | FP8 Q + NVFP4 KV → decode | Q/KV quant producers | FP8-Q / NVFP4-KV native representation | PrimTS decode | native H512 contract | quant → cache → attention | 更适合 representation/dtype factorial，不宜只叫 axis layout |
| A23 | [FlashInfer #5742](https://github.com/flashinfer-ai/flashinfer/pull/5742) | FlashInfer / DeepSeek-V4 sparse MLA | sparse KV producer → sparse MLA | sparse KV/context producer | Cake sparse-MLA layouts | sparse MLA | generated SM100/103 contract | index/gather → sparse MLA | 用真实 DSV4 shape 枚举 native/gathered representation |
| A24 | [TensorRT-LLM #18485](https://github.com/NVIDIA/TensorRT-LLM/pull/18485) | TRT-LLM / sparse attention | ragged-row builder → sparse attention | ragged metadata/KV row producer | ragged sparse row contract | sparse attention | same row contract | indexer → ragged-row preparation → attention | 改 row producer 的 physical row arrangement |
| A25 | [TensorRT-LLM #14327](https://github.com/NVIDIA/TensorRT-LLM/issues/14327) | TRT-LLM / DeepSeek-V4 | KV writer → SWA/COMPRESS attention | KV cache writer | TRT-LLM FP8 SWA/COMPRESS representation | DSV4 attention | reference FlashMLA/SGLang layout differs | cache update → attention | 明确测 TRT-LLM native vs reference-compatible direct write |
| A26 | [TensorRT-LLM #19502](https://github.com/NVIDIA/TensorRT-LLM/pull/19502) | TRT-LLM / DeepSeek-V4 | sparse KV residency/offload → attention | sparse KV restore/import | resident/offloaded sparse representation | DSV4 attention | sparse attention cache contract | residency → restore → attention | placement+layout case |

### Dense compute / projection / stride 类

| # | Issue / PR | Framework / real case | Test case | Producer | Producer Layout | Consumer | Consumer Layout | 其他算子 | 应改哪里 |
|---|---|---|---|---|---|---|---|---|---|
| D1 | [SGLang #34498](https://github.com/sgl-project/sglang/pull/34498) | SGLang / Kimi-K2.7-Code-MXFP4 | Attention BMM → O-Proj | absorbed value BMM | **`[H,T,Dv]` 或 `[T,H,Dv]`** | O-Proj GEMM | `[T,H,Dv]→[T,H·Dv]` | transpose-copy / flatten | **最干净 strict RQ1**：只改 BMM epilogue output mapping |
| D2 | [PyTorch #197100](https://github.com/pytorch/pytorch/pull/197100) | PyTorch/Inductor | view → matmul | upstream tensor/view | non-contiguous zero-copy stride vs contiguous | matmul | accepts compatible strided operand | decomposition/materialization | 改 graph decomposition：preserve view vs materialize |
| D3 | [PyTorch #195320](https://github.com/pytorch/pytorch/issues/195320) | PyTorch/Inductor / SDPA | permuted K → SDPA | K producer/permute | arbitrary permuted stride | SDPA fusion | expected logical transpose contract | permute/view → SDPA | 更适合作 correctness+legality control；枚举 stride-equivalent vs materialized |
| D4 | [Triton #10342](https://github.com/triton-lang/triton/issues/10342) | Triton | Dot1 → bias/FFMA → Dot2 | first `tl.dot` | MMA accumulator encoding | second `tl.dot` | DotOperand encoding | FFMA/bias + `convert_layout` | 改 Dot1 output/中间 mapping，比较 direct-compatible vs convert |
| D5 | [IREE #22489](https://github.com/iree-org/iree/issues/22489) | IREE | producer op → preferred-layout consumer | contraction/elementwise | incoming encoding | downstream contraction | preferred encoding | local relayout | 用真实 matmul→elementwise→matmul graph |
| D6 | [IREE #24969](https://github.com/iree-org/iree/pull/24969) | IREE | graph producer → graph consumer | encoded tensor producer | globally materialized layout | downstream graph op | target-aware preferred layout | layout propagation/materialization | 原生 heuristic vs forced producer layout vs graph oracle |

### MoE

| # | Issue / PR | Framework / real case | Test case | Producer | Producer Layout | Consumer | Consumer Layout | 其他算子 | 应改哪里 |
|---|---|---|---|---|---|---|---|---|---|
| M1 | [SGLang #37261](https://github.com/sgl-project/sglang/pull/37261) | SGLang / Qwen3-30B-A3B、MiMoV2 | DeepEP Dispatch → DeepGEMM | DeepEP v2 dispatch | **non-expanded vs expanded expert-major** | DeepGEMM grouped GEMM | expert-grouped contiguous rows | `ep_scatter`、routing-weight handling | **直接改 dispatch destination row layout** |
| M2 | [vLLM #54109](https://github.com/vllm-project/vllm/pull/54109) | vLLM / DeepEPV2 | Prepare/Finalize → experts | DeepEPV2 prepare | `Standard`, **`PaddedStandard`**, `BatchedExperts` | FusedMoE expert kernel | declared activation-format contract | padding markers `topk_ids=-1` | 强制同一 expert kernel 消费 Standard/PaddedStandard；只测合法交集 |
| M3 | [SGLang #39313](https://github.com/sgl-project/sglang/pull/39313) | SGLang / modern mixed-precision MoE | shared+routed experts → MegaMoE | shared/routed expert paths | FP8 shared vs MXFP4 routed representations | DeepGEMM/MegaMoE | fused expert representation | shared expert fusion | 这是 dtype+representation 联合 case，layout-only 需控制 dtype |
| M4 | [SGLang #38884](https://github.com/sgl-project/sglang/pull/38884) | SGLang / SM90 MoE | routed activations → CUTLASS grouped GEMM | MoE dispatcher | grouped expert rows | CUTLASS MXFP4×BF16 grouped GEMM | backend-specific expert-major rows | quant/dispatch | 改 dispatcher output rows，固定 CUTLASS backend |
| M5 | [FlashInfer #5208](https://github.com/flashinfer-ai/flashinfer/pull/5208) | FlashInfer/vLLM EP MoE | GEMM2 → Combine | GEMM2 epilogue | local output vs **`(token,k_slot)` route-slot** | reduce/combine | local/peer route-slot buffer | reduce_scatterv / peer-store / local reduce | 改 **GEMM2 destination layout+placement** |
| M6 | [SGLang #32712](https://github.com/sgl-project/sglang/issues/32712) | SGLang / SharedEP | Dispatch → expert execution | EP dispatch | copied/communicated vs shared zero-copy representation | expert GEMM | shared accessible representation | EP communication | 比较 materialized expert buffer vs zero-copy shared buffer |
| M7 | [vLLM #56177](https://github.com/vllm-project/vllm/pull/56177) | vLLM / Qwen3.8-Flash-Next NVFP4 | Host expert refill → Marlin GEMM | expert-cache H2D fill | checkpoint/compact vs **Marlin-converted physical rows** | Marlin MoE | Marlin physical-row representation | load-time conversion, LRU, H2D copy | 改 host-side stored weight representation |
| M8 | [vLLM #38256](https://github.com/vllm-project/vllm/issues/38256) | vLLM MoE offload | Host/GPU expert cache → expert GEMM | offload/cache manager | host-native vs GPU-consumer-native | expert kernel | backend-native weight layout | async prefetch/eviction | 适合扩展 M7 到多个 backend |
| M9 | [SGLang #29971](https://github.com/sgl-project/sglang/pull/29971) | SGLang Paged Experts | Host/paged expert → MoE | expert page loader | paged/offloaded representation | MoE expert kernel | device-native expert layout | page-in/cache | 改 page-in destination format |
| M10 | [vLLM #57794](https://github.com/vllm-project/vllm/issues/57794) | vLLM / expert-granular residency | host experts → GPU cache | residency producer | per-expert compact/cache representation | MoE GEMM | kernel-native | EPLB statistics → prefetch | placement RQ1，结合真实 routing frequency |

### KDA / GDN / recurrent-state

| # | Issue / PR | Framework / real case | Test case | Producer | Producer Layout | Consumer | Consumer Layout | 其他算子 | 应改哪里 |
|---|---|---|---|---|---|---|---|---|---|
| L1 | [FlashInfer #5846](https://github.com/flashinfer-ai/flashinfer/pull/5846) | FlashInfer / Kimi-K3、Kimi-Linear | QKV Projection → KDA Prefill | packed QKV projection | **packed row + strided Q/K/V views** vs dense separate Q/K/V | Cake KDA prefill | both dense and accepted strided pitch | `split(dim=-1)` / old densify copies | 最小：改变 materialization；strict：patch projection direct dense output |
| L2 | [SGLang #34299](https://github.com/sgl-project/sglang/pull/34299) | SGLang / Kimi-K3、Kimi-Linear | KDA Prefill → Decode | KDA prefill checkpoint writer | **native zero-copy checkpoint** | KDA decode | **packed decode state** / backend state pool | optional pack/convert once | patch prefill state writer：native checkpoint vs direct decode-packed |
| L3 | [FlashInfer #5452](https://github.com/flashinfer-ai/flashinfer/pull/5452) | FlashInfer / KDA | KDA prefill → persistent state | KDA prefill | FP32 intermediate/recurrent state/checkpoint representation | later KDA | prepared state contract | plan cache / checkpoint | 状态 precision 与 layout 分因素 |
| L4 | [vLLM #53078](https://github.com/vllm-project/vllm/pull/53078) | vLLM / hybrid GDN/Mamba P/D | Prefill recurrent state → Decode | state exporter | producer TP/state shard layout | remote decode GDN/Mamba | destination TP/state layout | Mooncake/NIXL redistribution | 改 state export packing/partition |
| L5 | [vLLM #58034](https://github.com/vllm-project/vllm/issues/58034) | vLLM / hybrid GDN/Mamba CPU | conv-state producer → consumer | Mamba/GDN state producer | DS vs SD conv layout（issue 中冲突） | CPU platform kernel | forced alternate layout | P/D state handling | 很适合 legality/cross-platform layout case |

### P/D、Host、Storage、Memory Placement

| # | Issue / PR | Framework / real case | Test case | Producer | Producer Layout / Placement | Consumer | Consumer Layout / Placement | 其他算子 | 应改哪里 |
|---|---|---|---|---|---|---|---|---|---|
| P1 | [SGLang #41953](https://github.com/sgl-project/sglang/pull/41953) | SGLang unified PD | Prefill export → transport/import | KV/state exporter | whole-page/whole-slot，含 byte offset/block stride/TP axis | decode importer | destination tensor rows | Mooncake transfer | 枚举 `PrefillNative` / packed canonical / `DecodeNative` wire layout |
| P2 | [SGLang #41958](https://github.com/sgl-project/sglang/pull/41958) | SGLang TP4/DCP2 | Prefill pages → Decode pages | prefill exporter | local DCP page layout | decode local pages | matching DCP layout | whole-page copy，no redistribution | direct page-to-page vs canonical pack/unpack |
| P3 | [SGLang #41182](https://github.com/sgl-project/sglang/pull/41182) | SGLang / GLM-5.3、Kimi-K3 | GPU→Host→Network→Host→GPU | prefill gather | GPU KV/state rows → pinned-host packed slots | decode scatter | destination GPU rows | Triton gather/scatter + NIXL H2H | 改 host staging packing/layout 与 destination-row ordering |
| P4 | [vLLM #59070](https://github.com/vllm-project/vllm/pull/59070) | vLLM ROCm MLA+DCP | FP8 context → AllGather → projection | DCP prefill context producer | FP8 | downstream MLA projection/pack | consumer-ready context | AllGather、dequant、reorg | FP8 wire vs BF16 wire；dtype/layout联合，单独做 layout control |
| P5 | [vLLM #46326](https://github.com/vllm-project/vllm/pull/46326) | vLLM / GLM-5.2 HiSparse | Host KV → GPU hot buffer → sparse decode | `hisparse_swap_in` | pinned-host full KV → compact GPU hot set | Sparse MLA decode | top-k GPU working-set layout | indexer top-k、LRU、gather | 改 hot-buffer destination：token-major vs consumer-native |
| P6 | [vLLM #55398](https://github.com/vllm-project/vllm/pull/55398) | vLLM HiSparse P/D | remote KV → host/GPU landing | NIXL destination mapper | host landing vs **direct GPU final-page landing** | decode | final page/hot-buffer contract | admission + transfer | memory placement 本身作为候选 |
| P7 | [SGLang #41365](https://github.com/sgl-project/sglang/pull/41365) | SGLang HiCache | GPU KV → host L2 pool | backup producer | normal precision vs **INT8 host representation** | restore/decode path | GPU attention representation | quantize/dequantize HBM↔host | layout/precision 分开；placement+compression case |
| P8 | [SGLang #39606](https://github.com/sgl-project/sglang/pull/39606) | SGLang HiCache unified KV | GPU page → L2 | staged write-back producer | page-unified KV layout | host cache | L2 page representation | staging/write-back | page-first vs staging-contiguous |
| P9 | [SGLang #40960](https://github.com/sgl-project/sglang/pull/40960) | SGLang HiCache | KV backup → host | batch backup producer | per-page vs batched buffer-only | host storage | host batch layout | flush batching | transfer granularity RQ1 extension |
| P10 | [SGLang #39784](https://github.com/sgl-project/sglang/pull/39784) | SGLang / DeepSeek-V4 L3 | L2 KV → L3 MemCache | L3 writer | host/cache representation | MemCache L3 | storage-tier representation | serialization/storage | Host-native vs L3-native pack |
| P11 | [vLLM #54129](https://github.com/vllm-project/vllm/pull/54129) | vLLM / Qwen3.8-Flash-Next PLE | mmap table → GPU PLE buffer | mmap row gather | disk-backed row-major checkpoint / optional pinned staging | PLE consumer | stable GPU gathered rows | mmap → page cache → gather → H2D | 改 host gather layout/chunking；placement不是普通 tensor layout |
| P12 | [vLLM #53899](https://github.com/vllm-project/vllm/pull/53899) | vLLM Qwen3.8 PLE | CPU PLE → GPU | offload worker | resident host table | PLE GPU consumer | gathered GPU rows | H2D offload | 与 #54129 比 resident-host vs mmap |
| P13 | [vLLM #58815](https://github.com/vllm-project/vllm/pull/58815) | vLLM Qwen3.8 PLE | checkpoint file → host/GPU | file gather | host-file gathered rows | PLE | GPU row buffer | safetensors/file gather | file range/coalescing vs consumer row order |
| P14 | [vLLM #38260](https://github.com/vllm-project/vllm/issues/38260) | vLLM multi-tier KV | GPU KV → CPU/NVMe tier → GPU | offload connector | GPU-native / canonical tier representation | restored attention | backend-native KV | pack/serialize/restore | system-level multi-tier extension |
| P15 | [TensorRT-LLM #19297](https://github.com/NVIDIA/TensorRT-LLM/pull/19297) | TRT-LLM KVCM V2 | GPU KV → host offload → restore | KVCM residency manager | sparse resident/offloaded layout | attention | device KV layout | host offload | placement + page format |
| P16 | [TensorRT-LLM #18921](https://github.com/NVIDIA/TensorRT-LLM/pull/18921) | TRT-LLM / Qwen3.8-Flash-Next | Prefill → Decode | PD exporter | prefill cache/state representation | remote decode | decode-native representation | disaggregated serving | 对照 producer-native/wire-native/consumer-native |

### Micro `ld/st/MMA` / compiler representation

| # | Issue / PR | Framework | Test case | Producer | Producer layout | Consumer | Consumer layout | 其他操作 | 应改哪里 |
|---|---|---|---|---|---|---|---|---|---|
| U1 | [Triton #8450](https://github.com/triton-lang/triton/pull/8450) | Triton / Paged Attention RFC | GMEM Load → Dot | `LoadOp` | coalesced `Blocked` | `DotOp/MFMA` | `DotOperandEncoding` | `convert_layout` / LDS | HBM operand physical order + load encoding；比较 LDS repair vs direct |
| U2 | [Triton #7968](https://github.com/triton-lang/triton/pull/7968) | Triton AMD | GMEM → MFMA | global operand producer | plain vs **preshuffled coalesced** | MFMA | dot-native | transpose/reshape, bypass LDS | 改 **HBM operand layout**，不是只改 IR tag |
| U3 | [Triton #10342](https://github.com/triton-lang/triton/issues/10342) | Triton | Dot1 → FFMA → Dot2 | Dot1 | accumulator layout | Dot2 | DotOperand | bias/FFMA + convert | 中间 register representation |
| U4 | [Triton #10563](https://github.com/triton-lang/triton/pull/10563) | Triton AMD | WMMA → Store | WMMA accumulator | WMMA-native | global store | coalesced store mapping | convert/reorder | accumulator→store layout |
| U5 | [Triton #11704](https://github.com/triton-lang/triton/pull/11704) | Triton | FMA/MMA → Store | compute result | accumulator-native | `tt.store` | coalesced Blocked | shared/register relayout | producer output encoding |
| U6 | [Triton #8775](https://github.com/triton-lang/triton/issues/8775) | Triton | Reduce → consumer | Reduce | reduction/scratch layout | downstream op | downstream preferred layout | redundant convert | reduction output mapping |
| U7 | [IREE #23782](https://github.com/iree-org/iree/issues/23782), [#22356](https://github.com/iree-org/iree/pull/22356) | IREE / Attention | HBM → LDS → Attention | coalesced DMA/gather | global coalesced representation | MMA/attention | LDS tile/swizzle | Reg staging or direct gather-to-LDS | `HBM→Reg→LDS` vs `HBM→LDS` |
| U8 | [CUTLASS #3313](https://github.com/NVIDIA/cutlass/pull/3313) | CUTLASS Blackwell | TMEM → register epilogue | MMA accumulator TMEM | canonical TMEM fragment | epilogue | register fragment/store mapping | TMEM load atom | sweep `32dp32b1x/16x/32x/128x`；属于 micro-layout |
| U9 | [CUTLASS #3273](https://github.com/NVIDIA/cutlass/pull/3273) | CUTLASS SM120 | quantized global operand → MMA | TMA loader | MXF4/NVFP4 **native-TMA** representation | MMA | native blockscaled operand | TMA staging | compare native-TMA vs transformed input |
| U10 | [TileLang #3256](https://github.com/tile-ai/tilelang/issues/3256) | TileLang / register-resident attention | attention fragment → blockscaled GEMM | register attention output | register-resident fragment | GEMM | blockscaled MMA fragment | scale fragments | direct register consumer vs shared materialization |
| U11 | [TileLang #2642](https://github.com/tile-ai/tilelang/issues/2642) | TileLang | Shared operand → GEMM | shared-buffer producer | annotated shared layout | `T.gemm` | MMA-compatible layout | relayout | legality/control case |
| U12 | [TileLang #2954](https://github.com/tile-ai/tilelang/issues/2954) | TileLang | GEMM operand fragment → Shared | register fragment | GEMM fragment layout | shared copy consumer | shared physical slots | `T.copy` | fragment→shared representation |
| U13 | [TileLang #3283](https://github.com/tile-ai/tilelang/issues/3283), [#3284](https://github.com/tile-ai/tilelang/pull/3284) | TileLang SM120 | Scale producer(s) → blockscaled GEMM | SFA/SFB fragments | potentially conflicting fragment layouts | blockscaled GEMM | separate scale contracts | shared fragment reuse | multi-consumer/alias conflict |
| U14 | [TileLang #3176](https://github.com/tile-ai/tilelang/pull/3176) | TileLang | producer → Reduction | vectorized producer | vector width/layout | reduction | reduction-aware layout | vectorization | micro control：vector width × reduction consumer |

---

## 这张表怎么真正变成 RQ1 实验

对于上面每一行，我建议统一生成四种 arm：

\[
\boxed{
\begin{aligned}
A&:\quad P(L_0)\rightarrow C(L_0)\\
B&:\quad P(L_0)\rightarrow R_{0\to1}\rightarrow C(L_1)\\
C&:\quad P(L_1)\rightarrow C(L_1)\\
D&:\quad \text{Native Framework Auto}
\end{aligned}}
\]

其中 `C` 最关键。`P(L1)` 必须真的修改 producer：

- BMM case：改 GEMM/BMM epilogue destination indexing；
- KV case：改 cache writer physical offset；
- Gather case：改 gather destination layout；
- MoE case：改 dispatch destination rows；
- KDA case：改 projection/state writer；
- PD case：改 exporter/gather/wire packing；
- Offload case：改 host/storage representation；
- micro case：改真正的 HBM/shared/register mapping。

而不是：

\[
P(L_0)\rightarrow transpose\rightarrow L_1
\]

然后把它叫作 `P(L1)`。

完整 RQ1 要分别得到：

\[
L_P^*=\arg\min T_P(L_P)
\]

和

\[
(L_P,R,L_C)^*
=
\arg\min T_{\rm edge}(P\rightarrow R\rightarrow C).
\]

只有：

\[
\boxed{
L_P^*\neq L_{P,\rm edge}^*
}
\]

才应计作严格的 cross-edge rank inversion。

另外每个 case 都应保留 `Native Framework Auto`。最终可以同时报告：

\[
Regret_{\rm native}
=
\frac{T_{\rm native}-T_{\rm oracle}}
{T_{\rm oracle}}.
\]

这样你最终不只是证明“P 和 C 喜欢不同 layout”，还可以回答更有价值的问题：

> **vLLM / SGLang / FlashInfer / TRT-LLM / compiler 当前的原生 heuristic，距离真实 edge oracle 还有多大差距。**

如果按“最值得优先真正实现”排序，我建议第一批先做 `A16 MSA gather→MSA`、`D1 Attention BMM→OProj`、`M1 DeepEP→DeepGEMM`、`L1 packed-QKV→KDA`、`A9 standard-KV→AITER-shuffled`、`P3 PD host staging`、`M7 HostExpert→Marlin`、`U1/U2 HBM→MFMA`。它们之间的 producer、consumer、memory level 和 representation 差异最大，最能解决当前 RQ1 variety 不足。

---

你指出的缺口是对的。真正做 RQ1 时，不能只说“这个 case 有 layout conflict”，必须对每个 case 明确：

\[
\boxed{
P\ 是谁,\quad C\ 是谁,\quad
\text{到底改谁的 layout},\quad
\mathcal L_P/\mathcal L_C\ 有哪些,\quad
\text{为什么这样改}
}
\]

而且严格 RQ1 最好固定 consumer 算法/backend，只改变 producer 输出 representation；否则“换 layout”同时换了 consumer kernel，因果就不干净。

我重新把前面筛出的 case 按这个标准逐个整理。先给一个统一规则：对所有 case，都尽量构造

\[
P(L_0)\rightarrow C(L_0),
\]

\[
P(L_0)\rightarrow R(L_0\to L_1)\rightarrow C(L_1),
\]

\[
P(L_1)\rightarrow C(L_1),
\]

必要时再有

\[
P(L_1)\rightarrow R(L_1\to L_0)\rightarrow C(L_0).
\]

其中最关键的是第三条：`P(L1)` 必须让 producer **直接产生** \(L_1\)，不能先产生 \(L_0\) 再偷偷转成 \(L_1\)。

---

## 1. SGLang：KV-cache writer → TRTLLM MHA Attention

这是最适合替代 v10 synthetic KV case 的真实框架版本。

真实源码在 SGLang `trtllm_mha_backend.py`。源码明确写着：

> Native pool format is NHD

而 decode / SM100 batch-context 路径又明确写：

> require HND layout

源码还提供 `SGLANG_USE_HND_KVCACHE`。  
[源码：trtllm_mha_backend.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/trtllm_mha_backend.py)

### Producer 是谁？

不是 projection GEMM，而是：

\[
\boxed{P=\text{KV cache update/write}}
\]

即当前 token 的 K/V 被写入 persistent paged KV pool 的那一步。

### Consumer 是谁？

\[
\boxed{C=\text{TRTLLM MHA decode/prefill attention}}
\]

严格实验最好固定 `trtllm_mha`，不要同时换 FlashInfer。

### 改谁的 layout？

改 producer 最终写入 KV pool 的 physical layout：

\[
P_{NHD}:\quad
[\text{page},N,H,D]
\]

和：

\[
P_{HND}:\quad
[\text{page},H,N,D].
\]

也就是改 KV writer 的 destination-offset mapping。

### 为什么改 producer？

因为源码已经暴露了真实矛盾：

\[
\text{persistent pool native}=NHD
\]

但：

\[
\text{TRTLLM decode consumer wants}=HND.
\]

所以这是最纯粹的：

\[
L_{P-best}\stackrel?=L_{edge-best}.
\]

### 应测的候选

| Arm | Producer | Repair | Consumer |
|---|---|---|---|
| NN | direct NHD | none | NHD-capable path |
| NH | direct NHD | NHD→HND | TRTLLM HND |
| HH | direct HND | none | TRTLLM HND |
| HN | direct HND | HND→NHD | NHD consumer/path |

如果 TRTLLM 某个具体 kernel 只合法接受 HND，那么 `NN/HN` 不要强行送给它；可以把 consumer-isolation 换成另一个同算法、合法 NHD path，或者将 strict search 限制到：

\[
NHD\ producer+repair
\quad vs\quad
HND\ direct-producer.
\]

这条我会列为 **最高优先级 RQ1 case**。

---

## 2. vLLM：KV cache update → FlashInfer Attention

vLLM 比 `{NHD,HND}` 更有意思，因为 cache layout 包含 Layer/Block 轴。

当前源码中 FlashInfer 在 SM100 上明确支持：

\[
\boxed{\{LBHNC,\ BLHNC\}}
\]

原因是 TRTLLM-gen kernels 要 head-major block interiors，而外部 `Layer/Block` nesting 对 kernel 不重要。  
[FlashInfer backend 源码](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flashinfer.py)

相反，FlexAttention 是：

\[
\boxed{\{LBNHC\}}
\]

因为它需要把 `(B,N)` flatten 成 zero-copy token dimension。  
[FlexAttention backend 源码](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flex_attention.py)

### Strict RQ1 推荐用谁？

用：

\[
\boxed{P=\text{vLLM KV cache writer}}
\]

\[
\boxed{C=\text{FlashInfer attention}}
\]

并固定 FlashInfer。

### 改 producer 什么？

让 KV update 直接产生：

\[
LBHNC=[L,B,H,N,C]
\]

或：

\[
BLHNC=[B,L,H,N,C].
\]

注意这里 **H/N 顺序不变**，变化的是：

\[
L\leftrightarrow B.
\]

这是和 v10 完全不同的一类 layout trade-off。

### 候选

\[
\boxed{\mathcal L_P=\{LBHNC,BLHNC\}}
\]

同一个 FlashInfer consumer 两种都合法，因此非常适合 strict RQ1。

测：

\[
P(LBHNC)\rightarrow FI(LBHNC)
\]

vs

\[
P(BLHNC)\rightarrow FI(BLHNC).
\]

如果 producer-local 更喜欢一种而完整 edge 更喜欢另一种，就是严格 inversion。

### Flex vs FlashInfer 呢？

vLLM Issue #55312：

[Issue #55312](https://github.com/vllm-project/vllm/issues/55312)

暴露的是：

\[
Flex=\{LBNHC\}
\]

与某些 FlashInfer/TRTLLM paths：

\[
\{LBHNC,BLHNC\}
\]

可以完全不相交。

但这个 case 同时改变 consumer backend，因此我不会把它作为 strict single-consumer RQ1；它更适合：

\[
\text{multi-consumer / shared persistent layout extension}.
\]

---

## 3. SGLang MiniMax-M3：KV Gather/Prepare → MSA Sparse Attention

这是非常强的真实 operator-level case。

PR：  
[SGLang #34525](https://github.com/sgl-project/sglang/pull/34525)

PR 写得非常清楚：

SGLang persistent pool：

\[
[\text{slot},H_{kv},D]
\]

即 NHD-like；

`fmha_sm100` sparse prefill consumer：

\[
[\text{page},H_{kv},page\_size,D]
\]

即 paged-HND。

### Producer 是谁？

这里不要把最早 KV writer 定义成 producer。

更合适的是：

\[
\boxed{P=\text{sparse KV gather/prepare}}
\]

因为这个算子真正决定交给 MSA 的 layout。

### Consumer

\[
\boxed{C=\text{fmha\_sm100 MSA sparse prefill}}
\]

### 改谁？

改 `gather_kv_*` 的输出 mapping。

现在 PR 的新 kernel 是：

```text
NHD full pool
    ↓
gather_kv_hnd
    ↓
compact contiguous HND
    ↓
MSA
```

### 候选 layout

我建议真正实现三种 producer 输出：

\[
L_0=CompactNHD
\]

\[
L_1=CompactHND
\]

\[
L_2=PagedCompactHND.
\]

再保留原始：

\[
FullPoolNHD\rightarrow permute/view/materialize.
\]

于是完整 arms：

| Arm | Gather output | Repair | MSA |
|---|---|---|---|
| Native-old | full NHD pool | whole-pool permute+contiguous | HND |
| Gather-NHD | compact NHD | NHD→HND | HND |
| Gather-HND | compact HND | none | HND |

### 为什么这是好 RQ1？

PR 已经证明 whole-pool HND materialization 非常昂贵；但还没有回答：

> `gather_kv_hnd` 自己直接写 HND，是不是 producer-local 最快？

所以应该测：

\[
T_{gather}(NHD)
\quad vs\quad
T_{gather}(HND)
\]

再比较：

\[
T_{gather\rightarrow MSA}.
\]

如果：

\[
gather\text{-only prefers NHD}
\]

但：

\[
edge prefers HND,
\]

就是标准 RQ1。

---

## 4. SGLang：Attention value BMM → O-Proj

PR：

[SGLang #34498](https://github.com/sgl-project/sglang/pull/34498)

这是我认为整个 suite 里最干净的 operator→operator case。

### Producer

\[
\boxed{
P=\text{absorbed value BMM}
}
\]

具体 kernel：

`batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant`

### Consumer

\[
\boxed{
C=O\text{-Proj GEMM}
}
\]

### 原生冲突

旧 producer 输出：

\[
L_H=[H,T,D_v]
\]

而 `o_proj` 前需要：

\[
L_T=[T,H,D_v].
\]

于是旧路径：

```text
BMM [H,T,D]
     ↓
transpose + contiguous copy
     ↓
[T,H,D]
     ↓
flatten
     ↓
O-Proj
```

PR 改成 BMM 直接：

\[
[T,H,D].
\]

### 改谁的 layout？

只改 **BMM output epilogue mapping**。

具体就是：

- `transpose_bm=False` → `[H,T,D]`
- `transpose_bm=True` + preallocated output → `[T,H,D]`

### 候选

\[
\boxed{
\mathcal L_P=
\{
[H,T,D],
[T,H,D]
\}
}
\]

如果 O-Proj backend 支持 strided input，还可以增加：

\[
[H,T,D]\text{ 的 zero-copy stride view}
\]

但只有 consumer contract 真接受才加。

### RQ1 测法

producer-only：

\[
T_{BMM}(H,T,D)
\]

vs

\[
T_{BMM}(T,H,D).
\]

edge：

\[
BMM(H,T,D)
\rightarrow transpose
\rightarrow OProj
\]

vs

\[
BMM(T,H,D)
\rightarrow OProj.
\]

这条几乎是教科书式 RQ1。

---

## 5. SGLang DeepEP：Dispatch → Grouped Expert GEMM

PR：

[SGLang #37261](https://github.com/sgl-project/sglang/pull/37261)

真实模型测试里有 Qwen3-30B-A3B、MiMoV2 等。

### Producer

\[
\boxed{
P=\text{DeepEP dispatch}
}
\]

包括 token 根据 expert routing 被复制/交换到 expert rank。

### Consumer

\[
\boxed{
C=\text{DeepGEMM grouped expert GEMM}
}
\]

### 原始 representation

旧 prefill：

\[
L_0=\text{non-expanded dispatch layout}
\]

之后需要：

\[
ep\_scatter
\]

得到 grouped-by-expert rows。

### 新 representation

PR 让 DeepEP dispatch 自己直接产生：

\[
L_1=\text{expanded expert-major}
\]

即：

\[
(token,expert)
\]

展开，并按 expert 分组。

### 改谁？

改：

\[
\boxed{\text{DeepEP dispatch 的 destination row layout}}
\]

不要改 DeepGEMM。

### 候选

最小严格集合：

\[
\boxed{
\{
NonExpanded,
ExpandedExpertMajor
\}
}
\]

repair：

\[
NonExpanded\rightarrow ep\_scatter\rightarrow Expanded.
\]

如果结合 vLLM activation contracts，可以扩到：

\[
Standard,\quad
PaddedStandard,\quad
BatchedExperts.
\]

但不要一次混在一个实验里，最好分两组。

### 测法

\[
T_{dispatch}(NonExpanded)
\]

vs

\[
T_{dispatch}(Expanded).
\]

完整：

\[
Dispatch(NonExpanded)
\rightarrow ep\_scatter
\rightarrow DeepGEMM
\]

vs

\[
Dispatch(Expanded)
\rightarrow DeepGEMM.
\]

这回答：

\[
L_{\rm dispatch-best}
\stackrel?=
L_{\rm MoE-edge-best}.
\]

---

## 6. vLLM DeepEPV2：Prepare/Finalize → Expert kernel

PR：

[vLLM #54109](https://github.com/vllm-project/vllm/pull/54109)

这个 case 和上一个不同，它研究的是 **activation contract**，不是 expert-major permutation 本身。

### Producer

\[
\boxed{
P=\text{DeepEPV2 Prepare/Finalize}
}
\]

它为 expert kernel 准备 activation tensor。

### Consumer

\[
\boxed{
C=\text{FusedMoE expert kernel}
}
\]

### 真实格式

PR 明确定义：

\[
Standard
\]

\[
PaddedStandard
\]

\[
BatchedExperts.
\]

其中 `PaddedStandard` 的 shape 仍是 2D：

\[
[num\_tokens,hidden]
\]

但 token dimension 中存在 inactive/padded rows，用 `topk_ids=-1` 表示。

### 改谁？

改 Prepare 阶段输出的：

\[
\boxed{\text{activation row contract}}
\]

而不是专家 GEMM 的 weight layout。

### 为什么？

不同 expert kernels 对 padding contract 的接受能力不同。

因此这里其实测试：

\[
\text{compact exact rows}
\]

vs

\[
\text{padded fixed-capacity rows}.
\]

### 候选

严格在 consumer 都支持的交集里测：

\[
\boxed{
\{
Standard,
PaddedStandard
\}
}
\]

`BatchedExperts` 是 disjoint representation，只有选定 expert kernel 原生支持时才加入。

### 很重要

这条 case 不能只看 latency，因为 `PaddedStandard` 的价值之一是避免 host-side token-count sync / 支持 CUDA graph。

所以完整 edge 应包括：

\[
Prepare
\rightarrow Expert
\]

以及 graph-capture regime。

---

## 7. FlashInfer：Expert GEMM2 → MoE Combine

PR：

[FlashInfer #5208](https://github.com/flashinfer-ai/flashinfer/pull/5208)

### Producer

\[
\boxed{
P=\text{Expert GEMM2 epilogue}
}
\]

### Consumer

\[
\boxed{
C=\text{cross-GPU combine / local weighted reduction}
}
\]

### 原生 layout/placement

Baseline GEMM2 写 local output，然后：

\[
reduce\_scatterv.
\]

可抽象为：

\[
L_0=LocalTokenOutput.
\]

### Alternate

PR 为每个：

\[
(token,k\_slot)
\]

分配 destination slot，并让 GEMM2 epilogue 直接 peer-store 到 owning GPU：

\[
L_1=PeerRouteSlotMajor.
\]

### 改谁？

改：

\[
\boxed{
GEMM2 epilogue 的 destination layout + memory placement
}
\]

不是改 combine kernel。

### 候选

\[
\boxed{
\{
LocalTokenMajor,
LocalRouteSlotMajor,
PeerRouteSlotMajor
\}
}
\]

其中第三个是跨 GPU placement。

### 路径

Baseline：

\[
GEMM2\rightarrow LocalHBM
\rightarrow ReduceScatter.
\]

Alternate：

\[
GEMM2\rightarrow PeerHBM(route-slot)
\rightarrow LocalReduce.
\]

这是 layout + placement 的 RQ1。

---

## 8. FlashInfer：Packed QKV projection → KDA Prefill

PR：

[FlashInfer #5846](https://github.com/flashinfer-ai/flashinfer/pull/5846)

这是 stride-layout 方向最好的真实 case。

### Producer

从系统角度：

\[
\boxed{
P=\text{QKV projection}
}
\]

它实际生成一行：

\[
[T,3\cdot H\cdot128].
\]

### Consumer

\[
\boxed{
C=\text{Cake KDA prefill}
}
\]

### 两种 representation

真实 serving 通过：

```python
split(dim=-1)
```

得到 Q/K/V。

它们逻辑 shape 是：

\[
[T,H,128]
\]

但 token stride 仍来自 packed row。

因此：

\[
L_0=PackedRow+StridedViews.
\]

旧部分 KDA body 要 dense pitches，于是 densify：

\[
L_1=DenseSeparateQKV.
\]

### 改谁？

这里有两个层次。

最小实验：不改 projection，改 **boundary materialization policy**：

\[
PackedStrided
\]

vs

\[
DenseSeparate.
\]

更严格的 RQ1：修改 QKV projection epilogue，让它直接产生：

\[
DenseSeparateQKV.
\]

这样才能测真正：

\[
T_P(Packed)
\]

vs

\[
T_P(DenseSeparate).
\]

### 候选

\[
\boxed{
\{
PackedRow,
PackedRow+StridedViews,
DenseSeparateQKV
\}
}
\]

其中 `PackedRow` 本身是 storage；`StridedViews` 是 zero-copy boundary contract。

PR 当前 `_validate_qkv_layout` 已经明确允许：

- dense `[B,T,H,128]`
- packed-row strided views。

因此这不是人为想象的 candidate。

---

## 9. SGLang：KDA Prefill → Persistent State → Decode

PR：

[SGLang #34299](https://github.com/sgl-project/sglang/pull/34299)

### Producer

\[
\boxed{
P=\text{KDA prefill kernel}
}
\]

更准确地说，是它产生/保存 recurrent state checkpoint 的部分。

### Consumer

\[
\boxed{
C=\text{KDA decode kernel}
}
\]

### 当前框架已经出现的两类 representation

PR 明确区分：

- `native zero-copy prefill checkpoints`
- `packed Cake decode`

所以可以定义：

\[
L_P=\text{PrefillNativeCheckpoint}
\]

和：

\[
L_D=\text{DecodePackedState}.
\]

### 改谁？

为了 strict RQ1，应修改：

\[
\boxed{
\text{prefill checkpoint/state writer}
}
\]

让它直接写：

1. prefill-native checkpoint layout；
2. decode-packed layout。

第二条如果当前 kernel 不支持，就做最小 patch。

### Arms

\[
PrefillNative
\rightarrow Decode
\]

\[
PrefillNative
\rightarrow Pack
\rightarrow DecodePacked
\]

\[
PrefillDirectPacked
\rightarrow DecodePacked.
\]

### 为什么值得测？

这是 persistent state，而不是瞬时 activation。

所以 producer-local 最佳 layout 可能更利于 prefill checkpoint emission，而 edge 最佳可能更利于后续几十/几百个 decode steps。

这条应再扫：

\[
reuse/decode\_steps.
\]

---

## 10. SGLang：Prefill-side KV export → Decode-side KV import

真实 PR：

- [#41953 Expose unified memory transfer layouts](https://github.com/sgl-project/sglang/pull/41953)
- [#41958 matching DCP peers](https://github.com/sgl-project/sglang/pull/41958)
- [#41182 NIXL HOST staging](https://github.com/sgl-project/sglang/pull/41182)

这里不要把整个 Transformer prefill 当 producer。

### Producer

\[
\boxed{
P=\text{prefill-side KV/state export/gather}
}
\]

### Consumer

\[
\boxed{
C=\text{decode-side import/scatter + first use}
}
\]

### 原生 representation

#41953 暴露真实 transfer layout：

- layer id；
- byte offset；
- payload size；
- block stride；
- TP shard axis/group。

#41958 在 matching DCP 时直接：

\[
P\ local\ pages
\rightarrow
D\ local\ pages
\]

不需要 packing/redistribution。

### 改谁？

改：

\[
\boxed{
\text{exporter/gather producer 输出的 wire representation}
}
\]

### 候选

我建议三个：

\[
L_0=PrefillNativeWholePage
\]

\[
L_1=PackedCanonicalWire
\]

\[
L_2=DecodeNativePage.
\]

其中：

- \(L_0\) / direct whole-page 已有原生机制；
- \(L_1\) 是实验需要增加的 canonical pack；
- \(L_2\) 是让 source gather 直接按 destination row order 产生。

### HOST staging 再增加 placement

#41182 真实路径：

\[
GPU_P
\rightarrow PinnedHost
\rightarrow PinnedHost
\rightarrow GPU_D.
\]

因此可以再比较：

\[
DirectGPU/registered-memory
\]

vs

\[
HostPackedStaging.
\]

这里真正改的是 exporter 的 layout/placement，不要改模型计算。

---

## 11. vLLM HiSparse：Host KV → GPU hot buffer → Sparse MLA Decode

PR：

[vLLM #46326](https://github.com/vllm-project/vllm/pull/46326)

以及 direct landing：

[vLLM #55398](https://github.com/vllm-project/vllm/pull/55398)

### Producer

对于 RQ1，不应该叫“host memory”。

真正 producer 是：

\[
\boxed{
P=\text{hisparse swap-in/gather}
}
\]

它把 host-resident KV miss rows 搬到 GPU hot buffer。

### Consumer

\[
\boxed{
C=\text{sparse MLA decode}
}
\]

### 当前 native

full sparse KV：

\[
HostPinned.
\]

GPU 上只保留：

\[
\text{per-request top-k hot buffer}.
\]

### 改谁？

改 swap-in/gather kernel 的 GPU destination representation。

### 候选

可以做：

\[
L_0=FullPageNative
\]

\[
L_1=CompactTopKTokenMajor
\]

\[
L_2=CompactTopKConsumerNative.
\]

其中 `CompactTopK` 是 HiSparse 的真实机制；`consumer-native` 需要根据实际 sparse MLA kernel stride contract 做最小 patch。

### 为什么？

H2D producer 最喜欢的往往是：

\[
\text{large contiguous copy}.
\]

但 consumer 可能更喜欢：

\[
\text{compact, head-/page-organized hot buffer}.
\]

所以这是非常典型的：

\[
transfer\text{-}best
\neq
decode\text{-}edge-best.
\]

---

## 12. vLLM：Host expert pool → Marlin MoE GEMM

PR：

[vLLM #56177](https://github.com/vllm-project/vllm/pull/56177)

这条很有价值，因为它不是 activation/KV，而是 weight representation。

### Producer

\[
\boxed{
P=\text{expert-cache H2D fill}
}
\]

即 cache miss 时，把 expert rows 从 pinned host memory 填到 GPU expert bank。

### Consumer

\[
\boxed{
C=\text{Marlin MoE expert GEMM}
}
\]

### PR 当前其实已经做了一个重要选择

NVFP4 expert weights 在 load time：

1. 先有原始 checkpoint representation；
2. 做一次 device round-trip；
3. 做 Marlin conversion；
4. 把 **converted representation 再放回 pinned host memory**。

随后 cache miss 时直接 row-copy 到 GPU bank。

也就是说 native 本身已经在做：

\[
\boxed{
\text{host 也存 consumer-native Marlin representation}
}
\]

而不是每次 H2D 后转换。

### RQ1 应该比较什么？

两个 producer representations：

\[
L_0=Checkpoint/CompactNVFP4
\]

\[
L_1=MarlinConvertedPhysicalRows.
\]

### 路径

\[
HostCompact
\rightarrow H2D
\rightarrow MarlinConvert
\rightarrow GEMM
\]

vs

\[
HostMarlin
\rightarrow H2D
\rightarrow GEMM.
\]

### 改谁？

改：

\[
\boxed{
\text{host expert-cache 的 stored representation}
}
\]

以及 refill producer 是否带 conversion。

这个 case 特别适合回答：

> storage-local optimal format 是否应该服从 GPU compute consumer？

---

# 13. Micro-control：Triton GMEM load → MFMA/Dot

真实 PR：

- [Triton #8450](https://github.com/triton-lang/triton/pull/8450)
- [Triton #7968](https://github.com/triton-lang/triton/pull/7968)

这个不是主要 external-validity case，但应该作为机制 control。

### Producer

\[
P=LoadOp/global-memory load.
\]

### Consumer

\[
C=DotOp/MFMA.
\]

### 真实矛盾

HBM load 喜欢：

\[
Blocked/coalesced.
\]

MFMA consumer 喜欢：

\[
DotOperand/MFMA layout.
\]

默认：

\[
HBM
\rightarrow Reg(Blocked)
\rightarrow LDS
\rightarrow Reg(Dot)
\rightarrow Dot.
\]

preshuffled/direct：

\[
HBM
\rightarrow Reg(Dot-compatible)
\rightarrow Dot.
\]

### 改谁？

改：

\[
\boxed{\text{global operand physical/preshuffled layout}}
\]

以及 load result encoding。

候选：

\[
Blocked,
PreshuffledDotCompatible,
DotOperand.
\]

这里不要只插 `convert_layout`；要真正比较 operand 在 HBM 中如何存。

---

# 14. Micro-control：CUTLASS TMEM → Registers → Epilogue

PR：

[CUTLASS #3313](https://github.com/NVIDIA/cutlass/pull/3313)

### Producer

\[
P=\text{MMA accumulator in TMEM}.
\]

### Consumer

\[
C=\text{register epilogue/store}.
\]

### 改谁？

这里不是改 global tensor axis order，而是改：

\[
\boxed{
TMEM\rightarrow register\ copy\ layout/atom width
}
\]

候选 PR 已经实际 sweep：

\[
32dp32b1x,
32dp32b16x,
32dp32b32x,
32dp32b128x
\]

一类。

它证明“更宽并不一定更好”，所以可以作为非常好的底层 control，但不要和上面 operator-level layout 混为同一种 `layout`。

---

## 哪些 case 最适合真正进入 strict RQ1？

如果你的目标是论文主实验，而不是把所有东西都做一遍，我会按可归因性分三档。

| Case | 推荐程度 | 原因 |
|---|---:|---|
| SGLang KV writer→TRTLLM MHA | ★★★ | NHD/HND 冲突明确、真实 cache |
| Attention BMM→OProj | ★★★ | 两个真实算子、只有 output layout 变化 |
| MSA gather→Sparse Attention | ★★★ | NHD/HND conflict + real MiniMax-M3 |
| DeepEP Dispatch→Grouped GEMM | ★★★ | real MoE representation conflict |
| Packed QKV→KDA | ★★★ | real stride contract，Kimi-K3 |
| Host expert→Marlin | ★★★ | weight representation + placement |
| KDA Prefill→Decode | ★★☆ | persistent-state representation，需额外 direct-pack patch |
| GEMM2→Combine | ★★☆ | layout + communication placement 混合 |
| PD export→import | ★★☆ | system-level，变量更多 |
| HiSparse swap-in→decode | ★★☆ | placement+layout，需要 consumer-native hot-buffer patch |
| Triton GMEM→MFMA | mechanism | 太底层，不作为主要 external validity |
| CUTLASS TMEM→Reg | mechanism | 同上 |

---

## 最关键的实验实现原则

对每一个 case，你都应该把“改 layout”落到 **producer 的最终 write mapping** 上。

也就是说：

\[
\boxed{
\text{不要：}
P(L_0)\rightarrow transpose\rightarrow L_1
\quad\text{然后声称测了 }P(L_1)
}
\]

真正的 producer variant 必须：

\[
\boxed{
P_{L_1}\text{ 自己直接写出 }L_1.
}
\]

而 `transpose/copy/gather/pack` 必须作为独立 repair arm：

\[
P(L_0)
\rightarrow
R(L_0,L_1)
\rightarrow
C(L_1).
\]

这样才能区分：

\[
T_P(L_0)
\]

\[
T_P(L_1)
\]

和：

\[
T_R(L_0\rightarrow L_1).
\]

最后 strict RQ1 判据仍然是：

\[
\boxed{
\arg\min_{L_P}T_P(L_P)
\neq
L_P\text{ component of }
\arg\min_{L_P,R,L_C}T_E(L_P,R,L_C)
}
\]

而不是只看到 consumer 喜欢另一种 layout 就宣布 RQ1 成立。

另一个关键点是：像 vLLM #57097 这种把 QK-Norm、RoPE、KV write 三个 kernel 融成一个的 PR，适合做 **native reproduction**，但不适合直接拿 PR before/after 当 strict RQ1，因为它同时改变了 fusion、launch count 和 layout materialization。严格 RQ1 必须另外做 layout-only controlled arms。

如果按这个原则实施，上面每一个 case 都能明确回答“究竟改谁、改成什么、为什么、是否只是 conversion、以及最终 edge 是否真的发生 rank inversion”。

---

可以。要把这些 case 真正做成符合 RQ1 的实验，核心不是“给 tensor 做几种 transpose”，而是对每一条真实 P→C edge 明确四件事：

\[
\boxed{
\text{改谁的输出 layout}
\quad
\text{consumer 接受什么}
\quad
\text{中间 repair 是什么}
\quad
\text{完整 edge 怎么直接计时}
}
\]

同时必须保留“原生框架未修改路径”作为 baseline。否则只能证明某个自制 kernel 更快，不能回答框架当前策略是否有 regret。

我建议统一采用下面的实验定义：

\[
E=(P,C,\mathcal L_P,\mathcal L_C,\mathcal R)
\]

其中

\[
\mathcal L_P=\text{producer 可直接产生的 representations}
\]

\[
\mathcal L_C=\text{consumer 合法接受的 representations}
\]

\[
\mathcal R(L_P,L_C)=
\{
identity,\ view,\ transpose,\ copy,\ gather,\ pack,\ transfer,\ldots
\}
\]

然后真正搜索：

\[
\boxed{
(L_P,R,L_C)^*
=
\arg\min
T_{\rm edge}(P(L_P)\rightarrow R\rightarrow C(L_C))
}
\]

而不是只比较 \(P\) 和 \(C\) 单独的时间。

---

## 一、每个 case 都固定做 4 类路径

这是最重要的统一规范。

对于每条真实 P→C，至少跑：

| Path | 含义 |
|---|---|
| `NATIVE` | 完全不改框架，让 vLLM/SGLang/FlashInfer 等自己选择 |
| `P-DIRECT(L)` | 修改 producer，使其**直接产生** layout \(L\)，不允许先产生 A 再偷偷 transpose 成 L |
| `P-NATIVE + REPAIR` | producer 保持原生最优/原生输出，中间显式转换成 consumer layout |
| `EDGE-ORACLE` | 枚举所有合法 `P layout × repair × C layout`，找完整 edge 最快方案 |

然后额外做 consumer-only：

\[
C(L)
\]

和 producer-only：

\[
P(L)
\]

用于解释 preference。

最终报告：

\[
L_P^*=\arg\min_L T_P(L)
\]

\[
L_C^*=\arg\min_L T_C(L)
\]

\[
(L_P,R,L_C)_E^*
=\arg\min T_E
\]

严格的 RQ1 positive 应该是：

\[
\boxed{
L_P^*
\neq
L_{P,E}^*
}
\]

注意，不是仅仅：

\[
L_P^*\neq L_C^*.
\]

后者只叫 preference conflict。

---

# 二、Case 1：QKV/QK-Norm/RoPE/KVPrep → GQA Attention

真实来源例如：

- vLLM Qwen3.8 QSA：[PR #57097](https://github.com/vllm-project/vllm/pull/57097)
- vLLM GQA/MLA/SparseMLA audit：[Issue #46979](https://github.com/vllm-project/vllm/issues/46979)
- SGLang MiniMax-M3：[PR #35357](https://github.com/sgl-project/sglang/pull/35357)

边界应该定义在：

```text
QKV projection
   ↓
Norm/RoPE
   ↓
KV preparation  ← Producer结束位置
   ↓
persistent/paged K,V
   ↓
real Attention  ← Consumer
```

### 改哪个 layout？

改：

\[
\boxed{\text{KVPrep/KV-write 的输出 physical layout}}
\]

不是改 QKV GEMM 内部 tile。

Producer variants：

\[
NHD,\quad HND,\quad paged\_NHD,\quad paged\_HND
\]

如果是 vLLM，再对应实际合法：

\[
LBNHC,LBHNC,BLNHC,BLHNC.
\]

实现时直接改 producer store address：

```text
logical K[b,t,h,d]
          ↓
physical_offset_NHD(...)
```

或

```text
physical_offset_HND(...)
```

不能：

```text
producer always NHD
→ transpose
→ 假装这是 HND producer
```

否则你测的是 repair，不是 producer preference。

### Consumer

必须使用框架真正 attention backend，例如：

- FlashInfer
- FlashAttention
- TRTLLM
- AITER
- Sparse backend

只运行其合法 layout。

如果 backend 不接受 NHD：

```text
P(NHD)
→ explicit NHD→HND repair
→ C(HND)
```

而不是强行把非法 stride 喂进去。

### 测量

\[
T_P(NHD),T_P(HND)
\]

预构造 cache 后：

\[
T_C(NHD),T_C(HND)
\]

然后直接计时：

\[
P(NHD)\rightarrow C(NHD)
\]

\[
P(NHD)\rightarrow convert\rightarrow C(HND)
\]

\[
P(HND)\rightarrow C(HND)
\]

完整 edge 要 CUDA event 包住整个 pipeline，不能把三个 median 相加。

### 原生框架 baseline

直接运行未修改：

```text
vLLM main / SGLang main
same backend
same model
same shape
```

记录：

- 实际选择的 backend；
- 实际 KV layout；
- 是否发生 conversion/materialization；
- edge latency。

最后得到：

\[
Regret_{\rm native}
=
\frac{T_{\rm native}-T_{\rm oracle}}
{T_{\rm oracle}}.
\]

---

# 三、Case 2：MLA projection/pack → MLA Attention

真实来源：

[FlashInfer PR #5667](https://github.com/flashinfer-ai/flashinfer/pull/5667)

真实边：

```text
kv_b_proj
   ↓
[k_nope | v] + k_pe
   ↓
pack / quant / concatenate
   ↓
K,V representation
   ↓
MLA Attention
```

### Producer 改什么？

改：

\[
\boxed{\text{kv\_b\_proj 后的 K/V physical representation}}
\]

例如固定 dtype=BF16 时先比较：

\[
\{
separate\ K/V,
packed\ K/V,
consumer\text{-}native\ contiguous
\}.
\]

FP8 再作为单独 factorial：

\[
dtype\in\{BF16,FP8\}.
\]

不要把：

> layout + BF16→FP8

混成一个变量。

### 更强的 representation experiment

第二层可以比较：

\[
latent
\]

vs

\[
expanded\ K/V.
\]

但这里已经不只是 layout，而是 computation placement，也就是：

\[
\text{representation-level RQ1}.
\]

论文里要单独标出来。

---

# 四、Case 3：Sparse KV Gather/Prepare → Sparse Attention

最好的真实来源之一：

[SGLang PR #34525](https://github.com/sgl-project/sglang/pull/34525)

真实：

```text
Persistent NHD KV pool
       ↓
selected pages/tokens
       ↓
gather / compact / transpose
       ↓
compact sparse-KV
       ↓
MSA/Sparse Attention
```

### 改什么？

这里 producer 应定义成：

\[
P=\text{sparse gather/prepare}
\]

而不是最早的 KV writer。

让它分别直接生成：

\[
CompactNHD
\]

\[
CompactHND
\]

\[
PagedCompactNHD
\]

\[
PagedCompactHND.
\]

另外加入：

```text
Full pool direct read
```

作为“不做 gather”的 strategy。

于是比较：

\[
FullNHD\rightarrow SparseAttention
\]

\[
FullNHD\rightarrow GatherHND\rightarrow SparseAttention
\]

\[
GatherNativeHND\rightarrow SparseAttention.
\]

这个 case 特别适合证明：

> producer-local 最快的 gather representation 不一定是 sparse-attention edge 最快 representation。

---

# 五、Case 4：Attention BMM → O-Proj

真实来源：

[SGLang PR #34498](https://github.com/sgl-project/sglang/pull/34498)

这是我最推荐的 operator-level case。

### Producer

\[
P=\text{Attention value BMM}
\]

### Consumer

\[
C=O\text{-Proj GEMM}
\]

### 改 producer 的什么？

直接修改 BMM output buffer layout / epilogue mapping：

\[
L_1=[H,T,D_v]
\]

\[
L_2=[T,H,D_v]
\]

\[
L_3=[T,H D_v].
\]

例如：

```text
BMM → [H,T,D]
```

和：

```text
BMM → [T,H,D]
```

应该都由 BMM 自己直接写出。

然后 repair path：

```text
BMM [H,T,D]
→ transpose/materialize
→ [T,H,D]
→ OProj
```

以及 zero-copy：

```text
BMM output
→ stride view
→ OProj
```

如果 GEMM backend 接受。

### RQ1

producer-local：

\[
\arg\min_L T_{\rm BMM}(L)
\]

edge：

\[
\arg\min_L T_{\rm BMM\rightarrow OProj}(L).
\]

这会是非常干净的 rank inversion case。

---

# 六、Case 5：MoE Dispatch → Grouped GEMM

真实来源：

- [SGLang PR #37261](https://github.com/sgl-project/sglang/pull/37261)
- [vLLM PR #54109](https://github.com/vllm-project/vllm/pull/54109)
- [SGLang PR #39039](https://github.com/sgl-project/sglang/pull/39039)

### Producer

\[
P=\text{Router + Dispatch}
\]

### Consumer

\[
C=\text{Grouped Expert GEMM}.
\]

### Producer 直接生成：

\[
TokenMajor
\]

\[
Standard
\]

\[
PaddedStandard
\]

\[
ExpandedExpertMajor
\]

\[
BatchedExperts
\]

\[
DeepGEMM\text{-}psum.
\]

重点是“direct emit”。

例如不要只测：

```text
DeepEP Standard
→ ep_scatter
→ ExpertMajor
```

还应该实现：

```text
DeepEP
→ directly ExpandedExpertMajor
```

这正是 #37261 的思想。

### repair 集合

\[
\{
scatter,
sort,
expand,
pad,
permute,
identity
\}.
\]

### Consumer-only

预先生成完全相同 routing/token data 的每种合法 format，再测：

\[
GroupedGEMM(L).
\]

### Edge

必须跑真实：

```text
Router/Dispatch
→ optional transform
→ Grouped GEMM
```

不能单纯：

\[
T_{dispatch}+T_{gemm}.
\]

---

# 七、Case 6：Expert GEMM2 → Combine

真实来源：

[FlashInfer PR #5208](https://github.com/flashinfer-ai/flashinfer/pull/5208)

这里需要同时改变 layout 和 memory placement。

### Producer

\[
P=\text{GEMM2 epilogue}.
\]

直接产生：

\[
LocalContiguous
\]

或：

\[
RouteSlotMajor
\]

甚至：

\[
PeerRouteSlotMajor.
\]

### Consumer

\[
C=\text{combine/reduce-scatter/local weighted reduction}.
\]

### 路径

Baseline：

```text
GEMM2
→ local GPU HBM
→ combine collective
→ destination GPU
```

Alternate：

```text
GEMM2 epilogue
→ direct peer-GPU route-slot buffer
→ local reduction
```

这里：

\[
M_P=Reg/SMEM,
\quad
M_C=PeerHBM.
\]

所以它也是 memory-placement RQ1。

### 注意

通信 overlap 会导致：

\[
T_E\neq T_P+T_R+T_C.
\]

因此完整 edge wall-time 更重要。

---

# 八、Case 7：Packed QKV → KDA/GDN

真实来源：

[FlashInfer PR #5846](https://github.com/flashinfer-ai/flashinfer/pull/5846)

### Producer

真实 QKV projection 输出：

\[
[T,3HD].
\]

候选输出：

\[
PackedRow
\]

\[
DenseSeparate(Q,K,V)
\]

以及：

\[
PackedRow + zero\text{-}copy\ strided\ split.
\]

### Consumer

真实 KDA/GDN prefill kernel。

### 最重要的三条路径

```text
Projection → packed
→ strided Q/K/V
→ KDA
```

```text
Projection → packed
→ materialize dense Q/K/V
→ KDA
```

```text
Projection directly emits dense Q/K/V
→ KDA
```

第三条需要修改 projection epilogue，否则无法真正判断：

\[
L_{P-best}
\]

是不是 packed。

这是一个很好的 stride-contract RQ1。

---

# 九、Case 8：KDA/GDN Prefill → Decode State

真实来源：

[SGLang PR #34299](https://github.com/sgl-project/sglang/pull/34299)

### Producer

\[
P=\text{Prefill}.
\]

修改 persistent-state store：

\[
PrefillNativeCheckpoint
\]

vs

\[
DecodePackedState.
\]

### Consumer

\[
C=\text{Decode KDA/GDN}.
\]

三条路径：

```text
Prefill-native
→ Decode directly
```

```text
Prefill-native
→ pack once
→ Decode
```

```text
Prefill directly writes DecodePacked
→ Decode
```

### 这里不能只测一个 decode step

因为这是 persistent state。

应该做：

\[
reuse\in\{1,4,16,64\}
\]

或实际 request decode horizon。

这和 RQ3 有交叉，但 RQ1 只负责回答：

> 最佳 producer state layout 与完整 prefill→decode edge layout 是否一致。

---

# 十、Case 9：Draft/MTP → Verify

这里需要谨慎。

[vLLM Issue #55312](https://github.com/vllm-project/vllm/issues/55312) 是真实 spec-decoding layout conflict，但不一定所有 draft/target 路径都有一个严格意义上的“同一个 tensor 从 draft handed off 给 target”。

因此只有在代码里能明确找到：

\[
State_D
\rightarrow
TargetVerify
\]

的 shared representation 时，才把它算 strict RQ1。

否则它更适合作为：

\[
\boxed{\text{multi-consumer layout-domain problem}}
\]

偏 RQ2/RQ7。

如果能明确共享 cache/state，则候选：

\[
DraftNative,
TargetNative,
SharedCanonical,
DualRepresentation.
\]

---

# 十一、Case 10：HBM → LDS/Register → MMA

真实：

- [Triton PR #8450](https://github.com/triton-lang/triton/pull/8450)
- [Triton PR #7968](https://github.com/triton-lang/triton/pull/7968)
- [IREE PR #22356](https://github.com/iree-org/iree/pull/22356)

这是 micro-level mechanism case。

### Producer

\[
P=\text{global-memory load/staging}.
\]

### Consumer

\[
C=MMA/MFMA/Dot.
\]

候选：

```text
HBM Blocked
→ LDS
→ DotOperand
```

```text
HBM preshuffled
→ register DotOperand
```

```text
HBM
→ direct LDS
→ MMA
```

### 改什么？

Triton：

- `BlockedEncoding`
- `DotOperandEncoding`
- `SharedEncoding`
- bypass LDS

IREE：

- coalesced DMA → LDS
- direct-to-LDS

这个 case 用来解释上层 layout conflict 的物理根源，不应该单独承担 external validity。

---

# 十二、Case 11：Prefill → Transfer → Decode

真实：

- [SGLang PR #41953](https://github.com/sgl-project/sglang/pull/41953)
- [PR #41956](https://github.com/sgl-project/sglang/pull/41956)
- [PR #41958](https://github.com/sgl-project/sglang/pull/41958)
- [PR #41182](https://github.com/sgl-project/sglang/pull/41182)

### Producer 怎么定义？

不要把整个 prefill Transformer 当 producer，否则 compute 会淹没 representation effect。

主实验定义：

\[
P=\text{prefill-side KV/state export/gather}.
\]

consumer：

\[
C=\text{decode-side KV/state import + first attention use}.
\]

candidate：

\[
PrefillNative
\]

\[
WireCanonical
\]

\[
DecodeNative.
\]

路径：

```text
PrefillNative
→ pack
→ transport
→ unpack
→ DecodeNative
```

```text
Prefill directly exports WireCanonical
→ transport
→ unpack
```

```text
Prefill exports DecodeNative
→ direct transfer
→ Decode
```

如果是 host staging：

```text
GPU
→ pinned host
→ network
→ pinned host
→ GPU
```

则 memory placement 本身也进入 candidate。

### 二级 external-validity

再跑真实：

```text
full prefill request
→ PD
→ first decode token
```

但这只能作为 serving-level corroboration，不用于 strict H1 adjudication。

---

# 十三、Case 12：Host KV → GPU Hot Buffer → Sparse Decode

真实：

- [vLLM PR #46326](https://github.com/vllm-project/vllm/pull/46326)
- [vLLM PR #55398](https://github.com/vllm-project/vllm/pull/55398)

producer：

\[
P=\text{host→GPU restore/gather}.
\]

consumer：

\[
C=\text{sparse decode attention}.
\]

让 restore producer 直接生成：

\[
FullGPUPage
\]

\[
CompactNHDHotBuffer
\]

\[
CompactHNDHotBuffer
\]

\[
ConsumerNativeHotBuffer.
\]

然后测：

\[
T_{\rm restore}(L)
\]

和：

\[
T_{\rm restore\rightarrow sparseDecode}(L).
\]

很可能 producer-local 喜欢大块 contiguous copy，而 consumer edge 更喜欢 compact top-k consumer-native representation。

这正是非常漂亮的 memory-placement rank inversion。

---

# 十四、补充 Case：Host Expert → GPU Expert Pool → MoE GEMM

真实：

[vLLM PR #56177](https://github.com/vllm-project/vllm/pull/56177)

producer：

\[
P=H2D\ expert\ cache\ fill.
\]

consumer：

\[
C=Marlin/FusedMoE.
\]

candidate：

\[
HostCompactNVFP4
\]

\[
HostMarlinPhysical
\]

\[
GPUConsumerNative.
\]

比较：

```text
host compact
→ H2D
→ GPU convert/preshuffle
→ GEMM
```

与：

```text
host consumer-native
→ direct H2D
→ GEMM
```

非常适合研究：

\[
storage\ efficiency
\quad vs\quad
compute\ efficiency.
\]

---

# 十五、和原生框架应该怎样比

我建议明确分成四个 baseline 层级，而不是只报一个 native number。

| Baseline | 目的 |
|---|---|
| `Native-Auto` | 框架原样运行，真实 heuristic |
| `Native-Forced-L` | 用框架已有 config/env 强制某 layout |
| `Native+Repair` | 原生 producer + 显式转换 + 原生 consumer |
| `RQ1 Oracle` | 枚举所有合法 direct-emission/repair 组合 |

其中：

### Native-Auto

完全不 patch。

记录框架实际：

```text
backend
layout
stride
dtype
page size
packing
memory tier
```

例如 vLLM 如果有：

```python
supported_kv_cache_layouts()
```

就把实际选择结果记下来。

这回答：

> 当前框架 heuristic 到底选了什么？

---

### Native-Forced-L

如果框架本身支持：

```text
NHD/HND
LBHNC/LBNHC
page size
backend
```

这种强制接口，就直接用。

这是最公平的 A/B。

如果框架根本不支持 alternate layout，不要把它写成：

> native alternate。

应该写：

> patched/native-kernel experiment。

---

### Native+Repair

保持 producer 完全原样：

```text
Native P
→ explicit repair
→ Native C
```

它回答：

> 当前框架保持 producer-native representation，再补转换，代价是多少？

---

### RQ1 Oracle

这里可以 patch producer。

但必须尽量：

- 使用同一个 native consumer kernel；
- 同一个算法；
- 同一个 dtype；
- 同一个 tile/backend；
- 只改变 producer output mapping/layout。

这样才能判断：

\[
\text{native framework selection regret}.
\]

最终：

\[
Regret_{\rm framework}
=
\frac{T_{\rm NativeAuto}}
{T_{\rm Oracle}}-1.
\]

---

# 十六、必须做两套实验：Controlled 和 Native reproduction

一些真实 PR 会同时改变：

- layout；
- fusion；
- dtype；
- kernel count；
- communication overlap。

所以不要直接拿 PR before/after 当 RQ1。

应该分：

### Controlled RQ1

固定：

\[
algorithm,
dtype,
backend,
fusion,
shape
\]

只改变：

\[
representation/layout/repair.
\]

这决定 H1。

### Native reproduction

直接运行：

```text
framework main
vs
PR/new implementation
```

它证明：

> 真实框架中这个问题有系统价值。

但不能把 PR 全部收益归因于 layout。

这个区分非常重要。

---

# 十七、每一个 case 都应该记录统一的 6 个时间

我建议 schema 固定为：

\[
T_P
\]

producer-only。

\[
T_C
\]

consumer-only。

\[
T_R
\]

显式 repair。

\[
T_E
\]

完整 edge，直接计时。

\[
T_{\rm Native}
\]

框架原生路径。

\[
T_{\rm Oracle}
\]

枚举后的 edge oracle。

不要：

\[
T_E=T_P+T_R+T_C
\]

作为主结果。

因为：

- cache state 不同；
- launch overhead 不同；
- overlap 不同；
- fusion 不同；
- memory reuse 不同。

\(T_E\) 必须直接跑。

---

# 十八、统计判据建议继续沿用 v10 思路

每个真实 case 建议：

\[
\ge5
\]

独立 process repetitions。

每 process：

- warmup 8；
- measured iterations ≥30；
- 得到 p50。

然后跨 process 判断：

- producer winner stable；
- edge winner stable；
- correctness 全通过；
- bootstrap 95% CI；
- adaptive epsilon。

严格 positive：

\[
L_P^*
\neq
L_{P,E}^*
\]

并且：

\[
CI_{\rm regret,low}>0
\]

\[
regret>\epsilon.
\]

这比简单：

\[
speedup\ge3\%
\]

可靠很多。

---

# 十九、real-world shape 不要自己拍脑袋 sweep

每个 case 至少保留三类 shape source：

\[
\boxed{
PR\ benchmark
+
model\ config
+
serving\ trace
}
\]

例如：

Attention：

```text
Qwen / DeepSeek / MiniMax
decode q=1
MTP q=2~4
prefill q≫1
```

MoE：

```text
real experts
real top-k
real hidden/intermediate size
real decode token count
```

KDA：

```text
real packed QKV pitch
real H/D/state_dim
```

PD：

```text
real page size
real TP/DCP
real prefix blocks
```

并把：

```text
model
phase
backend
hardware
dtype
parallelism
```

写进每一行结果。

---

# 二十、最终一个 case 的结果应该长这样

以 `Attention BMM → OProj` 为例：

| Strategy | Producer layout | Repair | Consumer layout | P p50 | C p50 | Edge p50 |
|---|---|---|---|---:|---:|---:|
| Native | H,T,D | transpose-copy | T,H,D | … | … | … |
| Direct-A | H,T,D | none | H,T,D accepted | … | … | … |
| Direct-B | T,H,D | none | T,H,D | … | … | … |
| View | H,T,D | stride view | compatible | … | … | … |

然后单独报告：

\[
ProducerWinner=H,T,D
\]

\[
ConsumerWinner=T,H,D
\]

\[
EdgeWinner=T,H,D\ direct
\]

如果成立：

\[
ProducerWinner\neq EdgeProducerWinner
\]

才是：

\[
\boxed{\text{strict RQ1 inversion}}
\]

再比较：

\[
Native=T_{\rm transpose-copy}
\]

vs

\[
Oracle=T_{\rm direct-T,H,D}.
\]

这样你就同时回答了两个问题：

> producer/consumer preference 是否冲突？

以及更重要的：

> 当前框架原生实现是否因为局部 layout 决策而错过了更好的完整 edge？

这应该成为你后续所有新增 RQ1 case 的统一实验模板。

---

可以。把前面所有讨论和这一轮 2025–2026 GitHub 检索合起来，我建议不要把 RQ1 继续定义成“更多 shape 下的 NHD/HND producer–consumer benchmark”，而是把它扩展成一个真实框架中的 P–C edge catalog。

当前 v10/v13 的主要不足很明确：

\[
\text{v10: KV write}\rightarrow\text{paged head-local scan}
\]

\[
\text{v13 CUDA: KV write}\rightarrow\text{flat head-local scan}
\]

它们很好地证明了一个机制：

\[
L_{\text{producer-best}}\neq L_{\text{consumer/edge-best}}
\]

是可能发生的。

但还没有充分回答：

> 2025–2026 真实 LLM 框架中，到底哪些真实 operator / subgraph / stage、哪些 memory hierarchy boundary、哪些真实 model shapes 上会出现这种冲突？

我建议新的 RQ1 case 定义为：

\[
\boxed{
E=
(P,C,M_P,M_C,\mathcal L_P,\mathcal L_C,\mathcal R,S)
}
\]

其中：

- \(P\)：真实 producer；
- \(C\)：真实 consumer；
- \(M_P,M_C\)：producer/consumer 所在 memory level；
- \(\mathcal L_P,\mathcal L_C\)：各自合法 representation/layout；
- \(\mathcal R\)：view/copy/transpose/gather/pack/offload 等 repair；
- \(S\)：真实 model/workload shape。

这样 variety 才真正完整。

---

# 一、我建议的 RQ1 四个正交 variety 轴

| 轴 | 应覆盖内容 |
|---|---|
| P–C 粒度 | micro-op → operator → subgraph → serving stage |
| Memory hierarchy | Register/TMEM → Shared/LDS → HBM → Peer HBM → Host DRAM → NVMe/remote |
| LLM workload | GQA、MLA、Sparse Attention、MoE、GDN/KDA、MTP/SpecDecode、PD serving |
| Representation | axis-order、stride、packing、quantized representation、page geometry、placement |

因此不要把“layout”只理解成：

\[
NHD\leftrightarrow HND.
\]

现实中的 representation conflict 还包括：

\[
\text{packed}\leftrightarrow\text{split},
\]

\[
\text{token-major}\leftrightarrow\text{expert-major},
\]

\[
\text{accumulator-native}\leftrightarrow\text{store-native},
\]

\[
\text{GPU-native}\leftrightarrow\text{wire/storage-native}.
\]

---

# 二、Micro-level：`ld/st/MMA` 与 on-chip memory hierarchy

这些不应该成为 RQ1 的唯一证据，但非常适合解释“为什么”上层 edge 会发生 inversion。

| Framework | 真实 P → C | Memory level | 真实 issue/PR | 推荐测试 |
|---|---|---|---|---|
| Triton | Global load → Dot/MFMA | HBM→Reg/LDS→Reg | [PR #8450](https://github.com/triton-lang/triton/pull/8450) | Blocked→LDS→DotOperand vs direct HBM→DotOperand |
| Triton | Global operand → Dot | HBM→LDS/Reg | [PR #7968](https://github.com/triton-lang/triton/pull/7968) | ordinary vs preshuffled global representation |
| Triton | tensor permute/reshape → consumer | Reg→LDS→Reg | [PR #10360](https://github.com/triton-lang/triton/pull/10360) | SMEM relayout vs warp/register shuffle |
| Triton | WMMA/FMA result → Store | Reg→HBM | [PR #10563](https://github.com/triton-lang/triton/pull/10563), [PR #11704](https://github.com/triton-lang/triton/pull/11704) | accumulator-native vs store-coalesced |
| Triton | Dot1 → FFMA → Dot2 | Reg layout → Reg layout | [Issue #10342](https://github.com/triton-lang/triton/issues/10342) | preserve Dot1 output vs convert for Dot2 |
| Triton | Reduce → downstream op | Reg/SMEM | [Issue #8775](https://github.com/triton-lang/triton/issues/8775) | reduction scratch-native vs consumer-native |
| IREE | HBM producer → Attention LDS tile | HBM→LDS | [Issue #23782](https://github.com/iree-org/iree/issues/23782), [PR #22356](https://github.com/iree-org/iree/pull/22356) | HBM→Reg→LDS vs direct HBM→LDS |
| CUTLASS/CuTe | MMA accumulator → epilogue | TMEM→Reg | [PR #3313](https://github.com/NVIDIA/cutlass/pull/3313) | TMEM load atom widths / register layout |
| TileLang | GEMM fragment → shared consumer | Reg fragment→SMEM | [Issue #2954](https://github.com/tile-ai/tilelang/issues/2954) | fragment-native vs shared-native |
| TileLang | Shared operand → MMA | SMEM→MMA | [Issue #2642](https://github.com/tile-ai/tilelang/issues/2642) | annotated layout legality/conflict |
| TileLang | register-resident attention → block-scaled GEMM | Reg→MMA | [Issue #3256](https://github.com/tile-ai/tilelang/issues/3256) | resident fragments vs rematerialization |

这里最适合作为 RQ1 mechanism control 的是：

\[
HBM\rightarrow LDS\rightarrow MMA
\]

和

\[
HBM\rightarrow Register\rightarrow MMA.
\]

Triton #8450/#7968 明确就是现实 Paged Attention/MFMA 路径，不是人为构造。

对应 candidate：

\[
\mathcal L_{\text{micro}}
=
\{
Blocked,
DotOperand,
MMA/WMMA,
Shared,
SwizzledShared,
RegisterNative
\}.
\]

---

# 三、真实 Attention operator/subgraph

这一组应该成为 RQ1 的主证据之一。

## 1. `QKV / Norm / RoPE / KV preparation → Attention`

这是所有现代 Transformer/MLA/Sparse Attention 中非常真实的边。

| Framework | Real-world case | PR / Issue |
|---|---|---|
| vLLM | GQA/MLA/SparseMLA 的 RoPE + KV-cache fusion coverage | [Issue #46979](https://github.com/vllm-project/vllm/issues/46979) |
| vLLM | Qwen3.8-Flash-Next：QK norm+RoPE+gate+QSA prep+KV write → QSA | [PR #57097](https://github.com/vllm-project/vllm/pull/57097) |
| vLLM | MiniMax-M3 MSA + NVFP4 KV | [PR #59300](https://github.com/vllm-project/vllm/pull/59300) |
| vLLM | MLA FP8/NVFP4 cache variants | [PR #59246](https://github.com/vllm-project/vllm/pull/59246), [PR #59342](https://github.com/vllm-project/vllm/pull/59342) |
| SGLang | MiniMax-M3 sparse QK norm + RoPE + cache write | [PR #35357](https://github.com/sgl-project/sglang/pull/35357) |
| SGLang | FA3 query quantization + RoPE | [PR #35142](https://github.com/sgl-project/sglang/pull/35142) |
| SGLang | FP8 MLA，避免 full-pool BF16 cast | [PR #35322](https://github.com/sgl-project/sglang/pull/35322) |
| FlashInfer | MLA projection → FP8 packed K/V → TRTLLM attention | [PR #5667](https://github.com/flashinfer-ai/flashinfer/pull/5667) |
| TensorRT-LLM | ragged sparse-attention row contract | [PR #18485](https://github.com/NVIDIA/TensorRT-LLM/pull/18485) |

这里应该测试：

```text
QKV projection
      ↓
QK Norm / RoPE
      ↓
KV preparation / quant / pack
      ↓
Attention
```

而不是只测试 `kv_write()`。

推荐 representation：

\[
\{
packed\ QKV,
split\ strided\ QKV,
dense\ Q/K/V,
NHD,
HND,
paged\_NHD,
paged\_HND,
consumer\text{-}native
\}.
\]

对于 MLA 再加入：

\[
\{
latent-native,
expanded\ KV,
packed\ FP8,
NVFP4
\}.
\]

---

# 四、Attention output → O-Proj：非常值得做

这是我认为最干净的 operator→operator RQ1 case 之一。

SGLang：

[SGLang PR #34498](https://github.com/sgl-project/sglang/pull/34498)

实际路径：

```text
Attention value BMM
        ↓
[heads,tokens,vdim]
        ↓
transpose/materialize
        ↓
[tokens,heads,vdim]
        ↓
O-Proj GEMM
```

PR 改成 BMM 直接写：

\[
[T,H,D_v]
\]

让 downstream flatten 成为 zero-copy view。

所以真实问题就是：

\[
L_{\text{BMM-best}}
\stackrel{?}{=}
L_{\text{BMM→OProj-best}}.
\]

候选：

\[
\{
[H,T,D],
[T,H,D],
[T,H D],
stride\text{-}view
\}.
\]

这个 case 的好处是它完全摆脱了 KV-cache 特例。

---

# 五、MoE：必须成为单独一个 P–C family

2025–2026 主流 LLM 中 MoE 已经足够常见，所以 RQ1 如果只研究 Attention，会显得外部有效性不足。

## `Router / Dispatch → Grouped Expert GEMM`

| Framework | 真实问题 | PR |
|---|---|---|
| SGLang | DeepEP dispatch 直接产生 expanded expert-major representation | [PR #37261](https://github.com/sgl-project/sglang/pull/37261) |
| vLLM | DeepEP V2 activation format | [PR #54109](https://github.com/vllm-project/vllm/pull/54109) |
| SGLang | MoonEP 直接采用 DeepGEMM psum layout | [PR #39039](https://github.com/sgl-project/sglang/pull/39039) |
| SGLang | shared/routed experts representation fusion | [PR #39313](https://github.com/sgl-project/sglang/pull/39313) |
| SGLang | MXFP4×BF16 grouped GEMM | [PR #38884](https://github.com/sgl-project/sglang/pull/38884) |

候选：

\[
\boxed{
\{
Standard,
PaddedStandard,
TokenMajor,
ExpandedExpertMajor,
BatchedExperts,
DeepGEMM\text{-}psum
\}}
\]

repair：

\[
scatter,\ expand,\ pad,\ permute.
\]

真正比较：

\[
T_{\text{dispatch}}(L_D)
\]

和：

\[
T_{\text{dispatch→GroupedGEMM}}(L_D,R,L_G).
\]

---

# 六、MoE GEMM2 → Combine / Communication

这是 operator 和 memory-placement 同时变化的优秀 case。

FlashInfer：

[PR #5208](https://github.com/flashinfer-ai/flashinfer/pull/5208)

原来：

```text
GEMM2
 ↓
local GPU HBM
 ↓
ReduceScatter/Combine
 ↓
peer GPU
```

候选新路径：

```text
GEMM2 epilogue
 ↓
direct peer store
 ↓
peer-GPU route-slot buffer
 ↓
local reduction
```

也就是：

\[
Reg/SMEM\rightarrow LocalHBM\rightarrow Collective
\]

vs

\[
Reg/SMEM\rightarrow PeerHBM.
\]

候选：

\[
\{
LocalContiguous,
TokenMajor,
RouteSlotMajor,
PeerScatter
\}.
\]

它同时增加：

- operator variety；
- multi-GPU variety；
- memory-placement variety。

---

# 七、GDN/KDA/Linear Attention：现代 LLM 中非常值得补

这一部分比旧 RQ1 新得多。

## `Packed QKV projection → KDA/GDN`

FlashInfer：

[PR #5846](https://github.com/flashinfer-ai/flashinfer/pull/5846)

serving 实际 producer 给：

\[
[T,3HD]
\]

packed projection row。

`split(dim=-1)` 后 Q/K/V 是 strided views。

旧 consumer 会把它们 densify：

\[
3\times[T,H,D]
\]

新 KDA prefill 可以直接读取真实 token pitch。

所以：

\[
\mathcal L=
\{
PackedRow,
StridedSplit,
DenseSeparateQKV
\}.
\]

这是一个非常真实的：

\[
Projection\rightarrow KDA/GDN
\]

P–C。

## `KDA/GDN Prefill → recurrent state → Decode`

SGLang：

[PR #34299](https://github.com/sgl-project/sglang/pull/34299)

真实 representation：

\[
\{
PrefillNativeCheckpoint,
DecodePackedState,
CanonicalState
\}.
\]

测试：

\[
Prefill_{native}\rightarrow Decode
\]

vs

\[
Prefill_{native}\rightarrow PackOnce\rightarrow Decode
\]

vs

\[
Prefill\ directly\ emits\ DecodePacked.
\]

这应该成为 persistent-state 版本的 RQ1。

---

# 八、Sparse Attention：QSA / MSA / Sparse MLA

这是 2026 特别值得补的一类。

## `NHD persistent pool → sparse-attention consumer`

SGLang：

[PR #34525](https://github.com/sgl-project/sglang/pull/34525)

它直接暴露：

\[
NHD\ pool
\rightarrow
gather/transpose
\rightarrow
paged/compact\ HND
\rightarrow
MSA.
\]

对于 \(H_{kv}>1\)，permute 不能简单当作免费 contiguous view。

候选：

\[
\{
FullPoolNHD,
FullPoolHND,
CompactGatheredNHD,
CompactGatheredHND,
PagedConsumerNative
\}.
\]

相关现实支持还有：

- vLLM MiniMax-M3 MSA NVFP4：[PR #59300](https://github.com/vllm-project/vllm/pull/59300)
- TensorRT-LLM sparse row contracts：[PR #18485](https://github.com/NVIDIA/TensorRT-LLM/pull/18485)
- SGLang GLM-5.3 sparse attention：[PR #41615](https://github.com/sgl-project/sglang/pull/41615)

这一类可以让 RQ1 不只覆盖 dense GQA/MLA。

---

# 九、Speculative/MTP：Draft → Verify

这也是现实的 stage，不是人为构造。

vLLM 已有：

[Issue #55312](https://github.com/vllm-project/vllm/issues/55312)

> draft 和 target attention backend 可以拥有不相交的 KV-cache layout support set。

所以实际可能出现：

\[
\mathcal L_D\cap\mathcal L_T=\varnothing.
\]

FlashInfer 又有：

[Issue #4731](https://github.com/flashinfer-ai/flashinfer/issues/4731)

multi-token verify 与 \(q\_len=1\) 的行为明显不同。

因此建议：

\[
\{
DraftNative,
TargetNative,
SharedCanonical,
ConvertOnce,
DualRepresentation
\}.
\]

这是现实的：

\[
Draft/MTP\rightarrow TargetVerify.
\]

---

# 十、Persistent KV：不要再只用 `{NHD,HND}`

vLLM/SGLang 已经说明现实 cache layout 空间远大于两种 axis order。

关键真实证据包括：

- vLLM draft/target layout conflict：[Issue #55312](https://github.com/vllm-project/vllm/issues/55312)
- SGLang unified token-major memory：[PR #38592](https://github.com/sgl-project/sglang/pull/38592)
- SGLang stride-based KV addressing：[PR #40326](https://github.com/sgl-project/sglang/pull/40326)
- SGLang paged KV view construction：[PR #40327](https://github.com/sgl-project/sglang/pull/40327)
- TensorRT-LLM DeepSeek-V4 FP8 layout mismatch：[Issue #14327](https://github.com/NVIDIA/TensorRT-LLM/issues/14327)

因此 candidate registry 至少应该支持：

\[
\begin{aligned}
\mathcal L_{KV}=\{&
NHD,HND,\\
&paged\_NHD,paged\_HND,\\
&LBNHC,LBHNC,BLNHC,BLHNC,\\
&interleaved,noninterleaved,\\
&token\text{-}major\ unified,\\
&backend\text{-}shuffled
\}.
\end{aligned}
\]

但必须先通过 backend legality contract 过滤。

---

# 十一、Memory hierarchy：HBM → Peer GPU

除了 FlashInfer MoE #5208，还有 PD / remote landing。

vLLM：

[PR #55398](https://github.com/vllm-project/vllm/pull/55398)

HiSparse P/D transfer 可以根据 GPU capacity：

\[
remote\ transfer
\rightarrow
final\ GPU\ page
\]

直接 landing；

否则：

\[
remote
\rightarrow
host\ page.
\]

这说明 placement 本身已经进入 edge decision：

\[
\boxed{
L+\text{placement}
}
\]

应该一起搜索。

---

# 十二、GPU HBM ↔ Host pinned DRAM

这是长上下文 LLM serving 中必须加入的一层。

## HiSparse / sparse MLA

vLLM：

[PR #46326](https://github.com/vllm-project/vllm/pull/46326)

真实结构：

```text
full sparse-MLA KV
        ↓
pinned host DRAM
        ↓
top-k gather
        ↓
GPU hot buffer
        ↓
Sparse MLA Decode
```

候选：

\[
\{
GPUFullKV,
HostFullKV+GPUHotBuffer,
HostCanonical+GPUConsumerNative
\}.
\]

这是很好的 real-world memory-placement RQ1。

## Host staging

SGLang：

[PR #41182](https://github.com/sgl-project/sglang/pull/41182)

真实 PD 路径：

```text
Prefill GPU
    ↓
Pinned host slot
    ↓
NIXL host-to-host
    ↓
Decode pinned host ring
    ↓
Decode GPU
```

对应：

\[
GPU\rightarrow Host\rightarrow Network\rightarrow Host\rightarrow GPU.
\]

strategy 可以比较：

\[
DirectGPU\rightarrow DirectGPU
\]

vs

\[
GPU\rightarrow HostCanonical
\rightarrow Host
\rightarrow GPUNative.
\]

---

# 十三、Host ↔ GPU copy granularity 本身也是 P–C edge

vLLM：

[PR #54483](https://github.com/vllm-project/vllm/pull/54483)

当 NIXL 走 CPU staging buffer 时，旧方案按 cache group 分别 H2D/D2H。

新方案将 block IDs 合并，减少：

\[
groups\times layers
\]

级 copy operations。

所以 memory representation 不仅有 layout，还有：

\[
\boxed{\text{transfer grouping/granularity}}
\]

这一维。

---

# 十四、Host DRAM ↔ NVMe / filesystem

2026 serving 已经存在真正的 multi-tier KV。

SGLang：

[PR #39880](https://github.com/sgl-project/sglang/pull/39880)

真实：

```text
Host KV pool
   ↓
staging
   ↓
NVMe
```

vs：

```text
page-first Host pool
   ↓
readv/writev direct
   ↓
NVMe
```

因此候选：

\[
\{
LogicalHostPool,
PageFirstHostPool,
ContiguousStaging,
DirectFilePage
\}.
\]

vLLM 也有 filesystem KV tier：

[PR #54327](https://github.com/vllm-project/vllm/pull/54327)

以及 multi-tier offloading RFC：

[Issue #38260](https://github.com/vllm-project/vllm/issues/38260)

所以：

\[
HBM\leftrightarrow Host\leftrightarrow NVMe
\]

已经是现实 serving memory hierarchy。

---

# 十五、Disk/mmap → Host → GPU：大 embedding/table 类

vLLM：

[PR #54129](https://github.com/vllm-project/vllm/pull/54129)

Qwen3.8-Flash-Next 的 PLE table 可直接 mmap checkpoint file：

```text
safetensors / disk
       ↓
mmap / OS page cache
       ↓
selected-row gather
       ↓
GPU stable buffer
       ↓
PLE consumer
```

这里的 representation candidates 是：

\[
\{
GPUResident,
HostResident,
PinnedHost,
DiskMmap
\}.
\]

这个 case 的意义是证明 RQ1 不是只适用于 KV cache。

---

# 十六、Host expert weights → GPU expert cache → MoE

vLLM：

[PR #56177](https://github.com/vllm-project/vllm/pull/56177)

真实路径：

```text
Pinned Host
 full NVFP4 experts
       ↓
device-managed cache miss copy
       ↓
GPU expert bank
       ↓
Marlin MoE GEMM
```

这里非常适合研究：

\[
L_{\text{host-storage-best}}
\stackrel{?}{=}
L_{\text{Marlin-consumer-best}}.
\]

例如：

\[
HostCompactNVFP4
\rightarrow
GPUConvert
\rightarrow
Marlin
\]

vs

\[
HostMarlinNative
\rightarrow
DirectCopy
\rightarrow
Marlin.
\]

这同时增加：

- MoE；
- weight representation；
- host/GPU placement；
- cache residency。

---

# 十七、PD disaggregation：Prefill → Transfer → Decode

这是明确的真实 serving stage。

SGLang：

- transfer layout metadata：[PR #41953](https://github.com/sgl-project/sglang/pull/41953)
- unequal TP：[PR #41956](https://github.com/sgl-project/sglang/pull/41956)
- matching DCP direct transfer：[PR #41958](https://github.com/sgl-project/sglang/pull/41958)
- hybrid SWA KV transfer：[PR #40909](https://github.com/sgl-project/sglang/pull/40909)

TensorRT-LLM：

- Qwen3.8-Flash-Next disaggregated serving：[PR #18921](https://github.com/NVIDIA/TensorRT-LLM/pull/18921)

vLLM：

- P/D parallel-layout transfer contract：[Issue #56123](https://github.com/vllm-project/vllm/issues/56123)

建议三个 representation：

\[
\{
PrefillNative,
CanonicalWire,
DecodeNative
\}.
\]

然后比较：

\[
P_{native}
\rightarrow
pack
\rightarrow
wire
\rightarrow
unpack
\rightarrow
D_{native}
\]

与：

\[
P_{decode-native}
\rightarrow
direct transfer
\rightarrow
D.
\]

---

# 十八、PyTorch/Inductor：zero-copy stride contract

这个也值得保留，因为它代表 framework graph level，而不是专用 attention runtime。

PyTorch：

[PR #197100](https://github.com/pytorch/pytorch/pull/197100)

真实问题：

\[
noncontiguous\ view
\rightarrow
matmul
\]

本来 stride 是合法 zero-copy，但 decomposition 会产生额外 materialization。

同时：

[Issue #195320](https://github.com/pytorch/pytorch/issues/195320)

涉及：

\[
permuted\ K\rightarrow SDPA.
\]

推荐：

\[
\{
Contiguous,
TransposeView,
PermutedStride,
PaddedStride,
Materialized
\}.
\]

这是很好的 framework-level stride-contract control。

---

# 十九、IREE：compiler graph-level layout propagation

IREE 特别适合补“graph representation”这一类。

相关：

- incoming layout mismatch preferred layout：[Issue #22489](https://github.com/iree-org/iree/issues/22489)
- dynamic layout decisions：[Issue #22370](https://github.com/iree-org/iree/issues/22370)
- attention direct-to-LDS：[Issue #23782](https://github.com/iree-org/iree/issues/23782)
- target-aware layout materialization：[PR #24969](https://github.com/iree-org/iree/pull/24969)

IREE 可以测试：

\[
Matmul\rightarrow Elementwise\rightarrow Matmul
\]

和：

\[
Attention\rightarrow Projection.
\]

其 candidate 应是 encoding/tiling，而非 NHD/HND。

---

# 二十、哪些不应该进入“核心 real-world RQ1”

前面筛出的有些 case 有价值，但不建议和真实 LLM operator 放在同一证据等级：

- generic row/column reducer；
- 单独 synthetic scan；
- generic `tt.load→tt.dot`，如果没有关联实际 attention/GEMM workload；
- CUTLASS 教程式 TMEM microbenchmark；
- TileLang fuzzer-only cases；
- generic Conv；
- 不对应真实模型 graph 的 arbitrary operator chain。

这些可以作为：

\[
\boxed{\text{mechanism controls}}
\]

而不是 external-validity evidence。

---

# 二十一、Real-world shape 应该怎么取

这里我建议不要再自己定义一个均匀 Cartesian sweep。

不同 P–C family 使用真实 workload contract：

| Family | 应记录的 real-world shape |
|---|---|
| GQA | \(B,Q,KV,H_q,H_{kv},D,page\) |
| MLA | \(B,Q,KV,H_q,D_{nope},D_{rope},D_v\) |
| Sparse Attention | \(B,Q,KV,H,topk,page\) |
| MoE | \(M,K,N,E,topk,TP,EP\) |
| GDN/KDA | \(T,H,D,state\_dim,token\_pitch\) |
| GEMM→OProj | \(T,H,D_v,H_{model}\) |
| MTP | batch、draft length、verify q_len、KV length |
| P/D | layers、blocks、page size、TP/DCP、bytes/page |
| Offload | working-set bytes、hot-set fraction、transfer block |
| Micro MMA | tile M/N/K、warps、vector width、SMEM swizzle |

优先从对应 PR 的 benchmark、模型 config 和 serving recipe 抽 shape。

几个已经有非常具体 real-world 参数的例子：

FlashInfer #5667 明确处理 MLA：

\[
kv\_nope:[T,H,256]
\]

\[
k_{pe}:[T,64]
\]

输出：

\[
K:[T,H,192],
\quad
V:[T,H,128].
\]

而且 Kimi-K3 特化路径使用 \(H=12\)。

[PR #5667](https://github.com/flashinfer-ai/flashinfer/pull/5667)

vLLM #57097 的 Qwen3.8-Flash-Next benchmark 明确是实际 TP8 attention path，而不是 synthetic head count：

[PR #57097](https://github.com/vllm-project/vllm/pull/57097)

vLLM HiSparse #46326 实际验证里包含：

- 65,536 context；
- sparse top-k 2048；
- GLM-5.2 production stack。

[PR #46326](https://github.com/vllm-project/vllm/pull/46326)

所以 shape source 本身也应该注册：

```text
shape_source =
    model_config
    | PR_benchmark
    | serving_recipe
```

避免无法解释为什么选这个 shape。

---

# 二十二、最终建议的 candidate registry

建议不要有一个全局 layout enum，而是 family-specific。

| Family | Candidate set |
|---|---|
| GQA | NHD, HND, paged_NHD, paged_HND |
| vLLM KV | LBNHC, LBHNC, BLNHC, BLHNC |
| MLA | latent, expanded-KV, packed-KV, FP8/NVFP4 |
| Sparse | full-pool, compact-gathered, paged consumer-native |
| QKV | packed-row, split-strided, dense-separated |
| Attention→OProj | H×T×D, T×H×D, flattened |
| MoE | Standard, PaddedStandard, ExpertMajor, BatchedExperts, psum |
| KDA/GDN | packed QKV, strided Q/K/V, dense Q/K/V, checkpoint/packed-state |
| Triton | Blocked, DotOperand, MMA/WMMA, Shared, SwizzledShared |
| CUTLASS | TMEM, RMEM, TMA-native, RowMajor, ColumnMajor |
| Host tier | logical, page-first, canonical-transfer, compressed |
| Communication | producer-native, wire-native, consumer-native |
| Filesystem | staging-contiguous, page-file-native, direct-I/O |
| Offloaded MoE | host compact, GPU consumer-native, cached physical rows |

---

# 二十三、我认为最合适的“核心 RQ1 benchmark suite”

如果你真的要实现实验，而不是无限扩张，我建议最终保留约 12 个核心 family。

| ID | P → C | 粒度 | Memory hierarchy |
|---|---|---|---|
| E1 | QKV/Norm/RoPE/KVPrep → GQA Attention | subgraph | HBM |
| E2 | MLA projection/pack → MLA Attention | operator/subgraph | HBM |
| E3 | Sparse KV Gather → MSA/QSA/SparseMLA | operator | HBM |
| E4 | Attention BMM → O-Proj | operator | Reg/HBM |
| E5 | MoE Dispatch → Grouped GEMM | operator | HBM |
| E6 | Expert GEMM2 → Combine | operator/distributed | Reg/SMEM→peer HBM |
| E7 | Packed QKV → KDA/GDN | operator | HBM |
| E8 | KDA/GDN Prefill → Decode state | stage | persistent HBM |
| E9 | Draft/MTP → Target Verify | stage | persistent HBM |
| E10 | Attention tile load → MMA | micro | HBM→LDS/Reg |
| E11 | Prefill KV → PD transfer → Decode | stage | GPU→network→GPU |
| E12 | Host sparse KV → GPU hot buffer → Decode | stage | Host→GPU |

然后 supplementary：

\[
TMEM\rightarrow Reg\rightarrow HBM,
\]

\[
GPU\leftrightarrow Host\leftrightarrow NVMe,
\]

\[
HostExpert\rightarrow GPUExpertPool\rightarrow MoE.
\]

这已经能够覆盖：

\[
\boxed{
Attention,\ MLA,\ SparseAttention,\ MoE,\ LinearAttention,\ SpecDecode
}
\]

以及：

\[
\boxed{
Register,\ TMEM,\ LDS,\ HBM,\ PeerGPU,\ HostDRAM,\ NVMe
}
\]

两个维度。

---

# 二十四、实验统计上仍然要保持 v10 的“完整 edge”原则

这是最重要的一点。

不要因为找到了这些真实 PR，就只分别测：

\[
T_P(L)
\]

和：

\[
T_C(L).
\]

应该同时直接运行：

\[
\boxed{
T_E(L_P,R,L_C)
=
time[
P(L_P)\rightarrow R\rightarrow C(L_C)
]
}
\]

然后报告三个不同的 winner：

\[
L_P^*=\arg\min T_P,
\]

\[
L_C^*=\arg\min T_C,
\]

以及：

\[
(L_P,R,L_C)^*
=
\arg\min T_E.
\]

这样才能严格区分：

\[
L_P^*\neq L_C^*
\]

只是“preference conflict”，

和真正的：

\[
\boxed{
L_P^*\neq L_{P,E}^*
}
\]

即 RQ1 要找的：

\[
\boxed{\text{cross-edge rank inversion}}.
\]

因此，我会把新的 RQ1 定义成：

\[
\boxed{
\textit{Across real 2025–2026 LLM operator, subgraph, stage,
and memory-hierarchy boundaries, when does the producer-local
optimal representation differ from the end-to-end edge-optimal representation?}
}
\]

这比当前“KV writer 和 head-local scan 是否喜欢不同的 NHD/HND”要强得多，同时这些新增 case 都能够追溯到真实框架的 issue/PR，而不是为了实验人为创造的数据流。

---

是的。前面主要改变的是 producer/consumer 的“算子粒度”，但 RQ1 还应该增加另一条独立维度：

\[
\boxed{\text{producer/consumer 所处的 memory hierarchy}}
\]

这很重要，因为同一个逻辑 tensor 的最佳 representation 往往取决于它究竟要被放在：

\[
\text{register/TMEM}
\rightarrow
\text{shared/LDS}
\rightarrow
\text{GPU HBM}
\rightarrow
\text{peer GPU HBM}
\rightarrow
\text{host pinned DRAM}
\rightarrow
\text{NVMe/remote store}
\]

中的哪一级。

我重新筛了 2025–2026 的实际 PR/issue 后，发现这条轴上有一批非常适合 RQ1 的真实 case，而且不少直接来自 Paged Attention、MoE、PD disaggregation、HiSparse、KDA/QSA 等现代 LLM 路径。

先给结论：我建议 RQ1 至少覆盖下面 7 种 memory-level boundary。

| 类别 | Producer 所在层级 | Consumer 所在层级 | 现实 LLM 场景 |
|---|---|---|---|
| M1 | GPU HBM | Registers | Attention/GEMM operand load |
| M2 | GPU HBM | Shared/LDS → Registers | Attention/GEMM staging |
| M3 | TMEM/Register | Register/Shared → HBM | GEMM/Attention epilogue |
| M4 | GPU HBM / SMEM | Peer-GPU HBM | MoE EP combine/dispatch |
| M5 | GPU HBM | Host pinned DRAM | KV offload / HiSparse |
| M6 | Host pinned DRAM | GPU HBM | KV restore / PD decode |
| M7 | GPU/Host DRAM | NVMe/filesystem/remote tier | long-context KV/embedding tiering |

这里的关键是：同一个 RQ1 不再只是比较

\[
NHD\quad\text{vs}\quad HND
\]

而是比较：

\[
\boxed{
\text{representation}
+
\text{memory placement}
+
\text{transfer strategy}
}
\]

---

## 1. M1：GPU HBM → Register，绕过 Shared/LDS

这是最干净的“不同 memory level” RQ1。

Triton PR #8450 的标题就是：

> Implement implicit layout conversion for DotOp to enable direct GMEM to reg loads

它来自 Paged Attention 优化 RFC。

原来的路径是：

```text
GPU HBM
   │
   │ coalesced load
   ▼
register: Blocked layout
   │
   ▼
Shared / LDS
   │ convert/rearrange
   ▼
register: DotOperand layout
   │
   ▼
Dot / MFMA
```

也就是：

\[
HBM
\rightarrow
Register_{blocked}
\rightarrow
LDS
\rightarrow
Register_{dot}
\rightarrow
Dot
\]

原因是：

- global-memory load 喜欢 coalesced `Blocked` layout；
- MFMA/Dot consumer 喜欢 `DotOperandEncoding`；
- 两边 preference 不一致；
- 所以传统方案借助 LDS 做 layout conversion。

PR #8450 要尝试：

```text
GPU HBM
   │
   │ 按 Dot 所需要的映射直接读取
   ▼
Register
   │
   ▼
Dot
```

即：

\[
\boxed{
HBM\rightarrow Register\rightarrow Dot
}
\]

消掉：

\[
Register\rightarrow LDS\rightarrow Register
\]

这一段。

这正是非常标准的 RQ1：

\[
L_{\text{HBM-load-best}}
\stackrel{?}{=}
L_{\text{Dot-consumer-best}}
\]

并且：

\[
\text{best memory path}
\stackrel{?}{=}
HBM\rightarrow LDS\rightarrow Reg
\]

还是：

\[
HBM\rightarrow Reg.
\]

相关真实证据还有 Triton PR #7968，已经 merged：

> Add bypassLDS feature to StreamPipeline

它明确说 Dot/MFMA operand 默认不是 coalesced layout，因此通常：

\[
GMEM\rightarrow coalesced\ blocked
\rightarrow LDS
\rightarrow MFMA
\]

但如果 operand 在 global memory 中已经被 preshuffle 成合适 representation，就可以：

\[
GMEM\rightarrow register\rightarrow MFMA
\]

直接绕过 LDS。

这非常适合 modern Attention：

\[
K/V\ HBM
\rightarrow
MFMA/Attention
\]

建议的 candidate strategy：

\[
\boxed{
\begin{aligned}
S_1 &: HBM_{plain}
\rightarrow Reg_{blocked}
\rightarrow LDS
\rightarrow Reg_{dot}\\
S_2 &: HBM_{preshuffled}
\rightarrow Reg_{dot}\\
S_3 &: HBM_{plain}
\rightarrow LDS_{consumer-native}
\rightarrow Reg_{dot}
\end{aligned}}
\]

这比仅仅测试 NHD/HND 更丰富，因为这里改变的是：

\[
\boxed{\text{physical layout + memory hierarchy path}}
\]

---

## 2. M2：GPU HBM → Shared/LDS → Registers

这条也有非常明确的真实 Attention 证据。

IREE issue #23782：

> Use direct-to-LDS in attention

以及 merged PR #22356：

> Add AMDGPULowerCoalescedDMAToGatherLDS pass for direct global to LDS loads

这个 PR 明确实现：

\[
\boxed{
GlobalMemory
\rightarrow
LDS
}
\]

直接 DMA/gather，而不是：

```text
HBM
 ↓
register
 ↓
LDS
```

所以可以比较：

### Path A

\[
HBM\rightarrow Register\rightarrow LDS\rightarrow Attention
\]

### Path B

\[
HBM\rightarrow LDS\rightarrow Attention
\]

### Path C

如果硬件允许：

\[
HBM\rightarrow Register\rightarrow MMA.
\]

这三条实际上对应不同的 producer-consumer contract。

例如 Attention 中 K/V tile：

```text
Paged KV in HBM
       │
       ▼
   tile loader
       │
       ▼
      LDS
       │
       ▼
MFMA / attention compute
```

问题不是：

> “NHD 好还是 HND 好？”

而是：

> “producer 的 HBM representation 应该为了 coalesced DMA 优化，还是为了 LDS swizzle / MMA consumer 优化？”

对应 candidate：

\[
\mathcal L_{HBM}
=
\{
plain,\ blocked,\ vectorized,\ preshuffled
\}
\]

\[
\mathcal L_{LDS}
=
\{
row-major,\ operand-major,\ swizzled
\}
\]

然后搜索：

\[
(L_{HBM},L_{LDS})^*.
\]

这其实是非常典型的 cross-memory-level RQ1。

---

## 3. M3：TMEM / accumulator → Register → HBM

Blackwell 上这一类特别值得测。

CUTLASS PR #3313 专门讨论：

\[
\boxed{
TMEM\rightarrow Register
}
\]

的 GEMM epilogue。

Blackwell MMA 的 accumulator 可以驻留在 TMEM。

于是完整路径类似：

```text
Tensor Cores
    │
    ▼
   TMEM
    │
    │ tcgen05.ld
    ▼
Registers
    │
    ▼
Epilogue transform
    │
    ▼
Global Memory / HBM
```

PR #3313 发现：

\[
TMEM\rightarrow Register
\]

的 copy atom 宽度本身就非常重要。

例如：

```text
32dp32b1x
```

vs

```text
32dp32b32x
```

在一个实验中，仅改变 TMEM→register load width，就把完整 kernel：

\[
23.8\mu s\rightarrow16.0\mu s
\]

约提升 \(1.49\times\)。

所以一个真实 LLM GEMM RQ1 可以是：

\[
\boxed{
GEMM\ accumulator_{TMEM}
\rightarrow
OProj/MoE/next-op
}
\]

候选不仅是 RowMajor/ColumnMajor，还包括：

\[
\{
TMEM\ native,
Register\ fragment,
consumer\text{-}native\ register,
HBM\ row-major,
HBM\ column-major
\}.
\]

路径也有：

\[
TMEM\rightarrow Reg\rightarrow HBM
\]

和可能的：

\[
TMEM\rightarrow Reg_{consumer-native}
\rightarrow next\ fused\ op.
\]

这类 case 很适合：

- Attention BMM → O-Proj；
- MoE GEMM1 → activation → GEMM2；
- GEMM → fused epilogue。

---

# 4. M4：GPU local memory → Peer GPU HBM

这是 distributed MoE 中非常现实的一条边。

FlashInfer PR #5208：

> Fuse the cross-GPU MoE combine into the Blackwell GEMM2 finalize epilogue

原方案：

```text
GPU i
Expert GEMM2
     │
     ▼
local GPU HBM
     │
     ▼
ReduceScatter / Combine
     │
     ▼
GPU j HBM
```

也就是：

\[
Reg/SMEM
\rightarrow
LocalHBM
\rightarrow
Collective
\rightarrow
PeerHBM.
\]

新方案让 GEMM2 epilogue：

```text
GEMM2 SMEM/register
        │
        │ peer store
        ▼
peer GPU combine_buffer
        │
        ▼
local weighted reduction
```

即：

\[
\boxed{
Reg/SMEM
\rightarrow
PeerGPU\ HBM
}
\]

绕过：

\[
LocalHBM\rightarrow Collective.
\]

而且 PR 明确说，底层仍使用：

```text
cp.async.bulk.global.shared::cta.bulk_group
```

只是 destination address 直接指向 peer GPU symmetric buffer。

因此这是一个非常漂亮的 memory-level RQ1：

\[
\boxed{
L_{\text{GEMM2-local-best}}
\stackrel{?}{=}
L_{\text{peer-combine-best}}
}
\]

候选：

\[
\{
LocalContiguous,
TokenMajor,
RouteSlotMajor,
PeerScatterLayout
\}.
\]

Strategy：

\[
S_1:
Reg\rightarrow LocalHBM
\rightarrow ReduceScatter
\rightarrow PeerHBM
\]

vs

\[
S_2:
Reg/SMEM
\rightarrow PeerHBM_{route-slot}
\rightarrow local\ reduce.
\]

这一条对 2025–2026 MoE 非常现实。

---

# 5. M5：GPU HBM → Host pinned DRAM

这是当前长上下文 serving 中最值得加入 RQ1 的 memory-level family。

最直接的现实例子是 HiSparse。

vLLM PR #46326 的设计是：

> full sparse-MLA KV lives in pinned host memory, decode is served from per-request GPU hot buffers holding the indexer top-k working set.

也就是说：

```text
            full KV
              │
              ▼
       pinned host DRAM
              │
       selected top-k
              │
              ▼
        GPU hot buffer
              │
              ▼
        Sparse MLA Decode
```

而不是传统：

```text
full KV
  │
  ▼
GPU HBM
  │
  ▼
Attention
```

因此这已经不是简单 layout 问题。

它是在选择：

\[
\boxed{
\text{persistent representation 的 memory placement}
}
\]

候选可以有：

\[
L_1 =
GPU\ HBM\ full\ KV
\]

\[
L_2 =
HostPinned\ full\ KV
+
GPU\ compact\ hot\ buffer
\]

\[
L_3 =
HostPinned\ canonical\ KV
+
GPU\ consumer\text{-}native\ gathered\ layout.
\]

然后测：

\[
T_{\text{producer/save}}
+
T_{\text{host→GPU gather}}
+
T_{\text{sparse attention}}.
\]

这种 case 对：

- GLM-5.x / sparse MLA；
- MiniMax-M3 MSA；
- DeepSeek-V4 sparse attention

都非常有现实意义。

---

SGLang PR #42009 也是很好的 D2H/H2D case。

它专门研究 HiCache：

\[
GPU\ HBM\leftrightarrow Host\ DRAM.
\]

PR 中明确有：

### D2H backup

\[
GPU\rightarrow Host
\]

而且优化：

\[
\text{one huge copy issue}
\]

vs

\[
\text{chunked D2H issue}
\]

让下一 batch descriptor issuance 与当前 copy overlap。

### H2D load-back

\[
Host\rightarrow GPU
\]

并且 rank-sharded host tier 后：

```text
owner rank:
Host → local GPU
       ↓
broadcast
       ↓
other GPU slots
```

所以这里可以研究：

\[
\boxed{
L_{\text{GPU persistent}}
\neq
L_{\text{Host tier optimal}}
\neq
L_{\text{reload consumer optimal}}
}
\]

这已经是一个非常标准的跨 hierarchy representation problem。

---

# 6. M6：Host pinned DRAM → GPU HBM

这和上一条方向相反，但应该单独做，因为 H2D 与 D2H 最佳 batching/layout 不一定一样。

vLLM PR #54483 是很直接的证据。

NIXL 无法直接 register device memory 时，会走：

\[
GPU
\leftrightarrow
CPU staging buffer.
\]

原先每个 KV cache group 各做一次 H2D/D2H：

```text
group0 → copy
group1 → copy
group2 → copy
...
```

PR 将多个 group 合并：

\[
\text{groups}\times\text{layers copies}
\]

降成：

\[
\text{layers copies}.
\]

这虽然字节数没变，但说明 host-tier representation 和 batching granularity 会直接影响 edge latency。

更典型的是 SGLang PR #41182：

> Add NIXL HOST staging for PD transfers through pinned host memory

真实数据流写得非常清楚：

```text
Prefill GPU
    │
    ▼
Pinned host slot
    │
    │ NIXL
    ▼
Decode pinned host ring
    │
    ▼
Decode GPU
```

即：

\[
\boxed{
GPU_{prefill}
\rightarrow
PinnedHost_{P}
\rightarrow
PinnedHost_{D}
\rightarrow
GPU_{decode}
}
\]

而不是：

\[
GPU\rightarrow GPU
\]

direct RDMA。

这里甚至有显式：

- Triton gather；
- host payload/header；
- host-to-host NIXL WRITE；
- Triton scatter。

因此一个完整 RQ1 case 可以比较：

\[
S_1:
GPU\ native
\rightarrow direct\ RDMA
\rightarrow GPU\ native
\]

\[
S_2:
GPU\ native
\rightarrow packed\ Host
\rightarrow Host
\rightarrow GPU\ native
\]

\[
S_3:
GPU\ transfer\text{-}native
\rightarrow HostCanonical
\rightarrow GPU\ consumer\text{-}native.
\]

这里 candidate representation 应包含：

\[
\{
GPU\ page\ layout,
Host\ packed\ layout,
Host\ canonical\ layout,
Decode\ native\ layout
\}.
\]

---

# 7. M7：GPU/Host DRAM → NVMe / filesystem

这一层以前的 RQ1 完全没有覆盖，但 2026 serving 已经是真实问题。

SGLang PR #39880：

> Add fast_file local storage backend

它明确指出原 `HiCacheFile`：

- 每一 page 都通过 staging tensor；
- serial reads；
- 不适合真实 local NVMe tier。

新实现支持：

\[
\boxed{
HostBuffer
\leftrightarrow
NVMe/File
}
\]

直接：

```text
readv / writev
```

在：

\[
page\text{-}first\ host\ pool
\]

和 page file 之间传输，避免中间 staging copy。

所以有两条路径：

```text
Host pool
  ↓
staging tensor
  ↓
filesystem
```

vs

```text
Host page-first pool
  ↓
direct readv/writev
  ↓
filesystem
```

这也是 RQ1：

\[
L_{\text{HostPool-best}}
\stackrel{?}{=}
L_{\text{NVMe-I/O-best}}.
\]

这里 candidate 可以是：

\[
\{
logical\ pool,
page\text{-}first,
contiguous\ staging,
direct\text{-}IO\ page
\}.
\]

vLLM PR #54327 也已经有 filesystem secondary KV tier：

\[
GPU/CPU\ cache
\rightarrow
filesystem\ KV.
\]

虽然这个 PR 本身主要做 capacity/LRU，不是 layout 优化，但它证明 filesystem 已经是正式的 KV memory tier，而不是研究者人为构造。

---

# 8. 更极端的一层：Disk/mmap → Host → GPU

这个在 2026 新模型里也有真实 case。

vLLM PR #54129：

> Support disk-backed (mmap) PLE table for Qwen3.8-Flash-Next

Qwen3.8-Flash-Next 的 PLE table 可达到约：

\[
47.68\ GiB
\]

PR 允许它不常驻 host RAM，而是：

```text
safetensors / file
       │
       │ mmap
       ▼
Linux page cache / host
       │
       │ gather selected rows
       ▼
stable GPU buffer
       │
       ▼
PLE consumer
```

也就是：

\[
\boxed{
Disk
\rightarrow
HostPageCache
\rightarrow
GPU
\rightarrow
Embedding/PLE
}
\]

这个 case 比 KV 更特别，但是真实 2026 LLM。

可以比较：

\[
\{
ResidentHostTable,
PinnedHostTable,
DiskMmapTable,
GPUResidentTable
\}.
\]

以及：

\[
\{
row\text{-}major,
gather\text{-}friendly,
coalesced\ file\ ranges,
GPU\ packed
\}.
\]

如果你希望 RQ1 证明不仅适用于 KV cache，而是适用于“现代 LLM persistent state / large table”，这个 case 很有价值。

---

# 9. Host expert weights → GPU expert cache → MoE GEMM

另一个非常现代的 real-world case 是 MoE expert offloading。

vLLM PR #56177：

> Shared GPU expert pool with a device-side planner for NVFP4 Marlin experts

针对 expert weights 放不下 GPU 的 MoE。

实际层级：

```text
Pinned Host DRAM
  full NVFP4 expert weights
          │
          │ cache miss
          ▼
shared GPU expert bank
          │
          ▼
Marlin MoE GEMM
```

即：

\[
\boxed{
HostPinned
\rightarrow
GPU\ expert\ pool
\rightarrow
Grouped\ GEMM
}
\]

而且这是 Qwen3.8-Flash-Next 的真实 end-to-end test path。

所以可以形成非常好的 RQ1：

\[
L_{\text{host-storage-best}}
\stackrel{?}{=}
L_{\text{GPU-Marlin-best}}.
\]

例如 host 可能偏向：

\[
\text{compact NVFP4 storage}
\]

GPU Marlin consumer 则偏向：

\[
\text{Marlin preshuffled / physical-row representation}.
\]

Strategy：

\[
HostCanonical
\rightarrow
GPUConversion
\rightarrow
Marlin
\]

vs

\[
HostMarlinNative
\rightarrow
direct\ copy
\rightarrow
Marlin.
\]

这很值得加入 MoE memory-hierarchy suite。

---

## 最终我建议把 memory hierarchy 作为 RQ1 的第二个正交轴

以前 case 大概是：

\[
(P,C,L,\text{shape})
\]

现在应该变成：

\[
\boxed{
(P,C,
M_P,M_C,
L_P,L_C,
R,
\text{shape})
}
\]

其中：

\[
M_P=\text{producer output memory level}
\]

\[
M_C=\text{consumer input memory level}.
\]

比如：

```text
Case A:
P = paged KV loader
C = Attention MFMA
Mp = HBM
Mc = Register
Lp = NHD/paged-HND/preshuffled
Lc = DotOperand
R  = LDS staging / direct load
```

又比如：

```text
Case B:
P = Prefill KV
C = Decode Attention
Mp = GPU HBM
Mc = GPU HBM on another node
Lp = prefill-native pages
Lc = decode-native pages
R  = direct RDMA / host staging / pack-unpack
```

又比如：

```text
Case C:
P = MoE GEMM2
C = peer combine
Mp = TMEM/register/SMEM
Mc = peer GPU HBM
Lp = local GEMM epilogue
Lc = route-slot-major
R  = local-store+collective / peer-direct-store
```

---

## 论文里最值得保留的 memory-level core suite

如果不想实验爆炸，我建议不要把每一级两两组合，而是保留下面 7 条真实且互补的路径：

| Core case | Memory hierarchy | Real evidence |
|---|---|---|
| Attention operand load | `HBM → Reg` vs `HBM → LDS → Reg` | Triton #8450/#7968，Paged Attention |
| Attention staging | `HBM → LDS → MMA` | IREE #23782/#22356 |
| GEMM epilogue | `TMEM → Reg → HBM` | CUTLASS #3313 |
| Distributed MoE | `Reg/SMEM → local HBM → collective` vs `→ peer HBM` | FlashInfer #5208 |
| Sparse KV serving | `Host DRAM → GPU hot buffer → Attention` | HiSparse/vLLM #46326/#55398 |
| PD serving | `GPU → pinned Host → network → Host → GPU` | SGLang #41182 |
| Long-term tiering | `GPU/Host → NVMe → Host → GPU` | SGLang #39880，vLLM FS KV tier |

再加一个扩展：

\[
\boxed{
HostExpertWeights
\rightarrow
GPUExpertPool
\rightarrow
MoEGEMM
}
\]

来自 vLLM #56177。

这样 memory hierarchy 这一维就真正覆盖了：

\[
\boxed{
\text{on-chip}
\rightarrow
\text{device memory}
\rightarrow
\text{peer device}
\rightarrow
\text{host}
\rightarrow
\text{storage/network}
}
\]

而不是只在 global memory 里改变 NHD/HND。

最重要的是，这些都不是人为创造的 memory path：它们分别来自 Paged Attention、Blackwell GEMM、MoE EP、HiSparse、PD disaggregation、KV offload、Qwen PLE / expert offload 等 2025–2026 实际 LLM 系统路径。这样扩展之后，RQ1 就可以真正研究：

\[
\boxed{
\text{局部最优 representation 是否会随着 memory hierarchy boundary 改变，
并导致整个 producer→consumer edge 的最优选择反转？}
}
\]

这会比当前 v10/v13 只在 GPU global-memory KV layout 上做实验强很多。

---

可以。重新按“2025–2026 主流 LLM 中真实存在、而且 producer/consumer 粒度足够粗”这个标准筛以后，我会明显收缩前一版候选。像 `WMMA→store`、`tt.load→tt.dot`、单纯 `convert_layout` 这类编译器内部边，不应该作为 RQ1 的主实验，只适合做 mechanism supplement。

更适合作为 RQ1 主体的是下面这些真实 recurring patterns。它们确实出现在 2025–2026 的主流模型族里：Qwen3 是 GQA+MoE，DeepSeek-V3 是 MLA+DeepSeekMoE，Kimi K2 是 MLA+MoE；Qwen3-Next 明确采用 `3×Gated DeltaNet + 1×Gated Attention` 的 hybrid layout，并带 MoE/MTP；Kimi K3 则是 69 个 KDA 层 + 24 个 Gated MLA 层 + LatentMoE；MiniMax-M3 引入 MSA sparse attention。citeturn126072search4turn126072search0turn372357search0turn765992search6turn372357search3turn372357search1

我建议最终把 RQ1 的 real-world producer–consumer variety 收敛到下面 9 类。

| 优先级 | Real-world edge | Producer | Consumer | 典型 2025–26 模型 |
|---|---|---|---|---|
| A1 | QKV/Norm/RoPE → Attention | QKV projection + QK-Norm/RoPE/KV prep | GQA/MLA Attention | Qwen3/3.5/3.8、Gemma3、DeepSeek、Kimi |
| A2 | Attention BMM → O-Proj | attention value aggregation/BMM | output projection GEMM | 几乎所有 Transformer/MLA |
| A3 | Router/Dispatch → Expert GEMM | TopK + token dispatch | grouped expert GEMM | Qwen3 MoE、DeepSeek、Llama4、Kimi |
| A4 | Expert GEMM2 → Combine | expert down projection | MoE combine / reduce-scatter | distributed MoE |
| A5 | KV Update → Attention | persistent KV-cache producer | decode/prefill attention | 几乎所有 cache-based LLM |
| A6 | Packed QKV → Linear Attention | fused QKV projection | GDN/KDA prefill/decode | Qwen3-Next/3.5–3.8、Kimi K3 |
| A7 | Prefill State → Decode | prefill attention/KDA/GDN | decode attention/recurrent op | 所有 serving，尤其 hybrid models |
| A8 | Sparse Indexer/Gather → Sparse Attention | QSA/MSA/indexer/KV gather | sparse attention | Qwen3.8、MiniMax-M3、DeepSeek-V4 类 |
| A9 | Draft → Target Verify | MTP/draft model | target verification | Qwen3-Next、现代 spec-decode serving |

另外可以保留两个 deployment-level 的 B 类 case：

\[
\text{Prefill worker}\rightarrow\text{KV transfer}\rightarrow\text{Decode worker}
\]

以及：

\[
\text{DCP/CP communication}\rightarrow\text{Attention}.
\]

它们是真实生产 serving stage，但不是每个单机 LLM forward 都有，所以不应该和 A1–A9 放在同一证据等级。

---

### A1. `QKV / QK-Norm / RoPE / KV preparation → Attention`

这是我认为最应该替换当前 synthetic KV-write benchmark 的第一类。

真实模型中的数据流通常是：

```text
Hidden states
     │
     ▼
QKV Projection
     │
     ▼
Q/K Norm + RoPE
     │
     ├──────── Q ──────────────┐
     │                         │
     └── K/V → KV preparation │
                │              │
                ▼              │
             KV cache          │
                │              │
                └──────────────┤
                               ▼
                           Attention
```

它不是人为拼出来的。DeepSeek-V3 的 MLA 模块源码就明确存在 `wq/wkv_a/wkv_b → MLA → wo`；Qwen3 是标准 GQA；Qwen3-Next 的 gated attention 是 16 Q heads / 2 KV heads；Kimi K2/K3 使用 MLA。citeturn126072search3turn126072search4turn765992search6turn372357search0

并且 2026 年框架 PR 正在直接优化这条边：

- vLLM #52901：`gated QKV split + Q/K RMSNorm + RoPE + KV update → Attention`
- vLLM #57097：Qwen3.8 的 `QK-Norm/RoPE/gate/QSA-prep/KV write → QSA`
- vLLM #52363：RoPE 后直接写 paged KV cache
- SGLang #41698：DSA decode 的 RoPE + KV preparation → AITER attention
- FlashInfer #5667：MLA projection result → packed FP8 K/V → ragged attention

其中 vLLM #57097 很有代表性：baseline 是三个真实 kernels——QK norm/RoPE、QSA pre-indexer、`reshape_and_cache_flash`；PR 将其变成一个 `qsa_prepare`，直接把 K/V 写进 paged cache，明确消除了 K intermediate round-trip。urlvLLM PR #57097https://github.com/vllm-project/vllm/pull/57097

建议不要只测 `{NHD,HND}`，而是根据 attention family 分组：

\[
\mathcal L_{\rm GQA}
=
\{NHD,HND,paged\_NHD,paged\_HND\}
\]

MLA：

\[
\mathcal L_{\rm MLA}
=
\{
latent,
expanded\ K/V,
packed\ K/V,
FP8\ packed
\}
\]

Sparse attention：

\[
\mathcal L_{\rm sparse}
=
\{
full\ pool,
paged,
compact\ gathered,
consumer\text{-}native
\}.
\]

---

### A2. `Attention BMM → O-Proj`

这是非常好的 operator→operator RQ1，而且比 KV-write 更普适。

真实计算是：

\[
Attention(Q,K,V)
\rightarrow
Y_{\text{heads}}
\rightarrow
W_OY.
\]

SGLang PR #34498 就是在真实 Kimi attention path 上解决这个问题。

原来 BMM 输出：

\[
[H,T,D_v]
\]

然后 `o_proj` 需要：

\[
[T,H,D_v]
\rightarrow
[T,H D_v].
\]

因此：

```text
Attention/BMM
     │
     │ [heads,tokens,vdim]
     ▼
 transpose + copy
     │
     ▼
[tokens,heads,vdim]
     │
     ▼
   O-Proj
```

PR 改成 producer 直接输出：

\[
[T,H,D_v]
\]

从而 downstream `flatten` 变成免费 view。这个 PR 报告整个 serving throughput 最多约 +3.3%。urlSGLang PR #34498https://github.com/sgl-project/sglang/pull/34498

所以这是非常干净的：

\[
\boxed{
L_{\text{attention-output-best}}
\stackrel{?}{=}
L_{\text{attention→OProj-best}}
}
\]

建议候选：

\[
\{
[H,T,D],
[T,H,D],
[T,H\!\cdot\!D]
\}
\]

以及：

- producer-native；
- consumer-native direct write；
- producer-native + transpose/materialize；
- stride-view（若合法）。

这个 case 我会列为核心必做。

---

### A3. `Router/TopK + Dispatch → Grouped Expert GEMM`

这可能是除了 Attention 外最重要的 real-world family。

原因是 MoE 已经不是边缘结构：

- Qwen3-30B-A3B 有 128 experts、每 token 选 8。citeturn126072search4
- DeepSeek-V3 使用 DeepSeekMoE。citeturn126072search0
- Kimi K2 有 384 experts。citeturn372357search0
- Kimi K3 有 896 experts、选 16。citeturn372357search3
- Llama 4 Maverick 有 128 routed experts + shared expert。citeturn696484search0

真实 edge 是：

```text
tokens [T,H]
    │
    ▼
 Router / TopK
    │
    ▼
 Dispatch
    │
    │ activation representation
    ▼
 Grouped Expert GEMM
```

SGLang #37261 就非常适合 RQ1：原来的 DeepEP dispatch 产生 non-expanded representation，再由 `ep_scatter` 重排；新的方案直接让 dispatch 产生按 `(token,expert)` 展开且 expert-grouped 的 rows，从而删除中间 full-hidden copy。urlSGLang PR #37261https://github.com/sgl-project/sglang/pull/37261

SGLang #39039 又直接使用：

> DeepGEMM psum layout + zero-copy dispatch/combine.

urlSGLang PR #39039https://github.com/sgl-project/sglang/pull/39039

vLLM #54109 则把真实 activation contract 显式定义成：

\[
\{
Standard,
PaddedStandard,
BatchedExperts
\}.
\]

urlvLLM PR #54109https://github.com/vllm-project/vllm/pull/54109

所以建议候选：

\[
\boxed{
\mathcal L_{\rm MoE}=
\{
TokenMajor,
Standard,
PaddedStandard,
ExpandedExpertMajor,
BatchedExperts,
DeepGEMM\text{-}psum
\}
}
\]

这里的 layout 已经不是简单 axis permutation，而是更一般的 physical representation。

这是非常有价值的 RQ1 扩展。

---

### A4. `Expert GEMM2 → Combine`

这同样是真实 MoE 热路径。

典型 distributed MoE：

```text
Dispatch
   ↓
Expert GEMM1
   ↓
Activation
   ↓
Expert GEMM2
   ↓
expert-local outputs
   ↓
Combine / ReduceScatter / All-to-All
```

FlashInfer #5208 正在做：

\[
\boxed{
GEMM2\ epilogue
\rightarrow
cross\text{-}GPU\ combine
}
\]

原方案：

```text
GEMM2
  ↓
global output
  ↓
separate combine collective
```

新方案：

```text
GEMM2 epilogue
  ↓
direct peer/token-slot destination
  ↓
local weighted reduction
```

即 producer 的 output representation 为 downstream communication consumer 服务。urlFlashInfer PR #5208https://github.com/flashinfer-ai/flashinfer/pull/5208

建议 strategy 集：

\[
\{
local\_contiguous,
route\_slot\_major,
peer\_scatter,
canonical\_combine
\}.
\]

这尤其适合多 GPU MoE，不适合作为单 GPU mandatory case。

---

### A5. `KV-cache Update → Attention`

这仍应该保留，因为它确实是真实且普遍，而不是因为 v10 已经用了它。

但应该替换当前人为的：

\[
KVWrite\rightarrow HeadLocalScan
\]

为真实 attention consumer：

\[
\boxed{
KVCacheUpdate
\rightarrow
FlashAttention/FlashInfer/TRTLLM/MLA/MSA
}
\]

真实 issue 已经证明 backend contract 不统一。例如 vLLM #55312 的标题就是：

> speculative decoding draft and target can select attention backends with disjoint KV cache layouts.

urlvLLM issue #55312https://github.com/vllm-project/vllm/issues/55312

SGLang #34525 更直接：

\[
NHD\ persistent\ pool
\rightarrow
MSA\ sparse\ prefill
\]

其中 consumer 需要：

\[
[page,H_{kv},page\_size,D]
\]

即 HND-like representation；当 \(H_{kv}>1\) 时，原来的 permute 并非 contiguous，需要 materialization。urlSGLang PR #34525https://github.com/sgl-project/sglang/pull/34525

因此这里建议测：

\[
\{
NHD,
HND,
paged\_NHD,
paged\_HND,
interleaved,
noninterleaved
\}
\]

但 consumer 必须换成框架真实 attention kernel。

---

### A6. `Packed QKV Projection → GDN/KDA`

这是 2025–2026 新架构中非常值得新增的一类。

Qwen3-Next 已经正式采用 Gated DeltaNet + gated attention hybrid architecture，且其后续 Qwen3.5/3.6/3.7/3.8 都沿用这类 hybrid design。citeturn765992search0turn765992search6 Kimi K3 则大量使用 KDA。citeturn372357search3

因此：

```text
QKV projection
      │
      ▼
packed [T, 3*H*D]
      │
      ├── strided Q view
      ├── strided K view
      └── strided V view
                │
                ▼
             KDA/GDN
```

已经是一个非常现实的 operator boundary。

FlashInfer #5846 的动机就是：serving 实际上传入的是 packed projection row 的 strided `split(dim=-1)` views；原先部分 KDA prefill kernels 会把 Q/K/V densify，产生三个完整 staging copies；现在 consumer 直接读取真实 stride。urlFlashInfer PR #5846https://github.com/flashinfer-ai/flashinfer/pull/5846

所以候选很清楚：

\[
\boxed{
\{
PackedRowStridedQKV,
DenseSeparateQKV,
ConsumerNativeStrided
\}
}
\]

这比人为设计 NHD/HND 更可信。

---

### A7. `Prefill → Persistent State → Decode`

这也是非常真实的 stage，但最好拆成两种，不要混。

传统 Transformer：

\[
PrefillAttention
\rightarrow
KVCache
\rightarrow
DecodeAttention.
\]

Hybrid recurrent model：

\[
GDN/KDA\ Prefill
\rightarrow
RecurrentState
\rightarrow
GDN/KDA\ Decode.
\]

后者在 2025–2026 特别重要，因为 Qwen3-Next 系列和 Kimi K3 都是现实大模型，而不是研究 toy。Qwen3-Next 的公开结构是 36 GDN + 12 gated-attention 风格层；Kimi K3 是 69 KDA + 24 Gated MLA。citeturn765992search6turn372357search3

SGLang #34299 就明确叫：

> zero-copy native prefill checkpoints and packed decode.

所以这条真实 edge 就是：

```text
KDA Prefill
    │
    ▼
persistent checkpoint/state
    │
    ▼
KDA Decode
```

urlSGLang PR #34299https://github.com/sgl-project/sglang/pull/34299

候选：

\[
\{
PrefillNativeState,
DecodePackedState,
CommonCanonicalState
\}
\]

以及：

```text
prefill-native → decode direct
prefill-native → pack once → decode
prefill directly emits decode-packed → decode
```

这一类非常适合研究“persistent representation”的 RQ1。

---

### A8. `Sparse Indexer / Gather → Sparse Attention`

这个也是 2026 非常现实的新 family，不是人为拼出来的。

Qwen3.8-Flash-Next 使用 QSA：轻量 indexer 先选择重要上下文，再执行 sparse attention。citeturn765992search0

MiniMax-M3 则正式以 MSA 作为其 million-context sparse attention。citeturn372357search1

因此真实计算图：

```text
KV / hidden states
       │
       ▼
Indexer / Top-K / Page selection
       │
       ▼
selected indices/pages
       │
       ▼
Gather / Compact / Layout prepare
       │
       ▼
Sparse Attention
```

RQ1 可以研究两条边：

\[
Indexer\rightarrow Gather
\]

和更重要的：

\[
\boxed{
Gather/Preparation
\rightarrow
SparseAttention
}
\]

候选 representation：

\[
\{
FullPoolNHD,
FullPoolHND,
CompactGatheredNHD,
CompactGatheredHND,
PagedConsumerNative
\}.
\]

SGLang #34525 恰好就是实际 MSA consumer 上的：

\[
NHD\ pool
\rightarrow
gather+transpose
\rightarrow
compact\ HND
\rightarrow
MSA.
\]

这应该成为 sparse-attention RQ1 的主 case，而不是再人工写一个 sparse scan。

---

### A9. `Draft/MTP → Target Verify`

这一类我前面没有强调够，但它非常符合 2025–2026 real-world。

Qwen3-Next 官方明确把 MTP 列为模型特性。citeturn765992search6 DeepSeek-V3 也将 multi-token prediction 作为模型训练目标。citeturn126072search0

serving 时真实 stage 是：

```text
Draft / MTP
     │
     │ γ candidate tokens
     ▼
draft state/KV
     │
     ▼
Target multi-token verification
```

这里不是人为构造 stage。

vLLM #55312 已经直接暴露：

\[
\mathcal L_{\rm draft}
\cap
\mathcal L_{\rm target}
=
\varnothing
\]

可能发生。

FlashInfer #4731 又显示 `q_len_per_req>1` 的 masked multi-token verify 可以出现严重性能问题，而 q_len=1 正常。urlFlashInfer issue #4731https://github.com/flashinfer-ai/flashinfer/issues/4731

所以它非常适合 RQ1：

\[
L_{\rm draft-best}
\stackrel{?}{=}
L_{\rm draft\rightarrow verify-best}.
\]

候选 strategy：

\[
\{
DraftNative,
TargetNative,
SharedCanonical,
DraftNative\rightarrow Convert,
DualRepresentation
\}.
\]

这比普通 single-token decode 更能体现 2025–2026 inference workload。

---

## 两个 B 类：是真实 stage，但不要和上述 A 类混在一起

### B1. `Prefill worker → KV transfer → Decode worker`

PD disaggregation 已经是现实 serving 架构，不是人为构造的 stage。

SGLang #41958 就是在真实：

\[
Prefill\rightarrow Mooncake/RDMA\rightarrow Decode
\]

路径上支持 matching DCP peers，匹配时可以直接 whole-page copy，无需 redistribution/packing。urlSGLang PR #41958https://github.com/sgl-project/sglang/pull/41958

vLLM #59111 则明确提出：

\[
standard\ KV\ pages
\rightarrow
NIXL
\rightarrow
AITER\ shuffled\ pages
\]

在 decode side 转换。urlvLLM issue #59111https://github.com/vllm-project/vllm/issues/59111

所以 layout：

\[
\{
PrefillNative,
CanonicalWire,
DecodeNative,
BackendShuffled
\}.
\]

这个适合作为 system-level external validity，而不是 RQ1 的第一核心实验。

### B2. `DCP/CP AllGather → Attention`

vLLM #59070 也是很好的真实例子：

原来：

```text
FP8 cached context
 → dequant
 → BF16 AllGather
 → reorg
 → KV projection
 → K/V pack
```

改成：

```text
FP8 cached context
 → FP8 AllGather
 → fused reorg/dequant/KV projection/KV pack
```

也就是说：

\[
\boxed{
communication\ representation
\rightarrow
attention-preparation\ consumer
}
\]

真实影响整条 edge。urlvLLM PR #59070https://github.com/vllm-project/vllm/pull/59070

但它属于 large-scale DCP deployment，所以应放 B 类。

---

## 这次我会明确剔除哪些前一版候选

以下内容不是没有研究价值，而是不应该成为“2025–2026 real-world LLM RQ1 主 case”：

- `tt.load → tt.dot`
- `WMMA → store`
- 单独 `reduce → layout conversion`
- TileLang fragment→shared-memory copy
- 单纯 CUTLASS TMEM→RMEM
- generic Conv→Pool
- 任意 synthetic row/column reducer

这些更适合：

\[
\boxed{\text{mechanism validation}}
\]

用来解释“为什么 layout conflict 会出现”，而不是用来证明：

> 现代 LLM 中真实 operator/subgraph/stage 确实存在 cross-edge layout inversion。

---

## 最终我建议你的 RQ1 核心 suite

如果实验预算允许，我会真正实现下面 9 条，而不是继续扩展 synthetic shape：

```text
1. QKV/Norm/RoPE/KVPrep
      → GQA Attention

2. MLA latent/KV preparation
      → MLA Attention

3. Attention BMM
      → O-Proj GEMM

4. MoE Router/Dispatch
      → Grouped Expert GEMM

5. Expert GEMM2
      → Combine / ReduceScatter

6. Packed QKV projection
      → GDN/KDA

7. KDA/GDN Prefill
      → Persistent State
      → Decode

8. Sparse KV Gather/Prepare
      → QSA/MSA/Sparse-MLA

9. Draft/MTP
      → Multi-token Target Verify
```

另外：

```text
10. Prefill worker
      → KV Transfer
      → Decode worker
```

作为 system-level extension。

这组 case 有一个很重要的优点：它们分别覆盖了当前主流架构的核心变化：

\[
\begin{array}{c}
\text{GQA}\\
\text{MLA}\\
\text{MoE}\\
\text{GDN/KDA}\\
\text{Sparse Attention}\\
\text{Speculative/MTP}\\
\text{Disaggregated Serving}
\end{array}
\]

而不是研究者为了制造 layout conflict 自己拼出来的算子链。

如果要进一步压缩成论文最核心、且 reviewer 最难质疑的 6 类，我会保留：

\[
\boxed{
\begin{aligned}
&QKV/Norm/RoPE/KVPrep\rightarrow Attention\\
&Attention\ BMM\rightarrow OProj\\
&MoE\ Dispatch\rightarrow GroupedGEMM\\
&PackedQKV\rightarrow GDN/KDA\\
&SparseKVPrep\rightarrow SparseAttention\\
&PrefillState\rightarrow Decode
\end{aligned}}
\]

这六条既足够新，又分别在多个真实 2025–2026 模型/框架里出现，而且 producer 和 consumer 都已经是有明确语义的真实算子或子图，而不是微型 memory primitive。

---

可以。把 RQ1 真正做强，应该从“找几个典型 PR”升级成一个系统性的 producer–consumer edge catalog。

我重新按 2025-01-01 到 2026-10-01 的 GitHub issue/PR 做了较宽的筛选，覆盖 vLLM、SGLang、FlashInfer、TensorRT-LLM、Triton、CUTLASS/CuTe、PyTorch/Inductor、IREE、TileLang、TVM。筛选条件不是标题里出现 `layout` 就算，而是尽量满足以下至少一项：

\[
\boxed{
\text{producer output representation}
\rightarrow
\text{consumer input representation}
}
\]

之间明确出现 layout/stride/packing/transpose/materialization/zero-copy/format/quantized representation 的选择、冲突或 repair。

这不能保证“GitHub 上绝对一个都没漏”，因为很多相关 PR 标题不含 layout 关键词，但已经比前面的一批扩大很多。下面我先给你最适合作为实验 case 的完整分类，然后再给候选 layout 集合。

---

## 一、我建议把 RQ1 的 producer–consumer 分成 7 个 family

不要再只有：

\[
KVWrite\rightarrow KVRead
\]

而是至少覆盖：

\[
\boxed{
\begin{array}{ll}
F1:& Projection/Norm/RoPE\rightarrow Attention\\
F2:& Attention/GEMM\rightarrow Projection/GEMM\\
F3:& MoE\ Dispatch\rightarrow Expert\ GEMM\rightarrow Combine\\
F4:& Prefill\rightarrow Persistent\ State\rightarrow Decode\\
F5:& Communication/Offload\rightarrow Consumer\\
F6:& Compiler\ IR\ Producer\rightarrow Consumer\\
F7:& Vision/Conv/Attention\ producer\rightarrow downstream\ op
\end{array}}
\]

这样既有算子级，也有子图级、阶段级。

---

# 二、F1：QKV / Norm / RoPE / KV preparation → Attention

这是和现有 RQ1 最接近，但 producer 已经不再只是“写内存”。

| 来源 | Producer | Consumer | RQ1 冲突点 |
|---|---|---|---|
| [vLLM PR #52363](https://github.com/vllm-project/vllm/pull/52363) | RoPE + K/V cache emission | FlashAttention decode | producer 直接生成 consumer cache layout vs 普通 RoPE→cache |
| [vLLM PR #52901](https://github.com/vllm-project/vllm/pull/52901) | gated QKV split + Q/K RMSNorm + RoPE + gate + KV update | Attention | fused producer 支持 NHD/HND |
| [vLLM PR #57097](https://github.com/vllm-project/vllm/pull/57097) | QSA prepare：QK norm+RoPE+gate+cache writes | QSA sparse attention | 消除 K intermediate round trip，直接写 paged cache |
| [vLLM PR #47757](https://github.com/vllm-project/vllm/pull/47757) | sparse MLA QK-RoPE + Q/KV concat + cache write | Sparse MLA | fused output representation |
| [vLLM PR #55230](https://github.com/vllm-project/vllm/pull/55230) | decode RoPE + MLA KV write + FP8 Q assembly | MLA attention | Q/KV representation共同决定 |
| [vLLM PR #59035](https://github.com/vllm-project/vllm/pull/59035) | QKNorm+RoPE+KV cache + Q quant | attention | quantized Q 与 cache layout 联合 |
| [vLLM issue #46979](https://github.com/vllm-project/vllm/issues/46979) | RoPE/KV producer families | GQA/MLA/SparseMLA | 明确要求审计 fusion coverage |
| [SGLang PR #41698](https://github.com/sgl-project/sglang/pull/41698) | DSA RoPE + KV preparation | AITER DSA attention | separate vs fused preparation |
| [SGLang PR #41533](https://github.com/sgl-project/sglang/pull/41533) | MLA absorb+RoPE+KV write | MLA decode | fused representation只在 decode-sized mode 使用 |
| [SGLang PR #35142](https://github.com/sgl-project/sglang/pull/35142) | RoPE + Q quant | FA3 | consumer 对 quantized Q representation 的要求 |
| [SGLang PR #34394](https://github.com/sgl-project/sglang/pull/34394) | DSA indexer Q/K preparation | DSA indexer | 多个 prep op 融成 consumer-native representation |
| [SGLang PR #38583](https://github.com/sgl-project/sglang/pull/38583) | fused DSA indexer decode preparation | sparse decode | 四 kernel producer 子图→decode |
| [SGLang PR #40710](https://github.com/sgl-project/sglang/pull/40710) | FP8 indexer writer | DSA/indexer consumer | writer representation |
| [FlashInfer PR #5667](https://github.com/flashinfer-ai/flashinfer/pull/5667) | MLA projection output | TRTLLM ragged attention | `[k_nope|v]` → contiguous FP8 K/V pack |
| [FlashInfer PR #5402](https://github.com/flashinfer-ai/flashinfer/pull/5402) | NVFP4 KV producer | MSA sparse decode | 明确定义 canonical page layout contract |
| [FlashInfer issue #5601](https://github.com/flashinfer-ai/flashinfer/issues/5601) | paged KV | cuDNN prefill | HND/NHD contract 错误可直接导致错误结果 |
| [FlashInfer issue #3856](https://github.com/flashinfer-ai/flashinfer/issues/3856) | KV pool | TRTLLM sparse MLA | consumer 没有正确尊重 `stride(0)` |
| [FlashInfer PR #5075](https://github.com/flashinfer-ai/flashinfer/pull/5075) | sparse MLA KV producer | SM120 sparse MLA | runtime KV row stride/canonical rows |
| [TensorRT-LLM issue #16801](https://github.com/NVIDIA/TensorRT-LLM/issues/16801) | layer-specific KV cache | vanilla attention | consumer 索引依赖真实 cache layout |
| [TensorRT-LLM PR #18485](https://github.com/NVIDIA/TensorRT-LLM/pull/18485) | ragged sparse rows | sparse attention | 显式 row representation contract |

这一组特别适合定义：

\[
P=
\text{QKV/Norm/RoPE/KV-prep}
\]

\[
C=
\text{Attention}
\]

候选 representation：

\[
\mathcal L_{\rm attn}=
\{
NHD,HND,paged\_NHD,paged\_HND,
LBNHC,LBHNC,BLNHC,BLHNC
\}
\]

再加：

\[
\{
\text{packed FP8},
\text{NVFP4 planar},
\text{strided QKV view}
\}
\]

但每个 backend 先做 legality filtering。

---

# 三、F2：GEMM/BMM/Attention output → 下游 GEMM/Projection

这一类特别重要，因为它完全摆脱了“KV cache 特例”。

| 来源 | Producer | Consumer | layout tension |
|---|---|---|---|
| [SGLang PR #34498](https://github.com/sgl-project/sglang/pull/34498) | absorbed BMM | `o_proj` | `(heads,tokens,vdim)` vs `(tokens,heads,vdim)` |
| [SGLang PR #33166](https://github.com/sgl-project/sglang/pull/33166) | quantized activation producer | downstream Linear/GEMM | row-major scale vs bpreshuffle consumer scale |
| [Triton issue #10342](https://github.com/triton-lang/triton/issues/10342) | `tl.dot #1` | FFMA/bias → `tl.dot #2` | 两 dot 中间插入 `convert_layout` |
| [Triton issue #6569](https://github.com/triton-lang/triton/issues/6569) | transposed operand producer | `tl.dot` | 会通过 shared memory rearrange |
| [Triton PR #10563](https://github.com/triton-lang/triton/pull/10563) | WMMA | global store | WMMA result layout 应依据 consumer store order 选择 |
| [Triton PR #11704](https://github.com/triton-lang/triton/pull/11704) | FMA/MMA result | store | accumulator-native vs coalesced store layout |
| [Triton PR #10360](https://github.com/triton-lang/triton/pull/10360) | small tensor op | next op | register permutation vs SMEM relayout |
| [CUTLASS issue #3540](https://github.com/NVIDIA/cutlass/issues/3540) | GEMM accumulator | broadcast epilogue | RowMajor/ColumnMajor 会改变 broadcast coordinate |
| [CUTLASS PR #3345](https://github.com/NVIDIA/cutlass/pull/3345) | SM120 blockscaled GEMM | epilogue store | accumulator→FP32 output layout |
| [CUTLASS PR #3313](https://github.com/NVIDIA/cutlass/pull/3313) | TMEM accumulator | RMEM/epilogue | TMEM-load layout 宽度影响 downstream |
| [CUTLASS PR #3650](https://github.com/NVIDIA/cutlass/pull/3650) | Stream-K accumulation | broadcast/store epilogue | fragment/output coordinate mapping |
| [PyTorch PR #197100](https://github.com/pytorch/pytorch/pull/197100) | non-contiguous tensor/view | matmul | zero-copy stride vs materialized contiguous |
| [PyTorch issue #188635](https://github.com/pytorch/pytorch/issues/188635) | producer tensor | view→transpose→contiguous→attention | compiler错失 producer→consumer fusion |
| [PyTorch issue #195320](https://github.com/pytorch/pytorch/issues/195320) | permuted K | SDPA | semantic transpose vs physical permute contract |

最值得直接做的一条是 SGLang #34498，因为它已经天然包含：

```text
BMM
 │
 │ output = (heads,tokens,vdim)
 ▼
transpose copy
 │
 ▼
(tokens,heads,vdim)
 │
 ▼
o_proj
```

PR 改成：

```text
BMM
 │
 │ direct output
 ▼
(tokens,heads,vdim)
 │
 ▼
o_proj
```

这就是：

\[
\boxed{
L_{\rm BMM-best}
\stackrel{?}{=}
L_{\rm BMM\rightarrow oproj-best}
}
\]

建议候选：

\[
\{
HND-like,\ NHD-like,\ RowMajor,\ ColumnMajor,
producer\_native,\ consumer\_native
\}
\]

这是应该加入 RQ1 的核心 case。

---

# 四、F3：MoE Dispatch → Grouped GEMM → Combine

这一组我非常建议作为一个完整 family。

| 来源 | Producer/Consumer boundary | representation 问题 |
|---|---|---|
| [vLLM PR #54109](https://github.com/vllm-project/vllm/pull/54109) | DeepEP Prepare/Finalize → Expert kernel | `Standard/PaddedStandard/BatchedExperts` |
| [vLLM issue #47095](https://github.com/vllm-project/vllm/issues/47095) | EP/quant producer → backend | physical representation 不一定只是轴 permutation |
| [SGLang PR #37261](https://github.com/sgl-project/sglang/pull/37261) | DeepEP dispatch → DeepGEMM | non-expanded vs expanded expert-major |
| [SGLang PR #39039](https://github.com/sgl-project/sglang/pull/39039) | MoonEP dispatch → DeepGEMM | DeepGEMM psum layout + zero-copy shard/combine |
| [SGLang PR #39313](https://github.com/sgl-project/sglang/pull/39313) | shared expert + routed experts | FP8 shared representation vs MXFP4 routed representation |
| [SGLang PR #29525](https://github.com/sgl-project/sglang/pull/29525) | DeepEP v2 dispatch | downstream MoE runner |
| [SGLang PR #38884](https://github.com/sgl-project/sglang/pull/38884) | MoE routed activations | CUTLASS grouped GEMM | MXFP4×BF16 grouped representation |
| [FlashInfer PR #5751](https://github.com/flashinfer-ai/flashinfer/pull/5751) | ragged expert rows | grouped GEMM | ragged expert representation |
| [FlashInfer PR #5208](https://github.com/flashinfer-ai/flashinfer/pull/5208) | GEMM2 epilogue | cross-GPU combine | direct peer scatter vs GEMM output→collective |
| [FlashInfer PR #5309](https://github.com/flashinfer-ai/flashinfer/pull/5309) | MegaMoE FC1 | FC2 | split representation/backend |
| [FlashInfer issue #3780](https://github.com/flashinfer-ai/flashinfer/issues/3780) | MegaMoE activation | MegaMoE | SM90 support/layout space |
| [TensorRT-LLM PR #16954](https://github.com/NVIDIA/TensorRT-LLM/pull/16954) | DeepEP NVFP4 transport | MoE expert compute | transport representation overhead |

这组候选 representation 应该至少包括：

\[
\boxed{
\mathcal L_{\rm MoE}=
\{
Standard,
PaddedStandard,
ExpandedExpertMajor,
BatchedExperts,
DeepGEMM\ psum,
RouteSlotMajor
\}}
\]

并另外控制 dtype/scale：

\[
FP8,\ MXFP4,\ NVFP4
\]

但 dtype 应作为第二个因素，不能和 layout 混成一个变量。

一个非常标准的 experiment：

\[
T_D(L_D)
\]

\[
T_G(L_G)
\]

\[
T_{D\rightarrow G}(L_D,R,L_G)
\]

其中 repair 可以是：

\[
scatter,\ permute,\ pad,\ expand,\ psum\ conversion
\]

然后再扩展：

\[
D\rightarrow GEMM1\rightarrow GEMM2\rightarrow Combine
\]

这会得到一个真正 coarse-grained RQ1。

---

# 五、F4：Prefill → persistent state → Decode

这是比单算子更粗的一层，而且非常适合研究 persistent representation。

| 来源 | Producer | Consumer |
|---|---|---|
| [SGLang PR #34299](https://github.com/sgl-project/sglang/pull/34299) | KDA Prefill | KDA Decode |
| [SGLang PR #41400](https://github.com/sgl-project/sglang/pull/41400) | KDA prefill checkpoint | later KDA execution |
| [FlashInfer PR #5846](https://github.com/flashinfer-ai/flashinfer/pull/5846) | packed QKV projection | KDA prefill |
| [FlashInfer PR #4845](https://github.com/flashinfer-ai/flashinfer/pull/4845) | KDA prefill | FP32 persistent state |
| [FlashInfer PR #4262](https://github.com/flashinfer-ai/flashinfer/pull/4262) | recurrent prefill | recurrent decode |
| [FlashInfer PR #4581](https://github.com/flashinfer-ai/flashinfer/pull/4581) | GDN prefill | decode/MTP |
| [vLLM PR #52928](https://github.com/vllm-project/vllm/pull/52928) | Mamba state | FlashInfer ReplaySSM |
| [vLLM PR #54103](https://github.com/vllm-project/vllm/pull/54103) | KDA state producer | ATOM KDA/ReplaySSM |
| [SGLang PR #41968](https://github.com/sgl-project/sglang/pull/41968) | speculative Mamba state | replay consumer |

尤其 FlashInfer #5846 非常典型：

原来 serving 给出：

\[
[T,3HD]
\]

packed projection。

通过：

```text
split(dim=-1)
```

得到三个 non-contiguous/strided：

\[
Q,K,V
\]

以前部分 KDA prefill kernels 要：

\[
\text{dense }Q,K,V
\]

所以需要三次 staging copy。

现在 consumer 直接接受：

\[
\boxed{\text{packed-row strided Q/K/V views}}
\]

因此可以测试：

\[
\{
packed\_strided,
dense\_QKV,
consumer\_native\_packed
\}
\]

这就是非常好的：

\[
Projection\rightarrow KDA\ Prefill
\]

算子/子图级 RQ1。

---

# 六、F5：Prefill / Cache / Communication → Decode

这组属于 stage-level RQ1。

| 来源 | Producer | Consumer | 中间表示 |
|---|---|---|---|
| [vLLM issue #59111](https://github.com/vllm-project/vllm/issues/59111) | standard KV / NIXL | ROCm AITER decode | standard pages → shuffled pages |
| [vLLM issue #59110](https://github.com/vllm-project/vllm/issues/59110) | P-side KV | decode backend | heterogeneous TP + split K/V slot layout |
| [vLLM issue #45997](https://github.com/vllm-project/vllm/issues/45997) | attention cache | KV connector | kernel-specific vs layer-major constant-stride |
| [vLLM issue #55312](https://github.com/vllm-project/vllm/issues/55312) | shared KV pool | draft + target attention | backend layout sets can be disjoint |
| [vLLM PR #58380](https://github.com/vllm-project/vllm/pull/58380) | allocator/cache writer | attention layers | layer-outermost vs block-outermost |
| [vLLM PR #59070](https://github.com/vllm-project/vllm/pull/59070) | cached FP8 context | AllGather → MLA projection | FP8 wire vs BF16 wire |
| [SGLang PR #41953](https://github.com/sgl-project/sglang/pull/41953) | unified-memory P-side | PD transfer | exposes exact byte-offset/stride layout |
| [SGLang PR #41956](https://github.com/sgl-project/sglang/pull/41956) | Prefill TP layout | Decode TP layout | unequal TP representation |
| [SGLang PR #41958](https://github.com/sgl-project/sglang/pull/41958) | Prefill DCP state | Decode DCP state | matching layouts permit direct page→page transfer |
| [SGLang PR #40909](https://github.com/sgl-project/sglang/pull/40909) | hybrid SWA prefill | decode | heterogeneous TP KV transfer |
| [SGLang PR #41607](https://github.com/sgl-project/sglang/pull/41607) | recurrent state | Mooncake receiver | stride compatibility |
| [SGLang issue #37838](https://github.com/sgl-project/sglang/issues/37838) | prefill KV | TRTLLM decode | uniform-FP8 KV handshake |
| [SGLang issue #32143](https://github.com/sgl-project/sglang/issues/32143) | Prefill | Decode | transfer vs token replay |
| [FlashInfer PR #4924](https://github.com/flashinfer-ai/flashinfer/pull/4924) | local QKV | Ulysses exchange | low-precision communication representation |
| [TensorRT-LLM PR #19502](https://github.com/NVIDIA/TensorRT-LLM/pull/19502) | sparse KV state | offload | sparse KV representation |
| [TensorRT-LLM PR #19171](https://github.com/NVIDIA/TensorRT-LLM/pull/19171) | KVCM | Mooncake | cache transfer representation |
| [TensorRT-LLM PR #19303](https://github.com/NVIDIA/TensorRT-LLM/pull/19303) | draft KV pool | target/shared pool | paired draft cache |

这类实验的候选应该不是简单 NHD/HND，而是：

\[
\mathcal L_{\rm transfer}
=
\{
L_{\rm producer-native},
L_{\rm canonical-wire},
L_{\rm consumer-native}
\}
\]

策略至少：

\[
P_{native}\rightarrow transfer\rightarrow C
\]

\[
P_{native}\rightarrow pack\rightarrow transfer\rightarrow unpack\rightarrow C
\]

\[
P_{consumer-native}\rightarrow transfer\rightarrow C
\]

\[
P_{compressed}\rightarrow compressed\ transfer\rightarrow fused\ expand\rightarrow C
\]

这一组能够直接测试：

> “为了通信最优而选择 canonical layout，是否会损害 compute edge；或者 producer 是否应该直接产生 receiver-native representation？”

---

# 七、F6：真实 persistent KV layout conflicts

这一组虽然仍然是 KV，但比 v10 的 `{NHD,HND}` 丰富很多。

相关证据至少包括：

[vLLM issue #55312](https://github.com/vllm-project/vllm/issues/55312)：draft/target backend 的 KV layout support 可不相交。

[vLLM issue #59176](https://github.com/vllm-project/vllm/issues/59176)：GLM-5.3-Flash 出现 LBHNC vs BLHNC compatibility problem。

[vLLM PR #58380](https://github.com/vllm-project/vllm/pull/58380)：mixed page size 时，block-outermost layout 能显著改变 cache packing/padding。

[vLLM issue #55904](https://github.com/vllm-project/vllm/issues/55904)：offload persistent bytes 如果不区分 LBNHC/LBHNC，会静默解释错误。

[SGLang PR #38592](https://github.com/sgl-project/sglang/pull/38592)：统一内存从 layer-block style 转为 token-major dense representation，以支持异构 K/V row width。

[SGLang PR #40326](https://github.com/sgl-project/sglang/pull/40326)、[#40327](https://github.com/sgl-project/sglang/pull/40327)、[#40330](https://github.com/sgl-project/sglang/pull/40330)：分别处理 stride addressing、paged view construction、layout translation。

[SGLang PR #34525](https://github.com/sgl-project/sglang/pull/34525)：NHD pool → MSA HND consumer，当前最经典的真实 conflict case。

[SGLang PR #32272](https://github.com/sgl-project/sglang/pull/32272)：consumer 改为直接接受 non-interleaved paged KV，以删除 interleaving copy。

[FlashInfer PR #5329](https://github.com/flashinfer-ai/flashinfer/pull/5329)：同样叫 NHD 的情况下，equal/unequal K/V strides 仍会改变 consumer specialization。

因此 Attention persistent candidates 至少应该扩为：

\[
\boxed{
\begin{aligned}
\mathcal L_{KV}=\{&
NHD,HND,\\
&paged\_NHD,paged\_HND,\\
&LBNHC,LBHNC,LHBNC,\\
&BLNHC,BLHNC,BHLNC,\\
&interleaved,\ noninterleaved,\\
&token\text{-}major\ unified,\\
&backend\text{-}shuffled
\}
\end{aligned}}
\]

当然不能每个 case 全部运行，应由 consumer contract 筛合法集合。

---

# 八、F7：Triton compiler IR producer→consumer

Triton 的 issue/PR 数量非常多，建议单独成为一个 RQ1 sub-suite。

高相关条目包括：

| PR/Issue | 实际边 |
|---|---|
| [#8450](https://github.com/triton-lang/triton/pull/8450) | `LoadOp → DotOp` |
| [#10342](https://github.com/triton-lang/triton/issues/10342) | `Dot → elementwise → Dot` |
| [#10360](https://github.com/triton-lang/triton/pull/10360) | register tensor op → permute/reshape → consumer |
| [#11704](https://github.com/triton-lang/triton/pull/11704) | FMA → Store |
| [#10563](https://github.com/triton-lang/triton/pull/10563) | WMMA → Store |
| [#8775](https://github.com/triton-lang/triton/issues/8775) | Reduce → consumer，scratch layout 导致 redundant conversion |
| [#8500](https://github.com/triton-lang/triton/pull/8500) | Reduce wave layout |
| [#8706](https://github.com/triton-lang/triton/issues/8706) | memory coalescing ↔ layout propagation |
| [#10158](https://github.com/triton-lang/triton/pull/10158) | layout through control-flow producer/consumer |
| [#11685](https://github.com/triton-lang/triton/pull/11685) | automatic layout propagation |
| [#11538](https://github.com/triton-lang/triton/pull/11538) | reuse relayout/rematerialization |
| [#11885](https://github.com/triton-lang/triton/pull/11885) | reuse dot-operand rematerialization |
| [#11387](https://github.com/triton-lang/triton/pull/11387) | FP4 producer → strided consumer layout |
| [#11360](https://github.com/triton-lang/triton/pull/11360) | FP4 conversion allocation reduction |
| [#7625](https://github.com/triton-lang/triton/pull/7625) | transposed MFMA → DotOp |
| [#6045](https://github.com/triton-lang/triton/pull/6045) | MMA operand → TMA encoding |
| [#11967](https://github.com/triton-lang/triton/pull/11967) | masks influence coalesced-layout choice |

Triton candidate set不应该出现 NHD/HND，而应该是：

\[
\boxed{
\{
BlockedEncoding,
DotOperandEncoding,
MMA/WMMAEncoding,
SliceEncoding,
SharedEncoding,
SwizzledShared,
RegisterLayout
\}}
\]

并把 `convert_layout` strategy 也作为 candidate。

---

# 九、IREE：这是一个非常值得补进来的框架

IREE 2025–2026 其实有非常直接的 RQ1 证据。

最重要的是：

[IREE issue #22489](https://github.com/iree-org/iree/issues/22489)：

> incoming layout mismatches preferred layout 时，应能生成 local relayout。

这句话几乎就是 RQ1。

另外：

- [#22370](https://github.com/iree-org/iree/issues/22370)：dynamic layout decisions；
- [#21858](https://github.com/iree-org/iree/issues/21858)：有时宁可 duplicate operations，而不是引入 layout conflict；
- [#23782](https://github.com/iree-org/iree/issues/23782)：attention direct-to-LDS；
- [#22700](https://github.com/iree-org/iree/issues/22700)：target transpose-load instructions；
- [PR #24969](https://github.com/iree-org/iree/pull/24969)：target-aware layout materialization，把 layout 决策提前到 global graph，向周围 tensor graph 传播；
- [PR #21554](https://github.com/iree-org/iree/pull/21554)：CPU data-layout propagation；
- [PR #20410](https://github.com/iree-org/iree/pull/20410)：single-dispatch reshape propagation；
- [PR #20484](https://github.com/iree-org/iree/pull/20484)：`matmul_k` encoding；
- [PR #24528](https://github.com/iree-org/iree/pull/24528)：attention VectorDistribute layout constraints；
- [PR #24400](https://github.com/iree-org/iree/pull/24400)：multi-reduction source layout propagation。

IREE 很适合测试：

\[
Matmul\rightarrow Elementwise\rightarrow Matmul
\]

\[
Attention\rightarrow Projection
\]

\[
Conv\rightarrow Elementwise
\]

并比较：

\[
producer\ preferred\ encoding
\]

vs

\[
consumer\ preferred\ encoding
\]

vs

\[
propagated\ graph\ encoding.
\]

---

# 十、PyTorch / Inductor：也有非常好的 operator-level case

高相关的包括：

- [issue #197083](https://github.com/pytorch/pytorch/issues/197083) / [PR #197100](https://github.com/pytorch/pytorch/pull/197100)：non-contiguous view → matmul，本来可以 zero-copy，却由于 decomposition 做 layout copy。
- [issue #188635](https://github.com/pytorch/pytorch/issues/188635)：producer→view/transpose/contiguous→attention fusion missed。
- [issue #195320](https://github.com/pytorch/pytorch/issues/195320)：permuted K 在 SDPA fusion 中 layout semantics 出问题。
- [PR #198543](https://github.com/pytorch/pytorch/pull/198543)：FlexAttention GQA view 前冻结 query layout。
- [issue #190154](https://github.com/pytorch/pytorch/issues/190154)：non-contiguous SDPA 输入性能显著下降。
- [issue #184014](https://github.com/pytorch/pytorch/issues/184014)：Conv→AvgPool→Flatten 下 layout optimization 导致 stride mismatch。
- [PR #195063](https://github.com/pytorch/pytorch/pull/195063)：non-row-major fused inputs 的 reduction logical index。
- [PR #184051](https://github.com/pytorch/pytorch/pull/184051)：custom-op layout constraints。
- [issues #197869/#197870](https://github.com/pytorch/pytorch/issues/197870)：padding/layout decision 会受 consumer FX node context 影响。
- [issue #182328](https://github.com/pytorch/pytorch/issues/182328)：vLLM MoE custom op stride contract mismatch。

PyTorch 的 layout candidates 更适合写成：

\[
\{
contiguous,
transpose\ view,
permuted\ stride,
padded\ stride,
channels\_last,
consumer\_required
\}
\]

而不是命名物理 format。

---

# 十一、CUTLASS/CuTe：MMA → Epilogue → downstream operator

强相关项：

- [issue #3540](https://github.com/NVIDIA/cutlass/issues/3540)：ColumnMajor output transpose 与 bias consumer coordinate；
- [PR #3650](https://github.com/NVIDIA/cutlass/pull/3650)：Stream-K accumulation → broadcast epilogue；
- [PR #3345](https://github.com/NVIDIA/cutlass/pull/3345)：SM120 FP32 epilogue store layout；
- [PR #3313](https://github.com/NVIDIA/cutlass/pull/3313)：TMEM→register epilogue load atom；
- [PR #3660](https://github.com/NVIDIA/cutlass/pull/3660)：TiledMMA C-layout K slices；
- [PR #3273](https://github.com/NVIDIA/cutlass/pull/3273)：MXF4/NVFP4 native-TMA representation；
- [issue #2996](https://github.com/NVIDIA/cutlass/issues/2996)：`NCxHWx<32>` vs NHWC Conv representations；
- [PR #3656](https://github.com/NVIDIA/cutlass/pull/3656)：CopyAtom source/destination layout-size contract。

适合：

\[
GEMM/MMA
\rightarrow
Epilogue
\rightarrow
NextGEMM/Activation/Store
\]

候选：

\[
\{
TMEM-native,
RMEM-native,
RowMajor,
ColumnMajor,
Interleaved,
TMA-native
\}
\]

---

# 十二、TileLang：fragment → GEMM / Shared / Reduction

TileLang 更底层，但可以补 compiler-level variety：

- [issue #2642](https://github.com/tile-ai/tilelang/issues/2642)：`T.gemm` shared operand 使用 incompatible annotated layout；
- [issue #2954](https://github.com/tile-ai/tilelang/issues/2954)：GEMM operand fragment → shared copy；
- [issue #2311](https://github.com/tile-ai/tilelang/issues/2311)：GEMM accumulator → sibling fragment consumer；
- [issue #2609](https://github.com/tile-ai/tilelang/issues/2609)：swizzled shared buffer → cross-dtype view；
- [issue #2398](https://github.com/tile-ai/tilelang/issues/2398)：layout inference 无合法 layout；
- [issue #3283](https://github.com/tile-ai/tilelang/issues/3283)：两个 scale producers 共用 fragment 时 layout conflict；
- [PR #3284](https://github.com/tile-ai/tilelang/pull/3284)：显式 reject conflicting scale fragment layouts；
- [PR #3176](https://github.com/tile-ai/tilelang/pull/3176)：default vector size 根据 reduction consumer 调整；
- [issue #3256](https://github.com/tile-ai/tilelang/issues/3256)：register-resident attention 的 block-scaled GEMM；
- [PR #3257](https://github.com/tile-ai/tilelang/pull/3257)：SM120 block-scaled GEMM fragment layouts。

---

# 十三、还有几个很值得作为“非 Attention”对照组

为了避免 reviewer 认为 RQ1 只是 KV/cache 问题，我建议至少加入：

| Case | Producer | Consumer |
|---|---|---|
| Triton Dot-Dot | GEMM1 | GEMM2 |
| SGLang BMM-o_proj | attention value BMM | output projection |
| MoE | Dispatch | Grouped GEMM |
| FlashInfer MoE | GEMM2 | cross-GPU combine |
| PyTorch Conv chain | Conv | AvgPool/Flatten |
| CUTLASS | MMA | epilogue/store |
| IREE | contraction | elementwise/next contraction |
| TileLang | GEMM fragment | shared/reduction |

这样你的实验结论才能跨：

\[
Attention,\ GEMM,\ MoE,\ Reduction,\ Conv,\ Communication,\ RecurrentState.
\]

---

# 十四、我建议最终的 RQ1 candidate-layout registry

不要写一个全局：

\[
\{NHD,HND\}
\]

而应该按 producer-consumer family 建 registry：

| Family | Candidate representation |
|---|---|
| Dense/paged attention | `NHD,HND,paged_NHD,paged_HND,LBNHC,LBHNC,BLNHC,BLHNC` |
| QKV projection | packed-QKV, split-strided Q/K/V, dense Q/K/V |
| MLA | latent-native, separate K/V, packed K/V, FP8/NVFP4 canonical |
| Sparse Attention | full pool NHD, paged HND, compact gathered HND |
| MoE | Standard, PaddedStandard, ExpandedExpertMajor, BatchedExperts, psum |
| GEMM chain | producer accumulator, RowMajor, ColumnMajor, transposed, interleaved |
| Triton | Blocked, DotOperand, MMA/WMMA, Slice, Shared/SwizzledShared |
| CUTLASS | TMEM, RMEM, TMA-native, Row/ColumnMajor, interleaved |
| KDA/GDN/Mamba | checkpoint-native, packed-decode, strided-state, dense-state |
| Communication | producer-native, canonical-wire, compressed-wire, consumer-native |
| PyTorch | contiguous, transpose-view, permuted-stride, padded-stride, channels-last |
| Vision | NCHW, NHWC, channels-last, packed/interleaved |

然后真正搜索：

\[
(P,L_P,R,L_C,C)
\]

而不是只搜索 \(L\)。

---

# 十五、如果要控制实验量，我建议分成 3 个层级

核心集最好不是所有 PR 各做一个 benchmark，而是从上面抽象成 edge families。

| 粒度 | 建议核心 edge |
|---|---|
| Operator | `BMM→o_proj`, `GEMM→GEMM`, `Dispatch→GroupedGEMM`, `RoPE→Attention` |
| Subgraph | `QKV+Norm+RoPE+KVUpdate→Attention`, `SparseKVPrep→SparseAttention`, `Prefill→KDA Decode` |
| Stage | `Prefill→Decode`, `Draft→Target`, `KV Offload→Attention`, `MoE GEMM2→cross-GPU Combine` |

每一条都统一执行：

\[
T_P(L_P)
\]

\[
T_C(L_C)
\]

\[
T_R(L_P,L_C)
\]

以及最重要的：

\[
\boxed{
T_E(L_P,R,L_C)
=
\operatorname{time}
[
P(L_P)\rightarrow R\rightarrow C(L_C)
]
}
\]

最后分别得到：

\[
L_P^*,
\qquad
L_C^*,
\qquad
(L_P,R,L_C)^*_E
\]

这样你就能严格区分：

\[
L_P^*\neq L_C^*
\]

只是 preference conflict；

而：

\[
\boxed{
L_P^*\neq L_{P,E}^*
}
\]

才是真正的 RQ1 rank inversion。

目前 v13 CUDA 最大的问题之一就是把这两者靠得太近了。

如果从论文实验设计角度排序，我会优先实现：`SGLang BMM→o_proj`、`DeepEP Dispatch→GroupedGEMM`、`SGLang NHD pool→MSA sparse prefill`、`Triton Dot→Dot`、`Triton WMMA→Store`、`FlashInfer packed QKV→KDA Prefill`、`KDA Prefill→Decode`、`Prefill→Decode transfer`、`vLLM QKNorm/RoPE/KV-update→Attention`。这九类已经能把当前 RQ1 从单一 KV microbenchmark 扩展成跨 operator、subgraph 和 serving-stage 的系统性实验。

---

有，而且这类 case 其实比当前 `KV write → head-local scan` 更适合作为 RQ1 的主验证。因为两端都可以直接对应框架中的真实算子/子图，而不是研究者自己构造的 memory micro-kernel。

我建议把它们分成三档：算子→算子、子图→算子、阶段→阶段。

第一类最干净的是算子→算子。

| 框架 / PR | Producer | Consumer | 中间 representation/layout |
|---|---|---|---|
| Triton #10342 | 第一个 `tl.dot` | 第二个 `tl.dot` | MMA accumulator / FFMA-compatible / DotOperand layout |
| SGLang PR #37261 | DeepEP dispatch | expert Grouped GEMM | token-major / expanded expert-major / padded expert rows |
| vLLM PR #54109 | MoE Prepare/Dispatch | FusedMoE expert kernel | `Standard` / `PaddedStandard` / `BatchedExperts` |
| SGLang PR #34299 | KDA prefill | KDA decode | native checkpoint state / packed decode state |
| vLLM/SGLang RoPE fusion PRs | RoPE/QK-Norm operator | Attention | NHD/HND/paged KV representation |

其中我认为最值得加入 RQ1 的是下面几个。

一，`MoE Dispatch → Grouped GEMM`。

SGLang PR #37261 是非常好的真实例子。它讨论的已经不是某个 load/store，而是完整的：

\[
\boxed{
\text{MoE Dispatch}
\rightarrow
\text{expert activation representation}
\rightarrow
\text{Grouped GEMM}
}
\]

DeepEP dispatch 原来可以产生一种普通非-expanded representation，然后 SGLang 再运行：

```text
ep_scatter
```

把它整理成 grouped GEMM 喜欢的 representation。

PR 改成让 DeepEP dispatch 自己直接产生：

\[
\text{expanded expert-major rows}
\]

也就是大致：

```text
原方案

tokens
  │
  ▼
DeepEP dispatch
  │
  ▼
non-expanded representation
  │
  ▼
ep_scatter
  │
  ▼
expert-grouped rows
  │
  ▼
Grouped GEMM
```

改成：

```text
tokens
  │
  ▼
DeepEP dispatch
  │
  │ 直接按 (token, expert)
  │ 并按 expert 分组
  ▼
expanded expert-major representation
  │
  ▼
Grouped GEMM
```

这就是非常标准的 RQ1：

\[
L_{\text{Dispatch-best}}
\stackrel{?}{=}
L_{\text{GroupedGEMM-best}}
\]

这里甚至不应该再称作简单 tensor layout，而应该称：

\[
\boxed{\text{activation representation}}
\]

建议候选集合至少有：

\[
\mathcal L=
\{
\text{token-major Standard},
\text{PaddedStandard},
\text{expanded expert-major},
\text{BatchedExperts}
\}
\]

然后直接测：

\[
T_{\text{dispatch}}(L)
\]

\[
T_{\text{expert-GEMM}}(L)
\]

以及：

\[
T_{\text{dispatch}\rightarrow\text{GEMM}}(L)
\]

如果需要 `scatter/permute`：

\[
T_{\text{edge}}
=
T_D(L_D)
+
T_{\text{permute}}
+
T_G(L_G)
\]

这比当前 NHD/HND KV scan 更接近“真正的算子间 layout 决策”。

vLLM PR #54109 也有类似证据。它专门引入：

```text
Standard
PaddedStandard
BatchedExperts
```

来描述 Prepare/Finalize 给 expert kernel 的 activation contract。

所以可以直接构造：

\[
\boxed{
\text{MoE Prepare/Dispatch}
\rightarrow
\text{FusedMoE Expert}
}
\]

这应该是 RQ1 的一个独立 producer-consumer family。

---

第二个非常好的例子是：

\[
\boxed{
\text{Dot/GEMM}
\rightarrow
\text{Dot/GEMM}
}
\]

Triton issue #10342 的标题本身就是：

> Layout convert inserted between two `tl.dot`s when an FFMA bias is fused on the accumulator

这几乎就是一个现成的 RQ1。

计算图可以看成：

```text
GEMM / tl.dot #1
        │
        ▼
 accumulator
        │
        ▼
    FFMA / bias
        │
   convert_layout ?
        │
        ▼
GEMM / tl.dot #2
```

Producer：

\[
P=\text{GEMM}_1
\]

Consumer：

\[
C=\text{GEMM}_2
\]

中间还有 elementwise bias，但你完全可以把它归到 edge adaptation。

第一 GEMM 的输出天然处于某个：

\[
L_{\text{acc}}
\]

第二 GEMM 的 operand 又要求：

\[
L_{\text{dot-op}}
\]

于是实验就可以直接比较：

\[
\text{GEMM}_1(L_1)
\rightarrow
convert
\rightarrow
\text{GEMM}_2
\]

和：

\[
\text{GEMM}_1
\text{直接生成下一 GEMM 可接受的 layout}
\rightarrow
\text{GEMM}_2
\]

candidate 不应该用 NHD/HND，而应该是：

\[
\{
\text{MMA accumulator layout},
\text{DotOperandEncoding},
\text{BlockedEncoding},
\text{shared-memory staged layout}
\}
\]

这个 case 的科研价值很高，因为它证明 RQ1 不是 attention/KV 专属问题。

---

第三个很适合的是：

\[
\boxed{
\text{RoPE/QK-Norm}
\rightarrow
\text{Attention}
}
\]

而且 2026 年这方面有大量真实 PR。

例如 vLLM PR #52901：

> Add AITER gated QKV + RoPE + kv-cache compile fusion

它融合的是：

```text
gated QKV split
      ↓
Q/K RMSNorm
      ↓
RoPE
      ↓
gate extraction
      ↓
paged KV-cache update
      ↓
Attention
```

而且 PR 明确支持：

\[
NHD/HND
\]

两种 KV-cache layout。

所以可以把 producer 提高到：

\[
\boxed{
P=
\text{QKV split + QK-Norm + RoPE + KV update}
}
\]

consumer：

\[
\boxed{
C=\text{Attention}
}
\]

这样 producer 本身已经是一个真正有算术计算的子图，而不是单纯 `kv_write_kernel()`。

candidate 可以是：

\[
\{
NHD,
HND,
paged\_NHD,
paged\_HND
\}
\]

再加 strategy 维度：

```text
unfused producer → cache
fused producer → NHD cache
fused producer → HND cache
producer NHD → convert → HND consumer
producer HND → convert → NHD consumer
```

这就是非常理想的 operator/subgraph-level RQ1。

SGLang PR #41698 也类似：

> Fuse AITER DSA decode RoPE and KV-cache preparation

原来：

```text
RoPE
  ↓
KV preparation
  ↓
DSA Attention
```

现在可以：

```text
fused RoPE + KV preparation
       ↓
DSA Attention
```

所以可以研究：

\[
L_{\text{RoPE/KV-prep-best}}
\stackrel{?}{=}
L_{\text{DSA-attention-best}}
\]

这个比 synthetic head scan 强很多。

---

第四类是：

\[
\boxed{
\text{Prefill operator}
\rightarrow
\text{Decode operator}
}
\]

这是比单算子还粗一级，而且特别适合 persistent state。

SGLang PR #34299：

> Add zero-copy native prefill checkpoints and packed decode

这里已经明确出现：

```text
KDA Prefill
    │
    ▼
persistent recurrent state
    │
    ▼
KDA Decode
```

prefill 会产生 checkpoint/state。

decode 又希望一种 packed representation。

因此真正的问题就是：

\[
\boxed{
L_{\text{prefill-output-best}}
\stackrel{?}{=}
L_{\text{decode-input-best}}
}
\]

candidate 可以设计成：

\[
\mathcal L=
\{
\text{prefill-native checkpoint},
\text{decode-packed state},
\text{common canonical state}
\}
\]

然后三条典型路径：

```text
A. prefill-native
   → decode direct

B. prefill-native
   → pack/convert once
   → decode-packed

C. prefill directly emits decode-packed
   → decode
```

这本质上就是 v10 的：

```text
NN
NH-copy
HH-native
```

但这里 producer 和 consumer 都已经是真实的大算子：

\[
\boxed{
KDA\ Prefill
\rightarrow
KDA\ Decode
}
\]

这会是非常强的 RQ1 case。

---

再粗一层甚至可以做：

\[
\boxed{
\text{Prefill stage}
\rightarrow
\text{Decode stage}
}
\]

比如 SGLang PR #41958：

> Support unified PD between matching DCP peers

其实际结构是：

```text
Prefill worker
     │
     │ KV pages +
     │ recurrent states
     ▼
PD transfer
     │
     ▼
Decode worker
```

这里 producer 已经不是一个 kernel，而是整个：

\[
\boxed{\text{Prefill stage}}
\]

consumer 是：

\[
\boxed{\text{Decode stage}}
\]

PR 的关键观察是，当 prefill 和 decode 的：

\[
TP/DCP/layout
\]

完全匹配时，可以：

\[
\text{page}\rightarrow\text{page}
\]

直接 copy，不需要 redistribution 或 packing。

所以这里可以研究：

\[
L_{\text{prefill-stage-best}}
\stackrel{?}{=}
L_{\text{transfer-best}}
\stackrel{?}{=}
L_{\text{decode-stage-best}}
\]

candidate 可以是：

\[
\{
\text{prefill-local layout},
\text{canonical transfer layout},
\text{decode-local layout}
\}
\]

策略：

```text
Prefill native
→ direct page transfer
→ Decode native
```

vs

```text
Prefill native
→ pack/canonicalize
→ transfer
→ unpack
→ Decode native
```

vs

```text
Prefill directly maintains transfer/decode-compatible layout
→ transfer
→ Decode
```

这其实已经是“system-level RQ1”。

---

还有一个值得加入的是：

\[
\boxed{
\text{Shared-expert MLP}
\rightarrow
\text{Routed-expert/MegaMoE}
}
\]

但这里更准确地说是两个 parallel producer/consumer branches 的 representation/fusion问题。

SGLang PR #39313 明确说 DeepSeek-V4 每个 token 有：

1. shared dense MLP；
2. routed grouped GEMM。

原来两个独立 GEMM 路径。

后来尝试把 shared expert 融进 MegaMoE。

问题是 shared expert 是：

\[
FP8
\]

routed experts 是：

\[
MXFP4
\]

所以：

\[
\text{dtype/layout representation}
\]

并不天然相同。

这个 case 很适合研究：

\[
\boxed{
\text{是否为了下游统一执行而改变 upstream representation}
}
\]

但它更偏向 RQ2/RQ8 一类 joint representation，而不是最纯粹的一对一 RQ1。所以我会把它作为扩展 case，而不是 RQ1 主 case。

---

因此我建议新的 RQ1 不要只有一个统一粒度。

可以明确规定三个层级。

第一级：Operator Edge

\[
\boxed{
Op_P
\rightarrow
Tensor/State
\rightarrow
Op_C
}
\]

最推荐：

- `GEMM → GEMM`
- `MoE Dispatch → Grouped GEMM`
- `RoPE/QK-Norm → Attention`
- `Attention/Projection → KV Update`
- `Reduction → GEMM`

第二级：Subgraph Edge

\[
\boxed{
Subgraph_P
\rightarrow
persistent/intermediate representation
\rightarrow
Subgraph_C
}
\]

例如：

- `QKV+Norm+RoPE+KV-update → Attention`
- `KDA Prefill → KDA Decode`
- `Sparse Prefill preparation → Sparse Attention`
- `MoE Prepare/Dispatch → Expert MLP`

第三级：Stage Edge

\[
\boxed{
Stage_P
\rightarrow
transfer/persistent state
\rightarrow
Stage_C
}
\]

例如：

- `Prefill → Decode`
- `Draft model → Target verify`
- `GPU KV → CPU/NVMe offload → GPU Attention`
- `TP/EP producer rank → consumer rank`

这样 RQ1 的外部有效性会强很多。

如果让我挑一组“论文核心 case”，我会用这 6 个：

| Family | Producer | Consumer | Layout/representation candidates |
|---|---|---|---|
| Attention persistent state | QK-Norm+RoPE+KV update | Attention | NHD/HND/paged variants |
| Dense compute | GEMM1 | GEMM2 | accumulator / DotOperand / Blocked |
| MoE | Dispatch | Grouped GEMM | Standard / Padded / expert-major |
| Sparse attention | KV preparation/gather | Sparse Attention | NHD / compact paged-HND |
| Linear attention | Prefill | Decode | checkpoint-native / packed state |
| Serving | Prefill stage | Decode stage | local / canonical-transfer / decode-native |

这 6 类已经足以回答一个比现有 RQ1 强得多的问题：

\[
\boxed{
\text{跨算子甚至跨阶段的 producer-local representation preference，
是否系统性地不同于 downstream edge-optimal representation？}
}
\]

而不是现在较窄的：

\[
\boxed{
\text{KV writer 与 head-local scan 是否喜欢不同的 NHD/HND}
}
\]

从科研设计上，我更建议把 primitive benchmark 降级为 mechanism control，把上述 operator/subgraph/stage cases 提升成 RQ1 的主要 external-validity experiments。

---

是的。对于 RQ1，如果目标是证明“真实框架中的 producer-local layout preference 会与 downstream consumer preference 冲突，并最终改变 edge-optimal layout”，那么现在的 v10/v13 设计还不够。它目前过度集中在：

\[
\text{KV write}\rightarrow \text{head-local KV scan}
\]

而真实 2025–2026 框架已经暴露出至少几类完全不同的 producer→consumer 冲突。更合理的 RQ1 应当把这些真实边纳入，而不是继续单纯增加 KV shape 数量。

截至 2026-10-01，我认为下面这套矩阵最值得补。

| 真实来源 | 应测试的 producer → consumer | 为什么确实存在 layout tension | 建议候选 layout / strategy |
|---|---|---|---|
| vLLM [#55312](https://github.com/vllm-project/vllm/issues/55312) | KV cache writer → speculative draft/target 两种 attention backend | draft 与 target 可以选择支持集合不相交的 backend；当前 `supported_kv_cache_layouts()` 已显式表达 consumer contract | `LBNHC`, `LBHNC`, `BLNHC`, `BLHNC`；shared direct、per-consumer copy、producer-native split |
| vLLM [PR #58380](https://github.com/vllm-project/vllm/pull/58380)（2026-09，当前 open） | KV allocator/writer → attention kernels | layer-outermost 与 block-outermost layout 对 packing/padding 和 kernel consumption 的利弊不同 | `LBNHC/LBHNC` vs `BLNHC/BLHNC` |
| vLLM [PR #52363](https://github.com/vllm-project/vllm/pull/52363)（open） | RoPE + K/V producer → paged KV → FlashAttention | 新实现直接把 rotary K/V 写进 cache；producer 不再只是 memcpy/write，而是带实际计算的 fused producer | 当前 vLLM 合法 `KVCacheLayout` 集合；普通 RoPE→cache、fused direct-emission、materialize |
| SGLang [PR #34525](https://github.com/sgl-project/sglang/pull/34525)（open） | NHD persistent KV pool → MSA sparse prefill | SGLang pool 是 slot-major NHD，但 `fmha_sm100` sparse prefill 消费 paged HND；`Hkv>1` 时 view 不连续，原来会整个 pool `.contiguous()` | `NHD pool`; whole-pool HND copy; compact batch `paged_HND`; native HND |
| SGLang [PR #32272](https://github.com/sgl-project/sglang/pull/32272)（open） | KV pool → TRTLLM fmha_v2 prefill | 以前为了 consumer API 要做 interleaved KV copy；新 indexing 可以直接读 non-interleaved paged KV | separate/non-interleaved paged K/V；interleaved KV；native producer→consumer direct |
| SGLang [PR #31652](https://github.com/sgl-project/sglang/pull/31652)（open） | quantization producer → KV-store consumer → attention | FP8 quant + store 原来 5 launches，融合后只对 NHD fast path 最自然，HND/page-major 等保留 fallback | `NHD`, `HND`, page-major/vectorized 5D；quant-then-store vs fused quant+store |
| SGLang [PR #34299](https://github.com/sgl-project/sglang/pull/34299)（open） | linear-attention/KDA state producer → prefill/decode consumer | prefill checkpoint 与 packed decode representation 不同，已经有 zero-copy/native checkpoint 与 packed decode 两种路径 | native checkpoint layout；packed-decode layout；convert-once；direct-native |
| vLLM [#59111](https://github.com/vllm-project/vllm/issues/59111) | standard KV producer / NIXL transfer → ROCm AITER decode | transfer/canonical representation 与 decode-side shuffled representation不同 | standard LBNHC/LBHNC；AITER shuffled; transfer-canonical→decode conversion |
| FlashInfer [PR #5402](https://github.com/flashinfer-ai/flashinfer/pull/5402)（merged） | NVFP4 paged-KV producer → MSA sparse decode | PR 专门定义 canonical planar page contract：K data/K scale/V data/V scale 的布局必须被 producer 与 reader共同遵守 | generic KV+scale representation；canonical MSA planar representation；producer-native canonical |
| FlashInfer [PR #5329](https://github.com/flashinfer-ai/flashinfer/pull/5329)（merged） | KV producer → persistent BatchAttention | equal K/V stride 与 unequal stride 走不同 specialization，说明 consumer 不只对“layout name”敏感，还对真实 stride contract 敏感 | equal-stride NHD、unequal-stride NHD；合法 stride variants |
| Triton [PR #8450](https://github.com/triton-lang/triton/pull/8450)（2025，closed/unmerged） | `tt.load` → `tt.dot` | global load 希望 coalesced mapping，而 DotOp 要 `DotOperandEncoding`; 原实现需要 shared-memory `convert_layout` | `BlockedEncoding`; `DotOperandEncoding`; shared staging；direct GMEM→dot-compatible registers |
| Triton [PR #11704](https://github.com/triton-lang/triton/pull/11704)（2026，open） | FMA/MMA accumulator → `tt.store` | accumulator 原生 layout 与 coalesced store layout 冲突；当前实现用 heuristic 判断是否值得绕过 relayout | accumulator-native；coalesced `BlockedEncoding`; shared/LDS relayout |
| Triton [#10342](https://github.com/triton-lang/triton/issues/10342) | `tt.dot` → FFMA/bias → second `tt.dot` | 中间 elementwise op 会导致两个 dot 之间插入 `convert_layout` | first-dot output layout；elementwise-compatible layout；second-dot operand layout |
| CUTLASS [PR #3650](https://github.com/NVIDIA/cutlass/pull/3650)（2026，open） | MMA accumulator → epilogue/broadcast → output | RowMajor/ColumnMajor 会改变 epilogue/broadcast coordinate contract；不是纯粹名字变化 | RowMajor, ColumnMajor；TMEM/RMEM accumulator-native；epilogue-native output |
| CUTLASS [#2996](https://github.com/NVIDIA/cutlass/issues/2996) | Conv producer → next consumer | `NCxHWx<32>`、NHWC 等 interleaved layout 的性能并不由单 kernel 简单决定 | NCHW, NHWC, `NCxHWx<32>`，只保留 consumer 合法集合 |

这几类比当前的 `KV write → head-local scan` 更有价值，因为 producer 本身开始出现真正不同的计算：GEMM/RoPE、quantization、gather-transpose、MMA、reduction、offload/transfer、state update，而 consumer 也不再只是一个 reduction-style scan。

最典型的新增 case，我认为是 SGLang PR #34525。它几乎就是一个现实版 RQ1。

原始数据流是：

```text
               Producer side
                    │
                    ▼
        persistent MHA KV pool
          [slot, Hkv, D]
               NHD-like
                    │
                    │
          ┌─────────┴───────────┐
          │                     │
        Hkv=1                 Hkv>1
          │                     │
          ▼                     ▼
 permute is contiguous    permute non-contiguous
          │                     │
          │                 .contiguous()
          │                     │
          ▼                     ▼
     zero-copy HND       whole-pool HND copy
          │                     │
          └─────────┬───────────┘
                    ▼
          MSA sparse prefill
     expects [page,Hkv,N,D]
                 HND
```

这里 producer 的 natural persistent representation 是：

\[
L_P=NHD
\]

但是 consumer 的 native representation 是：

\[
L_C=paged\_HND
\]

而且真实 PR 已经说明，当 \(H_{kv}>1\) 时，这不是一个免费的 view；会触发 materialization。于是 RQ1 应该真正比较：

\[
T_{\text{pool-write}}(NHD)
+
T_{\text{MSA}}(NHD\text{-view/copy})
\]

与：

\[
T_{\text{pool-write}}(HND)
+
T_{\text{MSA}}(HND)
\]

以及 PR 提出的第三种：

\[
T_{\text{pool-write}}(NHD)
+
T_{\text{gather-transpose selected pages}}
+
T_{\text{MSA}}(compact\ HND)
\]

这就比当前 `NHD→HND full transpose` 丰富得多，因为 repair 本身也拥有多个候选。

---

更重要的是，RQ1 不应该再把 layout candidate 固定成一个全局的：

\[
\{NHD,HND\}
\]

集合。正确做法应该是对每一条 edge 定义三个集合：

\[
\mathcal L_P=
\{\text{producer 能直接生成的 layouts}\}
\]

\[
\mathcal L_C=
\{\text{consumer 能合法直接消费的 layouts}\}
\]

以及：

\[
\mathcal R=
\{\text{两者之间合法的 repair/view/fused transforms}\}
\]

真正的搜索空间应该是：

\[
\boxed{
(P,L_P)
\rightarrow
R(L_P,L_C)
\rightarrow
(C,L_C)
}
\]

然后求：

\[
(L_P^*,R^*,L_C^*)
=
\arg\min
\left[
T_P(L_P)
+
T_R(L_P,L_C)
+
T_C(L_C)
\right]
\]

而不是只问：

\[
NHD\text{ or }HND?
\]

这点非常关键。

例如 vLLM speculative decoding 的真实 candidate space 应该来自 backend contract，而不是研究者自己编几个名字。当前 vLLM 已经有：

```python
supported_kv_cache_layouts()
```

那么测试 harness 应该直接读取这些集合。

假设：

```text
FlexAttention:
    {LBNHC}

FlashInfer:
    {LBHNC, BLHNC}
```

那么：

\[
\mathcal L_{\text{Flex}}
=
\{LBNHC\}
\]

\[
\mathcal L_{\text{FI}}
=
\{LBHNC,BLHNC\}
\]

如果 draft 和 target 同时需要：

\[
\mathcal L_{\text{draft}}
\cap
\mathcal L_{\text{target}}
=
\varnothing
\]

这时正确 RQ1 candidate 不应该强制一个共同 layout，而应该比较：

```text
shared single layout       → illegal

LBNHC
 → conversion
 → LBHNC                   → legal repair

producer directly keeps
two representations        → legal split

producer-native LBHNC
 → convert only for Flex   → legal repair
```

这样 RQ1 才真正研究“edge selection”，而不是“两个 layout 名字的 benchmark”。

---

针对 producer variety，我建议从现在只有的 memory writer 扩展成下面几种计算类型，但在实验实现上仍统一一个 edge contract：

| Producer class | 真实例子 | 为什么重要 |
|---|---|---|
| Pure store/scatter | KV cache writer | 当前 RQ1 已覆盖 |
| GEMM producer | K/V projection、QKV projection、MoE GEMM | 输出 layout 可能由 GEMM epilogue 决定 |
| Fused compute+store | RoPE+KV store、quantize+KV store | vLLM #52363、SGLang #31652 已真实出现 |
| Gather/transpose producer | sparse prefill batch-KV gather | SGLang #34525 |
| Reduction/MMA producer | Triton `tt.dot`, CUTLASS MMA accumulator | internal register/thread layout 与下游不同 |
| Persistent-state update | Mamba/KDA/linear-attention recurrent state | 不应该把 RQ1 限定在 Attention KV |
| Communication/offload producer | NIXL/HiCache/PD disaggregation | wire/host layout 与 decode layout可能不同 |
| Routing/scatter producer | MoE token→expert dispatch | expert-major layout可能利于 grouped GEMM，但不利于上游 token-major |

相应地，consumer variety 也应该包含：

\[
\text{decode attention}
\]

之外的：

\[
\text{prefill attention},
\quad
\text{sparse attention},
\quad
\text{GEMM/MMA},
\quad
\text{reduction},
\quad
\text{global store},
\quad
\text{MoE grouped GEMM},
\quad
\text{linear-attention state reader},
\quad
\text{KV transfer/offload}.
\]

这样才能测试 RQ1 到底是一个 Attention 特例，还是更普遍的 producer-consumer layout incompatibility。

---

对应的 layout space 我建议分层，而不是把所有 layout 混在一个枚举集合中。

对于 persistent/global-memory edge，Attention 类可以统一成：

\[
\mathcal L_{\text{attn}}
=
\{
NHD,\ HND,\ paged\_NHD,\ paged\_HND,
LBNHC,\ LBHNC,\ BLNHC,\ BLHNC
\}
\]

但每个 framework/case 必须先做 legality filtering，不能八个全部硬跑。

其中例如：

\[
LBNHC=[Layer,Block,Token,Head,Channel]
\]

\[
LBHNC=[Layer,Block,Head,Token,Channel]
\]

\[
BLNHC=[Block,Layer,Token,Head,Channel]
\]

\[
BLHNC=[Block,Layer,Head,Token,Channel]
\]

这样既能研究：

\[
N\leftrightarrow H
\]

也能研究：

\[
Layer\leftrightarrow Block
\]

这正好对应 vLLM PR #58380 暴露出来的新 trade-off。

对于 Triton 这种 kernel-internal edge，就绝对不应该继续用 NHD/HND，而应使用：

\[
\mathcal L_{\text{Triton}}
=
\{
Blocked,
DotOperand,
MMA/Linear,
Slice,
SwizzledShared
\}
\]

例如真正测试：

```text
tt.load
  │
  ├─ Blocked
  │    ↓ convert_layout
  │  DotOperand
  │
  └─ direct DotOperand-compatible load
          ↓
        tt.dot
```

和：

```text
tt.dot / FMA
      │
 accumulator-native
      │
      ├──── direct store
      │
      └──── convert_layout
                ↓
          coalesced Blocked
                ↓
             tt.store
```

PR #8450 和 #11704 正是在分别处理这两个方向。

对于 CUTLASS，layout 集合又应该不同：

\[
\mathcal L_{\text{CUTLASS boundary}}
=
\{
RowMajor,
ColumnMajor,
Interleaved,
\text{TMEM/RMEM native fragment},
\text{epilogue store layout}
\}
\]

内部 fragment layout 不一定能直接作为 global tensor layout，所以测试的是“strategy”而不是把它们都当作同一种 layout name：

\[
MMA\ native
\rightarrow
epilogue\ relayout
\rightarrow
RowMajor
\]

与：

\[
MMA\ native
\rightarrow
alternate\ epilogue
\rightarrow
ColumnMajor
\]

再让下一个 GEMM/consumer 使用它。

---

我建议把新的 RQ1 实验单元定义成：

\[
\boxed{
\text{EdgeCase}
=
(P,C,\mathcal L_P,\mathcal L_C,\mathcal R,\text{shape})
}
\]

而不是现在的：

\[
\text{Case}
=
(B,Q,KV,H_q,H_{kv},D)
\]

一个合格的 RQ1 case 至少要直接记录：

\[
T_P(L_P)
\]

\[
T_C(L_C)
\]

\[
T_R(L_P,L_C)
\]

以及最关键的：

\[
\boxed{
T_{\text{edge}}(L_P,R,L_C)
}
\]

而且 `T_edge` 必须真正执行 pipeline，不能只把前三个 median 相加。

随后同时报告三个 winner：

\[
L_P^*=\arg\min T_P
\]

\[
L_C^*=\arg\min T_C
\]

和：

\[
(L_P,R,L_C)^*=\arg\min T_{\text{edge}}
\]

这样就能区分三种非常不同的现象：

\[
L_P^*\neq L_C^*
\]

只说明 preference conflict；

\[
L_P^*\neq L_{\text{edge producer}}^*
\]

才是真正的 producer-local rank inversion；

而：

\[
L_P^*=L_{\text{edge producer}}^*,
\quad
L_C^*\neq L_{\text{edge consumer}}^*
\]

则说明 conversion 太贵，虽然 consumer 单独喜欢另一种 layout，但全 edge 不值得迁移。

这三个现象现在的 v13 CUDA 分析实际上混在了一起。

---

如果把实验规模控制在论文可执行范围内，我不会直接做“所有框架 × 所有 producer × 所有 consumer”。更合理的核心 RQ1 suite 是大约 8–10 类真实 edge：

\[
\boxed{
\begin{aligned}
&KVWrite\rightarrow DenseDecode\\
&KVWrite\rightarrow Prefill\\
&NHDPool\rightarrow SparsePrefill\\
&RoPE/Quant\rightarrow KVStore\rightarrow Attention\\
&KVTransfer\rightarrow Decode\\
&Draft/Target\ multi\text{-}consumer\\
&Load\rightarrow Dot\\
&Dot/FMA\rightarrow Store\\
&GEMM/Epilogue\rightarrow GEMM/Activation\\
&StateUpdate\rightarrow LinearAttentionDecode
\end{aligned}
}
\]

这套 variety 已经覆盖了 global persistent layout、zero-copy view、materialization、fused producer、register/thread layout、shared-memory relayout、communication layout 和 multi-consumer conflict。

因此，RQ1 最值得修改的地方不是把 256 cases 增加到 1024 cases，而是把“case 的维度”从只有 shape：

\[
(B,Q,KV,H_q,H_{kv},D)
\]

升级成：

\[
\boxed{
(\text{producer type},
\text{consumer type},
\text{representation level},
\mathcal L_P,
\mathcal L_C,
\mathcal R,
\text{shape})
}
\]

这样得到的结论才可以真正写成：

> “Cross-edge layout rank inversion occurs across multiple real producer–consumer classes and representation levels.”

而不是目前更窄的：

> “A synthetic KV writer and a head-local KV reader can prefer different layouts.”

这会从根本上提升 RQ1 的外部有效性和科研价值。

---

不完全一样。这里要区分 v13 的 `L-RQ1` 和仓库里较早 canonical `RQ1` 的编号。你现在问的 v13 `L-RQ1 = Cross-edge layout rank inversion`，它的 CUDA 证据确实仍然围绕 KV write 和 head-local KV read，但和 v10 有两个关键差别。

第一，v13 的核心 CUDA consumer 不是 paged KV scan，而是普通 flat NHD/HND scan。源码中 producer 是：

```cpp
kv_write_kernel(...)
```

它把逻辑上的 `[token, head, dim]` KV 写成：

\[
NHD=[N,H,D]
\]

或：

\[
HND=[H,N,D]
\]

consumer 是：

```cpp
decode_head_scan_kernel(...)
```

它固定一个 head，沿：

\[
[token,dim]
\]

扫描。所以 v13 H1.1 的基本结构是：

\[
\boxed{
KV\ write
\rightarrow
NHD/HND\ global\ memory
\rightarrow
head\text{-}local\ scan
}
\]

但这里没有 page table，也没有：

\[
page,\ within\_page
\]

这样的寻址。`decode_head_scan_kernel()` 使用的是：

```cpp
offset_of(layout, token, head, d, ...)
```

也就是 flat NHD/HND。

相比之下，v10 的 `v10_rq1_edge_bench.cu` 用的是：

```cpp
cache_offset(
    layout,
    b,
    token,
    head,
    d,
    pages,
    page_size,
    ...
)
```

其物理布局实际上是：

\[
[B,Page,N,H,D]
\]

或：

\[
[B,Page,H,N,D]
\]

所以 v10 才是真正的 paged-KV RQ1 benchmark。

---

第二，而且这是更重要的区别：**v13 CUDA 的 H1.1 实际上没有像 v10 那样直接计时完整 producer→consumer edge。**

v13 `analyze_v13_rqs.py` 对 CUDA H1.1 做的是：

先从：

```text
benchmark = kv_write
```

找：

\[
L_P^*
=
\arg\min_L T_{\text{write}}(L)
\]

然后从：

```text
benchmark = decode_head_scan
```

找：

\[
L_C^*
=
\arg\min_L T_{\text{head-scan}}(L)
\]

如果：

\[
L_P^*\neq L_C^*
\]

再计算：

\[
\frac{
T_C(L_P^*)
}{
T_C(L_C^*)
}
\]

如果 ≥1.03，就记成 `local_inversion`。

也就是说，CUDA 部分实际上是在检查：

```text
KV writer:
    NHD vs HND
        ↓
producer winner

head-local consumer:
    NHD vs HND
        ↓
consumer winner

producer winner != consumer winner ?
```

而不是 v10 那种直接测：

```text
NN
NH-copy
HH-native
HN-copy
NH-view
```

然后：

\[
\arg\min T_{\text{whole edge}}
\]

---

所以 v10 和 v13 的区别可以非常清楚地表示成：

| | v10 L-RQ1 | v13 L-RQ1 CUDA |
|---|---|---|
| Producer | KV append/write | KV write |
| Consumer | head-local KV scan | head-local KV scan |
| KV storage | **paged NHD/HND** | **flat NHD/HND** |
| Page table | 有 page/within-page | 无 |
| Producer 单独计时 | 有 | 有 |
| Consumer 单独计时 | 有 | 有 |
| Conversion | NHD↔HND | 有 primitive，但 H1.1 不主要用它 |
| 完整 edge 直接计时 | **有** | **H1.1 CUDA 判据里没有** |
| 候选 | NN/NH-copy/HH/HN/NH-view | producer NHD/HND + consumer NHD/HND |
| H1.1 比较 | producer-local winner vs **full-edge oracle** | producer winner vs **consumer winner** |
| 统计 | ≥5 process + bootstrap + adaptive epsilon | ≥3% practical gate |

因此，如果严格地说：

\[
\boxed{
\text{v13 CUDA H1.1 并不等价于 v10 的完整 edge experiment}
}
\]

它更像：

\[
\boxed{
\text{producer preference conflict test}
}
\]

即：

> KV writer 最喜欢的 layout，是否和 head-local consumer 最喜欢的 layout 不一致？

而 v10 才进一步问：

> 即使两边 preference 冲突，把 producer、conversion、consumer 全部放进总账以后，真正的 edge winner 是否还会改变？

---

还有一个容易混淆的地方：v13 的 CUDA 文件确实存在真正的 paged consumer：

```cpp
paged_decode_head_kernel(...)
```

它包含：

```text
page_size
block_table
logical_page
physical_page
within
```

所以它是真正 paged KV scan。

但是这条 kernel 主要被用于后面的 page/layout granularity、metadata、RQ10 等实验，并不是 v13 H1.1 的核心 `decode_head_scan` 判据。

所以不能因为 `rq_llm_layout_bench.cu` 里出现了 paged kernel，就说：

> v13 RQ1 = paged KV write → paged head scan。

并不是。

---

还有第三个细节值得注意：v13 的 Triton H1.1 反而比 CUDA H1.1 更接近真正的 “edge inversion”。

`analyze_v13_rqs.py` 对 Triton 会读取：

```text
producer_p50_ms
pipeline_p50_ms
```

然后分别找：

\[
L_P^*=\arg\min T_{\text{producer}}
\]

和：

\[
L_E^*=\arg\min T_{\text{pipeline}}
\]

再判断：

\[
L_P^*\neq L_E^*
\]

以及：

\[
\frac{
T_{\text{pipeline}}(L_P^*)
}{
T_{\text{pipeline}}(L_E^*)
}
\ge 1.03
\]

所以：

```text
v13 CUDA H1.1
    producer winner
       vs
    consumer winner
```

而：

```text
v13 Triton H1.1
    producer winner
       vs
    producer+consumer pipeline winner
```

后者才真正符合：

\[
L_{\text{producer-best}}
\stackrel{?}{=}
L_{\text{edge-best}}
\]

这个定义。

因此，从你前面指出的 RQ1 缺陷来看，v13 并没有根本解决它。**CUDA 部分仍然主要是在用人工设计的 KV writer + head-local consumer 制造 layout preference conflict；而且 v13 的 CUDA H1.1 比 v10 在“完整 edge 总账”这一点上反而更弱。** v13 的优势主要是 shape 覆盖更广，以及额外加入 Triton pipeline evidence。

---

是的，你抓到了 RQ1 当前设计里最关键的不足。不过先纠正一个方向问题：当前 CUDA RQ1 不是“Producer 读、Consumer 写”，而是

\[
\text{Producer: global input 读}\rightarrow\text{global KV cache 写}
\]

\[
\text{Consumer: global KV cache 读}\rightarrow\text{output 写}
\]

真正的问题是：它只构造了一个 synthetic 的 `KV writer → head-local scan`，没有把各框架源码里真实存在的 producer→consumer 边逐条拿出来验证。

更具体地说，真实框架里确实能找到比当前 RQ1 更有说服力的 layout-conflict 边：

| 框架 | 真实 producer→consumer 边 | layout 冲突是什么 | 当前 RQ1 是否真正验证 |
|---|---|---|---|
| SGLang | KV pool writer → TRTLLM-MHA/FlashInfer/HPC attention | native pool 是 NHD，但部分 decode kernel 要 HND；其他 kernel 又直接吃 NHD | 没有 |
| vLLM | `reshape_and_cache*` writer → attention backend | Flex/HPC 与 B12X/FlashInfer 对 KV layout 的合法/偏好集合不同 | 没有 |
| Triton | `tt.load`/elementwise producer → `tt.dot`；或 `tt.dot` → `tt.store` | load/store 偏向 coalesced BlockedEncoding，dot operand 要 DotOperandEncoding | 没有 |
| TVM | producer block → buffer → consumer block | producer/consumer 可拥有不同 IndexMap/access order，需要 `transform_layout`/reindex | 当前 harness 只测了 consumer，很不完整 |
| CUTLASS/CuTe | MMA accumulator → epilogue → global output/next kernel | MMA/TMEM fragment layout、epilogue shared/TMA layout、下游 global operand layout不同 | 当前只做邻近 boundary benchmark |
| Hexcute | 一个 tile value 被不同 instruction/consumer 使用 | 不同 anchor instruction 对 thread-value/shared layout 施加不同约束，冲突时需要 rearrange | 当前没有映射成真实 edge 实验 |

下面几个例子尤其能说明为什么你这个批评是成立的。

### SGLang 是最直接的真实例子

SGLang 当前 `trtllm_mha_backend.py` 里直接写明：

> Native pool format is NHD

即 persistent KV pool 原生格式类似：

\[
[\text{page},N,H,D]
\]

但是同一个文件又明确说：

> Decode and SM100 batch_context kernels require HND layout.

于是代码需要：

```text
native NHD KV pool
        │
        ▼
_reshape_paged_kv_cache(...)
        │
        ▼
HND view/layout
        │
        ▼
TRTLLM MHA decode
```

这已经是一个非常真实的 RQ1 edge：

\[
\boxed{
\text{KV writer/persistent pool}
\rightarrow
\text{layout adaptation}
\rightarrow
\text{TRTLLM-MHA decode}
}
\]

而且更有意思的是，同一个 SGLang persistent pool，如果 consumer 换成 HPC-Ops，HPC-Ops 的代码明确说它可以直接读取 token-major NHD cache，无需 copy。citeturn983460search1turn983460search7

于是同一个 producer：

```text
SGLang KV writer
       │
       ▼
   NHD KV pool
```

下游可能是：

```text
consumer A = HPC-Ops
             ↓
          喜欢/接受 NHD
```

也可能是：

```text
consumer B = TRTLLM MHA decode on SM100
             ↓
          要 HND
```

这比当前 synthetic `decode_consumer()` 有说服力得多。

真正应该测的是：

\[
T_{\text{writer}}(NHD),\quad
T_{\text{writer}}(HND)
\]

以及：

\[
T_{\text{TRTLLM}}(NHD),\quad
T_{\text{TRTLLM}}(HND)
\]

再加：

\[
T_{NHD\rightarrow HND}
\]

最终比较：

\[
NHD\rightarrow TRTLLM
\]

\[
NHD\rightarrow convert\rightarrow HND\rightarrow TRTLLM
\]

\[
HND\text{-native writer}\rightarrow TRTLLM
\]

这才是真正 framework-native 的 RQ1。

---

### vLLM 里甚至已经出现了“layout 集合冲突”的真实 issue

vLLM 的 `AttentionBackend` 接口现在直接提供：

```python
supported_kv_cache_layouts()
```

定义就是：

> 这个 backend 的 kernel 能消费哪些 KV layouts，并按 preference 排序。

vLLM 随后对所有 backend 做 layout intersection，并为整个模型选择一个 persistent KV layout。citeturn938870search0turn938870search1

这说明 vLLM 自己已经承认：

\[
\boxed{
\text{不同 consumer backend 对同一个 KV tensor 有不同 layout contract}
}
\]

甚至已经有非常具体的 bug。

2026 年 9 月的 issue #55312 报告：

```text
FlexAttention:
    {LBNHC}

B12X / FlashInfer(SM100):
    {LBHNC, BLHNC}
```

两边 layout 集合可以完全不相交：

\[
\{LBNHC\}
\cap
\{LBHNC,BLHNC\}
=
\varnothing
\]

最终 engine init 直接失败。citeturn938870search3

这里：

\[
LBNHC\approx NHD\text{-like}
\]

而：

\[
LBHNC\approx HND\text{-like}
\]

所以这是一个非常强的真实证据：

> 不同 consumer 对同一个 persistent tensor 的 layout preference/legality 确实会冲突。

而且 vLLM 还有 issue #42082，动机就是不同 attention backends 存在“语义上和物理上不同”的 KV-cache layouts，因此需要统一 layout abstraction。citeturn938870search2

但这里还缺一步：

vLLM 源码证明了

\[
\boxed{\text{consumer layout conflict}}
\]

却没有自动证明

\[
\boxed{
L_{\text{producer-best}}
\neq
L_{\text{edge-best}}
}
\]

因为还必须测 producer。

vLLM 中可以把 producer 定义成：

```text
K/V projection output
        ↓
reshape_and_cache_flash /
write_to_paged_cache
        ↓
persistent KV cache
```

源码里 `reshape_and_cache_flash` 本身就是 layout-aware 的，benchmark 也接受 `kv_cache_layout`；不同 backend 随后消费这个 cache。当前 vLLM 源码还显示例如 HPC backend 只支持 `LBNHC`，B12X 则支持 `LBHNC/BLHNC`。因此这是一组非常自然的真实 RQ1 candidate edges。citeturn938870search0

正确实验应该问：

\[
\arg\min_L T_{\text{reshape\_and\_cache}}(L)
\]

是否等于：

\[
\arg\min_L
[
T_{\text{reshape\_and\_cache}}(L)
+
T_{\text{attention backend}}(L)
]
\]

这才是原始 RQ1 的真正含义。

---

### Triton 的冲突甚至直接存在于 IR 中

Triton 是另一个非常典型的例子，只不过这里不是 persistent global layout，而是 kernel 内 distributed/register layout。

TritonGPU 对 `BlockedEncoding` 的说明非常明确：这种 layout 通常用于改善 `LoadInst/StoreInst` 的 memory coalescing。citeturn936472search0

所以：

```text
tt.load
```

通常希望产生类似：

```text
#blocked
```

的线程—元素映射。

但是到了：

```text
tt.dot
```

情况完全不同。

TritonGPU IR 直接规定，对于 pre-Hopper MMA：

> `tt.dot` 的 A/B operands 必须是 `DotOperandEncodingAttr`。citeturn936472search0

于是实际 IR 会出现：

```text
%a0 = tt.load ...
        │
        │ #blocked
        ▼
%a1 = ttg.convert_layout %a0
        │
        │ #dot_op
        ▼
%acc = tt.dot %a1, %b1
```

Triton 自己的 test IR 就包含这种：

```text
#blocked
   ↓
ttg.convert_layout
   ↓
#ttg.dot_op
   ↓
tt.dot
```

的真实例子。citeturn936472search0

甚至 `AssignCGALayouts.cpp` 的注释明确说：

> pass 会给 Dot/Reduce ops 它们“preferred CGA layout”，并在 boundary 上 materialize `ttg.convert_layout`。

这几乎就是 RQ1 在 IR 层的原生表述。citeturn936472search0

这里真正的 producer/consumer 是：

\[
\boxed{
tt.load
\rightarrow
tt.dot
}
\]

或：

\[
\boxed{
tt.dot
\rightarrow
tt.store
}
\]

第一条边：

- producer/load 更希望 coalesced blocked layout；
- consumer/dot 要 MMA operand layout。

第二条边：

- producer/dot 得到 MMA accumulator layout；
- consumer/store 又更希望适合 coalesced store 的 layout。

因此有：

\[
L_{\text{load-best}}
\neq
L_{\text{dot-best}}
\]

以及可能：

\[
L_{\text{dot-output-best}}
\neq
L_{\text{store-best}}
\]

这才是 Triton 真正应该用于 RQ1 的实验。

当前仓库里的 Triton KV-layout benchmark没有验证这种 IR-level edge。

---

### TVM 当前实验恰好证明了你说的“不足”

你这个仓库里的 `tvm_rq_observation_bench.py` 实际构造了：

```text
row_major
tiled_16x16
```

然后分别让：

```text
row reducer
column reducer
```

读取。

也就是说，它主要证明：

\[
\boxed{
\text{不同 consumer 可以有不同 layout preference}
}
\]

甚至构造：

```text
row consumers
+
column consumers
```

同时消费同一个 tensor。

但代码里非常关键的一行是：

```python
"producer_materialization_timed": False
```

这意味着 multi-consumer 实验没有把 producer materialization 纳入实际 runtime pipeline。

所以它测的是：

\[
C_{\text{row}}(L)
\]

和：

\[
C_{\text{column}}(L)
\]

而不是：

\[
P(L)
+
C(L)
\]

更不是：

\[
P(L_P)
+
Transform(L_P\rightarrow L_C)
+
C(L_C)
\]

因此 TVM 这一部分也确实没有完整回答原始 RQ1。

TVM IR 本身明明有：

```text
IndexMap
transform_layout
reindex
cache_read
cache_write
```

这些机制可以直接表达 producer block 与 consumer block 的 layout boundary。TVM 当前 API也明确提供 `transform_layout()`、`reindex_cache_read()` 和 `reindex_cache_write()`。citeturn677725search0

但当前实验没有选择一个真实：

```text
producer block
     ↓
buffer
     ↓
consumer block
```

然后枚举 producer-side 与 consumer-side layout。

所以这部分应该重做。

---

### CUTLASS/CuTe 也有真实的内部 producer-consumer boundary

CUTLASS/CuTe 中一个比较典型的真实边是：

\[
\boxed{
MMA
\rightarrow
Accumulator
\rightarrow
Epilogue
\rightarrow
Global\ C
}
\]

MMA 产生的数据首先处于 MMA/TMEM accumulator layout。

然后 epilogue warp 把 accumulator 从 TMEM/register 格式取出，再组织成适合 shared-memory/TMA/global store 的 layout。

当前 CUTLASS Blackwell GEMM 代码就明确包含：

```text
MMA producer
    ↓
TMEM accumulator
    ↓
epilogue consumer
    ↓
TMA store
    ↓
global C
```

而 epilogue 还单独构造 shared-memory swizzle/layout 与 TMA store pipeline。citeturn436480search0turn436480search2

所以 CUTLASS 中真实值得研究的 RQ1 不是简单的：

```text
flat
vs
tiled
```

而应该是：

\[
\boxed{
\text{MMA accumulator layout}
\rightarrow
\text{epilogue layout}
\rightarrow
\text{next operator operand layout}
}
\]

尤其是两个 GEMM 连起来：

```text
GEMM1 MMA
   ↓
epilogue/global C
   ↓
GEMM2 A/B input
```

这时 GEMM1 的最佳 epilogue store layout，很可能不是 GEMM2 的最佳 operand layout。

当前仓库里的 CUTLASS softmax boundary benchmark只能算这种问题的 proxy，并没有真正测这个 edge。

---

所以我会把你对当前 RQ1 的批评进一步表述成：

\[
\boxed{
\text{当前 RQ1 证明的是一个 synthetic mechanism，
不是 framework-native edge conflict。}
}
\]

它证明：

\[
\exists\ P,C,L_1,L_2:
L_P^*\neq L_E^*
\]

但是没有证明：

\[
\boxed{
\text{在 Triton/vLLM/SGLang/TVM/CUTLASS 的哪些真实 IR edge 上，
这种 inversion 实际发生。}
}
\]

这两个 claim 差别非常大。

如果重新设计 RQ1，我认为应该直接变成：

\[
\boxed{
\textbf{Framework-native Producer–Consumer Edge Matrix}
}
\]

例如至少做：

```text
SGLang:
KV writer
→ NHD/HND persistent cache
→ TRTLLM-MHA / FlashInfer / HPC-Ops

vLLM:
reshape_and_cache_flash
→ LBNHC/LBHNC/BLHNC
→ FlexAttention / FlashInfer / B12X / HPC

Triton:
tt.load (#blocked)
→ convert_layout
→ tt.dot (#dot_op)

Triton:
tt.dot (#mma)
→ convert_layout
→ tt.store (#blocked)

TVM:
producer TIR block
→ IndexMap/layout_transform
→ row/column/reduction consumer

CUTLASS:
MMA accumulator
→ epilogue/TMA layout
→ next GEMM/softmax consumer
```

然后每一条真实 edge 都统一测四个量：

\[
T_P(L)
\]

\[
T_C(L)
\]

\[
T_{\text{convert}}(L_i\rightarrow L_j)
\]

\[
T_{\text{edge}}(L_i,L_j)
\]

最后真正比较：

\[
\boxed{
L_P^*=\arg\min_L T_P(L)
}
\]

和：

\[
\boxed{
(L_P,L_C)^*
=
\arg\min_{L_P,L_C}
[
T_P(L_P)
+
T_{\text{adapt}}(L_P,L_C)
+
T_C(L_C)
]
}
\]

这样 RQ1 才能从“人工构造一个 producer/consumer 冲突”升级成：

> **真实框架 IR 与 runtime 中，哪些具体 producer–consumer edges 的局部 layout preference 确实冲突，以及这种冲突在什么 workload 下值得付 conversion/native-emission 成本。**

这会比目前的 RQ1 科学上强很多。

---

可以，但要把“谁读、谁写”区分准确。

在这个 RQ1 CUDA benchmark 里，真正的数据流是：

```text
global memory: input
      │
      │ producer 读
      ▼
producer_write()
      │
      │ producer 写
      ▼
global memory: KV cache (NHD/HND)
      │
      │ consumer 读
      ▼
decode_consumer()
      │
      ├─ block 内 reduction：shared memory
      │
      └─ 最终写一个结果
      ▼
global memory: output
```

所以并不是简单的“producer 读、consumer 写”。更准确是：

- Producer：从 global memory 的 `input` 读数据，再写到 global memory 的 paged KV cache。
- Consumer：从 global memory 的 KV cache 读大量数据，在 block 内用 shared memory 做 reduction，最后把结果写回 global memory 的 `output`。

源码里 producer：

```cpp
producer_write(
    const __nv_bfloat16* input,
    __nv_bfloat16* cache,
    ...
)
```

核心就是：

```cpp
cache[cache_offset(...)] = input[i];
```

这里 `input` 和 `cache` 都是通过 `cudaMalloc()` 分配的，所以都是 device global memory。

Consumer：

```cpp
decode_consumer(
    const __nv_bfloat16* cache,
    float* output,
    ...
)
```

主要读取：

```cpp
sum += __bfloat162float(
    cache[cache_offset(...)]
);
```

这个 `cache` 同样是 global memory。

之后：

```cpp
__shared__ float scratch[256];
```

进入 shared memory 做 block reduction。

最后：

```cpp
output[blockIdx.x] = scratch[0];
```

再写回 global memory。

因此这个实验真正关心的主要是：

\[
\boxed{
\text{global-memory KV layout 对 producer write 和 consumer read 的影响}
}
\]

而不是 shared-memory layout 优化。

---

至于第二个问题：**在这个特定 consumer kernel 中，可以认为 consumer 本身结构上偏好 HND。**

甚至可以说，这个 consumer 就是专门构造来体现这种偏好的。

原因是它的访问方式是：

\[
\boxed{\text{固定一个 KV head，沿 token 方向扫描}}
\]

代码逻辑相当于：

```text
固定 kvhead
for token = 0 ... KV-1:
    for d = 0 ... D-1:
        read cache[token, kvhead, d]
```

也就是逻辑访问顺序：

\[
[head=\text{fixed},\ token,\ d]
\]

而 HND 的物理布局恰好近似：

\[
[page,\ head,\ token,\ d]
\]

所以同一个 head 下：

```text
token0: D...
token1: D...
token2: D...
token3: D...
```

在内存中是连续或高度连续的。

相比之下，NHD 是：

\[
[page,\ token,\ head,\ d]
\]

内存大致是：

```text
token0:
  head0 D...
  head1 D...
  head2 D...
  head3 D...

token1:
  head0 D...
  head1 D...
  ...
```

如果 consumer 固定 `head0`，则它读：

```text
token0/head0
       ↓
跨过 head1/head2/head3
       ↓
token1/head0
       ↓
跨过 head1/head2/head3
       ↓
token2/head0
```

因此相邻 token 对同一 KV head 来说存在 stride。

源码自己就明确写了：

> `HND gives contiguous [within,dim] runs; NHD jumps over Hkv for adjacent tokens.`

所以对于这个 consumer：

\[
\boxed{
\text{access pattern 与 HND 的 physical order 更匹配}
}
\]

---

不过最好不要直接写成：

\[
\boxed{\text{Consumer 永远偏好 HND}}
\]

更严谨的是：

> **这个 head-local decode consumer 在其设计的访问模式下天然更有利于 HND；实际性能是否 HND 更快仍需要测量，并且会受到 Hkv、KV length、page size、GPU memory transaction/coalescing 等因素影响。**

特别是这里有一个很重要的因素：

\[
H_{kv}
\]

如果：

\[
H_{kv}=1
\]

那么 NHD 和 HND 的区别实际上会非常小，因为：

\[
[Token,1,D]
\]

和：

\[
[1,Token,D]
\]

在物理上几乎退化成同一种排列。

而如果：

\[
H_{kv}=8
\]

在 NHD 中同一个 head 的相邻 token 之间就要跨过另外 7 个 heads：

\[
\text{stride}\approx H_{kv}\times D
\]

这时 HND 的优势可能明显得多。

因此可以把这个实验理解成：

\[
\boxed{
Producer:
\text{固定 token，处理多个 heads}
}
\]

更接近 NHD 的自然顺序；

而：

\[
\boxed{
Consumer:
\text{固定 head，扫描多个 tokens}
}
\]

更接近 HND 的自然顺序。

所以 RQ1 才故意形成一个非常干净的冲突：

\[
\boxed{
\underbrace{\text{token-local write}}_{\text{producer}}
\quad\text{vs}\quad
\underbrace{\text{head-local scan}}_{\text{consumer}}
}
\]

也就是：

\[
\boxed{
\text{producer access order}
\neq
\text{consumer access order}
}
\]

这正是这个 RQ1 benchmark 能产生 layout rank inversion 的根本原因。

---

这里的 `edge` 不是图论里抽象的一条线那么简单。在这个 RQ1 实验里，它具体指：

\[
\boxed{
\text{producer 输出 tensor}
\rightarrow
\text{必要的 layout adaptation}
\rightarrow
\text{consumer 使用这个 tensor}
}
\]

也就是从 producer 到 consumer 之间这一整段“跨算子边界”的成本。

用仓库里的 `v10-core-r8` 这个 case 最容易理解。

这个 case 的配置是：

\[
B=8,\quad Q=1,\quad KV=2048,\quad H_q=32,\quad H_{kv}=4,\quad D=128,\quad page=64
\]

可以理解成：8 个请求，每次 decode 新产生 1 个 KV token，总 KV 长度 2048；32 个 query heads 共享 4 个 KV heads。

这里：

- producer = `producer_write()`，负责把新生成的 KV 写进 paged KV cache；
- consumer = `decode_consumer()`，负责固定一个 KV head，沿着 2048 个 token 扫描 KV；
- layout 可以是 NHD 或 HND。

先看 producer 自己。

假设实测：

\[
T_P(NHD)=2.0\ \mu s
\]

\[
T_P(HND)=2.6\ \mu s
\]

那么 producer 单独看，会认为：

\[
\boxed{NHD\ 最好}
\]

所以：

\[
L_{\text{producer-best}}=NHD
\]

因为 producer 只关心：

> “我把 KV 写出去，哪种 layout 最快？”

它完全不管后面的 consumer。

但 RQ1 不只看 producer。

它继续把这个 KV 交给 consumer。

假设 consumer 单独测出来：

\[
T_C(NHD)=10\ \mu s
\]

\[
T_C(HND)=5\ \mu s
\]

这说明 consumer 明显更喜欢 HND，因为它是固定 head 沿 token 扫描，而 HND 中同一个 head 的 token 更连续。

于是开始出现冲突：

```text
producer:
NHD 更快

consumer:
HND 更快
```

这时候真正应该选哪个 layout，不能只看 producer。

这就是为什么要看 `edge`。

假设第一条策略是 `NN`：

```text
producer
  │
  ▼
写 NHD
  │
  ▼
NHD cache
  │
  ▼
consumer 按 NHD 读取
```

那么这整个 edge 是：

\[
\boxed{
Producer_{NHD}
\rightarrow
NHD\ boundary
\rightarrow
Consumer_{NHD}
}
\]

完整 edge latency 假设测出来：

\[
T_{NN}=12.0\ \mu s
\]

注意，这里的 `12 μs` 不是简单说 “producer 2 + consumer 10” 的理论值，而是实际把这两个 kernel 连起来计时得到的整段 latency。

再看 `HH-native`：

```text
producer
  │
  ▼
直接写 HND
  │
  ▼
HND cache
  │
  ▼
consumer 按 HND 读取
```

整个 edge 是：

\[
\boxed{
Producer_{HND}
\rightarrow
HND\ boundary
\rightarrow
Consumer_{HND}
}
\]

虽然 producer 写 HND 更慢：

\[
2.6 > 2.0
\]

但 consumer 快很多。

所以假设完整 edge：

\[
T_{HH}=7.6\ \mu s
\]

现在就会出现：

\[
L_{\text{producer-best}}=NHD
\]

但：

\[
L_{\text{edge-best}}=HND
\]

因为：

\[
7.6<12.0
\]

所以 producer 自己最优的 NHD，放到整条 producer→consumer edge 上反而更差。

这就是 RQ1 所谓的：

\[
\boxed{
L_{\text{producer-best}}
\neq
L_{\text{edge-best}}
}
\]

再进一步，edge 甚至不只是 “producer + consumer”，因为中间可能有 layout conversion。

比如 `NH-copy`：

```text
producer
  │
  ▼
写 NHD
  │
  ▼
NHD cache
  │
  ▼
NHD → HND conversion
  │
  ▼
HND cache
  │
  ▼
consumer
```

那么这条 edge 的成本是：

\[
T_{NH}
=
T_{producer(NHD)}
+
T_{convert(NHD\to HND)}
+
T_{consumer(HND)}
\]

假设：

\[
T_P(NHD)=2.0
\]

\[
T_{convert}=3.0
\]

\[
T_C(HND)=5.0
\]

那么：

\[
T_{NH}\approx10.0\ \mu s
\]

于是现在三种选择可能是：

| Strategy | 含义 | Edge latency |
|---|---|---:|
| NN | NHD producer → NHD consumer | 12.0 μs |
| NH-copy | NHD producer → convert → HND consumer | 10.0 μs |
| HH-native | HND producer → HND consumer | 7.6 μs |

从 producer 自己看：

\[
NHD
\]

最好。

但从 edge 看：

\[
HH-native
\]

最好。

所以这个实验真正想表达的是：

> 一个 tensor 的 layout 决策，不能只在 producer 写完 tensor 的那个瞬间做。

因为这个 tensor 马上会被下游 consumer 使用。

应该看：

\[
\boxed{
\text{生产它的成本}
+
\text{跨边界适配成本}
+
\text{消费它的成本}
}
\]

这整段就是这里的 `edge cost`。

你也可以把它画成一个很简单的计算图：

```text
            producer-local view

     producer_write()
           │
           ▼
        KV tensor
           ↑
           │
     只看这里结束
     → NHD 可能最快


────────────────────────────────

              edge view

     producer_write()
           │
           ▼
        KV tensor
           │
     layout boundary
           │
     [optional convert]
           │
           ▼
     decode_consumer()
           │
           ▼
         output

     看完整这一段
     → HND 可能整体更快
```

因此这里的 `edge` 更准确地说，是计算图里：

\[
\boxed{
\text{producer output boundary + downstream consumer path}
}
\]

而不是整个模型，也不是完整 attention。

在这个 RQ1 的 CUDA case 中，它具体就是：

\[
\boxed{
\text{KV append}
\rightarrow
\text{KV physical layout / conversion}
\rightarrow
\text{paged KV scan}
}
\]

所以 `edge-best` 的含义是：

> 对这条 producer→consumer 边界来说，哪一种 layout 策略让整段执行时间最短。

---

如果你这里说的“架构”是指框架/后端，那么除了 CUDA reference，仓库对 RQ1 还涉及 Triton/PyTorch、vLLM、SGLang、TVM、CUTLASS/CuTe、TileLang、Hexcute，以及预留的 FlashInfer、TensorRT-LLM 证据接口。但它们的验证强度不同，不能都说成“和 CUDA 一样验证了 RQ1”。

最准确的分类如下。

| 框架/后端 | 是否实际有运行代码 | RQ1 中测什么 | 是否等价于 CUDA 的五路径实验 |
|---|---|---|---|
| CUDA reference | 是 | NHD/HND producer、conversion、consumer、完整 edge | **是，核心因果证据** |
| Triton / PyTorch | 是 | 真实 LLM shape、KV-layout/boundary layout、完整子图外部有效性 | 否 |
| vLLM | 是，可选 native serving | 强制不同 KV cache layout，测完整请求 | 否 |
| SGLang | 是，可选 native serving | NHD/HND KV cache，测完整请求 | 否 |
| TVM | 是 | row-major/tiled、layout transform、consumer/boundary | 否 |
| CUTLASS/CuTe | 是 | softmax/tiled boundary、direct vs transform | 否 |
| TileLang | 是 | 真实 LLM 子图的外部有效性 | 否 |
| Hexcute | 可选 artifact | upstream kernel/layout artifact | 否，默认甚至不运行 |
| FlashInfer | 分析器预留 artifact | 只有存在专门 artifact 才算 native edge decomposition | 当前仓库不能确认已经运行 |
| TensorRT-LLM | 同上 | 同上 | 当前仓库不能确认已经运行 |

其中最需要区分的是下面几类。

### 1. Triton：最接近第二套独立验证

在较广的 RQ runner `run_ref_talks_rq_gpu1.sh` 里，明确运行：

```text
triton_kv_layout_bench.py
```

输出：

```text
triton_kv_layout.csv
```

它针对 attention cases 比较真实的 KV layout，比如 NHD/HND，另外还会跑 PyTorch/Triton 的真实 LLM representative subgraphs。

因此 Triton 的作用主要是：

\[
\boxed{\text{检查 CUDA reference 发现的 layout sensitivity 是否也出现在另一套 kernel stack}}
\]

但它并没有严格复刻 v10 的：

\[
NN,\ NH-copy,\ HH-native,\ HN-copy,\ NH-view
\]

全部五条路径和完全相同的 statistical adjudication。

所以论文里更安全的说法是：

> Triton provides an independent layout-sensitive runtime replication / external-validity check.

而不是：

> Triton 完整复现了 v10 RQ1。

---

### 2. vLLM：原生 serving-level 验证

仓库通过 `native_serving_layout_bench.py` 真正启动 vLLM server。

它可以强制：

```text
auto
LBNHC
LBHNC
BLNHC
BLHNC
```

这些都是 paged KV-cache physical layouts。

其中可以近似理解：

```text
LBNHC
≈ page/token/head/channel 类型顺序

LBHNC
≈ page/head/token/channel 类型顺序
```

也就是和 RQ1 中 NHD/HND tension 有直接关系。

然后保持：

```text
same model
same requests
same memory fraction
process isolation
```

运行真实请求，测：

```text
request p50
request p80
wall time
output tokens/s
```

所以 vLLM 验证的是：

\[
\boxed{
\text{真实 serving engine 中，不同 persistent KV layout 是否产生性能差异}
}
\]

但这里整个请求包含很多东西：

```text
scheduler
GEMM
RoPE
attention
KV write/read
sampling
runtime overhead
...
```

因此无法拆成：

\[
P_{NHD},\quad M_{N\to H},\quad C_{HND}
\]

所以仓库自己明确说：

> vLLM whole-engine measurements cannot decompose producer, conversion edge and consumer costs.

因此 vLLM 属于 **native external validity**，不是 RQ1 的核心因果证明。

---

### 3. SGLang：原生 NHD/HND 验证

SGLang 更直接。

代码里有：

```text
auto_nhd
auto_hnd
```

分别通过：

```text
SGLANG_USE_HND_KVCACHE
```

控制 HND 与 NHD。

这两组被标成：

```text
comparison_scope = layout_only
```

意味着意图上保持：

```text
same backend
same page size
same model
same requests
```

只改变：

\[
\boxed{KV\ layout}
\]

因此这是比较有价值的一套 native evidence：

```text
SGLang
   │
   ├── NHD KV cache
   │
   └── HND KV cache
        ↓
真实 serving latency
```

但是同样，它不是：

```text
producer
→ conversion
→ consumer
```

的逐项拆分。

所以它证明的是：

> layout choice 在真实 serving engine 中确实可能改变性能。

而不是直接证明：

> producer-local winner 与 edge winner 发生严格 rank inversion。

---

### 4. TVM：用另一种 tensor/compiler boundary 验证同类现象

canonical runner 会运行：

```text
tvm_layout_bench.py
tvm_rq_observation_bench.py
```

TVM 这里主要研究：

```text
row-major
vs
tiled
```

以及：

```text
direct consumer
vs
layout conversion
vs
compiled function
```

所以抽象上对应：

\[
\text{producer/native representation}
\rightarrow
\text{layout transform}
\rightarrow
\text{consumer}
\]

这和 RQ1 的思想是一致的：

> 单个节点喜欢的 layout 与跨 boundary 的最优方案未必一致。

但它研究的 layout family 已经不是：

\[
NHD/HND
\]

而是：

\[
row\text{-}major/tiled
\]

因此属于：

\[
\boxed{\text{cross-framework mechanism evidence}}
\]

而不是同一 attention experiment 的 replication。

---

### 5. CUTLASS/CuTe：验证另一个 producer-consumer boundary

runner 中运行：

```text
run_softmax_layout_boundary.py
```

输出：

```text
cutlass_softmax_boundary.csv
```

大致比较：

```text
flat reference

tiled input
→ tiled/direct consumer

tiled input
→ transform
→ flat consumer
```

所以研究的仍然是：

\[
\boxed{
\text{直接消费 producer layout}
\quad vs\quad
\text{先转换再让 consumer 使用}
}
\]

这在概念上与：

```text
HH-native
vs
NH-copy
```

非常相似。

但这里是 softmax/tile boundary，不是 paged KV decode。

所以 CUTLASS 是 RQ1 的“邻近机制验证”。

---

### 6. TileLang

canonical runner 还会跑：

```text
llm_representative_tilelang.jsonl
```

覆盖真实 LLM representative cases。

但 TileLang 这里更偏向：

\[
\boxed{\text{真实子图 external validity / framework comparison}}
\]

而不是专门构造 producer-local vs edge-optimal rank inversion。

所以不能把 TileLang 结果单独说成“直接验证 H1.1”。

---

### 7. Hexcute

Hexcute 更弱一些。

代码是：

```bash
RUN_HEXCUTE_ARTIFACT=1
HEXCUTE_BENCH_DIR=...
```

才真正跑 upstream artifact。

否则记录：

```text
source_only
```

因此默认情况下它只是：

\[
\boxed{\text{source/artifact supporting evidence}}
\]

不能写成“Hexcute runtime 验证了 RQ1”。

---

## FlashInfer 和 TensorRT-LLM 要特别小心

`analyze_v10_rq1.py` 里确实定义了：

```python
"FlashInfer": "flashinfer_v10_rq1.csv"
"TensorRT-LLM": "tensorrt_llm_v10_rq1.csv"
```

而且这两个非常特殊。

只有这两个 artifact 存在时，分析器才会把它们标成：

```text
v10_edge_decomposition
```

也就是理论上可以作为：

\[
\boxed{\text{独立 native producer/boundary/consumer decomposition}}
\]

但是，我重新查了当前仓库：

没有找到对应的：

```text
flashinfer_v10_rq1.py
tensorrt_llm_v10_rq1.py
```

strict v10 runner 本身也没有生成这两个 CSV。

因此当前仓库源码只能说明：

> **分析器支持接收 FlashInfer/TensorRT-LLM 的 exact v10 edge artifact。**

不能据此说：

> **当前仓库已经实际用 FlashInfer 和 TensorRT-LLM 完整验证了 RQ1。**

这是一个很重要的区别。

---

因此，如果从“RQ1 的证据层次”来看，更准确的结构是：

```text
                         RQ1
                          │
          ┌───────────────┼────────────────┐
          │               │                │
      核心因果验证      独立机制验证       Native外部有效性
          │               │                │
          ▼               ▼                ▼
        CUDA           Triton             vLLM
 NN/NH/HH/HN/view       TVM               SGLang
 producer/consumer      CUTLASS
 decomposition
          │
          │
          ▼
   最强的 H1.1 证据

                     其他支持
                        │
               TileLang / Hexcute
                        │
                  framework evidence
```

另外，如果你说的“架构”是**GPU 硬件架构**，答案反而很简单：这套实验默认基本都运行在 NVIDIA A10、`sm_86` 上。当前 repository 的默认 runner 是：

```text
ARCH=sm_86
physical GPU 1
```

所以它**没有**完成 A100/H100/Blackwell/AMD 等跨硬件架构的 RQ1 验证；NVFP4 在这个 CUDA reference 里甚至明确标成 unsupported。也就是说，目前“多架构”主要是**多框架/多软件栈验证**，不是多 GPU generation 验证。

---

RQ1 的验证 case 可以分成两类：一类是严格受控的 synthetic/controlled cases，用来证明因果；另一类是从真实模型 manifest 中抽出的 decode shape，用来检查外部有效性。核心 case 和 sensitivity sweep 的配置都在 [`build_v10_rq1_cases.py`](https://github.com/idoOwhd/layout_Test/blob/main/build_v10_rq1_cases.py)，真正执行 producer/layout/consumer 的 CUDA 代码在 [`v10_rq1_edge_bench.cu`](https://github.com/idoOwhd/layout_Test/blob/main/v10_rq1_edge_bench.cu)。

先看最核心的 case。

| Case | B | Q | KV | Hq | Hkv | Hq/Hkv | D | Page | Reuse | Dtype | 目的 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `v10-neg-bf16-small` | 1 | 1 | 512 | 8 | 8 | 1 | 128 | 64 | 1 | BF16 | negative control |
| `v10-core-r1` | 8 | 1 | 2048 | 8 | 8 | 1 | 128 | 64 | 1 | BF16 | MHA-like core |
| `v10-core-r8` | 8 | 1 | 2048 | 32 | 4 | 8 | 128 | 64 | 1 | BF16 | GQA core |
| `v10-core-r32` | 8 | 1 | 2048 | 32 | 1 | 32 | 128 | 64 | 1 | BF16 | MQA-like core |
| `v10-reuse-r8-x4` | 8 | 1 | 2048 | 32 | 4 | 8 | 128 | 64 | 4 | BF16 | reuse |
| `v10-reuse-r8-x16` | 8 | 1 | 2048 | 32 | 4 | 8 | 128 | 64 | 16 | BF16 | reuse |
| `v10-reuse-r8-x64` | 8 | 1 | 2048 | 32 | 4 | 8 | 128 | 64 | 64 | BF16 | reuse |

另外还预注册了三个相同几何的 NVFP4 case：

```text
v10-core-nvfp4-r1
v10-core-nvfp4-r8
v10-core-nvfp4-r32
```

但是当前这个 CUDA reference 只真正实现 BF16；如果 dtype 不是 BF16，它明确输出：

```text
unsupported_native_NVFP4_on_this_reference
```

不会偷偷用 BF16 替代 NVFP4。

---

### 1. 这些 case 到底表示什么？

核心 case 全部是在模拟 decode 时的一段 KV-cache 边界：

\[
\boxed{
\text{new KV}
\rightarrow
\text{write into paged KV cache}
\rightarrow
\text{read existing KV cache}
}
\]

其中：

- `B`：batch size；
- `Q`：这次 producer 新写入多少 token；
- `KV`：写完后整个 KV cache 的有效 sequence length；
- `Hq`：query head 数；
- `Hkv`：KV head 数；
- `D`：head dimension；
- `Page`：paged KV cache 中每页多少 token；
- `Reuse`：同一份 cache 被 consumer 重复读多少次。

例如最典型的：

\[
B=8,\ Q=1,\ KV=2048,\ H_q=32,\ H_{kv}=4,\ D=128
\]

意味着：

> 有 8 个 request；每个 request 当前 append 1 个新 KV token；总上下文长度 2048；32 个 query heads 共享 4 个 KV heads，即每个 KV head 被 8 个 query heads 使用。

这就是 `v10-core-r8`。

---

# Producer 到底是什么？

这里的 producer 不是完整的：

```text
hidden states
→ K projection GEMM
→ RoPE
→ KV cache write
```

而只是最后的：

\[
\boxed{\text{KV-cache writer}}
\]

也就是：

```cpp
producer_write()
```

它收到已经准备好的 BF16 `input`，然后把当前的 `q_len` 个新 KV token 写到 cache 尾部。

代码：

```cpp
int first = kv_len - q_len;
```

因此假设：

\[
Q=1,\ KV=2048
\]

它写的是逻辑上的：

\[
token=2047
\]

如果：

\[
Q=8,\ KV=2048
\]

则写：

\[
token=2040,\ldots,2047
\]

所以 producer 更准确的名字是：

\[
\boxed{\text{paged KV append/write kernel}}
\]

它不包含 QK、softmax，也不包含真正模型的 K/V GEMM。

---

# Consumer 到底是什么？

Consumer 是：

```cpp
decode_consumer()
```

它也不是完整 Attention。

它是一个专门模拟 decode KV 读取模式的：

\[
\boxed{\text{head-local paged KV scan}}
\]

对于一个 query head：

```cpp
int group = hq / hkv;
int kvhead = qhead / group;
```

先确定它对应哪个 KV head。

例如：

\[
H_q=32,\ H_{kv}=4
\]

那么：

\[
group=8
\]

所以：

```text
qhead 0–7   → kvhead 0
qhead 8–15  → kvhead 1
qhead 16–23 → kvhead 2
qhead 24–31 → kvhead 3
```

然后 consumer 固定这个 KV head，沿整个：

\[
token=0\dots KV-1
\]

以及：

\[
d=0\dots D-1
\]

进行读取。

实际上它做的是：

\[
sum += cache[token,kvhead,d]
\]

最后 reduction 成一个 float。

所以它模拟的是 Attention decode 中最关键的内存行为：

> 一个 query head 对对应的 KV head 做长 sequence 扫描。

但它没有执行：

\[
QK^\top
\]

也没有执行 softmax/PV。

这点很重要。

---

# Layout 到底是怎么改变的？

这里不是给同一块连续内存换个“名字”。

它真的改变物理地址映射。

逻辑 tensor 可以理解成：

\[
KV[b,t,h,d]
\]

两种 layout 保存的是完全一样的逻辑数据，只是物理位置不同。

### NHD

代码地址公式对应：

\[
[B,\ Page,\ TokenWithinPage,\ H_{kv},D]
\]

也就是：

```text
page 0
 ├─ token 0
 │   ├─ head 0: D...
 │   ├─ head 1: D...
 │   ├─ head 2: D...
 │   └─ head 3: D...
 │
 ├─ token 1
 │   ├─ head 0: D...
 │   ├─ head 1: D...
 │   ...
```

本质上：

\[
\boxed{\text{token-major inside each page}}
\]

---

### HND

地址公式对应：

\[
[B,\ Page,\ H_{kv},TokenWithinPage,D]
\]

也就是：

```text
page 0
 ├─ head 0
 │   ├─ token 0: D...
 │   ├─ token 1: D...
 │   ├─ token 2: D...
 │
 ├─ head 1
 │   ├─ token 0: D...
 │   ├─ token 1: D...
 │   ...
```

本质上：

\[
\boxed{\text{head-major inside each page}}
\]

所以它真正改变的是 stride/address mapping：

\[
offset_{NHD}(b,t,h,d)
\neq
offset_{HND}(b,t,h,d)
\]

逻辑值：

\[
KV[b,t,h,d]
\]

则完全相同。

---

# 为什么这两个 layout 会让 producer / consumer 有不同表现？

因为 producer 和 consumer 的访问方向不同。

Producer 是在写“当前新 token”的所有 KV heads：

```text
固定 token
↓
遍历 Hkv
↓
遍历 D
```

也就是更接近：

\[
[token,\ head,\ d]
\]

所以 token-major 的 NHD 与这种访问方式比较自然。

但 consumer 是：

```text
固定 KV head
↓
遍历所有 token
↓
遍历 D
```

更接近：

\[
[head,\ token,\ d]
\]

所以 HND 会让同一个 head 的：

```text
token0 D...
token1 D...
token2 D...
...
```

连续排列。

源码自己的注释就是：

> HND gives contiguous `[within, dim]` runs; NHD jumps over Hkv for adjacent tokens.

因此 RQ1 故意构造了这种 tension：

\[
\boxed{
producer\ 偏好的访问顺序
\neq
consumer\ 偏好的访问顺序
}
\]

这正是它要验证的问题。

---

# 每一个 case 不是只跑 NHD 和 HND 两遍，而是跑 5 条路径

这是 RQ1 的真正实验变量。

| Strategy | Producer 输出 | 中间处理 | Consumer 实际读取 | 含义 |
|---|---|---|---|---|
| `NN` | NHD | 无 | NHD | 全程 NHD |
| `NH-copy` | NHD | NHD→HND materialize | HND | producer 保持 NHD，但为 consumer 转换 |
| `HH-native` | HND | 无 | HND | producer 直接原生产 HND |
| `HN-copy` | HND | HND→NHD materialize | NHD | 反方向转换 |
| `NH-view` | NHD | 无 copy | NHD stride | consumer 接受原 storage |

所以对同一个：

\[
B,Q,KV,H_q,H_{kv},D,page
\]

它改变的是：

\[
\boxed{\text{producer layout + adaptation + consumer access layout}}
\]

而不是改变模型 shape 或逻辑计算。

---

## `NN`

```text
input
  │
  ▼
producer_write(NHD)
  │
  ▼
NHD KV cache
  │
  ▼
decode_consumer(NHD)
```

完整时间：

\[
T_{NN}
\]

---

## `NH-copy`

```text
input
  │
  ▼
producer_write(NHD)
  │
  ▼
NHD cache
  │
  ▼
convert_cache(NHD → HND)
  │
  ▼
HND cache
  │
  ▼
decode_consumer(HND)
```

完整时间：

\[
T_{NH}
\]

这里 conversion 会遍历完整：

\[
B\times KV\times H_{kv}\times D
\]

cache，而不是只转换刚 append 的 token。

所以 conversion 是很实在的一笔成本。

---

## `HH-native`

```text
input
  │
  ▼
producer_write(HND)
  │
  ▼
HND KV cache
  │
  ▼
decode_consumer(HND)
```

这里完全没有转换。

所以它测试的是：

> 与其先 NHD 再转换，不如 producer 一开始直接写 HND 会怎样？

---

## `HN-copy`

与上面反向：

```text
producer HND
→ full-cache HND→NHD
→ NHD consumer
```

---

## `NH-view`

这一条比较容易误解。

它不是：

```text
NHD storage
→ 神奇地获得 HND locality
```

它实际上还是：

```cpp
writer(NHD)
reader(NHD)
```

只是实验把它解释成：

> consumer 是 stride-polymorphic 的，因此允许直接接受 producer 的 NHD storage，而不用 materialize HND。

所以：

\[
NH\text{-view}
\]

和 `NN` 在当前 CUDA 实现中的实际 kernel path 是相同的。

区别主要是**语义分类**：

```text
NN
= consumer 本来就是 NHD consumer

NH-view
= consumer 原本逻辑上接受“alternate layout contract”，
  但允许通过 stride contract 直接读 NHD storage
```

因此这条主要用于 H1.3 的 zero-copy 讨论。

---

# Stage 1 和 Stage 2 还会把 producer / consumer 单独测出来

完整 edge 之外，代码还分别测：

\[
P_N=T_P(NHD)
\]

\[
P_H=T_P(HND)
\]

即：

```cpp
measure(writer(NHD))
measure(writer(HND))
```

用来得到：

\[
L_P^*
=
\arg\min(P_N,P_H)
\]

这是 producer-local winner。

同时测：

\[
C_N=T_C(NHD)
\]

\[
C_H=T_C(HND)
\]

即：

```cpp
measure(reader(NHD))
measure(reader(HND))
```

这一步用的是已经提前构建好的、逻辑内容相同的 NHD/HND cache。

因此 consumer-only 测量**不包含 cache construction cost**。

这正是 Stage 2 的意义：

\[
\boxed{\text{单独问 consumer 自己喜欢 NHD 还是 HND}}
\]

然后 Stage 3 才把：

```text
producer
+ conversion if any
+ consumer
```

全部放进去。

---

# 为什么特意改变 Hq/Hkv？

这其实是 Stage 3 最关键的 shape variable。

三组：

\[
H_q/H_{kv}=1,8,32
\]

分别代表越来越强的 KV-head reuse。

对于 ratio=1：

```text
1 query head
→ 1 KV head
```

对于 ratio=8：

```text
8 query heads
→ share 1 KV head
```

对于 ratio=32：

```text
32 query heads
→ share 1 KV head
```

而 `decode_consumer()` 是按 query head 启动工作的。

因此同一 KV head 的内容会被更多 query heads 重复读取。

也就是说：

\[
H_q/H_{kv}\uparrow
\]

会强化：

\[
\boxed{\text{consumer-side layout locality 的重要性}}
\]

所以它不是随便选三个 head 数，而是在有意识地改变：

> producer cost 与 downstream reuse/locality cost 的权重。

---

# Reuse case 又进一步把这个机制放大

在：

```text
B=8
Q=1
KV=2048
Hq=32
Hkv=4
D=128
page=64
```

完全固定的情况下，把：

\[
reuse=1,4,16,64
\]

改变。

代码直接：

```cpp
for (int r = 0; r < c.reuse; ++r) {
    decode_consumer(...);
}
```

所以假设 HND consumer 每次比 NHD consumer 少：

\[
\Delta C
\]

那么重复 \(R\) 次以后收益大约会变成：

\[
R\Delta C
\]

而一次 conversion 仍然只有：

\[
M
\]

于是：

\[
M < R\Delta C
\]

可能在某个 reuse 之后成立。

这就是 H1.2。

---

# Stage 4 到底测试哪些 case？

如果 Stage 3 至少出现一个严格 positive inversion，才运行下面的 sensitivity sweep。

它以：

\[
B=8,\ Q=1,\ KV=2048,\ H_q=32,\ H_{kv}=4,\ D=128,\ Page=64
\]

为中心。

然后一次主要改变一个维度：

| Sweep | 取值 | 其他核心参数 |
|---|---|---|
| Batch | 1, 8, 32, 64 | Q=1, KV=2048, 32/4 heads, D=128, page=64 |
| KV length | 128, 512, 2048, 8192, 32768 | B=8，其余同中心 |
| GQA ratio | 1, 4, 8, 16, 32 | 对应 8/8、32/8、32/4、32/2、32/1 |
| Q length | 1, 4, 8 | B=8, KV=2048 |
| Page size | 16, 64, 128 | B=8, Q=1, KV=2048 |

总计：

\[
4+5+5+3+3=20
\]

个 sensitivity cases。

每一个又都跑：

\[
NN,\ NH-copy,\ HH-native,\ HN-copy,\ NH-view
\]

五种策略。

所以 Stage 4 不是找一个“最佳 layout”，而是在找：

\[
\boxed{
L^*=
f(B,Q,KV,H_q/H_{kv},Page)
}
\]

也就是哪一片 shape 区域发生 layout crossover。

---

# 真实模型 case 是怎么来的？

`build_v10_rq1_cases.py` 调用：

```python
choose_attention(load(manifest), 3)
```

源 manifest 有 640 个 real-world cases。

只保留：

```text
gqa
sliding_attention
```

并按照：

```text
structure
× phase
× attention kind
```

分组。

Attention kind 是：

```text
Hq == Hkv → MHA
Hkv == 1  → MQA
otherwise → GQA
```

在每一组内，先去重 tensor contract，再按估算 workload 排序，full 模式取：

\[
\boxed{\text{min / median / max}}
\]

三个 shape。

对于 v10 RQ1 core，又只保留：

```text
phase = decode
```

然后去重：

\[
(KV,H_q,H_{kv},D,structure)
\]

这些真实 shape进入 CUDA reference 后仍然统一：

\[
Q=1,\ Page=64,\ Reuse=1,\ dtype=BF16
\]

但保留模型的：

\[
B,\ KV,\ H_q,\ H_{kv},D
\]

所以它们不是完整真实模型执行，而是：

\[
\boxed{
\text{真实模型 geometry}
+
\text{统一 controlled producer/consumer mechanism}
}
\]

需要注意一点：这个 GitHub 仓库当前没有包含它代码中引用的 sibling `real_world_shapes/real_world_shape_manifest.json`，因此我现在能精确说明**真实模型 shape 的选择算法**，但不能仅从这个仓库快照列出那批真实模型 case 的全部具体 model ID 和数值。

---

# 最后把整个实验对象对应起来

可以把 RQ1 的每一个 case 理解为：

```text
                Case geometry
 B,Q,KV,Hq,Hkv,D,page,reuse
                     │
          logical KV values fixed
                     │
           ┌─────────┴─────────┐
           │                   │
      Producer NHD        Producer HND
        KV append           KV append
           │                   │
           │              physical HND
      physical NHD              │
           │                    │
      ┌────┴──────┐        ┌────┴──────┐
      │           │        │           │
      ▼           ▼        ▼           ▼
 NHD consumer   N→H     HND consumer  H→N
      │          copy       │          copy
      │           │         │           │
      │           ▼         │           ▼
      │      HND consumer   │      NHD consumer
      │                     │
      └──── NH-view ────────┘

               ↓

 NN / NH-copy / HH-native / HN-copy / NH-view

               ↓

      compare complete edge latency
```

因此，RQ1 中所谓的“改变 layout”并不是改变整个模型，而是：

\[
\boxed{
\text{保持同一个逻辑 KV tensor、同一个 shape、同一个 producer/consumer 计算，}
}
\]

只改变：

\[
\boxed{
\text{KV cache 的物理 address mapping}
}
\]

即：

\[
[B,P,N,H,D]
\]

与：

\[
[B,P,H,N,D]
\]

以及是否在 producer 和 consumer 之间加入：

\[
NHD\leftrightarrow HND
\]

materialization。

这正是为什么这个实验能比较干净地回答 RQ1：**当逻辑计算不变时，producer 的局部最优 physical layout 是否会因为 consumer 的访问模式和转换成本而变成整个 edge 的次优选择。**

---

我重新读取了这个仓库。当前仓库对 **RQ1 最严格、最完整的实现**是专门的 `layout_summary_v10 L-RQ1` protocol，而不是简单跑几个 NHD/HND kernel 后比延迟。

RQ1 的核心问题是：

> **producer 自己最喜欢的 physical layout，是否可能不是 producer → consumer 整条 edge 的最优 layout？**

也就是研究：

\[
L_{\text{producer-best}}
\stackrel{?}{=}
L_{\text{edge-best}}
\]

仓库把这个问题拆成一个受控的 `KV producer → layout boundary → decode consumer` 实验。

---

## 1. RQ1 实际测的是什么计算

最重要的一点：这里的主实验**不是完整 Attention**。

它不是：

\[
QK^\top \rightarrow softmax \rightarrow PV
\]

而是一个刻意简化的、用于隔离 layout 因果效应的 CUDA reference：

\[
\boxed{
\text{KV producer write}
\rightarrow
\text{layout boundary}
\rightarrow
\text{head-local paged KV scan}
}
\]

源码就是：

[`v10_rq1_edge_bench.cu`](https://github.com/idoOwhd/layout_Test/blob/main/v10_rq1_edge_bench.cu)

其中有三个关键 kernel：

```text
producer_write()
convert_cache()
decode_consumer()
```

它想回答的不是“哪个 attention implementation 最快”，而是：

> producer 把 KV 写成 NHD/HND 时的局部选择，和后面 decode consumer 真正读取这些 KV 后的全 edge 选择，是否会冲突？

---

## 2. NHD 和 HND 在这里具体是什么

源码里 physical address 明确定义成两种 paged layout。

### NHD

物理顺序近似：

\[
[B,\ page,\ token\_within\_page,\ H,\ D]
\]

源码：

```cpp
if (layout == NHD) {
    return (((b * pages + page) * page_size + within)
             * heads + head) * dim + d;
}
```

也就是：

```text
page
 └─ token
     └─ head
         └─ D
```

对于同一个 token：

```text
token0:
    head0 D...
    head1 D...
    head2 D...
token1:
    head0 D...
    ...
```

---

### HND

物理顺序：

\[
[B,\ page,\ H,\ token\_within\_page,\ D]
\]

源码：

```cpp
return (((b * pages + page) * heads + head)
         * page_size + within) * dim + d;
```

也就是：

```text
page
 └─ head
     └─ token
         └─ D
```

因此同一个 head 的一串 token 是连续的：

```text
head0:
    token0 D...
    token1 D...
    token2 D...
```

这也是为什么实验中的 `decode_consumer` 是一个 **head-local scan**。

---

# 3. Producer 到底做什么

`producer_write()` 模拟 KV append。

它把当前 `q_len` 个新 token 写进已有 KV cache 的最后：

```cpp
int first = kv_len - q_len;
```

然后：

```cpp
cache[cache_offset(layout,...)] = input[i];
```

非常重要的是，代码专门强调：

```cpp
// Genuine direct emission:
// each layout receives the logical producer value
// at its own physical address.
// No transpose is hidden inside this kernel.
```

所以：

### NHD producer

直接生成：

\[
\text{logical KV}
\rightarrow
\text{NHD physical storage}
\]

### HND producer

直接生成：

\[
\text{logical KV}
\rightarrow
\text{HND physical storage}
\]

不是：

```text
先生成 NHD
↓
transpose
↓
HND
```

因此它真的能测：

\[
C_P(NHD)
\]

和

\[
C_P(HND)
\]

哪个 producer-local 更便宜。

---

# 4. Consumer 到底做什么

`decode_consumer()` 模拟 decode 时：

> 一个 query head 对应一个 KV head，然后顺着这个 KV head 扫描整个 KV sequence。

映射：

```cpp
int group = hq / hkv;
int kvhead = qhead / group;
```

因此 GQA：

\[
H_q/H_{kv}
\]

越大，一个 KV head 被更多 query heads 使用。

每个 consumer block 处理某个：

```text
batch
× query head
× sequence chunk
```

然后扫描：

\[
KV[token,\ kvhead,\ d]
\]

最后做 reduction：

```cpp
sum += ...
```

所以这个 consumer 特意形成了：

\[
\boxed{\text{固定 head，沿 token 轴扫描}}
\]

对于这种 access pattern：

```text
HND:
head0 → token0 → token1 → token2 ...
```

通常比：

```text
NHD:
token0 → head0
token1 → head0
token2 → head0
```

更有利于 head-local locality。

源码注释甚至直接说明：

```cpp
// Head-local paged decode proxy.
// HND gives contiguous [within,dim] runs;
// NHD jumps over Hkv for adjacent tokens.
```

所以这个 benchmark 实际上是在制造一个很清楚的 tension：

\[
\boxed{
Producer\ preference
\quad vs \quad
Consumer\ preference
}
\]

---

# 5. RQ1 最核心：一次 case 会测 5 条路径

这是理解整个实验最关键的地方。

每一个 shape 都测：

\[
\boxed{
NN,\ NH\text{-copy},\ HH\text{-native},\ HN\text{-copy},\ NH\text{-view}
}
\]

---

## NN

```text
Producer
   │
   ▼
 NHD
   │
   ▼
NHD consumer
```

即：

\[
T_{NN}
=
T_P(NHD)+T_C(NHD)
\]

没有 conversion。

源码：

```cpp
writer(NHD, nhd);
reader(NHD, nhd);
```

---

## NH-copy

```text
Producer
   │
   ▼
 NHD
   │
   ▼
NHD → HND materialization
   │
   ▼
HND consumer
```

即：

\[
T_{NH-copy}
=
T_P(NHD)
+
T_{N\rightarrow H}
+
T_C(HND)
\]

源码：

```cpp
writer(NHD, nhd);
converter(NHD, HND, nhd, repair);
reader(HND, repair);
```

这条路径回答：

> consumer 虽然更喜欢 HND，但是为了满足它，要不要值得付一次 conversion？

---

## HH-native

```text
Producer
   │
   ▼
 HND
   │
   ▼
HND consumer
```

即：

\[
T_{HH}
=
T_P(HND)+T_C(HND)
\]

这里 producer **直接产生 HND**，没有 transpose。

源码：

```cpp
writer(HND, hnd);
reader(HND, hnd);
```

这条非常重要，因为它测试：

> 与其 NHD producer + conversion，为什么不直接让 producer 原生地产出 consumer 喜欢的 HND？

---

## HN-copy

反方向：

```text
Producer
   │
   ▼
 HND
   │
   ▼
HND → NHD
   │
   ▼
NHD consumer
```

即：

\[
T_{HN-copy}
=
T_P(HND)
+
T_{H\rightarrow N}
+
T_C(NHD)
\]

它保证实验是双向的，而不是只研究：

```text
NHD → HND
```

---

## NH-view

概念上表示：

```text
NHD physical storage
       │
       │ no materialization
       ▼
stride-polymorphic consumer
```

没有真正复制数据：

\[
T_{view}
\simeq
T_P(NHD)+T_C(\text{NHD strides})
\]

源码实际实现是：

```cpp
writer(NHD, nhd);
reader(NHD, nhd);
```

所以要非常准确地理解这一条：

> 它并不是把 NHD storage magically 变成了真正的 HND 连续访问；它测试的是“consumer 接受原有 stride，因此完全避免 conversion”的路径。

代码自己也明确写：

```cpp
// it has no materialization
// and retains the NHD access cost.
```

因此 `NH-view` 更准确地理解成：

\[
\boxed{\text{zero-copy compatibility experiment}}
\]

而不是“免费得到 HND locality”。

---

# 6. 实验不是只测完整 edge，它先把三个成本拆开

对于每个 case，代码首先单独测：

### Producer

\[
P_N=T_P(NHD)
\]

\[
P_H=T_P(HND)
\]

代码：

```cpp
p_nhd = measure(writer(NHD,...))
p_hnd = measure(writer(HND,...))
```

---

### Consumer

\[
C_N=T_C(NHD)
\]

\[
C_H=T_C(HND)
\]

代码：

```cpp
c_nhd = measure(reader(NHD,...))
c_hnd = measure(reader(HND,...))
```

---

### Conversion

\[
M_{NH}=T_{NHD\rightarrow HND}
\]

\[
M_{HN}=T_{HND\rightarrow NHD}
\]

代码：

```cpp
n2h = measure(converter(NHD,HND,...))
h2n = measure(converter(HND,NHD,...))
```

然后才直接测整个 pipeline：

\[
T_{NN},T_{NH},T_{HH},T_{HN},T_{view}
\]

注意这里不是简单做：

\[
P+M+C
\]

的后处理求和，而是又真正 launch 完整 sequence 来计时。

例如：

```cpp
Timing nh = measure([&](cudaStream_t s) {
    writer(NHD,...);
    converter(NHD,HND,...);
    reader(HND,...);
});
```

所以 `edge_us` 是真实测出来的整个 CUDA launch sequence latency。

这点设计是合理的，因为：

\[
T(P)+T(M)+T(C)
\]

不一定严格等于：

\[
T(P\rightarrow M\rightarrow C)
\]

---

# 7. Producer-local winner 怎么得到

在每个独立 process 中，分析器取：

\[
P_N
\]

和：

\[
P_H
\]

然后：

\[
\boxed{
L_P^*=\arg\min_{L\in\{NHD,HND\}}P(L)
}
\]

代码：

```python
producer_winner = min(producer_cost,
                      key=producer_cost.get)
```

假设结果是：

```text
NHD = 1.5 us
HND = 2.0 us
```

那么：

\[
L_P^*=NHD
\]

如果你只做 producer-local optimization，就会选择 NHD。

---

# 8. Edge winner 怎么得到

与此同时，它在五种策略中直接找：

\[
S^*
=
\arg\min
\{
T_{NN},
T_{NH-copy},
T_{HH},
T_{HN-copy},
T_{view}
\}
\]

代码：

```python
edge_winner = min(
    by_strategy,
    key=lambda name:
        float(by_strategy[name]["edge_us"])
)
```

例如：

```text
NN          = 10 us
NH-copy     = 8 us
HH-native   = 5 us
HN-copy     = 12 us
NH-view     = 7 us
```

那么：

```text
producer winner = NHD
edge winner     = HH-native
```

也就是：

\[
\boxed{
L_P^*=NHD
}
\]

但是：

\[
\boxed{
L_E^*=HND
}
\]

这就是 RQ1 要找的：

\[
\boxed{\text{rank inversion}}
\]

---

# 9. Regret 是怎么计算的

如果 producer winner 是 NHD，那么 producer-local policy 对应的完整 edge baseline 是：

```text
NN
```

如果 producer winner 是 HND，则 baseline 是：

```text
HH-native
```

代码：

```python
producer_direct = (
    "NN"
    if producer_winner == "NHD"
    else "HH-native"
)
```

然后：

\[
T_{\text{baseline}}
=
T(\text{producer-local direct strategy})
\]

而 oracle：

\[
T_{\text{oracle}}
=
\min_s T(s)
\]

regret：

\[
R
=
T_{\text{baseline}}
-
T_{\text{oracle}}
\]

代码：

```python
regret_us = baseline - oracle
```

仓库中的 percentage 定义实际是：

\[
\boxed{
R_{\%}
=
\frac{
\operatorname{mean}(T_{\text{baseline}}-T_{\text{oracle}})
}{
\operatorname{mean}(T_{\text{oracle}})
}
\times100
}
\]

注意分母是 **oracle**，不是 baseline。

---

# 10. 不是一次测量就宣布 rank inversion

这个实验比较严格的一点是：

\[
\boxed{\ge 5\text{ independent processes}}
\]

默认 runner：

```bash
V10_PROCESS_REPETITIONS=5
```

每一个 process 都重新运行整套：

```text
producer NHD
producer HND
consumer NHD
consumer HND
N→H
H→N
NN
NH-copy
HH-native
HN-copy
NH-view
```

例如：

```text
process 1
process 2
process 3
process 4
process 5
```

分析器要求 producer winner 在进程间一致：

```python
stable_producer = len(producer_winners) == 1
```

同时 edge winner 也必须一致：

```python
stable_edge = len(edge_winners) == 1
```

所以不是：

```text
3 次 NN 快
2 次 HH 快
→ 随便取平均
```

而是 winner 必须稳定。

---

# 11. 单个 process 内部又怎么计时

full mode 默认：

```text
warmup = 8
iterations = 30
```

每一种 measurement：

先 warmup：

```cpp
for (...) launch(stream);
cudaStreamSynchronize(stream);
```

然后 30 次 CUDA event：

```text
start
kernel / pipeline
stop
```

收集 30 个 latency。

最后取：

\[
p50
\]

和：

\[
p95
\]

代码：

```cpp
return {
    quantile(values, .5f),
    quantile(values, .95f)
};
```

因此有两层重复：

```text
一个 process:
    8 warmup
    + 30 timing iterations
    → p50

整个 experiment:
    ≥5 independent processes
    → statistical adjudication
```

---

# 12. 真正判定“positive rank inversion”需要同时通过 7 个条件

这是 RQ1 最关键的 statistical gate。

源码 [`analyze_v10_rq1.py`](https://github.com/idoOwhd/layout_Test/blob/main/analyze_v10_rq1.py)：

```python
positive = all((
    len(per_run) >= 5,
    stable_producer,
    stable_edge,
    producer_winner != edge_producer,
    ci_low is not None and ci_low > 0,
    regret_pct is not None
        and regret_pct > epsilon_pct,
    correctness,
))
```

翻成数学语言就是：

### 条件 1：至少五个独立进程

\[
N_{\rm proc}\ge5
\]

### 条件 2：producer winner 稳定

\[
L_{P,1}^*
=
L_{P,2}^*
=
\cdots
\]

### 条件 3：edge winner 稳定

\[
S_{E,1}^*
=
S_{E,2}^*
=
\cdots
\]

### 条件 4：edge winner 使用的 producer layout 不同于 producer-local winner

\[
L_P^*
\neq
L_{\text{producer of edge oracle}}^*
\]

这才是真正的“局部最优和 edge 最优冲突”。

### 条件 5：bootstrap 95% CI 下界 > 0

它对五次 process 的：

\[
R_i=T_{\text{baseline},i}-T_{\text{oracle},i}
\]

做 10,000 次 bootstrap。

要求：

\[
\boxed{
CI_{95\%,low}(R)>0
}
\]

也就是说收益不能只是随机波动。

### 条件 6：还必须超过 practical-effect threshold

阈值不是简单固定 3%。

代码是：

\[
\boxed{
\epsilon
=
\max
\left(
1\%,
3\times CV_{\rm baseline}
\right)
}
\]

其中：

\[
CV=\frac{\sigma}{\mu}
\]

要求：

\[
R_\%>\epsilon
\]

所以如果 baseline 很稳定：

```text
CV = 0.2%
3CV = 0.6%
```

则：

\[
\epsilon=1\%
\]

如果噪声较高：

```text
CV = 2%
```

则：

\[
\epsilon=6\%
\]

必须收益 >6% 才认。

这比简单“谁快 0.1 μs 就算 winner”严格很多。

### 条件 7：correctness 必须过

代码构造逻辑等价的 NHD/HND cache，然后分别跑 consumer：

\[
Y_{NHD}
\]

和：

\[
Y_{HND}
\]

计算：

\[
\max_i |Y_{NHD,i}-Y_{HND,i}|
\]

要求：

\[
\boxed{\le10^{-3}}
\]

---

# 13. Stage 0：先做 negative control

仓库不是一上来就找正例。

先构造：

```text
B=1
Q=1
KV=512
Hq=8
Hkv=8
D=128
page=64
BF16
```

这里：

\[
H_q/H_{kv}=1
\]

目的就是测试一个预期 layout effect 比较弱的 regime。

要求：

```text
NN 和 HH-native 的差异
≤ noise/practical threshold
```

同时不能成为 positive rank inversion。

这就是：

\[
H1.NEG
\]

它的作用是：

> 如果连这种 layout-insensitive control 都到处产生“rank inversion”，说明你的计时系统本身不可信。

---

# 14. Stage 3：真正的核心 RQ1 实验

核心中心点固定：

\[
B=8,\quad Q=1,\quad KV=2048,\quad D=128,\quad page=64
\]

改变 GQA ratio。

### ratio = 1

```text
Hq=8
Hkv=8
```

\[
H_q/H_{kv}=1
\]

### ratio = 8

```text
Hq=32
Hkv=4
```

\[
H_q/H_{kv}=8
\]

### ratio = 32

```text
Hq=32
Hkv=1
```

\[
H_q/H_{kv}=32
\]

所以 Stage 3 在问：

> 随着一个 KV head 被越来越多 query heads 重复消费，consumer locality 的价值变大以后，是否足以改变全 edge 的最优 layout？

这是很合理的控制变量实验。

---

# 15. H1.2：显式测 reuse

仓库额外固定：

```text
B=8
Q=1
KV=2048
Hq=32
Hkv=4
```

只改变：

\[
reuse\in\{1,4,16,64\}
\]

`reader()` 里面：

```cpp
for (int r = 0; r < c.reuse; ++r) {
    decode_consumer(...);
}
```

因此一次 producer/cache layout 被 consumer 重复读取：

```text
producer once
      │
      ▼
    cache
      │
      ├── consumer
      ├── consumer
      ├── consumer
      └── ...
```

目的就是测试：

\[
\boxed{
\text{conversion/native-layout cost}
\quad vs\quad
\text{consumer reuse benefit}
}
\]

直觉上：

\[
T_{NH}(R)
=
P_N+M_{NH}+R C_H
\]

而：

\[
T_{NN}(R)
=
P_N+R C_N
\]

随着 \(R\) 增大：

\[
R(C_N-C_H)
\]

可能逐渐超过：

\[
M_{NH}
\]

这就是 layout conversion 的 amortization。

H1.2 判据也不是只看某个点，而是要求 reuse 增大后 regret 总体增长。

---

# 16. H1.3：zero-copy view 是否比 materialization 更好

直接比较：

\[
NH\text{-view}
\]

和：

\[
NH\text{-copy}
\]

分析器要求：

\[
T_{view}<0.99T_{copy}
\]

也就是至少约 1% 收益。

它在回答：

> 如果 consumer 可以 stride-polymorphic 地接受现有 storage，那么是不是根本不应该 materialize 一个新 layout？

---

# 17. H1.4：直接原生产出 consumer layout 是否比转换更好

比较：

\[
HH\text{-native}
\]

vs

\[
NH\text{-copy}
\]

即：

```text
方案 A:
NHD producer
→ NHD→HND conversion
→ HND consumer

方案 B:
HND producer
→ HND consumer
```

如果：

\[
T_{HH}<0.99T_{NH-copy}
\]

则认为存在 native-emission benefit。

这个假设非常重要，因为它把问题从：

> “要不要转换？”

提升成：

> “producer 本来是不是就应该生成 downstream 想要的 layout？”

---

# 18. Stage 4：只有 RQ1 核心现象先成立，才做 crossover sweep

这是这个仓库设计得比较好的地方。

它不是无条件 sweep 一堆 shape。

必须先有至少一个 Stage-3 BF16 core case 满足完整 rank-inversion 判据：

\[
\boxed{
H1.1\text{ core positive}
}
\]

才：

```text
Stage4_GO = true
```

否则：

```text
STOP
```

Stage 4 才开始扫描：

\[
B\in\{1,8,32,64\}
\]

\[
KV\in\{128,512,2048,8192,32768\}
\]

\[
GQA\ ratio\in\{1,4,8,16,32\}
\]

\[
Q\in\{1,4,8\}
\]

\[
page\in\{16,64,128\}
\]

一次只主要改变一个轴。

目的是找：

\[
\boxed{
\text{crossover surface}
}
\]

即回答：

> 在什么 batch / KV length / GQA / q_len / page size 下，producer-local 与 edge-global 开始分离？

而不是只报告：

```text
“某个 shape 下 HND 快 8%”
```

---

# 19. 还加入了真实模型 decode shape

`build_v10_rq1_cases.py` 会从真实的 LLM manifest 中选 attention case。

不是把模型完整跑一遍，而是：

> 从真实模型 contract 中提取真实的 KV length、query heads、KV heads、head_dim，然后放进同一个受控 CUDA edge benchmark。

也就是：

```text
real model
   │
   └── extract shape/contract
             │
             ▼
controlled producer→KV→consumer benchmark
```

因此这些属于：

\[
\boxed{\text{real-world-derived controlled cases}}
\]

而不是完整模型 inference。

---

# 20. Stage 5 才加入更广泛的 external validity

Stage 3 通过以后，runner 才构造八类 LLM subgraph：

```text
GQA
sliding attention
sparse attention
MLA
SwiGLU
MoE
Mamba2
linear attention
```

然后运行：

```text
llm_boundary_layout_sweep.py
benchmark_real_world.py
```

以及可选的：

```text
vLLM
SGLang
```

native serving。

但是仓库非常明确：

> 这些 Stage-5 结果不能替代 Stage-3 的严格 causal decomposition。

所以证据层级实际上是：

```text
                    L-RQ1
                      │
          ┌───────────┴───────────┐
          │                       │
 controlled causal evidence   external validity
          │                       │
CUDA producer→boundary→consumer   real shapes
          │                       Triton/PyTorch
          │                       vLLM/SGLang
          │
      核心证明
```

这是正确的解读。

---

# 21. 因此，RQ1 整个实验流程可以压缩成这张逻辑图

```text
             Real / controlled shape
                     │
                     ▼
        ┌────────────────────────┐
        │ Direct KV producer     │
        │                        │
        │   P(NHD)    P(HND)     │
        └──────────┬─────────────┘
                   │
          producer-local winner
                   │
                   ▼
          Lp* = argmin P(L)
                   │
                   │
       ┌───────────┴──────────────────────────────┐
       │                                          │
       ▼                                          ▼
 NHD physical                                HND physical
       │                                          │
       ├─────────────┐                 ┌──────────┤
       │             │                 │          │
       ▼             ▼                 ▼          ▼
      NN         N→H copy          H→N copy   HH-native
       │             │                 │          │
       │             ▼                 ▼          │
       │            HND               NHD         │
       │          consumer          consumer      │
       │                                          │
       └──────── NH-view / zero-copy ─────────────┘
                          │
                          ▼
          compare five complete edge times
                          │
                          ▼
             Se* = argmin Tedge(strategy)
                          │
                          ▼
       Is producer layout of Se* different from Lp* ?
                          │
                 ┌────────┴────────┐
                 │                 │
                no                yes
                 │                 │
                 ▼                 ▼
           no inversion       ≥5 processes?
                                  │
                           stable winners?
                                  │
                          bootstrap CI > 0?
                                  │
                     regret > max(1%,3×CV)?
                                  │
                            correctness?
                                  │
                                  ▼
                        positive inversion
```

---

# 22. 我认为最需要注意的三个实验边界

第一，**这不是完整 Attention**。核心 direct evidence 只证明：

\[
\boxed{
KV\ producer
\rightarrow
layout
\rightarrow
head\text{-}local\ KV\ scan
}
\]

范围内局部最优可能不同于 edge 最优。

它不能直接推出：

\[
\boxed{
\text{完整 FlashAttention / vLLM / SGLang attention}
}
\]

一定发生同样 inversion。

仓库本身也明确把它标成：

```text
cuda-reference
runtime_empirical_controlled
```

而不是 vLLM/SGLang/FlashInfer native evidence。

第二，consumer 是人为设计的 **head-local paged decode proxy**。它非常适合研究 NHD/HND tension，但本质上是有针对性的 mechanism benchmark：

\[
HND
\]

天然对：

\[
[\text{token},D]\text{ within same head}
\]

更连续。

所以它证明的是一个因果机制：

> downstream access order 能使 producer-local layout ranking 失效。

而不是说：

> HND 普遍优于 NHD。

第三，当前 correctness gate 主要比较的是：

\[
consumer(NHD)
\]

与：

\[
consumer(HND)
\]

逻辑输出是否一致。`correctness_pass` 随后被复用到五种 strategy 行中。也就是说，从严格代码审计角度，它没有在 timed edge 中分别把 `NH-copy`、`HN-copy`、`NH-view` 每一条 pipeline 的最终输出再逐条与 reference 比较。`convert_cache()` 从索引逻辑上看保持语义，但如果要做到论文级最严格实验，我会建议把五条 edge 都分别做输出 correctness check。

---

所以一句话概括这个仓库的 RQ1 实验：

\[
\boxed{
\text{它先独立测 producer 的 NHD/HND 局部排名，}
}
\]

\[
\boxed{
\text{再真实执行 NN/NH-copy/HH-native/HN-copy/NH-view 五条完整 edge，}
}
\]

\[
\boxed{
\text{比较 producer-local policy 与 exhaustive edge oracle，}
}
\]

\[
\boxed{
\text{并要求 ≥5 个独立进程、winner 稳定、bootstrap CI>0、}
}
\]

\[
\boxed{
\text{regret}>\max(1\%,3CV)\text{、correctness 通过，才认定 RQ1 的 rank inversion。}
}
\]

因此它真正试图证明的是：**layout 不能只看单个 producer/kernel 的局部成本；当 consumer locality、conversion、zero-copy legality 和 reuse 一起进入总账以后，最优决策可能改变。**

---

我先说明一个限制：我已经按你指定的 GitHub 连接器去读取 `idoOwhd/layout_Test`，但当前连接器对这个仓库返回 404，而且在已授权仓库里也搜不到它，所以我现在不能假装已经逐行读到了该 GitHub 仓库的最新源码。公开网页搜索也没有索引到这个仓库。

不过，这个 Project 里保留了这套仓库实验的 v10/v13 完整分析与代码审计结果，其中明确记录了 RQ1 的实验对象、统计口径、框架证据边界和最终计数。下面我先基于这些已有实验材料，把 RQ1 的实验设计完整拆开。等该仓库授权给当前 GitHub App 后，我还可以继续精确追到“哪个 `.cu/.py` 文件、哪个函数、哪几行代码”来逐行验证。

RQ1 的核心其实不是“哪个 layout 更快”，而是问：

> **如果只看 producer 自己，选出来的最优 layout，是否可能不是整个 producer → consumer 边界的最优 layout？**

也就是研究：

\[
\boxed{
\arg\min_L T_{\text{producer}}(L)
\stackrel{?}{=}
\arg\min_L T_{\text{producer}\rightarrow\text{consumer}}(L)
}
\]

当前严格报告把它表述为：

> `v10-RQ1 / L-RQ1：局部最优与 edge/subgraph 最优是否冲突`

并明确把 CUDA reference 视为直接因果证据，Triton producer+consumer pipeline 作为独立复核。fileciteturn4file0L23-L27

---

## 1. RQ1 实际比较的是什么

最简单地说，它人为构造一条：

```text
producer
   │
   │ tensor layout
   ▼
consumer
```

然后允许中间 tensor 有不同物理布局，例如：

```text
NHD
HND
```

这里：

```text
NHD = [token, head, head_dim]
HND = [head, token, head_dim]
```

RQ1 分两层做选择。

第一层，只看 producer：

\[
T_P(NHD),\qquad T_P(HND)
\]

于是得到：

\[
L_{\text{producer}}
=
\arg\min_{L\in\{NHD,HND\}} T_P(L)
\]

这模拟很多编译器/框架的一种局部决策：

> “我的这个 kernel 用 HND 写最快，所以就输出 HND。”

但是 RQ1 接着问：

> consumer 真的喜欢这个布局吗？

所以第二层会测完整 edge：

\[
T_{\text{edge}}(L)
=
T_P(L)
+
T_C(L)
+
T_{\text{conversion/repair}}(L)
\]

然后得到：

\[
L_{\text{edge}}
=
\arg\min_L T_{\text{edge}}(L)
\]

如果：

\[
\boxed{
L_{\text{producer}}\neq L_{\text{edge}}
}
\]

就出现了所谓的 **rank inversion / ranking inversion（排序反转）**。

这就是 RQ1 真正要抓的现象。

---

## 2. 仓库中不是只运行一个 kernel，而是拆成 primitive 和完整 edge

这一点非常重要。

从严格结果里能看出，它至少独立考察了三个 primitive：

```text
kv_write
decode_head_scan
token_major_scan
```

并统计 NHD/HND 在这些 primitive 上的 robust winner。报告中给出的 3% gate 后结果是：

```text
kv_write:
    tie 214
    NHD 17
    HND 25

decode_head_scan:
    tie 204
    HND 32
    NHD 20

token_major_scan:
    tie 215
    NHD 30
    HND 11
```

这实际上告诉我们实验的基本结构是：

```text
                    primitive measurement
                 ┌──────────────────────────┐
                 │                          │
            producer                     consumer
            kv_write              decode_head_scan
                 │                          │
              NHD/HND                   NHD/HND
                 └───────────┬──────────────┘
                             │
                             ▼
                  producer → consumer
                     edge pipeline
                             │
                             ▼
                  compare edge winner
                  with producer winner
```

报告特别强调，大量 raw winner 改变其实位于微秒级噪声范围，因此不能看到 NHD/HND 名义 winner 不同就宣布存在 RQ1 效应。fileciteturn4file0L23-L27

---

# 3. RQ1 中最关键的几条执行路径

根据你之前这套实验的 RQ1 方法定义，它不是简单做：

```text
NHD vs HND
```

而是把 producer layout、consumer layout 和 conversion 分开组合。

典型的五条路径是：

| Path | Producer | 中间操作 | Consumer |
|---|---|---|---|
| `NN` | NHD | 无转换 | NHD |
| `NH-copy` | NHD | NHD→HND copy | HND |
| `HH-native` | HND | 无转换 | HND |
| `HN-copy` | HND | HND→NHD copy | NHD |
| `NH-view` | NHD storage | stride/view | HND-style consumer |

可以把它理解成下面这张逻辑图。

```text
                         Same logical tensor
                                │
             ┌──────────────────┴──────────────────┐
             │                                     │
         Producer NHD                         Producer HND
             │                                     │
       ┌─────┼─────────┐                      ┌────┴───────┐
       │     │         │                      │            │
       │     │         │                      │            │
      NN  NH-copy   NH-view              HH-native     HN-copy
       │     │         │                      │            │
       │   transpose/  │                      │         transpose/
       │     copy      │                      │            copy
       │     │         │                      │            │
       ▼     ▼         ▼                      ▼            ▼
    NHD C   HND C   stride-aware C          HND C        NHD C
```

这样才能区分四种完全不同的问题：

```text
① producer 本身喜欢哪个 layout？
② consumer 本身喜欢哪个 layout？
③ 两者不一致时 conversion 贵不贵？
④ 名称不同的 layout 是否其实能 zero-copy view？
```

如果只跑：

```text
consumer(NHD)
consumer(HND)
```

其实回答不了 RQ1。

因为那只能证明：

> consumer 对 layout 敏感。

RQ1 要证明的是：

> **producer-local 最优决策在完整 edge 上产生 regret。**

---

# 4. 最关键的实验逻辑：先制造一个“局部策略”

假设某个 case 测出来：

```text
producer:
NHD = 8.0 μs
HND = 7.0 μs
```

于是 producer-local policy 会选择：

\[
L_P=HND
\]

因为：

\[
7<8
\]

这就是所谓：

```text
producer-local winner = HND
```

但完整 edge 可能是：

```text
HND producer + HND consumer = 7 + 10 = 17 μs

NHD producer + NHD consumer = 8 + 6 = 14 μs
```

于是：

```text
producer-local decision: HND
global edge oracle:       NHD
```

虽然 producer 因 HND 节省了：

\[
1\mu s
\]

但是 consumer 多花了：

\[
4\mu s
\]

最后整个 edge 多花：

\[
3\mu s
\]

所以：

\[
\boxed{
\text{local optimum}
\neq
\text{edge optimum}
}
\]

这才是 RQ1 的正例。

---

# 5. RQ1 中的 `regret` 是核心指标

实验并不只是数：

```text
winner 是否不同
```

因为 winner 改变可能只是：

```text
10.00 μs
vs
10.01 μs
```

几乎没有科研意义。

真正关心的是：

\[
\text{Regret}
=
\frac{
T_{\text{edge}}(
L_{\text{producer-local}}
)
}{
\min_LT_{\text{edge}}(L)
}
\]

例如：

```text
producer-local policy = 11.25 μs
oracle edge           = 10.00 μs
```

则：

\[
Regret=1.125
\]

也就是说局部策略让整个 edge 慢了：

\[
12.5\%
\]

这也是为什么报告中的 RQ1 最终结果写成：

> 最大 `regret = 1.125×`

而不是只说“有多少次 winner 翻转”。fileciteturn4file0L23-L27

---

# 6. 当前版本为什么设置 3% practical-effect gate

这一版分析做了一个很重要的修正：

\[
\boxed{\text{difference}<3\%\Rightarrow tie}
\]

也就是说：

```text
NHD = 10.00 μs
HND = 10.15 μs
```

虽然 raw winner 是 NHD，但：

\[
10.15/10.00=1.015
\]

只有 1.5%，所以严格分析不会称：

```text
NHD wins
```

而会记：

```text
tie
```

整个 full 实验明确声明，所有 winner 结论使用预声明的 3% practical-effect gate，小于 3% 的差异视为 tie。fileciteturn4file0L15-L19

这是 RQ1 很关键的一点，因为 GPU 微 kernel 很容易出现：

```text
raw winner flipping
```

但并不代表：

```text
scientifically meaningful rank inversion
```

---

# 7. 它不是取一次最快值，而是做 5 个独立进程重复

当前严格统计方式是：

```text
case
 × experiment
 × candidate
 × causal-axis
```

每组 CUDA 数据有：

\[
5
\]

次独立进程重复。

然后：

\[
\boxed{
T =
\operatorname{median}
(
p50_1,p50_2,p50_3,p50_4,p50_5
)
}
\]

而不是：

\[
\min(p50_1,\dots,p50_5)
\]

报告明确说明：

> CUDA 的 5 次独立进程结果先按 `case × experiment × candidate × causal-axis` 取 p50 中位数，再比较候选；不选择最快的一次 repetition。fileciteturn4file0L13-L20

这一点很重要。

因为如果取 fastest repetition：

```text
NHD:
8.0 8.2 8.1 7.3 8.1

HND:
7.9 7.8 7.9 8.0 7.9
```

你可能错误得到：

```text
NHD = 7.3
HND = 7.8

NHD wins
```

但实际上：

```text
median(NHD) ≈ 8.1
median(HND) ≈ 7.9

HND wins
```

旧分析确实存在“raw winner / fastest repetition”可能夸大微秒级噪声的问题；后来的审计改成了 repetition median + 3% gate。fileciteturn4file0L125-L131

---

# 8. 一个 `case` 究竟是什么

这套 RQ1 不是只测试一个模型 shape。

full run 中最终有：

\[
256
\]

个独立 attention case。

整个实验集更大，但报告特别强调：

```text
1024 contracts
≠
1024 independent models
```

很多 shape 是同一个 parent model 的受控 counterfactual，所以不能把所有 contract 当成独立样本做统计。fileciteturn4file0L15-L20

大致相当于从真实模型配置里抽出：

```text
B
Q length
KV length
Hq
Hkv
head_dim
phase
...
```

然后固定逻辑计算，只改变：

```text
physical layout / stride / conversion path
```

这样才能把性能变化归因给 layout，而不是：

```text
模型变了
计算量变了
kernel 算法变了
```

---

# 9. 它如何判断真正的 RQ1 positive case

可以把当前版本理解成下面的流程。

```text
Case i
  │
  ├── Benchmark producer(NHD)
  ├── Benchmark producer(HND)
  │
  ▼
Producer-local winner
       L_P
        │
        │
        ├──────────────┐
        │              │
        ▼              ▼
 edge using NHD    edge using HND
        │              │
        └───────┬──────┘
                ▼
          Edge oracle L_E
                │
                ▼
          Compare L_P vs L_E
                │
          ┌─────┴──────┐
          │            │
       same        different
          │            │
         no        calculate regret
                       │
                       ▼
                 regret < 3% ?
                  /          \
                yes           no
                │              │
               tie        meaningful
                          RQ1 positive
```

所以：

\[
L_P\neq L_E
\]

只是第一层。

真正严格的条件更接近：

\[
\boxed{
L_P\neq L_E
\quad\land\quad
\frac{T(L_P)}{T(L_E)}\ge1.03
}
\]

---

# 10. 这就是为什么报告里会同时出现 99 和 18

这是理解 RQ1 最重要的数字。

当前 strict 分析说：

\[
256
\]

个 attention cases 里：

### raw winner 不一致

\[
\boxed{99/256}
\]

也就是有 99 个 case：

```text
producer-local winner
≠
edge raw winner
```

但是加入 3% practical-effect gate 后：

\[
\boxed{18/256}
\]

才是真正：

```text
producer-local decision
causes ≥3% edge regret
```

而最大值：

\[
\boxed{1.125\times}
\]

即约 12.5% regret。fileciteturn4file0L23-L27

所以千万不能把这个实验写成：

> “99/256 个 case 证明局部布局选择错误。”

更准确的是：

> 99/256 出现 raw ordering mismatch，但只有 18/256 在预声明的 3% practical-effect gate 后仍构成有意义的 edge regret。

这也是当前报告给 H1.1 的判定：

> `支持但效应稀疏`，依据为 `edge regret ≥ 3% = 18/256`。fileciteturn4file0L75-L82

---

# 11. `consumer-only mismatch = 19/256` 又是什么意思

报告还给了：

\[
19/256
\]

个 consumer-only meaningful mismatch。

这和 edge mismatch 不完全一样。

consumer-only 比较的是：

\[
T_C(NHD)
\quad vs \quad
T_C(HND)
\]

看：

```text
producer 喜欢的 layout
```

是否也是：

```text
consumer 喜欢的 layout
```

例如：

```text
Producer:
NHD 7 μs
HND 8 μs
→ producer chooses NHD

Consumer:
NHD 12 μs
HND 8 μs
→ consumer chooses HND
```

这是：

```text
local preference conflict
```

但还不能直接说：

```text
应该转换为 HND
```

因为 conversion 可能要：

```text
6 μs
```

那么：

\[
6+(8)<12?
\]

不成立。

所以 RQ1 的层次其实是：

```text
Level 1
producer preference vs consumer preference

                ↓

Level 2
producer-local policy vs complete edge oracle

                ↓

Level 3
meaningful edge regret after 3% gate
```

真正最有说服力的是 Level 3。

---

# 12. 为什么还有 zero-copy 路径

这是这个实验设计比较好的地方。

假设逻辑上：

```text
producer = NHD
consumer wants HND
```

传统分析可能马上写：

```text
需要 transpose
```

但实际上：

```text
layout name mismatch
```

并不一定等于：

```text
physical conversion required
```

如果 consumer 是 stride-polymorphic：

```text
same storage
    │
    └── different stride interpretation
```

就可能：

\[
T_{\text{conversion}}\approx0
\]

因此实验还要加入：

```text
NH-view
```

之类 zero-copy candidate。

v13 中 H1.3 的结果明确记录了：

\[
41/1024
\]

个 zero-copy meaningful wins。fileciteturn4file0L75-L82

因此 RQ1 最终研究的实际上不只是：

```text
NHD vs HND
```

而是：

\[
\boxed{
\text{producer decision}
+
\text{physical representation}
+
\text{consumer requirement}
+
\text{repair/conversion}
}
\]

共同决定 edge cost。

---

# 13. 为什么还要做 Triton 实验

CUDA reference 是主要的 controlled experiment。

但只做 CUDA reference 会有一个问题：

> 会不会这个现象只是你自己写的 synthetic CUDA kernel 造成的？

所以又用 Triton 做独立复核。

严格证据表里写得很明确：

| framework | 对 RQ1 的证据 |
|---|---|
| CUDA control | `direct` |
| Triton | `direct pipeline` |
| TVM | `projected direct boundary` |
| CUTLASS/CuTe | `
