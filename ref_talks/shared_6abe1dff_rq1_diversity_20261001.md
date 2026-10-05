# RQ1 多样性讨论存档

来源：https://chatgpt.com/share/6abe1dff-8264-83ec-a1dc-8247f4819946

读取日期：2026-10-01。以下为共享页面中实际提取的讨论文本；GitHub PR/issue 的断言需要另行源码核验。

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
