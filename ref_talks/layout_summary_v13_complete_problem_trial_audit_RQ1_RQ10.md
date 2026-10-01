# layout_summary_v13_complete_problem_trial_audit_RQ1_RQ10.md

> **Date:** 2026-09-18  
> **Purpose:** re-audit every canonical layout RQ against the exact requested “problem trial” chain:
>
> **all related source rules → rule proxy → ignored variables → framework fallback/repair → modern Attention/GQA/MLA/MoE trigger → falsifiable hypotheses.**
>
> Base evidence: 115 frozen semantic units from v14/v3, represented as 209 RQ-mapped rows in v9.  
> Current-source supplements: **16** unique current facts (10 from v12 + 6 new in this audit).  
> Unified complete rule-chain rows after multi-RQ mapping: **258**.
>
> The purpose of this document is depth control. It does not add a new parent RQ.

# 0. Verdict

After the completeness audit and supplementation:

- **L-RQ1–L-RQ5 and L-RQ7–L-RQ10:** complete source/reasoning problem trials within the declared evidence scope.
- **L-RQ6:** complete source/reasoning chain, but scientifically **conditional on RQ4** demonstrating that the preferred persistent layout changes enough for migration to have a benefit side.
- Every rule-chain row now has a non-empty:
  `source fact / proxy / assumption / ignored variables / fallback-repair / trigger / hypothesis link / provenance`.
- No new independent causal mechanism requires L-RQ11.

“Complete” still does not mean repository-exhaustive forever or empirically true. It means the RQ is sufficiently grounded to be falsified by the registered experiment.

# 1. Overall completeness matrix

| RQ | Question | Verdict | Rule-chain rows | Direct rows | Direct stacks | Current supplements | Hypotheses | Unlinked positive H |
|---|---|---|---|---|---|---|---|---|
| L-RQ1 | Cross-edge layout rank inversion | COMPLETE_STRONG | 23 | 20 | 8 | 2 | H1.1,H1.2,H1.3,H1.4,H1.NEG | NONE |
| L-RQ2 | Optimal layout-domain granularity | COMPLETE_STRONG | 29 | 27 | 4 | 6 | H2.1,H2.2,H2.3,H2.4,H2.5 | NONE |
| L-RQ3 | Layout adaptation amortization | COMPLETE_STRONG | 17 | 17 | 6 | 2 | H3.1,H3.2,H3.3,H3.4,H3.5 | NONE |
| L-RQ4 | Optimal-layout stability across workload/hardware | COMPLETE_STRONG | 50 | 19 | 6 | 5 | H4.1,H4.2,H4.3,H4.4,H4.5 | NONE |
| L-RQ5 | Communication/placement-aware layout inversion | COMPLETE_STRONG | 18 | 13 | 3 | 4 | H5.1,H5.2,H5.3,H5.4,H5.5 | NONE |
| L-RQ6 | Persistent-layout reconfiguration threshold | COMPLETE_CONDITIONAL_ON_RQ4 | 13 | 13 | 5 | 4 | H6.1,H6.2,H6.3,H6.4,H6.5 | NONE |
| L-RQ7 | Layout equivalence and zero-copy transformability | COMPLETE_STRONG | 17 | 16 | 7 | 7 | H7.1,H7.2,H7.3,H7.4 | NONE |
| L-RQ8 | Coupled data–metadata co-layout | COMPLETE_STRONG | 23 | 21 | 6 | 7 | H8.1,H8.2,H8.3,H8.4,H8.5 | NONE |
| L-RQ9 | Layout-space expressiveness and search-space truncation | COMPLETE_STRONG | 52 | 34 | 9 | 5 | H9.1,H9.2,H9.3,H9.4,H9.5 | NONE |
| L-RQ10 | Paged/block layout granularity and allocator coupling | COMPLETE_STRONG | 16 | 16 | 4 | 7 | H10.1,H10.2,H10.3,H10.4,H10.5 | NONE |

# 2. Framework relevance rule

The phrase **“all related frameworks”** is interpreted by abstraction ownership, not by forcing every repository into every RQ.

Examples:

- FlashInfer is highly relevant to RQ1/RQ3/RQ7 but does not own whole-engine pool partitioning for RQ2.
- Triton/CUTLASS are central to RQ7/RQ8/RQ9 but do not own serving-time distributed KV placement for RQ5.
- MLIR/IREE/TVM provide representation/legality/search counterevidence, not persistent serving allocators.

An absent framework is therefore a gap only when that framework owns the decision variable under trial.

# 3. Newly added current-source facts in this audit

| ID | Stack | RQs | Source fact | Proxy | Ignored | Framework fallback/repair | Modern trigger | Pinned source |
|---|---|---|---|---|---|---|---|---|
| S13-01 | vLLM | L-RQ6,L-RQ2 | EngineCore writes the resolved cache layout into KVCacheConfig, and GPUModelRunner uses get_resolved_kv_cache_layout() while creating/profiling KV cache state. | resolved deployment-time KV cache layout and cache configuration | future workload-regime drift, migration cost/benefit, per-phase re-layout, remaining request horizon | reinitialize/reconfigure the engine/cache state rather than changing the allocated layout per request | Persistent paged MHA/MQA/GQA/MLA cache initialization; any serving mode using resolved KV layout before profiling/allocation | https://github.com/vllm-project/vllm/blob/017dced6a6fd3cf430e4686a47b354da1cadfbf5/vllm/v1/engine/core.py |
| S13-02 | FlashInfer | L-RQ6,L-RQ4,L-RQ10 | cuDNN paged decode buckets max_seq_len_kv and block-table width because these values are baked into the built graph; bucketing lets consecutive decode steps replay a cached graph instead of rebuilding whenever the longest sequence grows by a page. | bucketed maximum KV length and block-table width | future sequence-length distribution, memory overhead of larger buckets, alternative page sizes/layouts, cross-request graph reuse lifetime | build/rebuild a graph for another bucket or use a non-captured/alternate path | CUDA-graph paged decode; changing KV length/page-table width; GQA/MHA decode families supported by the selected backend | https://github.com/flashinfer-ai/flashinfer/blob/01045366e15df38e19e76050592f82e49b4cff64/flashinfer/decode.py |
| S13-03 | SGLang | L-RQ6,L-RQ8,L-RQ4 | DeepSeek-v4 TRT-LLM attention requires a uniform-FP8 KV pool and requires a persistent TRT-LLM semaphore buffer to be created outside CUDA-graph capture during eager warmup. | uniform-FP8 pool contract and graph-capture lifecycle | future dtype/layout preference changes, migration/recapture cost under drift, mixed-layout pool alternatives | fail/route away from the incompatible backend contract or perform eager warmup/resource creation outside capture | DeepSeek-v4/MLA-like TRT-LLM attention path, FP8 KV pool, CUDA graph capture | https://github.com/sgl-project/sglang/blob/191172fa742f8837423ca4e4be4919ebb52f1b51/python/sglang/srt/layers/attention/deepseek_v4_trtllm_backend.py |
| S13-04 | TensorRT-LLM | L-RQ5,L-RQ10,L-RQ2 | Helix context partitioning assigns token blocks round-robin across CP ranks using tokens_per_block and changes both prefill→decode KV movement and how KV grows during decode when generation context parallelism is enabled. | tokens_per_block, CP rank count, generation CP mode | live interconnect contention, unequal request lengths, alternative block/page sizes, replication alternatives, per-rank memory skew | use the existing non-Helix/non-CP KV-cache infrastructure when the Helix mode is not enabled | multi-million-token long-context decode, generation CP > 1, distributed paged KV transfer/growth | https://github.com/NVIDIA/TensorRT-LLM/blob/c5c839308f10004ec6fc1e5a0020d82e6c4d1402/docs/source/blogs/tech_blog/blog22_Helix_Parallelism_Scaling_Multi_Million_Token_Decoding_with_KV_Cache_Sharding.md |
| S13-05 | CUTLASS/CuTe | L-RQ8,L-RQ9,L-RQ7 | CUTLASS Sm1xxBlockScaledConfig explicitly constructs SFA/SFB scale-factor tensor layouts for block-scaled Blackwell GEMM; scale-factor layouts are a first-class physical representation rather than scalar metadata. | scale-vector size, operand/tile geometry, architecture-specific block-scaled MMA contract | upstream producer scale layout, cost of interleaving/repacking scales, downstream reuse, graph-level choice among alternative scale layouts | construct/use the required scale layout or choose a different legal GEMM/precision path | Blackwell NVFP4/MXFP block-scaled dense or sparse GEMM, including MoE/grouped-GEMM style consumers | https://github.com/NVIDIA/cutlass/blob/f614dc40e17fb3ddb1b7b474318b48a8d5d21a2c/include/cutlass/detail/sm100_blockscaled_layout.hpp |
| S13-06 | Triton | L-RQ8,L-RQ9,L-RQ4 | Triton kernels inspect the storage layout of MX scale tensors and contain architecture-specific scale-layout/unswizzle implementations for Hopper/CDNA4/GFX1250; block-scaled matmul treats scale layout as part of the physical kernel contract. | scale tensor storage layout, precision config, microblock size, target architecture | upstream producer cost, serving-level reuse/fanout, alternative compound layouts, conversion amortization | canonicalize/convert scale layout or use another supported precision/kernel path | MXFP4/NVFP4/block-scaled GEMM on NVIDIA/AMD targets; dense MLP and MoE/grouped GEMM consumers | https://github.com/triton-lang/triton/blob/fe5a423a1e7c754c011205cc35ad0dfd94e3d6fa/python/triton_kernels/triton_kernels/matmul_details/opt_flags.py |

# 4. Hypothesis linkage rule

A source rule does not “prove” a hypothesis. It supports the **premise/mechanism** that makes the hypothesis testable.

The `hypothesis_links` column means:

> “This rule supplies a causal premise, alternative, counterexample, or boundary condition used by that hypothesis.”

Negative controls such as `H1.NEG` and `H8.4` are allowed to have no positive source rule; they exist specifically to falsify over-generalization.



# 5. L-RQ1 — Cross-edge layout rank inversion

## 5.1 Trial status

**Verdict:** `COMPLETE_STRONG`

- Complete rule-chain rows: **23**
- Direct mechanism rows: **20**
- Direct stacks: **CUTLASS/CuTe, FlashInfer, IREE, MLIR, SGLang, TVM Relax, TensorRT-LLM, vLLM**
- Current-source supplement rows: **2**
- Registered hypotheses: **H1.1, H1.2, H1.3, H1.4, H1.NEG**

## 5.2 Framework coverage judgment


| Framework/layer | Why it is or is not part of this RQ |
|---|---|
| vLLM | DIRECT: supported/common KV-layout negotiation and persistent consumer contract. |
| SGLang | DIRECT: phase consumers, specialized layouts, MoE/scale preparation. |
| TensorRT-LLM | DIRECT: backend/fusion/format restructuring. |
| FlashInfer | DIRECT: NHD/HND, copy and zero-copy/view variants, LSE layout. |
| Triton/CUTLASS | DIRECT/BASELINE: physical mapping and algebraic/native kernel alternatives. |
| MLIR/IREE/TVM | DIRECT/BASELINE: explicit layout transform/encoding legality.  |


## 5.3 Complete source-rule → proxy → ignored → fallback → trigger → hypothesis chain


| ID | Stack | Role | Source rule/fact | Proxy | Ignored variables | Fallback/repair | Trigger condition | Hypothesis link | Source status |
|---|---|---|---|---|---|---|---|---|---|
| HC04 | CUTLASS/CuTe | DIRECT_MECHANISM | CUTLASS documents an efficient row-major epilogue and, for column-major output, transposes/swaps the equivalent GEMM operands so the efficient epilogue mapping is preserved. | input/output layout | downstream consumer layout/fusion | Specialized template path. | The exact source predicate is in source_rule_fact; model-family envelope: Dense MLP/linear | H1.1,H1.4 | FROZEN_LEDGER |
| F-A01 | FlashInfer | DIRECT_MECHANISM | Layout is an explicit caller-visible NHD/HND contract. | Layout argument, tensor shape/strides. | Global serving objective unless caller models it. | Invalid layout fails or wrapper-specific repair. | The exact source predicate is in source_rule_fact; model-family envelope: Paged KV, Generic attention | H1.1,H1.3 | FROZEN_LEDGER |
| F-A02 | FlashInfer | DIRECT_MECHANISM | NHD can be accepted but transposed + materialized contiguously as HND, including scale tensors. | Backend, NVFP4 KV, input layout. | Reuse count, memory pressure, possibility of native HND allocation. | Automatic transpose/contiguous copy. | The exact source predicate is in source_rule_fact; model-family envelope: Quantized KV/activation | H1.1,H1.2,H1.4 | FROZEN_LEDGER |
| F-L1 | FlashInfer | DIRECT_MECHANISM | Paged KV APIs consume caller-selected NHD/HND layouts. | kv_layout and strides/shapes | cross-consumer/global serving objective | invalid layout fails or wrapper-specific repair | The exact source predicate is in source_rule_fact; model-family envelope: MHA, MQA, GQA, Paged KV | H1.1,H1.3 | FROZEN_LEDGER |
| F-L2 | FlashInfer | DIRECT_MECHANISM | Some TRTLLM-gen NVFP4 paths accept NHD but transpose/materialize HND. | backend, KV dtype, input layout | reuse count, memory pressure, native-HND allocation option | transpose + contiguous copy | The exact source predicate is in source_rule_fact; model-family envelope: Quantized KV/activation | H1.1,H1.2,H1.4 | FROZEN_LEDGER |
| HF02 | FlashInfer | DIRECT_MECHANISM | For trtllm-gen with NVFP4 KV cache, NHD input triggers automatic transpose plus contiguous copies of KV data and scale tensors to HND, with documented allocation/copy overhead. | layout; dtype; backend | reuse count; temporary memory; producer-native HND | Materialized transpose/copy. | The exact source predicate is in source_rule_fact; model-family envelope: Quantized KV/activation, Paged KV | H1.1,H1.2,H1.4 | FROZEN_LEDGER |
| HI04 | IREE | DIRECT_MECHANISM | The Encoding dialect attaches abstract data-layout encodings to tensors; downstream encoding/materialization infrastructure resolves these representations for targets. | op/operand/indexing maps/iteration/target | LLM persistent multi-consumer/online traffic | Unset/materialize encoding. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H1.1,H1.2 | FROZEN_LEDGER |
| HM03 | MLIR | DIRECT_MECHANISM | structured.pack_transpose can transpose a pack/compute(/unpack) chain; for pack it expects the consuming linalg.generic to be the sole consumer, and unsupported topology/specification yield… | pack chain; permutations; consumer topology | multi-consumer/persistent state/global cost | Silenceable failure. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H1.1,H1.2,H1.4 | FROZEN_LEDGER |
| HS03 | SGLang | DIRECT_MECHANISM | HybridAttnBackend explicitly routes decode/idle to the decode backend, prefill/extend to the prefill backend, and target_verify according to speculative_attention_mode. | semantic phase; spec mode | within-phase batch/KV/GQA; shared-layout conversion | Inherit global backend / reject unsupported combo. | The exact source predicate is in source_rule_fact; model-family envelope: Speculative, Generic attention | H1.1 | FROZEN_LEDGER |
| HS06 | SGLang | DIRECT_MECHANISM | For NVFP4 + FlashInfer CUTLASS MoE, received scale tensors are explicitly passed through nvfp4_block_scale_interleave; a TODO says to fuse this interleave into CUTLASS MoE when supported. | runner backend; scale layout | amortization; direct producer-native scale layout | Materialize scale interleave. | The exact source predicate is in source_rule_fact; model-family envelope: MoE, Quantized KV/activation | H1.1,H1.4 | FROZEN_LEDGER |
| S-A01 | SGLang | DIRECT_MECHANISM | Separate prefill and decode backend overrides exist; if they resolve differently, HybridAttnBackend is built. | Phase-specific config/backend names. | Within-phase batch/KV-length crossover, conversion cost. | Single backend if names match; invalid combination fails. | The exact source predicate is in source_rule_fact; model-family envelope: MHA, MLA | H1.1 | FROZEN_LEDGER |
| S-A02 | SGLang | DIRECT_MECHANISM | Decode/idle uses decode backend; extend/prefill uses prefill backend; target-verify backend depends on speculative-attention mode. | Forward mode and speculative-attention mode. | Actual q_len/KV_len, acceptance rate, per-batch crossover. | Route to the configured phase backend. | The exact source predicate is in source_rule_fact; model-family envelope: Speculative, Generic attention | H1.1 | FROZEN_LEDGER |
| HX05 | TVM Relax | DIRECT_MECHANISM | Relax layout_transform is an explicit graph operator taking an IndexMap and optional padding and materializing a transformed tensor. | index map; tensor shape | producer/consumer could avoid conversion jointly | Keep/lower/fuse transform. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H1.1,H1.4 | FROZEN_LEDGER |
| HT02 | TensorRT-LLM | DIRECT_MECHANISM | FuseSiluMul is a post-load transform intended to run after GEMM fusion; a suitable sole FP8-linear consumer enables further quantization fusion. | prior rewrite; consumer cardinality/type | joint gate/up packing + activation layout + down-GEMM format | Skip extra fusion if prerequisite absent. | The exact source predicate is in source_rule_fact; model-family envelope: Quantized KV/activation, Dense MLP/linear | H1.1 | FROZEN_LEDGER |
| HT05 | TensorRT-LLM | DIRECT_MECHANISM | AutoDeploy contains explicit fused-MoE restructuring logic that converts between stacked/backend-specific weight/graph representations and enforces backend/style support constraints. | MoE graph/weight representation style, backend family, fused-MoE support predicates | Runtime expert-token distribution, restructuring/copy cost, peak temporary memory, alternative fused-MoE layout families. | Apply only supported fused-MoE restructuring/backend style; otherwise retain/choose a supported general or alternate path rather than … | The exact source predicate is in source_rule_fact; model-family envelope: MoE | H1.1,H1.4 | FROZEN_LEDGER |
| V-A01 | vLLM | DIRECT_MECHANISM | Multimodal-prefix configurations can prepend composite backends such as Triton+FlashAttention or Triton+FlashInfer that route current queries between kernels while sharing KV. | SM generation, multimodal-prefix flag, causal/bidirectional query semantics, feature support. | Per-query measured latency, image/text mixture, conversion alternative, route-switch locality. | Fall through to ordinary candidates if composite backend is incompatible; internal component fallback applies. | The exact source predicate is in source_rule_fact; model-family envelope: Generic attention | H1.1,H1.4,H1.NEG | FROZEN_LEDGER |
| V-A07 | vLLM | DIRECT_MECHANISM | MLA prefill generally prefers FA, but Blackwell has extra fallbacks and a known geometry can try TRT-LLM Ragged before FA. | Phase, SM, qk-nope/rope dimensions, value dimension. | Actual prefill length/batch and prefix reuse. | Ordered fallback among compatible prefill kernels. | The exact source predicate is in source_rule_fact; model-family envelope: MLA | H1.1,H1.2,H1.NEG | FROZEN_LEDGER |
| V-L1 | vLLM | DIRECT_MECHANISM | Each backend declares supported/preferred KV layouts. | backend support/preference list | runtime frequency, bytes, cache locality, conversion cost | default layout preference if none declares restrictions | The exact source predicate is in source_rule_fact; model-family envelope: Paged KV, Generic attention | H1.1,H1.NEG | FROZEN_LEDGER |
| HF01 | FlashInfer | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | The API explicitly states that tensor-core decode can be faster for large grouped-query-attention group size; NHD/HND is an explicit KV-layout contract. | Hq/Hkv | global conversion/system contention | Non-tensor-core path. | The exact source predicate is in source_rule_fact; model-family envelope: Generic attention | H1.1 | FROZEN_LEDGER |
| HV01 | vLLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | The design doc states that RoPE+KV-cache update fusion is ROCm/AITER-specific and, by default, applies when num_tokens <= 256; the maximum is configurable. | num_tokens; platform/backend | head geometry; KV layout; concurrent load; reuse | Do not fuse above threshold / unsupported platform. | The exact source predicate is in source_rule_fact; model-family envelope: Paged KV, Generic attention | H1.1,H1.2,H1.NEG | FROZEN_LEDGER |
| HV04 | vLLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | The fusion design enumerates concrete graph patterns including AllReduce->RMSNorm, Attention->Quant, QK-Norm->RoPE, RoPE->KV write, RMSNorm->Quant and SiLU+Mul->Quant. | graph syntax/ops; backend; dtype; GPU | semantic variants; multi-user graphs; alternate output contracts | General unfused path. | The exact source predicate is in source_rule_fact; model-family envelope: Quantized KV/activation, Generic attention, Dense MLP/linear | H1.1 | FROZEN_LEDGER |
| S12-07 | FlashInfer | DIRECT_MECHANISM | For the cuDNN route, FlashInfer builds the graph from strides and can present an NHD cache as a transposed [pages, heads, page_size, head_dim] view without a copy. | backend route plus actual strides/layout | downstream reuse/fanout, copy-vs-view profitability on other backends, multi-consumer alias/lifetime constraints | use a compatible backend/path; non-view-compatible cases require another representation path | cuDNN paged decode with an NHD cache whose transposed strides satisfy the consumer graph | H1.1,H1.2,H1.3,H1.4 | CURRENT_SOURCE_SUPPLEMENT |
| S12-08 | FlashInfer | DIRECT_MECHANISM | Prefill wrappers expose returned LSE layout NH or HN; HN is a contiguous transpose and, for relevant backends, uses one fused transpose+rescale kernel; caller-provided LSE must match the re… | requested LSE layout and backend | which downstream merge consumer will reuse the LSE, fanout, total end-to-end value of producing HN directly | produce NH or pay the fused transpose/rescale path for HN according to the backend implementation | Attention prefill returning LSE for merge/normalization consumers; NH versus HN output layout | H1.1,H1.2,H1.4 | CURRENT_SOURCE_SUPPLEMENT |


## 5.4 Modern model trigger audit

| Family | Concrete trigger for this RQ |
|---|---|
| MHA/MQA/GQA | Primary: producer QKV/KV write vs decode-native layout; GQA head ratio can amplify consumer sensitivity. |
| MLA | Primary external-validity: latent-cache producer vs MLA decode/backend-native representation. |
| Sparse/DSA | Secondary: use only after dense/GQA because sparse metadata introduces RQ8 interaction. |
| MoE | Primary non-attention external-validity: dispatch/scale representation vs grouped-GEMM-native layout. |
| Speculative/Hybrid | Target/draft/verify fanout can increase reuse/consumer heterogeneity; hybrid state is secondary. |

## 5.5 Falsifiable hypotheses and direct falsifiers


| H | Hypothesis | What falsifies it | Experiment | Linked evidence rows |
|---|---|---|---|---|
| H1.1 | At least one common producer→consumer edge has stable local-best ≠ edge-best layout. | Producer-local and edge-global winners coincide in every tested legal configuration within the practical-noise threshold. | RQ1-E3 | 23 |
| H1.2 | Rank-inversion regret increases with downstream reuse/fanout. | Regret is flat/decreases with controlled reuse after boundary cost is amortized. | RQ1-E4 | 9 |
| H1.3 | Stride-polymorphic consumers reduce rank inversion by eliminating materialization. | Legal zero-copy paths do not reduce edge regret versus materialized adaptation. | RQ1-E4 | 3 |
| H1.4 | Producer-native consumer layout beats natural-producer+conversion in a measurable regime. | Direct consumer-native emission is always slower after producer cost is included. | RQ1-E3 | 11 |
| H1.NEG | Low-precision/layout-insensitive regimes may have no meaningful inversion. | A large stable inversion appears even in the negative-control region. | RQ1-E0 | 4 |


## 5.6 Closure judgment

This RQ now satisfies the requested chain:

```text
source rule
  → explicit proxy
  → explicit omitted variables
  → framework-native fallback / repair / alternative
  → modern-model trigger envelope
  → falsifiable hypothesis + falsifier
```

The evidence does **not** assert that the hypothesis is true. The remaining unknown is empirical and is already assigned to the v11 experiment protocol.



# 6. L-RQ2 — Optimal layout-domain granularity

## 6.1 Trial status

**Verdict:** `COMPLETE_STRONG`

- Complete rule-chain rows: **29**
- Direct mechanism rows: **27**
- Direct stacks: **IREE, SGLang, TensorRT-LLM, vLLM**
- Current-source supplement rows: **6**
- Registered hypotheses: **H2.1, H2.2, H2.3, H2.4, H2.5**

## 6.2 Framework coverage judgment


| Framework/layer | Why it is or is not part of this RQ |
|---|---|
| vLLM | DIRECT: whole-model resolution, HiSparse multi-groups, offload groups. |
| SGLang | DIRECT: phase routing and page-major hybrid-state unification. |
| TensorRT-LLM | DIRECT: multiple KV pools for incompatible geometry. |
| FlashInfer | N/A AS POLICY OWNER: kernel library does not own persistent pool/domain partition; its per-wrapper freedom is an implementation substrate, not a missing framework rule. |
| Compiler stacks | IREE per-tensor Encoding is a representation baseline; allocator/domain economics are outside generic compiler scope. |


## 6.3 Complete source-rule → proxy → ignored → fallback → trigger → hypothesis chain


| ID | Stack | Role | Source rule/fact | Proxy | Ignored variables | Fallback/repair | Trigger condition | Hypothesis link | Source status |
|---|---|---|---|---|---|---|---|---|---|
| HI04 | IREE | DIRECT_MECHANISM | The Encoding dialect attaches abstract data-layout encodings to tensors; downstream encoding/materialization infrastructure resolves these representations for targets. | op/operand/indexing maps/iteration/target | LLM persistent multi-consumer/online traffic | Unset/materialize encoding. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H2.1 | FROZEN_LEDGER |
| HS01 | SGLang | DIRECT_MECHANISM | The attention-backend guide states that prefix reuse occurs only for complete pages, page_size=1 maximizes reuse, larger pages generally improve attention-kernel performance, and backends h… | page size; backend | actual prefix distribution; fragmentation; transfer | Emulation for some backends or supported native size. | The exact source predicate is in source_rule_fact; model-family envelope: Generic attention | H2.1 | FROZEN_LEDGER |
| HS03 | SGLang | DIRECT_MECHANISM | HybridAttnBackend explicitly routes decode/idle to the decode backend, prefill/extend to the prefill backend, and target_verify according to speculative_attention_mode. | semantic phase; spec mode | within-phase batch/KV/GQA; shared-layout conversion | Inherit global backend / reject unsupported combo. | The exact source predicate is in source_rule_fact; model-family envelope: Speculative, Generic attention | H2.1,H2.3,H2.4,H2.5 | FROZEN_LEDGER |
| S-A01 | SGLang | DIRECT_MECHANISM | Separate prefill and decode backend overrides exist; if they resolve differently, HybridAttnBackend is built. | Phase-specific config/backend names. | Within-phase batch/KV-length crossover, conversion cost. | Single backend if names match; invalid combination fails. | The exact source predicate is in source_rule_fact; model-family envelope: MHA, MLA | H2.1,H2.5 | FROZEN_LEDGER |
| S-A02 | SGLang | DIRECT_MECHANISM | Decode/idle uses decode backend; extend/prefill uses prefill backend; target-verify backend depends on speculative-attention mode. | Forward mode and speculative-attention mode. | Actual q_len/KV_len, acceptance rate, per-batch crossover. | Route to the configured phase backend. | The exact source predicate is in source_rule_fact; model-family envelope: Speculative, Generic attention | H2.1,H2.4 | FROZEN_LEDGER |
| S-A05 | SGLang | DIRECT_MECHANISM | `page_size=1` maximizes prefix reuse because only full pages are reusable, while larger pages generally improve attention-kernel performance. | Page size, prefix-cache semantics. | Real prefix distribution, fragmentation, transfer granularity, graph state. | User/configuration choice; no universal joint optimizer. | The exact source predicate is in source_rule_fact; model-family envelope: Paged KV, Generic attention | H2.1 | FROZEN_LEDGER |
| S-A06 | SGLang | DIRECT_MECHANISM | Target verify/draft routing is conditioned on speculative mode; some backend/page/top-k combinations are forbidden. | Spec mode, top-k, page size, backend. | Acceptance distribution, draft depth, target/draft asymmetry. | Switch supported mode/backend or reject. | The exact source predicate is in source_rule_fact; model-family envelope: Speculative | H2.1,H2.4 | FROZEN_LEDGER |
| S-A07 | SGLang | DIRECT_MECHANISM | A draft worker can override both draft prefill and draft decode to a **single draft backend**. | Draft-worker flag, draft backend. | Cases where draft prefill/decode prefer different kernels, including FP4 paths. | Inherited/default draft backend; incompatible cases can fail. | The exact source predicate is in source_rule_fact; model-family envelope: Speculative, Generic attention | H2.1,H2.4 | FROZEN_LEDGER |
| S-A10 | SGLang | DIRECT_MECHANISM | Modern hybrid recurrent/linear-attention architectures use a separate linear-attention backend; some FlashInfer prefill paths require a conjunction over GPU/CUDA/state dtype/head dims/cache… | Architecture, SM, CUDA, recurrent-state dtype, head dimensions, cache/chunk mode. | Live batch, state reuse, MoE/system contention. | Base linear-attention backend. | The exact source predicate is in source_rule_fact; model-family envelope: Hybrid/recurrent, Generic attention | H2.1,H2.5 | FROZEN_LEDGER |
| T-A02 | TensorRT-LLM | DIRECT_MECHANISM | Layers with incompatible KV geometry/window characteristics use separate pools rather than one homogeneous pool. | KV-head count, attention-window/cache geometry. | Per-layer runtime weight, phase preference, remote consumer. | Create another pool. | The exact source predicate is in source_rule_fact; model-family envelope: Paged KV | H2.1,H2.2 | FROZEN_LEDGER |
| T-A10 | TensorRT-LLM | DIRECT_MECHANISM | Attention can use TP or DP; documentation treats TP as suitable for smaller batches and attention-DP for larger throughput regimes. | Deployment parallel configuration / expected batch regime. | Layout-locality and per-rank sequence skew. | User/deployment chooses alternative. | The exact source predicate is in source_rule_fact; model-family envelope: Generic attention | H2.1 | FROZEN_LEDGER |
| T-A11 | TensorRT-LLM | DIRECT_MECHANISM | With GQA/MQA/MLA, KV may be replicated if KV-head count is smaller than TP size. | KV-head count, TP size, attention family. | Extra memory pressure and possible alternative DP policy. | Replicate per rank. | The exact source predicate is in source_rule_fact; model-family envelope: MQA, GQA, MLA | H2.1 | FROZEN_LEDGER |
| HV05 | vLLM | DIRECT_MECHANISM | The modular MoE design distinguishes activation formats and states that input activation format is determined by the All2All dispatch; it motivates modular composition because implementatio… | A2A; activation format; quantization; architecture | joint network payload + expert GEMM layout cost | Pick compatible family or reject. | The exact source predicate is in source_rule_fact; model-family envelope: MoE | H2.1 | FROZEN_LEDGER |
| V-A01 | vLLM | DIRECT_MECHANISM | Multimodal-prefix configurations can prepend composite backends such as Triton+FlashAttention or Triton+FlashInfer that route current queries between kernels while sharing KV. | SM generation, multimodal-prefix flag, causal/bidirectional query semantics, feature support. | Per-query measured latency, image/text mixture, conversion alternative, route-switch locality. | Fall through to ordinary candidates if composite backend is incompatible; internal component fallback applies. | The exact source predicate is in source_rule_fact; model-family envelope: Generic attention | H2.1 | FROZEN_LEDGER |
| V-A10 | vLLM | DIRECT_MECHANISM | Uniform cross-layer blocks require a conjunction of transfer presence, connector preference, simple compatible attention grouping/spec, quantization compatibility and backend block-stride i… | Connector preference, group count/spec type, quantization, backend indexing. | Measured registration/copy cost, layer execution weight, network packetization. | Ordinary per-layer organization. | The exact source predicate is in source_rule_fact; model-family envelope: KV transfer/disagg, Quantized KV/activation, Generic attention | H2.1,H2.2 | FROZEN_LEDGER |
| V-A12 | vLLM | DIRECT_MECHANISM | Target and draft can independently select backend constraints whose supported-layout intersection is empty, causing initialization failure. | Target layout set, draft layout set. | Joint target/draft cost, second-best backend pair, conversion, separate pools. | Current observed behavior can be hard failure. | The exact source predicate is in source_rule_fact; model-family envelope: Speculative | H2.1,H2.4 | FROZEN_LEDGER |
| V-B3 | vLLM | DIRECT_MECHANISM | Attention/KV semantic kind can override the global backend. | MLA, sliding-window, attention type, config map | per-layer execution weight, bytes, live shape and phase cost | missing kind uses global backend | The exact source predicate is in source_rule_fact; model-family envelope: Generic attention | H2.1,H2.2 | FROZEN_LEDGER |
| V-L2 | vLLM | DIRECT_MECHANISM | Candidate layouts are the intersection across all consuming backends. | boolean membership in supported-layout sets | per-consumer benefit, conversion amortization, split-pool cost | empty intersection is hard error | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H2.1 | FROZEN_LEDGER |
| V-L3 | vLLM | DIRECT_MECHANISM | When declarations differ, candidates are ordered by how many backends rank a layout first. | unweighted first-choice vote count | layer count, executed KV bytes, phase frequency, performance-gap magnitude | tie keeps enum order | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H2.1,H2.2 | FROZEN_LEDGER |
| V-L4 | vLLM | DIRECT_MECHANISM | Resolve one KV-cache layout for the whole model. | worker candidate lists, cache specs/config | per-group/phase/consumer optimum and workload drift | mismatch/error; resolved layout is recorded | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H2.1 | FROZEN_LEDGER |
| V-L5 | vLLM | DIRECT_MECHANISM | Mixed H/N/C shapes narrow candidates to block-compact layouts. | static cache-spec shape heterogeneity | group execution weight, separate-pool alternative | no candidate -> failure | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H2.1,H2.2 | FROZEN_LEDGER |
| S-A04 | SGLang | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Blackwell favors TRTLLM MHA except speculative top-k > 1 changes the path/constraints. | Blackwell, speculative top-k. | Acceptance rate, verify length, KV length. | Compatible alternative backend. | The exact source predicate is in source_rule_fact; model-family envelope: MHA, Speculative | H2.1,H2.4 | FROZEN_LEDGER |
| V-A11 | vLLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Certain backend + batch-invariance/prefix-cache/chunk-lookback combinations disable a feature, warn, or require a specific backend. | Feature flags and backend identity. | Lost prefix-reuse opportunity cost. | Disable feature, warn, or reject incompatible backend. | The exact source predicate is in source_rule_fact; model-family envelope: Generic attention | H2.1 | FROZEN_LEDGER |
| S12-01 | vLLM | DIRECT_MECHANISM | OffloadingConfig explicitly carries tokens_per_block, blocks_per_chunk, resolved worker KV layout, canonical_layout, replicated_layout, and whether persisted bytes are parallelism-agnostic;… | configured block/chunk granularity, parallel topology, pure-MLA/TP-only replication condition, canonical-layout request | live network bandwidth, cache-hit/reuse distribution, conversion cost, host-cache locality, future topology changes | representation mode is configured/certified rather than jointly cost-optimized; unsupported portability must use a compatible represen… | KV offload/disaggregation for MHA/GQA/MLA; multi-rank TP/PP/PCP/DCP; canonical/direct/replicated host representation | H2.1 | CURRENT_SOURCE_SUPPLEMENT |
| S12-02 | vLLM | DIRECT_MECHANISM | HiSparse partitions sparse-MLA source and indexer specs, builds source/indexer/hot/resident groups, requires one resolved GPU block size, requires one page size within each hot-cache group,… | cache role, block size, page_size_bytes, indexer-page budget, host budget, block-outermost layout capability | live sparsity distribution, per-layer access frequency, host/device bandwidth variation, dynamic hot-set size, alternative grouping costs | raises on incompatible block/page/layout conditions; forms compatible groups and computes host/device layouts | HiSparse sparse-MLA with source+indexer caches, hot/resident device caches and host-resident source state | H2.1 | CURRENT_SOURCE_SUPPLEMENT |
| S12-05 | SGLang | DIRECT_MECHANISM | SGLang exposes a page-major KV layout that places Mamba state and full/SWA KV caches in one page-granularity envelope (page outermost, layer-major within page) instead of the default per-la… | explicit enable flag plus backend compatibility | per-state access frequency, model/workload-specific profitability, alternative hybrid-state domain partitions | default per-layer layout remains available when page-major mode is disabled/incompatible | Hybrid Mamba + full attention/SWA models using Triton attention/linear-attention/Mamba backends | H2.1,H2.5 | CURRENT_SOURCE_SUPPLEMENT |
| S12-10 | vLLM | DIRECT_MECHANISM | Current indexed vLLM source keeps a generalized default KV-layout preference beginning with LBNHC, LBHNC, BLNHC and resolves supported layouts through backend-declared capability sets rathe… | backend-supported layout sets and fixed preference order | live execution weights, measured performance gaps, layouts not implemented/exposed by any selected backend | intersect available sets and choose from the supported preference order | vLLM paged attention layout resolution across backends; generalized LBNHC/LBHNC/BLNHC/... families | H2.1,H2.2 | CURRENT_SOURCE_SUPPLEMENT |
| S13-01 | vLLM | DIRECT_MECHANISM | EngineCore writes the resolved cache layout into KVCacheConfig, and GPUModelRunner uses get_resolved_kv_cache_layout() while creating/profiling KV cache state. | resolved deployment-time KV cache layout and cache configuration | future workload-regime drift, migration cost/benefit, per-phase re-layout, remaining request horizon | reinitialize/reconfigure the engine/cache state rather than changing the allocated layout per request | Persistent paged MHA/MQA/GQA/MLA cache initialization; any serving mode using resolved KV layout before profiling/allocation | H2.1 | CURRENT_SOURCE_SUPPLEMENT |
| S13-04 | TensorRT-LLM | DIRECT_MECHANISM | Helix context partitioning assigns token blocks round-robin across CP ranks using tokens_per_block and changes both prefill→decode KV movement and how KV grows during decode when generation… | tokens_per_block, CP rank count, generation CP mode | live interconnect contention, unequal request lengths, alternative block/page sizes, replication alternatives, per-rank memory skew | use the existing non-Helix/non-CP KV-cache infrastructure when the Helix mode is not enabled | multi-million-token long-context decode, generation CP > 1, distributed paged KV transfer/growth | H2.1 | CURRENT_SOURCE_SUPPLEMENT |


## 6.4 Modern model trigger audit

| Family | Concrete trigger for this RQ |
|---|---|
| MHA/MQA/GQA | Different attention kinds/layers or phase consumers can prefer different persistent layouts. |
| MLA | MLA versus full/SWA cache geometry and HiSparse roles create separate/unified domain choices. |
| Sparse/DSA | Source/indexer/hot/resident groups are a direct domain-granularity trigger. |
| MoE | Only where activation/routing persistent domains are shared across communication/compute; not the primary KV case. |
| Speculative/Hybrid | Target/draft/verify and Mamba/full/SWA unification are direct heterogeneity triggers. |

## 6.5 Falsifiable hypotheses and direct falsifiers


| H | Hypothesis | What falsifies it | Experiment | Linked evidence rows |
|---|---|---|---|---|
| H2.1 | A consumer-heterogeneity threshold exists beyond which splitting layout domains wins. | A shared domain remains oracle-optimal across the tested heterogeneity range. | RQ2-E2 | 29 |
| H2.2 | Executed-byte/time-weighted regret predicts split decisions better than equal consumer voting. | Unweighted voting matches or beats weighted models on held-out traces. | RQ2-E3 | 6 |
| H2.3 | Low heterogeneity/reuse favors one shared domain. | Fine-grained domains win despite negligible consumer preference differences. | RQ2-E0 | 1 |
| H2.4 | Speculative acceptance rate moves the optimal target/draft/verify partition. | Partition winner is invariant to controlled acceptance-rate changes. | RQ2-E4 | 6 |
| H2.5 | Heterogeneous KV/recurrent state only merits separate domains above an overhead threshold. | Separate domains always win or never win regardless of storage/allocator overhead. | RQ2-E5 | 4 |


## 6.6 Closure judgment

This RQ now satisfies the requested chain:

```text
source rule
  → explicit proxy
  → explicit omitted variables
  → framework-native fallback / repair / alternative
  → modern-model trigger envelope
  → falsifiable hypothesis + falsifier
```

The evidence does **not** assert that the hypothesis is true. The remaining unknown is empirical and is already assigned to the v11 experiment protocol.



# 7. L-RQ3 — Layout adaptation amortization

## 7.1 Trial status

**Verdict:** `COMPLETE_STRONG`

- Complete rule-chain rows: **17**
- Direct mechanism rows: **17**
- Direct stacks: **CUTLASS/CuTe, FlashInfer, SGLang, TVM Relax, TensorRT-LLM, vLLM**
- Current-source supplement rows: **2**
- Registered hypotheses: **H3.1, H3.2, H3.3, H3.4, H3.5**

## 7.2 Framework coverage judgment


| Framework/layer | Why it is or is not part of this RQ |
|---|---|
| vLLM | DIRECT: connector preference and compatibility behavior. |
| SGLang | DIRECT: 5D/specialized path and explicit scale interleave. |
| TensorRT-LLM | DIRECT: endpoint format conversion and overlap. |
| FlashInfer | DIRECT: materialized conversion, stride/view alternative, LSE transform. |
| CUTLASS/TVM | DIRECT/BASELINE: algebraic rewrite and explicit layout_transform. |


## 7.3 Complete source-rule → proxy → ignored → fallback → trigger → hypothesis chain


| ID | Stack | Role | Source rule/fact | Proxy | Ignored variables | Fallback/repair | Trigger condition | Hypothesis link | Source status |
|---|---|---|---|---|---|---|---|---|---|
| HC04 | CUTLASS/CuTe | DIRECT_MECHANISM | CUTLASS documents an efficient row-major epilogue and, for column-major output, transposes/swaps the equivalent GEMM operands so the efficient epilogue mapping is preserved. | input/output layout | downstream consumer layout/fusion | Specialized template path. | The exact source predicate is in source_rule_fact; model-family envelope: Dense MLP/linear | H3.1,H3.5 | FROZEN_LEDGER |
| F-A02 | FlashInfer | DIRECT_MECHANISM | NHD can be accepted but transposed + materialized contiguously as HND, including scale tensors. | Backend, NVFP4 KV, input layout. | Reuse count, memory pressure, possibility of native HND allocation. | Automatic transpose/contiguous copy. | The exact source predicate is in source_rule_fact; model-family envelope: Quantized KV/activation | H3.1,H3.2,H3.5 | FROZEN_LEDGER |
| F-A06 | FlashInfer | DIRECT_MECHANISM | Some kernels can consume NHD-contiguous or HND-transposed/arbitrary supported strides without materialized conversion. | Strides and innermost head-dim condition. | Cache/TLB efficiency of each stride pattern. | Another kernel or reject unsupported stride. | The exact source predicate is in source_rule_fact; model-family envelope: GQA, Paged KV | H3.1,H3.4,H3.5 | FROZEN_LEDGER |
| F-L2 | FlashInfer | DIRECT_MECHANISM | Some TRTLLM-gen NVFP4 paths accept NHD but transpose/materialize HND. | backend, KV dtype, input layout | reuse count, memory pressure, native-HND allocation option | transpose + contiguous copy | The exact source predicate is in source_rule_fact; model-family envelope: Quantized KV/activation | H3.1,H3.2,H3.5 | FROZEN_LEDGER |
| HF02 | FlashInfer | DIRECT_MECHANISM | For trtllm-gen with NVFP4 KV cache, NHD input triggers automatic transpose plus contiguous copies of KV data and scale tensors to HND, with documented allocation/copy overhead. | layout; dtype; backend | reuse count; temporary memory; producer-native HND | Materialized transpose/copy. | The exact source predicate is in source_rule_fact; model-family envelope: Quantized KV/activation, Paged KV | H3.1,H3.2,H3.5 | FROZEN_LEDGER |
| HS01 | SGLang | DIRECT_MECHANISM | The attention-backend guide states that prefix reuse occurs only for complete pages, page_size=1 maximizes reuse, larger pages generally improve attention-kernel performance, and backends h… | page size; backend | actual prefix distribution; fragmentation; transfer | Emulation for some backends or supported native size. | The exact source predicate is in source_rule_fact; model-family envelope: Generic attention | H3.1,H3.2,H3.3 | FROZEN_LEDGER |
| HS06 | SGLang | DIRECT_MECHANISM | For NVFP4 + FlashInfer CUTLASS MoE, received scale tensors are explicitly passed through nvfp4_block_scale_interleave; a TODO says to fuse this interleave into CUTLASS MoE when supported. | runner backend; scale layout | amortization; direct producer-native scale layout | Materialize scale interleave. | The exact source predicate is in source_rule_fact; model-family envelope: MoE, Quantized KV/activation | H3.1 | FROZEN_LEDGER |
| S-A05 | SGLang | DIRECT_MECHANISM | `page_size=1` maximizes prefix reuse because only full pages are reusable, while larger pages generally improve attention-kernel performance. | Page size, prefix-cache semantics. | Real prefix distribution, fragmentation, transfer granularity, graph state. | User/configuration choice; no universal joint optimizer. | The exact source predicate is in source_rule_fact; model-family envelope: Paged KV, Generic attention | H3.1,H3.2,H3.3 | FROZEN_LEDGER |
| S-A09 | SGLang | DIRECT_MECHANISM | `vectorized_5d` cache layout routes writes/decode through different compatible paths because a 4D view is not representable. | Pool layout tag, backend/platform. | Cost of keeping 5D vs converting; workload crossover. | 5D-compatible writer/decode path. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H3.1,H3.4,H3.5 | FROZEN_LEDGER |
| HX05 | TVM Relax | DIRECT_MECHANISM | Relax layout_transform is an explicit graph operator taking an IndexMap and optional padding and materializing a transformed tensor. | index map; tensor shape | producer/consumer could avoid conversion jointly | Keep/lower/fuse transform. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H3.1 | FROZEN_LEDGER |
| T-A08 | TensorRT-LLM | DIRECT_MECHANISM | Producer/consumer KV layouts and parallel formats can be transformed during transfer. | Producer and consumer format/parallel config. | Where to place conversion under current load, conversion amortization. | Convert/format, otherwise unsupported path fails. | The exact source predicate is in source_rule_fact; model-family envelope: KV transfer/disagg | H3.1,H3.3,H3.5 | FROZEN_LEDGER |
| T-A09 | TensorRT-LLM | DIRECT_MECHANISM | Transfer can overlap computation. | Transfer stage and scheduler/connector capability. | Competition with MoE/copy engines. | Non-overlapped path. | The exact source predicate is in source_rule_fact; model-family envelope: KV transfer/disagg | H3.1,H3.3,H3.5 | FROZEN_LEDGER |
| V-A09 | vLLM | DIRECT_MECHANISM | Connector-specific paths can prefer HND for transfer while generic/default representations may be NHD-like. | Connector identity/mode, configured preference. | Actual link bandwidth, message size, local-kernel penalty. | Drop incompatible connector preference or retain local candidate. | The exact source predicate is in source_rule_fact; model-family envelope: KV transfer/disagg, Generic attention | H3.1,H3.3 | FROZEN_LEDGER |
| V-L2 | vLLM | DIRECT_MECHANISM | Candidate layouts are the intersection across all consuming backends. | boolean membership in supported-layout sets | per-consumer benefit, conversion amortization, split-pool cost | empty intersection is hard error | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H3.1 | FROZEN_LEDGER |
| V-L6 | vLLM | DIRECT_MECHANISM | KV connector preferred layout is advisory and only used if locally compatible. | connector preference plus local candidate set | link bandwidth, transfer frequency, overlap, remote/native speedup | drop connector preference with warning | The exact source predicate is in source_rule_fact; model-family envelope: KV transfer/disagg, Generic attention | H3.1,H3.3 | FROZEN_LEDGER |
| S12-07 | FlashInfer | DIRECT_MECHANISM | For the cuDNN route, FlashInfer builds the graph from strides and can present an NHD cache as a transposed [pages, heads, page_size, head_dim] view without a copy. | backend route plus actual strides/layout | downstream reuse/fanout, copy-vs-view profitability on other backends, multi-consumer alias/lifetime constraints | use a compatible backend/path; non-view-compatible cases require another representation path | cuDNN paged decode with an NHD cache whose transposed strides satisfy the consumer graph | H3.1,H3.4,H3.5 | CURRENT_SOURCE_SUPPLEMENT |
| S12-08 | FlashInfer | DIRECT_MECHANISM | Prefill wrappers expose returned LSE layout NH or HN; HN is a contiguous transpose and, for relevant backends, uses one fused transpose+rescale kernel; caller-provided LSE must match the re… | requested LSE layout and backend | which downstream merge consumer will reuse the LSE, fanout, total end-to-end value of producing HN directly | produce NH or pay the fused transpose/rescale path for HN according to the backend implementation | Attention prefill returning LSE for merge/normalization consumers; NH versus HN output layout | H3.1,H3.5 | CURRENT_SOURCE_SUPPLEMENT |


## 7.4 Modern model trigger audit

| Family | Concrete trigger for this RQ |
|---|---|
| MHA/MQA/GQA | NHD/HND view/copy/native emission; repeated KV reads determine amortization. |
| MLA | Endpoint/latent-format conversion and repeated decode reads. |
| Sparse/DSA | Possible but confounded with metadata; use after dense case. |
| MoE | Scale interleave/activation permutation before grouped GEMM is a direct adaptation boundary. |
| Distributed | Convert-before-send, wire-native, convert-after-receive and overlap. |

## 7.5 Falsifiable hypotheses and direct falsifiers


| H | Hypothesis | What falsifies it | Experiment | Linked evidence rows |
|---|---|---|---|---|
| H3.1 | Materialized conversion has a measurable reuse break-even R*. | No crossover appears over a broad reuse range, or the predicted R* is not reproducible. | RQ3-E2 | 17 |
| H3.2 | Temporary-memory pressure increases R*. | Changing memory pressure/capacity does not move the conversion crossover. | RQ3-E3 | 5 |
| H3.3 | Transfer-native format/overlap can lower R*. | Wire-native conversion and overlap do not change the crossover. | RQ3-E4 | 6 |
| H3.4 | Zero-copy view/stride wins over materialization in a non-trivial legal region. | Zero-copy is never competitive once native consumer performance is included. | RQ3-E1 | 3 |
| H3.5 | Short generation/low reuse should not materialize conversion. | Conversion wins even at R=1 after allocation/copy cost is fully charged. | RQ3-E0 | 10 |


## 7.6 Closure judgment

This RQ now satisfies the requested chain:

```text
source rule
  → explicit proxy
  → explicit omitted variables
  → framework-native fallback / repair / alternative
  → modern-model trigger envelope
  → falsifiable hypothesis + falsifier
```

The evidence does **not** assert that the hypothesis is true. The remaining unknown is empirical and is already assigned to the v11 experiment protocol.



# 8. L-RQ4 — Optimal-layout stability across workload/hardware

## 8.1 Trial status

**Verdict:** `COMPLETE_STRONG`

- Complete rule-chain rows: **50**
- Direct mechanism rows: **19**
- Direct stacks: **CUTLASS/CuTe, FlashInfer, IREE, SGLang, Triton, vLLM**
- Current-source supplement rows: **5**
- Registered hypotheses: **H4.1, H4.2, H4.3, H4.4, H4.5**

## 8.2 Framework coverage judgment


| Framework/layer | Why it is or is not part of this RQ |
|---|---|
| vLLM | DIRECT/BASELINE: fixed preferences/architecture predicates. |
| SGLang | DIRECT: documented page-size workload trade-off and architecture/backend rules. |
| TensorRT-LLM | COUNTEREVIDENCE: XQA/multi-block live workload heuristics show local adaptivity is possible. |
| FlashInfer | DIRECT/COUNTEREVIDENCE: GQA/geometry fast-path predicates and graph buckets. |
| Triton/CUTLASS/IREE/TVM | DIRECT/BASELINE: autotune, target-specific layout, dynamic encodings, measured search. |


## 8.3 Complete source-rule → proxy → ignored → fallback → trigger → hypothesis chain


| ID | Stack | Role | Source rule/fact | Proxy | Ignored variables | Fallback/repair | Trigger condition | Hypothesis link | Source status |
|---|---|---|---|---|---|---|---|---|---|
| HC02 | CUTLASS/CuTe | DIRECT_MECHANISM | The helper calls its policy a simple heuristic and selects SMEM layout/swizzle atoms from major mode, element type and major-dimension size; related helpers choose legal/vectorized store at… | output layout; dtype; tile; ownership | producer/consumer graph; concurrent pressure | Legal fallback store/layout. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H4.1,H4.5 | FROZEN_LEDGER |
| F-A06 | FlashInfer | DIRECT_MECHANISM | Some kernels can consume NHD-contiguous or HND-transposed/arbitrary supported strides without materialized conversion. | Strides and innermost head-dim condition. | Cache/TLB efficiency of each stride pattern. | Another kernel or reject unsupported stride. | The exact source predicate is in source_rule_fact; model-family envelope: GQA, Paged KV | H4.1,H4.3,H4.5 | FROZEN_LEDGER |
| F-A07 | FlashInfer | DIRECT_MECHANISM | Fast paths have conjunctions over dtype, head dimension, GQA ratio, page size, q_len and SM. | Geometry/dtype/page/q_len/SM. | Continuous system state inside the region. | Alternate supported kernel. | The exact source predicate is in source_rule_fact; model-family envelope: MHA, GQA, Speculative, Paged KV | H4.1,H4.3,H4.5 | FROZEN_LEDGER |
| HI01 | IREE | DIRECT_MECHANISM | IREE exposes per-dispatch tuning knobs such as MMA layout, subgroup/workgroup parameters and reduction tiling; defaults target generic performance while tuning specializes them. | dispatch/target/knobs | persistent KV and cross-dispatch serving interactions | Tuning spec overrides. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H4.1,H4.3,H4.4 | FROZEN_LEDGER |
| HI05 | IREE | DIRECT_MECHANISM | IREE set_encoding/unset_encoding accept dynamic encoding_dims (e.g. M/N/K); the operation description explicitly states these values are used for runtime layout selection based on problem s… | runtime dimensions | prefix reuse; queue; network/MoE contention | Resolver/default behavior. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H4.1,H4.3,H4.4 | FROZEN_LEDGER |
| HS01 | SGLang | DIRECT_MECHANISM | The attention-backend guide states that prefix reuse occurs only for complete pages, page_size=1 maximizes reuse, larger pages generally improve attention-kernel performance, and backends h… | page size; backend | actual prefix distribution; fragmentation; transfer | Emulation for some backends or supported native size. | The exact source predicate is in source_rule_fact; model-family envelope: Generic attention | H4.1,H4.3,H4.5 | FROZEN_LEDGER |
| S-A05 | SGLang | DIRECT_MECHANISM | `page_size=1` maximizes prefix reuse because only full pages are reusable, while larger pages generally improve attention-kernel performance. | Page size, prefix-cache semantics. | Real prefix distribution, fragmentation, transfer granularity, graph state. | User/configuration choice; no universal joint optimizer. | The exact source predicate is in source_rule_fact; model-family envelope: Paged KV, Generic attention | H4.1,H4.5 | FROZEN_LEDGER |
| S-A08 | SGLang | DIRECT_MECHANISM | DSA is architecture-triggered and uses phase/platform/dtype-specific sub-backend rules. | Model architecture, phase, GPU, BF16/FP8, some parallel-state constraints. | Actual selected-token count/locality, sparse density, batch/KV length. | Alternative supported DSA backend or reject. | The exact source predicate is in source_rule_fact; model-family envelope: MLA, Sparse/DSA | H4.1,H4.3,H4.4 | FROZEN_LEDGER |
| S-A09 | SGLang | DIRECT_MECHANISM | `vectorized_5d` cache layout routes writes/decode through different compatible paths because a 4D view is not representable. | Pool layout tag, backend/platform. | Cost of keeping 5D vs converting; workload crossover. | 5D-compatible writer/decode path. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H4.1 | FROZEN_LEDGER |
| V-A03 | vLLM | DIRECT_MECHANISM | MLA has separate SM10, SM12 and other-GPU ordered backend sets. | `use_mla`, SM generation, head size, KV dtype, number of heads. | Live batch, q_len, KV_len, actual sparsity, transfer/MoE contention. | Next compatible MLA backend. | The exact source predicate is in source_rule_fact; model-family envelope: MLA | H4.1,H4.4 | FROZEN_LEDGER |
| V-A04 | vLLM | DIRECT_MECHANISM | Sparse MLA preference changes with KV dtype and query-head count; BF16-like paths use a head-count threshold, while quantized KV favors a different sparse backend order. | KV dtype, query-head count, SM generation. | Selected-page count, sparse density, KV length, page locality, batch. | Next compatible sparse backend. | The exact source predicate is in source_rule_fact; model-family envelope: MLA, Sparse/DSA, Quantized KV/activation | H4.1,H4.2 | FROZEN_LEDGER |
| V-A06 | vLLM | DIRECT_MECHANISM | FA version is architecture-conditioned; specialized FA4 paths have block-size/head-size/feature conditions and can transparently fall back. | SM, head size, block size, unsupported feature flags. | Current sequence length, batch, graph state, prefix reuse. | Older compatible FA implementation or make the higher-priority backend ineligible. | The exact source predicate is in source_rule_fact; model-family envelope: MHA, Paged KV | H4.1,H4.2,H4.3,H4.4 | FROZEN_LEDGER |
| V-A07 | vLLM | DIRECT_MECHANISM | MLA prefill generally prefers FA, but Blackwell has extra fallbacks and a known geometry can try TRT-LLM Ragged before FA. | Phase, SM, qk-nope/rope dimensions, value dimension. | Actual prefill length/batch and prefix reuse. | Ordered fallback among compatible prefill kernels. | The exact source predicate is in source_rule_fact; model-family envelope: MLA | H4.1,H4.5 | FROZEN_LEDGER |
| V-L3 | vLLM | DIRECT_MECHANISM | When declarations differ, candidates are ordered by how many backends rank a layout first. | unweighted first-choice vote count | layer count, executed KV bytes, phase frequency, performance-gap magnitude | tie keeps enum order | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H4.1,H4.2 | FROZEN_LEDGER |
| HC01 | CUTLASS/CuTe | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | CUTLASS GEMM heuristics use NVIDIA's analytical matmul heuristic to rank a subset of valid kernels from problem size and hardware SKU, reducing runtime autotuning search; exhaustive coverag… | shape; SKU; registered kernels | cross-op layout; online serving state | Profile/rank subset. | The exact source predicate is in source_rule_fact; model-family envelope: Dense MLP/linear | H4.1,H4.4,H4.5 | FROZEN_LEDGER |
| F-A03 | FlashInfer | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Architecture/kernel availability can auto-select TRTLLM-gen or XQA. | SM generation, kernel availability. | Batch/KV-length crossover if both feasible. | Compatible available backend. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H4.1,H4.4,H4.5 | FROZEN_LEDGER |
| F-A04 | FlashInfer | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Tensor-core path can be better for large GQA group size. | `Hq/Hkv` group size and option/policy. | Full-system contention and conversion. | Non-tensor-core path. | The exact source predicate is in source_rule_fact; model-family envelope: GQA | H4.1 | FROZEN_LEDGER |
| F-B1 | FlashInfer | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Tensor-core decode can be faster for large GQA group size. | Hq/Hkv and policy option | global conversion/system contention | non-tensor-core path | The exact source predicate is in source_rule_fact; model-family envelope: GQA | H4.1 | FROZEN_LEDGER |
| HF01 | FlashInfer | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | The API explicitly states that tensor-core decode can be faster for large grouped-query-attention group size; NHD/HND is an explicit KV-layout contract. | Hq/Hkv | global conversion/system contention | Non-tensor-core path. | The exact source predicate is in source_rule_fact; model-family envelope: Generic attention | H4.1 | FROZEN_LEDGER |
| HF04 | FlashInfer | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | For the cuDNN grouped-GEMM backend, tactic=-1 is documented as using the heuristic-best plan; non-negative tactic values select an explicit execution-plan index. | problem shape/library plan | A2A token order; producer layout; serving interference | Heuristic or explicit tactic. | The exact source predicate is in source_rule_fact; model-family envelope: Quantized KV/activation, Dense MLP/linear | H4.1,H4.4,H4.5 | FROZEN_LEDGER |
| HS04 | SGLang | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Execution arguments expose multiple FP8/FP4 GEMM backends and an auto mode whose documented choice depends on hardware/backend availability. | SM; dtype; availability | live M/N/K; downstream layout/quant format | Compatible fallback, often Triton; explicit invalid choices error. | The exact source predicate is in source_rule_fact; model-family envelope: Quantized KV/activation, Dense MLP/linear | H4.1 | FROZEN_LEDGER |
| S-A03 | SGLang | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Hopper favors FA3 when CUDA/model support conditions hold. | GPU generation, CUDA version, model support. | Seq length, batch, GQA ratio, prefix reuse. | Alternative platform backend. | The exact source predicate is in source_rule_fact; model-family envelope: MHA, MQA, GQA | H4.1,H4.4 | FROZEN_LEDGER |
| S-A04 | SGLang | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Blackwell favors TRTLLM MHA except speculative top-k > 1 changes the path/constraints. | Blackwell, speculative top-k. | Acceptance rate, verify length, KV length. | Compatible alternative backend. | The exact source predicate is in source_rule_fact; model-family envelope: MHA, Speculative | H4.1 | FROZEN_LEDGER |
| S-B1 | SGLang | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Hopper standard attention prefers FA3 when supported. | GPU generation, CUDA/model support | live batch/sequence/GQA/reuse | alternative compatible backend | The exact source predicate is in source_rule_fact; model-family envelope: MHA, MQA, GQA | H4.1,H4.3,H4.4 | FROZEN_LEDGER |
| S-B2 | SGLang | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Blackwell MHA prefers TRTLLM MHA except speculative top-k constraints alter the path. | Blackwell, speculative top-k | acceptance rate, verify length, batch/KV length | compatible alternative or reject unsupported combination | The exact source predicate is in source_rule_fact; model-family envelope: MHA, Speculative | H4.1,H4.3 | FROZEN_LEDGER |
| S-B3 | SGLang | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Other CUDA platforms prefer FlashInfer if available, else Triton. | platform class, package availability | live workload and reuse | Triton | The exact source predicate is in source_rule_fact; model-family envelope: MHA, Paged KV | H4.1 | FROZEN_LEDGER |
| S-B4 | SGLang | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | MLA Hopper path prefers FA3. | GPU generation, MLA | live shape, latent geometry, reuse | platform-compatible alternative | The exact source predicate is in source_rule_fact; model-family envelope: MLA | H4.1,H4.4 | FROZEN_LEDGER |
| S-B5 | SGLang | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | MLA Blackwell prefers FlashInfer, with model-specific TRTLLM MLA auto selection in documented cases. | GPU generation, model identity | deployment request mix | general Blackwell MLA fallback | The exact source predicate is in source_rule_fact; model-family envelope: MLA | H4.1,H4.4 | FROZEN_LEDGER |
| S-B6 | SGLang | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Other architectures can fall back to Triton MLA. | platform class | live workload | Triton | The exact source predicate is in source_rule_fact; model-family envelope: MLA | H4.1,H4.4 | FROZEN_LEDGER |
| HX06 | TVM MetaSchedule | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | MetaSchedule documents structural and anchor-block module equality; anchor-block matching ignores surrounding context and enables tuning-record sharing across fused operators with the same … | anchor trace; target | LLM state/hardware/workload drift beyond anchor similarity | Tune new task if no record. | The exact source predicate is in source_rule_fact; model-family envelope: KV transfer/disagg | H4.1,H4.4 | FROZEN_LEDGER |
| HX04 | TVM TIR | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | MetaSchedule documentation states that it searches schedule design spaces and measures on real hardware; the snapshot exposes cost-model choices (including XGBoost/MLP/random), runners and … | TIR features; candidates; measured time | whole-serving temporal/persistent state | Best database record; DLight as zero-search alternative. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H4.1 | FROZEN_LEDGER |
| HT06 | TensorRT-LLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | The XQA heuristic computes work from num_kv_heads * batch_size * multiBlockCount(history) and compares it against an SM-count-derived threshold; force/speculative/MLA cases can bypass the h… | batch; Hkv; history; SM | persistent layout; MoE/network contention | Masked MHA/single-block alternative. | The exact source predicate is in source_rule_fact; model-family envelope: MLA, Speculative | H4.1,H4.2,H4.5 | FROZEN_LEDGER |
| T-A01 | TensorRT-LLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | TRT-LLM is the production/default backend family; Vanilla and FlashInfer are available alternatives/configurations. | Configured backend and feature support. | Live workload at top-level family choice. | Supported configured path or documented fallback/error. | The exact source predicate is in source_rule_fact; model-family envelope: Generic attention | H4.1,H4.3 | FROZEN_LEDGER |
| T-A03 | TensorRT-LLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Context attention can choose different algorithms for short versus larger sequences. | Context length and support predicates. | Global cache/transfer/MoE objective. | Alternate context algorithm. | The exact source predicate is in source_rule_fact; model-family envelope: MHA | H4.1 | FROZEN_LEDGER |
| T-A04 | TensorRT-LLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | XQA first checks support, then a performance heuristic; XQA may also be forced when legacy masked-MHA cannot support a dtype combination. | Support predicates, Q/KV dtype, runtime config. | Global conversion/transfer and other-layer contention. | Masked MHA when valid and heuristic says XQA is not useful; otherwise XQA. | The exact source predicate is in source_rule_fact; model-family envelope: MHA | H4.1,H4.4,H4.5 | FROZEN_LEDGER |
| T-A05 | TensorRT-LLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | XQA estimates available work approximately from KV-heads × batch-size × multi-block-count; multi-block count grows with history length; compares this against an SM-derived threshold. | KV-head count, batch size, history length, multi-block state, SM count. | Layout transaction efficiency, L2 locality, prefix reuse, MoE overlap. | Use masked-MHA path if supported. | The exact source predicate is in source_rule_fact; model-family envelope: MHA, MQA, GQA | H4.1,H4.2,H4.3,H4.5 | FROZEN_LEDGER |
| T-A06 | TensorRT-LLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Multi-block attention increases parallel work when batch×heads underutilizes SMs and can be forced by shared-memory limits. | Batch, head count, SM count, history/shared-memory needs. | Layout/coalescing and system contention. | Single-block mode if enough parallelism. | The exact source predicate is in source_rule_fact; model-family envelope: Generic attention | H4.1 | FROZEN_LEDGER |
| T-B1 | TensorRT-LLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | TRT-LLM attention is the production/default backend family unless configured otherwise. | backend configuration and feature support | live workload at family/layout commitment time | configured supported path / explicit fallback | The exact source predicate is in source_rule_fact; model-family envelope: MHA, GQA, MLA | H4.1,H4.3 | FROZEN_LEDGER |
| HR01 | Triton | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | The tutorial declares finite lists of BLOCK_SIZE_M/N/K, GROUP_SIZE_M, stage and warp configurations and autotunes them using M/N/K as the key. | M,N,K key; config list | unlisted tiles/swizzles; graph state | Best measured declared config. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H4.1,H4.5 | FROZEN_LEDGER |
| HV01 | vLLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | The design doc states that RoPE+KV-cache update fusion is ROCm/AITER-specific and, by default, applies when num_tokens <= 256; the maximum is configurable. | num_tokens; platform/backend | head geometry; KV layout; concurrent load; reuse | Do not fuse above threshold / unsupported platform. | The exact source predicate is in source_rule_fact; model-family envelope: Paged KV, Generic attention | H4.1,H4.2,H4.3 | FROZEN_LEDGER |
| V-A02 | vLLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Standard-attention priority differs by GPU generation: Blackwell begins with FlashInfer; Ampere/Hopper begins with FlashAttention. | Compute capability, non-causal flag, multimodal-prefix state. | Batch/KV-length crossover, GQA ratio, prefix hit rate, transfer pressure. | Next compatible candidate. | The exact source predicate is in source_rule_fact; model-family envelope: MHA, MQA, GQA | H4.1,H4.2,H4.4 | FROZEN_LEDGER |
| V-A05 | vLLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Source comments acknowledge a TokenSpeed MLA batch-size crossover, but the top-level priority order itself does **not** read live batch size. | Platform/backend static order. | Live batch size, queue composition, SLA objective. | Static next-candidate fallback, not a batch-conditioned switch. | The exact source predicate is in source_rule_fact; model-family envelope: MLA | H4.1,H4.2 | FROZEN_LEDGER |
| V-A11 | vLLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Certain backend + batch-invariance/prefix-cache/chunk-lookback combinations disable a feature, warn, or require a specific backend. | Feature flags and backend identity. | Lost prefix-reuse opportunity cost. | Disable feature, warn, or reject incompatible backend. | The exact source predicate is in source_rule_fact; model-family envelope: Generic attention | H4.1 | FROZEN_LEDGER |
| V-B1 | vLLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Explicit backend is validated then used. | backend name, dtype/KV dtype, head/block size, feature/capability predicates | live shape distribution, prefix reuse, conversion/transfer/system contention | hard error when explicit backend is incompatible | The exact source predicate is in source_rule_fact; model-family envelope: Generic attention | H4.1,H4.4 | FROZEN_LEDGER |
| V-B2 | vLLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Automatic selection takes the first compatible backend in a platform priority list. | SM generation, attention family, dtype/KV dtype, head/block size, feature flags | live batch/q_len/KV_len, GQA ratio as cost term, reuse, contention, conversion | try next backend; error if none valid | The exact source predicate is in source_rule_fact; model-family envelope: MHA, MQA, GQA, MLA | H4.1,H4.2 | FROZEN_LEDGER |
| S12-04 | SGLang | DIRECT_MECHANISM | HiCache documents page size as KV storage/retrieval granularity: larger pages reduce metadata overhead and improve storage I/O efficiency but can reduce partial-page cache-hit rate; long co… | configured page size and coarse workload prefix structure | exact request distribution, GPU-kernel locality, fragmentation, current bandwidth/contention, dynamic page-size switching cost | deployment selects a fixed page size; no source-level dynamic optimizer is claimed | HiCache host/storage KV caching; long-common-prefix versus diverse-prefix workloads | H4.1,H4.4,H4.5 | CURRENT_SOURCE_SUPPLEMENT |
| S12-10 | vLLM | DIRECT_MECHANISM | Current indexed vLLM source keeps a generalized default KV-layout preference beginning with LBNHC, LBHNC, BLNHC and resolves supported layouts through backend-declared capability sets rathe… | backend-supported layout sets and fixed preference order | live execution weights, measured performance gaps, layouts not implemented/exposed by any selected backend | intersect available sets and choose from the supported preference order | vLLM paged attention layout resolution across backends; generalized LBNHC/LBHNC/BLNHC/... families | H4.1,H4.2,H4.3 | CURRENT_SOURCE_SUPPLEMENT |
| S13-02 | FlashInfer | DIRECT_MECHANISM | cuDNN paged decode buckets max_seq_len_kv and block-table width because these values are baked into the built graph; bucketing lets consecutive decode steps replay a cached graph instead of… | bucketed maximum KV length and block-table width | future sequence-length distribution, memory overhead of larger buckets, alternative page sizes/layouts, cross-request graph reuse lifetime | build/rebuild a graph for another bucket or use a non-captured/alternate path | CUDA-graph paged decode; changing KV length/page-table width; GQA/MHA decode families supported by the selected backend | H4.1,H4.3 | CURRENT_SOURCE_SUPPLEMENT |
| S13-03 | SGLang | DIRECT_MECHANISM | DeepSeek-v4 TRT-LLM attention requires a uniform-FP8 KV pool and requires a persistent TRT-LLM semaphore buffer to be created outside CUDA-graph capture during eager warmup. | uniform-FP8 pool contract and graph-capture lifecycle | future dtype/layout preference changes, migration/recapture cost under drift, mixed-layout pool alternatives | fail/route away from the incompatible backend contract or perform eager warmup/resource creation outside capture | DeepSeek-v4/MLA-like TRT-LLM attention path, FP8 KV pool, CUDA graph capture | H4.1,H4.2 | CURRENT_SOURCE_SUPPLEMENT |
| S13-06 | Triton | DIRECT_MECHANISM | Triton kernels inspect the storage layout of MX scale tensors and contain architecture-specific scale-layout/unswizzle implementations for Hopper/CDNA4/GFX1250; block-scaled matmul treats s… | scale tensor storage layout, precision config, microblock size, target architecture | upstream producer cost, serving-level reuse/fanout, alternative compound layouts, conversion amortization | canonicalize/convert scale layout or use another supported precision/kernel path | MXFP4/NVFP4/block-scaled GEMM on NVIDIA/AMD targets; dense MLP and MoE/grouped GEMM consumers | H4.1,H4.3,H4.4,H4.5 | CURRENT_SOURCE_SUPPLEMENT |


## 8.4 Modern model trigger audit

| Family | Concrete trigger for this RQ |
|---|---|
| MHA/MQA/GQA | Batch, q_len, KV_len and Hq/Hkv move local kernel/layout operating points. |
| MLA | Phase/model-specific cache geometry and backend crossover. |
| Sparse/DSA | Page/head/sparsity envelopes can alter viable/winning layouts. |
| MoE | Secondary through grouped-GEMM/scale layout and token-count-dependent tactics. |
| Hardware | Hopper/Blackwell/ROCm plus dtype/quantization are mandatory external-validity axes. |

## 8.5 Falsifiable hypotheses and direct falsifiers


| H | Hypothesis | What falsifies it | Experiment | Linked evidence rows |
|---|---|---|---|---|
| H4.1 | At least one workload/hardware axis causes a layout-winner crossover. | The same legal layout wins across all tested axes and hardware. | RQ4-E1 | 50 |
| H4.2 | Static preferences incur measurable regret under shifted workload distributions. | Static policy matches the oracle within noise on held-out distributions. | RQ4-E2 | 11 |
| H4.3 | At least one robust layout occupies a broad stable region. | Every region is highly fragmented with no robust winner. | RQ4-E1 | 16 |
| H4.4 | Structured workload+hardware features explain crossover better than GPU-name-only rules. | Simple GPU-name/static rules match richer models on held-out data. | RQ4-E3 | 19 |
| H4.5 | Local kernel adaptivity does not eliminate persistent-layout regret. | Once local dispatch is enabled, persistent-layout choice has no measurable residual regret. | RQ4-E4 | 15 |


## 8.6 Closure judgment

This RQ now satisfies the requested chain:

```text
source rule
  → explicit proxy
  → explicit omitted variables
  → framework-native fallback / repair / alternative
  → modern-model trigger envelope
  → falsifiable hypothesis + falsifier
```

The evidence does **not** assert that the hypothesis is true. The remaining unknown is empirical and is already assigned to the v11 experiment protocol.



# 9. L-RQ5 — Communication/placement-aware layout inversion

## 9.1 Trial status

**Verdict:** `COMPLETE_STRONG`

- Complete rule-chain rows: **18**
- Direct mechanism rows: **13**
- Direct stacks: **SGLang, TensorRT-LLM, vLLM**
- Current-source supplement rows: **4**
- Registered hypotheses: **H5.1, H5.2, H5.3, H5.4, H5.5**

## 9.2 Framework coverage judgment


| Framework/layer | Why it is or is not part of this RQ |
|---|---|
| vLLM | DIRECT: connector/offload direct/canonical/replicated representations. |
| SGLang | DIRECT: page-interleave sharding/all-gather scratch and MoE communication/compute compatibility. |
| TensorRT-LLM | DIRECT: disaggregated format transform, TP/DP/replication, Helix CP. |
| FlashInfer | N/A AS GLOBAL POLICY OWNER: exposes local kernels; global placement/sharding belongs upstream. |
| Compiler/kernel stacks | LOCAL BASELINE only; distributed rank ownership/network state is outside their audited scope. |


## 9.3 Complete source-rule → proxy → ignored → fallback → trigger → hypothesis chain


| ID | Stack | Role | Source rule/fact | Proxy | Ignored variables | Fallback/repair | Trigger condition | Hypothesis link | Source status |
|---|---|---|---|---|---|---|---|---|---|
| HS05 | SGLang | DIRECT_MECHANISM | MoE communication and compute backends are represented as independent backend families with compatibility/configuration logic. | parallel config; model; hardware; quant | token ordering; A2A payload layout; expert GEMM packing | Validator rejection / other backend. | The exact source predicate is in source_rule_fact; model-family envelope: MoE | H5.1,H5.5 | FROZEN_LEDGER |
| T-A08 | TensorRT-LLM | DIRECT_MECHANISM | Producer/consumer KV layouts and parallel formats can be transformed during transfer. | Producer and consumer format/parallel config. | Where to place conversion under current load, conversion amortization. | Convert/format, otherwise unsupported path fails. | The exact source predicate is in source_rule_fact; model-family envelope: KV transfer/disagg | H5.1 | FROZEN_LEDGER |
| T-A09 | TensorRT-LLM | DIRECT_MECHANISM | Transfer can overlap computation. | Transfer stage and scheduler/connector capability. | Competition with MoE/copy engines. | Non-overlapped path. | The exact source predicate is in source_rule_fact; model-family envelope: KV transfer/disagg | H5.1,H5.3,H5.5 | FROZEN_LEDGER |
| T-A10 | TensorRT-LLM | DIRECT_MECHANISM | Attention can use TP or DP; documentation treats TP as suitable for smaller batches and attention-DP for larger throughput regimes. | Deployment parallel configuration / expected batch regime. | Layout-locality and per-rank sequence skew. | User/deployment chooses alternative. | The exact source predicate is in source_rule_fact; model-family envelope: Generic attention | H5.1,H5.4,H5.5 | FROZEN_LEDGER |
| T-A11 | TensorRT-LLM | DIRECT_MECHANISM | With GQA/MQA/MLA, KV may be replicated if KV-head count is smaller than TP size. | KV-head count, TP size, attention family. | Extra memory pressure and possible alternative DP policy. | Replicate per rank. | The exact source predicate is in source_rule_fact; model-family envelope: MQA, GQA, MLA | H5.1,H5.2,H5.5 | FROZEN_LEDGER |
| HV05 | vLLM | DIRECT_MECHANISM | The modular MoE design distinguishes activation formats and states that input activation format is determined by the All2All dispatch; it motivates modular composition because implementatio… | A2A; activation format; quantization; architecture | joint network payload + expert GEMM layout cost | Pick compatible family or reject. | The exact source predicate is in source_rule_fact; model-family envelope: MoE | H5.1,H5.5 | FROZEN_LEDGER |
| V-A09 | vLLM | DIRECT_MECHANISM | Connector-specific paths can prefer HND for transfer while generic/default representations may be NHD-like. | Connector identity/mode, configured preference. | Actual link bandwidth, message size, local-kernel penalty. | Drop incompatible connector preference or retain local candidate. | The exact source predicate is in source_rule_fact; model-family envelope: KV transfer/disagg, Generic attention | H5.1,H5.4 | FROZEN_LEDGER |
| V-A10 | vLLM | DIRECT_MECHANISM | Uniform cross-layer blocks require a conjunction of transfer presence, connector preference, simple compatible attention grouping/spec, quantization compatibility and backend block-stride i… | Connector preference, group count/spec type, quantization, backend indexing. | Measured registration/copy cost, layer execution weight, network packetization. | Ordinary per-layer organization. | The exact source predicate is in source_rule_fact; model-family envelope: KV transfer/disagg, Quantized KV/activation, Generic attention | H5.1 | FROZEN_LEDGER |
| V-L6 | vLLM | DIRECT_MECHANISM | KV connector preferred layout is advisory and only used if locally compatible. | connector preference plus local candidate set | link bandwidth, transfer frequency, overlap, remote/native speedup | drop connector preference with warning | The exact source predicate is in source_rule_fact; model-family envelope: KV transfer/disagg, Generic attention | H5.1,H5.3,H5.4 | FROZEN_LEDGER |
| HT06 | TensorRT-LLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | The XQA heuristic computes work from num_kv_heads * batch_size * multiBlockCount(history) and compares it against an SM-count-derived threshold; force/speculative/MLA cases can bypass the h… | batch; Hkv; history; SM | persistent layout; MoE/network contention | Masked MHA/single-block alternative. | The exact source predicate is in source_rule_fact; model-family envelope: MLA, Speculative | H5.1,H5.2,H5.5 | FROZEN_LEDGER |
| T-A05 | TensorRT-LLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | XQA estimates available work approximately from KV-heads × batch-size × multi-block-count; multi-block count grows with history length; compares this against an SM-derived threshold. | KV-head count, batch size, history length, multi-block state, SM count. | Layout transaction efficiency, L2 locality, prefix reuse, MoE overlap. | Use masked-MHA path if supported. | The exact source predicate is in source_rule_fact; model-family envelope: MHA, MQA, GQA | H5.1,H5.3,H5.4,H5.5 | FROZEN_LEDGER |
| T-A06 | TensorRT-LLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Multi-block attention increases parallel work when batch×heads underutilizes SMs and can be forced by shared-memory limits. | Batch, head count, SM count, history/shared-memory needs. | Layout/coalescing and system contention. | Single-block mode if enough parallelism. | The exact source predicate is in source_rule_fact; model-family envelope: Generic attention | H5.1 | FROZEN_LEDGER |
| T-A12 | TensorRT-LLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Attention and MoE can use different parallel decompositions; rank imbalance and EP communication can dominate iteration time. | TP/DP/EP configuration, per-rank workload. | Layout choice is not usually a direct input to the MoE/ADP policy. | Balancing, disaggregation, alternate parallel configuration. | The exact source predicate is in source_rule_fact; model-family envelope: GQA, MLA, MoE | H5.1,H5.5 | FROZEN_LEDGER |
| HV03 | vLLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | AsyncTP depends on the Sequence Parallelism rewrite; the doc states AsyncTP is a no-op when SP has not been applied. | pass state; tokens; hidden size; platform | alternative rewrite order; producer layout; communication topology | No AsyncTP when prerequisite not met. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H5.1 | FROZEN_LEDGER |
| S12-01 | vLLM | DIRECT_MECHANISM | OffloadingConfig explicitly carries tokens_per_block, blocks_per_chunk, resolved worker KV layout, canonical_layout, replicated_layout, and whether persisted bytes are parallelism-agnostic;… | configured block/chunk granularity, parallel topology, pure-MLA/TP-only replication condition, canonical-layout request | live network bandwidth, cache-hit/reuse distribution, conversion cost, host-cache locality, future topology changes | representation mode is configured/certified rather than jointly cost-optimized; unsupported portability must use a compatible represen… | KV offload/disaggregation for MHA/GQA/MLA; multi-rank TP/PP/PCP/DCP; canonical/direct/replicated host representation | H5.1,H5.2,H5.4,H5.5 | CURRENT_SOURCE_SUPPLEMENT |
| S12-02 | vLLM | DIRECT_MECHANISM | HiSparse partitions sparse-MLA source and indexer specs, builds source/indexer/hot/resident groups, requires one resolved GPU block size, requires one page size within each hot-cache group,… | cache role, block size, page_size_bytes, indexer-page budget, host budget, block-outermost layout capability | live sparsity distribution, per-layer access frequency, host/device bandwidth variation, dynamic hot-set size, alternative grouping costs | raises on incompatible block/page/layout conditions; forms compatible groups and computes host/device layouts | HiSparse sparse-MLA with source+indexer caches, hot/resident device caches and host-resident source state | H5.1 | CURRENT_SOURCE_SUPPLEMENT |
| S12-03 | SGLang | DIRECT_MECHANISM | Page-interleave KV sharding stripes logical pages across ranks, all-gathers prefix pages one layer ahead into a [prefix\|chunk\|trash] scratch layout, stages the current chunk locally, and ma… | shard_size/rank, page_size, aligned prefix/chunk lengths, page ownership, fixed scratch capacity | runtime interconnect contention, overlap variation, scratch-memory pressure, alternative page placement/layouts, heterogeneous link topology | unsupported direct prefix-valid writes raise; data are assembled into scratch through the dedicated gather path | Sharded prefill for GQA or MLA; logical-page KV sharding; prefix all-gather and per-layer scratch | H5.1,H5.3,H5.4,H5.5 | CURRENT_SOURCE_SUPPLEMENT |
| S13-04 | TensorRT-LLM | DIRECT_MECHANISM | Helix context partitioning assigns token blocks round-robin across CP ranks using tokens_per_block and changes both prefill→decode KV movement and how KV grows during decode when generation… | tokens_per_block, CP rank count, generation CP mode | live interconnect contention, unequal request lengths, alternative block/page sizes, replication alternatives, per-rank memory skew | use the existing non-Helix/non-CP KV-cache infrastructure when the Helix mode is not enabled | multi-million-token long-context decode, generation CP > 1, distributed paged KV transfer/growth | H5.1,H5.2,H5.5 | CURRENT_SOURCE_SUPPLEMENT |


## 9.4 Modern model trigger audit

| Family | Concrete trigger for this RQ |
|---|---|
| MHA/MQA/GQA | TP/DP/DCP, Hkv<TP replication, wire/offload layout. |
| MLA | Replicated latent KV, disaggregated transfer, HiSparse/long-context sharding. |
| Sparse/DSA | Distributed page/index/source roles can change transfer representation. |
| MoE | A2A payload layout vs expert-GEMM-native representation under EP/rank skew. |
| Speculative | Secondary through larger/multiple KV consumers; not required for primary claim. |

## 9.5 Falsifiable hypotheses and direct falsifiers


| H | Hypothesis | What falsifies it | Experiment | Linked evidence rows |
|---|---|---|---|---|
| H5.1 | Adding communication/placement creates a local-vs-system layout rank inversion. | System winner always equals single-GPU local winner. | RQ5-E2 | 18 |
| H5.2 | KV replication under Hkv<TP changes the preferred layout/page organization via memory pressure. | Replication changes memory use but never changes layout ranking. | RQ5-E3 | 4 |
| H5.3 | Overlap can reverse whether explicit format conversion is worthwhile. | Conversion decision is invariant to overlap scheduling. | RQ5-E4 | 4 |
| H5.4 | Removing communication/placement terms makes local/system rankings converge. | They remain different when communication and placement costs are neutralized. | RQ5-E0 | 6 |
| H5.5 | Max-rank critical-path objective predicts EP/DP choices better than mean-rank latency. | Mean-rank and max-rank objectives choose equally well under skew. | RQ5-E5 | 11 |


## 9.6 Closure judgment

This RQ now satisfies the requested chain:

```text
source rule
  → explicit proxy
  → explicit omitted variables
  → framework-native fallback / repair / alternative
  → modern-model trigger envelope
  → falsifiable hypothesis + falsifier
```

The evidence does **not** assert that the hypothesis is true. The remaining unknown is empirical and is already assigned to the v11 experiment protocol.



# 10. L-RQ6 — Persistent-layout reconfiguration threshold

## 10.1 Trial status

**Verdict:** `COMPLETE_CONDITIONAL_ON_RQ4`

- Complete rule-chain rows: **13**
- Direct mechanism rows: **13**
- Direct stacks: **FlashInfer, IREE, SGLang, TensorRT-LLM, vLLM**
- Current-source supplement rows: **4**
- Registered hypotheses: **H6.1, H6.2, H6.3, H6.4, H6.5**

## 10.2 Framework coverage judgment


| Framework/layer | Why it is or is not part of this RQ |
|---|---|
| vLLM | DIRECT: resolved layout enters profiling/allocation and persists as cache configuration. |
| SGLang | DIRECT: graph-capture lifecycle, persistent semaphore/uniform FP8 pool contract. |
| TensorRT-LLM | DIRECT: persistent cache/block geometry in executor/cache manager. |
| FlashInfer | DIRECT: wrapper planning, graph buckets and rebuild avoidance. |
| IREE | COUNTEREVIDENCE: runtime-size-dependent encoding shows dynamic layout resolution is representationally possible. |
| Other compiler stacks | N/A to persistent serving migration; compile-time retuning is not equivalent to live KV migration. |


## 10.3 Complete source-rule → proxy → ignored → fallback → trigger → hypothesis chain


| ID | Stack | Role | Source rule/fact | Proxy | Ignored variables | Fallback/repair | Trigger condition | Hypothesis link | Source status |
|---|---|---|---|---|---|---|---|---|---|
| F-A05 | FlashInfer | DIRECT_MECHANISM | Graph-compatible wrappers can avoid shape-dependent kernel switching to preserve replay. | Graph mode/planned workspace metadata. | Shape distribution and replan/recapture cost. | More flexible non-graph wrapper. | The exact source predicate is in source_rule_fact; model-family envelope: Speculative | H6.1,H6.2,H6.5 | FROZEN_LEDGER |
| F-A08 | FlashInfer | DIRECT_MECHANISM | Workspace/metadata planning and graph capture constrain what can change without re-planning. | Planned batch metadata, workspace, graph state. | Future queue composition. | Re-plan or use flexible wrapper. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H6.1,H6.2,H6.5,H6.4 | FROZEN_LEDGER |
| F-A09 | FlashInfer | DIRECT_MECHANISM | Batch decode wrapper pre-plans metadata/workspace; with CUDA Graph enabled the wrapper documents that batch_size cannot change during its lifecycle. | planned batch metadata, workspace, graph flag, configured layout | future queue/workload drift and layout crossover | re-plan/recreate wrapper or use non-graph mode | The exact source predicate is in source_rule_fact; model-family envelope: Speculative | H6.1,H6.2,H6.5,H6.4 | FROZEN_LEDGER |
| HF05 | FlashInfer | DIRECT_MECHANISM | When CUDA Graph mode is enabled, the wrapper documentation states that batch_size cannot change during the wrapper lifecycle; caller-owned buffers/workspace are planned around that lifetime… | max tokens/batch; graph mode | workload drift and reconfiguration benefit | Recreate/replan or non-graph path. | The exact source predicate is in source_rule_fact; model-family envelope: Paged KV | H6.1,H6.2,H6.5,H6.4 | FROZEN_LEDGER |
| HI05 | IREE | DIRECT_MECHANISM | IREE set_encoding/unset_encoding accept dynamic encoding_dims (e.g. M/N/K); the operation description explicitly states these values are used for runtime layout selection based on problem s… | runtime dimensions | prefix reuse; queue; network/MoE contention | Resolver/default behavior. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H6.1,H6.4 | FROZEN_LEDGER |
| S-A11 | SGLang | DIRECT_MECHANISM | Attention backend/page size are ServerArgs launch choices; hybrid backend is constructed from resolved prefill/decode choices; decode backend is CUDA-graph captured in documented hybrid/spe… | startup hardware/model/config, page size, phase backend names, graph mode | diurnal batch/KV distribution, prefix reuse drift, speculation acceptance drift | change launch config/restart; non-graph or differently captured path where available | The exact source predicate is in source_rule_fact; model-family envelope: Speculative, Paged KV, Generic attention | H6.1,H6.2,H6.5,H6.3,H6.4 | FROZEN_LEDGER |
| T-A13 | TensorRT-LLM | DIRECT_MECHANISM | tokens_per_block is model/executor/KV-manager configuration; manager construction receives fixed tokens_per_block and geometry. Legacy documentation also describes block-size selection at b… | build/launch config, attention/KV geometry | runtime prefix-reuse distribution, workload phase drift, current fragmentation | rebuild/recreate executor/manager/configure another run | The exact source predicate is in source_rule_fact; model-family envelope: MLA, Paged KV | H6.1,H6.3,H6.4 | FROZEN_LEDGER |
| V-A13 | vLLM | DIRECT_MECHANISM | Engine core resolves KV layout before memory profiling/full graph-related KV initialization; record_kv_cache_layout refuses changing an already-resolved layout. | startup model/backend supported-layout set, cache specs, connector/env preference | later workload drift: batch, phase mix, prefix hit rate, transfer/MoE pressure | restart/reinitialize/reallocate is effectively required for a persistent-layout change | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H6.1,H6.2,H6.5,H6.3,H6.4 | FROZEN_LEDGER |
| V-L4 | vLLM | DIRECT_MECHANISM | Resolve one KV-cache layout for the whole model. | worker candidate lists, cache specs/config | per-group/phase/consumer optimum and workload drift | mismatch/error; resolved layout is recorded | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H6.1,H6.3,H6.4 | FROZEN_LEDGER |
| S12-01 | vLLM | DIRECT_MECHANISM | OffloadingConfig explicitly carries tokens_per_block, blocks_per_chunk, resolved worker KV layout, canonical_layout, replicated_layout, and whether persisted bytes are parallelism-agnostic;… | configured block/chunk granularity, parallel topology, pure-MLA/TP-only replication condition, canonical-layout request | live network bandwidth, cache-hit/reuse distribution, conversion cost, host-cache locality, future topology changes | representation mode is configured/certified rather than jointly cost-optimized; unsupported portability must use a compatible represen… | KV offload/disaggregation for MHA/GQA/MLA; multi-rank TP/PP/PCP/DCP; canonical/direct/replicated host representation | H6.1,H6.3,H6.4 | CURRENT_SOURCE_SUPPLEMENT |
| S13-01 | vLLM | DIRECT_MECHANISM | EngineCore writes the resolved cache layout into KVCacheConfig, and GPUModelRunner uses get_resolved_kv_cache_layout() while creating/profiling KV cache state. | resolved deployment-time KV cache layout and cache configuration | future workload-regime drift, migration cost/benefit, per-phase re-layout, remaining request horizon | reinitialize/reconfigure the engine/cache state rather than changing the allocated layout per request | Persistent paged MHA/MQA/GQA/MLA cache initialization; any serving mode using resolved KV layout before profiling/allocation | H6.1,H6.3,H6.4 | CURRENT_SOURCE_SUPPLEMENT |
| S13-02 | FlashInfer | DIRECT_MECHANISM | cuDNN paged decode buckets max_seq_len_kv and block-table width because these values are baked into the built graph; bucketing lets consecutive decode steps replay a cached graph instead of… | bucketed maximum KV length and block-table width | future sequence-length distribution, memory overhead of larger buckets, alternative page sizes/layouts, cross-request graph reuse lifetime | build/rebuild a graph for another bucket or use a non-captured/alternate path | CUDA-graph paged decode; changing KV length/page-table width; GQA/MHA decode families supported by the selected backend | H6.1,H6.2,H6.5,H6.4 | CURRENT_SOURCE_SUPPLEMENT |
| S13-03 | SGLang | DIRECT_MECHANISM | DeepSeek-v4 TRT-LLM attention requires a uniform-FP8 KV pool and requires a persistent TRT-LLM semaphore buffer to be created outside CUDA-graph capture during eager warmup. | uniform-FP8 pool contract and graph-capture lifecycle | future dtype/layout preference changes, migration/recapture cost under drift, mixed-layout pool alternatives | fail/route away from the incompatible backend contract or perform eager warmup/resource creation outside capture | DeepSeek-v4/MLA-like TRT-LLM attention path, FP8 KV pool, CUDA graph capture | H6.1,H6.2,H6.5,H6.3,H6.4 | CURRENT_SOURCE_SUPPLEMENT |


## 10.4 Modern model trigger audit

| Family | Concrete trigger for this RQ |
|---|---|
| MHA/MQA/GQA | Persistent paged KV layout/page geometry under workload drift. |
| MLA | Uniform FP8/latent pool plus graph-capture resources; long-running decode regimes. |
| Sparse/DSA | Persistent sparse source/indexer pools can be migration targets, but not needed for first experiment. |
| MoE | Not a primary persistent-KV trigger; can apply to tuned persistent MoE buffers in later work. |
| Speculative | Acceptance/q_len distribution drift can change preferred persistent representation and graph bucket. |

## 10.5 Falsifiable hypotheses and direct falsifiers


| H | Hypothesis | What falsifies it | Experiment | Linked evidence rows |
|---|---|---|---|---|
| H6.1 | A measurable switching horizon N* exists. | No horizon crossover occurs between staying and switching. | RQ6-E2 | 13 |
| H6.2 | Per-request switching loses to epochal/hysteretic policies under realistic switch cost. | Immediate per-request switching remains best after all switch costs are charged. | RQ6-E3 | 8 |
| H6.3 | Stationary traffic favors no reconfiguration. | Switching repeatedly improves stationary traces beyond noise. | RQ6-E0 | 7 |
| H6.4 | Persistent drift can justify expensive migration. | Even long stable drift never amortizes migration. | RQ6-E2 | 12 |
| H6.5 | Graph recapture/duplicate-live memory materially move N*. | Removing these terms leaves the measured threshold unchanged. | RQ6-E4 | 8 |


## 10.6 Closure judgment

This RQ now satisfies the requested chain:

```text
source rule
  → explicit proxy
  → explicit omitted variables
  → framework-native fallback / repair / alternative
  → modern-model trigger envelope
  → falsifiable hypothesis + falsifier
```

The evidence does **not** assert that the hypothesis is true. The remaining unknown is empirical and is already assigned to the v11 experiment protocol.



**Conditionality:** RQ6 must be stopped or narrowed if RQ4 finds no meaningful persistent-layout crossover. This is not an evidence incompleteness; it is a deliberate causal dependency.


# 11. L-RQ7 — Layout equivalence and zero-copy transformability

## 11.1 Trial status

**Verdict:** `COMPLETE_STRONG`

- Complete rule-chain rows: **17**
- Direct mechanism rows: **16**
- Direct stacks: **CUTLASS/CuTe, FlashInfer, IREE, MLIR, SGLang, TVM Relax, TensorRT-LLM**
- Current-source supplement rows: **7**
- Registered hypotheses: **H7.1, H7.2, H7.3, H7.4**

## 11.2 Framework coverage judgment


| Framework/layer | Why it is or is not part of this RQ |
|---|---|
| vLLM | INDIRECT ONLY: capability sets gate legal layouts but do not provide a general zero-copy equivalence solver. |
| SGLang | DIRECT: vectorized-5D and sharded scratch paths define non-view/specialized cases. |
| TensorRT-LLM | DIRECT: sparse HND/page hard contracts. |
| FlashInfer | DIRECT: NHD zero-copy transposed-stride view and sparse materialization/remap countercases. |
| CUTLASS/MLIR/IREE/TVM | DIRECT: instruction/alignment/topology/index-map legality. |
| Triton | LOW-LEVEL BASELINE; explicit layout lowering can participate but the frozen RQ7 chain does not require it. |


## 11.3 Complete source-rule → proxy → ignored → fallback → trigger → hypothesis chain


| ID | Stack | Role | Source rule/fact | Proxy | Ignored variables | Fallback/repair | Trigger condition | Hypothesis link | Source status |
|---|---|---|---|---|---|---|---|---|---|
| HC04 | CUTLASS/CuTe | DIRECT_MECHANISM | CUTLASS documents an efficient row-major epilogue and, for column-major output, transposes/swaps the equivalent GEMM operands so the efficient epilogue mapping is preserved. | input/output layout | downstream consumer layout/fusion | Specialized template path. | The exact source predicate is in source_rule_fact; model-family envelope: Dense MLP/linear | H7.1 | FROZEN_LEDGER |
| HC05 | CUTLASS/CuTe | DIRECT_MECHANISM | CuTe DSL helpers fail fast on unsupported instruction/architecture combinations and enforce TMA contiguous-dimension alignment before performance choices are made. | swizzle; instruction; architecture | higher-level graph opportunity cost | Pick legal atom/template. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H7.3 | FROZEN_LEDGER |
| F-A06 | FlashInfer | DIRECT_MECHANISM | Some kernels can consume NHD-contiguous or HND-transposed/arbitrary supported strides without materialized conversion. | Strides and innermost head-dim condition. | Cache/TLB efficiency of each stride pattern. | Another kernel or reject unsupported stride. | The exact source predicate is in source_rule_fact; model-family envelope: GQA, Paged KV | H7.1,H7.3,H7.4 | FROZEN_LEDGER |
| HI02 | IREE | DIRECT_MECHANISM | IREE emits compiler pipeline constraints for legal knob combinations, serializes them to SMT-LIB in the tuner, and uses an SMT solver to enumerate valid assignments; an experimental verifie… | compiler constraints | cross-dispatch cost and online serving state | SMT excludes illegal assignments; verifier checks. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H7.1,H7.3 | FROZEN_LEDGER |
| HI06 | IREE | DIRECT_MECHANISM | IREE's GPU shared-memory reuse pass computes allocation liveness ranges, groups overlapping lifetimes, and reuses allocations only under explicit structural preconditions; a separate GPU pa… | resource limits; liveness; layout ops | whole-serving persistent state | Alter/reject/combine/reuse. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H7.2 | FROZEN_LEDGER |
| HM03 | MLIR | DIRECT_MECHANISM | structured.pack_transpose can transpose a pack/compute(/unpack) chain; for pack it expects the consuming linalg.generic to be the sole consumer, and unsupported topology/specification yield… | pack chain; permutations; consumer topology | multi-consumer/persistent state/global cost | Silenceable failure. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H7.1,H7.2 | FROZEN_LEDGER |
| S-A09 | SGLang | DIRECT_MECHANISM | `vectorized_5d` cache layout routes writes/decode through different compatible paths because a 4D view is not representable. | Pool layout tag, backend/platform. | Cost of keeping 5D vs converting; workload crossover. | 5D-compatible writer/decode path. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H7.1,H7.3,H7.4 | FROZEN_LEDGER |
| HX05 | TVM Relax | DIRECT_MECHANISM | Relax layout_transform is an explicit graph operator taking an IndexMap and optional padding and materializing a transformed tensor. | index map; tensor shape | producer/consumer could avoid conversion jointly | Keep/lower/fuse transform. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H7.1,H7.3 | FROZEN_LEDGER |
| T-A07 | TensorRT-LLM | DIRECT_MECHANISM | Specialized sparse kernels have hard HND/page-size/geometry constraints; unsupported classes can fall back to another sparse path or error. | Sparse algorithm, page/layout, dtype/head geometry. | Conversion cost from general layout. | Alternative sparse implementation or hard error. | The exact source predicate is in source_rule_fact; model-family envelope: MHA, MQA, GQA, MLA, Sparse/DSA | H7.3 | FROZEN_LEDGER |
| HT03 | TensorRT-LLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | The transform explicitly uses direct FX graph manipulation rather than the ordinary pattern matcher so that intermediate add/RMSNorm values with multiple users are handled correctly. | use-def topology | general clone/recompute/layout alternatives | Bespoke transform for known case. | The exact source predicate is in source_rule_fact; model-family envelope: Dense MLP/linear | H7.2 | FROZEN_LEDGER |
| S12-03 | SGLang | DIRECT_MECHANISM | Page-interleave KV sharding stripes logical pages across ranks, all-gathers prefix pages one layer ahead into a [prefix\|chunk\|trash] scratch layout, stages the current chunk locally, and ma… | shard_size/rank, page_size, aligned prefix/chunk lengths, page ownership, fixed scratch capacity | runtime interconnect contention, overlap variation, scratch-memory pressure, alternative page placement/layouts, heterogeneous link topology | unsupported direct prefix-valid writes raise; data are assembled into scratch through the dedicated gather path | Sharded prefill for GQA or MLA; logical-page KV sharding; prefix all-gather and per-layer scratch | H7.2,H7.3 | CURRENT_SOURCE_SUPPLEMENT |
| S12-05 | SGLang | DIRECT_MECHANISM | SGLang exposes a page-major KV layout that places Mamba state and full/SWA KV caches in one page-granularity envelope (page outermost, layer-major within page) instead of the default per-la… | explicit enable flag plus backend compatibility | per-state access frequency, model/workload-specific profitability, alternative hybrid-state domain partitions | default per-layer layout remains available when page-major mode is disabled/incompatible | Hybrid Mamba + full attention/SWA models using Triton attention/linear-attention/Mamba backends | H7.3 | CURRENT_SOURCE_SUPPLEMENT |
| S12-06 | TensorRT-LLM | DIRECT_MECHANISM | QSA sparse GQA stores/reads paged KV through block_table and tokens_per_block, consumes selected sparse tokens, requires query-head divisibility by local KV heads, and chooses a fused CUDA … | tokens_per_block, block table, sparse selected-token metadata, request indices, head dimension, Q/KV head divisibility | alternative KV physical layouts, joint route/page-layout profitability, metadata cache locality, dynamic page-granularity selection | reference sparse GQA path when fused CUDA predicate is false; validation errors for incompatible shapes/metadata | QSA sparse GQA/MHA/MQA-compatible head topology with block tables, selected sparse tokens and paged KV | H7.2,H7.3 | CURRENT_SOURCE_SUPPLEMENT |
| S12-07 | FlashInfer | DIRECT_MECHANISM | For the cuDNN route, FlashInfer builds the graph from strides and can present an NHD cache as a transposed [pages, heads, page_size, head_dim] view without a copy. | backend route plus actual strides/layout | downstream reuse/fanout, copy-vs-view profitability on other backends, multi-consumer alias/lifetime constraints | use a compatible backend/path; non-view-compatible cases require another representation path | cuDNN paged decode with an NHD cache whose transposed strides satisfy the consumer graph | H7.1,H7.2,H7.3,H7.4 | CURRENT_SOURCE_SUPPLEMENT |
| S12-08 | FlashInfer | DIRECT_MECHANISM | Prefill wrappers expose returned LSE layout NH or HN; HN is a contiguous transpose and, for relevant backends, uses one fused transpose+rescale kernel; caller-provided LSE must match the re… | requested LSE layout and backend | which downstream merge consumer will reuse the LSE, fanout, total end-to-end value of producing HN directly | produce NH or pay the fused transpose/rescale path for HN according to the backend implementation | Attention prefill returning LSE for merge/normalization consumers; NH versus HN output layout | H7.1 | CURRENT_SOURCE_SUPPLEMENT |
| S12-09 | FlashInfer | DIRECT_MECHANISM | Sparse CuTe paths expose concrete layout incompatibilities/repairs: SM100 allocates a fresh contiguous BHSD output even when the caller has BSHD; SM120 explicitly remaps BSHD inputs into th… | architecture, sparse/Sage mode, BSHD/BHSD layout, tensor-map/instruction constraints | whole-graph consumer requirements, whether a different producer layout could remove the repair, search over alternative sparse physical layouts | fresh contiguous output or architecture-specific view/remap | Blackwell sparse/Sage attention with BSHD/BHSD or architecture-specific physical mappings | H7.1,H7.3,H7.4 | CURRENT_SOURCE_SUPPLEMENT |
| S13-05 | CUTLASS/CuTe | DIRECT_MECHANISM | CUTLASS Sm1xxBlockScaledConfig explicitly constructs SFA/SFB scale-factor tensor layouts for block-scaled Blackwell GEMM; scale-factor layouts are a first-class physical representation rath… | scale-vector size, operand/tile geometry, architecture-specific block-scaled MMA contract | upstream producer scale layout, cost of interleaving/repacking scales, downstream reuse, graph-level choice among alternative scale layouts | construct/use the required scale layout or choose a different legal GEMM/precision path | Blackwell NVFP4/MXFP block-scaled dense or sparse GEMM, including MoE/grouped-GEMM style consumers | H7.3 | CURRENT_SOURCE_SUPPLEMENT |


## 11.4 Modern model trigger audit

| Family | Concrete trigger for this RQ |
|---|---|
| MHA/MQA/GQA | NHD/HND and arbitrary-stride/view consumption. |
| MLA | Latent/vectorized layouts and specialized consumer contracts. |
| Sparse/DSA | BHSD/BSHD, special token/value permutations and paged HND constraints provide negative/materialization cases. |
| MoE | Scale/operand layouts can be viewable or require interleave/repack; secondary but relevant. |
| Hybrid | Page-major Mamba/full/SWA is a structured layout, not automatically a zero-copy equivalent of per-layer storage. |

## 11.5 Falsifiable hypotheses and direct falsifiers


| H | Hypothesis | What falsifies it | Experiment | Linked evidence rows |
|---|---|---|---|---|
| H7.1 | A non-trivial fraction of named-layout mismatches are zero-copy representable. | All mismatches require materialization or native re-emission. | RQ7-E1 | 9 |
| H7.2 | Multi-consumer/topology constraints reduce zero-copy equivalence. | Adding consumers/alias constraints does not change legal-view coverage. | RQ7-E3 | 6 |
| H7.3 | Alignment/instruction constraints predict materialization need better than layout names. | Layout-name equality/difference predicts legality equally well. | RQ7-E2 | 12 |
| H7.4 | More stride-polymorphic kernels reduce the need for separate layout domains. | Expanded stride support does not reduce conversion/domain splitting. | RQ7-E4 | 4 |


## 11.6 Closure judgment

This RQ now satisfies the requested chain:

```text
source rule
  → explicit proxy
  → explicit omitted variables
  → framework-native fallback / repair / alternative
  → modern-model trigger envelope
  → falsifiable hypothesis + falsifier
```

The evidence does **not** assert that the hypothesis is true. The remaining unknown is empirical and is already assigned to the v11 experiment protocol.



# 12. L-RQ8 — Coupled data–metadata co-layout

## 12.1 Trial status

**Verdict:** `COMPLETE_STRONG`

- Complete rule-chain rows: **23**
- Direct mechanism rows: **21**
- Direct stacks: **CUTLASS/CuTe, FlashInfer, SGLang, TensorRT-LLM, Triton, vLLM**
- Current-source supplement rows: **7**
- Registered hypotheses: **H8.1, H8.2, H8.3, H8.4, H8.5**

## 12.2 Framework coverage judgment


| Framework/layer | Why it is or is not part of this RQ |
|---|---|
| vLLM | DIRECT: quant/sparse/HiSparse source-indexer-hot-resident compound state. |
| SGLang | DIRECT: NVFP4 scale interleave and sparse/quantized representations. |
| TensorRT-LLM | DIRECT: QSA paged KV + block table + selected sparse token metadata. |
| FlashInfer | DIRECT: KV+scale transform and LSE/quantized representation. |
| CUTLASS/Triton | DIRECT: scale-factor storage layout is part of block-scaled MMA/matmul contract. |
| Generic compilers | REPRESENTATION SUBSTRATE only; no audited LLM-specific data+metadata profitability policy. |


## 12.3 Complete source-rule → proxy → ignored → fallback → trigger → hypothesis chain


| ID | Stack | Role | Source rule/fact | Proxy | Ignored variables | Fallback/repair | Trigger condition | Hypothesis link | Source status |
|---|---|---|---|---|---|---|---|---|---|
| F-A02 | FlashInfer | DIRECT_MECHANISM | NHD can be accepted but transposed + materialized contiguously as HND, including scale tensors. | Backend, NVFP4 KV, input layout. | Reuse count, memory pressure, possibility of native HND allocation. | Automatic transpose/contiguous copy. | The exact source predicate is in source_rule_fact; model-family envelope: Quantized KV/activation | H8.1,H8.3 | FROZEN_LEDGER |
| F-L2 | FlashInfer | DIRECT_MECHANISM | Some TRTLLM-gen NVFP4 paths accept NHD but transpose/materialize HND. | backend, KV dtype, input layout | reuse count, memory pressure, native-HND allocation option | transpose + contiguous copy | The exact source predicate is in source_rule_fact; model-family envelope: Quantized KV/activation | H8.1,H8.3 | FROZEN_LEDGER |
| HF02 | FlashInfer | DIRECT_MECHANISM | For trtllm-gen with NVFP4 KV cache, NHD input triggers automatic transpose plus contiguous copies of KV data and scale tensors to HND, with documented allocation/copy overhead. | layout; dtype; backend | reuse count; temporary memory; producer-native HND | Materialized transpose/copy. | The exact source predicate is in source_rule_fact; model-family envelope: Quantized KV/activation, Paged KV | H8.1,H8.3 | FROZEN_LEDGER |
| HF03 | FlashInfer | DIRECT_MECHANISM | Grouped-MoE GEMM APIs expose explicit tensor-shape/dtype contracts, backend dispatch and tactic selection; quantized variants add scale/format requirements. | dtype; architecture; scale/layout; tactic | whole graph and network representation | Backend/tactic or unsupported error. | The exact source predicate is in source_rule_fact; model-family envelope: MoE, Quantized KV/activation | H8.1,H8.3 | FROZEN_LEDGER |
| HS05 | SGLang | DIRECT_MECHANISM | MoE communication and compute backends are represented as independent backend families with compatibility/configuration logic. | parallel config; model; hardware; quant | token ordering; A2A payload layout; expert GEMM packing | Validator rejection / other backend. | The exact source predicate is in source_rule_fact; model-family envelope: MoE | H8.1 | FROZEN_LEDGER |
| HS06 | SGLang | DIRECT_MECHANISM | For NVFP4 + FlashInfer CUTLASS MoE, received scale tensors are explicitly passed through nvfp4_block_scale_interleave; a TODO says to fuse this interleave into CUTLASS MoE when supported. | runner backend; scale layout | amortization; direct producer-native scale layout | Materialize scale interleave. | The exact source predicate is in source_rule_fact; model-family envelope: MoE, Quantized KV/activation | H8.1,H8.3 | FROZEN_LEDGER |
| S-A08 | SGLang | DIRECT_MECHANISM | DSA is architecture-triggered and uses phase/platform/dtype-specific sub-backend rules. | Model architecture, phase, GPU, BF16/FP8, some parallel-state constraints. | Actual selected-token count/locality, sparse density, batch/KV length. | Alternative supported DSA backend or reject. | The exact source predicate is in source_rule_fact; model-family envelope: MLA, Sparse/DSA | H8.1,H8.2,H8.5,H8.3 | FROZEN_LEDGER |
| HT02 | TensorRT-LLM | DIRECT_MECHANISM | FuseSiluMul is a post-load transform intended to run after GEMM fusion; a suitable sole FP8-linear consumer enables further quantization fusion. | prior rewrite; consumer cardinality/type | joint gate/up packing + activation layout + down-GEMM format | Skip extra fusion if prerequisite absent. | The exact source predicate is in source_rule_fact; model-family envelope: Quantized KV/activation, Dense MLP/linear | H8.1,H8.3 | FROZEN_LEDGER |
| HT05 | TensorRT-LLM | DIRECT_MECHANISM | AutoDeploy contains explicit fused-MoE restructuring logic that converts between stacked/backend-specific weight/graph representations and enforces backend/style support constraints. | MoE graph/weight representation style, backend family, fused-MoE support predicates | Runtime expert-token distribution, restructuring/copy cost, peak temporary memory, alternative fused-MoE layout families. | Apply only supported fused-MoE restructuring/backend style; otherwise retain/choose a supported general or alternate path rather than … | The exact source predicate is in source_rule_fact; model-family envelope: MoE | H8.1,H8.3 | FROZEN_LEDGER |
| T-A07 | TensorRT-LLM | DIRECT_MECHANISM | Specialized sparse kernels have hard HND/page-size/geometry constraints; unsupported classes can fall back to another sparse path or error. | Sparse algorithm, page/layout, dtype/head geometry. | Conversion cost from general layout. | Alternative sparse implementation or hard error. | The exact source predicate is in source_rule_fact; model-family envelope: MHA, MQA, GQA, MLA, Sparse/DSA | H8.1,H8.2,H8.5 | FROZEN_LEDGER |
| HV05 | vLLM | DIRECT_MECHANISM | The modular MoE design distinguishes activation formats and states that input activation format is determined by the All2All dispatch; it motivates modular composition because implementatio… | A2A; activation format; quantization; architecture | joint network payload + expert GEMM layout cost | Pick compatible family or reject. | The exact source predicate is in source_rule_fact; model-family envelope: MoE | H8.1 | FROZEN_LEDGER |
| V-A04 | vLLM | DIRECT_MECHANISM | Sparse MLA preference changes with KV dtype and query-head count; BF16-like paths use a head-count threshold, while quantized KV favors a different sparse backend order. | KV dtype, query-head count, SM generation. | Selected-page count, sparse density, KV length, page locality, batch. | Next compatible sparse backend. | The exact source predicate is in source_rule_fact; model-family envelope: MLA, Sparse/DSA, Quantized KV/activation | H8.1,H8.2,H8.5 | FROZEN_LEDGER |
| V-A08 | vLLM | DIRECT_MECHANISM | Backend validation includes sinks, sparse, multimodal prefix, per-head scales, non-causal, batch-invariance, connector, PCP/DCP, adaptive verification, RSWA and other flags. | Feature booleans plus dtype/head/block geometry. | Continuous runtime workload and end-to-end contention. | Invalidate candidate and move down priority. | The exact source predicate is in source_rule_fact; model-family envelope: Sparse/DSA, KV transfer/disagg, Quantized KV/activation | H8.1,H8.2,H8.5,H8.3 | FROZEN_LEDGER |
| V-A10 | vLLM | DIRECT_MECHANISM | Uniform cross-layer blocks require a conjunction of transfer presence, connector preference, simple compatible attention grouping/spec, quantization compatibility and backend block-stride i… | Connector preference, group count/spec type, quantization, backend indexing. | Measured registration/copy cost, layer execution weight, network packetization. | Ordinary per-layer organization. | The exact source predicate is in source_rule_fact; model-family envelope: KV transfer/disagg, Quantized KV/activation, Generic attention | H8.1,H8.2,H8.5 | FROZEN_LEDGER |
| HR02 | Triton | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Grouped GEMM launches a fixed number of CTAs with static device-side scheduling and autotunes a finite set of BLOCK sizes and NUM_SM values keyed by group_size. | group size; tile configs; NUM_SM | expert skew drift; A2A payload representation | Autotune among configs. | The exact source predicate is in source_rule_fact; model-family envelope: MoE, Paged KV | H8.1 | FROZEN_LEDGER |
| HV04 | vLLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | The fusion design enumerates concrete graph patterns including AllReduce->RMSNorm, Attention->Quant, QK-Norm->RoPE, RoPE->KV write, RMSNorm->Quant and SiLU+Mul->Quant. | graph syntax/ops; backend; dtype; GPU | semantic variants; multi-user graphs; alternate output contracts | General unfused path. | The exact source predicate is in source_rule_fact; model-family envelope: Quantized KV/activation, Generic attention, Dense MLP/linear | H8.1 | FROZEN_LEDGER |
| S12-02 | vLLM | DIRECT_MECHANISM | HiSparse partitions sparse-MLA source and indexer specs, builds source/indexer/hot/resident groups, requires one resolved GPU block size, requires one page size within each hot-cache group,… | cache role, block size, page_size_bytes, indexer-page budget, host budget, block-outermost layout capability | live sparsity distribution, per-layer access frequency, host/device bandwidth variation, dynamic hot-set size, alternative grouping costs | raises on incompatible block/page/layout conditions; forms compatible groups and computes host/device layouts | HiSparse sparse-MLA with source+indexer caches, hot/resident device caches and host-resident source state | H8.1,H8.2,H8.5 | CURRENT_SOURCE_SUPPLEMENT |
| S12-06 | TensorRT-LLM | DIRECT_MECHANISM | QSA sparse GQA stores/reads paged KV through block_table and tokens_per_block, consumes selected sparse tokens, requires query-head divisibility by local KV heads, and chooses a fused CUDA … | tokens_per_block, block table, sparse selected-token metadata, request indices, head dimension, Q/KV head divisibility | alternative KV physical layouts, joint route/page-layout profitability, metadata cache locality, dynamic page-granularity selection | reference sparse GQA path when fused CUDA predicate is false; validation errors for incompatible shapes/metadata | QSA sparse GQA/MHA/MQA-compatible head topology with block tables, selected sparse tokens and paged KV | H8.1,H8.2,H8.5 | CURRENT_SOURCE_SUPPLEMENT |
| S12-08 | FlashInfer | DIRECT_MECHANISM | Prefill wrappers expose returned LSE layout NH or HN; HN is a contiguous transpose and, for relevant backends, uses one fused transpose+rescale kernel; caller-provided LSE must match the re… | requested LSE layout and backend | which downstream merge consumer will reuse the LSE, fanout, total end-to-end value of producing HN directly | produce NH or pay the fused transpose/rescale path for HN according to the backend implementation | Attention prefill returning LSE for merge/normalization consumers; NH versus HN output layout | H8.1,H8.3 | CURRENT_SOURCE_SUPPLEMENT |
| S12-09 | FlashInfer | DIRECT_MECHANISM | Sparse CuTe paths expose concrete layout incompatibilities/repairs: SM100 allocates a fresh contiguous BHSD output even when the caller has BSHD; SM120 explicitly remaps BSHD inputs into th… | architecture, sparse/Sage mode, BSHD/BHSD layout, tensor-map/instruction constraints | whole-graph consumer requirements, whether a different producer layout could remove the repair, search over alternative sparse physical layouts | fresh contiguous output or architecture-specific view/remap | Blackwell sparse/Sage attention with BSHD/BHSD or architecture-specific physical mappings | H8.1,H8.2,H8.5 | CURRENT_SOURCE_SUPPLEMENT |
| S13-03 | SGLang | DIRECT_MECHANISM | DeepSeek-v4 TRT-LLM attention requires a uniform-FP8 KV pool and requires a persistent TRT-LLM semaphore buffer to be created outside CUDA-graph capture during eager warmup. | uniform-FP8 pool contract and graph-capture lifecycle | future dtype/layout preference changes, migration/recapture cost under drift, mixed-layout pool alternatives | fail/route away from the incompatible backend contract or perform eager warmup/resource creation outside capture | DeepSeek-v4/MLA-like TRT-LLM attention path, FP8 KV pool, CUDA graph capture | H8.1,H8.2,H8.5,H8.3 | CURRENT_SOURCE_SUPPLEMENT |
| S13-05 | CUTLASS/CuTe | DIRECT_MECHANISM | CUTLASS Sm1xxBlockScaledConfig explicitly constructs SFA/SFB scale-factor tensor layouts for block-scaled Blackwell GEMM; scale-factor layouts are a first-class physical representation rath… | scale-vector size, operand/tile geometry, architecture-specific block-scaled MMA contract | upstream producer scale layout, cost of interleaving/repacking scales, downstream reuse, graph-level choice among alternative scale layouts | construct/use the required scale layout or choose a different legal GEMM/precision path | Blackwell NVFP4/MXFP block-scaled dense or sparse GEMM, including MoE/grouped-GEMM style consumers | H8.1,H8.2,H8.5,H8.3 | CURRENT_SOURCE_SUPPLEMENT |
| S13-06 | Triton | DIRECT_MECHANISM | Triton kernels inspect the storage layout of MX scale tensors and contain architecture-specific scale-layout/unswizzle implementations for Hopper/CDNA4/GFX1250; block-scaled matmul treats s… | scale tensor storage layout, precision config, microblock size, target architecture | upstream producer cost, serving-level reuse/fanout, alternative compound layouts, conversion amortization | canonicalize/convert scale layout or use another supported precision/kernel path | MXFP4/NVFP4/block-scaled GEMM on NVIDIA/AMD targets; dense MLP and MoE/grouped GEMM consumers | H8.1,H8.3 | CURRENT_SOURCE_SUPPLEMENT |


## 12.4 Modern model trigger audit

| Family | Concrete trigger for this RQ |
|---|---|
| MHA/MQA/GQA | Quantized KV + scales and page-table metadata. |
| MLA | Sparse/quantized latent cache + indexer/source metadata. |
| Sparse/DSA | Primary: KV + selected tokens/routes/block tables/page tables. |
| MoE | Primary: activation/weight + block scales + routing/permutation metadata. |
| Negative regime | Tiny cache-resident metadata should make the interaction negligible. |

## 12.5 Falsifiable hypotheses and direct falsifiers


| H | Hypothesis | What falsifies it | Experiment | Linked evidence rows |
|---|---|---|---|---|
| H8.1 | Independent data/metadata optimization causes measurable regret in at least one regime. | The data×metadata interaction term is negligible across all tested regimes. | RQ8-E2 | 23 |
| H8.2 | Sparse route granularity and KV page layout have a joint optimum. | Best route/index representation is independent of KV layout/page choice. | RQ8-E3 | 10 |
| H8.3 | Joint scale/data-tile selection reduces preparation/interleave traffic. | Joint selection offers no advantage over separately optimal layouts. | RQ8-E1 | 13 |
| H8.4 | When metadata is tiny/cache-resident, co-layout benefit disappears. | A large interaction persists even when metadata is forced cache-resident and tiny. | RQ8-E0 | 0 |
| H8.5 | Metadata placement can change the preferred data page/tile size. | Data-layout winner is invariant to controlled metadata placement changes. | RQ8-E4 | 10 |


## 12.6 Closure judgment

This RQ now satisfies the requested chain:

```text
source rule
  → explicit proxy
  → explicit omitted variables
  → framework-native fallback / repair / alternative
  → modern-model trigger envelope
  → falsifiable hypothesis + falsifier
```

The evidence does **not** assert that the hypothesis is true. The remaining unknown is empirical and is already assigned to the v11 experiment protocol.



# 13. L-RQ9 — Layout-space expressiveness and search-space truncation

## 13.1 Trial status

**Verdict:** `COMPLETE_STRONG`

- Complete rule-chain rows: **52**
- Direct mechanism rows: **34**
- Direct stacks: **CUTLASS/CuTe, FlashInfer, IREE, MLIR, SGLang, TVM Relax, TensorRT-LLM, Triton, vLLM**
- Current-source supplement rows: **5**
- Registered hypotheses: **H9.1, H9.2, H9.3, H9.4, H9.5**

## 13.2 Framework coverage judgment


| Framework/layer | Why it is or is not part of this RQ |
|---|---|
| vLLM | DIRECT: generalized layout family and support-set intersection. |
| SGLang | DIRECT: vectorized-5D and page-major/hybrid representation. |
| TensorRT-LLM | DIRECT: narrow specialized dense/MLA/sparse families. |
| FlashInfer | DIRECT: named layout plus stride-capable/sparse mappings. |
| Triton/CUTLASS/MLIR/IREE/TVM | DIRECT: richer physical layout/encoding/index-map spaces and legality mechanisms. |


## 13.3 Complete source-rule → proxy → ignored → fallback → trigger → hypothesis chain


| ID | Stack | Role | Source rule/fact | Proxy | Ignored variables | Fallback/repair | Trigger condition | Hypothesis link | Source status |
|---|---|---|---|---|---|---|---|---|---|
| HC02 | CUTLASS/CuTe | DIRECT_MECHANISM | The helper calls its policy a simple heuristic and selects SMEM layout/swizzle atoms from major mode, element type and major-dimension size; related helpers choose legal/vectorized store at… | output layout; dtype; tile; ownership | producer/consumer graph; concurrent pressure | Legal fallback store/layout. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H9.1,H9.3 | FROZEN_LEDGER |
| HC03 | CUTLASS/CuTe | DIRECT_MECHANISM | CUTLASS 3.x explicitly decomposes GEMM into device, kernel, collective, tiled MMA/copy and atom levels; kernels are assembled from a collective mainloop and collective epilogue, with persis… | template types; dtype/layout/tile | global graph equivalence and runtime distribution | Another template/operator. | The exact source predicate is in source_rule_fact; model-family envelope: Dense MLP/linear | H9.1 | FROZEN_LEDGER |
| HC04 | CUTLASS/CuTe | DIRECT_MECHANISM | CUTLASS documents an efficient row-major epilogue and, for column-major output, transposes/swaps the equivalent GEMM operands so the efficient epilogue mapping is preserved. | input/output layout | downstream consumer layout/fusion | Specialized template path. | The exact source predicate is in source_rule_fact; model-family envelope: Dense MLP/linear | H9.1 | FROZEN_LEDGER |
| HC05 | CUTLASS/CuTe | DIRECT_MECHANISM | CuTe DSL helpers fail fast on unsupported instruction/architecture combinations and enforce TMA contiguous-dimension alignment before performance choices are made. | swizzle; instruction; architecture | higher-level graph opportunity cost | Pick legal atom/template. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H9.1,H9.3,H9.5 | FROZEN_LEDGER |
| F-A01 | FlashInfer | DIRECT_MECHANISM | Layout is an explicit caller-visible NHD/HND contract. | Layout argument, tensor shape/strides. | Global serving objective unless caller models it. | Invalid layout fails or wrapper-specific repair. | The exact source predicate is in source_rule_fact; model-family envelope: Paged KV, Generic attention | H9.1 | FROZEN_LEDGER |
| F-A06 | FlashInfer | DIRECT_MECHANISM | Some kernels can consume NHD-contiguous or HND-transposed/arbitrary supported strides without materialized conversion. | Strides and innermost head-dim condition. | Cache/TLB efficiency of each stride pattern. | Another kernel or reject unsupported stride. | The exact source predicate is in source_rule_fact; model-family envelope: GQA, Paged KV | H9.1 | FROZEN_LEDGER |
| F-A07 | FlashInfer | DIRECT_MECHANISM | Fast paths have conjunctions over dtype, head dimension, GQA ratio, page size, q_len and SM. | Geometry/dtype/page/q_len/SM. | Continuous system state inside the region. | Alternate supported kernel. | The exact source predicate is in source_rule_fact; model-family envelope: MHA, GQA, Speculative, Paged KV | H9.1 | FROZEN_LEDGER |
| F-L1 | FlashInfer | DIRECT_MECHANISM | Paged KV APIs consume caller-selected NHD/HND layouts. | kv_layout and strides/shapes | cross-consumer/global serving objective | invalid layout fails or wrapper-specific repair | The exact source predicate is in source_rule_fact; model-family envelope: MHA, MQA, GQA, Paged KV | H9.1 | FROZEN_LEDGER |
| HF03 | FlashInfer | DIRECT_MECHANISM | Grouped-MoE GEMM APIs expose explicit tensor-shape/dtype contracts, backend dispatch and tactic selection; quantized variants add scale/format requirements. | dtype; architecture; scale/layout; tactic | whole graph and network representation | Backend/tactic or unsupported error. | The exact source predicate is in source_rule_fact; model-family envelope: MoE, Quantized KV/activation | H9.1,H9.3 | FROZEN_LEDGER |
| HI01 | IREE | DIRECT_MECHANISM | IREE exposes per-dispatch tuning knobs such as MMA layout, subgroup/workgroup parameters and reduction tiling; defaults target generic performance while tuning specializes them. | dispatch/target/knobs | persistent KV and cross-dispatch serving interactions | Tuning spec overrides. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H9.1,H9.3,H9.4 | FROZEN_LEDGER |
| HI02 | IREE | DIRECT_MECHANISM | IREE emits compiler pipeline constraints for legal knob combinations, serializes them to SMT-LIB in the tuner, and uses an SMT solver to enumerate valid assignments; an experimental verifie… | compiler constraints | cross-dispatch cost and online serving state | SMT excludes illegal assignments; verifier checks. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H9.1 | FROZEN_LEDGER |
| HI04 | IREE | DIRECT_MECHANISM | The Encoding dialect attaches abstract data-layout encodings to tensors; downstream encoding/materialization infrastructure resolves these representations for targets. | op/operand/indexing maps/iteration/target | LLM persistent multi-consumer/online traffic | Unset/materialize encoding. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H9.1,H9.3,H9.5 | FROZEN_LEDGER |
| HI05 | IREE | DIRECT_MECHANISM | IREE set_encoding/unset_encoding accept dynamic encoding_dims (e.g. M/N/K); the operation description explicitly states these values are used for runtime layout selection based on problem s… | runtime dimensions | prefix reuse; queue; network/MoE contention | Resolver/default behavior. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H9.1,H9.4,H9.5 | FROZEN_LEDGER |
| HI06 | IREE | DIRECT_MECHANISM | IREE's GPU shared-memory reuse pass computes allocation liveness ranges, groups overlapping lifetimes, and reuses allocations only under explicit structural preconditions; a separate GPU pa… | resource limits; liveness; layout ops | whole-serving persistent state | Alter/reject/combine/reuse. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H9.1 | FROZEN_LEDGER |
| HM03 | MLIR | DIRECT_MECHANISM | structured.pack_transpose can transpose a pack/compute(/unpack) chain; for pack it expects the consuming linalg.generic to be the sole consumer, and unsupported topology/specification yield… | pack chain; permutations; consumer topology | multi-consumer/persistent state/global cost | Silenceable failure. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H9.1 | FROZEN_LEDGER |
| S-A06 | SGLang | DIRECT_MECHANISM | Target verify/draft routing is conditioned on speculative mode; some backend/page/top-k combinations are forbidden. | Spec mode, top-k, page size, backend. | Acceptance distribution, draft depth, target/draft asymmetry. | Switch supported mode/backend or reject. | The exact source predicate is in source_rule_fact; model-family envelope: Speculative | H9.1,H9.3 | FROZEN_LEDGER |
| S-A09 | SGLang | DIRECT_MECHANISM | `vectorized_5d` cache layout routes writes/decode through different compatible paths because a 4D view is not representable. | Pool layout tag, backend/platform. | Cost of keeping 5D vs converting; workload crossover. | 5D-compatible writer/decode path. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H9.1 | FROZEN_LEDGER |
| S-A10 | SGLang | DIRECT_MECHANISM | Modern hybrid recurrent/linear-attention architectures use a separate linear-attention backend; some FlashInfer prefill paths require a conjunction over GPU/CUDA/state dtype/head dims/cache… | Architecture, SM, CUDA, recurrent-state dtype, head dimensions, cache/chunk mode. | Live batch, state reuse, MoE/system contention. | Base linear-attention backend. | The exact source predicate is in source_rule_fact; model-family envelope: Hybrid/recurrent, Generic attention | H9.1,H9.3 | FROZEN_LEDGER |
| HX05 | TVM Relax | DIRECT_MECHANISM | Relax layout_transform is an explicit graph operator taking an IndexMap and optional padding and materializing a transformed tensor. | index map; tensor shape | producer/consumer could avoid conversion jointly | Keep/lower/fuse transform. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H9.1,H9.5 | FROZEN_LEDGER |
| HT05 | TensorRT-LLM | DIRECT_MECHANISM | AutoDeploy contains explicit fused-MoE restructuring logic that converts between stacked/backend-specific weight/graph representations and enforces backend/style support constraints. | MoE graph/weight representation style, backend family, fused-MoE support predicates | Runtime expert-token distribution, restructuring/copy cost, peak temporary memory, alternative fused-MoE layout families. | Apply only supported fused-MoE restructuring/backend style; otherwise retain/choose a supported general or alternate path rather than … | The exact source predicate is in source_rule_fact; model-family envelope: MoE | H9.1,H9.4 | FROZEN_LEDGER |
| T-A07 | TensorRT-LLM | DIRECT_MECHANISM | Specialized sparse kernels have hard HND/page-size/geometry constraints; unsupported classes can fall back to another sparse path or error. | Sparse algorithm, page/layout, dtype/head geometry. | Conversion cost from general layout. | Alternative sparse implementation or hard error. | The exact source predicate is in source_rule_fact; model-family envelope: MHA, MQA, GQA, MLA, Sparse/DSA | H9.1,H9.4 | FROZEN_LEDGER |
| V-A03 | vLLM | DIRECT_MECHANISM | MLA has separate SM10, SM12 and other-GPU ordered backend sets. | `use_mla`, SM generation, head size, KV dtype, number of heads. | Live batch, q_len, KV_len, actual sparsity, transfer/MoE contention. | Next compatible MLA backend. | The exact source predicate is in source_rule_fact; model-family envelope: MLA | H9.1 | FROZEN_LEDGER |
| V-A06 | vLLM | DIRECT_MECHANISM | FA version is architecture-conditioned; specialized FA4 paths have block-size/head-size/feature conditions and can transparently fall back. | SM, head size, block size, unsupported feature flags. | Current sequence length, batch, graph state, prefix reuse. | Older compatible FA implementation or make the higher-priority backend ineligible. | The exact source predicate is in source_rule_fact; model-family envelope: MHA, Paged KV | H9.1,H9.3 | FROZEN_LEDGER |
| V-A08 | vLLM | DIRECT_MECHANISM | Backend validation includes sinks, sparse, multimodal prefix, per-head scales, non-causal, batch-invariance, connector, PCP/DCP, adaptive verification, RSWA and other flags. | Feature booleans plus dtype/head/block geometry. | Continuous runtime workload and end-to-end contention. | Invalidate candidate and move down priority. | The exact source predicate is in source_rule_fact; model-family envelope: Sparse/DSA, KV transfer/disagg, Quantized KV/activation | H9.1 | FROZEN_LEDGER |
| V-A12 | vLLM | DIRECT_MECHANISM | Target and draft can independently select backend constraints whose supported-layout intersection is empty, causing initialization failure. | Target layout set, draft layout set. | Joint target/draft cost, second-best backend pair, conversion, separate pools. | Current observed behavior can be hard failure. | The exact source predicate is in source_rule_fact; model-family envelope: Speculative | H9.1,H9.2,H9.3 | FROZEN_LEDGER |
| V-B3 | vLLM | DIRECT_MECHANISM | Attention/KV semantic kind can override the global backend. | MLA, sliding-window, attention type, config map | per-layer execution weight, bytes, live shape and phase cost | missing kind uses global backend | The exact source predicate is in source_rule_fact; model-family envelope: Generic attention | H9.1 | FROZEN_LEDGER |
| V-L1 | vLLM | DIRECT_MECHANISM | Each backend declares supported/preferred KV layouts. | backend support/preference list | runtime frequency, bytes, cache locality, conversion cost | default layout preference if none declares restrictions | The exact source predicate is in source_rule_fact; model-family envelope: Paged KV, Generic attention | H9.1,H9.4 | FROZEN_LEDGER |
| V-L2 | vLLM | DIRECT_MECHANISM | Candidate layouts are the intersection across all consuming backends. | boolean membership in supported-layout sets | per-consumer benefit, conversion amortization, split-pool cost | empty intersection is hard error | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H9.1,H9.2 | FROZEN_LEDGER |
| V-L5 | vLLM | DIRECT_MECHANISM | Mixed H/N/C shapes narrow candidates to block-compact layouts. | static cache-spec shape heterogeneity | group execution weight, separate-pool alternative | no candidate -> failure | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H9.1,H9.4 | FROZEN_LEDGER |
| HC01 | CUTLASS/CuTe | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | CUTLASS GEMM heuristics use NVIDIA's analytical matmul heuristic to rank a subset of valid kernels from problem size and hardware SKU, reducing runtime autotuning search; exhaustive coverag… | shape; SKU; registered kernels | cross-op layout; online serving state | Profile/rank subset. | The exact source predicate is in source_rule_fact; model-family envelope: Dense MLP/linear | H9.1,H9.3 | FROZEN_LEDGER |
| F-A03 | FlashInfer | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Architecture/kernel availability can auto-select TRTLLM-gen or XQA. | SM generation, kernel availability. | Batch/KV-length crossover if both feasible. | Compatible available backend. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H9.1,H9.3 | FROZEN_LEDGER |
| HS04 | SGLang | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Execution arguments expose multiple FP8/FP4 GEMM backends and an auto mode whose documented choice depends on hardware/backend availability. | SM; dtype; availability | live M/N/K; downstream layout/quant format | Compatible fallback, often Triton; explicit invalid choices error. | The exact source predicate is in source_rule_fact; model-family envelope: Quantized KV/activation, Dense MLP/linear | H9.1,H9.3 | FROZEN_LEDGER |
| S-B1 | SGLang | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Hopper standard attention prefers FA3 when supported. | GPU generation, CUDA/model support | live batch/sequence/GQA/reuse | alternative compatible backend | The exact source predicate is in source_rule_fact; model-family envelope: MHA, MQA, GQA | H9.1,H9.4 | FROZEN_LEDGER |
| S-B2 | SGLang | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Blackwell MHA prefers TRTLLM MHA except speculative top-k constraints alter the path. | Blackwell, speculative top-k | acceptance rate, verify length, batch/KV length | compatible alternative or reject unsupported combination | The exact source predicate is in source_rule_fact; model-family envelope: MHA, Speculative | H9.1,H9.4 | FROZEN_LEDGER |
| S-B3 | SGLang | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Other CUDA platforms prefer FlashInfer if available, else Triton. | platform class, package availability | live workload and reuse | Triton | The exact source predicate is in source_rule_fact; model-family envelope: MHA, Paged KV | H9.1 | FROZEN_LEDGER |
| S-B4 | SGLang | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | MLA Hopper path prefers FA3. | GPU generation, MLA | live shape, latent geometry, reuse | platform-compatible alternative | The exact source predicate is in source_rule_fact; model-family envelope: MLA | H9.1,H9.4 | FROZEN_LEDGER |
| S-B5 | SGLang | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | MLA Blackwell prefers FlashInfer, with model-specific TRTLLM MLA auto selection in documented cases. | GPU generation, model identity | deployment request mix | general Blackwell MLA fallback | The exact source predicate is in source_rule_fact; model-family envelope: MLA | H9.1 | FROZEN_LEDGER |
| S-B6 | SGLang | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Other architectures can fall back to Triton MLA. | platform class | live workload | Triton | The exact source predicate is in source_rule_fact; model-family envelope: MLA | H9.1,H9.3 | FROZEN_LEDGER |
| HX04 | TVM TIR | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | MetaSchedule documentation states that it searches schedule design spaces and measures on real hardware; the snapshot exposes cost-model choices (including XGBoost/MLP/random), runners and … | TIR features; candidates; measured time | whole-serving temporal/persistent state | Best database record; DLight as zero-search alternative. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H9.1,H9.3,H9.4 | FROZEN_LEDGER |
| T-A01 | TensorRT-LLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | TRT-LLM is the production/default backend family; Vanilla and FlashInfer are available alternatives/configurations. | Configured backend and feature support. | Live workload at top-level family choice. | Supported configured path or documented fallback/error. | The exact source predicate is in source_rule_fact; model-family envelope: Generic attention | H9.1,H9.4 | FROZEN_LEDGER |
| T-B1 | TensorRT-LLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | TRT-LLM attention is the production/default backend family unless configured otherwise. | backend configuration and feature support | live workload at family/layout commitment time | configured supported path / explicit fallback | The exact source predicate is in source_rule_fact; model-family envelope: MHA, GQA, MLA | H9.1,H9.4 | FROZEN_LEDGER |
| HR01 | Triton | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | The tutorial declares finite lists of BLOCK_SIZE_M/N/K, GROUP_SIZE_M, stage and warp configurations and autotunes them using M/N/K as the key. | M,N,K key; config list | unlisted tiles/swizzles; graph state | Best measured declared config. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H9.1,H9.3 | FROZEN_LEDGER |
| HR02 | Triton | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Grouped GEMM launches a fixed number of CTAs with static device-side scheduling and autotunes a finite set of BLOCK sizes and NUM_SM values keyed by group_size. | group size; tile configs; NUM_SM | expert skew drift; A2A payload representation | Autotune among configs. | The exact source predicate is in source_rule_fact; model-family envelope: MoE, Paged KV | H9.1 | FROZEN_LEDGER |
| HR03 | Triton | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | The tutorial explains that tile traversal order changes L2 reuse; grouped ordering reduces loaded blocks and can materially outperform simple row-major ordering on the cited hardware. | tile grid; GROUP_SIZE_M; NUM_SMS | producer/consumer layout and communication overlap | Alternative config/kernel. | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H9.1,H9.3,H9.4 | FROZEN_LEDGER |
| V-A02 | vLLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Standard-attention priority differs by GPU generation: Blackwell begins with FlashInfer; Ampere/Hopper begins with FlashAttention. | Compute capability, non-causal flag, multimodal-prefix state. | Batch/KV-length crossover, GQA ratio, prefix hit rate, transfer pressure. | Next compatible candidate. | The exact source predicate is in source_rule_fact; model-family envelope: MHA, MQA, GQA | H9.1 | FROZEN_LEDGER |
| V-B1 | vLLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Explicit backend is validated then used. | backend name, dtype/KV dtype, head/block size, feature/capability predicates | live shape distribution, prefix reuse, conversion/transfer/system contention | hard error when explicit backend is incompatible | The exact source predicate is in source_rule_fact; model-family envelope: Generic attention | H9.1 | FROZEN_LEDGER |
| V-B2 | vLLM | ADJACENT_BASELINE_OR_COUNTEREVIDENCE | Automatic selection takes the first compatible backend in a platform priority list. | SM generation, attention family, dtype/KV dtype, head/block size, feature flags | live batch/q_len/KV_len, GQA ratio as cost term, reuse, contention, conversion | try next backend; error if none valid | The exact source predicate is in source_rule_fact; model-family envelope: MHA, MQA, GQA, MLA | H9.1 | FROZEN_LEDGER |
| S12-05 | SGLang | DIRECT_MECHANISM | SGLang exposes a page-major KV layout that places Mamba state and full/SWA KV caches in one page-granularity envelope (page outermost, layer-major within page) instead of the default per-la… | explicit enable flag plus backend compatibility | per-state access frequency, model/workload-specific profitability, alternative hybrid-state domain partitions | default per-layer layout remains available when page-major mode is disabled/incompatible | Hybrid Mamba + full attention/SWA models using Triton attention/linear-attention/Mamba backends | H9.1,H9.4 | CURRENT_SOURCE_SUPPLEMENT |
| S12-09 | FlashInfer | DIRECT_MECHANISM | Sparse CuTe paths expose concrete layout incompatibilities/repairs: SM100 allocates a fresh contiguous BHSD output even when the caller has BSHD; SM120 explicitly remaps BSHD inputs into th… | architecture, sparse/Sage mode, BSHD/BHSD layout, tensor-map/instruction constraints | whole-graph consumer requirements, whether a different producer layout could remove the repair, search over alternative sparse physical layouts | fresh contiguous output or architecture-specific view/remap | Blackwell sparse/Sage attention with BSHD/BHSD or architecture-specific physical mappings | H9.1,H9.3,H9.4,H9.5 | CURRENT_SOURCE_SUPPLEMENT |
| S12-10 | vLLM | DIRECT_MECHANISM | Current indexed vLLM source keeps a generalized default KV-layout preference beginning with LBNHC, LBHNC, BLNHC and resolves supported layouts through backend-declared capability sets rathe… | backend-supported layout sets and fixed preference order | live execution weights, measured performance gaps, layouts not implemented/exposed by any selected backend | intersect available sets and choose from the supported preference order | vLLM paged attention layout resolution across backends; generalized LBNHC/LBHNC/BLNHC/... families | H9.1,H9.4,H9.5 | CURRENT_SOURCE_SUPPLEMENT |
| S13-05 | CUTLASS/CuTe | DIRECT_MECHANISM | CUTLASS Sm1xxBlockScaledConfig explicitly constructs SFA/SFB scale-factor tensor layouts for block-scaled Blackwell GEMM; scale-factor layouts are a first-class physical representation rath… | scale-vector size, operand/tile geometry, architecture-specific block-scaled MMA contract | upstream producer scale layout, cost of interleaving/repacking scales, downstream reuse, graph-level choice among alternative scale layouts | construct/use the required scale layout or choose a different legal GEMM/precision path | Blackwell NVFP4/MXFP block-scaled dense or sparse GEMM, including MoE/grouped-GEMM style consumers | H9.1,H9.3,H9.4 | CURRENT_SOURCE_SUPPLEMENT |
| S13-06 | Triton | DIRECT_MECHANISM | Triton kernels inspect the storage layout of MX scale tensors and contain architecture-specific scale-layout/unswizzle implementations for Hopper/CDNA4/GFX1250; block-scaled matmul treats s… | scale tensor storage layout, precision config, microblock size, target architecture | upstream producer cost, serving-level reuse/fanout, alternative compound layouts, conversion amortization | canonicalize/convert scale layout or use another supported precision/kernel path | MXFP4/NVFP4/block-scaled GEMM on NVIDIA/AMD targets; dense MLP and MoE/grouped GEMM consumers | H9.1,H9.3,H9.4 | CURRENT_SOURCE_SUPPLEMENT |


## 13.4 Modern model trigger audit

| Family | Concrete trigger for this RQ |
|---|---|
| MHA/MQA/GQA | Generalized paged-KV axis orders, stride-capable variants and block/page organizations. |
| MLA | Latent/sparse/vectorized representations enlarge the native family. |
| Sparse/DSA | Architecture-specific BHSD/BSHD/paged layouts test richer search-space value. |
| MoE | Grouped-GEMM operand/scale layouts and block-scaled representations. |
| Compiler | CuTe/MLIR/IREE/TVM/Triton provide the extended legal representation space. |

## 13.5 Falsifiable hypotheses and direct falsifiers


| H | Hypothesis | What falsifies it | Experiment | Linked evidence rows |
|---|---|---|---|---|
| H9.1 | At least one realizable layout outside the native set is materially faster. | Extended search never beats the native framework oracle beyond noise. | RQ9-E2 | 52 |
| H9.2 | Backend-support intersection causes measurable expressiveness regret with heterogeneous consumers. | Intersection-restricted and extended oracles are statistically equivalent. | RQ9-E3 | 2 |
| H9.3 | Benefits of richer layout space are concentrated in specific geometries/phases. | Extended layouts win uniformly everywhere or nowhere. | RQ9-E4 | 19 |
| H9.4 | If the native set contains the oracle winner, extra layout complexity should be disabled. | Extended search changes decisions despite zero oracle regret and only adds overhead. | RQ9-E0 | 18 |
| H9.5 | Legality-first richer layout algebra can expose useful candidates not surfaced by named APIs. | All legal extended candidates collapse to already-exposed physical mappings. | RQ9-E1 | 6 |


## 13.6 Closure judgment

This RQ now satisfies the requested chain:

```text
source rule
  → explicit proxy
  → explicit omitted variables
  → framework-native fallback / repair / alternative
  → modern-model trigger envelope
  → falsifiable hypothesis + falsifier
```

The evidence does **not** assert that the hypothesis is true. The remaining unknown is empirical and is already assigned to the v11 experiment protocol.



# 14. L-RQ10 — Paged/block layout granularity and allocator coupling

## 14.1 Trial status

**Verdict:** `COMPLETE_STRONG`

- Complete rule-chain rows: **16**
- Direct mechanism rows: **16**
- Direct stacks: **FlashInfer, SGLang, TensorRT-LLM, vLLM**
- Current-source supplement rows: **7**
- Registered hypotheses: **H10.1, H10.2, H10.3, H10.4, H10.5**

## 14.2 Framework coverage judgment


| Framework/layer | Why it is or is not part of this RQ |
|---|---|
| vLLM | DIRECT: block-compact/HiSparse/offload tokens_per_block and blocks_per_chunk. |
| SGLang | DIRECT: explicit page-size metadata/I/O/cache-hit trade-off; page-interleave sharding. |
| TensorRT-LLM | DIRECT: tokens_per_block, QSA paged access, Helix block partition. |
| FlashInfer | DIRECT: page-geometry fast paths and graph block-table buckets. |
| Compiler/kernel stacks | LOCAL TILE BASELINE only; local tile size is not a substitute for serving allocator/page granularity. |


## 14.3 Complete source-rule → proxy → ignored → fallback → trigger → hypothesis chain


| ID | Stack | Role | Source rule/fact | Proxy | Ignored variables | Fallback/repair | Trigger condition | Hypothesis link | Source status |
|---|---|---|---|---|---|---|---|---|---|
| F-A07 | FlashInfer | DIRECT_MECHANISM | Fast paths have conjunctions over dtype, head dimension, GQA ratio, page size, q_len and SM. | Geometry/dtype/page/q_len/SM. | Continuous system state inside the region. | Alternate supported kernel. | The exact source predicate is in source_rule_fact; model-family envelope: MHA, GQA, Speculative, Paged KV | H10.1,H10.5 | FROZEN_LEDGER |
| HS01 | SGLang | DIRECT_MECHANISM | The attention-backend guide states that prefix reuse occurs only for complete pages, page_size=1 maximizes reuse, larger pages generally improve attention-kernel performance, and backends h… | page size; backend | actual prefix distribution; fragmentation; transfer | Emulation for some backends or supported native size. | The exact source predicate is in source_rule_fact; model-family envelope: Generic attention | H10.1,H10.2,H10.3,H10.5 | FROZEN_LEDGER |
| S-A05 | SGLang | DIRECT_MECHANISM | `page_size=1` maximizes prefix reuse because only full pages are reusable, while larger pages generally improve attention-kernel performance. | Page size, prefix-cache semantics. | Real prefix distribution, fragmentation, transfer granularity, graph state. | User/configuration choice; no universal joint optimizer. | The exact source predicate is in source_rule_fact; model-family envelope: Paged KV, Generic attention | H10.1,H10.2,H10.3,H10.5 | FROZEN_LEDGER |
| S-A06 | SGLang | DIRECT_MECHANISM | Target verify/draft routing is conditioned on speculative mode; some backend/page/top-k combinations are forbidden. | Spec mode, top-k, page size, backend. | Acceptance distribution, draft depth, target/draft asymmetry. | Switch supported mode/backend or reject. | The exact source predicate is in source_rule_fact; model-family envelope: Speculative | H10.1 | FROZEN_LEDGER |
| T-A02 | TensorRT-LLM | DIRECT_MECHANISM | Layers with incompatible KV geometry/window characteristics use separate pools rather than one homogeneous pool. | KV-head count, attention-window/cache geometry. | Per-layer runtime weight, phase preference, remote consumer. | Create another pool. | The exact source predicate is in source_rule_fact; model-family envelope: Paged KV | H10.1,H10.4,H10.5 | FROZEN_LEDGER |
| T-A07 | TensorRT-LLM | DIRECT_MECHANISM | Specialized sparse kernels have hard HND/page-size/geometry constraints; unsupported classes can fall back to another sparse path or error. | Sparse algorithm, page/layout, dtype/head geometry. | Conversion cost from general layout. | Alternative sparse implementation or hard error. | The exact source predicate is in source_rule_fact; model-family envelope: MHA, MQA, GQA, MLA, Sparse/DSA | H10.1,H10.5 | FROZEN_LEDGER |
| T-A13 | TensorRT-LLM | DIRECT_MECHANISM | tokens_per_block is model/executor/KV-manager configuration; manager construction receives fixed tokens_per_block and geometry. Legacy documentation also describes block-size selection at b… | build/launch config, attention/KV geometry | runtime prefix-reuse distribution, workload phase drift, current fragmentation | rebuild/recreate executor/manager/configure another run | The exact source predicate is in source_rule_fact; model-family envelope: MLA, Paged KV | H10.1,H10.2,H10.3,H10.5 | FROZEN_LEDGER |
| V-A06 | vLLM | DIRECT_MECHANISM | FA version is architecture-conditioned; specialized FA4 paths have block-size/head-size/feature conditions and can transparently fall back. | SM, head size, block size, unsupported feature flags. | Current sequence length, batch, graph state, prefix reuse. | Older compatible FA implementation or make the higher-priority backend ineligible. | The exact source predicate is in source_rule_fact; model-family envelope: MHA, Paged KV | H10.1,H10.3 | FROZEN_LEDGER |
| V-L5 | vLLM | DIRECT_MECHANISM | Mixed H/N/C shapes narrow candidates to block-compact layouts. | static cache-spec shape heterogeneity | group execution weight, separate-pool alternative | no candidate -> failure | The exact source predicate is in source_rule_fact; model-family envelope: Generic layout/compiler | H10.1,H10.3,H10.4 | FROZEN_LEDGER |
| S12-01 | vLLM | DIRECT_MECHANISM | OffloadingConfig explicitly carries tokens_per_block, blocks_per_chunk, resolved worker KV layout, canonical_layout, replicated_layout, and whether persisted bytes are parallelism-agnostic;… | configured block/chunk granularity, parallel topology, pure-MLA/TP-only replication condition, canonical-layout request | live network bandwidth, cache-hit/reuse distribution, conversion cost, host-cache locality, future topology changes | representation mode is configured/certified rather than jointly cost-optimized; unsupported portability must use a compatible represen… | KV offload/disaggregation for MHA/GQA/MLA; multi-rank TP/PP/PCP/DCP; canonical/direct/replicated host representation | H10.1,H10.5 | CURRENT_SOURCE_SUPPLEMENT |
| S12-02 | vLLM | DIRECT_MECHANISM | HiSparse partitions sparse-MLA source and indexer specs, builds source/indexer/hot/resident groups, requires one resolved GPU block size, requires one page size within each hot-cache group,… | cache role, block size, page_size_bytes, indexer-page budget, host budget, block-outermost layout capability | live sparsity distribution, per-layer access frequency, host/device bandwidth variation, dynamic hot-set size, alternative grouping costs | raises on incompatible block/page/layout conditions; forms compatible groups and computes host/device layouts | HiSparse sparse-MLA with source+indexer caches, hot/resident device caches and host-resident source state | H10.1,H10.4 | CURRENT_SOURCE_SUPPLEMENT |
| S12-03 | SGLang | DIRECT_MECHANISM | Page-interleave KV sharding stripes logical pages across ranks, all-gathers prefix pages one layer ahead into a [prefix\|chunk\|trash] scratch layout, stages the current chunk locally, and ma… | shard_size/rank, page_size, aligned prefix/chunk lengths, page ownership, fixed scratch capacity | runtime interconnect contention, overlap variation, scratch-memory pressure, alternative page placement/layouts, heterogeneous link topology | unsupported direct prefix-valid writes raise; data are assembled into scratch through the dedicated gather path | Sharded prefill for GQA or MLA; logical-page KV sharding; prefix all-gather and per-layer scratch | H10.1,H10.3,H10.5 | CURRENT_SOURCE_SUPPLEMENT |
| S12-04 | SGLang | DIRECT_MECHANISM | HiCache documents page size as KV storage/retrieval granularity: larger pages reduce metadata overhead and improve storage I/O efficiency but can reduce partial-page cache-hit rate; long co… | configured page size and coarse workload prefix structure | exact request distribution, GPU-kernel locality, fragmentation, current bandwidth/contention, dynamic page-size switching cost | deployment selects a fixed page size; no source-level dynamic optimizer is claimed | HiCache host/storage KV caching; long-common-prefix versus diverse-prefix workloads | H10.1,H10.2,H10.3,H10.5 | CURRENT_SOURCE_SUPPLEMENT |
| S12-06 | TensorRT-LLM | DIRECT_MECHANISM | QSA sparse GQA stores/reads paged KV through block_table and tokens_per_block, consumes selected sparse tokens, requires query-head divisibility by local KV heads, and chooses a fused CUDA … | tokens_per_block, block table, sparse selected-token metadata, request indices, head dimension, Q/KV head divisibility | alternative KV physical layouts, joint route/page-layout profitability, metadata cache locality, dynamic page-granularity selection | reference sparse GQA path when fused CUDA predicate is false; validation errors for incompatible shapes/metadata | QSA sparse GQA/MHA/MQA-compatible head topology with block tables, selected sparse tokens and paged KV | H10.1,H10.2,H10.5 | CURRENT_SOURCE_SUPPLEMENT |
| S13-02 | FlashInfer | DIRECT_MECHANISM | cuDNN paged decode buckets max_seq_len_kv and block-table width because these values are baked into the built graph; bucketing lets consecutive decode steps replay a cached graph instead of… | bucketed maximum KV length and block-table width | future sequence-length distribution, memory overhead of larger buckets, alternative page sizes/layouts, cross-request graph reuse lifetime | build/rebuild a graph for another bucket or use a non-captured/alternate path | CUDA-graph paged decode; changing KV length/page-table width; GQA/MHA decode families supported by the selected backend | H10.1 | CURRENT_SOURCE_SUPPLEMENT |
| S13-04 | TensorRT-LLM | DIRECT_MECHANISM | Helix context partitioning assigns token blocks round-robin across CP ranks using tokens_per_block and changes both prefill→decode KV movement and how KV grows during decode when generation… | tokens_per_block, CP rank count, generation CP mode | live interconnect contention, unequal request lengths, alternative block/page sizes, replication alternatives, per-rank memory skew | use the existing non-Helix/non-CP KV-cache infrastructure when the Helix mode is not enabled | multi-million-token long-context decode, generation CP > 1, distributed paged KV transfer/growth | H10.1,H10.5 | CURRENT_SOURCE_SUPPLEMENT |


## 14.4 Modern model trigger audit

| Family | Concrete trigger for this RQ |
|---|---|
| MHA/MQA/GQA | Paged KV page/block size controls kernel locality, table size and reuse granularity. |
| MLA | HiCache/HiSparse/latent pages and long-context storage. |
| Sparse/DSA | QSA/HiSparse page/block granularity couples sparse metadata and KV access. |
| MoE | Not a primary page-cache problem; exclude from core experiment unless a paged expert cache is studied. |
| Distributed | Offload blocks_per_chunk, page sharding and Helix tokens_per_block add transfer consequences. |

## 14.5 Falsifiable hypotheses and direct falsifiers


| H | Hypothesis | What falsifies it | Experiment | Linked evidence rows |
|---|---|---|---|---|
| H10.1 | Optimal page/block size shifts with prefix-hit probability/fanout. | Page-size winner is invariant to controlled reuse distribution. | RQ10-E2 | 16 |
| H10.2 | Very small pages are metadata-bound while larger pages become fragmentation/prefix-waste bound. | Profiler/allocator terms do not change direction with page size. | RQ10-E1 | 5 |
| H10.3 | One fixed page size has measurable regret on mixed traffic. | A single page size matches per-trace oracle on heterogeneous traces. | RQ10-E3 | 7 |
| H10.4 | Multiple page granularities win only after benefits exceed pool/registration overhead. | Multi-granularity always wins even after all pool/allocator costs are charged. | RQ10-E4 | 3 |
| H10.5 | Kernel-optimal and allocator-optimal granularity can differ. | The page size minimizing isolated kernel latency always minimizes end-to-end/cache-system cost. | RQ10-E1 | 11 |


## 14.6 Closure judgment

This RQ now satisfies the requested chain:

```text
source rule
  → explicit proxy
  → explicit omitted variables
  → framework-native fallback / repair / alternative
  → modern-model trigger envelope
  → falsifiable hypothesis + falsifier
```

The evidence does **not** assert that the hypothesis is true. The remaining unknown is empirical and is already assigned to the v11 experiment protocol.



# 15. Mechanical QA

Unified rule-chain rows: **258**

| Required field | Non-empty rows | Coverage |
|---|---|---|
| source_rule_fact | 258 | 100.0% |
| proxy | 258 | 100.0% |
| assumption | 258 | 100.0% |
| ignored_variables | 258 | 100.0% |
| fallback_repair | 258 | 100.0% |
| modern_trigger_family | 258 | 100.0% |
| trigger_condition | 258 | 100.0% |
| hypothesis_links | 258 | 100.0% |
| provenance | 258 | 100.0% |

Positive hypotheses requiring manual review because no rule-chain row linked to them: **0**.

If a hypothesis is a deliberate negative control, zero positive source links are acceptable; otherwise a zero link would block closure.

# 16. What changed versus v9/v12

The earlier documents were already deep, but three weaknesses remained:

1. **Current supplements were not fully integrated into the per-rule chain.**  
   They had fact/proxy/ignored/fallback but not always explicit assumption, trigger and hypothesis linkage. This is now fixed.

2. **“All related frameworks” was implicit.**  
   The new framework-coverage judgment explicitly distinguishes `DIRECT`, `BASELINE/COUNTEREVIDENCE`, and `N/A because the stack does not own this decision`.

3. **Modern-model triggers were mostly tags.**  
   Each RQ now has explicit MHA/MQA/GQA/MLA/Sparse/MoE/Speculative/Hybrid/Distributed trigger conditions and explicit non-primary cases.

# 17. Canonical status after v13

```text
RQ taxonomy                         FROZEN
historical Markdown audit           COMPLETE
frozen 115-unit source ledger       COMPLETE within declared scope
current-source supplementation      COMPLETE for identified weak chains
per-rule proxy audit                COMPLETE
per-rule ignored-variable audit     COMPLETE
per-rule fallback/repair audit      COMPLETE
modern-model trigger audit          COMPLETE
hypothesis/falsifier mapping        COMPLETE
experiment design RQ1-RQ10          COMPLETE DESIGN
actual GPU evidence                 OPEN
empirical verdicts                  OPEN
repository-future-exhaustiveness    OPEN by definition
```

No additional conceptual expansion is justified before measurements.

# 18. Files to use going forward

- **This document:** canonical human-readable source/reasoning trial.
- `layout_summary_v13_complete_rule_proxy_fallback_trigger_hypothesis_chain.csv`: canonical machine-readable rule chain.
- `layout_summary_v13_current_source_supplements.csv`: current-source supplements with assumptions/triggers.
- `layout_summary_v11_full_empirical_adjudication_protocol_RQ1_RQ10.md`: canonical experiment design.
- `layout_summary_v11_RQ1_RQ10_hypothesis_falsifier_matrix.csv`: hypothesis-to-falsifier/test mapping.

The next non-experimental step is implementation specification, not adding more research questions.
