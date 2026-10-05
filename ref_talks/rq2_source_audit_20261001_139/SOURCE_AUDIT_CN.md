# 全链接源码审计

源码可访问不是性能验证成功。每个链接独立记录。

| URL | status | title | native replay |
|---|---|---|---|
| https://github.com/Dao-AILab/flash-attention/pull/2336 | verified | Add SM120 split-KV (FlashDecoding) with FP32 partial outputs | not_run |
| https://github.com/Dao-AILab/flash-attention/pull/2443 | verified | [Fwd,Sm100] TMA-paged KV cache loading for small page sizes | not_run |
| https://github.com/Dao-AILab/flash-attention/pull/2667 | verified | fix(hd256/sm100): make the dedicated hd256 kernel stride-aware (supersede #2666 copy) | not_run |
| https://github.com/Dao-AILab/flash-attention/pull/2777 | verified | [FA2] Fix backward for non-contiguous Q/K/V inputs | not_run |
| https://github.com/Dao-AILab/flash-attention/pull/2785 | verified | [CuTe] Add packed QKV and KV public APIs | not_run |
| https://github.com/Dao-AILab/flash-attention/pull/2809 | verified | [CuTe, SM100] writeback LSE to shared memory when `return_lse=False` to avoid compiler-induced 14% perf diff | not_run |
| https://github.com/NVIDIA/TensorRT-LLM/pull/18763 | verified | [None][feat] add Rubin topology and prototype locality-domain sharding to GVR V2 decode | not_run |
| https://github.com/NVIDIA/TensorRT-LLM/pull/19115 | verified | [None][feat] Add PrimsTS MoE with Rubin locality domains | not_run |
| https://github.com/NVIDIA/TensorRT-LLM/pull/19338 | verified | [None][feat] Upgrade Blackwell cuteDSL MLA kernel for packed q heads and compact input | not_run |
| https://github.com/NVIDIA/TensorRT-LLM/pull/19358 | verified | [None][perf] Fuse strided MXFP8 packing for cuDNN attention | not_run |
| https://github.com/NVIDIA/TensorRT-LLM/pull/19541 | verified | [None][perf] MXFP8 locality-domain split: one parent reset for both shards, static planner decision | not_run |
| https://github.com/NVIDIA/TensorRT-LLM/pull/19542 | verified | [None][feat] CUTEDSL: switch the Rubin NVFP4 FC1 weight layout to gate16/up16 | not_run |
| https://github.com/NVIDIA/TensorRT-LLM/pull/19731 | verified | [None][perf] MiniMax-M3: native P128 draft KV view for the NVFP4 hybrid cache | not_run |
| https://github.com/NVIDIA/TensorRT-LLM/pull/19749 | verified | [None][feat] MoE scheduler: optional finalize-before-combine and FP4 scale-layout contract fix | not_run |
| https://github.com/NVIDIA/cutlass/issues/3612 | verified | [QST] Should `MMA_Atom::get_layoutC_TV()` mask or ignore `ThrK`? | not_run |
| https://github.com/NVIDIA/cutlass/pull/3030 | verified | [CuTeDSL] Flash Attention v2 for SM120 (Blackwell GeForce) | not_run |
| https://github.com/NVIDIA/cutlass/pull/3194 | verified | Add Hopper FP8 grouped blockwise GEMM CuTeDSL example | not_run |
| https://github.com/NVIDIA/cutlass/pull/3256 | verified | [Cutlass SM90] Per-group aux TMA descriptor update for grouped GEMM + Gated-SwiGLU example) | not_run |
| https://github.com/NVIDIA/cutlass/pull/3273 | verified | [CuTeDSL] Add SM120 MXF4/NVFP4 native-TMA path | not_run |
| https://github.com/NVIDIA/cutlass/pull/3345 | verified | Fix SM120 blockscaled FP32 epilogue store layout | not_run |
| https://github.com/NVIDIA/cutlass/pull/3453 | verified | [CuTe DSL] Preserve runtime tensor strides in Blackwell FMHA | not_run |
| https://github.com/NVIDIA/cutlass/pull/3660 | verified | [CuTe] Keep K-slices distinct in TiledMMA::get_layoutC_TV (#3612) | not_run |
| https://github.com/ROCm/aiter/pull/2919 | verified | Add paged_attention_ragged_nhd | not_run |
| https://github.com/ROCm/aiter/pull/4057 | verified | [Triton][GDN] Support V-major (hvk) state layout in decode kernel | not_run |
| https://github.com/ROCm/aiter/pull/4550 | verified | [FlyDSL] flydsl_gdr_decode: read strided q/k/v directly (qkv_contiguous=False) | not_run |
| https://github.com/ROCm/aiter/pull/5094 | verified | [Triton/Gluon] Take the MOE scale layout from shuffle_scale_moe | not_run |
| https://github.com/ROCm/aiter/pull/5495 | verified | [HIP] [FlyDSL] [JIT] [gfx950] Integrate layout-dynamic MXFP8 GEMM with tuning and AOT | not_run |
| https://github.com/ROCm/aiter/pull/5543 | verified | [FlyDSL] Add token-major layout option to GDN prefill h kernel | not_run |
| https://github.com/apache/tvm/pull/17599 | verified | [RELAX][PASS] Annotate Custom Scope layout pass for Adreno GPU | not_run |
| https://github.com/apache/tvm/pull/18523 | verified | [ADRENO][TEXTURE] Texture based lowering | not_run |
| https://github.com/apache/tvm/pull/18548 | verified | [Relax][PyTroch] Add NHWC layout support | not_run |
| https://github.com/apache/tvm/pull/19896 | verified | [TIRx] Bundle CUDA tile primitive and op dispatch updates | not_run |
| https://github.com/apache/tvm/pull/20076 | verified | [FIX][TIRx] Use physical order for Buffer.local views | not_run |
| https://github.com/apache/tvm/pull/20080 | verified | [TIRx] tcgen05 dispatch paths, buffer dim-surgery views, FlashMLA lowering, and typed-buffer migration fixes | not_run |
| https://github.com/apache/tvm/pull/20329 | verified | [FIX][TIRx][CUDA] Support SM100 weight-stationary B collectors | not_run |
| https://github.com/apache/tvm/pull/20426 | verified | [Relax][WebGPU] Sequence-sharded paged decode attention for long contexts | not_run |
| https://github.com/deepseek-ai/DeepGEMM/pull/183 | verified | [Feat] Single Batch Overlap (SBO): Overlaping of Down GEMM with Combine Send | not_run |
| https://github.com/deepseek-ai/DeepGEMM/pull/311 | verified | fix: correct fused KV cache stride in paged MQA logits | not_run |
| https://github.com/deepseek-ai/DeepGEMM/pull/394 | verified | MegaMoE: add cluster-paired FP4 weight packets | not_run |
| https://github.com/deepseek-ai/DeepGEMM/pull/403 | verified | Fix SM120 scale-factor layout transformation | not_run |
| https://github.com/deepseek-ai/DeepGEMM/pull/440 | verified | Add a fused SM90 FP8 MegaMoE: one kernel, a lag-ordered schedule and a ring activation pool | not_run |
| https://github.com/flashinfer-ai/flashinfer/issues/1474 | verified | TMA loads turned off for paged KV cache kernel | not_run |
| https://github.com/flashinfer-ai/flashinfer/issues/3617 | verified | [feat] support fp8 per-token-head kv cache with inline scale | not_run |
| https://github.com/flashinfer-ai/flashinfer/pull/4572 | verified | fix(gdn): honor intermediate cache view strides | not_run |
| https://github.com/flashinfer-ai/flashinfer/pull/4574 | verified | fix(moe): expose tactic-dependent MXFP8 activation scale layout | not_run |
| https://github.com/flashinfer-ai/flashinfer/pull/4657 | verified | feat: autotune dynamic MXFP8 layout and TRTLLM tactic | not_run |
| https://github.com/flashinfer-ai/flashinfer/pull/4971 | verified | perf(attention): TMA loads for paged KV in the SM90 prefill kernel | not_run |
| https://github.com/flashinfer-ai/flashinfer/pull/5094 | verified | perf(msa): support packed FP8 KV and token-major top-k | not_run |
| https://github.com/flashinfer-ai/flashinfer/pull/5117 | verified | feat: add experimental Frost DeepSeek V4.1 cache and quantization kernels | not_run |
| https://github.com/flashinfer-ai/flashinfer/pull/5272 | verified | feat(attention): FA2 per-(token, head) FP8 KV scale | not_run |
| https://github.com/flashinfer-ai/flashinfer/pull/5405 | verified | feat(rope): fuse QK norm, RoPE, and paged KV append | not_run |
| https://github.com/flashinfer-ai/flashinfer/pull/5447 | verified | feat(attention): add PrimTS DSV4 sparse MLA kernels | not_run |
| https://github.com/flashinfer-ai/flashinfer/pull/5493 | verified | feat(cake_minimax_h3): SM120 FP8/NVFP4 fused MiniMax-H3 pre-attention (RTX 5090 / RTX PRO 6000) | not_run |
| https://github.com/flashinfer-ai/flashinfer/pull/5667 | verified | feat: add concat_mla_kv_quant_fp8, a fused MLA context K/V pack with saturating fp8 cast | not_run |
| https://github.com/flashinfer-ai/flashinfer/pull/5709 | verified | perf(mla): prefetch split partials in the CuTe DSL MLA decode reducer | not_run |
| https://github.com/flashinfer-ai/flashinfer/pull/5737 | verified | feat(cake_dsa): SM107 (Rubin) generated programs and strided / packed GLM-5.2 trainer layouts for the native 64-head DSA sparse-attention training kernels (#5675) | not_run |
| https://github.com/flashinfer-ai/flashinfer/pull/5757 | verified | [prims-ts] flexible nvfp4 kv cache layout  | not_run |
| https://github.com/sgl-project/sglang/issues/12196 | verified | [Feature]  Implement Decode Context Parallel in SGLang | not_run |
| https://github.com/sgl-project/sglang/pull/12892 | verified | [GDN/Qwen3-Next] Avoid SSM and conv state copy for speculative decoding - up to 9.47% e2e speedup | not_run |
| https://github.com/sgl-project/sglang/pull/14982 | verified | [Feature] Add DCP support for GQA with flashinfer | not_run |
| https://github.com/sgl-project/sglang/pull/33651 | verified | Unified Full KV Cache Layout in L3 | not_run |
| https://github.com/sgl-project/sglang/pull/34299 | verified | [KDA] Add zero-copy native prefill checkpoints and packed decode | not_run |
| https://github.com/sgl-project/sglang/pull/37521 | verified | [Diffusion][Perf] Consume packed Ulysses QKV without copies | not_run |
| https://github.com/sgl-project/sglang/pull/38373 | verified | [AMD][DSV4] Rename unified_kv to the ring KV layout and select it via --dsv4-kv-layout | not_run |
| https://github.com/sgl-project/sglang/pull/38430 | verified | Enable compact GLM NoPE KV storage with native FlashInfer | not_run |
| https://github.com/sgl-project/sglang/pull/38592 | verified | [unified-memory] Token-major dense views for the unified memory pool (3/7) | not_run |
| https://github.com/sgl-project/sglang/pull/39039 | verified | [MoonEP] Use DeepGEMM psum layout, zero-copy shard and other optimization | not_run |
| https://github.com/sgl-project/sglang/pull/39388 | verified | [MegaMoE] Preserve W13 layout for ModelOpt NVFP4 experts | not_run |
| https://github.com/sgl-project/sglang/pull/39606 | html_verified_body_pending | [HiCache] Add staged write-back for the page-unified KV cache layout by huangtingwei9988 · Pull Request #39606 · sgl-project/sglang · GitHub | not_run |
| https://github.com/sgl-project/sglang/pull/39860 | html_verified_body_pending | Add optional strip layout for Inkling speculative convolutions by shenxiul · Pull Request #39860 · sgl-project/sglang · GitHub | not_run |
| https://github.com/sgl-project/sglang/pull/40326 | html_verified_body_pending | [unified-memory] Derive KV row addresses from strides, not shapes (1/7) by caihuali95 · Pull Request #40326 · sgl-project/sglang · GitHub | not_run |
| https://github.com/sgl-project/sglang/pull/40327 | html_verified_body_pending | [unified-memory] Build paged KV views through one helper (2/7) by caihuali95 · Pull Request #40327 · sgl-project/sglang · GitHub | not_run |
| https://github.com/sgl-project/sglang/pull/40330 | html_verified_body_pending | [unified-memory] Stride the KV translate kernel and route every translate through it (6/7) by caihuali95 · Pull Request #40330 · sgl-project/sglang · GitHub | not_run |
| https://github.com/sgl-project/sglang/pull/41030 | html_verified_body_pending | [AMD] Eliminate remaining gfx95 FP8 scale relayout copies by xiaofei-zheng · Pull Request #41030 · sgl-project/sglang · GitHub | not_run |
| https://github.com/sgl-project/sglang/pull/41944 | html_verified_body_pending | [diffusion][xpu] Feed row-strided q/k/v to XPU flash attention without copies by AgainstEntropy · Pull Request #41944 · sgl-project/sglang · GitHub | not_run |
| https://github.com/sgl-project/sglang/pull/41953 | html_verified_body_pending | [PD] Expose unified memory transfer layouts by ByronHsu · Pull Request #41953 · sgl-project/sglang · GitHub | not_run |
| https://github.com/tile-ai/tilelang/issues/1336 | html_verified_body_pending | [BUG] example_gqa_decode.py fails during tuning · Issue #1336 · tile-ai/tilelang · GitHub | not_run |
| https://github.com/tile-ai/tilelang/issues/1729 | html_verified_body_pending | [Feature Request] Improve layout when small fragments cause excessive thread replication · Issue #1729 · tile-ai/tilelang · GitHub | not_run |
| https://github.com/tile-ai/tilelang/issues/2408 | html_verified_body_pending | [BUG][Fuzzer][wrong-code]  `T.alloc_reducer(replication="all")` + `T.finalize_reducer` silently returns `floor(threads / block_N)` × the correct sum instead of `A @ x` when `block_N` doesn't divide the thread count · Issue #2408 · tile-ai/tilelang · GitHub | not_run |
| https://github.com/tile-ai/tilelang/issues/2897 | html_verified_body_pending | [Feature Request][RFC] Redesign reducers as ownership-safe deferred reductions · Issue #2897 · tile-ai/tilelang · GitHub | not_run |
| https://github.com/tile-ai/tilelang/issues/3283 | html_verified_body_pending | [BUG][CUDA] SM120 block-scaled GEMM overwrites layouts when SFA and SFB share a fragment · Issue #3283 · tile-ai/tilelang · GitHub | not_run |
| https://github.com/tile-ai/tilelang/pull/1386 | html_verified_body_pending | [BugFix] Fix split kernel layout bug of GQA decode by tzj-fxz · Pull Request #1386 · tile-ai/tilelang · GitHub | not_run |
| https://github.com/tile-ai/tilelang/pull/1509 | html_verified_body_pending | [Refactor] Support auto swizzling for tma store and phaseout related layout annotations by LeiWang1999 · Pull Request #1509 · tile-ai/tilelang · GitHub | not_run |
| https://github.com/tile-ai/tilelang/pull/1559 | html_verified_body_pending | [Parallel][Infer] Free-mode chooses minimal replication between buffer-based and PlanLoopPartition by LeiWang1999 · Pull Request #1559 · tile-ai/tilelang · GitHub | not_run |
| https://github.com/tile-ai/tilelang/pull/2452 | html_verified_body_pending | [Backend] [CUDA] Support GMMA/UMMA lowering for sliced SMEM layout (actually arbitrary layout) by Yongqi-Zhuo · Pull Request #2452 · tile-ai/tilelang · GitHub | not_run |
| https://github.com/tile-ai/tilelang/pull/2785 | html_verified_body_pending | [CUDA] Support arbitrary TMEM layouts by Yongqi-Zhuo · Pull Request #2785 · tile-ai/tilelang · GitHub | not_run |
| https://github.com/tile-ai/tilelang/pull/2807 | html_verified_body_pending | [BugFix] Respect layouts in logical reductions by lijinpei · Pull Request #2807 · tile-ai/tilelang · GitHub | not_run |
| https://github.com/tile-ai/tilelang/pull/2940 | html_verified_body_pending | [Lang] Reducer v2: first-class deferred reduction epochs with planned physical lowering by LeiWang1999 · Pull Request #2940 · tile-ai/tilelang · GitHub | not_run |
| https://github.com/tile-ai/tilelang/pull/2953 | html_verified_body_pending | [Layout][CUDA] Support reinterpreting (dtype-changing) T.view aliases by Yongqi-Zhuo · Pull Request #2953 · tile-ai/tilelang · GitHub | not_run |
| https://github.com/tile-ai/tilelang/pull/2960 | html_verified_body_pending | [Layout][Inference] Add IO-aware cost model for free-mode selection by LeiWang1999 · Pull Request #2960 · tile-ai/tilelang · GitHub | not_run |
| https://github.com/tile-ai/tilelang/pull/3084 | html_verified_body_pending | [Layout][Reducer] Steer reducer destinations during layout inference by LeiWang1999 · Pull Request #3084 · tile-ai/tilelang · GitHub | not_run |
| https://github.com/tile-ai/tilelang/pull/3176 | html_verified_body_pending | [Layout] Make default vector size selection reduction-aware by SiriusNEO · Pull Request #3176 · tile-ai/tilelang · GitHub | not_run |
| https://github.com/tile-ai/tilelang/pull/3180 | html_verified_body_pending | [Fix] Enhance Layout Inference of fragment slicing and partial visit in `T.Parallel` by KellyFrog · Pull Request #3180 · tile-ai/tilelang · GitHub | not_run |
| https://github.com/tile-ai/tilelang/pull/3255 | html_verified_body_pending | [ROCm] Fuse GLM-5.3 paged k-pool selection by andyluo7 · Pull Request #3255 · tile-ai/tilelang · GitHub | not_run |
| https://github.com/tile-ai/tilelang/pull/3279 | html_verified_body_pending | [Compiler] Unify async data staging and transfer-aware lowering by sepcnt · Pull Request #3279 · tile-ai/tilelang · GitHub | not_run |
| https://github.com/tile-ai/tilelang/pull/3284 | html_verified_body_pending | [CUDA] Reject conflicting SM120 scale fragment layouts by LeiWang1999 · Pull Request #3284 · tile-ai/tilelang · GitHub | not_run |
| https://github.com/tile-ai/tilelang/pull/3320 | html_verified_body_pending | feat(ascend): Add AutoSimtVF pass and register it in Ascend pipeline by yangsichan · Pull Request #3320 · tile-ai/tilelang · GitHub | not_run |
| https://github.com/tile-ai/tilelang/pull/3328 | html_verified_body_pending | feat(ascend): support L0C fp32->f16/bf16 dual_copy on-path cast  by CeleNewYear · Pull Request #3328 · tile-ai/tilelang · GitHub | not_run |
| https://github.com/triton-lang/triton/issues/10987 | html_verified_body_pending | Triton generates pathologically large PTX for chained fp32 division-by-broadcast feeding a reduction · Issue #10987 · triton-lang/triton · GitHub | not_run |
| https://github.com/triton-lang/triton/issues/11526 | html_verified_body_pending | [Regression] Increased SMEM usage with tf32x3 GEMM (3.6 -> 3.8) · Issue #11526 · triton-lang/triton · GitHub | not_run |
| https://github.com/triton-lang/triton/issues/8281 | html_verified_body_pending | [RFC][AMD] Optimizations for Paged Attention: Proposal with Multiple Features · Issue #8281 · triton-lang/triton · GitHub | not_run |
| https://github.com/triton-lang/triton/pull/10158 | html_verified_body_pending | Improve tensor layout propagations through the control flow by Hardcode84 · Pull Request #10158 · triton-lang/triton · GitHub | not_run |
| https://github.com/triton-lang/triton/pull/10360 | html_verified_body_pending | [BACKEND] Avoid Shared Memory for small-tensor layout conversions/transpositions by mwichro · Pull Request #10360 · triton-lang/triton · GitHub | not_run |
| https://github.com/triton-lang/triton/pull/10563 | html_verified_body_pending | [AMD] Select WMMA result layout based on consuming store order by dhernandez0 · Pull Request #10563 · triton-lang/triton · GitHub | not_run |
| https://github.com/triton-lang/triton/pull/10837 | html_verified_body_pending | [TRITONGPU] Propagate assigned CGA layouts to loads by Jokeren · Pull Request #10837 · triton-lang/triton · GitHub | not_run |
| https://github.com/triton-lang/triton/pull/11117 | html_verified_body_pending | [TritonGPU] Prefer least-replicated layout when resolving elementwise layout conflicts by ArchanaChetan07 · Pull Request #11117 · triton-lang/triton · GitHub | not_run |
| https://github.com/triton-lang/triton/pull/11295 | html_verified_body_pending | [AMD] Swizzle clamping for the direct-to-lds path by erizheng-amd · Pull Request #11295 · triton-lang/triton · GitHub | not_run |
| https://github.com/triton-lang/triton/pull/11360 | html_verified_body_pending | [KERNELS] Reduce MXFP4 layout conversion allocations by roman-openai · Pull Request #11360 · triton-lang/triton · GitHub | not_run |
| https://github.com/triton-lang/triton/pull/11538 | html_verified_body_pending | [TRITONGPU] Try reusing existing layout rematerialization before hoisting layout conversion  by kainzhong · Pull Request #11538 · triton-lang/triton · GitHub | not_run |
| https://github.com/triton-lang/triton/pull/11685 | html_verified_body_pending | [Gluon] Improve auto layout propagation by peterbell10 · Pull Request #11685 · triton-lang/triton · GitHub | not_run |
| https://github.com/triton-lang/triton/pull/11704 | html_verified_body_pending | [AMD] Bypass the epilogue relayout for FMA by pabloantoniom · Pull Request #11704 · triton-lang/triton · GitHub | not_run |
| https://github.com/triton-lang/triton/pull/11754 | html_verified_body_pending | Pipeline converted matmul operands with bounded buffers by roman-openai · Pull Request #11754 · triton-lang/triton · GitHub | not_run |
| https://github.com/triton-lang/triton/pull/11760 | html_verified_body_pending | Absorb compatible layouts into broadcasts by neildhar · Pull Request #11760 · triton-lang/triton · GitHub | not_run |
| https://github.com/triton-lang/triton/pull/11777 | html_verified_body_pending | [TritonGPU][Gluon] Support logical shared memory indexing by bangtianliu · Pull Request #11777 · triton-lang/triton · GitHub | not_run |
| https://github.com/triton-lang/triton/pull/11780 | html_verified_body_pending | Drop obsolete layout conversions when repartitioning warps by roman-openai · Pull Request #11780 · triton-lang/triton · GitHub | not_run |
| https://github.com/triton-lang/triton/pull/11967 | html_verified_body_pending | [Layouts] Account for masks when choosing coalesced layouts by peterbell10 · Pull Request #11967 · triton-lang/triton · GitHub | not_run |
| https://github.com/triton-lang/triton/pull/7447 | html_verified_body_pending | [Gluon] Add AutoLayout for backward layout inference by peterbell10 · Pull Request #7447 · triton-lang/triton · GitHub | not_run |
| https://github.com/triton-lang/triton/pull/7718 | html_verified_body_pending | [Gluon] Fix auto_encoding for ops which may infer multiple layouts by peterbell10 · Pull Request #7718 · triton-lang/triton · GitHub | not_run |
| https://github.com/triton-lang/triton/pull/8398 | html_verified_body_pending | [BACKEND] Add heuristic to rematerialize through forOps for pointer types in `RemoveLayoutConversion` by AlexAUT · Pull Request #8398 · triton-lang/triton · GitHub | not_run |
| https://github.com/triton-lang/triton/pull/8450 | html_verified_body_pending | [AMD][Draft] Implement implicit layout conversion for DotOp to enable direct GMEM to reg loads by the-strawhat · Pull Request #8450 · triton-lang/triton · GitHub | not_run |
| https://github.com/vllm-project/vllm/issues/26744 | html_verified_body_pending | [RFC]: Nixl Connector Heterogeneous BlockSize support · Issue #26744 · vllm-project/vllm · GitHub | not_run |
| https://github.com/vllm-project/vllm/issues/46204 | html_verified_body_pending | [Bug]: MiniMax‑M3 (MSA) + NixlConnector: connector forces HND KV layout while MSA is NHD‑native, permuting the head axis under P/D disaggregation · Issue #46204 · vllm-project/vllm · GitHub | not_run |
| https://github.com/vllm-project/vllm/pull/30448 | html_verified_body_pending | [NIXL] Heterogeneous KV Layout and block_size - prefill NHD and nP > nD support by xuechendi · Pull Request #30448 · vllm-project/vllm · GitHub | not_run |
| https://github.com/vllm-project/vllm/pull/37885 | html_verified_body_pending | [KV Connector] Canonical KV Cache Allocation for HMA Models by Etelis · Pull Request #37885 · vllm-project/vllm · GitHub | not_run |
| https://github.com/vllm-project/vllm/pull/44458 | html_verified_body_pending | [6/N][KV-Cache Layout Refactor] Standardize KV cache layout by LucasWilkinson · Pull Request #44458 · vllm-project/vllm · GitHub | not_run |
| https://github.com/vllm-project/vllm/pull/46223 | html_verified_body_pending | [Attention] MiniMax-M3: declare NHD-only KV cache layout via get_required_kv_cache_layout by Pranav-d33 · Pull Request #46223 · vllm-project/vllm · GitHub | not_run |
| https://github.com/vllm-project/vllm/pull/49335 | verified | [Bugfix][MoE] Swizzle mxfp8 activation scales after DP/EP dispatch for FlashInfer CUTLASS | not_run |
| https://github.com/vllm-project/vllm/pull/50208 | html_verified_body_pending | [Bugfix][KV Connector][Mooncake] Preserve HMA region strides by Dao007forever · Pull Request #50208 · vllm-project/vllm · GitHub | not_run |
| https://github.com/vllm-project/vllm/pull/50717 | html_verified_body_pending | [Bugfix][NIXL] Preserve block geometry for packed MLA caches by xijiaat · Pull Request #50717 · vllm-project/vllm · GitHub | not_run |
| https://github.com/vllm-project/vllm/pull/52240 | html_verified_body_pending | [Quantization] Clarify MXFP4 MoE W13 layout requirement from backends by BowenBao · Pull Request #52240 · vllm-project/vllm · GitHub | not_run |
| https://github.com/vllm-project/vllm/pull/54795 | html_verified_body_pending | [Bugfix] Intersect per-worker KV cache layout support by saichowdary007 · Pull Request #54795 · vllm-project/vllm · GitHub | not_run |
| https://github.com/vllm-project/vllm/pull/56989 | html_verified_body_pending | [Perf][DSv4.1] Keep FP8 Engram rows packed through TP exchange by wangyicong52 · Pull Request #56989 · vllm-project/vllm · GitHub | not_run |
| https://github.com/vllm-project/vllm/pull/57169 | html_verified_body_pending | [KV Cache] GLM-5.3-Flash: use the generic packed KV layout; CircularBufferSpec tail with speculative ring slots by andakai · Pull Request #57169 · vllm-project/vllm · GitHub | not_run |
| https://github.com/vllm-project/vllm/pull/58766 | html_verified_body_pending | [Perf][DSv4.1] Feed Engram's FP8 rows to wkv without dequant + requant by zyongye · Pull Request #58766 · vllm-project/vllm · GitHub | not_run |
| https://github.com/vllm-project/vllm/pull/59112 | html_verified_body_pending | [Attention] Support FlashInfer re-paging for packed BLHNC KV caches by leonardHONG · Pull Request #59112 · vllm-project/vllm · GitHub | not_run |
| https://github.com/vllm-project/vllm/pull/59265 | html_verified_body_pending | [KV Offload] Preserve packed layouts for layer-wise KV cache by meenchen · Pull Request #59265 · vllm-project/vllm · GitHub | not_run |
| https://github.com/vllm-project/vllm/pull/59337 | html_verified_body_pending | [MoE] Select DeepEP v2 dispatch layout per forward by tlrmchlsmth · Pull Request #59337 · vllm-project/vllm · GitHub | not_run |
| https://github.com/vllm-project/vllm/pull/59342 | html_verified_body_pending | [Attention][MLA] Support nvfp4_ds_mla on FLASHINFER_MLA_SPARSE, with native NVFP4 decode by stu-cao · Pull Request #59342 · vllm-project/vllm · GitHub | not_run |
| https://github.com/vllm-project/vllm/pull/59350 | html_verified_body_pending | [Perf][KV Connector] Drop the redundant index_select in kv_postprocess_layout_on_receive by konfan12345678 · Pull Request #59350 · vllm-project/vllm · GitHub | not_run |
