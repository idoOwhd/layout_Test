# layout_summary_v3_forensic_provenance_complete.md

> **Forensic snapshot date:** 2026-09-16  
> **Supersedes:** `layout_summary_v3_provenance_ledger.md` draft  
> **Parent synthesis:** `layout_summary_v2_horizontal_framework_semantic_research_map.md`
>
> **Goal:** pin every horizontal semantic evidence unit to an immutable repository snapshot and an audited source/document range, while preserving the distinction between source fact and research inference.

---

# 0. Final provenance verdict

Within the **declared 50-unit horizontal evidence scope**, the provenance step is complete:

- **50/50** evidence IDs pinned to immutable repository SHAs.
- **50/50** have primary file + symbol/section.
- **50/50** have an audited source/document range at the pinned SHA.
- **49 FACT / 1 INFERENCE**.
- `CODE`, `DOC`, and `CODE+DOC` evidence remain explicitly distinguished.
- Two v2 overclaims were corrected instead of being carried forward.
- One initially conservative v3 correction (IREE HI05) was **reversed after direct source confirmation**.

This supports the claim:

> **Forensic provenance is complete for the declared semantic evidence ledger.**

It does **not** support the stronger claim:

> “Every relevant branch in all nine repositories has been mechanically enumerated.”

Repository-exhaustive branch enumeration remains a different task.

---

# 1. Evidence semantics and confidence

## 1.1 Provenance kind

- **CODE** — behavior/predicate/contract is directly represented in implementation or source docstring.
- **DOC** — immutable repository design/reference documentation is the authoritative policy statement.
- **CODE+DOC** — code and same-snapshot explanatory material jointly establish the fact.

## 1.2 Evidence role

- **FACT** — directly established by the audited range.
- **INFERENCE** — architectural conclusion drawn from direct facts. It must never be phrased as a literal code statement.

Only **HR05** remains an intentional INFERENCE: Triton autotune operates over caller-declared kernels/configuration spaces, so “whole-LLM graph semantics remain an outer policy” is a scope inference.

## 1.3 Audited range

`AUDITED-RANGE` means the immutable file at the pinned SHA was re-fetched and the stated range was inspected.  
Ranges are deliberately sometimes broader than a single predicate so the engineering context/fallback is preserved.

---

# 2. Immutable repository manifest

| repository | snapshot_SHA | immutable_root |
|---|---|---|
| NVIDIA/TensorRT-LLM | 7c79c15fa19c4a3b4e6bae15ea741541557f6c92 | https://github.com/NVIDIA/TensorRT-LLM/tree/7c79c15fa19c4a3b4e6bae15ea741541557f6c92 |
| NVIDIA/cutlass | 147295a3d4b75f3aeff247c25b8927cea9a7006a | https://github.com/NVIDIA/cutlass/tree/147295a3d4b75f3aeff247c25b8927cea9a7006a |
| apache/tvm | 8312a17f8734ddfd56e5f3977cd5df25b83ec49f | https://github.com/apache/tvm/tree/8312a17f8734ddfd56e5f3977cd5df25b83ec49f |
| flashinfer-ai/flashinfer | c039288594d3f97404166e1f8111f4c5c88143ad | https://github.com/flashinfer-ai/flashinfer/tree/c039288594d3f97404166e1f8111f4c5c88143ad |
| iree-org/iree | 83265e80a3b99d9896e32b7c36795f83e9a23e45 | https://github.com/iree-org/iree/tree/83265e80a3b99d9896e32b7c36795f83e9a23e45 |
| llvm/llvm-project | e94698b83a74e6ee40438898b17bc49cbd47fcf5 | https://github.com/llvm/llvm-project/tree/e94698b83a74e6ee40438898b17bc49cbd47fcf5 |
| sgl-project/sglang | ad28b91faec29cd9e51ecc5b76b06327ed93389e | https://github.com/sgl-project/sglang/tree/ad28b91faec29cd9e51ecc5b76b06327ed93389e |
| triton-lang/triton | 570e5b4dd1e2d1daf785a894d19268fe7b281cb5 | https://github.com/triton-lang/triton/tree/570e5b4dd1e2d1daf785a894d19268fe7b281cb5 |
| vllm-project/vllm | 9ca6dbba71329a7b08711e2acfa17b31e542553c | https://github.com/vllm-project/vllm/tree/9ca6dbba71329a7b08711e2acfa17b31e542553c |

---

# 3. Full forensic evidence ledger

| id | framework | semantic_class | subgraph | evidence_role | provenance_kind | confidence | repo | sha | path | symbol_or_section | audited_range | direct_fact | proxy | assumption | ignored | fallback | optimization_abstraction | rq | correction | secondary_sources | immutable_url |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HC01 | CUTLASS/CuTe | C1 Analytical heuristic | SG5 | FACT | DOC | AUDITED-RANGE | NVIDIA/cutlass | 147295a3d4b75f3aeff247c25b8927cea9a7006a | media/docs/cpp/heuristics.md | GEMM Heuristics overview / coverage | 1-140 | CUTLASS GEMM heuristics use NVIDIA's analytical matmul heuristic to rank a subset of valid kernels from problem size and hardware SKU, reducing runtime autotuning search; exhaustive coverage is explicitly not guaranteed. | shape; SKU; registered kernels | Analytical ranking reduces profiling while preserving strong candidates. | cross-op layout; online serving state | Profile/rank subset. | Analytical candidate pruning. | RQ9 |  |  | https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/media/docs/cpp/heuristics.md#L1-L140 |
| HC02 | CUTLASS/CuTe | C1 Layout heuristic | SG5 | FACT | CODE | AUDITED-RANGE | NVIDIA/cutlass | 147295a3d4b75f3aeff247c25b8927cea9a7006a | python/CuTeDSL/cutlass/utils/blackwell_helpers.py | get_smem_layout_atom_ab / SMEM store helpers | 520-760 | The helper calls its policy a simple heuristic and selects SMEM layout/swizzle atoms from major mode, element type and major-dimension size; related helpers choose legal/vectorized store atoms. | output layout; dtype; tile; ownership | Local structural features suffice for SMEM choice. | producer/consumer graph; concurrent pressure | Legal fallback store/layout. | Bank conflict/alignment/vectorization crossover. | RQ7/RQ9 |  |  | https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/python/CuTeDSL/cutlass/utils/blackwell_helpers.py#L520-L760 |
| HC03 | CUTLASS/CuTe | C3 Template family | SG5/6 | FACT | DOC | AUDITED-RANGE | NVIDIA/cutlass | 147295a3d4b75f3aeff247c25b8927cea9a7006a | media/docs/cpp/gemm_api_3x.md | CUTLASS 3.x GEMM components / kernel assembly | 1-180 | CUTLASS 3.x explicitly decomposes GEMM into device, kernel, collective, tiled MMA/copy and atom levels; kernels are assembled from a collective mainloop and collective epilogue, with persistent kernels querying a work-tile scheduler. | template types; dtype/layout/tile | Composable templates span useful hardware mappings. | global graph equivalence and runtime distribution | Another template/operator. | Expressive kernel registry. | RQ7/RQ8 |  |  | https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/media/docs/cpp/gemm_api_3x.md#L1-L180 |
| HC04 | CUTLASS/CuTe | C2/C4 Algebraic repair | SG5 | FACT | DOC | AUDITED-RANGE | NVIDIA/cutlass | 147295a3d4b75f3aeff247c25b8927cea9a7006a | media/docs/cpp/gemm_api.md | Efficient Epilogue | 430-620 | CUTLASS documents an efficient row-major epilogue and, for column-major output, transposes/swaps the equivalent GEMM operands so the efficient epilogue mapping is preserved. | input/output layout | Algebraic transform is cheaper than poor epilogue access. | downstream consumer layout/fusion | Specialized template path. | Layout repair by equivalent computation. | RQ7/RQ10 |  |  | https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/media/docs/cpp/gemm_api.md#L430-L620 |
| HC05 | CUTLASS/CuTe | C4 Legality | SG5/6 | FACT | CODE | AUDITED-RANGE | NVIDIA/cutlass | 147295a3d4b75f3aeff247c25b8927cea9a7006a | python/CuTeDSL/cutlass/utils/blackwell_helpers.py | TMA alignment / architecture legality checks | 1-220 | CuTe DSL helpers fail fast on unsupported instruction/architecture combinations and enforce TMA contiguous-dimension alignment before performance choices are made. | swizzle; instruction; architecture | Legality must prune search before performance ranking. | higher-level graph opportunity cost | Pick legal atom/template. | Formal constraint layer. | RQ8/RQ10 |  |  | https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/python/CuTeDSL/cutlass/utils/blackwell_helpers.py#L1-L220 |
| HF01 | FlashInfer | C1 Heuristic | SG2 | FACT | CODE | AUDITED-RANGE | flashinfer-ai/flashinfer | c039288594d3f97404166e1f8111f4c5c88143ad | flashinfer/decode.py | single_decode_with_kv_cache(use_tensor_cores) | 520-760 | The API explicitly states that tensor-core decode can be faster for large grouped-query-attention group size; NHD/HND is an explicit KV-layout contract. | Hq/Hkv | GQA ratio predicts local compute/memory crossover. | global conversion/system contention | Non-tensor-core path. | Shape-aware local kernel selection. | RQ1/RQ9 |  |  | https://github.com/flashinfer-ai/flashinfer/blob/c039288594d3f97404166e1f8111f4c5c88143ad/flashinfer/decode.py#L520-L760 |
| HF02 | FlashInfer | C4 Conversion | SG2 | FACT | CODE | AUDITED-RANGE | flashinfer-ai/flashinfer | c039288594d3f97404166e1f8111f4c5c88143ad | flashinfer/decode.py | trtllm_batch_decode_with_kv_cache | 3100-3420 | For trtllm-gen with NVFP4 KV cache, NHD input triggers automatic transpose plus contiguous copies of KV data and scale tensors to HND, with documented allocation/copy overhead. | layout; dtype; backend | Conversion is preferable to rejection. | reuse count; temporary memory; producer-native HND | Materialized transpose/copy. | Explicit repair edge. | RQ3/RQ7/RQ10 |  |  | https://github.com/flashinfer-ai/flashinfer/blob/c039288594d3f97404166e1f8111f4c5c88143ad/flashinfer/decode.py#L3100-L3420 |
| HF03 | FlashInfer | C3 Kernel family | SG5/6 | FACT | CODE | AUDITED-RANGE | flashinfer-ai/flashinfer | c039288594d3f97404166e1f8111f4c5c88143ad | flashinfer/grouped_mm/core.py | grouped_mm_* public facade | 1-260 | Grouped-MoE GEMM APIs expose explicit tensor-shape/dtype contracts, backend dispatch and tactic selection; quantized variants add scale/format requirements. | dtype; architecture; scale/layout; tactic | Upstream policy can select from explicit native representations. | whole graph and network representation | Backend/tactic or unsupported error. | Kernel capability set for contract solver. | RQ7/RQ9 | v2 overstated this row as a general row/column-major and scale-major contract. v3 narrows it to the representation/backend/tactic contracts directly verified in grouped_mm/core.py. |  | https://github.com/flashinfer-ai/flashinfer/blob/c039288594d3f97404166e1f8111f4c5c88143ad/flashinfer/grouped_mm/core.py#L1-L260 |
| HF04 | FlashInfer | C1 Heuristic | SG6 | FACT | CODE | AUDITED-RANGE | flashinfer-ai/flashinfer | c039288594d3f97404166e1f8111f4c5c88143ad | flashinfer/grouped_mm/core.py | grouped_mm_bf16 / grouped_mm_fp8 tactic | 1-260 | For the cuDNN grouped-GEMM backend, tactic=-1 is documented as using the heuristic-best plan; non-negative tactic values select an explicit execution-plan index. | problem shape/library plan | Local library heuristic ranks execution plans well. | A2A token order; producer layout; serving interference | Heuristic or explicit tactic. | Local heuristic baseline. | RQ7/RQ9 |  |  | https://github.com/flashinfer-ai/flashinfer/blob/c039288594d3f97404166e1f8111f4c5c88143ad/flashinfer/grouped_mm/core.py#L1-L260 |
| HF05 | FlashInfer | C4 Planning | Runtime | FACT | CODE | AUDITED-RANGE | flashinfer-ai/flashinfer | c039288594d3f97404166e1f8111f4c5c88143ad | flashinfer/decode.py | BatchDecodeWithPagedKVCacheWrapper.__init__ | 760-1120 | When CUDA Graph mode is enabled, the wrapper documentation states that batch_size cannot change during the wrapper lifecycle; caller-owned buffers/workspace are planned around that lifetime constraint. | max tokens/batch; graph mode | Stable planning amortizes launch/workspace overhead. | workload drift and reconfiguration benefit | Recreate/replan or non-graph path. | Temporal switching-cost problem. | RQ6/RQ10 |  |  | https://github.com/flashinfer-ai/flashinfer/blob/c039288594d3f97404166e1f8111f4c5c88143ad/flashinfer/decode.py#L760-L1120 |
| HI01 | IREE | C1 Tuning | GPU dispatch | FACT | DOC | AUDITED-RANGE | iree-org/iree | 83265e80a3b99d9896e32b7c36795f83e9a23e45 | docs/website/docs/reference/tuning.md | Knobs in dispatches / generic tuning workflow | 1-260 | IREE exposes per-dispatch tuning knobs such as MMA layout, subgroup/workgroup parameters and reduction tiling; defaults target generic performance while tuning specializes them. | dispatch/target/knobs | Generic defaults are robust; tuning specializes. | persistent KV and cross-dispatch serving interactions | Tuning spec overrides. | Explicit constrained tuning space. | RQ9 |  |  | https://github.com/iree-org/iree/blob/83265e80a3b99d9896e32b7c36795f83e9a23e45/docs/website/docs/reference/tuning.md#L1-L260 |
| HI02 | IREE | C4 Formal legality | GPU tuning | FACT | DOC | AUDITED-RANGE | iree-org/iree | 83265e80a3b99d9896e32b7c36795f83e9a23e45 | docs/website/docs/reference/tuning.md | Constraint generation | 1-260 | IREE emits compiler pipeline constraints for legal knob combinations, serializes them to SMT-LIB in the tuner, and uses an SMT solver to enumerate valid assignments; an experimental verifier checks configurations. | compiler constraints | Compiler should own legality; tuner selection can be external. | cross-dispatch cost and online serving state | SMT excludes illegal assignments; verifier checks. | SMT legality already exists. | RQ8/RQ10 |  |  | https://github.com/iree-org/iree/blob/83265e80a3b99d9896e32b7c36795f83e9a23e45/docs/website/docs/reference/tuning.md#L1-L260 |
| HI03 | IREE | C3 Semantic match | GPU tuning | FACT | DOC | AUDITED-RANGE | iree-org/iree | 83265e80a3b99d9896e32b7c36795f83e9a23e45 | docs/website/docs/reference/tuning.md | Tuning specs / semantic match operations | 260-520 | IREE tuning specs are Transform-dialect libraries; IREE provides semantic match operations for contraction, convolution and attention that are described as more robust than generic DAG matching. | op semantics; shape/type; target | Semantic matcher robustly applies target configs. | multi-dispatch/persistent objective | Default/user spec fallback behavior. | Semantic configuration baseline. | RQ8/RQ9 |  |  | https://github.com/iree-org/iree/blob/83265e80a3b99d9896e32b7c36795f83e9a23e45/docs/website/docs/reference/tuning.md#L260-L520 |
| HI04 | IREE | C2 Layout propagation | Graph→GPU | FACT | CODE | AUDITED-RANGE | iree-org/iree | 83265e80a3b99d9896e32b7c36795f83e9a23e45 | compiler/src/iree/compiler/Dialect/Encoding/IR/EncodingOps.td | set_encoding / unset_encoding | 1-260 | The Encoding dialect attaches abstract data-layout encodings to tensors; downstream encoding/materialization infrastructure resolves these representations for targets. | op/operand/indexing maps/iteration/target | Deferred layout resolution enables target-specific data tiling. | LLM persistent multi-consumer/online traffic | Unset/materialize encoding. | Closest generic substrate to edge contracts. | RQ7 |  | compiler/src/iree/compiler/Dialect/Stream/Transforms/MaterializeEncodings.cpp; compiler/src/iree/compiler/Codegen/Common/MaterializeEncodingPatterns.cpp | https://github.com/iree-org/iree/blob/83265e80a3b99d9896e32b7c36795f83e9a23e45/compiler/src/iree/compiler/Dialect/Encoding/IR/EncodingOps.td#L1-L260 |
| HI05 | IREE | C1 Dynamic encoding | Generic layout | FACT | CODE | AUDITED-RANGE | iree-org/iree | 83265e80a3b99d9896e32b7c36795f83e9a23e45 | compiler/src/iree/compiler/Dialect/Encoding/IR/EncodingOps.td | dynamic encoding_dims | 1-260 | IREE set_encoding/unset_encoding accept dynamic encoding_dims (e.g. M/N/K); the operation description explicitly states these values are used for runtime layout selection based on problem size. | runtime dimensions | Problem size is sufficient dynamic layout state. | prefix reuse; queue; network/MoE contention | Resolver/default behavior. | Counterexample to fully static layout. | RQ6/RQ9 |  | compiler/src/iree/compiler/Codegen/Common/test/materialize_encoding_vmvx.mlir | https://github.com/iree-org/iree/blob/83265e80a3b99d9896e32b7c36795f83e9a23e45/compiler/src/iree/compiler/Dialect/Encoding/IR/EncodingOps.td#L1-L260 |
| HI06 | IREE | C4 Resource repair | GPU codegen | FACT | CODE | AUDITED-RANGE | iree-org/iree | 83265e80a3b99d9896e32b7c36795f83e9a23e45 | compiler/src/iree/compiler/Codegen/Common/GPU/GPUReuseSharedMemoryAllocs.cpp | GPUReuseSharedMemoryAllocs | 1-240 | IREE's GPU shared-memory reuse pass computes allocation liveness ranges, groups overlapping lifetimes, and reuses allocations only under explicit structural preconditions; a separate GPU pass combines layout transformations. | resource limits; liveness; layout ops | Compiler can repair/materialize legal local memory plans. | whole-serving persistent state | Alter/reject/combine/reuse. | Legality/resource layer for higher-level optimizer. | RQ7/RQ10 |  | compiler/src/iree/compiler/Codegen/Common/GPU/GPUCombineLayoutTransformation.cpp | https://github.com/iree-org/iree/blob/83265e80a3b99d9896e32b7c36795f83e9a23e45/compiler/src/iree/compiler/Codegen/Common/GPU/GPUReuseSharedMemoryAllocs.cpp#L1-L240 |
| HM01 | MLIR | C2 Programmable order | Generic | FACT | DOC | AUDITED-RANGE | llvm/llvm-project | e94698b83a74e6ee40438898b17bc49cbd47fcf5 | mlir/docs/Tutorials/transform/Ch2.md | Transform sequence: tile then fuse; order is important | 220-399 | The Transform dialect tutorial tiles a target then fuses producers one-by-one and explicitly states that the order of these fusions is important because the producer must define a value used in the loop. | transform IR | Compiler/user can express desired order. | No built-in LLM-serving cost chooses the order globally | Transform failure modes. | Expressive transform control plane. | RQ8 |  |  | https://github.com/llvm/llvm-project/blob/e94698b83a74e6ee40438898b17bc49cbd47fcf5/mlir/docs/Tutorials/transform/Ch2.md#L220-L399 |
| HM02 | MLIR | C3 Pattern IR | Generic | FACT | CODE | AUDITED-RANGE | llvm/llvm-project | e94698b83a74e6ee40438898b17bc49cbd47fcf5 | mlir/include/mlir/Dialect/PDL/IR/PDLDialect.td | PDL_Dialect / pdl.pattern example | 1-140 | PDL represents rewrite patterns as IR and carries RewritePattern-like metadata including benefit. | pattern structure; static benefit | Pattern representation and driver prioritize rewrites. | benefit is not live serving-level cost | Pattern miss/continue. | Pattern synthesis infrastructure already exists. | RQ8/RQ10 |  |  | https://github.com/llvm/llvm-project/blob/e94698b83a74e6ee40438898b17bc49cbd47fcf5/mlir/include/mlir/Dialect/PDL/IR/PDLDialect.td#L1-L140 |
| HM03 | MLIR | C2/C4 Layout transform | Generic layout | FACT | CODE | AUDITED-RANGE | llvm/llvm-project | e94698b83a74e6ee40438898b17bc49cbd47fcf5 | mlir/include/mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td | PackTransposeOp | 1120-1355 | structured.pack_transpose can transpose a pack/compute(/unpack) chain; for pack it expects the consuming linalg.generic to be the sole consumer, and unsupported topology/specification yields a silenceable failure. | pack chain; permutations; consumer topology | Restricted local chain makes transposition manageable. | multi-consumer/persistent state/global cost | Silenceable failure. | Topology-bounded layout transform. | RQ7/RQ10 |  |  | https://github.com/llvm/llvm-project/blob/e94698b83a74e6ee40438898b17bc49cbd47fcf5/mlir/include/mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td#L1120-L1355 |
| HM04 | MLIR | C3 Semantic fusion | Generic tensor | FACT | DOC | AUDITED-RANGE | llvm/llvm-project | e94698b83a74e6ee40438898b17bc49cbd47fcf5 | mlir/docs/Tutorials/transform/Ch2.md | structured.tile_using_forall + fuse_into_containing_op | 220-399 | Structured tiling/fusion is expressed over operation/dataflow semantics and producer handles rather than a hard-coded LLM op-name registry; this is used as the semantic-fusion baseline. | indexing maps; iteration semantics | Structured semantics generalize fusion. | target/system profitability | No fusion when transform cannot apply. | Strong semantic baseline. | RQ8 |  |  | https://github.com/llvm/llvm-project/blob/e94698b83a74e6ee40438898b17bc49cbd47fcf5/mlir/docs/Tutorials/transform/Ch2.md#L220-L399 |
| HM05 | MLIR | C1 Static benefit | Generic rewrite | FACT | DOC | AUDITED-RANGE | llvm/llvm-project | e94698b83a74e6ee40438898b17bc49cbd47fcf5 | mlir/docs/PatternRewriter.md | PatternBenefit | 1-80 | Pattern benefit is static after pattern construction, although it may be computed from domain/target information at pattern initialization time. | pattern metadata; target initialization | Static priority is adequate during rewrite driving. | current shape distribution/runtime system state | Driver applies available patterns. | Legality/representation separated from dynamic profitability. | RQ9 |  |  | https://github.com/llvm/llvm-project/blob/e94698b83a74e6ee40438898b17bc49cbd47fcf5/mlir/docs/PatternRewriter.md#L1-L80 |
| HR01 | Triton | C1 Candidate set | SG5 | FACT | CODE | AUDITED-RANGE | triton-lang/triton | 570e5b4dd1e2d1daf785a894d19268fe7b281cb5 | python/tutorials/03-matrix-multiplication.py | get_cuda_autotune_config / get_hip_autotune_config / @triton.autotune | 150-260 | The tutorial declares finite lists of BLOCK_SIZE_M/N/K, GROUP_SIZE_M, stage and warp configurations and autotunes them using M/N/K as the key. | M,N,K key; config list | Declared candidates contain a near-optimal point. | unlisted tiles/swizzles; graph state | Best measured declared config. | Finite design-space approximation. | RQ9 |  |  | https://github.com/triton-lang/triton/blob/570e5b4dd1e2d1daf785a894d19268fe7b281cb5/python/tutorials/03-matrix-multiplication.py#L150-L260 |
| HR02 | Triton | C1/C2 | SG6 | FACT | CODE | AUDITED-RANGE | triton-lang/triton | 570e5b4dd1e2d1daf785a894d19268fe7b281cb5 | python/tutorials/08-grouped-gemm.py | grouped_matmul_kernel @triton.autotune | 1-220 | Grouped GEMM launches a fixed number of CTAs with static device-side scheduling and autotunes a finite set of BLOCK sizes and NUM_SM values keyed by group_size. | group size; tile configs; NUM_SM | A few local scheduling configs cover grouped GEMM regimes. | expert skew drift; A2A payload representation | Autotune among configs. | Local search under irregular groups. | RQ7/RQ9 |  |  | https://github.com/triton-lang/triton/blob/570e5b4dd1e2d1daf785a894d19268fe7b281cb5/python/tutorials/08-grouped-gemm.py#L1-L220 |
| HR03 | Triton | C2 Scheduling | SG5/6 | FACT | CODE+DOC | AUDITED-RANGE | triton-lang/triton | 570e5b4dd1e2d1daf785a894d19268fe7b281cb5 | python/tutorials/03-matrix-multiplication.py | L2 Cache Optimizations / grouped program ordering | 20-150 | The tutorial explains that tile traversal order changes L2 reuse; grouped ordering reduces loaded blocks and can materially outperform simple row-major ordering on the cited hardware. | tile grid; GROUP_SIZE_M; NUM_SMS | Ordering improves cache reuse/occupancy for target shape. | producer/consumer layout and communication overlap | Alternative config/kernel. | Schedule-layout coupling. | RQ8/RQ9 |  |  | https://github.com/triton-lang/triton/blob/570e5b4dd1e2d1daf785a894d19268fe7b281cb5/python/tutorials/03-matrix-multiplication.py#L20-L150 |
| HR04 | Triton | C4 Workaround | SG5 | FACT | CODE | AUDITED-RANGE | triton-lang/triton | 570e5b4dd1e2d1daf785a894d19268fe7b281cb5 | python/tutorials/09-persistent-matmul.py | matmul_kernel_persistent | 1-360 | The persistent-matmul tutorial documents a Blackwell pipelining bug and duplicates counters as an explicit workaround. | architecture/pipeline semantics | Local workaround restores correctness with limited change. | whether different schedule/layout removes overhead | Workaround/alternate path. | Long-tail low-level constraint. | RQ10 |  |  | https://github.com/triton-lang/triton/blob/570e5b4dd1e2d1daf785a894d19268fe7b281cb5/python/tutorials/09-persistent-matmul.py#L1-L360 |
| HR05 | Triton | C3 JIT boundary | Multiple | INFERENCE | CODE+DOC | AUDITED-RANGE | triton-lang/triton | 570e5b4dd1e2d1daf785a894d19268fe7b281cb5 | python/tutorials/03-matrix-multiplication.py | Triton JIT/autotune tutorial scope | 150-260 | The audited APIs optimize a caller-defined kernel/configuration set. The conclusion that whole-LLM graph semantics and persistent-state boundaries remain external is a scope inference, not a literal source predicate. | constexpr/search key | Caller supplies correct search/fusion boundary. | persistent KV; multi-consumer; communication | Caller/framework policy. | Powerful inner optimizer; outer search space external. | RQ7/RQ8/RQ9 |  | python/tutorials/08-grouped-gemm.py@1-220; python/tutorials/09-persistent-matmul.py@1-360 | https://github.com/triton-lang/triton/blob/570e5b4dd1e2d1daf785a894d19268fe7b281cb5/python/tutorials/03-matrix-multiplication.py#L150-L260 |
| HS01 | SGLang | C1 Threshold | SG2 | FACT | DOC | AUDITED-RANGE | sgl-project/sglang | ad28b91faec29cd9e51ecc5b76b06327ed93389e | docs/docs/advanced_features/attention_backend.mdx | Page-size support / prefix-cache tradeoff | 220-360 | The attention-backend guide states that prefix reuse occurs only for complete pages, page_size=1 maximizes reuse, larger pages generally improve attention-kernel performance, and backends have native page-size constraints. | page size; backend | One launch-time page size balances reuse and kernel speed. | actual prefix distribution; fragmentation; transfer | Emulation for some backends or supported native size. | Cache granularity trade-off. | RQ2/RQ3/RQ9 |  |  | https://github.com/sgl-project/sglang/blob/ad28b91faec29cd9e51ecc5b76b06327ed93389e/docs/docs/advanced_features/attention_backend.mdx#L220-L360 |
| HS02 | SGLang | C1 Threshold | SG7 | FACT | CODE | AUDITED-RANGE | sgl-project/sglang | ad28b91faec29cd9e51ecc5b76b06327ed93389e | python/sglang/srt/arg_groups/fields/exec_.py | ExecOverlap.tbo_token_distribution_threshold | 650-900 | The TBO token-distribution threshold is defined as 0.48 and controls whether overlap uses two-batch-overlap or two-chunk-overlap. | token distribution ratio | One ratio predicts overlap mode. | expert skew; network latency; per-stage timings | Alternative overlap behavior. | Scheduling crossover surface. | RQ4/RQ9 |  |  | https://github.com/sgl-project/sglang/blob/ad28b91faec29cd9e51ecc5b76b06327ed93389e/python/sglang/srt/arg_groups/fields/exec_.py#L650-L900 |
| HS03 | SGLang | C2 Routing | SG2 | FACT | CODE | AUDITED-RANGE | sgl-project/sglang | ad28b91faec29cd9e51ecc5b76b06327ed93389e | python/sglang/srt/layers/attention/hybrid_attn_backend.py | HybridAttnBackend._select_backend | 1-180 | HybridAttnBackend explicitly routes decode/idle to the decode backend, prefill/extend to the prefill backend, and target_verify according to speculative_attention_mode. | semantic phase; spec mode | Phase identity captures most backend crossover. | within-phase batch/KV/GQA; shared-layout conversion | Inherit global backend / reject unsupported combo. | Semantic routing vs cost-based multi-consumer selection. | RQ1/RQ5/RQ7 |  |  | https://github.com/sgl-project/sglang/blob/ad28b91faec29cd9e51ecc5b76b06327ed93389e/python/sglang/srt/layers/attention/hybrid_attn_backend.py#L1-L180 |
| HS04 | SGLang | C3 Registry | SG5 | FACT | CODE | AUDITED-RANGE | sgl-project/sglang | ad28b91faec29cd9e51ecc5b76b06327ed93389e | python/sglang/srt/arg_groups/fields/exec_.py | GEMM runner backend arguments / auto policy | 1-260 | Execution arguments expose multiple FP8/FP4 GEMM backends and an auto mode whose documented choice depends on hardware/backend availability. | SM; dtype; availability | Hardware/backend family is a strong performance proxy. | live M/N/K; downstream layout/quant format | Compatible fallback, often Triton; explicit invalid choices error. | Piecewise kernel dispatch. | RQ7/RQ9 |  |  | https://github.com/sgl-project/sglang/blob/ad28b91faec29cd9e51ecc5b76b06327ed93389e/python/sglang/srt/arg_groups/fields/exec_.py#L1-L260 |
| HS05 | SGLang | C3 Registry | SG6 | FACT | CODE | AUDITED-RANGE | sgl-project/sglang | ad28b91faec29cd9e51ecc5b76b06327ed93389e | python/sglang/srt/layers/moe/utils.py | MoeA2ABackend / MoeRunnerBackend and compatibility utilities | 1-520 | MoE communication and compute backends are represented as independent backend families with compatibility/configuration logic. | parallel config; model; hardware; quant | Component-wise selection works if compatibility is checked. | token ordering; A2A payload layout; expert GEMM packing | Validator rejection / other backend. | Network representation + expert compute co-design. | RQ4/RQ7/RQ10 |  | python/sglang/srt/arg_groups/fields/exec_.py@650-900 | https://github.com/sgl-project/sglang/blob/ad28b91faec29cd9e51ecc5b76b06327ed93389e/python/sglang/srt/layers/moe/utils.py#L1-L520 |
| HS06 | SGLang | C4 Repair | SG6 | FACT | CODE | AUDITED-RANGE | sgl-project/sglang | ad28b91faec29cd9e51ecc5b76b06327ed93389e | python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py | FlashinferDispatcher.dispatch: NVFP4 scale interleave | 360-560 | For NVFP4 + FlashInfer CUTLASS MoE, received scale tensors are explicitly passed through nvfp4_block_scale_interleave; a TODO says to fuse this interleave into CUTLASS MoE when supported. | runner backend; scale layout | Explicit transformation is acceptable until producer/consumer fusion exists. | amortization; direct producer-native scale layout | Materialize scale interleave. | Concrete layout-repair edge. | RQ3/RQ7 |  |  | https://github.com/sgl-project/sglang/blob/ad28b91faec29cd9e51ecc5b76b06327ed93389e/python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py#L360-L560 |
| HT01 | TensorRT-LLM | C2 Ordering | Whole graph | FACT | CODE | AUDITED-RANGE | NVIDIA/TensorRT-LLM | 7c79c15fa19c4a3b4e6bae15ea741541557f6c92 | tensorrt_llm/_torch/auto_deploy/transform/interface.py | Stages enum | 1-180 | AutoDeploy declares an ordered stage sequence including export, post-export, pattern matching, sharding, weight loading, post-load fusion, cache initialization and compilation. | stage | One global order is predictable and legal. | feedback between sharding/fusion/cache/layout | Later stage consumes earlier committed graph. | Pass-order optimization. | RQ8 |  |  | https://github.com/NVIDIA/TensorRT-LLM/blob/7c79c15fa19c4a3b4e6bae15ea741541557f6c92/tensorrt_llm/_torch/auto_deploy/transform/interface.py#L1-L180 |
| HT02 | TensorRT-LLM | C2/C3 | SG5 | FACT | CODE | AUDITED-RANGE | NVIDIA/TensorRT-LLM | 7c79c15fa19c4a3b4e6bae15ea741541557f6c92 | tensorrt_llm/_torch/auto_deploy/transform/library/fuse_silu_mul.py | FuseSiluMul | 1-220 | FuseSiluMul is a post-load transform intended to run after GEMM fusion; a suitable sole FP8-linear consumer enables further quantization fusion. | prior rewrite; consumer cardinality/type | GEMM fusion should precede activation/quant fusion. | joint gate/up packing + activation layout + down-GEMM format | Skip extra fusion if prerequisite absent. | Producer epilogue/consumer contract + pass order. | RQ7/RQ8 |  |  | https://github.com/NVIDIA/TensorRT-LLM/blob/7c79c15fa19c4a3b4e6bae15ea741541557f6c92/tensorrt_llm/_torch/auto_deploy/transform/library/fuse_silu_mul.py#L1-L220 |
| HT03 | TensorRT-LLM | C4 Repair | SG4 | FACT | CODE | AUDITED-RANGE | NVIDIA/TensorRT-LLM | 7c79c15fa19c4a3b4e6bae15ea741541557f6c92 | tensorrt_llm/_torch/auto_deploy/transform/library/fused_add_rms_norm.py | FuseAddRMSNorm | 1-260 | The transform explicitly uses direct FX graph manipulation rather than the ordinary pattern matcher so that intermediate add/RMSNorm values with multiple users are handled correctly. | use-def topology | Known multi-user topology deserves a special repair. | general clone/recompute/layout alternatives | Bespoke transform for known case. | Multi-consumer repair space. | RQ7/RQ10 |  |  | https://github.com/NVIDIA/TensorRT-LLM/blob/7c79c15fa19c4a3b4e6bae15ea741541557f6c92/tensorrt_llm/_torch/auto_deploy/transform/library/fused_add_rms_norm.py#L1-L260 |
| HT04 | TensorRT-LLM | C3 Semantic discovery | Elementwise | FACT | CODE | AUDITED-RANGE | NVIDIA/TensorRT-LLM | 7c79c15fa19c4a3b4e6bae15ea741541557f6c92 | tensorrt_llm/_torch/auto_deploy/transform/library/mlir_elementwise_fusion.py | MLIR elementwise fusion pipeline | 1-130 | The transform converts/decomposes elementwise regions, discovers maximal fusible subgraphs, generates Triton and replaces the original region; unsupported conversion can be skipped. | semantic decomposition; graph region | Semantic discovery can exceed exact syntax patterns. | cross-op layout/resource cost; persistent state | Skip gracefully if unavailable / configured skip_on_error. | Semantic subgraph discovery + codegen. | RQ8/RQ10 |  |  | https://github.com/NVIDIA/TensorRT-LLM/blob/7c79c15fa19c4a3b4e6bae15ea741541557f6c92/tensorrt_llm/_torch/auto_deploy/transform/library/mlir_elementwise_fusion.py#L1-L130 |
| HT05 | TensorRT-LLM | C4 Assert/repair | SG6 | FACT | CODE | AUDITED-RANGE | NVIDIA/TensorRT-LLM | 7c79c15fa19c4a3b4e6bae15ea741541557f6c92 | tensorrt_llm/_torch/auto_deploy/transform/library/fused_moe.py | Fused-MoE graph/weight restructuring transforms | 1-360 | AutoDeploy contains explicit fused-MoE restructuring logic that converts between stacked/backend-specific weight/graph representations and enforces backend/style support constraints. | per-expert scales | Uniform scale matches preferred backend representation. | alternative scale grouping; accuracy/performance frontier | Assert or coarsen scale. | Repair with accuracy constraint. | RQ7/RQ10 | v2's specific claim about equal per-expert FP8 input scales and max-scale repair was not re-located at this pinned HEAD; it is replaced by the currently verified fused-MoE representation-restructuring and support-boundary evidence. |  | https://github.com/NVIDIA/TensorRT-LLM/blob/7c79c15fa19c4a3b4e6bae15ea741541557f6c92/tensorrt_llm/_torch/auto_deploy/transform/library/fused_moe.py#L1-L360 |
| HT06 | TensorRT-LLM | C1 Local heuristic | SG2 | FACT | CODE | AUDITED-RANGE | NVIDIA/TensorRT-LLM | 7c79c15fa19c4a3b4e6bae15ea741541557f6c92 | cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunner.cpp | DecoderXQARunner::mayHavePerfGain | 150-270 | The XQA heuristic computes work from num_kv_heads * batch_size * multiBlockCount(history) and compares it against an SM-count-derived threshold; force/speculative/MLA cases can bypass the heuristic. | batch; Hkv; history; SM | Available parallel work predicts local crossover. | persistent layout; MoE/network contention | Masked MHA/single-block alternative. | Existing live local cost heuristic. | RQ1/RQ9 |  |  | https://github.com/NVIDIA/TensorRT-LLM/blob/7c79c15fa19c4a3b4e6bae15ea741541557f6c92/cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunner.cpp#L150-L270 |
| HV01 | vLLM | C1 Threshold | SG1 | FACT | DOC | AUDITED-RANGE | vllm-project/vllm | 9ca6dbba71329a7b08711e2acfa17b31e542553c | docs/design/fusions.md | RoPE + KV Cache Update fusion | 170-195 | The design doc states that RoPE+KV-cache update fusion is ROCm/AITER-specific and, by default, applies when num_tokens <= 256; the maximum is configurable. | num_tokens; platform/backend | One token threshold approximates the fusion crossover. | head geometry; KV layout; concurrent load; reuse | Do not fuse above threshold / unsupported platform. | Fusion benefit vs resource/implementation penalty. | RQ7/RQ9 |  |  | https://github.com/vllm-project/vllm/blob/9ca6dbba71329a7b08711e2acfa17b31e542553c/docs/design/fusions.md#L170-L195 |
| HV02 | vLLM | C1 Threshold | SG4/SG7 | FACT | DOC | AUDITED-RANGE | vllm-project/vllm | 9ca6dbba71329a7b08711e2acfa17b31e542553c | docs/design/fusions.md | AllReduce + RMSNorm fusion | 100-125 | The design doc gives a hardware-dependent maximum tensor-size regime for fused AllReduce+RMSNorm and a 64 MB example for TP=2 on SM90/SM100. | tensor bytes; TP; SM | Tensor size approximates collective/fusion crossover. | network contention; downstream quant/GEMM layout; MoE traffic | Unfused collective+norm. | Communication+memory fusion crossover. | RQ7/RQ9 |  |  | https://github.com/vllm-project/vllm/blob/9ca6dbba71329a7b08711e2acfa17b31e542553c/docs/design/fusions.md#L100-L125 |
| HV03 | vLLM | C2 Ordering | SG3/SG7 | FACT | DOC | AUDITED-RANGE | vllm-project/vllm | 9ca6dbba71329a7b08711e2acfa17b31e542553c | docs/design/fusions.md | Sequence Parallelism / AsyncTP | 160-235 | AsyncTP depends on the Sequence Parallelism rewrite; the doc states AsyncTP is a no-op when SP has not been applied. | pass state; tokens; hidden size; platform | A fixed staged rewrite is adequate. | alternative rewrite order; producer layout; communication topology | No AsyncTP when prerequisite not met. | Non-commutative sharding/fusion/layout search. | RQ8 |  |  | https://github.com/vllm-project/vllm/blob/9ca6dbba71329a7b08711e2acfa17b31e542553c/docs/design/fusions.md#L160-L235 |
| HV04 | vLLM | C3 Pattern | SG1/3/4/5 | FACT | DOC | AUDITED-RANGE | vllm-project/vllm | 9ca6dbba71329a7b08711e2acfa17b31e542553c | docs/design/fusions.md | Fusion quick-reference matrix | 20-90 | The fusion design enumerates concrete graph patterns including AllReduce->RMSNorm, Attention->Quant, QK-Norm->RoPE, RoPE->KV write, RMSNorm->Quant and SiLU+Mul->Quant. | graph syntax/ops; backend; dtype; GPU | Finite pattern set covers common profitable cases. | semantic variants; multi-user graphs; alternate output contracts | General unfused path. | Pattern coverage + producer/consumer contract. | RQ7/RQ8/RQ10 |  |  | https://github.com/vllm-project/vllm/blob/9ca6dbba71329a7b08711e2acfa17b31e542553c/docs/design/fusions.md#L20-L90 |
| HV05 | vLLM | C3 Registry | SG6 | FACT | DOC | AUDITED-RANGE | vllm-project/vllm | 9ca6dbba71329a7b08711e2acfa17b31e542553c | docs/design/fused_moe_modular_kernel.md | Modular MoE activation formats and composition | 1-180 | The modular MoE design distinguishes activation formats and states that input activation format is determined by the All2All dispatch; it motivates modular composition because implementation combinations become intractable. | A2A; activation format; quantization; architecture | Compatibility families tame combinatorial composition. | joint network payload + expert GEMM layout cost | Pick compatible family or reject. | Typed component compatibility graph. | RQ4/RQ7 |  |  | https://github.com/vllm-project/vllm/blob/9ca6dbba71329a7b08711e2acfa17b31e542553c/docs/design/fused_moe_modular_kernel.md#L1-L180 |
| HV06 | vLLM | C1/C4 | SG6/7 | FACT | CODE | AUDITED-RANGE | vllm-project/vllm | 9ca6dbba71329a7b08711e2acfa17b31e542553c | vllm/config/parallel.py | ParallelConfig DBO thresholds | 220-340 | ParallelConfig defines dbo_decode_token_threshold=32 and dbo_prefill_token_threshold=512; above-threshold batches can be microbatched when DBO is enabled. | token count; phase; EP/A2A | Token count predicts overlap amortization. | expert skew; network state; attention contention | Single batch / no DBO. | Piecewise compute-communication overlap policy. | RQ4/RQ9/RQ10 |  |  | https://github.com/vllm-project/vllm/blob/9ca6dbba71329a7b08711e2acfa17b31e542553c/vllm/config/parallel.py#L220-L340 |
| HX01 | TVM Relax/TIR | C2 Pipeline | Graph compiler | FACT | CODE | AUDITED-RANGE | apache/tvm | 8312a17f8734ddfd56e5f3977cd5df25b83ec49f | python/tvm/relax/backend/cuda/pipeline.py | CUDA Relax backend pipeline | 1-180 | The CUDA Relax backend pipeline explicitly runs LegalizeOps -> AnnotateTIROpPattern -> FoldConstant -> FuseOps -> FuseTIR before default scheduling and later dataflow/finalization passes. | pass order/pattern annotations | Staged lowering is tractable and predictable. | joint order with layout/backend decisions | Later pass consumes earlier graph. | Non-commutative pipeline. | RQ8 |  |  | https://github.com/apache/tvm/blob/8312a17f8734ddfd56e5f3977cd5df25b83ec49f/python/tvm/relax/backend/cuda/pipeline.py#L1-L180 |
| HX02 | TVM Relax | C3 Semantic fusion | Graph fusion | FACT | CODE | AUDITED-RANGE | apache/tvm | 8312a17f8734ddfd56e5f3977cd5df25b83ec49f | src/relax/transform/fuse_ops.cc | FuseOps post-dominator algorithm | 1-180 | FuseOps documents and implements a post-dominator-based graph fusion algorithm for diamond dataflow: build DAG, build post-dominator tree, CheckPath, then CommitFuse; the source also has kMaxFusedOps=256. | pattern kinds; dataflow graph | Structured graph analysis generalizes fusion. | target layout/resource cost not directly global objective | Leave boundaries unfused. | Counterexample to simplistic greedy fusion claim. | RQ8 |  |  | https://github.com/apache/tvm/blob/8312a17f8734ddfd56e5f3977cd5df25b83ec49f/src/relax/transform/fuse_ops.cc#L1-L180 |
| HX03 | TVM Relax | C3 Pattern registry | Backend dispatch | FACT | DOC | AUDITED-RANGE | apache/tvm | 8312a17f8734ddfd56e5f3977cd5df25b83ec49f | docs/arch/fusion.rst | FuseOpsByPattern / FusionPattern | 1-280 | TVM documents two complementary mechanisms: automatic FuseOps/FuseTIR using OpPatternKind + post-dominator analysis, and FuseOpsByPattern for user/backend patterns with optional context-aware checks. | pattern priority; structural match; checks | Backend-specific pattern library encodes profitable offload. | unseen semantic variants and joint layout | Next pattern/general compiler. | Registry coverage/priority. | RQ8/RQ10 |  |  | https://github.com/apache/tvm/blob/8312a17f8734ddfd56e5f3977cd5df25b83ec49f/docs/arch/fusion.rst#L1-L280 |
| HX04 | TVM TIR | C1/C4 Search | Kernel schedule | FACT | DOC | AUDITED-RANGE | apache/tvm | 8312a17f8734ddfd56e5f3977cd5df25b83ec49f | docs/deep_dive/tensor_ir/tutorials/meta_schedule.py | MetaSchedule cost model / runner / database summary | 250-380 | MetaSchedule documentation states that it searches schedule design spaces and measures on real hardware; the snapshot exposes cost-model choices (including XGBoost/MLP/random), runners and persistent tuning databases. | TIR features; candidates; measured time | Measurement-guided search finds strong per-task schedules. | whole-serving temporal/persistent state | Best database record; DLight as zero-search alternative. | Cost-model/search already exists. | RQ9 |  | tests/python/s_tir/meta_schedule/test_meta_schedule_cost_model.py; tests/python/s_tir/meta_schedule/testing/tune_te.py | https://github.com/apache/tvm/blob/8312a17f8734ddfd56e5f3977cd5df25b83ec49f/docs/deep_dive/tensor_ir/tutorials/meta_schedule.py#L250-L380 |
| HX05 | TVM Relax | C2 Layout op | Graph layout | FACT | CODE | AUDITED-RANGE | apache/tvm | 8312a17f8734ddfd56e5f3977cd5df25b83ec49f | python/tvm/relax/op/manipulate.py | layout_transform | 90-180 | Relax layout_transform is an explicit graph operator taking an IndexMap and optional padding and materializing a transformed tensor. | index map; tensor shape | Explicit conversion is a valid graph action. | producer/consumer could avoid conversion jointly | Keep/lower/fuse transform. | Explicit conversion baseline. | RQ3/RQ7 |  |  | https://github.com/apache/tvm/blob/8312a17f8734ddfd56e5f3977cd5df25b83ec49f/python/tvm/relax/op/manipulate.py#L90-L180 |
| HX06 | TVM MetaSchedule | C3 Transfer/reuse | Tuning DB | FACT | DOC | AUDITED-RANGE | apache/tvm | 8312a17f8734ddfd56e5f3977cd5df25b83ec49f | docs/deep_dive/tensor_ir/tutorials/meta_schedule.py | Cross-model database reuse / anchor-block equality | 200-270 | MetaSchedule documents structural and anchor-block module equality; anchor-block matching ignores surrounding context and enables tuning-record sharing across fused operators with the same core computation but different fusion boundaries. | anchor trace; target | Schedule knowledge transfers across related tasks. | LLM state/hardware/workload drift beyond anchor similarity | Tune new task if no record. | Existing transfer baseline. | RQ9 |  |  | https://github.com/apache/tvm/blob/8312a17f8734ddfd56e5f3977cd5df25b83ec49f/docs/deep_dive/tensor_ir/tutorials/meta_schedule.py#L200-L270 |

---

# 4. Appendix-ready compact ledger

| id | framework | semantic_class | subgraph | provenance_kind | path | symbol_or_section | audited_range | rq | immutable_url |
|---|---|---|---|---|---|---|---|---|---|
| HC01 | CUTLASS/CuTe | C1 Analytical heuristic | SG5 | DOC | media/docs/cpp/heuristics.md | GEMM Heuristics overview / coverage | 1-140 | RQ9 | https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/media/docs/cpp/heuristics.md#L1-L140 |
| HC02 | CUTLASS/CuTe | C1 Layout heuristic | SG5 | CODE | python/CuTeDSL/cutlass/utils/blackwell_helpers.py | get_smem_layout_atom_ab / SMEM store helpers | 520-760 | RQ7/RQ9 | https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/python/CuTeDSL/cutlass/utils/blackwell_helpers.py#L520-L760 |
| HC03 | CUTLASS/CuTe | C3 Template family | SG5/6 | DOC | media/docs/cpp/gemm_api_3x.md | CUTLASS 3.x GEMM components / kernel assembly | 1-180 | RQ7/RQ8 | https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/media/docs/cpp/gemm_api_3x.md#L1-L180 |
| HC04 | CUTLASS/CuTe | C2/C4 Algebraic repair | SG5 | DOC | media/docs/cpp/gemm_api.md | Efficient Epilogue | 430-620 | RQ7/RQ10 | https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/media/docs/cpp/gemm_api.md#L430-L620 |
| HC05 | CUTLASS/CuTe | C4 Legality | SG5/6 | CODE | python/CuTeDSL/cutlass/utils/blackwell_helpers.py | TMA alignment / architecture legality checks | 1-220 | RQ8/RQ10 | https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/python/CuTeDSL/cutlass/utils/blackwell_helpers.py#L1-L220 |
| HF01 | FlashInfer | C1 Heuristic | SG2 | CODE | flashinfer/decode.py | single_decode_with_kv_cache(use_tensor_cores) | 520-760 | RQ1/RQ9 | https://github.com/flashinfer-ai/flashinfer/blob/c039288594d3f97404166e1f8111f4c5c88143ad/flashinfer/decode.py#L520-L760 |
| HF02 | FlashInfer | C4 Conversion | SG2 | CODE | flashinfer/decode.py | trtllm_batch_decode_with_kv_cache | 3100-3420 | RQ3/RQ7/RQ10 | https://github.com/flashinfer-ai/flashinfer/blob/c039288594d3f97404166e1f8111f4c5c88143ad/flashinfer/decode.py#L3100-L3420 |
| HF03 | FlashInfer | C3 Kernel family | SG5/6 | CODE | flashinfer/grouped_mm/core.py | grouped_mm_* public facade | 1-260 | RQ7/RQ9 | https://github.com/flashinfer-ai/flashinfer/blob/c039288594d3f97404166e1f8111f4c5c88143ad/flashinfer/grouped_mm/core.py#L1-L260 |
| HF04 | FlashInfer | C1 Heuristic | SG6 | CODE | flashinfer/grouped_mm/core.py | grouped_mm_bf16 / grouped_mm_fp8 tactic | 1-260 | RQ7/RQ9 | https://github.com/flashinfer-ai/flashinfer/blob/c039288594d3f97404166e1f8111f4c5c88143ad/flashinfer/grouped_mm/core.py#L1-L260 |
| HF05 | FlashInfer | C4 Planning | Runtime | CODE | flashinfer/decode.py | BatchDecodeWithPagedKVCacheWrapper.__init__ | 760-1120 | RQ6/RQ10 | https://github.com/flashinfer-ai/flashinfer/blob/c039288594d3f97404166e1f8111f4c5c88143ad/flashinfer/decode.py#L760-L1120 |
| HI01 | IREE | C1 Tuning | GPU dispatch | DOC | docs/website/docs/reference/tuning.md | Knobs in dispatches / generic tuning workflow | 1-260 | RQ9 | https://github.com/iree-org/iree/blob/83265e80a3b99d9896e32b7c36795f83e9a23e45/docs/website/docs/reference/tuning.md#L1-L260 |
| HI02 | IREE | C4 Formal legality | GPU tuning | DOC | docs/website/docs/reference/tuning.md | Constraint generation | 1-260 | RQ8/RQ10 | https://github.com/iree-org/iree/blob/83265e80a3b99d9896e32b7c36795f83e9a23e45/docs/website/docs/reference/tuning.md#L1-L260 |
| HI03 | IREE | C3 Semantic match | GPU tuning | DOC | docs/website/docs/reference/tuning.md | Tuning specs / semantic match operations | 260-520 | RQ8/RQ9 | https://github.com/iree-org/iree/blob/83265e80a3b99d9896e32b7c36795f83e9a23e45/docs/website/docs/reference/tuning.md#L260-L520 |
| HI04 | IREE | C2 Layout propagation | Graph→GPU | CODE | compiler/src/iree/compiler/Dialect/Encoding/IR/EncodingOps.td | set_encoding / unset_encoding | 1-260 | RQ7 | https://github.com/iree-org/iree/blob/83265e80a3b99d9896e32b7c36795f83e9a23e45/compiler/src/iree/compiler/Dialect/Encoding/IR/EncodingOps.td#L1-L260 |
| HI05 | IREE | C1 Dynamic encoding | Generic layout | CODE | compiler/src/iree/compiler/Dialect/Encoding/IR/EncodingOps.td | dynamic encoding_dims | 1-260 | RQ6/RQ9 | https://github.com/iree-org/iree/blob/83265e80a3b99d9896e32b7c36795f83e9a23e45/compiler/src/iree/compiler/Dialect/Encoding/IR/EncodingOps.td#L1-L260 |
| HI06 | IREE | C4 Resource repair | GPU codegen | CODE | compiler/src/iree/compiler/Codegen/Common/GPU/GPUReuseSharedMemoryAllocs.cpp | GPUReuseSharedMemoryAllocs | 1-240 | RQ7/RQ10 | https://github.com/iree-org/iree/blob/83265e80a3b99d9896e32b7c36795f83e9a23e45/compiler/src/iree/compiler/Codegen/Common/GPU/GPUReuseSharedMemoryAllocs.cpp#L1-L240 |
| HM01 | MLIR | C2 Programmable order | Generic | DOC | mlir/docs/Tutorials/transform/Ch2.md | Transform sequence: tile then fuse; order is important | 220-399 | RQ8 | https://github.com/llvm/llvm-project/blob/e94698b83a74e6ee40438898b17bc49cbd47fcf5/mlir/docs/Tutorials/transform/Ch2.md#L220-L399 |
| HM02 | MLIR | C3 Pattern IR | Generic | CODE | mlir/include/mlir/Dialect/PDL/IR/PDLDialect.td | PDL_Dialect / pdl.pattern example | 1-140 | RQ8/RQ10 | https://github.com/llvm/llvm-project/blob/e94698b83a74e6ee40438898b17bc49cbd47fcf5/mlir/include/mlir/Dialect/PDL/IR/PDLDialect.td#L1-L140 |
| HM03 | MLIR | C2/C4 Layout transform | Generic layout | CODE | mlir/include/mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td | PackTransposeOp | 1120-1355 | RQ7/RQ10 | https://github.com/llvm/llvm-project/blob/e94698b83a74e6ee40438898b17bc49cbd47fcf5/mlir/include/mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td#L1120-L1355 |
| HM04 | MLIR | C3 Semantic fusion | Generic tensor | DOC | mlir/docs/Tutorials/transform/Ch2.md | structured.tile_using_forall + fuse_into_containing_op | 220-399 | RQ8 | https://github.com/llvm/llvm-project/blob/e94698b83a74e6ee40438898b17bc49cbd47fcf5/mlir/docs/Tutorials/transform/Ch2.md#L220-L399 |
| HM05 | MLIR | C1 Static benefit | Generic rewrite | DOC | mlir/docs/PatternRewriter.md | PatternBenefit | 1-80 | RQ9 | https://github.com/llvm/llvm-project/blob/e94698b83a74e6ee40438898b17bc49cbd47fcf5/mlir/docs/PatternRewriter.md#L1-L80 |
| HR01 | Triton | C1 Candidate set | SG5 | CODE | python/tutorials/03-matrix-multiplication.py | get_cuda_autotune_config / get_hip_autotune_config / @triton.autotune | 150-260 | RQ9 | https://github.com/triton-lang/triton/blob/570e5b4dd1e2d1daf785a894d19268fe7b281cb5/python/tutorials/03-matrix-multiplication.py#L150-L260 |
| HR02 | Triton | C1/C2 | SG6 | CODE | python/tutorials/08-grouped-gemm.py | grouped_matmul_kernel @triton.autotune | 1-220 | RQ7/RQ9 | https://github.com/triton-lang/triton/blob/570e5b4dd1e2d1daf785a894d19268fe7b281cb5/python/tutorials/08-grouped-gemm.py#L1-L220 |
| HR03 | Triton | C2 Scheduling | SG5/6 | CODE+DOC | python/tutorials/03-matrix-multiplication.py | L2 Cache Optimizations / grouped program ordering | 20-150 | RQ8/RQ9 | https://github.com/triton-lang/triton/blob/570e5b4dd1e2d1daf785a894d19268fe7b281cb5/python/tutorials/03-matrix-multiplication.py#L20-L150 |
| HR04 | Triton | C4 Workaround | SG5 | CODE | python/tutorials/09-persistent-matmul.py | matmul_kernel_persistent | 1-360 | RQ10 | https://github.com/triton-lang/triton/blob/570e5b4dd1e2d1daf785a894d19268fe7b281cb5/python/tutorials/09-persistent-matmul.py#L1-L360 |
| HR05 | Triton | C3 JIT boundary | Multiple | CODE+DOC | python/tutorials/03-matrix-multiplication.py | Triton JIT/autotune tutorial scope | 150-260 | RQ7/RQ8/RQ9 | https://github.com/triton-lang/triton/blob/570e5b4dd1e2d1daf785a894d19268fe7b281cb5/python/tutorials/03-matrix-multiplication.py#L150-L260 |
| HS01 | SGLang | C1 Threshold | SG2 | DOC | docs/docs/advanced_features/attention_backend.mdx | Page-size support / prefix-cache tradeoff | 220-360 | RQ2/RQ3/RQ9 | https://github.com/sgl-project/sglang/blob/ad28b91faec29cd9e51ecc5b76b06327ed93389e/docs/docs/advanced_features/attention_backend.mdx#L220-L360 |
| HS02 | SGLang | C1 Threshold | SG7 | CODE | python/sglang/srt/arg_groups/fields/exec_.py | ExecOverlap.tbo_token_distribution_threshold | 650-900 | RQ4/RQ9 | https://github.com/sgl-project/sglang/blob/ad28b91faec29cd9e51ecc5b76b06327ed93389e/python/sglang/srt/arg_groups/fields/exec_.py#L650-L900 |
| HS03 | SGLang | C2 Routing | SG2 | CODE | python/sglang/srt/layers/attention/hybrid_attn_backend.py | HybridAttnBackend._select_backend | 1-180 | RQ1/RQ5/RQ7 | https://github.com/sgl-project/sglang/blob/ad28b91faec29cd9e51ecc5b76b06327ed93389e/python/sglang/srt/layers/attention/hybrid_attn_backend.py#L1-L180 |
| HS04 | SGLang | C3 Registry | SG5 | CODE | python/sglang/srt/arg_groups/fields/exec_.py | GEMM runner backend arguments / auto policy | 1-260 | RQ7/RQ9 | https://github.com/sgl-project/sglang/blob/ad28b91faec29cd9e51ecc5b76b06327ed93389e/python/sglang/srt/arg_groups/fields/exec_.py#L1-L260 |
| HS05 | SGLang | C3 Registry | SG6 | CODE | python/sglang/srt/layers/moe/utils.py | MoeA2ABackend / MoeRunnerBackend and compatibility utilities | 1-520 | RQ4/RQ7/RQ10 | https://github.com/sgl-project/sglang/blob/ad28b91faec29cd9e51ecc5b76b06327ed93389e/python/sglang/srt/layers/moe/utils.py#L1-L520 |
| HS06 | SGLang | C4 Repair | SG6 | CODE | python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py | FlashinferDispatcher.dispatch: NVFP4 scale interleave | 360-560 | RQ3/RQ7 | https://github.com/sgl-project/sglang/blob/ad28b91faec29cd9e51ecc5b76b06327ed93389e/python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py#L360-L560 |
| HT01 | TensorRT-LLM | C2 Ordering | Whole graph | CODE | tensorrt_llm/_torch/auto_deploy/transform/interface.py | Stages enum | 1-180 | RQ8 | https://github.com/NVIDIA/TensorRT-LLM/blob/7c79c15fa19c4a3b4e6bae15ea741541557f6c92/tensorrt_llm/_torch/auto_deploy/transform/interface.py#L1-L180 |
| HT02 | TensorRT-LLM | C2/C3 | SG5 | CODE | tensorrt_llm/_torch/auto_deploy/transform/library/fuse_silu_mul.py | FuseSiluMul | 1-220 | RQ7/RQ8 | https://github.com/NVIDIA/TensorRT-LLM/blob/7c79c15fa19c4a3b4e6bae15ea741541557f6c92/tensorrt_llm/_torch/auto_deploy/transform/library/fuse_silu_mul.py#L1-L220 |
| HT03 | TensorRT-LLM | C4 Repair | SG4 | CODE | tensorrt_llm/_torch/auto_deploy/transform/library/fused_add_rms_norm.py | FuseAddRMSNorm | 1-260 | RQ7/RQ10 | https://github.com/NVIDIA/TensorRT-LLM/blob/7c79c15fa19c4a3b4e6bae15ea741541557f6c92/tensorrt_llm/_torch/auto_deploy/transform/library/fused_add_rms_norm.py#L1-L260 |
| HT04 | TensorRT-LLM | C3 Semantic discovery | Elementwise | CODE | tensorrt_llm/_torch/auto_deploy/transform/library/mlir_elementwise_fusion.py | MLIR elementwise fusion pipeline | 1-130 | RQ8/RQ10 | https://github.com/NVIDIA/TensorRT-LLM/blob/7c79c15fa19c4a3b4e6bae15ea741541557f6c92/tensorrt_llm/_torch/auto_deploy/transform/library/mlir_elementwise_fusion.py#L1-L130 |
| HT05 | TensorRT-LLM | C4 Assert/repair | SG6 | CODE | tensorrt_llm/_torch/auto_deploy/transform/library/fused_moe.py | Fused-MoE graph/weight restructuring transforms | 1-360 | RQ7/RQ10 | https://github.com/NVIDIA/TensorRT-LLM/blob/7c79c15fa19c4a3b4e6bae15ea741541557f6c92/tensorrt_llm/_torch/auto_deploy/transform/library/fused_moe.py#L1-L360 |
| HT06 | TensorRT-LLM | C1 Local heuristic | SG2 | CODE | cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunner.cpp | DecoderXQARunner::mayHavePerfGain | 150-270 | RQ1/RQ9 | https://github.com/NVIDIA/TensorRT-LLM/blob/7c79c15fa19c4a3b4e6bae15ea741541557f6c92/cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunner.cpp#L150-L270 |
| HV01 | vLLM | C1 Threshold | SG1 | DOC | docs/design/fusions.md | RoPE + KV Cache Update fusion | 170-195 | RQ7/RQ9 | https://github.com/vllm-project/vllm/blob/9ca6dbba71329a7b08711e2acfa17b31e542553c/docs/design/fusions.md#L170-L195 |
| HV02 | vLLM | C1 Threshold | SG4/SG7 | DOC | docs/design/fusions.md | AllReduce + RMSNorm fusion | 100-125 | RQ7/RQ9 | https://github.com/vllm-project/vllm/blob/9ca6dbba71329a7b08711e2acfa17b31e542553c/docs/design/fusions.md#L100-L125 |
| HV03 | vLLM | C2 Ordering | SG3/SG7 | DOC | docs/design/fusions.md | Sequence Parallelism / AsyncTP | 160-235 | RQ8 | https://github.com/vllm-project/vllm/blob/9ca6dbba71329a7b08711e2acfa17b31e542553c/docs/design/fusions.md#L160-L235 |
| HV04 | vLLM | C3 Pattern | SG1/3/4/5 | DOC | docs/design/fusions.md | Fusion quick-reference matrix | 20-90 | RQ7/RQ8/RQ10 | https://github.com/vllm-project/vllm/blob/9ca6dbba71329a7b08711e2acfa17b31e542553c/docs/design/fusions.md#L20-L90 |
| HV05 | vLLM | C3 Registry | SG6 | DOC | docs/design/fused_moe_modular_kernel.md | Modular MoE activation formats and composition | 1-180 | RQ4/RQ7 | https://github.com/vllm-project/vllm/blob/9ca6dbba71329a7b08711e2acfa17b31e542553c/docs/design/fused_moe_modular_kernel.md#L1-L180 |
| HV06 | vLLM | C1/C4 | SG6/7 | CODE | vllm/config/parallel.py | ParallelConfig DBO thresholds | 220-340 | RQ4/RQ9/RQ10 | https://github.com/vllm-project/vllm/blob/9ca6dbba71329a7b08711e2acfa17b31e542553c/vllm/config/parallel.py#L220-L340 |
| HX01 | TVM Relax/TIR | C2 Pipeline | Graph compiler | CODE | python/tvm/relax/backend/cuda/pipeline.py | CUDA Relax backend pipeline | 1-180 | RQ8 | https://github.com/apache/tvm/blob/8312a17f8734ddfd56e5f3977cd5df25b83ec49f/python/tvm/relax/backend/cuda/pipeline.py#L1-L180 |
| HX02 | TVM Relax | C3 Semantic fusion | Graph fusion | CODE | src/relax/transform/fuse_ops.cc | FuseOps post-dominator algorithm | 1-180 | RQ8 | https://github.com/apache/tvm/blob/8312a17f8734ddfd56e5f3977cd5df25b83ec49f/src/relax/transform/fuse_ops.cc#L1-L180 |
| HX03 | TVM Relax | C3 Pattern registry | Backend dispatch | DOC | docs/arch/fusion.rst | FuseOpsByPattern / FusionPattern | 1-280 | RQ8/RQ10 | https://github.com/apache/tvm/blob/8312a17f8734ddfd56e5f3977cd5df25b83ec49f/docs/arch/fusion.rst#L1-L280 |
| HX04 | TVM TIR | C1/C4 Search | Kernel schedule | DOC | docs/deep_dive/tensor_ir/tutorials/meta_schedule.py | MetaSchedule cost model / runner / database summary | 250-380 | RQ9 | https://github.com/apache/tvm/blob/8312a17f8734ddfd56e5f3977cd5df25b83ec49f/docs/deep_dive/tensor_ir/tutorials/meta_schedule.py#L250-L380 |
| HX05 | TVM Relax | C2 Layout op | Graph layout | CODE | python/tvm/relax/op/manipulate.py | layout_transform | 90-180 | RQ3/RQ7 | https://github.com/apache/tvm/blob/8312a17f8734ddfd56e5f3977cd5df25b83ec49f/python/tvm/relax/op/manipulate.py#L90-L180 |
| HX06 | TVM MetaSchedule | C3 Transfer/reuse | Tuning DB | DOC | docs/deep_dive/tensor_ir/tutorials/meta_schedule.py | Cross-model database reuse / anchor-block equality | 200-270 | RQ9 | https://github.com/apache/tvm/blob/8312a17f8734ddfd56e5f3977cd5df25b83ec49f/docs/deep_dive/tensor_ir/tutorials/meta_schedule.py#L200-L270 |

---

# 5. Correction and confirmation log

## 5.1 Claims narrowed/replaced

| ID | Correction | Verified replacement |
|---|---|---|
| HF03 | v2 overstated this row as a general row/column-major and scale-major contract. v3 narrows it to the representation/backend/tactic contracts directly verified in grouped_mm/core.py. | Grouped-MoE GEMM APIs expose explicit tensor-shape/dtype contracts, backend dispatch and tactic selection; quantized variants add scale/format requirements. |
| HT05 | v2's specific claim about equal per-expert FP8 input scales and max-scale repair was not re-located at this pinned HEAD; it is replaced by the currently verified fused-MoE representation-restructuring and support-boundary evidence. | AutoDeploy contains explicit fused-MoE restructuring logic that converts between stacked/backend-specific weight/graph representations and enforces backend/style support constraints. |

## 5.2 Claims strengthened by direct source confirmation

| ID | Confirmation |
|---|---|
| HI05 | The forensic pass found direct wording in EncodingOps.td confirming the original v2 statement that dynamic encoding_dims are used for runtime layout selection based on problem size. |

The IREE result is methodologically important: the audit should be willing to move in **both** directions.  
If source evidence is weaker than the research prose, narrow the claim.  
If the pinned source directly supports a stronger claim, restore it.

---

# 6. Provenance statistics

## 6.1 By framework

| Framework | Evidence units |
|---|---|
| CUTLASS/CuTe | 5 |
| FlashInfer | 5 |
| IREE | 6 |
| MLIR | 5 |
| SGLang | 6 |
| TVM MetaSchedule | 1 |
| TVM Relax | 3 |
| TVM Relax/TIR | 1 |
| TVM TIR | 1 |
| TensorRT-LLM | 6 |
| Triton | 5 |
| vLLM | 6 |

## 6.2 By provenance kind

| Kind | Count |
|---|---|
| CODE | 30 |
| CODE+DOC | 2 |
| DOC | 18 |

## 6.3 By evidence role

| Role | Count |
|---|---|
| FACT | 49 |
| INFERENCE | 1 |

## 6.4 By confidence

| Confidence | Count |
|---|---|
| AUDITED-RANGE | 50 |

---

# 7. Research-question support audit

| RQ | Horizontal evidence stacks |
|---|---|
| RQ1 | FlashInfer, SGLang, TensorRT-LLM |
| RQ2 | SGLang |
| RQ3 | FlashInfer, SGLang, TVM Relax |
| RQ4 | SGLang, vLLM |
| RQ5 | SGLang |
| RQ6 | FlashInfer, IREE |
| RQ7 | CUTLASS/CuTe, FlashInfer, IREE, MLIR, SGLang, TVM Relax, TensorRT-LLM, Triton, vLLM |
| RQ8 | CUTLASS/CuTe, IREE, MLIR, TVM Relax, TVM Relax/TIR, TensorRT-LLM, Triton, vLLM |
| RQ9 | CUTLASS/CuTe, FlashInfer, IREE, MLIR, SGLang, TVM MetaSchedule, TVM TIR, TensorRT-LLM, Triton, vLLM |
| RQ10 | CUTLASS/CuTe, FlashInfer, IREE, MLIR, SGLang, TVM Relax, TensorRT-LLM, Triton, vLLM |

The provenance audit preserves the v2 research structure:

- **RQ7:** producer-consumer / cross-subgraph representation contracts.
- **RQ8:** non-commutative pass/fusion/layout search.
- **RQ9:** system-level heuristic synthesis beyond mature local tuners.
- **RQ10:** bounded legal repair search for long-tail incompatibilities.

No new RQ was created during provenance work. That is intentional: provenance should verify or correct the evidence, not create questions merely because more source lines were inspected.

---

# 8. Four provenance chains that are now paper-ready

## 8.1 C1 Threshold chain → RQ9

**vLLM:** 256-token RoPE/KV fusion, 64 MB collective+norm example, DBO 32/512.  
**SGLang:** page-size reuse/throughput trade-off and TBO=0.48.  
**Counter-baselines:** TensorRT-LLM XQA runtime heuristic, Triton autotune, CUTLASS analytical heuristic, IREE tuning+SMT legality, TVM MetaSchedule.

Scientific consequence:

> The novelty cannot be “replace magic numbers with a cost model.”  
> RQ9 must model system variables absent from local tuners and demonstrate lower decision regret under distribution/hardware shift.

---

## 8.2 C2 Ordering chain → RQ8

**vLLM:** Sequence Parallelism must precede AsyncTP.  
**TensorRT-LLM:** ordered AutoDeploy stages; SiLU+Mul depends on prior GEMM fusion.  
**TVM:** explicit CUDA Relax pass order.  
**MLIR:** Transform tutorial explicitly states that producer-fusion order matters.

Scientific consequence:

> The problem is not whether transforms can be sequenced; MLIR already provides that representation.  
> The research gap is detecting the sparse non-commutative interaction graph and jointly searching only that component with layout.

---

## 8.3 Cross-subgraph representation chain → RQ7

**SGLang:** current NVFP4 path materializes scale interleave before CUTLASS MoE and contains a TODO to fuse it.  
**FlashInfer:** NVFP4 NHD→HND can materialize transpose/contiguous copies.  
**CUTLASS:** SMEM layout heuristics + efficient-epilogue algebraic transpose/swap.  
**MLIR:** pack_transpose can rewrite a pack/compute/unpack chain but has sole-consumer/topology restrictions.  
**IREE:** abstract Encoding with dynamic dimensions and runtime problem-size layout selection.  
**TVM:** explicit graph-level layout_transform.

Scientific consequence:

> A new system should treat physical representation as a producer-consumer edge contract and reuse these existing legality/execution mechanisms.

---

## 8.4 C4 Repair chain → RQ10

**TensorRT-LLM:** bespoke multi-user Add+RMSNorm transform; fused-MoE representation restructuring/support checks.  
**SGLang:** materialized scale interleave.  
**FlashInfer:** compatibility conversion.  
**Triton:** architecture-specific pipeline workaround.  
**MLIR:** pack_transpose silenceable failures for unsupported topology/specification.  
**IREE:** liveness-based shared-memory reuse and layout-transform combination.

Scientific consequence:

> Fallback space already contains recurring repair primitives.  
> RQ10 can formalize them as a bounded legal repair graph rather than invent arbitrary graph synthesis.

---

# 9. Novelty collision conclusions after source pinning

The provenance step makes the following negative conclusions robust:

1. **SMT legality is not novel by itself.** IREE already emits constraints and uses an SMT solver for valid tuning assignments.
2. **A cost model is not novel by itself.** TVM MetaSchedule and other local tuners already expose measured cost/search infrastructure.
3. **Pattern IR is not novel by itself.** MLIR PDL represents rewrite patterns as IR.
4. **Semantic fusion is not novel by itself.** TVM post-dominator fusion, MLIR structured fusion, IREE semantic matchers and TensorRT-LLM MLIR fusion are strong baselines.
5. **Dynamic problem-size layout is not novel by itself.** IREE Encoding directly supports dynamic dimensions used for runtime layout selection by problem size.
6. **Local kernel autotuning is not the missing layer.** Triton/CUTLASS/FlashInfer/TensorRT-LLM already cover substantial local adaptation.

The defensible novelty remains the **joint scope**:

\[
\text{cross-subgraph representation}
+\text{persistent state}
+\text{multi-consumer constraints}
+\text{communication}
+\text{online serving state}
\]

under compiler/backend legality.

---

# 10. Artifact-Evaluation schema

The CSV companion is the machine-readable source of truth. A reproduction script should enforce:

```text
unique(evidence_id)
sha is immutable and 40 hex chars
audited_range exists for every row
CODE rows point to source files
DOC rows point to versioned repository documentation
INFERENCE rows are explicitly labeled
RQ7-RQ10 each have multi-stack support
correction log is append-only
```

A later source-update audit should:

1. pin new repository SHAs;
2. re-fetch each range/symbol;
3. diff the normalized predicate/fact;
4. classify changes as semantic/no-semantic-change;
5. rerun Evidence→RQ coverage;
6. version the ledger rather than mutating old provenance.

---

# 11. Current project status after this step

| Layer | Status |
|---|---|
| v14 Attention/KV source semantics | 65 stable evidence units |
| v2 whole-LLM horizontal semantic evidence | 50 units |
| v3 immutable SHA pinning | **50/50 complete** |
| v3 audited range | **50/50 complete** |
| FACT vs INFERENCE | **50/50 complete** |
| provenance correction pass | **complete for the declared 50-unit ledger** |
| RQ1–RQ10 conceptual evidence chain | **closed enough for experimental formulation** |
| repository-wide mechanical branch enumeration | OPEN |
| empirical falsification | OPEN |

The next step should therefore move from **evidence formulation** to **measurement**.  
The recommended first empirical target remains the four-prototype set from v2:

1. `QKV → QK-Norm → RoPE → KV write`
2. `Gate+Up → SwiGLU → quant → Down GEMM`
3. `Collective → Residual → RMSNorm → quant`
4. `MoE dispatch/A2A → grouped GEMM → combine`

Start with RQ7 because it unifies the original KV-layout problem with the newly audited non-KV representation boundaries.


---

# 12. 从 evidence 到解法：如何真正解决上述问题

本章回答三个问题：

1. 每个 RQ 应该用什么机制解决？
2. 哪些不同框架的 evidence 可以互相反证过强结论？
3. 哪些 evidence 对方案设计和论文论证是决定性的？

## 12.1 建议的统一系统结构

这些问题不适合用十个彼此独立的优化器解决。更合适的是一个分层控制结构：

```text
               ┌───────────────────────────────────┐
               │  Runtime telemetry / workload     │
               │  batch, q/KV, reuse, accept rate │
               │  comm pressure, rank skew         │
               └────────────────┬──────────────────┘
                                │
                       epochal / live signals
                                │
┌───────────────────────────────▼────────────────────────────────┐
│ L4. Global policy / reconfiguration controller                 │
│ persistent layout scope, backend family, pool split, recapture │
│ hysteresis + migration/replan cost                             │
└───────────────────────────────┬────────────────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────┐
│ L3. Joint cost model                                            │
│ kernel + memory + conversion + communication + sync + launch   │
│ + peak memory + switching cost                                 │
└───────────────────────────────┬────────────────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────┐
│ L2. Legality + candidate generator                             │
│ backend capability ∩ compiler constraints ∩ ABI/metadata rules │
│ SMT/verifier / alignment / graph-capture constraints           │
└───────────────────────────────┬────────────────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────┐
│ L1. Typed representation / Edge Contract IR                    │
│ axis, stride, tile/page, packing, quant format, placement,     │
│ persistence, fanout, producer/consumer accepted forms          │
└───────────────────────────────┬────────────────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────┐
│ L0. Existing local experts                                    │
│ XQA / FlashInfer tactic / Triton autotune / CUTLASS heuristic  │
│ IREE tuning / TVM MetaSchedule / framework fast paths          │
└────────────────────────────────────────────────────────────────┘
```

最重要的设计原则是：

- 不替代成熟 local tuner，而是把它们当作 L0 专家。
- compiler 拥有 legality；性能模型不能把非法方案排在第一名后再补救。
- persistent state 与 ephemeral tensor 分开处理。
- conversion 是一条有成本的合法边，不是默认失败，也不是默认免费。
- fallback 是 repair graph 的终端高代价边。
- kernel 可 per-call 自适应，persistent layout 更适合 epochal + hysteresis。

## 12.2 RQ1–RQ10 解法总表

| RQ | 问题 | 建议解法 | 必须进入目标/状态的变量 | 搜索/控制形式 | 关键相互反证 | 成功判据 |
|---|---|---|---|---|---|---|
| RQ1 | Persistent control-plane vs local adaptivity | 两层 selector：persistent-plan 选择 layout/backend family；leaf/local 层继续复用 XQA/tactic/autotune。persistent 层用系统成本替代 first-compatible/priority。 | deployment/model/hardware + batch/q/KV/GQA + prefix reuse + conversion/transfer + communication contention | 分层 contextual policy；offline profiling + online residual correction | vLLM/SGLang priority ↔ HT06/HF01/HF04 | 证明 persistent 层新增变量降低 end-to-end regret，而不是替代 local tuner。 |
| RQ2 | Layout granularity | 把 layout scope 建模成 graph partition：whole model / pool / attention kind / layer group / phase。切分收益=局部 regret 减少；代价=额外 pool、metadata、fragmentation、conversion。 | group read count、bytes、phase frequency、consumer preference、pool overhead、fragmentation | 带切边成本的 partition / DP / min-cut-like formulation | vLLM global layout ↔ HS03/HI04；HS01 提供 page/reuse trade-off | 低异质性时 global layout 应作为负结果胜出。 |
| RQ3 | Conversion-aware optimization | 把 conversion 当作有成本的合法边。比较 common layout、strided view、one-time convert、producer-native emission。 | conversion bytes/time、temporary peak memory、reuse count R、remaining lifetime、network path | break-even model + shortest-path/DP over representation states | HF02/HX05 ↔ HC04/HS06 | 求出 R*，并证明 memory/network 会移动 R*。 |
| RQ4 | GQA/MLA/MoE/parallelism coupling | 目标从 isolated attention 扩成 token critical path；显式建模 KV replication/sharding、EP A2A、TP/DP/DCP、overlap、rank imbalance。 | Hq/Hkv、TP/EP sizes、rank token counts、A2A bytes、overlap slack、HBM/copy contention | critical-path DAG model + max-rank objective + measured contention correction | HF01/local kernel evidence ↔ HS05 + v14 distributed evidence | 证明 system winner 与 isolated winner 发生 inversion。 |
| RQ5 | Speculative multi-consumer negotiation | 在 cache allocation 前，把 target/draft/verify/transfer 作为共享 persistent-state 的多消费者 hyperedge，联合选择 backend/layout/page/metadata contract。 | acceptance rate、draft depth、verify length、target/draft dtype/head geometry、metadata location、memory premium | hypergraph constraint solving + weighted consumer cost | HS03 phase routing + v14 speculative constraints/repair evidence | 减少 independent-choice 后的 repair/failure，并保留低异质性负结果。 |
| RQ6 | Static commitment vs workload drift | 把 persistent config 当成 epochal state。用 change detection + hysteresis 判断何时 replan/recapture/migrate；预计收益超过 switching cost 才切换。 | traffic regime、prefix-hit drift、acceptance drift、remaining requests/reads、migration/recapture cost | renewal/epoch model + hysteresis controller | HI05/HT06 dynamic evidence ↔ HF05 + v14 startup commitment | 不能 per-request 粗暴切 persistent layout；必须测 amortization threshold。 |
| RQ7 | Cross-subgraph layout-contract synthesis | 建立 typed Edge Contract IR：axis/stride/tile/page/packing/quant/placement + producer/consumer capability + persistence。允许 producer-native emission、consumer view、conversion、split pool。 | producer/consumer accepted representations、fanout、reuse、quant scale format、placement | contract propagation + capability intersection + costed repair edges | HS06/HF02/HF03/HC04/HM03/HI04/HX05 | 优先实现；它是 whole-LLM layout 的核心抽象。 |
| RQ8 | Joint pass-order / fusion / layout search | 先做 dependence/legality，构造 transform interaction graph；只对 non-commutative connected components 搜索 pass order、fusion、layout，其他 pass 固定。 | read/write sets、producer-consumer topology、layout preconditions、fusion legality、resource limits | partial-order/beam search + 局部 e-graph + SMT legality | HV03/HT01/HT02/HM01/HX01 + HI02/HC05 | 避免全排列爆炸；证明 interaction graph 稀疏且收益集中。 |
| RQ9 | Portable threshold / heuristic synthesis | 保留现有 local tuner 作为专家，新增系统级 meta-policy；用稳定语义特征预测何时覆盖默认规则，并用 online measurements 做 residual calibration。 | shape/GQA + reuse + conversion + communication + persistent lifetime + hardware descriptors | mixture-of-experts policy / residual model / contextual bandit with safety gates | HV01/HV02/HV06/HS02 ↔ HT06/HR01/HC01/HI01/HX04 | novelty 是跨层特征、迁移与 regret，不是“用了 cost model”。 |
| RQ10 | Repair-oriented optimization | 把 ad-hoc fallback 变成有向 repair graph；节点是合法 representation/config state，边是 view/convert/split/duplicate/recompute/backend switch/partial fuse/fallback。 | legality、copy bytes、extra memory、compile/replan cost、accuracy/quant constraints | bounded shortest-path / A* over legal repairs；SMT/verifier 剪枝 | HF02/HS06/HC04/HM03/HT03/HT05/HI06/HR04 + HI02/HC05 | fallback 是终端高代价边；repair search 必须 bounded。 |

---

# 13. 哪些框架之间的 evidence 可以“相互反证”

这里的“反证”不是说两个框架实现互相矛盾。更严格地说：

> 当某个框架的 evidence 容易诱导出一个普遍化命题时，另一个框架的 evidence 是否提供现实 counterexample，使这个普遍化命题不能成立。

因此下面每一组都是：

```text
overclaim -> counterexample -> surviving research question
```

| ID | 容易产生的过强命题 | Evidence A | Counter-evidence B | 被反证的结论 | 设计含义 |
|---|---|---|---|---|---|
| CE01 | “当前框架的阈值/规则足够稳定，可以直接跨硬件和工作负载复用。” | vLLM HV01/HV02/HV06、SGLang HS02：存在明确固定阈值或离散规则。 | TensorRT-LLM HT06、IREE HI01/HI05、TVM HX04：已有 runtime/shape-aware heuristic 与 tuning 机制。 | 固定阈值不是不可避免的设计，也不能被当作普适最优代理。 | RQ9 应做“局部专家 + 系统级 residual/meta-policy”，而不是简单用另一个 magic number 替换现有阈值。 |
| CE02 | “现有系统完全没有 workload-aware 运行时优化。” | vLLM/SGLang 的许多 control-plane evidence 是 capability gate、priority、phase/model rule。 | TensorRT-LLM HT06、FlashInfer HF01/HF04 明确存在 batch/history/GQA/tactic-sensitive 的局部运行时选择。 | “所有框架都是纯静态规则”这一强命题。 | RQ1 必须研究 persistent commitment 与 local adaptivity 的边界，而不是“heuristic vs no heuristic”。 |
| CE03 | “一个全局/whole-model layout 或 backend scope 是结构上必需的。” | v14 的 vLLM whole-model KV layout resolver/全局 layout negotiation。 | SGLang HS03 支持 prefill/decode 不同 backend；IREE HI04 把 encoding 作为可传播/可物化的表示属性。 | 全局单一 scope 是唯一可行抽象。 | RQ2/RQ7 应把 layout scope 本身作为优化变量，但显式计入 pool/metadata/fragmentation 成本。 |
| CE04 | “避免 conversion 总是比 materialize conversion 更好。” | 全局共同 layout/兼容性优先策略倾向减少转换。 | FlashInfer HF02 会显式 NHD→HND；TVM HX05 把 layout_transform 作为合法图操作；SGLang HS01 显示 kernel 与 reuse trade-off。 | conversion 是绝对坏事。 | RQ3 应求 break-even reuse R*，比较 common layout、strided view、一次转换和 producer-native emission。 |
| CE05 | “repair 就等价于 memcpy/transpose。” | FlashInfer HF02、SGLang HS06 是 materialized conversion/interleave。 | CUTLASS HC04 可做代数等价 transpose/swap；MLIR HM03、TensorRT-LLM HT05、IREE HI06 暴露结构性/资源型 repair。 | repair action space 只有 copy/repack。 | RQ10 应搜索有限 repair graph：view、producer rewrite、algebraic remap、split、duplicate、convert、backend switch、fallback。 |
| CE06 | “只要有固定正确的 pass order，就不需要搜索 transform order。” | TensorRT-LLM HT01/HT02、TVM HX01、vLLM HV03 都有固定/依赖式 pass ordering。 | MLIR HM01 明确说明 fusion order matters；Triton HR03 显示 traversal/schedule order 改变 locality/performance。 | 合法顺序等价于性能最优顺序。 | RQ8 应先构造 non-commutativity interaction graph，只对冲突组件枚举/搜索 topological order。 |
| CE07 | “通用 pattern/fusion infrastructure 已经足够表达所有 LLM 优化。” | MLIR HM02/HM04、TVM HX02/HX03、IREE HI03 有通用 pattern/semantic fusion infrastructure。 | TensorRT-LLM HT03 仍需 multi-user 特判，SGLang HS06/HT05 等存在 backend-specific representation repair。 | 有通用 IR/pattern system 就自动解决 LLM-specific topology/ABI。 | RQ7/RQ8 的 IR 必须支持 backend contract、multi-consumer、persistent state 和 distributed ownership，而不只是 DAG pattern。 |
| CE08 | “引入 autotune/cost model 本身就是新的系统贡献。” | 静态阈值/priority evidence 提供明显改进动机。 | Triton HR01/HR02、CUTLASS HC01/HC02、IREE HI01、TVM HX04/HX06、FlashInfer HF04 已有成熟 local search/heuristics/cost model。 | “我们用了 cost model/autotune”足以构成 novelty。 | RQ9 的 novelty 必须是跨 persistent state、conversion、communication、reuse、serving drift 的 joint feature/objective。 |
| CE09 | “attention kernel latency 可以代表整个 token critical path。” | FlashInfer/Triton/CUTLASS 多数 local heuristic 直接优化 kernel/problem。 | SGLang HS05 与 v14 RQ4 evidence 暴露 MoE A2A、EP/TP/DP、overlap、rank geometry 等系统路径。 | 单 kernel winner 必然是系统 winner。 | RQ4 的目标应至少包含 max-rank critical path、HBM/copy contention、communication overlap。 |
| CE10 | “既然有 dynamic heuristic/layout，就可以随请求自由切换。” | IREE HI05、TensorRT-LLM HT06 证明动态选择机制可存在。 | FlashInfer HF05 与 v14 RQ6 的 vLLM/SGLang/TRT-LLM commitment evidence 显示 graph/planning/cache geometry 有生命周期和重配置成本。 | 动态选择等价于零成本任意时刻重配置。 | RQ6 应采用 epochal reoptimization + hysteresis + migration/recapture break-even。 |
| CE11 | “出现不兼容时唯一正确做法是直接 fallback。” | 许多框架存在 hard error、fallback 或 workaround 路径。 | SGLang HS06、CUTLASS HC04、IREE HI06、Triton HR04 显示可通过 representation/resource/schedule repair 保留更优实现。 | fallback 是唯一安全修复。 | RQ10 需要 legality-preserving bounded repair search，并把 fallback 设为最后一条高代价边。 |
| CE12 | “先枚举所有候选，再靠 benchmark/cost model 排序即可。” | Triton/TVM 等能枚举/测量丰富候选。 | CUTLASS HC05、IREE HI02 证明 alignment/instruction/pipeline 约束应在性能排序前剪枝并可由 compiler 生成。 | 性能搜索可以代替 legality reasoning。 | RQ8/RQ10 使用 legality-first candidate generation：compiler/SMT 剪枝后才进行成本搜索。 |

## 13.1 最重要的双向反证关系

### A. vLLM / SGLang 静态 control-plane ↔ TensorRT-LLM / FlashInfer 局部动态 heuristic

一边是 platform priority、capability gate、phase/model special case、startup/persistent commitment；另一边是 XQA 的 batch/history proxy、GQA group ratio、cuDNN tactic heuristic。

所以两个极端说法都不成立：

```text
错误命题 1：所有框架都是纯静态规则。
错误命题 2：既然已有 runtime heuristic，系统已经 workload-aware。
```

真正的问题是：

```text
persistent commitment 之前看见了什么？
local kernel 之后又能适应什么？
两层之间缺失了哪些 system-level variables？
```

这就是 RQ1 最稳健的形式。

### B. global/common layout ↔ conversion / algebraic repair

如果只看 whole-model/common-layout 机制，很容易推出：

```text
最好所有 consumer 共享一种兼容 layout，从而避免 conversion。
```

HF02 与 HC04 一起反证这个强结论：

- HF02：生产路径会付出 NHD→HND transpose/contiguous；
- HC04：还可以通过代数等价 swap/transpose computation 改善 epilogue，而非 materialize copy。

因此 candidate set 至少应包含：

```text
common layout
strided/view-only adaptation
materialized conversion
producer-native emission
algebraic producer/consumer rewrite
separate layout pools
```

### C. 通用 compiler infrastructure ↔ LLM-specific special cases

MLIR/TVM/IREE 已有 pattern IR、semantic matching、structured fusion、formal legality、tuning/search；但 HT03/HT05/HS06 又说明生产系统仍需 multi-user topology、checkpoint/runtime representation、backend-specific scale ABI、persistent lifetime 处理。

因此：

```text
“需要一个 compiler IR”不够新；
“generic compiler 已经自动解决 serving”也不成立。
```

RQ7/RQ8 应填补的是 compiler infrastructure 与 serving-specific persistent/distributed representation semantics 之间的缺口。

### D. dynamic layout/heuristic ↔ graph/persistent lifetime

HI05/HT06 说明动态选择可实现；HF05 与 v14 RQ6 说明 CUDA Graph、workspace、cache geometry 把切换变成有成本事件。

所以 RQ6 不应做：

```text
per-request persistent-layout switching
```

而应做：

```text
epochal reoptimization
+ hysteresis
+ explicit migration/recapture cost
```

---

# 14. 哪些 evidence 对解决问题最重要

不是所有 50 条 evidence 的科研价值相同。建议按作用分层。

## 14.1 P0：问题定义 / counterexample / causal-gap evidence

P0 能直接证明当前规则观察了什么、漏掉了什么、存在什么反例，是 Motivation、Problem Statement 和 falsifiable hypothesis 的核心。

当前：**28 / 50**。

## 14.2 P1：solution-enabling / hard-baseline evidence

P1 决定新系统不能重新发明什么，例如 Triton autotune、CUTLASS heuristic、IREE tuning/SMT、TVM MetaSchedule、MLIR pattern IR。

当前：**22 / 50**。

## 14.3 P2：context / coverage evidence

P2 用于覆盖工程语境，但单条不足以支撑顶层科研命题。

当前：**0 / 50**。

## 14.4 每个 RQ 最决定性的 evidence

| RQ | P0 / 最关键 evidence | 为什么关键 |
|---|---|---|
| RQ1 | HT06, HF01, HF04 | 直接建立“静态 persistent control-plane 与局部 runtime adaptivity 共存”的问题定义。 |
| RQ2 | HS01, HS03, HI04 | 同时覆盖 reuse/页粒度、phase differentiation 和可传播 representation，决定 layout scope 是否应细化。 |
| RQ3 | HF02, HX05, HC04 | 分别提供 materialized conversion、显式 layout op、algebraic repair 三种 repair 语义。 |
| RQ4 | HF01, HS05, HS02 | 把 GQA 的 local crossover 与通信/overlap 系统变量放进同一目标。 |
| RQ5 | HS03, HF05, HT05 | 证明共享 persistent state 存在 phase/consumer routing、生命周期和 representation 约束。 |
| RQ6 | HF05, HI05, HT06 | 同时证明“动态决策可存在”与“persistent state 不能零成本随时切换”。 |
| RQ7 | HS06, HF02, HC04, HM03, HI04, HX05 | 完整覆盖 producer/consumer contract、materialized repair、algebraic repair、compiler encoding 和 graph transform。 |
| RQ8 | HT01, HT02, HM01, HI02, HC05, HX01 | 直接证明固定 pass dependency、order sensitivity 和 legality-first pruning。 |
| RQ9 | HV01, HV02, HV06, HT06, HR01, HC01, HI01, HX04 | 形成 static threshold ↔ runtime heuristic ↔ mature tuner/cost-model 的三角反证，避免 novelty 过度声明。 |
| RQ10 | HT03, HT05, HF02, HS06, HC04, HM03, HI06, HR04, HI02, HC05 | 覆盖 topology、representation、resource、architecture repair 与 formal legality。 |

---

# 15. Evidence 如何直接进入系统设计

Evidence 不应该只出现在 Related Work。建议转换成四种工程输入：

| Evidence role | 在系统里的作用 |
|---|---|
| Problem-defining | 决定 cost model 必须观察哪些变量 |
| Counterexample | 防止过强假设；决定系统必须允许哪种 alternative |
| Mechanism evidence | 复用已有 IR、legality、kernel/tuner mechanism |
| Repair evidence | 定义 bounded repair action space |

例如：

```text
HF02
  -> variable: conversion_bytes, temp_memory, remaining_reads
  -> candidate: materialize NHD->HND

HC04
  -> candidate: algebraic producer rewrite
  -> proves repair action space 不能只有 memcpy

HI02 + HC05
  -> legality constraints
  -> prune candidate before cost ranking

HT06 + HF01
  -> local runtime expert
  -> global policy should compose with it, not replace it

HF05
  -> switching_cost / epoch boundary
  -> persistent reconfiguration cannot be per-request
```

---

# 16. 推荐求解算法：不要做一个“大一统黑盒模型”

## 16.1 Step 1 — Legality-first

先构造：

```text
C_legal =
    backend capability
  ∩ layout/stride/alignment constraints
  ∩ graph/pattern topology constraints
  ∩ quantization/metadata ABI
  ∩ CUDA-Graph/persistent-state constraints
  ∩ distributed ownership constraints
```

关键 evidence：HC05、HI02、HM03、HF03、HS05。

非法 candidate 直接删除，不交给性能模型。

## 16.2 Step 2 — Edge Contract propagation

每条 edge 保存：

```text
Representation:
  logical_shape
  axis_order
  strides
  page_or_tile
  packing
  quant_format
  scale_format
  placement
  persistence
  lifetime
  producer_accepts
  consumer_accepts
```

先求：

```text
R_direct = R_producer ∩ R_consumer
```

若为空，不直接 fallback，而是生成 repair edges。

关键 evidence：HI04、HF03、HM03、HX05、HS06。

## 16.3 Step 3 — Repair graph

固定有限 action：

```text
view / stride reinterpret
producer-native emission
consumer alternate kernel
algebraic transpose/swap
materialized transpose/repack
split layout pool
duplicate state
recompute
quant/dequant or scale-format conversion
backend switch
partial defusion
fallback
```

每条边必须记录：

```text
legality predicate
latency estimate
bytes moved
temporary memory
persistent memory delta
compile/replan cost
accuracy constraints
```

关键 evidence：HF02、HC04、HS06、HT03、HT05、HI06、HR04。

## 16.4 Step 4 — Hierarchical cost model

不要训练一个模型直接输出“最佳 layout”。

更稳健的是：

```text
J =
  J_local_kernel
+ J_representation
+ J_communication
+ J_persistent_reuse
+ J_switch
```

其中：

```text
J_local_kernel
  <- XQA / Triton / CUTLASS / library heuristic

J_representation
  <- conversion bytes, view penalty, cache/TLB, temp memory

J_communication
  <- A2A/TP/DCP bytes, overlap slack, rank skew

J_persistent_reuse
  <- remaining reads, prefix fanout, cache lifetime

J_switch
  <- replan, graph recapture, migration, compile
```

## 16.5 Step 5 — 时间尺度分离

建议三个时间尺度：

```text
per-call:
    local tactic / kernel heuristic

per-batch or short epoch:
    backend variant / overlap / routing policy

long epoch:
    persistent layout pool / page geometry / graph recapture / state migration
```

---

# 17. 推荐解决顺序

不要按 RQ 编号做，而按系统依赖做。

### 第一阶段：RQ7 + RQ3

先解决：

```text
怎样表达 representation contract？
怎样表达 conversion/repair cost？
```

没有这两项，后续 joint search 没有状态空间。

第一实验建议仍是：

```text
QKV -> QK-Norm -> RoPE -> KV write -> decode attention
```

比较：

```text
framework baseline
consumer-native layout
producer-native layout
common neutral layout
common + explicit conversion
strided/view path
offline oracle
```

### 第二阶段：RQ8

Edge Contract IR 可用后，再把：

```text
fusion / rewrite / pass order / layout
```

共同放进 interaction graph。

### 第三阶段：RQ9 + RQ1

合法方案和 cost decomposition 稳定后，再研究：

```text
default rule 什么时候应被 override？
```

### 第四阶段：RQ6

selector 稳定后，再加入：

```text
when to switch
```

否则会混淆“选什么”和“何时切换”。

### 第五阶段：RQ4 + RQ5

最后扩展 distributed/speculative coupling。

### 第六阶段：RQ10

repair graph 从第一阶段就应作为基础设施存在，但作为独立科研命题，应在 RQ7/RQ3 已有实测 repair cost 后评估。

---

# 18. 最小可实现 prototype

最先只实现四个组件：

```text
1. ContractExtractor
   framework/backend metadata -> typed edge contracts

2. LegalCandidateGenerator
   contracts + compiler/backend constraints -> legal candidates

3. CostEvaluator
   local kernel time
   + conversion
   + memory
   + communication proxy
   + switching cost

4. Planner
   static oracle first
   -> then heuristic/meta-policy
```

最初不要做 learned model。

先用 exhaustive/offline oracle 证明：

```text
regret(default framework rule) > measurement noise
```

以及：

```text
joint representation plan
    < best local-independent plan
```

如果 oracle 没有稳定 gain，RQ7/RQ8 不应继续扩张。

---

# 19. 用 evidence 构造 falsification，而不是只构造支持

| Proposed claim | 最危险的 counter-evidence | 必须做的 negative experiment |
|---|---|---|
| per-group layout 更好 | RQ2-H6 / stride-polymorphism | 低异质性和低 reuse 下不应强行 split |
| conversion 有价值 | HF02 copy overhead | short decode / memory pressure 下应选择不转换 |
| joint pass search 有价值 | fixed mature pipeline | 大多数 graph 上 search 应快速退化为 baseline |
| meta-policy 优于阈值 | Triton/CUTLASS/TVM/IREE local tuner | 只给 local shape 时不能宣称胜过成熟 tuner |
| dynamic reconfiguration 有价值 | HF05 graph lifetime | stationary trace 上 hysteresis 应保持不切换 |
| repair search 优于 fallback | legality complexity | repair cost 高时必须选择 fallback |

---

# 20. 本轮结论

十个 RQ 可以归并为四个核心机制：

1. **Representation Contract**：RQ2/RQ3/RQ5/RQ7。
2. **Legality-aware Joint Search**：RQ8/RQ10，并为其它 RQ 提供候选空间。
3. **Hierarchical System Cost Model**：RQ1/RQ4/RQ9。
4. **Epochal Reconfiguration Controller**：RQ6。

最重要的跨框架反证关系是：

```text
static priority       ↔ live local heuristic
global common layout  ↔ explicit/algebraic conversion
generic compiler IR   ↔ LLM-specific topology/ABI repair
dynamic selection     ↔ persistent/graph switching cost
local kernel optimum  ↔ distributed token critical path
benchmark search      ↔ compiler legality constraints
```

因此更准确的研究目标不是：

> 找一个比所有 framework heuristic 更聪明的 selector。

而是：

> **建立一个 legality-preserving、representation-aware、multi-timescale 的上层 planner，复用现有 local tuners，并显式处理 conversion、persistent reuse、communication 和 reconfiguration cost。**

这也是目前 115 个 evidence units 最一致地指向的系统设计。
