# layout_framework_compare_continued_v14_canonical_evidence_RQ1_RQ6_audited.md

> Audit date: 2026-09-16  
> Canonical successor to v13.  
> Purpose: remove the obsolete pre-counterexample RQ1 wording, normalize the original v12 rules into stable IDs, add reverse evidence coverage, and adjudicate the newly justified RQ6.
>
> Scope: vLLM, SGLang, TensorRT-LLM and FlashInfer public LLM-serving/KV-layout decision paths that materially affect backend selection, persistent KV physical layout, cache/page/pool organization, conversion, phase/speculative routing or distributed interaction.
>
> Excluded unless they affect LLM persistent KV layout: diffusion-only attention selection, unrelated training kernels, optional experimental plugins with no persistent-KV consequence. For example, SGLang diffusion has a warmup attention-backend autotune facility; it is not used here as evidence about LLM persistent KV layout.

# 0. Current completion verdict

v13 was **scientifically advanced but not canonical**: it physically contained the old v12 RQ1 plus the corrected v13 RQ1 supplement. v14 removes that ambiguity.

Within the declared LLM/KV source scope, the current audit contains:

- **65 stable semantic decision-rule units**: 19 normalized from the original v12 audit, 42 added by v13, and 4 new temporal-commitment rules in v14.
- **6 top-level RQs**.
- **33 falsifiable hypotheses**: RQ1=8, RQ2=6, RQ3=5, RQ4=5, RQ5=4, RQ6=5.
- a reverse **Evidence ID → RQ → Hypothesis** coverage table.

The key distinction remains:

- `SOURCE_CHAIN`: closed at semantic-decision-path level within this declared scope.
- `REPOSITORY_EXHAUSTIVENESS`: not yet mechanically proven for every optional plugin/commit branch.
- `PROVENANCE_PINNING`: file/function URLs are present, but not every rule is pinned to an immutable commit SHA + exact line range.
- `EMPIRICAL_HYPOTHESIS`: open until benchmark/falsification.

Therefore “证据链完善” has two answers:

1. **Research-logic completeness: yes, substantially closed.**
2. **Absolute repository-forensic completeness: not yet claimable.**

# 1. Normalized original v12 rules that remain part of the canonical evidence ledger

| ID | Framework | Strategy | Primary RQ | Hypothesis link | Rule | Proxy | Assumption | Ignored variables | Fallback/repair | Trigger | Layout consequence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V-B1 | vLLM | explicit backend override | RQ1 | RQ1-H1/H2 | Explicit backend is validated then used. | backend name, dtype/KV dtype, head/block size, feature/capability predicates | user choice is authoritative once feasible | live shape distribution, prefix reuse, conversion/transfer/system contention | hard error when explicit backend is incompatible | all supported attention families | backend constrains admissible persistent layouts/pages |
| V-B2 | vLLM | first-compatible platform priority | RQ1 | RQ1-H1/H2 | Automatic selection takes the first compatible backend in a platform priority list. | SM generation, attention family, dtype/KV dtype, head/block size, feature flags | platform priority is a good global performance surrogate after feasibility | live batch/q_len/KV_len, GQA ratio as cost term, reuse, contention, conversion | try next backend; error if none valid | MHA/MQA/GQA/MLA | chosen backend contributes layout/page constraints |
| V-B3 | vLLM | backend-per-kind override | RQ1,RQ2 | RQ1-H5; RQ2-H1 | Attention/KV semantic kind can override the global backend. | MLA, sliding-window, attention type, config map | semantic kind is sufficient specialization granularity | per-layer execution weight, bytes, live shape and phase cost | missing kind uses global backend | hybrid attention kinds | heterogeneous backends must still negotiate cache layout |
| V-L1 | vLLM | backend-declared layout support | RQ1,RQ2 | RQ1-H4; RQ2-H1 | Each backend declares supported/preferred KV layouts. | backend support/preference list | backend preference is an adequate performance proxy | runtime frequency, bytes, cache locality, conversion cost | default layout preference if none declares restrictions | all paged attention | defines candidate persistent layouts |
| V-L2 | vLLM | layout intersection | RQ2,RQ3 | RQ2-H1/H6; RQ3-H1 | Candidate layouts are the intersection across all consuming backends. | boolean membership in supported-layout sets | one common physical layout is preferable to conversion/multiple pools | per-consumer benefit, conversion amortization, split-pool cost | empty intersection is hard error | multi-backend/group models | forces one common feasible layout |
| V-L3 | vLLM | first-preference voting | RQ1,RQ2 | RQ1-H4; RQ2-H2 | When declarations differ, candidates are ordered by how many backends rank a layout first. | unweighted first-choice vote count | one backend vote is a comparable utility unit | layer count, executed KV bytes, phase frequency, performance-gap magnitude | tie keeps enum order | multiple layout-preferring backends | global preference can be selected without workload weighting |
| V-L4 | vLLM | whole-model layout resolution | RQ2,RQ6 | RQ2-H1/H6; RQ6-H1/H4 | Resolve one KV-cache layout for the whole model. | worker candidate lists, cache specs/config | uniform persistent layout is preferable operationally | per-group/phase/consumer optimum and workload drift | mismatch/error; resolved layout is recorded | all models using resolver | strong global and temporal commitment |
| V-L5 | vLLM | mixed-shape compactness gate | RQ2 | RQ2-H1/H4 | Mixed H/N/C shapes narrow candidates to block-compact layouts. | static cache-spec shape heterogeneity | block compactness is required/robust for mixed shapes | group execution weight, separate-pool alternative | no candidate -> failure | hybrid KV specs | prunes generalized layout permutations |
| V-L6 | vLLM | connector layout preference | RQ3,RQ4 | RQ3-H2/H5; RQ4-H1 | KV connector preferred layout is advisory and only used if locally compatible. | connector preference plus local candidate set | local attention compatibility dominates transfer-native format | link bandwidth, transfer frequency, overlap, remote/native speedup | drop connector preference with warning | disaggregated/prefix KV transfer | creates local-attention vs transfer-layout tradeoff |
| S-B1 | SGLang | Hopper MHA default | RQ1 | RQ1-H1/H5 | Hopper standard attention prefers FA3 when supported. | GPU generation, CUDA/model support | hardware class predicts best backend | live batch/sequence/GQA/reuse | alternative compatible backend | MHA/MQA/GQA | backend-specific page/layout behavior |
| S-B2 | SGLang | Blackwell MHA + speculation rule | RQ1,RQ5 | RQ1-H5; RQ5-H2 | Blackwell MHA prefers TRTLLM MHA except speculative top-k constraints alter the path. | Blackwell, speculative top-k | top-k is a sufficient speculative classifier | acceptance rate, verify length, batch/KV length | compatible alternative or reject unsupported combination | speculative decoding | page/backend contract can change |
| S-B3 | SGLang | other-CUDA availability priority | RQ1 | RQ1-H1 | Other CUDA platforms prefer FlashInfer if available, else Triton. | platform class, package availability | platform+availability approximates performance | live workload and reuse | Triton | standard attention | different paged implementation |
| S-B4 | SGLang | MLA Hopper rule | RQ1,RQ2 | RQ1-H1; RQ2-H3 | MLA Hopper path prefers FA3. | GPU generation, MLA | hardware generation dominates phase/kernel choice | live shape, latent geometry, reuse | platform-compatible alternative | MLA | phase/layout compatibility differs from decode-native kernels |
| S-B5 | SGLang | MLA Blackwell/model special case | RQ1,RQ2 | RQ1-H1; RQ2-H3 | MLA Blackwell prefers FlashInfer, with model-specific TRTLLM MLA auto selection in documented cases. | GPU generation, model identity | model identity stands in for geometry/performance regime | deployment request mix | general Blackwell MLA fallback | MLA/DeepSeek-like | backend-native page/layout constraints |
| S-B6 | SGLang | MLA other-architecture fallback | RQ1 | RQ1-H1 | Other architectures can fall back to Triton MLA. | platform class | broad platform class is enough | live workload | Triton | MLA | fallback layout/page semantics |
| T-B1 | TensorRT-LLM | top-level backend default/config | RQ1 | RQ1-H1/H3 | TRT-LLM attention is the production/default backend family unless configured otherwise. | backend configuration and feature support | production default is a strong baseline | live workload at family/layout commitment time | configured supported path / explicit fallback | MHA/GQA/MLA as supported | backend family fixes cache/kernel contracts |
| F-L1 | FlashInfer | explicit NHD/HND layout contract | RQ2,RQ3 | RQ2-H5; RQ3-H3 | Paged KV APIs consume caller-selected NHD/HND layouts. | kv_layout and strides/shapes | system-level policy belongs to caller | cross-consumer/global serving objective | invalid layout fails or wrapper-specific repair | MHA/MQA/GQA APIs | layout is first-class caller contract |
| F-B1 | FlashInfer | GQA tensor-core path | RQ1,RQ4 | RQ1-H5; RQ4-H2 | Tensor-core decode can be faster for large GQA group size. | Hq/Hkv and policy option | GQA group size predicts kernel crossover | global conversion/system contention | non-tensor-core path | GQA | proves Q:KV ratio is performance-relevant |
| F-L2 | FlashInfer | NVFP4 layout conversion | RQ3 | RQ3-H1/H4/H5 | Some TRTLLM-gen NVFP4 paths accept NHD but transpose/materialize HND. | backend, KV dtype, input layout | conversion is preferable to rejection | reuse count, memory pressure, native-HND allocation option | transpose + contiguous copy | NVFP4 context/decode | feasible layouts are not cost-equivalent |

# 2. v13 detailed rule additions and serial RQ1→RQ5 adjudication

# A1. Additional vLLM rules missing from v12

| ID | Decision site | Rule | Proxy actually used | Implicit assumption | Important ignored variables | Fallback / repair | Trigger / layout consequence |
|---|---|---|---|---|---|---|---|
| V-A01 | `vllm/platforms/cuda.py::_get_backend_priorities` | Multimodal-prefix configurations can prepend composite backends such as Triton+FlashAttention or Triton+FlashInfer that route current queries between kernels while sharing KV. | SM generation, multimodal-prefix flag, causal/bidirectional query semantics, feature support. | Query semantic type is enough to route kernels and sharing one cache contract is preferable to per-route cache copies. | Per-query measured latency, image/text mixture, conversion alternative, route-switch locality. | Fall through to ordinary candidates if composite backend is incompatible; internal component fallback applies. | Multimodal mixed causal/non-causal attention; forces a common cache contract across routed kernels. |
| V-A02 | same | Standard-attention priority differs by GPU generation: Blackwell begins with FlashInfer; Ampere/Hopper begins with FlashAttention. | Compute capability, non-causal flag, multimodal-prefix state. | GPU generation is a high-signal performance classifier. | Batch/KV-length crossover, GQA ratio, prefix hit rate, transfer pressure. | Next compatible candidate. | MHA/MQA/GQA; selected backend changes layout/page constraints. |
| V-A03 | same | MLA has separate SM10, SM12 and other-GPU ordered backend sets. | `use_mla`, SM generation, head size, KV dtype, number of heads. | A small piecewise architecture/geometry map approximates MLA performance. | Live batch, q_len, KV_len, actual sparsity, transfer/MoE contention. | Next compatible MLA backend. | MLA; backend-specific latent-cache layout. |
| V-A04 | same | Sparse MLA preference changes with KV dtype and query-head count; BF16-like paths use a head-count threshold, while quantized KV favors a different sparse backend order. | KV dtype, query-head count, SM generation. | Head count + quantization approximate sparse-kernel crossover. | Selected-page count, sparse density, KV length, page locality, batch. | Next compatible sparse backend. | Sparse MLA; sparse page/index layout can change. |
| V-A05 | same | Source comments acknowledge a TokenSpeed MLA batch-size crossover, but the top-level priority order itself does **not** read live batch size. | Platform/backend static order. | Fixed ordering is acceptable despite a known workload-dependent crossover. | Live batch size, queue composition, SLA objective. | Static next-candidate fallback, not a batch-conditioned switch. | MLA on applicable Blackwell paths. |
| V-A06 | FA backend selection/version policy | FA version is architecture-conditioned; specialized FA4 paths have block-size/head-size/feature conditions and can transparently fall back. | SM, head size, block size, unsupported feature flags. | Architecture/version and a few geometry gates capture the optimized region. | Current sequence length, batch, graph state, prefix reuse. | Older compatible FA implementation or make the higher-priority backend ineligible. | Standard attention; can change required page/block size. |
| V-A07 | MLA prefill backend logic | MLA prefill generally prefers FA, but Blackwell has extra fallbacks and a known geometry can try TRT-LLM Ragged before FA. | Phase, SM, qk-nope/rope dimensions, value dimension. | Known model geometry is a stable performance classifier. | Actual prefill length/batch and prefix reuse. | Ordered fallback among compatible prefill kernels. | MLA prefill; demonstrates phase-specific kernel policy. |
| V-A08 | `selector.py::AttentionSelectorConfig` | Backend validation includes sinks, sparse, multimodal prefix, per-head scales, non-causal, batch-invariance, connector, PCP/DCP, adaptive verification, RSWA and other flags. | Feature booleans plus dtype/head/block geometry. | Feasibility can be separated from performance optimization. | Continuous runtime workload and end-to-end contention. | Invalidate candidate and move down priority. | Modern hybrid/distributed features constrain feasible layout/backend set. |
| V-A09 | KV connector model-runner / transfer utilities | Connector-specific paths can prefer HND for transfer while generic/default representations may be NHD-like. | Connector identity/mode, configured preference. | Connector type approximates transfer-efficient organization. | Actual link bandwidth, message size, local-kernel penalty. | Drop incompatible connector preference or retain local candidate. | Disaggregated/prefix KV transfer; exposes transfer-vs-attention tension. |
| V-A10 | cross-layer block gate | Uniform cross-layer blocks require a conjunction of transfer presence, connector preference, simple compatible attention grouping/spec, quantization compatibility and backend block-stride indexing. | Connector preference, group count/spec type, quantization, backend indexing. | Cross-layer packing is beneficial only under structurally simple compatibility. | Measured registration/copy cost, layer execution weight, network packetization. | Ordinary per-layer organization. | Transfer workloads; may allocate a contiguous all-layer tensor. |
| V-A11 | attention layer feature repair | Certain backend + batch-invariance/prefix-cache/chunk-lookback combinations disable a feature, warn, or require a specific backend. | Feature flags and backend identity. | Correctness dominates cache optimization. | Lost prefix-reuse opportunity cost. | Disable feature, warn, or reject incompatible backend. | Prefix caching/batch invariance; changes effective KV reuse even without axis permutation. |
| V-A12 | target/draft + layout resolver, current issue evidence | Target and draft can independently select backend constraints whose supported-layout intersection is empty, causing initialization failure. | Target layout set, draft layout set. | Independent local selection followed by global intersection is sufficient. | Joint target/draft cost, second-best backend pair, conversion, separate pools. | Current observed behavior can be hard failure. | Speculative decoding; motivates joint negotiation RQ5. |

### vLLM assumption summary

The added rules make the key assumption pattern clearer: vLLM has a rich **feasibility model**, but its persistent backend/layout arbitration is not a general runtime cost minimizer. A particularly strong falsifiable clue is V-A05: a source-known batch-size crossover is represented by static priority rather than a live batch-conditioned decision.

---

# A2. Additional SGLang rules missing from v12

| ID | Decision site | Rule | Proxy actually used | Implicit assumption | Important ignored variables | Fallback / repair | Trigger / layout consequence |
|---|---|---|---|---|---|---|---|
| S-A01 | attention backend guide / setup | Separate prefill and decode backend overrides exist; if they resolve differently, HybridAttnBackend is built. | Phase-specific config/backend names. | Phase is a sufficiently important discriminator to justify separate policy. | Within-phase batch/KV-length crossover, conversion cost. | Single backend if names match; invalid combination fails. | MHA/MLA; two consumers share persistent cache. |
| S-A02 | `HybridAttnBackend` | Decode/idle uses decode backend; extend/prefill uses prefill backend; target-verify backend depends on speculative-attention mode. | Forward mode and speculative-attention mode. | Semantic mode is sufficient online routing state. | Actual q_len/KV_len, acceptance rate, per-batch crossover. | Route to the configured phase backend. | Hybrid phase routing; shared cache must satisfy both consumers. |
| S-A03 | auto MHA policy | Hopper favors FA3 when CUDA/model support conditions hold. | GPU generation, CUDA version, model support. | Hopper+FA3 is broadly best after feasibility. | Seq length, batch, GQA ratio, prefix reuse. | Alternative platform backend. | MHA/MQA/GQA. |
| S-A04 | auto MHA policy | Blackwell favors TRTLLM MHA except speculative top-k > 1 changes the path/constraints. | Blackwell, speculative top-k. | Speculative width is an adequate classifier. | Acceptance rate, verify length, KV length. | Compatible alternative backend. | Speculative decoding. |
| S-A05 | page-size guide | `page_size=1` maximizes prefix reuse because only full pages are reusable, while larger pages generally improve attention-kernel performance. | Page size, prefix-cache semantics. | Prefix reuse and kernel efficiency are dominant opposing terms. | Real prefix distribution, fragmentation, transfer granularity, graph state. | User/configuration choice; no universal joint optimizer. | Paged attention; direct RQ2/RQ3 tradeoff. |
| S-A06 | speculative backend rules | Target verify/draft routing is conditioned on speculative mode; some backend/page/top-k combinations are forbidden. | Spec mode, top-k, page size, backend. | Mode/top-k capture principal speculative constraints. | Acceptance distribution, draft depth, target/draft asymmetry. | Switch supported mode/backend or reject. | Speculative target/verify. |
| S-A07 | `attention_backend_setup.py` | A draft worker can override both draft prefill and draft decode to a **single draft backend**. | Draft-worker flag, draft backend. | One draft backend simplifies cache/graph integration enough to outweigh phase specialization. | Cases where draft prefill/decode prefer different kernels, including FP4 paths. | Inherited/default draft backend; incompatible cases can fail. | Speculative draft model; forces one draft cache contract. |
| S-A08 | DSA/DeepSeek sparse backend policy | DSA is architecture-triggered and uses phase/platform/dtype-specific sub-backend rules. | Model architecture, phase, GPU, BF16/FP8, some parallel-state constraints. | A discrete platform/dtype/phase map approximates sparse-kernel optimum. | Actual selected-token count/locality, sparse density, batch/KV length. | Alternative supported DSA backend or reject. | Sparse MLA/DSA; sparse index + KV layout. |
| S-A09 | AITER backend | `vectorized_5d` cache layout routes writes/decode through different compatible paths because a 4D view is not representable. | Pool layout tag, backend/platform. | Representability is a hard execution constraint. | Cost of keeping 5D vs converting; workload crossover. | 5D-compatible writer/decode path. | ROCm/AITER; layout changes control flow. |
| S-A10 | hybrid linear/GDN setup | Modern hybrid recurrent/linear-attention architectures use a separate linear-attention backend; some FlashInfer prefill paths require a conjunction over GPU/CUDA/state dtype/head dims/cache mode/chunk size. | Architecture, SM, CUDA, recurrent-state dtype, head dimensions, cache/chunk mode. | A hand-specified conjunction captures the optimized region. | Live batch, state reuse, MoE/system contention. | Base linear-attention backend. | Hybrid attention introduces persistent recurrent state in addition to KV. |

### SGLang assumption summary

SGLang is materially more dynamic than a single fixed-priority selector because it can route by phase and speculative mode. But the routing variable is primarily **semantic phase**, not a measured global cost. Its own page-size guidance explicitly exposes a prefix-reuse versus kernel-efficiency tradeoff that remains configuration/policy rather than a universal runtime optimization.

---

# A3. TensorRT-LLM additions — the critical RQ1 counterexample

| ID | Decision site | Rule | Proxy actually used | Implicit assumption | Important ignored variables | Fallback / repair | Trigger / layout consequence |
|---|---|---|---|---|---|---|---|
| T-A01 | PyTorch attention backend config | TRT-LLM is the production/default backend family; Vanilla and FlashInfer are available alternatives/configurations. | Configured backend and feature support. | Production default is a strong global baseline. | Live workload at top-level family choice. | Supported configured path or documented fallback/error. | Top-level backend family. |
| T-A02 | KV cache manager | Layers with incompatible KV geometry/window characteristics use separate pools rather than one homogeneous pool. | KV-head count, attention-window/cache geometry. | Geometry-homogeneous pooling is the right allocation scope. | Per-layer runtime weight, phase preference, remote consumer. | Create another pool. | Direct production counterexample to “one pool for entire model.” |
| T-A03 | context FMHA | Context attention can choose different algorithms for short versus larger sequences. | Context length and support predicates. | Sequence length predicts algorithm crossover. | Global cache/transfer/MoE objective. | Alternate context algorithm. | Prefill; local adaptive kernel. |
| T-A04 | `decoderXQARunner::shouldUse` | XQA first checks support, then a performance heuristic; XQA may also be forced when legacy masked-MHA cannot support a dtype combination. | Support predicates, Q/KV dtype, runtime config. | Local feasibility/performance can be decided after persistent cache setup. | Global conversion/transfer and other-layer contention. | Masked MHA when valid and heuristic says XQA is not useful; otherwise XQA. | Decode. |
| T-A05 | `decoderXQARunner::mayHavePerfGain` | XQA estimates available work approximately from KV-heads × batch-size × multi-block-count; multi-block count grows with history length; compares this against an SM-derived threshold. | KV-head count, batch size, history length, multi-block state, SM count. | Parallel work count approximates occupancy and XQA crossover. | Layout transaction efficiency, L2 locality, prefix reuse, MoE overlap. | Use masked-MHA path if supported. | MHA/GQA/MQA decode. **True live workload heuristic.** |
| T-A06 | generation multi-block | Multi-block attention increases parallel work when batch×heads underutilizes SMs and can be forced by shared-memory limits. | Batch, head count, SM count, history/shared-memory needs. | Parallel-block count and shared-memory capacity predict benefit/necessity. | Layout/coalescing and system contention. | Single-block mode if enough parallelism. | Small-batch/long-context decode. |
| T-A07 | sparse attention | Specialized sparse kernels have hard HND/page-size/geometry constraints; unsupported classes can fall back to another sparse path or error. | Sparse algorithm, page/layout, dtype/head geometry. | Narrow physical contract is required for specialized sparse efficiency/correctness. | Conversion cost from general layout. | Alternative sparse implementation or hard error. | Sparse MHA/MQA/GQA/MLA. |
| T-A08 | disaggregated serving | Producer/consumer KV layouts and parallel formats can be transformed during transfer. | Producer and consumer format/parallel config. | Explicit formatting is better than requiring identical endpoint layout. | Where to place conversion under current load, conversion amortization. | Convert/format, otherwise unsupported path fails. | Disaggregated prefill/decode. |
| T-A09 | disaggregated serving | Transfer can overlap computation. | Transfer stage and scheduler/connector capability. | Overlap can hide data-motion cost. | Competition with MoE/copy engines. | Non-overlapped path. | Changes conversion break-even threshold. |
| T-A10 | parallel strategy | Attention can use TP or DP; documentation treats TP as suitable for smaller batches and attention-DP for larger throughput regimes. | Deployment parallel configuration / expected batch regime. | Batch regime is a major predictor of parallel strategy. | Layout-locality and per-rank sequence skew. | User/deployment chooses alternative. | Changes KV ownership/partitioning. |
| T-A11 | parallel strategy | With GQA/MQA/MLA, KV may be replicated if KV-head count is smaller than TP size. | KV-head count, TP size, attention family. | Replication is preferable/necessary to over-sharding. | Extra memory pressure and possible alternative DP policy. | Replicate per rank. | GQA/MQA/MLA; changes bytes/rank. |
| T-A12 | attention-DP + MoE EP | Attention and MoE can use different parallel decompositions; rank imbalance and EP communication can dominate iteration time. | TP/DP/EP configuration, per-rank workload. | Module-specific parallelism improves system utilization. | Layout choice is not usually a direct input to the MoE/ADP policy. | Balancing, disaggregation, alternate parallel configuration. | MoE+MLA/GQA; motivates RQ4. |

### Why T-A05 changes the scientific question

T-A05 means the statement “current frameworks do not use workload-aware heuristics” is false. TensorRT-LLM uses a real runtime performance proxy. Therefore RQ1 must distinguish:

- **persistent control-plane choice**: backend family, cache pool/layout, page organization;
- **local execution choice**: XQA vs masked MHA, multi-block vs single-block, context algorithm.

The research opportunity is whether the **first** layer should learn from the same kind of live variables already used by the **second** layer.

---

# A4. FlashInfer additions missing from v12

| ID | Decision site | Rule | Proxy actually used | Implicit assumption | Important ignored variables | Fallback / repair | Trigger / layout consequence |
|---|---|---|---|---|---|---|---|
| F-A01 | paged attention APIs | Layout is an explicit caller-visible NHD/HND contract. | Layout argument, tensor shape/strides. | System-level policy belongs upstream. | Global serving objective unless caller models it. | Invalid layout fails or wrapper-specific repair. | Makes layout policy explicit. |
| F-A02 | TRTLLM-gen NVFP4 wrappers | NHD can be accepted but transposed + materialized contiguously as HND, including scale tensors. | Backend, NVFP4 KV, input layout. | Conversion is preferable to rejection. | Reuse count, memory pressure, possibility of native HND allocation. | Automatic transpose/contiguous copy. | Concrete feasible-but-not-cost-equivalent case. |
| F-A03 | TRTLLM decode wrapper | Architecture/kernel availability can auto-select TRTLLM-gen or XQA. | SM generation, kernel availability. | Architecture predicts specialized kernel. | Batch/KV-length crossover if both feasible. | Compatible available backend. | Decode; backend can impose layout/page constraints. |
| F-A04 | decode API | Tensor-core path can be better for large GQA group size. | `Hq/Hkv` group size and option/policy. | GQA group size predicts arithmetic-intensity crossover. | Full-system contention and conversion. | Non-tensor-core path. | GQA; proves head ratio is a performance variable. |
| F-A05 | CUDA-graph wrappers | Graph-compatible wrappers can avoid shape-dependent kernel switching to preserve replay. | Graph mode/planned workspace metadata. | Graph-replay savings can dominate shape-specialization benefit. | Shape distribution and replan/recapture cost. | More flexible non-graph wrapper. | Decode/speculative; dynamic selection freedom is reduced. |
| F-A06 | CuTe DSL paged GQA | Some kernels can consume NHD-contiguous or HND-transposed/arbitrary supported strides without materialized conversion. | Strides and innermost head-dim condition. | Stride polymorphism can make multiple logical layouts cheap. | Cache/TLB efficiency of each stride pattern. | Another kernel or reject unsupported stride. | Counterexample to “layout mismatch always requires copy.” |
| F-A07 | specialized FMHA/XQA fast paths | Fast paths have conjunctions over dtype, head dimension, GQA ratio, page size, q_len and SM. | Geometry/dtype/page/q_len/SM. | Hard-coded region captures optimized envelope. | Continuous system state inside the region. | Alternate supported kernel. | MHA/GQA/speculative. |
| F-A08 | batch wrapper planning | Workspace/metadata planning and graph capture constrain what can change without re-planning. | Planned batch metadata, workspace, graph state. | Planning amortization is valuable. | Future queue composition. | Re-plan or use flexible wrapper. | Continuous batching; switching has temporal cost. |

---

# A5. Rule-level statistics after supplement

The exact number of raw source branches is not a scientifically stable metric because adjacent `if` statements may implement one policy and one branch can encode several assumptions. The stable unit used here is a **semantically distinct decision rule**.

The supplement adds:

- vLLM: 12 rule units;
- SGLang: 10 rule units;
- TensorRT-LLM: 12 rule units;
- FlashInfer: 8 rule units;

for **42 additional rule units** on top of the v12 ledger.

The dominant strategy classes across the combined audit are:

- compatibility / capability gates;
- fixed or piecewise platform/model priority;
- semantic phase routing;
- hand thresholds;
- user override;
- global supported-layout intersection;
- pool partitioning;
- feature disable/repair;
- explicit conversion;
- hard failure;
- local runtime performance heuristics;
- scheduling/parallelism repair.

The important scientific distinction is not the raw count. It is **which objective each class optimizes**:

- capability gates optimize correctness/feasibility;
- priority rules approximate performance using static proxies;
- local runtime heuristics optimize one kernel decision;
- conversion repairs enlarge feasibility;
- pool/layout rules determine persistent memory organization;
- system scheduling rules optimize overlap/balance.

A research selector must not compare these as if they were equivalent “heuristics.”

---

# B. Evidence chain condensed into revised research questions

## RQ1 — Persistent control-plane versus local adaptivity

> When choosing a persistent KV-cache layout and its consuming backend family, do current serving systems optimize a joint live workload/system cost, or do they mostly commit through capability gates, fixed/piecewise priorities and semantic routing, leaving workload-aware optimization to local kernels after backend/layout commitment?

This is the corrected RQ1. It survives the TensorRT-LLM counterexample because it no longer claims “no runtime heuristic exists.”

## RQ2 — Layout granularity

> Is one shared/whole-model persistent KV layout too coarse for heterogeneous layer groups, phases, sparse/dense consumers and state types? What is the smallest useful layout scope—whole model, cache pool, attention kind, layer group, phase or consumer—whose benefit exceeds metadata/fragmentation cost?

## RQ3 — Conversion-aware optimization

> When two consumers prefer different physical layouts, when is explicit conversion/packing cheaper than forcing both to use a common but locally inferior layout?

## RQ4 — GQA/MLA/MoE/parallelism coupling

> Does the optimal KV layout change when the objective includes KV replication/sharding, TP/DP/DCP, MoE EP All2All, overlap and rank imbalance rather than isolated attention-kernel latency?

## RQ5 — Speculative multi-consumer negotiation

> Should target, draft, verify and transfer consumers negotiate backend/layout jointly before cache allocation instead of selecting independently and repairing conflicts afterward?

No additional RQ is created for page size, graph capture, sparse locality or recurrent state. They are currently explanatory variables inside RQ1–RQ4; promoting them now would recreate the “many shallow questions” problem.

---

# C. RQ1 — full problem trial

## C1. Claim under trial

**RQ1-P:** The persistent backend/layout control-plane is predominantly feasibility + priority + semantic-rule driven. Genuine live workload heuristics exist, but they mostly operate after the persistent cache/backend family has been constrained.

A source counterexample that shows joint live-cost comparison of multiple persistent layouts **before allocation** would falsify this proposition.

## C2. Source-rule chain

### vLLM

The central automatic backend path validates candidates in an ordered platform/model priority list and chooses the first valid candidate. The input vector is rich—SM, dtype, head size, block size, MLA/sparse/non-causal/connector/parallel feature flags—but it is primarily a feasibility and static/piecewise policy vector.

The strongest evidence is not merely that the priority is fixed. V-A05 records a source-known MLA batch-size crossover while the priority branch itself does not observe live batch. That gives a direct, falsifiable gap between a known performance variable and the control-plane proxy.

After backend selection, the KV resolver:
1. collects backend-supported layouts;
2. intersects them;
3. may order common candidates by first-preference votes;
4. narrows mixed-spec layouts to block-compact forms;
5. applies user/connector preferences;
6. resolves one persistent layout for the whole model.

This is a compatibility aggregation algorithm, not a measured joint cost model.

### SGLang

SGLang already moves beyond a single static backend: it can select separate prefill/decode backends and routes online by semantic forward mode. That is meaningful adaptivity.

However, the router's primary proxy is **phase identity**, not a measured crossover for the current q_len/KV_len/batch. The page-size documentation independently exposes a performance tradeoff—prefix reuse versus kernel throughput—without resolving it via a universal live cost model.

### TensorRT-LLM — required counterevidence

TensorRT-LLM's XQA path is real workload-aware selection. Its performance proxy uses:
- KV-head count;
- batch size;
- history-length-derived multi-block count;
- SM count.

Generation multi-block logic likewise responds to available parallel work and shared-memory constraints.

Therefore the old binary hypothesis is rejected.

The refined hypothesis survives because these are **local execution decisions under an already established cache/backend environment**. The public top-level backend family and pool organization are not shown to run a general live optimization over all persistent layout/backend alternatives, conversion, prefix reuse and system contention.

### FlashInfer

FlashInfer reinforces the layer boundary. It exposes:
- explicit caller layout;
- local GQA-sensitive kernel choices;
- architecture-specific wrappers;
- graph/planning constraints;
- conversion or stride-polymorphic paths.

It is an optimized kernel substrate, not a universal cross-consumer persistent-layout allocator.

## C3. Proxy taxonomy

Current rules use four levels of proxy:

1. **Feasibility:** dtype, SM, head size, page/block size, feature support.
2. **Semantic:** prefill/decode, MLA/sparse, speculative top-k, model identity.
3. **Local performance:** batch, KV-heads, history length, SM count, GQA ratio.
4. **Global system:** prefix-reuse distribution, conversion bytes, target/draft interaction, remote transfer, MoE overlap, rank imbalance, allocator fragmentation.

The first three are present. The persistent layout control-plane sparsely uses the fourth.

## C4. Variables omitted by persistent layout arbitration

Workload:
- q_len and KV_len distributions rather than a mode flag;
- continuous-batching composition;
- prefix-hit probability, fanout and prefix length;
- speculative acceptance rate/draft depth;
- TTFT/ITL objective weighting.

Memory:
- measured HBM traffic;
- L2 hit/transaction efficiency;
- TLB/page-table locality;
- metadata bytes/useful KV byte;
- fragmentation.

Cross-consumer:
- expected reads before eviction;
- consumer-specific native-layout speedup;
- conversion/packing cost;
- transfer bandwidth/topology.

System:
- TP/DP/DCP replication/sharding;
- MoE All2All overlap;
- rank skew;
- copy-engine/HBM contention.

## C5. Framework fallback / repair comparison

| Repair type | vLLM | SGLang | TensorRT-LLM | FlashInfer |
|---|---|---|---|---|
| ordered next-candidate | yes | yes | path-specific | wrapper-specific |
| hard error | yes | yes | yes | yes |
| disable conflicting feature | yes | path-specific | path-specific | caller-level |
| semantic phase routing | composite/per-kind, less general | **explicit** | local algorithms | wrapper-level |
| live local performance heuristic | backend-dependent | backend-dependent | **explicit XQA/multi-block** | yes in selected wrappers |
| global layout intersection | **explicit** | common-pool compatibility rather than equivalent universal resolver | pool-specific | caller |
| separate geometry pools | limited by current global physical-layout resolver | pool implementation dependent | **explicit** | caller |
| materialized conversion | connector/integration-specific | path-specific | **explicit transfer formatter** | **explicit NVFP4 NHD→HND** |
| zero-copy/stride alternative | backend-specific | backend-specific | kernel-specific | **explicit in supported CuTe paths** |

## C6. Modern triggers

MHA:
large conventional KV-head count makes decode byte traffic important.

MQA/GQA:
define `g = Hq/Hkv`. Increasing `g` reduces persistent KV bytes but changes KV reuse and kernel parallelism. TensorRT-LLM uses KV-head count in XQA's runtime work estimate; FlashInfer exposes GQA-group-sensitive tensor-core behavior. Therefore GQA ratio must be a continuous experimental variable.

MLA:
latent KV changes bytes/token and geometry. Frameworks already separate MLA prefill/decode and sparse/dense paths; the optimum is not one universal kernel.

Sparse/DSA:
selected-page/token count and locality are runtime variables. “Sparse supported” does not determine useful-byte efficiency.

Hybrid GDN/linear attention:
persistent recurrent state coexists with KV, broadening layout optimization to heterogeneous state storage.

MoE:
does not alter attention mathematics but can alter system optimum via All2All, HBM pressure, overlap and rank imbalance.

## C7. Falsifiable hypotheses

**RQ1-H1 Priority inversion.** For a fixed GPU/model with two valid backends, there exists a workload region `(batch, q_len, KV_len, g, page_size)` where a lower-priority control-plane choice beats the default in end-to-end latency.

**RQ1-H2 Known crossover regret.** A live batch-conditioned choice beats the static ordering on a mixed-batch trace in a source branch where a batch crossover is already acknowledged.

**RQ1-H3 Local heuristic ≠ global optimum.** XQA's locally faster choice can coexist with a different persistent layout/backend family that yields lower full-token critical path after conversion/transfer/reuse are included.

**RQ1-H4 Vote weighting.** Runtime-weighting layout preference by executed KV bytes or measured time beats equal backend-first-choice voting in a heterogeneous model.

**RQ1-H5 Same-phase inversion.** Within one phase, varying batch/KV length/GQA ratio reverses the ordering of two valid kernels/layouts, showing semantic phase routing alone is insufficient.

**RQ1-H6 Graph regret.** Graph-stable fixed dispatch wins launch overhead but loses enough kernel time on a variable-shape trace to create a measurable crossover against more dynamic dispatch.

**RQ1-H7 Prefix reuse shift.** Optimal page/layout changes as prefix-hit rate/fanout changes with hardware/model held fixed.

**RQ1-H8 System shift.** The layout/backend minimizing isolated attention latency differs from that minimizing token critical path under KV transfer or MoE communication contention.

## C8. Minimum experiment

Sweep:
- Hopper and Blackwell minimum; Ampere/ROCm for external validity;
- MHA, at least two GQA ratios, MLA, sparse/DSA, MoE+GQA/MLA;
- prefill/decode/mixed/speculative verify;
- batch 1→64+;
- q_len from 1 to long prefill;
- KV_len logarithmic bins;
- prefix reuse 0/medium/high and fanout;
- all mutually supported pages/layouts;
- single GPU plus at least one distributed TP/ADP/EP setting.

Measure:
TTFT, ITL, tokens/s, kernel time, conversion time, HBM bytes/bandwidth, L2 hit, occupancy, graph overhead, fragmentation, transfer time, All2All and whole-token critical path.

## C9. RQ1 verdict

**SOURCE_CHAIN = CLOSED.**

The evidence supports the refined statement:

> Persistent backend/layout-family arbitration is primarily feasibility/priority/semantic-rule driven, while lower-level kernels can be genuinely workload-aware.

**EMPIRICAL_HYPOTHESIS = OPEN.**

No claim that a new selector is faster is considered proven until RQ1-H1…H8 are measured.

---

# D. RQ2 — full problem trial: layout granularity

## D1. Claim under trial

A whole-model/shared physical layout can be too coarse when consumers differ. A smaller layout scope—cache pool, attention group, phase or consumer—may win if its fragmentation/metadata/conversion overhead is smaller than the avoided layout regret.

## D2. All relevant source mechanisms

vLLM:
- one whole-model layout is explicitly resolved;
- selected backends' supported-layout sets are intersected;
- first-preference votes can order the common candidates;
- mixed H/N/C shapes force block-compact candidates;
- backend-per-kind still allows heterogeneous backends even while persistent physical layout remains global;
- composite backends share KV across distinct current-query kernels.

SGLang:
- prefill and decode can use different backends;
- HybridAttnBackend routes by phase;
- page-size tradeoff depends on prefix reuse;
- AITER vectorized-5D layout changes execution path;
- hybrid linear/GDN models introduce another persistent state type.

TensorRT-LLM:
- layers with different KV-head/window geometry can live in separate pools;
- sparse kernels can require different page/HND contracts;
- disaggregated endpoints can transform cache format.

FlashInfer:
- some consumers penalize NHD by materializing HND;
- other consumers can read alternative strides without a copy.

These are four real design points:
1. global intersection;
2. phase routing over shared state;
3. multi-pool partition;
4. conversion/stride polymorphism.

## D3. Current proxies for layout scope

Observed:
- geometry compatibility;
- KV-head count;
- attention window;
- backend support;
- semantic attention kind;
- page-size support;
- connector requirements.

Mostly omitted:
- reads per group;
- KV bytes per group;
- per-group performance delta;
- conversion frequency;
- prefix reuse by group;
- pool fragmentation;
- registration count;
- transfer locality.

## D4. Fallback/repair

vLLM:
common intersection or hard error.

SGLang:
phase-specific kernels but common compatible state; choose supported backend/page.

TensorRT-LLM:
split pools for incompatible geometry; conversion between disaggregated endpoints.

FlashInfer:
consume supported strides, convert, or choose another kernel.

## D5. Modern triggers

- GQA layers with differing KV-head counts;
- sliding-window + full-attention hybrids;
- MLA + conventional attention;
- DSA/sparse + dense layers;
- recurrent/linear state + KV;
- target/draft consumers;
- disaggregated prefill/decode endpoints.

## D6. Falsifiable hypotheses

**RQ2-H1 Per-group win.** Two layout pools outperform the best feasible common layout for a heterogeneous model while staying below a fixed memory-overhead budget.

**RQ2-H2 Weighted split criterion.** A split rule based on `reads_group × layout_regret - split_overhead` predicts when per-group layout wins.

**RQ2-H3 Phase-native persistence.** For long decode generations, decode-native persistent layout plus one prefill→decode format step beats a prefill-native common layout; for short outputs the result reverses.

**RQ2-H4 Window-specific policy.** Sliding-window/full-attention groups prefer different page/layout granularity because useful live KV length and reuse differ.

**RQ2-H5 Stride polymorphism shrinks need for multiple layouts.** If both consumers read the same allocation efficiently through different strides/views, splitting or copying provides no significant gain.

**RQ2-H6 Global layout is optimal in a low-heterogeneity regime.** At low reuse or small performance delta, extra pools/fragmentation lose. This negative hypothesis prevents bias toward always splitting.

## D7. Experiment

Compare:
1. current common layout;
2. per-attention-kind layout;
3. per-geometry-pool layout;
4. phase-native layout;
5. oracle per-consumer layout;
6. cost-model-selected scope.

Report:
latency/throughput **and** memory overhead, number of pools, fragmentation, registration count, prefix hit rate, conversion count/bytes and transfer bytes.

## D8. RQ2 verdict

**SOURCE_CHAIN = CLOSED.**

The sources prove the structural conflict and prove that production multi-pool/conversion alternatives exist.

**EMPIRICAL_HYPOTHESIS = OPEN.**

They do not prove per-group layout is always faster.

---

# E. RQ3 — full problem trial: conversion-aware optimization

## E1. Claim under trial

“Common layout or fail” is not the only feasible policy. Conversion is a measurable third action and should be selected by break-even cost.

## E2. Source mechanisms

FlashInfer:
NVFP4 TRTLLM-gen can accept NHD but materializes HND with transpose/contiguous copies, including scale tensors.

TensorRT-LLM:
disaggregated serving explicitly transforms producer/consumer cache/parallel formats and can overlap transfer with compute.

FlashInfer CuTe:
some kernels accept stride-polymorphic NHD/HND-style views without materialization.

vLLM:
connector preference is advisory; local common-layout compatibility can override transfer preference. Empty target/draft intersection can still become a hard boundary.

## E3. Cost equation

For producer layout `A` and consumer-native layout `B`:

`C_convert(A→B) + C_memory < R × [T_consume(A) - T_consume(B)] + ΔT_transfer`

where:
- `R` = expected number of consumer reads before eviction/replacement;
- `C_memory` = temporary/duplicate allocation + fragmentation cost;
- `ΔT_transfer` = transfer/packing advantage of B.

If transfer/conversion overlaps compute, compare **critical-path** cost rather than simply summing wall-clock components.

## E4. Ignored variables in rule-based repair

- reuse count and eviction probability;
- temporary-memory headroom;
- link bandwidth/topology;
- copy-engine overlap;
- phase duration;
- graph disruption/re-planning;
- zero-copy stride alternative.

## E5. Falsifiable hypotheses

**RQ3-H1 Break-even reuse exists.** There is a measurable `R*` beyond which one conversion to a native consumer layout beats repeated non-native consumption.

**RQ3-H2 Network changes R*.** RDMA/NVLink/disaggregated transfer shifts the threshold.

**RQ3-H3 Strided view region.** A supported strided view beats materialization until cache/TLB inefficiency exceeds the one-time copy.

**RQ3-H4 Memory-pressure reversal.** Temporary duplicate layout can reduce cache capacity enough to erase kernel speedup.

**RQ3-H5 Long-decode asymmetry.** Conversion amortizes for long generation but often loses for short outputs.

## E6. RQ3 verdict

**SOURCE_CHAIN = CLOSED.**

Conversion, overlap and no-copy alternative layouts are production mechanisms.

**EMPIRICAL_HYPOTHESIS = OPEN.**

The break-even surface is not yet established.

---

# F. RQ4 — full problem trial: GQA/MLA/MoE/parallelism coupling

## F1. Claim under trial

Isolated attention-kernel latency is an incomplete layout objective once KV ownership/replication and MoE/network overlap enter the critical path.

## F2. Source mechanisms

GQA/MQA:
TensorRT-LLM XQA uses KV-head count in its runtime work estimate. Under TP, KV may be replicated if KV-head count is below TP size. Thus `Hkv` changes both local execution and bytes per rank.

MLA:
vLLM/SGLang already have separate MLA prefill/decode/sparse paths. TensorRT-LLM can use attention TP/DP and MLA graph-overlap transformations. MLA changes representation and scheduling.

MoE:
vLLM DBO exists specifically to overlap MoE sparse All2All with surrounding compute and uses separate decode/prefill token thresholds. TensorRT-LLM allows attention TP/DP while MoE uses EP/ETP.

Rank imbalance:
attention-DP systems can become limited by the slowest rank when prefill/decode loads differ.

## F3. Proxies currently used

- KV-head count;
- TP/DP/EP sizes;
- token thresholds for DBO;
- phase mix;
- All2All backend;
- per-rank workload for balancing.

## F4. Variables not normally fed back to layout choice

- EP All2All bytes;
- expert skew;
- overlap window;
- per-rank phase skew;
- replicated KV bytes;
- HBM contention;
- copy-engine contention;
- critical-path maximum across ranks.

## F5. Framework fallback/repair

vLLM:
choose All2All backend, enable DBO above thresholds, use DP+EP constraints.

TensorRT-LLM:
choose attention TP vs DP, MoE TP/EP/ETP, balancing/disaggregation strategies.

These solve scheduling/parallelism problems but are not generally a persistent layout optimizer.

## F6. Falsifiable hypotheses

**RQ4-H1 System optimum differs.** Under EP All2All overlap, the isolated-attention winner is not always the whole-token critical-path winner.

**RQ4-H2 GQA replication shift.** When `Hkv < TP`, replicated KV memory pressure changes preferred layout/page/parallel strategy relative to single-GPU execution.

**RQ4-H3 DBO shift.** Enabling MoE overlap moves the batch/KV-length crossover between two attention layouts/backends.

**RQ4-H4 Rank-aware objective.** Minimizing maximum rank time chooses a different layout than minimizing mean local attention time in heterogeneous attention-DP workloads.

**RQ4-H5 Conversion overlap reversal.** A conversion can be nearly free if hidden under a communication stall, or more expensive if competing for the same HBM/copy resource.

## F7. RQ4 verdict

**SOURCE_CHAIN = CLOSED.**

The coupling mechanisms are documented.

**EMPIRICAL_HYPOTHESIS = OPEN.**

The causal effect of layout on system optimum must be isolated experimentally.

---

# G. RQ5 — full problem trial: speculative multi-consumer negotiation

## G1. Claim under trial

Target, draft, verify and transfer consumers should be treated as one constraint/cost graph before persistent cache allocation.

## G2. Source chain

vLLM:
target and draft can independently induce backend/layout constraints; the whole-model intersection can become empty. A current issue documents initialization failure rather than a joint second-best choice.

SGLang:
target verify routing depends on speculative-attention mode; page/top-k/backend constraints exist; a draft worker may force one draft backend for both prefill and decode. Recent FP4/speculative issue evidence shows compatibility also includes metadata/phase-native requirements, not just axes.

FlashInfer:
speculative/multi-query-token fast paths have explicit q_len/page/layout/geometry constraints.

Thus the consumer set is at least:

`target decode, target verify, draft extend/prefill, draft decode, KV writer, optional transfer endpoint`.

## G3. Variables ignored by independent local selection

- acceptance rate;
- draft length;
- target/draft reuse;
- verify q_len;
- read frequency by consumer;
- conversion cost;
- shared versus separate cache capacity;
- graph compatibility;
- host/GPU metadata contracts.

## G4. Repair alternatives

1. common intersection;
2. change one consumer to second-best backend;
3. separate target/draft pools;
4. boundary conversion;
5. shared stride-polymorphic storage;
6. reject configuration.

The research problem is to pick among 1–6 by cost rather than hard-coded precedence.

## G5. Falsifiable hypotheses

**RQ5-H1 Feasible joint repair.** Some configurations that fail under independent choices are feasible with a jointly selected second-best backend pair and outperform disabling speculation.

**RQ5-H2 Acceptance-rate shift.** Low acceptance increases verify weight enough to change optimal backend/layout; high acceptance favors draft-side optimization.

**RQ5-H3 Separate-pool win.** Strong target/draft geometry or dtype asymmetry makes separate layouts faster within a fixed memory premium.

**RQ5-H4 Metadata-aware solver.** A solver that checks only layout/page shape still fails on paths where host/GPU metadata requirements differ.

## G6. RQ5 verdict

**SOURCE_CHAIN = CLOSED.**

Independent multi-consumer selection can create compatibility/repair boundaries.

**EMPIRICAL_HYPOTHESIS = OPEN.**

---

# H. Unified model after serial adjudication

The five RQs reduce to one constrained optimization problem.

For consumers `c` (prefill, decode, sparse, target, draft, transfer, etc.) and persistent layout scopes `s`, choose:

- physical layout `L_s`;
- backend/kernel family `B_c`;
- page size `P_s`;
- optional conversion edge `X`;
- parallel/scheduling policy `Q`;

to minimize an end-to-end objective such as:

`J = α·TTFT + β·ITL_p99 - γ·throughput + λ·memory + μ·network_critical_path`

subject to correctness/support, memory capacity, graph constraints, target/draft compatibility, and distributed ownership/replication.

A measurable decomposition is:

`J ≈ Σ_c freq_c · T_c(B_c, L_s, P_s, workload_c)`
`  + Σ_edges C_convert`
`  + C_metadata + C_fragmentation`
`  + C_transfer_critical`
`  + C_MoE/parallel_interference`
`  + C_rank_imbalance`.

Current frameworks cover many constraints and several local terms. The research opportunity is to estimate enough cross-consumer/global terms to choose persistent layout scope and conversion intentionally.

---

# I. Required experimental baselines and ablations

Baselines:
1. framework default;
2. forced each feasible backend/layout;
3. local kernel heuristic intact;
4. layout-only oracle;
5. backend-only oracle;
6. joint offline oracle;
7. proposed cost model.

Ablations:
- remove prefix-reuse term;
- remove conversion term;
- remove GQA-ratio term;
- remove transfer term;
- remove MoE/parallel contention;
- force global layout;
- disable runtime re-selection.

Generalization:
- train/tune on one request trace, test different context distribution;
- different concurrency;
- different GQA/MLA model;
- different GPU generation;
- speculation on/off;
- disaggregated serving on/off;
- MoE EP on/off.

Required negative results:
- regimes where framework default already wins;
- dynamic selection overhead does not amortize;
- multi-layout loses to fragmentation;
- conversion loses under memory pressure;
- global layout is preferable.

A result that reports only positive regimes is not a falsifiable validation of the research thesis.

---

# J. Completeness status after v13 supplement

| Dimension | vLLM | SGLang | TensorRT-LLM | FlashInfer |
|---|---|---|---|---|
| central backend selection | closed for audited public path | closed for audited public path | closed at documented PyTorch/backend layer | not a global serving selector; wrapper layer closed |
| platform/model special cases | closed for audited standard/MLA/sparse/composite paths | closed for MHA/MLA/DSA/GDN paths | closed for audited attention/XQA/sparse paths | closed for audited wrappers |
| phase routing | MLA/composite/per-kind captured | **HybridAttnBackend + speculative routing captured** | local phase algorithms captured | wrapper-level captured |
| persistent layout/page policy | global resolver captured | support/tradeoff/vectorized path captured | multi-pool/sparse constraints captured | NHD/HND/stride contracts captured |
| runtime workload heuristic | top-level gap + known static crossover captured | semantic routing and graph policy captured | **XQA/multi-block counterexample captured** | GQA/graph/local policy captured |
| fallback/error | captured at audited decision sites | captured | captured | captured |
| conversion/no-copy alternative | connector side captured | path-specific | disaggregated formatter captured | materialized + stride-polymorphic captured |
| speculative multi-consumer | target/draft conflict captured | routing/draft/FP4 boundary captured | relevant wrapper/local constraints captured | geometry/layout fast paths captured |
| MoE/system coupling | DBO/EP captured in RQ4 | treated as system variable, no unsupported layout claim | ADP+EP/ETP captured | not a MoE scheduler |

**Overall source status:** within the declared public-source scope, the evidence chain is now sufficiently complete to close RQ1–RQ5 as scientific problem statements.

**Not claimed:** empirical superiority of any new layout selector.

---

# K. Source additions used by v13 supplement

vLLM:
- https://docs.vllm.ai/en/latest/design/attention_backends/
- https://github.com/vllm-project/vllm/blob/main/vllm/platforms/cuda.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/selector.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/utils.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/kv_cache_layout.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/kv_connector_model_runner_mixin.py
- https://docs.vllm.ai/en/latest/design/dbo/
- https://docs.vllm.ai/en/latest/api/vllm/config/parallel/
- https://docs.vllm.ai/en/latest/design/moe_kernel_features/
- https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/

SGLang:
- https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/attention_backend.mdx
- https://docs.sglang.io/docs/advanced_features/attention_backend
- https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/attention_backend_setup.py
- https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/aiter_backend.py
- https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/kvcache/cache_ops.py

TensorRT-LLM:
- https://nvidia.github.io/TensorRT-LLM/latest/torch/attention.html
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/xqa/decoderXQARunner.h
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/xqa/decoderXQARunner.cpp
- https://nvidia.github.io/TensorRT-LLM/1.2.0/features/parallel-strategy.html
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/sparse-attention.md
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/disagg-serving.md

FlashInfer:
- https://docs.flashinfer.ai/api/attention.html
- https://docs.flashinfer.ai/generated/flashinfer.prefill.trtllm_batch_context_with_kv_cache.html
- https://docs.flashinfer.ai/generated/flashinfer.decode.trtllm_batch_decode_with_kv_cache.html
- https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/decode.py

---

# L. Merge rule when v11 becomes available

Do not simply paste this supplement after v11. Merge by evidence identity:

1. Keep v11 bibliography/history.
2. Replace any binary RQ1 wording (“heuristic vs workload-aware”) with the refined **persistent-control-plane vs local-adaptivity** wording.
3. Preserve contradictory evidence; especially do not delete TensorRT-LLM XQA because it weakens the original claim.
4. For every rule require:
   `decision site → rule → proxy → assumption → ignored vars → fallback/repair → modern trigger → layout consequence → falsifiable hypothesis`.
5. Maintain two independent status fields:
   - `SOURCE_CHAIN`
   - `EMPIRICAL_HYPOTHESIS`
6. A new RQ is permitted only when a source/measured causal mechanism cannot be represented as a variable or hypothesis of RQ1–RQ5.

This is the guardrail against returning to “many questions, insufficient evidence depth.”

# 3. v14 temporal-commitment rules

| ID | Framework | Strategy | Rule | Proxy | Assumption | Ignored variables | Fallback/repair | Trigger | Layout consequence |
|---|---|---|---|---|---|---|---|---|---|
| V-A13 | vLLM | one-time layout commitment before memory profiling | Engine core resolves KV layout before memory profiling/full graph-related KV initialization; record_kv_cache_layout refuses changing an already-resolved layout. | startup model/backend supported-layout set, cache specs, connector/env preference | startup-resolved layout remains suitable for the serving epoch | later workload drift: batch, phase mix, prefix hit rate, transfer/MoE pressure | restart/reinitialize/reallocate is effectively required for a persistent-layout change | all KV models; especially long-running mixed traffic | persistent physical layout becomes an epoch-level decision |
| S-A11 | SGLang | startup backend/page policy + graph commitment | Attention backend/page size are ServerArgs launch choices; hybrid backend is constructed from resolved prefill/decode choices; decode backend is CUDA-graph captured in documented hybrid/speculative paths. | startup hardware/model/config, page size, phase backend names, graph mode | deployment-time configuration remains appropriate over later traffic | diurnal batch/KV distribution, prefix reuse drift, speculation acceptance drift | change launch config/restart; non-graph or differently captured path where available | continuous serving, hybrid/speculative attention | persistent page/backend policy has switching/recapture cost |
| T-A13 | TensorRT-LLM | KV block geometry fixed in model/executor configuration | tokens_per_block is model/executor/KV-manager configuration; manager construction receives fixed tokens_per_block and geometry. Legacy documentation also describes block-size selection at build time. | build/launch config, attention/KV geometry | configured block size is adequate for the deployment workload | runtime prefix-reuse distribution, workload phase drift, current fragmentation | rebuild/recreate executor/manager/configure another run | paged attention, cache reuse, MLA paths that may force a block size | page granularity is persistent over a serving epoch |
| F-A09 | FlashInfer | wrapper planning / CUDA-graph lifetime stability | Batch decode wrapper pre-plans metadata/workspace; with CUDA Graph enabled the wrapper documents that batch_size cannot change during its lifecycle. | planned batch metadata, workspace, graph flag, configured layout | stable shape/planning amortization outweighs per-shape re-selection | future queue/workload drift and layout crossover | re-plan/recreate wrapper or use non-graph mode | graph-captured decode/speculative serving | dynamic layout/kernel switching has explicit temporal re-planning cost |

# 4. Reverse coverage audit: every stable evidence rule must terminate in an RQ/hypothesis

| Evidence ID | RQ coverage | Hypothesis coverage |
|---|---|---|
| F-A01 | RQ2,RQ3 | RQ2-H5; RQ3-H3 |
| F-A02 | RQ3 | RQ3-H1/H4/H5 |
| F-A03 | RQ1,RQ3 | RQ1-H1; RQ3-H1 |
| F-A04 | RQ1,RQ4 | RQ1-H5; RQ4-H2 |
| F-A05 | RQ1,RQ6 | RQ1-H6; RQ6-H2/H3 |
| F-A06 | RQ2,RQ3 | RQ2-H5; RQ3-H3 |
| F-A07 | RQ1,RQ2,RQ5 | RQ1-H5; RQ2-H3; RQ5-H1 |
| F-A08 | RQ1,RQ6 | RQ1-H6; RQ6-H2/H3 |
| F-A09 | RQ6 | RQ6-H2/H3 |
| F-B1 | RQ1,RQ4 | RQ1-H5; RQ4-H2 |
| F-L1 | RQ2,RQ3 | RQ2-H5; RQ3-H3 |
| F-L2 | RQ3 | RQ3-H1/H4/H5 |
| S-A01 | RQ1,RQ2 | RQ1-H5; RQ2-H3 |
| S-A02 | RQ1,RQ2,RQ5 | RQ1-H5; RQ2-H3; RQ5-H2 |
| S-A03 | RQ1 | RQ1-H1 |
| S-A04 | RQ1,RQ5 | RQ1-H5; RQ5-H2 |
| S-A05 | RQ2,RQ3 | RQ2-H4; RQ3-H1 |
| S-A06 | RQ5 | RQ5-H1/H2 |
| S-A07 | RQ2,RQ5 | RQ2-H3; RQ5-H3 |
| S-A08 | RQ1,RQ2,RQ4 | RQ1-H5; RQ2-H1; RQ4-H1 |
| S-A09 | RQ2,RQ3 | RQ2-H5; RQ3-H3 |
| S-A10 | RQ2,RQ4 | RQ2-H1; RQ4-H1 |
| S-A11 | RQ6 | RQ6-H1/H2/H3 |
| S-B1 | RQ1 | RQ1-H1/H5 |
| S-B2 | RQ1,RQ5 | RQ1-H5; RQ5-H2 |
| S-B3 | RQ1 | RQ1-H1 |
| S-B4 | RQ1,RQ2 | RQ1-H1; RQ2-H3 |
| S-B5 | RQ1,RQ2 | RQ1-H1; RQ2-H3 |
| S-B6 | RQ1 | RQ1-H1 |
| T-A01 | RQ1 | RQ1-H1/H3 |
| T-A02 | RQ2 | RQ2-H1/H6 |
| T-A03 | RQ1,RQ2 | RQ1-H5; RQ2-H3 |
| T-A04 | RQ1 | RQ1-H3 |
| T-A05 | RQ1,RQ4 | RQ1-H3/H5; RQ4-H2 |
| T-A06 | RQ1,RQ4 | RQ1-H3; RQ4-H1 |
| T-A07 | RQ2,RQ3,RQ4 | RQ2-H1; RQ3-H1; RQ4-H1 |
| T-A08 | RQ3,RQ4 | RQ3-H2; RQ4-H1 |
| T-A09 | RQ3,RQ4 | RQ3-H2; RQ4-H5 |
| T-A10 | RQ4 | RQ4-H1/H4 |
| T-A11 | RQ4 | RQ4-H2 |
| T-A12 | RQ4 | RQ4-H1/H4/H5 |
| T-A13 | RQ6 | RQ6-H1/H4/H5 |
| T-B1 | RQ1 | RQ1-H1/H3 |
| V-A01 | RQ1,RQ2 | RQ1-H5; RQ2-H5 |
| V-A02 | RQ1 | RQ1-H1 |
| V-A03 | RQ1,RQ4 | RQ1-H1; RQ4-H2 |
| V-A04 | RQ1,RQ4 | RQ1-H5; RQ4-H2 |
| V-A05 | RQ1 | RQ1-H2 |
| V-A06 | RQ1,RQ2 | RQ1-H1/H5; RQ2-H4 |
| V-A07 | RQ1,RQ2 | RQ1-H5; RQ2-H3 |
| V-A08 | RQ1,RQ4,RQ5 | RQ1-H1; RQ4-H1; RQ5-H4 |
| V-A09 | RQ3,RQ4 | RQ3-H2; RQ4-H1 |
| V-A10 | RQ2,RQ3,RQ4 | RQ2-H1; RQ3-H2; RQ4-H1 |
| V-A11 | RQ1,RQ2 | RQ1-H7; RQ2-H6 |
| V-A12 | RQ5 | RQ5-H1 |
| V-A13 | RQ6 | RQ6-H1/H4 |
| V-B1 | RQ1 | H1/H2 |
| V-B2 | RQ1 | H1/H2 |
| V-B3 | RQ1,RQ2 | RQ1-H5; RQ2-H1 |
| V-L1 | RQ1,RQ2 | RQ1-H4; RQ2-H1 |
| V-L2 | RQ2,RQ3 | RQ2-H1/H6; RQ3-H1 |
| V-L3 | RQ1,RQ2 | RQ1-H4; RQ2-H2 |
| V-L4 | RQ2,RQ6 | RQ2-H1/H6; RQ6-H1/H4 |
| V-L5 | RQ2 | RQ2-H1/H4 |
| V-L6 | RQ3,RQ4 | RQ3-H2/H5; RQ4-H1 |

## 4.1 Orphan-evidence test

For the 65 stable rule IDs in the current ledger, **no rule is intentionally left without an RQ mapping**.

This is stronger than merely listing RQs: it checks that a source rule is actually “consumed” by a scientific question.

## 4.2 Unsupported-RQ test

Each top-level RQ is supported by source mechanisms from at least two frameworks/kernel stacks:

- RQ1: all four;
- RQ2: all four;
- RQ3: vLLM + SGLang + TensorRT-LLM + FlashInfer;
- RQ4: vLLM + SGLang + TensorRT-LLM + FlashInfer kernel signals;
- RQ5: vLLM + SGLang + FlashInfer/consumer constraints;
- RQ6: all four.

This does not prove each hypothesis true; it prevents an RQ from existing only because of speculative prose.


# M. RQ6 — full problem trial: static commitment versus workload drift

## M1. Why RQ6 is independent rather than another RQ1 sub-hypothesis

RQ1 asks **what variables are used when the control-plane chooses**. RQ6 asks a different causal question: **when is that choice allowed to change again?**

The current source chain shows persistent configuration at an epoch/startup boundary:

- vLLM resolves KV layout in the engine core before memory profiling / full cache initialization and records the result; attempting to record a different already-resolved layout is rejected.
- SGLang exposes attention backend and page size as launch configuration; hybrid prefill/decode backends are constructed from resolved startup choices, and documented CUDA-graph behavior captures the decode backend.
- TensorRT-LLM configures `tokens_per_block` and cache-manager geometry when building/creating the model executor; block size is not a per-request adaptive variable.
- FlashInfer batch wrappers plan workspace/metadata, and with CUDA Graph enabled document that the wrapper's batch size cannot change during its lifecycle.

This is a distinct **temporal granularity** problem. A control-plane may use the right variables at `t0` and still become suboptimal at `t1` when traffic changes.

## M2. Source rules

| ID | Framework | Persistent decision | Proxy at commitment | Ignored future variables | Current repair |
|---|---|---|---|---|---|
| V-A13 | vLLM | one whole-model KV layout is resolved/recorded before memory profiling | model/backends/specs/env/connector at startup | later batch, phase mix, prefix reuse, transfer/MoE contention | restart/reinitialize/reallocate |
| S-A11 | SGLang | backend/page/hybrid configuration is launch-time; graph captures execution path | model/hardware/server args/phase config | traffic drift, prefix-hit drift, acceptance-rate drift | relaunch, recapture/change mode where supported |
| T-A13 | TensorRT-LLM | tokens-per-block and manager geometry are executor/model configuration | build/launch config and cache geometry | runtime reuse distribution, fragmentation and phase mix | recreate/reconfigure executor/model |
| F-A09 | FlashInfer | planned wrapper / CUDA-graph state has lifetime stability constraints | planned metadata, graph flag, batch/layout config | future queue shape and crossover | re-plan/recreate wrapper or non-graph path |

## M3. Proxy and assumption trial

The common proxy is a **deployment snapshot**:
`hardware + model geometry + launch configuration + supported backend set`.

The common assumption is temporal stationarity:
the relative ranking of persistent layout/page/backend choices will remain good enough for the serving epoch.

Variables omitted by that assumption:
- diurnal/request-source changes in concurrency;
- prompt/decode length distribution drift;
- prefix reuse/fanout drift;
- speculative acceptance-rate drift;
- disaggregated-transfer fraction drift;
- MoE/communication pressure drift;
- memory fragmentation and cache occupancy evolution.

## M4. Framework fallback/repair

Current systems mostly repair temporal mismatch at a coarse boundary:

- restart/recreate the serving engine with another layout/page/backend;
- re-plan/recreate kernel wrappers;
- leave graph mode for a more flexible path;
- use local kernel adaptation while leaving persistent layout unchanged.

This differs from RQ3 conversion: conversion repairs **consumer-format mismatch**. RQ6 repairs **time-varying optimality**.

## M5. Modern-attention triggers

GQA:
traffic drift changes batch/KV-length distribution while `Hq/Hkv` stays fixed, so a fixed persistent layout can cross a kernel/layout performance boundary without model changes.

MLA:
prefill/decode ratio can change over time. A deployment that was prefill-heavy at startup can become decode-heavy while retaining the same persistent page/layout choice.

Sparse/DSA:
selected-token density/locality can drift with task/domain mix.

MoE:
expert/communication contention can change with batch composition and parallel load, changing whether attention should favor bandwidth, overlap or conversion.

Speculative decoding:
acceptance rate and draft depth are workload/model-state dependent and can change the weight of draft versus verify consumers.

## M6. Falsifiable hypotheses

**RQ6-H1 Workload-drift regret.** A layout/page/backend configuration chosen for trace A becomes measurably suboptimal on trace B without any model or hardware change.

**RQ6-H2 Hysteresis beats per-request switching.** An epochal reconfiguration policy with hysteresis outperforms both a permanently fixed policy and per-request switching after accounting for migration/replan overhead.

**RQ6-H3 Graph recapture threshold.** There exists a minimum expected duration/volume of the new workload regime beyond which graph recapture/replanning is amortized.

**RQ6-H4 Layout-migration threshold.** Repacking live KV state into a new persistent layout is worthwhile only above a threshold in remaining expected reads / requests / serving-epoch duration.

**RQ6-H5 Stable-workload negative result.** Under stationary traffic, reconfiguration overhead is not repaid and a fixed configuration remains optimal or statistically indistinguishable.

## M7. Minimum experiment

Create two or more traffic regimes on the same model/GPU:
- short-prompt/high-concurrency decode-heavy;
- long-prefill/low-concurrency;
- high prefix reuse versus low reuse;
- speculation high-acceptance versus low-acceptance;
- with and without MoE/transfer contention where applicable.

Compare:
1. fixed startup default;
2. oracle fixed configuration for each full trace;
3. epochal reconfiguration;
4. per-request/local-only adaptation;
5. no-graph versus graph-recapture variants.

Measure the switching cost explicitly:
KV migration bytes/time, graph recapture or wrapper re-plan time, temporary memory, request pause/stall, and post-switch amortization horizon.

## M8. RQ6 verdict

**SOURCE_CHAIN = CLOSED.**

The source evidence is sufficient to establish that persistent layout/page/backend choices have a temporal commitment boundary and that local dynamic kernel policy does not imply persistent-layout reconfiguration.

**EMPIRICAL_HYPOTHESIS = OPEN.**

No claim is made that online reconfiguration is normally worthwhile.



# 5. New current-source evidence added in v14

## vLLM
Current source documents `resolve_kv_cache_layout` as resolving one layout for the whole model, with an already-recorded layout winning and disagreement/error checks. Engine core resolves the layout before memory profiling and propagates it to workers.

Sources:
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/utils.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core.py
- https://docs.vllm.ai/en/latest/design/attention_backends/

## SGLang
Current attention-backend documentation confirms:
- native page-size differences;
- page-size=1 maximizes prefix reuse while larger pages usually improve attention-kernel performance;
- hybrid prefill/decode backends;
- speculative top-k/page/backend constraints;
- CUDA-graph capture rules.

Sources:
- https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/attention_backend.mdx
- https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/arg_groups/speculative_hook.py

## TensorRT-LLM
Current source/docs show fixed cache-manager geometry/tokens-per-block configuration and production code that can force/adjust tokens-per-block for specific MLA backends, demonstrating that page granularity is a persistent compatibility parameter rather than a free per-request variable.

Sources:
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/torch/kv_cache_manager.md
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/pyexecutor/py_executor_creator.py
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/include/tensorrt_llm/runtime/modelConfig.h

## FlashInfer
Current decode API documents:
- NHD/HND caller layout;
- GQA tensor-core advantage at large group size;
- wrapper planning;
- CUDA-graph wrapper lifetime constraint that batch size cannot change.

Source:
- https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/decode.py

# 6. What is still required before writing “all heuristics in the repositories are exhaustively audited”

The remaining task is forensic rather than conceptual:

1. pin every rule to `repository + immutable commit SHA + file + function + line range`;
2. enumerate backend registration and support-predicate entrypoints mechanically for each repository;
3. diff those entrypoints against the 65 semantic rules;
4. inspect optional platform/plugin branches (ROCm/NPU/XPU/CPU) only when they can change the scientific conclusion or claimed scope;
5. rerun the coverage matrix after every upstream version bump.

Until that is done, the correct wording is:

> “source-chain complete within the declared public LLM/KV decision-path scope,”

not:

> “every `if` statement in all four repositories has been proven exhaustively covered.”

# 7. Canonical RQ list after v14

1. **RQ1 — Persistent control-plane versus local adaptivity**
2. **RQ2 — Layout granularity**
3. **RQ3 — Conversion-aware optimization**
4. **RQ4 — GQA/MLA/MoE/parallelism coupled objective**
5. **RQ5 — Speculative multi-consumer negotiation**
6. **RQ6 — Static commitment versus workload drift**

No other mechanism is promoted to a top-level RQ in v14. Page size, prefix reuse, graph capture, sparse locality, generalized axis permutation, recurrent state and topology remain variables/sub-hypotheses unless future evidence shows an independent causal mechanism.

