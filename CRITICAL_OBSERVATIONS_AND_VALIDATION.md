# Critical observations behind the layout RQs

This note records the cross-document observations that are important precisely
because no single framework source is expected to state them as a global
claim.  It also defines what the experiment may and may not prove.

## Evidence rule

“Present in most frameworks” has to use an applicable-layer denominator.
SGLang/vLLM-style persistent-cache decisions and CUTLASS/Triton-style
thread-value layouts are not interchangeable votes.  The audit therefore uses:

- serving: vLLM, SGLang, TensorRT-LLM, FlashInfer; majority = 3/4;
- compiler/kernel: CUTLASS, Triton, TVM, IREE, MLIR; majority = 3/5;
- cross-layer: all nine audited systems; majority = 5/9.

Hexcute is retained as an important synthesis baseline but is not silently
counted as an A10 native result: its public artifact targets A100/H100.  An
unsupported run cannot be converted into either a positive or negative vote.

For every observation the report emits three independent verdicts:

1. **Source-majority**: pinned official code exposes the component rules from
   which the observation is inferred.
2. **Controlled-causal**: a paired, correctness-checked counterfactual changes
   only the decision in question and improves more than measurement noise.
3. **Native-majority**: the same counterfactual improves by more than 1% in a
   majority of applicable native frameworks.

Only the third verdict supports the statement “this decision improves most
frameworks.” Source-majority alone proves prevalence of the limitation, not a
performance gain.

## Observation map

The documents use two incompatible RQ numberings.  The executable catalog uses
the v14 serving RQ1--RQ6 followed by the v4 compiler RQ7--RQ10; it does not
silently equate those numbers with the six RQs in
`FRAMEWORK_COMPARISON_AND_RESEARCH_QUESTIONS_V2.md`.  Their semantic crosswalk
is:

| V2 question | Executable observation(s) |
|---|---|
| V2-RQ1 local versus subgraph optimum | L-RQ1 edge inversion; canonical RQ8 non-commutative search |
| V2-RQ2 persistent versus internal two-level layout | canonical RQ1 two-level control; RQ2 adaptive domains |
| V2-RQ3 representation/metadata/layout interaction | canonical RQ3 conversion investment; RQ7 typed edge contract |
| V2-RQ4 cross-framework relation model | canonical RQ7 typed edge contract; RQ10 legality-first repair |
| V2-RQ5 constraint plus measurement-efficient selection | canonical RQ9 residual meta-policy |
| V2-RQ6 diffusion lifecycle | canonical RQ3 reuse break-even and RQ6 epochal state; visual external-validity run remains separate |

This crosswalk is semantic, not an assertion that one experiment fully closes
both questions. In particular, V2-RQ4 still needs CuTe/LinearLayout/IndexMap/
Hexcute importer coverage, V2-RQ5 needs regret-versus-trial-budget curves, and
V2-RQ6 needs the existing modern visual suite in addition to the LLM run.

| RQ | Important inferred observation | Decision tested | Primary discriminator |
|---|---|---|---|
| L-RQ1 | producer-local winner can lose on the complete producer→consumer edge | direct native emission/view/copy/no-copy edge oracle | ranking inversion with five-process bootstrap CI |
| RQ1 | persistent commitment and live kernel adaptation are two different control layers | persistent selector plus existing local expert | mixed-shape regret versus one fixed priority |
| RQ2 | profitable layout scope is conditional, neither always global nor always split | global/group/phase domains under memory budget | positive heterogeneous and negative homogeneous regimes |
| RQ3 | conversion is a reusable investment, not a per-consumer constant penalty | view/materialize-once/producer-native/no-conversion | reuse break-even and memory-pressure reversal |
| RQ4 | token critical path/max-rank time can reverse isolated-kernel ranking | contention/rank-aware objective | isolated versus concurrent HBM pressure; multi-GPU extension remains open |
| RQ5 | independently legal consumer choices can have an empty joint intersection | joint pair selection plus bounded repair/separate pool | common, repaired, and feature-disable baselines |
| RQ6 | layout choice is stateful under drift, but switching has migration cost | epochal hysteresis | fixed versus per-request versus hysteretic trace policy |
| RQ7 | shape/stride is not a complete boundary contract | typed axes/dtype/packing/page/metadata/lifetime/ownership contract | legality-first intersection and zero-copy/materialized repair |
| RQ8 | fusion/pass order/layout search is non-commutative | joint legal partial-order search | RoPE-before-convert, convert-before-RoPE, fused native layout |
| RQ9 | mature heuristics are useful experts but non-portable global policies | residual meta-policy over legal framework candidates | static regret versus per-shape oracle; held-out GPU is mandatory |
| RQ10 | direct mismatch is not impossibility, but repair must be bounded | legality-first shortest repair path | reject, transpose/pack, producer-native, over-budget negative control |

## Runtime interpretation

The canonical runner selects min/median/max contracts from all common LLM
subgraph/phase groups in the pinned 640-case catalog.  Its explicit Triton KV
sweep additionally covers MHA, MQA and GQA shapes and compares NHD, HND,
paged-NHD and paged-HND with identical arithmetic.  The v10 runner is a stricter
L-RQ1 test and must not be treated as eleven independent RQs.

Native artifact extraction is conservative:

- CUTLASS/CUDA softmax rows contribute only when correctness passes and a
  paired boundary strategy changes by more than 1%;
- Triton contributes to RQ9 only when the winning physical layout differs
  across cases, rather than merely showing that one fixed layout is fastest;
- vLLM contributes only for `comparison_scope=layout_only` rows;
- SGLang backend+page experiments are excluded from layout-only causal claims
  because both variables change together.

The generated report grades each observation A/B/C/D/OPEN.  A is deliberately
hard: it requires native-majority causal reproduction.  B is the appropriate
claim when official-source prevalence and a controlled mechanism are both
validated but native framework coverage remains incomplete.
