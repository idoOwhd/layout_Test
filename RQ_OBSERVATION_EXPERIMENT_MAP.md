# RQ / observation experiment map

## Claim rule

Four statements are reported independently:

1. **Source prevalence**: pinned source shows that the relevant control or constraint exists.
2. **Controlled problem**: a common-LLM multi-shape counterfactual reproduces the failure mode.
3. **Controlled solution**: changing only the proposed decision produces a correct speedup.
4. **Native reproduction**: the same paired counterfactual runs inside an applicable framework.

The first statement cannot prove performance. A default framework throughput row cannot prove
either the problem or the solution. `RQ_NATIVE_COVERAGE.md` is the fail-closed result of these
rules after every run.

## Exact mapping

| RQ | Important observation | Problem experiment | Solution experiment | Installed native scope |
|---|---|---|---|---|
| RQ1 | Persistent storage choice and live kernel adaptivity are different levels | NHD/HND head-local versus token-local winners; v10 edge inversion | consumer/edge-aware oracle versus one persistent layout | vLLM and SGLang force legal KV layouts over four request regimes |
| RQ2 | Profitable layout scope depends on consumer heterogeneity and split overhead | opposite consumer winners | common NHD/HND versus split storage with conversion | no faithful native adapter yet; SGLang page size is not substituted for split domains |
| RQ3 | Conversion is an investment whose break-even depends on reuse | reuse-dependent winner change | no-convert, convert-once and convert-each-use at reuse 1/4/16/64 | TVM times row/tiled consumers plus conversion and composes measured primitive costs |
| RQ4 | Isolated latency is not the token/max-rank critical path | isolated versus HBM-contention inversion | contention-aware selection | single-card proxy only; TP/EP/All2All remains unverified |
| RQ5 | Independent consumers can have an empty direct layout intersection | head-local/token-local typed conflict | common, repaired split and disable alternatives | controlled proxy only; native speculative target/draft run is missing |
| RQ6 | Layout choice is stateful and migration prevents per-request switching | prefill/decode winner drift | fixed, aggressive and hysteretic epoch policy with measured migration | vLLM/SGLang can prove static regret only; live migration solution is not exposed |
| RQ7 | Shape/stride alone is not a full edge contract | typed empty intersection | legality-first repair search | TVM executes a correctness-checked row-to-tiled edge; this is feasibility, not automatically speedup |
| RQ8 | Fusion, pass order and layout are non-commutative | RoPE/convert order and TVM materialize/convert order | fused native-layout candidate | TVM native primitives and fused TE kernel over four shapes |
| RQ9 | Framework heuristics are local experts; one threshold/layout is not portable | heterogeneous winners across shape/consumer | best static policy versus per-shape oracle | vLLM, SGLang and TVM on A10; held-out GPU portability remains missing |
| RQ10 | Direct mismatch does not imply impossibility; bounded repair can restore legality | empty direct intersection | legal repair, convert-once versus repeated repair | TVM native row-to-tiled repair; full repair-family/resource search stays in controlled planner |

Strict `layout_summary_v10` L-RQ1 is an additional protocol, not a replacement for RQ2--RQ10.
It tests producer-local versus complete-edge selection with independent process repetitions,
bootstrap intervals and a practical-significance threshold.

## Native TVM evidence labels

- `native_runtime_counterfactual`: the TVM CUDA function itself was timed.
- `native_measured_cost_model`: totals add medians from timed TVM primitives. They are not called
  a directly timed end-to-end pipeline.
- `feasibility_supported`: the conversion/repair compiled, ran and passed correctness, but no
  speedup claim follows from legality alone.

## Known hard limits of one GPU

- RQ4 distributed placement/collective overlap cannot be validated on physical GPU 1 alone.
- RQ9 cross-hardware portability requires at least one held-out GPU generation.
- vLLM/SGLang whole-engine measurements cannot decompose producer, conversion edge and consumer
  costs unless those engines expose or are instrumented with a lower-level trace.
