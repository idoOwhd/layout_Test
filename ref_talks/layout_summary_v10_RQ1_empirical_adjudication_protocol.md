# layout_summary_v10_RQ1_empirical_adjudication_protocol.md

> **Date:** 2026-09-17  
> **Scope:** empirical adjudication plan for **L-RQ1 only**.  
> **L-RQ1:** *When does the physical layout minimizing producer-local cost differ from the layout minimizing producer→consumer subgraph cost?*  
> **Input:** `layout_summary_v9_serial_problem_trials_RQ1_RQ10.md`, which closed the source/reasoning trial for L-RQ1 before proceeding to later RQs.
>
> This document does **not** claim an empirical result. The current environment has no suitable H100/B200/MI300 execution target, so the output of this step is a preregistered, directly runnable empirical protocol. The scientific verdict remains `OPEN`.

---

# 0. Why L-RQ1 is now experimentally ready

The source trial established a real producer-consumer tension rather than a hypothetical one.

FlashInfer's current KV-layout documentation states:

- **NHD** follows the natural output organization of \(xW_k/xW_v\);
- **HND** is friendlier to GPU implementation for low-precision KV;
- for FP16 KV the library reports little difference between NHD/HND in its documented experience.

Its TRTLLM-gen NVFP4 decode API further states that NHD input triggers an automatic transpose and `.contiguous()` conversion of **both KV data and block-scale tensors** to HND, with explicit allocation/copy overhead.

That gives a clean empirical test:

```text
producer preference: natural NHD
consumer preference: potentially HND for low-precision KV
boundary action: none / zero-copy / materialized conversion / producer-native HND
```

Current vLLM source also treats NHD/HND as aliases for generalized KV layouts (`LBNHC`/`LBHNC`), supports a broader six-layout default preference family, and resolves one whole-model layout once before memory profiling/allocation.

Therefore L-RQ1 is not asking whether multiple layouts exist. It asks whether **their ranking changes when the graph edge is optimized rather than the producer in isolation**.

---

# 1. Scientific claim and null hypothesis

## 1.1 Primary claim

For workload \(w\), let:

\[
L_P^*(w)=\arg\min_L T_P(L,w)
\]

be the producer-local winner.

For an edge strategy \(s=(L_P,a,L_C)\), where \(a\) is the boundary adaptation:

\[
J_e(s,w)=T_P(L_P,w)+T_a(s,w)+T_C(L_C,w).
\]

The edge oracle is:

\[
s_e^*(w)=\arg\min_s J_e(s,w).
\]

A **rank inversion** exists if the producer layout used by the edge oracle differs from \(L_P^*\), or if using the producer-local winner creates statistically significant edge regret:

\[
R_e(w)=
J_e(s(L_P^*),w)-J_e(s_e^*,w)>0.
\]

## 1.2 Null hypothesis

\[
H_0:
L_P^*=L_e^*
\quad\text{or}\quad
R_e \le \epsilon_{\text{noise}}
\]

for all tested workloads.

A useful paper requires rejecting \(H_0\) in a non-trivial, reproducible regime—not merely finding a one-off microbenchmark win.

---

# 2. Primary subgraph

Freeze the first experiment to one edge family:

```text
QKV / KV producer
   ↓
RoPE / cache write
   ↓
paged KV representation
   ↓
decode attention consumer
```

Do **not** begin with full-engine throughput. Scheduler/queueing/prefix-cache effects would make it unclear whether a layout ranking change came from the edge or from the serving system.

The first benchmark must therefore measure:

1. producer only;
2. boundary adaptation only;
3. consumer only;
4. composed edge.

Full vLLM/SGLang integration comes only after the edge-level effect is established.

---

# 3. Candidate strategies

For NHD/HND-capable consumers, enumerate at least:

| Strategy | Producer | Boundary | Consumer | Scientific role |
|---|---|---|---|---|
| **NN** | native NHD | none | NHD | natural-producer baseline |
| **NH-copy** | native NHD | transpose/repack | HND | production-style repair |
| **HH-native** | direct HND emission | none | HND | producer-native consumer layout |
| **HN-copy** | direct HND emission | transpose/repack | NHD | reverse negative/control case |
| **view/stride** | legal source layout | zero-copy stride/index adaptation | compatible consumer | include only where the backend proves it legal |

Important: `HH-native` must not be implemented as `NHD producer + transpose`. Otherwise it is not a producer-native strategy and cannot adjudicate H1.4.

For vLLM generalized layouts, the same schema later extends from `{NHD,HND}` to:

```text
LBNHC
LBHNC
BLNHC
BLHNC
BHLNC
LHBNC
```

but **only** if producer and consumer implementations for those layouts are legal and equivalent.

---

# 4. Stage 0 — Measurement and correctness controls

Before ranking layouts:

1. validate identical logical K/V values across physical layouts;
2. validate attention output against a reference within dtype-appropriate tolerance;
3. verify no hidden allocation occurs inside timed loops unless allocation is intentionally part of the adaptation;
4. preallocate workspace;
5. record exact GPU, driver, CUDA and library commit/version;
6. pin clocks/power policy when the environment allows;
7. isolate the GPU from unrelated jobs.

Negative control:

- BF16/FP16;
- batch 1;
- q_len 1;
- moderate KV length;
- NHD versus HND.

The documented FlashInfer expectation is that FP16 NHD/HND often have little performance difference. If this control shows a large unstable difference, the measurement harness is suspect or the current hardware/library has materially changed the behavior.

---

# 5. Stage 1 — Producer-local ranking

Measure **only the writer**.

For each layout:

\[
T_P(L)
\]

must include the real address mapping/write pattern used to create the cache representation.

Minimum producer implementations:

```text
P-NHD: write K/V directly as NHD
P-HND: write K/V directly as HND
```

For quantized KV:

```text
P-NHD-Q: data + scale production in NHD-compatible format
P-HND-Q: data + scale production directly in HND consumer-native format
```

Do not hide a post-write transpose inside a “producer-native” implementation.

### Primary producer variables

Hold most variables fixed initially:

- head_dim = 128;
- batch = 8;
- q_len = 1;
- KV history = 2048;
- page size = 64;
- Hq/Hkv = {1, 8, 32};
- dtype = {BF16, NVFP4}.

The purpose is simply to establish:

\[
\text{rank}_P(NHD,HND).
\]

---

# 6. Stage 2 — Consumer-local ranking

Prebuild logically identical KV caches in both layouts and measure decode independently:

\[
T_C(NHD),\quad T_C(HND).
\]

Do not include cache construction in this stage.

Minimum axes:

- dtype `{BF16, NVFP4}`;
- Hq/Hkv `{1, 8, 32}`;
- batch `8`;
- q_len `1`;
- KV len `2048`;
- page size `64`.

Where supported, record:

- FlashInfer XQA / TRTLLM-gen path;
- whether tensor cores are used;
- whether the consumer silently performs a layout conversion.

A consumer measurement that silently converts is **not** a native consumer measurement and must be classified under Stage 3 boundary adaptation.

---

# 7. Stage 3 — Primary L-RQ1 edge test

Compose the four principal strategies:

```text
NN
NH-copy
HH-native
HN-copy
```

and measure:

\[
T_{\text{edge}}
=
T_P+T_{\text{boundary}}+T_C.
\]

The critical comparison is not `NHD vs HND` in isolation.

It is:

```text
producer-local winner
       versus
edge-global strategy winner.
```

For every workload:

1. identify producer-local winner;
2. identify edge oracle;
3. compute edge regret of producer-local choice;
4. record whether a rank inversion occurs.

---

# 8. Statistical decision rule

For each configuration:

- at least **5 independent process-level repetitions**;
- each repetition performs warmup followed by enough timed iterations to stabilize the median;
- report process-level medians, p95 and coefficient of variation;
- use a paired/bootstrap 95% confidence interval for edge-regret difference.

Declare a **positive L-RQ1 inversion** only if all hold:

1. producer-local winner is stable across repetitions;
2. edge-global winner is different;
3. 95% CI for edge regret excludes zero;
4. relative regret exceeds:

\[
\epsilon =
\max(1\%, 3\times CV_{\text{baseline}})
\]

5. numerical correctness passes.

This avoids publishing a “rank inversion” that is smaller than measurement noise.

---

# 9. Stage 4 — Crossover surface

Only after Stage 3 finds a positive inversion should the experiment expand.

Sweep one axis at a time around the Stage-3 center:

### Batch

```text
1, 8, 32, 64
```

### KV history

```text
128, 512, 2048, 8192, 32768
```

### GQA ratio

```text
Hq/Hkv = 1, 4, 8, 16, 32
```

Ratio 1 acts as MHA/MQA-like control depending on head topology; larger ratios stress GQA.

### Query/speculative length

```text
1, 4, 8
```

This tests whether speculative/MTP-style query length changes consumer preference.

### Page size

```text
16, 64, 128
```

This is a **sensitivity control**, not an attempt to answer L-RQ10. If page size materially changes the L-RQ1 inversion region, record the interaction and leave global page-size optimization to L-RQ10.

---

# 10. Stage 5 — Modern-attention external validity

Do not expand here until the core NHD/HND edge result is established.

## 10.1 GQA

Primary external-validity target because:

- FlashInfer explicitly documents GQA-group-size sensitivity for tensor-core decode;
- head-ratio changes the amount of Q work relative to shared K/V consumption.

Test whether inversion magnitude increases with Hq/Hkv.

## 10.2 MLA

Test a latent-cache producer and MLA consumer.

Question:

> Does the producer-natural latent representation differ from the consumer/backend-preferred representation strongly enough to create the same rank-inversion phenomenon?

This is external validity, not a separate RQ.

## 10.3 Sparse attention

Use only after the dense/GQA result.

TensorRT-LLM's current sparse contracts couple:
- MHA/MQA/GQA topology;
- paged HND;
- fixed page sizes/profiles;
- routes/indices/page tables.

This makes sparse attention a useful stress case but also introduces L-RQ8 data–metadata coupling, so it is deliberately not the first L-RQ1 benchmark.

## 10.4 MoE

Use:

```text
dispatch/receive representation
→ optional scale interleave
→ grouped GEMM
```

Compare:

```text
dispatcher-local format
dispatcher + materialized scale interleave
producer-native expert-kernel scale layout
```

If rank inversion appears here too, L-RQ1 generalizes beyond Attention/KV.

---

# 11. Hypothesis-to-experiment map

## H1.1 — Rank inversion exists

Test: Stage 3.

Success:

```text
argmin producer != argmin edge
AND statistically/practically significant edge regret.
```

Falsifier:

```text
producer-local winner is also edge winner for every legal configuration tested.
```

## H1.2 — Reuse/fanout increases inversion

KV history alone is not reuse count.

Test explicit repeated consumer reads or downstream consumers while reusing the same produced representation.

Expected:

\[
\frac{\partial R_e}{\partial R}>0
\]

when consumer-native advantage is positive.

Falsifier: regret does not grow with reuse.

## H1.3 — Stride-polymorphism suppresses inversion

Compare:

```text
materialized adaptation
vs
legal zero-copy stride/view consumer.
```

Falsifier: stride-polymorphic path still pays equivalent cost or cannot match native consumer performance.

## H1.4 — Producer-native emission wins

Compare:

```text
natural producer layout + conversion
vs
direct consumer-native producer layout.
```

Falsifier: producer-native write penalty always exceeds saved conversion/consumer gain.

## H1.NEG — Layout-insensitive regime

BF16/FP16 + small/simple workload is the principal negative control.

A good optimizer should **not** invent a conversion or exotic layout when edge regret is statistically negligible.

---

# 12. Metrics

Primary:

```text
producer_us
boundary_us
consumer_us
edge_us
edge_regret_pct
```

Mandatory resource metrics:

```text
bytes_copied
temporary_bytes
```

Profiler metrics when available:

```text
HBM read/write bytes
L2 hit rate
SM occupancy
memory throughput
```

Do not use profiler metrics to define the winner; use them to explain the mechanism after timing establishes the effect.

---

# 13. Baselines

Minimum baselines:

1. framework/library native default;
2. producer-local winner;
3. consumer-native layout;
4. explicit materialized conversion;
5. direct producer-native consumer layout;
6. zero-copy/view adaptation where legal;
7. offline exhaustive edge oracle over the legal candidate set.

The oracle is essential.

Without it, a learned or heuristic policy can appear good merely because all compared baselines are weak.

---

# 14. Implementation strategy

## 14.1 First implementation

Use a standalone CUDA/PyTorch benchmark process:

```text
producer writer
→ optional boundary adaptation
→ FlashInfer decode
```

Why:

- minimal scheduler noise;
- FlashInfer directly exposes NHD/HND;
- low-precision NHD→HND conversion is explicitly documented;
- GQA ratio is directly controllable.

The producer writer can initially be a small Triton/CUDA kernel so both NHD and HND are genuinely direct-write alternatives.

## 14.2 Second implementation

Integrate with vLLM.

Current vLLM source:

- maps legacy NHD/HND to generalized `LBNHC/LBHNC`;
- has a six-layout default preference family;
- intersects backend support sets;
- resolves one layout before memory profiling;
- makes the resolved value final.

This makes vLLM a strong whole-engine external-validity target **after** edge-level rank inversion is measured.

## 14.3 Do not start with a learned policy

L-RQ1 first needs an oracle effect size.

Only if:

\[
R_e > \text{measurement noise}
\]

over a meaningful region should a later policy be trained to predict the winner.

---

# 15. Stop/go criteria before L-RQ2

L-RQ2 empirical work is blocked until one of the following L-RQ1 verdicts is recorded.

## GO: positive phenomenon

Proceed if:

- at least one common workload family has reproducible rank inversion;
- practical regret is non-trivial;
- effect survives an independent hardware or backend check.

Then L-RQ2 asks whether multiple consumers justify multiple layout domains.

## STOP/NARROW: weak phenomenon

Narrow the research program if:

- inversion appears only in a pathological configuration;
- edge regret is below noise/practical threshold;
- low-level stride-polymorphism removes almost all mismatch cost.

In that case a full cross-subgraph layout planner is weakly motivated.

## REFRAME

If inversion exists only because the framework's layout space is too narrow, move the main causal explanation toward L-RQ9 rather than forcing L-RQ1.

---

# 16. Current status

| Gate | Status |
|---|---|
| Source-rule trial | CLOSED_WITHIN_DECLARED_LEDGER |
| Experimental variable definition | CLOSED |
| Candidate/action-space definition | CLOSED |
| Baseline/oracle definition | CLOSED |
| Statistical decision rule | CLOSED |
| Negative controls | CLOSED |
| Modern GQA/MLA/MoE expansion plan | CLOSED |
| Runnable GPU measurements | **OPEN — no target GPU in current environment** |
| L-RQ1 empirical verdict | **OPEN** |
| Permission to begin L-RQ2 empirical trial | **BLOCKED until L-RQ1 verdict** |

The next executable action on a GPU machine is **Stage 0→Stage 3 only**. Do not begin L-RQ2 experiments until L-RQ1 has a positive, negative, or narrowed empirical verdict.
