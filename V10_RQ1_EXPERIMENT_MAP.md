# v10 L-RQ1 experiment map

`layout_summary_v10_*` defines one research question:

> **L-RQ1:** When does the physical layout minimizing producer-local cost
> differ from the layout minimizing producer→consumer subgraph cost?

H1.1, H1.2, H1.3, H1.4 and H1.NEG are hypotheses under L-RQ1, not additional
RQ numbers.

| Experiment | What is held fixed / varied | L-RQ1 claim tested | Required success condition |
|---|---|---|---|
| Stage 0 BF16 negative control | B=1, Q=1, KV=512, Hq/Hkv=1 | H1.NEG: layout-insensitive regime | no statistically/practically significant inversion |
| Stage 1 producer isolation | identical logical KV and direct NHD/HND emission | producer-local rank | stable producer winner across ≥5 processes |
| Stage 2 consumer isolation | prebuilt equal NHD/HND paged caches | consumer-local rank | records native consumer preference without construction cost |
| Stage 3 NN/NH-copy/HH-native/HN-copy | complete producer→adaptation→consumer edge | H1.1 rank inversion | stable different edge winner, bootstrap CI > 0, regret > max(1%,3CV), correctness |
| reuse=1/4/16/64 | same B/Q/KV/Hq/Hkv, repeated reads only | H1.2 reuse/fanout | edge regret grows with reuse |
| NH-view vs NH-copy | zero-copy stride path vs materialization | H1.3 stride polymorphism | view avoids conversion and reduces edge cost |
| HH-native vs NH-copy | direct HND producer vs NHD plus transpose | H1.4 native emission | direct consumer-native emission wins in a reproducible regime |
| Stage 4 B/KV/GQA/Q/PAGE sweeps | one axis at a time | crossover surface for L-RQ1 | executed only after Stage-3 GO |
| Stage 5 real-model shapes | GQA/sliding attention, then MLA/sparse/MoE boundaries | external validity of L-RQ1 | conditional; never substitutes for Stage 3 |

The CUDA reference emits all required v10 measurement fields and is labeled
`runtime_empirical_controlled`. It is not relabeled as FlashInfer, vLLM or
SGLang. Source auditing establishes that layout decisions/tensions exist in
multiple frameworks; only an exact native producer/boundary/consumer run may
claim native runtime validation.

## Shape coverage

The core full run contains:

- the v10 small BF16 negative control;
- Stage-3 centers at Hq/Hkv 1, 8 and 32;
- explicit reuse 1, 4, 16 and 64;
- preregistered NVFP4 rows, reported unsupported when no native path exists;
- min/median/max decode contracts drawn from the pinned 640-case LLM catalog,
  stratified across MHA/MQA/GQA and GQA/sliding-attention families.

After a positive Stage-3 gate, Stage 5 selects 46 contracts across GQA,
sliding/sparse attention, MLA, SwiGLU, MoE, Mamba2 and linear attention, with
prefill and decode represented separately.

## Run

```bash
cd /home/liangyilei/ladder_home
bash staged/baseline_framework/layout_research/run_v10_rq1_gpu1.sh --full
```

The primary outputs are `V10_RQ1_VALIDATION_REPORT.md`,
`v10_rq1_validation.json`, `v10_rq1_adjudicated.csv`,
`v10_rq1_experiment_map.json`, per-process raw CSVs and `status.jsonl`.
