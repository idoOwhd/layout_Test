# Layout research harness

This directory turns layout claims into paired, falsifiable experiments.  Read
[`SCIENTIFIC_QUESTIONS.md`](SCIENTIFIC_QUESTIONS.md) for the hypotheses,
framework taxonomy, source links, interpretation rules and output schema.

Run every enabled LLM and 2025–2026 image/video-generation experiment on physical GPU 1:

```bash
cd /home/liangyilei/ladder_home
bash staged/baseline_framework/layout_research/run_all_gpu1.sh --full
```

The controller never silently substitutes a framework.  Optional dependencies
are marked `unavailable`; SGLang/vLLM kernel-internal decisions are marked
`delegated`.  Re-run a smoke suite with `--quick` before occupying the GPU for
the full 640-case run.

The vision catalog is independent of CUDA and contains 12 modern architecture
families, 13 motifs meeting a 50% all/image/video incidence threshold, and 83
multi-shape cases. Regenerate it with:

```bash
python3 staged/baseline_framework/vision_shapes/build_vision_manifest.py
```

To add exact shapes from already generated ONNX models to a timestamped run:

```bash
VISION_ONNX_ROOT=/absolute/path/to/onnx \
  bash staged/baseline_framework/layout_research/run_all_gpu1.sh --full
```

To include native SGLang/vLLM end-to-end policy checks, point to one identical
local or already-cached model (the script does not silently download a model):

```bash
SERVING_MODEL_PATH=/absolute/path/to/model \
  bash staged/baseline_framework/layout_research/run_all_gpu1.sh --full
```

vLLM rows force its native `VLLM_KV_CACHE_LAYOUT` candidates. SGLang compares
its native NHD/HND KV-cache control with backend/page/model/requests fixed;
the additional attention-backend/page-policy rows remain labelled confounded.

The supplied ChatGPT shares are stored as ten separate HTTPS URLs in
`shared_sources.json`; the Codex thread is a separate `codex://` local-client
reference. Each full run content-validates the HTTPS pages and stores
`shared_link_validation.json`. Set `CHECK_SHARED_LINKS=0` only for an offline
run. The same network phase resolves each mutable GitHub/Hugging Face visual
model source to a commit and stores the immutable links in
`vision_source_revisions.json`.

Framework source evidence follows the same rule: local `*_DIR` checkouts take
precedence; otherwise the network-enabled run resolves the configured official
GitHub repository to a commit and stores exact matched file/line snippets in
`source_evidence.json`.

Every run also produces `REQUIREMENTS_AUDIT.md` and
`requirements_audit.json`.  They distinguish code that has been implemented
from claims that have actually been validated, and list every partial, missing,
blocked or source-dependent requirement.

The optimality analyzer keeps the full requested Cartesian product visible:
640 LLM cases and 83 visual cases across all six frameworks, for 4,338 cells.
Unsupported or delegated framework/case pairs remain missing with a reason;
they are never removed from the denominator.

The machine-readable source of truth is `requirements_status.json`.  To audit
the implementation without occupying a GPU, run:

```bash
python3 staged/baseline_framework/layout_research/audit_requirements.py \
  --run-dir /tmp/layout_requirements_audit
```

`ncu` is invoked only when the NVIDIA driver responds and is bounded by
`NCU_TIMEOUT_SECONDS` (300 seconds by default), so a broken profiler cannot
stall the complete batch indefinitely.

## `ref_talks` RQ1--RQ10 validation

The focused runner below audits the pinned `ref_talks` evidence, selects
multiple LLM shapes, runs causal CUDA layout mechanisms, runs every installed
native/compiler adapter, and produces one RQ report:

```bash
cd /home/liangyilei/ladder_home
CUDA_VISIBLE_DEVICES=1 \
  bash staged/baseline_framework/layout_research/run_ref_talks_rq_gpu1.sh --full
```

The script itself exports `CUDA_VISIBLE_DEVICES=1`; the explicit prefix above
only makes the physical-card choice visible at launch.  `--full` selects up to
three distinct min/median/max contracts for every `(subgraph, phase)` from the
pinned 640-case catalog.  This is 46 representative PyTorch/Triton cases over
GQA, sliding/sparse attention, MLA, SwiGLU, MoE, Mamba2 and linear attention,
plus 26 attention-layout cases stratified into MHA, MQA and GQA.  `--quick`
still retains one case per stratum instead of truncating the suite to two
shapes.

Useful optional settings are:

```bash
PYTHON_BIN=/path/to/gpu/env/bin/python \
SGLANG_PYTHON=/path/to/sglang/env/bin/python \
VLLM_PYTHON=/path/to/vllm/env/bin/python \
SERVING_MODEL_PATH=/absolute/path/to/local/model \
CHECK_SOURCE_URLS=1 \
RUN_ALL_640=1 \
  bash staged/baseline_framework/layout_research/run_ref_talks_rq_gpu1.sh --full \
  --output /absolute/path/to/results
```

`RUN_ALL_640=1` is optional because the representative set already performs
the requested multi-shape validation. `SERVING_MODEL_PATH` enables real vLLM
and SGLang server runs if those packages are installed. Framework-specific
environments can also be supplied with `TORCH_PYTHON`, `TILELANG_PYTHON`,
`TVM_PYTHON`, `SGLANG_PYTHON`, and `VLLM_PYTHON`, while keeping one bash entry
point. Hexcute's upstream
artifact is opt-in with `RUN_HEXCUTE_ARTIFACT=1` and `HEXCUTE_BENCH_DIR` because
it can take hours and owns its environment.

The main outputs are `RQ_VALIDATION_REPORT.md`, `rq_validation.json`,
`rq_observations.jsonl`, `rq_cuda_reference.csv`,
`rq_llm_representative_cases.json`, and `status.jsonl`. The report never treats
repository evidence as runtime evidence, never calls the controlled CUDA
kernels a native framework result, and explicitly leaves multi-GPU RQ4 and
cross-GPU RQ9 portability unverified on a single A10.

## `layout_summary_v10` L-RQ1 protocol

v10 is a stricter, preregistered protocol for L-RQ1 only. Run it separately:

```bash
cd /home/liangyilei/ladder_home
bash staged/baseline_framework/layout_research/run_v10_rq1_gpu1.sh --full
```

It launches at least five independent processes, computes paired bootstrap
confidence intervals, enforces `max(1%, 3×CV)` practical significance, and
executes Stage 4/5 only if Stage 3 passes. See
[`V10_RQ1_EXPERIMENT_MAP.md`](V10_RQ1_EXPERIMENT_MAP.md) for the exact mapping
from every experiment to H1.1/H1.2/H1.3/H1.4/H1.NEG. Native NVFP4 and native
framework edge decompositions remain explicitly unsupported/missing when the
hardware or packages do not provide them; they are never replaced by a CUDA
reference result.

## Cross-RQ observation validation

The observation runner combines the canonical RQ1--RQ10 protocol and the
strict v10 L-RQ1 protocol, then keeps three claims separate: official-source
prevalence, controlled causal speedup, and native-framework causal speedup.
This prevents a common inference error: a policy found in three repositories
does not by itself prove that changing the policy improves performance.
The observation-to-RQ rationale and exact proof boundary are documented in
[`CRITICAL_OBSERVATIONS_AND_VALIDATION.md`](CRITICAL_OBSERVATIONS_AND_VALIDATION.md).

Run the complete protocol on physical GPU 1 with one command:

```bash
cd /home/liangyilei/ladder_home
CHECK_SOURCE_URLS=1 \
  bash staged/baseline_framework/layout_research/run_observation_validation_gpu1.sh --full
```

The final report is
`results/observations_<timestamp>/OBSERVATION_VALIDATION_REPORT.md`. The same
directory contains `observation_validation.{json,csv}`, the automatically
extracted `native_observation_results.jsonl`, and raw canonical/v10 artifacts.
An observation receives grade A only when a majority of the relevant native
framework denominator has a correct paired counterfactual with greater than
1% speedup. Grade B means source-majority plus controlled causal evidence;
grade C means source-majority only; grade D means controlled evidence only.

Additional exact native-framework results can be merged without modifying the
runner. Each JSONL row must contain `observation_id`, `framework`, `status`,
`speedup`, `artifact`, and `evidence_level`:

```bash
NATIVE_OBSERVATION_RESULTS=/absolute/path/to/native_results.jsonl \
  bash staged/baseline_framework/layout_research/run_observation_validation_gpu1.sh --full
```

## Install and validate vLLM, SGLang and CUDA TVM

These frameworks use isolated environments because their PyTorch/CUDA package
constraints are not interchangeable. The recommended setup uses vLLM/SGLang
wheels and compiles TVM with CUDA:

```bash
cd /home/liangyilei/ladder_home
bash staged/baseline_framework/layout_research/setup_and_run_native_frameworks_gpu1.sh \
  --install-only
```

To compile the audited vLLM/SGLang revisions as well as TVM from source:

```bash
bash staged/baseline_framework/layout_research/setup_and_run_native_frameworks_gpu1.sh \
  --install-only --source-builds
```

Installation records package freezes and source commits under
`framework_install_logs/`, and writes `framework_envs.generated.sh`. Once a
small model that fits the A10 is present locally, perform preflight and run the
native adapters on physical GPU 1:

```bash
bash staged/baseline_framework/layout_research/run_native_framework_validation_gpu1.sh \
  --model /absolute/path/to/local/model --preflight-only

bash staged/baseline_framework/layout_research/setup_and_run_native_frameworks_gpu1.sh \
  --run-only --model /absolute/path/to/local/model --full
```

The validation wrapper treats `unavailable`, `skipped`, and `failed` as a
failure for vLLM, SGLang, TVM, CUTLASS and Triton. It therefore cannot report a
successful all-framework run when a dependency or model is missing.

### Run all installed-framework RQ/observation experiments on card 1

After installation, the following is the preferred single entry point. It
verifies/downloads the pinned small Qwen checkpoint, runs four exact-token
serving regimes in vLLM and SGLang, runs the multi-shape TVM native RQ suite,
the controlled RQ1--RQ10 suite, strict v10 L-RQ1, and (by default) all 640
external-validity cases:

```bash
cd /home/liangyilei/ladder_home
bash staged/baseline_framework/layout_research/run_installed_rq_observations_gpu1.sh --full
```

Use `--model /absolute/local/model --no-download` to force an existing pinned
checkpoint, or `--skip-640` when the prior 640 replay is being reused. The two
final decision reports are `OBSERVATION_VALIDATION_REPORT.md` (broad framework
denominators) and `RQ_NATIVE_COVERAGE.md` (vLLM/SGLang/TVM per-RQ fail-closed
audit). The static experiment map is
[`RQ_OBSERVATION_EXPERIMENT_MAP.md`](RQ_OBSERVATION_EXPERIMENT_MAP.md).

### Complete only the previously skipped native serving rows

For the 640-case catalog, the recommended small common checkpoint is the
pinned `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf...`. It is itself a catalog
source and maps to six GQA/SwiGLU/sliding-attention prefill/decode rows. The
following command downloads that exact revision when absent, runs only SGLang
and vLLM on physical card 1, appends their status to an existing run, and
refreshes the aggregate observation report:

```bash
cd /home/liangyilei/ladder_home
bash staged/baseline_framework/layout_research/complete_missing_native_serving_gpu1.sh \
  --result /home/liangyilei/ladder_home/staged/baseline_framework/layout_research/results/card1_all_frameworks_20260917_153155 \
  --full
```

The two full request shapes use exact token-ID prompts of 2,048 and 8,192
tokens. The generated `native_model_case_mapping.json` records which catalog
cases they cover. These are end-to-end framework-native measurements: they do
not falsely claim that every one of the 640 synthetic subgraphs was isolated
inside a serving engine, and the long-context decode row necessarily includes
prefill before its decode tokens.

## v13 RQ validation on physical GPU 1

The v13 suite is separate from the older RQ numbering. It reads the canonical
49 hypotheses from `ref_talks/layout_summary_v13_*`, checks exact registry
coverage, audits source-observation prevalence, runs common LLM subgraphs, and
emits a fail-closed RQ × framework matrix.

Subgraph-only smoke (no model download or whole-engine serving):

```bash
cd /home/liangyilei/ladder_home
GPU_PHYSICAL_INDEX=1 \
bash staged/baseline_framework/layout_research/run_v13_validation_gpu1.sh \
  --smoke --skip-serving
```

Full min/median/max multi-shape suite. PyTorch/Triton real-world cases run in
separate processes with a per-case timeout, so one compiler outlier cannot
block the remaining subgraphs:

```bash
cd /home/liangyilei/ladder_home
GPU_PHYSICAL_INDEX=1 \
REAL_WORLD_CASE_TIMEOUT=180 \
RUN_ALL_640=0 \
CUDA_WARMUP=8 CUDA_ITERATIONS=30 \
TRITON_WARMUP=8 TRITON_ITERATIONS=30 \
bash staged/baseline_framework/layout_research/run_v13_validation_gpu1.sh \
  --full --skip-serving
```

Add vLLM/SGLang native offline-engine evidence to an existing v13 result. This
uses exact-token prompts and avoids the HTTP control plane; it creates a new
combined directory and never changes the base result:

```bash
cd /home/liangyilei/ladder_home
GPU_PHYSICAL_INDEX=1 \
bash staged/baseline_framework/layout_research/run_v13_native_serving_gpu1.sh \
  --full \
  --base-result /absolute/path/to/v13_full_TIMESTAMP \
  --model /home/liangyilei/ladder_home/staged/baseline_framework/layout_research/models/Qwen2.5-0.5B-Instruct-7ae5576
```

To run smoke and then the complete multi-shape/native campaign with one command:

```bash
cd /home/liangyilei/ladder_home
GPU_PHYSICAL_INDEX=1 \
RUN_ALL_640=0 \
CUDA_WARMUP=8 CUDA_ITERATIONS=30 \
TRITON_WARMUP=8 TRITON_ITERATIONS=30 \
bash staged/baseline_framework/layout_research/run_v13_all_gpu1.sh --all
```

Set `RUN_ALL_640=1` only for the separate exhaustive legacy replay. The full
v13 default already selects min/median/max tensor contracts for every common
subgraph and phase.

Each invocation creates a new `results/v13_smoke_TIMESTAMP` or
`results/v13_full_TIMESTAMP` directory. The primary outputs are:

- `V13_RQ_VALIDATION_REPORT.md`: hypothesis-by-hypothesis problem/solution verdicts;
- `V13_SOURCE_OBSERVATIONS.md`: direct-stack source prevalence only;
- `V13_FRAMEWORK_MATRIX.md`: native runtime, source-only, missing-adapter,
  not-applicable and single-GPU-blocked cells;
- `v13_hypothesis_results.json`: machine-readable metrics and evidence levels.

RQ5 and H4.4 are intentionally not validated on card1. CUDA HBM contention is
never relabeled as a distributed collective or cross-hardware experiment.
