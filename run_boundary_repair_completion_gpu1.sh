#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/liangyilei/ladder_home
LR="$ROOT/staged/baseline_framework/layout_research"
PYTHON_BIN=${PYTHON_BIN:-/home/liangyilei/conda/envs/cuda-opt/bin/python}
GPU_PHYSICAL_INDEX=${GPU_PHYSICAL_INDEX:-1}
BASE_RESULT=${BASE_RESULT:-$LR/results/v10_v13_640_all_frameworks_20260920_143843}
PRIMARY_RUN=${PRIMARY_RUN:-$LR/results/repaired_non_oom_20260920_213642}
WARMUP=${REPAIR_WARMUP:-3}
ITERATIONS=${REPAIR_ITERATIONS:-10}
TIMEOUT=${REPAIR_CASE_TIMEOUT:-3600}

if [[ ${1:-} == "--output" ]]; then
  OUT=$2
  shift 2
else
  OUT=${OUT:-$LR/results/repaired_boundary_completion_$(date +%Y%m%d_%H%M%S)}
fi
if [[ $# -ne 0 ]]; then
  echo "unexpected arguments: $*" >&2
  exit 2
fi
if [[ -e "$OUT" ]]; then
  echo "refusing to overwrite existing output: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT/logs" "$OUT/raw" "$OUT/manifests"

"$PYTHON_BIN" -m unittest "$LR/test_repaired_workloads.py" -v \
  > "$OUT/logs/unit_tests.log" 2>&1
"$PYTHON_BIN" "$LR/build_boundary_supplement_manifest.py" \
  --primary-run "$PRIMARY_RUN" --output "$OUT/manifests/mamba_retry.json" \
  | tee "$OUT/logs/build_manifest.log"

CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL_INDEX" "$PYTHON_BIN" "$LR/run_boundary_isolated.py" \
  --runner "$LR/llm_boundary_layout_sweep.py" \
  --python "$PYTHON_BIN" \
  --manifest "$OUT/manifests/mamba_retry.json" \
  --output "$OUT/raw/mamba_boundary_isolated.jsonl" \
  --log-dir "$OUT/logs/mamba_isolated" \
  --physical-device-index "$GPU_PHYSICAL_INDEX" \
  --warmup "$WARMUP" --iterations "$ITERATIONS" --timeout "$TIMEOUT" \
  | tee "$OUT/logs/driver.log"

"$PYTHON_BIN" "$LR/analyze_repair_completion.py" \
  --base-result "$BASE_RESULT" --primary-run "$PRIMARY_RUN" --supplement-run "$OUT" \
  | tee "$OUT/logs/analysis.log"
echo "[repair-completion] output=$OUT"
echo "[repair-completion] report=$OUT/REPAIR_COMPLETION_CN.md"
