#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/liangyilei/ladder_home
LR="$ROOT/staged/baseline_framework/layout_research"
BASELINE="$ROOT/staged/baseline_framework"
PYTHON_BIN=${PYTHON_BIN:-/home/liangyilei/conda/envs/cuda-opt/bin/python}
GPU_PHYSICAL_INDEX=${GPU_PHYSICAL_INDEX:-1}
BASE_RESULT=${BASE_RESULT:-$LR/results/v10_v13_640_all_frameworks_20260920_143843}
WARMUP=${REPAIR_WARMUP:-3}
ITERATIONS=${REPAIR_ITERATIONS:-10}
TIMEOUT=${REPAIR_CASE_TIMEOUT:-3600}

if [[ ${1:-} == "--output" ]]; then
  OUT=$2
  shift 2
else
  OUT=${OUT:-$LR/results/repaired_non_oom_$(date +%Y%m%d_%H%M%S)}
fi
if [[ $# -ne 0 ]]; then
  echo "unexpected arguments: $*" >&2
  exit 2
fi
if [[ -e "$OUT" ]]; then
  echo "refusing to overwrite existing output: $OUT" >&2
  exit 2
fi

mkdir -p "$OUT/logs" "$OUT/raw"
{
  echo "BASE_RESULT=$BASE_RESULT"
  echo "OUT=$OUT"
  echo "GPU_PHYSICAL_INDEX=$GPU_PHYSICAL_INDEX"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "WARMUP=$WARMUP"
  echo "ITERATIONS=$ITERATIONS"
  echo "TIMEOUT=$TIMEOUT"
  date --iso-8601=seconds
} > "$OUT/RUN_METADATA.txt"

"$PYTHON_BIN" -m unittest "$LR/test_repaired_workloads.py" -v \
  > "$OUT/logs/unit_tests.log" 2>&1
"$PYTHON_BIN" "$LR/build_repair_rerun_manifests.py" \
  --base-result "$BASE_RESULT" --output-dir "$OUT/manifests" \
  > "$OUT/logs/build_manifests.log" 2>&1

CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL_INDEX" "$PYTHON_BIN" "$LR/run_real_world_isolated.py" \
  --runner "$BASELINE/benchmark_real_world.py" \
  --python "$PYTHON_BIN" \
  --manifest "$OUT/manifests/whole_moe_non_oom.json" \
  --output "$OUT/raw/whole_moe_non_oom.jsonl" \
  --log-dir "$OUT/logs/whole_isolated" \
  --physical-device-index "$GPU_PHYSICAL_INDEX" \
  --backend pytorch --backend triton \
  --warmup "$WARMUP" --iterations "$ITERATIONS" --timeout "$TIMEOUT" \
  | tee "$OUT/logs/whole_driver.log"

CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL_INDEX" "$PYTHON_BIN" "$LR/llm_boundary_layout_sweep.py" \
  --manifest "$OUT/manifests/boundary_moe_mamba_non_oom.json" \
  --output "$OUT/raw/boundary_moe_mamba_non_oom.jsonl" \
  --physical-device-index "$GPU_PHYSICAL_INDEX" \
  --warmup "$WARMUP" --iterations "$ITERATIONS" \
  | tee "$OUT/logs/boundary_driver.log"

set +e
"$PYTHON_BIN" "$LR/analyze_repair_rerun.py" \
  --base-result "$BASE_RESULT" --rerun-dir "$OUT" \
  | tee "$OUT/logs/analysis.log"
ANALYSIS_EXIT=${PIPESTATUS[0]}
set -e
echo "$ANALYSIS_EXIT" > "$OUT/ANALYSIS_EXIT_CODE"
echo "[repair-rerun] output=$OUT"
echo "[repair-rerun] report=$OUT/REPAIR_RERUN_ANALYSIS_CN.md"
exit "$ANALYSIS_EXIT"
