#!/usr/bin/env bash
# Non-overwriting GPU1 runner. Native failures are not converted to successes.
set -euo pipefail
HERE=${RQ2_RUNNER_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
# Bash reads scripts incrementally. Freeze the launcher itself as well as the
# worker modules so later development cannot corrupt a long-running script.
if [[ ${RQ2_FROZEN_RUNNER:-0} != 1 ]]; then
  FROZEN_RUNNER=$(mktemp /tmp/layout_rq2_gpu1_runner_XXXXXXXX.sh)
  cp -- "${BASH_SOURCE[0]}" "$FROZEN_RUNNER"
  exec env RQ2_FROZEN_RUNNER=1 RQ2_RUNNER_ROOT="$HERE" \
    RQ2_FROZEN_RUNNER_PATH="$FROZEN_RUNNER" bash "$FROZEN_RUNNER" "$@"
fi
MODE=smoke
OUT=""
FRAMEWORKS=${RQ2_FRAMEWORKS:-pytorch,triton,cutlass,tvm,vllm,sglang}
GROUP_INDEX=""
SMOKE_EXTENDED=0
APPROVED_SMOKE=""
while (($#)); do
  case "$1" in
    --full) MODE=full; shift;;
    --smoke) MODE=smoke; shift;;
    --output) OUT=$2; shift 2;;
    --frameworks) FRAMEWORKS=$2; shift 2;;
    --group-index) GROUP_INDEX=$2; shift 2;;
    --smoke-extended) SMOKE_EXTENDED=1; shift;;
    --approved-smoke) APPROVED_SMOKE=$2; shift 2;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2;;
  esac
done
BASE_PYTHON=${PYTHON_BIN:-/home/liangyilei/conda/envs/cuda-opt/bin/python}
if [[ -f ${FRAMEWORK_ENV_FILE:-$HERE/framework_envs.generated.sh} ]]; then
  source "${FRAMEWORK_ENV_FILE:-$HERE/framework_envs.generated.sh}"
fi
GPU_PHYSICAL_INDEX=${GPU_PHYSICAL_INDEX:-1}
if [[ $GPU_PHYSICAL_INDEX != 1 ]]; then printf 'This runner is restricted to physical card1.\n' >&2; exit 2; fi
GPU_UUID=$(nvidia-smi -i "$GPU_PHYSICAL_INDEX" --query-gpu=uuid --format=csv,noheader)
export CUDA_VISIBLE_DEVICES=$GPU_UUID
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
exec 9>"/tmp/layout_rq2_gpu1_${UID}.lock"
if ! flock -n 9; then printf 'Another RQ2 card1 runner holds the lock.\n' >&2; exit 75; fi
if [[ -z $OUT ]]; then OUT=$HERE/results/rq2_graph_growth_${MODE}_$(date +%Y%m%d_%H%M%S)_$$; fi
if [[ -e $OUT ]]; then printf 'Refusing to overwrite: %s\n' "$OUT" >&2; exit 2; fi
mkdir -p "$OUT/artifacts" "$OUT/logs" "$OUT/source_snapshot"
export RQ2_PROJECT_ROOT=$HERE
for SOURCE in rq2_growth_bench.py rq2_growth_search.py rq2_growth_kernels.py rq2_growth_cutlass.cu \
  rq1_diverse_kernels.py rq1_flashinfer_workspace.py analyze_rq2_graph_growth.py rq2_source_execution_coverage.py \
  audit_rq2_holdout_statistics.py check_rq2_graph_growth_progress.py; do
  cp "$HERE/$SOURCE" "$OUT/source_snapshot/$SOURCE"
done
cp "${RQ2_FROZEN_RUNNER_PATH:-${BASH_SOURCE[0]}}" "$OUT/source_snapshot/run_rq2_graph_growth_gpu1.sh"
sha256sum "$OUT"/source_snapshot/* > "$OUT/source_sha256.txt"
printf '[rq2-growth] output=%s gpu=%s mode=%s\n' "$OUT" "$GPU_UUID" "$MODE"
printf '%s\n' "$$" > "$OUT/runner.pid"
nvidia-smi -i 1 --query-gpu=uuid,name,memory.total,memory.used,utilization.gpu --format=csv > "$OUT/gpu_preflight.csv"
cp "$HERE/ref_talks/rq2_qwen3_real_forward_growth_mixed_design_20261002_v3.json" "$OUT/design_manifest.json"
if [[ -n $APPROVED_SMOKE ]]; then
  "$BASE_PYTHON" "$HERE/check_rq2_smoke_approval.py" --smoke "$APPROVED_SMOKE" \
    --run "$OUT" --frameworks "$FRAMEWORKS" > "$OUT/logs/smoke_approval.log" 2>&1
fi
"$BASE_PYTHON" "$OUT/source_snapshot/rq2_source_execution_coverage.py" \
  --audit "$HERE/ref_talks/rq2_fp16_test_audit_20261001_v3/RQ2_ALL_SOURCE_TEST_AUDIT.json" \
  --output "$OUT/SOURCE_REPLAY_COVERAGE.json"
"$BASE_PYTHON" -m unittest discover -s "$HERE" -p 'test_rq2*.py' > "$OUT/logs/cpu_tests.log" 2>&1
if [[ $MODE == smoke ]]; then WARMUP=${RQ2_WARMUP:-2}; ITERATIONS=${RQ2_ITERATIONS:-3}; HOLDOUT=${RQ2_HOLDOUT_REPETITIONS:-3}; BUDGET=${RQ2_CANDIDATE_BUDGET:-32};
else WARMUP=${RQ2_WARMUP:-3}; ITERATIONS=${RQ2_ITERATIONS:-10}; HOLDOUT=${RQ2_HOLDOUT_REPETITIONS:-5}; BUDGET=${RQ2_CANDIDATE_BUDGET:-48}; fi
IFS=, read -ra TARGETS <<< "$FRAMEWORKS"
for FRAMEWORK in "${TARGETS[@]}"; do
  if [[ $FRAMEWORK == cutlass ]]; then
    CUTLASS_DIR=${CUTLASS_DIR:-/home/liangyilei/cursor_home/Ladder/3rdparty/cutlass}
    "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" -std=c++17 -O3 -shared -Xcompiler=-fPIC \
      -gencode arch=compute_80,code=sm_86 -I "$CUTLASS_DIR/include" \
      "$OUT/source_snapshot/rq2_growth_cutlass.cu" -o "$OUT/artifacts/rq2_cutlass.so" > "$OUT/logs/cutlass_build.log" 2>&1
    if [[ -x ${CUDA_HOME:-/usr/local/cuda}/bin/cuobjdump ]]; then
      "${CUDA_HOME:-/usr/local/cuda}/bin/cuobjdump" --dump-sass "$OUT/artifacts/rq2_cutlass.so" > "$OUT/artifacts/rq2_cutlass.sass"
    fi
  fi
  case "$FRAMEWORK" in
    vllm) EXECUTOR=${VLLM_PYTHON:-$HERE/framework_envs/layout-vllm/bin/python};;
    sglang) EXECUTOR=${SGLANG_PYTHON:-$HERE/framework_envs/layout-sglang/bin/python};;
    pytorch|triton|cutlass|tvm) EXECUTOR=$BASE_PYTHON;;
    *) printf 'Unknown framework: %s\n' "$FRAMEWORK" >&2; exit 2;;
  esac
  EXTRA=()
  if [[ -n $GROUP_INDEX ]]; then EXTRA+=(--group-index "$GROUP_INDEX"); fi
  if ((SMOKE_EXTENDED)); then EXTRA+=(--smoke-extended); fi
  printf '[rq2-growth] starting %s\n' "$FRAMEWORK"
  set +e
  "$EXECUTOR" "$OUT/source_snapshot/rq2_growth_bench.py" --framework "$FRAMEWORK" --mode "$MODE" \
    --manifest "$OUT/design_manifest.json" \
    --output "$OUT/$FRAMEWORK.jsonl" --artifacts "$OUT/artifacts" \
    --warmup "$WARMUP" --iterations "$ITERATIONS" --holdout-repetitions "$HOLDOUT" \
    --candidate-budget "$BUDGET" "${EXTRA[@]}" > "$OUT/logs/$FRAMEWORK.log" 2>&1
  STATUS=$?
  set -e
  printf '%s\t%s\n' "$FRAMEWORK" "$STATUS" >> "$OUT/process_status.tsv"
  printf '[rq2-growth] %s exit=%s log=%s\n' "$FRAMEWORK" "$STATUS" "$OUT/logs/$FRAMEWORK.log"
done
"$BASE_PYTHON" "$OUT/source_snapshot/analyze_rq2_graph_growth.py" --output "$OUT"
"$BASE_PYTHON" "$OUT/source_snapshot/audit_rq2_holdout_statistics.py" --run "$OUT" \
  --output "$OUT/holdout_statistical_audit"
if awk '$2 != 0 {found=1} END {exit !found}' "$OUT/process_status.tsv"; then
  printf '[rq2-growth] FAILED: inspect saved failures; full run is not smoke-approved.\n' >&2
  exit 1
fi
printf '[rq2-growth] complete=%s\n' "$OUT"
