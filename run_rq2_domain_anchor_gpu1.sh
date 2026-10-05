#!/usr/bin/env bash
# Distinct experiment output. No baseline mutation, no model downloads.
set -euo pipefail
HERE=${RQ2_RUNNER_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
if [[ ${RQ2_FROZEN_RUNNER:-0} != 1 ]]; then
  FROZEN=$(mktemp /tmp/layout_rq2_domain_anchor_XXXXXXXX.sh)
  cp -- "${BASH_SOURCE[0]}" "$FROZEN"
  exec env RQ2_FROZEN_RUNNER=1 RQ2_RUNNER_ROOT="$HERE" RQ2_FROZEN_SCRIPT="$FROZEN" bash "$FROZEN" "$@"
fi
BASELINE="";OUT="";MODE=smoke;WAIT_LOCK=0;APPROVED_SMOKE=""
SOURCE_ROOT=${RQ2_DOMAIN_SOURCE_ROOT:-$HERE}
FRAMEWORKS=${RQ2_FRAMEWORKS:-pytorch,triton,cutlass,tvm,vllm,sglang}
while (($#)); do
  case "$1" in
    --baseline-run) BASELINE=$2;shift 2;;
    --output) OUT=$2;shift 2;;
    --smoke) MODE=smoke;shift;;
    --full) MODE=full;shift;;
    --frameworks) FRAMEWORKS=$2;shift 2;;
    --wait-for-gpu-lock) WAIT_LOCK=1;shift;;
    --approved-smoke) APPROVED_SMOKE=$2;shift 2;;
    *) printf 'Unknown argument: %s\n' "$1" >&2;exit 2;;
  esac
done
[[ -n $BASELINE && -f $BASELINE/design_manifest.json ]] || { printf 'Valid --baseline-run required.\n' >&2;exit 2; }
BASE_PYTHON=${PYTHON_BIN:-/home/liangyilei/conda/envs/cuda-opt/bin/python}
source "${FRAMEWORK_ENV_FILE:-$HERE/framework_envs.generated.sh}"
[[ ${GPU_PHYSICAL_INDEX:-1} == 1 ]] || { printf 'Restricted to physical card1.\n' >&2;exit 2; }
exec 9>"/tmp/layout_rq2_gpu1_${UID}.lock"
if ((WAIT_LOCK)); then
  printf '[rq2-domain-anchor] waiting for existing RQ2 card1 run to release its lock\n'
  flock 9
else
  flock -n 9 || { printf 'Card1 RQ2 runner busy; no overlapping GPU work started.\n' >&2;exit 75; }
fi
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RQ2_PROJECT_ROOT=$HERE
if [[ -z $OUT ]]; then OUT=$HERE/results/rq2_domain_anchor_${MODE}_$(date +%Y%m%d_%H%M%S)_$$;fi
[[ ! -e $OUT ]] || { printf 'Refusing to overwrite: %s\n' "$OUT" >&2;exit 2; }
mkdir -p "$OUT/source_snapshot" "$OUT/logs" "$OUT/artifacts"
# Use the EXACT base GPU implementations measured by the baseline run.
for FILE in rq2_growth_bench.py rq2_growth_search.py rq2_growth_kernels.py rq2_growth_cutlass.cu rq1_diverse_kernels.py rq1_flashinfer_workspace.py; do
  cp -- "$BASELINE/source_snapshot/$FILE" "$OUT/source_snapshot/$FILE"
done
for FILE in rq2_domain_anchor_bench.py analyze_rq2_domain_anchor.py audit_rq2_holdout_statistics.py check_rq2_graph_growth_progress.py; do
  cp -- "$SOURCE_ROOT/$FILE" "$OUT/source_snapshot/$FILE"
done
cp -- "$RQ2_FROZEN_SCRIPT" "$OUT/source_snapshot/run_rq2_domain_anchor_gpu1.sh"
cp -- "$BASELINE/design_manifest.json" "$OUT/design_manifest.json"
if [[ -n $APPROVED_SMOKE ]];then
  "$BASE_PYTHON" "$SOURCE_ROOT/check_rq2_conditional_smoke.py" --smoke "$APPROVED_SMOKE" \
    --run "$OUT" > "$OUT/logs/smoke_approval.log" 2>&1
fi
sha256sum "$OUT"/source_snapshot/* > "$OUT/source_sha256.txt"
nvidia-smi -i 1 --query-gpu=uuid,name,memory.total,memory.used --format=csv > "$OUT/gpu_preflight.csv"
"$BASE_PYTHON" -m unittest discover -s "$HERE" -p 'test_rq2*.py' > "$OUT/logs/cpu_tests.log" 2>&1
IFS=, read -ra TARGETS <<< "$FRAMEWORKS"
WARMUP=${RQ2_WARMUP:-3};ITERATIONS=${RQ2_ITERATIONS:-10};HOLDOUT=${RQ2_HOLDOUT_REPETITIONS:-5}
if [[ $MODE == smoke ]]; then WARMUP=${RQ2_WARMUP:-2};ITERATIONS=${RQ2_ITERATIONS:-3};HOLDOUT=${RQ2_HOLDOUT_REPETITIONS:-3};fi
printf '[rq2-domain-anchor] output=%s mode=%s baseline=%s\n' "$OUT" "$MODE" "$BASELINE"
for FRAMEWORK in "${TARGETS[@]}"; do
  case "$FRAMEWORK" in
    vllm) EXECUTOR=$VLLM_PYTHON;;
    sglang) EXECUTOR=$SGLANG_PYTHON;;
    pytorch|triton|cutlass|tvm) EXECUTOR=$BASE_PYTHON;;
    *) printf 'Unknown framework: %s\n' "$FRAMEWORK" >&2;exit 2;;
  esac
  if [[ $FRAMEWORK == cutlass ]]; then cp -- "$BASELINE/artifacts/rq2_cutlass.so" "$OUT/artifacts/rq2_cutlass.so";fi
  set +e
  "$EXECUTOR" "$OUT/source_snapshot/rq2_domain_anchor_bench.py" --framework "$FRAMEWORK" \
    --manifest "$OUT/design_manifest.json" --baseline-results "$BASELINE/$FRAMEWORK.jsonl" \
    --mode "$MODE" --output "$OUT/$FRAMEWORK.jsonl" --artifacts "$OUT/artifacts" \
    --warmup "$WARMUP" --iterations "$ITERATIONS" --holdout-repetitions "$HOLDOUT" > "$OUT/logs/$FRAMEWORK.log" 2>&1
  STATUS=$?
  set -e
  printf '%s\t%s\n' "$FRAMEWORK" "$STATUS" >> "$OUT/process_status.tsv"
  printf '[rq2-domain-anchor] %s exit=%s\n' "$FRAMEWORK" "$STATUS"
done
"$BASE_PYTHON" "$OUT/source_snapshot/analyze_rq2_domain_anchor.py" --output "$OUT"
if awk '$2 != 0 {found=1} END {exit !found}' "$OUT/process_status.tsv";then exit 1;fi
printf '[rq2-domain-anchor] complete=%s\n' "$OUT"
