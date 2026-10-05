#!/usr/bin/env bash
# Exact failed-cell retry; originals are read-only, all output is new.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=full
PREVIOUS=""
OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --previous) PREVIOUS="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --smoke) MODE=smoke; shift ;;
    --full) MODE=full; shift ;;
    --help|-h) echo 'usage: run_rq1_failure_retry_gpu1.sh --previous OLD_RESULTS [--smoke|--full] [--output NEW_DIR]'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$PREVIOUS" && -f "$PREVIOUS/manifest.json" ]] || { echo '--previous must identify a measured run' >&2; exit 2; }
[[ "${GPU_PHYSICAL_INDEX:-1}" == 1 ]] || { echo 'only physical GPU1 is allowed' >&2; exit 2; }
ENV_FILE="${FRAMEWORK_ENV_FILE:-$HERE/framework_envs.generated.sh}"
[[ -f "$ENV_FILE" ]] || { echo "missing env file $ENV_FILE" >&2; exit 2; }
# shellcheck disable=SC1090
source "$ENV_FILE"
BASE_PYTHON="${TORCH_PYTHON:-${PYTHON_BIN:-/home/liangyilei/conda/envs/cuda-opt/bin/python}}"
for task_python in "$BASE_PYTHON" "${VLLM_PYTHON:-}" "${SGLANG_PYTHON:-}"; do
  [[ -x "$task_python" ]] || { echo "missing interpreter $task_python" >&2; exit 2; }
done
if [[ -z "$OUTPUT" ]]; then
  OUTPUT="$HERE/results/rq1_failed_retry_${MODE}_$(date +%Y%m%d_%H%M%S)_${BASHPID}"
fi
PREPARE_ARGS=()
[[ "$MODE" == smoke ]] && PREPARE_ARGS=(--smoke)
"$BASE_PYTHON" "$HERE/rq1_failure_retry.py" prepare --previous "$PREVIOUS" --output "$OUTPUT" "${PREPARE_ARGS[@]}"
OUTPUT="$(cd "$OUTPUT" && pwd)"
IFS=$'\t' read -r WARMUP ITERATIONS REPS < <("$BASE_PYTHON" -c \
  'import json,sys;p=json.load(open(sys.argv[1]));print(p["warmup"],p["iterations"],p["source_repetitions"],sep="\t")' \
  "$OUTPUT/retry_plan.json")
# Even retry smoke keeps the previous warmup/iteration protocol. Smoke differs
# only by the number of selected independent processes, not input shapes.
[[ "$MODE" == smoke ]] && REPS=1
export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
"$BASE_PYTHON" "$HERE/rq1_diversity_preflight.py" --output "$OUTPUT/preflight.json" \
  --manifest "$OUTPUT/manifest.json" --warmup "$WARMUP" --iterations "$ITERATIONS" --repetitions "$REPS"
failures=0
while IFS=$'\t' read -r mode rep selection; do
  case "$mode" in
    torch) task_python="$BASE_PYTHON" ;;
    vllm) task_python="$VLLM_PYTHON" ;;
    sglang) task_python="$SGLANG_PYTHON" ;;
    *) echo "invalid mode $mode" >&2; exit 2 ;;
  esac
  stem="${mode}_rep${rep}"
  echo "[rq1-retry] $stem; exact selection=$OUTPUT/selections/$selection"
  if timeout --signal=TERM --kill-after=120 "${RQ1_PROCESS_TIMEOUT:-28800}" \
    "$task_python" "$HERE/rq1_diverse_edge_bench.py" --manifest "$OUTPUT/manifest.json" \
    --output "$OUTPUT/raw/$stem.jsonl" --consumer-mode "$mode" \
    --case-ids-file "$OUTPUT/selections/$selection" --repetition "$rep" \
    --warmup "$WARMUP" --iterations "$ITERATIONS" --max-bytes "${RQ1_MAX_CASE_BYTES:-8589934592}" \
    > "$OUTPUT/logs/$stem.log" 2>&1; then
    echo "[rq1-retry] $stem finished"
  else
    status=$?;failures=$((failures+1))
    echo "$stem exit=$status" >> "$OUTPUT/process_failures.txt"
    echo "[rq1-retry] $stem failed; retaining partial records" >&2
  fi
done < "$OUTPUT/jobs.tsv"
"$BASE_PYTHON" "$HERE/analyze_rq1_diversity.py" --output-dir "$OUTPUT"
"$BASE_PYTHON" "$HERE/check_rq1_diversity_run.py" --output-dir "$OUTPUT" || failures=$((failures+1))
# Successful corrected cells supersede whole old failed cells, never individual
# faster rows. Unrepaired cells stay visibly failed in the new merged report.
"$BASE_PYTHON" "$HERE/rq1_failure_retry.py" merge --retry-dir "$OUTPUT"
"$BASE_PYTHON" "$HERE/analyze_rq1_diversity.py" --output-dir "$OUTPUT/merged"
if [[ "$MODE" == smoke ]]; then
  "$BASE_PYTHON" "$HERE/check_rq1_diversity_run.py" --output-dir "$OUTPUT/merged" || true
  echo '[rq1-retry] smoke does not replace all five process repetitions; merged preview may retain failures.'
else
  "$BASE_PYTHON" "$HERE/check_rq1_diversity_run.py" --output-dir "$OUTPUT/merged" || failures=$((failures+1))
fi
echo "[rq1-retry] output=$OUTPUT"
echo "[rq1-retry] correctness=$OUTPUT/CORRECTNESS_AND_COVERAGE_CN.md"
echo "[rq1-retry] merged-report=$OUTPUT/merged/RQ1_DIVERSITY_REPORT_CN.md"
echo '[rq1-retry] previous files unchanged; PR-native replay completeness is a separate requirement.'
[[ "$failures" == 0 ]]
