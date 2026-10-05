#!/usr/bin/env bash
# New supplementary suite; never changes old runners or old result directories.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=smoke
OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) MODE=smoke; shift ;;
    --extended-smoke) MODE=extended-smoke; shift ;;
    --architecture-smoke) MODE=architecture-smoke; shift ;;
    --validation-smoke) MODE=validation-smoke; shift ;;
    --full) MODE=full; shift ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --help|-h) echo 'usage: run_rq1_diversity_gpu1.sh [--smoke|--extended-smoke|--architecture-smoke|--validation-smoke|--full] [--output NEW_DIR]'; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX:-1}"
[[ "$GPU_PHYSICAL_INDEX" == 1 ]] || { echo 'This entry point is restricted to physical GPU 1.' >&2; exit 2; }
ENV_FILE="${FRAMEWORK_ENV_FILE:-${HERE}/framework_envs.generated.sh}"
[[ -f "$ENV_FILE" ]] || { echo "missing env file: $ENV_FILE" >&2; exit 2; }
# shellcheck disable=SC1090
source "$ENV_FILE"
BASE_PYTHON="${TORCH_PYTHON:-${PYTHON_BIN:-/home/liangyilei/conda/envs/cuda-opt/bin/python}}"
for task_python in "$BASE_PYTHON" "${VLLM_PYTHON:-}" "${SGLANG_PYTHON:-}"; do
  [[ -x "$task_python" ]] || { echo "missing interpreter: $task_python" >&2; exit 2; }
done
if [[ -z "$OUTPUT" ]]; then
  OUTPUT="${HERE}/results/rq1_diversity_${MODE}_$(date +%Y%m%d_%H%M%S)_${BASHPID}"
fi
[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite $OUTPUT" >&2; exit 2; }
mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd)"
mkdir "$OUTPUT/raw" "$OUTPUT/logs"
export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
REPS="${RQ1_PROCESS_REPETITIONS:-5}"
WARMUP="${RQ1_WARMUP:-8}"
ITERATIONS="${RQ1_ITERATIONS:-30}"
if [[ "$MODE" != full ]]; then
  REPS="${RQ1_PROCESS_REPETITIONS:-1}"
  WARMUP="${RQ1_WARMUP:-2}"
  ITERATIONS="${RQ1_ITERATIONS:-5}"
fi
[[ "$REPS" =~ ^[1-9][0-9]*$ ]] || { echo 'invalid repetitions' >&2; exit 2; }
GEN_ARGS=()
[[ "$MODE" == smoke ]] && GEN_ARGS=(--smoke)
[[ "$MODE" == extended-smoke ]] && GEN_ARGS=(--extended-smoke)
[[ "$MODE" == architecture-smoke ]] && GEN_ARGS=(--architecture-smoke)
[[ "$MODE" == validation-smoke ]] && GEN_ARGS=(--validation-smoke)
"$BASE_PYTHON" "$HERE/build_rq1_diversity_cases.py" "${GEN_ARGS[@]}" --output "$OUTPUT/manifest.json"
SOURCE_ARGS=()
if [[ -n "${RQ1_SOURCE_AUDIT_JSON:-}" ]]; then
  [[ -f "$RQ1_SOURCE_AUDIT_JSON" ]] || { echo 'source audit does not exist' >&2; exit 2; }
  SOURCE_ARGS=(--source-audit "$RQ1_SOURCE_AUDIT_JSON")
elif [[ "${CHECK_SOURCE_URLS:-1}" == 1 ]]; then
  "$BASE_PYTHON" "$HERE/audit_rq1_github_catalog.py" --output-dir "$OUTPUT/source_audit" \
    --workers "${RQ1_SOURCE_WORKERS:-3}" > "$OUTPUT/logs/source_audit.log" 2>&1
  SOURCE_ARGS=(--source-audit "$OUTPUT/source_audit/source_audit.json")
fi
"$BASE_PYTHON" "$HERE/build_rq1_source_registry.py" "${SOURCE_ARGS[@]}" \
  --manifest "$OUTPUT/manifest.json" --output-dir "$OUTPUT/source_coverage"
# Save implementation fingerprints and environment provenance, not credentials.
"$BASE_PYTHON" "$HERE/rq1_diversity_preflight.py" --output "$OUTPUT/preflight.json" \
  --manifest "$OUTPUT/manifest.json" --warmup "$WARMUP" --iterations "$ITERATIONS" --repetitions "$REPS"
failures=0
for ((rep=0;rep<REPS;rep++)); do
  for consumer in torch vllm sglang; do
    task_python="$BASE_PYTHON"
    FAMILY_ARGS=()
    if [[ "$consumer" == torch ]]; then
      # Native-only families are tested below, not silently replaced by torch.
      for family in attention_bmm_oproj projection_sdpa sparse_gather_sdpa projection_swiglu_down \
        moe_dispatch_expert moe_gemm_combine state_prefill_decode gemm_bias_gemm reduce_norm_gemm host_weight_gemm host_kv_sdpa; do
        FAMILY_ARGS+=(--family "$family")
      done
    else
      [[ "$consumer" == vllm ]] && task_python="$VLLM_PYTHON"
      [[ "$consumer" == sglang ]] && task_python="$SGLANG_PYTHON"
      FAMILY_ARGS=(--family projection_swiglu_down --family projection_gdn \
                   --family kv_writer_flashinfer --family kv_writer_swa_flashinfer)
    fi
    stem="${consumer}_rep${rep}"
    echo "[rq1-diversity] $stem; log=$OUTPUT/logs/$stem.log"
    # Frameworks are deliberately sequential; never time two tenants together.
    if timeout --signal=TERM --kill-after=120 "${RQ1_PROCESS_TIMEOUT:-28800}" \
      "$task_python" "$HERE/rq1_diverse_edge_bench.py" --manifest "$OUTPUT/manifest.json" \
      --output "$OUTPUT/raw/$stem.jsonl" --consumer-mode "$consumer" "${FAMILY_ARGS[@]}" \
      --repetition "$rep" --warmup "$WARMUP" --iterations "$ITERATIONS" \
      --max-bytes "${RQ1_MAX_CASE_BYTES:-8589934592}" \
      > "$OUTPUT/logs/$stem.log" 2>&1; then
      echo "[rq1-diversity] $stem finished"
    else
      status=$?; failures=$((failures+1))
      echo "[rq1-diversity] $stem FAILED exit=$status (retaining partial records)" >&2
      echo "$stem exit=$status" >> "$OUTPUT/process_failures.txt"
    fi
  done
done
"$BASE_PYTHON" "$HERE/analyze_rq1_diversity.py" --output-dir "$OUTPUT"
"$BASE_PYTHON" "$HERE/check_rq1_diversity_run.py" --output-dir "$OUTPUT" || failures=$((failures+1))
echo "[rq1-diversity] report=$OUTPUT/RQ1_DIVERSITY_REPORT_CN.md"
echo "[rq1-diversity] correctness=$OUTPUT/CORRECTNESS_AND_COVERAGE_CN.md"
echo "[rq1-diversity] all-links=$OUTPUT/source_coverage/ALL_PR_ISSUE_COVERAGE_CN.md"
echo '[rq1-diversity] controlled suite != all PR/issue native replays; consult coverage ledger.'
[[ "$failures" == 0 ]]
