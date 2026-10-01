#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=full
OUTPUT=""
MANIFEST=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) MODE=smoke; shift ;;
    --full) MODE=full; shift ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VLLM_PYTHON="${VLLM_PYTHON:-${HERE}/framework_envs/layout-vllm/bin/python}"
SGLANG_PYTHON="${SGLANG_PYTHON:-${HERE}/framework_envs/layout-sglang/bin/python}"
OUTPUT="${OUTPUT:-${HERE}/results/kv_request_batch_${MODE}_$(date +%Y%m%d_%H%M%S)}"
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 1; }
mkdir -p "${OUTPUT}/raw" "${OUTPUT}/logs" "${OUTPUT}/cache"

if [[ -z "${MANIFEST}" ]]; then
  MANIFEST="${HERE}/results/persuasive_real_world_full_20260921_005344/cases/persuasive_real_world_128.json"
fi
[[ -f "${MANIFEST}" ]] || { echo "manifest missing: ${MANIFEST}" >&2; exit 2; }
cp "${MANIFEST}" "${OUTPUT}/executed_manifest.json"

WARMUP="${KV_BATCH_WARMUP:-3}"
ITERATIONS="${KV_BATCH_ITERATIONS:-10}"
LIMIT_ARGS=()
if [[ "${MODE}" == smoke ]]; then
  WARMUP=1; ITERATIONS=3; LIMIT_ARGS=(--balanced-limit 8)
fi

: >"${OUTPUT}/step_status.tsv"
run_one() {
  local framework="$1" python_bin="$2"
  local output="${OUTPUT}/raw/${framework}_kv_request_batch.jsonl"
  set +e
  timeout --signal=TERM --kill-after=60s "${KV_BATCH_TIMEOUT:-14400}" \
    env -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES="${GPU_PHYSICAL_INDEX}" \
      XDG_CACHE_HOME="${OUTPUT}/cache/${framework}" \
      "${python_bin}" "${HERE}/native_kv_request_batch_probe.py" \
      --framework "${framework}" --manifest "${OUTPUT}/executed_manifest.json" \
      --output "${output}" --request-batches "${KV_REQUEST_BATCHES:-1,2,4,8}" \
      --max-bytes "${KV_BATCH_MAX_BYTES:-2147483648}" \
      --warmup "${WARMUP}" --iterations "${ITERATIONS}" "${LIMIT_ARGS[@]}" \
      >"${OUTPUT}/logs/${framework}.log" 2>&1
  local code=$?
  set -e
  printf '%s\t%s\t%s\n' "${framework}" "${code}" "${OUTPUT}/logs/${framework}.log" \
    >>"${OUTPUT}/step_status.tsv"
}

run_one vllm "${VLLM_PYTHON}"
run_one sglang "${SGLANG_PYTHON}"
"${PYTHON_BIN}" "${HERE}/analyze_kv_request_batch.py" \
  --result-dir "${OUTPUT}" --manifest "${OUTPUT}/executed_manifest.json" \
  >"${OUTPUT}/logs/analyze.log" 2>&1
echo "[kv-request-batch] complete=${OUTPUT}"
echo "[kv-request-batch] report=${OUTPUT}/KV_REQUEST_BATCH_REPORT_CN.md"
