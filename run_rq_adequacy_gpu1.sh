#!/usr/bin/env bash
# Single-card front door for the v10/v13 RQ adequacy supplement.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
MODE=smoke
OUTPUT=""
GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX:-1}"
ENV_FILE="${FRAMEWORK_ENV_FILE:-${HERE}/framework_envs.generated.sh}"
MODEL_PATH="${SERVING_MODEL_PATH:-${HERE}/models/Qwen2.5-0.5B-Instruct-7ae5576}"
CUTLASS_ROOT="${CUTLASS_DIR:-/home/liangyilei/cursor_home/Ladder/3rdparty/cutlass}"
RQ_ONLY=""

usage() {
  cat <<'EOF'
usage: run_rq_adequacy_gpu1.sh [--smoke|--full] [--rq L-RQ1..L-RQ10] [--output NEW_DIR] [--model DIR]

Runs the redesigned single-GPU adequacy suite on physical GPU 1:
1024 real-model-derived contracts in full mode, dense reuse/fanout/state/page
counterfactuals, Triton/TVM/CUTLASS and native vLLM/SGLang slices.  Hexcute,
cross-architecture claims and true multi-rank L-RQ5 are deliberately excluded.
Existing output directories are rejected.

With --rq, the runner selects every manifest case whose target_rqs contains
that RQ and launches only the applicable framework adapters. L-RQ5 is recorded
as blocked because a physical single-GPU run cannot test a distributed RQ.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) MODE=smoke; shift ;;
    --full) MODE=full; shift ;;
    --rq) RQ_ONLY="${2^^}"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ "${RQ_ONLY}" =~ ^RQ([1-9]|10)$ ]] && RQ_ONLY="L-${RQ_ONLY}"
if [[ -n "${RQ_ONLY}" && ! "${RQ_ONLY}" =~ ^L-RQ([1-9]|10)$ ]]; then
  echo "invalid --rq ${RQ_ONLY}; expected L-RQ1..L-RQ10" >&2
  exit 2
fi

[[ -f "${ENV_FILE}" ]] || { echo "missing framework env: ${ENV_FILE}" >&2; exit 2; }
# shellcheck disable=SC1090
source "${ENV_FILE}"
PYTHON_BIN="${PYTHON_BIN:-${VLLM_PYTHON:-}}"
TORCH_PYTHON="${TORCH_PYTHON:-${VLLM_PYTHON:-}}"
[[ -x "${PYTHON_BIN}" && -x "${TORCH_PYTHON}" ]] || {
  echo "PYTHON_BIN/TORCH_PYTHON is missing from ${ENV_FILE}" >&2; exit 2; }
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "missing model: ${MODEL_PATH}" >&2; exit 2; }
[[ -f "${CUTLASS_ROOT}/include/cutlass/cutlass.h" ]] || {
  echo "missing CUTLASS headers: ${CUTLASS_ROOT}" >&2; exit 2; }
if [[ -z "${OUTPUT}" ]]; then
  OUTPUT="${HERE}/results/rq_adequacy_gpu1_${MODE}_$(date +%Y%m%d_%H%M%S)"
fi
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 1; }
RQ_ARGS=()
[[ -n "${RQ_ONLY}" ]] && RQ_ARGS=(--rq "${RQ_ONLY}")

cd "${ROOT}"
GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX}" \
PYTHON_BIN="${PYTHON_BIN}" TORCH_PYTHON="${TORCH_PYTHON}" \
FRAMEWORK_ENV_FILE="${ENV_FILE}" SERVING_MODEL_PATH="${MODEL_PATH}" \
CUTLASS_DIR="${CUTLASS_ROOT}" \
PERSUASIVE_WARMUP="${PERSUASIVE_WARMUP:-3}" \
PERSUASIVE_ITERATIONS="${PERSUASIVE_ITERATIONS:-10}" \
CUDA_WARMUP="${CUDA_WARMUP:-8}" CUDA_ITERATIONS="${CUDA_ITERATIONS:-30}" \
NATIVE_SUBGRAPH_TIMEOUT="${NATIVE_SUBGRAPH_TIMEOUT:-28800}" \
KV_BATCH_TIMEOUT="${KV_BATCH_TIMEOUT:-28800}" \
REAL_WORLD_CASE_TIMEOUT="${REAL_WORLD_CASE_TIMEOUT:-14400}" \
SERVING_STARTUP_TIMEOUT="${SERVING_STARTUP_TIMEOUT:-3600}" \
SERVING_REQUEST_TIMEOUT="${SERVING_REQUEST_TIMEOUT:-3600}" \
bash "${HERE}/run_persuasive_real_world_rqs_gpu1.sh" \
  "--${MODE}" --adequacy-v3 "${RQ_ARGS[@]}" \
  --output "${OUTPUT}" --model "${MODEL_PATH}"

echo "[rq-adequacy] complete=${OUTPUT}"
if [[ -n "${RQ_ONLY}" ]]; then
  echo "[rq-adequacy] selected-rq=${RQ_ONLY}"
  echo "[rq-adequacy] selection=${OUTPUT}/cases/SINGLE_RQ_SELECTION.json"
  echo "[rq-adequacy] report=${OUTPUT}/SINGLE_RQ_REPRODUCTION_CN.md"
else
  echo "[rq-adequacy] report=${OUTPUT}/RQ_ADEQUACY_ANALYSIS_CN.md"
  echo "[rq-adequacy] strict-per-rq=${OUTPUT}/RQ_FULL_PER_RQ_STRICT_ANALYSIS_CN.md"
fi
