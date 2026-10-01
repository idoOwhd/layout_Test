#!/usr/bin/env bash
# One command for the installed vLLM, SGLang and CUDA-TVM RQ/observation suite.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${FRAMEWORK_ENV_FILE:-${HERE}/framework_envs.generated.sh}"
GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX:-1}"
MODEL_PATH="${SERVING_MODEL_PATH:-${HERE}/models/Qwen2.5-0.5B-Instruct-7ae5576}"
OUTPUT=""
MODE=full
RUN_640="${RUN_ALL_640:-1}"
DOWNLOAD=1

usage() {
  cat <<'EOF'
usage: run_installed_rq_observations_gpu1.sh [options]
  --output DIR       Result root (default timestamped)
  --model DIR        Existing/download target checkpoint directory
  --no-download      Require the checkpoint to exist locally
  --quick | --full   Default full
  --skip-640         Skip the exhaustive 640-case external-validity replay
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --no-download) DOWNLOAD=0; shift ;;
    --quick) MODE=quick; shift ;;
    --full) MODE=full; shift ;;
    --skip-640) RUN_640=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -f "${ENV_FILE}" ]] || { echo "missing ${ENV_FILE}; install environments first" >&2; exit 1; }
# shellcheck disable=SC1090
source "${ENV_FILE}"
BASE_PYTHON="${PYTHON_BIN:-/home/liangyilei/conda/envs/cuda-opt/bin/python}"
if [[ -z "${OUTPUT}" ]]; then
  OUTPUT="${HERE}/results/installed_rq_observations_$(date +%Y%m%d_%H%M%S)"
fi

MODEL_ARGS=(--output "${MODEL_PATH}")
[[ "${DOWNLOAD}" == 0 ]] && MODEL_ARGS+=(--no-download)
env -u PYTHONPATH -u PYTHONHOME "${VLLM_PYTHON}" "${HERE}/prepare_pinned_serving_model.py" "${MODEL_ARGS[@]}"

export GPU_PHYSICAL_INDEX PYTHON_BIN="${BASE_PYTHON}" TORCH_PYTHON="${BASE_PYTHON}"
export VLLM_PYTHON SGLANG_PYTHON TVM_PYTHON TVM_LIBRARY_PATH CUDA_HOME
export SERVING_MODEL_PATH="${MODEL_PATH}"
export SERVING_CASES="${SERVING_CASES:-128:1:8:1,2048:1:5:1,8192:16:5:1,2048:32:16:4}"
export RUN_ALL_640="${RUN_640}"
export CHECK_SOURCE_URLS="${CHECK_SOURCE_URLS:-1}"
export CUDA_WARMUP="${CUDA_WARMUP:-8}" CUDA_ITERATIONS="${CUDA_ITERATIONS:-30}"
export TRITON_WARMUP="${TRITON_WARMUP:-8}" TRITON_ITERATIONS="${TRITON_ITERATIONS:-30}"
export SERVING_STARTUP_TIMEOUT="${SERVING_STARTUP_TIMEOUT:-900}"
export SERVING_REQUEST_TIMEOUT="${SERVING_REQUEST_TIMEOUT:-600}"

RUN_ARGS=(--model "${MODEL_PATH}" "--${MODE}" --output "${OUTPUT}")
[[ "${RUN_640}" == 1 ]] && RUN_ARGS+=(--run-640)
bash "${HERE}/run_native_framework_validation_gpu1.sh" "${RUN_ARGS[@]}"

echo "[installed-rq] observation report=${OUTPUT}/OBSERVATION_VALIDATION_REPORT.md"
echo "[installed-rq] per-RQ native coverage=${OUTPUT}/RQ_NATIVE_COVERAGE.md"
echo "[installed-rq] raw native evidence=${OUTPUT}/native_observation_results.jsonl"
