#!/usr/bin/env bash
# Hardware-portable front door for the 256-case v10/v13 shape-diversity suite.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_INDEX=0
MODE=smoke
MODEL_PATH="${SERVING_MODEL_PATH:-}"
CUTLASS_ROOT="${CUTLASS_DIR:-}"
ENV_FILE="${FRAMEWORK_ENV_FILE:-${HERE}/framework_envs.generated.sh}"
OUTPUT=""

usage() {
  cat <<'EOF'
usage: run_llm_shape_diversity_v2_portable.sh --model PATH --cutlass PATH [options]

Options:
  --gpu INDEX       Physical GPU index on the target host (default: 0)
  --smoke           Run 16-case smoke validation (default)
  --full            Run all 256 cases
  --env-file PATH   framework_envs.generated.sh created on the target host
  --output DIR      Fresh result directory; never overwrites an old run
  --model PATH      Local pinned Hugging Face model directory
  --cutlass PATH    CUTLASS source directory containing include/cutlass/cutlass.h

The script detects the CUDA compute capability through the target PyTorch
environment and passes ARCH=sm_XY/TVM_CUDA_ARCH=sm_XY to every backend.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_INDEX="$2"; shift 2 ;;
    --smoke) MODE=smoke; shift ;;
    --full) MODE=full; shift ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --cutlass) CUTLASS_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -f "${ENV_FILE}" ]] || {
  echo "environment file is missing: ${ENV_FILE}" >&2
  echo "run install_native_framework_envs.sh on the target host first" >&2
  exit 2
}
# shellcheck disable=SC1090
source "${ENV_FILE}"

[[ -n "${MODEL_PATH}" && -d "${MODEL_PATH}" ]] || {
  echo "--model must point to a local model directory" >&2
  exit 2
}
[[ -n "${CUTLASS_ROOT}" && -f "${CUTLASS_ROOT}/include/cutlass/cutlass.h" ]] || {
  echo "--cutlass must point to a CUTLASS source checkout" >&2
  exit 2
}

ORCHESTRATOR_PYTHON="${PYTHON_BIN:-${VLLM_PYTHON:-}}"
TORCH_RUNNER="${TORCH_PYTHON:-${VLLM_PYTHON:-}}"
[[ -x "${ORCHESTRATOR_PYTHON}" && -x "${TORCH_RUNNER}" ]] || {
  echo "no executable PyTorch Python found in ${ENV_FILE}" >&2
  exit 2
}

DETECTED_ARCH="$(CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${TORCH_RUNNER}" -c \
  'import torch; assert torch.cuda.is_available(); a,b=torch.cuda.get_device_capability(0); print(f"sm_{a}{b}")')"
ARCH="${DETECTED_ARCH}"
if [[ -z "${OUTPUT}" ]]; then
  OUTPUT="${HERE}/results/shape_diversity_v2_${ARCH}_${MODE}_$(date +%Y%m%d_%H%M%S)"
fi
[[ ! -e "${OUTPUT}" ]] || {
  echo "refusing to overwrite ${OUTPUT}" >&2
  exit 1
}

echo "[portable] gpu=${GPU_INDEX} arch=${ARCH} mode=${MODE} output=${OUTPUT}"

GPU_PHYSICAL_INDEX="${GPU_INDEX}" \
ARCH="${ARCH}" \
TVM_CUDA_ARCH="${ARCH}" \
PYTHON_BIN="${ORCHESTRATOR_PYTHON}" \
TORCH_PYTHON="${TORCH_RUNNER}" \
FRAMEWORK_ENV_FILE="${ENV_FILE}" \
SERVING_MODEL_PATH="${MODEL_PATH}" \
CUTLASS_DIR="${CUTLASS_ROOT}" \
PERSUASIVE_WARMUP="${PERSUASIVE_WARMUP:-3}" \
PERSUASIVE_ITERATIONS="${PERSUASIVE_ITERATIONS:-10}" \
CUDA_WARMUP="${CUDA_WARMUP:-8}" \
CUDA_ITERATIONS="${CUDA_ITERATIONS:-30}" \
NATIVE_SUBGRAPH_TIMEOUT="${NATIVE_SUBGRAPH_TIMEOUT:-14400}" \
KV_BATCH_TIMEOUT="${KV_BATCH_TIMEOUT:-28800}" \
REAL_WORLD_CASE_TIMEOUT="${REAL_WORLD_CASE_TIMEOUT:-7200}" \
SERVING_STARTUP_TIMEOUT="${SERVING_STARTUP_TIMEOUT:-3600}" \
SERVING_REQUEST_TIMEOUT="${SERVING_REQUEST_TIMEOUT:-3600}" \
bash "${HERE}/run_persuasive_real_world_rqs_gpu1.sh" \
  "--${MODE}" --diversity-v2 --output "${OUTPUT}" --model "${MODEL_PATH}"
