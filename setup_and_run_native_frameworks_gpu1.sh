#!/usr/bin/env bash
# One front door for installation followed by native-framework validation.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE=all
MODEL_PATH="${SERVING_MODEL_PATH:-}"
OUTPUT=""
MODE=full
RUN_640=0
VLLM_METHOD="${VLLM_INSTALL_METHOD:-wheel}"
SGLANG_METHOD="${SGLANG_INSTALL_METHOD:-wheel}"
TVM_METHOD="${TVM_INSTALL_METHOD:-source}"

usage() {
  cat <<'EOF'
usage: setup_and_run_native_frameworks_gpu1.sh --model PATH [options]

Options:
  --install-only       Install/build environments but do not occupy GPU
  --run-only           Reuse existing environments and run validation
  --source-builds      Build vLLM, SGLang and TVM from audited source revisions
  --quick | --full     Validation mode; default full
  --run-640            Also repeat the complete 640-case suite
  --output DIR         Explicit validation output directory
  --model PATH         Local Hugging Face model; required unless --install-only
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-only) PHASE=install; shift ;;
    --run-only) PHASE=run; shift ;;
    --source-builds) VLLM_METHOD=source; SGLANG_METHOD=source; TVM_METHOD=source; shift ;;
    --quick) MODE=quick; shift ;;
    --full) MODE=full; shift ;;
    --run-640) RUN_640=1; shift ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "${PHASE}" != run ]]; then
  bash "${HERE}/install_native_framework_envs.sh" \
    --vllm "${VLLM_METHOD}" --sglang "${SGLANG_METHOD}" --tvm "${TVM_METHOD}"
fi

if [[ "${PHASE}" == install ]]; then
  echo "[setup] installation complete; run later with:"
  echo "bash ${HERE}/setup_and_run_native_frameworks_gpu1.sh --run-only --model /absolute/model/path"
  exit 0
fi

[[ -n "${MODEL_PATH}" ]] || { echo "--model is required for validation" >&2; exit 2; }
RUN_ARGS=(--model "${MODEL_PATH}" "--${MODE}")
if [[ "${RUN_640}" == 1 ]]; then RUN_ARGS+=(--run-640); fi
if [[ -n "${OUTPUT}" ]]; then RUN_ARGS+=(--output "${OUTPUT}"); fi

GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX:-1}" \
  bash "${HERE}/run_native_framework_validation_gpu1.sh" "${RUN_ARGS[@]}"
