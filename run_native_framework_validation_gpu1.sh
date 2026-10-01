#!/usr/bin/env bash
# Validate installed vLLM, SGLang and CUDA-enabled TVM, then run the layout RQ
# suite on physical GPU 1.  The script fails if a requested native adapter is
# skipped/unavailable/failed, even though the underlying research runner keeps
# going to preserve all artifacts.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX:-1}"
ENV_FILE="${FRAMEWORK_ENV_FILE:-${HERE}/framework_envs.generated.sh}"
BASE_PYTHON="${PYTHON_BIN:-/home/liangyilei/conda/envs/cuda-opt/bin/python}"
CUTLASS_ROOT="${CUTLASS_DIR:-/home/liangyilei/cursor_home/Ladder/3rdparty/cutlass}"
MODEL_PATH="${SERVING_MODEL_PATH:-}"
MODE=full
OUTPUT=""
PREFLIGHT_ONLY=0
RUN_640="${RUN_ALL_640:-0}"

usage() {
  cat <<'EOF'
usage: run_native_framework_validation_gpu1.sh --model PATH [options]

Options:
  --model PATH          Local Hugging Face-format model directory (required)
  --output DIR          Result directory; default is timestamped
  --quick | --full      Benchmark mode; default full
  --run-640             Re-run the full 640-case PyTorch/Triton suite
  --preflight-only      Check installations/GPU/model without benchmarks
  --env-file PATH       Generated environment file from the installer

Environment:
  GPU_PHYSICAL_INDEX=1, SERVING_MODEL_NAME, SGLANG_PORT, VLLM_PORT, SERVING_CASES
  CHECK_SOURCE_URLS=0|1, SERVING_STARTUP_TIMEOUT, ARCH=sm_86
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_PATH="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --quick) MODE=quick; shift ;;
    --full) MODE=full; shift ;;
    --run-640) RUN_640=1; shift ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -f "${ENV_FILE}" ]] || {
  echo "missing ${ENV_FILE}; run install_native_framework_envs.sh first" >&2; exit 1;
}
# shellcheck disable=SC1090
source "${ENV_FILE}"

[[ -x "${VLLM_PYTHON:-}" ]] || { echo "VLLM_PYTHON is not executable" >&2; exit 1; }
[[ -x "${SGLANG_PYTHON:-}" ]] || { echo "SGLANG_PYTHON is not executable" >&2; exit 1; }
[[ -x "${TVM_PYTHON:-}" ]] || { echo "TVM_PYTHON is not executable" >&2; exit 1; }
[[ -x "${BASE_PYTHON}" ]] || { echo "base benchmark Python is missing: ${BASE_PYTHON}" >&2; exit 1; }
[[ -f "${CUTLASS_ROOT}/include/cutlass/cutlass.h" ]] || {
  echo "CUTLASS headers missing: ${CUTLASS_ROOT}" >&2; exit 1;
}
[[ -n "${MODEL_PATH}" ]] || { echo "--model/ SERVING_MODEL_PATH is required" >&2; exit 2; }
[[ -f "${MODEL_PATH}/config.json" ]] || {
  echo "model does not look like a local Hugging Face model: ${MODEL_PATH}/config.json missing" >&2; exit 1;
}

if ! nvidia-smi --query-gpu=index,name,driver_version,memory.total \
  --format=csv,noheader -i "${GPU_PHYSICAL_INDEX}"; then
  echo "physical GPU ${GPU_PHYSICAL_INDEX} is not visible to the NVIDIA driver" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_PHYSICAL_INDEX}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export TVM_LIBRARY_PATH="${TVM_LIBRARY_PATH:-}"
if [[ -n "${TVM_LIBRARY_PATH}" ]]; then
  export LD_LIBRARY_PATH="${TVM_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
fi

echo "[preflight] vLLM"
"${VLLM_PYTHON}" -c \
  'import torch,vllm; assert torch.cuda.is_available(); print(vllm.__version__,torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))'
"${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server --help >/dev/null

echo "[preflight] SGLang"
"${SGLANG_PYTHON}" -c \
  'import torch,sglang; assert torch.cuda.is_available(); print(getattr(sglang,"__version__","unknown"),torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))'
"${SGLANG_PYTHON}" -m sglang.launch_server --help >/dev/null

echo "[preflight] TVM"
"${TVM_PYTHON}" -c \
  'import ctypes,tvm; import cuda.bindings; d=ctypes.CDLL("libcuda.so.1"); assert d.cuInit(0)==0; i=tvm.support.libinfo(); assert i.get("USE_CUDA") not in (None,"OFF"); assert tvm.cuda(0).exist; print(tvm.__file__,i.get("USE_CUDA"),i.get("CUDA_VERSION"),tvm.cuda(0))'

echo "[preflight] base PyTorch/Triton"
"${BASE_PYTHON}" -c \
  'import torch,triton; assert torch.cuda.is_available(); print(torch.__version__,triton.__version__,torch.cuda.get_device_name(0))'

if [[ "${PREFLIGHT_ONLY}" == 1 ]]; then
  echo "[preflight] all requested environments are ready"
  exit 0
fi

if [[ -z "${OUTPUT}" ]]; then
  OUTPUT="${HERE}/results/native_frameworks_$(date +%Y%m%d_%H%M%S)"
fi

export GPU_PHYSICAL_INDEX
export PYTHON_BIN="${BASE_PYTHON}"
export TORCH_PYTHON="${BASE_PYTHON}"
export VLLM_PYTHON
export SGLANG_PYTHON
export TVM_PYTHON
export CUTLASS_DIR="${CUTLASS_ROOT}"
export SERVING_MODEL_PATH="${MODEL_PATH}"
export CHECK_SOURCE_URLS="${CHECK_SOURCE_URLS:-1}"
export RUN_ALL_640="${RUN_640}"
export SERVING_STARTUP_TIMEOUT="${SERVING_STARTUP_TIMEOUT:-900}"
export ARCH="${ARCH:-sm_86}"

bash "${HERE}/run_observation_validation_gpu1.sh" "--${MODE}" --output "${OUTPUT}"

"${BASE_PYTHON}" -c '
import json,sys
path=sys.argv[1]
required={"triton_kv_layout","tvm_layout","tvm_rq_observations","cutlass_softmax","sglang_native_serving","vllm_native_serving"}
rows=[json.loads(line) for line in open(path,encoding="utf-8") if line.strip()]
latest={row["step"]:row for row in rows}
bad=[]
for name in sorted(required):
    row=latest.get(name)
    status=row.get("status") if row else "missing"
    detail=row.get("detail","") if row else ""
    print(f"{name}\t{status}\t{detail}")
    if status != "success": bad.append(f"{name}={status}")
if bad:
    raise SystemExit("native framework validation incomplete: " + ", ".join(bad))
' "${OUTPUT}/canonical/status.jsonl"

echo "[native-validation] all required framework adapters succeeded"
echo "[native-validation] report=${OUTPUT}/OBSERVATION_VALIDATION_REPORT.md"
