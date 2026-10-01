#!/usr/bin/env bash
# One non-overwriting command for strict v10 L-RQ1 plus repaired v13 640 suite.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX:-1}"
PYTHON_BIN="${PYTHON_BIN:-/home/liangyilei/conda/envs/cuda-opt/bin/python}"
TORCH_PYTHON="${TORCH_PYTHON:-${PYTHON_BIN}}"
ENV_FILE="${FRAMEWORK_ENV_FILE:-${HERE}/framework_envs.generated.sh}"
MODEL_PATH="${SERVING_MODEL_PATH:-${HERE}/models/Qwen2.5-0.5B-Instruct-7ae5576}"
OUTPUT=""

usage() {
  cat <<'EOF'
usage: run_v10_v13_640_all_frameworks_gpu1.sh [--output NEW_DIR] [--model DIR]

Runs strict v10 L-RQ1 and the repaired v13 640-case/six-framework fail-closed
suite on physical GPU 1. Existing output paths are always rejected.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
if [[ -z "${OUTPUT}" ]]; then
  OUTPUT="${HERE}/results/v10_v13_640_all_frameworks_$(date +%Y%m%d_%H%M%S)"
fi
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 1; }
[[ -x "${PYTHON_BIN}" ]] || { echo "missing PYTHON_BIN=${PYTHON_BIN}" >&2; exit 1; }
[[ -x "${TORCH_PYTHON}" ]] || { echo "missing TORCH_PYTHON=${TORCH_PYTHON}" >&2; exit 1; }
[[ -f "${ENV_FILE}" ]] || { echo "missing framework env ${ENV_FILE}" >&2; exit 1; }
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "missing model ${MODEL_PATH}" >&2; exit 1; }
mkdir -p "${OUTPUT}"

# Fail before a long GPU run if the repaired semantics regress.
PYTHONPATH="${HERE}:${HERE}/.." "${PYTHON_BIN}" -m unittest \
  "${HERE}/test_repaired_workloads.py" \
  "${HERE}/test_v13_640_matrix.py" \
  "${HERE}/test_native_serving_layout_bench.py" \
  "${HERE}/test_native_framework_subgraph_probe.py" \
  "${HERE}/test_v13_rqs.py" \
  "${HERE}/test_v10_rq1.py"

# shellcheck disable=SC1090
source "${ENV_FILE}"
export GPU_PHYSICAL_INDEX PYTHON_BIN TORCH_PYTHON
export FRAMEWORK_ENV_FILE="${ENV_FILE}" SERVING_MODEL_PATH="${MODEL_PATH}"
export CHECK_SOURCE_URLS="${CHECK_SOURCE_URLS:-0}"
export CUTLASS_DIR="${CUTLASS_DIR:-/home/liangyilei/cursor_home/Ladder/3rdparty/cutlass}"
export V10_PROCESS_REPETITIONS="${V10_PROCESS_REPETITIONS:-5}"
export V10_WARMUP="${V10_WARMUP:-8}" V10_ITERATIONS="${V10_ITERATIONS:-30}"
export CUDA_WARMUP="${CUDA_WARMUP:-8}" CUDA_ITERATIONS="${CUDA_ITERATIONS:-30}"
export TRITON_WARMUP="${TRITON_WARMUP:-8}" TRITON_ITERATIONS="${TRITON_ITERATIONS:-30}"
export ALL_640_WARMUP="${ALL_640_WARMUP:-3}" ALL_640_ITERATIONS="${ALL_640_ITERATIONS:-10}"
export KV_BATCH_WARMUP="${KV_BATCH_WARMUP:-${ALL_640_WARMUP}}"
export KV_BATCH_ITERATIONS="${KV_BATCH_ITERATIONS:-${ALL_640_ITERATIONS}}"
export KV_BATCH_TIMEOUT="${KV_BATCH_TIMEOUT:-28800}"
export KV_BATCH_MAX_BYTES="${KV_BATCH_MAX_BYTES:-2147483648}"
export REAL_WORLD_CASE_TIMEOUT="${REAL_WORLD_CASE_TIMEOUT:-1200}"
export SERVING_STARTUP_TIMEOUT="${SERVING_STARTUP_TIMEOUT:-1200}"
export SERVING_REQUEST_TIMEOUT="${SERVING_REQUEST_TIMEOUT:-1200}"

set +e
bash "${HERE}/run_v10_rq1_gpu1.sh" --full --output "${OUTPUT}/v10_strict"
V10_RUN_EXIT=$?
set -e

set +e
GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX}" \
PYTHON_BIN="${PYTHON_BIN}" TORCH_PYTHON="${TORCH_PYTHON}" \
FRAMEWORK_ENV_FILE="${ENV_FILE}" SERVING_MODEL_PATH="${MODEL_PATH}" \
CHECK_SOURCE_URLS="${CHECK_SOURCE_URLS}" \
CUDA_WARMUP="${CUDA_WARMUP}" CUDA_ITERATIONS="${CUDA_ITERATIONS}" \
TRITON_WARMUP="${TRITON_WARMUP}" TRITON_ITERATIONS="${TRITON_ITERATIONS}" \
ALL_640_WARMUP="${ALL_640_WARMUP}" ALL_640_ITERATIONS="${ALL_640_ITERATIONS}" \
REAL_WORLD_CASE_TIMEOUT="${REAL_WORLD_CASE_TIMEOUT}" \
SERVING_STARTUP_TIMEOUT="${SERVING_STARTUP_TIMEOUT}" \
SERVING_REQUEST_TIMEOUT="${SERVING_REQUEST_TIMEOUT}" \
bash "${HERE}/run_v13_640_all_frameworks_gpu1.sh" \
  --output "${OUTPUT}/v13_640" --model "${MODEL_PATH}"
V13_RUN_EXIT=$?
set -e

set +e
"${PYTHON_BIN}" "${HERE}/audit_v10_v13_completion.py" \
  --v10 "${OUTPUT}/v10_strict" --v13 "${OUTPUT}/v13_640" --output-dir "${OUTPUT}"
AUDIT_EXIT=$?
set -e
export V10_RUN_EXIT V13_RUN_EXIT AUDIT_EXIT

"${PYTHON_BIN}" - "${OUTPUT}" "${ROOT}" "${HERE}" "${ENV_FILE}" "${MODEL_PATH}" <<'PY'
import json, os, shlex, sys
from pathlib import Path

output, root, here, env_file, model = map(Path, sys.argv[1:])
replay_env_names = [
    "CHECK_SOURCE_URLS", "CUTLASS_DIR", "V10_PROCESS_REPETITIONS", "V10_WARMUP",
    "V10_ITERATIONS", "CUDA_WARMUP", "CUDA_ITERATIONS", "TRITON_WARMUP",
    "TRITON_ITERATIONS", "ALL_640_WARMUP", "ALL_640_ITERATIONS",
    "KV_BATCH_WARMUP", "KV_BATCH_ITERATIONS", "KV_BATCH_TIMEOUT",
    "KV_BATCH_MAX_BYTES",
    "REAL_WORLD_CASE_TIMEOUT", "SERVING_STARTUP_TIMEOUT",
    "SERVING_REQUEST_TIMEOUT"]
env_lines = " \\\n".join(
    f"{name}={shlex.quote(os.environ.get(name, ''))}"
    for name in replay_env_names)
text = f'''#!/usr/bin/env bash
set -euo pipefail
cd {shlex.quote(str(root))}
OUT={shlex.quote(str(output) + '_replay')}_$(date +%Y%m%d_%H%M%S)
GPU_PHYSICAL_INDEX={shlex.quote(os.environ['GPU_PHYSICAL_INDEX'])} \\
PYTHON_BIN={shlex.quote(os.environ['PYTHON_BIN'])} TORCH_PYTHON={shlex.quote(os.environ['TORCH_PYTHON'])} \\
FRAMEWORK_ENV_FILE={shlex.quote(str(env_file))} SERVING_MODEL_PATH={shlex.quote(str(model))} \\
{env_lines} \\
bash {shlex.quote(str(here / 'run_v10_v13_640_all_frameworks_gpu1.sh'))} --output "$OUT" --model {shlex.quote(str(model))}
'''
path = output / "REPRODUCE_THIS_RUN.sh"
path.write_text(text, encoding="utf-8")
path.chmod(0o755)
(output / "EXECUTION_STATUS.json").write_text(json.dumps({
    "v10_run_exit": int(os.environ["V10_RUN_EXIT"]),
    "v13_run_exit": int(os.environ["V13_RUN_EXIT"]),
    "audit_exit": int(os.environ["AUDIT_EXIT"]),
    "all_stages_succeeded": all(int(os.environ[name]) == 0 for name in
                                ("V10_RUN_EXIT", "V13_RUN_EXIT", "AUDIT_EXIT")),
}, indent=2) + "\n", encoding="utf-8")
PY

echo "[v10-v13-640] complete=${OUTPUT}"
echo "[v10-v13-640] audit=${OUTPUT}/CROSS_VERSION_COMPLETENESS.md"
echo "[v10-v13-640] exits: v10=${V10_RUN_EXIT} v13=${V13_RUN_EXIT} audit=${AUDIT_EXIT}"
if (( V10_RUN_EXIT != 0 || V13_RUN_EXIT != 0 || AUDIT_EXIT != 0 )); then
  echo "[v10-v13-640] EXECUTION FAILED: inspect EXECUTION_STATUS.json and audit report" >&2
  exit 1
fi
