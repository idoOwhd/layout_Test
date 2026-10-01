#!/usr/bin/env bash
# v13 RQ validation on physical GPU 1. Results are always timestamped.
set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${FRAMEWORK_ENV_FILE:-${HERE}/framework_envs.generated.sh}"
GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX:-1}"
MODE=smoke
OUTPUT=""
MODEL_PATH="${SERVING_MODEL_PATH:-${HERE}/models/Qwen2.5-0.5B-Instruct-7ae5576}"
RUN_SERVING=0
ANALYSIS_ONLY=0

usage() {
  cat <<'EOF'
usage: run_v13_validation_gpu1.sh [--smoke|--full] [--output DIR] [--model DIR]
                                  [--with-serving] [--skip-serving] [--analysis-only]

--smoke runs one representative shape per LLM subgraph and a minimal native
        vLLM/SGLang policy pair.
--full  runs min/median/max shapes per subgraph, all native policy variants,
        and the 640-case external-validity suite unless RUN_ALL_640=0.
--with-serving runs the legacy HTTP-control-plane diagnostic. For measured
        native evidence use run_v13_native_serving_gpu1.sh, or use
        run_v13_all_gpu1.sh to run subgraphs and native engines together.

Distributed RQ5 and cross-hardware H4.4 are recorded as blocked_single_gpu.
Every default output directory contains v13_smoke_* or v13_full_* and never
overwrites an older result.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) MODE=smoke; shift ;;
    --full) MODE=full; shift ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --with-serving) RUN_SERVING=1; shift ;;
    --skip-serving) RUN_SERVING=0; shift ;;
    --analysis-only) ANALYSIS_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${OUTPUT}" ]]; then
  OUTPUT="${HERE}/results/v13_${MODE}_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${OUTPUT}" "${OUTPUT}/raw" "${OUTPUT}/logs"
STATUS="${OUTPUT}/v13_status.jsonl"
: >"${STATUS}"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi
PYTHON_BIN="${PYTHON_BIN:-/home/liangyilei/conda/envs/cuda-opt/bin/python}"
TORCH_PYTHON="${TORCH_PYTHON:-${PYTHON_BIN}}"
VLLM_PYTHON="${VLLM_PYTHON:-${PYTHON_BIN}}"
SGLANG_PYTHON="${SGLANG_PYTHON:-${PYTHON_BIN}}"
TVM_PYTHON="${TVM_PYTHON:-${PYTHON_BIN}}"

record() {
  "${PYTHON_BIN}" -c 'import json,sys; open(sys.argv[1],"a",encoding="utf-8").write(json.dumps({"step":sys.argv[2],"status":sys.argv[3],"detail":sys.argv[4]})+"\n")' \
    "${STATUS}" "$1" "$2" "${3:-}"
}

run_logged() {
  local name="$1"; shift
  echo "[v13] ${name}"
  if "$@" >"${OUTPUT}/logs/${name}.log" 2>&1; then
    record "${name}" success "${OUTPUT}/logs/${name}.log"
  else
    local code=$?
    record "${name}" failed "exit=${code}; ${OUTPUT}/logs/${name}.log"
    echo "[v13] WARNING ${name} failed; see ${OUTPUT}/logs/${name}.log" >&2
  fi
}

echo "[v13] output=${OUTPUT} physical_gpu=${GPU_PHYSICAL_INDEX} mode=${MODE}"
run_logged source_observation_audit "${PYTHON_BIN}" "${HERE}/analyze_v13_source_chain.py" \
  --output-dir "${OUTPUT}"

if [[ "${ANALYSIS_ONLY}" == 0 ]]; then
  if [[ "${RUN_SERVING}" == 1 ]] && ! compgen -G "${MODEL_PATH}/*.safetensors" >/dev/null; then
    run_logged prepare_model env -u PYTHONPATH -u PYTHONHOME HF_HUB_DISABLE_XET=1 "${VLLM_PYTHON}" \
      "${HERE}/prepare_pinned_serving_model.py" --output "${MODEL_PATH}"
  fi

  export GPU_PHYSICAL_INDEX PYTHON_BIN TORCH_PYTHON VLLM_PYTHON SGLANG_PYTHON TVM_PYTHON
  export CUDA_WARMUP="${CUDA_WARMUP:-8}" CUDA_ITERATIONS="${CUDA_ITERATIONS:-30}"
  export TRITON_WARMUP="${TRITON_WARMUP:-8}" TRITON_ITERATIONS="${TRITON_ITERATIONS:-30}"
  export CHECK_SOURCE_URLS="${CHECK_SOURCE_URLS:-0}"
  export SERVING_STARTUP_TIMEOUT="${SERVING_STARTUP_TIMEOUT:-900}"
  export SERVING_REQUEST_TIMEOUT="${SERVING_REQUEST_TIMEOUT:-600}"
  if [[ "${RUN_SERVING}" == 1 ]] && [[ -f "${MODEL_PATH}/config.json" ]] && \
      compgen -G "${MODEL_PATH}/*.safetensors" >/dev/null; then
    if CUDA_VISIBLE_DEVICES="${GPU_PHYSICAL_INDEX}" "${VLLM_PYTHON}" \
        "${HERE}/check_serving_model_fit.py" --model "${MODEL_PATH}" \
        >"${OUTPUT}/logs/serving_model_fit.log" 2>&1; then
      record serving_model_fit success "${OUTPUT}/logs/serving_model_fit.log"
      export SERVING_MODEL_PATH="${MODEL_PATH}"
    else
      record serving_model_fit skipped_insufficient_memory "${OUTPUT}/logs/serving_model_fit.log"
      unset SERVING_MODEL_PATH
    fi
  else
    unset SERVING_MODEL_PATH
  fi

  if [[ "${MODE}" == smoke ]]; then
    export RUN_ALL_640=0
    export SERVING_CASES="${SERVING_CASES:-128:1:3:1,1024:4:3:1}"
    export SGLANG_VARIANTS="${SGLANG_VARIANTS:-auto_nhd,auto_hnd}"
    export VLLM_VARIANTS="${VLLM_VARIANTS:-auto,LBNHC}"
    run_logged raw_framework_suite bash "${HERE}/run_ref_talks_rq_gpu1.sh" \
      --quick --output "${OUTPUT}/raw"
  else
    # The min/median/max suite already covers every common subgraph with
    # multiple shapes. The legacy 640-case replay is separately opt-in.
    export RUN_ALL_640="${RUN_ALL_640:-0}"
    export SERVING_CASES="${SERVING_CASES:-128:1:8:1,2048:1:5:1,8192:16:5:1,2048:32:16:4}"
    unset SGLANG_VARIANTS VLLM_VARIANTS
    run_logged raw_framework_suite bash "${HERE}/run_ref_talks_rq_gpu1.sh" \
      --full --output "${OUTPUT}/raw"
  fi
else
  record raw_framework_suite skipped "analysis-only"
fi

run_logged v13_adjudication "${PYTHON_BIN}" "${HERE}/analyze_v13_rqs.py" \
  --raw-dir "${OUTPUT}/raw" --output-dir "${OUTPUT}" \
  --source-json "${OUTPUT}/v13_source_observations.json"
run_logged v13_framework_matrix "${PYTHON_BIN}" "${HERE}/analyze_v13_framework_matrix.py" \
  --raw-dir "${OUTPUT}/raw" --output-dir "${OUTPUT}" \
  --source-json "${OUTPUT}/v13_source_observations.json"
run_logged v13_640_framework_rq_matrix "${PYTHON_BIN}" "${HERE}/analyze_v13_640_matrix.py" \
  --raw-dir "${OUTPUT}/raw" --output-dir "${OUTPUT}"

echo "[v13] complete=${OUTPUT}"
echo "[v13] report=${OUTPUT}/V13_RQ_VALIDATION_REPORT.md"
echo "[v13] source-report=${OUTPUT}/V13_SOURCE_OBSERVATIONS.md"
echo "[v13] framework-matrix=${OUTPUT}/V13_FRAMEWORK_MATRIX.md"
echo "[v13] 640-framework-rq-matrix=${OUTPUT}/V13_640_FRAMEWORK_RQ_MATRIX.md"
echo "[v13] status=${STATUS}"
