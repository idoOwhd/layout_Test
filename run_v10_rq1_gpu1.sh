#!/usr/bin/env bash
# Preregistered layout_summary_v10 L-RQ1 runner on physical GPU 1.
set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="$(cd "${HERE}/.." && pwd)"
GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX:-1}"
ARCH="${ARCH:-sm_86}"
MODE=full
OUTPUT=""

if [[ -z "${PYTHON_BIN:-}" ]] && [[ -x /home/liangyilei/conda/envs/cuda-opt/bin/python ]]; then
  PYTHON_BIN=/home/liangyilei/conda/envs/cuda-opt/bin/python
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
TORCH_PYTHON="${TORCH_PYTHON:-${PYTHON_BIN}}"
SGLANG_PYTHON="${SGLANG_PYTHON:-${PYTHON_BIN}}"
VLLM_PYTHON="${VLLM_PYTHON:-${PYTHON_BIN}}"

usage() {
  echo "usage: $0 [--quick|--full] [--output DIR]"
  echo "physical GPU defaults to 1; v10 requires >=5 process repetitions"
  echo "env: PYTHON_BIN TORCH_PYTHON SGLANG_PYTHON VLLM_PYTHON SERVING_MODEL_PATH"
  echo "     V10_PROCESS_REPETITIONS=5 GPU_PHYSICAL_INDEX=1 ARCH=sm_86 CHECK_SOURCE_URLS=0|1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) MODE=quick; shift ;;
    --full) MODE=full; shift ;;
    --output) OUTPUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
if [[ -z "${OUTPUT}" ]]; then
  OUTPUT="${HERE}/results/v10_rq1_$(date +%Y%m%d_%H%M%S)"
fi
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 1; }
mkdir -p "${OUTPUT}/logs" "${OUTPUT}/native_server_logs" "${OUTPUT}/compiler_dumps"
STATUS="${OUTPUT}/status.jsonl"
: >"${STATUS}"
export CUDA_VISIBLE_DEVICES="${GPU_PHYSICAL_INDEX}"
export TORCHINDUCTOR_CACHE_DIR="${OUTPUT}/torchinductor_cache"
export TRITON_KERNEL_DUMP="${TRITON_KERNEL_DUMP:-1}"
export TRITON_DUMP_DIR="${OUTPUT}/compiler_dumps/triton"

record() {
  "${PYTHON_BIN}" -c 'import json,sys; open(sys.argv[1],"a",encoding="utf-8").write(json.dumps({"step":sys.argv[2],"status":sys.argv[3],"detail":sys.argv[4]})+"\n")' \
    "${STATUS}" "$1" "$2" "${3:-}"
}

run_logged() {
  local name="$1"; shift
  echo "[v10-rq1] ${name}"
  if "$@" >"${OUTPUT}/logs/${name}.log" 2>&1; then
    record "${name}" success "${OUTPUT}/logs/${name}.log"
    return 0
  fi
  local code=$?
  record "${name}" failed "exit=${code}; ${OUTPUT}/logs/${name}.log"
  return "${code}"
}

run_csv() {
  local name="$1"; local csv_path="$2"; shift 2
  echo "[v10-rq1] ${name}"
  if "$@" >"${csv_path}" 2>"${OUTPUT}/logs/${name}.log"; then
    record "${name}" success "${csv_path}"
    return 0
  fi
  local code=$?
  record "${name}" failed "exit=${code}; ${OUTPUT}/logs/${name}.log"
  return "${code}"
}

echo "[v10-rq1] output=${OUTPUT} physical_gpu=${GPU_PHYSICAL_INDEX} logical_gpu=0 mode=${MODE}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
    --format=csv,noheader -i "${GPU_PHYSICAL_INDEX}" \
    >"${OUTPUT}/logs/gpu_identity.log" 2>&1 || true
fi

AUDIT_ARGS=(--output "${OUTPUT}/ref_talks_source_audit.json" --report "${OUTPUT}/REF_TALKS_SOURCE_AUDIT.md")
if [[ "${CHECK_SOURCE_URLS:-0}" == 1 ]]; then AUDIT_ARGS+=(--verify-urls); fi
run_logged source_audit "${PYTHON_BIN}" "${HERE}/audit_ref_talks_evidence.py" "${AUDIT_ARGS[@]}" || true
run_logged v10_unit_tests "${PYTHON_BIN}" -m unittest discover -s "${HERE}" -p 'test_v10_rq1.py' -v || true

CASE_ARGS=(--output-dir "${OUTPUT}")
if [[ "${MODE}" == quick ]]; then CASE_ARGS+=(--quick); fi
run_logged build_v10_cases "${PYTHON_BIN}" "${HERE}/build_v10_rq1_cases.py" "${CASE_ARGS[@]}" || true

if ! command -v nvcc >/dev/null 2>&1; then
  record build_v10_cuda unavailable "nvcc not found"
else
  run_logged build_v10_cuda nvcc -O3 -std=c++17 "-arch=${ARCH}" \
    "${HERE}/v10_rq1_edge_bench.cu" -o "${OUTPUT}/v10_rq1_edge_bench" || true
fi

REPETITIONS="${V10_PROCESS_REPETITIONS:-5}"
if (( REPETITIONS < 5 )); then
  record repetition_warning protocol_violation "v10 requires at least 5 process repetitions; configured ${REPETITIONS}"
fi
if [[ "${MODE}" == quick ]]; then
  WARMUP="${V10_WARMUP:-3}"; ITERATIONS="${V10_ITERATIONS:-8}"
else
  WARMUP="${V10_WARMUP:-8}"; ITERATIONS="${V10_ITERATIONS:-30}"
fi

if [[ -x "${OUTPUT}/v10_rq1_edge_bench" ]] && [[ -s "${OUTPUT}/v10_rq1_core_cases.tsv" ]]; then
  for ((rep=1; rep<=REPETITIONS; rep++)); do
    run_csv "v10_core_rep${rep}" "${OUTPUT}/v10_rq1_core_rep${rep}.csv" \
      "${OUTPUT}/v10_rq1_edge_bench" --cases "${OUTPUT}/v10_rq1_core_cases.tsv" \
      --run-id "core-rep-${rep}" --warmup "${WARMUP}" --iterations "${ITERATIONS}" || true
  done
else
  record v10_core unavailable "compiled benchmark or core case file missing"
fi

run_logged v10_stage3_gate "${PYTHON_BIN}" "${HERE}/analyze_v10_rq1.py" \
  --input-dir "${OUTPUT}" --pattern 'v10_rq1_core_rep*.csv' \
  --source-audit "${OUTPUT}/ref_talks_source_audit.json" --output-dir "${OUTPUT}" --gate-only || true

STAGE4_GO=0
if [[ -f "${OUTPUT}/v10_rq1_gate.json" ]]; then
  STAGE4_GO="$("${PYTHON_BIN}" -c 'import json,sys; print(1 if json.load(open(sys.argv[1]))["stage4_gate"]["go"] else 0)' "${OUTPUT}/v10_rq1_gate.json")"
fi

if [[ "${STAGE4_GO}" == 1 ]]; then
  record stage4_gate GO "Stage3 passed; executing preregistered crossover surface"
  for ((rep=1; rep<=REPETITIONS; rep++)); do
    run_csv "v10_sensitivity_rep${rep}" "${OUTPUT}/v10_rq1_sensitivity_rep${rep}.csv" \
      "${OUTPUT}/v10_rq1_edge_bench" --cases "${OUTPUT}/v10_rq1_sensitivity_cases.tsv" \
      --run-id "sensitivity-rep-${rep}" --warmup "${WARMUP}" --iterations "${ITERATIONS}" || true
  done

  # Stage 5 external validity is conditional. It covers all eight common LLM
  # subgraphs and multiple min/median/max contracts without claiming that the
  # generic boundary stride experiment is a native FlashInfer/vLLM result.
  BROAD_ARGS=(--output-dir "${OUTPUT}/stage5_cases")
  if [[ "${MODE}" == quick ]]; then BROAD_ARGS+=(--quick); else BROAD_ARGS+=(--per-group 3); fi
  run_logged stage5_build_common_llm_cases "${PYTHON_BIN}" "${HERE}/build_rq_llm_cases.py" "${BROAD_ARGS[@]}" || true
  if "${TORCH_PYTHON}" -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
    run_logged stage5_boundary_layout_sweep "${TORCH_PYTHON}" "${HERE}/llm_boundary_layout_sweep.py" \
      --manifest "${OUTPUT}/stage5_cases/rq_llm_representative_cases.json" \
      --output "${OUTPUT}/llm_boundary_layout_sweep.jsonl" \
      --physical-device-index "${GPU_PHYSICAL_INDEX}" --warmup "${WARMUP}" --iterations "${ITERATIONS}" || true
    run_logged stage5_pytorch_triton "${TORCH_PYTHON}" "${BASELINE}/benchmark_real_world.py" \
      --manifest "${OUTPUT}/stage5_cases/rq_llm_representative_cases.json" \
      --output "${OUTPUT}/llm_representative_pytorch_triton.jsonl" \
      --physical-device-index "${GPU_PHYSICAL_INDEX}" --no-deduplicate --no-resume \
      --warmup "${WARMUP}" --iterations "${ITERATIONS}" || true
  else
    record stage5_boundary_layout_sweep unavailable "torch CUDA unavailable in ${TORCH_PYTHON}"
    record stage5_pytorch_triton unavailable "torch CUDA unavailable in ${TORCH_PYTHON}"
  fi
else
  record stage4_gate STOP "Stage3 did not pass v10 statistical gate; Stage4/5 intentionally skipped"
fi

# Whole-engine runs are optional external validity and never replace Stage 3.
if [[ "${STAGE4_GO}" == 1 ]] && [[ -n "${SERVING_MODEL_PATH:-}" ]]; then
  SERVING=(--model "${SERVING_MODEL_PATH}" --log-dir "${OUTPUT}/native_server_logs" \
    --startup-timeout "${SERVING_STARTUP_TIMEOUT:-600}")
  if [[ -n "${SERVING_MODEL_NAME:-}" ]]; then SERVING+=(--served-model-name "${SERVING_MODEL_NAME}"); fi
  if "${SGLANG_PYTHON}" -c 'import importlib.util; assert importlib.util.find_spec("sglang")' >/dev/null 2>&1; then
    run_logged stage5_sglang "${SGLANG_PYTHON}" "${HERE}/native_serving_layout_bench.py" \
      --framework sglang --port "${SGLANG_PORT:-31000}" \
      --output "${OUTPUT}/sglang_native_serving.jsonl" "${SERVING[@]}" || true
  else
    record stage5_sglang unavailable "sglang not installed"
  fi
  if "${VLLM_PYTHON}" -c 'import importlib.util; assert importlib.util.find_spec("vllm")' >/dev/null 2>&1; then
    run_logged stage5_vllm "${VLLM_PYTHON}" "${HERE}/native_serving_layout_bench.py" \
      --framework vllm --port "${VLLM_PORT:-32000}" \
      --output "${OUTPUT}/vllm_native_serving.jsonl" "${SERVING[@]}" || true
  else
    record stage5_vllm unavailable "vllm not installed"
  fi
else
  record stage5_native_serving skipped "requires Stage3 GO and SERVING_MODEL_PATH"
fi

run_logged v10_final_adjudication "${PYTHON_BIN}" "${HERE}/analyze_v10_rq1.py" \
  --input-dir "${OUTPUT}" --pattern 'v10_rq1_*_rep*.csv' \
  --source-audit "${OUTPUT}/ref_talks_source_audit.json" --output-dir "${OUTPUT}" || true

echo "[v10-rq1] complete: ${OUTPUT}"
echo "[v10-rq1] report: ${OUTPUT}/V10_RQ1_VALIDATION_REPORT.md"
echo "[v10-rq1] experiment map: ${OUTPUT}/v10_rq1_experiment_map.json"
