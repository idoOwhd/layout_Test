#!/usr/bin/env bash
# One-command validation for the RQs defined in ref_talks.
# Physical GPU 1 is exposed as logical cuda:0 inside every child process.
set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="$(cd "${HERE}/.." && pwd)"
GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX:-1}"
ARCH="${ARCH:-sm_86}"
MODE="full"
OUTPUT=""

if [[ -z "${PYTHON_BIN:-}" ]] && [[ -x /home/liangyilei/conda/envs/cuda-opt/bin/python ]]; then
  PYTHON_BIN=/home/liangyilei/conda/envs/cuda-opt/bin/python
elif [[ -z "${PYTHON_BIN:-}" ]] && [[ -x /tmp/tilelang-a10-env/bin/python ]]; then
  PYTHON_BIN=/tmp/tilelang-a10-env/bin/python
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
TORCH_PYTHON="${TORCH_PYTHON:-${PYTHON_BIN}}"
TILELANG_PYTHON="${TILELANG_PYTHON:-${TORCH_PYTHON}}"
TVM_PYTHON="${TVM_PYTHON:-${PYTHON_BIN}}"
SGLANG_PYTHON="${SGLANG_PYTHON:-${PYTHON_BIN}}"
VLLM_PYTHON="${VLLM_PYTHON:-${PYTHON_BIN}}"
TVM_LD_LIBRARY_PATH="${TVM_LIBRARY_PATH:-}${TVM_LIBRARY_PATH:+:}${LD_LIBRARY_PATH:-}"

usage() {
  echo "usage: $0 [--quick|--full] [--output DIR]"
  echo "default: physical GPU 1, full representative multi-shape suite"
  echo "env: PYTHON_BIN TORCH_PYTHON TILELANG_PYTHON TVM_PYTHON SGLANG_PYTHON VLLM_PYTHON"
  echo "     GPU_PHYSICAL_INDEX=1 ARCH=sm_86 CHECK_SOURCE_URLS=0|1 RUN_ALL_640=0|1"
  echo "optional native serving: SERVING_MODEL_PATH, SERVING_MODEL_NAME, SGLANG_PORT, VLLM_PORT"
  echo "  SERVING_CASES=128:1:8:1,2048:1:5:1,8192:16:5:1,2048:32:16:4"
  echo "optional artifact: RUN_HEXCUTE_ARTIFACT=1 HEXCUTE_BENCH_DIR=/path/to/artifact"
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
  OUTPUT="${HERE}/results/ref_talks_rq_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${OUTPUT}/logs" "${OUTPUT}/compiler_dumps" "${OUTPUT}/native_server_logs"
STATUS="${OUTPUT}/status.jsonl"
: >"${STATUS}"

export CUDA_VISIBLE_DEVICES="${GPU_PHYSICAL_INDEX}"
export TORCHINDUCTOR_CACHE_DIR="${OUTPUT}/torchinductor_cache"
export TORCH_COMPILE_DEBUG_DIR="${OUTPUT}/compiler_dumps/torch"
export TRITON_KERNEL_DUMP="${TRITON_KERNEL_DUMP:-1}"
export TRITON_DUMP_DIR="${OUTPUT}/compiler_dumps/triton"
export TVM_CUDA_ARCH="${TVM_CUDA_ARCH:-${ARCH}}"

record() {
  "${PYTHON_BIN}" -c 'import json,sys; open(sys.argv[1],"a",encoding="utf-8").write(json.dumps({"step":sys.argv[2],"status":sys.argv[3],"detail":sys.argv[4]})+"\n")' \
    "${STATUS}" "$1" "$2" "${3:-}"
}

run_logged() {
  local name="$1"; shift
  echo "[ref-talks-rq] ${name}"
  if "$@" >"${OUTPUT}/logs/${name}.log" 2>&1; then
    record "${name}" success "${OUTPUT}/logs/${name}.log"
  else
    local code=$?
    record "${name}" failed "exit=${code}; ${OUTPUT}/logs/${name}.log"
  fi
}

run_csv() {
  local name="$1"; local csv_path="$2"; shift 2
  echo "[ref-talks-rq] ${name}"
  if "$@" >"${csv_path}" 2>"${OUTPUT}/logs/${name}.log"; then
    record "${name}" success "${csv_path}"
  else
    local code=$?
    record "${name}" failed "exit=${code}; ${OUTPUT}/logs/${name}.log"
  fi
}

echo "[ref-talks-rq] output=${OUTPUT} physical_gpu=${GPU_PHYSICAL_INDEX} logical_gpu=0 mode=${MODE}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
    --format=csv,noheader -i "${GPU_PHYSICAL_INDEX}" \
    >"${OUTPUT}/logs/gpu_identity.log" 2>&1 || true
fi

AUDIT_ARGS=(--output "${OUTPUT}/ref_talks_source_audit.json" --report "${OUTPUT}/REF_TALKS_SOURCE_AUDIT.md")
if [[ "${CHECK_SOURCE_URLS:-0}" == 1 ]]; then AUDIT_ARGS+=(--verify-urls); fi
run_logged source_audit "${PYTHON_BIN}" "${HERE}/audit_ref_talks_evidence.py" "${AUDIT_ARGS[@]}"

run_logged rq_unit_tests "${PYTHON_BIN}" -m unittest discover -s "${HERE}" -p 'test_*rq*.py' -v
run_logged source_audit_unit_tests "${PYTHON_BIN}" -m unittest discover -s "${HERE}" -p 'test_ref_talks_evidence.py' -v

CASE_ARGS=(--output-dir "${OUTPUT}")
if [[ "${MODE}" == quick ]]; then CASE_ARGS+=(--quick); else CASE_ARGS+=(--per-group 3); fi
run_logged build_representative_cases "${PYTHON_BIN}" "${HERE}/build_rq_llm_cases.py" "${CASE_ARGS[@]}"

if [[ "${RUN_ALL_640:-0}" == 1 ]]; then
  run_logged build_all_640_cases "${PYTHON_BIN}" "${HERE}/build_rq_llm_cases.py" \
    --output-dir "${OUTPUT}" --all-cases
else
  record build_all_640_cases skipped "set RUN_ALL_640=1 for the exhaustive manifest"
fi

if command -v nvcc >/dev/null 2>&1; then
  run_logged build_cuda_reference nvcc -O3 -std=c++17 "-arch=${ARCH}" \
    "${HERE}/rq_llm_layout_bench.cu" -o "${OUTPUT}/rq_llm_layout_bench"
  if [[ -x "${OUTPUT}/rq_llm_layout_bench" ]] && [[ -s "${OUTPUT}/rq_attention_cases.tsv" ]]; then
    CUDA_ARGS=(--case-file "${OUTPUT}/rq_attention_cases.tsv")
    if [[ "${MODE}" == quick ]]; then
      CUDA_ARGS+=(--quick --warmup 3 --iterations 8)
    else
      CUDA_ARGS+=(--warmup "${CUDA_WARMUP:-8}" --iterations "${CUDA_ITERATIONS:-30}")
    fi
    run_csv cuda_rq_mechanisms "${OUTPUT}/rq_cuda_reference.csv" \
      "${OUTPUT}/rq_llm_layout_bench" "${CUDA_ARGS[@]}"
    if command -v cuobjdump >/dev/null 2>&1; then
      run_logged cuda_reference_sass cuobjdump --dump-sass "${OUTPUT}/rq_llm_layout_bench"
      cp "${OUTPUT}/logs/cuda_reference_sass.log" "${OUTPUT}/compiler_dumps/rq_cuda_reference.sass" 2>/dev/null || true
    fi
  else
    record cuda_rq_mechanisms unavailable "CUDA binary or case TSV missing"
  fi
else
  record build_cuda_reference unavailable "nvcc not found"
  record cuda_rq_mechanisms unavailable "nvcc not found"
fi

# External-validity suite: all eight common LLM subgraph families, both phases,
# and multiple min/median/max tensor contracts in full mode.
if "${TORCH_PYTHON}" -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  REP_ARGS=(--manifest "${OUTPUT}/rq_llm_representative_cases.json" \
    --physical-device-index "${GPU_PHYSICAL_INDEX}" \
    --output "${OUTPUT}/llm_representative_pytorch_triton.jsonl" \
    --no-deduplicate --no-resume)
  if [[ "${MODE}" == quick ]]; then
    # Smoke validates every subgraph without triggering a long torch.compile
    # search for each one. Triton is still exercised by the dedicated explicit
    # KV-layout benchmark below. Full mode retains both eager and compiled rows.
    REP_ARGS+=(--backend pytorch --warmup 1 --iterations 3)
  else
    REP_ARGS+=(--warmup 3 --iterations 10)
  fi
  if [[ "${MODE}" == full ]]; then
    run_logged llm_representative_pytorch_triton "${PYTHON_BIN}" "${HERE}/run_real_world_isolated.py" \
      --python "${TORCH_PYTHON}" --runner "${BASELINE}/benchmark_real_world.py" \
      --manifest "${OUTPUT}/rq_llm_representative_cases.json" \
      --physical-device-index "${GPU_PHYSICAL_INDEX}" \
      --output "${OUTPUT}/llm_representative_pytorch_triton.jsonl" \
      --log-dir "${OUTPUT}/logs/isolated_real_world" \
      --warmup 3 --iterations 10 --timeout "${REAL_WORLD_CASE_TIMEOUT:-300}"
  else
    run_logged llm_representative_pytorch_triton "${TORCH_PYTHON}" "${BASELINE}/benchmark_real_world.py" "${REP_ARGS[@]}"
  fi

  if [[ "${MODE}" == quick ]]; then
    BOUNDARY_WARMUP=1
    BOUNDARY_ITERATIONS=3
  else
    BOUNDARY_WARMUP=3
    BOUNDARY_ITERATIONS=10
  fi
  run_logged llm_boundary_layout_sweep "${PYTHON_BIN}" \
    "${HERE}/run_boundary_hybrid.py" \
    --runner "${HERE}/llm_boundary_layout_sweep.py" \
    --isolated-runner "${HERE}/run_boundary_isolated.py" \
    --python "${TORCH_PYTHON}" \
    --manifest "${OUTPUT}/rq_llm_representative_cases.json" \
    --physical-device-index "${GPU_PHYSICAL_INDEX}" \
    --output "${OUTPUT}/llm_boundary_layout_sweep.jsonl" \
    --log-dir "${OUTPUT}/logs/boundary_representative" \
    --warmup "${BOUNDARY_WARMUP}" \
    --iterations "${BOUNDARY_ITERATIONS}" \
    --timeout "${REAL_WORLD_CASE_TIMEOUT:-300}"

  TRITON_LAYOUT_ARGS=(--attention-tsv "${OUTPUT}/rq_attention_cases.tsv" \
    --output "${OUTPUT}/triton_kv_layout.csv")
  if [[ "${MODE}" == quick ]]; then
    TRITON_LAYOUT_ARGS+=(--warmup 3 --iterations 8)
  else
    TRITON_LAYOUT_ARGS+=(--warmup "${TRITON_WARMUP:-8}" --iterations "${TRITON_ITERATIONS:-30}")
  fi
  if "${TORCH_PYTHON}" -c 'import triton' >/dev/null 2>&1; then
    run_logged triton_kv_layout "${TORCH_PYTHON}" "${HERE}/triton_kv_layout_bench.py" \
      "${TRITON_LAYOUT_ARGS[@]}"
  else
    record triton_kv_layout unavailable "triton not installed in TORCH_PYTHON"
  fi

  if "${TILELANG_PYTHON}" -c 'import tilelang, torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
    TILE_ARGS=(--manifest "${OUTPUT}/rq_llm_representative_cases.json" \
      --physical-device-index "${GPU_PHYSICAL_INDEX}" \
      --output "${OUTPUT}/llm_representative_tilelang.jsonl" --no-resume)
    if [[ "${MODE}" == quick ]]; then TILE_ARGS+=(--warmup 1 --iterations 3); fi
    run_logged llm_representative_tilelang "${TILELANG_PYTHON}" "${BASELINE}/tilelang/benchmark_real_world.py" "${TILE_ARGS[@]}"
  else
    record llm_representative_tilelang unavailable "tilelang not installed in PYTHON_BIN"
  fi

  if [[ "${RUN_ALL_640:-0}" == 1 ]]; then
    run_logged llm_all_640_pytorch_triton "${PYTHON_BIN}" "${HERE}/run_real_world_isolated.py" \
      --python "${TORCH_PYTHON}" --runner "${BASELINE}/benchmark_real_world.py" \
      --manifest "${OUTPUT}/rq_llm_all_640_cases.json" \
      --physical-device-index "${GPU_PHYSICAL_INDEX}" \
      --output "${OUTPUT}/llm_640_pytorch_triton.jsonl" \
      --log-dir "${OUTPUT}/logs/isolated_all_640" \
      --warmup "${ALL_640_WARMUP:-3}" --iterations "${ALL_640_ITERATIONS:-10}" \
      --timeout "${REAL_WORLD_CASE_TIMEOUT:-300}"
    run_logged llm_all_640_boundary_layout_sweep "${PYTHON_BIN}" \
      "${HERE}/run_boundary_hybrid.py" \
      --runner "${HERE}/llm_boundary_layout_sweep.py" \
      --isolated-runner "${HERE}/run_boundary_isolated.py" \
      --python "${TORCH_PYTHON}" \
      --manifest "${OUTPUT}/rq_llm_all_640_cases.json" \
      --physical-device-index "${GPU_PHYSICAL_INDEX}" \
      --output "${OUTPUT}/llm_640_boundary_layout_sweep.jsonl" \
      --log-dir "${OUTPUT}/logs/boundary_all_640" \
      --warmup "${ALL_640_WARMUP:-3}" --iterations "${ALL_640_ITERATIONS:-10}" \
      --timeout "${REAL_WORLD_CASE_TIMEOUT:-300}"
    if [[ -x "${OUTPUT}/rq_llm_layout_bench" ]]; then
      run_csv cuda_rq_all_attention "${OUTPUT}/rq_cuda_all_attention.csv" \
        "${OUTPUT}/rq_llm_layout_bench" \
        --case-file "${OUTPUT}/rq_attention_all_cases.tsv" \
        --warmup "${CUDA_WARMUP:-8}" --iterations "${CUDA_ITERATIONS:-30}"
    else
      record cuda_rq_all_attention unavailable "CUDA reference binary missing"
    fi
    if "${TORCH_PYTHON}" -c 'import triton' >/dev/null 2>&1; then
      run_logged triton_all_640_attention "${TORCH_PYTHON}" "${HERE}/triton_kv_layout_bench.py" \
        --attention-tsv "${OUTPUT}/rq_attention_all_cases.tsv" \
        --output "${OUTPUT}/triton_640_kv_layout.csv" \
        --warmup "${TRITON_WARMUP:-8}" --iterations "${TRITON_ITERATIONS:-30}"
    else
      record triton_all_640_attention unavailable "triton not installed in TORCH_PYTHON"
    fi
  else
    record llm_all_640_pytorch_triton skipped "set RUN_ALL_640=1 for the exhaustive manifest"
    record llm_all_640_boundary_layout_sweep skipped "set RUN_ALL_640=1"
    record cuda_rq_all_attention skipped "set RUN_ALL_640=1"
    record triton_all_640_attention skipped "set RUN_ALL_640=1"
  fi
else
  record llm_representative_pytorch_triton unavailable "torch CUDA unavailable in ${TORCH_PYTHON}"
  record llm_boundary_layout_sweep unavailable "torch CUDA unavailable in ${TORCH_PYTHON}"
  record llm_representative_tilelang unavailable "torch CUDA unavailable in ${TORCH_PYTHON}"
fi

# Framework-native/local compiler probes.  Their outputs remain separate from
# the CUDA reference so attribution is explicit.
if env LD_LIBRARY_PATH="${TVM_LD_LIBRARY_PATH}" "${TVM_PYTHON}" -c 'import ctypes,tvm; d=ctypes.CDLL("libcuda.so.1"); assert d.cuInit(0)==0; assert tvm.cuda(0).exist' >/dev/null 2>&1; then
  TVM_ARGS=(--output "${OUTPUT}/tvm_layout.csv" --dump-dir "${OUTPUT}/compiler_dumps/tvm")
  if [[ "${MODE}" == quick ]]; then TVM_ARGS+=(--rows 1024 --cols 1024 --number 5 --repeat 3); fi
  run_logged tvm_layout env LD_LIBRARY_PATH="${TVM_LD_LIBRARY_PATH}" \
    "${TVM_PYTHON}" "${HERE}/tvm_layout_bench.py" "${TVM_ARGS[@]}"
  TVM_RQ_ARGS=(--output "${OUTPUT}/tvm_rq_observations.csv" --dump-dir "${OUTPUT}/compiler_dumps/tvm_rq")
  if [[ "${MODE}" == quick ]]; then
    TVM_RQ_ARGS+=(--shape 512x512 --shape 1024x1024 --number 5 --repeat 3)
  else
    TVM_RQ_ARGS+=(--number "${TVM_NUMBER:-20}" --repeat "${TVM_REPEAT:-7}")
  fi
  run_logged tvm_rq_observations env LD_LIBRARY_PATH="${TVM_LD_LIBRARY_PATH}" \
    "${TVM_PYTHON}" "${HERE}/tvm_rq_observation_bench.py" "${TVM_RQ_ARGS[@]}"
  if [[ "${RUN_ALL_640:-0}" == 1 ]]; then
    run_logged tvm_all_640_rq_observations env LD_LIBRARY_PATH="${TVM_LD_LIBRARY_PATH}" \
      "${TVM_PYTHON}" "${HERE}/tvm_rq_observation_bench.py" \
      --manifest "${OUTPUT}/rq_llm_all_640_cases.json" \
      --output "${OUTPUT}/tvm_640_rq_observations.csv" \
      --dump-dir "${OUTPUT}/compiler_dumps/tvm_640_rq" \
      --number "${TVM_640_NUMBER:-5}" --repeat "${TVM_640_REPEAT:-3}"
  else
    record tvm_all_640_rq_observations skipped "set RUN_ALL_640=1"
  fi
  if [[ -s "${OUTPUT}/tvm_rq_observations.csv" ]]; then
    run_logged tvm_rq_analysis "${PYTHON_BIN}" "${HERE}/analyze_tvm_rq_observations.py" \
      --input "${OUTPUT}/tvm_rq_observations.csv" --output-dir "${OUTPUT}"
  else
    record tvm_rq_analysis unavailable "TVM RQ CSV missing"
  fi
else
  record tvm_layout unavailable "TVM CUDA runtime unavailable"
  record tvm_rq_observations unavailable "TVM CUDA runtime unavailable"
  record tvm_rq_analysis unavailable "TVM CUDA runtime unavailable"
fi

DEFAULT_CUTLASS=/home/liangyilei/cursor_home/Ladder/3rdparty/cutlass
CUTLASS_ROOT="${CUTLASS_DIR:-${DEFAULT_CUTLASS}}"
if [[ -f "${CUTLASS_ROOT}/include/cutlass/cutlass.h" ]]; then
  CUTLASS_ARGS=(--cutlass "${CUTLASS_ROOT}" --arch "${ARCH}" \
    --output "${OUTPUT}/cutlass_softmax_boundary.csv" --binary "${OUTPUT}/softmax_layout_boundary")
  if [[ "${MODE}" == quick ]]; then CUTLASS_ARGS+=(--rows 1024 --cols 1024 --tiles 1x16,16x16 --warmup 3 --iterations 10); fi
  run_logged cutlass_softmax "${PYTHON_BIN}" \
    "${BASELINE}/../../Ladder_LL/benchmarks/run_softmax_layout_boundary.py" "${CUTLASS_ARGS[@]}"
  if [[ "${RUN_ALL_640:-0}" == 1 ]]; then
    run_logged cutlass_all_640_attention "${PYTHON_BIN}" \
      "${HERE}/cutlass_640_attention_layout_bench.py" \
      --manifest "${OUTPUT}/rq_llm_all_640_cases.json" \
      --driver "${BASELINE}/../../Ladder_LL/benchmarks/run_softmax_layout_boundary.py" \
      --cutlass "${CUTLASS_ROOT}" --arch "${ARCH}" \
      --binary "${OUTPUT}/softmax_layout_boundary_640" \
      --output "${OUTPUT}/cutlass_640_softmax_boundary.csv" \
      --warmup "${CUTLASS_640_WARMUP:-8}" --iterations "${CUTLASS_640_ITERATIONS:-30}"
  else
    record cutlass_all_640_attention skipped "set RUN_ALL_640=1"
  fi
else
  record cutlass_softmax unavailable "CUTLASS headers not found at ${CUTLASS_ROOT}"
fi

if [[ -n "${SERVING_MODEL_PATH:-}" ]]; then
  SERVING=(--model "${SERVING_MODEL_PATH}" --log-dir "${OUTPUT}/native_server_logs" \
    --startup-timeout "${SERVING_STARTUP_TIMEOUT:-600}" \
    --request-timeout "${SERVING_REQUEST_TIMEOUT:-600}" \
    --memory-fraction "${SERVING_MEMORY_FRACTION:-0.60}")
  if [[ -n "${SERVING_MODEL_NAME:-}" ]]; then SERVING+=(--served-model-name "${SERVING_MODEL_NAME}"); fi
  IFS=',' read -r -a SERVING_CASE_LIST <<< "${SERVING_CASES:-128:1:8:1,2048:1:5:1,8192:16:5:1,2048:32:16:4}"
  for serving_case in "${SERVING_CASE_LIST[@]}"; do
    [[ -n "${serving_case}" ]] && SERVING+=(--case "${serving_case}")
  done
  SGLANG_SERVING=("${SERVING[@]}")
  VLLM_SERVING=("${SERVING[@]}")
  if [[ -n "${SGLANG_VARIANTS:-}" ]]; then
    IFS=',' read -r -a SGLANG_VARIANT_LIST <<< "${SGLANG_VARIANTS}"
    for variant in "${SGLANG_VARIANT_LIST[@]}"; do
      [[ -n "${variant}" ]] && SGLANG_SERVING+=(--variant "${variant}")
    done
  fi
  if [[ -n "${VLLM_VARIANTS:-}" ]]; then
    IFS=',' read -r -a VLLM_VARIANT_LIST <<< "${VLLM_VARIANTS}"
    for variant in "${VLLM_VARIANT_LIST[@]}"; do
      [[ -n "${variant}" ]] && VLLM_SERVING+=(--variant "${variant}")
    done
  fi
  if "${SGLANG_PYTHON}" -c 'import importlib.util; assert importlib.util.find_spec("sglang")' >/dev/null 2>&1; then
    run_logged sglang_native_serving "${SGLANG_PYTHON}" "${HERE}/native_serving_layout_bench.py" \
      --framework sglang --port "${SGLANG_PORT:-31000}" \
      --output "${OUTPUT}/sglang_native_serving.jsonl" "${SGLANG_SERVING[@]}"
  else
    record sglang_native_serving unavailable "sglang not installed"
  fi
  if "${VLLM_PYTHON}" -c 'import importlib.util; assert importlib.util.find_spec("vllm")' >/dev/null 2>&1; then
    run_logged vllm_native_serving "${VLLM_PYTHON}" "${HERE}/native_serving_layout_bench.py" \
      --framework vllm --port "${VLLM_PORT:-32000}" \
      --output "${OUTPUT}/vllm_native_serving.jsonl" "${VLLM_SERVING[@]}"
  else
    record vllm_native_serving unavailable "vllm not installed"
  fi
else
  record sglang_native_serving skipped "set SERVING_MODEL_PATH to a local model"
  record vllm_native_serving skipped "set SERVING_MODEL_PATH to a local model"
fi

if [[ "${RUN_HEXCUTE_ARTIFACT:-0}" == 1 ]] && [[ -d "${HEXCUTE_BENCH_DIR:-}" ]]; then
  HEXCUTE_RUN_SCRIPT="${HEXCUTE_RUN_SCRIPT:-run_smoke.sh}"
  run_logged hexcute_artifact bash -c 'cd "$1/scripts" && export root="$1" && bash "./$2" --host' \
    _ "${HEXCUTE_BENCH_DIR}" "${HEXCUTE_RUN_SCRIPT}"
else
  record hexcute_artifact source_only "set RUN_HEXCUTE_ARTIFACT=1 and HEXCUTE_BENCH_DIR"
fi

run_logged analyze_rqs "${PYTHON_BIN}" "${HERE}/analyze_ref_talks_rqs.py" \
  --cuda-csv "${OUTPUT}/rq_cuda_reference.csv" \
  --source-audit "${OUTPUT}/ref_talks_source_audit.json" \
  --case-summary "${OUTPUT}/rq_case_summary.json" \
  --boundary-jsonl "${OUTPUT}/llm_boundary_layout_sweep.jsonl" --output-dir "${OUTPUT}"

echo "[ref-talks-rq] complete: ${OUTPUT}"
echo "[ref-talks-rq] main report: ${OUTPUT}/RQ_VALIDATION_REPORT.md"
echo "[ref-talks-rq] machine-readable: ${OUTPUT}/rq_validation.json"
