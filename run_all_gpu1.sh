#!/usr/bin/env bash
set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="$(cd "${HERE}/.." && pwd)"
GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX:-1}"
if [[ -z "${PYTHON_BIN:-}" ]] && [[ -x /tmp/tilelang-a10-env/bin/python ]]; then
  PYTHON_BIN="/tmp/tilelang-a10-env/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
ARCH="${ARCH:-sm_86}"
MODE="full"
RUN_640="${RUN_640:-1}"
RUN_VISION="${RUN_VISION:-1}"

usage() {
  echo "usage: $0 [--quick|--full] [--output DIR]"
  echo "environment: PYTHON_BIN, GPU_PHYSICAL_INDEX=1, ARCH=sm_86, RUN_640=0|1, RUN_VISION=0|1"
  echo "network/ONNX: CHECK_SHARED_LINKS=0|1, VISION_ONNX_ROOT, VISION_ONNX_MODEL_MAP"
  echo "native serving: SERVING_MODEL_PATH, SERVING_MODEL_NAME, SERVING_STARTUP_TIMEOUT"
  echo "optional source roots: SGLANG_DIR VLLM_DIR CUTLASS_DIR TRITON_DIR TVM_DIR HEXCUTE_DIR"
}

OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) MODE="quick"; RUN_640=0; shift ;;
    --full) MODE="full"; RUN_640=1; shift ;;
    --output) OUTPUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${OUTPUT}" ]]; then
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
  OUTPUT="${HERE}/results/${RUN_ID}"
fi
mkdir -p "${OUTPUT}/logs" "${OUTPUT}/compiler_dumps"
STATUS="${OUTPUT}/status.jsonl"
: >"${STATUS}"
export CUDA_VISIBLE_DEVICES="${GPU_PHYSICAL_INDEX}"
export TORCHINDUCTOR_CACHE_DIR="${OUTPUT}/torchinductor_cache"
export TORCH_LOGS="${TORCH_LOGS:-output_code}"
export TORCH_COMPILE_DEBUG="${TORCH_COMPILE_DEBUG:-1}"
export TORCH_COMPILE_DEBUG_DIR="${OUTPUT}/compiler_dumps/torch"
export TRITON_KERNEL_DUMP="${TRITON_KERNEL_DUMP:-1}"
export TRITON_DUMP_DIR="${OUTPUT}/compiler_dumps/triton"
export LAYOUT_PROBE_OUTPUT="${OUTPUT}/environment.json"

DEFAULT_CUTLASS="/home/liangyilei/cursor_home/Ladder/3rdparty/cutlass"
if [[ -z "${CUTLASS_DIR:-}" ]] && [[ -d "${DEFAULT_CUTLASS}" ]]; then
  export CUTLASS_DIR="${DEFAULT_CUTLASS}"
fi

record() {
  "${PYTHON_BIN}" - "$STATUS" "$1" "$2" "${3:-}" <<'PY'
import json, sys
with open(sys.argv[1], "a", encoding="utf-8") as f:
    f.write(json.dumps({"step": sys.argv[2], "status": sys.argv[3], "detail": sys.argv[4]}) + "\n")
PY
}

run_logged() {
  local name="$1"; shift
  echo "[layout-study] ${name}"
  if "$@" >"${OUTPUT}/logs/${name}.log" 2>&1; then
    record "$name" success "${OUTPUT}/logs/${name}.log"
  else
    local code=$?
    record "$name" failed "exit=${code}; ${OUTPUT}/logs/${name}.log"
  fi
}

run_logged environment "${PYTHON_BIN}" "${HERE}/runtime_probe.py"
run_logged static_unit_tests "${PYTHON_BIN}" -m unittest discover -s "${HERE}" -p 'test_*.py'
run_logged static_layout_analysis "${PYTHON_BIN}" "${HERE}/static_layout_analysis.py" --output "${OUTPUT}/static_layout_metrics.jsonl"
run_logged graph_layout_optimizer "${PYTHON_BIN}" "${HERE}/graph_layout_optimizer.py" \
  "${HERE}/graph_layout_example.json" --output "${OUTPUT}/graph_layout_optimizer.json"
SOURCE_EVIDENCE_ARGS=(--output "${OUTPUT}/source_evidence.json")
if [[ "${CHECK_SHARED_LINKS:-1}" == 1 ]]; then SOURCE_EVIDENCE_ARGS+=(--remote-fallback); fi
run_logged source_evidence "${PYTHON_BIN}" "${HERE}/source_evidence.py" "${SOURCE_EVIDENCE_ARGS[@]}"
run_logged vision_manifest "${PYTHON_BIN}" "${BASELINE}/vision_shapes/build_vision_manifest.py" --output-dir "${OUTPUT}/vision_catalog"
if [[ "${CHECK_SHARED_LINKS:-1}" == 1 ]]; then
  run_logged shared_link_validation "${PYTHON_BIN}" "${HERE}/validate_shared_links.py" --output "${OUTPUT}/shared_link_validation.json" --strict
  run_logged vision_source_pins "${PYTHON_BIN}" "${BASELINE}/vision_shapes/pin_vision_sources.py" \
    --output "${OUTPUT}/vision_source_revisions.json"
else
  record shared_link_validation skipped "CHECK_SHARED_LINKS=0"
  record vision_source_pins skipped "CHECK_SHARED_LINKS=0"
fi
if [[ -n "${VISION_ONNX_ROOT:-}" ]] && [[ -d "${VISION_ONNX_ROOT}" ]]; then
  ONNX_ARGS=(--onnx-root "${VISION_ONNX_ROOT}" --output "${OUTPUT}/vision_onnx_shapes.jsonl")
  if [[ -n "${VISION_ONNX_MODEL_MAP:-}" ]]; then ONNX_ARGS+=(--model-map "${VISION_ONNX_MODEL_MAP}"); fi
  run_logged vision_onnx_discovery "${PYTHON_BIN}" "${BASELINE}/vision_shapes/discover_vision_onnx.py" "${ONNX_ARGS[@]}"
else
  record vision_onnx_discovery skipped "set VISION_ONNX_ROOT to the directory containing generated ONNX files"
fi

if command -v nvcc >/dev/null 2>&1; then
  run_logged build_shared_bank nvcc -O3 -std=c++17 "-arch=${ARCH}" "${HERE}/shared_bank_bench.cu" -o "${OUTPUT}/shared_bank_bench"
  if [[ -x "${OUTPUT}/shared_bank_bench" ]]; then
    run_logged shared_bank "${OUTPUT}/shared_bank_bench"
    if grep -q '^framework,' "${OUTPUT}/logs/shared_bank.log"; then cp "${OUTPUT}/logs/shared_bank.log" "${OUTPUT}/shared_bank.csv"; fi
    if command -v cuobjdump >/dev/null 2>&1; then
      run_logged shared_bank_sass cuobjdump --dump-sass "${OUTPUT}/shared_bank_bench"
      if [[ -s "${OUTPUT}/logs/shared_bank_sass.log" ]]; then cp "${OUTPUT}/logs/shared_bank_sass.log" "${OUTPUT}/compiler_dumps/shared_bank.sass"; fi
    fi
    if command -v ncu >/dev/null 2>&1 && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
      if command -v timeout >/dev/null 2>&1; then
        run_logged shared_bank_ncu timeout "${NCU_TIMEOUT_SECONDS:-300}" \
          ncu -f -o "${OUTPUT}/shared_bank_ncu" --set basic \
          "${OUTPUT}/shared_bank_bench" --blocks 256 --repeats 1024 --warmup 0 --iterations 1
      else
        run_logged shared_bank_ncu ncu -f -o "${OUTPUT}/shared_bank_ncu" --set basic \
          "${OUTPUT}/shared_bank_bench" --blocks 256 --repeats 1024 --warmup 0 --iterations 1
      fi
    else
      record shared_bank_ncu unavailable "ncu missing or NVIDIA driver unavailable"
    fi
  fi
else
  record build_shared_bank unavailable "nvcc not found"
fi

if "${PYTHON_BIN}" -c 'import torch, triton; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  TRITON_ARGS=(--output "${OUTPUT}/triton_kv_layout.csv")
  TORCH_ARGS=(--output "${OUTPUT}/torch_layout.csv")
  if [[ "${MODE}" == quick ]]; then TORCH_ARGS+=(--quick); fi
  run_logged triton_kv_layout "${PYTHON_BIN}" "${HERE}/triton_kv_layout_bench.py" "${TRITON_ARGS[@]}"
  run_logged torch_layout "${PYTHON_BIN}" "${HERE}/torch_layout_bench.py" "${TORCH_ARGS[@]}"
  if [[ "${RUN_VISION}" == 1 ]]; then
    VISION_ARGS=(--output "${OUTPUT}/vision_layout_reference.jsonl")
    if [[ "${MODE}" == quick ]]; then VISION_ARGS+=(--quick --warmup 2 --iterations 5); fi
    run_logged vision_layout_reference "${PYTHON_BIN}" "${HERE}/vision_layout_bench.py" "${VISION_ARGS[@]}"
  else
    record vision_layout_reference skipped "RUN_VISION=0"
  fi
else
  record triton_kv_layout unavailable "torch/triton/CUDA unavailable"
  record torch_layout unavailable "torch/CUDA unavailable"
  record vision_layout_reference unavailable "torch/CUDA unavailable"
fi

if "${PYTHON_BIN}" -c 'import tvm; assert tvm.cuda(0).exist' >/dev/null 2>&1; then
  TVM_ARGS=(--output "${OUTPUT}/tvm_layout.csv" --dump-dir "${OUTPUT}/compiler_dumps/tvm")
  if [[ "${MODE}" == quick ]]; then TVM_ARGS+=(--rows 1024 --cols 1024 --number 5 --repeat 3); fi
  run_logged tvm_layout "${PYTHON_BIN}" "${HERE}/tvm_layout_bench.py" "${TVM_ARGS[@]}"
else
  record tvm_layout unavailable "TVM CUDA runtime unavailable"
fi

CUTLASS_ROOT="${CUTLASS_DIR:-${DEFAULT_CUTLASS}}"
if [[ -f "${CUTLASS_ROOT}/include/cutlass/cutlass.h" ]]; then
  SOFTMAX_ARGS=(--cutlass "${CUTLASS_ROOT}" --arch "${ARCH}" --output "${OUTPUT}/cutlass_softmax_boundary.csv" --binary "${OUTPUT}/softmax_layout_boundary")
  if [[ "${MODE}" == quick ]]; then SOFTMAX_ARGS+=(--rows 1024 --cols 1024 --tiles 1x16,16x16 --warmup 5 --iterations 20); fi
  run_logged cutlass_softmax "${PYTHON_BIN}" "${BASELINE}/../../Ladder_LL/benchmarks/run_softmax_layout_boundary.py" "${SOFTMAX_ARGS[@]}"
  if [[ -x "${OUTPUT}/softmax_layout_boundary" ]] && command -v cuobjdump >/dev/null 2>&1; then
    run_logged cutlass_softmax_sass cuobjdump --dump-sass "${OUTPUT}/softmax_layout_boundary"
    if [[ -s "${OUTPUT}/logs/cutlass_softmax_sass.log" ]]; then cp "${OUTPUT}/logs/cutlass_softmax_sass.log" "${OUTPUT}/compiler_dumps/cutlass_softmax.sass"; fi
  fi
else
  record cutlass_softmax unavailable "CUTLASS headers missing: ${CUTLASS_ROOT}"
fi

if [[ "${RUN_640}" == 1 ]] && "${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  run_logged llm_640_pytorch_triton "${PYTHON_BIN}" "${BASELINE}/benchmark_real_world.py" \
    --physical-device-index "${GPU_PHYSICAL_INDEX}" --output "${OUTPUT}/llm_640_pytorch_triton.jsonl" \
    --no-deduplicate --no-resume --warmup 3 --iterations 10
  if "${PYTHON_BIN}" -c 'import tilelang' >/dev/null 2>&1; then
    run_logged llm_640_tilelang "${PYTHON_BIN}" "${BASELINE}/tilelang/benchmark_real_world.py" \
      --physical-device-index "${GPU_PHYSICAL_INDEX}" --output "${OUTPUT}/llm_640_tilelang.jsonl" \
      --no-deduplicate --no-resume --warmup 3 --iterations 10
  else
    record llm_640_tilelang unavailable "tilelang not installed"
  fi
else
  record llm_640_pytorch_triton skipped "RUN_640=${RUN_640} or CUDA unavailable"
fi

# Hexcute's public artifact owns its environment and frequency-locking scripts.
# It is opt-in because the upstream full runs take hours and may require sudo.
if [[ "${RUN_HEXCUTE_ARTIFACT:-0}" == 1 ]] && [[ -d "${HEXCUTE_BENCH_DIR:-}" ]]; then
  HEXCUTE_RUN_SCRIPT="${HEXCUTE_RUN_SCRIPT:-run_smoke.sh}"
  run_logged hexcute_artifact bash -c 'cd "$1/scripts" && export root="$1" && bash "./$2" --host' \
    _ "${HEXCUTE_BENCH_DIR}" "${HEXCUTE_RUN_SCRIPT}"
else
  record hexcute_artifact source_only "set RUN_HEXCUTE_ARTIFACT=1 and HEXCUTE_BENCH_DIR to execute upstream artifact"
fi

# Native serving adapters are opt-in because the exact model must be identical
# across policies and already available locally (the runner never downloads one
# implicitly). vLLM varies only KV layout; SGLang backend/page experiments are
# marked confounded and therefore excluded from pure layout-regret claims.
if [[ -n "${SERVING_MODEL_PATH:-}" ]]; then
  SERVING_COMMON=(--model "${SERVING_MODEL_PATH}" --log-dir "${OUTPUT}/native_server_logs" \
    --startup-timeout "${SERVING_STARTUP_TIMEOUT:-600}")
  if [[ -n "${SERVING_MODEL_NAME:-}" ]]; then SERVING_COMMON+=(--served-model-name "${SERVING_MODEL_NAME}"); fi
  if "${PYTHON_BIN}" -c 'import importlib.util; assert importlib.util.find_spec("sglang")' >/dev/null 2>&1; then
    run_logged sglang_native_serving "${PYTHON_BIN}" "${HERE}/native_serving_layout_bench.py" \
      --framework sglang --port "${SGLANG_PORT:-31000}" --output "${OUTPUT}/sglang_native_serving.jsonl" "${SERVING_COMMON[@]}"
  else
    record sglang_native_serving unavailable "sglang package unavailable"
  fi
  if "${PYTHON_BIN}" -c 'import importlib.util; assert importlib.util.find_spec("vllm")' >/dev/null 2>&1; then
    run_logged vllm_native_serving "${PYTHON_BIN}" "${HERE}/native_serving_layout_bench.py" \
      --framework vllm --port "${VLLM_PORT:-32000}" --output "${OUTPUT}/vllm_native_serving.jsonl" "${SERVING_COMMON[@]}"
  else
    record vllm_native_serving unavailable "vllm package unavailable"
  fi
else
  record sglang_native_serving skipped "set SERVING_MODEL_PATH to a local/cached model"
  record vllm_native_serving skipped "set SERVING_MODEL_PATH to a local/cached model"
fi

run_logged compiler_layout_analysis "${PYTHON_BIN}" "${HERE}/analyze_layout_dumps.py" \
  --dump-dir "${OUTPUT}/compiler_dumps" --output "${OUTPUT}/compiler_layout_evidence.jsonl"
run_logged layout_optimality_analysis "${PYTHON_BIN}" "${HERE}/analyze_layout_results.py" \
  --run-dir "${OUTPUT}" --output-prefix "${OUTPUT}/layout_analysis"
run_logged requirements_audit "${PYTHON_BIN}" "${HERE}/audit_requirements.py" --run-dir "${OUTPUT}"
run_logged summarize "${PYTHON_BIN}" "${HERE}/summarize.py" "${OUTPUT}"
echo "[layout-study] complete: ${OUTPUT}"
echo "[layout-study] report: ${OUTPUT}/REPORT.md"
