#!/usr/bin/env bash
# Run the v10/v13 discriminative supplement on physical GPU 1.
# Every invocation requires a fresh output directory and never mutates an old run.
set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="$(cd "${HERE}/.." && pwd)"
ENV_FILE="${FRAMEWORK_ENV_FILE:-${HERE}/framework_envs.generated.sh}"
PYTHON_BIN="${PYTHON_BIN:-/home/liangyilei/conda/envs/cuda-opt/bin/python}"
TORCH_PYTHON="${TORCH_PYTHON:-${PYTHON_BIN}}"
GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX:-1}"
CUTLASS_ROOT="${CUTLASS_DIR:-/home/liangyilei/cursor_home/Ladder/3rdparty/cutlass}"
MODEL_PATH="${SERVING_MODEL_PATH:-${HERE}/models/Qwen2.5-0.5B-Instruct-7ae5576}"
ARCH="${ARCH:-sm_86}"
MODE=full
OUTPUT=""
CASE_PROFILE=persuasive128
RQ_ONLY=""

usage() {
  cat <<'EOF'
usage: run_persuasive_real_world_rqs_gpu1.sh [--smoke|--full] [--diversity-v2|--adequacy-v3] [--rq L-RQ1..L-RQ10] [--output DIR] [--model DIR]

Default full mode runs the original 128 real-world-derived contracts.
--diversity-v2 runs 256 contracts balanced across structure, phase and
batch=1/2/4/8. Smoke mode runs each structure×phase path. vLLM/SGLang
include both fixed-model whole-engine counterfactuals and case-for-case native
operator slices.  The latter are explicitly not labelled full-graph evidence.
--adequacy-v3 runs 1024 real-parent-derived contracts and the denser causal
interventions for RQ2/RQ3/RQ6/RQ8/RQ10. L-RQ5 stays single-GPU blocked.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) MODE=smoke; shift ;;
    --full) MODE=full; shift ;;
    --diversity-v2) CASE_PROFILE=diversity_v2; shift ;;
    --adequacy-v3) CASE_PROFILE=adequacy_v3; shift ;;
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
if [[ -n "${RQ_ONLY}" && "${CASE_PROFILE}" != adequacy_v3 ]]; then
  echo "--rq requires --adequacy-v3 so target_rqs is available" >&2
  exit 2
fi

if [[ -z "${OUTPUT}" ]]; then
  OUTPUT="${HERE}/results/persuasive_real_world_${MODE}_$(date +%Y%m%d_%H%M%S)"
fi
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 1; }
mkdir -p "${OUTPUT}/raw" "${OUTPUT}/logs" "${OUTPUT}/compiler_dumps" \
  "${OUTPUT}/raw/native_server_logs"
STATUS="${OUTPUT}/status.jsonl"
: >"${STATUS}"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi
TVM_PYTHON="${TVM_PYTHON:-${PYTHON_BIN}}"
VLLM_PYTHON="${VLLM_PYTHON:-${PYTHON_BIN}}"
SGLANG_PYTHON="${SGLANG_PYTHON:-${PYTHON_BIN}}"
export CUDA_VISIBLE_DEVICES="${GPU_PHYSICAL_INDEX}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
TVM_LD_LIBRARY_PATH="${TVM_LIBRARY_PATH:-}${TVM_LIBRARY_PATH:+:}${LD_LIBRARY_PATH:-}"
export TVM_CUDA_ARCH="${TVM_CUDA_ARCH:-${ARCH}}"
export TORCHINDUCTOR_CACHE_DIR="${OUTPUT}/compiler_dumps/torchinductor_cache"
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR="${OUTPUT}/compiler_dumps/triton"

record() {
  "${PYTHON_BIN}" -c 'import json,sys; open(sys.argv[1],"a",encoding="utf-8").write(json.dumps({"step":sys.argv[2],"status":sys.argv[3],"exit_code":int(sys.argv[4]),"detail":sys.argv[5]})+"\n")' \
    "${STATUS}" "$1" "$2" "$3" "$4"
}

run_step() {
  local name="$1"; shift
  echo "[persuasive] ${name}"
  set +e
  env -u PYTHONPATH -u PYTHONHOME "$@" >"${OUTPUT}/logs/${name}.log" 2>&1
  local code=$?
  set -e
  if [[ "${code}" == 0 ]]; then
    record "${name}" success "${code}" "${OUTPUT}/logs/${name}.log"
  else
    record "${name}" failed "${code}" "${OUTPUT}/logs/${name}.log"
  fi
  return 0
}

run_csv() {
  local name="$1" csv="$2"; shift 2
  echo "[persuasive] ${name}"
  set +e
  env -u PYTHONPATH -u PYTHONHOME "$@" >"${csv}" 2>"${OUTPUT}/logs/${name}.log"
  local code=$?
  set -e
  if [[ "${code}" == 0 ]]; then
    record "${name}" success "${code}" "${csv}"
  else
    record "${name}" failed "${code}" "${OUTPUT}/logs/${name}.log"
  fi
}

rq_wants() {
  [[ -z "${RQ_ONLY}" ]] && return 0
  local candidate
  for candidate in "$@"; do
    [[ "${RQ_ONLY}" == "${candidate}" ]] && return 0
  done
  return 1
}

skip_for_rq() {
  record "$1" skipped_by_rq 0 "not applicable to ${RQ_ONLY}"
}

echo "[persuasive] output=${OUTPUT} mode=${MODE} physical_gpu=${GPU_PHYSICAL_INDEX} logical_gpu=0"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
  --format=csv,noheader -i "${GPU_PHYSICAL_INDEX}" >"${OUTPUT}/gpu_identity.txt" 2>&1 || true

run_step persuasive_case_tests "${PYTHON_BIN}" -m unittest discover \
  -s "${HERE}" -p 'test_persuasive_real_world_cases.py' -v
run_step native_adapter_tests "${PYTHON_BIN}" -m unittest discover \
  -s "${HERE}" -p 'test_native_serving_layout_bench.py' -v
run_step native_subgraph_adapter_tests "${PYTHON_BIN}" -m unittest discover \
  -s "${HERE}" -p 'test_native_framework_subgraph_probe.py' -v
if [[ "${CASE_PROFILE}" == adequacy_v3 ]]; then
  run_step rq_adequacy_tests env \
    PYTHONPATH="${HERE}:${BASELINE}" \
    "${PYTHON_BIN}" -m unittest "${HERE}/test_rq_adequacy_suite.py" -v
fi
if [[ "${CASE_PROFILE}" == adequacy_v3 ]]; then
  run_step build_cases "${PYTHON_BIN}" "${HERE}/build_rq_adequacy_suite.py" \
    --output-dir "${OUTPUT}/cases"
  FULL_MANIFEST="${OUTPUT}/cases/rq_adequacy_1024.json"
  FULL_ATTENTION_TSV="${OUTPUT}/cases/rq_adequacy_attention.tsv"
  SMOKE_MANIFEST="${OUTPUT}/cases/rq_adequacy_smoke_32.json"
  SMOKE_ATTENTION_TSV="${OUTPUT}/cases/rq_adequacy_attention_smoke.tsv"
elif [[ "${CASE_PROFILE}" == diversity_v2 ]]; then
  run_step build_cases "${PYTHON_BIN}" "${HERE}/build_llm_shape_diversity_v2.py" \
    --output-dir "${OUTPUT}/cases"
  FULL_MANIFEST="${OUTPUT}/cases/llm_shape_diversity_256.json"
  FULL_ATTENTION_TSV="${OUTPUT}/cases/llm_shape_diversity_attention.tsv"
  SMOKE_MANIFEST="${OUTPUT}/cases/llm_shape_diversity_smoke_16.json"
  SMOKE_ATTENTION_TSV="${OUTPUT}/cases/llm_shape_diversity_attention_smoke.tsv"
else
  run_step build_cases "${PYTHON_BIN}" "${HERE}/build_persuasive_real_world_cases.py" \
    --output-dir "${OUTPUT}/cases"
  FULL_MANIFEST="${OUTPUT}/cases/persuasive_real_world_128.json"
  FULL_ATTENTION_TSV="${OUTPUT}/cases/persuasive_attention.tsv"
  SMOKE_MANIFEST="${OUTPUT}/cases/persuasive_smoke_8.json"
  SMOKE_ATTENTION_TSV="${OUTPUT}/cases/persuasive_attention_smoke.tsv"
fi
if [[ "${CASE_PROFILE}" == diversity_v2 || "${CASE_PROFILE}" == adequacy_v3 ]]; then
  run_step shape_diversity_audit "${PYTHON_BIN}" "${HERE}/audit_llm_shape_diversity.py" \
    --manifest "${FULL_MANIFEST}" --output-dir "${OUTPUT}/cases"
fi

if [[ "${MODE}" == smoke ]]; then
  MANIFEST="${SMOKE_MANIFEST}"
  ATTENTION_TSV="${SMOKE_ATTENTION_TSV}"
  WARMUP=1; ITERATIONS=3; EXPLICIT_WARMUP=2; EXPLICIT_ITERATIONS=5
  TVM_NUMBER=3; TVM_REPEAT=2; CUTLASS_LIMIT=(--balanced-limit 8)
  # Equal prompt counts, paired concurrency: isolate single-request waves from
  # genuinely simultaneous multi-request engine batches.
  NATIVE_CASES=(--case 128:1:3:1 --case 128:1:3:3 \
    --case 1024:4:3:1 --case 1024:4:3:3)
  NATIVE_REPETITIONS=2; GRAPH_ARGS=()
else
  MANIFEST="${FULL_MANIFEST}"
  ATTENTION_TSV="${FULL_ATTENTION_TSV}"
  WARMUP="${PERSUASIVE_WARMUP:-3}"; ITERATIONS="${PERSUASIVE_ITERATIONS:-10}"
  EXPLICIT_WARMUP="${CUDA_WARMUP:-8}"; EXPLICIT_ITERATIONS="${CUDA_ITERATIONS:-30}"
  TVM_NUMBER="${TVM_128_NUMBER:-5}"; TVM_REPEAT="${TVM_128_REPEAT:-3}"; CUTLASS_LIMIT=()
  NATIVE_CASES=(--case 128:1:8:1 --case 128:1:8:8 \
    --case 512:1:5:1 --case 512:1:5:5 \
    --case 4096:16:5:1 --case 4096:16:5:5 \
    --case 8192:16:3:1 --case 8192:16:3:3)
  NATIVE_REPETITIONS="${NATIVE_REPETITIONS:-5}"; GRAPH_ARGS=(--enable-cuda-graph)
fi
if [[ "${CASE_PROFILE}" == adequacy_v3 && "${MODE}" == full ]]; then
  # Paired prompt counts with concurrency 1 versus N, plus page/tile boundary
  # prompts.  Equal prompt counts are essential: otherwise throughput and
  # layout effects are confounded by different amounts of work.
  NATIVE_CASES=(
    --case 31:1:8:1 --case 31:1:8:8
    --case 32:1:8:1 --case 32:1:8:8
    --case 33:1:8:1 --case 33:1:8:8
    --case 127:4:8:1 --case 127:4:8:8
    --case 128:4:8:1 --case 128:4:8:8
    --case 129:4:8:1 --case 129:4:8:8
    --case 511:8:4:1 --case 511:8:4:4
    --case 1024:16:4:1 --case 1024:16:4:4
    --case 4096:16:2:1 --case 4096:16:2:2
    --case 8192:16:2:1 --case 8192:16:2:2)
fi
cp "${MANIFEST}" "${OUTPUT}/cases/executed_manifest.json"
if [[ -n "${RQ_ONLY}" ]]; then
  run_step select_single_rq_cases "${PYTHON_BIN}" "${HERE}/filter_manifest_for_rq.py" \
    --rq "${RQ_ONLY}" --manifest "${MANIFEST}" --attention-tsv "${ATTENTION_TSV}" \
    --output-manifest "${OUTPUT}/cases/executed_${RQ_ONLY}.json" \
    --output-attention-tsv "${OUTPUT}/cases/executed_${RQ_ONLY}_attention.tsv" \
    --summary "${OUTPUT}/cases/SINGLE_RQ_SELECTION.json"
  if [[ ! -s "${OUTPUT}/cases/executed_${RQ_ONLY}.json" || \
        ! -s "${OUTPUT}/cases/SINGLE_RQ_SELECTION.json" ]]; then
    echo "single-RQ case selection failed; refusing to run an unscoped suite" >&2
    exit 2
  fi
  MANIFEST="${OUTPUT}/cases/executed_${RQ_ONLY}.json"
  ATTENTION_TSV="${OUTPUT}/cases/executed_${RQ_ONLY}_attention.tsv"
  cp "${MANIFEST}" "${OUTPUT}/cases/executed_manifest.json"
fi
BOUNDARY_EXTRA_ARGS=()
[[ "${CASE_PROFILE}" == adequacy_v3 ]] && BOUNDARY_EXTRA_ARGS+=(--extended-rq-counterfactuals)

# Fail before producing a directory full of unrelated adapter errors when the
# assigned card/driver is temporarily unavailable.  torch allocation is the
# decisive test; NVML alone can be unavailable in otherwise valid containers.
if ! env -u PYTHONPATH -u PYTHONHOME "${TORCH_PYTHON}" -c \
  'import torch; assert torch.cuda.is_available(); x=torch.empty(1,device="cuda"); torch.cuda.synchronize(); print(torch.cuda.get_device_name())' \
  >"${OUTPUT}/logs/gpu_preflight.log" 2>&1; then
  record gpu_preflight failed 3 "CUDA allocation failed; ${OUTPUT}/logs/gpu_preflight.log"
  "${PYTHON_BIN}" "${HERE}/analyze_persuasive_real_world_rqs.py" --result-dir "${OUTPUT}" \
    >"${OUTPUT}/logs/analyze.log" 2>&1 || true
  echo "[persuasive] GPU preflight failed; no framework result is claimed: ${OUTPUT}" >&2
  exit 3
fi
record gpu_preflight success 0 "${OUTPUT}/logs/gpu_preflight.log"

if [[ "${RQ_ONLY}" == "L-RQ5" ]]; then
  record distributed_rq5 blocked_single_gpu 0 \
    "L-RQ5 requires at least two GPUs/ranks; physical GPU 1 cannot validate it"
  record hexcute blocked_architecture_unsupported 0 \
    "physical GPU 1 is NVIDIA A10 (sm_86); public Hexcute artifact targets A100/H100"
  "${PYTHON_BIN}" "${HERE}/analyze_single_rq_run.py" \
    --result-dir "${OUTPUT}" --rq "${RQ_ONLY}" || true
  echo "[persuasive] ${RQ_ONLY} is correctly blocked on one card: ${OUTPUT}"
  exit 0
fi

run_step whole_graph_pytorch_torchinductor "${PYTHON_BIN}" "${HERE}/run_real_world_isolated.py" \
  --python "${TORCH_PYTHON}" --runner "${BASELINE}/benchmark_real_world.py" \
  --manifest "${MANIFEST}" --output "${OUTPUT}/raw/whole_graph.jsonl" \
  --log-dir "${OUTPUT}/logs/whole_graph_isolated" \
  --physical-device-index "${GPU_PHYSICAL_INDEX}" --backend pytorch --backend triton \
  --warmup "${WARMUP}" --iterations "${ITERATIONS}" \
  --timeout "${REAL_WORLD_CASE_TIMEOUT:-3600}"

if rq_wants L-RQ1 L-RQ3 L-RQ4 L-RQ7 L-RQ9; then
run_step pytorch_boundary_counterfactual "${PYTHON_BIN}" "${HERE}/run_boundary_hybrid.py" \
  --runner "${HERE}/llm_boundary_layout_sweep.py" \
  --isolated-runner "${HERE}/run_boundary_isolated.py" \
  --python "${TORCH_PYTHON}" --manifest "${MANIFEST}" \
  --output "${OUTPUT}/raw/boundary_layout.jsonl" \
  --log-dir "${OUTPUT}/logs/boundary" \
  --physical-device-index "${GPU_PHYSICAL_INDEX}" \
  --warmup "${WARMUP}" --iterations "${ITERATIONS}" \
  --timeout "${REAL_WORLD_CASE_TIMEOUT:-3600}" \
  "${BOUNDARY_EXTRA_ARGS[@]}"
else
  skip_for_rq pytorch_boundary_counterfactual
fi

if rq_wants L-RQ1 L-RQ2 L-RQ3 L-RQ4 L-RQ6 L-RQ8 L-RQ9 L-RQ10 && command -v nvcc >/dev/null 2>&1; then
  run_step build_cuda_reference nvcc -O3 -std=c++17 "-arch=${ARCH}" \
    "${HERE}/rq_llm_layout_bench.cu" -o "${OUTPUT}/raw/rq_llm_layout_bench"
  if [[ -x "${OUTPUT}/raw/rq_llm_layout_bench" ]]; then
    if [[ "${CASE_PROFILE}" == adequacy_v3 ]]; then
      CUDA_REP_ARGS=()
      [[ "${MODE}" == smoke ]] && CUDA_REP_ARGS+=(--quick)
      run_step cuda_reference "${PYTHON_BIN}" "${HERE}/run_cuda_reference_repetitions.py" \
        --binary "${OUTPUT}/raw/rq_llm_layout_bench" --case-file "${ATTENTION_TSV}" \
        --output "${OUTPUT}/raw/cuda_reference.csv" \
        --warmup "${EXPLICIT_WARMUP}" --iterations "${EXPLICIT_ITERATIONS}" \
        --repetitions "${CUDA_PROCESS_REPETITIONS:-5}" "${CUDA_REP_ARGS[@]}"
    else
      run_csv cuda_reference "${OUTPUT}/raw/cuda_reference.csv" \
        "${OUTPUT}/raw/rq_llm_layout_bench" --case-file "${ATTENTION_TSV}" \
        --warmup "${EXPLICIT_WARMUP}" --iterations "${EXPLICIT_ITERATIONS}"
    fi
    if command -v cuobjdump >/dev/null 2>&1; then
      run_step cuda_reference_sass cuobjdump --dump-sass "${OUTPUT}/raw/rq_llm_layout_bench"
      cp "${OUTPUT}/logs/cuda_reference_sass.log" \
        "${OUTPUT}/compiler_dumps/cuda_reference.sass" 2>/dev/null || true
    fi
  fi
elif rq_wants L-RQ1 L-RQ2 L-RQ3 L-RQ4 L-RQ6 L-RQ8 L-RQ9 L-RQ10; then
  record build_cuda_reference unavailable 127 "nvcc not found"
else
  skip_for_rq build_cuda_reference
fi

if rq_wants L-RQ1 L-RQ2 L-RQ4 L-RQ7 L-RQ9 && "${TORCH_PYTHON}" -c 'import triton' >/dev/null 2>&1; then
  run_step explicit_triton "${TORCH_PYTHON}" "${HERE}/triton_kv_layout_bench.py" \
    --attention-tsv "${ATTENTION_TSV}" --output "${OUTPUT}/raw/triton_explicit.csv" \
    --warmup "${EXPLICIT_WARMUP}" --iterations "${EXPLICIT_ITERATIONS}"
elif rq_wants L-RQ1 L-RQ2 L-RQ4 L-RQ7 L-RQ9; then
  record explicit_triton unavailable 127 "triton unavailable in TORCH_PYTHON"
else
  skip_for_rq explicit_triton
fi

if rq_wants L-RQ1 L-RQ2 L-RQ3 L-RQ4 L-RQ7 L-RQ9 && env LD_LIBRARY_PATH="${TVM_LD_LIBRARY_PATH}" "${TVM_PYTHON}" -c 'import ctypes,tvm; d=ctypes.CDLL("libcuda.so.1"); assert d.cuInit(0)==0; assert tvm.cuda(0).exist' >/dev/null 2>&1; then
  run_step tvm_projected_boundaries env LD_LIBRARY_PATH="${TVM_LD_LIBRARY_PATH}" \
    "${TVM_PYTHON}" "${HERE}/tvm_rq_observation_bench.py" \
    --manifest "${MANIFEST}" --output "${OUTPUT}/raw/tvm_projected.csv" \
    --dump-dir "${OUTPUT}/compiler_dumps/tvm" --number "${TVM_NUMBER}" --repeat "${TVM_REPEAT}"
elif rq_wants L-RQ1 L-RQ2 L-RQ3 L-RQ4 L-RQ7 L-RQ9; then
  record tvm_projected_boundaries unavailable 127 "TVM CUDA runtime unavailable"
else
  skip_for_rq tvm_projected_boundaries
fi

if rq_wants L-RQ1 L-RQ3 L-RQ4 L-RQ7 L-RQ9 && [[ -f "${CUTLASS_ROOT}/include/cutlass/cutlass.h" ]]; then
  run_step cutlass_attention_boundary "${PYTHON_BIN}" \
    "${HERE}/cutlass_640_attention_layout_bench.py" \
    --manifest "${MANIFEST}" \
    --driver "${BASELINE}/../../Ladder_LL/benchmarks/run_softmax_layout_boundary.py" \
    --cutlass "${CUTLASS_ROOT}" --arch "${ARCH}" \
    --binary "${OUTPUT}/raw/cutlass_softmax_boundary" \
    --output "${OUTPUT}/raw/cutlass_attention.csv" \
    --warmup "${EXPLICIT_WARMUP}" --iterations "${EXPLICIT_ITERATIONS}" \
    "${CUTLASS_LIMIT[@]}"
elif rq_wants L-RQ1 L-RQ3 L-RQ4 L-RQ7 L-RQ9; then
  record cutlass_attention_boundary unavailable 127 "CUTLASS headers missing: ${CUTLASS_ROOT}"
else
  skip_for_rq cutlass_attention_boundary
fi

run_native_framework() {
  local framework="$1" python_bin="$2" aggregate="$3"; shift 3
  : >"${aggregate}"
  local variant part
  for variant in "$@"; do
    part="${OUTPUT}/raw/native_server_logs/${framework}_${variant}.jsonl"
    run_step "${framework}_${variant}" "${python_bin}" "${HERE}/native_offline_layout_bench.py" \
      --framework "${framework}" --variant "${variant}" --model "${MODEL_PATH}" \
      --output "${part}" --memory-fraction "${SERVING_MEMORY_FRACTION:-0.60}" \
      --repetitions "${NATIVE_REPETITIONS}" "${NATIVE_CASES[@]}" "${GRAPH_ARGS[@]}"
    [[ -s "${part}" ]] && cat "${part}" >>"${aggregate}"
  done
  return 0
}

if rq_wants L-RQ1 L-RQ2 L-RQ3 L-RQ4 L-RQ6 L-RQ8 L-RQ9 L-RQ10 && [[ -f "${MODEL_PATH}/config.json" ]] && "${SGLANG_PYTHON}" -c 'import sglang' >/dev/null 2>&1; then
  run_native_framework sglang "${SGLANG_PYTHON}" "${OUTPUT}/raw/sglang_native.jsonl" \
    auto_nhd auto_hnd flashinfer_p1 flashinfer_p16
elif rq_wants L-RQ1 L-RQ2 L-RQ3 L-RQ4 L-RQ6 L-RQ8 L-RQ9 L-RQ10; then
  record sglang_native unavailable 127 "model or SGLang environment missing"
else
  skip_for_rq sglang_native
fi
if rq_wants L-RQ1 L-RQ2 L-RQ3 L-RQ4 L-RQ6 L-RQ8 L-RQ9 L-RQ10 && [[ -f "${MODEL_PATH}/config.json" ]] && "${VLLM_PYTHON}" -c 'import vllm' >/dev/null 2>&1; then
  run_native_framework vllm "${VLLM_PYTHON}" "${OUTPUT}/raw/vllm_native.jsonl" \
    LBNHC LBHNC block16 block32
elif rq_wants L-RQ1 L-RQ2 L-RQ3 L-RQ4 L-RQ6 L-RQ8 L-RQ9 L-RQ10; then
  record vllm_native unavailable 127 "model or vLLM environment missing"
else
  skip_for_rq vllm_native
fi

# The serving rows above are fixed-model end-to-end evidence.  This additional
# pass consumes every manifest case and invokes layout-sensitive operators
# shipped by vLLM/SGLang.  Its report keeps operator-slice and full-subgraph
# evidence separate and accounts for unsupported cases fail-closed.
if rq_wants L-RQ1 L-RQ2 L-RQ3 L-RQ4 L-RQ6 L-RQ7 L-RQ9; then
run_step vllm_sglang_native_subgraph_completion env \
  GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX}" \
  VLLM_PYTHON="${VLLM_PYTHON}" SGLANG_PYTHON="${SGLANG_PYTHON}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  NATIVE_SUBGRAPH_WARMUP="${NATIVE_SUBGRAPH_WARMUP:-${WARMUP}}" \
  NATIVE_SUBGRAPH_ITERATIONS="${NATIVE_SUBGRAPH_ITERATIONS:-${ITERATIONS}}" \
  NATIVE_SUBGRAPH_TIMEOUT="${NATIVE_SUBGRAPH_TIMEOUT:-7200}" \
  bash "${HERE}/run_vllm_sglang_subgraph_completion_gpu1.sh" \
  "--${MODE}" --manifest "${MANIFEST}" --output "${OUTPUT}/native_subgraphs"
else
  skip_for_rq vllm_sglang_native_subgraph_completion
fi

# Isolate the KV materialization boundary.  The source catalogs are batch=1;
# this runner explicitly labels B>1 as a derived request-batch counterfactual.
KV_BATCH_MANIFEST="${MANIFEST}"
if [[ "${MODE}" == smoke && -z "${RQ_ONLY}" ]]; then
  # The compact 8-case smoke manifest has only prefill attention.  Use the
  # generated 128 catalog with balanced limiting so decode is exercised too.
  KV_BATCH_MANIFEST="${FULL_MANIFEST}"
fi
if rq_wants L-RQ2 L-RQ4 L-RQ6 L-RQ8 L-RQ9 L-RQ10; then
run_step vllm_sglang_kv_request_batch env \
  GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX}" \
  VLLM_PYTHON="${VLLM_PYTHON}" SGLANG_PYTHON="${SGLANG_PYTHON}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  KV_BATCH_WARMUP="${KV_BATCH_WARMUP:-${WARMUP}}" \
  KV_BATCH_ITERATIONS="${KV_BATCH_ITERATIONS:-${ITERATIONS}}" \
  KV_BATCH_TIMEOUT="${KV_BATCH_TIMEOUT:-14400}" \
  bash "${HERE}/run_kv_request_batch_gpu1.sh" \
  "--${MODE}" --manifest "${KV_BATCH_MANIFEST}" --output "${OUTPUT}/kv_request_batch"
else
  skip_for_rq vllm_sglang_kv_request_batch
fi

# Hexcute's public kernels require Ampere datacenter tensor-core paths that the
# A10 artifact does not support. Keep an explicit blocked record; never replace
# it with CUDA-reference evidence.
record hexcute blocked_architecture_unsupported 0 \
  "physical GPU 1 is NVIDIA A10 (sm_86); public Hexcute artifact targets A100/H100"

run_step analyze "${PYTHON_BIN}" "${HERE}/analyze_persuasive_real_world_rqs.py" \
  --result-dir "${OUTPUT}"
if [[ -n "${RQ_ONLY}" ]]; then
  run_step analyze_single_rq "${PYTHON_BIN}" "${HERE}/analyze_single_rq_run.py" \
    --result-dir "${OUTPUT}" --rq "${RQ_ONLY}"
fi
if [[ "${CASE_PROFILE}" == adequacy_v3 && -z "${RQ_ONLY}" ]]; then
  run_step analyze_rq_adequacy "${PYTHON_BIN}" "${HERE}/analyze_rq_adequacy_supplement.py" \
    --result-dir "${OUTPUT}" --manifest "${MANIFEST}"
  run_step audit_attention_kv_multiconsumer "${PYTHON_BIN}" \
    "${HERE}/audit_attention_kv_multiconsumer_results.py" \
    --result-dir "${OUTPUT}" --manifest "${MANIFEST}"
  run_step analyze_rq_strict "${PYTHON_BIN}" \
    "${HERE}/analyze_rq_full_per_rq.py" --result-dir "${OUTPUT}"
  "${PYTHON_BIN}" "${HERE}/audit_rq_adequacy_completion.py" \
    --result-dir "${OUTPUT}" --mode "${MODE}"
fi

echo "[persuasive] complete=${OUTPUT}"
echo "[persuasive] report=${OUTPUT}/PERSUASIVE_RQ_FRAMEWORK_REPORT_CN.md"
echo "[persuasive] figure=${OUTPUT}/figures/rq_framework_evidence.svg"
if [[ "${CASE_PROFILE}" == adequacy_v3 && -z "${RQ_ONLY}" ]]; then
  echo "[persuasive] adequacy=${OUTPUT}/RQ_ADEQUACY_ANALYSIS_CN.md"
  echo "[persuasive] strict-per-rq=${OUTPUT}/RQ_FULL_PER_RQ_STRICT_ANALYSIS_CN.md"
  echo "[persuasive] attention-kv=${OUTPUT}/ATTENTION_KV_MULTICONSUMER_AUDIT_CN.md"
fi
if [[ -n "${RQ_ONLY}" ]]; then
  echo "[persuasive] selected-rq=${RQ_ONLY}"
  echo "[persuasive] selection=${OUTPUT}/cases/SINGLE_RQ_SELECTION.json"
  echo "[persuasive] single-rq-report=${OUTPUT}/SINGLE_RQ_REPRODUCTION_CN.md"
fi
