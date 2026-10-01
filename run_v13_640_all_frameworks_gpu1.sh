#!/usr/bin/env bash
# Reproducible full v13 run over the pinned 640-case LLM catalog on card 1.
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
usage: run_v13_640_all_frameworks_gpu1.sh [--output DIR] [--model DIR]

Runs the pinned 640 LLM cases, all v13 RQs, CUDA/PyTorch, Triton, TVM,
CUTLASS, and native vLLM/SGLang adapters.  It always creates a new output
directory. Hexcute remains an explicit architecture-blocked/missing cell on
the A10 because its public artifact targets A100/H100.
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
  OUTPUT="${HERE}/results/v13_640_all_frameworks_$(date +%Y%m%d_%H%M%S)"
fi
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 1; }
[[ -x "${PYTHON_BIN}" ]] || { echo "missing PYTHON_BIN=${PYTHON_BIN}" >&2; exit 1; }
[[ -x "${TORCH_PYTHON}" ]] || { echo "missing TORCH_PYTHON=${TORCH_PYTHON}" >&2; exit 1; }
[[ -f "${ENV_FILE}" ]] || { echo "missing framework env file ${ENV_FILE}" >&2; exit 1; }
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "missing model ${MODEL_PATH}/config.json" >&2; exit 1; }

# shellcheck disable=SC1090
source "${ENV_FILE}"
export GPU_PHYSICAL_INDEX PYTHON_BIN TORCH_PYTHON
export FRAMEWORK_ENV_FILE="${ENV_FILE}" SERVING_MODEL_PATH="${MODEL_PATH}"
export RUN_ALL_640=1 CHECK_SOURCE_URLS="${CHECK_SOURCE_URLS:-0}"
export CUTLASS_DIR="${CUTLASS_DIR:-/home/liangyilei/cursor_home/Ladder/3rdparty/cutlass}"
export CUDA_WARMUP="${CUDA_WARMUP:-8}" CUDA_ITERATIONS="${CUDA_ITERATIONS:-30}"
export TRITON_WARMUP="${TRITON_WARMUP:-8}" TRITON_ITERATIONS="${TRITON_ITERATIONS:-30}"
export ALL_640_WARMUP="${ALL_640_WARMUP:-3}" ALL_640_ITERATIONS="${ALL_640_ITERATIONS:-10}"
export KV_BATCH_WARMUP="${KV_BATCH_WARMUP:-${ALL_640_WARMUP}}"
export KV_BATCH_ITERATIONS="${KV_BATCH_ITERATIONS:-${ALL_640_ITERATIONS}}"
export KV_BATCH_TIMEOUT="${KV_BATCH_TIMEOUT:-28800}"
export KV_BATCH_MAX_BYTES="${KV_BATCH_MAX_BYTES:-2147483648}"

echo "[v13-640] preflight physical GPU ${GPU_PHYSICAL_INDEX}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
  --format=csv,noheader -i "${GPU_PHYSICAL_INDEX}"

bash "${HERE}/run_v13_all_gpu1.sh" --full --all-640 \
  --output "${OUTPUT}" --model "${MODEL_PATH}"

# Add exact 640-case accounting and framework-native operator slices for
# vLLM/SGLang.  This is kept separate from the fixed-Qwen serving experiment;
# its report explicitly marks every row as an operator slice, never E2E.
GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX}" \
PYTHON_BIN="${PYTHON_BIN}" VLLM_PYTHON="${VLLM_PYTHON}" \
SGLANG_PYTHON="${SGLANG_PYTHON}" \
NATIVE_SUBGRAPH_WARMUP="${NATIVE_SUBGRAPH_WARMUP:-${ALL_640_WARMUP}}" \
NATIVE_SUBGRAPH_ITERATIONS="${NATIVE_SUBGRAPH_ITERATIONS:-${ALL_640_ITERATIONS}}" \
NATIVE_SUBGRAPH_TIMEOUT="${NATIVE_SUBGRAPH_TIMEOUT:-14400}" \
bash "${HERE}/run_vllm_sglang_subgraph_completion_gpu1.sh" --full \
  --manifest "${HERE}/../real_world_shapes/real_world_shape_manifest.json" \
  --output "${OUTPUT}/vllm_sglang_native_subgraphs_640"

# Preserve each catalog attention shape, then derive B=1/2/4/8 with two
# physical slot schedules.  This closes the previous batch=1-only KV evidence.
GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX}" \
PYTHON_BIN="${PYTHON_BIN}" VLLM_PYTHON="${VLLM_PYTHON}" \
SGLANG_PYTHON="${SGLANG_PYTHON}" \
KV_BATCH_WARMUP="${KV_BATCH_WARMUP:-${ALL_640_WARMUP}}" \
KV_BATCH_ITERATIONS="${KV_BATCH_ITERATIONS:-${ALL_640_ITERATIONS}}" \
KV_BATCH_TIMEOUT="${KV_BATCH_TIMEOUT:-28800}" \
bash "${HERE}/run_kv_request_batch_gpu1.sh" --full \
  --manifest "${HERE}/../real_world_shapes/real_world_shape_manifest.json" \
  --output "${OUTPUT}/kv_request_batch_640"

COMBINED="${OUTPUT}/full_combined"
SUBGRAPHS="${OUTPUT}/full_subgraphs"
"${PYTHON_BIN}" - "${OUTPUT}" "${SUBGRAPHS}" "${COMBINED}" "${ROOT}" \
  "${MODEL_PATH}" "${ENV_FILE}" <<'PY'
import json, os, shlex, sys
from pathlib import Path

out, subgraphs, combined, root, model, env_file = map(Path, sys.argv[1:])
summary = json.loads((combined / "v13_640_matrix_summary.json").read_text())
native = json.loads((combined / "v13_native_completeness.json").read_text())
kv_batch = json.loads((out / "kv_request_batch_640" / "KV_REQUEST_BATCH_SUMMARY.json").read_text())
status_rows = [json.loads(line) for line in
               (subgraphs / "raw/status.jsonl").read_text().splitlines() if line.strip()]
latest = {row["step"]: row for row in status_rows}
required = [
    "build_all_640_cases", "llm_all_640_pytorch_triton",
    "llm_all_640_boundary_layout_sweep", "cuda_rq_all_attention",
    "triton_all_640_attention", "tvm_all_640_rq_observations",
    "cutlass_all_640_attention",
]
steps = {name: latest.get(name, {"status": "missing"}) for name in required}
kv_expected = kv_batch["expected_rows_per_framework_for_batches_1_2_4_8_and_two_policies"] * 2
kv_batch_complete = (kv_batch["rows"] == kv_expected and
                     kv_batch["successful_correct_rows"] == kv_expected and
                     kv_batch["status_counts"] == {"success": kv_expected})
infrastructure_complete = (
    summary["matrix_cell_count"] == 640 * 6 * 10
    and all(row.get("status") == "success" for row in steps.values())
    and all(row.get("complete") for row in native.values())
    and kv_batch_complete
)
metadata = {
    "schema_version": 1,
    "output": str(out), "subgraph_result": str(subgraphs),
    "combined_result": str(combined), "model": str(model),
    "framework_env_file": str(env_file), "physical_gpu": int(os.environ["GPU_PHYSICAL_INDEX"]),
    "required_steps": steps, "native_completeness": native,
    "kv_request_batch": {"expected_rows": kv_expected,
                          "complete": kv_batch_complete, **kv_batch},
    "matrix_cells": summary["matrix_cell_count"],
    "applicable_runtime_cells": summary["applicable_runtime_cell_count"],
    "paired_native_runtime_cells": summary["paired_native_runtime_cell_count"],
    "paired_native_projected_runtime_cells": summary.get(
        "paired_native_projected_runtime_cell_count", 0),
    "infrastructure_complete": infrastructure_complete,
    "scientific_all_cells_supported": summary["all_applicable_cells_natively_validated"],
    "guardrail": "Infrastructure completion is not the same as every framework owning or validating every RQ.",
}
(out / "V13_640_RUN_COMPLETENESS.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
replay_prefix = str(out) + "_replay"
command = f'''#!/usr/bin/env bash
set -euo pipefail
cd {shlex.quote(str(root))}
OUT_PREFIX={shlex.quote(replay_prefix)}
OUT="${{OUT_PREFIX}}_$(date +%Y%m%d_%H%M%S)"
GPU_PHYSICAL_INDEX={os.environ['GPU_PHYSICAL_INDEX']} \\
PYTHON_BIN={shlex.quote(os.environ['PYTHON_BIN'])} \\
TORCH_PYTHON={shlex.quote(os.environ['TORCH_PYTHON'])} \\
FRAMEWORK_ENV_FILE={shlex.quote(str(env_file))} \\
SERVING_MODEL_PATH={shlex.quote(str(model))} \\
CUTLASS_DIR={shlex.quote(os.environ['CUTLASS_DIR'])} \\
CUDA_WARMUP={os.environ['CUDA_WARMUP']} CUDA_ITERATIONS={os.environ['CUDA_ITERATIONS']} \\
TRITON_WARMUP={os.environ['TRITON_WARMUP']} TRITON_ITERATIONS={os.environ['TRITON_ITERATIONS']} \\
ALL_640_WARMUP={os.environ['ALL_640_WARMUP']} ALL_640_ITERATIONS={os.environ['ALL_640_ITERATIONS']} \\
KV_BATCH_WARMUP={os.environ['KV_BATCH_WARMUP']} KV_BATCH_ITERATIONS={os.environ['KV_BATCH_ITERATIONS']} \\
KV_BATCH_TIMEOUT={os.environ['KV_BATCH_TIMEOUT']} KV_BATCH_MAX_BYTES={os.environ['KV_BATCH_MAX_BYTES']} \\
bash {shlex.quote(str(Path(os.environ.get('FRAMEWORK_ENV_FILE', '')).parent / 'run_v13_640_all_frameworks_gpu1.sh'))} \\
  --output "$OUT" --model {shlex.quote(str(model))}
'''
(out / "REPRODUCE_THIS_RUN.sh").write_text(command, encoding="utf-8")
(out / "REPRODUCE_THIS_RUN.sh").chmod(0o755)
print(json.dumps(metadata, ensure_ascii=False, indent=2))
if not infrastructure_complete:
    raise SystemExit("full infrastructure run is incomplete; inspect V13_640_RUN_COMPLETENESS.json")
PY

echo "[v13-640] complete=${OUTPUT}"
echo "[v13-640] RQ report=${COMBINED}/V13_RQ_VALIDATION_REPORT.md"
echo "[v13-640] matrix=${COMBINED}/V13_640_FRAMEWORK_RQ_MATRIX.md"
echo "[v13-640] completeness=${OUTPUT}/V13_640_RUN_COMPLETENESS.json"
echo "[v13-640] reproduce=${OUTPUT}/REPRODUCE_THIS_RUN.sh"
echo "[v13-640] vLLM/SGLang subgraphs=${OUTPUT}/vllm_sglang_native_subgraphs_640/VLLM_SGLANG_SUBGRAPH_COMPLETION_REPORT_CN.md"
echo "[v13-640] KV request batch=${OUTPUT}/kv_request_batch_640/KV_REQUEST_BATCH_REPORT_CN.md"
