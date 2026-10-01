#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=full
OUTPUT=""
MANIFEST=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) MODE=smoke; shift ;;
    --full) MODE=full; shift ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX:-1}"
VLLM_PYTHON="${VLLM_PYTHON:-${HERE}/framework_envs/layout-vllm/bin/python}"
SGLANG_PYTHON="${SGLANG_PYTHON:-${HERE}/framework_envs/layout-sglang/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -z "${OUTPUT}" ]]; then
  OUTPUT="${HERE}/results/vllm_sglang_subgraphs_${MODE}_$(date +%Y%m%d_%H%M%S)"
fi
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 1; }
mkdir -p "${OUTPUT}/raw/parts" "${OUTPUT}/logs" "${OUTPUT}/cache"

if [[ -z "${MANIFEST}" ]]; then
  CANDIDATE="${HERE}/results/persuasive_real_world_full_20260921_005344/cases/persuasive_real_world_128.json"
  [[ "${MODE}" == smoke ]] && CANDIDATE="${HERE}/results/persuasive_real_world_full_20260921_005344/cases/persuasive_smoke_8.json"
  MANIFEST="${CANDIDATE}"
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "manifest missing: ${MANIFEST}" >&2; exit 2
fi
cp "${MANIFEST}" "${OUTPUT}/executed_manifest.json"

WARMUP="${NATIVE_SUBGRAPH_WARMUP:-3}"
ITERATIONS="${NATIVE_SUBGRAPH_ITERATIONS:-10}"
[[ "${MODE}" == smoke ]] && WARMUP=1 && ITERATIONS=2

run_framework() {
  local framework="$1" python_bin="$2" output="$3" log="$4"
  echo "[native-subgraph] ${framework}"
  : >"${output}"
  local structure phase label part code
  for structure in gqa sliding_attention sparse_attention mla swiglu moe mamba2 linear_attention; do
    if [[ "${structure}" == mamba2 ]]; then
      phase_list=(prefill decode)
    else
      phase_list=(all)
    fi
    for phase in "${phase_list[@]}"; do
      label="${structure}"; phase_args=()
      if [[ "${phase}" != all ]]; then
        label="${structure}_${phase}"; phase_args=(--phase "${phase}")
      fi
      part="${OUTPUT%/raw/*}/raw/parts/${framework}_${label}.jsonl"
      set +e
      timeout --signal=TERM --kill-after=60s \
        "${NATIVE_SUBGRAPH_STRUCTURE_TIMEOUT:-${NATIVE_SUBGRAPH_TIMEOUT:-1800}}" \
        env -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES="${GPU_PHYSICAL_INDEX}" \
          XDG_CACHE_HOME="${OUTPUT%/raw/*}/cache/${framework}" \
          "${python_bin}" "${HERE}/native_framework_subgraph_probe.py" \
          --framework "${framework}" --manifest "${OUTPUT%/raw/*}/executed_manifest.json" \
          --structure "${structure}" "${phase_args[@]}" --output "${part}" \
          --warmup "${WARMUP}" --iterations "${ITERATIONS}" \
          >"${log%.log}_${label}.log" 2>&1
      code=$?
      set -e
      [[ -s "${part}" ]] && cat "${part}" >>"${output}"
      printf '%s_%s\t%s\t%s\n' "${framework}" "${label}" "${code}" \
        "${log%.log}_${label}.log" >>"${OUTPUT%/raw/*}/step_status.tsv"
    done
  done
  return 0
}

: >"${OUTPUT}/step_status.tsv"
run_framework vllm "${VLLM_PYTHON}" "${OUTPUT}/raw/vllm_subgraph_native.jsonl" "${OUTPUT}/logs/vllm.log"
run_framework sglang "${SGLANG_PYTHON}" "${OUTPUT}/raw/sglang_subgraph_native.jsonl" "${OUTPUT}/logs/sglang.log"

"${PYTHON_BIN}" "${HERE}/analyze_native_subgraph_completion.py" \
  --manifest "${OUTPUT}/executed_manifest.json" --result-dir "${OUTPUT}" \
  >"${OUTPUT}/logs/analyze.log" 2>&1
echo "[native-subgraph] complete=${OUTPUT}"
echo "[native-subgraph] report=${OUTPUT}/VLLM_SGLANG_SUBGRAPH_COMPLETION_REPORT_CN.md"
