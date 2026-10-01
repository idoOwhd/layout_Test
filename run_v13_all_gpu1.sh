#!/usr/bin/env bash
# One-command v13 smoke/full validation on physical GPU 1.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX:-1}"
MODEL_PATH="${SERVING_MODEL_PATH:-${HERE}/models/Qwen2.5-0.5B-Instruct-7ae5576}"
MODE=all
OUTPUT=""
RUN_640="${RUN_ALL_640:-0}"

usage() {
  cat <<'EOF'
usage: run_v13_all_gpu1.sh [--smoke|--full|--all] [--all-640] [--output DIR] [--model DIR]

Runs the model-free multi-subgraph suite first, then adds vLLM/SGLang native
offline-engine evidence in a separate combined directory. --all (default)
runs smoke followed by full. Every output is new and no prior run is modified.
--all-640 expands the full run to every pinned LLM case and writes the exact
640 × 6-framework × 10-RQ fail-closed matrix. It is intentionally not enabled
for smoke mode.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) MODE=smoke; shift ;;
    --full) MODE=full; shift ;;
    --all) MODE=all; shift ;;
    --all-640) RUN_640=1; shift ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${OUTPUT}" ]]; then
  OUTPUT="${HERE}/results/v13_all_$(date +%Y%m%d_%H%M%S)"
fi
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 1; }
mkdir -p "${OUTPUT}"

run_mode() {
  local mode="$1"
  local run_640_mode=0
  if [[ "${mode}" == full ]]; then run_640_mode="${RUN_640}"; fi
  local subgraphs="${OUTPUT}/${mode}_subgraphs"
  local combined="${OUTPUT}/${mode}_combined"
  echo "[v13-all] ${mode} subgraphs -> ${subgraphs}"
  GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX}" \
    RUN_ALL_640="${run_640_mode}" \
    bash "${HERE}/run_v13_validation_gpu1.sh" "--${mode}" --skip-serving \
    --output "${subgraphs}" --model "${MODEL_PATH}"
  echo "[v13-all] ${mode} native frameworks -> ${combined}"
  GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX}" \
    bash "${HERE}/run_v13_native_serving_gpu1.sh" "--${mode}" \
    --base-result "${subgraphs}" --output "${combined}" --model "${MODEL_PATH}"
  echo "[v13-all] ${mode} report=${combined}/V13_RQ_VALIDATION_REPORT.md"
}

case "${MODE}" in
  smoke) run_mode smoke ;;
  full) run_mode full ;;
  all) run_mode smoke; run_mode full ;;
esac
echo "[v13-all] complete=${OUTPUT}"
