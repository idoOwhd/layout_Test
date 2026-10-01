#!/usr/bin/env bash
# One entry point for all cross-document observations on physical GPU 1.
set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX:-1}"
MODE=full
OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) MODE=quick; shift ;;
    --full) MODE=full; shift ;;
    --output) OUTPUT="$2"; shift 2 ;;
    -h|--help)
      echo "usage: $0 [--quick|--full] [--output DIR]"
      echo "runs canonical RQ1-10, strict v10 L-RQ1, then cross-framework observation audit"
      exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "${OUTPUT}" ]]; then
  OUTPUT="${HERE}/results/observations_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${OUTPUT}/logs"

if [[ -z "${PYTHON_BIN:-}" ]] && [[ -x /home/liangyilei/conda/envs/cuda-opt/bin/python ]]; then
  PYTHON_BIN=/home/liangyilei/conda/envs/cuda-opt/bin/python
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
export GPU_PHYSICAL_INDEX
export CUDA_VISIBLE_DEVICES="${GPU_PHYSICAL_INDEX}"

echo "[observations] output=${OUTPUT} physical_gpu=${GPU_PHYSICAL_INDEX} mode=${MODE}"
bash "${HERE}/run_ref_talks_rq_gpu1.sh" "--${MODE}" --output "${OUTPUT}/canonical" \
  >"${OUTPUT}/logs/canonical.log" 2>&1
CANONICAL_CODE=$?
bash "${HERE}/run_v10_rq1_gpu1.sh" "--${MODE}" --output "${OUTPUT}/v10" \
  >"${OUTPUT}/logs/v10.log" 2>&1
V10_CODE=$?

EXTRACT_ARGS=(
  --canonical-dir "${OUTPUT}/canonical"
  --v10-dir "${OUTPUT}/v10"
  --output "${OUTPUT}/native_observation_results.jsonl"
)
if [[ -n "${NATIVE_OBSERVATION_RESULTS:-}" ]]; then
  EXTRACT_ARGS+=(--additional "${NATIVE_OBSERVATION_RESULTS}")
fi
"${PYTHON_BIN}" "${HERE}/extract_native_observation_results.py" "${EXTRACT_ARGS[@]}" \
  >"${OUTPUT}/logs/native_observation_extract.log" 2>&1
EXTRACT_CODE=$?

VALIDATE_ARGS=(
  --source-audit "${OUTPUT}/canonical/ref_talks_source_audit.json"
  --general-validation "${OUTPUT}/canonical/rq_validation.json"
  --v10-validation "${OUTPUT}/v10/v10_rq1_validation.json"
  --native-results "${OUTPUT}/native_observation_results.jsonl"
  --output-dir "${OUTPUT}"
)
"${PYTHON_BIN}" "${HERE}/validate_layout_observations.py" "${VALIDATE_ARGS[@]}" \
  >"${OUTPUT}/logs/observation_analysis.log" 2>&1
ANALYSIS_CODE=$?

"${PYTHON_BIN}" "${HERE}/audit_rq_native_coverage.py" \
  --result-root "${OUTPUT}" --output-dir "${OUTPUT}" \
  >"${OUTPUT}/logs/rq_native_coverage.log" 2>&1
COVERAGE_CODE=$?

"${PYTHON_BIN}" -c '
import json,sys
path=sys.argv[1]
data={"canonical_exit":int(sys.argv[2]),"v10_exit":int(sys.argv[3]),
      "native_extract_exit":int(sys.argv[4]),"analysis_exit":int(sys.argv[5]),
      "coverage_audit_exit":int(sys.argv[6]),
      "report":sys.argv[7],"native_results":sys.argv[8],"coverage_report":sys.argv[9]}
open(path,"w",encoding="utf-8").write(json.dumps(data,indent=2)+"\n")
' "${OUTPUT}/run_summary.json" "${CANONICAL_CODE}" "${V10_CODE}" "${EXTRACT_CODE}" \
  "${ANALYSIS_CODE}" "${COVERAGE_CODE}" "${OUTPUT}/OBSERVATION_VALIDATION_REPORT.md" \
  "${OUTPUT}/native_observation_results.jsonl" "${OUTPUT}/RQ_NATIVE_COVERAGE.md"

echo "[observations] complete: ${OUTPUT}"
echo "[observations] report: ${OUTPUT}/OBSERVATION_VALIDATION_REPORT.md"
echo "[observations] raw canonical report: ${OUTPUT}/canonical/RQ_VALIDATION_REPORT.md"
echo "[observations] strict v10 report: ${OUTPUT}/v10/V10_RQ1_VALIDATION_REPORT.md"
echo "[observations] native coverage: ${OUTPUT}/RQ_NATIVE_COVERAGE.md"
if [[ "${ANALYSIS_CODE}" -ne 0 || "${EXTRACT_CODE}" -ne 0 || "${COVERAGE_CODE}" -ne 0 ]]; then
  exit 1
fi
exit 0
