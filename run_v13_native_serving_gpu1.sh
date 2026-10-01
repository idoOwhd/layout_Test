#!/usr/bin/env bash
# Complete the v13 vLLM/SGLang native evidence on physical GPU 1 without
# modifying an earlier result directory. Existing raw evidence is copied into
# a new timestamped result before the two serving adapters are run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${FRAMEWORK_ENV_FILE:-${HERE}/framework_envs.generated.sh}"
GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX:-1}"
PYTHON_BIN="${PYTHON_BIN:-/home/liangyilei/conda/envs/cuda-opt/bin/python}"
MODEL_PATH="${SERVING_MODEL_PATH:-${HERE}/models/Qwen2.5-0.5B-Instruct-7ae5576}"
BASE_RESULT=""
OUTPUT=""
MODE=smoke

usage() {
  cat <<'EOF'
usage: run_v13_native_serving_gpu1.sh --base-result V13_RESULT [options]

Options:
  --smoke | --full  Two short requests or four multi-shape requests.
  --output DIR      New result directory (default is timestamped).
  --model DIR       Pinned local checkpoint; no download is attempted.
  --env-file FILE   Generated framework environment file.

The script never writes into BASE_RESULT. It copies BASE_RESULT/raw, runs only
vLLM/SGLang through their native offline APIs, and creates a fresh combined
v13 report. The earlier HTTP adapter remains available for server-level tests.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-result) BASE_RESULT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --smoke) MODE=smoke; shift ;;
    --full) MODE=full; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "${BASE_RESULT}" && -d "${BASE_RESULT}/raw" ]] || {
  echo "--base-result must contain raw/: ${BASE_RESULT:-<unset>}" >&2; exit 2;
}
[[ -f "${ENV_FILE}" ]] || { echo "missing env file: ${ENV_FILE}" >&2; exit 1; }
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "missing model config: ${MODEL_PATH}" >&2; exit 1; }
compgen -G "${MODEL_PATH}/*.safetensors" >/dev/null || {
  echo "missing safetensors in ${MODEL_PATH}" >&2; exit 1;
}
if [[ -z "${OUTPUT}" ]]; then
  OUTPUT="${HERE}/results/v13_native_${MODE}_$(date +%Y%m%d_%H%M%S)"
fi
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 1; }
mkdir -p "${OUTPUT}/raw" "${OUTPUT}/logs" "${OUTPUT}/raw/native_server_logs"
cp -a "${BASE_RESULT}/raw/." "${OUTPUT}/raw/"

# shellcheck disable=SC1090
source "${ENV_FILE}"
[[ -x "${VLLM_PYTHON:-}" ]] || { echo "invalid VLLM_PYTHON" >&2; exit 1; }
[[ -x "${SGLANG_PYTHON:-}" ]] || { echo "invalid SGLANG_PYTHON" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${GPU_PHYSICAL_INDEX}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

run_clean() { env -u PYTHONPATH -u PYTHONHOME "$@"; }
run_step() {
  local name="$1"; shift
  echo "[v13-native] ${name}"
  set +e
  run_clean "$@" >"${OUTPUT}/logs/${name}.log" 2>&1
  local code=$?
  set -e
  "${PYTHON_BIN}" -c 'import json,sys; open(sys.argv[1],"a",encoding="utf-8").write(json.dumps({"step":sys.argv[2],"status":"success" if sys.argv[3]=="0" else "failed","exit_code":int(sys.argv[3]),"log":sys.argv[4]})+"\n")' \
    "${OUTPUT}/v13_native_status.jsonl" "${name}" "${code}" "${OUTPUT}/logs/${name}.log"
  return "${code}"
}

run_step model_identity "${VLLM_PYTHON}" "${HERE}/prepare_pinned_serving_model.py" \
  --output "${MODEL_PATH}" --no-download
run_step model_fit "${VLLM_PYTHON}" "${HERE}/check_serving_model_fit.py" --model "${MODEL_PATH}"

COMMON=(--model "${MODEL_PATH}" \
  --memory-fraction "${SERVING_MEMORY_FRACTION:-0.60}")
if [[ "${MODE}" == smoke ]]; then
  CASES=(--case 128:1:3:1 --case 1024:4:3:1)
  REPETITIONS=3
  GRAPH_ARGS=()
else
  CASES=(--case 128:1:8:1 --case 2048:1:5:1 \
    --case 8192:16:5:1 --case 2048:32:16:4)
  REPETITIONS=5
  GRAPH_ARGS=(--enable-cuda-graph)
fi

run_offline_framework() {
  local framework="$1" python_bin="$2"; shift 2
  local aggregate="${OUTPUT}/raw/${framework}_native_serving.jsonl"
  : >"${aggregate}"
  local any_success=0 variant_code=0 variant part
  for variant in "$@"; do
    part="${OUTPUT}/raw/native_server_logs/${framework}_${variant}.jsonl"
    run_step "${framework}_${variant}" "${python_bin}" "${HERE}/native_offline_layout_bench.py" \
      --framework "${framework}" --variant "${variant}" --output "${part}" \
      --repetitions "${REPETITIONS}" "${COMMON[@]}" "${CASES[@]}" "${GRAPH_ARGS[@]}" \
      || variant_code=$?
    if [[ -s "${part}" ]]; then
      cat "${part}" >>"${aggregate}"
      if "${PYTHON_BIN}" -c 'import json,sys; raise SystemExit(0 if any(json.loads(x).get("status")=="success" for x in open(sys.argv[1],encoding="utf-8") if x.strip()) else 1)' "${part}"; then
        any_success=1
      fi
    fi
  done
  [[ "${any_success}" == 1 ]]
}

SGLANG_CODE=0
VLLM_CODE=0
run_offline_framework sglang "${SGLANG_PYTHON}" \
  auto_nhd auto_hnd flashinfer_p1 flashinfer_p16 || SGLANG_CODE=$?
run_offline_framework vllm "${VLLM_PYTHON}" \
  LBNHC LBHNC block16 block32 || VLLM_CODE=$?

# Preserve the copied historical status while exposing one canonical view in
# which successfully completed native adapters supersede the earlier skipped
# placeholders.  The original raw/status.jsonl is never modified.
"${PYTHON_BIN}" - "${OUTPUT}" "${SGLANG_CODE}" "${VLLM_CODE}" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1]); codes = {"sglang": int(sys.argv[2]), "vllm": int(sys.argv[3])}
source = root / "raw/status.jsonl"
rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()] if source.exists() else []
rows = [row for row in rows if row.get("step") not in
        {"sglang_native_serving", "vllm_native_serving"}]
for framework, code in codes.items():
    rows.append({"step": f"{framework}_native_serving", "status": "success" if code == 0 else "failed",
                 "detail": str(root / "raw" / f"{framework}_native_serving.jsonl"),
                 "supersedes_copied_status": True})
(root / "canonical_status.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
PY

run_step source_audit "${PYTHON_BIN}" "${HERE}/analyze_v13_source_chain.py" --output-dir "${OUTPUT}"
run_step adjudication "${PYTHON_BIN}" "${HERE}/analyze_v13_rqs.py" \
  --raw-dir "${OUTPUT}/raw" --output-dir "${OUTPUT}" \
  --source-json "${OUTPUT}/v13_source_observations.json"
run_step framework_matrix "${PYTHON_BIN}" "${HERE}/analyze_v13_framework_matrix.py" \
  --raw-dir "${OUTPUT}/raw" --output-dir "${OUTPUT}" \
  --source-json "${OUTPUT}/v13_source_observations.json"
run_step framework_640_rq_matrix "${PYTHON_BIN}" "${HERE}/analyze_v13_640_matrix.py" \
  --raw-dir "${OUTPUT}/raw" --output-dir "${OUTPUT}"

EXPECTED_ROWS=8
[[ "${MODE}" == full ]] && EXPECTED_ROWS=16
COMPLETENESS_CODE=0
"${PYTHON_BIN}" -c '
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); expected=int(sys.argv[2]); summary={}
for framework in ("sglang","vllm"):
    path=root/"raw"/f"{framework}_native_serving.jsonl"
    rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    success=sum(row.get("status")=="success" for row in rows)
    summary[framework]={"rows":len(rows),"successful_rows":success,
                        "expected_successful_rows":expected,"complete":success==expected}
(root/"v13_native_completeness.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
raise SystemExit(0 if all(item["complete"] for item in summary.values()) else 1)
' "${OUTPUT}" "${EXPECTED_ROWS}" || COMPLETENESS_CODE=$?

"${PYTHON_BIN}" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps({"base_result":sys.argv[2],"mode":sys.argv[3],"sglang_exit":int(sys.argv[4]),"vllm_exit":int(sys.argv[5])},indent=2)+"\n",encoding="utf-8")' \
  "${OUTPUT}/v13_native_run.json" "${BASE_RESULT}" "${MODE}" "${SGLANG_CODE}" "${VLLM_CODE}"
echo "[v13-native] complete=${OUTPUT}"
echo "[v13-native] report=${OUTPUT}/V13_RQ_VALIDATION_REPORT.md"
echo "[v13-native] matrix=${OUTPUT}/V13_FRAMEWORK_MATRIX.md"
echo "[v13-native] 640-matrix=${OUTPUT}/V13_640_FRAMEWORK_RQ_MATRIX.md"
echo "[v13-native] completeness=${OUTPUT}/v13_native_completeness.json"
echo "[v13-native] canonical-status=${OUTPUT}/canonical_status.jsonl"
[[ "${SGLANG_CODE}" == 0 && "${VLLM_CODE}" == 0 && "${COMPLETENESS_CODE}" == 0 ]]
