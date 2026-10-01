#!/usr/bin/env bash
# Download the pinned small checkpoint (if needed), run only the previously
# skipped SGLang/vLLM native-serving experiments on physical GPU 1, and refresh
# the cross-framework observation report in an existing result directory.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="$(cd "${HERE}/.." && pwd)"
ENV_FILE="${FRAMEWORK_ENV_FILE:-${HERE}/framework_envs.generated.sh}"
GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX:-1}"
BASE_PYTHON="${PYTHON_BIN:-/home/liangyilei/conda/envs/cuda-opt/bin/python}"
MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION="7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_PATH="${SERVING_MODEL_PATH:-${HERE}/models/Qwen2.5-0.5B-Instruct-7ae5576}"
RESULT_ROOT=""
MODE=full
DOWNLOAD=1
MEMORY_FRACTION="${SERVING_MEMORY_FRACTION:-0.60}"

usage() {
  cat <<'EOF'
usage: complete_missing_native_serving_gpu1.sh --result EXISTING_RESULT [options]

Required:
  --result DIR       Existing observation result root containing canonical/ and v10/

Options:
  --model DIR        Existing local Hugging Face checkpoint. If absent, the pinned
                     Qwen/Qwen2.5-0.5B-Instruct revision is downloaded here.
  --no-download      Fail instead of downloading when --model lacks config.json
  --quick | --full   Quick smoke or model-derived full cases (default: full)
  --memory-fraction  Server GPU memory fraction (default: 0.60)
  --env-file FILE    Generated native-framework environment file

Full cases map to the pinned 640-case model shapes:
  prompt=2048, output=1   -> prefill-0215/0216/0217
  prompt=8192, output=16  -> decode-0215/0216/0217 (whole-request proxy)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --result) RESULT_ROOT="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --no-download) DOWNLOAD=0; shift ;;
    --quick) MODE=quick; shift ;;
    --full) MODE=full; shift ;;
    --memory-fraction) MEMORY_FRACTION="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "${RESULT_ROOT}" ]] || { echo "--result is required" >&2; exit 2; }
[[ -d "${RESULT_ROOT}/canonical" ]] || {
  echo "missing existing canonical result directory: ${RESULT_ROOT}/canonical" >&2; exit 1;
}
[[ -f "${ENV_FILE}" ]] || { echo "missing framework env file: ${ENV_FILE}" >&2; exit 1; }
[[ -x "${BASE_PYTHON}" ]] || { echo "base Python is not executable: ${BASE_PYTHON}" >&2; exit 1; }

# shellcheck disable=SC1090
source "${ENV_FILE}"
[[ -x "${VLLM_PYTHON:-}" ]] || { echo "VLLM_PYTHON is not executable" >&2; exit 1; }
[[ -x "${SGLANG_PYTHON:-}" ]] || { echo "SGLANG_PYTHON is not executable" >&2; exit 1; }

run_clean() {
  env -u PYTHONPATH -u PYTHONHOME "$@"
}

MODEL_WEIGHT="$(find "${MODEL_PATH}" -maxdepth 1 -type f -name '*.safetensors' -print -quit 2>/dev/null || true)"
if [[ ! -f "${MODEL_PATH}/config.json" || -z "${MODEL_WEIGHT}" ]]; then
  [[ "${DOWNLOAD}" == 1 ]] || {
    echo "model missing ${MODEL_PATH}/config.json and --no-download was set" >&2; exit 1;
  }
  echo "[native-completion] downloading ${MODEL_ID}@${MODEL_REVISION} -> ${MODEL_PATH}"
  mkdir -p "${MODEL_PATH}"
  run_clean "${VLLM_PYTHON}" -c '
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])
' "${MODEL_ID}" "${MODEL_REVISION}" "${MODEL_PATH}"
fi

# Reject an arbitrary checkpoint accidentally placed at the recommended path.
run_clean "${BASE_PYTHON}" -c '
import json,sys
c=json.load(open(sys.argv[1],encoding="utf-8"))
expected={"hidden_size":896,"num_attention_heads":14,"num_key_value_heads":2}
bad={k:(c.get(k),v) for k,v in expected.items() if c.get(k)!=v}
if bad: raise SystemExit(f"checkpoint is not the pinned Qwen2.5-0.5B shape: {bad}")
print("[native-completion] model config verified", expected)
' "${MODEL_PATH}/config.json"

if ! nvidia-smi --query-gpu=index,name,driver_version,memory.total \
  --format=csv,noheader -i "${GPU_PHYSICAL_INDEX}"; then
  echo "physical GPU ${GPU_PHYSICAL_INDEX} is not visible" >&2
  exit 1
fi

CANONICAL="${RESULT_ROOT}/canonical"
mkdir -p "${CANONICAL}/logs" "${CANONICAL}/native_server_logs"
STATUS="${CANONICAL}/status.jsonl"
touch "${STATUS}"
export CUDA_VISIBLE_DEVICES="${GPU_PHYSICAL_INDEX}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

if [[ "${MODE}" == quick ]]; then
  CASE_ARGS=(--case 512:1:2:1 --case 2048:4:2:1)
else
  # Exact input-token counts corresponding to the checkpoint's entries in the
  # 640-case manifest. Decode remains a whole-request proxy because a serving
  # request necessarily performs prefill before the first decode step.
  CASE_ARGS=(--case 2048:1:5:1 --case 8192:16:5:1)
fi

record() {
  run_clean "${BASE_PYTHON}" -c '
import json,sys
with open(sys.argv[1],"a",encoding="utf-8") as f:
    f.write(json.dumps({"step":sys.argv[2],"status":sys.argv[3],"detail":sys.argv[4]})+"\n")
' "${STATUS}" "$1" "$2" "$3"
}

run_framework() {
  local framework="$1" python_bin="$2" port="$3"
  local output="${CANONICAL}/${framework}_native_serving.jsonl"
  local log="${CANONICAL}/logs/${framework}_native_serving_completion.log"
  echo "[native-completion] ${framework} -> ${output}"
  set +e
  run_clean "${python_bin}" "${HERE}/native_serving_layout_bench.py" \
    --framework "${framework}" --model "${MODEL_PATH}" \
    --output "${output}" --log-dir "${CANONICAL}/native_server_logs" \
    --port "${port}" --startup-timeout "${SERVING_STARTUP_TIMEOUT:-900}" \
    --request-timeout "${SERVING_REQUEST_TIMEOUT:-600}" \
    --memory-fraction "${MEMORY_FRACTION}" "${CASE_ARGS[@]}" >"${log}" 2>&1
  local code=$?
  set -e
  if [[ "${code}" == 0 ]]; then
    record "${framework}_native_serving" success "${output}"
  else
    record "${framework}_native_serving" failed "exit=${code}; ${log}"
  fi
  return "${code}"
}

SGLANG_CODE=0
VLLM_CODE=0
run_framework sglang "${SGLANG_PYTHON}" "${SGLANG_PORT:-31000}" || SGLANG_CODE=$?
run_framework vllm "${VLLM_PYTHON}" "${VLLM_PORT:-32000}" || VLLM_CODE=$?

# Store the exact model-to-640-case mapping and summarize which requested
# variants really reached successful requests. Failed/unsupported layouts stay
# visible rather than being dropped from the denominator.
run_clean "${BASE_PYTHON}" -c '
import json,sys
from collections import defaultdict
from pathlib import Path
root=Path(sys.argv[1])
manifest_path=Path(sys.argv[2])
model_id,revision,model_path=sys.argv[3:6]
obj=json.loads(manifest_path.read_text(encoding="utf-8"))
catalog=obj["cases"] if isinstance(obj,dict) else obj
matched=[r for r in catalog if r.get("model_id")==model_id and r.get("model_revision")==revision]
summary={
  "model_id":model_id,"model_revision":revision,"model_path":model_path,
  "matched_640_cases":[{"case_id":r["case_id"],"phase":r["phase"],"structure":r["structure"],"shape":r["shape"]} for r in matched],
  "native_case_mapping":{
    "serving_t2048_o1_n5_c1":[r["case_id"] for r in matched if r["phase"]=="prefill"],
    "serving_t8192_o16_n5_c1":[r["case_id"] for r in matched if r["phase"]=="decode"],
  },
  "claim_boundary":"Serving rows are end-to-end. They exercise the same checkpoint/shapes but do not isolate every Triton synthetic subgraph; the 8192-token row includes prefill before decode."
}
(root/"native_model_case_mapping.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
lines=["# Native serving completion", "", f"- Model: `{model_id}@{revision}`", f"- Local path: `{model_path}`", f"- Matched 640-case rows: {len(matched)}", ""]
for fw in ("sglang","vllm"):
    path=root/"canonical"/f"{fw}_native_serving.jsonl"
    rows=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []
    good=[r for r in rows if r.get("status")=="success"]
    by_case=defaultdict(set)
    for r in good: by_case[r["case_id"]].add(r["variant"])
    lines += [f"## {fw}", "", f"- Successful rows: {len(good)}/{len(rows)}"]
    for case,variants in sorted(by_case.items()):
        joined=", ".join(sorted(variants))
        lines.append(f"- `{case}`: {joined}")
    lines.append("")
lines += ["## Claim boundary", "", summary["claim_boundary"], ""]
(root/"NATIVE_SERVING_COMPLETION.md").write_text("\n".join(lines),encoding="utf-8")
' "${RESULT_ROOT}" "${BASELINE}/real_world_shapes/real_world_shape_manifest.json" \
  "${MODEL_ID}" "${MODEL_REVISION}" "${MODEL_PATH}"

# Refresh the cross-framework native evidence and final report using the old
# controlled CUDA/v10 artifacts plus the newly completed server measurements.
if [[ -f "${CANONICAL}/ref_talks_source_audit.json" && \
      -f "${CANONICAL}/rq_validation.json" && \
      -f "${RESULT_ROOT}/v10/v10_rq1_validation.json" ]]; then
  run_clean "${BASE_PYTHON}" "${HERE}/extract_native_observation_results.py" \
    --canonical-dir "${CANONICAL}" --v10-dir "${RESULT_ROOT}/v10" \
    --output "${RESULT_ROOT}/native_observation_results.jsonl"
  run_clean "${BASE_PYTHON}" "${HERE}/validate_layout_observations.py" \
    --source-audit "${CANONICAL}/ref_talks_source_audit.json" \
    --general-validation "${CANONICAL}/rq_validation.json" \
    --v10-validation "${RESULT_ROOT}/v10/v10_rq1_validation.json" \
    --native-results "${RESULT_ROOT}/native_observation_results.jsonl" \
    --output-dir "${RESULT_ROOT}"
else
  echo "[native-completion] prior analysis inputs incomplete; native raw files were kept but the aggregate report was not refreshed" >&2
fi

echo "[native-completion] summary=${RESULT_ROOT}/NATIVE_SERVING_COMPLETION.md"
echo "[native-completion] mapping=${RESULT_ROOT}/native_model_case_mapping.json"
if [[ "${SGLANG_CODE}" != 0 || "${VLLM_CODE}" != 0 ]]; then
  echo "native completion failed: sglang=${SGLANG_CODE} vllm=${VLLM_CODE}" >&2
  exit 1
fi
