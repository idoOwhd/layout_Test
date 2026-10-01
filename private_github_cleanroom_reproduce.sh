#!/usr/bin/env bash
# Export the validation source to a private GitHub repository, or rebuild all
# dependencies and reproduce the registered GPU experiments from a clean clone.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="$(cd "${HERE}/.." && pwd)"
REPO_ROOT="$(cd "${HERE}/../../.." && pwd)"
VERSIONS_FILE="${HERE}/reproduction_versions.env"
ORIGINAL_ARGV=("$@")

ACTION="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "${ACTION}" == "-h" || "${ACTION}" == "--help" ]]; then ACTION=help; fi

GPU_INDEX="${GPU_PHYSICAL_INDEX:-0}"
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
BOOTSTRAP_PYTHON="${FRAMEWORK_BOOTSTRAP_PYTHON:-}"
EXPORT_DIR=""
GITHUB_REPO=""
OUTPUT_ROOT=""
MODEL_PATH=""
CUTLASS_ROOT=""
MAX_JOBS="${MAX_BUILD_JOBS:-16}"
SOURCE_BUILD_SERVING=0

usage() {
  cat <<'EOF'
Usage:
  private_github_cleanroom_reproduce.sh export --export-dir DIR
  private_github_cleanroom_reproduce.sh publish --export-dir DIR --github-repo OWNER/NAME
  private_github_cleanroom_reproduce.sh system-deps
  private_github_cleanroom_reproduce.sh source-check
  private_github_cleanroom_reproduce.sh doctor [options]
  private_github_cleanroom_reproduce.sh install [options]
  private_github_cleanroom_reproduce.sh prepare [options]
  private_github_cleanroom_reproduce.sh smoke [options]
  private_github_cleanroom_reproduce.sh full-diversity [options]
  private_github_cleanroom_reproduce.sh full-640 [options]
  private_github_cleanroom_reproduce.sh all [options]

Options:
  --gpu INDEX          Physical GPU index on this host (default: 0)
  --cuda-home PATH     CUDA toolkit containing bin/nvcc (default: /usr/local/cuda)
  --python PATH        Bootstrap Python; if omitted, uv installs managed Python 3.12
  --jobs N             Maximum parallel build jobs (default: 16)
  --model PATH         Local pinned serving-model directory
  --cutlass PATH       Local pinned CUTLASS checkout
  --output-root DIR    Fresh output directory; existing paths are rejected
  --export-dir DIR     Fresh source-only directory prepared for GitHub
  --github-repo NAME   OWNER/NAME for an explicitly requested private publish
  --source-serving     Build pinned vLLM/SGLang sources instead of pinned wheels

Actions:
  export          Copy only source/manifests/references needed by the suite.
                  It excludes results, models, environments, framework sources,
                  compiler caches, binaries and generated install logs.
  publish         export + git commit + `gh repo create --private --push`.
  system-deps     Install Debian/Ubuntu host build prerequisites with sudo.
                  NVIDIA driver and CUDA toolkit are intentionally not modified.
  source-check    Verify the clone contains every required source/input file.
  doctor          Validate source tree, CUDA, GPU, LLVM and free disk space.
  install         Create clean local environments and clone pinned CUTLASS.
  prepare         Verify/download the pinned 0.5B model after install.
  smoke           Run the 16-case, all-backend discriminative smoke suite.
  full-diversity  Reproduce the 256-case result family used by the scientific report.
  full-640        Run strict v10 plus v13 640-case/all-framework accounting.
  all             doctor + install + prepare + smoke + both full suites.

Fresh-clone example:
  bash staged/baseline_framework/layout_research/private_github_cleanroom_reproduce.sh all \
    --gpu 1 --output-root "$PWD/reproduction_outputs/run_$(date +%Y%m%d_%H%M%S)"

The suite reproduces the experimental procedure and result schemas. GPU timing
values are not expected to be bit-identical across cards, drivers or clocks.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_INDEX="$2"; shift 2 ;;
    --cuda-home) CUDA_ROOT="$2"; shift 2 ;;
    --python) BOOTSTRAP_PYTHON="$2"; shift 2 ;;
    --jobs) MAX_JOBS="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --cutlass) CUTLASS_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --export-dir) EXPORT_DIR="$2"; shift 2 ;;
    --github-repo) GITHUB_REPO="$2"; shift 2 ;;
    --source-serving) SOURCE_BUILD_SERVING=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -f "${VERSIONS_FILE}" ]] || {
  echo "missing version lock: ${VERSIONS_FILE}" >&2
  exit 2
}
# shellcheck disable=SC1090
source "${VERSIONS_FILE}"

ENV_ROOT="${HERE}/.repro_envs"
SOURCE_ROOT="${HERE}/.repro_framework_sources"
CACHE_ROOT="${HERE}/.repro_cache"
INSTALL_LOG_ROOT="${HERE}/.repro_install_logs"
ENV_FILE="${HERE}/.repro_framework_envs.sh"
BOOTSTRAP_ROOT="${HERE}/.repro_bootstrap"
DEFAULT_MODEL="${HERE}/.repro_models/Qwen2.5-0.5B-Instruct-${MODEL_REF:0:7}"
DEFAULT_CUTLASS="${REPO_ROOT}/third_party/cutlass"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL}}"
CUTLASS_ROOT="${CUTLASS_ROOT:-${DEFAULT_CUTLASS}}"

required_source_paths=(
  staged/baseline_framework/layout_research
  staged/baseline_framework/benchmark_real_world.py
  staged/baseline_framework/real_world_workloads.py
  staged/baseline_framework/discover_real_world_shapes.py
  staged/baseline_framework/real_world_shapes
  staged/baseline_framework/vision_shapes
  staged/baseline_framework/tilelang/benchmark_llm_subgraphs.py
  staged/baseline_framework/tilelang/benchmark_real_world.py
  staged/baseline_framework/tilelang/smoke_test.py
  staged/baseline_framework/tilelang/tilelang_kernels.py
  Ladder_LL/benchmarks/run_softmax_layout_boundary.py
  Ladder_LL/benchmarks/softmax_layout_boundary.cu
)

source_check() {
  local item
  for item in "${required_source_paths[@]}"; do
    [[ -e "${REPO_ROOT}/${item}" ]] || {
      echo "required reproduction source is missing: ${REPO_ROOT}/${item}" >&2
      return 1
    }
  done
  [[ -f "${HERE}/rq_llm_layout_bench.cu" ]]
  [[ -f "${HERE}/v10_rq1_edge_bench.cu" ]]
  [[ -f "${HERE}/rq_experiment_registry.json" ]]
  [[ -f "${HERE}/v13_experiment_registry.json" ]]
  echo "[source-check] PASS"
}

write_export_metadata() {
  local destination="$1"
  cat >"${destination}/.gitignore" <<'EOF'
# Generated results and launch logs
reproduction_outputs/
staged/baseline_framework/layout_research/results/
*.launcher.log

# Clean-room environments, downloaded frameworks/models and caches
staged/baseline_framework/layout_research/.repro_*/
staged/baseline_framework/layout_research/.repro_*
staged/baseline_framework/layout_research/framework_envs/
staged/baseline_framework/layout_research/framework_sources/
staged/baseline_framework/layout_research/framework_cache/
staged/baseline_framework/layout_research/framework_install_logs/
staged/baseline_framework/layout_research/models/
staged/baseline_framework/layout_research/framework_envs.generated.sh
third_party/

# Models, compiler products and Python caches
*.safetensors
*.pt
*.pth
*.so
*.o
*.a
*.pyc
__pycache__/
.pytest_cache/
Ladder_LL/benchmarks/softmax_layout_boundary
EOF
  cat >"${destination}/PRIVATE_REPRODUCTION_README.md" <<'EOF'
# Clean-room layout validation reproduction

This repository contains validation source, frozen case manifests and RQ
references. It intentionally contains no result directory, model weights,
framework installation, downloaded framework source, binary or compiler cache.

On a CUDA host, run:

```bash
bash staged/baseline_framework/layout_research/private_github_cleanroom_reproduce.sh all \
  --gpu 0 --output-root "$PWD/reproduction_outputs/run_$(date +%Y%m%d_%H%M%S)"
```

For a safer staged run, use `doctor`, `install`, `prepare`, `smoke`, then
`full-diversity` and/or `full-640`. Run the script with `--help` for details.
EOF
}

assert_clean_export() {
  local destination="$1" found
  found="$(find "${destination}" \
    \( -type d \( -name results -o -name framework_envs -o \
       -name framework_sources -o -name framework_cache -o -name models -o \
       -name __pycache__ \) -o \
       -type f \( -name '*.safetensors' -o -name '*.pyc' -o -name '*.so' -o \
       -name 'framework_envs.generated.sh' \) \) -print -quit)"
  if [[ -n "${found}" ]]; then
    echo "forbidden generated artifact entered export: ${found}" >&2
    return 1
  fi
  [[ -f "${destination}/Ladder_LL/benchmarks/softmax_layout_boundary.cu" ]]
  [[ -f "${destination}/staged/baseline_framework/discover_real_world_shapes.py" ]]
  echo "[export-check] PASS: no results/environments/models/binaries"
}

export_source() {
  source_check
  [[ -n "${EXPORT_DIR}" ]] || { echo "--export-dir is required" >&2; exit 2; }
  [[ ! -e "${EXPORT_DIR}" ]] || {
    echo "refusing to overwrite export directory: ${EXPORT_DIR}" >&2
    exit 1
  }
  command -v rsync >/dev/null || { echo "rsync is required for export" >&2; exit 1; }
  mkdir -p "${EXPORT_DIR}/staged/baseline_framework" \
    "${EXPORT_DIR}/Ladder_LL/benchmarks"
  rsync -a \
    --exclude='results/' \
    --exclude='framework_envs/' \
    --exclude='framework_sources/' \
    --exclude='framework_cache/' \
    --exclude='framework_install_logs/' \
    --exclude='models/' \
    --exclude='.repro_*' \
    --exclude='framework_envs.generated.sh' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' --exclude='*.so' --exclude='*.o' --exclude='*.a' \
    "${HERE}/" "${EXPORT_DIR}/staged/baseline_framework/layout_research/"
  rsync -a "${BASELINE}/benchmark_real_world.py" \
    "${BASELINE}/real_world_workloads.py" \
    "${BASELINE}/discover_real_world_shapes.py" \
    "${EXPORT_DIR}/staged/baseline_framework/"
  rsync -a "${BASELINE}/real_world_shapes/" \
    "${EXPORT_DIR}/staged/baseline_framework/real_world_shapes/"
  rsync -a --exclude='__pycache__/' --exclude='*.pyc' \
    "${BASELINE}/vision_shapes/" \
    "${EXPORT_DIR}/staged/baseline_framework/vision_shapes/"
  mkdir -p "${EXPORT_DIR}/staged/baseline_framework/tilelang"
  rsync -a \
    "${BASELINE}/tilelang/benchmark_llm_subgraphs.py" \
    "${BASELINE}/tilelang/benchmark_real_world.py" \
    "${BASELINE}/tilelang/smoke_test.py" \
    "${BASELINE}/tilelang/tilelang_kernels.py" \
    "${EXPORT_DIR}/staged/baseline_framework/tilelang/"
  rsync -a "${REPO_ROOT}/Ladder_LL/benchmarks/run_softmax_layout_boundary.py" \
    "${REPO_ROOT}/Ladder_LL/benchmarks/softmax_layout_boundary.cu" \
    "${EXPORT_DIR}/Ladder_LL/benchmarks/"
  write_export_metadata "${EXPORT_DIR}"
  assert_clean_export "${EXPORT_DIR}"
  (
    cd "${EXPORT_DIR}"
    find . -type f ! -name SOURCE_SHA256SUMS -print0 | sort -z | \
      xargs -0 sha256sum >SOURCE_SHA256SUMS
    find . -type f | sort >SOURCE_FILE_LIST.txt
  )
  echo "[export] ready=${EXPORT_DIR}"
  echo "[export] next: git -C '${EXPORT_DIR}' init -b main"
}

publish_source() {
  [[ -n "${GITHUB_REPO}" ]] || { echo "--github-repo OWNER/NAME is required" >&2; exit 2; }
  command -v gh >/dev/null || { echo "GitHub CLI (gh) is required" >&2; exit 1; }
  gh auth status >/dev/null
  export_source
  git -C "${EXPORT_DIR}" init -b main
  git -C "${EXPORT_DIR}" add .
  git -C "${EXPORT_DIR}" commit -m "Add clean-room layout validation reproduction"
  gh repo create "${GITHUB_REPO}" --private --source "${EXPORT_DIR}" \
    --remote origin --push
  echo "[publish] private repository=${GITHUB_REPO}"
}

install_system_deps() {
  command -v sudo >/dev/null || { echo "sudo is required" >&2; exit 1; }
  sudo apt-get update
  sudo apt-get install -y \
    build-essential ca-certificates curl git git-lfs rsync \
    python3 python3-venv python3-dev llvm-dev libedit-dev libtinfo-dev zlib1g-dev
  echo "[system-deps] CUDA toolkit/driver were not changed"
}

resolve_bootstrap_python() {
  if [[ -n "${BOOTSTRAP_PYTHON}" ]]; then
    [[ -x "${BOOTSTRAP_PYTHON}" ]] || {
      echo "bootstrap Python is not executable: ${BOOTSTRAP_PYTHON}" >&2
      return 1
    }
  else
    command -v python3 >/dev/null || {
      echo "python3 is required to bootstrap managed Python ${REFERENCE_PYTHON}" >&2
      return 1
    }
    if [[ ! -x "${BOOTSTRAP_ROOT}/bin/uv" ]]; then
      python3 -m venv "${BOOTSTRAP_ROOT}"
      "${BOOTSTRAP_ROOT}/bin/python" -m pip install --upgrade pip uv
    fi
    "${BOOTSTRAP_ROOT}/bin/uv" python install "${REFERENCE_PYTHON}"
    BOOTSTRAP_PYTHON="$("${BOOTSTRAP_ROOT}/bin/uv" python find "${REFERENCE_PYTHON}")"
  fi
  "${BOOTSTRAP_PYTHON}" -c \
    'import sys; assert sys.version_info[:2] == (3,12), sys.version; import ensurepip,venv; print(sys.version)'
}

doctor() {
  source_check
  command -v git >/dev/null || { echo "git is missing" >&2; return 1; }
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi is missing" >&2; return 1; }
  [[ -x "${CUDA_ROOT}/bin/nvcc" ]] || {
    echo "CUDA toolkit missing: ${CUDA_ROOT}/bin/nvcc" >&2
    return 1
  }
  if ! nvidia-smi -i "${GPU_INDEX}" \
      --query-gpu=index,name,uuid,driver_version,memory.total \
      --format=csv,noheader; then
    echo "GPU ${GPU_INDEX} is not currently accessible through the NVIDIA driver" >&2
    return 1
  fi
  "${CUDA_ROOT}/bin/nvcc" --version | tail -n 1
  local llvm=""
  for candidate in /usr/bin/llvm-config-19 /usr/bin/llvm-config-18 \
                   /usr/bin/llvm-config-17 /usr/bin/llvm-config; do
    if [[ -x "${candidate}" ]]; then llvm="${candidate}"; break; fi
  done
  [[ -n "${llvm}" ]] || { echo "llvm-config is missing" >&2; return 1; }
  echo "llvm=$(${llvm} --version) path=${llvm}"
  local available_kib
  available_kib="$(df -Pk "${REPO_ROOT}" | awk 'NR==2 {print $4}')"
  if (( available_kib < 30 * 1024 * 1024 )); then
    echo "at least 30 GiB free disk is required; available_kib=${available_kib}" >&2
    return 1
  fi
  echo "disk_free_gib=$(( available_kib / 1024 / 1024 ))"
  echo "[doctor] PASS"
}

prepare_cutlass() {
  if [[ ! -d "${CUTLASS_ROOT}/.git" ]]; then
    [[ ! -e "${CUTLASS_ROOT}" ]] || {
      echo "CUTLASS target exists but is not a git checkout: ${CUTLASS_ROOT}" >&2
      return 1
    }
    mkdir -p "$(dirname "${CUTLASS_ROOT}")"
    git clone --recursive "${CUTLASS_REPOSITORY}" "${CUTLASS_ROOT}"
  fi
  [[ -z "$(git -C "${CUTLASS_ROOT}" status --porcelain)" ]] || {
    echo "CUTLASS checkout is dirty: ${CUTLASS_ROOT}" >&2
    return 1
  }
  git -C "${CUTLASS_ROOT}" fetch origin "${CUTLASS_REF}"
  git -C "${CUTLASS_ROOT}" checkout --detach "${CUTLASS_REF}"
  git -C "${CUTLASS_ROOT}" submodule update --init --recursive
  [[ "$(git -C "${CUTLASS_ROOT}" rev-parse HEAD)" == "${CUTLASS_REF}" ]]
  [[ -f "${CUTLASS_ROOT}/include/cutlass/cutlass.h" ]]
  echo "[cutlass] commit=${CUTLASS_REF}"
}

install_frameworks() {
  doctor
  resolve_bootstrap_python
  prepare_cutlass
  local vllm_method=wheel sglang_method=wheel
  if [[ "${SOURCE_BUILD_SERVING}" == 1 ]]; then
    vllm_method=source
    sglang_method=source
  fi
  GPU_PHYSICAL_INDEX="${GPU_INDEX}" \
  CUDA_HOME_PATH="${CUDA_ROOT}" \
  CUTLASS_DIR="${CUTLASS_ROOT}" \
  FRAMEWORK_BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON}" \
  FRAMEWORK_ENV_ROOT="${ENV_ROOT}" \
  FRAMEWORK_SOURCE_ROOT="${SOURCE_ROOT}" \
  FRAMEWORK_UV_CACHE_DIR="${CACHE_ROOT}/uv" \
  FRAMEWORK_INSTALL_LOG_DIR="${INSTALL_LOG_ROOT}" \
  FRAMEWORK_ENV_FILE="${ENV_FILE}" \
  MAX_BUILD_JOBS="${MAX_JOBS}" \
  VLLM_PACKAGE="${VLLM_PACKAGE}" SGLANG_PACKAGE="${SGLANG_PACKAGE}" \
  VLLM_REF="${VLLM_REF}" SGLANG_REF="${SGLANG_REF}" TVM_REF="${TVM_REF}" \
  bash "${HERE}/install_native_framework_envs.sh" \
    --vllm "${vllm_method}" --sglang "${sglang_method}" --tvm source \
    --cuda-home "${CUDA_ROOT}" --jobs "${MAX_JOBS}" \
    --env-root "${ENV_ROOT}" --source-root "${SOURCE_ROOT}" \
    --bootstrap-python "${BOOTSTRAP_PYTHON}"
  verify_install
}

load_framework_env() {
  [[ -f "${ENV_FILE}" ]] || {
    echo "clean-room environment is absent; run the install action first" >&2
    return 1
  }
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  export PYTHON_BIN="${VLLM_PYTHON}"
  export TORCH_PYTHON="${VLLM_PYTHON}"
  export FRAMEWORK_ENV_FILE="${ENV_FILE}"
  export CUTLASS_DIR="${CUTLASS_ROOT}"
  export SERVING_MODEL_PATH="${MODEL_PATH}"
  export GPU_PHYSICAL_INDEX="${GPU_INDEX}"
  export CUDA_HOME="${CUDA_ROOT}"
}

verify_install() {
  load_framework_env
  env -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
    "${VLLM_PYTHON}" -c \
    'import torch,vllm; assert torch.cuda.is_available(); print("vllm",vllm.__version__,torch.__version__,torch.cuda.get_device_name(0))'
  env -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
    "${SGLANG_PYTHON}" -c \
    'import torch,sglang; assert torch.cuda.is_available(); print("sglang",sglang.__version__,torch.__version__,torch.cuda.get_device_name(0))'
  env -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
    TVM_LIBRARY_PATH="${TVM_LIBRARY_PATH}" LD_LIBRARY_PATH="${TVM_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}" \
    "${TVM_PYTHON}" -c \
    'import tvm; assert tvm.cuda(0).exist; print("tvm",tvm.__file__,tvm.cuda(0))'
  echo "[frameworks] PASS"
}

prepare_model() {
  load_framework_env
  HF_HUB_DISABLE_XET=1 env -u PYTHONPATH -u PYTHONHOME \
    "${VLLM_PYTHON}" "${HERE}/prepare_pinned_serving_model.py" \
    --output "${MODEL_PATH}"
  echo "[model] verified=${MODEL_PATH}"
}

require_runtime() {
  source_check
  load_framework_env
  verify_install
  [[ -f "${MODEL_PATH}/config.json" ]] || {
    echo "model is absent; run the prepare action first: ${MODEL_PATH}" >&2
    return 1
  }
  [[ -f "${CUTLASS_ROOT}/include/cutlass/cutlass.h" ]] || {
    echo "CUTLASS is absent; run the install action first" >&2
    return 1
  }
  ARCH="$(CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${TORCH_PYTHON}" -c \
    'import torch; a,b=torch.cuda.get_device_capability(0); print(f"sm_{a}{b}")')"
  export ARCH TVM_CUDA_ARCH="${ARCH}"
  echo "[runtime] gpu=${GPU_INDEX} arch=${ARCH}"
}

fresh_output() {
  local label="$1"
  if [[ -z "${OUTPUT_ROOT}" ]]; then
    OUTPUT_ROOT="${REPO_ROOT}/reproduction_outputs/${label}_$(date +%Y%m%d_%H%M%S)"
  fi
  [[ ! -e "${OUTPUT_ROOT}" ]] || {
    echo "refusing to overwrite output: ${OUTPUT_ROOT}" >&2
    return 1
  }
}

write_run_metadata() {
  local destination="$1"
  mkdir -p "${destination}"
  cp "${VERSIONS_FILE}" "${destination}/reproduction_versions.env"
  printf '%q ' "$0" "${ORIGINAL_ARGV[@]}" >"${destination}/invocation.txt"
  printf '\n' >>"${destination}/invocation.txt"
  if [[ -f "${REPO_ROOT}/SOURCE_SHA256SUMS" ]]; then
    cp "${REPO_ROOT}/SOURCE_SHA256SUMS" "${destination}/SOURCE_SHA256SUMS"
  fi
  if [[ -d "${INSTALL_LOG_ROOT}" ]]; then
    cp -a "${INSTALL_LOG_ROOT}" "${destination}/framework_install_manifests"
  fi
  git -C "${CUTLASS_ROOT}" rev-parse HEAD >"${destination}/cutlass.commit"
  nvidia-smi -i "${GPU_INDEX}" -q >"${destination}/nvidia_smi_q.txt"
  "${CUDA_ROOT}/bin/nvcc" --version >"${destination}/nvcc_version.txt"
}

validate_discriminative_status() {
  local directory="$1"
  "${PYTHON_BIN}" - "${directory}/status.jsonl" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
p = Path(sys.argv[1])
rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
c = Counter(str(x.get("status")) for x in rows)
bad = [x for x in rows if x.get("status") in {"failed", "unavailable"}]
print(json.dumps({"status_counts": c, "bad_steps": bad}, default=dict, indent=2))
if bad:
    raise SystemExit("required discriminative-suite step failed or was unavailable")
PY
}

run_smoke_into() {
  local destination="$1"
  bash "${HERE}/run_llm_shape_diversity_v2_portable.sh" \
    --gpu "${GPU_INDEX}" --smoke --env-file "${ENV_FILE}" \
    --output "${destination}" --model "${MODEL_PATH}" --cutlass "${CUTLASS_ROOT}"
  validate_discriminative_status "${destination}"
}

run_diversity_into() {
  local destination="$1"
  PERSUASIVE_WARMUP="${PERSUASIVE_WARMUP:-3}" \
  PERSUASIVE_ITERATIONS="${PERSUASIVE_ITERATIONS:-10}" \
  CUDA_WARMUP="${CUDA_WARMUP:-8}" CUDA_ITERATIONS="${CUDA_ITERATIONS:-30}" \
  NATIVE_SUBGRAPH_TIMEOUT="${NATIVE_SUBGRAPH_TIMEOUT:-14400}" \
  KV_BATCH_TIMEOUT="${KV_BATCH_TIMEOUT:-28800}" \
  REAL_WORLD_CASE_TIMEOUT="${REAL_WORLD_CASE_TIMEOUT:-7200}" \
  SERVING_STARTUP_TIMEOUT="${SERVING_STARTUP_TIMEOUT:-3600}" \
  SERVING_REQUEST_TIMEOUT="${SERVING_REQUEST_TIMEOUT:-3600}" \
  bash "${HERE}/run_llm_shape_diversity_v2_portable.sh" \
    --gpu "${GPU_INDEX}" --full --env-file "${ENV_FILE}" \
    --output "${destination}" --model "${MODEL_PATH}" --cutlass "${CUTLASS_ROOT}"
  validate_discriminative_status "${destination}"
}

run_640_into() {
  local destination="$1"
  GPU_PHYSICAL_INDEX="${GPU_INDEX}" ARCH="${ARCH}" TVM_CUDA_ARCH="${ARCH}" \
  PYTHON_BIN="${PYTHON_BIN}" TORCH_PYTHON="${TORCH_PYTHON}" \
  FRAMEWORK_ENV_FILE="${ENV_FILE}" SERVING_MODEL_PATH="${MODEL_PATH}" \
  CUTLASS_DIR="${CUTLASS_ROOT}" CHECK_SOURCE_URLS="${CHECK_SOURCE_URLS:-0}" \
  V10_PROCESS_REPETITIONS="${V10_PROCESS_REPETITIONS:-5}" \
  V10_WARMUP="${V10_WARMUP:-8}" V10_ITERATIONS="${V10_ITERATIONS:-30}" \
  CUDA_WARMUP="${CUDA_WARMUP:-8}" CUDA_ITERATIONS="${CUDA_ITERATIONS:-30}" \
  TRITON_WARMUP="${TRITON_WARMUP:-8}" TRITON_ITERATIONS="${TRITON_ITERATIONS:-30}" \
  ALL_640_WARMUP="${ALL_640_WARMUP:-3}" ALL_640_ITERATIONS="${ALL_640_ITERATIONS:-10}" \
  REAL_WORLD_CASE_TIMEOUT="${REAL_WORLD_CASE_TIMEOUT:-7200}" \
  SERVING_STARTUP_TIMEOUT="${SERVING_STARTUP_TIMEOUT:-3600}" \
  SERVING_REQUEST_TIMEOUT="${SERVING_REQUEST_TIMEOUT:-3600}" \
  bash "${HERE}/run_v10_v13_640_all_frameworks_gpu1.sh" \
    --output "${destination}" --model "${MODEL_PATH}"
}

run_single() {
  local kind="$1"
  require_runtime
  fresh_output "${kind}"
  case "${kind}" in
    smoke) run_smoke_into "${OUTPUT_ROOT}" ;;
    diversity_256) run_diversity_into "${OUTPUT_ROOT}" ;;
    v10_v13_640) run_640_into "${OUTPUT_ROOT}" ;;
  esac
  write_run_metadata "${OUTPUT_ROOT}/cleanroom_metadata"
  echo "[reproduction] complete=${OUTPUT_ROOT}"
}

run_all() {
  fresh_output cleanroom_all
  local run_root="${OUTPUT_ROOT}"
  doctor
  install_frameworks
  prepare_model
  require_runtime
  mkdir -p "${run_root}"
  write_run_metadata "${run_root}/cleanroom_metadata"
  run_smoke_into "${run_root}/smoke_16"
  run_diversity_into "${run_root}/diversity_256"
  run_640_into "${run_root}/v10_v13_640"
  cat >"${run_root}/CLEANROOM_REPRODUCTION_INDEX.md" <<EOF
# Clean-room reproduction outputs

- Source and environment metadata: \`cleanroom_metadata/\`
- 16-case gate: \`smoke_16/\`
- 256-case high-discrimination report: \`diversity_256/PERSUASIVE_RQ_FRAMEWORK_REPORT_CN.md\`
- v10/v13 640-case audit: \`v10_v13_640/CROSS_VERSION_COMPLETENESS.md\`
- v13 framework/RQ matrix: \`v10_v13_640/v13_640/full_combined/V13_640_FRAMEWORK_RQ_MATRIX.md\`

Timing values are measurements of this recorded GPU/software environment and
are not expected to be bit-identical to a run on different hardware or clocks.
EOF
  echo "[reproduction] all complete=${run_root}"
  echo "[reproduction] diversity report=${run_root}/diversity_256/PERSUASIVE_RQ_FRAMEWORK_REPORT_CN.md"
  echo "[reproduction] 640 audit=${run_root}/v10_v13_640/CROSS_VERSION_COMPLETENESS.md"
}

case "${ACTION}" in
  help) usage ;;
  export) export_source ;;
  publish) publish_source ;;
  system-deps) install_system_deps ;;
  source-check) source_check ;;
  doctor) doctor ;;
  install) install_frameworks ;;
  prepare) prepare_model ;;
  smoke) run_single smoke ;;
  full-diversity) run_single diversity_256 ;;
  full-640) run_single v10_v13_640 ;;
  all) run_all ;;
  *) echo "unknown action: ${ACTION}" >&2; usage; exit 2 ;;
esac
