#!/usr/bin/env bash
# Install isolated vLLM, SGLang and CUDA-enabled TVM environments.
#
# Defaults:
#   vLLM   prebuilt wheel selected by uv/driver
#   SGLang prebuilt wheel (current official CUDA-13 lane)
#   TVM    source build with CUDA enabled
#
# Source mode uses the immutable commits audited by this repository unless the
# corresponding *_REF environment variable is overridden.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# venv is deliberately the default: framework installation must not depend on
# repo.anaconda.com/conda-forge being reachable.  Conda remains an opt-in
# backend for sites that explicitly want it.
ENV_CREATOR="${FRAMEWORK_ENV_CREATOR:-venv}"
DEFAULT_BOOTSTRAP_PYTHON="$(command -v python3 || true)"
BOOTSTRAP_PYTHON="${FRAMEWORK_BOOTSTRAP_PYTHON:-${DEFAULT_BOOTSTRAP_PYTHON}}"
CONDA_BIN="${CONDA_BIN:-/home/liangyilei/conda/bin/conda}"
ENV_ROOT="${FRAMEWORK_ENV_ROOT:-${HERE}/framework_envs}"
SOURCE_ROOT="${FRAMEWORK_SOURCE_ROOT:-${HERE}/framework_sources}"
CUDA_HOME_PATH="${CUDA_HOME_PATH:-/usr/local/cuda}"
PYTHON_VERSION="${FRAMEWORK_PYTHON_VERSION:-3.12}"
MAX_BUILD_JOBS="${MAX_BUILD_JOBS:-40}"
PIP_RETRIES="${FRAMEWORK_PIP_RETRIES:-10}"
PIP_TIMEOUT="${FRAMEWORK_PIP_TIMEOUT:-120}"
LLVM_CONFIG_OVERRIDE="${LLVM_CONFIG:-}"
UV_CACHE_DIR_VALUE="${FRAMEWORK_UV_CACHE_DIR:-${HERE}/framework_cache/uv}"
UV_LINK_MODE_VALUE="${FRAMEWORK_UV_LINK_MODE:-copy}"
UV_CONCURRENT_DOWNLOADS_VALUE="${FRAMEWORK_UV_CONCURRENT_DOWNLOADS:-4}"
UV_CONCURRENT_BUILDS_VALUE="${FRAMEWORK_UV_CONCURRENT_BUILDS:-1}"
UV_CONCURRENT_INSTALLS_VALUE="${FRAMEWORK_UV_CONCURRENT_INSTALLS:-1}"

export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-${PIP_TIMEOUT}}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-${PIP_TIMEOUT}}"
if [[ -n "${FRAMEWORK_PYPI_INDEX:-}" ]]; then
  export PIP_INDEX_URL="${FRAMEWORK_PYPI_INDEX}"
  export UV_DEFAULT_INDEX="${FRAMEWORK_PYPI_INDEX}"
fi

VLLM_METHOD="${VLLM_INSTALL_METHOD:-wheel}"
SGLANG_METHOD="${SGLANG_INSTALL_METHOD:-wheel}"
TVM_METHOD="${TVM_INSTALL_METHOD:-source}"
VLLM_PACKAGE="${VLLM_PACKAGE:-vllm==0.29.0}"
SGLANG_PACKAGE="${SGLANG_PACKAGE:-sglang==0.5.20}"
INSTALL_GPU_INDEX="${GPU_PHYSICAL_INDEX:-0}"

# These revisions match the pinned source-evidence run. Override explicitly to
# test a newer framework revision, and keep the generated manifest.
VLLM_REF="${VLLM_REF:-9ca6dbba71329a7b08711e2acfa17b31e542553c}"
SGLANG_REF="${SGLANG_REF:-ad28b91faec29cd9e51ecc5b76b06327ed93389e}"
TVM_REF="${TVM_REF:-8312a17f8734ddfd56e5f3977cd5df25b83ec49f}"

VLLM_ENV_OVERRIDE="${VLLM_ENV:-}"
SGLANG_ENV_OVERRIDE="${SGLANG_ENV:-}"
TVM_ENV_OVERRIDE="${TVM_ENV:-}"
MANIFEST_DIR="${FRAMEWORK_INSTALL_LOG_DIR:-${HERE}/framework_install_logs}"
ENV_FILE="${FRAMEWORK_ENV_FILE:-${HERE}/framework_envs.generated.sh}"

usage() {
  cat <<'EOF'
usage: install_native_framework_envs.sh [options]

Options:
  --vllm wheel|source|skip
  --sglang wheel|source|skip
  --tvm source|skip
  --cuda-home PATH
  --jobs N
  --env-root PATH
  --source-root PATH
  --env-creator venv|conda
  --bootstrap-python PATH

Useful environment overrides:
  VLLM_REF, SGLANG_REF, TVM_REF
  VLLM_PACKAGE (default: vllm==0.29.0)
  SGLANG_PACKAGE (default: sglang==0.5.20)
  VLLM_ENV, SGLANG_ENV, TVM_ENV
  VLLM_INSTALL_METHOD, SGLANG_INSTALL_METHOD, TVM_INSTALL_METHOD
  CUTLASS_DIR, LLVM_CONFIG, REQUIRE_GPU_AT_INSTALL=0|1
  FRAMEWORK_ENV_CREATOR=venv|conda (default: venv)
  FRAMEWORK_BOOTSTRAP_PYTHON (default: python3 resolved on this host)
  FRAMEWORK_PYPI_INDEX, FRAMEWORK_PIP_RETRIES, FRAMEWORK_PIP_TIMEOUT
  FRAMEWORK_UV_CACHE_DIR, FRAMEWORK_UV_LINK_MODE (default: copy)
  FRAMEWORK_UV_CONCURRENT_DOWNLOADS/BUILDS/INSTALLS

Examples:
  # Recommended binary/source mix:
  bash install_native_framework_envs.sh

  # Compile all available framework sources:
  bash install_native_framework_envs.sh --vllm source --sglang source --tvm source
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vllm) VLLM_METHOD="$2"; shift 2 ;;
    --sglang) SGLANG_METHOD="$2"; shift 2 ;;
    --tvm) TVM_METHOD="$2"; shift 2 ;;
    --cuda-home) CUDA_HOME_PATH="$2"; shift 2 ;;
    --jobs) MAX_BUILD_JOBS="$2"; shift 2 ;;
    --env-root) ENV_ROOT="$2"; shift 2 ;;
    --source-root) SOURCE_ROOT="$2"; shift 2 ;;
    --env-creator) ENV_CREATOR="$2"; shift 2 ;;
    --bootstrap-python) BOOTSTRAP_PYTHON="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

VLLM_ENV="${VLLM_ENV_OVERRIDE:-${ENV_ROOT}/layout-vllm}"
SGLANG_ENV="${SGLANG_ENV_OVERRIDE:-${ENV_ROOT}/layout-sglang}"
TVM_ENV="${TVM_ENV_OVERRIDE:-${ENV_ROOT}/layout-tvm}"

for value in "${VLLM_METHOD}" "${SGLANG_METHOD}"; do
  [[ "${value}" =~ ^(wheel|source|skip)$ ]] || {
    echo "vLLM/SGLang method must be wheel, source or skip" >&2; exit 2;
  }
done
[[ "${TVM_METHOD}" =~ ^(source|skip)$ ]] || {
  echo "TVM method must be source or skip" >&2; exit 2;
}
[[ "${ENV_CREATOR}" =~ ^(venv|conda)$ ]] || {
  echo "environment creator must be venv or conda" >&2; exit 2;
}

if [[ "${ENV_CREATOR}" == venv ]]; then
  [[ -x "${BOOTSTRAP_PYTHON}" ]] || {
    echo "bootstrap Python not found: ${BOOTSTRAP_PYTHON}" >&2; exit 1;
  }
  env -u PYTHONPATH -u PYTHONHOME \
    "${BOOTSTRAP_PYTHON}" -c 'import ensurepip, venv' || {
    echo "bootstrap Python lacks ensurepip/venv: ${BOOTSTRAP_PYTHON}" >&2; exit 1;
  }
else
  [[ -x "${CONDA_BIN}" ]] || { echo "conda not found: ${CONDA_BIN}" >&2; exit 1; }
fi
[[ -x "${CUDA_HOME_PATH}/bin/nvcc" ]] || {
  echo "CUDA toolkit not found under ${CUDA_HOME_PATH}" >&2; exit 1;
}
command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
mkdir -p "${SOURCE_ROOT}" "${MANIFEST_DIR}" "${UV_CACHE_DIR_VALUE}"

echo "[install] CUDA_HOME=${CUDA_HOME_PATH}"
echo "[install] environment_creator=${ENV_CREATOR} env_root=${ENV_ROOT}"
echo "[install] sanitized_python_path=1 uv_link_mode=${UV_LINK_MODE_VALUE} uv_cache=${UV_CACHE_DIR_VALUE}"
"${CUDA_HOME_PATH}/bin/nvcc" --version | tail -n 1

# The parent Ladder development shell exports PYTHONPATH entries that expose
# Ladder's package metadata inside otherwise isolated venvs.  Always remove
# PYTHONPATH/PYTHONHOME for framework installation and validation commands.
run_clean() {
  env -u PYTHONPATH -u PYTHONHOME "$@"
}

# uv's clone/reflink mode can fail with EAGAIN on the workspace filesystem,
# especially while creating cuda-tile's isolated build environment.  Plain
# copies plus bounded concurrency are slower but deterministic and resumable.
run_uv() {
  env -u PYTHONPATH -u PYTHONHOME \
    UV_CACHE_DIR="${UV_CACHE_DIR_VALUE}" \
    UV_LINK_MODE="${UV_LINK_MODE_VALUE}" \
    UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS_VALUE}" \
    UV_CONCURRENT_BUILDS="${UV_CONCURRENT_BUILDS_VALUE}" \
    UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS_VALUE}" \
    "$@"
}

ensure_env() {
  local prefix="$1"
  if [[ ! -x "${prefix}/bin/python" ]]; then
    if [[ "${ENV_CREATOR}" == venv ]]; then
      mkdir -p "$(dirname "${prefix}")"
      run_clean "${BOOTSTRAP_PYTHON}" -m venv --copies "${prefix}"
    else
      "${CONDA_BIN}" create -y -p "${prefix}" "python=${PYTHON_VERSION}" pip
    fi
  fi
  run_clean "${prefix}/bin/python" -m pip install --retries "${PIP_RETRIES}" \
    --timeout "${PIP_TIMEOUT}" --upgrade \
    pip uv setuptools wheel packaging cmake ninja
}

resolve_llvm_config() {
  if [[ -n "${LLVM_CONFIG_OVERRIDE}" ]]; then
    [[ -x "${LLVM_CONFIG_OVERRIDE}" ]] || {
      echo "LLVM_CONFIG is not executable: ${LLVM_CONFIG_OVERRIDE}" >&2; return 1;
    }
    printf '%s\n' "${LLVM_CONFIG_OVERRIDE}"
    return 0
  fi

  local candidate
  for candidate in /usr/bin/llvm-config-19 /usr/bin/llvm-config-18 \
                   /usr/bin/llvm-config-17 /usr/bin/llvm-config; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  echo "llvm-config not found; set LLVM_CONFIG=/absolute/path/to/llvm-config" >&2
  return 1
}

checkout_source() {
  local repository="$1" destination="$2" revision="$3"
  if [[ ! -d "${destination}/.git" ]]; then
    git clone --recursive "${repository}" "${destination}"
  fi
  if [[ -n "$(git -C "${destination}" status --porcelain)" ]]; then
    echo "source checkout is dirty; refusing to switch revisions: ${destination}" >&2
    exit 1
  fi
  git -C "${destination}" fetch --tags origin
  git -C "${destination}" checkout --detach "${revision}"
  git -C "${destination}" submodule update --init --recursive
  git -C "${destination}" rev-parse HEAD >"${MANIFEST_DIR}/$(basename "${destination}").commit"
}

install_vllm() {
  [[ "${VLLM_METHOD}" != skip ]] || return 0
  ensure_env "${VLLM_ENV}"
  local python_bin="${VLLM_ENV}/bin/python" uv_bin="${VLLM_ENV}/bin/uv"
  if [[ "${VLLM_METHOD}" == wheel ]]; then
    run_uv "${uv_bin}" pip install --python "${python_bin}" \
      "${VLLM_PACKAGE}" --torch-backend=auto
  else
    local source_dir="${SOURCE_ROOT}/vllm"
    checkout_source https://github.com/vllm-project/vllm.git "${source_dir}" "${VLLM_REF}"
    local cutlass_root="${CUTLASS_DIR:-/home/liangyilei/cursor_home/Ladder/3rdparty/cutlass}"
    CUDA_HOME="${CUDA_HOME_PATH}" MAX_JOBS="${MAX_BUILD_JOBS}" \
      VLLM_CUTLASS_SRC_DIR="${cutlass_root}" \
      run_uv "${uv_bin}" pip install --python "${python_bin}" --torch-backend=auto \
      --editable "${source_dir}"
  fi
  CUDA_HOME="${CUDA_HOME_PATH}" run_clean "${python_bin}" -c \
    'import torch,vllm; print("vllm",vllm.__version__,"torch",torch.__version__,"cuda",torch.version.cuda)'
  run_clean "${python_bin}" -m pip freeze >"${MANIFEST_DIR}/vllm.freeze.txt"
}

install_sglang() {
  [[ "${SGLANG_METHOD}" != skip ]] || return 0
  ensure_env "${SGLANG_ENV}"
  local python_bin="${SGLANG_ENV}/bin/python" uv_bin="${SGLANG_ENV}/bin/uv"
  if [[ "${SGLANG_METHOD}" == wheel ]]; then
    CUDA_HOME="${CUDA_HOME_PATH}" run_uv "${uv_bin}" pip install --python "${python_bin}" \
      --prerelease=allow "${SGLANG_PACKAGE}"
  else
    local source_dir="${SOURCE_ROOT}/sglang"
    checkout_source https://github.com/sgl-project/sglang.git "${source_dir}" "${SGLANG_REF}"
    CUDA_HOME="${CUDA_HOME_PATH}" MAX_JOBS="${MAX_BUILD_JOBS}" \
      run_uv "${uv_bin}" pip install --python "${python_bin}" --prerelease=allow \
      --editable "${source_dir}/python"
  fi
  CUDA_HOME="${CUDA_HOME_PATH}" run_clean "${python_bin}" -c \
    'import torch,sglang; print("sglang",getattr(sglang,"__version__","unknown"),"torch",torch.__version__,"cuda",torch.version.cuda)'
  run_clean "${python_bin}" -m sglang.launch_server --help >/dev/null
  run_clean "${python_bin}" -m pip freeze >"${MANIFEST_DIR}/sglang.freeze.txt"
}

install_tvm() {
  [[ "${TVM_METHOD}" != skip ]] || return 0
  ensure_env "${TVM_ENV}"
  run_clean "${TVM_ENV}/bin/python" -m pip install --retries "${PIP_RETRIES}" \
    --timeout "${PIP_TIMEOUT}" --upgrade \
    pip setuptools wheel numpy cython psutil cloudpickle decorator scipy tornado typing_extensions \
    cuda-bindings
  local source_dir="${SOURCE_ROOT}/tvm" build_dir="${SOURCE_ROOT}/tvm-build" llvm_config
  llvm_config="$(resolve_llvm_config)"
  echo "[install] TVM LLVM_CONFIG=${llvm_config} ($("${llvm_config}" --version))"
  checkout_source https://github.com/apache/tvm.git "${source_dir}" "${TVM_REF}"
  mkdir -p "${build_dir}"
  PATH="${TVM_ENV}/bin:${PATH}" "${TVM_ENV}/bin/cmake" \
    -S "${source_dir}" -B "${build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA="${CUDA_HOME_PATH}" \
    -DUSE_LLVM="${llvm_config}" \
    -DHIDE_PRIVATE_SYMBOLS=ON \
    -DUSE_CCACHE=AUTO
  PATH="${TVM_ENV}/bin:${PATH}" "${TVM_ENV}/bin/cmake" \
    --build "${build_dir}" --parallel "${MAX_BUILD_JOBS}"
  TVM_LIBRARY_PATH="${build_dir}/lib" run_clean "${TVM_ENV}/bin/python" -m pip install -e "${source_dir}"
  TVM_LIBRARY_PATH="${build_dir}/lib" run_clean "${TVM_ENV}/bin/python" -c \
    'import tvm; i=tvm.support.libinfo(); assert i.get("USE_CUDA") not in (None,"OFF"); print("tvm",tvm.__file__,"USE_CUDA",i.get("USE_CUDA"),"CUDA_VERSION",i.get("CUDA_VERSION"))'
  run_clean "${TVM_ENV}/bin/python" -m pip freeze >"${MANIFEST_DIR}/tvm.freeze.txt"
}

install_vllm
install_sglang
install_tvm

# This generated file is consumed by the validation runner.  It contains no
# credentials and is safe to inspect/version separately from model paths.
{
  printf '%s\n' '# Generated by install_native_framework_envs.sh'
  printf 'export VLLM_PYTHON=%q\n' "${VLLM_ENV}/bin/python"
  printf 'export SGLANG_PYTHON=%q\n' "${SGLANG_ENV}/bin/python"
  printf 'export TVM_PYTHON=%q\n' "${TVM_ENV}/bin/python"
  printf 'export TVM_LIBRARY_PATH=%q\n' "${SOURCE_ROOT}/tvm-build/lib"
  printf 'export CUDA_HOME=%q\n' "${CUDA_HOME_PATH}"
  printf 'export PATH=%q:$PATH\n' "${CUDA_HOME_PATH}/bin"
  [[ -d "${SOURCE_ROOT}/vllm" ]] && printf 'export VLLM_DIR=%q\n' "${SOURCE_ROOT}/vllm"
  [[ -d "${SOURCE_ROOT}/sglang" ]] && printf 'export SGLANG_DIR=%q\n' "${SOURCE_ROOT}/sglang"
  [[ -d "${SOURCE_ROOT}/tvm" ]] && printf 'export TVM_DIR=%q\n' "${SOURCE_ROOT}/tvm"
} >"${ENV_FILE}"

if nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader >/dev/null 2>&1; then
  echo "[install] GPU runtime is visible; running CUDA import checks"
  if [[ "${VLLM_METHOD}" != skip ]]; then
    CUDA_VISIBLE_DEVICES="${INSTALL_GPU_INDEX}" run_clean "${VLLM_ENV}/bin/python" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
  fi
  if [[ "${SGLANG_METHOD}" != skip ]]; then
    CUDA_VISIBLE_DEVICES="${INSTALL_GPU_INDEX}" run_clean "${SGLANG_ENV}/bin/python" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
  fi
  if [[ "${TVM_METHOD}" != skip ]]; then
    CUDA_VISIBLE_DEVICES="${INSTALL_GPU_INDEX}" TVM_LIBRARY_PATH="${SOURCE_ROOT}/tvm-build/lib" \
      run_clean "${TVM_ENV}/bin/python" -c 'import ctypes,tvm; d=ctypes.CDLL("libcuda.so.1"); assert d.cuInit(0)==0; assert tvm.cuda(0).exist; print(tvm.cuda(0))'
  fi
else
  echo "[install] WARNING: NVIDIA driver is not currently visible; import/build checks passed, GPU runtime checks are deferred."
  if [[ "${REQUIRE_GPU_AT_INSTALL:-0}" == 1 ]]; then exit 1; fi
fi

echo "[install] environments ready"
echo "[install] source ${ENV_FILE}"
echo "[install] package manifests: ${MANIFEST_DIR}"
