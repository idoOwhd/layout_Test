#!/usr/bin/env bash
# Create a source-only, relocatable bundle for an L20/other-host rerun.
# Environments, compiler caches, models, and old results are deliberately not
# copied: binary environments and generated code are not portable across hosts.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../.." && pwd)"
OUTPUT="${1:-${REPO_ROOT}/layout_research_l20_bundle_$(date +%Y%m%d_%H%M%S).tar.gz}"

[[ ! -e "${OUTPUT}" ]] || {
  echo "refusing to overwrite ${OUTPUT}" >&2
  exit 1
}

for path in \
  staged/baseline_framework/layout_research \
  staged/baseline_framework/benchmark_real_world.py \
  staged/baseline_framework/real_world_workloads.py \
  staged/baseline_framework/discover_real_world_shapes.py \
  staged/baseline_framework/real_world_shapes \
  Ladder_LL/benchmarks/run_softmax_layout_boundary.py \
  Ladder_LL/benchmarks/softmax_layout_boundary.cu; do
  [[ -e "${REPO_ROOT}/${path}" ]] || {
    echo "required path is missing: ${REPO_ROOT}/${path}" >&2
    exit 2
  }
done

tar -C "${REPO_ROOT}" -czf "${OUTPUT}" \
  --exclude='staged/baseline_framework/layout_research/results' \
  --exclude='staged/baseline_framework/layout_research/framework_envs' \
  --exclude='staged/baseline_framework/layout_research/framework_sources' \
  --exclude='staged/baseline_framework/layout_research/framework_cache' \
  --exclude='staged/baseline_framework/layout_research/framework_install_logs' \
  --exclude='staged/baseline_framework/layout_research/models' \
  --exclude='staged/baseline_framework/layout_research/.repro_*' \
  --exclude='staged/baseline_framework/layout_research/__pycache__' \
  --exclude='*.pyc' \
  staged/baseline_framework/layout_research \
  staged/baseline_framework/benchmark_real_world.py \
  staged/baseline_framework/real_world_workloads.py \
  staged/baseline_framework/discover_real_world_shapes.py \
  staged/baseline_framework/real_world_shapes \
  Ladder_LL/benchmarks/run_softmax_layout_boundary.py \
  Ladder_LL/benchmarks/softmax_layout_boundary.cu

echo "bundle=${OUTPUT}"
echo "not included: framework_envs, framework_sources, framework_cache, models, results, compiler caches"
echo "copy a local serving model separately, or rerun prepare_pinned_serving_model.py on the target"
echo "provide CUTLASS_DIR on the target (clone/copy a pinned CUTLASS checkout)"
