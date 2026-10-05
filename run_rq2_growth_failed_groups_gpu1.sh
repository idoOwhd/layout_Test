#!/usr/bin/env bash
# Run only repaired RQ2 growth groups, two validation phases, exclusive card1.
set -euo pipefail
HERE=${RQ2_RUNNER_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
if [[ ${RQ2_RETRY_FROZEN:-0} != 1 ]];then
  FROZEN=$(mktemp /tmp/layout_rq2_retry_XXXXXXXX.sh);cp -- "${BASH_SOURCE[0]}" "$FROZEN"
  exec env RQ2_RETRY_FROZEN=1 RQ2_RUNNER_ROOT="$HERE" RQ2_RETRY_SCRIPT="$FROZEN" bash "$FROZEN" "$@"
fi
PRIOR="";OUT=""
while (($#));do
  case "$1" in --prior) PRIOR=$2;shift 2;; --output) OUT=$2;shift 2;; *) printf 'Unknown argument\n' >&2;exit 2;; esac
done
[[ -n $PRIOR && ${GPU_PHYSICAL_INDEX:-1} == 1 ]] || { printf '--prior required; card1 only\n' >&2;exit 2; }
if [[ -z $OUT ]];then OUT=$HERE/results/rq2_failed_group_retry_$(date +%Y%m%d_%H%M%S)_$$;fi
[[ ! -e $OUT ]] || { printf 'Refusing overwrite\n' >&2;exit 2; }
BASE_PYTHON=${PYTHON_BIN:-/home/liangyilei/conda/envs/cuda-opt/bin/python}
"$BASE_PYTHON" "$HERE/rq2_failed_group_plan.py" --run "$PRIOR" --output "$OUT"
if [[ $($BASE_PYTHON -c 'import json,sys;print(len(json.load(open(sys.argv[1]))["case_ids_by_framework"]))' "$OUT/RQ2_FAILED_GROUP_PLAN.json") == 0 ]];then
  printf 'No failed measured groups to retry; nothing launched\n';exit 0
fi
exec 9>"/tmp/layout_rq2_gpu1_${UID}.lock"
flock -n 9 || { printf 'Card1 busy; retry plan saved but no GPU work launched\n' >&2;exit 75; }
source "${FRAMEWORK_ENV_FILE:-$HERE/framework_envs.generated.sh}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RQ2_PROJECT_ROOT=$HERE
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)
if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F, -v id="$CUDA_VISIBLE_DEVICES" '$1==id{found=1}END{exit !found}';then
  printf 'Card1 has an external process; retry plan saved, no GPU launched\n' >&2;exit 75
fi
mkdir -p "$OUT/source_snapshot" "$OUT/logs"
for FILE in rq2_failed_group_plan.py rq2_growth_bench.py rq2_growth_search.py rq2_growth_kernels.py \
            rq2_growth_cutlass.cu rq1_diverse_kernels.py rq1_flashinfer_workspace.py \
            check_rq2_graph_growth_progress.py analyze_rq2_graph_growth.py audit_rq2_holdout_statistics.py;do
  cp -- "$HERE/$FILE" "$OUT/source_snapshot/$FILE"
done
cp -- "$RQ2_RETRY_SCRIPT" "$OUT/source_snapshot/run_rq2_growth_failed_groups_gpu1.sh"
sha256sum "$OUT"/source_snapshot/* > "$OUT/source_sha256.txt"
mapfile -t FRAMES < <("$BASE_PYTHON" -c 'import json,sys;print("\n".join(json.load(open(sys.argv[1]))["case_ids_by_framework"]))' "$OUT/RQ2_FAILED_GROUP_PLAN.json")
for FW in "${FRAMES[@]}";do
  case "$FW" in vllm) PY=$VLLM_PYTHON;;sglang) PY=$SGLANG_PYTHON;;*) PY=$BASE_PYTHON;;esac
  for PHASE in smoke full;do
    RUN=$OUT/${FW}_${PHASE};mkdir -p "$RUN/artifacts" "$RUN/logs"
    cp -- "$OUT/${FW}_manifest.json" "$RUN/design_manifest.json"
    "$BASE_PYTHON" -c 'import json,sys;json.dump({"phase":sys.argv[2],"declared_subset_only":True,"smoke_runs_all_selected_real_growth_stages":True,"not_all_RQ2_completed":True},open(sys.argv[1],"x"),indent=2)' "$RUN/RETRY_SCOPE.json" "$PHASE"
    if [[ $FW == cutlass ]];then
      "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" -std=c++17 -O3 -shared -Xcompiler=-fPIC \
        -gencode arch=compute_80,code=sm_86 -I"${CUTLASS_DIR:-/home/liangyilei/cursor_home/Ladder/3rdparty/cutlass}/include" \
        "$OUT/source_snapshot/rq2_growth_cutlass.cu" -o "$RUN/artifacts/rq2_cutlass.so" > "$RUN/logs/build.log" 2>&1
    fi
    W=${RQ2_WARMUP:-3};I=${RQ2_ITERATIONS:-10};H=${RQ2_HOLDOUT_REPETITIONS:-5}
    if [[ $PHASE == smoke ]];then W=1;I=2;H=2;fi
    set +e
    "$PY" "$OUT/source_snapshot/rq2_growth_bench.py" --framework "$FW" --mode full --manifest "$RUN/design_manifest.json" \
      --output "$RUN/$FW.jsonl" --artifacts "$RUN/artifacts" --warmup "$W" --iterations "$I" \
      --holdout-repetitions "$H" --candidate-budget "${RQ2_CANDIDATE_BUDGET:-48}" > "$RUN/logs/$FW.log" 2>&1
    STATUS=$?;set -e
    printf '%s\t%s\n' "$FW" "$STATUS" > "$RUN/process_status.tsv"
    "$BASE_PYTHON" "$OUT/source_snapshot/rq2_failed_group_plan.py" --run "$RUN" --check-selected-complete > "$RUN/SELECTED_COMPLETION.json"
    if [[ $PHASE == full ]];then
      "$BASE_PYTHON" "$OUT/source_snapshot/analyze_rq2_graph_growth.py" --output "$RUN"
      "$BASE_PYTHON" "$OUT/source_snapshot/audit_rq2_holdout_statistics.py" --run "$RUN" --output "$RUN/strict_statistics"
    fi
  done
done
printf '[rq2-retry] repaired groups validated in new paths: %s\n' "$OUT"
