#!/usr/bin/env bash
# Growth suite -> verified baseline -> conditional smoke -> conditional full.
# This is NOT a certification of full-domain/compiler/source-PR requirements.
set -euo pipefail
HERE=${RQ2_SUITE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
if [[ ${RQ2_SUITE_FROZEN:-0} != 1 ]];then
  FROZEN=$(mktemp /tmp/layout_rq2_controlled_suite_XXXXXXXX.sh)
  cp -- "${BASH_SOURCE[0]}" "$FROZEN"
  exec env RQ2_SUITE_FROZEN=1 RQ2_SUITE_ROOT="$HERE" bash "$FROZEN" "$@"
fi
OUT="";GROWTH_FULL="";GROWTH_SMOKE=""
while (($#));do
  case "$1" in
    --output) OUT=$2;shift 2;;
    --reuse-growth-full) GROWTH_FULL=$2;shift 2;;
    --reuse-growth-smoke) GROWTH_SMOKE=$2;shift 2;;
    *) printf 'Unknown argument: %s\n' "$1" >&2;exit 2;;
  esac
done
BASE_PYTHON=${PYTHON_BIN:-/home/liangyilei/conda/envs/cuda-opt/bin/python}
if [[ -z $OUT ]];then OUT=$HERE/results/rq2_controlled_suite_$(date +%Y%m%d_%H%M%S)_$$;fi
[[ ! -e $OUT ]] || { printf 'Refusing to overwrite: %s\n' "$OUT" >&2;exit 2; }
mkdir -p "$OUT/source_snapshot" "$OUT/logs"
for FILE in rq2_domain_anchor_bench.py analyze_rq2_domain_anchor.py audit_rq2_holdout_statistics.py \
  check_rq2_graph_growth_progress.py check_rq2_conditional_smoke.py await_rq2_growth_completion.py \
  run_rq2_domain_anchor_gpu1.sh;do cp -- "$HERE/$FILE" "$OUT/source_snapshot/$FILE";done
sha256sum "$OUT"/source_snapshot/* > "$OUT/source_sha256.txt"
export RQ2_DOMAIN_SOURCE_ROOT=$OUT/source_snapshot
export RQ2_RUNNER_ROOT=$HERE
if [[ -z $GROWTH_SMOKE ]];then
  GROWTH_SMOKE=$OUT/growth_smoke
  bash "$HERE/run_rq2_graph_growth_gpu1.sh" --smoke --smoke-extended --output "$GROWTH_SMOKE"
fi
if [[ -z $GROWTH_FULL ]];then
  GROWTH_FULL=$OUT/growth_full
  bash "$HERE/run_rq2_graph_growth_gpu1.sh" --full --approved-smoke "$GROWTH_SMOKE" --output "$GROWTH_FULL"
fi
"$BASE_PYTHON" "$OUT/source_snapshot/await_rq2_growth_completion.py" --run "$GROWTH_FULL" \
  --require-full > "$OUT/logs/growth_completion_gate.log" 2>&1
printf '[rq2-controlled-suite] growth gate passed; starting conditional smoke\n'
bash "$OUT/source_snapshot/run_rq2_domain_anchor_gpu1.sh" --smoke --wait-for-gpu-lock \
  --baseline-run "$GROWTH_SMOKE" --output "$OUT/conditional_smoke"
printf '[rq2-controlled-suite] conditional smoke passed; starting conditional full\n'
bash "$OUT/source_snapshot/run_rq2_domain_anchor_gpu1.sh" --full --wait-for-gpu-lock \
  --baseline-run "$GROWTH_FULL" --approved-smoke "$OUT/conditional_smoke" --output "$OUT/conditional_full"
printf '[rq2-controlled-suite] implemented controlled suite complete=%s; full RQ2 requirements still separate\n' "$OUT"
