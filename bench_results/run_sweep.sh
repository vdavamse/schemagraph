#!/usr/bin/env bash
# Full Spider 2.0-Lite schema-linking sweep. Usage: bench_results/run_sweep.sh /path/to/Spider2 [part]
#   part = 1 (main + budgets), 2 (ablations), omitted = both
set -euo pipefail
ROOT="${1:?path to Spider2 clone}"
PART="${2:-all}"
OUT="$(cd "$(dirname "$0")" && pwd)"
cd "$OUT/.."
run() {  # resumable: skip configs whose result file already exists
  local tag="$2"
  if [ -f "$OUT/spider2_lite_${tag}.json" ]; then echo "=== $* (cached)"; return; fi
  echo "=== $*"; uv run schemagraph bench-spider2-lite "$ROOT" --out "$OUT" "$@"
}
if [ "$PART" = "1" ] || [ "$PART" = "all" ]; then
  run --tag default
  run --tag large_default --min-db-tables 100
  run --tag mt6_k3  --max-tables 6  --anchor-k 3
  run --tag mt20_k6 --max-tables 20 --anchor-k 6
  run --tag no_docs --no-docs
  run --tag no_infer --no-infer
fi
if [ "$PART" = "2" ] || [ "$PART" = "all" ]; then
  run --tag no_families --no-families
  run --tag no_idf --opt idf=false
  run --tag agg_sum --opt agg=sum
  run --tag no_bypass --opt bypass_if_fits=false
  run --tag no_adaptive --opt adaptive_budget=false
  run --tag routing_on --opt schema_routing=0.5
fi
