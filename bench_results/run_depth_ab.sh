#!/usr/bin/env bash
# Run A (new judge context, old search) then run B (AB-MCTS-M, rolling, deeper) on the 21-task subset.
# Usage: bench_results/run_depth_ab.sh /path/to/Spider2 [path/to/.env]; resumable per tag.
set -u
ROOT=${1:?path to the Spider2 clone}
ENV_FILE=${2:-.env}
ONLY=$(tr -d '\r' < bench_results/spider2_exec_subset21.txt | sed 's/^/--only /' | tr '\n' ' ')
export SCHEMAGRAPH_REASONING=high
uv run --env-file "$ENV_FILE" schemagraph bench-spider2-exec "$ROOT" \
  --strategy abmcts --budget 16 --batch 4 $ONLY --tag abmcts16_high_ctx > bench_results/run_a.log 2>&1
uv run --env-file "$ENV_FILE" schemagraph bench-spider2-exec "$ROOT" \
  --strategy abmcts --budget 16 --batch 4 --algorithm m --rolling \
  --early-stop-min-nodes 8 --early-stop-agree 2 --draft-temperature 1.0 \
  $ONLY --tag abmcts16_high_m_deep > bench_results/run_b.log 2>&1
