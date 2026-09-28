#!/usr/bin/env bash
# Multi-LLM AB-MCTS-M against single-LLM AB-MCTS-M (Qwen) on the 60 held-out tasks, with the same
# budget and run B's deep search settings (AB-MCTS-M, rolling, 8 nodes before an early stop on 2
# agreeing results, draft temperature 1.0). In the multi-LLM run the search's actions are the five
# generator models: AB-MCTS-M decides between a new node and a refinement, then samples the new
# node's model with the models as groups of its mixed model (arXiv 2503.04412, appendix D,
# generator selection algorithm I). Every node links the wide context, so the Qwen run searches
# the wide context only too (--action wide): the two runs differ only in the models.
#
# Usage: bench_results/run_multi_llm.sh /path/to/Spider2 [path/to/.env]
#   SMOKE=1  instead, one draft from each model on 2 held-out tasks (best_of_n, 5 drafts): checks
#            every model's tool calls and cost per call, for about $0.30. Run it first.
# Estimated cost at list prices and Qwen's token profile (terser models cost less): single-LLM
# about $40, multi-LLM about $36; about 10 hours each, one task at a time (OpenRouter holds new
# accounts to 20 requests a minute on Qwen). Resumable per tag; logs go to bench_results/run_*.log.
set -euo pipefail
ROOT="${1:?path to the Spider2 clone}"
ENV_FILE="$(realpath -m "${2:-.env}")"
OUT="$(cd "$(dirname "$0")" && pwd)"
cd "$OUT/.."
HELDOUT="$OUT/spider2_exec_heldout60.txt"
[ -s "$HELDOUT" ] || { echo "missing $HELDOUT: run bench_results/select_heldout.py" >&2; exit 1; }
[ -f "$ENV_FILE" ] || { echo "missing $ENV_FILE: the model keys" >&2; exit 1; }

# The first model is also the default generator and the critic.
GENERATORS=(
  openrouter:qwen/qwen3.8-max-0902
  openrouter:z-ai/glm-5.3
  openrouter:x-ai/grok-4.7
  openrouter:google/gemini-3.8-flash
  openrouter:openai/gpt-6-sol
)
MULTI=()
for model in "${GENERATORS[@]}"; do
  MULTI+=(--gen-model "$model")
done
DEEP=(
  --strategy abmcts --budget 16 --batch 4 --algorithm m --rolling
  --early-stop-min-nodes 8 --early-stop-agree 2 --draft-temperature 1.0
)

# held-out ids (the list may have CRLF line ends on a Windows checkout); the smoke test takes two
mapfile -t TASKS < <(tr -d '\r' < "$HELDOUT" | grep -v '^$')
if [ "${SMOKE:-0}" = 1 ]; then
  TASKS=("${TASKS[@]:0:2}")
fi
ONLY=()
for task in "${TASKS[@]}"; do
  ONLY+=(--only "$task")
done
export SCHEMAGRAPH_REASONING=high

bench_run() {  # tag, then bench-spider2-exec options
  local tag=$1
  shift
  echo "=== $tag: ${#TASKS[@]} tasks, log $OUT/run_${tag}.log"
  uv run --env-file "$ENV_FILE" schemagraph bench-spider2-exec "$ROOT" "${ONLY[@]}" --tag "$tag" \
    "$@" > "$OUT/run_${tag}.log" 2>&1
}

if [ "${SMOKE:-0}" = 1 ]; then
  bench_run multi_llm_smoke --strategy best_of_n --budget 5 --batch 5 --no-selector "${MULTI[@]}"
  exit
fi
bench_run abmcts16_high_m_qwen_wide_heldout60 "${DEEP[@]}" --action wide --gen-model "${GENERATORS[0]}"
bench_run abmcts16_high_m_multi5_heldout60 "${DEEP[@]}" "${MULTI[@]}"
