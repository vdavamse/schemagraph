#!/usr/bin/env bash
# Multi-LLM AB-MCTS (arXiv 2503.04412, appendix D) on 21 held-out tasks, 64 nodes a task, with the
# protocol of the paper's ARC-AGI-2 experiment (D.2.1): temperature 0.6 for drafts and refinements,
# the whole budget spent (no early stop), Pass@k over generator calls read from each row's
# first_correct_node. The paper's search reward came from demonstration cases; ours is the judge.
# Every multi-LLM node links the wide context, so the single-LLM arms search it alone too.
#
# Usage: bench_results/run_multi_llm.sh /path/to/Spider2 path/to/.env ARM...
# Arms (estimated at list prices and Qwen's token profile, about 16 hours each at one task at a
# time; OpenRouter holds new accounts to 20 requests a minute on Qwen):
#   smoke     one draft from each model on 2 tasks: checks every model's tool calls, about $0.30
#   qwen_m    single-LLM AB-MCTS-M (Qwen), about $63
#   multi_m1  multi-LLM AB-MCTS-M, generator selection algorithm I (eq. 22), about $60
#   multi_m2  multi-LLM AB-MCTS-M, generator selection algorithm II, about $60
#   qwen_a    single-LLM AB-MCTS-A (Qwen), the paper's single-LLM baseline, about $63
#   multi_a2  multi-LLM AB-MCTS-A, algorithm II: the paper's D.2.1 configuration, about $60
# Resumable per tag; logs go to bench_results/run_<tag>.log.
set -euo pipefail
ROOT="${1:?path to the Spider2 clone}"
ENV_FILE="$(realpath -m "${2:?path to the .env with the model keys}")"
shift 2
ARMS=("$@")
OUT="$(cd "$(dirname "$0")" && pwd)"
cd "$OUT/.."
HELDOUT="$OUT/spider2_exec_heldout21.txt"
[ ${#ARMS[@]} -gt 0 ] || { echo "name the arms to run (see the header)" >&2; exit 1; }
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
PROTOCOL=(
  --strategy abmcts --budget 64 --batch 4 --rolling --early-stop-min-nodes 64
  --draft-temperature 0.6 --refine-temperature 0.6
)
SINGLE=(--action wide --gen-model "${GENERATORS[0]}")
# held-out ids (the list may have CRLF line ends on a Windows checkout)
mapfile -t TASKS < <(tr -d '\r' < "$HELDOUT" | grep -v '^$')
export SCHEMAGRAPH_REASONING=high

arm_options() {  # sets OPTIONS and ARM_TASKS for one arm; fails on an unknown arm
  ARM_TASKS=("${TASKS[@]}")
  case $1 in
    smoke)
      ARM_TASKS=("${TASKS[@]:0:2}")
      OPTIONS=(--strategy best_of_n --budget 5 --batch 5 --no-selector "${MULTI[@]}") ;;
    qwen_m)   OPTIONS=("${PROTOCOL[@]}" --algorithm m "${SINGLE[@]}") ;;
    multi_m1) OPTIONS=("${PROTOCOL[@]}" --algorithm m --generator-selection 1 "${MULTI[@]}") ;;
    multi_m2) OPTIONS=("${PROTOCOL[@]}" --algorithm m --generator-selection 2 "${MULTI[@]}") ;;
    qwen_a)   OPTIONS=("${PROTOCOL[@]}" --algorithm a "${SINGLE[@]}") ;;
    multi_a2) OPTIONS=("${PROTOCOL[@]}" --algorithm a --generator-selection 2 "${MULTI[@]}") ;;
    *) echo "unknown arm $1 (see the header)" >&2; return 1 ;;
  esac
}

for arm in "${ARMS[@]}"; do  # check every arm before paying for any
  arm_options "$arm"
done
for arm in "${ARMS[@]}"; do
  arm_options "$arm"
  tag="abmcts64_high_${arm}_heldout21"
  [ "$arm" = smoke ] && tag=multi_llm_smoke
  ONLY=()
  for task in "${ARM_TASKS[@]}"; do
    ONLY+=(--only "$task")
  done
  echo "=== $tag: ${#ARM_TASKS[@]} tasks, log $OUT/run_${tag}.log"
  uv run --env-file "$ENV_FILE" schemagraph bench-spider2-exec "$ROOT" "${ONLY[@]}" --tag "$tag" \
    "${OPTIONS[@]}" > "$OUT/run_${tag}.log" 2>&1
done
