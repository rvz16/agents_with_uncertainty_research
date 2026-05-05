#!/usr/bin/env bash
# Launches Phase 2 SWE-bench harness eval for one generator's iterative
# refinement steps. Reads predictions_iter_step{1..N}.jsonl, runs harness
# on each, writes per-step harness reports under eval/.
#
# Usage: launch_iter_phase2.sh <generator> [n_steps]
set -euo pipefail
cd /mnt/data/users/vlad.smirnov/agents_with_uncertainty_research/.claude/worktrees/reverent-vaughan-017bf5/experiments/orchestration_hypothesis_testing
export DOCKER_HOST="unix:///run/user/$(id -u)/podman/podman.sock"
export SWEBENCH_PODMAN_COMPAT=1

GEN="${1:?usage: $0 <generator> [n_steps]}"
N_STEPS="${2:-5}"
OUTDIR="data/spot_check_n50/$GEN"
EVALDIR="data/spot_check_n50/eval"

# Run for each refinement step (1..N-1 — step 0 already evaluated in spot-check)
for STEP in $(seq 1 $((N_STEPS - 1))); do
  PRED="$OUTDIR/predictions_iter_step$STEP.jsonl"
  if [ ! -f "$PRED" ]; then
    echo "skip step $STEP: $PRED not found"
    continue
  fi
  RUN_ID="${GEN}_iter_step${STEP}"
  echo "==== eval $GEN step $STEP (run_id=$RUN_ID) ===="
  python3 -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path "$(realpath "$PRED")" \
    --max_workers 4 \
    --run_id "$RUN_ID" \
    --cache_level instance
done
