#!/usr/bin/env bash
# Launch sonnet45 pipeline on mbz1, resuming from where it died at Step 6.
#
# Prerequisites (verified before running this):
#   1. mbz2 sonnet45 data rsynced to mbz1 under
#      data/swebench_{lite,verified}_{calibration_full,realbaselines_selfrefine_full}/sonnet45/
#   2. eval_steps.py with empty-predictions skip deployed (commit 21b0190b)
#   3. Sharded harness wrapper present (commit 7edbaf1d)
#   4. Workspace + key both positive (key $400 cap, workspace $3,350 cap)
#   5. qwen3 pipeline finished (mbz1 + mbz3 free)
#
# What it does:
#   Resume guard skips Steps 1-4 (Cal Lite + Cal Verified + SR Lite refine/eval).
#   Step 5 SR Verified refine is idempotent on iter_records (no new work — keeps
#     the existing sparse 10/7/3/0 trajectory predictions; this is a research
#     finding, not a bug). Step 6 evals the existing SR Ver reports
#     (step1/2/3 already on disk; step4 predictions are empty and the patched
#     eval_steps.py will skip it cleanly).
#   Step 7 Rfx Lite refine runs fresh on Cal Lite data (~$25-35 LLM).
#   Step 8 Rfx Lite eval sharded mbz1+mbz3 (Reflexion attrition cuts work).
#   Step 9 Rfx Ver refine runs fresh on Cal Verified data (~$10-15 LLM).
#   Step 10 Rfx Ver eval sharded mbz1+mbz3.
#
# Cost caps tuned to remaining budget (workspace $149 remaining, key $96):
#   REFINE_LITE_COST=15 (Rfx Lite cap)
#   REFINE_VER_COST=10 (Rfx Ver cap)
#   SPOTCHECK_COST defaults to $10
#
# Verified subset: same as qwen3, 200 instances.

set -euo pipefail

REPO=/mnt/data/users/vlad.smirnov/agents_with_uncertainty_research
SUBSET=$REPO/experiments/orchestration_hypothesis_testing/data/swebench_verified_calibration_full/verified_200_instance_ids.json

tmux new -d -s swebench_sonnet45 \
    "EVAL_SHARDS=mbz3 \
     VERIFIED_SUBSET=$SUBSET \
     MAX_REFINE_WORKERS=8 \
     REFINE_LITE_COST=15 \
     REFINE_VER_COST=10 \
     SWEBENCH_NAMESPACE=none \
     bash $REPO/experiments/orchestration_hypothesis_testing/scripts_pipeline/run_swebench_pipeline.sh \
        sonnet45 50 25 2>&1 \
     | tee -a $REPO/logs/swebench_pipeline_sonnet45/master.log"

sleep 3
tmux list-sessions
echo "---tail master log:---"
tail -5 $REPO/logs/swebench_pipeline_sonnet45/master.log
