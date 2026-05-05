#!/usr/bin/env bash
# Launch lcb_calibrate.py fully detached from the SSH session.
# - setsid + nohup so SIGHUP from SSH disconnect doesn't kill it
# - stdout/stderr to log file
# - PID written to file for monitoring/cleanup
# - resumes from existing critic_results.jsonl thanks to the patched calibrator

set -eu
cd /mnt/data/users/vlad.smirnov/agents_with_uncertainty_research/.claude/worktrees/reverent-vaughan-017bf5/experiments/orchestration_hypothesis_testing

export PYTHONUNBUFFERED=1

# Load env (OPENROUTER_API_KEY)
for f in ../.env ./.env /mnt/data/users/vlad.smirnov/agents_with_uncertainty_research/.env; do
  if [ -f "$f" ]; then set -a; source "$f"; set +a; break; fi
done

OUT_DIR=data/lcb_calibration_v2
LOG="$OUT_DIR/calibration.log"
PID_FILE="$OUT_DIR/calibration.pid"

# stamp the resume start so we can grep later
echo "$(date -u +%FT%TZ) [LAUNCH] resuming under setsid" >> "$LOG"

setsid nohup python3 scripts/lcb_calibrate.py \
  --output-dir "$OUT_DIR" \
  --generators gpt5_mini,qwen3_coder \
  --n-instances 29 \
  --n-patches 3 \
  --difficulty hard \
  --platform leetcode \
  --max-cost-usd-per-model "gpt5_mini=3.0,qwen3_coder=3.0" \
  >> "$LOG" 2>&1 < /dev/null &
disown

PID=$!
echo "$PID" > "$PID_FILE"
echo "launched pid=$PID, log=$LOG"
