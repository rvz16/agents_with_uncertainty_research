#!/usr/bin/env bash
# Parallel-friendly launcher: one detached process per (generator, difficulty).
# Patched calibrator resumes from any existing critic_results.jsonl, so
# re-launching is safe.
#
# Usage:
#   bash launch_lcb_medium_parallel.sh <gen> [difficulty] [out_dir] [cap_usd] [n_inst]
#   gen        gpt5_mini | qwen3_coder | haiku45 | sonnet45
#   difficulty hard | medium | easy   (default: medium)
#   out_dir    output dir (default: data/lcb_calibration_medium for medium,
#                                   data/lcb_calibration_v2 for hard)
#   cap_usd    per-model cap (default: 5.0)
#   n_inst     instances (default: matches LCB pool size for the difficulty)
set -eu
GEN="${1:?usage: $0 <gen> [difficulty] [out_dir] [cap_usd] [n_inst]}"
case "$GEN" in
  gpt5_mini|qwen3_coder|haiku45|sonnet45) ;;
  *) echo "unknown generator: $GEN" >&2; exit 1 ;;
esac

DIFF="${2:-medium}"
case "$DIFF" in
  hard|medium|easy) ;;
  *) echo "unknown difficulty: $DIFF" >&2; exit 1 ;;
esac

# Default out_dir per difficulty
case "$DIFF" in
  hard)   DEFAULT_OUT=data/lcb_calibration_v2 ;;
  medium) DEFAULT_OUT=data/lcb_calibration_medium ;;
  easy)   DEFAULT_OUT=data/lcb_calibration_easy ;;
esac
OUT_DIR="${3:-$DEFAULT_OUT}"

CAP="${4:-5.0}"

# Default n_inst per difficulty (LCB pool sizes)
case "$DIFF" in
  hard)   DEFAULT_N=29 ;;
  medium) DEFAULT_N=90 ;;
  easy)   DEFAULT_N=62 ;;
esac
N_INST="${5:-$DEFAULT_N}"

cd /mnt/data/users/vlad.smirnov/agents_with_uncertainty_research/.claude/worktrees/reverent-vaughan-017bf5/experiments/orchestration_hypothesis_testing

export PYTHONUNBUFFERED=1
for f in ../.env ./.env /mnt/data/users/vlad.smirnov/agents_with_uncertainty_research/.env; do
  if [ -f "$f" ]; then set -a; source "$f"; set +a; break; fi
done

LOG="$OUT_DIR/calibration.${GEN}.log"
PID_FILE="$OUT_DIR/calibration.${GEN}.pid"
mkdir -p "$OUT_DIR"
echo "$(date -u +%FT%TZ) [LAUNCH] [$GEN/$DIFF/n=$N_INST/cap=\$$CAP] under setsid" >> "$LOG"

setsid nohup python3 scripts/lcb_calibrate.py \
  --output-dir "$OUT_DIR" \
  --generators "$GEN" \
  --n-instances "$N_INST" \
  --n-patches 3 \
  --difficulty "$DIFF" \
  --platform leetcode \
  --max-cost-usd-per-model "${GEN}=${CAP}" \
  >> "$LOG" 2>&1 < /dev/null &
disown

PID=$!
echo "$PID" > "$PID_FILE"
echo "[$GEN/$DIFF] launched pid=$PID, log=$LOG, out=$OUT_DIR, cap=\$$CAP"
