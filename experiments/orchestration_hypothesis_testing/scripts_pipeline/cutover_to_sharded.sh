#!/usr/bin/env bash
# Stop the running pipeline tmux session cleanly, then relaunch the same
# arguments with EVAL_SHARDS exported so the remaining eval steps fan out
# across shard hosts.
#
# Idempotency: the pipeline shell's underlying scripts (refine_swe.py,
# spot_check_generators.py, the harness itself) all skip already-completed
# work, so relaunching at any point picks up exactly where the previous run
# left off. The killer concern was only the SR Ver refine→eval transition,
# which is now what we are actually targeting on purpose.
#
# Usage:
#   cutover_to_sharded.sh <session_name> <eval_shards>
#
# Example:
#   cutover_to_sharded.sh swebench_qwen3_coder mbz3
#
# Reads the tmux session's command line via `tmux list-windows -F '#{pane_start_command}'`
# so the exact relaunch matches the original launch.

set -euo pipefail

SESSION="${1:?usage: $0 <tmux_session> <eval_shards>}"
EVAL_SHARDS_VAL="${2:?eval_shards (e.g. mbz3 or mbz3,mbz4)}"

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' not found" >&2
    exit 1
fi

# Capture original launch command line from the session's started_command.
ORIG_CMD=$(tmux list-windows -t "$SESSION" -F '#{pane_start_command}' | head -1)
if [ -z "$ORIG_CMD" ]; then
    echo "ERROR: could not read pane_start_command for session '$SESSION'" >&2
    exit 1
fi
echo "original command line:"
echo "  $ORIG_CMD"

# Kill cleanly. The pipeline's `set -e` plus the harness's per-instance
# log persistence means SIGTERM mid-step is safe: completed instances'
# reports are on disk; in-flight container will be reaped by the runtime.
echo ">> killing session $SESSION"
tmux kill-session -t "$SESSION"
sleep 2

# Relaunch with EVAL_SHARDS prepended. Use eval to handle the embedded
# env-var assignments and 2>&1 redirection from the original line.
echo ">> relaunching with EVAL_SHARDS=$EVAL_SHARDS_VAL"
RELAUNCH="EVAL_SHARDS=$EVAL_SHARDS_VAL $ORIG_CMD"
tmux new -d -s "$SESSION" "$RELAUNCH"
sleep 2

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo ">> tmux session $SESSION relaunched"
else
    echo "ERROR: relaunch failed; session not present" >&2
    exit 1
fi
