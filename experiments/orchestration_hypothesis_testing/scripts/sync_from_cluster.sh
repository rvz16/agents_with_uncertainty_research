#!/usr/bin/env bash
# Sync experiment artifacts from MBZUAI-Artem-1 to local.
# Excludes raw response/call payloads, harness logs, pyc caches.
#
# Usage:
#   ./scripts/sync_from_cluster.sh           # full sync (default)
#   ./scripts/sync_from_cluster.sh dry       # dry-run, show items
#   ./scripts/sync_from_cluster.sh paper-table  # just refresh PAPER_TABLE.json + backups
#
# Run from: experiments/orchestration_hypothesis_testing/

set -euo pipefail

REMOTE_HOST="MBZUAI-Artem-1"
REMOTE_PATH="/mnt/data/users/vlad.smirnov/agents_with_uncertainty_research/experiments/orchestration_hypothesis_testing/data"
LOCAL_PATH="data"

# Common excludes: bulky raw payloads we don't need for analysis
EXCLUDES=(
  --exclude='raw_responses/'
  --exclude='raw_calls/'
  --exclude='iter_raw_responses/'
  --exclude='eval_logs/'
  --exclude='*.log'
  --exclude='*.pid'
  --exclude='__pycache__/'
  --exclude='.DS_Store'
)

mode="${1:-full}"

case "$mode" in
  dry)
    echo "[dry-run] Showing what would transfer (excludes applied)..."
    rsync -avn "${EXCLUDES[@]}" \
      "$REMOTE_HOST:$REMOTE_PATH/" "$LOCAL_PATH/" | head -100
    echo
    echo "Item count:"
    rsync -avn "${EXCLUDES[@]}" --itemize-changes \
      "$REMOTE_HOST:$REMOTE_PATH/" "$LOCAL_PATH/" 2>&1 | grep -cE '^[<>]'
    ;;
  paper-table)
    echo "Refreshing only PAPER_TABLE.{json,csv} + backups..."
    rsync -av \
      --include='PAPER_TABLE*' \
      --include='*.bak' \
      --exclude='*' \
      "$REMOTE_HOST:$REMOTE_PATH/" "$LOCAL_PATH/"
    ;;
  full|*)
    echo "Full sync from $REMOTE_HOST (excludes applied)..."
    rsync -av --progress --stats "${EXCLUDES[@]}" \
      "$REMOTE_HOST:$REMOTE_PATH/" "$LOCAL_PATH/"
    echo
    echo "Done. New local size:"
    du -sh "$LOCAL_PATH"
    ;;
esac
