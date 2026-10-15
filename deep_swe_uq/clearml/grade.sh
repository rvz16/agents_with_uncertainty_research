#!/usr/bin/env bash
# Grade DeepSWE patches outside pier, using each task's own verifier container.
# The validation gate runs first: a task whose golden solution does not score
# 1.0 has an environment problem, and any label from it would be meaningless.
set -uo pipefail
OUT_DIR="${OUT_DIR:-/tmp/grade_runs}"
SHARED="${SHARED:-/tmp/grade_shared}"
DEEPSWE_COMMIT="${DEEPSWE_COMMIT:-e016041}"
COMMAND="${COMMAND:-validate}"
COHORT="${COHORT:-DG}"
SNAPSHOTS="${SNAPSHOTS:-8}"
# no default: a run that silently covers 5 of 112 episodes looks like a success
LIMIT="${LIMIT:-}"
ONLY="${ONLY:-}"
# SHARD is "I/N": replay every Nth episode from I. Each shard keeps its own
# patches directory and labels file, so shards can share a worker and a failure
# costs one shard rather than the cohort.
SHARD="${SHARD:-}"
TAG=""
if [ -n "$SHARD" ]; then TAG="_s${SHARD//\//of}"; fi

mkdir -p "$OUT_DIR" "$SHARED"
source "$(dirname "${BASH_SOURCE[0]}")/docker_client.sh" || exit 20

if [ ! -d "${SHARED}/deep-swe/tasks" ]; then
  git clone -q https://github.com/datacurve-ai/deep-swe "${SHARED}/deep-swe" || {
    echo "[grade] VERDICT: clone failed"; exit 22; }
  git -C "${SHARED}/deep-swe" checkout -q "${DEEPSWE_COMMIT}" || exit 23
fi
echo "[grade] tasks: $(ls "${SHARED}/deep-swe/tasks" | wc -l) at ${DEEPSWE_COMMIT}"

cd deep_swe_uq
if [ "$COMMAND" = "replay" ]; then
    # reconstruct the repository partway through each trajectory, then grade those
    # snapshots with the task's own verifier -- the golden check gates this
    python -m experiments.replay_patches \
        --compact "data/replay_inputs/${COHORT}.jsonl.gz" \
        --tasks "${SHARED}/deep-swe/tasks" --out "${SHARED}/patches_${COHORT}${TAG}" \
        --snapshots "$SNAPSHOTS" ${LIMIT:+--limit "$LIMIT"} ${SHARD:+--shard "$SHARD"} || exit 31
    python -m experiments.grade_patch grade \
        --tasks "${SHARED}/deep-swe/tasks" --out "$OUT_DIR" --work "${SHARED}/work${TAG}" \
        --patches "${SHARED}/patches_${COHORT}${TAG}"
    mv "$OUT_DIR/labels.jsonl" "$OUT_DIR/labels_${COHORT}${TAG}.jsonl" 2>/dev/null || true
elif [ "$COMMAND" = "grade" ]; then
    # grade snapshots a previous replay already wrote. The patches live under the
    # host-mounted SHARED directory, and both high_q workers are on one host, so a
    # replay that cost two hours does not have to be repeated after a grading bug.
    PATCHES="${PATCHES:-${SHARED}/patches_${COHORT}${TAG}}"
    if [ ! -d "$PATCHES" ]; then
        echo "[grade] VERDICT: no patches at $PATCHES"; exit 24
    fi
    echo "[grade] grading snapshots already in $PATCHES ($(ls "$PATCHES" | wc -l) episodes)"
    python -m experiments.grade_patch grade \
        --tasks "${SHARED}/deep-swe/tasks" --out "$OUT_DIR" --work "${SHARED}/work${TAG}" \
        --patches "$PATCHES" ${LIMIT:+--limit "$LIMIT"}
    mv "$OUT_DIR/labels.jsonl" "$OUT_DIR/labels_${COHORT}${TAG}.jsonl" 2>/dev/null || true
else
    python -m experiments.grade_patch "$COMMAND" \
        --tasks "${SHARED}/deep-swe/tasks" --out "$OUT_DIR" --work "${SHARED}/work" \
        ${LIMIT:+--limit "$LIMIT"} ${ONLY:+--only $ONLY}
fi
echo "[grade] done"; ls -la "$OUT_DIR"
