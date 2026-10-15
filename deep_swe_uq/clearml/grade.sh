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
# Only the client is needed: the daemon is the host's, reached through the mounted
# socket. apt installed it on the earlier runs and then silently failed on this
# worker, and because every attempt was "|| true" with output to /dev/null the log
# said nothing but "command not found". So: report each attempt, and prefer the
# static binary, which needs no package manager and no mirror.
DOCKER_CLI_VERSION="${DOCKER_CLI_VERSION:-29.9.0}"
if ! command -v docker >/dev/null 2>&1; then
  echo "[grade] no docker client, installing"
  arch="$(uname -m)"
  url="https://download.docker.com/linux/static/stable/${arch}/docker-${DOCKER_CLI_VERSION}.tgz"
  if curl -fsSL --max-time 300 "$url" 2>/tmp/dl.err \
       | tar xz -C /tmp --strip-components=1 docker/docker 2>/tmp/tar.err \
       && install -m 0755 /tmp/docker /usr/local/bin/docker; then
    echo "[grade] installed the static client ${DOCKER_CLI_VERSION} for ${arch}"
  else
    echo "[grade] static client failed: $(tail -c 200 /tmp/dl.err 2>/dev/null) $(tail -c 200 /tmp/tar.err 2>/dev/null)"
    apt-get update -qq -o Acquire::AllowInsecureRepositories=true >/tmp/apt.log 2>&1 \
      || echo "[grade] apt-get update failed: $(tail -c 200 /tmp/apt.log)"
    apt-get install -y -qq --no-install-recommends docker.io >>/tmp/apt.log 2>&1 \
      || echo "[grade] apt-get install failed: $(tail -c 300 /tmp/apt.log)"
  fi
  command -v docker >/dev/null 2>&1 \
    || curl -fsSL --max-time 300 https://get.docker.com 2>/tmp/get.err | sh >/tmp/get.log 2>&1 \
    || echo "[grade] get.docker.com failed: $(tail -c 200 /tmp/get.err 2>/dev/null) $(tail -c 200 /tmp/get.log 2>/dev/null)"
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "[grade] VERDICT: no docker client could be installed on this worker"; exit 21
fi
ls -la /var/run/docker.sock 2>&1 | head -2
docker version --format 'client={{.Client.Version}} server={{.Server.Version}}' 2>&1 | head -2
if ! docker ps >/dev/null 2>&1; then
  echo "[grade] VERDICT: docker daemon unreachable from this worker"; exit 20
fi
echo "[grade] docker reachable, arch $(uname -m)"

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
