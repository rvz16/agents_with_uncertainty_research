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
LIMIT="${LIMIT:-5}"
ONLY="${ONLY:-}"

mkdir -p "$OUT_DIR" "$SHARED"
# The same client install the full DeepSWE runs use, curl fallback included: the
# first attempt dropped that fallback and died with "docker daemon unreachable"
# on a worker where apt could not reach the package.
if ! command -v docker >/dev/null 2>&1; then
  echo "[grade] no docker client, installing"
  (apt-get update -qq -o Acquire::AllowInsecureRepositories=true >/dev/null 2>&1 || true)
  (apt-get install -y -qq --no-install-recommends docker.io >/dev/null 2>&1 || true)
  command -v docker >/dev/null 2>&1 || curl -fsSL https://get.docker.com | sh >/dev/null 2>&1 || true
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
        --tasks "${SHARED}/deep-swe/tasks" --out "${SHARED}/patches_${COHORT}" \
        --snapshots "$SNAPSHOTS" ${LIMIT:+--limit "$LIMIT"} || exit 31
    python -m experiments.grade_patch grade \
        --tasks "${SHARED}/deep-swe/tasks" --out "$OUT_DIR" --work "${SHARED}/work" \
        --patches "${SHARED}/patches_${COHORT}"
    mv "$OUT_DIR/labels.jsonl" "$OUT_DIR/labels_${COHORT}.jsonl" 2>/dev/null || true
else
    python -m experiments.grade_patch "$COMMAND" \
        --tasks "${SHARED}/deep-swe/tasks" --out "$OUT_DIR" --work "${SHARED}/work" \
        ${LIMIT:+--limit "$LIMIT"} ${ONLY:+--only $ONLY}
fi
echo "[grade] done"; ls -la "$OUT_DIR"
