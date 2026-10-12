#!/usr/bin/env bash
# Grade DeepSWE patches outside pier, using each task's own verifier container.
# The validation gate runs first: a task whose golden solution does not score
# 1.0 has an environment problem, and any label from it would be meaningless.
set -uo pipefail
OUT_DIR="${OUT_DIR:-/tmp/grade_runs}"
SHARED="${SHARED:-/tmp/grade_shared}"
DEEPSWE_COMMIT="${DEEPSWE_COMMIT:-e016041}"
COMMAND="${COMMAND:-validate}"
LIMIT="${LIMIT:-5}"
ONLY="${ONLY:-}"

mkdir -p "$OUT_DIR" "$SHARED"
command -v docker >/dev/null 2>&1 || {
  apt-get update -qq -o Acquire::AllowInsecureRepositories=true >/dev/null 2>&1 || true
  apt-get install -y -qq --no-install-recommends docker.io >/dev/null 2>&1 || true; }
docker ps >/dev/null 2>&1 || { echo "[grade] VERDICT: docker daemon unreachable"; exit 20; }
echo "[grade] docker $(docker version --format '{{.Server.Version}}' 2>/dev/null)"
echo "[grade] arch $(uname -m)"

if [ ! -d "${SHARED}/deep-swe/tasks" ]; then
  git clone -q https://github.com/datacurve-ai/deep-swe "${SHARED}/deep-swe" || {
    echo "[grade] VERDICT: clone failed"; exit 22; }
  git -C "${SHARED}/deep-swe" checkout -q "${DEEPSWE_COMMIT}" || exit 23
fi
echo "[grade] tasks: $(ls "${SHARED}/deep-swe/tasks" | wc -l) at ${DEEPSWE_COMMIT}"

cd deep_swe_uq
python -m experiments.grade_patch "$COMMAND" \
    --tasks "${SHARED}/deep-swe/tasks" --out "$OUT_DIR" --work "${SHARED}/work" \
    ${LIMIT:+--limit "$LIMIT"} ${ONLY:+--only $ONLY}
echo "[grade] done"; ls -la "$OUT_DIR"
