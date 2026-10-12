#!/usr/bin/env bash
# Replay recorded ALFWorld episodes to recover a label at every step.
# No model is served and nothing is generated: the executed actions are already
# in the log, so this only needs the environment and the dataset.
set -uo pipefail

OUT_DIR="${OUT_DIR:-/tmp/replay_runs}"
INPUTS="${INPUTS:-data/replay_inputs}"
COHORTS="${COHORTS:-RG SG RQ SQ}"
PROBE="${PROBE:-0}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-/root/.cache/alfworld}"

mkdir -p "$OUT_DIR"
rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list || true
rm -rf /var/lib/apt/lists/* || true
mkdir -p /tmp/aptcache/partial
apt-get -o Dir::Cache::archives=/tmp/aptcache -o Acquire::AllowInsecureRepositories=true update -qq >/dev/null 2>&1 || true
apt-get -o Dir::Cache::archives=/tmp/aptcache install -y -qq --no-install-recommends \
    --allow-unauthenticated build-essential libffi-dev unzip >/dev/null 2>&1 || true
python -m pip install -q --no-cache-dir "alfworld==0.4.2" "textworld[pddl]==1.7.0"

games=$(find "${ALFWORLD_DATA}/json_2.1.1" -name game.tw-pddl 2>/dev/null | wc -l | tr -d ' ') || games=0
if [ "$games" -lt 100 ]; then
    echo "[replay] downloading ALFWorld data into ${ALFWORLD_DATA}"
    for attempt in 1 2 3; do
        alfworld-download || true
        games=$(find "${ALFWORLD_DATA}/json_2.1.1" -name game.tw-pddl 2>/dev/null | wc -l | tr -d ' ') || games=0
        [ "$games" -ge 100 ] && break
    done
fi
echo "[replay] ${games} games available"

cd alfworld_uq
for c in $COHORTS; do
    echo "=== cohort $c ==="
    if [ "$PROBE" != "0" ]; then
        # before building anything on a graded label, see what the environment returns
        python -m experiments.replay_labels --input "$INPUTS/$c.jsonl" --cohort "$c" \
            --out "$OUT_DIR" --probe "$PROBE" || echo "[replay] $c probe failed"
    else
        python -m experiments.replay_labels --input "$INPUTS/$c.jsonl" --cohort "$c" \
            --out "$OUT_DIR" || echo "[replay] $c failed, continuing"
    fi
done
echo "[replay] done"
ls -la "$OUT_DIR"
