#!/usr/bin/env bash
# Full SWE-Bench pipeline (Lite + Verified) for one generator.
# 14 steps total: calibration eval -> from_spotcheck -> Self-Refine refine/eval/backfill ->
# Reflexion refine/eval/backfill, on both SWE-Bench Lite and SWE-Bench Verified.
#
# Prerequisites (one-time per host)
# ---------------------------------
# 1. Rootless podman 3.4.4 running with user systemd socket:
#       systemctl --user enable --now podman.socket
#
# 2. Patch the installed swebench harness for podman compatibility:
#       python experiments/orchestration_hypothesis_testing/scripts/patch_swebench_harness.py
#    (see that script for why — TL;DR: docker SDK build path is broken on
#    podman 3.4.4 for unqualified FROM image names, and the platform kwarg
#    is unsupported on API < v1.41.)
#
# 3. .env at repo root with OPENROUTER_API_KEY=sk-or-v1-...
#
# Usage
# -----
#   ./run_swebench_pipeline.sh <generator> <lite_cost> <verified_cost>
#
# Example:
#   ./run_swebench_pipeline.sh qwen3_coder 15 25
#
# Notes
# -----
# * The cost caps bound LLM spend per benchmark for the cal step. Each cell
#   is 300 inst x 3 patches (Lite) or 500 inst x 3 patches (Verified). Tune
#   for your generator price tier (claude-sonnet-4.5 ~ 8x qwen3_coder).
# * LLM gen is resumable: re-running with the same args after a partial run
#   detects existing predictions and only re-evaluates.
# * Local builds use TMPDIR on /mnt/data so buildah scratch doesn't blow
#   through the root quota (typically 200 GB on /).
# * Expected wall-clock: 24-36 hours for one generator on a single host.
#   Local builds are slower than Docker Hub pulls (5-15 min vs 30s per
#   first build per env-image hash), but they avoid the ~28% error rate
#   from missing swebench/* Docker Hub images (astropy, matplotlib,
#   recent django).

set -euo pipefail

GEN="${1:?usage: $0 <generator> <lite_cost> <verified_cost>}"
LITE_COST="${2:?lite_cost (USD)}"
VER_COST="${3:?verified_cost (USD)}"

SPOTCHECK_COST=10
N_LITE=300
N_VER=500
N_PATCHES=3
N_STEPS=5

# Refinement cost caps (Self-Refine + Reflexion, per benchmark).
# refine_swe.py defaults to $5 which is too low for 300/500-instance paid
# runs and would stop trajectories early with stop_reason="cost_cap",
# producing partial baselines. Default to the calibration caps (same
# order of magnitude per benchmark) so the budget scales with generator
# pricing. Override via env if needed.
REFINE_LITE_COST="${REFINE_LITE_COST:-$LITE_COST}"
REFINE_VER_COST="${REFINE_VER_COST:-$VER_COST}"

# Refinement concurrency. Default 1 (serial) because refine_swe.py's
# per-model --max-cost-usd-per-model cap is only checked at step
# boundaries: with N>1 workers, all N can pass the cap check before
# any of them debits their spend, leading to a worst-case overrun of
# ~N×cost_per_step. Override via MAX_REFINE_WORKERS=4 (or higher) when
# the budget cap has comfortable headroom and you want the ~Nx wall-
# clock speedup. The refinement loop itself is pure LLM and tolerates
# concurrency fine — the only concern is the cap-check race.
MAX_REFINE_WORKERS="${MAX_REFINE_WORKERS:-1}"

# Optional: pin all SWE-Bench Verified steps to a pre-chosen subset by setting
# VERIFIED_SUBSET=/abs/path/to/<file>.json. The file MAY be either a flat
# JSON array of instance_ids OR a dict with an 'instance_ids' key; the
# script normalises both. When set, N_VER is also overridden so the cap
# matches the subset size, and refine_swe.py / spot_check_generators both
# receive --instance-ids-file. Leave unset to run all 500 Verified.
#
# Parsing of this file happens AFTER conda activation below, because tmux
# non-interactive shells don't have `python` on PATH until conda is sourced.
VERIFIED_SUBSET="${VERIFIED_SUBSET:-}"
VER_SUBSET_ARG=()

# Optional: fan eval calls out across multiple SSH hosts to compress the
# Docker-bound wall clock. Set EVAL_SHARDS=<host1>[,<host2>...] (comma list of
# ssh aliases). The local host is always shard 0; each listed host becomes an
# additional shard. Each remote host MUST already have the same git checkout,
# patched swebench install, .env, podman socket, and TMPDIR configured. See
# scripts/sharded_swebench_eval.py for details and the per-host bootstrap
# checklist. Leave unset to run single-host (current default).
EVAL_SHARDS="${EVAL_SHARDS:-}"
export EVAL_SHARDS

# Resolve the repo root from this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO="$(cd "${PIPE_DIR}/../.." && pwd)"
LOG_DIR="$REPO/logs/swebench_pipeline_${GEN}"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"

log() {
    printf "==== %s | %s ====\n" "$(date '+%F %T')" "$1" | tee -a "$MASTER_LOG"
}

# Activate conda base env (tmux's non-interactive shell doesn't inherit PATH).
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate base
fi

# Load .env (OPENROUTER_API_KEY etc.)
if [ -f "$REPO/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$REPO/.env"
    set +a
fi

# Rootless podman socket.
export DOCKER_HOST="unix:///run/user/$(id -u)/podman/podman.sock"
export SWEBENCH_PODMAN_COMPAT=1

# Force local image builds so the patcher's TAR_OPTIONS / pip-downgrade fixes
# take effect at instance-build time. Without this the harness pulls from
# docker.io/swebench/* and ignores the patches. The wrapper in
# scripts/spot_check_generators.py:run_swebench_eval() honours this env var.
export SWEBENCH_NAMESPACE=none

# Buildah layer-commit scratch -> /mnt/data so we don't hit the root quota.
# (Override BUILDAH_SCRATCH_DIR via env if /mnt/data isn't appropriate on your host.)
SCRATCH_DIR="${BUILDAH_SCRATCH_DIR:-/mnt/data/users/$(whoami)/buildah-tmp}"
mkdir -p "$SCRATCH_DIR"
export TMPDIR="$SCRATCH_DIR"
export BUILDAH_TMPDIR="$SCRATCH_DIR"

cd "$PIPE_DIR"

# Subset parsing (deferred from config block so python is available).
if [ -n "$VERIFIED_SUBSET" ]; then
    if [ ! -f "$VERIFIED_SUBSET" ]; then
        echo "ERROR: VERIFIED_SUBSET='$VERIFIED_SUBSET' does not exist" >&2
        exit 1
    fi
    VER_SUBSET_ARG=(--instance-ids-file "$VERIFIED_SUBSET")
    # Override Verified instance count to subset size. Handle both
    # flat-list and dict-with-instance_ids-key JSON formats.
    N_VER=$(python - "$VERIFIED_SUBSET" <<'PYEOF'
import json, sys
d = json.loads(open(sys.argv[1]).read())
ids = d if isinstance(d, list) else d["instance_ids"]
print(len(ids))
PYEOF
)
    echo "VERIFIED_SUBSET active: $N_VER instances from '$VERIFIED_SUBSET'"
fi

if [ -n "$EVAL_SHARDS" ]; then
    log "EVAL_SHARDS active: local + [$EVAL_SHARDS]"
fi
log "START: $GEN  cap_lite=\$$LITE_COST  cap_verified=\$$VER_COST"

# ─── Step 1: Cal Lite (resumes LLM, evals with patched harness) ───────
log "1/14 Cal Lite (n=$N_LITE patches=$N_PATCHES cap=\$$LITE_COST)"
python scripts/spot_check_generators.py \
    --dataset princeton-nlp/SWE-bench_Lite \
    --n-instances $N_LITE --n-patches $N_PATCHES \
    --generators "$GEN" \
    --output-dir data/swebench_lite_calibration_full \
    --max-cost-usd-per-model "$GEN=$LITE_COST" \
    2>&1 | tee "$LOG_DIR/01_cal_lite.log"

# ─── Step 1b: from_spotcheck Lite -> critic_results.jsonl ─────────────
log "1b/14 from_spotcheck Lite -> critic_results.jsonl"
python calibration/from_spotcheck.py \
    --output-dir data/swebench_lite_calibration_full \
    --generators "$GEN" \
    --dataset princeton-nlp/SWE-bench_Lite \
    --max-cost-usd-per-model $SPOTCHECK_COST \
    2>&1 | tee "$LOG_DIR/01b_from_spotcheck_lite.log"

# ─── Step 2: Cal Verified ─────────────────────────────────────────────
log "2/14 Cal Verified (n=$N_VER patches=$N_PATCHES cap=\$$VER_COST)"
python scripts/spot_check_generators.py \
    --dataset princeton-nlp/SWE-bench_Verified \
    --n-instances $N_VER --n-patches $N_PATCHES \
    --generators "$GEN" \
    --output-dir data/swebench_verified_calibration_full \
    --max-cost-usd-per-model "$GEN=$VER_COST" \
    "${VER_SUBSET_ARG[@]}" \
    2>&1 | tee "$LOG_DIR/02_cal_verified.log"

# ─── Step 2b: from_spotcheck Verified ─────────────────────────────────
log "2b/14 from_spotcheck Verified -> critic_results.jsonl"
python calibration/from_spotcheck.py \
    --output-dir data/swebench_verified_calibration_full \
    --generators "$GEN" \
    --dataset princeton-nlp/SWE-bench_Verified \
    --max-cost-usd-per-model $SPOTCHECK_COST \
    "${VER_SUBSET_ARG[@]}" \
    2>&1 | tee "$LOG_DIR/02b_from_spotcheck_verified.log"

# ─── Helper: backfill Y into iter_records.jsonl after each eval ───────
backfill() {
    local cell_dir="$1"
    python -c "
import sys
sys.path.insert(0, 'iter')
from swe_backfill_y import backfill_cell
from pathlib import Path
cell_dir = Path('$cell_dir')
print('backfill:', cell_dir, '->', backfill_cell(cell_dir, dry_run=False))
"
}

# ─── Step 3-4b: Self-Refine on Lite ───────────────────────────────────
log "3/14 SR Lite"
python iter/refine_swe.py --method selfrefine \
    --dataset princeton-nlp/SWE-bench_Lite \
    --src-dir data/swebench_lite_calibration_full \
    --output-dir data/swebench_lite_realbaselines_selfrefine_full \
    --generators "$GEN" \
    --n-instances $N_LITE --steps $N_STEPS --max-workers $MAX_REFINE_WORKERS \
    --max-cost-usd-per-model "$GEN=$REFINE_LITE_COST" \
    2>&1 | tee "$LOG_DIR/03_sr_lite.log"

log "4/14 Eval SR Lite"
python iter/eval_steps.py \
    --gen "$GEN" \
    --method selfrefine \
    --data-dir data/swebench_lite_realbaselines_selfrefine_full \
    --dataset princeton-nlp/SWE-bench_Lite \
    --n-steps $N_STEPS \
    2>&1 | tee "$LOG_DIR/04_eval_sr_lite.log"

log "4b/14 backfill Y (SR Lite)"
backfill "data/swebench_lite_realbaselines_selfrefine_full/$GEN/selfrefine" \
    2>&1 | tee "$LOG_DIR/04b_backfill_sr_lite.log"

# ─── Step 5-6b: Self-Refine on Verified ───────────────────────────────
log "5/14 SR Verified"
python iter/refine_swe.py --method selfrefine \
    --dataset princeton-nlp/SWE-bench_Verified \
    --src-dir data/swebench_verified_calibration_full \
    --output-dir data/swebench_verified_realbaselines_selfrefine_full \
    --generators "$GEN" \
    --n-instances $N_VER --steps $N_STEPS --max-workers $MAX_REFINE_WORKERS \
    --max-cost-usd-per-model "$GEN=$REFINE_VER_COST" \
    "${VER_SUBSET_ARG[@]}" \
    2>&1 | tee "$LOG_DIR/05_sr_verified.log"

log "6/14 Eval SR Verified"
python iter/eval_steps.py \
    --gen "$GEN" \
    --method selfrefine \
    --data-dir data/swebench_verified_realbaselines_selfrefine_full \
    --dataset princeton-nlp/SWE-bench_Verified \
    --n-steps $N_STEPS \
    2>&1 | tee "$LOG_DIR/06_eval_sr_verified.log"

log "6b/14 backfill Y (SR Verified)"
backfill "data/swebench_verified_realbaselines_selfrefine_full/$GEN/selfrefine" \
    2>&1 | tee "$LOG_DIR/06b_backfill_sr_verified.log"

# ─── Step 7-8b: Reflexion on Lite ─────────────────────────────────────
log "7/14 Rfx Lite"
python iter/refine_swe.py --method reflexion \
    --dataset princeton-nlp/SWE-bench_Lite \
    --src-dir data/swebench_lite_calibration_full \
    --output-dir data/swebench_lite_realbaselines_reflexion_full \
    --generators "$GEN" \
    --n-instances $N_LITE --steps $N_STEPS --max-workers $MAX_REFINE_WORKERS \
    --max-cost-usd-per-model "$GEN=$REFINE_LITE_COST" \
    2>&1 | tee "$LOG_DIR/07_rfx_lite.log"

log "8/14 Eval Rfx Lite"
python iter/eval_steps.py \
    --gen "$GEN" \
    --method reflexion \
    --data-dir data/swebench_lite_realbaselines_reflexion_full \
    --dataset princeton-nlp/SWE-bench_Lite \
    --n-steps $N_STEPS \
    2>&1 | tee "$LOG_DIR/08_eval_rfx_lite.log"

log "8b/14 backfill Y (Rfx Lite)"
backfill "data/swebench_lite_realbaselines_reflexion_full/$GEN/reflexion" \
    2>&1 | tee "$LOG_DIR/08b_backfill_rfx_lite.log"

# ─── Step 9-10b: Reflexion on Verified ────────────────────────────────
log "9/14 Rfx Verified"
python iter/refine_swe.py --method reflexion \
    --dataset princeton-nlp/SWE-bench_Verified \
    --src-dir data/swebench_verified_calibration_full \
    --output-dir data/swebench_verified_realbaselines_reflexion_full \
    --generators "$GEN" \
    --n-instances $N_VER --steps $N_STEPS --max-workers $MAX_REFINE_WORKERS \
    --max-cost-usd-per-model "$GEN=$REFINE_VER_COST" \
    "${VER_SUBSET_ARG[@]}" \
    2>&1 | tee "$LOG_DIR/09_rfx_verified.log"

log "10/14 Eval Rfx Verified"
python iter/eval_steps.py \
    --gen "$GEN" \
    --method reflexion \
    --data-dir data/swebench_verified_realbaselines_reflexion_full \
    --dataset princeton-nlp/SWE-bench_Verified \
    --n-steps $N_STEPS \
    2>&1 | tee "$LOG_DIR/10_eval_rfx_verified.log"

log "10b/14 backfill Y (Rfx Verified)"
backfill "data/swebench_verified_realbaselines_reflexion_full/$GEN/reflexion" \
    2>&1 | tee "$LOG_DIR/10b_backfill_rfx_verified.log"

log "ALL DONE: $GEN"
