#!/usr/bin/env bash
# Verified-only SWE-Bench expansion for ONE generator, scoped to a missing-IDs
# subset, into fresh *_exp dirs. Leaves any existing *_full data untouched;
# you merge the _exp cells into *_full offline afterwards (see
# README_verified_expansion.md). This is the incremental complement to
# run_swebench_pipeline.sh (which runs Lite+Verified end-to-end for a fixed
# subset and would re-run/clobber finished Lite baselines if reused here).
#
# Usage:
#   run_verified_expand.sh <generator> <subset_ids.json> [cal_cap refine_cap spotcheck_cap]
#
# Examples (from experiments/orchestration_hypothesis_testing/):
#   scripts_pipeline/run_verified_expand.sh sonnet45    data/swebench_verified_calibration_full/verified_missing_289.json
#   scripts_pipeline/run_verified_expand.sh qwen3_coder data/swebench_verified_calibration_full/verified_missing_289.json
#   scripts_pipeline/run_verified_expand.sh haiku45     data/swebench_verified_calibration_full/verified_missing_289.json
#   scripts_pipeline/run_verified_expand.sh gpt5_mini   data/swebench_verified_calibration_full/verified_gpt5_mini_missing_489.json
#
# Prereqs on the host (same as run_swebench_pipeline.sh):
#   - .env at repo root with a FUNDED OPENROUTER_API_KEY
#   - rootless podman socket + patched swebench harness (patch_swebench_harness.py)
#   - enough /mnt/data quota headroom (~5-10 GB per generator)
#
# Re-run semantics: Cal resumes (spot_check skips already-generated patches),
# critics are recomputed (from_spotcheck overwrites), and the SR/Rfx _exp cells
# are cleared and rebuilt (refine_swe appends to iter_records, so a re-run would
# otherwise duplicate rows).
#
# Cost caps default per generator (override via args or CAL_CAP/REFINE_CAP/
# SPOTCHECK_CAP env). They are ceilings; set them high enough to avoid
# cost_cap early-stops that would leave partial trajectories.
set -euo pipefail

GEN="${1:?usage: $0 <generator> <subset_ids.json> [cal_cap refine_cap spotcheck_cap]}"
SUBSET_IN="${2:?subset ids json (flat list or {\"instance_ids\":[...]})}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO="$(cd "${PIPE_DIR}/../.." && pwd)"
cd "$PIPE_DIR"

case "$SUBSET_IN" in /*) SUBSET="$SUBSET_IN";; *) SUBSET="$PWD/$SUBSET_IN";; esac
[ -f "$SUBSET" ] || { echo "ERROR: subset not found: $SUBSET" >&2; exit 1; }

# Per-generator default caps (USD). Sonnet is ~8x qwen/haiku/gpt5-mini per token.
case "$GEN" in
    sonnet45)    DCAL=50; DREF=25; DSPOT=25;;
    haiku45)     DCAL=25; DREF=15; DSPOT=15;;
    qwen3_coder) DCAL=20; DREF=12; DSPOT=12;;
    gpt5_mini)   DCAL=20; DREF=12; DSPOT=15;;   # cheap model, but 489 instances
    *)           DCAL=30; DREF=15; DSPOT=15;;
esac
CAL_CAP="${3:-${CAL_CAP:-$DCAL}}"
REFINE_CAP="${4:-${REFINE_CAP:-$DREF}}"
SPOTCHECK_CAP="${5:-${SPOTCHECK_CAP:-$DSPOT}}"
MAX_REFINE_WORKERS="${MAX_REFINE_WORKERS:-8}"
N_PATCHES=3; N_STEPS=5

LOG_DIR="$REPO/logs/swebench_verified_expand_${GEN}"; mkdir -p "$LOG_DIR"
CAL_DIR=data/swebench_verified_calibration_exp
SR_DIR=data/swebench_verified_realbaselines_selfrefine_exp
RFX_DIR=data/swebench_verified_realbaselines_reflexion_exp

log() { printf "==== %s | %s ====\n" "$(date '+%F %T')" "$1"; }
warn() { printf "==== %s | WARN: %s ====\n" "$(date '+%F %T')" "$1" >&2; }

# Env — identical to run_swebench_pipeline.sh.
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate base
fi
if [ -f "$REPO/.env" ]; then set -a; . "$REPO/.env"; set +a; fi
export DOCKER_HOST="unix:///run/user/$(id -u)/podman/podman.sock"
export SWEBENCH_PODMAN_COMPAT=1
export SWEBENCH_NAMESPACE=none
SCRATCH_DIR="${BUILDAH_SCRATCH_DIR:-/mnt/data/users/$(whoami)/buildah-tmp}"; mkdir -p "$SCRATCH_DIR"
export TMPDIR="$SCRATCH_DIR"; export BUILDAH_TMPDIR="$SCRATCH_DIR"

# Subset size — computed AFTER conda so `python` is on PATH (tmux non-interactive
# shells don't inherit it). Abort on an empty/invalid subset.
N=$(python - "$SUBSET" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ids = d if isinstance(d, list) else d["instance_ids"]
print(len(set(ids)))
PY
)
[ "${N:-0}" -gt 0 ] 2>/dev/null || { echo "FATAL: subset '$SUBSET' has 0 usable ids" >&2; exit 1; }

log "START verified-expand $GEN  n=$N  caps(cal=$CAL_CAP refine=$REFINE_CAP spot=$SPOTCHECK_CAP)  workers=$MAX_REFINE_WORKERS  subset=$SUBSET"

backfill() { python -c "import sys; sys.path.insert(0,'iter'); from swe_backfill_y import backfill_cell; from pathlib import Path; print('backfill:', backfill_cell(Path('$1'), dry_run=False))"; }

# ── Cal: generate base patches (resumable; skips already-generated) ──
log "Cal ($GEN +$N, cap \$$CAL_CAP)"
python scripts/spot_check_generators.py --dataset princeton-nlp/SWE-bench_Verified \
    --n-instances "$N" --n-patches $N_PATCHES --generators "$GEN" \
    --output-dir "$CAL_DIR" --instance-ids-file "$SUBSET" \
    --max-cost-usd-per-model "$GEN=$CAL_CAP" 2>&1 | tee "$LOG_DIR/02_cal.log"
# Hard gate: no base predictions => generation failed (e.g. API key/credit).
# Abort loudly instead of cascading empty data through the rest of the pipeline.
[ -s "$CAL_DIR/$GEN/predictions_p0.jsonl" ] || {
    echo "FATAL: Cal produced no predictions for $GEN — check OPENROUTER_API_KEY credit/limit and $LOG_DIR/02_cal.log" >&2; exit 1; }

log "Critics (from_spotcheck, cap \$$SPOTCHECK_CAP)"
python calibration/from_spotcheck.py --output-dir "$CAL_DIR" --generators "$GEN" \
    --dataset princeton-nlp/SWE-bench_Verified --instance-ids-file "$SUBSET" \
    --max-cost-usd-per-model "$SPOTCHECK_CAP" 2>&1 | tee "$LOG_DIR/02b_critics.log"

# ── Self-Refine: clear stale exp cell first (refine_swe appends) ──
log "SR refine (workers=$MAX_REFINE_WORKERS, cap \$$REFINE_CAP)"
if [ -d "$SR_DIR/$GEN/selfrefine" ]; then warn "clearing stale SR exp cell $SR_DIR/$GEN/selfrefine"; rm -rf "$SR_DIR/$GEN/selfrefine"; fi
python iter/refine_swe.py --method selfrefine --dataset princeton-nlp/SWE-bench_Verified \
    --src-dir "$CAL_DIR" --output-dir "$SR_DIR" --generators "$GEN" \
    --n-instances "$N" --steps $N_STEPS --max-workers "$MAX_REFINE_WORKERS" \
    --instance-ids-file "$SUBSET" --max-cost-usd-per-model "$GEN=$REFINE_CAP" \
    2>&1 | tee "$LOG_DIR/05_sr.log"
log "Eval SR"
python iter/eval_steps.py --gen "$GEN" --method selfrefine --data-dir "$SR_DIR" \
    --dataset princeton-nlp/SWE-bench_Verified --n-steps $N_STEPS \
    2>&1 | tee "$LOG_DIR/06_eval_sr.log" || warn "SR eval returned nonzero (continuing)"
backfill "$SR_DIR/$GEN/selfrefine" 2>&1 | tee "$LOG_DIR/06b_backfill_sr.log" || warn "SR backfill failed"

# ── Reflexion: clear stale exp cell first ──
log "Rfx refine (workers=$MAX_REFINE_WORKERS, cap \$$REFINE_CAP)"
if [ -d "$RFX_DIR/$GEN/reflexion" ]; then warn "clearing stale Rfx exp cell $RFX_DIR/$GEN/reflexion"; rm -rf "$RFX_DIR/$GEN/reflexion"; fi
python iter/refine_swe.py --method reflexion --dataset princeton-nlp/SWE-bench_Verified \
    --src-dir "$CAL_DIR" --output-dir "$RFX_DIR" --generators "$GEN" \
    --n-instances "$N" --steps $N_STEPS --max-workers "$MAX_REFINE_WORKERS" \
    --instance-ids-file "$SUBSET" --max-cost-usd-per-model "$GEN=$REFINE_CAP" \
    2>&1 | tee "$LOG_DIR/09_rfx.log"
log "Eval Rfx"
python iter/eval_steps.py --gen "$GEN" --method reflexion --data-dir "$RFX_DIR" \
    --dataset princeton-nlp/SWE-bench_Verified --n-steps $N_STEPS \
    2>&1 | tee "$LOG_DIR/10_eval_rfx.log" || warn "Rfx eval returned nonzero (continuing)"
backfill "$RFX_DIR/$GEN/reflexion" 2>&1 | tee "$LOG_DIR/10b_backfill_rfx.log" || warn "Rfx backfill failed"

log "DONE $GEN  cal=$(wc -l < "$CAL_DIR/$GEN/predictions_p0.jsonl" 2>/dev/null)  SR=$(wc -l < "$SR_DIR/$GEN/selfrefine/iter_records.jsonl" 2>/dev/null)  Rfx=$(wc -l < "$RFX_DIR/$GEN/reflexion/iter_records.jsonl" 2>/dev/null)"
