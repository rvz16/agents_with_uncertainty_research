#!/usr/bin/env bash
# Re-score recorded DeepSWE actions to recover the per-step uncertainty signals.
# Inference only: one teacher-forced forward pass per step, no environment and no
# agent loop, so this needs a GPU and nothing else.
set -uo pipefail
OUT_DIR="${OUT_DIR:-/tmp/signal_runs}"
COHORT="${COHORT:-DG}"
MODEL="${MODEL:-openai/gpt-oss-20b}"
TOP_K="${TOP_K:-20}"
WINDOW="${WINDOW:-10}"
OBS_CHARS="${OBS_CHARS:-400}"
LIMIT="${LIMIT:-}"
# this image ships mkl-service alongside libgomp and already exports
# MKL_THREADING_LAYER=INTEL, so torch aborts at import; a :- default never applied
# because the variable was set. libgomp is the one present, so name GNU outright.
export MKL_THREADING_LAYER=GNU

mkdir -p "$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[score] no nvidia-smi"
cd deep_swe_uq
python -m experiments.score_signals \
    --inputs "data/replay_inputs/${COHORT}.jsonl.gz" \
    --views "../alfworld_uq/data/judge_views/${COHORT}.jsonl" \
    --model "$MODEL" --out "${OUT_DIR}/compact_${COHORT}.jsonl" \
    --top-k "$TOP_K" --window "$WINDOW" --obs-chars "$OBS_CHARS" \
    ${LIMIT:+--limit "$LIMIT"}
rc=$?
echo "[score] done rc=$rc"; ls -la "$OUT_DIR"
exit $rc
