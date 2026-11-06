#!/usr/bin/env bash
# Our own decision model: LoRA over a small backbone, a three-answer schema head,
# supervised by the verified outcome, our own calibrated posterior and the
# structural facts ALFWorld labels mechanically. Ablations and the permuted
# control are flags, so each ingredient's contribution is measured.
set -uo pipefail
MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
OUT_DIR="${OUT_DIR:-/tmp/decision_runs}"
VIEWS="${VIEWS:-data/judge_views}"
TARGETS="${TARGETS:-data/distill_targets}"
LORA_RANK="${LORA_RANK:-16}"
EPOCHS="${EPOCHS:-2}"
BATCH="${BATCH:-8}"
LR="${LR:-1e-4}"
W_DISTILL="${W_DISTILL:-1.0}"
W_AUX="${W_AUX:-0.3}"
SHUFFLE="${SHUFFLE:-0}"
TAG="${TAG:-ours}"
HOLDOUT="${HOLDOUT:-}"
DROP="${DROP:-}"
TASK_ONLY="${TASK_ONLY:-0}"
W_ACTION="${W_ACTION:-0}"
W_ENTROPY="${W_ENTROPY:-0.01}"
C_STEP="${C_STEP:-0.02}"
C_ASK="${C_ASK:-0.5}"
DOMAIN_BIAS="${DOMAIN_BIAS:-0}"
SAVE_WEIGHTS="${SAVE_WEIGHTS:-1}"
EVAL_EACH_EPOCH="${EVAL_EACH_EPOCH:-1}"
FOLDS="${FOLDS:-5}"
# a whole 12-cohort partition is 90k optimiser steps, measured at 0.42 a second
# on one 80GB card: sixty hours, and nothing is written until the last fold ends.
# Two folds of the same partition is the same 80/20 split on 40% of the episodes
FOLD_LIMIT="${FOLD_LIMIT:-0}"

mkdir -p "$OUT_DIR"
nvidia-smi || echo "no nvidia-smi"
python -m pip install -q --upgrade pip
python -m pip install -q torch "transformers>=5.10" peft accelerate safetensors numpy

cd alfworld_uq
flag=""
[ "$SHUFFLE" = "1" ] && flag="--shuffle-labels"
[ -n "$HOLDOUT" ] && flag="$flag --holdout $HOLDOUT"
[ -n "$DROP" ] && flag="$flag --drop $DROP"
[ "$TASK_ONLY" = "1" ] && flag="$flag --task-only"
# default on: eight runs were trained before this and every adapter was lost
[ "$SAVE_WEIGHTS" = "1" ] && flag="$flag --save-weights $OUT_DIR/weights"
python -m experiments.train_decision \
    --views "$VIEWS" --targets "$TARGETS" --out "$OUT_DIR" --model "$MODEL" --tag "$TAG" \
    --lora-rank "$LORA_RANK" --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR" \
    --w-action "$W_ACTION" --w-entropy "$W_ENTROPY" --c-step "$C_STEP" --c-ask "$C_ASK" \
    --w-distill "$W_DISTILL" --w-aux "$W_AUX" \
    --domain-bias "$DOMAIN_BIAS" --eval-each-epoch "$EVAL_EACH_EPOCH" \
    --folds "$FOLDS" --fold-limit "$FOLD_LIMIT" $flag
# The trailing diagnostics would otherwise overwrite $? and report a crashed
# run as completed; set -e is not used here because those lines must still run.
rc=$?
echo "[train] done rc=$rc"; ls -la "$OUT_DIR"
exit $rc
