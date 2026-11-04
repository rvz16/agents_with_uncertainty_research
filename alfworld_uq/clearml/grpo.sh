#!/usr/bin/env bash
# GRPO over the confidence head. The supervised recipe trains the number; this
# trains the decision the number causes, which is the part a threshold makes
# non-differentiable. Three arms: supervised alone (train_decision.sh on the
# same cohorts), supervised then this with INIT_ADAPTER set, and this alone.
set -uo pipefail
MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
OUT_DIR="${OUT_DIR:-/tmp/decision_runs}"
VIEWS="${VIEWS:-data/judge_views}"
TARGETS="${TARGETS:-data/distill_targets}"
KEEP="${KEEP:-DG,DQ}"
INIT_ADAPTER="${INIT_ADAPTER:-}"
LORA_RANK="${LORA_RANK:-16}"
EPOCHS="${EPOCHS:-2}"
BATCH="${BATCH:-8}"
LR="${LR:-5e-5}"
GROUP="${GROUP:-8}"
WARMUP="${WARMUP:-100}"
W_ENTROPY="${W_ENTROPY:-0.01}"
W_CAL="${W_CAL:-0.3}"
KAPPA="${KAPPA:-0.3}"
C_ASK="${C_ASK:-0.5}"
TAU_STOP="${TAU_STOP:-0.0}"
TAU_ASK="${TAU_ASK:-0.5}"
FOLDS="${FOLDS:-5}"
FOLD_LIMIT="${FOLD_LIMIT:-0}"
TAG="${TAG:-grpo}"
SAVE_WEIGHTS="${SAVE_WEIGHTS:-1}"
EVAL_EACH_EPOCH="${EVAL_EACH_EPOCH:-1}"

mkdir -p "$OUT_DIR"
nvidia-smi || echo "no nvidia-smi"
python -m pip install -q --upgrade pip
python -m pip install -q torch "transformers>=5.10" peft accelerate safetensors numpy

cd alfworld_uq
flag=""
[ -n "$INIT_ADAPTER" ] && flag="$flag --init-adapter $INIT_ADAPTER"
[ "$SAVE_WEIGHTS" = "1" ] && flag="$flag --save-weights $OUT_DIR/weights"
python -m experiments.grpo_decision \
    --views "$VIEWS" --targets "$TARGETS" --out "$OUT_DIR" --model "$MODEL" --tag "$TAG" \
    --keep "$KEEP" --lora-rank "$LORA_RANK" --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR" \
    --group "$GROUP" --warmup "$WARMUP" --w-entropy "$W_ENTROPY" --w-cal "$W_CAL" \
    --kappa "$KAPPA" --c-ask "$C_ASK" --tau-stop "$TAU_STOP" --tau-ask "$TAU_ASK" \
    --folds "$FOLDS" --fold-limit "$FOLD_LIMIT" --eval-each-epoch "$EVAL_EACH_EPOCH" $flag
echo "[grpo] done"; ls -la "$OUT_DIR"
