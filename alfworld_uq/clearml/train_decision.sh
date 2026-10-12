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

mkdir -p "$OUT_DIR"
nvidia-smi || echo "no nvidia-smi"
python -m pip install -q --upgrade pip
python -m pip install -q torch "transformers>=5.10" peft accelerate safetensors numpy

cd alfworld_uq
flag=""
[ "$SHUFFLE" = "1" ] && flag="--shuffle-labels"
python -m experiments.train_decision \
    --views "$VIEWS" --targets "$TARGETS" --out "$OUT_DIR" --model "$MODEL" --tag "$TAG" \
    --lora-rank "$LORA_RANK" --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR" \
    --w-distill "$W_DISTILL" --w-aux "$W_AUX" $flag
echo "[train] done"; ls -la "$OUT_DIR"
