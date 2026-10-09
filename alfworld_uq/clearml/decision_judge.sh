#!/usr/bin/env bash
# A decision model as the per-step judge. The views travel with the repo: one
# JSON object per episode with the task and the action/result of every scored
# step, under the giveup+acted rule, so the prefixes match what the tables score.
set -euo pipefail

BACKEND="${BACKEND:-liquid}"
MODEL="${MODEL:-LiquidAI/d1-omni-600M}"
OUT_DIR="${OUT_DIR:-/tmp/decision_judge}"
VIEWS_DIR="${VIEWS_DIR:-alfworld_uq/data/judge_views}"
BATCH="${BATCH:-16}"
WINDOW="${WINDOW:-10}"

mkdir -p "$OUT_DIR"
nvidia-smi || echo "no nvidia-smi"
python -m pip install -q --upgrade pip
python -m pip install -q torch torchvision "transformers>=5.10" huggingface_hub accelerate safetensors pillow

# the answer file carries the model's name: this directory is mounted from the
# host and a previous model's verdicts here were re-uploaded as a new model's
TAG="$(echo "$MODEL" | tr '/.' '__')"
cd alfworld_uq
for view in ../"$VIEWS_DIR"/*.jsonl; do
    cohort="$(basename "$view" .jsonl)"
    echo "[judge] $cohort with $MODEL"
    python -m experiments.prefix_decision \
        --view "$view" --out "$OUT_DIR/judge_${TAG}_${cohort}.jsonl" \
        --backend "$BACKEND" --model "$MODEL" --batch "$BATCH" --window "$WINDOW" \
        || echo "[judge] $cohort failed, continuing"
done
echo "[judge] done"
ls -la "$OUT_DIR"
