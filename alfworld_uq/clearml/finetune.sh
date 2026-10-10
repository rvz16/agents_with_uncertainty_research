#!/usr/bin/env bash
# Fine-tune a small encoder on trajectory prefixes: how much of a decision
# model's advantage is the trained head, and how much a transformer reading text.
set -euo pipefail

MODEL="${MODEL:-answerdotai/ModernBERT-base}"
OUT_DIR="${OUT_DIR:-/tmp/finetune_runs}"
VIEWS_DIR="${VIEWS_DIR:-data/judge_views}"
STATES_DIR="${STATES_DIR:-data/decision_states_clean}"
EPOCHS="${EPOCHS:-2}"
BATCH="${BATCH:-16}"
MAX_LENGTH="${MAX_LENGTH:-1536}"
OOD="${OOD:-1}"

mkdir -p "$OUT_DIR"
nvidia-smi || echo "no nvidia-smi"
python -m pip install -q --upgrade pip
python -m pip install -q torch "transformers>=5.10" huggingface_hub accelerate safetensors scikit-learn

cd alfworld_uq
flag=""
[ "$OOD" = "1" ] && flag="--ood"
python -m experiments.finetune_outcome \
    --views "$VIEWS_DIR" --states "$STATES_DIR" --out "$OUT_DIR" \
    --model "$MODEL" --epochs "$EPOCHS" --batch "$BATCH" --max-length "$MAX_LENGTH" $flag
echo "[finetune] done"
ls -la "$OUT_DIR"
