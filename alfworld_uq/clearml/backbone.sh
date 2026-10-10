#!/usr/bin/env bash
# The backbone's own confidence, for the ablation against the decision model
# built on it: Qwen3.5-9B under clef-flash, Qwen3.8-27B under clef.
set -euo pipefail

MODELS="${MODELS:-Qwen/Qwen3.5-9B Qwen/Qwen3.8-27B}"
MODES="${MODES:-token verbal}"
OUT_DIR="${OUT_DIR:-/tmp/backbone_runs}"
STATES_DIR="${STATES_DIR:-alfworld_uq/data/decision_states_clean}"

mkdir -p "$OUT_DIR"
nvidia-smi || echo "no nvidia-smi"
python -m pip install -q --upgrade pip
python -m pip install -q torch "transformers>=5.10" huggingface_hub accelerate safetensors

cd alfworld_uq
for model in $MODELS; do
    tag="$(echo "$model" | tr '/.' '__')"
    for mode in $MODES; do
        echo "[backbone] $model ($mode)"
        python -m experiments.backbone_confidence \
            --states ../"$STATES_DIR"/*.jsonl \
            --out "$OUT_DIR/answers_${tag}_${mode}.jsonl" \
            --model "$model" --mode "$mode" || echo "[backbone] $model $mode failed, continuing"
    done
done
echo "[backbone] done"
ls -la "$OUT_DIR"
