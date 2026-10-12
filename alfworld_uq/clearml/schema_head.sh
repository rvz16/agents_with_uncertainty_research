#!/usr/bin/env bash
# Train a schema head ourselves on a frozen backbone: the third point the
# decision-model ablation is missing. Features come from one no-gradient pass
# over the backbone and are cached, so the heads are cheap to retrain.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
OUT_DIR="${OUT_DIR:-/tmp/schema_runs}"
STATES_DIR="${STATES_DIR:-data/decision_states_clean}"
VIEWS_DIR="${VIEWS_DIR:-data/judge_views}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
FEATURE_BATCH="${FEATURE_BATCH:-4}"
EPOCHS="${EPOCHS:-40}"
KINDS="${KINDS:-logreg mlp schema}"
AUX="${AUX:-1}"

mkdir -p "$OUT_DIR"
nvidia-smi || echo "no nvidia-smi"
python -m pip install -q --upgrade pip
python -m pip install -q torch "transformers>=5.10" huggingface_hub accelerate safetensors scikit-learn numpy

cd alfworld_uq
# the plain heads first, so a failure in the auxiliary variant still leaves a result
python -m experiments.schema_head \
    --states "$STATES_DIR" --views "$VIEWS_DIR" --out "$OUT_DIR" --model "$MODEL" \
    --max-length "$MAX_LENGTH" --feature-batch "$FEATURE_BATCH" --epochs "$EPOCHS" --kinds $KINDS
if [ "$AUX" = "1" ]; then
    python -m experiments.schema_head \
        --states "$STATES_DIR" --views "$VIEWS_DIR" --out "$OUT_DIR" --model "$MODEL" \
        --max-length "$MAX_LENGTH" --epochs "$EPOCHS" --kinds mlp schema --aux
fi
echo "[schema] done"
ls -la "$OUT_DIR"
