#!/usr/bin/env bash
# Run Cloudflare's Clef over the recorded trajectories. Clef ships no server:
# the model and its joint_schema_model.py come from the Hub and run in process.
set -euo pipefail

MODELS="${MODELS:-Cloudflare/clef-flash Cloudflare/clef}"
OUT_DIR="${OUT_DIR:-/tmp/decision_runs}"
STATES_DIR="${STATES_DIR:-alfworld_uq/data/decision_states_clean}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
TORCH_SPEC="${TORCH_SPEC:-torch}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers>=5.10}"

mkdir -p "$OUT_DIR"
nvidia-smi || echo "no nvidia-smi"

python -m pip install -q --upgrade pip
python -m pip install -q "$TORCH_SPEC" "$TRANSFORMERS_SPEC" "huggingface_hub[cli]" accelerate safetensors pillow

cd alfworld_uq
for model in $MODELS; do
    tag="$(echo "$model" | tr '/' '_')"
    echo "[clef] $model"
    python -m experiments.decision_model_clef \
        --states ../"$STATES_DIR"/*.jsonl \
        --out "$OUT_DIR/answers_${tag}.jsonl" \
        --model "$model" --max-length "$MAX_LENGTH" || echo "[clef] $model failed, continuing"
done
echo "[clef] done"
ls -la "$OUT_DIR"
