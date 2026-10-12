#!/usr/bin/env bash
# Decision 3.0 across its released sizes, with each backbone measured raw and
# with a head we train ourselves. Three points per size is what makes the
# head-versus-backbone question a trend rather than one release's quirk.
set -uo pipefail

D3_MODELS="${D3_MODELS:-vllm-sr/d3-edge vllm-sr/d3-lite vllm-sr/d3-nano vllm-sr/d3-mini}"
BACKBONES="${BACKBONES:-Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B}"
OUT_DIR="${OUT_DIR:-/tmp/d3_runs}"
STATES_DIR="${STATES_DIR:-data/decision_states_clean}"
VIEWS_DIR="${VIEWS_DIR:-data/judge_views}"
EPOCHS="${EPOCHS:-40}"
STAGES="${STAGES:-decision backbone head}"
OOD="${OOD:-0}"
OOD_FLAG=""
[ "$OOD" = "1" ] && OOD_FLAG="--ood"

mkdir -p "$OUT_DIR"
nvidia-smi || echo "no nvidia-smi"
python -m pip install -q --upgrade pip
# d3 asks for transformers 5.17 and needs torchvision for its vision tower
python -m pip install -q torch torchvision "transformers==5.17.0" huggingface_hub accelerate \
    safetensors pillow opencv-python-headless scikit-learn numpy

cd alfworld_uq
STATES=$(ls "$STATES_DIR"/*.jsonl)

for stage in $STAGES; do
case "$stage" in
  decision)
    for m in $D3_MODELS; do
      echo "=== decision $m ==="
      python -m experiments.decision_model_d3 --states $STATES --out "$OUT_DIR" \
          --model "$m" --tag clean || echo "[sweep] $m failed, continuing"
    done ;;
  backbone)
    for b in $BACKBONES; do
      echo "=== backbone $b ==="
      safe=$(echo "$b" | tr '/.' '__')
      python -m experiments.backbone_confidence --states $STATES \
          --out "$OUT_DIR/answers_${safe}_token.jsonl" --model "$b" --mode token \
          || echo "[sweep] $b failed, continuing"
    done ;;
  head)
    for b in $BACKBONES; do
      echo "=== trained head on $b ==="
      python -m experiments.schema_head --states "$STATES_DIR" --views "$VIEWS_DIR" \
          --out "$OUT_DIR" --model "$b" --epochs "$EPOCHS" --kinds logreg schema $OOD_FLAG \
          || echo "[sweep] head on $b failed, continuing"
    done ;;
esac
done
echo "[sweep] done"
ls -la "$OUT_DIR"
