#!/usr/bin/env bash
# Serve a System One decision model on the agent and ask it about every recorded
# trajectory. The states travel with the repo (alfworld_uq/data/decision_states),
# so nothing but the weights is downloaded here.
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-PostHog/jeeves}"
JEEVES_REPO="${JEEVES_REPO:-https://github.com/PostHog/jeeves}"
PORT="${PORT:-8009}"
MAX_LEN="${MAX_LEN:-8192}"
WORKERS="${WORKERS:-4}"
THINK_MODES="${THINK_MODES:-nothink think}"
MAX_THINK="${MAX_THINK:-768}"
OUT_DIR="${OUT_DIR:-/tmp/decision_runs}"
STATES_DIR="${STATES_DIR:-alfworld_uq/data/decision_states}"
SERVE_TIMEOUT="${SERVE_TIMEOUT:-1800}"

mkdir -p "$OUT_DIR"
nvidia-smi || echo "no nvidia-smi"

python -m pip install -q --upgrade pip
python -m pip install -q "huggingface_hub[cli]" requests

echo "[jeeves] downloading $MODEL_REPO"
hf download "$MODEL_REPO" --local-dir "$OUT_DIR/weights"

echo "[jeeves] cloning the inference code"
git clone --depth 1 "$JEEVES_REPO" "$OUT_DIR/jeeves"
python -m pip install -q -r "$OUT_DIR/jeeves/requirements.txt"
python -m pip install -q "$OUT_DIR/jeeves/sdk" || echo "[jeeves] no sdk package; the REST client is enough"

DRAFTER="$OUT_DIR/weights/drafter_k4.safetensors"
SERVE_ARGS=(--model "$OUT_DIR/weights" --port "$PORT" --max-len "$MAX_LEN")
[ -f "$DRAFTER" ] && SERVE_ARGS+=(--drafter "$DRAFTER")

echo "[jeeves] serving: ${SERVE_ARGS[*]}"
( cd "$OUT_DIR/jeeves" && python -m inference.serve "${SERVE_ARGS[@]}" > "$OUT_DIR/serve.log" 2>&1 ) &
SERVE_PID=$!
trap 'kill $SERVE_PID 2>/dev/null || true' EXIT

deadline=$((SECONDS + SERVE_TIMEOUT))
until curl -sf "http://localhost:${PORT}/v1/systemone" -X POST -H 'Content-Type: application/json' \
        -d '{"state":"ping","questions":{"q":{"type":"noul","instructions":"Is this a ping?"}}}' >/dev/null 2>&1; do
    if ! kill -0 $SERVE_PID 2>/dev/null; then
        echo "[jeeves] the server died"; tail -40 "$OUT_DIR/serve.log"; exit 1
    fi
    [ $SECONDS -gt $deadline ] && { echo "[jeeves] server did not come up"; tail -40 "$OUT_DIR/serve.log"; exit 1; }
    sleep 10
done
echo "[jeeves] server is up after ${SECONDS}s"

cd alfworld_uq
for mode in $THINK_MODES; do
    flag=""
    [ "$mode" = "think" ] && flag="--think"
    echo "[jeeves] asking in mode: $mode"
    python -m experiments.decision_model_query \
        --states ../"$STATES_DIR"/*.jsonl \
        --out "$OUT_DIR/answers_${mode}.jsonl" \
        --url "http://localhost:${PORT}/v1/systemone" \
        --workers "$WORKERS" --max-think "$MAX_THINK" $flag
done
echo "[jeeves] done"
ls -la "$OUT_DIR"
