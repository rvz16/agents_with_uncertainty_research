#!/usr/bin/env bash
# In-container wrapper: serve the model, then run the shared pipeline.
#
# Runs inside vllm/vllm-openai, which already ships vLLM and torch — building
# vLLM from source on the agent is what this whole path exists to avoid.
#
# Everything about *what* is run lives in run_sage_uncertainty_experiments.sh
# and JOINT_RUN_CONFIG.md. This file only stands the endpoint up and hands over.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
cd "${REPO_DIR}"
echo "[wrapper] repo=${REPO_DIR}"

# The image has python3 but no `python`; the pipeline calls `python`.
command -v python >/dev/null 2>&1 || ln -sf "$(command -v python3)" /usr/local/bin/python

MODEL="${MODEL:-openai/gpt-oss-20b}"
PORT="${VLLM_PORT:-8010}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
VLLM_LOG="${VLLM_LOG:-${REPO_DIR}/vllm_serve.log}"

# Preflight the reviewer before spending ~8 min loading the model. A dead or
# restricted key surfaces only as `critic_L3 0/N success_rate=0.000` hours later,
# and httpx logs the status line without the body, so the reason is invisible.
# The body is printed here; the key never is.
if [ -n "${OPENROUTER_API_KEY:-}" ]; then
  echo "[wrapper] preflight: OpenRouter reviewer (${L3_REVIEW_MODEL:-anthropic/claude-haiku-4.5})"
  code=$(curl -s -o /tmp/l3_preflight.json -w '%{http_code}' \
    https://openrouter.ai/api/v1/chat/completions \
    -H "Authorization: Bearer ${OPENROUTER_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${L3_REVIEW_MODEL:-anthropic/claude-haiku-4.5}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":8}" || echo 000)
  echo "[wrapper] preflight HTTP ${code}"
  if [ "${code}" != "200" ]; then
    echo "[wrapper] preflight body:"; head -c 800 /tmp/l3_preflight.json; echo
    echo "[wrapper] WARNING: the L3 critic will return None for every candidate."
    echo "[wrapper]          Set CALIBRATE_L3=0 to let the analysis finish without it."
  fi
else
  echo "[wrapper] OPENROUTER_API_KEY unset -> L3 disabled"
fi

echo "[wrapper] installing deps (vllm/torch already in the image)"
python -m pip install --no-cache-dir -e '.[langgraph,openrouter]' >/dev/null
python -m pip install --no-cache-dir \
  -r experiments/orchestration_hypothesis_testing/scripts/requirements-sage-uncertainty.txt >/dev/null
python -m pip install --no-cache-dir ruff >/dev/null   # the L1 critic shells out to it
python -c "import vllm, torch; print('[wrapper] vllm', vllm.__version__, 'torch', torch.__version__)"

# No --served-model-name: the client addresses the model by its HF id, and
# renaming it here makes every request 404.
echo "[wrapper] serving ${MODEL} on :${PORT}"
vllm serve "${MODEL}" --host 127.0.0.1 --port "${PORT}" --max-model-len "${MAX_MODEL_LEN}" \
  >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
trap 'echo "[wrapper] stopping vLLM ${VLLM_PID}"; kill ${VLLM_PID} 2>/dev/null || true' EXIT

echo "[wrapper] waiting for /health (up to 20 min)"
for _ in $(seq 1 240); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
  if ! kill -0 ${VLLM_PID} 2>/dev/null; then
    echo "[wrapper] FATAL: vLLM died:"; tail -n 60 "${VLLM_LOG}"; exit 1
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || {
  echo "[wrapper] FATAL: health never came up"; tail -n 60 "${VLLM_LOG}"; exit 1; }
echo "[wrapper] vLLM healthy"

export GENERATOR_KEY="${GENERATOR_KEY:-gpt_oss_20b_local}"
export GPT_OSS_20B_BASE_URL="http://127.0.0.1:${PORT}/v1"
export RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/runs/sage_uq/${GENERATOR_KEY}}"

# Defaults come from run_sage_uncertainty_experiments.sh, which is the single
# source of truth for the shared config. Only the two knobs a caller normally
# varies are surfaced here.
export BENCHMARKS="${BENCHMARKS:-lcb_hard}"
export N_INSTANCES="${N_INSTANCES:-0}"
# The analysis step refits the critic likelihoods on the train split and needs
# L3 verdicts there, which the generation pass does not produce on its own:
#   RuntimeError: missing saved L3 train-calibration results for N instances
export CALIBRATE_L3="${CALIBRATE_L3:-1}"

echo "[wrapper] BENCHMARKS=${BENCHMARKS} N_INSTANCES=${N_INSTANCES}"
echo "[wrapper] MAX_VERIFICATIONS=${MAX_VERIFICATIONS:-<script default: 0>}"
echo "[wrapper] RUN_ROOT=${RUN_ROOT}"

bash experiments/orchestration_hypothesis_testing/scripts/run_sage_uncertainty_experiments.sh
echo "[wrapper] DONE -> ${RUN_ROOT}"
