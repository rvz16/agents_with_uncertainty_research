#!/usr/bin/env bash
# In-container wrapper for a CLEAN code-UQ run on a ClearML GPU agent.
# Runs inside the vllm/vllm-openai:v0.12.0 docker image (vLLM + torch preinstalled).
#
# What it does:
#   1. install our package deps (NOT vllm/torch — image already has them)
#   2. serve gpt-oss-20b via the image's vLLM on 127.0.0.1:8010
#   3. wait for /health
#   4. run scripts/run_code_uq.sh with CLEAN, no-leak flags
#      (terminal single verify, no private-test cap, 32k gen tokens from the fixed agent default)
#
# All extra UQ analysis (experiment2 sep/lr_neg/double, entropy_kl, ntokens,
# belief_logit, multi-critic) is run LOCALLY afterwards on the downloaded artifacts.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
cd "${REPO_DIR}"
echo "[wrapper] repo=${REPO_DIR}"
command -v python >/dev/null 2>&1 || ln -sf "$(command -v python3)" /usr/local/bin/python

MODEL="${MODEL:-openai/gpt-oss-20b}"
PORT="${VLLM_PORT:-8010}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
VLLM_LOG="${VLLM_LOG:-${REPO_DIR}/vllm_serve.log}"

# ---------------------------------------------------------------- deps
echo "[wrapper] installing package deps (excluding vllm/torch)"
python -m pip install --no-cache-dir -e ".[code]" >/dev/null
# code extras minus torch/vllm (image ships them); ignore transient resolver noise
python -m pip install --no-cache-dir datasets docker evalplus >/dev/null || true
python -c "import vllm, torch; print('[wrapper] vllm', vllm.__version__, 'torch', torch.__version__)"

# ---------------------------------------------------------------- serve
echo "[wrapper] starting vLLM: ${MODEL} on :${PORT}"
vllm serve "${MODEL}" \
  --host 127.0.0.1 --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
trap 'echo "[wrapper] stopping vLLM ${VLLM_PID}"; kill ${VLLM_PID} 2>/dev/null || true' EXIT

echo "[wrapper] waiting for /health (up to 20 min)"
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[wrapper] vLLM healthy after ${i}x5s"; break
  fi
  if ! kill -0 ${VLLM_PID} 2>/dev/null; then
    echo "[wrapper] FATAL: vLLM died. Last log lines:"; tail -n 60 "${VLLM_LOG}"; exit 1
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || { echo "[wrapper] FATAL: health never came up"; tail -n 60 "${VLLM_LOG}"; exit 1; }

# ---------------------------------------------------------------- run (CLEAN flags)
export GENERATOR_KEY="gpt_oss_20b_local"
export GPT_OSS_20B_BASE_URL="http://127.0.0.1:${PORT}/v1"

export BENCHMARKS="${BENCHMARKS:-lcb_hard}"
export N_INSTANCES="${N_INSTANCES:-0}"
export MAX_VERIFICATIONS="${MAX_VERIFICATIONS:-1}"   # terminal single final verify => NO oracle leak
export PRIVATE_TEST_CAP="${PRIVATE_TEST_CAP:-0}"     # use ALL private tests => real balance (student PR §9)
export MAX_GENERATIONS="${MAX_GENERATIONS:-5}"
export MAX_STEPS="${MAX_STEPS:-20}"
export FINAL_VERIFY="${FINAL_VERIFY:-0}"
export SAVE_VERBALIZED_2S="${SAVE_VERBALIZED_2S:-1}"
export RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
export RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/runs/code_uq_clean/gpt_oss_20b_local}"
export RESUME="${RESUME:-1}"

echo "[wrapper] CLEAN config:"
echo "  BENCHMARKS=${BENCHMARKS} N_INSTANCES=${N_INSTANCES}"
echo "  MAX_VERIFICATIONS=${MAX_VERIFICATIONS} PRIVATE_TEST_CAP=${PRIVATE_TEST_CAP} MAX_GENERATIONS=${MAX_GENERATIONS}"
echo "  RUN_ROOT=${RUN_ROOT}"

bash scripts/run_code_uq.sh

echo "[wrapper] DONE. results under ${RUN_ROOT}"
