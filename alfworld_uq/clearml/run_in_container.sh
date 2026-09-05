#!/usr/bin/env bash
# In-container wrapper for an ALFWorld run on a ClearML GPU agent.
#
# Runs inside vllm/vllm-openai, which already ships vLLM and torch. This file
# only installs the ALFWorld/agent dependencies, fetches the dataset, stands the
# endpoint up and hands over to experiments/run_alfworld*.py. Everything about
# *what* is run stays in the runner's own flags.
#
# Deliberately does NOT install requirements.txt as-is: its numpy/matplotlib
# pins would fight the image's torch build, and the analysis runs locally on the
# downloaded artifacts anyway.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
PROJECT_DIR="${REPO_DIR}/alfworld_uq"
cd "${PROJECT_DIR}"
echo "[wrapper] project=${PROJECT_DIR}"
command -v python >/dev/null 2>&1 || ln -sf "$(command -v python3)" /usr/local/bin/python

MODEL="${MODEL:-openai/gpt-oss-20b}"
PORT="${VLLM_PORT:-8010}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
VLLM_LOG="${VLLM_LOG:-${PROJECT_DIR}/vllm_serve.log}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
# A 72 GB checkpoint has to be downloaded before the server can answer, which
# takes far longer than loading an already-cached one.
HEALTH_TIMEOUT_STEPS="${HEALTH_TIMEOUT_STEPS:-240}"

POLICY="${POLICY:-smolagents}"
SPLIT="${SPLIT:-valid_seen}"
NUM_EPISODES="${NUM_EPISODES:-100}"
MAX_STEPS="${MAX_STEPS:-30}"
AGENT_MAX_STEPS="${AGENT_MAX_STEPS:-0}"
MAX_GENERATION_TOKENS="${MAX_GENERATION_TOKENS:-2048}"
SMOL_CODE_TAGS="${SMOL_CODE_TAGS:-markdown}"
# A smolagents turn carries a multi-thousand-token prompt; 60s is too tight.
API_TIMEOUT="${API_TIMEOUT:-300}"
EMPTY_RESPONSE_RETRIES="${EMPTY_RESPONSE_RETRIES:-1}"
WORKERS="${WORKERS:-1}"
SEED="${SEED:-0}"
RUN_NAME="${RUN_NAME:-alfworld_${POLICY}}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/${RUN_NAME}}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-/root/.cache/alfworld}"

# ---------------------------------------------------------------- deps
echo "[wrapper] installing ALFWorld + agent deps (vllm/torch stay untouched)"
# The agent mounts the host's /var/cache/apt/archives into the container, and on
# some workers that cache is corrupt: every repository, not just NVIDIA's, then
# fails with "At least one invalid signature was encountered", apt installs
# nothing, and the agent dies with `Cannot find "git" executable` before it can
# clone the repo. Dropping the NVIDIA list, clearing the stale package lists and
# pointing the archive cache at /tmp bypasses the mounted cache entirely.
rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list || true
rm -rf /var/lib/apt/lists/* || true
mkdir -p /tmp/aptcache/partial
apt-get -o Dir::Cache::archives=/tmp/aptcache -o Acquire::AllowInsecureRepositories=true update -qq >/dev/null 2>&1 || true
apt-get -o Dir::Cache::archives=/tmp/aptcache install -y -qq --no-install-recommends --allow-unauthenticated build-essential libffi-dev unzip >/dev/null 2>&1 || true
python -m pip install --no-cache-dir \
  "alfworld==0.4.2" "textworld[pddl]==1.7.0" "openai==2.50.0" \
  "python-dotenv==1.2.2" "smolagents==1.26.0" >/dev/null
# Images without vLLM (used on agents that strip --entrypoint) get it from pip;
# the wheel brings its own torch and CUDA runtime, so only the driver matters.
if ! python -c "import vllm" >/dev/null 2>&1; then
  echo "[wrapper] no vLLM in the image, installing vllm==${VLLM_VERSION:-0.12.0}"
  python -m pip install "vllm==${VLLM_VERSION:-0.12.0}" || {
    echo "[wrapper] FATAL: could not install vLLM"; exit 1; }
fi
python -c "import vllm, torch, smolagents, alfworld; print('[wrapper] vllm', vllm.__version__, 'torch', torch.__version__, 'smolagents', smolagents.__version__)"

# ---------------------------------------------------------------- data
# The dataset arrives as two archives; the second carries initial_state.pddl
# and the pre-generated game.tw-pddl files. A dropped connection still exits 0
# and leaves a tree the environment reports as "0 supported games", so the game
# files are counted and the download retried.
# `find` exits non-zero before the first download because the directory does not
# exist yet, and under `set -euo pipefail` that killed the whole script right
# here -- silently, since the failing command was a command substitution.
count_games() {
  local found
  found=$(find "${ALFWORLD_DATA}/json_2.1.1" -name game.tw-pddl 2>/dev/null | wc -l | tr -d ' ') || found=0
  echo "${found:-0}"
}

echo "[wrapper] ALFWORLD_DATA=${ALFWORLD_DATA}"
games=$(count_games)
for attempt in 1 2 3; do
  if [ "${games}" -gt 1000 ]; then break; fi
  echo "[wrapper] alfworld-download attempt ${attempt} (games so far: ${games})"
  alfworld-download || true
  games=$(count_games)
done
echo "[wrapper] ALFWorld game files: ${games}"
[ "${games}" -gt 1000 ] || { echo "[wrapper] FATAL: dataset incomplete"; exit 1; }

# ---------------------------------------------------------------- serve
# No --served-model-name: the client addresses the model by its HF id.
echo "[wrapper] serving ${MODEL} on :${PORT} (tp=${TENSOR_PARALLEL_SIZE})"
serve_args=(
  --host 127.0.0.1 --port "${PORT}"
  --max-model-len "${MAX_MODEL_LEN}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
)
if [ -n "${GPU_MEMORY_UTILIZATION}" ]; then
  serve_args+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
fi
vllm serve "${MODEL}" "${serve_args[@]}" >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
trap 'echo "[wrapper] stopping vLLM ${VLLM_PID}"; kill ${VLLM_PID} 2>/dev/null || true' EXIT

echo "[wrapper] waiting for /health (up to $((HEALTH_TIMEOUT_STEPS * 5 / 60)) min)"
for i in $(seq 1 "${HEALTH_TIMEOUT_STEPS}"); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[wrapper] vLLM healthy after ${i}x5s"; break
  fi
  if ! kill -0 ${VLLM_PID} 2>/dev/null; then
    echo "[wrapper] FATAL: vLLM died. Last log lines:"; tail -n 300 "${VLLM_LOG}"; exit 1
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || {
  echo "[wrapper] FATAL: health never came up"; tail -n 300 "${VLLM_LOG}"; exit 1; }

# The runner reads the endpoint from .env / the environment. `logprobs` is a
# plain OpenAI parameter here, so none of the OpenRouter provider routing
# applies -- that is the point of serving locally.
export LLM_BASE_URI="http://127.0.0.1:${PORT}/v1"
export LLM_API_KEY="local"
export MODEL_NAME="${MODEL}"

echo "[wrapper] config: policy=${POLICY} model=${MODEL} split=${SPLIT}"
echo "  NUM_EPISODES=${NUM_EPISODES} MAX_STEPS=${MAX_STEPS} AGENT_MAX_STEPS=${AGENT_MAX_STEPS}"
echo "  MAX_GENERATION_TOKENS=${MAX_GENERATION_TOKENS} WORKERS=${WORKERS}"
echo "  RUN_ROOT=${RUN_ROOT}"

common_args=(
  --policy "${POLICY}"
  --num-episodes "${NUM_EPISODES}"
  --max-steps "${MAX_STEPS}"
  --agent-max-steps "${AGENT_MAX_STEPS}"
  --max-generation-tokens "${MAX_GENERATION_TOKENS}"
  --empty-response-retries "${EMPTY_RESPONSE_RETRIES}"
  --smol-code-tags "${SMOL_CODE_TAGS}"
  --api-timeout "${API_TIMEOUT}"
  --split "${SPLIT}"
  --seed "${SEED}"
  --output-dir "${RUN_ROOT}"
  --overwrite
)

if [ "${WORKERS}" -gt 1 ]; then
  python -m experiments.run_alfworld_sharded --workers "${WORKERS}" "${common_args[@]}"
else
  python -m experiments.run_alfworld "${common_args[@]}"
fi

echo "[wrapper] DONE. results under ${RUN_ROOT}"
