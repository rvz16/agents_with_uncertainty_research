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
# Two tasks on one worker share the host network, so a fixed port makes the
# second one adopt the first one's server: its health check passes, and the
# failure only surfaces later as a connection error from somewhere else. Take a
# free port unless one was named.
if [ -z "${VLLM_PORT:-}" ]; then
  PORT=$(python - <<'PYPORT'
import socket
sock = socket.socket()
sock.bind(("127.0.0.1", 0))
print(sock.getsockname()[1])
sock.close()
PYPORT
)
else
  PORT="${VLLM_PORT}"
fi
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
VLLM_LOG="${VLLM_LOG:-${PROJECT_DIR}/vllm_serve.log}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
# vLLM takes 90% of the card unless told otherwise, which leaves no room for a
# reviewer on the same GPU. When one is asked for, the agent's share is capped
# so that both fit, unless a share was set explicitly.
AGENT_GPU_FRACTION_WITH_JUDGE="${AGENT_GPU_FRACTION_WITH_JUDGE:-0.60}"
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
# On vLLM 0.28 gpt-oss returns its whole generation as reasoning and leaves the
# visible answer empty: 2637 of 2728 smolagents generations came back blank,
# the framework read that as nothing to do and called final_answer in 105 of
# 140 episodes. The code-tag stop sequence firing inside the hidden channel is
# the known cause of an empty answer here, so it has to be switchable.
SMOL_STOP_SEQUENCES="${SMOL_STOP_SEQUENCES:-1}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
TOP_LOGPROBS="${TOP_LOGPROBS:-0}"
ALLOW_GIVE_UP="${ALLOW_GIVE_UP:-0}"
VERBALIZED="${VERBALIZED:-0}"
JUDGE_TOOL_BUDGET="${JUDGE_TOOL_BUDGET:-0}"
# A reviewer served next to the agent on the same GPU. The probe showed why it
# is needed: gpt-oss reviewing itself answered 0.99 to everything, five PASS
# verdicts on an episode that failed and five FAIL verdicts on another, never
# once expressing doubt. A second model is the cheapest thing that is not the
# agent's own opinion, and the cluster blocks every hosted endpoint.
JUDGE_SERVE_MODEL="${JUDGE_SERVE_MODEL:-}"
JUDGE_GPU_FRACTION="${JUDGE_GPU_FRACTION:-0.25}"
JUDGE_TOOL_MODEL="${JUDGE_TOOL_MODEL:-}"
JUDGE_TOOL_BASE_URL="${JUDGE_TOOL_BASE_URL:-}"
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

# ---------------------------------------------------------------- cuda toolkit
# vLLM 0.28 samples through FlashInfer, which JIT-compiles its kernels with
# nvcc. A plain python image has none -- torch wheels carry the CUDA runtime,
# not the compiler -- and the engine dies with "Could not find nvcc and default
# cuda_home='/usr/local/cuda' doesn't exist". Two independent guards:
# the PyTorch sampler needs no compilation at all, and a pip-installed toolkit
# satisfies anything else that still wants to build.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
if ! command -v nvcc >/dev/null 2>&1 && [ ! -d /usr/local/cuda ]; then
  echo "[wrapper] no nvcc in the image, installing the pip CUDA toolkit"
  python -m pip install "nvidia-cuda-nvcc-cu12" "nvidia-cuda-runtime-cu12" >/dev/null 2>&1 || true
  nvcc_root=$(python -c "import pathlib,nvidia; print(pathlib.Path(nvidia.__file__).parent)" 2>/dev/null || true)
  if [ -n "${nvcc_root}" ] && [ -x "${nvcc_root}/cuda_nvcc/bin/nvcc" ]; then
    mkdir -p /usr/local/cuda
    ln -sfn "${nvcc_root}/cuda_nvcc/bin" /usr/local/cuda/bin
    ln -sfn "${nvcc_root}/cuda_runtime/include" /usr/local/cuda/include
    export CUDA_HOME=/usr/local/cuda
    export PATH="/usr/local/cuda/bin:${PATH}"
    echo "[wrapper] nvcc at $(command -v nvcc || echo missing)"
  else
    echo "[wrapper] pip CUDA toolkit unavailable; relying on the PyTorch sampler"
  fi
fi

# ---------------------------------------------------------------- serve
# No --served-model-name: the client addresses the model by its HF id.
# A worker whose disk is full kills the run 25 minutes in, inside vLLM's engine
# startup, as "I/O error: No space left on device". Both gpt-oss runs died that
# way on a node with 7.9G left of 1.1T. Check first: it costs a second.
# aiagent01:gpu0 has MIG enabled with no MIG devices configured, so nvidia-smi
# shows an A100 while torch reports "No CUDA GPUs are available" -- 9 of our
# runs died there, each after installing vLLM. One csv query settles it before
# any of that work happens.
# Read every line and take the first in the shell. `| head -1` closes the pipe
# after one line, nvidia-smi takes SIGPIPE, and under `set -euo pipefail` the
# 141 kills the whole run -- which is what happened on the two-GPU worker,
# where nvidia-smi prints two lines and the check that was meant to protect the
# run destroyed it instead.
MIG_ALL=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null || true)
MIG_MODE=$(printf '%s' "${MIG_ALL%%$'\n'*}" | tr -d ' ')
if [ "${MIG_MODE}" = "Enabled" ] && ! nvidia-smi -L 2>/dev/null | grep -q "MIG"; then
  echo "[wrapper] VERDICT: this GPU has MIG enabled with no MIG device; nothing can use it"
  nvidia-smi -L || true
  exit 31
fi

MIN_FREE_GB="${MIN_FREE_GB:-40}"
find "${HF_HOME:-$HOME/.cache/huggingface}" -name '*.incomplete' -delete 2>/dev/null || true
# Weights already in the cache need no room, so the bar is only for a download.
CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--$(echo "${MODEL:-}" | sed 's|/|--|g')"
if [ -d "${CACHE_DIR}" ]; then
  MIN_FREE_GB=8
  echo "[wrapper] weights already cached, requiring only ${MIN_FREE_GB}G"
fi
FREE_GB=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')
echo "[wrapper] free space on /: ${FREE_GB:-?}G (need ${MIN_FREE_GB}G)"
if [ -n "${FREE_GB}" ] && [ "${FREE_GB}" -lt "${MIN_FREE_GB}" ]; then
  echo "[wrapper] VERDICT: not enough disk for the model weights on this worker"
  df -h / || true
  exit 30
fi

echo "[wrapper] serving ${MODEL} on :${PORT} (tp=${TENSOR_PARALLEL_SIZE})"
serve_args=(
  --host 127.0.0.1 --port "${PORT}"
  --max-model-len "${MAX_MODEL_LEN}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
)
if [ -z "${GPU_MEMORY_UTILIZATION}" ] && [ -n "${JUDGE_SERVE_MODEL:-}" ] \
   && [ "${JUDGE_TOOL_BUDGET:-0}" != "0" ]; then
  GPU_MEMORY_UTILIZATION="${AGENT_GPU_FRACTION_WITH_JUDGE}"
  echo "[wrapper] reviewer requested: capping the agent at ${GPU_MEMORY_UTILIZATION} of the card"
fi
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

# A health endpoint only proves that something is listening. Ask it what it is:
# adopting another task's model silently produces a whole run of the wrong data.
SERVED=$(curl -sf "http://127.0.0.1:${PORT}/v1/models" | python -c \
  "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "")
echo "[wrapper] endpoint serves: ${SERVED:-unknown}"
if [ -n "${SERVED}" ] && [ "${SERVED}" != "${MODEL}" ]; then
  echo "[wrapper] FATAL: port ${PORT} is serving ${SERVED}, not ${MODEL}"
  exit 1
fi

# The runner reads the endpoint from .env / the environment. `logprobs` is a
# plain OpenAI parameter here, so none of the OpenRouter provider routing
# applies -- that is the point of serving locally.
export LLM_BASE_URI="http://127.0.0.1:${PORT}/v1"
export LLM_API_KEY="local"
export MODEL_NAME="${MODEL}"

echo "[wrapper] config: policy=${POLICY} model=${MODEL} split=${SPLIT}"
echo "  NUM_EPISODES=${NUM_EPISODES} MAX_STEPS=${MAX_STEPS} AGENT_MAX_STEPS=${AGENT_MAX_STEPS}"
echo "  MAX_GENERATION_TOKENS=${MAX_GENERATION_TOKENS} WORKERS=${WORKERS}"
echo "  TOP_LOGPROBS=${TOP_LOGPROBS} VERBALIZED=${VERBALIZED} JUDGE_TOOL_BUDGET=${JUDGE_TOOL_BUDGET}"
echo "  RUN_ROOT=${RUN_ROOT}"

common_args=(
  --policy "${POLICY}"
  --num-episodes "${NUM_EPISODES}"
  --max-steps "${MAX_STEPS}"
  --agent-max-steps "${AGENT_MAX_STEPS}"
  --max-generation-tokens "${MAX_GENERATION_TOKENS}"
  --empty-response-retries "${EMPTY_RESPONSE_RETRIES}"
  --smol-code-tags "${SMOL_CODE_TAGS}"
  $([ "${SMOL_STOP_SEQUENCES}" = "0" ] && echo "--no-smol-stop-sequences" || echo "--smol-stop-sequences")
  --api-timeout "${API_TIMEOUT}"
  --split "${SPLIT}"
  --seed "${SEED}"
  --output-dir "${RUN_ROOT}"
  --top-logprobs "${TOP_LOGPROBS}"
  --context-limit "${MAX_MODEL_LEN}"
  $([ -n "${REASONING_EFFORT}" ] && echo "--reasoning-effort ${REASONING_EFFORT}")
  --overwrite
)
if [ -n "${JUDGE_SERVE_MODEL}" ] && [ "${JUDGE_TOOL_BUDGET}" != "0" ]; then
  JUDGE_PORT=$(python - <<'PYPORT'
import socket
sock = socket.socket(); sock.bind(("127.0.0.1", 0))
print(sock.getsockname()[1]); sock.close()
PYPORT
)
  echo "[wrapper] serving reviewer ${JUDGE_SERVE_MODEL} on :${JUDGE_PORT}"
  # The reviewer only ever reads a transcript and answers one JSON object, so
  # it needs neither a long context nor graph capture. Both were charged
  # against a card the agent had already taken 60% of, and the server died
  # during engine start -- silently, because its log was never collected.
  vllm serve "${JUDGE_SERVE_MODEL}" --host 127.0.0.1 --port "${JUDGE_PORT}" \
    --max-model-len "${JUDGE_MAX_MODEL_LEN:-8192}" \
    --enforce-eager \
    --gpu-memory-utilization "${JUDGE_GPU_FRACTION}" \
    > /tmp/vllm_judge.log 2>&1 &
  JUDGE_PID=$!
  for i in $(seq 1 "${HEALTH_TIMEOUT_STEPS:-240}"); do
    curl -sf "http://127.0.0.1:${JUDGE_PORT}/health" >/dev/null 2>&1 && break
    kill -0 ${JUDGE_PID} 2>/dev/null || { echo "[wrapper] reviewer died"; tail -n 40 /tmp/vllm_judge.log; break; }
    sleep 5
  done
  if curl -sf "http://127.0.0.1:${JUDGE_PORT}/health" >/dev/null 2>&1; then
    JUDGE_TOOL_BASE_URL="http://127.0.0.1:${JUDGE_PORT}/v1"
    JUDGE_TOOL_MODEL="${JUDGE_SERVE_MODEL}"
    echo "[wrapper] reviewer ready: ${JUDGE_TOOL_MODEL} (independent of the agent)"
  else
    echo "[wrapper] reviewer failed to start; the judge falls back to self-assessment"
    echo "[wrapper] last lines of the reviewer log:"
    tail -n 25 /tmp/vllm_judge.log 2>/dev/null || true
  fi
fi

if [ "${JUDGE_TOOL_BUDGET}" != "0" ]; then
  # The cluster answers hosted endpoints with 403, so an outside reviewer is
  # only available if egress happens to be open. Check rather than assume: a
  # judge served by the same local model is a self-assessment, and the run has
  # to say which of the two it was.
  JUDGE_URL="${JUDGE_TOOL_BASE_URL}"
  if [ -n "${JUDGE_URL}" ]; then
    if curl -sf --max-time 20 "${JUDGE_URL}/models" >/dev/null 2>&1; then
      echo "[wrapper] judge reviewer: ${JUDGE_TOOL_MODEL} at ${JUDGE_URL} (independent)"
    else
      echo "[wrapper] judge reviewer: ${JUDGE_URL} unreachable, falling back to the local model (self-assessment)"
      JUDGE_URL=""
    fi
  fi
  if [ -z "${JUDGE_URL}" ]; then
    JUDGE_URL="${LLM_BASE_URI}"
    JUDGE_TOOL_MODEL="${MODEL}"
    echo "[wrapper] judge reviewer: ${MODEL} on the local endpoint (self-assessment)"
  fi
  common_args+=(
    --judge-tool-budget "${JUDGE_TOOL_BUDGET}"
    --judge-tool-model "${JUDGE_TOOL_MODEL}"
    --judge-tool-base-url "${JUDGE_URL}"
  )
fi
# A locally served vLLM honours top_logprobs and the confidence prompt; both are
# off by default so an existing run reproduces byte for byte.
case "${VERBALIZED}" in
  1|true|True|yes) common_args+=(--verbalized) ;;
esac
case "${ALLOW_GIVE_UP}" in
  1|true|True|yes) common_args+=(--allow-give-up) ;;
esac

if [ "${WORKERS}" -gt 1 ]; then
  python -m experiments.run_alfworld_sharded --workers "${WORKERS}" "${common_args[@]}"
else
  python -m experiments.run_alfworld "${common_args[@]}"
fi

echo "[wrapper] DONE. results under ${RUN_ROOT}"
