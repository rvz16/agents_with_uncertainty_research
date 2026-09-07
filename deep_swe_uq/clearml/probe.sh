#!/usr/bin/env bash
# Feasibility probe: can Pier start its own Docker containers from inside a
# ClearML task? Everything here is diagnosis first, work second -- the point is
# to learn where the path breaks, not to produce results.
set -uo pipefail

echo "=== [1/6] docker reachable from inside the task container? ==="
ls -la /var/run/docker.sock 2>&1 | head -2
if ! command -v docker >/dev/null 2>&1; then
  echo "[probe] no docker client, installing"
  (apt-get update -qq -o Acquire::AllowInsecureRepositories=true >/dev/null 2>&1 || true)
  (apt-get install -y -qq --no-install-recommends docker.io >/dev/null 2>&1 || true)
  command -v docker >/dev/null 2>&1 || curl -fsSL https://get.docker.com | sh >/dev/null 2>&1 || true
fi
docker version --format '{{.Server.Version}}' 2>&1 | head -2
docker info --format 'containers={{.Containers}} images={{.Images}} driver={{.Driver}}' 2>&1 | head -2
if ! docker ps >/dev/null 2>&1; then
  echo "[probe] VERDICT: docker daemon unreachable -- Pier cannot run here"
  exit 20
fi
echo "[probe] docker reachable"

echo "=== [2/6] deps ==="
python -m pip install --no-cache-dir "datacurve-pier==0.3.0" >/dev/null 2>&1 || {
  echo "[probe] VERDICT: pier install failed"; exit 21; }
python -c "import pier; print('[probe] pier', pier.__version__ if hasattr(pier,'__version__') else 'ok')"

echo "=== [3/6] tasks ==="
# The task containers are started by the *host* daemon through the mounted
# socket, so every bind mount it resolves is a host path. Anything living only
# inside this container is invisible to them: the first attempt cloned to
# /tmp/deep-swe and both trials died in `docker compose` with empty mounts.
# SHARED is bind-mounted at the same path on both sides, so it resolves alike.
SHARED="${RUN_ROOT:-/tmp/probe_runs}"
mkdir -p "${SHARED}"
git clone --depth 1 https://github.com/datacurve-ai/deep-swe "${SHARED}/deep-swe" >/dev/null 2>&1 || {
  echo "[probe] VERDICT: task clone failed"; exit 22; }
echo "[probe] tasks: $(ls "${SHARED}/deep-swe/tasks" | wc -l) under ${SHARED}"

echo "=== [4/6] can we pull a task image? ==="
IMAGE=$(grep -ho 'public.ecr.aws[^"]*' "${SHARED}"/deep-swe/tasks/*/environment/Dockerfile 2>/dev/null | head -1)
echo "[probe] image: ${IMAGE:-<none found>}"
if [ -n "${IMAGE}" ]; then
  timeout 900 docker pull "${IMAGE}" >/dev/null 2>&1 && echo "[probe] pull OK" || echo "[probe] pull FAILED (registry throttling was the local failure mode)"
fi

echo "=== [5/6] serve the model locally ==="
# The cluster's egress filter answers OpenRouter with HTTP 403 ("Access denied
# by security policy"), so the agent cannot reach a hosted endpoint at all.
# Serving the model here removes the outbound call and, as a bonus, is the only
# way we ever got complete token log-probabilities.
SERVE_MODEL="${SERVE_MODEL:-openai/gpt-oss-20b}"
PORT="${VLLM_PORT:-8010}"
python -m pip install --no-cache-dir "vllm==${VLLM_VERSION:-0.28.0}" >/dev/null 2>&1 || {
  echo "[probe] VERDICT: vllm install failed"; exit 23; }
export VLLM_USE_FLASHINFER_SAMPLER=0
# 0.0.0.0, not localhost: this task runs with --network=host, so the port lands
# in the host namespace where Pier's task containers can reach it.
vllm serve "${SERVE_MODEL}" --host 0.0.0.0 --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" > /tmp/vllm.log 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT
for i in $(seq 1 "${HEALTH_TIMEOUT_STEPS:-360}"); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
  kill -0 ${VLLM_PID} 2>/dev/null || { echo "[probe] VERDICT: vLLM died"; tail -n 40 /tmp/vllm.log; exit 24; }
  sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || { echo "[probe] VERDICT: vLLM never healthy"; exit 24; }
echo "[probe] vLLM healthy"

# A task container's own localhost is not ours; reach the host over the bridge.
GATEWAY=$(docker network inspect bridge -f '{{(index .IPAM.Config 0).Gateway}}' 2>/dev/null || echo 172.17.0.1)
BASE_URL="http://${GATEWAY}:${PORT}/v1"
echo "[probe] agents will call ${BASE_URL}"
curl -sf "${BASE_URL}/models" >/dev/null && echo "[probe] endpoint reachable via gateway" || echo "[probe] WARNING: gateway address not reachable from here"

echo "=== [6/6] two real tasks through pier ==="
timeout "${PROBE_TIMEOUT_SEC:-3600}" pier run \
  --path "${SHARED}/deep-swe/tasks" \
  --model "openai/${SERVE_MODEL}" \
  --n-tasks "${N_TASKS:-2}" --sample-seed 0 --n-concurrent 1 \
  --jobs-dir "${SHARED}/jobs" --env docker --yes \
  --agent mini-swe-agent \
  --agent-kwarg 'model_kwargs={"logprobs":true}' \
  --agent-env "OPENAI_API_KEY=local" \
  --agent-env "OPENAI_API_BASE=${BASE_URL}" \
  --agent-env "OPENAI_BASE_URL=${BASE_URL}" \
  --job-name probe
rc=$?
echo "[probe] pier rc=${rc}"
exit ${rc}
