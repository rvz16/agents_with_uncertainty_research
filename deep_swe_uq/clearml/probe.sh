#!/usr/bin/env bash
# Feasibility probe: can Pier start its own Docker containers from inside a
# ClearML task? Everything here is diagnosis first, work second -- the point is
# to learn where the path breaks, not to produce results.
set -uo pipefail

echo "=== [1/5] docker reachable from inside the task container? ==="
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

echo "=== [2/5] deps ==="
python -m pip install --no-cache-dir "datacurve-pier==0.3.0" >/dev/null 2>&1 || {
  echo "[probe] VERDICT: pier install failed"; exit 21; }
python -c "import pier; print('[probe] pier', pier.__version__ if hasattr(pier,'__version__') else 'ok')"

echo "=== [3/5] tasks ==="
git clone --depth 1 https://github.com/datacurve-ai/deep-swe /tmp/deep-swe >/dev/null 2>&1 || {
  echo "[probe] VERDICT: task clone failed"; exit 22; }
echo "[probe] tasks: $(ls /tmp/deep-swe/tasks | wc -l)"

echo "=== [4/5] can we pull a task image? ==="
IMAGE=$(grep -ho 'public.ecr.aws[^"]*' /tmp/deep-swe/tasks/*/environment/Dockerfile 2>/dev/null | head -1)
echo "[probe] image: ${IMAGE:-<none found>}"
if [ -n "${IMAGE}" ]; then
  timeout 900 docker pull "${IMAGE}" >/dev/null 2>&1 && echo "[probe] pull OK" || echo "[probe] pull FAILED (registry throttling was the local failure mode)"
fi

echo "=== [5/5] two real tasks through pier ==="
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
timeout "${PROBE_TIMEOUT_SEC:-3600}" pier run \
  --path /tmp/deep-swe/tasks \
  --model "${MODEL:-openrouter/openai/gpt-oss-20b}" \
  --n-tasks "${N_TASKS:-2}" --sample-seed 0 --n-concurrent 1 \
  --jobs-dir "${RUN_ROOT:-/tmp/probe_runs}" --env docker --yes \
  --agent mini-swe-agent \
  --agent-kwarg 'model_kwargs={"logprobs":true}' \
  --job-name probe
rc=$?
echo "[probe] pier rc=${rc}"
exit ${rc}
