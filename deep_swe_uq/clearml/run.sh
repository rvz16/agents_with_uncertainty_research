#!/usr/bin/env bash
# Full DeepSWE run on a ClearML agent: serve the model in this task, let Pier
# drive its own containers through the mounted host daemon, grade in a pristine
# container. The probe (probe.sh) established every step here; this only adds
# the knobs a real run needs.
set -uo pipefail

echo "=== [1/6] docker reachable from inside the task container? ==="
ls -la /var/run/docker.sock 2>&1 | head -2
if ! command -v docker >/dev/null 2>&1; then
  echo "[run] no docker client, installing"
  (apt-get update -qq -o Acquire::AllowInsecureRepositories=true >/dev/null 2>&1 || true)
  (apt-get install -y -qq --no-install-recommends docker.io >/dev/null 2>&1 || true)
  command -v docker >/dev/null 2>&1 || curl -fsSL https://get.docker.com | sh >/dev/null 2>&1 || true
fi
docker version --format '{{.Server.Version}}' 2>&1 | head -2
docker info --format 'containers={{.Containers}} images={{.Images}} driver={{.Driver}}' 2>&1 | head -2
if ! docker ps >/dev/null 2>&1; then
  echo "[run] VERDICT: docker daemon unreachable -- Pier cannot run here"
  exit 20
fi
echo "[run] docker reachable"

echo "=== [2/6] deps ==="
python -m pip install --no-cache-dir "datacurve-pier==0.3.0" >/dev/null 2>&1 || {
  echo "[run] VERDICT: pier install failed"; exit 21; }
python -c "import pier; print('[probe] pier', pier.__version__ if hasattr(pier,'__version__') else 'ok')"

echo "=== [3/6] tasks ==="
# The task containers are started by the *host* daemon through the mounted
# socket, so every bind mount it resolves is a host path. Anything living only
# inside this container is invisible to them: the first attempt cloned to
# /tmp/deep-swe and both trials died in `docker compose` with empty mounts.
# SHARED is bind-mounted at the same path on both sides, so it resolves alike.
SHARED="${RUN_ROOT:-/tmp/probe_runs}"
mkdir -p "${SHARED}"
# SHARED is bind-mounted from the host, so it survives between tasks: a clone
# into it fails the second time. Reuse what is already there.
if [ -d "${SHARED}/deep-swe/tasks" ]; then
  echo "[run] tasks already present from an earlier run, reusing"
else
  git clone --depth 1 https://github.com/datacurve-ai/deep-swe "${SHARED}/deep-swe" >/dev/null 2>&1 || {
    echo "[run] VERDICT: task clone failed"; exit 22; }
fi
echo "[run] tasks: $(ls "${SHARED}/deep-swe/tasks" | wc -l) under ${SHARED}"

echo "=== [4/6] can we pull a task image? ==="
IMAGE=$(grep -ho 'public.ecr.aws[^"]*' "${SHARED}"/deep-swe/tasks/*/environment/Dockerfile 2>/dev/null | head -1)
echo "[run] image: ${IMAGE:-<none found>}"
if [ -n "${IMAGE}" ]; then
  timeout 900 docker pull "${IMAGE}" >/dev/null 2>&1 && echo "[run] pull OK" || echo "[run] pull FAILED (registry throttling was the local failure mode)"
fi

echo "=== [5/6] serve the model locally ==="
# The cluster's egress filter answers OpenRouter with HTTP 403 ("Access denied
# by security policy"), so the agent cannot reach a hosted endpoint at all.
# Serving the model here removes the outbound call and, as a bonus, is the only
# way we ever got complete token log-probabilities.
SERVE_MODEL="${SERVE_MODEL:-openai/gpt-oss-20b}"
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
python -m pip install --no-cache-dir "vllm==${VLLM_VERSION:-0.28.0}" >/dev/null 2>&1 || {
  echo "[run] VERDICT: vllm install failed"; exit 23; }
export VLLM_USE_FLASHINFER_SAMPLER=0
# 0.0.0.0, not localhost: this task runs with --network=host, so the port lands
# in the host namespace where Pier's task containers can reach it.
# mini-swe-agent sends tool_choice="auto". gpt-oss parses that natively through
# harmony, but every other checkpoint needs the parser named explicitly, and
# without it vLLM rejects the very first call with
#   "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser
# so the whole run finishes with empty patches and no model call ever made.
# A wrong parser name makes vLLM exit during startup, which reads like a CUDA
# failure in the log 20 minutes later. Print what this build actually has.
python - <<'PYPARSERS' || true
try:
    from vllm.entrypoints.openai.tool_parsers import ToolParserManager
    print("[run] tool-call parsers available:", ", ".join(sorted(ToolParserManager.tool_parsers)))
except Exception as exc:
    print("[run] could not list tool-call parsers:", exc)
PYPARSERS

TOOL_ARGS=()
if [ -n "${TOOL_CALL_PARSER:-}" ]; then
  TOOL_ARGS=(--enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER}")
fi
vllm serve "${SERVE_MODEL}" --host 0.0.0.0 --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  ${TOOL_ARGS[@]+"${TOOL_ARGS[@]}"} \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" > /tmp/vllm.log 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT
for i in $(seq 1 "${HEALTH_TIMEOUT_STEPS:-360}"); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
  kill -0 ${VLLM_PID} 2>/dev/null || { echo "[run] VERDICT: vLLM died"; tail -n 40 /tmp/vllm.log; exit 24; }
  sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || { echo "[run] VERDICT: vLLM never healthy"; exit 24; }
echo "[run] vLLM healthy"

# A task container's own localhost is not ours; reach the host over the bridge.
GATEWAY=$(docker network inspect bridge -f '{{(index .IPAM.Config 0).Gateway}}' 2>/dev/null || echo 172.17.0.1)
BASE_URL="http://${GATEWAY}:${PORT}/v1"
echo "[run] agents will call ${BASE_URL}"
# The agents reach the server over the bridge, so that is the address whose
# health matters -- and the model it names has to be ours, not a neighbour
# task's server that happened to take the same port.
SERVED=$(curl -sf "${BASE_URL}/models" | python -c \
  "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "")
if [ -z "${SERVED}" ]; then
  echo "[run] VERDICT: ${BASE_URL} is not reachable; the agents cannot call the model"
  exit 25
fi
echo "[run] endpoint serves: ${SERVED}"
if [ "${SERVED}" != "${SERVE_MODEL}" ]; then
  echo "[run] VERDICT: ${BASE_URL} serves ${SERVED}, not ${SERVE_MODEL}"
  exit 26
fi

echo "=== [6/7] which litellm entry point returns logprobs? ==="
# The agent's own config showed {"drop_params": true, "logprobs": true}: litellm
# silently drops parameters it believes the provider does not support, and a
# locally served model is absent from its registry. This asks litellm directly,
# so one probe cycle settles whether the client or the server is at fault.
python -m pip install --no-cache-dir litellm >/dev/null 2>&1 || true
python - <<PY
import json
try:
    import litellm
    # The adapter picks litellm_response for an openai/ model, and that path calls
    # litellm.responses() -- the Responses API, which has no logprobs at all. The
    # completion path does. This prints the difference rather than assuming it.
    try:
        r = litellm.completion(
            model="openai/${SERVE_MODEL}", api_base="${BASE_URL}", api_key="local",
            messages=[{"role": "user", "content": "say ok"}], max_tokens=8, logprobs=True,
        )
        print(f"[probe] litellm.completion: logprobs={'PRESENT' if r.choices[0].logprobs else 'ABSENT'}")
    except Exception as exc:
        print(f"[probe] litellm.completion FAILED {type(exc).__name__}: {str(exc)[:120]}")
    # The plain call above succeeds even when the server cannot serve the agent:
    # mini-swe-agent always sends tools with tool_choice="auto", and a server
    # started without a parser rejects exactly that, on the first call of every
    # task. Probe the shape the agent actually sends.
    try:
        r = litellm.completion(
            model="openai/${SERVE_MODEL}", api_base="${BASE_URL}", api_key="local",
            messages=[{"role": "user",
                       "content": "Use the bash tool to list the files in /tmp."}],
            max_tokens=64, tool_choice="auto",
            tools=[{"type": "function", "function": {
                "name": "bash", "description": "run a shell command",
                "parameters": {"type": "object",
                               "properties": {"command": {"type": "string"}},
                               "required": ["command"]}}}],
        )
        calls = r.choices[0].message.tool_calls or []
        names = [call.function.name for call in calls]
        print(f"[probe] tool_choice=auto: ACCEPTED, tool_calls={names}")
        # gpt-oss on a hosted endpoint returned 'bash<|channel|>commentary' as
        # the tool name, which the agent rejects as an unknown tool until it
        # gives up with RepeatedFormatError. A clean round-trip here is the
        # difference between a run and 113 empty patches.
        if not names:
            print("[probe] VERDICT: the model returned no tool call for a request "
                  "that plainly needs one; the agent will loop on format errors")
        elif any(name != "bash" for name in names):
            print(f"[probe] VERDICT: tool name is mangled: {names}")
    except Exception as exc:
        print(f"[probe] VERDICT: tool_choice=auto REJECTED -- {str(exc)[:200]}")
    try:
        r = litellm.responses(
            model="openai/${SERVE_MODEL}", api_base="${BASE_URL}", api_key="local",
            input="say ok", max_output_tokens=16,
        )
        print("[probe] litellm.responses: returned, logprobs are not part of that API")
    except Exception as exc:
        print(f"[probe] litellm.responses FAILED {type(exc).__name__}: {str(exc)[:120]}")
except Exception as exc:
    print("[probe] litellm unavailable:", exc)
PY

# Every failure mode so far ended the same way -- a run that finished with
# nothing to grade -- and it was only visible hours later, on the laptop, after
# downloading the artifact. Count the patches here instead.
summarise_patches() {
  local root="${1}"
  python - "${root}" <<'PYSUM'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
patches = sorted(root.glob("*/artifacts/model.patch"))
sizes = [path.stat().st_size for path in patches]
non_empty = [size for size in sizes if size > 0]
print(f"[run] patches: {len(patches)} captured, {len(non_empty)} non-empty, "
      f"{sum(sizes)} bytes total")
statuses = {}
for path in root.glob("*/agent/mini-swe-agent.trajectory.json"):
    try:
        info = json.loads(path.read_text()).get("info", {})
    except Exception:
        continue
    statuses[info.get("exit_status")] = statuses.get(info.get("exit_status"), 0) + 1
print(f"[run] agent exit statuses: {statuses}")
PYSUM
}

# mini-swe-agent gives up after three CONSECUTIVE malformed responses. gpt-oss
# leaks harmony markers into the tool name ("bash<|channel|>commentary") and
# emits truncated JSON arguments every so often, interleaved with dozens of
# perfectly good calls -- 33 successful tool calls and then an exit. The default
# assumes a model whose tool calling never slips; raise it and the agent reads
# the format-error message and carries on.
AGENT_CONFIG=/tmp/mswea_custom.yaml
cat > "${AGENT_CONFIG}" <<YAML
agent:
  max_consecutive_format_errors: ${MAX_FORMAT_ERRORS:-20}
YAML
echo "[run] agent config: $(tr '\n' ' ' < ${AGENT_CONFIG})"

echo "=== [7/7] two real tasks through pier ==="
timeout "${RUN_TIMEOUT_SEC:-21600}" pier run \
  --path "${SHARED}/deep-swe/tasks" \
  --model "openai/${SERVE_MODEL}" \
  --n-tasks "${N_TASKS:-113}" --sample-seed "${SAMPLE_SEED:-0}" --n-concurrent "${N_CONCURRENT:-4}" \
  --jobs-dir "${SHARED}/jobs" --env docker --yes \
  --agent mini-swe-agent \
  --agent-kwarg 'model_kwargs={"logprobs":true}' \
  --agent-kwarg model_class=litellm \
  --agent-kwarg "config_file=${AGENT_CONFIG}" \
  --agent-env "OPENAI_API_KEY=local" \
  --agent-env "OPENAI_API_BASE=${BASE_URL}" \
  --agent-env "OPENAI_BASE_URL=${BASE_URL}" \
  --job-name "${RUN_NAME:-deepswe}"
rc=$?
echo "[run] pier rc=${rc}"
summarise_patches "${SHARED}/jobs/${RUN_NAME:-deepswe}" || true
exit ${rc}
