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
# Pinned, and not to master. Upstream has since removed pre_artifacts.sh from
# every task -- the script pier runs to turn the agent's commits into
# /logs/artifacts/model.patch. A shallow clone of master therefore ships no
# capture at all: pier skips the missing script without a word, nothing writes
# the patch, and collection fails with "Could not find the file". That, and not
# the agent, is why every cluster run reported empty patches. This commit is
# the one vendored in the repo, and the one our pier 0.3.0 was validated
# against.
DEEPSWE_COMMIT="${DEEPSWE_COMMIT:-e016041}"
if [ -d "${SHARED}/deep-swe/tasks" ] \
   && [ "$(git -C "${SHARED}/deep-swe" rev-parse --short HEAD 2>/dev/null)" = "${DEEPSWE_COMMIT}" ]; then
  echo "[run] tasks already present at ${DEEPSWE_COMMIT}, reusing"
else
  rm -rf "${SHARED}/deep-swe"
  git clone https://github.com/datacurve-ai/deep-swe "${SHARED}/deep-swe" >/dev/null 2>&1 || {
    echo "[run] VERDICT: task clone failed"; exit 22; }
  git -C "${SHARED}/deep-swe" checkout -q "${DEEPSWE_COMMIT}" || {
    echo "[run] VERDICT: cannot check out ${DEEPSWE_COMMIT}"; exit 22; }
fi
echo "[run] tasks: $(ls "${SHARED}/deep-swe/tasks" | wc -l) under ${SHARED} at ${DEEPSWE_COMMIT}"
capture_scripts=$(ls "${SHARED}"/deep-swe/tasks/*/pre_artifacts.sh 2>/dev/null | wc -l)
echo "[run] tasks shipping pre_artifacts.sh: ${capture_scripts}"
if [ "${capture_scripts}" -eq 0 ]; then
  echo "[run] VERDICT: no task ships pre_artifacts.sh, so no patch can ever be captured"
  exit 27
fi

# Grade the working tree, not only what the agent remembered to commit.
# Upstream captures "git diff base HEAD", so an agent that edits, tests and
# submits without committing scores an empty patch. Telling gpt-oss to commit
# as it goes worked (87 of 113 did); Qwen ignored the same instruction, and
# 41 of its 81 trajectories ended by context overflow or the step budget with
# real edits and no commit -- 1 of them graded. SWE-bench grades the working
# tree; so do we, by committing it in the capture script before the diff.
if [ "${COMMIT_WORKING_TREE:-1}" = "1" ]; then
  python - "${SHARED}"/deep-swe/tasks/*/pre_artifacts.sh <<'PY' || { echo "[run] VERDICT: capture-script patch failed"; exit 28; }
import sys
MARK = "harness: working tree"
PRE = ('git add -A . >/dev/null 2>&1 || true\n'
       'git -c user.email=harness@local -c user.name=harness commit -qm "' + MARK + '" >/dev/null 2>&1 || true\n')
done = 0
for path in sys.argv[1:]:
    text = open(path).read()
    if MARK not in text:
        lines = text.splitlines(keepends=True)
        idx = [i for i, l in enumerate(lines) if l.startswith("git diff --binary")]
        if not idx:
            continue
        lines.insert(idx[0], PRE)
        open(path, "w").write("".join(lines))
    done += 1
print(f"[run] capture scripts committing the working tree before the diff: {done} of {len(sys.argv) - 1}")
sys.exit(0 if done == len(sys.argv) - 1 else 28)
PY
fi

echo "=== [4/6] can we pull a task image? ==="
IMAGE=$(grep -ho 'public.ecr.aws[^"]*' "${SHARED}"/deep-swe/tasks/*/environment/Dockerfile 2>/dev/null | head -1)
echo "[run] image: ${IMAGE:-<none found>}"
if [ -n "${IMAGE}" ]; then
  timeout 900 docker pull "${IMAGE}" >/dev/null 2>&1 && echo "[run] pull OK" || echo "[run] pull FAILED (registry throttling was the local failure mode)"
fi

# The capture itself is now the failure. An agent committed
# ("[master 41493e4] 1 file changed"), yet /logs/artifacts/model.patch did not
# exist in the container: pier's own copy reported "Could not find the file".
# /logs/verifier, unlike /logs/artifacts, is bind-mounted to the host trial
# directory, so anything written there survives whatever happens to the
# container. Append a second, mounted copy plus a few facts about the
# environment; nothing existing is changed, so grading is unaffected.
echo "=== [4.5/6] add a mounted copy to each task's artifact capture ==="
capture_patched=0
for script in "${SHARED}"/deep-swe/tasks/*/pre_artifacts.sh; do
  # An unmatched glob expands to itself, and the append below would then create
  # a file named after the pattern: the first attempt reported "1 task" patched
  # and nothing else, which is how the missing scripts stayed hidden.
  [ -f "${script}" ] || continue
  grep -q "pier-cluster-diagnostic" "${script}" && continue
  cat >> "${script}" <<'CAPTURE'

# pier-cluster-diagnostic: /logs/artifacts did not survive on the cluster.
mkdir -p /logs/verifier 2>/dev/null || true
{
  echo "pwd=$(pwd)"
  echo "app=$([ -d /app ] && echo present || echo missing)"
  echo "head=$(git -C /app rev-parse --short HEAD 2>&1)"
  echo "commits_since_base=$(git -C /app rev-list --count HEAD 2>&1)"
  echo "artifacts_dir=$([ -d /logs/artifacts ] && echo present || echo missing)"
  echo "patch_bytes=$(wc -c < /logs/artifacts/model.patch 2>/dev/null || echo missing)"
} > /logs/verifier/pre_artifacts_debug.txt 2>&1
cp /logs/artifacts/model.patch /logs/verifier/model.patch 2>/dev/null || true
CAPTURE
  capture_patched=$((capture_patched + 1))
done
echo "[run] artifact capture extended in ${capture_patched} tasks"

echo "=== [5/6] serve the model locally ==="
# The cluster's egress filter answers OpenRouter with HTTP 403 ("Access denied
# by security policy"), so the agent cannot reach a hosted endpoint at all.
# Serving the model here removes the outbound call and, as a bonus, is the only
# way we ever got complete token log-probabilities.
# A worker whose disk is full kills the run 25 minutes in, inside vLLM's engine
# startup, as "I/O error: No space left on device". Both gpt-oss runs died that
# way on a node with 7.9G left of 1.1T. Check first: it costs a second.
MIN_FREE_GB="${MIN_FREE_GB:-40}"
find "${HF_HOME:-$HOME/.cache/huggingface}" -name '*.incomplete' -delete 2>/dev/null || true
# Weights already in the cache need no room, so the bar is only for a download.
CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--$(echo "${SERVE_MODEL:-}" | sed 's|/|--|g')"
if [ -d "${CACHE_DIR}" ]; then
  MIN_FREE_GB=8
  echo "[run] weights already cached, requiring only ${MIN_FREE_GB}G"
fi
FREE_GB=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')
echo "[run] free space on /: ${FREE_GB:-?}G (need ${MIN_FREE_GB}G)"
if [ -n "${FREE_GB}" ] && [ "${FREE_GB}" -lt "${MIN_FREE_GB}" ]; then
  echo "[run] VERDICT: not enough disk for the model weights on this worker"
  df -h / || true
  exit 30
fi

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
# vLLM moved the parser registry between versions, so ask the CLI rather than
# importing: --help lists the accepted names for this build.
PARSER_CHOICES=$(vllm serve --help 2>/dev/null | tr ',' '\n' | tr -d ' {}' | sort -u)
echo "[run] tool-call parsers offered by this build: $(echo "${PARSER_CHOICES}" | grep -ciE 'hermes|qwen|openai') matches"

# Qwen3.6 does not emit Hermes JSON. It writes the Qwen-Coder XML shape --
#   <tool_call><function=bash><parameter=command>...
# -- so the hermes parser returned tool_calls: None on every single call, and
# all 113 tasks ended in RepeatedFormatError with 2260 "No tool calls found".
# Pick the first name this build actually offers.
pick_parser() {
  for candidate in "$@"; do
    if echo "${PARSER_CHOICES}" | grep -qx "${candidate}"; then
      echo "${candidate}"; return 0
    fi
  done
  echo "$1"  # nothing matched: keep the request and let vLLM complain loudly
}
case "${SERVE_MODEL}" in
  *gpt-oss*) TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-openai}" ;;
  *Qwen3.6*|*qwen3.6*|*Qwen3-*|*qwen3-*)
    TOOL_CALL_PARSER=$(pick_parser qwen3_coder qwen3_xml hermes)
    REASONING_PARSER="${REASONING_PARSER:-qwen3}"
    ;;
esac
echo "[run] tool-call parser: ${TOOL_CALL_PARSER:-none}, reasoning parser: ${REASONING_PARSER:-none}"

TOOL_ARGS=()
if [ -n "${TOOL_CALL_PARSER:-}" ]; then
  TOOL_ARGS=(--enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER}")
fi
# Qwen writes its chain of thought into the answer unless a reasoning parser
# splits it off, which also keeps our log-probability accounting honest.
if [ -n "${REASONING_PARSER:-}" ]; then
  TOOL_ARGS+=(--reasoning-parser "${REASONING_PARSER}")
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
mounted = sorted(root.glob("*/verifier/model.patch"))
if mounted:
    sizes_mounted = [path.stat().st_size for path in mounted]
    print(f"[run] mounted copies: {len(mounted)}, "
          f"{sum(1 for size in sizes_mounted if size > 0)} non-empty, "
          f"{sum(sizes_mounted)} bytes total")
for path in sorted(root.glob("*/verifier/pre_artifacts_debug.txt"))[:2]:
    print(f"[run] {path.parent.parent.name}: "
          + "; ".join(path.read_text().split())) 
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
#
# The second half of the same problem: deep-swe grades "git diff base..HEAD",
# so only committed work is submitted -- upstream says the agent "commits its
# work upon completion" -- but mini-swe-agent's own workflow ends at
# "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and never mentions committing.
# Our two agents edited files, ran the tests, submitted, and left everything
# uncommitted: 80 and 25 commands, zero calls to git commit, empty patches.
# The system prompt states the grading rule and nothing about how to solve the
# task, which is the smallest change that makes the harness agree with the
# benchmark it is running.
# Two more things the second Qwen run taught us. 44 of 109 trajectories died
# of ContextWindowExceededError (from step 75 on, median step 158) and 3 of
# 113 never finished inside the 10-hour budget; every one of those left its
# edits uncommitted, so the benchmark saw an empty patch even where the agent
# had done the work. A step budget bounds the run, and "commit as you go"
# turns a death mid-trajectory into a graded partial patch instead of nothing.
# The budget is stated in the prompt so it is a rule the agent can plan for,
# the way the ALFWorld agents know their 50-step budget.
STEP_LIMIT="${STEP_LIMIT:-0}"
BUDGET_LINE=""
if [ "${STEP_LIMIT}" -gt 0 ]; then
  BUDGET_LINE="    You have a budget of at most ${STEP_LIMIT} commands; the episode ends when it runs out."
fi
AGENT_CONFIG=/tmp/mswea_custom.yaml
cat > "${AGENT_CONFIG}" <<YAML
agent:
  max_consecutive_format_errors: ${MAX_FORMAT_ERRORS:-20}
  step_limit: ${STEP_LIMIT}
  system_template: |
    You are a helpful assistant that can interact with a computer.
    Your work is submitted as git commits: anything left uncommitted in the
    working tree is discarded and counts as no work at all. Commit after every
    meaningful change (for example: git add -A && git commit -m "wip"), and
    make sure everything is committed before you issue the final submit command.
${BUDGET_LINE}
YAML
echo "[run] agent config: $(tr '\n' ' ' < ${AGENT_CONFIG})"

echo "=== [7/7] two real tasks through pier ==="
timeout "${RUN_TIMEOUT_SEC:-21600}" pier run \
  --path "${SHARED}/deep-swe/tasks" \
  --model "openai/${SERVE_MODEL}" \
  --n-tasks "${N_TASKS:-113}" --sample-seed "${SAMPLE_SEED:-0}" --n-concurrent "${N_CONCURRENT:-4}" \
  --jobs-dir "${SHARED}/jobs" --env docker --yes \
  --agent mini-swe-agent \
  --agent-kwarg "model_kwargs={\"logprobs\":true,\"top_logprobs\":${TOP_LOGPROBS:-20}}" \
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
