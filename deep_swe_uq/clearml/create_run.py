#!/usr/bin/env python3
"""Enqueue a full DeepSWE run on a ClearML agent.

The model is served inside the task: the cluster answers hosted endpoints with
HTTP 403, and a local server is also the only configuration that returns token
log-probabilities. `model_class=litellm` is not optional -- Pier otherwise picks
the Responses API for an openai/ model, where logprobs do not exist.
"""
from __future__ import annotations

import argparse
import os

from clearml import Task

REPO = "https://github.com/rvz16/agents_with_uncertainty_research.git"
# The agent's default output_uri is the s3 bucket, and ClearML prepends its own
# Task.init() to the entry script: without an explicit destination that init
# fails before our code runs, with no boto3 and no credentials for that bucket.
FILE_STORE = "https://files.clearai.innopolis.university"
# A plain python image: it ships git, and nothing here needs a GPU.
DOCKER_IMAGE = "python:3.12"  # vLLM comes from pip; the wheels carry their own CUDA runtime
DOCKER_ARGS = (
    "--entrypoint= --network=host "
    "-v /var/run/docker.sock:/var/run/docker.sock "
    "-v /tmp/deepswe_runs:/tmp/deepswe_runs -v /root/.clearml/hf-cache:/root/.cache/huggingface"
)
SETUP = """
df -h /
rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list || true
apt-get update -qq -o Acquire::AllowInsecureRepositories=true || true
apt-get install -y -qq --no-install-recommends git curl || true
ls -la /var/run/docker.sock || echo "no docker socket mounted"
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", default="agentic-uq")
    p.add_argument("--name", default="deepswe run")
    p.add_argument("--queue", default="high_q_2xA100_80")
    p.add_argument("--branch", default="alfworld_smolagents")
    p.add_argument("--n-tasks", type=int, default=113)
    p.add_argument("--n-concurrent", type=int, default=4)
    p.add_argument("--tensor-parallel-size", type=int, default=2)
    p.add_argument("--run-name", default="deepswe")
    p.add_argument("--model", default="openrouter/openai/gpt-oss-20b")
    p.add_argument("--serve-model", default="Qwen/Qwen3.6-35B-A3B",
                   help="served locally with vLLM; the cluster blocks hosted endpoints")
    p.add_argument("--max-model-len", type=int, default=131072,
                   help="a DeepSWE trajectory is long: one agent died at "
                        "32814 tokens against a 32768 window, 94 calls in")
    p.add_argument("--max-format-errors", type=int, default=20,
                   help="consecutive malformed responses mini-swe-agent tolerates "
                        "before giving up; its default of 3 ends a gpt-oss run "
                        "that is otherwise making progress")
    p.add_argument("--tool-call-parser", default=None,
                   help="vLLM parser for tool_choice=auto, which mini-swe-agent "
                        "always sends. Defaults to openai for gpt-oss and hermes "
                        "otherwise; pass an empty string to serve without one.")
    a = p.parse_args()
    if a.tool_call_parser is None:
        # gpt-oss speaks harmony, which vLLM parses with its `openai` parser;
        # everything else here is a Qwen3-family checkpoint, which uses hermes.
        # Serving gpt-oss without one is what let OpenRouter-style harmony
        # markers ("bash<|channel|>commentary") reach the agent as a tool name.
        a.tool_call_parser = (
            "openai" if "gpt-oss" in a.serve_model.lower() else "hermes"
        )

    docker_args = DOCKER_ARGS
    if key := os.environ.get("OPENROUTER_API_KEY", ""):
        docker_args += f" -e OPENROUTER_API_KEY={key}"
    else:
        print("WARNING: OPENROUTER_API_KEY unset; the probe stops after the docker check")

    task = Task.create(
        project_name=a.project,
        task_name=a.name,
        repo=REPO,
        branch=a.branch,
        script="deep_swe_uq/clearml/run_entry.py",
        docker=f"{DOCKER_IMAGE} {docker_args}",
        docker_bash_setup_script=SETUP,
        packages=["clearml", "boto3"],
    )
    task.output_uri = FILE_STORE
    task.set_parameters({
        "Args/N_TASKS": str(a.n_tasks),
        "Args/MODEL": a.model,
        "Args/RUN_ROOT": "/tmp/deepswe_runs",
        "Args/RUN_TIMEOUT_SEC": "21600",
        "Args/N_CONCURRENT": str(a.n_concurrent),
        "Args/TENSOR_PARALLEL_SIZE": str(a.tensor_parallel_size),
        "Args/RUN_NAME": a.run_name,
        "Args/SERVE_MODEL": a.serve_model,
        "Args/TOOL_CALL_PARSER": a.tool_call_parser,
        "Args/MAX_FORMAT_ERRORS": str(a.max_format_errors),
        "Args/MAX_MODEL_LEN": str(a.max_model_len),
        "Args/VLLM_VERSION": "0.28.0",
        "Args/HEALTH_TIMEOUT_STEPS": "720",
    })
    print(f"Created task {task.id}")
    Task.enqueue(task, queue_name=a.queue)
    print(f"Enqueued to '{a.queue}'")


if __name__ == "__main__":
    main()
