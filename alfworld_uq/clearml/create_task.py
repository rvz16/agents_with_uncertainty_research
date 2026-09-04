#!/usr/bin/env python3
"""Create and enqueue a ClearML task for an ALFWorld run.

    python create_task.py --smoke                      # 2 episodes, gpt-oss-20b
    python create_task.py --num-episodes 100
    python create_task.py --model Qwen/Qwen3.6-35B-A3B --tensor-parallel-size 2

The agent clones this repo/branch itself, so the branch has to be pushed first.
Queue notes carried over from the SAGE runs (checked 2026-08-27):
high_q_2xA100_80 -> aiagent02:gpu1,2 works; aiagent01:gpu0 pulls tasks and then
dies with "No CUDA GPUs are available"; aiagent03 (high_q / sience) strips
--entrypoint=, so the vLLM image runs its own entrypoint and the task fails.
"""
from __future__ import annotations

import argparse

from clearml import Task

REPO = "https://github.com/rvz16/agents_with_uncertainty_research.git"
DOCKER_IMAGE = "vllm/vllm-openai:v0.12.0"
# --entrypoint= : the image's entrypoint is `vllm`; clear it so ClearML runs python.
# --network=host: the client reaches the in-container endpoint on 127.0.0.1.
DOCKER_ARGS = "--entrypoint= --network=host --shm-size=16g"
SETUP = """
df -h /
# The agent mounts the host's /var/cache/apt/archives into the container, and on
# some workers that cache is corrupt: every repository, not just NVIDIA's, then
# fails with "At least one invalid signature was encountered", apt installs
# nothing, and the agent dies with `Cannot find "git" executable` before it can
# clone the repo. Dropping the NVIDIA list, clearing the stale package lists and
# pointing the archive cache at /tmp bypasses the mounted cache entirely.
rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list || true
rm -rf /var/lib/apt/lists/* || true
mkdir -p /tmp/aptcache/partial
apt-get -o Dir::Cache::archives=/tmp/aptcache -o Acquire::AllowInsecureRepositories=true update -qq || true
apt-get -o Dir::Cache::archives=/tmp/aptcache install -y -qq --no-install-recommends --allow-unauthenticated git curl unzip || true
command -v git || echo "FATAL: git is still missing, the agent cannot clone the repo"
nvidia-smi || echo "no nvidia-smi"
"""


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--project", default="agentic-uq")
    p.add_argument("--name", default=None, help="defaults to 'alfworld <policy> <model>'")
    p.add_argument("--queue", default="high_q_2xA100_80")
    p.add_argument("--branch", default="alfworld_smolagents")
    p.add_argument("--model", default="openai/gpt-oss-20b", help="HF id served by vLLM")
    p.add_argument("--policy", default="smolagents", choices=["smolagents", "llm"])
    p.add_argument("--num-episodes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=30, help="environment action budget")
    p.add_argument(
        "--agent-max-steps",
        type=int,
        default=45,
        help="generation budget for the smolagents loop; keep it above --max-steps so "
        "the environment budget is what ends an episode, as it does for ReAct",
    )
    p.add_argument("--max-generation-tokens", type=int, default=2048)
    p.add_argument("--smol-code-tags", default="markdown", choices=["markdown", "xml"],
                   help="action format for the smolagents loop; gpt-oss follows "
                        "markdown fences far more reliably than <code> tags")
    p.add_argument("--split", default="valid_seen",
                   choices=["train", "valid_seen", "valid_unseen"])
    p.add_argument("--workers", type=int, default=1,
                   help="shards; they share one vLLM endpoint")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--run-name", default=None)
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="extra environment variable for the run; repeatable")
    p.add_argument("--smoke", action="store_true", help="2 episodes, end-to-end check")
    a = p.parse_args()

    model_tag = a.model.split("/")[-1].lower()
    run_name = a.run_name or f"alfworld_{a.policy}_{model_tag}"

    task = Task.create(
        project_name=a.project,
        task_name=a.name or f"alfworld {a.policy} {model_tag}",
        repo=REPO,
        branch=a.branch,
        script="alfworld_uq/clearml/entry.py",
        docker=f"{DOCKER_IMAGE} {DOCKER_ARGS}",
        docker_bash_setup_script=SETUP,
        packages=["clearml"],
    )
    params = {
        "Args/MODEL": a.model,
        "Args/POLICY": a.policy,
        "Args/NUM_EPISODES": "2" if a.smoke else str(a.num_episodes),
        "Args/MAX_STEPS": str(a.max_steps),
        "Args/AGENT_MAX_STEPS": str(a.agent_max_steps),
        "Args/MAX_GENERATION_TOKENS": str(a.max_generation_tokens),
        "Args/SMOL_CODE_TAGS": a.smol_code_tags,
        "Args/SPLIT": a.split,
        "Args/WORKERS": str(a.workers),
        "Args/SEED": str(a.seed),
        "Args/TENSOR_PARALLEL_SIZE": str(a.tensor_parallel_size),
        "Args/MAX_MODEL_LEN": str(a.max_model_len),
        "Args/RUN_NAME": f"{run_name}_smoke" if a.smoke else run_name,
    }
    for item in a.env:
        key, _, value = item.partition("=")
        if key:
            params[f"Args/{key.strip()}"] = value.strip()
    task.set_parameters(params)
    print(f"Created task {task.id}")
    for key, value in params.items():
        print(f"  {key}={value}")
    Task.enqueue(task, queue_name=a.queue)
    print(f"Enqueued to '{a.queue}'")


if __name__ == "__main__":
    main()
