#!/usr/bin/env python3
"""Create + enqueue a ClearML task for a CLEAN code-UQ generation run.

Runs on the GPU agent inside vllm/vllm-openai:v0.12.0 (vLLM + torch preinstalled),
which sidesteps the driver/compiler problems we hit building vLLM on the server.

The task clones this PUBLIC repo/branch (no git creds needed on the agent),
serves gpt-oss-20b, generates clean trajectories, and uploads RUN_ROOT as
artifacts. All extra UQ analysis is run locally on the downloaded artifacts.

Usage:
    python scripts/create_clean_uq_task.py --smoke          # 6 instances, lcb_hard
    python scripts/create_clean_uq_task.py                  # full lcb_hard
    python scripts/create_clean_uq_task.py \
        --benchmarks lcb_hard,codecontests --n-instances 0
"""
import argparse
import os

from clearml import Task

REPO = "https://github.com/rvz16/agents_with_uncertainty_research.git"
BRANCH = "clearml_clean_run"
SCRIPT = "scripts/clearml_entry.py"

DOCKER_IMAGE = "vllm/vllm-openai:v0.12.0"
# --entrypoint= : the image's default entrypoint is `vllm`; clear it so ClearML runs python.
# --network=host: client reaches the in-container vLLM on 127.0.0.1.
DOCKER_ARGS = "--entrypoint= --network=host --shm-size=16g -e SKIP_LM_POLYGRAPH=1"

# ClearML flattens this to one line joined by ";" — no brace-groups / heredocs.
DOCKER_BASH_SETUP = r"""
df -h /
apt-get update -qq --allow-insecure-repositories || true
apt-get install -y -qq --no-install-recommends git curl || true
git --version || echo "ERROR: git missing"
curl --version | head -1 || echo "ERROR: curl missing"
nvidia-smi || echo "no nvidia-smi"
"""


def create(args: argparse.Namespace) -> None:
    docker_args = DOCKER_ARGS
    orkey = os.environ.get("OPENROUTER_API_KEY", "")
    if orkey:
        docker_args += f" -e OPENROUTER_API_KEY={orkey}"
        print("OPENROUTER_API_KEY: injected (L3 critic enabled)")
    else:
        print("WARNING: OPENROUTER_API_KEY not set locally -> L3 critic will be skipped")

    task = Task.create(
        project_name=args.project,
        task_name=args.name,
        repo=REPO,
        branch=BRANCH,
        script=SCRIPT,
        docker=f"{DOCKER_IMAGE} {docker_args}",
        docker_bash_setup_script=DOCKER_BASH_SETUP,
        packages=["clearml"],
    )

    n_instances = "6" if args.smoke else str(args.n_instances)
    benchmarks = args.benchmarks
    params = {
        "Args/BENCHMARKS": benchmarks,
        "Args/N_INSTANCES": n_instances,
        "Args/MAX_VERIFICATIONS": "0",   # terminal single final verify => no oracle leak
        "Args/PRIVATE_TEST_CAP": "0",    # all private tests => real balance
        "Args/MAX_GENERATIONS": "5",
        "Args/MAX_STEPS": "20",
    }
    task.set_parameters(params)
    print(f"Created task {task.id}")
    for k, v in params.items():
        print(f"  {k}={v}")

    Task.enqueue(task, queue_name=args.queue)
    print(f"Enqueued to queue '{args.queue}' (id={task.id})")
    print(f"Watch: task console in ClearML UI, project '{args.project}'")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="agentic-uq")
    p.add_argument("--name", default="clean code-UQ gpt-oss-20b")
    p.add_argument("--queue", default="high_q_80")
    p.add_argument("--benchmarks", default="lcb_hard")
    p.add_argument("--n-instances", type=int, default=0, help="0 = all")
    p.add_argument("--smoke", action="store_true", help="6 instances for a fast end-to-end test")
    create(p.parse_args())
