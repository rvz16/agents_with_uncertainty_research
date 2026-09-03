#!/usr/bin/env python3
"""Create and enqueue a ClearML task for a SAGE uncertainty run.

    python create_task.py --benchmarks lcb_hard --smoke
    python create_task.py --benchmarks codecontests

The agent clones this repo/branch itself, so the branch has to be pushed first.
Queue notes (checked 2026-08-27): high_q_80 -> aiagent01:gpu0/gpu1, aiagent02:gpu0;
high_q_2xA100_80 -> aiagent02:gpu1,2. aiagent01:gpu0 pulls tasks and then dies with
"No CUDA GPUs are available"; aiagent03 (high_q / sience) strips --entrypoint=, so
the vLLM image runs its own entrypoint and the task fails immediately.
"""
from __future__ import annotations

import argparse
import os

from clearml import Task

REPO = "https://github.com/rvz16/agents_with_uncertainty_research.git"
DOCKER_IMAGE = "vllm/vllm-openai:v0.12.0"
# --entrypoint= : the image's entrypoint is `vllm`; clear it so ClearML runs python.
# --network=host: the client reaches the in-container endpoint on 127.0.0.1.
DOCKER_ARGS = "--entrypoint= --network=host --shm-size=16g"
SETUP = """
df -h /
apt-get update -qq --allow-insecure-repositories || true
apt-get install -y -qq --no-install-recommends git curl || true
nvidia-smi || echo "no nvidia-smi"
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default="agentic-uq")
    p.add_argument("--name", default=None, help="defaults to 'sage-uq <benchmarks>'")
    p.add_argument("--queue", default="high_q_2xA100_80")
    p.add_argument("--branch", default="joint_exps_clearml")
    p.add_argument("--benchmarks", default="lcb_hard")
    p.add_argument("--n-instances", type=int, default=0, help="0 = all")
    p.add_argument("--max-verifications", type=int, default=1,
                   help="budget for the TERMINAL verification; 0 skips it and the run "
                        "produces no labels at all. See JOINT_RUN_CONFIG.md")
    p.add_argument("--l3-model", default=None,
                   help="reviewer model for the L3 critic; default anthropic/claude-haiku-4.5")
    p.add_argument("--l3-local", type=int, default=0, choices=[0, 1],
                   help="1 points the L3 reviewer at the container's own vLLM instead of "
                        "OpenRouter, which the cluster egress filter blocks. Makes L3 a "
                        "SELF-review by the generator, not an independent judge")
    p.add_argument("--calibrate-l3", type=int, default=1, choices=[0, 1],
                   help="0 lets the analysis finish when the reviewer is unavailable")
    p.add_argument("--lcb-platform", default=None,
                   choices=["leetcode", "atcoder", "codeforces", "all"],
                   help="LCB source platform. Difficulty comes from the benchmark name, so "
                        "lcb_hard + codeforces is a different task pool from lcb_hard + leetcode")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="extra environment variable for the run; repeatable")
    p.add_argument("--smoke", action="store_true", help="6 instances, end-to-end check")
    a = p.parse_args()

    docker_args = DOCKER_ARGS
    if key := os.environ.get("OPENROUTER_API_KEY", ""):
        # The agent echoes the whole docker command into the task console, so the
        # key ends up readable by anyone with access to the task.
        docker_args += f" -e OPENROUTER_API_KEY={key}"
        print("OPENROUTER_API_KEY: injected (L3 critic enabled)")
    else:
        print("WARNING: OPENROUTER_API_KEY unset -> the L3 critic will be skipped")
    if a.l3_model:
        docker_args += f" -e L3_REVIEW_MODEL={a.l3_model}"
        print(f"L3 reviewer model: {a.l3_model}")

    task = Task.create(
        project_name=a.project,
        task_name=a.name or f"sage-uq {a.benchmarks}",
        repo=REPO,
        branch=a.branch,
        script="experiments/orchestration_hypothesis_testing/scripts/clearml/entry.py",
        docker=f"{DOCKER_IMAGE} {docker_args}",
        docker_bash_setup_script=SETUP,
        packages=["clearml"],
    )
    params = {
        "Args/BENCHMARKS": a.benchmarks,
        "Args/N_INSTANCES": "6" if a.smoke else str(a.n_instances),
        "Args/MAX_VERIFICATIONS": str(a.max_verifications),
        "Args/CALIBRATE_L3": str(a.calibrate_l3),
        "Args/L3_LOCAL": str(a.l3_local),
    }
    if a.lcb_platform:
        params["Args/LCB_PLATFORM"] = a.lcb_platform
    for item in a.env:
        key, _, value = item.partition("=")
        if key:
            params[f"Args/{key.strip()}"] = value.strip()
    task.set_parameters(params)
    print(f"Created task {task.id}")
    for k, v in params.items():
        print(f"  {k}={v}")
    Task.enqueue(task, queue_name=a.queue)
    print(f"Enqueued to '{a.queue}'")


if __name__ == "__main__":
    main()
