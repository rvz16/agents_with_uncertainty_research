#!/usr/bin/env python3
"""Enqueue the per-step decision judge on a ClearML agent.

A decision model answers the four-question schema at every prefix of every
episode. The views travel with the repo; only the checkpoint is downloaded.
"""
from __future__ import annotations

import argparse

from clearml import Task

REPO = "https://github.com/rvz16/agents_with_uncertainty_research.git"
FILE_STORE = "https://files.clearai.innopolis.university"
DOCKER_IMAGE = "python:3.12"
DOCKER_ARGS = (
    "--entrypoint= --network=host "
    "-v /tmp/decision_judge:/tmp/decision_judge "
    "-v /root/.clearml/hf-cache:/root/.cache/huggingface"
)
SETUP = """
df -h /
rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list || true
apt-get update -qq -o Acquire::AllowInsecureRepositories=true || true
apt-get install -y -qq --no-install-recommends git curl || true
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", default="agentic-uq")
    p.add_argument("--name", default="per-step decision judge")
    p.add_argument("--queue", default="high_q_80")
    p.add_argument("--branch", default="alfworld_smolagents")
    p.add_argument("--commit", default=None, help="commit to pin; the branch head otherwise")
    p.add_argument("--model", default="LiquidAI/d1-omni-600M")
    p.add_argument("--backend", default="liquid", choices=["liquid", "clef"])
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--views-dir", default="alfworld_uq/data/judge_views")
    a = p.parse_args()

    task = Task.create(
        project_name=a.project,
        task_name=a.name,
        repo=REPO,
        branch=a.branch,
        script="alfworld_uq/clearml/decision_judge_entry.py",
        docker=f"{DOCKER_IMAGE} {DOCKER_ARGS}",
        docker_bash_setup_script=SETUP,
        packages=["clearml", "boto3"],
    )
    task.output_uri = FILE_STORE
    if a.commit:
        task.set_script(commit=a.commit)
    task.set_parameters({
        "Args/MODEL": a.model,
        "Args/BACKEND": a.backend,
        "Args/BATCH": str(a.batch),
        "Args/WINDOW": str(a.window),
        "Args/OUT_DIR": "/tmp/decision_judge",
        "Args/VIEWS_DIR": a.views_dir,
    })
    print(f"Created task {task.id}")
    Task.enqueue(task, queue_name=a.queue)
    print(f"Enqueued to '{a.queue}'")


if __name__ == "__main__":
    main()
