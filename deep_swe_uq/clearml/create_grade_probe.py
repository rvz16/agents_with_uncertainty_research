#!/usr/bin/env python3
"""Enqueue the pier capability probe: can it grade a patch we supply?"""
from __future__ import annotations

import argparse

from clearml import Task

REPO = "https://github.com/rvz16/agents_with_uncertainty_research.git"
DOCKER_IMAGE = "python:3.12"
DOCKER_ARGS = "--entrypoint= --network=host -v /var/run/docker.sock:/var/run/docker.sock"
SETUP = """
rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list || true
apt-get update -qq -o Acquire::AllowInsecureRepositories=true || true
apt-get install -y -qq --no-install-recommends git curl || true
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", default="high_q")
    p.add_argument("--branch", default="decision-models-audit")
    p.add_argument("--commit", default=None)
    p.add_argument("--name", default="pier: can it grade a supplied patch")
    a = p.parse_args()
    task = Task.create(project_name="agentic-uq", task_name=a.name, repo=REPO, branch=a.branch,
                       script="deep_swe_uq/clearml/grade_probe_entry.py",
                       docker=f"{DOCKER_IMAGE} {DOCKER_ARGS}", docker_bash_setup_script=SETUP,
                       packages=["clearml", "boto3<1.43"])
    if a.commit:
        task.set_script(commit=a.commit)
    print(f"Created task {task.id}")
    Task.enqueue(task, queue_name=a.queue)
    print(f"Enqueued to '{a.queue}'")


if __name__ == "__main__":
    main()
