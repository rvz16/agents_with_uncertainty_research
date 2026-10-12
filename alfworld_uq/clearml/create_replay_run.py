#!/usr/bin/env python3
"""Enqueue the per-step label replay.

Every prefix in this work carries its episode's final outcome. This recovers
what was true at each step instead, by replaying the recorded actions in the
same game. No model is served: the actions are in the log, so the job needs
only the ALFWorld dataset and a CPU.
"""
from __future__ import annotations

import argparse

from clearml import Task

REPO = "https://github.com/rvz16/agents_with_uncertainty_research.git"
FILE_STORE = "https://files.clearai.innopolis.university"
DOCKER_IMAGE = "python:3.12"
DOCKER_ARGS = ("--entrypoint= --network=host "
               "-v /tmp/replay_runs:/tmp/replay_runs "
               "-v /root/.cache/alfworld:/root/.cache/alfworld")
SETUP = """
df -h /
rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list || true
apt-get update -qq -o Acquire::AllowInsecureRepositories=true || true
apt-get install -y -qq --no-install-recommends git curl || true
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", default="agentic-uq")
    p.add_argument("--name", default="per-step labels by replay")
    p.add_argument("--queue", default="high_q")
    p.add_argument("--branch", default="decision-models-audit")
    p.add_argument("--commit", default=None)
    p.add_argument("--cohorts", default="RG SG RQ SQ")
    p.add_argument("--probe", default="0",
                   help="replay this many episodes and print the environment's info keys instead")
    a = p.parse_args()

    task = Task.create(
        project_name=a.project, task_name=a.name, repo=REPO, branch=a.branch,
        script="alfworld_uq/clearml/replay_entry.py",
        docker=f"{DOCKER_IMAGE} {DOCKER_ARGS}", docker_bash_setup_script=SETUP,
        packages=["clearml", "boto3<1.43"],
    )
    task.output_uri = FILE_STORE
    if a.commit:
        task.set_script(commit=a.commit)
    task.set_parameters({"Args/OUT_DIR": "/tmp/replay_runs",
                         "Args/INPUTS": "data/replay_inputs",
                         "Args/COHORTS": a.cohorts,
                         "Args/PROBE": a.probe})
    print(f"Created task {task.id}")
    Task.enqueue(task, queue_name=a.queue)
    print(f"Enqueued to '{a.queue}'")


if __name__ == "__main__":
    main()
