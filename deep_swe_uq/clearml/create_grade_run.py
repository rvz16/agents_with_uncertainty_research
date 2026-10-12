#!/usr/bin/env python3
"""Enqueue DeepSWE grading. Validation first: golden solutions must score 1.0."""
from __future__ import annotations

import argparse

from clearml import Task

REPO = "https://github.com/rvz16/agents_with_uncertainty_research.git"
FILE_STORE = "https://files.clearai.innopolis.university"
DOCKER_IMAGE = "python:3.12"
DOCKER_ARGS = ("--entrypoint= --network=host "
               "-v /var/run/docker.sock:/var/run/docker.sock "
               "-v /tmp/grade_shared:/tmp/grade_shared -v /tmp/grade_runs:/tmp/grade_runs")
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
    p.add_argument("--command", default="validate", choices=["validate", "grade"])
    p.add_argument("--limit", default="5")
    p.add_argument("--only", default="")
    p.add_argument("--name", default="deepswe: golden validation")
    a = p.parse_args()
    task = Task.create(project_name="agentic-uq", task_name=a.name, repo=REPO, branch=a.branch,
                       script="deep_swe_uq/clearml/grade_entry.py",
                       docker=f"{DOCKER_IMAGE} {DOCKER_ARGS}", docker_bash_setup_script=SETUP,
                       packages=["clearml", "boto3<1.43"])
    task.output_uri = FILE_STORE
    if a.commit:
        task.set_script(commit=a.commit)
    task.set_parameters({"Args/COMMAND": a.command, "Args/LIMIT": a.limit, "Args/ONLY": a.only,
                         "Args/OUT_DIR": "/tmp/grade_runs", "Args/SHARED": "/tmp/grade_shared"})
    print(f"Created task {task.id}")
    Task.enqueue(task, queue_name=a.queue)
    print(f"Enqueued to '{a.queue}'")


if __name__ == "__main__":
    main()
