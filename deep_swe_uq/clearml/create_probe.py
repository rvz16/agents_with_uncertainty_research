#!/usr/bin/env python3
"""Enqueue the DeepSWE feasibility probe.

Pier starts its own Docker containers, but a ClearML task already runs inside
one. The probe answers whether the agent lets us reach the host daemon; it
mounts the socket, so it fails fast and cheaply if the cluster forbids that.
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
DOCKER_IMAGE = "python:3.12"
DOCKER_ARGS = (
    "--entrypoint= --network=host "
    "-v /var/run/docker.sock:/var/run/docker.sock "
    "-v /tmp/probe_runs:/tmp/probe_runs"
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
    p.add_argument("--name", default="deepswe docker-in-docker probe")
    p.add_argument("--queue", default="CPU_queue")
    p.add_argument("--branch", default="alfworld_smolagents")
    p.add_argument("--n-tasks", type=int, default=2)
    p.add_argument("--model", default="openrouter/openai/gpt-oss-20b")
    a = p.parse_args()

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
        script="deep_swe_uq/clearml/probe_entry.py",
        docker=f"{DOCKER_IMAGE} {docker_args}",
        docker_bash_setup_script=SETUP,
        packages=["clearml", "boto3"],
    )
    task.output_uri = FILE_STORE
    task.set_parameters({
        "Args/N_TASKS": str(a.n_tasks),
        "Args/MODEL": a.model,
        "Args/RUN_ROOT": "/tmp/probe_runs",
        "Args/PROBE_TIMEOUT_SEC": "3600",
    })
    print(f"Created task {task.id}")
    Task.enqueue(task, queue_name=a.queue)
    print(f"Enqueued to '{a.queue}'")


if __name__ == "__main__":
    main()
