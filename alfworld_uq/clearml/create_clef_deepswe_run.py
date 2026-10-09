#!/usr/bin/env python3
"""Enqueue the Cloudflare Clef decision-model baseline on a ClearML agent.

Clef ships no server, so the model runs in process from the Hub release.
clef-flash is 9B and clef is 27B: both fit one 80 GB card in bf16.
"""
from __future__ import annotations

import argparse

from clearml import Task

REPO = "https://github.com/rvz16/agents_with_uncertainty_research.git"
FILE_STORE = "https://files.clearai.innopolis.university"
DOCKER_IMAGE = "python:3.12"
DOCKER_ARGS = (
    "--entrypoint= --network=host "
    "-v /tmp/clef_runs:/tmp/clef_runs "
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
    p.add_argument("--name", default="clef on DeepSWE states with output")
    p.add_argument("--queue", default="high_q_80")
    p.add_argument("--branch", default="alfworld_smolagents")
    p.add_argument("--commit", default=None, help="commit to pin; the branch head otherwise")
    p.add_argument("--models", default="Cloudflare/clef-flash Cloudflare/clef")
    p.add_argument("--max-length", type=int, default=16384)
    p.add_argument("--states-dir", default="/tmp/clef_runs/deepswe_states",
                   help="decision_states (every recorded step) or decision_states_clean (giveup+acted)")
    a = p.parse_args()

    task = Task.create(
        project_name=a.project,
        task_name=a.name,
        repo=REPO,
        branch=a.branch,
        script="alfworld_uq/clearml/clef_deepswe_entry.py",
        docker=f"{DOCKER_IMAGE} {DOCKER_ARGS}",
        docker_bash_setup_script=SETUP,
        packages=["clearml", "boto3<1.43"],
    )
    task.output_uri = FILE_STORE
    if a.commit:
        task.set_script(commit=a.commit)
    task.set_parameters({
        "Args/MODELS": a.models,
        "Args/OUT_DIR": "/tmp/clef_runs",
        "Args/MAX_LENGTH": str(a.max_length),
        "Args/STATES_DIR": a.states_dir,
    })
    print(f"Created task {task.id}")
    Task.enqueue(task, queue_name=a.queue)
    print(f"Enqueued to '{a.queue}'")


if __name__ == "__main__":
    main()
