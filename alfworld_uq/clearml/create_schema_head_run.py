#!/usr/bin/env python3
"""Enqueue the schema head we train ourselves over a frozen backbone.

The released decision models are a frozen Qwen with a trained schema head. This
trains that head on our own episodes over the same frozen backbone, which is
the comparison the appendix is missing: a logistic regression measures the
features alone, and the schema head under the released recipe -- a
label-smoothed cross-entropy with a Brier term -- measures what the head adds.
Qwen3.5-9B in bf16 is some 18 GB, so one 80 GB card is ample.
"""
from __future__ import annotations

import argparse

from clearml import Task

REPO = "https://github.com/rvz16/agents_with_uncertainty_research.git"
FILE_STORE = "https://files.clearai.innopolis.university"
DOCKER_IMAGE = "python:3.12"
DOCKER_ARGS = (
    "--entrypoint= --network=host "
    "-v /tmp/schema_runs:/tmp/schema_runs "
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
    p.add_argument("--name", default="schema head on a frozen backbone")
    p.add_argument("--queue", default="high_q_80")
    p.add_argument("--branch", default="alfworld_smolagents")
    p.add_argument("--commit", default=None, help="commit to pin; the branch head otherwise")
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--feature-batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--aux", default="1")
    p.add_argument("--shuffle", default="0", help="1 runs the permuted-label control instead")
    p.add_argument("--ood", default="0", help="1 also trains on each cohort and scores the others")
    a = p.parse_args()

    task = Task.create(
        project_name=a.project,
        task_name=a.name,
        repo=REPO,
        branch=a.branch,
        script="alfworld_uq/clearml/schema_head_entry.py",
        docker=f"{DOCKER_IMAGE} {DOCKER_ARGS}",
        docker_bash_setup_script=SETUP,
        packages=["clearml", "boto3<1.43"],
    )
    task.output_uri = FILE_STORE
    if a.commit:
        task.set_script(commit=a.commit)
    task.set_parameters({
        "Args/MODEL": a.model,
        "Args/OUT_DIR": "/tmp/schema_runs",
        "Args/STATES_DIR": "data/decision_states_clean",
        "Args/VIEWS_DIR": "data/judge_views",
        "Args/MAX_LENGTH": str(a.max_length),
        "Args/FEATURE_BATCH": str(a.feature_batch),
        "Args/EPOCHS": str(a.epochs),
        "Args/AUX": a.aux,
        "Args/SHUFFLE": a.shuffle,
        "Args/OOD": a.ood,
    })
    print(f"Created task {task.id}")
    Task.enqueue(task, queue_name=a.queue)
    print(f"Enqueued to '{a.queue}'")


if __name__ == "__main__":
    main()
