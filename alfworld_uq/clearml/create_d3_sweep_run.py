#!/usr/bin/env python3
"""Enqueue the Decision 3.0 size sweep.

Six released models on five Qwen backbones, Apache-2.0 throughout. For each
size we want three numbers: the released decision head, the same backbone read
raw at its language-model head, and a head we train ourselves on frozen
features. The small end fits any card; d3-flash and d3 want the 80 GB one.
"""
from __future__ import annotations

import argparse

from clearml import Task

REPO = "https://github.com/rvz16/agents_with_uncertainty_research.git"
FILE_STORE = "https://files.clearai.innopolis.university"
DOCKER_IMAGE = "python:3.12"
DOCKER_ARGS = (
    "--entrypoint= --network=host "
    "-v /tmp/d3_runs:/tmp/d3_runs "
    "-v /root/.clearml/hf-cache:/root/.cache/huggingface"
)
SETUP = """
df -h /
rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list || true
apt-get update -qq -o Acquire::AllowInsecureRepositories=true || true
apt-get install -y -qq --no-install-recommends git curl || true
"""
SMALL = "vllm-sr/d3-edge vllm-sr/d3-lite vllm-sr/d3-nano vllm-sr/d3-mini"
SMALL_BACKBONES = "Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", default="agentic-uq")
    p.add_argument("--name", default="Decision 3.0 size sweep")
    p.add_argument("--queue", default="high_q_80")
    p.add_argument("--branch", default="decision-models-audit")
    p.add_argument("--commit", default=None)
    p.add_argument("--models", default=SMALL)
    p.add_argument("--backbones", default=SMALL_BACKBONES)
    p.add_argument("--stages", default="decision backbone head")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--ood", default="0", help="1 also transfers each trained head across cohorts")
    a = p.parse_args()

    task = Task.create(
        project_name=a.project,
        task_name=a.name,
        repo=REPO,
        branch=a.branch,
        script="alfworld_uq/clearml/d3_sweep_entry.py",
        docker=f"{DOCKER_IMAGE} {DOCKER_ARGS}",
        docker_bash_setup_script=SETUP,
        packages=["clearml", "boto3<1.43"],
    )
    task.output_uri = FILE_STORE
    if a.commit:
        task.set_script(commit=a.commit)
    task.set_parameters({
        "Args/D3_MODELS": a.models,
        "Args/BACKBONES": a.backbones,
        "Args/STAGES": a.stages,
        "Args/OUT_DIR": "/tmp/d3_runs",
        "Args/STATES_DIR": "data/decision_states_clean",
        "Args/VIEWS_DIR": "data/judge_views",
        "Args/EPOCHS": str(a.epochs),
        "Args/OOD": a.ood,
    })
    print(f"Created task {task.id}")
    Task.enqueue(task, queue_name=a.queue)
    print(f"Enqueued to '{a.queue}'")


if __name__ == "__main__":
    main()
