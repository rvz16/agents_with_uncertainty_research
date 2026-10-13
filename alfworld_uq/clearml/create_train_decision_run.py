#!/usr/bin/env python3
"""Enqueue training of our own decision model.

Four runs make the claim: the full recipe, then one ablation each for the
distilled posterior and the structural questions, then the permuted control.
"""
from __future__ import annotations

import argparse

from clearml import Task

REPO = "https://github.com/rvz16/agents_with_uncertainty_research.git"
FILE_STORE = "https://files.clearai.innopolis.university"
DOCKER_IMAGE = "python:3.12"
DOCKER_ARGS = ("--entrypoint= --network=host -v /tmp/decision_runs:/tmp/decision_runs "
               "-v /root/.clearml/hf-cache:/root/.cache/huggingface")
SETUP = """
df -h /
rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list || true
apt-get update -qq -o Acquire::AllowInsecureRepositories=true || true
apt-get install -y -qq --no-install-recommends git curl || true
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", default="high_q_80")
    p.add_argument("--branch", default="decision-models-audit")
    p.add_argument("--commit", default=None)
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--tag", default="ours")
    p.add_argument("--lora-rank", default="16")
    p.add_argument("--epochs", default="2")
    p.add_argument("--batch", default="8")
    p.add_argument("--w-distill", default="1.0")
    p.add_argument("--w-aux", default="0.3")
    p.add_argument("--shuffle", default="0")
    p.add_argument("--holdout", default="", help="train on the other cohorts and score this one")
    p.add_argument("--name", default="our decision model: full recipe")
    a = p.parse_args()
    task = Task.create(project_name="agentic-uq", task_name=a.name, repo=REPO, branch=a.branch,
                       script="alfworld_uq/clearml/train_decision_entry.py",
                       docker=f"{DOCKER_IMAGE} {DOCKER_ARGS}", docker_bash_setup_script=SETUP,
                       packages=["clearml", "boto3<1.43"])
    task.output_uri = FILE_STORE
    if a.commit:
        task.set_script(commit=a.commit)
    task.set_parameters({"Args/MODEL": a.model, "Args/TAG": a.tag, "Args/OUT_DIR": "/tmp/decision_runs",
                         "Args/LORA_RANK": a.lora_rank, "Args/EPOCHS": a.epochs, "Args/BATCH": a.batch,
                         "Args/W_DISTILL": a.w_distill, "Args/W_AUX": a.w_aux,
                         "Args/SHUFFLE": a.shuffle, "Args/HOLDOUT": a.holdout})
    print(f"Created task {task.id}")
    Task.enqueue(task, queue_name=a.queue)
    print(f"Enqueued to '{a.queue}'")


if __name__ == "__main__":
    main()
