#!/usr/bin/env python3
"""Enqueue GRPO over the confidence head.

Two runs make the comparison the supervised table cannot: the same recipe with
the reward layered on the finished adapter (--sft-task), and the reward alone
from the base model. The supervised arm itself is create_train_decision_run.py
with the same --drop, so all three see one set of folds.
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
    p.add_argument("--tag", default="grpo")
    p.add_argument("--keep", default="DG,DQ")
    p.add_argument("--sft-task", default="",
                   help="a finished supervised run whose weights_s0f* artifacts start this one: "
                        "the on-top-of-SFT arm. Omit for the from-scratch arm")
    p.add_argument("--data-task", default="")
    p.add_argument("--epochs", default="2")
    p.add_argument("--batch", default="8")
    p.add_argument("--lr", default="5e-5")
    p.add_argument("--group", default="8")
    p.add_argument("--warmup", default="100")
    p.add_argument("--w-cal", default="0.3")
    p.add_argument("--w-entropy", default="0.01")
    p.add_argument("--kappa", default="0.3")
    p.add_argument("--c-ask", default="0.5")
    p.add_argument("--tau-stop", default="0.0")
    p.add_argument("--tau-ask", default="0.5")
    p.add_argument("--folds", default="5")
    p.add_argument("--fold-limit", default="0")
    p.add_argument("--name", default="GRPO on the confidence head")
    a = p.parse_args()
    task = Task.create(project_name="agentic-uq", task_name=a.name, repo=REPO, branch=a.branch,
                       script="alfworld_uq/clearml/grpo_entry.py",
                       docker=f"{DOCKER_IMAGE} {DOCKER_ARGS}", docker_bash_setup_script=SETUP,
                       packages=["clearml", "boto3<1.43"])
    task.output_uri = FILE_STORE
    if a.commit:
        task.set_script(commit=a.commit)
    task.set_parameters({"Args/MODEL": a.model, "Args/TAG": a.tag, "Args/KEEP": a.keep,
                         "Args/OUT_DIR": "/tmp/decision_runs", "Args/EPOCHS": a.epochs,
                         "Args/BATCH": a.batch, "Args/LR": a.lr, "Args/GROUP": a.group,
                         "Args/WARMUP": a.warmup, "Args/W_CAL": a.w_cal,
                         "Args/W_ENTROPY": a.w_entropy, "Args/KAPPA": a.kappa,
                         "Args/C_ASK": a.c_ask, "Args/TAU_STOP": a.tau_stop,
                         "Args/TAU_ASK": a.tau_ask, "Args/FOLDS": a.folds,
                         "Args/FOLD_LIMIT": a.fold_limit, "Args/SFT_TASK": a.sft_task,
                         "Args/DATA_TASK": a.data_task})
    print(f"Created task {task.id}")
    Task.enqueue(task, queue_name=a.queue)
    print(f"Enqueued to '{a.queue}'")


if __name__ == "__main__":
    main()
