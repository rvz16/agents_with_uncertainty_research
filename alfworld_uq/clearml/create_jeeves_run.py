#!/usr/bin/env python3
"""Enqueue the decision-model baseline (PostHog Jeeves) on a ClearML agent.

The model is served inside the task: the cluster answers hosted endpoints with
HTTP 403, and the weights are 9B in bf16, which no laptop here holds. The
trajectory states travel with the repo, so the only download is the checkpoint.
"""
from __future__ import annotations

import argparse

from clearml import Task

REPO = "https://github.com/rvz16/agents_with_uncertainty_research.git"
FILE_STORE = "https://files.clearai.innopolis.university"
DOCKER_IMAGE = "python:3.12"
DOCKER_ARGS = (
    "--entrypoint= --network=host "
    "-v /tmp/decision_runs:/tmp/decision_runs "
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
    p.add_argument("--name", default="jeeves decision baseline")
    p.add_argument("--queue", default="high_q_80")
    p.add_argument("--branch", default="alfworld_smolagents")
    p.add_argument("--commit", default=None, help="commit to pin; the branch head otherwise")
    p.add_argument("--model-repo", default="PostHog/jeeves")
    p.add_argument("--max-len", type=int, default=8192,
                   help="the longest state is ~1.6k tokens, so the default cache is ample")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-think", type=int, default=768)
    p.add_argument("--states-dir", default="alfworld_uq/data/decision_states",
                   help="decision_states (every recorded step) or decision_states_clean (giveup+acted)")
    p.add_argument("--think-modes", default="nothink think",
                   help="nothink answers in ~0.3 s per episode, think in ~3 s")
    a = p.parse_args()

    task = Task.create(
        project_name=a.project,
        task_name=a.name,
        repo=REPO,
        branch=a.branch,
        script="alfworld_uq/clearml/jeeves_entry.py",
        docker=f"{DOCKER_IMAGE} {DOCKER_ARGS}",
        docker_bash_setup_script=SETUP,
        packages=["clearml", "boto3<1.43"],
    )
    task.output_uri = FILE_STORE
    if a.commit:
        task.set_script(commit=a.commit)
    task.set_parameters({
        "Args/MODEL_REPO": a.model_repo,
        "Args/OUT_DIR": "/tmp/decision_runs",
        "Args/MAX_LEN": str(a.max_len),
        "Args/WORKERS": str(a.workers),
        "Args/MAX_THINK": str(a.max_think),
        "Args/THINK_MODES": a.think_modes,
        "Args/STATES_DIR": a.states_dir,
    })
    print(f"Created task {task.id}")
    Task.enqueue(task, queue_name=a.queue)
    print(f"Enqueued to '{a.queue}'")


if __name__ == "__main__":
    main()
