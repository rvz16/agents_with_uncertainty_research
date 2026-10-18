#!/usr/bin/env python3
"""Enqueue the re-scoring that recovers DeepSWE's per-step uncertainty signals."""
from __future__ import annotations

import argparse

from clearml import Task

REPO = "https://github.com/rvz16/agents_with_uncertainty_research.git"
FILE_STORE = "https://files.clearai.innopolis.university"
DOCKER_IMAGE = "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"
DOCKER_ARGS = "--entrypoint= --network=host --shm-size=8g -v /tmp/signal_runs:/tmp/signal_runs"
SETUP = "pip install -q -U transformers accelerate 2>&1 | tail -2 || true"
#: the executor whose generations we are re-scoring, per cohort
MODELS = {"DG": "openai/gpt-oss-20b", "DL": "openai/gpt-oss-120b",
          "DQ": "Qwen/Qwen3.6-35B-A3B"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", default="high_q_80")
    p.add_argument("--branch", default="decision-models-audit")
    p.add_argument("--commit", default=None)
    p.add_argument("--cohort", default="DG", choices=[*MODELS])
    p.add_argument("--model", default=None, help="overrides the cohort's executor")
    p.add_argument("--top-k", default="20")
    p.add_argument("--limit", default="", help="episodes, for a smoke run")
    p.add_argument("--name", default=None)
    a = p.parse_args()

    model = a.model or MODELS[a.cohort]
    name = a.name or f"signals {a.cohort}: re-score actions with {model.split('/')[-1]}"
    task = Task.create(project_name="agentic-uq", task_name=name, repo=REPO, branch=a.branch,
                       script="deep_swe_uq/clearml/score_signals_entry.py",
                       docker=f"{DOCKER_IMAGE} {DOCKER_ARGS}", docker_bash_setup_script=SETUP,
                       packages=["clearml", "boto3<1.43", "transformers", "accelerate"])
    task.output_uri = FILE_STORE
    if a.commit:
        task.set_script(commit=a.commit)
    task.set_parameters({"Args/COHORT": a.cohort, "Args/MODEL": model,
                         "Args/TOP_K": a.top_k, "Args/LIMIT": a.limit,
                         "Args/OUT_DIR": "/tmp/signal_runs"})
    print(f"Created task {task.id}")
    Task.enqueue(task, queue_name=a.queue)
    print(f"Enqueued to '{a.queue}'")


if __name__ == "__main__":
    main()
