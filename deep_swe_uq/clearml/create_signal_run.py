#!/usr/bin/env python3
"""Enqueue the re-scoring that recovers DeepSWE's per-step uncertainty signals."""
from __future__ import annotations

import argparse

from clearml import Task

REPO = "https://github.com/rvz16/agents_with_uncertainty_research.git"
FILE_STORE = "https://files.clearai.innopolis.university"
# A plain python image: vLLM comes from pip and the wheels carry their own CUDA
# runtime, which is how this project already served all three of these models.
DOCKER_IMAGE = "python:3.12"
DOCKER_ARGS = ("--entrypoint= --network=host --shm-size=8g "
               "-v /tmp/signal_runs:/tmp/signal_runs "
               "-v /root/.clearml/hf-cache:/root/.cache/huggingface")
SETUP = """
df -h /
rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list || true
apt-get update -qq -o Acquire::AllowInsecureRepositories=true || true
apt-get install -y -qq --no-install-recommends git curl || true
"""
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
    p.add_argument("--max-len", default="32768")
    p.add_argument("--tp", default="1", help="tensor parallel; the 120B needs 2")
    p.add_argument("--gpu-util", default="0.90")
    p.add_argument("--batch", default="8")
    p.add_argument("--limit", default="", help="episodes, for a smoke run")
    p.add_argument("--name", default=None)
    a = p.parse_args()

    model = a.model or MODELS[a.cohort]
    name = a.name or f"signals {a.cohort}: re-score actions with {model.split('/')[-1]}"
    task = Task.create(project_name="agentic-uq", task_name=name, repo=REPO, branch=a.branch,
                       script="deep_swe_uq/clearml/score_signals_entry.py",
                       docker=f"{DOCKER_IMAGE} {DOCKER_ARGS}", docker_bash_setup_script=SETUP,
                       packages=["clearml", "boto3<1.43"])
    task.output_uri = FILE_STORE
    if a.commit:
        task.set_script(commit=a.commit)
    task.set_parameters({"Args/COHORT": a.cohort, "Args/MODEL": model,
                         "Args/TOP_K": a.top_k, "Args/LIMIT": a.limit,
                         "Args/MAX_LEN": a.max_len, "Args/TP": a.tp,
                         "Args/GPU_UTIL": a.gpu_util, "Args/BATCH": a.batch,
                         "Args/OUT_DIR": "/tmp/signal_runs"})
    print(f"Created task {task.id}")
    Task.enqueue(task, queue_name=a.queue)
    print(f"Enqueued to '{a.queue}'")


if __name__ == "__main__":
    main()
