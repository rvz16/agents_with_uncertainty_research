#!/usr/bin/env python3
"""Reclaim this project's disk on a worker, before anything that needs a clone.

The cleanup is inlined into the docker setup script rather than run from the
repository, because the thing it fixes is a full disk and a full disk is exactly
what stops the repository from being cloned: the DG shards died with ENOSPC while
git was unpacking experiments/orchestration_hypothesis_testing/data, which carries
committed SWE-bench evaluation logs. The setup script runs first, so the clone
that follows has room.
"""
from __future__ import annotations

import argparse
from base64 import b64encode
from pathlib import Path

from clearml import Task

REPO = "https://github.com/rvz16/agents_with_uncertainty_research.git"
DOCKER_IMAGE = "python:3.12"
DOCKER_ARGS = ("--entrypoint= --network=host "
               "-v /var/run/docker.sock:/var/run/docker.sock "
               "-v /tmp/grade_shared:/tmp/grade_shared -v /tmp/grade_runs:/tmp/grade_runs")
HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", default="high_q")
    p.add_argument("--branch", default="decision-models-audit")
    p.add_argument("--commit", default=None)
    p.add_argument("--report-only", action="store_true",
                   help="print the disk and what is ours, remove nothing")
    p.add_argument("--name", default="deepswe: reclaim our disk on the worker")
    a = p.parse_args()

    # ClearML joins the setup script's lines with " ; " into one bash -c, which
    # destroys every multi-line construct: a heredoc, an if block and a for loop
    # all came back as "syntax error near unexpected token ';'". So each line here
    # has to stand alone, and the scripts travel base64-encoded. They are read
    # from the committed files, so there is one source of truth.
    def deliver(name: str) -> str:
        blob = b64encode((HERE / name).read_bytes()).decode()
        return f"echo {blob} | base64 -d > /tmp/{name}"

    setup = " \n".join([
        "apt-get install -y -qq --no-install-recommends curl >/dev/null 2>&1 || true",
        f"export DO_CLEAN={'0' if a.report_only else '1'}",
        deliver("docker_client.sh"),
        deliver("reclaim_disk.sh"),
        # both land in /tmp, so reclaim_disk.sh finds its sibling where it looks
        "bash /tmp/reclaim_disk.sh || true",
    ]).replace(" \n", "\n")

    task = Task.create(project_name="agentic-uq", task_name=a.name, repo=REPO, branch=a.branch,
                       script="deep_swe_uq/clearml/grade_entry.py",
                       docker=f"{DOCKER_IMAGE} {DOCKER_ARGS}", docker_bash_setup_script=setup,
                       packages=["clearml", "boto3<1.43"])
    if a.commit:
        task.set_script(commit=a.commit)
    # the entry still runs; validate on nothing is a no-op that proves the clone works
    task.set_parameters({"Args/COMMAND": "validate", "Args/LIMIT": "1",
                         "Args/OUT_DIR": "/tmp/grade_runs", "Args/SHARED": "/tmp/grade_shared"})
    print(f"Created task {task.id}")
    Task.enqueue(task, queue_name=a.queue)
    print(f"Enqueued to '{a.queue}'")


if __name__ == "__main__":
    main()
