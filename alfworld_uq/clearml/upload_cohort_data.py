#!/usr/bin/env python3
"""Upload cohort training inputs to the File Store, for runs that cannot carry them in git.

The 45 MB of code-cohort views and targets will not push over HTTPS from here
(every attempt from 3.6 MB up returns HTTP 408), so they travel as artifacts of
one task instead and the training entry fetches them by task id. Artifact names
are "<directory>__<stem>", which is how the entry decides where each file lands.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from clearml import Task

DIRS = ("judge_views", "distill_targets")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--cohorts", default="CB,CC,CL,CS",
                   help="comma-separated cohort codes to upload")
    p.add_argument("--name", default="code cohort training inputs")
    a = p.parse_args()

    task = Task.init(project_name="agentic-uq", task_name=a.name,
                     task_type=Task.TaskTypes.data_processing, reuse_last_task_id=False)
    task.output_uri = "https://files.clearai.innopolis.university"
    sent = 0
    for cohort in a.cohorts.split(","):
        for directory in DIRS:
            stem = cohort if directory == "judge_views" else f"targets_{cohort}"
            path = a.data / directory / f"{stem}.jsonl"
            if not path.exists():
                print(f"[upload] {path} missing, skipped", flush=True)
                continue
            task.upload_artifact(f"{directory}__{stem}", artifact_object=path,
                                 wait_on_upload=True)
            print(f"[upload] {directory}/{stem}.jsonl "
                  f"({path.stat().st_size / 1e6:.1f} MB)", flush=True)
            sent += 1
    print(f"[upload] {sent} artifacts -> task {task.id}", flush=True)
    print(f"DATA_TASK={task.id}", flush=True)


if __name__ == "__main__":
    main()
