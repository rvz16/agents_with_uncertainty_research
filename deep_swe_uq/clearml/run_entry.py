#!/usr/bin/env python3
"""ClearML entry for a full DeepSWE run."""

import os
import subprocess
import sys
from pathlib import Path

from clearml import Task

FILE_STORE = "https://files.clearai.innopolis.university"


def main() -> int:
    task = Task.current_task() or Task.init(project_name="agentic-uq", task_name="deepswe-run")
    task.output_uri = FILE_STORE
    for key, value in (task.get_parameters_as_dict().get("Args", {}) or {}).items():
        if value not in (None, ""):
            os.environ[key] = str(value)

    repo = Path(__file__).resolve().parents[2]
    script = repo / "deep_swe_uq" / "clearml" / "run.sh"
    run_root = Path(os.environ.setdefault("RUN_ROOT", "/tmp/deepswe_runs"))
    rc = subprocess.call(["bash", str(script)], cwd=str(repo))
    print(f"[entry] run rc={rc}", flush=True)

    # A worker whose card is unusable (rc 25: MIG mode with no device, someone
    # else's memory) or whose disk cannot hold the weights (rc 30) is idle
    # precisely because nothing can run on it, so it is the first to take the
    # next task from the queue. Re-enqueue a copy and keep this worker busy
    # for a while, so that a healthy worker gets a chance to pick the copy up.
    if rc in (25, 30):
        left = int(os.environ.get("REQUEUE_LEFT", "12") or 0)
        if left > 0:
            import time
            queue = task.data.execution.queue
            clone = Task.clone(source_task=task, name=task.name)
            clone.set_parameter("Args/REQUEUE_LEFT", str(left - 1))
            Task.enqueue(clone, queue_id=queue)
            print(f"[entry] card unusable here; re-enqueued as {clone.id} ({left - 1} retries left), holding this worker for 15 min", flush=True)
            time.sleep(900)

    # Upload only the job results: the shared directory also holds the cloned
    # task repository, which the host daemon needs but nobody needs afterwards.
    jobs = run_root / "jobs"
    if jobs.exists() and any(jobs.iterdir()):
        task.upload_artifact("run_root", artifact_object=jobs, wait_on_upload=True)
        print(f"[entry] uploaded {jobs}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
