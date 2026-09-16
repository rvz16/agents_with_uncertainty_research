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

    # Collect mode: no agent run. A finished run whose archive never uploaded
    # (a 6.4 GB zip of the whole shared jobs directory was refused) is reduced
    # on the host to its compact per-step records and those are uploaded.
    if os.environ.get("COLLECT_ONLY") == "1":
        name = os.environ["RUN_NAME"]; jobs = run_root / "jobs"
        out = Path("/tmp") / f"compact_{name}.jsonl"
        rc = subprocess.call([sys.executable, str(repo / "deep_swe_uq" / "experiments" / "compact_jobs.py"),
                              str(jobs), "--run", name, "--out", str(out)], cwd=str(repo))
        print(f"[entry] compact rc={rc}", flush=True)
        if out.exists():
            task.upload_artifact("compact", artifact_object=out, wait_on_upload=True)
            print(f"[entry] uploaded {out} ({out.stat().st_size} bytes)", flush=True)
        for extra in sorted(jobs.glob("verb_*.jsonl")):
            if extra.stat().st_size > 0:
                task.upload_artifact(extra.stem, artifact_object=extra, wait_on_upload=True)
                print(f"[entry] uploaded {extra}", flush=True)
        return rc

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
    replay = os.environ.get("REPLAY_RUN", "")
    if replay:
        # replay mode writes one small jsonl; the shared jobs directory on the
        # host holds every earlier run and would be a multi-GB upload
        out = jobs / f"verb_{replay}.jsonl"
        if out.exists():
            task.upload_artifact("verb", artifact_object=out, wait_on_upload=True)
            print(f"[entry] uploaded {out}", flush=True)
    elif jobs.exists() and any(jobs.iterdir()):
        # only this run's directory: the shared jobs directory on the host
        # holds every earlier run and grows past what the file server takes
        mine = jobs / os.environ.get("RUN_NAME", "")
        target = mine if mine.is_dir() else jobs
        task.upload_artifact("run_root", artifact_object=target, wait_on_upload=True)
        print(f"[entry] uploaded {target}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
