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

    # Upload only the job results: the shared directory also holds the cloned
    # task repository, which the host daemon needs but nobody needs afterwards.
    jobs = run_root / "jobs"
    if jobs.exists() and any(jobs.iterdir()):
        task.upload_artifact("run_root", artifact_object=jobs, wait_on_upload=True)
        print(f"[entry] uploaded {jobs}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
