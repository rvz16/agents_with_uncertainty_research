#!/usr/bin/env python3
"""ClearML task entrypoint for an ALFWorld run: run the pipeline, upload the run.

Artifacts go to the ClearML File Store rather than the s3 bucket: the bucket is
full and returns XMinioStorageFull, which surfaces only after the run finishes
and loses hours of GPU time at the last step.
"""

# No `from __future__ import annotations` here: ClearML prepends its own code to
# the entry script, so the __future__ import would stop being the first
# statement and the task would die with SyntaxError.

import os
import subprocess
import sys
from pathlib import Path

from clearml import Task

FILE_STORE = "https://files.clearai.innopolis.university"


def main() -> int:
    task = Task.current_task() or Task.init(
        project_name="agentic-uq", task_name="alfworld-smolagents"
    )
    task.output_uri = FILE_STORE

    params = task.get_parameters_as_dict().get("Args", {}) or {}
    for key, value in params.items():
        if value not in (None, ""):
            os.environ[key] = str(value)
            print(f"[entry] param {key}={value}", flush=True)

    repo = Path(__file__).resolve().parents[2]
    project = repo / "alfworld_uq"
    run_name = os.environ.get("RUN_NAME") or f"alfworld_{os.environ.get('POLICY', 'smolagents')}"
    run_root = Path(os.environ.setdefault("RUN_ROOT", str(project / "runs" / run_name)))
    os.environ["REPO_DIR"] = str(repo)

    script = project / "clearml" / "run_in_container.sh"
    print(f"[entry] repo={repo}", flush=True)
    print(f"[entry] run_root={run_root}", flush=True)
    rc = subprocess.call(["bash", str(script)], cwd=str(project))
    print(f"[entry] pipeline rc={rc}", flush=True)

    # Upload whatever exists even on failure: a partial run still carries the
    # trajectories and log-probabilities already written, and re-running costs
    # hours of GPU time.
    if run_root.exists():
        task.upload_artifact("run_root", artifact_object=run_root, wait_on_upload=True)
        print(f"[entry] uploaded {run_root}", flush=True)
    else:
        print(f"[entry] nothing to upload: {run_root} does not exist", flush=True)

    # The engine's real error scrolls out of the console's rolling window, and
    # the container is gone by the time anyone reads the task, so keep the file.
    serve_log = project / "vllm_serve.log"
    if serve_log.exists():
        task.upload_artifact("vllm_serve_log", artifact_object=serve_log, wait_on_upload=True)
        print(f"[entry] uploaded {serve_log}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
