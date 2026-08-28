#!/usr/bin/env python3
"""ClearML task entrypoint: run the pipeline, upload the run root.

Artifacts go to the ClearML File Store rather than the s3 bucket: the bucket
(api.blackhole2.../clearml-example) is full and returns XMinioStorageFull, which
surfaces only after the run finishes — losing hours of GPU time at the last step.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from clearml import Task

FILE_STORE = "https://files.clearai.innopolis.university"


def main() -> int:
    task = Task.current_task() or Task.init(project_name="agentic-uq", task_name="sage-uq")
    task.output_uri = FILE_STORE

    params = task.get_parameters_as_dict().get("Args", {}) or {}
    for key, value in params.items():
        if value not in (None, ""):
            os.environ[key] = str(value)

    repo = Path(__file__).resolve().parents[4]
    generator = os.environ.get("GENERATOR_KEY", "gpt_oss_20b_local")
    run_root = Path(os.environ.setdefault("RUN_ROOT", str(repo / "runs" / "sage_uq" / generator)))
    os.environ["REPO_DIR"] = str(repo)

    script = repo / "experiments/orchestration_hypothesis_testing/scripts/clearml/run_in_container.sh"
    print(f"[entry] repo={repo}", flush=True)
    print(f"[entry] run_root={run_root}", flush=True)
    rc = subprocess.call(["bash", str(script)], cwd=str(repo))
    print(f"[entry] pipeline rc={rc}", flush=True)

    # Upload whatever exists even on failure: a partial run still carries the
    # trajectories and logprobs already written, and re-running costs hours.
    if run_root.exists():
        task.upload_artifact("run_root", artifact_object=run_root, wait_on_upload=True)
        print(f"[entry] uploaded {run_root}", flush=True)
    else:
        print(f"[entry] nothing to upload: {run_root} does not exist", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
