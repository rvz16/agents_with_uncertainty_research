#!/usr/bin/env python3
"""ClearML entry for the DeepSWE feasibility probe."""

import os
import subprocess
import sys
from pathlib import Path

from clearml import Task

FILE_STORE = "https://files.clearai.innopolis.university"


def main() -> int:
    task = Task.current_task() or Task.init(project_name="agentic-uq", task_name="deepswe-probe")
    task.output_uri = FILE_STORE
    for key, value in (task.get_parameters_as_dict().get("Args", {}) or {}).items():
        if value not in (None, ""):
            os.environ[key] = str(value)

    repo = Path(__file__).resolve().parents[2]
    script = repo / "deep_swe_uq" / "clearml" / "probe.sh"
    run_root = Path(os.environ.setdefault("RUN_ROOT", "/tmp/probe_runs"))
    rc = subprocess.call(["bash", str(script)], cwd=str(repo))
    print(f"[entry] probe rc={rc}", flush=True)

    if run_root.exists() and any(run_root.iterdir()):
        task.upload_artifact("probe_runs", artifact_object=run_root, wait_on_upload=True)
        print(f"[entry] uploaded {run_root}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
