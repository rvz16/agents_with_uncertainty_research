#!/usr/bin/env python3
"""ClearML entry for grading DeepSWE patches outside pier."""
import os
import subprocess
import sys
from pathlib import Path

from clearml import Task

FILE_STORE = "https://files.clearai.innopolis.university"


def main() -> int:
    task = Task.current_task() or Task.init(project_name="agentic-uq", task_name="deepswe-grade")
    task.output_uri = FILE_STORE
    for key, value in (task.get_parameters_as_dict().get("Args", {}) or {}).items():
        if value not in (None, ""):
            os.environ[key] = str(value)
    repo = Path(__file__).resolve().parents[2]
    out_dir = Path(os.environ.setdefault("OUT_DIR", "/tmp/grade_runs"))
    rc = subprocess.call(["bash", str(repo / "deep_swe_uq" / "clearml" / "grade.sh")], cwd=str(repo))
    print(f"[entry] grade rc={rc}", flush=True)
    for f in sorted(out_dir.glob("*.jsonl")):
        if f.stat().st_size:
            task.upload_artifact(f.stem, artifact_object=f, wait_on_upload=True)
            print(f"[entry] uploaded {f.name} ({f.stat().st_size} bytes)", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
