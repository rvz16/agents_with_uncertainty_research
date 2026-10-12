#!/usr/bin/env python3
"""ClearML entry for training our own decision model."""
import os
import subprocess
import sys
from pathlib import Path

from clearml import Task

FILE_STORE = "https://files.clearai.innopolis.university"


def main() -> int:
    task = Task.current_task() or Task.init(project_name="agentic-uq", task_name="train-decision")
    task.output_uri = FILE_STORE
    for key, value in (task.get_parameters_as_dict().get("Args", {}) or {}).items():
        if value not in (None, ""):
            os.environ[key] = str(value)
    repo = Path(__file__).resolve().parents[2]
    out_dir = Path(os.environ.setdefault("OUT_DIR", "/tmp/decision_runs"))
    rc = subprocess.call(["bash", str(repo / "alfworld_uq" / "clearml" / "train_decision.sh")],
                         cwd=str(repo))
    print(f"[entry] train rc={rc}", flush=True)
    for f in sorted(list(out_dir.glob("answers_*.jsonl")) + list(out_dir.glob("steps_*.jsonl"))):
        if f.stat().st_size:
            task.upload_artifact(f.stem, artifact_object=f, wait_on_upload=True)
            print(f"[entry] uploaded {f.name} ({f.stat().st_size} bytes)", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
