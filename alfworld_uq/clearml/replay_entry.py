#!/usr/bin/env python3
"""ClearML entry for the per-step label replay."""

import os
import subprocess
import sys
from pathlib import Path

from clearml import Task

FILE_STORE = "https://files.clearai.innopolis.university"


def main() -> int:
    task = Task.current_task() or Task.init(project_name="agentic-uq", task_name="replay-labels")
    task.output_uri = FILE_STORE
    for key, value in (task.get_parameters_as_dict().get("Args", {}) or {}).items():
        if value not in (None, ""):
            os.environ[key] = str(value)

    repo = Path(__file__).resolve().parents[2]
    # own directory per task: OUT_DIR is mounted from the host and shared, and
    # uploading the parent once shipped another run's files and none of this one's
    out_dir = Path(os.environ.get("OUT_DIR") or "/tmp/replay_runs") / task.id
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OUT_DIR"] = str(out_dir)
    rc = subprocess.call(["bash", str(repo / "alfworld_uq" / "clearml" / "replay.sh")], cwd=str(repo))
    print(f"[entry] replay rc={rc}", flush=True)
    found = sorted(out_dir.glob("labels_*.jsonl"))
    if not found:
        print(f"[entry] WARNING: nothing in {out_dir}", flush=True)
    for labels in found:
        if labels.stat().st_size > 0:
            task.upload_artifact(labels.stem, artifact_object=labels, wait_on_upload=True)
            print(f"[entry] uploaded {labels.name} ({labels.stat().st_size} bytes)", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
