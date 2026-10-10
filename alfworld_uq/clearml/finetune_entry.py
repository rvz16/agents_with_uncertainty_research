#!/usr/bin/env python3
"""ClearML entry for the decision-model baseline: serve Jeeves, ask, upload."""

import os
import subprocess
import sys
from pathlib import Path

from clearml import Task

FILE_STORE = "https://files.clearai.innopolis.university"


def main() -> int:
    task = Task.current_task() or Task.init(project_name="agentic-uq", task_name="finetune-outcome")
    task.output_uri = FILE_STORE
    for key, value in (task.get_parameters_as_dict().get("Args", {}) or {}).items():
        if value not in (None, ""):
            os.environ[key] = str(value)

    repo = Path(__file__).resolve().parents[2]
    out_dir = Path(os.environ.setdefault("OUT_DIR", "/tmp/finetune_runs"))
    rc = subprocess.call(["bash", str(repo / "alfworld_uq" / "clearml" / "finetune.sh")], cwd=str(repo))
    print(f"[entry] finetune rc={rc}", flush=True)

    for answers in sorted(out_dir.glob("*.jsonl")):
        if answers.stat().st_size > 0:
            task.upload_artifact(answers.stem, artifact_object=answers, wait_on_upload=True)
            print(f"[entry] uploaded {answers} ({answers.stat().st_size} bytes)", flush=True)
    log = out_dir / "serve.log"
    if log.exists():
        task.upload_artifact("serve_log", artifact_object=log, wait_on_upload=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
