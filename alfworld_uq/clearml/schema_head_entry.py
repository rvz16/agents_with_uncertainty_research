#!/usr/bin/env python3
"""ClearML entry for the schema head we train ourselves on a frozen backbone."""

import os
import subprocess
import sys
from pathlib import Path

from clearml import Task

FILE_STORE = "https://files.clearai.innopolis.university"


def main() -> int:
    task = Task.current_task() or Task.init(project_name="agentic-uq", task_name="schema-head")
    task.output_uri = FILE_STORE
    for key, value in (task.get_parameters_as_dict().get("Args", {}) or {}).items():
        if value not in (None, ""):
            os.environ[key] = str(value)

    repo = Path(__file__).resolve().parents[2]
    out_dir = Path(os.environ.setdefault("OUT_DIR", "/tmp/schema_runs"))
    rc = subprocess.call(["bash", str(repo / "alfworld_uq" / "clearml" / "schema_head.sh")], cwd=str(repo))
    print(f"[entry] schema head rc={rc}", flush=True)

    # the answer files are what the scorer reads; the cached features stay on the host
    for answers in sorted(out_dir.glob("answers_*.jsonl")):
        if answers.stat().st_size > 0:
            task.upload_artifact(answers.stem, artifact_object=answers, wait_on_upload=True)
            print(f"[entry] uploaded {answers.name} ({answers.stat().st_size} bytes)", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
