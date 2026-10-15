#!/usr/bin/env python3
"""ClearML entry for the decision-model baseline: serve Jeeves, ask, upload."""

import os
import subprocess
import sys
from pathlib import Path

from clearml import Task

FILE_STORE = "https://files.clearai.innopolis.university"


def main() -> int:
    task = Task.current_task() or Task.init(project_name="agentic-uq", task_name="decision-judge")
    task.output_uri = FILE_STORE
    for key, value in (task.get_parameters_as_dict().get("Args", {}) or {}).items():
        if value not in (None, ""):
            os.environ[key] = str(value)

    repo = Path(__file__).resolve().parents[2]
    # OUT_DIR is mounted from the host and shared by every task on the worker, so
    # give this run its own directory. Uploading the parent instead once shipped
    # eight files from an earlier run and none of this one's.
    out_dir = Path(os.environ.get("OUT_DIR") or "/tmp/decision_judge") / task.id
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OUT_DIR"] = str(out_dir)
    rc = subprocess.call(["bash", str(repo / "alfworld_uq" / "clearml" / "decision_judge.sh")], cwd=str(repo))
    print(f"[entry] judge rc={rc}", flush=True)

    # the gate schema writes gate_*.jsonl, not judge_*.jsonl: globbing one prefix
    # silently dropped the whole run
    answers_files = sorted(f for pattern in ("judge_*.jsonl", "gate_*.jsonl")
                           for f in out_dir.glob(pattern))
    if not answers_files:
        print(f"[entry] WARNING: no answer files in {out_dir}", flush=True)
    for answers in answers_files:
        if answers.stat().st_size > 0:
            task.upload_artifact(answers.stem, artifact_object=answers, wait_on_upload=True)
            print(f"[entry] uploaded {answers} ({answers.stat().st_size} bytes)", flush=True)
    log = out_dir / "serve.log"
    if log.exists():
        task.upload_artifact("serve_log", artifact_object=log, wait_on_upload=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
