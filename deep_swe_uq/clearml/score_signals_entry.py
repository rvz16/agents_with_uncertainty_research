#!/usr/bin/env python3
"""ClearML entry for re-scoring DeepSWE actions into per-step signals."""
import os
import subprocess
import sys
from pathlib import Path

from clearml import Task

FILE_STORE = "https://files.clearai.innopolis.university"


def main() -> int:
    task = Task.current_task() or Task.init(project_name="agentic-uq", task_name="score-signals")
    task.output_uri = FILE_STORE
    for key, value in (task.get_parameters_as_dict().get("Args", {}) or {}).items():
        if value not in (None, ""):
            os.environ[key] = str(value)
    repo = Path(__file__).resolve().parents[2]
    # own directory per task: OUT_DIR is mounted from the host and shared, and
    # uploading the parent once shipped another run's files and none of this one's
    out_dir = Path(os.environ.get("OUT_DIR") or "/tmp/signal_runs") / task.id
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OUT_DIR"] = str(out_dir)
    rc = subprocess.call(["bash", str(repo / "deep_swe_uq" / "clearml" / "score_signals.sh")],
                         cwd=str(repo))
    print(f"[entry] score rc={rc}", flush=True)
    found = sorted(out_dir.glob("compact_*.jsonl"))
    if not found:
        print(f"[entry] WARNING: nothing in {out_dir}", flush=True)
    for f in found:
        if f.stat().st_size:
            task.upload_artifact(f.stem, artifact_object=f, wait_on_upload=True)
            print(f"[entry] uploaded {f.name} ({f.stat().st_size} bytes)", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
