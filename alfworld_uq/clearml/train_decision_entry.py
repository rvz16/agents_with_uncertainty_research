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
    # Cohort inputs that could not be pushed to the branch travel as artifacts of
    # a data task instead (see clearml/upload_cohort_data.py). Artifact names are
    # "<directory>__<stem>", so each file lands beside the cohorts git does carry.
    data_task = os.environ.get("DATA_TASK", "").strip()
    if data_task:
        source = Task.get_task(task_id=data_task)
        for name, artifact in (source.artifacts or {}).items():
            if "__" not in name:
                continue
            directory, stem = name.split("__", 1)
            target = repo / "alfworld_uq" / "data" / directory / f"{stem}.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            local = Path(artifact.get_local_copy())
            target.write_bytes(local.read_bytes())
            print(f"[entry] fetched {directory}/{stem}.jsonl "
                  f"({target.stat().st_size} bytes)", flush=True)
    # one directory per task: /tmp/decision_runs is mounted from the host, and a
    # previous job's files there were uploaded as this run's results once already
    out_dir = Path(os.environ.get("OUT_DIR", "/tmp/decision_runs")) / task.id
    os.environ["OUT_DIR"] = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = subprocess.call(["bash", str(repo / "alfworld_uq" / "clearml" / "train_decision.sh")],
                         cwd=str(repo))
    print(f"[entry] train rc={rc}", flush=True)
    for f in sorted(list(out_dir.glob("answers_*.jsonl")) + list(out_dir.glob("steps_*.jsonl"))):
        if f.stat().st_size:
            task.upload_artifact(f.stem, artifact_object=f, wait_on_upload=True)
            print(f"[entry] uploaded {f.name} ({f.stat().st_size} bytes)", flush=True)
    # the trainer uploads each fold itself, but a run that died after training
    # would still leave them on a worker nobody can reach; send the lot once more
    weights = out_dir / "weights"
    if weights.is_dir() and any(weights.iterdir()):
        task.upload_artifact("weights", artifact_object=weights, wait_on_upload=True)
        print(f"[entry] uploaded weights ({sum(1 for _ in weights.iterdir())} folds)", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
