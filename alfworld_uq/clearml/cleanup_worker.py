#!/usr/bin/env python3
"""Reclaim the disk our own runs left on a worker.

A node fills up and then refuses our jobs -- aiagent03 reached 12 GB free and
the ALFWorld probe was turned away by its own preflight. Much of that is ours:
DeepSWE keeps every trial's containers, logs and artifacts under
/tmp/deepswe_runs, and those have already been uploaded to ClearML.

Only paths this project created are touched. The model cache is reported but
never deleted: re-downloading 40 GB of weights to free 40 GB of disk is not a
trade worth making, and other runs depend on it.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

OURS = [
    Path("/tmp/deepswe_runs/jobs"),      # per-trial outputs, already uploaded
    Path("/tmp/probe_runs"),
]
REPORT_ONLY = [
    Path("/root/.clearml/hf-cache"),     # weights: expensive to lose
    Path("/root/.clearml/venvs-cache"),
    Path("/root/.clearml/pip-cache"),
]


def size_of(path: Path) -> str:
    if not path.exists():
        return "absent"
    out = subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True)
    return out.stdout.split()[0] if out.stdout else "?"


def df() -> str:
    out = subprocess.run(["df", "-h", "/"], capture_output=True, text=True)
    return out.stdout.strip().splitlines()[-1]


def main() -> int:
    from clearml import Task

    task = Task.current_task() or Task.init(
        project_name="agentic-uq", task_name="worker cleanup"
    )
    task.output_uri = "https://files.clearai.innopolis.university"

    print(f"[cleanup] worker: {os.environ.get('CLEARML_WORKER_ID', '?')}", flush=True)
    print(f"[cleanup] before: {df()}", flush=True)
    for path in REPORT_ONLY:
        print(f"[cleanup] keeping {path}: {size_of(path)}", flush=True)

    for path in OURS:
        if not path.exists():
            print(f"[cleanup] {path}: absent", flush=True)
            continue
        print(f"[cleanup] removing {path}: {size_of(path)}", flush=True)
        try:
            shutil.rmtree(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[cleanup] could not remove {path}: {exc}", flush=True)

    # Half-finished downloads are pure waste wherever they are.
    freed = 0
    cache = Path(os.environ.get("HF_HOME", "/root/.cache/huggingface"))
    if cache.exists():
        for blob in cache.rglob("*.incomplete"):
            try:
                freed += blob.stat().st_size
                blob.unlink()
            except Exception:  # noqa: BLE001
                pass
    print(f"[cleanup] removed {freed / 1e9:.1f} GB of incomplete downloads", flush=True)
    print(f"[cleanup] after:  {df()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
