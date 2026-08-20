#!/usr/bin/env python
import os
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        from clearml import Task
        task = Task.current_task()
    except Exception:
        task = None
    if task is not None:
        try:
            task.output_uri = False
        except Exception:
            pass
        for key, val in (task.get_parameters() or {}).items():
            name = key.split("/", 1)[-1]
            if val not in (None, "") and name.isupper():
                os.environ[name] = str(val)
                print(f"[entry] param {name}={val}", flush=True)
    env = dict(os.environ, REPO_DIR=str(REPO_DIR))
    rc = subprocess.run(
        ["bash", str(REPO_DIR / "scripts/run_clean_uq_clearml.sh")],
        env=env, cwd=str(REPO_DIR),
    ).returncode
    print(f"[entry] run finished rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
