#!/usr/bin/env python
import os
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
RUN_ROOT = Path(
    os.environ.get("RUN_ROOT", REPO_DIR / "runs/code_uq_clean/gpt_oss_20b_local")
)
# ClearML File Store (NOT the full s3 bucket) -> artifacts land here.
FILE_STORE = os.environ.get(
    "CLEARML_FILES_URI", "https://files.clearai.innopolis.university"
)


def main() -> int:
    try:
        from clearml import Task
        task = Task.current_task()
    except Exception:
        task = None
    if task is not None:
        try:
            task.output_uri = FILE_STORE
        except Exception as exc:
            print(f"[entry] set output_uri failed: {exc}", flush=True)
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

    if task is not None and RUN_ROOT.exists():
        try:
            task.upload_artifact("run_root", artifact_object=str(RUN_ROOT))
            print(f"[entry] uploaded run_root -> {FILE_STORE}", flush=True)
        except Exception as exc:
            print(f"[entry] run_root upload failed: {exc}", flush=True)

    print(f"[entry] run finished rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
