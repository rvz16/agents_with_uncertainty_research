#!/usr/bin/env python
"""ClearML task entrypoint for a clean code-UQ generation run.

Runs scripts/run_clean_uq_clearml.sh (serve gpt-oss + clean generation), then
uploads the whole RUN_ROOT (JSONL logprobs, verbalized, belief, readable CSVs)
as ClearML artifacts so we can pull them and run all extra UQ analysis locally.

Task-level parameters (override in the ClearML UI before enqueue) map to env vars
consumed by the wrapper: BENCHMARKS, N_INSTANCES, MAX_VERIFICATIONS,
PRIVATE_TEST_CAP, MAX_GENERATIONS, OPENROUTER_API_KEY (optional, for L3 critic).
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
RUN_ROOT = Path(
    os.environ.get("RUN_ROOT", REPO_DIR / "runs/code_uq_clean/gpt_oss_20b_local")
)


def _apply_clearml_params() -> None:
    """Pull Args/* task parameters into the environment for the bash wrapper."""
    try:
        from clearml import Task
    except Exception:
        return
    task = Task.current_task()
    if task is None:
        return
    params = task.get_parameters() or {}
    for key, val in params.items():
        name = key.split("/", 1)[-1]  # strip "Args/" section prefix
        if val not in (None, "") and name.isupper():
            os.environ[name] = str(val)
            print(f"[entry] param {name}={val}", flush=True)


def main() -> int:
    _apply_clearml_params()
    env = dict(os.environ, REPO_DIR=str(REPO_DIR))

    rc = subprocess.run(
        ["bash", str(REPO_DIR / "scripts/run_clean_uq_clearml.sh")],
        env=env,
        cwd=str(REPO_DIR),
    ).returncode

    # Upload artifacts regardless of rc so a partial run is still inspectable.
    try:
        from clearml import Task

        task = Task.current_task()
        if task is not None and RUN_ROOT.exists():
            task.upload_artifact("run_root", artifact_object=str(RUN_ROOT))
            vlog = REPO_DIR / "vllm_serve.log"
            if vlog.exists():
                task.upload_artifact("vllm_serve_log", artifact_object=str(vlog))
            print(f"[entry] uploaded artifacts from {RUN_ROOT}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[entry] artifact upload failed: {exc}", flush=True)

    return rc


if __name__ == "__main__":
    sys.exit(main())
