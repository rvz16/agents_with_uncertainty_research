#!/usr/bin/env python3
"""ClearML entry for the pier capability probe."""
import subprocess
import sys
from pathlib import Path

from clearml import Task


def main() -> int:
    Task.current_task() or Task.init(project_name="agentic-uq", task_name="pier-grade-probe")
    repo = Path(__file__).resolve().parents[2]
    rc = subprocess.call(["bash", str(repo / "deep_swe_uq" / "clearml" / "grade_probe.sh")],
                         cwd=str(repo))
    print(f"[entry] probe rc={rc}", flush=True)
    return 0  # the point is the log, not the exit code


if __name__ == "__main__":
    sys.exit(main())
