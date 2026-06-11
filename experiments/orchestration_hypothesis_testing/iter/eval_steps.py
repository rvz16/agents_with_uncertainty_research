"""Run SWE-bench harness on the iterative refinement predictions.

Wraps spot_check_generators.run_swebench_eval (which includes the podman
compat shim) and runs it for steps 1..N-1 of one generator.

Usage:
  python3 eval_iter_steps.py <generator> [n_steps]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path("/mnt/data/users/vlad.smirnov/agents_with_uncertainty_research/.claude/worktrees/reverent-vaughan-017bf5/experiments/orchestration_hypothesis_testing")
sys.path.insert(0, str(ROOT / "scripts"))

# Force podman shim env vars (mirrors launch_v3.sh)
os.environ.setdefault("DOCKER_HOST", f"unix:///run/user/{os.geteuid()}/podman/podman.sock")
os.environ["SWEBENCH_PODMAN_COMPAT"] = "1"

import spot_check_generators as scg  # noqa: E402

DATA = ROOT / "data" / "spot_check_n50"

gen = sys.argv[1]
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
# Default max_workers=1 to avoid the containerd metadata race that produces
# silent "Docker API timeout" failures on 100+ instances when 2+ workers
# claim images concurrently. Override via SWE_EVAL_MAX_WORKERS=N env var.
max_workers = int(os.environ.get("SWE_EVAL_MAX_WORKERS", "1"))
work_dir = DATA / "eval"
work_dir.mkdir(parents=True, exist_ok=True)

print(f"eval config: gen={gen}  n_steps={n_steps}  max_workers={max_workers}")

for step in range(1, n_steps):
    pred_path = DATA / gen / f"predictions_iter_step{step}.jsonl"
    if not pred_path.exists():
        print(f"skip step {step}: {pred_path} not found")
        continue
    run_id = f"{gen}_iter_step{step}"
    print(f"\n==== eval {gen} step {step} (run_id={run_id}) ====")
    try:
        report_path = scg.run_swebench_eval(
            predictions_path=pred_path,
            run_id=run_id,
            max_workers=max_workers,
            work_dir=work_dir,
        )
        print(f"  -> {report_path.name}")
    except Exception as e:
        print(f"  ERROR: {e}")
