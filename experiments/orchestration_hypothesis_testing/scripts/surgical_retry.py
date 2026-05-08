"""Surgical retry for harness reports with err>0.

For each report with error_ids:
  1. Filter the original predictions.jsonl down to just the error_ids.
  2. Run swebench harness on that filtered set (--namespace none, max_workers=8).
  3. Merge results back into the original report:
       - resolved_ids: union with new resolved
       - unresolved_ids: union with new unresolved
       - error_ids: only the ones still erroring
       - update count fields

Saves ~80% of harness wall-time vs re-running the whole step file.

Usage:
  python3 surgical_retry.py <eval_report_path>
  python3 surgical_retry.py --auto    # find all err>0 reports and retry them
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

EXP_DIR = Path("/mnt/data/users/vlad.smirnov/agents_with_uncertainty_research/experiments/orchestration_hypothesis_testing")


def parse_run_id(report_path: Path) -> tuple[str, str]:
    """eval/<run_id>.<run_id>.json → (run_id, predictions_path inferred)."""
    name = report_path.name
    if not name.endswith(".json") or "." not in name[:-5]:
        raise SystemExit(f"unexpected report name: {name}")
    base = name[:-5]
    parts = base.split(".")
    run_id = parts[-1]
    return run_id


def find_predictions(report_path: Path, run_id: str) -> Path:
    """Walk up from eval/ to find the predictions file matching run_id step."""
    cell_dir = report_path.parent.parent
    if "_iter_step" in run_id:
        step = run_id.split("_iter_step")[1]
        return cell_dir / f"predictions_iter_step{step}.jsonl"
    if run_id.startswith(f"qwen25_32b_p"):
        pid = run_id.split("_p")[-1]
        return cell_dir / f"predictions_p{pid}.jsonl"
    raise SystemExit(f"can't infer predictions path for run_id={run_id}")


def retry_one(report_path: Path, dry_run: bool = False) -> dict:
    rep = json.loads(report_path.read_text())
    error_ids = list(rep.get("error_ids", []))
    if not error_ids:
        return {"status": "no_errors", "report": str(report_path)}

    run_id = parse_run_id(report_path)
    pred_path = find_predictions(report_path, run_id)
    if not pred_path.exists():
        return {"status": "no_predictions", "pred_path": str(pred_path)}

    # Determine dataset
    if "verified" in str(report_path):
        dataset = "princeton-nlp/SWE-bench_Verified"
    else:
        dataset = "princeton-nlp/SWE-bench_Lite"

    # Filter predictions to error_ids
    err_set = set(error_ids)
    filtered_path = pred_path.with_suffix(".retry.jsonl")
    n_filtered = 0
    with open(pred_path) as src, open(filtered_path, "w") as dst:
        for line in src:
            r = json.loads(line)
            if r["instance_id"] in err_set:
                dst.write(line)
                n_filtered += 1

    if n_filtered == 0:
        filtered_path.unlink()
        return {"status": "no_filtered_predictions"}

    if dry_run:
        filtered_path.unlink()
        return {"status": "DRY", "would_retry": n_filtered, "error_ids": error_ids[:3]}

    # Run harness
    retry_run_id = f"{run_id}_retry_{datetime.now().strftime('%H%M%S')}"
    eval_dir = report_path.parent
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] retrying {n_filtered} instances → {retry_run_id}")
    cmd = [
        "python3", "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset,
        "--predictions_path", str(filtered_path),
        "--max_workers", "8",
        "--run_id", retry_run_id,
        "--cache_level", "instance",
        "--namespace", "none",
    ]
    env = os.environ.copy()
    env["DOCKER_HOST"] = f"unix:///run/user/{os.getuid()}/podman/podman.sock"
    env["SWEBENCH_PODMAN_COMPAT"] = "1"
    log_path = eval_dir / f"surgical_retry_{retry_run_id}.log"
    with open(log_path, "w") as logf:
        rc = subprocess.call(cmd, cwd=str(eval_dir), stdout=logf, stderr=subprocess.STDOUT, env=env)

    # Find the retry's report and merge back
    retry_report = eval_dir / f"{retry_run_id}.{retry_run_id}.json"
    if not retry_report.exists():
        return {"status": "retry_no_report", "rc": rc, "log": str(log_path)}

    new = json.loads(retry_report.read_text())
    new_resolved = set(new.get("resolved_ids", []))
    new_unresolved = set(new.get("unresolved_ids", []))
    new_error = set(new.get("error_ids", []))

    # Merge into original report
    backup_path = report_path.with_suffix(".pre_surgical.bak")
    if not backup_path.exists():
        shutil.copy(report_path, backup_path)

    merged_resolved = set(rep.get("resolved_ids", [])) | new_resolved
    merged_unresolved = set(rep.get("unresolved_ids", [])) | new_unresolved
    merged_error = (set(rep.get("error_ids", [])) - new_resolved - new_unresolved) | new_error

    rep["resolved_ids"] = sorted(merged_resolved)
    rep["unresolved_ids"] = sorted(merged_unresolved)
    rep["error_ids"] = sorted(merged_error)
    rep["resolved_instances"] = len(merged_resolved)
    rep["unresolved_instances"] = len(merged_unresolved)
    rep["error_instances"] = len(merged_error)
    rep["completed_instances"] = len(merged_resolved) + len(merged_unresolved)
    rep["surgical_retry_history"] = rep.get("surgical_retry_history", []) + [{
        "ts": datetime.now().isoformat(),
        "retry_run_id": retry_run_id,
        "n_attempted": n_filtered,
        "n_recovered_resolved": len(new_resolved - set(rep.get("resolved_ids", []))),
        "n_recovered_unresolved": len(new_unresolved),
        "n_still_errored": len(new_error),
    }]

    report_path.write_text(json.dumps(rep, indent=4))
    filtered_path.unlink()  # cleanup

    return {
        "status": "merged",
        "n_attempted": n_filtered,
        "recovered_resolved": len(new_resolved),
        "recovered_unresolved": len(new_unresolved),
        "still_errored": len(new_error),
        "rc": rc,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", help="path to eval/<run_id>.<run_id>.json")
    parser.add_argument("--auto", action="store_true", help="find all err>0 reports and retry")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = []
    if args.auto:
        for p in sorted(EXP_DIR.glob("data/swebench_*_realbaselines/*/*/eval/*.json")):
            if "harness" in p.name or "surgical_retry" in p.name or ".bak" in p.name or "retry_" in p.name: continue
            try:
                d = json.loads(p.read_text())
                if d.get("error_instances", 0) > 0:
                    targets.append(p)
            except Exception:
                pass
        # Also the qwen32b calibration dirs
        for p in sorted(EXP_DIR.glob("data/swebench_*_qwen32b/*/eval/*.json")):
            if "harness" in p.name or ".bak" in p.name or "retry_" in p.name: continue
            try:
                d = json.loads(p.read_text())
                if d.get("error_instances", 0) > 0:
                    targets.append(p)
            except Exception:
                pass
    elif args.report:
        targets = [Path(args.report)]
    else:
        parser.print_help()
        sys.exit(1)

    print(f"found {len(targets)} reports with err>0")
    for t in targets:
        print(f"\n{t.relative_to(EXP_DIR)}")
        result = retry_one(t, dry_run=args.dry_run)
        print(f"  result: {result}")


if __name__ == "__main__":
    main()
