"""Merge existing retry .json files into the original reports.
The retry harness already ran; we just need to fold the results back."""
import json, glob, os, shutil
from pathlib import Path
from datetime import datetime

ROOT = Path("/mnt/data/users/vlad.smirnov/agents_with_uncertainty_research/experiments/orchestration_hypothesis_testing")

def merge_one(retry_path: Path):
    """retry_path = .../eval/<orig_run_id>.<orig_run_id>_retry_HHMMSS.json
       original  = .../eval/<orig_run_id>.<orig_run_id>.json"""
    name = retry_path.name
    if "_retry_" not in name: return None
    parts = name[:-5].split(".")  # strip .json then split by .
    if len(parts) != 2: return None
    orig_run_id = parts[0]
    retry_run_id = parts[1]
    orig_path = retry_path.parent / f"{orig_run_id}.{orig_run_id}.json"
    if not orig_path.exists():
        return f"orig missing: {orig_path}"

    new = json.loads(retry_path.read_text())
    rep = json.loads(orig_path.read_text())

    # Check if this retry was already merged
    if "surgical_retry_history" in rep and any(h.get("retry_run_id") == retry_run_id for h in rep["surgical_retry_history"]):
        return "already merged"

    new_resolved = set(new.get("resolved_ids", []))
    new_unresolved = set(new.get("unresolved_ids", []))
    new_error = set(new.get("error_ids", []))

    backup_path = orig_path.with_suffix(".pre_surgical.bak")
    if not backup_path.exists():
        shutil.copy(orig_path, backup_path)

    merged_resolved = set(rep.get("resolved_ids", [])) | new_resolved
    merged_unresolved = set(rep.get("unresolved_ids", [])) | new_unresolved
    # error: take original errors, remove ones that are now resolved/unresolved, add new errors
    merged_error = (set(rep.get("error_ids", [])) - new_resolved - new_unresolved) | new_error

    n_recovered_resolved = len(new_resolved - set(rep.get("resolved_ids", [])))
    n_recovered_unresolved = len(new_unresolved - set(rep.get("unresolved_ids", [])))

    rep["resolved_ids"] = sorted(merged_resolved)
    rep["unresolved_ids"] = sorted(merged_unresolved)
    rep["error_ids"] = sorted(merged_error)
    rep["resolved_instances"] = len(merged_resolved)
    rep["unresolved_instances"] = len(merged_unresolved)
    rep["error_instances"] = len(merged_error)
    rep["completed_instances"] = len(merged_resolved) + len(merged_unresolved)
    rep.setdefault("surgical_retry_history", []).append({
        "ts": datetime.now().isoformat(),
        "retry_run_id": retry_run_id,
        "n_attempted": new.get("submitted_instances", 0),
        "n_recovered_resolved": n_recovered_resolved,
        "n_recovered_unresolved": n_recovered_unresolved,
        "n_still_errored": len(new_error),
    })

    orig_path.write_text(json.dumps(rep, indent=4))
    return f"merged: +{n_recovered_resolved}R +{n_recovered_unresolved}U, {len(new_error)} still err"


def main():
    retry_files = sorted(ROOT.glob("data/swebench_*_realbaselines/*/*/eval/*_retry_*.json"))
    print(f"found {len(retry_files)} retry .json files")
    for r in retry_files:
        result = merge_one(r)
        cell = str(r).split("realbaselines/")[1][:60]
        print(f"  {cell:<62s} {result}")


if __name__ == "__main__":
    main()
