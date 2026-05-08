"""Populate Y_t per step in iter_records.jsonl from harness reports.

Reads each <gen>__iter_step{S}.<gen>_iter_step{S}.json report's resolved_ids,
then updates the iter_records.jsonl row matching (instance_id, step) with
Y in {0, 1}. Step 0's Y is taken from the spot-check eval reports
(<gen>__p0.<gen>_p0.json).
"""
import json
import sys
from pathlib import Path

ROOT = Path("/mnt/data/users/vlad.smirnov/agents_with_uncertainty_research/.claude/worktrees/reverent-vaughan-017bf5/experiments/orchestration_hypothesis_testing")
DATA = ROOT / "data" / "spot_check_n50"

gen = sys.argv[1]
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 5

# Load resolved sets per step
resolved_by_step: dict[int, set[str]] = {}
# step 0 from spot-check (use patch_id=0 since iter started from step 0 = spot-check p0)
step0_path = DATA / "eval" / f"{gen}__p0.{gen}_p0.json"
if step0_path.exists():
    rep = json.loads(step0_path.read_text())
    resolved_by_step[0] = set(rep.get("resolved_ids", []))
else:
    print(f"WARN: no step-0 report at {step0_path}")
    resolved_by_step[0] = set()

for s in range(1, n_steps):
    p = DATA / "eval" / f"{gen}__iter_step{s}.{gen}_iter_step{s}.json"
    if p.exists():
        rep = json.loads(p.read_text())
        resolved_by_step[s] = set(rep.get("resolved_ids", []))
    else:
        print(f"WARN: no step-{s} report")
        resolved_by_step[s] = set()

# Update iter_records.jsonl
records_path = DATA / gen / "iter_records.jsonl"
with open(records_path) as f:
    records = [json.loads(line) for line in f]

n_updated = 0
for r in records:
    s = r["step"]
    inst = r["instance_id"]
    new_y = 1 if inst in resolved_by_step.get(s, set()) else 0
    if r.get("Y") != new_y:
        r["Y"] = new_y
        n_updated += 1

records_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
print(f"updated Y on {n_updated} records in {records_path}")

# Print per-step Y=1 counts
print(f"\n{gen} resolved counts per step:")
for s in range(n_steps):
    n_y1 = sum(1 for r in records if r["step"] == s and r.get("Y") == 1)
    n_total = sum(1 for r in records if r["step"] == s)
    print(f"  step {s}: {n_y1}/{n_total} resolved ({100*n_y1/max(n_total,1):.1f}%)")
