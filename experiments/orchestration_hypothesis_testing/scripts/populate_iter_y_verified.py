"""Populate Y_t per step in iter_records.jsonl from SWE-Verified harness reports.

For each generator, reads:
  - step 0 Y from <gen>/critic_results.jsonl (key: instance_id, patch_id=0)
  - steps 1..N-1 Y from data/swebench_verified_iter/eval/<gen>_iter_step{S}.<gen>_iter_step{S}.json

Updates the iter_records.jsonl in place.

Usage:
  python3 populate_iter_y_verified.py \\
    --iter-dir data/swebench_verified_iter \\
    --src-dir data/swebench_verified \\
    --generators gpt5_mini,qwen3_coder,haiku45,sonnet45 \\
    --steps 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iter-dir", required=True, type=Path)
    parser.add_argument("--src-dir", required=True, type=Path,
                        help="dir containing <gen>/critic_results.jsonl with step-0 Y")
    parser.add_argument("--generators", required=True)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    iter_dir = args.iter_dir.resolve()
    src_dir = args.src_dir.resolve()
    eval_dir = iter_dir / "eval"

    for gen in [g.strip() for g in args.generators.split(",") if g.strip()]:
        # Step 0: from critic_results.jsonl in src_dir
        step0_y: dict[str, int] = {}
        crit_path = src_dir / gen / "critic_results.jsonl"
        if crit_path.exists():
            for line in open(crit_path):
                if not line.strip(): continue
                r = json.loads(line)
                if r.get("patch_id") == 0:
                    step0_y[r["instance_id"]] = int(r.get("Y") or 0)

        # Steps 1..N-1: from harness summary reports
        resolved_by_step: dict[int, set[str]] = {0: {k for k, v in step0_y.items() if v == 1}}
        for s in range(1, args.steps):
            p = eval_dir / f"{gen}_iter_step{s}.{gen}_iter_step{s}.json"
            if p.exists():
                rep = json.loads(p.read_text())
                resolved_by_step[s] = set(rep.get("resolved_ids", []))
                print(f"[{gen}/step{s}] {len(resolved_by_step[s])} resolved")
            else:
                print(f"[{gen}/step{s}] WARN: no report at {p.name}")
                resolved_by_step[s] = set()

        # Update iter_records.jsonl
        records_path = iter_dir / gen / "iter_records.jsonl"
        if not records_path.exists():
            print(f"[{gen}] no iter_records.jsonl, skipping")
            continue
        records = [json.loads(l) for l in open(records_path) if l.strip()]
        n_updated = 0
        for r in records:
            s = r["step"]
            if s in resolved_by_step:
                Y = 1 if r["instance_id"] in resolved_by_step[s] else 0
                if r.get("Y") != Y:
                    r["Y"] = Y
                    n_updated += 1
        with open(records_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        n_y1 = sum(1 for r in records if r.get("Y") == 1)
        print(f"[{gen}] wrote {len(records)} records ({n_updated} Y updated, {n_y1} now Y=1)")


if __name__ == "__main__":
    main()
