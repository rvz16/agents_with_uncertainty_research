"""Compute per-generator transition kernel from iter_records.jsonl.

For each generator, counts (Y_t, Y_{t+1}) transitions across all (instance, step)
pairs and computes:
  - P_fix_given_broken = (Y_t=0 → Y_{t+1}=1) / total Y_t=0
  - P_break_given_correct = (Y_t=1 → Y_{t+1}=0) / total Y_t=1

With Beta(1,1) smoothing.

Usage:
  python3 compute_iter_kernel.py \\
    --iter-dir data/swebench_verified_iter \\
    --generators gpt5_mini,qwen3_coder,haiku45,sonnet45
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iter-dir", required=True, type=Path)
    parser.add_argument("--generators", required=True)
    args = parser.parse_args()

    iter_dir = args.iter_dir.resolve()
    print(f"{'gen':<14} {'n_pairs':>8} {'0->0':>5} {'0->1':>5} {'1->0':>5} {'1->1':>5} {'P_fix':>7} {'P_break':>9}")

    for gen in [g.strip() for g in args.generators.split(",") if g.strip()]:
        path = iter_dir / gen / "iter_records.jsonl"
        if not path.exists():
            print(f"[{gen}] missing iter_records.jsonl"); continue

        # Group by instance, sorted by step
        by_inst: dict[str, list[dict]] = {}
        for line in open(path):
            if not line.strip(): continue
            r = json.loads(line)
            by_inst.setdefault(r["instance_id"], []).append(r)
        for inst in by_inst:
            by_inst[inst].sort(key=lambda r: r["step"])

        # Count transitions
        counts = {"0->0": 0, "0->1": 0, "1->0": 0, "1->1": 0}
        for inst, traj in by_inst.items():
            for i in range(len(traj) - 1):
                yt = traj[i].get("Y")
                yt1 = traj[i + 1].get("Y")
                if yt is None or yt1 is None:
                    continue
                counts[f"{yt}->{yt1}"] += 1

        n_broken = counts["0->0"] + counts["0->1"]
        n_correct = counts["1->0"] + counts["1->1"]
        # Beta(1,1)
        P_fix = (counts["0->1"] + 1) / (n_broken + 2)
        P_break = (counts["1->0"] + 1) / (n_correct + 2)
        n_pairs = n_broken + n_correct

        print(f"{gen:<14} {n_pairs:>8} {counts['0->0']:>5} {counts['0->1']:>5} "
              f"{counts['1->0']:>5} {counts['1->1']:>5} {P_fix:>7.3f} {P_break:>9.3f}")

        # Save kernel json (matches lcb_compare_swap_reviewer's expected schema)
        kernel = {
            "generator": gen,
            "kernel_all": {
                "P_fix_given_broken": P_fix,
                "P_stay_broken": 1 - P_fix,
                "P_break_given_correct": P_break,
                "P_stay_correct": 1 - P_break,
                "raw_counts": counts,
                "n_pairs": n_pairs,
                "smoothing": "Beta(1,1)",
            },
        }
        out_path = iter_dir / gen / "transition_kernel.json"
        out_path.write_text(json.dumps(kernel, indent=2))


if __name__ == "__main__":
    main()
