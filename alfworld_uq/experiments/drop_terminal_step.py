"""A copy of a run with the terminal step of every episode removed.

On a finished episode the last step is the outcome in disguise: a success
ends on the action that satisfied the goal (state changed, tool succeeded),
a failure ends on give-up or final_answer (no environment action at all).
Any critic or feature that sees that step reads the label -- the tempered
critic scored .85 on the finished cohorts with it and .66 without. Every
method is therefore scored on the episode up to, not including, its last
generation: the state a monitor would act on before the outcome exists.
Episode-level fields derived from steps (tool_success_rate,
state_changed_rate, verbalized_*) are recomputed from the kept steps.

    python -m experiments.drop_terminal_step RUN_DIR OUT_DIR
"""
from __future__ import annotations

import json
import shutil
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    if (src / "run_config.json").exists():
        shutil.copy(src / "run_config.json", dst / "run_config.json")
    steps = defaultdict(list)
    for line in open(src / "trajectories.jsonl"):
        if line.strip():
            row = json.loads(line); steps[row["episode_id"]].append(row)
    kept = {i: (rows[:-1] if len(rows) > 1 else rows) for i, rows in steps.items()}
    with open(dst / "trajectories.jsonl", "w") as f:
        for rows in kept.values():
            for row in rows:
                f.write(json.dumps(row) + "\n")
    with open(dst / "episodes.jsonl", "w") as f:
        for line in open(src / "episodes.jsonl"):
            if not line.strip():
                continue
            e = json.loads(line); rows = kept.get(e["episode_id"], [])
            if rows:
                e["tool_success_rate"] = st.fmean(bool(r.get("tool_success")) for r in rows)
                e["state_changed_rate"] = st.fmean(bool(r.get("state_changed")) for r in rows)
                verbs = [r.get("verb") for r in rows if r.get("verb") is not None]
                e["verbalized_mean"] = st.fmean(verbs) if verbs else None
                e["verbalized_last"] = verbs[-1] if verbs else None
                e["final_verbalized"] = verbs[-1] if verbs else None
                e["num_steps_scored"] = len(rows)
            f.write(json.dumps(e) + "\n")
    print(f"{dst.name}: {len(kept)} episodes, {sum(len(v) for v in kept.values())} of {sum(len(v) for v in steps.values())} steps kept")


if __name__ == "__main__":
    main()
