"""The main-branch logistic regression, scored on our ALFWorld cohorts.

`trajectory_uq_toolkit.regression.TrajectoryRegression` (agentic-uq, main) is
the discriminative counterpart of the Bayesian rows: every (signal,
aggregation) pair and every critic goes into one feature vector, columns are
forward-selected on inner folds, and an L2 logistic regression is fitted.
It is given exactly what the other rows see -- the per-step UQ signals of all
four response segments, the five step critics -- and nothing about length.
Same split seeds and the same PRR@0.5 as the rest of the table.

    python -m experiments.regression_rows --toolkit PATH/src RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from experiments.analyze_trajectories import (
    STEP_CRITIC_NAMES,
    _prr_references,
    _split_ids,
    _step_critic_observation,
    prediction_rejection_area,
)

SEGMENTS = ("thought", "action", "reasoning", "combined")
SIGNALS = ("mean_token_logprob", "perplexity", "mean_token_entropy", "sequence_probability", "sum_logprob", "verbalized_confidence")


def _prr(y, conf):
    a = prediction_rejection_area([-c for c in conf], y, 0.5)
    o, r = _prr_references(tuple(y), 0.5)
    return None if None in (a, o, r) or abs(o - r) < 1e-9 else (a - r) / (o - r)


def records_of(run: Path) -> tuple[dict[str, dict], dict[str, int]]:
    steps = defaultdict(list)
    for line in open(run / "trajectories.jsonl"):
        if line.strip():
            row = json.loads(line); steps[row["episode_id"]].append(row)
    records, labels = {}, {}
    for line in open(run / "episodes.jsonl"):
        if not line.strip():
            continue
        e = json.loads(line); rows = steps.get(e["episode_id"], [])
        if not rows:
            continue
        generations = []
        for k, row in enumerate(rows):
            signals = {}
            for seg in SEGMENTS:
                bundle = (row.get("uq") or {}).get(seg) or {}
                for sig in SIGNALS:
                    v = bundle.get(sig)
                    if v is not None:
                        signals[f"{seg}_{sig}"] = float(v)
            crit = _step_critic_observation(row)
            generations.append({"index": k, "signals": signals, "critics": {c: bool(crit[c]) for c in STEP_CRITIC_NAMES}})
        records[e["episode_id"]] = {"episode_id": e["episode_id"], "environment": "alfworld", "success": int(bool(e["final_success"])), "generations": generations}
        labels[e["episode_id"]] = int(bool(e["final_success"]))
    return records, labels


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--toolkit", required=True, help="path to agentic-uq/src (main branch)")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("runs", nargs="+", type=Path)
    a = p.parse_args()
    sys.path.insert(0, a.toolkit)
    from trajectory_uq_toolkit.regression import REGRESSION_CONFIGURATIONS, TrajectoryRegression

    out = {name: [] for name in REGRESSION_CONFIGURATIONS}
    chosen = defaultdict(lambda: defaultdict(int))
    for run in a.runs:
        records, labels = records_of(run); ids = list(records)
        acc = {name: [] for name in REGRESSION_CONFIGURATIONS}
        for seed in range(a.seeds):
            cal, test = _split_ids(ids, 0.5, seed)
            for name, settings in REGRESSION_CONFIGURATIONS.items():
                model = TrajectoryRegression.fit([records[i] for i in cal], [labels[i] for i in cal], seed=seed, **settings)
                v = _prr([labels[i] for i in test], model.predict([records[i] for i in test]))
                if v is not None:
                    acc[name].append(v)
                for feat in (model.describe().get("weights") or {}):
                    chosen[(name, run.name)][feat] += 1
        for name in acc:
            out[name].append(st.fmean(acc[name]) if acc[name] else None)
    print(f"{'method':32s}" + "".join(f"{r.name[-22:]:>24s}" for r in a.runs))
    for name, vals in out.items():
        print(f"{'Logistic regression ('+name+')':32s}" + "".join((f"{v:+24.3f}" if v is not None else f"{'NA':>24s}") for v in vals))
    print("\nmost selected features (seeds):")
    for (name, run), feats in chosen.items():
        top = sorted(feats.items(), key=lambda kv: -kv[1])[:4]
        print(f"  {name:9s} {run[-30:]:>30s}  " + ", ".join(f"{k}:{n}" for k, n in top))


if __name__ == "__main__":
    main()
