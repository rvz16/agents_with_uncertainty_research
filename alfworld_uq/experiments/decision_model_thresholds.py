"""Are the probabilities usable as thresholds?

Expected calibration error averages over the whole range and hides the thing a
harness actually needs: if it accepts an outcome at p >= 0.8, does four in five
hold up. This reports, for a grid of thresholds, how much of the cohort clears
it and what the realised success rate is there, against the threshold itself.
It also reports both tails of confident error -- the share of accepted episodes
that fail at p >= 0.9, and of rejected ones that succeed at p <= 0.1 -- which
is the number the model cards quote.

Our own estimators answer in log-odds, so they come through a sigmoid and are
measured on exactly the same grid; nothing here is specific to a decision model.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.decision_model_score import apply_temperature, ece, extract, fit_temperature
from experiments.prr_report_v2 import SEEDS, stratified_folds

GRID = (0.5, 0.6, 0.7, 0.8, 0.9)


def reliability(labels, probabilities, bins: int = 5) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    labels = np.asarray(labels, dtype=float); probabilities = np.asarray(probabilities, dtype=float)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities > low) & (probabilities <= high) if low > 0 else (probabilities <= high)
        if not mask.sum():
            continue
        rows.append({"bin": f"{low:.1f}-{high:.1f}", "n": int(mask.sum()),
                     "mean_p": float(probabilities[mask].mean()),
                     "observed": float(labels[mask].mean())})
    return rows


def thresholds(labels, probabilities) -> list[dict]:
    labels = np.asarray(labels, dtype=float); probabilities = np.asarray(probabilities, dtype=float)
    rows = []
    for tau in GRID:
        kept = probabilities >= tau
        rows.append({"tau": tau, "coverage": float(kept.mean()),
                     "observed": float(labels[kept].mean()) if kept.sum() else float("nan"),
                     "n": int(kept.sum())})
    return rows


def confident_errors(labels, probabilities) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=float); probabilities = np.asarray(probabilities, dtype=float)
    high = probabilities >= 0.9
    low = probabilities <= 0.1
    return (float((1 - labels[high]).mean()) if high.sum() else float("nan"),
            float(labels[low].mean()) if low.sum() else float("nan"))


def scaled_out_of_fold(ids, labels, probabilities) -> list[float]:
    """The temperature every one of these models fits on a dev set, fitted here
    on each fold's training part so no episode is scaled by its own label."""
    episodes = {i: {"label": int(labels[k]), "score": labels[k], "id": i} for k, i in enumerate(ids)}
    index = {i: k for k, i in enumerate(ids)}
    per_seed = []
    for seed in SEEDS:
        out = [0.0] * len(ids)
        for fold in stratified_folds(episodes, seed):
            held = set(fold)
            train = [index[i] for i in episodes if i not in held]
            temperature = fit_temperature([labels[i] for i in train], [probabilities[i] for i in train])
            for i in (index[j] for j in fold):
                out[i] = apply_temperature([probabilities[i]], temperature)[0]
        per_seed.append(out)
    return [st.fmean(values) for values in zip(*per_seed)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--answers", type=Path, nargs="+", required=True)
    p.add_argument("--question", default="success")
    p.add_argument("--pool", action="store_true", help="pool the cohorts of a file instead of splitting")
    a = p.parse_args()

    for path in a.answers:
        rows = [json.loads(line) for line in open(path) if line.strip()]
        groups: dict[str, list] = defaultdict(list)
        for row in rows:
            value = extract(row["answer"], a.question)
            if value is None:
                continue
            row["p"] = value
            groups["all" if a.pool else row["cohort"]].append(row)

        print(f"\n{'=' * 78}\n{path.name}\n{'=' * 78}")
        for name, items in sorted(groups.items()):
            scores = [r["score"] for r in items]
            continuous = len(set(scores)) > 2
            labels = ([1.0 if s > st.median(scores) else 0.0 for s in scores] if continuous
                      else [float(s) for s in scores])
            probabilities = [r["p"] for r in items]
            ids = [f'{r["cohort"]}:{r["id"]}' for r in items]
            scaled = scaled_out_of_fold(ids, labels, probabilities)

            base = st.fmean(labels)
            print(f"\n-- {name}: n={len(items)}, base rate {base:.2f}, "
                  f"ECE {ece(labels, probabilities):.3f} -> {ece(labels, scaled):.3f} after a fitted temperature")
            print(f"   {'bin':10} {'n':>4} {'mean p':>7} {'observed':>9} {'gap':>7}   |   "
                  f"{'bin':10} {'n':>4} {'mean p':>7} {'observed':>9} {'gap':>7}")
            raw_rows, scaled_rows = reliability(labels, probabilities), reliability(labels, scaled)
            for left, right in zip(raw_rows + [None] * len(scaled_rows), scaled_rows + [None] * len(raw_rows)):
                if left is None and right is None:
                    break
                def cell(r):
                    if r is None:
                        return " " * 40
                    return (f"   {r['bin']:10} {r['n']:4d} {r['mean_p']:7.2f} {r['observed']:9.2f} "
                            f"{r['observed'] - r['mean_p']:+7.2f}")
                print(f"{cell(left)}   |{cell(right)}")

            print(f"   threshold:   " + "  ".join(f"{t:>5.1f}" for t in GRID))
            for label, values in (("raw", probabilities), ("scaled", scaled)):
                th = thresholds(labels, values)
                cover = "  ".join(f"{r['coverage']:5.2f}" for r in th)
                obs = "  ".join((f"{r['observed']:5.2f}" if r["n"] else "    -") for r in th)
                gap = "  ".join((f"{r['observed'] - r['tau']:+5.2f}" if r["n"] else "    -") for r in th)
                print(f"     {label:6} coverage:  {cover}")
                print(f"     {label:6} observed:  {obs}")
                print(f"     {label:6} gap:       {gap}")
            hi, lo = confident_errors(labels, probabilities)
            hi_s, lo_s = confident_errors(labels, scaled)
            print(f"   confident errors  p>=0.9: {hi:.3f} raw, {hi_s:.3f} scaled   "
                  f"|  p<=0.1 but succeeded: {lo:.3f} raw, {lo_s:.3f} scaled")


if __name__ == "__main__":
    main()
