"""How many labelled episodes the thin layer over a decision model needs.

The decision model itself is never trained. What is fitted is the logistic over
its four answers, and the question the paper asks of every estimator applies to
it too: how much labelled data before it is as good as it gets. Same protocol as
the learning curves of the report -- 3 seeds x 5 folds, each fold's training
part subsampled in a label-stratified way, held-out predictions pooled per seed.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.decision_model_score import _fuse, extract, fit_temperature, apply_temperature
from experiments.prr_report_v2 import SEEDS, prr, stratified_folds


def subsample(train: list[int], labels: list[float], n: int, rng: np.random.RandomState) -> list[int]:
    by_label: dict[int, list[int]] = defaultdict(list)
    for i in train:
        by_label[int(labels[i])].append(i)
    out: list[int] = []
    for label, members in by_label.items():
        take = max(1, round(n * len(members) / len(train)))
        picked = rng.permutation(members)[: min(take, len(members))]
        out += list(int(i) for i in picked)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--answers", type=Path, required=True)
    p.add_argument("--question", default="success")
    p.add_argument("--extra", nargs="*", default=["stuck", "stale_evidence", "progress"])
    p.add_argument("--sizes", type=int, nargs="*", default=[10, 20, 30, 40, 60, 80])
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    rows = [json.loads(line) for line in open(a.answers) if line.strip()]
    by_cohort: dict[str, list] = defaultdict(list)
    for row in rows:
        value = extract(row["answer"], a.question)
        if value is None:
            continue
        row["p"] = value
        row["extra"] = [extract(row["answer"], q) for q in a.extra]
        by_cohort[row["cohort"]].append(row)

    results: dict[str, dict] = {}
    for cohort, items in sorted(by_cohort.items()):
        scores = [r["score"] for r in items]
        continuous = len(set(scores)) > 2
        labels = ([1.0 if s > st.median(scores) else 0.0 for s in scores] if continuous
                  else [float(s) for s in scores])
        episodes = {r["id"]: {"label": int(labels[i]), "score": scores[i], "id": r["id"]}
                    for i, r in enumerate(items)}
        index = {r["id"]: i for i, r in enumerate(items)}
        raw = prr(scores, [r["p"] for r in items])
        curve = {}
        for size in a.sizes + [0]:  # 0 = the whole training part
            per_seed = []
            for seed in SEEDS:
                rng = np.random.RandomState(seed)
                folds = stratified_folds(episodes, seed)
                fused = [0.0] * len(items)
                for fold in folds:
                    held = set(fold)
                    train = [index[i] for i in episodes if i not in held]
                    if size:
                        train = subsample(train, labels, size, rng)
                    test = [index[i] for i in fold]
                    if len({int(labels[i]) for i in train}) < 2:
                        for i in test:
                            fused[i] = items[i]["p"]
                        continue
                    for i, v in zip(test, _fuse(items, labels, train, test, a.extra)):
                        fused[i] = v
                per_seed.append(prr(scores, fused))
            curve["full" if size == 0 else size] = (st.fmean(per_seed), st.pstdev(per_seed))
        results[cohort] = {"raw": raw, "curve": curve, "n": len(items)}

    header = f"{'cohort':7} {'n':>4} {'raw':>6} " + " ".join(f"{s:>7}" for s in a.sizes) + f" {'full':>7}"
    print(header); print("-" * len(header))
    for cohort, row in results.items():
        cells = " ".join(f"{row['curve'][s][0]:+7.2f}" for s in a.sizes)
        print(f"{cohort:7} {row['n']:4d} {row['raw']:+6.2f} {cells} {row['curve']['full'][0]:+7.2f}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({c: {"n": v["n"], "raw": v["raw"],
                       "curve": {str(k): list(val) for k, val in v["curve"].items()}}
                   for c, v in results.items()}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
