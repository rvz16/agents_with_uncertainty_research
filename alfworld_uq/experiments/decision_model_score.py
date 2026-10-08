"""Score a decision model's answers against the cohorts, on the report's folds.

The model needs no training, so its probability ranks episodes as a raw signal
does. Two things still need the folds: a temperature fitted on the training part
of each fold, and the small logistic that asks whether the three auxiliary
questions add anything over the success probability alone. Both are fitted out
of fold, so nothing is scored by a model that has seen it.

Reports PRR@0.5 (the report's metric), and ECE and Brier, which the main tables
do not carry: a decision model's claim is calibration, and ranking alone would
not test it.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.prr_report_v2 import FOLDS, SEEDS, prr, stratified_folds

PROBABILITY_FIELDS = ("value", "probability", "prob", "p", "true", "yes", "score")


def extract(answer: dict, question: str) -> float | None:
    """Pull a probability for ``question`` out of a /v1/systemone response."""
    if not isinstance(answer, dict) or "error" in answer:
        return None
    for container in (answer, answer.get("answers"), answer.get("nouls"), answer.get("scores"),
                      answer.get("choices"), answer.get("result"), answer.get("questions")):
        if not isinstance(container, dict):
            continue
        node = container.get(question)
        if node is None:
            continue
        if isinstance(node, (int, float)):
            return float(node)
        if isinstance(node, dict):
            for field in PROBABILITY_FIELDS:
                if isinstance(node.get(field), (int, float)):
                    return float(node[field])
            options = node.get("options") or node.get("criteria")
            if isinstance(options, dict):  # a score: expectation over the ordered levels
                items = [(k, v) for k, v in options.items() if isinstance(v, (int, float))]
                if items:
                    total = sum(v for _, v in items) or 1.0
                    return sum(i * v for i, (_, v) in enumerate(items)) / (len(items) - 1 or 1) / total
    return None


def ece(labels, probabilities, bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=float); probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities > low) & (probabilities <= high) if low > 0 else (probabilities <= high)
        if mask.sum():
            total += mask.sum() / len(labels) * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(total)


def brier(labels, probabilities) -> float:
    return float(np.mean((np.asarray(probabilities, dtype=float) - np.asarray(labels, dtype=float)) ** 2))


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def fit_temperature(labels, probabilities) -> float:
    """The one scalar every published decision model fits on a dev set."""
    z = np.array([_logit(p) for p in probabilities]); y = np.asarray(labels, dtype=float)
    best, best_loss = 1.0, float("inf")
    for temperature in np.concatenate([np.linspace(0.2, 5.0, 97), np.linspace(5.5, 12.0, 14)]):
        q = 1.0 / (1.0 + np.exp(-z / temperature))
        loss = -float(np.mean(y * np.log(q + 1e-9) + (1 - y) * np.log(1 - q + 1e-9)))
        if loss < best_loss:
            best, best_loss = float(temperature), loss
    return best


def apply_temperature(probabilities, temperature: float):
    return [1.0 / (1.0 + math.exp(-_logit(p) / temperature)) for p in probabilities]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--answers", type=Path, nargs="+", required=True)
    p.add_argument("--question", default="success")
    p.add_argument("--extra", nargs="*", default=["stuck", "stale_evidence", "progress"])
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    for path in a.answers:
        rows = [json.loads(line) for line in open(path) if line.strip()]
        by_cohort: dict[str, list] = defaultdict(list)
        missing = 0
        for row in rows:
            value = extract(row["answer"], a.question)
            if value is None:
                missing += 1
                continue
            row["p"] = value
            row["extra"] = [extract(row["answer"], q) for q in a.extra]
            by_cohort[row["cohort"]].append(row)
        print(f"\n=== {path.name}: {len(rows)} answers, {missing} without a parsable probability ===")
        if missing == len(rows) and rows:
            print("    response shape:", json.dumps(rows[0]["answer"])[:400])
            continue
        table = {}
        for cohort, items in sorted(by_cohort.items()):
            scores = [r["score"] for r in items]
            continuous = len(set(scores)) > 2
            labels = ([1.0 if s > st.median(scores) else 0.0 for s in scores] if continuous
                      else [float(s) for s in scores])
            probabilities = [r["p"] for r in items]
            raw_prr = prr(scores, probabilities)
            episodes = {r["id"]: {"label": int(labels[i]), "score": scores[i], "id": r["id"]}
                        for i, r in enumerate(items)}
            index = {r["id"]: i for i, r in enumerate(items)}
            scaled_per_seed, prr_scaled, prr_fused = [], [], []
            for seed in SEEDS:
                folds = stratified_folds(episodes, seed)
                scaled = [0.0] * len(items); fused = [0.0] * len(items)
                for fold in folds:
                    held = set(fold)
                    train = [index[i] for i in episodes if i not in held]
                    test = [index[i] for i in fold]
                    temperature = fit_temperature([labels[i] for i in train], [probabilities[i] for i in train])
                    for i in test:
                        scaled[i] = apply_temperature([probabilities[i]], temperature)[0]
                    fused_values = _fuse(items, labels, train, test, a.extra)
                    for i, v in zip(test, fused_values):
                        fused[i] = v
                scaled_per_seed.append(scaled)
                prr_scaled.append(prr(scores, scaled)); prr_fused.append(prr(scores, fused))
            mean_scaled = [st.fmean(values) for values in zip(*scaled_per_seed)]
            table[cohort] = {
                "n": len(items), "prr_raw": raw_prr,
                "prr_scaled": st.fmean(prr_scaled), "prr_fused": st.fmean(prr_fused),
                "ece_raw": ece(labels, probabilities), "ece_scaled": ece(labels, mean_scaled),
                "brier_raw": brier(labels, probabilities), "brier_scaled": brier(labels, mean_scaled),
                "mean_p": st.fmean(probabilities), "base": st.fmean(labels),
            }
        header = f"{'cohort':7} {'n':>4} {'base':>5} {'mean p':>7} | {'PRR raw':>8} {'PRR temp':>9} {'PRR +3q':>8} | {'ECE raw':>8} {'ECE temp':>9} | {'Brier':>7}"
        print(header); print("-" * len(header))
        for cohort, row in table.items():
            print(f"{cohort:7} {row['n']:4d} {row['base']:5.2f} {row['mean_p']:7.2f} | "
                  f"{row['prr_raw']:+8.2f} {row['prr_scaled']:+9.2f} {row['prr_fused']:+8.2f} | "
                  f"{row['ece_raw']:8.3f} {row['ece_scaled']:9.3f} | {row['brier_raw']:7.3f}")
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            json.dump(table, open(a.out, "w"), indent=1)


def _fuse(items, labels, train, test, extra) -> list[float]:
    """Logistic on the success probability plus the auxiliary answers."""
    from sklearn.linear_model import LogisticRegression

    def design(indices):
        rows = []
        for i in indices:
            values = [_logit(items[i]["p"])] + [v if v is not None else 0.0 for v in items[i]["extra"]]
            rows.append(values)
        return np.asarray(rows, dtype=float)

    y = np.asarray([labels[i] for i in train], dtype=float)
    if len(set(y.tolist())) < 2:
        return [items[i]["p"] for i in test]
    model = LogisticRegression(C=1.0, max_iter=1000).fit(design(train), y)
    return list(model.predict_proba(design(test))[:, 1])


if __name__ == "__main__":
    main()
