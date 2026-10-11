"""What the reported number costs once the aggregation mode is also chosen honestly.

Each signal in this work can be turned into an episode-level confidence about
thirteen ways: six Bayesian update modes, each with or without the tool
posterior, plus the history-and-last fusion. The tables report one of them, and
that one was chosen by looking at all thirteen on the same episodes it is scored
on. On this sample the spread across modes for a single signal runs from .37 to
.76, so the choice is worth more than most of the differences the paper
discusses, and a reader is entitled to ask what is left when it is paid for.

Two honest alternatives are measured against the selected number.

``nested`` chooses the mode inside each outer training fold, by an inner
cross-validation that never sees the outer test fold, and then applies it. This
is the strict answer and it also reports how often the inner selection changes
its mind, which is itself informative: a choice that is stable across folds was
probably not overfitted, one that flips every fold was.

``transfer`` chooses the mode on the other three cohorts and applies it to the
held-out one. This is the question a practitioner actually faces -- the mode is
a hyperparameter you fix once and carry to a new agent -- and it is the weaker
but more realistic bound.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time
from collections import Counter
from pathlib import Path

from experiments.prr_report_v2 import (SEEDS, SIGNALS, is_continuous, load_alfworld, method_table,
                                       oof, prr, stratified_folds, with_labels)

RUNS = {"RG": "react_gptoss_140_giveup_sc_finished_nolast", "SG": "smol_gptoss_140_sc_finished_nolast",
        "RQ": "react_qwen_140_giveup_sc_finished_nolast", "SQ": "smol_qwen_140_sc_finished_nolast"}
INJECTED = {"Decision p(success)": ("judge_Cloudflare_clef-flash", "p", False),
            "Decision progress": ("judge_Cloudflare_clef-flash", "progress", False)}


def read_steps(path: Path, field: str) -> dict[str, list[float]]:
    out: dict[str, list[tuple[int, float]]] = {}
    if not path.exists():
        return {}
    for line in open(path):
        if line.strip():
            row = json.loads(line)
            if row.get(field) is not None:
                out.setdefault(row["id"], []).append((int(row["step"]), float(row[field])))
    return {k: [v for _, v in sorted(pairs)] for k, pairs in out.items()}


def candidates(methods: dict, signal: str) -> list[tuple]:
    """Every key the tables could have reported for this signal."""
    return [k for k in methods if k[0] == signal and "\\ensuremath" in k[1]]


def select(subset: dict, methods: dict, keys: list[tuple], seed: int) -> tuple:
    """The key with the best out-of-fold PRR on this subset alone."""
    best, choice = None, keys[0]
    for key in keys:
        values = [v for s in (seed,) if (v := oof(subset, methods[key], s)) is not None]
        if values and (best is None or values[0] > best):
            best, choice = values[0], key
    return choice


def nested(eps: dict, methods: dict, keys: list[tuple], seed: int) -> tuple[float | None, list[tuple]]:
    """Choose the mode inside each outer training fold, then apply it."""
    continuous = is_continuous(eps)
    pred: dict[str, float] = {}
    chosen: list[tuple] = []
    for fold in stratified_folds(eps, seed):
        held = set(fold)
        inner = {i: eps[i] for i in eps if i not in held}
        key = select(inner, methods, keys, seed + 100)
        chosen.append(key)
        train = with_labels(list(inner.values()), continuous)
        for i, c in zip(fold, methods[key](train, [eps[i] for i in fold])):
            if c is None:
                return None, chosen
            pred[i] = float(c)
    ids = list(eps)
    return prr([eps[i]["score"] for i in ids], [pred[i] for i in ids]), chosen


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--judge-dir", type=Path, required=True)
    p.add_argument("--toolkit", required=True)
    p.add_argument("--step-rule", default="giveup+acted")
    p.add_argument("--signals", nargs="+", default=["LLM judge", "MTE", "Decision p(success)"])
    a = p.parse_args()

    for label, (_, _, hiu) in INJECTED.items():
        SIGNALS[label] = (label, None, hiu)
    methods = method_table(a.toolkit)

    cohorts = {}
    for cohort, run in RUNS.items():
        eps = load_alfworld(a.runs / run, "react" if "react" in run else "smolagents", a.step_rule)
        eps = {k: v for k, v in eps.items() if v["n_steps"]}
        for label, (stem, field, _) in INJECTED.items():
            sequences = read_steps(a.judge_dir / f"{stem}_{cohort}.jsonl", field)
            for key, e in eps.items():
                e["signals"][label] = (sequences.get(key) or [])[: e["n_steps"]]
        cohorts[cohort] = eps

    order = list(RUNS)
    for signal in a.signals:
        keys = candidates(methods, signal)
        print(f"\n=== {signal}: {len(keys)} candidate modes ===", flush=True)
        started = time.time()

        # every mode, scored the way the tables score it
        grid: dict[tuple, dict[str, float]] = {}
        for key in keys:
            row = {}
            for cohort in order:
                values = [v for s in SEEDS if (v := oof(cohorts[cohort], methods[key], s)) is not None]
                if values:
                    row[cohort] = st.fmean(values)
            if len(row) == len(order):
                grid[key] = row
        if not grid:
            print("  no mode covers every cohort", flush=True)
            continue
        spread = sorted(st.fmean(r.values()) for r in grid.values())
        best_key = max(grid, key=lambda k: st.fmean(grid[k].values()))
        print(f"  spread across modes: {spread[0]:+.2f} to {spread[-1]:+.2f}  "
              f"(median {st.median(spread):+.2f})")
        print(f"  {'selected on the evaluation data':44} "
              + " ".join(f"{grid[best_key][c]:+6.2f}" for c in order)
              + f" {st.fmean(grid[best_key].values()):+7.2f}   {best_key[1].split('} ')[-1]} / {best_key[2]}")

        # chosen on the other three cohorts
        row, picks = {}, {}
        for cohort in order:
            others = [c for c in order if c != cohort]
            key = max(grid, key=lambda k: st.fmean(grid[k][c] for c in others))
            row[cohort] = grid[key][cohort]
            picks[cohort] = key[2]
        print(f"  {'chosen on the other three cohorts':44} "
              + " ".join(f"{row[c]:+6.2f}" for c in order)
              + f" {st.fmean(row.values()):+7.2f}   {Counter(picks.values()).most_common()}")

        # chosen inside each outer training fold
        row, flips = {}, []
        for cohort in order:
            values = []
            for seed in SEEDS:
                value, chosen = nested(cohorts[cohort], methods, keys, seed)
                if value is not None:
                    values.append(value)
                flips.append(len(set(chosen)))
            if values:
                row[cohort] = st.fmean(values)
        if len(row) == len(order):
            print(f"  {'chosen inside each training fold (nested)':44} "
                  + " ".join(f"{row[c]:+6.2f}" for c in order)
                  + f" {st.fmean(row.values()):+7.2f}   "
                  f"distinct modes per 5 folds: {st.fmean(flips):.1f}")
        print(f"  [{time.time() - started:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
