"""Learning curves on ALFWorld: PRR@0.5 against the number of training episodes.

Same protocol as prr_report_v2 (3 seeds x 5 stratified folds, pooled OOF PRR
per seed, mean +- SD over seeds), except that each fold's training set (the
other four folds) is subsampled to n episodes, stratified on the label with the
seed's RNG, before fitting. Raw UQ needs no training and is drawn as a constant
reference. Curves: raw MTE (last), Bayes Fused MTE (Last only: tempered step
critics + one MTE update), logistic regression (pinned C=0.03), TemporalBelief B4.

    python -m experiments.learning_curves --toolkit agentic-uq/src --out reports/learning_curves_alfworld \\
        --alfworld RG runs/react_gptoss_140_giveup_sc_finished_nolast ...
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

from experiments.prr_report_v2 import (FOLDS, SEEDS, load_alfworld, method_table, prr, stratified_folds,
                                       is_continuous, with_labels)

CURVES = {
    "MTE — raw UQ, last (no training)": ("MTE", "MTE", "last"),
    "MTE — Bayes Fused, Last only": ("MTE", "MTE \\ensuremath{-} Bayes UQ + tools", "Last only"),
    "Self-certainty — raw UQ, last (no training)": ("Self-certainty", "Self-certainty", "last"),
    "Self-certainty — Bayes Fused, Last only": ("Self-certainty", "Self-certainty \\ensuremath{-} Bayes UQ + tools", "Last only"),
    "Logistic regression, pinned C=0.03": ("Reference", "Logistic regression", "pinned"),
    "TemporalBelief / B4": ("Reference", "TemporalBelief (B4)", "final checkpoint"),
}
STYLE = {
    "MTE — raw UQ, last (no training)": dict(color="0.3", ls="--", marker=None),
    "MTE — Bayes Fused, Last only": dict(color="#1f77b4", ls="-", marker="o"),
    "Self-certainty — raw UQ, last (no training)": dict(color="#2ca25f", ls="--", marker=None),
    "Self-certainty — Bayes Fused, Last only": dict(color="#2ca25f", ls="-", marker="^"),
    "Logistic regression, pinned C=0.03": dict(color="#e6550d", ls="-", marker="s"),
    "TemporalBelief / B4": dict(color="#7b4fbf", ls="-", marker="D"),
}


def subsample(train: list[dict], n: int, rng: np.random.RandomState) -> list[dict]:
    """n episodes, label-stratified, at least one of each class when possible."""
    if n >= len(train):
        return train
    by = {0: [e for e in train if e["label"] == 0], 1: [e for e in train if e["label"] == 1]}
    share = {c: len(v) / len(train) for c, v in by.items()}
    take = {c: max(1, int(round(n * share[c]))) if by[c] else 0 for c in by}
    while sum(take.values()) > n:
        c = max(take, key=take.get); take[c] -= 1
    out = []
    for c, v in by.items():
        idx = rng.choice(len(v), size=min(take[c], len(v)), replace=False)
        out += [v[i] for i in idx]
    return out


def curve(eps: dict, fn, n: int, seed: int) -> float | None:
    continuous = is_continuous(eps); folds = stratified_folds(eps, seed); pred = {}
    rng = np.random.RandomState(1000 * seed + n)
    for fold in folds:
        held = set(fold)
        train = with_labels([eps[i] for i in eps if i not in held], continuous)
        train = subsample(train, n, rng)
        if len({e["label"] for e in train}) < 2:
            return None
        for i, c in zip(fold, fn(train, [eps[i] for i in fold])):
            pred[i] = c
    ids = list(eps)
    if any(pred[i] is None for i in ids):
        return None
    return prr([eps[i]["score"] for i in ids], [pred[i] for i in ids])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alfworld", nargs=2, action="append", metavar=("KEY", "RUN_DIR"), required=True)
    p.add_argument("--toolkit", required=True)
    p.add_argument("--out", type=Path, required=True, help="stem: .json, .png and .pdf are written")
    p.add_argument("--step", type=int, default=10)
    a = p.parse_args()
    methods = method_table(a.toolkit)
    previous = json.load(open(a.out.with_suffix(".json"))) if a.out.with_suffix(".json").exists() else {}
    results: dict[str, dict] = {}
    for key, run in a.alfworld:
        eps = load_alfworld(Path(run), key)
        full = int(round(len(eps) * (FOLDS - 1) / FOLDS))
        grid = list(range(a.step, full, a.step)) + [full]
        results[key] = {"n_episodes": len(eps), "full": full, "grid": grid, "curves": {}}
        for label, mkey in CURVES.items():
            fn = methods[mkey]
            done = (previous.get(key) or {}).get("curves", {}).get(label)
            if done and (previous[key].get("grid") == grid or "constant" in done):
                results[key]["curves"][label] = done; continue
            if "raw UQ" in label:
                vals = [curve(eps, fn, full, s) for s in SEEDS]
                results[key]["curves"][label] = {"constant": [st.fmean(v for v in vals if v is not None)]}
                continue
            rows = []
            for n in grid:
                vals = [curve(eps, fn, n, s) for s in SEEDS]
                ok = [v for v in vals if v is not None]
                rows.append((st.fmean(ok), st.stdev(ok) if len(ok) > 1 else 0.0) if ok else (None, None))
                print(f"[curve] {key} {label[:28]:28s} n={n:4d} PRR={rows[-1][0]}", flush=True)
            results[key]["curves"][label] = {"mean": [r[0] for r in rows], "sd": [r[1] for r in rows]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(results, indent=1))
    plot(results, a.out)


def plot(results: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(results)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.9), sharey=True)
    axes = np.atleast_1d(axes)
    lo, hi = 1.0, -1.0
    for ax, key in zip(axes, keys):
        r = results[key]; grid = r["grid"]
        for label, c in r["curves"].items():
            sty = STYLE[label]
            if "constant" in c:
                ax.axhline(c["constant"][0], color=sty["color"], ls=sty["ls"], lw=1.2, label=label)
                lo, hi = min(lo, c["constant"][0]), max(hi, c["constant"][0]); continue
            m = np.array([v if v is not None else np.nan for v in c["mean"]], dtype=float)
            s = np.array([v if v is not None else np.nan for v in c["sd"]], dtype=float)
            ax.plot(grid, m, color=sty["color"], ls=sty["ls"], marker=sty["marker"], ms=3, lw=1.4, label=label)
            ax.fill_between(grid, m - s, m + s, color=sty["color"], alpha=0.15, lw=0)
            lo, hi = min(lo, np.nanmin(m - s)), max(hi, np.nanmax(m + s))
        ax.set_title(f"ALFWorld · {key}", loc="left", fontsize=11, fontweight="bold", pad=20)
        ax.annotate(f"Full training: {r['full']}/fold · OOF episodes = {r['n_episodes']}", xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 4), textcoords="offset points", fontsize=7.5, color="0.35")
        ax.set_xlabel("Training episodes per source fold", fontsize=8.5)
        ticks = [g for g in grid[:-1] if g % 20 == 0 or g == grid[0]] + [grid[-1]]
        ax.set_xticks(ticks); ax.set_xticklabels([str(t) if t != grid[-1] else f"Full\n{t}" for t in ticks], fontsize=7.5)
        ax.axvline(grid[-1], color="0.8", lw=0.8, ls=":")
        ax.grid(True, axis="y", color="0.9", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=7.5)
    axes[0].set_ylabel("PRR@0.5", fontsize=8.5)
    axes[0].set_ylim(min(-0.05, lo - 0.03), max(0.95, hi + 0.03))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Learning curves: ALFWorld", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.90, "PRR@0.5 · 3 seeds × 5 folds · mean ± 1 SD · common vertical scale · pre-terminal finished cohorts", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0.13, 1, 0.9))
    fig.savefig(out.with_suffix(".png"), dpi=170); fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out.with_suffix('.png')} and .pdf")


if __name__ == "__main__":
    main()
