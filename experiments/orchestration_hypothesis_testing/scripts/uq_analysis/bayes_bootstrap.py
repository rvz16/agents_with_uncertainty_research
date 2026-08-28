#!/usr/bin/env python3
"""Paired bootstrap over episodes for the key PRR contrasts.

Cross-fitted scores are averaged over split seeds first (one score per
episode), then episodes are resampled with replacement; PRR is recomputed for
both members of a pair on each resample, so the pairing is preserved.
"""
from __future__ import annotations

import _compat  # noqa: F401  # регистрирует code_uq.* / trajectory_uq_toolkit.*

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "askutsakov" / "src"))
sys.path.insert(0, str(HERE))

from code_uq.analysis.analyze_lcb_llm_tool_agent_logs import (  # noqa: E402
    prediction_rejection_area, prr)
from functools import lru_cache  # noqa: E402
from bayes_cross_fit import (  # noqa: E402
    STEP_FEATURES, cached_steps, col, cross_fitted, episode_series, folds,
    published, read_jsonl,
)
from code_uq.analysis import uq_features as UF  # noqa: E402
from code_uq.analysis.bayes_trajectory import belief_logit, calibrate  # noqa: E402
from trajectory_uq_toolkit.bayes import BinaryBayes  # noqa: E402


@lru_cache(maxsize=None)
def _refs(n: int, k: int, max_rej: float):
    """Oracle and random reference areas. Both depend on the label multiset
    only -- the random baseline is an average over shuffles -- so memoising on
    (n, k) turns the inner loop of the bootstrap from 1000 shuffles per PRR
    call into one computation per distinct success count."""
    labels = [1] * k + [0] * (n - k)
    oracle = prediction_rejection_area([-float(q) for q in labels], labels, max_rej)
    rng = np.random.RandomState(42)
    order = np.arange(n)
    areas = []
    for _ in range(500):
        rng.shuffle(order)
        area = prediction_rejection_area(order.tolist(), labels, max_rej)
        if area is not None:
            areas.append(area)
    return oracle, (float(np.mean(areas)) if areas else None)


def _prr(conf, labels, max_rej, refs):
    oracle, random = refs
    if oracle is None or random is None or abs(oracle - random) < 1e-12:
        return None
    if max(conf) == min(conf):
        return 0.0
    area = prediction_rejection_area([-c for c in conf], labels, max_rej)
    return None if area is None else (area - random) / (oracle - random)


def boot(a, b, labels, keep, n_boot, max_rej, seed=0):
    rng = np.random.RandomState(seed)
    idx = np.array([i for i in keep if np.isfinite(a[i]) and np.isfinite(b[i])])
    deltas = []
    for _ in range(n_boot):
        pick = idx[rng.randint(0, len(idx), len(idx))]
        y = [int(labels[i]) for i in pick]
        if len(set(y)) < 2:
            continue
        refs = _refs(len(y), sum(y), max_rej)
        pa = _prr([float(a[i]) for i in pick], y, max_rej, refs)
        pb = _prr([float(b[i]) for i in pick], y, max_rej, refs)
        if pa is not None and pb is not None:
            deltas.append(pa - pb)
    d = np.asarray(deltas)
    return float(d.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)), \
        float((d > 0).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--cache-dir", type=Path, default=HERE / "cache")
    args = ap.parse_args()

    root, stem = args.run_root, f"{args.benchmark}__{args.generator}"
    episodes = read_jsonl(root / f"{stem}.jsonl")
    steps = cached_steps(root, stem, args.cache_dir)
    cap = max((int(e.get("n_steps") or 0) for e in episodes), default=0)
    rows = [UF.episode_features(e, steps, cap) for e in episodes]
    series = [episode_series(e, steps) for e in episodes]
    labels = np.array([int(bool(e["fixed"])) for e in episodes])
    instances = [str(e["instance_id"]) for e in episodes]
    n = len(labels)
    table = published(root / "readable" / args.benchmark / "final_logprob_bayes_quality.csv")
    keep = [i for i, inst in enumerate(instances) if inst in table] or list(range(n))

    def student(train, held):
        cal = calibrate([rows[int(i)] for i in train], labels[train],
                        [series[int(i)] for i in train], STEP_FEATURES)
        return [belief_logit(cal, rows[int(i)], series[int(i)], STEP_FEATURES)
                for i in held]

    prior_col = col(table, instances, "bayes_state_after_generation")

    def sep_fusion(train, held):
        seqs = [series[int(i)].get("sum", []) for i in train]
        labs = [int(labels[int(i)]) for i in train]
        usable = [(s, l) for s, l in zip(seqs, labs) if s]
        model = BinaryBayes.fit([s for s, _ in usable], [l for _, l in usable],
                                mode="sep", higher_is_uncertain=False)
        out = []
        for i in held:
            seq, start = series[int(i)].get("sum", []), prior_col[int(i)]
            out.append(np.nan if not seq or not np.isfinite(start)
                       else model.predict(seq, start=float(np.clip(start, 1e-6, 1 - 1e-6))))
        return out

    def avg_cross(fp):
        acc = np.stack([cross_fitted(fp, n, args.folds, s) for s in range(args.seeds)])
        return np.nanmean(acc, axis=0)

    signals = {
        "belief_logit": avg_cross(student),
        "fusion_sep": avg_cross(sep_fusion),
        "sum:last": np.array([r.get("sum:last", np.nan) for r in rows], dtype=float),
        "sum:min": np.array([r.get("sum:min", np.nan) for r in rows], dtype=float),
        "ntok:mean": np.array([r.get("ntok:mean", np.nan) for r in rows], dtype=float),
        "sum:mean": np.array([r.get("sum:mean", np.nan) for r in rows], dtype=float),
        "verbalized": col(table, instances, "verbalized_2s_confidence"),
        "bayes_state": col(table, instances, "bayes_state"),
        "bayes_after_gen": col(table, instances, "bayes_state_after_generation"),
    }

    pairs = [
        # the published scheme, and the kernel-free correction
        ("belief_logit", "bayes_state"), ("belief_logit", "bayes_after_gen"),
        ("belief_logit", "verbalized"), ("belief_logit", "ntok:mean"),
        ("belief_logit", "sum:mean"), ("belief_logit", "fusion_sep"),
        # our aggregation claim: mean/min over the trajectory vs the last generation
        ("sum:mean", "sum:last"), ("sum:min", "sum:last"),
        ("ntok:mean", "verbalized"), ("sum:mean", "verbalized"),
        # our experiment2 fusion, into the honest belief
        ("fusion_sep", "bayes_after_gen"), ("fusion_sep", "bayes_state"),
        ("fusion_sep", "ntok:mean"), ("fusion_sep", "verbalized"),
    ]

    print(f"paired bootstrap, {args.boot} resamples, {len(keep)} episodes, "
          f"{int((1 - labels[keep]).sum())} errors\n")
    for max_rej in (0.5, 1.0):
        print(f"--- PRR@{max_rej} " + "-" * 52)
        print(f"{'contrast':<34} {'delta':>7}  {'95% CI':>18}  {'P(>0)':>6}")
        for a, b in pairs:
            m, lo, hi, p = boot(signals[a], signals[b], labels, keep, args.boot, max_rej)
            print(f"{a + ' - ' + b:<34} {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {p:5.2f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
