#!/usr/bin/env python3
"""Полная сетка «bayes + UQ»: какой сигнал вливать, как агрегировать, каким режимом.

До сих пор считался только один UQ-признак — llm_log_seq_prob (полная сумма
логпробов). Здесь в ту же честную k-fold конструкцию (kfold_fuse из
experiment2_uq_bayes_critic) подставляются все per-generation метрики из
сайдкара логпробов, включая answer-канал, длину и self-certainty, а также
per-instance verbalized. Приор — bayes_state (можно переключить на
bayes_state_after_generation).
"""

import _compat  # noqa: F401  # регистрирует code_uq.* / trajectory_uq_toolkit.*
import argparse
import csv
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "askutsakov" / "src"))
sys.path.insert(0, str(HERE))
from code_uq.analysis.experiment2_uq_bayes_critic import (  # noqa: E402
    kfold_continuous_fuse, kfold_fuse, load_final)
from code_uq.analysis.analyze_lcb_llm_tool_agent_logs import (  # noqa: E402
    prediction_rejection_area, prr)
from bayes_cross_fit import cached_steps, episode_series, read_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run-root", type=Path, required=True)
ap.add_argument("--benchmark", required=True)
ap.add_argument("--generator", required=True)
ap.add_argument("--prior", default="bayes_state",
                choices=["bayes_state", "bayes_state_after_generation"])
ap.add_argument("--max-rej", type=float, default=0.5)
ap.add_argument("--boot", type=int, default=1000)
args = ap.parse_args()

root, stem = args.run_root, f"{args.benchmark}__{args.generator}"
rd = root / "readable" / args.benchmark
episodes = read_jsonl(root / f"{stem}.jsonl")
steps = cached_steps(root, stem, HERE / "cache")
series = {str(e["instance_id"]): episode_series(e, steps) for e in episodes}

# per-instance verbalized как отдельный «признак без агрегации»
verb = {}
with (rd / "final_logprob_bayes_quality.csv").open() as fh:
    for r in csv.DictReader(fh):
        try:
            verb[str(r["instance_id"])] = float(r["verbalized_2s_confidence"])
        except (TypeError, ValueError):
            pass

AGGS = {"last": lambda v: v[-1], "mean": lambda v: float(np.mean(v)),
        "min": min, "max": max, "first": lambda v: v[0],
        "sum": lambda v: float(np.sum(v))}
METRICS = ["sum", "answer_sum", "ntok", "answer_mean", "mean", "min"]

rows0 = load_final(rd / "final_logprob_bayes_quality.csv", "llm_log_seq_prob")
ids = [r["iid"] for r in rows0]
quality = [r["quality"] for r in rows0]
base = prr([r["bayes"] for r in rows0], quality, args.max_rej)


@lru_cache(maxsize=None)
def refs(m, k, mr):
    lab = [1]*k + [0]*(m-k)
    o = prediction_rejection_area([-float(q) for q in lab], lab, mr)
    rng = np.random.RandomState(42); order = np.arange(m); ar = []
    for _ in range(500):
        rng.shuffle(order)
        a = prediction_rejection_area(order.tolist(), lab, mr)
        if a is not None:
            ar.append(a)
    return o, (float(np.mean(ar)) if ar else None)


def fast(conf, lab, mr, rf):
    o, rnd = rf
    if o is None or rnd is None or abs(o-rnd) < 1e-12:
        return None
    if max(conf) == min(conf):
        return 0.0
    a = prediction_rejection_area([-c for c in conf], lab, mr)
    return None if a is None else (a-rnd)/(o-rnd)


def ci_vs_base(vals):
    a, b = np.array(vals, float), np.array([r["bayes"] for r in rows0], float)
    rng = np.random.RandomState(0); idx = np.arange(len(a)); d = []
    for _ in range(args.boot):
        pick = idx[rng.randint(0, len(idx), len(idx))]
        lab = [quality[i] for i in pick]
        if len(set(lab)) < 2:
            continue
        rf = refs(len(lab), sum(lab), args.max_rej)
        pa, pb = fast([a[i] for i in pick], lab, args.max_rej, rf), \
            fast([b[i] for i in pick], lab, args.max_rej, rf)
        if pa is not None and pb is not None:
            d.append(pa-pb)
    d = np.asarray(d)
    return float(np.quantile(d, .025)), float(np.quantile(d, .975))


print(f"{args.benchmark}: n={len(ids)}, ошибок={len(quality)-sum(quality)}, "
      f"приор={args.prior}, PRR@{args.max_rej:g} приора = {base:+.3f}\n")
print(f"{'UQ-признак':<18}{'агрег.':<8}{'режим':<12}{'PRR':>8}{'дельта':>9}  95% CI")
print("-" * 72)

results = []
combos = [(m, a) for m in METRICS for a in AGGS] + [("verbalized", "—")]
for metric, aggname in combos:
    if metric == "verbalized":
        ov = dict(verb)
    else:
        ov = {}
        for i in ids:
            vals = series.get(i, {}).get(metric) or []
            if vals:
                ov[i] = float(AGGS[aggname](vals))
    if len(ov) < len(ids) * 0.8:
        continue
    rows = load_final(rd / "final_logprob_bayes_quality.csv", "llm_log_seq_prob", ov)
    if args.prior != "bayes_state":
        with (rd / "final_logprob_bayes_quality.csv").open() as fh:
            pr = {str(r["instance_id"]): r.get(args.prior) for r in csv.DictReader(fh)}
        for r in rows:
            try:
                r["bayes"] = float(pr[r["iid"]])
            except (TypeError, ValueError):
                pass
    for mode in ("sep", "lr_pos", "lr_neg", "double", "continuous", "tempered"):
        if mode == "continuous":
            fused = kfold_continuous_fuse(rows, k=5, seed=0, lambda_=1.0)
        elif mode == "tempered":
            fused = kfold_continuous_fuse(rows, k=5, seed=0, lambda_=0.25)
        else:
            fused = kfold_fuse(rows, True, k=5, seed=0, mode=mode)
        vals = [fused[i] for i in ids]
        p = prr(vals, quality, args.max_rej)
        results.append((p - base, p, metric, aggname, mode, vals))

results.sort(reverse=True)
for delta, p, metric, aggname, mode, vals in results[:18]:
    lo, hi = ci_vs_base(vals)
    star = " ✔" if lo > 0 else ""
    print(f"{metric:<18}{aggname:<8}{mode:<12}{p:>8.3f}{delta:>+9.3f}  "
          f"[{lo:+.3f}, {hi:+.3f}]{star}")
