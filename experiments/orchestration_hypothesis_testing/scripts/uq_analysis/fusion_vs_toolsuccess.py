#!/usr/bin/env python3
"""Наш fusion (bayes_state + sep, агрегация min) против тривиального tool_success.

Вопрос: даёт ли байесовская конструкция что-то сверх простого счётчика
"доля пройденных критиков". Всё на одних и тех же эпизодах, парный бутстрап.
"""

import _compat  # noqa: F401  # регистрирует code_uq.* / trajectory_uq_toolkit.*
import csv
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "askutsakov" / "src"))
from code_uq.analysis.experiment2_uq_bayes_critic import (  # noqa: E402
    kfold_fuse, load_final, load_trajectory_feature)
from code_uq.analysis.analyze_lcb_llm_tool_agent_logs import (  # noqa: E402
    prediction_rejection_area, prr)

rd = Path(sys.argv[1])
overrides = load_trajectory_feature(rd / "generation_trajectory_scores.jsonl",
                                    "llm_log_seq_prob", sys.argv[2] if len(sys.argv) > 2 else "min")
rows = load_final(rd / "final_logprob_bayes_quality.csv", "llm_log_seq_prob", overrides)
quality = [r["quality"] for r in rows]
ids = [r["iid"] for r in rows]

ts = {}
with (rd / "final_logprob_bayes_quality.csv").open() as fh:
    for r in csv.DictReader(fh):
        try:
            ts[str(r["instance_id"])] = float(r["tool_success"])
        except (TypeError, ValueError):
            pass

sig = {
    "bayes_state": [r["bayes"] for r in rows],
    "fusion sep+min": [kfold_fuse(rows, True, k=5, seed=0, mode="sep")[i] for i in ids],
    "tool_success": [ts.get(i, np.nan) for i in ids],
}
print(f"n={len(rows)}  ошибок={len(quality)-sum(quality)}\n")
for k, v in sig.items():
    print(f"  {k:<16} {prr(v, quality, 0.5):+.3f} / {prr(v, quality, 1.0):+.3f}")


@lru_cache(maxsize=None)
def refs(m, k, mr):
    lab = [1]*k + [0]*(m-k)
    oracle = prediction_rejection_area([-float(q) for q in lab], lab, mr)
    rng = np.random.RandomState(42); order = np.arange(m); areas = []
    for _ in range(500):
        rng.shuffle(order)
        a = prediction_rejection_area(order.tolist(), lab, mr)
        if a is not None:
            areas.append(a)
    return oracle, (float(np.mean(areas)) if areas else None)


def fast(conf, lab, mr, rf):
    o, rnd = rf
    if o is None or rnd is None or abs(o-rnd) < 1e-12:
        return None
    if max(conf) == min(conf):
        return 0.0
    a = prediction_rejection_area([-c for c in conf], lab, mr)
    return None if a is None else (a-rnd)/(o-rnd)


def boot(na, nb, mr):
    a, b = np.array(sig[na], float), np.array(sig[nb], float)
    idx = np.array([i for i in range(len(a)) if np.isfinite(a[i]) and np.isfinite(b[i])])
    rng = np.random.RandomState(0); d = []
    for _ in range(2000):
        pick = idx[rng.randint(0, len(idx), len(idx))]
        lab = [quality[i] for i in pick]
        if len(set(lab)) < 2:
            continue
        rf = refs(len(lab), sum(lab), mr)
        pa, pb = fast([a[i] for i in pick], lab, mr, rf), fast([b[i] for i in pick], lab, mr, rf)
        if pa is not None and pb is not None:
            d.append(pa-pb)
    d = np.asarray(d)
    return d.mean(), np.quantile(d, .025), np.quantile(d, .975), (d > 0).mean()


for mr in (0.5, 1.0):
    print(f"\n--- PRR@{mr} ---")
    for a, b in (("fusion sep+min", "tool_success"), ("fusion sep+min", "bayes_state"),
                 ("bayes_state", "tool_success")):
        m, lo, hi, p = boot(a, b, mr)
        print(f"  {a+' − '+b:<34} {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  P={p:.2f}")
