#!/usr/bin/env python3
"""bayes+UQ против простой агрегации — обе с отбором варианта на train.

Сравнивать вложенно-отобранный bayes+UQ с лучшей ячейкой таблицы агрегаций
нечестно: вторая выбрана по тем же данным, на которых меряется. Здесь обе
стороны проходят одну процедуру — выбор варианта только по обучающим фолдам,
замер на отложенных, — и обе на ОДНИХ И ТЕХ ЖЕ разбиениях, поэтому бутстрап
парный.

Простая агрегация ничего не обучает, поэтому её вариант отбирается по PRR
на обучающей части напрямую; у bayes+UQ пороги обучаются, поэтому там
внутренняя кросс-валидация.
"""

import _compat  # noqa: F401  # регистрирует code_uq.* / trajectory_uq_toolkit.*
import argparse
import csv
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "askutsakov" / "src"))
sys.path.insert(0, str(HERE))
from code_uq.analysis.experiment2_uq_bayes_critic import (  # noqa: E402
    bayes_update, fit_threshold_theta, load_final)
from code_uq.analysis.analyze_lcb_llm_tool_agent_logs import (  # noqa: E402
    prediction_rejection_area, prr)
from bayes_cross_fit import cached_steps, episode_series, read_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run-root", type=Path, required=True)
ap.add_argument("--benchmark", required=True)
ap.add_argument("--generator", default="gpt_oss_20b_local")
ap.add_argument("--outer", type=int, default=5)
ap.add_argument("--inner", type=int, default=5)
ap.add_argument("--seeds", type=int, default=5)
args = ap.parse_args()

root, stem = args.run_root, f"{args.benchmark}__{args.generator}"
rd = root / "readable" / args.benchmark
eps = read_jsonl(root / f"{stem}.jsonl")
steps = cached_steps(root, stem, HERE / "cache")
ser = {str(e["instance_id"]): episode_series(e, steps) for e in eps}
rows = load_final(rd / "final_logprob_bayes_quality.csv", "llm_log_seq_prob")
ids = [r["iid"] for r in rows]
y = np.array([r["quality"] for r in rows])
bayes = np.array([r["bayes"] for r in rows], float)
n = len(ids)

AGGS = {"first": lambda v: v[0], "last": lambda v: v[-1], "max": max, "min": min,
        "mean": lambda v: float(np.mean(v)), "sum": lambda v: float(np.sum(v))}
METRICS = ["sum", "answer_sum", "ntok", "answer_mean", "mean", "min"]
MODES = ("sep", "lr_pos", "lr_neg", "double")

feat = {}
for m in METRICS:
    for a, fn in AGGS.items():
        col = np.full(n, np.nan)
        for j, i in enumerate(ids):
            v = ser.get(i, {}).get(m) or []
            if v:
                col[j] = float(fn(v))
        if np.isfinite(col).mean() > 0.8:
            feat[(m, a)] = col
verb = {}
with (rd / "final_logprob_bayes_quality.csv").open() as fh:
    for r in csv.DictReader(fh):
        try:
            verb[str(r["instance_id"])] = float(r["verbalized_2s_confidence"])
        except (TypeError, ValueError):
            pass
feat[("verbalized", "—")] = np.array([verb.get(i, np.nan) for i in ids])


def fuse(train, apply_to, key, mode):
    x = feat[key]
    tr = [i for i in train if np.isfinite(x[i])]
    if len({int(y[i]) for i in tr}) < 2:
        return None
    ms = ["lr_pos", "lr_neg"] if mode == "double" else [mode]
    fitted = [fit_threshold_theta([float(x[i]) for i in tr], [int(y[i]) for i in tr],
                                  True, mode=m) for m in ms]
    out = {}
    for i in apply_to:
        b = float(bayes[i])
        if np.isfinite(x[i]):
            for thr, p1, p0 in fitted:
                b = bayes_update(b, p1, p0, x[i] >= thr)
        out[i] = b
    return out


def folds(k, seed):
    order = np.random.RandomState(seed).permutation(n)
    return [order[i::k] for i in range(k)]


def train_prr(idx, vals):
    ok = [i for i in idx if np.isfinite(vals[i])]
    if len(ok) < 10 or len({int(y[i]) for i in ok}) < 2:
        return None
    return prr([float(vals[i]) for i in ok], [int(y[i]) for i in ok], 0.5)


oof_b, oof_s = np.full((args.seeds, n), np.nan), np.full((args.seeds, n), np.nan)
picks_s = defaultdict(int)
for seed in range(args.seeds):
    parts = folds(args.outer, seed)
    for f_i, held in enumerate(parts):
        train = np.concatenate([p for j, p in enumerate(parts) if j != f_i])
        # --- простая агрегация: обучать нечего, вариант выбираем по PRR на train
        best_s, best_v = None, -np.inf
        for key in feat:
            v = train_prr(train, feat[key])
            if v is not None and v > best_v:
                best_v, best_s = v, key
        if best_s:
            picks_s[best_s] += 1
            for i in held:
                oof_s[seed, i] = feat[best_s][i]
        # --- bayes+UQ: пороги обучаются => внутренняя кросс-валидация
        best_b, best_v = None, -np.inf
        inner_parts = folds(args.inner, seed + 100)
        for key in feat:
            for mode in MODES:
                inner = np.full(n, np.nan)
                for g_i, ih in enumerate(inner_parts):
                    ih_abs = np.array([i for i in ih if i in set(train.tolist())])
                    it_abs = np.array([i for i in train if i not in set(ih_abs.tolist())])
                    if len(ih_abs) == 0 or len(it_abs) < 10:
                        continue
                    got = fuse(it_abs, ih_abs, key, mode)
                    if got:
                        for i, val in got.items():
                            inner[i] = val
                v = train_prr(train, inner)
                if v is not None and v > best_v:
                    best_v, best_b = v, (key, mode)
        if best_b:
            got = fuse(train, held, *best_b)
            if got:
                for i, val in got.items():
                    oof_b[seed, i] = val

b = np.nanmean(oof_b, axis=0)
s = np.nanmean(oof_s, axis=0)


@lru_cache(maxsize=None)
def refs(m, k, mr):
    lab = [1] * k + [0] * (m - k)
    o = prediction_rejection_area([-float(x) for x in lab], lab, mr)
    rng = np.random.RandomState(42)
    order = np.arange(m)
    ar = []
    for _ in range(400):
        rng.shuffle(order)
        a = prediction_rejection_area(order.tolist(), lab, mr)
        if a is not None:
            ar.append(a)
    return o, (float(np.mean(ar)) if ar else None)


def fast(c, l, mr, rf):
    o, rn = rf
    if o is None or rn is None or abs(o - rn) < 1e-12:
        return None
    if max(c) == min(c):
        return 0.0
    a = prediction_rejection_area([-x for x in c], l, mr)
    return None if a is None else (a - rn) / (o - rn)


print(f"{args.benchmark}: n={n}, не решено {int((1-y).sum())}")
for mr in (0.5, 1.0):
    pb = prr(b.tolist(), y.tolist(), mr)
    ps = prr(s.tolist(), y.tolist(), mr)
    idx = np.arange(n)
    rng = np.random.RandomState(0)
    d = []
    for _ in range(2000):
        p = idx[rng.randint(0, n, n)]
        lab = [int(y[i]) for i in p]
        if len(set(lab)) < 2:
            continue
        rf = refs(len(lab), sum(lab), mr)
        pa, pbb = fast([b[i] for i in p], lab, mr, rf), fast([s[i] for i in p], lab, mr, rf)
        if pa is not None and pbb is not None:
            d.append(pa - pbb)
    d = np.asarray(d)
    print(f"  @{mr}: bayes+UQ {pb:+.3f}   простая агрегация {ps:+.3f}   "
          f"разность {d.mean():+.3f} [{np.quantile(d,.025):+.3f}, {np.quantile(d,.975):+.3f}] "
          f"P={np.mean(d>0):.2f}")
top = sorted(picks_s.items(), key=lambda kv: -kv[1])[:3]
print("  простая агрегация выбирала: " +
      ", ".join(f"{k[0]}:{k[1]} ({c}/{args.outer*args.seeds})" for k, c in top))
