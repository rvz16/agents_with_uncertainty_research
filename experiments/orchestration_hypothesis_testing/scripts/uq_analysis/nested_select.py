#!/usr/bin/env python3
"""Честная оценка семейства «bayes + UQ» с автоматическим выбором варианта.

Таблица-перебор по ~150 комбинациям (признак x агрегация x режим) даёт смещённый
максимум: её верхняя строка выбрана по тем же данным, на которых измерена.
Здесь выбор варианта делается ВЛОЖЕННО — на обучающих фолдах, — а замеряется
на отложенных. Это то число, которое можно писать в статью.
"""

import _compat  # noqa: F401  # регистрирует code_uq.* / trajectory_uq_toolkit.*
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "askutsakov" / "src"))
sys.path.insert(0, str(HERE))
from code_uq.analysis.experiment2_uq_bayes_critic import (  # noqa: E402
    bayes_update, fit_threshold_theta, load_final)
from code_uq.analysis.analyze_lcb_llm_tool_agent_logs import prr  # noqa: E402
from bayes_cross_fit import cached_steps, episode_series, read_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run-root", type=Path, required=True)
ap.add_argument("--benchmark", required=True)
ap.add_argument("--generator", required=True)
ap.add_argument("--outer", type=int, default=5)
ap.add_argument("--inner", type=int, default=5)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--dump", type=Path)
args = ap.parse_args()

root, stem = args.run_root, f"{args.benchmark}__{args.generator}"
rd = root / "readable" / args.benchmark
episodes = read_jsonl(root / f"{stem}.jsonl")
steps = cached_steps(root, stem, HERE / "cache")
series = {str(e["instance_id"]): episode_series(e, steps) for e in episodes}
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
MODES = ("sep", "lr_pos", "lr_neg", "double")

rows = load_final(rd / "final_logprob_bayes_quality.csv", "llm_log_seq_prob")
ids = [r["iid"] for r in rows]
y = np.array([r["quality"] for r in rows])
bayes = np.array([r["bayes"] for r in rows], float)

feat = {}
for metric in METRICS:
    for aggname, fn in AGGS.items():
        col = np.full(len(ids), np.nan)
        for j, i in enumerate(ids):
            vals = series.get(i, {}).get(metric) or []
            if vals:
                col[j] = float(fn(vals))
        if np.isfinite(col).mean() > 0.8:
            feat[(metric, aggname)] = col
feat[("verbalized", "—")] = np.array([verb.get(i, np.nan) for i in ids])
COMBOS = [(k, m) for k in feat for m in MODES]
print(f"{args.benchmark}: n={len(ids)}, ошибок={int((1-y).sum())}, "
      f"комбинаций в поиске: {len(COMBOS)}")


def fuse(train_idx, apply_idx, key, mode):
    """Обучить критик(и) на train, применить к вере на apply."""
    x = feat[key]
    tr = [i for i in train_idx if np.isfinite(x[i])]
    if len({int(y[i]) for i in tr}) < 2:
        return None
    modes = ["lr_pos", "lr_neg"] if mode == "double" else [mode]
    fitted = []
    for m in modes:
        thr, p1, p0 = fit_threshold_theta([float(x[i]) for i in tr],
                                          [int(y[i]) for i in tr], True, mode=m)
        fitted.append((thr, p1, p0))
    out = {}
    for i in apply_idx:
        if not np.isfinite(x[i]):
            out[i] = float(bayes[i]); continue
        b = float(bayes[i])
        for thr, p1, p0 in fitted:
            b = bayes_update(b, p1, p0, x[i] >= thr)
        out[i] = b
    return out


def folds(n, k, seed):
    order = np.random.RandomState(seed).permutation(n)
    return [order[i::k] for i in range(k)]


n = len(ids)
per_seed, picks = [], defaultdict(int)
all_oof = []
for seed in range(args.seeds):
    oof = np.full(n, np.nan)
    for f_i, held in enumerate(folds(n, args.outer, seed)):
        train = np.concatenate([p for j, p in enumerate(folds(n, args.outer, seed)) if j != f_i])
        # --- внутренний отбор варианта только по train
        best, best_prr = None, -np.inf
        for key, mode in COMBOS:
            inner = np.full(n, np.nan)
            for g_i, ih in enumerate(folds(len(train), args.inner, seed + 100)):
                ih_abs = train[ih]
                it_abs = np.array([t for t in train if t not in set(ih_abs)])
                got = fuse(it_abs, ih_abs, key, mode)
                if got:
                    for i, v in got.items():
                        inner[i] = v
            ok = [i for i in train if np.isfinite(inner[i])]
            if len(ok) < 10:
                continue
            p = prr([float(inner[i]) for i in ok], [int(y[i]) for i in ok], 0.5)
            if p is not None and p > best_prr:
                best_prr, best = p, (key, mode)
        if best is None:
            continue
        picks[best] += 1
        got = fuse(train, held, *best)
        if got:
            for i, v in got.items():
                oof[i] = v
    ok = [i for i in range(n) if np.isfinite(oof[i])]
    all_oof.append(oof.copy())
    per_seed.append((prr([float(oof[i]) for i in ok], [int(y[i]) for i in ok], 0.5),
                     prr([float(oof[i]) for i in ok], [int(y[i]) for i in ok], 1.0)))

b05 = prr(bayes.tolist(), y.tolist(), 0.5)
b10 = prr(bayes.tolist(), y.tolist(), 1.0)
a = float(np.mean([p[0] for p in per_seed])); b = float(np.mean([p[1] for p in per_seed]))
print(f"\nbayes_state (приор)                      {b05:+.3f} / {b10:+.3f}")
print(f"bayes + UQ, вариант выбран вложенно     {a:+.3f} / {b:+.3f}   "
      f"(sd {np.std([p[0] for p in per_seed]):.3f} / {np.std([p[1] for p in per_seed]):.3f})")
print(f"дельта                                  {a-b05:+.3f} / {b-b10:+.3f}")
print("\nчто выбирал внутренний отбор (топ-6):")
for (key, mode), c in sorted(picks.items(), key=lambda kv: -kv[1])[:6]:
    print(f"  {key[0]:<12}{key[1]:<7}{mode:<10} выбран {c}/{args.outer*args.seeds} раз")

if args.dump:
    np.savez(args.dump, oof=np.nanmean(np.stack(all_oof), axis=0),
             y=y, ids=np.array(ids, dtype=object), bayes=bayes)
    print(f"\nOOF-предсказания сохранены: {args.dump}")
