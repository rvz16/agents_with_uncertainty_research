#!/usr/bin/env python3
"""Есть ли ОДНА связка Bayes+UQ, годная на всех датасетах?

Отбор варианта под каждый датасет — это риск переобучиться под датасет.
Здесь считается вся сетка (признак x агрегация x режим) на всех трёх прогонах
честной k-fold (обучаются только пороги и таблицы правдоподобия, выбор варианта
НЕ делается), после чего:

  1) варианты ранжируются по среднему и по худшему из трёх датасетов;
  2) leave-one-dataset-out: вариант выбирается по двум датасетам, меряется на
     третьем — это и есть честная оценка переносимости фиксированной связки.
"""

import _compat  # noqa: F401  # регистрирует code_uq.* / trajectory_uq_toolkit.*
import argparse, csv, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "askutsakov" / "src")); sys.path.insert(0, str(HERE))
from code_uq.analysis.experiment2_uq_bayes_critic import (bayes_update, continuous_bayes_update,
    fit_class_gaussians, fit_threshold_theta, load_final)
from code_uq.analysis.analyze_lcb_llm_tool_agent_logs import prr
from bayes_cross_fit import cached_steps, episode_series, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--max-rej", type=float, default=0.5)
a = ap.parse_args()

RUNS = [("lcb_hard", "gpt_oss_20b_local-3"), ("lcb_medium", "gpt_oss_20b_local-5"),
        ("codecontests", "gpt_oss_20b_local-4")]
AGGS = {"first": lambda v: v[0], "last": lambda v: v[-1], "max": max, "min": min,
        "mean": lambda v: float(np.mean(v)), "sum": lambda v: float(np.sum(v))}
METRICS = ["sum", "answer_sum", "ntok", "answer_mean", "mean", "min"]
MODES = ("sep", "lr_pos", "lr_neg", "double", "continuous", "tempered")

data = {}
for bench, d in RUNS:
    root = Path(f"/Users/victor/Downloads/{d}"); stem = f"{bench}__gpt_oss_20b_local"
    rd = root / "readable" / bench
    eps = read_jsonl(root / f"{stem}.jsonl")
    steps = cached_steps(root, stem, HERE / "cache")
    ser = {str(e["instance_id"]): episode_series(e, steps) for e in eps}
    rows = load_final(rd / "final_logprob_bayes_quality.csv", "llm_log_seq_prob")
    ids = [r["iid"] for r in rows]
    y = np.array([r["quality"] for r in rows])
    bayes = np.array([r["bayes"] for r in rows], float)
    verb = {}
    with (rd / "final_logprob_bayes_quality.csv").open() as fh:
        for r in csv.DictReader(fh):
            try: verb[str(r["instance_id"])] = float(r["verbalized_2s_confidence"])
            except (TypeError, ValueError): pass
    feat = {}
    for m in METRICS:
        for ag, fn in AGGS.items():
            c = np.full(len(ids), np.nan)
            for j, i in enumerate(ids):
                v = ser.get(i, {}).get(m) or []
                if v: c[j] = float(fn(v))
            if np.isfinite(c).mean() > 0.8: feat[(m, ag)] = c
    feat[("verbalized", "—")] = np.array([verb.get(i, np.nan) for i in ids])
    data[bench] = (y, bayes, feat)

def kfold_variant(bench, key, mode):
    y, bayes, feat = data[bench]
    x = feat.get(key)
    if x is None: return None
    n = len(y); scores = []
    for seed in range(a.seeds):
        order = np.random.RandomState(seed).permutation(n)
        parts = [order[i::a.folds] for i in range(a.folds)]
        out = np.full(n, np.nan)
        for k, held in enumerate(parts):
            train = np.concatenate([p for j, p in enumerate(parts) if j != k])
            tr = [i for i in train if np.isfinite(x[i])]
            if len({int(y[i]) for i in tr}) < 2: continue
            if mode in ("continuous", "tempered"):
                lam = 1.0 if mode == "continuous" else 0.25
                params = fit_class_gaussians([float(x[i]) for i in tr], [int(y[i]) for i in tr])
                for i in held:
                    b = float(bayes[i])
                    if np.isfinite(x[i]):
                        b = continuous_bayes_update(b, float(x[i]), params, lambda_=lam)
                    out[i] = b
                continue
            ms = ["lr_pos", "lr_neg"] if mode == "double" else [mode]
            try:
                fitted = [fit_threshold_theta([float(x[i]) for i in tr],
                                              [int(y[i]) for i in tr], True, mode=m) for m in ms]
            except Exception:
                continue
            for i in held:
                b = float(bayes[i])
                if np.isfinite(x[i]):
                    for thr, p1, p0 in fitted: b = bayes_update(b, p1, p0, x[i] >= thr)
                out[i] = b
        ok = [i for i in range(n) if np.isfinite(out[i])]
        if len(ok) > 10:
            v = prr([float(out[i]) for i in ok], [int(y[i]) for i in ok], a.max_rej)
            if v is not None: scores.append(v)
    return float(np.mean(scores)) if scores else None

keys = sorted(set().union(*[set(data[b][2]) for b, _ in RUNS]))
grid = {}
for key in keys:
    for mode in MODES:
        vals = {b: kfold_variant(b, key, mode) for b, _ in RUNS}
        if all(v is not None for v in vals.values()):
            grid[(key, mode)] = vals

benches = [b for b, _ in RUNS]
ranks = {b: {k: r for r, k in enumerate(sorted(grid, key=lambda k: -grid[k][b]))} for b in benches}
rows_out = []
for k, v in grid.items():
    rows_out.append((np.mean([v[b] for b in benches]), min(v[b] for b in benches),
                     np.mean([ranks[b][k] for b in benches]) + 1, k, v))

print(f"Всего связок: {len(grid)}.  Метрика PRR@{a.max_rej:g}, честная {a.folds}-fold "
      f"x {a.seeds} сида, БЕЗ отбора варианта.\n")
print(f"{'признак':<13}{'агрег.':<7}{'режим':<9}{'среднее':>9}{'худший':>8}{'ср.ранг':>9}   "
      + "".join(f"{b:>14}" for b in benches))
print("-" * 96)
for mean, worst, mrank, (key, mode), v in sorted(rows_out, reverse=True)[:12]:
    print(f"{key[0]:<13}{key[1]:<7}{mode:<9}{mean:>9.3f}{worst:>8.3f}{mrank:>9.1f}   "
          + "".join(f"{v[b]:>14.3f}" for b in benches))
print("\nЛучшие по ХУДШЕМУ датасету:")
for mean, worst, mrank, (key, mode), v in sorted(rows_out, key=lambda r: -r[1])[:5]:
    print(f"  {key[0]:<13}{key[1]:<7}{mode:<9} худший {worst:+.3f}, среднее {mean:+.3f}   "
          + "".join(f"{v[b]:>10.3f}" for b in benches))

print("\nLeave-one-dataset-out: вариант выбран по двум датасетам, замерен на третьем")
for held in benches:
    others = [b for b in benches if b != held]
    best = max(grid, key=lambda k: np.mean([grid[k][b] for b in others]))
    own = max(grid, key=lambda k: grid[k][held])
    print(f"  {held:<14} выбран {best[0][0]}/{best[0][1]}/{best[1]:<9} -> {grid[best][held]:+.3f}"
          f"   (потолок этого датасета {grid[own][held]:+.3f}, "
          f"потеря {grid[own][held]-grid[best][held]:+.3f})")
