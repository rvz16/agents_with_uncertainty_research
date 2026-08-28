#!/usr/bin/env python3
"""Наши fusion-методы на экспорте (124 эпизода codecontests, 74 ошибки).

ВНИМАНИЕ: это данные "протекающей" эпохи. bayes_state там заражён промежуточным
verify, и через правило остановки заражены last-статистики. Но приор
bayes_state_after_generation и ряды llm_log_seq_prob по генерациям есть, поэтому
вопрос "добавляет ли наш fusion поверх честного belief на сбалансированном
бенчмарке" здесь ставится осмысленно — просто ответ направленный, не финальный.
Критиков в экспорте нет, поэтому belief_logit не считается.
"""

import _compat  # noqa: F401  # регистрирует code_uq.* / trajectory_uq_toolkit.*
import csv
import json
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "askutsakov" / "src"))
from code_uq.analysis.analyze_lcb_llm_tool_agent_logs import (  # noqa: E402
    prediction_rejection_area, prr)
from trajectory_uq_toolkit.bayes import BinaryBayes, DoubleBinaryBayes  # noqa: E402

root = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/Users/victor/Downloads/sage_uncertainty_export/gpt_oss_20b/codecontests")
FOLDS, SEEDS, BOOT = 4, 5, 1000

rows = {str(r["instance_id"]): r for r in
        csv.DictReader((root / "final_logprob_bayes_quality.csv").open())}


def num(r, key):
    try:
        v = float(r[key])
        return v if np.isfinite(v) else None
    except (TypeError, ValueError, KeyError):
        return None


series = defaultdict(list)
for line in (root / "generation_trajectory_scores.jsonl").open():
    if not line.strip():
        continue
    rec = json.loads(line)
    v = rec.get("llm_log_seq_prob")
    if isinstance(v, (int, float)) and np.isfinite(v):
        series[str(rec["instance_id"])].append((int(rec.get("action_step", 0)), float(v)))

insts, y, seq, prior = [], [], [], []
for i, r in rows.items():
    q = num(r, "quality")
    p = num(r, "bayes_state_after_generation")
    if q is None or p is None or not series.get(i):
        continue
    insts.append(i)
    y.append(int(q))
    seq.append([v for _, v in sorted(series[i])])
    prior.append(float(np.clip(p, 1e-6, 1 - 1e-6)))
y = np.array(y)
n = len(y)
print(f"{root.parent.name}/{root.name}: {n} эпизодов, pass@1 {y.mean():.3f}, "
      f"ошибок {n - y.sum()}, генераций/эпизод медиана "
      f"{int(np.median([len(s) for s in seq]))}\n")


def folds(seed):
    order = np.random.RandomState(seed).permutation(n)
    return [order[i::FOLDS] for i in range(FOLDS)]


def cross_fit(mode, seed):
    out = np.full(n, np.nan)
    parts = folds(seed)
    for k, held in enumerate(parts):
        train = np.concatenate([p for j, p in enumerate(parts) if j != k])
        cls = DoubleBinaryBayes if mode == "double" else BinaryBayes
        kw = {} if mode == "double" else {"mode": mode}
        model = cls.fit([seq[int(i)] for i in train], [int(y[int(i)]) for i in train],
                        higher_is_uncertain=False, **kw)
        for i in held:
            out[int(i)] = model.predict(seq[int(i)], start=prior[int(i)])
    return out


def agg(fn):
    return np.array([fn(s) for s in seq], dtype=float)


signals = {
    "bayes_state (заражён verify)": np.array([num(rows[i], "bayes_state") or np.nan for i in insts]),
    "bayes_after_gen (приор)": np.array(prior),
    "verbalized": np.array([num(rows[i], "verbalized_2s_confidence") or np.nan for i in insts]),
    "seqprob:last": agg(lambda s: s[-1]),
    "seqprob:mean": agg(np.mean),
    "seqprob:min": agg(min),
}
for mode, name in (("sep", "fusion sep"), ("double", "fusion double"),
                   ("lr_neg", "fusion lr_neg"), ("lr_pos", "fusion lr_pos")):
    signals[name] = np.nanmean(np.stack([cross_fit(mode, s) for s in range(SEEDS)]), axis=0)

print(f"{'сигнал':<32} {'PRR@0.5 / PRR@1.0':>18}")
print("-" * 52)
for name, v in signals.items():
    ok = np.isfinite(v)
    a = prr([float(x) for x in v[ok]], [int(q) for q in y[ok]], 0.5)
    b = prr([float(x) for x in v[ok]], [int(q) for q in y[ok]], 1.0)
    print(f"{name:<32} {a:+.3f} / {b:+.3f}")


@lru_cache(maxsize=None)
def refs(m, k, mr):
    lab = [1] * k + [0] * (m - k)
    oracle = prediction_rejection_area([-float(q) for q in lab], lab, mr)
    rng = np.random.RandomState(42)
    order = np.arange(m)
    areas = []
    for _ in range(500):
        rng.shuffle(order)
        a = prediction_rejection_area(order.tolist(), lab, mr)
        if a is not None:
            areas.append(a)
    return oracle, (float(np.mean(areas)) if areas else None)


def fast_prr(conf, lab, mr, rf):
    o, rnd = rf
    if o is None or rnd is None or abs(o - rnd) < 1e-12:
        return None
    if max(conf) == min(conf):
        return 0.0
    a = prediction_rejection_area([-c for c in conf], lab, mr)
    return None if a is None else (a - rnd) / (o - rnd)


def boot(na, nb, mr, seed=0):
    a, b = signals[na], signals[nb]
    idx = np.array([i for i in range(n) if np.isfinite(a[i]) and np.isfinite(b[i])])
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(BOOT):
        pick = idx[rng.randint(0, len(idx), len(idx))]
        lab = [int(y[i]) for i in pick]
        if len(set(lab)) < 2:
            continue
        rf = refs(len(lab), sum(lab), mr)
        pa = fast_prr([float(a[i]) for i in pick], lab, mr, rf)
        pb = fast_prr([float(b[i]) for i in pick], lab, mr, rf)
        if pa is not None and pb is not None:
            d.append(pa - pb)
    d = np.asarray(d)
    return d.mean(), np.quantile(d, .025), np.quantile(d, .975), (d > 0).mean()


pairs = [("fusion sep", "bayes_after_gen (приор)"), ("fusion double", "bayes_after_gen (приор)"),
         ("fusion lr_neg", "bayes_after_gen (приор)"), ("fusion sep", "seqprob:mean"),
         ("fusion double", "seqprob:mean"), ("seqprob:mean", "bayes_after_gen (приор)")]
for mr in (0.5, 1.0):
    print(f"\n--- парный бутстрап, PRR@{mr} ---")
    for a, b in pairs:
        m, lo, hi, p = boot(a, b, mr)
        print(f"{a + ' − ' + b:<50} {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  P={p:.2f}")
