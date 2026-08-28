#!/usr/bin/env python3
"""Во что вливать UQ: в байесовский belief или прямо в tool_success?

bayes_state (tool only) во всех шести ячейках хуже тривиального tool_success.
Значит напрашивается вопрос: нужен ли belief вообще, если можно тем же
механизмом обновить tool_success — он тоже в [0,1] и тоже строится из критиков.
Здесь оба варианта проходят одинаковый вложенный отбор на одних разбиениях.
"""

import _compat  # noqa: F401  # регистрирует code_uq.* / trajectory_uq_toolkit.*
import argparse, csv, sys
from functools import lru_cache
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "askutsakov" / "src")); sys.path.insert(0, str(HERE))
from code_uq.analysis.experiment2_uq_bayes_critic import bayes_update, fit_threshold_theta, load_final
from code_uq.analysis.analyze_lcb_llm_tool_agent_logs import prediction_rejection_area, prr
from bayes_cross_fit import cached_steps, episode_series, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--run-root", type=Path, required=True)
ap.add_argument("--benchmark", required=True)
ap.add_argument("--generator", default="gpt_oss_20b_local")
ap.add_argument("--outer", type=int, default=5); ap.add_argument("--inner", type=int, default=5)
ap.add_argument("--seeds", type=int, default=5)
a = ap.parse_args()

root, stem = a.run_root, f"{a.benchmark}__{a.generator}"
rd = root / "readable" / a.benchmark
eps = read_jsonl(root / f"{stem}.jsonl")
steps = cached_steps(root, stem, HERE / "cache")
ser = {str(e["instance_id"]): episode_series(e, steps) for e in eps}
rows = load_final(rd / "final_logprob_bayes_quality.csv", "llm_log_seq_prob")
ids = [r["iid"] for r in rows]; y = np.array([r["quality"] for r in rows]); n = len(ids)

raw = {str(r["instance_id"]): r for r in csv.DictReader((rd/"final_logprob_bayes_quality.csv").open())}
def col(key):
    out = np.full(n, np.nan)
    for j, i in enumerate(ids):
        try: out[j] = float(raw[i][key])
        except (TypeError, ValueError, KeyError): pass
    return out
PRIORS = {"bayes_state": np.clip(col("bayes_state"), 1e-6, 1-1e-6),
          "tool_success": np.clip(col("tool_success"), 1e-6, 1-1e-6)}

AGGS = {"first": lambda v: v[0], "last": lambda v: v[-1], "max": max, "min": min,
        "mean": lambda v: float(np.mean(v)), "sum": lambda v: float(np.sum(v))}
feat = {}
for m in ["sum","answer_sum","ntok","answer_mean","mean","min"]:
    for ag, fn in AGGS.items():
        c = np.full(n, np.nan)
        for j, i in enumerate(ids):
            v = ser.get(i, {}).get(m) or []
            if v: c[j] = float(fn(v))
        if np.isfinite(c).mean() > 0.8: feat[(m, ag)] = c
MODES = ("sep","lr_pos","lr_neg","double")

def fuse(train, apply_to, key, mode, prior):
    x = feat[key]; tr = [i for i in train if np.isfinite(x[i])]
    if len({int(y[i]) for i in tr}) < 2: return None
    ms = ["lr_pos","lr_neg"] if mode == "double" else [mode]
    fitted = [fit_threshold_theta([float(x[i]) for i in tr], [int(y[i]) for i in tr], True, mode=m) for m in ms]
    out = {}
    for i in apply_to:
        b = float(prior[i])
        if np.isfinite(x[i]) and np.isfinite(b):
            for thr, p1, p0 in fitted: b = bayes_update(b, p1, p0, x[i] >= thr)
        out[i] = b
    return out

def folds(k, seed):
    o = np.random.RandomState(seed).permutation(n); return [o[i::k] for i in range(k)]
def tprr(idx, v):
    ok = [i for i in idx if np.isfinite(v[i])]
    if len(ok) < 10 or len({int(y[i]) for i in ok}) < 2: return None
    return prr([float(v[i]) for i in ok], [int(y[i]) for i in ok], 0.5)
def rank_norm(A):
    out = np.full_like(A, np.nan, dtype=float)
    for r in range(A.shape[0]):
        ok = np.where(np.isfinite(A[r]))[0]
        if len(ok) > 1: out[r, ok] = np.argsort(np.argsort(A[r][ok])) / (len(ok)-1)
    return out

oof = {p: np.full((a.seeds, n), np.nan) for p in PRIORS}
for seed in range(a.seeds):
    parts = folds(a.outer, seed); inner_parts = folds(a.inner, seed+100)
    for f_i, held in enumerate(parts):
        train = np.concatenate([p for j, p in enumerate(parts) if j != f_i])
        tset = set(train.tolist())
        for pname, prior in PRIORS.items():
            best, bv = None, -np.inf
            for key in feat:
                for mode in MODES:
                    inner = np.full(n, np.nan)
                    for ih in inner_parts:
                        ih_abs = np.array([i for i in ih if i in tset])
                        it_abs = np.array([i for i in train if i not in set(ih_abs.tolist())])
                        if len(ih_abs) == 0 or len(it_abs) < 10: continue
                        got = fuse(it_abs, ih_abs, key, mode, prior)
                        if got:
                            for i, v in got.items(): inner[i] = v
                    v = tprr(train, inner)
                    if v is not None and v > bv: bv, best = v, (key, mode)
            if best:
                got = fuse(train, held, *best, prior)
                if got:
                    for i, v in got.items(): oof[pname][seed, i] = v

@lru_cache(maxsize=None)
def refs(m, k, mr):
    lab = [1]*k + [0]*(m-k); o = prediction_rejection_area([-float(x) for x in lab], lab, mr)
    rng = np.random.RandomState(42); order = np.arange(m); ar = []
    for _ in range(400):
        rng.shuffle(order); v = prediction_rejection_area(order.tolist(), lab, mr)
        if v is not None: ar.append(v)
    return o, (float(np.mean(ar)) if ar else None)
def fast(c, l, mr, rf):
    o, rn = rf
    if o is None or rn is None or abs(o-rn) < 1e-12: return None
    if max(c) == min(c): return 0.0
    v = prediction_rejection_area([-x for x in c], l, mr)
    return None if v is None else (v-rn)/(o-rn)
def boot(A, B, mr):
    rng = np.random.RandomState(0); idx = np.arange(n); d = []
    for _ in range(2000):
        p = idx[rng.randint(0, n, n)]; lab = [int(y[i]) for i in p]
        if len(set(lab)) < 2: continue
        rf = refs(len(lab), sum(lab), mr)
        pa, pb = fast([A[i] for i in p], lab, mr, rf), fast([B[i] for i in p], lab, mr, rf)
        if pa is not None and pb is not None: d.append(pa-pb)
    d = np.asarray(d); return d.mean(), np.quantile(d,.025), np.quantile(d,.975), (d>0).mean()

bs = np.nanmean(rank_norm(oof["bayes_state"]), axis=0)
ts = np.nanmean(rank_norm(oof["tool_success"]), axis=0)
print(f"\n{a.benchmark}: n={n}, не решено {int((1-y).sum())}")
for mr in (0.5, 1.0):
    p_b = prr(PRIORS['bayes_state'].tolist(), y.tolist(), mr)
    p_t = prr(PRIORS['tool_success'].tolist(), y.tolist(), mr)
    m1, l1, h1, q1 = boot(PRIORS['tool_success'], PRIORS['bayes_state'], mr)
    m2, l2, h2, q2 = boot(bs, ts, mr)
    print(f"  @{mr}: bayes_state {p_b:+.3f} | tool_success {p_t:+.3f} | "
          f"tool−bayes {m1:+.3f} [{l1:+.3f}, {h1:+.3f}] P={q1:.2f}")
    print(f"        bayes+UQ {prr(bs.tolist(),y.tolist(),mr):+.3f} | "
          f"tool_success+UQ {prr(ts.tolist(),y.tolist(),mr):+.3f} | "
          f"разность {m2:+.3f} [{l2:+.3f}, {h2:+.3f}] P={q2:.2f}")
