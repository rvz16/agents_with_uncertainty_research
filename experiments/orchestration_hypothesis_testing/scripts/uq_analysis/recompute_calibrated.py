#!/usr/bin/env python3
"""Пересчёт строк таблицы, зависящих от transition kernel.

Меняются только они: bayes_state, belief-after-generation и наш fused, который
стартует от bayes_state. Логпробные сигналы, verbalized, tool_success,
belief_logit (у него kernel нет) и Binary Bayes UQ-only не затронуты.
Прокси калиброванного kernel — p_fix=0.11, значение, которое сам анализатор
называет измеренным на прогонах проекта.
"""

import _compat  # noqa: F401  # регистрирует code_uq.* / trajectory_uq_toolkit.*
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "askutsakov" / "src")); sys.path.insert(0, str(HERE))
from code_uq.analysis.analyze_lcb_llm_tool_agent_logs import prr
from code_uq.analysis.experiment2_uq_bayes_critic import (
    bayes_update as fuse_update, continuous_bayes_update, fit_class_gaussians,
    fit_threshold_theta)
from bayes_cross_fit import cached_steps, episode_series, read_jsonl

CRITIC_MAP = {"critic_L0": "L0_syntax", "critic_L2": "L2_public_tests"}
CAL = {"p_fix_broken": 0.11, "p_break_correct": 0.05}
OLD = {"p_fix_broken": 0.50, "p_break_correct": 0.05}


def bupd(b, th, passed):
    p = th["p_pass_y1"] if passed else 1 - th["p_pass_y1"]
    q = th["p_pass_y0"] if passed else 1 - th["p_pass_y0"]
    num = b * p; den = num + (1 - b) * q
    return num / den if den > 0 else b


def replay2(actions, prior, theta, kern):
    """Возвращает (bayes_state, belief_after_generation) — как в simulate_instance."""
    b = float(prior); after_gen = None
    for row in sorted(actions, key=lambda r: int(r.get("step", 0))):
        act = str(row.get("action"))
        if act == "generate":
            if row.get("skipped") is True:
                continue
            b = b * (1 - kern["p_break_correct"]) + (1 - b) * kern["p_fix_broken"]
            after_gen = b
            continue
        cr = CRITIC_MAP.get(act)
        if cr and row.get("passed") in (True, False) and cr in theta:
            b = bupd(b, theta[cr], bool(row["passed"])); continue
        if act in {"verify", "final_verify"} and row.get("passed") in (True, False):
            return b, (after_gen if after_gen is not None else b)
    return b, (after_gen if after_gen is not None else b)


AGGS = {"first": lambda v: v[0], "last": lambda v: v[-1], "max": max, "min": min,
        "mean": lambda v: float(np.mean(v)), "sum": lambda v: float(np.sum(v))}
METRICS = ["sum", "answer_sum", "ntok", "answer_mean", "mean", "min"]
MODES = ("sep", "lr_pos", "lr_neg", "double", "continuous")


def fuse(prior_col, x, y, idx_tr, idx_ap, mode):
    tr = [i for i in idx_tr if np.isfinite(x[i])]
    if len({int(y[i]) for i in tr}) < 2:
        return None
    out = {}
    if mode == "continuous":
        par = fit_class_gaussians([float(x[i]) for i in tr], [int(y[i]) for i in tr])
        for i in idx_ap:
            out[i] = (continuous_bayes_update(float(prior_col[i]), float(x[i]), par, lambda_=1.0)
                      if np.isfinite(x[i]) else float(prior_col[i]))
    else:
        ms = ["lr_pos", "lr_neg"] if mode == "double" else [mode]
        fitted = [fit_threshold_theta([float(x[i]) for i in tr], [int(y[i]) for i in tr], True, mode=m)
                  for m in ms]
        for i in idx_ap:
            b = float(prior_col[i])
            if np.isfinite(x[i]):
                for thr, p1, p0 in fitted:
                    b = fuse_update(b, p1, p0, x[i] >= thr)
            out[i] = b
    return out


def nested(prior_col, feats, y, outer=5, inner=5, seeds=5):
    n = len(y); acc = np.full((seeds, n), np.nan)
    for seed in range(seeds):
        order = np.random.RandomState(seed).permutation(n)
        parts = [order[i::outer] for i in range(outer)]
        iparts = [np.random.RandomState(seed + 100).permutation(n)[i::inner] for i in range(inner)]
        for k, held in enumerate(parts):
            train = np.concatenate([p for j, p in enumerate(parts) if j != k]); tset = set(train.tolist())
            best, bv = None, -np.inf
            for key, x in feats.items():
                for mode in MODES:
                    inner_pred = np.full(n, np.nan)
                    for ih in iparts:
                        ia = np.array([i for i in ih if i in tset])
                        it = np.array([i for i in train if i not in set(ia.tolist())])
                        if len(ia) == 0 or len(it) < 10:
                            continue
                        got = fuse(prior_col, x, y, it, ia, mode)
                        if got:
                            for i, v in got.items():
                                inner_pred[i] = v
                    ok = [i for i in train if np.isfinite(inner_pred[i])]
                    if len(ok) < 10 or len({int(y[i]) for i in ok}) < 2:
                        continue
                    v = prr([float(inner_pred[i]) for i in ok], [int(y[i]) for i in ok], 0.5)
                    if v is not None and v > bv:
                        bv, best = v, (key, mode)
            if best:
                got = fuse(prior_col, feats[best[0]], y, train, held, best[1])
                if got:
                    for i, v in got.items():
                        acc[seed, i] = v
    return np.nanmean(acc, axis=0)


for bench, d in (("lcb_hard","gpt_oss_20b_local-3"),("lcb_medium","gpt_oss_20b_local-5"),
                 ("codecontests","gpt_oss_20b_local-4")):
    root = Path(f"/Users/victor/Downloads/{d}"); stem = f"{bench}__gpt_oss_20b_local"
    summ = json.load(open(root/"readable"/bench/"analysis_summary.json"))
    prior, theta = summ["prior"]["prior_Y1"], summ["theta"]
    acts = {}
    for line in open(root/f"{stem}.actions.jsonl"):
        if line.strip():
            r = json.loads(line); acts.setdefault(str(r["instance_id"]), []).append(r)
    eps = read_jsonl(root/f"{stem}.jsonl"); steps = cached_steps(root, stem, HERE/"cache")
    ids, y, ser = [], [], []
    for ep in eps:
        i = str(ep["instance_id"])
        if i in acts:
            ids.append(i); y.append(int(bool(ep["fixed"]))); ser.append(episode_series(ep, steps))
    y = np.array(y)
    feats = {}
    for m in METRICS:
        for ag, fn in AGGS.items():
            c = np.array([fn(s[m]) if s.get(m) else np.nan for s in ser], dtype=float)
            if np.isfinite(c).mean() > 0.8:
                feats[(m, ag)] = c
    print(f"\n### {bench} (n={len(y)}, не решено {int((1-y).sum())})")
    print(f"{'строка':<40}{'плейсхолдер':>22}{'калиброванный':>22}")
    for kname, kern in (("old", OLD), ("cal", CAL)):
        pairs = [replay2(acts[i], prior, theta, kern) for i in ids]
        bs = np.array([p[0] for p in pairs]); ag = np.array([p[1] for p in pairs])
        fu = nested(bs, feats, y)
        if kname == "old":
            store = (bs, ag, fu)
        else:
            for label, o, c in (("Original (tool only), bayes_state", store[0], bs),
                                ("Belief after generation", store[1], ag),
                                ("Bayes + UQ (fused, nested)", store[2], fu)):
                so = f"{prr(o.tolist(),y.tolist(),0.5):+.3f} / {prr(o.tolist(),y.tolist(),1.0):+.3f}"
                sc = f"{prr(c.tolist(),y.tolist(),0.5):+.3f} / {prr(c.tolist(),y.tolist(),1.0):+.3f}"
                print(f"{label:<40}{so:>22}{sc:>22}")
