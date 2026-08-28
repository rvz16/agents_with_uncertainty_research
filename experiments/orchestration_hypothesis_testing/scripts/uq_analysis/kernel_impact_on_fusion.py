#!/usr/bin/env python3
"""Насколько сдвинутся таблицы, если kernel откалибровать.

Настоящую калибровку ещё предстоит прогнать; здесь как прокси взято p_fix=0.11 —
значение, которое сам анализатор называет измеренным на прогонах этого проекта.
Пересчитываются только строки, зависящие от kernel: bayes_state и наши fusion-
варианты, которые стартуют от него. Логпробные сигналы, verbalized, tool_success,
belief_logit и Binary Bayes (UQ only) от kernel не зависят вообще.
"""

import _compat  # noqa: F401  # регистрирует code_uq.* / trajectory_uq_toolkit.*
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "askutsakov" / "src"))
sys.path.insert(0, str(HERE))
from code_uq.analysis.analyze_lcb_llm_tool_agent_logs import prr  # noqa: E402
from code_uq.analysis.experiment2_uq_bayes_critic import (  # noqa: E402
    bayes_update as fuse_update, continuous_bayes_update, fit_class_gaussians,
    fit_threshold_theta)
from bayes_cross_fit import cached_steps, episode_series, read_jsonl  # noqa: E402
from kernel_ablation import CRITIC_MAP, bayes_update, replay  # noqa: E402

KERNELS = {
    "плейсхолдер 0.50/0.05": {"p_fix_broken": 0.50, "p_break_correct": 0.05},
    "калиброванный ~0.11": {"p_fix_broken": 0.11, "p_break_correct": 0.05},
}
RUNS = [("lcb_hard", "gpt_oss_20b_local-3"), ("lcb_medium", "gpt_oss_20b_local-5"),
        ("codecontests", "gpt_oss_20b_local-4")]


def fused(prior_col, x, y, mode, folds=5, seeds=5):
    n = len(y)
    acc = np.full((seeds, n), np.nan)
    for seed in range(seeds):
        order = np.random.RandomState(seed).permutation(n)
        parts = [order[i::folds] for i in range(folds)]
        for k, held in enumerate(parts):
            tr = [i for i in np.concatenate([p for j, p in enumerate(parts) if j != k])
                  if np.isfinite(x[i])]
            if len({int(y[i]) for i in tr}) < 2:
                continue
            if mode == "continuous":
                par = fit_class_gaussians([float(x[i]) for i in tr], [int(y[i]) for i in tr])
                for i in held:
                    acc[seed, i] = (continuous_bayes_update(float(prior_col[i]), float(x[i]), par,
                                                            lambda_=1.0)
                                    if np.isfinite(x[i]) else prior_col[i])
            else:
                thr, p1, p0 = fit_threshold_theta([float(x[i]) for i in tr],
                                                  [int(y[i]) for i in tr], True, mode=mode)
                for i in held:
                    acc[seed, i] = (fuse_update(float(prior_col[i]), p1, p0, x[i] >= thr)
                                    if np.isfinite(x[i]) else prior_col[i])
    return np.nanmean(acc, axis=0)


print(f"{'бенчмарк':<14}{'строка':<34}{'плейсхолдер':>13}{'калиброванный':>15}{'сдвиг':>9}")
print("-" * 86)
for bench, d in RUNS:
    root = Path(f"/Users/victor/Downloads/{d}")
    stem = f"{bench}__gpt_oss_20b_local"
    summ = json.load(open(root / "readable" / bench / "analysis_summary.json"))
    prior, theta = summ["prior"]["prior_Y1"], summ["theta"]
    acts = {}
    for line in open(root / f"{stem}.actions.jsonl"):
        if line.strip():
            r = json.loads(line)
            acts.setdefault(str(r["instance_id"]), []).append(r)
    eps = read_jsonl(root / f"{stem}.jsonl")
    steps = cached_steps(root, stem, HERE / "cache")
    ids, y, ser = [], [], []
    for ep in eps:
        i = str(ep["instance_id"])
        if i in acts:
            ids.append(i); y.append(int(bool(ep["fixed"]))); ser.append(episode_series(ep, steps))
    y = np.array(y)
    feats = {
        "ntok:mean + continuous": (np.array([float(np.mean(s["ntok"])) if s.get("ntok") else np.nan
                                             for s in ser]), "continuous"),
        "ntok:sum + sep": (np.array([float(np.sum(s["ntok"])) if s.get("ntok") else np.nan
                                     for s in ser]), "sep"),
    }
    out = {}
    for kname, kern in KERNELS.items():
        b = np.array([replay(acts[i], prior, theta, kern) for i in ids])
        out[("bayes_state", kname)] = prr(b.tolist(), y.tolist(), 0.5)
        for fname, (x, mode) in feats.items():
            v = fused(b, x, y, mode)
            out[(fname, kname)] = prr(v.tolist(), y.tolist(), 0.5)
    for row in ("bayes_state", *feats):
        a = out[(row, "плейсхолдер 0.50/0.05")]
        c = out[(row, "калиброванный ~0.11")]
        print(f"{bench:<14}{row:<34}{a:>13.3f}{c:>15.3f}{c-a:>+9.3f}")
    print()
