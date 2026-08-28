#!/usr/bin/env python3
"""Binary Bayes (UQ only): вера, построенная ТОЛЬКО на UQ, без tool-канала.

Отличие от нашего fusion: там приор — уже посчитанная bayes_state (вера по
критикам), и UQ её лишь домножает. Здесь приор — базовая частота успеха из
обучающих фолдов, а вся вера набирается из последовательности UQ-значений по
генерациям. То есть это байесовская агрегация UQ, а не слияние двух каналов.
"""

import _compat  # noqa: F401  # регистрирует code_uq.* / trajectory_uq_toolkit.*
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "askutsakov" / "src"))
sys.path.insert(0, str(HERE))
from trajectory_uq_toolkit.bayes import (  # noqa: E402
    BinaryBayes, ContinuousBayes, DoubleBinaryBayes)
from code_uq.analysis.analyze_lcb_llm_tool_agent_logs import prr  # noqa: E402
from bayes_cross_fit import cached_steps, episode_series, read_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--folds", type=int, default=4)
ap.add_argument("--seeds", type=int, default=5)
args = ap.parse_args()

RUNS = [("lcb_hard", Path("/Users/victor/Downloads/gpt_oss_20b_local-3")),
        ("lcb_medium", Path("/Users/victor/Downloads/gpt_oss_20b_local-5")),
        ("codecontests", Path("/Users/victor/Downloads/gpt_oss_20b_local-4"))]
METRICS = ["sum", "answer_sum", "ntok", "answer_mean", "mean", "min"]
MODES = ["quantile", "sep", "lr_pos", "lr_neg", "double", "continuous", "tempered"]


def fit_model(seqs, labs, mode):
    if mode == "double":
        return DoubleBinaryBayes.fit(seqs, labs, higher_is_uncertain=False)
    if mode == "continuous":
        return ContinuousBayes.fit(seqs, labs, lambda_=1.0)
    if mode == "tempered":
        return ContinuousBayes.fit(seqs, labs, lambda_=0.25)
    return BinaryBayes.fit(seqs, labs, mode=mode, higher_is_uncertain=False)


cells = {}
for bench, root in RUNS:
    stem = f"{bench}__gpt_oss_20b_local"
    eps = read_jsonl(root / f"{stem}.jsonl")
    steps = cached_steps(root, stem, HERE / "cache")
    ser = [episode_series(e, steps) for e in eps]
    y = np.array([int(bool(e["fixed"])) for e in eps])
    n = len(y)
    for metric in METRICS:
        seqs_all = [s.get(metric) or [] for s in ser]
        if np.mean([bool(s) for s in seqs_all]) < 0.8:
            continue
        for mode in MODES:
            per_seed = []
            for seed in range(args.seeds):
                order = np.random.RandomState(seed).permutation(n)
                parts = [order[i::args.folds] for i in range(args.folds)]
                out = np.full(n, np.nan)
                for k, held in enumerate(parts):
                    train = np.concatenate([p for j, p in enumerate(parts) if j != k])
                    pairs = [(seqs_all[int(i)], int(y[int(i)])) for i in train
                             if seqs_all[int(i)]]
                    if len({l for _, l in pairs}) < 2:
                        continue
                    try:
                        model = fit_model([s for s, _ in pairs], [l for _, l in pairs], mode)
                    except (ValueError, TypeError) as exc:
                        print(f"[skip] {metric}/{mode}: {exc}", file=sys.stderr)
                        break
                    for i in held:
                        s = seqs_all[int(i)]
                        if s:
                            out[int(i)] = model.predict(s)      # start=None => приор = базовая частота
                ok = [i for i in range(n) if np.isfinite(out[i])]
                if len(ok) > 10:
                    per_seed.append((prr([float(out[i]) for i in ok], [int(y[i]) for i in ok], 0.5),
                                     prr([float(out[i]) for i in ok], [int(y[i]) for i in ok], 1.0)))
            if per_seed:
                cells[(metric, mode, bench)] = (float(np.mean([p[0] for p in per_seed])),
                                                float(np.mean([p[1] for p in per_seed])))

print("Binary Bayes (UQ only) — приор = базовая частота, вера набирается из UQ\n")
print("| признак | режим | " + " | ".join(b for b, _ in RUNS) + " |")
print("|" + "---|" * (2 + len(RUNS)))
for metric in METRICS:
    for mode in MODES:
        row = [cells.get((metric, mode, b)) for b, _ in RUNS]
        if all(v is None for v in row):
            continue
        cs = ["—" if v is None else f"{v[0]:+.3f} / {v[1]:+.3f}" for v in row]
        print(f"| `{metric}` | {mode} | " + " | ".join(cs) + " |")
