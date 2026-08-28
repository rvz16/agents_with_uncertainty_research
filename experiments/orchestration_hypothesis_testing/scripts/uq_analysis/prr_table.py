#!/usr/bin/env python3
"""Единая таблица PRR по трём чистым прогонам. Всё считается одним кодом,
из кэша разобранных логпробов, чтобы числа были сопоставимы построчно."""

import _compat  # noqa: F401  # регистрирует code_uq.* / trajectory_uq_toolkit.*
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "askutsakov" / "src"))
sys.path.insert(0, str(HERE))
from code_uq.analysis.analyze_lcb_llm_tool_agent_logs import prr  # noqa: E402
from bayes_cross_fit import cached_steps, episode_series, read_jsonl  # noqa: E402

RUNS = [("lcb_hard", Path("/Users/victor/Downloads/gpt_oss_20b_local-3")),
        ("lcb_medium", Path("/Users/victor/Downloads/gpt_oss_20b_local-5")),
        ("codecontests", Path("/Users/victor/Downloads/gpt_oss_20b_local-4"))]

CSV_ROWS = [("bayes_state", "bayes_state"),
            ("bayes_state_after_generation", "bayes_state_after_generation"),
            ("tool_success", "tool_success"),
            ("verbalized_2s", "verbalized_2s_confidence"),
            ("llm_perplexity", "llm_perplexity")]
AGGS = {"first": lambda v: v[0], "last": lambda v: v[-1],
        "mean": lambda v: float(np.mean(v)), "min": min, "max": max,
        "sum": lambda v: float(np.sum(v))}
SERIES_ROWS = ([("sum", a) for a in ("last", "mean", "max", "min", "first")]
               + [("answer_sum", a) for a in ("last", "mean", "max", "min")]
               + [("ntok", a) for a in ("last", "sum", "mean", "max", "min")]
               + [("perplexity", a) for a in ("last", "mean", "max")]
               + [("answer_perplexity", a) for a in ("last", "mean", "max")]
               + [("answer_mean", "mean"), ("mean", "mean")])

cells, meta = {}, {}
for bench, root in RUNS:
    stem = f"{bench}__gpt_oss_20b_local"
    rd = root / "readable" / bench
    rows = {str(r["instance_id"]): r for r in
            csv.DictReader((rd / "final_logprob_bayes_quality.csv").open())}
    eps = read_jsonl(root / f"{stem}.jsonl")
    steps = cached_steps(root, stem, HERE / "cache")
    ser = {str(e["instance_id"]): episode_series(e, steps) for e in eps}
    for sr in ser.values():                      # perplexity = exp(-mean logprob), берём со знаком
        for src, dst in (("mean", "perplexity"), ("answer_mean", "answer_perplexity")):
            if sr.get(src):
                sr[dst] = [-float(np.exp(-v)) for v in sr[src]]

    def q(i):
        try:
            return int(float(rows[i]["quality"]))
        except (TypeError, ValueError, KeyError):
            return None

    insts = [i for i in rows if q(i) is not None]
    y = [q(i) for i in insts]
    meta[bench] = (len(y), len(y) - sum(y), sum(y) / len(y))

    def put(label, vals):
        ok = [(v, l) for v, l in zip(vals, y) if v is not None and np.isfinite(v)]
        if len(ok) < 10:
            return
        c, l = [a for a, _ in ok], [b for _, b in ok]
        cells[(label, bench)] = (prr(c, l, 0.5), prr(c, l, 1.0))

    def num(r, k):
        try:
            v = float(r[k])
            return v if np.isfinite(v) else None
        except (TypeError, ValueError, KeyError):
            return None

    for label, key in CSV_ROWS:
        put(label, [num(rows[i], key) for i in insts])
    for metric, aggname in SERIES_ROWS:
        put(f"{metric}:{aggname}",
            [AGGS[aggname](ser[i][metric]) if ser.get(i, {}).get(metric) else None
             for i in insts])

order = [l for l, _ in CSV_ROWS] + [f"{m}:{a}" for m, a in SERIES_ROWS]

def head_cell(bench):
    n, err, _ = meta[bench]
    word = "задача" if n % 10 == 1 and n % 100 != 11 else (
        "задачи" if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14) else "задач")
    return f"{bench}<br>{n} {word}, {err} не решено"

print("В ячейках: PRR@0.5 / PRR@1.0\n")
print("| сигнал | " + " | ".join(head_cell(b) for b, _ in RUNS) + " |")
print("|" + "---|" * (1 + len(RUNS)))
for label in order:
    cs = []
    for bench, _ in RUNS:
        v = cells.get((label, bench))
        cs.append("—" if v is None else f"{v[0]:+.3f} / {v[1]:+.3f}")
    print(f"| `{label}` | " + " | ".join(cs) + " |")
