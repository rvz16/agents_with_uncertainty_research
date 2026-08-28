#!/usr/bin/env python3
"""The aggregation claim on the July codecontests export (124 episodes, 74 errors).

Only two tables survive in that export -- the per-instance CSV and the
per-generation scores -- so the critic verdicts needed by belief_logit are gone.
What can be recomputed is exactly the honest part: per-generation logprob under
last/mean/min/first, plus the per-instance signals for reference. Whether that
run carried the intermediate-verify leak is read off bayes_state itself.
"""

import _compat  # noqa: F401  # регистрирует code_uq.* / trajectory_uq_toolkit.*
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "askutsakov" / "src"))
from code_uq.analysis.analyze_lcb_llm_tool_agent_logs import prr  # noqa: E402

root = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/Users/victor/Downloads/sage_uncertainty_export/gpt_oss_20b/codecontests")

rows = list(csv.DictReader((root / "final_logprob_bayes_quality.csv").open()))
label = {}
for r in rows:
    try:
        label[str(r["instance_id"])] = int(float(r["quality"]))
    except (TypeError, ValueError):
        pass


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

agg = {}
for inst, pairs in series.items():
    vals = [v for _, v in sorted(pairs)]
    agg[inst] = {"seqprob:last": vals[-1], "seqprob:first": vals[0],
                 "seqprob:mean": float(np.mean(vals)), "seqprob:min": float(min(vals)),
                 "n_generations": -float(len(vals))}

insts = [i for i in label if i in agg]
y = [label[i] for i in insts]
print(f"{root.parent.name}/{root.name}: {len(insts)} episodes, pass@1 "
      f"{np.mean(y):.3f}, errors {len(y) - sum(y)}\n")
print(f"{'signal':<34} {'PRR@0.5 / PRR@1.0':>18}")
print("-" * 54)

per_instance = {str(r["instance_id"]): r for r in rows}
for name in ("bayes_state", "bayes_state_after_generation", "tool_success",
             "verbalized_2s_confidence", "llm_log_seq_prob"):
    conf, lab = [], []
    for i in insts:
        v = num(per_instance[i], name)
        if v is not None:
            conf.append(v)
            lab.append(label[i])
    if len(conf) > 2:
        print(f"{name:<34} {prr(conf, lab, 0.5):+.3f} / {prr(conf, lab, 1.0):+.3f}")

print("-" * 54)
for name in ("seqprob:last", "seqprob:first", "seqprob:mean", "seqprob:min",
             "n_generations"):
    conf = [agg[i][name] for i in insts]
    print(f"{name:<34} {prr(conf, y, 0.5):+.3f} / {prr(conf, y, 1.0):+.3f}")
