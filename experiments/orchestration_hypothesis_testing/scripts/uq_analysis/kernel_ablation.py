#!/usr/bin/env python3
"""Чего стоит некалиброванный transition kernel.

В analyze_lcb_llm_tool_agent_logs вера на КАЖДОЙ генерации прогоняется через
kernel_update: b' = b*(1-p_break) + (1-b)*p_fix. Константы (0.50 / 0.05)
помечены в исходниках как "initial uninformative" и ни в одном прогоне не
калибровались (--kernel никто не передавал; kernel_source во всех наших
артефактах = hardcoded_uninformative_default).

Здесь вера переигрывается из actions.jsonl с разными kernel и сравнивается PRR.
Неподвижная точка дефолтного kernel = p_fix/(p_fix+p_break) = 0.909: он тянет
любую веру к 0.91 на каждой генерации, независимо от свидетельств.
"""

import _compat  # noqa: F401  # регистрирует code_uq.* / trajectory_uq_toolkit.*
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "askutsakov" / "src"))
from code_uq.analysis.analyze_lcb_llm_tool_agent_logs import prr  # noqa: E402

CRITIC_MAP = {"critic_L0": "L0_syntax", "critic_L2": "L2_public_tests"}


def bayes_update(b, th, passed):
    p = th["p_pass_y1"] if passed else 1.0 - th["p_pass_y1"]
    q = th["p_pass_y0"] if passed else 1.0 - th["p_pass_y0"]
    num = b * p
    den = num + (1.0 - b) * q
    return num / den if den > 0 else b


def replay(actions, prior, theta, kernel):
    b = float(prior)
    for row in sorted(actions, key=lambda r: int(r.get("step", 0))):
        act = str(row.get("action"))
        if act == "generate":
            if row.get("skipped") is True:
                continue
            if kernel is not None:
                b = b * (1.0 - kernel["p_break_correct"]) + (1.0 - b) * kernel["p_fix_broken"]
            continue
        cr = CRITIC_MAP.get(act)
        if cr and row.get("passed") in (True, False) and cr in theta:
            b = bayes_update(b, theta[cr], bool(row["passed"]))
            continue
        if act in {"verify", "final_verify"} and row.get("passed") in (True, False):
            return b          # значение ДО финальной метки, как в анализаторе
    return b


KERNELS = {
    "дефолтный 0.50/0.05 (в наших прогонах)": {"p_fix_broken": 0.50, "p_break_correct": 0.05},
    "измеренный p_fix=0.11": {"p_fix_broken": 0.11, "p_break_correct": 0.05},
    "БЕЗ kernel": None,
}

for bench, d in (("lcb_hard", "gpt_oss_20b_local-3"), ("lcb_medium", "gpt_oss_20b_local-5"),
                 ("codecontests", "gpt_oss_20b_local-4")):
    root = Path(f"/Users/victor/Downloads/{d}")
    stem = f"{bench}__gpt_oss_20b_local"
    summ = json.load(open(root / "readable" / bench / "analysis_summary.json"))
    prior, theta = summ["prior"]["prior_Y1"], summ["theta"]
    acts = {}
    for line in open(root / f"{stem}.actions.jsonl"):
        if line.strip():
            r = json.loads(line)
            acts.setdefault(str(r["instance_id"]), []).append(r)
    y, per = [], {k: [] for k in KERNELS}
    for line in open(root / f"{stem}.jsonl"):
        if not line.strip():
            continue
        ep = json.loads(line)
        i = str(ep["instance_id"])
        if i not in acts:
            continue
        y.append(int(bool(ep["fixed"])))
        for name, kern in KERNELS.items():
            per[name].append(replay(acts[i], prior, theta, kern))
    print(f"\n{bench} (n={len(y)}, не решено {len(y)-sum(y)}):")
    for name in KERNELS:
        v = per[name]
        print(f"  {name:<40} PRR@0.5 {prr(v, y, 0.5):+.3f} / @1.0 {prr(v, y, 1.0):+.3f}   "
              f"медиана веры: решено {np.median([x for x,q in zip(v,y) if q]):.3f}, "
              f"провал {np.median([x for x,q in zip(v,y) if not q]):.3f}")
