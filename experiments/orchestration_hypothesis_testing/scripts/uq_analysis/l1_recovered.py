#!/usr/bin/env python3
"""Вера с восстановленным критиком L1 — без перегенерации.

`critic_L1_lint` в репозитории падает (subprocess не импортирован, переменная
`path` не существует, `result.timed_out` у CompletedProcess отсутствует), поэтому
во всех прогонах L1 возвращал None: 504 / 980 / 736 вердиктов выброшено, и канал
в веру не входил. Линтер — локальная статическая проверка, так что его можно
пересчитать на сохранённых кандидатах и доиграть веру заново.
"""

import _compat  # noqa: F401  # регистрирует code_uq.* / trajectory_uq_toolkit.*
import json, pickle, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "askutsakov" / "src")); sys.path.insert(0, str(HERE))
from code_uq.analysis.analyze_lcb_llm_tool_agent_logs import prr
from lint_fixed import lint_ok

CAL = {"p_fix_broken": 0.11, "p_break_correct": 0.05}
OLD = {"p_fix_broken": 0.50, "p_break_correct": 0.05}
MAP = {"critic_L0": "L0_syntax", "critic_L2": "L2_public_tests", "critic_L1": "L1_lint"}


def bupd(b, th, passed):
    p = th["p1"] if passed else 1 - th["p1"]
    q = th["p0"] if passed else 1 - th["p0"]
    num = b * p; den = num + (1 - b) * q
    return num / den if den > 0 else b


def fit_l1(train_rows):
    n1 = n0 = c1 = c0 = 0
    for r in train_rows:
        y = r.get("passed")
        if y is None:
            continue
        v = lint_ok(r.get("code") or "")
        if v is None:
            continue
        if y: n1 += 1; c1 += int(v)
        else: n0 += 1; c0 += int(v)
    return {"p1": (c1 + 1) / (n1 + 2), "p0": (c0 + 1) / (n0 + 2), "n1": n1, "n0": n0}


for bench, d in (("lcb_hard","gpt_oss_20b_local-3"),("lcb_medium","gpt_oss_20b_local-5"),
                 ("codecontests","gpt_oss_20b_local-4")):
    root = Path(f"/Users/victor/Downloads/{d}"); stem = f"{bench}__gpt_oss_20b_local"
    summ = json.load(open(root/"readable"/bench/"analysis_summary.json"))
    prior = summ["prior"]["prior_Y1"]
    theta = {k: {"p1": v["p_pass_y1"], "p0": v["p_pass_y0"]} for k, v in summ["theta"].items()}
    train = [json.loads(l) for l in open(root/f"{stem}.train_prior_calibration.jsonl") if l.strip()]
    theta["L1_lint"] = fit_l1(train)
    codes = pickle.load(open(HERE/"cache"/f"codes__{d}__{bench}.pkl","rb"))

    l1_cache = {}
    def l1_of(iid, gen):
        key = (iid, gen)
        if key not in l1_cache:
            c = codes.get(key)
            l1_cache[key] = None if c is None else lint_ok(c)
        return l1_cache[key]

    acts = {}
    for line in open(root/f"{stem}.actions.jsonl"):
        if line.strip():
            r = json.loads(line)
            if r.get("split") == "test":
                acts.setdefault(str(r["instance_id"]), []).append(r)

    def replay(iid, rows, kern, use_l1):
        b = float(prior); gen = -1
        for row in sorted(rows, key=lambda r: int(r.get("step", 0))):
            a = str(row.get("action"))
            if a == "generate":
                if row.get("skipped") is True: continue
                gen += 1
                b = b*(1-kern["p_break_correct"]) + (1-b)*kern["p_fix_broken"]
                continue
            name = MAP.get(a)
            if name == "L1_lint":
                if not use_l1: continue
                v = l1_of(iid, max(gen, 0))
                if v is not None and theta["L1_lint"]["n1"] and theta["L1_lint"]["n0"]:
                    b = bupd(b, theta["L1_lint"], bool(v))
                continue
            if name and row.get("passed") in (True, False) and name in theta:
                b = bupd(b, theta[name], bool(row["passed"])); continue
            if a in {"verify","final_verify"} and row.get("passed") in (True, False):
                return b
        return b

    eps = [json.loads(l) for l in open(root/f"{stem}.jsonl") if l.strip()]
    ids = [str(e["instance_id"]) for e in eps if str(e["instance_id"]) in acts]
    y = [int(bool(e["fixed"])) for e in eps if str(e["instance_id"]) in acts]
    t1 = theta["L1_lint"]
    print(f"\n### {bench} (n={len(y)}, не решено {len(y)-sum(y)})")
    print(f"L1 обучен на train: P(чисто|Y=1)={t1['p1']:.3f}, P(чисто|Y=0)={t1['p0']:.3f} "
          f"(n1={t1['n1']}, n0={t1['n0']})")
    print(f"{'вариант веры':<44}{'PRR@0.5 / PRR@1.0':>20}")
    for kname, kern in (("плейсхолдер kernel", OLD), ("калиброванный kernel", CAL)):
        for use_l1, tag in ((False, "L0+L2 (как сейчас)"), (True, "L0+L1+L2 (L1 восстановлен)")):
            v = [replay(i, acts[i], kern, use_l1) for i in ids]
            print(f"  {kname}, {tag:<26}{prr(v,y,0.5):+.3f} / {prr(v,y,1.0):+.3f}")
