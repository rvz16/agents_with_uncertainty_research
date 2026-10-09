"""Is the confidence head better than a plain uncertainty score, given the same post-hoc fit?

Every estimator gets the same treatment: one monotone mapping to probability
fitted out of fold (a temperature for something already a probability, a
one-feature logistic for a raw signal), then the same calibration measures.
"""
import json, math, statistics as st, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
from experiments.decision_model_score import apply_temperature, ece, brier, extract, fit_temperature
from experiments.prr_report_v2 import SEEDS, SEGMENT, prr, stratified_folds

R = Path("/Users/victor/Documents/vs_files/research/article_implementation/agents_with_uncertainty_research/alfworld_uq/runs")
S = Path("/private/tmp/claude-501/-Users-victor-Documents-vs-files-research/7fd0e6ed-037d-4010-a39a-26dcaace1004/scratchpad/jeeves")
RUNS = {"RG": "react_gptoss_140_giveup_sc_finished_nolast", "SG": "smol_gptoss_140_sc_finished_nolast",
        "RQ": "react_qwen_140_giveup_sc_finished_nolast", "SQ": "smol_qwen_140_sc_finished_nolast"}
SIGNALS = (("MTE", "mean_token_entropy", True), ("Self-certainty", "self_certainty", False),
           ("Logprob", "sum_logprob", False), ("Perplexity", "perplexity", True))


def platt_oof(ids, labels, values):
    from sklearn.linear_model import LogisticRegression
    episodes = {i: {"label": int(labels[k]), "score": labels[k], "id": i} for k, i in enumerate(ids)}
    index = {i: k for k, i in enumerate(ids)}
    x = np.asarray(values, dtype=float).reshape(-1, 1)
    y = np.asarray(labels, dtype=float)
    per_seed = []
    for seed in SEEDS:
        out = [0.0] * len(ids)
        for fold in stratified_folds(episodes, seed):
            held = set(fold); train = [index[i] for i in episodes if i not in held]
            test = [index[i] for i in fold]
            if len(set(y[train])) < 2:
                for i in test: out[i] = float(y[train].mean())
                continue
            m = LogisticRegression(C=1e6, max_iter=2000).fit(x[train], y[train])
            for i, p in zip(test, m.predict_proba(x[test])[:, 1]): out[i] = float(p)
        per_seed.append(out)
    return [st.fmean(v) for v in zip(*per_seed)]


def temp_oof(ids, labels, probs):
    episodes = {i: {"label": int(labels[k]), "score": labels[k], "id": i} for k, i in enumerate(ids)}
    index = {i: k for k, i in enumerate(ids)}
    per_seed = []
    for seed in SEEDS:
        out = [0.0] * len(ids)
        for fold in stratified_folds(episodes, seed):
            held = set(fold); train = [index[i] for i in episodes if i not in held]
            t = fit_temperature([labels[i] for i in train], [probs[i] for i in train])
            for i in (index[j] for j in fold): out[i] = apply_temperature([probs[i]], t)[0]
        per_seed.append(out)
    return [st.fmean(v) for v in zip(*per_seed)]


def report(name, labels, probs, scores):
    a = np.asarray(probs); y = np.asarray(labels, dtype=float)
    hi = a >= 0.9
    at9 = float(y[hi].mean()) if hi.sum() else float("nan")
    at8 = float(y[a >= 0.8].mean()) if (a >= 0.8).sum() else float("nan")
    return (f"{name:28} PRR {prr(scores, probs):+5.2f}  ECE {ece(labels, probs):.3f}  "
            f"Brier {brier(labels, probs):.3f}  p>=.8 -> {at8:5.2f} (n={int((a>=0.8).sum()):3d})  "
            f"p>=.9 -> {at9:5.2f} (n={int(hi.sum()):3d})")


for coh, d in RUNS.items():
    eps = {json.loads(l)["episode_id"]: float(bool(json.loads(l)["final_success"])) for l in open(R / d / "episodes.jsonl")}
    steps = defaultdict(list)
    for l in open(R / d / "trajectories.jsonl"):
        if l.strip():
            r = json.loads(l); steps[r["episode_id"]].append(r)
    ids, labels, raw = [], [], defaultdict(list)
    for e, yy in eps.items():
        rr = sorted(steps.get(e) or [], key=lambda r: int(r.get("step", 0)))
        if not rr: continue
        vals = {}
        for name, key, hiu in SIGNALS:
            v = [((x.get("uq") or {}).get(SEGMENT) or {}).get(key) for x in rr]
            v = [float(z) for z in v if z is not None and math.isfinite(float(z))]
            if not v: break
            vals[name] = (-v[-1] if hiu else v[-1])
        if len(vals) < len(SIGNALS): continue
        ids.append(e); labels.append(yy)
        for k, v in vals.items(): raw[k].append(v)
    print(f"\n=== {coh}: n={len(ids)}, base {st.fmean(labels):.2f}")
    for name, _, _ in SIGNALS:
        p = platt_oof(ids, labels, raw[name])
        print("  " + report(f"{name} + Platt (OOF)", labels, p, labels))
    # the decision head on the same episodes
    for tag, f in (("Jeeves", "clean_answers_nothink.jsonl"), ("clef-flash", "answers_Cloudflare_clef-flash.jsonl"),
                   ("clef-27B", "answers_Cloudflare_clef.jsonl")):
        rows = {r["id"]: r for r in (json.loads(l) for l in open(S / f)) if r["cohort"] == coh}
        sub = [(i, labels[k]) for k, i in enumerate(ids) if i in rows]
        if not sub: continue
        sids = [i for i, _ in sub]; sy = [y for _, y in sub]
        probs = [extract(rows[i]["answer"], "success") for i in sids]
        print("  " + report(f"{tag} head, raw", sy, probs, sy))
        print("  " + report(f"{tag} head + temperature", sy, temp_oof(sids, sy, probs), sy))
