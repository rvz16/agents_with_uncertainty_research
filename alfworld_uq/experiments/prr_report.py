"""PRR report for ALFWorld in the format of the OSWorld/WebArena report.

Sections 1-4 use 5-fold out-of-fold evaluation on each cohort (shuffle with
numpy RandomState(0), fold i = order[i::5], not stratified; every episode
predicted once by a model fitted on the other four folds; one PRR on the
pooled prediction vector). Section 7 averages OOD over five repeated
source-train -> target-test holdouts (seeds 0-4): all parameters fitted on a
random half of the source cohort (the calibration size the in-domain
protocol gets), scored on the whole target.

    python -m experiments.prr_report --out REPORT.md --toolkit agentic-uq/src \\
        --cohort "ReAct · gpt-oss" runs/react_gptoss_..._finished_nolast ...
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from belief.binary_bayes import BinaryBayesUQ, DoubleBinaryBayesUQ
from belief.continuous_bayes import ContinuousBayesUQ
from belief.critic_bayes import CriticBayesState
from experiments.analyze_trajectories import (
    STEP_CRITIC_NAMES,
    _prr_references,
    _step_critic_observation,
    prediction_rejection_area,
)

SEGMENT = "combined"
# name -> (stored signal, higher value means more uncertain)
SIGNALS = {
    "Logprob": ("sum_logprob", False),
    "Perplexity": ("perplexity", True),
    "MTE": ("mean_token_entropy", True),
    "Verb actions": ("verbalized_confidence", False),
}
FUSED_MODES = ("Quantile", "SEP", "LR+", "LR−", "Double", "Continuous (λ=1)", "Tempered (λ=0.25)", "Last only")
ID_MODES = ("SEP", "Double", "Continuous (λ=1)", "Tempered (λ=0.25)", "Last only")


# ----------------------------------------------------------------------------- data
def load(run: Path) -> dict[str, dict[str, Any]]:
    steps = defaultdict(list)
    for line in open(run / "trajectories.jsonl"):
        if line.strip():
            row = json.loads(line); steps[row["episode_id"]].append(row)
    episodes = {}
    for line in open(run / "episodes.jsonl"):
        if not line.strip():
            continue
        e = json.loads(line); rows = steps.get(e["episode_id"], [])
        if not rows:
            continue
        seqs = {}
        for name, (key, _) in SIGNALS.items():
            vals = [((r.get("uq") or {}).get(SEGMENT) or {}).get(key) for r in rows]
            seqs[name] = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
        crit = [{c: bool(_step_critic_observation(r)[c]) for c in STEP_CRITIC_NAMES} for r in rows]
        verbs = seqs["Verb actions"]
        episodes[e["episode_id"]] = {
            "label": int(bool(e["final_success"])), "signals": seqs, "critics": crit,
            "verb_final": verbs[-1] if verbs else None, "n_steps": len(rows),
            "tool_rate": st.fmean(v for c in crit for v in c.values()),
            "rows": rows,
        }
    return episodes


def prr(labels, conf) -> float | None:
    a = prediction_rejection_area([-c for c in conf], labels, 0.5)
    o, r = _prr_references(tuple(labels), 0.5)
    return None if None in (a, o, r) or abs(o - r) < 1e-9 else (a - r) / (o - r)


def folds(ids: list[str], k: int = 5, seed: int = 0) -> list[list[str]]:
    order = np.random.RandomState(seed).permutation(len(ids))
    return [[ids[j] for j in order[i::k]] for i in range(k)]


# ----------------------------------------------------------------------------- models
def fit_tools(train: list[dict]) -> CriticBayesState:
    obs = [c for e in train for c in e["critics"]]
    lab = [e["label"] for e in train for _ in e["critics"]]
    prior = sum(e["label"] for e in train) / len(train)
    return CriticBayesState.fit(obs, lab, prior=prior)


def fit_uq(train: list[dict], signal: str, mode: str):
    key, hiu = SIGNALS[signal][0], SIGNALS[signal][1]
    seqs = [e["signals"][signal] for e in train if e["signals"][signal]]
    labs = [e["label"] for e in train if e["signals"][signal]]
    if not seqs:
        return None
    if mode == "Quantile":
        return BinaryBayesUQ.fit(seqs, labs, threshold_mode="quantile", higher_is_uncertain=hiu)
    if mode == "SEP":
        return BinaryBayesUQ.fit(seqs, labs, threshold_mode="sep", higher_is_uncertain=hiu)
    if mode == "LR+":
        return BinaryBayesUQ.fit(seqs, labs, threshold_mode="lr_pos", higher_is_uncertain=hiu)
    if mode == "LR−":
        return BinaryBayesUQ.fit(seqs, labs, threshold_mode="lr_neg", higher_is_uncertain=hiu)
    if mode == "Double":
        return DoubleBinaryBayesUQ.fit(seqs, labs, higher_is_uncertain=hiu)
    if mode == "Continuous (λ=1)":
        return ContinuousBayesUQ.fit(seqs, labs, lambda_=1.0)
    if mode == "Tempered (λ=0.25)":
        return ContinuousBayesUQ.fit(seqs, labs, lambda_=0.25)
    if mode == "Last only":
        return ContinuousBayesUQ.fit([[s[-1]] for s in seqs], labs, lambda_=1.0)
    raise ValueError(mode)


def apply_uq(model, seq: list[float], belief: float, mode: str) -> float:
    if not seq:
        return belief
    values = [seq[-1]] if mode == "Last only" else seq
    for v in values:
        belief = model.update(belief, v)
    return belief


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6); return math.log(p / (1 - p))


# ----------------------------------------------------------------------------- in-domain
def oof(episodes: dict[str, dict], predict_fn) -> float | None:
    """predict_fn(train_list, test_list) -> list of confidences for test."""
    ids = list(episodes); pred = {}
    for fold in folds(ids):
        train = [episodes[i] for i in ids if i not in set(fold)]
        for i, c in zip(fold, predict_fn(train, [episodes[i] for i in fold])):
            pred[i] = c
    return prr([episodes[i]["label"] for i in ids], [pred[i] for i in ids])


def raw_rows(cohorts):
    rows = {}
    for signal, (key, hiu) in SIGNALS.items():
        for agg in ("last", "mean", "max"):
            vals = []
            for eps in cohorts.values():
                conf = []
                for e in eps.values():
                    s = e["signals"][signal]
                    v = (s[-1] if agg == "last" else st.fmean(s) if agg == "mean" else max(s)) if s else float("nan")
                    conf.append(v)
                ok = [i for i, c in enumerate(conf) if not math.isnan(c)]
                labels = [list(eps.values())[i]["label"] for i in ok]
                vals.append(prr(labels, [(-conf[i] if hiu else conf[i]) for i in ok]))
            rows[(signal, agg)] = vals
    return rows


def bayes_rows(cohorts):
    rows = {}
    for signal in SIGNALS:
        for family in ("Bayes UQ-only", "Bayes UQ + tools"):
            for mode in ID_MODES:
                vals = []
                for eps in cohorts.values():
                    def fn(train, test, signal=signal, mode=mode, family=family):
                        uq = fit_uq(train, signal, mode); tools = fit_tools(train) if family.endswith("tools") else None
                        prior = sum(e["label"] for e in train) / len(train)
                        out = []
                        for e in test:
                            b = tools.predict_sequence_tempered(e["critics"]) if tools else prior
                            out.append(apply_uq(uq, e["signals"][signal], b, mode) if uq else b)
                        return out
                    vals.append(oof(eps, fn))
                rows[(f"{signal} — {family}", mode)] = vals
    return rows


def reference_rows(cohorts):
    rows = {}
    def verb_final_fused(train, test):
        tools = fit_tools(train)
        seqs = [[e["verb_final"]] for e in train if e["verb_final"] is not None]
        labs = [e["label"] for e in train if e["verb_final"] is not None]
        uq = ContinuousBayesUQ.fit(seqs, labs, lambda_=1.0) if seqs else None
        out = []
        for e in test:
            b = tools.predict_sequence_tempered(e["critics"])
            out.append(uq.update(b, e["verb_final"]) if (uq and e["verb_final"] is not None) else b)
        return out
    rows[("Verb final — Bayes UQ + tools", "Last only")] = [oof(eps, verb_final_fused) for eps in cohorts.values()]
    rows[("Verb final", "final")] = [
        prr([e["label"] for e in eps.values() if e["verb_final"] is not None],
            [e["verb_final"] for e in eps.values() if e["verb_final"] is not None]) for eps in cohorts.values()]
    rows[("Tool success rate", "mean of step tool critics")] = [prr([e["label"] for e in eps.values()], [e["tool_rate"] for e in eps.values()]) for eps in cohorts.values()]
    rows[("N steps", "−N")] = [prr([e["label"] for e in eps.values()], [-e["n_steps"] for e in eps.values()]) for eps in cohorts.values()]
    rows[("Bayes tool-only", "step critics, tempered")] = [oof(eps, lambda tr, te: [fit_tools(tr).predict_sequence_tempered(e["critics"]) for e in te]) for eps in cohorts.values()]
    rows[("Bayes tool-only", "step critics, multiplied")] = [oof(eps, lambda tr, te: [fit_tools(tr).predict_sequence(e["critics"]) for e in te]) for eps in cohorts.values()]
    return rows


def regression_rows(cohorts, toolkit: str):
    sys.path.insert(0, toolkit)
    from trajectory_uq_toolkit.regression import REGRESSION_CONFIGURATIONS, TrajectoryRegression

    def record(eid, e):
        gens = []
        for k, (row, crit) in enumerate(zip(e["rows"], e["critics"])):
            sig = {}
            for name, (key, _) in SIGNALS.items():
                v = ((row.get("uq") or {}).get(SEGMENT) or {}).get(key)
                if v is not None and math.isfinite(float(v)):
                    sig[key] = float(v)
            gens.append({"index": k, "signals": sig, "critics": crit})
        return {"episode_id": eid, "environment": "alfworld", "success": e["label"], "generations": gens}

    rows = {}
    for name, settings in REGRESSION_CONFIGURATIONS.items():
        vals = []
        for eps in cohorts.values():
            recs = {id(e): record(i, e) for i, e in eps.items()}
            def fn(train, test, settings=settings):
                model = TrajectoryRegression.fit([recs[id(e)] for e in train], [e["label"] for e in train], seed=0, **settings)
                return model.predict([recs[id(e)] for e in test])  # in test order
            vals.append(oof(eps, fn))
        rows[("Logistic regression", name)] = vals
    return rows, REGRESSION_CONFIGURATIONS


# ----------------------------------------------------------------------------- OOD
def ood_cell(source: dict, target: dict, predict_fn, seeds=range(5)) -> tuple[float, float]:
    vals = []
    ids = list(source)
    for seed in seeds:
        rng = np.random.RandomState(seed); order = rng.permutation(len(ids))
        train = [source[ids[j]] for j in order[: len(ids) // 2]]
        conf = predict_fn(train, list(target.values()))
        v = prr([e["label"] for e in target.values()], conf)
        if v is not None:
            vals.append(v)
    return (st.fmean(vals), st.stdev(vals) if len(vals) > 1 else 0.0) if vals else (float("nan"), float("nan"))


def ood_methods(toolkit: str | None):
    methods = {}
    for signal in SIGNALS:
        for mode in FUSED_MODES:
            def fn(train, test, signal=signal, mode=mode):
                uq = fit_uq(train, signal, mode); tools = fit_tools(train)
                return [apply_uq(uq, e["signals"][signal], tools.predict_sequence_tempered(e["critics"]), mode) if uq else tools.predict_sequence_tempered(e["critics"]) for e in test]
            methods[(f"{signal} — Bayes UQ + tools", mode)] = fn
    def verb_final(train, test):
        tools = fit_tools(train)
        seqs = [[e["verb_final"]] for e in train if e["verb_final"] is not None]; labs = [e["label"] for e in train if e["verb_final"] is not None]
        uq = ContinuousBayesUQ.fit(seqs, labs, lambda_=1.0) if seqs else None
        return [uq.update(tools.predict_sequence_tempered(e["critics"]), e["verb_final"]) if (uq and e["verb_final"] is not None) else tools.predict_sequence_tempered(e["critics"]) for e in test]
    methods[("Verb final — Bayes UQ + tools", "Last only")] = verb_final
    methods[("Verb final", "final")] = lambda train, test: [e["verb_final"] if e["verb_final"] is not None else 0.5 for e in test]
    methods[("Perplexity", "last")] = lambda train, test: [-(e["signals"]["Perplexity"][-1]) if e["signals"]["Perplexity"] else 0.0 for e in test]
    if toolkit:
        sys.path.insert(0, toolkit)
        from trajectory_uq_toolkit.regression import TrajectoryRegression
        def rec(e, k):
            gens = []
            for j, (row, crit) in enumerate(zip(e["rows"], e["critics"])):
                sig = {}
                for name, (key, _) in SIGNALS.items():
                    v = ((row.get("uq") or {}).get(SEGMENT) or {}).get(key)
                    if v is not None and math.isfinite(float(v)):
                        sig[key] = float(v)
                gens.append({"index": j, "signals": sig, "critics": crit})
            return {"episode_id": f"e{k}", "environment": "alfworld", "success": e["label"], "generations": gens}
        def regression(train, test):
            model = TrajectoryRegression.fit([rec(e, k) for k, e in enumerate(train)], [e["label"] for e in train], seed=0, inverse_penalties=(0.03,))
            return model.predict([rec(e, k) for k, e in enumerate(test)])
        methods[("Logistic regression", "pinned")] = regression
    return methods


# ----------------------------------------------------------------------------- rendering
def f4(v):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


def table(rows: dict, cols: list[str], bold_best=True) -> str:
    keys = list(rows); avgs = {}
    for k in keys:
        ok = [v for v in rows[k] if v is not None and not math.isnan(v)]
        avgs[k] = st.fmean(ok) if ok else float("-inf")
    ranked = sorted(keys, key=lambda k: -avgs[k]); rank = {}
    for pos, k in enumerate(ranked):
        rank[k] = 1 + sum(1 for j in ranked if avgs[j] > avgs[k])
    best = ranked[0] if ranked else None
    out = ["| Rank | Method | Aggregation | " + " | ".join(cols) + " | Avg |", "|---:|---|---|" + "---:|" * len(cols) + "---:|"]
    for k in keys:
        avg = f4(avgs[k]) if avgs[k] != float("-inf") else "n/a"
        if bold_best and k == best: avg = f"**{avg}**"
        out.append(f"| {rank[k]} | {k[0]} | {k[1]} | " + " | ".join(f4(v) for v in rows[k]) + f" | {avg} |")
    return "\n".join(out)


def ood_table(cells: dict, dirs: list[str]) -> str:
    keys = list(cells); avg = {k: st.fmean(m for m, s in cells[k]) for k in keys}
    sd = {k: st.fmean(s for m, s in cells[k]) for k in keys}
    ranked = sorted(keys, key=lambda k: -avg[k]); rank = {k: 1 + sum(1 for j in ranked if avg[j] > avg[k]) for k in keys}
    out = ["| Rank | UQ | Mode | " + " | ".join(dirs) + " | Avg |", "|---:|---|---|" + "---:|" * len(dirs) + "---:|"]
    for k in keys:
        a = f"{avg[k]:.4f} ± {sd[k]:.4f}"
        if k == ranked[0]: a = f"**{a}**"
        out.append(f"| {rank[k]} | {k[0].replace(' — Bayes UQ + tools','')} | {k[1]} | " + " | ".join(f"{m:.4f} ± {s:.4f}" for m, s in cells[k]) + f" | {a} |")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", nargs=2, action="append", metavar=("NAME", "RUN_DIR"), required=True)
    p.add_argument("--toolkit", default=None, help="agentic-uq/src for the main-branch regression")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pairs", default="", help="OOD groups: 'title:A>B,B>A;title:...' using cohort names")
    a = p.parse_args()
    cohorts = {name: load(Path(run)) for name, run in a.cohort}
    cols = list(cohorts)

    md = ["# PRR report — ALFWorld, finished-only, pre-terminal", ""]
    md += ["## Evaluation protocol", "",
           "Sections 1–4 use 5-fold out-of-fold evaluation on the complete cohort: episode indices are shuffled with `numpy.random.RandomState(0)`, fold *i* is `order[i::5]` (not stratified); every episode is predicted exactly once by a model fitted on the other four folds, and each method's PRR@0.5 is computed once on the pooled prediction vector. Raw baselines, Verb final, tool success rate and −N fit nothing.",
           "", "Cohort = ALFWorld valid-seen, 50-step budget, **finished-only** (the agent ended the episode itself: success, `final_answer`, or `give up`; budget exhaustion and external errors excluded — the analogue of Answer-only), and **pre-terminal**: every method sees the episode up to, not including, its last generation, because on a finished episode the last step reveals the outcome (a success ends on the goal-satisfying action, a failure on give-up / final_answer). Labels are binary success; PRR uses them (no partial scores exist).",
           "", f"UQ signals are read from the `{SEGMENT}` response segment of each generation: Logprob = `sum_logprob`, Perplexity, MTE = mean token entropy (top-20 alternatives), Verb actions = per-step verbalised confidence. Self-certainty is not collected on ALFWorld. Tool critics are the five per-step checks: format valid, action admissible, no repeated action, tool success, state changed. **Bayes tool-only** is the tempered critic posterior — per-step log-likelihood ratios averaged over steps, not multiplied (the multiplied form is shown in Section 3 for reference).",
           "", "### Cohorts and folds", ""]
    for name, eps in cohorts.items():
        n = len(eps); s = sum(e["label"] for e in eps.values())
        md.append(f"- **{name}:** {n} finished episodes; successes {s}/{n}. Per fold: train {n - math.ceil(n/5)}–{n - n//5}, test {n//5}–{math.ceil(n/5)}.")
    md += ["", "## 1. UQ baselines — last, mean, max", "",
           "Confidence is the aggregated raw value for logprob and Verb actions and the negative of the aggregated raw value for perplexity and MTE. No parameters are fitted.", ""]
    raw = raw_rows(cohorts)
    for signal in SIGNALS:
        md += [f"### {signal}", "", table({k: v for k, v in raw.items() if k[0] == signal}, cols), ""]
    md += ["## 2. Bayesian UQ — UQ-only and UQ + tools", "",
           "Five variants per signal. SEP, Double, Continuous and Tempered consume the full UQ sequence; Last only fits and applies one ContinuousBayes (λ=1) update on the last value per trajectory. UQ-only starts from the fitted prior; UQ + tools starts from the tempered posterior of the five step critics fitted on the same train fold.", ""]
    bay = bayes_rows(cohorts)
    for signal in SIGNALS:
        md += [f"### {signal}", "", table({k: v for k, v in bay.items() if k[0].startswith(signal + " —")}, cols), ""]
    md += ["## 3. Reference methods", "", table(reference_rows(cohorts), cols), "",
           "Tool success rate is the mean over steps and critics of the five step-critic booleans. N steps uses −N (shorter ranks higher); N counts generations before the terminal step. Verb final is the last verbalised confidence before the terminal step.", ""]
    if a.toolkit:
        reg, cfgs = regression_rows(cohorts, a.toolkit)
        md += ["## 4. Logistic regression — outer cross-fitting", "",
               "The unchanged main implementation (`trajectory_uq_toolkit.regression`, agentic-uq main) fitted independently in each outer train fold: forward feature selection over every (signal, aggregation) column of the four signals plus the five step critics (share of passing steps), two internal hash splits, train-only preprocessing. `selected` searches the penalty; `pinned` fixes C=0.03. No length feature is supplied.", "",
               table(reg, cols), ""]
    if a.pairs:
        md += ["## 7. OOD Bayes fused sweep — five-split means", "",
               "Each signal has eight fused variants (Quantile, SEP, LR+, LR−, Double, Continuous, Tempered, Last only) on top of the tempered tool posterior; Verb final adds a Last-only fused row. For each direction and seed 0–4, every parameter (prior, critic likelihoods, thresholds, Gaussians, regression) is fitted on a random half of the source cohort and scored on the whole target cohort. Cells are mean PRR ± sample SD over the five seeds; Avg/Rank are local to each direction pair.", ""]
        methods = ood_methods(a.toolkit)
        groups = []
        for group in a.pairs.split(";"):
            title, dirs = group.split(":", 1); dirs = [d.strip() for d in dirs.split(",")]
            cells = {k: [] for k in methods}
            for d in dirs:
                src, tgt = [x.strip() for x in d.split(">")]
                for k, fn in methods.items():
                    cells[k].append(ood_cell(cohorts[src], cohorts[tgt], fn))
            groups.append((title, dirs, cells))
            md += [f"### {title}", "", ood_table(cells, [d.replace(">", " → ") for d in dirs]), ""]
        # overall
        md += ["### Overall summary — all methods", "", "Overall Avg equally weights all direction means; SD is the mean of the per-cell SDs.", ""]
        keys = list(methods); overall = {}
        for k in keys:
            per_group = [st.fmean(m for m, s in g[2][k]) for g in groups]
            overall[k] = (per_group, st.fmean(per_group), st.fmean(s for g in groups for m, s in g[2][k]))
        ranked = sorted(keys, key=lambda k: -overall[k][1])
        md += ["| Rank | Method | Aggregation | " + " | ".join(f"Avg {g[0].split(' — ')[0].replace('Directions ','')}" for g in groups) + " | Overall Avg |",
               "|---:|---|---|" + "---:|" * len(groups) + "---:|"]
        for pos, k in enumerate(ranked):
            pg, ov, sd = overall[k]; o = f"{ov:.4f} ± {sd:.4f}"
            if pos == 0: o = f"**{o}**"
            md.append(f"| {pos+1} | {k[0]} | {k[1]} | " + " | ".join(f"{v:.4f}" for v in pg) + f" | {o} |")
        md.append("")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(md))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
