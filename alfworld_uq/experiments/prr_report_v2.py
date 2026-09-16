"""Standard PRR report for ALFWorld / DeepSWE, in the OSWorld/WebArena format.

Protocol (the one the reference report uses):

* Cross-fitting: 3 seeds (0, 1, 2) x 5 stratified folds; fit on four folds,
  predict the held-out fold; pool all held-out predictions and compute one
  PRR@0.5 per seed; report mean +- sample SD over seeds.
* Score: the cohort's native score (binary success on ALFWorld, the verifier's
  continuous partial score on DeepSWE). Binary labels are used for fitting and
  stratification only; on DeepSWE they are the training fold's median split.
* OOD: fit on the source's four training folds and reuse the parameters,
  unchanged, on the target's corresponding test fold; all five folds, three
  seeds; no target-side fitting.
* Methods: raw last / mean / max of every UQ signal; Bayes UQ-only and
  UQ + tools with SEP, Double, Continuous (lambda=1), Tempered (lambda=.25),
  Last only, Mean only; reference methods (logistic regression pinned /
  selected, TemporalBelief B4 at the final checkpoint, -N, tool success rate,
  Bayes tool-only as episode critics, step critics multiplied and tempered).

    python -m experiments.prr_report_v2 --out REPORT.tex --toolkit agentic-uq/src \\
        --alfworld "RG=ReAct/gpt-oss" runs/react_gptoss_140_giveup_sc_finished_nolast ... \\
        --deepswe "DG=gpt-oss-20b" deep_swe_uq/runs/deepswe_gptoss_113_v5/compact.jsonl ...
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from belief.binary_bayes import BinaryBayesUQ, DoubleBinaryBayesUQ
from belief.continuous_bayes import ContinuousBayesUQ
from belief.critic_bayes import CriticBayesState
from experiments.analyze_trajectories import (
    STEP_CRITIC_NAMES,
    _critic_observations,
    _step_critic_observation,
)
from experiments.deepswe_table import _TEST_CMD, prr_continuous

SEEDS = (0, 1, 2)
FOLDS = 5
SEGMENT = "combined"
# label -> (stored key on ALFWorld, stored key on DeepSWE, higher value = more uncertain)
SIGNALS = {
    "Logprob": ("sum_logprob", "mean_logprob", False),
    "Perplexity": ("perplexity", "perplexity", True),
    "MTE": ("mean_token_entropy", "mean_entropy", True),
    "Self-certainty": ("self_certainty", "self_certainty", False),
    "Verb actions": ("verbalized_confidence", "confidence", False),
}
UQ_MODES = ("SEP", "Double", "Continuous", "Tempered", "Last only", "Mean only")
OOD_MODES = ("Quantile", "SEP", "LR+", "LR-", "Double", "Continuous", "Tempered", "Last only", "Mean only")
MODE_LABEL = {"Continuous": "Continuous (\\ensuremath{\\lambda}=1)", "Tempered": "Tempered (\\ensuremath{\\lambda}=0.25)", "LR-": "LR\\ensuremath{-}"}


# ----------------------------------------------------------------------------- loading
def load_alfworld(run: Path, harness: str) -> dict[str, dict[str, Any]]:
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
        for name, (key, _, _) in SIGNALS.items():
            vals = [((r.get("uq") or {}).get(SEGMENT) or {}).get(key) for r in rows]
            seqs[name] = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
        crit = [{c: bool(_step_critic_observation(r)[c]) for c in STEP_CRITIC_NAMES} for r in rows]
        episodes[e["episode_id"]] = {
            "id": e["episode_id"], "harness": harness,
            "score": float(bool(e["final_success"])), "label": int(bool(e["final_success"])),
            "signals": seqs, "critics": crit, "episode_critics": _critic_observations(rows),
            "n_steps": len(rows), "tool_rate": st.fmean(v for c in crit for v in c.values()),
            "record": _record_alfworld(e["episode_id"], rows, crit, int(bool(e["final_success"]))),
        }
    return episodes


def _record_alfworld(eid, rows, crit, label):
    gens = []
    for k, (row, c) in enumerate(zip(rows, crit)):
        sig = {}
        for name, (key, _, _) in SIGNALS.items():
            v = ((row.get("uq") or {}).get(SEGMENT) or {}).get(key)
            if v is not None and math.isfinite(float(v)):
                sig[key] = float(v)
        gens.append({"index": k, "signals": sig, "critics": dict(c)})
    return {"episode_id": eid, "environment": "alfworld", "success": label, "generations": gens}


DEEPSWE_CRITICS = ("ran_tests", "ran_tests_twice", "last_test_passed", "test_flipped", "committed_twice", "no_format_errors")


def load_deepswe(path: Path, harness: str, verb: Path | None = None) -> dict[str, dict[str, Any]]:
    confidences: dict[tuple[str, int], float] = {}
    if verb and verb.exists():
        for line in open(verb):
            r = json.loads(line)
            if r.get("confidence") is not None:
                confidences[(r["id"], r["step"])] = float(r["confidence"]) / 100.0
    episodes = {}
    for line in open(path):
        r = json.loads(line); rw = r["rewards"]; steps = r["steps"]
        if not rw or r["patch_bytes"] <= 0 or r["exit_status"] != "Submitted":
            continue
        for k, s in enumerate(steps):
            if (r["id"], k) in confidences:
                s["confidence"] = confidences[(r["id"], k)]
        if len(steps) > 1:
            steps = steps[:-1]  # pre-terminal: before the submit command
        lp = [s["mean_logprob"] for s in steps if s["mean_logprob"] is not None]
        if not lp:
            continue
        seqs = {
            "Logprob": lp,
            "Perplexity": [math.exp(-v) for v in lp],
            "MTE": [s["mean_entropy"] for s in steps if s["mean_entropy"] is not None],
            "Self-certainty": [s["self_certainty"] for s in steps if s.get("self_certainty") is not None],
            "Verb actions": [s["confidence"] for s in steps if s.get("confidence") is not None],
        }
        cmds = [s["command"] for s in steps if s["command"]]
        test_rcs = [s["returncode"] for s in steps if s["command"] and _TEST_CMD.search(s["command"]) and s["returncode"] is not None]
        first_fail = next((k for k, rc in enumerate(test_rcs) if rc != 0), None)
        ep_crit = {
            "ran_tests": len(test_rcs) > 0, "ran_tests_twice": len(test_rcs) >= 2,
            "last_test_passed": bool(test_rcs) and test_rcs[-1] == 0,
            "test_flipped": first_fail is not None and any(rc == 0 for rc in test_rcs[first_fail:]),
            "committed_twice": sum("git commit" in c for c in cmds) >= 2,
            "no_format_errors": not any(s["format_error"] for s in steps),
        }
        rc0 = [s["returncode"] == 0 for s in steps if s["returncode"] is not None]
        gens = []
        for k, s in enumerate(steps):
            sig = {kk: float(v) for kk, v in (("mean_logprob", s["mean_logprob"]), ("perplexity", math.exp(-s["mean_logprob"]) if s["mean_logprob"] is not None else None),
                                               ("mean_entropy", s["mean_entropy"]), ("self_certainty", s.get("self_certainty")), ("verbalized_confidence", s.get("confidence"))) if v is not None}  # B4 excludes verbalized by name
            gens.append({"index": k, "signals": sig, "critics": dict(ep_crit)})
        episodes[r["id"]] = {
            "id": r["id"], "harness": harness, "score": float(rw.get("partial", 0.0)), "label": 0,
            "signals": seqs, "critics": [dict(ep_crit) for _ in steps], "episode_critics": ep_crit,
            "n_steps": len(steps), "tool_rate": st.fmean(rc0) if rc0 else 0.0,
            "record": {"episode_id": r["id"], "environment": "deepswe", "success": 0, "generations": gens},
        }
    # A signal most episodes do not carry is unavailable for every method,
    # including the regression and B4 records (a column present on one
    # cohort and absent on another breaks fold-matched OOD scoring).
    if episodes and sum(1 for e in episodes.values() if e["signals"]["Verb actions"]) < 0.5 * len(episodes):
        for e in episodes.values():
            e["signals"]["Verb actions"] = []
            for g in e["record"]["generations"]:
                g["signals"].pop("verbalized_confidence", None)
    return episodes


# ----------------------------------------------------------------------------- protocol
def is_continuous(eps: dict) -> bool:
    return any(not float(e["score"]).is_integer() for e in eps.values())


def with_labels(train: list[dict], continuous: bool) -> list[dict]:
    """Binary training labels: the score itself, or the fold's own median split."""
    if not continuous:
        return [dict(e, label=int(e["score"]), record=dict(e["record"], success=int(e["score"]))) for e in train]
    med = st.median(e["score"] for e in train)
    return [dict(e, label=int(e["score"] > med), record=dict(e["record"], success=int(e["score"] > med))) for e in train]


def stratified_folds(eps: dict, seed: int) -> list[list[str]]:
    ids = list(eps); continuous = is_continuous(eps)
    if continuous:
        med = st.median(eps[i]["score"] for i in ids); strata = {i: int(eps[i]["score"] > med) for i in ids}
    else:
        strata = {i: int(eps[i]["score"]) for i in ids}
    rng = np.random.RandomState(seed); folds = [[] for _ in range(FOLDS)]
    for value in sorted(set(strata.values())):
        members = [i for i in ids if strata[i] == value]; rng.shuffle(members)
        for k, i in enumerate(members):
            folds[k % FOLDS].append(i)
    return folds


def prr(scores, conf):
    return prr_continuous(scores, conf) if len(scores) > 1 else None


def oof(eps: dict, fn: Callable, seed: int) -> float | None:
    continuous = is_continuous(eps); folds = stratified_folds(eps, seed); pred = {}
    for fold in folds:
        held = set(fold)
        train = with_labels([eps[i] for i in eps if i not in held], continuous)
        for i, c in zip(fold, fn(train, [eps[i] for i in fold])):
            pred[i] = c
    ids = list(eps)
    if any(pred[i] is None for i in ids):
        return None
    return prr([eps[i]["score"] for i in ids], [pred[i] for i in ids])


def ood(src: dict, tgt: dict, fn: Callable, seed: int) -> float | None:
    """Fit on the source's four training folds, score the target's test fold f."""
    sf = stratified_folds(src, seed); tf = stratified_folds(tgt, seed); pred = {}
    continuous = is_continuous(src)
    for f in range(FOLDS):
        held = set(sf[f])
        train = with_labels([src[i] for i in src if i not in held], continuous)
        for i, c in zip(tf[f], fn(train, [tgt[i] for i in tf[f]])):
            pred[i] = c
    ids = list(tgt)
    if any(pred[i] is None for i in ids):
        return None
    return prr([tgt[i]["score"] for i in ids], [pred[i] for i in ids])


def mean_sd(values: list[float | None]) -> tuple[float, float] | None:
    ok = [v for v in values if v is not None]
    if not ok:
        return None
    return (st.fmean(ok), st.stdev(ok) if len(ok) > 1 else 0.0)


# ----------------------------------------------------------------------------- models
def fit_tools(train: list[dict], kind: str):
    if kind == "episode":
        return CriticBayesState.fit([e["episode_critics"] for e in train], [e["label"] for e in train],
                                    prior=st.fmean(e["label"] for e in train))
    obs = [c for e in train for c in e["critics"]]; lab = [e["label"] for e in train for _ in e["critics"]]
    return CriticBayesState.fit(obs, lab, prior=st.fmean(e["label"] for e in train))


def tool_belief(model, e: dict, kind: str) -> float:
    if kind == "episode":
        return model.predict(e["episode_critics"])
    if kind == "multiplied":
        return model.predict_sequence(e["critics"])
    return model.predict_sequence_tempered(e["critics"])


def fit_uq(train: list[dict], signal: str, mode: str):
    hiu = SIGNALS[signal][2]
    seqs = [e["signals"][signal] for e in train if e["signals"][signal]]
    labs = [e["label"] for e in train if e["signals"][signal]]
    if not seqs or len(set(labs)) < 2:
        return None
    if mode == "Quantile":
        return BinaryBayesUQ.fit(seqs, labs, threshold_mode="quantile", higher_is_uncertain=hiu)
    if mode == "SEP":
        return BinaryBayesUQ.fit(seqs, labs, threshold_mode="sep", higher_is_uncertain=hiu)
    if mode == "LR+":
        return BinaryBayesUQ.fit(seqs, labs, threshold_mode="lr_pos", higher_is_uncertain=hiu)
    if mode == "LR-":
        return BinaryBayesUQ.fit(seqs, labs, threshold_mode="lr_neg", higher_is_uncertain=hiu)
    if mode == "Double":
        return DoubleBinaryBayesUQ.fit(seqs, labs, higher_is_uncertain=hiu)
    if mode == "Continuous":
        return ContinuousBayesUQ.fit(seqs, labs, lambda_=1.0)
    if mode == "Tempered":
        return ContinuousBayesUQ.fit(seqs, labs, lambda_=0.25)
    if mode == "Last only":
        return ContinuousBayesUQ.fit([[s[-1]] for s in seqs], labs, lambda_=1.0)
    if mode == "Mean only":
        return ContinuousBayesUQ.fit([[st.fmean(s)] for s in seqs], labs, lambda_=1.0)
    raise ValueError(mode)


def apply_uq(model, seq: list[float], belief: float, mode: str) -> float:
    if model is None or not seq:
        return belief
    values = [seq[-1]] if mode == "Last only" else [st.fmean(seq)] if mode == "Mean only" else seq
    for v in values:
        belief = model.update(belief, v)
    return belief


def covered(eps: list[dict], signal: str, minimum: float = 0.5) -> bool:
    """A signal that most episodes do not carry is reported as unavailable."""
    return bool(eps) and sum(1 for e in eps if e["signals"][signal]) >= minimum * len(eps)


def method_table(toolkit: str | None, tool_kind: str = "tempered") -> dict[tuple[str, str, str], Callable]:
    """(family, method, aggregation) -> fn(train, test) -> confidences."""
    methods: dict[tuple[str, str, str], Callable] = {}
    for signal, (_, _, hiu) in SIGNALS.items():
        sgn = -1.0 if hiu else 1.0
        for agg in ("last", "mean", "max"):
            def raw(train, test, signal=signal, agg=agg, sgn=sgn):
                if not covered(test, signal):
                    return [None] * len(test)
                out = []
                for e in test:
                    s = e["signals"][signal]
                    if not s:
                        out.append(0.0); continue
                    v = s[-1] if agg == "last" else st.fmean(s) if agg == "mean" else max(s)
                    out.append(sgn * v)
                return out
            methods[(signal, signal, agg)] = raw
        for family in ("Bayes UQ-only", "Bayes UQ + tools"):
            for mode in UQ_MODES:
                def bayes(train, test, signal=signal, mode=mode, family=family):
                    if not covered(test, signal) or not covered(train, signal):
                        return [None] * len(test)
                    uq = fit_uq(train, signal, mode)
                    prior = st.fmean(e["label"] for e in train)
                    tools = fit_tools(train, tool_kind) if family.endswith("tools") else None
                    return [apply_uq(uq, e["signals"][signal], tool_belief(tools, e, tool_kind) if tools else prior, mode) for e in test]
                methods[(signal, f"{signal} \\ensuremath{{-}} {family}", mode)] = bayes
    # reference methods
    methods[("Reference", "N steps", "\\ensuremath{-}N")] = lambda train, test: [-float(e["n_steps"]) for e in test]
    methods[("Reference", "Tool success rate", "mean of observed tool critics")] = lambda train, test: [e["tool_rate"] for e in test]
    methods[("Reference", "Bayes tool-only", "critic:all")] = lambda train, test: [tool_belief(fit_tools(train, "episode"), e, "episode") for e in test]
    methods[("Reference", "Bayes tool-only", "Step critics, multiplied")] = lambda train, test: [tool_belief(fit_tools(train, "multiplied"), e, "multiplied") for e in test]
    methods[("Reference", "Bayes tool-only", "Step critics, tempered")] = lambda train, test: [tool_belief(fit_tools(train, "tempered"), e, "tempered") for e in test]
    if toolkit:
        sys.path.insert(0, toolkit)
        from trajectory_uq_toolkit import TemporalBelief, TrajectoryRegression
        from trajectory_uq_toolkit.regression import REGRESSION_CONFIGURATIONS

        for name, settings in REGRESSION_CONFIGURATIONS.items():
            def regression(train, test, settings=settings):
                model = TrajectoryRegression.fit([e["record"] for e in train], [e["label"] for e in train], seed=0, **settings)
                try:
                    return model.predict([e["record"] for e in test])
                except ValueError:  # a selected column the target does not supply
                    return [None] * len(test)
            methods[("Reference", "Logistic regression", name)] = regression

        def b4(train, test):
            model = TemporalBelief.fit([e["record"] for e in train], checkpoint=None, harnesses=[e["harness"] for e in train])
            try:
                return [model.predict(e["record"], harness=e["harness"]) for e in test]
            except (ValueError, KeyError):
                return [None] * len(test)
        methods[("Reference", "TemporalBelief (B4)", "final checkpoint")] = b4
    return methods


# ----------------------------------------------------------------------------- rendering
def fmt(v: tuple[float, float] | None) -> str:
    return "n/a" if v is None else f"\\PRR{{{v[0]:.2f}}}{{{v[1]:.2f}}}"


def longtable(title: str, cols: list[str], rows: list[tuple[str, str, list[tuple[float, float] | None]]], widths: str) -> str:
    """rows: (method, aggregation, per-column (mean, sd)); Avg and Rank are added."""
    scored = []
    for method, agg, cells in rows:
        ok = [c for c in cells if c is not None]
        avg = (st.fmean(c[0] for c in ok), st.fmean(c[1] for c in ok)) if ok else None
        scored.append((method, agg, cells, avg))
    order = sorted(range(len(scored)), key=lambda k: -(scored[k][3][0] if scored[k][3] else -9))
    rank = {k: r + 1 for r, k in enumerate(order)}
    head = " & ".join(["\\textbf{Rank}", "\\textbf{Method}", "\\textbf{Aggregation}"] + [f"\\textbf{{{c}}}" for c in cols] + ["\\textbf{Avg}"]) + " \\\\"
    n = len(cols) + 4
    out = [f"\\Needspace{{45mm}}", f"\\section*{{{title}}}", "\\begingroup", "\\small", f"\\begin{{longtable}}{{{widths}}}", "\\toprule", head, "\\midrule", "\\endfirsthead",
           f"\\multicolumn{{{n}}}{{@{{}}l}}{{\\emph{{{title} (continued)}}}} \\\\", "\\toprule", head, "\\midrule", "\\endhead", "\\midrule",
           f"\\multicolumn{{{n}}}{{r@{{}}}}{{\\emph{{Continued on next page}}}} \\\\", "\\endfoot", "\\bottomrule", "\\endlastfoot"]
    for k, (method, agg, cells, avg) in enumerate(scored):
        out.append(f"{rank[k]} & {method} & {agg} & " + " & ".join(fmt(c) for c in cells) + f" & {fmt(avg)} \\\\")
    out += ["\\end{longtable}", "\\endgroup"]
    return "\n".join(out) + "\n"


def widths_for(ncols: int) -> str:
    cell = {2: 40, 3: 32, 4: 28, 6: 22}.get(ncols, 22)
    method = {2: 70, 3: 63, 4: 58, 6: 48}.get(ncols, 48)
    return "@{}C{7mm}L{" + str(method) + "mm}L{37mm}" + "".join(f"C{{{cell}mm}}" for _ in range(ncols)) + f"C{{{cell}mm}}@{{}}"


PREAMBLE = r"""% Self-contained report; no external figures or input files required.
% Compile with pdfLaTeX twice, or upload to Overleaf.
\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[landscape,margin=16mm]{geometry}
\usepackage{array,booktabs,longtable}
\usepackage{enumitem,needspace}
\usepackage[hidelinks]{hyperref}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}
\newcommand{\PRR}[2]{\ensuremath{#1\,\pm\,#2}}
\setlength{\tabcolsep}{3pt}
\setlength{\LTleft}{0pt}
\setlength{\LTright}{\fill}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\renewcommand{\arraystretch}{1.18}
\setlist[itemize]{leftmargin=*,itemsep=3pt,topsep=4pt}
\emergencystretch=2em
"""


def render_dataset(name: str, cohorts: dict[str, dict], labels: dict[str, str], groups: list[tuple[str, list[tuple[str, str]]]], methods, seeds, notes: str) -> str:
    cols = list(cohorts); col_labels = [labels[c] for c in cols]
    out = [f"\\clearpage", f"\\section*{{{name}}}", notes, ""]
    # datasets table
    out += ["\\Needspace{50mm}", "\\subsection*{Datasets}", "\\begin{tabular}{@{}L{60mm}C{30mm}C{30mm}C{40mm}@{}}", "\\toprule",
            "\\textbf{Cohort} & \\textbf{Episodes} & \\textbf{Mean score} & \\textbf{Continuous score} \\\\", "\\midrule"]
    for c in cols:
        eps = cohorts[c]; out.append(f"{labels[c]} & {len(eps)} & {st.fmean(e['score'] for e in eps.values()):.2f} & {'yes (partial credit)' if is_continuous(eps) else 'no (binary success)'} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", ""]
    # in-domain
    cache: dict[tuple, list] = {}
    def idcell(c, key):
        if (c, key) not in cache:
            cache[(c, key)] = [oof(cohorts[c], methods[key], s) for s in seeds]
        return mean_sd(cache[(c, key)])
    families = [s for s in SIGNALS]
    best_rows = []
    for signal in families:
        raw_rows = [(m, a, [idcell(c, k) for c in cols]) for k in methods if k[0] == signal and k[1] == signal for (_, m, a) in [k]]
        bay_rows = [(m, a, [idcell(c, k) for c in cols]) for k in methods if k[0] == signal and k[1] != signal for (_, m, a) in [k]]
        bay_rows = [(m, MODE_LABEL.get(a, a), cells) for m, a, cells in bay_rows]
        if all(all(c is None for c in cells) for _, _, cells in raw_rows):
            out.append(f"\\Needspace{{20mm}}\\subsection*{{{signal}}}\nNot available on this benchmark.\n")
            continue
        out.append(longtable(f"{name} \\ensuremath{{-}} {signal} \\ensuremath{{-}} raw UQ", col_labels, raw_rows, widths_for(len(cols))))
        out.append(longtable(f"{name} \\ensuremath{{-}} {signal} \\ensuremath{{-}} Bayesian variants", col_labels, bay_rows, widths_for(len(cols))))
        for rows in (raw_rows, bay_rows):
            ok = [(m, a, cells) for m, a, cells in rows if any(c is not None for c in cells)]
            if ok:
                best = max(ok, key=lambda r: st.fmean(c[0] for c in r[2] if c is not None))
                best_rows.append(best)
    ref_rows = [(m, a, [idcell(c, k) for c in cols]) for k in methods if k[0] == "Reference" for (_, m, a) in [k]]
    out.append(longtable(f"{name} \\ensuremath{{-}} Reference methods", col_labels, ref_rows, widths_for(len(cols))))
    out.append(longtable(f"{name} \\ensuremath{{-}} ID summary \\ensuremath{{-}} best per UQ family", col_labels, best_rows + ref_rows, widths_for(len(cols))))
    # OOD
    ood_methods = {k: fn for k, fn in methods.items() if k[0] == "Reference" or (k[1] != k[0] and "tools" in k[1])}
    overall: dict[tuple, list] = defaultdict(list)
    for title, dirs in groups:
        cells_by_method = {k: [] for k in ood_methods}
        for s_name, t_name in dirs:
            for k, fn in ood_methods.items():
                cells_by_method[k].append(mean_sd([ood(cohorts[s_name], cohorts[t_name], fn, s) for s in seeds]))
        rows = [(m, MODE_LABEL.get(a, a), cells_by_method[k]) for k in ood_methods for (_, m, a) in [k]]
        for k in ood_methods:
            ok = [c for c in cells_by_method[k] if c is not None]
            overall[k].append((st.fmean(c[0] for c in ok), st.fmean(c[1] for c in ok)) if ok else None)
        dir_labels = [f"{labels[s]} \\ensuremath{{\\to}} {labels[t]}" for s, t in dirs]
        out.append(longtable(f"{name} \\ensuremath{{-}} OOD {title}", dir_labels, rows, widths_for(len(dirs))))
    if len(groups) > 1:
        rows = [(m, MODE_LABEL.get(a, a), overall[k]) for k in ood_methods for (_, m, a) in [k]]
        out.append(longtable(f"{name} \\ensuremath{{-}} OOD overall \\ensuremath{{-}} all methods", [g[0] for g in groups], rows, widths_for(len(groups))))
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--toolkit", default=None)
    p.add_argument("--alfworld", nargs=2, action="append", metavar=("KEY=LABEL", "RUN_DIR"), default=[])
    p.add_argument("--deepswe", nargs=2, action="append", metavar=("KEY=LABEL", "COMPACT"), default=[])
    p.add_argument("--deepswe-verb", nargs=2, action="append", metavar=("KEY", "VERB_JSONL"), default=[])
    p.add_argument("--title", default="Standard PRR \\ensuremath{-} ALFWorld and DeepSWE")
    a = p.parse_args()
    methods = method_table(a.toolkit)
    doc = [PREAMBLE, f"\\hypersetup{{pdftitle={{{a.title}}}}}", f"\\title{{{a.title}}}", "\\author{}", "\\date{}", "\\begin{document}", "\\maketitle", "\\vspace{-8mm}",
           r"""\begin{itemize}
\item \textbf{Unit:} one confidence score and one native task score per episode; PRR is episode-level, including for stepwise methods.
\item \textbf{Cohorts:} ALFWorld: finished-only (the agent ended the episode itself: success, final answer, or give-up) and scored before the terminal generation; ReAct uses the give-up variant. DeepSWE: submitted-only and scored before the submit command. Both ALFWorld and DeepSWE trajectories are truncated at the outcome-revealing terminal action; the reference OSWorld/WebArena report does not truncate.
\item \textbf{Cross-fitting:} 3 seeds (0, 1, 2) \ensuremath{\times} 5 stratified folds; fit on 4 folds and predict the held-out fold. ALFWorld stratifies on success; DeepSWE on the cohort median of the partial score.
\item \textbf{Metric:} PRR@0.5 on native scores: binary success on ALFWorld, the verifier's continuous partial score (passed / all hidden tests) on DeepSWE, where no model resolves a task outright. Pool all held-out predictions before computing one PRR per seed; report mean \ensuremath{\pm} sample SD over seeds. Binary labels are used for fitting and stratification only; on DeepSWE they are the training fold's own median split.
\item \textbf{Signals:} Logprob = summed token log-probability of a generation (ALFWorld) / mean token log-probability of a command (DeepSWE); Perplexity; MTE = mean token entropy over the top-\ensuremath{k} alternatives; Self-certainty = \ensuremath{-\mathrm{mean}(\log p_{\mathrm{top}\text{-}k}) - \log k} per token; Verb actions = per-step verbalized confidence (ALFWorld: a required line of the response format; DeepSWE: a separate side query after every command, replayed on the recorded trajectories, since both models ignore an in-loop instruction).
\item \textbf{Step critics:} ALFWorld: format valid, action admissible, no repeat, tool success, state changed, one row per generation. DeepSWE: ran the test suite, ran it twice, last run passed, a run that failed then passed, committed twice, no format errors; these are episode-level and repeated per row, so the multiplied and tempered forms coincide up to the step count. \texttt{critic:all} = the episode-level formats-valid / actions-admissible / no-repeat flags of the original formulation.
\item \textbf{Bayes Fused:} start from the tempered step-critic posterior (log-likelihood ratios averaged over rows), then update with UQ. Tool success rate alone is the arithmetic mean of the step-critic flags (ALFWorld) / the share of commands with return code 0 (DeepSWE); Fused does not use that mean as its tool input.
\item \textbf{UQ updates:} SEP/Double/Continuous/Tempered process the UQ sequence; Tempered uses \ensuremath{\lambda}=0.25. Last only fits and applies one update on the last UQ value; Mean only does so on the trajectory's arithmetic mean UQ. Raw last/mean/max have no learned transform.
\item \textbf{Reference methods:} logistic regression = the unchanged main-branch \texttt{TrajectoryRegression} (pinned \ensuremath{C}=0.03 / selected), given the same signals and critics; TemporalBelief (B4) = the main-branch model at the final checkpoint, one harness per cohort; elicited signals (Verb) are excluded from B4 and regression by construction.
\item \textbf{OOD:} fit only on the source's 4 training folds and reuse those parameters, unchanged, on the target's corresponding test fold; all 5 folds and 3 seeds, no target-side fitting. Avg is the equal-weight mean over directions; Overall is the mean over direction groups.
\item \textbf{Averages and ranks:} Avg is the equal-weight mean over cohorts/directions of the seed means; its \ensuremath{\pm} is the mean of the per-cell SDs. Ranks are descriptive selections by Avg. Raw \ensuremath{\pm}0 reflects split-invariant rankings, not zero statistical uncertainty.
\end{itemize}"""]
    if a.alfworld:
        labels = {}; cohorts = {}
        for kl, run in a.alfworld:
            key, label = kl.split("=", 1); labels[key] = label; cohorts[key] = load_alfworld(Path(run), key)
        keys = list(cohorts)
        # groups by what changes; keys are expected as RG, SG, RQ, SQ (agent letter, model letter)
        def agent(k): return k[0]
        def model(k): return k[1]
        pairs = [(s, t) for s in keys for t in keys if s != t]
        groups = [("directions 1\\ensuremath{-}4 \\ensuremath{-} same model, different agent", [(s, t) for s, t in pairs if model(s) == model(t) and agent(s) != agent(t)]),
                  ("directions 5\\ensuremath{-}8 \\ensuremath{-} different model, different agent", [(s, t) for s, t in pairs if model(s) != model(t) and agent(s) != agent(t)]),
                  ("directions 9\\ensuremath{-}12 \\ensuremath{-} different model, same agent", [(s, t) for s, t in pairs if model(s) != model(t) and agent(s) == agent(t)])]
        notes = "RG = ReAct/gpt-oss-20b; SG = smolagents/gpt-oss-20b; RQ = ReAct/Qwen3.6-35B-A3B; SQ = smolagents/Qwen3.6-35B-A3B. 140 episodes per setup before the finished-only filter, 50-step budget; ReAct with the give-up action."
        doc.append(render_dataset("ALFWorld", cohorts, labels, groups, methods, SEEDS, notes))
    if a.deepswe:
        labels = {}; cohorts = {}; verbs = dict(a.deepswe_verb)
        for kl, path in a.deepswe:
            key, label = kl.split("=", 1); labels[key] = label
            cohorts[key] = load_deepswe(Path(path), key, Path(verbs[key]) if key in verbs else None)
        keys = list(cohorts)
        groups = [("directions \\ensuremath{-} same agent, different model", [(s, t) for s in keys for t in keys if s != t])]
        notes = "DG = mini-swe-agent/gpt-oss-20b; DQ = mini-swe-agent/Qwen3.6-35B-A3B. 113 DeepSWE tasks, 200-command budget, working-tree grading; the score is the verifier's partial credit."
        doc.append(render_dataset("DeepSWE", cohorts, labels, groups, methods, SEEDS, notes))
    doc.append("\\end{document}\n")
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text("\n".join(doc)); print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
