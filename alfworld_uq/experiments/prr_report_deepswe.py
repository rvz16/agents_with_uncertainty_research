"""The PRR report (sections 1-4, 7) for DeepSWE, from the compact runs.

Same layout and protocol as `prr_report` (5-fold OOF, five-seed OOD), on the
verifier's continuous partial score instead of a binary label: PRR ranks
against the native score without binarisation, the oracle ranks by the true
score and the random reference is the mean score. Bayes and regression need
a binary training label; each training fold is split at its own median.

    python -m experiments.prr_report_deepswe --out REPORT.md --toolkit agentic-uq/src \\
        --cohort "gpt-oss-20b" runs/deepswe_gptoss_113_v5/compact.jsonl ...
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

import numpy as np

from experiments import prr_report as base
from experiments.deepswe_table import _TEST_CMD, prr_continuous

SIGNALS = {
    "Logprob": ("mean_logprob", False),
    "Perplexity": ("perplexity", True),
    "MTE": ("mean_entropy", True),
}
CRITICS = ("no_format_errors", "ran_tests", "committed", "no_repeated_command", "test_flipped", "last_test_passed")


def load(path: Path) -> dict[str, dict]:
    episodes = {}
    for line in open(path):
        r = json.loads(line); rw = r["rewards"]; steps = r["steps"]
        if not rw or r["patch_bytes"] <= 0 or r["exit_status"] != "Submitted":
            continue
        if len(steps) > 1:
            steps = steps[:-1]  # pre-terminal, as on ALFWorld
        lp = [s["mean_logprob"] for s in steps if s["mean_logprob"] is not None]
        ent = [s["mean_entropy"] for s in steps if s["mean_entropy"] is not None]
        if not lp:
            continue
        cmds = [s["command"] for s in steps if s["command"]]
        test_rcs = [s["returncode"] for s in steps if s["command"] and _TEST_CMD.search(s["command"]) and s["returncode"] is not None]
        first_fail = next((k for k, rc in enumerate(test_rcs) if rc != 0), None)
        flipped = first_fail is not None and any(rc == 0 for rc in test_rcs[first_fail:])
        # per-step critics: the process ones are per step, the test ones are per episode and repeated
        rows = []; seen = {}
        for s in steps:
            c = s["command"] or ""; seen[c] = seen.get(c, 0) + 1
            rows.append({
                "no_format_errors": not s["format_error"],
                "ran_tests": bool(_TEST_CMD.search(c)),
                "committed": "git commit" in c,
                "no_repeated_command": seen[c] < 3,
                "test_flipped": bool(flipped),
                "last_test_passed": bool(test_rcs) and test_rcs[-1] == 0,
            })
        rc0 = [s["returncode"] == 0 for s in steps if s["returncode"] is not None]
        episodes[r["id"]] = {
            "_key": r["id"], "label": 0, "score": float(rw.get("partial", 0.0)),
            "signals": {"Logprob": lp, "Perplexity": [math.exp(-v) for v in lp], "MTE": ent},
            "critics": rows, "verb_final": None, "n_steps": len(steps),
            "tool_rate": st.fmean(rc0) if rc0 else 0.0,
            "rows": [{"uq": {"combined": {"mean_logprob": s["mean_logprob"], "perplexity": (math.exp(-s["mean_logprob"]) if s["mean_logprob"] is not None else None), "mean_entropy": s["mean_entropy"]}}} for s in steps],
        }
    return episodes


def with_median_labels(train: list[dict]) -> list[dict]:
    med = st.median(e["score"] for e in train)
    return [dict(e, label=int(e["score"] > med)) for e in train]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", nargs=2, action="append", metavar=("NAME", "COMPACT"), required=True)
    p.add_argument("--toolkit", default=None)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    # --- adapt the ALFWorld report to continuous scores and DeepSWE signals
    base.SIGNALS = SIGNALS
    base.STEP_CRITIC_NAMES = CRITICS
    base.prr = lambda labels, conf: prr_continuous(labels, conf) if len(labels) > 1 else None

    def oof(episodes, predict_fn):
        ids = list(episodes); pred = {}
        for fold in base.folds(ids):
            train = with_median_labels([episodes[i] for i in ids if i not in set(fold)])
            for i, c in zip(fold, predict_fn(train, [episodes[i] for i in fold])):
                pred[i] = c
        return prr_continuous([episodes[i]["score"] for i in ids], [pred[i] for i in ids])
    base.oof = oof

    _ood = base.ood_cell
    def ood_cell(source, target, predict_fn, seeds=range(5)):
        vals = []; ids = list(source)
        for seed in seeds:
            order = np.random.RandomState(seed).permutation(len(ids))
            train = with_median_labels([source[ids[j]] for j in order[: len(ids) // 2]])
            v = prr_continuous([e["score"] for e in target.values()], predict_fn(train, list(target.values())))
            if v is not None:
                vals.append(v)
        return (st.fmean(vals), st.stdev(vals) if len(vals) > 1 else 0.0) if vals else (float("nan"), float("nan"))
    base.ood_cell = ood_cell

    # raw rows and references use e["label"] as the score: give them the score
    cohorts = {name: load(Path(path)) for name, path in a.cohort}
    for eps in cohorts.values():
        for e in eps.values():
            e["label"] = e["score"]
    cols = list(cohorts)

    md = ["# PRR report — DeepSWE, submitted-only, pre-terminal", "",
          "## Evaluation protocol", "",
          "Same protocol as the ALFWorld report: Sections 1–4 use 5-fold out-of-fold evaluation (`numpy.random.RandomState(0)`, fold *i* = `order[i::5]`, one PRR on the pooled predictions); Section 7 averages OOD over five source-train → target-test holdouts (seeds 0–4, all parameters fitted on a random half of the source).",
          "", "**Scores are continuous.** Neither model resolves a DeepSWE task outright (binary reward 0/113 for both), so PRR uses the verifier's native `partial` score (passed / all hidden tests, F2P + P2P) without binarisation, as the OSWorld/WebArena report does: the oracle ranks by the true score, the random reference is the mean score. Bayes and regression need a binary training label; every training fold is split at its own median score.",
          "", "Cohort: 113 DeepSWE tasks, mini-swe-agent, 200-command budget, working-tree grading; **submitted-only** (the agent issued the submit command itself; context-window deaths and budget exhaustion excluded) and **pre-terminal** (scored before the submit command). Signals per command: Logprob = mean token log-probability of the generation, Perplexity = exp(−Logprob), MTE = mean token entropy (top-20). Verbalised confidence was requested in the system prompt and ignored by both models (Qwen 0/1806 commands, gpt-oss 49/6365), so no Verb rows. Tool critics per command: no format error, ran tests, committed, no command repeated ≥3×, and two episode-level test critics repeated per step: a test that failed then passed, last test run passed. Tool success rate = share of commands with return code 0. **Bayes tool-only** is the tempered critic posterior.",
          "", "**Two things that look like copy-paste errors and are not.** (1) Perplexity rows repeat the Logprob rows wherever the method is rank-based (raw `last`, and every binarised Bayes variant): on DeepSWE Perplexity is exp(−Logprob) per command, a monotone transform, and PRR only sees ranks; they differ only where the value enters a Gaussian (Continuous / Tempered / Last only) or a mean over steps. (2) UQ-only Continuous (λ=1) and Tempered (λ=0.25) coincide: λ rescales the summed evidence, which does not change the ranking; they separate only once fused with the tool posterior.",
          "", "**Headline.** All values are low (best Avg ≈ .24 in-domain, ≈ .30 OOD, against .5–.9 on ALFWorld): the label is partial credit rather than success, the outcome is largely a task property, and the environment offers no progress signal (process critics saturate, failing test runs are how work is done, so the tool success rate is *negative*). Within that, Bayes UQ + tools (Double) is top-1 for every signal in Section 2, the tempered/multiplied tool-only posterior beats the regression and every raw baseline in Sections 3–4, and the fused binary variants (LR+ / SEP) transfer between the two models with a drop of ≈ .05.",
          "", "### Cohorts and folds", ""]
    for name, eps in cohorts.items():
        n = len(eps); md.append(f"- **{name}:** {n} submitted episodes; mean partial score {st.fmean(e['score'] for e in eps.values()):.3f}; median {st.median(e['score'] for e in eps.values()):.3f}. Per fold: train {n - math.ceil(n/5)}–{n - n//5}, test {n//5}–{math.ceil(n/5)}.")
    md += ["", "## 1. UQ baselines — last, mean, max", "", "Confidence is the aggregated raw value for Logprob and the negative of the aggregated raw value for Perplexity and MTE. No parameters are fitted.", ""]
    raw = base.raw_rows(cohorts)
    for signal in SIGNALS:
        md += [f"### {signal}", "", base.table({k: v for k, v in raw.items() if k[0] == signal}, cols), ""]
    md += ["## 2. Bayesian UQ — UQ-only and UQ + tools", "", "Five variants per signal, as in the ALFWorld report. UQ + tools starts from the tempered posterior of the six critics fitted on the same train fold.", ""]
    bay = base.bayes_rows(cohorts)
    for signal in SIGNALS:
        md += [f"### {signal}", "", base.table({k: v for k, v in bay.items() if k[0].startswith(signal + " —")}, cols), ""]
    ref = base.reference_rows(cohorts)
    ref = {k: v for k, v in ref.items() if not k[0].startswith("Verb")}
    ref[("Tool success rate", "share of commands with return code 0")] = ref.pop(("Tool success rate", "mean of step tool critics"))
    md += ["## 3. Reference methods", "", base.table(ref, cols), "", "N steps uses −N (shorter ranks higher); N counts commands before the submit.", ""]
    if a.toolkit:
        reg, _ = base.regression_rows(cohorts, a.toolkit)
        md += ["## 4. Logistic regression — outer cross-fitting", "", "The unchanged main implementation (`trajectory_uq_toolkit.regression`) fitted independently in each outer train fold on the three signals' aggregations and the six critics' pass shares; the outer training label is the fold's median split.", "", base.table(reg, cols), ""]
    md += ["## 7. OOD Bayes fused sweep — five-split means", "", "Two directions: fit on a random half of one model's cohort, score the other model's whole cohort. Eight fused variants per signal on top of the tempered tool posterior. Cells are mean PRR ± sample SD over seeds 0–4.", ""]
    methods = {k: fn for k, fn in base.ood_methods(a.toolkit).items() if not k[0].startswith("Verb")}
    names = cols; dirs = [f"{names[0]}>{names[1]}", f"{names[1]}>{names[0]}"]
    cells = {k: [] for k in methods}
    for d in dirs:
        src, tgt = d.split(">")
        for k, fn in methods.items():
            cells[k].append(base.ood_cell(cohorts[src], cohorts[tgt], fn))
    md += ["### Directions 1–2 — Same agent, different model", "", base.ood_table(cells, [d.replace(">", " → ") for d in dirs]), ""]
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text("\n".join(md)); print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
