"""Bring the external per-step judgements inside the fusion.

The decision models were measured beside our estimators; this puts their
per-step answers through the same machinery as mean token entropy or the prefix
judge -- raw aggregation, the Bayesian update on top of the tool posterior, and
the history-and-last fusion -- so the comparison is no longer baseline against
baseline but signal against signal.

Four sequences come from a decision judge (the probability of success and the
three structural answers) and one from the fine-tuned encoder. The structural
ones are the point: whether the agent is stuck and whether its last check is
stale are exactly the conditional dependencies an additive log-likelihood ratio
over token statistics cannot express.

One caveat is recorded rather than hidden. The encoder's per-step probability is
out of fold with respect to the episode it scores, but the model that produced
it saw episodes that later serve as the fusion's test fold, so its fused rows
are optimistic in the way any stacked model is without nested cross-validation.
The decision judges are trained on none of our data and carry no such debt.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from experiments.prr_report_v2 import (SEEDS, SIGNALS, load_alfworld, method_table, oof, prr)

RUNS = {"RG": "react_gptoss_140_giveup_sc_finished_nolast", "SG": "smol_gptoss_140_sc_finished_nolast",
        "RQ": "react_qwen_140_giveup_sc_finished_nolast", "SQ": "smol_qwen_140_sc_finished_nolast"}
#: label -> (field, higher value means more uncertain); the judge's file stem is
#: the model under test, so a second decision model needs no edit here
JUDGE_FIELDS = {"Decision p(success)": ("p", False), "Decision stuck": ("stuck", True),
                "Decision stale": ("stale", True), "Decision progress": ("progress", False)}
ENCODER = {"Encoder p(success)": ("steps", "p", False)}


def external_for(model: str) -> dict[str, tuple[str, str, bool]]:
    stem = f"judge_{model}"
    return {label: (stem, field, hiu) for label, (field, hiu) in JUDGE_FIELDS.items()} | ENCODER


def read_steps(path: Path, field: str) -> dict[str, list[tuple[int, float]]]:
    out: dict[str, list[tuple[int, float]]] = defaultdict(list)
    if not path.exists():
        return out
    for line in open(path):
        if not line.strip():
            continue
        row = json.loads(line)
        value = row.get(field)
        if value is not None:
            out[row["id"]].append((int(row["step"]), float(value)))
    return {k: [v for _, v in sorted(pairs)] for k, pairs in out.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True, help="ALFWorld run directories live here")
    p.add_argument("--judge-dir", type=Path, required=True, help="per-step verdicts from prefix_decision")
    p.add_argument("--encoder-dir", type=Path, required=True, help="steps_*.jsonl from finetune_outcome")
    p.add_argument("--toolkit", required=True)
    p.add_argument("--step-rule", default="giveup+acted")
    p.add_argument("--judge-model", default="Cloudflare_clef-flash",
                   help="the per-step verdict files are judge_<model>_<cohort>.jsonl")
    a = p.parse_args()

    external = external_for(a.judge_model)
    for label, (_, _, hiu) in external.items():
        SIGNALS[label] = (label, None, hiu)  # the key is unused: the sequence is injected directly
    methods = method_table(a.toolkit)

    variants = [("raw, last", "last"), ("raw, mean", "mean")]
    families = [("Bayes UQ + tools, Last only", "Last only"),
                ("Bayes UQ + tools, Tempered", "Tempered"),
                ("Bayes Fused H+L + tools", "History + Last")]

    results: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for cohort, run in RUNS.items():
        episodes = load_alfworld(a.runs / run, "react" if "react" in run else "smolagents", a.step_rule)
        episodes = {k: v for k, v in episodes.items() if v["n_steps"]}
        for label, (stem, field, _) in external.items():
            directory = a.encoder_dir if stem == "steps" else a.judge_dir
            sequences = read_steps(directory / f"{stem}_{cohort}.jsonl", field)
            missing = 0
            for key, episode in episodes.items():
                values = sequences.get(key) or []
                episode["signals"][label] = values[: episode["n_steps"]]
                missing += not values
            if missing:
                print(f"[fuse] {cohort} {label}: {missing} episodes without the signal", flush=True)

        for label in external:
            for name, aggregate in variants:
                key = (label, aggregate)
                fn = methods[(label, label, aggregate)]
                values = [v for s in SEEDS if (v := oof(episodes, fn, s)) is not None]
                if values:
                    results[(label, name)][cohort] = st.fmean(values)
            for name, mode in families:
                fn = methods[(label, f"{label} \\ensuremath{{-}} Bayes UQ + tools", mode)] if "Fused" not in name \
                    else methods[(label, f"{label} \\ensuremath{{-}} Bayes Fused H+L + tools", mode)]
                values = [v for s in SEEDS if (v := oof(episodes, fn, s)) is not None]
                if values:
                    results[(label, name)][cohort] = st.fmean(values)
        print(f"[fuse] {cohort} done", flush=True)

    order = list(RUNS)
    print(f"\n{'signal':24} {'aggregation':28} " + " ".join(f"{c:>6}" for c in order) + f" {'Avg':>7}")
    print("-" * 92)
    for label in external:
        for name, _ in variants + [(n, m) for n, m in families]:
            row = results.get((label, name))
            if not row or len(row) < len(order):
                continue
            cells = " ".join(f"{row[c]:+6.2f}" for c in order)
            print(f"{label:24} {name:28} {cells} {st.fmean(row[c] for c in order):+7.2f}")
        print()


if __name__ == "__main__":
    main()
