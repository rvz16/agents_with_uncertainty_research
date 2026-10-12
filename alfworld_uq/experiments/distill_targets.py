"""Export our calibrated posterior per prefix, as a target to distil into.

Our best-calibrated estimator is the tempered Bayesian update over the tool
critics, at an expected calibration error of .041 -- better than any released
decision head we measured and better than our own fused variants. It has one
drawback: it reads critic observations, which are harness-specific, so it does
not travel to a new agent.

A model that reproduces this posterior from the trajectory text alone would
inherit the calibration and lose the dependence. That is what these targets are
for. Each prefix gets the posterior our layer assigns it, computed out of fold
so no episode contributes to the belief that scores it, alongside the binary
outcome and the mechanical structural labels, so a trainer can weigh a
calibrated soft target against a verified hard one.

The posterior is written per prefix rather than per episode because the point of
distillation here is the whole curve: how the belief moves as evidence arrives
is the part a single final number throws away.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from belief.continuous_bayes import ContinuousBayesUQ
from belief.critic_bayes import CriticBayesState
from experiments.prr_report_v2 import SEEDS, load_alfworld, stratified_folds
from experiments.schema_head import structural_labels

RUNS = {"RG": "react_gptoss_140_giveup_sc_finished_nolast", "SG": "smol_gptoss_140_sc_finished_nolast",
        "RQ": "react_qwen_140_giveup_sc_finished_nolast", "SQ": "smol_qwen_140_sc_finished_nolast"}


def posteriors(episodes: dict[str, dict], signal: str | None, seed: int,
               lambda_: float) -> dict[str, list[float]]:
    """Per-prefix belief: the tempered tool posterior, then the UQ update."""
    out: dict[str, list[float]] = {}
    for fold in stratified_folds(episodes, seed):
        held = set(fold)
        train = [episodes[i] for i in episodes if i not in held]
        tools = CriticBayesState.fit([c for e in train for c in e["critics"]],
                                     [e["label"] for e in train for _ in e["critics"]],
                                     prior=st.fmean(e["label"] for e in train))
        uq = None
        if signal:
            sequences = [s for e in train if (s := [v for v in e["signals"].get(signal, []) if v is not None])]
            labels = [e["label"] for e in train if e["signals"].get(signal)]
            if sequences and len(set(labels)) > 1:
                uq = ContinuousBayesUQ.fit(sequences, labels, lambda_=lambda_)
        for i in fold:
            episode = episodes[i]
            values = episode["signals"].get(signal) or [] if signal else []
            beliefs = []
            for k in range(episode["n_steps"]):
                belief = tools.predict_sequence_tempered(episode["critics"][: k + 1])
                if uq is not None:
                    for v in values[: k + 1]:
                        belief = uq.update(belief, v)
                beliefs.append(float(belief))
            out[i] = beliefs
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--views", type=Path, default=Path("data/judge_views"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--signal", default=None,
                   help="fold this UQ signal into the posterior as well; omit for tool-only, "
                        "which is the best-calibrated variant we measured")
    p.add_argument("--lambda", dest="lambda_", type=float, default=0.25)
    p.add_argument("--step-rule", default="giveup+acted")
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    for cohort, run in RUNS.items():
        episodes = load_alfworld(a.runs / run, "react" if "react" in run else "smolagents", a.step_rule)
        episodes = {k: v for k, v in episodes.items() if v["n_steps"]}
        structural = {}
        view_path = a.views / f"{cohort}.jsonl"
        if view_path.exists():
            for line in open(view_path):
                if line.strip():
                    view = json.loads(line)
                    structural[view["id"]] = [structural_labels(view["steps"][: k + 1])
                                              for k in range(len(view["steps"]))]

        per_seed = [posteriors(episodes, a.signal, seed, a.lambda_) for seed in SEEDS]
        path = a.out / f"targets_{cohort}.jsonl"
        with open(path, "w") as handle:
            for key, episode in episodes.items():
                curves = [s[key] for s in per_seed if key in s]
                if not curves:
                    continue
                facts = structural.get(key) or []
                for k in range(episode["n_steps"]):
                    repeat, stale = facts[k] if k < len(facts) else (None, None)
                    handle.write(json.dumps({
                        "id": key, "cohort": cohort, "step": k,
                        "posterior": st.fmean(c[k] for c in curves if k < len(c)),
                        "label": episode["label"], "score": episode["score"],
                        "repeat_share": repeat, "stale_share": stale}) + "\n")
        rows = sum(1 for _ in open(path))
        print(f"[distill] {cohort}: {len(episodes)} episodes -> {rows} prefix targets  {path}",
              flush=True)


if __name__ == "__main__":
    main()
