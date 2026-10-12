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

Which estimator teaches was decided by measuring all four rather than by
argument. Averaged over the cohorts: the temporal belief model reaches .64 PRR
at .077 expected calibration error, the judge fusion .65 at .107, the tempered
tool posterior .47 at .046, mean token entropy .46 at .053. The tool posterior
is the best calibrated and the weakest at ranking, and distilling calibration
without discrimination teaches a student the base rate, so the temporal belief
model is the default: it buys .17 of ranking for .03 of calibration. ``--compare``
reproduces that table.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from belief.continuous_bayes import ContinuousBayesUQ
from belief.critic_bayes import CriticBayesState
from experiments.decision_model_score import apply_temperature, ece, fit_temperature
from experiments.deepswe_table import prr_continuous
from experiments.prr_report_v2 import SEEDS, SIGNALS, load_alfworld, method_table, stratified_folds, with_labels
from experiments.schema_head import structural_labels

#: label -> the method-table key that produces it, for the teachers that are not
#: the tool posterior. Each is evaluated per prefix by truncating the record.
TEACHERS = {
    "b4": ("Reference", "TemporalBelief (B4)", "final checkpoint"),
    "judge": ("LLM judge", "LLM judge \\ensuremath{-} Bayes UQ + tools", "Last only"),
    "mte": ("MTE", "MTE \\ensuremath{-} Bayes UQ + tools", "Last only"),
}

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


def prefix_records(episode: dict, cut: int) -> dict:
    """The episode as it looked after ``cut`` steps, for a method that reads a record."""
    record = dict(episode["record"])
    record["generations"] = record["generations"][:cut]
    return dict(episode, record=record, n_steps=cut,
                signals={k: list(v)[:cut] for k, v in episode["signals"].items()},
                critics=episode["critics"][:cut], judge_steps=episode["judge_steps"][:cut],
                mte_steps=episode["mte_steps"][:cut], saup=episode["saup"][:cut],
                htc_tokens=tuple(side[:cut] for side in episode["htc_tokens"]))


def teacher_curves(episodes: dict[str, dict], key, methods, seed: int) -> dict[str, list[float]]:
    """A method-table teacher, evaluated at every prefix, out of fold."""
    fn = methods[key]
    out: dict[str, list[float]] = {}
    for fold in stratified_folds(episodes, seed):
        held = set(fold)
        train = with_labels([episodes[i] for i in episodes if i not in held], False)
        for i in fold:
            episode = episodes[i]
            cuts = [prefix_records(episode, k + 1) for k in range(episode["n_steps"])]
            values = fn(train, cuts)
            out[i] = [float(v) if v is not None else float("nan") for v in values]
    return out


def recalibrate(curves: dict[str, list[float]], episodes: dict[str, dict],
                seed: int) -> dict[str, list[float]]:
    """One temperature, fitted out of fold on the last step and applied to the curve.

    A teacher that ranks well but is overconfident -- our fused variants sit at
    .20 expected calibration error -- would hand the student its scale along with
    its ranking. Fitting the map on the training folds only keeps the target
    honest about the episodes it scores.
    """
    out = dict(curves)
    for fold in stratified_folds(episodes, seed):
        held = set(fold)
        train = [i for i in episodes if i not in held and curves.get(i)]
        if not train:
            continue
        temperature = fit_temperature([episodes[i]["label"] for i in train],
                                      [curves[i][-1] for i in train])
        for i in fold:
            if curves.get(i):
                out[i] = apply_temperature(curves[i], temperature)
    return out


def report(name: str, cohort: str, curves: dict[str, list[float]], episodes: dict[str, dict]) -> None:
    last = [(episodes[i]["score"], c[-1]) for i, c in curves.items() if c]
    if len(last) < 5:
        return
    y = [s for s, _ in last]; p = [v for _, v in last]
    print(f"[distill] {cohort} teacher {name:10} PRR {prr_continuous(y, p):+.2f}  "
          f"ECE {ece(y, p):.3f}  mean {st.fmean(p):.2f} vs base {st.fmean(y):.2f}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--views", type=Path, default=Path("data/judge_views"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--signal", default=None,
                   help="fold this UQ signal into the posterior as well; omit for tool-only, "
                        "which is the best-calibrated variant we measured")
    p.add_argument("--lambda", dest="lambda_", type=float, default=0.25)
    p.add_argument("--teacher", default="b4", choices=["tools", *TEACHERS],
                   help="measured over the four cohorts: b4 .64 PRR at .077 ECE, judge .65 at .107, "
                        "tools .47 at .046, mte .46 at .053. b4 is the default because it buys "
                        ".17 of ranking for .03 of calibration")
    p.add_argument("--toolkit", default=None, help="required for the b4 and fused teachers")
    p.add_argument("--recalibrate", action="store_true",
                   help="off by default because it was measured and does not help: these posteriors "
                        "are already near-calibrated, and one temperature fitted on the last step "
                        "distorts the early ones (tools on SG .048 -> .106, judge on RQ .097 -> .167)")
    p.add_argument("--compare", action="store_true", help="report every teacher and write none")
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

        methods = method_table(a.toolkit) if (a.toolkit and (a.compare or a.teacher != "tools")) else {}
        wanted = list(TEACHERS) if a.compare else ([] if a.teacher == "tools" else [a.teacher])

        def curves_for(name: str, seed: int) -> dict[str, list[float]]:
            if name == "tools":
                return posteriors(episodes, a.signal, seed, a.lambda_)
            return teacher_curves(episodes, TEACHERS[name], methods, seed)

        if a.compare:
            for name in ("tools", *[w for w in wanted if TEACHERS[w] in methods]):
                curves = curves_for(name, SEEDS[0])
                report(name, cohort, curves, episodes)
                report(f"{name}+temp", cohort, recalibrate(curves, episodes, SEEDS[0]), episodes)
            continue

        per_seed = [curves_for(a.teacher, seed) for seed in SEEDS]
        if a.recalibrate:
            per_seed = [recalibrate(c, episodes, seed) for c, seed in zip(per_seed, SEEDS)]
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
