"""What a decision model is actually for: the in-loop gate.

PRR asks whether a signal ranks finished episodes. That is our question, not
a System One model's: these models are sold for a decision taken while the
agent is still running -- keep going, escalate, or abandon -- with a
calibrated threshold as the whole point. So this measures the thing they are
built for.

At every step the controller sees the prefix and one probability of eventual
success. Two thresholds define the policy: at or below ``tau_stop`` abandon the
episode, at or below ``tau_ask`` hand it to a human, otherwise continue. The
realised utility of an episode is the reward actually collected minus the steps
actually paid for,

    continue to the end   label - c_step * n
    abandon at step k     -c_step * (k + 1)
    escalate at step k    1 - c_step * (k + 1) - c_ask

which is the realised-utility accounting of the POMDP sections, not a proxy.
Thresholds are fitted on each fold's training episodes and applied to the held
out ones, over the same three seeds and five folds as every other table, so a
policy cannot tune itself on the episodes it is scored on.

Three references bound the result. Never gating is the agent as it ran; always
escalating at step 0 is the cost of distrusting it entirely; the oracle knows
the label and picks the best of the three actions per episode. A policy is
reported both as its utility and as the share of the oracle's margin over
never gating that it captures, which is the only number comparable across
cohorts with base rates from .34 to .90.

The off-the-shelf row is the point of the exercise: the same policy with the
thresholds the model cards quote (0.1 and 0.5) and nothing fitted. A decision
model that is calibrated in the advertised sense should not need our folds.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from belief.continuous_bayes import ContinuousBayesUQ
from belief.critic_bayes import CriticBayesState
from experiments.prr_report_v2 import SEEDS, load_alfworld, stratified_folds

RUNS = {"RG": "react_gptoss_140_giveup_sc_finished_nolast", "SG": "smol_gptoss_140_sc_finished_nolast",
        "RQ": "react_qwen_140_giveup_sc_finished_nolast", "SQ": "smol_qwen_140_sc_finished_nolast"}
#: label -> (file stem, field, orientation: +1 if higher means more likely to succeed)
EXTERNAL = {
    "Decision p(success)": ("judge_Cloudflare_clef-flash", "p", +1),
    "Decision 1-stuck": ("judge_Cloudflare_clef-flash", "stuck", -1),
    "Decision 1-stale": ("judge_Cloudflare_clef-flash", "stale", -1),
    "Decision progress": ("judge_Cloudflare_clef-flash", "progress", +1),
    "Encoder p(success)": ("steps", "p", +1),
}
GRID = (-1.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9)
OFF_THE_SHELF = (0.1, 0.5)


def read_steps(path: Path, field: str, sign: int) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = defaultdict(dict)
    if not path.exists():
        return out
    for line in open(path):
        if not line.strip():
            continue
        row = json.loads(line)
        value = row.get(field)
        if value is not None:
            out[row["id"]][int(row["step"])] = float(value) if sign > 0 else 1.0 - float(value)
    return out


def decide(episode: dict, sequence: list[tuple[int, float]], tau_stop: float,
           tau_ask: float, c_step: float, c_ask: float) -> tuple[float, str]:
    """The realised utility of the threshold policy on one episode.

    ``sequence`` carries the step each probability belongs to, so a signal with
    gaps still pays for the steps the agent really took before the gate fired.
    An episode with no probability at all is never gated.
    """
    for k, p in sequence:
        if p <= tau_stop:
            return -c_step * (k + 1), "stop"
        if p <= tau_ask:
            return 1.0 - c_step * (k + 1) - c_ask, "ask"
    return episode["label"] - c_step * episode["n_steps"], "run"


def utility(episode: dict, sequence: list[tuple[int, float]], tau_stop: float, tau_ask: float,
            c_step: float, c_ask: float) -> float:
    return decide(episode, sequence, tau_stop, tau_ask, c_step, c_ask)[0]


def references(episodes: list[dict], c_step: float, c_ask: float,
               ask: bool = True) -> tuple[float, float, float]:
    never = st.fmean(e["label"] - c_step * e["n_steps"] for e in episodes)
    always = 1.0 - c_step - c_ask if ask else float("nan")
    choices = (lambda e: (e["label"] - c_step * e["n_steps"], -c_step, 1.0 - c_step - c_ask)) if ask \
        else (lambda e: (e["label"] - c_step * e["n_steps"], -c_step))
    oracle = st.fmean(max(choices(e)) for e in episodes)
    return never, always, oracle


def best_thresholds(train: list[dict], sequences: dict[str, list[tuple[int, float]]],
                    c_step: float, c_ask: float, ask: bool = True) -> tuple[float, float]:
    best, choice = None, (-1.0, -1.0)
    for tau_stop in GRID:
        for tau_ask in (GRID if ask else (-1.0,)):
            if ask and tau_ask < tau_stop:
                continue
            value = st.fmean(utility(e, sequences[e["id"]], tau_stop, tau_ask, c_step, c_ask)
                             for e in train)
            if best is None or value > best:
                best, choice = value, (tau_stop, tau_ask)
    return choice


def our_sequences(episodes: dict[str, dict], signal: str | None,
                  seed: int) -> dict[str, list[tuple[int, float]]]:
    """Out-of-fold per-step posteriors: the tool belief over the prefix, then the
    UQ update over the prefix's signal values -- the tempered variant of the
    tables, evaluated at every step instead of only the last."""
    out: dict[str, list[tuple[int, float]]] = {}
    for fold in stratified_folds(episodes, seed):
        held = set(fold)
        train = [episodes[i] for i in episodes if i not in held]
        tools = CriticBayesState.fit([c for e in train for c in e["critics"]],
                                     [e["label"] for e in train for _ in e["critics"]],
                                     prior=st.fmean(e["label"] for e in train))
        uq = None
        if signal is not None:
            seqs = [s for e in train if (s := [v for v in e["aligned"][signal] if v is not None])]
            labs = [e["label"] for e in train if any(v is not None for v in e["aligned"][signal])]
            if seqs and len(set(labs)) > 1:
                uq = ContinuousBayesUQ.fit(seqs, labs, lambda_=0.25)
        for i in fold:
            e = episodes[i]
            values = e["aligned"][signal] if signal is not None else [None] * e["n_steps"]
            beliefs: list[tuple[int, float]] = []
            prefix: list[float] = []
            for k in range(e["n_steps"]):
                belief = tools.predict_sequence_tempered(e["critics"][: k + 1])
                if values[k] is not None:
                    prefix.append(values[k])
                if uq is not None:
                    for v in prefix:
                        belief = uq.update(belief, v)
                beliefs.append((k, belief))
            out[i] = beliefs
    return out


def external_sequences(episodes: dict[str, dict], label: str) -> dict[str, list[tuple[int, float]]]:
    return {i: [(k, v) for k, v in enumerate(e["external"][label]) if v is not None]
            for i, e in episodes.items()}


def fused_sequences(episodes: dict[str, dict], label: str, seed: int) -> dict[str, list[float]]:
    """The same machinery with an external per-step answer as the signal."""
    for e in episodes.values():
        e["aligned"][label] = e["external"].get(label, [None] * e["n_steps"])
    return our_sequences(episodes, label, seed)


def evaluate(episodes: dict[str, dict], sequences_for, c_step: float, c_ask: float,
             fixed: tuple[float, float] | None = None,
             ask: bool = True) -> tuple[float, float, float] | None:
    """Mean out-of-fold utility over the seeds, or None if the signal is too sparse.

    The threshold is fitted on each fold's training episodes. For our own rows
    the sequences those training episodes carry come from a model fitted on a
    different split, so the threshold inherits the mild optimism of any
    two-stage out-of-fold stack; the rows whose sequences come from a model that
    never saw our data carry no such debt.
    """
    per_seed = []
    for seed in SEEDS:
        sequences = sequences_for(seed)
        if sum(bool(sequences.get(i)) for i in episodes) < 0.5 * len(episodes):
            return None
        sequences = {i: sequences.get(i) or [] for i in episodes}
        values, actions = {}, {}
        for fold in stratified_folds(episodes, seed):
            held = set(fold)
            train = [episodes[i] for i in episodes if i not in held]
            taus = fixed or best_thresholds(train, sequences, c_step, c_ask, ask)
            for i in fold:
                values[i], actions[i] = decide(episodes[i], sequences[i], *taus, c_step, c_ask)
        per_seed.append((st.fmean(values.values()),
                         st.fmean(a == "stop" for a in actions.values()),
                         st.fmean(a == "ask" for a in actions.values())))
    return tuple(st.fmean(column) for column in zip(*per_seed))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--judge-dir", type=Path, required=True)
    p.add_argument("--encoder-dir", type=Path, required=True)
    p.add_argument("--step-rule", default="giveup+acted")
    p.add_argument("--c-step", type=float, default=0.02)
    p.add_argument("--c-ask", type=float, default=0.5)
    p.add_argument("--no-ask", action="store_true",
                   help="drop the escalation action: the gate only chooses between continuing and\n"
                        "abandoning, which is the setting where a signal has to discriminate rather\n"
                        "than fire on everything")
    a = p.parse_args()

    cohorts: dict[str, dict[str, dict]] = {}
    for cohort, run in RUNS.items():
        episodes = load_alfworld(a.runs / run, "react" if "react" in run else "smolagents", a.step_rule)
        episodes = {k: v for k, v in episodes.items() if v["n_steps"]}
        for label, (stem, field, sign) in EXTERNAL.items():
            directory = a.encoder_dir if stem == "steps" else a.judge_dir
            per_episode = read_steps(directory / f"{stem}_{cohort}.jsonl", field, sign)
            for key, e in episodes.items():
                by_step = per_episode.get(key) or {}
                e.setdefault("external", {})[label] = [by_step.get(k) for k in range(e["n_steps"])]
        for e in episodes.values():
            e["aligned"] = {"MTE": [None if v is None else float(v) for v in e["mte_steps"]],
                            "LLM judge": list(e["judge_steps"])}
        cohorts[cohort] = episodes
        print(f"[gate] {cohort}: {len(episodes)} episodes, base rate "
              f"{st.fmean(e['label'] for e in episodes.values()):.2f}, mean steps "
              f"{st.fmean(e['n_steps'] for e in episodes.values()):.1f}", flush=True)

    order = list(RUNS)
    ask = not a.no_ask
    refs = {c: references(list(cohorts[c].values()), a.c_step, a.c_ask, ask) for c in order}
    print(f"\ncost of a step {a.c_step}, "
          + (f"cost of escalation {a.c_ask}" if ask else "no escalation action") + "\n")
    print(f"{'policy':34} " + " ".join(f"{c:>13}" for c in order) + f" {'Avg gap':>8}")
    print("-" * 102)

    def row(name: str, utilities: dict[str, object]) -> None:
        cells, gaps, fired = [], [], []
        for c in order:
            value = utilities.get(c)
            if value is None:
                cells.append(f"{'--':>13}")
                continue
            u, stop, askr = value if isinstance(value, tuple) else (value, float("nan"), float("nan"))
            never, _, oracle = refs[c]
            gap = (u - never) / (oracle - never) if oracle > never else float("nan")
            gaps.append(gap)
            if stop == stop:
                fired.append((stop, askr))
            cells.append(f"{u:+6.3f} {gap * 100:5.0f}%")
        avg = f"{st.fmean(gaps) * 100:7.0f}%" if len(gaps) == len(order) else "      --"
        tail = ""
        if len(fired) == len(order):
            tail = (f"   stop {st.fmean(s for s, _ in fired) * 100:3.0f}%"
                    f" ask {st.fmean(a for _, a in fired) * 100:3.0f}%")
        print(f"{name:34} " + " ".join(cells) + f" {avg}{tail}")

    row("never gate (the agent as it ran)", {c: refs[c][0] for c in order})
    if ask:
        row("escalate everything at step 0", {c: refs[c][1] for c in order})
    row("oracle (knows the outcome)", {c: refs[c][2] for c in order})
    print()

    for label in EXTERNAL:
        row(f"{label}, fitted",
            {c: evaluate(cohorts[c], lambda s, c=c, l=label: external_sequences(cohorts[c], l),
                         a.c_step, a.c_ask, ask=ask) for c in order})
    print()
    for label in ("Decision p(success)", "Encoder p(success)"):
        row(f"{label}, off-the-shelf 0.1/0.5",
            {c: evaluate(cohorts[c], lambda s, c=c, l=label: external_sequences(cohorts[c], l),
                         a.c_step, a.c_ask, fixed=OFF_THE_SHELF if ask else (OFF_THE_SHELF[0], -1.0),
                         ask=ask) for c in order})
    print()
    row("ours: tools only, tempered",
        {c: evaluate(cohorts[c], lambda s, c=c: our_sequences(cohorts[c], None, s),
                     a.c_step, a.c_ask, ask=ask) for c in order})
    for signal in ("MTE", "LLM judge"):
        row(f"ours: Bayes {signal} + tools",
            {c: evaluate(cohorts[c], lambda s, c=c, g=signal: our_sequences(cohorts[c], g, s),
                         a.c_step, a.c_ask, ask=ask) for c in order})
    print()
    for label in ("Decision p(success)", "Decision 1-stuck", "Decision progress"):
        row(f"ours: Bayes [{label}] + tools",
            {c: evaluate(cohorts[c], lambda s, c=c, l=label: fused_sequences(cohorts[c], l, s),
                         a.c_step, a.c_ask, ask=ask) for c in order})


if __name__ == "__main__":
    main()
