"""The released model's own gate policy, against one we fit.

Everywhere else we read a probability out of a decision model and fit thresholds
around it, which is our construction, not theirs. These models answer a
``choice`` question natively: name the action. So here the model is simply asked
what the harness should do -- continue, retry, ask a human, or stop -- and the
harness does what it says. No thresholds, no labels, no fitting.

That makes two policies comparable on identical prefixes under identical
accounting: the model's own decision, and the threshold policy of
``gate_utility`` fitted on our folds. The references are the same too -- never
gating, escalating everything at step 0, and an oracle that knows the outcome --
so the whole table reads in one currency, the realised utility of an episode.

``retry`` is counted as continuing. The agent cannot be made to restart from a
recorded trajectory, so the honest reading of that answer is that the episode
runs on; counting it as a stop would credit the policy with an intervention the
data cannot show the effect of.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from experiments.prr_report_v2 import load_alfworld

RUNS = {"RG": "react_gptoss_140_giveup_sc_finished_nolast", "SG": "smol_gptoss_140_sc_finished_nolast",
        "RQ": "react_qwen_140_giveup_sc_finished_nolast", "SQ": "smol_qwen_140_sc_finished_nolast"}
ACTS = ("continue", "retry", "ask", "stop")


def read_choices(path: Path) -> dict[str, list[tuple[int, str, dict]]]:
    out: dict[str, list[tuple[int, str, dict]]] = defaultdict(list)
    if not path.exists():
        return out
    for line in open(path):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("choice") is None:
            continue
        shares = {k[2:]: v for k, v in row.items() if k.startswith("p_")}
        out[row["id"]].append((int(row["step"]), str(row["choice"]), shares))
    return {k: sorted(v) for k, v in out.items()}


def utility(episode: dict, decisions: list[tuple[int, str, dict]], c_step: float,
            c_ask: float, threshold: float | None = None) -> tuple[float, str]:
    """Follow the first decision that is not 'continue'; retry counts as continue."""
    for k, choice, shares in decisions:
        if threshold is not None:
            # the fitted variant: act only when the model is confident enough
            if choice in ("ask", "stop") and shares.get(choice, 0.0) < threshold:
                continue
        if choice == "stop":
            return -c_step * (k + 1), "stop"
        if choice == "ask":
            return 1.0 - c_step * (k + 1) - c_ask, "ask"
    return episode["label"] - c_step * episode["n_steps"], "run"


def references(episodes: list[dict], c_step: float, c_ask: float) -> tuple[float, float, float]:
    never = st.fmean(e["label"] - c_step * e["n_steps"] for e in episodes)
    always = 1.0 - c_step - c_ask
    oracle = st.fmean(max(e["label"] - c_step * e["n_steps"], -c_step, 1.0 - c_step - c_ask)
                      for e in episodes)
    return never, always, oracle


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--choices", type=Path, required=True, help="directory of gate_<cohort>.jsonl")
    p.add_argument("--stem", default="gate")
    p.add_argument("--step-rule", default="giveup+acted")
    p.add_argument("--c-step", type=float, default=0.02)
    p.add_argument("--c-ask", type=float, default=0.5)
    p.add_argument("--thresholds", type=float, nargs="*", default=[0.5, 0.7, 0.9])
    a = p.parse_args()

    order = list(RUNS)
    rows: dict[str, dict[str, float]] = defaultdict(dict)
    spread: dict[str, Counter] = {}
    refs = {}
    for cohort, run in RUNS.items():
        episodes = load_alfworld(a.runs / run, "react" if "react" in run else "smolagents", a.step_rule)
        episodes = {k: v for k, v in episodes.items() if v["n_steps"]}
        choices = read_choices(a.choices / f"{a.stem}_{cohort}.jsonl")
        if not choices:
            print(f"[gate-choice] {cohort}: no verdicts, skipped", flush=True)
            continue
        refs[cohort] = references(list(episodes.values()), a.c_step, a.c_ask)
        spread[cohort] = Counter(c for v in choices.values() for _, c, _ in v)

        for label, threshold in [("as answered", None)] + [(f"only when p>={t:g}", t)
                                                           for t in a.thresholds]:
            values, taken = [], Counter()
            for key, episode in episodes.items():
                value, action = utility(episode, choices.get(key) or [], a.c_step, a.c_ask, threshold)
                values.append(value); taken[action] += 1
            rows[label][cohort] = st.fmean(values)
            rows[f"{label} · fired"][cohort] = 1.0 - taken["run"] / max(len(episodes), 1)

    print(f"\ncost of a step {a.c_step}, cost of escalation {a.c_ask}\n")
    print("what the model answers, over all prefixes:")
    for cohort in order:
        if cohort in spread:
            total = sum(spread[cohort].values())
            share = "  ".join(f"{act} {spread[cohort][act] / total * 100:4.1f}%" for act in ACTS)
            print(f"  {cohort}: {total:5d} prefixes   {share}")

    print(f"\n{'policy':28} " + " ".join(f"{c:>14}" for c in order) + f" {'Avg gap':>8}")
    print("-" * 96)

    def line(name: str, values: dict[str, float]) -> None:
        cells, gaps = [], []
        for cohort in order:
            if cohort not in values or cohort not in refs:
                cells.append(f"{'--':>14}"); continue
            never, _, oracle = refs[cohort]
            gap = (values[cohort] - never) / (oracle - never) if oracle > never else float("nan")
            gaps.append(gap)
            cells.append(f"{values[cohort]:+6.3f} {gap * 100:5.0f}%")
        avg = f"{st.fmean(gaps) * 100:7.0f}%" if gaps else "      --"
        print(f"{name:28} " + " ".join(cells) + f" {avg}")

    line("never gate", {c: refs[c][0] for c in refs})
    line("escalate everything at 0", {c: refs[c][1] for c in refs})
    line("oracle", {c: refs[c][2] for c in refs})
    print()
    for label in [k for k in rows if "fired" not in k]:
        line(label, rows[label])
        fired = rows.get(f"{label} · fired") or {}
        if fired:
            print(f"{'':28} " + " ".join(f"{fired.get(c, float('nan')) * 100:13.0f}%" for c in order)
                  + "   episodes acted on")


if __name__ == "__main__":
    main()
