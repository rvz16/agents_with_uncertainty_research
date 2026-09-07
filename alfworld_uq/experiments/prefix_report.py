"""Episode-level UQ scored where the outcome does not yet exist.

An ALFWorld episode ends the moment the agent wins and otherwise runs to the
step budget, so on ReAct every failure is exactly `max_steps` long and every
success is shorter: `num_steps < budget` alone scores AUROC 0.978 (gpt-oss) and
1.000 (Qwen). Anything that can see how many steps happened -- the LLM judge
included, since it reads the transcript -- is then recognising the outcome
rather than predicting it.

This report removes that channel by construction. It keeps only episodes with at
least `--prefix-steps` steps and reads only those first steps, so every episode
contributes a transcript of exactly the same length and no signal can encode the
ending. What survives is the honest question: at step N, before the episode is
decided, what did we know?

    python -m experiments.prefix_report --run runs/alfworld_baseline_140 \\
        --prefix-steps 10 --seeds 20
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

from belief.critic_bayes import CriticBayesState
from experiments.analyze_trajectories import (
    _critic_observations,
    _split_ids,
    metric_values,
)

SIGNALS = (
    "mean_token_logprob",
    "perplexity",
    "sum_logprob",
    "sequence_probability",
    "mean_token_entropy",
    "verbalized_confidence",
)
AGGREGATIONS = ("mean", "last", "min", "max")
# Lower is more confident for these two, so their sign is flipped to keep every
# score oriented the same way: larger means more likely to succeed.
DESCENDING = {"perplexity", "mean_token_entropy"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _aggregate(values: list[float], how: str) -> float:
    if not values:
        return 0.0
    return {
        "mean": st.fmean(values),
        "last": values[-1],
        "min": min(values),
        "max": max(values),
    }[how]


def _auroc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = sorted(zip(scores, labels))
    ranks: dict[int, float] = {}
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and order[stop + 1][0] == order[index][0]:
            stop += 1
        rank = (index + stop) / 2 + 1
        for position in range(index, stop + 1):
            ranks[position] = rank
        index = stop + 1
    positive_rank_sum = sum(
        ranks[position] for position, (_, label) in enumerate(order) if label == 1
    )
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def load_cohort(run: Path, prefix_steps: int) -> dict[str, Any]:
    episodes = {row["episode_id"]: row for row in _read_jsonl(run / "episodes.jsonl")}
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(run / "trajectories.jsonl"):
        by_episode[row["episode_id"]].append(row)

    ids, labels, prefixes = [], [], {}
    for episode_id, rows in by_episode.items():
        if episode_id not in episodes or len(rows) < prefix_steps:
            continue
        ids.append(episode_id)
        labels.append(int(bool(episodes[episode_id]["final_success"])))
        prefixes[episode_id] = sorted(rows, key=lambda row: int(row["step"]))[
            :prefix_steps
        ]
    return {"ids": ids, "labels": labels, "prefixes": prefixes, "episodes": episodes}


def _signal_scores(
    cohort: dict[str, Any], signal: str, how: str, target: str
) -> list[float]:
    scores = []
    for episode_id in cohort["ids"]:
        values = [
            float(value)
            for row in cohort["prefixes"][episode_id]
            if (value := (row.get("uq") or {}).get(target, {}).get(signal)) is not None
        ]
        score = _aggregate(values, how)
        scores.append(-score if signal in DESCENDING else score)
    return scores


def _belief_over_splits(cohort: dict[str, Any], seeds: int, fraction: float) -> dict[str, Any]:
    """Critic-Bayes fitted on one half of the cohort, scored on the other."""
    observations = {
        episode_id: _critic_observations(cohort["prefixes"][episode_id])
        for episode_id in cohort["ids"]
    }
    label_of = dict(zip(cohort["ids"], cohort["labels"]))
    per_seed = []
    for seed in range(seeds):
        calibration, test = _split_ids(cohort["ids"], fraction, seed)
        state = CriticBayesState.fit(
            [observations[i] for i in calibration], [label_of[i] for i in calibration]
        )
        probabilities = [state.predict(observations[i]) for i in test]
        per_seed.append(metric_values([label_of[i] for i in test], probabilities))
    return {
        metric: (
            st.fmean([row[metric] for row in per_seed if row.get(metric) is not None]),
            st.stdev([row[metric] for row in per_seed if row.get(metric) is not None])
            if seeds > 1
            else 0.0,
        )
        for metric in ("auroc", "prr_at_0_5", "brier")
        if any(row.get(metric) is not None for row in per_seed)
    }


def _judge_scores(run: Path, cohort: dict[str, Any], filename: str) -> list[float] | None:
    path = run / filename
    if not path.exists():
        return None
    scored = {
        row["episode_id"]: row
        for row in _read_jsonl(path)
        if row.get("status") == "ok"
    }
    if not all(episode_id in scored for episode_id in cohort["ids"]):
        return None
    out = []
    for episode_id in cohort["ids"]:
        row = scored[episode_id]
        confidence = float(row.get("confidence") or 0.5)
        out.append(confidence if row["judge_pass"] else 1.0 - confidence)
    return out


def report(run: Path, prefix_steps: int, seeds: int, fraction: float) -> dict[str, Any]:
    cohort = load_cohort(run, prefix_steps)
    labels = cohort["labels"]
    lengths = [-int(cohort["episodes"][i]["num_steps"]) for i in cohort["ids"]]

    best = None
    per_signal: dict[str, dict[str, Any]] = {}
    for target in ("combined", "thought"):
        for signal in SIGNALS:
            for how in AGGREGATIONS:
                scores = _signal_scores(cohort, signal, how, target)
                if not any(scores):
                    continue  # the run never recorded this signal
                value = _auroc(labels, scores)
                if value is None:
                    continue
                if best is None or value > best["auroc"]:
                    best = {
                        "auroc": value,
                        "signal": f"{target}/{signal}",
                        "aggregation": how,
                    }
                seen = per_signal.get(signal)
                if seen is None or value > seen["auroc"]:
                    per_signal[signal] = {
                        "auroc": value,
                        "target": target,
                        "aggregation": how,
                    }

    belief = _belief_over_splits(cohort, seeds, fraction)
    judge_full = _judge_scores(run, cohort, "llm_judge_scores.jsonl")
    judge_prefix = _judge_scores(run, cohort, f"llm_judge_prefix{prefix_steps}.jsonl")
    return {
        "run": run.name,
        "episodes_in_cohort": len(labels),
        "successes": sum(labels),
        "prefix_steps": prefix_steps,
        "length_artefact_auroc": _auroc(labels, lengths),
        "best_prefix_signal": best,
        "per_signal_prefix": {
            name: f"{row['auroc']:.3f} ({row['target']}/{row['aggregation']})"
            for name, row in sorted(per_signal.items())
        },
        "belief_prefix": {
            name: f"{mean:.3f} ± {sd:.3f}" for name, (mean, sd) in belief.items()
        },
        "judge_full_transcript_auroc": _auroc(labels, judge_full) if judge_full else None,
        "judge_prefix_auroc": _auroc(labels, judge_prefix) if judge_prefix else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--prefix-steps", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reports = [
        report(run, args.prefix_steps, args.seeds, args.calibration_fraction)
        for run in args.run
    ]
    print(json.dumps(reports, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
