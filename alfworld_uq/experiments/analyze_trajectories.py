from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

MATPLOTLIB_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from belief.binary_bayes import BinaryBayesUQ, DoubleBinaryBayesUQ
from belief.continuous_bayes import ContinuousBayesUQ
from belief.critic_bayes import CriticBayesState
from uq.aggregation import aggregate_trajectory


UQ_METHODS = {
    "perplexity": True,
    "sum_logprob": False,
    "mean_token_logprob": False,
    "sequence_probability": False,
    "verbalized_confidence": False,
}
# `reasoning` exists only for locally served models that return log-probabilities
# for their hidden channel; it is empty for hosted endpoints and drops out then.
TARGETS = ("thought", "action", "combined", "reasoning")
AGGREGATIONS = (
    "last",
    "mean",
    "min",
    "max",
    "median",
    "ewma",
    "fraction_uncertain",
    "last_k_mean",
    "cvar",
)
PREFIXES = (0.25, 0.5, 0.75, 1.0)
EPS = 1e-6
EPISODE_CRITIC_NAMES = (
    "all_formats_valid",
    "all_actions_valid",
    "no_repeated_fallback",
)
STEP_CRITIC_NAMES = ("format_valid", "action_valid", "no_repeated_fallback")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _split_ids(
    episode_ids: list[str], calibration_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    if len(episode_ids) < 2:
        raise ValueError("Analysis requires at least two episodes")
    ordered = sorted(
        episode_ids,
        key=lambda episode_id: hashlib.sha256(
            f"{seed}:{episode_id}".encode("utf-8")
        ).hexdigest(),
    )
    calibration_size = min(
        len(ordered) - 1,
        max(1, round(len(ordered) * calibration_fraction)),
    )
    return ordered[:calibration_size], ordered[calibration_size:]


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _sequence(
    rows: list[dict[str, Any]], target: str, method: str
) -> list[float]:
    values = []
    for row in rows:
        value = row.get("uq", {}).get(target, {}).get(method)
        value = _finite(value)
        if value is not None:
            values.append(value)
    return values


def _critic_observations(rows: list[dict[str, Any]]) -> dict[str, bool]:
    """Summarize cheap pre-outcome checks once per episode."""
    return {
        "all_formats_valid": all(row.get("format_valid") is True for row in rows),
        "all_actions_valid": all(row.get("action_valid") is True for row in rows),
        "no_repeated_fallback": all(
            row.get("fallback_reason") != "repeated_action" for row in rows
        ),
    }


def _step_critic_observation(row: dict[str, Any]) -> dict[str, bool]:
    return {
        "format_valid": row.get("format_valid") is True,
        "action_valid": row.get("action_valid") is True,
        "no_repeated_fallback": row.get("fallback_reason") != "repeated_action",
    }


def _predict_from_belief(model: Any, sequence: list[float], belief: float) -> float:
    for value in sequence:
        belief = model.update(belief, value)
    return belief


def _prefix(sequence: list[float], fraction: float) -> list[float]:
    count = max(1, math.ceil(len(sequence) * fraction))
    return sequence[:count]


def _clip_probabilities(values: np.ndarray) -> np.ndarray:
    return np.clip(values, EPS, 1.0 - EPS)


def auroc(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return None
    comparisons = [
        float(pos > neg) + 0.5 * float(pos == neg)
        for pos in positives
        for neg in negatives
    ]
    return sum(comparisons) / len(comparisons)


def auprc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if positives == 0 or positives == len(labels):
        return None
    ranked = sorted(zip(scores, labels), reverse=True)
    true_positives = 0
    precisions = []
    for rank, (_, label) in enumerate(ranked, 1):
        true_positives += label
        if label:
            precisions.append(true_positives / rank)
    return sum(precisions) / positives


def prediction_rejection_area(
    uncertainty: list[float], labels: list[int], max_rejection: float
) -> float | None:
    """Area under retained accuracy while rejecting uncertain predictions."""
    pairs = [
        (float(value), int(label))
        for value, label in zip(uncertainty, labels)
        if value is not None and math.isfinite(float(value))
    ]
    if len(pairs) < 2:
        return None
    uncertainty_array = np.asarray([value for value, _ in pairs], dtype=float)
    outcomes = np.asarray([label for _, label in pairs], dtype=float)
    minimum, maximum = np.min(outcomes), np.max(outcomes)
    if np.isclose(minimum, maximum):
        minimum -= 1.0
        maximum += 1.0
    outcomes = (outcomes - minimum) / (maximum - minimum)
    ranked = outcomes[np.argsort(uncertainty_array)]
    rejected = int(max_rejection * len(ranked))
    if rejected <= 0:
        return None
    cumulative = np.cumsum(ranked)[-rejected:]
    retained = np.arange((len(ranked) - rejected) + 1, len(ranked) + 1)
    return float(np.sum((cumulative / retained)[::-1]) / rejected)


@lru_cache(maxsize=None)
def _prr_references(
    labels: tuple[int, ...], max_rejection: float
) -> tuple[float | None, float | None]:
    outcomes = list(labels)
    oracle = prediction_rejection_area(
        [-float(label) for label in outcomes], outcomes, max_rejection
    )
    rng = np.random.RandomState(42)
    random_order = np.arange(len(outcomes))
    random_areas = []
    for _ in range(1000):
        rng.shuffle(random_order)
        area = prediction_rejection_area(
            random_order.tolist(), outcomes, max_rejection
        )
        if area is not None:
            random_areas.append(area)
    random = float(np.mean(random_areas)) if random_areas else None
    return oracle, random


def prr(
    confidence: list[float], labels: list[int], max_rejection: float = 0.5
) -> float | None:
    """Prediction Rejection Ratio normalized by random and oracle rankings."""
    pairs = [
        (float(value), int(label))
        for value, label in zip(confidence, labels)
        if value is not None and math.isfinite(float(value))
    ]
    if len(pairs) < 2:
        return None
    filtered_confidence = [value for value, _ in pairs]
    filtered_labels = [label for _, label in pairs]
    area = prediction_rejection_area(
        [-value for value in filtered_confidence],
        filtered_labels,
        max_rejection,
    )
    oracle, random = _prr_references(tuple(filtered_labels), max_rejection)
    if (
        area is None
        or oracle is None
        or random is None
        or abs(oracle - random) < 1e-12
    ):
        return None
    return (area - random) / (oracle - random)


def ece(labels: list[int], probabilities: list[float], bins: int = 10) -> float:
    labels_array = np.asarray(labels, dtype=float)
    probabilities_array = np.asarray(probabilities, dtype=float)
    total = len(labels)
    result = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        mask = (probabilities_array >= lower) & (
            probabilities_array <= upper if upper >= 1.0 else probabilities_array < upper
        )
        if np.any(mask):
            result += (
                np.sum(mask)
                / total
                * abs(
                    float(np.mean(probabilities_array[mask]))
                    - float(np.mean(labels_array[mask]))
                )
            )
    return result


def metric_values(labels: list[int], probabilities: list[float]) -> dict[str, Any]:
    clipped = _clip_probabilities(np.asarray(probabilities, dtype=float))
    outcomes = np.asarray(labels, dtype=float)
    return {
        "n": len(labels),
        "n_success": int(np.sum(outcomes)),
        "auroc": auroc(labels, probabilities),
        "auprc": auprc(labels, probabilities),
        "prr_at_0_5": prr(probabilities, labels, 0.5),
        "brier": float(np.mean((clipped - outcomes) ** 2)),
        "nll": float(
            -np.mean(outcomes * np.log(clipped) + (1.0 - outcomes) * np.log(1.0 - clipped))
        ),
        "ece": ece(labels, clipped.tolist()),
    }


@dataclass
class PlattModel:
    mean: float
    std: float
    slope: float
    intercept: float

    def predict(self, values: list[float]) -> list[float]:
        scaled = (np.asarray(values, dtype=float) - self.mean) / self.std
        logits = np.clip(self.slope * scaled + self.intercept, -30.0, 30.0)
        return (1.0 / (1.0 + np.exp(-logits))).tolist()


def fit_platt(values: list[float], labels: list[int]) -> PlattModel:
    array = np.asarray(values, dtype=float)
    outcomes = np.asarray(labels, dtype=float)
    feature_mean = float(np.mean(array))
    feature_std = max(float(np.std(array)), 1e-8)
    scaled = (array - feature_mean) / feature_std
    prior = min(1.0 - EPS, max(EPS, float(np.mean(outcomes))))
    slope = 0.0
    intercept = math.log(prior / (1.0 - prior))
    if feature_std > 1e-7 and len(set(labels)) > 1:
        for _ in range(1500):
            logits = np.clip(slope * scaled + intercept, -30.0, 30.0)
            predictions = 1.0 / (1.0 + np.exp(-logits))
            error = predictions - outcomes
            slope -= 0.05 * float(np.mean(error * scaled))
            intercept -= 0.05 * float(np.mean(error))
    return PlattModel(feature_mean, feature_std, slope, intercept)


def _metric_row(
    *,
    target: str,
    method: str,
    model: str,
    labels: list[int],
    probabilities: list[float],
    split: str = "test",
    prefix: float | None = None,
) -> dict[str, Any]:
    return {
        "split": split,
        "target": target,
        "uq_method": method,
        "model": model,
        "prefix": prefix,
        **metric_values(labels, probabilities),
    }


def _risk_coverage(
    labels: list[int],
    probabilities: list[float],
    *,
    target: str,
    method: str,
    model: str,
) -> list[dict[str, Any]]:
    ranked = sorted(zip(probabilities, labels), reverse=True)
    rows = []
    failures = 0
    for retained, (probability, label) in enumerate(ranked, 1):
        failures += 1 - label
        rows.append(
            {
                "target": target,
                "uq_method": method,
                "model": model,
                "coverage": retained / len(ranked),
                "risk": failures / retained,
                "probability_threshold": probability,
            }
        )
    return rows


def _plot_prefix_metrics(rows: list[dict[str, Any]], output_dir: Path) -> None:
    primary = [
        row
        for row in rows
        if row["target"] == "combined"
        and row["uq_method"] == "perplexity"
        and row["model"]
        in {
            "prior",
            "feature_last",
            "feature_mean",
            "feature_max",
            "binary_bayes",
            "continuous_bayes",
            "tempered_continuous_bayes",
        }
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if primary:
        for model in sorted({row["model"] for row in primary}):
            model_rows = sorted(
                [row for row in primary if row["model"] == model],
                key=lambda row: row["prefix"],
            )
            x = [100 * row["prefix"] for row in model_rows]
            for axis, metric in zip(axes, ("auroc", "brier")):
                y = [np.nan if row[metric] is None else row[metric] for row in model_rows]
                axis.plot(x, y, marker="o", label=model)
        axes[0].set_ylabel("AUROC")
        axes[1].set_ylabel("Brier score")
    else:
        for axis in axes:
            axis.text(0.5, 0.5, "No combined perplexity values", ha="center", va="center")
    for axis in axes:
        axis.set_xlabel("Trajectory observed (%)")
        axis.grid(alpha=0.25)
    if primary:
        axes[-1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "prefix_metrics.png", dpi=180)
    plt.close(fig)


def _plot_belief_by_outcome(rows: list[dict[str, Any]], output_dir: Path) -> None:
    primary = [
        row
        for row in rows
        if row["target"] == "combined"
        and row["uq_method"] == "perplexity"
        and row["model"] == "tempered_continuous_bayes"
    ]
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    if primary:
        for outcome, label in ((0, "failure"), (1, "success")):
            grouped = defaultdict(list)
            for row in primary:
                if row["label"] == outcome:
                    grouped[row["prefix"]].append(row["probability"])
            if grouped:
                x = sorted(grouped)
                y = [float(np.mean(grouped[position])) for position in x]
                axis.plot([100 * value for value in x], y, marker="o", label=label)
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No belief trajectories available", ha="center", va="center")
    axis.set_xlabel("Trajectory observed (%)")
    axis.set_ylabel("Mean predicted success probability")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "belief_by_outcome.png", dpi=180)
    plt.close(fig)


def _plot_examples(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    successes = [row for row in rows if row["label"] == 1][:2]
    failures = [row for row in rows if row["label"] == 0][:2]
    examples = successes + failures
    if examples:
        for row in examples:
            steps = range(1, len(row["uq"]) + 1)
            axes[0].plot(steps, row["uq"], marker=".", label=row["episode_id"][:18])
            axes[1].plot(
                steps,
                row["tempered_belief"],
                marker=".",
                label=f"{row['episode_id'][:14]} y={row['label']}",
            )
        axes[0].set_ylabel("Perplexity")
        axes[1].set_ylabel("Tempered Bayes belief")
        axes[0].legend(fontsize=6)
        axes[1].legend(fontsize=6)
    else:
        for axis in axes:
            axis.text(0.5, 0.5, "No trajectory examples", ha="center", va="center")
    for axis in axes:
        axis.set_xlabel("Step")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_examples.png", dpi=180)
    plt.close(fig)


def _plot_risk_coverage(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    primary = [
        row
        for row in rows
        if row["target"] == "combined"
        and row["uq_method"] == "perplexity"
        and row["model"]
        in {
            "feature_last",
            "feature_mean",
            "feature_max",
            "binary_bayes",
            "continuous_bayes",
            "tempered_continuous_bayes",
            "bayes_state",
            "bayes_state_plus_binary",
            "bayes_state_plus_sep",
            "bayes_state_plus_tempered",
            "stepwise_bayes_state",
            "stepwise_bayes_state_plus_binary",
            "stepwise_bayes_state_plus_sep",
            "stepwise_bayes_state_plus_tempered",
        }
    ]
    for model in sorted({row["model"] for row in primary}):
        model_rows = [row for row in primary if row["model"] == model]
        axis.plot(
            [row["coverage"] for row in model_rows],
            [row["risk"] for row in model_rows],
            label=model,
        )
    if not primary:
        axis.text(0.5, 0.5, "No risk-coverage data", ha="center", va="center")
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Failure risk")
    axis.grid(alpha=0.25)
    if primary:
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "risk_coverage.png", dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Post-hoc trajectory uncertainty and Bayesian belief analysis."
    )
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--episodes", type=Path)
    parser.add_argument(
        "--judge-scores",
        type=Path,
        help="Optional JSONL from experiments.judge_trajectories.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold-quantile", type=float, default=0.9)
    parser.add_argument(
        "--threshold-mode",
        choices=["quantile", "grid", "sep", "lr_pos", "lr_neg"],
        default="quantile",
    )
    parser.add_argument("--tempered-lambda", type=float, default=0.25)
    parser.add_argument("--ewma-alpha", type=float, default=0.3)
    parser.add_argument("--last-k", type=int, default=3)
    parser.add_argument("--cvar-fraction", type=float, default=0.2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.calibration_fraction < 1.0:
        raise SystemExit("--calibration-fraction must be in (0, 1)")
    episodes_path = args.episodes or args.trajectories.with_name("episodes.jsonl")
    trajectory_rows = _read_jsonl(args.trajectories)
    episode_rows = _read_jsonl(episodes_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trajectories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectory_rows:
        trajectories[row["episode_id"]].append(row)
    for rows in trajectories.values():
        rows.sort(key=lambda row: int(row["step"]))

    summaries = {row["episode_id"]: row for row in episode_rows}
    episode_ids = [episode_id for episode_id in summaries if episode_id in trajectories]
    judge_scores: dict[str, dict[str, Any]] = {}
    if args.judge_scores:
        for row in _read_jsonl(args.judge_scores):
            if row.get("status") == "ok":
                judge_scores[str(row["episode_id"])] = row
        missing_judgments = [
            episode_id for episode_id in episode_ids if episode_id not in judge_scores
        ]
        if missing_judgments:
            raise ValueError(
                f"Judge scores are missing {len(missing_judgments)} episode(s); "
                "rerun experiments.judge_trajectories to complete the cache"
            )
    calibration_ids, test_ids = _split_ids(
        episode_ids, args.calibration_fraction, args.seed
    )
    labels = {
        episode_id: int(bool(summaries[episode_id]["final_success"]))
        for episode_id in episode_ids
    }
    split_rows = [
        {
            "episode_id": episode_id,
            "split": "calibration" if episode_id in calibration_ids else "test",
            "task_type": summaries[episode_id]["task_type"],
            "final_success": labels[episode_id],
        }
        for episode_id in episode_ids
    ]
    _write_csv(
        args.output_dir / "split.csv",
        split_rows,
        ["episode_id", "split", "task_type", "final_success"],
    )

    metric_rows: list[dict[str, Any]] = []
    prefix_metric_rows: list[dict[str, Any]] = []
    prefix_prediction_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    model_metadata: list[dict[str, Any]] = []

    prior = min(
        1.0 - EPS,
        max(EPS, sum(labels[i] for i in calibration_ids) / len(calibration_ids)),
    )
    critic_observations = {
        episode_id: _critic_observations(trajectories[episode_id])
        for episode_id in episode_ids
    }
    critic_bayes = CriticBayesState.fit(
        [critic_observations[i] for i in calibration_ids],
        [labels[i] for i in calibration_ids],
    )
    bayes_states = {
        episode_id: critic_bayes.predict(critic_observations[episode_id])
        for episode_id in episode_ids
    }
    judge_observations = {
        episode_id: {"llm_judge_pass": bool(judge_scores[episode_id]["judge_pass"])}
        for episode_id in episode_ids
        if episode_id in judge_scores
    }
    judge_bayes: CriticBayesState | None = None
    critics_plus_judge_bayes: CriticBayesState | None = None
    judge_states: dict[str, float] = {}
    bayes_states_plus_judge: dict[str, float] = {}
    if judge_observations:
        judge_bayes = CriticBayesState.fit(
            [judge_observations[i] for i in calibration_ids],
            [labels[i] for i in calibration_ids],
            prior=prior,
        )
        combined_observations = {
            episode_id: {
                **critic_observations[episode_id],
                **judge_observations[episode_id],
            }
            for episode_id in episode_ids
        }
        critics_plus_judge_bayes = CriticBayesState.fit(
            [combined_observations[i] for i in calibration_ids],
            [labels[i] for i in calibration_ids],
            prior=prior,
        )
        judge_states = {
            episode_id: judge_bayes.predict(judge_observations[episode_id])
            for episode_id in episode_ids
        }
        bayes_states_plus_judge = {
            episode_id: critics_plus_judge_bayes.predict(
                combined_observations[episode_id]
            )
            for episode_id in episode_ids
        }
    step_critic_sequences = {
        episode_id: [
            _step_critic_observation(row) for row in trajectories[episode_id]
        ]
        for episode_id in episode_ids
    }
    step_calibration_observations = [
        observation
        for episode_id in calibration_ids
        for observation in step_critic_sequences[episode_id]
    ]
    step_calibration_labels = [
        labels[episode_id]
        for episode_id in calibration_ids
        for _ in step_critic_sequences[episode_id]
    ]
    stepwise_critic_bayes = CriticBayesState.fit(
        step_calibration_observations,
        step_calibration_labels,
        prior=prior,
    )
    stepwise_bayes_states = {
        episode_id: stepwise_critic_bayes.predict_sequence(
            step_critic_sequences[episode_id]
        )
        for episode_id in episode_ids
    }
    bayes_state_rows = [
        {
            "episode_id": episode_id,
            "split": "calibration" if episode_id in calibration_ids else "test",
            "final_success": labels[episode_id],
            **critic_observations[episode_id],
            "llm_judge_pass": (
                judge_observations[episode_id]["llm_judge_pass"]
                if episode_id in judge_observations
                else None
            ),
            "llm_judge_confidence": (
                judge_scores[episode_id]["confidence"]
                if episode_id in judge_scores
                else None
            ),
            "bayes_state": bayes_states[episode_id],
            "llm_judge_state": judge_states.get(episode_id),
            "bayes_state_plus_judge": bayes_states_plus_judge.get(episode_id),
            "stepwise_bayes_state": stepwise_bayes_states[episode_id],
        }
        for episode_id in episode_ids
    ]
    critic_likelihood_rows = []
    critic_models: list[tuple[str, CriticBayesState, tuple[str, ...]]] = [
        ("episode", critic_bayes, EPISODE_CRITIC_NAMES),
        ("stepwise_uq_exps", stepwise_critic_bayes, STEP_CRITIC_NAMES),
    ]
    if judge_bayes is not None:
        critic_models.append(("llm_judge", judge_bayes, ("llm_judge_pass",)))
    if critics_plus_judge_bayes is not None:
        critic_models.append(
            (
                "episode_plus_llm_judge",
                critics_plus_judge_bayes,
                (*EPISODE_CRITIC_NAMES, "llm_judge_pass"),
            )
        )
    for state_model, model, critic_names in critic_models:
        for critic in critic_names:
            likelihood = model.likelihoods[critic]
            critic_likelihood_rows.append(
                {
                    "state_model": state_model,
                    "critic": critic,
                    "p_pass_success": likelihood.p_pass_success,
                    "p_pass_failure": likelihood.p_pass_failure,
                    "informativeness": (
                        likelihood.p_pass_success - likelihood.p_pass_failure
                    ),
                }
            )
    test_labels = [labels[i] for i in test_ids]
    global_prior_predictions = [prior] * len(test_ids)
    metric_rows.append(
        _metric_row(
            target="all",
            method="none",
            model="prior",
            labels=test_labels,
            probabilities=global_prior_predictions,
        )
    )
    risk_rows.extend(
        _risk_coverage(
            test_labels,
            global_prior_predictions,
            target="all",
            method="none",
            model="prior",
        )
    )
    global_stepwise_predictions = [stepwise_bayes_states[i] for i in test_ids]
    metric_rows.append(
        _metric_row(
            target="all",
            method="stepwise_critics",
            model="stepwise_bayes_state",
            labels=test_labels,
            probabilities=global_stepwise_predictions,
        )
    )
    risk_rows.extend(
        _risk_coverage(
            test_labels,
            global_stepwise_predictions,
            target="all",
            method="stepwise_critics",
            model="stepwise_bayes_state",
        )
    )
    global_bayes_predictions = [bayes_states[i] for i in test_ids]
    metric_rows.append(
        _metric_row(
            target="all",
            method="critics",
            model="bayes_state",
            labels=test_labels,
            probabilities=global_bayes_predictions,
        )
    )
    risk_rows.extend(
        _risk_coverage(
            test_labels,
            global_bayes_predictions,
            target="all",
            method="critics",
            model="bayes_state",
        )
    )
    if judge_states:
        for model_name, method_name, states in (
            ("llm_judge_state", "llm_judge", judge_states),
            (
                "bayes_state_plus_judge",
                "critics_plus_llm_judge",
                bayes_states_plus_judge,
            ),
        ):
            predictions = [states[i] for i in test_ids]
            metric_rows.append(
                _metric_row(
                    target="all",
                    method=method_name,
                    model=model_name,
                    labels=test_labels,
                    probabilities=predictions,
                )
            )
            risk_rows.extend(
                _risk_coverage(
                    test_labels,
                    predictions,
                    target="all",
                    method=method_name,
                    model=model_name,
                )
            )

    for target in TARGETS:
        for method, higher_is_uncertain in UQ_METHODS.items():
            sequences = {
                episode_id: _sequence(trajectories[episode_id], target, method)
                for episode_id in episode_ids
            }
            calibration_available = [
                episode_id for episode_id in calibration_ids if sequences[episode_id]
            ]
            test_available = [
                episode_id for episode_id in test_ids if sequences[episode_id]
            ]
            if not calibration_available or not test_available:
                continue

            calibration_values = [
                value
                for episode_id in calibration_available
                for value in sequences[episode_id]
            ]
            threshold_q = (
                args.threshold_quantile
                if higher_is_uncertain
                else 1.0 - args.threshold_quantile
            )
            threshold = float(np.quantile(calibration_values, threshold_q))
            aggregated = {
                episode_id: aggregate_trajectory(
                    sequences[episode_id],
                    threshold=threshold,
                    higher_is_uncertain=higher_is_uncertain,
                    ewma_alpha=args.ewma_alpha,
                    last_k=args.last_k,
                    cvar_fraction=args.cvar_fraction,
                )
                for episode_id in episode_ids
            }

            for aggregation in AGGREGATIONS:
                train_ids = [
                    episode_id
                    for episode_id in calibration_available
                    if aggregated[episode_id][aggregation] is not None
                ]
                eval_ids = [
                    episode_id
                    for episode_id in test_available
                    if aggregated[episode_id][aggregation] is not None
                ]
                if not train_ids or not eval_ids:
                    continue
                model = fit_platt(
                    [float(aggregated[i][aggregation]) for i in train_ids],
                    [labels[i] for i in train_ids],
                )
                predictions = model.predict(
                    [float(aggregated[i][aggregation]) for i in eval_ids]
                )
                outcomes = [labels[i] for i in eval_ids]
                model_name = f"feature_{aggregation}"
                metric_rows.append(
                    _metric_row(
                        target=target,
                        method=method,
                        model=model_name,
                        labels=outcomes,
                        probabilities=predictions,
                    )
                )
                risk_rows.extend(
                    _risk_coverage(
                        outcomes,
                        predictions,
                        target=target,
                        method=method,
                        model=model_name,
                    )
                )

            calibration_sequences = [sequences[i] for i in calibration_available]
            calibration_labels = [labels[i] for i in calibration_available]
            binary = BinaryBayesUQ.fit(
                calibration_sequences,
                calibration_labels,
                threshold_quantile=args.threshold_quantile,
                threshold_mode=args.threshold_mode,
                higher_is_uncertain=higher_is_uncertain,
            )
            binary_sep = BinaryBayesUQ.fit(
                calibration_sequences,
                calibration_labels,
                threshold_mode="sep",
                higher_is_uncertain=higher_is_uncertain,
            )
            binary_lr_pos = BinaryBayesUQ.fit(
                calibration_sequences,
                calibration_labels,
                threshold_mode="lr_pos",
                higher_is_uncertain=higher_is_uncertain,
            )
            binary_lr_neg = BinaryBayesUQ.fit(
                calibration_sequences,
                calibration_labels,
                threshold_mode="lr_neg",
                higher_is_uncertain=higher_is_uncertain,
            )
            binary_double = DoubleBinaryBayesUQ.fit(
                calibration_sequences,
                calibration_labels,
                higher_is_uncertain=higher_is_uncertain,
            )
            continuous = ContinuousBayesUQ.fit(
                calibration_sequences, calibration_labels, lambda_=1.0
            )
            tempered = ContinuousBayesUQ.fit(
                calibration_sequences,
                calibration_labels,
                lambda_=args.tempered_lambda,
            )
            bayes_models = {
                "binary_bayes": binary,
                "binary_bayes_sep": binary_sep,
                "binary_bayes_lr_pos": binary_lr_pos,
                "binary_bayes_lr_neg": binary_lr_neg,
                "binary_bayes_double": binary_double,
                "continuous_bayes": continuous,
                "tempered_continuous_bayes": tempered,
            }
            model_metadata.append(
                {
                    "target": target,
                    "uq_method": method,
                    "aggregation_threshold": threshold,
                    "binary_threshold": binary.threshold,
                    "prior": binary.prior,
                    "p_certain_success": binary.p_certain_success,
                    "p_certain_failure": binary.p_certain_failure,
                    "binary_sep_threshold": binary_sep.threshold,
                    "binary_lr_pos_threshold": binary_lr_pos.threshold,
                    "binary_lr_neg_threshold": binary_lr_neg.threshold,
                    "binary_double_positive_threshold": (
                        binary_double.positive.threshold
                    ),
                    "binary_double_negative_threshold": (
                        binary_double.negative.threshold
                    ),
                    "continuous_success_mean": continuous.success_mean,
                    "continuous_success_std": continuous.success_std,
                    "continuous_failure_mean": continuous.failure_mean,
                    "continuous_failure_std": continuous.failure_std,
                    "tempered_lambda": args.tempered_lambda,
                }
            )

            outcomes = [labels[i] for i in test_available]
            for model_name, model in bayes_models.items():
                predictions = [model.predict(sequences[i]) for i in test_available]
                metric_rows.append(
                    _metric_row(
                        target=target,
                        method=method,
                        model=model_name,
                        labels=outcomes,
                        probabilities=predictions,
                    )
                )
                risk_rows.extend(
                    _risk_coverage(
                        outcomes,
                        predictions,
                        target=target,
                        method=method,
                        model=model_name,
                    )
                )

            base_predictions = [bayes_states[i] for i in test_available]
            metric_rows.append(
                _metric_row(
                    target=target,
                    method=method,
                    model="bayes_state",
                    labels=outcomes,
                    probabilities=base_predictions,
                )
            )
            risk_rows.extend(
                _risk_coverage(
                    outcomes,
                    base_predictions,
                    target=target,
                    method=method,
                    model="bayes_state",
                )
            )
            if bayes_states_plus_judge:
                judge_base_predictions = [
                    bayes_states_plus_judge[i] for i in test_available
                ]
                metric_rows.append(
                    _metric_row(
                        target=target,
                        method=method,
                        model="bayes_state_plus_judge",
                        labels=outcomes,
                        probabilities=judge_base_predictions,
                    )
                )
                risk_rows.extend(
                    _risk_coverage(
                        outcomes,
                        judge_base_predictions,
                        target=target,
                        method=method,
                        model="bayes_state_plus_judge",
                    )
                )
            fusion_names = {
                "binary_bayes": "bayes_state_plus_binary",
                "binary_bayes_sep": "bayes_state_plus_sep",
                "binary_bayes_lr_pos": "bayes_state_plus_lr_pos",
                "binary_bayes_lr_neg": "bayes_state_plus_lr_neg",
                "binary_bayes_double": "bayes_state_plus_double",
                "continuous_bayes": "bayes_state_plus_continuous",
                "tempered_continuous_bayes": "bayes_state_plus_tempered",
            }
            for source_name, fused_name in fusion_names.items():
                model = bayes_models[source_name]
                predictions = [
                    _predict_from_belief(model, sequences[i], bayes_states[i])
                    for i in test_available
                ]
                metric_rows.append(
                    _metric_row(
                        target=target,
                        method=method,
                        model=fused_name,
                        labels=outcomes,
                        probabilities=predictions,
                    )
                )
                risk_rows.extend(
                    _risk_coverage(
                        outcomes,
                        predictions,
                        target=target,
                        method=method,
                        model=fused_name,
                    )
                )
                if bayes_states_plus_judge:
                    judge_fused_name = fused_name.replace(
                        "bayes_state_plus_", "bayes_state_plus_judge_plus_"
                    )
                    judge_predictions = [
                        _predict_from_belief(
                            model,
                            sequences[i],
                            bayes_states_plus_judge[i],
                        )
                        for i in test_available
                    ]
                    metric_rows.append(
                        _metric_row(
                            target=target,
                            method=method,
                            model=judge_fused_name,
                            labels=outcomes,
                            probabilities=judge_predictions,
                        )
                    )
                    risk_rows.extend(
                        _risk_coverage(
                            outcomes,
                            judge_predictions,
                            target=target,
                            method=method,
                            model=judge_fused_name,
                        )
                    )

            stepwise_base_predictions = [
                stepwise_bayes_states[i] for i in test_available
            ]
            metric_rows.append(
                _metric_row(
                    target=target,
                    method=method,
                    model="stepwise_bayes_state",
                    labels=outcomes,
                    probabilities=stepwise_base_predictions,
                )
            )
            risk_rows.extend(
                _risk_coverage(
                    outcomes,
                    stepwise_base_predictions,
                    target=target,
                    method=method,
                    model="stepwise_bayes_state",
                )
            )
            for source_name, episode_fused_name in fusion_names.items():
                model = bayes_models[source_name]
                fused_name = episode_fused_name.replace(
                    "bayes_state_plus_", "stepwise_bayes_state_plus_"
                )
                predictions = [
                    _predict_from_belief(
                        model,
                        sequences[i],
                        stepwise_bayes_states[i],
                    )
                    for i in test_available
                ]
                metric_rows.append(
                    _metric_row(
                        target=target,
                        method=method,
                        model=fused_name,
                        labels=outcomes,
                        probabilities=predictions,
                    )
                )
                risk_rows.extend(
                    _risk_coverage(
                        outcomes,
                        predictions,
                        target=target,
                        method=method,
                        model=fused_name,
                    )
                )

            for fraction in PREFIXES:
                truncated_calibration = {
                    i: _prefix(sequences[i], fraction) for i in calibration_available
                }
                truncated_test = {
                    i: _prefix(sequences[i], fraction) for i in test_available
                }
                prefix_outcomes = [labels[i] for i in test_available]
                prior_predictions = [binary.prior] * len(test_available)
                prefix_metric_rows.append(
                    _metric_row(
                        target=target,
                        method=method,
                        model="prior",
                        labels=prefix_outcomes,
                        probabilities=prior_predictions,
                        prefix=fraction,
                    )
                )
                for episode_id, probability in zip(test_available, prior_predictions):
                    prefix_prediction_rows.append(
                        {
                            "target": target,
                            "uq_method": method,
                            "model": "prior",
                            "prefix": fraction,
                            "episode_id": episode_id,
                            "label": labels[episode_id],
                            "probability": probability,
                        }
                    )

                for aggregation in ("last", "mean", "max"):
                    train_features = [
                        float(
                            aggregate_trajectory(
                                truncated_calibration[i],
                                threshold=threshold,
                                higher_is_uncertain=higher_is_uncertain,
                            )[aggregation]
                        )
                        for i in calibration_available
                    ]
                    test_features = [
                        float(
                            aggregate_trajectory(
                                truncated_test[i],
                                threshold=threshold,
                                higher_is_uncertain=higher_is_uncertain,
                            )[aggregation]
                        )
                        for i in test_available
                    ]
                    platt = fit_platt(train_features, calibration_labels)
                    predictions = platt.predict(test_features)
                    model_name = f"feature_{aggregation}"
                    prefix_metric_rows.append(
                        _metric_row(
                            target=target,
                            method=method,
                            model=model_name,
                            labels=prefix_outcomes,
                            probabilities=predictions,
                            prefix=fraction,
                        )
                    )
                    for episode_id, probability in zip(test_available, predictions):
                        prefix_prediction_rows.append(
                            {
                                "target": target,
                                "uq_method": method,
                                "model": model_name,
                                "prefix": fraction,
                                "episode_id": episode_id,
                                "label": labels[episode_id],
                                "probability": probability,
                            }
                        )

                for model_name, model in bayes_models.items():
                    predictions = [
                        model.predict(truncated_test[i]) for i in test_available
                    ]
                    prefix_metric_rows.append(
                        _metric_row(
                            target=target,
                            method=method,
                            model=model_name,
                            labels=prefix_outcomes,
                            probabilities=predictions,
                            prefix=fraction,
                        )
                    )
                    for episode_id, probability in zip(test_available, predictions):
                        prefix_prediction_rows.append(
                            {
                                "target": target,
                                "uq_method": method,
                                "model": model_name,
                                "prefix": fraction,
                                "episode_id": episode_id,
                                "label": labels[episode_id],
                                "probability": probability,
                            }
                        )

            if target == "combined" and method == "perplexity":
                for episode_id in test_available:
                    example_rows.append(
                        {
                            "episode_id": episode_id,
                            "label": labels[episode_id],
                            "uq": sequences[episode_id],
                            "binary_belief": binary.predict_sequence(
                                sequences[episode_id]
                            ),
                            "continuous_belief": continuous.predict_sequence(
                                sequences[episode_id]
                            ),
                            "tempered_belief": tempered.predict_sequence(
                                sequences[episode_id]
                            ),
                        }
                    )

    metric_fields = [
        "split",
        "target",
        "uq_method",
        "model",
        "prefix",
        "n",
        "n_success",
        "auroc",
        "auprc",
        "prr_at_0_5",
        "brier",
        "nll",
        "ece",
    ]
    _write_csv(args.output_dir / "metrics.csv", metric_rows, metric_fields)
    _write_csv(
        args.output_dir / "prefix_metrics.csv", prefix_metric_rows, metric_fields
    )
    _write_csv(
        args.output_dir / "risk_coverage.csv",
        risk_rows,
        [
            "target",
            "uq_method",
            "model",
            "coverage",
            "risk",
            "probability_threshold",
        ],
    )
    _write_csv(
        args.output_dir / "model_parameters.csv",
        model_metadata,
        [
            "target",
            "uq_method",
            "aggregation_threshold",
            "binary_threshold",
            "prior",
            "p_certain_success",
            "p_certain_failure",
            "binary_sep_threshold",
            "binary_lr_pos_threshold",
            "binary_lr_neg_threshold",
            "binary_double_positive_threshold",
            "binary_double_negative_threshold",
            "continuous_success_mean",
            "continuous_success_std",
            "continuous_failure_mean",
            "continuous_failure_std",
            "tempered_lambda",
        ],
    )
    _write_csv(
        args.output_dir / "critic_likelihoods.csv",
        critic_likelihood_rows,
        [
            "state_model",
            "critic",
            "p_pass_success",
            "p_pass_failure",
            "informativeness",
        ],
    )
    _write_csv(
        args.output_dir / "bayes_states.csv",
        bayes_state_rows,
        [
            "episode_id",
            "split",
            "final_success",
            *EPISODE_CRITIC_NAMES,
            "llm_judge_pass",
            "llm_judge_confidence",
            "bayes_state",
            "llm_judge_state",
            "bayes_state_plus_judge",
            "stepwise_bayes_state",
        ],
    )
    with (args.output_dir / "belief_trajectories.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in example_rows:
            handle.write(json.dumps(row) + "\n")

    _plot_prefix_metrics(prefix_metric_rows, args.output_dir)
    _plot_belief_by_outcome(prefix_prediction_rows, args.output_dir)
    _plot_examples(example_rows, args.output_dir)
    _plot_risk_coverage(risk_rows, args.output_dir)

    available_methods = sorted(
        {
            (row["target"], row["uq_method"])
            for row in metric_rows
            if row["uq_method"] != "none"
        }
    )
    logprob_steps = sum(bool(row.get("logprobs_available")) for row in trajectory_rows)
    fallback_counts = defaultdict(int)
    provider_counts = defaultdict(int)
    for row in trajectory_rows:
        fallback_counts[str(row.get("fallback_reason") or "none")] += 1
        provider_counts[str(row.get("provider") or "unknown")] += 1
    stop_counts = defaultdict(int)
    for row in episode_rows:
        stop_counts[str(row.get("stop_reason", "unknown"))] += 1
    durations = [float(row.get("duration_seconds", 0.0)) for row in episode_rows]
    primary_models = {
        "feature_last",
        "feature_mean",
        "feature_max",
        "binary_bayes",
        "binary_bayes_sep",
        "binary_bayes_lr_pos",
        "binary_bayes_lr_neg",
        "binary_bayes_double",
        "continuous_bayes",
        "tempered_continuous_bayes",
        "bayes_state",
        "bayes_state_plus_binary",
        "bayes_state_plus_sep",
        "bayes_state_plus_lr_pos",
        "bayes_state_plus_lr_neg",
        "bayes_state_plus_double",
        "bayes_state_plus_continuous",
        "bayes_state_plus_tempered",
        "bayes_state_plus_judge",
        "bayes_state_plus_judge_plus_binary",
        "bayes_state_plus_judge_plus_sep",
        "bayes_state_plus_judge_plus_continuous",
        "bayes_state_plus_judge_plus_tempered",
        "stepwise_bayes_state",
        "stepwise_bayes_state_plus_binary",
        "stepwise_bayes_state_plus_sep",
        "stepwise_bayes_state_plus_lr_pos",
        "stepwise_bayes_state_plus_lr_neg",
        "stepwise_bayes_state_plus_double",
        "stepwise_bayes_state_plus_continuous",
        "stepwise_bayes_state_plus_tempered",
    }
    primary_metrics = [
        row
        for row in metric_rows
        if row["target"] == "combined"
        and row["uq_method"] == "perplexity"
        and row["model"] in primary_models
    ]
    thought_fusion_models = {
        "bayes_state",
        "bayes_state_plus_binary",
        "bayes_state_plus_sep",
        "bayes_state_plus_lr_pos",
        "bayes_state_plus_lr_neg",
        "bayes_state_plus_double",
        "bayes_state_plus_continuous",
        "bayes_state_plus_tempered",
        "bayes_state_plus_judge",
        "bayes_state_plus_judge_plus_binary",
        "bayes_state_plus_judge_plus_sep",
        "bayes_state_plus_judge_plus_continuous",
        "bayes_state_plus_judge_plus_tempered",
    }
    thought_fusion_metrics = [
        row
        for row in metric_rows
        if row["target"] == "thought"
        and row["uq_method"] == "sum_logprob"
        and row["model"] in thought_fusion_models
    ]
    thought_fusion_base = next(
        (row for row in thought_fusion_metrics if row["model"] == "bayes_state"),
        None,
    )
    stepwise_fusion_models = {
        model for model in primary_models if model.startswith("stepwise_bayes_state")
    }
    stepwise_fusion_metrics = [
        row
        for row in metric_rows
        if row["target"] == "thought"
        and row["uq_method"] == "sum_logprob"
        and row["model"] in stepwise_fusion_models
    ]
    stepwise_fusion_base = next(
        (
            row
            for row in stepwise_fusion_metrics
            if row["model"] == "stepwise_bayes_state"
        ),
        None,
    )
    comparable_metrics = [
        row
        for row in metric_rows
        if row["uq_method"] != "none" and row["auroc"] is not None
    ]
    best_auroc = max(
        comparable_metrics,
        key=lambda row: float(row["auroc"]),
        default=None,
    )
    best_brier = min(
        comparable_metrics,
        key=lambda row: float(row["brier"]),
        default=None,
    )
    prr_metrics = [
        row for row in comparable_metrics if row["prr_at_0_5"] is not None
    ]
    best_prr = max(
        prr_metrics,
        key=lambda row: float(row["prr_at_0_5"]),
        default=None,
    )
    task_counts = defaultdict(lambda: [0, 0])
    for episode_id in episode_ids:
        task_type = summaries[episode_id]["task_type"]
        task_counts[task_type][0] += int(labels[episode_id])
        task_counts[task_type][1] += 1
    report = [
        "# ALFWorld trajectory UQ report",
        "",
        f"- Episodes: {len(episode_ids)} "
        f"({len(calibration_ids)} calibration, {len(test_ids)} test)",
        f"- Successes: {sum(labels.values())}/{len(labels)}",
        f"- Steps with token logprobs: {logprob_steps}/{len(trajectory_rows)}",
        f"- Stop reasons: {dict(sorted(stop_counts.items()))}",
        f"- Fallbacks: {dict(sorted(fallback_counts.items()))}",
        f"- Providers: {dict(sorted(provider_counts.items()))}",
        f"- Total tokens: {sum(int(row.get('total_tokens', 0)) for row in episode_rows)}",
        f"- Median episode time: {float(np.median(durations)):.1f}s",
        f"- P95 episode time: {float(np.quantile(durations, 0.95)):.1f}s",
        f"- Available target/method pairs: {len(available_methods)}",
        f"- Metric rows: {len(metric_rows)}",
        f"- LLM judge scores: {len(judge_scores)}/{len(episode_ids)}",
        "",
        "## Critic Bayes state",
        "",
        "Episode state applies each summarized critic once; stepwise_uq_exps applies all critics after every generation.",
        "",
        "| State model | Critic | P(pass | success) | P(pass | failure) | Informativeness |",
        "|---|---|---:|---:|---:|",
    ]
    for row in critic_likelihood_rows:
        report.append(
            f"| {row['state_model']} | {row['critic']} | "
            f"{row['p_pass_success']:.4f} | "
            f"{row['p_pass_failure']:.4f} | {row['informativeness']:+.4f} |"
        )
    report.extend(
        [
            "",
        "## Combined perplexity",
        "",
        "| Model | AUROC | AUPRC | PRR@0.5 | Brier | NLL | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(primary_metrics, key=lambda item: item["model"]):
        values = [
            "NA" if row[name] is None else f"{float(row[name]):.4f}"
            for name in ("auroc", "auprc", "prr_at_0_5", "brier", "nll", "ece")
        ]
        report.append(f"| {row['model']} | {' | '.join(values)} |")
    report.extend(
        [
            "",
            "These pilot metrics use only "
            f"{len(test_ids)} test episodes with {sum(labels[i] for i in test_ids)} "
            "positive outcome(s). They validate the pipeline but are not stable "
            "performance estimates.",
        ]
    )
    if thought_fusion_metrics:
        report.extend(
            [
                "",
                "## Bayes state + thought sum_logprob",
                "",
                "| Model | AUROC | Delta AUROC | PRR@0.5 | Delta PRR@0.5 | Brier | Delta Brier | NLL | Delta NLL |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(thought_fusion_metrics, key=lambda item: item["model"]):
            assert thought_fusion_base is not None
            report.append(
                f"| {row['model']} | {float(row['auroc']):.4f} | "
                f"{float(row['auroc']) - float(thought_fusion_base['auroc']):+.4f} | "
                f"{float(row['prr_at_0_5']):.4f} | "
                f"{float(row['prr_at_0_5']) - float(thought_fusion_base['prr_at_0_5']):+.4f} | "
                f"{float(row['brier']):.4f} | "
                f"{float(row['brier']) - float(thought_fusion_base['brier']):+.4f} | "
                f"{float(row['nll']):.4f} | "
                f"{float(row['nll']) - float(thought_fusion_base['nll']):+.4f} |"
            )
    if stepwise_fusion_metrics:
        report.extend(
            [
                "",
                "## Stepwise uq_exps-style state + thought sum_logprob",
                "",
                "| Model | AUROC | Delta AUROC | PRR@0.5 | Delta PRR@0.5 | Brier | Delta Brier | NLL | Delta NLL |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(stepwise_fusion_metrics, key=lambda item: item["model"]):
            assert stepwise_fusion_base is not None
            report.append(
                f"| {row['model']} | {float(row['auroc']):.4f} | "
                f"{float(row['auroc']) - float(stepwise_fusion_base['auroc']):+.4f} | "
                f"{float(row['prr_at_0_5']):.4f} | "
                f"{float(row['prr_at_0_5']) - float(stepwise_fusion_base['prr_at_0_5']):+.4f} | "
                f"{float(row['brier']):.4f} | "
                f"{float(row['brier']) - float(stepwise_fusion_base['brier']):+.4f} | "
                f"{float(row['nll']):.4f} | "
                f"{float(row['nll']) - float(stepwise_fusion_base['nll']):+.4f} |"
            )
        report.extend(
            [
                "",
                "The stepwise variant is a mechanics-matching stress test: proxy critics "
                "are correlated and failed episodes contribute more observations because "
                "they are usually longer.",
            ]
        )
    report.extend(["", "## Best observed features", ""])
    if best_auroc is not None and best_brier is not None:
        report.extend(
            [
                f"- Best AUROC: `{best_auroc['target']}/{best_auroc['uq_method']}/"
                f"{best_auroc['model']}` = {float(best_auroc['auroc']):.4f} "
                f"(Brier {float(best_auroc['brier']):.4f}).",
                f"- Best Brier: `{best_brier['target']}/{best_brier['uq_method']}/"
                f"{best_brier['model']}` = {float(best_brier['brier']):.4f} "
                f"(AUROC {float(best_brier['auroc']):.4f}).",
            ]
        )
        if best_prr is not None:
            report.append(
                f"- Best PRR@0.5: `{best_prr['target']}/{best_prr['uq_method']}/"
                f"{best_prr['model']}` = {float(best_prr['prr_at_0_5']):.4f}."
            )
        report.append(
            "- Sum log-probability is length-sensitive; compare it against normalized "
            "mean token log-probability/perplexity before treating it as epistemic UQ."
        )
    else:
        report.append("No comparable UQ features were available for ranking.")
    report.extend(
        [
            "",
            "## Success by task type",
            "",
            "| Task type | Success | Total | Rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for task_type, (successes, total) in sorted(task_counts.items()):
        report.append(
            f"| {task_type} | {successes} | {total} | {successes / total:.1%} |"
        )
    report.extend(
        [
            "",
            "## Example episodes",
            "",
        ]
    )
    success_examples = sorted([value for value in episode_ids if labels[value]])[:2]
    failure_examples = sorted([value for value in episode_ids if not labels[value]])[:2]
    for episode_id in success_examples + failure_examples:
        summary = summaries[episode_id]
        report.append(
            f"- `{episode_id}`: success={bool(labels[episode_id])}, "
            f"steps={summary.get('num_steps')}, stop={summary.get('stop_reason')}"
        )
    report.append("")
    if not available_methods:
        report.extend(
            [
                "No token-level UQ values were present. The prior baseline was still "
                "evaluated, but logprob-based methods require an endpoint that returns "
                "chat completion token logprobs.",
                "",
            ]
        )
    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "episodes": len(episode_ids),
                "calibration": len(calibration_ids),
                "test": len(test_ids),
                "metric_rows": len(metric_rows),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
