from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MATPLOTLIB_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from belief.binary_bayes import BinaryBayesUQ
from belief.continuous_bayes import ContinuousBayesUQ
from uq.aggregation import aggregate_trajectory


UQ_METHODS = {
    "perplexity": True,
    "sum_logprob": False,
    "mean_token_logprob": False,
    "sequence_probability": False,
    "verbalized_confidence": False,
}
TARGETS = ("thought", "action", "combined")
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold-quantile", type=float, default=0.9)
    parser.add_argument(
        "--threshold-mode", choices=["quantile", "grid"], default="quantile"
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
            "continuous_success_mean",
            "continuous_success_std",
            "continuous_failure_mean",
            "continuous_failure_std",
            "tempered_lambda",
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
        "continuous_bayes",
        "tempered_continuous_bayes",
    }
    primary_metrics = [
        row
        for row in metric_rows
        if row["target"] == "combined"
        and row["uq_method"] == "perplexity"
        and row["model"] in primary_models
    ]
    comparable_metrics = [
        row
        for row in metric_rows
        if row["uq_method"] != "none" and row["auroc"] is not None
    ]
    best_auroc = max(comparable_metrics, key=lambda row: float(row["auroc"]))
    best_brier = min(comparable_metrics, key=lambda row: float(row["brier"]))
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
        "",
        "## Combined perplexity",
        "",
        "| Model | AUROC | AUPRC | Brier | NLL | ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(primary_metrics, key=lambda item: item["model"]):
        values = [
            "NA" if row[name] is None else f"{float(row[name]):.4f}"
            for name in ("auroc", "auprc", "brier", "nll", "ece")
        ]
        report.append(f"| {row['model']} | {' | '.join(values)} |")
    report.extend(
        [
            "",
            "These pilot metrics use only "
            f"{len(test_ids)} test episodes with {sum(labels[i] for i in test_ids)} "
            "positive outcome(s). They validate the pipeline but are not stable "
            "performance estimates.",
            "",
            "## Best observed features",
            "",
            f"- Best AUROC: `{best_auroc['target']}/{best_auroc['uq_method']}/"
            f"{best_auroc['model']}` = {float(best_auroc['auroc']):.4f} "
            f"(Brier {float(best_auroc['brier']):.4f}).",
            f"- Best Brier: `{best_brier['target']}/{best_brier['uq_method']}/"
            f"{best_brier['model']}` = {float(best_brier['brier']):.4f} "
            f"(AUROC {float(best_brier['auroc']):.4f}).",
            "- Sum log-probability is length-sensitive; compare it against normalized "
            "mean token log-probability/perplexity before treating it as epistemic UQ.",
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
