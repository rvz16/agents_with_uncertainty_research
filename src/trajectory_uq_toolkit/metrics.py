from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


EPS = 1e-6


def auroc(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return None
    return float(
        np.mean(
            [float(pos > neg) + 0.5 * float(pos == neg) for pos in positives for neg in negatives]
        )
    )


def auprc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if positives == 0 or positives == len(labels):
        return None
    groups: dict[float, list[int]] = {}
    for score, label in zip(scores, labels):
        groups.setdefault(float(score), []).append(label)
    true_positives = 0
    retrieved = 0
    weighted_precision = 0.0
    for score in sorted(groups, reverse=True):
        group = groups[score]
        group_positives = sum(group)
        true_positives += group_positives
        retrieved += len(group)
        weighted_precision += group_positives * (true_positives / retrieved)
    return weighted_precision / positives


def prediction_rejection_area(
    uncertainty: list[float], labels: list[int], max_rejection: float = 0.5
) -> float | None:
    if len(labels) < 2:
        return None
    uncertainty_array = np.asarray(uncertainty, dtype=float)
    outcomes = np.asarray(labels, dtype=float)
    if np.all(outcomes == outcomes[0]):
        return None
    ranked = outcomes[np.argsort(uncertainty_array)]
    rejected = int(max_rejection * len(ranked))
    if rejected <= 0:
        return None
    cumulative = np.cumsum(ranked)[-rejected:]
    retained = np.arange(len(ranked) - rejected + 1, len(ranked) + 1)
    return float(np.sum((cumulative / retained)[::-1]) / rejected)


@lru_cache(maxsize=None)
def _prr_references(labels: tuple[int, ...], max_rejection: float) -> tuple[float | None, float | None]:
    outcomes = list(labels)
    oracle = prediction_rejection_area([-float(label) for label in outcomes], outcomes, max_rejection)
    rng = np.random.RandomState(42)
    order = np.arange(len(outcomes))
    random_areas = []
    for _ in range(500):
        rng.shuffle(order)
        area = prediction_rejection_area(order.tolist(), outcomes, max_rejection)
        if area is not None:
            random_areas.append(area)
    return oracle, float(np.mean(random_areas)) if random_areas else None


def prr(confidence: list[float], labels: list[int], max_rejection: float = 0.5) -> float | None:
    if confidence and max(confidence) == min(confidence):
        return 0.0
    area = prediction_rejection_area([-value for value in confidence], labels, max_rejection)
    oracle, random = _prr_references(tuple(labels), max_rejection)
    if area is None or oracle is None or random is None or abs(oracle - random) < 1e-12:
        return None
    return (area - random) / (oracle - random)


def ece(labels: list[int], probabilities: list[float], bins: int = 10) -> float:
    labels_array = np.asarray(labels, dtype=float)
    probability_array = np.asarray(probabilities, dtype=float)
    result = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        mask = (probability_array >= lower) & (
            probability_array <= upper if upper >= 1.0 else probability_array < upper
        )
        if np.any(mask):
            result += float(np.mean(mask)) * abs(
                float(np.mean(probability_array[mask])) - float(np.mean(labels_array[mask]))
            )
    return result


def metric_values(labels: list[int], probabilities: list[float]) -> dict[str, float | int | None]:
    clipped = np.clip(np.asarray(probabilities, dtype=float), EPS, 1.0 - EPS)
    outcomes = np.asarray(labels, dtype=float)
    return {
        "n": len(labels),
        "n_success": int(np.sum(outcomes)),
        "auroc": auroc(labels, probabilities),
        "auprc": auprc(labels, probabilities),
        "prr_at_0_5": prr(probabilities, labels),
        "brier": float(np.mean((clipped - outcomes) ** 2)),
        "nll": float(-np.mean(outcomes * np.log(clipped) + (1.0 - outcomes) * np.log(1.0 - clipped))),
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
    mean = float(np.mean(array))
    std = max(float(np.std(array)), 1e-8)
    scaled = (array - mean) / std
    prior = min(1.0 - EPS, max(EPS, float(np.mean(outcomes))))
    slope = 0.0
    intercept = math.log(prior / (1.0 - prior))
    if std > 1e-7 and len(set(labels)) > 1:
        for _ in range(1500):
            predictions = 1.0 / (1.0 + np.exp(-np.clip(slope * scaled + intercept, -30.0, 30.0)))
            error = predictions - outcomes
            slope -= 0.05 * float(np.mean(error * scaled))
            intercept -= 0.05 * float(np.mean(error))
    return PlattModel(mean, std, slope, intercept)
