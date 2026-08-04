from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


EPS = 1e-6


def _clip_probability(value: float) -> float:
    return min(1.0 - EPS, max(EPS, value))


def _logit(value: float) -> float:
    value = _clip_probability(value)
    return math.log(value / (1.0 - value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


@dataclass
class BinaryBayesUQ:
    prior: float
    threshold: float
    p_certain_success: float
    p_certain_failure: float
    higher_is_uncertain: bool = True

    @classmethod
    def fit(
        cls,
        sequences: list[list[float]],
        labels: list[int],
        *,
        threshold_quantile: float = 0.9,
        threshold_mode: str = "quantile",
        higher_is_uncertain: bool = True,
        smoothing: float = 1.0,
    ) -> "BinaryBayesUQ":
        if len(sequences) != len(labels) or not sequences:
            raise ValueError("sequences and labels must be non-empty and aligned")
        prior = _clip_probability(sum(labels) / len(labels))
        all_values = [value for sequence in sequences for value in sequence]
        if not all_values:
            raise ValueError("at least one finite UQ value is required")

        quantile = threshold_quantile if higher_is_uncertain else 1.0 - threshold_quantile
        candidates = [float(np.quantile(all_values, quantile))]
        if threshold_mode == "grid":
            candidates = [
                float(np.quantile(all_values, q))
                for q in np.linspace(0.05, 0.95, 19)
            ]
        elif threshold_mode != "quantile":
            raise ValueError("threshold_mode must be 'quantile' or 'grid'")

        best: BinaryBayesUQ | None = None
        best_brier = math.inf
        for threshold in candidates:
            model = cls._fit_at_threshold(
                sequences,
                labels,
                prior=prior,
                threshold=threshold,
                higher_is_uncertain=higher_is_uncertain,
                smoothing=smoothing,
            )
            predictions = [model.predict(sequence) for sequence in sequences]
            brier = sum((p - y) ** 2 for p, y in zip(predictions, labels)) / len(labels)
            if brier < best_brier:
                best, best_brier = model, brier
        assert best is not None
        return best

    @classmethod
    def _fit_at_threshold(
        cls,
        sequences: list[list[float]],
        labels: list[int],
        *,
        prior: float,
        threshold: float,
        higher_is_uncertain: bool,
        smoothing: float,
    ) -> "BinaryBayesUQ":
        counts = {0: [0, 0], 1: [0, 0]}
        for sequence, label in zip(sequences, labels):
            for value in sequence:
                certain = value <= threshold if higher_is_uncertain else value >= threshold
                counts[int(label)][0] += int(certain)
                counts[int(label)][1] += 1
        p_success = (counts[1][0] + smoothing) / (counts[1][1] + 2.0 * smoothing)
        p_failure = (counts[0][0] + smoothing) / (counts[0][1] + 2.0 * smoothing)
        return cls(
            prior=prior,
            threshold=threshold,
            p_certain_success=_clip_probability(p_success),
            p_certain_failure=_clip_probability(p_failure),
            higher_is_uncertain=higher_is_uncertain,
        )

    def update(self, belief: float, value: float) -> float:
        certain = (
            value <= self.threshold
            if self.higher_is_uncertain
            else value >= self.threshold
        )
        if certain:
            likelihood_ratio = self.p_certain_success / self.p_certain_failure
        else:
            likelihood_ratio = (1.0 - self.p_certain_success) / (
                1.0 - self.p_certain_failure
            )
        return _clip_probability(
            _sigmoid(_logit(belief) + math.log(likelihood_ratio))
        )

    def predict_sequence(self, sequence: list[float]) -> list[float]:
        beliefs = []
        belief = self.prior
        for value in sequence:
            belief = self.update(belief, value)
            beliefs.append(belief)
        return beliefs

    def predict(self, sequence: list[float]) -> float:
        beliefs = self.predict_sequence(sequence)
        return beliefs[-1] if beliefs else self.prior
