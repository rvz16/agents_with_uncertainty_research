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


def _gaussian_logpdf(value: float, mean: float, std: float) -> float:
    z = (value - mean) / std
    return -0.5 * z * z - math.log(std) - 0.5 * math.log(2.0 * math.pi)


@dataclass
class ContinuousBayesUQ:
    prior: float
    success_mean: float
    success_std: float
    failure_mean: float
    failure_std: float
    lambda_: float = 1.0

    @classmethod
    def fit(
        cls,
        sequences: list[list[float]],
        labels: list[int],
        *,
        lambda_: float = 1.0,
        min_std: float = 1e-3,
    ) -> "ContinuousBayesUQ":
        if len(sequences) != len(labels) or not sequences:
            raise ValueError("sequences and labels must be non-empty and aligned")
        if lambda_ < 0.0:
            raise ValueError("lambda_ must be non-negative")

        pooled = np.asarray([value for sequence in sequences for value in sequence])
        if pooled.size == 0:
            raise ValueError("at least one finite UQ value is required")
        global_mean = float(np.mean(pooled))
        global_std = max(float(np.std(pooled)), min_std)

        by_label = {
            label: np.asarray(
                [
                    value
                    for sequence, outcome in zip(sequences, labels)
                    if outcome == label
                    for value in sequence
                ]
            )
            for label in (0, 1)
        }

        def parameters(label: int) -> tuple[float, float]:
            values = by_label[label]
            if values.size == 0:
                return global_mean, global_std
            return float(np.mean(values)), max(float(np.std(values)), min_std)

        failure_mean, failure_std = parameters(0)
        success_mean, success_std = parameters(1)
        return cls(
            prior=_clip_probability(sum(labels) / len(labels)),
            success_mean=success_mean,
            success_std=success_std,
            failure_mean=failure_mean,
            failure_std=failure_std,
            lambda_=lambda_,
        )

    def log_likelihood_ratio(self, value: float) -> float:
        return _gaussian_logpdf(
            value, self.success_mean, self.success_std
        ) - _gaussian_logpdf(value, self.failure_mean, self.failure_std)

    def update(self, belief: float, value: float) -> float:
        return _clip_probability(
            _sigmoid(
                _logit(belief) + self.lambda_ * self.log_likelihood_ratio(value)
            )
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
