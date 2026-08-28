from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


EPS = 1e-6


def clip_probability(value: float) -> float:
    return min(1.0 - EPS, max(EPS, float(value)))


def logit(value: float) -> float:
    value = clip_probability(value)
    return math.log(value / (1.0 - value))


def sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


@dataclass
class BinaryBayes:
    prior: float
    threshold: float
    p_pass_success: float
    p_pass_failure: float
    higher_is_uncertain: bool

    @classmethod
    def fit(
        cls,
        sequences: list[list[float]],
        labels: list[int],
        *,
        mode: str = "quantile",
        higher_is_uncertain: bool,
        smoothing: float = 1.0,
    ) -> "BinaryBayes":
        if len(sequences) != len(labels) or not sequences:
            raise ValueError("sequences and labels must be non-empty and aligned")
        values = [value for sequence in sequences for value in sequence]
        if not values:
            raise ValueError("at least one UQ value is required")
        if mode == "quantile":
            q = 0.9 if higher_is_uncertain else 0.1
            candidates = [float(np.quantile(values, q))]
        elif mode in {"sep", "lr_pos", "lr_neg"}:
            candidates = [float(np.quantile(values, q)) for q in np.linspace(0.05, 0.95, 19)]
        else:
            raise ValueError(f"unsupported binary mode: {mode}")
        prior = clip_probability(sum(labels) / len(labels))
        best: BinaryBayes | None = None
        best_score = -math.inf
        for threshold in candidates:
            counts = {0: [0, 0], 1: [0, 0]}
            for sequence, label in zip(sequences, labels):
                for value in sequence:
                    passed = value <= threshold if higher_is_uncertain else value >= threshold
                    counts[label][0] += int(passed)
                    counts[label][1] += 1
            p1 = (counts[1][0] + smoothing) / (counts[1][1] + 2.0 * smoothing)
            p0 = (counts[0][0] + smoothing) / (counts[0][1] + 2.0 * smoothing)
            model = cls(prior, threshold, clip_probability(p1), clip_probability(p0), higher_is_uncertain)
            if mode == "sep":
                score = abs(p1 - p0)
            elif mode == "lr_pos":
                score = p1 / max(p0, EPS)
            elif mode == "lr_neg":
                score = -((1.0 - p1) / max(1.0 - p0, EPS))
            else:
                predictions = [model.predict(sequence) for sequence in sequences]
                score = -sum((prediction - label) ** 2 for prediction, label in zip(predictions, labels))
            if score > best_score:
                best, best_score = model, score
        assert best is not None
        return best

    def update(self, belief: float, value: float) -> float:
        passed = value <= self.threshold if self.higher_is_uncertain else value >= self.threshold
        if passed:
            ratio = self.p_pass_success / self.p_pass_failure
        else:
            ratio = (1.0 - self.p_pass_success) / (1.0 - self.p_pass_failure)
        return clip_probability(sigmoid(logit(belief) + math.log(ratio)))

    def predict(self, sequence: list[float], start: float | None = None) -> float:
        belief = self.prior if start is None else start
        for value in sequence:
            belief = self.update(belief, value)
        return belief


@dataclass
class DoubleBinaryBayes:
    prior: float
    positive: BinaryBayes
    negative: BinaryBayes

    @classmethod
    def fit(
        cls, sequences: list[list[float]], labels: list[int], *, higher_is_uncertain: bool
    ) -> "DoubleBinaryBayes":
        positive = BinaryBayes.fit(
            sequences, labels, mode="lr_pos", higher_is_uncertain=higher_is_uncertain
        )
        negative = BinaryBayes.fit(
            sequences, labels, mode="lr_neg", higher_is_uncertain=higher_is_uncertain
        )
        return cls(positive.prior, positive, negative)

    def predict(self, sequence: list[float], start: float | None = None) -> float:
        belief = self.prior if start is None else start
        for value in sequence:
            belief = self.negative.update(self.positive.update(belief, value), value)
        return belief


def _gaussian_logpdf(value: float, mean: float, std: float) -> float:
    z = (value - mean) / std
    return -0.5 * z * z - math.log(std) - 0.5 * math.log(2.0 * math.pi)


@dataclass
class ContinuousBayes:
    prior: float
    success_mean: float
    success_std: float
    failure_mean: float
    failure_std: float
    lambda_: float

    @classmethod
    def fit(
        cls, sequences: list[list[float]], labels: list[int], *, lambda_: float = 1.0
    ) -> "ContinuousBayes":
        if lambda_ < 0.0 or len(sequences) != len(labels) or not sequences:
            raise ValueError("invalid continuous Bayes inputs")
        pooled = np.asarray([value for sequence in sequences for value in sequence], dtype=float)
        if pooled.size == 0:
            raise ValueError("at least one UQ value is required")
        global_mean = float(np.mean(pooled))
        global_std = max(float(np.std(pooled)), 1e-3)

        def parameters(label: int) -> tuple[float, float]:
            values = np.asarray(
                [value for sequence, outcome in zip(sequences, labels) if outcome == label for value in sequence],
                dtype=float,
            )
            if values.size == 0:
                return global_mean, global_std
            return float(np.mean(values)), max(float(np.std(values)), 1e-3)

        failure_mean, failure_std = parameters(0)
        success_mean, success_std = parameters(1)
        return cls(
            clip_probability(sum(labels) / len(labels)),
            success_mean,
            success_std,
            failure_mean,
            failure_std,
            lambda_,
        )

    def update(self, belief: float, value: float) -> float:
        evidence = _gaussian_logpdf(value, self.success_mean, self.success_std)
        evidence -= _gaussian_logpdf(value, self.failure_mean, self.failure_std)
        return clip_probability(sigmoid(logit(belief) + self.lambda_ * evidence))

    def predict(self, sequence: list[float], start: float | None = None) -> float:
        belief = self.prior if start is None else start
        for value in sequence:
            belief = self.update(belief, value)
        return belief


@dataclass(frozen=True)
class CriticLikelihood:
    p_pass_success: float
    p_pass_failure: float


@dataclass
class CriticBayes:
    prior: float
    likelihoods: dict[str, CriticLikelihood]

    @classmethod
    def fit(
        cls,
        observations: list[dict[str, bool]],
        labels: list[int],
        *,
        names: list[str],
        smoothing: float = 1.0,
    ) -> "CriticBayes":
        likelihoods = {}
        for name in names:
            probabilities = {}
            for label in (0, 1):
                values = [row[name] for row, outcome in zip(observations, labels) if outcome == label]
                probabilities[label] = (sum(values) + smoothing) / (len(values) + 2.0 * smoothing)
            likelihoods[name] = CriticLikelihood(
                clip_probability(probabilities[1]), clip_probability(probabilities[0])
            )
        return cls(clip_probability(sum(labels) / len(labels)), likelihoods)

    def update(self, belief: float, name: str, passed: bool) -> float:
        likelihood = self.likelihoods[name]
        p1 = likelihood.p_pass_success if passed else 1.0 - likelihood.p_pass_success
        p0 = likelihood.p_pass_failure if passed else 1.0 - likelihood.p_pass_failure
        return clip_probability(sigmoid(logit(belief) + math.log(p1 / p0)))

    def predict(self, observations: dict[str, bool]) -> float:
        belief = self.prior
        for name in sorted(self.likelihoods):
            belief = self.update(belief, name, observations[name])
        return belief

    def predict_sequence(self, sequence: list[dict[str, bool]]) -> float:
        belief = self.prior
        for observations in sequence:
            for name in sorted(self.likelihoods):
                belief = self.update(belief, name, observations[name])
        return belief
