from __future__ import annotations

import math
from dataclasses import dataclass


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


@dataclass(frozen=True)
class CriticLikelihood:
    p_pass_success: float
    p_pass_failure: float


@dataclass
class CriticBayesState:
    """Episode-level belief from binary, non-oracle critic observations."""

    prior: float
    likelihoods: dict[str, CriticLikelihood]

    @classmethod
    def fit(
        cls,
        observations: list[dict[str, bool]],
        labels: list[int],
        *,
        smoothing: float = 1.0,
        prior: float | None = None,
    ) -> "CriticBayesState":
        if len(observations) != len(labels) or not observations:
            raise ValueError("observations and labels must be non-empty and aligned")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        critic_names = sorted(set.intersection(*(set(row) for row in observations)))
        if not critic_names:
            raise ValueError("at least one critic observation is required")

        likelihoods = {}
        for critic in critic_names:
            by_label = {}
            for label in (0, 1):
                values = [
                    bool(row[critic])
                    for row, outcome in zip(observations, labels)
                    if outcome == label
                ]
                by_label[label] = (sum(values) + smoothing) / (
                    len(values) + 2.0 * smoothing
                )
            likelihoods[critic] = CriticLikelihood(
                p_pass_success=_clip_probability(by_label[1]),
                p_pass_failure=_clip_probability(by_label[0]),
            )
        return cls(
            prior=_clip_probability(
                sum(labels) / len(labels) if prior is None else prior
            ),
            likelihoods=likelihoods,
        )

    def update(self, belief: float, critic: str, passed: bool) -> float:
        likelihood = self.likelihoods[critic]
        if passed:
            p_success = likelihood.p_pass_success
            p_failure = likelihood.p_pass_failure
        else:
            p_success = 1.0 - likelihood.p_pass_success
            p_failure = 1.0 - likelihood.p_pass_failure
        return _clip_probability(
            _sigmoid(_logit(belief) + math.log(p_success / p_failure))
        )

    def predict(self, observations: dict[str, bool]) -> float:
        belief = self.prior
        for critic in sorted(self.likelihoods):
            belief = self.update(belief, critic, bool(observations[critic]))
        return belief

    def predict_sequence(self, sequence: list[dict[str, bool]]) -> float:
        belief = self.prior
        for observations in sequence:
            for critic in sorted(self.likelihoods):
                belief = self.update(belief, critic, bool(observations[critic]))
        return belief

    def predict_sequence_tempered(self, sequence: list[dict[str, bool]]) -> float:
        """The same evidence averaged over steps instead of multiplied.

        One likelihood ratio per step treats steps as independent draws; a
        long episode then saturates the posterior after a handful of steps
        and the ranking degenerates into a count of failed steps -- which is
        why the plain tool-success *rate* outranked the multiplied belief
        (.79 vs .58 mean PRR across the ALFWorld grid). Dividing the total
        log-evidence by the number of steps -- the geometric mean of the
        per-step likelihood ratios -- keeps the rate's resolution and the
        prior, and is the critic counterpart of the tempered UQ update.
        """
        if not sequence:
            return self.prior
        total = 0.0
        for observations in sequence:
            for critic, likelihood in self.likelihoods.items():
                passed = bool(observations[critic])
                p_success = likelihood.p_pass_success if passed else 1 - likelihood.p_pass_success
                p_failure = likelihood.p_pass_failure if passed else 1 - likelihood.p_pass_failure
                total += math.log(_clip_probability(p_success) / _clip_probability(p_failure))
        return _clip_probability(_sigmoid(_logit(self.prior) + total / len(sequence)))
