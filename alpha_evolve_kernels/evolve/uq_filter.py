"""Uncertainty quantification pre-compilation filter.

Uses UncertaintyEstimator classes from sgr+uncertainty/estimators.py
to score LLM outputs by confidence. High uncertainty generations are
skipped before expensive compilation and evaluation.

Inspired by SAGE's tau_exec=0.85 execution threshold.
"""

import logging
import sys
import os
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Add sgr+uncertainty to path for imports
_ESTIMATORS_DIR = str(Path(__file__).resolve().parents[2] / "sgr+uncertainty")
if _ESTIMATORS_DIR not in sys.path:
    sys.path.insert(0, _ESTIMATORS_DIR)

from estimators import (
    GenerationLogprobs,
    TokenLogprob,
    UncertaintyEstimator,
    MaxSequenceProbability,
    MeanTokenEntropy,
    PerplexityEstimator,
    NormalizedSequenceEntropy,
    TokenLevelVariance,
    MinTokenProbability,
    CompositeEstimator,
)

# Registry of available estimators
ESTIMATOR_REGISTRY: dict[str, type[UncertaintyEstimator]] = {
    "max_seq_prob": MaxSequenceProbability,
    "mean_token_entropy": MeanTokenEntropy,
    "perplexity": PerplexityEstimator,
    "normalized_seq_entropy": NormalizedSequenceEntropy,
    "token_variance": TokenLevelVariance,
    "min_token_prob": MinTokenProbability,
}


@dataclass
class UQFilterConfig:
    """Configuration for the UQ pre-compilation filter."""
    threshold: float = 5.0  # initial uncertainty threshold (perplexity scale)
    estimator_type: str = "perplexity"  # which estimator to use
    auto_calibrate: bool = True  # adjust threshold from observations
    calibration_percentile: float = 90.0  # keep 90% of correct programs
    min_observations_for_calibration: int = 10
    # For composite estimator
    composite_weights: dict[str, float] | None = None


class UQPreCompilationFilter:
    """Filters out low-confidence LLM generations before compilation.

    Uses UncertaintyEstimator from sgr+uncertainty/estimators.py to
    compute uncertainty scores from token logprobs. High uncertainty
    means the LLM was not confident in its output, so we skip the
    expensive compile+correctness+profiling pipeline.

    The threshold is adaptive: it calibrates from observed (uncertainty,
    correctness) pairs to find a cutoff that retains most correct programs
    while filtering out uncertain junk.
    """

    def __init__(self, config: UQFilterConfig | None = None):
        self.config = config or UQFilterConfig()
        self.estimator = self._build_estimator()
        self._threshold = self.config.threshold
        self._calibration_data: list[tuple[float, bool]] = []
        self._stats = {"filtered": 0, "passed": 0}

    def _build_estimator(self) -> UncertaintyEstimator:
        """Instantiate estimator by name from the registry."""
        name = self.config.estimator_type

        if name == "composite" and self.config.composite_weights:
            estimators = []
            for est_name, weight in self.config.composite_weights.items():
                cls = ESTIMATOR_REGISTRY.get(est_name)
                if cls:
                    estimators.append((cls(), weight))
            if estimators:
                return CompositeEstimator(estimators)
            logger.warning("No valid estimators for composite, falling back to perplexity")
            return PerplexityEstimator()

        cls = ESTIMATOR_REGISTRY.get(name)
        if cls is None:
            logger.warning(f"Unknown estimator '{name}', falling back to perplexity")
            cls = PerplexityEstimator
        return cls()

    def should_evaluate(
        self, logprobs: GenerationLogprobs
    ) -> tuple[bool, float]:
        """Decide whether to proceed with compilation.

        Args:
            logprobs: Token-level logprobs from the LLM generation.

        Returns:
            (should_evaluate, uncertainty_score)
            True = confident enough, proceed with compilation.
            False = too uncertain, skip this candidate.
        """
        score = self.estimator.estimate(logprobs)
        should_eval = score < self._threshold

        if should_eval:
            self._stats["passed"] += 1
        else:
            self._stats["filtered"] += 1
            logger.debug(
                f"UQ filter: SKIP (score={score:.3f} >= threshold={self._threshold:.3f})"
            )

        return should_eval, score

    def observe(self, score: float, was_correct: bool):
        """Feed back evaluation result to calibrate threshold.

        Args:
            score: The uncertainty score that was computed.
            was_correct: Whether the evaluated program was correct.
        """
        self._calibration_data.append((score, was_correct))

        if self.config.auto_calibrate:
            self._recalibrate()

    def _recalibrate(self):
        """Adjust threshold based on observed correlation.

        Strategy: set threshold so that calibration_percentile% of
        correct programs would pass the filter.
        """
        if len(self._calibration_data) < self.config.min_observations_for_calibration:
            return

        correct_scores = [s for s, c in self._calibration_data if c]
        if not correct_scores:
            return

        new_threshold = float(
            np.percentile(correct_scores, self.config.calibration_percentile)
        )

        if abs(new_threshold - self._threshold) > 0.01:
            logger.info(
                f"UQ filter recalibrated: threshold {self._threshold:.3f} → {new_threshold:.3f} "
                f"(from {len(correct_scores)} correct observations)"
            )
            self._threshold = new_threshold

    def get_stats(self) -> dict:
        """Return filter statistics for logging."""
        total = self._stats["passed"] + self._stats["filtered"]
        return {
            "passed": self._stats["passed"],
            "filtered": self._stats["filtered"],
            "filter_rate": (
                self._stats["filtered"] / total if total > 0 else 0.0
            ),
            "threshold": self._threshold,
            "calibration_points": len(self._calibration_data),
            "estimator": self.estimator.name,
        }


def parse_openai_logprobs(choice_logprobs) -> GenerationLogprobs | None:
    """Parse OpenAI API logprobs response into GenerationLogprobs.

    Args:
        choice_logprobs: The `logprobs` field from a ChatCompletion choice.
            Expected structure: {content: [{token, logprob, top_logprobs: [{token, logprob}]}]}

    Returns:
        GenerationLogprobs instance, or None if logprobs unavailable.
    """
    if choice_logprobs is None:
        return None

    content = getattr(choice_logprobs, "content", None)
    if content is None:
        return None

    tokens = []
    for token_info in content:
        top_lps = None
        if hasattr(token_info, "top_logprobs") and token_info.top_logprobs:
            top_lps = {
                t.token: t.logprob for t in token_info.top_logprobs
            }

        tokens.append(
            TokenLogprob(
                token=token_info.token,
                logprob=token_info.logprob,
                top_logprobs=top_lps,
            )
        )

    return GenerationLogprobs(tokens=tokens)
