from __future__ import annotations

import math
from statistics import fmean
from typing import Iterable, Sequence


def _finite(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def summarize_token_logprobs(
    chosen_logprobs: Sequence[float],
    top_logprobs: Sequence[Sequence[float]] | None = None,
) -> dict[str, float]:
    """Reduce token distributions immediately so raw arrays need not be saved.

    ``top_logprobs`` contains one sequence of log-probabilities per generated
    token. They may be top-k rather than full-vocabulary distributions.
    """
    chosen = _finite(chosen_logprobs)
    if not chosen:
        return {}
    total = sum(chosen)
    mean = fmean(chosen)
    result = {
        "sum_logprob": total,
        "mean_token_logprob": mean,
        "sequence_probability": math.exp(total) if total > -745.0 else 0.0,
        "perplexity": math.exp(min(700.0, -mean)),
        "num_tokens": float(len(chosen)),
    }
    if not top_logprobs:
        return result

    entropies: list[float] = []
    kl_to_uniform: list[float] = []
    self_certainties: list[float] = []
    for raw_distribution in top_logprobs:
        logps = _finite(raw_distribution)
        if len(logps) < 2:
            continue
        maximum = max(logps)
        weights = [math.exp(value - maximum) for value in logps]
        normalizer = sum(weights)
        probabilities = [value / normalizer for value in weights]
        entropy = -sum(p * math.log(p) for p in probabilities if p > 0.0)
        entropies.append(entropy)
        kl_to_uniform.append(math.log(len(probabilities)) - entropy)
        self_certainties.append(-fmean(logps) - math.log(len(logps)))
    if entropies:
        result.update(
            {
                "mean_token_entropy": fmean(entropies),
                "kl_to_uniform": fmean(kl_to_uniform),
                "self_certainty": fmean(self_certainties),
            }
        )
    return result
