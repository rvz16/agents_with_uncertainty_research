from __future__ import annotations

import math


def sum_log_probability(token_logprobs: list[float]) -> float | None:
    return sum(token_logprobs) if token_logprobs else None


def mean_token_log_probability(token_logprobs: list[float]) -> float | None:
    return sum(token_logprobs) / len(token_logprobs) if token_logprobs else None


def sequence_probability(token_logprobs: list[float]) -> float | None:
    total = sum_log_probability(token_logprobs)
    return math.exp(total) if total is not None else None
