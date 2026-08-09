from __future__ import annotations

import math


def perplexity(token_logprobs: list[float]) -> float | None:
    if not token_logprobs:
        return None
    return math.exp(-sum(token_logprobs) / len(token_logprobs))
