from __future__ import annotations

import math
from statistics import fmean, median, pstdev


def aggregate(values: list[float], *, higher_is_uncertain: bool) -> dict[str, float]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {}
    alpha = 0.3
    ewma = clean[0]
    for value in clean[1:]:
        ewma = alpha * value + (1.0 - alpha) * ewma
    tail_count = max(1, math.ceil(0.2 * len(clean)))
    ordered_uncertain = sorted(clean, reverse=higher_is_uncertain)
    return {
        "last": clean[-1],
        "first": clean[0],
        "mean": fmean(clean),
        "min": min(clean),
        "max": max(clean),
        "median": median(clean),
        "std": pstdev(clean) if len(clean) > 1 else 0.0,
        "range": max(clean) - min(clean),
        "ewma": ewma,
        "last_k_mean": fmean(clean[-3:]),
        "cvar": fmean(ordered_uncertain[:tail_count]),
    }
