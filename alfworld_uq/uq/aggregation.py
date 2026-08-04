from __future__ import annotations

import math
from statistics import mean, median


def ewma(values: list[float], alpha: float = 0.3) -> float | None:
    if not values:
        return None
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1.0 - alpha) * result
    return result


def cvar(
    values: list[float],
    *,
    fraction: float = 0.2,
    higher_is_uncertain: bool = True,
) -> float | None:
    if not values:
        return None
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    count = max(1, math.ceil(len(values) * fraction))
    ordered = sorted(values, reverse=higher_is_uncertain)
    return mean(ordered[:count])


def aggregate_trajectory(
    values: list[float | None],
    *,
    threshold: float | None = None,
    higher_is_uncertain: bool = True,
    ewma_alpha: float = 0.3,
    last_k: int = 3,
    cvar_fraction: float = 0.2,
) -> dict[str, float | None]:
    clean = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return {
            name: None
            for name in (
                "last",
                "mean",
                "min",
                "max",
                "median",
                "ewma",
                "fraction_uncertain",
                "last_k_mean",
                "cvar",
            )
        }
    if last_k < 1:
        raise ValueError("last_k must be positive")

    fraction_uncertain = None
    if threshold is not None:
        flags = (
            [value > threshold for value in clean]
            if higher_is_uncertain
            else [value < threshold for value in clean]
        )
        fraction_uncertain = sum(flags) / len(flags)

    return {
        "last": clean[-1],
        "mean": mean(clean),
        "min": min(clean),
        "max": max(clean),
        "median": median(clean),
        "ewma": ewma(clean, ewma_alpha),
        "fraction_uncertain": fraction_uncertain,
        "last_k_mean": mean(clean[-last_k:]),
        "cvar": cvar(
            clean,
            fraction=cvar_fraction,
            higher_is_uncertain=higher_is_uncertain,
        ),
    }
