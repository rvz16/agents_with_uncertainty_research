"""Acquisition functions for Bayesian optimization.

These serve the same role as get_optimal_policy_value() in
article_implementation/bayesian/example.py — they balance
exploration (uncertainty) vs exploitation (predicted performance)
to choose the next experiment.
"""

import numpy as np
from scipy.stats import norm


def expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_so_far: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected Improvement acquisition function.

    EI(x) = E[max(f(x) - f(x*), 0)]

    Analogous to the Bellman value in example.py:
    - mean ~ expected reward for choosing this action
    - std ~ information gain potential
    - xi ~ exploration bonus (like C_TEST cost tradeoff)

    Args:
        mean: Predicted means from GP.
        std: Predicted stds from GP.
        best_so_far: Best observed value (f(x*)).
        xi: Exploration-exploitation tradeoff (higher = more exploration).

    Returns:
        EI values for each candidate.
    """
    with np.errstate(divide="warn"):
        improvement = mean - best_so_far - xi
        Z = np.where(std > 1e-8, improvement / std, 0.0)
        ei = np.where(
            std > 1e-8,
            improvement * norm.cdf(Z) + std * norm.pdf(Z),
            np.maximum(improvement, 0.0),
        )
    return ei


def upper_confidence_bound(
    mean: np.ndarray,
    std: np.ndarray,
    kappa: float = 2.0,
) -> np.ndarray:
    """Upper Confidence Bound acquisition function.

    UCB(x) = mu(x) + kappa * sigma(x)

    Simple and effective. kappa controls exploration:
    - kappa=0: pure exploitation (greedy)
    - kappa=2: standard (good default)
    - kappa>3: aggressive exploration

    Args:
        mean: Predicted means from GP.
        std: Predicted stds from GP.
        kappa: Exploration weight.

    Returns:
        UCB values for each candidate.
    """
    return mean + kappa * std


def probability_of_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_so_far: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Probability of Improvement acquisition function.

    PI(x) = P(f(x) > f(x*) + xi)

    Simplest acquisition function. Tends to over-exploit.

    Args:
        mean: Predicted means.
        std: Predicted stds.
        best_so_far: Best observed value.
        xi: Minimum improvement threshold.

    Returns:
        PI values for each candidate.
    """
    with np.errstate(divide="warn"):
        Z = np.where(std > 1e-8, (mean - best_so_far - xi) / std, 0.0)
    return norm.cdf(Z)
