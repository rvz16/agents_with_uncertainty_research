"""POMDP-based mutation strategy selector for the AlphaEvolve loop.

Adapts the Bayesian POMDP controller from
article_implementation/bayesian/example.py to select which GPU kernel
optimization strategy to emphasize in the LLM prompt each iteration.

Key adaptation:
- Hypotheses H = optimization strategies (tiling, fusion, shared_memory, ...)
- Tests T = trying strategy X in the LLM prompt
- Outcomes Z = improved / not_improved
- Belief b(H) updated via Bayes rule after each observation
- DP solver picks strategy that maximizes expected information gain
"""

import logging
import numpy as np
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Strategy descriptions for the LLM prompt
STRATEGY_DESCRIPTIONS = {
    "tiling": (
        "Use tiling / loop blocking to improve data locality. "
        "Break the computation into tiles that fit in shared memory or registers."
    ),
    "fusion": (
        "Fuse multiple operations into a single kernel to reduce memory bandwidth. "
        "Eliminate intermediate global memory reads/writes."
    ),
    "shared_memory": (
        "Use shared memory to cache frequently accessed data. "
        "Reduce global memory accesses by loading tiles into __shared__ memory."
    ),
    "vectorize": (
        "Use vectorized loads/stores (float4, int4) and warp-level primitives. "
        "Maximize memory throughput with coalesced, wide memory transactions."
    ),
    "algorithmic": (
        "Apply algorithmic improvements: change the computation order, "
        "use mathematical identities, reduce redundant work, or apply "
        "domain-specific optimizations."
    ),
}

DEFAULT_STRATEGIES = list(STRATEGY_DESCRIPTIONS.keys())


@dataclass
class StrategyConfig:
    """Configuration for the POMDP strategy selector."""
    strategies: list[str] = field(default_factory=lambda: DEFAULT_STRATEGIES)
    budget_depth: int = 2  # DP lookahead depth
    prior_diagonal: float = 0.7  # P(improve | correct strategy)
    prior_off_diagonal: float = 0.2  # P(improve | wrong strategy)
    smoothing_alpha: float = 0.5  # weight for prior vs empirical in updates
    min_observations_for_update: int = 10  # min obs before updating likelihood


class MutationStrategySelector:
    """POMDP-based mutation strategy selector.

    Adapts compute_posterior() and get_optimal_policy_value() from
    article_implementation/bayesian/example.py to select which
    optimization strategy to emphasize in the LLM prompt.

    The belief vector b[i] = P(best_strategy = strategies[i]) is updated
    after each iteration using Bayes' rule, just like the diagnostic
    controller in example.py updates P(bug_type = H).
    """

    def __init__(self, config: StrategyConfig | None = None):
        self.config = config or StrategyConfig()
        self.strategies = self.config.strategies
        self.n = len(self.strategies)
        self.belief = np.ones(self.n) / self.n  # uniform prior
        self.budget_depth = self.config.budget_depth

        # P(improve | best_strategy=H, tried_strategy=T)
        # Diagonal-dominant: trying the "right" strategy most likely to improve
        # Mirrors P_FAIL_GIVEN_H_T in example.py
        self._p_improve = self._make_prior_matrix()
        self._history: list[dict] = []

    def _make_prior_matrix(self) -> np.ndarray:
        """Prior: P(improve | H, T) -- high on diagonal, low off-diagonal.

        Same structure as Parameters.P_FAIL_GIVEN_H_T in example.py:
        each row is a hypothesis, each column is a test/strategy.
        """
        mat = np.full(
            (self.n, self.n), self.config.prior_off_diagonal
        )
        np.fill_diagonal(mat, self.config.prior_diagonal)
        return mat

    def select_strategy(self) -> str:
        """Use DP solver (get_optimal_policy_value) to pick the next strategy.

        Returns the strategy name to emphasize in the LLM prompt.
        """
        _, best_idx = self._get_optimal_policy_value(
            self.belief, self.budget_depth
        )
        if best_idx < 0:
            best_idx = int(np.argmax(self.belief))

        strategy = self.strategies[best_idx]
        logger.debug(
            f"POMDP selected strategy '{strategy}' "
            f"(belief={self.belief[best_idx]:.3f})"
        )
        return strategy

    def observe(self, strategy: str, improved: bool):
        """Update belief via compute_posterior() after trying a strategy.

        Mirrors the belief update in EpisodeSimulator.run_episode_bayesian()
        from example.py.

        Args:
            strategy: The strategy that was emphasized.
            improved: Whether the child's fitness exceeded the parent's.
        """
        if strategy not in self.strategies:
            logger.warning(f"Unknown strategy '{strategy}', skipping observe")
            return

        t_idx = self.strategies.index(strategy)
        self.belief = self._compute_posterior(self.belief, t_idx, improved)
        self._history.append({"strategy": strategy, "improved": improved})
        self._update_likelihood_matrix()

        logger.debug(
            f"POMDP observe: strategy='{strategy}', improved={improved}, "
            f"belief={dict(zip(self.strategies, self.belief.round(3)))}"
        )

    def get_strategy_description(self, strategy: str) -> str:
        """Get the description for a strategy (for prompt injection)."""
        return STRATEGY_DESCRIPTIONS.get(strategy, "")

    def get_belief_summary(self) -> dict:
        """Return current belief state for logging."""
        return {
            "belief": dict(zip(self.strategies, self.belief.round(4).tolist())),
            "map_strategy": self.strategies[int(np.argmax(self.belief))],
            "map_confidence": float(np.max(self.belief)),
            "total_observations": len(self._history),
        }

    # --- Core Bayesian logic (adapted from example.py) ---

    def _compute_posterior(
        self,
        belief: np.ndarray,
        test_idx: int,
        outcome_improved: bool,
    ) -> np.ndarray:
        """Update belief using Bayes' rule.

        Direct adaptation of compute_posterior(belief, test_idx, outcome_is_fail)
        from article_implementation/bayesian/example.py.

        Args:
            belief: Prior P(H), shape (n,).
            test_idx: Which strategy was tried.
            outcome_improved: Whether the outcome was an improvement.

        Returns:
            Posterior belief vector, normalized.
        """
        p_improve = self._p_improve[:, test_idx]
        likelihood = p_improve if outcome_improved else (1.0 - p_improve)

        unnormalized = belief * likelihood
        evidence = np.sum(unnormalized)

        if evidence == 0:
            return belief

        return unnormalized / evidence

    def _get_terminal_value(self, belief: np.ndarray) -> float:
        """Expected value of committing to MAP strategy.

        Adapted from get_terminal_value(belief) in example.py.
        In the original: R * max(belief) - C_PATCH - C_VER
        Here we just use max(belief) as confidence score (no costs).
        """
        return float(np.max(belief))

    def _get_optimal_policy_value(
        self, belief: np.ndarray, depth: int
    ) -> tuple[float, int]:
        """Solve DP recursion for optimal strategy selection.

        Direct adaptation of get_optimal_policy_value(belief, depth)
        from article_implementation/bayesian/example.py.

        Args:
            belief: Current belief state.
            depth: Number of "probes" remaining in lookahead budget.

        Returns:
            (expected_value, best_strategy_index)
        """
        if depth == 0:
            return self._get_terminal_value(belief), -1

        best_value = -float("inf")
        best_action = -1

        for t_idx in range(self.n):
            # P(improve | b, T) = sum_H P(improve | H, T) * b(H)
            p_imp = np.dot(belief, self._p_improve[:, t_idx])

            # Posterior for both outcomes
            b_imp = self._compute_posterior(belief, t_idx, True)
            b_no = self._compute_posterior(belief, t_idx, False)

            # Recurse
            v_imp, _ = self._get_optimal_policy_value(b_imp, depth - 1)
            v_no, _ = self._get_optimal_policy_value(b_no, depth - 1)

            # Bellman equation (no explicit cost term here)
            val = p_imp * v_imp + (1 - p_imp) * v_no

            if val > best_value:
                best_value = val
                best_action = t_idx

        return best_value, best_action

    def _update_likelihood_matrix(self):
        """Update P(improve|H,T) from accumulated history.

        Uses simple counting with smoothing to blend the prior matrix
        with empirical observations.
        """
        if len(self._history) < self.config.min_observations_for_update:
            return

        counts: dict[str, dict[str, int]] = {
            s: {"total": 0, "improved": 0} for s in self.strategies
        }
        for obs in self._history:
            counts[obs["strategy"]]["total"] += 1
            if obs["improved"]:
                counts[obs["strategy"]]["improved"] += 1

        alpha = self.config.smoothing_alpha
        for i, s in enumerate(self.strategies):
            if counts[s]["total"] > 0:
                empirical = counts[s]["improved"] / counts[s]["total"]
                # Blend prior diagonal with empirical success rate
                self._p_improve[i, i] = (
                    alpha * self._p_improve[i, i] + (1 - alpha) * empirical
                )
