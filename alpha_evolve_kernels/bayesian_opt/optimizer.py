"""Bayesian optimizer for CUDA kernel hyperparameters.

Extends the belief-update pattern from article_implementation/bayesian/example.py:
- example.py: discrete hypotheses, exact Bayes update, DP solver
- This: continuous HP space, GP surrogate, acquisition function

The core loop mirrors example.py's diagnostic loop:
1. Current belief (GP posterior) over HP space
2. Choose next experiment (acquisition function ~ DP solver)
3. Run experiment (evaluate kernel)
4. Update belief (GP.fit() ~ compute_posterior())
"""

import re
import logging
import numpy as np
from itertools import product

from .models import HyperparamConfig, ParameterDomain, ObservationPoint, COMMON_DOMAINS
from .surrogate import GPSurrogate
from .acquisition import expected_improvement, upper_confidence_bound

logger = logging.getLogger(__name__)


class BayesianKernelOptimizer:
    """Bayesian optimizer for CUDA kernel hyperparameters.

    Usage:
        optimizer = BayesianKernelOptimizer(domains=[...])

        for i in range(budget):
            config = optimizer.suggest()          # acquisition function
            speedup = evaluate_kernel(config)     # run experiment
            optimizer.observe(config, speedup)    # belief update

        best = optimizer.get_best()
    """

    def __init__(
        self,
        domains: list[ParameterDomain] | None = None,
        acquisition: str = "ei",  # "ei", "ucb", "pi"
        n_initial: int = 5,       # Random exploration budget
        seed: int = 42,
        kappa: float = 2.0,       # UCB exploration weight
        xi: float = 0.01,        # EI/PI exploration bonus
    ):
        self.domains = domains or [
            COMMON_DOMAINS["block_size_x"],
            COMMON_DOMAINS["tile_size"],
            COMMON_DOMAINS["num_warps"],
        ]
        self.acquisition_name = acquisition
        self.n_initial = n_initial
        self.kappa = kappa
        self.xi = xi

        self.surrogate = GPSurrogate()
        self.observations: list[ObservationPoint] = []
        self._rng = np.random.default_rng(seed)

        # Build candidate grid (all combinations of domain values)
        self._candidate_grid = self._build_grid()

    def _build_grid(self) -> list[HyperparamConfig]:
        """Build full grid of candidate HP configurations."""
        all_values = [d.values for d in self.domains]
        grid = []
        for combo in product(*all_values):
            values = {d.name: v for d, v in zip(self.domains, combo)}
            grid.append(HyperparamConfig(values=values))
        return grid

    def suggest(self) -> HyperparamConfig:
        """Suggest the next HP configuration to evaluate.

        If < n_initial observations: random exploration.
        Otherwise: maximize acquisition function over candidate grid.

        This is analogous to get_optimal_policy_value() in example.py —
        it picks the action (HP config) with highest expected value.
        """
        # Exploration phase: random sampling
        if len(self.observations) < self.n_initial:
            idx = self._rng.integers(0, len(self._candidate_grid))
            return self._candidate_grid[idx]

        # Exploitation phase: GP + acquisition function
        self.surrogate.fit(self.observations, self.domains)

        mean, std = self.surrogate.predict(self._candidate_grid, self.domains)

        best_so_far = max(
            (o.speedup for o in self.observations if o.correct),
            default=0.0,
        )

        if self.acquisition_name == "ucb":
            scores = upper_confidence_bound(mean, std, kappa=self.kappa)
        elif self.acquisition_name == "pi":
            from .acquisition import probability_of_improvement
            scores = probability_of_improvement(mean, std, best_so_far, xi=self.xi)
        else:  # "ei"
            scores = expected_improvement(mean, std, best_so_far, xi=self.xi)

        best_idx = int(np.argmax(scores))
        return self._candidate_grid[best_idx]

    def observe(
        self,
        config: HyperparamConfig,
        speedup: float,
        correct: bool = True,
        metadata: dict | None = None,
    ) -> None:
        """Record an observation and update the GP posterior.

        This is the Bayesian belief update step — analogous to
        compute_posterior() in example.py.
        """
        obs = ObservationPoint(
            config=config,
            speedup=speedup,
            correct=correct,
            metadata=metadata or {},
        )
        self.observations.append(obs)

        # Refit GP if enough data
        if len(self.observations) >= 2:
            self.surrogate.fit(self.observations, self.domains)

    def observe_from_code(self, code: str, speedup: float, correct: bool = True) -> None:
        """Extract HP values from kernel code and record observation.

        Parses constants like BLOCK_SIZE = 256 from the source.
        """
        config = self.extract_hyperparams(code)
        if config.values:
            self.observe(config, speedup, correct)

    def extract_hyperparams(self, code: str) -> HyperparamConfig:
        """Extract hyperparameter values from CUDA/Python source code.

        Looks for patterns like:
        - BLOCK_SIZE = 256
        - TILE_SIZE = 32
        - threadsPerBlock(16, 16)
        - num_warps = 4
        """
        values = {}

        # Pattern: NAME = value (Python/CUDA constants)
        const_pattern = re.compile(
            r'\b([A-Z_]+(?:SIZE|BLOCK|TILE|WARP|UNROLL|THREADS)[A-Z_]*)\s*=\s*(\d+)',
            re.IGNORECASE,
        )
        for match in const_pattern.finditer(code):
            name = match.group(1).lower()
            val = int(match.group(2))

            # Map to our domain names
            for domain in self.domains:
                if self._name_matches(name, domain.name):
                    if val in domain.values:
                        values[domain.name] = val
                    else:
                        # Snap to nearest valid value
                        values[domain.name] = min(domain.values, key=lambda v: abs(v - val))

        # Pattern: dim3 threadsPerBlock(X, Y)
        dim3_pattern = re.compile(r'dim3\s+\w+\s*\(\s*(\d+)\s*,\s*(\d+)')
        match = dim3_pattern.search(code)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            for d in self.domains:
                if "block_size_x" in d.name and d.name not in values:
                    values[d.name] = min(d.values, key=lambda v: abs(v - x))
                if "block_size_y" in d.name and d.name not in values:
                    values[d.name] = min(d.values, key=lambda v: abs(v - y))

        return HyperparamConfig(values=values)

    def _name_matches(self, code_name: str, domain_name: str) -> bool:
        """Check if a code constant name matches a domain name."""
        code_name = code_name.lower().replace("_", "")
        domain_name = domain_name.lower().replace("_", "")

        # Direct match
        if code_name == domain_name:
            return True

        # Partial matches
        mappings = {
            "blocksize": "blocksizex",
            "threadsperblock": "blocksizex",
            "tilesize": "tilesize",
            "tiledim": "tilesize",
            "numwarps": "numwarps",
            "nwarps": "numwarps",
            "unrollfactor": "unrollfactor",
            "unroll": "unrollfactor",
        }
        return mappings.get(code_name, code_name) == domain_name

    def get_best(self) -> ObservationPoint | None:
        """Get the best observation so far."""
        valid = [o for o in self.observations if o.correct]
        if not valid:
            return None
        return max(valid, key=lambda o: o.speedup)

    def get_belief_summary(self) -> dict:
        """Summary of the current posterior belief over HP space.

        Analogous to examining the belief vector in example.py.
        """
        if not self.surrogate.is_fitted:
            return {
                "n_observations": len(self.observations),
                "n_correct": sum(1 for o in self.observations if o.correct),
                "best_speedup": max((o.speedup for o in self.observations if o.correct), default=0),
                "phase": "exploration" if len(self.observations) < self.n_initial else "exploitation",
                "fitted": False,
            }

        mean, std = self.surrogate.predict(self._candidate_grid, self.domains)

        top_idx = np.argsort(mean)[-5:][::-1]
        top_configs = [
            {
                "config": self._candidate_grid[i].to_dict(),
                "predicted_speedup": float(mean[i]),
                "uncertainty": float(std[i]),
            }
            for i in top_idx
        ]

        return {
            "n_observations": len(self.observations),
            "n_correct": sum(1 for o in self.observations if o.correct),
            "best_speedup": max((o.speedup for o in self.observations if o.correct), default=0),
            "mean_predicted_speedup": float(np.mean(mean)),
            "mean_uncertainty": float(np.mean(std)),
            "phase": "exploitation",
            "fitted": True,
            "top_5_predicted": top_configs,
            "grid_size": len(self._candidate_grid),
        }
