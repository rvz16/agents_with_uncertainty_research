"""Gaussian Process surrogate model for speedup prediction.

The GP posterior plays the same role as compute_posterior() in
article_implementation/bayesian/example.py — it maintains a belief
over the objective function (speedup) given observations.
"""

import numpy as np
from .models import HyperparamConfig, ParameterDomain, ObservationPoint


class GPSurrogate:
    """GP surrogate model wrapping sklearn's GaussianProcessRegressor.

    Bayesian analogy to example.py:
    - Prior: GP kernel (Matern 5/2) = initial belief
    - Likelihood: observation noise model
    - Posterior: GP after fit() = updated belief
    - predict() returns (mean, std) = posterior belief summary
    """

    def __init__(
        self,
        noise: float = 1e-2,
        length_scale: float = 1.0,
        length_scale_bounds: tuple = (1e-2, 1e2),
    ):
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, WhiteKernel
        except ImportError:
            raise ImportError("scikit-learn required: pip install scikit-learn")

        kernel = Matern(
            nu=2.5,
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds,
        ) + WhiteKernel(noise_level=noise)

        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,  # Numerical stability
            normalize_y=True,
            n_restarts_optimizer=5,
        )
        self._fitted = False

    def fit(
        self,
        observations: list[ObservationPoint],
        domains: list[ParameterDomain],
    ) -> None:
        """Fit GP to observed (config, speedup) pairs.

        This is the Bayesian belief update step — analogous to
        compute_posterior() but for continuous function estimation.
        """
        if len(observations) < 2:
            self._fitted = False
            return

        # Only use correct observations for speedup modeling
        valid_obs = [o for o in observations if o.correct]
        if len(valid_obs) < 2:
            self._fitted = False
            return

        X = np.array([o.config.to_vector(domains) for o in valid_obs])
        y = np.array([o.speedup for o in valid_obs])

        self._gp.fit(X, y)
        self._fitted = True

    def predict(
        self,
        configs: list[HyperparamConfig],
        domains: list[ParameterDomain],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and std for a list of configs.

        Returns:
            (means, stds) — posterior belief at each point.
        """
        if not self._fitted:
            n = len(configs)
            return np.zeros(n), np.ones(n)  # Uninformative prior

        X = np.array([c.to_vector(domains) for c in configs])
        mean, std = self._gp.predict(X, return_std=True)
        return mean, std

    @property
    def is_fitted(self) -> bool:
        return self._fitted
