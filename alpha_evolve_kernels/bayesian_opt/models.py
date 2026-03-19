"""Data models for Bayesian hyperparameter optimization of CUDA kernels."""

import time
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ParameterDomain:
    """Domain for a single CUDA kernel hyperparameter.

    Represents discrete values that a hyperparameter can take.
    CUDA parameters are typically powers of 2 or small integers.
    """
    name: str
    values: list[int | float]
    log_scale: bool = False  # Whether to optimize in log space
    description: str = ""

    @property
    def size(self) -> int:
        return len(self.values)

    def encode(self, value: int | float) -> float:
        """Encode a value for the GP (optionally in log space)."""
        v = float(value)
        return np.log2(v) if self.log_scale and v > 0 else v

    def decode(self, encoded: float) -> int | float:
        """Decode from GP space back to original value. Snaps to nearest valid value."""
        if self.log_scale:
            target = 2.0 ** encoded
        else:
            target = encoded
        # Snap to nearest valid value
        return min(self.values, key=lambda v: abs(v - target))


# Common CUDA kernel hyperparameter domains
COMMON_DOMAINS = {
    "block_size_x": ParameterDomain(
        name="block_size_x",
        values=[32, 64, 128, 256, 512, 1024],
        log_scale=True,
        description="CUDA block size in X dimension",
    ),
    "block_size_y": ParameterDomain(
        name="block_size_y",
        values=[1, 2, 4, 8, 16, 32],
        log_scale=True,
        description="CUDA block size in Y dimension",
    ),
    "tile_size": ParameterDomain(
        name="tile_size",
        values=[8, 16, 32, 64, 128],
        log_scale=True,
        description="Tile size for tiled matrix operations",
    ),
    "num_warps": ParameterDomain(
        name="num_warps",
        values=[1, 2, 4, 8, 16],
        log_scale=True,
        description="Number of warps per block",
    ),
    "unroll_factor": ParameterDomain(
        name="unroll_factor",
        values=[1, 2, 4, 8],
        log_scale=False,
        description="Loop unrolling factor",
    ),
    "shared_mem_kb": ParameterDomain(
        name="shared_mem_kb",
        values=[0, 8, 16, 32, 48, 64, 96],
        log_scale=False,
        description="Shared memory allocation in KB",
    ),
    "elements_per_thread": ParameterDomain(
        name="elements_per_thread",
        values=[1, 2, 4, 8, 16],
        log_scale=True,
        description="Elements processed per thread",
    ),
}


@dataclass
class HyperparamConfig:
    """A specific point in the hyperparameter space."""
    values: dict[str, int | float] = field(default_factory=dict)

    def to_vector(self, domains: list[ParameterDomain]) -> np.ndarray:
        """Encode to a numeric vector for the GP."""
        vec = []
        for d in domains:
            val = self.values.get(d.name, d.values[len(d.values) // 2])  # default=median
            vec.append(d.encode(val))
        return np.array(vec, dtype=np.float64)

    @classmethod
    def from_vector(cls, vec: np.ndarray, domains: list[ParameterDomain]) -> "HyperparamConfig":
        """Decode from GP space to a config with valid values."""
        values = {}
        for i, d in enumerate(domains):
            values[d.name] = d.decode(vec[i])
        return cls(values=values)

    def to_dict(self) -> dict[str, int | float]:
        """Return plain dict for prompt injection."""
        return dict(self.values)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.values.items())
        return f"HyperparamConfig({params})"


@dataclass
class ObservationPoint:
    """A single observation: hyperparams -> measured speedup."""
    config: HyperparamConfig
    speedup: float
    correct: bool
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
