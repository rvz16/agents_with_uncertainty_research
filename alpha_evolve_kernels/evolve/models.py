"""Data models for the AlphaEvolve evolutionary loop."""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from ..evaluator.models import KernelExecResult, EvalResult


@dataclass
class Program:
    """A program (kernel candidate) in the evolutionary database."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    code: str = ""
    parent_id: str | None = None
    generation: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgramScore:
    """A program paired with its evaluation result."""
    program: Program
    eval_result: KernelExecResult

    @property
    def fitness(self) -> float:
        """Primary fitness = speedup if correct, else 0."""
        return self.eval_result.score

    @property
    def is_valid(self) -> bool:
        """Program compiled without errors."""
        return self.eval_result.compiled

    def summary(self) -> str:
        return (
            f"[gen={self.program.generation} id={self.program.id}] "
            f"fitness={self.fitness:.3f} {self.eval_result.summary()}"
        )


@dataclass
class IslandConfig:
    """Configuration for a MAP-Elites island (feature dimension).

    Feature functions map (Program, EvalResult) -> float for binning.
    Example: code length, number of CUDA kernels, use of shared memory, etc.
    """
    name: str
    feature_fn: Callable[[Program, EvalResult], float]
    bins: list[float]  # Bin boundaries (N boundaries -> N+1 bins)

    def get_bin(self, program: Program, eval_result: EvalResult) -> int:
        """Return bin index for a program."""
        val = self.feature_fn(program, eval_result)
        for i, boundary in enumerate(self.bins):
            if val < boundary:
                return i
        return len(self.bins)
