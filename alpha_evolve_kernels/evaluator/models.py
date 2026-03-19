"""Data models for KernelBench evaluation, aligned with the original KernelBench repo."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class KernelTask:
    """A kernel optimization task, mirroring KernelBench structure."""
    task_id: str
    reference_code: str  # Full Python file with class Model(nn.Module)
    level: int = 1       # KernelBench level: 1=single op, 2=fused ops, 3=full arch


@dataclass
class KernelExecResult:
    """Single kernel execution result — mirrors KernelBench's KernelExecResult.

    Original uses pydantic BaseModel; we use dataclass for zero dependencies.
    """
    compiled: bool = False
    correctness: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    runtime: float = -1.0           # Candidate mean runtime (ms)
    runtime_stats: dict[str, Any] = field(default_factory=dict)
    ref_runtime: float = -1.0       # Reference mean runtime (ms)
    ref_runtime_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def speedup(self) -> float:
        """ref_time / new_time. >1 means candidate is faster."""
        if self.runtime <= 0 or self.ref_runtime <= 0:
            return 0.0
        return self.ref_runtime / self.runtime

    @property
    def fast_p1(self) -> float:
        """KernelBench fast_p=1: correct AND speedup > 1."""
        return 1.0 if (self.correctness and self.speedup > 1.0) else 0.0

    def fast_at_p(self, p: float) -> float:
        """KernelBench fast_p metric: correct AND speedup > p."""
        return 1.0 if (self.correctness and self.speedup > p) else 0.0

    @property
    def score(self) -> float:
        """Combined score: speedup if correct, else 0."""
        return self.speedup if self.correctness else 0.0

    def summary(self) -> str:
        if not self.compiled:
            err = self.metadata.get("compilation_error", "unknown")
            return f"COMPILE_ERROR: {str(err)[:120]}"
        if not self.correctness:
            issue = self.metadata.get("correctness_issue",
                    self.metadata.get("runtime_error", "unknown"))
            return f"INCORRECT: {str(issue)[:120]}"
        return (f"correct=True speedup={self.speedup:.2f}x "
                f"ref={self.ref_runtime:.2f}ms new={self.runtime:.2f}ms")


# Keep EvalResult as alias for backward compatibility
EvalResult = KernelExecResult
