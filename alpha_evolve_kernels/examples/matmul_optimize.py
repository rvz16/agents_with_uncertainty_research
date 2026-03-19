"""Example: Optimize a matrix multiplication kernel using the full pipeline.

This mirrors the KernelBench Appendix A task format — a simple matmul Model
that we want to replace with an optimized CUDA kernel via ModelNew.

Usage:
    # With a local LLM (e.g., via vLLM or ollama):
    python -m alpha_evolve_kernels.examples.matmul_optimize \\
        --base-url http://localhost:8000/v1 \\
        --model "deepseek-r1" \\
        --iterations 10

    # With OpenAI:
    OPENAI_API_KEY=... python -m alpha_evolve_kernels.examples.matmul_optimize \\
        --model "gpt-4o" \\
        --iterations 10
"""

import argparse
import logging
import sys

from alpha_evolve_kernels.evaluator.models import KernelTask
from alpha_evolve_kernels.pipeline import AlphaEvolveKernelPipeline, PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ─── KernelBench-style reference code (Appendix A from paper) ───

REFERENCE_CODE = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    """Simple model that performs a single matrix multiplication (C = A * B)
    with a large K dimension."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Performs matrix multiplication of A and B."""
        return torch.matmul(A, B)

M = 256
N = 256
K = 131072

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []
'''

# ─── Initial naive ModelNew (starting point for evolution) ───

INITIAL_CANDIDATE = '''
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """Naive ModelNew — just wraps torch.matmul (same as reference).
    The evolutionary loop will optimize this."""

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)
'''


def main():
    parser = argparse.ArgumentParser(description="Optimize matmul kernel via AlphaEvolve")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--no-bayesian", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task = KernelTask(
        task_id="matmul_large_k",
        reference_code=REFERENCE_CODE,
        level=1,
    )

    config = PipelineConfig(
        max_iterations=args.iterations,
        candidates_per_iteration=args.candidates,
        seed=args.seed,
        use_bayesian_opt=not args.no_bayesian,
        device=args.device,
    )

    pipeline = AlphaEvolveKernelPipeline(
        task=task,
        llm_config={
            "base_url": args.base_url,
            "model": args.model,
            "api_key": args.api_key,
        },
        config=config,
    )

    result = pipeline.run(INITIAL_CANDIDATE)

    # Print results
    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE")
    print("=" * 60)
    print(f"Total evaluations: {result.total_evaluations}")
    print(f"Total time: {result.total_time_s:.1f}s")
    print(f"Database stats: {result.db_stats}")

    if result.best:
        print(f"\nBest program: {result.best.summary()}")
        print(f"\n--- Best Code ---\n{result.best.program.code}")
    else:
        print("\nNo successful programs found.")


if __name__ == "__main__":
    main()
'''
