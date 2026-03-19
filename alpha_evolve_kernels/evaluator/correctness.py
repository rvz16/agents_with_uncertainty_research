"""Correctness checking — aligned with KernelBench's run_and_check_correctness().

Key behaviors from original:
- Deterministic seeds per trial (torch.randint from initial seed)
- set_seed() before each get_inputs() and model instantiation
- Tolerance from precision (1e-4 fp32, 1e-2 fp16/bf16)
- Tracks max_difference and avg_difference in metadata
- Returns KernelExecResult
"""

import torch
import traceback

from .models import KernelExecResult


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_tolerance(precision: torch.dtype = torch.float32) -> float:
    """Get tolerance for a given precision — mirrors get_tolerance_for_precision()."""
    if precision in (torch.float16, torch.bfloat16):
        return 1e-2
    return 1e-4


def run_and_check_correctness(
    original_model: torch.nn.Module,
    custom_model: torch.nn.Module,
    get_inputs_fn: callable,
    metadata: dict,
    num_correct_trials: int = 5,
    verbose: bool = False,
    seed: int = 42,
    device: torch.device | str = "cuda",
    precision: torch.dtype = torch.float32,
) -> KernelExecResult:
    """Check functional equivalence — mirrors KernelBench's implementation.

    Generates deterministic seeds from initial seed, runs both models,
    compares outputs with torch.allclose().
    """
    pass_count = 0

    # Generate deterministic trial seeds (same as original)
    torch.manual_seed(seed)
    trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item()
        for _ in range(num_correct_trials)
    ]

    tolerance = get_tolerance(precision)

    with torch.no_grad():
        for trial in range(num_correct_trials):
            trial_seed = trial_seeds[trial]

            if verbose:
                print(f"[Eval] Trial {trial}, seed={trial_seed}")

            # Generate inputs with trial seed
            set_seed(trial_seed)
            inputs = get_inputs_fn()
            inputs = [
                x.to(device=device, dtype=precision) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]

            # Move models to device
            set_seed(trial_seed)
            model = original_model.to(device=device, dtype=precision)
            set_seed(trial_seed)
            model_new = custom_model.to(device=device, dtype=precision)

            # Run reference
            output = model(*inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device=device)

            # Run candidate
            try:
                output_new = model_new(*inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device=device)

                # Shape check
                if output.shape != output_new.shape:
                    metadata["correctness_issue"] = (
                        f"Shape mismatch: {output.shape} vs {output_new.shape}"
                    )
                    metadata["correctness_issue_name"] = "correctness_issue"
                    if verbose:
                        print(f"[FAIL] Trial {trial}: Shape mismatch")
                    return KernelExecResult(
                        compiled=True, correctness=False, metadata=metadata
                    )

                # Value check
                if not torch.allclose(output, output_new, atol=tolerance, rtol=tolerance):
                    max_diff = torch.max(torch.abs(output - output_new)).item()
                    avg_diff = torch.mean(torch.abs(output - output_new)).item()
                    metadata.setdefault("max_difference", []).append(f"{max_diff:.6f}")
                    metadata.setdefault("avg_difference", []).append(f"{avg_diff:.6f}")
                    metadata["correctness_issue"] = "Output mismatch"
                    if verbose:
                        print(f"[FAIL] Trial {trial}: max_diff={max_diff:.6f}")
                else:
                    pass_count += 1
                    if verbose:
                        print(f"[PASS] Trial {trial}")

            except Exception as e:
                metadata["runtime_error"] = str(e)
                metadata["runtime_error_name"] = type(e).__name__
                metadata["runtime_error_traceback"] = traceback.format_exc()
                if verbose:
                    print(f"[ERROR] Trial {trial}: {e}")
                return KernelExecResult(
                    compiled=True, correctness=False, metadata=metadata
                )

    metadata["correctness_trials"] = f"({pass_count} / {num_correct_trials})"

    if verbose:
        print(f"[Eval] Pass: {pass_count}/{num_correct_trials}")

    return KernelExecResult(
        compiled=True,
        correctness=(pass_count == num_correct_trials),
        metadata=metadata,
    )
