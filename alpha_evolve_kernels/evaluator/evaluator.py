"""Top-level KernelBench evaluator — aligned with eval_kernel_against_ref().

Mirrors the original flow:
1. Load original model + get_inputs/get_init_inputs via exec
2. Compile candidate (exec or tempfile depending on backend)
3. Instantiate both models on device
4. run_and_check_correctness() with deterministic seeds
5. Performance measurement (timing + ref timing)
6. Excessive speedup detection
7. graceful_eval_cleanup()
"""

import os
import torch
import warnings
import tempfile
import traceback

from .models import KernelTask, KernelExecResult
from .compiler import extract_and_compile, load_model_via_exec
from .correctness import run_and_check_correctness, set_seed, get_tolerance
from .profiler import measure_time_cuda_events, measure_time_cpu, _flush_l2_cache

import statistics


class KernelBenchEvaluator:
    """Evaluates candidate CUDA kernels against a reference PyTorch model."""

    def __init__(
        self,
        device: str | None = None,
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
        n_warmup: int = 3,
        precision: str = "float32",
        backend: str = "cuda",
        verbose: bool = False,
        check_excessive_speedup: bool = True,
        excessive_speedup_threshold: float = 10.0,
    ):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda", torch.cuda.current_device())
            else:
                warnings.warn("CUDA not available. Using CPU — timing won't reflect GPU performance.")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.num_correct_trials = num_correct_trials
        self.num_perf_trials = num_perf_trials
        self.n_warmup = n_warmup
        self.backend = backend
        self.verbose = verbose
        self.check_excessive_speedup = check_excessive_speedup
        self.excessive_speedup_threshold = excessive_speedup_threshold

        # Precision
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        self.precision = dtype_map.get(precision, torch.float32)

    def _load_reference(self, task: KernelTask) -> tuple[dict | None, str | None]:
        """Load reference code via exec — extract Model, get_inputs, get_init_inputs."""
        namespace = {}
        try:
            exec(task.reference_code, namespace)
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

        if "Model" not in namespace:
            return None, "Model class not found in reference code"

        return namespace, None

    def _graceful_cleanup(self, context: dict | None = None):
        """Clean up GPU cache and context — mirrors graceful_eval_cleanup()."""
        if context:
            context.clear()
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device=self.device)
                torch.cuda.synchronize(device=self.device)

    def evaluate(self, task: KernelTask, candidate_code: str, seed: int = 42) -> KernelExecResult:
        """Evaluate a single candidate — mirrors eval_kernel_against_ref().

        Args:
            task: KernelBench task with reference code.
            candidate_code: Python source defining ModelNew.
            seed: Random seed for reproducibility.

        Returns:
            KernelExecResult with compilation, correctness, and timing data.
        """
        metadata = {}
        if self.device.type == "cuda":
            metadata["hardware"] = torch.cuda.get_device_name(self.device)
        metadata["device"] = str(self.device)

        context = {}

        # 1. Load reference
        if self.verbose:
            print("[Eval] Loading Original Model")

        ref_ns, ref_err = self._load_reference(task)
        if ref_err:
            metadata["compilation_error"] = f"Reference: {ref_err}"
            return KernelExecResult(compiled=False, metadata=metadata)

        Model = ref_ns["Model"]
        get_inputs = ref_ns.get("get_inputs")
        get_init_inputs = ref_ns.get("get_init_inputs")

        if get_inputs is None:
            metadata["compilation_error"] = "get_inputs() not found"
            return KernelExecResult(compiled=False, metadata=metadata)

        # 2. Compile candidate
        if self.verbose:
            print("[Eval] Loading and Compiling New Model")

        cand_ns, cand_err = extract_and_compile(
            candidate_code, backend=self.backend
        )
        if cand_err:
            metadata["compilation_error"] = cand_err
            metadata["compilation_error_name"] = "CompilationError"
            self._graceful_cleanup(context)
            return KernelExecResult(compiled=False, metadata=metadata)

        ModelNew = cand_ns.get("ModelNew")
        if ModelNew is None:
            metadata["compilation_error"] = "ModelNew class not found"
            self._graceful_cleanup(context)
            return KernelExecResult(compiled=False, metadata=metadata)

        # 3. Instantiate models
        try:
            set_seed(seed)
            init_inputs = get_init_inputs() if get_init_inputs else []
            init_inputs = [
                x.to(device=self.device, dtype=self.precision) if isinstance(x, torch.Tensor) else x
                for x in init_inputs
            ]

            with torch.no_grad():
                set_seed(seed)
                original_model = Model(*init_inputs)
                set_seed(seed)
                custom_model = ModelNew(*init_inputs)

                original_model = original_model.to(device=self.device, dtype=self.precision)
                custom_model = custom_model.to(device=self.device, dtype=self.precision)

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)

        except Exception as e:
            metadata["runtime_error"] = str(e)
            metadata["runtime_error_name"] = type(e).__name__
            self._graceful_cleanup(context)
            return KernelExecResult(compiled=True, correctness=False, metadata=metadata)

        if self.verbose:
            print("[Eval] Checking Correctness")

        # 4. Correctness check
        try:
            result = run_and_check_correctness(
                original_model,
                custom_model,
                get_inputs,
                metadata=metadata,
                num_correct_trials=self.num_correct_trials,
                verbose=self.verbose,
                seed=seed,
                device=self.device,
                precision=self.precision,
            )
        except Exception as e:
            metadata["runtime_error"] = str(e)
            metadata["runtime_error_name"] = type(e).__name__
            self._graceful_cleanup(context)
            return KernelExecResult(compiled=True, correctness=False, metadata=metadata)

        # 5. Performance measurement (only if correct)
        if result.correctness:
            if self.verbose:
                print("[Eval] Measuring Performance")
            try:
                result = self._measure_performance(
                    result, original_model, custom_model, get_inputs, seed
                )
            except Exception as e:
                result.metadata["error_during_performance"] = str(e)
                if self.verbose:
                    print(f"[Eval] Performance error: {e}")

        self._graceful_cleanup(context)
        return result

    def _measure_performance(
        self,
        result: KernelExecResult,
        original_model: torch.nn.Module,
        custom_model: torch.nn.Module,
        get_inputs_fn: callable,
        seed: int,
    ) -> KernelExecResult:
        """Measure timing for both models — mirrors original performance section."""
        use_cuda = self.device.type == "cuda"
        device_str = str(self.device)

        def make_inputs():
            inputs = get_inputs_fn()
            return [
                x.to(device=self.device, dtype=self.precision) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]

        if use_cuda:
            # Candidate timing
            cand_mean, cand_times = measure_time_cuda_events(
                custom_model, make_inputs,
                n_warmup=self.n_warmup, n_trials=self.num_perf_trials,
                device=device_str, flush_cache=True,
            )
            # Reference timing
            ref_mean, ref_times = measure_time_cuda_events(
                original_model, make_inputs,
                n_warmup=self.n_warmup, n_trials=self.num_perf_trials,
                device=device_str, flush_cache=True,
            )
        else:
            cand_mean, cand_times = measure_time_cpu(
                custom_model, make_inputs,
                n_warmup=self.n_warmup, n_trials=self.num_perf_trials,
                device=device_str,
            )
            ref_mean, ref_times = measure_time_cpu(
                original_model, make_inputs,
                n_warmup=self.n_warmup, n_trials=self.num_perf_trials,
                device=device_str,
            )

        result.runtime = cand_mean
        result.runtime_stats = {
            "mean": cand_mean,
            "std": statistics.stdev(cand_times) if len(cand_times) > 1 else 0.0,
            "min": min(cand_times),
            "max": max(cand_times),
            "trials": len(cand_times),
        }
        result.ref_runtime = ref_mean
        result.ref_runtime_stats = {
            "mean": ref_mean,
            "std": statistics.stdev(ref_times) if len(ref_times) > 1 else 0.0,
            "min": min(ref_times),
            "max": max(ref_times),
            "trials": len(ref_times),
        }

        # Excessive speedup detection
        if self.check_excessive_speedup and result.speedup > self.excessive_speedup_threshold:
            result.metadata["excessive_speedup"] = True
            if self.verbose:
                print(f"[WARNING] Excessive speedup {result.speedup:.2f}x detected")

        return result

    def evaluate_batch(self, task: KernelTask, candidates: list[str]) -> list[KernelExecResult]:
        """Evaluate multiple candidates for the same task."""
        return [self.evaluate(task, code) for code in candidates]

    def get_feedback_str(self, result: KernelExecResult) -> str:
        """Generate human-readable feedback for LLM iterative refinement."""
        lines = []
        if not result.compiled:
            err = result.metadata.get("compilation_error", "unknown")
            lines.append(f"COMPILATION FAILED:\n{err}")
        elif not result.correctness:
            issue = result.metadata.get("correctness_issue",
                    result.metadata.get("runtime_error", "unknown"))
            lines.append(f"INCORRECT OUTPUT:\n{issue}")
            trials = result.metadata.get("correctness_trials", "")
            if trials:
                lines.append(f"Trials: {trials}")
            max_diffs = result.metadata.get("max_difference", [])
            if max_diffs:
                lines.append(f"Max differences: {max_diffs}")
        else:
            lines.append(f"Status: CORRECT")
            lines.append(f"Speedup: {result.speedup:.4f}x")
            lines.append(f"Candidate: {result.runtime:.4f} ms")
            lines.append(f"Reference: {result.ref_runtime:.4f} ms")
            if result.speedup > 1.0:
                lines.append("Kernel is faster than baseline. Try to improve further.")
            else:
                lines.append("Kernel is correct but slower. Optimize for speed.")
        return "\n".join(lines)
