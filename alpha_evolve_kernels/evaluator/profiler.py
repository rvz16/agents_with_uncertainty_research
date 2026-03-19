"""Performance profiling with L2 cache flushing (mirrors KernelBench timing.py)."""

import torch
import time
import statistics


def _flush_l2_cache(device: str = "cuda"):
    """Flush GPU L2 cache by thrashing with a large tensor.

    KernelBench uses 256MB to cover modern GPU L2 caches
    (A100=40MB, H100=50MB, L40S=96MB, B200=128MB).
    """
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return
    try:
        flush_size = 256 * 1024 * 1024 // 4  # 256MB of float32
        flush_tensor = torch.zeros(flush_size, dtype=torch.float32, device=device)
        del flush_tensor
    except Exception:
        pass


def get_tolerance(precision: str = "float32") -> tuple[float, float]:
    """Get atol/rtol for a given precision.

    Mirrors KernelBench's get_tolerance_for_precision():
    - float32: 1e-4
    - float16/bfloat16: 1e-2
    """
    if precision in ("float16", "bfloat16"):
        return 1e-2, 1e-2
    return 1e-4, 1e-4


def measure_time_cuda_events(
    model: torch.nn.Module,
    get_inputs_fn: callable,
    n_warmup: int = 3,
    n_trials: int = 100,
    device: str = "cuda",
    flush_cache: bool = True,
) -> tuple[float, list[float]]:
    """Measure model execution time using CUDA events.

    Args:
        model: Model to time.
        get_inputs_fn: Callable returning input tensors.
        n_warmup: Warmup iterations (not timed).
        n_trials: Timed iterations.
        device: Device string.
        flush_cache: Whether to flush L2 cache between trials.

    Returns:
        (mean_time_ms, per_trial_times_ms)
    """
    model = model.to(device).eval()

    # Warmup
    for _ in range(n_warmup):
        inputs = get_inputs_fn()
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
        with torch.no_grad():
            model(*inputs)
    torch.cuda.synchronize()

    # Timed trials
    times = []
    for _ in range(n_trials):
        if flush_cache:
            _flush_l2_cache(device)

        inputs = get_inputs_fn()
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            model(*inputs)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    mean_ms = statistics.mean(times)
    return mean_ms, times


def measure_time_cpu(
    model: torch.nn.Module,
    get_inputs_fn: callable,
    n_warmup: int = 3,
    n_trials: int = 100,
    device: str = "cpu",
) -> tuple[float, list[float]]:
    """Measure model execution time using time.perf_counter (CPU fallback)."""
    model = model.to(device).eval()

    for _ in range(n_warmup):
        inputs = get_inputs_fn()
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
        with torch.no_grad():
            model(*inputs)

    times = []
    for _ in range(n_trials):
        inputs = get_inputs_fn()
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

        start = time.perf_counter()
        with torch.no_grad():
            model(*inputs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times.append(elapsed_ms)

    mean_ms = statistics.mean(times)
    return mean_ms, times


def measure_speedup(
    ref_model: torch.nn.Module,
    new_model: torch.nn.Module,
    get_inputs_fn: callable,
    n_warmup: int = 3,
    n_trials: int = 100,
    device: str = "cuda",
    flush_cache: bool = True,
) -> tuple[float, float, float, str]:
    """Measure wall-clock speedup of new_model over ref_model.

    Returns:
        (ref_time_ms, new_time_ms, speedup, profiler_output_str)
    """
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    time_fn = measure_time_cuda_events if use_cuda else measure_time_cpu

    kwargs = dict(
        get_inputs_fn=get_inputs_fn,
        n_warmup=n_warmup,
        n_trials=n_trials,
        device=device,
    )
    if use_cuda:
        kwargs["flush_cache"] = flush_cache

    ref_mean, ref_times = time_fn(model=ref_model, **kwargs)
    new_mean, new_times = time_fn(model=new_model, **kwargs)

    speedup = ref_mean / new_mean if new_mean > 0 else 0.0

    profiler_output = (
        f"Reference: {ref_mean:.4f} ms (std={statistics.stdev(ref_times):.4f})\n"
        f"Candidate: {new_mean:.4f} ms (std={statistics.stdev(new_times):.4f})\n"
        f"Speedup:   {speedup:.4f}x\n"
        f"Trials:    {n_trials}\n"
        f"Device:    {device} ({'CUDA events' if use_cuda else 'CPU timer'})\n"
        f"L2 flush:  {flush_cache and use_cuda}"
    )

    return ref_mean, new_mean, speedup, profiler_output


def compute_fast_p_batch(results: list[dict], p: float = 1.0) -> float:
    """Compute KernelBench fast_p metric over a batch.

    fast_p = (1/N) * sum(correct_i AND speedup_i > p)
    """
    if not results:
        return 0.0
    n_fast = sum(
        1 for r in results
        if r.get("correct", False) and r.get("speedup", 0.0) > p
    )
    return n_fast / len(results)
