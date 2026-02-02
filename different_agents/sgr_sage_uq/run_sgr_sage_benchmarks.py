"""Benchmark evaluation for SGR-SAGE-UQ Agent.

Evaluates on:
1. BFCL (Berkeley Function Calling Leaderboard) - Tool calling accuracy
2. GSM8K - Math reasoning with uncertainty
3. HumanEval - Code generation with uncertainty

The key insight: Uncertainty should correlate with correctness.
- High confidence + correct = good calibration
- High uncertainty + wrong = appropriate caution
- High confidence + wrong = overconfident (bad)

Usage:
    # Test on BFCL (function calling)
    python run_sgr_sage_benchmarks.py --benchmark bfcl --limit 20

    # Test on GSM8K (math)
    python run_sgr_sage_benchmarks.py --benchmark gsm8k --limit 50

    # Test on HumanEval (code)
    python run_sgr_sage_benchmarks.py --benchmark humaneval --limit 20

    # All benchmarks
    python run_sgr_sage_benchmarks.py --benchmark all --limit 10
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

# Project root is 2 levels up from different_agents/sgr_sage_uq/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = Path(__file__).resolve().parent
SHARED_DIR = PROJECT_ROOT / "different_agents" / "shared"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LOCAL_DIR))
sys.path.insert(0, str(SHARED_DIR))

from sgr_sage_uq_agent import (
    TTSConfig,
    SelfConsistencyClient,
    SAGEConfig,
    ActionSchema,
    get_self_consistency_samples,
)
from sage_agent import ParameterDomain, ToolSchema, UNK
from sage_agent.core.uncertainty_propagation import create_sage_propagator


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SGR-SAGE-UQ Benchmark Evaluation")

    parser.add_argument(
        "--benchmark",
        choices=["bfcl", "gsm8k", "humaneval", "all"],
        default="gsm8k",
        help="Benchmark to evaluate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of examples (0=all)",
    )
    parser.add_argument(
        "--service-url",
        default=os.getenv("TTS_SERVICE_URL", "http://localhost:8001/v1"),
        help="TTS service URL",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("TTS_MODEL", "openai/gpt-4o-mini"),
        help="Model name",
    )
    parser.add_argument(
        "--tts-budget",
        type=int,
        default=8,
        help="Number of samples for self-consistency",
    )
    parser.add_argument(
        "--print-each",
        action="store_true",
        help="Print each example result",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )

    return parser.parse_args()


# ============================================================================
# Result Tracking
# ============================================================================

@dataclass
class ExampleResult:
    """Result for a single example."""
    example_id: str
    correct: bool
    uncertainty: float
    confidence: float
    prediction: str
    expected: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""
    benchmark_name: str
    total_examples: int
    accuracy: float
    avg_uncertainty: float

    # Calibration metrics
    confident_accuracy: float  # Accuracy when uncertainty < 0.3
    uncertain_accuracy: float  # Accuracy when uncertainty > 0.7
    ece: float  # Expected Calibration Error

    # Uncertainty quality
    auroc: float  # How well uncertainty predicts errors

    examples: List[ExampleResult] = field(default_factory=list)


def compute_ece(results: List[ExampleResult], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    if not results:
        return 0.0

    bins = [[] for _ in range(n_bins)]
    for r in results:
        bin_idx = min(int(r.confidence * n_bins), n_bins - 1)
        bins[bin_idx].append(r)

    ece = 0.0
    total = len(results)

    for bin_results in bins:
        if not bin_results:
            continue
        avg_conf = sum(r.confidence for r in bin_results) / len(bin_results)
        avg_acc = sum(1 for r in bin_results if r.correct) / len(bin_results)
        ece += (len(bin_results) / total) * abs(avg_conf - avg_acc)

    return ece


def compute_auroc(results: List[ExampleResult]) -> float:
    """Compute AUROC for uncertainty predicting errors."""
    if len(results) < 2:
        return 0.5

    # Sort by uncertainty (high uncertainty should predict errors)
    sorted_results = sorted(results, key=lambda r: r.uncertainty, reverse=True)

    # Count correct and incorrect
    n_correct = sum(1 for r in results if r.correct)
    n_incorrect = len(results) - n_correct

    if n_correct == 0 or n_incorrect == 0:
        return 0.5

    # Compute AUROC
    rank_sum = 0
    for i, r in enumerate(sorted_results):
        if not r.correct:  # Incorrect predictions should have high uncertainty
            rank_sum += i + 1

    auroc = (rank_sum - n_incorrect * (n_incorrect + 1) / 2) / (n_correct * n_incorrect)
    return auroc


# ============================================================================
# GSM8K Evaluation
# ============================================================================

def extract_gsm8k_answer(text: str) -> Optional[float]:
    """Extract numeric answer from GSM8K response."""
    # Look for #### format
    match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if match:
        return float(match.group(1).replace(',', ''))

    # Look for "answer is X" or "= X"
    match = re.search(r'(?:answer is|=)\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text, re.I)
    if match:
        return float(match.group(1).replace(',', ''))

    # Find last number
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        return float(numbers[-1].replace(',', ''))

    return None


def evaluate_gsm8k(args: argparse.Namespace, tts_client: SelfConsistencyClient) -> BenchmarkResult:
    """Evaluate on GSM8K math benchmark."""
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("Evaluating on GSM8K (Math Reasoning)")
    print("=" * 60)

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    results: List[ExampleResult] = []

    for i, example in enumerate(dataset):
        question = example["question"]
        answer_text = example["answer"]

        # Extract expected answer
        expected_match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', answer_text)
        expected = float(expected_match.group(1).replace(',', '')) if expected_match else 0

        # Create prompt
        prompt = f"""Solve this math problem step by step. End with "#### [answer]"

Question: {question}

Solution:"""

        # Get response with uncertainty
        samples, metadata = tts_client.get_samples(prompt, n=args.tts_budget)
        response = samples[0] if samples else ""
        uncertainty = metadata.get("uncertainty_score", 0.5)

        # Extract predicted answer
        predicted = extract_gsm8k_answer(response)
        correct = predicted is not None and abs(predicted - expected) < 0.01

        result = ExampleResult(
            example_id=f"gsm8k_{i}",
            correct=correct,
            uncertainty=uncertainty,
            confidence=1.0 - uncertainty,
            prediction=str(predicted),
            expected=str(expected),
            metadata={"question": question[:100]},
        )
        results.append(result)

        if args.print_each:
            status = "✓" if correct else "✗"
            print(f"[{i+1}/{len(dataset)}] {status} pred={predicted} exp={expected} unc={uncertainty:.2f}")

    # Compute metrics
    accuracy = sum(1 for r in results if r.correct) / len(results) if results else 0
    avg_uncertainty = sum(r.uncertainty for r in results) / len(results) if results else 0

    confident = [r for r in results if r.uncertainty < 0.3]
    uncertain = [r for r in results if r.uncertainty > 0.7]

    confident_acc = sum(1 for r in confident if r.correct) / len(confident) if confident else 0
    uncertain_acc = sum(1 for r in uncertain if r.correct) / len(uncertain) if uncertain else 0

    return BenchmarkResult(
        benchmark_name="GSM8K",
        total_examples=len(results),
        accuracy=accuracy,
        avg_uncertainty=avg_uncertainty,
        confident_accuracy=confident_acc,
        uncertain_accuracy=uncertain_acc,
        ece=compute_ece(results),
        auroc=compute_auroc(results),
        examples=results,
    )


# ============================================================================
# HumanEval Evaluation
# ============================================================================

def execute_humaneval_test(code: str, test: str, entry_point: str, timeout: float = 5.0) -> bool:
    """Execute HumanEval test."""
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError()

    try:
        # Extract function
        func_match = re.search(
            rf'(def\s+{re.escape(entry_point)}\s*\([^)]*\).*?)(?=\ndef\s|\nclass\s|\Z)',
            code, re.DOTALL
        )
        if func_match:
            clean_code = func_match.group(1)
        else:
            # Try code block
            block_match = re.search(r'```python\n(.*?)```', code, re.DOTALL)
            clean_code = block_match.group(1) if block_match else code

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))

        namespace = {}
        exec(clean_code, namespace)
        exec(test, namespace)

        signal.alarm(0)
        return True

    except Exception:
        signal.alarm(0)
        return False


def evaluate_humaneval(args: argparse.Namespace, tts_client: SelfConsistencyClient) -> BenchmarkResult:
    """Evaluate on HumanEval code benchmark."""
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("Evaluating on HumanEval (Code Generation)")
    print("=" * 60)

    dataset = load_dataset("openai_humaneval", split="test")
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    results: List[ExampleResult] = []

    for i, example in enumerate(dataset):
        task_id = example["task_id"]
        prompt = example["prompt"]
        test = example["test"]
        entry_point = example["entry_point"]

        # Create completion prompt
        completion_prompt = f"""Complete the following Python function. Only output the code, no explanation.

{prompt}"""

        # Get response with uncertainty
        samples, metadata = tts_client.get_samples(completion_prompt, n=args.tts_budget)
        response = samples[0] if samples else ""
        uncertainty = metadata.get("uncertainty_score", 0.5)

        # Test execution
        correct = execute_humaneval_test(response, test, entry_point)

        result = ExampleResult(
            example_id=task_id,
            correct=correct,
            uncertainty=uncertainty,
            confidence=1.0 - uncertainty,
            prediction=response[:200],
            expected=entry_point,
            metadata={"prompt": prompt[:100]},
        )
        results.append(result)

        if args.print_each:
            status = "✓" if correct else "✗"
            print(f"[{i+1}/{len(dataset)}] {status} {task_id} unc={uncertainty:.2f}")

    # Compute metrics
    accuracy = sum(1 for r in results if r.correct) / len(results) if results else 0
    avg_uncertainty = sum(r.uncertainty for r in results) / len(results) if results else 0

    confident = [r for r in results if r.uncertainty < 0.3]
    uncertain = [r for r in results if r.uncertainty > 0.7]

    confident_acc = sum(1 for r in confident if r.correct) / len(confident) if confident else 0
    uncertain_acc = sum(1 for r in uncertain if r.correct) / len(uncertain) if uncertain else 0

    return BenchmarkResult(
        benchmark_name="HumanEval",
        total_examples=len(results),
        accuracy=accuracy,
        avg_uncertainty=avg_uncertainty,
        confident_accuracy=confident_acc,
        uncertain_accuracy=uncertain_acc,
        ece=compute_ece(results),
        auroc=compute_auroc(results),
        examples=results,
    )


# ============================================================================
# BFCL Evaluation (Function Calling)
# ============================================================================

def evaluate_bfcl(args: argparse.Namespace, tts_client: SelfConsistencyClient) -> BenchmarkResult:
    """Evaluate on BFCL function calling benchmark."""
    print("\n" + "=" * 60)
    print("Evaluating on BFCL (Function Calling)")
    print("=" * 60)

    # BFCL dataset loading
    try:
        from datasets import load_dataset
        dataset = load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard", split="test")
    except Exception as e:
        print(f"Could not load BFCL dataset: {e}")
        print("Using synthetic function calling examples instead...")
        return evaluate_synthetic_function_calling(args, tts_client)

    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    results: List[ExampleResult] = []

    for i, example in enumerate(dataset):
        # Parse BFCL example
        question = example.get("question", "")
        functions = example.get("function", [])
        expected = example.get("ground_truth", "")

        # Build tool schemas from functions
        tool_schemas = {}
        for func in functions[:5]:  # Limit functions
            if isinstance(func, dict):
                name = func.get("name", f"func_{i}")
                params = func.get("parameters", {}).get("properties", {})
                tool_schemas[name] = ToolSchema(
                    name=name,
                    parameters={p: ParameterDomain.continuous() for p in params},
                    required=frozenset(func.get("parameters", {}).get("required", [])),
                )

        if not tool_schemas:
            continue

        # Get samples
        prompt = f"""Given this question, call the appropriate function with correct arguments.

Question: {question}

Available functions: {list(tool_schemas.keys())}

Respond with JSON: {{"function": "name", "arguments": {{...}}}}"""

        samples, metadata = tts_client.get_samples(prompt, n=args.tts_budget)
        response = samples[0] if samples else ""
        uncertainty = metadata.get("uncertainty_score", 0.5)

        # Parse response
        try:
            # Extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                predicted_func = parsed.get("function", "")
            else:
                predicted_func = ""
        except Exception:
            predicted_func = ""

        # Check correctness (simplified)
        correct = predicted_func in str(expected)

        result = ExampleResult(
            example_id=f"bfcl_{i}",
            correct=correct,
            uncertainty=uncertainty,
            confidence=1.0 - uncertainty,
            prediction=predicted_func,
            expected=str(expected)[:100],
        )
        results.append(result)

        if args.print_each:
            status = "✓" if correct else "✗"
            print(f"[{i+1}/{len(dataset)}] {status} pred={predicted_func} unc={uncertainty:.2f}")

    # Compute metrics
    accuracy = sum(1 for r in results if r.correct) / len(results) if results else 0
    avg_uncertainty = sum(r.uncertainty for r in results) / len(results) if results else 0

    confident = [r for r in results if r.uncertainty < 0.3]
    uncertain = [r for r in results if r.uncertainty > 0.7]

    confident_acc = sum(1 for r in confident if r.correct) / len(confident) if confident else 0
    uncertain_acc = sum(1 for r in uncertain if r.correct) / len(uncertain) if uncertain else 0

    return BenchmarkResult(
        benchmark_name="BFCL",
        total_examples=len(results),
        accuracy=accuracy,
        avg_uncertainty=avg_uncertainty,
        confident_accuracy=confident_acc,
        uncertain_accuracy=uncertain_acc,
        ece=compute_ece(results),
        auroc=compute_auroc(results),
        examples=results,
    )


def evaluate_synthetic_function_calling(args: argparse.Namespace, tts_client: SelfConsistencyClient) -> BenchmarkResult:
    """Synthetic function calling evaluation when BFCL not available."""
    print("Using synthetic function calling examples...")

    # Define test cases
    test_cases = [
        {
            "question": "Book a flight from New York to Los Angeles on March 15th",
            "expected_func": "book_flight",
            "expected_args": {"origin": "NYC", "destination": "LAX", "date": "2024-03-15"},
        },
        {
            "question": "What's the weather in Tokyo?",
            "expected_func": "get_weather",
            "expected_args": {"city": "Tokyo"},
        },
        {
            "question": "Send an email to john@example.com about the meeting",
            "expected_func": "send_email",
            "expected_args": {"to": "john@example.com", "subject": "meeting"},
        },
        {
            "question": "Calculate 15% tip on $85.50",
            "expected_func": "calculate_tip",
            "expected_args": {"amount": 85.50, "percentage": 15},
        },
        {
            "question": "Set a reminder for tomorrow at 3pm",
            "expected_func": "set_reminder",
            "expected_args": {"time": "15:00", "date": "tomorrow"},
        },
    ]

    # Duplicate to match limit
    while len(test_cases) < args.limit:
        test_cases = test_cases + test_cases

    test_cases = test_cases[:args.limit]

    results: List[ExampleResult] = []

    for i, tc in enumerate(test_cases):
        prompt = f"""Call the appropriate function for this request.

Request: {tc['question']}

Available functions: book_flight, get_weather, send_email, calculate_tip, set_reminder

Respond with JSON: {{"function": "name", "arguments": {{...}}}}"""

        samples, metadata = tts_client.get_samples(prompt, n=args.tts_budget)
        response = samples[0] if samples else ""
        uncertainty = metadata.get("uncertainty_score", 0.5)

        # Parse response
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                predicted_func = parsed.get("function", "")
            else:
                predicted_func = ""
        except Exception:
            predicted_func = ""

        correct = predicted_func == tc["expected_func"]

        result = ExampleResult(
            example_id=f"synthetic_{i}",
            correct=correct,
            uncertainty=uncertainty,
            confidence=1.0 - uncertainty,
            prediction=predicted_func,
            expected=tc["expected_func"],
        )
        results.append(result)

        if args.print_each:
            status = "✓" if correct else "✗"
            print(f"[{i+1}/{len(test_cases)}] {status} pred={predicted_func} exp={tc['expected_func']} unc={uncertainty:.2f}")

    # Compute metrics
    accuracy = sum(1 for r in results if r.correct) / len(results) if results else 0
    avg_uncertainty = sum(r.uncertainty for r in results) / len(results) if results else 0

    confident = [r for r in results if r.uncertainty < 0.3]
    uncertain = [r for r in results if r.uncertainty > 0.7]

    confident_acc = sum(1 for r in confident if r.correct) / len(confident) if confident else 0
    uncertain_acc = sum(1 for r in uncertain if r.correct) / len(uncertain) if uncertain else 0

    return BenchmarkResult(
        benchmark_name="BFCL (Synthetic)",
        total_examples=len(results),
        accuracy=accuracy,
        avg_uncertainty=avg_uncertainty,
        confident_accuracy=confident_acc,
        uncertain_accuracy=uncertain_acc,
        ece=compute_ece(results),
        auroc=compute_auroc(results),
        examples=results,
    )


# ============================================================================
# Main
# ============================================================================

def print_benchmark_result(result: BenchmarkResult):
    """Print benchmark results."""
    print("\n" + "=" * 60)
    print(f"Results: {result.benchmark_name}")
    print("=" * 60)

    print(f"\nAccuracy: {result.accuracy:.1%} ({sum(1 for r in result.examples if r.correct)}/{result.total_examples})")
    print(f"Avg Uncertainty: {result.avg_uncertainty:.3f}")

    print(f"\nCalibration:")
    print(f"  ECE (Expected Calibration Error): {result.ece:.3f}")
    print(f"  AUROC (Uncertainty → Errors): {result.auroc:.3f}")

    print(f"\nUncertainty Breakdown:")
    confident_n = sum(1 for r in result.examples if r.uncertainty < 0.3)
    uncertain_n = sum(1 for r in result.examples if r.uncertainty > 0.7)
    print(f"  Confident (unc<0.3): {result.confident_accuracy:.1%} accuracy ({confident_n} examples)")
    print(f"  Uncertain (unc>0.7): {result.uncertain_accuracy:.1%} accuracy ({uncertain_n} examples)")

    # Interpretation
    print(f"\nInterpretation:")
    if result.auroc > 0.6:
        print(f"  ✓ Uncertainty is predictive of errors (AUROC={result.auroc:.2f})")
    else:
        print(f"  ✗ Uncertainty is not well calibrated (AUROC={result.auroc:.2f})")

    if result.confident_accuracy > result.accuracy:
        print(f"  ✓ High confidence correlates with correctness")
    else:
        print(f"  ✗ Model is overconfident on some predictions")


def main():
    args = parse_args()

    print("=" * 60)
    print("SGR-SAGE-UQ Benchmark Evaluation")
    print("=" * 60)
    print(f"TTS URL: {args.service_url}")
    print(f"Model: {args.model}")
    print(f"TTS Budget: {args.tts_budget}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Limit: {args.limit}")

    # Create TTS client
    tts_config = TTSConfig(
        service_url=args.service_url,
        model=args.model,
        tts_budget=args.tts_budget,
    )
    tts_client = SelfConsistencyClient(tts_config)

    # Test connection
    try:
        samples, metadata = tts_client.get_samples("Test", n=1)
        print(f"TTS connection OK (uncertainty={metadata.get('uncertainty_score', 'N/A')})")
    except Exception as e:
        print(f"TTS connection failed: {e}")
        return

    # Run benchmarks
    all_results: List[BenchmarkResult] = []

    if args.benchmark in ["gsm8k", "all"]:
        result = evaluate_gsm8k(args, tts_client)
        all_results.append(result)
        print_benchmark_result(result)

    if args.benchmark in ["humaneval", "all"]:
        result = evaluate_humaneval(args, tts_client)
        all_results.append(result)
        print_benchmark_result(result)

    if args.benchmark in ["bfcl", "all"]:
        result = evaluate_bfcl(args, tts_client)
        all_results.append(result)
        print_benchmark_result(result)

    # Summary
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        for r in all_results:
            print(f"  {r.benchmark_name}: {r.accuracy:.1%} acc, {r.avg_uncertainty:.2f} unc, {r.auroc:.2f} AUROC")

    # Save results
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "service_url": args.service_url,
                "model": args.model,
                "tts_budget": args.tts_budget,
            },
            "results": [
                {
                    "benchmark": r.benchmark_name,
                    "accuracy": r.accuracy,
                    "avg_uncertainty": r.avg_uncertainty,
                    "ece": r.ece,
                    "auroc": r.auroc,
                    "confident_accuracy": r.confident_accuracy,
                    "uncertain_accuracy": r.uncertain_accuracy,
                    "total_examples": r.total_examples,
                }
                for r in all_results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
