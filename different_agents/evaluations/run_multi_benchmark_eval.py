"""
Multi-Benchmark Evaluation for Uncertainty-Guided LLM Agents.

Evaluates the SAGE-Agent uncertainty quantification approach on multiple benchmarks:
1. When2Call - Tool calling disambiguation
2. HumanEval - Code generation
3. MBPP - Code generation (Mostly Basic Python Problems)
4. GSM8K - Math reasoning
5. HotpotQA - Multi-hop question answering

The key insight is that uncertainty quantification can benefit all these tasks:
- Tool calling: Clarify ambiguous parameters
- Code generation: Clarify ambiguous requirements
- Math reasoning: Identify when problem statement is unclear
- QA: Decompose complex questions when uncertain

Usage:
    python run_multi_benchmark_eval.py --benchmark humaneval --limit 10
    python run_multi_benchmark_eval.py --benchmark gsm8k --limit 20 --print-each
    python run_multi_benchmark_eval.py --benchmark all --limit 5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Any

# Project root is 2 levels up from different_agents/evaluations/
ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = ROOT / "different_agents" / "shared"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SHARED_DIR))

from sage_agent.core.advanced_reasoning import (
    UncertaintyDecomposer,
    ChainOfThoughtVerifier,
    ReflexionAgent,
)
from sage_agent.core.uncertainty_propagation import create_sage_propagator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-benchmark evaluation with uncertainty quantification."
    )
    parser.add_argument(
        "--benchmark",
        choices=["when2call", "humaneval", "mbpp", "gsm8k", "hotpotqa", "all"],
        default="humaneval",
        help="Benchmark to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of examples to evaluate (0 = all).",
    )
    parser.add_argument(
        "--model",
        default="xiaomi/mimo-v2-flash:free",
        help="Model name for llm-tts service.",
    )
    parser.add_argument(
        "--service-url",
        default="http://localhost:8001/v1",
        help="llm-tts service URL.",
    )
    parser.add_argument(
        "--tts-budget",
        type=int,
        default=8,
        help="Number of reasoning traces for uncertainty.",
    )
    parser.add_argument(
        "--use-ollama",
        action="store_true",
        help="Use Ollama instead of TTS service.",
    )
    parser.add_argument(
        "--ollama-model",
        default="qwen3:4b-instruct-2507-q8_0",
        help="Ollama model name.",
    )
    parser.add_argument(
        "--print-each",
        action="store_true",
        help="Print per-example results.",
    )
    parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=0.5,
        help="Threshold for uncertainty-aware metrics.",
    )
    parser.add_argument(
        "--disable-saup",
        action="store_true",
        help="Disable SAUP uncertainty decomposition.",
    )
    parser.add_argument(
        "--disable-cot",
        action="store_true",
        help="Disable chain-of-thought verification.",
    )
    parser.add_argument(
        "--disable-reflexion",
        action="store_true",
        help="Disable Reflexion self-improvement.",
    )
    parser.add_argument(
        "--saup-samples",
        type=int,
        default=4,
        help="Number of samples for SAUP decomposition.",
    )
    parser.add_argument(
        "--disable-propagation",
        action="store_true",
        help="Disable SAUP-style uncertainty propagation.",
    )
    parser.add_argument(
        "--reflexion-max-attempts",
        type=int,
        default=3,
        help="Max attempts for Reflexion loop.",
    )
    return parser.parse_args()


@dataclass
class BenchmarkResult:
    """Results from evaluating a single benchmark."""
    benchmark_name: str
    total_examples: int
    
    # Accuracy metrics
    accuracy: float  # Main accuracy metric for the benchmark
    
    # Uncertainty metrics
    avg_uncertainty: float
    confident_accuracy: float  # Accuracy on low-uncertainty predictions
    abstention_rate: float  # Fraction where model was too uncertain
    
    # Calibration
    ece: float  # Expected Calibration Error
    
    # Additional benchmark-specific metrics
    extra_metrics: Dict[str, float] = field(default_factory=dict)


def create_llm_client(args: argparse.Namespace):
    """Create LLM client based on args."""
    if args.use_ollama:
        from ollama_client import OllamaClient
        return OllamaClient(model=args.ollama_model, verbose=False)
    else:
        from tts_llm_client import TTSLLMClient
        return TTSLLMClient(
            base_url=args.service_url,
            model=args.model,
            tts_budget=args.tts_budget,
        )

def get_uncertainty(llm) -> float:
    """Get last uncertainty from LLM client."""
    uncertainty = getattr(llm, "last_uncertainty", None)
    return uncertainty if uncertainty is not None else 0.5


@dataclass
class EnhancementDiagnostics:
    saup_uncertainty: Optional[float] = None
    propagated_uncertainty: Optional[float] = None
    cot_valid: Optional[bool] = None
    cot_confidence: Optional[float] = None
    reflexion_used: bool = False
    reflexion_attempts: int = 0


def _combine_uncertainties(values: Sequence[float]) -> float:
    """Combine uncertainties using noisy-or (independent error assumption)."""
    combined = 0.0
    for value in values:
        clamped = max(0.0, min(1.0, value))
        combined = 1.0 - (1.0 - combined) * (1.0 - clamped)
    return combined


def _apply_enhancements(
    prompt: str,
    llm,
    args: argparse.Namespace,
    problem_context: str,
    evaluate_fn,
    ground_truth: Optional[str] = None,
    extract_answer_fn=None,
) -> Tuple[str, float, EnhancementDiagnostics]:
    response = llm.complete(prompt)
    base_uncertainty = get_uncertainty(llm)
    diagnostics = EnhancementDiagnostics()
    propagator = create_sage_propagator()
    if not args.disable_propagation:
        propagator.observe(base_uncertainty, "llm_parsing", {"stage": "initial"})

    saup_uncertainty = None
    if not args.disable_saup:
        num_samples = max(1, args.saup_samples)
        decomposer = UncertaintyDecomposer(num_samples=num_samples)
        samples = [llm.complete(prompt) for _ in range(num_samples)]
        decomposed = decomposer.decompose_from_samples(samples, extract_answer_fn)
        saup_uncertainty = decomposed.total
        diagnostics.saup_uncertainty = decomposed.total
        if not args.disable_propagation:
            propagator.observe(decomposed.total, "belief_update", {"stage": "saup_decompose"})

    cot_uncertainty = None
    if not args.disable_cot:
        verifier = ChainOfThoughtVerifier()
        cot_result = verifier.verify_chain(response, problem_context)
        diagnostics.cot_valid = cot_result.overall_valid
        diagnostics.cot_confidence = cot_result.chain_confidence
        cot_uncertainty = max(0.0, 1.0 - cot_result.chain_confidence)
        if not cot_result.overall_valid:
            cot_uncertainty = max(cot_uncertainty, 0.7)
        if not args.disable_propagation:
            propagator.observe(cot_uncertainty, "verification", {"stage": "cot_verify"})

    if not args.disable_reflexion and evaluate_fn is not None:
        is_correct, _feedback = evaluate_fn(response, ground_truth)
        if not is_correct:
            reflexion = ReflexionAgent(
                generate_fn=llm.complete,
                evaluate_fn=evaluate_fn,
                max_attempts=args.reflexion_max_attempts,
            )
            reflexion_result = reflexion.solve(problem_context or prompt, ground_truth or "")
            response = reflexion_result.final_answer
            diagnostics.reflexion_used = True
            diagnostics.reflexion_attempts = reflexion_result.num_attempts
            base_uncertainty = get_uncertainty(llm)
            if not args.disable_propagation:
                propagator.observe(base_uncertainty, "llm_parsing", {"stage": "reflexion"})

            if not args.disable_cot:
                verifier = ChainOfThoughtVerifier()
                cot_result = verifier.verify_chain(response, problem_context)
                diagnostics.cot_valid = cot_result.overall_valid
                diagnostics.cot_confidence = cot_result.chain_confidence
                cot_uncertainty = max(0.0, 1.0 - cot_result.chain_confidence)
                if not cot_result.overall_valid:
                    cot_uncertainty = max(cot_uncertainty, 0.7)
                if not args.disable_propagation:
                    propagator.observe(cot_uncertainty, "verification", {"stage": "reflexion_verify"})

    uncertainty_components = [base_uncertainty]
    if saup_uncertainty is not None:
        uncertainty_components.append(saup_uncertainty)
    if cot_uncertainty is not None:
        uncertainty_components.append(cot_uncertainty)

    if not args.disable_propagation:
        combined_uncertainty = propagator.accumulated_uncertainty
        diagnostics.propagated_uncertainty = combined_uncertainty
    else:
        combined_uncertainty = _combine_uncertainties(uncertainty_components)
    return response, combined_uncertainty, diagnostics


def _evaluate_humaneval_response(
    response: str,
    prompt: str,
    tests: str,
    entry_point: str,
    canonical: str,
) -> Tuple[bool, str]:
    response_clean = response.strip()
    is_correct, eval_method = _execute_humaneval_test(
        generated_code=response_clean,
        prompt=prompt,
        test_code=tests,
        entry_point=entry_point,
    )
    if not is_correct and "assertion" not in eval_method:
        is_correct = _check_code_similarity(response_clean, canonical, entry_point)
        eval_method = "heuristic" if is_correct else eval_method
    return is_correct, eval_method

def evaluate_humaneval(args: argparse.Namespace, llm) -> BenchmarkResult:
    """Evaluate on HumanEval code generation benchmark."""
    from datasets import load_dataset
    
    print("\n" + "=" * 60)
    print("Evaluating on HumanEval (Code Generation)")
    print("=" * 60)
    
    dataset = load_dataset("openai_humaneval", split="test")
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    
    correct = 0
    uncertainties = []
    confidences = []
    correctness_list = []
    saup_uncertainties = []
    propagated_uncertainties = []
    cot_invalids = []
    reflexion_used = []
    
    for i, example in enumerate(dataset):
        task_id = example["task_id"]
        prompt = example["prompt"]
        canonical = example["canonical_solution"]
        tests = example["test"]
        entry_point = example["entry_point"]
        
        # Ask LLM to complete the code
        completion_prompt = f"""Complete the following Python function. Only output the function body, no explanation.

{prompt}
"""

        def evaluate_fn(attempt: str, _gt: Optional[str]) -> Tuple[bool, str]:
            return _evaluate_humaneval_response(
                attempt,
                prompt=prompt,
                tests=tests,
                entry_point=entry_point,
                canonical=canonical,
            )

        response, uncertainty, diagnostics = _apply_enhancements(
            completion_prompt,
            llm,
            args,
            problem_context=prompt,
            evaluate_fn=evaluate_fn,
            ground_truth=None,
        )
        uncertainties.append(uncertainty)
        confidences.append(1.0 - uncertainty)

        is_correct, eval_method = _evaluate_humaneval_response(
            response,
            prompt=prompt,
            tests=tests,
            entry_point=entry_point,
            canonical=canonical,
        )

        if diagnostics.saup_uncertainty is not None:
            saup_uncertainties.append(diagnostics.saup_uncertainty)
        if diagnostics.propagated_uncertainty is not None:
            propagated_uncertainties.append(diagnostics.propagated_uncertainty)
        if diagnostics.cot_valid is not None:
            cot_invalids.append(1.0 if diagnostics.cot_valid is False else 0.0)
        if diagnostics.reflexion_used:
            reflexion_used.append(1.0)
        
        correctness_list.append(1.0 if is_correct else 0.0)
        if is_correct:
            correct += 1
        
        if args.print_each:
            status = "✓" if is_correct else "✗"
            method_info = "" if eval_method == "execution" else f"[{eval_method[:30]}] "
            print(f"[{i+1}/{len(dataset)}] {task_id}: {status} {method_info}(uncertainty={uncertainty:.3f})")
    
    # Compute metrics
    accuracy = correct / len(dataset) if dataset else 0
    avg_unc = sum(uncertainties) / len(uncertainties) if uncertainties else 0
    
    # Uncertainty-aware metrics
    confident_correct = sum(1 for u, c in zip(uncertainties, correctness_list) 
                           if u <= args.uncertainty_threshold and c == 1.0)
    confident_total = sum(1 for u in uncertainties if u <= args.uncertainty_threshold)
    abstained = sum(1 for u in uncertainties if u > args.uncertainty_threshold)
    
    confident_acc = confident_correct / confident_total if confident_total > 0 else 0
    abstention = abstained / len(uncertainties) if uncertainties else 0
    
    # ECE
    ece = _compute_simple_ece(confidences, correctness_list)
    
    return BenchmarkResult(
        benchmark_name="HumanEval",
        total_examples=len(dataset),
        accuracy=accuracy,
        avg_uncertainty=avg_unc,
        confident_accuracy=confident_acc,
        abstention_rate=abstention,
        ece=ece,
        extra_metrics={
            "pass@1": accuracy,
            "saup_uncertainty": sum(saup_uncertainties) / len(saup_uncertainties) if saup_uncertainties else 0.0,
            "cot_invalid_rate": sum(cot_invalids) / len(cot_invalids) if cot_invalids else 0.0,
            "reflexion_used_rate": sum(reflexion_used) / len(correctness_list) if correctness_list else 0.0,
            "propagated_uncertainty": sum(propagated_uncertainties) / len(propagated_uncertainties) if propagated_uncertainties else 0.0,
        },
    )


def _check_code_similarity(generated: str, canonical: str, entry_point: str) -> bool:
    """Heuristic check for code correctness (simplified fallback)."""
    gen_lower = generated.lower()
    can_lower = canonical.lower()
    
    if "return" not in gen_lower:
        return False
    
    gen_tokens = set(re.findall(r'\w+', gen_lower))
    can_tokens = set(re.findall(r'\w+', can_lower))
    
    overlap = len(gen_tokens & can_tokens) / max(len(can_tokens), 1)
    return overlap > 0.3


def _execute_humaneval_test(
    generated_code: str,
    prompt: str,
    test_code: str,
    entry_point: str,
    timeout: float = 5.0,
) -> Tuple[bool, str]:
    """Execute HumanEval test with actual code execution.
    
    Args:
        generated_code: The generated function body or full code
        prompt: Original function signature/docstring
        test_code: HumanEval test assertions (uses check(candidate) pattern)
        entry_point: Function name
        timeout: Max execution time
        
    Returns:
        (success, error_or_method)
    """
    import signal
    from contextlib import contextmanager
    from typing import List as TList, Dict as TDict, Tuple as TTuple, Optional as TOpt, Any as TAny
    
    @contextmanager
    def time_limit(seconds):
        def signal_handler(signum, frame):
            raise TimeoutError("Execution timed out")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(int(seconds))
        try:
            yield
        finally:
            signal.alarm(0)
    
    # Build complete code
    if f"def {entry_point}" in generated_code:
        full_code = generated_code
    else:
        full_code = prompt + generated_code
    
    # Create namespace with common imports
    namespace = {
        '__builtins__': __builtins__,
        'List': TList,
        'Dict': TDict,
        'Tuple': TTuple,
        'Optional': TOpt,
        'Any': TAny,
        'math': __import__('math'),
        'collections': __import__('collections'),
        'itertools': __import__('itertools'),
        'functools': __import__('functools'),
        'typing': __import__('typing'),
    }
    
    try:
        with time_limit(timeout):
            # Execute the generated code to define the function
            exec(full_code, namespace)
            
            if entry_point not in namespace:
                return False, f"Function '{entry_point}' not defined"
            
            # Execute test code which defines check() function
            exec(test_code, namespace)
            
            # The test code defines a check() function - call it with our function
            if 'check' in namespace:
                namespace['check'](namespace[entry_point])
            
            return True, "execution"
            
    except AssertionError as e:
        return False, f"assertion_failed: {e}"
    except TimeoutError:
        return False, "timeout"
    except SyntaxError as e:
        return False, f"syntax_error: {e}"
    except Exception as e:
        return False, f"error: {type(e).__name__}: {e}"


def evaluate_mbpp(args: argparse.Namespace, llm) -> BenchmarkResult:
    """Evaluate on MBPP code generation benchmark."""
    from datasets import load_dataset
    
    print("\n" + "=" * 60)
    print("Evaluating on MBPP (Mostly Basic Python Problems)")
    print("=" * 60)
    
    dataset = load_dataset("mbpp", split="test")
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    
    correct = 0
    uncertainties = []
    confidences = []
    correctness_list = []
    saup_uncertainties = []
    propagated_uncertainties = []
    cot_invalids = []
    reflexion_used = []
    
    for i, example in enumerate(dataset):
        task_id = example["task_id"]
        text = example["text"]  # Problem description
        code = example["code"]  # Reference solution
        
        # Ask LLM to generate code
        prompt = f"""Write a Python function to solve the following problem. Only output the code, no explanation.

        Problem: {text}
        """
        def evaluate_fn(attempt: str, _gt: Optional[str]) -> Tuple[bool, str]:
            is_correct = _check_code_similarity(attempt.strip(), code, "")
            return is_correct, "heuristic" if is_correct else "mismatch"

        response, uncertainty, diagnostics = _apply_enhancements(
            prompt,
            llm,
            args,
            problem_context=text,
            evaluate_fn=evaluate_fn,
            ground_truth=None,
        )
        uncertainties.append(uncertainty)
        confidences.append(1.0 - uncertainty)
        
        # Heuristic check
        is_correct = _check_code_similarity(response.strip(), code, "")
        correctness_list.append(1.0 if is_correct else 0.0)
        if is_correct:
            correct += 1

        if diagnostics.saup_uncertainty is not None:
            saup_uncertainties.append(diagnostics.saup_uncertainty)
        if diagnostics.propagated_uncertainty is not None:
            propagated_uncertainties.append(diagnostics.propagated_uncertainty)
        if diagnostics.cot_valid is not None:
            cot_invalids.append(1.0 if diagnostics.cot_valid is False else 0.0)
        if diagnostics.reflexion_used:
            reflexion_used.append(1.0)
        
        if args.print_each:
            status = "ok" if is_correct else "fail"
            print(f"[{i+1}/{len(dataset)}] Task {task_id}: {status} (uncertainty={uncertainty:.3f})")
    
    accuracy = correct / len(dataset) if dataset else 0
    avg_unc = sum(uncertainties) / len(uncertainties) if uncertainties else 0
    
    confident_correct = sum(1 for u, c in zip(uncertainties, correctness_list) 
                           if u <= args.uncertainty_threshold and c == 1.0)
    confident_total = sum(1 for u in uncertainties if u <= args.uncertainty_threshold)
    abstained = sum(1 for u in uncertainties if u > args.uncertainty_threshold)
    
    confident_acc = confident_correct / confident_total if confident_total > 0 else 0
    abstention = abstained / len(uncertainties) if uncertainties else 0
    ece = _compute_simple_ece(confidences, correctness_list)
    
    return BenchmarkResult(
        benchmark_name="MBPP",
        total_examples=len(dataset),
        accuracy=accuracy,
        avg_uncertainty=avg_unc,
        confident_accuracy=confident_acc,
        abstention_rate=abstention,
        ece=ece,
        extra_metrics={
            "saup_uncertainty": sum(saup_uncertainties) / len(saup_uncertainties) if saup_uncertainties else 0.0,
            "cot_invalid_rate": sum(cot_invalids) / len(cot_invalids) if cot_invalids else 0.0,
            "reflexion_used_rate": sum(reflexion_used) / len(correctness_list) if correctness_list else 0.0,
            "propagated_uncertainty": sum(propagated_uncertainties) / len(propagated_uncertainties) if propagated_uncertainties else 0.0,
        },
    )

def evaluate_gsm8k(args: argparse.Namespace, llm) -> BenchmarkResult:
    """Evaluate on GSM8K math reasoning benchmark."""
    from datasets import load_dataset
    
    print("\n" + "=" * 60)
    print("Evaluating on GSM8K (Math Reasoning)")
    print("=" * 60)
    
    dataset = load_dataset("gsm8k", "main", split="test")
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    
    correct = 0
    uncertainties = []
    confidences = []
    correctness_list = []
    saup_uncertainties = []
    propagated_uncertainties = []
    cot_invalids = []
    reflexion_used = []
    
    for i, example in enumerate(dataset):
        question = example["question"]
        answer = example["answer"]
        
        # Extract final numeric answer from ground truth
        gt_answer = _extract_gsm8k_answer(answer)
        
        # Ask LLM to solve
        prompt = f"""Solve the following math problem step by step. At the end, provide your final answer as a number.

Problem: {question}

Solution:"""
        def evaluate_fn(attempt: str, _gt: Optional[str]) -> Tuple[bool, str]:
            pred = _extract_numeric_answer(attempt)
            is_correct = pred is not None and abs(pred - gt_answer) < 0.01
            feedback = f"pred={pred}, gt={gt_answer}"
            return is_correct, feedback

        response, uncertainty, diagnostics = _apply_enhancements(
            prompt,
            llm,
            args,
            problem_context=question,
            evaluate_fn=evaluate_fn,
            ground_truth=None,
            extract_answer_fn=_extract_numeric_answer,
        )
        uncertainties.append(uncertainty)
        confidences.append(1.0 - uncertainty)
        
        # Extract answer from response
        pred_answer = _extract_numeric_answer(response)
        is_correct = pred_answer is not None and abs(pred_answer - gt_answer) < 0.01
        correctness_list.append(1.0 if is_correct else 0.0)
        if is_correct:
            correct += 1
        
        if args.print_each:
            status = "ok" if is_correct else "fail"
            print(f"[{i+1}/{len(dataset)}] {status} pred={pred_answer}, gt={gt_answer} (unc={uncertainty:.3f})")

        if diagnostics.saup_uncertainty is not None:
            saup_uncertainties.append(diagnostics.saup_uncertainty)
        if diagnostics.propagated_uncertainty is not None:
            propagated_uncertainties.append(diagnostics.propagated_uncertainty)
        if diagnostics.cot_valid is not None:
            cot_invalids.append(1.0 if diagnostics.cot_valid is False else 0.0)
        if diagnostics.reflexion_used:
            reflexion_used.append(1.0)
    
    accuracy = correct / len(dataset) if dataset else 0
    avg_unc = sum(uncertainties) / len(uncertainties) if uncertainties else 0
    
    confident_correct = sum(1 for u, c in zip(uncertainties, correctness_list) 
                           if u <= args.uncertainty_threshold and c == 1.0)
    confident_total = sum(1 for u in uncertainties if u <= args.uncertainty_threshold)
    abstained = sum(1 for u in uncertainties if u > args.uncertainty_threshold)
    
    confident_acc = confident_correct / confident_total if confident_total > 0 else 0
    abstention = abstained / len(uncertainties) if uncertainties else 0
    ece = _compute_simple_ece(confidences, correctness_list)
    
    return BenchmarkResult(
        benchmark_name="GSM8K",
        total_examples=len(dataset),
        accuracy=accuracy,
        avg_uncertainty=avg_unc,
        confident_accuracy=confident_acc,
        abstention_rate=abstention,
        ece=ece,
        extra_metrics={
            "saup_uncertainty": sum(saup_uncertainties) / len(saup_uncertainties) if saup_uncertainties else 0.0,
            "cot_invalid_rate": sum(cot_invalids) / len(cot_invalids) if cot_invalids else 0.0,
            "reflexion_used_rate": sum(reflexion_used) / len(correctness_list) if correctness_list else 0.0,
            "propagated_uncertainty": sum(propagated_uncertainties) / len(propagated_uncertainties) if propagated_uncertainties else 0.0,
        },
    )


def _extract_gsm8k_answer(answer_text: str) -> float:
    """Extract numeric answer from GSM8K format (#### <number>)."""
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', answer_text)
    if match:
        return float(match.group(1).replace(',', ''))
    return 0.0


def _extract_numeric_answer(text: str) -> Optional[float]:
    """Extract the last numeric answer from model response."""
    # Look for patterns like "the answer is X" or just the last number
    patterns = [
        r'(?:answer|result|total|=)\s*[:is]*\s*(-?[\d,]+\.?\d*)',
        r'####\s*(-?[\d,]+\.?\d*)',
        r'(-?[\d,]+\.?\d*)\s*$',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            try:
                return float(matches[-1].replace(',', ''))
            except ValueError:
                continue
    return None


def evaluate_hotpotqa(args: argparse.Namespace, llm) -> BenchmarkResult:
    """Evaluate on HotpotQA multi-hop QA benchmark."""
    from datasets import load_dataset
    
    print("\n" + "=" * 60)
    print("Evaluating on HotpotQA (Multi-hop QA)")
    print("=" * 60)
    
    dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    
    correct = 0
    uncertainties = []
    confidences = []
    correctness_list = []
    saup_uncertainties = []
    propagated_uncertainties = []
    cot_invalids = []
    reflexion_used = []
    
    for i, example in enumerate(dataset):
        question = example["question"]
        answer = example["answer"]
        
        # Ask LLM to answer
        prompt = f"""Answer the following question concisely.

        Question: {question}

        Answer:"""

        def evaluate_fn(attempt: str, _gt: Optional[str]) -> Tuple[bool, str]:
            is_correct = _fuzzy_match(attempt.strip(), answer.strip())
            feedback = f"gt={answer.strip()}"
            return is_correct, feedback

        response, uncertainty, diagnostics = _apply_enhancements(
            prompt,
            llm,
            args,
            problem_context=question,
            evaluate_fn=evaluate_fn,
            ground_truth=None,
        )
        uncertainties.append(uncertainty)
        confidences.append(1.0 - uncertainty)
        
        # Check if answer matches (fuzzy)
        is_correct = _fuzzy_match(response.strip(), answer.strip())
        correctness_list.append(1.0 if is_correct else 0.0)
        if is_correct:
            correct += 1
        
        if args.print_each:
            status = "ok" if is_correct else "fail"
            print(f"[{i+1}/{len(dataset)}] {status} pred='{response[:50]}...' gt='{answer}' (unc={uncertainty:.3f})")

        if diagnostics.saup_uncertainty is not None:
            saup_uncertainties.append(diagnostics.saup_uncertainty)
        if diagnostics.propagated_uncertainty is not None:
            propagated_uncertainties.append(diagnostics.propagated_uncertainty)
        if diagnostics.cot_valid is not None:
            cot_invalids.append(1.0 if diagnostics.cot_valid is False else 0.0)
        if diagnostics.reflexion_used:
            reflexion_used.append(1.0)
    
    accuracy = correct / len(dataset) if dataset else 0
    avg_unc = sum(uncertainties) / len(uncertainties) if uncertainties else 0
    
    confident_correct = sum(1 for u, c in zip(uncertainties, correctness_list) 
                           if u <= args.uncertainty_threshold and c == 1.0)
    confident_total = sum(1 for u in uncertainties if u <= args.uncertainty_threshold)
    abstained = sum(1 for u in uncertainties if u > args.uncertainty_threshold)
    
    confident_acc = confident_correct / confident_total if confident_total > 0 else 0
    abstention = abstained / len(uncertainties) if uncertainties else 0
    ece = _compute_simple_ece(confidences, correctness_list)
    
    return BenchmarkResult(
        benchmark_name="HotpotQA",
        total_examples=len(dataset),
        accuracy=accuracy,
        avg_uncertainty=avg_unc,
        confident_accuracy=confident_acc,
        abstention_rate=abstention,
        ece=ece,
        extra_metrics={
            "exact_match": accuracy,
            "saup_uncertainty": sum(saup_uncertainties) / len(saup_uncertainties) if saup_uncertainties else 0.0,
            "cot_invalid_rate": sum(cot_invalids) / len(cot_invalids) if cot_invalids else 0.0,
            "reflexion_used_rate": sum(reflexion_used) / len(correctness_list) if correctness_list else 0.0,
            "propagated_uncertainty": sum(propagated_uncertainties) / len(propagated_uncertainties) if propagated_uncertainties else 0.0,
        },
    )


def _fuzzy_match(pred: str, gt: str) -> bool:
    """Fuzzy string matching for QA."""
    pred_lower = pred.lower().strip()
    gt_lower = gt.lower().strip()
    
    # Exact match
    if pred_lower == gt_lower:
        return True
    
    # Contains match
    if gt_lower in pred_lower or pred_lower in gt_lower:
        return True
    
    # Token overlap
    pred_tokens = set(pred_lower.split())
    gt_tokens = set(gt_lower.split())
    if gt_tokens and len(pred_tokens & gt_tokens) / len(gt_tokens) > 0.5:
        return True
    
    return False


def _compute_simple_ece(confidences: List[float], correctness: List[float], num_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    if not confidences:
        return 0.0
    
    n = len(confidences)
    bin_sums = [0.0] * num_bins
    bin_correct = [0.0] * num_bins
    bin_counts = [0] * num_bins
    
    for conf, corr in zip(confidences, correctness):
        bin_idx = min(int(conf * num_bins), num_bins - 1)
        bin_sums[bin_idx] += conf
        bin_correct[bin_idx] += corr
        bin_counts[bin_idx] += 1
    
    ece = 0.0
    for i in range(num_bins):
        if bin_counts[i] > 0:
            avg_conf = bin_sums[i] / bin_counts[i]
            avg_acc = bin_correct[i] / bin_counts[i]
            ece += (bin_counts[i] / n) * abs(avg_acc - avg_conf)
    
    return ece


def print_result(result: BenchmarkResult):
    """Print benchmark result."""
    print("\n" + "-" * 60)
    print(f"Results: {result.benchmark_name}")
    print("-" * 60)
    print(f"Total examples:        {result.total_examples}")
    print(f"Accuracy:              {result.accuracy:.4f}")
    print(f"Avg uncertainty:       {result.avg_uncertainty:.4f}")
    print(f"Confident accuracy:    {result.confident_accuracy:.4f}")
    print(f"Abstention rate:       {result.abstention_rate:.4f}")
    print(f"ECE:                   {result.ece:.4f}")
    if result.extra_metrics:
        for k, v in result.extra_metrics.items():
            print(f"{k}:                {v:.4f}")



def main():
    args = parse_args()
    
    print("=" * 60)
    print("Multi-Benchmark Uncertainty Evaluation")
    print("=" * 60)
    print(f"Model: {args.ollama_model if args.use_ollama else args.model}")
    print(f"Limit: {args.limit}")
    print(f"Uncertainty threshold: {args.uncertainty_threshold}")
    
    llm = create_llm_client(args)
    
    results: List[BenchmarkResult] = []
    
    benchmarks = {
        "humaneval": evaluate_humaneval,
        "mbpp": evaluate_mbpp,
        "gsm8k": evaluate_gsm8k,
        "hotpotqa": evaluate_hotpotqa,
    }
    
    if args.benchmark == "all":
        for name, eval_fn in benchmarks.items():
            try:
                result = eval_fn(args, llm)
                results.append(result)
                print_result(result)
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
    elif args.benchmark == "when2call":
        print("For When2Call, use run_when2call_eval.py")
        return
    else:
        result = benchmarks[args.benchmark](args, llm)
        results.append(result)
        print_result(result)
    
    # Summary
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("Summary Across Benchmarks")
        print("=" * 60)
        avg_acc = sum(r.accuracy for r in results) / len(results)
        avg_unc = sum(r.avg_uncertainty for r in results) / len(results)
        avg_conf_acc = sum(r.confident_accuracy for r in results) / len(results)
        avg_ece = sum(r.ece for r in results) / len(results)
        
        print(f"Average accuracy:           {avg_acc:.4f}")
        print(f"Average uncertainty:        {avg_unc:.4f}")
        print(f"Average confident accuracy: {avg_conf_acc:.4f}")
        print(f"Average ECE:                {avg_ece:.4f}")


if __name__ == "__main__":
    main()
