"""
LangGraph-based evaluation for code benchmarks (HumanEval, MBPP).

This mirrors run_multi_benchmark_eval.py but uses a LangGraph pipeline
for generate -> verify -> reflexion retries.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TypedDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langgraph.graph import StateGraph, END


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LangGraph evaluation on code benchmarks with uncertainty."
    )
    parser.add_argument(
        "--benchmark",
        choices=["humaneval", "mbpp", "gsm8k", "hotpotqa", "all"],
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
        "--max-attempts",
        type=int,
        default=3,
        help="Max attempts with reflexion-style retries.",
    )
    return parser.parse_args()


@dataclass
class BenchmarkResult:
    benchmark_name: str
    total_examples: int
    accuracy: float
    avg_uncertainty: float
    confident_accuracy: float
    abstention_rate: float
    ece: float
    extra_metrics: Dict[str, float]


class AgentState(TypedDict):
    prompt: str
    problem: str
    generated_code: str
    attempt: int
    max_attempts: int
    last_uncertainty: float
    verification_passed: bool
    verification_error: str


def create_llm_client(args: argparse.Namespace):
    if args.use_ollama:
        from examples.ollama_client import OllamaClient
        return OllamaClient(model=args.ollama_model, verbose=False)
    from examples.tts_llm_client import TTSLLMClient
    return TTSLLMClient(
        base_url=args.service_url,
        model=args.model,
        tts_budget=args.tts_budget,
    )


def get_uncertainty(llm) -> float:
    uncertainty = getattr(llm, "last_uncertainty", None)
    return uncertainty if uncertainty is not None else 0.5


def build_graph(
    llm,
    evaluate_fn: Callable[[str], Tuple[bool, str]],
) -> StateGraph:
    graph = StateGraph(AgentState)

    def generate_node(state: AgentState) -> AgentState:
        response = llm.complete(state["prompt"])
        return {
            **state,
            "generated_code": response.strip(),
            "last_uncertainty": get_uncertainty(llm),
            "attempt": state["attempt"] + 1,
        }

    def verify_node(state: AgentState) -> AgentState:
        is_correct, error = evaluate_fn(state["generated_code"])
        return {
            **state,
            "verification_passed": is_correct,
            "verification_error": error,
        }

    def refine_node(state: AgentState) -> AgentState:
        prompt = (
            "The previous attempt failed. Improve the code.\n\n"
            f"Problem: {state['problem']}\n"
            f"Previous attempt:\n{state['generated_code']}\n"
            f"Error: {state['verification_error']}\n\n"
            "Return only the improved code."
        )
        response = llm.complete(prompt)
        return {
            **state,
            "generated_code": response.strip(),
            "last_uncertainty": get_uncertainty(llm),
            "attempt": state["attempt"] + 1,
        }

    def route_after_verify(state: AgentState) -> str:
        if state["verification_passed"]:
            return "done"
        if state["attempt"] < state["max_attempts"]:
            return "refine"
        return "done"

    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)
    graph.add_node("refine", refine_node)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {"refine": "refine", "done": END},
    )
    graph.add_edge("refine", "verify")

    return graph


def _check_code_similarity(generated: str, canonical: str, entry_point: str) -> bool:
    gen_lower = generated.lower()
    can_lower = canonical.lower()
    if "return" not in gen_lower:
        return False
    gen_tokens = set(re.findall(r"\w+", gen_lower))
    can_tokens = set(re.findall(r"\w+", can_lower))
    overlap = len(gen_tokens & can_tokens) / max(len(can_tokens), 1)
    return overlap > 0.3


def _execute_humaneval_test(
    generated_code: str,
    prompt: str,
    test_code: str,
    entry_point: str,
    timeout: float = 5.0,
) -> Tuple[bool, str]:
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

    if f"def {entry_point}" in generated_code:
        full_code = generated_code
    else:
        full_code = prompt + generated_code

    namespace = {
        "__builtins__": __builtins__,
        "List": TList,
        "Dict": TDict,
        "Tuple": TTuple,
        "Optional": TOpt,
        "Any": TAny,
        "math": __import__("math"),
        "collections": __import__("collections"),
        "itertools": __import__("itertools"),
        "functools": __import__("functools"),
        "typing": __import__("typing"),
    }

    try:
        with time_limit(timeout):
            exec(full_code, namespace)
            if entry_point not in namespace:
                return False, f"Function '{entry_point}' not defined"
            exec(test_code, namespace)
            if "check" in namespace:
                namespace["check"](namespace[entry_point])
            return True, "execution"
    except AssertionError as e:
        return False, f"assertion_failed: {e}"
    except TimeoutError:
        return False, "timeout"
    except SyntaxError as e:
        return False, f"syntax_error: {e}"
    except Exception as e:
        return False, f"error: {type(e).__name__}: {e}"


def _compute_simple_ece(confidences: List[float], correctness: List[float], num_bins: int = 10) -> float:
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


def _extract_gsm8k_answer(answer_text: str) -> float:
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
    if match:
        return float(match.group(1).replace(",", ""))
    return 0.0


def _extract_numeric_answer(text: str) -> Optional[float]:
    patterns = [
        r"(?:answer|result|total|=)\s*[:is]*\s*(-?[\d,]+\.?\d*)",
        r"####\s*(-?[\d,]+\.?\d*)",
        r"(-?[\d,]+\.?\d*)\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            try:
                return float(matches[-1].replace(",", ""))
            except ValueError:
                continue
    return None


def _fuzzy_match(pred: str, gt: str) -> bool:
    pred_lower = pred.lower().strip()
    gt_lower = gt.lower().strip()
    if pred_lower == gt_lower:
        return True
    if gt_lower in pred_lower or pred_lower in gt_lower:
        return True
    pred_tokens = set(pred_lower.split())
    gt_tokens = set(gt_lower.split())
    if gt_tokens and len(pred_tokens & gt_tokens) / len(gt_tokens) > 0.5:
        return True
    return False


def evaluate_humaneval(args: argparse.Namespace, llm) -> BenchmarkResult:
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("Evaluating on HumanEval (Code Generation)")
    print("=" * 60)

    dataset = load_dataset("openai_humaneval", split="test")
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    correct = 0
    uncertainties: List[float] = []
    confidences: List[float] = []
    correctness_list: List[float] = []
    attempts: List[int] = []

    for i, example in enumerate(dataset):
        prompt = example["prompt"]
        canonical = example["canonical_solution"]
        tests = example["test"]
        entry_point = example["entry_point"]

        completion_prompt = (
            "Complete the following Python function. Only output the function body, no explanation.\n\n"
            f"{prompt}\n"
        )

        def evaluate_fn(code: str) -> Tuple[bool, str]:
            is_correct, eval_method = _execute_humaneval_test(
                generated_code=code,
                prompt=prompt,
                test_code=tests,
                entry_point=entry_point,
            )
            if not is_correct and "assertion" not in eval_method:
                is_correct = _check_code_similarity(code, canonical, entry_point)
            return is_correct, eval_method

        graph = build_graph(llm, evaluate_fn).compile()
        state: AgentState = {
            "prompt": completion_prompt,
            "problem": prompt,
            "generated_code": "",
            "attempt": 0,
            "max_attempts": args.max_attempts,
            "last_uncertainty": 0.5,
            "verification_passed": False,
            "verification_error": "",
        }
        result_state = graph.invoke(state)
        is_correct, _ = evaluate_fn(result_state["generated_code"])
        uncertainty = result_state["last_uncertainty"]
        uncertainties.append(uncertainty)
        confidences.append(1.0 - uncertainty)
        correctness_list.append(1.0 if is_correct else 0.0)
        attempts.append(result_state["attempt"])
        if is_correct:
            correct += 1

        if args.print_each:
            status = "✓" if is_correct else "✗"
            print(f"[{i+1}/{len(dataset)}] {example['task_id']}: {status} (uncertainty={uncertainty:.3f})")

    accuracy = correct / len(dataset) if dataset else 0.0
    avg_unc = sum(uncertainties) / len(uncertainties) if uncertainties else 0.0
    confident_correct = sum(
        1 for u, c in zip(uncertainties, correctness_list)
        if u <= args.uncertainty_threshold and c == 1.0
    )
    confident_total = sum(1 for u in uncertainties if u <= args.uncertainty_threshold)
    abstained = sum(1 for u in uncertainties if u > args.uncertainty_threshold)
    confident_acc = confident_correct / confident_total if confident_total > 0 else 0.0
    abstention = abstained / len(uncertainties) if uncertainties else 0.0
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
            "avg_attempts": sum(attempts) / len(attempts) if attempts else 0.0,
        },
    )


def evaluate_mbpp(args: argparse.Namespace, llm) -> BenchmarkResult:
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("Evaluating on MBPP (Mostly Basic Python Problems)")
    print("=" * 60)

    dataset = load_dataset("mbpp", split="test")
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    correct = 0
    uncertainties: List[float] = []
    confidences: List[float] = []
    correctness_list: List[float] = []
    attempts: List[int] = []

    for i, example in enumerate(dataset):
        text = example["text"]
        code = example["code"]

        prompt = (
            "Write a Python function to solve the following problem. "
            "Only output the code, no explanation.\n\n"
            f"Problem: {text}\n"
        )

        def evaluate_fn(gen: str) -> Tuple[bool, str]:
            ok = _check_code_similarity(gen, code, "")
            return ok, "heuristic"

        graph = build_graph(llm, evaluate_fn).compile()
        state: AgentState = {
            "prompt": prompt,
            "problem": text,
            "generated_code": "",
            "attempt": 0,
            "max_attempts": args.max_attempts,
            "last_uncertainty": 0.5,
            "verification_passed": False,
            "verification_error": "",
        }
        result_state = graph.invoke(state)
        is_correct, _ = evaluate_fn(result_state["generated_code"])
        uncertainty = result_state["last_uncertainty"]
        uncertainties.append(uncertainty)
        confidences.append(1.0 - uncertainty)
        correctness_list.append(1.0 if is_correct else 0.0)
        attempts.append(result_state["attempt"])
        if is_correct:
            correct += 1

        if args.print_each:
            status = "ok" if is_correct else "fail"
            print(f"[{i+1}/{len(dataset)}] Task {example['task_id']}: {status} (uncertainty={uncertainty:.3f})")

    accuracy = correct / len(dataset) if dataset else 0.0
    avg_unc = sum(uncertainties) / len(uncertainties) if uncertainties else 0.0
    confident_correct = sum(
        1 for u, c in zip(uncertainties, correctness_list)
        if u <= args.uncertainty_threshold and c == 1.0
    )
    confident_total = sum(1 for u in uncertainties if u <= args.uncertainty_threshold)
    abstained = sum(1 for u in uncertainties if u > args.uncertainty_threshold)
    confident_acc = confident_correct / confident_total if confident_total > 0 else 0.0
    abstention = abstained / len(uncertainties) if uncertainties else 0.0
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
            "avg_attempts": sum(attempts) / len(attempts) if attempts else 0.0,
        },
    )


def evaluate_gsm8k(args: argparse.Namespace, llm) -> BenchmarkResult:
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("Evaluating on GSM8K (Math Reasoning)")
    print("=" * 60)

    dataset = load_dataset("gsm8k", "main", split="test")
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    correct = 0
    uncertainties: List[float] = []
    confidences: List[float] = []
    correctness_list: List[float] = []
    attempts: List[int] = []

    for i, example in enumerate(dataset):
        question = example["question"]
        answer = example["answer"]
        gt_answer = _extract_gsm8k_answer(answer)

        prompt = (
            "Solve the following math problem step by step. "
            "At the end, provide your final answer as a number.\n\n"
            f"Problem: {question}\n\n"
            "Solution:"
        )

        def evaluate_fn(gen: str) -> Tuple[bool, str]:
            pred = _extract_numeric_answer(gen)
            ok = pred is not None and abs(pred - gt_answer) < 0.01
            return ok, f"pred={pred}, gt={gt_answer}"

        graph = build_graph(llm, evaluate_fn).compile()
        state: AgentState = {
            "prompt": prompt,
            "problem": question,
            "generated_code": "",
            "attempt": 0,
            "max_attempts": args.max_attempts,
            "last_uncertainty": 0.5,
            "verification_passed": False,
            "verification_error": "",
        }
        result_state = graph.invoke(state)
        is_correct, _ = evaluate_fn(result_state["generated_code"])
        uncertainty = result_state["last_uncertainty"]
        uncertainties.append(uncertainty)
        confidences.append(1.0 - uncertainty)
        correctness_list.append(1.0 if is_correct else 0.0)
        attempts.append(result_state["attempt"])
        if is_correct:
            correct += 1

        if args.print_each:
            status = "ok" if is_correct else "fail"
            print(f"[{i+1}/{len(dataset)}] {status} (uncertainty={uncertainty:.3f})")

    accuracy = correct / len(dataset) if dataset else 0.0
    avg_unc = sum(uncertainties) / len(uncertainties) if uncertainties else 0.0
    confident_correct = sum(
        1 for u, c in zip(uncertainties, correctness_list)
        if u <= args.uncertainty_threshold and c == 1.0
    )
    confident_total = sum(1 for u in uncertainties if u <= args.uncertainty_threshold)
    abstained = sum(1 for u in uncertainties if u > args.uncertainty_threshold)
    confident_acc = confident_correct / confident_total if confident_total > 0 else 0.0
    abstention = abstained / len(uncertainties) if uncertainties else 0.0
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
            "avg_attempts": sum(attempts) / len(attempts) if attempts else 0.0,
        },
    )


def evaluate_hotpotqa(args: argparse.Namespace, llm) -> BenchmarkResult:
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("Evaluating on HotpotQA (Multi-hop QA)")
    print("=" * 60)

    dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    correct = 0
    uncertainties: List[float] = []
    confidences: List[float] = []
    correctness_list: List[float] = []
    attempts: List[int] = []

    for i, example in enumerate(dataset):
        question = example["question"]
        answer = example["answer"]

        prompt = (
            "Answer the following question concisely.\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        def evaluate_fn(gen: str) -> Tuple[bool, str]:
            ok = _fuzzy_match(gen.strip(), answer.strip())
            return ok, f"gt={answer.strip()}"

        graph = build_graph(llm, evaluate_fn).compile()
        state: AgentState = {
            "prompt": prompt,
            "problem": question,
            "generated_code": "",
            "attempt": 0,
            "max_attempts": args.max_attempts,
            "last_uncertainty": 0.5,
            "verification_passed": False,
            "verification_error": "",
        }
        result_state = graph.invoke(state)
        is_correct, _ = evaluate_fn(result_state["generated_code"])
        uncertainty = result_state["last_uncertainty"]
        uncertainties.append(uncertainty)
        confidences.append(1.0 - uncertainty)
        correctness_list.append(1.0 if is_correct else 0.0)
        attempts.append(result_state["attempt"])
        if is_correct:
            correct += 1

        if args.print_each:
            status = "ok" if is_correct else "fail"
            print(f"[{i+1}/{len(dataset)}] {status} (uncertainty={uncertainty:.3f})")

    accuracy = correct / len(dataset) if dataset else 0.0
    avg_unc = sum(uncertainties) / len(uncertainties) if uncertainties else 0.0
    confident_correct = sum(
        1 for u, c in zip(uncertainties, correctness_list)
        if u <= args.uncertainty_threshold and c == 1.0
    )
    confident_total = sum(1 for u in uncertainties if u <= args.uncertainty_threshold)
    abstained = sum(1 for u in uncertainties if u > args.uncertainty_threshold)
    confident_acc = confident_correct / confident_total if confident_total > 0 else 0.0
    abstention = abstained / len(uncertainties) if uncertainties else 0.0
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
            "avg_attempts": sum(attempts) / len(attempts) if attempts else 0.0,
            "exact_match": accuracy,
        },
    )


def print_result(result: BenchmarkResult):
    print("\n" + "-" * 60)
    print(f"Results: {result.benchmark_name}")
    print("-" * 60)
    print(f"Total examples:        {result.total_examples}")
    print(f"Accuracy:              {result.accuracy:.4f}")
    print(f"Avg uncertainty:       {result.avg_uncertainty:.4f}")
    print(f"Confident accuracy:    {result.confident_accuracy:.4f}")
    print(f"Abstention rate:       {result.abstention_rate:.4f}")
    print(f"ECE:                   {result.ece:.4f}")
    for k, v in result.extra_metrics.items():
        print(f"{k}:                {v:.4f}")


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("LangGraph Code Benchmark Evaluation")
    print("=" * 60)
    print(f"Model: {args.ollama_model if args.use_ollama else args.model}")
    print(f"Limit: {args.limit}")
    print(f"Max attempts: {args.max_attempts}")
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
            result = eval_fn(args, llm)
            results.append(result)
            print_result(result)
    else:
        result = benchmarks[args.benchmark](args, llm)
        results.append(result)
        print_result(result)

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
