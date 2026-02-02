"""
SAGE Agent v3 evaluation on code generation benchmarks (HumanEval, MBPP, GSM8K).

Adapts SAGE v3 (designed for tool calling) to code generation by treating
code generation as a structured tool with parameters.

Usage:
    python examples/run_multi_benchmark_sage_v3.py \
      --benchmark all --limit 30 --print-each \
      --model "mistralai/devstral-2512:free" \
      --use-v3 --v3-config balanced
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Project root is 2 levels up from different_agents/v3/
ROOT = Path(__file__).resolve().parents[2]
V3_DIR = Path(__file__).resolve().parent
SHARED_DIR = ROOT / "different_agents" / "shared"
MISC_DIR = ROOT / "different_agents" / "misc"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V3_DIR))
sys.path.insert(0, str(SHARED_DIR))
sys.path.insert(0, str(MISC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAGE v3 evaluation on code benchmarks."
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
        default="qwen/qwen-2.5-7b-instruct",
        help="Model name for llm-tts service.",
    )
    parser.add_argument(
        "--service-url",
        default="http://localhost:8001/v1",
        help="llm-tts service URL.",
    )
    parser.add_argument(
        "--use-ollama",
        action="store_true",
        help="Use Ollama instead of TTS service.",
    )
    parser.add_argument(
        "--ollama-model",
        default="qwen2.5:7b-instruct",
        help="Ollama model name.",
    )
    parser.add_argument(
        "--print-each",
        action="store_true",
        help="Print per-example results.",
    )
    parser.add_argument(
        "--use-v3",
        action="store_true",
        help="Use SAGE v3 agent (default: v2).",
    )
    parser.add_argument(
        "--v3-config",
        choices=["conservative", "balanced", "aggressive", "lite"],
        default="balanced",
        help="v3 configuration profile.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=2,
        help="Max clarification questions for SAGE.",
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


def create_code_generation_tools():
    """Create tool schemas for code generation.

    Treats code generation as structured tool calling:
    - generate_python_code: For Python code generation
    - generate_solution: For math/reasoning problems
    """
    from sage_agent import ParameterDomain, ToolSchema

    python_code_tool = ToolSchema(
        name="generate_python_code",
        parameters={
            "function_name": ParameterDomain.continuous(),
            "approach": ParameterDomain.from_values([
                "iterative",
                "recursive",
                "dynamic_programming",
                "greedy",
                "divide_and_conquer",
                "brute_force",
            ]),
            "code": ParameterDomain.continuous(),
            "explanation": ParameterDomain.continuous(),
        },
        required=frozenset({"code"}),
    )

    solution_tool = ToolSchema(
        name="generate_solution",
        parameters={
            "reasoning": ParameterDomain.continuous(),
            "solution": ParameterDomain.continuous(),
            "answer": ParameterDomain.continuous(),
        },
        required=frozenset({"answer"}),
    )

    return {
        python_code_tool.name: python_code_tool,
        solution_tool.name: solution_tool,
    }


def create_sage_agent(args, tool_schemas):
    """Create SAGE agent (v2 or v3) for code generation."""
    from sage_agent import (
        LLMBackedCandidateGenerator,
        LLMBackedQuestionGenerator,
        SageAgentConfig,
        SimpleConstraintExtractor,
        ToolRegistryExecutor,
    )
    from sage_agent.core.types import ExecutionResult

    # Create LLM client
    if args.use_ollama:
        from ollama_client import OllamaClient
        llm = OllamaClient(model=args.ollama_model, verbose=False)
    else:
        from tts_llm_client import TTSLLMClient
        llm = TTSLLMClient(
            base_url=args.service_url,
            model=args.model,
            tts_strategy="self_consistency",
            tts_budget=8,
        )

    # Dummy question asker (code gen doesn't ask questions usually)
    class NoOpQuestionAsker:
        def ask(self, question):
            return "Please proceed with your best guess."

    # Dummy tool executor (we just collect the code, don't execute via tools)
    def dummy_executor(tool_call):
        return ExecutionResult(success=True, output={"result": tool_call})

    tool_registry = {name: dummy_executor for name in tool_schemas.keys()}

    # Import appropriate version
    if args.use_v3:
        from langgraph_sage_agent_v3 import (
            GraphDeps,
            build_graph,
            create_initial_state,
        )
        from v3_configs import get_config
        from sage_agent import create_sage_propagator
        from sage_agent.core.advanced_reasoning import UncertaintyDecomposer

        config_dict = get_config(args.v3_config)

        deps = GraphDeps(
            tool_schemas=tool_schemas,
            candidate_generator=LLMBackedCandidateGenerator(llm),
            question_generator=LLMBackedQuestionGenerator(llm),
            question_asker=NoOpQuestionAsker(),
            tool_executor=ToolRegistryExecutor(tool_registry),
            constraint_extractor=SimpleConstraintExtractor(),
            config=SageAgentConfig(
                max_questions=args.max_questions,
                redundancy_weight=0.02,
                tau_execute=0.8,
                alpha=0.1,
            ),
            uncertainty_propagator=create_sage_propagator(),
            uncertainty_decomposer=UncertaintyDecomposer(num_samples=config_dict["max_samples"]),
        )

        print(f"Using SAGE v3 ({args.v3_config} profile)")
    else:
        from langgraph_sage_agent_v2 import (
            GraphDeps,
            build_graph,
            create_initial_state,
        )
        from sage_agent import create_sage_propagator

        deps = GraphDeps(
            tool_schemas=tool_schemas,
            candidate_generator=LLMBackedCandidateGenerator(llm),
            question_generator=LLMBackedQuestionGenerator(llm),
            question_asker=NoOpQuestionAsker(),
            tool_executor=ToolRegistryExecutor(tool_registry),
            constraint_extractor=SimpleConstraintExtractor(),
            config=SageAgentConfig(
                max_questions=args.max_questions,
                redundancy_weight=0.02,
                tau_execute=0.8,
                alpha=0.1,
            ),
            uncertainty_propagator=create_sage_propagator(),
        )

        print("Using SAGE v2")

    return build_graph(deps).compile(), create_initial_state


def format_code_prompt(problem: str, tool_name: str) -> str:
    """Format problem as tool calling request."""
    if tool_name == "generate_python_code":
        return f"""Generate Python code to solve this problem:

{problem}

You should call the 'generate_python_code' tool with:
- function_name: the name of the function to implement
- approach: the algorithmic approach (iterative/recursive/etc)
- code: the complete Python implementation
- explanation: brief explanation of your approach

Focus on generating correct, efficient code."""
    else:
        return f"""Solve this math problem and provide the final numeric answer:

{problem}

You should call the 'generate_solution' tool with:
- reasoning: step-by-step reasoning (show your work)
- solution: detailed solution with calculations
- answer: ONLY the final numeric answer (just the number, e.g., "42" or "3.14")

IMPORTANT: The 'answer' field must contain ONLY the final number, nothing else.

Example answer format: "42" or "3.14" or "1000"

Solve the problem step-by-step."""


def extract_code_from_result(result_state) -> Optional[str]:
    """Extract generated code from SAGE result."""
    result = result_state.get("result")
    if not result:
        return None

    args = result.arguments

    # Try to get code field
    if "code" in args:
        return args["code"]

    # Try to get solution field
    if "solution" in args:
        return args["solution"]

    # Try to get answer field (for GSM8K)
    if "answer" in args:
        answer = str(args["answer"])

        # Try to extract final number from answer
        # Look for patterns like "Answer: 42" or "= 42" or "#### 42"
        import re

        # Try "#### number" format
        match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', answer)
        if match:
            return match.group(1).replace(',', '')

        # Try "Answer: number" or "answer is number"
        match = re.search(r'(?:answer|result|total)(?:\s+is)?[:=\s]+(-?\d+(?:,\d{3})*(?:\.\d+)?)', answer, re.IGNORECASE)
        if match:
            return match.group(1).replace(',', '')

        # Try "= number" at end
        match = re.search(r'=\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)(?:\s|$)', answer)
        if match:
            return match.group(1).replace(',', '')

        # Last resort: any number in the string
        match = re.search(r'(-?\d+(?:,\d{3})*(?:\.\d+)?)', answer)
        if match:
            return match.group(1).replace(',', '')

        return answer

    return None


def load_benchmark(benchmark_name: str, limit: int):
    """Load benchmark examples."""
    if benchmark_name == "humaneval":
        from datasets import load_dataset
        dataset = load_dataset("openai/openai_humaneval", split="test")

        examples = []
        for i, item in enumerate(dataset):
            if limit > 0 and i >= limit:
                break
            examples.append({
                "id": item["task_id"],
                "prompt": item["prompt"],
                "canonical_solution": item["canonical_solution"],
                "test": item["test"],
                "entry_point": item["entry_point"],
                "tool_name": "generate_python_code",
            })
        return examples

    elif benchmark_name == "mbpp":
        from datasets import load_dataset
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")

        examples = []
        for i, item in enumerate(dataset):
            if limit > 0 and i >= limit:
                break

            # MBPP sanitized uses 'prompt' field, not 'text'
            prompt_text = item.get("prompt") or item.get("text", "")
            test_list = item.get("test_list", [])
            code = item.get("code", "")
            task_id = item.get("task_id", i)

            examples.append({
                "id": f"mbpp_{task_id}",
                "prompt": prompt_text + "\n\n" + "\n".join(test_list[:1]) if test_list else prompt_text,
                "canonical_solution": code,
                "test": "\n".join(test_list),
                "entry_point": "solution",  # MBPP typically uses 'solution' as function name
                "tool_name": "generate_python_code",
            })
        return examples

    elif benchmark_name == "gsm8k":
        from datasets import load_dataset
        dataset = load_dataset("openai/gsm8k", "main", split="test")

        examples = []
        for i, item in enumerate(dataset):
            if limit > 0 and i >= limit:
                break

            # Extract numeric answer
            answer_text = item["answer"]
            match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", answer_text)
            answer = match.group(1).replace(",", "") if match else "0"

            examples.append({
                "id": f"gsm8k_{i}",
                "prompt": item["question"],
                "canonical_solution": answer,
                "test": None,
                "entry_point": None,
                "tool_name": "generate_solution",
            })
        return examples

    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")


def evaluate_code(code: str, test: str, entry_point: str) -> bool:
    """Execute code and check if tests pass."""
    if not code or not test:
        return False

    try:
        # Create namespace
        namespace = {}

        # Execute the code
        exec(code, namespace)

        # Execute the tests
        exec(test, namespace)

        return True
    except Exception as e:
        return False


def evaluate_gsm8k_answer(generated: str, expected: str) -> bool:
    """Check if GSM8K answer is correct."""
    if not generated:
        return False

    # Extract number from generated answer
    match = re.search(r"(-?\d+(?:,\d{3})*(?:\.\d+)?)", str(generated))
    if not match:
        return False

    generated_num = match.group(1).replace(",", "")
    expected_num = expected.replace(",", "")

    try:
        return abs(float(generated_num) - float(expected_num)) < 1e-6
    except ValueError:
        return False


def run_benchmark(args, benchmark_name: str) -> BenchmarkResult:
    """Run SAGE agent on a benchmark."""

    # Load examples
    examples = load_benchmark(benchmark_name, args.limit)
    print(f"\nEvaluating {benchmark_name}: {len(examples)} examples")

    # Create tool schemas
    tool_schemas = create_code_generation_tools()

    # Create SAGE agent
    graph, create_initial_state = create_sage_agent(args, tool_schemas)

    # Evaluate
    results = []
    uncertainties = []
    correct_count = 0

    for idx, example in enumerate(examples):
        if args.print_each:
            print(f"\n{'='*60}")
            print(f"Example {idx+1}/{len(examples)}: {example['id']}")
            print(f"{'='*60}")

        # Create prompt
        prompt = format_code_prompt(example["prompt"], example["tool_name"])

        # Run SAGE agent
        initial_state = create_initial_state(prompt, tool_schemas)
        result_state = graph.invoke(initial_state, {"recursion_limit": 50})

        # Extract code
        generated = extract_code_from_result(result_state)

        # Evaluate
        if example["tool_name"] == "generate_python_code":
            is_correct = evaluate_code(
                generated,
                example["test"],
                example["entry_point"]
            )
        else:
            is_correct = evaluate_gsm8k_answer(
                generated,
                example["canonical_solution"]
            )

        if is_correct:
            correct_count += 1

        # Get uncertainty
        uncertainty = result_state.get("combined_uncertainty", 1.0)
        uncertainties.append(uncertainty)

        results.append({
            "id": example["id"],
            "correct": is_correct,
            "uncertainty": uncertainty,
            "status": result_state.get("status"),
        })

        if args.print_each:
            print(f"Status: {result_state.get('status')}")
            print(f"Correct: {is_correct}")
            print(f"Uncertainty: {uncertainty:.3f}")

            if args.use_v3:
                print(f"Epistemic: {result_state.get('epistemic_uncertainty', 'N/A')}")
                print(f"Aleatoric: {result_state.get('aleatoric_uncertainty', 'N/A')}")
                print(f"Trajectory: {result_state.get('trajectory_uncertainty', 'N/A')}")
                print(f"Samples: {result_state.get('num_samples', 1)}")
                if result_state.get('warning'):
                    print(f"⚠️ Warning: {result_state['warning']}")

            if generated:
                print(f"\nGenerated:\n{generated[:200]}...")

    # Compute metrics
    accuracy = correct_count / len(examples) if examples else 0.0
    avg_unc = sum(uncertainties) / len(uncertainties) if uncertainties else 0.0

    # Confident accuracy (correct on low uncertainty examples)
    low_unc_results = [r for r in results if r["uncertainty"] < 0.5]
    confident_acc = (
        sum(1 for r in low_unc_results if r["correct"]) / len(low_unc_results)
        if low_unc_results else 0.0
    )

    # Abstention rate (high uncertainty = abstain)
    abstention_rate = sum(1 for r in results if r["uncertainty"] > 0.7) / len(results) if results else 0.0

    # Simple ECE approximation
    ece = abs(accuracy - (1.0 - avg_unc))

    return BenchmarkResult(
        benchmark_name=benchmark_name,
        total_examples=len(examples),
        accuracy=accuracy,
        avg_uncertainty=avg_unc,
        confident_accuracy=confident_acc,
        abstention_rate=abstention_rate,
        ece=ece,
        extra_metrics={},
    )


def main():
    args = parse_args()

    # Determine benchmarks to run
    if args.benchmark == "all":
        benchmarks = ["humaneval", "mbpp", "gsm8k"]
    else:
        benchmarks = [args.benchmark]

    # Run benchmarks
    all_results = []
    for benchmark_name in benchmarks:
        try:
            result = run_benchmark(args, benchmark_name)
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Error on {benchmark_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "="*80)
    print("SAGE AGENT CODE GENERATION BENCHMARK RESULTS")
    print("="*80)

    for result in all_results:
        print(f"\n{result.benchmark_name.upper()}")
        print("-"*80)
        print(f"  Examples:           {result.total_examples}")
        print(f"  Accuracy:           {result.accuracy:.4f}")
        print(f"  Avg Uncertainty:    {result.avg_uncertainty:.4f}")
        print(f"  Confident Accuracy: {result.confident_accuracy:.4f}")
        print(f"  Abstention Rate:    {result.abstention_rate:.4f}")
        print(f"  ECE:                {result.ece:.4f}")

    # Overall average
    if all_results:
        avg_acc = sum(r.accuracy for r in all_results) / len(all_results)
        avg_unc = sum(r.avg_uncertainty for r in all_results) / len(all_results)
        print(f"\n{'='*80}")
        print(f"OVERALL AVERAGE")
        print(f"  Accuracy:        {avg_acc:.4f}")
        print(f"  Avg Uncertainty: {avg_unc:.4f}")
        print("="*80)


if __name__ == "__main__":
    main()
