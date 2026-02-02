#!/usr/bin/env python3
"""Evaluate SAGE-Agent on Real ClarifyBench Dataset.

ClarifyBench is a benchmark for evaluating LLM agents on tasks requiring clarification.
Dataset source: /Users/victor/Documents/vs_files/research/article_implementation/ClarifyBench/

Usage:
    # Evaluate on sample data
    python different_agents/evaluations/run_clarifybench_eval.py --split sample --limit 5 --print-each

    # Evaluate on ClarifyBench_A (ambiguous)
    python different_agents/evaluations/run_clarifybench_eval.py --split A --limit 20

    # Evaluate on ClarifyBench_E (explicit)
    python different_agents/evaluations/run_clarifybench_eval.py --split E --limit 20

    # Evaluate on ClarifyBench_I (implicit)
    python different_agents/evaluations/run_clarifybench_eval.py --split I --limit 20

    # With different LLM
    python different_agents/evaluations/run_clarifybench_eval.py --split sample --use-openrouter --model openai/gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Project root is 2 levels up from different_agents/evaluations/
ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = ROOT / "different_agents" / "shared"
PURE_SAGE_DIR = ROOT / "different_agents" / "pure_sage"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SHARED_DIR))
sys.path.insert(0, str(PURE_SAGE_DIR))

from sage_agent import (
    LLMBackedCandidateGenerator,
    LLMBackedQuestionGenerator,
    ParameterDomain,
    SimpleConstraintExtractor,
    ToolCall,
    ToolRegistryExecutor,
    ToolSchema,
    evaluate_metrics,
)
from sage_agent.core.types import ExecutionResult, Question

from langgraph_sage_agent import (
    GraphDeps,
    SAGEConfig,
    build_graph,
    create_initial_state,
)


# =============================================================================
# ClarifyBench Data Loading
# =============================================================================

CLARIFYBENCH_ROOT = Path("/Users/victor/Documents/vs_files/research/article_implementation/ClarifyBench/ClarifyBench")

SPLIT_DIRS = {
    "sample": CLARIFYBENCH_ROOT / "sample",
    "A": CLARIFYBENCH_ROOT / "ClarifyBench_A",
    "E": CLARIFYBENCH_ROOT / "ClarifyBench_E",
    "I": CLARIFYBENCH_ROOT / "ClarifyBench_I",
}


@dataclass
class ClarifyBenchExample:
    """A single ClarifyBench example."""
    example_id: str
    user_query: str
    follow_ups: List[str]
    ground_truth_calls: List[Dict[str, Any]]
    user_intention: str
    initial_config: Dict[str, Any]
    primary_api: str


def load_clarifybench(split: str, limit: int = 0) -> List[ClarifyBenchExample]:
    """Load ClarifyBench examples from JSON files."""
    split_dir = SPLIT_DIRS.get(split)
    if split_dir is None or not split_dir.exists():
        raise ValueError(f"Unknown or missing split: {split}. Available: {list(SPLIT_DIRS.keys())}")

    examples = []
    json_files = sorted(split_dir.glob("*.json"))

    if limit > 0:
        json_files = json_files[:limit]

    for json_path in json_files:
        with open(json_path, "r") as f:
            data = json.load(f)

        examples.append(ClarifyBenchExample(
            example_id=json_path.stem,
            user_query=data.get("user_query", ""),
            follow_ups=data.get("potential_follow_ups", []),
            ground_truth_calls=data.get("ground_truth_tool_calls", []),
            user_intention=data.get("user_intention", ""),
            initial_config=data.get("initial_config", {}),
            primary_api=data.get("primary_api", ""),
        ))

    return examples


# =============================================================================
# Tool Schema Generation
# =============================================================================

# Common file system tools used in ClarifyBench
FILESYSTEM_TOOLS = {
    "touch": ToolSchema(
        name="touch",
        parameters={
            "file_name": ParameterDomain.continuous(),
        },
        required=frozenset({"file_name"}),
    ),
    "cp": ToolSchema(
        name="cp",
        parameters={
            "source": ParameterDomain.continuous(),
            "destination": ParameterDomain.continuous(),
        },
        required=frozenset({"source", "destination"}),
    ),
    "mv": ToolSchema(
        name="mv",
        parameters={
            "source": ParameterDomain.continuous(),
            "destination": ParameterDomain.continuous(),
        },
        required=frozenset({"source", "destination"}),
    ),
    "rm": ToolSchema(
        name="rm",
        parameters={
            "file_name": ParameterDomain.continuous(),
        },
        required=frozenset({"file_name"}),
    ),
    "mkdir": ToolSchema(
        name="mkdir",
        parameters={
            "dir_name": ParameterDomain.continuous(),
        },
        required=frozenset({"dir_name"}),
    ),
    "ls": ToolSchema(
        name="ls",
        parameters={
            "path": ParameterDomain.continuous(),
        },
        required=frozenset(),
    ),
    "cat": ToolSchema(
        name="cat",
        parameters={
            "file_name": ParameterDomain.continuous(),
        },
        required=frozenset({"file_name"}),
    ),
    "grep": ToolSchema(
        name="grep",
        parameters={
            "file_name": ParameterDomain.continuous(),
            "pattern": ParameterDomain.continuous(),
        },
        required=frozenset({"file_name", "pattern"}),
    ),
    "echo": ToolSchema(
        name="echo",
        parameters={
            "content": ParameterDomain.continuous(),
            "file_name": ParameterDomain.continuous(),
        },
        required=frozenset({"content"}),
    ),
    "cd": ToolSchema(
        name="cd",
        parameters={
            "path": ParameterDomain.continuous(),
        },
        required=frozenset({"path"}),
    ),
    "pwd": ToolSchema(
        name="pwd",
        parameters={},
        required=frozenset(),
    ),
    "find": ToolSchema(
        name="find",
        parameters={
            "path": ParameterDomain.continuous(),
            "name": ParameterDomain.continuous(),
        },
        required=frozenset({"path"}),
    ),
    "head": ToolSchema(
        name="head",
        parameters={
            "file_name": ParameterDomain.continuous(),
            "lines": ParameterDomain.continuous(),
        },
        required=frozenset({"file_name"}),
    ),
    "tail": ToolSchema(
        name="tail",
        parameters={
            "file_name": ParameterDomain.continuous(),
            "lines": ParameterDomain.continuous(),
        },
        required=frozenset({"file_name"}),
    ),
    "wc": ToolSchema(
        name="wc",
        parameters={
            "file_name": ParameterDomain.continuous(),
            "mode": ParameterDomain.from_values(["lines", "words", "chars"]),
        },
        required=frozenset({"file_name"}),
    ),
    "sort": ToolSchema(
        name="sort",
        parameters={
            "file_name": ParameterDomain.continuous(),
        },
        required=frozenset({"file_name"}),
    ),
    "diff": ToolSchema(
        name="diff",
        parameters={
            "file1": ParameterDomain.continuous(),
            "file2": ParameterDomain.continuous(),
        },
        required=frozenset({"file1", "file2"}),
    ),
}


def get_tools_for_example(example: ClarifyBenchExample) -> Dict[str, ToolSchema]:
    """Get tool schemas relevant to an example."""
    # Collect all tool names from ground truth
    tool_names: Set[str] = set()
    for call in example.ground_truth_calls:
        tool_names.add(call.get("tool_name", ""))

    # Return matching schemas
    tools = {}
    for name in tool_names:
        if name in FILESYSTEM_TOOLS:
            tools[name] = FILESYSTEM_TOOLS[name]
        else:
            # Create a generic schema for unknown tools
            params = example.ground_truth_calls[0].get("parameters", {}) if example.ground_truth_calls else {}
            tools[name] = ToolSchema(
                name=name,
                parameters={p: ParameterDomain.continuous() for p in params},
                required=frozenset(params.keys()),
            )

    # Always include common tools
    for name in ["touch", "cp", "mv", "rm", "cat", "grep", "ls"]:
        if name not in tools:
            tools[name] = FILESYSTEM_TOOLS[name]

    return tools


# =============================================================================
# User Simulator
# =============================================================================

class ClarifyBenchUserSimulator:
    """Simulates user responses based on ClarifyBench user_intention."""

    def __init__(self, example: ClarifyBenchExample):
        self.example = example
        self.intention = example.user_intention.lower()
        self.ground_truth = example.ground_truth_calls
        self.call_count = 0

    def ask(self, question: Question) -> str:
        """Answer a clarification question based on user intention."""
        self.call_count += 1
        q_text = question.text.lower()

        # Try to extract relevant info from user_intention
        # Look for patterns like "user clarifies that X is Y"
        clarifies_pattern = r"user clarifies that (?:the )?(\w+) is ['\"]?([^'\"]+)['\"]?"
        matches = re.findall(clarifies_pattern, self.intention, re.IGNORECASE)

        for key, value in matches:
            if key.lower() in q_text:
                return value

        # Try to match ground truth parameters
        for call in self.ground_truth:
            params = call.get("parameters", {})
            for param_name, param_value in params.items():
                if param_name.lower() in q_text or param_name.replace("_", " ") in q_text:
                    return str(param_value)

        # Default response
        return "I'm not sure"


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAGE-Agent on ClarifyBench.")

    parser.add_argument(
        "--split",
        choices=["sample", "A", "E", "I"],
        default="sample",
        help="ClarifyBench split: sample, A (ambiguous), E (explicit), I (implicit)",
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of examples (0=all)")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="Model name")
    parser.add_argument("--service-url", default="http://localhost:8001/v1", help="LLM service URL")
    parser.add_argument("--use-openrouter", action="store_true", help="Use OpenRouter")
    parser.add_argument("--use-ollama", action="store_true", help="Use Ollama")
    parser.add_argument("--ollama-model", default="qwen2.5:7b-instruct", help="Ollama model")

    # SAGE hyperparameters
    parser.add_argument("--tau", type=float, default=0.85, help="τ_exec threshold")
    parser.add_argument("--alpha", type=float, default=0.1, help="α termination factor")
    parser.add_argument("--lambda-r", type=float, default=0.5, help="λ redundancy weight")
    parser.add_argument("--max-questions", type=int, default=6, help="T_max")

    parser.add_argument("--print-each", action="store_true", help="Print per-example results")
    parser.add_argument("--recursion-limit", type=int, default=50, help="LangGraph recursion limit")

    return parser.parse_args()


# =============================================================================
# LLM Client
# =============================================================================

def get_llm_client(args: argparse.Namespace):
    """Get LLM client based on arguments."""
    if args.use_ollama:
        from ollama_client import OllamaClient
        return OllamaClient(model=args.ollama_model)

    if args.use_openrouter:
        from openrouter_client import OpenRouterClient
        return OpenRouterClient(model=args.model)

    # Default: TTS service
    try:
        from tts_llm_client import TTSLLMClient
        return TTSLLMClient(
            base_url=args.service_url,
            model=args.model,
            tts_budget=8,
        )
    except Exception:
        from openrouter_client import OpenRouterClient
        return OpenRouterClient(model=args.model)


# =============================================================================
# Main Evaluation
# =============================================================================

def main():
    args = parse_args()

    print("=" * 60)
    print("SAGE-Agent Evaluation on ClarifyBench")
    print("=" * 60)
    print(f"Split: {args.split}")
    print(f"Model: {args.model}")
    print(f"Hyperparameters: τ={args.tau}, α={args.alpha}, λ={args.lambda_r}, T_max={args.max_questions}")
    print()

    # Load data
    try:
        examples = load_clarifybench(args.split, args.limit)
    except ValueError as e:
        print(f"Error: {e}")
        print(f"\nMake sure ClarifyBench is at: {CLARIFYBENCH_ROOT}")
        sys.exit(1)

    print(f"Loaded {len(examples)} examples\n")

    if not examples:
        print("No examples found!")
        sys.exit(1)

    # Get LLM client
    llm = get_llm_client(args)

    # SAGE config
    config = SAGEConfig(
        tau_exec=args.tau,
        alpha=args.alpha,
        lambda_redundancy=args.lambda_r,
        max_questions=args.max_questions,
    )

    # Results tracking
    predictions: List[ToolCall] = []
    ground_truths: List[ToolCall] = []
    question_counts: List[int] = []

    for i, example in enumerate(examples):
        # Get tools for this example
        tool_schemas = get_tools_for_example(example)

        # Create user simulator
        user_sim = ClarifyBenchUserSimulator(example)

        # Build graph dependencies
        deps = GraphDeps(
            tool_schemas=tool_schemas,
            candidate_generator=LLMBackedCandidateGenerator(llm),
            question_generator=LLMBackedQuestionGenerator(llm),
            question_asker=user_sim,
            tool_executor=ToolRegistryExecutor({
                t: lambda _: ExecutionResult(success=True, output={"ok": True})
                for t in tool_schemas
            }),
            constraint_extractor=SimpleConstraintExtractor(),
            config=config,
        )

        # Build and run graph
        graph = build_graph(deps).compile()
        initial = create_initial_state(example.user_query, tool_schemas)

        try:
            result = graph.invoke(initial, {"recursion_limit": args.recursion_limit})
        except Exception as e:
            if args.print_each:
                print(f"[{i+1}/{len(examples)}] {example.example_id} - ERROR: {e}")
            predictions.append(ToolCall("", {}))
            if example.ground_truth_calls:
                gt = example.ground_truth_calls[0]
                ground_truths.append(ToolCall(gt.get("tool_name", ""), gt.get("parameters", {})))
            else:
                ground_truths.append(ToolCall("", {}))
            question_counts.append(0)
            continue

        # Extract prediction (first tool call only for now)
        pred = result.get("result") or ToolCall("", {})
        predictions.append(pred)

        # Ground truth (first call)
        if example.ground_truth_calls:
            gt = example.ground_truth_calls[0]
            ground_truths.append(ToolCall(gt.get("tool_name", ""), gt.get("parameters", {})))
        else:
            ground_truths.append(ToolCall("", {}))

        question_counts.append(user_sim.call_count)

        if args.print_each:
            print(f"[{i+1}/{len(examples)}] {example.example_id}")
            print(f"  Query: {example.user_query[:60]}...")
            print(f"  Pred: {pred.tool_name}({dict(pred.arguments)})")
            gt = example.ground_truth_calls[0] if example.ground_truth_calls else {}
            print(f"  Truth: {gt.get('tool_name', '')}({gt.get('parameters', {})})")
            print(f"  Questions: {user_sim.call_count}, Status: {result.get('status')}")
            print(f"  Max prob: {result.get('max_prob', 0):.2%}")
            print("-" * 40)
        elif (i + 1) % 5 == 0:
            print(f"Processed {i+1}/{len(examples)}")

    # Compute metrics
    metrics = evaluate_metrics(predictions, ground_truths, question_counts)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Examples evaluated: {len(predictions)}")
    print()
    print("Metrics:")
    print(f"  Tool match rate:      {metrics.tool_match_rate:.4f}")
    print(f"  Parameter match rate: {metrics.parameter_match_rate:.4f}")
    print(f"  Coverage rate:        {metrics.coverage_rate:.4f}")
    print(f"  Avg questions:        {metrics.avg_questions:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
