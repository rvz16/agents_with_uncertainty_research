#!/usr/bin/env python3
"""Evaluate Pure SAGE-Agent on Different Datasets.

This script evaluates the clean SAGE implementation (Algorithm 1 from the paper)
on various benchmarks:
- When2Call: Tool calling disambiguation
- ClarifyBench: Simulated clarification scenarios

Usage:
    # When2Call evaluation
    python different_agents/pure_sage/run_sage_eval.py --dataset when2call --limit 10 --print-each

    # With different LLM
    python different_agents/pure_sage/run_sage_eval.py --dataset when2call --model openai/gpt-4o-mini

    # Custom hyperparameters
    python different_agents/pure_sage/run_sage_eval.py --dataset when2call --tau 0.9 --alpha 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

# Project root is 2 levels up from different_agents/pure_sage/
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "different_agents" / "shared"))

from langgraph_sage_agent import (
    GraphDeps,
    SAGEConfig,
    build_graph,
    create_initial_state,
)
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
from sage_agent.core.types import ExecutionResult


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Pure SAGE-Agent on benchmarks."
    )
    parser.add_argument(
        "--dataset",
        choices=["when2call", "clarifybench"],
        default="when2call",
        help="Dataset to evaluate on.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of examples to evaluate (0 = all).",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="Model name for LLM.",
    )
    parser.add_argument(
        "--service-url",
        default="http://localhost:8001/v1",
        help="LLM service URL (for TTS service).",
    )
    parser.add_argument(
        "--use-openrouter",
        action="store_true",
        help="Use OpenRouter instead of TTS service.",
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

    # SAGE hyperparameters (paper defaults)
    parser.add_argument("--tau", type=float, default=0.85, help="τ_exec threshold")
    parser.add_argument("--alpha", type=float, default=0.1, help="α termination factor")
    parser.add_argument("--lambda-r", type=float, default=0.5, help="λ redundancy weight")
    parser.add_argument("--max-questions", type=int, default=6, help="T_max")

    parser.add_argument("--print-each", action="store_true", help="Print per-example results")
    parser.add_argument("--recursion-limit", type=int, default=50, help="LangGraph recursion limit")

    return parser.parse_args()


# =============================================================================
# LLM Client Setup
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
        # Fallback to OpenRouter
        from openrouter_client import OpenRouterClient
        return OpenRouterClient(model=args.model)


# =============================================================================
# When2Call Dataset
# =============================================================================

def load_when2call(limit: int = 0) -> List[dict]:
    """Load When2Call dataset."""
    from datasets import load_dataset

    dataset = load_dataset("nvidia/When2Call", "test")
    rows = list(dataset["llm_judge"])

    # Filter to tool_call examples (where ground truth is a tool call)
    rows = [r for r in rows if r.get("correct_answer") == "tool_call"]

    if limit > 0:
        rows = rows[:limit]

    return rows


def parse_tool_schema(tool_json: dict) -> ToolSchema:
    """Parse When2Call tool definition to ToolSchema."""
    name = tool_json.get("name", "")
    params = tool_json.get("parameters", {}) or {}
    required = params.get("required", []) or []
    properties = params.get("properties", {}) or {}

    domains: Dict[str, ParameterDomain] = {}
    for param_name, prop in properties.items():
        enum = prop.get("enum")
        if isinstance(enum, list) and enum:
            domains[param_name] = ParameterDomain.from_values(enum)
        elif prop.get("type") == "boolean":
            domains[param_name] = ParameterDomain.from_values([True, False])
        else:
            domains[param_name] = ParameterDomain.continuous()

    for param_name in required:
        if param_name not in domains:
            domains[param_name] = ParameterDomain.continuous()

    return ToolSchema(name=name, parameters=domains, required=frozenset(required))


def parse_tool_call(raw: str) -> ToolCall:
    """Parse When2Call tool call answer."""
    payload = json.loads(raw)
    return ToolCall(
        tool_name=payload.get("name", ""),
        arguments=payload.get("arguments", {}),
    )


@dataclass
class GroundTruthAsker:
    """Question asker that returns ground truth values."""
    truth: ToolCall
    count: int = 0

    def ask(self, question) -> str:
        self.count += 1
        if not getattr(question, "aspects", None):
            return ""

        values = []
        for aspect in question.aspects:
            if aspect.tool_name != self.truth.tool_name:
                continue
            value = self.truth.arguments.get(aspect.param_name)
            if value is not None:
                values.append(str(value))

        return "; ".join(values)


def evaluate_when2call(args: argparse.Namespace) -> None:
    """Evaluate on When2Call dataset."""
    print("=" * 60)
    print("Pure SAGE-Agent Evaluation on When2Call")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Hyperparameters: τ={args.tau}, α={args.alpha}, λ={args.lambda_r}, T_max={args.max_questions}")
    print()

    # Load dataset
    rows = load_when2call(args.limit)
    print(f"Loaded {len(rows)} examples\n")

    # Get LLM client
    llm = get_llm_client(args)

    # SAGE config with paper defaults (or custom)
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

    for i, row in enumerate(rows):
        # Parse tools
        tool_payloads = row.get("tools", [])
        if not tool_payloads:
            continue

        tool_schemas = []
        for tool_str in tool_payloads:
            tool_json = json.loads(tool_str)
            tool_schemas.append(parse_tool_schema(tool_json))

        tool_schemas_dict = {t.name: t for t in tool_schemas}

        # Parse ground truth
        tool_call_raw = row.get("answers", {}).get("tool_call")
        if not tool_call_raw:
            continue
        truth = parse_tool_call(tool_call_raw)

        # Create question asker with ground truth
        question_asker = GroundTruthAsker(truth=truth)

        # Build dependencies
        deps = GraphDeps(
            tool_schemas=tool_schemas_dict,
            candidate_generator=LLMBackedCandidateGenerator(llm),
            question_generator=LLMBackedQuestionGenerator(llm),
            question_asker=question_asker,
            tool_executor=ToolRegistryExecutor({
                t.name: lambda _: ExecutionResult(success=True, output={"ok": True})
                for t in tool_schemas
            }),
            constraint_extractor=SimpleConstraintExtractor(),
            config=config,
        )

        # Build and run graph
        graph = build_graph(deps).compile()

        initial_state = create_initial_state(
            user_input=row.get("question", ""),
            tool_schemas=tool_schemas_dict,
        )

        result = graph.invoke(initial_state, {"recursion_limit": args.recursion_limit})

        # Extract prediction
        pred = result.get("result") or ToolCall("", {})
        predictions.append(pred)
        ground_truths.append(truth)
        question_counts.append(question_asker.count)

        if args.print_each:
            print(f"[{i+1}/{len(rows)}] {row.get('uuid', '')[:8]}")
            print(f"  Question: {row.get('question', '')[:80]}...")
            print(f"  Pred: {pred.tool_name}({dict(pred.arguments)})")
            print(f"  Truth: {truth.tool_name}({dict(truth.arguments)})")
            print(f"  Questions: {question_asker.count}, Status: {result['status']}")
            print(f"  Max prob: {result['max_prob']:.2%}")
            print("-" * 40)
        elif (i + 1) % 5 == 0:
            print(f"Processed {i+1}/{len(rows)}")

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


# =============================================================================
# ClarifyBench Simulation
# =============================================================================

def evaluate_clarifybench(args: argparse.Namespace) -> None:
    """Evaluate on simulated ClarifyBench scenarios."""
    print("=" * 60)
    print("Pure SAGE-Agent Evaluation on ClarifyBench (Simulated)")
    print("=" * 60)

    from sage_agent import ClarifyBenchSimulator, SimulationScenario

    # Get LLM client
    llm = get_llm_client(args)

    # SAGE config
    config = SAGEConfig(
        tau_exec=args.tau,
        alpha=args.alpha,
        lambda_redundancy=args.lambda_r,
        max_questions=args.max_questions,
    )

    # Define example scenarios
    scenarios = [
        SimulationScenario(
            scenario_id="flight_booking",
            requests=["Book me a flight to New York"],
            ground_truth=[ToolCall("book_flight", {"origin": "LAX", "destination": "NYC", "date": "2024-03-15"})],
        ),
        SimulationScenario(
            scenario_id="restaurant_search",
            requests=["Find me a good Italian restaurant"],
            ground_truth=[ToolCall("find_restaurant", {"cuisine": "italian", "location": "downtown"})],
        ),
    ]

    if args.limit > 0:
        scenarios = scenarios[:args.limit]

    # Define tool schemas for scenarios
    flight_tool = ToolSchema(
        name="book_flight",
        parameters={
            "origin": ParameterDomain.from_values(["NYC", "LAX", "SFO", "BOS"]),
            "destination": ParameterDomain.from_values(["NYC", "LAX", "SFO", "BOS"]),
            "date": ParameterDomain.continuous(),
        },
        required=frozenset({"origin", "destination", "date"}),
    )

    restaurant_tool = ToolSchema(
        name="find_restaurant",
        parameters={
            "cuisine": ParameterDomain.from_values(["italian", "japanese", "mexican", "chinese"]),
            "location": ParameterDomain.from_values(["downtown", "midtown", "uptown"]),
        },
        required=frozenset({"cuisine", "location"}),
    )

    tool_schemas = {flight_tool.name: flight_tool, restaurant_tool.name: restaurant_tool}

    # Simple user simulator
    class SimpleUserSimulator:
        def answer(self, question, scenario_id=None):
            # Return simple answers based on scenario
            text = question.text.lower()
            if "origin" in text or "from" in text or "departing" in text:
                return "LAX"
            if "destination" in text or "to" in text:
                return "NYC"
            if "date" in text or "when" in text:
                return "2024-03-15"
            if "cuisine" in text or "type" in text:
                return "italian"
            if "location" in text or "neighborhood" in text:
                return "downtown"
            return "I don't know"

    # Build SAGE agent
    from sage_agent import SageAgent, SageAgentConfig

    agent = SageAgent(
        tool_schemas=list(tool_schemas.values()),
        candidate_generator=LLMBackedCandidateGenerator(llm),
        question_generator=LLMBackedQuestionGenerator(llm),
        question_asker=None,  # Will be set by simulator
        tool_executor=ToolRegistryExecutor({
            t: lambda _: ExecutionResult(success=True, output={"ok": True})
            for t in tool_schemas
        }),
        constraint_extractor=SimpleConstraintExtractor(),
        config=SageAgentConfig(
            max_questions=config.max_questions,
            tau_execute=config.tau_exec,
            alpha=config.alpha,
            redundancy_weight=config.lambda_redundancy,
        ),
    )

    # Run simulator
    simulator = ClarifyBenchSimulator(
        agent=agent,
        user_simulator=SimpleUserSimulator(),
    )

    for scenario in scenarios:
        print(f"\nScenario: {scenario.scenario_id}")
        print("-" * 40)

        result = simulator.run(scenario)

        for turn in result.turns:
            print(f"  Request: {turn.request}")
            print(f"  Result: {turn.tool_call}")
            print(f"  Questions: {turn.questions_asked}")

        print(f"\nMetrics:")
        print(f"  Tool match: {result.metrics.tool_match_rate:.2%}")
        print(f"  Param match: {result.metrics.parameter_match_rate:.2%}")
        print(f"  Avg questions: {result.metrics.avg_questions:.1f}")

    print("\n" + "=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if args.dataset == "when2call":
        evaluate_when2call(args)
    elif args.dataset == "clarifybench":
        evaluate_clarifybench(args)
    else:
        print(f"Unknown dataset: {args.dataset}")
        sys.exit(1)


if __name__ == "__main__":
    main()
