#!/usr/bin/env python3
"""LangGraph integration example for SAGE-Agent.

This example demonstrates using SAGE-Agent with LangGraph for
structured state management and graph-based execution.

Usage:
    pip install sage-agent[langgraph]
    export OPENROUTER_API_KEY="your-key"
    python examples/langgraph_example.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Mapping

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from langgraph.graph import StateGraph
except ImportError:
    print("LangGraph is required. Install with: pip install langgraph")
    sys.exit(1)

from sage_agent import (
    SAGEGraph,
    SAGEGraphConfig,
    ParameterDomain,
    ToolSchema,
    ToolCall,
    ToolCallCandidate,
    ExecutionResult,
    Question,
    Aspect,
    UNK,
)
from sage_agent.core.types import (
    CandidateGenerator,
    QuestionGenerator,
    QuestionAsker,
    ToolExecutor,
    ConstraintExtractor,
)

# Try to import OpenRouter client
try:
    from sage_agent.llm import OpenRouterClient
    HAS_OPENROUTER = True
except ImportError:
    HAS_OPENROUTER = False


# =============================================================================
# Tool Definitions
# =============================================================================

RESTAURANT_TOOL = ToolSchema(
    name="find_restaurant",
    parameters={
        "cuisine": ParameterDomain.from_values([
            "italian", "japanese", "mexican", "chinese", "indian", "french"
        ]),
        "location": ParameterDomain.from_values([
            "downtown", "midtown", "uptown", "brooklyn", "queens"
        ]),
        "price_range": ParameterDomain.from_values(["$", "$$", "$$$", "$$$$"]),
        "party_size": ParameterDomain.continuous(),  # Any integer
    },
    required=frozenset({"cuisine", "location"}),
)


# =============================================================================
# Implementations
# =============================================================================

@dataclass
class SimpleCandidateGenerator:
    """Generate candidates from user input."""

    def generate_candidates(
        self,
        user_input: str,
        observations: List[str],
        tool_schemas: Mapping[str, ToolSchema],
    ) -> List[ToolCallCandidate]:
        args = {"cuisine": UNK, "location": UNK, "price_range": "$$", "party_size": 2}
        user_lower = user_input.lower()

        # Parse cuisine
        cuisines = ["italian", "japanese", "mexican", "chinese", "indian", "french"]
        for c in cuisines:
            if c in user_lower:
                args["cuisine"] = c
                break

        # Parse location
        locations = ["downtown", "midtown", "uptown", "brooklyn", "queens"]
        for loc in locations:
            if loc in user_lower:
                args["location"] = loc
                break

        # Parse from observations
        for obs in observations:
            obs_lower = obs.lower()
            for c in cuisines:
                if c in obs_lower:
                    args["cuisine"] = c
            for loc in locations:
                if loc in obs_lower:
                    args["location"] = loc
            for price in ["$", "$$", "$$$", "$$$$"]:
                if price in obs:
                    args["price_range"] = price

        return [ToolCallCandidate(tool_name="find_restaurant", arguments=args)]


@dataclass
class SimpleQuestionGenerator:
    """Generate clarifying questions."""

    def generate_questions(
        self,
        user_input: str,
        candidates: List[ToolCallCandidate],
        observations: List[str],
        tool_schemas: Mapping[str, ToolSchema],
    ) -> List[Question]:
        questions = []

        for candidate in candidates:
            schema = tool_schemas.get(candidate.tool_name)
            if not schema:
                continue

            for param in schema.required:
                if candidate.arguments.get(param, UNK) == UNK:
                    if param == "cuisine":
                        questions.append(Question(
                            text="What type of cuisine are you in the mood for?",
                            aspects=(Aspect(candidate.tool_name, param),),
                        ))
                    elif param == "location":
                        questions.append(Question(
                            text="Which neighborhood would you prefer?",
                            aspects=(Aspect(candidate.tool_name, param),),
                        ))

        return questions


class ConsoleAsker:
    """Interactive console question asker."""

    def ask(self, question: Question) -> str:
        print(f"\n🤔 Agent: {question.text}")
        return input("You: ").strip()


@dataclass
class MockRestaurantExecutor:
    """Simulate restaurant search."""

    def execute(self, tool_call: ToolCall) -> ExecutionResult:
        args = dict(tool_call.arguments)
        return ExecutionResult(
            success=True,
            output={
                "restaurants": [
                    {"name": f"Best {args.get('cuisine', 'Food')} Place", "rating": 4.5},
                    {"name": f"The {args.get('location', 'Local')} Kitchen", "rating": 4.2},
                ],
                "query": args,
            },
        )


@dataclass
class SimpleExtractor:
    """Extract constraints from responses."""

    def update_domain(self, domain: ParameterDomain, response: str) -> ParameterDomain:
        response_lower = response.lower()

        if domain.values:
            for value in domain.values:
                if isinstance(value, str) and value.lower() in response_lower:
                    return ParameterDomain.from_values([value])

        return domain


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the LangGraph SAGE-Agent demo."""
    print("=" * 60)
    print("SAGE-Agent with LangGraph")
    print("=" * 60)
    print("\nThis demo shows SAGE-Agent using LangGraph for state management.\n")

    # Configuration matching paper defaults
    config = SAGEGraphConfig(
        tau_exec=0.85,
        alpha=0.1,
        lambda_redundancy=0.5,
        max_questions=4,
    )

    # Create the SAGE graph
    graph = SAGEGraph(
        tool_schemas=[RESTAURANT_TOOL],
        candidate_generator=SimpleCandidateGenerator(),
        question_generator=SimpleQuestionGenerator(),
        question_asker=ConsoleAsker(),
        tool_executor=MockRestaurantExecutor(),
        constraint_extractor=SimpleExtractor(),
        config=config,
    )

    # User request
    user_request = "I want to find a good restaurant"
    print(f"📝 User: {user_request}\n")

    # Run the graph
    result = graph.run(user_request)

    # Display results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    if result["status"] == "escalated":
        print("⚠️  Request escalated")
        print(f"   Error: {result.get('error')}")
    elif result["status"] == "done":
        if result["tool_call"]:
            print(f"✅ Executed: {result['tool_call'].tool_name}")
            print(f"   Arguments: {dict(result['tool_call'].arguments)}")
        if result["execution_result"]:
            print(f"   Success: {result['execution_result'].success}")
            if result["execution_result"].output:
                output = result["execution_result"].output
                print(f"   Found {len(output.get('restaurants', []))} restaurants")

    print(f"\n📊 Statistics:")
    print(f"   Steps: {result['step']}")
    print(f"   Final max_prob: {result['max_prob']:.2%}")
    print(f"   Status: {result['status']}")
    if result.get("termination_reason"):
        print(f"   Termination: {result['termination_reason']}")


if __name__ == "__main__":
    main()
