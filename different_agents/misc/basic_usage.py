#!/usr/bin/env python3
"""Basic usage example for SAGE-Agent.

This example demonstrates the canonical SAGE-Agent implementation
following Algorithm 1 from the paper.

Usage:
    export OPENROUTER_API_KEY="your-key"
    python examples/basic_usage.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Mapping

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sage_agent import (
    SAGEAgent,
    SAGEConfig,
    SAGEResult,
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
from sage_agent.core.domains import ParameterDomain


# =============================================================================
# Example Tool Schema
# =============================================================================

FLIGHT_TOOL = ToolSchema(
    name="book_flight",
    parameters={
        "origin": ParameterDomain.from_values(["NYC", "BOS", "LAX", "SFO", "ORD"]),
        "destination": ParameterDomain.from_values(["NYC", "BOS", "LAX", "SFO", "ORD"]),
        "date": ParameterDomain.from_values([
            "2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18"
        ]),
        "class": ParameterDomain.from_values(["economy", "business", "first"]),
    },
    required=frozenset({"origin", "destination", "date"}),
)


# =============================================================================
# Simple Mock Implementations
# =============================================================================

@dataclass
class MockCandidateGenerator:
    """Simple candidate generator for demonstration."""

    def generate_candidates(
        self,
        user_input: str,
        observations: List[str],
        tool_schemas: Mapping[str, ToolSchema],
    ) -> List[ToolCallCandidate]:
        """Generate candidates based on user input."""
        # Parse simple patterns from input
        args = {"origin": UNK, "destination": UNK, "date": UNK, "class": "economy"}

        user_lower = user_input.lower()

        # Check for destinations
        for city in ["nyc", "bos", "lax", "sfo", "ord"]:
            if city in user_lower:
                if args["destination"] == UNK:
                    args["destination"] = city.upper()
                elif args["origin"] == UNK:
                    args["origin"] = city.upper()

        # Check observations for additional info
        for obs in observations:
            obs_lower = obs.lower()
            for city in ["nyc", "bos", "lax", "sfo", "ord"]:
                if city in obs_lower:
                    if "from" in obs_lower and args["origin"] == UNK:
                        args["origin"] = city.upper()
                    elif "to" in obs_lower and args["destination"] == UNK:
                        args["destination"] = city.upper()
            for date in ["2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18"]:
                if date in obs:
                    args["date"] = date

        return [ToolCallCandidate(tool_name="book_flight", arguments=args)]


@dataclass
class MockQuestionGenerator:
    """Simple question generator for demonstration."""

    def generate_questions(
        self,
        user_input: str,
        candidates: List[ToolCallCandidate],
        observations: List[str],
        tool_schemas: Mapping[str, ToolSchema],
    ) -> List[Question]:
        """Generate questions about unknown parameters."""
        questions = []

        for candidate in candidates:
            if candidate.tool_name not in tool_schemas:
                continue

            schema = tool_schemas[candidate.tool_name]

            # Ask about unknown required parameters
            for param in schema.required:
                if candidate.arguments.get(param, UNK) == UNK:
                    if param == "origin":
                        questions.append(Question(
                            text="Which city will you be departing from?",
                            aspects=(Aspect(tool_name=candidate.tool_name, param_name=param),),
                        ))
                    elif param == "destination":
                        questions.append(Question(
                            text="Which city do you want to fly to?",
                            aspects=(Aspect(tool_name=candidate.tool_name, param_name=param),),
                        ))
                    elif param == "date":
                        questions.append(Question(
                            text="When would you like to travel?",
                            aspects=(Aspect(tool_name=candidate.tool_name, param_name=param),),
                        ))

        return questions


class ConsoleQuestionAsker:
    """Ask questions via console input."""

    def ask(self, question: Question) -> str:
        print(f"\n🤔 Agent: {question.text}")
        response = input("You: ").strip()
        return response


@dataclass
class MockToolExecutor:
    """Mock tool executor that simulates booking."""

    def execute(self, tool_call: ToolCall) -> ExecutionResult:
        """Simulate tool execution."""
        print(f"\n✈️ Booking flight: {tool_call.arguments}")
        return ExecutionResult(
            success=True,
            output={
                "confirmation": "FL12345",
                "details": dict(tool_call.arguments),
            },
        )


@dataclass
class SimpleConstraintExtractor:
    """Extract constraints from user responses."""

    def update_domain(
        self,
        domain: ParameterDomain,
        response: str,
    ) -> ParameterDomain:
        """Update domain based on response."""
        response_lower = response.lower().strip()

        # Try to find matching values in domain
        if domain.values:
            for value in domain.values:
                if isinstance(value, str) and value.lower() in response_lower:
                    return ParameterDomain.from_values([value])

        # For cities
        city_map = {"new york": "NYC", "boston": "BOS", "los angeles": "LAX",
                    "san francisco": "SFO", "chicago": "ORD"}
        for name, code in city_map.items():
            if name in response_lower:
                if domain.values and code in domain.values:
                    return ParameterDomain.from_values([code])

        return domain


# =============================================================================
# Main Demo
# =============================================================================

def main():
    """Run the SAGE-Agent demo."""
    print("=" * 60)
    print("SAGE-Agent: Structured Uncertainty Guided Clarification")
    print("=" * 60)
    print("\nThis demo shows the SAGE-Agent decision loop (Algorithm 1)")
    print("asking clarifying questions when uncertain about parameters.\n")

    # Create agent with paper-default configuration
    config = SAGEConfig(
        tau_exec=0.85,      # Execute when 85% confident
        alpha=0.1,          # Stop asking when score < 10% of max_prob
        lambda_redundancy=0.5,  # Redundancy cost weight
        epsilon=1e-4,       # Small prob for infinite domains
        max_questions=6,    # Max clarification questions
    )

    agent = SAGEAgent(
        tool_schemas=[FLIGHT_TOOL],
        candidate_generator=MockCandidateGenerator(),
        question_generator=MockQuestionGenerator(),
        question_asker=ConsoleQuestionAsker(),
        tool_executor=MockToolExecutor(),
        constraint_extractor=SimpleConstraintExtractor(),
        config=config,
    )

    # Example request with missing information
    user_request = "I need to book a flight to LAX"
    print(f"📝 User: {user_request}\n")

    # Run the agent
    result: SAGEResult = agent.run(user_request)

    # Display results
    print("\n" + "=" * 60)
    print("SAGE-Agent Results")
    print("=" * 60)

    if result.escalated:
        print("⚠️  Request escalated to human operator")
    elif result.tool_call:
        print(f"✅ Executed: {result.tool_call.tool_name}")
        print(f"   Arguments: {dict(result.tool_call.arguments)}")
        if result.execution_result:
            print(f"   Success: {result.execution_result.success}")
            if result.execution_result.output:
                print(f"   Output: {result.execution_result.output}")

    print(f"\n📊 Statistics:")
    print(f"   Questions asked: {result.total_questions}")
    print(f"   Final confidence: {result.final_probability:.2%}")
    print(f"   Termination: {result.termination_reason.name}")

    if result.steps:
        print(f"\n📈 Decision Steps:")
        for i, step in enumerate(result.steps):
            print(f"   Step {i}: max_prob={step.max_probability:.2%}")
            if step.selected_question:
                print(f"      Asked: {step.selected_question.text}")
            if step.response:
                print(f"      Response: {step.response}")


if __name__ == "__main__":
    main()
