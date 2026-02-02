"""Pure SAGE-Agent LangGraph Implementation (Algorithm 1 from Paper).

This is a clean implementation of the SAGE-Agent decision loop exactly as
described in the paper "Structured Uncertainty Guided Clarification for LLM Agents".

NO extra features - just the core algorithm:
- BeliefState with domain-based candidate weighting
- EVPI-based question selection with redundancy cost
- Termination conditions from the paper (τ, α thresholds)

For experimental features (SAUP, reflexion, LLM uncertainty), see:
    langgraph_sage_agent_experimental.py

Algorithm 1: SAGE-Agent Decision Loop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1: Initialize belief B(0), aspect counts n(0), t ← 0
2: while t < T_max do
3:     Generate candidates C, compute π_c(t)
4:     if max_c π_c(t) ≥ τ_exec then execute
5:     Generate questions Q
6:     Score: q* = argmax[EVPI(q) - Cost(q)]
7:     if Score(q*) < α · max_c π_c(t) then execute
8:     Ask q*, update domains, t++
9: end while
10: Final execution or escalation

Paper hyperparameters (defaults):
- τ_exec = 0.85 (execution confidence threshold)
- α = 0.1 (termination threshold factor)
- λ = 0.5 (redundancy weight)
- ε = 1e-4 (epsilon for infinite domains)
- T_max = 6 (maximum clarification questions)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Protocol, TypedDict, Literal

# Project root is 2 levels up from different_agents/pure_sage/
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "different_agents", "shared"))

from langgraph.graph import StateGraph, END

from sage_agent import (
    ParameterDomain,
    ToolCall,
    ToolCallCandidate,
    ToolSchema,
    Question,
    UNK,
)
from sage_agent.core.belief import BeliefState
from sage_agent.core.evpi import compute_evpi
from sage_agent.core.types import ExecutionResult


# =============================================================================
# Protocols for dependency injection
# =============================================================================

class LLMClient(Protocol):
    """Protocol for LLM completions."""
    def complete(self, prompt: str) -> str:
        ...


class CandidateGenerator(Protocol):
    """Protocol for generating tool call candidates."""
    def generate_candidates(
        self,
        user_input: str,
        observations: List[str],
        tool_schemas: Mapping[str, ToolSchema],
    ) -> List[ToolCallCandidate]:
        ...


class QuestionGenerator(Protocol):
    """Protocol for generating clarification questions."""
    def generate_questions(
        self,
        user_input: str,
        candidates: List[ToolCallCandidate],
        observations: List[str],
        tool_schemas: Mapping[str, ToolSchema],
    ) -> List[Question]:
        ...


class QuestionAsker(Protocol):
    """Protocol for asking questions to users."""
    def ask(self, question: Question) -> str:
        ...


class ToolExecutor(Protocol):
    """Protocol for executing tool calls."""
    def execute(self, tool_call: ToolCall) -> ExecutionResult:
        ...


class ConstraintExtractor(Protocol):
    """Protocol for extracting constraints from responses."""
    def update_domain(self, domain: ParameterDomain, response: str) -> ParameterDomain:
        ...


# =============================================================================
# Configuration - Paper Hyperparameters Only
# =============================================================================

@dataclass(frozen=True)
class SAGEConfig:
    """SAGE-Agent configuration with paper hyperparameters.

    All defaults match the paper exactly.
    """
    tau_exec: float = 0.85          # τ - execution confidence threshold
    alpha: float = 0.1              # α - termination threshold factor
    lambda_redundancy: float = 0.5  # λ - redundancy weight
    epsilon: float = 1e-4           # ε - small prob for infinite domains
    max_questions: int = 6          # T_max - maximum clarification questions
    min_confidence: float = 0.3     # Minimum confidence to execute at termination


# =============================================================================
# Agent State
# =============================================================================

class AgentState(TypedDict):
    """State for SAGE-Agent LangGraph."""
    # Input
    user_input: str

    # Belief state
    observations: List[str]
    domains: Dict[str, Dict[str, ParameterDomain]]

    # Candidates
    candidates: List[ToolCallCandidate]
    probabilities: List[float]
    max_prob: float
    best_idx: int

    # Questions
    questions: List[Question]
    best_question: Optional[Question]
    best_score: float
    aspect_counts: Dict[str, int]

    # Loop control
    step: int  # t in the algorithm

    # Output
    status: Literal["pending", "executing", "done", "escalated"]
    result: Optional[ToolCall]
    execution_result: Optional[ExecutionResult]
    error: Optional[str]


# =============================================================================
# Dependencies
# =============================================================================

@dataclass
class GraphDeps:
    """All dependencies for SAGE graph nodes."""
    tool_schemas: Dict[str, ToolSchema]
    candidate_generator: CandidateGenerator
    question_generator: QuestionGenerator
    question_asker: QuestionAsker
    tool_executor: ToolExecutor
    constraint_extractor: ConstraintExtractor
    config: SAGEConfig


# =============================================================================
# Helper Functions
# =============================================================================

def _has_required_unknowns(candidate: ToolCallCandidate, schema: ToolSchema) -> bool:
    """Check if candidate has unknown required parameters."""
    return any(candidate.arguments.get(p, UNK) == UNK for p in schema.required)


def _redundancy_cost(
    question: Question,
    aspect_counts: Dict[str, int],
    lambda_weight: float,
) -> float:
    """Compute redundancy cost: Cost(q, t) = λ · Σ_{a∈A(q)} n_a(t)."""
    total = sum(
        aspect_counts.get(f"{a.tool_name}:{a.param_name}", 0)
        for a in question.aspects
    )
    return lambda_weight * total


# =============================================================================
# Graph Nodes
# =============================================================================

def build_graph(deps: GraphDeps) -> StateGraph:
    """Build the pure SAGE-Agent LangGraph."""

    graph = StateGraph(AgentState)

    # -------------------------------------------------------------------------
    # Node: Generate Candidates (Lines 3-4 of Algorithm 1)
    # -------------------------------------------------------------------------
    def generate_candidates_node(state: AgentState) -> dict:
        """Generate candidates and compute probabilities π_c(t)."""

        # Generate candidates from LLM
        candidates = deps.candidate_generator.generate_candidates(
            state["user_input"],
            state["observations"],
            deps.tool_schemas,
        )

        # Filter to known tools
        candidates = [c for c in candidates if c.tool_name in deps.tool_schemas]

        if not candidates:
            return {
                "candidates": [],
                "probabilities": [],
                "max_prob": 0.0,
                "best_idx": 0,
                "error": "No valid candidates generated",
                "status": "done",
            }

        # Compute weights using BeliefState (Section 3.2)
        belief = BeliefState(
            domains=state["domains"],
            epsilon=deps.config.epsilon,
        )

        weights = [
            belief.candidate_weight(c, deps.tool_schemas[c.tool_name])
            for c in candidates
        ]

        # Normalize to get probabilities π_c(t)
        probabilities = belief.normalize(weights)
        max_prob = max(probabilities) if probabilities else 0.0
        best_idx = probabilities.index(max_prob) if probabilities else 0

        return {
            "candidates": candidates,
            "probabilities": probabilities,
            "max_prob": max_prob,
            "best_idx": best_idx,
        }

    # -------------------------------------------------------------------------
    # Node: Generate Questions (Line 5 of Algorithm 1)
    # -------------------------------------------------------------------------
    def generate_questions_node(state: AgentState) -> dict:
        """Generate and score clarification questions."""

        if not state["candidates"]:
            return {"questions": [], "best_question": None, "best_score": 0.0}

        # Generate candidate questions
        questions = deps.question_generator.generate_questions(
            state["user_input"],
            state["candidates"],
            state["observations"],
            deps.tool_schemas,
        )

        if not questions:
            return {"questions": [], "best_question": None, "best_score": 0.0}

        # Score each question: Score(q) = EVPI(q) - Cost(q)
        scored = []
        for question in questions:
            evpi = compute_evpi(
                state["candidates"],
                state["probabilities"],
                question.aspects,
            )
            cost = _redundancy_cost(
                question,
                state["aspect_counts"],
                deps.config.lambda_redundancy,
            )
            score = evpi - cost
            scored.append((score, question))

        # Select best question q* = argmax Score(q)
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_question = scored[0]

        return {
            "questions": questions,
            "best_question": best_question,
            "best_score": best_score,
        }

    # -------------------------------------------------------------------------
    # Node: Ask Question (Lines 8-10 of Algorithm 1)
    # -------------------------------------------------------------------------
    def ask_question_node(state: AgentState) -> dict:
        """Ask the selected question and update belief."""

        question = state["best_question"]
        if question is None:
            return {}

        # Ask question, get response
        response = deps.question_asker.ask(question)

        # Update observations
        new_observations = list(state["observations"]) + [response]

        # Update domains based on response
        new_domains = {
            tool: dict(params)
            for tool, params in state["domains"].items()
        }

        for aspect in question.aspects:
            tool = deps.tool_schemas.get(aspect.tool_name)
            if tool is None:
                continue
            if aspect.param_name not in new_domains.get(tool.name, {}):
                continue

            current = new_domains[tool.name][aspect.param_name]
            refined = deps.constraint_extractor.update_domain(current, response)
            new_domains[tool.name][aspect.param_name] = refined

        # Update aspect counts: n_a(t+1) ← n_a(t) + 1
        new_counts = dict(state["aspect_counts"])
        for aspect in question.aspects:
            key = f"{aspect.tool_name}:{aspect.param_name}"
            new_counts[key] = new_counts.get(key, 0) + 1

        return {
            "observations": new_observations,
            "domains": new_domains,
            "aspect_counts": new_counts,
            "step": state["step"] + 1,
        }

    # -------------------------------------------------------------------------
    # Node: Execute (Line 6 of Algorithm 1)
    # -------------------------------------------------------------------------
    def execute_node(state: AgentState) -> dict:
        """Execute the best candidate tool call."""

        if state.get("error") or not state["candidates"]:
            return {"status": "done"}

        candidate = state["candidates"][state["best_idx"]]
        schema = deps.tool_schemas[candidate.tool_name]

        # Create tool call
        tool_call = ToolCall(
            tool_name=candidate.tool_name,
            arguments=dict(candidate.arguments),
        )

        # Validate required parameters
        try:
            schema.validate_call(tool_call.arguments)
        except ValueError as exc:
            return {
                "result": tool_call,
                "execution_result": ExecutionResult(success=False, error=str(exc)),
                "error": str(exc),
                "status": "done",
            }

        # Execute
        result = deps.tool_executor.execute(tool_call)

        return {
            "result": tool_call,
            "execution_result": result,
            "status": "done",
        }

    # -------------------------------------------------------------------------
    # Node: Escalate
    # -------------------------------------------------------------------------
    def escalate_node(state: AgentState) -> dict:
        """Escalate to human operator."""
        return {
            "status": "escalated",
            "error": "Escalated due to high uncertainty or max questions reached",
        }

    # -------------------------------------------------------------------------
    # Routing: Core SAGE decision logic
    # -------------------------------------------------------------------------
    def route_after_questions(state: AgentState) -> Literal["execute", "ask", "escalate"]:
        """Route based on SAGE termination conditions."""

        # Error or no candidates -> execute (will fail gracefully)
        if state.get("error") or not state["candidates"]:
            return "execute"

        # Line 4: if max_c π_c(t) ≥ τ_exec then execute
        if state["max_prob"] >= deps.config.tau_exec:
            return "execute"

        # Check if we have questions to ask
        if not state["questions"] or state["best_question"] is None:
            return "execute"

        # Line 2: Check max questions (t < T_max)
        if state["step"] >= deps.config.max_questions:
            # Final check: execute if confident enough, else escalate
            if state["max_prob"] >= deps.config.min_confidence:
                return "execute"
            return "escalate"

        # Line 7: if Score(q*) < α · max_c π_c(t) then execute
        if state["best_score"] < deps.config.alpha * state["max_prob"]:
            # Cost exceeds benefit, but check required params first
            candidate = state["candidates"][state["best_idx"]]
            schema = deps.tool_schemas[candidate.tool_name]
            if not _has_required_unknowns(candidate, schema):
                return "execute"
            # Has required unknowns, must keep asking

        return "ask"

    # -------------------------------------------------------------------------
    # Build Graph
    # -------------------------------------------------------------------------
    graph.add_node("generate_candidates", generate_candidates_node)
    graph.add_node("generate_questions", generate_questions_node)
    graph.add_node("ask_question", ask_question_node)
    graph.add_node("execute", execute_node)
    graph.add_node("escalate", escalate_node)

    # Entry point
    graph.set_entry_point("generate_candidates")

    # Edges
    graph.add_edge("generate_candidates", "generate_questions")

    graph.add_conditional_edges(
        "generate_questions",
        route_after_questions,
        {
            "execute": "execute",
            "ask": "ask_question",
            "escalate": "escalate",
        },
    )

    # After asking, loop back to generate new candidates
    graph.add_edge("ask_question", "generate_candidates")

    # Terminal nodes
    graph.add_edge("execute", END)
    graph.add_edge("escalate", END)

    return graph


# =============================================================================
# Helper: Create Initial State
# =============================================================================

def create_initial_state(
    user_input: str,
    tool_schemas: Mapping[str, ToolSchema],
) -> AgentState:
    """Create initial state for SAGE graph execution."""
    return AgentState(
        user_input=user_input,
        observations=[],
        domains={
            name: dict(schema.parameters)
            for name, schema in tool_schemas.items()
        },
        candidates=[],
        probabilities=[],
        max_prob=0.0,
        best_idx=0,
        questions=[],
        best_question=None,
        best_score=0.0,
        aspect_counts={},
        step=0,
        status="pending",
        result=None,
        execution_result=None,
        error=None,
    )


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    from sage_agent import (
        LLMBackedCandidateGenerator,
        LLMBackedQuestionGenerator,
        SimpleConstraintExtractor,
        ToolRegistryExecutor,
    )

    # Try to import LLM client
    try:
        from examples.tts_llm_client import TTSLLMClient
        llm = TTSLLMClient(
            base_url="http://localhost:8001/v1",
            model="openai/gpt-4o-mini",
            tts_budget=8,
        )
    except Exception:
        try:
            from examples.openrouter_client import OpenRouterClient
            llm = OpenRouterClient(
                model=os.getenv("SAGE_MODEL", "openai/gpt-4o-mini"),
            )
        except Exception as e:
            print(f"No LLM client available: {e}")
            sys.exit(1)

    # Define a simple tool
    tool = ToolSchema(
        name="book_flight",
        parameters={
            "origin": ParameterDomain.from_values(["NYC", "BOS", "LAX"]),
            "destination": ParameterDomain.from_values(["NYC", "BOS", "LAX"]),
            "date": ParameterDomain.from_values(["2024-03-01", "2024-03-02"]),
        },
        required=frozenset({"origin", "destination", "date"}),
    )

    # Interactive question asker
    class ConsoleAsker:
        def ask(self, question: Question) -> str:
            print(f"\n🤔 Agent: {question.text}")
            return input("You: ").strip()

    # Build dependencies
    deps = GraphDeps(
        tool_schemas={tool.name: tool},
        candidate_generator=LLMBackedCandidateGenerator(llm),
        question_generator=LLMBackedQuestionGenerator(llm),
        question_asker=ConsoleAsker(),
        tool_executor=ToolRegistryExecutor({
            "book_flight": lambda args: ExecutionResult(
                success=True,
                output={"confirmation": "FL123", "details": args},
            )
        }),
        constraint_extractor=SimpleConstraintExtractor(),
        config=SAGEConfig(),  # Paper defaults
    )

    # Build and run graph
    graph = build_graph(deps).compile()

    initial_state = create_initial_state(
        user_input="Book me a flight to LAX",
        tool_schemas=deps.tool_schemas,
    )

    print("=" * 60)
    print("Pure SAGE-Agent (Algorithm 1 from Paper)")
    print("=" * 60)
    print(f"\nUser: {initial_state['user_input']}\n")

    result = graph.invoke(initial_state, {"recursion_limit": 50})

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Status: {result['status']}")
    print(f"Questions asked: {result['step']}")
    print(f"Final max_prob: {result['max_prob']:.2%}")

    if result["result"]:
        print(f"Tool call: {result['result'].tool_name}")
        print(f"Arguments: {dict(result['result'].arguments)}")

    if result.get("error"):
        print(f"Error: {result['error']}")
