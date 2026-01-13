"""LangGraph wrapper around SAGE-Agent decision loop - Enhanced Version.

This implements the full SAGE-Agent POMDP loop with proper structure:

┌─────────────────┐
│   START         │
└────────┬────────┘
         ▼
┌─────────────────┐
│ generate_cands  │◄────────────────────────────┐
└────────┬────────┘                             │
         ▼                                      │
┌─────────────────┐     ┌───────────────────┐   │
│ check_confidence│────►│ execute           │   │
└────────┬────────┘     └─────────┬─────────┘   │
         │ (need more info)       │             │
         ▼                        ▼             │
┌─────────────────┐     ┌───────────────────┐   │
│ generate_qs     │     │ validate_result   │   │
└────────┬────────┘     └─────────┬─────────┘   │
         ▼                        │             │
┌─────────────────┐               │             │
│ select_question │               ▼             │
└────────┬────────┘         ┌───────────┐       │
         │                  │  SUCCESS  │       │
         ▼                  └───────────┘       │
┌─────────────────┐               │             │
│ ask_question    │               ▼             │
└────────┬────────┘         ┌───────────┐       │
         │                  │  handle   │       │
         ▼                  │  error    │───────┘
┌─────────────────┐         └───────────┘
│ update_belief   │               │
└────────┬────────┘               ▼
         │                  ┌───────────┐
         │                  │ ESCALATE  │
         └──────────────────►───────────┘

Key improvements over v1:
1. Separate check_confidence node for early exit
2. Explicit update_belief node for domain refinement
3. validate_result node with error recovery loop
4. Better separation of concerns
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TypedDict, Literal, Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from langgraph.graph import StateGraph, END

from sage_agent import (
    LLMBackedCandidateGenerator,
    LLMBackedQuestionGenerator,
    ParameterDomain,
    Question,
    SageAgentConfig,
    ToolCall,
    ToolCallCandidate,
    ToolRegistryExecutor,
    ToolSchema,
    UNK,
    create_sage_propagator,
)
from sage_agent.core.belief import BeliefState
from sage_agent.core.constraints import SimpleConstraintExtractor, HybridConstraintExtractor
from sage_agent.core.evpi import compute_evpi
from sage_agent.core.types import ToolExecutor, ConstraintExtractor, ExecutionResult
from sage_agent.core.uncertainty_propagation import UncertaintyPropagator


class AgentState(TypedDict):
    """Complete state for SAGE-Agent graph.
    
    This state maintains all information needed for the POMDP belief update loop.
    """
    # Input
    user_input: str
    
    # Belief state components
    observations: List[str]
    domains: Dict[str, Dict[str, ParameterDomain]]
    
    # Candidate tracking
    candidates: List[ToolCallCandidate]
    probabilities: List[float]
    best_candidate_index: int
    
    # Question selection
    questions: List[Question]
    best_question: Optional[Question]
    best_score: float
    aspect_counts: Dict[str, int]
    
    # Uncertainty tracking
    uncertainty: float  # Structured (from belief state)
    llm_uncertainty: float  # From LLM sampling
    combined_uncertainty: float  # Weighted combination
    
    # Execution tracking
    steps: int
    attempts: int
    execution_attempts: int  # Track execution retries
    last_execution_error: Optional[str]
    
    # Output
    status: Literal["pending", "confident", "asking", "executing", "done", "escalated"]
    result: Optional[ToolCall]
    execution_result: Optional[ExecutionResult]
    error: Optional[str]


CONFIG = {
    # Uncertainty thresholds
    "confidence_threshold": 0.7,  # min prob to skip questions
    "uncertainty_threshold": 0.3,  # max uncertainty to execute
    "max_attempts": 3,  # max clarification rounds
    "max_execution_retries": 2,  # max execution error recovery attempts
    
    # Uncertainty combination
    "structured_weight": 0.7,
    "llm_modulation": 0.5,
    
    # Critical operations
    "critical_patterns": ["delete", "cancel", "remove", "drop", "terminate", "destroy"],
    "critical_threshold_multiplier": 0.5,
    
    # Escalation
    "escalation_uncertainty": 0.85,
    "max_high_uncertainty_steps": 3,
}

@dataclass
class GraphDeps:
    """All dependencies injected into the graph."""
    tool_schemas: Dict[str, ToolSchema]
    candidate_generator: LLMBackedCandidateGenerator
    question_generator: LLMBackedQuestionGenerator
    question_asker: Any  # QuestionAsker protocol
    tool_executor: ToolExecutor
    constraint_extractor: ConstraintExtractor
    config: SageAgentConfig
    uncertainty_propagator: Optional[UncertaintyPropagator] = None


def _compute_adaptive_threshold(tool_name: str, base_threshold: float) -> float:
    """Lower threshold for critical operations."""
    tool_lower = tool_name.lower()
    is_critical = any(p in tool_lower for p in CONFIG["critical_patterns"])
    return base_threshold * CONFIG["critical_threshold_multiplier"] if is_critical else base_threshold


def _has_required_unknowns(candidate: ToolCallCandidate, schema: ToolSchema) -> bool:
    """Check if candidate has unknown required parameters."""
    return any(candidate.arguments.get(p, UNK) == UNK for p in schema.required)


def _compute_evpi_score(
    question: Question,
    candidates: List[ToolCallCandidate],
    probs: List[float],
    aspect_counts: Dict[str, int],
    redundancy_weight: float,
) -> float:
    """Compute EVPI minus redundancy cost for question selection."""
    evpi = compute_evpi(candidates, probs, question.aspects)
    cost = sum(aspect_counts.get(f"{a.tool_name}:{a.param_name}", 0) for a in question.aspects)
    return evpi - cost * redundancy_weight


def build_graph(deps: GraphDeps) -> StateGraph:
    """Build the enhanced SAGE-Agent graph."""
    
    graph = StateGraph(AgentState)

    def generate_candidates_node(state: AgentState) -> AgentState:
        """Generate tool call candidates and compute belief-based probabilities."""
        candidates = deps.candidate_generator.generate_candidates(
            state["user_input"], 
            state["observations"], 
            deps.tool_schemas
        )
        
        if not candidates:
            return {
                **state,
                "error": "No candidates generated",
                "status": "done",
            }
        
        # Compute belief-based weights
        belief = BeliefState(domains=state["domains"], epsilon=deps.config.epsilon)
        weights = [
            belief.candidate_weight(c, deps.tool_schemas[c.tool_name])
            for c in candidates
        ]
        
        # Get LLM uncertainty and modulate weights
        llm = getattr(deps.candidate_generator, "llm", None)
        llm_unc_raw = getattr(llm, "last_uncertainty", None)
        llm_unc = llm_unc_raw if llm_unc_raw is not None else 0.5
        
        if llm_unc_raw is not None:
            factor = 1.0 - (llm_unc * CONFIG["llm_modulation"])
            weights = [w * factor for w in weights]
        
        probs = belief.normalize(weights)
        best_idx = probs.index(max(probs)) if probs else 0
        
        # Compute uncertainties
        struct_unc = 1.0 - (max(probs) if probs else 0.0)
        combined = CONFIG["structured_weight"] * struct_unc + (1 - CONFIG["structured_weight"]) * llm_unc
        
        # Track in propagator
        if deps.uncertainty_propagator:
            deps.uncertainty_propagator.observe(struct_unc, "candidate_generation")
            if llm_unc_raw is not None:
                deps.uncertainty_propagator.observe(llm_unc, "llm_parsing")
        
        return {
            **state,
            "candidates": candidates,
            "probabilities": probs,
            "best_candidate_index": best_idx,
            "uncertainty": struct_unc,
            "llm_uncertainty": llm_unc,
            "combined_uncertainty": combined,
            "status": "pending",
        }
    
    def check_confidence_router(state: AgentState) -> Literal["confident", "need_questions", "escalate"]:
        """Decide if we're confident enough to execute, need questions, or should escalate."""
        if state["error"]:
            return "confident"  # Will handle error in execute
        
        candidate = state["candidates"][state["best_candidate_index"]]
        schema = deps.tool_schemas[candidate.tool_name]
        
        # Must ask if required params unknown
        if _has_required_unknowns(candidate, schema):
            return "need_questions"
        
        # Check escalation conditions
        if deps.uncertainty_propagator and deps.uncertainty_propagator.should_escalate(
            escalation_threshold=CONFIG["escalation_uncertainty"],
            max_high_uncertainty_steps=CONFIG["max_high_uncertainty_steps"],
        ):
            return "escalate"
        
        if state["attempts"] >= CONFIG["max_attempts"]:
            return "escalate"
        
        # Check confidence
        max_prob = max(state["probabilities"]) if state["probabilities"] else 0.0
        threshold = _compute_adaptive_threshold(
            candidate.tool_name, 
            CONFIG["confidence_threshold"]
        )
        
        if max_prob >= threshold:
            return "confident"
        
        if state["combined_uncertainty"] <= CONFIG["uncertainty_threshold"]:
            return "confident"
        
        return "need_questions"

    def generate_questions_node(state: AgentState) -> AgentState:
        """Generate clarifying questions for ambiguous candidates."""
        questions = deps.question_generator.generate_questions(
            state["user_input"],
            state["candidates"],
            state["observations"],
            deps.tool_schemas,
        )
        
        if not questions:
            # No questions generated, proceed to execution
            return {**state, "questions": [], "best_question": None, "best_score": 0.0}
        
        # Score questions by EVPI - redundancy cost
        scored = [
            (q, _compute_evpi_score(
                q, state["candidates"], state["probabilities"],
                state["aspect_counts"], deps.config.redundancy_weight
            ))
            for q in questions
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        best_q, best_score = scored[0] if scored else (None, 0.0)
        
        return {
            **state,
            "questions": questions,
            "best_question": best_q,
            "best_score": best_score,
        }
    
    def select_question_router(state: AgentState) -> Literal["ask", "execute"]:
        """Decide whether to ask the best question or proceed to execution."""
        if not state["questions"] or state["best_question"] is None:
            return "execute"
        
        # Check if question is worth asking (EVPI > alpha * max_prob)
        max_prob = max(state["probabilities"]) if state["probabilities"] else 0.0
        if state["best_score"] < deps.config.alpha * max_prob:
            return "execute"
        
        # Check step limit
        if state["steps"] >= deps.config.max_questions:
            return "execute"
        
        return "ask"
    
    def ask_question_node(state: AgentState) -> AgentState:
        """Ask the selected question and get user response."""
        question = state["best_question"]
        if question is None:
            return state
        
        response = deps.question_asker.ask(question)
        new_observations = list(state["observations"]) + [response]
        
        # Update aspect counts
        new_counts = dict(state["aspect_counts"])
        for aspect in question.aspects:
            key = f"{aspect.tool_name}:{aspect.param_name}"
            new_counts[key] = new_counts.get(key, 0) + 1
        
        return {
            **state,
            "observations": new_observations,
            "aspect_counts": new_counts,
            "steps": state["steps"] + 1,
            "attempts": state["attempts"] + 1,
            "status": "asking",
        }
    
    def update_belief_node(state: AgentState) -> AgentState:
        """Update domain constraints based on user response."""
        question = state["best_question"]
        if question is None or not state["observations"]:
            return state
        
        # Latest response
        response = state["observations"][-1]
        
        # Update domains for each aspect
        new_domains = {tool: dict(params) for tool, params in state["domains"].items()}
        
        for aspect in question.aspects:
            schema = deps.tool_schemas[aspect.tool_name]
            current = new_domains[schema.name][aspect.param_name]
            refined = deps.constraint_extractor.update_domain(current, response)
            new_domains[schema.name][aspect.param_name] = refined
            
            # Apply domain refiner if available
            if schema.domain_refiner is not None:
                new_domains[schema.name] = dict(
                    schema.domain_refiner.refine(schema, new_domains[schema.name], {})
                )
        
        # Track belief update in propagator
        if deps.uncertainty_propagator:
            deps.uncertainty_propagator.observe(0.1, "belief_update")  # Low uncertainty for update step
        
        return {**state, "domains": new_domains}
    
    def execute_node(state: AgentState) -> AgentState:
        """Execute the best candidate tool call."""
        if state["error"]:
            return {**state, "status": "done"}
        
        candidate = state["candidates"][state["best_candidate_index"]]
        schema = deps.tool_schemas[candidate.tool_name]
        
        # Validate before execution
        try:
            schema.validate_call(candidate.arguments)
        except ValueError as e:
            return {
                **state,
                "error": str(e),
                "status": "done",
            }
        
        # Execute
        tool_call = ToolCall(tool_name=candidate.tool_name, arguments=candidate.arguments)
        result = deps.tool_executor.execute(tool_call)
        
        return {
            **state,
            "result": tool_call,
            "execution_result": result,
            "status": "executing",
        }
    
    def validate_result_router(state: AgentState) -> Literal["success", "retry", "escalate"]:
        """Check execution result and decide next step."""
        exec_result = state.get("execution_result")
        
        if exec_result is None:
            return "escalate"
        
        if exec_result.success:
            return "success"
        
        # Execution failed - check if we should retry
        retry_count = state.get("execution_attempts", 0)
        if retry_count < CONFIG["max_execution_retries"]:
            return "retry"
        
        return "escalate"
    
    def handle_success_node(state: AgentState) -> AgentState:
        """Mark successful completion."""
        return {**state, "status": "done", "error": None}
    
    def handle_error_node(state: AgentState) -> AgentState:
        """Handle execution error - prepare for retry."""
        exec_result = state.get("execution_result")
        error_msg = exec_result.error if exec_result else "Unknown error"
        
        return {
            **state,
            "last_execution_error": error_msg,
            "execution_attempts": state.get("execution_attempts", 0) + 1,
            "observations": state["observations"] + [f"Execution failed: {error_msg}"],
        }
    
    def escalate_node(state: AgentState) -> AgentState:
        """Escalate to human when agent cannot resolve."""
        return {
            **state,
            "status": "escalated",
            "error": state.get("error") or state.get("last_execution_error") or "Escalated due to high uncertainty",
        }
    
    # Add all nodes
    graph.add_node("generate_candidates", generate_candidates_node)
    graph.add_node("generate_questions", generate_questions_node)
    graph.add_node("ask_question", ask_question_node)
    graph.add_node("update_belief", update_belief_node)
    graph.add_node("execute", execute_node)
    graph.add_node("handle_success", handle_success_node)
    graph.add_node("handle_error", handle_error_node)
    graph.add_node("escalate", escalate_node)
    
    # Entry point
    graph.set_entry_point("generate_candidates")
    
    # Edges from generate_candidates
    graph.add_conditional_edges(
        "generate_candidates",
        check_confidence_router,
        {
            "confident": "execute",
            "need_questions": "generate_questions",
            "escalate": "escalate",
        }
    )
    
    # Edges from generate_questions
    graph.add_conditional_edges(
        "generate_questions",
        select_question_router,
        {
            "ask": "ask_question",
            "execute": "execute",
        }
    )
    
    # Ask -> Update Belief -> Generate Candidates (loop back)
    graph.add_edge("ask_question", "update_belief")
    graph.add_edge("update_belief", "generate_candidates")
    
    # Execute -> Validate
    graph.add_conditional_edges(
        "execute",
        validate_result_router,
        {
            "success": "handle_success",
            "retry": "handle_error",
            "escalate": "escalate",
        }
    )
    
    # Error -> back to candidates (for retry)
    graph.add_edge("handle_error", "generate_candidates")
    
    # Terminal nodes
    graph.add_edge("handle_success", END)
    graph.add_edge("escalate", END)
    
    return graph


def create_initial_state(
    user_input: str,
    tool_schemas: Dict[str, ToolSchema],
) -> AgentState:
    """Create initial state for the agent."""
    return {
        "user_input": user_input,
        "observations": [],
        "domains": {name: dict(schema.parameters) for name, schema in tool_schemas.items()},
        "candidates": [],
        "probabilities": [],
        "best_candidate_index": 0,
        "questions": [],
        "best_question": None,
        "best_score": 0.0,
        "aspect_counts": {},
        "uncertainty": 1.0,
        "llm_uncertainty": 0.5,
        "combined_uncertainty": 1.0,
        "steps": 0,
        "attempts": 0,
        "execution_attempts": 0,
        "last_execution_error": None,
        "status": "pending",
        "result": None,
        "execution_result": None,
        "error": None,
    }


class InteractiveQuestionAsker:
    """Ask questions interactively via console."""
    def ask(self, question: Question) -> str:
        print(f"\n Agent: {question.text}")
        return input("You: ").strip()


def main():
    """Demo the enhanced SAGE-Agent."""
    from examples.ollama_client import OllamaClient
    
    # Define tool
    tool = ToolSchema(
        name="book_flight",
        parameters={
            "origin": ParameterDomain.from_values(["NYC", "BOS", "LAX", "SFO"]),
            "dest": ParameterDomain.from_values(["NYC", "BOS", "LAX", "SFO"]),
            "date": ParameterDomain.from_values(["2024-01-15", "2024-01-16", "2024-01-17"]),
        },
        required=frozenset({"origin", "dest", "date"}),
    )
    
    # Create LLM
    llm = OllamaClient(model="qwen3:4b-instruct-2507-q8_0", verbose=False)
    
    # Create propagator
    propagator = create_sage_propagator()
    
    # Create dependencies
    deps = GraphDeps(
        tool_schemas={tool.name: tool},
        candidate_generator=LLMBackedCandidateGenerator(llm),
        question_generator=LLMBackedQuestionGenerator(llm),
        question_asker=InteractiveQuestionAsker(),
        tool_executor=ToolRegistryExecutor({
            "book_flight": lambda args: ExecutionResult(success=True, output={"booked": args})
        }),
        constraint_extractor=SimpleConstraintExtractor(),
        config=SageAgentConfig(max_questions=4, tau_execute=0.85, alpha=0.1),
        uncertainty_propagator=propagator,
    )
    
    # Build and compile
    graph = build_graph(deps).compile()
    
    # Run
    initial = create_initial_state("I need to fly to LAX", deps.tool_schemas)
    result = graph.invoke(initial, {"recursion_limit": 50})
    
    print("\n" + "=" * 50)
    print(f"Status: {result['status']}")
    print(f"Result: {result['result']}")
    print(f"Steps: {result['steps']}")
    print(f"Attempts: {result['attempts']}")
    print(f"Final uncertainty: {result['combined_uncertainty']:.3f}")
    if result['error']:
        print(f"Error: {result['error']}")
    print("=" * 50)


if __name__ == "__main__":
    main()

