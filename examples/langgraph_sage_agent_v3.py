"""LangGraph SAGE-Agent v3: Enhanced with Advanced Uncertainty & Constrained Decoding.

This implements the full SAGE-Agent with:

Phase 1: Constrained decoding + Schema Guided Reasoning (SGR)
- JSON schema validation for guaranteed-valid tool calls
- Per-field uncertainty tracking
- Structured reasoning traces

Phase 2: Uncertainty-driven budget allocation
- Self-consistency / Best-of-N resampling
- Dynamic sampling based on uncertainty
- Uncertainty-aware early stopping

Phase 3: SAUP trajectory-level uncertainty
- Comprehensive step-level uncertainty propagation
- Failure prediction and localization
- Targeted fixes based on uncertainty breakdown

Phase 4: Smart Reflexion
- Reflexion only on failures or high uncertainty
- Test-driven reflexion for code tasks
- Memory-aware improvement

Graph structure (same as v2 but with enhanced nodes):

┌─────────────────┐
│   START         │
└────────┬────────┘
         ▼
┌─────────────────┐
│ generate_cands  │◄────────────────────────────┐
│ (+ resampling)  │                             │
└────────┬────────┘                             │
         ▼                                      │
┌─────────────────┐     ┌───────────────────┐   │
│ check_confidence│────►│ execute           │   │
│ (+ SAUP)        │     │ (+ validation)    │   │
└────────┬────────┘     └─────────┬─────────┘   │
         │                        ▼             │
         ▼                  ┌───────────┐       │
┌─────────────────┐         │ validate  │       │
│ generate_qs     │         │ (+ CoT)   │       │
└────────┬────────┘         └─────┬─────┘       │
         ▼                        ▼             │
┌─────────────────┐         ┌───────────┐       │
│ select_question │         │  SUCCESS  │       │
└────────┬────────┘         └───────────┘       │
         ▼                        │             │
┌─────────────────┐               ▼             │
│ ask_question    │         ┌───────────┐       │
└────────┬────────┘         │  handle   │       │
         ▼                  │  error    │───────┘
┌─────────────────┐         │(+Reflexion)
│ update_belief   │         └───────────┘
└────────┬────────┘               │
         │                        ▼
         │                  ┌───────────┐
         │                  │ ESCALATE  │
         └──────────────────►───────────┘
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TypedDict, Literal, Any, Tuple

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
from sage_agent.core.advanced_reasoning import (
    UncertaintyDecomposer,
    ChainOfThoughtVerifier,
    DecomposedUncertainty,
)


# =============================================================================
# Phase 1: Schema Guided Reasoning & Per-Field Uncertainty
# =============================================================================

@dataclass
class FieldUncertainty:
    """Per-field uncertainty for a parameter."""
    param_name: str
    value: object
    uncertainty: float  # 0=certain, 1=uncertain
    source: Literal["inferred", "asked", "default", "unknown"]
    reasoning: str = ""


@dataclass
class ReasoningTrace:
    """Structured reasoning trace for SGR."""
    step: str
    thought: str
    action: str
    uncertainty: float
    fields_affected: List[str]


class StructuredToolCall(TypedDict):
    """Constrained tool call with validation."""
    tool: str
    arguments: Dict[str, object]
    reasoning: str
    field_uncertainties: Dict[str, FieldUncertainty]
    overall_uncertainty: float


# =============================================================================
# Agent State (Enhanced)
# =============================================================================

class AgentState(TypedDict):
    """Enhanced state with per-field uncertainty and reasoning traces."""
    # Input
    user_input: str

    # Belief state components
    observations: List[str]
    domains: Dict[str, Dict[str, ParameterDomain]]

    # Candidate tracking (with per-field uncertainty)
    candidates: List[ToolCallCandidate]
    probabilities: List[float]
    best_candidate_index: int
    field_uncertainties: Dict[str, Dict[str, FieldUncertainty]]  # NEW: per-candidate field unc

    # Question selection
    questions: List[Question]
    best_question: Optional[Question]
    best_score: float
    aspect_counts: Dict[str, int]

    # Uncertainty tracking (enhanced)
    uncertainty: float  # Structured (from belief state)
    llm_uncertainty: float  # From LLM sampling
    combined_uncertainty: float  # Weighted combination
    epistemic_uncertainty: float  # NEW: Reducible uncertainty
    aleatoric_uncertainty: float  # NEW: Irreducible uncertainty

    # Phase 2: Resampling state
    num_samples: int  # NEW: How many samples to generate
    samples: List[ToolCallCandidate]  # NEW: Multiple sampled candidates
    sample_agreement: float  # NEW: Agreement between samples

    # Phase 3: SAUP trajectory tracking
    reasoning_traces: List[ReasoningTrace]  # NEW: Step-by-step reasoning
    trajectory_uncertainty: float  # NEW: Accumulated uncertainty across steps
    high_uncertainty_steps: List[int]  # NEW: Which steps have high uncertainty

    # Execution tracking
    steps: int
    attempts: int
    execution_attempts: int
    reflexion_attempts: int
    last_execution_error: Optional[str]

    # Phase 4: Smart Reflexion
    should_reflect: bool  # NEW: Whether reflexion is needed
    reflection_trigger: Optional[str]  # NEW: Why reflexion was triggered

    # Output
    status: Literal["pending", "confident", "asking", "executing", "done", "escalated"]
    result: Optional[ToolCall]
    execution_result: Optional[ExecutionResult]
    error: Optional[str]
    warning: Optional[str]  # NEW: Soft escalation warning
    confidence_score: Optional[float]  # NEW: Confidence for soft escalated results


# =============================================================================
# Configuration (Enhanced)
# =============================================================================

CONFIG = {
    # Uncertainty thresholds
    "confidence_threshold": 0.7,
    "uncertainty_threshold": 0.3,
    "max_attempts": 3,
    "max_execution_retries": 2,

    # Phase 1: SGR
    "enable_sgr": True,
    "per_field_uncertainty": True,
    "sgr_reasoning_steps": True,

    # Phase 2: Resampling & Budget Allocation
    "enable_resampling": True,
    "base_samples": 1,  # Start with 1 sample
    "max_samples": 3,  # TUNED: Reduced from 5 to 3 (less overhead)
    "high_uncertainty_sample_threshold": 0.7,  # TUNED: Increased from 0.6 (less aggressive)
    "agreement_threshold": 0.7,  # Stop early if agreement > 0.7

    # Phase 3: SAUP
    "enable_saup_tracking": True,
    "saup_escalation_threshold": 0.95,  # TUNED: Increased from 0.85 (less aggressive)
    "track_reasoning_traces": True,

    # Phase 4: Smart Reflexion
    "enable_reflexion": True,
    "max_reflexion_attempts": 2,
    "reflexion_only_on_failure": True,
    "reflexion_uncertainty_threshold": 0.7,

    # Uncertainty combination
    "structured_weight": 0.7,
    "llm_modulation": 0.5,

    # Critical operations
    "critical_patterns": ["delete", "cancel", "remove", "drop", "terminate", "destroy"],
    "critical_threshold_multiplier": 0.5,

    # Escalation
    "escalation_uncertainty": 0.95,  # TUNED: Increased from 0.85
    "max_high_uncertainty_steps": 5,  # TUNED: Increased from 3

    # Soft Escalation (NEW)
    "enable_soft_escalation": True,  # Return best guess for non-critical ops instead of giving up
}


@dataclass
class GraphDeps:
    """All dependencies injected into the graph (enhanced)."""
    tool_schemas: Dict[str, ToolSchema]
    candidate_generator: LLMBackedCandidateGenerator
    question_generator: LLMBackedQuestionGenerator
    question_asker: Any
    tool_executor: ToolExecutor
    constraint_extractor: ConstraintExtractor
    config: SageAgentConfig
    uncertainty_propagator: Optional[UncertaintyPropagator] = None

    # Phase 1: SGR components
    uncertainty_decomposer: Optional[UncertaintyDecomposer] = None

    # Phase 3: CoT verifier
    cot_verifier: Optional[ChainOfThoughtVerifier] = None


# =============================================================================
# Helper Functions (from v2, with enhancements)
# =============================================================================

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


def _coerce_to_domain(value: object, domain: ParameterDomain) -> object:
    if not domain.is_finite():
        return value
    values = list(domain.values or [])
    if not values:
        return value
    if isinstance(value, list):
        if all(isinstance(v, bool) for v in value) and len(value) == 1:
            return value[0]
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        for option in values:
            if isinstance(option, str) and option.lower() == lowered:
                return option
        for option in values:
            if isinstance(option, str) and lowered in option.lower():
                return option
    return value


def _normalize_candidate_arguments(
    candidate: ToolCallCandidate,
    schema: ToolSchema,
) -> ToolCallCandidate:
    normalized = dict(candidate.arguments)
    for param, domain in schema.parameters.items():
        if param in normalized:
            normalized[param] = _coerce_to_domain(normalized[param], domain)
    return ToolCallCandidate(tool_name=candidate.tool_name, arguments=normalized)


# =============================================================================
# Phase 1: Schema Guided Reasoning Functions
# =============================================================================

def _compute_field_uncertainty(
    param_name: str,
    value: object,
    domain: ParameterDomain,
    observations: List[str],
) -> FieldUncertainty:
    """Compute uncertainty for a specific field."""

    # If unknown, max uncertainty
    if value == UNK:
        return FieldUncertainty(
            param_name=param_name,
            value=value,
            uncertainty=1.0,
            source="unknown",
            reasoning="Parameter value is unknown"
        )

    # If directly from user observation, low uncertainty
    for obs in observations:
        obs_lower = obs.lower()
        if param_name.lower() in obs_lower and str(value).lower() in obs_lower:
            return FieldUncertainty(
                param_name=param_name,
                value=value,
                uncertainty=0.1,
                source="asked",
                reasoning=f"Directly stated in observation: {obs[:50]}"
            )

    # If finite domain with few options, check ambiguity
    if domain.is_finite():
        num_options = len(list(domain.values or []))
        if num_options <= 2:
            return FieldUncertainty(
                param_name=param_name,
                value=value,
                uncertainty=0.3,
                source="inferred",
                reasoning=f"Inferred from {num_options} options"
            )
        else:
            return FieldUncertainty(
                param_name=param_name,
                value=value,
                uncertainty=0.6,
                source="inferred",
                reasoning=f"Inferred from {num_options} options (high ambiguity)"
            )

    # Continuous domain - medium uncertainty
    return FieldUncertainty(
        param_name=param_name,
        value=value,
        uncertainty=0.5,
        source="inferred",
        reasoning="Inferred from continuous domain"
    )


def _validate_with_json_schema(
    candidate: ToolCallCandidate,
    schema: ToolSchema,
) -> Tuple[bool, Optional[str]]:
    """Phase 1: Constrained decoding - validate against schema.

    NOTE: This validation allows <UNK> values! The whole point of SAGE is to
    handle unknowns and ask clarifying questions. We only validate:
    1. Parameter names are valid (in schema)
    2. Non-UNK values are within their domains

    Returns:
        (is_valid, error_message)
    """
    # Check parameter names and domains (but ALLOW <UNK> values)
    for param, value in candidate.arguments.items():
        if param not in schema.parameters:
            return False, f"Unknown parameter: {param}"

        # Skip UNK values - they're allowed!
        if value == UNK:
            continue

        domain = schema.parameters[param]

        # Validate finite domains for non-UNK values
        if domain.is_finite():
            valid_values = list(domain.values or [])
            if value not in valid_values:
                return False, f"Invalid value for {param}: {value} not in {valid_values}"

    return True, None


# =============================================================================
# Phase 2: Self-Consistency / Best-of-N Resampling
# =============================================================================

def _resample_candidates(
    deps: GraphDeps,
    user_input: str,
    observations: List[str],
    num_samples: int,
) -> Tuple[List[ToolCallCandidate], DecomposedUncertainty]:
    """Phase 2: Generate multiple samples and compute uncertainty via disagreement."""

    if num_samples <= 1:
        # No resampling - generate single candidate
        candidates = deps.candidate_generator.generate_candidates(
            user_input, observations, deps.tool_schemas
        )
        return candidates[:1], DecomposedUncertainty(0.5, 0.5, 1.0)

    # Generate multiple samples
    all_samples = []
    for _ in range(num_samples):
        samples = deps.candidate_generator.generate_candidates(
            user_input, observations, deps.tool_schemas
        )
        if samples:
            all_samples.append(samples[0])  # Take best from each sample

    if not all_samples:
        return [], DecomposedUncertainty(0.5, 0.5, 1.0)

    # Use UncertaintyDecomposer if available
    if deps.uncertainty_decomposer:
        # Convert to string representations for comparison
        sample_strs = [
            f"{c.tool_name}({json.dumps(dict(c.arguments), sort_keys=True)})"
            for c in all_samples
        ]

        decomposed = deps.uncertainty_decomposer.decompose_from_samples(
            sample_strs,
            extract_answer=lambda s: s,  # Already strings
        )

        return all_samples, decomposed

    # Fallback: simple disagreement
    unique_tools = len(set(c.tool_name for c in all_samples))
    epistemic = 1.0 - (1.0 / unique_tools) if unique_tools > 0 else 0.5

    return all_samples, DecomposedUncertainty(0.3, epistemic, epistemic + 0.3)


def _compute_dynamic_sample_budget(
    current_uncertainty: float,
    epistemic_uncertainty: float,
) -> int:
    """Phase 2: Decide how many samples to generate based on uncertainty."""

    if not CONFIG["enable_resampling"]:
        return CONFIG["base_samples"]

    # If epistemic uncertainty is high, more samples help
    if epistemic_uncertainty > CONFIG["high_uncertainty_sample_threshold"]:
        return CONFIG["max_samples"]
    elif epistemic_uncertainty > 0.4:
        return min(3, CONFIG["max_samples"])
    else:
        return CONFIG["base_samples"]


# =============================================================================
# Phase 3: SAUP Trajectory Tracking
# =============================================================================

def _add_reasoning_trace(
    state: AgentState,
    step: str,
    thought: str,
    action: str,
    uncertainty: float,
    fields_affected: List[str],
) -> AgentState:
    """Phase 3: Add a reasoning trace to trajectory."""

    if not CONFIG["track_reasoning_traces"]:
        return state

    trace = ReasoningTrace(
        step=step,
        thought=thought,
        action=action,
        uncertainty=uncertainty,
        fields_affected=fields_affected,
    )

    new_traces = list(state.get("reasoning_traces", []))
    new_traces.append(trace)

    # Track high uncertainty steps
    high_unc_steps = list(state.get("high_uncertainty_steps", []))
    if uncertainty > 0.6:
        high_unc_steps.append(len(new_traces) - 1)

    return {
        **state,
        "reasoning_traces": new_traces,
        "high_uncertainty_steps": high_unc_steps,
    }


# =============================================================================
# Phase 4: Smart Reflexion
# =============================================================================

def _should_trigger_reflexion(
    state: AgentState,
    execution_failed: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Phase 4: Decide if reflexion should be triggered.

    Returns:
        (should_reflect, trigger_reason)
    """
    if not CONFIG["enable_reflexion"]:
        return False, None

    # Check if we've exhausted reflexion attempts
    if state.get("reflexion_attempts", 0) >= CONFIG["max_reflexion_attempts"]:
        return False, None

    # Only on failure mode
    if CONFIG["reflexion_only_on_failure"]:
        if execution_failed:
            return True, "execution_failure"

        # Or if uncertainty remains very high after multiple attempts
        if (state.get("attempts", 0) >= 2 and
            state.get("combined_uncertainty", 0) > CONFIG["reflexion_uncertainty_threshold"]):
            return True, "persistent_high_uncertainty"

        return False, None

    # Original behavior: always reflect on errors
    if execution_failed:
        return True, "execution_failure"

    return False, None


def _generate_smart_reflection(
    deps: GraphDeps,
    state: AgentState,
) -> str:
    """Phase 4: Generate contextual reflection based on failure mode."""

    trigger = state.get("reflection_trigger", "unknown")
    error = state.get("last_execution_error", "Unknown error")
    candidate = state["candidates"][state["best_candidate_index"]]

    llm = getattr(deps.candidate_generator, "llm", None)
    if llm is None or not hasattr(llm, "complete"):
        return f"Execution failed: {error}"

    # Tailored prompts based on trigger
    if trigger == "execution_failure":
        prompt = (
            "Analyze why this tool call failed and what constraints were violated.\n\n"
            f"User input: {state['user_input']}\n"
            f"Tool call: {candidate.tool_name}({dict(candidate.arguments)})\n"
            f"Error: {error}\n\n"
            "Focus on:\n"
            "1. Which parameter values were incorrect?\n"
            "2. What constraints or dependencies were missed?\n"
            "3. What should be asked to clarify?\n\n"
            "Reflection:"
        )
    elif trigger == "persistent_high_uncertainty":
        # Include uncertainty breakdown
        breakdown = ""
        if state.get("field_uncertainties"):
            field_uncs = state["field_uncertainties"].get(str(state["best_candidate_index"]), {})
            breakdown = "\n".join([
                f"- {param}: {fu.uncertainty:.2f} ({fu.reasoning})"
                for param, fu in field_uncs.items()
            ])

        prompt = (
            "We remain uncertain after multiple clarification rounds.\n\n"
            f"User input: {state['user_input']}\n"
            f"Current candidate: {candidate.tool_name}({dict(candidate.arguments)})\n"
            f"Observations so far: {state['observations']}\n\n"
            f"Uncertainty breakdown:\n{breakdown}\n\n"
            "What key information are we still missing? What's the root cause of uncertainty?\n\n"
            "Reflection:"
        )
    else:
        prompt = (
            f"Reflect on why we're struggling with this task.\n\n"
            f"User input: {state['user_input']}\n"
            f"Error: {error}\n\n"
            "Reflection:"
        )

    return llm.complete(prompt).strip()


# =============================================================================
# Graph Nodes (Enhanced)
# =============================================================================

def build_graph(deps: GraphDeps) -> StateGraph:
    """Build the enhanced SAGE-Agent v3 graph."""

    graph = StateGraph(AgentState)

    def generate_candidates_node(state: AgentState) -> AgentState:
        """Generate candidates with Phase 1 (SGR) + Phase 2 (resampling) + Phase 3 (SAUP)."""

        # Phase 2: Compute dynamic sample budget
        epistemic_unc = state.get("epistemic_uncertainty", 0.5)
        num_samples = _compute_dynamic_sample_budget(
            state.get("combined_uncertainty", 1.0),
            epistemic_unc,
        )

        # Phase 2: Resample candidates
        samples, decomposed_unc = _resample_candidates(
            deps,
            state["user_input"],
            state["observations"],
            num_samples,
        )

        # Normalize and validate
        candidates = []
        for c in samples:
            if c.tool_name not in deps.tool_schemas:
                continue

            schema = deps.tool_schemas[c.tool_name]
            normalized = _normalize_candidate_arguments(c, schema)

            # Phase 1: Validate with JSON schema
            is_valid, error = _validate_with_json_schema(normalized, schema)
            if is_valid:
                candidates.append(normalized)

        if not candidates:
            return {
                **state,
                "error": "No valid candidates after schema validation",
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

        # Phase 1: Compute per-field uncertainties
        field_uncs = {}
        for idx, candidate in enumerate(candidates):
            schema = deps.tool_schemas[candidate.tool_name]
            field_uncs[str(idx)] = {
                param: _compute_field_uncertainty(
                    param, value, schema.parameters[param], state["observations"]
                )
                for param, value in candidate.arguments.items()
            }

        # Phase 3: Track in SAUP propagator
        if deps.uncertainty_propagator:
            deps.uncertainty_propagator.observe(struct_unc, "candidate_generation")
            if llm_unc_raw is not None:
                deps.uncertainty_propagator.observe(llm_unc, "llm_parsing")

        # Phase 3: Add reasoning trace
        updated_state = _add_reasoning_trace(
            state,
            step="candidate_generation",
            thought=f"Generated {len(candidates)} valid candidates with {num_samples} samples",
            action=f"Best: {candidates[best_idx].tool_name} (p={probs[best_idx]:.2f})",
            uncertainty=combined,
            fields_affected=list(candidates[best_idx].arguments.keys()),
        )

        return {
            **updated_state,
            "candidates": candidates,
            "probabilities": probs,
            "best_candidate_index": best_idx,
            "uncertainty": struct_unc,
            "llm_uncertainty": llm_unc,
            "combined_uncertainty": combined,
            "epistemic_uncertainty": decomposed_unc.epistemic,
            "aleatoric_uncertainty": decomposed_unc.aleatoric,
            "field_uncertainties": field_uncs,
            "num_samples": num_samples,
            "samples": samples,
            "sample_agreement": 1.0 - decomposed_unc.epistemic,
            "trajectory_uncertainty": deps.uncertainty_propagator.accumulated_uncertainty if deps.uncertainty_propagator else combined,
            "status": "pending",
        }

    def check_confidence_router(state: AgentState) -> Literal["confident", "need_questions", "escalate"]:
        """Decide if we're confident enough to execute (with Phase 3 SAUP escalation)."""
        if state["error"]:
            return "confident"

        candidate = state["candidates"][state["best_candidate_index"]]
        schema = deps.tool_schemas[candidate.tool_name]

        # Must ask if required params unknown
        if _has_required_unknowns(candidate, schema):
            return "need_questions"

        # Phase 3: SAUP-based escalation
        if CONFIG["enable_saup_tracking"] and deps.uncertainty_propagator:
            if deps.uncertainty_propagator.should_escalate(
                escalation_threshold=CONFIG["saup_escalation_threshold"],
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
        """Generate clarifying questions."""
        questions = deps.question_generator.generate_questions(
            state["user_input"],
            state["candidates"],
            state["observations"],
            deps.tool_schemas,
        )

        if not questions:
            candidate = state["candidates"][state["best_candidate_index"]]
            schema = deps.tool_schemas[candidate.tool_name]
            if _has_required_unknowns(candidate, schema):
                return {
                    **state,
                    "questions": [],
                    "best_question": None,
                    "best_score": 0.0,
                    "error": "Missing required parameters and no questions generated",
                    "status": "done",
                }
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

        max_prob = max(state["probabilities"]) if state["probabilities"] else 0.0
        if state["best_score"] < deps.config.alpha * max_prob:
            return "execute"

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
            deps.uncertainty_propagator.observe(0.1, "belief_update")

        return {**state, "domains": new_domains}

    def execute_node(state: AgentState) -> AgentState:
        """Execute the best candidate tool call (with Phase 1 validation)."""
        if state["error"]:
            return {**state, "status": "done"}

        candidate = state["candidates"][state["best_candidate_index"]]
        schema = deps.tool_schemas[candidate.tool_name]

        # Phase 1: Final validation before execution
        is_valid, error = _validate_with_json_schema(candidate, schema)
        if not is_valid:
            return {
                **state,
                "error": f"Schema validation failed: {error}",
                "status": "done",
            }

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
        """Handle execution error - Phase 4: Smart Reflexion."""
        exec_result = state.get("execution_result")
        error_msg = exec_result.error if exec_result else "Unknown error"
        observations = list(state["observations"])
        observations.append(f"Execution failed: {error_msg}")

        # Phase 4: Decide if reflexion should be triggered
        should_reflect, trigger = _should_trigger_reflexion(state, execution_failed=True)

        updated_state = {
            **state,
            "last_execution_error": error_msg,
            "execution_attempts": state.get("execution_attempts", 0) + 1,
            "observations": observations,
            "should_reflect": should_reflect,
            "reflection_trigger": trigger,
        }

        if should_reflect:
            # Generate smart reflection
            reflection = _generate_smart_reflection(deps, updated_state)
            observations.append(f"Reflection: {reflection}")

            if deps.uncertainty_propagator:
                deps.uncertainty_propagator.observe(
                    step_uncertainty=state.get("combined_uncertainty", state["uncertainty"]),
                    step_type="reflexion",
                    metadata={"trigger": trigger},
                )

            return {
                **updated_state,
                "observations": observations,
                "reflexion_attempts": state.get("reflexion_attempts", 0) + 1,
            }

        return updated_state

    def escalate_node(state: AgentState) -> AgentState:
        """Escalate to human when agent cannot resolve.

        Uses soft escalation for non-critical operations:
        - Critical ops (delete, cancel, etc.): Hard stop
        - Non-critical ops: Return best guess with warning
        """

        # Phase 3: Provide detailed uncertainty breakdown for debugging
        breakdown = ""
        if deps.uncertainty_propagator:
            unc_breakdown = deps.uncertainty_propagator.get_uncertainty_breakdown()
            breakdown = "\n".join([
                f"  {step_type}: {unc:.3f}"
                for step_type, unc in unc_breakdown.items()
            ])

        error_msg = state.get("error") or state.get("last_execution_error") or "Escalated due to high uncertainty"

        if breakdown:
            error_msg += f"\n\nUncertainty breakdown:\n{breakdown}"

        # Check if we have a valid candidate for soft escalation
        enable_soft = CONFIG.get("enable_soft_escalation", True)
        candidates = state.get("candidates", [])

        if enable_soft and candidates:
            best_idx = state.get("best_candidate_index", 0)
            if best_idx < len(candidates):
                candidate = candidates[best_idx]

                # Check if operation is critical
                is_critical = any(
                    p in candidate.tool_name.lower()
                    for p in CONFIG["critical_patterns"]
                )

                if not is_critical:
                    # Soft escalation: return best guess with warning
                    uncertainty = state.get("trajectory_uncertainty", state.get("combined_uncertainty", 1.0))

                    # Try to create tool call (handle UNK values)
                    try:
                        schema = deps.tool_schemas.get(candidate.tool_name)
                        if schema:
                            # Check if we can execute (no required UNK)
                            has_required_unk = any(
                                candidate.arguments.get(p, UNK) == UNK
                                for p in schema.required
                            )

                            if not has_required_unk:
                                # We have all required params - soft escalate with execution
                                tool_call = ToolCall(
                                    tool_name=candidate.tool_name,
                                    arguments=candidate.arguments
                                )

                                return {
                                    **state,
                                    "status": "done",
                                    "result": tool_call,
                                    "error": None,
                                    "warning": f"High uncertainty ({uncertainty:.2f}), result may be incorrect",
                                    "confidence_score": 1.0 - uncertainty,
                                }
                    except Exception:
                        pass  # Fall through to hard escalation

        # Hard escalation (critical ops or soft escalation failed)
        return {
            **state,
            "status": "escalated",
            "error": error_msg,
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

    # Edges
    graph.add_conditional_edges(
        "generate_candidates",
        check_confidence_router,
        {
            "confident": "execute",
            "need_questions": "generate_questions",
            "escalate": "escalate",
        }
    )

    graph.add_conditional_edges(
        "generate_questions",
        select_question_router,
        {
            "ask": "ask_question",
            "execute": "execute",
        }
    )

    graph.add_edge("ask_question", "update_belief")
    graph.add_edge("update_belief", "generate_candidates")

    graph.add_conditional_edges(
        "execute",
        validate_result_router,
        {
            "success": "handle_success",
            "retry": "handle_error",
            "escalate": "escalate",
        }
    )

    graph.add_edge("handle_error", "generate_candidates")

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
        "field_uncertainties": {},
        "questions": [],
        "best_question": None,
        "best_score": 0.0,
        "aspect_counts": {},
        "uncertainty": 1.0,
        "llm_uncertainty": 0.5,
        "combined_uncertainty": 1.0,
        "epistemic_uncertainty": 0.5,
        "aleatoric_uncertainty": 0.5,
        "num_samples": CONFIG["base_samples"],
        "samples": [],
        "sample_agreement": 0.0,
        "reasoning_traces": [],
        "trajectory_uncertainty": 1.0,
        "high_uncertainty_steps": [],
        "steps": 0,
        "attempts": 0,
        "execution_attempts": 0,
        "reflexion_attempts": 0,
        "last_execution_error": None,
        "should_reflect": False,
        "reflection_trigger": None,
        "status": "pending",
        "result": None,
        "execution_result": None,
        "error": None,
        "warning": None,
        "confidence_score": None,
    }


# =============================================================================
# Demo
# =============================================================================

class InteractiveQuestionAsker:
    """Ask questions interactively via console."""
    def ask(self, question: Question) -> str:
        print(f"\n🤖 Agent: {question.text}")
        return input("👤 You: ").strip()


def main():
    """Demo the enhanced SAGE-Agent v3."""
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
    llm = OllamaClient(model="qwen2.5:3b-instruct-q8_0", verbose=False)

    # Phase 3: Create SAUP propagator
    propagator = create_sage_propagator()

    # Phase 1: Create uncertainty decomposer
    decomposer = UncertaintyDecomposer(num_samples=CONFIG["max_samples"])

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
        uncertainty_decomposer=decomposer,
    )

    # Build and compile
    graph = build_graph(deps).compile()

    # Run
    initial = create_initial_state("I need to fly to LAX", deps.tool_schemas)
    result = graph.invoke(initial, {"recursion_limit": 50})

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Status: {result['status']}")
    print(f"Result: {result['result']}")
    print(f"Steps: {result['steps']}")
    print(f"Attempts: {result['attempts']}")
    print(f"\nUncertainty Analysis:")
    print(f"  Combined: {result['combined_uncertainty']:.3f}")
    print(f"  Epistemic: {result['epistemic_uncertainty']:.3f}")
    print(f"  Aleatoric: {result['aleatoric_uncertainty']:.3f}")
    print(f"  Trajectory: {result['trajectory_uncertainty']:.3f}")
    print(f"\nResampling:")
    print(f"  Samples used: {result['num_samples']}")
    print(f"  Sample agreement: {result['sample_agreement']:.3f}")
    print(f"\nReflexion:")
    print(f"  Reflexion attempts: {result['reflexion_attempts']}")
    print(f"  Reflection triggered: {result.get('should_reflect', False)}")
    if result.get('reflection_trigger'):
        print(f"  Trigger reason: {result['reflection_trigger']}")

    if result['error']:
        print(f"\nError: {result['error']}")

    # Phase 1: Show per-field uncertainties
    if result.get('field_uncertainties'):
        best_idx = result['best_candidate_index']
        field_uncs = result['field_uncertainties'].get(str(best_idx), {})
        if field_uncs:
            print(f"\nPer-Field Uncertainties:")
            for param, fu in field_uncs.items():
                print(f"  {param}: {fu.uncertainty:.3f} ({fu.source}) - {fu.reasoning}")

    print("=" * 80)


if __name__ == "__main__":
    main()
