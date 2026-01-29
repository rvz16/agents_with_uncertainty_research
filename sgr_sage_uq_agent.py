"""SGR-SAGE-UQ Agent: High-reliability AI agent combining Schema-Guided Reasoning,
Self-Consistency for Uncertainty Quantification, and SAGE decision logic.

This implements the technical plan for integrating:
- Phase 1: SGR Foundation - Pydantic schemas for structured tool calls
- Phase 2: Self-Consistency Bridge - UQ via TTS endpoint sampling
- Phase 3: SAGE Decision Engine - Risk-based act/clarify decisions
- Phase 4: LangGraph Topology - Full graph with Bayesian belief updates
- Phase 5: Testing & Benchmarking Ready - Traceability for ClarifyBench

Architecture:
    ┌──────────────────────────────────────────────────────────────────────┐
    │                         SGR-SAGE-UQ Agent                            │
    │                                                                      │
    │  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────────┐  │
    │  │ SGR Layer   │───>│ UQ Layer (TTS)  │───>│ SAGE Decision Engine │  │
    │  │ (Schemas)   │    │ (Self-Consist.) │    │ (EVPI + Risk)        │  │
    │  └─────────────┘    └─────────────────┘    └──────────────────────┘  │
    │         │                    │                       │               │
    │         ▼                    ▼                       ▼               │
    │  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────────┐  │
    │  │ Tool Schema │    │ N Hypotheses    │    │ ACT / CLARIFY        │  │
    │  │ Validation  │    │ → Probability   │    │ Decision             │  │
    │  └─────────────┘    └─────────────────┘    └──────────────────────┘  │
    └──────────────────────────────────────────────────────────────────────┘

Graph Flow:
    START → generate_hypotheses → analyze_uncertainty →
        [ACT] → execute_tool → validate_result → END
        [CLARIFY] → human_clarification → update_belief → generate_hypotheses

Usage:
    from sgr_sage_uq_agent import SGRSageUQAgent, create_agent, run_agent

    agent = create_agent(tool_schemas, tts_config)
    result = run_agent(agent, "Book a flight from NYC to LA")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    TypedDict,
    Union,
)

from pydantic import BaseModel, Field, ValidationError, field_validator

# Add parent directory for imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from langgraph.graph import END, StateGraph

# Import SAGE agent core components
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
from sage_agent.core.constraints import (
    HybridConstraintExtractor,
    LLMConstraintExtractor,
    SimpleConstraintExtractor,
)
from sage_agent.core.evpi import compute_evpi
from sage_agent.core.types import ConstraintExtractor, ExecutionResult, ToolExecutor
from sage_agent.core.uncertainty_propagation import (
    PropagationMode,
    UncertaintyObservation,
    UncertaintyPropagator,
)
from sage_agent.core.advanced_reasoning import (
    ChainOfThoughtVerifier,
    DecomposedUncertainty,
    UncertaintyDecomposer,
)

# TTS integration for self-consistency sampling
from llm_tts.integration import ChatTTS

# Import TTSLLMClient (handle case where examples is not a package)
try:
    from examples.tts_llm_client import TTSLLMClient
except ImportError:
    # Define inline if import fails
    from langchain_core.messages import HumanMessage, SystemMessage
    from sage_agent import LLMClient

    @dataclass
    class TTSLLMClient(LLMClient):
        """TTS-enabled LLM client using self-consistency for uncertainty."""
        base_url: str
        model: str
        tts_strategy: str = "self_consistency"
        tts_budget: int = 8
        temperature: float = 0.7
        max_tokens: int = 4096
        timeout: float = 120.0
        system_prompt: str = (
            "You are a precise assistant. "
            "Respond with one line only in this exact format: Answer: <final answer>."
        )
        last_metadata: dict = field(default_factory=dict)

        def __post_init__(self) -> None:
            self._llm = ChatTTS(
                base_url=self.base_url,
                model=self.model,
                tts_strategy=self.tts_strategy,
                tts_budget=self.tts_budget,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
            if self.last_metadata is None:
                self.last_metadata = {}

        def complete(self, prompt: str) -> str:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt),
            ]
            response = self._llm.invoke(messages)
            self.last_metadata = response.response_metadata.get("tts_metadata", {})
            return response.content

        @property
        def last_uncertainty(self) -> Optional[float]:
            if not self.last_metadata:
                return None
            return self.last_metadata.get("uncertainty_score")

        @property
        def last_confidence(self) -> Optional[float]:
            if not self.last_metadata:
                return None
            return self.last_metadata.get("consensus_score")

log = logging.getLogger(__name__)

# =============================================================================
# Phase 1: Core Schema Definitions (SGR Foundation)
# =============================================================================


class ActionType(str, Enum):
    """Types of actions the agent can take."""
    TOOL_CALL = "tool_call"
    CLARIFY = "clarify"
    ESCALATE = "escalate"


class ParameterValue(BaseModel):
    """A validated parameter value with uncertainty tracking."""
    name: str
    value: Any
    uncertainty: float = Field(ge=0.0, le=1.0, default=0.5)
    source: Literal["user", "inferred", "default", "unknown"] = "unknown"

    @property
    def is_unknown(self) -> bool:
        return self.value == UNK or self.source == "unknown"


class ActionSchema(BaseModel):
    """SGR-compliant action schema with full validation.

    This is the Pydantic-based schema for guaranteed-valid tool calls.
    Each action is hashable for belief distribution mapping.
    """
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

    # Per-parameter uncertainty tracking (Phase 1 SGR)
    parameter_uncertainties: Dict[str, float] = Field(default_factory=dict)

    @field_validator('arguments', mode='before')
    @classmethod
    def validate_arguments(cls, v):
        if v is None:
            return {}
        return dict(v)

    def compute_hash(self) -> str:
        """Compute unique hash for this action (for belief distribution)."""
        content = json.dumps({
            "tool": self.tool_name,
            "args": self.arguments,
        }, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def to_tool_call_candidate(self) -> ToolCallCandidate:
        """Convert to SAGE ToolCallCandidate."""
        return ToolCallCandidate(
            tool_name=self.tool_name,
            arguments=self.arguments,
        )

    @classmethod
    def from_tool_call_candidate(
        cls,
        candidate: ToolCallCandidate,
        confidence: float = 0.5,
        reasoning: str = "",
    ) -> "ActionSchema":
        """Create from SAGE ToolCallCandidate."""
        return cls(
            tool_name=candidate.tool_name,
            arguments=dict(candidate.arguments),
            confidence=confidence,
            reasoning=reasoning,
        )


class ClarificationRequest(BaseModel):
    """Request for user clarification."""
    question: str
    aspects: List[Tuple[str, str]]  # [(tool_name, param_name), ...]
    priority: float = Field(ge=0.0, le=1.0, default=0.5)
    evpi_score: float = 0.0


class DecisionMetrics(BaseModel):
    """SAGE decision metrics for act/clarify logic."""
    evpi_score: float = 0.0
    risk_score: float = 0.0
    cost_of_clarification: float = 0.1
    penalty_of_error: float = 1.0
    threshold_exceeded: bool = False
    chosen_action: Literal["ACT", "CLARIFY", "ESCALATE"] = "CLARIFY"

    @property
    def expected_risk(self) -> float:
        """Expected risk if we act now."""
        return self.risk_score * self.penalty_of_error

    @property
    def should_clarify(self) -> bool:
        """SAGE decision: clarify if risk > cost."""
        return self.expected_risk > self.cost_of_clarification


# =============================================================================
# Phase 2: Self-Consistency Bridge (UQ Layer)
# =============================================================================


@dataclass
class TTSConfig:
    """Configuration for TTS (Test-Time Scaling) service."""
    service_url: str = "http://localhost:8001/v1"
    model: str = "openai/gpt-4o-mini"
    tts_strategy: str = "self_consistency"
    tts_budget: int = 10  # Number of samples for self-consistency
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 120.0


class SelfConsistencyClient:
    """Client for getting self-consistency samples from TTS service.

    This wraps the TTS endpoint to provide N structured samples
    for uncertainty quantification via agreement analysis.
    """

    def __init__(self, config: TTSConfig):
        self.config = config
        self._llm = ChatTTS(
            base_url=config.service_url,
            model=config.model,
            tts_strategy=config.tts_strategy,
            tts_budget=config.tts_budget,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
        )
        self._last_metadata: Dict[str, Any] = {}

    def get_samples(
        self,
        prompt: str,
        system_prompt: str = "",
        n: int = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Get N self-consistency samples from TTS service.

        Args:
            prompt: The user prompt
            system_prompt: System instructions
            n: Number of samples (defaults to tts_budget)

        Returns:
            Tuple of (samples list, metadata dict)
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        # Use configured budget if n not specified
        num_samples = n or self.config.tts_budget

        # Build messages
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        # Get response with TTS metadata
        response = self._llm.invoke(messages)
        self._last_metadata = response.response_metadata.get("tts_metadata", {})

        # TTS service returns aggregated response with metadata about samples
        # The metadata contains info about the N internal samples
        return [response.content], self._last_metadata

    @property
    def last_uncertainty(self) -> Optional[float]:
        """Get uncertainty score from last call."""
        return self._last_metadata.get("uncertainty_score")

    @property
    def last_confidence(self) -> Optional[float]:
        """Get consensus/confidence score from last call."""
        return self._last_metadata.get("consensus_score")


def get_self_consistency_samples(
    client: SelfConsistencyClient,
    prompt: str,
    tool_schemas: Dict[str, ToolSchema],
    n: int = 10,
) -> List[ActionSchema]:
    """Get N self-consistency samples as ActionSchema objects.

    This is the main API for Phase 2 - it calls the TTS service
    and parses the responses into validated action schemas.

    Args:
        client: The TTS client
        prompt: The prompt to send
        tool_schemas: Available tool schemas for validation
        n: Number of samples

    Returns:
        List of parsed and validated ActionSchema objects
    """
    from sage_agent.llm.prompts import build_candidate_prompt

    # Build structured prompt for candidate generation
    system_prompt = (
        "You are a precise assistant that proposes tool calls.\n"
        "Return a JSON list with tool call candidates.\n"
        "Each item: {\"tool\": string, \"arguments\": {param: value or <UNK>}}\n"
        "Use <UNK> for unknown values. Only use valid tool names and parameters."
    )

    # Get samples from TTS service
    samples, metadata = client.get_samples(prompt, system_prompt, n)

    # Parse samples into ActionSchema objects
    actions: List[ActionSchema] = []

    for sample in samples:
        try:
            # Try to parse JSON from sample
            parsed = _parse_candidates_from_response(sample)

            for candidate in parsed:
                # Validate against tool schema
                if candidate.tool_name in tool_schemas:
                    schema = tool_schemas[candidate.tool_name]

                    # Normalize arguments
                    normalized_args = _normalize_arguments(
                        candidate.arguments,
                        schema,
                    )

                    action = ActionSchema(
                        tool_name=candidate.tool_name,
                        arguments=normalized_args,
                        confidence=metadata.get("consensus_score", 0.5),
                    )
                    actions.append(action)

        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            log.warning(f"Failed to parse sample: {e}")
            continue

    return actions


def _parse_candidates_from_response(response: str) -> List[ActionSchema]:
    """Parse candidates from LLM response."""
    import re

    # Try to extract JSON array
    json_match = re.search(r'\[.*\]', response, re.DOTALL)
    if json_match:
        data = json.loads(json_match.group())
        if isinstance(data, list):
            return [
                ActionSchema(
                    tool_name=item.get("tool", item.get("tool_name", "")),
                    arguments=item.get("arguments", item.get("args", {})),
                )
                for item in data
                if isinstance(item, dict)
            ]

    # Try single object
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        item = json.loads(json_match.group())
        return [ActionSchema(
            tool_name=item.get("tool", item.get("tool_name", "")),
            arguments=item.get("arguments", item.get("args", {})),
        )]

    return []


def _normalize_arguments(
    arguments: Dict[str, Any],
    schema: ToolSchema,
) -> Dict[str, Any]:
    """Normalize and validate arguments against schema."""
    normalized = {}

    for param_name, value in arguments.items():
        if param_name not in schema.parameters:
            continue

        domain = schema.parameters[param_name]

        # Handle UNK values
        if value == UNK or value == "<UNK>":
            normalized[param_name] = UNK
            continue

        # Coerce to domain if finite
        if domain.is_finite():
            valid_values = list(domain.values or [])

            # Try exact match
            if value in valid_values:
                normalized[param_name] = value
                continue

            # Try case-insensitive match
            if isinstance(value, str):
                for v in valid_values:
                    if isinstance(v, str) and v.lower() == value.lower():
                        normalized[param_name] = v
                        break
                else:
                    # Partial match
                    for v in valid_values:
                        if isinstance(v, str) and value.lower() in v.lower():
                            normalized[param_name] = v
                            break
                    else:
                        normalized[param_name] = value
            else:
                normalized[param_name] = value
        else:
            normalized[param_name] = value

    return normalized


# =============================================================================
# Phase 3: SAGE Decision Engine (Control Layer)
# =============================================================================


@dataclass
class SAGEConfig:
    """Configuration for SAGE decision engine."""
    # Base cost of asking a clarification question
    cost_of_clarification: float = 0.1

    # Base penalty for executing with error (dynamic per tool)
    base_penalty_of_error: float = 1.0

    # Tool patterns that increase penalty (critical operations)
    critical_tool_patterns: List[str] = field(default_factory=lambda: [
        "delete", "cancel", "remove", "drop", "terminate", "destroy",
        "purchase", "send", "transfer", "submit",
    ])

    # Penalty multiplier for critical tools
    critical_penalty_multiplier: float = 2.0

    # Minimum probability to execute without clarification
    min_confidence_to_act: float = 0.7

    # Maximum allowed accumulated uncertainty
    max_accumulated_uncertainty: float = 0.85

    # Maximum clarification rounds before escalation
    max_clarification_rounds: int = 5

    # Weight for structured vs LLM uncertainty
    structured_uncertainty_weight: float = 0.7


class SAGEDecisionEngine:
    """SAGE decision engine implementing the act/clarify logic.

    The core SAGE paper logic:
    1. Compute probability distribution P(h) over hypotheses
    2. Calculate Expected Risk = (1 - P_max) × Penalty
    3. IF Risk > Cost_clarify THEN CLARIFY ELSE ACT

    This implements utility-based decision making with:
    - Dynamic penalty based on tool criticality
    - EVPI (Expected Value of Perfect Information) for question selection
    - Accumulated uncertainty tracking via SAUP
    """

    def __init__(
        self,
        config: SAGEConfig,
        tool_schemas: Dict[str, ToolSchema],
        propagator: Optional[UncertaintyPropagator] = None,
    ):
        self.config = config
        self.tool_schemas = tool_schemas
        self.propagator = propagator or create_sage_propagator()

    def compute_belief_distribution(
        self,
        hypotheses: List[ActionSchema],
        domains: Dict[str, Dict[str, ParameterDomain]],
    ) -> Dict[str, float]:
        """Convert hypothesis frequency into probability distribution P(h).

        This implements Phase 3 probability mapping:
        - Count unique hypotheses
        - Weight by belief state compatibility
        - Normalize to probability distribution

        Returns:
            Dict mapping hypothesis hash → probability
        """
        if not hypotheses:
            return {}

        # Count hypothesis frequencies
        hash_counts: Counter = Counter()
        hash_to_action: Dict[str, ActionSchema] = {}

        for h in hypotheses:
            h_hash = h.compute_hash()
            hash_counts[h_hash] += 1
            hash_to_action[h_hash] = h

        # Compute belief-weighted probabilities
        belief = BeliefState(domains=domains, epsilon=1e-4)

        weighted_probs: Dict[str, float] = {}

        for h_hash, count in hash_counts.items():
            action = hash_to_action[h_hash]

            # Base frequency weight
            freq_weight = count / len(hypotheses)

            # Belief state weight (compatibility with domains)
            if action.tool_name in self.tool_schemas:
                schema = self.tool_schemas[action.tool_name]
                candidate = action.to_tool_call_candidate()
                belief_weight = belief.candidate_weight(candidate, schema)
            else:
                belief_weight = 0.0

            # Combined weight
            weighted_probs[h_hash] = freq_weight * max(belief_weight, 1e-10)

        # Normalize
        total = sum(weighted_probs.values())
        if total > 0:
            weighted_probs = {k: v / total for k, v in weighted_probs.items()}
        else:
            # Uniform distribution if all zero
            n = len(weighted_probs)
            weighted_probs = {k: 1.0 / n for k in weighted_probs}

        return weighted_probs

    def compute_penalty(self, action: ActionSchema) -> float:
        """Compute dynamic penalty based on tool criticality.

        Critical operations (delete, purchase, etc.) have higher penalties
        to encourage more clarification before execution.
        """
        base_penalty = self.config.base_penalty_of_error

        tool_lower = action.tool_name.lower()
        is_critical = any(
            pattern in tool_lower
            for pattern in self.config.critical_tool_patterns
        )

        if is_critical:
            return base_penalty * self.config.critical_penalty_multiplier

        return base_penalty

    def compute_risk(
        self,
        belief_distribution: Dict[str, float],
        best_action: ActionSchema,
    ) -> float:
        """Compute expected risk of acting now.

        Risk = (1 - P_max) × Penalty

        Where P_max is the probability of the best hypothesis.
        """
        if not belief_distribution:
            return 1.0

        # Get max probability
        p_max = max(belief_distribution.values())

        # Get penalty for best action
        penalty = self.compute_penalty(best_action)

        # Expected risk
        return (1.0 - p_max) * penalty

    def decide(
        self,
        hypotheses: List[ActionSchema],
        domains: Dict[str, Dict[str, ParameterDomain]],
        questions: List[Question],
        aspect_counts: Dict[str, int],
        clarification_rounds: int,
        llm_uncertainty: Optional[float] = None,
    ) -> Tuple[DecisionMetrics, Optional[ActionSchema], Optional[ClarificationRequest]]:
        """Make SAGE decision: ACT, CLARIFY, or ESCALATE.

        This is the core decision logic implementing:
        1. If risk > cost → CLARIFY
        2. Else → ACT
        3. If too many rounds or accumulated uncertainty too high → ESCALATE

        Returns:
            Tuple of (metrics, chosen_action, clarification_request)
        """
        # Compute belief distribution
        belief_dist = self.compute_belief_distribution(hypotheses, domains)

        if not belief_dist:
            return (
                DecisionMetrics(
                    chosen_action="ESCALATE",
                    risk_score=1.0,
                ),
                None,
                None,
            )

        # Find best hypothesis
        best_hash = max(belief_dist, key=belief_dist.get)
        best_prob = belief_dist[best_hash]
        best_action = next(
            h for h in hypotheses if h.compute_hash() == best_hash
        )

        # Check if best action has required unknowns
        has_required_unknowns = False
        if best_action.tool_name in self.tool_schemas:
            schema = self.tool_schemas[best_action.tool_name]
            for param in schema.required:
                if best_action.arguments.get(param, UNK) == UNK:
                    has_required_unknowns = True
                    break

        # Compute risk
        risk = self.compute_risk(belief_dist, best_action)
        cost = self.config.cost_of_clarification

        # Record uncertainty in propagator
        structured_uncertainty = 1.0 - best_prob
        if llm_uncertainty is not None:
            sw = self.config.structured_uncertainty_weight
            combined_uncertainty = sw * structured_uncertainty + (1 - sw) * llm_uncertainty
        else:
            combined_uncertainty = structured_uncertainty

        self.propagator.observe(
            step_uncertainty=combined_uncertainty,
            step_type="decision_point",
            metadata={"risk": risk, "best_prob": best_prob},
        )

        # Initialize metrics
        metrics = DecisionMetrics(
            evpi_score=0.0,
            risk_score=risk,
            cost_of_clarification=cost,
            penalty_of_error=self.compute_penalty(best_action),
        )

        # Check escalation conditions
        if clarification_rounds >= self.config.max_clarification_rounds:
            metrics.chosen_action = "ESCALATE"
            return (metrics, best_action, None)

        if self.propagator.should_escalate(
            escalation_threshold=self.config.max_accumulated_uncertainty,
            max_high_uncertainty_steps=4,
        ):
            metrics.chosen_action = "ESCALATE"
            return (metrics, best_action, None)

        # SAGE decision: Risk > Cost → CLARIFY
        if has_required_unknowns or risk > cost:
            metrics.threshold_exceeded = True

            # Select best question using EVPI
            clarify_request = self._select_best_question(
                questions,
                hypotheses,
                belief_dist,
                aspect_counts,
            )

            if clarify_request:
                metrics.evpi_score = clarify_request.evpi_score
                metrics.chosen_action = "CLARIFY"
                return (metrics, best_action, clarify_request)

            # No good questions available - act anyway if no required unknowns
            if not has_required_unknowns:
                metrics.chosen_action = "ACT"
                return (metrics, best_action, None)

            # Required unknowns but no questions - escalate
            metrics.chosen_action = "ESCALATE"
            return (metrics, best_action, None)

        # Risk <= Cost → ACT
        metrics.chosen_action = "ACT"
        return (metrics, best_action, None)

    def _select_best_question(
        self,
        questions: List[Question],
        hypotheses: List[ActionSchema],
        belief_dist: Dict[str, float],
        aspect_counts: Dict[str, int],
        redundancy_weight: float = 0.1,
    ) -> Optional[ClarificationRequest]:
        """Select best clarifying question using EVPI.

        EVPI (Expected Value of Perfect Information) measures how much
        the answer to a question would help us decide between hypotheses.

        Special case: If there's only one hypothesis with unknown required
        params, EVPI will be 0 but we should still allow clarification.
        """
        if not questions:
            return None

        # Convert hypotheses to candidates for EVPI computation
        candidates = [h.to_tool_call_candidate() for h in hypotheses]

        # Build probability list aligned with candidates
        probs = []
        for h in hypotheses:
            h_hash = h.compute_hash()
            probs.append(belief_dist.get(h_hash, 0.0))

        # Identify parameters with unknown values in hypotheses
        unknown_params: Set[Tuple[str, str]] = set()
        for h in hypotheses:
            if h.tool_name in self.tool_schemas:
                schema = self.tool_schemas[h.tool_name]
                for param in schema.required:
                    if h.arguments.get(param, UNK) == UNK:
                        unknown_params.add((h.tool_name, param))

        best_question = None
        best_score = float("-inf")
        best_evpi = 0.0

        for question in questions:
            # Compute EVPI
            evpi = compute_evpi(candidates, probs, question.aspects)

            # Compute redundancy cost
            redundancy_cost = sum(
                aspect_counts.get(f"{a.tool_name}:{a.param_name}", 0)
                for a in question.aspects
            ) * redundancy_weight

            # Check if question addresses required unknowns (bonus for these)
            addresses_unknown = any(
                (a.tool_name, a.param_name) in unknown_params
                for a in question.aspects
            )

            # Score = EVPI - redundancy cost
            # Add small bonus for questions addressing required unknowns
            # This ensures we can clarify even with single hypothesis
            score = evpi - redundancy_cost
            if addresses_unknown:
                score = max(score, 0.01)  # Minimum score for required unknowns

            if score > best_score:
                best_score = score
                best_question = question
                best_evpi = evpi

        if best_question is None or best_score <= 0:
            return None

        return ClarificationRequest(
            question=best_question.text,
            aspects=[(a.tool_name, a.param_name) for a in best_question.aspects],
            priority=min(best_evpi, 1.0),
            evpi_score=best_evpi,
        )


# =============================================================================
# Phase 4: LangGraph State & Topology
# =============================================================================


class SAGEAgentState(TypedDict):
    """Complete agent state for SGR-SAGE-UQ.

    This extends the LangGraph TypedDict to include:
    - messages: Conversation history
    - belief_distribution: Mapping of hypothesis hashes → probabilities
    - hypotheses: Raw list of N samples from self-consistency
    - decision_metrics: EVPI, risk, chosen action
    """
    # User input and conversation
    user_input: str
    messages: List[Dict[str, str]]
    observations: List[str]

    # Phase 2: Hypotheses from self-consistency
    hypotheses: List[ActionSchema]
    num_samples: int

    # Belief state (Phase 1 SGR)
    belief_distribution: Dict[str, float]
    domains: Dict[str, Dict[str, ParameterDomain]]

    # Best candidate tracking
    best_action: Optional[ActionSchema]
    best_probability: float

    # Uncertainty tracking
    structured_uncertainty: float
    llm_uncertainty: float
    combined_uncertainty: float
    accumulated_uncertainty: float

    # Question selection
    questions: List[Question]
    current_question: Optional[ClarificationRequest]
    aspect_counts: Dict[str, int]

    # Phase 3: SAGE decision metrics
    decision_metrics: Optional[DecisionMetrics]

    # Execution tracking
    clarification_rounds: int
    execution_attempts: int

    # Status and results
    status: Literal["pending", "generating", "analyzing", "acting", "clarifying", "done", "escalated"]
    result: Optional[ToolCall]
    execution_result: Optional[ExecutionResult]
    error: Optional[str]

    # Traceability (Phase 5)
    trace: List[Dict[str, Any]]


@dataclass
class GraphDependencies:
    """All dependencies injected into the LangGraph nodes."""
    tool_schemas: Dict[str, ToolSchema]
    tts_client: SelfConsistencyClient
    sage_engine: SAGEDecisionEngine
    candidate_generator: LLMBackedCandidateGenerator
    question_generator: LLMBackedQuestionGenerator
    question_asker: Any  # Protocol for asking questions
    tool_executor: ToolExecutor
    constraint_extractor: ConstraintExtractor
    config: SageAgentConfig
    propagator: UncertaintyPropagator


def build_sgr_sage_uq_graph(deps: GraphDependencies) -> StateGraph:
    """Build the complete SGR-SAGE-UQ LangGraph.

    Topology:
        generate_hypotheses → analyze_uncertainty →
            [ACT] → execute_tool → validate_result → END
            [CLARIFY] → human_clarification → update_belief → generate_hypotheses
            [ESCALATE] → escalate → END
    """
    graph = StateGraph(SAGEAgentState)

    # -------------------------------------------------------------------------
    # Node: generate_hypotheses
    # -------------------------------------------------------------------------
    def generate_hypotheses_node(state: SAGEAgentState) -> SAGEAgentState:
        """Generate N hypotheses via self-consistency sampling from TTS.

        This is the Phase 2 Self-Consistency Bridge node.
        """
        from sage_agent.llm.prompts import build_candidate_prompt

        # Build prompt
        prompt = build_candidate_prompt(
            state["user_input"],
            deps.tool_schemas.values(),
            state["observations"],
        )

        # Get self-consistency samples
        hypotheses = get_self_consistency_samples(
            deps.tts_client,
            prompt,
            deps.tool_schemas,
            n=state.get("num_samples", 10),
        )

        # If no hypotheses from TTS, fall back to direct generation
        if not hypotheses:
            candidates = deps.candidate_generator.generate_candidates(
                state["user_input"],
                state["observations"],
                deps.tool_schemas,
            )
            hypotheses = [
                ActionSchema.from_tool_call_candidate(c)
                for c in candidates
                if c.tool_name in deps.tool_schemas
            ]

        # Get LLM uncertainty from TTS
        llm_uncertainty = deps.tts_client.last_uncertainty or 0.5

        # Track uncertainty observation
        deps.propagator.observe(
            step_uncertainty=llm_uncertainty,
            step_type="hypothesis_generation",
            metadata={"num_hypotheses": len(hypotheses)},
        )

        # Add trace entry
        trace = list(state.get("trace", []))
        trace.append({
            "step": "generate_hypotheses",
            "num_hypotheses": len(hypotheses),
            "llm_uncertainty": llm_uncertainty,
        })

        return {
            **state,
            "hypotheses": hypotheses,
            "llm_uncertainty": llm_uncertainty,
            "status": "generating",
            "trace": trace,
        }

    # -------------------------------------------------------------------------
    # Node: analyze_uncertainty
    # -------------------------------------------------------------------------
    def analyze_uncertainty_node(state: SAGEAgentState) -> SAGEAgentState:
        """Analyze uncertainty and compute belief distribution.

        This implements the SAGE decision engine (Phase 3).
        """
        hypotheses = state["hypotheses"]

        if not hypotheses:
            return {
                **state,
                "error": "No valid hypotheses generated",
                "status": "escalated",
            }

        # Compute belief distribution
        belief_dist = deps.sage_engine.compute_belief_distribution(
            hypotheses,
            state["domains"],
        )

        # Find best action
        if belief_dist:
            best_hash = max(belief_dist, key=belief_dist.get)
            best_prob = belief_dist[best_hash]
            best_action = next(
                h for h in hypotheses if h.compute_hash() == best_hash
            )
        else:
            best_action = hypotheses[0] if hypotheses else None
            best_prob = 0.0

        # Compute structured uncertainty
        structured_unc = 1.0 - best_prob

        # Combined uncertainty
        sw = deps.sage_engine.config.structured_uncertainty_weight
        llm_unc = state.get("llm_uncertainty", 0.5)
        combined_unc = sw * structured_unc + (1 - sw) * llm_unc

        # Check if we have required unknowns (need questions before deciding)
        has_required_unknowns = False
        if best_action and best_action.tool_name in deps.tool_schemas:
            schema = deps.tool_schemas[best_action.tool_name]
            for param in schema.required:
                if best_action.arguments.get(param, UNK) == UNK:
                    has_required_unknowns = True
                    break

        # Generate questions FIRST if we have unknowns or high uncertainty
        # This ensures the decision engine has questions to select from
        questions = state.get("questions", [])
        if not questions and (has_required_unknowns or structured_unc > 0.3):
            questions = deps.question_generator.generate_questions(
                state["user_input"],
                [h.to_tool_call_candidate() for h in hypotheses],
                state["observations"],
                deps.tool_schemas,
            )

        # Make SAGE decision WITH questions
        metrics, _, clarify_request = deps.sage_engine.decide(
            hypotheses=hypotheses,
            domains=state["domains"],
            questions=questions,
            aspect_counts=state.get("aspect_counts", {}),
            clarification_rounds=state.get("clarification_rounds", 0),
            llm_uncertainty=llm_unc,
        )

        # Add trace
        trace = list(state.get("trace", []))
        trace.append({
            "step": "analyze_uncertainty",
            "belief_distribution": belief_dist,
            "structured_uncertainty": structured_unc,
            "combined_uncertainty": combined_unc,
            "decision": metrics.chosen_action,
            "risk_score": metrics.risk_score,
            "evpi_score": metrics.evpi_score,
        })

        return {
            **state,
            "belief_distribution": belief_dist,
            "best_action": best_action,
            "best_probability": best_prob,
            "structured_uncertainty": structured_unc,
            "combined_uncertainty": combined_unc,
            "accumulated_uncertainty": deps.propagator.accumulated_uncertainty,
            "questions": questions,
            "current_question": clarify_request,
            "decision_metrics": metrics,
            "status": "analyzing",
            "trace": trace,
        }

    # -------------------------------------------------------------------------
    # Router: conditional_router
    # -------------------------------------------------------------------------
    def conditional_router(state: SAGEAgentState) -> Literal["act", "clarify", "escalate"]:
        """Route based on SAGE decision."""
        metrics = state.get("decision_metrics")

        if metrics is None:
            return "escalate"

        if metrics.chosen_action == "ACT":
            return "act"
        elif metrics.chosen_action == "CLARIFY":
            return "clarify"
        else:
            return "escalate"

    # -------------------------------------------------------------------------
    # Node: execute_tool
    # -------------------------------------------------------------------------
    def execute_tool_node(state: SAGEAgentState) -> SAGEAgentState:
        """Execute the chosen tool action."""
        best_action = state.get("best_action")

        if best_action is None:
            return {
                **state,
                "error": "No action to execute",
                "status": "escalated",
            }

        # Validate against schema
        if best_action.tool_name not in deps.tool_schemas:
            return {
                **state,
                "error": f"Unknown tool: {best_action.tool_name}",
                "status": "escalated",
            }

        schema = deps.tool_schemas[best_action.tool_name]

        # Check for required unknowns
        for param in schema.required:
            if best_action.arguments.get(param, UNK) == UNK:
                return {
                    **state,
                    "error": f"Missing required parameter: {param}",
                    "status": "escalated",
                }

        # Create tool call
        tool_call = ToolCall(
            tool_name=best_action.tool_name,
            arguments=best_action.arguments,
        )

        # Execute
        exec_result = deps.tool_executor.execute(tool_call)

        # Track execution
        trace = list(state.get("trace", []))
        trace.append({
            "step": "execute_tool",
            "tool_name": tool_call.tool_name,
            "arguments": dict(tool_call.arguments),
            "success": exec_result.success,
            "error": exec_result.error,
        })

        if exec_result.success:
            return {
                **state,
                "result": tool_call,
                "execution_result": exec_result,
                "status": "done",
                "trace": trace,
            }
        else:
            return {
                **state,
                "error": exec_result.error or "Execution failed",
                "execution_result": exec_result,
                "execution_attempts": state.get("execution_attempts", 0) + 1,
                "status": "escalated",
                "trace": trace,
            }

    # -------------------------------------------------------------------------
    # Node: human_clarification
    # -------------------------------------------------------------------------
    def human_clarification_node(state: SAGEAgentState) -> SAGEAgentState:
        """Get clarification from human (interrupts graph for input)."""
        clarify_request = state.get("current_question")

        if clarify_request is None:
            # No question selected - try to generate one
            return {
                **state,
                "clarification_rounds": state.get("clarification_rounds", 0) + 1,
                "status": "clarifying",
            }

        # Convert to SAGE Question format
        from sage_agent.core.types import Aspect

        question = Question(
            text=clarify_request.question,
            aspects=tuple(
                Aspect(tool_name=t, param_name=p)
                for t, p in clarify_request.aspects
            ),
        )

        # Ask question
        response = deps.question_asker.ask(question)

        # Update observations
        observations = list(state["observations"])
        observations.append(response)

        # Update aspect counts
        aspect_counts = dict(state.get("aspect_counts", {}))
        for aspect in question.aspects:
            key = f"{aspect.tool_name}:{aspect.param_name}"
            aspect_counts[key] = aspect_counts.get(key, 0) + 1

        # Track in propagator (clarification reduces uncertainty)
        deps.propagator.observe(
            step_uncertainty=0.1,  # Low uncertainty from direct user input
            step_type="clarification",
            metadata={"question": question.text},
        )

        trace = list(state.get("trace", []))
        trace.append({
            "step": "human_clarification",
            "question": question.text,
            "response": response,
        })

        return {
            **state,
            "observations": observations,
            "aspect_counts": aspect_counts,
            "clarification_rounds": state.get("clarification_rounds", 0) + 1,
            "status": "clarifying",
            "trace": trace,
        }

    # -------------------------------------------------------------------------
    # Node: update_belief (Bayesian update)
    # -------------------------------------------------------------------------
    def update_belief_node(state: SAGEAgentState) -> SAGEAgentState:
        """Update belief state based on clarification (Bayesian update).

        This implements Phase 4's Bayesian update: instead of starting from
        scratch, we filter the previous belief_distribution based on new info.
        """
        if not state["observations"]:
            return state

        # Get the last observation (user response)
        response = state["observations"][-1]

        # Get the question that was asked
        clarify_request = state.get("current_question")
        if clarify_request is None:
            return state

        # Update domains based on response
        new_domains = {
            tool: dict(params)
            for tool, params in state["domains"].items()
        }

        for tool_name, param_name in clarify_request.aspects:
            if tool_name not in deps.tool_schemas:
                continue

            schema = deps.tool_schemas[tool_name]
            if param_name not in new_domains.get(tool_name, {}):
                continue

            current_domain = new_domains[tool_name][param_name]

            # Use constraint extractor to refine domain
            refined_domain = deps.constraint_extractor.update_domain(
                current_domain,
                response,
            )
            new_domains[tool_name][param_name] = refined_domain

            # Apply domain refiner if available
            if schema.domain_refiner is not None:
                new_domains[tool_name] = dict(
                    schema.domain_refiner.refine(schema, new_domains[tool_name], {})
                )

        # Bayesian update: filter previous hypotheses based on new constraints
        old_hypotheses = state.get("hypotheses", [])
        old_belief = state.get("belief_distribution", {})

        # Re-weight hypotheses based on new domains
        belief = BeliefState(domains=new_domains, epsilon=1e-4)
        new_belief: Dict[str, float] = {}

        for h in old_hypotheses:
            h_hash = h.compute_hash()
            old_prob = old_belief.get(h_hash, 0.0)

            if h.tool_name in deps.tool_schemas:
                schema = deps.tool_schemas[h.tool_name]
                candidate = h.to_tool_call_candidate()
                new_weight = belief.candidate_weight(candidate, schema)

                # Bayesian update: P(h|evidence) ∝ P(evidence|h) × P(h)
                # Here P(evidence|h) ≈ new_weight (how compatible is h with new domains)
                new_belief[h_hash] = old_prob * max(new_weight, 1e-10)

        # Normalize
        total = sum(new_belief.values())
        if total > 0:
            new_belief = {k: v / total for k, v in new_belief.items()}

        trace = list(state.get("trace", []))
        trace.append({
            "step": "update_belief",
            "response": response,
            "domains_updated": list(
                f"{t}:{p}" for t, p in clarify_request.aspects
            ),
        })

        return {
            **state,
            "domains": new_domains,
            "belief_distribution": new_belief,
            "current_question": None,  # Clear after update
            "trace": trace,
        }

    # -------------------------------------------------------------------------
    # Node: escalate
    # -------------------------------------------------------------------------
    def escalate_node(state: SAGEAgentState) -> SAGEAgentState:
        """Escalate to human when agent cannot resolve."""
        # Provide detailed trace for debugging
        unc_breakdown = deps.propagator.get_uncertainty_breakdown()

        error_msg = state.get("error") or "Escalated due to high uncertainty"
        if unc_breakdown:
            error_msg += "\n\nUncertainty breakdown:\n"
            for step_type, unc in unc_breakdown.items():
                error_msg += f"  {step_type}: {unc:.3f}\n"

        trace = list(state.get("trace", []))
        trace.append({
            "step": "escalate",
            "reason": state.get("error") or "high_uncertainty",
            "accumulated_uncertainty": deps.propagator.accumulated_uncertainty,
            "clarification_rounds": state.get("clarification_rounds", 0),
        })

        return {
            **state,
            "error": error_msg,
            "status": "escalated",
            "trace": trace,
        }

    # -------------------------------------------------------------------------
    # Build graph topology
    # -------------------------------------------------------------------------

    graph.add_node("generate_hypotheses", generate_hypotheses_node)
    graph.add_node("analyze_uncertainty", analyze_uncertainty_node)
    graph.add_node("execute_tool", execute_tool_node)
    graph.add_node("human_clarification", human_clarification_node)
    graph.add_node("update_belief", update_belief_node)
    graph.add_node("escalate", escalate_node)

    # Entry point
    graph.set_entry_point("generate_hypotheses")

    # Edges
    graph.add_edge("generate_hypotheses", "analyze_uncertainty")

    graph.add_conditional_edges(
        "analyze_uncertainty",
        conditional_router,
        {
            "act": "execute_tool",
            "clarify": "human_clarification",
            "escalate": "escalate",
        },
    )

    graph.add_edge("human_clarification", "update_belief")
    graph.add_edge("update_belief", "generate_hypotheses")

    graph.add_edge("execute_tool", END)
    graph.add_edge("escalate", END)

    return graph


# =============================================================================
# Phase 5: Agent Factory & Runner
# =============================================================================


class InteractiveQuestionAsker:
    """Default question asker that prompts via console."""

    def ask(self, question: Question) -> str:
        print(f"\n[AGENT] {question.text}")
        return input("[USER] ").strip()


def create_initial_state(
    user_input: str,
    tool_schemas: Dict[str, ToolSchema],
    num_samples: int = 10,
) -> SAGEAgentState:
    """Create initial state for the agent."""
    return {
        "user_input": user_input,
        "messages": [],
        "observations": [],
        "hypotheses": [],
        "num_samples": num_samples,
        "belief_distribution": {},
        "domains": {
            name: dict(schema.parameters)
            for name, schema in tool_schemas.items()
        },
        "best_action": None,
        "best_probability": 0.0,
        "structured_uncertainty": 1.0,
        "llm_uncertainty": 0.5,
        "combined_uncertainty": 1.0,
        "accumulated_uncertainty": 0.0,
        "questions": [],
        "current_question": None,
        "aspect_counts": {},
        "decision_metrics": None,
        "clarification_rounds": 0,
        "execution_attempts": 0,
        "status": "pending",
        "result": None,
        "execution_result": None,
        "error": None,
        "trace": [],
    }


def create_agent(
    tool_schemas: Dict[str, ToolSchema],
    tts_config: Optional[TTSConfig] = None,
    sage_config: Optional[SAGEConfig] = None,
    question_asker: Any = None,
    tool_registry: Optional[Dict[str, Callable]] = None,
) -> Tuple[StateGraph, GraphDependencies]:
    """Create a configured SGR-SAGE-UQ agent.

    Args:
        tool_schemas: Available tool definitions
        tts_config: TTS service configuration
        sage_config: SAGE decision engine configuration
        question_asker: Custom question asker (default: InteractiveQuestionAsker)
        tool_registry: Mapping of tool name → execution function

    Returns:
        Tuple of (compiled graph, dependencies)
    """
    # Use defaults if not provided
    tts_config = tts_config or TTSConfig()
    sage_config = sage_config or SAGEConfig()

    # Create TTS client
    tts_client = SelfConsistencyClient(tts_config)

    # Create TTSLLMClient for candidate/question generators
    llm = TTSLLMClient(
        base_url=tts_config.service_url,
        model=tts_config.model,
        tts_budget=tts_config.tts_budget,
        temperature=tts_config.temperature,
        max_tokens=tts_config.max_tokens,
        timeout=tts_config.timeout,
    )

    # Create uncertainty propagator
    propagator = create_sage_propagator(
        structured_weight=sage_config.structured_uncertainty_weight,
        llm_weight=1.0 - sage_config.structured_uncertainty_weight,
    )

    # Create SAGE decision engine
    sage_engine = SAGEDecisionEngine(
        config=sage_config,
        tool_schemas=tool_schemas,
        propagator=propagator,
    )

    # Create constraint extractor
    constraint_extractor = HybridConstraintExtractor(
        llm=llm,
        ambiguity_threshold=0.5,
    )

    # Create tool executor
    if tool_registry is None:
        tool_registry = {}
    tool_executor = ToolRegistryExecutor(tool_registry)

    # Create dependencies
    deps = GraphDependencies(
        tool_schemas=tool_schemas,
        tts_client=tts_client,
        sage_engine=sage_engine,
        candidate_generator=LLMBackedCandidateGenerator(llm),
        question_generator=LLMBackedQuestionGenerator(llm),
        question_asker=question_asker or InteractiveQuestionAsker(),
        tool_executor=tool_executor,
        constraint_extractor=constraint_extractor,
        config=SageAgentConfig(
            max_questions=sage_config.max_clarification_rounds,
            tau_execute=sage_config.min_confidence_to_act,
            alpha=0.1,
        ),
        propagator=propagator,
    )

    # Build and return graph
    graph = build_sgr_sage_uq_graph(deps)

    return graph, deps


def run_agent(
    graph: StateGraph,
    user_input: str,
    tool_schemas: Dict[str, ToolSchema],
    num_samples: int = 10,
    recursion_limit: int = 50,
) -> SAGEAgentState:
    """Run the SGR-SAGE-UQ agent.

    Args:
        graph: Compiled LangGraph
        user_input: User's request
        tool_schemas: Available tools
        num_samples: Number of self-consistency samples
        recursion_limit: Maximum graph iterations

    Returns:
        Final agent state
    """
    compiled = graph.compile()

    initial_state = create_initial_state(
        user_input=user_input,
        tool_schemas=tool_schemas,
        num_samples=num_samples,
    )

    result = compiled.invoke(initial_state, {"recursion_limit": recursion_limit})

    return result


def print_result(result: SAGEAgentState) -> None:
    """Pretty-print agent result."""
    print("\n" + "=" * 80)
    print("SGR-SAGE-UQ Agent Result")
    print("=" * 80)

    print(f"\nStatus: {result['status']}")

    if result['result']:
        print(f"\nTool Call: {result['result'].tool_name}")
        print(f"Arguments: {dict(result['result'].arguments)}")

    if result['error']:
        print(f"\nError: {result['error']}")

    print(f"\nMetrics:")
    print(f"  Clarification rounds: {result['clarification_rounds']}")
    print(f"  Combined uncertainty: {result['combined_uncertainty']:.3f}")
    print(f"  Accumulated uncertainty: {result['accumulated_uncertainty']:.3f}")

    if result.get('decision_metrics'):
        m = result['decision_metrics']
        print(f"\nSAGE Decision Metrics:")
        print(f"  Risk score: {m.risk_score:.3f}")
        print(f"  EVPI score: {m.evpi_score:.3f}")
        print(f"  Cost of clarification: {m.cost_of_clarification:.3f}")
        print(f"  Penalty of error: {m.penalty_of_error:.3f}")
        print(f"  Decision: {m.chosen_action}")

    if result.get('trace'):
        print(f"\nTrace ({len(result['trace'])} steps):")
        for i, step in enumerate(result['trace']):
            print(f"  {i+1}. {step.get('step', 'unknown')}")

    print("=" * 80)


# =============================================================================
# Demo / Main
# =============================================================================


def main():
    """Demo the SGR-SAGE-UQ agent."""

    # Define sample tool
    book_flight_tool = ToolSchema(
        name="book_flight",
        parameters={
            "origin": ParameterDomain.from_values(["NYC", "BOS", "LAX", "SFO", "ORD"]),
            "destination": ParameterDomain.from_values(["NYC", "BOS", "LAX", "SFO", "ORD"]),
            "date": ParameterDomain.from_values([
                "2024-03-01", "2024-03-02", "2024-03-03",
                "2024-03-04", "2024-03-05",
            ]),
            "class": ParameterDomain.from_values(["economy", "business", "first"]),
        },
        required=frozenset({"origin", "destination", "date"}),
    )

    tool_schemas = {book_flight_tool.name: book_flight_tool}

    # Create tool registry (mock execution)
    def execute_book_flight(args: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "confirmation": "FL-12345",
            "details": dict(args),
            "status": "booked",
        }

    tool_registry = {
        "book_flight": execute_book_flight,
    }

    # Create agent
    tts_config = TTSConfig(
        service_url=os.getenv("TTS_SERVICE_URL", "http://localhost:8001/v1"),
        model=os.getenv("TTS_MODEL", "openai/gpt-4o-mini"),
        tts_budget=8,
    )

    sage_config = SAGEConfig(
        cost_of_clarification=0.15,
        base_penalty_of_error=1.0,
        min_confidence_to_act=0.7,
        max_clarification_rounds=4,
    )

    graph, _deps = create_agent(
        tool_schemas=tool_schemas,
        tts_config=tts_config,
        sage_config=sage_config,
        tool_registry=tool_registry,
    )

    # Run with sample input
    user_input = "I need to fly from New York to Los Angeles."

    print(f"\nUser: {user_input}")
    print("-" * 40)

    result = run_agent(
        graph=graph,
        user_input=user_input,
        tool_schemas=tool_schemas,
        num_samples=8,
    )

    print_result(result)


if __name__ == "__main__":
    main()
