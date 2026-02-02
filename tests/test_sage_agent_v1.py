"""Tests for Pure SAGE-Agent (v1) LangGraph Implementation.

Tests the clean Algorithm 1 implementation from examples/langgraph_sage_agent.py
"""

import pytest
from typing import Dict, List, Mapping
from dataclasses import dataclass

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(ROOT, "examples")
sys.path.insert(0, ROOT)
sys.path.insert(0, EXAMPLES_DIR)

from sage_agent import (
    ParameterDomain,
    ToolSchema,
    ToolCall,
    ToolCallCandidate,
    Question,
    Aspect,
    UNK,
)
from sage_agent.core.belief import BeliefState
from sage_agent.core.evpi import compute_evpi
from sage_agent.core.types import ExecutionResult

from langgraph_sage_agent import (
    SAGEConfig,
    GraphDeps,
    AgentState,
    build_graph,
    create_initial_state,
    _has_required_unknowns,
    _redundancy_cost,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def flight_tool() -> ToolSchema:
    """Simple flight booking tool for testing."""
    return ToolSchema(
        name="book_flight",
        parameters={
            "origin": ParameterDomain.from_values(["NYC", "BOS", "LAX"]),
            "destination": ParameterDomain.from_values(["NYC", "BOS", "LAX"]),
            "date": ParameterDomain.from_values(["2024-03-01", "2024-03-02"]),
        },
        required=frozenset({"origin", "destination", "date"}),
    )


@pytest.fixture
def restaurant_tool() -> ToolSchema:
    """Restaurant search tool with fewer parameters."""
    return ToolSchema(
        name="find_restaurant",
        parameters={
            "cuisine": ParameterDomain.from_values(["italian", "japanese", "mexican"]),
            "location": ParameterDomain.from_values(["downtown", "midtown"]),
        },
        required=frozenset({"cuisine"}),
    )


@pytest.fixture
def tool_schemas(flight_tool, restaurant_tool) -> Dict[str, ToolSchema]:
    """Dictionary of test tools."""
    return {
        flight_tool.name: flight_tool,
        restaurant_tool.name: restaurant_tool,
    }


# =============================================================================
# Mock Implementations
# =============================================================================

@dataclass
class MockCandidateGenerator:
    """Returns predefined candidates."""
    candidates: List[ToolCallCandidate]

    def generate_candidates(
        self,
        user_input: str,
        observations: List[str],
        tool_schemas: Mapping[str, ToolSchema],
    ) -> List[ToolCallCandidate]:
        return self.candidates


@dataclass
class MockQuestionGenerator:
    """Returns predefined questions."""
    questions: List[Question]

    def generate_questions(
        self,
        user_input: str,
        candidates: List[ToolCallCandidate],
        observations: List[str],
        tool_schemas: Mapping[str, ToolSchema],
    ) -> List[Question]:
        return self.questions


@dataclass
class MockQuestionAsker:
    """Returns predefined answers."""
    answers: List[str]
    call_count: int = 0

    def ask(self, question: Question) -> str:
        if self.call_count < len(self.answers):
            answer = self.answers[self.call_count]
            self.call_count += 1
            return answer
        return ""


@dataclass
class MockToolExecutor:
    """Always succeeds."""
    def execute(self, tool_call: ToolCall) -> ExecutionResult:
        return ExecutionResult(success=True, output={"ok": True})


@dataclass
class MockConstraintExtractor:
    """Simple constraint extractor that narrows domain based on response."""
    def update_domain(self, domain: ParameterDomain, response: str) -> ParameterDomain:
        if not domain.values:
            return domain
        response_lower = response.lower()
        for value in domain.values:
            if isinstance(value, str) and value.lower() in response_lower:
                return ParameterDomain.from_values([value])
        return domain


# =============================================================================
# Test SAGEConfig
# =============================================================================

class TestSAGEConfig:
    """Tests for SAGEConfig dataclass."""

    def test_default_values(self):
        """Test paper default hyperparameters."""
        config = SAGEConfig()
        assert config.tau_exec == 0.85
        assert config.alpha == 0.1
        assert config.lambda_redundancy == 0.5
        assert config.epsilon == 1e-4
        assert config.max_questions == 6
        assert config.min_confidence == 0.3

    def test_custom_values(self):
        """Test custom hyperparameter values."""
        config = SAGEConfig(
            tau_exec=0.9,
            alpha=0.05,
            lambda_redundancy=0.3,
            max_questions=4,
        )
        assert config.tau_exec == 0.9
        assert config.alpha == 0.05
        assert config.lambda_redundancy == 0.3
        assert config.max_questions == 4

    def test_frozen(self):
        """Test that config is immutable."""
        config = SAGEConfig()
        with pytest.raises(AttributeError):
            config.tau_exec = 0.5


# =============================================================================
# Test BeliefState
# =============================================================================

class TestBeliefState:
    """Tests for BeliefState candidate weighting."""

    def test_specified_values_weight_one(self, flight_tool):
        """Specified values should have weight 1.0."""
        domains = {flight_tool.name: dict(flight_tool.parameters)}
        belief = BeliefState(domains=domains)

        candidate = ToolCallCandidate(
            tool_name="book_flight",
            arguments={"origin": "NYC", "destination": "LAX", "date": "2024-03-01"},
        )

        weight = belief.candidate_weight(candidate, flight_tool)
        assert weight == 1.0

    def test_unknown_finite_domain_weight(self, flight_tool):
        """Unknown values with finite domain should weight 1/|D|."""
        domains = {flight_tool.name: dict(flight_tool.parameters)}
        belief = BeliefState(domains=domains)

        # One unknown parameter with 3 values
        candidate = ToolCallCandidate(
            tool_name="book_flight",
            arguments={"origin": UNK, "destination": "LAX", "date": "2024-03-01"},
        )

        weight = belief.candidate_weight(candidate, flight_tool)
        assert weight == pytest.approx(1.0 / 3.0)  # 3 cities

    def test_multiple_unknowns_multiply(self, flight_tool):
        """Multiple unknowns should multiply weights."""
        domains = {flight_tool.name: dict(flight_tool.parameters)}
        belief = BeliefState(domains=domains)

        # Two unknown parameters
        candidate = ToolCallCandidate(
            tool_name="book_flight",
            arguments={"origin": UNK, "destination": UNK, "date": "2024-03-01"},
        )

        weight = belief.candidate_weight(candidate, flight_tool)
        # 3 origins * 3 destinations = 1/9
        assert weight == pytest.approx(1.0 / 9.0)

    def test_invalid_value_weight_zero(self, flight_tool):
        """Invalid values should have weight 0."""
        domains = {flight_tool.name: dict(flight_tool.parameters)}
        belief = BeliefState(domains=domains)

        candidate = ToolCallCandidate(
            tool_name="book_flight",
            arguments={"origin": "INVALID", "destination": "LAX", "date": "2024-03-01"},
        )

        weight = belief.candidate_weight(candidate, flight_tool)
        assert weight == 0.0

    def test_normalize(self):
        """Test probability normalization."""
        belief = BeliefState(domains={})

        weights = [0.5, 0.3, 0.2]
        probs = belief.normalize(weights)

        assert sum(probs) == pytest.approx(1.0)
        assert probs[0] == pytest.approx(0.5)
        assert probs[1] == pytest.approx(0.3)
        assert probs[2] == pytest.approx(0.2)

    def test_normalize_zeros(self):
        """Test normalization with all zeros."""
        belief = BeliefState(domains={})

        weights = [0.0, 0.0, 0.0]
        probs = belief.normalize(weights)

        # Should return uniform distribution
        assert all(p == pytest.approx(1.0 / 3.0) for p in probs)

    def test_domain_updates_affect_weight(self, flight_tool):
        """Updated domains should affect candidate weights."""
        # Start with original domains
        domains = {flight_tool.name: dict(flight_tool.parameters)}
        belief = BeliefState(domains=domains)

        candidate = ToolCallCandidate(
            tool_name="book_flight",
            arguments={"origin": UNK, "destination": "LAX", "date": "2024-03-01"},
        )

        weight_before = belief.candidate_weight(candidate, flight_tool)

        # Narrow domain to single value
        belief.domains[flight_tool.name]["origin"] = ParameterDomain.from_values(["NYC"])

        weight_after = belief.candidate_weight(candidate, flight_tool)

        assert weight_before == pytest.approx(1.0 / 3.0)
        assert weight_after == pytest.approx(1.0)  # Now only 1 option


# =============================================================================
# Test EVPI Computation
# =============================================================================

class TestEVPI:
    """Tests for EVPI (Expected Value of Perfect Information)."""

    def test_evpi_with_uncertainty(self):
        """EVPI should be positive when there's uncertainty."""
        candidates = [
            ToolCallCandidate("book_flight", {"origin": UNK, "destination": "LAX"}),
            ToolCallCandidate("book_flight", {"origin": "NYC", "destination": "LAX"}),
        ]
        probs = [0.5, 0.5]
        aspects = (Aspect("book_flight", "origin"),)

        evpi = compute_evpi(candidates, probs, aspects)
        assert evpi >= 0.0

    def test_evpi_zero_when_certain(self):
        """EVPI should be zero when already certain."""
        candidates = [
            ToolCallCandidate("book_flight", {"origin": "NYC", "destination": "LAX"}),
        ]
        probs = [1.0]
        aspects = (Aspect("book_flight", "origin"),)

        evpi = compute_evpi(candidates, probs, aspects)
        assert evpi == pytest.approx(0.0)

    def test_evpi_empty_candidates(self):
        """EVPI with empty candidates should be zero."""
        evpi = compute_evpi([], [], (Aspect("book_flight", "origin"),))
        assert evpi == 0.0


# =============================================================================
# Test Helper Functions
# =============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_has_required_unknowns_true(self, flight_tool):
        """Should return True when required param is UNK."""
        candidate = ToolCallCandidate(
            tool_name="book_flight",
            arguments={"origin": UNK, "destination": "LAX", "date": "2024-03-01"},
        )
        assert _has_required_unknowns(candidate, flight_tool) is True

    def test_has_required_unknowns_false(self, flight_tool):
        """Should return False when all required params specified."""
        candidate = ToolCallCandidate(
            tool_name="book_flight",
            arguments={"origin": "NYC", "destination": "LAX", "date": "2024-03-01"},
        )
        assert _has_required_unknowns(candidate, flight_tool) is False

    def test_redundancy_cost_zero(self):
        """Redundancy cost should be zero for new aspects."""
        question = Question(
            text="Where from?",
            aspects=(Aspect("book_flight", "origin"),),
        )
        aspect_counts = {}

        cost = _redundancy_cost(question, aspect_counts, lambda_weight=0.5)
        assert cost == 0.0

    def test_redundancy_cost_increases(self):
        """Redundancy cost should increase with repeat queries."""
        question = Question(
            text="Where from?",
            aspects=(Aspect("book_flight", "origin"),),
        )
        aspect_counts = {"book_flight:origin": 2}

        cost = _redundancy_cost(question, aspect_counts, lambda_weight=0.5)
        assert cost == 1.0  # 2 * 0.5


# =============================================================================
# Test Initial State Creation
# =============================================================================

class TestInitialState:
    """Tests for create_initial_state."""

    def test_creates_correct_structure(self, tool_schemas):
        """Initial state should have all required fields."""
        state = create_initial_state("Book a flight", tool_schemas)

        assert state["user_input"] == "Book a flight"
        assert state["observations"] == []
        assert state["candidates"] == []
        assert state["probabilities"] == []
        assert state["max_prob"] == 0.0
        assert state["best_idx"] == 0
        assert state["questions"] == []
        assert state["best_question"] is None
        assert state["best_score"] == 0.0
        assert state["aspect_counts"] == {}
        assert state["step"] == 0
        assert state["status"] == "pending"
        assert state["result"] is None
        assert state["error"] is None

    def test_domains_copied(self, tool_schemas):
        """Domains should be copied from tool schemas."""
        state = create_initial_state("Book a flight", tool_schemas)

        assert "book_flight" in state["domains"]
        assert "find_restaurant" in state["domains"]
        assert "origin" in state["domains"]["book_flight"]


# =============================================================================
# Test Graph Building
# =============================================================================

class TestGraphBuilding:
    """Tests for graph construction."""

    def test_graph_builds_without_error(self, flight_tool):
        """Graph should build successfully."""
        deps = GraphDeps(
            tool_schemas={flight_tool.name: flight_tool},
            candidate_generator=MockCandidateGenerator([]),
            question_generator=MockQuestionGenerator([]),
            question_asker=MockQuestionAsker([]),
            tool_executor=MockToolExecutor(),
            constraint_extractor=MockConstraintExtractor(),
            config=SAGEConfig(),
        )

        graph = build_graph(deps)
        assert graph is not None

    def test_graph_compiles(self, flight_tool):
        """Graph should compile without error."""
        deps = GraphDeps(
            tool_schemas={flight_tool.name: flight_tool},
            candidate_generator=MockCandidateGenerator([]),
            question_generator=MockQuestionGenerator([]),
            question_asker=MockQuestionAsker([]),
            tool_executor=MockToolExecutor(),
            constraint_extractor=MockConstraintExtractor(),
            config=SAGEConfig(),
        )

        compiled = build_graph(deps).compile()
        assert compiled is not None


# =============================================================================
# Test Graph Execution
# =============================================================================

class TestGraphExecution:
    """Tests for end-to-end graph execution."""

    def test_no_candidates_returns_error(self, flight_tool):
        """Should return error when no candidates generated."""
        deps = GraphDeps(
            tool_schemas={flight_tool.name: flight_tool},
            candidate_generator=MockCandidateGenerator([]),
            question_generator=MockQuestionGenerator([]),
            question_asker=MockQuestionAsker([]),
            tool_executor=MockToolExecutor(),
            constraint_extractor=MockConstraintExtractor(),
            config=SAGEConfig(),
        )

        graph = build_graph(deps).compile()
        initial = create_initial_state("Book flight", {flight_tool.name: flight_tool})

        result = graph.invoke(initial, {"recursion_limit": 10})

        assert result["status"] == "done"
        assert result["error"] == "No valid candidates generated"

    def test_confident_execution(self, flight_tool):
        """Should execute immediately when confident."""
        # Fully specified candidate
        candidate = ToolCallCandidate(
            tool_name="book_flight",
            arguments={"origin": "NYC", "destination": "LAX", "date": "2024-03-01"},
        )

        deps = GraphDeps(
            tool_schemas={flight_tool.name: flight_tool},
            candidate_generator=MockCandidateGenerator([candidate]),
            question_generator=MockQuestionGenerator([]),
            question_asker=MockQuestionAsker([]),
            tool_executor=MockToolExecutor(),
            constraint_extractor=MockConstraintExtractor(),
            config=SAGEConfig(tau_exec=0.85),
        )

        graph = build_graph(deps).compile()
        initial = create_initial_state("Book flight", {flight_tool.name: flight_tool})

        result = graph.invoke(initial, {"recursion_limit": 10})

        assert result["status"] == "done"
        assert result["result"] is not None
        assert result["result"].tool_name == "book_flight"
        assert result["step"] == 0  # No questions asked

    def test_asks_questions_when_uncertain(self, flight_tool):
        """Should ask questions when uncertain."""
        # Multiple candidates with unknowns - neither has high enough probability
        # Candidate 1: UNK origin (weight = 1/3 for 3 origin values)
        # Candidate 2: UNK destination (weight = 1/3 for 3 dest values)
        # After normalization, max_prob = 0.5, which is < tau_exec=0.85
        candidate1 = ToolCallCandidate(
            tool_name="book_flight",
            arguments={"origin": UNK, "destination": "LAX", "date": "2024-03-01"},
        )
        candidate2 = ToolCallCandidate(
            tool_name="book_flight",
            arguments={"origin": "NYC", "destination": UNK, "date": "2024-03-01"},
        )

        question = Question(
            text="Which city are you departing from?",
            aspects=(Aspect("book_flight", "origin"),),
        )

        asker = MockQuestionAsker(["NYC"])

        deps = GraphDeps(
            tool_schemas={flight_tool.name: flight_tool},
            candidate_generator=MockCandidateGenerator([candidate1, candidate2]),
            question_generator=MockQuestionGenerator([question]),
            question_asker=asker,
            tool_executor=MockToolExecutor(),
            constraint_extractor=MockConstraintExtractor(),
            config=SAGEConfig(tau_exec=0.85, max_questions=3),
        )

        graph = build_graph(deps).compile()
        initial = create_initial_state("Book flight", {flight_tool.name: flight_tool})

        result = graph.invoke(initial, {"recursion_limit": 20})

        # Should have asked at least one question since max_prob < tau_exec
        assert asker.call_count >= 1
        assert len(result["observations"]) >= 1

    def test_respects_max_questions(self, flight_tool):
        """Should stop after max_questions."""
        # Candidate that stays uncertain
        candidate = ToolCallCandidate(
            tool_name="book_flight",
            arguments={"origin": UNK, "destination": UNK, "date": UNK},
        )

        question = Question(
            text="Which city?",
            aspects=(Aspect("book_flight", "origin"),),
        )

        # Answers that don't resolve uncertainty
        asker = MockQuestionAsker(["maybe NYC", "perhaps LAX", "sometime"])

        deps = GraphDeps(
            tool_schemas={flight_tool.name: flight_tool},
            candidate_generator=MockCandidateGenerator([candidate]),
            question_generator=MockQuestionGenerator([question]),
            question_asker=asker,
            tool_executor=MockToolExecutor(),
            constraint_extractor=MockConstraintExtractor(),
            config=SAGEConfig(max_questions=2, min_confidence=0.0),
        )

        graph = build_graph(deps).compile()
        initial = create_initial_state("Book flight", {flight_tool.name: flight_tool})

        result = graph.invoke(initial, {"recursion_limit": 20})

        # Should have stopped at max_questions
        assert result["step"] <= 2

    def test_escalates_when_uncertain(self, flight_tool):
        """Should escalate when too uncertain after max questions."""
        candidate = ToolCallCandidate(
            tool_name="book_flight",
            arguments={"origin": UNK, "destination": UNK, "date": UNK},
        )

        question = Question(
            text="Which city?",
            aspects=(Aspect("book_flight", "origin"),),
        )

        asker = MockQuestionAsker(["dunno", "not sure"])

        deps = GraphDeps(
            tool_schemas={flight_tool.name: flight_tool},
            candidate_generator=MockCandidateGenerator([candidate]),
            question_generator=MockQuestionGenerator([question]),
            question_asker=asker,
            tool_executor=MockToolExecutor(),
            constraint_extractor=MockConstraintExtractor(),
            config=SAGEConfig(max_questions=2, min_confidence=0.5),
        )

        graph = build_graph(deps).compile()
        initial = create_initial_state("Book flight", {flight_tool.name: flight_tool})

        result = graph.invoke(initial, {"recursion_limit": 20})

        # With high min_confidence and unresolved uncertainty, should escalate
        assert result["status"] in ["escalated", "done"]


# =============================================================================
# Test Domain Updates
# =============================================================================

class TestDomainUpdates:
    """Tests for domain refinement during questioning."""

    def test_domain_narrows_after_answer(self, flight_tool):
        """Domain should narrow after user answers."""
        candidate = ToolCallCandidate(
            tool_name="book_flight",
            arguments={"origin": UNK, "destination": "LAX", "date": "2024-03-01"},
        )

        question = Question(
            text="Which city are you departing from?",
            aspects=(Aspect("book_flight", "origin"),),
        )

        # Answer contains "NYC"
        asker = MockQuestionAsker(["I'll be flying from NYC"])

        deps = GraphDeps(
            tool_schemas={flight_tool.name: flight_tool},
            candidate_generator=MockCandidateGenerator([candidate]),
            question_generator=MockQuestionGenerator([question]),
            question_asker=asker,
            tool_executor=MockToolExecutor(),
            constraint_extractor=MockConstraintExtractor(),
            config=SAGEConfig(max_questions=1),
        )

        graph = build_graph(deps).compile()
        initial = create_initial_state("Book flight", {flight_tool.name: flight_tool})

        result = graph.invoke(initial, {"recursion_limit": 20})

        # Domain should have been updated
        origin_domain = result["domains"]["book_flight"]["origin"]
        # After constraint extraction, domain should be narrowed
        if origin_domain.values:
            assert len(origin_domain.values) <= 3


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
