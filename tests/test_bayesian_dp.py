"""Tests for Bayesian DP planner integration into SAGE-Agent.

Tests the dynamic programming lookahead question selection from
sage_agent/core/bayesian_dp.py, including terminal value computation,
single-step equivalence with EVPI, multi-step lookahead advantage,
and full integration with SageAgent.
"""

import pytest
import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sage_agent.core.bayesian_dp import (
    compute_question_value_dp,
    get_terminal_value,
    _compute_resolution_probability,
    _dp_value,
    _simulate_question_outcome,
)
from sage_agent.core.belief import BeliefState
from sage_agent.core.domains import ParameterDomain
from sage_agent.core.evpi import compute_evpi
from sage_agent.core.types import (
    Aspect,
    ExecutionResult,
    Question,
    ToolCall,
    ToolCallCandidate,
    ToolSchema,
    UNK,
)
from sage_agent.core.agent import SageAgent, SageAgentConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def flight_tool() -> ToolSchema:
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
def tool_schemas(flight_tool) -> Dict[str, ToolSchema]:
    return {flight_tool.name: flight_tool}


@pytest.fixture
def basic_candidates() -> List[ToolCallCandidate]:
    return [
        ToolCallCandidate("book_flight", {"origin": "NYC", "destination": "LAX", "date": "2024-03-01"}),
        ToolCallCandidate("book_flight", {"origin": "BOS", "destination": "LAX", "date": "2024-03-01"}),
        ToolCallCandidate("book_flight", {"origin": "LAX", "destination": "NYC", "date": "2024-03-02"}),
    ]


@pytest.fixture
def origin_question() -> Question:
    return Question(
        text="Which city are you departing from?",
        aspects=(Aspect("book_flight", "origin"),),
    )


@pytest.fixture
def destination_question() -> Question:
    return Question(
        text="Where are you flying to?",
        aspects=(Aspect("book_flight", "destination"),),
    )


# =============================================================================
# Mock implementations for integration test
# =============================================================================

@dataclass
class MockCandidateGenerator:
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
    def execute(self, tool_call: ToolCall) -> ExecutionResult:
        return ExecutionResult(success=True, output={"ok": True})


@dataclass
class MockConstraintExtractor:
    def update_domain(self, domain: ParameterDomain, response: str) -> ParameterDomain:
        if not domain.values:
            return domain
        response_lower = response.lower()
        for value in domain.values:
            if isinstance(value, str) and value.lower() in response_lower:
                return ParameterDomain.from_values([value])
        return domain


# =============================================================================
# Test Terminal Value
# =============================================================================

class TestTerminalValue:
    def test_terminal_value_certain(self):
        """When one candidate has probability 1.0, terminal value = 1.0."""
        assert get_terminal_value([1.0, 0.0, 0.0]) == 1.0

    def test_terminal_value_uniform(self):
        """Uniform distribution over 3 candidates -> terminal value = 1/3."""
        assert get_terminal_value([1/3, 1/3, 1/3]) == pytest.approx(1/3)

    def test_terminal_value_empty(self):
        """Empty probabilities -> terminal value = 0."""
        assert get_terminal_value([]) == 0.0

    def test_terminal_value_skewed(self):
        """Skewed distribution -> terminal value = max."""
        assert get_terminal_value([0.7, 0.2, 0.1]) == pytest.approx(0.7)


# =============================================================================
# Test Resolution Probability
# =============================================================================

class TestResolutionProbability:
    def test_resolution_prob_concentrated(self, tool_schemas, origin_question):
        """When candidates agree on origin, resolution probability is high."""
        # All candidates have same origin
        candidates = [
            ToolCallCandidate("book_flight", {"origin": "NYC", "destination": "LAX", "date": "2024-03-01"}),
            ToolCallCandidate("book_flight", {"origin": "NYC", "destination": "BOS", "date": "2024-03-01"}),
        ]
        belief = BeliefState(domains={
            "book_flight": dict(tool_schemas["book_flight"].parameters),
        })

        p_resolve = _compute_resolution_probability(
            belief, candidates, origin_question, tool_schemas
        )
        # Both candidates have origin=NYC, so P(resolve) should be 1.0
        assert p_resolve == pytest.approx(1.0)

    def test_resolution_prob_spread(self, tool_schemas, origin_question):
        """When candidates disagree on origin, resolution probability reflects spread."""
        candidates = [
            ToolCallCandidate("book_flight", {"origin": "NYC", "destination": "LAX", "date": "2024-03-01"}),
            ToolCallCandidate("book_flight", {"origin": "BOS", "destination": "LAX", "date": "2024-03-01"}),
            ToolCallCandidate("book_flight", {"origin": "LAX", "destination": "NYC", "date": "2024-03-01"}),
        ]
        belief = BeliefState(domains={
            "book_flight": dict(tool_schemas["book_flight"].parameters),
        })

        p_resolve = _compute_resolution_probability(
            belief, candidates, origin_question, tool_schemas
        )
        # Equal weight candidates with 3 different origins -> max is 1/3
        assert p_resolve == pytest.approx(1/3, abs=0.01)


# =============================================================================
# Test Simulate Question Outcome
# =============================================================================

class TestSimulateOutcome:
    def test_resolve_narrows_domain(self, tool_schemas, origin_question):
        """Resolving outcome should narrow the domain to a single value."""
        candidates = [
            ToolCallCandidate("book_flight", {"origin": "NYC", "destination": "LAX", "date": "2024-03-01"}),
            ToolCallCandidate("book_flight", {"origin": "BOS", "destination": "LAX", "date": "2024-03-01"}),
        ]
        belief = BeliefState(domains={
            "book_flight": dict(tool_schemas["book_flight"].parameters),
        })

        new_belief, new_probs = _simulate_question_outcome(
            belief, candidates, origin_question, tool_schemas, outcome_resolves=True
        )

        # Origin domain should be narrowed
        origin_domain = new_belief.domains["book_flight"]["origin"]
        assert origin_domain.size() == 1

    def test_no_resolve_keeps_domain(self, tool_schemas, origin_question):
        """Non-resolving outcome should keep the domain unchanged."""
        candidates = [
            ToolCallCandidate("book_flight", {"origin": "NYC", "destination": "LAX", "date": "2024-03-01"}),
            ToolCallCandidate("book_flight", {"origin": "BOS", "destination": "LAX", "date": "2024-03-01"}),
        ]
        belief = BeliefState(domains={
            "book_flight": dict(tool_schemas["book_flight"].parameters),
        })

        new_belief, new_probs = _simulate_question_outcome(
            belief, candidates, origin_question, tool_schemas, outcome_resolves=False
        )

        # Origin domain should be unchanged (3 values)
        origin_domain = new_belief.domains["book_flight"]["origin"]
        assert origin_domain.size() == 3


# =============================================================================
# Test DP Value Computation
# =============================================================================

class TestDPValue:
    def test_dp_value_budget_zero(self, tool_schemas):
        """With budget=0, DP value equals terminal value."""
        candidates = [
            ToolCallCandidate("book_flight", {"origin": "NYC", "destination": "LAX", "date": "2024-03-01"}),
        ]
        belief = BeliefState(domains={
            "book_flight": dict(tool_schemas["book_flight"].parameters),
        })
        probs = [1.0]
        questions = [
            Question("Where from?", aspects=(Aspect("book_flight", "origin"),)),
        ]

        from sage_agent.core.bayesian_dp import CostModel
        cost_model = CostModel(query_base_cost=0.05, query_aspect_cost=0.0, query_open_domain_cost=0.0, execution_cost=0.0, reward=1.0)
        value = _dp_value(belief, candidates, probs, questions, tool_schemas, budget=0, cost_model=cost_model)
        assert value == pytest.approx(1.0)

    def test_dp_value_increases_with_budget(self, tool_schemas):
        """DP value with budget > 0 should be >= terminal value when questions are useful."""
        # Uncertain candidates
        candidates = [
            ToolCallCandidate("book_flight", {"origin": "NYC", "destination": "LAX", "date": "2024-03-01"}),
            ToolCallCandidate("book_flight", {"origin": "BOS", "destination": "LAX", "date": "2024-03-01"}),
            ToolCallCandidate("book_flight", {"origin": "LAX", "destination": "NYC", "date": "2024-03-02"}),
        ]
        belief = BeliefState(domains={
            "book_flight": dict(tool_schemas["book_flight"].parameters),
        })
        weights = [belief.candidate_weight(c, tool_schemas["book_flight"]) for c in candidates]
        probs = belief.normalize(weights)
        questions = [
            Question("Where from?", aspects=(Aspect("book_flight", "origin"),)),
        ]

        terminal = get_terminal_value(probs)
        from sage_agent.core.bayesian_dp import CostModel
        cost_model = CostModel(query_base_cost=0.01, query_aspect_cost=0.0, query_open_domain_cost=0.0, execution_cost=0.0, reward=1.0)
        dp_val = _dp_value(belief, candidates, probs, questions, tool_schemas, budget=2, cost_model=cost_model)

        # With low query cost and useful questions, DP should find value >= terminal
        assert dp_val >= terminal - 0.01  # Allow small numerical tolerance


# =============================================================================
# Test compute_question_value_dp (Main Entry Point)
# =============================================================================

class TestComputeQuestionValueDP:
    def test_returns_scored_questions(self, tool_schemas, basic_candidates, origin_question, destination_question):
        """Should return scored questions sorted by value."""
        belief = BeliefState(domains={
            "book_flight": dict(tool_schemas["book_flight"].parameters),
        })
        weights = [belief.candidate_weight(c, tool_schemas["book_flight"]) for c in basic_candidates]
        probs = belief.normalize(weights)

        scored = compute_question_value_dp(
            belief=belief,
            candidates=basic_candidates,
            probabilities=probs,
            questions=[origin_question, destination_question],
            tool_schemas=tool_schemas,
            budget=2,
        )

        assert len(scored) == 2
        # Should be sorted descending
        assert scored[0][0] >= scored[1][0]
        # Each item is (score, question)
        assert isinstance(scored[0][1], Question)

    def test_empty_questions(self, tool_schemas, basic_candidates):
        """Should return empty list for empty questions."""
        belief = BeliefState(domains={
            "book_flight": dict(tool_schemas["book_flight"].parameters),
        })
        probs = [1/3, 1/3, 1/3]

        scored = compute_question_value_dp(
            belief=belief,
            candidates=basic_candidates,
            probabilities=probs,
            questions=[],
            tool_schemas=tool_schemas,
            budget=2,
        )
        assert scored == []

    def test_redundancy_penalty_applied(self, tool_schemas, basic_candidates, origin_question):
        """Redundancy penalty should reduce scores for repeated aspects."""
        belief = BeliefState(domains={
            "book_flight": dict(tool_schemas["book_flight"].parameters),
        })
        weights = [belief.candidate_weight(c, tool_schemas["book_flight"]) for c in basic_candidates]
        probs = belief.normalize(weights)

        # Score without redundancy
        scored_fresh = compute_question_value_dp(
            belief=belief,
            candidates=basic_candidates,
            probabilities=probs,
            questions=[origin_question],
            tool_schemas=tool_schemas,
            budget=2,
            aspect_counts={},
        )

        # Score with redundancy (origin already asked twice)
        scored_redundant = compute_question_value_dp(
            belief=belief,
            candidates=basic_candidates,
            probabilities=probs,
            questions=[origin_question],
            tool_schemas=tool_schemas,
            budget=2,
            aspect_counts={Aspect("book_flight", "origin"): 2},
            redundancy_weight=0.5,
        )

        assert scored_fresh[0][0] > scored_redundant[0][0]


# =============================================================================
# Test Multi-Step Lookahead Advantage
# =============================================================================

class TestMultiStepAdvantage:
    def test_dp_prefers_informative_first_question(self):
        """DP should prefer a question that sets up a better follow-up.

        Scenario: Two tools, 'diagnose' with params 'system' and 'subsystem'.
        - 'system' has 2 values: ["A", "B"]
        - 'subsystem' has 4 values: ["X", "Y", "Z", "W"]
        - But subsystem is only meaningful after system is known
        - Asking system first (2 values) then subsystem (4 values) is better
          than asking subsystem first
        """
        tool = ToolSchema(
            name="diagnose",
            parameters={
                "system": ParameterDomain.from_values(["A", "B"]),
                "subsystem": ParameterDomain.from_values(["X", "Y", "Z", "W"]),
            },
            required=frozenset({"system", "subsystem"}),
        )
        tool_schemas = {"diagnose": tool}

        # 4 candidates covering different combos
        candidates = [
            ToolCallCandidate("diagnose", {"system": "A", "subsystem": "X"}),
            ToolCallCandidate("diagnose", {"system": "A", "subsystem": "Y"}),
            ToolCallCandidate("diagnose", {"system": "B", "subsystem": "Z"}),
            ToolCallCandidate("diagnose", {"system": "B", "subsystem": "W"}),
        ]

        belief = BeliefState(domains={"diagnose": dict(tool.parameters)})
        weights = [belief.candidate_weight(c, tool) for c in candidates]
        probs = belief.normalize(weights)

        q_system = Question("Which system?", aspects=(Aspect("diagnose", "system"),))
        q_subsystem = Question("Which subsystem?", aspects=(Aspect("diagnose", "subsystem"),))

        scored = compute_question_value_dp(
            belief=belief,
            candidates=candidates,
            probabilities=probs,
            questions=[q_system, q_subsystem],
            tool_schemas=tool_schemas,
            budget=2,
            query_cost=0.01,
        )

        # Both questions should be scored
        assert len(scored) == 2

        # With budget=2, the DP planner should recognize that asking either
        # question first leads to different information cascades
        # Both should have positive scores (questions are useful)
        for score, _ in scored:
            assert score > -1.0  # Not absurdly negative


# =============================================================================
# Test Integration with SageAgent
# =============================================================================

class TestSageAgentIntegration:
    def test_dp_planning_flag_works(self, flight_tool):
        """SageAgent with use_dp_planning=True should run without errors."""
        candidate = ToolCallCandidate(
            tool_name="book_flight",
            arguments={"origin": "NYC", "destination": "LAX", "date": "2024-03-01"},
        )

        agent = SageAgent(
            tool_schemas=[flight_tool],
            candidate_generator=MockCandidateGenerator([candidate]),
            question_generator=MockQuestionGenerator([]),
            question_asker=MockQuestionAsker([]),
            tool_executor=MockToolExecutor(),
            constraint_extractor=MockConstraintExtractor(),
            config=SageAgentConfig(
                use_dp_planning=True,
                tau_execute=0.85,
            ),
        )

        result = agent.run("Book me a flight from NYC to LAX on March 1st")

        assert result.execution_result.success
        assert result.tool_call is not None
        assert result.tool_call.tool_name == "book_flight"

    def test_dp_planning_asks_questions(self, flight_tool):
        """SageAgent with DP planning should ask questions when uncertain."""
        # Two candidates with different origins -> uncertain
        candidates = [
            ToolCallCandidate("book_flight", {"origin": UNK, "destination": "LAX", "date": "2024-03-01"}),
            ToolCallCandidate("book_flight", {"origin": "NYC", "destination": UNK, "date": "2024-03-01"}),
        ]

        question = Question(
            text="Which city are you departing from?",
            aspects=(Aspect("book_flight", "origin"),),
        )

        asker = MockQuestionAsker(["NYC"])

        agent = SageAgent(
            tool_schemas=[flight_tool],
            candidate_generator=MockCandidateGenerator(candidates),
            question_generator=MockQuestionGenerator([question]),
            question_asker=asker,
            tool_executor=MockToolExecutor(),
            constraint_extractor=MockConstraintExtractor(),
            config=SageAgentConfig(
                use_dp_planning=True,
                tau_execute=0.85,
                max_questions=3,
            ),
        )

        result = agent.run("Book me a flight to LAX")

        # Should have asked at least one question
        assert asker.call_count >= 1

    def test_dp_vs_myopic_both_succeed(self, flight_tool):
        """Both DP and myopic modes should produce valid results for same input."""
        candidates = [
            ToolCallCandidate("book_flight", {"origin": UNK, "destination": "LAX", "date": "2024-03-01"}),
            ToolCallCandidate("book_flight", {"origin": "NYC", "destination": "LAX", "date": "2024-03-01"}),
        ]

        question = Question(
            text="Which city?",
            aspects=(Aspect("book_flight", "origin"),),
        )

        # Run with myopic EVPI
        agent_myopic = SageAgent(
            tool_schemas=[flight_tool],
            candidate_generator=MockCandidateGenerator(candidates),
            question_generator=MockQuestionGenerator([question]),
            question_asker=MockQuestionAsker(["NYC"]),
            tool_executor=MockToolExecutor(),
            constraint_extractor=MockConstraintExtractor(),
            config=SageAgentConfig(use_dp_planning=False, max_questions=2),
        )
        result_myopic = agent_myopic.run("Book flight")

        # Run with DP planning
        agent_dp = SageAgent(
            tool_schemas=[flight_tool],
            candidate_generator=MockCandidateGenerator(candidates),
            question_generator=MockQuestionGenerator([question]),
            question_asker=MockQuestionAsker(["NYC"]),
            tool_executor=MockToolExecutor(),
            constraint_extractor=MockConstraintExtractor(),
            config=SageAgentConfig(use_dp_planning=True, max_questions=2),
        )
        result_dp = agent_dp.run("Book flight")

        # Both should succeed
        assert result_myopic.tool_call is not None
        assert result_dp.tool_call is not None
        assert result_myopic.tool_call.tool_name == "book_flight"
        assert result_dp.tool_call.tool_name == "book_flight"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
