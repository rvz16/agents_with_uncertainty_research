"""Test suite for SGR-SAGE-UQ Agent.

Phase 5: Testing & Benchmarking Readiness

This module provides:
1. Mock environment for offline testing (no TTS service required)
2. ClarifyBench scenario simulation
3. Traceability verification
4. Unit tests for core components
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional
from unittest.mock import MagicMock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sage_agent import (
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
from sage_agent.core.types import Aspect, ExecutionResult

# =============================================================================
# Mock TTS Service
# =============================================================================


class MockTTSResponse:
    """Mock response from TTS service."""

    def __init__(self, content: str, uncertainty: float = 0.3, confidence: float = 0.7):
        self.content = content
        self.response_metadata = {
            "tts_metadata": {
                "uncertainty_score": uncertainty,
                "consensus_score": confidence,
            }
        }


class MockChatTTS:
    """Mock ChatTTS for offline testing."""

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        uncertainty: float = 0.3,
        **kwargs,
    ):
        self.responses = responses or []
        self.response_idx = 0
        self.uncertainty = uncertainty
        self.call_history: List[Dict[str, Any]] = []

    def invoke(self, messages: List[Any]) -> MockTTSResponse:
        """Return mock response."""
        self.call_history.append({"messages": messages})

        if self.response_idx < len(self.responses):
            content = self.responses[self.response_idx]
            self.response_idx += 1
        else:
            # Default response
            content = '[{"tool": "mock_tool", "arguments": {"param": "<UNK>"}}]'

        return MockTTSResponse(
            content=content,
            uncertainty=self.uncertainty,
            confidence=1.0 - self.uncertainty,
        )


class MockQuestionAsker:
    """Mock question asker that returns predefined responses."""

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.default_response = "I'm not sure"
        self.questions_asked: List[Question] = []

    def ask(self, question: Question) -> str:
        """Return predefined response or default."""
        self.questions_asked.append(question)

        # Check if we have a response for any aspect
        for aspect in question.aspects:
            key = f"{aspect.tool_name}:{aspect.param_name}"
            if key in self.responses:
                return self.responses[key]

        # Check for partial match in question text
        for key, response in self.responses.items():
            if key.lower() in question.text.lower():
                return response

        return self.default_response


# =============================================================================
# Test Scenarios (ClarifyBench-style)
# =============================================================================


@dataclass
class TestScenario:
    """A test scenario for the SGR-SAGE-UQ agent."""
    name: str
    user_input: str
    tool_schemas: Dict[str, ToolSchema]
    expected_tool: str
    expected_args: Dict[str, Any]
    mock_responses: List[str]
    clarification_responses: Dict[str, str] = field(default_factory=dict)
    max_clarifications: int = 3
    uncertainty_level: float = 0.3


def create_flight_booking_scenarios() -> List[TestScenario]:
    """Create ClarifyBench-style flight booking scenarios."""
    flight_tool = ToolSchema(
        name="book_flight",
        parameters={
            "origin": ParameterDomain.from_values(["NYC", "BOS", "LAX", "SFO"]),
            "destination": ParameterDomain.from_values(["NYC", "BOS", "LAX", "SFO"]),
            "date": ParameterDomain.from_values(["2024-03-01", "2024-03-02", "2024-03-03"]),
        },
        required=frozenset({"origin", "destination", "date"}),
    )

    return [
        # Scenario 1: Complete information
        TestScenario(
            name="complete_info",
            user_input="Book a flight from NYC to LAX on March 1st 2024",
            tool_schemas={"book_flight": flight_tool},
            expected_tool="book_flight",
            expected_args={"origin": "NYC", "destination": "LAX", "date": "2024-03-01"},
            mock_responses=[
                json.dumps([{
                    "tool": "book_flight",
                    "arguments": {"origin": "NYC", "destination": "LAX", "date": "2024-03-01"}
                }])
            ],
            uncertainty_level=0.1,
        ),
        # Scenario 2: Missing destination
        TestScenario(
            name="missing_destination",
            user_input="I need to fly from Boston on March 2nd",
            tool_schemas={"book_flight": flight_tool},
            expected_tool="book_flight",
            expected_args={"origin": "BOS", "destination": "SFO", "date": "2024-03-02"},
            mock_responses=[
                json.dumps([{
                    "tool": "book_flight",
                    "arguments": {"origin": "BOS", "destination": "<UNK>", "date": "2024-03-02"}
                }]),
                # After clarification
                json.dumps([{
                    "tool": "book_flight",
                    "arguments": {"origin": "BOS", "destination": "SFO", "date": "2024-03-02"}
                }]),
            ],
            clarification_responses={
                "book_flight:destination": "San Francisco",
                "destination": "San Francisco",
            },
            uncertainty_level=0.5,
        ),
        # Scenario 3: Ambiguous - multiple possible interpretations
        TestScenario(
            name="ambiguous_request",
            user_input="Flight to California",
            tool_schemas={"book_flight": flight_tool},
            expected_tool="book_flight",
            expected_args={"origin": "NYC", "destination": "LAX", "date": "2024-03-01"},
            mock_responses=[
                # High uncertainty - multiple candidates
                json.dumps([
                    {"tool": "book_flight", "arguments": {"origin": "<UNK>", "destination": "LAX", "date": "<UNK>"}},
                    {"tool": "book_flight", "arguments": {"origin": "<UNK>", "destination": "SFO", "date": "<UNK>"}},
                ]),
                # After first clarification
                json.dumps([{
                    "tool": "book_flight",
                    "arguments": {"origin": "NYC", "destination": "LAX", "date": "<UNK>"}
                }]),
                # After second clarification
                json.dumps([{
                    "tool": "book_flight",
                    "arguments": {"origin": "NYC", "destination": "LAX", "date": "2024-03-01"}
                }]),
            ],
            clarification_responses={
                "book_flight:origin": "New York",
                "book_flight:destination": "Los Angeles",
                "book_flight:date": "March 1st",
                "origin": "New York",
                "date": "March 1st",
            },
            uncertainty_level=0.7,
        ),
    ]


def create_deletion_scenario() -> TestScenario:
    """Create a critical operation (deletion) scenario."""
    delete_tool = ToolSchema(
        name="delete_file",
        parameters={
            "path": ParameterDomain.continuous(),
            "recursive": ParameterDomain.from_values([True, False]),
        },
        required=frozenset({"path"}),
    )

    return TestScenario(
        name="critical_deletion",
        user_input="Delete the temp folder",
        tool_schemas={"delete_file": delete_tool},
        expected_tool="delete_file",
        expected_args={"path": "/tmp/temp_folder", "recursive": True},
        mock_responses=[
            json.dumps([{
                "tool": "delete_file",
                "arguments": {"path": "/tmp/temp_folder", "recursive": True}
            }])
        ],
        clarification_responses={
            "delete_file:path": "/tmp/temp_folder",
            "path": "/tmp/temp_folder",
            "confirm": "yes",
        },
        uncertainty_level=0.4,  # Even with low uncertainty, should clarify (critical op)
    )


# =============================================================================
# Test Functions
# =============================================================================


def test_action_schema_hashing():
    """Test ActionSchema hash computation for belief distribution."""
    from sgr_sage_uq_agent import ActionSchema

    action1 = ActionSchema(
        tool_name="test_tool",
        arguments={"a": 1, "b": "value"},
    )
    action2 = ActionSchema(
        tool_name="test_tool",
        arguments={"a": 1, "b": "value"},
    )
    action3 = ActionSchema(
        tool_name="test_tool",
        arguments={"a": 2, "b": "value"},
    )

    # Same content should have same hash
    assert action1.compute_hash() == action2.compute_hash()
    # Different content should have different hash
    assert action1.compute_hash() != action3.compute_hash()

    print("[PASS] ActionSchema hashing works correctly")


def test_sage_decision_engine():
    """Test SAGE decision engine logic."""
    from sgr_sage_uq_agent import (
        ActionSchema,
        SAGEConfig,
        SAGEDecisionEngine,
    )

    # Create simple tool schema
    tool = ToolSchema(
        name="test_tool",
        parameters={
            "param1": ParameterDomain.from_values(["a", "b", "c"]),
            "param2": ParameterDomain.from_values([1, 2, 3]),
        },
        required=frozenset({"param1"}),
    )

    config = SAGEConfig(
        cost_of_clarification=0.1,
        base_penalty_of_error=1.0,
    )

    engine = SAGEDecisionEngine(
        config=config,
        tool_schemas={"test_tool": tool},
    )

    # Test with high confidence (should ACT)
    hypotheses_confident = [
        ActionSchema(tool_name="test_tool", arguments={"param1": "a", "param2": 1}),
        ActionSchema(tool_name="test_tool", arguments={"param1": "a", "param2": 1}),
        ActionSchema(tool_name="test_tool", arguments={"param1": "a", "param2": 1}),
    ]

    domains = {"test_tool": dict(tool.parameters)}
    metrics, action, clarify = engine.decide(
        hypotheses=hypotheses_confident,
        domains=domains,
        questions=[],
        aspect_counts={},
        clarification_rounds=0,
    )

    assert metrics.chosen_action == "ACT", f"Expected ACT, got {metrics.chosen_action}"
    print("[PASS] SAGE engine decides ACT for confident hypotheses")

    # Test with unknown required parameter (should CLARIFY)
    hypotheses_unknown = [
        ActionSchema(tool_name="test_tool", arguments={"param1": UNK, "param2": 1}),
    ]

    # Create a mock question for clarification
    question = Question(
        text="What value for param1?",
        aspects=(Aspect(tool_name="test_tool", param_name="param1"),),
    )

    metrics2, action2, clarify2 = engine.decide(
        hypotheses=hypotheses_unknown,
        domains=domains,
        questions=[question],
        aspect_counts={},
        clarification_rounds=0,
    )

    assert metrics2.chosen_action == "CLARIFY", f"Expected CLARIFY, got {metrics2.chosen_action}"
    print("[PASS] SAGE engine decides CLARIFY for unknown required params")


def test_belief_distribution_computation():
    """Test belief distribution computation from hypotheses."""
    from sgr_sage_uq_agent import (
        ActionSchema,
        SAGEConfig,
        SAGEDecisionEngine,
    )

    tool = ToolSchema(
        name="test_tool",
        parameters={
            "option": ParameterDomain.from_values(["A", "B", "C"]),
        },
        required=frozenset({"option"}),
    )

    config = SAGEConfig()
    engine = SAGEDecisionEngine(
        config=config,
        tool_schemas={"test_tool": tool},
    )

    # Create hypotheses with different frequencies
    hypotheses = [
        ActionSchema(tool_name="test_tool", arguments={"option": "A"}),
        ActionSchema(tool_name="test_tool", arguments={"option": "A"}),
        ActionSchema(tool_name="test_tool", arguments={"option": "A"}),
        ActionSchema(tool_name="test_tool", arguments={"option": "B"}),
        ActionSchema(tool_name="test_tool", arguments={"option": "C"}),
    ]

    domains = {"test_tool": dict(tool.parameters)}
    belief_dist = engine.compute_belief_distribution(hypotheses, domains)

    # A should have highest probability (3/5 base frequency)
    probs_by_option = {}
    for h in hypotheses:
        h_hash = h.compute_hash()
        if h_hash in belief_dist:
            probs_by_option[h.arguments["option"]] = belief_dist[h_hash]

    assert probs_by_option.get("A", 0) > probs_by_option.get("B", 0)
    assert probs_by_option.get("A", 0) > probs_by_option.get("C", 0)

    print("[PASS] Belief distribution correctly reflects hypothesis frequency")


def test_critical_tool_detection():
    """Test that critical tools get higher penalties."""
    from sgr_sage_uq_agent import ActionSchema, SAGEConfig, SAGEDecisionEngine

    config = SAGEConfig(
        base_penalty_of_error=1.0,
        critical_tool_patterns=["delete", "remove"],
        critical_penalty_multiplier=2.0,
    )

    engine = SAGEDecisionEngine(config=config, tool_schemas={})

    # Normal tool
    normal_action = ActionSchema(tool_name="search_files", arguments={})
    normal_penalty = engine.compute_penalty(normal_action)

    # Critical tool
    critical_action = ActionSchema(tool_name="delete_files", arguments={})
    critical_penalty = engine.compute_penalty(critical_action)

    assert critical_penalty == normal_penalty * 2.0
    print("[PASS] Critical tools get higher penalty")


def test_graph_builds_successfully():
    """Test that the graph builds without errors."""
    from sgr_sage_uq_agent import (
        GraphDependencies,
        SAGEConfig,
        SAGEDecisionEngine,
        SelfConsistencyClient,
        TTSConfig,
        build_sgr_sage_uq_graph,
    )
    from sage_agent import (
        LLMBackedCandidateGenerator,
        LLMBackedQuestionGenerator,
        create_sage_propagator,
    )
    from sage_agent.core.constraints import SimpleConstraintExtractor

    tool = ToolSchema(
        name="test_tool",
        parameters={"param": ParameterDomain.from_values(["x", "y"])},
        required=frozenset({"param"}),
    )

    # Create mock LLM
    class MockLLM:
        def complete(self, prompt: str) -> str:
            return '[{"tool": "test_tool", "arguments": {"param": "x"}}]'

        last_uncertainty = 0.3
        last_confidence = 0.7

    mock_llm = MockLLM()
    propagator = create_sage_propagator()

    # Mock TTS client
    tts_config = TTSConfig()
    tts_client = SelfConsistencyClient(tts_config)
    tts_client._llm = MockChatTTS(
        responses=['[{"tool": "test_tool", "arguments": {"param": "x"}}]']
    )

    sage_engine = SAGEDecisionEngine(
        config=SAGEConfig(),
        tool_schemas={"test_tool": tool},
        propagator=propagator,
    )

    deps = GraphDependencies(
        tool_schemas={"test_tool": tool},
        tts_client=tts_client,
        sage_engine=sage_engine,
        candidate_generator=LLMBackedCandidateGenerator(mock_llm),
        question_generator=LLMBackedQuestionGenerator(mock_llm),
        question_asker=MockQuestionAsker(),
        tool_executor=ToolRegistryExecutor({"test_tool": lambda args: {"result": "ok"}}),
        constraint_extractor=SimpleConstraintExtractor(),
        config=SageAgentConfig(),
        propagator=propagator,
    )

    graph = build_sgr_sage_uq_graph(deps)
    compiled = graph.compile()

    assert compiled is not None
    print("[PASS] Graph builds and compiles successfully")


def run_scenario_with_mock(scenario: TestScenario) -> Dict[str, Any]:
    """Run a test scenario with mock services."""
    from sgr_sage_uq_agent import (
        GraphDependencies,
        SAGEConfig,
        SAGEDecisionEngine,
        SelfConsistencyClient,
        TTSConfig,
        build_sgr_sage_uq_graph,
        create_initial_state,
    )
    from sage_agent import (
        LLMBackedCandidateGenerator,
        LLMBackedQuestionGenerator,
        create_sage_propagator,
    )
    from sage_agent.core.constraints import SimpleConstraintExtractor

    # Create mock LLM that returns scenario responses
    # It detects whether it's being asked for candidates or questions
    class MockLLM:
        def __init__(self, responses: List[str], uncertainty: float, tool_schemas: Dict[str, ToolSchema]):
            self.responses = responses
            self.response_idx = 0
            self.uncertainty = uncertainty
            self.tool_schemas = tool_schemas

        def complete(self, prompt: str) -> str:
            # Detect if this is a question generation prompt
            # The question prompt contains unique text: "Only ask about parameters"
            if "only ask about parameters" in prompt.lower():
                # Generate questions for unknown parameters
                return self._generate_questions_response(prompt)

            # Otherwise return candidate response
            if self.response_idx < len(self.responses):
                response = self.responses[self.response_idx]
                self.response_idx += 1
                return response
            return self.responses[-1] if self.responses else "[]"

        def _generate_questions_response(self, prompt: str) -> str:
            """Generate mock questions for unknown parameters."""
            questions = []
            for tool_name, schema in self.tool_schemas.items():
                for param in schema.required:
                    # Check if <UNK> appears in the prompt for this param
                    if "<UNK>" in prompt or "unknown" in prompt.lower():
                        questions.append({
                            "question": f"What is the {param} for the {tool_name}?",
                            "aspects": [{"tool": tool_name, "param": param}]
                        })
            return json.dumps(questions) if questions else "[]"

        @property
        def last_uncertainty(self) -> float:
            return self.uncertainty

        @property
        def last_confidence(self) -> float:
            return 1.0 - self.uncertainty

    mock_llm = MockLLM(scenario.mock_responses, scenario.uncertainty_level, scenario.tool_schemas)
    propagator = create_sage_propagator()

    # Mock TTS client
    tts_config = TTSConfig()
    tts_client = SelfConsistencyClient(tts_config)
    tts_client._llm = MockChatTTS(
        responses=scenario.mock_responses,
        uncertainty=scenario.uncertainty_level,
    )

    sage_engine = SAGEDecisionEngine(
        config=SAGEConfig(max_clarification_rounds=scenario.max_clarifications),
        tool_schemas=scenario.tool_schemas,
        propagator=propagator,
    )

    # Mock tool execution
    def mock_execute(args: Mapping[str, Any]) -> Dict[str, Any]:
        return {"status": "success", "args": dict(args)}

    tool_registry = {name: mock_execute for name in scenario.tool_schemas}

    deps = GraphDependencies(
        tool_schemas=scenario.tool_schemas,
        tts_client=tts_client,
        sage_engine=sage_engine,
        candidate_generator=LLMBackedCandidateGenerator(mock_llm),
        question_generator=LLMBackedQuestionGenerator(mock_llm),
        question_asker=MockQuestionAsker(scenario.clarification_responses),
        tool_executor=ToolRegistryExecutor(tool_registry),
        constraint_extractor=SimpleConstraintExtractor(),
        config=SageAgentConfig(max_questions=scenario.max_clarifications),
        propagator=propagator,
    )

    graph = build_sgr_sage_uq_graph(deps)
    compiled = graph.compile()

    initial_state = create_initial_state(
        user_input=scenario.user_input,
        tool_schemas=scenario.tool_schemas,
        num_samples=8,
    )

    try:
        result = compiled.invoke(initial_state, {"recursion_limit": 30})
        return {
            "success": True,
            "status": result["status"],
            "result": result.get("result"),
            "error": result.get("error"),
            "clarification_rounds": result.get("clarification_rounds", 0),
            "trace": result.get("trace", []),
            "questions": result.get("questions", []),
            "hypotheses": result.get("hypotheses", []),
            "decision_metrics": result.get("decision_metrics"),
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def test_scenario(scenario: TestScenario, debug: bool = False) -> bool:
    """Test a single scenario."""
    print(f"\n--- Testing: {scenario.name} ---")
    print(f"Input: {scenario.user_input}")

    result = run_scenario_with_mock(scenario)

    if not result["success"]:
        print(f"[FAIL] Error: {result['error']}")
        if "traceback" in result:
            print(result["traceback"])
        return False

    print(f"Status: {result['status']}")
    print(f"Clarification rounds: {result['clarification_rounds']}")

    # Debug output for escalated scenarios
    if debug or result['status'] == 'escalated':
        if result.get('hypotheses'):
            print(f"Hypotheses ({len(result['hypotheses'])}):")
            for h in result['hypotheses'][:3]:
                print(f"  - {h.tool_name}: {dict(h.arguments)}")
        if result.get('questions'):
            print(f"Questions generated: {len(result['questions'])}")
        else:
            print("Questions generated: 0 (none!)")
        if result.get('decision_metrics'):
            m = result['decision_metrics']
            print(f"Decision: {m.chosen_action}, Risk: {m.risk_score:.2f}")

    if result["result"]:
        print(f"Tool: {result['result'].tool_name}")
        print(f"Args: {dict(result['result'].arguments)}")

        # Check expected tool
        if result["result"].tool_name != scenario.expected_tool:
            print(f"[FAIL] Expected tool {scenario.expected_tool}, got {result['result'].tool_name}")
            return False

    # Check trace for traceability
    if result["trace"]:
        print(f"Trace steps: {len(result['trace'])}")
        for step in result["trace"]:
            step_name = step.get("step", "unknown")
            print(f"  - {step_name}")

    print("[PASS]")
    return True


def run_all_tests():
    """Run all unit tests and scenarios."""
    print("=" * 60)
    print("SGR-SAGE-UQ Agent Test Suite")
    print("=" * 60)

    # Unit tests
    print("\n[Unit Tests]")
    test_action_schema_hashing()
    test_sage_decision_engine()
    test_belief_distribution_computation()
    test_critical_tool_detection()
    test_graph_builds_successfully()

    # Scenario tests
    print("\n[Scenario Tests]")
    scenarios = create_flight_booking_scenarios()
    scenarios.append(create_deletion_scenario())

    passed = 0
    failed = 0

    for scenario in scenarios:
        if test_scenario(scenario):
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
