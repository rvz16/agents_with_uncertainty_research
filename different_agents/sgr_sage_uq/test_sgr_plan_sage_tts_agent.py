"""Tests for SGRPlanSageTTSAgent with offline mocks."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

# Project root is 2 levels up from different_agents/sgr_sage_uq/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = os.path.join(REPO_ROOT, "different_agents", "shared")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, LOCAL_DIR)
sys.path.insert(0, SHARED_DIR)

from sage_agent import ParameterDomain, ToolRegistryExecutor, ToolSchema
from sage_agent.core.constraints import SimpleConstraintExtractor
from sage_agent.core.types import Aspect, Question

from sgr_sage_uq_agent import ActionSchema, SAGEConfig
from sgr_plan_sage_tts_agent import (
    SGRPlanSageTTSAgent,
    TTSChooser,
    TTSConfig,
    GeneratePlanTool,
    ReasoningTool,
)


class MockTTSResponse:
    def __init__(self, content: str, uncertainty: float) -> None:
        self.content = content
        self.response_metadata = {
            "tts_metadata": {
                "uncertainty_score": uncertainty,
                "consensus_score": 1.0 - uncertainty,
            }
        }


class MockChatTTS:
    def __init__(self, responses: List[str], uncertainties: List[float]) -> None:
        self._responses = responses
        self._uncertainties = uncertainties
        self._idx = 0

    def invoke(self, messages: List[Any]) -> MockTTSResponse:
        if self._idx >= len(self._responses):
            return MockTTSResponse("{}", 1.0)
        content = self._responses[self._idx]
        uncertainty = self._uncertainties[self._idx]
        self._idx += 1
        return MockTTSResponse(content, uncertainty)


class DummyQuestionGenerator:
    def __init__(self, questions: Optional[List[Question]] = None) -> None:
        self._questions = questions or []

    def generate_questions(self, *args, **kwargs) -> List[Question]:
        return list(self._questions)


class DummyAsker:
    def __init__(self, responses: Dict[str, str]) -> None:
        self.responses = responses
        self.questions: List[Question] = []

    def ask(self, question: Question) -> str:
        self.questions.append(question)
        for aspect in question.aspects:
            key = f"{aspect.tool_name}:{aspect.param_name}"
            if key in self.responses:
                return self.responses[key]
        return "NYC"


def _make_tool_schema() -> ToolSchema:
    return ToolSchema(
        name="book_flight",
        parameters={
            "origin": ParameterDomain.from_values(["NYC", "BOS"]),
            "destination": ParameterDomain.from_values(["LAX", "SFO"]),
            "date": ParameterDomain.from_values(["2024-03-01", "2024-03-02"]),
        },
        required=frozenset({"origin", "destination", "date"}),
    )


def test_tts_best_of_picks_lowest_uncertainty(monkeypatch):
    responses = ['{"ok": 1}', '{"ok": 2}', '{"ok": 3}']
    uncertainties = [0.4, 0.1, 0.2]
    mock_llm = MockChatTTS(responses, uncertainties)

    monkeypatch.setattr("sgr_plan_sage_tts_agent.ChatTTS", lambda **_: mock_llm)

    chooser = TTSChooser(
        config=TTSConfig(),
        system_prompt="Return JSON only.",
    )
    best, best_unc, all_attempts = chooser.best_of("test", attempts=3)

    assert best == '{"ok": 2}'
    assert best_unc == 0.1
    assert len(all_attempts) == 3


def test_agent_executes_plan_steps(monkeypatch):
    tool_schema = _make_tool_schema()
    tool_registry = {
        "book_flight": lambda args: {"ok": True, "args": dict(args)},
    }

    agent = SGRPlanSageTTSAgent(
        tool_schemas={tool_schema.name: tool_schema},
        tool_executor=ToolRegistryExecutor(tool_registry),
        tts_config=TTSConfig(),
        sage_config=SAGEConfig(max_clarification_rounds=2),
        constraint_extractor=SimpleConstraintExtractor(),
    )
    agent._question_generator = DummyQuestionGenerator([])

    def fake_plan(_user_input: str) -> GeneratePlanTool:
        return GeneratePlanTool(
            reasoning="Plan for booking",
            research_goal="Book flight",
            planned_steps=["Pick flight", "Confirm booking", "Finalize details"],
        )

    def fake_reasoning(*_args, **_kwargs) -> ReasoningTool:
        return ReasoningTool(
            reasoning_steps=["Step 1", "Step 2"],
            current_situation="Working on booking",
            plan_status="In progress",
            enough_data=False,
            remaining_steps=["Pick flight"],
            task_completed=False,
        )

    def fake_candidates(_step: str, _obs: List[str]):
        action = ActionSchema(
            tool_name="book_flight",
            arguments={"origin": "NYC", "destination": "LAX", "date": "2024-03-01"},
            parameter_uncertainties={},
        )
        return [action], 0.1

    monkeypatch.setattr(agent, "_make_plan", fake_plan)
    monkeypatch.setattr(agent, "_make_reasoning", fake_reasoning)
    monkeypatch.setattr(agent, "_select_candidates", fake_candidates)

    result = agent.run("Book me a flight.")

    assert result["plan"]["planned_steps"] == ["Pick flight", "Confirm booking", "Finalize details"]
    assert len(result["results"]) == 3
    assert all(r["status"] == "executed" for r in result["results"])


def test_agent_clarifies_then_executes(monkeypatch):
    tool_schema = _make_tool_schema()
    tool_registry = {
        "book_flight": lambda args: {"ok": True, "args": dict(args)},
    }
    asker = DummyAsker({"book_flight:origin": "NYC"})

    agent = SGRPlanSageTTSAgent(
        tool_schemas={tool_schema.name: tool_schema},
        tool_executor=ToolRegistryExecutor(tool_registry),
        tts_config=TTSConfig(),
        sage_config=SAGEConfig(max_clarification_rounds=2),
        question_asker=asker,
        constraint_extractor=SimpleConstraintExtractor(),
    )

    question = Question(
        text="Which origin?",
        aspects=(Aspect(tool_name="book_flight", param_name="origin"),),
    )
    agent._question_generator = DummyQuestionGenerator([question])

    def fake_plan(_user_input: str) -> GeneratePlanTool:
        return GeneratePlanTool(
            reasoning="Plan for booking",
            research_goal="Book flight",
            planned_steps=["Pick flight", "Confirm booking", "Finalize details"],
        )

    def fake_reasoning(*_args, **_kwargs) -> ReasoningTool:
        return ReasoningTool(
            reasoning_steps=["Step 1", "Step 2"],
            current_situation="Need origin",
            plan_status="In progress",
            enough_data=False,
            remaining_steps=["Pick flight"],
            task_completed=False,
        )

    call_count = {"n": 0}

    def fake_candidates(_step: str, _obs: List[str]):
        call_count["n"] += 1
        if call_count["n"] == 1:
            action = ActionSchema(
                tool_name="book_flight",
                arguments={"origin": "<UNK>", "destination": "LAX", "date": "2024-03-01"},
                parameter_uncertainties={},
            )
            return [action], 0.6
        action = ActionSchema(
            tool_name="book_flight",
            arguments={"origin": "NYC", "destination": "LAX", "date": "2024-03-01"},
            parameter_uncertainties={},
        )
        return [action], 0.2

    monkeypatch.setattr(agent, "_make_plan", fake_plan)
    monkeypatch.setattr(agent, "_make_reasoning", fake_reasoning)
    monkeypatch.setattr(agent, "_select_candidates", fake_candidates)

    result = agent.run("Book me a flight.")

    assert result["results"][0]["status"] == "executed"
    assert len(asker.questions) == 1
