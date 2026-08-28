"""Regression tests for the LCB tool-agent action and L3 constraints."""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_ORCH_ROOT = _REPO_ROOT / "experiments" / "orchestration_hypothesis_testing"
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_ORCH_ROOT))
sys.path.insert(0, str(_ORCH_ROOT / "scripts"))

from _common.critics import critic_L3_review, critic_L3_review_detailed  # noqa: E402

from different_agents.v4.lcb_llm_tool_agent import (  # noqa: E402
    ACTION_SPACE,
    _available_actions,
    _decision_prompt,
    _parse_decision,
    _record,
    _trajectory_brief,
    _used_critic_actions,
    fallback_decision_after_parse_failure,
    initial_state,
)


def test_action_space_excludes_legacy_and_direct_verify_actions():
    assert ACTION_SPACE == (
        "generate",
        "critic_L0",
        "critic_L2",
        "critic_L3",
        "finish",
    )


def test_each_critic_is_available_once_per_real_generation():
    state = {
        "candidate_payload": "print(1)",
        "n_generations": 1,
        "max_generations": 2,
        "trajectory": [
            {"action": "generate"},
            {"action": "critic_L2"},
            {"action": "generate", "skipped": True},
        ],
    }
    assert _used_critic_actions(state) == {"critic_L2"}
    assert "critic_L2" not in _available_actions(state)

    state["trajectory"].append({"action": "generate"})
    assert _used_critic_actions(state) == set()
    assert "critic_L2" in _available_actions(state)


def test_no_candidate_only_allows_generate_or_finish():
    state = {
        "candidate_payload": "",
        "n_generations": 0,
        "max_generations": 1,
        "trajectory": [],
    }
    assert _available_actions(state) == ("generate",)
    state["n_generations"] = 1
    assert _available_actions(state) == ("finish",)
    assert fallback_decision_after_parse_failure(state)[0] == "finish"


def test_initial_state_caps_hidden_verification_to_one():
    state = initial_state(
        instance={},
        instance_id="example",
        benchmark="lcb_hard",
        max_steps=10,
        max_generations=3,
        max_verifications=99,
    )
    assert state["max_verifications"] == 1


def _review_client(
    *,
    content=None,
    reasoning_content=None,
    prompt_tokens=10,
    completion_tokens=2,
    calls=None,
):
    message = SimpleNamespace(content=content, reasoning_content=reasoning_content)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )
    def create(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        return response

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_router_requires_reasoning_before_action_in_json():
    state = {
        "candidate_payload": "print(1)",
        "n_generations": 1,
        "max_generations": 3,
        "step": 1,
        "max_steps": 10,
        "trajectory": [{"action": "generate"}],
    }
    prompt = _decision_prompt(state)
    assert (
        '{"reasoning":"short reason, at most two sentences",'
        '"action":"generate|critic_L0|critic_L2|critic_L3|finish"}'
    ) in prompt
    assert _parse_decision('{"reasoning":"Need tests.","action":"critic_L2"}') == (
        "critic_L2",
        "Need tests.",
    )


def test_router_marks_critic_evidence_with_candidate_version():
    state = {
        "candidate_payload": "print(2)",
        "n_generations": 2,
        "max_generations": 3,
        "step": 3,
        "max_steps": 10,
        "trajectory": [
            {"step": 0, "action": "generate", "candidate_version": 1},
            {
                "step": 1,
                "action": "critic_L3",
                "candidate_version": 1,
                "passed": False,
            },
            {"step": 2, "action": "generate", "candidate_version": 2},
        ],
    }
    prompt = _decision_prompt(state)
    assert "current_candidate_version: v2" in prompt
    assert "critic failures from an older version" in prompt
    assert "candidate=v1 action=critic_L3 passed=False" in prompt
    assert "candidate=v2 action=generate" in prompt

    critic_row = _record(
        state,
        action="critic_L0",
        reasoning="Check syntax.",
        passed=True,
    )[-1]
    assert critic_row["candidate_version"] == 2

    generation_row = _record(
        state,
        action="generate",
        reasoning="Try another candidate.",
    )[-1]
    assert generation_row["candidate_version"] == 3
    assert "candidate=v3 action=generate" in _trajectory_brief([generation_row])


def test_l3_uses_reasoning_content_and_last_explicit_verdict():
    passed, cost = critic_L3_review(
        "problem",
        "code",
        _review_client(reasoning_content="Initial FAIL concern, final verdict: PASS"),
    )
    assert passed is True
    assert cost == pytest.approx(20 / 1_000_000)


def test_l3_unparseable_response_is_missing_evidence_not_failure():
    passed, cost = critic_L3_review(
        "problem",
        "code",
        _review_client(content="The implementation needs another look."),
    )
    assert passed is None
    assert cost == pytest.approx(20 / 1_000_000)


def test_l3_json_response_has_short_reasoning_and_4096_token_limit():
    calls = []
    result = critic_L3_review_detailed(
        "problem",
        "code",
        _review_client(
            content='{"reasoning":"The loop is off by one.","verdict":"FAIL"}',
            calls=calls,
        ),
    )
    assert result.passed is False
    assert result.reasoning == "The loop is off by one."
    assert result.raw_response.startswith('{"reasoning"')
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 2
    assert calls[0]["max_tokens"] == 4096
    l3_prompt = calls[0]["messages"][0]["content"]
    assert l3_prompt.index('{"reasoning"') < l3_prompt.index('"verdict"')
