"""Single terminal verification: the oracle is out of the model's reach.

Achieved by configuration rather than a dedicated flag: --max-verifications 0
together with --hide-exhausted-actions removes verify from the action space from
step zero, while the terminal check still runs.
"""
from __future__ import annotations

from code_uq.agents.lcb_llm_tool_agent import _decision_prompt, available_actions


def _state(**over):
    base = {
        "unavailable_actions": (),
        "hide_exhausted_actions": True,
        "no_repeat_verify": False,
        "critic_policy": "",
        "describe_actions": True,
        "cost_profile": "",
        "final_verify": True,
        "step": 3,
        "max_steps": 20,
        "n_generations": 1,
        "max_generations": 10,
        "n_verifications": 0,
        "max_verifications": 0,
        "candidate_payload": "class Solution: pass",
        "trajectory": [],
        "instance_id": "x",
    }
    base.update(over)
    return base


def test_verify_absent_when_budget_is_zero():
    assert "verify" not in available_actions(_state())


def test_verify_present_with_a_normal_budget():
    assert "verify" in available_actions(_state(max_verifications=3))


def test_prompt_states_the_single_terminal_check():
    text = _decision_prompt(_state())
    assert "exactly once" in text
    assert "not an action you can call" in text


def test_prompt_stays_silent_when_verify_is_callable():
    text = _decision_prompt(_state(max_verifications=3))
    assert "exactly once" not in text


def test_prompt_stays_silent_without_a_terminal_check():
    """No terminal verification means the claim would simply be false."""
    text = _decision_prompt(_state(final_verify=False))
    assert "exactly once" not in text


def test_generate_and_finish_remain_available():
    actions = available_actions(_state())
    assert "generate" in actions
    assert "finish" in actions
