"""Guards against silently wrong measurements in the agent loop.

Each test here corresponds to a way the pipeline used to produce a number that
looked fine and meant nothing: a verifier that never ran recorded as a failed
patch, a truncated generation parsed as if it were an answer, a critic that
could not reach its judge answering FAIL for every episode. None of these
show up as errors -- they show up as plausible results -- so they need tests.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from code_uq.agents import lcb_llm_tool_agent as agent
from code_uq.environments.fitted_live.common import Candidate, CriticResult, VerifyResult


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeAdapter:
    """A benchmark adapter whose behaviour each test dictates."""

    benchmark = "fake"

    def __init__(
        self,
        *,
        verify_result: VerifyResult | None = None,
        unavailable: set[str] | None = None,
        critic_result: CriticResult | None = None,
    ) -> None:
        self._verify_result = verify_result or VerifyResult(True, detail="1/1")
        self._unavailable = unavailable or set()
        self._critic_result = critic_result or CriticResult(True)
        self.verify_calls = 0

    def unavailable_actions(self) -> set[str]:
        return set(self._unavailable)

    def load_instances(self) -> list[dict]:
        return [{"id": "one"}]

    def instance_id(self, instance: dict) -> str:
        return str(instance["id"])

    def build_prompt(self, instance, previous, action_log) -> str:
        return "solve it"

    def extract_candidate(self, instance, response_text: str) -> Candidate:
        return Candidate(payload=response_text, raw_text=response_text, kind="code")

    def run_critic(self, critic, instance, candidate, reviewer_client) -> CriticResult:
        return self._critic_result

    def verify(self, instance, candidate, run_id) -> VerifyResult:
        self.verify_calls += 1
        return self._verify_result


def _deps(adapter: FakeAdapter) -> Any:
    return agent.AgentDeps(
        adapter=adapter,
        llm_client=SimpleNamespace(),
        reviewer_client=SimpleNamespace(),
        model_id="fake-model",
        decision_temperature=0.0,
        generation_temperature=0.0,
        max_tokens_decision=32,
        max_tokens_generation=256,
        max_code_chars=1000,
    )


def _state(**overrides: Any) -> Any:
    state = agent.initial_state(
        instance={"id": "one"},
        instance_id="one",
        benchmark="fake",
        max_steps=10,
        max_generations=3,
        max_verifications=2,
        unavailable_actions=overrides.pop("unavailable_actions", ()),
    )
    state.update(overrides)
    return state


def _fake_response(
    *, content: str | None, reasoning: str = "", finish_reason: str = "stop"
) -> Any:
    message = SimpleNamespace(content=content, reasoning_content=reasoning)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason, logprobs=None)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20)
    return SimpleNamespace(choices=[choice], usage=usage)


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

def test_action_space_is_complete_by_default() -> None:
    assert agent.available_actions(_state()) == agent.ACTION_SPACE


def test_unavailable_actions_are_removed_from_the_action_space() -> None:
    actions = agent.available_actions(_state(unavailable_actions=("verify", "critic_L2")))

    assert "verify" not in actions
    assert "critic_L2" not in actions
    assert "generate" in actions and "finish" in actions


def test_the_decision_prompt_does_not_offer_unavailable_actions() -> None:
    prompt = agent._decision_prompt(_state(unavailable_actions=("verify", "critic_L2")))

    offered = prompt.split("Valid actions:\n")[1].split("\n")[0]
    assert "verify" not in offered
    assert "critic_L2" not in offered


def test_an_unavailable_action_produces_no_evidence_if_chosen_anyway() -> None:
    """A model may still name a dropped action; it must not read as a failure."""
    adapter = FakeAdapter(unavailable={"verify"})
    state = _state(
        unavailable_actions=("verify",),
        chosen_action="verify",
        candidate_payload="print(1)",
    )

    update = agent.execute_action_node(state, _deps(adapter))
    record = update["trajectory"][-1]

    assert adapter.verify_calls == 0
    assert record["passed"] is None
    assert record["skipped"] is True
    assert "fixed" not in update


def test_both_backends_agree_on_the_action_space() -> None:
    """The SAGE backend's tool list must match what the prompt advertises."""
    state = _state(unavailable_actions=("verify", "critic_L2"))

    schema_names = {schema.name for schema in agent.sage_tool_schemas(state)}

    assert schema_names == set(agent.available_actions(state))


def test_the_prior_ignores_episodes_nothing_could_verify() -> None:
    """An unverifiable patch is not evidence of a failing generator."""
    rows = [
        {"passed": True},
        {"passed": False},
        {"passed": None, "label_available": False},
    ]

    summary = agent.summarize_prior(rows)

    assert summary["prior_calibration_n"] == 2
    assert summary["prior_calibration_correct"] == 1


def test_the_fallback_decision_avoids_unavailable_actions() -> None:
    action, _ = agent.fallback_decision_after_parse_failure(
        _state(
            unavailable_actions=("verify", "critic_L2"),
            candidate_payload="print(1)",
            n_critic_runs=0,
        )
    )

    assert action not in {"verify", "critic_L2"}


# ---------------------------------------------------------------------------
# A verifier that cannot run is not a failing patch
# ---------------------------------------------------------------------------

def test_an_environment_without_a_verifier_starts_unlabelled() -> None:
    """No verifier in the action space means no label will ever be produced.

    Defaulting to "labelled" here would mark every SWE-Bench row as carrying
    ground truth it never had -- and `fixed=False` on all of them would look
    like a model that solved nothing.
    """
    assert _state(unavailable_actions=("critic_L2", "verify"))["label_available"] is False
    assert _state()["label_available"] is True


def test_an_unlabelled_episode_reports_that_in_its_row() -> None:
    adapter = FakeAdapter(unavailable={"verify"})
    state = _state(unavailable_actions=("verify",), candidate_payload="a diff")

    row = agent.result_record(
        state, _deps(adapter), 1.0, split_summary={}, prior_summary={}
    )

    assert row["label_available"] is False
    assert row["final_code"] == "a diff"  # the patch is kept for later labelling


def test_an_unavailable_verifier_leaves_the_episode_unlabelled() -> None:
    adapter = FakeAdapter(
        verify_result=VerifyResult(False, detail="harness_unavailable", available=False)
    )
    state = _state(chosen_action="verify", candidate_payload="print(1)")

    update = agent.execute_action_node(state, _deps(adapter))

    assert update["label_available"] is False
    assert "fixed" not in update  # the label is absent, not negative
    assert update["trajectory"][-1]["passed"] is None


def test_a_real_verifier_failure_is_still_a_label() -> None:
    adapter = FakeAdapter(verify_result=VerifyResult(False, detail="0/5"))
    state = _state(chosen_action="verify", candidate_payload="print(1)")

    update = agent.execute_action_node(state, _deps(adapter))

    assert update["label_available"] is True
    assert update["fixed"] is False
    assert update["trajectory"][-1]["passed"] is False


def test_final_verify_marks_the_episode_unlabelled_when_the_harness_is_missing() -> None:
    adapter = FakeAdapter(
        verify_result=VerifyResult(False, detail="harness_unavailable", available=False)
    )
    state = _state(candidate_payload="print(1)")

    final = agent.maybe_final_verify(state, _deps(adapter))

    assert final["label_available"] is False
    assert final["final_action"] == "final_verify_unavailable"
    assert final.get("fixed") is False  # untouched default, guarded by the flag


def test_final_verify_records_a_real_pass() -> None:
    adapter = FakeAdapter(verify_result=VerifyResult(True, detail="5/5"))
    state = _state(candidate_payload="print(1)")

    final = agent.maybe_final_verify(state, _deps(adapter))

    assert final["label_available"] is True
    assert final["fixed"] is True
    assert final["final_action"] == "final_verify_pass"


def test_the_result_row_carries_the_label_availability_flag() -> None:
    adapter = FakeAdapter()
    state = _state(candidate_payload="print(1)", label_available=False)

    row = agent.result_record(
        state, _deps(adapter), 1.0, split_summary={}, prior_summary={}
    )

    assert row["label_available"] is False


# ---------------------------------------------------------------------------
# Truncated generations are an absent answer, not a low-quality one
# ---------------------------------------------------------------------------

def test_text_source_distinguishes_the_answer_channel_from_reasoning() -> None:
    answered = _fake_response(content="the answer")
    thinking_only = _fake_response(content=None, reasoning="hmm, let me think")

    assert agent._message_text_with_source(answered.choices[0].message) == (
        "the answer",
        "content",
    )
    text, source = agent._message_text_with_source(thinking_only.choices[0].message)
    assert source == "reasoning"
    assert text == "hmm, let me think"


def test_a_truncated_generation_does_not_become_a_candidate(monkeypatch) -> None:
    """The classic failure: reasoning text parsed as if it were the solution."""
    monkeypatch.setattr(
        agent,
        "_create_completion",
        lambda deps, params, **kwargs: _fake_response(
            content=None,
            reasoning="I should probably write a loop here...",
            finish_reason="length",
        ),
    )
    adapter = FakeAdapter()
    state = _state(chosen_action="generate")

    update = agent.execute_action_node(state, _deps(adapter))
    record = update["trajectory"][-1]

    assert "candidate_payload" not in update
    assert update.get("n_generations", 0) == 0
    assert record["skipped"] is True
    assert record["truncated"] is True
    assert record["finish_reason"] == "length"


def test_a_complete_generation_becomes_a_candidate(monkeypatch) -> None:
    monkeypatch.setattr(
        agent,
        "_create_completion",
        lambda deps, params, **kwargs: _fake_response(content="print(42)"),
    )
    adapter = FakeAdapter()
    state = _state(chosen_action="generate")

    update = agent.execute_action_node(state, _deps(adapter))

    assert update["candidate_payload"] == "print(42)"
    assert update["n_generations"] == 1
    assert update["trajectory"][-1]["truncated"] is False


def test_the_token_cost_of_a_truncated_generation_is_still_counted(monkeypatch) -> None:
    """Abstaining must not hide what the attempt spent."""
    monkeypatch.setattr(
        agent,
        "_create_completion",
        lambda deps, params, **kwargs: _fake_response(
            content=None, reasoning="thinking", finish_reason="length"
        ),
    )
    state = _state(chosen_action="generate")

    update = agent.execute_action_node(state, _deps(FakeAdapter()))

    assert update["prompt_tokens"] == 10
    assert update["completion_tokens"] == 20


# ---------------------------------------------------------------------------
# A critic with no verdict
# ---------------------------------------------------------------------------

def test_a_critic_without_a_verdict_records_none_not_false() -> None:
    adapter = FakeAdapter(
        critic_result=CriticResult(None, detail="reviewer_unavailable")
    )
    state = _state(chosen_action="critic_L3", candidate_payload="print(1)")

    update = agent.execute_action_node(state, _deps(adapter))

    assert update["trajectory"][-1]["passed"] is None


@pytest.mark.parametrize("verdict", [True, False])
def test_a_real_critic_verdict_is_preserved(verdict: bool) -> None:
    adapter = FakeAdapter(critic_result=CriticResult(verdict, detail="3/3"))
    state = _state(chosen_action="critic_L0", candidate_payload="print(1)")

    update = agent.execute_action_node(state, _deps(adapter))

    assert update["trajectory"][-1]["passed"] is verdict


# ---------------------------------------------------------------------------
# Budgets, repeat verification and the single-terminal-check setup
# ---------------------------------------------------------------------------

def test_a_spent_budget_removes_the_action() -> None:
    """An action that can only answer 'budget exhausted' still costs a step."""
    state = _state(hide_exhausted_actions=True, candidate_payload="code")
    state["n_generations"] = 3   # max_generations is 3 in _state
    state["n_verifications"] = 2  # max_verifications is 2

    actions = agent.available_actions(state)

    assert "generate" not in actions
    assert "verify" not in actions
    assert "finish" in actions and "think" in actions


def test_budgets_still_shown_while_they_last() -> None:
    state = _state(hide_exhausted_actions=True, candidate_payload="code")
    state["n_generations"] = 1

    actions = agent.available_actions(state)

    assert "generate" in actions and "verify" in actions
    assert '"generations": 2' in agent._decision_prompt(state)


def test_exhausted_actions_stay_on_the_menu_by_default() -> None:
    """Opt-in, so the earlier runs remain comparable."""
    state = _state(candidate_payload="code")
    state["n_generations"] = 99
    state["n_verifications"] = 99

    assert agent.available_actions(state) == agent.ACTION_SPACE


def test_the_prompt_states_remaining_calls_per_tool() -> None:
    """Grouped budgets make the controller infer which cap binds which action."""
    state = _state(candidate_payload="code")
    state["n_generations"] = 1   # max_generations is 3 in _state
    state["n_verifications"] = 1  # max_verifications is 2

    prompt = agent._decision_prompt(state)
    line = prompt.split("calls_left_per_action: ")[1].split("\n")[0]

    assert '"generate": 2' in line
    assert '"verify": 1' in line
    assert '"critic_L0": "unlimited"' in line


def test_verify_is_withheld_after_the_oracle_ruled_on_this_candidate() -> None:
    """Re-verifying unchanged code returns the same answer for the top price."""
    state = _state(no_repeat_verify=True, candidate_payload="code")
    state["trajectory"] = [{"action": "generate"}, {"action": "verify", "passed": False}]

    assert "verify" not in agent.available_actions(state)


def test_regenerating_makes_the_verifier_available_again() -> None:
    state = _state(no_repeat_verify=True, candidate_payload="code")
    state["trajectory"] = [
        {"action": "generate"},
        {"action": "verify", "passed": False},
        {"action": "generate"},
    ]

    assert agent.verified_current_candidate(state) is False
    assert "verify" in agent.available_actions(state)


def test_the_prompt_says_whether_the_oracle_already_ruled() -> None:
    state = _state(describe_actions=True, no_repeat_verify=True, candidate_payload="code")
    state["trajectory"] = [{"action": "generate"}, {"action": "verify", "passed": False}]

    assert "verifier_already_ran_on_this_candidate: true" in agent._decision_prompt(state)


def test_repeat_verification_is_allowed_by_default() -> None:
    state = _state(candidate_payload="code")
    state["trajectory"] = [{"action": "generate"}, {"action": "verify", "passed": False}]

    assert "verify" in agent.available_actions(state)
