from types import SimpleNamespace

import pytest

from agents.smolagents_agent import SmolagentsPolicy, _response_spans
from environments.alfworld_env import StepResult


def _tokenise(text: str) -> list[SimpleNamespace]:
    """Whitespace-preserving split so token offsets reconstruct the response."""
    tokens = []
    for index, chunk in enumerate(text.split(" ")):
        token = chunk if index == 0 else " " + chunk
        tokens.append(SimpleNamespace(token=token, logprob=-0.1 * (index + 1)))
    return tokens


def _response(text: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(role="assistant", content=text, tool_calls=None),
                logprobs=SimpleNamespace(content=_tokenise(text)),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=30, completion_tokens=10, total_tokens=40),
        provider="Novita",
    )


class ScriptedCompletions:
    def __init__(self, texts):
        self.texts = list(texts)
        self.kwargs = []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        text = (
            self.texts.pop(0)
            if self.texts
            else "```python\nfinal_answer('stop')\n```"
        )
        return _response(text)


class ScriptedPolicy(SmolagentsPolicy):
    """Policy whose recording model talks to a scripted OpenAI client."""

    def __init__(self, texts, **kwargs):
        super().__init__(
            base_url="http://unused/v1",
            api_key="unused",
            model="test-model",
            **kwargs,
        )
        self.completions = ScriptedCompletions(texts)

    def _build_model(self, session, generations):
        model = super()._build_model(session, generations)
        model.client = SimpleNamespace(
            chat=SimpleNamespace(completions=self.completions)
        )
        return model


class FakeEnv:
    """ALFWorld stand-in: scripted observations, admissible sets and outcomes."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.actions: list[str] = []

    def step(self, action: str) -> StepResult:
        self.actions.append(action)
        return self.steps.pop(0)


def _initial(admissible):
    return SimpleNamespace(
        episode_id="pick_and_place_simple-abc123",
        task_type="pick_and_place_simple",
        task="put a mug in the cabinet",
        observation="You are in the middle of a room.",
        admissible_actions=list(admissible),
        gamefile="/tmp/game.tw-pddl",
    )


def _step(observation, admissible, *, done=False, won=False, progress=0.0):
    return StepResult(
        observation=observation,
        admissible_actions=list(admissible),
        done=done,
        won=won,
        progress=progress,
    )


def test_response_spans_splits_thought_and_code() -> None:
    thought_span, action_span, code = _response_spans(
        'Thought: go there\n<code>\ntake_action("go to drawer 1")\n</code>'
    )
    assert thought_span == (0, len("Thought: go there\n"))
    assert "take_action" in code
    assert action_span is not None

    fenced_thought, fenced_action, fenced_code = _response_spans(
        'Reasoning\n```python\ntake_action("look")\n```'
    )
    assert fenced_thought is not None and fenced_action is not None
    assert "take_action" in fenced_code

    assert _response_spans("no code at all")[1] is None


def test_episode_records_one_row_per_generation_with_logprobs() -> None:
    env = FakeEnv(
        [
            _step("You arrive at drawer 1.", ["open drawer 1", "look"]),
            _step("You win.", ["look"], done=True, won=True, progress=1.0),
        ]
    )
    policy = ScriptedPolicy(
        [
            'Thought: walk over\n```python\nprint(take_action("go to drawer 1"))\n```',
            'Thought: open it\n```python\nprint(take_action("open drawer 1"))\n```',
        ]
    )
    result = policy.run_episode(env, _initial(["go to drawer 1", "look"]), max_steps=30)

    assert env.actions == ["go to drawer 1", "open drawer 1"]
    assert result.stop_reason == "success"
    assert result.final_success is True
    assert result.total_tokens == 80
    assert [row["step"] for row in result.records] == [1, 2]

    first = result.records[0]
    assert first["env_actions"] == ["go to drawer 1"]
    assert first["action"] == "go to drawer 1"
    assert first["format_valid"] and first["action_valid"]
    assert first["fallback_reason"] is None
    assert first["logprobs_available"]
    assert first["provider"] == "Novita"
    assert first["admissible_actions"] == ["go to drawer 1", "look"]
    assert first["uq"]["thought"]["num_tokens"] > 0
    assert first["uq"]["action"]["num_tokens"] > 0
    assert first["uq"]["combined"]["num_tokens"] > first["uq"]["action"]["num_tokens"]
    assert first["usage"]["total_tokens"] == 40

    # The admissible list handed to generation 2 is the one the env returned.
    assert result.records[1]["admissible_actions"] == ["open drawer 1", "look"]
    assert result.records[1]["done"] is True
    assert policy.completions.kwargs[0]["logprobs"] is True


def test_inadmissible_action_is_recorded_as_a_fallback() -> None:
    env = FakeEnv([_step("Nothing happens.", ["look"], done=True, won=False)])
    policy = ScriptedPolicy(
        ['Thought: teleport\n```python\nprint(take_action("teleport"))\n```']
    )
    result = policy.run_episode(env, _initial(["look", "inventory"]), max_steps=30)

    assert env.actions == ["look"]  # resolved onto the admissible set
    row = result.records[0]
    assert row["proposed_action"] == "teleport"
    assert row["action_valid"] is False
    assert row["fallback_reason"] == "inadmissible_action"
    assert result.final_success is False
    assert result.stop_reason == "environment_done"


def test_generation_without_code_is_marked_invalid_format() -> None:
    env = FakeEnv([_step("You arrive.", ["look"], done=True, won=False)])
    policy = ScriptedPolicy(
        [
            "I will think about it in prose only.",
            'Thought: act\n```python\nprint(take_action("look"))\n```',
        ]
    )
    result = policy.run_episode(env, _initial(["look"]), max_steps=30)

    assert result.records[0]["format_valid"] is False
    assert result.records[0]["fallback_reason"] == "invalid_format"
    assert result.records[0]["env_actions"] == []
    assert result.records[-1]["env_actions"] == ["look"]


def test_env_step_budget_ends_the_episode() -> None:
    env = FakeEnv(
        [
            _step("step one", ["look"]),
            _step("step two", ["look"]),
        ]
    )
    policy = ScriptedPolicy(
        [
            'Thought: one\n```python\nprint(take_action("look"))\n```',
            'Thought: two\n```python\nprint(take_action("look"))\n```',
            'Thought: three\n```python\nprint(take_action("look"))\n```',
        ]
    )
    result = policy.run_episode(env, _initial(["look"]), max_steps=2)

    assert len(env.actions) == 2
    assert result.stop_reason == "max_steps"
    assert len(result.records) == 2


def test_final_answer_generation_counts_as_a_deliberate_stop() -> None:
    env = FakeEnv([_step("You arrive.", ["look"])])
    policy = ScriptedPolicy(
        [
            'Thought: act\n```python\nprint(take_action("look"))\n```',
            'Thought: give up\n```python\nfinal_answer("stuck")\n```',
        ]
    )
    result = policy.run_episode(env, _initial(["look"]), max_steps=30)

    assert result.stop_reason == "agent_stopped"
    last = result.records[-1]
    assert last["env_actions"] == []
    assert last["action_valid"] is True
    assert last["fallback_reason"] is None


@pytest.mark.parametrize("budget", [1, 5])
def test_agent_budget_is_positive(budget: int) -> None:
    policy = ScriptedPolicy([], agent_max_steps=budget)
    assert policy.agent_max_steps == budget


def test_empty_response_is_retried_with_a_doubled_limit() -> None:
    env = FakeEnv([_step("You arrive.", ["look"], done=True, won=True)])
    policy = ScriptedPolicy(
        ["", 'Thought: act\n```python\nprint(take_action("look"))\n```'],
        max_tokens=1024,
        empty_response_retries=1,
    )
    result = policy.run_episode(env, _initial(["look"]), max_steps=30)

    assert len(result.records) == 1
    usage = result.records[0]["usage"]
    assert usage["request_attempts"] == 2
    assert usage["empty_response_retries"] == 1
    assert usage["generation_token_limit"] == 2048
    assert usage["total_tokens"] == 80  # the discarded attempt is still paid for
    assert policy.completions.kwargs[0]["max_tokens"] == 1024
    assert policy.completions.kwargs[1]["max_tokens"] == 2048
    assert result.final_success is True


def test_empty_response_gives_up_after_the_retry_budget() -> None:
    env = FakeEnv([_step("You arrive.", ["look"], done=True, won=False)])
    policy = ScriptedPolicy(["", "", ""], empty_response_retries=1)
    result = policy.run_episode(env, _initial(["look"]), max_steps=30)

    assert result.records[0]["usage"]["request_attempts"] == 2
    assert result.records[0]["format_valid"] is False


def test_framework_prose_stop_sequences_are_dropped() -> None:
    """smolagents also stops on "Observation:" and "Calling tools:".

    Both are words a reasoning model uses while it thinks, and a hit inside
    hidden reasoning ends the turn with empty content. Only the framework's own
    code-tag rule survives: no stop at all for markdown fences (the close tag is
    a prefix of the open one), the close tag for the <code> format.
    """
    env = FakeEnv([_step("You arrive.", ["look"], done=True, won=True)])
    policy = ScriptedPolicy(
        ['Thought: act\n```python\nprint(take_action("look"))\n```']
    )
    policy.run_episode(env, _initial(["look"]), max_steps=30)
    assert "stop" not in policy.completions.kwargs[0]

    xml_env = FakeEnv([_step("You arrive.", ["look"], done=True, won=True)])
    xml_policy = ScriptedPolicy(
        ['Thought: act\n<code>\nprint(take_action("look"))\n</code>'],
        code_block_tags=None,
    )
    xml_policy.run_episode(xml_env, _initial(["look"]), max_steps=30)
    assert xml_policy.completions.kwargs[0]["stop"] == ["</code>"]


def test_transient_endpoint_failure_is_retried_inside_the_episode() -> None:
    class FlakyCompletions(ScriptedCompletions):
        def __init__(self, texts):
            super().__init__(texts)
            self.failures = 1

        def create(self, **kwargs):
            if self.failures:
                self.failures -= 1
                raise type("APIConnectionError", (Exception,), {})("Connection error.")
            return super().create(**kwargs)

    env = FakeEnv([_step("You arrive.", ["look"], done=True, won=True)])
    policy = ScriptedPolicy(
        ['Thought: act\n```python\nprint(take_action("look"))\n```'], max_retries=2
    )
    policy.completions = FlakyCompletions(list(policy.completions.texts))
    result = policy.run_episode(env, _initial(["look"]), max_steps=30)

    assert result.stop_reason == "success"
    assert len(result.records) == 1


def test_generation_without_env_action_carries_its_own_state() -> None:
    """`done`/`progress` are per-row, not the end-of-episode state."""
    env = FakeEnv(
        [
            _step("You arrive.", ["look"], progress=0.5),
            _step("You win.", ["look"], done=True, won=True, progress=1.0),
        ]
    )
    policy = ScriptedPolicy(
        [
            'Thought: act\n```python\nprint(take_action("look"))\n```',
            "prose with no code at all",
            'Thought: act\n```python\nprint(take_action("look"))\n```',
        ]
    )
    result = policy.run_episode(env, _initial(["look"]), max_steps=30)

    middle = result.records[1]
    assert middle["env_actions"] == []
    assert middle["done"] is False
    assert middle["progress"] == 0.5  # state when that generation was made
    assert result.records[-1]["done"] is True


def test_the_confidence_request_appears_only_when_asked_for() -> None:
    """The ReAct policy asks for a Confidence line; smolagents needs its own.

    The framework owns its system prompt, so the request goes into the task
    prompt as a comment line the existing parser already recognises.
    """
    from agents.smolagents_agent import SmolagentsPolicy, _EnvSession

    initial = SimpleNamespace(
        episode_id="e", task_type="t", task="put a mug on the desk",
        observation="a room", admissible_actions=["look"], gamefile="g",
    )
    session = _EnvSession(
        env=SimpleNamespace(), initial=initial, max_steps=30,
        repeat_action_limit=2, seed=0,
    )

    plain = SmolagentsPolicy._task_prompt(initial, session)
    assert "Confidence" not in plain

    session.verbalized = True
    asked = SmolagentsPolicy._task_prompt(initial, session)
    assert "# Confidence: <number between 0.00 and 1.00>" in asked
    # the rules survive the insertion
    assert "take_action" in asked and "Rules:" in asked


def test_the_judge_is_exempted_from_the_one_action_rule() -> None:
    """The rule that fixed multi-action blocks also silenced the reviewer.

    "Make exactly ONE take_action call per code block" reads as a ban on any
    second call, and the agent obeyed: zero judge calls in 140 episodes against
    264 for ReAct on the same model. The exemption appears only when the tool
    is actually registered.
    """
    from agents.smolagents_agent import SmolagentsPolicy, _EnvSession

    initial = SimpleNamespace(
        episode_id="e", task_type="t", task="put a mug on the desk",
        observation="a room", admissible_actions=["look"], gamefile="g",
    )
    session = _EnvSession(
        env=SimpleNamespace(), initial=initial, max_steps=50,
        repeat_action_limit=2, seed=0,
    )

    without = SmolagentsPolicy._task_prompt(initial, session)
    assert "check_progress" not in without
    assert "exactly ONE take_action call" in without

    session.judge_available = True
    with_judge = SmolagentsPolicy._task_prompt(initial, session)
    assert "check_progress() is not an environment action" in with_judge
    assert "does not count" in with_judge
    # the original rule survives: it is narrowed, not removed
    assert "exactly ONE take_action call" in with_judge


def test_the_hidden_channel_is_used_only_when_the_answer_is_missing() -> None:
    """gpt-oss finishes inside its analysis channel on the code prompt.

    6113 of 7085 generations returned an empty visible answer at a median of
    621 tokens out of 4096 allowed, so it is not a budget: the model considers
    itself done. The work is in the reasoning -- take_action appears in 295 of
    300 sampled empty generations -- and reading it is the difference between
    a usable run and 111 episodes ending in final_answer.
    """
    from agents.smolagents_agent import _visible_or_reasoning

    answered = SimpleNamespace(content="```py\ntake_action('look')\n```")
    text, recovered = _visible_or_reasoning(answered, None)
    assert "take_action" in text and not recovered

    silent = SimpleNamespace(content="", reasoning_content="We should look.\ntake_action('look')")
    text, recovered = _visible_or_reasoning(silent, None)
    assert "take_action" in text and recovered

    # the visible answer always wins, even when both are present
    both = SimpleNamespace(content="visible", reasoning_content="hidden")
    text, recovered = _visible_or_reasoning(both, None)
    assert text == "visible" and not recovered

    nothing = SimpleNamespace(content="")
    assert _visible_or_reasoning(nothing, None) == ("", False)
