"""The judge the agent calls itself, as opposed to the offline one."""
from types import SimpleNamespace

from agents.judge_tool import JudgeTool
from agents.react_agent import JUDGE_TOOL_ACTION, ReActAgent, resolve_action
import random


class _Client:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    class _Completions:
        def __init__(self, outer):
            self.outer = outer

        def create(self, **kwargs):
            self.outer.calls.append(kwargs)
            text = self.outer.replies.pop(0)
            if isinstance(text, Exception):
                raise text
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                usage=SimpleNamespace(total_tokens=42),
            )

    @property
    def chat(self):
        return SimpleNamespace(completions=self._Completions(self))


def _tool(*replies, budget=5):
    client = _Client(*replies)
    return JudgeTool(client=client, model="reviewer", budget=budget, retries=0), client


HISTORY = [{"thought": "t", "action": "go to desk 1", "observation": "a desk"}]


def test_the_verdict_comes_back_as_an_observation_the_agent_can_read():
    tool, _ = _tool('{"verdict":"FAIL","confidence":0.2,"reason":"the mug is elsewhere"}')
    verdict = tool.check("put a mug on the desk", HISTORY, step=3)
    assert verdict.passed is False and verdict.confidence == 0.2
    text = verdict.as_observation()
    assert "NOT COMPLETE" in text and "the mug is elsewhere" in text
    assert verdict.as_record()["step"] == 3


def test_the_judge_never_sees_the_environment_answer():
    """The prompt is built from the allow-list, so no label can leak back."""
    tool, client = _tool('{"verdict":"PASS","confidence":0.9,"reason":"done"}')
    history = [
        {
            "thought": "t",
            "action": "go to desk 1",
            "observation": "a desk",
            "won": True,
            "progress": 1.0,
            "final_success": True,
        }
    ]
    tool.check("put a mug on the desk", history)
    prompt = client.calls[0]["messages"][1]["content"]
    assert "put a mug on the desk" in prompt and "go to desk 1" in prompt
    for leak in ("won", "progress", "final_success", "True", "1.0"):
        assert leak not in prompt


def test_the_budget_is_enforced_and_costs_no_request():
    tool, client = _tool(
        '{"verdict":"FAIL","confidence":0.1,"reason":"no"}',
        '{"verdict":"FAIL","confidence":0.1,"reason":"no"}',
        budget=1,
    )
    tool.check("task", HISTORY)
    second = tool.check("task", HISTORY)
    assert second.status == "budget_exhausted"
    assert "used all of your checks" in second.as_observation()
    assert len(client.calls) == 1  # the refused call never reached the endpoint
    assert tool.remaining == 0


def test_a_failing_reviewer_does_not_fail_the_episode():
    tool, _ = _tool(RuntimeError("endpoint down"))
    verdict = tool.check("task", HISTORY)
    assert verdict.status == "error" and verdict.passed is None
    assert "unavailable" in verdict.as_observation()


def test_reset_returns_the_budget_between_episodes():
    tool, _ = _tool('{"verdict":"PASS","confidence":0.9,"reason":"done"}', budget=1)
    tool.check("task", HISTORY)
    tool.reset()
    assert tool.remaining == 1 and tool.calls == []


def test_the_meta_action_survives_action_resolution():
    """It is not in the admissible list, yet it must not become a fallback."""
    action, valid, reason = resolve_action(
        "check progress",
        ["look", "go to desk 1"],
        [],
        rng=random.Random(0),
        repeat_action_limit=2,
    )
    assert action == JUDGE_TOOL_ACTION and valid and reason is None

    # ... while a genuinely inadmissible action still falls back
    action, valid, reason = resolve_action(
        "fly to the moon",
        ["look"],
        [],
        rng=random.Random(0),
        repeat_action_limit=2,
    )
    assert action == "look" and not valid and reason == "inadmissible_action"


def test_the_prompt_advertises_the_action_only_when_there_is_a_budget():
    def prompt_of(**kwargs):
        agent = ReActAgent(
            base_url="http://unused", api_key="unused", model="test",
            client=SimpleNamespace(), **kwargs
        )
        return agent._system_prompt()

    assert "check progress" not in prompt_of()
    assert "check progress" in prompt_of(judge_tool_budget=3)
    assert "at most 3 times" in prompt_of(judge_tool_budget=3)
    # the two switches compose rather than override each other
    both = prompt_of(judge_tool_budget=3, verbalized=True)
    assert "check progress" in both and "Confidence:" in both
