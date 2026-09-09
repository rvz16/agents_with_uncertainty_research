from types import SimpleNamespace

from agents.react_agent import (
    ReActAgent,
    _segment_logprobs,
    parse_react_response,
)


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class SequencedCompletions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.max_tokens = []

    def create(self, **kwargs):
        self.max_tokens.append(kwargs["max_tokens"])
        return next(self.responses)


def _client(text: str):
    token_strings = ["Thought:", " inspect", "\n", "Action:", " look"]
    token_items = [
        SimpleNamespace(token=token, logprob=-0.1 * (index + 1))
        for index, token in enumerate(token_strings)
    ]
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                logprobs=SimpleNamespace(content=token_items),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=20, completion_tokens=5, total_tokens=25
        ),
    )
    completions = FakeCompletions(response)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def test_parse_react_response() -> None:
    parsed = parse_react_response("Thought: inspect\nAction: look")
    assert parsed.valid
    assert parsed.thought == "inspect"
    assert parsed.action == "look"
    assert not parse_react_response("look").valid


def test_agent_records_segmented_logprobs() -> None:
    client, completions = _client("Thought: inspect\nAction: look")
    agent = ReActAgent(
        base_url="http://unused",
        api_key="unused",
        model="test",
        client=client,
        extra_body={"provider": {"require_parameters": True}},
    )
    result = agent.act("inspect", [], ["look", "inventory"])
    assert result.action == "look"
    assert result.logprobs_available
    assert result.uq["thought"]["num_tokens"] > 0
    assert result.uq["action"]["num_tokens"] > 0
    assert result.uq["combined"]["num_tokens"] == 5
    assert completions.kwargs["logprobs"] is True
    assert completions.kwargs["extra_body"]["provider"]["require_parameters"] is True


def test_agent_falls_back_for_invalid_action() -> None:
    client, _ = _client("Thought: inspect\nAction: teleport")
    agent = ReActAgent(
        base_url="http://unused",
        api_key="unused",
        model="test",
        client=client,
    )
    result = agent.act("inspect", [], ["look", "inventory"])
    assert result.action == "look"
    assert not result.action_valid
    assert result.fallback_reason == "inadmissible_action"


def test_agent_retries_empty_reasoning_response_with_larger_limit() -> None:
    empty_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=""),
                logprobs=SimpleNamespace(content=[]),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=20, completion_tokens=512, total_tokens=532
        ),
    )
    valid_client, _ = _client("Thought: inspect\nAction: look")
    valid_response = valid_client.chat.completions.response
    completions = SequencedCompletions([empty_response, valid_response])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    agent = ReActAgent(
        base_url="http://unused",
        api_key="unused",
        model="test",
        client=client,
        max_tokens=512,
        max_empty_response_retries=1,
    )
    result = agent.act("inspect", [], ["look"])
    assert result.action == "look"
    assert result.empty_response_retries == 1
    assert result.request_attempts == 2
    assert result.generation_token_limit == 1024
    assert result.total_tokens == 557
    assert completions.max_tokens == [512, 1024]


def test_segments_survive_tokens_the_provider_dropped() -> None:
    """OpenRouter/Novita omits a few percent of tokens from `logprobs`.

    Concatenating what survives shifts every later token to the left, which
    used to leave the action segment empty and fill the thought segment with
    the wrong tokens; positions are searched for in the response text instead.
    """
    raw = "Thought: inspect\nAction: look"
    # " inspect" is the token the provider dropped.
    kept = ["Thought", ":", "\n", "Action", ":", " look"]
    records = [
        {"token": token, "logprob": -1.0 if token == " look" else -0.1}
        for token in kept
    ]
    uq = _segment_logprobs(raw, parse_react_response(raw), records)

    assert uq["combined"]["num_tokens"] == len(kept)
    # The action segment holds exactly the action token, at its real position.
    assert uq["action"]["num_tokens"] == 1
    assert uq["action"]["sum_logprob"] == -1.0
    # The thought segment keeps only its surviving token, and never the
    # "Action" ones that the concatenated offsets used to slide into it.
    assert uq["thought"]["num_tokens"] == 1
    assert uq["thought"]["sum_logprob"] == -0.1


def test_hidden_reasoning_is_scored_separately_from_the_answer() -> None:
    """A locally served gpt-oss returns log-probabilities for its whole stream.

    The visible answer is only the final channel, so counting the reasoning as
    part of the generation would make `combined` mean something different than
    it does on a hosted endpoint that returns the answer alone.
    """
    from agents.react_agent import split_reasoning_tokens

    raw = "Thought: inspect\nAction: look"
    stream = (
        ["<|channel|>", "analysis", "<|message|>", "We", " should", " look"]
        + ["<|end|>", "<|start|>", "assistant", "<|channel|>", "final", "<|message|>"]
        + ["Thought", ":", " inspect", "\n", "Action", ":", " look"]
        + ["<|return|>"]
    )
    records = [{"token": token, "logprob": -0.2} for token in stream]

    reasoning, content = split_reasoning_tokens(raw, records)
    assert "".join(r["token"] for r in content) == raw
    assert any(r["token"] == "analysis" for r in reasoning)
    assert not any(r["token"].startswith("<|") for r in content)


def test_a_plain_endpoint_has_no_reasoning_to_split() -> None:
    from agents.react_agent import split_reasoning_tokens

    records = [{"token": t, "logprob": -0.1} for t in ["Thought", ": ", "look"]]
    reasoning, content = split_reasoning_tokens("Thought: look", records)
    assert reasoning == []
    assert content == records


def test_trailing_punctuation_does_not_make_an_action_inadmissible() -> None:
    """Qwen3.6 ends its actions with a full stop.

    Exact matching turned 62% of that run's steps into fallbacks -- two thirds
    of them from responses that parsed perfectly -- so the measured policy was
    the harness, not the agent.
    """
    import random

    from agents.react_agent import resolve_action

    admissible = ["go to drawer 1", "look"]
    for proposed in ("go to drawer 1.", " go to drawer 1 ", "Go to drawer 1!"):
        action, valid, fallback = resolve_action(
            proposed, admissible, [], rng=random.Random(0), repeat_action_limit=2
        )
        assert (action, valid, fallback) == ("go to drawer 1", True, None), proposed

    # something genuinely absent still falls back
    action, valid, fallback = resolve_action(
        "teleport", admissible, [], rng=random.Random(0), repeat_action_limit=2
    )
    assert not valid and fallback == "inadmissible_action"


def _client_with_top_logprobs(text: str, alternatives: list[list[float]]):
    """A response whose tokens carry the top-k the server was asked for."""
    token_strings = ["Thought:", " inspect", "\n", "Action:", " look"]
    token_items = [
        SimpleNamespace(
            token=token,
            logprob=-0.1 * (index + 1),
            top_logprobs=[
                SimpleNamespace(token="x", logprob=value)
                for value in alternatives[index]
            ],
        )
        for index, token in enumerate(token_strings)
    ]
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                logprobs=SimpleNamespace(content=token_items),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=5, total_tokens=25),
    )
    completions = FakeCompletions(response)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def _agent(client, **kwargs):
    return ReActAgent(
        base_url="http://unused", api_key="unused", model="test", client=client, **kwargs
    )


def test_top_logprobs_are_requested_only_when_asked_for() -> None:
    client, completions = _client("Thought: inspect\nAction: look")
    _agent(client).act("inspect", [], ["look"])
    assert "top_logprobs" not in completions.kwargs

    client, completions = _client("Thought: inspect\nAction: look")
    _agent(client, top_logprobs=5).act("inspect", [], ["look"])
    assert completions.kwargs["top_logprobs"] == 5


def test_mean_token_entropy_separates_a_flat_head_from_a_peaked_one() -> None:
    """Entropy is not recoverable from the sampled token's log-probability.

    Both responses below draw exactly the same tokens with the same
    log-probabilities; only the shape of the distribution they were drawn from
    differs, which is the whole point of asking for `top_logprobs`.
    """
    import math

    peaked = [[math.log(0.97), math.log(0.02), math.log(0.01)]] * 5
    flat = [[math.log(0.34), math.log(0.33), math.log(0.33)]] * 5

    client, _ = _client_with_top_logprobs("Thought: inspect\nAction: look", peaked)
    sharp = _agent(client, top_logprobs=3).act("inspect", [], ["look"])
    client, _ = _client_with_top_logprobs("Thought: inspect\nAction: look", flat)
    vague = _agent(client, top_logprobs=3).act("inspect", [], ["look"])

    assert sharp.uq["combined"]["mean_token_logprob"] == vague.uq["combined"]["mean_token_logprob"]
    assert sharp.uq["combined"]["mean_token_entropy"] < vague.uq["combined"]["mean_token_entropy"]
    assert sharp.uq["combined"]["entropy_coverage"] == 1.0


def test_entropy_is_none_when_the_server_sent_no_alternatives() -> None:
    client, _ = _client("Thought: inspect\nAction: look")
    result = _agent(client).act("inspect", [], ["look"])
    assert result.uq["combined"]["mean_token_entropy"] is None
    assert result.uq["combined"]["perplexity"] is not None


def test_confidence_is_parsed_only_when_the_prompt_asks_for_it() -> None:
    text = "Thought: inspect\nAction: look\nConfidence: 0.35"
    client, completions = _client(text)
    plain = _agent(client).act("inspect", [], ["look"])
    assert "Confidence:" not in completions.kwargs["messages"][0]["content"]

    client, completions = _client(text)
    asked = _agent(client, verbalized=True).act("inspect", [], ["look"])
    assert "Confidence:" in completions.kwargs["messages"][0]["content"]
    # The parser reads the line wherever it appears; the prompt decides whether
    # the model ever writes one.
    assert asked.uq["combined"]["verbalized_confidence"] == 0.35
    assert plain.uq["combined"]["verbalized_confidence"] == 0.35


def test_final_confidence_is_a_separate_call_over_the_finished_trajectory() -> None:
    client, completions = _client("Confidence: 0.10")
    agent = _agent(client, verbalized=True)
    history = [{"thought": "t", "action": "look", "observation": "a room"}]
    result = agent.final_confidence("put a mug on the desk", history)
    assert result["verbalized_confidence"] == 0.10
    prompt = completions.kwargs["messages"][1]["content"]
    assert "put a mug on the desk" in prompt and "a room" in prompt


def test_an_action_is_read_out_of_an_unformatted_response() -> None:
    """Qwen deliberates in prose and runs past the token budget mid-sentence.

    Discarding those steps sent `look` to the environment on 50% of Qwen's
    turns, which measured our format rule rather than the policy.
    """
    text = (
        "The user wants a knife. I searched drawers 1-7 and found nothing.\n"
        "Let's try cabinet 1.\n"
        "Action: go to cabinet 1\n"
        "Wait, I should check whether there are more dra"
    )
    parsed = parse_react_response(text)
    assert parsed.action == "go to cabinet 1"
    assert parsed.recovered
    assert not parsed.valid  # never promoted to a clean parse
    assert "searched drawers" in parsed.thought


def test_the_last_complete_action_wins_when_the_model_changes_its_mind() -> None:
    text = (
        "Action: go to cabinet 1\n"
        "No, the counter is closer.\n"
        "Action: go to countertop 1\n"
        "Actually let me reconsider once more, because Action: go to sink"
    )
    parsed = parse_react_response(text)
    # the trailing line has no newline: it was cut off, so it is not trusted
    assert parsed.action == "go to countertop 1"


def test_a_clean_response_is_not_marked_as_recovered() -> None:
    parsed = parse_react_response("Thought: inspect\nAction: look")
    assert parsed.valid and not parsed.recovered


def test_a_response_with_no_action_at_all_stays_invalid() -> None:
    parsed = parse_react_response("I am thinking about what to do next and then")
    assert not parsed.valid and not parsed.recovered and parsed.action == ""


def test_the_history_is_trimmed_to_fit_the_server_context() -> None:
    """An untrimmed ReAct prompt carries the whole episode and eventually 400s.

    42 of Qwen's 140 episodes died that way at a 50-step budget, all of them
    late, which also bent the length distribution they were then measured by.
    """
    agent = ReActAgent(
        base_url="http://unused", api_key="unused", model="test",
        client=SimpleNamespace(), max_tokens=100, context_limit=1000,
    )
    history = [
        {"thought": "t" * 200, "action": "look", "observation": "o" * 200}
        for _ in range(40)
    ]
    prompt = agent._prompt("put a mug on the desk", history, ["look"])

    assert len(prompt) < 40 * 400          # most of the history is gone
    assert "earlier steps omitted" in prompt
    assert "put a mug on the desk" in prompt   # task and actions always survive
    assert "- look" in prompt


def test_without_a_limit_the_whole_history_is_sent() -> None:
    agent = ReActAgent(
        base_url="http://unused", api_key="unused", model="test",
        client=SimpleNamespace(),
    )
    history = [{"thought": "t", "action": "look", "observation": "o"} for _ in range(30)]
    prompt = agent._prompt("task", history, ["look"])
    assert prompt.count("Action: look") == 30
    assert "omitted" not in prompt


def test_give_up_is_a_valid_action_but_not_an_environment_one() -> None:
    """Without an exit, every ReAct failure runs to the budget.

    Episode length alone then scores PRR .948 on gpt-oss and 1.000 on Qwen,
    and no trajectory signal can be measured against that. smolagents has
    final_answer; this is the same affordance for the same protocol.
    """
    import random
    from agents.react_agent import GIVE_UP_ACTION, resolve_action

    action, valid, reason = resolve_action(
        "give up", ["look", "go to desk 1"], [], rng=random.Random(0),
        repeat_action_limit=2,
    )
    assert action == GIVE_UP_ACTION and valid and reason is None

    # trailing punctuation is handled like any other action
    action, valid, _ = resolve_action(
        "Give up.", ["look"], [], rng=random.Random(0), repeat_action_limit=2,
    )
    assert action == GIVE_UP_ACTION and valid


def test_the_exit_is_offered_only_when_enabled() -> None:
    def prompt_of(**kwargs):
        agent = ReActAgent(
            base_url="http://unused", api_key="unused", model="test",
            client=SimpleNamespace(), **kwargs
        )
        return agent._system_prompt()

    assert "give up" not in prompt_of()
    assert "give up" in prompt_of(allow_give_up=True)
    # it composes with the other two switches rather than replacing them
    both = prompt_of(allow_give_up=True, judge_tool_budget=3, verbalized=True)
    assert "give up" in both and "check progress" in both and "Confidence:" in both


def test_a_refused_prompt_is_retried_with_half_the_history() -> None:
    """Any character estimate of a token count can be wrong; ours was.

    39 of Qwen's episodes died on "maximum context length" with the history
    never trimmed once, because the budget was computed at 3.5 characters per
    token and the real text ran closer to 2.5. Reacting to the refusal itself
    cannot misjudge the ratio.
    """
    class _Refusing:
        """Refuses until the prompt is short enough, then answers."""

        def __init__(self, limit_chars):
            self.limit = limit_chars
            self.calls = []

        def create(self, **kwargs):
            prompt = kwargs["messages"][1]["content"]
            self.calls.append(len(prompt))
            if len(prompt) > self.limit:
                raise RuntimeError(
                    "Error code: 400 - This model's maximum context length is "
                    "131072 tokens. However, your prompt contains more."
                )
            token_items = [SimpleNamespace(token=t, logprob=-0.1)
                           for t in ["Thought:", " go", "\n", "Action:", " look"]]
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="Thought: go\nAction: look"),
                    logprobs=SimpleNamespace(content=token_items))],
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            )

    completions = _Refusing(limit_chars=3000)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    agent = ReActAgent(
        base_url="http://unused", api_key="unused", model="test", client=client,
        max_retries=0,
    )
    history = [
        {"thought": "t" * 100, "action": "look", "observation": "o" * 100}
        for _ in range(60)
    ]
    result = agent.act("put a mug on the desk", history, ["look"])

    assert result.action == "look"                 # it recovered
    assert len(completions.calls) > 1              # and only after shrinking
    assert completions.calls[-1] < completions.calls[0]
