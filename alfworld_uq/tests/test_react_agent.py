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
