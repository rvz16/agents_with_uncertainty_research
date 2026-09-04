from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import OpenAI

from uq.perplexity import perplexity
from uq.seqprob import mean_token_log_probability, sequence_probability, sum_log_probability
from uq.verbalized import parse_verbalized_confidence


SYSTEM_PROMPT = """You are a text-only household agent. Solve the task by reasoning
briefly and selecting exactly one action from the admissible action list.
Return exactly:
Thought: <brief reasoning>
Action: <one admissible action>
Do not add any other fields or formatting."""


class AgentError(RuntimeError):
    """Raised when the model request cannot be completed."""


class Agent(Protocol):
    def act(
        self,
        task: str,
        history: list[dict[str, str]],
        admissible_actions: list[str],
    ) -> "AgentGeneration": ...


@dataclass
class ParsedResponse:
    thought: str
    action: str
    valid: bool
    thought_span: tuple[int, int] | None = None
    action_span: tuple[int, int] | None = None


@dataclass
class AgentGeneration:
    thought: str
    action: str
    proposed_action: str
    raw_text: str
    format_valid: bool
    action_valid: bool
    fallback_reason: str | None
    token_logprobs: list[dict[str, Any]] = field(default_factory=list)
    uq: dict[str, dict[str, float | int | None]] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    logprobs_available: bool = False
    provider: str | None = None
    request_attempts: int = 1
    empty_response_retries: int = 0
    generation_token_limit: int = 0


def parse_react_response(text: str) -> ParsedResponse:
    thought_match = re.search(
        r"(?ims)^\s*Thought:\s*(.*?)(?=^\s*Action:\s*)", text
    )
    action_match = re.search(r"(?im)^\s*Action:\s*([^\r\n]+)", text)
    if not thought_match or not action_match:
        return ParsedResponse(thought=text.strip(), action="", valid=False)

    thought = thought_match.group(1).strip()
    action = action_match.group(1).strip().strip("`\"'")
    return ParsedResponse(
        thought=thought,
        action=action,
        valid=bool(thought and action),
        thought_span=thought_match.span(1),
        action_span=action_match.span(1),
    )


def _metric_bundle(logprobs: list[float]) -> dict[str, float | int | None]:
    return {
        "num_tokens": len(logprobs),
        "perplexity": perplexity(logprobs),
        "sum_logprob": sum_log_probability(logprobs),
        "mean_token_logprob": mean_token_log_probability(logprobs),
        "sequence_probability": sequence_probability(logprobs),
    }


def token_offsets(
    raw_text: str, token_records: list[dict[str, Any]]
) -> list[tuple[int, int, float]]:
    """Locate every scored token inside the response text.

    Providers may omit tokens from the logprobs array -- OpenRouter/Novita
    drops a few percent of them -- and simply concatenating what survives then
    shifts every later token to the left, which silently mis-assigns the
    thought/action segments. Each token is searched for from the end of the
    previous one instead, so a gap costs only the dropped token.
    """
    offsets: list[tuple[int, int, float]] = []
    cursor = 0
    for record in token_records:
        token = str(record["token"])
        start = raw_text.find(token, cursor) if token else -1
        if start < 0:
            # Not locatable (dropped or rewritten by the provider): keep it
            # zero-width so it counts in `combined` but in no segment.
            start = end = cursor
        else:
            end = start + len(token)
        offsets.append((start, end, float(record["logprob"])))
        cursor = end
    return offsets


def metrics_by_span(
    raw_text: str,
    token_records: list[dict[str, Any]],
    spans: dict[str, tuple[int, int] | None],
) -> dict[str, dict[str, float | int | None]]:
    """UQ bundles for character spans of the response text.

    `combined` always covers the whole generation; every other segment is the
    subset of tokens overlapping its span. Shared with the smolagents policy,
    which segments a code-action response instead of a ReAct one.
    """
    offsets = token_offsets(raw_text, token_records)

    def select(span: tuple[int, int] | None) -> list[float]:
        if span is None:
            return []
        start, end = span
        return [lp for left, right, lp in offsets if right > start and left < end]

    bundles = {name: _metric_bundle(select(span)) for name, span in spans.items()}
    bundles["combined"] = _metric_bundle(
        [float(record["logprob"]) for record in token_records]
    )
    return bundles


def _segment_logprobs(
    raw_text: str,
    parsed: ParsedResponse,
    token_records: list[dict[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
    return metrics_by_span(
        raw_text,
        token_records,
        {"thought": parsed.thought_span, "action": parsed.action_span},
    )


def _extract_token_records(response: Any) -> list[dict[str, Any]]:
    try:
        content = response.choices[0].logprobs.content
    except (AttributeError, IndexError, TypeError):
        return []
    if not content:
        return []
    records = []
    for item in content:
        token = getattr(item, "token", None)
        logprob = getattr(item, "logprob", None)
        if token is not None and logprob is not None:
            records.append({"token": token, "logprob": float(logprob)})
    return records


def _usage(response: Any, name: str) -> int:
    value = getattr(getattr(response, "usage", None), name, 0)
    return int(value or 0)


def is_transient_error(exc: Exception) -> bool:
    """Whether an endpoint failure is worth retrying rather than failing on."""
    status = getattr(exc, "status_code", None)
    return status in {408, 409, 429, 500, 502, 503, 504, 529} or type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }


def resolve_action(
    proposed: str,
    admissible: list[str],
    history: list[dict[str, str]],
    *,
    rng: random.Random,
    repeat_action_limit: int,
) -> tuple[str, bool, str | None]:
    """Map a proposed action onto the admissible set, with the repeat fallback.

    Returns (action, action_valid, fallback_reason). Shared by every policy so
    the mechanical critics mean the same thing across runs.
    """
    by_lower = {action.lower(): action for action in admissible}
    action = by_lower.get(proposed.strip().lower())
    if action is None:
        fallback = by_lower.get("look") or (admissible[0] if admissible else "look")
        return fallback, False, "inadmissible_action"

    recent = [item["action"] for item in history[-repeat_action_limit:]]
    if len(recent) == repeat_action_limit and all(
        previous == action for previous in recent
    ):
        alternatives = [candidate for candidate in admissible if candidate != action]
        if alternatives:
            return rng.choice(alternatives), True, "repeated_action"
    return action, True, None


class ReActAgent:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        max_tokens: int = 1024,
        request_logprobs: bool = True,
        repeat_action_limit: int = 2,
        seed: int = 0,
        extra_body: dict[str, Any] | None = None,
        max_empty_response_retries: int = 1,
        client: Any | None = None,
    ) -> None:
        self.client = client or OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
        )
        self.model = model
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.request_logprobs = request_logprobs
        self.repeat_action_limit = max(1, repeat_action_limit)
        self.rng = random.Random(seed)
        self.extra_body = extra_body
        self.max_empty_response_retries = max(0, max_empty_response_retries)

    @staticmethod
    def _prompt(
        task: str,
        history: list[dict[str, str]],
        admissible_actions: list[str],
    ) -> str:
        transcript = []
        for item in history:
            transcript.extend(
                [
                    f"Thought: {item['thought']}",
                    f"Action: {item['action']}",
                    f"Observation: {item['observation']}",
                ]
            )
        history_text = "\n".join(transcript) if transcript else "(no previous steps)"
        actions = "\n".join(f"- {action}" for action in admissible_actions)
        return (
            f"Task: {task}\n\nHistory:\n{history_text}\n\n"
            f"Admissible actions:\n{actions}\n\nChoose the next action."
        )

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        return is_transient_error(exc)

    @staticmethod
    def _logprobs_unsupported(exc: Exception) -> bool:
        message = str(exc).lower()
        return getattr(exc, "status_code", None) == 400 and "logprob" in message

    def _request(self, messages: list[dict[str, str]]) -> tuple[Any, dict[str, int]]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        if self.request_logprobs:
            kwargs["logprobs"] = True
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        last_error: Exception | None = None
        transient_attempt = 0
        request_attempts = 0
        empty_retries = 0
        discarded_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        while True:
            request_attempts += 1
            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as exc:  # OpenAI-compatible endpoints vary in errors.
                last_error = exc
                if "logprobs" in kwargs and self._logprobs_unsupported(exc):
                    kwargs.pop("logprobs")
                    continue
                if not self._is_transient(exc) or transient_attempt >= self.max_retries:
                    break
                time.sleep(min(2**transient_attempt, 8))
                transient_attempt += 1
                continue

            raw_text = response.choices[0].message.content or ""
            if not raw_text.strip() and empty_retries < self.max_empty_response_retries:
                for name in discarded_usage:
                    discarded_usage[name] += _usage(response, name)
                empty_retries += 1
                kwargs["max_tokens"] = int(kwargs["max_tokens"]) * 2
                continue
            metadata = {
                "request_attempts": request_attempts,
                "empty_response_retries": empty_retries,
                "generation_token_limit": int(kwargs["max_tokens"]),
                "prompt_tokens": discarded_usage["prompt_tokens"]
                + _usage(response, "prompt_tokens"),
                "completion_tokens": discarded_usage["completion_tokens"]
                + _usage(response, "completion_tokens"),
                "total_tokens": discarded_usage["total_tokens"]
                + _usage(response, "total_tokens"),
            }
            return response, metadata
        raise AgentError(f"LLM request failed: {type(last_error).__name__}: {last_error}")

    def _resolve_action(
        self,
        proposed: str,
        admissible: list[str],
        history: list[dict[str, str]],
    ) -> tuple[str, bool, str | None]:
        return resolve_action(
            proposed,
            admissible,
            history,
            rng=self.rng,
            repeat_action_limit=self.repeat_action_limit,
        )

    def act(
        self,
        task: str,
        history: list[dict[str, str]],
        admissible_actions: list[str],
    ) -> AgentGeneration:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._prompt(task, history, admissible_actions),
            },
        ]
        response, request_metadata = self._request(messages)
        raw_text = response.choices[0].message.content or ""
        parsed = parse_react_response(raw_text)
        action, action_valid, fallback_reason = self._resolve_action(
            parsed.action, admissible_actions, history
        )
        if not parsed.valid and fallback_reason is None:
            fallback_reason = "invalid_format"

        token_records = _extract_token_records(response)
        uq = _segment_logprobs(raw_text, parsed, token_records)
        verbalized = parse_verbalized_confidence(raw_text)
        for segment in uq.values():
            segment["verbalized_confidence"] = verbalized

        return AgentGeneration(
            thought=parsed.thought or "Model response did not match the ReAct format.",
            action=action,
            proposed_action=parsed.action,
            raw_text=raw_text,
            format_valid=parsed.valid,
            action_valid=action_valid,
            fallback_reason=fallback_reason,
            token_logprobs=token_records,
            uq=uq,
            prompt_tokens=request_metadata["prompt_tokens"],
            completion_tokens=request_metadata["completion_tokens"],
            total_tokens=request_metadata["total_tokens"],
            logprobs_available=bool(token_records),
            provider=str(
                getattr(response, "provider", None)
                or (getattr(response, "model_extra", None) or {}).get("provider")
                or ""
            )
            or None,
            request_attempts=request_metadata["request_attempts"],
            empty_response_retries=request_metadata["empty_response_retries"],
            generation_token_limit=request_metadata["generation_token_limit"],
        )


class RandomAdmissibleAgent:
    """Offline smoke-test policy using the same trajectory schema."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(
        self,
        task: str,
        history: list[dict[str, str]],
        admissible_actions: list[str],
    ) -> AgentGeneration:
        action = self.rng.choice(admissible_actions) if admissible_actions else "look"
        return AgentGeneration(
            thought="Offline smoke-test policy selected an admissible action.",
            action=action,
            proposed_action=action,
            raw_text=f"Thought: Offline smoke-test policy.\nAction: {action}",
            format_valid=True,
            action_valid=True,
            fallback_reason=None,
            uq={
                segment: _metric_bundle([])
                for segment in ("thought", "action", "combined")
            },
        )
