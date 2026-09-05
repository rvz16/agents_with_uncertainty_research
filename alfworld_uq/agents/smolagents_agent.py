"""smolagents CodeAgent policy for text-only ALFWorld.

The ReAct policy is a per-step `act()` and the runner owns the environment
loop. smolagents owns the loop instead: the model writes Python code that calls
`take_action(...)`, so one generation can produce zero, one, or several
environment steps. Trajectory rows stay one per *generation*, because that is
the unit the UQ analysis consumes -- `env_actions` records what the generation
actually did in the environment.

Token log-probabilities are the reason this is not a plain smolagents run:
`OpenAIServerModel` drops them, so the model here is subclassed to request
`logprobs=True` and keep the raw response of every call.
"""
from __future__ import annotations

import ast
import random
import re
import time
from dataclasses import dataclass
from typing import Any

from agents.react_agent import (
    _extract_token_records,
    _metric_bundle,
    _usage,
    is_transient_error,
    metrics_by_span,
    resolve_action,
    split_reasoning_tokens,
)
from uq.verbalized import parse_verbalized_confidence


AGENT_NAME = "smolagents"

# smolagents 1.26 defaults to <code> tags; markdown fences are accepted too so
# the segmentation survives a model that ignores the tag instruction. The
# closing tag is a stop sequence, so the stored response usually ends without
# it -- end-of-text closes the block, exactly as the framework's own parser
# does when it re-appends the missing tag.
_CODE_BLOCK = re.compile(
    r"(?s)(?:<code>|```(?:python|py)?[ \t]*\n)(.*?)(?:</code>|\n```|\Z)"
)
_FINAL_ANSWER = re.compile(r"\bfinal_answer\s*\(")

# smolagents stops generation on ["Observation:", "Calling tools:", <close tag>].
# The two prose markers are words a reasoning model uses while it thinks, and a
# hit inside hidden reasoning ends the turn with empty content -- with gpt-oss
# that silently wasted a quarter of all generations. Only the framework's own
# code-tag rule is kept.
#
# Markdown fences are the default action format here: gpt-oss-20b frequently
# answers with a bare "Thought:" line and no <code> block at all, which the
# framework charges as a malformed step.
DEFAULT_CODE_BLOCK_TAGS = "markdown"


class _EpisodeComplete(BaseException):
    """Unwind the framework loop the moment the episode is over.

    smolagents catches `Exception` around tool execution and turns it into an
    observation, so only a `BaseException` stops the agent immediately instead
    of letting it spend more generations on a finished episode.
    """


@dataclass
class _EnvStep:
    proposed_action: str
    action: str
    action_valid: bool
    fallback_reason: str | None
    observation: str
    progress: float | None
    done: bool
    won: bool


@dataclass
class EpisodeResult:
    records: list[dict[str, Any]]
    final_success: bool
    stop_reason: str
    total_tokens: int


def _response_spans(
    raw_text: str,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None, str]:
    """Split a code-action response into (thought span, code span, code text)."""
    match = _CODE_BLOCK.search(raw_text)
    if match is not None:
        thought_span = (0, match.start()) if match.start() > 0 else None
        return thought_span, match.span(1), match.group(1)
    # smolagents also accepts a bare code blob when the whole response parses
    # as Python; mirror that so such a generation is not counted as malformed.
    if raw_text.strip():
        try:
            ast.parse(raw_text)
        except SyntaxError:
            return (0, len(raw_text)), None, ""
        return None, (0, len(raw_text)), raw_text
    return None, None, ""


class _EnvSession:
    """Environment state shared between the tool and the recording model."""

    def __init__(
        self,
        env: Any,
        initial: Any,
        *,
        max_steps: int,
        repeat_action_limit: int = 2,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.max_steps = max_steps
        self.repeat_action_limit = max(1, repeat_action_limit)
        self.rng = random.Random(seed)
        self.observation = initial.observation
        self.admissible = list(initial.admissible_actions)
        self.history: list[dict[str, str]] = []
        self.pending: list[_EnvStep] = []
        self.env_steps = 0
        self.done = False
        self.won = False
        self.progress: float | None = None

    def render_actions(self) -> str:
        return "\n".join(f"- {action}" for action in self.admissible) or "- (none)"

    def take(self, proposed: str) -> str:
        if self.done:
            raise _EpisodeComplete
        action, action_valid, fallback_reason = resolve_action(
            str(proposed),
            self.admissible,
            self.history,
            rng=self.rng,
            repeat_action_limit=self.repeat_action_limit,
        )
        result = self.env.step(action)
        self.env_steps += 1
        step = _EnvStep(
            proposed_action=str(proposed),
            action=action,
            action_valid=action_valid,
            fallback_reason=fallback_reason,
            observation=result.observation,
            progress=result.progress,
            done=bool(result.done),
            won=bool(result.won),
        )
        self.pending.append(step)
        self.history.append({"action": action, "observation": result.observation})
        self.observation = result.observation
        self.admissible = list(result.admissible_actions)
        self.progress = result.progress
        if step.done or self.env_steps >= self.max_steps:
            self.done = True
            self.won = step.won
            raise _EpisodeComplete
        return self._render(step)

    def _render(self, step: _EnvStep) -> str:
        lines = [f"Observation: {step.observation}"]
        if step.fallback_reason == "inadmissible_action":
            lines.append(
                f"[warning] {step.proposed_action!r} is not admissible; "
                f"the environment ran {step.action!r} instead."
            )
        elif step.fallback_reason == "repeated_action":
            lines.append(
                f"[warning] {step.action!r} was repeated; "
                "the environment ran a different admissible action instead."
            )
        lines.append(f"Steps used: {self.env_steps}/{self.max_steps}")
        lines.append(f"Admissible actions:\n{self.render_actions()}")
        return "\n".join(lines)

    def drain(self) -> list[_EnvStep]:
        pending, self.pending = self.pending, []
        return pending


def _build_tool(session: _EnvSession) -> Any:
    from smolagents import Tool

    class TakeAction(Tool):
        name = "take_action"
        description = (
            "Execute exactly one admissible action in the household environment and "
            "return the resulting observation together with the new admissible "
            "action list. The admissible list changes after every action."
        )
        inputs = {
            "action": {
                "type": "string",
                "description": "One action string copied verbatim from the admissible list.",
            }
        }
        output_type = "string"

        def forward(self, action: str) -> str:
            return session.take(action)

    return TakeAction()


def _provider_of(response: Any) -> str | None:
    provider = (
        getattr(response, "provider", None)
        or (getattr(response, "model_extra", None) or {}).get("provider")
        or ""
    )
    return str(provider) or None


class SmolagentsPolicy:
    """Runs whole ALFWorld episodes through a smolagents `CodeAgent`."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        request_logprobs: bool = True,
        repeat_action_limit: int = 2,
        seed: int = 0,
        extra_body: dict[str, Any] | None = None,
        agent_max_steps: int = 30,
        empty_response_retries: int = 1,
        stop_sequences: list[str] | None = None,
        code_block_tags: str | None = DEFAULT_CODE_BLOCK_TAGS,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_logprobs = request_logprobs
        self.repeat_action_limit = repeat_action_limit
        self.seed = seed
        self.extra_body = extra_body
        self.agent_max_steps = max(1, agent_max_steps)
        self.empty_response_retries = max(0, empty_response_retries)
        # None derives the framework's own code-tag rule; [] removes stops.
        self.stop_sequences = stop_sequences
        self.code_block_tags = code_block_tags

    # -- model ---------------------------------------------------------------

    def _build_model(
        self, session: _EnvSession, generations: list[dict[str, Any]]
    ) -> Any:
        from smolagents import OpenAIServerModel

        policy = self

        class RecordingModel(OpenAIServerModel):
            """`OpenAIServerModel` that keeps the raw response of every call."""

            def generate(self, messages, **kwargs):  # type: ignore[override]
                # Environment steps produced since the previous call belong to
                # the generation that requested them.
                if generations:
                    generations[-1]["env_steps"].extend(session.drain())
                entry: dict[str, Any] = {
                    "raw_text": "",
                    "token_records": [],
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "provider": None,
                    "admissible": list(session.admissible),
                    "observation": session.observation,
                    "progress": session.progress,
                    "env_steps": [],
                    "latency_seconds": 0.0,
                    "request_attempts": 1,
                    "empty_response_retries": 0,
                    "generation_token_limit": policy.max_tokens,
                }
                started = time.monotonic()
                # gpt-oss style models can spend the whole budget on hidden
                # reasoning and return empty content, which smolagents would
                # charge as a wasted step. Retry with a doubled limit, exactly
                # as the ReAct policy does, and keep the discarded usage.
                limit = int(self.kwargs.get("max_tokens") or policy.max_tokens)
                discarded = {"prompt": 0, "completion": 0, "total": 0}
                attempts = 0
                empty_retries = 0
                try:
                    while True:
                        attempts += 1
                        # smolagents turns any exception here into a dead
                        # episode, and a long ALFWorld episode meets enough
                        # transient endpoint failures to lose several that way.
                        for transient in range(policy.max_retries + 1):
                            try:
                                message = super().generate(messages, **kwargs)
                                break
                            except Exception as exc:
                                if (
                                    not is_transient_error(exc)
                                    or transient >= policy.max_retries
                                ):
                                    raise
                                time.sleep(min(2**transient, 8))
                        response = getattr(message, "raw", None)
                        if (message.content or "").strip():
                            break
                        if empty_retries >= policy.empty_response_retries:
                            break
                        discarded["prompt"] += _usage(response, "prompt_tokens")
                        discarded["completion"] += _usage(response, "completion_tokens")
                        discarded["total"] += _usage(response, "total_tokens")
                        empty_retries += 1
                        limit *= 2
                        self.kwargs["max_tokens"] = limit
                finally:
                    self.kwargs["max_tokens"] = policy.max_tokens

                usage = getattr(message, "token_usage", None)
                kept_prompt = _usage(response, "prompt_tokens") or int(
                    getattr(usage, "input_tokens", 0) or 0
                )
                kept_completion = _usage(response, "completion_tokens") or int(
                    getattr(usage, "output_tokens", 0) or 0
                )
                entry.update(
                    raw_text=message.content or "",
                    token_records=_extract_token_records(response),
                    prompt_tokens=discarded["prompt"] + kept_prompt,
                    completion_tokens=discarded["completion"] + kept_completion,
                    provider=_provider_of(response),
                    latency_seconds=time.monotonic() - started,
                    request_attempts=attempts,
                    empty_response_retries=empty_retries,
                    generation_token_limit=limit,
                )
                entry["total_tokens"] = discarded["total"] + (
                    _usage(response, "total_tokens") or (kept_prompt + kept_completion)
                )
                generations.append(entry)
                return message

        completion_kwargs: dict[str, Any] = {
            "temperature": policy.temperature,
            "max_tokens": policy.max_tokens,
        }
        if policy.request_logprobs:
            completion_kwargs["logprobs"] = True
        if policy.extra_body:
            completion_kwargs["extra_body"] = policy.extra_body
        return RecordingModel(
            model_id=policy.model,
            api_base=policy.base_url,
            api_key=policy.api_key,
            client_kwargs={"timeout": policy.timeout, "max_retries": policy.max_retries},
            **completion_kwargs,
        )

    # -- episode -------------------------------------------------------------

    @staticmethod
    def _task_prompt(initial: Any, session: _EnvSession) -> str:
        return (
            "You are a household agent acting in a text-only ALFWorld environment.\n\n"
            f"Task: {initial.task}\n\n"
            f"Initial observation:\n{initial.observation}\n\n"
            f"Admissible actions right now:\n{session.render_actions()}\n\n"
            "Rules:\n"
            '- Call take_action("<action>") with exactly ONE admissible action, '
            "copied verbatim from the admissible list.\n"
            "- Make exactly ONE take_action call per code block, then stop and read "
            "the result. The room is only partially observable: the next action "
            "depends on what the previous one revealed, so a script of several "
            "actions written in advance is guesswork.\n"
            "- take_action returns the new observation and the new admissible list; "
            "the list changes after every action, so never reuse a stale one.\n"
            f"- The environment allows at most {session.max_steps} actions in this "
            "episode and ends the episode by itself once the task is solved or the "
            "budget runs out.\n"
            '- Call final_answer("<summary>") only if you decide to give up: it '
            "ends the episode immediately, and the task is graded by the "
            "environment, not by what you claim."
        )

    def run_episode(self, env: Any, initial: Any, max_steps: int) -> EpisodeResult:
        from smolagents import CodeAgent
        from smolagents.monitoring import LogLevel

        session = _EnvSession(
            env,
            initial,
            max_steps=max_steps,
            repeat_action_limit=self.repeat_action_limit,
            seed=self.seed,
        )
        generations: list[dict[str, Any]] = []
        model = self._build_model(session, generations)
        agent = CodeAgent(
            tools=[_build_tool(session)],
            model=model,
            max_steps=self.agent_max_steps,
            code_block_tags=self.code_block_tags,
            verbosity_level=LogLevel.OFF,
        )
        # `self.kwargs` wins over the per-call arguments, so this replaces the
        # framework's stop sequences rather than adding to them.
        model.kwargs["stop"] = self._stop_sequences_for(agent)

        error: str | None = None
        try:
            agent.run(self._task_prompt(initial, session))
        except _EpisodeComplete:
            pass
        except Exception as exc:  # framework and endpoint errors both land here
            error = f"{type(exc).__name__}: {exc}"
        if generations:
            generations[-1]["env_steps"].extend(session.drain())

        records = self._records(initial, generations, session)
        stop_reason = self._stop_reason(session, generations, error)
        if not records:
            records = [self._error_record(initial, session, error or "no generation")]
        elif error:
            records[-1]["agent_error"] = error
        return EpisodeResult(
            records=records,
            final_success=session.won,
            stop_reason=stop_reason,
            total_tokens=sum(int(gen["total_tokens"]) for gen in generations),
        )

    def _stop_sequences_for(self, agent: Any) -> Any:
        """The framework's code-tag stop rule, without its prose markers."""
        from smolagents.models import REMOVE_PARAMETER

        if self.stop_sequences is not None:
            return list(self.stop_sequences) or REMOVE_PARAMETER
        opening, closing = agent.code_block_tags
        # smolagents skips the close tag when it is a prefix of the open tag
        # (markdown fences), because it would cut the code short.
        return [closing] if closing not in opening else REMOVE_PARAMETER

    @staticmethod
    def _stop_reason(
        session: _EnvSession, generations: list[dict[str, Any]], error: str | None
    ) -> str:
        if session.won:
            return "success"
        if error and not generations:
            return "api_error"
        if session.env_steps >= session.max_steps:
            return "max_steps"
        if session.done:
            return "environment_done"
        if error:
            return "agent_error"
        return "agent_stopped"

    @staticmethod
    def _error_record(
        initial: Any, session: _EnvSession, error: str
    ) -> dict[str, Any]:
        return {
            "episode_id": initial.episode_id,
            "task_type": initial.task_type,
            "task": initial.task,
            "step": 1,
            "thought": "",
            "action": "",
            "observation": session.observation,
            "admissible_actions": list(session.admissible),
            "token_logprobs": [],
            "perplexity": None,
            "seqprob": None,
            "verb": None,
            "progress": None,
            "done": True,
            "final_success": False,
            "error": error,
            "uq": {},
        }

    def _records(
        self, initial: Any, generations: list[dict[str, Any]], session: _EnvSession
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for index, generation in enumerate(generations, start=1):
            raw_text = generation["raw_text"]
            thought_span, action_span, code = _response_spans(raw_text)
            # A locally served model may score its hidden reasoning channel too;
            # keep it out of `combined` and report it as its own target.
            reasoning, content = split_reasoning_tokens(
                raw_text, generation["token_records"]
            )
            uq = metrics_by_span(
                raw_text,
                content,
                {"thought": thought_span, "action": action_span},
            )
            uq["reasoning"] = _metric_bundle(
                [float(record["logprob"]) for record in reasoning]
            )
            verbalized = parse_verbalized_confidence(raw_text)
            for segment in uq.values():
                segment["verbalized_confidence"] = verbalized

            env_steps: list[_EnvStep] = generation["env_steps"]
            format_valid = action_span is not None
            if env_steps:
                fallbacks = [
                    step.fallback_reason for step in env_steps if step.fallback_reason
                ]
                fallback_reason = (
                    "repeated_action"
                    if "repeated_action" in fallbacks
                    else (fallbacks[0] if fallbacks else None)
                )
                action_valid = all(step.action_valid for step in env_steps)
                last = env_steps[-1]
                action = last.action
                proposed_action = last.proposed_action
                observation = last.observation
                progress = last.progress
                done = last.done
            else:
                # A generation that took no environment step is either the
                # agent's own final answer or a wasted turn; both are real
                # generations and stay in the trajectory.
                stopped_deliberately = format_valid and bool(_FINAL_ANSWER.search(code))
                action_valid = stopped_deliberately
                fallback_reason = (
                    None
                    if stopped_deliberately
                    else ("no_env_action" if format_valid else "invalid_format")
                )
                action = ""
                proposed_action = ""
                observation = generation["observation"]
                progress = generation["progress"]
                # No environment transition happened in this generation, so it
                # carries the state it started from -- `session` here holds the
                # end-of-episode state, which would mark middle rows done.
                done = False

            combined = uq["combined"]
            records.append(
                {
                    "episode_id": initial.episode_id,
                    "task_type": initial.task_type,
                    "task": initial.task,
                    "step": index,
                    "thought": raw_text[slice(*thought_span)].strip()
                    if thought_span
                    else "",
                    "action": action,
                    "proposed_action": proposed_action,
                    "observation": observation,
                    "admissible_actions": generation["admissible"],
                    "token_logprobs": generation["token_records"],
                    "perplexity": combined["perplexity"],
                    "seqprob": combined["sequence_probability"],
                    "verb": combined["verbalized_confidence"],
                    "progress": progress,
                    "done": done,
                    "final_success": False,
                    "format_valid": format_valid,
                    "action_valid": action_valid,
                    "fallback_reason": fallback_reason,
                    "raw_response": raw_text,
                    "logprobs_available": bool(content),
                    "provider": generation["provider"],
                    "uq": uq,
                    "usage": {
                        "prompt_tokens": generation["prompt_tokens"],
                        "completion_tokens": generation["completion_tokens"],
                        "total_tokens": generation["total_tokens"],
                        "request_attempts": generation["request_attempts"],
                        "empty_response_retries": generation[
                            "empty_response_retries"
                        ],
                        "generation_token_limit": generation[
                            "generation_token_limit"
                        ],
                    },
                    "code_action": code,
                    "reasoning_tokens": len(reasoning),
                    "content_tokens": len(content),
                    "env_actions": [step.action for step in env_steps],
                    "env_action_count": len(env_steps),
                    "latency_seconds": generation["latency_seconds"],
                    "agent_error": None,
                }
            )
        return records
