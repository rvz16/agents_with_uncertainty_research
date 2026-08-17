#!/usr/bin/env python3
"""LLM-controlled tool agent for LiveCodeBench.

This is intentionally small: by default, the SAGE-Agent LangGraph pipeline
selects one tool action at a time, and the actual benchmark tools are the
existing LCB adapter methods used by the fitted-controller experiments.

Example:
    python different_agents/v4/lcb_llm_tool_agent.py \
      --benchmark lcb_easy \
      --generator haiku45 \
      --n-instances 5 \
      --output experiments/orchestration_hypothesis_testing/sim_results/lcb_agent.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = PROJECT_ROOT
ORCH_ROOT = PACKAGE_ROOT

from code_uq.common.cost import cost_for_call
from code_uq.common.generators import GENERATORS, _make_client, canonical_generator_key
from code_uq.environments.fitted_live.common import (
    BenchmarkAdapter,
    Candidate,
    CriticResult,
    VerifyResult,
    safe_stem,
)
from code_uq.environments.fitted_live.function_adapters import make_function_adapter
from code_uq.environments.fitted_live.swe_adapter import make_swe_adapter
from code_uq.sage_agent import ExecutionResult, ToolCall, ToolCallCandidate, ToolSchema
from code_uq.sage_agent.langgraph import SAGEGraph, SAGEGraphConfig

ACTION_SPACE: tuple[str, ...] = (
    "generate",
    "critic_L0",
    "critic_L1",
    "critic_L2",
    "critic_L3",
    "verify",
    "think",
    "finish",
)
VALID_ACTIONS: set[str] = set(ACTION_SPACE)

CRITIC_ACTIONS = {
    "critic_L0": "L0",
    "critic_L1": "L1",
    "critic_L2": "L2",
    "critic_L3": "L3",
}

ACTION_ALIASES = {
    "generate": "generate",
    "gen": "generate",
    "regenerate": "generate",
    "revise": "generate",
    "critic_l0": "critic_L0",
    "l0": "critic_L0",
    "syntax": "critic_L0",
    "syntax_check": "critic_L0",
    "critic_l1": "critic_L1",
    "l1": "critic_L1",
    "lint": "critic_L1",
    "critic_l2": "critic_L2",
    "l2": "critic_L2",
    "public_tests": "critic_L2",
    "public_test": "critic_L2",
    "run_public_tests": "critic_L2",
    "critic_l3": "critic_L3",
    "l3": "critic_L3",
    "llm_review": "critic_L3",
    "review": "critic_L3",
    "verify": "verify",
    "verifier": "verify",
    "hidden_tests": "verify",
    "private_tests": "verify",
    "think": "think",
    "finish": "finish",
    "stop": "finish",
}


class AgentState(TypedDict, total=False):
    instance: dict[str, Any]
    instance_id: str
    benchmark: str
    step: int
    max_steps: int
    max_generations: int
    max_verifications: int
    final_verify: bool
    candidate_payload: str
    candidate_raw: str
    candidate_kind: str
    chosen_action: str
    chosen_reasoning: str
    decision_raw: str
    trajectory: list[dict[str, Any]]
    n_decisions: int
    n_generations: int
    n_critic_runs: int
    n_verifications: int
    prompt_tokens: int
    completion_tokens: int
    api_cost_usd: float
    fixed: bool
    #: False when no verifier could produce a terminal label for this episode.
    #: `fixed` is then meaningless and the row must be excluded from any metric
    #: that needs ground truth.
    label_available: bool
    unavailable_actions: tuple[str, ...]
    #: Show the controller what each action does, what evidence it already has
    #: on this candidate, and the prior. Off by default -- the original prompt
    #: says none of this.
    describe_actions: bool
    #: Drop `generate`/`verify` from the action space once their budget is
    #: spent, instead of offering an action that can only refuse.
    hide_exhausted_actions: bool
    #: Withhold `verify` when the oracle has already ruled on this candidate.
    no_repeat_verify: bool
    done: bool
    final_action: str
    error: str
    last_generation_prompt: str
    prior_Y1: float
    prior_calibration_n: int
    prior_calibration_correct: int


@dataclass
class AgentDeps:
    adapter: BenchmarkAdapter
    llm_client: Any
    reviewer_client: Any
    model_id: str
    decision_temperature: float
    generation_temperature: float
    max_tokens_decision: int
    max_tokens_generation: int
    max_code_chars: int
    save_generation_logprobs: bool = False
    require_generation_logprobs: bool = False
    top_logprobs: int = 0
    logprobs_output: Path | None = None


class ContextOverflowError(RuntimeError):
    pass


def _load_env_chain() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    candidates = [REPO_ROOT / ".env", ORCH_ROOT / ".env"]
    cur = REPO_ROOT.parent
    for _ in range(5):
        candidates.append(cur / ".env")
        if cur.parent == cur:
            break
        cur = cur.parent
    for env_path in candidates:
        if env_path.exists() and env_path.stat().st_size > 0:
            load_dotenv(env_path, override=False)


def _usage(resp: Any) -> tuple[int, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0, 0
    return int(getattr(usage, "prompt_tokens", 0) or 0), int(
        getattr(usage, "completion_tokens", 0) or 0
    )


def _dump_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _dump_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_dump_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_dump_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _response_logprobs(resp: Any) -> Any:
    try:
        choice = resp.choices[0]
        logprobs = getattr(choice, "logprobs", None)
    except Exception:
        return None
    return _dump_jsonable(logprobs)


def _message_text_with_source(message: Any) -> tuple[str, str]:
    """Return the message text and where it came from.

    The source matters. Reasoning models emit the answer on a separate channel
    from their thinking (harmony channels for gpt-oss, ``<think>`` tags for
    Qwen3), and when a generation is cut off mid-thought the answer channel is
    simply absent. Falling back to the reasoning text then hands the caller a
    monologue to parse as if it were an action or a patch -- it is not a
    low-confidence answer, it is the absence of one.

    Callers use the source to abstain instead of guessing; see
    :func:`_message_text` for the plain-text shorthand.
    """
    content = getattr(message, "content", None)
    if isinstance(content, str) and content:
        return content, "content"
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        if parts:
            return "\n".join(parts), "content"

    for attr in ("reasoning_content", "reasoning"):
        value = getattr(message, attr, None)
        if isinstance(value, str) and value:
            return value, "reasoning"

    extra = getattr(message, "model_extra", None) or {}
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning"):
            value = extra.get(key)
            if isinstance(value, str) and value:
                return value, "reasoning"

    kwargs = getattr(message, "additional_kwargs", None) or {}
    if isinstance(kwargs, dict):
        for key in ("reasoning_content", "reasoning"):
            value = kwargs.get(key)
            if isinstance(value, str) and value:
                return value, "reasoning"
    return "", "none"


def _message_text(message: Any) -> str:
    return _message_text_with_source(message)[0]


def _finish_reason(resp: Any) -> str:
    try:
        return str(getattr(resp.choices[0], "finish_reason", "") or "")
    except (AttributeError, IndexError):
        return ""


def _message_content_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return "\n".join(parts)
    return ""


def _retry_max_tokens_for_context_error(exc: Exception, requested: int) -> int | None:
    text = str(exc)
    if "max_tokens" not in text and "max_completion_tokens" not in text:
        return None
    if "maximum context length" not in text:
        return None
    match = re.search(
        r"maximum context length is (\d+) tokens and your request has (\d+) input tokens",
        text,
    )
    if not match:
        return None
    available = int(match.group(1)) - int(match.group(2))
    for candidate in (2048, 1024, 512, 256):
        if candidate <= available - 64:
            return min(requested, candidate)
    return None


def _is_context_overflow_error(exc: Exception) -> bool:
    text = str(exc)
    return (
        "maximum context length" in text
        or "max_tokens must be at least 1" in text
        or "max_completion_tokens must be at least 1" in text
    )


def _openrouter_logprobs_provider_routing(client: Any) -> dict[str, Any] | None:
    """OpenRouter provider routing so logprobs requests only hit capable providers.

    Many OpenRouter providers for a given model do not return logprobs. Setting
    ``provider.require_parameters=true`` makes OpenRouter route only to providers
    that support every parameter in the request (here: logprobs/top_logprobs).
    Optionally pin/prefer providers via ``OPENROUTER_PROVIDER_ORDER`` (comma list,
    e.g. "DeepSeek,Fireworks,Parasail"). No-op for non-OpenRouter clients.
    """
    base = str(getattr(client, "base_url", "") or "")
    if "openrouter" not in base:
        return None
    routing: dict[str, Any] = {"require_parameters": True}
    order = os.environ.get("OPENROUTER_PROVIDER_ORDER", "").strip()
    if order:
        routing["order"] = [p.strip() for p in order.split(",") if p.strip()]
        routing["allow_fallbacks"] = (
            os.environ.get("OPENROUTER_ALLOW_FALLBACKS", "1").strip().lower()
            not in {"0", "false", "no", "off"}
        )
    return {"provider": routing}


def _create_completion(deps: AgentDeps, params: dict[str, Any], *, tries: int = 6) -> Any:
    """chat.completions.create with backoff on transient upstream errors.

    OpenRouter frequently returns 429 (provider rate-limited upstream) or 5xx.
    Retry those with exponential backoff instead of crashing a long run.
    Non-transient errors (e.g. 400 context overflow) propagate immediately so
    the caller's context-window handling still applies.
    """
    last: Exception | None = None
    for i in range(tries):
        try:
            return deps.llm_client.chat.completions.create(**params)
        except Exception as exc:  # noqa: BLE001
            last = exc
            status = getattr(exc, "status_code", None)
            name = type(exc).__name__
            transient = status in (429, 500, 502, 503, 529) or name in {
                "RateLimitError", "APITimeoutError", "APIConnectionError",
                "InternalServerError",
            }
            if not transient or i == tries - 1:
                raise
            wait = min(2 ** i, 30)
            print(
                f"transient API error {name} (status={status}); "
                f"retry {i + 1}/{tries} in {wait}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
    assert last is not None
    raise last


def _chat(
    deps: AgentDeps,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    include_logprobs: bool = False,
    content_only: bool = False,
    extra_body: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> tuple[str, int, int, float, Any]:
    """Call the model. When ``meta`` is given it receives ``finish_reason``,
    ``text_source`` and ``truncated`` for the response, so callers can tell a
    real answer from a cut-off one."""
    params: dict[str, Any] = {
        "model": deps.model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if include_logprobs:
        params["logprobs"] = True
        if deps.top_logprobs > 0:
            params["top_logprobs"] = deps.top_logprobs
        routing = _openrouter_logprobs_provider_routing(deps.llm_client)
        if routing:
            extra_body = {**(extra_body or {}), **routing}
    if extra_body:
        params["extra_body"] = extra_body
    try:
        resp = _create_completion(deps, params)
    except Exception as exc:
        retry_max_tokens = _retry_max_tokens_for_context_error(exc, max_tokens)
        if retry_max_tokens is None or retry_max_tokens >= max_tokens:
            if _is_context_overflow_error(exc):
                raise ContextOverflowError(str(exc)) from exc
            raise
        print(
            f"retrying context-limited request with max_tokens={retry_max_tokens} "
            f"(was {max_tokens})",
            file=sys.stderr,
            flush=True,
        )
        params["max_tokens"] = retry_max_tokens
        resp = _create_completion(deps, params)
    logprobs_val = _response_logprobs(resp) if include_logprobs else None
    # OpenRouter can route a logprobs request to a provider that returns an
    # empty logprobs payload (or transiently drops it). Rather than fail the
    # whole run, re-request a couple of times; provider routing keeps us on a
    # logprobs-capable provider.
    if include_logprobs and not logprobs_val:
        for attempt in range(2):
            print(
                f"logprobs missing from response, retrying generation "
                f"({attempt + 1}/2)",
                file=sys.stderr,
                flush=True,
            )
            resp = _create_completion(deps, params)
            logprobs_val = _response_logprobs(resp)
            if logprobs_val:
                break
    if content_only:
        text = _message_content_text(resp.choices[0].message)
        source = "content" if text else "none"
    else:
        text, source = _message_text_with_source(resp.choices[0].message)
    prompt_tokens, completion_tokens = _usage(resp)
    if meta is not None:
        meta["finish_reason"] = _finish_reason(resp)
        meta["text_source"] = source
        meta["truncated"] = meta["finish_reason"] == "length" or source == "reasoning"
    return (
        text,
        prompt_tokens,
        completion_tokens,
        cost_for_call(deps.model_id, prompt_tokens, completion_tokens),
        logprobs_val,
    )


def _candidate_from_state(state: AgentState) -> Candidate | None:
    payload = state.get("candidate_payload") or ""
    if not payload:
        return None
    return Candidate(
        payload=payload,
        raw_text=state.get("candidate_raw", ""),
        kind=state.get("candidate_kind", "code"),
    )


def _adapter_log(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert agent trajectory action names to fitted_live feedback names."""
    out: list[dict[str, Any]] = []
    for item in trajectory:
        action = item.get("action")
        if action in CRITIC_ACTIONS:
            out.append(
                {
                    "action": CRITIC_ACTIONS[action],
                    "passed": item.get("passed"),
                    "detail": item.get("detail") or item.get("observation") or "",
                }
            )
        elif action == "verify":
            out.append(
                {
                    "action": "verify",
                    "passed": item.get("passed"),
                    "detail": item.get("detail") or item.get("observation") or "",
                }
            )
        elif action == "generate":
            out.append({"action": "generate"})
    return out


def _problem_brief(instance: dict[str, Any]) -> str:
    title = str(instance.get("question_title") or "")
    content = str(instance.get("question_content") or "")
    starter = str(instance.get("starter_code") or "")
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if content:
        parts.append(f"Problem:\n{content[:3500]}")
    if starter:
        parts.append(f"Starter code:\n```python\n{starter[:1200]}\n```")
    return "\n\n".join(parts)[:5500]


def _trajectory_brief(trajectory: list[dict[str, Any]]) -> str:
    if not trajectory:
        return "No actions yet."
    rows = []
    for item in trajectory[-10:]:
        action = item.get("action")
        status = ""
        if "passed" in item:
            status = f" passed={item.get('passed')}"
        obs = str(item.get("observation") or item.get("detail") or "")[:300]
        reasoning = str(item.get("reasoning") or "")[:200]
        rows.append(
            f"- step={item.get('step')} action={action}{status} "
            f"reason={reasoning!r} observation={obs!r}"
        )
    return "\n".join(rows)


VERBALIZED_2S_CONFIDENCE_PROMPT = (
    "Provide the probability that your guess is correct. Give ONLY the probability, "
    "no other words or explanation.\n\nFor example:\n\nProbability: <the probability "
    "between 0.0 and 1.0 that your guess is correct, without any extra commentary "
    "whatsoever; just the probability!>"
)


def _parse_verbalized_2s_confidence(text: str) -> float | None:
    patterns = [
        r"confidence\s*[:=]\s*((?:[0-9]+(?:\.[0-9]+)?)|(?:\.[0-9]+))\s*(%)?",
        r"probability\s*[:=]\s*((?:[0-9]+(?:\.[0-9]+)?)|(?:\.[0-9]+))\s*(%)?",
        r"certainty\s*[:=]\s*((?:[0-9]+(?:\.[0-9]+)?)|(?:\.[0-9]+))\s*(%)?",
        r"\b((?:[0-9]+(?:\.[0-9]+)?)|(?:\.[0-9]+))\s*%",
        r"^\s*((?:0(?:\.[0-9]+)?)|(?:1(?:\.0+)?)|(?:\.[0-9]+))\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            value = float(match.group(1))
        except (TypeError, ValueError):
            continue
        has_percent = len(match.groups()) >= 2 and bool(match.group(2))
        if has_percent or value > 1.0:
            value /= 100.0
        return max(0.0, min(1.0, value))
    return None


def run_verbalized_2s(
    state: AgentState,
    deps: AgentDeps,
    *,
    temperature: float,
    max_tokens: int,
    require_parse: bool,
) -> dict[str, Any]:
    """LM-Polygraph-style Verbalized2S: ask confidence after the answer."""
    candidate = _candidate_from_state(state)
    if candidate is None:
        return {
            "verbalized_2s_confidence": None,
            "verbalized_2s_uncertainty": None,
            "verbalized_2s_raw": "",
            "verbalized_2s_error": "no_candidate",
            "verbalized_2s_prompt_tokens": 0,
            "verbalized_2s_completion_tokens": 0,
            "verbalized_2s_api_cost_usd": 0.0,
        }

    input_prompt = str(state.get("last_generation_prompt") or "")
    if not input_prompt:
        input_prompt = deps.adapter.build_prompt(state["instance"], None, [])
    input_prompt = input_prompt[:12000]
    answer = candidate.payload[: deps.max_code_chars]

    try:
        raw, pt, ct, cost, _logprobs = _chat(
            deps,
            [
                {"role": "user", "content": input_prompt},
                {"role": "assistant", "content": answer},
                {"role": "user", "content": VERBALIZED_2S_CONFIDENCE_PROMPT},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            content_only=True,
            extra_body={"reasoning_effort": "low"},
        )
        confidence = _parse_verbalized_2s_confidence(raw)
        if confidence is None and require_parse:
            raise RuntimeError(f"could not parse Verbalized2S confidence: {raw[:500]}")
        return {
            "verbalized_2s_confidence": confidence,
            "verbalized_2s_uncertainty": None if confidence is None else 1.0 - confidence,
            "verbalized_2s_raw": raw,
            "verbalized_2s_error": "" if confidence is not None else "parse_failed",
            "verbalized_2s_prompt_tokens": pt,
            "verbalized_2s_completion_tokens": ct,
            "verbalized_2s_api_cost_usd": cost,
        }
    except Exception as exc:
        if require_parse:
            raise
        return {
            "verbalized_2s_confidence": None,
            "verbalized_2s_uncertainty": None,
            "verbalized_2s_raw": "",
            "verbalized_2s_error": f"{type(exc).__name__}: {exc}",
            "verbalized_2s_prompt_tokens": 0,
            "verbalized_2s_completion_tokens": 0,
            "verbalized_2s_api_cost_usd": 0.0,
        }


def verdicts_on_current_candidate(state: AgentState) -> dict[str, str]:
    """PASS/FAIL per critic for the candidate in hand, most recent verdict wins."""
    trajectory = state.get("trajectory") or []
    last_generation = -1
    for index, step in enumerate(trajectory):
        if step.get("action") == "generate" and not step.get("skipped"):
            last_generation = index
    return {
        str(step["action"]): ("PASS" if step["passed"] else "FAIL")
        for step in trajectory[last_generation + 1:]
        if step.get("action") in CRITIC_ACTIONS and isinstance(step.get("passed"), bool)
    }


def verified_current_candidate(state: AgentState) -> bool:
    """Whether the oracle has already ruled on the candidate now in hand.

    Re-verifying unchanged code is deterministic -- same code, same hidden
    tests, same answer -- so the second call buys nothing and costs the most
    expensive action in the space. Measured on a 76-episode run: 55 of 149
    verifier calls were repeats of a candidate that had not changed.
    """
    trajectory = state.get("trajectory") or []
    last_generation = -1
    for index, step in enumerate(trajectory):
        if step.get("action") == "generate" and not step.get("skipped"):
            last_generation = index
    return any(
        step.get("action") in ("verify", "final_verify")
        and isinstance(step.get("passed"), bool)
        for step in trajectory[last_generation + 1:]
    )


def critics_on_current_candidate(state: AgentState) -> set[str]:
    """Distinct critics already run on the candidate now in hand.

    Counted from the last real generation onwards: verdicts recorded before it
    describe a candidate that has since been replaced.
    """
    trajectory = state.get("trajectory") or []
    last_generation = -1
    for index, step in enumerate(trajectory):
        if step.get("action") == "generate" and not step.get("skipped"):
            last_generation = index
    return {
        str(step.get("action"))
        for step in trajectory[last_generation + 1:]
        if step.get("action") in CRITIC_ACTIONS and isinstance(step.get("passed"), bool)
    }


def available_actions(state: AgentState) -> tuple[str, ...]:
    """The action space for this episode, minus what it cannot or should not do.

    Two filters. The first is capability: SWE-Bench has no public-test critic,
    and without a container runtime it has no verifier either. Offering those
    anyway would spend the step budget on actions that can only report their
    own absence.

    The second is policy: with ``min_critics_before_verify`` set, the verifier
    stays out of the action space until that many distinct critics have run on
    the current candidate. The point is to stop the controller from going
    straight to the oracle -- which is what it does by default, leaving the
    critics almost never consulted and the belief state with nothing to
    aggregate. The quota is capped by how many critics this environment
    actually has, so it can never lock the verifier away for good.
    """
    unavailable = set(state.get("unavailable_actions") or ())

    if state.get("hide_exhausted_actions"):
        # An action whose budget is spent can only report that fact, and the
        # report costs a step like any other. Leaving them on the menu is not
        # free: measured over a 76-episode run, 50 of 299 `generate` choices
        # and 13 of 108 `verify` choices were no-ops -- 11% of every step taken.
        if int(state.get("n_generations", 0)) >= int(state.get("max_generations", 0)):
            unavailable = unavailable | {"generate"}
        if int(state.get("n_verifications", 0)) >= int(state.get("max_verifications", 0)):
            unavailable = unavailable | {"verify"}

    if state.get("no_repeat_verify") and verified_current_candidate(state):
        unavailable = unavailable | {"verify"}

    return tuple(action for action in ACTION_SPACE if action not in unavailable)


def _decision_prompt(state: AgentState) -> str:
    candidate_chars = len(state.get("candidate_payload") or "")
    remaining = {
        "steps": state["max_steps"] - state.get("step", 0),
        "generations": state["max_generations"] - state.get("n_generations", 0),
        "verifications": state["max_verifications"] - state.get("n_verifications", 0),
    }
    actions = available_actions(state)

    # Per-tool budgets. `budget_remaining` above is grouped by resource, which
    # leaves the controller to work out for itself that "generations" is the
    # cap on `generate` and that the critics are not capped at all. Spelling it
    # out per action costs nothing and removes the inference.
    calls_left: dict[str, Any] = {}
    for action in actions:
        if action == "generate":
            calls_left[action] = remaining["generations"]
        elif action == "verify":
            calls_left[action] = remaining["verifications"]
        else:
            calls_left[action] = "unlimited"
    action_list = ", ".join(actions)
    action_union = "|".join(actions)

    described = ""
    evidence = ""
    if state.get("describe_actions"):
        # The original prompt names the actions and stops there, so a
        # controller told only "critic_L0" cannot know it is a syntax check or
        # that `verify` runs the hidden tests. Those meanings exist in the
        # codebase only inside ACTION_ALIASES, which parses the model's reply
        # and is never shown to it.
        # Statements of fact only. Calling a critic "the most informative
        # signal" or the verifier "expensive" tells the model which action to
        # prefer, and it obliges -- a pilot with that wording moved the
        # controller onto whichever critic the sentence praised. That is the
        # prompt steering the result rather than measuring the model.
        catalogue = {
            "generate": "produce a new candidate solution, replacing the current one",
            "critic_L0": "check the candidate parses (ast.parse)",
            "critic_L1": "static lint: undefined names, redefinitions",
            "critic_L2": "run the public example tests",
            "critic_L3": "ask a reviewer model for a PASS/FAIL opinion",
            "verify": "run the hidden test suite; the episode ends if it passes",
            "think": "record a note without calling any tool",
            "finish": "stop and submit the current candidate",
        }
        lines = "\n".join(f"- {a}: {catalogue[a]}" for a in actions if a in catalogue)
        described = f"\nWhat each action does:\n{lines}\n"

        # One-verification setup: the controller cannot call the oracle at all,
        # but a terminal check still happens. Left unsaid, the model reads the
        # missing action as "there is no test suite" and submits its first
        # draft. This states the arrangement and nothing else -- no advice on
        # how careful to be, which would be the prompt steering the result.
        if "verify" not in actions and state.get("final_verify"):
            described += (
                "\nThe hidden test suite is not an action you can call. "
                "Whatever candidate you hold when the episode ends is run "
                "against it exactly once, and that single result is final. "
                "You will not see its output.\n"
            )

        # Evidence about the candidate in hand. `_trajectory_brief` shows raw
        # log lines from which the model cannot tell which verdicts describe
        # the current candidate and which describe one already thrown away.
        seen = critics_on_current_candidate(state)
        verdicts = verdicts_on_current_candidate(state)
        not_run = [a for a in actions if a in CRITIC_ACTIONS and a not in seen]
        evidence = (
            f"\nEvidence on the CURRENT candidate: {json.dumps(verdicts)}\n"
            f"verifier_already_ran_on_this_candidate: "
            f"{json.dumps(verified_current_candidate(state))}\n"
            f"critics_not_yet_run: {json.dumps(not_run)}\n"
            f"prior_P(correct) for a fresh candidate: "
            f"{float(state.get('prior_Y1', 0.5)):.3f}\n"
        )

    gate = ""
    quota = int(state.get("min_critics_before_verify") or 0)
    if quota > 0:
        already = sorted(critics_on_current_candidate(state))
        gate = (
            f"\ncritics_run_on_current_candidate: {json.dumps(already)}\n"
            f"verify_unlocks_after: {quota} distinct critics on this candidate\n"
        )

    return f"""Choose one next tool action for a coding episode.

You are only routing tools. Do not solve the programming task and do not write code.

Valid actions:
{action_list}

Return exactly one compact JSON object and nothing else:
{{"action":"{action_union}","reasoning":"short reason"}}

State:
candidate_exists: {bool(candidate_chars)}
candidate_chars: {candidate_chars}
budget_remaining: {json.dumps(remaining)}
calls_left_per_action: {json.dumps(calls_left)}
{described}{evidence}{gate}
Recent trajectory:
{_trajectory_brief(state.get("trajectory", []))}
"""


def _decision_messages(state: AgentState) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a tool-routing controller. Return only valid JSON. "
                "Never solve the coding problem. Never output code."
            ),
        },
        {"role": "user", "content": _decision_prompt(state)},
    ]


def _normalize_action(value: Any) -> str:
    raw = str(value or "").strip().strip("\"'`.,")
    if not raw:
        return ""
    if raw in VALID_ACTIONS:
        return raw
    key = raw.lower()
    key = re.sub(r"^(action|tool|next_action|tool_name)\s*[:=]\s*", "", key)
    key = key.strip().strip("\"'`.,")
    key = key.replace("-", "_").replace(" ", "_").replace(":", "_")
    key = re.sub(r"^critic_?([0-3])$", r"critic_l\1", key)
    return ACTION_ALIASES.get(key, "")


def _action_from_json_obj(obj: Any) -> tuple[str, str]:
    if not isinstance(obj, dict):
        return "", ""
    action = ""
    for key in ("action", "next_action", "tool", "tool_name", "name"):
        action = _normalize_action(obj.get(key))
        if action:
            break
    reasoning = str(obj.get("reasoning") or obj.get("reason") or "").strip()
    return action, reasoning


def _parse_decision(text: str) -> tuple[str, str]:
    if not text.strip():
        return "", "empty decision response"
    decoder = json.JSONDecoder()
    candidates = [text]
    for fence in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
        candidates.insert(0, fence)
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            obj = json.loads(candidate)
            action, reasoning = _action_from_json_obj(obj)
            if action:
                return action, reasoning
        except Exception:
            pass
        for idx, ch in enumerate(candidate):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(candidate[idx:])
            except Exception:
                continue
            action, reasoning = _action_from_json_obj(obj)
            if action:
                return action, reasoning

    for line in text.splitlines()[:12]:
        stripped = line.strip()
        action = _normalize_action(stripped)
        if action:
            return action, f"parsed bare action: {action}"
        match = re.search(
            r"\b(action|tool|next_action|tool_name)\b\s*[:=]\s*[\"'`]?([A-Za-z0-9_: -]+)",
            stripped,
            flags=re.IGNORECASE,
        )
        if match:
            action = _normalize_action(match.group(2))
            if action:
                return action, f"parsed action field: {action}"

    lowered = text.lower()
    for alias, action in sorted(ACTION_ALIASES.items(), key=lambda x: -len(x[0])):
        alias_pattern = re.escape(alias).replace("_", r"[_\s:-]*")
        pattern = r"(?<![a-z0-9_])" + alias_pattern + r"(?![a-z0-9_])"
        if re.search(pattern, lowered):
            return action, f"parsed action mention: {action}"
    return "", "decision parse failed"


def fallback_decision_after_parse_failure(state: AgentState) -> tuple[str, str]:
    allowed = set(available_actions(state))
    if not (state.get("candidate_payload") or ""):
        return "generate", "fallback: decision parse failed and no candidate exists"
    if int(state.get("n_critic_runs", 0)) == 0 and "critic_L2" in allowed:
        return "critic_L2", "fallback: decision parse failed after candidate generation"
    if int(state.get("n_critic_runs", 0)) == 0 and "critic_L0" in allowed:
        return "critic_L0", "fallback: decision parse failed after candidate generation"
    if (
        "verify" in allowed
        and int(state.get("n_verifications", 0)) < int(state.get("max_verifications", 0))
    ):
        return "verify", "fallback: decision parse failed after critics"
    if int(state.get("n_generations", 0)) < int(state.get("max_generations", 0)):
        return "generate", "fallback: decision parse failed after verification budget"
    return "finish", "fallback: decision parse failed and no useful budget remains"


def _merge_usage(state: AgentState, pt: int, ct: int, cost: float) -> dict[str, Any]:
    return {
        "prompt_tokens": int(state.get("prompt_tokens", 0)) + pt,
        "completion_tokens": int(state.get("completion_tokens", 0)) + ct,
        "api_cost_usd": float(state.get("api_cost_usd", 0.0)) + cost,
    }


def decide_action_node(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    if state.get("step", 0) >= state["max_steps"]:
        return {"done": True, "final_action": "max_steps"}

    if not (state.get("candidate_payload") or ""):
        if int(state.get("n_generations", 0)) >= state["max_generations"]:
            return {"done": True, "final_action": "no_candidate"}
        return {
            "chosen_action": "generate",
            "chosen_reasoning": "auto: no candidate exists",
            "decision_raw": "",
        }

    try:
        raw, pt, ct, cost, _logprobs = _chat(
            deps,
            _decision_messages(state),
            temperature=deps.decision_temperature,
            max_tokens=deps.max_tokens_decision,
        )
    except ContextOverflowError as exc:
        return {
            "done": True,
            "fixed": False,
            "final_action": "context_overflow_skip",
            "error": str(exc),
            "trajectory": _record(
                state,
                action="think",
                reasoning="context overflow",
                observation="skipped: context overflow",
                skipped=True,
                error=str(exc),
            ),
        }
    action, reasoning = _parse_decision(raw)
    if not action:
        action, reasoning = fallback_decision_after_parse_failure(state)
    updates = _merge_usage(state, pt, ct, cost)
    updates.update(
        {
            "chosen_action": action,
            "chosen_reasoning": reasoning,
            "decision_raw": raw,
            "n_decisions": int(state.get("n_decisions", 0)) + 1,
        }
    )
    return updates


def _record(
    state: AgentState,
    *,
    action: str,
    reasoning: str,
    observation: str = "",
    **extra: Any,
) -> list[dict[str, Any]]:
    row = {
        "step": state.get("step", 0),
        "action": action,
        "reasoning": reasoning,
        "decision_raw": str(state.get("decision_raw") or "")[:2000],
        "observation": observation,
    }
    row.update(extra)
    return list(state.get("trajectory", [])) + [row]


def execute_action_node(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    action = state.get("chosen_action", "think")
    reasoning = state.get("chosen_reasoning", "")
    step_next = int(state.get("step", 0)) + 1
    candidate = _candidate_from_state(state)

    if action in VALID_ACTIONS and action not in set(available_actions(state)):
        # Reached only if the model names an action the prompt did not offer.
        # Recorded with passed=None: the action produced no evidence either way.
        return {
            "step": step_next,
            "trajectory": _record(
                state,
                action=action,
                reasoning=reasoning,
                observation=f"skipped: {action} is unavailable in this environment",
                passed=None,
                skipped=True,
            ),
        }

    if action == "finish":
        return {
            "step": step_next,
            "done": True,
            "final_action": "finish",
            "trajectory": _record(state, action=action, reasoning=reasoning),
        }

    if action == "think":
        return {
            "step": step_next,
            "trajectory": _record(state, action=action, reasoning=reasoning),
        }

    if action == "generate":
        if int(state.get("n_generations", 0)) >= state["max_generations"]:
            return {
                "step": step_next,
                "trajectory": _record(
                    state,
                    action=action,
                    reasoning=reasoning,
                    observation="skipped: generation budget exhausted",
                    skipped=True,
                ),
            }
        prompt = deps.adapter.build_prompt(
            state["instance"],
            candidate,
            _adapter_log(state.get("trajectory", [])),
        )
        generation_meta: dict[str, Any] = {}
        try:
            raw, pt, ct, cost, logprobs = _chat(
                deps,
                [{"role": "user", "content": prompt}],
                temperature=deps.generation_temperature,
                max_tokens=deps.max_tokens_generation,
                include_logprobs=deps.save_generation_logprobs,
                meta=generation_meta,
            )
        except ContextOverflowError as exc:
            return {
                "step": step_next,
                "done": True,
                "fixed": False,
                "final_action": "context_overflow_skip",
                "error": str(exc),
                "trajectory": _record(
                    state,
                    action=action,
                    reasoning=reasoning,
                    observation="skipped: context overflow",
                    skipped=True,
                    error=str(exc),
                ),
            }
        if deps.require_generation_logprobs and logprobs is None:
            raise RuntimeError(
                "generation logprobs were requested but the model endpoint "
                "returned no logprobs"
            )
        if generation_meta.get("text_source") == "reasoning":
            # The answer channel never arrived: the generation was cut off
            # mid-thought. Parsing the reasoning would manufacture a candidate
            # out of the model's deliberation, and that candidate would then be
            # scored as if the model had committed to it.
            return {
                **_merge_usage(state, pt, ct, cost),
                "step": step_next,
                "trajectory": _record(
                    state,
                    action=action,
                    reasoning=reasoning,
                    observation=(
                        "skipped: generation truncated before the answer channel "
                        f"(finish_reason={generation_meta.get('finish_reason', '')})"
                    ),
                    skipped=True,
                    truncated=True,
                    finish_reason=generation_meta.get("finish_reason", ""),
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    api_cost_usd=cost,
                ),
            }
        new_candidate = deps.adapter.extract_candidate(state["instance"], raw)
        logprobs_saved = False
        generation_index = int(state.get("n_generations", 0))
        if deps.save_generation_logprobs and deps.logprobs_output is not None:
            append_jsonl(
                deps.logprobs_output,
                {
                    "benchmark": state.get("benchmark"),
                    "instance_id": state.get("instance_id"),
                    "step": state.get("step", 0),
                    "generation_index": generation_index,
                    "model_id": deps.model_id,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "api_cost_usd": cost,
                    "top_logprobs": deps.top_logprobs,
                    "raw_text": raw,
                    "code": new_candidate.payload,
                    "logprobs": logprobs,
                },
            )
            logprobs_saved = True
        updates = _merge_usage(state, pt, ct, cost)
        updates.update(
            {
                "step": step_next,
                "candidate_payload": new_candidate.payload,
                "candidate_raw": new_candidate.raw_text[: deps.max_code_chars],
                "candidate_kind": new_candidate.kind,
                "last_generation_prompt": prompt[:12000],
                "n_generations": int(state.get("n_generations", 0)) + 1,
                "trajectory": _record(
                    state,
                    action=action,
                    reasoning=reasoning,
                    observation=f"generated {len(new_candidate.payload)} chars",
                    candidate_chars=len(new_candidate.payload),
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    api_cost_usd=cost,
                    logprobs_saved=logprobs_saved,
                    logprobs_output=str(deps.logprobs_output) if deps.logprobs_output else "",
                    finish_reason=generation_meta.get("finish_reason", ""),
                    truncated=bool(generation_meta.get("truncated", False)),
                ),
            }
        )
        return updates

    if candidate is None:
        return {
            "step": step_next,
            "trajectory": _record(
                state,
                action=action,
                reasoning=reasoning,
                observation="skipped: no candidate exists",
                skipped=True,
            ),
        }

    if action in CRITIC_ACTIONS:
        critic = CRITIC_ACTIONS[action]
        if critic == "L3" and deps.reviewer_client is None:
            return {
                "step": step_next,
                "trajectory": _record(
                    state,
                    action=action,
                    reasoning=reasoning,
                    observation="skipped: L3 reviewer client is unavailable",
                    passed=None,
                    skipped=True,
                ),
            }
        started = time.perf_counter()
        try:
            result: CriticResult = deps.adapter.run_critic(
                critic,
                state["instance"],
                candidate,
                deps.reviewer_client,
            )
            elapsed = time.perf_counter() - started
            return {
                "step": step_next,
                "api_cost_usd": float(state.get("api_cost_usd", 0.0))
                + float(result.api_cost_usd),
                "n_critic_runs": int(state.get("n_critic_runs", 0)) + 1,
                "trajectory": _record(
                    state,
                    action=action,
                    reasoning=reasoning,
                    observation=result.detail,
                    passed=result.passed,
                    detail=result.detail,
                    wall_clock_s=round(elapsed, 4),
                    api_cost_usd=result.api_cost_usd,
                ),
            }
        except Exception as exc:
            return {
                "step": step_next,
                "trajectory": _record(
                    state,
                    action=action,
                    reasoning=reasoning,
                    observation=f"critic error: {type(exc).__name__}: {exc}",
                    passed=None,
                    error=str(exc),
                ),
            }

    if action == "verify":
        if int(state.get("n_verifications", 0)) >= state["max_verifications"]:
            return {
                "step": step_next,
                "trajectory": _record(
                    state,
                    action=action,
                    reasoning=reasoning,
                    observation="skipped: verification budget exhausted",
                    skipped=True,
                ),
            }
        started = time.perf_counter()
        run_id = safe_stem(
            f"{state.get('benchmark')}__llm_agent__{state.get('instance_id')}__v{state.get('n_verifications', 0)}",
            180,
        )
        result: VerifyResult = deps.adapter.verify(state["instance"], candidate, run_id)
        elapsed = time.perf_counter() - started
        if not result.available:
            # No verdict was produced. Leave `fixed` untouched so the episode
            # ends without a terminal label rather than with a false negative.
            return {
                "step": step_next,
                "label_available": False,
                "trajectory": _record(
                    state,
                    action=action,
                    reasoning=reasoning,
                    observation=result.detail,
                    passed=None,
                    detail=result.detail,
                    verifier_available=False,
                    wall_clock_s=round(elapsed, 4),
                ),
            }
        passed = bool(result.passed)
        return {
            "step": step_next,
            "done": passed,
            "fixed": passed,
            "label_available": True,
            "final_action": "verify_pass" if passed else state.get("final_action", ""),
            "n_verifications": int(state.get("n_verifications", 0)) + 1,
            "trajectory": _record(
                state,
                action=action,
                reasoning=reasoning,
                observation=result.detail,
                passed=passed,
                detail=result.detail,
                verifier_available=True,
                wall_clock_s=round(elapsed, 4),
            ),
        }

    return {
        "step": step_next,
        "trajectory": _record(
            state,
            action=action,
            reasoning=reasoning,
            observation="unknown action",
            error="unknown action",
        ),
    }


def build_agent_graph(deps: AgentDeps):
    try:
        from langgraph.graph import END, StateGraph
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: langgraph. Install it in this environment with "
            "`pip install 'sage-agent[langgraph]'` from the repo root, or "
            "`pip install langgraph`."
        ) from exc

    def route_after_execute(state: AgentState) -> str:
        if state.get("done"):
            return END
        if state.get("step", 0) >= state["max_steps"]:
            return END
        return "decide_action"

    graph = StateGraph(AgentState)
    graph.add_node("decide_action", lambda s: decide_action_node(s, deps))
    graph.add_node("execute_action", lambda s: execute_action_node(s, deps))
    graph.set_entry_point("decide_action")
    graph.add_edge("decide_action", "execute_action")
    graph.add_conditional_edges(
        "execute_action",
        route_after_execute,
        {"decide_action": "decide_action", END: END},
    )
    return graph.compile()


class NoQuestionGenerator:
    def generate_questions(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return []


class NoQuestionAsker:
    def ask(self, _question: Any) -> str:
        return ""


@dataclass
class SageActionCandidateGenerator:
    deps: AgentDeps
    state: AgentState

    def generate_candidates(
        self,
        _user_input: str,
        _observations: list[str],
        _tool_schemas: dict[str, ToolSchema],
    ) -> list[ToolCallCandidate]:
        if not (self.state.get("candidate_payload") or ""):
            self.state.update(
                {
                    "chosen_action": "generate",
                    "chosen_reasoning": "auto: no candidate exists",
                    "decision_raw": "",
                }
            )
            return [
                ToolCallCandidate(
                    tool_name="generate",
                    arguments={"reasoning": "auto: no candidate exists"},
                )
            ]

        try:
            raw, pt, ct, cost, _logprobs = _chat(
                self.deps,
                _decision_messages(self.state),
                temperature=self.deps.decision_temperature,
                max_tokens=self.deps.max_tokens_decision,
            )
        except ContextOverflowError as exc:
            self.state.update(
                {
                    "done": True,
                    "fixed": False,
                    "final_action": "context_overflow_skip",
                    "error": str(exc),
                    "trajectory": _record(
                        self.state,
                        action="think",
                        reasoning="context overflow",
                        observation="skipped: context overflow",
                        skipped=True,
                        error=str(exc),
                    ),
                }
            )
            return [ToolCallCandidate(tool_name="finish", arguments={"reasoning": "context overflow"})]
        action, reasoning = _parse_decision(raw)
        if not action:
            action, reasoning = fallback_decision_after_parse_failure(self.state)
        updates = _merge_usage(self.state, pt, ct, cost)
        updates.update(
            {
                "chosen_action": action,
                "chosen_reasoning": reasoning,
                "decision_raw": raw,
                "n_decisions": int(self.state.get("n_decisions", 0)) + 1,
            }
        )
        self.state.update(updates)
        return [ToolCallCandidate(tool_name=action, arguments={"reasoning": reasoning})]


@dataclass
class SageActionToolExecutor:
    deps: AgentDeps
    state: AgentState

    def execute(self, tool_call: ToolCall) -> ExecutionResult:
        action = str(tool_call.tool_name)
        reasoning = str(tool_call.arguments.get("reasoning") or "")
        self.state.update({"chosen_action": action, "chosen_reasoning": reasoning})
        try:
            updates = execute_action_node(self.state, self.deps)
            self.state.update(updates)
        except Exception as exc:
            self.state.update(
                {
                    "done": True,
                    "final_action": f"exception:{type(exc).__name__}",
                    "error": str(exc),
                }
            )
            return ExecutionResult(success=False, error=str(exc))
        return ExecutionResult(
            success=not bool(self.state.get("error")),
            output={
                "action": action,
                "done": bool(self.state.get("done")),
                "fixed": bool(self.state.get("fixed")),
                "final_action": self.state.get("final_action", ""),
            },
            error=self.state.get("error") or None,
        )


def sage_tool_schemas(state: AgentState | None = None) -> list[ToolSchema]:
    """Tool schemas offered to the SAGE controller.

    Mirrors the action space the decision prompt advertises, so the two
    backends cannot disagree about which actions exist.
    """
    actions = available_actions(state) if state is not None else ACTION_SPACE
    return [
        ToolSchema(name=action, parameters={}, required=frozenset())
        for action in actions
    ]


def run_sage_agent_episode(state: AgentState, deps: AgentDeps) -> AgentState:
    """Run the episode using the repo's SAGE-Agent LangGraph wrapper."""
    tool_schemas = sage_tool_schemas(state)
    while not state.get("done") and int(state.get("step", 0)) < state["max_steps"]:
        if (
            not (state.get("candidate_payload") or "")
            and int(state.get("n_generations", 0)) >= state["max_generations"]
        ):
            state.update({"done": True, "final_action": "no_candidate"})
            break
        graph = SAGEGraph(
            tool_schemas=tool_schemas,
            candidate_generator=SageActionCandidateGenerator(deps, state),
            question_generator=NoQuestionGenerator(),
            question_asker=NoQuestionAsker(),
            tool_executor=SageActionToolExecutor(deps, state),
            constraint_extractor=None,
            config=SAGEGraphConfig(
                tau_exec=0.0,
                alpha=0.0,
                max_questions=0,
                enable_escalation=False,
                recursion_limit=8,
            ),
        )
        graph.run("Choose and execute the next LiveCodeBench tool action.")
    if not state.get("done") and int(state.get("step", 0)) >= state["max_steps"]:
        state.update({"done": True, "final_action": "max_steps"})
    return state


def instance_id_set(instances: list[dict[str, Any]], adapter: BenchmarkAdapter) -> set[str]:
    return {adapter.instance_id(inst) for inst in instances}


def split_instances(
    instances: list[dict[str, Any]],
    adapter: BenchmarkAdapter,
    *,
    n_train: int | None,
    train_fraction: float,
    split_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    shuffled = list(instances)
    random.Random(split_seed).shuffle(shuffled)
    n_total = len(shuffled)
    if n_train is None:
        n_train_final = int(round(n_total * train_fraction))
    else:
        n_train_final = int(n_train)
    n_train_final = max(0, min(n_train_final, n_total))
    train = shuffled[:n_train_final]
    test = shuffled[n_train_final:]
    split = {
        "split_seed": split_seed,
        "train_fraction": train_fraction if n_train is None else None,
        "n_train_requested": n_train,
        "n_total": n_total,
        "n_train": len(train),
        "n_test": len(test),
        "train_ids": [adapter.instance_id(inst) for inst in train],
        "test_ids": [adapter.instance_id(inst) for inst in test],
    }
    return train, test, split


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Records of a JSONL file, skipping blank and unparseable lines.

    Iterating the file is not the same as `read_text().splitlines()`, which is
    what this used to do: `splitlines` also breaks on \\v, \\f, \\x1c and
    U+2028, and these files hold raw model text written with
    `ensure_ascii=False`, so those characters survive into the line. One
    measured sidecar held 14 of them, which turned 150 records into 164
    fragments and silently lost 16% of the data.
    """
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_prior_rows(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def summarize_prior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("passed") in (True, False)]
    n = len(usable)
    correct = sum(1 for row in usable if row.get("passed") is True)
    return {
        "prior_Y1": (correct + 1) / (n + 2) if n else 0.5,
        "prior_calibration_n": n,
        "prior_calibration_correct": correct,
        "prior_smoothing": "Beta(1,1)",
    }


def append_actions(
    path: Path | None,
    *,
    split: str,
    benchmark: str,
    instance_id: str,
    model_id: str,
    actions: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> None:
    if path is None:
        return
    for action in actions:
        row = {
            "split": split,
            "benchmark": benchmark,
            "instance_id": instance_id,
            "model_id": model_id,
            **action,
        }
        if extra:
            row.update(extra)
        append_jsonl(path, row)


def calibrate_prior(
    *,
    train_instances: list[dict[str, Any]],
    benchmark: str,
    deps: AgentDeps,
    prior_patches: int,
    output_path: Path,
    logprobs_output_path: Path | None,
    actions_output_path: Path | None,
    resume: bool,
    print_each: bool,
    workers: int = 1,
) -> dict[str, Any]:
    existing_rows = load_prior_rows(output_path) if resume else []
    done = {
        (str(row.get("instance_id")), int(row.get("patch_id", 0)))
        for row in existing_rows
        if row.get("instance_id") is not None
    }
    rows = list(existing_rows)
    total_jobs = len(train_instances) * max(0, prior_patches)
    completed_now = 0
    jobs = [
        (instance, patch_id)
        for instance in train_instances
        for patch_id in range(prior_patches)
        if (deps.adapter.instance_id(instance), patch_id) not in done
    ]
    progress_lock = threading.Lock()

    def _record_prior(row: dict[str, Any], inst_id: str, patch_id: int) -> None:
        """Shared bookkeeping for one finished calibration patch."""
        nonlocal completed_now
        with progress_lock:
            done.add((inst_id, patch_id))
            completed_now += 1

    def calibrate_one(job: tuple[dict[str, Any], int]) -> dict[str, Any] | None:
        nonlocal completed_now
        instance, patch_id = job
        inst_id = deps.adapter.instance_id(instance)
        started = time.perf_counter()
        prompt = deps.adapter.build_prompt(instance, None, [])
        try:
            raw, pt, ct, cost, logprobs = _chat(
                deps,
                [{"role": "user", "content": prompt}],
                temperature=deps.generation_temperature,
                max_tokens=deps.max_tokens_generation,
                include_logprobs=deps.save_generation_logprobs,
            )
        except ContextOverflowError as exc:
            wall_s = time.perf_counter() - started
            actions = [
                {
                    "step": 0,
                    "action": "generate",
                    "observation": "skipped: context overflow",
                    "skipped": True,
                    "error": str(exc),
                }
            ]
            row = {
                "split": "train_calibration",
                "benchmark": benchmark,
                "instance_id": inst_id,
                "patch_id": patch_id,
                "model_id": deps.model_id,
                "passed": None,
                "detail": "skipped_context_overflow",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "api_cost_usd": 0.0,
                "wall_clock_s": round(wall_s, 4),
                "actions": actions,
                "code": "",
                "error": str(exc),
            }
            append_jsonl(output_path, row)
            append_actions(
                actions_output_path,
                split="train_calibration",
                benchmark=benchmark,
                instance_id=inst_id,
                model_id=deps.model_id,
                actions=actions,
                extra={"patch_id": patch_id},
            )
            _record_prior(row, inst_id, patch_id)
            if print_each:
                print(
                    f"[prior {completed_now}/{total_jobs}] {inst_id} p{patch_id} "
                    "passed=None detail=skipped_context_overflow"
                )
            return row
        if deps.require_generation_logprobs and logprobs is None:
            raise RuntimeError(
                "train prior calibration requested generation logprobs, "
                "but the model endpoint returned none"
            )
        candidate = deps.adapter.extract_candidate(instance, raw)
        if deps.save_generation_logprobs and logprobs_output_path is not None:
            append_jsonl(
                logprobs_output_path,
                {
                    "split": "train_calibration",
                    "benchmark": benchmark,
                    "instance_id": inst_id,
                    "patch_id": patch_id,
                    "model_id": deps.model_id,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "api_cost_usd": cost,
                    "top_logprobs": deps.top_logprobs,
                    "raw_text": raw,
                    "code": candidate.payload,
                    "logprobs": logprobs,
                },
            )
        verify_started = time.perf_counter()
        run_id = safe_stem(f"{benchmark}__prior__{inst_id}__p{patch_id}", 180)
        verify = deps.adapter.verify(instance, candidate, run_id)
        verify_s = time.perf_counter() - verify_started
        wall_s = time.perf_counter() - started
        actions = [
            {
                "step": 0,
                "action": "generate",
                "observation": f"generated {len(candidate.payload)} chars",
                "candidate_chars": len(candidate.payload),
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "api_cost_usd": cost,
                "logprobs_saved": bool(deps.save_generation_logprobs and logprobs_output_path),
            },
            {
                "step": 1,
                "action": "verify",
                "passed": bool(verify.passed) if verify.available else None,
                "detail": verify.detail,
                "observation": verify.detail,
                "verifier_available": verify.available,
                "wall_clock_s": round(verify_s, 4),
            },
        ]
        row = {
            "split": "train_calibration",
            "benchmark": benchmark,
            "instance_id": inst_id,
            "patch_id": patch_id,
            "model_id": deps.model_id,
            # None rather than False when nothing verified it: the prior is
            # a base rate, and counting unverifiable patches as failures
            # would drag it toward zero for purely infrastructural reasons.
            "passed": bool(verify.passed) if verify.available else None,
            "label_available": verify.available,
            "detail": verify.detail,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "api_cost_usd": cost,
            "wall_clock_s": round(wall_s, 4),
            "actions": actions,
            "code": candidate.payload[: deps.max_code_chars],
        }
        append_jsonl(output_path, row)
        append_actions(
            actions_output_path,
            split="train_calibration",
            benchmark=benchmark,
            instance_id=inst_id,
            model_id=deps.model_id,
            actions=actions,
            extra={"patch_id": patch_id},
        )
        _record_prior(row, inst_id, patch_id)
        if print_each:
            print(
                f"[prior {completed_now}/{total_jobs}] {inst_id} p{patch_id} "
                f"passed={bool(verify.passed)} detail={verify.detail}"
            )
        return row

    def calibrate_one_guarded(job: tuple[dict[str, Any], int]) -> dict[str, Any] | None:
        """One bad instance must not take the whole calibration down with it.

        `calibrate_one` already returns None for instances it cannot use, so a
        failure here is expressible without inventing a label. It is logged
        loudly rather than swallowed: a run that quietly drops instances is
        worse than one that reports how many it dropped.
        """
        try:
            return calibrate_one(job)
        except Exception:
            inst_id = deps.adapter.instance_id(job[0])
            log.exception("[prior] instance %s failed; skipping it", inst_id)
            return None

    if workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            produced = list(pool.map(calibrate_one_guarded, jobs))
    else:
        produced = [calibrate_one_guarded(job) for job in jobs]
    dropped = sum(1 for row in produced if row is None)
    if dropped:
        log.warning("[prior] %d of %d calibration instances were dropped",
                    dropped, len(jobs))
    rows.extend(row for row in produced if row is not None)

    summary = summarize_prior(rows)
    summary.update(
        {
            "prior_calibration_output": str(output_path),
            "prior_calibration_logprobs_output": str(logprobs_output_path or ""),
            "prior_patches": prior_patches,
        }
    )
    return summary


def initial_state(
    *,
    instance: dict[str, Any],
    instance_id: str,
    benchmark: str,
    max_steps: int,
    max_generations: int,
    max_verifications: int,
    prior_summary: dict[str, Any] | None = None,
    unavailable_actions: tuple[str, ...] = (),
    describe_actions: bool = False,
    hide_exhausted_actions: bool = False,
    no_repeat_verify: bool = False,
    final_verify: bool = False,
) -> AgentState:
    state: AgentState = {
        "instance": instance,
        "instance_id": instance_id,
        "benchmark": benchmark,
        "unavailable_actions": unavailable_actions,
        "describe_actions": describe_actions,
        "hide_exhausted_actions": hide_exhausted_actions,
        "no_repeat_verify": no_repeat_verify,
        # The prompt needs this: when `verify` is not an action the model can
        # call but a terminal check still happens, saying so is the honest
        # description of the setup. Left unsaid, the model reads the missing
        # action as "there is no test suite" and submits its first draft.
        "final_verify": final_verify,
        # When the environment has no verifier at all, the episode starts
        # without a label and never acquires one. Defaulting to True here would
        # mark every SWE-Bench row as carrying ground truth it never had.
        "label_available": "verify" not in unavailable_actions,
        "step": 0,
        "max_steps": max_steps,
        "max_generations": max_generations,
        "max_verifications": max_verifications,
        "trajectory": [],
        "n_decisions": 0,
        "n_generations": 0,
        "n_critic_runs": 0,
        "n_verifications": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "api_cost_usd": 0.0,
        "fixed": False,
        "done": False,
        "final_action": "",
    }
    if prior_summary:
        state["prior_Y1"] = float(prior_summary.get("prior_Y1", 0.5))
        state["prior_calibration_n"] = int(prior_summary.get("prior_calibration_n", 0))
        state["prior_calibration_correct"] = int(
            prior_summary.get("prior_calibration_correct", 0)
        )
    return state


def result_record(
    state: AgentState,
    deps: AgentDeps,
    wall_clock_s: float,
    *,
    split_summary: dict[str, Any],
    prior_summary: dict[str, Any],
) -> dict[str, Any]:
    instance = state["instance"]
    return {
        "benchmark": state.get("benchmark"),
        "split": "test",
        "instance_id": state.get("instance_id"),
        "question_title": instance.get("question_title"),
        "difficulty": instance.get("difficulty"),
        "platform": instance.get("platform"),
        "model_id": deps.model_id,
        "split_seed": split_summary.get("split_seed"),
        "n_train": split_summary.get("n_train"),
        "n_test": split_summary.get("n_test"),
        "prior_Y1": prior_summary.get("prior_Y1", 0.5),
        "prior_calibration_n": prior_summary.get("prior_calibration_n", 0),
        "prior_calibration_correct": prior_summary.get("prior_calibration_correct", 0),
        "prior_smoothing": prior_summary.get("prior_smoothing", ""),
        "fixed": bool(state.get("fixed", False)),
        # Downstream analysis must filter on this before using `fixed`: an
        # episode whose verifier never ran has no ground truth, and counting it
        # as a negative is the difference between a measurement and a fiction.
        "label_available": bool(state.get("label_available", True)),
        "unavailable_actions": list(state.get("unavailable_actions") or ()),
        "final_action": state.get("final_action") or "max_steps",
        "n_steps": int(state.get("step", 0)),
        "n_decisions": int(state.get("n_decisions", 0)),
        "n_generations": int(state.get("n_generations", 0)),
        "n_critic_runs": int(state.get("n_critic_runs", 0)),
        "n_verifications": int(state.get("n_verifications", 0)),
        "prompt_tokens": int(state.get("prompt_tokens", 0)),
        "completion_tokens": int(state.get("completion_tokens", 0)),
        "api_cost_usd": float(state.get("api_cost_usd", 0.0)),
        "wall_clock_s": round(wall_clock_s, 4),
        "trajectory": state.get("trajectory", []),
        "final_code": (state.get("candidate_payload") or "")[: deps.max_code_chars],
    }


def maybe_final_verify(state: AgentState, deps: AgentDeps) -> AgentState:
    """Run one terminal hidden-test verification when a candidate exists."""
    if state.get("fixed"):
        return state
    candidate = _candidate_from_state(state)
    if candidate is None:
        return state

    started = time.perf_counter()
    run_id = safe_stem(
        f"{state.get('benchmark')}__llm_agent__{state.get('instance_id')}__final_verify",
        180,
    )
    result: VerifyResult = deps.adapter.verify(state["instance"], candidate, run_id)
    elapsed = time.perf_counter() - started
    previous_final = str(state.get("final_action") or "max_steps")
    verified_state: AgentState = dict(state)

    if not result.available:
        verified_state.update(
            {
                "step": int(state.get("step", 0)) + 1,
                "label_available": False,
                "final_action": "final_verify_unavailable",
                "trajectory": _record(
                    state,
                    action="final_verify",
                    reasoning=f"terminal verification after {previous_final}",
                    observation=result.detail,
                    passed=None,
                    detail=result.detail,
                    verifier_available=False,
                    wall_clock_s=round(elapsed, 4),
                    previous_final_action=previous_final,
                ),
            }
        )
        return verified_state

    passed = bool(result.passed)
    verified_state.update(
        {
            "step": int(state.get("step", 0)) + 1,
            "done": passed,
            "fixed": passed,
            "label_available": True,
            "final_action": "final_verify_pass" if passed else "final_verify_fail",
            "n_verifications": int(state.get("n_verifications", 0)) + 1,
            "trajectory": _record(
                state,
                action="final_verify",
                reasoning=f"terminal verification after {previous_final}",
                observation=result.detail,
                passed=passed,
                detail=result.detail,
                verifier_available=True,
                wall_clock_s=round(elapsed, 4),
                previous_final_action=previous_final,
            ),
        }
    )
    return verified_state


#: Serialises every JSONL append. Episodes run concurrently and several of
#: them write to the same files (results, actions, generation logprobs,
#: verbalized scores); interleaved partial lines would corrupt all of them.
_WRITE_LOCK = threading.Lock()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    payload = json.dumps(row, ensure_ascii=False) + "\n"
    with _WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(payload)


def build_parser() -> argparse.ArgumentParser:
    """Kept separate from parsing so tests can inspect the flags."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--benchmark",
        choices=[
            "lcb_easy",
            "lcb_medium",
            "lcb_med",
            "lcb_hard",
            "lcb_all",
            "mbpp",
            "humaneval",
            "swebench_lite",
            "swebench_verified",
            "humanevalfix",
            "codecontests",
        ],
        default="lcb_easy",
    )
    p.add_argument("--generator", default="haiku45", help="Key from code_uq.common.generators")
    p.add_argument("--n-instances", type=int, default=5, help="0 = all matching benchmark tasks")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--train-fraction", type=float, default=0.5)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--prior-patches", type=int, default=1)
    p.add_argument("--skip-prior-calibration", action="store_true")
    p.add_argument("--platform", default="leetcode", choices=["leetcode", "atcoder", "codeforces", "all"])
    p.add_argument("--lcb-version", default="all", choices=["v1", "all"])
    p.add_argument(
        "--private-test-cap",
        type=int,
        default=0,
        help=(
            "Cap on private tests used for the label; 0 means all of them. A cap "
            "checks a weaker condition than the benchmark defines: at 12 against "
            "a median suite of 35 the measured success rate was 0.93, well above "
            "the real one. Cap only when profiling runtime."
        ),
    )
    p.add_argument("--plus-input-cap", type=int, default=200)
    p.add_argument("--swe-harness-workers", type=int, default=1)
    p.add_argument(
        "--max-steps",
        type=int,
        default=25,
        help=(
            "Step ceiling per episode. One generation plus a sweep of four "
            "critics plus the terminal check already costs six, so 12 left room "
            "for two candidates and the loop could not be observed. Measured: "
            "the last generation of a successful episode lands by step 22."
        ),
    )
    p.add_argument("--max-generations", type=int, default=10)
    p.add_argument(
        "--max-verifications",
        type=int,
        default=0,
        help=(
            "Oracle calls the controller may spend. Zero by default because an "
            "episode ends the moment the oracle passes: with a budget above "
            "zero, 'how many steps were taken' and 'did verify ever fail' become "
            "structurally equivalent to the label, and any estimator reading the "
            "trajectory gets part of the answer for free. Measured: the belief "
            "state scores 0.915 with that leak and 0.224 without, while agent "
            "quality is unchanged. Raise it only to study the leak itself."
        ),
    )
    p.add_argument(
        "--agent-backend",
        choices=["sage", "langgraph"],
        default="sage",
        help="sage uses sage_agent.langgraph.SAGEGraph; langgraph uses the old local two-node loop.",
    )
    p.add_argument(
        "--no-final-verify",
        dest="final_verify",
        action="store_false",
        help=(
            "Skip the single automatic terminal verification. It runs by "
            "default: with --max-verifications 0 it is the only source of a "
            "label, and being automatic is the point -- a verdict the controller "
            "cannot schedule cannot leak into the trajectory."
        ),
    )
    p.add_argument("--decision-temperature", type=float, default=0.2)
    p.add_argument("--generation-temperature", type=float, default=0.7)
    p.add_argument("--max-tokens-decision", type=int, default=128)
    p.add_argument(
        "--max-tokens-generation",
        type=int,
        default=32768,
        help=(
            "Token ceiling for one generation: a circuit breaker for degenerate "
            "loops, not a budget. At 4000 about 10 percent of steps were "
            "truncated, and a truncated generation yields no extractable code, "
            "so it scored as a wrong answer. At 32768 truncation is nil and the "
            "average cost barely moves -- only the tail pays."
        ),
    )
    p.add_argument("--max-code-chars", type=int, default=20000)
    p.add_argument(
        "--save-generation-logprobs",
        action="store_true",
        help="Request and save token logprobs/top_logprobs for every code generation call.",
    )
    p.add_argument(
        "--require-generation-logprobs",
        action="store_true",
        help="Fail an episode if the model endpoint returns no generation logprobs.",
    )
    p.add_argument(
        "--top-logprobs",
        type=int,
        default=20,
        help="Number of top token alternatives to request with logprobs.",
    )
    p.add_argument(
        "--logprobs-output",
        type=Path,
        default=None,
        help="JSONL path for generation logprobs. Default: <output stem>.generation_logprobs.jsonl.",
    )
    p.add_argument(
        "--save-verbalized-2s",
        action="store_true",
        help="Save LM-Polygraph-style Verbalized2S confidence for the final candidate.",
    )
    p.add_argument(
        "--require-verbalized-2s",
        action="store_true",
        help="Fail an episode if Verbalized2S confidence cannot be parsed.",
    )
    p.add_argument("--verbalized-2s-temperature", type=float, default=0.0)
    p.add_argument("--verbalized-2s-max-tokens", type=int, default=1024)
    p.add_argument(
        "--verbalized-2s-output",
        type=Path,
        default=None,
        help="JSONL path for Verbalized2S rows. Default: <output stem>.verbalized_2s.jsonl.",
    )
    p.add_argument(
        "--split-output",
        type=Path,
        default=None,
        help="JSON path for train/test ids. Default: <output stem>.split.json.",
    )
    p.add_argument(
        "--prior-calibration-output",
        type=Path,
        default=None,
        help="JSONL path for train prior calibration. Default: <output stem>.train_prior_calibration.jsonl.",
    )
    p.add_argument(
        "--prior-calibration-logprobs-output",
        type=Path,
        default=None,
        help="JSONL path for train calibration generation logprobs.",
    )
    p.add_argument(
        "--actions-output",
        type=Path,
        default=None,
        help="Flat JSONL path with every train/test action. Default: <output stem>.actions.jsonl.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "code_uq" / "lcb_llm_tool_agent.jsonl",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--print-each", action="store_true")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Episodes to run concurrently. Each episode is an independent "
            "sequence of API calls, and vLLM batches concurrent requests, so "
            "the wall clock drops roughly linearly until the server saturates. "
            "1 (default) keeps the original sequential behaviour."
        ),
    )
    p.add_argument(
        "--allow-repeat-verify",
        dest="no_repeat_verify",
        action="store_false",
        help=(
            "Keep offering `verify` after the oracle has ruled on the current "
            "candidate. Withheld by default: re-verifying unchanged code returns "
            "the same answer, so the call is pure cost -- 55 of 149 verifier "
            "calls in a measured run were such repeats."
        ),
    )
    p.add_argument(
        "--offer-exhausted-actions",
        dest="hide_exhausted_actions",
        action="store_false",
        help=(
            "Keep offering `generate`/`verify` after their budget is spent. "
            "Hidden by default: on a 76-episode run 11%% of all steps went to "
            "picking an action that could only answer 'budget exhausted', and a "
            "no-op recorded as a step distorts both cost and trajectory shape."
        ),
    )
    p.add_argument(
        "--bare-action-names",
        dest="describe_actions",
        action="store_false",
        help=(
            "List bare action names in the prompt, as the original did, instead "
            "of describing what each action does, which critics have already "
            "judged the current candidate, and the prior. Descriptions are on by "
            "default: without them the controller goes straight from generate to "
            "verify and no pre-oracle evidence exists at all -- 13 episodes out "
            "of 76 had any, against 6 of 6 in a pilot with descriptions."
        ),
    )
    p.add_argument(
        "--run-config-output",
        type=Path,
        default=None,
        help="JSON path recording the run settings. "
             "Default: <output stem>.run_config.json.",
    )
    return p


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()




def adapter_unavailable_actions(adapter: BenchmarkAdapter) -> set[str]:
    """Actions the adapter cannot perform. Adapters may omit the method."""
    getter = getattr(adapter, "unavailable_actions", None)
    return set(getter()) if callable(getter) else set()


def make_adapter(args: argparse.Namespace) -> BenchmarkAdapter:
    benchmark = "lcb_medium" if args.benchmark == "lcb_med" else args.benchmark
    if benchmark in {"swebench_lite", "swebench_verified"}:
        return make_swe_adapter(
            benchmark=benchmark,
            n_instances=args.n_instances,
            seed=args.seed,
            output_dir=args.output.parent,
            harness_workers=args.swe_harness_workers,
        )
    if benchmark == "mbpp":
        from code_uq.environments.calibration import mbpp as mbpp_calibrate
    elif benchmark == "humaneval":
        from code_uq.environments.calibration import humaneval as humaneval_calibrate
    elif benchmark == "humanevalfix":
        from code_uq.environments.calibration import humanevalfix as humanevalfix_calibrate
    elif benchmark == "codecontests":
        from code_uq.environments.calibration import codecontests as codecontests_calibrate
    return make_function_adapter(
        benchmark=benchmark,
        n_instances=args.n_instances,
        seed=args.seed,
        lcb_version=args.lcb_version,
        plus_input_cap=args.plus_input_cap,
        lcb_private_test_cap=args.private_test_cap,
        platform=args.platform,
    )


def main() -> None:
    args = parse_args()
    _load_env_chain()
    if args.logprobs_output is None:
        args.logprobs_output = args.output.with_name(
            args.output.stem + ".generation_logprobs.jsonl"
        )
    if args.verbalized_2s_output is None:
        args.verbalized_2s_output = args.output.with_name(
            args.output.stem + ".verbalized_2s.jsonl"
        )
    if args.split_output is None:
        args.split_output = args.output.with_name(args.output.stem + ".split.json")
    if args.prior_calibration_output is None:
        args.prior_calibration_output = args.output.with_name(
            args.output.stem + ".train_prior_calibration.jsonl"
        )
    if args.prior_calibration_logprobs_output is None:
        args.prior_calibration_logprobs_output = args.output.with_name(
            args.output.stem + ".train_prior_calibration.generation_logprobs.jsonl"
        )
    if args.actions_output is None:
        args.actions_output = args.output.with_name(args.output.stem + ".actions.jsonl")
    if args.run_config_output is None:
        args.run_config_output = args.output.with_name(
            args.output.stem + ".run_config.json"
        )

    generator = canonical_generator_key(args.generator)
    model_id = GENERATORS[generator][0]
    if generator == "gpt_oss_120b_local":
        model_id = os.environ.get("GPT_OSS_120B_MODEL", model_id)
    benchmark = "lcb_medium" if args.benchmark == "lcb_med" else args.benchmark

    args.run_config_output.parent.mkdir(parents=True, exist_ok=True)
    args.run_config_output.write_text(
        json.dumps(
            {
                "benchmark": benchmark,
                "generator": generator,
                "model_id": model_id,
                "agent_backend": args.agent_backend,
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in sorted(vars(args).items())
                },
            },
            indent=2,
        )
    )

    adapter = make_adapter(args)
    instances = adapter.load_instances()

    train_instances, test_instances, split_summary = split_instances(
        instances,
        adapter,
        n_train=args.n_train,
        train_fraction=args.train_fraction,
        split_seed=args.split_seed,
    )
    args.split_output.parent.mkdir(parents=True, exist_ok=True)
    args.split_output.write_text(json.dumps(split_summary, indent=2))

    if args.resume and args.output.exists():
        # A record lost here does not merely disappear: it reads as "this
        # instance is not done yet" and the episode is silently run again.
        done = {str(row.get("instance_id")) for row in iter_jsonl(args.output)}
        test_instances = [
            inst for inst in test_instances if adapter.instance_id(inst) not in done
        ]
    else:
        args.output.unlink(missing_ok=True)
        if args.save_generation_logprobs:
            args.logprobs_output.unlink(missing_ok=True)
            args.prior_calibration_logprobs_output.unlink(missing_ok=True)
        if args.save_verbalized_2s:
            args.verbalized_2s_output.unlink(missing_ok=True)
        args.prior_calibration_output.unlink(missing_ok=True)
        args.actions_output.unlink(missing_ok=True)

    llm_client = _make_client(generator)
    # REVIEWER_BASE_URL points the L3 reviewer at a local OpenAI-compatible
    # endpoint (e.g. the same local vLLM) so the whole pipeline runs offline;
    # pair with L3_REVIEW_MODEL. Falls back to the OpenRouter reviewer.
    reviewer_base_url = os.environ.get("REVIEWER_BASE_URL", "").strip()
    if reviewer_base_url:
        from openai import OpenAI
        reviewer_client = OpenAI(api_key="EMPTY", base_url=reviewer_base_url)
    else:
        try:
            reviewer_client = _make_client(None)
        except SystemExit:
            reviewer_client = None
    deps = AgentDeps(
        adapter=adapter,
        llm_client=llm_client,
        reviewer_client=reviewer_client,
        model_id=model_id,
        decision_temperature=args.decision_temperature,
        generation_temperature=args.generation_temperature,
        max_tokens_decision=args.max_tokens_decision,
        max_tokens_generation=args.max_tokens_generation,
        max_code_chars=args.max_code_chars,
        save_generation_logprobs=args.save_generation_logprobs,
        require_generation_logprobs=args.require_generation_logprobs,
        top_logprobs=args.top_logprobs,
        logprobs_output=args.logprobs_output,
    )

    if args.skip_prior_calibration:
        prior_summary = summarize_prior(load_prior_rows(args.prior_calibration_output))
        prior_summary.update(
            {
                "prior_calibration_output": str(args.prior_calibration_output),
                "prior_calibration_logprobs_output": str(
                    args.prior_calibration_logprobs_output
                ),
                "prior_patches": args.prior_patches,
                "skipped_prior_calibration": True,
            }
        )
    else:
        prior_summary = calibrate_prior(
            train_instances=train_instances,
            benchmark=benchmark,
            deps=deps,
            prior_patches=args.prior_patches,
            output_path=args.prior_calibration_output,
            logprobs_output_path=(
                args.prior_calibration_logprobs_output
                if args.save_generation_logprobs
                else None
            ),
            actions_output_path=args.actions_output,
            resume=args.resume,
            print_each=args.print_each,
            workers=args.workers,
        )

    graph = build_agent_graph(deps) if args.agent_backend == "langgraph" else None

    blocked_actions = tuple(sorted(adapter_unavailable_actions(adapter)))

    print(
        f"LCB LLM tool-agent: benchmark={benchmark} backend={args.agent_backend} model={model_id} "
        f"train={split_summary['n_train']} test={split_summary['n_test']} "
        f"remaining_test={len(test_instances)} output={args.output}"
    )
    if blocked_actions:
        print(
            f"unavailable actions in this environment: {', '.join(blocked_actions)} "
            f"(removed from the action space)"
        )
        if "verify" in blocked_actions:
            print(
                "NOTE: no verifier -> episodes carry no terminal label; rows are "
                "written with label_available=false and must be excluded from "
                "any metric that needs ground truth."
            )
    print(
        f"prior_Y1={prior_summary['prior_Y1']:.3f} "
        f"n={prior_summary['prior_calibration_n']} "
        f"correct={prior_summary['prior_calibration_correct']}"
    )
    print(f"split={args.split_output}")
    print(f"prior_calibration={args.prior_calibration_output}")
    print(f"actions={args.actions_output}")
    if args.save_generation_logprobs:
        print(f"generation_logprobs={args.logprobs_output}")
        print(f"prior_calibration_logprobs={args.prior_calibration_logprobs_output}")
    if args.save_verbalized_2s:
        print(f"verbalized_2s={args.verbalized_2s_output}")
    def run_one_episode(indexed: tuple[int, dict[str, Any]]) -> None:
        idx, instance = indexed
        inst_id = adapter.instance_id(instance)
        started = time.perf_counter()
        state = initial_state(
            instance=instance,
            instance_id=inst_id,
            benchmark=benchmark,
            max_steps=args.max_steps,
            max_generations=args.max_generations,
            max_verifications=args.max_verifications,
            prior_summary=prior_summary,
            unavailable_actions=blocked_actions,
            describe_actions=args.describe_actions,
            hide_exhausted_actions=args.hide_exhausted_actions,
            no_repeat_verify=args.no_repeat_verify,
            final_verify=args.final_verify,
        )
        try:
            if args.agent_backend == "sage":
                final_state = run_sage_agent_episode(state, deps)
            else:
                final_state = graph.invoke(state)
            verbalized_2s_row: dict[str, Any] = {}
            if args.save_verbalized_2s:
                verbalized_2s_row = run_verbalized_2s(
                    final_state,
                    deps,
                    temperature=args.verbalized_2s_temperature,
                    max_tokens=args.verbalized_2s_max_tokens,
                    require_parse=args.require_verbalized_2s,
                )
                append_jsonl(
                    args.verbalized_2s_output,
                    {
                        "split": "test",
                        "benchmark": benchmark,
                        "instance_id": inst_id,
                        "model_id": model_id,
                        "agent_backend": args.agent_backend,
                        **verbalized_2s_row,
                    },
                )
            if args.final_verify:
                final_state = maybe_final_verify(final_state, deps)
            row = result_record(
                final_state,
                deps,
                time.perf_counter() - started,
                split_summary=split_summary,
                prior_summary=prior_summary,
            )
            row["agent_backend"] = args.agent_backend
            row.update(verbalized_2s_row)
        except Exception as exc:
            row = {
                "benchmark": benchmark,
                "split": "test",
                "instance_id": inst_id,
                "model_id": model_id,
                "agent_backend": args.agent_backend,
                "prior_Y1": prior_summary.get("prior_Y1", 0.5),
                "fixed": False,
                "final_action": f"exception:{type(exc).__name__}",
                "error": str(exc),
                "wall_clock_s": round(time.perf_counter() - started, 4),
                "trajectory": [],
            }
        append_jsonl(args.output, row)
        append_actions(
            args.actions_output,
            split="test",
            benchmark=benchmark,
            instance_id=inst_id,
            model_id=model_id,
            actions=row.get("trajectory", []),
            extra={
                "prior_Y1": prior_summary.get("prior_Y1", 0.5),
                "agent_backend": args.agent_backend,
            },
        )
        if args.print_each:
            print(
                f"[test {idx}/{len(test_instances)}] {inst_id} "
                f"fixed={row.get('fixed')} final={row.get('final_action')} "
                f"steps={row.get('n_steps', 0)} cost=${row.get('api_cost_usd', 0.0):.4f}"
            )


    work = list(enumerate(test_instances, start=1))
    if args.workers > 1 and len(work) > 1:
        # Episodes share nothing but append-only output files, which
        # `append_jsonl` serialises, so they parallelise directly. The server
        # batches the concurrent requests; the test subprocesses are
        # independent by construction.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run_one_episode, work))
    else:
        for item in work:
            run_one_episode(item)

    print(f"done: {args.output}")


if __name__ == "__main__":
    main()
