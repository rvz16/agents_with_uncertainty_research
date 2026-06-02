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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

REPO_ROOT = Path(__file__).resolve().parents[2]
ORCH_ROOT = REPO_ROOT / "experiments" / "orchestration_hypothesis_testing"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ORCH_ROOT))
sys.path.insert(0, str(ORCH_ROOT / "scripts"))

from calibration import lcb as lcb_calibrate  # noqa: E402

# fitted_live adapters still import the pre-refactor module name.
sys.modules.setdefault("lcb_calibrate", lcb_calibrate)

from _common.cost import cost_for_call  # noqa: E402
from _common.generators import GENERATORS, _make_client, canonical_generator_key  # noqa: E402
from fitted_live.common import Candidate, CriticResult, VerifyResult, safe_stem  # noqa: E402
from fitted_live.function_adapters import LCBAdapter  # noqa: E402

from sage_agent import ExecutionResult, ToolCall, ToolCallCandidate, ToolSchema  # noqa: E402
from sage_agent.langgraph import SAGEGraph, SAGEGraphConfig  # noqa: E402

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
    done: bool
    final_action: str
    error: str
    prior_Y1: float
    prior_calibration_n: int
    prior_calibration_correct: int


@dataclass
class AgentDeps:
    adapter: LCBAdapter
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


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str) and content:
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
        if parts:
            return "\n".join(parts)

    for attr in ("reasoning_content", "reasoning"):
        value = getattr(message, attr, None)
        if isinstance(value, str) and value:
            return value

    extra = getattr(message, "model_extra", None) or {}
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning"):
            value = extra.get(key)
            if isinstance(value, str) and value:
                return value

    kwargs = getattr(message, "additional_kwargs", None) or {}
    if isinstance(kwargs, dict):
        for key in ("reasoning_content", "reasoning"):
            value = kwargs.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _chat(
    deps: AgentDeps,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    include_logprobs: bool = False,
) -> tuple[str, int, int, float, Any]:
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
    resp = deps.llm_client.chat.completions.create(**params)
    text = _message_text(resp.choices[0].message)
    prompt_tokens, completion_tokens = _usage(resp)
    return (
        text,
        prompt_tokens,
        completion_tokens,
        cost_for_call(deps.model_id, prompt_tokens, completion_tokens),
        _response_logprobs(resp) if include_logprobs else None,
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


def _decision_prompt(state: AgentState) -> str:
    candidate_chars = len(state.get("candidate_payload") or "")
    remaining = {
        "steps": state["max_steps"] - state.get("step", 0),
        "generations": state["max_generations"] - state.get("n_generations", 0),
        "verifications": state["max_verifications"] - state.get("n_verifications", 0),
    }
    return f"""Choose one next tool action for a LiveCodeBench coding episode.

You are only routing tools. Do not solve the programming task and do not write code.

Valid actions:
generate, critic_L0, critic_L1, critic_L2, critic_L3, verify, think, finish

Return exactly one compact JSON object and nothing else:
{{"action":"generate|critic_L0|critic_L1|critic_L2|critic_L3|verify|think|finish","reasoning":"short reason"}}

State:
candidate_exists: {bool(candidate_chars)}
candidate_chars: {candidate_chars}
budget_remaining: {json.dumps(remaining)}

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
    if not (state.get("candidate_payload") or ""):
        return "generate", "fallback: decision parse failed and no candidate exists"
    if int(state.get("n_critic_runs", 0)) == 0:
        return "critic_L2", "fallback: decision parse failed after candidate generation"
    if int(state.get("n_verifications", 0)) < int(state.get("max_verifications", 0)):
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

    raw, pt, ct, cost, _logprobs = _chat(
        deps,
        _decision_messages(state),
        temperature=deps.decision_temperature,
        max_tokens=deps.max_tokens_decision,
    )
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
        raw, pt, ct, cost, logprobs = _chat(
            deps,
            [{"role": "user", "content": prompt}],
            temperature=deps.generation_temperature,
            max_tokens=deps.max_tokens_generation,
            include_logprobs=deps.save_generation_logprobs,
        )
        if deps.require_generation_logprobs and logprobs is None:
            raise RuntimeError(
                "generation logprobs were requested but the model endpoint "
                "returned no logprobs"
            )
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
        passed = bool(result.passed)
        return {
            "step": step_next,
            "done": passed,
            "fixed": passed,
            "final_action": "verify_pass" if passed else state.get("final_action", ""),
            "n_verifications": int(state.get("n_verifications", 0)) + 1,
            "trajectory": _record(
                state,
                action=action,
                reasoning=reasoning,
                observation=result.detail,
                passed=passed,
                detail=result.detail,
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

        raw, pt, ct, cost, _logprobs = _chat(
            self.deps,
            _decision_messages(self.state),
            temperature=self.deps.decision_temperature,
            max_tokens=self.deps.max_tokens_decision,
        )
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


def sage_tool_schemas() -> list[ToolSchema]:
    return [
        ToolSchema(name=action, parameters={}, required=frozenset())
        for action in ACTION_SPACE
    ]


def run_sage_agent_episode(state: AgentState, deps: AgentDeps) -> AgentState:
    """Run the LCB episode using the repo's SAGE-Agent LangGraph wrapper."""
    tool_schemas = sage_tool_schemas()
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


def instance_id_set(instances: list[dict[str, Any]], adapter: LCBAdapter) -> set[str]:
    return {adapter.instance_id(inst) for inst in instances}


def split_instances(
    instances: list[dict[str, Any]],
    adapter: LCBAdapter,
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


def load_prior_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


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
    for instance in train_instances:
        inst_id = deps.adapter.instance_id(instance)
        for patch_id in range(prior_patches):
            if (inst_id, patch_id) in done:
                continue
            started = time.perf_counter()
            prompt = deps.adapter.build_prompt(instance, None, [])
            raw, pt, ct, cost, logprobs = _chat(
                deps,
                [{"role": "user", "content": prompt}],
                temperature=deps.generation_temperature,
                max_tokens=deps.max_tokens_generation,
                include_logprobs=deps.save_generation_logprobs,
            )
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
                    "passed": bool(verify.passed),
                    "detail": verify.detail,
                    "observation": verify.detail,
                    "wall_clock_s": round(verify_s, 4),
                },
            ]
            row = {
                "split": "train_calibration",
                "benchmark": benchmark,
                "instance_id": inst_id,
                "patch_id": patch_id,
                "model_id": deps.model_id,
                "passed": bool(verify.passed),
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
            rows.append(row)
            done.add((inst_id, patch_id))
            completed_now += 1
            if print_each:
                print(
                    f"[prior {completed_now}/{total_jobs}] {inst_id} p{patch_id} "
                    f"passed={bool(verify.passed)} detail={verify.detail}"
                )
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
) -> AgentState:
    state: AgentState = {
        "instance": instance,
        "instance_id": instance_id,
        "benchmark": benchmark,
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
    passed = bool(result.passed)
    previous_final = str(state.get("final_action") or "max_steps")
    verified_state: AgentState = dict(state)
    verified_state.update(
        {
            "step": int(state.get("step", 0)) + 1,
            "done": passed,
            "fixed": passed,
            "final_action": "final_verify_pass" if passed else "final_verify_fail",
            "n_verifications": int(state.get("n_verifications", 0)) + 1,
            "trajectory": _record(
                state,
                action="final_verify",
                reasoning=f"terminal verification after {previous_final}",
                observation=result.detail,
                passed=passed,
                detail=result.detail,
                wall_clock_s=round(elapsed, 4),
                previous_final_action=previous_final,
            ),
        }
    )
    return verified_state


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--benchmark",
        choices=["lcb_easy", "lcb_medium", "lcb_med", "lcb_hard"],
        default="lcb_easy",
    )
    p.add_argument("--generator", default="haiku45", help="Key from _common.generators")
    p.add_argument("--n-instances", type=int, default=5, help="0 = all matching LCB tasks")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--train-fraction", type=float, default=0.5)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--prior-patches", type=int, default=1)
    p.add_argument("--skip-prior-calibration", action="store_true")
    p.add_argument("--platform", default="leetcode", choices=["leetcode", "atcoder", "codeforces", "all"])
    p.add_argument("--lcb-version", default="all", choices=["v1", "all"])
    p.add_argument("--private-test-cap", type=int, default=12, help="0 = all private tests")
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--max-generations", type=int, default=3)
    p.add_argument("--max-verifications", type=int, default=2)
    p.add_argument(
        "--agent-backend",
        choices=["sage", "langgraph"],
        default="sage",
        help="sage uses sage_agent.langgraph.SAGEGraph; langgraph uses the old local two-node loop.",
    )
    p.add_argument(
        "--final-verify",
        action="store_true",
        help="Always run one terminal hidden-test verification if a candidate exists.",
    )
    p.add_argument("--decision-temperature", type=float, default=0.2)
    p.add_argument("--generation-temperature", type=float, default=0.7)
    p.add_argument("--max-tokens-decision", type=int, default=128)
    p.add_argument("--max-tokens-generation", type=int, default=4000)
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
        default=ORCH_ROOT / "sim_results" / "lcb_llm_tool_agent.jsonl",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--print-each", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _load_env_chain()
    if args.logprobs_output is None:
        args.logprobs_output = args.output.with_name(
            args.output.stem + ".generation_logprobs.jsonl"
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

    generator = canonical_generator_key(args.generator)
    model_id = GENERATORS[generator][0]
    if generator == "gpt_oss_120b_local":
        model_id = os.environ.get("GPT_OSS_120B_MODEL", model_id)
    benchmark = "lcb_medium" if args.benchmark == "lcb_med" else args.benchmark
    difficulty = benchmark.split("_", 1)[1]

    adapter = LCBAdapter(
        benchmark=benchmark,
        difficulty=difficulty,
        n_instances=args.n_instances,
        seed=args.seed,
        platform=args.platform,
        lcb_version=args.lcb_version,
        private_test_cap=args.private_test_cap,
    )
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
        done = set()
        for line in args.output.read_text().splitlines():
            if not line.strip():
                continue
            try:
                done.add(str(json.loads(line).get("instance_id")))
            except json.JSONDecodeError:
                pass
        test_instances = [
            inst for inst in test_instances if adapter.instance_id(inst) not in done
        ]
    else:
        args.output.unlink(missing_ok=True)
        if args.save_generation_logprobs:
            args.logprobs_output.unlink(missing_ok=True)
            args.prior_calibration_logprobs_output.unlink(missing_ok=True)
        args.prior_calibration_output.unlink(missing_ok=True)
        args.actions_output.unlink(missing_ok=True)

    llm_client = _make_client(generator)
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
        )

    graph = build_agent_graph(deps) if args.agent_backend == "langgraph" else None

    print(
        f"LCB LLM tool-agent: benchmark={benchmark} backend={args.agent_backend} model={model_id} "
        f"train={split_summary['n_train']} test={split_summary['n_test']} "
        f"remaining_test={len(test_instances)} output={args.output}"
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
    for idx, instance in enumerate(test_instances, start=1):
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
        )
        try:
            if args.agent_backend == "sage":
                final_state = run_sage_agent_episode(state, deps)
            else:
                final_state = graph.invoke(state)
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

    print(f"done: {args.output}")


if __name__ == "__main__":
    main()
