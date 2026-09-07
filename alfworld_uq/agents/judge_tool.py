"""A judge the agent calls itself, mid-episode, as one more tool.

The offline judge in `experiments.judge_trajectories` runs once per episode
after the data is collected: it cannot touch the trajectory it scores, which is
what makes its verdict an independent signal. This is the opposite object. The
agent decides when to ask, the verdict lands in its context, and the episode
continues -- so the run measures something the offline judge cannot see: whether
an agent knows when it needs checking.

The two are not interchangeable and a run with this tool is not comparable to a
run without it. Keep it behind a flag.

The judge is shown the same allow-listed fields as the offline one (task,
thought, action, observation) and never `won`, `progress`, `done` or
`final_success`, so it cannot leak the environment's answer to the agent.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from experiments.judge_trajectories import (
    SYSTEM_PROMPT,
    build_judge_prompt,
    parse_judge_response,
)


@dataclass
class JudgeVerdict:
    passed: bool | None
    confidence: float | None
    reason: str
    status: str
    step: int = 0
    total_tokens: int = 0

    def as_observation(self) -> str:
        """What the agent reads back. Deliberately terse and never an order."""
        if self.status == "budget_exhausted":
            return "Progress check unavailable: you have used all of your checks."
        if self.status != "ok":
            return f"Progress check unavailable: {self.reason}"
        verdict = "COMPLETE" if self.passed else "NOT COMPLETE"
        confidence = f"{self.confidence:.2f}" if self.confidence is not None else "n/a"
        return (
            f"Progress check: the task looks {verdict} "
            f"(reviewer confidence {confidence}). Reason: {self.reason}"
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "judge_pass": self.passed,
            "confidence": self.confidence,
            "reason": self.reason,
            "status": self.status,
            "total_tokens": self.total_tokens,
        }


@dataclass
class JudgeTool:
    """Bounded, agent-initiated calls to an independent reviewer.

    `budget` caps the calls per episode: without one an agent that distrusts
    itself can spend the whole episode asking, and the cost of the run stops
    being a property of the policy.
    """

    client: Any
    model: str
    budget: int = 5
    max_tokens: int = 256
    retries: int = 2
    calls: list[JudgeVerdict] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        *,
        base_url: str,
        api_key: str,
        model: str,
        budget: int = 5,
        timeout: float = 60.0,
        client: Any | None = None,
    ) -> "JudgeTool":
        return cls(
            client=client
            or OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0),
            model=model,
            budget=budget,
        )

    @property
    def remaining(self) -> int:
        return max(0, self.budget - len(self.calls))

    def reset(self) -> None:
        self.calls = []

    def _rows(self, task: str, history: list[dict[str, str]]) -> list[dict[str, Any]]:
        return [
            {
                "step": index,
                "task": task,
                "thought": item.get("thought", ""),
                "action": item.get("action", ""),
                "observation": item.get("observation", ""),
            }
            for index, item in enumerate(history, 1)
        ]

    def check(self, task: str, history: list[dict[str, str]], *, step: int = 0) -> JudgeVerdict:
        if self.remaining <= 0:
            verdict = JudgeVerdict(None, None, "budget exhausted", "budget_exhausted", step)
            self.calls.append(verdict)
            return verdict
        if not history:
            verdict = JudgeVerdict(
                None, None, "nothing has happened yet", "empty_history", step
            )
            self.calls.append(verdict)
            return verdict

        prompt = build_judge_prompt(self._rows(task, history))
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 2):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                )
                text = response.choices[0].message.content or ""
                passed, confidence, reason = parse_judge_response(text)
                usage = getattr(response, "usage", None)
                verdict = JudgeVerdict(
                    passed,
                    confidence,
                    reason,
                    "ok",
                    step,
                    int(getattr(usage, "total_tokens", 0) or 0),
                )
                self.calls.append(verdict)
                return verdict
            except Exception as exc:
                last_error = exc
                if attempt <= self.retries:
                    time.sleep(min(2 ** (attempt - 1), 4))
        verdict = JudgeVerdict(
            None, None, f"{type(last_error).__name__}", "error", step
        )
        self.calls.append(verdict)
        return verdict
