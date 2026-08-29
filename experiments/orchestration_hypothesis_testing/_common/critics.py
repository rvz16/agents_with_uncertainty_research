"""Language-agnostic Python critics (L0 syntax, L1 lint, L3 LLM review).

These critics take a Python `code` string and return a bool (or for L3,
a (bool | None, cost_usd) tuple). They are independent of benchmark format -- a
critic doesn't know if it's reviewing an LCB solution or a MBPP function.

Originally lived inside lcb_calibrate.py; extracted here so every
calibration/iter pipeline shares one implementation.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass

log = logging.getLogger(__name__)


def critic_L0_syntax(code: str) -> bool:
    """L0: parses cleanly under ast.parse(). Essentially free."""
    if not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def critic_L1_lint(code: str) -> bool:
    """L1: ruff with conservative ruleset (F821, F811, E999).

    Same ruleset as the SWE-bench Lite production critic. If ruff isn't
    installed, this returns True (don't fail-closed on missing tooling).
    """
    if not code.strip():
        return False
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        proc = subprocess.run(
            ["ruff", "check", "--quiet", "--no-cache", "--select", "F821,F811,E999", tmp],
            capture_output=True, text=True, timeout=15,
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return True
    finally:
        os.unlink(tmp)


@dataclass(frozen=True)
class L3ReviewResult:
    passed: bool | None
    reasoning: str
    raw_response: str
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int


def _parse_l3_response(text: str) -> tuple[bool | None, str]:
    candidates = [text.strip()]
    candidates[:0] = re.findall(
        r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
    )
    decoder = json.JSONDecoder()
    for candidate in candidates:
        starts = [0] + [i for i, char in enumerate(candidate) if char == "{"]
        for start in starts:
            try:
                obj, _ = decoder.raw_decode(candidate[start:].lstrip())
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(obj, dict):
                continue
            verdict = str(
                obj.get("verdict") or obj.get("answer") or obj.get("result") or ""
            ).strip().upper()
            if verdict not in {"PASS", "FAIL"}:
                continue
            reasoning = str(obj.get("reasoning") or obj.get("reason") or "").strip()
            return verdict == "PASS", reasoning

    # Keep accepting legacy one-word/free-text reviewer responses.
    verdicts = re.findall(r"\b(PASS|FAIL)\b", text.upper())
    return (verdicts[-1] == "PASS", "") if verdicts else (None, "")


def critic_L3_review_detailed(problem: str, code: str, client) -> L3ReviewResult:
    """L3: JSON PASS/FAIL judgment with a short explanation.

    Default reviewer model is claude-haiku-4.5. Caller passes an
    OpenAI-compatible client (typically the OpenRouter client). On an API or
    parsing failure, ``passed`` is None and a warning is logged.
    """
    prompt = (
        "You are a senior software engineer reviewing a code submission.\n\n"
        f"## Problem\n{problem[:3000]}\n\n"
        f"## Submitted code\n```python\n{code[:6000]}\n```\n\n"
        "Does this code correctly solve the problem? Reason briefly, then return "
        "exactly one compact JSON object with keys in this order:\n"
        '{"reasoning":"short reason, at most two sentences","verdict":"PASS|FAIL"}\n'
        "Return no markdown and no text outside the JSON object."
    )
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("L3_REVIEW_MODEL", "anthropic/claude-haiku-4.5"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            # 4096 is ample for a non-reasoning judge, but a local reasoning model
            # spends its budget deliberating and the JSON verdict never arrives:
            # the review comes back unparseable, and the analyzer refuses to fit
            # likelihoods when any train instance lacks a verdict. Raise it via
            # L3_MAX_TOKENS when the reviewer reasons before answering.
            max_tokens=int(os.environ.get("L3_MAX_TOKENS", "4096")),
        )
        message = resp.choices[0].message
        text = (message.content or getattr(message, "reasoning_content", "") or "").strip()
        passed, reasoning = _parse_l3_response(text)
        usage = resp.usage
        cost = (usage.prompt_tokens / 1_000_000) * 1.0 + (usage.completion_tokens / 1_000_000) * 5.0
        if passed is None:
            log.warning("L3 returned no PASS/FAIL verdict")
        return L3ReviewResult(
            passed=passed,
            reasoning=reasoning,
            raw_response=text,
            cost_usd=cost,
            prompt_tokens=int(usage.prompt_tokens),
            completion_tokens=int(usage.completion_tokens),
        )
    except Exception as e:
        log.warning("L3 failed: %s", e)
        return L3ReviewResult(None, "", "", 0.0, 0, 0)


def critic_L3_review(problem: str, code: str, client) -> tuple[bool | None, float]:
    """Backward-compatible wrapper for calibration and legacy callers."""
    result = critic_L3_review_detailed(problem, code, client)
    return result.passed, result.cost_usd
