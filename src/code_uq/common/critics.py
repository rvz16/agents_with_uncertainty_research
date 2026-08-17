"""Language-agnostic Python critics (L0 syntax, L1 lint, L3 LLM review).

These critics take a Python `code` string and return a bool (or for L3,
a (bool, cost_usd) tuple). They are independent of benchmark format -- a
critic doesn't know if it's reviewing an LCB solution or a MBPP function.

Originally lived inside lcb_calibrate.py; extracted here so every
calibration/iter pipeline shares one implementation.
"""
from __future__ import annotations

import ast
import logging
import os
import shutil
import tempfile


log = logging.getLogger(__name__)

#: Only real errors, no stylistic rules: undefined name and redefinition.
#: Style warnings fire constantly on large legacy code bases and would make the
#: critic a near-constant signal.
#:
#: ``E999`` (syntax error) was part of this set historically but modern ruff
#: removed it as a selectable rule and exits with a *configuration error* when
#: asked for it -- which made the critic answer the same way for every input.
#: Syntax is reported unconditionally by ruff, and the L0 critic covers it via
#: ``ast.parse`` anyway.
RUFF_RULES = "F821,F811"

#: ruff exits 0 when clean, 1 when it found violations, and 2 when it could not
#: run at all. Only the first two are verdicts.
_RUFF_CLEAN = 0
_RUFF_VIOLATIONS = 1


class LintToolMissing(RuntimeError):
    """``ruff`` is not installed.

    Raised rather than returning a verdict.  A lint critic that answers "pass"
    whenever its tool is missing is a constant, and a constant critic scores a
    textbook AUROC of exactly 0.500 that reads as an honest negative result.
    Failing loudly at start-up costs one confused minute; failing quietly costs
    a whole measurement.
    """


def resolve_ruff() -> str:
    """Absolute path to ``ruff``.

    Resolved in the parent so the child does not depend on PATH lookup
    that does not include the ``--target`` install directory.
    """
    found = shutil.which("ruff")
    if not found:
        raise LintToolMissing(
            "ruff not found on PATH; install it into the pinned dependency "
            "directory and re-run (the L1 critic must not silently pass)"
        )
    return found


def critic_L0_syntax(code: str) -> bool:
    """L0: parses cleanly under ast.parse(). Essentially free."""
    if not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _is_loopback_url(url: str) -> bool:
    """True when the endpoint is local, so no third party is billed for it."""
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def critic_L1_lint(code: str) -> bool:
    """L1: ruff with a conservative ruleset (see :data:`RUFF_RULES`).

    Raises :class:`LintToolMissing` when ruff is absent; a timeout is reported
    as a failed check rather than a pass, so an unusable verdict never
    masquerades as a clean one.
    """
    if not code.strip():
        return False
    ruff = resolve_ruff()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        result = subprocess.run(
            [ruff, "check", "--quiet", "--no-cache", "--select", RUFF_RULES, path],
            capture_output=True, text=True, timeout=RUFF_TIMEOUT_S,
        )
        if result.timed_out:
            log.warning("ruff timed out; reporting L1 as failed")
            return False
        if result.returncode not in (_RUFF_CLEAN, _RUFF_VIOLATIONS):
            raise LintToolMissing(
                f"ruff could not run (exit={result.returncode}): "
                f"{result.stderr.strip()[:300]}"
            )
        return result.returncode == _RUFF_CLEAN
    finally:
        os.unlink(tmp)


#: A reasoning model spends its budget on the analysis channel before it says
#: anything on the answer channel. The historical value here was 10 tokens,
#: which is fine for a non-reasoning judge and never reaches a verdict with one
#: -- the judge then returned no opinion on every single episode.
#:
#: This is a safety valve, not a budget. The verdict itself is one word, so the
#: only thing a larger ceiling buys is room for the hard cases to finish
#: thinking; the median call costs the same either way. Measured on LCB
#: atcoder/hard, a 2048-token ceiling lost the verdict on 4 of 5 calls
#: (``finish_reason=length``), which is a critic that mostly abstains.
L3_REVIEW_MAX_TOKENS = int(os.environ.get("L3_REVIEW_MAX_TOKENS", "8192"))


#: Hosted-reviewer pricing (USD per million tokens), roughly Haiku 4.5.
_REVIEW_PRICE_IN, _REVIEW_PRICE_OUT = 1.0, 5.0


def _review_cost_usd(client, prompt_tokens: int, completion_tokens: int) -> float:
    """Dollar cost of one reviewer call.

    A judge served from the local vLLM costs nothing, so charging it hosted
    prices puts an invented number in every run's ``api_cost_usd`` -- and a
    fabricated cost is worse than no cost, because it looks like a measurement.
    """
    base_url = getattr(client, "base_url", None)
    if base_url is not None and _is_loopback_url(str(base_url)):
        return 0.0
    return (
        prompt_tokens / 1_000_000 * _REVIEW_PRICE_IN
        + completion_tokens / 1_000_000 * _REVIEW_PRICE_OUT
    )


def _judge_verdict(client, prompt: str, *, label: str) -> tuple[bool | None, float]:
    """Ask the reviewer model for a PASS/FAIL verdict.

    Returns ``(passed, cost_usd)`` with ``passed=None`` when no verdict was
    obtained -- unreachable endpoint, or a response that never reached the
    answer channel. ``None`` is deliberately distinct from ``False``: an
    unreachable judge that reports FAIL turns L3 into a constant, and a
    constant critic scores AUROC 0.500 while looking like a real finding.
    """
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("L3_REVIEW_MODEL", "anthropic/claude-haiku-4.5"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=L3_REVIEW_MAX_TOKENS,
        )
    except Exception as exc:
        log.warning("L3 reviewer unreachable for %s: %s", label, exc)
        return None, 0.0

    message = resp.choices[0].message
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    cost = _review_cost_usd(client, prompt_tokens, completion_tokens)

    # Only the answer channel counts. Reasoning text weighs both words while it
    # deliberates, so scanning it for "PASS" would read the judge's thinking as
    # its conclusion.
    content = getattr(message, "content", None)
    if not content or not str(content).strip():
        finish_reason = getattr(resp.choices[0], "finish_reason", "")
        log.warning(
            "L3 reviewer produced no answer for %s (finish_reason=%s, "
            "completion_tokens=%s); raise L3_REVIEW_MAX_TOKENS if this is common",
            label, finish_reason, completion_tokens,
        )
        return None, cost

    text = str(content).strip().upper()
    if "PASS" not in text and "FAIL" not in text:
        log.warning("L3 reviewer gave no verdict for %s: %r", label, text[:120])
        return None, cost
    return ("PASS" in text and "FAIL" not in text), cost


def critic_L3_review(
    problem: str, code: str, client, *, label: str = "candidate"
) -> tuple[bool | None, float]:
    """L3: reviewer-model PASS/FAIL judgment on (problem, code)."""
    prompt = (
        "You are a senior software engineer reviewing a code submission.\n\n"
        f"## Problem\n{problem[:3000]}\n\n"
        f"## Submitted code\n```python\n{code[:6000]}\n```\n\n"
        "Does this code correctly solve the problem? Answer with exactly one "
        "word: PASS or FAIL."
    )
    return _judge_verdict(client, prompt, label=label)
