"""Extract Python code from a freeform LLM response.

LLMs typically wrap code in ```python ... ``` fences. This helper tries
the fenced form first, falling back to the raw response if no fence is
present.
"""
from __future__ import annotations

import re

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n([\s\S]*?)```")
_PY_TAGS = {"", "python", "python3", "py"}


def _compiles(src: str) -> bool:
    try:
        compile(src, "<candidate>", "exec")
    except (SyntaxError, ValueError):
        return False
    return True


def extract_code(response: str) -> str:
    """Pull the Python solution out of a model response.

    Reasoning models routinely fence things that are not the solution --
    derivations, complexity bounds, pseudo-code, sample I/O -- *before* the
    real code. Taking the first fence therefore returned fragments like
    ``dist[1] = 0`` or ``E = Σ P(i) · pos(i)``: on LCB-Hard/atcoder this hit
    51 of 102 gpt-oss-120b generations, every one of which then failed. The
    effect scales with how much a model explains itself, so it penalises the
    stronger model hardest and silently corrupts any cross-model comparison.

    So: consider every fenced block, keep the ones that actually parse as
    Python, and return the longest. Falls back to the longest block when none
    parses, and to the raw response when there is no fence at all.
    """
    if not response:
        return ""
    blocks = [
        body.strip()
        for tag, body in _FENCE.findall(response)
        if tag.lower() in _PY_TAGS
    ]
    blocks = [b for b in blocks if b]
    if not blocks:
        return response.strip()
    parsing = [b for b in blocks if _compiles(b)]
    return max(parsing or blocks, key=len)
