"""Extract Python code from a freeform LLM response.

LLMs typically wrap code in ```python ... ``` fences. This helper tries
the fenced form first, falling back to the raw response if no fence is
present.
"""
from __future__ import annotations

import re

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n([\s\S]*?)```")
_PY_TAGS = {"", "python", "python3", "py"}


def _compiles(source: str) -> bool:
    try:
        compile(source, "<candidate>", "exec")
    except (SyntaxError, ValueError):
        return False
    return True


def extract_code(response: str) -> str:
    """Pull the most complete Python code block out of a model response.

    Prefer the longest compilable Python or unlabeled fenced block. Fall back
    to the longest such block, then to the raw response when no fence exists.
    """
    if not response:
        return ""
    blocks = [
        body.strip()
        for tag, body in _FENCE.findall(response)
        if tag.lower() in _PY_TAGS and body.strip()
    ]
    if not blocks:
        return response.strip()
    compilable = [block for block in blocks if _compiles(block)]
    return max(compilable or blocks, key=len)
