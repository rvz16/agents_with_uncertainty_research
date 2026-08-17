"""Tests for solution extraction.

The regression these pin down: reasoning models fence their derivations before
they fence the solution, and taking the first fence returned the derivation.
"""
from __future__ import annotations

from code_uq.common.extract import extract_code

SOLUTION = "import sys\n\n\ndef solve():\n    print(sum(map(int, sys.stdin.read().split())))\n"


def test_single_fence():
    assert extract_code(f"Here you go:\n```python\n{SOLUTION}```\n") == SOLUTION.strip()


def test_unlabelled_fence():
    assert extract_code(f"```\n{SOLUTION}```") == SOLUTION.strip()


def test_prose_without_fence_returns_raw():
    assert extract_code("  no fences here  ") == "no fences here"


def test_empty():
    assert extract_code("") == ""


def test_skips_leading_derivation_block():
    """The bug: a fenced formula before the solution used to win."""
    resp = (
        "First the recurrence:\n\n```\ndist[1] = 0\nsub[1] = N\n```\n\n"
        "which gives the answer. Final solution:\n\n"
        f"```python\n{SOLUTION}```\n"
    )
    assert extract_code(resp) == SOLUTION.strip()


def test_skips_non_python_fence_that_does_not_parse():
    resp = (
        "```\nE = Σ  P(i) · pos(i)\n```\n"
        f"```python\n{SOLUTION}```\n"
    )
    assert extract_code(resp) == SOLUTION.strip()


def test_ignores_non_python_language_tags():
    resp = (
        "```text\n5 3\n1 2 3 4 5\n```\n"
        "```bash\npython3 main.py < input.txt\n```\n"
        f"```python\n{SOLUTION}```\n"
    )
    assert extract_code(resp) == SOLUTION.strip()


def test_prefers_longest_parsing_block_over_trailing_snippet():
    resp = (
        f"```python\n{SOLUTION}```\n"
        "Example call:\n```python\nsolve()\n```\n"
    )
    assert extract_code(resp) == SOLUTION.strip()


def test_falls_back_to_longest_when_nothing_parses():
    resp = "```\nfoo(\n```\n```\nbar((( and a much longer broken fragment\n```\n"
    assert "much longer broken fragment" in extract_code(resp)


def test_leetcode_class_solution_still_extracted():
    code = "class Solution:\n    def f(self, n: int) -> int:\n        return n\n"
    assert extract_code(f"Reasoning...\n```python\n{code}```") == code.strip()
