"""Unit tests for the indentation-tolerant SEARCH/REPLACE matcher.

Run with:
    pytest scripts/test_indent_matcher.py -v

These cases lock in behaviour around the bug discovered in the R1
spot-check, where Qwen models emitted SEARCH blocks at a shallower
indent than the file actually had (e.g. inner-`if` with 8sp instead of
12sp), causing every SEARCH to silently miss.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from spot_check_generators import (  # noqa: E402
    _common_indent,
    _try_indent_tolerant_replace,
    apply_change_blocks,
)


# ---------- _common_indent ----------

def test_common_indent_uniform():
    assert _common_indent(["    a", "    b"]) == "    "


def test_common_indent_increasing():
    assert _common_indent(["    a", "        b"]) == "    "


def test_common_indent_partial():
    assert _common_indent(["  a", "    b"]) == "  "


def test_common_indent_blank_lines_ignored():
    assert _common_indent(["", "    a", ""]) == "    "


def test_common_indent_whitespace_only_ignored():
    # whitespace-only "indent" carries no signal
    assert _common_indent(["        ", "    a"]) == "    "


def test_common_indent_empty():
    assert _common_indent([]) == ""


def test_common_indent_no_indent():
    assert _common_indent(["a", "b"]) == ""


# ---------- _try_indent_tolerant_replace ----------

def test_indent_tolerant_replace_matches_shallower_search():
    """Model emits SEARCH at shallower indent; matcher dedents both and
    re-indents REPLACE to the file's actual nesting level."""
    file_text = (
        "def f():\n"
        "    if True:\n"
        "        x = 1\n"
        "        y = 2\n"
    )
    search = "if True:\n    x = 1\n    y = 2"
    replace = "if True:\n    x = 99\n    y = 2"
    out = _try_indent_tolerant_replace(file_text, search, replace)
    assert out is not None
    assert "        x = 99" in out
    assert "        y = 2" in out
    # untouched lines are preserved
    assert "def f():" in out


def test_indent_tolerant_replace_strict_match_path_unchanged():
    file_text = "alpha\nbeta\n"
    out = _try_indent_tolerant_replace(file_text, "alpha\nbeta", "gamma\ndelta")
    assert out is not None
    assert "gamma" in out and "delta" in out
    assert "alpha" not in out


def test_indent_tolerant_replace_no_spurious_match():
    """SEARCH text that doesn't actually exist must not be wedged in."""
    file_text = "def a():\n    return 1\ndef b():\n    return 2\n"
    out = _try_indent_tolerant_replace(file_text, "return 99", "return 100")
    assert out is None


def test_indent_tolerant_replace_handles_blank_lines_in_block():
    file_text = (
        "class C:\n"
        "    def m(self):\n"
        "        a = 1\n"
        "\n"
        "        b = 2\n"
    )
    search = "a = 1\n\nb = 2"
    replace = "a = 10\n\nb = 20"
    out = _try_indent_tolerant_replace(file_text, search, replace)
    assert out is not None
    assert "        a = 10" in out
    assert "        b = 20" in out


def test_indent_tolerant_replace_first_match_wins():
    """When the dedented SEARCH matches multiple windows, take the first
    (matches str.replace(..., 1) semantics)."""
    file_text = (
        "if A:\n"
        "    x = 1\n"
        "if B:\n"
        "    x = 1\n"
    )
    out = _try_indent_tolerant_replace(file_text, "x = 1", "x = 99")
    assert out is not None
    # First occurrence flipped, second untouched
    assert out.count("x = 99") == 1
    assert out.count("x = 1") == 1


def test_indent_tolerant_replace_blank_replace_lines_not_padded():
    """Blank lines in REPLACE must stay blank (no leading-indent prefix
    applied), to avoid trailing whitespace warnings."""
    file_text = "    def m():\n        pass\n"
    search = "def m():\n    pass"
    replace = "def m():\n\n    pass"
    out = _try_indent_tolerant_replace(file_text, search, replace)
    assert out is not None
    assert "    def m():\n\n        pass" in out


# ---------- apply_change_blocks integration ----------

def test_apply_change_blocks_uses_indent_tier_when_strict_fails():
    oracle = {
        "m.py": "class C:\n    def m(self):\n        if x:\n            return 1\n",
    }
    blocks = [(
        "m.py",
        "if x:\n    return 1",       # 0/4-space SEARCH
        "if x:\n    return 99",      # same 0/4-space REPLACE
    )]
    modified = apply_change_blocks(oracle, blocks)
    assert "m.py" in modified
    text = modified["m.py"]
    assert "        if x:\n            return 99" in text


def test_apply_change_blocks_strict_still_takes_priority():
    """If the strict literal substring matches, we don't fall through to
    the indent path (faster and less ambiguous)."""
    oracle = {"m.py": "alpha\nbeta\ngamma\n"}
    blocks = [("m.py", "alpha\nbeta", "ALPHA\nBETA")]
    modified = apply_change_blocks(oracle, blocks)
    assert modified["m.py"] == "ALPHA\nBETA\ngamma\n"


def test_apply_change_blocks_returns_empty_when_no_match():
    oracle = {"m.py": "hello\nworld\n"}
    blocks = [("m.py", "not present", "something")]
    modified = apply_change_blocks(oracle, blocks)
    assert "m.py" not in modified
