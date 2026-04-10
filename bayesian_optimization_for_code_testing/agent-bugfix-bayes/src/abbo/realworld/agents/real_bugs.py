"""Realistic bug definitions for comparing Simple Agent vs Bayesian POMDP Agent.

Uses a log parser module — a common real-world component — with bugs at
graduated difficulty levels designed to hit ~30-60% fix rate on 7B models.

Bug difficulty is tuned so the Bayesian agent's adaptive decisions matter:
- Easy bugs: both agents fix them, but Bayesian saves cost by verifying sooner
- Medium bugs: Bayesian uses cheap critics to decide strategy adaptively
- Hard bugs: Bayesian bails out early, saving cost on unfixable bugs
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Clean source: a log parser with timestamp handling, filtering, aggregation
# ---------------------------------------------------------------------------

LOG_PARSER_CLEAN = '''\
"""Log parser: parse, filter, and summarize log records."""

import re
from datetime import datetime

LEVELS = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
LOG_RE = re.compile(
    r"\\[(\\d{4}-\\d{1,2}-\\d{1,2} \\d{1,2}:\\d{2}:\\d{2})\\]"
    r" (DEBUG|INFO|WARNING|ERROR|CRITICAL)"
    r" (\\S+):(\\d+) - (.*)"
)

def parse_line(line):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    m = LOG_RE.match(line)
    if not m:
        return None
    ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    return {"ts": ts, "level": m.group(2), "src": m.group(3),
            "lineno": int(m.group(4)), "msg": m.group(5).strip()}

def parse_log(text):
    return [r for line in text.splitlines() if (r := parse_line(line)) is not None]

def filter_by_level(records, min_level):
    threshold = LEVELS.get(min_level.upper(), 0)
    return [r for r in records if LEVELS.get(r["level"], 0) >= threshold]

def filter_by_time(records, start=None, end=None):
    out = []
    for r in records:
        if start and r["ts"] < start:
            continue
        if end and r["ts"] > end:
            continue
        out.append(r)
    return out

def filter_by_source(records, source, exact=False):
    if exact:
        return [r for r in records if r["src"] == source]
    return [r for r in records if source in r["src"]]

def count_by_level(records):
    counts = {}
    for r in records:
        counts[r["level"]] = counts.get(r["level"], 0) + 1
    return counts

def count_by_source(records):
    counts = {}
    for r in records:
        counts[r["src"]] = counts.get(r["src"], 0) + 1
    return counts

def error_windows(records, n_before=2, n_after=2):
    windows = []
    for i, r in enumerate(records):
        if r["level"] in ("ERROR", "CRITICAL"):
            s = max(0, i - n_before)
            e = min(len(records), i + n_after + 1)
            windows.append(records[s:e])
    return windows

def error_rate(records):
    if not records:
        return 0.0
    errs = sum(1 for r in records if r["level"] in ("ERROR", "CRITICAL"))
    return errs / len(records)

def latest_by_source(records):
    latest = {}
    for r in records:
        if r["src"] not in latest or r["ts"] > latest[r["src"]]["ts"]:
            latest[r["src"]] = r
    return latest

def top_sources(records, n=5):
    counts = count_by_source(records)
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]
'''


# ---------------------------------------------------------------------------
# Buggy variants — graduated difficulty
# ---------------------------------------------------------------------------

BUGGY_SOURCES_REAL: dict[str, str] = {}

# Bug R1 (EASY): filter_by_level uses > instead of >=
BUGGY_SOURCES_REAL["bug_r1"] = LOG_PARSER_CLEAN.replace(
    'LEVELS.get(r["level"], 0) >= threshold',
    'LEVELS.get(r["level"], 0) > threshold',
)

# Bug R2 (EASY): filter_by_source exact match uses 'in' instead of '=='
BUGGY_SOURCES_REAL["bug_r2"] = LOG_PARSER_CLEAN.replace(
    '        return [r for r in records if r["src"] == source]',
    '        return [r for r in records if source in r["src"]]',
)

# Bug R3 (MEDIUM): filter_by_time end boundary inverted
BUGGY_SOURCES_REAL["bug_r3"] = LOG_PARSER_CLEAN.replace(
    'if end and r["ts"] > end:',
    'if end and r["ts"] < end:',
)

# Bug R4 (MEDIUM): error_windows off-by-one (missing +1)
BUGGY_SOURCES_REAL["bug_r4"] = LOG_PARSER_CLEAN.replace(
    'e = min(len(records), i + n_after + 1)',
    'e = min(len(records), i + n_after)',
)

# Bug R5 (MEDIUM): error_rate counts only ERROR, not CRITICAL
BUGGY_SOURCES_REAL["bug_r5"] = LOG_PARSER_CLEAN.replace(
    'errs = sum(1 for r in records if r["level"] in ("ERROR", "CRITICAL"))',
    'errs = sum(1 for r in records if r["level"] == "ERROR")',
)

# Bug R6 (HARD): latest_by_source uses < instead of > (returns oldest)
BUGGY_SOURCES_REAL["bug_r6"] = LOG_PARSER_CLEAN.replace(
    'r["ts"] > latest[r["src"]]["ts"]',
    'r["ts"] < latest[r["src"]]["ts"]',
)

# Bug R7 (HARD): top_sources sorts ascending instead of descending
BUGGY_SOURCES_REAL["bug_r7"] = LOG_PARSER_CLEAN.replace(
    "return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]",
    "return sorted(counts.items(), key=lambda x: x[1])[:n]",
)

# Bug R8 (HARD): LEVELS dict missing CRITICAL entry
BUGGY_SOURCES_REAL["bug_r8"] = LOG_PARSER_CLEAN.replace(
    'LEVELS = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}',
    'LEVELS = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}',
)

# Bug R9 (EASY): filter_by_time start boundary inverted (< vs >)
BUGGY_SOURCES_REAL["bug_r9"] = LOG_PARSER_CLEAN.replace(
    'if start and r["ts"] < start:',
    'if start and r["ts"] > start:',
)

# Bug R10 (EASY): count_by_level off-by-one (default 1 instead of 0)
BUGGY_SOURCES_REAL["bug_r10"] = LOG_PARSER_CLEAN.replace(
    'counts[r["level"]] = counts.get(r["level"], 0) + 1',
    'counts[r["level"]] = counts.get(r["level"], 1) + 1',
)

# Bug R11 (EASY): parse_log splits on whitespace instead of newlines
BUGGY_SOURCES_REAL["bug_r11"] = LOG_PARSER_CLEAN.replace(
    'return [r for line in text.splitlines() if (r := parse_line(line)) is not None]',
    'return [r for line in text.split() if (r := parse_line(line)) is not None]',
)

# Bug R12 (EASY): LEVELS dict wrong WARNING value (2 -> 3, same as ERROR)
BUGGY_SOURCES_REAL["bug_r12"] = LOG_PARSER_CLEAN.replace(
    '"WARNING": 2, "ERROR": 3,',
    '"WARNING": 3, "ERROR": 3,',
)

# Bug R13 (MEDIUM): parse_line swaps level and src groups
BUGGY_SOURCES_REAL["bug_r13"] = LOG_PARSER_CLEAN.replace(
    '"level": m.group(2), "src": m.group(3),',
    '"level": m.group(3), "src": m.group(2),',
)

# Bug R14 (MEDIUM): error_windows start index uses n_after instead of n_before
BUGGY_SOURCES_REAL["bug_r14"] = LOG_PARSER_CLEAN.replace(
    's = max(0, i - n_before)',
    's = max(0, i - n_after)',
)

# Bug R15 (MEDIUM): error_rate divides by len-1 (off-by-one)
BUGGY_SOURCES_REAL["bug_r15"] = LOG_PARSER_CLEAN.replace(
    'return errs / len(records)',
    'return errs / (len(records) - 1)',
)

# Bug R16 (MEDIUM): count_by_source uses r["level"] instead of r["src"]
BUGGY_SOURCES_REAL["bug_r16"] = LOG_PARSER_CLEAN.replace(
    'counts[r["src"]] = counts.get(r["src"], 0) + 1',
    'counts[r["level"]] = counts.get(r["level"], 0) + 1',
)

# Bug R17 (HARD): latest_by_source uses level as dict key instead of src
BUGGY_SOURCES_REAL["bug_r17"] = LOG_PARSER_CLEAN.replace(
    'latest[r["src"]] = r',
    'latest[r["level"]] = r',
)

# Bug R18 (HARD): error_windows only catches ERROR, not CRITICAL
BUGGY_SOURCES_REAL["bug_r18"] = LOG_PARSER_CLEAN.replace(
    'if r["level"] in ("ERROR", "CRITICAL"):',
    'if r["level"] == "ERROR":',
)

# Bug R19 (HARD): error_rate inverted fraction (len/errs instead of errs/len)
BUGGY_SOURCES_REAL["bug_r19"] = LOG_PARSER_CLEAN.replace(
    'return errs / len(records)',
    'return len(records) / errs',
)

# Bug R20 (HARD): filter_by_level uses .lower() instead of .upper()
BUGGY_SOURCES_REAL["bug_r20"] = LOG_PARSER_CLEAN.replace(
    'threshold = LEVELS.get(min_level.upper(), 0)',
    'threshold = LEVELS.get(min_level.lower(), 0)',
)


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

LOG_PARSER_TEST_CONTENT = '''"""Tests for the log parser module."""

import sys
sys.path.insert(0, "{src_dir}")

from datetime import datetime
from log_parser import (
    parse_line, parse_log, filter_by_level, filter_by_time,
    filter_by_source, count_by_level, count_by_source,
    error_windows, error_rate, latest_by_source, top_sources,
)

SAMPLE_LOG = """\\
[2024-01-15 08:00:00] DEBUG server.py:10 - Starting server
[2024-01-15 08:00:01] INFO server.py:20 - Listening on 8080
[2024-01-15 08:00:05] INFO auth.py:15 - User admin logged in
[2024-01-15 08:01:00] WARNING db.py:45 - Pool running low
[2024-01-15 08:01:30] ERROR db.py:52 - Connection timeout
[2024-01-15 08:01:31] INFO db.py:60 - Retrying connection
[2024-01-15 08:02:00] ERROR server.py:88 - Handler failed
[2024-01-15 08:02:01] CRITICAL server.py:90 - Shutting down
[2024-01-15 08:05:00] INFO server.py:95 - Server restarted
[2024-01-15 09:00:00] DEBUG auth.py:10 - Session cleanup
"""


# ===== Parsing tests (critic group: parsing_check) =====

def test_parse_line_basic():
    r = parse_line("[2024-01-15 08:00:00] DEBUG server.py:10 - Starting")
    assert r is not None
    assert r["level"] == "DEBUG"
    assert r["src"] == "server.py"

def test_parse_log_count():
    assert len(parse_log(SAMPLE_LOG)) == 10

def test_parse_line_blank():
    assert parse_line("") is None
    assert parse_line("# comment") is None


# ===== Filtering tests (critic group: filtering_check) =====

def test_filter_by_level_warning():
    recs = parse_log(SAMPLE_LOG)
    filtered = filter_by_level(recs, "WARNING")
    assert len(filtered) == 4  # WARNING + 2 ERROR + CRITICAL

def test_filter_by_level_error():
    recs = parse_log(SAMPLE_LOG)
    filtered = filter_by_level(recs, "ERROR")
    assert len(filtered) == 3  # 2 ERROR + 1 CRITICAL

def test_filter_by_level_critical():
    recs = parse_log(SAMPLE_LOG)
    filtered = filter_by_level(recs, "CRITICAL")
    assert len(filtered) == 1

def test_filter_by_level_includes_threshold():
    recs = parse_log(SAMPLE_LOG)
    filtered = filter_by_level(recs, "ERROR")
    error_recs = [r for r in filtered if r["level"] == "ERROR"]
    assert len(error_recs) == 2

def test_filter_by_time_inclusive():
    recs = parse_log(SAMPLE_LOG)
    start = datetime(2024, 1, 15, 8, 1, 0)
    end = datetime(2024, 1, 15, 8, 2, 0)
    filtered = filter_by_time(recs, start=start, end=end)
    assert any(r["ts"] == start for r in filtered)
    assert any(r["ts"] == end for r in filtered)

def test_filter_by_time_end_boundary():
    recs = parse_log(SAMPLE_LOG)
    end = datetime(2024, 1, 15, 8, 1, 30)
    filtered = filter_by_time(recs, end=end)
    assert any(r["ts"] == end for r in filtered)

def test_filter_by_source_exact():
    recs = parse_log(SAMPLE_LOG)
    assert len(filter_by_source(recs, "server.py", exact=True)) == 5
    assert len(filter_by_source(recs, "server", exact=True)) == 0


# ===== Aggregation tests (critic group: aggregation_check) =====

def test_count_by_level():
    recs = parse_log(SAMPLE_LOG)
    c = count_by_level(recs)
    assert c["DEBUG"] == 2
    assert c["INFO"] == 4
    assert c["ERROR"] == 2
    assert c["CRITICAL"] == 1

def test_error_rate_calculation():
    recs = parse_log(SAMPLE_LOG)
    assert abs(error_rate(recs) - 0.3) < 0.01  # 3 errors out of 10

def test_error_rate_empty():
    assert error_rate([]) == 0.0


# ===== Context & summary tests (critic group: context_check) =====

def test_error_windows_count():
    recs = parse_log(SAMPLE_LOG)
    wins = error_windows(recs, n_before=2, n_after=2)
    assert len(wins) == 3  # 2 ERROR + 1 CRITICAL

def test_error_windows_includes_after():
    recs = parse_log(SAMPLE_LOG)
    wins = error_windows(recs, n_before=0, n_after=2)
    assert wins[0][0]["level"] == "ERROR"
    assert len(wins[0]) == 3  # error + 2 after

def test_latest_by_source():
    recs = parse_log(SAMPLE_LOG)
    latest = latest_by_source(recs)
    assert latest["server.py"]["ts"] == datetime(2024, 1, 15, 8, 5, 0)
    assert latest["auth.py"]["ts"] == datetime(2024, 1, 15, 9, 0, 0)
    assert latest["db.py"]["ts"] == datetime(2024, 1, 15, 8, 1, 31)

def test_top_sources_order():
    recs = parse_log(SAMPLE_LOG)
    top = top_sources(recs)
    assert top[0][0] == "server.py"  # most common (5 records)
    assert top[0][1] == 5
'''


# ---------------------------------------------------------------------------
# Critic test subsets (cheap diagnostic tests for the Bayesian agent)
# ---------------------------------------------------------------------------

REAL_CRITIC_TESTS = {
    "parsing_check": [
        "test_parse_line_basic",
        "test_parse_log_count",
        "test_parse_line_blank",
    ],
    "filtering_check": [
        "test_filter_by_level_warning",
        "test_filter_by_level_error",
        "test_filter_by_source_exact",
        "test_filter_by_level_includes_threshold",
    ],
    "aggregation_check": [
        "test_count_by_level",
        "test_error_rate_calculation",
    ],
    "context_check": [
        "test_error_windows_includes_after",
        "test_latest_by_source",
        "test_top_sources_order",
    ],
}


# ---------------------------------------------------------------------------
# LLM prompt templates for generating fixes
# ---------------------------------------------------------------------------

REAL_ARM_PROMPTS = {
    "g1_direct_fix": (
        "Below is a Python module with a bug. Some tests are failing.\n\n"
        "IMPORTANT RULES:\n"
        "- Return the ENTIRE source code of the module, not just a snippet.\n"
        "- Keep ALL imports, classes, functions, and dataclasses EXACTLY as they are.\n"
        "- Only change the ONE line that has the bug. Do NOT rewrite or restructure.\n"
        "- The fix is likely a single character or operator change.\n\n"
        "Source code:\n```python\n{source_code}\n```\n\n"
        "Failing test output:\n```\n{test_output}\n```\n\n"
        "Return the COMPLETE fixed module in a single ```python ... ``` block. "
        "The output must contain every function from the original."
    ),
    "g2_localized_fix": (
        "A Python module has a bug causing test failures. "
        "Your job: find the ONE broken line and fix it.\n\n"
        "RULES:\n"
        "- Copy the ENTIRE source code below and fix only the broken line.\n"
        "- Do NOT remove, rename, or restructure any functions.\n"
        "- Do NOT change imports or class definitions.\n"
        "- The bug is a wrong operator, wrong variable, or wrong constant.\n\n"
        "Source code:\n```python\n{source_code}\n```\n\n"
        "Test failures:\n```\n{test_output}\n```\n\n"
        "Think step by step:\n"
        "1. Which tests fail and what do they check?\n"
        "2. Which function has the wrong behavior?\n"
        "3. What single-line change fixes it?\n\n"
        "Return the COMPLETE module (all functions) in a ```python ... ``` block."
    ),
    "g3_test_guided_fix": (
        "Fix the bug in this module. The test code shows expected behavior.\n\n"
        "CRITICAL: Return the ENTIRE source code with the fix applied. "
        "Do NOT rewrite the module. Do NOT remove any functions. "
        "Only change the line that causes the test failure.\n\n"
        "Source code:\n```python\n{source_code}\n```\n\n"
        "Test code:\n```python\n{test_code}\n```\n\n"
        "Test output:\n```\n{test_output}\n```\n\n"
        "Return the COMPLETE fixed module in a ```python ... ``` block."
    ),
    "g4_error_focused_fix": (
        "A Python module has a subtle bug — likely a wrong operator or value.\n\n"
        "RULES: Copy the ENTIRE source code and change ONLY the broken line. "
        "Keep all functions, classes, imports exactly as-is.\n\n"
        "Source code:\n```python\n{source_code}\n```\n\n"
        "Errors:\n```\n{test_output}\n```\n\n"
        "Return the COMPLETE fixed module in a ```python ... ``` block."
    ),
}


# ---------------------------------------------------------------------------
# Bug metadata (for reporting)
# ---------------------------------------------------------------------------

BUG_METADATA = {
    "bug_r1":  {"difficulty": "easy",   "description": "filter_by_level: > instead of >="},
    "bug_r2":  {"difficulty": "easy",   "description": "filter_by_source: exact match uses 'in'"},
    "bug_r3":  {"difficulty": "medium", "description": "filter_by_time: inverted end boundary"},
    "bug_r4":  {"difficulty": "medium", "description": "error_windows: off-by-one in context"},
    "bug_r5":  {"difficulty": "medium", "description": "error_rate: missing CRITICAL"},
    "bug_r6":  {"difficulty": "hard",   "description": "latest_by_source: inverted comparison"},
    "bug_r7":  {"difficulty": "hard",   "description": "top_sources: ascending instead of descending"},
    "bug_r8":  {"difficulty": "hard",   "description": "LEVELS: missing CRITICAL entry"},
    "bug_r9":  {"difficulty": "easy",   "description": "filter_by_time: inverted start boundary"},
    "bug_r10": {"difficulty": "easy",   "description": "count_by_level: off-by-one default"},
    "bug_r11": {"difficulty": "easy",   "description": "parse_log: split on whitespace not newlines"},
    "bug_r12": {"difficulty": "easy",   "description": "LEVELS: WARNING same value as ERROR"},
    "bug_r13": {"difficulty": "medium", "description": "parse_line: level/src groups swapped"},
    "bug_r14": {"difficulty": "medium", "description": "error_windows: n_after used for start index"},
    "bug_r15": {"difficulty": "medium", "description": "error_rate: off-by-one denominator"},
    "bug_r16": {"difficulty": "medium", "description": "count_by_source: counts by level instead"},
    "bug_r17": {"difficulty": "hard",   "description": "latest_by_source: keyed by level not src"},
    "bug_r18": {"difficulty": "hard",   "description": "error_windows: ignores CRITICAL errors"},
    "bug_r19": {"difficulty": "hard",   "description": "error_rate: inverted fraction"},
    "bug_r20": {"difficulty": "hard",   "description": "filter_by_level: .lower() breaks lookup"},
}
