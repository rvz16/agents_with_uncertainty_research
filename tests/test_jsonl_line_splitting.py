"""JSONL must not be split with `str.splitlines()`.

`splitlines()` breaks on more than newline: also on vertical tab, form feed,
file separator, U+2028 and U+2029. Records hold raw model text, and U+2028 does
occur there -- one run's logprob sidecar contained 14 of them, turning 150
records into 164 fragments and losing 16 percent of the data.

On the `--resume` path the cost is worse than lost data: an unreadable record
reads as "instance not finished", so the episode is silently run again.
"""
from __future__ import annotations

import json

from code_uq.environments.fitted_live.common import load_jsonl_keys


def _write_record_with_line_separator(path, instance_id: str) -> None:
    """A record whose text contains a raw U+2028, as model output does."""
    payload = {
        "instance_id": instance_id,
        "policy": "sage",
        "raw_text": f"before after",
    }
    # ensure_ascii=False keeps the separator raw, as in real sidecars
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def test_line_separator_does_not_hide_a_finished_instance(tmp_path):
    path = tmp_path / "episodes.jsonl"
    _write_record_with_line_separator(path, "task-1")

    assert load_jsonl_keys(path) == {("task-1", "sage")}


def test_splitlines_would_have_lost_it(tmp_path):
    """Pin the mechanism itself so the fix cannot be reverted unnoticed."""
    path = tmp_path / "episodes.jsonl"
    _write_record_with_line_separator(path, "task-1")
    text = path.read_text(encoding="utf-8")

    assert len(text.split("\n")) - 1 == 1          # one real line
    assert len([x for x in text.splitlines() if x.strip()]) == 2   # splitlines tears it in two


def test_several_records_survive(tmp_path):
    path = tmp_path / "episodes.jsonl"
    rows = [
        {"instance_id": f"task-{i}", "policy": "sage", "raw_text": f"a b{i}"}
        for i in range(5)
    ]
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    assert load_jsonl_keys(path) == {(f"task-{i}", "sage") for i in range(5)}


def test_blank_and_broken_lines_are_skipped(tmp_path):
    path = tmp_path / "episodes.jsonl"
    path.write_text(
        json.dumps({"instance_id": "ok", "policy": "sage"}) + "\n"
        + "\n"
        + "{ not json\n",
        encoding="utf-8",
    )
    assert load_jsonl_keys(path) == {("ok", "sage")}
