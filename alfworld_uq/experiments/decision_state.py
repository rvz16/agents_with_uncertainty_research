"""Serialise a recorded trajectory into the `state` a decision model reads.

A System One decision model (Jeeves, Clef, Jev, Laya) takes a state and a
schema of typed questions and returns a calibrated probability per option. The
state has to be the trajectory as text, not the per-step float signals of the
toolkit export: those models read a situation.

Nothing that reveals the outcome may enter the state. We therefore drop the
episode's stop reason, its reward and its terminal generation (the cohorts are
already scored pre-terminally), and keep only what the agent itself could see
while acting: its actions, the environment's replies, and the mechanical
critics of each step.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

OBS_CHARS = 180
CMD_CHARS = 160
HEAD_STEPS = 12
TAIL_STEPS = 18


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _window(rows: list, head: int = HEAD_STEPS, tail: int = TAIL_STEPS) -> Iterable[tuple[int, Any]]:
    """First ``head`` and last ``tail`` steps, with a marker for what was cut."""
    if len(rows) <= head + tail:
        yield from enumerate(rows)
        return
    yield from enumerate(rows[:head])
    yield -1, None
    yield from ((i + len(rows) - tail, r) for i, r in enumerate(rows[-tail:]))


def apply_step_rule(rows: list[dict], rule: str) -> list[dict]:
    """The report's step-selection rules, so the state covers what the tables score."""
    if rule == "all":
        return rows
    from experiments.prr_report_v2 import _is_refusal

    if rule in ("giveup", "giveup+acted"):
        cut = next((i for i, r in enumerate(rows) if _is_refusal(r)), len(rows))
        rows = rows[:cut]
    if rule in ("acted", "giveup+acted"):
        rows = [r for r in rows if (r.get("action") or "").strip()]
    return rows


def alfworld_state(rows: list[dict], task: str, harness: str) -> str:
    """ALFWorld: the task, then one line per step with the action and the reply."""
    lines = [f"Environment: ALFWorld, household task, {harness} agent.", f"Task: {task}", ""]
    n_sub = n_noop = 0
    for index, row in _window(rows):
        if row is None:
            lines.append(f"  ... {len(rows) - HEAD_STEPS - TAIL_STEPS} further steps omitted ...")
            continue
        proposed = (row.get("proposed_action") or "").strip()
        action = (row.get("action") or "").strip()
        note = ""
        if not action:
            n_noop += 1
            note = "  [no tool call: the model produced no executable action]"
            shown = _clip(row.get("raw_response") or "", 80) or "(empty response)"
        elif row.get("fallback_reason") == "inadmissible_action":
            n_sub += 1
            shown = proposed or action
            note = f"  [not available here; the harness ran '{action}' instead]"
        else:
            shown = action
        lines.append(f"step {index + 1}: {shown}{note}")
        lines.append(f"   result: {_clip(row.get('observation') or '', OBS_CHARS)}")
    summary = f"The agent has taken {len(rows)} steps so far, {n_sub} of which proposed an action " \
              f"the environment did not allow."
    if n_noop:
        summary += f" In {n_noop} further steps the model produced no executable action."
    lines += ["", summary]
    return "\n".join(lines)


def deepswe_state(record: dict) -> str:
    """DeepSWE: the shell commands the agent ran and their return codes."""
    rows = record.get("steps") or []
    lines = ["Environment: DeepSWE, a software engineering task in a repository, mini-swe-agent.",
             f"Task instance: {record.get('id')}", "",
             "The agent works by running shell commands. One line per command, with its exit status."]
    bad = sum(1 for r in rows if r.get("returncode"))
    fmt = sum(1 for r in rows if r.get("format_error"))
    tests = sum(1 for r in rows if _is_test(r.get("command") or ""))
    for index, row in _window(rows):
        if row is None:
            lines.append(f"  ... {len(rows) - HEAD_STEPS - TAIL_STEPS} further commands omitted ...")
            continue
        if row.get("format_error"):
            lines.append(f"cmd {index + 1}: (no command: the response did not call the tool)")
            continue
        rc = row.get("returncode")
        lines.append(f"cmd {index + 1}: {_clip(row.get('command') or '', CMD_CHARS)}   -> exit {rc}")
    lines += ["", f"The agent has run {len(rows)} commands so far: {bad} returned a non-zero exit status, "
              f"{tests} invoked the test suite, and {fmt} responses called no tool at all."]
    return "\n".join(lines)


def _is_test(command: str) -> bool:
    low = command.lower()
    return any(token in low for token in ("pytest", "unittest", "tox", "nosetests", " test"))


#: the schema we ask about every trajectory; the first question is the one we score
QUESTIONS = {
    "success": {
        "type": "noul",
        "instructions": "Will this agent ultimately complete the task correctly? "
                        "Judge only from the steps shown; the trajectory is cut before its final action.",
    },
    "stuck": {
        "type": "noul",
        "instructions": "Is the agent repeating actions or retrying the same approach without making progress?",
    },
    "stale_evidence": {
        "type": "noul",
        "instructions": "Has the agent changed the state since its most recent check, so that the result "
                        "of that check no longer describes the current state?",
    },
    "progress": {
        "type": "score",
        "instructions": "How much of the task has the agent completed so far?",
        "criteria": ["none", "a little", "about half", "most", "all but the final action"],
    },
}


def main() -> None:
    import argparse
    import statistics as st

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alfworld", nargs=2, action="append", metavar=("KEY", "RUN_DIR"), default=[])
    p.add_argument("--deepswe", nargs=2, action="append", metavar=("KEY", "COMPACT"), default=[])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--step-rule", default="all", choices=["all", "acted", "giveup", "giveup+acted"],
                   help="which recorded generations enter the state; see prr_report_v2.load_alfworld. "
                        "giveup+acted is the rule under which a step means the same on both harnesses")
    p.add_argument("--drop-last", type=int, default=0,
                   help="drop this many further steps from the end of every episode; an audit of "
                        "whether a high score lives in the last steps or across the trajectory")
    p.add_argument("--show", action="store_true", help="print the first state of each cohort")
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    for key, run in a.alfworld:
        run = Path(run)
        steps: dict[str, list] = {}
        for line in open(run / "trajectories.jsonl"):
            if line.strip():
                row = json.loads(line); steps.setdefault(row["episode_id"], []).append(row)
        out = []
        for line in open(run / "episodes.jsonl"):
            if not line.strip():
                continue
            episode = json.loads(line); rows = steps.get(episode["episode_id"]) or []
            if not rows:
                continue
            rows.sort(key=lambda r: int(r.get("step", 0)))
            rows = apply_step_rule(rows, a.step_rule)
            if a.drop_last:
                rows = rows[: -a.drop_last]
            if not rows:
                continue
            harness = "ReAct" if "react" in run.name else "smolagents CodeAgent"
            out.append({"id": episode["episode_id"], "cohort": key,
                        "label": int(bool(episode["final_success"])),
                        "score": float(bool(episode["final_success"])),
                        "state": alfworld_state(rows, episode.get("task") or "", harness)})
        _write(a.out / f"{key}.jsonl", out, key, a.show)
    for key, compact in a.deepswe:
        out = []
        for line in open(compact):
            if not line.strip():
                continue
            record = json.loads(line)
            rewards = record.get("rewards") or {}
            if "partial" not in rewards:  # a run the verifier never scored
                continue
            out.append({"id": record["id"], "cohort": key,
                        "label": None, "score": float(rewards["partial"]),
                        "state": deepswe_state(record)})
        _write(a.out / f"{key}.jsonl", out, key, a.show)


def _write(path: Path, rows: list[dict], key: str, show: bool) -> None:
    import statistics as st
    with open(path, "w") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in rows)
    sizes = [len(r["state"]) for r in rows]
    print(f"{key}: {len(rows)} states -> {path}  chars median {st.median(sizes):.0f} "
          f"max {max(sizes)} (~{max(sizes) // 4} tokens)")
    if show and rows:
        print("-" * 70); print(rows[0]["state"][:1200]); print("-" * 70)


if __name__ == "__main__":
    main()
