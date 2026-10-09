"""Build decision-model states for DeepSWE from the run archive.

The compact records carry the commands and their return codes but not what the
commands printed, and a decision model asked to judge a repository task without
seeing any output is being asked to guess. This reads the archived
mini-swe-agent trajectories instead and keeps, per step, the command and the
head and tail of its output.

Nothing that reveals the verdict may enter the state: the reward, the exit
status and the final assistant turn stay out, and the trajectory is cut before
its last command, as the cohorts are scored.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

OUT_HEAD = 400
OUT_TAIL = 200
HEAD_STEPS = 10
TAIL_STEPS = 16
CMD_CHARS = 200


def _clip_output(text: str) -> str:
    text = text.strip()
    if len(text) <= OUT_HEAD + OUT_TAIL:
        return " ".join(text.split())
    head = " ".join(text[:OUT_HEAD].split())
    tail = " ".join(text[-OUT_TAIL:].split())
    return f"{head} … [{len(text)} chars] … {tail}"


def steps_of(trajectory: dict) -> list[dict]:
    """(command, output, returncode) per step, from the recorded messages."""
    steps: list[dict] = []
    pending: dict | None = None
    for message in trajectory.get("messages", []):
        role = message.get("role")
        if role == "assistant":
            command = None
            for call in message.get("tool_calls") or []:
                try:
                    command = json.loads(call["function"]["arguments"]).get("command")
                except Exception:  # noqa: BLE001 - a malformed call is a format error, not a command
                    command = None
            pending = {"command": command, "output": "", "returncode": None,
                       "format_error": command is None}
            steps.append(pending)
        elif role in ("tool", "user") and pending is not None:
            text = str(message.get("content") or "")
            try:
                payload = json.loads(text)
                pending["returncode"] = int(payload.get("returncode"))
                pending["output"] = str(payload.get("output") or payload.get("stdout") or "")
            except Exception:  # noqa: BLE001 - not every reply is the tool's JSON
                pending["output"] = text
            if "Tool call error" in text or "format error" in text.lower():
                pending["format_error"] = True
            pending = None
    return steps


def state_of(task_id: str, steps: list[dict]) -> str:
    lines = ["Environment: DeepSWE, a software engineering task in a repository, mini-swe-agent.",
             f"Task instance: {task_id}", "",
             "The agent works by running shell commands. Each entry is one command and what it printed."]
    windowed: list[tuple[int, dict | None]] = []
    if len(steps) <= HEAD_STEPS + TAIL_STEPS:
        windowed = list(enumerate(steps))
    else:
        windowed = list(enumerate(steps[:HEAD_STEPS]))
        windowed.append((-1, None))
        windowed += [(i + len(steps) - TAIL_STEPS, s) for i, s in enumerate(steps[-TAIL_STEPS:])]
    for index, step in windowed:
        if step is None:
            lines.append(f"  ... {len(steps) - HEAD_STEPS - TAIL_STEPS} further commands omitted ...")
            continue
        if step["format_error"] or not step["command"]:
            lines.append(f"cmd {index + 1}: (the response called no tool)")
            continue
        command = " ".join(str(step["command"]).split())[:CMD_CHARS]
        lines.append(f"cmd {index + 1}: {command}   -> exit {step['returncode']}")
        output = _clip_output(step["output"])
        if output:
            lines.append(f"   output: {output}")
    bad = sum(1 for s in steps if s.get("returncode"))
    fmt = sum(1 for s in steps if s.get("format_error"))
    lines += ["", f"The agent has run {len(steps)} commands so far: {bad} returned a non-zero exit "
              f"status and {fmt} responses called no tool at all."]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path, help="jobs zip, or an extracted jobs directory")
    p.add_argument("--run", required=True, help="top-level directory inside the archive")
    p.add_argument("--cohort", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--drop-last", type=int, default=1,
                   help="steps cut from the end; one by default, as the cohorts are scored "
                        "before the outcome-revealing terminal command")
    a = p.parse_args()

    prefix = a.run.rstrip("/") + "/"
    if a.archive.is_dir():
        root = a.archive / a.run
        tasks = sorted(d.name for d in root.iterdir() if d.is_dir())
        read = lambda member: (a.archive / member).read_bytes()  # noqa: E731
    else:
        archive = zipfile.ZipFile(a.archive)
        tasks = sorted({n[len(prefix):].split("/")[0] for n in archive.namelist()
                        if n.startswith(prefix) and "/" in n[len(prefix):]})
        read = archive.read

    a.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(a.out, "w") as out:
        for task in tasks:
            base = f"{prefix}{task}/"
            try:
                result = json.loads(read(base + "result.json"))
                trajectory = json.loads(read(base + "agent/mini-swe-agent.trajectory.json"))
            except KeyError:
                continue
            rewards = (result.get("verifier_result") or {}).get("rewards") or {}
            if "partial" not in rewards:
                continue
            steps = steps_of(trajectory)
            if a.drop_last:
                steps = steps[: -a.drop_last]
            if not steps:
                continue
            out.write(json.dumps({"id": task, "cohort": a.cohort, "label": None,
                                  "score": float(rewards["partial"]),
                                  "state": state_of(task, steps)}) + "\n")
            written += 1
    print(f"[deepswe-state] {written} states -> {a.out}")


if __name__ == "__main__":
    main()
