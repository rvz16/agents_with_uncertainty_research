"""Judge view of a DeepSWE run: task statement plus (command, truncated output) per step.

Steps are indexed exactly as in compact_jobs.py (one per assistant message, the
following tool / user message is its result), so judge verdicts line up with the
compact records. Reads the jobs.zip archive or an extracted jobs directory.

    python -m experiments.deepswe_judge_view jobs.zip --run deepswe_gptoss_113_v5 --out runs/deepswe_gptoss_113_v5/judge_view.jsonl
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

OBS_CHARS = 600
TASK_CHARS = 3000


def view(tid: str, tr: dict) -> dict:
    msgs = tr.get("messages", [])
    task = next((str(m.get("content") or "") for m in msgs if m.get("role") == "user"), "")
    steps: list[dict] = []
    pending = None
    for m in msgs:
        role = m.get("role")
        if role == "assistant":
            resp = ((m.get("extra") or {}).get("response") or {})
            choice = (resp.get("choices") or [{}])[0]
            cmd = ""
            for tc in (choice.get("message") or {}).get("tool_calls") or []:
                try:
                    c = json.loads(tc["function"]["arguments"]).get("command", "")
                    cmd = c if isinstance(c, str) else " ".join(map(str, c))
                except Exception:  # noqa: BLE001
                    cmd = ""
            if not cmd:
                cmd = str(m.get("content") or "")
            pending = {"action": cmd, "observation": ""}
            steps.append(pending)
        elif role in ("tool", "user") and pending is not None:
            text = str(m.get("content") or "")
            if role == "tool":
                try:
                    d = json.loads(text)
                    text = f"[rc={d.get('returncode')}] {d.get('output', '')}"
                except Exception:  # noqa: BLE001
                    pass
            pending["observation"] = text[:OBS_CHARS]
            pending = None
    return {"id": tid, "task": task[:TASK_CHARS], "steps": steps}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("zip", type=Path); p.add_argument("--run", required=True); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    prefix = a.run.rstrip("/") + "/"
    if a.zip.is_dir():
        root = a.zip / a.run
        tasks = sorted(d.name for d in root.iterdir() if d.is_dir())
        read = lambda member: (a.zip / member).read_bytes()  # noqa: E731
    else:
        z = zipfile.ZipFile(a.zip)
        tasks = sorted({n[len(prefix):].split("/")[0] for n in z.namelist() if n.startswith(prefix) and "/" in n[len(prefix):]})
        read = z.read
    n = 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as out:
        for t in tasks:
            try:
                tr = json.loads(read(f"{prefix}{t}/agent/mini-swe-agent.trajectory.json"))
            except (KeyError, FileNotFoundError):
                continue
            out.write(json.dumps(view(t, tr)) + "\n"); n += 1
    print(f"wrote {n} episodes to {a.out}")


if __name__ == "__main__":
    main()
