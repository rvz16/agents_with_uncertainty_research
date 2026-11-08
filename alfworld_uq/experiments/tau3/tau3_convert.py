"""tau3-bench results.json -> our judge_view + distill_targets schema.

A step is an assistant turn that carries a tool call, paired with the ``tool``
message that answers it: that pair is (action, observation), the same unit the
other three environments give us. User turns are folded into the observation of
the step they precede, because in tau3 the task is revealed progressively by the
user rather than stated at step 0 -- the one structural difference from DeepSWE,
ALFWorld and OSWorld, and it is recorded here rather than hidden.

Labels are the simulation's own reward (continuous: tau3 grades on several bases
and ``reward_breakdown`` can be partial), so this cohort joins CONTINUOUS.

Critics, by analogy with the DeepSWE set (ran tests / last run passed / committed):
  called_write      a state-changing tool was called in the prefix
  tool_ok           the last tool result parsed and carried no error
  no_error_yet      no tool result in the prefix carried an error
  read_before_write at least one read preceded the first write
  asked_user        the agent asked the user something after its first tool call

Structural shares, the same two the head is supervised on:
  repeat_share  tool calls duplicating an earlier (name, arguments)
  stale_share   reads whose result predates a later write on the same entity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

WRITE_HINTS = ("cancel", "modify", "update", "create", "book", "exchange", "return",
               "transfer", "send", "set_", "add_", "remove_", "delete")
ERROR_HINTS = ("error", "not found", "invalid", "cannot", "failed", "denied")


def is_write(name: str) -> bool:
    return any(h in name.lower() for h in WRITE_HINTS)


def looks_error(text: str) -> bool:
    head = (text or "")[:200].lower()
    return any(h in head for h in ERROR_HINTS)


def steps_of(messages: list[dict]) -> list[dict]:
    """Assistant-with-tool-call paired to the tool reply; user turns ride along."""
    out: list[dict] = []
    pending_user: list[str] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        role = m.get("role")
        if role == "user":
            pending_user.append(str(m.get("content") or ""))
            i += 1
            continue
        calls = m.get("tool_calls") or []
        if role == "assistant" and calls:
            call = calls[0]
            observation = ""
            if i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
                observation = str(messages[i + 1].get("content") or "")
                i += 1
            action = f"{call.get('name')}({json.dumps(call.get('arguments'), sort_keys=True)})"
            said = "\n".join(f"[user] {u}" for u in pending_user)
            out.append({"action": action, "observation": (said + "\n" + observation).strip(),
                        "name": str(call.get("name") or ""),
                        "arguments": json.dumps(call.get("arguments"), sort_keys=True)})
            pending_user = []
        elif role == "assistant" and (m.get("content") or "").strip():
            # a turn that only talks: an action with no tool, kept so the prefix
            # length matches the dialogue the agent actually produced
            said = "\n".join(f"[user] {u}" for u in pending_user)
            out.append({"action": "say(" + str(m.get("content"))[:400] + ")",
                        "observation": said.strip(), "name": "", "arguments": ""})
            pending_user = []
        i += 1
    return out


def shares(steps: list[dict], cut: int) -> tuple[float, float]:
    seen: set[tuple[str, str]] = set()
    repeats = 0
    first_write = None
    reads_before: list[int] = []
    for k, s in enumerate(steps[:cut]):
        if not s["name"]:
            continue
        key = (s["name"], s["arguments"])
        if key in seen:
            repeats += 1
        seen.add(key)
        if is_write(s["name"]):
            if first_write is None:
                first_write = k
        else:
            reads_before.append(k)
    n = max(1, cut)
    stale = 0 if first_write is None else sum(1 for k in reads_before if k < first_write)
    return repeats / n, stale / n


def critics(steps: list[dict], cut: int) -> dict:
    window = steps[:cut]
    tools = [s for s in window if s["name"]]
    errors = [looks_error(s["observation"]) for s in tools]
    writes = [is_write(s["name"]) for s in tools]
    first_write = next((i for i, w in enumerate(writes) if w), None)
    return {
        "called_write": bool(first_write is not None),
        "tool_ok": bool(tools and not errors[-1]),
        "no_error_yet": bool(not any(errors)),
        "read_before_write": bool(first_write is not None and first_write > 0),
        "asked_user": any("[user]" in s["observation"] for s in window[1:]),
    }


def episodes_of(blob) -> list[dict]:
    if isinstance(blob, dict):
        for key in ("simulations", "results", "episodes"):
            if isinstance(blob.get(key), list):
                return blob[key]
        return [blob]
    return list(blob)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--cohort", required=True, help="e.g. TA for tau3-airline")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min-steps", type=int, default=4)
    a = p.parse_args()

    blob = json.loads(a.results.read_text())
    eps = episodes_of(blob)
    views, targets = [], []
    kept = skipped = 0
    for e in eps:
        msgs = e.get("messages") or []
        ri = e.get("reward_info") or {}
        reward = ri.get("reward")
        st = steps_of(msgs)
        if reward is None or len(st) < a.min_steps:
            skipped += 1
            continue
        kept += 1
        eid = f"{e.get('task_id')}::{e.get('id')}"
        task = str(e.get("policy") or "")[:4000]
        views.append({"id": eid, "task": task,
                      "steps": [{"action": s["action"], "observation": s["observation"]} for s in st]})
        for k in range(0, len(st), a.stride):
            rep, sta = shares(st, k + 1)
            targets.append({"id": eid, "cohort": a.cohort, "step": k,
                            "posterior": float(reward), "label": int(float(reward) > 0.5),
                            "score": float(reward), "repeat_share": rep, "stale_share": sta,
                            "critics": critics(st, k + 1)})
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "judge_views").mkdir(exist_ok=True)
    (a.out / "distill_targets").mkdir(exist_ok=True)
    with open(a.out / "judge_views" / f"{a.cohort}.jsonl", "w") as h:
        for v in views:
            h.write(json.dumps(v) + "\n")
    with open(a.out / "distill_targets" / f"targets_{a.cohort}.jsonl", "w") as h:
        for t in targets:
            h.write(json.dumps(t) + "\n")
    print(f"[tau3] {a.cohort}: {kept} episodes kept, {skipped} skipped "
          f"(no reward or < {a.min_steps} steps), {len(targets)} prefixes")


if __name__ == "__main__":
    main()
