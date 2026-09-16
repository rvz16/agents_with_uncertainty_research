"""Prefix LLM judge, one verdict per step, shared by ALFWorld and DeepSWE.

At step k the judge sees the task and the last ``--window`` steps up to and
including k (action plus a truncated result), nothing after it, and no step
counter, and answers with an integer 0-100: the probability that the agent
will complete the task successfully. The verdict is therefore formed before
the outcome exists, which is what makes it a usable UQ signal (a judge that
reads a full ALFWorld transcript mostly reads its length).

Input is a "view": one JSON object per episode, ``{"id", "task", "steps":
[{"action", "observation"}]}``. ``--alfworld RUN_DIR`` builds it from a run's
trajectories.jsonl (the rows of the finished / pre-terminal cohort as they
are); DeepSWE views come from experiments.deepswe_judge_view.

    python -m experiments.prefix_judge --alfworld runs/react_gptoss_140_giveup_sc_finished_nolast \\
        --out runs/react_gptoss_140_giveup_sc_finished_nolast/judge.jsonl
    python -m experiments.prefix_judge --view deep_swe_uq/runs/deepswe_gptoss_113_v5/judge_view.jsonl \\
        --out deep_swe_uq/runs/deepswe_gptoss_113_v5/judge.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SYSTEM = (
    "You are evaluating a partial trajectory of an autonomous agent. You see the task and the most recent "
    "steps; the episode is still in progress and you do not know how it ends. Estimate the probability that "
    "the agent will eventually complete the task successfully. Reply with a single integer from 0 to 100 and "
    "nothing else."
)
NUM = re.compile(r"(\d{1,3})")


def alfworld_view(run: Path) -> list[dict]:
    eps: dict[str, dict] = {}
    for line in open(run / "trajectories.jsonl"):
        if not line.strip():
            continue
        r = json.loads(line)
        e = eps.setdefault(r["episode_id"], {"id": r["episode_id"], "task": r["task"], "steps": []})
        e["steps"].append({"action": str(r.get("action") or ""), "observation": str(r.get("observation") or "")})
    return list(eps.values())


def prompt(task: str, steps: list[dict], window: int, obs_chars: int, act_chars: int) -> str:
    recent = steps[-window:]
    lines = [f"Task: {task.strip()[:3000]}", "", "Most recent steps, oldest first:"]
    for s in recent:
        lines.append(f"[action] {s['action'].strip()[:act_chars]}")
        lines.append(f"[result] {s['observation'].strip()[:obs_chars]}")
    return "\n".join(lines)


def ask(client, model: str, text: str, extra: dict, max_tokens: int) -> tuple[int | None, str]:
    err = ""
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
                max_tokens=max_tokens, temperature=0.0, **extra)
            msg = r.choices[0].message
            out = (msg.content or "").strip()  # the answer channel only: digits inside the reasoning are not a verdict
            m = NUM.findall(out)
            if not m:  # reasoning ate the budget, or prose: one more try is cheap
                err = "no integer: " + (out or (getattr(msg, "reasoning", None) or ""))[-120:]
                continue
            p = int(m[-1])
            return min(max(p, 0), 100), out[-200:]
        except Exception as exc:  # noqa: BLE001
            err = str(exc)[-200:]
            time.sleep(2 * (attempt + 1))
    return None, err


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alfworld", type=Path, default=None, help="run directory with trajectories.jsonl")
    p.add_argument("--view", type=Path, default=None, help="judge view jsonl (DeepSWE)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--obs-chars", type=int, default=400)
    p.add_argument("--act-chars", type=int, default=300)
    p.add_argument("--reasoning-effort", default="low")
    p.add_argument("--max-tokens", type=int, default=256, help="the reasoning channel is billed against it")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--limit", type=int, default=0, help="episodes (debug)")
    a = p.parse_args()
    from openai import OpenAI
    client = OpenAI(base_url=a.base_url, api_key=os.environ[a.api_key_env], timeout=a.timeout, max_retries=0)
    extra: dict = {"extra_body": {}}
    if a.reasoning_effort:
        extra["extra_body"]["reasoning"] = {"effort": a.reasoning_effort}
    if "openrouter" in a.base_url:  # a slow provider stalls a whole episode: prefer the fast ones
        extra["extra_body"]["provider"] = {"sort": "throughput"}
    if a.alfworld:
        episodes = alfworld_view(a.alfworld)
    else:
        episodes = [json.loads(l) for l in open(a.view) if l.strip()]
    if a.limit:
        episodes = episodes[: a.limit]
    done = set()
    if a.out.exists():
        rows = [json.loads(l) for l in open(a.out) if l.strip()]
        rows = [r for r in rows if r.get("p") is not None]
        with open(a.out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        done = {(r["id"], r["step"]) for r in rows}
        print(f"[judge] resuming: {len(done)} verdicts kept", flush=True)
    out = open(a.out, "a"); total = asked = failed = 0
    with ThreadPoolExecutor(a.workers) as pool:
        for e in episodes:
            jobs = [k for k in range(len(e["steps"])) if (e["id"], k) not in done]
            total += len(e["steps"])
            texts = [prompt(e["task"], e["steps"][: k + 1], a.window, a.obs_chars, a.act_chars) for k in jobs]
            for k, (pv, raw) in zip(jobs, pool.map(lambda t: ask(client, a.model, t, extra, a.max_tokens), texts)):
                out.write(json.dumps({"id": e["id"], "step": k, "p": pv, "raw": raw}) + "\n"); asked += 1
                failed += pv is None
            out.flush()
            print(f"[judge] {e['id']}: {len(e['steps'])} steps ({len(jobs)} asked)", flush=True)
    print(f"[judge] done: {asked} queries ({failed} without a number), {total} steps -> {a.out}")


if __name__ == "__main__":
    main()
