"""Approximate per-step decision samples for UProp on the ALFWorld smolagents cohorts.

The ReAct prompt is a string we can rebuild exactly (agents/react_agent.py), but a
CodeAgent's prompt is assembled by smolagents from its system template, tool
descriptions and memory. This script rebuilds it the way the framework does: it
instantiates a CodeAgent with the same tool signature and code-block tags as the run,
replays the recorded steps into its memory as ``TaskStep`` / ``ActionStep`` entries, and
takes ``write_memory_to_messages()`` as the prefix for step k. N decisions are then
sampled from the same model at temperature 0.8 with log-probabilities, exactly as in
experiments/uprop_samples.py.

The reconstruction is approximate in one respect: ALFWorld's initial room description is
not stored in the trajectory rows, so the task message carries the recorded task and the
first admissible-action list with the initial observation marked as unavailable. Every
later message (the model's own code blocks and the environment's returns) is verbatim.
Rows are written in the format experiments/uprop_samples.py produces.

    python -m experiments.uprop_samples_smol --alfworld runs/smol_gptoss_140_sc \\
        --model openai/gpt-oss-20b --out runs/smol_gptoss_140_sc/uprop_samples.jsonl
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

_ACTION = re.compile(r"take_action\(\s*['\"](.+?)['\"]\s*\)", re.S)


def build_agent(code_block_tags: str, max_steps: int):
    """A CodeAgent whose system prompt matches the run's, with a stub tool."""
    from smolagents import CodeAgent, Tool
    from smolagents.models import Model

    class TakeAction(Tool):
        name = "take_action"
        description = (
            "Execute exactly one admissible action in the household environment and "
            "return the resulting observation together with the new admissible "
            "action list. The admissible list changes after every action."
        )
        inputs = {
            "action": {
                "type": "string",
                "description": "One action string copied verbatim from the admissible list.",
            }
        }
        output_type = "string"

        def forward(self, action: str) -> str:  # never called: memory is replayed, not run
            return ""

    class _Silent(Model):
        def generate(self, messages, **kwargs):  # pragma: no cover - never called
            raise RuntimeError("the reconstruction never generates")

    return CodeAgent(tools=[TakeAction()], model=_Silent(), max_steps=max_steps,
                     code_block_tags=code_block_tags)


def task_message(task: str, admissible: list[str]) -> str:
    actions = "\n".join(f"- {a}" for a in admissible)
    return (
        "You are a household agent acting in a text-only ALFWorld environment.\n\n"
        f"Task: {task}\n\n"
        "Initial observation:\n(not recorded in the trajectory)\n\n"
        f"Admissible actions right now:\n{actions}\n\n"
        "Rules:\n"
        '- Call take_action("<action>") with exactly ONE admissible action, '
        "copied verbatim from the admissible list.\n"
        "- Make exactly ONE take_action call per code block, then stop and read "
        "the result. The room is only partially observable: the next action "
        "depends on what the previous one revealed, so a script of several "
        "actions written in advance is guesswork.\n"
        "- take_action returns the new observation and the new admissible list; "
        "the list changes after every action, so never reuse a stale one.\n"
    )


def prefixes(rows: list[dict], code_block_tags: str, max_steps: int):
    """(step index, realised action, messages) for every recorded step."""
    from smolagents.memory import ActionStep, TaskStep
    from smolagents.models import get_clean_message_list, tool_role_conversions
    from smolagents.monitoring import Timing

    agent = build_agent(code_block_tags, max_steps)
    agent.memory.steps = [TaskStep(task=task_message(rows[0]["task"], rows[0].get("admissible_actions") or []))]
    out = []
    for k, r in enumerate(rows):
        # the framework's own conversion: tool responses become user turns and
        # consecutive turns of one role are merged, as when the run was collected
        messages = [{"role": str(getattr(m["role"], "value", m["role"])), "content": m["content"]}
                    for m in get_clean_message_list(agent.write_memory_to_messages(),
                                                    role_conversions=tool_role_conversions,
                                                    flatten_messages_as_text=True)]
        out.append((k, str(r.get("action") or ""), messages))
        observation = (
            f"{r.get('observation') or ''}\n\nAdmissible actions right now:\n"
            + "\n".join(f"- {a}" for a in (r.get("admissible_actions") or []))
        )
        agent.memory.steps.append(ActionStep(
            step_number=k + 1, timing=Timing(start_time=0.0, end_time=0.0),
            model_output=str(r.get("raw_response") or ""), observations=observation))
    return out


def sample(client, model: str, messages: list[dict], extra: dict, max_tokens: int) -> dict:
    err = ""
    for attempt in range(3):
        try:
            r = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens,
                                               temperature=0.8, logprobs=True, **extra)
            choice = r.choices[0]
            text = (choice.message.content or "").strip()
            toks = (choice.logprobs.content if choice.logprobs else None) or []
            lps = [t.logprob for t in toks if t.logprob is not None]
            m = _ACTION.search(text)
            return {"text": text, "action": m.group(1).strip() if m else text,
                    "nll": -sum(lps) / len(lps) if lps else None}
        except Exception as exc:  # noqa: BLE001
            err = str(exc)[-160:]
            time.sleep(2 * (attempt + 1))
    return {"text": "", "action": "", "nll": None, "error": err}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alfworld", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", required=True); p.add_argument("--n", type=int, default=10)
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--workers", type=int, default=16); p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--timeout", type=float, default=60.0); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--code-block-tags", default="markdown"); p.add_argument("--agent-max-steps", type=int, default=75)
    a = p.parse_args()
    from openai import OpenAI
    client = OpenAI(base_url=a.base_url, api_key=os.environ[a.api_key_env], timeout=a.timeout, max_retries=0)
    extra: dict = {"extra_body": {"provider": {"require_parameters": True, "sort": "throughput"},
                                 "reasoning": {"effort": "low"} if "gpt-oss" in a.model else {"enabled": False}}}
    rows = defaultdict(list)
    for line in open(a.alfworld / "trajectories.jsonl"):
        if line.strip():
            r = json.loads(line); rows[r["episode_id"]].append(r)
    episodes = list(rows.items())[: a.limit or None]
    done = set()
    if a.out.exists():
        kept = [json.loads(l) for l in open(a.out) if l.strip()]
        kept = [r for r in kept if sum(1 for s in r["samples"] if s.get("text")) >= a.n // 2]
        with open(a.out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in kept)
        done = {(r["id"], r["step"]) for r in kept}
        print(f"[uprop-smol] resuming: {len(done)} steps kept", flush=True)
    out = open(a.out, "a"); asked = 0
    with ThreadPoolExecutor(a.workers) as pool:
        for eid, steps in episodes:
            steps.sort(key=lambda r: int(r.get("step", 0)))
            jobs = [(k, realised, msgs) for k, realised, msgs in prefixes(steps, a.code_block_tags, a.agent_max_steps)
                    if (eid, k) not in done]
            for (k, realised, msgs), samples in zip(jobs, pool.map(
                    lambda j: [sample(client, a.model, j[2], extra, a.max_tokens) for _ in range(a.n)], jobs)):
                out.write(json.dumps({"id": eid, "step": k, "realised": realised, "samples": samples}) + "\n"); asked += 1
            out.flush()
            print(f"[uprop-smol] {eid}: {len(steps)} steps ({len(jobs)} sampled x {a.n})", flush=True)
    print(f"[uprop-smol] done: {asked} steps -> {a.out}")


if __name__ == "__main__":
    main()
