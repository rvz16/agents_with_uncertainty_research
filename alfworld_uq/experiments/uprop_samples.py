"""Per-step decision samples for UProp on the ALFWorld ReAct cohorts.

For every recorded step the ReAct prompt is rebuilt exactly as the agent built
it (system prompt with the verbalised-confidence and give-up clauses, task,
history of thought / action / observation, the step's admissible actions) and
N decisions are sampled at temperature 0.8 with token log-probabilities. The
recorded trajectory is never changed: the samples only measure how typical the
realised decision was among its alternatives (UProp's extrinsic term) and how
peaked the step's decision distribution is (its intrinsic term).

Output rows: {"id", "step", "realised", "samples": [{"text", "action", "nll"}]}
where ``nll`` is the per-token negative log-probability of the whole sample.
smolagents cohorts are not covered: their prompts are assembled by the
framework from tool descriptions and memory and cannot be rebuilt from the rows.

    python -m experiments.uprop_samples --alfworld runs/<react run> --model openai/gpt-oss-20b \\
        --out runs/<react run>/uprop_samples.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agents.react_agent import GIVE_UP_CLAUSE, VERBALIZED_SYSTEM_PROMPT, parse_react_response


def prompt_for(task: str, history: list[dict], admissible: list[str]) -> str:
    transcript = []
    for item in history:
        transcript += [f"Thought: {item['thought']}", f"Action: {item['action']}", f"Observation: {item['observation']}"]
    history_text = "\n".join(transcript) if transcript else "(no previous steps)"
    actions = "\n".join(f"- {a}" for a in admissible)
    return f"Task: {task}\n\nHistory:\n{history_text}\n\nAdmissible actions:\n{actions}\n\nChoose the next action."


def sample_once(client, model: str, messages: list[dict], extra: dict, max_tokens: int, timeout_attempts: int = 3):
    err = ""
    for attempt in range(timeout_attempts):
        try:
            r = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=0.8,
                                               logprobs=True, **extra)
            choice = r.choices[0]
            text = (choice.message.content or "").strip()
            toks = (choice.logprobs.content if choice.logprobs else None) or []
            lps = [t.logprob for t in toks if t.logprob is not None]
            nll = -sum(lps) / len(lps) if lps else None
            return {"text": text, "action": parse_react_response(text).action or text, "nll": nll}
        except Exception as exc:  # noqa: BLE001
            err = str(exc)[-160:]
            time.sleep(2 * (attempt + 1))
    return {"text": "", "action": "", "nll": None, "error": err}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alfworld", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", required=True, help="openai/gpt-oss-20b or qwen/qwen3.6-35b-a3b")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    from openai import OpenAI
    client = OpenAI(base_url=a.base_url, api_key=os.environ[a.api_key_env], timeout=a.timeout, max_retries=0)
    extra: dict = {"extra_body": {"provider": {"require_parameters": True, "sort": "throughput"}}}
    if "gpt-oss" in a.model:
        extra["extra_body"]["reasoning"] = {"effort": "low"}
    else:
        extra["extra_body"]["reasoning"] = {"enabled": False}
    system = VERBALIZED_SYSTEM_PROMPT + GIVE_UP_CLAUSE
    rows = defaultdict(list)
    for line in open(a.alfworld / "trajectories.jsonl"):
        if line.strip():
            r = json.loads(line); rows[r["episode_id"]].append(r)
    episodes = list(rows.items())[: a.limit or None]
    done = set()
    if a.out.exists():
        kept = [json.loads(l) for l in open(a.out) if l.strip()]
        kept = [r for r in kept if r["samples"] and sum(1 for s in r["samples"] if s["text"]) >= a.n // 2]
        with open(a.out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in kept)
        done = {(r["id"], r["step"]) for r in kept}
        print(f"[uprop] resuming: {len(done)} steps kept", flush=True)
    out = open(a.out, "a"); asked = 0
    with ThreadPoolExecutor(a.workers) as pool:
        for eid, steps in episodes:
            jobs = []
            for k, r in enumerate(steps):
                if (eid, k) in done:
                    continue
                history = [{"thought": s.get("thought") or "", "action": s.get("action") or "", "observation": s.get("observation") or ""} for s in steps[:k]]
                messages = [{"role": "system", "content": system},
                            {"role": "user", "content": prompt_for(r["task"], history, r.get("admissible_actions") or [])}]
                realised = r.get("proposed_action") or r.get("action") or ""
                jobs.append((k, realised, messages))
            for (k, realised, messages), samples in zip(jobs, pool.map(
                    lambda j: [sample_once(client, a.model, j[2], extra, a.max_tokens) for _ in range(a.n)], jobs)):
                out.write(json.dumps({"id": eid, "step": k, "realised": realised, "samples": samples}) + "\n"); asked += 1
            out.flush()
            print(f"[uprop] {eid}: {len(steps)} steps ({len(jobs)} sampled x {a.n})", flush=True)
    print(f"[uprop] done: {asked} steps -> {a.out}")


if __name__ == "__main__":
    main()
