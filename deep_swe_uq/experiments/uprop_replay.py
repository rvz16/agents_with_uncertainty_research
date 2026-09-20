"""Per-step decision samples for UProp on a recorded DeepSWE run (cluster replay).

For every assistant turn of every trajectory the recorded message history up
to that turn is sent back to the model with the agent's bash tool, and N
decisions are sampled at temperature 0.8 with log-probabilities (vLLM's ``n``
gives them in one request, sharing the prefix). The decision compared by
UProp's distance is the sampled command (tool-call argument), or the text when
the sample carried no tool call; ``nll`` is the per-token negative
log-probability of the whole sample. Output rows match
alfworld_uq/experiments/uprop_samples.py: {"id", "step", "realised", "samples"}.

Reasoning is kept short (gpt-oss: reasoning_effort low; Qwen: thinking off),
which departs from the recorded run's own settings but keeps 10 samples per
prefix tractable over ~30k prefixes.

    python uprop_replay.py --jobs jobs.zip --run deepswe_gptoss_113_v5 --base-url http://host/v1 \\
        --model openai/gpt-oss-20b --out uprop_deepswe_gptoss_113_v5.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from verb_replay import clean, trajectories  # same archive walk and message cleaning

BASH_TOOL = {"type": "function", "function": {"name": "bash", "description": "Execute a bash command",
             "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
                            "required": ["command"]}}}


def realised_decision(msg: dict) -> str:
    for c in msg.get("tool_calls") or []:
        try:
            return str(json.loads(c["function"]["arguments"]).get("command", ""))
        except Exception:  # noqa: BLE001
            pass
    return str(msg.get("content") or "")


def sample(client, model: str, prefix: list[dict], n: int, extra: dict, max_tokens: int) -> list[dict]:
    err = ""
    for attempt in range(3):
        try:
            r = client.chat.completions.create(model=model, messages=prefix, tools=[BASH_TOOL], tool_choice="auto",
                                               n=n, temperature=0.8, max_tokens=max_tokens, logprobs=True, **extra)
            out = []
            for ch in r.choices:
                text = (ch.message.content or "").strip()
                action = ""
                for c in ch.message.tool_calls or []:
                    try:
                        action = str(json.loads(c.function.arguments).get("command", "")); break
                    except Exception:  # noqa: BLE001
                        pass
                toks = (ch.logprobs.content if ch.logprobs else None) or []
                lps = [t.logprob for t in toks if t.logprob is not None]
                out.append({"text": action or text, "action": action or text, "nll": -sum(lps) / len(lps) if lps else None})
            return out
        except Exception as exc:  # noqa: BLE001
            err = str(exc)[-200:]
            if "context" in err.lower() or "maximum" in err.lower():
                return [{"text": "", "action": "", "nll": None, "error": "context_overflow"}]
            time.sleep(3 * (attempt + 1))
    return [{"text": "", "action": "", "nll": None, "error": err}]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", required=True); p.add_argument("--run", required=True)
    p.add_argument("--base-url", required=True); p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--n", type=int, default=10); p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--reasoning-effort", default=None, help="gpt-oss: low; Qwen: enable_thinking=false is sent instead")
    a = p.parse_args()
    from openai import OpenAI
    client = OpenAI(base_url=a.base_url, api_key="local", timeout=900)
    extra: dict = {"extra_body": {"reasoning_effort": a.reasoning_effort}} if a.reasoning_effort else \
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    done = set()
    if a.out.exists():
        rows = [json.loads(l) for l in open(a.out) if l.strip()]
        rows = [r for r in rows if sum(1 for s in r["samples"] if s.get("text")) >= a.n // 2]
        with open(a.out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        done = {(r["id"], r["step"]) for r in rows}
        print(f"[uprop] resuming: {len(done)} steps kept", flush=True)
    out = open(a.out, "a"); total = asked = 0
    for tid, tr in trajectories(a.jobs, a.run):
        msgs = clean(tr.get("messages", []))
        jobs = []
        k = 0
        for j, m in enumerate(msgs):
            if m["role"] == "assistant":
                if (tid, k) not in done:
                    jobs.append((k, realised_decision(m), msgs[:j]))
                k += 1
        total += k
        with ThreadPoolExecutor(a.workers) as pool:
            for (kk, realised, prefix), samples in zip(jobs, pool.map(lambda jb: sample(client, a.model, jb[2], a.n, extra, a.max_tokens), jobs)):
                out.write(json.dumps({"id": tid, "step": kk, "realised": realised, "samples": samples}) + "\n"); asked += 1
        out.flush()
        print(f"[uprop] {tid}: {k} steps ({len(jobs)} sampled x {a.n})", flush=True)
    print(f"[uprop] done: {asked} steps sampled, {total} steps in total -> {a.out}")


if __name__ == "__main__":
    main()
